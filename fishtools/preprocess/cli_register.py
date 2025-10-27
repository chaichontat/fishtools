# %%
import json
import pickle
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import rich_click as click
import toml
from basicpy import BaSiC
from loguru import logger
from pydantic import ValidationError
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools.preprocess.chromatic import Affine
from fishtools.preprocess.config import (
    Config,
    Fiducial,
    NumpyEncoder,
    RegisterConfig,
)
from fishtools.preprocess.deconv.helpers import scale_deconv
from fishtools.preprocess.downsample import gpu_downsample_xy
from fishtools.preprocess.fiducial import Shifts, align_fiducials
from fishtools.utils.io import Workspace, safe_imwrite
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.pretty_print import progress_bar_threadpool

FORBIDDEN_PREFIXES = ["10x", "registered", "shifts", "fids"]

if TYPE_CHECKING:
    pass


# %%

DATA = Path("/working/fishtools/data")


# %%

As = {}
ats = {}
for λ in ["650", "750"]:
    a_ = np.loadtxt(DATA / f"560to{λ}.txt")
    A = np.zeros((3, 3), dtype=np.float64)
    A[:2, :2] = a_[:4].reshape(2, 2)
    t = np.zeros(3, dtype=np.float64)
    t[:2] = a_[-2:]

    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    t[2] = 0
    As[λ] = A
    ats[λ] = t


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    scaled = spiller * corr
    return np.where(spillee >= scaled, spillee - scaled, 0)


def parse_nofids(
    nofids: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    channels: dict[str, str],
):
    """Converts nofids into bits and perform shift correction.

    Args:
        nofids: {name: zcyx images}
        shifts: {name: shift vector}
        channels: channel {bit: name, must be 488, 560, 650, 750}


    Raises:
        ValueError: Duplicate bit names.

    Returns:
        out: {bit: zyx images}
        out_shift: {bit: shift vector}
        bit_name_mapping: {bit: (name, idx)} for deconv scaling.
    """

    out: dict[str, Annotated[np.ndarray, "z,y,x"]] = {}
    out_shift: dict[str, Annotated[np.ndarray, "shifts"]] = {}
    bit_name_mapping: dict[str, tuple[str, int]] = {}

    for name, img in nofids.items():
        curr_bits = name.split("-")[0].split("_")
        assert img.shape[1] == len(curr_bits)

        for i, bit in enumerate(curr_bits):
            bit_name_mapping[bit] = (name, i)

            if bit in out:
                raise ValueError(f"Duplicated bit {bit} in {name}")
            out[bit] = img[:, i]  # sliced.max(0, keepdims=True) if max_proj else sliced

        # cs = {str(channels[bit]): bit for bit in bits}
        # if "560" in cs and "647" in cs:
        #     out[cs["560"]] = spillover_correction(out[cs["560"]], out[cs["647"]], 0.22)

        # if "647" in cs and "750" in cs:
        #     out[cs["647"]] = spillover_correction(out[cs["647"]], out[cs["750"]], 0.05)

    for name, shift_ in shifts.items():
        curr_bits = name.split("-")[0].split("_")
        for i, bit in enumerate(curr_bits):
            out_shift[bit] = shift_

    return out, out_shift, bit_name_mapping


def _run_child_cli(argv: list[str], *, check: bool = True):
    """Wrapper around subprocess.run for easy monkeypatching in tests."""

    return subprocess.run(argv, check=check)


def _copy_codebook_to_workspace(cli_path: Path, codebook_path: Path) -> Path:
    """Materialize the provided codebook inside the workspace for reproducibility.

    Args:
        cli_path: Path value provided to the register CLI (typically analysis/deconv).
        codebook_path: Source codebook file to replicate inside the workspace.

    Returns:
        Destination path of the codebook inside ``ws.deconved / codebooks``.
    """

    workspace = Workspace(cli_path)
    codebooks_dir = workspace.deconved / "codebooks"
    codebooks_dir.mkdir(parents=True, exist_ok=True)

    source = Path(codebook_path)
    destination = codebooks_dir / source.name

    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved:
        logger.debug(f"Codebook already present at workspace destination {destination_resolved}")
        return destination

    try:
        shutil.copy2(source, destination)
    except shutil.SameFileError:
        logger.debug(f"Codebook already present at workspace destination {destination_resolved}")
    else:
        logger.debug(f"Copied codebook from {source_resolved} to {destination_resolved}")

    return destination


def sort_key(x: tuple[str, np.ndarray]) -> int | str:
    """Key function for dict.items() by numerical order.

    Args:
        x: dict.items() tuple with bit name as first element and array as second element.

    Returns:
        Key for sorting.
    """
    try:
        return f"{int(x[0]):02d}"
    except ValueError:
        return x[0]


def apply_deconv_scaling(
    img: np.ndarray,
    *,
    idx: int,
    orig_name: str,
    global_deconv_scaling: np.ndarray,
    metadata: dict[str, Any],
    debug: bool,
) -> np.ndarray:
    """Apply deconvolution rescaling unless input is already prenormalized."""

    if metadata.get("prenormalized"):
        if debug:
            logger.debug(f"Skipping deconvolution scaling for {orig_name}: metadata prenormalized flag set.")
        return img

    return scale_deconv(
        img,
        idx,
        name=orig_name,
        global_deconv_scaling=global_deconv_scaling,
        metadata=metadata,
        debug=debug,
    )


@dataclass
class Image:
    name: str
    idx: int
    nofid: np.ndarray
    fid: np.ndarray
    fid_raw: np.ndarray
    bits: list[str]
    powers: dict[str, float]
    metadata: dict[str, Any]
    global_deconv_scaling: np.ndarray | None
    basic: Callable[[], dict[str, BaSiC] | None]

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        discards: dict[str, list[str]] | None = None,
        n_fids: int = 1,
    ):
        ws = Workspace(path.parent.parent)
        stem = path.stem
        name, idx = stem.split("-")
        bits = name.split("_")
        if discards is None:
            discards = {}

        to_discard_idxs = []
        for k, v in discards.items():
            if k in bits and name in v:
                logger.debug(f"Discarding {k} (index {bits.index(k)}) from {stem}.")
                to_discard_idxs.append(bits.index(k))

        with TiffFile(path) as tif:
            try:
                img = tif.asarray()
                try:
                    metadata = tif.shaped_metadata[0]  # type: ignore
                except IndexError:
                    metadata = tif.imagej_metadata
                # tifffile throws IndexError if the file is truncated
            except IndexError as e:
                raise Exception(f"File {path} is corrupted. Please check the file.") from e
            assert metadata is not None

        try:
            waveform = (
                metadata["waveform"]
                if isinstance(metadata["waveform"], dict)
                else json.loads(metadata["waveform"])
            )
        except KeyError:
            waveform = toml.load(path.with_name(f"{path.name.split('-')[0]}.toml"))

        counts = {key: sum(waveform[key]["sequence"]) for key in cls.CHANNELS}

        # To remove ilm from, say, ilm405.

        powers = {
            key[3:]: waveform[key]["power"]
            for key in cls.CHANNELS
            if (key == "ilm405" and counts[key] > n_fids) or (key != "ilm405" and counts[key])
        }

        if waveform.get("params"):
            powers = waveform["params"]["powers"]

        if len(powers) != len(bits):
            raise ValueError(f"{path}: Expected {len(bits)} channels, got {len(powers)}")

        prenormalized = metadata.get("prenormalized", False)
        if not prenormalized:
            try:
                global_deconv_scaling = (
                    np.loadtxt(ws.deconv_scaling(name)).astype(np.float32).reshape((2, -1))
                )
            except FileNotFoundError:
                raise ValueError(f"No deconv_scaling found for {name} and prenormalized not set.")
        else:
            global_deconv_scaling = None

        nofid = img[:-n_fids].reshape(-1, len(powers), 2048, 2048)

        if to_discard_idxs:
            _bits = name.split("_")
            power_keys = list(powers.keys())
            for _idx_d in to_discard_idxs:
                _bits.pop(_idx_d)
                del powers[power_keys[_idx_d]]
            name = "_".join(_bits)
            keeps = list(sorted(set(range(len(bits))) - set(to_discard_idxs)))
            nofid = nofid[:, keeps]
            bits = [bits[i] for i in keeps]
            global_deconv_scaling = (
                global_deconv_scaling[:, keeps] if global_deconv_scaling is not None else None
            )
            assert len(_bits) == nofid.shape[1]

        path_basic = path.parent.parent / "basic" / f"{name}.pkl"
        if path_basic.exists():

            def b():
                basic = pickle.loads(path_basic.read_bytes())
                return dict(zip(bits, basic.values()))

            basic = b
        else:
            # raise Exception(f"No basic template found at {path_basic}")
            basic = lambda: None

        fids_raw = np.atleast_3d(img[-n_fids:]).max(axis=0)
        return cls(
            name=name,
            idx=int(idx),
            nofid=nofid,
            fid=cls.loG_fids(fids_raw),
            fid_raw=fids_raw,
            bits=bits,
            powers=powers,
            metadata=metadata,
            global_deconv_scaling=global_deconv_scaling,
            basic=basic,
        )

    @staticmethod
    def loG_fids(fid: np.ndarray):
        if len(fid.shape) == 3:
            fid = fid.max(axis=0)

        temp = -ndimage.gaussian_laplace(fid.astype(np.float32).copy(), sigma=3)  # type: ignore
        temp -= temp.min()
        percs = np.percentile(temp, [1, 99.99])

        if percs[1] - percs[0] == 0:
            raise ValueError("Uniform image")
        temp = (temp - percs[0]) / (percs[1] - percs[0])
        return temp


def run_fiducial(
    path: Path,
    fids: dict[str, np.ndarray],
    codebook_name: str,
    config: Config,
    *,
    roi: str,
    idx: int,
    reference: str,
    debug: bool,
    no_priors: bool = False,
):
    logger.info("Aligning fiducials")

    prior_mapping: dict[str, str] = {}

    if (
        not config.registration.fiducial.use_fft
        and len(shifts_existing := sorted((path / f"shifts--{roi}+{codebook_name}").glob("*.json"))) > 10
        and not no_priors
    ):
        _priors: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for shift_path in shifts_existing:
            try:
                shift_dicts = Shifts.validate_json(shift_path.read_text())
            except ValidationError as e:
                logger.warning(f"Error decoding {shift_path} {e}. Skipping.")
                continue
            for name, shift_dict in shift_dicts.items():
                if shift_dict.residual < 0.3:
                    _priors[name].append(shift_dict.shifts)
        logger.debug(f"Using priors from {len(shifts_existing)} existing shifts.")
        config.registration.fiducial.priors = {
            name: tuple(np.median(np.array(shifts), axis=0)) for name, shifts in _priors.items()
        }
        logger.debug(config.registration.fiducial.priors)

    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            for file in fids:
                if file.startswith(name):
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(
                    f"{idx}: Searched {list(fids.keys())}. Could not find file that starts with {name} for prior shift."
                )

    if config.registration.fiducial.overrides is not None:
        for name, sh in config.registration.fiducial.overrides.items():
            for file in fids:
                if file.startswith(name):
                    logger.info(f"Overriding shift for {name} to {sh}")
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(f"Could not find file that starts with {name} for override shift.")

    # Write reference fiducial
    (fid_path := path / f"fids--{roi}").mkdir(exist_ok=True)
    crop = config.registration.crop
    safe_imwrite(
        fid_path / f"fids-{idx:04d}.tif",
        fids[reference][crop:-crop, crop:-crop],
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "YX"},
    )

    _fids_path = path / f"registered--{roi}+{codebook_name}" / "_fids"
    _fids_path.mkdir(exist_ok=True, parents=True)

    safe_imwrite(
        _fids_path / f"_fids-{idx:04d}.tif",
        np.stack([v for v in fids.values()]),
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "CYX", "key": list(fids.keys())},
    )

    shifts, residuals = align_fiducials(
        fids,
        reference=reference,
        debug=debug,
        max_iters=5,
        threshold_sigma=config.registration.fiducial.threshold,
        fwhm=config.registration.fiducial.fwhm,
        use_fft=config.registration.fiducial.use_fft,
    )

    assert shifts  # type: ignore
    assert residuals  # type: ignore

    shifted = {k: shift(fid, [shifts[k][1], shifts[k][0]]) for k, fid in fids.items()}

    if debug:
        safe_imwrite(
            path / f"fids_shifted-{idx:04d}.tif",
            # Prior shifts already applied.
            np.stack(list(shifted.values())),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )

    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            if (
                not config.registration.fiducial.overrides
                or name not in config.registration.fiducial.overrides
            ):
                shifts[prior_mapping[name]][0] += sh[0]
                shifts[prior_mapping[name]][1] += sh[1]

    (shift_path := path / f"shifts--{roi}+{codebook_name}").mkdir(exist_ok=True)

    _fid_ref = fids[reference][500:-500:2, 500:-500:2].flatten()
    validated = Shifts.validate_python({
        k: {
            "shifts": (shifts[k][0], shifts[k][1]),
            "residual": residuals[k],
            "corr": 1.0
            if reference == k
            else np.corrcoef(shifted[k][500:-500:2, 500:-500:2].flatten(), _fid_ref)[0, 1],
        }
        for k in fids
    })
    jsoned = Shifts.dump_json(validated)
    (shift_path / f"shifts-{idx:04d}.json").write_bytes(jsoned)
    logger.debug({k: f"{r.corr:03f}" for k, r in validated.items()})
    return shifts


def _run(
    path: Path,
    roi: str,
    idx: int,
    *,
    codebook: str | Path,
    reference: str,
    config: Config,
    debug: bool = False,
    overwrite: bool = False,
    no_priors: bool = False,
):
    logger.info("Reading files")
    codebook_name = Path(codebook).stem
    out_path = path / f"registered--{roi}+{codebook_name}"
    reg_file = out_path / f"reg-{idx:04d}.tif"

    if not overwrite and reg_file.exists():
        logger.info(f"Skipping {idx}")
        return

    out_path.mkdir(exist_ok=True, parents=True)

    # path_prevfids = out_path / "_fids" / f"_fids-{idx:04d}.tif"
    # if path_prevfids.exists():
    #     with TiffFile(path_prevfids) as tif:
    #         _fids = tif.asarray()
    #         fids = {k: v for k, v in zip(tif.shaped_metadata[0]["key"], _fids)}
    #         del _fids
    #     shifts = run_fiducial(
    #         path,
    #         fids,
    #         Path(codebook).stem,
    #         config,
    #         roi=roi,
    #         reference=reference,
    #         debug=debug,
    #         idx=idx,
    #         no_priors=no_priors,
    #     )
    #     del fids
    # else:
    shifts = {}

    cb = json.loads(Path(codebook).read_text())
    codebook_bits = {str(bit) for bit in chain.from_iterable(cb.values())}

    roi_dirs = [
        p
        for p in Path(path).glob(f"*--{roi}")
        if p.is_dir()
        and not any(p.name.startswith(bad) for bad in FORBIDDEN_PREFIXES + (config.exclude or []))
    ]

    available_bits = {bit for p in roi_dirs for bit in p.name.split("--")[0].split("_") if bit}

    missing_bits = sorted(bit for bit in codebook_bits if bit not in available_bits)
    available_display = ", ".join(sorted(available_bits)) or "none"
    if missing_bits:
        raise ValueError(
            f"Missing codebook bits for ROI {roi}: {', '.join(missing_bits)} (available: {available_display})"
        )

    reference_bits = set(reference.split("_"))
    folders = {
        p for p in roi_dirs if set(p.name.split("--")[0].split("_")) & (codebook_bits | reference_bits)
    }

    # Convert file name to bit
    _imgs: list[Image] = [
        Image.from_file(
            file,
            discards=config.registration and config.registration.discards,
            n_fids=config.registration.fiducial.n_fids,
        )
        for file in chain.from_iterable(p.glob(f"*-{idx:04d}.tif") for p in folders)
        if not any(file.parent.name.startswith(bad) for bad in FORBIDDEN_PREFIXES + (config.exclude or []))
    ]
    imgs = {img.name: img for img in _imgs}
    del _imgs

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    if not shifts:
        shifts = run_fiducial(
            path,
            {name: img.fid for name, img in imgs.items()},
            codebook_name,
            config,
            roi=roi,
            reference=reference,
            debug=debug,
            idx=idx,
            no_priors=no_priors,
        )

    for _img in imgs.values():
        del _img.fid, _img.fid_raw

    # Remove reference if not in codebook since we're done with fiducials.
    if not (set(reference.split("_")) & codebook_bits):
        del imgs[reference]

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, [p[-3:] for p in img.powers]))

    assert all(v.isdigit() for v in channels.values())

    logger.debug(f"Channels: {channels}")

    # Split into individual bits.
    # Spillover correction, max projection
    nofids = {name: img.nofid for name, img in imgs.items()}
    bits, bits_shifted, bit_name_mapping = parse_nofids(nofids, shifts, channels)

    del nofids
    for _img in imgs.values():
        del _img.nofid

    # if debug:
    #     for name, img in bits.items():
    #         logger.info(f"{name}: {img.max()}")

    def collapse_z(
        img: np.ndarray,
        slices: list[tuple[int | None, int | None]] | slice | None,
    ) -> np.ndarray:
        if isinstance(slices, list):
            return np.stack([img[slice(*sl)].max(axis=0) for sl in slices])
        if slices is None:
            slices = slice(None)
        return img[slices]

    transformed: dict[str, np.ndarray] = {}
    ref = None

    affine = Affine(As=As, ats=ats)
    bits_in_output = sorted(codebook_bits & set(bits))
    for i, bit in enumerate(bits_in_output):
        bit = str(bit)
        img = bits[bit]
        del bits[bit]
        c = str(channels[bit])
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        img = collapse_z(img, config.registration.slices).astype(np.float32)
        metadata = imgs[orig_name].metadata

        if not metadata.get("prenormalized"):
            scaling = imgs[orig_name].global_deconv_scaling
            assert scaling is not None
            img = apply_deconv_scaling(
                img,
                idx=orig_idx,
                orig_name=orig_name,
                global_deconv_scaling=scaling,
                metadata=metadata,
                debug=debug,
            )
            del scaling

        if ref is None:
            # Need to put this here because of shape change during collapse_z.
            affine.ref_image = ref = img

        # Within-tile alignment. Chromatic corrections.
        logger.debug(f"{bit}: before affine channel={c}, shiftpx={-bits_shifted[bit]}")
        img = affine(img, channel=c, shiftpx=-bits_shifted[bit], debug=debug)
        crop = config.registration.crop
        downsample = config.registration.downsample
        if crop:
            img = img[:, crop:-crop, crop:-crop]

        if downsample > 1:
            transformed_img = gpu_downsample_xy(
                img,
                crop=0,
                factor=downsample,
                clip_range=(0, 65534),
                output_dtype=np.uint16,
            )
        else:
            transformed_img = np.clip(img, 0, 65534).astype(np.uint16)

        transformed[bit] = transformed_img
        logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")

    if not len(transformed):
        raise ValueError("No images were transformed.")

    if not len({img.shape for img in transformed.values()}) == 1:
        raise ValueError(
            f"Transformed images have different shapes: {set(img.shape for img in transformed.values())}"
        )

    sample = next(iter(transformed.values()))
    out = np.zeros(
        (sample.shape[0], len(transformed), sample.shape[1], sample.shape[2]),
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        out[:, i] = v
    del transformed

    # out[0, -1] = fids[reference][crop:-crop:downsample, crop:-crop:downsample]

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))

    # (path / "down2").mkdir(exist_ok=True)

    safe_imwrite(
        out_path / f"reg-{idx:04d}.tif",
        out,
        compression=22610,
        compressionargs={"level": 0.75},
        metadata={
            "key": keys,
            "axes": "ZCYX",
            "shifts": json.dumps(shifts, cls=NumpyEncoder),
            "config": json.dumps(config.model_dump(), cls=NumpyEncoder),
        },
    )

    # for i in range(0, len(out), 3):
    #     (path / "down2" / str(i)).mkdir(exist_ok=True, parents=True)
    #     tifffile.imwrite(
    #         path / "down2" / str(i) / f"{i:03d}-{idx:04d}.tif",
    #         out[i : i + 3] >> config.reduce_bit_depth,
    #         compression=22610,
    #         compressionargs={"level": 0.9},
    #         metadata={"axes": "CYX", "channels": ",".join(keys[i : i + 3])},
    #         imagej=True,
    #     )


def get_rois(path: Path, roi: str):
    rois = (
        {r.name.split("+")[0].split("--")[1] for r in path.glob("*--*") if r.is_dir()}
        if roi == "*"
        else [roi]
    )
    return {r for r in rois if r and r != "*"}


@click.group()
def register(): ...


@register.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("idx", type=int)
@click.option("--codebook", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option("--roi", type=str, default="*")
@click.option("--reference", "-r", type=str, default="4_12_20")
@click.option("--debug", is_flag=True)
@click.option("--threshold", type=float, default=5.0)
@click.option("--fwhm", type=float, default=4.0)
@click.option("--overwrite", is_flag=True)
@click.option("--no-priors", is_flag=True)
def run(
    path: Path,
    idx: int,
    codebook: Path,
    roi: str | None,
    debug: bool = False,
    reference: str = "4_12_20",
    overwrite: bool = False,
    threshold: float = 6,
    fwhm: float = 4,
    no_priors: bool = False,
):
    """Preprocess image sets before spot calling.

    Args:
        path: Workspace path.
        idx: Index of the image to process.
        roi: ROI to work on.
        debug: More logs and write fids. Defaults to False.
        overwrite: Defaults to False.
        threshold: σ above median to call fiducial spots. Defaults to 6.
        fwhm: FWHM for the Gaussian spot detector. More == more spots but slower.Defaults to 4.
    """
    rois = get_rois(path, roi)
    codebook_name = codebook.stem

    for roi in rois:
        reg_file = path / f"registered--{roi}+{codebook_name}" / f"reg-{idx:04d}.tif"
        if not overwrite and reg_file.exists():
            logger.info(f"Skipping {idx}: registration already present at {reg_file}")
            continue

        log_file_tag = f"{roi}+{codebook_name}+{idx:04d}"
        setup_cli_logging(
            path,
            component="preprocess.register.run",
            file=log_file_tag,
            idx=idx,
            debug=debug,
            extra={"roi": roi, "codebook": codebook_name},
        )

        _run(
            path,
            roi,
            idx,
            codebook=codebook,
            debug=debug,
            reference=reference,
            no_priors=no_priors,
            config=Config(
                dataPath=str(DATA),
                exclude=None,
                registration=RegisterConfig(
                    chromatic_shifts={
                        "650": str(DATA / "560to650.txt"),
                        "750": str(DATA / "560to750.txt"),
                    },
                    fiducial=Fiducial(
                        use_fft=False,
                        fwhm=fwhm,
                        threshold=threshold,
                        priors={},
                        overrides={},
                        n_fids=2,
                    ),
                    reference=reference,
                    downsample=1,
                    crop=40,
                    slices=slice(None),
                    reduce_bit_depth=0,
                    discards=None,
                ),
            ),
            overwrite=overwrite,
        )


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--codebook",
    help="Path to the codebook file",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
@click.option("--ref", default=None, help="Reference identifier")
@click.option("--fwhm", type=float, default=4, help="FWHM value")
@click.option("--threshold", type=float, default=6, help="Threshold value")
@click.option("--threads", type=int, default=15, help="Number of threads to use")
@click.option("--overwrite", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option(
    "--verify",
    is_flag=True,
    help=(
        "After batch registration, verify each registered TIFF can be read and has a "
        "consistent shape (based on the first registered file). If a file fails, rerun "
        "the single 'run' command with --overwrite."
    ),
)
def batch(
    path: Path,
    ref: str | None,
    codebook: Path,
    fwhm: int,
    threshold: int,
    threads: int,
    overwrite: bool,
    debug: bool,
    verify: bool,
):
    # idxs = None
    # use_custom_idx = idxs is not None
    codebook = _copy_codebook_to_workspace(path, codebook)
    codebook_name = codebook.stem
    setup_cli_logging(
        path,
        component="preprocess.register.batch",
        file=f"register-batch-{codebook_name}",
        debug=debug,
        extra={"codebook": codebook_name},
    )
    ws = Workspace(path.parent.parent)
    logger.info(f"Found {ws.rois}")

    if ref is None:
        if "2_10_18" in ws.rounds:
            ref = "2_10_18"
        elif "7_15_23" in ws.rounds:
            ref = "7_15_23"
        else:
            raise ValueError("No reference specified and no default found.")

    for roi in ws.rois:
        names = sorted({name for name in path.rglob(f"{ref}--{roi}/{ref}*.tif")})
        if not len(names):
            raise ValueError(f"No images found for {ref}--{roi}")
        idxs = [
            int(name.stem.split("-")[1])
            for name in names
            if overwrite
            or not (path / f"registered--{roi}+{codebook.stem}/reg-{name.stem.split('-')[1]}.tif").exists()
        ]

        if not idxs and not verify:
            logger.warning(f"Skipping {ref}--{roi}, already registered.")
            continue

        with progress_bar_threadpool(len(idxs), threads=threads, debug=debug) as submit:
            for i in idxs:
                submit(
                    _run_child_cli,
                    [
                        "preprocess",
                        "register",
                        "run",
                        str(path),
                        str(i),
                        f"--codebook={codebook}",
                        f"--fwhm={fwhm}",
                        f"--threshold={threshold}",
                        "--reference",
                        ref,
                        f"--roi={roi}",
                        *(["--overwrite"] if overwrite else []),
                    ],
                    check=True,
                )

        # Optional verification phase: ensure files are readable and shapes match.
        if verify:
            codebook_name = codebook.stem
            reg_dir = path / f"registered--{roi}+{codebook_name}"
            expected_shape: tuple[int, int, int, int] | None = None
            verify_idxs = sorted({int(name.stem.split("-")[1]) for name in names})

            def _read_shape(p: Path) -> tuple[int, int, int, int]:
                try:
                    with TiffFile(p) as tif:  # Verify decoding and read shape
                        arr = tif.asarray()
                        shape = tuple(arr.shape)
                        assert len(shape) == 4, f"Unexpected ndim {arr.ndim} for {p}"
                        return shape  # type: ignore[return-value]
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Failed to read registered file {p}: {e}") from e

            # Establish baseline expected shape from the first index that exists and is readable
            for i in verify_idxs:
                p = reg_dir / f"reg-{i:04d}.tif"
                if not p.exists():
                    # If the file does not exist (e.g., child run skipped), skip baseline attempt.
                    continue
                try:
                    expected_shape = _read_shape(p)
                    logger.info(f"[{roi}] Baseline registered shape from {p.name}: {expected_shape}")
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{roi}] Baseline read failed for {p.name}: {e}")

            # Verify each produced file; if any fails, rerun the single index with overwrite
            failed: list[int] = []
            for i in verify_idxs:
                p = reg_dir / f"reg-{i:04d}.tif"
                if not p.exists():
                    logger.warning(f"[{roi}] Missing output {p.name}; scheduling rerun.")
                    failed.append(i)
                    continue
                try:
                    shape = _read_shape(p)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{roi}] Read failed for {p.name}: {e}; scheduling rerun.")
                    failed.append(i)
                    continue

                if expected_shape is not None and shape != expected_shape:
                    logger.warning(
                        f"[{roi}] Shape mismatch for {p.name}: got {shape}, expected {expected_shape}; scheduling rerun."
                    )
                    failed.append(i)

            for i in failed:
                logger.info(f"[{roi}] Re-running index {i:04d} with --overwrite due to verification failure.")
                _run_child_cli(
                    [
                        "preprocess",
                        "register",
                        "run",
                        str(path),
                        str(i),
                        f"--codebook={codebook}",
                        f"--fwhm={fwhm}",
                        f"--threshold={threshold}",
                        "--reference",
                        ref,
                        f"--roi={roi}",
                        "--overwrite",
                    ],
                    check=True,
                )

                # Re-verify just this file; log if still failing
                p = reg_dir / f"reg-{i:04d}.tif"
                try:
                    shape = _read_shape(p)
                    if expected_shape is not None and shape != expected_shape:
                        logger.error(
                            f"[{roi}] Post-rerun shape still mismatched for {p.name}: {shape} vs {expected_shape}."
                        )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[{roi}] Post-rerun read still failing for {p.name}: {e}")


register.add_command(batch)

if __name__ == "__main__":
    register()

# %%
