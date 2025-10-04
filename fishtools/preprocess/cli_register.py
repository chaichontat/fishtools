# %%
import json
import subprocess
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import line_profiler as line_profiler
import numpy as np
import rich_click as click
import tifffile
from loguru import logger
from pydantic import ValidationError
from scipy.ndimage import shift
from tifffile import TiffFile  # For test-time patching expectations

from fishtools.gpu.memory import release_all as gpu_release_all
from fishtools.io.image import Image
from fishtools.io.workspace import Workspace
from fishtools.preprocess.chromatic import Affine
from fishtools.preprocess.config import (
    Config,
    Fiducial,
    NumpyEncoder,
    RegisterConfig,
)
from fishtools.preprocess.deconv.core import PRENORMALIZED
from fishtools.preprocess.deconv.helpers import scale_deconv
from fishtools.preprocess.downsample import gpu_downsample_xy
from fishtools.preprocess.fiducial import Shifts, align_fiducials
from fishtools.utils.logging import setup_workspace_logging
from fishtools.utils.pretty_print import progress_bar_threadpool

FORBIDDEN_PREFIXES = ["10x", "registered", "shifts", "fids"]


def _shift_json_path(base: Path, roi: str, idx: int, codebook_name: str) -> Path:
    """Return the expected shifts JSON path for a ROI/index/codebook trio."""

    return base / f"shifts--{roi}+{codebook_name}" / f"shifts-{idx:04d}.json"


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


@line_profiler.profile
def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    scaled = spiller * corr
    return np.where(spillee >= scaled, spillee - scaled, 0)


@line_profiler.profile
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


@line_profiler.profile
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

    if metadata.get(PRENORMALIZED):
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


__all__ = ["Image"]


@line_profiler.profile
def run_fiducial(
    path: Path,
    fids: Annotated[dict[str, np.ndarray], "Dict of {name: yx image}"],
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

    configured_priors = dict(config.registration.fiducial.priors or {})
    derived_priors: dict[str, tuple[float, float]] = {}

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
        if _priors:
            medianized = {
                name: tuple(np.median(np.array(shifts), axis=0)) for name, shifts in _priors.items()
            }
            unmatched = {name for name in medianized if not any(fid.startswith(name) for fid in fids)}
            if unmatched:
                logger.debug(
                    "Dropping %s derived priors without matching fid tiles: %s",
                    len(unmatched),
                    sorted(unmatched),
                )
            derived_priors = {name: shift for name, shift in medianized.items() if name not in unmatched}
            logger.debug(
                "Using priors derived from %s existing shifts: %s",
                len(shifts_existing),
                derived_priors,
            )

    effective_priors = configured_priors.copy()
    effective_priors.update(derived_priors)

    for name, sh in effective_priors.items():
        for file in fids:
            if file.startswith(name):
                fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                prior_mapping[name] = file
                break
        else:
            raise ValueError(
                f"{idx}: Detected priors: {effective_priors}. Could not find file that starts with {name} for prior shift. "
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
    crop, downsample = config.registration.crop, config.registration.downsample
    tifffile.imwrite(
        fid_path / f"fids-{idx:04d}.tif",
        fids[reference][crop:-crop, crop:-crop],
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "YX"},
    )

    _fids_path = path / f"registered--{roi}+{codebook_name}" / "_fids"
    _fids_path.mkdir(exist_ok=True, parents=True)

    tifffile.imwrite(
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
        tifffile.imwrite(
            path / f"fids_shifted-{idx:04d}.tif",
            # Prior shifts already applied.
            np.stack(list(shifted.values())),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )

    if effective_priors:
        for name, sh in effective_priors.items():
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


@line_profiler.profile
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
    codebook_name = Path(codebook).stem
    shift_json = _shift_json_path(path, roi, idx, codebook_name)

    if not overwrite and shift_json.exists():
        logger.info(f"Skipping {roi}/{idx:04d}: existing shift file at {shift_json.relative_to(path)}.")
        return

    logger.info("Reading files")
    out_path = path / f"registered--{roi}+{codebook_name}"
    out_path.mkdir(exist_ok=True, parents=True)

    if not overwrite and (out_path / f"reg-{idx:04d}.tif").exists():
        logger.info(f"Skipping {idx}")
        return

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
    bits_cb = {str(bit) for bit in chain.from_iterable(cb.values())}

    ws = Workspace(path)
    round_tokens = bits_cb | {token for token in reference.split("_") if token}
    candidate_rounds = []
    for round_name in ws.rounds:
        tokens = set(ws._split_round_tokens(round_name))
        if tokens & round_tokens and not any(round_name.startswith(bad) for bad in FORBIDDEN_PREFIXES):
            candidate_rounds.append(round_name)

    imgs: dict[str, Image] = {}
    for round_name in candidate_rounds:
        tile_path = ws.img(round_name, roi, idx)
        if not tile_path.exists():
            continue
        img = Image.from_file(
            tile_path,
            discards=config.registration and config.registration.discards,
            n_fids=config.registration.fiducial.n_fids,
        )
        if any(round_name.startswith(bad) for bad in FORBIDDEN_PREFIXES + (config.exclude or [])):
            continue
        imgs[img.name] = img

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    available_bits = {str(bit) for img in imgs.values() for bit in img.bits}
    missing_bits = sorted(bits_cb - available_bits)
    if missing_bits:
        raise FileNotFoundError(
            "Missing registered tiles for bits "
            f"{missing_bits} (ROI={roi}, index={idx:04d}). "
            "Ensure required rounds are present under analysis/deconv."
        )

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
    if not (set(reference.split("_")) & set(map(str, bits_cb))):
        del imgs[reference]

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= {str(bit): str(channel)[-3:] for bit, channel in zip(img.bits, img.powers)}

    assert all(v.isdigit() for v in channels.values())

    missing_channels = [bit for bit in bits_cb if bit not in channels]
    if missing_channels:
        raise ValueError(
            "Missing channel metadata for bits "
            f"{sorted(missing_channels)}. Available bit channels: {sorted(channels)}"
        )

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
    bits_cb_sorted = sorted(bits_cb & set(bits))
    for i, bit in enumerate(bits_cb_sorted):
        img = bits[bit]
        del bits[bit]
        c = str(channels[bit])
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        img = collapse_z(img, config.registration.slices).astype(np.float32)
        metadata = imgs[orig_name].metadata

        img = apply_deconv_scaling(
            img,
            idx=orig_idx,
            orig_name=orig_name,
            global_deconv_scaling=imgs[orig_name].global_deconv_scaling,
            metadata=metadata,
            debug=debug,
        )

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
            try:
                transformed_img = gpu_downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=(0, 65534),
                    output_dtype=np.uint16,
                )
            finally:
                try:
                    gpu_release_all()
                except Exception:
                    logger.opt(exception=True).debug(
                        "GPU cleanup failed in cli_register downsample; continuing."
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

    tifffile.imwrite(
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
@line_profiler.profile
def run(
    path: Path,
    idx: int,
    codebook: Path,
    roi: str,
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
    # Workspace-scoped logging: write all logs (not progress bars) to
    # {workspace}/analysis/logs/preprocess.register.log
    setup_workspace_logging(
        path,
        component="preprocess.register",
        idx=idx,
        file=roi,
        debug=debug,
        # Console stays quiet unless --debug; file captures INFO+ by default
        console_level="WARNING",
        file_level="INFO",
    )

    rois = get_rois(path, roi)
    codebook_name = codebook.stem

    for roi in rois:
        shift_json = _shift_json_path(path, roi, idx, codebook_name)
        if not overwrite and shift_json.exists():
            logger.info(f"Skipping {roi}/{idx:04d}: existing shift file at {shift_json.relative_to(path)}.")
            continue
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
def batch(
    path: Path,
    ref: str | None,
    codebook: Path,
    fwhm: int,
    threshold: int,
    threads: int,
    overwrite: bool,
):
    # idxs = None
    # use_custom_idx = idxs is not None
    # Enable logging for the batch driver itself
    setup_workspace_logging(path, component="preprocess.register.batch", file="batch")
    ws = Workspace(path.parent.parent)
    logger.info(f"Found ROIs: {ws.rois}")

    codebook_name = codebook.stem

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
        idxs: list[int] = []
        for name in names:
            idx = int(name.stem.split("-")[1])
            if not overwrite:
                reg_exists = (path / f"registered--{roi}+{codebook_name}" / f"reg-{idx:04d}.tif").exists()
                if reg_exists:
                    continue
            idxs.append(idx)

        if not idxs:
            logger.warning(f"Skipping {ref}--{roi}, all tiles are registered and --overwrite not given.")
            continue

        with progress_bar_threadpool(len(idxs), threads=threads) as submit:
            for i in idxs:
                submit(
                    subprocess.run,
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


register.add_command(batch)

if __name__ == "__main__":
    register()

# %%
