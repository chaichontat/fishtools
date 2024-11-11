# %%
import json
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import numpy as np
import tifffile
import toml
from basicpy import BaSiC
from loguru import logger
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools import align_fiducials
from fishtools.preprocess.chromatic import Affine
from fishtools.preprocess.config import ChannelConfig, Config, Fiducial, NumpyEncoder, RegisterConfig
from fishtools.preprocess.deconv import scale_deconv

FORBIDDEN_PREFIXES = ["10x", "registered", "shifts", "fids"]

if TYPE_CHECKING:
    from loguru import Logger


# %%

DATA = Path("/fast2/fishtools/data")


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


def parse_nofids(nofids: dict[str, np.ndarray], shifts: dict[str, np.ndarray], channels: dict[str, str]):
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
    global_deconv_scaling: np.ndarray
    basic: Callable[[], dict[str, BaSiC] | None]

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(cls, path: Path, *, discards: dict[str, list[str]] | None = None, n_fids: int = 1):
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
            img = tif.asarray()
            try:
                metadata = tif.shaped_metadata[0]  # type: ignore
            except IndexError:
                metadata = tif.imagej_metadata
        try:
            waveform = json.loads(metadata["waveform"])
        except KeyError:
            waveform = toml.load(path.with_name(f"{path.name.split('-')[0]}.toml"))

        counts = {key: sum(waveform[key]["sequence"]) for key in cls.CHANNELS}

        # To remove ilm from, say, ilm405.
        powers = {
            key[3:]: waveform[key]["power"]
            for key in cls.CHANNELS
            if (key == "ilm405" and counts[key] > n_fids) or (key != "ilm405" and counts[key])
        }
        if len(powers) != len(bits):
            raise ValueError(f"{path}: Expected {len(bits)} channels, got {len(powers)}")

        try:
            global_deconv_scaling = (
                np.loadtxt((path.parent.parent / "deconv_scaling" / f"{name}.txt"))
                .astype(np.float32)
                .reshape((2, -1))
            )
        except FileNotFoundError:
            if "deconv" in path.resolve().as_posix():
                raise ValueError("No deconv_scaling found.")
            logger.debug("No deconv_scaling found. Using ones.")
            global_deconv_scaling = np.ones((2, len(bits)))

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
            global_deconv_scaling = global_deconv_scaling[:, keeps]
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

        fids_raw = np.atleast_3d(img[-n_fids:])
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
            temp = fid.max(axis=0)
        temp = -ndimage.gaussian_laplace(fid.astype(np.float32).copy(), sigma=3)  # type: ignore
        temp -= temp.min()
        return temp


def run(
    path: Path,
    roi: str,
    idx: int,
    reference: str = "3_11_19",
    *,
    config: Config,
    debug: bool = False,
    overwrite: bool = False,
):
    logger.info("Reading files")

    if not overwrite and (path / f"registered--{roi}" / f"reg-{idx:04d}.tif").exists():
        logger.info(f"Skipping {idx}")
        return

    # Convert file name to bit
    _imgs: list[Image] = [
        Image.from_file(
            file,
            discards=config.channels and config.channels.discards,
            n_fids=config.registration.fiducial.n_fids,
        )
        for file in sorted(Path(path).glob(f"*--{roi}/*-{idx:04d}.tif" if roi else f"*/*-{idx:04d}.tif"))
        if not any(file.parent.name.startswith(bad) for bad in FORBIDDEN_PREFIXES)
    ]
    imgs = {img.name: img for img in _imgs}
    del _imgs

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    logger.info("Aligning fiducials")
    nofids = {name: img.nofid for name, img in imgs.items()}
    fids = {name: img.fid for name, img in imgs.items()}

    prior_mapping: dict[str, str] = {}
    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            for file in fids:
                if file.startswith(name):
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(f"Could not find file that starts with {name} for prior shift.")

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

    if debug:
        _fids_path = path / f"fids-{idx:04d}.tif"
        tifffile.imwrite(
            _fids_path,
            np.stack([v for v in fids.values()]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )
        logger.debug(f"Written fiducials to {_fids_path}.")

    # Write reference fiducial
    (fid_path := path / f"fids--{roi}").mkdir(exist_ok=True)
    crop, downsample = config.registration.crop, config.registration.downsample
    tifffile.imwrite(
        fid_path / f"fids-{idx:04d}.tif",
        fids[reference][:, crop:-crop, crop:-crop],
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "YX"},
    )

    shifts = align_fiducials(
        fids,
        reference=reference,
        debug=debug,
        max_iters=6,
        threshold_sigma=config.registration.fiducial.threshold,
        fwhm=config.registration.fiducial.fwhm,
    )

    if debug:
        tifffile.imwrite(
            path / f"fids_shifted-{idx:04d}.tif",
            # Prior shifts already applied.
            np.stack([shift(fid.max(axis=0), [shifts[k][1], shifts[k][0]]) for k, fid in fids.items()]),
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

    (shift_path := path / f"shifts--{roi}").mkdir(exist_ok=True)

    _fid_ref = fids[reference][::4, ::4].flatten()
    with open(shift_path / f"shifts-{idx:04d}.json", "w") as f:
        to_dump = {
            k: {
                "shifts": v.tolist(),
                "corr": 1.0 if reference == k else np.corrcoef(fids[k][::4, ::4].flatten(), _fid_ref)[0, 1],
            }
            for k, v in shifts.items()
        }
        json.dump(to_dump, f)
        logger.debug(f"Corr: {[x['corr'] for x in to_dump.values()]}")

    del fids, _fid_ref
    for _img in imgs.values():
        del _img.fid, _img.fid_raw
    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, img.powers))
    logger.debug(f"Channels: {channels}")

    # del imgs

    # Split into individual bits.
    # Spillover correction, max projection
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

    for i, bit in enumerate(list(bits)):
        img = bits[bit]
        del bits[bit]
        c = str(channels[bit])
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        img = collapse_z(img, config.registration.slices).astype(np.float32)
        img = scale_deconv(
            img,
            orig_idx,
            name=orig_name,
            global_deconv_scaling=imgs[orig_name].global_deconv_scaling,
            metadata=imgs[orig_name].metadata,
        )

        # Illumination correction
        basic = imgs[orig_name].basic()
        if basic is not None:
            logger.debug("Running BaSiC")
            img = np.stack(basic[bit].transform(np.array(img)))
        else:
            ...
            # raise Exception("No basic template found.")

        if ref is None:
            # Need to put this here because of shape change during collapse_z.
            affine.ref_image = ref = img

        # Within-tile alignment. Chromatic corrections.
        img = affine(img, channel=c, shiftpx=-bits_shifted[bit], debug=False)
        transformed[bit] = np.clip(img, 0, 65535).astype(np.uint16)
        logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")

    if not len({img.shape for img in transformed.values()}) == 1:
        raise ValueError("Transformed images have different shapes.")

    crop, downsample = config.registration.crop, config.registration.downsample
    out = np.zeros(
        [
            next(iter(transformed.values())).shape[0],
            # if isinstance(config.registration.slices, slice)
            # else len(config.registration.slices),
            len(transformed),
            (2048 - 2 * crop) // downsample,
            (2048 - 2 * crop) // downsample,
        ],
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        out[:, i] = v[:, crop:-crop:downsample, crop:-crop:downsample]
    del transformed

    # out[0, -1] = fids[reference][crop:-crop:downsample, crop:-crop:downsample]

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))

    (outpath := (path / f"registered--{roi}")).mkdir(exist_ok=True, parents=True)
    # (path / "down2").mkdir(exist_ok=True)

    tifffile.imwrite(
        outpath / f"reg-{idx:04d}.tif",
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


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("idx", type=int)
@click.option("--roi", type=str, default="")
@click.option("--debug", is_flag=True)
@click.option("--reference", "-r", type=str)
@click.option("--threshold", type=float, default=2.0)
@click.option("--fwhm", type=float, default=4.0)
@click.option("--overwrite", is_flag=True)
def main(
    path: Path,
    idx: int,
    reference: str,
    roi: str,
    debug: bool = False,
    overwrite: bool = False,
    threshold: float = 6,
    fwhm: float = 4,
):
    """Preprocess image sets before spot calling.

    Args:
        path: Workspace path.
        idx: Index of the image to process.
        reference: Reference round to use for registration.
        roi: ROI to work on.
        debug: More logs and write fids. Defaults to False.
        overwrite: Defaults to False.
        threshold: σ above median to call fiducial spots. Defaults to 6.
        fwhm: FWHM for the Gaussian spot detector. More == more spots but slower.Defaults to 4.
    """
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> {extra[idx]} {extra[file]}| "
        "- <level>{message}</level>"
    )
    logger.remove()
    logger.configure(extra=dict(idx=f"{idx:04d}", file=""))
    logger.add(sys.stderr, level="DEBUG" if debug else "WARNING", format=logger_format)

    run(
        path,
        roi,
        idx,
        reference=reference,
        debug=debug,
        config=Config(
            dataPath=str(DATA),
            channels=ChannelConfig(discards={"af": ["af_3_11_19"]}),
            basic=None,
            # basic_template=dict(
            #     zip(
            #         ["560", "650", "750"],
            #         [str(f"/mnt/archive/starmap/e155_trc/basic_{c}.pkl") for c in ["560", "650", "750"]],
            #     )
            # ),
            registration=RegisterConfig(
                chromatic_shifts={
                    "647": str(DATA / "560to647.txt"),
                    "750": str(DATA / "560to750.txt"),
                },
                fiducial=Fiducial(
                    fwhm=fwhm,
                    threshold=threshold,
                    priors={
                        # "WGAfiducial_fiducial_tdT_29_polyT": (9, 26),
                        # "blank568_blank647_blank750": (-6, -11),
                        # "6_14_22": (-160, -175),
                        # "7_15_23": (-160, -175),
                        # "8_16_24": (-160, -175),
                        # "25_26_27": (-160, -175),
                        # "dapi_29_polyA": (10, 30),
                        # "polyA_1_9_17": (-10, -5)
                    },
                    overrides={
                        # "polyA_1_9_17": (12.01, -12.43),
                    },
                    n_fids=1,
                ),
                downsample=1,
                crop=30,
                slices=slice(None),
                reduce_bit_depth=0,
            ),
        ),
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()

# %%
