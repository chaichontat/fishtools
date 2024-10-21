# %%

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import click
import numpy as np
import tifffile
import toml
from basicpy import BaSiC
from loguru import logger
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools import align_fiducials
from fishtools.analysis.chromatic import Affine

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
import jax

jax.config.update("jax_platform_name", "cpu")
FORBIDDEN_PREFIXES = ["10x", "registered", "shifts", "fids"]

if TYPE_CHECKING:
    from loguru import Logger


class Fiducial(BaseModel):
    fwhm: float = Field(
        default=4.0,
        description="Full width at half maximum for fiducial spot detection. The higher this is, the more spots will be detected.",
    )
    threshold: float = Field(
        default=3.0,
        description="Threshold for fiducial spot detection in standard deviation above the median.",
    )
    priors: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Shifts to apply before alignment. Name must match round name.",
    )
    overrides: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Overrides for fiducial spot detection. Name must match round name.",
    )


class RegisterConfig(BaseModel):
    fiducial: Fiducial
    downsample: int = Field(default=1, description="Downsample factor")
    reduce_bit_depth: int = Field(
        default=0,
        description="Reduce bit depth by n bits. 0 to disable. This is to assist in compression of output intended for visualization.",
    )
    crop: int = Field(
        default=25,
        description="Pixels to crop from each edge. This is to account for translation during alignment.",
    )
    slices: list[tuple[int | None, int | None]] = Field(
        default=[(0, 5)], description="Slice range to use for registration"
    )
    split_channels: bool = False
    chromatic_shifts: dict[str, Annotated[str, "path for 560to{channel}.txt"]]


class ChannelConfig(BaseModel):
    discards: dict[str, list[str]] = Field(
        description="In case of duplicated key(s), discard the ones from the file in value."
    )


class Config(BaseModel):
    dataPath: str
    registration: RegisterConfig
    channels: ChannelConfig | None = None
    basic_template: dict[str, Annotated[str, "path for basic_{channel}.pkl"]] | None = None


# print(json.dumps(Config.model_json_schema()))

# %%

DATA = Path("/home/chaichontat/fishtools/data")


# %%


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

    for name, shift in shifts.items():
        curr_bits = name.split("-")[0].split("_")
        for i, bit in enumerate(curr_bits):
            out_shift[bit] = shift

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
    deconv_scaling: np.ndarray

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(cls, path: Path, *, discards: dict[str, list[str]] | None = None):
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
        powers = {key[3:]: waveform[key]["power"] for key in cls.CHANNELS if counts[key] > 1}
        if len(powers) != len(bits):
            raise ValueError(f"{path}: Expected {len(bits)} channels, got {len(powers)}")

        try:
            deconv_scaling = (
                np.loadtxt(next(path.parent.glob("deconv_scaling.txt"))).astype(np.float32).reshape((2, -1))
            )
        except StopIteration:
            if "deconv" in path.as_posix():
                raise ValueError("No deconv_scaling found.")
            logger.debug("No deconv_scaling found. Using ones.")
            deconv_scaling = np.ones((2, len(bits)))

        nofid = img[:-1].reshape(-1, len(powers), 2048, 2048)

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
            deconv_scaling = deconv_scaling[:, keeps]
            assert len(_bits) == nofid.shape[1]

        return cls(
            name=name,
            idx=int(idx),
            nofid=nofid,
            fid=cls.loG_fids(img[-1]),
            fid_raw=img[-1],
            bits=bits,
            powers=powers,
            metadata=metadata,
            deconv_scaling=deconv_scaling,
        )

    def scale_deconv(self, img: np.ndarray, idx: int):
        m_ = self.deconv_scaling[0, idx]
        s_ = self.deconv_scaling[1, idx]
        return np.clip(
            (img * (s_ / self.metadata["deconv_scale"])[0] + (s_ * (self.metadata["deconv_min"] - m_))[0]),
            0,
            65535,
        )

    @staticmethod
    def loG_fids(fid: np.ndarray):
        temp = -ndimage.gaussian_laplace(fid.astype(np.float32), sigma=3)  # type: ignore
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

    if config.basic_template:
        basic = {
            channel: pickle.loads(Path(path).read_bytes()) for channel, path in config.basic_template.items()
        }
        try:
            basic = {c: pickle.loads((path / f"basic_{c}.pkl").read_bytes()) for c in ["560", "650", "750"]}
        except FileNotFoundError:
            basic = {c: pickle.loads((DATA / f"basic_{c}.pkl").read_bytes()) for c in ["560", "650", "750"]}
        basic["488"] = basic["560"]
    else:
        basic = {}

    # print(sorted(Path(path).glob(f"*/*-{idx:04d}.tif")))
    # Convert file name to bit
    _imgs: list[Image] = [
        Image.from_file(file, discards=config.channels and config.channels.discards)
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
    tifffile.imwrite(
        fid_path / f"fids-{idx:04d}.tif",
        fids[reference],
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "YX"},
    )

    shifts = align_fiducials(
        fids,
        reference=reference,
        debug=debug,
        iterations=4,
        threshold_sigma=config.registration.fiducial.threshold,
        fwhm=config.registration.fiducial.fwhm,
    )

    if debug:
        tifffile.imwrite(
            path / f"fids_shifted-{idx:04d}.tif",
            # Prior shifts already applied.
            np.stack([shift(img, [shifts[k][1], shifts[k][0]]) for k, img in fids.items()]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )

    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            shifts[prior_mapping[name]][0] += sh[0]
            shifts[prior_mapping[name]][1] += sh[1]

    if config.registration.fiducial.overrides is not None:
        for name, sh in config.registration.fiducial.overrides.items():
            for file in fids:
                if file.startswith(name):
                    shifts[file][0] = sh[0]
                    shifts[file][1] = sh[1]
                    break
            else:
                raise ValueError(f"Could not find file that starts with {name} for override shift.")

    (shift_path := path / f"shifts--{roi}").mkdir(exist_ok=True)

    with open(shift_path / f"shifts-{idx:04d}.json", "w") as f:
        json.dump({k: v.tolist() for k, v in shifts.items()}, f)

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, img.powers))
    logger.debug(f"Channels: {channels}")

    # del imgs

    # Split into individual bits.
    # Spillover correction, max projection
    bits, bits_shifted, bit_name_mapping = parse_nofids(nofids, shifts, channels)

    if debug:
        for name, img in bits.items():
            logger.info(f"{name}: {img.max()}")

    def collapse_z(
        img: np.ndarray,
        slices: list[tuple[int | None, int | None]],
    ) -> np.ndarray:
        return np.stack([img[slice(*sl)].max(axis=0) for sl in slices])
        # return img.reshape(-1, 4, *img.shape[1:]).max(axis=1)

    transformed: dict[str, np.ndarray] = {}
    n_z = -1
    ref = None

    affine = Affine(As=As, ats=ats)

    for i, (bit, img) in enumerate(bits.items()):
        logger.debug(f"Processing {bit}")
        c = str(channels[bit])
        img = collapse_z(img, config.registration.slices).astype(np.float32)
        n_z = img.shape[0]
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        # img = imgs[orig_name].scale_deconv(img, orig_idx)

        # Illumination correction
        if config.basic_template:
            img = np.stack(basic.get(c, basic["560"]).transform(np.array(img)))

        if ref is None:
            # Need to put this here because of shape change during collapse_z.
            affine.ref_image = ref = img

        # Within-tile alignment. Chromatic corrections.
        img = affine(img, channel=c, shiftpx=-bits_shifted[bit], debug=False)
        transformed[bit] = np.clip(img, 0, 65535).astype(np.uint16)
        logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")

    crop, downsample = config.registration.crop, config.registration.downsample
    out = np.zeros(
        [
            n_z,
            len(transformed),
            (2048 - 2 * crop) // downsample,
            (2048 - 2 * crop) // downsample,
        ],
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        out[:, i] = v[:, crop:-crop:downsample, crop:-crop:downsample]

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
        metadata={"key": keys, "axes": "ZCYX"},
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
@click.option("--ignore", type=str)
def main(
    path: Path,
    idx: int,
    reference: str,
    roi: str,
    debug: bool = False,
    overwrite: bool = False,
    threshold: float = 2,
    fwhm: float = 4,
    downsample: int = 1,
    crop: int = 30,
    ignore: str | None = None,
):
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
                        # "1_9_17": (-21, -18)
                    },
                ),
                downsample=1,
                crop=30,
                slices=[(i, i + 5) for i in range(0, 55, 5)],
                reduce_bit_depth=0,
            ),
        ),
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()

# %%
