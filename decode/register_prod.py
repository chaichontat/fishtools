# %%

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import click

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
import jax

jax.config.update("jax_platform_name", "cpu")

import numpy as np
import SimpleITK as sitk
import tifffile
import toml
from basicpy import BaSiC
from loguru import logger
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools import align_fiducials
from fishtools.analysis.fiducial import align_phase, phase_shift

if TYPE_CHECKING:
    from loguru import Logger


class Fiducial(BaseModel):
    fwhm: float = Field(
        4.0,
        description="Full width at half maximum for fiducial spot detection. The higher this is, the more spots will be detected.",
    )
    threshold: float = Field(
        3.0, description="Threshold for fiducial spot detection in standard deviation above the median."
    )


class RegisterConfig(BaseModel):
    fiducial: Fiducial
    downsample: int = Field(1, description="Downsample factor")
    reduce_bit_depth: int = Field(
        0,
        description="Reduce bit depth by n bits. 0 to disable. This is to assist in compression of output intended for visualization.",
    )
    crop: int = Field(
        25, description="Pixels to crop from each edge. This is to account for translation during alignment."
    )
    slices: list[tuple[int | None, int | None]] = Field(
        [(0, 5)], description="Slice range to use for registration"
    )
    max_proj: bool = True
    split_channels: bool = False


class Config(BaseModel):
    dataPath: str
    registration: RegisterConfig


# print(json.dumps(Config.model_json_schema()))

# %%

DATA = Path("/home/chaichontat/fishtools/data")


# %%


# %%


As = {}
ats = {}
for λ in ["405", "488", "560", "750"]:
    a_ = np.loadtxt(DATA / f"650to{λ}.txt")
    A, t = a_[:9].reshape(3, 3), a_[-3:]
    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    t[2] = 0
    As[λ] = A
    ats[λ] = t


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    return np.clip(spillee - spiller * corr, 0, 65535)


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


def st(ref: sitk.Image, img: np.ndarray[np.float32, Any], transform: sitk.Transform):
    """Execute a sitk transform on an image.

    Args:
        ref: Reference image in sitk format.
        img: Image to transform. Must be in float32 format.
        transform: sitk transform to apply.

    Returns:
        Transformed image.
    """
    image = sitk.GetImageFromArray(img)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(image))


def affine(img: np.ndarray[np.float32, Any], channel: str, shiftpx: np.ndarray, ref: sitk.Image):
    """Chromatic and shift correction. Repeated 2D operations of zyx image.
    Assumes 650 is the reference channel.

    Args:
        img: Single-bit zyx image.
        channel: channel name. Must be 488, 560, 650, 750.
        shiftpx: Vector of shift in pixels.
        ref: Reference image in sitk format.

    Raises:
        ValueError: Invalid shift vector dimension.

    Returns:
        Corrected image.
    """
    if len(shiftpx) != 2:
        raise ValueError

    translate = sitk.TranslationTransform(3)
    translate.SetParameters((float(shiftpx[0]), float(shiftpx[1]), 0.0))

    if channel == "650":
        return st(ref, img, translate)

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(As[channel].flatten())
    affine.SetTranslation([*ats[channel], 0])
    affine.SetCenter([1023.5 + shiftpx[0], 1023.5 + shiftpx[1], 0])

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(translate)
    composite.AddTransform(affine)

    return st(ref, img, composite)


@dataclass
class Image:
    name: str
    idx: int
    nofid: np.ndarray
    fid: np.ndarray
    bits: list[str]
    powers: dict[str, float]
    metadata: dict[str, Any]
    deconv_scaling: np.ndarray

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(cls, path: Path):
        stem = path.stem
        name, idx = stem.split("-")
        bits = name.split("_")
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

        deconv_scaling = np.loadtxt(path.parent / "deconv_scaling.txt").astype(np.float32)
        nofid = img[:-1].reshape(-1, len(powers), 2048, 2048)
        if "dapi" in stem:
            nofid = nofid[::2]

        return cls(
            name=name,
            idx=int(idx),
            nofid=nofid,
            fid=cls.loG_fids(img[-1]),
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
    config: RegisterConfig,
    debug: bool = False,
    overwrite: bool = False,
):
    # channels = toml.load("/fast2/3t3clean/channels.toml")
    logger.info("Reading files")

    if not overwrite and (path / "registered" / f"reg-{idx:04d}.tif").exists():
        logger.info(f"Skipping {idx}")
        return

    try:
        basic = {
            c: pickle.loads((path / f"basic_deconv_{c}.pkl").read_bytes())
            for c in ["405", "560", "650", "750"]
        }
    except FileNotFoundError:
        basic = {
            c: pickle.loads((Path("/home/chaichontat/fishtools/data") / f"basic_{c}.pkl").read_bytes())
            for c in ["405", "560", "650", "750"]
        }

    basic["488"] = basic["560"]

    if roi:
        imgs: dict[str, Image] = {
            file.name: Image.from_file(file)
            for file in sorted(Path(path).glob(f"*--{roi}/*-{idx:04d}.tif"))
            if not file.parent.name in ["10x", "registered"]
        }
    else:
        imgs = {
            file.name: Image.from_file(file)
            for file in sorted(Path(path).rglob(f"*-{idx:04d}.tif"))
            if not file.parent.name in ["10x", "registered"]
        }

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    if debug:
        _fids_path = path / f"fids_{idx}.tif"
        tifffile.imwrite(
            _fids_path,
            np.stack([i.fid for i in imgs.values()]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )
        logger.debug(f"Written fiducials to {_fids_path}.")

    fids = {name: img.fid for name, img in imgs.items()}
    nofids = {name: img.nofid for name, img in imgs.items()}
    logger.info("Aligning fiducials")
    shifts = align_fiducials(
        fids,
        reference=reference,
        debug=debug,
        iterations=3,
        threshold=config.fiducial.threshold,
        fwhm=config.fiducial.fwhm,
    )

    (path / "shifts").mkdir(exist_ok=True)
    with open(path / f"shifts/shifts_{idx}.json", "w") as f:
        json.dump({k: v.tolist() for k, v in shifts.items()}, f)

    if debug:
        tifffile.imwrite(
            path / f"fids_shifted_{idx}.tif",
            np.stack([shift(img, [shifts[k][1], shifts[k][0]]) for k, img in fids.items()]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, img.powers))

    # del imgs

    # Split into individual bits.
    # Spillover correction, max projection
    bits, bits_shifted, bit_name_mapping = parse_nofids(nofids, shifts, channels)

    def collapse_z(
        img: np.ndarray, slices: list[tuple[int | None, int | None]], max_proj: bool = True
    ) -> np.ndarray:
        # return np.stack([img[slice(*sl)].max(axis=0) for sl in slices])
        return img.reshape(-1, 4, *img.shape[1:]).max(axis=1)

    transformed: dict[str, np.ndarray] = {}
    n_z = -1
    ref = None

    for i, (bit, img) in enumerate(bits.items()):
        c = str(channels[bit])
        img = collapse_z(img, config.slices, config.max_proj).astype(np.float32)
        n_z = img.shape[0]
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        img = imgs[orig_name].scale_deconv(img, orig_idx)

        # Illumination correction
        img = np.stack(basic[c].transform(img))

        if ref is None:
            # Need to put this here because of shape change during collapse_z.
            ref = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)

        # Within-tile alignment. Chromatic corrections.
        img = affine(img, c, -bits_shifted[bit], ref)
        transformed[bit] = np.clip(img, 0, 65535).astype(np.uint16)
        logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")

    crop, downsample = config.crop, config.downsample
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

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))

    (outpath := (path / "registered")).mkdir(exist_ok=True, parents=True)
    # (path / "down2").mkdir(exist_ok=True)

    tifffile.imwrite(
        outpath / f"reg-{idx:04d}.tif",
        out,
        compression=22610,
        compressionargs={"level": 0.8},
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
@click.option("--roi", type=str, default="full")
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
            registration=RegisterConfig(
                fiducial=Fiducial(fwhm=4, threshold=8),
                downsample=1,
                crop=25,
                slices=[(0, 5), (5, 10)],
                reduce_bit_depth=0,
                max_proj=True,
            ),
        ).registration,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()

# %%
