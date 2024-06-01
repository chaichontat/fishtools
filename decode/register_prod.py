# %%

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import jax
import numpy as np
import SimpleITK as sitk
import tifffile
import toml
from basicpy import BaSiC
from loguru import logger
from scipy import ndimage
from scipy.ndimage import shift
from skimage.measure import block_reduce
from tifffile import TiffFile, imread

from fishtools import align_fiducials
from fishtools.analysis.fiducial import align_phase, phase_shift

if TYPE_CHECKING:
    from loguru import Logger

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

jax.config.update("jax_platform_name", "cpu")
DATA = Path("/home/chaichontat/fishtools/data")

with open(DATA / "basic_all.pkl", "rb") as f:
    basic: dict[str, BaSiC] = pickle.load(f)

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


def parse_nofids(
    nofids: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    channels: dict[str, str],
    *,
    z_slice: slice = np.s_[:],
    max_proj: bool = True,
):
    out, out_shift = {}, {}
    for name, img in nofids.items():
        bits = name.split("-")[0].split("_")
        assert img.shape[1] == len(bits)
        for i, bit in enumerate(bits):
            sliced = img[z_slice, i]
            if bit in out:
                raise ValueError(f"Duplicated bit {bit} in {name}")
            out[bit] = sliced.max(0, keepdims=True) if max_proj else sliced

        cs = {str(channels[bit]): bit for bit in bits}
        if "560" in cs and "647" in cs:
            out[cs["560"]] = spillover_correction(out[cs["560"]], out[cs["647"]], 0.22)

        # if "647" in cs and "750" in cs:
        #     out[cs["647"]] = spillover_correction(out[cs["647"]], out[cs["750"]], 0.05)

    for name, shift in shifts.items():
        bits = name.split("-")[0].split("_")
        for i, bit in enumerate(bits):
            if bit == "polyA" and "dapi" in bits:
                continue
            out_shift[bit] = shift

    return out, out_shift


def sort_key(x):
    try:
        return f"{int(x[0]):02d}"
    except ValueError:
        return x[0]


def st(ref: sitk.Image, img: np.ndarray[np.float32, Any], transform: sitk.Transform):
    image = sitk.GetImageFromArray(img)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(image))


def affine(img: np.ndarray[np.float32, Any], channel: str, shiftpx: np.ndarray, ref: sitk.Image):
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
        # To remove ilm from ilm405.
        powers = {key[3:]: waveform[key]["power"] for key in cls.CHANNELS if counts[key] > 1}
        if len(powers) != len(bits):
            raise ValueError(f"{path}: Expected {len(bits)} channels, got {len(powers)}")
        return cls(
            name=name,
            idx=int(idx),
            nofid=img[:-1].reshape(-1, len(powers), 2048, 2048),
            fid=cls.log_fids(img[-1]),
            bits=bits,
            powers=powers,
            metadata=metadata,
        )

    @staticmethod
    def log_fids(fid: np.ndarray):
        temp = -ndimage.gaussian_laplace(fid.astype(np.float32), sigma=3)  # type: ignore
        temp -= temp.min()
        return temp


def run(
    path: Path,
    roi: str | None,
    idx: int,
    reference: str = "3_11_19",
    *,
    debug: bool = False,
    threshold: float = 6,
    fwhm: float = 4,
    max_proj: bool = True,
):
    # channels = toml.load("/fast2/3t3clean/channels.toml")
    logger.info("Reading files")

    if roi:
        imgs: dict[str, Image] = {
            file.name: Image.from_file(file)
            for file in sorted(Path(path).glob(f"{((roi) + '--') if roi else ''}*/*-{idx:04d}.tif"))
            if not file.name.startswith("10x")
        }
    else:
        imgs = {
            file.name: Image.from_file(file)
            for file in sorted(Path(path).rglob(f"*-{idx:04d}.tif"))
            if not file.name.startswith("10x")
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
        fids, reference=reference, debug=debug, iterations=3, threshold=threshold, fwhm=fwhm
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

    del imgs

    # Split into individual bits.
    # Spillover correction, max projection

    bits, bits_shifted = parse_nofids(nofids, shifts, channels, max_proj=max_proj, z_slice=np.s_[:])
    ref = sitk.Cast(sitk.GetImageFromArray(next(iter(bits.values()))), sitk.sitkFloat32)
    transformed = {}
    for i, (name, img) in enumerate(bits.items()):
        c = str(channels[name])
        img = img.astype(np.float32)
        # Within-tile alignment. Chromatic corrections.
        t_ = affine(img, c, -bits_shifted[name], ref)
        # Illumination correction
        transformed[name] = np.stack(basic[c].transform(t_)).astype(np.uint16)
        logger.debug(f"Transformed {name}: max={t_.max()}, min={t_.min()}")

    downsample = 2

    out = np.zeros(
        [
            *([] if max_proj else [next(iter(nofids.values())).shape[0]]),
            len(transformed),
            1988 // downsample,
            1988 // downsample,
        ],
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        if max_proj:
            out[i] = v[0, 30:-30:downsample, 30:-30:downsample]
        else:
            out[:, i] = v[:, 30:-30:downsample, 30:-30:downsample]

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))
    # (path / "down2").mkdir(exist_ok=True)
    logger.info(f"Writing to {path / 'down2' / f'full_{idx:04d}.tif'}")
    # tifffile.imwrite(
    #     path / f"full_{idx:04d}.tif",
    #     out,
    #     compression=22610,
    #     compressionargs={"level": 0.9},
    #     metadata={"axes": ("" if max_proj else "Z") + "CYX", "channels": ",".join(keys)},
    #     imagej=True,
    # )

    for i in range(0, len(out), 3):
        (path / "down2" / str(i)).mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(
            path / "down2" / str(i) / f"{idx:03d}_{i:03d}.tif",
            out[i : i + 3] >> 4,
            compression=22610,
            compressionargs={"level": 0.9},
            metadata={"axes": "CYX", "channels": ",".join(keys[i : i + 3])},
            imagej=True,
        )


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("idx", type=int)
@click.option("--roi", type=str)
@click.option("--debug", is_flag=True)
@click.option("--threshold", type=float, default=2.0)
@click.option("--fwhm", type=float, default=4.0)
@click.option("--reference", "-r", type=str)
def main(
    path: Path,
    idx: int,
    reference: str,
    roi: str | None = None,
    debug: bool = False,
    threshold: float = 2,
    fwhm: float = 4,
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
        threshold=threshold,
        fwhm=fwhm,
        max_proj=True,
    )


if __name__ == "__main__":
    main()

# %%
