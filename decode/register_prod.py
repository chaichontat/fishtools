# %%

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import SimpleITK as sitk
import tifffile
import toml
from loguru import logger
from scipy import ndimage
from scipy.ndimage import shift
from skimage.measure import block_reduce
from tifffile import imread

from fishtools import align_fiducials
from fishtools.analysis.fiducial import align_phase, phase_shift

DATA = Path("/home/chaichontat/fishtools/data")
channels = toml.load("/home/chaichontat/fishtools/decode/channels.toml")

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

dark = imread(DATA / "dark.tif").astype(np.float32)
flats = {}

for λ in ["405", "488", "560", "647", "750"]:
    flat = (imread(DATA / f"flat_{λ if λ != '750' else '647'}.tif") - dark).astype(np.float32)
    flat /= np.min(flat)  # so that we don't divide with less than 1.
    flats[λ] = flat


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    return np.clip(spillee - spiller * corr, 0, 65535)


def parse_nofids(
    nofids: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    channels: dict[str, int],
    *,
    z_slice: slice = np.s_[5:15],
    max_proj: bool = True,
):
    out, out_shift = {}, {}
    for name, img in nofids.items():
        bits = name.split("-")[0].split("_")
        assert img.shape[1] == len(bits)
        for i, bit in enumerate(bits):
            sliced = img[z_slice, i]
            out[bit] = sliced.max(0, keepdims=True) if max_proj else sliced

        cs = {str(channels[bit]): bit for bit in bits}
        if "560" in cs and "647" in cs:
            out[cs["560"]] = spillover_correction(out[cs["560"]], out[cs["647"]], 0.22)

        # if "647" in cs and "750" in cs:
        #     out[cs["647"]] = spillover_correction(out[cs["647"]], out[cs["750"]], 0.05)

    for name, shift in shifts.items():
        bits = name.split("-")[0].split("_")
        for i, bit in enumerate(bits):
            out_shift[bit] = shift

    return out, out_shift


def sort_key(x):
    try:
        return f"{int(x[0]):02d}"
    except ValueError:
        return x[0]


def st(ref: sitk.Image, img: np.ndarray[np.float32, Any], transform: sitk.Transform):
    image = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(image))


def run_image(img: np.ndarray[np.float32, Any], channel: str, shiftpx: np.ndarray, ref: sitk.Image):
    if len(shiftpx) != 2:
        raise ValueError

    translate = sitk.TranslationTransform(3)
    translate.SetParameters((float(shiftpx[0]), float(shiftpx[1]), 0.0))

    if channel == "647":
        return st(ref, img, translate)

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(As[channel].flatten())
    affine.SetTranslation([*ats[channel], 0])
    affine.SetCenter([1023.5 + shiftpx[0], 1023.5 + shiftpx[1], 0])

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(translate)
    composite.AddTransform(affine)

    return st(ref, img, composite)


def run(
    path: Path,
    idx: int,
    reference: str = "3_11_19",
    debug: bool = False,
    threshold: float = 3,
    fwhm: float = 4,
    max_proj: bool = True,
):
    logger.info("Reading files")
    imgs = {
        file.name: tifffile.imread(file)
        for file in sorted(Path(path).rglob(f"*-{idx:04d}.tif"))
        if not file.name.startswith("10x")
    }
    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")
    # Fiducials
    fids = {}
    for name, img in imgs.items():
        temp = -ndimage.gaussian_laplace(img[-1].astype(np.float32), sigma=3)
        temp -= temp.min()
        fids[name] = temp  # filtered_image

    if debug:
        _fids_path = path / f"fids_{idx}.tif"
        tifffile.imwrite(
            _fids_path,
            np.stack(list(fids.values())),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX"},
        )
        logger.debug(f"Written fiducials to {_fids_path}.")

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

    nofids = {name: img[:-1].reshape(-1, len(name.split("_")), 2048, 2048) for name, img in imgs.items()}
    del imgs

    # Split into individual bits.
    # Spillover correction, max projection
    bits, bits_shifted = parse_nofids(nofids, shifts, channels, max_proj=max_proj, z_slice=np.s_[3:])
    ref = sitk.Cast(sitk.GetImageFromArray(next(iter(bits.values()))), sitk.sitkFloat32)
    transformed = {}
    for i, (name, img) in enumerate(bits.items()):
        c = str(channels[name])
        img = img.astype(np.float32) / flats[c]
        # Within-tile alignment. Chromatic corrections.
        t_ = transformed[name] = run_image(img, c, -bits_shifted[name], ref).astype(np.uint16)
        logger.debug(f"Transformed {name}: max={t_.max()}, min={t_.min()}")

    out = np.zeros(
        [*([] if max_proj else [next(iter(nofids.values())).shape[0]]), len(transformed), 994, 994],
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        if max_proj:
            out[i] = v[0, 30:-30:2, 30:-30:2]
        else:
            out[:, i] = v[:, 30:-30:2, 30:-30:2]

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))
    (path / "down2").mkdir(exist_ok=True)
    # logger.info(f"Writing to {path / 'down2' / f'full_{idx:03d}.tif'}")
    # tifffile.imwrite(
    #     path / "down2" / f"full_{idx:03d}.tif",
    #     out,
    #     compression=22610,
    #     compressionargs={"level": 0.9},
    #     metadata={"axes": "ZCYX", "channels": ",".join(keys)},
    #     imagej=True,
    # )

    for i in range(0, len(out), 3):
        (path / "down2" / str(i)).mkdir(exist_ok=True)
        tifffile.imwrite(
            path / "down2" / str(i) / f"{idx:03d}_{i:03d}.tif",
            out[i : i + 3],
            compression=22610,
            compressionargs={"level": 0.9},
            metadata={"axes": "CYX", "channels": ",".join(keys[i : i + 3])},
            imagej=True,
        )


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("idx", type=int)
@click.option("--debug", is_flag=True)
@click.option("--threshold", type=float, default=3.0)
@click.option("--fwhm", type=float, default=4.0)
def main(path: Path, idx: int, debug: bool = False, threshold: float = 3, fwhm: float = 4):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    run(path, idx, debug=debug, threshold=threshold, fwhm=fwhm, max_proj=True)


if __name__ == "__main__":
    main()
