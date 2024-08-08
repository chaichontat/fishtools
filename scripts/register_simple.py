# %%
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rich
import rich_click as click
import SimpleITK as sitk
import tifffile
from loguru import logger
from rich.logging import RichHandler

from fishtools.analysis.fiducial import calc_shift

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
console = rich.get_console()


def imread_page(path: Path | str, page: int):
    with tifffile.TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


As = {}
ats = {}
for λ in [560, 750]:
    a_ = np.loadtxt(f"650to{λ}.txt")
    A, t = a_[:9].reshape(3, 3), a_[-3:]
    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    t[2] = 0
    As[λ] = A
    ats[λ] = t


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def st(img: np.ndarray[np.uint16, Any], transform: sitk.Transform, ref_img: sitk.Image):
    image = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.uint16)


def run_image(key: str, img: np.ndarray[np.uint16, Any], shiftpx: np.ndarray, ref_img):
    if len(shiftpx) != 2:
        raise ValueError

    translate = sitk.TranslationTransform(3)
    translate.SetParameters((shiftpx[1], shiftpx[0], 0.0))

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(As[750].flatten())
    affine.SetTranslation([*ats[750], 0])
    affine.SetCenter([1023.5 + shiftpx[1], 1023.5 + shiftpx[0], 0])

    comp750 = sitk.CompositeTransform(3)
    comp750.AddTransform(translate)
    comp750.AddTransform(affine)

    if img.shape[1] == 2:
        return {
            key.split("-")[0].split("_")[0]: st(img[:, 0], translate, ref_img),
            key.split("-")[0].split("_")[1]: st(img[:, 1], comp750, ref_img),
        }

    elif img.shape[1] == 3:
        affine560 = sitk.AffineTransform(3)
        affine560.SetMatrix(As[560].flatten())
        affine560.SetTranslation([*ats[560], 0])
        affine560.SetCenter([1023.5 + shiftpx[1], 1023.5 + shiftpx[0], 0])

        comp560 = sitk.CompositeTransform(3)
        comp560.AddTransform(translate)
        comp560.AddTransform(affine560)
        return {
            key.split("-")[0].split("_")[0]: st(img[:, 0], comp560, ref_img),
            key.split("-")[0].split("_")[1]: st(img[:, 1], translate, ref_img),
            key.split("-")[0].split("_")[2]: st(img[:, 2], comp750, ref_img),
        }
    else:
        raise ValueError(f"Wrong number of channels. {img.shape[1]}")


def try_int(x: str):
    try:
        return f"{int(x):02d}"
    except ValueError:
        return x


def run(path: Path, idx: int, ref: str, n_channels: int = 3):
    imgs = {
        file.name: tifffile.imread(file)
        for file in path.glob(f"*-{idx:03d}.tif")
        if not file.name.startswith("aligned")
    }
    nofids = {
        name: img[:-1].reshape(-1, n_channels, 2048, 2048)[3:].max(axis=0, keepdims=True)
        for name, img in imgs.items()
    }
    fids = {name: img[-1] for name, img in imgs.items()}
    del imgs

    # ref_clipped = np.clip(fids[ref], 0, 9000)
    shifts: dict[str, np.ndarray] = {}
    for k, v in fids.items():
        shifts[k] = np.array([0.0, 0.0])  # TODO: Zero here  # calc_shift(ref_clipped, np.clip(v, 0, 9000))
        if np.abs(shifts[k]).max() > 20:
            logger.warning(f"Image {idx} - Shift {k}: {shifts[k]} > 20 px")

    ref_img = sitk.Cast(sitk.GetImageFromArray(nofids[ref][:, 0]), sitk.sitkFloat32)
    transformed = {}
    for name, img in nofids.items():
        transformed |= run_image(name, img, -shifts[name], ref_img)
    transformed = {k: transformed[k][:, 24:2024, 24:2024] for k in sorted(transformed, key=try_int)}
    tifffile.imwrite(
        path / f"aligned-{idx:03d}.tif",
        np.concatenate(list(transformed.values())),
        compression=22610,
        imagej=True,
        metadata={"axes": "CYX"},
        compressionargs={"level": 98},
    )
    logger.info(f"Finished {idx}")


with ThreadPoolExecutor(8) as exc:
    path = Path("/fast2/alinastar")
    # files = sorted(path.glob("2_10_18-*.tif"))
    futs = [exc.submit(run, path, idx, f"2_10_18-{idx:03d}.tif") for idx in range(589)]
    for fut in futs:
        fut.result()


# %%
