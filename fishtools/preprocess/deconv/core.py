from __future__ import annotations

import fcntl
import functools
import tempfile
from os import environ, fsync
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve as cconvolve
from loguru import logger
from tifffile import imread

DATA_DIR = (
    Path(environ["DATA_PATH"]).expanduser().resolve()
    if "DATA_PATH" in environ
    else (Path(__file__).resolve().parent.parent.parent.parent / "data")
)
PSF_FILENAME = "PSF GL.tif"


PRENORMALIZED = "prenormalized"

if not DATA_DIR.joinpath(PSF_FILENAME).exists():
    raise FileNotFoundError(f"Missing PSF reference: {DATA_DIR / PSF_FILENAME}")


EPS = np.float32(1e-9)
I_MAX = np.float32(2**16 - 1)


@functools.cache
def projectors(step: int = 6):
    psf_path = Path(DATA_DIR / PSF_FILENAME)
    cache_path = psf_path.with_name(f"{psf_path.stem}.step{step}.npy")

    if not cache_path.exists():
        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        with open(lock_path, "w") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            if not cache_path.exists():
                make_projector(psf_path, step=step, max_z=7, output=cache_path)

    return tuple([x.astype(cp.float32) for x in cp.load(cache_path)])


def center_index(center: int, nz: int, step: int):
    ns = []
    curr = center
    while curr > 0:
        ns.insert(0, curr)
        curr -= step
    ns.extend(list(range(center, nz, step))[1 : len(ns)])
    return ns


def reverse_psf(inPSF: np.ndarray):
    Sx, Sy, Sz = inPSF.shape
    outPSF = np.zeros_like(inPSF)

    i = np.arange(Sx)
    j = np.arange(Sy)
    k = np.arange(Sz)

    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    outPSF = inPSF[Sx - 1 - ii, Sy - 1 - jj, Sz - 1 - kk]

    return outPSF


def _calculate_projectors_3d(
    pf: cp.ndarray, σ_G: float, a: float = 0.001, b: float = 0.001, n: int = 8
) -> tuple[cp.ndarray, cp.ndarray]:
    pfFFT = cp.fft.fftn(pf)
    α = a
    β = b

    # This is the cut-off frequency.
    kc = 1.0 / (0.5 * 2.355 * σ_G)

    # FFT frequencies
    kz = cp.fft.fftfreq(pfFFT.shape[0])
    kw = cp.fft.fftfreq(pfFFT.shape[1])
    kk = cp.sqrt((cp.array(cp.meshgrid(kz, kw, kw, indexing="ij")) ** 2).sum())

    # Wiener filter
    bWiener = pfFFT / (cp.abs(pfFFT) ** 2 + α)
    # Butterworth filter
    # OTF_butterworth = 1/sqrt(1+ee*(kx/kcx)^pn)
    eps = cp.sqrt(1.0 / (β**2) - 1)
    # https://github.com/eguomin/regDeconProject/blob/3ca800fce083f2e936105f8334bf5ecc6ee8438b/WBDeconvolution/BackProjector.m#L223
    bBWorth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    pbFFT = bWiener * bBWorth
    pb = cp.real(cp.fft.ifftn(pbFFT))

    return pf, pb


def make_projector(
    path: Path | str,
    *,
    step: int = 10,
    max_z: int = 7,
    size: int = 31,
    center: int = 50,
    output: Path | None = None,
):
    gen = imread(path := Path(path))
    assert gen.shape[1] == gen.shape[2]

    zs = center_index(center, gen.shape[0], step)
    z_crop = (len(zs) - max_z) // 2
    zs = zs[z_crop:-z_crop] if z_crop > 0 else zs

    crop = (gen.shape[1] - size) // 2
    # print(gen.shape, zs, z_crop)
    psf = gen[zs][::-1, crop:-crop, crop:-crop]
    psf /= psf.sum()
    logger.debug(f"PSF shape: {psf.shape}")
    print(path)
    p = [
        cp.asnumpy(x)[:, np.newaxis, ...]
        for x in _calculate_projectors_3d(cp.array(psf), σ_G=1.7, a=0.02, b=0.02, n=10)
    ]

    out_path = (Path(output) if output is not None else path.with_name(f"{path.stem}.step{step}.npy")).resolve()
    if out_path.exists():
        return p

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".partial",
        delete=False,
    ) as tmp_fp:
        np.save(tmp_fp, p)
        tmp_fp.flush()
        fsync(tmp_fp.fileno())
        tmp_path = Path(tmp_fp.name)

    try:
        try:
            out_path.hardlink_to(tmp_path)
        except FileExistsError:
            return p
        except OSError:
            if out_path.exists():
                return p
            tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return p


def deconvolve_lucyrichardson_guo(
    img: np.ndarray[cp.float32, Any],
    projectors: tuple[cp.ndarray, cp.ndarray],
    iters: int = 1,
) -> cp.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:
    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.
    """

    if img.dtype not in [cp.float32, cp.float16]:
        raise ValueError("Image must be float32")
    forward_projector, backward_projector = projectors

    estimate = cp.clip(img, EPS, None)
    if iters > 1:
        raise NotImplementedError

    filtered_estimate = cconvolve(estimate, forward_projector, mode="reflect").clip(
        EPS,
        I_MAX,
    )
    img /= filtered_estimate
    # Correction
    estimate *= cconvolve(img, backward_projector, mode="reflect")

    return estimate


def rescale(img: cp.ndarray, scale: float):
    return ((img - img.min()) * scale).get().astype(np.uint16)
