from __future__ import annotations

import functools
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve as cconvolve
from loguru import logger
from tifffile import imread

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PSF_FILENAME = "PSF GL.tif"

if not DATA_DIR.joinpath(PSF_FILENAME).exists():
    raise FileNotFoundError(f"Missing PSF reference: {DATA_DIR / PSF_FILENAME}")


EPS = np.float32(1e-9)
I_MAX = np.float32(2**16 - 1)


def center_index(center: int, nz: int, step: int) -> list[int]:
    indices: list[int] = []
    curr = center
    while curr > 0:
        indices.insert(0, curr)
        curr -= step
    indices.extend(list(range(center, nz, step))[1 : len(indices)])
    return indices


def _calculate_projectors_3d(
    pf: cp.ndarray,
    σ_G: float,
    a: float = 0.001,
    b: float = 0.001,
    n: int = 8,
) -> tuple[cp.ndarray, cp.ndarray]:
    pfFFT = cp.fft.fftn(pf)
    α = a
    β = b

    kc = 1.0 / (0.5 * 2.355 * σ_G)
    kz = cp.fft.fftfreq(pfFFT.shape[0])
    kw = cp.fft.fftfreq(pfFFT.shape[1])
    kk = cp.sqrt((cp.array(cp.meshgrid(kz, kw, kw, indexing="ij")) ** 2).sum())

    bWiener = pfFFT / (cp.abs(pfFFT) ** 2 + α)
    eps = cp.sqrt(1.0 / (β**2) - 1)
    bBWorth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    pbFFT = bWiener * bBWorth
    pb = cp.real(cp.fft.ifftn(pbFFT))

    return pf, pb


def make_projector(
    path: Path,
    *,
    step: int = 10,
    max_z: int = 7,
    size: int = 31,
    center: int = 50,
    σ_G: float = 1.7,
    a: float = 0.02,
    b: float = 0.02,
    n: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    gen = imread(path)
    if gen.ndim != 3:
        raise ValueError(f"Expected 3D PSF volume, got shape {gen.shape} for {path}")
    if gen.shape[1] != gen.shape[2]:
        raise ValueError(f"PSF must be square in XY, got {gen.shape[1:]} for {path}")

    zs = [z for z in center_index(center, gen.shape[0], step) if 0 <= z < gen.shape[0]]
    if not zs:
        logger.warning(
            "PSF selection yielded no planes with center=%s, step=%s; using middle slice.",
            center,
            step,
        )
        zs = list(range(gen.shape[0]))

    if len(zs) < max_z:
        logger.warning(
            "PSF has only %s slices for requested max_z=%s; reducing max_z to %s.",
            len(zs),
            max_z,
            len(zs),
        )
        max_z = len(zs)

    z_crop = (len(zs) - max_z) // 2
    zs = zs[z_crop:-z_crop] if z_crop > 0 else zs

    spatial_size = min(size, gen.shape[1], gen.shape[2])
    if spatial_size % 2 == 0 and spatial_size > 1:
        spatial_size -= 1
    crop_y = max((gen.shape[1] - spatial_size) // 2, 0)
    crop_x = max((gen.shape[2] - spatial_size) // 2, 0)
    end_y = crop_y + spatial_size
    end_x = crop_x + spatial_size
    psf = gen[zs][:, crop_y:end_y, crop_x:end_x]
    if psf.size == 0:
        raise ValueError(
            "PSF crop resulted in empty array; verify `size`, `center`, and `step` against PSF volume."
        )
    psf = psf.astype(np.float32)
    psf /= psf.sum()
    logger.debug("PSF shape: %s", psf.shape)
    pf, pb = _calculate_projectors_3d(cp.array(psf), σ_G=σ_G, a=a, b=b, n=n)
    forward = pf.get() if hasattr(pf, "get") else pf
    backward = pb.get() if hasattr(pb, "get") else pb
    return forward[:, np.newaxis, ...], backward[:, np.newaxis, ...]


def _cache_path_for_step(step: int) -> Path:
    return DATA_DIR / f"PSF GL_step{step}.npy"


def _ensure_projector_cache(step: int) -> Path:
    psf_path = DATA_DIR / PSF_FILENAME
    if not psf_path.exists():
        raise FileNotFoundError(f"PSF reference not found: {psf_path}")
    cache_path = _cache_path_for_step(step)
    if cache_path.exists():
        return cache_path
    forward, backward = make_projector(psf_path, step=step)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        np.save(fh, [forward, backward])
    return cache_path


@functools.cache
def load_projectors_cached(step: int = 6) -> tuple[cp.ndarray, cp.ndarray]:
    """Return CuPy projectors for the requested PSF step."""
    cache_path = _ensure_projector_cache(step)
    forward, backward = np.load(cache_path, allow_pickle=True)
    return cp.asarray(forward, dtype=cp.float32), cp.asarray(backward, dtype=cp.float32)


def deconvolve_lucyrichardson_guo(
    img: cp.ndarray,
    projectors: tuple[cp.ndarray, cp.ndarray],
    iters: int = 1,
) -> cp.ndarray:
    if img.dtype not in (cp.float32, cp.float16):
        raise ValueError("Image must be float32")
    forward_projector, backward_projector = projectors
    estimate = cp.clip(img, EPS, None)
    if iters > 1:
        raise NotImplementedError("Only single-iteration supported")

    filtered_estimate = cconvolve(estimate, forward_projector, mode="reflect").clip(
        EPS,
        I_MAX,
    )
    img /= filtered_estimate
    estimate *= cconvolve(img, backward_projector, mode="reflect")
    return estimate


def rescale(img: cp.ndarray, scale: float) -> np.ndarray:
    return ((img - img.min()) * scale).get().astype(np.uint16)
