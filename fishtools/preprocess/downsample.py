"""Shared GPU downsampling helpers used across preprocessing CLIs."""

from __future__ import annotations

import cupy as cp
import numpy as np
from cucim.skimage.transform import downscale_local_mean

ClipRange = tuple[float, float] | None


def ensure_cuda_available() -> None:
    """Verify that the CUDA runtime is accessible for CuPy operations."""

    try:
        runtime = cp.cuda.runtime  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise RuntimeError(
            "CuPy was imported without CUDA support; GPU-based downsampling cannot proceed."
        ) from exc

    try:
        count = runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            "CuPy downsampling requires a CUDA-capable GPU, but the CUDA runtime is unavailable."
        ) from exc

    if count < 1:
        raise RuntimeError("CuPy downsampling requires at least one CUDA-capable GPU.")


def gpu_downsample_xy(
    volume: np.ndarray,
    *,
    crop: int = 0,
    factor: int = 1,
    clip_range: ClipRange = None,
    output_dtype: np.dtype | type[np.generic] | None = None,
) -> np.ndarray:
    """Crop and downsample the final two spatial axes of ``volume`` using CuPy."""

    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    if crop < 0:
        raise ValueError("Crop must be non-negative.")

    ensure_cuda_available()

    xp = cp
    gpu_volume = xp.asarray(volume, dtype=xp.float32)

    if crop > 0:
        if crop * 2 >= gpu_volume.shape[-1] or crop * 2 >= gpu_volume.shape[-2]:
            raise ValueError("Crop is larger than the spatial dimensions of the image.")
        crop_slices: list[slice] = [slice(None)] * (gpu_volume.ndim - 2) + [
            slice(crop, -crop),
            slice(crop, -crop),
        ]
        gpu_volume = gpu_volume[tuple(crop_slices)]

    if factor > 1:
        if gpu_volume.shape[-2] % factor != 0 or gpu_volume.shape[-1] % factor != 0:
            raise ValueError("Downsample factor must evenly divide the cropped spatial dimensions.")

        zoom_factors: list[int] = [1] * gpu_volume.ndim
        zoom_factors[-2] = factor
        zoom_factors[-1] = factor

        gpu_volume = downscale_local_mean(gpu_volume, tuple(map(int, zoom_factors)))

    if clip_range is not None:
        gpu_volume = xp.clip(gpu_volume, clip_range[0], clip_range[1])

    result = xp.asnumpy(gpu_volume)

    return result.astype(output_dtype, copy=False)


__all__ = ["ensure_cuda_available", "gpu_downsample_xy"]
