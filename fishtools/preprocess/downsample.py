"""Shared downsampling helpers used across preprocessing CLIs.

Provides GPU-accelerated downsampling when CUDA is available,
with automatic fallback to CPU-based processing.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

ClipRange = tuple[float, float] | None

# Lazy-loaded modules
_cp = None
_cucim_downscale = None
_skimage_downscale = None
_cuda_available: bool | None = None


def _check_cuda_available() -> bool:
    """Check if CUDA is available for GPU operations."""
    global _cuda_available, _cp, _cucim_downscale
    if _cuda_available is not None:
        return _cuda_available

    try:
        import cupy as cp
        _cp = cp
        runtime = cp.cuda.runtime
        count = runtime.getDeviceCount()
        if count < 1:
            _cuda_available = False
        else:
            from cucim.skimage.transform import downscale_local_mean
            _cucim_downscale = downscale_local_mean
            _cuda_available = True
    except Exception:
        _cuda_available = False

    return _cuda_available


def _get_skimage_downscale():
    """Lazy load skimage downscale_local_mean."""
    global _skimage_downscale
    if _skimage_downscale is None:
        from skimage.transform import downscale_local_mean
        _skimage_downscale = downscale_local_mean
    return _skimage_downscale


def _cpu_downsample_xy(
    volume: np.ndarray,
    *,
    crop: int = 0,
    factor: int = 1,
    clip_range: ClipRange = None,
    output_dtype: np.dtype | type[np.generic] | None = None,
) -> np.ndarray:
    """CPU-based crop and downsample using skimage."""
    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    if crop < 0:
        raise ValueError("Crop must be non-negative.")

    result = volume.astype(np.float32, copy=True)

    if crop > 0:
        if crop * 2 >= result.shape[-1] or crop * 2 >= result.shape[-2]:
            raise ValueError("Crop is larger than the spatial dimensions of the image.")
        crop_slices: list[slice] = [slice(None)] * (result.ndim - 2) + [
            slice(crop, -crop),
            slice(crop, -crop),
        ]
        result = result[tuple(crop_slices)]

    if factor > 1:
        if result.shape[-2] % factor != 0 or result.shape[-1] % factor != 0:
            raise ValueError("Downsample factor must evenly divide the cropped spatial dimensions.")

        zoom_factors: tuple[int, ...] = tuple([1] * (result.ndim - 2) + [factor, factor])
        downscale = _get_skimage_downscale()
        result = downscale(result, zoom_factors)

    if clip_range is not None:
        result = np.clip(result, clip_range[0], clip_range[1])

    return result.astype(output_dtype, copy=False)


def _gpu_downsample_xy(
    volume: np.ndarray,
    *,
    crop: int = 0,
    factor: int = 1,
    clip_range: ClipRange = None,
    output_dtype: np.dtype | type[np.generic] | None = None,
) -> np.ndarray:
    """GPU-based crop and downsample using CuPy/cuCIM."""
    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    if crop < 0:
        raise ValueError("Crop must be non-negative.")

    cp = _cp
    gpu_volume = cp.asarray(volume, dtype=cp.float32)

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

        gpu_volume = _cucim_downscale(gpu_volume, tuple(map(int, zoom_factors)))

    if clip_range is not None:
        gpu_volume = cp.clip(gpu_volume, clip_range[0], clip_range[1])

    result = cp.asnumpy(gpu_volume)
    return result.astype(output_dtype, copy=False)


def downsample_xy(
    volume: np.ndarray,
    *,
    crop: int = 0,
    factor: int = 1,
    clip_range: ClipRange = None,
    output_dtype: np.dtype | type[np.generic] | None = None,
) -> np.ndarray:
    """Crop and downsample the final two spatial axes of ``volume``.

    Uses GPU acceleration when available, falls back to CPU otherwise.
    """
    if _check_cuda_available():
        return _gpu_downsample_xy(
            volume, crop=crop, factor=factor, clip_range=clip_range, output_dtype=output_dtype
        )
    else:
        logger.debug("CUDA not available, using CPU downsampling")
        return _cpu_downsample_xy(
            volume, crop=crop, factor=factor, clip_range=clip_range, output_dtype=output_dtype
        )


# Legacy alias for backwards compatibility
def gpu_downsample_xy(
    volume: np.ndarray,
    *,
    crop: int = 0,
    factor: int = 1,
    clip_range: ClipRange = None,
    output_dtype: np.dtype | type[np.generic] | None = None,
) -> np.ndarray:
    """Crop and downsample with GPU if available, CPU fallback.

    .. deprecated::
        Use :func:`downsample_xy` instead.
    """
    return downsample_xy(
        volume, crop=crop, factor=factor, clip_range=clip_range, output_dtype=output_dtype
    )


def ensure_cuda_available() -> None:
    """Verify that the CUDA runtime is accessible for CuPy operations.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not _check_cuda_available():
        raise RuntimeError(
            "CUDA is not available. Set up a CUDA-capable GPU or use downsample_xy() for automatic fallback."
        )


__all__ = ["downsample_xy", "gpu_downsample_xy", "ensure_cuda_available"]
