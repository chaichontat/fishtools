"""Shared downsampling helpers used across preprocessing CLIs.

Provides GPU-accelerated downsampling when CUDA is available,
with automatic fallback to CPU-based processing.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

ClipRange = tuple[float, float] | None

# Public, monkeypatchable GPU backends (tests rely on these names)
cp = None
downscale_local_mean = None
_skimage_downscale = None
_cuda_available: bool | None = None


def _check_cuda_available() -> bool:
    """Check if CUDA is available for GPU operations."""
    global _cuda_available, cp, downscale_local_mean
    # Cache only the positive result. A previous "no CUDA" check should not
    # permanently poison the module (tests monkeypatch `cp` at runtime).
    if _cuda_available is True:
        return True

    if cp is None:
        try:
            import cupy as imported_cupy  # pragma: no cover - exercised by integration runs
        except ImportError:
            _cuda_available = False
            return _cuda_available
        cp = imported_cupy

    cuda = getattr(cp, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    get_device_count = getattr(runtime, "getDeviceCount", None)
    if not callable(get_device_count) or int(get_device_count()) < 1:
        _cuda_available = False
        return _cuda_available

    _cuda_available = True

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
    global downscale_local_mean
    if factor < 1:
        raise ValueError("Downsample factor must be >= 1.")
    if crop < 0:
        raise ValueError("Crop must be non-negative.")

    if cp is None:
        raise RuntimeError("CUDA is not available. Set up a CUDA-capable GPU or use downsample_xy().")
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
        if downscale_local_mean is None:
            from cucim.skimage.transform import downscale_local_mean as cucim_downscale  # pragma: no cover

            downscale_local_mean = cucim_downscale
        if gpu_volume.shape[-2] % factor != 0 or gpu_volume.shape[-1] % factor != 0:
            raise ValueError("Downsample factor must evenly divide the cropped spatial dimensions.")

        zoom_factors: list[int] = [1] * gpu_volume.ndim
        zoom_factors[-2] = factor
        zoom_factors[-1] = factor

        gpu_volume = downscale_local_mean(gpu_volume, tuple(map(int, zoom_factors)))

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
    """Crop and downsample using the GPU, failing fast if CUDA is unavailable."""
    ensure_cuda_available()
    return _gpu_downsample_xy(volume, crop=crop, factor=factor, clip_range=clip_range, output_dtype=output_dtype)


def ensure_cuda_available() -> None:
    """Verify that the CUDA runtime is accessible for CuPy operations.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not _check_cuda_available():
        raise RuntimeError(
            "CUDA is not available. Set up a CUDA-capable GPU or use downsample_xy() for automatic fallback."
        )


__all__ = ["cp", "downscale_local_mean", "downsample_xy", "gpu_downsample_xy", "ensure_cuda_available"]
