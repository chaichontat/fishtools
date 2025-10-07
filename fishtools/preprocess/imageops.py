from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clip_range_for_dtype(dtype: np.dtype) -> tuple[float, float] | None:
    """Return the numeric clip range for an integer dtype.

    For non-integer dtypes, returns None so callers can skip clipping.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return None


def crop_xy(array: np.ndarray, trim: int) -> np.ndarray:
    """Crop equally from both X and Y borders of the last two axes.

    Args:
        array: Input array with at least 2 spatial trailing dimensions.
        trim: Pixels to trim from each edge; 0 is a no-op.
    """
    if trim <= 0:
        return array
    if trim * 2 >= array.shape[-2] or trim * 2 >= array.shape[-1]:
        raise ValueError("Trim size is larger than the spatial dimensions of the image.")
    idx = [slice(None)] * array.ndim
    idx[-2] = slice(trim, -trim)
    idx[-1] = slice(trim, -trim)
    return array[tuple(idx)]


def apply_low_range_zcyx(
    img: NDArray[np.generic],
    low: NDArray[np.floating],
    rng: NDArray[np.floating],
    clamp: tuple[float, float] = (1.0 / 3.0, 3.0),
) -> NDArray[np.float32]:
    """Apply LOW/RANGE correction to a sanitized Z×C×Y×X stack.

    Computes: out = max(img - LOW, 0) * clip(RANGE, clamp[0], clamp[1]) and returns float32.

    Preconditions (inputs are already sanitized upstream):
    - ``img`` has shape (Z, C, Y, X)
    - ``low`` has shape (C, Y, X)
    - ``rng`` has shape (C, Y, X)

    Args:
        img: Input image stack with axes Z, C, Y, X (any numeric dtype).
        low: Additive LOW field, shape (C, Y, X).
        rng: Multiplicative RANGE field, shape (C, Y, X).
        clamp: RANGE clamp bounds (min, max), default (1/3, 3).

    Returns:
        Float32 array with the same shape as ``img``.

    Raises:
        ValueError: If shapes do not strictly match the Z×C×Y×X and C×Y×X expectations.
    """
    if img.ndim != 4:
        raise ValueError(f"img must be Z×C×Y×X; got shape {img.shape!r}")
    if low.ndim != 3:
        raise ValueError(f"low must be (C,Y,X); got shape {low.shape!r}")
    if rng.ndim != 3:
        raise ValueError(f"rng must be (C,Y,X); got shape {rng.shape!r}")

    z, c, y, x = img.shape
    if low.shape != (c, y, x):
        raise ValueError(f"low shape {low.shape!r} must equal (C,Y,X)={(c, y, x)!r} inferred from img")
    if rng.shape != (c, y, x):
        raise ValueError(f"rng shape {rng.shape!r} must equal (C,Y,X)={(c, y, x)!r} inferred from img")

    cmin, cmax = float(clamp[0]), float(clamp[1])
    if not (np.isfinite(cmin) and np.isfinite(cmax) and 0.0 < cmin <= cmax):
        raise ValueError(f"Invalid clamp bounds {clamp!r}; require 0 < min <= max")

    out = img.astype(np.float32, copy=True)

    # Broadcast across Z via leading axis of size 1 for per-channel fields
    out -= low[None, ...].astype(np.float32, copy=False)
    np.maximum(out, 0.0, out=out)

    rng_clamped = np.clip(rng.astype(np.float32, copy=False), cmin, cmax)
    out *= rng_clamped[None, ...]

    return out
