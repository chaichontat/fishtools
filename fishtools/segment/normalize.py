from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from skimage.filters import unsharp_mask


def sample_percentiles(
    img: npt.ArrayLike,
    channels: list[int],
    block: tuple[int, int] = (512, 512),
    *,
    n: int = 25,
    low: float = 1,
    high: float = 99,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample percentile ranges from a large 4D stack.

    Assumes input is shaped (Z, Y, X, C) and returns per-channel
    low/high percentiles computed from up to ``n`` randomly sampled
    spatial crops of size ``block`` after a light unsharp filter.

    Behavior matches the historical implementation used by the
    distributed segmentation scripts and preserves the 1-based
    channel indexing convention those scripts expect.

    Parameters
    - img: Array-like of shape (Z, Y, X, C)
    - channels: 1-based channel indices to include (e.g., [1, 2])
    - block: (height, width) of sampled crops
    - n: max number of crops to sample
    - low, high: percentile bounds to compute (0-100)
    - seed: RNG seed for reproducibility

    Returns
    - mean_perc: array (n_channels, 2) with [[low, high], ...]
    - all_samples: array (n_samples, 2, n_channels) of individual crop percentiles
    """
    # Accept numpy, zarr, or array-likes that support shape and slicing
    arr = img  # type: ignore[assignment]
    try:
        ndim = arr.ndim  # type: ignore[attr-defined]
        shape = arr.shape  # type: ignore[attr-defined]
    except Exception:
        arr = np.asarray(img)
        ndim = arr.ndim
        shape = arr.shape

    if ndim != 4:
        raise ValueError("Expected image with shape (Z, Y, X, C).")
    if shape[1] < block[0] or shape[2] < block[1]:
        raise ValueError("Block size larger than image spatial dimensions.")

    rng = np.random.default_rng(seed)
    # Historical convention: incoming channels are 1-based
    ch_idx = [c - 1 for c in channels]

    # Over-sample starts to allow skipping zero-padded/stitch-edge crops
    x_start = rng.integers(0, shape[1] - block[0], n * 2)
    y_start = rng.integers(0, shape[2] - block[1], n * 2)

    samples: list[np.ndarray] = []
    taken = 0
    for x, y in zip(x_start, y_start):
        if taken >= n:
            break
        crop = arr[:, x : x + block[0], y : y + block[1], :]
        # Skip crops containing any zero pixel in the selected channels (avoid stitched borders)
        if np.any(crop[..., ch_idx] == 0):
            continue
        # Light sharpening before measuring percentiles
        crop_filt = unsharp_mask(crop, preserve_range=True, radius=3, channel_axis=3)
        samples.append(np.percentile(crop_filt[..., ch_idx], [low, high], axis=(0, 1, 2)))
        taken += 1

    if not samples:
        raise RuntimeError("No valid crops sampled; image may be mostly zeros or block too large.")

    all_samples = np.asarray(samples)  # (n_samples, 2, n_channels)
    mean_perc = all_samples.mean(axis=0).T  # -> (n_channels, 2)
    return mean_perc, all_samples


# Backwards-compatible name used throughout the repo
def calc_percentile(
    img: npt.ArrayLike,
    channels: list[int],
    block: tuple[int, int] = (512, 512),
    *,
    n: int = 25,
    low: float = 1,
    high: float = 99,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return sample_percentiles(img, channels, block, n=n, low=low, high=high, seed=seed)
