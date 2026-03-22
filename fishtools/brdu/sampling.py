from __future__ import annotations

import numpy as np


def proportional_sample_sizes(*, counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate ``total`` samples across groups while preserving observed ratios as closely as possible."""

    counts_arr = np.asarray(counts, dtype=int).reshape(-1)
    if counts_arr.ndim != 1:
        raise ValueError(f"`counts` must be 1D, found shape {counts_arr.shape}.")
    if np.any(counts_arr < 0):
        raise ValueError("`counts` must be non-negative.")
    if total < 0:
        raise ValueError("`total` must be non-negative.")

    available = int(counts_arr.sum())
    if total >= available:
        return counts_arr.copy()
    if available == 0 or total == 0:
        return np.zeros_like(counts_arr)

    expected = counts_arr.astype(np.float64) * (float(total) / float(available))
    alloc = np.floor(expected).astype(int)
    remainder = int(total - alloc.sum())
    if remainder > 0:
        fractional = expected - alloc
        order = np.argsort(-fractional, kind="stable")
        alloc[order[:remainder]] += 1
    return alloc
