from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np


def _choose_offsets(strides: tuple[int, int, int], *, seed: int | None = None) -> tuple[int, int, int]:
    SZ, SY, SX = strides
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    off_z = int(rng.integers(SZ)) if SZ > 1 else 0
    off_y = int(rng.integers(SY)) if SY > 1 else 0
    off_x = int(rng.integers(SX)) if SX > 1 else 0
    return off_z, off_y, off_x


def sample_histograms(
    res: cp.ndarray,
    *,
    bins: int = 8192,
    strides: tuple[int, int, int] = (4, 2, 2),
    offsets: tuple[int, int, int] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute per-channel histograms on a subsample of a (Z,C,Y,X) CuPy array.

    - Uses per-channel min/max from the same subsample to build bin edges.
    - Returns CPU-friendly payload (numpy arrays) ready to serialize.
    """
    if res.ndim != 4:
        raise ValueError("res must be (Z,C,Y,X)")

    Z, C, Y, X = map(int, res.shape)
    SZ, SY, SX = strides
    if offsets is None:
        offsets = _choose_offsets(strides, seed=seed)
    off_z, off_y, off_x = offsets

    # compute per-channel mins/maxs on the subsample
    mins_ch = cp.min(res[off_z::SZ, :, off_y::SY, off_x::SX], axis=(0, 2, 3))  # (C,)
    maxs_ch = cp.max(res[off_z::SZ, :, off_y::SY, off_x::SX], axis=(0, 2, 3))  # (C,)

    counts_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    for c in range(C):
        lo = float(cp.asnumpy(mins_ch[c]))
        hi = float(cp.asnumpy(maxs_ch[c]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        edges_gpu = cp.linspace(lo, hi, int(bins) + 1, dtype=cp.float32)
        sample_c = res[off_z::SZ, c, off_y::SY, off_x::SX]
        h_gpu, _ = cp.histogram(sample_c, bins=edges_gpu)
        counts_list.append(cp.asnumpy(h_gpu).astype(np.int64, copy=False))
        edges_list.append(cp.asnumpy(edges_gpu).astype(np.float32, copy=False))

    n_samp = int(res[off_z::SZ, 0, off_y::SY, off_x::SX].size)
    payload = {
        "C": int(C),
        "bins": int(bins),
        "counts": counts_list,
        "edges": edges_list,
        "strides": (int(SZ), int(SY), int(SX)),
        "offsets": (int(off_z), int(off_y), int(off_x)),
        "sampled_per_channel": n_samp,
        "tile_shape": (int(Z), int(C), int(Y), int(X)),
    }
    return payload


def _quantile_from_hist(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """
    Linear CDF interpolation within bins. `counts` length = len(edges)-1.
    Returns approximate quantile value in the same units as edges.
    """
    counts = counts.astype(np.float64, copy=False)
    total = counts.sum()
    if total <= 0:
        return float(edges[0])
    cdf = np.cumsum(counts) / total
    mids = 0.5 * (edges[:-1] + edges[1:])

    idx = int(np.searchsorted(cdf, q, side="left"))
    if idx <= 0:
        return float(mids[0])
    if idx >= mids.size:
        return float(mids[-1])
    x0, x1 = mids[idx - 1], mids[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    t = 0.0 if y1 <= y0 else (q - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))
