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

    subset = res[off_z::SZ, :, off_y::SY, off_x::SX]
    if subset.size == 0:
        subset = res
        SZ = SY = SX = 1
        off_z = off_y = off_x = 0

    # compute per-channel mins/maxs on the subsample
    mins_ch = cp.min(subset, axis=(0, 2, 3))  # (C,)
    maxs_ch = cp.max(subset, axis=(0, 2, 3))  # (C,)

    counts_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    for c in range(C):
        lo = float(cp.asnumpy(mins_ch[c]))
        hi = float(cp.asnumpy(maxs_ch[c]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        edges_gpu = cp.linspace(lo, hi, int(bins) + 1, dtype=cp.float32)
        sample_c = subset[:, c, :, :]
        h_gpu, _ = cp.histogram(sample_c, bins=edges_gpu)
        counts_list.append(cp.asnumpy(h_gpu).astype(np.int64, copy=False))
        edges_list.append(cp.asnumpy(edges_gpu).astype(np.float32, copy=False))

    n_samp = int(subset[:, 0, :, :].size)
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
    Piecewise-linear CDF interpolation for histogram quantile estimation.

    Uses an edge-based CDF definition:
      - CDF(edges[0]) = 0
      - CDF(edges[i]) = cumsum(counts[0:i]) / total  for i in 1..n-1
      - CDF(edges[n]) = 1.0

    This allows interpolation across the full range [edges[0], edges[-1]],
    with q=0 returning edges[0] and q=1 returning edges[-1].
    """
    counts = counts.astype(np.float64, copy=False)
    total = counts.sum()
    if total <= 0:
        return float(edges[0])

    # Boundary cases
    if q <= 0.0:
        return float(edges[0])
    if q >= 1.0:
        return float(edges[-1])

    # Build CDF at edges: CDF[0]=0, CDF[i]=cumsum(counts[0:i])/total, CDF[n]=1
    cumsum = np.cumsum(counts)
    cdf_at_edges = np.concatenate([[0.0], cumsum / total])  # len = n+1

    # Find which bin q falls into
    idx = int(np.searchsorted(cdf_at_edges, q, side="left"))

    # Clamp idx to valid range [1, len(edges)-1]
    idx = max(1, min(idx, len(edges) - 1))

    # Interpolate within bin [idx-1, idx]
    x0, x1 = edges[idx - 1], edges[idx]
    y0, y1 = cdf_at_edges[idx - 1], cdf_at_edges[idx]
    t = 0.0 if y1 <= y0 else (q - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))
