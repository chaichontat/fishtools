from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cupy as cp
import matplotlib
import numpy as np
import polars as pl
from loguru import logger

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

I_MAX = 2**16 - 1  # use 65535 unless you must reserve it


def _rebin_counts_to_global(
    src_edges: np.ndarray,  # shape (Ns+1,)
    src_counts: np.ndarray,  # shape (Ns,)
    dst_edges: np.ndarray,  # shape (Nd+1,)
) -> np.ndarray:
    """
    Rebin histogram counts from piecewise-constant bins defined by src_edges
    into dst_edges. Assumes uniform density within each source bin and uses
    exact length-weighted overlap.
    Returns counts on dst grid (shape (Nd,)).
    """
    Ns = src_counts.size
    Nd = dst_edges.size - 1
    out = np.zeros(Nd, dtype=np.float64)
    i = j = 0
    s_left = src_edges[0]
    s_right = src_edges[1]
    d_left = dst_edges[0]
    d_right = dst_edges[1]

    # Fast‐forward to any initial non‐overlap (if grids don't start aligned)
    while i < Ns and s_right <= d_left:
        i += 1
        if i < Ns:
            s_left, s_right = src_edges[i], src_edges[i + 1]

    while i < Ns and j < Nd:
        # no overlap: advance the earlier interval
        if s_right <= d_left:
            i += 1
            if i < Ns:
                s_left, s_right = src_edges[i], src_edges[i + 1]
            continue
        if d_right <= s_left:
            j += 1
            if j < Nd:
                d_left, d_right = dst_edges[j], dst_edges[j + 1]
            continue

        # overlap length
        left = max(s_left, d_left)
        right = min(s_right, d_right)
        if right > left:
            # fraction of the src bin going into this dst bin
            frac = (right - left) / (s_right - s_left) if s_right > s_left else 0.0
            out[j] += src_counts[i] * frac

        # advance whichever interval ends first
        if s_right <= d_right:
            i += 1
            if i < Ns:
                s_left, s_right = src_edges[i], src_edges[i + 1]
        else:
            j += 1
            if j < Nd:
                d_left, d_right = dst_edges[j], dst_edges[j + 1]

    return out


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


def _read_histogram_csv(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """
    Read one .histogram.csv using Polars for performance.

    Returns a dict mapping channel to arrays describing the histogram:
        { channel: {"edges": float64[b+1], "counts": int64[b]} }

    The CSV may include a header. If absent, we assume the column order is
    [channel, bin_left, bin_right, count].
    """
    # Detect header cheaply
    with open(path, "rb") as fh:
        first = fh.readline(2048).decode("utf-8", "ignore").lower()
    has_header = ("channel" in first) or ("bin_left" in first) or ("bin_right" in first)

    df = pl.read_csv(
        str(path),
        has_header=has_header,
        new_columns=["channel", "bin_left", "bin_right", "count"],
        schema_overrides={
            "channel": pl.Int64,
            "bin_left": pl.Float64,
            "bin_right": pl.Float64,
            "count": pl.Int64,
        },
        infer_schema_length=0,
        try_parse_dates=False,
        ignore_errors=False,
    )

    # Sort to ensure increasing bin_left within each channel
    df = df.sort(["channel", "bin_left"])  # vectorized, multi-threaded

    grouped = df.group_by("channel", maintain_order=True).agg(
        pl.col("bin_left").implode().alias("bin_left"),
        pl.col("bin_right").implode().alias("bin_right"),
        pl.col("count").implode().alias("count"),
    )

    out: dict[int, dict[str, np.ndarray]] = {}
    for row in grouped.iter_rows(named=True):
        c = int(row["channel"])  # type: ignore[arg-type]
        lefts: list[float] = row["bin_left"]
        rights: list[float] = row["bin_right"]
        counts_l: list[int] = row["count"]
        if not lefts:
            continue
        edges = np.asarray([*lefts, rights[-1]], dtype=np.float64)
        counts = np.asarray(counts_l, dtype=np.int64)
        out[c] = {"edges": edges, "counts": counts}
    return out


def precompute_global_quantization(
    path: Path,
    round_name: str,
    *,
    bins: int = 8192,
    p_low: float = 0.001,  # 0.1%
    p_high: float = 0.99999,  # 99.999%
    gamma: float = 1.05,
    i_max: int = I_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-tile deconvolved histograms (CSV) into a global per-channel histogram,
    compute (m_glob, s_glob), and write them to analysis/deconv/deconv_scaling/<round>.txt.

    Returns:
        m_glob: (C,) float64
        s_glob: (C,) float64  (includes gamma)
    """
    path = Path(path)
    scale_dir = path / "analysis" / "deconv32" / "deconv_scaling"
    scale_dir.mkdir(parents=True, exist_ok=True)
    out_txt = scale_dir / f"{round_name}.txt"
    out_json = scale_dir / f"{round_name}.json"

    # --- 1) discover histogram CSVs and/or float32 stacks ---
    hist_paths = sorted(
        (path / "analysis" / "deconv32").glob(f"{round_name}--*/*{round_name}-*.histogram.csv")
    )
    logger.info(f"[{round_name}] Aggregating {len(hist_paths)} histogram CSVs …")
    # Single pass using Polars parsing: cache per-file hists and track global ranges per channel
    ch_min: dict[int, float] = {}
    ch_max: dict[int, float] = {}
    per_file: list[dict[int, dict[str, np.ndarray]]] = []

    for p in hist_paths:
        per = _read_histogram_csv(p)
        per_file.append(per)
        for c, d in per.items():
            e = d["edges"]
            ch_min[c] = min(ch_min.get(c, e[0]), float(e[0]))
            ch_max[c] = max(ch_max.get(c, e[-1]), float(e[-1]))

    if not ch_min:
        raise RuntimeError(f"[{round_name}] No histogram content found in CSVs.")

    channels = sorted(ch_min.keys())
    C = len(channels)

    # Build global edges per channel
    glob_edges: dict[int, np.ndarray] = {
        c: np.linspace(ch_min[c], ch_max[c], bins + 1, dtype=np.float64) for c in channels
    }
    # Global counts per channel
    glob_counts: dict[int, np.ndarray] = {c: np.zeros(bins, dtype=np.float64) for c in channels}

    # Rebin each cached per-file histogram into the global edges and accumulate
    for per in per_file:
        for c in channels:
            if c not in per:
                continue
            src_e = per[c]["edges"].astype(np.float64, copy=False)
            src_h = per[c]["counts"].astype(np.float64, copy=False)
            dst_e = glob_edges[c]
            glob_counts[c] += _rebin_counts_to_global(src_e, src_h, dst_e)

    # Compute m_glob / s_glob and retain low/high quantiles per channel
    m_glob = np.zeros(C, dtype=np.float64)
    s_glob = np.zeros(C, dtype=np.float64)
    q_lo = np.zeros(C, dtype=np.float64)
    q_hi = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(channels):
        edges = glob_edges[c]
        counts = glob_counts[c]
        lo = _quantile_from_hist(counts, edges, p_low)
        hi = _quantile_from_hist(counts, edges, p_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            logger.warning(f"[{round_name}] channel {c}: degenerate quantiles; falling back to edges.")
            lo, hi = float(edges[0]), float(edges[-1])
        m_glob[i] = lo
        q_lo[i] = lo
        q_hi[i] = hi
        dyn = (hi - lo) * float(gamma)
        s_glob[i] = (i_max / dyn) if dyn > 0 else 1.0

    # Persist
    arr = np.vstack([m_glob, s_glob])
    np.savetxt(out_txt, arr)
    meta = {
        "round": round_name,
        "source": "histogram_csv",
        "files": len(hist_paths),
        "channels_indexed": channels,
        "bins": int(bins),
        "p_low": float(p_low),
        "p_high": float(p_high),
        "gamma": float(gamma),
        "i_max": int(i_max),
        "notes": "Aggregated per-tile deconvolved histograms with rebinning to a global grid.",
    }
    out_json.write_text(json.dumps(meta, indent=2))
    # --- Save a logarithmic histogram plot (counts on log-scale) ---
    try:
        out_png = scale_dir / f"{round_name}.hist.png"
        logger.info(f"[{round_name}] Histogram plot path: {out_png}")
        # Use standard seaborn styling
        sns.set_style("whitegrid")

        # Subplot grid: per-channel panel to draw channel-specific cutoffs cleanly
        ncols = 4 if C >= 4 else C
        nrows = int(np.ceil(C / ncols)) if C > 0 else 1
        # Make the plot larger for readability
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False)
        axes = axes.ravel()

        # Format percentiles for legend labels (trim trailing zeros)
        p_low_pct = ("%.5f" % (p_low * 100)).rstrip("0").rstrip(".")
        p_high_pct = ("%.5f" % (p_high * 100)).rstrip("0").rstrip(".")

        for i, c in enumerate(channels):
            ax = axes[i]
            edges = glob_edges[c]
            counts = glob_counts[c]
            mids = 0.5 * (edges[:-1] + edges[1:])
            # Clip out non-positive intensities for log-scale x-axis
            mask = mids > 0
            if not np.any(mask):
                ax.text(0.5, 0.5, "no >0 bins", transform=ax.transAxes, ha="center", va="center")
            else:
                ax.plot(mids[mask], counts[mask] + 1.0)
                ax.set_xscale("log")
                ax.set_yscale("log")
            # Vertical lines at low/high cutoffs with percentile labels (only if >0)
            if q_lo[i] > 0:
                ax.axvline(
                    q_lo[i], color="tab:red", linestyle="--", linewidth=1.2, label=f"low {p_low_pct}%"
                )
            if q_hi[i] > 0:
                ax.axvline(
                    q_hi[i], color="tab:green", linestyle="--", linewidth=1.2, label=f"high {p_high_pct}%"
                )
            ax.set_title(f"ch {c}")
            ax.set_xlabel("Intensity (log)")
            ax.set_ylabel("Count (log)")
            if i == 0:
                ax.legend(fontsize=8, loc="best")

        # Hide any unused axes
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        fig.suptitle(f"Global Histograms with Cutoffs (round={round_name})", y=0.995)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        logger.info(f"[{round_name}] Saved histogram plot to {out_png}")
    except Exception as e:
        logger.warning(f"[{round_name}] Failed to write histogram plot: {e}")
    logger.info(f"[{round_name}] Wrote global quantization to {out_txt}")
    return m_glob, s_glob


ArrayLike = Union[np.ndarray, cp.ndarray]


def quantize_global(
    res: ArrayLike,
    m_glob: np.ndarray,
    s_glob: np.ndarray,
    *,
    i_max: int = 2**16 - 1,  # use 65535; set 65534 if you reserve 65535
    return_stats: bool = False,
    as_numpy: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Map deconvolved float32 data to uint16 with a single global affine per channel.

    code = s_glob[c] * (res - m_glob[c]), rounded, then clipped to [0, i_max]

    Args
    ----
    res : (Z, C, Y, X) float32 array (cupy or numpy accepted)
        Deconvolved, BaSiC-corrected data. If (C, Y, X), Z is assumed 1.
    m_glob : (C,) array-like (float)
        Global low quantile per channel.
    s_glob : (C,) array-like (float)
        Global scale per channel (already includes any headroom factor).
    i_max : int
        Saturation code value (65535 by default).
    return_stats : bool
        If True, also return dict with clipping stats.
    as_numpy : bool
        If True, returns a NumPy array (moves from GPU if needed). If False and
        `res` is a CuPy array, returns a CuPy array.

    Returns
    -------
    out_uint16 : (-1, Y, X) uint16 array
        Z and C are flattened to the first axis to match your file layout.
    stats : dict (optional)
        {'clipped_low': int, 'clipped_high': int, 'total': int} for quick telemetry.

    Notes
    -----
    - Rounds with unbiased `rint` before casting to uint16 to avoid truncation bias.
    - Keeps everything in the same array module as `res` (CPU/GPU) until the final
      return; set `as_numpy=False` to keep it on GPU.
    """
    # Accept (C, Y, X) by promoting to (1, C, Y, X)
    if res.ndim == 3:
        res = res[None, ...]
    if res.ndim != 4:
        raise ValueError("res must have shape (Z, C, Y, X) or (C, Y, X)")

    Z, C, Y, X = map(int, res.shape)

    # Pick array module based on input
    xp = cp.get_array_module(res) if isinstance(res, (cp.ndarray,)) else np

    # Ensure dtype float32 for arithmetic
    res = res.astype(xp.float32, copy=False)

    # Broadcast global affine over channel axis
    m = xp.asarray(m_glob, dtype=xp.float32)[xp.newaxis, :, xp.newaxis, xp.newaxis]  # (1,C,1,1)
    s = xp.asarray(s_glob, dtype=xp.float32)[xp.newaxis, :, xp.newaxis, xp.newaxis]

    # Affine map, clip, round, cast
    arr = s * (res - m)
    xp.clip(arr, 0.0, float(i_max), out=arr)
    arr = xp.rint(arr)

    # Optional stats (computed before casting; cheap)
    stats: Optional[Dict[str, Any]] = None
    if return_stats:
        clipped_low = int((arr <= 0).sum().item())
        clipped_high = int((arr >= float(i_max)).sum().item())
        total = int(arr.size)
        stats = {"clipped_low": clipped_low, "clipped_high": clipped_high, "total": total}

    arr = arr.astype(xp.uint16, copy=False)  # (Z, C, Y, X)
    arr = arr.reshape(Z * C, Y, X)  # (-1, Y, X)

    # Move to NumPy if requested or if input was NumPy
    if as_numpy or xp is np:
        out = arr.get() if xp is cp else arr
    else:
        out = arr  # keep on GPU

    return (out, stats) if return_stats else out


if __name__ == "__main__":
    # Example manual invocation; keep out of import side-effects
    precompute_global_quantization(
        "/working/20250929_JaxA3_Coro4",
        round_name="pi",
    )
