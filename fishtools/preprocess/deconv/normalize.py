from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import cupy as cp
import matplotlib
import numpy as np
import polars as pl
import rich_click as click
import tifffile
from loguru import logger

from fishtools.preprocess.deconv.hist import _quantile_from_hist
from fishtools.utils.pretty_print import progress_bar
from fishtools.utils.tiff import normalize_channel_names as _norm_channel_names
from fishtools.utils.tiff import read_metadata_from_tif

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

I_MAX = 2**16 - 1  # use 65535 unless you must reserve it

# Optional hardcoded overrides for upper percentile (p_high) per round and channel.
#
# Structure:
#   {
#       "<round_name>": {
#           <channel_index>: <p_high_fraction>,  # e.g., 0.9999 means 99.99%
#           ...
#       },
#       ...
#   }
#
# Leave empty by default. Populate with your experiment-specific values when you
# want automatic overrides without passing --p-high on the CLI. Channels not
# listed fall back to the CLI/default p_high.
P_HIGH_OVERRIDES: dict[str, dict[int, float]] = {
    # Example (uncomment and edit to use):
    # "1_9_17": {0: 0.9999, 1: 0.9995, 2: 0.9999},
}

# Global name-based overrides applied when channel names are discovered in TIFF metadata.
# Keys are matched case-insensitively after basic normalization.
P_HIGH_NAME_OVERRIDES_GLOBAL: dict[str, float] = {
    # Global mapping provided by user; case-insensitive matching.
    "wga": 0.99,
    "edu": 0.9999,
    "brdu": 0.9999,
    "pi": 0.999,
}


def _p_high_overrides_for(round_name: str) -> dict[int, float] | None:
    """Return per-channel p_high overrides for a round if configured.

    Values must be in (0, 1]. Invalid entries are ignored at use time.
    """
    mapping = P_HIGH_OVERRIDES.get(round_name)
    return dict(mapping) if mapping else None


def _first_tile_for_round(workspace: Path, round_name: str) -> Path | None:
    for _roi, tile in _iter_float32_tiles(workspace, round_name):
        return tile
    return None


def _normalize_name(name: str) -> str:
    """Validate and canonicalize a channel name: underscore-separated, case-insensitive.

    Rules:
    - Only letters, digits, and underscores allowed
    - Underscore is the only separator; spaces or other punctuation are invalid
    - Returns lowercased name for matching
    """
    s = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", s):
        raise click.ClickException(
            f"Invalid channel name '{name}'. Expected only letters, digits, and underscores (_)."
        )
    return s.lower()


def _p_high_by_channel_from_names(
    workspace: Path,
    round_name: str,
    name_overrides: dict[str, float],
) -> dict[int, float] | None:
    """Resolve name-based p_high overrides to channel indices for a given round.

    Looks up the first float32 tile of the round, reads channel names from metadata,
    and maps any provided name-based overrides to indices. Returns None when names
    cannot be determined.
    """
    if not name_overrides:
        return None
    tile = _first_tile_for_round(workspace, round_name)
    if tile is None:
        return None
    try:
        with tifffile.TiffFile(str(tile)) as tif:
            meta = read_metadata_from_tif(tif)
            # Determine channel count from series axes when available
            count = None
            try:
                s = tif.series[0]
                axes = getattr(s, "axes", None)
                shape = getattr(s, "shape", None)
                if axes and shape and "C" in axes:
                    count = int(shape[axes.index("C")])
                elif shape and len(shape) == 3:
                    # Heuristic: (C, Y, X)
                    count = int(shape[0])
                elif shape and len(shape) == 4:
                    # Heuristic: (Z, C, Y, X)
                    count = int(shape[1])
            except Exception:
                count = None
            if not count or count <= 0:
                return None
            names = _norm_channel_names(count, meta)
    except Exception:
        return None

    if not names:
        return None

    norm_map = {_normalize_name(k): float(v) for k, v in name_overrides.items()}
    out: dict[int, float] = {}
    for idx, nm in enumerate(names):
        key = _normalize_name(nm)
        if key in norm_map:
            val = norm_map[key]
            if 0.0 < val <= 1.0:
                out[idx] = val
    return out or None


# CLI entry points removed; this module now exposes library functions only.


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
    p_high_by_channel: dict[int, float] | None = None,
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
    # Prepare per-channel p_high to use (fall back to scalar p_high)
    effective_p_high: dict[int, float] = {}
    for c in channels:
        if p_high_by_channel and c in p_high_by_channel:
            try:
                val = float(p_high_by_channel[c])
            except Exception:
                val = float(p_high)
        else:
            val = float(p_high)
        # clamp to (0, 1]
        if not (0.0 < val <= 1.0):
            val = float(p_high)
        effective_p_high[c] = val

    # For reporting/plotting
    used_p_high = np.zeros(C, dtype=np.float64)

    for i, c in enumerate(channels):
        edges = glob_edges[c]
        counts = glob_counts[c]
        lo = _quantile_from_hist(counts, edges, p_low)
        p_hi_c = effective_p_high[c]
        hi = _quantile_from_hist(counts, edges, p_hi_c)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            logger.warning(f"[{round_name}] channel {c}: degenerate quantiles; falling back to edges.")
            lo, hi = float(edges[0]), float(edges[-1])
        m_glob[i] = lo
        q_lo[i] = lo
        q_hi[i] = hi
        used_p_high[i] = p_hi_c
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
        "p_high": float(p_high),  # scalar default requested
        "p_high_by_channel": {int(k): float(v) for k, v in (p_high_by_channel or {}).items()},
        "p_high_used": [float(v) for v in used_p_high],
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
        # Per-channel high label (since it may vary)
        p_high_pct_list = [
            ("%.5f" % (used_p_high[i] * 100)).rstrip("0").rstrip(".") if used_p_high.size > i else ""
            for i in range(C)
        ]

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
                ax.axvline(q_lo[i], color="tab:red", linestyle="--", linewidth=1.2, label=f"low {p_low_pct}%")
            if q_hi[i] > 0:
                ax.axvline(
                    q_hi[i],
                    color="tab:green",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"high {p_high_pct_list[i]}%" if p_high_pct_list[i] else "high",
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


def _iter_float32_tiles(
    workspace: Path,
    round_name: str,
    rois: Iterable[str] | None = None,
) -> Iterable[tuple[str, Path]]:
    """Yield (roi_name, float32_path) pairs for the requested round."""

    base = workspace / "analysis" / "deconv32"
    if not base.exists():
        return

    roi_filter = {r for r in rois} if rois else None
    pattern = f"{round_name}--*"
    for roi_dir in sorted(p for p in base.glob(pattern) if p.is_dir()):
        roi_name = roi_dir.name.split("--", 1)[-1]
        if roi_filter is not None and roi_name not in roi_filter:
            continue
        for tile in sorted(roi_dir.glob(f"{round_name}-*.tif")):
            if tile.suffix.lower() != ".tif":
                continue
            yield roi_name, tile


def quantize(
    workspace: Path,
    round_name: str,
    *,
    rois: tuple[str, ...] = (),
    n_fids: int = 2,
    overwrite: bool = False,
) -> None:
    """Quantize float32 deconvolved tiles in analysis/deconv32 to uint16 deliverables."""

    if n_fids <= 0:
        raise click.BadParameter("--n-fids must be positive.")

    workspace = workspace.resolve()
    logger.info(f"Workspace: {workspace}")

    scaling_path = workspace / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.txt"
    if not scaling_path.exists():
        raise click.ClickException(
            f"Scaling file not found at {scaling_path}. Run precompute_global_quantization first."
        )

    scaling = np.loadtxt(scaling_path, dtype=np.float32).reshape(2, -1)
    if scaling.shape[1] == 0:
        raise click.ClickException(f"Scaling file {scaling_path} is empty or malformed.")

    m_glob = scaling[0]
    s_glob = scaling[1]

    float32_tiles = list(_iter_float32_tiles(workspace, round_name, rois or None))
    if not float32_tiles:
        raise click.ClickException(f"No float32 tiles found for round '{round_name}' in analysis/deconv32.")

    out_root = workspace / "analysis" / "deconv"
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with progress_bar(len(float32_tiles)) as advance:
        for roi_name, float32_path in float32_tiles:
            rel_dir = float32_path.parent.name
            output_dir = out_root / rel_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / float32_path.name
            if not overwrite and output_path.exists():
                skipped += 1
                logger.info(f"Skipping existing {output_path}")
                advance()
                continue

            raw_path = workspace / rel_dir / float32_path.name
            if not raw_path.exists():
                raise click.ClickException(f"Missing raw tile required for fiducials: {raw_path}")

            float32_stack = tifffile.imread(float32_path)
            if float32_stack.ndim != 3:
                raise click.ClickException(
                    f"Expected 3D array in {float32_path}, got shape {float32_stack.shape}."
                )

            channels = m_glob.size
            if float32_stack.shape[0] % channels != 0:
                raise click.ClickException(
                    f"Tile {float32_path} has {float32_stack.shape[0]} planes, which is not divisible by {channels} channels."
                )

            z_slices = float32_stack.shape[0] // channels
            gpu_stack = cp.asarray(float32_stack, dtype=cp.float32)
            reshaped = gpu_stack.reshape(z_slices, channels, float32_stack.shape[1], float32_stack.shape[2])

            quantized = quantize_global(
                reshaped,
                m_glob,
                s_glob,
                i_max=I_MAX,
                return_stats=False,
                as_numpy=True,
            )

            with tifffile.TiffFile(raw_path) as tif:
                try:
                    metadata = tif.shaped_metadata[0]  # type: ignore[index]
                except (TypeError, IndexError):
                    metadata = tif.imagej_metadata or {}
                raw_stack = tif.asarray()

            if raw_stack.ndim != 3:
                raise click.ClickException(f"Raw tile {raw_path} has unexpected shape {raw_stack.shape}.")

            if raw_stack.shape[0] < n_fids:
                raise click.ClickException(
                    f"Raw tile {raw_path} has only {raw_stack.shape[0]} planes; cannot extract {n_fids} fiducials."
                )

            fid = raw_stack[-n_fids:].astype(np.uint16, copy=False)

            deliverable = np.concatenate([quantized, fid], axis=0)

            if metadata is None:
                metadata_dict: Dict[str, Any] = {}
            elif isinstance(metadata, dict):
                metadata_dict = dict(metadata)
            else:
                metadata_dict = dict(metadata)  # type: ignore[arg-type]

            metadata_dict.update({
                "deconv_min": [float(x) for x in np.ravel(m_glob)],
                "deconv_scale": [float(x) for x in np.ravel(s_glob)],
                "prenormalized": True,
            })

            tifffile.imwrite(
                output_path,
                deliverable,
                compression=22610,
                compressionargs={"level": 0.75},
                metadata=metadata_dict,
            )

            processed += 1
            logger.info(f"Quantized {output_path.relative_to(workspace)} (roi={roi_name})")
            advance()

    logger.info(f"Quantization complete: {processed} written, {skipped} skipped (overwrite={overwrite}).")


def precompute(
    workspace: Path,
    round_name: str,
    *,
    bins: int = 8192,
    p_low: float = 0.001,
    p_high: float = 0.99999,
    gamma: float = 1.05,
    i_max: int = I_MAX,
) -> None:
    """Aggregate histogram CSVs to produce global quantization parameters."""

    if bins <= 0:
        raise click.BadParameter("--bins must be positive.")
    if not (0.0 <= p_low < p_high <= 1.0):
        raise click.BadParameter("Require 0 <= p_low < p_high <= 1.")
    if gamma <= 0:
        raise click.BadParameter("--gamma must be positive.")

    workspace = workspace.resolve()
    logger.info(f"Workspace: {workspace}")
    # Apply automatic p_high overrides (priority: per-round indices > by-name global)
    overrides = _p_high_overrides_for(round_name) or {}
    if P_HIGH_NAME_OVERRIDES_GLOBAL:
        by_name = _p_high_by_channel_from_names(workspace, round_name, P_HIGH_NAME_OVERRIDES_GLOBAL) or {}
        # Merge only for channels not already specified by per-round mapping
        for k, v in by_name.items():
            overrides.setdefault(k, v)
    if overrides:
        logger.info(
            f"[{round_name}] Applying p_high overrides for channels: "
            + ", ".join(f"{c}->{v:.6f}" for c, v in sorted(overrides.items()))
        )
    m_glob, s_glob = precompute_global_quantization(
        workspace,
        round_name,
        bins=bins,
        p_low=p_low,
        p_high=p_high,
        gamma=gamma,
        i_max=i_max,
        p_high_by_channel=overrides or None,
    )
    logger.info(
        f"Computed global quantization for {round_name}: m_glob shape={m_glob.shape}, s_glob shape={s_glob.shape}."
    )


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
    raise SystemExit("This module provides library functions; use 'preprocess deconv' CLI.")


def load_global_scaling(path: Path, round_name: str) -> tuple[np.ndarray, np.ndarray]:
    scale_path = path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.txt"
    if not scale_path.exists():
        raise FileNotFoundError(
            f"Missing global scaling at {scale_path}. Run 'multi_deconv prepare' to generate histogram scaling first."
        )
    arr = np.loadtxt(scale_path).astype(np.float32).reshape((2, -1))
    return arr[0], arr[1]
