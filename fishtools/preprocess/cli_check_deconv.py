from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.utils.logging import setup_cli_logging

sns.set_theme()


@dataclass(frozen=True, slots=True)
class DeconvScaling:
    round_name: str
    channels_indexed: list[int]
    m_glob: np.ndarray  # shape (C,)
    s_glob: np.ndarray  # shape (C,)
    p_high_used: list[float] | None = None
    p_high_default: float | None = None
    p_high_by_channel: dict[int, float] | None = None


def _load_scaling(ws: Workspace, round_name: str) -> DeconvScaling:
    """Load deconvolution scaling arrays (txt) and paired JSON metadata.

    The txt contains a 2xC array: first row `m_glob` (low quantile), second row
    `s_glob` (scale). The JSON provides metadata including the channel indices
    and any per-channel percentile overrides used to compute the global scale.
    """
    txt_path = ws.deconv_scaling(round_name)
    json_path = txt_path.with_suffix(".json")

    if not txt_path.exists():
        raise click.ClickException(
            f"Scaling file not found: {txt_path}. Run 'preprocess deconvnew precompute' or equivalent first."
        )
    try:
        arr = np.loadtxt(txt_path).astype(np.float64).reshape(2, -1)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"Failed to read scaling txt at {txt_path}: {e}") from e

    m_glob = arr[0]
    s_glob = arr[1]

    channels_indexed: list[int]
    p_high_used: list[float] | None = None
    p_high_default: float | None = None
    p_high_by_channel: dict[int, float] | None = None

    if json_path.exists():
        try:
            import json as _json

            meta = _json.loads(json_path.read_text())
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(f"Failed to parse JSON sidecar at {json_path}: {e}") from e
        # Extract with defensive defaults
        channels_indexed = [int(x) for x in meta.get("channels_indexed", list(range(m_glob.size)))]
        p_high_used = [float(x) for x in meta.get("p_high_used", [])] or None
        p_high_default = float(meta.get("p_high", 0.0)) or None
        raw_over = meta.get("p_high_by_channel", {}) or {}
        try:
            p_high_by_channel = {int(k): float(v) for k, v in dict(raw_over).items()} if raw_over else None
        except Exception:  # pragma: no cover - best-effort parsing
            p_high_by_channel = None
    else:
        logger.warning(
            f"JSON sidecar not found next to scaling txt. Proceeding without metadata: {json_path}"
        )
        channels_indexed = list(range(m_glob.size))

    if len(channels_indexed) != m_glob.size:
        raise click.ClickException(
            "JSON channels_indexed length does not match scaling array length: "
            f"{len(channels_indexed)} vs {m_glob.size}."
        )

    return DeconvScaling(
        round_name=round_name,
        channels_indexed=channels_indexed,
        m_glob=m_glob,
        s_glob=s_glob,
        p_high_used=p_high_used,
        p_high_default=p_high_default,
        p_high_by_channel=p_high_by_channel,
    )


def build_scaling_table(
    scaling: DeconvScaling,
    *,
    channel_names: list[str] | None,
) -> pd.DataFrame:
    """Return a tidy table for plotting/export with one row per channel.

    Columns: [channel, name, m_glob, s_glob, p_high_used, override]
    """
    rows: list[dict[str, Any]] = []
    C = int(scaling.m_glob.size)
    used = scaling.p_high_used or [np.nan] * C
    overrides = scaling.p_high_by_channel or {}
    for i in range(C):
        ch = int(scaling.channels_indexed[i])
        nm = None
        if channel_names and ch < len(channel_names):
            nm = str(channel_names[ch])
        rows.append({
            "channel": ch,
            "name": nm or f"ch{ch}",
            "m_glob": float(scaling.m_glob[i]),
            "s_glob": float(scaling.s_glob[i]),
            "p_high_used": float(used[i]) if i < len(used) else np.nan,
            "override": bool(ch in overrides),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("channel").reset_index(drop=True)


def _make_scaling_figure(df: pd.DataFrame, round_name: str, *, annotate: bool = False) -> plt.Figure:
    """Produce a compact 2-row panel: m_glob and s_glob per channel, plus scatter."""
    C = df.shape[0]
    ncols = 2
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 4.0 * 2), squeeze=False)
    ax_m, ax_s = axes[0, 0], axes[0, 1]
    ax_scatter = axes[1, 0]
    ax_dummy = axes[1, 1]
    ax_dummy.axis("off")

    order = df["channel"].to_numpy().argsort()
    plot_df = df.iloc[order].reset_index(drop=True)
    x_labels = [f"{row['channel']}\n{row['name']}" for _, row in plot_df.iterrows()]

    sns.barplot(x=np.arange(C), y=plot_df["m_glob"], ax=ax_m, color="#3b82f6", edgecolor="#1e3a8a")
    ax_m.set_title("Global Low per Channel (m_glob)")
    ax_m.set_xlabel("channel (idx\nname)")
    ax_m.set_ylabel("intensity")
    ax_m.set_xticks(np.arange(C), x_labels, rotation=0)

    sns.barplot(x=np.arange(C), y=plot_df["s_glob"], ax=ax_s, color="#10b981", edgecolor="#064e3b")
    ax_s.set_title("Global Scale per Channel (s_glob)")
    ax_s.set_xlabel("channel (idx\nname)")
    ax_s.set_ylabel("scale")
    ax_s.set_xticks(np.arange(C), x_labels, rotation=0)

    # Scatter of s_glob vs m_glob, highlight overrides
    colors = np.where(plot_df["override"].to_numpy(), "#dc2626", "#6b7280")
    ax_scatter.scatter(plot_df["m_glob"], plot_df["s_glob"], c=colors, s=24, alpha=0.9)
    for _, row in plot_df.iterrows():
        ax_scatter.annotate(
            f"ch{int(row['channel'])}",
            (float(row["m_glob"]), float(row["s_glob"])),
            fontsize=8,
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax_scatter.set_xlabel("m_glob")
    ax_scatter.set_ylabel("s_glob")
    ax_scatter.set_title("Scale vs Low (override channels in red)")

    if annotate and "p_high_used" in plot_df:
        # Add a small table-like text showing p_high_used per channel
        vals = [f"{v * 100:.3f}%" if np.isfinite(v) else "" for v in plot_df["p_high_used"].to_numpy()]
        text = "p_high_used: " + ", ".join(
            f"ch{int(c)}={v}" for c, v in zip(plot_df["channel"], vals, strict=False)
        )
        fig.text(0.01, 0.01, text, fontsize=8)

    fig.suptitle(f"Deconvolution Global Scaling — round={round_name}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


@click.command("check-deconv")
@click.argument(
    "path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument("round_name", type=str)
@click.option(
    "--roi",
    "rois",
    multiple=True,
    help="ROI(s) to include. Default: all detected.",
)
@click.option(
    "--label-skip",
    type=int,
    default=2,
    show_default=True,
    help="Label every Nth tile index in the layout figure.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path),
    default=None,
    help="Output directory for PNG/CSV [default: '<workspace_parent>/output']",
)
@click.option("--annotate", is_flag=True, help="Annotate figure with p_high_used values per channel.")
def check_deconv(
    path: Path,
    round_name: str,
    rois: tuple[str, ...],
    label_skip: int,
    output_dir: Path | None,
    annotate: bool,
) -> None:
    """Plot deconvolution global scaling (m_glob, s_glob) per channel.

    Reads the paired JSON for metadata and the TXT scaling matrix saved under
    analysis/deconv_scaling/<round>.{json,txt}.
    """

    setup_cli_logging(
        path,
        component="preprocess.check_deconv",
        file=f"check-deconv-{round_name}",
        extra={"round": round_name},
    )

    ws = Workspace(path)
    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")

    # Per-ROI grid: rows = [scale, min], columns = channels
    _emit_per_roi_channel_grid(
        ws,
        round_name,
        rois=ws.resolve_rois(rois) if rois else ws.resolve_rois(None),
        label_skip=int(label_skip),
        output_dir=output_dir,
    )


__all__ = [
    "DeconvScaling",
    "build_scaling_table",
    "check_deconv",
]


# --- Tile layout helpers -------------------------------------------------------


def _infer_tile_size_px(ws: Workspace, round_name: str, roi: str, default: int = 1968) -> int:
    try:
        tile = next((p for p in (ws.deconved / f"{round_name}--{roi}").glob(f"{round_name}-*.tif")), None)
        if tile is None:
            return int(default)
        from tifffile import TiffFile

        with TiffFile(tile) as tif:
            shape = tif.series[0].shape if tif.series else None  # type: ignore[assignment]
        if not shape:
            return int(default)
        if len(shape) == 2:
            y, x = shape
        elif len(shape) == 3:
            _, y, x = shape
        elif len(shape) == 4:
            _, _, y, x = shape
        else:
            return int(default)
        if y != x:
            logger.warning(f"Non-square tile detected ({x}x{y}); using X={x} as tile size.")
        return int(x)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed inferring tile size for ROI {roi}: {e}")
        return int(default)


def _collect_per_tile_values(
    ws: Workspace,
    round_name: str,
    roi: str,
) -> tuple[list[int], np.ndarray, np.ndarray, int]:
    """Return (tile_indices, scale_vals, min_vals, C) for a ROI.

    scale_vals/min_vals have shape (N_tiles, C). Missing data filled with NaN.
    """
    from json import loads as json_loads

    from fishtools.io.workspace import get_metadata

    base = ws.deconved / f"{round_name}--{roi}"
    entries: list[tuple[int, list[float] | None, list[float] | None]] = []
    for tif in sorted(base.glob(f"{round_name}-*.tif")):
        try:
            idx = int(tif.stem.rsplit("-", 1)[-1])
        except Exception:
            continue
        meta: dict[str, Any] | None = None
        sidecar = tif.with_suffix(".deconv.json")
        try:
            if sidecar.exists():
                meta = json_loads(sidecar.read_text())
            else:
                meta = get_metadata(tif)
        except Exception:
            meta = None
        if not meta:
            entries.append((idx, None, None))
            continue
        try:
            mins = meta.get("deconv_min")
            scales = meta.get("deconv_scale")
            mins_list = [float(x) for x in mins] if mins is not None else None
            scales_list = [float(x) for x in scales] if scales is not None else None
        except Exception:
            mins_list, scales_list = None, None
        entries.append((idx, scales_list, mins_list))

    if not entries:
        return [], np.empty((0, 0)), np.empty((0, 0)), 0

    # Determine channel count
    C = 0
    for _, s, m in entries:
        if s:
            C = max(C, len(s))
        if m:
            C = max(C, len(m))
    if C == 0:
        return [idx for idx, *_ in entries], np.empty((len(entries), 0)), np.empty((len(entries), 0)), 0

    tiles = [idx for idx, *_ in entries]
    scale_arr = np.full((len(entries), C), np.nan, dtype=float)
    min_arr = np.full((len(entries), C), np.nan, dtype=float)
    for r, (_idx, s, m) in enumerate(entries):
        if s:
            n = min(C, len(s))
            scale_arr[r, :n] = np.asarray(s[:n], dtype=float)
        if m:
            n = min(C, len(m))
            min_arr[r, :n] = np.asarray(m[:n], dtype=float)
    return tiles, scale_arr, min_arr, C


def _emit_per_roi_channel_grid(
    ws: Workspace,
    round_name: str,
    *,
    rois: list[str],
    label_skip: int,
    output_dir: Path,
) -> None:
    """Draw a simple grid per ROI: rows [scale, min], columns = channels.

    Coloring is by absolute values; per-row (stat) normalization is shared across
    all channels within the ROI to aid comparison.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    for roi in rois:
        try:
            tc = ws.tileconfig(roi)
        except FileNotFoundError:
            logger.warning(f"TileConfiguration not found for ROI {roi}; skipping.")
            continue

        tile_size_px = _infer_tile_size_px(ws, round_name, roi, default=1968)
        xs = tc.df["x"].to_numpy()
        ys = tc.df["y"].to_numpy()
        if xs.size == 0 or ys.size == 0:
            logger.warning(f"Empty TileConfiguration for ROI {roi}; skipping.")
            continue

        x0 = float(xs.min())
        y0 = float(ys.min())
        width = float(xs.max() - xs.min()) + float(tile_size_px)
        height = float(ys.max() - ys.min()) + float(tile_size_px)

        tiles, scale_arr, min_arr, C = _collect_per_tile_values(ws, round_name, roi)
        if C == 0:
            logger.warning(f"No deconvolution metadata found for ROI {roi}; skipping.")
            continue

        # Compute per-row normalization (shared across channels) using 1st/99th percentiles
        def _row_norm(arr: np.ndarray) -> Normalize:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return Normalize(vmin=0.0, vmax=1.0)
            try:
                vmin = float(np.percentile(finite, 1))
                vmax = float(np.percentile(finite, 99))
            except Exception:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                # Final fallback to a safe default range
                vmin, vmax = (0.0, 1.0)
            return Normalize(vmin=vmin, vmax=vmax)

        norm_scale = _row_norm(scale_arr)
        norm_min = _row_norm(min_arr)
        cm = get_cmap("viridis")

        # Figure sizing: each panel reuses same physical extent; scale grid area
        inch_per_px = 0.00035
        fig_w = min(max(width * inch_per_px * max(1, C / 3), 4.0), 16.0)
        fig_h = min(max(height * inch_per_px * 2.0, 4.0), 12.0)
        fig, axs = plt.subplots(nrows=2, ncols=C, figsize=(fig_w, fig_h), dpi=200)
        axs = np.atleast_2d(axs)

        # Prepare lookup for tile positions
        pos = {
            int(i): (float(x) - x0, float(y) - y0)
            for i, x, y in tc.df.select(["index", "x", "y"]).iter_rows()
        }

        for c in range(C):
            for r, (arr, norm, label) in enumerate((
                (scale_arr, norm_scale, "scale"),
                (min_arr, norm_min, "min"),
            )):
                ax = axs[r, c]
                ax.set_aspect("equal")
                colvals = arr[:, c] if c < arr.shape[1] else np.full((arr.shape[0],), np.nan)
                # Draw tiles
                for t_idx, val in zip(tiles, colvals, strict=False):
                    xy = pos.get(int(t_idx))
                    if xy is None:
                        continue
                    color = cm(norm(val)) if np.isfinite(val) else (0.3, 0.3, 0.3, 0.4)
                    rect = Rectangle(
                        xy, float(tile_size_px), float(tile_size_px), facecolor=color, edgecolor="none"
                    )
                    ax.add_patch(rect)
                # Labels
                if label_skip > 0:
                    for i, (idx, (x_rel, y_rel)) in enumerate(pos.items()):
                        if i % int(label_skip) != 0:
                            continue
                        cx = x_rel + 0.5 * float(tile_size_px)
                        cy = y_rel + 0.5 * float(tile_size_px)
                        ax.text(cx, cy, str(int(idx)), fontsize=6, ha="center", va="center", color="white")
                ax.set_xlim(0.0, width)
                ax.set_ylim(0.0, height)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(label)
                if r == 0:
                    ax.set_title(f"ch{c}")

        # Shared colorbars for each row
        for r, norm in enumerate((norm_scale, norm_min)):
            sm = ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            fig.colorbar(sm, ax=axs[r, :].tolist(), fraction=0.02, pad=0.01, location="right")

        fig.suptitle(f"{roi} — {round_name}: row0=scale, row1=min")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = output_dir / f"deconv_tiles_grid--{roi}+{round_name}.png"
        fig.savefig(out.as_posix(), dpi=200)
        plt.close(fig)
        logger.info(f"Saved per-ROI channel grid: {out}")
