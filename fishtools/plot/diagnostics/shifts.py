"""Registration shift diagnostics plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, colormaps
from matplotlib.figure import Figure
from pydantic import BaseModel, TypeAdapter

from fishtools.utils.plot import micron_tick_formatter

__all__ = [
    "Shift",
    "ShiftsAdapter",
    "RoundPanels",
    "infer_rounds",
    "grid_panels",
    "make_shifts_scatter_figure",
    "make_corr_vs_l2_figure",
    "make_corr_hist_figure",
    "build_shift_layout_table",
    "make_shifts_layout_figure",
]


class Shift(BaseModel):
    """Per-tile registration metadata for a single round."""

    shifts: tuple[float, float]
    corr: float
    residual: float


ShiftsAdapter = TypeAdapter(dict[str, Shift])


@dataclass(slots=True)
class RoundPanels:
    """Panel grid layout for multi-round figures."""

    ncols: int
    nrows: int
    rounds: list[str]


def infer_rounds(shifts_by_tile: Mapping[int, Mapping[str, Shift]]) -> list[str]:
    """Return sorted round identifiers present in the shifts mapping."""

    if not shifts_by_tile:
        return []
    first = next(iter(shifts_by_tile.values()))
    return sorted(first.keys())


def grid_panels(rounds: Iterable[str], ncols: int) -> RoundPanels:
    """Compute subplot layout for the given rounds."""

    rounds_list = list(rounds)
    if ncols <= 0:
        raise ValueError("ncols must be positive")
    nrows = int(np.ceil(len(rounds_list) / ncols)) if rounds_list else 1
    return RoundPanels(ncols=ncols, nrows=nrows, rounds=rounds_list)


def _base_figure(panels: RoundPanels) -> tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(
        ncols=panels.ncols,
        nrows=panels.nrows,
        figsize=(12, 3 * panels.nrows),
        dpi=200,
    )
    axs = np.atleast_1d(axs).ravel()
    return fig, axs


def make_shifts_scatter_figure(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    ncols: int = 4,
    corr_threshold: float = 0.8,
) -> Figure:
    """Plot per-round X/Y shift scatter panels coloured by correlation."""

    rounds = infer_rounds(shifts_by_tile)
    panels = grid_panels(rounds, ncols)

    fig, axs = _base_figure(panels)

    for ax, round_ in zip(axs, panels.rounds, strict=False):
        pts = np.array([v[round_].shifts for v in shifts_by_tile.values()])
        corrs = (
            np.array([v[round_].corr for v in shifts_by_tile.values()])
            if shifts_by_tile
            else np.array([])
        )
        lim = max(10.0, 1.25 * float(np.abs(pts).max())) if pts.size else 10.0

        if pts.size:
            ax.scatter(*pts.T, c=corrs, alpha=0.6, s=8, cmap="bwr_r")
            low = corrs < corr_threshold
            if np.any(low):
                ax.scatter(
                    pts[low, 0],
                    pts[low, 1],
                    s=60,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=1.0,
                )
                for (tile_id, rec), is_low in zip(shifts_by_tile.items(), low, strict=False):
                    if is_low:
                        sx, sy = rec[round_].shifts
                        ax.text(sx, sy, str(tile_id), fontsize=7, color="yellow")
        else:
            ax.scatter([], [], s=5)

        ax.axhline(0.0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        if pts.size:
            sigma_px = float(pts.std(axis=0).mean())
            ax.set_title(f"{round_} σ={sigma_px:.1f} px")
        else:
            ax.set_title(f"{round_} (no data)")
        ax.set_xlabel("Shift X (px)")
        ax.set_ylabel("Shift Y (px)")

    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)

    fig.tight_layout()
    return fig


def make_corr_vs_l2_figure(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    ncols: int = 4,
    corr_threshold: float = 0.8,
) -> Figure:
    """Plot correlation vs L2 distance from mean shift panels."""

    rounds = infer_rounds(shifts_by_tile)
    panels = grid_panels(rounds, ncols)

    fig, axs = _base_figure(panels)

    for ax, round_ in zip(axs, panels.rounds, strict=False):
        corrs = np.array([v[round_].corr for v in shifts_by_tile.values()])
        pts = np.array([v[round_].shifts for v in shifts_by_tile.values()])
        mean_shift = np.mean(pts, axis=0) if pts.size else np.array([0.0, 0.0])
        l2 = np.linalg.norm(pts - mean_shift, axis=1) if pts.size else np.array([])

        low = corrs < corr_threshold if corrs.size else np.array([])
        ax.scatter(corrs, l2, c=corrs, alpha=0.6, s=8, cmap="bwr_r", vmin=0.0, vmax=1.0)
        if low.size and np.any(low):
            ax.scatter(
                corrs[low],
                l2[low],
                s=60,
                facecolors="none",
                edgecolors="yellow",
                linewidths=1.0,
            )
            for is_low, tile_id, dist, corr in zip(low, shifts_by_tile.keys(), l2, corrs, strict=False):
                if is_low and corr < corr_threshold:
                    ax.text(float(corr) + 0.01, float(dist) + 0.1, str(tile_id), fontsize=6, color="yellow")
        ax.set_xlabel("Correlation")
        ax.set_ylabel("L2 distance from mean (px)")
        ax.set_title(f"{round_}")
        if corrs.size:
            ax.set_xlim(min(0.0, float(corrs.min()) - 0.1), 1.0)
        ax.set_ylim(0, max(2.0, float(l2.max()) + 1.0) if l2.size else 2.0)

    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)

    fig.tight_layout()
    return fig


def make_corr_hist_figure(
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    ncols: int = 4,
) -> Figure:
    """Plot correlation histograms for each round."""

    rounds = infer_rounds(shifts_by_tile)
    panels = grid_panels(rounds, ncols)

    fig, axs = _base_figure(panels)

    for ax, round_ in zip(axs, panels.rounds, strict=False):
        corrs = np.array([v[round_].corr for v in shifts_by_tile.values()])
        ax.hist(corrs, linewidth=0)
        ax.set_title(round_)
        ax.set_xlim(0, 1)

    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)

    fig.tight_layout()
    return fig


def _apply_micron_ticks(ax: plt.Axes, pixel_size_um: float | None) -> None:
    """Format axis ticks/labels using micron units when a pixel size is provided."""

    if pixel_size_um and pixel_size_um > 0:
        formatter = micron_tick_formatter(pixel_size_um)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")


def _compute_round_l2(shifts_by_tile: Mapping[int, Mapping[str, Shift]]) -> dict[str, dict[int, float]]:
    """Compute L2 distance from the per-round mean shift for each tile."""

    l2_lookup: dict[str, dict[int, float]] = {}
    rounds = infer_rounds(shifts_by_tile)
    for round_name in rounds:
        vectors: list[tuple[int, np.ndarray]] = []
        for tile_id, per_tile in shifts_by_tile.items():
            shift = per_tile.get(round_name)
            if shift is None:
                continue
            vectors.append((tile_id, np.asarray(shift.shifts, dtype=float)))
        if not vectors:
            continue
        stacked = np.vstack([vec for _tile_id, vec in vectors])
        mean_vec = stacked.mean(axis=0)
        distances = np.linalg.norm(stacked - mean_vec, axis=1)
        l2_lookup[round_name] = {
            tile_id: float(dist)
            for (tile_id, _), dist in zip(vectors, distances, strict=False)
        }
    return l2_lookup


def build_shift_layout_table(
    tile_positions: dict[int, tuple[float, float]],
    shifts_by_tile: Mapping[int, Mapping[str, Shift]],
    *,
    roi: str,
) -> list[dict[str, object]]:
    """Return records with per-round positions (x,y) after applying shifts.

    tile_positions: mapping of tile index -> (x_center, y_center) in pixels.
    """

    l2_lookup = _compute_round_l2(shifts_by_tile)
    records: list[dict[str, object]] = []

    for tile_id, per_round in shifts_by_tile.items():
        center = tile_positions.get(int(tile_id))
        if center is None:
            continue
        base_x, base_y = center
        for round_name, shift in per_round.items():
            sx, sy = float(shift.shifts[0]), float(shift.shifts[1])
            l2 = l2_lookup.get(round_name, {}).get(int(tile_id), float("nan"))
            records.append(
                {
                    "roi": roi,
                    "round": round_name,
                    "tile": int(tile_id),
                    "x": base_x + sx,
                    "y": base_y + sy,
                    "correlation": float(shift.corr),
                    "L2": l2,
                }
            )

    return records


def make_shifts_layout_figure(
    records: list[dict[str, object]],
    *,
    tile_size_px: float,
    pixel_size_um: float | None = 0.108,
    label_skip: int = 2,
    corr_threshold: float | None = None,
) -> Figure:
    """Create correlation and L2 scatter panels positioned by tile layout.

    This function expects precomputed records from `build_shift_layout_table`.
    """

    import pandas as pd

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No shift layout records to plot.")

    rounds = sorted(df["round"].unique())
    n_rounds = len(rounds)

    xmin = float(df["x"].min())
    xmax = float(df["x"].max())
    ymin = float(df["y"].min())
    ymax = float(df["y"].max())

    fig, axs = plt.subplots(nrows=2, ncols=n_rounds, figsize=(max(4.0, 3.0 * n_rounds), 6.0), dpi=200)
    axs = np.atleast_2d(axs)

    # Row 1: correlation (draw true tile squares, no borders)
    corr_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    corr_cmap = colormaps["RdYlGn"]
    corr_mappable = None
    for c, round_name in enumerate(rounds):
        sub = df[df["round"] == round_name]
        ax = axs[0, c]
        patches = []
        from matplotlib.patches import Rectangle

        for row in sub.itertuples(index=False):
            color = corr_cmap(corr_norm(float(row.correlation)))
            rect = Rectangle((float(row.x) - tile_size_px / 2.0, float(row.y) - tile_size_px / 2.0), tile_size_px, tile_size_px, facecolor=color, edgecolor="none")
            ax.add_patch(rect)
            patches.append(rect)
        # build a mappable for colorbar (no need to draw an artist)
        if corr_mappable is None:
            import matplotlib as mpl

            corr_mappable = mpl.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
        # No low-correlation outlines; color-only for clarity
        for row in sub.iloc[:: max(1, int(label_skip))][["tile", "x", "y"]].itertuples(index=False):
            ax.text(row.x, row.y, str(int(row.tile)), fontsize=4, ha="center", va="center", color="white")
        ax.set_title(f"{round_name} — corr")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        _apply_micron_ticks(ax, pixel_size_um)
        # Hide x-axis on first row; keep limits identical
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        # Hide y-axis labels for all but first column
        if c > 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
        # Remove all spines for the first row (no external borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if corr_mappable is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        right_ax = axs[0, -1]
        divider = make_axes_locatable(right_ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(corr_mappable, cax=cax, orientation="vertical", label="Correlation")

    # Row 2: L2 (draw tile squares colored by L2)
    l2_mappable = None
    # Fixed upper bound for colormap scaling per request
    vmax = 50.0
    l2_norm = colors.Normalize(vmin=0.0, vmax=vmax)
    l2_cmap = colormaps["inferno_r"]
    for c, round_name in enumerate(rounds):
        sub = df[df["round"] == round_name]
        ax = axs[1, c]
        from matplotlib.patches import Rectangle

        for row in sub.itertuples(index=False):
            color = l2_cmap(l2_norm(float(row.L2)))
            rect = Rectangle((float(row.x) - tile_size_px / 2.0, float(row.y) - tile_size_px / 2.0), tile_size_px, tile_size_px, facecolor=color, edgecolor="none")
            ax.add_patch(rect)
        if l2_mappable is None:
            import matplotlib as mpl

            l2_mappable = mpl.cm.ScalarMappable(norm=l2_norm, cmap=l2_cmap)
        for row in sub.iloc[:: max(1, int(label_skip))][["tile", "x", "y"]].itertuples(index=False):
            ax.text(row.x, row.y, str(int(row.tile)), fontsize=4, ha="center", va="center", color="white")
        ax.set_title(f"{round_name} — L2")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        _apply_micron_ticks(ax, pixel_size_um)
        # Hide y-axis labels for all but first column
        if c > 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")

    if l2_mappable is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        right_ax = axs[1, -1]
        divider = make_axes_locatable(right_ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(l2_mappable, cax=cax, orientation="vertical", label="L2 distance from mean (px)")

    # Bottom row keeps X-axis labels; first column retains Y-axis labels

    fig.tight_layout()
    return fig
