"""Registration shift diagnostics plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, TypeAdapter

__all__ = [
    "Shift",
    "ShiftsAdapter",
    "RoundPanels",
    "infer_rounds",
    "grid_panels",
    "make_shifts_scatter_figure",
    "make_corr_vs_l2_figure",
    "make_corr_hist_figure",
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
