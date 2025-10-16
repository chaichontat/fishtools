"""Reusable stitch layout plotting utilities.

These helpers generate Matplotlib figures showing the spatial layout defined
by ``TileConfiguration`` files. They encapsulate the sizing heuristics and
axis formatting used by the legacy ``check-stitch`` CLI, allowing other CLIs
or library consumers to reuse the same visuals without re-implementing the
plotting logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.plot import micron_tick_formatter

__all__ = [
    "StitchLayoutOptions",
    "make_combined_stitch_layout",
    "make_roi_stitch_layout",
]


@dataclass(frozen=True, slots=True)
class StitchLayoutOptions:
    """Configuration for stitch layout figures."""

    pixel_size_um: float | None = 0.108
    label_skip: int = 2
    tile_size_px: int = 1968
    inch_per_px: float = 0.0006


def _normalized_label_skip(label_skip: int) -> int:
    return max(1, label_skip)


def _compute_grid(n_items: int, ncols: int | None) -> tuple[int, int]:
    if n_items <= 0:
        return 1, 1
    if ncols is None or ncols <= 0:
        ncols = max(1, int(math.floor(math.sqrt(n_items))))
    nrows = int(math.ceil(n_items / ncols))
    return nrows, ncols


def _coverage_px(tc: TileConfiguration, tile_size_px: int) -> tuple[float, float]:
    xs = tc.df["x"].to_numpy()
    ys = tc.df["y"].to_numpy()
    if xs.size == 0 or ys.size == 0:
        return float(tile_size_px), float(tile_size_px)
    width = float(xs.max() - xs.min()) + float(tile_size_px)
    height = float(ys.max() - ys.min()) + float(tile_size_px)
    return width, height


def _format_axes(ax: Axes, pixel_size_um: float | None) -> None:
    ax.set_aspect("equal")
    if pixel_size_um and pixel_size_um > 0:
        formatter = micron_tick_formatter(pixel_size_um)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")


def _draw_tile_layout(ax: Axes, tc: TileConfiguration, options: StitchLayoutOptions) -> int:
    tc.plot(ax, show_labels=False)

    tile_size = float(options.tile_size_px)
    for x, y in zip(tc.df["x"].to_numpy(), tc.df["y"].to_numpy()):
        rect = Rectangle(
            (float(x), float(y)),
            tile_size,
            tile_size,
            facecolor="#4c78a8",
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(rect)

    label_skip = _normalized_label_skip(options.label_skip)
    df = tc.df[::label_skip]
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    labels = [str(int(i)) for i in df["index"].to_numpy()]
    for x, y, lab in zip(xs, ys, labels, strict=False):
        cx = float(x) + 0.5 * tile_size
        cy = float(y) + 0.5 * tile_size
        ax.text(cx, cy, lab, fontsize=6, ha="center", va="center", color="white")

    _format_axes(ax, options.pixel_size_um)
    return len(labels)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def make_combined_stitch_layout(
    ordered_rois: Sequence[str],
    tileconfigs: Mapping[str, TileConfiguration | None],
    *,
    ncols: int | None = None,
    options: StitchLayoutOptions | None = None,
    missing_message: str = "TileConfiguration not found",
) -> tuple[Figure, dict[str, Axes]]:
    """Create a multi-panel figure of stitch layouts for the given ROIs.

    Parameters
    ----------
    ordered_rois
        Sequence of ROI identifiers determining panel order.
    tileconfigs
        Mapping from ROI to ``TileConfiguration`` (or ``None`` if missing).
    ncols
        Optional number of grid columns (defaults to ``floor(sqrt(n_rois))``).
    options
        Plotting options (pixel size, label cadence, sizing heuristic).
    missing_message
        Text rendered when a ROI is missing a tile configuration.

    Returns
    -------
    Figure
        Matplotlib figure containing the combined layout.
    dict[str, Axes]
        Mapping of ROI to the axes used (only for ROIs with tile configuration).
    """
    opts = options or StitchLayoutOptions()
    nrows, ncols = _compute_grid(len(ordered_rois), ncols)

    coverages: list[tuple[float, float]] = []
    for roi in ordered_rois:
        tc = tileconfigs.get(roi)
        if tc is None:
            coverages.append((float(opts.tile_size_px), float(opts.tile_size_px)))
            continue
        coverages.append(_coverage_px(tc, opts.tile_size_px))

    col_widths: list[float] = [1.0] * ncols
    for idx, (width, _height) in enumerate(coverages):
        _row, col = divmod(idx, ncols)
        col_widths[col] = max(col_widths[col], float(width))
    global_max_height = max((h for _w, h in coverages), default=float(opts.tile_size_px))
    row_heights: list[float] = [float(global_max_height)] * nrows

    fig_width = _clamp(sum(col_widths) * opts.inch_per_px, 6.0, 18.0)
    fig_height = _clamp(sum(row_heights) * opts.inch_per_px, 4.0, 14.0)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        dpi=200,
        gridspec_kw={"width_ratios": col_widths, "height_ratios": row_heights},
    )
    if hasattr(axs, "flatten"):
        flat_axes = list(axs.flatten())
    else:
        flat_axes = [axs]
    axes_by_roi: dict[str, Axes] = {}

    for ax, roi in zip(flat_axes, ordered_rois, strict=False):
        tc = tileconfigs.get(roi)
        if tc is None:
            ax.axis("off")
            ax.text(0.5, 0.5, missing_message, ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{roi} (missing)")
            continue
        _draw_tile_layout(ax, tc, opts)
        ax.set_title(roi)
        axes_by_roi[roi] = ax

    for ax in flat_axes[len(ordered_rois) :]:
        fig.delaxes(ax)

    fig.tight_layout()
    return fig, axes_by_roi


def make_roi_stitch_layout(
    tc: TileConfiguration,
    roi: str,
    *,
    options: StitchLayoutOptions | None = None,
) -> Figure:
    """Create a single-ROI stitch layout figure."""

    opts = options or StitchLayoutOptions()
    width_px, height_px = _coverage_px(tc, opts.tile_size_px)
    width_in = _clamp(width_px * opts.inch_per_px, 3.5, 12.0)
    height_in = _clamp(height_px * opts.inch_per_px, 3.0, 10.0)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
    _draw_tile_layout(ax, tc, opts)
    ax.set_title(roi)
    fig.tight_layout()
    return fig
