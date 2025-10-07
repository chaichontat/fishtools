"""Preprocess plotting helpers.

Tile/grid visualizations for acquisition layouts and sampling diagnostics.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from numpy.typing import NDArray


def plot_tile_sizes(
    xy: NDArray[np.floating | np.integer],
    sizes: NDArray[np.floating | np.integer],
    *,
    selected: Iterable[int] | NDArray[np.integer] | None = None,
    percentile_clip: float | None = 99.0,
    cmap: str = "viridis",
    ax: Axes | None = None,
    title: str | None = None,
    show_colorbar: bool = True,
    annotate: Literal["none", "selected", "all"] | bool = "selected",
    fontsize: int = 6,
    selected_color: str = "red",
) -> Axes:
    """Render a grid heatmap of tile file sizes and optionally highlight selections.

    Parameters
    ----------
    xy
        Array of shape (N, 2) with stage or grid coordinates: columns [x, y].
    sizes
        Array of shape (N,) giving file sizes (bytes) per tile index. Missing
        tiles can be encoded as 0.
    selected
        Optional iterable of tile indices (0-based row indices into ``xy``)
        to highlight with a red square overlay.
    percentile_clip
        Upper percentile for color scale clipping (e.g., 99 to reduce the
        impact of outliers). Set to ``None`` to disable.
    cmap
        Matplotlib colormap name.
    ax
        Optional Matplotlib axis. If None, a new figure/axis is created.
    title
        Optional title for the plot.
    show_colorbar
        If True, add a colorbar to the axis.

    Returns
    -------
    Axes
        The axis used for plotting.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be of shape (N, 2)")
    if sizes.shape[0] != xy.shape[0]:
        raise ValueError("sizes length must match the number of tiles in xy")

    x = np.asarray(xy[:, 0])
    y = np.asarray(xy[:, 1])
    ux = np.unique(x)
    uy = np.unique(y)

    # Build grid mapping: (row=y_rank, col=x_rank)
    x_rank = {v: i for i, v in enumerate(ux)}
    y_rank = {v: i for i, v in enumerate(uy)}
    grid = np.full((len(uy), len(ux)), np.nan, dtype=float)
    rr = np.fromiter((y_rank[v] for v in y), count=len(y), dtype=int)
    cc = np.fromiter((x_rank[v] for v in x), count=len(x), dtype=int)
    grid[rr, cc] = sizes.astype(float)

    if percentile_clip is not None and np.isfinite(grid).any():
        vmax = float(np.nanpercentile(grid, percentile_clip))
    else:
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    vmin = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)

    im = ax.imshow(
        grid,
        origin="lower",
        interpolation="none",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        aspect="equal",
    )

    # Normalize annotate flag
    if isinstance(annotate, bool):
        annotate_mode: Literal["none", "selected", "all"] = "selected" if annotate else "none"
    else:
        annotate_mode = annotate

    sel_set: set[int] = set()
    if selected is not None:
        sel = np.asarray(list(selected), dtype=int)
        sel_set = set(int(i) for i in sel.tolist())
        # Map indices to ranks and plot a hollow red square overlay
        sel_rr = rr[sel]
        sel_cc = cc[sel]
        ax.scatter(
            sel_cc, sel_rr, s=100, facecolors="none", edgecolors=selected_color, linewidths=0.8, marker="s"
        )

    # Optionally annotate indices
    if annotate_mode != "none":
        try:
            import matplotlib.patheffects as pe

            effects = [pe.withStroke(linewidth=1.2, foreground="white")]
        except Exception:
            effects = None

        for i in range(len(xy)):
            if annotate_mode == "selected" and i not in sel_set:
                continue
            ax.text(
                cc[i],
                rr[i],
                str(i),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=(selected_color if i in sel_set else "black"),
                path_effects=effects,
            )

    ax.set_xlabel("x (grid col)")
    ax.set_ylabel("y (grid row)")
    if title:
        ax.set_title(title)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="File size (bytes)")
    ax.set_xlim(-0.5, len(ux) - 0.5)
    ax.set_ylim(-0.5, len(uy) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


__all__ = ["plot_tile_sizes"]
