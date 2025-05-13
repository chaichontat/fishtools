from typing import Any, NamedTuple

import cmocean  # colormap, do not remove
import colorcet as cc  # colormap, do not remove
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def plot_wheel(
    pcs: npt.NDArray[np.float32],
    cmap: str = "cet_colorwheel",
    c: npt.NDArray[np.float32] | None = None,
    scatter_cmap: str = "cet_colorwheel",
    fig: Figure | None = None,
    colorize_background: bool = True,
    colorbar_label: str | None = None,
    **kwargs: Any,
) -> tuple[Figure, NamedTuple]:
    θ = np.arctan2(pcs[:, 0], pcs[:, 1])
    fig = fig or plt.figure(1, figsize=(6, 6))
    axs = NamedTuple("axs", scatter=Axes, histx=Axes, histy=Axes)(
        scatter=plt.subplot2grid((6, 6), (1, 0), colspan=5, rowspan=5),
        histx=plt.subplot2grid((6, 6), (0, 0), colspan=5),
        histy=plt.subplot2grid((6, 6), (1, 5), rowspan=5),
    )

    H, xedges, yedges = np.histogram2d(pcs[:, 0], pcs[:, 1], bins=(200, 200))
    axs.scatter.axhline(0, color="black", alpha=0.25, markeredgewidth=0, linewidth=0.75)
    axs.scatter.axvline(0, color="black", alpha=0.25, markeredgewidth=0, linewidth=0.75)
    scatter_kw = (
        dict(alpha=1, s=5, c=c if c is not None else θ, cmap=scatter_cmap, zorder=2, linewidth=0.2) | kwargs
    )
    sc = axs.scatter.scatter(pcs[:, 0], pcs[:, 1], **scatter_kw)
    sc.set_edgecolor((0.4, 0.4, 0.4, 0.5))
    axs.scatter.set(xlabel="PC1", ylabel="PC2")

    # Background color wheel
    if colorize_background:
        xlim, ylim = axs.scatter.get_xlim(), axs.scatter.get_ylim()
        re, im = np.mgrid[xlim[0] : xlim[1] : 100j, ylim[0] : ylim[1] : 100j]
        n = re + 1j * im
        angle = np.arctan2(n.real, n.imag)
        axs.scatter.pcolormesh(re, im, angle, cmap=cmap, alpha=0.5)

    axs.histx.hist(pcs[:, 0], bins=xedges, alpha=0.8, edgecolor=None, linewidth=0)
    axs.histy.hist(pcs[:, 1], bins=yedges, alpha=0.8, orientation="horizontal", edgecolor=None, linewidth=0)

    nullfmt = mpl.ticker.NullFormatter()
    axs.histx.xaxis.set_major_formatter(nullfmt)
    axs.histy.yaxis.set_major_formatter(nullfmt)

    if colorbar_label is not None:
        divider = make_axes_locatable(axs.histy)
        # Append a new axes to the right of axs.histy for the colorbar
        # Adjust "size" (percentage of histy's width) and "pad" (in inches) as needed
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(sc, cax=cax, label=colorbar_label)

    fig.tight_layout()
    return fig, axs


def add_scalebar(
    ax: Axes,
    length: float,
    label: str,
    location: str = "lower right",
    pad: float = 0.5,
    borderpad: float = 0.5,
    sep: int = 5,
    bar_thickness: int = 2,
    font_size: int | None = None,
    color: str = "black",
    label_top: bool = False,
) -> Axes:
    """
    Adds a scale bar to a matplotlib Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes to add the scale bar to.
    - length: float
        The length of the scale bar in data units (typically x-axis units).
    - label: str
        The label for the scale bar (e.g., "100 µm").
    - location: str
        Location of the scale bar. Valid locations are:
        'upper right', 'upper left', 'lower left', 'lower right', 'right',
        'center left', 'center right', 'lower center', 'upper center', 'center'.
    - pad: float
        Padding around the scale bar, in fraction of font size.
    - borderpad: float
        Border padding, in fraction of font size (used if frameon=True).
    - sep: int
        Separation between bar and label in points.
    - bar_thickness: float
        Thickness (height) of the scale bar line.
    - font_size: int, optional
        Font size for the label. If None, Matplotlib's default is used.
    - color: str
        Color for the scale bar and text.
    - label_top: bool
        If True, the label is placed above the bar; otherwise, below.
    """
    fontprops = fm.FontProperties(size=font_size) if font_size else None

    scalebar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        location,
        pad=pad,
        borderpad=borderpad,
        sep=sep,
        frameon=False,
        size_vertical=bar_thickness,
        color=color,
        fontproperties=fontprops,
        label_top=label_top,
    )
    ax.add_artist(scalebar)
    return ax


def _generate_hsv_colors(n_colors: int, start_idx: int = 0):
    """Generate colors using HSV color space"""
    golden_ratio = (1 + 5**0.5) / 2
    hues = np.arange(n_colors) * golden_ratio % 1
    saturations = np.linspace(0.4, 1, n_colors)
    values = np.linspace(0.4, 1, n_colors)

    colors = np.zeros((n_colors, 3))
    for i in range(n_colors):
        h, s, v = hues[i], saturations[i], values[i]
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        # Convert HSV to RGB
        if h < 1 / 6:
            rgb = [c + m, x + m, m]
        elif h < 2 / 6:
            rgb = [x + m, c + m, m]
        elif h < 3 / 6:
            rgb = [m, c + m, x + m]
        elif h < 4 / 6:
            rgb = [m, x + m, c + m]
        elif h < 5 / 6:
            rgb = [x + m, m, c + m]
        else:
            rgb = [c + m, m, x + m]

        colors[start_idx + i] = rgb

    return colors


def create_label_colormap(label_image: np.ndarray):
    unique_labels = np.unique(label_image)
    n_colors = len(unique_labels)

    colors = np.zeros((n_colors, 3))
    if 0 in unique_labels:
        remaining_labels = unique_labels[unique_labels != 0]
        colors[1:] = _generate_hsv_colors(len(remaining_labels))
    else:
        colors = _generate_hsv_colors(n_colors)

    # Create lookup table
    max_label = unique_labels.max()
    lookup = np.zeros(max_label + 1, dtype=int)
    for i, label in enumerate(unique_labels):
        lookup[label] = i
    normalized_image = lookup[label_image]
    cmap = ListedColormap(colors)

    return normalized_image, cmap
