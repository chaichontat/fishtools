from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import cmocean  # colormap, do not remove
import colorcet as cc  # colormap, do not remove
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import scanpy as sc
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from fishtools.utils.utils import copy_signature

if TYPE_CHECKING:
    import anndata as ad


# Common plot styling primitives -------------------------------------------------

DARK_PANEL_STYLE = {
    "figure.facecolor": "#000000",
    "savefig.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "axes.edgecolor": "#9ca3af",
    "axes.grid": False,
    "axes.labelcolor": "#e5e7eb",
    "text.color": "#e5e7eb",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#e5e7eb",
    "ytick.color": "#e5e7eb",
}

SI_PREFIXES: list[tuple[float, str]] = [
    (1e24, "Y"),  # yotta
    (1e21, "Z"),  # zetta
    (1e18, "E"),  # exa
    (1e15, "P"),  # peta
    (1e12, "T"),  # tera
    (1e9, "G"),  # giga
    (1e6, "M"),  # mega
    (1e3, "k"),  # kilo
]


def format_si(x: float) -> str:
    """Format a number using an SI prefix without spaces or trailing .0.

    Examples: 950 -> "950", 1500 -> "1.5k", 1000 -> "1k", 1_000_000 -> "1M".
    """
    if not np.isfinite(x):
        return ""
    sign = "-" if x < 0 else ""
    ax = abs(x)
    factor, prefix = 1.0, ""
    for f, p in SI_PREFIXES:
        if ax >= f:
            factor, prefix = f, p
            break
    scaled = ax / factor
    if np.isclose(scaled, round(scaled)):
        num = f"{int(round(scaled))}"
    else:
        num = f"{scaled:.1f}".rstrip("0").rstrip(".")
    return f"{sign}{num}{prefix}"


def si_tick_formatter() -> FuncFormatter:
    """Return a Matplotlib `FuncFormatter` that applies `format_si` to tick values."""
    return FuncFormatter(lambda x, _pos: format_si(x))


def configure_dark_axes(
    ax: Axes,
    *,
    spine_color: str = "#9ca3af",
    tick_color: str = "#e5e7eb",
    enable_grid: bool = False,
) -> None:
    """Apply a consistent dark theme to an axes' spines, ticks, and grid."""

    ax.grid(enable_grid)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(spine_color)
    ax.tick_params(bottom=True, left=True, top=False, right=False, colors=tick_color)


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


def plot_with_hist(
    pcs: npt.NDArray[np.float32],
    fig: Figure | None = None,
    **kwargs: Any,
) -> tuple[Figure, NamedTuple]:
    axs = NamedTuple("axs", scatter=Axes, histx=Axes)(
        scatter=plt.subplot2grid((6, 6), (1, 0), colspan=5, rowspan=5),
        histx=plt.subplot2grid((6, 6), (0, 0), colspan=5),
    )
    fig = fig or plt.figure(1, figsize=(6, 6))
    H, xedges, yedges = np.histogram2d(pcs[:, 0], pcs[:, 1], bins=(200, 200))

    scatter_kw = kwargs
    sc = axs.scatter.scatter(pcs[:, 0], pcs[:, 1], **scatter_kw)
    sc.set_edgecolor((0.4, 0.4, 0.4, 0.5))

    axs.histx.hist(pcs[:, 0], bins=xedges, alpha=0.8, edgecolor=None, linewidth=0)

    nullfmt = mpl.ticker.NullFormatter()
    axs.histx.xaxis.set_major_formatter(nullfmt)

    fig.tight_layout()
    return fig, axs


def add_scale_bar(
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


def make_rgb(
    r: np.ndarray | None = None, g: np.ndarray | None = None, b: np.ndarray | None = None
) -> np.ndarray:
    """
    Create an RGB image from three separate channels.
    If a channel is None, it will be treated as an array of zeros.

    Parameters:
    - r: Red channel (2D array) or None
    - g: Green channel (2D array) or None
    - b: Blue channel (2D array) or None

    Returns:
    - RGB image (3D array with shape (height, width, 3))
    """
    channels = [r, g, b]
    provided_channels = [c for c in channels if c is not None]

    if not provided_channels:
        raise ValueError("At least one channel must be provided.")

    ref_shape = provided_channels[0].shape
    ref_dtype = provided_channels[0].dtype

    for c in provided_channels:
        if c.shape != ref_shape:
            raise ValueError("All provided input arrays must have the same shape.")
        if c.ndim != 2:
            raise ValueError("Input arrays must be 2D.")

    final_channels = []
    for c in channels:
        if c is None:
            final_channels.append(np.zeros(ref_shape, dtype=ref_dtype))
        else:
            final_channels.append(c)

    return np.dstack(final_channels)


def imshow_perc(ax: Axes, img: np.ndarray, percs: Sequence[float] | None = None, **kwargs: Any) -> None:
    """
    Display an image on a given Axes with percentiles for color scaling.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes to display the image on.
    - img: numpy.ndarray
        The image to display.
    - kwargs: Additional keyword arguments for imshow.
    """
    if percs is None:
        percs = [1, 99]
    vmin = np.percentile(img, percs[0])
    vmax = np.percentile(img, percs[1])

    ax.imshow(img, vmin=vmin, vmax=vmax, **kwargs)


def normalize(x: np.ndarray, percs: Sequence[float] = (1, 99)):
    """Normalize the image to the given percentiles."""
    vmin = np.percentile(x, percs[0])
    vmax = np.percentile(x, percs[1])
    return (x - vmin) / (vmax - vmin)


def add_legend(
    ax: plt.Axes,
    channel_names: Sequence[str | None],
    colors: Sequence[str] | None = None,
    loc: str = "best",
    **kwargs,
):
    """
    Adds a legend to an Axes object for an RGB image, with colored text.

    Parameters:
    - ax: The Matplotlib Axes to add the legend to.
    - channel_names: A sequence of names for the R, G, B channels.
                     If a name is None, that channel is skipped.
                     Example: ["Red Channel", "Green Channel", "Blue Channel"]
    - colors: A sequence of colors corresponding to R, G, B.
              Defaults to ["magenta", "lime", "blue"].
    - loc: Location of the legend (e.g., "upper right").
    - kwargs: Additional keyword arguments to pass to ax.legend().
    """
    if colors is None:
        colors = ["red", "lime", "blue"]

    active_labels = []
    active_colors = []
    dummy_handles = []

    for name, color_str in zip(channel_names, colors):
        if name is not None:
            active_labels.append(name)
            active_colors.append(color_str)
            # Create a dummy handle that won't be visible
            dummy_handles.append(Line2D([0], [0], marker="", color="none", linestyle="None"))

    if active_labels:
        legend = ax.legend(
            dummy_handles,
            active_labels,
            loc=loc,
            handlelength=0,
            handletextpad=0,
            frameon=False,
            **kwargs,
        )
        shift = max([t.get_window_extent().width for t in legend.get_texts()])

        for text, color_str in zip(legend.get_texts(), active_colors):
            text.set_color(color_str)
            text.set_fontweight("bold")
            if "right" in loc:
                text.set_horizontalalignment("right")
                text.set_position((shift - text.get_window_extent().width, 0))
    return ax


@copy_signature(sc.pl.embedding)
def plot_embedding(
    adata: "ad.AnnData", figsize: tuple[float, float] | None = None, dpi: float = 200, **kwargs: Any
):
    """
    Plot the embedding of an AnnData object.

    Parameters:
    - adata: AnnData object containing the embedding data.
    """

    kwargs = kwargs | {"return_fig": True}
    fig = cast(Figure, sc.pl.embedding(adata, **kwargs))

    if figsize:
        fig.set_size_inches(*figsize)
    if dpi:
        fig.set_dpi(dpi)

    axes: list[plt.Axes] = []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        if not ax.has_data():
            fig.delaxes(ax)
            continue

        ax.set_aspect("equal")
        axes.append(ax)

    return fig, axes


def plot_all_genes(
    spots: pl.DataFrame,
    *,
    dark: bool = False,
    only_blank: bool = False,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = None,
):
    genes = spots.group_by("target").len().sort("len", descending=True)["target"]
    genes = spots["target"].unique()
    genes = filter(lambda x: x.startswith("Blank"), genes) if only_blank else genes
    genes = sorted(genes)

    nrows = int(np.ceil(len(genes) / 20))
    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }):
        default_size = (72, int(2.4 * nrows))
        fig, axs = plt.subplots(
            ncols=20,
            nrows=nrows,
            figsize=figsize or default_size,
            dpi=160,
            facecolor="black" if dark else "white",
        )
    axs = axs.flatten()
    for ax, gene in zip(axs, genes):
        selected = spots.filter(pl.col("target") == gene)
        if not len(selected):
            logger.warning(f"No spots found for {gene}")
            continue
        ax.set_aspect("equal")
        ax.set_title(gene, fontsize=16, color="white" if dark else "black")
        ax.axis("off")
        use_hexbin = dark or (cmap is not None)
        if use_hexbin:
            ax.hexbin(selected["x"], selected["y"], gridsize=250, cmap=cmap or "magma")
        else:
            ax.scatter(selected["x"], selected["y"], s=2000 / len(selected), alpha=0.1)

    for ax in axs.flat:
        if not ax.has_data():
            fig.delaxes(ax)
    return fig, axs
