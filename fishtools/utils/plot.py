from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from loguru import logger
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.lib.index_tricks import IndexExpression

if TYPE_CHECKING:
    import anndata as ad
    import polars as pl


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
    # Ensure colorcet colormaps (e.g., 'cet_colorwheel') are registered with Matplotlib
    import cmocean  # noqa: F401
    import colorcet as cc  # noqa: F401

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
        dict(
            alpha=1,
            s=5,
            c=c if c is not None else θ,
            cmap=scatter_cmap,
            zorder=2,
            linewidth=0.2,
        )
        | kwargs
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
    axs.histy.hist(
        pcs[:, 1],
        bins=yedges,
        alpha=0.8,
        orientation="horizontal",
        edgecolor=None,
        linewidth=0,
    )

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
    **kwargs: Any,
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
        **kwargs,
    )
    ax.add_artist(scalebar)


def save_figure(
    fig: Figure,
    output_dir: Path,
    name: str,
    roi: str,
    codebook: str,
    *,
    dpi: int = 300,
    log_level: str = "DEBUG",
) -> Path:
    """Save a Matplotlib figure with a standardized filename and close it.

    Filename format: ``{name}--{roi}+{codebook}.png`` under ``output_dir``.

    Args:
        fig: Matplotlib Figure to save and close.
        output_dir: Destination directory (created if needed).
        name: Logical figure stem (e.g., "contours", "scree_final").
        roi: Region of interest.
        codebook: Codebook identifier.
        dpi: Output DPI. Default 300 for parity with other preprocess plots.
        log_level: One of {"DEBUG","INFO","WARNING"} for loguru level.

    Returns:
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (output_dir / f"{name}--{roi}+{codebook}.png").resolve()
    fig.savefig(filename.as_posix(), dpi=dpi, bbox_inches="tight")
    if log_level == "DEBUG":
        logger.debug(f"Saved {name}: {filename}")
    elif log_level == "INFO":
        logger.info(f"Saved {name}: {filename}")
    elif log_level == "WARNING":
        logger.warning(f"Saved {name}: {filename}")
    else:
        raise ValueError(f"Unsupported log level: {log_level}")
    plt.close(fig)
    return filename


def scatter_spots(
    ax: Axes,
    spots_df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    max_points: int | None = None,
    point_size: float = 0.1,
    alpha: float = 0.3,
    include_scale_bar: bool = False,
    scale_bar_length: float | None = None,
    scale_bar_label: str | None = None,
    title: str | None = None,
    empty_message: str = "No spots",
    facecolor: str | None = None,
) -> None:
    """Render a downsampled scatter of spots on the provided axis."""
    # Polars not required here at runtime; type checked via from __future__ annotations
    ax.set_aspect("equal")
    # ax.axis("off")

    total_spots = spots_df.height
    if total_spots == 0:
        ax.text(
            0.5,
            0.5,
            empty_message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=DARK_PANEL_STYLE["text.color"],
        )
        if title:
            ax.set_title(title, color=DARK_PANEL_STYLE["axes.titlecolor"])
        return

    stride = 1
    if max_points is not None and max_points > 0:
        stride = max(1, math.ceil(total_spots / max_points))

    x_data = spots_df[x_col].to_numpy()[::stride]
    y_data = spots_df[y_col].to_numpy()[::stride]
    scatter = ax.scatter(x_data, y_data, s=point_size, alpha=alpha)
    if facecolor is not None:
        scatter.set_facecolor(facecolor)

    if include_scale_bar:
        if scale_bar_length is None or scale_bar_label is None:
            raise ValueError(
                "scale_bar_length and scale_bar_label are required when include_scale_bar is True."
            )
        add_scale_bar(ax, scale_bar_length, scale_bar_label)

    if title:
        ax.set_title(title, color=DARK_PANEL_STYLE["axes.titlecolor"])


def place_labels_avoid_overlap(
    ax: Axes,
    xs: npt.ArrayLike,
    ys: npt.ArrayLike,
    labels: list[str],
    *,
    fontsize: int = 6,
    use_arrows: bool = True,
    offsets: list[tuple[float, float]] | None = None,
) -> list[Text]:
    """Annotate points with labels trying to avoid collisions.

    Tries a set of offset positions around each anchor, accepting the first that
    doesn't overlap previously placed labels (tested in display coordinates).
    Returns the list of Text/Annotation artists.
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if offsets is None:
        offsets = [(6, 6), (6, -6), (-6, 6), (-6, -6), (9, 0), (-9, 0), (0, 9), (0, -9)]

    placed_texts: list[Text] = []
    bboxes: list[Any] = []
    fig = ax.figure
    # One initial draw to initialize the renderer and font metrics. Re-drawing
    # inside the loop is very expensive and can make large plots appear stuck.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for x, y, label in zip(xs, ys, labels):
        accepted: Text | None = None
        accepted_bb: Any | None = None
        for dx, dy in offsets:
            ann = ax.annotate(
                label,
                (float(x), float(y)),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#888") if use_arrows and (dx or dy) else None,
            )
            # Use the cached renderer to compute extents without forcing a full draw
            bb = ann.get_window_extent(renderer=renderer).expanded(1.05, 1.2)
            if not any(bb.overlaps(prev) for prev in bboxes):
                accepted, accepted_bb = ann, bb
                break
            ann.remove()

        if accepted is None:
            dx, dy = offsets[0]
            accepted = ax.annotate(
                label,
                (float(x), float(y)),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#888") if use_arrows and (dx or dy) else None,
            )
            # No redraw; compute extent against cached renderer
            accepted_bb = accepted.get_window_extent(renderer=renderer)

        placed_texts.append(accepted)
        if accepted_bb is not None:
            bboxes.append(accepted_bb)

    return placed_texts


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
    r: np.ndarray | None = None,
    g: np.ndarray | None = None,
    b: np.ndarray | None = None,
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


def plot_embedding(
    adata: "ad.AnnData",
    figsize: tuple[float, float] | None = None,
    dpi: float = 200,
    **kwargs: Any,
):
    """
    Plot the embedding of an AnnData object.

    Parameters:
    - adata: AnnData object containing the embedding data.
    """
    # Lazy import to avoid importing scanpy at module import time
    try:
        import scanpy as sc  # type: ignore
    except Exception as e:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("plot_embedding requires 'scanpy'. Please install it to use this function.") from e

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
    # Local import to avoid importing polars when plotting utilities are imported
    import polars as pl  # type: ignore

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


def micron_tick_formatter(pixel_size_um: float = 0.108, *, round_: int = 1) -> FuncFormatter:
    """Return a formatter that renders pixel tick values in micrometers."""
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive to format tick labels in micrometers.")

    def _format(value: float, _pos: int) -> str:
        if not np.isfinite(value):
            raise ValueError("Non-finite value cannot be formatted.")
        micron_value = value * pixel_size_um
        if abs(micron_value) < 1e-9:
            return "0"

        return str(round(micron_value, round_))

    return FuncFormatter(_format)


def configure_micron_axes(
    ax: Axes,
    pixel_size_um: float,
    *,
    x_label: str = "X",
    y_label: str = "Y",
    round_: int = 1,
) -> None:
    """Format both axes in micrometers and set axis labels.

    Applies a micrometer tick formatter to both axes based on the provided
    ``pixel_size_um`` and sets labels to ``"{x_label} (µm)"`` and
    ``"{y_label} (µm)"``.

    Args:
        ax: Target Matplotlib axes.
        pixel_size_um: Physical pixel size in micrometers. Must be positive.
        x_label: Base label for the x‑axis (default "X").
        y_label: Base label for the y‑axis (default "Y").
        round_: Number of decimal places for tick label rounding.

    Raises:
        ValueError: If ``pixel_size_um`` is not positive or if a non‑finite
            tick value is encountered by the formatter.
    """
    fmt = micron_tick_formatter(pixel_size_um, round_=round_)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(f"{x_label} (µm)")
    ax.set_ylabel(f"{y_label} (µm)")


def plot_img(
    path: Path | str,
    sl: slice | IndexExpression = slice(None, None, None),
    fig_kwargs: dict | None = None,
    **kwargs,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        img = tifffile.imread(path)[sl]

    elif path.suffix.lower() in {".zarr"}:
        import zarr

        img = zarr.open_array(path, mode="r")[sl]

    else:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    fig, ax = plt.subplots(**({"figsize": (10, 10)} | (fig_kwargs or {})))
    ax.imshow(img, zorder=1, **kwargs)
    return fig, ax


def tableau20_label_cmap(
    labels: np.ndarray,
    *,
    background_label: int | None = 0,
    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    seed: int | None = None,
    connectivity: int = 4,
    fill_interiors: bool = True,
    add_border: bool = False,
    border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    border_label: int = -1,
) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, dict[int, int]]:
    """Generate a Tableau-20-based colormap that avoids adjacent label clashes.

    The function builds a greedy graph coloring over the label adjacency graph.
    It keeps sampling colors from the Tableau 20 palette until it finds one that
    does not conflict with already-colored neighbors, ensuring visually distinct
    boundaries in the rendered segmentation.

    Parameters
    ----------
    labels
        Integer label image (2-D or 3-D) to analyze.
    background_label
        Label value that should map to the background color. Set to ``None`` to
        disable a dedicated background slot.
    background_color
        RGBA tuple to use for the background entry. Defaults to transparent.
    seed
        Seed forwarded to ``numpy.random.Generator`` for reproducible color
        assignments. Uses nondeterministic entropy when omitted.
    connectivity
        Neighborhood definition. ``4`` uses axial neighbors, ``8`` additionally
        considers diagonals. Only supported for 2-D inputs at the moment.
    fill_interiors
        When ``False`` the per-label colors are emitted fully transparent
        (alpha equals ``0``) so that only optional borders remain visible.

    Returns
    -------
    cmap, norm, label_to_index
        ``ListedColormap`` and matching ``BoundaryNorm`` suitable for
        ``matplotlib.pyplot.imshow`` along with the label → palette index
        mapping used to build a dense index image. When ``add_border`` is set,
        ``label_to_index`` also contains ``border_label`` so border pixels can
        be encoded consistently.
    """

    label_array = np.asarray(labels)
    if label_array.ndim < 2:
        msg = "`labels` must be at least 2-D to define adjacency."
        raise ValueError(msg)
    if connectivity not in {4, 8}:
        msg = "`connectivity` must be either 4 or 8."
        raise ValueError(msg)
    if connectivity == 8 and label_array.ndim != 2:
        msg = "8-connectivity is only implemented for 2-D label arrays."
        raise ValueError(msg)

    tableau_colors = np.array(colormaps["tab20"].colors)
    rng = np.random.default_rng(seed)

    unique_labels = np.unique(label_array)
    if background_label is not None:
        unique_labels = unique_labels[unique_labels != background_label]

    colorable_labels = [int(value) for value in unique_labels]
    adjacency: dict[int, set[int]] = {value: set() for value in colorable_labels}

    def _register_pairs(first: np.ndarray, second: np.ndarray) -> None:
        mask = first != second
        if not np.any(mask):
            return
        if background_label is not None:
            mask &= first != background_label
            mask &= second != background_label
        if not np.any(mask):
            return
        pairs = np.stack((first[mask], second[mask]), axis=-1)
        if pairs.size == 0:
            return
        unique_pairs = np.unique(pairs, axis=0)
        for left, right in unique_pairs:
            left_val = int(left)
            right_val = int(right)
            if left_val in adjacency:
                adjacency[left_val].add(right_val)
            if right_val in adjacency:
                adjacency.setdefault(right_val, set()).add(left_val)

    # Axial neighbors (N, S, E, W) for any dimensionality ≥ 2.
    for axis in range(label_array.ndim):
        slicer_head = [slice(None)] * label_array.ndim
        slicer_tail = [slice(None)] * label_array.ndim
        slicer_head[axis] = slice(0, -1)
        slicer_tail[axis] = slice(1, None)
        head = label_array[tuple(slicer_head)]
        tail = label_array[tuple(slicer_tail)]
        _register_pairs(head, tail)

    if connectivity == 8 and label_array.ndim == 2:
        diag_pairs = [
            (label_array[:-1, :-1], label_array[1:, 1:]),
            (label_array[1:, :-1], label_array[:-1, 1:]),
        ]
        for left, right in diag_pairs:
            _register_pairs(left, right)

    background_rgba = _normalize_rgba(background_color)
    border_rgba = _normalize_rgba(border_color)

    color_assignments: dict[int, tuple[float, float, float, float]] = {}
    color_rgb: dict[int, tuple[float, float, float]] = {}

    if colorable_labels:
        shuffled_labels = rng.permutation(colorable_labels)
        palette_indices = np.arange(tableau_colors.shape[0])

        for label_value in shuffled_labels:
            neighbor_colors = {
                color_rgb[neighbor] for neighbor in adjacency.get(label_value, set()) if neighbor in color_rgb
            }
            assigned_rgb: tuple[float, float, float] | None = None
            for palette_index in rng.permutation(palette_indices):
                candidate = tuple(float(value) for value in tableau_colors[palette_index])
                if candidate not in neighbor_colors:
                    assigned_rgb = candidate
                    break
            if assigned_rgb is None:
                base = tableau_colors[rng.integers(len(tableau_colors))][:3]
                perturbation = rng.normal(loc=0.0, scale=0.05, size=3)
                rgb = np.clip(base + perturbation, 0.0, 1.0)
                assigned_rgb = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
            assert assigned_rgb is not None
            alpha = 1.0 if fill_interiors else 0.0
            color_assignments[label_value] = (
                assigned_rgb[0],
                assigned_rgb[1],
                assigned_rgb[2],
                alpha,
            )
            color_rgb[label_value] = assigned_rgb

    ordered_labels = sorted(color_assignments)
    color_list = [background_rgba] if background_label is not None else []
    color_list.extend(color_assignments[label_value] for label_value in ordered_labels)

    if add_border:
        color_list.append(border_rgba)

    cmap = mcolors.ListedColormap(color_list)
    boundaries = np.arange(len(color_list) + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    label_to_index: dict[int, int] = {}
    if background_label is not None:
        label_to_index[int(background_label)] = 0
    for offset, label_value in enumerate(ordered_labels, start=len(label_to_index)):
        label_to_index[label_value] = offset

    if add_border:
        label_to_index[int(border_label)] = len(color_list) - 1

    return cmap, norm, label_to_index


def encode_labels_for_colormap(
    labels: np.ndarray,
    label_to_index: dict[int, int],
    *,
    background_label: int | None = 0,
    border_label: int = -1,
    connectivity: int = 4,
) -> np.ndarray:
    """Convert a label image into palette indices, optionally painting borders.

    Parameters
    ----------
    labels
        Source label array (2-D or higher) that matches the data used to build
        ``label_to_index``.
    label_to_index
        Mapping returned by :func:`tableau20_label_cmap`. When it contains
        ``border_label`` the function will encode thin borders around regions.
    background_label
        Value to treat as background; pixels are assigned its palette index.
    border_label
        Sentinel stored in ``label_to_index`` for the border color. Set to a
        value absent from ``label_to_index`` to disable border encoding.
    connectivity
        Neighbor definition (4 or 8) for the border extraction in 2-D.
    """

    label_array = np.asarray(labels)
    if label_array.ndim < 2:
        msg = "`labels` must be at least 2-D to define borders."
        raise ValueError(msg)
    if connectivity not in {4, 8}:
        msg = "`connectivity` must be either 4 or 8."
        raise ValueError(msg)
    if connectivity == 8 and label_array.ndim != 2:
        msg = "8-connectivity is only implemented for 2-D label arrays."
        raise ValueError(msg)

    index_image = np.empty(label_array.shape, dtype=np.int32)
    if background_label is None:
        try:
            default_index = next(iter(label_to_index.values()))
        except StopIteration as exc:
            msg = "`label_to_index` must contain at least one entry."
            raise ValueError(msg) from exc
    else:
        default_index = label_to_index[int(background_label)]
    index_image.fill(default_index)

    for label_value, palette_index in label_to_index.items():
        if label_value in {border_label, background_label}:
            continue
        index_image[label_array == label_value] = palette_index

    border_index = label_to_index.get(int(border_label))
    if border_index is None:
        return index_image

    border_mask = _compute_border_mask(
        label_array, background_label=background_label, connectivity=connectivity
    )
    index_image[border_mask] = border_index
    return index_image


def _normalize_rgba(color: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(color) == 4:
        return color[0], color[1], color[2], color[3]
    if len(color) == 3:
        return color[0], color[1], color[2], 1.0
    msg = "Color tuples must provide 3 (RGB) or 4 (RGBA) components."
    raise ValueError(msg)


def _compute_border_mask(
    labels: np.ndarray,
    *,
    background_label: int | None,
    connectivity: int,
) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)

    def _mark_border(first: np.ndarray, second: np.ndarray, slicer_first, slicer_second) -> None:
        different = first != second
        if background_label is not None:
            different &= (first != background_label) | (second != background_label)
        if not np.any(different):
            return
        mask[tuple(slicer_first)][different] = True
        mask[tuple(slicer_second)][different] = True

    for axis in range(labels.ndim):
        slicer_head = [slice(None)] * labels.ndim
        slicer_tail = [slice(None)] * labels.ndim
        slicer_head[axis] = slice(0, -1)
        slicer_tail[axis] = slice(1, None)
        _mark_border(
            labels[tuple(slicer_head)],
            labels[tuple(slicer_tail)],
            slicer_head,
            slicer_tail,
        )

    if connectivity == 8 and labels.ndim == 2:
        diag_configs = [
            ([slice(0, -1), slice(0, -1)], [slice(1, None), slice(1, None)]),
            ([slice(1, None), slice(0, -1)], [slice(0, -1), slice(1, None)]),
        ]
        for head_slices, tail_slices in diag_configs:
            _mark_border(
                labels[tuple(head_slices)],
                labels[tuple(tail_slices)],
                head_slices,
                tail_slices,
            )

    return mask
