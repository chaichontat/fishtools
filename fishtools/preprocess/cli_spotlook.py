import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import questionary
import seaborn as sns
import typer
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rich.console import Console
from rich.text import Text
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from fishtools.io.codebook import Codebook
from fishtools.io.workspace import Workspace
from fishtools.preprocess.config import SpotThresholdParams
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.plot import (
    DARK_PANEL_STYLE,
    configure_dark_axes,
    configure_micron_axes,
    save_figure,
    scatter_spots,
    si_tick_formatter,
)

console = Console()


@dataclass(slots=True)
class DensitySurface:
    """Container for density grid data used across plotting stages."""

    x_coords: np.ndarray
    y_coords: np.ndarray
    z_raw: np.ndarray
    z_smooth: np.ndarray
    contour_levels: np.ndarray

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x_coords, self.y_coords)


@dataclass(slots=True)
class ThresholdCurve:
    """Threshold curve metadata used for interactive selection."""

    levels: list[int]
    spot_counts: list[int]
    blank_proportions: list[float]
    max_level: int


@dataclass(slots=True)
class ROIThresholdContext:
    """Cached per-ROI artifacts between the preparation and execution phases.

    Columns expected in ``spots`` and ``spots_final``
    - ``x``, ``y``: global mosaic pixel coordinates (preferred for spatial plots).
    - ``x_local``, ``y_local``: tile‑local coordinates (may be present from loaders).
    - ``area``: connected‑component area in pixels.
    - ``norm``: L2 norm of spot intensity vector across rounds×channels.
    - ``distance``: decode metric distance to the assigned target.
    - ``target``: decoded gene label; blanks typically start with ``"Blank"``.
    - ``x_``, ``y_``: feature‑space axes created by spotlook for thresholding;
      not spatial (see ``_apply_initial_filters`` for definitions).
    - Additional columns like ``bit0``, ``bit1`` may appear after codebook join.
    """

    spots: pl.DataFrame
    contours: QuadContourSet
    interpolator: RegularGridInterpolator
    curve: ThresholdCurve
    artifact_paths: dict[str, Path | None]
    spots_final: pl.DataFrame | None = None  # Filled after final filtering


def build_spotlook_params(
    config_path: Path | None = None,
    *,
    area_min: float | None = None,
    area_max: float | None = None,
    norm_threshold: float | None = None,
    min_norm: float | None = None,
    distance_threshold: float | None = None,
    seed: int | None = None,
    contour_mode: str | None = None,
    contour_levels: int | None = None,
) -> SpotThresholdParams:
    """Create SpotlookParams from project config (if provided) with CLI overrides."""
    params: SpotThresholdParams
    if config_path:
        from fishtools.preprocess.config_loader import load_config

        cfg = load_config(config_path)
        params = cfg.spot_threshold
    else:
        params = SpotThresholdParams()

    overrides = {
        k: v
        for k, v in {
            "area_min": area_min,
            "area_max": area_max,
            "norm_threshold": norm_threshold,
            "min_norm": min_norm,
            "distance_threshold": distance_threshold,
            "seed": seed,
            "contour_mode": contour_mode,
            "contour_levels": contour_levels,
        }.items()
        if v is not None
    }
    return params.model_copy(update=overrides) if overrides else params


# --- Core Helper Functions ---


def create_shimmer_text(text: str, style: str = "cyan") -> Text:
    """Creates a shimmering text effect using Rich styling."""
    shimmer_text = Text(text)
    shimmer_text.stylize(style)
    return shimmer_text


def count_by_gene(spots: pl.DataFrame) -> pl.DataFrame:
    """
    Count spots per target and attach plotting metadata.

    Input expectations
    - ``spots`` contains at least ``target`` and optionally codebook join columns
      (``bit0``, ``bit1``, ``bit2``).

    Output columns
    - ``target`` (Utf8): Target/gene.
    - ``count`` (UInt32): Number of spots for the target.
    - ``bit0``, ``bit1``, ``bit2`` (UInt8): First observed bit/channel ids for
      this target (copied from joined codebook; may be null if join not present).
    - ``is_blank`` (Boolean): True when target name starts with ``"Blank"``.
    - ``color`` (Utf8): Convenience color label (``"red"`` for blanks,
      ``"blue"`` otherwise) for scree plots.
    """
    return (
        spots.group_by("target")
        .agg([
            pl.len().alias("count"),
            pl.col("bit0").first(),
            pl.col("bit1").first(),
            pl.col("bit2").first(),
        ])
        .sort("count", descending=True)
        .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
        .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
    )


def _save_combined_spots_plot(
    contexts: dict[str, ROIThresholdContext],
    output_dir: Path,
    codebook: str,
    params: SpotThresholdParams,
) -> Path | None:
    """Create a grid of downsampled spot plots across ROIs."""
    if not contexts:
        raise ValueError("No ROIs available to plot.")

    n_rois = len(contexts)
    grid = int(math.ceil(math.sqrt(n_rois)))
    n_cols = max(1, min(grid, 12))
    n_rows = int(math.ceil(n_rois / n_cols))

    fig_width = 4 * n_cols
    fig_height = 4 * n_rows

    logger.debug(
        "CombSpots grid: rois=%d, rows=%d, cols=%d, size=(%.2f in, %.2f in) @ %d dpi (~%dx%d px)",
        n_rois,
        n_rows,
        n_cols,
        fig_width,
        fig_height,
        params.dpi,
        int(fig_width * params.dpi),
        int(fig_height * params.dpi),
    )

    max_inches = max(fig_width, fig_height)
    max_safe_dpi = int(65535 // max(1, max_inches))
    render_dpi = max(1, min(int(params.dpi), max_safe_dpi))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=render_dpi)
    axes = np.atleast_1d(axes).ravel()
    ordered_rois = sorted(contexts)

    scale_bar_length = params.scale_bar_um / params.pixel_size_um

    for idx, (ax, roi) in enumerate(zip(axes, ordered_rois)):
        ctx = contexts[roi]
        # Prefer final filtered spots; fall back to pre-threshold if needed (tests/mock contexts)
        spots = ctx.spots_final if ctx.spots_final is not None else ctx.spots
        title = f"Filtered spots for {roi} (n={spots.height:,})"
        # Use the same spatial orientation as the single-ROI plot
        x_col = "y" if "y" in spots.columns else "y_"
        y_col = "x" if "x" in spots.columns else "x_"
        scatter_spots(
            ax,
            spots,
            x_col=x_col,
            y_col=y_col,
            max_points=params.subsample,
            include_scale_bar=(idx == 0),
            scale_bar_length=scale_bar_length if idx == 0 else None,
            scale_bar_label=f"{params.scale_bar_um} μm" if idx == 0 else None,
            title=title,
        )

    for ax in axes:
        if not ax.has_data():
            fig.delaxes(ax)

    fig.suptitle(f"Spots Overview — {codebook}", color=DARK_PANEL_STYLE["axes.titlecolor"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    combined_path = (output_dir / f"spots_all--{codebook}.png").resolve()
    # Clamp DPI to avoid exceeding Agg backend limits (~65535 px on a side)
    save_dpi = render_dpi
    fig.savefig(combined_path.as_posix(), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved combined spots overview: {combined_path}")
    return combined_path


def _label_contour_levels(contour_set: QuadContourSet, max_level_index: int = 10) -> None:
    """Annotate contour lines with their level index up to a provided maximum."""
    levels = getattr(contour_set, "levels", None)
    if levels is None:
        return
    labels = {level: f"{idx}" for idx, level in enumerate(levels) if 1 <= idx <= max_level_index}
    if not labels:
        return
    contour_set.clabel(levels=list(labels.keys()), fmt=labels, inline=True, fontsize=8)


def _filter_contour_levels(levels: np.ndarray, cutoff: int = 10, spacing: int = 5) -> np.ndarray:
    """Reduce contour levels after a cutoff to every `spacing`th entry."""
    if levels.size == 0:
        return levels
    indices = np.arange(levels.size)
    mask = (indices <= cutoff) | ((indices > cutoff) & (((indices - cutoff) % spacing) == 0))
    return levels[mask]


def _render_hexbin_panel(
    ax: Axes,
    spots_df: pl.DataFrame,
    title: str,
    X: np.ndarray,
    Y: np.ndarray,
    z_smooth: np.ndarray,
    contour_levels: np.ndarray,
    cmap: str,
    *,
    show_ylabel: bool,
) -> None:
    """Render a single hexbin + contour panel on the provided axis."""
    if spots_df.is_empty():
        ax.text(
            0.5,
            0.5,
            f"No {title.lower()}",
            ha="center",
            va="center",
            color=DARK_PANEL_STYLE["text.color"],
        )
    else:
        ax.hexbin(
            spots_df["x_"].to_numpy(),
            spots_df["y_"].to_numpy(),
            gridsize=250,
            cmap=cmap,
            mincnt=5,
            linewidths=0,
        )
    contour = ax.contour(X, Y, z_smooth, levels=contour_levels, colors="white", linewidths=0.4, alpha=0.6)
    _label_contour_levels(contour)
    ax.set_xlabel("Area (cube-root, jittered)")
    if show_ylabel:
        ax.set_ylabel("log10(norm * (1 - distance))")
    ax.set_title(title, color=DARK_PANEL_STYLE["axes.titlecolor"])
    configure_dark_axes(ax)


# --- Pipeline Stage Functions ---


def _load_spots_data(path: Path, roi: str, codebook: Codebook) -> pl.DataFrame | None:
    """
    Load decoded spots for a ROI and enrich with convenience columns.

    Adds
    - ``roi`` (Utf8): The ROI identifier used to load the file.
    - ``is_blank`` (Boolean): Targets flagged by name (``startswith("Blank")``).
    - ``bit0``/``bit1``/``bit2`` (UInt8): From the codebook join on ``target``.

    Passes through the core spatial/QC columns that exist in the parquet, e.g.,
    ``x``, ``y``, ``z``, ``area``, ``distance``, ``norm``, ``tile``,
    ``passes_thresholds`` (see module ``fishtools.analysis.spots``).
    """
    spots_path = path / f"registered--{roi}+{codebook.name}" / f"decoded-{codebook.name}" / "spots.parquet"
    if not spots_path.exists():
        logger.warning(f"Spots file not found for ROI {roi}, skipping: {spots_path}")
        return None

    logger.debug(f"Loading spots for ROI {roi} from {spots_path.name}")
    df = (
        pl.read_parquet(spots_path)
        .with_columns(is_blank=pl.col("target").str.starts_with("Blank"), roi=pl.lit(roi))
        .join(codebook.to_dataframe(), on="target", how="left")
    )
    mtime = datetime.fromtimestamp(spots_path.stat().st_mtime)
    logger.debug(f"-> Found {len(df):,} spots. Data timestamp: {mtime:%Y-%m-%d %H:%M:%S}")
    return df


def _apply_initial_filters(
    spots: pl.DataFrame,
    rng: np.random.Generator,
    params: SpotThresholdParams,
) -> pl.DataFrame:
    """
    Apply basic filters and add feature-space axes used for density analysis.

    Returns a DataFrame that includes two engineered columns:
    - ``x_`` = ``area ** (1/3)`` plus small jitter (feature-space, not spatial)
    - ``y_`` = ``log10(norm * (1 - distance))`` (feature-space)

    Important
    - These ``x_``/``y_`` columns are not tile-local pixel coordinates. Tile-local
      spatial coordinates, when present in decoded data produced by
      ``preprocess spots batch``/``stitch``, are named ``x_local`` and
      ``y_local``. Global mosaic coordinates are named ``x`` and ``y``.
    """
    logger.debug("Applying initial filters and engineering features...")
    spots_ = spots.filter(
        pl.col("area").is_between(params.area_min, params.area_max) & pl.col("norm").gt(params.norm_threshold)
    )
    logger.debug(f"Spots after area/norm filter: {len(spots_):,}")

    if params.min_norm:
        spots_ = spots_.filter(pl.col("norm") >= params.min_norm)
        logger.debug(f"Spots after minimum norm filter ({params.min_norm}): {len(spots_):,}")

    spots_ = spots_.with_columns(
        x_=(pl.col("area")) ** (1 / 3) + rng.uniform(-0.75, 0.75, size=len(spots_)),
        y_=(pl.col("norm") * (1 - pl.col("distance"))).log10(),
    ).filter(pl.col("distance") < params.distance_threshold)
    logger.debug(f"Spots after distance filter: {len(spots_):,}")
    return spots_


def _compute_contour_levels(z_smooth: np.ndarray, mode: str, n_levels: int) -> np.ndarray:
    """Compute monotonic contour levels for a density map under different spacings.

    - linear: evenly spaced in value
    - log: evenly spaced in log10(value); requires positive min; uses a tiny floor if needed
    - sqrt: evenly spaced in sqrt(value) then squared back
    """
    z = np.asarray(z_smooth, dtype=float)
    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        # Fallback: Matplotlib will error on degenerate levels; caller guards earlier too
        return np.linspace(vmin, vmax + 1e-12, n_levels)

    if mode == "linear":
        return np.linspace(vmin, vmax, n_levels)
    if mode == "sqrt":
        vmin_clamped = max(vmin, 0.0)
        return np.linspace(np.sqrt(vmin_clamped), np.sqrt(vmax), n_levels) ** 2
    if mode == "log":
        # Ensure strictly positive lower bound; pick smallest positive or a small fraction of vmax
        positive = z[z > 0]
        if positive.size == 0:
            # If all non-positive, fall back to linear
            return np.linspace(vmin, vmax, n_levels)
        lo = float(np.min(positive))
        # Guard extremely tiny lower bound to avoid exploding ranges
        lo = max(lo, vmax * 1e-6)
        return 10 ** np.linspace(np.log10(lo), np.log10(vmax), n_levels)
    raise ValueError(f"Unsupported contour mode: {mode}")


def _calculate_density_map(
    spots_: pl.DataFrame,
    params: SpotThresholdParams,
) -> tuple[Figure, QuadContourSet, RegularGridInterpolator, DensitySurface]:
    """Calculates and smooths the blank proportion density map."""
    logger.debug("Calculating blank proportion density map...")
    bounds = spots_.select([
        pl.col("x_").min().alias("x_min"),
        pl.col("x_").max().alias("x_max"),
        pl.col("y_").min().alias("y_min"),
        pl.col("y_").max().alias("y_max"),
    ]).row(0, named=True)

    x_coords = np.linspace(bounds["x_min"], bounds["x_max"], params.density_grid_size)
    y_coords = np.linspace(bounds["y_min"], bounds["y_max"], params.density_grid_size)
    X, Y = np.meshgrid(x_coords, y_coords)

    step_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1
    step_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1

    binned_counts = (
        spots_.with_columns([
            ((pl.col("x_") - bounds["x_min"]) / step_x).floor().cast(pl.Int32).alias("i"),
            ((pl.col("y_") - bounds["y_min"]) / step_y).floor().cast(pl.Int32).alias("j"),
        ])
        .filter(
            (pl.col("i") >= 0)
            & (pl.col("i") < params.density_grid_size)
            & (pl.col("j") >= 0)
            & (pl.col("j") < params.density_grid_size)
        )
        .group_by(["i", "j"])
        .agg([pl.sum("is_blank").alias("blank_count"), pl.len().alias("total_count")])
        .with_columns((pl.col("blank_count") / (pl.col("total_count") + 1e-9)).alias("proportion"))
    )

    if params.min_spots_per_bin > 1:
        before = binned_counts.height
        binned_counts = binned_counts.filter(pl.col("total_count") >= params.min_spots_per_bin)
        removed = before - binned_counts.height
        if removed:
            logger.debug(
                "Removed %d sparse bins (< %d spots) before smoothing",
                removed,
                params.min_spots_per_bin,
            )

    Z = np.zeros_like(X, dtype=float)
    metric = params.density_metric
    for row in binned_counts.iter_rows(named=True):
        value = float(row["blank_count"]) if metric == "count" else float(row["proportion"])
        Z[row["j"], row["i"]] = value
    if x_coords[0] > 1.0:
        x_coords = np.insert(x_coords, 0, 1.0)
        Z = np.insert(Z, 0, Z[:, 0], axis=1)

    Z_smooth = gaussian_filter(Z, sigma=params.density_smooth_sigma) if params.density_smooth_sigma > 0 else Z
    X, Y = np.meshgrid(x_coords, y_coords)

    # Build a real figure showing the raw density heatmap (Z) with smoothed contours overlaid.
    fig, ax = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    if np.allclose(Z_smooth, Z_smooth.flat[0]):
        logger.warning("Histogram-based density map is constant; skipping contour generation.")
        raise RuntimeError("Constant density map. No spots or all spots identical?")

    vmax = np.percentile(Z_smooth, 99.9)
    if np.isclose(vmax, 0.0):
        vmax = 1e-6
    im = ax.pcolormesh(X, Y, Z, shading="auto", vmin=0, vmax=vmax)  # Z is intentional to show raw values.
    levels = _compute_contour_levels(Z_smooth, params.contour_mode, params.contour_levels)
    contours = ax.contour(X, Y, Z_smooth, levels=levels, colors="white", alpha=0.5, linewidths=0.5)
    ax.set_xlabel("Area")
    ax.set_ylabel("Norm * (1 - Distance) [log10]")
    title_metric = "Blank Count" if metric == "count" else "Blank Proportion"
    ax.set_title(f"{title_metric} Density with Smoothed Contours")
    fig.colorbar(im, ax=ax, label=title_metric)

    if not contours.levels.size:  # type: ignore[attr-defined]
        raise RuntimeError("Could not generate density contours. The data may be too sparse or uniform.")

    surface = DensitySurface(
        x_coords=x_coords,
        y_coords=y_coords,
        z_raw=Z,
        z_smooth=Z_smooth,
        contour_levels=np.asarray(contours.levels).copy(),
    )
    interp_func = RegularGridInterpolator((y_coords, x_coords), Z_smooth, bounds_error=False, fill_value=0)
    return fig, contours, interp_func, surface


def _create_spots_contours_figure(
    spots_: pl.DataFrame,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
    surface: DensitySurface,
) -> Figure | None:
    """Render side-by-side scatter panels for blank and non-blank spots."""
    if spots_.is_empty():
        logger.warning(f"No spots available to plot for ROI {roi}; skipping blank/non-blank panels.")
        return None

    non_blank = spots_.filter(~pl.col("is_blank"))
    blank = spots_.filter(pl.col("is_blank"))

    if non_blank.is_empty() and blank.is_empty():
        logger.warning(
            f"ROI {roi} contains no classified blank or non-blank spots after initial filtering; skipping scatter panels."
        )
        return

    with (
        sns.axes_style("dark", rc=DARK_PANEL_STYLE),
        sns.plotting_context(rc={"axes.titlesize": 14}),
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(params.figsize_thresh[0] * 2, params.figsize_thresh[1]),
            dpi=params.dpi,
            sharey=True,
        )

        X, Y = surface.meshgrid()
        filtered_levels = _filter_contour_levels(surface.contour_levels, cutoff=10, spacing=5)

        panels = (
            (axes[0], non_blank, "Non-blank spots"),
            (axes[1], blank, "Blank spots"),
        )
        for idx, (ax, spots_panel, title) in enumerate(panels):
            _render_hexbin_panel(
                ax,
                spots_panel,
                title,
                X,
                Y,
                surface.z_smooth,
                filtered_levels,
                "inferno",
                show_ylabel=idx == 0,
            )

        x_min = min(float(spots_["x_"].min()), surface.x_coords[0])
        x_max = max(float(spots_["x_"].max()), surface.x_coords[-1])
        y_min, y_max = surface.y_coords[0], surface.y_coords[-1]
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.tight_layout()

    return fig


def _compute_threshold_curve(
    spots_: pl.DataFrame,
    contours: QuadContourSet,
    interp_func: RegularGridInterpolator,
) -> ThresholdCurve:
    """Compute threshold curve statistics for a single ROI."""
    threshold_levels = list(range(1, min(len(contours.levels), 15), 2))  # type: ignore
    spot_counts: list[int] = []
    blank_proportions: list[float] = []
    point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())

    for level_idx in threshold_levels:
        threshold_value = contours.levels[level_idx]  # type: ignore
        spots_ok = spots_.filter(pl.lit(point_densities) < threshold_value)
        spot_counts.append(len(spots_ok))
        blank_proportions.append(spots_ok.filter(pl.col("is_blank")).height / max(1, len(spots_ok)))

    max_level = len(contours.levels) - 1  # type: ignore
    return ThresholdCurve(
        levels=threshold_levels,
        spot_counts=spot_counts,
        blank_proportions=blank_proportions,
        max_level=max_level,
    )


def _save_threshold_plot(
    curve: ThresholdCurve,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
) -> Path:
    """Persist the per-ROI threshold selection plot."""
    sns.set_theme()
    fig_thresh, ax1 = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    ax1.plot(
        curve.levels,
        curve.spot_counts,
        color="g",
        linestyle="-",
        label="Remaining Spots",
    )
    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    # Use SI-prefix formatting without spaces and without unnecessary trailing .0
    ax1.yaxis.set_major_formatter(si_tick_formatter())
    ax1.set_ylim(0, None)  # type: ignore[arg-type]

    ax2 = ax1.twinx()
    ax2.plot(
        curve.levels,
        curve.blank_proportions,
        color="r",
        linestyle="--",
        label="Blank Proportion",
    )
    ax2.set_ylabel("Blank Proportion", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title(f"Filter Threshold Selection for ROI: {roi}")
    fig_thresh.tight_layout()
    save_figure(fig_thresh, output_dir, "threshold_selection", roi, codebook)
    return (output_dir / f"threshold_selection--{roi}+{codebook}.png").resolve()


def _save_combined_threshold_plot(
    curves_by_roi: dict[str, ThresholdCurve],
    output_dir: Path,
    codebook: str,
    params: SpotThresholdParams,
) -> Path:
    """Create an overlay threshold-selection plot for all ROIs."""
    if not curves_by_roi:
        raise ValueError("No threshold curves available to plot.")

    ordered_rois = sorted(curves_by_roi)
    palette = sns.color_palette("husl", len(ordered_rois))

    sns.set_theme()
    fig, ax1 = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    ax2 = ax1.twinx()

    roi_handles: list[Line2D] = []
    for color, roi in zip(palette, ordered_rois):
        curve = curves_by_roi[roi]
        ax1.plot(curve.levels, curve.spot_counts, color=color, linestyle="-")
        ax2.plot(curve.levels, curve.blank_proportions, color=color, linestyle="--")
        roi_handles.append(Line2D([0], [0], color=color, linewidth=2, label=roi))

    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=2, label="Remaining Spots"),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Blank Proportion",
        ),
    ]

    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots")
    # Apply SI-prefix formatting without spaces and without unnecessary trailing .0
    ax1.yaxis.set_major_formatter(si_tick_formatter())
    ax2.set_ylabel("Blank Proportion")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"Threshold Selection Curves | Codebook {codebook}")

    legend1 = ax1.legend(handles=roi_handles, title="ROI", loc="upper right")
    ax1.legend(handles=style_handles, title=None, loc="lower right")
    ax1.add_artist(legend1)

    fig.tight_layout()
    combined_path = (output_dir / f"threshold_selection_all+{codebook}.png").resolve()
    fig.savefig(combined_path.as_posix(), dpi=params.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved plot: {combined_path}")
    return combined_path


def _prompt_threshold_levels(
    ordered_rois: list[str],
    contexts: dict[str, ROIThresholdContext],
    output_dir: Path,
    codebook: str,
    *,
    combined_spots_path: Path | None = None,
) -> dict[str, int]:
    """Prompt for comma-separated threshold levels across all ROIs."""

    if not ordered_rois:
        return {}

    combined_plot_path = output_dir / f"threshold_selection_all+{codebook}.png"
    lines = ["Generated artifacts:"]
    for roi in ordered_rois:
        artifacts = contexts[roi].artifact_paths
        contour = artifacts.get("contours")
        scatter = artifacts.get("spots_contours")
        threshold = artifacts.get("threshold")
        lines.append(f"  {roi}:")
        if scatter:
            lines.append(f"    Spots:     {scatter}")
        lines.append(f"    Contours:  {contour}")
        lines.append(f"    Threshold: {threshold}")

    lines.append("")
    lines.append(f"Combined plot: {combined_plot_path.resolve()}")
    if combined_spots_path is not None:
        lines.append(f"Combined spots: {combined_spots_path}")
    lines.append("")
    lines.append("Enter threshold levels for each ROI (comma-separated) in the order shown below:")
    for roi in ordered_rois:
        max_level = contexts[roi].curve.max_level
        lines.append(f"  - {roi}: 0-{max_level}")

    lines.append("")
    lines.append("Please enter the threshold levels now (comma-separated integers):")

    def _validate(value: str) -> bool | str:
        values = [part.strip() for part in value.split(",")]
        if len(values) != len(ordered_rois):
            return f"Expected {len(ordered_rois)} comma-separated values."

        levels: list[int] = []
        for roi, raw_level in zip(ordered_rois, values):
            if raw_level == "":
                return f"ROI {roi}: level is required."
            try:
                level = int(raw_level)
            except ValueError:
                return f"ROI {roi}: level must be an integer."
            max_level = contexts[roi].curve.max_level
            if not 0 <= level <= max_level:
                return f"ROI {roi}: level must be between 0 and {max_level}."
            levels.append(level)
        return True

    response = questionary.text("\n".join(lines), validate=_validate).ask()
    if response is None:
        logger.warning("KeyboardInterrupt detected during threshold input. Exiting immediately.")
        raise typer.Exit(code=1)

    levels = [int(part.strip()) for part in response.split(",")]
    return dict(zip(ordered_rois, levels))


def _apply_final_filter(
    spots_: pl.DataFrame,
    interp_func: RegularGridInterpolator,
    contours: QuadContourSet,
    threshold_level: int,
) -> pl.DataFrame:
    """Apply the final density-based filter to the spot data.

    Temporary columns
    - ``point_density``: interpolated density value at each spot in feature
      space (based on ``x_``, ``y_``); added for filtering and dropped in the
      saved parquet.
    """
    logger.debug(f"Applying final filter at threshold level: {threshold_level}")
    final_threshold_value = contours.levels[threshold_level]  # type: ignore
    point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())
    spots_ok = spots_.with_columns(point_density=point_densities).filter(
        pl.col("point_density") < final_threshold_value
    )

    logger.debug(f"Final filtered spots: {len(spots_ok):,}")
    blank_prop = spots_ok.filter(pl.col("is_blank")).height / max(1, len(spots_ok))
    logger.debug(f"Blank proportion in final set: {blank_prop:.2%}")
    return spots_ok


def _generate_final_outputs(
    spots_ok: pl.DataFrame,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
):
    """Generates all final plots and saves the filtered data for a single ROI."""
    logger.debug("Generating final plots and saving data...")
    # Final Spots Spatial Plot
    fig_spots, ax = plt.subplots(figsize=params.figsize_spots, dpi=params.dpi)
    x_col = "y" if "y" in spots_ok.columns else "y_"
    y_col = "x" if "x" in spots_ok.columns else "x_"
    scatter_spots(
        ax,
        spots_ok,
        x_col=x_col,
        y_col=y_col,
        max_points=params.subsample,
        include_scale_bar=True,
        scale_bar_length=params.scale_bar_um / params.pixel_size_um,
        scale_bar_label=f"{params.scale_bar_um} μm",
        title=f"Filtered spots for {roi} (n={len(spots_ok):,})",
    )

    if {"x", "y"}.issubset(spots_ok.columns):
        configure_micron_axes(ax, params.pixel_size_um, x_label="X", y_label="Y")
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    save_figure(fig_spots, output_dir, "spots_final", roi, codebook, log_level="INFO")

    # Save blank counts
    per_gene_final = count_by_gene(spots_ok)
    per_gene_final.filter(pl.col("is_blank")).sort("count", descending=True).write_csv(
        output_dir / f"blanks--{roi}+{codebook}.csv"
    )

    # Final Scree Plot
    fig_scree, ax_scree = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    blank_prop = per_gene_final.filter(pl.col("is_blank"))["count"].sum() / per_gene_final["count"].sum()
    total_spots = per_gene_final["count"].sum()
    ax_scree.bar(
        per_gene_final["target"],
        per_gene_final["count"],
        color=per_gene_final["color"],
        width=1,
        align="edge",
        linewidth=0,
    )
    ax_scree.set_xticks([])
    ax_scree.set_yscale("log")
    ax_scree.set_xlabel("Gene")
    ax_scree.set_ylabel("Count (log scale)")
    ax_scree.set_title(
        f"{codebook} | ROI {roi} (n={total_spots:,}, {blank_prop:.2%} blank)",
        loc="left",
    )
    fig_scree.tight_layout()
    save_figure(fig_scree, output_dir, "scree_final", roi, codebook, log_level="INFO")

    # Save final filtered data
    output_parquet = output_dir / f"{roi}+{codebook}.parquet"
    spots_ok.drop("point_density", "x_", "y_").write_parquet(output_parquet)
    logger.debug(f"Saved filtered spots for ROI {roi} to {output_parquet}")


# --- Main CLI Command ---


@click.command()
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
@click.option(
    "--codebook",
    "-c",
    "codebook_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
    help="Name of the codebook (stem of the .json file).",
)
@click.argument(
    "roi",
    type=str,
    default="*",
)
@click.argument("rois", nargs=-1)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path),
    default=None,
    help="Directory to save outputs. [default: 'path.parent/output']",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path (TOML format)",
)
@click.option("--area-min", type=float, help="Override minimum spot area")
@click.option("--area-max", type=float, help="Override maximum spot area")
@click.option("--norm-threshold", type=float, help="Override norm threshold")
@click.option("--min-norm", type=float, help="Minimum norm filter applied before density analysis")
@click.option("--distance-threshold", type=float, help="Override distance threshold")
@click.option("--seed", type=int, help="Override random seed")
@click.option(
    "--contour-mode",
    type=click.Choice(["linear", "log", "sqrt"], case_sensitive=False),
    help="Spacing mode for density contours",
)
@click.option("--contour-levels", type=int, help="Number of contour levels")
def threshold(
    path: Path,
    codebook_path: Path,
    roi: str,
    rois: tuple[str, ...],
    output_dir: Path | None = None,
    config: Path | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    norm_threshold: float | None = None,
    min_norm: float | None = None,
    distance_threshold: float | None = None,
    seed: int | None = None,
    contour_mode: str | None = None,
    contour_levels: int | None = None,
):
    """
    Process spot data for each ROI individually with an interactive threshold selection step.
    """
    wildcard_tokens = {"*", "all"}
    requested_tokens = [token for token in (roi, *rois) if token]
    includes_wildcard = any(token in wildcard_tokens for token in requested_tokens)
    explicit_requested = [token for token in requested_tokens if token not in wildcard_tokens]
    log_roi_value = ",".join(requested_tokens) if requested_tokens else "all"
    if includes_wildcard and not explicit_requested:
        log_roi_value = "all"

    setup_cli_logging(
        path,
        component="preprocess.spotlook.threshold",
        file=f"spotlook-threshold-{codebook_path.stem}",
        extra={
            "codebook": codebook_path.stem,
            "roi": log_roi_value,
        },
    )

    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Using output directory: {output_dir}")

    # Load configuration and create parameters model
    params = build_spotlook_params(
        config_path=config,
        area_min=area_min,
        area_max=area_max,
        norm_threshold=norm_threshold,
        min_norm=min_norm,
        distance_threshold=distance_threshold,
        seed=seed,
        contour_mode=contour_mode.lower() if contour_mode else None,
        contour_levels=contour_levels,
    )
    logger.debug(
        "Using analysis parameters: seed=%s, area=[%.3f, %.3f], norm_threshold=%.4f, min_norm=%s",
        params.seed,
        params.area_min,
        params.area_max,
        params.norm_threshold,
        params.min_norm,
    )

    codebook = Codebook(codebook_path)

    ws = Workspace(path)
    if includes_wildcard or not explicit_requested:
        rois_to_process = ws.rois
    else:
        unique_requested: list[str] = []
        seen: set[str] = set()
        for token in explicit_requested:
            if token not in seen:
                unique_requested.append(token)
                seen.add(token)
        rois_to_process = ws.resolve_rois(unique_requested)
    if not rois_to_process:
        logger.error(f"No ROIs found or specified in workspace: {path}")
        raise typer.Exit(code=1)

    logger.debug(f"Found {len(rois_to_process)} ROIs to process individually: {rois_to_process}")

    ordered_rois = sorted(rois_to_process)
    contexts: dict[str, ROIThresholdContext] = {}
    skipped_rois: list[str] = []

    # --- Phase 1: Pre-compute plots and curves ---
    for i, roi in enumerate(ordered_rois, 1):
        logger.info(f"Starting ROI {roi} ({i}/{len(ordered_rois)}) — preparing initial plots...")
        rng = np.random.default_rng(params.seed)

        logger.info(f"Analyzing spot density patterns for ROI {roi}...")
        logger.debug("Attempting to load raw spots parquet")
        spots_raw = _load_spots_data(path, roi, codebook)
        if spots_raw is None or spots_raw.is_empty():
            logger.warning(f"No data loaded for ROI {roi}. Skipping to next.")
            skipped_rois.append(roi)
            continue

        logger.debug("Applying initial filters to %d raw spots", len(spots_raw))
        spots_intermediate = _apply_initial_filters(spots_raw, rng, params)
        if spots_intermediate.is_empty():
            logger.warning(f"No spots remained after initial filters for ROI {roi}. Skipping.")
            skipped_rois.append(roi)
            continue

        logger.debug("Computing density map on %d filtered spots", len(spots_intermediate))
        density_results = _calculate_density_map(spots_intermediate, params)

        fig_contours, contours, interp_func, surface = density_results
        save_figure(fig_contours, output_dir, "contours", roi, codebook.name)
        contour_path = (output_dir / f"contours--{roi}+{codebook.name}.png").resolve()

        fig_blank_panels = _create_spots_contours_figure(
            spots_intermediate,
            roi,
            codebook.name,
            params,
            surface,
        )
        spots_contours_path: Path | None = None
        if fig_blank_panels is not None:
            save_figure(fig_blank_panels, output_dir, "spots_contours", roi, codebook.name)
            spots_contours_path = (output_dir / f"spots_contours--{roi}+{codebook.name}.png").resolve()

        logger.debug("Computing threshold curve statistics")
        curve = _compute_threshold_curve(spots_intermediate, contours, interp_func)
        threshold_plot_path = _save_threshold_plot(curve, output_dir, roi, codebook.name, params)

        contexts[roi] = ROIThresholdContext(
            spots=spots_intermediate,
            contours=contours,
            interpolator=interp_func,
            curve=curve,
            artifact_paths={
                "contours": contour_path,
                "spots_contours": spots_contours_path,
                "threshold": threshold_plot_path,
            },
        )

    if skipped_rois:
        logger.warning(f"Skipped {len(skipped_rois)} ROI(s) due to missing data: {skipped_rois}")

    if not contexts:
        logger.error("No ROIs produced threshold curves. Exiting.")
        raise typer.Exit(code=1)

    curves_for_plot = {roi: ctx.curve for roi, ctx in contexts.items()}
    logger.debug("Saving combined threshold selection plot for %d ROIs", len(curves_for_plot))
    combined_plot_path = _save_combined_threshold_plot(curves_for_plot, output_dir, codebook.name, params)
    logger.debug(f"Saved combined threshold plot: {combined_plot_path}")

    active_rois = sorted(contexts)
    selected_levels = _prompt_threshold_levels(active_rois, contexts, output_dir, codebook.name)

    # --- Phase 2: Apply thresholds and finalize outputs ---
    for i, roi in enumerate(active_rois, 1):
        chosen_level = selected_levels[roi]
        logger.info(
            f"Starting post-threshold processing for ROI {roi} ({i}/{len(active_rois)}) at level {chosen_level}"
        )
        ctx = contexts[roi]

        logger.debug(f"Applying final density-based filter at level {chosen_level} for ROI {roi}...")
        logger.debug("Interpolating densities for %d candidate spots", len(ctx.spots))
        ctx.spots_final = _apply_final_filter(ctx.spots, ctx.interpolator, ctx.contours, chosen_level)

        logger.debug("Generating plots and saving results...")
        _generate_final_outputs(ctx.spots_final, output_dir, roi, codebook.name, params)

        logger.debug(f"Finished ROI {roi} ({i}/{len(active_rois)})")

    combined_spots_all = _save_combined_spots_plot(contexts, output_dir, codebook.name, params)

    logger.info(f"Saved combined spots overview: {combined_spots_all.absolute()}")

    logger.debug("All specified ROIs have been processed.")
