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

from fishtools.preprocess.config import SpotThresholdParams
from fishtools.utils.io import Codebook, Workspace
from fishtools.utils.plot import (
    DARK_PANEL_STYLE,
    add_scale_bar,
    configure_dark_axes,
    si_tick_formatter,
)
from fishtools.utils.utils import configure_cli_logging, initialize_logger

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
    """Cached per-ROI artifacts between the preparation and execution phases."""

    spots: pl.DataFrame
    contours: QuadContourSet
    interpolator: RegularGridInterpolator
    curve: ThresholdCurve
    artifact_paths: dict[str, Path | None]


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
    """Counts spots per gene, adding metadata for plotting."""
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


def save_figure(fig: Figure, output_dir: Path, name: str, roi: str, codebook: str):
    """Saves a matplotlib figure to the output directory with a standardized name for a single ROI."""
    filename = output_dir / f"{name}--{roi}+{codebook}.png"
    fig.savefig(filename.as_posix(), dpi=300, bbox_inches="tight")
    logger.debug(f"Saved plot: {filename}")
    plt.close(fig)


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
    """Loads spot data from a parquet file for a single ROI."""
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
    """Applies initial area/norm filters and engineers features for density analysis."""
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
                "Removed %d sparse bins (< %d spots) before smoothing", removed, params.min_spots_per_bin
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

    with sns.axes_style("dark", rc=DARK_PANEL_STYLE), sns.plotting_context(rc={"axes.titlesize": 14}):
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
    ax1.plot(curve.levels, curve.spot_counts, color="g", linestyle="-", label="Remaining Spots")
    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    # Use SI-prefix formatting without spaces and without unnecessary trailing .0
    ax1.yaxis.set_major_formatter(si_tick_formatter())
    ax1.set_ylim(0, None)  # type: ignore[arg-type]

    ax2 = ax1.twinx()
    ax2.plot(curve.levels, curve.blank_proportions, color="r", linestyle="--", label="Blank Proportion")
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
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Blank Proportion"),
    ]

    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots")
    # Apply SI-prefix formatting without spaces and without unnecessary trailing .0
    ax1.yaxis.set_major_formatter(si_tick_formatter())
    ax2.set_ylabel("Blank Proportion")
    ax1.set_title(f"Threshold Selection Curves | Codebook {codebook}")

    legend1 = ax1.legend(handles=roi_handles, title="ROI", loc="upper right")
    legend2 = ax1.legend(handles=style_handles, title=None, loc="lower right")
    ax1.add_artist(legend1)

    fig.tight_layout()
    combined_path = (output_dir / f"threshold_selection_all+{codebook}.png").resolve()
    fig.savefig(combined_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved plot: {combined_path}")
    return combined_path


def _prompt_threshold_levels(
    ordered_rois: list[str],
    contexts: dict[str, ROIThresholdContext],
    output_dir: Path,
    codebook: str,
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
    lines.append("")
    lines.append("Enter threshold levels for each ROI (comma-separated) in the order shown below:")
    for roi in ordered_rois:
        max_level = contexts[roi].curve.max_level
        lines.append(f"  - {roi}: 0-{max_level}")

    lines.append("")
    lines.append("Please enter the threshold levels now (comma-separated integers):")

    def _validate(value: str) -> bool | str:
        values = [part.strip() for part in value.split(",") if part.strip()]
        if len(values) != len(ordered_rois):
            return f"Expected {len(ordered_rois)} comma-separated values."
        try:
            levels = [int(part) for part in values]
        except ValueError:
            return "Thresholds must be integers."

        for roi, level in zip(ordered_rois, levels):
            max_level = contexts[roi].curve.max_level
            if not 0 <= level <= max_level:
                return f"ROI {roi}: level must be between 0 and {max_level}."
        return True

    response = questionary.text("\n".join(lines), validate=_validate).ask()
    if response is None:
        logger.warning("KeyboardInterrupt detected during threshold input. Exiting immediately.")
        raise typer.Exit(code=1)

    levels = [int(part.strip()) for part in response.split(",")]
    return dict(zip(ordered_rois, levels))


def _apply_final_filter(
    spots_: pl.DataFrame, interp_func: RegularGridInterpolator, contours: QuadContourSet, threshold_level: int
) -> pl.DataFrame:
    """Applies the final density-based filter to the spot data."""
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
    spots_ok: pl.DataFrame, output_dir: Path, roi: str, codebook: str, params: SpotThresholdParams
):
    """Generates all final plots and saves the filtered data for a single ROI."""
    logger.debug("Generating final plots and saving data...")
    # Final Spots Spatial Plot
    fig_spots, ax = plt.subplots(figsize=params.figsize_spots, dpi=params.dpi)
    subsample = max(1, len(spots_ok) // params.subsample)
    ax.scatter(spots_ok["y"][::subsample], spots_ok["x"][::subsample], s=0.1, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    add_scale_bar(ax, params.scale_bar_um / params.pixel_size_um, f"{params.scale_bar_um} μm")
    ax.set_title(f"Filtered Spots for ROI {roi} (n={len(spots_ok):,})")
    save_figure(fig_spots, output_dir, "spots_final", roi, codebook)

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
        f"Gene Counts for {codebook} | ROI {roi} (n={total_spots:,}, {blank_prop:.1%} blank)", loc="left"
    )
    fig_scree.tight_layout()
    save_figure(fig_scree, output_dir, "scree_final", roi, codebook)

    # Save final filtered data
    output_parquet = output_dir / f"{roi}+{codebook}.parquet"
    spots_ok.drop("point_density", "x_", "y_").write_parquet(output_parquet)
    logger.debug(f"Saved filtered spots for ROI {roi} to {output_parquet}")


# --- Main CLI Command ---


@click.command()
@click.argument(
    "path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--codebook",
    "-c",
    "codebook_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Name of the codebook (stem of the .json file).",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path),
    default=None,
    help="Directory to save outputs. [default: 'path.parent/output']",
)
@click.option(
    "--roi",
    "rois",
    multiple=True,
    help="Specific ROI to process. Can be used multiple times. [default: all found ROIs]",
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
    output_dir: Path | None = None,
    rois: list[str] | None = None,
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
    initialize_logger()
    # Raise console verbosity to INFO for a clean, minimal UX.
    configure_cli_logging(
        workspace=None,
        component="spotlook",
        console_level="INFO",
        file_level="CRITICAL",
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
    rois_to_process = rois if rois else ws.rois
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
        spots_final = _apply_final_filter(ctx.spots, ctx.interpolator, ctx.contours, chosen_level)

        logger.debug("Generating plots and saving results...")
        _generate_final_outputs(spots_final, output_dir, roi, codebook.name, params)

        logger.debug(f"Finished ROI {roi} ({i}/{len(active_rois)})")

    logger.debug("All specified ROIs have been processed.")
