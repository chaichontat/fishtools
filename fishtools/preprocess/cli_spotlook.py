import sys
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
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from fishtools.utils.io import Workspace
from fishtools.utils.plot import add_scale_bar
from fishtools.utils.utils import initialize_logger


class SpotlookParams(BaseModel):
    """Parameters for spotlook analysis with validation and type safety."""

    # Core filtering parameters
    area_min: float = Field(default=10.0, gt=0, description="Minimum spot area in pixels")
    area_max: float = Field(default=200.0, gt=0, description="Maximum spot area in pixels")
    norm_threshold: float = Field(
        default=0.007, gt=0, description="Normalization threshold for spot filtering"
    )
    distance_threshold: float = Field(default=0.3, gt=0, description="Distance threshold for spot filtering")

    # Density analysis parameters
    density_grid_size: int = Field(default=50, gt=0, description="Grid size for density map calculation")
    density_smooth_sigma: float = Field(
        default=4.0, gt=0, description="Gaussian smoothing sigma for density map"
    )
    min_spots_for_density: int = Field(
        default=100, gt=0, description="Minimum spots required for density analysis"
    )

    # Reproducibility
    seed: int = Field(default=0, ge=0, description="Random seed for reproducible analysis")

    # Visualization settings
    subsample: int = Field(default=200000, gt=0, description="Maximum spots to display in plots")
    scale_bar_um: float = Field(default=1000.0, gt=0, description="Scale bar size in micrometers")
    dpi: int = Field(default=200, gt=0, description="Plot resolution in DPI")
    figsize_spots: tuple[float, float] = Field(default=(10.0, 10.0), description="Figure size for spot plots")
    figsize_thresh: tuple[float, float] = Field(
        default=(8.0, 6.0), description="Figure size for threshold plots"
    )

    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
        area_min: float | None = None,
        area_max: float | None = None,
        norm_threshold: float | None = None,
        distance_threshold: float | None = None,
        seed: int | None = None,
    ) -> "SpotlookParams":
        """Create SpotlookParams from config file with CLI overrides."""
        try:
            from fishtools.preprocess.config_loader import load_config

            if config_path:
                config = load_config(config_path)
                spot_config = config.spot_analysis

                # Extract from nested config structure
                params = cls(
                    area_min=spot_config.area_range[0],
                    area_max=spot_config.area_range[1],
                    norm_threshold=spot_config.norm_threshold,
                    distance_threshold=spot_config.distance_threshold,
                    density_grid_size=int(spot_config.density["grid_size"]),
                    density_smooth_sigma=float(spot_config.density["smooth_sigma"]),
                    min_spots_for_density=int(spot_config.density["min_spots"]),
                    seed=spot_config.seed,
                    subsample=int(spot_config.visualization["subsample"]),
                    scale_bar_um=float(spot_config.visualization["scale_bar_um"]),
                    dpi=int(spot_config.visualization["dpi"]),
                    figsize_spots=tuple(spot_config.visualization["figsize_spots"]),
                    figsize_thresh=tuple(spot_config.visualization["figsize_thresh"]),
                )
            else:
                # Use defaults
                params = cls()

        except ImportError:
            # Fallback if config system unavailable
            params = cls()

        # Apply CLI overrides using Pydantic's update mechanism
        overrides = {
            k: v
            for k, v in {
                "area_min": area_min,
                "area_max": area_max,
                "norm_threshold": norm_threshold,
                "distance_threshold": distance_threshold,
                "seed": seed,
            }.items()
            if v is not None
        }

        if overrides:
            return params.model_copy(update=overrides)
        return params


# --- Analysis Parameters ---
# DEPRECATED: These global constants will be replaced with config-based parameters
SEED = 0
AREA_MIN = 10.0
AREA_MAX = 200.0
NORM_GT = 0.007
DISTANCE_LT = 0.3
DENSITY_GRID_SIZE = 50
DENSITY_SMOOTH_SIGMA = 4.0
MIN_SPOTS_FOR_DENSITY_ANALYSIS = 100


# --- Core Helper Functions ---


def count_by_gene(spots: pl.DataFrame) -> pl.DataFrame:
    """Counts spots per gene, adding metadata for plotting."""
    return (
        spots.group_by("target")
        .len("count")
        .sort("count", descending=True)
        .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
        .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
    )


def save_figure(fig: Figure, output_dir: Path, name: str, roi: str, codebook: str):
    """Saves a matplotlib figure to the output directory with a standardized name for a single ROI."""
    filename = output_dir / f"{name}--{roi}+{codebook}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {filename}")
    plt.close(fig)


# --- Pipeline Stage Functions ---


def _load_spots_data(path: Path, roi: str, codebook: str) -> pl.DataFrame | None:
    """Loads spot data from a parquet file for a single ROI."""
    spots_path = path / f"registered--{roi}+{codebook}" / f"decoded-{codebook}" / "spots.parquet"
    if not spots_path.exists():
        logger.warning(f"Spots file not found for ROI {roi}, skipping: {spots_path}")
        return None

    logger.info(f"Loading spots for ROI {roi} from {spots_path.name}")
    df = pl.read_parquet(spots_path)
    df = df.with_columns(is_blank=pl.col("target").str.starts_with("Blank"), roi=pl.lit(roi))
    mtime = datetime.fromtimestamp(spots_path.stat().st_mtime)
    logger.info(f"-> Found {len(df):,} spots. Data timestamp: {mtime:%Y-%m-%d %H:%M:%S}")
    return df


def _apply_initial_filters(
    spots: pl.DataFrame,
    rng: np.random.Generator,
    params: SpotlookParams,
) -> pl.DataFrame:
    """Applies initial area/norm filters and engineers features for density analysis."""
    logger.info("Applying initial filters and engineering features...")
    spots_ = spots.filter(
        pl.col("area").is_between(params.area_min, params.area_max) & pl.col("norm").gt(params.norm_threshold)
    )
    logger.info(f"Spots after area/norm filter: {len(spots_):,}")

    spots_ = spots_.with_columns(
        x_=(pl.col("area")) ** (1 / 3) + rng.uniform(-0.75, 0.75, size=len(spots_)),
        y_=(pl.col("norm") * (1 - pl.col("distance"))).log10(),
    ).filter(pl.col("distance") < params.distance_threshold)
    logger.info(f"Spots after distance filter: {len(spots_):,}")
    return spots_


def _calculate_density_map(
    spots_: pl.DataFrame,
    params: SpotlookParams,
) -> tuple[QuadContourSet, RegularGridInterpolator] | None:
    """Calculates and smooths the blank proportion density map."""
    logger.info("Calculating blank proportion density map...")
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

    Z = np.zeros_like(X)
    for row in binned_counts.iter_rows(named=True):
        Z[row["j"], row["i"]] = row["proportion"]
    Z_smooth = gaussian_filter(Z, sigma=params.density_smooth_sigma)

    # Use a dummy figure to generate contours, as we only need the contour object itself.
    with plt.ioff():  # Turn off interactive plotting temporarily
        fig_dummy, ax_dummy = plt.subplots()
        contours = ax_dummy.contour(X, Y, Z_smooth, levels=50)
        plt.close(fig_dummy)

    if not contours.levels.size:
        logger.warning("Could not generate density contours. The data may be too sparse or uniform.")
        return None

    interp_func = RegularGridInterpolator((y_coords, x_coords), Z_smooth, bounds_error=False, fill_value=0)
    return contours, interp_func


def _get_interactive_threshold(
    spots_: pl.DataFrame,
    contours: QuadContourSet,
    interp_func: RegularGridInterpolator,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotlookParams,
) -> int:
    """Generates a threshold plot and interactively asks the user for a threshold level."""
    logger.info("Generating threshold selection plot...")
    threshold_levels = list(range(1, min(len(contours.levels), 15), 2))
    spot_counts, blank_proportions = [], []
    for level_idx in threshold_levels:
        threshold_value = contours.levels[level_idx]
        point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())
        spots_ok = spots_.filter(pl.lit(point_densities) < threshold_value)
        spot_counts.append(len(spots_ok))
        blank_proportions.append(spots_ok.filter(pl.col("is_blank")).height / max(1, len(spots_ok)))

    sns.set_theme()
    fig_thresh, ax1 = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    ax1.plot(threshold_levels, spot_counts, "g-", label="Remaining Spots")
    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax1.set_ylim(0, None)
    ax2 = ax1.twinx()
    ax2.plot(threshold_levels, blank_proportions, "r-", label="Blank Proportion")
    ax2.set_ylabel("Blank Proportion", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"Filter Threshold Selection for ROI: {roi}")
    fig_thresh.tight_layout()
    save_figure(fig_thresh, output_dir, "threshold_selection", roi, codebook)

    max_level = len(contours.levels) - 1
    level = questionary.text(
        f"For ROI '{roi}', please inspect 'threshold_selection--{roi}+{codebook}.png' in your output directory.\nEnter a threshold level (0-{max_level}):",
        validate=lambda val: val.isdigit()
        and 0 <= int(val) <= max_level
        or f"Please enter a valid integer between 0 and {max_level}.",
    ).ask()

    if level is None:  # User pressed Ctrl+C
        logger.warning("User cancelled input. Aborting process for this ROI.")
        return -1

    return int(level)


def _apply_final_filter(
    spots_: pl.DataFrame, interp_func: RegularGridInterpolator, contours: QuadContourSet, threshold_level: int
) -> pl.DataFrame:
    """Applies the final density-based filter to the spot data."""
    logger.info(f"Applying final filter at threshold level: {threshold_level}")
    final_threshold_value = contours.levels[threshold_level]
    point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())
    spots_ok = spots_.with_columns(point_density=point_densities).filter(
        pl.col("point_density") < final_threshold_value
    )

    logger.info(f"Final filtered spots: {len(spots_ok):,}")
    blank_prop = spots_ok.filter(pl.col("is_blank")).height / max(1, len(spots_ok))
    logger.info(f"Blank proportion in final set: {blank_prop:.2%}")
    return spots_ok


def _generate_final_outputs(
    spots_ok: pl.DataFrame, output_dir: Path, roi: str, codebook: str, params: SpotlookParams
):
    """Generates all final plots and saves the filtered data for a single ROI."""
    logger.info("Generating final plots and saving data...")
    # Final Spots Spatial Plot
    fig_spots, ax = plt.subplots(figsize=params.figsize_spots, dpi=params.dpi)
    subsample = max(1, len(spots_ok) // params.subsample)
    ax.scatter(spots_ok["y"][::subsample], spots_ok["x"][::subsample], s=0.1, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    add_scale_bar(ax, params.scale_bar_um / 0.108, f"{params.scale_bar_um} μm")
    ax.set_title(f"Filtered Spots for ROI {roi} (n={len(spots_ok):,})")
    save_figure(fig_spots, output_dir, "spots_final", roi, codebook)

    # Final Scree Plot
    per_gene_final = count_by_gene(spots_ok)
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
    logger.success(f"Saved {len(spots_ok):,} filtered spots for ROI {roi} to {output_parquet}")


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
@click.option("--distance-threshold", type=float, help="Override distance threshold")
@click.option("--seed", type=int, help="Override random seed")
def threshold(
    path: Path,
    codebook_path: Path,
    output_dir: Path | None = None,
    rois: list[str] | None = None,
    config: Path | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    norm_threshold: float | None = None,
    distance_threshold: float | None = None,
    seed: int | None = None,
):
    """
    Process spot data for each ROI individually with an interactive threshold selection step.
    """
    initialize_logger()

    if output_dir is None:
        output_dir = path.parent / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using output directory: {output_dir}")

    # Load configuration and create parameters model
    params = SpotlookParams.from_config(
        config_path=config,
        area_min=area_min,
        area_max=area_max,
        norm_threshold=norm_threshold,
        distance_threshold=distance_threshold,
        seed=seed,
    )
    logger.info(
        f"Using analysis parameters: seed={params.seed}, area=[{params.area_min}, {params.area_max}], norm_threshold={params.norm_threshold}"
    )

    codebook = codebook_path.stem

    ws = Workspace(path)
    rois_to_process = rois if rois else ws.rois
    if not rois_to_process:
        logger.error(f"No ROIs found or specified in workspace: {path}")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(rois_to_process)} ROIs to process individually: {rois_to_process}")

    # --- Main Processing Loop ---
    for i, roi in enumerate(rois_to_process, 1):
        logger.info(f"--- Starting processing for ROI: {roi} ({i}/{len(rois_to_process)}) ---")
        rng = np.random.default_rng(params.seed)

        spots_raw = _load_spots_data(path, roi, codebook)
        if spots_raw is None or spots_raw.is_empty():
            logger.warning(f"No data loaded for ROI {roi}. Skipping to next.")
            continue

        spots_intermediate = _apply_initial_filters(spots_raw, rng, params)
        if spots_intermediate.height < params.min_spots_for_density:
            logger.warning(
                f"Insufficient spots ({spots_intermediate.height}) for ROI {roi} after initial filtering. Skipping density analysis."
            )
            continue

        density_results = _calculate_density_map(spots_intermediate, params)
        if density_results is None:
            continue
        contours, interp_func = density_results

        chosen_level = _get_interactive_threshold(
            spots_intermediate, contours, interp_func, output_dir, roi, codebook, params
        )
        if chosen_level == -1:  # User cancelled
            continue

        spots_final = _apply_final_filter(spots_intermediate, interp_func, contours, chosen_level)
        _generate_final_outputs(spots_final, output_dir, roi, codebook, params)

        logger.success(f"--- Finished processing for ROI: {roi} ---")

    logger.success("All specified ROIs have been processed.")
