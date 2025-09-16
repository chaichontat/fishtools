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
from rich.console import Console
from rich.text import Text
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from fishtools.preprocess.config import SpotThresholdParams
from fishtools.utils.io import Codebook, Workspace
from fishtools.utils.plot import add_scale_bar
from fishtools.utils.utils import initialize_logger

console = Console()


def build_spotlook_params(
    config_path: Path | None = None,
    *,
    area_min: float | None = None,
    area_max: float | None = None,
    norm_threshold: float | None = None,
    min_norm: float | None = None,
    distance_threshold: float | None = None,
    seed: int | None = None,
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
        }.items()
        if v is not None
    }
    return params.model_copy(update=overrides) if overrides else params


# --- Core Helper Functions ---


def create_shimmer_text(text: str, style: str = "bold cyan") -> Text:
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
    logger.info(f"Saved plot: {filename}")
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


# --- Pipeline Stage Functions ---


def _load_spots_data(path: Path, roi: str, codebook: Codebook) -> pl.DataFrame | None:
    """Loads spot data from a parquet file for a single ROI."""
    spots_path = path / f"registered--{roi}+{codebook.name}" / f"decoded-{codebook.name}" / "spots.parquet"
    if not spots_path.exists():
        logger.warning(f"Spots file not found for ROI {roi}, skipping: {spots_path}")
        return None

    logger.info(f"Loading spots for ROI {roi} from {spots_path.name}")
    df = (
        pl.read_parquet(spots_path)
        .with_columns(is_blank=pl.col("target").str.starts_with("Blank"), roi=pl.lit(roi))
        .join(codebook.to_dataframe(), on="target", how="left")
    )
    mtime = datetime.fromtimestamp(spots_path.stat().st_mtime)
    logger.info(f"-> Found {len(df):,} spots. Data timestamp: {mtime:%Y-%m-%d %H:%M:%S}")
    return df


def _apply_initial_filters(
    spots: pl.DataFrame,
    rng: np.random.Generator,
    params: SpotThresholdParams,
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
    params: SpotThresholdParams,
) -> tuple[Figure, QuadContourSet, RegularGridInterpolator, np.ndarray, np.ndarray, np.ndarray]:
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
    contours = ax.contour(X, Y, Z_smooth, levels=50, colors="white", alpha=0.5, linewidths=0.5)
    ax.set_xlabel("Area")
    ax.set_ylabel("Norm * (1 - Distance) [log10]")
    title_metric = "Blank Count" if metric == "count" else "Blank Proportion"
    ax.set_title(f"{title_metric} Density with Smoothed Contours")
    fig.colorbar(im, ax=ax, label=title_metric)

    if not contours.levels.size:  # type: ignore[attr-defined]
        raise RuntimeError("Could not generate density contours. The data may be too sparse or uniform.")

    interp_func = RegularGridInterpolator((y_coords, x_coords), Z_smooth, bounds_error=False, fill_value=0)
    return fig, contours, interp_func, x_coords, y_coords, Z_smooth


def _plot_blank_vs_nonblank_panels(
    spots_: pl.DataFrame,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_smooth: np.ndarray,
    contour_levels: np.ndarray,
) -> None:
    """Render side-by-side scatter panels for blank and non-blank spots."""
    if spots_.is_empty():
        logger.warning(f"No spots available to plot for ROI {roi}; skipping blank/non-blank panels.")
        return

    non_blank = spots_.filter(~pl.col("is_blank"))
    blank = spots_.filter(pl.col("is_blank"))

    if non_blank.is_empty() and blank.is_empty():
        logger.warning(
            f"ROI {roi} contains no classified blank or non-blank spots after initial filtering; skipping scatter panels."
        )
        return

    # Dark theme for high-contrast titles on pure black background
    style_rc = {
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
    with sns.axes_style("dark", rc=style_rc), sns.plotting_context(rc={"axes.titlesize": 14}):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(params.figsize_thresh[0] * 2, params.figsize_thresh[1]),
            dpi=params.dpi,
            sharey=True,
        )

        X, Y = np.meshgrid(x_coords, y_coords)

        # After contour index 15, only include every 5th level to reduce clutter
        if contour_levels.size:
            idx = np.arange(len(contour_levels))
            mask = (idx <= 10) | ((idx > 10) & (((idx - 10) % 5) == 0))
            filtered_levels = contour_levels[mask]
        else:
            filtered_levels = contour_levels

        if non_blank.is_empty():
            axes[0].text(0.5, 0.5, "No non-blank spots", ha="center", va="center")
        else:
            axes[0].hexbin(
                non_blank["x_"].to_numpy(),
                non_blank["y_"].to_numpy(),
                gridsize=250,
                cmap="inferno",
                mincnt=5,
                linewidths=0,
            )
        contour_non_blank = axes[0].contour(
            X, Y, z_smooth, levels=filtered_levels, colors="white", linewidths=0.4, alpha=0.6
        )
        _label_contour_levels(contour_non_blank)
        axes[0].set_xlabel("Area (cube-root, jittered)")
        axes[0].set_ylabel("log10(norm * (1 - distance))")
        axes[0].set_title("Non-blank spots", color=style_rc["axes.titlecolor"])

        if blank.is_empty():
            axes[1].text(0.5, 0.5, "No blank spots", ha="center", va="center")
        else:
            axes[1].hexbin(
                blank["x_"].to_numpy(),
                blank["y_"].to_numpy(),
                gridsize=250,
                cmap="inferno",
                mincnt=5,
                linewidths=0,
            )
        contour_blank = axes[1].contour(
            X, Y, z_smooth, levels=filtered_levels, colors="white", linewidths=0.4, alpha=0.6
        )
        _label_contour_levels(contour_blank)
        axes[1].set_xlabel("Area (cube-root, jittered)")
        axes[1].set_ylabel("log10(norm * (1 - distance))")
        axes[1].set_title("Blank spots", color=style_rc["axes.titlecolor"])

        x_min = min(float(spots_["x_"].min()), x_coords[0])
        x_max = max(float(spots_["x_"].max()), x_coords[-1])
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_coords[0], y_coords[-1])
            # No grid, spine on left and bottom only
            ax.grid(False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("#9ca3af")
            ax.tick_params(bottom=True, left=True, top=False, right=False, colors="#e5e7eb")

        fig.tight_layout()
        # End seaborn context

    save_figure(fig, output_dir, "spots_contours", roi, codebook)


def _get_interactive_threshold(
    spots_: pl.DataFrame,
    contours: QuadContourSet,
    interp_func: RegularGridInterpolator,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
) -> int:
    """Generates a threshold plot and interactively asks the user for a threshold level."""
    with console.status(
        create_shimmer_text(f"🧮 Generating threshold selection plot for ROI {roi}...", "magenta"),
        spinner="dots4",
    ):
        threshold_levels = list(range(1, min(len(contours.levels), 15), 2))  # type: ignore
        spot_counts, blank_proportions = [], []
        for level_idx in threshold_levels:
            threshold_value = contours.levels[level_idx]  # type: ignore
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
        ax1.set_ylim(0, None)  # type: ignore
        ax2 = ax1.twinx()
        ax2.plot(threshold_levels, blank_proportions, "r-", label="Blank Proportion")
        ax2.set_ylabel("Blank Proportion", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax1.set_title(f"Filter Threshold Selection for ROI: {roi}")
        fig_thresh.tight_layout()
        save_figure(fig_thresh, output_dir, "threshold_selection", roi, codebook)

        max_level = len(contours.levels) - 1  # type: ignore

    # Show the absolute path to the threshold plot for clarity
    threshold_png = (output_dir / f"threshold_selection--{roi}+{codebook}.png").resolve()
    level = questionary.text(
        f"""Generated:
  Spots:     {output_dir / f"spots_contours--{roi}+{codebook}.png"}
  Contours:  {output_dir / f"contours--{roi}+{codebook}.png"}
  Threshold: {threshold_png}

Enter a threshold level (0-{max_level}):""",
        validate=lambda val: val.isdigit()
        and 0 <= int(val) <= max_level
        or f"Please enter a valid integer between 0 and {max_level}.",
    ).ask()

    # Always exit on KeyboardInterrupt/Ctrl+C instead of continuing to next ROI
    # questionary returns None on Ctrl+C; treat as an immediate, global abort.
    if level is None:
        logger.warning("KeyboardInterrupt detected during threshold input. Exiting immediately.")
        raise typer.Exit(code=1)

    return int(level)


def _apply_final_filter(
    spots_: pl.DataFrame, interp_func: RegularGridInterpolator, contours: QuadContourSet, threshold_level: int
) -> pl.DataFrame:
    """Applies the final density-based filter to the spot data."""
    logger.info(f"Applying final filter at threshold level: {threshold_level}")
    final_threshold_value = contours.levels[threshold_level]  # type: ignore
    point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())
    spots_ok = spots_.with_columns(point_density=point_densities).filter(
        pl.col("point_density") < final_threshold_value
    )

    logger.info(f"Final filtered spots: {len(spots_ok):,}")
    blank_prop = spots_ok.filter(pl.col("is_blank")).height / max(1, len(spots_ok))
    logger.info(f"Blank proportion in final set: {blank_prop:.2%}")
    return spots_ok


def _generate_final_outputs(
    spots_ok: pl.DataFrame, output_dir: Path, roi: str, codebook: str, params: SpotThresholdParams
):
    """Generates all final plots and saves the filtered data for a single ROI."""
    logger.info("Generating final plots and saving data...")
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
@click.option("--min-norm", type=float, help="Minimum norm filter applied before density analysis")
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
    min_norm: float | None = None,
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
    params = build_spotlook_params(
        config_path=config,
        area_min=area_min,
        area_max=area_max,
        norm_threshold=norm_threshold,
        min_norm=min_norm,
        distance_threshold=distance_threshold,
        seed=seed,
    )
    logger.info(
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

    logger.info(f"Found {len(rois_to_process)} ROIs to process individually: {rois_to_process}")

    # --- Main Processing Loop ---
    for i, roi in enumerate(rois_to_process, 1):
        logger.info(f"--- Starting processing for ROI: {roi} ({i}/{len(rois_to_process)}) ---")
        rng = np.random.default_rng(params.seed)

        with console.status(
            create_shimmer_text(f"🧮 Analyzing spot density patterns for ROI {roi}...", "magenta"),
            spinner="dots4",
        ) as status:
            spots_raw = _load_spots_data(path, roi, codebook)
            if spots_raw is None or spots_raw.is_empty():
                logger.warning(f"No data loaded for ROI {roi}. Skipping to next.")
                continue

            spots_intermediate = _apply_initial_filters(spots_raw, rng, params)
            density_results = _calculate_density_map(spots_intermediate, params)

        fig_contours, contours, interp_func, x_coords, y_coords, z_smooth = density_results

        save_figure(fig_contours, output_dir, "contours", roi, codebook.name)
        _plot_blank_vs_nonblank_panels(
            spots_intermediate,
            output_dir,
            roi,
            codebook.name,
            params,
            x_coords,
            y_coords,
            z_smooth,
            np.asarray(contours.levels),  # type: ignore
        )

        if params.min_norm:
            spots_intermediate = spots_intermediate.filter(pl.col("norm") >= params.min_norm)
            logger.info(f"Spots after minimum norm filter ({params.min_norm}): {len(spots_intermediate):,}")

        chosen_level = _get_interactive_threshold(
            spots_intermediate, contours, interp_func, output_dir, roi, codebook.name, params
        )
        if chosen_level == -1:  # User cancelled
            continue

        with console.status(
            create_shimmer_text(
                f"✨ Processing threshold level {chosen_level} for ROI {roi}...", "bold green"
            ),
            spinner="dots4",
        ) as status:
            status.update("🔄 Applying final density-based filter...")
            spots_final = _apply_final_filter(spots_intermediate, interp_func, contours, chosen_level)

            status.update("📊 Generating plots and saving results...")
            _generate_final_outputs(spots_final, output_dir, roi, codebook.name, params)

        # Show completion message with visual flair
        console.print(create_shimmer_text(f"Successfully processed ROI {roi}!", "bright_green"))
        logger.success(f"--- Finished processing for ROI: {roi} ---")

    logger.success("All specified ROIs have been processed.")
