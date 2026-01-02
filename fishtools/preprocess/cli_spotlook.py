import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rich.console import Console
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, label

from fishtools.io.codebook import Codebook
from fishtools.io.workspace import Workspace, WorkspaceOutput
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


def _maybe_render_terminal_image(path: Path, *, title: str | None = None) -> None:
    if not sys.stdout.isatty():
        return
    if not path.exists():
        return

    try:
        from PIL import Image as PILImage
        from term_image.image import AutoImage, BlockImage
    except ImportError:
        logger.debug("term-image (or Pillow) not installed; skipping terminal image preview")
        return

    if title is not None:
        console.print(title, style="bold")

    try:
        BlockImage.set_render_method("direct")
    except (TypeError, ValueError):
        pass

    max_width = max(20, (min(console.size.width, 120) // 2))
    try:
        with PILImage.open(path) as img:
            click.echo(AutoImage(img, width=max_width))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning(f"Failed to render terminal image preview: {exc}")


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
    point_densities: np.ndarray | None = None


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
) -> Path:
    """Create a grid of downsampled spot plots across ROIs."""
    if not contexts:
        raise ValueError("No ROIs available to plot.")

    output_root = WorkspaceOutput(output_dir)
    spotlook_out = output_root.spotlook

    n_rois = len(contexts)
    grid = int(math.ceil(math.sqrt(n_rois)))
    n_cols = max(1, min(grid, 12))
    n_rows = int(math.ceil(n_rois / n_cols))

    fig_width = 4 * n_cols
    fig_height = 4 * n_rows

    logger.debug(
        f"CombSpots grid: rois={n_rois}, rows={n_rows}, cols={n_cols}, "
        f"size=({fig_width:.2f} in, {fig_height:.2f} in) @ {params.dpi} dpi "
        f"(~{int(fig_width * params.dpi)}x{int(fig_height * params.dpi)} px)"
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

    spots_final_dir = spotlook_out.spots_final_dir
    spots_final_dir.mkdir(parents=True, exist_ok=True)
    combined_path = spotlook_out.combined_spots_png(codebook).resolve()
    # Clamp DPI to avoid exceeding Agg backend limits (~65535 px on a side)
    save_dpi = render_dpi
    fig.savefig(combined_path.as_posix(), dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved combined spots overview: {combined_path}")
    return combined_path


def _estimate_fdr(blank_count: int, total_count: int, n_total_codes: int, n_blank_codes: int) -> float | None:
    if total_count <= 0 or n_blank_codes <= 0:
        return None
    scale_factor = n_total_codes / n_blank_codes
    return float(min((blank_count / total_count) * scale_factor, 1.0))


def _blank_proportion_for_fdr(fdr: float, n_total_codes: int, n_blank_codes: int) -> float | None:
    """Convert an FDR target into an equivalent blank proportion given codebook composition."""
    if n_total_codes <= 0 or n_blank_codes <= 0:
        return None
    blank_prop = fdr * (n_blank_codes / n_total_codes)
    return float(min(max(blank_prop, 0.0), 1.0))


def _write_spotlook_summary_json(
    output_dir: Path,
    *,
    contexts: dict[str, ROIThresholdContext],
    selected_levels: dict[str, int],
    codebook_name: str,
    n_total_codes: int,
    n_blank_codes: int,
    params: SpotThresholdParams,
) -> Path:
    spotlook_out = WorkspaceOutput(output_dir).spotlook
    spotlook_out.root.mkdir(parents=True, exist_ok=True)
    out_path = spotlook_out.root / "summary.json"

    params_for_summary = params.model_dump(
        include={
            "area_min",
            "area_max",
            "norm_threshold",
            "min_norm",
            "distance_threshold",
            "density_grid_size",
            "density_smooth_sigma",
            "min_spots_per_bin",
            "density_metric",
            "contour_mode",
            "contour_levels",
            "use_main_mass_mask",
            "seed",
        }
    )

    new_entries: dict[str, dict[str, object]] = {}
    for roi in sorted(contexts):
        ctx = contexts[roi]
        level = int(selected_levels[roi])
        threshold_value = float(ctx.contours.levels[level])  # type: ignore[index]

        total = int(ctx.spots.height)
        total_blank = int(ctx.spots.filter(pl.col("is_blank")).height)
        total_non_blank = total - total_blank

        filtered = ctx.spots_final
        if filtered is None:
            raise ValueError(f"ROI {roi} has no filtered spots; summary must be written after Phase 2.")
        filtered_total = int(filtered.height)
        filtered_blank = int(filtered.filter(pl.col("is_blank")).height)
        filtered_non_blank = filtered_total - filtered_blank
        filtered_blank_proportion = (filtered_blank / filtered_total) if filtered_total > 0 else 0.0

        entry = {
            "roi": roi,
            "codebook": codebook_name,
            "total_spots": total,
            "total_blank": total_blank,
            "total_non_blank": total_non_blank,
            "threshold_level": level,
            "threshold_value": threshold_value,
            "threshold_params": params_for_summary,
            "filtered_spots": filtered_total,
            "filtered_blank": filtered_blank,
            "filtered_blank_proportion": float(filtered_blank_proportion),
            "filtered_non_blank": filtered_non_blank,
            "fdr_estimated": _estimate_fdr(filtered_blank, filtered_total, n_total_codes, n_blank_codes),
        }
        new_entries[roi] = entry

    # Merge with existing file, replacing only ROIs updated in this run.
    summary: list[dict[str, object]] = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if not isinstance(existing, list):
            raise ValueError(f"Expected {out_path} to contain a JSON list, got {type(existing).__name__}.")
        for item in existing:
            if not isinstance(item, dict):
                continue
            roi = item.get("roi")
            if isinstance(roi, str) and roi in new_entries:
                summary.append(new_entries.pop(roi))
            else:
                summary.append(item)

    # Append any new ROIs not present in the previous file.
    for roi in sorted(new_entries):
        summary.append(new_entries[roi])

    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"Saved spotlook summary: {out_path}")
    return out_path


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


def _load_spots_data(path: Path, roi: str, codebook: Codebook, *, output_dir: Path | None = None) -> pl.DataFrame | None:
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
    ws = Workspace(path)
    decoded_path = ws.decoded_spots_parquet(roi, codebook.name)
    effective_output_dir = ws.output.root if output_dir is None else output_dir
    raw_copy_path = ws.threshold_parquet(roi, codebook.name, raw=True, output_dir=effective_output_dir)
    legacy_raw_copy_path = effective_output_dir / f"{roi}+{codebook.name}.raw.parquet"

    spots_path = decoded_path
    if not decoded_path.exists():
        if raw_copy_path.exists():
            spots_path = raw_copy_path
            logger.warning(f"Decoded spots parquet missing for ROI {roi}; using output copy: {raw_copy_path}")
        elif legacy_raw_copy_path.exists():
            spots_path = legacy_raw_copy_path
            logger.warning(
                f"Decoded spots parquet missing for ROI {roi}; using legacy output copy: {legacy_raw_copy_path}"
            )
        else:
            logger.warning(f"Spots file not found for ROI {roi}, skipping: {decoded_path}")
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

    spots_ = spots_.filter(pl.col("distance") < params.distance_threshold)
    logger.debug(f"Spots after distance filter: {len(spots_):,}")

    spots_ = spots_.with_columns(
        x_=(pl.col("area")) ** (1 / 3) + rng.uniform(-0.75, 0.75, size=len(spots_)),
        y_=(pl.col("norm") * (1 - pl.col("distance"))).log10(),
    )
    return spots_


def _compute_contour_levels(z_smooth: np.ndarray, mode: str, n_levels: int) -> np.ndarray:
    """Compute monotonic contour levels for a density map under different spacings.

    - linear: evenly spaced in value
    - log: evenly spaced in log10(value); requires positive min; uses a tiny floor if needed
    - sqrt: evenly spaced in sqrt(value) then squared back (more levels at high values)
    - square: evenly spaced in value^2 then sqrt back (more levels at low values)
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
    if mode == "square":
        vmin_clamped = max(vmin, 0.0)
        return np.sqrt(np.linspace(vmin_clamped**2, vmax**2, n_levels))
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


def _compute_main_mass_mask(
    count_grid: np.ndarray,
    hdr_percentile: float = 50.0,
) -> np.ndarray:
    """Compute mask keeping only the main mass of spots.

    1. Smooth count field to bridge small gaps
    2. HDR mask: bins above hdr_percentile of smoothed density
    3. Main mass: connected component containing the global maximum
    """
    # Smooth the count grid to bridge small gaps and find coherent regions
    count_smooth = gaussian_filter(count_grid.astype(float), sigma=1.5)

    # HDR mask: keep bins above percentile of smoothed density
    nonzero_counts = count_smooth[count_smooth > 0]
    if len(nonzero_counts) == 0:
        return np.ones_like(count_grid, dtype=bool)

    hdr_threshold = np.percentile(nonzero_counts, hdr_percentile)
    hdr_mask = count_smooth >= hdr_threshold

    if not hdr_mask.any():
        logger.warning("No bins passed HDR filter, using all non-zero bins")
        return count_grid > 0

    # Find connected components
    labeled, num_features = label(hdr_mask)
    if num_features == 0:
        return count_grid > 0

    # Find component containing global maximum of count_grid
    max_idx = np.unravel_index(np.argmax(count_grid * hdr_mask), count_grid.shape)
    main_label = labeled[max_idx]

    if main_label == 0:
        # Global max not in any component (shouldn't happen), fall back to largest
        component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
        main_label = np.argmax(component_sizes) + 1

    main_mass = labeled == main_label
    n_excluded = hdr_mask.sum() - main_mass.sum()
    if n_excluded > 0:
        logger.debug(f"Main-mass filter excluded {n_excluded} bins from {num_features} components")

    return main_mass


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

    # Build count grid and proportion grid
    count_grid = np.zeros((len(y_coords), len(x_coords)), dtype=float)
    Z = np.zeros_like(count_grid)
    metric = params.density_metric
    for row in binned_counts.iter_rows(named=True):
        count_grid[row["j"], row["i"]] = float(row["total_count"])
        value = float(row["blank_count"]) if metric == "count" else float(row["proportion"])
        Z[row["j"], row["i"]] = value

    # Pad toward lower sizes (x) and lower norm values (y) to avoid selecting dim spots
    # X padding: extend halfway toward 1.0
    x_padded = False
    if x_coords[0] > 1.0:
        x_pad = (x_coords[0] + 1.0) / 2
        x_coords = np.insert(x_coords, 0, x_pad)
        Z = np.insert(Z, 0, Z[:, 0], axis=1)
        count_grid = np.insert(count_grid, 0, 0, axis=1)
        x_padded = True
    # Y padding: extend downward by half the grid step
    y_step = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0.1
    y_pad = y_coords[0] - y_step / 2
    y_coords = np.insert(y_coords, 0, y_pad)
    Z = np.insert(Z, 0, Z[0, :], axis=0)
    count_grid = np.insert(count_grid, 0, 0, axis=0)

    # Optionally compute main-mass mask (HDR + connected component)
    if params.use_main_mass_mask:
        main_mass_mask = _compute_main_mass_mask(count_grid)
        # Dilate and fill holes in mask
        main_mass_mask = binary_dilation(main_mass_mask, iterations=1)
        main_mass_mask = binary_fill_holes(main_mass_mask)
        # Include padding regions in main mask only where blank proportion is non-zero
        main_mass_mask[0, :] |= Z[0, :] > 0  # Y padding row
        if x_padded:
            main_mass_mask[:, 0] |= Z[:, 0] > 0  # X padding column
        Z_masked = np.where(main_mass_mask, Z, 0.0)
        fill_value = np.inf  # spots outside mask will fail threshold
    else:
        # Legacy behavior: no masking
        main_mass_mask = np.ones_like(Z, dtype=bool)
        Z_masked = Z
        fill_value = 0.0

    Z_smooth = gaussian_filter(Z_masked, sigma=params.density_smooth_sigma) if params.density_smooth_sigma > 0 else Z_masked
    X, Y = np.meshgrid(x_coords, y_coords)

    # Build figure showing density heatmap with contours overlaid
    fig, ax = plt.subplots(figsize=params.figsize_thresh, dpi=params.dpi)
    if np.allclose(Z_smooth, Z_smooth.flat[0]):
        logger.warning("Histogram-based density map is constant; skipping contour generation.")
        raise RuntimeError("Constant density map. No spots or all spots identical?")

    if params.use_main_mass_mask:
        vmax = np.percentile(Z[Z > 0], 99.9) if (Z > 0).any() else 1e-6
    else:
        # Legacy: use Z_smooth for vmax
        vmax = np.percentile(Z_smooth, 99.9)
    if np.isclose(vmax, 0.0):
        vmax = 1e-6

    im = ax.pcolormesh(X, Y, Z, shading="auto", vmin=0, vmax=vmax)
    levels = _compute_contour_levels(Z_smooth, params.contour_mode, params.contour_levels)
    contours = ax.contour(X, Y, Z_smooth, levels=levels, colors="white", alpha=0.5, linewidths=0.5)

    # Show mask boundary if masking is enabled
    if params.use_main_mass_mask:
        ax.contour(X, Y, main_mass_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5, linestyles="--")

    ax.set_xlabel("Area")
    ax.set_ylabel("Norm * (1 - Distance) [log10]")
    title_metric = "Blank Count" if metric == "count" else "Blank Proportion"
    if params.use_main_mass_mask:
        title_suffix = " (cyan = main mass boundary)"
    else:
        title_suffix = " with Smoothed Contours"
    ax.set_title(f"{title_metric} Density{title_suffix}")
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
    interp_func = RegularGridInterpolator((y_coords, x_coords), Z_smooth, bounds_error=False, fill_value=fill_value)
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
    *,
    point_densities: np.ndarray | None = None,
) -> ThresholdCurve:
    """Compute threshold curve statistics for a single ROI."""
    threshold_levels = list(range(1, min(len(contours.levels), 30), 2))  # type: ignore
    spot_counts: list[int] = []
    blank_proportions: list[float] = []
    if point_densities is None:
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


def _counts_at_contour_level(
    spots: pl.DataFrame,
    contours: QuadContourSet,
    level_idx: int,
    *,
    point_densities: np.ndarray | None = None,
    interp_func: RegularGridInterpolator | None = None,
) -> tuple[int, float]:
    if point_densities is None:
        if interp_func is None:
            raise ValueError("Must provide either point_densities or interp_func.")
        point_densities = interp_func(spots.select(["y_", "x_"]).to_numpy())
    threshold_value = contours.levels[level_idx]  # type: ignore[index]
    mask = point_densities < threshold_value
    spot_count = int(mask.sum())
    if spot_count <= 0:
        return 0, 0.0
    blank_mask = mask & spots["is_blank"].to_numpy()
    blank_prop = float(blank_mask.sum() / spot_count)
    return spot_count, blank_prop


def _save_threshold_plot(
    curve: ThresholdCurve,
    output_dir: Path,
    roi: str,
    codebook: str,
    params: SpotThresholdParams,
    *,
    fdr_blank_proportion: float | None = None,
    selected_level: int | None = None,
    selected_spot_count: int | None = None,
) -> Path:
    """Persist the per-ROI threshold selection plot."""
    spotlook_out = WorkspaceOutput(output_dir).spotlook
    sns.set_theme()
    fig_thresh, (ax1, ax_diff) = plt.subplots(
        2, 1, figsize=(params.figsize_thresh[0], params.figsize_thresh[1] * 1.5), dpi=params.dpi
    )
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax1.plot(
        curve.levels,
        curve.spot_counts,
        color="g",
        linestyle="-",
        label="Remaining Spots",
    )
    if selected_level is not None and selected_spot_count is not None:
        ax1.plot(
            [selected_level],
            [selected_spot_count],
            marker="*",
            markersize=12,
            linestyle="None",
            color="g",
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Selected",
        )
    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Number of Spots", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    # Use SI-prefix formatting without spaces and without unnecessary trailing .0
    ax1.yaxis.set_major_formatter(si_tick_formatter())
    ax1.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax1.set_ylim(0, None)  # type: ignore[arg-type]

    ax2 = ax1.twinx()
    ax2.plot(
        curve.levels,
        curve.blank_proportions,
        color="r",
        linestyle="--",
        label="Blank Proportion",
    )
    if fdr_blank_proportion is not None:
        ax2.axhline(
            fdr_blank_proportion,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="1% FDR",
        )
    ax2.set_ylabel("Blank Proportion", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title(f"Filter Threshold Selection for ROI: {roi}")

    # Diff plot
    ax_diff.grid(True, axis="x")
    ax_diff.grid(False, axis="y")
    diff_levels = (np.array(curve.levels[:-1]) + np.array(curve.levels[1:])) / 2
    spot_diff = np.diff(curve.spot_counts)
    blank_diff = np.diff(curve.blank_proportions)
    ax_diff.plot(diff_levels, spot_diff, color="g", linestyle="-", label="Δ Spots")
    ax_diff.set_xlabel("Threshold Contour Level")
    ax_diff.set_ylabel("Δ Spots per Step", color="g")
    ax_diff.tick_params(axis="y", labelcolor="g")
    ax_diff.yaxis.set_major_formatter(si_tick_formatter())
    ax_diff.xaxis.set_major_locator(plt.MultipleLocator(2))

    ax_diff2 = ax_diff.twinx()
    blank_diff_abs = np.abs(blank_diff)
    max_blank_delta = float(blank_diff_abs.max()) if blank_diff_abs.size else 0.0
    ax_diff2.plot(diff_levels, blank_diff_abs, color="r", linestyle="--", label="Δ Blank Prop")
    ax_diff2.set_ylabel("Δ Blank Proportion", color="r")
    ax_diff2.tick_params(axis="y", labelcolor="r")
    ax_diff2.set_ylim(0.0, max_blank_delta if max_blank_delta > 0 else 1.0)

    ax_diff.set_title("Rate of Change per Step")

    fig_thresh.tight_layout()
    thresh_dir = spotlook_out.threshold_selection_dir
    save_figure(fig_thresh, thresh_dir, "threshold_selection", roi, codebook)
    return spotlook_out.threshold_selection_png(roi, codebook).resolve()


def _save_fdr_diagnostic_plots(
    spots: pl.DataFrame,
    contours: QuadContourSet,
    interp_func: RegularGridInterpolator,
    output_dir: Path,
    roi: str,
    codebook_name: str,
    _n_total_codes: int,  # Reserved for post-threshold FDR display
    _n_blank_codes: int,  # Reserved for post-threshold FDR display
    params: SpotThresholdParams,
) -> Path:
    """Generate PP-plot and histogram diagnostics for FDR validation.

    PP-plot: Target survival (y) vs blank survival (x) across contour levels.
    Points above y=x: targets survive more than blanks (filtering removes noise).
    Points on y=x: equal survival (blanks behave like random targets).

    Histogram: Direction-corrected norm (norm × (1-distance)) distributions.
    """
    spotlook_out = WorkspaceOutput(output_dir).spotlook
    fdr_dir = spotlook_out.root / "fdr_diagnostic"
    fdr_dir.mkdir(parents=True, exist_ok=True)

    point_densities = interp_func(spots.select(["y_", "x_"]).to_numpy())
    blanks_mask = spots["is_blank"].to_numpy()
    n_blanks_total = int(blanks_mask.sum())
    n_targets_total = int((~blanks_mask).sum())

    if n_blanks_total == 0 or n_targets_total == 0:
        logger.warning(
            f"ROI {roi}: insufficient data for FDR diagnostics (blanks={n_blanks_total}, targets={n_targets_total})"
        )
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=params.dpi)
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        out_path = fdr_dir / f"fdr_diagnostic--{roi}+{codebook_name}.png"
        fig.savefig(out_path, dpi=params.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    contour_levels = contours.levels  # type: ignore
    target_survival = []
    blank_survival = []

    for threshold in contour_levels:
        passes = point_densities < threshold
        target_survival.append(passes[~blanks_mask].sum() / n_targets_total)
        blank_survival.append(passes[blanks_mask].sum() / n_blanks_total)

    target_survival = np.array(target_survival)
    blank_survival = np.array(blank_survival)

    sns.set_theme()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=params.dpi)

    ax_pp = axes[0]
    ax_pp.plot(blank_survival, target_survival, "b.-", markersize=4, label="Observed")
    ax_pp.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y = x (equal survival)")
    ax_pp.set_xlabel("Blank Survival Rate")
    ax_pp.set_ylabel("Target Survival Rate")
    ax_pp.set_title(f"PP-plot: Survival Rates Across Stringency | ROI {roi}")
    ax_pp.legend(loc="lower right")
    ax_pp.set_xlim(0, 1)
    ax_pp.set_ylim(0, 1)

    surviving_blanks = blank_survival * n_blanks_total
    surviving_targets = target_survival * n_targets_total
    total_surviving = surviving_blanks + surviving_targets
    blank_pct = (
        np.divide(
            surviving_blanks,
            total_surviving,
            out=np.zeros_like(surviving_blanks),
            where=total_surviving > 0,
        )
        * 100
    )

    ax_pp2 = ax_pp.twiny()
    ax_pp2.set_xlim(ax_pp.get_xlim())
    valid_mask = blank_pct > 0
    if valid_mask.any():
        pct_min = blank_pct[valid_mask].min()
        pct_max = blank_pct[valid_mask].max()
        pct_lo = pct_min * 0.9
        pct_hi = pct_max * 1.1
        in_range = (blank_pct >= pct_lo) & (blank_pct <= pct_hi)
        valid_indices = np.where(in_range)[0]
        n_ticks = min(8, len(valid_indices))
        if n_ticks > 1:
            sampled = valid_indices[np.linspace(0, len(valid_indices) - 1, n_ticks, dtype=int)]
            tick_positions = [blank_survival[i] for i in sampled]
            tick_labels = [f"{blank_pct[i]:.2f}%" for i in sampled]
            ax_pp2.set_xticks(tick_positions)
            ax_pp2.set_xticklabels(tick_labels, fontsize=9)
    ax_pp2.tick_params(axis="x", width=0.5)
    ax_pp2.grid(False)  # Disable grid lines from secondary axis
    ax_pp2.set_xlabel("Blank % of Surviving Spots", fontsize=10)

    ax_hist = axes[1]
    blanks = spots.filter(pl.col("is_blank"))
    targets = spots.filter(~pl.col("is_blank"))
    # Direction-corrected norm: norm * (1 - distance), higher = better
    blank_corrected = (blanks["norm"] * (1 - blanks["distance"])).to_numpy()
    target_corrected = (targets["norm"] * (1 - targets["distance"])).to_numpy()
    # Filter to positive values for log scale
    blank_corrected = blank_corrected[blank_corrected > 0]
    target_corrected = target_corrected[target_corrected > 0]
    vmin = min(blank_corrected.min(), target_corrected.min())
    vmax = max(blank_corrected.max(), target_corrected.max())
    bins = np.geomspace(vmin, vmax, 150)

    # Compute histograms and normalize each to max=1 for visual comparison
    target_counts, _ = np.histogram(target_corrected, bins=bins)
    blank_counts, _ = np.histogram(blank_corrected, bins=bins)
    target_norm = target_counts / target_counts.max() if target_counts.max() > 0 else target_counts
    blank_norm = blank_counts / blank_counts.max() if blank_counts.max() > 0 else blank_counts

    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean for log scale
    ax_hist.fill_between(bin_centers, 0, target_norm, alpha=0.5, color="blue", label="Targets", step="mid", linewidth=0)
    ax_hist.fill_between(bin_centers, 0, blank_norm, alpha=0.7, color="red", label="Blanks", step="mid", linewidth=0)
    ax_hist.set_xscale("log")

    ax_hist.set_xlabel("Norm × (1 - Distance)")
    ax_hist.set_ylabel("Normalized Density")
    ax_hist.set_title("Direction-Corrected Norm Distributions (normalized)")
    ax_hist.legend(loc="upper right")

    # Histogram (bottom): absolute counts with log y-scale
    ax_counts = axes[2]
    ax_counts.fill_between(bin_centers, 0.5, target_counts + 0.5, alpha=0.5, color="blue", label="Targets", step="mid", linewidth=0)
    ax_counts.fill_between(bin_centers, 0.5, blank_counts + 0.5, alpha=0.7, color="red", label="Blanks", step="mid", linewidth=0)
    ax_counts.set_xscale("log")
    ax_counts.set_yscale("log")
    ax_counts.set_xlabel("Norm × (1 - Distance)")
    ax_counts.set_ylabel("Count")
    ax_counts.set_title("Direction-Corrected Norm Distributions (absolute counts)")
    ax_counts.legend(loc="upper right")

    fig.tight_layout()
    out_path = fdr_dir / f"fdr_diagnostic--{roi}+{codebook_name}.png"
    fig.savefig(out_path, dpi=params.dpi, bbox_inches="tight")
    fig.savefig(fdr_dir / f"fdr_diagnostic--{roi}+{codebook_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved FDR diagnostic: {out_path}")
    return out_path


def _save_combined_threshold_plot(
    curves_by_roi: dict[str, ThresholdCurve],
    output_dir: Path,
    codebook: str,
    params: SpotThresholdParams,
    *,
    fdr_blank_proportion: float | None = None,
    selected_levels: dict[str, int] | None = None,
    selected_spot_counts: dict[str, int] | None = None,
) -> Path:
    """Create an overlay threshold-selection plot for all ROIs."""
    if not curves_by_roi:
        raise ValueError("No threshold curves available to plot.")

    spotlook_out = WorkspaceOutput(output_dir).spotlook
    ordered_rois = sorted(curves_by_roi)
    palette = sns.color_palette("husl", len(ordered_rois))

    sns.set_theme()
    fig, (ax1, ax_diff) = plt.subplots(2, 1, figsize=(params.figsize_thresh[0], params.figsize_thresh[1] * 1.5), dpi=params.dpi)
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax2 = ax1.twinx()

    ax_diff.grid(True, axis="x")
    ax_diff.grid(False, axis="y")
    ax_diff2 = ax_diff.twinx()
    roi_handles: list[Line2D] = []
    max_blank_delta_all = 0.0
    for color, roi in zip(palette, ordered_rois):
        curve = curves_by_roi[roi]
        max_count = max(curve.spot_counts) if curve.spot_counts else 0
        spot_counts_norm = [
            (count / max_count) if max_count > 0 else 0.0
            for count in curve.spot_counts
        ]
        ax1.plot(curve.levels, spot_counts_norm, color=color, linestyle="-")
        ax2.plot(curve.levels, curve.blank_proportions, color=color, linestyle="--")
        roi_handles.append(Line2D([0], [0], color=color, linewidth=2, label=roi))
        if selected_levels is not None and selected_spot_counts is not None:
            selected_level = selected_levels.get(roi)
            selected_count = selected_spot_counts.get(roi)
            if selected_level is not None and selected_count is not None:
                selected_norm = (selected_count / max_count) if max_count > 0 else 0.0
                ax1.plot(
                    [selected_level],
                    [selected_norm],
                    marker="*",
                    markersize=12,
                    linestyle="None",
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                )
        # Plot diff per level step
        diff_levels = (np.array(curve.levels[:-1]) + np.array(curve.levels[1:])) / 2
        spot_diff = np.diff(spot_counts_norm)
        blank_diff = np.diff(curve.blank_proportions)
        blank_diff_abs = np.abs(blank_diff)
        max_blank_delta = float(blank_diff_abs.max()) if blank_diff_abs.size else 0.0
        ax_diff.plot(diff_levels, spot_diff, color=color, linestyle="-")
        ax_diff2.plot(diff_levels, blank_diff_abs, color=color, linestyle="--")
        if max_blank_delta > max_blank_delta_all:
            max_blank_delta_all = max_blank_delta

    if fdr_blank_proportion is not None:
        ax2.axhline(
            fdr_blank_proportion,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="1% FDR",
        )

    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=2, label="Remaining Spots (normalized)"),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Blank Proportion",
        ),
    ]
    if fdr_blank_proportion is not None:
        style_handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="1% FDR"))
    if selected_levels is not None:
        style_handles.append(
            Line2D([0], [0], color="gray", linestyle="None", marker="*", markersize=10, label="Selected")
        )

    ax1.set_xlabel("Threshold Contour Level")
    ax1.set_ylabel("Remaining Spots (normalized)")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Blank Proportion")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"Threshold Selection Curves | Codebook {codebook}")

    legend1 = ax1.legend(handles=roi_handles, title="ROI", loc="upper right")
    ax1.legend(handles=style_handles, title=None, loc="lower right")
    ax1.add_artist(legend1)

    # Diff plot
    ax_diff.set_xlabel("Threshold Contour Level")
    ax_diff.set_ylabel("Δ Spots per Step (normalized)")
    ax_diff.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax_diff2.set_ylabel("Δ Blank Proportion")
    ax_diff2.set_ylim(0.0, max_blank_delta_all if max_blank_delta_all > 0 else 1.0)
    ax_diff.set_title("Rate of Change per Step")

    fig.tight_layout()
    thresh_dir = spotlook_out.threshold_selection_dir
    thresh_dir.mkdir(parents=True, exist_ok=True)
    combined_path = spotlook_out.combined_threshold_png(codebook).resolve()
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

    combined_plot_path = WorkspaceOutput(output_dir).spotlook.combined_threshold_png(codebook)
    lines = ["Generated artifacts:"]
    for roi in ordered_rois:
        artifacts = contexts[roi].artifact_paths
        contour = artifacts.get("contours")
        scatter = artifacts.get("spots_contours")
        threshold = artifacts.get("threshold")
        fdr_diag = artifacts.get("fdr_diagnostic")
        lines.append(f"  {roi}:")
        if scatter:
            lines.append(f"    Spots:     {scatter}")
        lines.append(f"    Contours:  {contour}")
        lines.append(f"    Threshold: {threshold}")
        if fdr_diag:
            lines.append(f"    FDR:       {fdr_diag}")

    lines.append("")
    lines.append(f"Combined plot: {combined_plot_path.resolve()}")
    if combined_spots_path is not None:
        lines.append(f"Combined spots: {combined_spots_path}")
    lines.append("")
    max_level = max(contexts[roi].curve.max_level for roi in ordered_rois)
    lines.append(f"Enter threshold levels for each ROI (comma-separated) between 0-{max_level}")

    lines.append("")
    lines.append("Please enter the threshold levels now (comma-separated integers):")
    lines.append("  - Or enter 'blank=<prop>' (e.g. blank=0.001) to auto-select for all ROIs.")

    # NOTE: We avoid questionary/prompt_toolkit here because it may switch the TTY
    # into alternate-screen mode, which prevents long artifact paths from wrapping.
    click.echo("\n".join(lines))
    _maybe_render_terminal_image(combined_plot_path.resolve(), title="Combined threshold plot:")

    while True:
        try:
            response = click.prompt("Threshold levels", type=str)
        except click.Abort:
            logger.warning("KeyboardInterrupt detected during threshold input.")
            raise KeyboardInterrupt

        try:
            return _parse_threshold_levels_response(response, ordered_rois, contexts)
        except ValueError as exc:
            click.secho(str(exc), fg="red", err=True)


def _parse_threshold_levels_response(
    response: str,
    ordered_rois: list[str],
    contexts: dict[str, ROIThresholdContext],
) -> dict[str, int]:
    raw = response.strip()
    lower = raw.lower()

    # Convenience: if the user enters a single decimal like "0.001" (no commas),
    # interpret it as a blank proportion (same as "blank=0.001").
    if "," not in raw and raw.startswith("0."):
        value_raw = raw
        if value_raw.endswith("%"):
            raise ValueError("Percent inputs are not supported. Use a proportion like 0.001.")
        try:
            value = float(value_raw)
        except ValueError as exc:
            raise ValueError("Blank proportion must be a number (e.g. 0.001).") from exc
        target = value
        if not 0.0 <= target <= 1.0:
            raise ValueError("Blank proportion must be between 0 and 1.")
        if target > 0.001:
            click.secho(
                f"Warning: blank proportion {target:g} (>0.001). If you meant a percent, convert it to a proportion.",
                fg="yellow",
                err=True,
            )
            if not click.confirm("Proceed with this blank proportion?", default=False):
                raise ValueError("Cancelled; please re-enter threshold levels.")
        return _select_threshold_levels_by_blank_proportion(ordered_rois, contexts, target)

    for prefix in ("blank=", "blank:", "blank "):
        if lower.startswith(prefix):
            value_raw = raw[len(prefix) :].strip()
            if not value_raw:
                raise ValueError("Blank proportion value is required after 'blank='.")
            if value_raw.endswith("%"):
                raise ValueError("Percent inputs are not supported. Use a proportion like blank=0.001.")
            try:
                value = float(value_raw)
            except ValueError as exc:
                raise ValueError("Blank proportion must be a number (e.g. 0.001).") from exc
            target = value
            if not 0.0 <= target <= 1.0:
                raise ValueError("Blank proportion must be between 0 and 1.")
            if target > 0.001:
                click.secho(
                    f"Warning: blank proportion {target:g} (>0.001). "
                    "If you meant a percent, convert it to a proportion (e.g. 0.75% -> 0.0075).",
                    fg="yellow",
                    err=True,
                )
                if not click.confirm("Proceed with this blank proportion?", default=False):
                    raise ValueError("Cancelled; please re-enter threshold levels.")
            return _select_threshold_levels_by_blank_proportion(ordered_rois, contexts, target)

    values = [part.strip() for part in response.split(",")]
    if len(values) != len(ordered_rois):
        raise ValueError(f"Expected {len(ordered_rois)} comma-separated values.")

    parsed: dict[str, int] = {}
    for roi, raw_level in zip(ordered_rois, values):
        if raw_level == "":
            raise ValueError(f"ROI {roi}: level is required.")
        try:
            level = int(raw_level)
        except ValueError as exc:
            raise ValueError(f"ROI {roi}: level must be an integer.") from exc
        max_level = contexts[roi].curve.max_level
        if not 0 <= level <= max_level:
            raise ValueError(f"ROI {roi}: level must be between 0 and {max_level}.")
        parsed[roi] = level
    return parsed


def _select_threshold_levels_by_blank_proportion(
    ordered_rois: list[str],
    contexts: dict[str, ROIThresholdContext],
    target_blank_proportion: float,
) -> dict[str, int]:
    """Select contour levels per ROI whose blank proportion best matches a target."""
    if not 0.0 <= target_blank_proportion <= 1.0:
        raise ValueError("target_blank_proportion must be between 0 and 1.")

    selected: dict[str, int] = {}
    for roi in ordered_rois:
        curve = contexts[roi].curve
        if not curve.levels:
            raise ValueError(f"ROI {roi}: no threshold curve levels available.")

        best_span: float | None = None
        best_level: float | None = None
        for idx in range(len(curve.levels) - 1):
            p0 = float(curve.blank_proportions[idx])
            p1 = float(curve.blank_proportions[idx + 1])
            lo = min(p0, p1)
            hi = max(p0, p1)
            if not (lo <= target_blank_proportion <= hi):
                continue
            if np.isclose(p0, p1):
                candidate_level = float(curve.levels[idx])
                span = 0.0
            else:
                t = (target_blank_proportion - p0) / (p1 - p0)
                level0 = float(curve.levels[idx])
                level1 = float(curve.levels[idx + 1])
                candidate_level = level0 + t * (level1 - level0)
                span = abs(p1 - p0)

            if best_span is None or span < best_span:
                best_span = span
                best_level = candidate_level

        if best_level is None:
            diffs = np.abs(np.asarray(curve.blank_proportions, dtype=float) - target_blank_proportion)
            best_idx = int(np.argmin(diffs))
            best_level = float(curve.levels[best_idx])

        level_float = float(np.clip(best_level, 0.0, float(curve.max_level)))
        # We only support integer contour indices. Use "round half up" so midpoints
        # (e.g. 11.5) round to the next integer consistently.
        selected[roi] = int(math.floor(level_float + 0.5))
    return selected


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
    final_threshold_value = contours.levels[threshold_level]  # type: ignore[index]
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
    *,
    metadata: dict[str, object] | None = None,
):
    """Generates all final plots and saves the filtered data for a single ROI."""
    output_root = WorkspaceOutput(output_dir)
    spotlook_out = output_root.spotlook
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
    save_figure(fig_spots, spotlook_out.spots_final_dir, "spots_final", roi, codebook, log_level="INFO")

    # Save blank counts and scree plot together
    scree_dir = spotlook_out.scree_final_dir
    scree_dir.mkdir(parents=True, exist_ok=True)

    per_gene_final = count_by_gene(spots_ok)
    per_gene_final.filter(pl.col("is_blank")).sort("count", descending=True).write_csv(
        scree_dir / f"blanks--{roi}+{codebook}.csv"
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
    save_figure(fig_scree, scree_dir, "scree_final", roi, codebook, log_level="INFO")

    # Save final filtered data with metadata
    parquets_dir = output_root.parquets
    parquets_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = parquets_dir / f"{roi}+{codebook}.parquet"

    df_out = spots_ok.drop("point_density", "x_", "y_")
    if metadata is not None:
        import pyarrow.parquet as pq

        table = df_out.to_arrow()
        # Encode metadata as JSON strings in the schema metadata
        metadata_encoded = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()}
        table = table.replace_schema_metadata({**metadata_encoded, **(table.schema.metadata or {})})
        pq.write_table(table, output_parquet)
    else:
        df_out.write_parquet(output_parquet)
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
    help="Directory to save outputs. [default: '<workspace>/analysis/output']",
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
    type=click.Choice(["linear", "log", "sqrt", "square"], case_sensitive=False),
    help="Spacing mode for density contours",
)
@click.option("--contour-levels", type=int, help="Number of contour levels")
@click.option(
    "--fixed-blank-proportion",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Auto-select threshold levels per ROI to match this blank proportion (skips prompt).",
)
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
    fixed_blank_proportion: float | None = None,
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

    ws = Workspace(path)
    if output_dir is None:
        output_dir = ws.output.root
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
        "Using analysis parameters: "
        f"seed={params.seed}, "
        f"area=[{params.area_min:.3f}, {params.area_max:.3f}], "
        f"norm_threshold={params.norm_threshold:.4f}, "
        f"min_norm={params.min_norm}"
    )

    codebook = Codebook(codebook_path)
    n_total_codes, n_blank_codes = codebook.blank_stats()
    logger.debug(f"Codebook stats: {n_blank_codes}/{n_total_codes} blanks ({n_blank_codes/n_total_codes:.1%})")
    fdr_blank_proportion = _blank_proportion_for_fdr(0.01, n_total_codes, n_blank_codes)

    output_root = WorkspaceOutput(output_dir)
    spotlook_out = output_root.spotlook
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
        raise click.ClickException(f"No ROIs found or specified in workspace: {path}")

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
        spots_raw = _load_spots_data(path, roi, codebook, output_dir=output_dir)
        decoded_spots_path = ws.decoded_spots_parquet(roi, codebook.name)
        if spots_raw is not None and decoded_spots_path.exists():
            raw_parquet_path = ws.threshold_parquet(roi, codebook.name, raw=True, output_dir=output_dir)
            raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(decoded_spots_path, raw_parquet_path)

        if spots_raw is None or spots_raw.is_empty():
            logger.warning(f"No data loaded for ROI {roi}. Skipping to next.")
            skipped_rois.append(roi)
            continue

        logger.debug(f"Applying initial filters to {len(spots_raw)} raw spots")
        spots_intermediate = _apply_initial_filters(spots_raw, rng, params)
        if spots_intermediate.is_empty():
            logger.warning(f"No spots remained after initial filters for ROI {roi}. Skipping.")
            skipped_rois.append(roi)
            continue

        logger.debug(f"Computing density map on {len(spots_intermediate)} filtered spots")
        density_results = _calculate_density_map(spots_intermediate, params)

        fig_contours, contours, interp_func, surface = density_results
        point_densities = interp_func(spots_intermediate.select(["y_", "x_"]).to_numpy())
        contours_dir = spotlook_out.contours_dir
        save_figure(fig_contours, contours_dir, "contours", roi, codebook.name)
        contour_path = spotlook_out.contours_png(roi, codebook.name).resolve()

        fig_blank_panels = _create_spots_contours_figure(
            spots_intermediate,
            roi,
            codebook.name,
            params,
            surface,
        )
        spots_contours_path: Path | None = None
        if fig_blank_panels is not None:
            spots_contours_dir = spotlook_out.spots_contours_dir
            save_figure(fig_blank_panels, spots_contours_dir, "spots_contours", roi, codebook.name)
            spots_contours_path = spotlook_out.spots_contours_png(roi, codebook.name).resolve()

        logger.debug("Computing threshold curve statistics")
        curve = _compute_threshold_curve(spots_intermediate, contours, interp_func, point_densities=point_densities)
        threshold_plot_path = _save_threshold_plot(
            curve,
            output_dir,
            roi,
            codebook.name,
            params,
            fdr_blank_proportion=fdr_blank_proportion,
        )

        # FDR validation diagnostics
        fdr_diagnostic_path = _save_fdr_diagnostic_plots(
            spots_intermediate,
            contours,
            interp_func,
            output_dir,
            roi,
            codebook.name,
            n_total_codes,
            n_blank_codes,
            params,
        )

        contexts[roi] = ROIThresholdContext(
            spots=spots_intermediate,
            contours=contours,
            interpolator=interp_func,
            curve=curve,
            artifact_paths={
                "contours": contour_path,
                "spots_contours": spots_contours_path,
                "threshold": threshold_plot_path,
                "fdr_diagnostic": fdr_diagnostic_path,
            },
            point_densities=point_densities,
        )

    if skipped_rois:
        logger.warning(f"Skipped {len(skipped_rois)} ROI(s) due to missing data: {skipped_rois}")

    if not contexts:
        raise click.ClickException("No ROIs produced threshold curves.")

    curves_for_plot = {roi: ctx.curve for roi, ctx in contexts.items()}
    logger.debug(f"Saving combined threshold selection plot for {len(curves_for_plot)} ROIs")
    combined_plot_path = _save_combined_threshold_plot(
        curves_for_plot,
        output_dir,
        codebook.name,
        params,
        fdr_blank_proportion=fdr_blank_proportion,
    )
    logger.debug(f"Saved combined threshold plot: {combined_plot_path}")

    active_rois = sorted(contexts)
    if fixed_blank_proportion is not None:
        logger.info(
            f"Selecting threshold levels automatically using fixed_blank_proportion={fixed_blank_proportion:.2%}"
        )
        selected_levels = _select_threshold_levels_by_blank_proportion(
            active_rois,
            contexts,
            fixed_blank_proportion,
        )
    else:
        selected_levels = _prompt_threshold_levels(active_rois, contexts, output_dir, codebook.name)

    # Overwrite threshold selection plots with markers at the user-selected levels.
    for roi in active_rois:
        ctx = contexts[roi]
        chosen_level = selected_levels[roi]
        spot_count, _blank_prop = _counts_at_contour_level(
            ctx.spots,
            ctx.contours,
            chosen_level,
            point_densities=ctx.point_densities,
            interp_func=ctx.interpolator,
        )
        _save_threshold_plot(
            ctx.curve,
            output_dir,
            roi,
            codebook.name,
            params,
            fdr_blank_proportion=fdr_blank_proportion,
            selected_level=chosen_level,
            selected_spot_count=spot_count,
        )

    # --- Phase 2: Apply thresholds and finalize outputs ---
    for i, roi in enumerate(active_rois, 1):
        chosen_level = selected_levels[roi]
        logger.info(
            f"Starting post-threshold processing for ROI {roi} ({i}/{len(active_rois)}) at level {chosen_level}"
        )
        ctx = contexts[roi]

        logger.debug(f"Applying final density-based filter at level {chosen_level} for ROI {roi}...")
        logger.debug(f"Interpolating densities for {len(ctx.spots)} candidate spots")
        ctx.spots_final = _apply_final_filter(ctx.spots, ctx.interpolator, ctx.contours, chosen_level)

        logger.debug("Generating plots and saving results...")

        # Build metadata matching summary.json structure
        total = int(ctx.spots.height)
        total_blank = int(ctx.spots.filter(pl.col("is_blank")).height)
        filtered_total = int(ctx.spots_final.height)
        filtered_blank = int(ctx.spots_final.filter(pl.col("is_blank")).height)
        threshold_value = float(ctx.contours.levels[int(chosen_level)])  # type: ignore[index]

        parquet_metadata = {
            "roi": roi,
            "codebook": codebook.name,
            "total_spots": total,
            "total_blank": total_blank,
            "total_non_blank": total - total_blank,
            "threshold_level": chosen_level,
            "threshold_value": threshold_value,
            "filtered_spots": filtered_total,
            "filtered_blank": filtered_blank,
            "filtered_blank_proportion": filtered_blank / filtered_total if filtered_total > 0 else 0.0,
            "filtered_non_blank": filtered_total - filtered_blank,
            "fdr_estimated": _estimate_fdr(filtered_blank, filtered_total, n_total_codes, n_blank_codes),
        }

        _generate_final_outputs(ctx.spots_final, output_dir, roi, codebook.name, params, metadata=parquet_metadata)

        logger.debug(f"Finished ROI {roi} ({i}/{len(active_rois)})")

    combined_spots_all = _save_combined_spots_plot(contexts, output_dir, codebook.name, params)

    logger.info(f"Saved combined spots overview: {combined_spots_all.absolute()}")
    _write_spotlook_summary_json(
        output_dir,
        contexts=contexts,
        selected_levels=selected_levels,
        codebook_name=codebook.name,
        n_total_codes=n_total_codes,
        n_blank_codes=n_blank_codes,
        params=params,
    )

    logger.debug("All specified ROIs have been processed.")
