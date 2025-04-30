#!/usr/bin/env python3
"""
Processes segmented image slices to identify regions (e.g., cells) and
assigns detected spots (e.g., RNA molecules) to these regions.

Refactored for improved testability and maintainability.
"""

import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import rich_click as click
import zarr
from loguru import logger
from scipy.ndimage import binary_erosion, binary_fill_holes
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.strtree import STRtree
from skimage import measure
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties

from fishtools.analysis.labelimage import (
    isotropic_label_closing,
    isotropic_label_dilation,
    isotropic_label_opening,
)
from fishtools.preprocess.tileconfig import TileConfiguration

# --- Configuration ---
CONTOUR_PAD = 1
MIN_CONTOUR_POINTS = 6
DOWNSAMPLE_FACTOR = 2
Z_FILTER_TOLERANCE = 0.5

# --- Data Loading and Preparation Functions ---


def load_segmentation_slice(segmentation_zarr_path: Path, idx: int) -> np.ndarray:
    """Loads a specific Z-slice from the segmentation Zarr store."""
    logger.info(f"Slice {idx}: Loading segmentation image from {segmentation_zarr_path}...")
    try:
        img_stack = zarr.open_array(str(segmentation_zarr_path), mode="r")
        if idx >= img_stack.shape[0]:
            raise IndexError(f"Index {idx} out of bounds for Zarr array (shape: {img_stack.shape}).")
        img = img_stack[idx]
        logger.info(f"Slice {idx}: Loaded segmentation slice with shape {img.shape}.")
        return img
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to load segmentation Zarr slice: {e}")
        raise  # Re-raise after logging


def apply_morphological_smoothing(
    img: np.ndarray,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    idx: int,  # For logging context
) -> np.ndarray:
    """Applies morphological smoothing operations to the segmentation mask."""
    logger.info(
        f"Slice {idx}: Applying morphological smoothing (open={opening_radius}, close={closing_radius}, dilate={dilation_radius})."
    )
    try:
        opened_mask = isotropic_label_opening(img, radius=opening_radius)
        closed_mask = isotropic_label_closing(opened_mask, radius=closing_radius)
        if dilation_radius > 0:
            smoothed_img = isotropic_label_dilation(closed_mask, radius=dilation_radius)
        else:
            smoothed_img = closed_mask
        logger.info(f"Slice {idx}: Smoothing complete.")
        return smoothed_img
    except Exception as e:
        logger.error(f"Slice {idx}: Error during smoothing: {e}. Returning original image.")
        return img


def calculate_coordinate_offsets(
    tile_config_path: Path, downsample_factor: int, idx: int
) -> tuple[float, float]:
    """Calculates coordinate offsets from TileConfiguration."""
    logger.info(f"Slice {idx}: Loading TileConfiguration from {tile_config_path}...")
    tc = TileConfiguration.from_file(tile_config_path).downsample(downsample_factor)
    coords = tc.df
    x_offset = coords["x"].min()
    y_offset = coords["y"].min()
    logger.info(f"Slice {idx}: Calculated offsets: x={x_offset:.2f}, y={y_offset:.2f}")
    return x_offset, y_offset


def load_and_prepare_spots(
    spots_parquet_path: Path,
    idx: int,
    z_filter_tolerance: float,
    downsample_factor: int,
    x_offset: float,
    y_offset: float,
) -> pl.DataFrame:
    """Loads spots, filters by Z, adjusts coordinates, and adds unique IDs."""
    logger.info(f"Slice {idx}: Loading spots from {spots_parquet_path}...")
    all_spots = pl.read_parquet(spots_parquet_path)

    if "index" not in all_spots.columns:
        all_spots = all_spots.with_row_index()

    logger.info(
        f"Slice {idx}: Filtering spots for Z = {idx} +/- {z_filter_tolerance} and applying offsets..."
    )
    spots = (
        all_spots.filter(
            pl.col("z").is_between(idx - z_filter_tolerance, idx + z_filter_tolerance, closed="both")
        )
        .with_columns(
            x_adj=pl.col("x") / downsample_factor - x_offset,
            y_adj=pl.col("y") / downsample_factor - y_offset,
            spot_id=pl.col("index"),
        )
        .select(["spot_id", "x_adj", "y_adj", "z", "target"])
    )

    n_spots = len(spots)
    logger.info(f"Slice {idx}: Found {n_spots} spots in the Z range.")
    if not n_spots:
        logger.warning(
            f"Slice {idx}: No spots found for Z range {idx - z_filter_tolerance:.1f} to {idx + z_filter_tolerance:.1f}."
        )
        # Return empty dataframe with correct schema
        return pl.DataFrame(
            # schema={
            #     "spot_id": pl.Int64,
            #     "x_adj": pl.Float64,
            #     "y_adj": pl.Float64,
            #     "z": pl.Float64,
            # }
        )

    return spots


# --- Polygon Extraction Functions ---


def _extract_polygon_from_region(
    region: RegionProperties,
    polygon_index: int,  # The index in the final list
    image_shape: tuple[int, int],
    idx: int,  # For logging context
) -> tuple[Polygon | MultiPolygon, dict[str, Any]]:
    """Extracts Shapely polygon(s) and metadata from a single skimage region."""

    region_meta = {
        "polygon_id": polygon_index,
        "label": region.label,
        "area": region.area,
        "centroid_y": region.centroid[0],
        "centroid_x": region.centroid[1],
    }
    min_row, min_col, max_row, max_col = region.bbox
    region_image = region.image

    # Basic checks
    if region.area == 0 or region_image.shape[0] < 2 or region_image.shape[1] < 2:
        return Polygon(), region_meta  # Return empty polygon

    # Pad, fill holes, erode slightly
    padded_mask = np.pad(region_image, pad_width=CONTOUR_PAD, mode="constant", constant_values=0)
    filled_mask = binary_fill_holes(padded_mask)
    eroded_mask = binary_erosion(filled_mask, iterations=1)

    # Find contours
    contours = measure.find_contours(eroded_mask, 0.5)

    region_polygons = []
    for contour in contours:
        if len(contour) < MIN_CONTOUR_POINTS:
            continue

        # Shift contour coordinates to absolute image coordinates
        contour[:, 0] = contour[:, 0] + min_row - CONTOUR_PAD
        contour[:, 1] = contour[:, 1] + min_col - CONTOUR_PAD

        # Clip coordinates
        contour[:, 0] = np.clip(contour[:, 0], 0, image_shape[0] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, image_shape[1] - 1)

        # Create polygon if valid
        if len(np.unique(contour, axis=0)) >= 3:
            shapely_contour = contour[:, ::-1]  # Swap to (x, y)
            try:
                poly = Polygon(shapely_contour)
                if poly.is_valid:
                    region_polygons.append(poly)
                else:
                    buffered_poly = poly.buffer(0)  # Attempt to fix
                    if buffered_poly.is_valid and isinstance(buffered_poly, Polygon):  # type: ignore
                        region_polygons.append(buffered_poly)
            except Exception as e:
                logger.debug(
                    f"Slice {idx}, Polygon {polygon_index} (Label {region.label}): Error creating polygon: {e}"
                )

    # Consolidate results
    if not region_polygons:
        return Polygon(), region_meta
    elif len(region_polygons) == 1:
        return region_polygons[0], region_meta
    else:
        try:
            multi_poly = MultiPolygon(region_polygons)
            if multi_poly.is_valid:
                return multi_poly, region_meta
            else:
                buffered_multi = multi_poly.buffer(0)  # Attempt to fix
                if buffered_multi.is_valid and isinstance(buffered_multi, (Polygon, MultiPolygon)):  # type: ignore
                    return buffered_multi, region_meta
                else:
                    return Polygon(), region_meta  # Fallback
        except Exception as e:
            logger.debug(
                f"Slice {idx}, Polygon {polygon_index} (Label {region.label}): Error creating MultiPolygon: {e}"
            )
            return Polygon(), region_meta  # Fallback


def extract_polygons_from_mask(
    mask: np.ndarray, idx: int
) -> list[tuple[Polygon | MultiPolygon, dict[str, Any]]]:
    """Finds regions in a mask and extracts Shapely polygons for each."""
    logger.info(f"Slice {idx}: Finding regions in the segmentation mask...")
    props = regionprops(mask)
    n_regions = len(props)
    logger.info(f"Slice {idx}: Found {n_regions} regions.")

    if not n_regions:
        logger.warning(f"Slice {idx}: No regions found in the segmentation image.")
        return []

    polygons_with_meta = []
    total_region_area = 0
    processed_regions = 0

    logger.info(f"Slice {idx}: Extracting polygons from {n_regions} regions...")
    for i, region in enumerate(props):
        if i > 0 and i % 5000 == 0:
            logger.info(f"Slice {idx}: Processed {i}/{n_regions} regions for polygon extraction...")

        poly, meta = _extract_polygon_from_region(region, i, mask.shape, idx)
        polygons_with_meta.append((poly, meta))
        total_region_area += meta["area"]
        processed_regions += 1

    if processed_regions > 0:
        logger.info(
            f"Slice {idx}: Extracted {len(polygons_with_meta)} polygon entries. Mean region area: {total_region_area / processed_regions:.2f} px²"
        )
    else:
        logger.info(f"Slice {idx}: No valid regions processed for polygon extraction.")

    return polygons_with_meta


# --- Spatial Indexing and Assignment Functions ---


def build_spatial_index(
    polygons_with_meta: list[tuple[Polygon | MultiPolygon, dict[str, Any]]], idx: int
) -> tuple[STRtree | None, list[int]]:
    """Builds an STRtree from the valid polygons."""
    valid_polygons = [(i, poly) for i, (poly, _) in enumerate(polygons_with_meta) if not poly.is_empty]

    if not valid_polygons:
        logger.warning(f"Slice {idx}: No valid polygons found to build spatial index.")
        return None, []

    tree_indices, tree_geoms = zip(*valid_polygons)
    logger.info(f"Slice {idx}: Building STRtree with {len(tree_geoms)} valid geometries...")
    tree = STRtree(tree_geoms)
    logger.info(f"Slice {idx}: STRtree built.")
    return tree, list(tree_indices)  # Return tree and the original indices corresponding to tree geometries


def assign_spots_to_polygons(
    spots_df: pl.DataFrame,
    tree: STRtree | None,
    tree_indices: list[int],  # Maps tree geometry index to original polygons_with_meta index
    polygons_with_meta: list[tuple[Polygon | MultiPolygon, dict[str, Any]]],
    idx: int,
) -> pl.DataFrame:
    """Assigns spots to polygons using the spatial index."""
    if tree is None or not tree_indices or spots_df.is_empty():
        logger.warning(
            f"Slice {idx}: Cannot assign spots (no tree, polygons, or spots). Returning empty assignments."
        )
        return pl.DataFrame()
    spots_df = spots_df.with_row_index("local_id")

    n_spots = len(spots_df)
    logger.info(f"Slice {idx}: Creating Shapely points for {n_spots} spots...")
    points = [Point(xy) for xy in zip(spots_df["x_adj"], spots_df["y_adj"])]

    logger.info(f"Slice {idx}: Querying STRtree to assign spots to polygons...")
    # query returns [point_idx, tree_idx]
    query_result = tree.query(points, predicate="intersects").T

    logger.info(f"Slice {idx}: Found {query_result.shape[0]} potential intersections. Refining...")

    assignments_list = []
    for i, (point_idx, tree_geom_idx) in enumerate(query_result):
        if i % 50000 == 0:
            logger.info(f"Slice {idx}: Refined {i}/{query_result.shape[0]} intersections...")

        # Map tree geom index back to original list index
        original_polygon_list_idx = tree_indices[tree_geom_idx]
        # polygon_geom = polygons_with_meta[original_polygon_list_idx][0]
        # point_geom = points[point_idx]
        # # Precise check
        # if polygon_geom.contains(point_geom):
        spot = spots_df[int(point_idx)]
        polygon_meta = polygons_with_meta[int(original_polygon_list_idx)][1]
        assignments_list.append({
            "spot_id": spot["spot_id"].item(),
            "target": spot["target"].item(),
            "label": polygon_meta["label"],
        })

    logger.info(f"Slice {idx}: Found {len(assignments_list)} confirmed spot assignments.")
    if not assignments_list:
        return pl.DataFrame()
    else:
        return pl.from_dicts(assignments_list)


# --- Saving and Plotting Functions ---


def save_results(
    assignments_df: pl.DataFrame,
    polygons_with_meta: list[tuple[Polygon | MultiPolygon, dict[str, Any]]],
    ident_path: Path,
    polygons_path: Path,
    idx: int,
):
    """Saves the spot assignments and polygon metadata to Parquet files."""
    logger.info(f"Slice {idx}: Saving spot assignments ({len(assignments_df)} rows) to {ident_path}...")
    try:
        assignments_df.write_parquet(ident_path)
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to write spot assignments parquet: {e}")
        raise

    logger.info(
        f"Slice {idx}: Saving polygon metadata ({len(polygons_with_meta)} entries) to {polygons_path}..."
    )
    try:
        poly_meta_list = [p[1] for p in polygons_with_meta] if polygons_with_meta else []
        if not poly_meta_list:
            df_polygons = pl.DataFrame(
                schema={
                    "polygon_id": pl.UInt32,
                    "label": pl.UInt32,
                    "area": pl.Float64,
                    "centroid_y": pl.Float64,
                    "centroid_x": pl.Float64,
                }
            )
        else:
            df_polygons = pl.from_dicts(
                poly_meta_list,
                schema={
                    "polygon_id": pl.UInt32,
                    "label": pl.UInt32,
                    "area": pl.Float64,
                    "centroid_y": pl.Float64,
                    "centroid_x": pl.Float64,
                },
            )
        df_polygons.write_parquet(polygons_path)
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to write polygon metadata parquet: {e}")
        raise


def generate_debug_plot(
    img: np.ndarray,
    spots_df: pl.DataFrame,
    segmentation_zarr_path: Path,  # Used to find input_image.zarr
    output_dir: Path,
    idx: int,
):
    """Generates and saves a debug plot showing intensity, mask, and spots."""
    logger.info(f"Slice {idx}: Generating debug plots...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()

        # Attempt to load corresponding intensity image
        intensity = None
        intensity_zarr_path = segmentation_zarr_path.parent / "input_image.zarr"
        if intensity_zarr_path.exists():
            try:
                intensity_stack = zarr.open_array(str(intensity_zarr_path), mode="r")
                if idx < len(intensity_stack):
                    intensity_slice = intensity_stack[idx]
                    # Handle potential channel dimension (assuming C is first if present)
                    intensity = intensity_slice[0] if intensity_slice.ndim == 3 else intensity_slice
            except Exception as e:
                logger.warning(f"Slice {idx}: Could not load intensity image for debug plot: {e}")

        # Define plot region (e.g., center crop)
        img_h, img_w = img.shape
        sl = np.s_[img_h // 4 : img_h * 3 // 4, img_w // 4 : img_w * 3 // 4]
        # sl = np.s_[:,:] # Or full slice

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 6), dpi=150)

        # Plot 1: Intensity
        if intensity is not None:
            vmin = np.percentile(intensity[sl][intensity[sl] > 0], 1) if np.any(intensity[sl] > 0) else 0
            vmax = np.percentile(intensity[sl], 99)
            axs[0].imshow(intensity[sl], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            axs[0].text(0.5, 0.5, "Intensity N/A", ha="center", va="center", transform=axs[0].transAxes)
        axs[0].set_title(f"Slice {idx}: Intensity (Input)")

        # Plot 2: Segmentation Mask
        cmap = plt.cm.get_cmap("tab20", np.max(img) + 1)
        cmap.set_under(color="black")
        axs[1].imshow(img[sl], origin="lower", cmap=cmap, interpolation="none", vmin=1)
        axs[1].set_title(f"Slice {idx}: Segmentation Mask (Used)")

        # Plot 3: Spots on Mask
        axs[2].imshow(img[sl], origin="lower", cmap=cmap, interpolation="none", vmin=1, alpha=0.6)
        if not spots_df.is_empty():
            # Need utility function from original code
            # Assuming filter_spots_for_imshow exists globally or is imported
            try:
                from __main__ import filter_spots_for_imshow  # Hacky way if running as script
            except ImportError:
                # If filter_spots_for_imshow is defined elsewhere, import it properly
                # from your_utility_module import filter_spots_for_imshow
                logger.warning("filter_spots_for_imshow function not found for debug plot.")
                plot_spots_x, plot_spots_y = [], []  # Avoid error
            else:
                plot_spots_x, plot_spots_y = filter_spots_for_imshow(spots_df, sl, columns=("x_adj", "y_adj"))

            axs[2].scatter(plot_spots_x, plot_spots_y, s=1, alpha=0.7, c="red")
        axs[2].set_title(f"Slice {idx}: Spots on Segmentation")

        # Final plot adjustments
        for ax in axs:
            ax.axis("off")
            xlim = (0, (sl[1].stop or img_w) - (sl[1].start or 0))
            ylim = (0, (sl[0].stop or img_h) - (sl[0].start or 0))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")

        plt.tight_layout()
        plot_path = output_dir / f"debug_plot_{idx}.png"
        fig.savefig(plot_path)
        logger.info(f"Slice {idx}: Saved debug plot to {plot_path}")
        plt.close(fig)

    except ImportError:
        logger.warning(f"Slice {idx}: Matplotlib or Seaborn not installed. Cannot generate debug plots.")
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to generate debug plot: {e}")


# --- Orchestration Function ---


def process_slice(
    idx: int,
    segmentation_zarr_path: Path,
    spots_parquet_path: Path,
    tile_config_path: Path | None,
    output_dir: Path,  # This should be the 'chunks' directory
    apply_smoothing: bool,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    overwrite: bool,
    debug: bool,
) -> None:
    """
    Orchestrates the processing steps for a single Z-slice.
    """
    ident_path = output_dir / f"ident_{idx}.parquet"
    polygons_path = output_dir / f"polygons_{idx}.parquet"

    if not overwrite and ident_path.exists() and polygons_path.exists():
        logger.info(f"Slice {idx}: Skipping, output files already exist ({ident_path}, {polygons_path}).")
        return

    try:
        # 1. Load Segmentation
        img = load_segmentation_slice(segmentation_zarr_path, idx)

        # 2. Apply Smoothing (optional)
        if apply_smoothing:
            img = apply_morphological_smoothing(img, opening_radius, closing_radius, dilation_radius, idx)

        # 3. Load and Prepare Spots
        x_offset, y_offset = 0.0, 0.0
        if tile_config_path:
            x_offset, y_offset = calculate_coordinate_offsets(tile_config_path, DOWNSAMPLE_FACTOR, idx)

        spots_df = load_and_prepare_spots(
            spots_parquet_path, idx, Z_FILTER_TOLERANCE, DOWNSAMPLE_FACTOR, x_offset, y_offset
        )

        # Handle case with no spots early
        if spots_df.is_empty():
            logger.warning(f"Slice {idx}: No spots loaded or filtered. Writing empty outputs.")
            save_results(spots_df, [], ident_path, polygons_path, idx)
            return

        # 4. Extract Polygons
        polygons_with_meta = extract_polygons_from_mask(img, idx)

        # Handle case with no polygons
        if not polygons_with_meta:
            logger.warning(f"Slice {idx}: No polygons extracted. Writing empty assignments.")
            assignments_df = pl.DataFrame(
                # schema={"spot_id": pl.Int64, "polygon_id": pl.Int64, "label": pl.UInt32}
            )
            save_results(assignments_df, polygons_with_meta, ident_path, polygons_path, idx)
            return

        # 5. Build Spatial Index
        tree, tree_indices = build_spatial_index(polygons_with_meta, idx)

        # 6. Assign Spots
        assignments_df = assign_spots_to_polygons(spots_df, tree, tree_indices, polygons_with_meta, idx)

        # 7. Save Results
        save_results(assignments_df, polygons_with_meta, ident_path, polygons_path, idx)

        # 8. Debug Plot (optional)
        if debug:
            # Pass the potentially smoothed image to the plot function
            generate_debug_plot(img, spots_df, segmentation_zarr_path, output_dir, idx)

        logger.info(f"Slice {idx}: Processing finished successfully.")

    except Exception as e:
        logger.error(f"Slice {idx}: Failed during processing pipeline: {e}")
        # Decide if partial results should be cleaned up or left
        # For simplicity, we don't clean up here, but in production, you might want to.
        raise e  # Re-raise to indicate failure at the main level


# --- Utility Functions (Keep or move to a separate utils file) ---
# Need to make filter_spots_for_imshow available if debug plot is used


def filter_spots_by_bounds(
    spots: pl.DataFrame,
    lim: tuple[tuple[float | None, float | None], tuple[float | None, float | None]] | tuple[slice, slice],
    x_col: str = "x",
    y_col: str = "y",
) -> pl.DataFrame:
    """Filter spots DataFrame within spatial bounds."""
    match lim:
        case ((x_min, x_max), (y_min, y_max)):
            return spots.filter(
                (pl.col(x_col) >= x_min if x_min is not None else True)
                & (pl.col(y_col) >= y_min if y_min is not None else True)
                & (pl.col(x_col) < x_max if x_max is not None else True)
                & (pl.col(y_col) < y_max if y_max is not None else True)
            )
        case (sl_y, sl_x):
            # Ensure start/stop are handled correctly if None
            x_start = sl_x.start if sl_x.start is not None else -float("inf")
            x_stop = sl_x.stop if sl_x.stop is not None else float("inf")
            y_start = sl_y.start if sl_y.start is not None else -float("inf")
            y_stop = sl_y.stop if sl_y.stop is not None else float("inf")
            return spots.filter(
                (pl.col(x_col) >= x_start)
                & (pl.col(y_col) >= y_start)
                & (pl.col(x_col) < x_stop)
                & (pl.col(y_col) < y_stop)
            )
        case _:
            raise ValueError(
                f"Invalid limit format: {lim}. Use ((xmin, xmax), (ymin, ymax)) or np.s_[ymin:ymax, xmin:xmax]."
            )


def filter_spots_for_imshow(
    spots: pl.DataFrame,
    lim: tuple[tuple[float | None, float | None], tuple[float | None, float | None]] | tuple[slice, slice],
    columns: tuple[str, str] = ("x", "y"),
) -> list[pl.Series]:
    """Filter spots and adjust coordinates relative to the bounds' origin for plotting."""
    filtered = filter_spots_by_bounds(spots, lim, x_col=columns[0], y_col=columns[1])
    if filtered.is_empty():
        return [pl.Series(values=[], dtype=pl.Float64), pl.Series(values=[], dtype=pl.Float64)]

    x_col, y_col = columns
    match lim:
        case ((x_min, _), (y_min, _)):
            x_offset = x_min or 0
            y_offset = y_min or 0
        case (sl_y, sl_x):
            x_offset = sl_x.start or 0
            y_offset = sl_y.start or 0
        case _:
            raise ValueError(f"Invalid limit format: {lim}.")

    filtered = filtered.with_columns(**{x_col: pl.col(x_col) - x_offset}, **{y_col: pl.col(y_col) - y_offset})
    # Ensure output is always list of Series, even if empty
    return [filtered.get_column(c) for c in columns]


# --- CLI Definition ---


# --- Define Common Click Options ---
# Store the decorator functions themselves in a list
common_options = [
    click.argument(
        "input-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
        required=True,
    ),
    click.option(
        "--output-dir",
        type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
        required=True,  # Make required if no sensible default can be derived
        help="Directory to save the output Parquet files ('chunks' subdirectory will be created/used).",
    ),
    click.option(
        "--segmentation-name",
        type=str,
        default="output_segmentation.zarr",
        show_default=True,
        help="Relative path to the segmentation Zarr store within the input directory.",
    ),
    click.option(
        "--spots-name",
        type=str,
        # Note: Original code had different defaults. Choose one or make required.
        default="../mousecommon--brain+mousecommon.parquet",
        show_default=True,
        help="Relative path to the spots Parquet file within the input directory.",
    ),
    click.option(
        "--tileconfig-name",
        type=str,
        default="../stitch--brain/TileConfiguration.registered.txt",
        show_default=True,
        help="Relative path to the TileConfiguration file within input dir (optional). Set to '' to disable.",
    ),
    click.option(
        "--opening-radius",
        type=float,
        default=4.0,
        show_default=True,
        help="Radius for morphological opening.",
    ),
    click.option(
        "--closing-radius",
        type=float,
        default=6.0,
        show_default=True,
        help="Radius for morphological closing.",
    ),
    click.option(
        "--dilation-radius",
        type=float,
        default=2.0,
        show_default=True,
        help="Radius for final dilation (0 to disable).",
    ),
    click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files."),
    click.option(
        "--debug", is_flag=True, default=False, help="Enable debug logging and generate debug plots."
    ),
]


# --- Helper Function to Apply Decorators ---
def add_options(options):
    """
    Decorator factory that applies a list of click options/arguments.
    """

    def _add_options(func):
        for option in reversed(options):  # Apply decorators bottom-up
            func = option(func)
        return func

    return _add_options


def run_(
    input_dir: Path,
    output_dir: Path,
    segmentation_name: str,
    spots: Path,
    tile_config_path: Path,
    idx: int,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    overwrite: bool,
    debug: bool,
):
    """
    Main function to process a specific Z-slice of segmented image data
    and assign detected spots to segmented regions. Refactored for clarity.
    """

    # --- Prepare Paths and Directories ---
    segmentation_path = input_dir / segmentation_name
    spots_path = spots
    output_chunk_dir = output_dir / f"chunks+{spots_path.stem.split('+')[1]}"

    try:
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_chunk_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_chunk_dir}: {e}")
        sys.exit(1)

    # --- Input Validation ---
    if not segmentation_path.exists():
        logger.error(f"Segmentation Zarr store not found: {segmentation_path}")
        sys.exit(1)
    if not spots_path.exists():
        logger.error(f"Spots Parquet file not found: {spots_path}")
        sys.exit(1)

    # --- Log Configuration ---
    logger.info(f"--- Configuration ---")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_chunk_dir}")
    logger.info(f"Segmentation: {segmentation_path}")
    logger.info(f"Spots: {spots_path}")
    logger.info(f"TileConfig: {tile_config_path if tile_config_path else 'Not used'}")
    logger.info(f"Overwrite: {overwrite}")

    # --- Execute Processing Pipeline ---
    try:
        process_slice(
            idx=idx,
            segmentation_zarr_path=segmentation_path,
            spots_parquet_path=spots_path,
            tile_config_path=tile_config_path,
            output_dir=output_chunk_dir,  # Pass the specific chunk dir
            apply_smoothing=False,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            dilation_radius=dilation_radius,
            overwrite=overwrite,
            debug=debug,
        )
        logger.info(f"Successfully finished processing slice {idx}.")
    except Exception as e:
        # The error should have been logged within process_slice or its sub-functions
        logger.critical(f"Pipeline execution failed for slice {idx}.")
        raise e


@click.group()
def cli(): ...


@cli.command()
@click.argument(
    "input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Directory to save the output Parquet files ('chunks' subdirectory will be created/used).",
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation.zarr",
    show_default=True,
    help="Relative path to the segmentation Zarr store within the input directory.",
)
@click.option(
    "--spots-name",
    type=str,
    default="../mousecommon--brain+mousecommon.parquet",
    show_default=True,
    help="Relative path to the spots Parquet file within the input directory.",
)
@click.option("-i", "--idx", type=int, required=True, help="The Z-slice index to process.")
@click.option(
    "--opening-radius", type=float, default=4.0, show_default=True, help="Radius for morphological opening."
)
@click.option(
    "--closing-radius", type=float, default=6.0, show_default=True, help="Radius for morphological closing."
)
@click.option(
    "--dilation-radius",
    type=float,
    default=2.0,
    show_default=True,
    help="Radius for final dilation (0 to disable).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging and generate debug plots.")
def run(
    input_dir: Path,
    output_dir: Path,
    segmentation_name: str,
    spots_name: str,
    idx: int,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    overwrite: bool,
    debug: bool,
):
    roi, codebook = spots.stem.split("+")
    run_(
        input_dir,
        output_dir,
        segmentation_name,
        spots_name,
        tile_config_path,
        idx,
        opening_radius,
        closing_radius,
        dilation_radius,
        overwrite,
        debug,
    )


def initialize():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@cli.command()
@click.argument(
    "input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Directory to save the output Parquet files ('chunks' subdirectory will be created/used).",
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation.zarr",
    show_default=True,
    help="Relative path to the segmentation Zarr store within the input directory.",
)
@click.option(
    "--spots",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    show_default=True,
    help="Relative path to the spots Parquet file within the input directory.",
)
@click.option(
    "--opening-radius", type=float, default=4.0, show_default=True, help="Radius for morphological opening."
)
@click.option(
    "--closing-radius", type=float, default=6.0, show_default=True, help="Radius for morphological closing."
)
@click.option(
    "--dilation-radius",
    type=float,
    default=2.0,
    show_default=True,
    help="Radius for final dilation (0 to disable).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging and generate debug plots.")
def batch(
    input_dir: Path,
    output_dir: Path,
    segmentation_name: str,
    spots: Path,
    opening_radius: float,
    closing_radius: float,
    dilation_radius: float,
    overwrite: bool,
    debug: bool,
):
    with ProcessPoolExecutor(
        max_workers=8,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = []
        roi, codebook = spots.stem.split("+")
        z = zarr.open_array(input_dir / segmentation_name, mode="r")
        for i in range(z.shape[0]):
            futures.append(
                executor.submit(
                    run_,
                    input_dir,
                    output_dir,
                    segmentation_name,
                    spots,
                    spots.parent / f"stitch--{roi}/TileConfiguration.registered.txt",
                    i,
                    opening_radius,
                    closing_radius,
                    dilation_radius,
                    overwrite,
                    debug,
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing slice: {e}")


if __name__ == "__main__":
    cli()
