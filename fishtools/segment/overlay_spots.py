#!/usr/bin/env python3
"""
Consolidated implementation for "segment overlay spots".

This module is moved from fishtools.preprocess.spots.overlay_spots to make
Segmentation the canonical home. The former path remains as a thin wrapper
to preserve backwards compatibility.
"""

import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import rich_click as click
from loguru import logger

from fishtools.io.workspace import Workspace
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.segment.utils import compute_regionprops_table
from fishtools.utils.logging import setup_cli_logging

# --- Configuration ---
DOWNSAMPLE_FACTOR = 2
Z_FILTER_TOLERANCE = 0.5


def load_segmentation_slice(segmentation_zarr_path: Path, idx: int) -> np.ndarray:
    """Loads a specific Z-slice from the segmentation Zarr store."""
    logger.info(f"Slice {idx}: Loading segmentation image from {segmentation_zarr_path}...")
    import zarr

    try:
        img_stack = zarr.open_array(str(segmentation_zarr_path), mode="r")
        if idx >= img_stack.shape[0]:
            raise IndexError(f"Index {idx} out of bounds for Zarr array (shape: {img_stack.shape}).")
        img = img_stack[idx]
        logger.info(f"Slice {idx}: Loaded segmentation slice with shape {img.shape}.")
        return img
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to load segmentation Zarr slice: {e}")
        raise


def calculate_coordinate_offsets(tile_config: TileConfiguration, downsample_factor: int) -> tuple[float, float]:
    """Calculates coordinate offsets from TileConfiguration."""
    tc = tile_config.downsample(downsample_factor)
    coords = tc.df
    x_offset = coords["x"].min()
    y_offset = coords["y"].min()
    logger.info(f"Calculated offsets: x={x_offset:.2f}, y={y_offset:.2f}")
    return x_offset, y_offset


def load_and_prepare_spots(
    spots_parquet_path: Path,
    idx: int,
    z_filter_tolerance: float,
    downsample_factor: int,
    x_offset: float,
    y_offset: float,
    max_proj: bool = False,
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
        (
            all_spots.filter(
                pl.col("z").is_between(idx - z_filter_tolerance, idx + z_filter_tolerance, closed="both")
            )
            if not max_proj
            else all_spots
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
        return pl.DataFrame()

    return spots


def compute_polygon_metadata(seg_mask: np.ndarray, idx: int) -> pl.DataFrame:
    """Compute per-label region metadata for a single slice."""
    df = compute_regionprops_table(seg_mask, properties=("label", "area", "centroid"))
    if df.is_empty():
        logger.warning(f"Slice {idx}: No regions found in the segmentation image.")
        return pl.DataFrame(
            schema={
                "polygon_id": pl.UInt32,
                "label": pl.UInt32,
                "area": pl.Float32,
                "centroid_y": pl.Float32,
                "centroid_x": pl.Float32,
            }
        )

    df = df.rename({"centroid-0": "centroid_y", "centroid-1": "centroid_x"})
    df = df.with_row_index("polygon_id").select(["polygon_id", "label", "area", "centroid_y", "centroid_x"])
    return df.with_columns(
        pl.col("polygon_id").cast(pl.UInt32),
        pl.col("area").cast(pl.Float32),
        pl.col("centroid_x").cast(pl.Float32),
        pl.col("centroid_y").cast(pl.Float32),
    )


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int64)


def assign_spots_to_labels(spots_df: pl.DataFrame, seg_mask: np.ndarray, idx: int) -> pl.DataFrame:
    """Assign spots to labels via direct label lookup at rounded pixel coordinates."""
    if spots_df.is_empty():
        logger.warning(f"Slice {idx}: Empty spots dataframe; returning empty assignments.")
        return pl.DataFrame()

    xs = spots_df.get_column("x_adj").to_numpy()
    ys = spots_df.get_column("y_adj").to_numpy()
    cols = _round_half_up(xs)
    rows = _round_half_up(ys)

    in_bounds = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < int(seg_mask.shape[0]))
        & (cols < int(seg_mask.shape[1]))
    )
    if not np.any(in_bounds):
        return pl.DataFrame()

    rows_in = rows[in_bounds]
    cols_in = cols[in_bounds]
    labels = np.asarray(seg_mask[rows_in, cols_in], dtype=np.int64)
    valid = labels > 0
    if not np.any(valid):
        return pl.DataFrame()

    spot_ids = spots_df.get_column("spot_id").to_numpy()[in_bounds][valid]
    targets = spots_df.get_column("target").to_numpy()[in_bounds][valid]
    out_labels = labels[valid]
    return pl.DataFrame({"spot_id": spot_ids, "target": targets, "label": out_labels}).with_columns(
        pl.col("spot_id").cast(pl.UInt32),
        pl.col("label").cast(pl.UInt32),
    )


def save_results(
    assignments_df: pl.DataFrame,
    polygons_df: pl.DataFrame,
    ident_path: Path,
    polygons_path: Path,
    idx: int,
) -> None:
    """Saves the spot assignments and polygon metadata to Parquet files."""
    logger.info(f"Slice {idx}: Saving spot assignments ({len(assignments_df)} rows) to {ident_path}...")
    try:
        assignments_df.write_parquet(ident_path)
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to write spot assignments parquet: {e}")
        raise

    logger.info(
        f"Slice {idx}: Saving polygon metadata ({len(polygons_df)} entries) to {polygons_path}..."
    )
    try:
        polygons_df.write_parquet(polygons_path)
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to write polygon metadata parquet: {e}")
        raise


def generate_debug_plot(
    img: np.ndarray,
    spots_df: pl.DataFrame,
    segmentation_zarr_path: Path,
    output_dir: Path,
    idx: int,
) -> None:
    import zarr

    logger.info(f"Slice {idx}: Generating debug plots...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()

        intensity = None
        intensity_zarr_path = segmentation_zarr_path.parent / "input_image.zarr"
        if intensity_zarr_path.exists():
            try:
                intensity_stack = zarr.open_array(str(intensity_zarr_path), mode="r")
                if idx < len(intensity_stack):
                    intensity_slice = intensity_stack[idx]
                    intensity = intensity_slice[0] if intensity_slice.ndim == 3 else intensity_slice
            except Exception as e:
                logger.warning(f"Slice {idx}: Could not load intensity image for debug plot: {e}")

        img_h, img_w = img.shape
        sl = np.s_[img_h // 4 : img_h * 3 // 4, img_w // 4 : img_w * 3 // 4]

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 6), dpi=150)

        if intensity is not None:
            vmin = np.percentile(intensity[sl][intensity[sl] > 0], 1) if np.any(intensity[sl] > 0) else 0
            vmax = np.percentile(intensity[sl], 99)
            axs[0].imshow(intensity[sl], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            axs[0].text(0.5, 0.5, "Intensity N/A", ha="center", va="center", transform=axs[0].transAxes)
        axs[0].set_title(f"Slice {idx}: Intensity (Input)")

        import matplotlib.pyplot as plt  # for colormap

        cmap = plt.cm.get_cmap("tab20", np.max(img) + 1)
        cmap.set_under(color="black")
        axs[1].imshow(img[sl], origin="lower", cmap=cmap, interpolation="none", vmin=1)
        axs[1].set_title(f"Slice {idx}: Segmentation Mask (Used)")

        axs[2].imshow(img[sl], origin="lower", cmap=cmap, interpolation="none", vmin=1, alpha=0.6)
        if not spots_df.is_empty():
            plot_spots_x, plot_spots_y = filter_spots_for_imshow(spots_df, sl, columns=("x_adj", "y_adj"))
        else:
            plot_spots_x, plot_spots_y = [], []

        axs[2].scatter(plot_spots_x, plot_spots_y, s=1, alpha=0.7, c="red")
        axs[2].set_title(f"Slice {idx}: Spots on Segmentation")

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


def process_slice(
    idx: int,
    segmentation_zarr_path: Path,
    spots_parquet_path: Path,
    x_offset: float,
    y_offset: float,
    output_dir: Path,
    overwrite: bool,
    debug: bool,
    max_proj: bool = False,
) -> None:
    ident_path = output_dir / f"ident_{idx}.parquet"
    polygons_path = output_dir / f"polygons_{idx}.parquet"

    if not overwrite and ident_path.exists() and polygons_path.exists():
        logger.info(f"Slice {idx}: Skipping, output files already exist ({ident_path}, {polygons_path}).")
        return

    try:
        img = load_segmentation_slice(segmentation_zarr_path, idx)

        spots_df = load_and_prepare_spots(
            spots_parquet_path,
            idx,
            Z_FILTER_TOLERANCE,
            DOWNSAMPLE_FACTOR,
            x_offset,
            y_offset,
            max_proj=max_proj,
        )

        if spots_df.is_empty():
            logger.warning(f"Slice {idx}: No spots loaded or filtered. Writing empty outputs.")
            save_results(spots_df, compute_polygon_metadata(img, idx), ident_path, polygons_path, idx)
            return

        polygons_df = compute_polygon_metadata(img, idx)
        if polygons_df.is_empty():
            logger.warning(f"Slice {idx}: No regions found. Writing empty assignments.")
            save_results(pl.DataFrame(), polygons_df, ident_path, polygons_path, idx)
            return

        assignments_df = assign_spots_to_labels(spots_df, img, idx)
        save_results(assignments_df, polygons_df, ident_path, polygons_path, idx)

        if debug:
            generate_debug_plot(img, spots_df, segmentation_zarr_path, output_dir, idx)

        logger.info(f"Slice {idx}: Processing finished successfully.")

    except Exception as e:
        logger.error(f"Slice {idx}: Failed during processing pipeline: {e}")
        raise e


def filter_spots_by_bounds(
    spots: pl.DataFrame,
    lim: tuple[tuple[float | None, float | None], tuple[float | None, float | None]] | tuple[slice, slice],
    x_col: str = "x",
    y_col: str = "y",
) -> pl.DataFrame:
    match lim:
        case ((x_min, x_max), (y_min, y_max)):
            return spots.filter(
                (pl.col(x_col) >= x_min if x_min is not None else True)
                & (pl.col(y_col) >= y_min if y_min is not None else True)
                & (pl.col(x_col) < x_max if x_max is not None else True)
                & (pl.col(y_col) < y_max if y_max is not None else True)
            )
        case (sl_y, sl_x):
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
    return [filtered.get_column(c) for c in columns]


def run_(
    segmentation_zarr_path: Path,
    spots: Path,
    idx: int,
    overwrite: bool,
    debug: bool,
    x_offset: float,
    y_offset: float,
    max_proj: bool = False,
    spot_cb_label: str | None = None,
) -> None:
    """Process a specific Z-slice and assign detected spots to regions."""
    spots_path = spots

    cb_for_chunks = spot_cb_label
    if cb_for_chunks is None:
        try:
            cb_for_chunks = spots_path.stem.split("+")[1]
        except Exception:
            cb_for_chunks = "spots"
    # Put chunks inside the segmentation zarr folder
    output_chunk_dir = segmentation_zarr_path / f"chunks+{cb_for_chunks}"

    try:
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_chunk_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_chunk_dir}: {e}")
        sys.exit(1)

    if not segmentation_zarr_path.exists():
        logger.error(f"Segmentation Zarr store not found: {segmentation_zarr_path}")
        sys.exit(1)
    if not spots_path.exists():
        logger.error(f"Spots Parquet file not found: {spots_path}")
        sys.exit(1)

    logger.info("--- Configuration ---")
    logger.info(f"Output directory: {output_chunk_dir}")
    logger.info(f"Segmentation: {segmentation_zarr_path}")
    logger.info(f"Spots: {spots_path}")
    logger.info(f"Offsets: x={x_offset:.2f}, y={y_offset:.2f}")
    logger.info(f"Overwrite: {overwrite}")

    try:
        process_slice(
            idx=idx,
            segmentation_zarr_path=segmentation_zarr_path,
            spots_parquet_path=spots_path,
            x_offset=x_offset,
            y_offset=y_offset,
            output_dir=output_chunk_dir,
            overwrite=overwrite,
            debug=debug,
            max_proj=max_proj,
        )
        logger.info(f"Successfully finished processing slice {idx}.")
    except Exception as e:
        logger.critical(f"Pipeline execution failed for slice {idx}.")
        raise e


def initialize() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.argument("roi", type=str, default="*")
@click.option(
    "--codebook",
    type=str,
    required=True,
    help="Codebook name used for spots.",
)
@click.option(
    "--seg-codebook",
    type=str,
    required=True,
    default=None,
    help="Segmentation codebook.",
)
@click.option(
    "--spots",
    "spots_opt",
    type=click.Path(dir_okay=True, file_okay=True, path_type=Path),
    default=None,
    help=(
        "Path to spots parquet file, or directory with per-ROI spots parquets "
        "named '{roi}+{codebook}.parquet'. Defaults to <workspace>/analysis/output."
    ),
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation-sam_postproc_s1-2-2_v500.zarr",
    show_default=True,
    help="Relative path to the segmentation Zarr store within the input directory.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging and generate debug plots.")
def overlay(
    path: Path,
    roi: str,
    codebook: str,
    spots_opt: Path | None,
    seg_codebook: str | None,
    segmentation_name: str,
    overwrite: bool,
    debug: bool,
) -> None:
    import zarr

    setup_cli_logging(path, component="segment.overlay_spots", file="overlay_spots.py", debug=debug)

    ws = Workspace(path)
    batch_mode = roi == "*"
    try:
        rois = ws.resolve_rois() if batch_mode else ws.resolve_rois([roi])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not rois:
        raise click.ClickException(f"No ROIs discovered under workspace {path}.")

    seg_cb = seg_codebook or codebook
    if seg_codebook is None:
        logger.info(f"Segmentation codebook not provided; defaulting to --codebook '{codebook}'.")

    for current_roi in rois:
        try:
            if spots_opt is not None:
                p = Path(spots_opt)
                if p.is_dir():
                    spots = p / f"{current_roi}+{ws.sanitize_codebook_name(codebook)}.parquet"
                else:
                    spots = p
                    if batch_mode:
                        logger.warning(
                            f"Batch mode with a single spots file '{p}'; this file will be reused for all ROIs."
                        )
            else:
                try:
                    spots = ws.spots_parquet(current_roi, codebook, must_exist=True)
                except FileNotFoundError as e:
                    msg = f"Skipping ROI '{current_roi}': {e}"
                    if batch_mode:
                        logger.warning(msg)
                        continue
                    raise click.ClickException(str(e))
            stitch_root = ws.stitch(current_roi, seg_cb)
            seg_path = stitch_root / segmentation_name

            if not seg_path.exists():
                msg = f"Skipping ROI '{current_roi}': segmentation not found at {seg_path}."
                if batch_mode:
                    logger.warning(msg)
                    continue
                raise click.ClickException(msg)

            if not spots.exists():
                msg = f"Skipping ROI '{current_roi}': spots parquet not found at {spots}."
                if batch_mode:
                    logger.warning(msg)
                    continue
                raise click.ClickException(msg)

            spot_cb_label = ws.sanitize_codebook_name(codebook)
            output_chunk_dir = seg_path / f"chunks+{spot_cb_label}"

            try:
                z = zarr.open_array(seg_path, mode="r")
            except Exception as exc:
                msg = f"Skipping ROI '{current_roi}': failed to open segmentation '{seg_path}': {exc}"
                if batch_mode:
                    logger.warning(msg)
                    continue
                raise click.ClickException(msg) from exc

            pending_slices: list[int] = []
            if overwrite:
                pending_slices = list(range(z.shape[0]))
            else:
                for idx in range(z.shape[0]):
                    ident_path = output_chunk_dir / f"ident_{idx}.parquet"
                    polygons_path = output_chunk_dir / f"polygons_{idx}.parquet"
                    if ident_path.exists() and polygons_path.exists():
                        continue
                    pending_slices.append(idx)

            if not pending_slices:
                logger.info(
                    f"ROI '{current_roi}': overlay spots already complete under {output_chunk_dir} "
                    "(use --overwrite to recompute)."
                )
                continue

            try:
                tileconfig = ws.tileconfig(current_roi)
            except FileNotFoundError as exc:
                msg = f"Skipping ROI '{current_roi}': {exc}"
                if batch_mode:
                    logger.warning(msg)
                    continue
                raise click.ClickException(msg) from exc

            x_offset, y_offset = calculate_coordinate_offsets(tileconfig, DOWNSAMPLE_FACTOR)

            with ProcessPoolExecutor(max_workers=8, mp_context=get_context("spawn")) as executor:
                futures = []
                for i in pending_slices:
                    futures.append(
                        executor.submit(
                            run_,
                            seg_path,
                            spots,
                            i,
                            overwrite,
                            debug,
                            x_offset,
                            y_offset,
                            max_proj=z.shape[0] == 1,
                            spot_cb_label=spot_cb_label,
                        )
                    )
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        if batch_mode:
                            logger.error(f"ROI '{current_roi}': error processing slice: {e}")
                            continue
                        raise
        except Exception as e:
            if batch_mode:
                logger.error(f"Skipping ROI '{current_roi}' due to error: {e}")
                continue
            raise


if __name__ == "__main__":
    overlay()
