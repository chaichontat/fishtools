"""
Extracts region properties (including intensity measures) for each slice
of a segmentation mask and saves them along with the corresponding intensity image.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import rich_click as click
import zarr
from loguru import logger
from skimage.measure import regionprops_table


def load_slice(zarr_path: Path, idx: int) -> np.ndarray | None:
    """Loads a specific Z-slice from a Zarr store."""
    logger.info(f"Slice {idx}: Loading from {zarr_path}...")
    try:
        img_stack = zarr.open_array(str(zarr_path), mode="r")
        img_slice = img_stack[idx]
        logger.info(f"Slice {idx}: Loaded slice with shape {img_slice.shape}.")
        return img_slice
    except Exception as e:
        logger.error(f"Slice {idx}: Failed to load Zarr slice: {e}")
        return None


def process_slice_regionprops(
    idx: int,
    segmentation_zarr_path: Path,
    intensity_zarr_path: Path,
    name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Loads segmentation and intensity slices, calculates regionprops,
    and saves results.
    """
    props_path = output_dir / f"intensity_{name}" / f"intensity-{idx:02d}.parquet"
    props_path.parent.mkdir(exist_ok=True, parents=True)

    if not overwrite and props_path.exists():
        logger.info(f"Slice {idx}: Skipping, output files already exist.")
        return

        # 2. Load Intensity Image
    intensity_img = zarr.open_array(str(intensity_zarr_path), mode="r")
    keys = intensity_img.attrs["key"]
    try:
        c_idx = keys.index(name)
    except ValueError:
        raise ValueError(f"Channel {name} not found in {keys}")
    intensity_img = intensity_img[idx, :, :, c_idx]

    # 1. Load Segmentation Mask
    seg_mask = load_slice(segmentation_zarr_path, idx)

    if seg_mask.shape != intensity_img.shape:
        raise ValueError(
            f"Slice {idx}: Shape mismatch between segmentation ({seg_mask.shape}) "
            f"and intensity ({intensity_img.shape}). Cannot calculate intensity props."
        )

    # 3. Calculate Region Properties
    logger.info(f"Slice {idx}: Calculating region properties...")
    try:
        # Define properties to extract, including intensity-based ones
        properties = [
            "label",
            "area",
            "centroid",
            "bbox",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ]
        # Use intensity_image argument for intensity stats
        props_table = regionprops_table(seg_mask, intensity_image=intensity_img, properties=properties)
        props_df = pl.DataFrame(props_table)
        n_regions = len(props_df)
        logger.info(f"Slice {idx}: Found {n_regions} regions.")

    except Exception as e:
        logger.error(f"Slice {idx}: Failed to calculate region properties: {e}")
        return  # Don't save partial results if props calculation fails

    # 4. Save Results
    props_df.write_parquet(props_path)
    logger.info(f"Slice {idx}: Saved region properties ({n_regions} regions) to {props_path}")


@click.command()
@click.argument(
    "input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation.zarr",
    show_default=True,
    help="Name of the segmentation Zarr store within the input directory.",
)
@click.option(
    "--intensity-name",
    type=str,
    default="input_image.zarr",
    show_default=True,
    help="Name of the intensity image Zarr store within the input directory.",
)
@click.option("--channel", type=str, help="Channel name.")
@click.option(
    "--max-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of parallel processes to use.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
def main(
    input_dir: Path,
    segmentation_name: str,
    intensity_name: str,
    channel: str,
    max_workers: int,
    overwrite: bool,
):
    """
    Extracts region properties and intensity images for each Z-slice
    from segmentation and intensity Zarr stores.
    """
    segmentation_zarr_path = input_dir / segmentation_name
    intensity_zarr_path = input_dir / intensity_name

    # --- Input Validation ---
    if not segmentation_zarr_path.exists():
        logger.error(f"Segmentation Zarr store not found: {segmentation_zarr_path}")
        sys.exit(1)
    if not intensity_zarr_path.exists():
        logger.error(f"Intensity Zarr store not found: {intensity_zarr_path}")
        sys.exit(1)

    # --- Determine Number of Slices ---
    try:
        seg_stack = zarr.open_array(str(segmentation_zarr_path), mode="r")
        num_slices = seg_stack.shape[0]
        logger.info(f"Found {num_slices} slices in {segmentation_zarr_path}.")
    except Exception as e:
        logger.error(f"Failed to open segmentation Zarr to determine slice count: {e}")
        sys.exit(1)

    # --- Process Slices in Parallel ---
    processed_count = 0
    failed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {
            executor.submit(
                process_slice_regionprops,
                idx,
                segmentation_zarr_path,
                intensity_zarr_path,
                channel,
                input_dir,
                overwrite,
            ): idx
            for idx in range(num_slices)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()  # Raise exception if one occurred in the process
                processed_count += 1
                logger.info(f"Successfully processed slice {idx}.")
            except Exception as e:
                logger.error(f"Slice {idx} failed: {e}")
                failed_count += 1

    logger.info("--- Extraction Summary ---")
    logger.info(f"Total slices: {num_slices}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
