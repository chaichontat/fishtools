"""
Extracts region properties (including intensity measures) for each slice
of a segmentation mask and saves them along with the corresponding intensity image.
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import rich_click as click
import zarr
from loguru import logger

from fishtools.segment.utils import StitchPaths, process_slice_regionprops


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
    if not channel:
        logger.error("Channel name must be provided via --channel.")
        sys.exit(1)

    stitch_paths = StitchPaths.from_stitch_root(input_dir)
    segmentation_zarr_path = stitch_paths.segmentation(segmentation_name)
    intensity_zarr_path = stitch_paths.intensity_store(intensity_name)

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
