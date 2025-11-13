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

# Share the exact same region detection primitives used by
# `segment overlay spots` to keep behavior identical up to the
# intersection step.
from fishtools.segment.overlay_spots import (
    extract_polygons_from_mask,
    load_segmentation_slice,
)
from fishtools.segment.utils import (
    StitchPaths,
    compute_regionprops_table,
    slice_intensity_channel,
    write_regionprops_parquet,
)


def _process_slice_shared_detection(
    idx: int,
    segmentation_zarr_path: Path,
    intensity_zarr_path: Path,
    channel: str,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Compute per-label intensity using the same region detection code
    as `segment overlay spots`.

    Steps shared with spots overlay:
    - load segmentation slice via `load_segmentation_slice`
    - detect regions via `extract_polygons_from_mask`

    Divergence (this tool):
    - intersect regions with an intensity Zarr (per-channel) and compute
      region properties with intensity measures.
    """

    props_path = output_dir / f"intensity_{channel}" / f"intensity-{idx:02d}.parquet"
    if not overwrite and props_path.exists():
        logger.info(f"Slice {idx} [channel={channel}]: Skipping, output files already exist.")
        return

    # Load intensity slice for the requested channel
    intensity_arr = zarr.open_array(str(intensity_zarr_path), mode="r")
    intensity_img = slice_intensity_channel(intensity_arr, idx, channel)

    # Load segmentation slice using the same helper as spots overlay
    seg_mask = load_segmentation_slice(segmentation_zarr_path, idx)

    if seg_mask.shape != intensity_img.shape:
        raise ValueError(
            f"Slice {idx}: Shape mismatch between segmentation ({seg_mask.shape}) "
            f"and intensity ({intensity_img.shape})."
        )

    # Detect regions using the shared code path. We don't need the polygons
    # here, but running this ensures region discovery is identical to the
    # spots overlay pipeline up to this point.
    polygons_with_meta = extract_polygons_from_mask(seg_mask, idx)
    labels_from_polygons = {meta["label"] for _, meta in polygons_with_meta}

    logger.info(f"Slice {idx} [channel={channel}]: Calculating intensity region properties…")
    df = compute_regionprops_table(seg_mask, intensity_image=intensity_img)

    # Sanity: ensure label sets match to guarantee parity with spots overlay
    if not labels_from_polygons:
        logger.warning(f"Slice {idx}: No regions found by shared detector; writing empty output.")
    else:
        if "label" in df.columns:
            labels_from_props = set(df["label"].to_list())  # type: ignore[arg-type]
            if labels_from_props != labels_from_polygons:
                missing = labels_from_polygons - labels_from_props
                extra = labels_from_props - labels_from_polygons
                raise RuntimeError(
                    "Region detection parity check failed: label sets differ "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})."
                )

    write_regionprops_parquet(df, output_dir, channel, idx, overwrite=overwrite)


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
@click.option(
    "--channel",
    type=str,
    help=(
        "Channel name. If omitted, processes all channels from the intensity Zarr "
        "(attrs['key'])."
    ),
)
@click.option(
    "--max-workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of parallel processes to use.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
def overlay_intensity(
    input_dir: Path,
    segmentation_name: str,
    intensity_name: str,
    channel: str | None,
    max_workers: int,
    overwrite: bool,
):
    """
    Extracts region properties and intensity images for each Z-slice
    from segmentation and intensity Zarr stores.
    """
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

    # --- Determine Number of Slices & Channels ---
    try:
        seg_stack = zarr.open_array(str(segmentation_zarr_path), mode="r")
        num_slices = seg_stack.shape[0]
        logger.info(f"Found {num_slices} slices in {segmentation_zarr_path}.")
    except Exception as e:
        logger.error(f"Failed to open segmentation Zarr to determine slice count: {e}")
        sys.exit(1)

    try:
        intensity_arr = zarr.open_array(str(intensity_zarr_path), mode="r")
        keys = intensity_arr.attrs.get("key")
        if channel is None:
            if keys is None:
                raise ValueError(
                    "Intensity Zarr missing attrs['key']; provide --channel explicitly."
                )
            if isinstance(keys, (str, bytes)):
                channels: list[str] = [str(keys)]
            else:
                channels = [str(k) for k in keys]
        else:
            channels = [channel]
    except Exception as e:
        logger.error(f"Failed to resolve channels from intensity Zarr: {e}")
        sys.exit(1)

    # --- Process Slices in Parallel ---
    processed_count = 0
    failed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        futures = {}
        for ch in channels:
            for idx in range(num_slices):
                fut = executor.submit(
                    _process_slice_shared_detection,
                    idx,
                    segmentation_zarr_path,
                    intensity_zarr_path,
                    ch,
                    input_dir,
                    overwrite,
                )
                futures[fut] = (ch, idx)

        for future in as_completed(futures):
            ch, idx = futures[future]
            try:
                future.result()  # Raise exception if one occurred in the process
                processed_count += 1
                logger.info(f"Successfully processed slice {idx} [channel={ch}].")
            except Exception as e:
                logger.error(f"Slice {idx} [channel={ch}] failed: {e}")
                failed_count += 1

    logger.info("--- Extraction Summary ---")
    logger.info(f"Total slices: {num_slices}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed: {failed_count}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    overlay_intensity()
