"""
Extracts region properties (including intensity measures) for each slice
of a segmentation mask and saves them along with the corresponding intensity image.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import rich_click as click
import zarr
from loguru import logger

from fishtools.io.workspace import Workspace

from fishtools.segment.overlay_spots import load_segmentation_slice
from fishtools.segment.utils import (
    StitchPaths,
    compute_regionprops_table,
    resolve_intensity_store,
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
    """Compute per-label intensity statistics for a single slice."""

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

    logger.info(f"Slice {idx} [channel={channel}]: Calculating intensity region properties…")
    df = compute_regionprops_table(seg_mask, intensity_image=intensity_img)

    write_regionprops_parquet(df, output_dir, channel, idx, overwrite=overwrite)


def _discover_channels(intensity_zarr_path: Path, channel: str | None) -> list[str]:
    try:
        intensity_arr = zarr.open_array(str(intensity_zarr_path), mode="r")
        if channel is not None:
            return [channel]

        keys = intensity_arr.attrs.get("key")
        if keys is None:
            raise ValueError("Intensity Zarr missing attrs['key']; provide --channel explicitly.")
        if isinstance(keys, (str, bytes)):
            return [str(keys)]
        return [str(k) for k in keys]
    except Exception as exc:  # pragma: no cover - IO failure path
        raise RuntimeError(
            f"Failed to resolve channels from intensity Zarr {intensity_zarr_path}: {exc}"
        ) from exc


def _run_overlay_for_roi(
    workspace: Workspace,
    roi: str,
    seg_codebook: str,
    intensity_codebook: str,
    segmentation_name: str,
    intensity_store: str,
    channel: str | None,
    threads: int,
    overwrite: bool,
) -> None:
    stitch_paths = StitchPaths.from_workspace(workspace, roi, seg_codebook)
    segmentation_zarr_path = stitch_paths.segmentation(segmentation_name)
    if not segmentation_zarr_path.exists():
        raise FileNotFoundError(f"ROI '{roi}': segmentation Zarr not found at {segmentation_zarr_path}")

    intensity_zarr_path = resolve_intensity_store(
        stitch_paths, intensity_codebook, store_name=intensity_store
    )

    try:
        seg_stack = zarr.open_array(str(segmentation_zarr_path), mode="r")
        num_slices = seg_stack.shape[0]
        logger.info(f"ROI '{roi}': Found {num_slices} slices in {segmentation_zarr_path}.")
    except Exception as exc:  # pragma: no cover - IO failure path
        raise RuntimeError(
            f"ROI '{roi}': failed to open segmentation Zarr {segmentation_zarr_path}: {exc}"
        ) from exc

    channels = _discover_channels(intensity_zarr_path, channel)
    logger.info(f"ROI '{roi}': Processing channels {channels} from {intensity_zarr_path}.")

    # Put intensity outputs inside the segmentation zarr folder
    output_dir = segmentation_zarr_path

    processed_count = 0
    failed_count = 0
    with ProcessPoolExecutor(max_workers=threads, mp_context=get_context("spawn")) as executor:
        futures: dict = {}
        for ch in channels:
            for idx in range(num_slices):
                fut = executor.submit(
                    _process_slice_shared_detection,
                    idx,
                    segmentation_zarr_path,
                    intensity_zarr_path,
                    ch,
                    output_dir,
                    overwrite,
                )
                futures[fut] = (ch, idx)

        for future in as_completed(futures):
            ch, idx = futures[future]
            try:
                future.result()
                processed_count += 1
                logger.info(f"ROI '{roi}': Successfully processed slice {idx} [channel={ch}].")
            except Exception as exc:
                logger.error(f"ROI '{roi}': Slice {idx} [channel={ch}] failed: {exc}")
                failed_count += 1

    logger.info(
        f"ROI '{roi}': Completed intensity overlay (processed={processed_count}, failed={failed_count})."
    )

    if failed_count > 0:
        raise RuntimeError(
            f"ROI '{roi}': Failed to process {failed_count} slice(s); inspect logs for details."
        )


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("roi", required=False)
@click.option(
    "--seg-codebook",
    type=str,
    required=True,
    help="Codebook label used for segmentation artifacts.",
)
@click.option(
    "--intensity-codebook",
    type=str,
    required=True,
    help="Codebook label used to resolve the intensity store.",
)
@click.option(
    "--segmentation-name",
    type=str,
    default="output_segmentation-sam.zarr",
    show_default=True,
    help="Relative path to the segmentation Zarr within the stitched ROI directory.",
)
@click.option(
    "--intensity-store",
    type=str,
    default="fused.zarr",
    show_default=True,
    help="Filename of the intensity Zarr inside stitch--ROI+<codebook>.",
)
@click.option(
    "--channel",
    type=str,
    help="Channel name. When omitted, all channels in attrs['key'] are processed.",
)
@click.option(
    "--threads",
    type=int,
    default=12,
    show_default=True,
    help="Number of parallel worker processes to use per ROI.",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
def overlay_intensity(
    path: Path,
    roi: str | None,
    seg_codebook: str,
    intensity_codebook: str,
    segmentation_name: str,
    intensity_store: str,
    channel: str | None,
    threads: int,
    overwrite: bool,
):
    """Overlay stitched intensity volumes onto segmentation masks for one or more ROIs."""

    workspace = Workspace(path)
    roi_token = (roi or "*").strip()
    batch_mode = roi_token == "*" or roi_token.lower() == "all"
    try:
        rois: Iterable[str] = (
            workspace.resolve_rois() if batch_mode else workspace.resolve_rois([roi_token])
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not rois:
        raise click.ClickException(f"No ROIs discovered under workspace {path}.")

    for current_roi in rois:
        try:
            _run_overlay_for_roi(
                workspace,
                current_roi,
                seg_codebook,
                intensity_codebook,
                segmentation_name,
                intensity_store,
                channel,
                threads,
                overwrite,
            )
        except Exception as exc:
            logger.error(f"ROI '{current_roi}': {exc}")
            if batch_mode:
                continue
            raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    overlay_intensity()
