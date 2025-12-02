"""
Distributed 3D post-processing pipeline for segmentation masks.

Applies the 4-phase postproc3d pipeline to chunked zarr data using Dask,
with stitching via face-matching union-find.

Design Rationale
----------------
The postproc3d pipeline consists of 4 phases:
  1. gaussian_smooth_labels - Gaussian-weighted voting to smooth jagged boundaries
  2. relabel_connected_components - Assign unique IDs to disconnected fragments
  3. compute_metadata_and_adjacency - Compute volumes and contact areas
  4. donate_small_cells - Absorb tiny fragments into neighboring cells

For very large volumes (>4000^3 voxels), running these phases on the full volume
is infeasible due to memory constraints. This module implements a chunked approach.

Key Design Decisions
--------------------
1. **Wrap all 4 phases in one function per chunk**
   Rather than chunking each phase separately (which would require complex
   cross-chunk coordination for phases 2-4), we run the entire pipeline on
   each overlapped chunk. This keeps the per-chunk logic identical to the
   non-distributed case.

2. **Use the same overlap removal as distributed_segmentation**
   - Each chunk is read with XY overlap by default (configurable via `margin` → `overlap = 2*margin`)
   - Chunks always span the full Z extent (no chunking in Z)
   - After processing, overlaps are removed using `remove_overlaps`, matching the stitching behavior
     of `distributed_segmentation`
   - The overlap must be large enough that the trimmed core region has correct context from all directions

3. **Reuse stitching from distributed_segmentation**
   After processing, chunks have locally-processed labels that need stitching:
   - global_segment_ids() encodes block index into label IDs (avoids collisions)
   - block_faces() extracts boundary faces
   - adjacent_faces() pairs faces between neighboring blocks
   - block_face_adjacency_graph() finds which labels touch across boundaries
   - scipy.sparse.csgraph.connected_components() determines merges (union-find)
   - Final relabeling via dask.array.map_blocks

4. **Overlap sizing for Gaussian smoothing**
   Gaussian smoothing with sigma=4 has effective radius ~4*sigma = 16 voxels.
   With 10% overlap on 2048×2048 XY blocks (~205px) and 50px margin crop, the
   core region still sits well inside the chunk boundary, comfortably beyond
   16px. This ensures Gaussian voting at the core boundary has essentially
   identical context from neighboring chunks.

5. **Small cell donation at boundaries**
   Small cells near chunk boundaries might have their best neighbor in another
   chunk. With sufficient overlap (>100px), small cells in the core region
   have their full neighborhood visible, so donation decisions are correct.

Pipeline Flow
-------------
```
Input zarr (from distributed_segmentation)
        |
        v
+--------------------------------------------------+
| Per-chunk (parallel via Dask):                   |
|   1. Read chunk with overlap                     |
|   2. gaussian_smooth_labels_cupy (sigma=4)       |
|   3. relabel_connected_components                |
|   4. compute_metadata_and_adjacency              |
|   5. donate_small_cells                          |
|   6. Remove overlaps (match distributed_segmentation) |
|   7. Assign globally unique IDs                  |
|   8. Extract faces, write to temp zarr           |
+--------------------------------------------------+
        |
        v
Stitch (reuse from distributed_segmentation):
   - adjacent_faces -> block_face_adjacency_graph
   - connected_components (union-find)
   - Apply remap via dask.array.map_blocks
        |
        v
Output zarr (post-processed masks)
```

Usage
-----
CLI:
    python -m fishtools.segmentation.distributed.distributed_postproc \\
        /path/to/segmentation.zarr \\
        --blocksize 512 --sigma 4.0 --v-min 8000

Programmatic:
    from fishtools.segmentation.distributed.distributed_postproc import distributed_postproc
    result = distributed_postproc(input_zarr, write_path, sigma=4.0, V_min=8000, ...)
"""

import logging
import os
from pathlib import Path
from typing import Any

import dask
import dask.array
import numpy as np
import typer
import zarr
from numpy.typing import NDArray
from rich.logging import RichHandler

from fishtools.segment.postproc3d import compute_metadata_and_adjacency, donate_small_cells, gaussian_smooth_labels_cupy, relabel_connected_components
from fishtools.segmentation.distributed.gpu_cluster import cluster, myGPUCluster, myLocalCluster
from fishtools.segmentation.distributed.merge_utils import (
    block_faces,
    bounding_boxes_in_global_coordinates,
    determine_merge_relabeling,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    relabel_and_write,
    remove_overlaps,
)

logging.basicConfig(level="INFO", handlers=[RichHandler(level="INFO")])
logger = logging.getLogger("rich")


def process_postproc_block(
    block_index: tuple[int, ...],
    crop: tuple[slice, ...],
    input_zarr: zarr.Array,
    output_zarr: zarr.Array,
    blocksize: tuple[int, ...],
    overlap: int,
    nblocks: NDArray[np.int_],
    postproc_kwargs: dict[str, Any],
) -> tuple[list[NDArray[Any]], list[tuple[slice, ...]], NDArray[np.uint32]]:
    """
    Process one block through all 4 post-processing phases.

    Parameters
    ----------
    block_index : tuple of int
        The (z, y, x) index of this block in the block grid
    crop : tuple of slice
        The crop coordinates (with overlap) to read from input
    input_zarr : zarr.Array
        Input segmentation masks
    output_zarr : zarr.Array
        Output zarr to write results (after overlap removal)
    blocksize : tuple of int
        Target block size (without overlap)
    overlap : int
        Number of voxels of spatial overlap used when building crops
    nblocks : NDArray
        Number of blocks along each axis
    postproc_kwargs : dict
        Parameters for post-processing (sigma, V_min, bg_scale, etc.)

    Returns
    -------
    tuple[list[NDArray], list[tuple[slice, ...]], NDArray[np.uint32]]
        (faces, boxes, box_ids) for stitching
    """
    logger.info(f"Processing block {block_index}")

    # 1. Read chunk
    masks = np.asarray(input_zarr[crop])

    # Ensure integer dtype
    if not np.issubdtype(masks.dtype, np.integer):
        masks = masks.astype(np.int32)

    # 2. Run 4-phase pipeline
    # Phase 1: Gaussian smooth
    sigma = postproc_kwargs.get("sigma", 4.0)
    bg_scale = postproc_kwargs.get("bg_scale", 1.0)
    max_expansion = postproc_kwargs.get("max_expansion", 1)

    try:
        masks = gaussian_smooth_labels_cupy(
            masks,
            sigma=sigma,
            in_place=True,
            bg_scale=bg_scale,
            max_expansion=max_expansion,
        )
    except ImportError:
        # Fall back to CPU if CuPy/CUDA not available
        from fishtools.segment.postproc3d import gaussian_smooth_labels

        masks = gaussian_smooth_labels(
            masks,
            sigma=sigma,
            in_place=True,
            bg_scale=bg_scale,
            max_expansion=max_expansion,
        )

    # Phase 2: Relabel connected components
    masks = relabel_connected_components(masks, in_place=True)

    # Phase 3: Compute metadata
    V_min = postproc_kwargs.get("V_min", 8000)
    min_contact_fraction = postproc_kwargs.get("min_contact_fraction", 0.0)

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    # Phase 4: Donate small cells
    masks = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=V_min,
        min_contact_fraction=min_contact_fraction,
        in_place=True,
    )

    logger.info(f"Block {block_index}: {int(masks.max())} labels after postproc")

    # 3. Remove overlaps to match distributed_segmentation behavior
    masks_cropped, crop_trimmed = remove_overlaps(
        masks,
        crop,
        overlap,
        blocksize,
    )
    crop_trimmed = tuple(crop_trimmed)

    # 4. Assign globally unique IDs
    masks_global, _ = global_segment_ids(masks_cropped, block_index, nblocks)

    # 5. Extract faces for stitching
    faces = block_faces(masks_global)

    # 6. Write to output zarr
    output_zarr[crop_trimmed] = masks_global.astype(np.uint32)

    # 7. Compute boxes and IDs for this block
    box_ids = np.unique(masks_global)
    box_ids = box_ids[box_ids > 0].astype(np.uint32)
    boxes = bounding_boxes_in_global_coordinates(masks_global, crop_trimmed)

    return faces, boxes, box_ids


def _copy_zarr_metadata(input_zarr: zarr.Array, output_path: Path, input_path: Path | None = None) -> None:
    """Copy metadata from input zarr to output zarr, including source mtime."""
    output_zarr = zarr.open(output_path, mode="r+")

    # Copy all attributes from input
    for key, value in input_zarr.attrs.items():
        output_zarr.attrs[key] = value

    # Add source file mtime if we can determine the input path
    if input_path is not None:
        try:
            input_mtime = os.path.getmtime(input_path)
            output_zarr.attrs["source_mtime"] = input_mtime
            output_zarr.attrs["source_path"] = str(input_path)
        except OSError:
            pass  # Can't get mtime, skip

    # Add processing metadata
    output_zarr.attrs["postproc_version"] = "distributed_postproc_v1"


@cluster
def distributed_postproc(
    input_zarr: zarr.Array,
    write_path: Path | str,
    blocksize: tuple[int, ...] | None = None,
    margin: int = 100,
    sigma: float = 4.0,
    V_min: int = 8000,
    bg_scale: float = 1.0,
    max_expansion: int = 1,
    min_contact_fraction: float = 0.0,
    input_path: Path | None = None,
    cluster: myLocalCluster | myGPUCluster | None = None,
    cluster_kwargs: dict[str, Any] | None = None,
    temporary_directory: Path | None = None,
) -> zarr.Array:
    """
    Distributed post-processing of 3D segmentation masks.

    Applies Gaussian smoothing, connected component relabeling, and small cell
    donation in a tiled manner with overlap, then stitches results.

    Parameters
    ----------
    input_zarr : zarr.Array
        Input segmentation masks (3D integer array)
    write_path : Path or str
        Output path for final post-processed zarr
    blocksize : tuple of int, optional
        ZYX block size for tiled processing. If None, uses
        (Z_full, 2048, 2048) clipped to the volume shape. Z is
        never chunked; chunks always span the full Z extent.
    margin : int
        Margin parameter (in voxels) used to derive the spatial overlap
        between blocks (default 100). The actual overlap passed to
        `get_block_crops` and `remove_overlaps` is `overlap = 2*margin`.
    sigma : float
        Gaussian smoothing sigma (default 4.0)
    V_min : int
        Minimum volume threshold for small cell donation (default 8000)
    bg_scale : float
        Background scale factor for Gaussian voting (default 1.0)
    max_expansion : int
        Maximum expansion for Gaussian smooth (default 1)
    min_contact_fraction : float
        Minimum contact fraction for donation (default 0.0)
    input_path : Path, optional
        Path to input zarr for metadata copying (source mtime)
    cluster : cluster object, optional
        Existing Dask cluster to use
    cluster_kwargs : dict, optional
        Arguments for cluster creation if cluster not provided
    temporary_directory : Path, optional
        Directory for temporary files

    Returns
    -------
    zarr.Array
        Post-processed segmentation masks
    """
    write_path = Path(write_path)

    if input_zarr.ndim != 3:
        raise ValueError("distributed_postproc expects a 3D ZYX zarr array.")

    z_dim, y_dim, x_dim = map(int, input_zarr.shape)

    if blocksize is None:
        y_block = min(2048, y_dim)
        x_block = min(2048, x_dim)
    else:
        if len(blocksize) != 3:
            raise ValueError("blocksize must be a 3-tuple (Z, Y, X).")
        _, y_block_raw, x_block_raw = blocksize
        y_block = min(int(y_block_raw), y_dim)
        x_block = min(int(x_block_raw), x_dim)

    blocksize = (z_dim, y_block, x_block)
    overlap = 2 * margin

    logger.info(
        f"Starting distributed postproc: blocksize={blocksize}, "
        f"overlap={overlap}, margin={margin}, sigma={sigma}, V_min={V_min}"
    )

    # Set up temporary directory
    if temporary_directory is None:
        temporary_directory = write_path.parent / "postproc_temp"
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)

    # Get block indices and crops
    block_indices, block_crops = get_block_crops(input_zarr.shape, np.array(blocksize), overlap, mask=None)
    nblocks = get_nblocks(input_zarr.shape, np.array(blocksize))

    logger.info(f"Processing {len(block_indices)} blocks")

    # Create temp zarr for unstitched output
    temp_zarr_path = temporary_directory / "postproc_unstitched.zarr"
    temp_zarr = zarr.open(
        temp_zarr_path,
        mode="w",
        shape=input_zarr.shape,
        chunks=blocksize,
        dtype=np.uint32,
        codecs=[
            zarr.codecs.BytesCodec(),
            zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=4,
                shuffle=zarr.codecs.BloscShuffle.shuffle,
                typesize=2,
            ),
        ],
    )

    # Prepare postproc kwargs
    postproc_kwargs = {
        "sigma": sigma,
        "V_min": V_min,
        "bg_scale": bg_scale,
        "max_expansion": max_expansion,
        "min_contact_fraction": min_contact_fraction,
    }

    # Map over blocks
    assert cluster is not None
    futures = cluster.client.map(
        process_postproc_block,
        block_indices,
        block_crops,
        input_zarr=input_zarr,
        output_zarr=temp_zarr,
        blocksize=blocksize,
        overlap=overlap,
        nblocks=nblocks,
        postproc_kwargs=postproc_kwargs,
    )

    # Gather results
    from fishtools.utils.pretty_print import progress_bar

    with progress_bar(len(block_indices)) as submit:
        [fut.add_done_callback(submit) for fut in futures]
        results = cluster.client.gather(futures)

    # Unpack results
    faces_list, boxes_list, box_ids_list = [], [], []
    for faces, boxes, box_ids in results:
        if len(box_ids) > 0:
            faces_list.append(faces)
            boxes_list.extend(boxes)
            box_ids_list.append(box_ids)

    if len(box_ids_list) == 0:
        logger.warning("No labels found in any block")
        # Just copy temp to output
        dask.array.to_zarr(dask.array.from_zarr(temp_zarr), str(write_path), overwrite=True)
        _copy_zarr_metadata(input_zarr, write_path, input_path=input_path)
        return zarr.open(write_path, mode="r")

    # Flatten faces to match block_indices
    # faces_list contains faces for non-empty blocks only
    # Need to match with their block_indices
    non_empty_indices = [block_indices[i] for i, (_, _, box_ids) in enumerate(results) if len(box_ids) > 0]

    all_box_ids = np.concatenate(box_ids_list)

    logger.info(f"Stitching {len(all_box_ids)} labels across {len(non_empty_indices)} non-empty blocks")

    # Determine merge relabeling
    new_labeling = determine_merge_relabeling(non_empty_indices, faces_list, all_box_ids, compact=True)
    new_labeling_path = temporary_directory / "new_labeling.npy"
    np.save(new_labeling_path, new_labeling)

    logger.info(f"Relabeling to {int(new_labeling.max())} final labels")

    # Apply relabeling via dask
    relabel_and_write(temp_zarr, new_labeling_path, write_path)

    logger.info(f"Post-processing complete. Output saved to {write_path}")

    _copy_zarr_metadata(input_zarr, write_path, input_path=input_path)


# CLI
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to input segmentation zarr"),
    output_path: Path = typer.Option(None, help="Output path (default: input_postproc.zarr)"),
    blocksize: int = typer.Option(1024, help="XY block size for tiled processing"),
    sigma: float = typer.Option(4.0, help="Gaussian smoothing sigma"),
    v_min: int = typer.Option(3000, help="Minimum volume threshold for small cell donation"),
    margin: int = typer.Option(50, help="Margin parameter (overlap = 2*margin for overlap removal)"),
    workers_per_gpu: int = typer.Option(2, help="Workers per GPU"),
) -> None:
    """
    Post-process 3D segmentation masks with Gaussian smoothing and small cell donation.
    """
    input_zarr = zarr.open(input_path, mode="r")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_postproc.zarr"

    cluster_kwargs = {
        "workers_per_gpu": workers_per_gpu,
        "threads_per_worker": 1,
    }

    distributed_postproc(
        input_zarr=input_zarr,
        write_path=output_path,
        blocksize=(input_zarr.shape[0], blocksize, blocksize),
        margin=margin,
        sigma=sigma,
        V_min=v_min,
        input_path=input_path,
        cluster_kwargs=cluster_kwargs,
    )


if __name__ == "__main__":
    import cupy as cp
    cp.cuda.set_allocator(None)
    app()
