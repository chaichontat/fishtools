import argparse
import logging
import time
from pathlib import Path
from typing import Sequence

import cupy as cp
from cupyx.scipy import ndimage as cpx_ndimage
import numpy as np
import zarr
from numpy.typing import NDArray

from fishtools.segmentation.distributed.merge_utils import (
    block_faces,
    determine_merge_relabeling,
    get_block_crops,
    get_nblocks,
    global_segment_ids,
    merge_all_boxes,
    relabel_and_write,
    remove_overlaps,
)


def _parse_triplet(arg: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected three comma-separated ints, got {arg!r}")
    try:
        z, y, x = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer in {arg!r}") from exc
    if z <= 0 or y <= 0 or x <= 0:
        raise argparse.ArgumentTypeError(f"Shape/blocksize must be positive, got {arg!r}")
    return z, y, x


def _make_synthetic_segmentation(
    base_dir: Path,
    shape: tuple[int, int, int],
    blocksize: tuple[int, int, int],
    overlap: int,
    labels_per_block: int,
) -> tuple[zarr.Array, list[tuple[int, ...]], list[tuple[slice, ...]]]:
    """Construct a synthetic segmentation_unstitched.zarr that mimics process_block output.

    Each block receives a set of box-shaped local segments, which are then:
    - trimmed to remove overlaps
    - assigned globally-unique IDs via global_segment_ids
    - written into a temporary Zarr store at the trimmed crop
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_zarr_path = base_dir / "segmentation_unstitched.zarr"

    blocksize_arr = np.asarray(blocksize, dtype=int)
    shape_arr = (int(shape[0]), int(shape[1]), int(shape[2]))

    temp_zarr = zarr.open_array(
        temp_zarr_path,
        mode="w",
        shape=shape_arr,
        chunks=tuple(blocksize_arr.tolist()),
        dtype=np.uint32,
    )

    block_indices, block_crops = get_block_crops(shape_arr, blocksize_arr, overlap, mask=None)
    nblocks = get_nblocks(shape_arr, blocksize_arr)

    rng = np.random.default_rng(0)

    for block_index, crop in zip(block_indices, block_crops):
        local_shape = tuple(slc.stop - slc.start for slc in crop)
        seg_local = np.zeros(local_shape, dtype=np.uint32)

        # Populate each block with several local labels (axis-aligned boxes).
        current_label = 1
        for _ in range(labels_per_block):
            if current_label >= (1 << 16) - 1:
                break
            z_len, y_len, x_len = local_shape
            if z_len < 2 or y_len < 2 or x_len < 2:
                break

            z0 = int(rng.integers(0, max(1, z_len - 4)))
            z1 = int(min(z_len, z0 + rng.integers(2, min(8, z_len - z0) + 1)))

            y0 = int(rng.integers(0, max(1, y_len - 16)))
            y1 = int(min(y_len, y0 + rng.integers(8, min(64, y_len - y0) + 1)))

            x0 = int(rng.integers(0, max(1, x_len - 16)))
            x1 = int(min(x_len, x0 + rng.integers(8, min(64, x_len - x0) + 1)))

            seg_local[z0:z1, y0:y1, x0:x1] = current_label
            current_label += 1

        seg_trimmed, crop_trimmed = remove_overlaps(seg_local, crop, overlap, blocksize)
        seg_global, _ = global_segment_ids(seg_trimmed, block_index, nblocks)

        temp_zarr[tuple(crop_trimmed)] = seg_global

    return temp_zarr, block_indices, block_crops


def _profile_stitch(
    temp_zarr: zarr.Array,
    block_indices: Sequence[tuple[int, ...]],
    block_crops: Sequence[tuple[slice, ...]],
    blocksize: tuple[int, int, int],
    overlap: int,
    output_path: Path,
) -> None:
    """Run the stitching phase on a pre-populated temp_zarr and report timings."""
    # 1) Recompute faces/boxes/IDs as in distributed_eval
    t_faces_start = time.perf_counter()
    read_time_total = 0.0
    faces_time_total = 0.0
    boxes_time_total = 0.0
    unique_time_total = 0.0
    # Detailed breakdown for the bounding-box (GPU) step
    gpu_upload_total = 0.0
    gpu_find_objects_total = 0.0
    boxes_filter_total = 0.0
    boxes_translate_total = 0.0

    results: list[tuple[list[NDArray[np.uint32]], list[tuple[slice, ...]], NDArray[np.uint32]]] = []
    for block_index, block_crop in zip(block_indices, block_crops):
        spatial_crop = block_crop
        spatial_blocksize = blocksize
        trimmed_crop: list[slice] = []
        for slc, bs in zip(spatial_crop, spatial_blocksize):
            start = slc.start if slc.start == 0 else slc.start + overlap
            stop = min(start + bs, slc.stop)
            trimmed_crop.append(slice(start, stop))
        trimmed_crop_tuple = tuple(trimmed_crop)

        t0 = time.perf_counter()
        seg_block = temp_zarr[trimmed_crop_tuple]
        read_time_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        faces = block_faces(seg_block)
        faces_time_total += time.perf_counter() - t0

        # Bounding boxes + GPU upload + unique all use the same GPU array
        t0 = time.perf_counter()
        seg_block_gpu = cp.asarray(seg_block)
        gpu_upload = time.perf_counter() - t0
        gpu_upload_total += gpu_upload

        t0 = time.perf_counter()
        boxes_local = cpx_ndimage.find_objects(seg_block_gpu)
        gpu_find = time.perf_counter() - t0
        gpu_find_objects_total += gpu_find

        t0 = time.perf_counter()
        boxes_local = [b for b in boxes_local if b is not None]
        boxes_filter = time.perf_counter() - t0
        boxes_filter_total += boxes_filter

        t0 = time.perf_counter()
        translate = lambda a, b: slice(a.start + b.start, a.start + b.stop)
        boxes = [tuple(translate(a, b) for a, b in zip(trimmed_crop_tuple, box)) for box in boxes_local]
        boxes_translate = time.perf_counter() - t0
        boxes_translate_total += boxes_translate

        boxes_time_total += gpu_upload + gpu_find + boxes_filter + boxes_translate

        t0 = time.perf_counter()
        unique_ids_gpu = cp.unique(seg_block_gpu)
        unique_ids = cp.asnumpy(unique_ids_gpu)
        unique_time_total += time.perf_counter() - t0
        box_ids = unique_ids[unique_ids != 0]
        results.append((faces, boxes, box_ids))
    t_faces_end = time.perf_counter()

    # 2) Filter to non-empty blocks and flatten lists
    faces_list: list[list[NDArray[np.uint32]]] = []
    boxes_list: list[list[tuple[slice, ...]]] = []
    box_ids_list: list[NDArray[np.uint32]] = []
    non_empty_indices: list[tuple[int, ...]] = []

    for i, (faces, boxes, box_ids) in enumerate(results):
        if box_ids.size > 0:
            faces_list.append(faces)
            boxes_list.append(boxes)
            box_ids_list.append(box_ids)
            non_empty_indices.append(block_indices[i])

    if not box_ids_list:
        print("No non-empty blocks found; nothing to stitch.")
        return

    boxes = [box for sublist in boxes_list for box in sublist]
    all_box_ids = np.concatenate(box_ids_list).astype(int)

    # 3) Determine merge relabeling
    t_relabel_lut_start = time.perf_counter()
    lut = determine_merge_relabeling(
        [(bi[0], bi[1], bi[2]) for bi in non_empty_indices],
        faces_list,
        all_box_ids,
    )
    t_relabel_lut_end = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lut_path = output_path.parent / "new_labeling.npy"
    np.save(lut_path, lut)

    # 4) Apply LUT over full temp_zarr via dask
    t_apply_start = time.perf_counter()
    relabel_and_write(temp_zarr, lut_path, output_path)
    t_apply_end = time.perf_counter()

    # 5) Merge bounding boxes with final IDs
    t_boxes_start = time.perf_counter()
    merged_boxes = merge_all_boxes(boxes, lut[all_box_ids])
    t_boxes_end = time.perf_counter()

    print("=== Stitching profile (synthetic) ===")
    print(f"Blocks: {len(block_indices)}, non-empty blocks: {len(non_empty_indices)}")
    print(f"Total labels before merge: {len(all_box_ids)}")
    print(f"Final non-background labels: {int(lut.max())}")
    print(f"Number of merged boxes: {len(merged_boxes)}")
    total_faces_boxes = t_faces_end - t_faces_start
    print(f"Faces/boxes extraction (total): {total_faces_boxes:.2f} s")
    n_blocks = max(len(block_indices), 1)
    print(
        f"  - read seg_block: {read_time_total:.2f} s "
        f"({read_time_total / n_blocks:.3f} s/block)"
    )
    print(
        f"  - block_faces: {faces_time_total:.2f} s "
        f"({faces_time_total / n_blocks:.3f} s/block)"
    )
    print(
        f"  - bounding_boxes: {boxes_time_total:.2f} s "
        f"({boxes_time_total / n_blocks:.3f} s/block)"
    )
    print(
        f"      * GPU upload (cp.asarray): {gpu_upload_total:.2f} s "
        f"({gpu_upload_total / n_blocks:.3f} s/block)"
    )
    print(
        f"      * cupyx.scipy.ndimage.find_objects: {gpu_find_objects_total:.2f} s "
        f"({gpu_find_objects_total / n_blocks:.3f} s/block)"
    )
    print(
        f"      * filter non-None boxes: {boxes_filter_total:.2f} s "
        f"({boxes_filter_total / n_blocks:.3f} s/block)"
    )
    print(
        f"      * translate to global slices: {boxes_translate_total:.2f} s "
        f"({boxes_translate_total / n_blocks:.3f} s/block)"
    )
    print(
        f"  - np.unique: {unique_time_total:.2f} s "
        f"({unique_time_total / n_blocks:.3f} s/block)"
    )
    print(f"determine_merge_relabeling: {t_relabel_lut_end - t_relabel_lut_start:.2f} s")
    print(f"relabel_and_write: {t_apply_end - t_apply_start:.2f} s")
    print(f"merge_all_boxes: {t_boxes_end - t_boxes_start:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the stitching phase of fishtools.segmentation.distributed.distributed_segmentation "
            "using synthetic label data."
        )
    )
    parser.add_argument(
        "--shape",
        type=_parse_triplet,
        default="32,512,512",
        help="ZYX shape of the synthetic volume (default: 32,512,512).",
    )
    parser.add_argument(
        "--blocksize",
        type=_parse_triplet,
        default="16,256,256",
        help="ZYX blocksize used for tiling (default: 16,256,256).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap in voxels along each spatial axis (default: 32).",
    )
    parser.add_argument(
        "--labels-per-block",
        type=int,
        default=32,
        help="Approximate number of synthetic objects per block (default: 32).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("tmp-workspace/profile_distributed_seg"),
        help="Base directory for synthetic Zarr stores and outputs.",
    )

    args = parser.parse_args()

    # Enable INFO-level timing logs from merge_utils (adjacent_faces, adjacency graph, etc.).
    logging.basicConfig(level=logging.INFO)

    overlap = int(args.overlap)
    if overlap < 0:
        raise SystemExit("overlap must be non-negative")

    t0 = time.perf_counter()
    temp_zarr, block_indices, block_crops = _make_synthetic_segmentation(
        base_dir=args.base_dir,
        shape=args.shape,
        blocksize=args.blocksize,
        overlap=overlap,
        labels_per_block=int(args.labels_per_block),
    )
    t1 = time.perf_counter()

    print(
        f"Synthetic segmentation_unstitched.zarr created at {temp_zarr.store} "
        f"with shape={temp_zarr.shape}, chunks={temp_zarr.chunks}, "
        f"in {t1 - t0:.2f} s"
    )

    stitched_path = args.base_dir / "segmentation_stitched.zarr"
    _profile_stitch(
        temp_zarr=temp_zarr,
        block_indices=block_indices,
        block_crops=block_crops,
        blocksize=args.blocksize,
        overlap=overlap,
        output_path=stitched_path,
    )


if __name__ == "__main__":
    main()
