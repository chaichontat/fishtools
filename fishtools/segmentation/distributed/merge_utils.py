"""
Shared merging utilities for distributed segmentation pipelines.

Provides:
- Block overlap removal and face extraction
- Face adjacency graph construction for stitching
- Label merging via union-find (connected components)
- Bounding box merging
"""

from pathlib import Path
from typing import Any

import dask.array
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import zarr
from numpy.typing import NDArray


def determine_merge_relabeling(
    block_indices: list[tuple[int, ...]],
    faces: list[list[NDArray[Any]]],
    used_labels: NDArray[Any],
    *,
    compact: bool = False,
) -> NDArray[np.uint32]:
    """
    Determine boundary segment mergers and return a relabeling lookup table.

    Parameters
    ----------
    block_indices
        Block grid indices corresponding to ``faces``.
    faces
        Boundary faces per block (from ``block_faces``).
    used_labels
        1D array of labels present in the volume.
    compact
        When True, compacts labels to the contiguous range [0..N]; when False,
        keeps the original label IDs (unused labels set to 0).
    """
    used_labels = used_labels.astype(int)
    label_range = int(np.max(used_labels)) if len(used_labels) > 0 else 0

    if label_range == 0:
        return np.array([0], dtype=np.uint32)

    faces_paired = adjacent_faces(block_indices, faces)
    label_groups = block_face_adjacency_graph(faces_paired, label_range)
    new_labeling = scipy.sparse.csgraph.connected_components(label_groups, directed=False)[1]

    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0

    if compact:
        unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
        new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]

    return new_labeling.astype(np.uint32)


def relabel_and_write(
    temp_zarr: zarr.Array,
    new_labeling_path: Path | str,
    write_path: Path | str,
) -> None:
    """
    Apply a relabeling LUT (saved to disk) across a temporary zarr store and write output.
    """
    write_path = Path(write_path)
    segmentation_da = dask.array.from_zarr(temp_zarr)
    relabeled = dask.array.map_blocks(
        lambda block: np.load(new_labeling_path)[block],
        segmentation_da,
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
    write_path.parent.mkdir(parents=True, exist_ok=True)
    dask.array.to_zarr(relabeled, str(write_path), overwrite=True)


def get_block_crops(
    shape: tuple[int, ...],
    blocksize: np.ndarray,
    overlap: int,
    mask: NDArray[Any] | None,
) -> tuple[list[tuple[int, ...]], list[tuple[slice, ...]]]:
    """Given a voxel grid shape, blocksize, and overlap size, construct slice tuples per block."""
    blocksize = np.array(blocksize)
    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def get_nblocks(shape: tuple[int, ...], blocksize: np.ndarray) -> NDArray[np.int_]:
    """Number of blocks along each axis."""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def block_faces(segmentation: NDArray[Any]) -> list[NDArray[Any]]:
    """Extract start/end faces along each axis for a segmented block."""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None)] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None)] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


def bounding_boxes_in_global_coordinates(
    segmentation: NDArray[Any], crop: tuple[slice, ...]
) -> list[tuple[slice, ...]]:
    """Compute bounding boxes for all segments in global coordinates."""
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]
    translate = lambda a, b: slice(a.start + b.start, a.start + b.stop)
    return [tuple(translate(a, b) for a, b in zip(crop, box)) for box in boxes]


def global_segment_ids(
    segmentation: NDArray[Any], block_index: tuple[int, ...], nblocks: NDArray[np.int_]
) -> tuple[NDArray[np.uint32], list[np.uint32]]:
    """
    Generate globally unique segment IDs by encoding block indices.
    """
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    block_token = str(np.ravel_multi_index(block_index, tuple(nblocks.tolist())))
    remap = [np.uint32(block_token + str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = np.uint32(0)
    segmentation_global = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
    return segmentation_global, remap


def remove_overlaps(
    array: NDArray[Any],
    crop: tuple[slice, ...],
    overlap: int,
    blocksize: tuple[int, ...],
) -> tuple[NDArray[Any], list[slice]]:
    """
    Remove overlapping regions from segmented block.

    Overlapping regions are added during processing to provide context for boundary
    pixels/voxels during segmentation. After segmentation is complete, these overlaps
    must be removed to prevent double-counting during block stitching.
    """
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [slice(None)] * array.ndim
            slc[axis] = slice(overlap, None)
            array = array[tuple(slc)]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlap, b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None)] * array.ndim
            slc[axis] = slice(None, blocksize[axis])
            array = array[tuple(slc)]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


def adjacent_faces(
    block_indices: list[tuple[int, ...]], faces: list[list[NDArray[Any]]]
) -> list[NDArray[Any]]:
    """Find faces which touch and pair them together in new data structure."""
    face_pairs = []
    faces_index_lookup = {a: b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2 * ax + 1]
                b = faces_index_lookup[neighbor_index][2 * ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def shrink_labels(plane: NDArray[Any], threshold: float) -> NDArray[Any]:
    """Shrink labels in plane by some distance from their boundary."""
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def block_face_adjacency_graph(faces: list[NDArray[Any]], nlabels: int) -> scipy.sparse.csr_matrix:
    """Shrink labels in face plane, then find which labels touch across the face boundary."""
    import dask_image.ndmeasure

    nlabels = int(nlabels)

    all_mappings = []
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)
        a = shrink_labels(face[sl0], 1.0)
        b = shrink_labels(face[sl1], 1.0)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = dask_image.ndmeasure._utils._label._across_block_label_grouping(face, structure)
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    return scipy.sparse.coo_matrix((v, (i, j)), shape=(nlabels + 1, nlabels + 1)).tocsr()


def merge_boxes(boxes: NDArray[Any]) -> tuple[slice, ...]:
    """Take union of two or more parallelpipeds."""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union


def merge_all_boxes(boxes: list[tuple[slice, ...]], box_ids: NDArray[Any]) -> list[tuple[slice, ...]]:
    """Merge all boxes that map to the same box_ids."""
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    box_ids = box_ids.astype(int)

    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes
