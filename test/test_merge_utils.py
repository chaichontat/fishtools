import numpy as np

from fishtools.segmentation.distributed.merge_utils import (
    adjacent_faces,
    block_face_adjacency_graph,
    block_faces,
    determine_merge_relabeling,
)


def _make_simple_faces() -> tuple[list[tuple[int, int, int]], list[list[np.ndarray]], np.ndarray]:
    """
    Build a tiny 2x1 block grid with a single touching label across the boundary.

    Blocks:
      - block (0, 0, 0): label 1 on the rightmost column
      - block (0, 1, 0): label 1 on the leftmost column
    """
    block_indices = [(0, 0, 0), (0, 1, 0)]

    seg0 = np.zeros((1, 2, 2), dtype=np.uint32)
    seg0[:, :, 1] = 1  # right column is label 1

    seg1 = np.zeros((1, 2, 2), dtype=np.uint32)
    seg1[:, :, 0] = 1  # left column is label 1

    # Faces per block as produced in the pipeline
    faces_per_block = [block_faces(seg0), block_faces(seg1)]

    # used_labels contains the global IDs; in this synthetic case we treat 1 as the only label
    used_labels = np.array([1], dtype=np.uint32)
    return block_indices, faces_per_block, used_labels


def test_block_face_adjacency_graph_uses_compact_size():
    block_indices, faces, used_labels = _make_simple_faces()
    unique_labels = np.unique(used_labels[used_labels > 0])

    faces_paired = adjacent_faces(block_indices, faces)
    graph = block_face_adjacency_graph(faces_paired, unique_labels)

    # Graph should allocate only (N_labels + 1) nodes, not based on max ID
    assert graph.shape == (unique_labels.size + 1, unique_labels.size + 1)


def test_determine_merge_relabeling_builds_dense_lut():
    block_indices, faces, used_labels = _make_simple_faces()

    lut = determine_merge_relabeling(block_indices, faces, used_labels)

    # LUT domain should be [0..max_label]
    assert lut.shape[0] == int(used_labels.max()) + 1

    # Background stays 0
    assert lut[0] == 0

    # Single label remains non-zero after merging (self-connected component)
    assert lut[1] != 0
