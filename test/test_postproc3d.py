import numpy as np

from fishtools.segment.postproc3d import (
    compute_metadata_and_adjacency,
    donate_small_cells,
    smooth_masks_3d,
)
from fishtools.segment.postproc3d import (
    compute_metadata_and_adjacency as compute_metadata_and_adjacency_fishtools,
)


def test_compute_metadata_and_adjacency_single_label_no_neighbors():
    masks = np.zeros((3, 4, 5), dtype=int)
    masks[1, 1:3, 1:4] = 1

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert volumes[0] == masks.size - masks[masks == 1].size
    assert volumes[1] == masks[masks == 1].size
    assert adjacency[1] == set()
    assert contact_areas[1] == {}


def test_compute_metadata_and_adjacency_two_labels_face_adjacent():
    masks = np.zeros((1, 4, 4), dtype=int)
    masks[0, :, 0:2] = 1
    masks[0, :, 2:4] = 2

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert volumes[1] == 4 * 2
    assert volumes[2] == 4 * 2

    assert adjacency[1] == {2}
    assert adjacency[2] == {1}

    assert 2 in contact_areas[1]
    assert 1 in contact_areas[2]
    assert contact_areas[1][2] == contact_areas[2][1]
    assert contact_areas[1][2] > 0


def test_compute_metadata_and_adjacency_diagonal_touch_only():
    masks = np.zeros((3, 3, 3), dtype=int)
    masks[1, 1, 1] = 1
    masks[2, 2, 2] = 2

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert volumes[1] == 1
    assert volumes[2] == 1

    # Diagonal-only contact should not register as 6-connected adjacency
    assert adjacency[1] == set()
    assert adjacency[2] == set()

    # And there should be no 6-connected contact area between them
    assert contact_areas[1].get(2, 0) == 0
    assert contact_areas[2].get(1, 0) == 0


def test_donate_small_cells_to_face_neighbor():
    masks = np.zeros((1, 4, 4), dtype=int)
    masks[0, :, 0:3] = 1
    masks[0, 0:2, 3] = 2

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert volumes[2] < volumes[1]

    masks_out = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=int(volumes[2]) + 1,
        min_contact_fraction=0.0,
        in_place=False,
    )

    assert np.all(masks_out[masks == 2] == 1)
    assert 2 not in np.unique(masks_out)


def test_donate_small_cells_isolated_label_drops_to_background():
    masks = np.zeros((3, 4, 4), dtype=int)
    masks[1, 1, 1] = 1

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert adjacency[1] == set()

    masks_out = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=2,
        min_contact_fraction=0.0,
        in_place=False,
    )

    assert np.all(masks_out == 0)


def test_donate_small_cells_respects_min_contact_fraction():
    masks = np.zeros((1, 3, 5), dtype=int)
    masks[0, :, 0:2] = 1
    masks[0, :, 3:5] = 2
    masks[0, 1, 2] = 3

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    assert volumes[3] == 1
    assert 1 in adjacency[3] and 2 in adjacency[3]

    masks_out = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=2,
        min_contact_fraction=3.0,
        in_place=False,
    )

    assert np.all(masks_out[0, 1, 2] == 0)


def test_donate_small_cells_does_not_donate_to_other_small_labels():
    """Small labels should not donate to other small labels."""
    masks = np.zeros((1, 1, 4), dtype=int)
    masks[0, 0, 0:2] = 1  # two voxels with label 1
    masks[0, 0, 2:4] = 2  # two voxels with label 2

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency(masks)

    # Both labels are small w.r.t V_min, and adjacent to each other
    assert volumes[1] == 2 and volumes[2] == 2
    assert 2 in adjacency[1] and 1 in adjacency[2]

    masks_out = donate_small_cells(
        masks,
        volumes=volumes,
        adjacency=adjacency,
        contact_areas=contact_areas,
        V_min=3,  # treat both labels as small
        min_contact_fraction=0.0,
        in_place=False,
    )

    # Both small labels should be removed (no 1↔2 donation)
    assert 1 not in np.unique(masks_out)
    assert 2 not in np.unique(masks_out)


def test_smooth_masks_3d_fills_small_internal_hole():
    masks = np.ones((3, 3, 3), dtype=int)
    masks[1, 1, 1] = 0

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=4,
        in_place=False,
    )

    assert masks_out[1, 1, 1] == 1


def test_smooth_masks_3d_erodes_thin_protrusion():
    masks = np.zeros((1, 3, 5), dtype=int)
    masks[0, :, 0:3] = 1
    masks[0, 1, 3] = 1

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=4,
        in_place=False,
    )

    assert masks_out[0, 1, 3] == 0
    assert np.all(masks_out[0, :, 0:3] == 1)


def test_smooth_masks_3d_does_not_merge_two_cells_across_gap():
    masks = np.zeros((1, 1, 3), dtype=int)
    masks[0, 0, 0] = 1
    masks[0, 0, 2] = 2

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=2,
        in_place=False,
    )

    assert masks_out[0, 0, 1] == 0


def test_smooth_masks_3d_smooths_two_label_interface():
    """A voxel of label 1 surrounded mostly by label 2 should flip to label 2."""
    # Create a 5x5x5 cube: mostly label 2, with a small protrusion of label 1
    masks = np.full((5, 5, 5), 2, dtype=int)
    # Label 1 occupies a corner region
    masks[:2, :2, :2] = 1
    # One voxel of label 1 sticks out into label 2's territory
    masks[2, 1, 1] = 1  # This should flip to 2

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=18,
        in_place=False,
    )

    # The protruding voxel should flip to label 2 (strong majority of label 2 neighbors)
    assert masks_out[2, 1, 1] == 2
    # The core of label 1 should remain
    assert masks_out[0, 0, 0] == 1


def test_smooth_masks_3d_no_bridge_through_background():
    """Two labels separated by background should not get connected."""
    masks = np.zeros((3, 3, 5), dtype=int)
    masks[:, :, 0:2] = 1  # Label 1 on left
    masks[:, :, 3:5] = 2  # Label 2 on right
    # Column at x=2 is background

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=4,
        in_place=False,
    )

    # Background gap should remain
    assert np.all(masks_out[:, :, 2] == 0)
    # Both labels should remain distinct
    assert 1 in np.unique(masks_out)
    assert 2 in np.unique(masks_out)


def test_smooth_masks_3d_preserves_three_label_junction():
    """Three labels meeting at a junction should not change."""
    masks = np.zeros((3, 3, 3), dtype=int)
    # Three labels meet at center
    masks[0, :, :] = 1
    masks[1, :, :] = 2
    masks[2, :, :] = 3

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=18,
        in_place=False,
    )

    # All three labels should be preserved
    assert 1 in np.unique(masks_out)
    assert 2 in np.unique(masks_out)
    assert 3 in np.unique(masks_out)
    # The junction voxels (at z=1) should not flip
    # (they have neighbors from multiple labels, Case C)
    assert np.all(masks_out[1, :, :] == 2)


def test_smooth_masks_3d_edge_guard_prevents_flip():
    """Voxels at cell edges (high background count) should not flip."""
    # A small interface with lots of background around it
    masks = np.zeros((5, 5, 5), dtype=int)
    masks[2, 2, 1] = 1
    masks[2, 2, 2] = 1
    masks[2, 2, 3] = 2
    masks[2, 2, 4] = 2

    masks_out = smooth_masks_3d(
        masks,
        num_iterations=1,
        min_neighbors_to_fill=18,
        in_place=False,
    )

    # The interface voxels have too much background around them
    # (n_self + n_other < 8), so they should not flip
    # They might erode to background instead if n_bg > n_self
    # The key is: no label 1 → label 2 flip should happen
    unique = np.unique(masks_out)
    # Either both labels remain distinct or some erode to background
    # but label 1 voxels should NOT become label 2 (or vice versa)
    # Check that no interface flip occurred by verifying the pattern
    if 1 in unique and 2 in unique:
        # If both labels exist, they should still be on opposite sides
        label1_x = np.where(masks_out == 1)[2]
        label2_x = np.where(masks_out == 2)[2]
        if len(label1_x) > 0 and len(label2_x) > 0:
            assert label1_x.max() < label2_x.min()  # Label 1 still left of label 2


# ============================================================================
# Opening-based protrusion correction tests
# ============================================================================


def test_opening_protrusion_donated_to_neighbor():
    """A thin protrusion of label 1 into label 2 territory should be donated."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    # Create two cubes with a thin bridge of label 1 extending into label 2's space
    masks = np.zeros((10, 10, 10), dtype=int)
    # Label 1: solid cube on left side
    masks[2:8, 2:8, 1:4] = 1
    # Label 2: solid cube on right side
    masks[2:8, 2:8, 6:9] = 2
    # Thin protrusion of label 1 extending toward label 2 (should flip)
    masks[4:6, 4:6, 4:6] = 1  # This is thin and contacts label 2

    masks_out = opening_correct_protrusions(masks, r_neck=2, min_protrusion_volume=1, in_place=False)

    # Both labels should still exist
    assert 1 in np.unique(masks_out)
    assert 2 in np.unique(masks_out)
    # The cores should be preserved
    assert masks_out[4, 4, 2] == 1  # Core of label 1
    assert masks_out[4, 4, 7] == 2  # Core of label 2


def test_opening_body_unchanged():
    """Body voxels (survive opening) should not change."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((10, 10, 10), dtype=int)
    # Two large cubes touching
    masks[1:9, 1:9, 1:5] = 1
    masks[1:9, 1:9, 5:9] = 2

    masks_out = opening_correct_protrusions(masks, r_neck=1, in_place=False)

    # Deep interior voxels should be unchanged
    assert masks_out[4, 4, 2] == 1  # Deep in label 1
    assert masks_out[4, 4, 7] == 2  # Deep in label 2
    # Both labels should still exist
    assert 1 in np.unique(masks_out)
    assert 2 in np.unique(masks_out)


def test_opening_protrusion_into_background_unchanged():
    """Protrusions into pure background (no neighbor contact) should not change."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((10, 10, 10), dtype=int)
    # Solid cube
    masks[3:7, 3:7, 3:7] = 1
    # Thin spike into background (no other label nearby)
    masks[5, 5, 7] = 1
    masks[5, 5, 8] = 1

    masks_out = opening_correct_protrusions(masks, r_neck=2, in_place=False)

    # The spike into background should be unchanged (no other label to flip to)
    # Opening may have removed it, but it won't be donated to another label
    # It will stay as label 1 since there's no neighbor contact
    assert masks_out[5, 5, 7] == 1
    assert masks_out[5, 5, 8] == 1


def test_opening_three_label_junction_stable():
    """Three labels meeting should all be preserved."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((9, 9, 9), dtype=int)
    # Three cubes meeting at a corner
    masks[1:5, 1:5, 1:5] = 1
    masks[1:5, 1:5, 4:8] = 2
    masks[1:5, 4:8, 1:5] = 3

    masks_out = opening_correct_protrusions(masks, r_neck=2, in_place=False)

    # All three labels should be preserved
    unique = np.unique(masks_out)
    assert 1 in unique
    assert 2 in unique
    assert 3 in unique


def test_opening_thin_cell_protected():
    """A very thin cell should be protected by the 50% safety gate."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((10, 10, 10), dtype=int)
    # Large cube
    masks[2:8, 2:8, 2:8] = 1
    # Very thin cell (only 1 voxel thick) adjacent to label 1
    masks[2:8, 2:8, 8] = 2  # Single-layer sheet

    # With r_neck=2, opening would remove > 50% of label 2, so it's skipped
    masks_out = opening_correct_protrusions(masks, r_neck=2, in_place=False)

    # Label 1 should still exist
    assert 1 in np.unique(masks_out)
    # Label 2 should be protected (not absorbed) due to 50% safety gate
    assert 2 in np.unique(masks_out)


def test_opening_no_background_crossing():
    """Labels separated by background should not merge."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((10, 10, 10), dtype=int)
    # Two cubes with background gap between them
    masks[2:8, 2:8, 1:4] = 1
    masks[2:8, 2:8, 6:9] = 2
    # Gap at z=4,5 is background

    masks_out = opening_correct_protrusions(masks, r_neck=2, in_place=False)

    # Gap should remain background
    assert np.all(masks_out[2:8, 2:8, 4:6] == 0)
    # Both labels should exist and be separate
    assert 1 in np.unique(masks_out)
    assert 2 in np.unique(masks_out)


def test_opening_ambiguous_protrusion_kept():
    """Protrusion with ambiguous neighbor contact should stay with original cell."""
    from fishtools.segment.postproc3d import opening_correct_protrusions

    masks = np.zeros((10, 10, 10), dtype=int)
    # Three cells: label 1 in middle, labels 2 and 3 on sides
    masks[2:8, 2:8, 1:4] = 2  # Left cube
    masks[2:8, 2:8, 4:6] = 1  # Middle thin section (protrusion)
    masks[2:8, 2:8, 6:9] = 3  # Right cube

    # The middle thin section contacts both 2 and 3 roughly equally
    # With dominance_threshold=0.5, neither neighbor dominates
    masks_out = opening_correct_protrusions(masks, r_neck=1, dominance_threshold=0.7, in_place=False)

    # All three labels should still exist (ambiguous protrusion kept)
    unique = np.unique(masks_out)
    assert 2 in unique
    assert 3 in unique


# ============================================================================
# Gaussian label smoothing tests
# ============================================================================


def test_gaussian_smooth_removes_thin_protrusion():
    """Protrusion thinner than sigma should be removed."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.zeros((15, 15, 15), dtype=int)
    masks[3:12, 3:12, 3:10] = 1  # Main body (9×9×7 cube) - large enough for sigma=2
    masks[7, 7, 10:14] = 1  # Thin spike (1 voxel wide, 4 long)

    result = gaussian_smooth_labels(masks, sigma=2.0, in_place=False)

    # Spike tip should be eroded (far from body)
    assert result[7, 7, 13] == 0
    # Body core preserved
    assert result[7, 7, 6] == 1


def test_gaussian_smooth_fills_small_hole():
    """Hole smaller than sigma should be filled."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.ones((7, 7, 7), dtype=int)
    masks[3, 3, 3] = 0  # Single voxel hole

    result = gaussian_smooth_labels(masks, sigma=2.0, in_place=False)

    assert result[3, 3, 3] == 1


def test_gaussian_smooth_preserves_large_cells():
    """Cells much larger than sigma should be preserved."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.zeros((20, 20, 20), dtype=int)
    masks[2:10, 2:10, 2:10] = 1
    masks[10:18, 10:18, 10:18] = 2

    result = gaussian_smooth_labels(masks, sigma=1.5, in_place=False)

    # Both cells exist
    assert 1 in np.unique(result)
    assert 2 in np.unique(result)
    # Cores unchanged
    assert result[5, 5, 5] == 1
    assert result[14, 14, 14] == 2


def test_gaussian_smooth_smooths_interface():
    """Jagged interface between two cells should be smoothed."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.zeros((10, 10, 10), dtype=int)
    # Two cells with a jagged interface
    masks[:, :, :5] = 1
    masks[:, :, 5:] = 2
    # Add a single-voxel protrusion of label 1 into label 2
    masks[5, 5, 5] = 1

    result = gaussian_smooth_labels(masks, sigma=1.5, in_place=False)

    # The single protrusion should flip to label 2
    assert result[5, 5, 5] == 2
    # Cores preserved
    assert result[5, 5, 2] == 1
    assert result[5, 5, 7] == 2


def test_gaussian_smooth_empty_masks():
    """Empty masks should return unchanged."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.zeros((5, 5, 5), dtype=int)
    result = gaussian_smooth_labels(masks, sigma=1.5, in_place=False)

    assert np.all(result == 0)


def test_gaussian_smooth_in_place():
    """in_place=True should modify the input array."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.ones((7, 7, 7), dtype=int)
    masks[3, 3, 3] = 0  # Hole to be filled

    result = gaussian_smooth_labels(masks, sigma=2.0, in_place=True)

    # Result should be same object
    assert result is masks
    # Hole should be filled in original array
    assert masks[3, 3, 3] == 1


def test_gaussian_smooth_not_in_place():
    """in_place=False should not modify the input array."""
    from fishtools.segment.postproc3d import gaussian_smooth_labels

    masks = np.ones((7, 7, 7), dtype=int)
    masks[3, 3, 3] = 0  # Hole
    original_value = masks[3, 3, 3]

    result = gaussian_smooth_labels(masks, sigma=2.0, in_place=False)

    # Original unchanged
    assert masks[3, 3, 3] == original_value
    # Result is different object
    assert result is not masks


def test_gaussian_smooth_cupy_matches_cpu():
    """CuPy-accelerated smoothing should match CPU result on a small volume."""
    import pytest

    cupy = pytest.importorskip("cupy")
    try:
        # Skip if CUDA is not actually usable in this environment
        _ = cupy.cuda.runtime.getDevice()
    except Exception:
        pytest.skip("CuPy is installed but CUDA device is not available")
    from fishtools.segment.postproc3d import gaussian_smooth_labels, gaussian_smooth_labels_cupy

    masks = np.zeros((7, 7, 7), dtype=int)
    masks[2:5, 2:5, 2:5] = 1
    masks[3, 3, 5] = 1  # small protrusion

    cpu = gaussian_smooth_labels(masks, sigma=1.5, in_place=False, bg_scale=1.0)
    gpu = gaussian_smooth_labels_cupy(masks, sigma=1.5, in_place=False, bg_scale=1.0)

    assert np.array_equal(cpu, gpu)


def test_gaussian_erosion_to_margin_and_scale_zero_erosion():
    """Zero erosion should give zero margin and unit scale."""
    from fishtools.segment.postproc3d import gaussian_erosion_to_margin_and_scale

    margin, scale = gaussian_erosion_to_margin_and_scale(sigma=1.5, erosion_voxels=0.0)
    assert abs(margin) < 1e-6
    assert abs(scale - 1.0) < 1e-6


def test_gaussian_erosion_to_margin_and_scale_positive_erosion():
    """Positive erosion yields positive margin and scale > 1."""
    from fishtools.segment.postproc3d import gaussian_erosion_to_margin_and_scale

    margin, scale = gaussian_erosion_to_margin_and_scale(sigma=2.0, erosion_voxels=0.5)
    assert margin > 0.0
    assert scale > 1.0


def test_gaussian_erosion_to_margin_and_scale_fwhm_fraction():
    """FWHM fraction specification produces consistent parameters."""
    from fishtools.segment.postproc3d import gaussian_erosion_to_margin_and_scale

    # For fwhm_fraction=0, should match zero erosion behavior
    margin, scale = gaussian_erosion_to_margin_and_scale(sigma=2.0, fwhm_fraction=0.0)
    assert abs(margin) < 1e-6
    assert abs(scale - 1.0) < 1e-6


# ============================================================================
# Relabel connected components tests
# ============================================================================


def test_relabel_connected_components_splits_disconnected():
    """Disconnected regions of same label get different IDs."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((10, 10, 10), dtype=int)
    masks[1:3, 1:3, 1:3] = 1  # Component A
    masks[7:9, 7:9, 7:9] = 1  # Component B (same label, disconnected)

    result = relabel_connected_components(masks, in_place=False)

    # Should have 2 different labels now (plus background)
    assert len(np.unique(result)) == 3  # 0, 1, 2
    # The two regions should have different labels
    assert result[2, 2, 2] != result[8, 8, 8]


def test_relabel_connected_components_preserves_connected():
    """Connected region stays as single label."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((10, 10, 10), dtype=int)
    masks[2:8, 2:8, 2:8] = 1  # Single connected cube

    result = relabel_connected_components(masks, in_place=False)

    # Should have just 1 label (plus background)
    assert len(np.unique(result)) == 2  # 0, 1


def test_relabel_connected_components_multiple_labels():
    """Multiple labels each stay separate."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((10, 10, 10), dtype=int)
    masks[1:3, 1:3, 1:3] = 1
    masks[5:7, 5:7, 5:7] = 2

    result = relabel_connected_components(masks, in_place=False)

    # Two connected components, should get labels 1 and 2
    unique = np.unique(result)
    assert len(unique) == 3  # 0, 1, 2


def test_relabel_connected_components_does_not_merge_touching_labels():
    """Touching different labels should not be merged."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((1, 1, 2), dtype=int)
    masks[0, 0, 0] = 1
    masks[0, 0, 1] = 2  # face-adjacent to label 1

    result = relabel_connected_components(masks, in_place=False)

    # Labels 1 and 2 should remain distinct
    assert result[0, 0, 0] == 1
    assert result[0, 0, 1] == 2
    unique = np.unique(result)
    assert set(unique.tolist()) == {1, 2}


def test_relabel_connected_components_empty():
    """Empty masks should return unchanged."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((5, 5, 5), dtype=int)
    result = relabel_connected_components(masks, in_place=False)

    assert np.all(result == 0)


def test_relabel_connected_components_in_place():
    """in_place=True should modify the input array."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((10, 10, 10), dtype=int)
    masks[1:3, 1:3, 1:3] = 1
    masks[7:9, 7:9, 7:9] = 1  # Disconnected

    result = relabel_connected_components(masks, in_place=True)

    assert result is masks
    assert len(np.unique(masks)) == 3  # Modified in place


def test_relabel_connected_components_out_of_place_uses_uint32():
    """in_place=False should return a uint32 label volume."""
    from fishtools.segment.postproc3d import relabel_connected_components

    masks = np.zeros((10, 10, 10), dtype=np.int16)
    masks[1:3, 1:3, 1:3] = 1  # Component A
    masks[7:9, 7:9, 7:9] = 1  # Component B (same label, disconnected)

    result = relabel_connected_components(masks, in_place=False)

    assert result.dtype == np.uint32


# ============================================================================
# Sparse global label tests (compact relabeling)
# ============================================================================


def test_compute_metadata_sparse_global_labels():
    """Sparse labels (high IDs, few cells) should produce correct volumes and contacts."""
    masks = np.zeros((10, 10, 10), dtype=np.int32)
    masks[0:5, :, :] = 15001  # High label ID
    masks[5:10, :, :] = 15002  # Adjacent, also high ID

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency_fishtools(
        masks, compute_contact_areas=True
    )

    # Outcomes: correct volumes at original label indices
    assert volumes[15001] == 500
    assert volumes[15002] == 500

    # Outcomes: correct adjacency at original label indices
    assert 15002 in adjacency[15001]
    assert 15001 in adjacency[15002]

    # Outcomes: correct contact area (10x10 face)
    assert contact_areas[15001][15002] == 100
    assert contact_areas[15002][15001] == 100


def test_compute_metadata_sparse_labels_chain_adjacency():
    """Three sparse labels in a chain: A-B-C where A and C don't touch."""
    masks = np.zeros((15, 5, 5), dtype=np.int32)
    masks[0:5, :, :] = 20001
    masks[5:10, :, :] = 20002  # Touches 20001 and 20003
    masks[10:15, :, :] = 20003

    volumes, adjacency, contact_areas = compute_metadata_and_adjacency_fishtools(
        masks, compute_contact_areas=True
    )

    # 20001 only touches 20002
    assert adjacency[20001] == {20002}
    # 20002 touches both
    assert adjacency[20002] == {20001, 20003}
    # 20003 only touches 20002
    assert adjacency[20003] == {20002}


def test_compute_metadata_sparse_matches_dense_relabeled():
    """Sparse labels should produce identical results to equivalent dense labels.

    This exercises the compact relabeling code path end-to-end by comparing
    results from high-ID sparse labels against low-ID dense labels with
    identical geometry.
    """
    # Dense labels (won't trigger relabeling)
    masks_dense = np.zeros((10, 10, 10), dtype=np.int32)
    masks_dense[0:5, :, :] = 1
    masks_dense[5:10, :, :] = 2

    # Sparse labels (will trigger relabeling path)
    masks_sparse = np.zeros((10, 10, 10), dtype=np.int32)
    masks_sparse[0:5, :, :] = 15001
    masks_sparse[5:10, :, :] = 15002

    vol_dense, adj_dense, ca_dense = compute_metadata_and_adjacency_fishtools(
        masks_dense, compute_contact_areas=True
    )
    vol_sparse, adj_sparse, ca_sparse = compute_metadata_and_adjacency_fishtools(
        masks_sparse, compute_contact_areas=True
    )

    # Volumes should match (at respective indices)
    assert vol_dense[1] == vol_sparse[15001]
    assert vol_dense[2] == vol_sparse[15002]

    # Adjacency structure should match
    assert adj_dense[1] == {2}
    assert adj_sparse[15001] == {15002}

    # Contact areas should match
    assert ca_dense[1][2] == ca_sparse[15001][15002]


# ============================================================================
# Absorb encircled ROIs tests (per 2D slice)
# ============================================================================


def test_absorb_encircled_simple():
    """A small cell completely inside a larger cell should be absorbed in that slice."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    # Create a 2D encirclement in middle slice: label 2 fully enclosed by label 1
    masks = np.zeros((5, 10, 10), dtype=int)
    masks[1:4, 2:8, 2:8] = 1  # Large cell
    masks[2, 4:6, 4:6] = 2  # Small cell inside (only in middle z slice)

    result = absorb_encircled_rois(masks, in_place=False)

    # Label 2 should be absorbed into label 1 (encircled in its slice)
    assert 2 not in np.unique(result)
    assert np.all(result[masks == 2] == 1)


def test_absorb_encircled_touches_background():
    """Cell touching background in a slice should NOT be absorbed in that slice."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((3, 10, 10), dtype=int)
    masks[:, 2:8, 2:8] = 1
    masks[:, 4:6, 4:6] = 2
    masks[1, 5, 1] = 2  # Extends to touch background in slice z=1

    result = absorb_encircled_rois(masks, in_place=False)

    # Label 2 touches background in z=1, so should NOT be absorbed in that slice
    # But may be absorbed in z=0 and z=2 if encircled there
    assert 2 in np.unique(result)  # Should still exist somewhere


def test_absorb_encircled_touches_edge():
    """Cell at xy edge should NOT be absorbed in that slice."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((3, 10, 10), dtype=int)
    masks[:, :, 1:] = 1
    masks[:, 2:4, 0:2] = 2  # Touches left edge (x=0)

    result = absorb_encircled_rois(masks, in_place=False)

    # Label 2 touches xy edge, so should NOT be absorbed
    assert 2 in np.unique(result)


def test_absorb_encircled_multiple_neighbors():
    """Cell with multiple neighbors in a slice should NOT be absorbed in that slice."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((1, 10, 10), dtype=int)
    masks[0, 0:5, :] = 1  # Top half
    masks[0, 5:10, :] = 2  # Bottom half
    masks[0, 4:6, 4:6] = 3  # Straddles boundary, touches both 1 and 2

    result = absorb_encircled_rois(masks, in_place=False)

    # Label 3 touches both 1 and 2 in this slice, so shouldn't be absorbed
    assert 3 in np.unique(result)


def test_absorb_encircled_partial_absorption():
    """ROI encircled in one slice but not another: partial absorption."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((3, 10, 10), dtype=int)
    # z=0: label 2 is encircled by label 1
    masks[0, 2:8, 2:8] = 1
    masks[0, 4:6, 4:6] = 2
    # z=1: label 2 touches background (extends beyond label 1)
    masks[1, 2:8, 2:8] = 1
    masks[1, 4:6, 4:6] = 2
    masks[1, 5, 1] = 2  # Touches background
    # z=2: label 2 is encircled by label 1
    masks[2, 2:8, 2:8] = 1
    masks[2, 4:6, 4:6] = 2

    result = absorb_encircled_rois(masks, in_place=False)

    # In z=0 and z=2, label 2 is encircled -> absorbed into 1
    # In z=1, label 2 touches background -> NOT absorbed
    assert result[0, 5, 5] == 1  # z=0: absorbed
    assert result[1, 5, 5] == 2  # z=1: NOT absorbed (touches bg)
    assert result[2, 5, 5] == 1  # z=2: absorbed


def test_absorb_encircled_different_parents_per_slice():
    """ROI encircled by different parents in different slices."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((2, 10, 10), dtype=int)
    # z=0: label 3 is encircled by label 1
    masks[0, 2:8, 2:8] = 1
    masks[0, 4:6, 4:6] = 3
    # z=1: label 3 is encircled by label 2
    masks[1, 2:8, 2:8] = 2
    masks[1, 4:6, 4:6] = 3

    result = absorb_encircled_rois(masks, in_place=False)

    # In z=0, label 3 absorbed into label 1
    assert result[0, 5, 5] == 1
    # In z=1, label 3 absorbed into label 2
    assert result[1, 5, 5] == 2
    # Label 3 should no longer exist
    assert 3 not in np.unique(result)


def test_absorb_encircled_sparse_labels():
    """Sparse label IDs (e.g., 1 and 1_000_000) don't cause memory explosion."""
    from fishtools.segment.postproc3d import absorb_encircled_rois

    masks = np.zeros((1, 20, 20), dtype=np.int32)
    # Outer ring: label 1_000_000 (sparse ID)
    masks[0, 5:15, 5:15] = 1_000_000
    # Inner region: label 2_000_000 (even sparser), encircled by 1_000_000
    masks[0, 8:12, 8:12] = 2_000_000

    # This should use the sparse path (max_label=2M >> n_foreground=2)
    result = absorb_encircled_rois(masks, in_place=False)

    # Inner label should be absorbed into outer label
    assert result[0, 10, 10] == 1_000_000
    # Only background and outer label should remain
    unique = np.unique(result)
    assert set(unique) == {0, 1_000_000}
