"""
Tests for tiled GPU correction in n4_tiled.py.

These tests verify that the tiled implementation produces identical results
to the original full-plane _correct_plane_gpu() from n4.py.

Note: Tests with unsharp mask require real GPU and cucim. They are skipped
when running with the test stubs (conftest.py provides numpy-backed stubs).
"""

from __future__ import annotations

import numpy as np
import pytest

# Import GPU libraries - tests will be skipped if not available
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Check if we have real cucim (not the conftest stub)
# Real cucim has __file__; the stub is a types.ModuleType without it
HAS_REAL_CUCIM = False
try:
    import cucim

    HAS_REAL_CUCIM = hasattr(cucim, "__file__") and cucim.__file__ is not None
except ImportError:
    pass

# Import the modules under test
if HAS_CUPY:
    from fishtools.preprocess.n4 import _correct_plane_gpu
    from fishtools.preprocess.n4_tiled import (
        DEFAULT_TILE_SIZE,
        correct_plane_gpu_tiled,
        get_auto_tile_size,
    )


pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")

# Marker for tests that require real cucim (not stub)
requires_cucim = pytest.mark.skipif(
    not HAS_REAL_CUCIM, reason="Requires real cucim (not test stub) for unsharp mask"
)


class TestGetAutoTileSize:
    """Tests for tile size auto-detection logic."""

    def test_explicit_tile_size(self):
        """Explicit tile size should be returned unchanged."""
        assert get_auto_tile_size(2048, 2048, 512) == 512
        assert get_auto_tile_size(8192, 8192, 1024) == 1024

    def test_disabled_tiling(self):
        """tile_size=0 should disable tiling (return max dimension)."""
        assert get_auto_tile_size(2048, 2048, 0) == 2048
        assert get_auto_tile_size(1000, 2000, 0) == 2000

    def test_auto_small_image(self):
        """Small images (< threshold) should not be tiled."""
        assert get_auto_tile_size(2048, 2048, None) == 2048
        assert get_auto_tile_size(4096, 4096, None) == 4096

    def test_auto_large_image(self):
        """Large images (> threshold) should be tiled with default size."""
        assert get_auto_tile_size(8192, 8192, None) == DEFAULT_TILE_SIZE
        assert get_auto_tile_size(4097, 4097, None) == DEFAULT_TILE_SIZE
        # One dimension over threshold
        assert get_auto_tile_size(2048, 8192, None) == DEFAULT_TILE_SIZE


class TestTiledEquivalence:
    """Tests verifying tiled output matches full-plane output."""

    @pytest.fixture
    def random_plane(self):
        """Generate a random test plane with realistic values."""
        np.random.seed(42)
        return (np.random.rand(2048, 2048).astype(np.float32) * 1000 + 100).astype(np.float32)

    @pytest.fixture
    def random_field(self):
        """Generate a random correction field (positive values near 1.0)."""
        np.random.seed(43)
        return (np.ones((2048, 2048), dtype=np.float32) + np.random.rand(2048, 2048) * 0.3).astype(np.float32)

    def test_tiled_matches_full_no_unsharp(self, random_plane, random_field):
        """Tiled processing without unsharp mask should be bit-exact."""
        result_full = _correct_plane_gpu(
            random_plane,
            field_cpu=random_field,
            use_unsharp_mask=False,
            mask_cpu=None,
        )
        result_tiled = correct_plane_gpu_tiled(
            random_plane,
            field_cpu=random_field,
            use_unsharp_mask=False,
            mask_cpu=None,
            tile_size=512,
        )

        # Without unsharp mask, should be exactly equal
        np.testing.assert_array_equal(result_full, result_tiled)

    @requires_cucim
    def test_tiled_matches_full_with_unsharp(self, random_plane, random_field):
        """Tiled processing with unsharp mask should match within tolerance."""
        mask = random_plane > 200

        result_full = _correct_plane_gpu(
            random_plane,
            field_cpu=random_field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            random_plane,
            field_cpu=random_field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=512,
        )

        # With unsharp mask, allow small floating-point tolerance
        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shape,tile_size",
        [
            ((1000, 1000), 512),  # Not divisible
            ((513, 513), 512),  # Just over one tile
            ((100, 100), 512),  # Smaller than tile
            ((1023, 1025), 512),  # Asymmetric, not divisible
            ((2000, 3000), 512),  # Rectangular
        ],
    )
    def test_arbitrary_dimensions_no_unsharp(self, shape, tile_size):
        """Handle images not evenly divisible by tile size (no unsharp)."""
        np.random.seed(44)
        H, W = shape
        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = (np.ones((H, W), dtype=np.float32) + np.random.rand(H, W) * 0.2).astype(np.float32)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=False,
            mask_cpu=None,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=False,
            mask_cpu=None,
            tile_size=tile_size,
        )

        # Without unsharp, should be exactly equal
        np.testing.assert_array_equal(result_full, result_tiled)

    @requires_cucim
    @pytest.mark.parametrize(
        "shape,tile_size",
        [
            ((1000, 1000), 512),  # Not divisible
            ((513, 513), 512),  # Just over one tile
            ((100, 100), 512),  # Smaller than tile
            ((1023, 1025), 512),  # Asymmetric, not divisible
            ((2000, 3000), 512),  # Rectangular
        ],
    )
    def test_arbitrary_dimensions_with_unsharp(self, shape, tile_size):
        """Handle images not evenly divisible by tile size."""
        np.random.seed(44)
        H, W = shape
        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = (np.ones((H, W), dtype=np.float32) + np.random.rand(H, W) * 0.2).astype(np.float32)
        mask = plane > 200

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=tile_size,
        )

        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)


class TestMaskBoundaries:
    """Tests for correct handling of mask transitions at tile boundaries."""

    @requires_cucim
    def test_mask_transition_at_tile_boundary(self):
        """Mask transitions at tile boundaries shouldn't cause artifacts."""
        np.random.seed(45)
        H, W = 1024, 1024
        tile_size = 512

        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = np.ones((H, W), dtype=np.float32) * 1.1

        # Mask transition exactly at tile boundary (x=512)
        mask = np.zeros((H, W), dtype=bool)
        mask[:, :512] = True  # Left half only

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=tile_size,
        )

        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)

    @requires_cucim
    def test_mask_transition_at_y_boundary(self):
        """Mask transitions at Y tile boundaries work correctly."""
        np.random.seed(46)
        H, W = 1024, 1024
        tile_size = 512

        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = np.ones((H, W), dtype=np.float32) * 1.1

        # Mask transition exactly at y=512
        mask = np.zeros((H, W), dtype=bool)
        mask[:512, :] = True  # Top half only

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=tile_size,
        )

        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)


class TestEdgeContinuity:
    """Tests for seamless results at tile boundaries."""

    @requires_cucim
    def test_no_seams_on_smooth_gradient(self):
        """Smooth gradients should have no visible seams at tile boundaries."""
        H, W = 2048, 2048
        tile_size = 512

        # Create smooth horizontal gradient
        plane = np.tile(np.linspace(100, 1100, W, dtype=np.float32), (H, 1))
        field = np.ones((H, W), dtype=np.float32)
        mask = np.ones((H, W), dtype=bool)

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=tile_size,
        )

        # Check continuity at tile boundaries
        for x in range(tile_size, W, tile_size):
            left = result_tiled[:, x - 1]
            right = result_tiled[:, x]
            # The difference at boundary should match the expected gradient
            expected_diff = result_tiled[:, x - 2] - result_tiled[:, x - 1]
            actual_diff = left - right
            # Allow small tolerance for floating-point
            np.testing.assert_allclose(
                np.abs(actual_diff),
                np.abs(expected_diff),
                rtol=0.1,
                err_msg=f"Seam detected at x={x}",
            )

        # Overall results should match
        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)


class TestTileSizeVariations:
    """Tests for different tile sizes."""

    @requires_cucim
    @pytest.mark.parametrize("tile_size", [256, 512, 1024, 2048])
    def test_various_tile_sizes(self, tile_size):
        """Different tile sizes should all produce equivalent results."""
        np.random.seed(47)
        H, W = 2048, 2048

        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = (np.ones((H, W), dtype=np.float32) + np.random.rand(H, W) * 0.2).astype(np.float32)
        mask = plane > 200

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=mask,
            tile_size=tile_size,
        )

        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)


class TestNoMask:
    """Tests for processing without explicit mask."""

    @requires_cucim
    def test_no_mask_provided(self):
        """Processing without mask should work (falls back to >0 mask)."""
        np.random.seed(48)
        H, W = 1024, 1024

        plane = (np.random.rand(H, W).astype(np.float32) * 1000 + 100).astype(np.float32)
        field = (np.ones((H, W), dtype=np.float32) + np.random.rand(H, W) * 0.2).astype(np.float32)

        field_gpu = cp.asarray(field)

        result_full = _correct_plane_gpu(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=None,
        )
        result_tiled = correct_plane_gpu_tiled(
            plane,
            field_cpu=field,
            use_unsharp_mask=True,
            mask_cpu=None,
            tile_size=512,
        )

        np.testing.assert_allclose(result_full, result_tiled, rtol=1e-5, atol=1e-5)
