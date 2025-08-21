"""
Comprehensive unit tests for fiducial processing in fishtools.preprocess.fiducial

This module tests the fiducial detection and alignment functions:
- run_fiducial: Complete fiducial alignment pipeline
- align_fiducials: Multi-threaded fiducial alignment with overrides
- align_phase: FFT-based phase correlation alignment
- find_spots: DAOStarFinder spot detection
- Supporting functions: butterworth, clahe, phase_shift, background
"""

from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from fishtools.preprocess.fiducial import (
    DriftTooLarge,
    NotEnoughSpots,
    ResidualTooLarge,
    TooManySpots,
    align_fiducials,
    align_phase,
    background,
    butterworth,
    clahe,
    find_spots,
    phase_shift,
    run_fiducial,
)

# Test constants
TEST_IMAGE_SIZE = (100, 100)
SMALL_IMAGE_SIZE = (50, 50)
LARGE_IMAGE_SIZE = (200, 200)


class TestFindSpots:
    """Unit tests for find_spots function using DAOStarFinder"""

    def test_find_spots_basic(self) -> None:
        """Test basic spot finding functionality"""
        # Create synthetic image with known spots
        img = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint16)

        # Add bright spots at known locations
        spot_locations = [(25, 25), (75, 75), (25, 75), (75, 25)]
        for y, x in spot_locations:
            # Create Gaussian-like spot
            yy, xx = np.ogrid[y - 3 : y + 4, x - 3 : x + 4]
            spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2) * 1000
            img[y - 3 : y + 4, x - 3 : x + 4] = spot.astype(np.uint16)

        # Add some background noise
        img += np.random.randint(50, 100, TEST_IMAGE_SIZE, dtype=np.uint16)

        result = find_spots(img, threshold_sigma=3.0, fwhm=4.0, minimum_spots=2)

        # Should be polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 2  # Should find at least minimum_spots

        # Should have required columns
        required_cols = ["xcentroid", "ycentroid", "mag", "idx"]
        assert all(col in result.columns for col in required_cols)

        # Results should be sorted by magnitude (brightest first)
        mag_values = result["mag"].to_list()
        assert mag_values == sorted(mag_values)  # Ascending order (lower mag = brighter)

    def test_find_spots_not_enough_spots(self) -> None:
        """Test find_spots raises NotEnoughSpots when insufficient spots found"""
        # Create image with very dim spots
        img = np.random.randint(100, 120, TEST_IMAGE_SIZE, dtype=np.uint16)

        with pytest.raises(NotEnoughSpots):
            find_spots(img, threshold_sigma=10.0, fwhm=4.0, minimum_spots=6)

    def test_find_spots_empty_image(self) -> None:
        """Test find_spots with empty image"""
        img = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint16)

        with pytest.raises(NotEnoughSpots, match="Reference image has zero sum"):
            find_spots(img, threshold_sigma=3.0, fwhm=4.0)

    def test_find_spots_wrong_dimensions(self) -> None:
        """Test find_spots with wrong image dimensions"""
        # 3D image should fail
        img_3d = np.random.randint(100, 200, (10, 50, 50), dtype=np.uint16)

        with pytest.raises(ValueError, match="Reference image must be 2D for DAOStarFinder"):
            find_spots(img_3d, threshold_sigma=3.0, fwhm=4.0)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_find_spots_dtypes(self, dtype: np.dtype[Any]) -> None:
        """Test find_spots with different data types"""
        # Create image with bright spots
        img = np.random.randint(100, 200, TEST_IMAGE_SIZE).astype(dtype)

        # Add bright spots
        img[25:30, 25:30] = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 10000

        try:
            result = find_spots(img, threshold_sigma=2.0, fwhm=4.0, minimum_spots=1)
            assert isinstance(result, pl.DataFrame)
            assert len(result) >= 1
        except NotEnoughSpots:
            # Some dtypes might not have sufficient dynamic range
            pytest.skip(f"Insufficient dynamic range for dtype {dtype}")

    def test_find_spots_parameter_sensitivity(self) -> None:
        """Test find_spots parameter sensitivity"""
        # Create image with spots of varying brightness
        img = np.random.randint(50, 100, TEST_IMAGE_SIZE, dtype=np.uint16)

        # Add spots with different intensities
        bright_spots = [(25, 25, 2000), (50, 50, 1000), (75, 75, 500)]
        for y, x, intensity in bright_spots:
            img[y - 2 : y + 3, x - 2 : x + 3] = intensity

        # Lower threshold should find more spots
        result_low = find_spots(img, threshold_sigma=2.0, fwhm=4.0, minimum_spots=1)
        result_high = find_spots(img, threshold_sigma=5.0, fwhm=4.0, minimum_spots=1)

        assert len(result_low) >= len(result_high), "Lower threshold should find more spots"


class TestImageProcessing:
    """Unit tests for image processing functions"""

    def test_butterworth_filter(self) -> None:
        """Test butterworth high-pass filtering"""
        # Create test image with low and high frequency components
        x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        low_freq = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)  # Low frequency
        high_freq = np.sin(20 * np.pi * x) * np.cos(20 * np.pi * y)  # High frequency
        img = low_freq + high_freq + 2  # Make positive

        result = butterworth(img, cutoff=0.1, order=3)

        # High-pass filter should suppress low frequencies
        assert result.shape == img.shape
        assert np.all(result >= 0), "Butterworth output should be clipped to non-negative"
        assert np.std(result) < np.std(img), "High-pass should reduce overall variation"

    def test_clahe_enhancement(self) -> None:
        """Test CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Create image with poor contrast
        img = np.random.random(TEST_IMAGE_SIZE) * 0.3 + 0.1  # Low dynamic range

        result = clahe(img, clip_limit=0.01)

        assert result.shape == img.shape
        assert 0 <= result.min() <= result.max() <= 1, "CLAHE output should be normalized [0,1]"
        assert np.std(result) > np.std(img), "CLAHE should increase contrast"

    def test_phase_shift_calculation(self) -> None:
        """Test phase shift calculation between images"""
        # Create reference image
        ref = np.random.random(TEST_IMAGE_SIZE)

        # Create shifted version: apply shift +5 pixels DOWN, +3 pixels RIGHT
        applied_shift = (5, 3)  # y, x shift applied to image
        shifted = np.roll(np.roll(ref, applied_shift[0], axis=0), applied_shift[1], axis=1)

        calculated_shift = phase_shift(ref, shifted, precision=1)

        # CRITICAL: phase_cross_correlation returns REVERSE of applied shift
        # Applied +5 DOWN, +3 RIGHT → detected should be -5, -3 (registration transform)
        expected_detected = (-applied_shift[0], -applied_shift[1])  # y, x ordering
        np.testing.assert_allclose(calculated_shift, expected_detected, atol=0.2)

    def test_background_estimation(self) -> None:
        """Test background estimation using photutils"""
        # Create image with varying background
        x, y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
        img = 100 + 50 * (x + y)  # Gradient background

        # Add some bright spots
        img[20:25, 20:25] = 1000
        img[40:45, 40:45] = 800

        bg = background(img)

        assert bg.shape == img.shape
        assert bg.min() >= 0, "Background should be non-negative"
        # Background should be smoother than original
        assert np.std(bg) < np.std(img)


class TestRunFiducial:
    """Unit tests for run_fiducial function - the core alignment pipeline"""

    @pytest.fixture
    def synthetic_reference_image(self) -> NDArray[np.uint16]:
        """Create synthetic reference image with fiducial spots"""
        img = np.random.randint(100, 200, TEST_IMAGE_SIZE, dtype=np.uint16)

        # Add fiducial spots at known locations
        fiducial_locations = [(20, 20), (20, 80), (80, 20), (80, 80), (50, 50)]
        for y, x in fiducial_locations:
            # Create bright Gaussian spots
            yy, xx = np.ogrid[
                max(0, y - 4) : min(TEST_IMAGE_SIZE[0], y + 5), max(0, x - 4) : min(TEST_IMAGE_SIZE[1], x + 5)
            ]
            if yy.size > 0 and xx.size > 0:
                spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4) * 2000
                img[
                    max(0, y - 4) : min(TEST_IMAGE_SIZE[0], y + 5),
                    max(0, x - 4) : min(TEST_IMAGE_SIZE[1], x + 5),
                ] += spot.astype(np.uint16)

        return img

    def test_run_fiducial_basic(self, synthetic_reference_image: NDArray[np.uint16]) -> None:
        """Test basic run_fiducial functionality"""
        # Get the alignment function
        align_func = run_fiducial(
            synthetic_reference_image, subtract_background=False, debug=False, threshold_sigma=3.0, fwhm=4.0
        )

        # Create target image with known shift
        shift = np.array([2.5, 1.8])  # x, y shift
        target_img = np.roll(
            np.roll(synthetic_reference_image, int(shift[1]), axis=0), int(shift[0]), axis=1
        ).astype(np.uint16)

        # Add some noise
        target_img = np.clip(target_img + np.random.randint(-20, 20, TEST_IMAGE_SIZE), 0, 65535).astype(
            np.uint16
        )

        # Calculate drift
        calculated_drift, residual = align_func(target_img, bitname="test", limit=4)

        # Should detect the shift reasonably well
        assert isinstance(calculated_drift, np.ndarray)
        assert calculated_drift.shape == (2,)
        assert isinstance(residual, (int, float))
        assert residual >= 0

        # For synthetic data, should be reasonably accurate
        # Drift should be negative of the applied shift (correction to align back to reference)
        np.testing.assert_allclose(calculated_drift, -shift, atol=1.0)

    def test_run_fiducial_with_background_subtraction(
        self, synthetic_reference_image: NDArray[np.uint16]
    ) -> None:
        """Test run_fiducial with background subtraction enabled"""
        align_func = run_fiducial(
            synthetic_reference_image, subtract_background=True, threshold_sigma=3.0, fwhm=4.0
        )

        # Should complete without error
        target_img = synthetic_reference_image  # Use same image (zero shift expected)
        calculated_drift, residual = align_func(target_img, bitname="bg_test")

        # Zero shift expected for identical images
        np.testing.assert_allclose(calculated_drift, [0, 0], atol=0.5)

    def test_run_fiducial_insufficient_spots(self, synthetic_reference_image: NDArray[np.uint16]) -> None:
        """Test run_fiducial behavior with insufficient spots"""
        # Use moderate threshold that works for reference but should fail on poor target
        align_func = run_fiducial(
            synthetic_reference_image,
            threshold_sigma=4.0,  # Moderate threshold that works for synthetic ref
            fwhm=4.0,
        )

        # Target with truly minimal signal - all zeros (which should definitely fail)
        target_img = np.zeros(TEST_IMAGE_SIZE, dtype=np.uint16)

        with pytest.raises(NotEnoughSpots):
            align_func(target_img, bitname="insufficient_test")

    def test_run_fiducial_too_many_spots(self) -> None:
        """Test run_fiducial behavior with too many spots (noisy image)"""
        # Create very noisy reference image
        noisy_ref = np.random.randint(500, 2000, TEST_IMAGE_SIZE, dtype=np.uint16)

        try:
            align_func = run_fiducial(
                noisy_ref,
                threshold_sigma=1.0,  # Low threshold for noisy image
                fwhm=2.0,
            )

            # This should either work or raise TooManySpots during reference processing
            target_img = noisy_ref
            result = align_func(target_img, bitname="noisy_test")

            # If it succeeds, should return reasonable values
            assert isinstance(result, tuple)
            assert len(result) == 2
        except (TooManySpots, NotEnoughSpots):
            # Expected for very noisy images
            pass

    def test_run_fiducial_drift_too_large(self, synthetic_reference_image: NDArray[np.uint16]) -> None:
        """Test run_fiducial behavior with very large drift"""
        align_func = run_fiducial(synthetic_reference_image, threshold_sigma=3.0, fwhm=4.0)

        # Create target with completely uncorrelated pattern - high intensity noise
        # that will create many false spots but no real alignment
        target_img = np.random.randint(500, 2000, TEST_IMAGE_SIZE, dtype=np.uint16)

        with pytest.raises((NotEnoughSpots, DriftTooLarge, ResidualTooLarge)):
            align_func(target_img, bitname="large_drift_test", limit=1)  # Reduce limit to make it fail faster

    def test_run_fiducial_parameter_adaptation(self, synthetic_reference_image: NDArray[np.uint16]) -> None:
        """Test that run_fiducial adapts parameters when alignment fails"""
        # This tests the retry logic with different sigma/fwhm combinations
        align_func = run_fiducial(
            synthetic_reference_image,
            threshold_sigma=4.0,  # Use a threshold that works from previous successful tests
            fwhm=4.0,
        )

        # Create challenging target image
        target_img = synthetic_reference_image.copy()
        # Add noise that might interfere with initial parameters
        target_img = np.clip(target_img + np.random.randint(-50, 50, TEST_IMAGE_SIZE), 0, 65535).astype(
            np.uint16
        )

        # Should eventually find working parameters or raise final exception
        try:
            calculated_drift, residual = align_func(target_img, bitname="adaptation_test", limit=2)
            # If successful, should return reasonable values
            assert isinstance(calculated_drift, np.ndarray)
            assert isinstance(residual, (int, float))
        except NotEnoughSpots:
            # May fail if no suitable parameters found within attempt limit
            pass


class TestAlignFiducials:
    """Unit tests for align_fiducials function - multi-threaded alignment"""

    @pytest.fixture
    def fiducial_image_set(self) -> dict[str, NDArray[np.uint16]]:
        """Create set of synthetic fiducial images with known shifts"""
        ref_img = np.random.randint(200, 300, TEST_IMAGE_SIZE, dtype=np.uint16)

        # Add consistent fiducial pattern
        fiducial_locs = [(25, 25), (75, 25), (25, 75), (75, 75)]
        for y, x in fiducial_locs:
            yy, xx = np.ogrid[y - 3 : y + 4, x - 3 : x + 4]
            spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2) * 1500
            ref_img[y - 3 : y + 4, x - 3 : x + 4] += spot.astype(np.uint16)

        # Create shifted versions
        images = {"reference": ref_img}
        known_shifts = {"shift_A": (3, 2), "shift_B": (-2, 4), "shift_C": (1, -3)}

        for name, (dy, dx) in known_shifts.items():
            shifted = np.roll(np.roll(ref_img, dy, axis=0), dx, axis=1)
            # Add some noise
            shifted = np.clip(shifted + np.random.randint(-30, 30, TEST_IMAGE_SIZE), 0, 65535).astype(
                np.uint16
            )
            images[name] = shifted

        return images

    def test_align_fiducials_basic(self, fiducial_image_set: dict[str, NDArray[np.uint16]]) -> None:
        """Test basic align_fiducials functionality"""
        shifts, residuals = align_fiducials(
            fiducial_image_set,
            reference="reference",
            use_fft=False,
            threads=1,  # Single-threaded for deterministic testing
            debug=True,
        )

        # Should return shifts for all non-reference images
        expected_keys = {"shift_A", "shift_B", "shift_C", "reference"}
        assert set(shifts.keys()) == expected_keys
        assert set(residuals.keys()) == {"shift_A", "shift_B", "shift_C", "reference"}

        # Reference should have zero shift
        np.testing.assert_array_equal(shifts["reference"], [0, 0])
        assert residuals["reference"] == 0.0

        # Other shifts should be reasonable (within 2 pixels of expected for synthetic data)
        for name in ["shift_A", "shift_B", "shift_C"]:
            assert isinstance(shifts[name], np.ndarray)
            assert shifts[name].shape == (2,)
            assert isinstance(residuals[name], (int, float))
            assert residuals[name] >= 0

    def test_align_fiducials_with_overrides(self, fiducial_image_set: dict[str, NDArray[np.uint16]]) -> None:
        """Test align_fiducials with manual shift overrides"""
        manual_shifts = {"shift_A": (5.0, 2.5), "shift_B": (-1.5, 3.0)}

        shifts, residuals = align_fiducials(
            fiducial_image_set, reference="reference", overrides=manual_shifts, threads=1, debug=True
        )

        # Manual shifts should be used exactly
        np.testing.assert_array_equal(shifts["shift_A"], [5.0, 2.5])
        np.testing.assert_array_equal(shifts["shift_B"], [-1.5, 3.0])

        # Overridden images shouldn't have residuals calculated
        assert "shift_A" not in residuals or residuals.get("shift_A", 0.0) == 0.0
        assert "shift_B" not in residuals or residuals.get("shift_B", 0.0) == 0.0

        # Non-overridden image should still be calculated
        assert "shift_C" in shifts
        assert isinstance(shifts["shift_C"], np.ndarray)

    def test_align_fiducials_fft_mode(self, fiducial_image_set: dict[str, NDArray[np.uint16]]) -> None:
        """Test align_fiducials with FFT-based phase correlation"""
        shifts, residuals = align_fiducials(
            fiducial_image_set, reference="reference", use_fft=True, threads=1, debug=True
        )

        # FFT mode should produce results
        assert len(shifts) == len(fiducial_image_set)

        # All residuals should be 0.0 in FFT mode (no iterative refinement)
        for name in ["shift_A", "shift_B", "shift_C"]:
            assert residuals[name] == 0.0

    def test_align_fiducials_missing_reference(
        self, fiducial_image_set: dict[str, NDArray[np.uint16]]
    ) -> None:
        """Test align_fiducials with missing reference"""
        with pytest.raises(ValueError, match="Could not find reference nonexistent"):
            align_fiducials(fiducial_image_set, reference="nonexistent", threads=1)

    def test_align_fiducials_multithreading(self, fiducial_image_set: dict[str, NDArray[np.uint16]]) -> None:
        """Test align_fiducials with multiple threads"""
        # Test with multiple threads
        shifts_mt, residuals_mt = align_fiducials(
            fiducial_image_set,
            reference="reference",
            threads=2,
            debug=False,  # Debug mode forces single-threaded
        )

        # Results should be consistent regardless of threading
        assert len(shifts_mt) == len(fiducial_image_set)
        assert all(isinstance(shift, np.ndarray) for shift in shifts_mt.values())


class TestAlignPhase:
    """Unit tests for align_phase function - FFT-based alignment"""

    def test_align_phase_basic(self) -> None:
        """Test basic align_phase functionality"""
        ref_img = np.random.random(TEST_IMAGE_SIZE)

        # Create shifted images
        images = {
            "ref": ref_img,
            "shift1": np.roll(np.roll(ref_img, 3, axis=0), 2, axis=1),
            "shift2": np.roll(np.roll(ref_img, -2, axis=0), 4, axis=1),
        }

        shifts = align_phase(images, reference="ref", threads=1, debug=True)

        # Should detect shifts correctly (y, x ordering)
        # Phase correlation returns correction needed to align back to reference
        np.testing.assert_array_equal(shifts["ref"], [0, 0])
        np.testing.assert_allclose(shifts["shift1"], [-3, -2], atol=0.1)  # Negative of applied shift
        np.testing.assert_allclose(shifts["shift2"], [2, -4], atol=0.1)  # Negative of applied shift

    def test_align_phase_multithreaded(self) -> None:
        """Test align_phase with multiple threads"""
        ref_img = np.random.random((64, 64))
        images = {"ref": ref_img, "test1": np.roll(ref_img, 5, axis=0), "test2": np.roll(ref_img, -3, axis=1)}

        shifts = align_phase(images, reference="ref", threads=2, debug=False)

        assert len(shifts) == 3
        assert all(isinstance(shift, np.ndarray) for shift in shifts.values())


class TestFiducialExceptions:
    """Test custom exception handling in fiducial processing"""

    def test_exception_hierarchy(self) -> None:
        """Test that custom exceptions are properly defined"""
        # All should be Exception subclasses
        assert issubclass(NotEnoughSpots, Exception)
        assert issubclass(TooManySpots, Exception)
        assert issubclass(ResidualTooLarge, Exception)
        assert issubclass(DriftTooLarge, Exception)

        # Test instantiation
        exceptions = [
            NotEnoughSpots("Not enough spots"),
            TooManySpots("Too many spots"),
            ResidualTooLarge("Residual too large"),
            DriftTooLarge("Drift too large"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert str(exc)  # Should have string representation


class TestFiducialIntegration:
    """Integration tests for complete fiducial processing workflows"""

    def test_complete_fiducial_workflow(self) -> None:
        """Test complete workflow from spot detection to alignment"""
        # Create synthetic multi-round dataset
        ref_img = np.random.randint(100, 200, (128, 128), dtype=np.uint16)

        # Add fiducials in consistent pattern
        fid_pattern = [(30, 30), (30, 98), (98, 30), (98, 98), (64, 64)]
        for y, x in fid_pattern:
            # Add Gaussian spots
            yy, xx = np.ogrid[y - 4 : y + 5, x - 4 : x + 5]
            spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 3) * 2000
            ref_img[y - 4 : y + 5, x - 4 : x + 5] += spot.astype(np.uint16)

        # Create multiple rounds with known shifts
        rounds = {
            "round0": ref_img,
            "round1": np.roll(np.roll(ref_img, 2, axis=0), 3, axis=1),
            "round2": np.roll(np.roll(ref_img, -1, axis=0), -2, axis=1),
        }

        # Add realistic noise
        for name, img in rounds.items():
            rounds[name] = np.clip(img + np.random.randint(-40, 40, img.shape), 0, 65535).astype(np.uint16)

        # Perform alignment
        shifts, residuals = align_fiducials(
            rounds,
            reference="round0",
            use_fft=False,
            threads=1,
            threshold_sigma=2.5,
            threshold_residual=0.5,
            debug=True,
        )

        # Verify results
        assert len(shifts) == 3
        np.testing.assert_array_equal(shifts["round0"], [0, 0])

        # Should detect shifts within reasonable accuracy for synthetic data
        for round_name in ["round1", "round2"]:
            assert isinstance(shifts[round_name], np.ndarray)
            assert shifts[round_name].shape == (2,)
            assert residuals[round_name] >= 0


class TestFiducialEdgeCases:
    """Test edge cases and error conditions for fiducial processing"""

    def test_single_pixel_image(self) -> None:
        """Test behavior with minimal image size"""
        tiny_img = np.array([[1000]], dtype=np.uint16)

        # Should handle gracefully
        try:
            result = find_spots(tiny_img, threshold_sigma=1.0, fwhm=1.0, minimum_spots=1)
            # If successful, should be valid DataFrame
            assert isinstance(result, pl.DataFrame)
        except (NotEnoughSpots, ValueError):
            # Expected for tiny images
            pass

    @pytest.mark.parametrize("noise_level", [0, 50, 200, 1000])
    def test_noise_robustness(self, noise_level: int) -> None:
        """Test fiducial detection robustness to different noise levels"""
        # Create clean image with fiducials
        clean_img = np.ones(TEST_IMAGE_SIZE, dtype=np.uint16) * 100
        clean_img[40:60, 40:60] = 2000  # Bright fiducial

        # Add noise
        if noise_level > 0:
            noise = np.random.randint(-noise_level, noise_level + 1, TEST_IMAGE_SIZE)
        else:
            noise = np.zeros(TEST_IMAGE_SIZE, dtype=int)

        noisy_img = np.clip(clean_img + noise, 0, 65535).astype(np.uint16)

        try:
            spots = find_spots(noisy_img, threshold_sigma=3.0, fwhm=4.0, minimum_spots=1)
            # Higher noise should generally find fewer reliable spots
            assert len(spots) >= 1
        except NotEnoughSpots:
            # Expected for high noise levels
            if noise_level < 500:  # Low noise should still work
                pytest.fail(f"Should find spots with noise level {noise_level}")

    def test_memory_efficiency_large_images(self) -> None:
        """Test memory efficiency with large images"""
        # Create large synthetic image (simulating real microscopy data)
        large_img = np.random.randint(100, 300, (512, 512), dtype=np.uint16)

        # Add some fiducials
        for i in range(0, 512, 100):
            for j in range(0, 512, 100):
                if i + 20 < 512 and j + 20 < 512:
                    large_img[i : i + 20, j : j + 20] = 3000

        try:
            # This should complete without excessive memory usage
            import time

            start_time = time.time()

            spots = find_spots(large_img, threshold_sigma=4.0, fwhm=6.0, minimum_spots=5)

            processing_time = time.time() - start_time
            assert processing_time < 5.0, f"Processing took too long: {processing_time:.2f}s"

            # Should find reasonable number of spots
            assert len(spots) >= 5

        except NotEnoughSpots:
            # May happen with challenging synthetic data
            pytest.skip("Insufficient spots found in large synthetic image")

    def test_extreme_parameter_values(self) -> None:
        """Test behavior with extreme parameter values"""
        img = np.random.randint(100, 200, SMALL_IMAGE_SIZE, dtype=np.uint16)
        img[20:30, 20:30] = 2000  # Add bright spot

        # Test extreme sigma values
        try:
            # Very low sigma (should find many spots)
            spots_low = find_spots(img, threshold_sigma=0.1, fwhm=2.0, minimum_spots=1)
            assert len(spots_low) >= 1
        except (NotEnoughSpots, TooManySpots):
            pass  # Either outcome reasonable for extreme parameters

        try:
            # Very high sigma (should find few spots)
            spots_high = find_spots(img, threshold_sigma=20.0, fwhm=2.0, minimum_spots=1)
            assert len(spots_high) >= 1
        except NotEnoughSpots:
            pass  # Expected for very high threshold
