"""Tests for FFT-based fiducial alignment.

This module tests the use_fft=True path in align_fiducials to ensure
phase cross-correlation returns correct shift values.
"""

import numpy as np
import pytest
from synthetic_fiducials import FiducialParams, SyntheticFiducialImage


@pytest.mark.unit
class TestPhaseShiftCoordinates:
    """Test coordinate conventions for phase_shift function."""

    def test_phase_shift_coordinate_convention(self) -> None:
        """Verify phase_shift returns shifts in [dy, dx] format (row, col).

        phase_cross_correlation returns shifts in row, column order.
        This test documents the expected convention.
        """
        from fishtools.preprocess.fiducial import phase_shift

        # Create a simple test image
        params = FiducialParams(
            image_size=(256, 256),
            background_level=200.0,
            peak_intensity=3000.0,
            psf_sigma=1.5,
            noise_factor=0.1,  # Low noise for precise measurement
            seed=42,
        )
        ref_img = SyntheticFiducialImage(params)

        # Apply shift: +5 in x direction only (horizontal shift right)
        dx, dy = 5.0, 0.0
        shifted_img = ref_img.create_shifted_copy(dx, dy)

        # Get the detected shift
        detected = phase_shift(
            ref_img.image.astype(np.float64),
            shifted_img.image.astype(np.float64),
        )

        # phase_cross_correlation returns [row, col] = [dy, dx]
        # When we shift the image right (+dx), the detected shift should be negative
        # because phase_cross_correlation returns the shift to register moving to ref
        # So if moving was shifted +5 in x, we need -5 to bring it back
        detected_dy, detected_dx = detected[0], detected[1]

        # Check that the x-shift is detected (should be negative of applied shift)
        assert abs(detected_dx - (-dx)) < 1.0, (
            f"X-shift not detected correctly: expected ~{-dx}, got {detected_dx}"
        )
        # Check that y-shift is near zero
        assert abs(detected_dy) < 1.0, (
            f"Y-shift should be near zero: expected ~0, got {detected_dy}"
        )

    def test_phase_shift_normalizes_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """phase_shift should zero-center and scale inputs before FFT."""
        from fishtools.preprocess import fiducial as fiducial_module

        captured: dict[str, float] = {}

        def fake_phase_cross_correlation(ref: np.ndarray, img: np.ndarray, upsample_factor: int):
            captured["ref_mean"] = float(ref.mean())
            captured["ref_std"] = float(ref.std())
            captured["img_mean"] = float(img.mean())
            captured["img_std"] = float(img.std())
            return np.array([0.0, 0.0]), None, None

        monkeypatch.setattr(fiducial_module, "phase_cross_correlation", fake_phase_cross_correlation)

        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        moving = base * 4.0 + 500.0

        fiducial_module.phase_shift(base, moving)

        assert captured, "phase_cross_correlation should have been invoked"
        assert captured["ref_mean"] == pytest.approx(0.0, abs=1e-6)
        assert captured["img_mean"] == pytest.approx(0.0, abs=1e-6)
        assert captured["ref_std"] == pytest.approx(1.0, rel=1e-3)
        assert captured["img_std"] == pytest.approx(1.0, rel=1e-3)

    def test_align_fiducials_fft_uses_log_preprocessing(self) -> None:
        """align_fiducials(use_fft=True) should run phase correlation on LoG-filtered images."""
        from fishtools.preprocess.fiducial import align_fiducials

        from scipy import ndimage
        from skimage.registration import phase_cross_correlation

        def normalize(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img, dtype=np.float32)
            centered = arr - float(np.median(arr))
            scale = float(np.std(centered))
            if not np.isfinite(scale) or scale < 1e-6:
                max_abs = float(np.max(np.abs(centered)))
                scale = max_abs if max_abs > 0 else 1.0
            return centered / scale

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1.0, (128, 128)).astype(np.float32)
        y, x = np.mgrid[: ref.shape[0], : ref.shape[1]]
        ref = ref + (x / ref.shape[1]) * 10.0 + (y / ref.shape[0]) * 5.0

        # Use wrap so the Fourier-domain assumption matches the synthetic transform.
        moving = ndimage.shift(ref, shift=(4.0, -3.0), order=1, mode="wrap")

        fwhm = 4.0
        shifts, _ = align_fiducials(
            {"reference": ref, "moving": moving},
            reference="reference",
            use_fft=True,
            fwhm=fwhm,
            threads=1,
            debug=True,
        )

        detected = shifts["moving"]  # [dx, dy]

        sigma = max(0.5, fwhm / 2.355)

        def log_for_fft(img: np.ndarray) -> np.ndarray:
            filtered = -ndimage.gaussian_laplace(img.astype(np.float32), sigma=sigma)
            filtered -= float(filtered.min())
            p1, p99999 = np.percentile(filtered, [1, 99.99])
            if not np.isfinite(p1) or not np.isfinite(p99999) or p99999 - p1 <= 0:
                return np.zeros_like(filtered, dtype=np.float32)
            filtered = (filtered - float(p1)) / float(p99999 - p1)
            return np.clip(filtered, 0.0, 1.0).astype(np.float32)

        ref_log = log_for_fft(ref)
        moving_log = log_for_fft(moving)
        shift_dy_dx, _, _ = phase_cross_correlation(
            normalize(ref_log),
            normalize(moving_log),
            upsample_factor=100,
        )
        expected = shift_dy_dx[::-1]  # [dx, dy]

        np.testing.assert_allclose(detected, expected, atol=0.2)

    def test_phase_shift_vs_spot_based_coordinate_order(self) -> None:
        """Compare coordinate ordering between FFT and spot-based alignment.

        This test verifies that both methods return shifts in the same coordinate
        order, or documents any mismatch.
        """
        from fishtools.preprocess.fiducial import align_fiducials

        params = FiducialParams(
            image_size=(256, 256),
            background_level=200.0,
            peak_intensity=3000.0,
            psf_sigma=1.5,
            noise_factor=0.3,
            seed=42,
        )
        ref_img = SyntheticFiducialImage(params)

        # Apply known shift: +3 in x, +2 in y
        dx, dy = 3.0, 2.0
        shifted_img = ref_img.create_shifted_copy(dx, dy)

        fids = {
            "reference": ref_img.image.astype(np.float64),
            "shifted": shifted_img.image.astype(np.float64),
        }

        # Get FFT-based shifts
        fft_shifts, _ = align_fiducials(
            fids.copy(),
            reference="reference",
            use_fft=True,
            threads=1,
            debug=True,
        )

        # Get spot-based shifts
        spot_shifts, _ = align_fiducials(
            fids.copy(),
            reference="reference",
            use_fft=False,
            threads=1,
            debug=True,
            threshold_sigma=3.0,
            fwhm=4.0,
        )

        fft_shift = fft_shifts["shifted"]
        spot_shift = spot_shifts["shifted"]

        print(f"Applied shift: dx={dx}, dy={dy}")
        print(f"FFT shift: {fft_shift}")
        print(f"Spot shift: {spot_shift}")

        # Both should detect shifts of similar magnitude
        fft_mag = np.sqrt(np.sum(fft_shift**2))
        spot_mag = np.sqrt(np.sum(spot_shift**2))
        expected_mag = np.sqrt(dx**2 + dy**2)

        assert abs(fft_mag - expected_mag) < 2.0, (
            f"FFT magnitude {fft_mag} differs from expected {expected_mag}"
        )
        assert abs(spot_mag - expected_mag) < 2.0, (
            f"Spot magnitude {spot_mag} differs from expected {expected_mag}"
        )

        # Both methods should return shifts in the same [dx, dy] coordinate order
        # Allow for some tolerance due to different algorithms
        assert np.allclose(fft_shift, spot_shift, atol=1.5), (
            f"FFT and spot-based shifts differ too much: FFT={fft_shift}, Spot={spot_shift}. "
            f"Check coordinate ordering is consistent."
        )


@pytest.mark.unit
class TestFFTAlignment:
    """Test FFT-based alignment in align_fiducials."""

    def test_align_fiducials_fft_clamps_shift_to_50px(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FFT-based alignment should clamp extreme shifts to +/- 50 px per axis."""
        from loguru import logger

        from fishtools.preprocess import fiducial as fiducial_module
        from fishtools.preprocess.fiducial import align_fiducials

        monkeypatch.setattr(fiducial_module, "phase_shift", lambda *_args, **_kwargs: np.array([60.0, -100.0]))

        messages: list[str] = []

        def sink(message: str) -> None:
            messages.append(message)

        sink_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            fids = {
                "reference": np.zeros((64, 64), dtype=np.float32),
                "round1": np.ones((64, 64), dtype=np.float32),
            }

            shifts, _ = align_fiducials(
                fids,
                reference="reference",
                use_fft=True,
                threads=1,
                debug=True,
            )
        finally:
            logger.remove(sink_id)

        # phase_shift returns [dy, dx]; align_fiducials exposes [dx, dy]
        np.testing.assert_allclose(shifts["round1"], [-50.0, 50.0], atol=0.0)
        assert any("Clamped FFT shift" in msg and "round1" in msg for msg in messages), messages

    def test_align_fiducials_warns_on_exact_zero_shift_for_nonreference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-reference rounds with exactly [0,0] shift should raise a warning."""
        from loguru import logger

        from fishtools.preprocess import fiducial as fiducial_module
        from fishtools.preprocess.fiducial import align_fiducials

        messages: list[str] = []

        def sink(message: str) -> None:
            messages.append(message)

        sink_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            monkeypatch.setattr(
                fiducial_module,
                "phase_cross_correlation",
                lambda *_args, **_kwargs: (np.array([0.0, 0.0]), None, None),
            )

            fids = {
                "reference": np.zeros((64, 64), dtype=np.float32),
                "round1": np.ones((64, 64), dtype=np.float32),
            }

            shifts, _ = align_fiducials(
                fids,
                reference="reference",
                use_fft=True,
                threads=1,
                debug=True,
            )
            np.testing.assert_allclose(shifts["round1"], [0.0, 0.0], atol=0.0)

        finally:
            logger.remove(sink_id)

        assert any("exactly [0, 0]" in msg and "round1" in msg for msg in messages), messages

    def test_align_fiducials_fft_returns_nonzero_shifts(self) -> None:
        """Test that align_fiducials with use_fft=True returns non-zero shifts for shifted images.

        This is a regression test for the bug where FFT alignment returned all zeros
        due to a closure issue in the lambda function.
        """
        from fishtools.preprocess.fiducial import align_fiducials

        # Create reference image with clear fiducials
        params = FiducialParams(
            image_size=(256, 256),
            background_level=200.0,
            peak_intensity=3000.0,
            psf_sigma=1.5,
            noise_factor=0.3,
            seed=42,
        )
        ref_img = SyntheticFiducialImage(params)

        # Create shifted images with known shifts
        known_dx, known_dy = 5.0, 3.0
        shifted_img = ref_img.create_shifted_copy(known_dx, known_dy)

        # Create a second shifted image with different shift
        known_dx2, known_dy2 = -2.0, 4.0
        shifted_img2 = ref_img.create_shifted_copy(known_dx2, known_dy2)

        # Build fids dict - the reference must match the pattern
        fids = {
            "reference": ref_img.image.astype(np.float64),
            "round1": shifted_img.image.astype(np.float64),
            "round2": shifted_img2.image.astype(np.float64),
        }

        # Call align_fiducials with use_fft=True
        shifts, residuals = align_fiducials(
            fids,
            reference="reference",
            use_fft=True,
            threads=1,
            debug=True,
        )

        # The reference should have zero shift
        assert "reference" in shifts
        np.testing.assert_allclose(shifts["reference"], [0.0, 0.0], atol=1e-6)

        # Other rounds should have NON-ZERO shifts
        # This is the key assertion - if the bug exists, these will be zeros
        assert "round1" in shifts
        assert "round2" in shifts

        # phase_cross_correlation returns [dy, dx] format
        # The shift to align moving to reference is the negative of the applied shift
        round1_shift = shifts["round1"]
        round2_shift = shifts["round2"]

        # At minimum, the shifts should not all be zero (the bug symptom)
        assert not np.allclose(round1_shift, [0.0, 0.0], atol=0.1), (
            f"round1 shift is near-zero: {round1_shift}, expected non-zero for applied shift ({known_dx}, {known_dy})"
        )
        assert not np.allclose(round2_shift, [0.0, 0.0], atol=0.1), (
            f"round2 shift is near-zero: {round2_shift}, expected non-zero for applied shift ({known_dx2}, {known_dy2})"
        )

        # Verify the shifts have correct magnitude (within tolerance for phase correlation)
        expected_magnitude_1 = np.sqrt(known_dx**2 + known_dy**2)
        actual_magnitude_1 = np.sqrt(np.sum(round1_shift**2))
        assert abs(actual_magnitude_1 - expected_magnitude_1) < 2.0, (
            f"round1 shift magnitude {actual_magnitude_1} doesn't match expected {expected_magnitude_1}"
        )

        expected_magnitude_2 = np.sqrt(known_dx2**2 + known_dy2**2)
        actual_magnitude_2 = np.sqrt(np.sum(round2_shift**2))
        assert abs(actual_magnitude_2 - expected_magnitude_2) < 2.0, (
            f"round2 shift magnitude {actual_magnitude_2} doesn't match expected {expected_magnitude_2}"
        )

    def test_align_fiducials_fft_multiple_rounds_different_shifts(self) -> None:
        """Test that multiple rounds get distinct shift values, not the same value.

        A closure bug would cause all lambdas to capture the same final loop value,
        resulting in all rounds getting identical (incorrect) shifts.
        """
        from fishtools.preprocess.fiducial import align_fiducials

        params = FiducialParams(
            image_size=(256, 256),
            background_level=200.0,
            peak_intensity=3000.0,
            psf_sigma=1.5,
            noise_factor=0.3,
            seed=42,
        )
        ref_img = SyntheticFiducialImage(params)

        # Create multiple shifted images with distinctly different shifts
        shifts_to_apply = [
            ("round_a", 3.0, 1.0),
            ("round_b", -4.0, 2.0),
            ("round_c", 1.0, -5.0),
        ]

        fids = {"reference": ref_img.image.astype(np.float64)}
        for name, dx, dy in shifts_to_apply:
            shifted = ref_img.create_shifted_copy(dx, dy)
            fids[name] = shifted.image.astype(np.float64)

        shifts, residuals = align_fiducials(
            fids,
            reference="reference",
            use_fft=True,
            threads=1,
            debug=True,
        )

        # Get non-reference shifts
        non_ref_shifts = {k: v for k, v in shifts.items() if k != "reference"}

        # All shifts should be different from each other
        for i, (name_i, shift_i) in enumerate(non_ref_shifts.items()):
            for name_j, shift_j in list(non_ref_shifts.items())[i + 1 :]:
                assert not np.allclose(shift_i, shift_j, atol=0.5), (
                    f"Shifts for {name_i} and {name_j} are nearly identical: "
                    f"{shift_i} vs {shift_j}. This suggests a closure bug."
                )
