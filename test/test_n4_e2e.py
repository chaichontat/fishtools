from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from fishtools.preprocess import n4

pytestmark = pytest.mark.timeout(30)


def _synthetic_bias_field(shape: tuple[int, int]) -> np.ndarray:
    """Create a smooth, strictly positive multiplicative field on [0.4, ~1.2].

    Field increases gently from center to edges to simulate vignetting/illumination bias.
    """
    h, w = shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)  # 0 at center, ~1.41 at corners
    rr = (rr / rr.max()).astype(np.float32)  # [0, 1]
    field = 0.4 + 0.8 * (rr**2)  # [0.4, 1.2]
    np.maximum(field, np.finfo(np.float32).tiny, out=field)
    return field.astype(np.float32)


def _ellipse_mask(shape: tuple[int, int], ry: float = 0.85, rx: float = 0.85) -> np.ndarray:
    h, w = shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    inside = (yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0
    return inside.astype(np.float32)


def _cv(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    med = float(np.nanmedian(a))
    if med == 0.0 or not np.isfinite(med):
        return float("inf")
    return float(np.nanstd(a) / abs(med))


def test_n4_field_estimation_no_nan_no_blowout_and_improves_uniformity() -> None:
    # Synthetic image: elliptical foreground with multiplicative bias
    shape = (64, 64)
    mask = _ellipse_mask(shape)
    base = 1000.0
    field_true = _synthetic_bias_field(shape)
    image = (base * field_true * mask).astype(np.float32)

    # Run SimpleITK N4 with fast settings suitable for CI
    field_est = n4.compute_correction_field(
        image,
        shrink=2,
        spline_lowres_px=12.0,
        iterations=(20, 12, 8),
    )

    # Basic properties
    assert field_est.shape == shape
    assert field_est.dtype == np.float32
    assert np.isfinite(field_est).all(), "Field must be finite (no NaN/inf)."
    assert float(field_est.min()) > 0.0, "Field must be strictly positive."

    # No blowouts: robust percentile bounds on foreground
    fg = field_est[mask > 0]
    p1, p50, p99 = np.percentile(fg, [1.0, 50.0, 99.0])
    assert 0.85 <= p50 <= 1.15, f"Median should be ~1.0, got {p50:.3f}"
    # Very loose but guards against pathological explosion/collapse
    assert p1 > 0.2, f"Lower percentile too small: {p1:.3f}"
    assert p99 < 5.0, f"Upper percentile too large: {p99:.3f}"

    # Validate that applying the estimated field flattens the foreground
    corrected = (image / field_est)[mask > 0]
    cv_before = _cv(image[mask > 0])
    cv_after = _cv(corrected)
    # Require a clear improvement and an absolute bound on residual variation
    assert cv_before / max(cv_after, 1e-12) >= 2.5, (
        f"Uniformity improvement too small: before={cv_before:.4f}, after={cv_after:.4f}"
    )
    assert cv_after <= 0.25, f"Residual variation too high: CV={cv_after:.4f}"


@pytest.mark.slow
@pytest.mark.timeout(120)  # override default 30s timeout
def test_n4_itk_128_boundary_bug() -> None:
    """Test that N4 works when image triggers ITK B-spline boundary condition.

    When shrink > 1, GetLogBiasFieldAsImage(img_full) can fail if the full image
    dimensions cause the B-spline parametric domain boundary to be hit exactly.
    This triggers "collapse point component N is outside parametric domain [0, N)".

    The bug occurs when full_size / 128 >= ~261 with shrink=4.
    """
    # 33408 = 128 * 261, triggers the boundary condition with shrink=4
    shape = (33408, 33408)
    y = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    mask = (rr < 0.9).astype(np.float32)
    image = (1000.0 * (0.5 + 0.5 * rr) * mask).astype(np.float32)

    # This would raise without the fix
    field_est = n4.compute_correction_field(
        image,
        shrink=4,
        spline_lowres_px=256.0,
        iterations=(3,),  # minimal iterations for speed
    )

    assert field_est.shape == shape
    assert np.isfinite(field_est).all()


def test_n4_resample_alignment() -> None:
    """Verify that resample-based reconstruction aligns with direct reconstruction.

    The fix for the ITK boundary bug reconstructs the bias field at small resolution
    then resamples to full resolution. This test verifies the resampled field matches
    the direct reconstruction (on sizes that work with both methods).
    """
    h, w = 1024, 1024
    shrink = 4

    # Create test image with radial bias
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    mask = (rr < 0.9).astype(np.float32)
    image = (1000.0 * (0.5 + 0.5 * rr) * mask).astype(np.float32)

    # Setup SimpleITK images
    img_full = sitk.GetImageFromArray(image)
    img_full.SetSpacing((1.0, 1.0))
    img_small = sitk.Shrink(img_full, [shrink, shrink])

    # Run N4
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.SetMaximumNumberOfIterations([10, 8])

    mask_arr = (sitk.GetArrayFromImage(img_small) > 0).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(img_small)

    _ = n4_filter.Execute(img_small, mask_img)

    # Method 1: Direct reconstruction to full (old method, works for this size)
    log_bias_direct = sitk.Cast(n4_filter.GetLogBiasFieldAsImage(img_full), sitk.sitkFloat32)
    field_direct = sitk.GetArrayFromImage(sitk.Exp(log_bias_direct))

    # Method 2: Reconstruct to small, then resample (new method)
    log_bias_small = sitk.Cast(n4_filter.GetLogBiasFieldAsImage(img_small), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_full)
    resampler.SetInterpolator(sitk.sitkBSpline)
    log_bias_resampled = resampler.Execute(log_bias_small)
    field_resampled = sitk.GetArrayFromImage(sitk.Exp(log_bias_resampled))

    # Verify alignment
    assert field_direct.shape == field_resampled.shape

    diff = np.abs(field_direct - field_resampled)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    relative_diff = max_diff / float(field_direct.mean())

    # Tolerance: resample interpolation introduces small differences
    assert max_diff < 0.01, f"Max difference too large: {max_diff:.6f}"
    assert mean_diff < 0.001, f"Mean difference too large: {mean_diff:.6f}"
    assert relative_diff < 0.01, f"Relative difference too large: {relative_diff:.4%}"
