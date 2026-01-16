import numpy as np
import pytest
from skimage.registration import phase_cross_correlation

from fishtools.preprocess.chromatic import FitAffine


def _gaussian2d(*, h: int, w: int, y0: float, x0: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    return np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))).astype(np.float32)


@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:builtin type swigvarlink has no __module__ attribute")
def test_fit_affine_reduces_residual_translation() -> None:
    h = w = 64
    fixed = _gaussian2d(h=h, w=w, y0=32.0, x0=32.0, sigma=5.0)
    moving = _gaussian2d(h=h, w=w, y0=32.0 - 2.0, x0=32.0 + 3.0, sigma=5.0)

    shift_before, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=50)
    resid_before = float(np.linalg.norm(np.asarray(shift_before, dtype=np.float64)))

    A, t, warped = FitAffine(
        optimizer="QuasiNewtonLBFGS",
        max_iterations=128,
        log_to_console=False,
    ).fit(fixed, moving)

    assert A.shape == (2, 2)
    assert t.shape == (2,)
    assert warped.shape == fixed.shape
    assert warped.dtype == np.float32
    assert np.isfinite(warped).all()

    shift_after, _, _ = phase_cross_correlation(fixed, warped, upsample_factor=50)
    resid_after = float(np.linalg.norm(np.asarray(shift_after, dtype=np.float64)))

    assert resid_after < 1.0
    assert resid_after < resid_before
