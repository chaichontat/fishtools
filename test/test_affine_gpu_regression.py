import numpy as np
import pytest

from fishtools.preprocess.chromatic import Affine


def _make_affine(ref_img: np.ndarray, ref_channel: str = "560") -> Affine:
    # Identity 3x3 transforms for a non-reference channel; zeros for translations
    A = np.zeros((3, 3), dtype=np.float64)
    A[:2, :2] = np.eye(2, dtype=np.float64)
    A[2] = [0, 0, 1]
    A[:, 2] = [0, 0, 1]
    As = {"650": A}
    ats = {"650": np.zeros(3, dtype=np.float64)}
    # Provide a reference image so SimpleITK resampler has output geometry
    return Affine(ref_img=ref_img.astype(np.float32, copy=False), As=As, ats=ats, ref=ref_channel)


def test_affine_identity_preserves_array() -> None:
    rng = np.random.default_rng(0)
    z, y, x = 2, 64, 64
    vol = rng.standard_normal((z, y, x), dtype=np.float32)
    aff = _make_affine(ref_img=vol)

    out = aff(vol.astype(np.float32), channel="560", shiftpx=np.array([0.0, 0.0], dtype=np.float64))

    assert out.shape == vol.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Identity affine with zero shift should be exact for float32
    assert np.allclose(out, vol, rtol=1e-6, atol=1e-6)


def test_affine_translation_integer_shift_ref_channel_positions() -> None:
    z, y, x = 1, 128, 128
    vol = np.zeros((z, y, x), dtype=np.float32)
    y0, x0 = 30, 30
    vol[0, y0, x0] = 1000.0
    aff = _make_affine(ref_img=vol)

    # Reference channel uses pure translation path
    shift = np.array([5.0, -3.0], dtype=np.float64)  # (dy, dx)
    out = aff(vol, channel="560", shiftpx=shift)

    yy, xx = np.unravel_index(np.argmax(out[0]), out[0].shape)
    # SimpleITK TranslationTransform parameters are (tx, ty, tz) and map
    # output→input; positive tx,ty shift content by -tx,-ty in image index space.
    assert (yy, xx) == (y0 - int(shift[1]), x0 - int(shift[0]))
    assert np.isclose(out[0, yy, xx], 1000.0, atol=1e-3)


def test_affine_translation_integer_shift_nonref_channel_positions() -> None:
    z, y, x = 1, 128, 128
    vol = np.zeros((z, y, x), dtype=np.float32)
    y0, x0 = 40, 50
    vol[0, y0, x0] = 1000.0
    aff = _make_affine(ref_img=vol)

    # Non-reference channel takes composite(translate + affine)
    shift = np.array([-7.0, 4.0], dtype=np.float64)  # (dy, dx)
    out = aff(vol, channel="650", shiftpx=shift)

    yy, xx = np.unravel_index(np.argmax(out[0]), out[0].shape)
    assert (yy, xx) == (y0 - int(shift[1]), x0 - int(shift[0]))
    assert np.isclose(out[0, yy, xx], 1000.0, atol=1e-3)


def test_affine_translation_fractional_shift_com_matches_expected() -> None:
    z, y, x = 1, 128, 128
    vol = np.zeros((z, y, x), dtype=np.float32)
    y0, x0 = 60, 60
    vol[0, y0, x0] = 1000.0
    aff = _make_affine(ref_img=vol)

    shift = np.array([2.5, 3.5], dtype=np.float64)
    out = aff(vol, channel="560", shiftpx=shift)

    grid_y, grid_x = np.mgrid[0:y, 0:x]
    # Subtract cval background (100) to avoid COM bias from padding
    mass = np.clip(out[0] - 100.0, 0.0, None)
    cy = float((mass * grid_y).sum() / max(mass.sum(), 1e-6))
    cx = float((mass * grid_x).sum() / max(mass.sum(), 1e-6))
    # CoM should move by (-dx, -dy) due to output→input mapping
    assert np.isclose(cy - y0, -shift[1], atol=1e-3)
    assert np.isclose(cx - x0, -shift[0], atol=1e-3)
