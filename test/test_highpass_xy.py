import numpy as np
import pytest

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as cgaussian_filter

from fishtools.preprocess.cli_deconv import high_pass_filter_xy


def _reference_highpass_xy(img: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """CPU reference: per-channel, per-z 2D Gaussian blur then subtract+clip.

    img shape: (Z, C, Y, X)
    """
    z, c, h, w = img.shape
    out = np.empty_like(img, dtype=np.float32)
    for ci in range(c):
        for zi in range(z):
            low = cgaussian_filter(img[zi, ci], sigma=sigma, mode="reflect", truncate=truncate)
            hp = img[zi, ci] - low
            out[zi, ci] = np.clip(hp, 0, None)
    return out


@pytest.mark.parametrize("shape,tile,sigma", [((3, 2, 127, 131), 32, 2.5), ((4, 1, 64, 64), 48, 3.0)])
def test_highpass_xy_tiled_matches_full(shape, tile, sigma):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 5000, size=shape).astype(np.float32)

    ref = _reference_highpass_xy(arr.copy(), sigma=sigma)

    gpu_in = cp.asarray(arr)
    got = high_pass_filter_xy(gpu_in, sigma_xy=sigma, tile=tile)
    got_np = cp.asnumpy(got)

    assert got_np.shape == ref.shape
    assert np.allclose(got_np, ref, rtol=1e-5, atol=5e-4)


def test_highpass_xy_full_vs_tiled_identical():
    rng = np.random.default_rng(1)
    arr = rng.uniform(0, 1_000, size=(2, 3, 200, 150)).astype(np.float32)

    full = high_pass_filter_xy(cp.asarray(arr), sigma_xy=3.3, tile=None)
    tiled = high_pass_filter_xy(cp.asarray(arr), sigma_xy=3.3, tile=64)

    np_full = cp.asnumpy(full)
    np_tiled = cp.asnumpy(tiled)

    assert np_full.shape == np_tiled.shape
    # Expect near-machine precision equality; tolerate tiny numeric drift
    assert np.allclose(np_full, np_tiled, rtol=1e-6, atol=1e-4)


def test_highpass_xy_non_negative():
    rng = np.random.default_rng(2)
    base = rng.uniform(100, 200, size=(1, 1, 50, 50)).astype(np.float32)
    trend = np.linspace(0, 50, 50, dtype=np.float32)[None, None, :, None]
    arr = base + trend

    out = high_pass_filter_xy(cp.asarray(arr), sigma_xy=2.0, tile=32)
    out_np = cp.asnumpy(out)
    assert out_np.min() >= -1e-6
