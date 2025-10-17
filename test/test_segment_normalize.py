import numpy as np
from skimage.filters import unsharp_mask

from fishtools.segment.normalize import sample_percentile, sample_percentiles


def test_sample_percentiles_shapes_and_order():
    # Synthetic (Z, Y, X, C)
    rng = np.random.default_rng(0)
    img = rng.integers(1, 2000, size=(3, 128, 128, 2), dtype=np.uint16)

    mean_perc, all_samples = sample_percentiles(
        img, channels=[1, 2], block=(32, 32), n=5, low=5, high=95, seed=1
    )

    assert mean_perc.shape == (2, 2)
    assert all_samples.ndim == 3 and all_samples.shape[1] == 2 and all_samples.shape[2] == 2
    # low <= high per channel
    assert np.all(mean_perc[:, 0] <= mean_perc[:, 1])

    filtered = unsharp_mask(img, preserve_range=True, radius=3, channel_axis=3)
    for idx, channel in enumerate([0, 1]):
        np.testing.assert_allclose(mean_perc[idx, 0], all_samples[:, 0, idx].mean())
        np.testing.assert_allclose(mean_perc[idx, 1], all_samples[:, 1, idx].mean())

        global_low = np.percentile(filtered[..., channel], 5)
        global_high = np.percentile(filtered[..., channel], 95)
        assert np.isfinite(global_low) and np.isfinite(global_high)
        np.testing.assert_allclose(mean_perc[idx, 0], global_low, rtol=0.15, atol=50.0)
        np.testing.assert_allclose(mean_perc[idx, 1], global_high, rtol=0.15, atol=50.0)


def test_sample_percentile_alias_matches():
    rng = np.random.default_rng(1)
    img = rng.integers(1, 65535, size=(2, 64, 64, 3), dtype=np.uint16)

    a, _ = sample_percentiles(img, channels=[1, 3], block=(32, 32), n=3, low=1, high=99, seed=0)
    b, _ = sample_percentile(img, channels=[1, 3], block=(32, 32), n=3, low=1, high=99, seed=0)

    np.testing.assert_allclose(a, b)
