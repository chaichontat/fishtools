import numpy as np

from fishtools.segment.normalize import sample_percentiles, calc_percentile


def test_sample_percentiles_shapes_and_order():
    # Synthetic (Z, Y, X, C)
    rng = np.random.default_rng(0)
    img = rng.integers(1, 2000, size=(3, 128, 128, 2), dtype=np.uint16)

    mean_perc, all_samples = sample_percentiles(img, channels=[1, 2], block=(32, 32), n=5, low=5, high=95, seed=1)

    assert mean_perc.shape == (2, 2)
    assert all_samples.ndim == 3 and all_samples.shape[1] == 2 and all_samples.shape[2] == 2
    # low <= high per channel
    assert np.all(mean_perc[:, 0] <= mean_perc[:, 1])


def test_calc_percentile_alias_matches():
    rng = np.random.default_rng(1)
    img = rng.integers(1, 65535, size=(2, 64, 64, 3), dtype=np.uint16)

    a, _ = sample_percentiles(img, channels=[1, 3], block=(32, 32), n=3, low=1, high=99, seed=0)
    b, _ = calc_percentile(img, channels=[1, 3], block=(32, 32), n=3, low=1, high=99, seed=0)

    np.testing.assert_allclose(a, b)
