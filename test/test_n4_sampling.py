from __future__ import annotations

import numpy as np

from fishtools.preprocess import n4


def test_sample_masked_values_respects_max_samples() -> None:
    h, w = 400, 400
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    mask = np.ones_like(arr, dtype=bool)

    samples, total = n4._sample_masked_values(arr, mask, max_samples=10_000, seed=123)

    assert total == h * w
    assert samples.size == 10_000
    # All samples should be drawn from the original array
    assert samples.min() >= arr.min()
    assert samples.max() <= arr.max()


def test_sample_masked_values_small_mask_returns_all() -> None:
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros_like(arr, dtype=bool)
    mask[2:4, 2:4] = True  # 4 pixels

    samples, total = n4._sample_masked_values(arr, mask, max_samples=50, seed=0)

    assert total == 4
    assert samples.size == 4
    expected = arr[mask]
    assert np.array_equal(np.sort(samples), np.sort(expected))


def test_stratified_sampling_covers_all_quadrants() -> None:
    """Stratified sampling should draw samples from all spatial regions."""
    h, w = 400, 400
    # Create array where each quadrant has distinct values
    arr = np.zeros((h, w), dtype=np.float32)
    arr[:200, :200] = 1.0  # top-left: 1
    arr[:200, 200:] = 2.0  # top-right: 2
    arr[200:, :200] = 3.0  # bottom-left: 3
    arr[200:, 200:] = 4.0  # bottom-right: 4
    mask = np.ones_like(arr, dtype=bool)

    samples, total = n4._sample_masked_values(
        arr, mask, max_samples=1000, seed=42, stratified=True, grid_size=4
    )

    assert total == h * w
    # Allow small variation due to proportional allocation rounding
    assert 900 <= samples.size <= 1000

    # All four quadrant values should be present in samples
    unique_vals = set(np.unique(samples))
    assert 1.0 in unique_vals, "Missing samples from top-left quadrant"
    assert 2.0 in unique_vals, "Missing samples from top-right quadrant"
    assert 3.0 in unique_vals, "Missing samples from bottom-left quadrant"
    assert 4.0 in unique_vals, "Missing samples from bottom-right quadrant"


def test_stratified_sampling_is_deterministic() -> None:
    """Stratified sampling should produce identical results with same seed."""
    h, w = 200, 200
    rng = np.random.default_rng(999)
    arr = rng.random((h, w)).astype(np.float32)
    mask = arr > 0.2

    samples1, total1 = n4._sample_masked_values(
        arr, mask, max_samples=500, seed=123, stratified=True
    )
    samples2, total2 = n4._sample_masked_values(
        arr, mask, max_samples=500, seed=123, stratified=True
    )

    assert total1 == total2
    assert np.array_equal(samples1, samples2)


def test_stratified_vs_random_on_gradient_image() -> None:
    """Stratified sampling should have lower variance on spatially heterogeneous images."""
    h, w = 400, 400
    # Create image with spatial gradient (0 to 1 from corner to corner)
    y, x = np.mgrid[0:h, 0:w]
    gradient = ((y + x) / (h + w - 2)).astype(np.float32)
    mask = np.ones((h, w), dtype=bool)

    true_p99 = float(np.percentile(gradient, 99))

    # Run multiple trials and compare variance
    n_trials = 20
    random_estimates = []
    stratified_estimates = []

    for seed in range(n_trials):
        # Random sampling
        samples_r, _ = n4._sample_masked_values(
            gradient, mask, max_samples=1000, seed=seed, stratified=False
        )
        random_estimates.append(float(np.percentile(samples_r, 99)))

        # Stratified sampling
        samples_s, _ = n4._sample_masked_values(
            gradient, mask, max_samples=1000, seed=seed, stratified=True
        )
        stratified_estimates.append(float(np.percentile(samples_s, 99)))

    random_std = np.std(random_estimates)
    stratified_std = np.std(stratified_estimates)

    # Stratified should have lower or similar variance
    # (may not always be strictly lower due to randomness, so use generous threshold)
    assert stratified_std <= random_std * 1.5, (
        f"Stratified variance ({stratified_std:.4f}) much higher than "
        f"random ({random_std:.4f})"
    )

    # Both should be reasonably close to true value
    random_mean_error = abs(np.mean(random_estimates) - true_p99)
    stratified_mean_error = abs(np.mean(stratified_estimates) - true_p99)
    assert random_mean_error < 0.05, f"Random mean error too high: {random_mean_error}"
    assert stratified_mean_error < 0.05, f"Stratified mean error too high: {stratified_mean_error}"


def test_stratified_sampling_sparse_mask() -> None:
    """Stratified sampling should handle sparse masks correctly."""
    h, w = 400, 400
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)

    # Sparse mask: only small regions in each quadrant
    mask = np.zeros((h, w), dtype=bool)
    mask[50:60, 50:60] = True  # 100 pixels in top-left
    mask[50:60, 250:260] = True  # 100 pixels in top-right
    mask[250:260, 50:60] = True  # 100 pixels in bottom-left
    mask[250:260, 250:260] = True  # 100 pixels in bottom-right

    samples, total = n4._sample_masked_values(
        arr, mask, max_samples=200, seed=0, stratified=True, grid_size=4
    )

    assert total == 400
    assert samples.size == 200

    # Check that samples come from all four regions
    # Values in each region are distinct due to sequential numbering
    samples_set = set(samples)
    has_top_left = any(50 * 400 + 50 <= s < 60 * 400 + 60 for s in samples_set)
    has_top_right = any(50 * 400 + 250 <= s < 60 * 400 + 260 for s in samples_set)
    has_bottom_left = any(250 * 400 + 50 <= s < 260 * 400 + 60 for s in samples_set)
    has_bottom_right = any(250 * 400 + 250 <= s < 260 * 400 + 260 for s in samples_set)

    assert has_top_left, "Missing samples from top-left region"
    assert has_top_right, "Missing samples from top-right region"
    assert has_bottom_left, "Missing samples from bottom-left region"
    assert has_bottom_right, "Missing samples from bottom-right region"

