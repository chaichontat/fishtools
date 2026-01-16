"""Tests for histogram quantile estimation boundary behavior."""

from __future__ import annotations

import numpy as np
import pytest

from fishtools.preprocess.deconv.hist import _quantile_from_hist


class TestQuantileFromHistBoundaries:
    """Test boundary behavior of _quantile_from_hist."""

    def test_quantile_zero_returns_first_edge(self) -> None:
        """q=0 should return the leftmost edge, not the first midpoint."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([10, 20, 30], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 0.0)

        # q=0 means "minimum value" which is edges[0]
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_quantile_one_returns_last_edge(self) -> None:
        """q=1 should return the rightmost edge, not the last midpoint."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([10, 20, 30], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 1.0)

        # q=1 means "maximum value" which is edges[-1]
        assert result == pytest.approx(3.0, abs=1e-9)

    def test_quantile_very_high_returns_near_last_edge(self) -> None:
        """q=0.99999 should approach edges[-1], not be clamped to mids[-1]."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([10, 20, 30], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 0.99999)

        # Should be very close to 3.0, not clamped to 2.5 (the last midpoint)
        assert result > 2.5, f"Expected > 2.5 (last midpoint), got {result}"
        assert result <= 3.0

    def test_quantile_very_low_returns_near_first_edge(self) -> None:
        """q=0.00001 should approach edges[0], not be clamped to mids[0]."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([10, 20, 30], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 0.00001)

        # Should be very close to 0.0, not clamped to 0.5 (the first midpoint)
        assert result < 0.5, f"Expected < 0.5 (first midpoint), got {result}"
        assert result >= 0.0

    def test_quantile_median_interpolates_correctly(self) -> None:
        """q=0.5 should interpolate within the histogram."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        # Uniform distribution: total = 30, cumsum = [10, 20, 30]
        # CDF at midpoints: [10/30, 20/30, 30/30] = [0.333, 0.667, 1.0]
        counts = np.array([10, 10, 10], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 0.5)

        # With uniform counts, q=0.5 should be near the middle of the range
        assert 1.0 < result < 2.0

    def test_empty_histogram_returns_first_edge(self) -> None:
        """Empty histogram (all zeros) should return first edge."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([0, 0, 0], dtype=np.int64)

        result = _quantile_from_hist(counts, edges, 0.5)

        assert result == pytest.approx(0.0, abs=1e-9)

    def test_single_bin_with_all_mass(self) -> None:
        """All mass in last bin: high quantiles should approach edges[-1]."""
        edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        counts = np.array([0, 0, 100], dtype=np.int64)

        result_high = _quantile_from_hist(counts, edges, 0.99)

        # All mass in [2, 3], so q=0.99 should be near 3.0
        assert result_high > 2.5


class TestQuantileFromHistEdgeBasedInterpolation:
    """Test edge-based CDF interpolation behavior."""

    def test_interior_interpolation_uses_edges(self) -> None:
        """Interior quantiles use linear interpolation between bin edges."""
        edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        counts = np.array([100, 100, 100], dtype=np.int64)
        # Edge-based CDF: [0, 1/3, 2/3, 1.0] at edges [0, 10, 20, 30]

        result = _quantile_from_hist(counts, edges, 0.5)

        # q=0.5 falls between CDF[1]=1/3 and CDF[2]=2/3
        # Interpolate between edges[1]=10 and edges[2]=20
        # t = (0.5 - 1/3) / (2/3 - 1/3) = 0.5
        # result = 10 + 0.5 * (20 - 10) = 15
        assert result == pytest.approx(15.0, rel=0.01)
