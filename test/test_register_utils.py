"""
Comprehensive unit tests for utility functions in fishtools.preprocess.cli_register

This module tests the utility functions that support image registration:
- spillover_correction: Spectral bleed-through correction
- parse_nofids: Convert nofids into bits with shift correction
- sort_key: Sort dictionary items by numerical order
- get_rois: Extract ROI names from directory structure
"""

import tempfile
from pathlib import Path
from typing import Any

import fishtools.preprocess.cli_register as cli_register_module
from fishtools.preprocess import downsample as downsample_module
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage
from types import SimpleNamespace

from fishtools.preprocess.cli_register import (
    apply_deconv_scaling,
    get_rois,
    parse_nofids,
    sort_key,
    spillover_correction,
)
from fishtools.preprocess.downsample import gpu_downsample_xy

# Test constants
TEST_IMAGE_SIZE = (10, 100, 100)  # z, y, x
SMALL_IMAGE_SIZE = (5, 50, 50)


class TestSpilloverCorrection:
    """Unit tests for spillover_correction function"""

    def test_spillover_correction_basic(self) -> None:
        """Test basic spillover correction functionality"""
        # Create test data where spillover correction should occur
        spillee = np.array([100, 200, 300, 400], dtype=np.float32)
        spiller = np.array([50, 100, 150, 200], dtype=np.float32)
        corr = 0.5

        result = spillover_correction(spillee, spiller, corr)

        expected = np.array([75, 150, 225, 300], dtype=np.float32)  # spillee - (spiller * 0.5)
        np.testing.assert_array_equal(result, expected)

    def test_spillover_correction_negative_prevention(self) -> None:
        """Test that spillover correction prevents negative values"""
        spillee = np.array([10, 20, 30], dtype=np.float32)
        spiller = np.array([50, 100, 150], dtype=np.float32)  # Higher than spillee
        corr = 0.5  # spiller * corr = [25, 50, 75], higher than spillee

        result = spillover_correction(spillee, spiller, corr)

        # ACTUAL BEHAVIOR: Only corrects when spillee >= scaled, otherwise returns 0
        # spillee >= scaled: [10>=25, 20>=50, 30>=75] = [False, False, False]
        expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_spillover_correction_mixed_threshold(self) -> None:
        """Test spillover correction with mixed threshold behavior"""
        spillee = np.array([100, 50, 200], dtype=np.float32)
        spiller = np.array([120, 200, 100], dtype=np.float32)
        corr = 0.5  # scaled = [60, 100, 50]

        result = spillover_correction(spillee, spiller, corr)

        # spillee >= scaled: [100>=60, 50>=100, 200>=50] = [True, False, True]
        # Where True: spillee - scaled, Where False: 0
        expected = np.array([40, 0, 150], dtype=np.float32)  # [100-60, 0, 200-50]
        np.testing.assert_array_equal(result, expected)

    def test_spillover_correction_zero_correction(self) -> None:
        """Test spillover correction with zero correction factor"""
        spillee = np.array([100, 200, 300], dtype=np.float32)
        spiller = np.array([50, 100, 150], dtype=np.float32)
        corr = 0.0

        result = spillover_correction(spillee, spiller, corr)

        # No correction should be applied
        np.testing.assert_array_equal(result, spillee)

    def test_spillover_correction_full_correction(self) -> None:
        """Test spillover correction with full correction factor"""
        spillee = np.array([100, 200, 300], dtype=np.float32)
        spiller = np.array([50, 100, 150], dtype=np.float32)
        corr = 1.0

        result = spillover_correction(spillee, spiller, corr)

        expected = np.array([50, 100, 150], dtype=np.float32)  # spillee - spiller
        np.testing.assert_array_equal(result, expected)

    def test_spillover_correction_multidimensional(self) -> None:
        """Test spillover correction with multidimensional arrays"""
        spillee = np.ones((3, 4, 5)) * 100
        spiller = np.ones((3, 4, 5)) * 20
        corr = 0.25

        result = spillover_correction(spillee, spiller, corr)

        expected = np.ones((3, 4, 5)) * 95  # 100 - (20 * 0.25)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.int32])
    def test_spillover_correction_dtypes(self, dtype: np.dtype[Any]) -> None:
        """Test spillover correction with different data types"""
        spillee = np.array([100, 200, 300], dtype=dtype)
        spiller = np.array([20, 40, 60], dtype=dtype)
        corr = 0.5

        result = spillover_correction(spillee, spiller, corr)

        # Result should preserve appropriate numeric behavior
        assert result.shape == spillee.shape
        assert np.all(result >= 0)  # No negative values


class TestSortKey:
    """Unit tests for sort_key function"""

    def test_sort_key_numeric_strings(self) -> None:
        """Test sort_key with numeric strings"""
        test_cases = [
            (("1", np.array([1])), "01"),
            (("5", np.array([1])), "05"),
            (("12", np.array([1])), "12"),
            (("123", np.array([1])), "123"),
        ]

        for input_tuple, expected in test_cases:
            result = sort_key(input_tuple)
            assert result == expected

    def test_sort_key_non_numeric_strings(self) -> None:
        """Test sort_key with non-numeric strings"""
        test_cases = [
            (("A", np.array([1])), "A"),
            (("abc", np.array([1])), "abc"),
            (("1a", np.array([1])), "1a"),  # Mixed alphanumeric
            (("", np.array([1])), ""),  # Empty string
        ]

        for input_tuple, expected in test_cases:
            result = sort_key(input_tuple)
            assert result == expected

    def test_sort_key_sorting_behavior(self) -> None:
        """Test that sort_key produces correct sorting order"""
        # Create test data with mixed numeric and non-numeric keys
        test_data = [
            ("10", np.array([1])),
            ("2", np.array([1])),
            ("A", np.array([1])),
            ("1", np.array([1])),
            ("B", np.array([1])),
            ("20", np.array([1])),
        ]

        # Sort using our sort_key function
        sorted_data = sorted(test_data, key=sort_key)
        sorted_keys = [item[0] for item in sorted_data]

        # Numeric keys should be zero-padded and come first, then alphabetic
        expected_order = ["01", "02", "10", "20", "A", "B"]
        actual_order = [sort_key(item) for item in sorted_data]

        assert actual_order == expected_order

    def test_sort_key_edge_cases(self) -> None:
        """Test sort_key with edge cases"""
        edge_cases = [
            (("0", np.array([1])), "00"),  # Zero
            (("-1", np.array([1])), "-1"),  # Negative (should be non-numeric)
            (("1.5", np.array([1])), "1.5"),  # Decimal (should be non-numeric)
            ((" 1", np.array([1])), "01"),  # ACTUAL BEHAVIOR: int(" 1") works, strips whitespace
        ]

        for input_tuple, expected in edge_cases:
            result = sort_key(input_tuple)
            assert result == expected


class TestApplyDeconvScaling:
    """Unit tests for apply_deconv_scaling helper"""

    def test_apply_deconv_scaling_calls_scale_for_non_prenormalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Images without prenormalized metadata should use scale_deconv."""

        img = np.ones((2, 3, 4), dtype=np.float32)
        global_scaling = np.array([[0.2], [2.0]], dtype=np.float32)
        metadata = {"deconv_scale": [1.0], "deconv_min": [0.0]}

        def fake_scale(
            img_in: NDArray[np.float32],
            idx: int,
            *,
            name: str | None,
            global_deconv_scaling: NDArray[np.float32],
            metadata: dict[str, Any],
            debug: bool,
        ) -> NDArray[np.float32]:
            assert idx == 0
            assert name == "img-0"
            assert global_deconv_scaling is global_scaling
            assert metadata is metadata_dict
            assert not debug
            return img_in + 1.0

        metadata_dict = metadata
        monkeypatch.setattr(
            cli_register_module,
            "scale_deconv",
            fake_scale,
        )

        result = apply_deconv_scaling(
            img,
            idx=0,
            orig_name="img-0",
            global_deconv_scaling=global_scaling,
            metadata=metadata_dict,
            debug=False,
        )

        np.testing.assert_array_equal(result, img + 1.0)

    def test_apply_deconv_scaling_skips_when_prenormalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Images tagged prenormalized should bypass scale_deconv entirely."""

        img = np.full((2, 3, 4), 5.0, dtype=np.float32)
        global_scaling = np.array([[0.1], [1.5]], dtype=np.float32)
        metadata = {
            "deconv_scale": [1.0],
            "deconv_min": [0.0],
            "prenormalized": True,
        }

        def fail_scale(*_: Any, **__: Any) -> NDArray[np.float32]:
            raise AssertionError("scale_deconv must not be invoked for prenormalized images")

        monkeypatch.setattr(
            cli_register_module,
            "scale_deconv",
            fail_scale,
        )

        result = apply_deconv_scaling(
            img,
            idx=0,
            orig_name="img-0",
            global_deconv_scaling=global_scaling,
            metadata=metadata,
            debug=True,
        )

        assert result is img


class TestParseNofids:
    """Unit tests for parse_nofids function"""

    def test_parse_nofids_basic(self) -> None:
        """Test basic parse_nofids functionality"""
        # Create test data
        nofids = {
            "A_B-001": np.random.random((3, 2, 10, 10)).astype(np.float32),  # 2 bits: A, B
            "C_D-002": np.random.random((3, 2, 10, 10)).astype(np.float32),  # 2 bits: C, D
        }
        shifts = {
            "A_B-001": np.array([1.0, 2.0]),
            "C_D-002": np.array([3.0, 4.0]),
        }
        channels = {"A": "488", "B": "560", "C": "650", "D": "750"}

        out, out_shift, bit_name_mapping = parse_nofids(nofids, shifts, channels)

        # Verify output structure
        assert set(out.keys()) == {"A", "B", "C", "D"}
        assert set(out_shift.keys()) == {"A", "B", "C", "D"}
        assert set(bit_name_mapping.keys()) == {"A", "B", "C", "D"}

        # Verify shapes - should be (z, y, x) after slicing channel dimension
        for bit in ["A", "B", "C", "D"]:
            assert out[bit].shape == (3, 10, 10)

        # Verify bit name mapping
        assert bit_name_mapping["A"] == ("A_B-001", 0)
        assert bit_name_mapping["B"] == ("A_B-001", 1)
        assert bit_name_mapping["C"] == ("C_D-002", 0)
        assert bit_name_mapping["D"] == ("C_D-002", 1)

        # Verify shifts are propagated
        np.testing.assert_array_equal(out_shift["A"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(out_shift["B"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(out_shift["C"], np.array([3.0, 4.0]))
        np.testing.assert_array_equal(out_shift["D"], np.array([3.0, 4.0]))

    def test_parse_nofids_duplicate_bits_error(self) -> None:
        """Test parse_nofids raises error on duplicate bits"""
        # Create test data with duplicate bit names
        nofids = {
            "A_B-001": np.random.random((3, 2, 10, 10)).astype(np.float32),
            "A_C-002": np.random.random((3, 2, 10, 10)).astype(np.float32),  # Duplicate 'A'
        }
        shifts = {
            "A_B-001": np.array([1.0, 2.0]),
            "A_C-002": np.array([3.0, 4.0]),
        }
        channels = {"A": "488", "B": "560", "C": "650"}

        with pytest.raises(ValueError, match="Duplicated bit A in A_C-002"):
            parse_nofids(nofids, shifts, channels)

    def test_parse_nofids_shape_mismatch_error(self) -> None:
        """Test parse_nofids handles shape mismatches"""
        # Create test data with wrong number of channels
        nofids = {
            "A_B_C-001": np.random.random((3, 2, 10, 10)).astype(np.float32),  # 3 bits but 2 channels
        }
        shifts = {"A_B_C-001": np.array([1.0, 2.0])}
        channels = {"A": "488", "B": "560", "C": "650"}

        with pytest.raises(AssertionError):  # Should fail on shape assertion
            parse_nofids(nofids, shifts, channels)

    def test_parse_nofids_single_bit(self) -> None:
        """Test parse_nofids with single-bit images"""
        nofids = {"A-001": np.random.random((5, 1, 20, 20)).astype(np.float32)}
        shifts = {"A-001": np.array([0.5, 1.5])}
        channels = {"A": "488"}

        out, out_shift, bit_name_mapping = parse_nofids(nofids, shifts, channels)

        assert list(out.keys()) == ["A"]
        assert out["A"].shape == (5, 20, 20)
        assert bit_name_mapping["A"] == ("A-001", 0)
        np.testing.assert_array_equal(out_shift["A"], np.array([0.5, 1.5]))

    def test_parse_nofids_empty_input(self) -> None:
        """Test parse_nofids with empty input"""
        out, out_shift, bit_name_mapping = parse_nofids({}, {}, {})

        assert out == {}
        assert out_shift == {}
        assert bit_name_mapping == {}


class TestGetRois:
    """Unit tests for get_rois function"""

    def test_get_rois_wildcard(self) -> None:
        """Test get_rois with wildcard to extract all ROIs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test directories with ROI pattern
            (temp_path / "data--roi1").mkdir()
            (temp_path / "data--roi2").mkdir()
            (temp_path / "data--roi3+extra").mkdir()
            (temp_path / "not_roi_pattern").mkdir()
            (temp_path / "file.txt").touch()  # Not a directory

            result = get_rois(temp_path, "*")

            expected = {"roi1", "roi2", "roi3"}
            assert result == expected

    def test_get_rois_specific_roi(self) -> None:
        """Test get_rois with specific ROI name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = get_rois(temp_path, "specific_roi")

            assert result == {"specific_roi"}

    def test_get_rois_empty_directory(self) -> None:
        """Test get_rois with empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = get_rois(temp_path, "*")

            assert result == set()

    def test_get_rois_filters_empty_and_wildcard(self) -> None:
        """Test that get_rois filters out empty strings and wildcards"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directories with edge case names
            (temp_path / "data--").mkdir()  # Empty ROI name
            (temp_path / "data--*").mkdir()  # ROI name is wildcard
            (temp_path / "data--valid").mkdir()  # Valid ROI

            result = get_rois(temp_path, "*")

            assert result == {"valid"}  # Should filter out empty and '*'

    def test_get_rois_complex_patterns(self) -> None:
        """Test get_rois with complex directory patterns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create various directory patterns
            (temp_path / "prefix--roi1+suffix").mkdir()
            (temp_path / "other--roi2").mkdir()
            (temp_path / "prefix--roi3+suffix+more").mkdir()

            result = get_rois(temp_path, "*")

            expected = {"roi1", "roi2", "roi3"}
            assert result == expected


class TestParseNofidsIntegration:
    """Integration tests for parse_nofids with realistic data"""

    def test_parse_nofids_realistic_microscopy_data(self) -> None:
        """Test parse_nofids with realistic microscopy data structure"""
        # Simulate realistic FISH data structure
        nofids = {
            "1_5_9-0001": np.random.randint(100, 4000, (4, 3, 64, 64), dtype=np.uint16).astype(np.float32),
            "2_6_10-0001": np.random.randint(100, 4000, (4, 3, 64, 64), dtype=np.uint16).astype(np.float32),
            "3_7_11-0001": np.random.randint(100, 4000, (4, 3, 64, 64), dtype=np.uint16).astype(np.float32),
        }
        shifts = {
            "1_5_9-0001": np.array([0.1, 0.2]),
            "2_6_10-0001": np.array([0.3, 0.4]),
            "3_7_11-0001": np.array([0.5, 0.6]),
        }
        channels = {
            "1": "488",
            "2": "488",
            "3": "488",
            "5": "560",
            "6": "560",
            "7": "560",
            "9": "650",
            "10": "650",
            "11": "650",
        }

        out, out_shift, bit_name_mapping = parse_nofids(nofids, shifts, channels)

        # Should have all 9 bits
        expected_bits = {"1", "2", "3", "5", "6", "7", "9", "10", "11"}
        assert set(out.keys()) == expected_bits

        # All arrays should have correct shape (Z, Y, X)
        for bit in expected_bits:
            assert out[bit].shape == (4, 64, 64)

        # Verify bit name mapping preserves source information
        assert bit_name_mapping["1"] == ("1_5_9-0001", 0)
        assert bit_name_mapping["5"] == ("1_5_9-0001", 1)
        assert bit_name_mapping["9"] == ("1_5_9-0001", 2)

    def test_parse_nofids_data_integrity(self) -> None:
        """Test that parse_nofids preserves data integrity"""
        # Create test data with known values
        test_array_1 = np.ones((2, 2, 3, 3), dtype=np.float32)
        test_array_1[:, 0] = 100  # First channel
        test_array_1[:, 1] = 200  # Second channel

        nofids = {"A_B-001": test_array_1}
        shifts = {"A_B-001": np.array([1.0, 2.0])}
        channels = {"A": "488", "B": "560"}

        out, out_shift, bit_name_mapping = parse_nofids(nofids, shifts, channels)

        # Verify data integrity
        np.testing.assert_array_equal(out["A"], test_array_1[:, 0])  # Should extract first channel
        np.testing.assert_array_equal(out["B"], test_array_1[:, 1])  # Should extract second channel

        # All extracted arrays should be views/copies with correct values
        assert np.all(out["A"] == 100)
        assert np.all(out["B"] == 200)


# Test fixtures for realistic data
@pytest.fixture
def sample_nofids_data() -> dict[str, NDArray[np.float32]]:
    """Create sample nofids data for testing"""
    np.random.seed(42)  # For reproducible tests
    return {
        "1_2_3-0001": np.random.random((3, 3, 10, 10)).astype(np.float32),
        "4_5_6-0001": np.random.random((3, 3, 10, 10)).astype(np.float32),
    }


@pytest.fixture
def sample_shifts_data() -> dict[str, NDArray[np.float64]]:
    """Create sample shifts data for testing"""
    return {
        "1_2_3-0001": np.array([0.1, 0.2]),
        "4_5_6-0001": np.array([0.3, 0.4]),
    }


def test_sample_fixtures(
    sample_nofids_data: dict[str, NDArray[np.float32]], sample_shifts_data: dict[str, NDArray[np.float64]]
) -> None:
    """Test that fixtures work correctly"""
    assert len(sample_nofids_data) == 2
    assert len(sample_shifts_data) == 2

    for name, data in sample_nofids_data.items():
        assert data.shape == (3, 3, 10, 10)
        assert data.dtype == np.float32

    for name, shift in sample_shifts_data.items():
        assert shift.shape == (2,)
        assert shift.dtype == np.float64


# Edge cases and error handling
class TestRegisterUtilsEdgeCases:
    """Test edge cases and error conditions for register utilities"""

    def test_spillover_correction_edge_values(self) -> None:
        """Test spillover correction with edge values"""
        # Test with very small values
        spillee = np.array([1e-10, 1e-5, 1e-3])
        spiller = np.array([1e-11, 1e-6, 1e-4])
        corr = 0.5

        result = spillover_correction(spillee, spiller, corr)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    @pytest.mark.parametrize("special_value", [np.nan, np.inf, -np.inf])
    def test_spillover_correction_special_values(self, special_value: float) -> None:
        """Test spillover correction handles special floating point values"""
        spillee = np.array([100, special_value, 200], dtype=np.float32)
        spiller = np.array([50, 50, 50], dtype=np.float32)
        corr = 0.5

        result = spillover_correction(spillee, spiller, corr)

        # Test basic structure preservation
        assert result.shape == spillee.shape
        assert result.dtype == spillee.dtype

        # Test normal values are still processed correctly
        assert result[0] == 75.0  # 100 - (50 * 0.5)
        assert result[2] == 175.0  # 200 - (50 * 0.5)

        # Document actual behavior with special values
        # Implementation uses np.where, so special values propagate through comparison
        if np.isnan(special_value):
            # NaN comparison always False, so np.where returns 0
            assert result[1] == 0.0
        elif special_value == np.inf:
            # inf >= 25.0 is True, so returns inf - 25.0 = inf
            assert np.isinf(result[1]) and result[1] > 0
        elif special_value == -np.inf:
            # -inf >= 25.0 is False, so returns 0
            assert result[1] == 0.0

    def test_spillover_correction_numerical_precision(self) -> None:
        """Test spillover correction with floating point precision issues"""
        # Test values that are very close to threshold
        spillee = np.array([100.0000001, 100.0, 99.9999999], dtype=np.float32)
        spiller = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        corr = 0.5  # scaled = [100.0, 100.0, 100.0]

        result = spillover_correction(spillee, spiller, corr)

        # Test that floating point precision is handled appropriately
        assert np.all(np.isfinite(result))

        # Due to float32 precision, very small differences might be lost
        # Document the actual behavior rather than assuming
        expected_behavior = spillee >= 100.0  # threshold comparison
        for i in range(len(spillee)):
            if expected_behavior[i]:
                assert result[i] == spillee[i] - 100.0
            else:
                assert result[i] == 0.0

    def test_sort_key_with_large_numbers(self) -> None:
        """Test sort_key with large numbers"""
        large_num_tuple = ("999999", np.array([1]))
        result = sort_key(large_num_tuple)
        assert result == "999999"  # Should handle large numbers

    def test_parse_nofids_memory_efficiency(self) -> None:
        """Test that parse_nofids is memory efficient (doesn't copy unnecessarily)"""
        # Create a large array to test memory behavior
        large_array = np.ones((2, 1, 100, 100), dtype=np.float32)
        nofids = {"A-001": large_array}
        shifts = {"A-001": np.array([0.0, 0.0])}
        channels = {"A": "488"}

        out, out_shift, bit_name_mapping = parse_nofids(nofids, shifts, channels)

        # Verify output structure
        assert out["A"].shape == (2, 100, 100)
        assert out["A"].dtype == np.float32

        # CRITICAL: Verify memory efficiency - output should be a view, not a copy
        # This is essential for large microscopy datasets (gigabytes)
        assert np.shares_memory(out["A"], large_array), (
            "parse_nofids should return views, not copies for memory efficiency"
        )

        # Verify that the view extracts the correct data
        # large_array[:, 0] should be the extracted channel
        np.testing.assert_array_equal(out["A"], large_array[:, 0])

        # Test that modifying the original affects the view (confirms it's a view)
        original_value = large_array[0, 0, 50, 50]
        large_array[0, 0, 50, 50] = 999.0
        assert out["A"][0, 50, 50] == 999.0, "View should reflect changes to original array"

        # Clean up
        large_array[0, 0, 50, 50] = original_value


class TestGpuDownsample:
    """Unit tests for GPU-powered downsampling utilities."""

    def test_downsample_requires_gpu(self, monkeypatch: Any) -> None:
        """Ensure we fail fast when no CUDA device is available."""

        fake_cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        fake_cp = SimpleNamespace(
            float32=np.float32,
            uint16=np.uint16,
            cuda=fake_cuda,
            asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
            asnumpy=lambda arr: np.asarray(arr),
            clip=lambda arr, a_min, a_max: np.clip(arr, a_min, a_max),
            empty=lambda shape, dtype=np.float32: np.empty(shape, dtype=dtype),
        )

        monkeypatch.setattr(downsample_module, "cp", fake_cp)

        with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
            gpu_downsample_xy(np.zeros((1, 4, 4), dtype=np.float32), crop=0, factor=1)

    def test_downsample_xy_gpu_matches_numpy_reference(self, monkeypatch: Any) -> None:
        """Validate CuPy pipeline (patched) matches NumPy reference implementation."""

        fake_cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1))

        def fake_asarray(arr: NDArray[np.float32], dtype: Any | None = None) -> NDArray[np.float32]:
            return np.asarray(arr, dtype=dtype)

        fake_cp = SimpleNamespace(
            float32=np.float32,
            uint16=np.uint16,
            cuda=fake_cuda,
            asarray=fake_asarray,
            asnumpy=lambda arr: np.asarray(arr),
            clip=lambda arr, a_min, a_max: np.clip(arr, a_min, a_max),
            empty=lambda shape, dtype=np.float32: np.empty(shape, dtype=dtype),
        )

        monkeypatch.setattr(downsample_module, "cp", fake_cp)
        monkeypatch.setattr(
            downsample_module,
            "cp_zoom",
            lambda arr, zoom, output, order, mode, grid_mode: ndimage.zoom(
                arr,
                zoom,
                output=output,
                order=order,
                mode=mode,
                grid_mode=grid_mode,
            ),
        )

        data = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        result = gpu_downsample_xy(
            data,
            crop=2,
            factor=2,
            clip_range=(0, 65534),
            output_dtype=np.uint16,
        )

        cropped = data[:, 2:-2, 2:-2]
        target_shape = (cropped.shape[0], cropped.shape[1] // 2, cropped.shape[2] // 2)
        reference = ndimage.zoom(
            cropped,
            (1.0, 0.5, 0.5),
            output=np.empty(target_shape, dtype=np.float32),
            order=3,
            mode="reflect",
            grid_mode=True,
        )
        reference = np.clip(reference, 0, 65534).astype(np.uint16)

        np.testing.assert_array_equal(result, reference)
        assert result.dtype == np.uint16

    def test_downsample_xy_gpu_factor_one(self, monkeypatch: Any) -> None:
        """Factor of 1 should only crop while still using the CuPy pipeline."""

        fake_cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1))

        fake_cp = SimpleNamespace(
            float32=np.float32,
            uint16=np.uint16,
            cuda=fake_cuda,
            asarray=lambda arr, dtype=None: np.asarray(arr, dtype=dtype),
            asnumpy=lambda arr: np.asarray(arr),
            clip=lambda arr, a_min, a_max: np.clip(arr, a_min, a_max),
            empty=lambda shape, dtype=np.float32: np.empty(shape, dtype=dtype),
        )

        monkeypatch.setattr(downsample_module, "cp", fake_cp)

        data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        result = gpu_downsample_xy(
            data,
            crop=1,
            factor=1,
            clip_range=(0, 65534),
            output_dtype=np.uint16,
        )
        expected = np.clip(data[:, 1:-1, 1:-1], 0, 65534).astype(np.uint16)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint16
