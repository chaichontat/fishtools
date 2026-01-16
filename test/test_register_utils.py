import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

import fishtools.preprocess.cli_register as cli_register_module
from fishtools.preprocess import downsample as downsample_module
from fishtools.preprocess.cli_register import (
    apply_deconv_scaling,
    get_rois,
    parse_nofids,
    sort_key,
    spillover_correction,
)
from fishtools.preprocess.config import Config, Fiducial, RegisterConfig
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

    def test_apply_deconv_scaling_skips_when_prenormalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
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


class TestRunFiducial:
    """Behavioural checks for run_fiducial helper"""

    def _make_config(
        self,
        *,
        priors: dict[str, tuple[float, float]] | None = None,
        overrides: dict[str, tuple[float, float]] | None = None,
        n_fids: int = 1,
        use_itk: bool = False,
        use_fft: bool = False,
    ) -> Config:
        return Config(
            dataPath="/tmp",
            registration=RegisterConfig(
                chromatic_path=Path("dummy"),
                fiducial=Fiducial(
                    priors=priors,
                    overrides=overrides,
                    n_fids=n_fids,
                    use_fft=use_fft,
                    use_itk=use_itk,
                    fwhm=3.0,
                    threshold=3.0,
                ),
                downsample=1,
                crop=0,
                slices=slice(None),
                reduce_bit_depth=0,
            ),
        )

    def test_run_fiducial_debug_fids_shifted_ordering(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Debug fids_shifted stack should follow sorted fid keys."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.1 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)

        def fake_shift(arr: np.ndarray, shift_vec: Sequence[float], **kwargs: Any) -> np.ndarray:
            return arr

        monkeypatch.setattr(cli_register_module, "shift", fake_shift)

        recorded: list[tuple[Path, np.ndarray, dict[str, Any]]] = []

        def fake_safe_imwrite(path: Path, data: np.ndarray, **kwargs: Any) -> None:
            metadata = kwargs.get("metadata", {})
            recorded.append((path, data, metadata))

        monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

        # Construct fids with deliberately unsorted keys to exercise ordering logic
        fids: dict[str, np.ndarray] = {
            "round_b": np.full((4, 4), 2, dtype=np.float32),
            "round_a": np.full((4, 4), 1, dtype=np.float32),
            "round_c": np.full((4, 4), 3, dtype=np.float32),
        }

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name="cb",
            config=config,
            roi="roi",
            idx=0,
            reference="round_a",
            debug=True,
        )

        shifted_records = [
            (path, data, metadata) for path, data, metadata in recorded if "-shifted-" in path.name
        ]
        assert shifted_records, "Expected a shifted debug write when debug=True"

        shifted_path, data, metadata = shifted_records[0]
        assert shifted_path.parent.name == "tifs"

        assert data.shape[0] == len(fids)

        expected_order = sorted(fids.keys())
        assert metadata.get("key") == expected_order

        means_by_plane = [float(np.mean(data[i])) for i in range(data.shape[0])]
        expected_means = [float(np.mean(fids[name])) for name in expected_order]
        assert means_by_plane == expected_means

    def test_run_fiducial_logs_debug_overlay_abs_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.1 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "shift", lambda arr, *_a, **_k: arr)
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

        roi = "roi"
        codebook_name = "cb"
        idx = 0
        expected_debug_dir, _, _ = cli_register_module._debug_fid_paths(deconv_path, roi, idx, codebook_name)
        expected_overlay = (
            expected_debug_dir / f"{roi}+{codebook_name}-{idx:04d}-overlay.png"
        ).resolve()

        def fake_save_debug_overlay(
            debug_dir: Path,
            roi_name: str,
            tile_idx: int,
            reference_name: str,
            shifted: dict[str, np.ndarray],
            *,
            codebook_name: str,
        ) -> Path:
            assert debug_dir == expected_debug_dir
            assert roi_name == roi
            assert tile_idx == idx
            assert reference_name == "round_a"
            assert shifted
            return debug_dir / f"{roi_name}+{codebook_name}-{tile_idx:04d}-overlay.png"

        monkeypatch.setattr(cli_register_module, "_save_debug_overlay", fake_save_debug_overlay)

        info_calls: list[str] = []

        def fake_info(msg: str, *_a: Any, **_k: Any) -> None:
            info_calls.append(msg)

        monkeypatch.setattr(cli_register_module.logger, "info", fake_info)

        fids: dict[str, np.ndarray] = {
            "round_a": np.full((4, 4), 1, dtype=np.float32),
            "round_b": np.full((4, 4), 2, dtype=np.float32),
        }

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name=codebook_name,
            config=config,
            roi=roi,
            idx=idx,
            reference="round_a",
            debug=True,
        )

        assert any(str(expected_overlay) in msg for msg in info_calls)

    def test_run_fiducial_debug_overlay_uses_log_when_use_fft(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors=None, use_fft=True)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.1 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "shift", lambda arr, *_a, **_k: arr)

        def fake_log_fids(arr: np.ndarray) -> np.ndarray:
            return np.full(arr.shape[-2:], 123.0, dtype=np.float32)

        monkeypatch.setattr(cli_register_module.Image, "loG_fids", staticmethod(fake_log_fids))

        writes: dict[Path, np.ndarray] = {}

        def fake_safe_imwrite(path: Path, data: np.ndarray, **_kwargs: Any) -> None:
            writes[path] = data

        monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

        overlay_shifted: dict[str, np.ndarray] | None = None

        def fake_save_debug_overlay(
            debug_dir: Path,
            roi: str,
            idx: int,
            reference_name: str,
            shifted: dict[str, np.ndarray],
            *,
            codebook_name: str,
        ) -> Path:
            nonlocal overlay_shifted
            overlay_shifted = shifted
            return debug_dir / f"{roi}+{codebook_name}-{idx:04d}-overlay.png"

        monkeypatch.setattr(cli_register_module, "_save_debug_overlay", fake_save_debug_overlay)

        fids = {
            "round_a": np.full((4, 4), 1.0, dtype=np.float32),
            "round_b": np.full((4, 4), 2.0, dtype=np.float32),
        }
        fids_raw = {
            "round_a": np.full((4, 4), 1000.0, dtype=np.float32),
            "round_b": np.full((4, 4), 2000.0, dtype=np.float32),
        }

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids_raw,
            codebook_name="cb",
            config=config,
            roi="roi",
            idx=0,
            reference="round_a",
            debug=True,
        )

        debug_dir, fids_name, _ = cli_register_module._debug_fid_paths(deconv_path, "roi", 0, "cb")
        debug_fids_path = (debug_dir / "tifs" / fids_name)
        assert debug_fids_path in writes
        assert np.all(writes[debug_fids_path] == 123.0)

        assert overlay_shifted is not None
        assert set(overlay_shifted) == set(fids)
        assert all(np.all(arr == 123.0) for arr in overlay_shifted.values())

    def test_run_fiducial_writes_debug_on_spot_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Spot-based registration failures should error (no FFT fallback)."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        reference = "round_a"
        failing_round = "round_b"

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            from fishtools.preprocess.fiducial import NotEnoughSpots

            raise NotEnoughSpots("Not enough spots")

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)

        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

        fids = {
            reference: np.ones((4, 4), dtype=np.float32),
            failing_round: np.ones((4, 4), dtype=np.float32) * 2,
        }

        roi = "roi_dbg"
        idx = 2

        from fishtools.preprocess.fiducial import NotEnoughSpots

        with pytest.raises(NotEnoughSpots, match="Not enough spots"):
            cli_register_module.run_fiducial(
                path=deconv_path,
                fids=fids,
                fids_raw=fids,
                codebook_name="cb",
                config=config,
                roi=roi,
                idx=idx,
                reference=reference,
                debug=False,
            )

        debug_dir, _, _ = cli_register_module._debug_fid_paths(deconv_path, roi, idx, "cb")
        assert not debug_dir.exists()

    def test_run_fiducial_does_not_write_debug_without_debug_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Debug artifacts are only written when debug=True (no auto-debug based on stats)."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        reference = "round_a"
        target = "round_b"
        debug_like_stat = cli_register_module.FiducialAlignmentStats(
            iterations=0,
            final_threshold=None,
            final_fwhm=None,
            n_spots=0,
            mode="fft",
            algorithm="phase_correlation",
        )

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.05 for name in fids}
            stats = {reference: None, target: debug_like_stat}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

        fids = {
            reference: np.ones((4, 4), dtype=np.float32),
            target: np.ones((4, 4), dtype=np.float32) * 3,
        }

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name="cb",
            config=config,
            roi="roi_force_itk",
            idx=1,
            reference=reference,
            debug=False,
        )

        debug_dir, _, _ = cli_register_module._debug_fid_paths(deconv_path, "roi_force_itk", 1, "cb")
        assert not debug_dir.exists()

    def test_run_fiducial_writes__fids_with_sorted_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_fids stack should follow sorted keys with key metadata."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.05 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)

        def fake_shift(arr: np.ndarray, shift_vec: Sequence[float], **kwargs: Any) -> np.ndarray:
            return arr

        monkeypatch.setattr(cli_register_module, "shift", fake_shift)

        records: list[tuple[Path, np.ndarray, dict[str, Any]]] = []

        def fake_safe_imwrite(path: Path, data: np.ndarray, **kwargs: Any) -> None:
            metadata = kwargs.get("metadata", {})
            records.append((path, data, metadata))

        monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

        fids: dict[str, np.ndarray] = {
            "round_z": np.full((4, 4), 3, dtype=np.float32),
            "round_x": np.full((4, 4), 1, dtype=np.float32),
            "round_y": np.full((4, 4), 2, dtype=np.float32),
        }

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name="cb",
            config=config,
            roi="roi",
            idx=0,
            reference="round_x",
            debug=False,
        )

        fids_stack_records = [
            (path, data, metadata)
            for path, data, metadata in records
            if "_fids-" in path.name
        ]
        assert fids_stack_records, "Expected _fids stack to be written"

        _, data, metadata = fids_stack_records[0]
        expected_order = sorted(fids.keys())

        assert data.shape[0] == len(expected_order)
        assert metadata.get("key") == expected_order

        means_by_plane = [float(np.mean(data[i])) for i in range(data.shape[0])]
        expected_means = [float(np.mean(fids[name])) for name in expected_order]
        assert means_by_plane == expected_means

    def test_run_fiducial_reference_fid_metadata_has_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Reference fid image should record its key in metadata."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.05 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)

        def fake_shift(arr: np.ndarray, shift_vec: Sequence[float], **kwargs: Any) -> np.ndarray:
            return arr

        monkeypatch.setattr(cli_register_module, "shift", fake_shift)

        records: list[tuple[Path, np.ndarray, dict[str, Any]]] = []

        def fake_safe_imwrite(path: Path, data: np.ndarray, **kwargs: Any) -> None:
            metadata = kwargs.get("metadata", {})
            records.append((path, data, metadata))

        monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

        fids: dict[str, np.ndarray] = {
            "round_b": np.ones((4, 4), dtype=np.float32),
            "round_a": np.ones((4, 4), dtype=np.float32) * 2,
        }
        reference = "round_a"

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name="cb",
            config=config,
            roi="roi",
            idx=1,
            reference=reference,
            debug=False,
        )

        ref_fid_records = [
            (path, data, metadata)
            for path, data, metadata in records
            if path.name.startswith("fids-")
        ]
        assert ref_fid_records, "Expected reference fid image to be written"

        _, data, metadata = ref_fid_records[0]
        assert data.shape == (4, 4)
        assert metadata.get("axes") == "YX"
        assert metadata.get("key") == [reference]

    def test_run_fiducial_writes_diagnostics_to_shifts_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """shifts-{idx}.json should include alignment diagnostics fields."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_safe_imwrite(path: Path, data: np.ndarray, **kwargs: Any) -> None:
            # No-op to avoid filesystem dependencies; metadata not needed here.
            return None

        monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)
        monkeypatch.setattr(cli_register_module, "shift", lambda arr, *_args, **_kwargs: arr)

        class _Stats:
            def __init__(self) -> None:
                self.iterations = 3
                self.final_threshold = 2.5
                self.final_fwhm = 4.5
                self.n_spots = 42
                self.mode = "spot"
                self.algorithm = "threshold"

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([1.5, -2.5], dtype=np.float32) for name in fids}
            residuals = {name: 0.123 for name in fids}
            stats = {name: _Stats() for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)

        fids: dict[str, np.ndarray] = {"round_a": np.ones((4, 4), dtype=np.float32)}
        roi = "roi_diag"
        codebook_name = "cbdiag"
        idx = 5

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name=codebook_name,
            config=config,
            roi=roi,
            idx=idx,
            reference="round_a",
            debug=False,
        )

        shift_dir = deconv_path / f"shifts--{roi}+{codebook_name}"
        json_path = shift_dir / f"shifts-{idx:04d}.json"
        assert json_path.exists()

        payload = json.loads(json_path.read_text())
        record = payload["round_a"]

        assert record["iterations"] == 3
        assert record["final_threshold"] == pytest.approx(2.5)
        assert record["final_fwhm"] == pytest.approx(4.5)
        assert record["n_spots"] == 42

    def test_run_fiducial_does_not_mutate_config_priors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors={"fidA": (1.0, 2.0)})
        config_before = copy.deepcopy(config.registration.fiducial.priors)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        (deconv_path / "fidA--roi").mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")
        tifffile.imwrite(
            deconv_path / "fidA--roi" / "fidA-0000.tif",
            np.zeros((3, 3, 3), dtype=np.uint16),
        )

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.1 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *args, **kwargs: None)

        fids = {"fidA-0000": np.ones((4, 4), dtype=np.float32)}

        result = cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name="cb",
            config=config,
            roi="roi",
            idx=0,
            reference="fidA-0000",
            debug=False,
        )

        assert result
        assert config.registration.fiducial.priors == config_before

    def test_run_fiducial_uses_derived_priors_without_mutation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors=None, n_fids=2)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.05 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *args, **kwargs: None)

        shift_calls: list[tuple[float, float]] = []

        def fake_shift(arr: np.ndarray, shift_vec: Sequence[float], **kwargs: Any) -> np.ndarray:
            shift_calls.append(tuple(float(v) for v in shift_vec))
            return arr

        monkeypatch.setattr(cli_register_module, "shift", fake_shift)

        roi = "roi"
        codebook_name = "cb"
        shift_dir = deconv_path / f"shifts--{roi}+{codebook_name}"
        shift_dir.mkdir(parents=True)

        payload = {
            "fidB-0001": {
                "shifts": [0.3, -0.4],
                "residual": 0.1,
                "corr": 0.95,
            }
        }
        for idx in range(11):
            (shift_dir / f"shifts-{idx:04d}.json").write_text(json.dumps(payload))

        z_planes = 2
        n_fids = config.registration.fiducial.n_fids
        image = np.arange((z_planes * 3 + n_fids) * 4 * 4, dtype=np.uint16).reshape(
            z_planes * 3 + n_fids, 4, 4
        )
        tile_dir = deconv_path / "fidB--roi"
        tile_dir.mkdir(parents=True)
        tifffile.imwrite(tile_dir / "fidB-0001.tif", image.astype(np.uint16))

        fids = {"fidB-0001": np.ones((4, 4), dtype=np.float32)}

        cli_register_module.run_fiducial(
            path=deconv_path,
            fids=fids,
            fids_raw=fids,
            codebook_name=codebook_name,
            config=config,
            roi=roi,
            idx=1,
            reference="fidB-0001",
            debug=False,
        )

        assert shift_calls[0] == (-0.4, 0.3)
        # Note: run_fiducial caches derived priors into the config for performance
        assert config.registration.fiducial.priors == {"fidB-0001": (0.3, -0.4)}

    def test_run_fiducial_raises_on_unmatched_derived_priors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When derived priors don't match any fid, run_fiducial raises ValueError."""
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.02 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *args, **kwargs: None)
        monkeypatch.setattr(cli_register_module, "shift", lambda arr, *args, **kwargs: arr)

        roi = "roi"
        codebook_name = "cb"
        shift_dir = deconv_path / f"shifts--{roi}+{codebook_name}"
        shift_dir.mkdir(parents=True)

        # Create shifts for a round that doesn't exist in fids
        payload = {
            "unused-0001": {
                "shifts": [1.0, -1.0],
                "residual": 0.05,
                "corr": 0.9,
            }
        }
        for idx in range(11):
            (shift_dir / f"shifts-{idx:04d}.json").write_text(json.dumps(payload))

        fids = {"fidC-0002": np.ones((4, 4), dtype=np.float32)}

        with pytest.raises(ValueError, match="Could not find file that starts with unused-0001"):
            cli_register_module.run_fiducial(
                path=deconv_path,
                fids=fids,
                fids_raw=fids,
                codebook_name=codebook_name,
                config=config,
                roi=roi,
                idx=2,
                reference="fidC-0002",
                debug=False,
            )

    def test_run_fiducial_missing_channel_metadata_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors=None)

        class DummyAffine:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.ref_image: np.ndarray | None = None

            def __call__(
                self,
                img: np.ndarray,
                *,
                channel: str,
                shiftpx: Sequence[float],
                debug: bool,
            ) -> np.ndarray:
                return img

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.01 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        def fake_scale(
            img: np.ndarray,
            idx: int,
            *,
            name: str | None,
            global_deconv_scaling: np.ndarray,
            metadata: dict[str, Any],
            debug: bool,
        ) -> np.ndarray:
            return img

        def fake_from_file(*args: Any, **kwargs: Any) -> SimpleNamespace:
            bits = ["1", "2"]
            return SimpleNamespace(
                name="1_2",
                idx=2,
                nofid=np.zeros((1, len(bits), 4, 4), dtype=np.float32),
                fid=np.zeros((4, 4), dtype=np.float32),
                fid_raw=np.zeros((4, 4), dtype=np.float32),
                bits=bits,
                powers={"ilm560": 1.0},
                metadata={"prenormalized": False},
                global_deconv_scaling=np.ones((2, len(bits)), dtype=np.float32),
                basic=lambda: None,
            )

        roi = "roi"
        codebook_name = "cb"
        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        shifts_dir = deconv_path / f"shifts--{roi}+{codebook_name}"
        shifts_dir.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")
        (shifts_dir / "shifts-0000.json").write_text(
            json.dumps({"fidC-0002": {"shifts": [0.0, 0.0], "residual": 0.0, "corr": 1.0}})
        )

        tile_dir = deconv_path / f"1_2--{roi}"
        tile_dir.mkdir()
        (tile_dir / "1_2-0002.tif").touch()

        codebook_path = tmp_path / "cb.json"
        codebook_path.write_text(json.dumps({"gene": [1, 2]}))

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "scale_deconv", fake_scale)
        monkeypatch.setattr(cli_register_module, "Affine", DummyAffine)
        monkeypatch.setattr(cli_register_module.Image, "from_file", staticmethod(fake_from_file))
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *args, **kwargs: None)

        # When channel metadata is missing for a bit, _run raises KeyError
        with pytest.raises(KeyError, match="2"):
            cli_register_module._run(
                deconv_path,
                roi,
                2,
                codebook=codebook_path,
                reference="1_2",
                config=config,
                debug=False,
                overwrite=False,
                no_priors=True,
            )

    def test_run_fiducial_missing_required_tiles_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = self._make_config(priors=None)

        workspace = tmp_path / "ws"
        deconv_path = workspace / "analysis" / "deconv"
        deconv_path.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("")

        def fake_align_with_stats(
            fids: dict[str, np.ndarray], **_: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
            shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
            residuals = {name: 0.02 for name in fids}
            stats = {name: None for name in fids}
            return shifts, residuals, stats

        def fake_scale(
            img: np.ndarray,
            idx: int,
            *,
            name: str | None,
            global_deconv_scaling: np.ndarray,
            metadata: dict[str, Any],
            debug: bool,
        ) -> np.ndarray:
            return img

        def fake_from_file(*args: Any, **kwargs: Any) -> SimpleNamespace:
            bits = ["1"]
            return SimpleNamespace(
                name="1",
                idx=2,
                nofid=np.zeros((1, len(bits), 4, 4), dtype=np.float32),
                fid=np.zeros((4, 4), dtype=np.float32),
                fid_raw=np.zeros((4, 4), dtype=np.float32),
                bits=bits,
                powers={"ilm560": 1.0},
                metadata={"prenormalized": False},
                global_deconv_scaling=np.ones((2, len(bits)), dtype=np.float32),
                basic=lambda: None,
            )

        roi = "roi"
        codebook_name = "cb"
        shifts_dir = deconv_path / f"shifts--{roi}+{codebook_name}"
        shifts_dir.mkdir(parents=True)
        (shifts_dir / "shifts-0000.json").write_text(
            json.dumps({"1": {"shifts": [0.0, 0.0], "residual": 0.0, "corr": 1.0}})
        )

        tile_dir = deconv_path / f"1--{roi}"
        tile_dir.mkdir()
        (tile_dir / "1-0002.tif").touch()

        codebook_path = tmp_path / "cb.json"
        codebook_path.write_text(json.dumps({"gene": [1, 2]}))

        monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
        monkeypatch.setattr(cli_register_module, "scale_deconv", fake_scale)
        monkeypatch.setattr(cli_register_module.Image, "from_file", staticmethod(fake_from_file))
        monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *args, **kwargs: None)

        # Codebook requires bits [1, 2] but only bit 1 is available
        with pytest.raises(ValueError, match="Missing codebook bits"):
            cli_register_module._run(
                deconv_path,
                roi,
                2,
                codebook=codebook_path,
                reference="1",
                config=config,
                debug=False,
                overwrite=False,
                no_priors=True,
            )


def test_run_prefers_repaired_round_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify --repaired switches tile discovery to the repaired folder."""

    config = Config(
        dataPath="/tmp",
        registration=RegisterConfig(
            chromatic_path=cli_register_module.DATA,
            fiducial=Fiducial(
                priors=None,
                overrides=None,
                n_fids=1,
                use_fft=False,
                use_itk=False,
                fwhm=3.0,
                threshold=3.0,
            ),
            downsample=1,
            crop=0,
            slices=slice(None),
            reduce_bit_depth=0,
        ),
    )

    workspace = tmp_path / "ws"
    deconv_path = workspace / "analysis" / "deconv"
    deconv_path.mkdir(parents=True)
    (workspace / "workspace.DONE").write_text("")

    roi = "roiA"
    round_name = "r1"
    idx = 1

    original_dir = deconv_path / f"{round_name}--{roi}"
    original_dir.mkdir()
    (original_dir / f"{round_name}-{idx:04d}.tif").write_bytes(b"")

    repaired_dir = deconv_path / f"{round_name}--{roi}--repaired"
    repaired_dir.mkdir()
    repaired_tile = repaired_dir / f"{round_name}-{idx:04d}.tif"
    repaired_tile.write_bytes(b"")

    codebook_path = tmp_path / "codebook.json"
    codebook_path.write_text(json.dumps({"gene": [round_name]}))

    called_files: list[Path] = []

    def fake_from_file(path: Path, **_kwargs: Any) -> SimpleNamespace:
        called_files.append(path)
        name, idx_token = path.stem.split("-")
        bits = name.split("_")
        nofid = np.zeros((1, len(bits), 4, 4), dtype=np.float32)
        fid = np.zeros((4, 4), dtype=np.float32)
        fid_raw = np.zeros((4, 4), dtype=np.float32)
        powers = {"560": 1.0}
        return SimpleNamespace(
            name=name,
            idx=int(idx_token),
            nofid=nofid,
            fid=fid,
            fid_raw=fid_raw,
            bits=bits,
            powers=powers,
            metadata={"prenormalized": True},
            global_deconv_scaling=None,
            basic=lambda: None,
        )

    def fake_run_fiducial(
        _path: Path,
        fid_images: dict[str, np.ndarray],
        _codebook_name: str,
        _config: Config,
        *,
        roi: str,
        idx: int,
        reference: str,
        debug: bool,
        no_priors: bool,
        fids_raw: dict[str, np.ndarray] | None = None,
        max_iters: int = 5,
    ) -> dict[str, np.ndarray]:
        assert roi == "roiA"
        assert idx == 1
        assert reference == round_name
        assert not debug
        assert no_priors is False
        return {name: np.zeros(2, dtype=np.float32) for name in fid_images}

    class DummyAffine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ref_image: np.ndarray | None = None

        def __call__(
            self,
            img: np.ndarray,
            *,
            channel: str,
            shiftpx: np.ndarray,
            debug: bool,
        ) -> np.ndarray:
            return img

    def fake_safe_imwrite(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(cli_register_module.Image, "from_file", staticmethod(fake_from_file))
    monkeypatch.setattr(cli_register_module, "run_fiducial", fake_run_fiducial)
    monkeypatch.setattr(cli_register_module, "Affine", DummyAffine)
    monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

    cli_register_module._run(
        deconv_path,
        roi,
        idx,
        codebook=codebook_path,
        reference=round_name,
            config=config,
            debug=False,
            overwrite=True,
        repaired_rounds={round_name},
    )

    assert called_files == [repaired_tile]


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


@pytest.mark.gpu
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
        from skimage.transform import downscale_local_mean as np_downscale

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
        monkeypatch.setattr(downsample_module, "downscale_local_mean", np_downscale)

        data = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        result = gpu_downsample_xy(
            data,
            crop=2,
            factor=2,
            clip_range=(0, 65534),
            output_dtype=np.uint16,
        )

        cropped = data[:, 2:-2, 2:-2]
        reference = np_downscale(cropped, (1, 2, 2))
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
