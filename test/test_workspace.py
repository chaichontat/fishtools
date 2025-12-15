"""Comprehensive test suite for Workspace class.

Tests directory structure parsing, path resolution, regex pattern matching,
and all public methods of the Workspace class. Includes scientific computing
robustness, memory efficiency, and performance benchmarks.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from fishtools.utils.io import CorruptedTiffError, OptimizePath, Workspace


def _write_done_sentinel(workspace_root: Path) -> None:
    (workspace_root / "workspace.DONE").write_text("ok\n", encoding="utf-8")


class TestWorkspaceInitialization:
    """Test Workspace initialization and path resolution."""

    def test_init_with_workspace_root(self):
        """Test initialization with workspace root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            _write_done_sentinel(workspace_path)
            ws = Workspace(workspace_path)
            assert ws.path == workspace_path.resolve()

    def test_init_with_deconv_subdirectory(self):
        """Test initialization with analysis/deconv subdirectory auto-resolves to root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            _write_done_sentinel(workspace_path)
            deconv_path = workspace_path / "analysis" / "deconv"
            deconv_path.mkdir(parents=True)

            ws = Workspace(deconv_path)
            assert ws.path == workspace_path.resolve()

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_done_sentinel(Path(tmpdir))
            ws = Workspace(tmpdir)
            assert ws.path == Path(tmpdir).resolve()

    def test_init_expands_user_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that ~ is expanded in paths."""
        monkeypatch.setenv("HOME", str(tmp_path))
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace("~/workspace")
        assert ws.path == workspace_path.resolve()


class TestWorkspaceRegexPatterns:
    """Test the regex patterns used for directory name parsing."""

    def test_round_roi_pattern_basic(self):
        """Test ROUND_ROI_PATTERN with basic round--roi format."""
        pattern = Workspace.ROUND_ROI_PATTERN

        # Valid patterns
        assert pattern.match("1_9_17--cortex")
        assert pattern.match("2_10_18--hippocampus")
        assert pattern.match("round1--roi_a")

        # Extract groups
        match = pattern.match("1_9_17--cortex")
        assert match.group(1) == "1_9_17"
        assert match.group(2) == "cortex"

    def test_round_roi_pattern_with_codebook(self):
        """Test ROUND_ROI_PATTERN with codebook suffix."""
        pattern = Workspace.ROUND_ROI_PATTERN

        # Valid patterns with suffix
        assert pattern.match("1_9_17--cortex+codebook_v1")
        assert pattern.match("registered--roi1+cb")

        # Extract groups (suffix ignored)
        match = pattern.match("1_9_17--cortex+codebook_v1")
        assert match.group(1) == "1_9_17"
        assert match.group(2) == "cortex"

    def test_round_roi_pattern_invalid(self):
        """Test ROUND_ROI_PATTERN rejects invalid formats."""
        pattern = Workspace.ROUND_ROI_PATTERN

        # Invalid patterns
        assert not pattern.match("no_double_dash")
        assert not pattern.match("--starts_with_dash")
        assert not pattern.match("ends_with_dash--")
        assert not pattern.match("")

    def test_roi_codebook_pattern(self):
        """Test ROI_CODEBOOK_PATTERN extracts ROI and codebook."""
        pattern = Workspace.ROI_CODEBOOK_PATTERN

        # Without codebook
        match = pattern.match("1_9_17--cortex")
        assert match.group(1) == "cortex"
        assert match.group(2) is None

        # With codebook
        match = pattern.match("registered--cortex+codebook_v1")
        assert match.group(1) == "cortex"
        assert match.group(2) == "codebook_v1"

    def test_numeric_sort_pattern(self):
        """Test NUMERIC_SORT_PATTERN for round sorting."""
        pattern = Workspace.NUMERIC_SORT_PATTERN

        # Valid numeric prefixes
        assert pattern.match("1_9_17").group(1) == "1"
        assert pattern.match("23_11_20").group(1) == "23"

        # Invalid patterns
        assert not pattern.match("round_1")
        assert not pattern.match("_1_start")


class TestWorkspaceStructureDiscovery:
    """Test workspace structure discovery methods."""

    def create_mock_workspace(self, directories: list[str]) -> Path:
        """Create mock workspace with specified directories."""
        tmpdir = tempfile.mkdtemp()
        workspace_path = Path(tmpdir)
        _write_done_sentinel(workspace_path)

        for dirname in directories:
            (workspace_path / dirname).mkdir(parents=True, exist_ok=True)

        return workspace_path

    def test_rounds_discovery_basic(self):
        """Test basic rounds discovery functionality."""
        directories = [
            "1_9_17--cortex",
            "2_10_18--cortex",
            "3_11_19--cortex",
            "1_9_17--hippocampus",
            "analysis",  # Should be filtered out
            "shifts--cortex+codebook",  # Should be filtered out
        ]

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        rounds = ws.rounds
        assert set(rounds) == {"1_9_17", "2_10_18", "3_11_19"}
        assert rounds == ["1_9_17", "2_10_18", "3_11_19"]  # Should be sorted

    def test_rounds_discovery_numerical_sorting(self):
        """Test rounds are sorted numerically by leading number."""
        directories = [
            "10_1_1--roi",
            "2_1_1--roi",
            "1_1_1--roi",
            "round_a--roi",  # Non-numeric should sort after
        ]

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        rounds = ws.rounds
        assert rounds == ["1_1_1", "2_1_1", "10_1_1", "round_a"]

    def test_rounds_forbidden_prefixes_filtered(self):
        """Test that forbidden prefixes are filtered out."""
        directories = [
            "1_9_17--roi",
            "10x--roi",
            "analysis--roi",
            "shifts--roi",
            "stitch--roi",
            "fid--roi",
            "registered--roi",
            "old--roi",
            "basic--roi",
        ]

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        rounds = ws.rounds
        assert rounds == ["1_9_17"]

    def test_rounds_empty_raises_error(self):
        """Test that empty workspace raises ValueError."""
        directories = ["analysis", "shifts--roi"]  # Only forbidden directories

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        with pytest.raises(ValueError, match="No round subdirectories found"):
            _ = ws.rounds

    def test_rois_discovery_basic(self):
        """Test basic ROI discovery functionality."""
        directories = [
            "1_9_17--cortex",
            "1_9_17--hippocampus",
            "2_10_18--cortex",
            "registered--cortex+codebook",
            "stitch--striatum",
        ]

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        rois = ws.rois
        assert set(rois) == {"cortex", "hippocampus", "striatum"}
        assert rois == ["cortex", "hippocampus", "striatum"]  # Should be sorted

    def test_rois_with_codebook_suffix(self):
        """Test ROI discovery strips codebook suffixes correctly."""
        directories = [
            "registered--cortex+codebook_v1",
            "stitch--hippocampus+cb_new",
            "segment--cortex+final_cb",
        ]

        workspace_path = self.create_mock_workspace(directories)
        ws = Workspace(workspace_path)

        rois = ws.rois
        assert set(rois) == {"cortex", "hippocampus"}


class TestWorkspaceImageAccess:
    """Test image access methods."""

    def create_mock_workspace_with_deconv(self) -> Path:
        """Create workspace with deconv directory structure."""
        tmpdir = tempfile.mkdtemp()
        workspace_path = Path(tmpdir)
        _write_done_sentinel(workspace_path)
        deconv_path = workspace_path / "analysis" / "deconv"

        # Create directory structure
        (deconv_path / "1_9_17--cortex").mkdir(parents=True)
        (deconv_path / "registered--cortex+cb").mkdir(parents=True)

        return workspace_path

    def test_deconved_property(self):
        """Test deconved property returns correct path."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        expected_path = workspace_path / "analysis" / "deconv"
        assert ws.deconved == expected_path

    def test_img_path_construction(self):
        """Test img method constructs correct paths."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        img_path = ws.img("1_9_17", "cortex", 42)
        expected_path = ws.deconved / "1_9_17--cortex" / "1_9_17-0042.tif"
        assert img_path == expected_path

    def test_img_with_read_true(self):
        """Test img method with read=True loads image data."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        img_path = ws.img("1_9_17", "cortex", 42)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        expected = np.zeros((10, 10), dtype=np.uint16)
        tifffile.imwrite(img_path, expected)

        img_data = ws.img("1_9_17", "cortex", 42, read=True)
        assert np.array_equal(img_data, expected)

    def test_registered_path_construction(self):
        """Test registered method constructs correct paths."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        reg_path = ws.registered("cortex", "codebook_v1")
        expected_path = ws.deconved / "registered--cortex+codebook_v1"
        assert reg_path == expected_path

    def test_regimg_path_construction(self):
        """Test regimg method constructs correct paths."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        regimg_path = ws.regimg("cortex", "codebook_v1", 42)
        expected_path = ws.deconved / "registered--cortex+codebook_v1" / "reg-0042.tif"
        assert regimg_path == expected_path

    def test_regimg_with_read_true(self):
        """Test regimg method with read=True loads image data."""
        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        regimg_path = ws.regimg("cortex", "cb", 42)
        regimg_path.parent.mkdir(parents=True, exist_ok=True)
        expected = np.zeros((5, 10, 10), dtype=np.uint16)
        tifffile.imwrite(regimg_path, expected)

        regimg_data = ws.regimg("cortex", "cb", 42, read=True)
        assert np.array_equal(regimg_data, expected)


class TestWorkspaceProcessingDirectories:
    """Test methods for accessing processing result directories."""

    def test_stitch_without_codebook(self, tmp_path: Path):
        """Test stitch method without codebook parameter."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        stitch_path = ws.stitch("cortex")
        expected_path = workspace_path / "analysis" / "deconv" / "stitch--cortex"
        assert stitch_path == expected_path

    def test_stitch_with_codebook(self, tmp_path: Path):
        """Test stitch method with codebook parameter."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        stitch_path = ws.stitch("cortex", "codebook_v1")
        expected_path = workspace_path / "analysis" / "deconv" / "stitch--cortex+codebook_v1"
        assert stitch_path == expected_path

    def test_segment_path_construction(self, tmp_path: Path):
        """Test segment method constructs correct paths."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        segment_path = ws.segment("cortex", "codebook_v1")
        expected_path = workspace_path / "analysis" / "deconv" / "segment--cortex+codebook_v1"
        assert segment_path == expected_path

    def test_opt_path_construction(self, tmp_path: Path):
        """Test opt method constructs correct paths with underscore pattern."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        opt_path = ws.opt("ebe_tricycle_targets")
        expected_path = workspace_path / "analysis" / "deconv" / "opt_ebe_tricycle_targets"
        assert opt_path.path == expected_path
        assert isinstance(opt_path, OptimizePath)

    def test_opt_properties_access(self, tmp_path: Path):
        """Test OptimizePath properties are accessible with real codebook names."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        opt_path = ws.opt("ebe_devprobeset_targets")
        mse_path = opt_path.mse
        scaling_path = opt_path.scaling_factor

        base_path = workspace_path / "analysis" / "deconv" / "opt_ebe_devprobeset_targets"
        assert mse_path == base_path / "mse.txt"
        assert scaling_path == base_path / "global_scale.txt"

    def test_tileconfig_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test tileconfig method loads configuration successfully."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        expected_path = ws.tileconfig_registered_txt("cortex")
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text("dim=2\n", encoding="utf-8")

        sentinel = object()

        def _fake_from_file(p: Path) -> object:
            assert p == expected_path
            return sentinel

        monkeypatch.setattr("fishtools.preprocess.tileconfig.TileConfiguration.from_file", _fake_from_file)
        assert ws.tileconfig("cortex") is sentinel

    def test_tileconfig_returns_canonical(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test tileconfig loads from the canonical location."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        expected = ws.tileconfig_registered_txt("cortex")
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text("dim=2\n", encoding="utf-8")

        sentinel = object()

        def _fake_from_file(p: Path) -> object:
            assert p == expected
            return sentinel

        monkeypatch.setattr("fishtools.preprocess.tileconfig.TileConfiguration.from_file", _fake_from_file)
        assert ws.tileconfig("cortex") is sentinel

    def test_tileconfig_file_not_found(self, tmp_path: Path):
        """Test tileconfig method raises meaningful error when file not found."""
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        with pytest.raises(FileNotFoundError, match="No registered TileConfig found at"):
            ws.tileconfig("cortex")


class TestWorkspaceFiducialsAndPositions:
    """Tests for fiducial and tile position helpers."""

    def test_fids_directory_path(self, tmp_path: Path):
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        expected = workspace_path / "analysis" / "deconv" / "fids--cortex"
        assert ws.fids("cortex") == expected

    def test_fid_path_accepts_int_and_str(self, tmp_path: Path):
        workspace_path = tmp_path / "ws"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        base = workspace_path / "analysis" / "deconv" / "fids--cortex"
        assert ws.fid("cortex", 3) == base / "fids-0003.tif"
        assert ws.fid("cortex", "0012") == base / "fids-0012.tif"

    def test_tile_positions_csv_prefers_override(self, tmp_path: Path):
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        override = tmp_path / "custom.csv"
        override.write_text("0,0,0\n", encoding="utf-8")

        ws = Workspace(workspace_root)
        assert ws.tile_positions_csv("cortex", position_file=override) == override

    def test_tile_positions_csv_from_workspace_root(self, tmp_path: Path):
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        csv_path = workspace_root / "cortex.csv"
        csv_path.write_text("0,0,0\n", encoding="utf-8")

        ws = Workspace(workspace_root)
        assert ws.tile_positions_csv("cortex") == csv_path

    def test_tile_positions_csv_missing_raises(self, tmp_path: Path):
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)

        ws = Workspace(workspace_root)
        with pytest.raises(FileNotFoundError, match="Tile position CSV not found"):
            ws.tile_positions_csv("cortex")


class TestWorkspaceRegisteredArtifacts:
    """Tests for registered TIFF discovery and validation helpers."""

    def test_registered_file_map_collects_registered_files(self, tmp_path: Path) -> None:
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        registered_dir = workspace_root / "analysis" / "deconv" / "registered--cortex+cb1"
        registered_dir.mkdir(parents=True)
        tif_path = registered_dir / "reg-0000.tif"
        tifffile.imwrite(tif_path, np.zeros((1, 1, 2, 2), dtype=np.uint16))

        ws = Workspace(workspace_root)
        mapping, missing = ws.registered_file_map("cb1")

        assert missing == []
        assert mapping == {"cortex": [tif_path]}

    def test_registered_file_map_reports_missing_roi(self, tmp_path: Path) -> None:
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        (workspace_root / "analysis" / "deconv" / "registered--cortex+other").mkdir(parents=True)

        ws = Workspace(workspace_root)
        mapping, missing = ws.registered_file_map("cb1", rois=["cortex"])

        assert mapping == {}
        assert missing == ["cortex"]

    def test_registered_codebooks_discovers_unique_sorted(self, tmp_path: Path) -> None:
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        base = workspace_root / "analysis" / "deconv"
        (base / "registered--cortex+cb2").mkdir(parents=True)
        (base / "registered--cortex+cb1").mkdir(parents=True)
        (base / "registered--hippocampus+cb3").mkdir(parents=True)
        (base / "registered--hippocampus+cb1").mkdir(parents=True)

        ws = Workspace(workspace_root)

        assert ws.registered_codebooks() == ["cb1", "cb2", "cb3"]

    def test_registered_codebooks_filters_by_rois(self, tmp_path: Path) -> None:
        workspace_root = tmp_path / "ws"
        workspace_root.mkdir()
        _write_done_sentinel(workspace_root)
        base = workspace_root / "analysis" / "deconv"
        (base / "registered--cortex+cb1").mkdir(parents=True)
        (base / "registered--hippocampus+cb2").mkdir(parents=True)
        (workspace_root / "1_0_0--striatum").mkdir(parents=True)

        ws = Workspace(workspace_root)

        assert ws.registered_codebooks(rois=["striatum"]) == []

    def test_ensure_tiff_readable_detects_corruption(self, tmp_path: Path) -> None:
        valid = tmp_path / "valid.tif"
        tifffile.imwrite(valid, np.zeros((1, 1), dtype=np.uint16))
        Workspace.ensure_tiff_readable(valid)  # Should not raise

        corrupted = tmp_path / "corrupted.tif"
        corrupted.write_bytes(b"not-a-tiff")

        with pytest.raises(CorruptedTiffError):
            Workspace.ensure_tiff_readable(corrupted)


class TestWorkspaceStringRepresentation:
    """Test string representation methods."""

    def test_str_method(self, tmp_path: Path):
        """Test __str__ returns path as string."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        assert str(ws) == str(workspace_path.resolve())

    def test_repr_method(self, tmp_path: Path):
        """Test __repr__ returns formatted representation."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        _write_done_sentinel(workspace_path)
        ws = Workspace(workspace_path)

        expected_repr = f"Workspace({workspace_path.resolve()})"
        assert repr(ws) == expected_repr


class TestWorkspaceEdgeCases:
    """Test edge cases and error conditions."""

    def test_nonexistent_directory(self):
        """Test behavior with nonexistent directory paths."""
        workspace_path = Path("/nonexistent/path")
        with pytest.raises(ValueError, match="does not exist"):
            _ = Workspace(workspace_path)

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            _write_done_sentinel(workspace_path)
            ws = Workspace(workspace_path)

            with pytest.raises(ValueError, match="No round subdirectories found"):
                _ = ws.rounds

            # ROIs should return empty list for empty directory
            assert ws.rois == []

    def test_deconved_exists_check(self):
        """Test that deconved directory existence affects path selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            _write_done_sentinel(workspace_path)
            deconv_path = workspace_path / "analysis" / "deconv"

            # Without deconv directory
            ws = Workspace(workspace_path)
            with pytest.raises(ValueError):  # No directories to parse
                _ = ws.rounds

            # With deconv directory
            deconv_path.mkdir(parents=True)
            (deconv_path / "1_9_17--roi").mkdir()

            rounds = ws.rounds
            assert rounds == ["1_9_17"]


class TestWorkspaceBackwardCompatibility:
    """Test that refactored methods maintain backward compatibility."""

    def test_rounds_compatibility_with_original(self):
        """Test that new regex-based rounds parsing produces same results as original."""
        directories = [
            "1_9_17--cortex",
            "2_10_18--hippocampus",
            "10_1_2--striatum",
            "analysis--ignore",
            "registered--cortex+cb",
        ]

        workspace_path = Path(tempfile.mkdtemp())
        _write_done_sentinel(workspace_path)
        for dirname in directories:
            (workspace_path / dirname).mkdir(parents=True, exist_ok=True)

        ws = Workspace(workspace_path)

        # Expected result based on original logic
        expected_rounds = ["1_9_17", "2_10_18", "10_1_2"]
        assert ws.rounds == expected_rounds

    def test_rois_compatibility_with_original(self):
        """Test that new regex-based ROIs parsing produces same results as original."""
        directories = [
            "1_9_17--cortex",
            "2_10_18--hippocampus",
            "registered--cortex+codebook",
            "stitch--striatum+cb",
        ]

        workspace_path = Path(tempfile.mkdtemp())
        _write_done_sentinel(workspace_path)
        for dirname in directories:
            (workspace_path / dirname).mkdir(parents=True, exist_ok=True)

        ws = Workspace(workspace_path)

        # Expected result based on original logic
        expected_rois = ["cortex", "hippocampus", "striatum"]
        assert ws.rois == expected_rois


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
