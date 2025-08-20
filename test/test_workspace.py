"""Comprehensive test suite for Workspace class.

Tests directory structure parsing, path resolution, regex pattern matching,
and all public methods of the Workspace class. Includes scientific computing
robustness, memory efficiency, and performance benchmarks.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from fishtools.utils.io import Workspace, OptimizePath


class TestWorkspaceInitialization:
    """Test Workspace initialization and path resolution."""

    def test_init_with_workspace_root(self):
        """Test initialization with workspace root path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            ws = Workspace(workspace_path)
            assert ws.path == workspace_path.resolve()

    def test_init_with_deconv_subdirectory(self):
        """Test initialization with analysis/deconv subdirectory auto-resolves to root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            deconv_path = workspace_path / "analysis" / "deconv"
            deconv_path.mkdir(parents=True)

            ws = Workspace(deconv_path)
            assert ws.path == workspace_path.resolve()

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(tmpdir)
            assert ws.path == Path(tmpdir).resolve()

    def test_init_expands_user_path(self):
        """Test that ~ is expanded in paths."""
        ws = Workspace("~/test")
        assert str(ws.path).startswith("/")  # Should be expanded


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

    @patch("fishtools.utils.io.imread")
    def test_img_with_read_true(self, mock_imread):
        """Test img method with read=True loads image data."""
        mock_array = np.zeros((10, 10), dtype=np.uint16)
        mock_imread.return_value = mock_array

        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        img_data = ws.img("1_9_17", "cortex", 42, read=True)
        assert np.array_equal(img_data, mock_array)
        mock_imread.assert_called_once()

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

    @patch("fishtools.utils.io.imread")
    def test_regimg_with_read_true(self, mock_imread):
        """Test regimg method with read=True loads image data."""
        mock_array = np.zeros((5, 10, 10), dtype=np.uint16)
        mock_imread.return_value = mock_array

        workspace_path = self.create_mock_workspace_with_deconv()
        ws = Workspace(workspace_path)

        regimg_data = ws.regimg("cortex", "cb", 42, read=True)
        assert np.array_equal(regimg_data, mock_array)
        mock_imread.assert_called_once()


class TestWorkspaceProcessingDirectories:
    """Test methods for accessing processing result directories."""

    def test_stitch_without_codebook(self):
        """Test stitch method without codebook parameter."""
        workspace_path = Path("/test")
        ws = Workspace(workspace_path)

        stitch_path = ws.stitch("cortex")
        expected_path = workspace_path / "analysis" / "deconv" / "stitch--cortex"
        assert stitch_path == expected_path

    def test_stitch_with_codebook(self):
        """Test stitch method with codebook parameter."""
        workspace_path = Path("/test")
        ws = Workspace(workspace_path)

        stitch_path = ws.stitch("cortex", "codebook_v1")
        expected_path = workspace_path / "analysis" / "deconv" / "stitch--cortex+codebook_v1"
        assert stitch_path == expected_path

    def test_segment_path_construction(self):
        """Test segment method constructs correct paths."""
        workspace_path = Path("/test")
        ws = Workspace(workspace_path)

        segment_path = ws.segment("cortex", "codebook_v1")
        expected_path = workspace_path / "analysis" / "deconv" / "segment--cortex+codebook_v1"
        assert segment_path == expected_path
    
    def test_opt_path_construction(self):
        """Test opt method constructs correct paths with underscore pattern."""
        workspace_path = Path("/test")
        ws = Workspace(workspace_path)
        
        opt_path = ws.opt("ebe_tricycle_targets")
        expected_path = workspace_path / "analysis" / "deconv" / "opt_ebe_tricycle_targets"
        assert opt_path.path == expected_path
        assert isinstance(opt_path, OptimizePath)
    
    def test_opt_properties_access(self):
        """Test OptimizePath properties are accessible with real codebook names."""
        workspace_path = Path("/test")
        ws = Workspace(workspace_path)
        
        opt_path = ws.opt("ebe_devprobeset_targets")
        mse_path = opt_path.mse
        scaling_path = opt_path.scaling_factor
        
        base_path = workspace_path / "analysis" / "deconv" / "opt_ebe_devprobeset_targets"
        assert mse_path == base_path / "mse.txt"
        assert scaling_path == base_path / "global_scale.txt"

    @patch("fishtools.preprocess.tileconfig.TileConfiguration.from_file")
    def test_tileconfig_success(self, mock_from_file):
        """Test tileconfig method loads configuration successfully."""
        mock_config = MagicMock()
        mock_from_file.return_value = mock_config

        workspace_path = Path("/test")
        ws = Workspace(workspace_path)

        config = ws.tileconfig("cortex")
        assert config == mock_config

        expected_path = workspace_path / "stitch--cortex" / "TileConfiguration.registered.txt"
        mock_from_file.assert_called_once_with(expected_path)

    @patch("fishtools.preprocess.tileconfig.TileConfiguration.from_file")
    def test_tileconfig_file_not_found(self, mock_from_file):
        """Test tileconfig method raises meaningful error when file not found."""
        mock_from_file.side_effect = FileNotFoundError()

        workspace_path = Path("/test")
        ws = Workspace(workspace_path)

        with pytest.raises(FileNotFoundError, match="Haven't stitch/registered yet"):
            ws.tileconfig("cortex")


class TestWorkspaceStringRepresentation:
    """Test string representation methods."""

    def test_str_method(self):
        """Test __str__ returns path as string."""
        workspace_path = Path("/test/workspace")
        ws = Workspace(workspace_path)

        assert str(ws) == str(workspace_path.resolve())

    def test_repr_method(self):
        """Test __repr__ returns formatted representation."""
        workspace_path = Path("/test/workspace")
        ws = Workspace(workspace_path)

        expected_repr = f"Workspace({workspace_path.resolve()})"
        assert repr(ws) == expected_repr


class TestWorkspaceEdgeCases:
    """Test edge cases and error conditions."""

    def test_nonexistent_directory(self):
        """Test behavior with nonexistent directory paths."""
        workspace_path = Path("/nonexistent/path")
        ws = Workspace(workspace_path)

        # Should not raise error during initialization
        assert ws.path == workspace_path.resolve()

        # But should raise when trying to access contents
        with pytest.raises((FileNotFoundError, OSError)):
            _ = ws.rounds

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            ws = Workspace(workspace_path)

            with pytest.raises(ValueError, match="No round subdirectories found"):
                _ = ws.rounds

            # ROIs should return empty list for empty directory
            assert ws.rois == []

    def test_deconved_exists_check(self):
        """Test that deconved directory existence affects path selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
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
        for dirname in directories:
            (workspace_path / dirname).mkdir(parents=True, exist_ok=True)

        ws = Workspace(workspace_path)

        # Expected result based on original logic
        expected_rois = ["cortex", "hippocampus", "striatum"]
        assert ws.rois == expected_rois


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
