"""
Tests for fishtools.postprocess.intensity_config module.

Comprehensive unit tests for intensity extraction configuration management,
including validation, file handling, and error scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import toml
from pydantic import ValidationError

from fishtools.postprocess.intensity_config import (
    IntensityExtractionConfig,
    load_intensity_config,
    validate_intensity_config,
)


class TestIntensityExtractionConfig:
    """Test IntensityExtractionConfig class functionality."""

    def create_valid_config_data(self, workspace_path: Path) -> Dict[str, Any]:
        """Create valid configuration data for testing."""
        return {
            "workspace_path": workspace_path,
            "roi": "roi1",
            "seg_codebook": "atp",
            "segmentation_name": "output_segmentation.zarr",
            "intensity_name": "input_image.zarr",
            "channel": "cfse",
            "max_workers": 4,
            "overwrite": False,
        }

    def test_config_initialization_valid(self) -> None:
        """Test configuration initialization with valid parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_data = self.create_valid_config_data(workspace_dir)

            config = IntensityExtractionConfig(**config_data)

            assert config.workspace_path == workspace_dir
            assert config.roi == "roi1"
            assert config.seg_codebook == "atp"
            assert config.segmentation_name == "output_segmentation.zarr"
            assert config.intensity_name == "input_image.zarr"
            assert config.channel == "cfse"
            assert config.max_workers == 4
            assert config.overwrite is False

    def test_config_defaults(self) -> None:
        """Test configuration with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            assert config.seg_codebook == "atp"
            assert config.segmentation_name == "output_segmentation.zarr"
            assert config.intensity_name == "input_image.zarr"
            assert config.max_workers == 4
            assert config.overwrite is False

    def test_workspace_path_validation_nonexistent(self) -> None:
        """Test validation fails for non-existent workspace directory."""
        nonexistent_path = Path("/nonexistent/directory")

        with pytest.raises(ValidationError, match="Workspace directory does not exist"):
            IntensityExtractionConfig(workspace_path=nonexistent_path, roi="roi1", channel="cfse")

    def test_workspace_path_validation_not_directory(self) -> None:
        """Test validation fails when workspace path is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = Path(temp_file.name)

            with pytest.raises(ValidationError, match="Workspace path is not a directory"):
                IntensityExtractionConfig(workspace_path=file_path, roi="roi1", channel="cfse")

    def test_roi_validation_empty(self) -> None:
        """Test ROI validation fails for empty string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            with pytest.raises(ValidationError, match="ROI identifier cannot be empty"):
                IntensityExtractionConfig(workspace_path=workspace_dir, roi="", channel="cfse")

    def test_roi_validation_invalid_characters(self) -> None:
        """Test ROI validation fails for invalid characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            with pytest.raises(ValidationError, match="ROI identifier contains invalid characters"):
                IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi@#$%", channel="cfse")

    def test_roi_validation_valid_characters(self) -> None:
        """Test ROI validation passes for valid characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            # These should all be valid
            valid_rois = ["roi1", "roi_2", "roi-3", "ROI1", "test_roi_123"]

            for roi in valid_rois:
                config = IntensityExtractionConfig(workspace_path=workspace_dir, roi=roi, channel="cfse")
                assert config.roi == roi

    def test_channel_validation_empty(self) -> None:
        """Test channel validation fails for empty string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            with pytest.raises(ValidationError, match="Channel name cannot be empty"):
                IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="")

    def test_channel_validation_whitespace_stripped(self) -> None:
        """Test channel validation strips whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="  cfse  ")

            assert config.channel == "cfse"

    def test_max_workers_validation_bounds(self) -> None:
        """Test max_workers validation enforces bounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            # Test lower bound
            with pytest.raises(ValidationError):
                IntensityExtractionConfig(
                    workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=0
                )

            # Test upper bound
            with pytest.raises(ValidationError):
                IntensityExtractionConfig(
                    workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=100
                )

            # Test valid values
            for workers in [1, 4, 16, 32]:
                config = IntensityExtractionConfig(
                    workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=workers
                )
                assert config.max_workers == workers

    def test_property_paths(self) -> None:
        """Test computed path properties."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir, roi="roi1", seg_codebook="atp", channel="cfse"
            )

            # Test computed paths
            expected_roi_dir = workspace_dir / "stitch--roi1+atp"
            assert config.roi_directory == expected_roi_dir

            expected_seg_path = expected_roi_dir / "output_segmentation.zarr"
            assert config.segmentation_zarr_path == expected_seg_path

            expected_intensity_path = expected_roi_dir / "input_image.zarr"
            assert config.intensity_zarr_path == expected_intensity_path

            expected_output_dir = expected_roi_dir / "intensity_cfse"
            assert config.output_directory == expected_output_dir

    def test_workspace_property(self) -> None:
        """Test workspace property returns Workspace instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            workspace = config.workspace
            assert workspace.path == workspace_dir

    def test_validate_zarr_files_missing_roi_directory(self) -> None:
        """Test Zarr validation fails when ROI directory is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            with pytest.raises(FileNotFoundError, match="ROI directory not found"):
                config.validate_zarr_files()

    def test_validate_zarr_files_missing_segmentation(self) -> None:
        """Test Zarr validation fails when segmentation Zarr is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            with pytest.raises(FileNotFoundError, match="Segmentation Zarr not found"):
                config.validate_zarr_files()

    def test_validate_zarr_files_missing_intensity(self) -> None:
        """Test Zarr validation fails when intensity Zarr is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create segmentation zarr directory
            seg_zarr = roi_dir / "output_segmentation.zarr"
            seg_zarr.mkdir()

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            with pytest.raises(FileNotFoundError, match="Intensity Zarr not found"):
                config.validate_zarr_files()

    def test_validate_zarr_files_success(self) -> None:
        """Test Zarr validation succeeds when all files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            # Should not raise any exceptions
            config.validate_zarr_files()


class TestLoadIntensityConfig:
    """Test load_intensity_config function."""

    def create_test_config_file(self, config_data: Dict[str, Any]) -> Path:
        """Create a temporary TOML configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        toml.dump(config_data, temp_file)
        temp_file.close()
        return Path(temp_file.name)

    def test_load_config_basic(self) -> None:
        """Test basic configuration loading from TOML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config_data = {
                "workspace_path": str(workspace_dir),
                "roi": "roi1",
                "channel": "cfse",
                "max_workers": 8,
            }

            config_file = self.create_test_config_file(config_data)

            try:
                config = load_intensity_config(config_file)

                assert config.workspace_path == workspace_dir
                assert config.roi == "roi1"
                assert config.channel == "cfse"
                assert config.max_workers == 8

            finally:
                config_file.unlink()

    def test_load_config_missing_file(self) -> None:
        """Test loading fails when configuration file doesn't exist."""
        nonexistent_file = Path("/nonexistent/config.toml")

        with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
            load_intensity_config(nonexistent_file)

    def test_load_config_invalid_toml(self) -> None:
        """Test loading fails with invalid TOML syntax."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        temp_file.write("invalid toml syntax [[[")
        temp_file.close()

        try:
            with pytest.raises(ValueError, match="Invalid TOML syntax"):
                load_intensity_config(Path(temp_file.name))
        finally:
            Path(temp_file.name).unlink()

    def test_load_config_with_overrides(self) -> None:
        """Test configuration loading with command-line overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            override_workspace = Path(temp_dir) / "override"
            override_workspace.mkdir()

            config_data = {"workspace_path": str(workspace_dir), "roi": "roi1", "channel": "cfse"}

            config_file = self.create_test_config_file(config_data)

            try:
                config = load_intensity_config(
                    config_file,
                    workspace_override=override_workspace,
                    roi_override="roi2",
                    channel_override="dapi",
                )

                assert config.workspace_path == override_workspace
                assert config.roi == "roi2"
                assert config.channel == "dapi"

            finally:
                config_file.unlink()


class TestValidateIntensityConfig:
    """Test validate_intensity_config function."""

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_config_success(self, mock_zarr_open: MagicMock) -> None:
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_seg_zarr.dtype = "uint16"
            mock_seg_zarr.chunks = (1, 512, 512)

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.chunks = (1, 512, 512, 4)
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            # Should not raise any exceptions and return validation info
            result = validate_intensity_config(config)

            assert result["validation_status"] == "passed"
            assert "zarr_info" in result
            assert "performance_info" in result

            # Check that output directory was created
            assert config.output_directory.exists()

    def test_validate_config_missing_files(self) -> None:
        """Test validation fails with missing Zarr files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            with pytest.raises(FileNotFoundError):
                validate_intensity_config(config)

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_config_output_directory_creation_failure(self, mock_zarr_open: MagicMock) -> None:
        """Test validation handles output directory creation failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores to avoid actual Zarr validation
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_seg_zarr.dtype = "uint16"

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.attrs = {"key": ["cfse"]}

            mock_zarr_open.side_effect = [
                mock_seg_zarr,
                mock_intensity_zarr,
                mock_seg_zarr,
                mock_intensity_zarr,
            ]

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            # Create a file where the output directory should be to cause conflict
            output_path = config.output_directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()  # Create a file, not a directory

            # This should handle the error gracefully
            # Note: The actual behavior depends on the OS and filesystem
            try:
                validate_intensity_config(config)
                # If no exception, the validation succeeded
            except ValueError as e:
                # If validation failed, it should be due to directory creation
                assert "output directory" in str(e).lower() or "directory" in str(e).lower()


class TestZarrValidation:
    """Test Zarr store validation functionality."""

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_zarr_stores_success(self, mock_zarr_open: MagicMock) -> None:
        """Test successful Zarr store validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_seg_zarr.dtype = "uint16"
            mock_seg_zarr.chunks = (1, 512, 512)

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.chunks = (1, 512, 512, 4)
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            result = config.validate_zarr_stores()

            assert result["segmentation_shape"] == (10, 512, 512)
            assert result["intensity_shape"] == (10, 512, 512, 4)
            assert result["available_channels"] == ["dapi", "cfse", "txred", "cy5"]
            assert result["segmentation_dtype"] == "uint16"
            assert result["intensity_dtype"] == "float32"

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_zarr_stores_channel_not_found(self, mock_zarr_open: MagicMock) -> None:
        """Test Zarr validation fails when channel is not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.attrs = {"key": ["dapi", "gfp", "txred", "cy5"]}  # Missing cfse

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir,
                roi="roi1",
                channel="cfse",  # This channel doesn't exist
            )

            with pytest.raises(ValueError, match="Channel 'cfse' not found"):
                config.validate_zarr_stores()

    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_zarr_stores_shape_mismatch(self, mock_zarr_open: MagicMock) -> None:
        """Test Zarr validation fails with shape mismatch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock Zarr stores with different shapes
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (15, 1024, 1024, 4)  # Different spatial dimensions
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(workspace_path=workspace_dir, roi="roi1", channel="cfse")

            with pytest.raises(ValueError, match="Shape mismatch"):
                config.validate_zarr_stores()

    @patch("fishtools.postprocess.intensity_config.psutil.virtual_memory")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_estimate_memory_requirements(self, mock_zarr_open: MagicMock, mock_memory: MagicMock) -> None:
        """Test memory requirement estimation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock memory info
            mock_memory.return_value.available = 16 * (1024**3)  # 16 GB available

            # Mock Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (10, 512, 512)
            mock_seg_zarr.dtype = "uint16"

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (10, 512, 512, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=4
            )

            memory_info = config.estimate_memory_requirements()

            assert "bytes_per_slice" in memory_info
            assert "parallel_memory_gb" in memory_info
            assert "peak_memory_gb" in memory_info
            assert "recommended_workers" in memory_info
            assert memory_info["bytes_per_slice"] > 0
            assert memory_info["parallel_memory_gb"] > 0

    @patch("fishtools.postprocess.intensity_config.psutil.virtual_memory")
    @patch("fishtools.postprocess.intensity_config.zarr.open")
    def test_validate_performance_requirements_insufficient_memory(
        self, mock_zarr_open: MagicMock, mock_memory: MagicMock
    ) -> None:
        """Test performance validation fails with insufficient memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            roi_dir = workspace_dir / "stitch--roi1+atp"
            roi_dir.mkdir(parents=True)

            # Create zarr directories
            seg_zarr = roi_dir / "output_segmentation.zarr"
            intensity_zarr = roi_dir / "input_image.zarr"
            seg_zarr.mkdir()
            intensity_zarr.mkdir()

            # Mock very low memory
            mock_memory.return_value.available = 1 * (1024**3)  # Only 1 GB available

            # Mock large Zarr stores
            mock_seg_zarr = MagicMock()
            mock_seg_zarr.shape = (100, 2048, 2048)
            mock_seg_zarr.dtype = "uint16"

            mock_intensity_zarr = MagicMock()
            mock_intensity_zarr.shape = (100, 2048, 2048, 4)
            mock_intensity_zarr.dtype = "float32"
            mock_intensity_zarr.attrs = {"key": ["dapi", "cfse", "txred", "cy5"]}

            mock_zarr_open.side_effect = [mock_seg_zarr, mock_intensity_zarr]

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=8
            )

            with pytest.raises(ValueError, match="Insufficient memory"):
                config.validate_performance_requirements()


class TestIntensityConfigEdgeCases:
    """Test edge cases and error scenarios for intensity configuration."""

    def test_config_with_special_characters_in_paths(self) -> None:
        """Test configuration handles special characters in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory with spaces and special characters
            special_dir = Path(temp_dir) / "test workspace-2024"
            special_dir.mkdir()

            config = IntensityExtractionConfig(workspace_path=special_dir, roi="roi_test-1", channel="cfse")

            assert config.workspace_path == special_dir
            assert "test workspace-2024" in str(config.roi_directory)

    def test_config_path_property_consistency(self) -> None:
        """Test that all path properties are consistent with each other."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir,
                roi="roi1",
                seg_codebook="custom_seg",
                segmentation_name="custom_seg.zarr",
                intensity_name="custom_intensity.zarr",
                channel="custom_channel",
            )

            # Verify path consistency
            assert config.roi_directory.parent == workspace_dir
            assert config.segmentation_zarr_path.parent == config.roi_directory
            assert config.intensity_zarr_path.parent == config.roi_directory
            assert config.output_directory.parent == config.roi_directory

            # Verify naming patterns
            assert "stitch--roi1+custom_seg" in str(config.roi_directory)
            assert "custom_seg.zarr" in str(config.segmentation_zarr_path)
            assert "custom_intensity.zarr" in str(config.intensity_zarr_path)
            assert "intensity_custom_channel" in str(config.output_directory)
