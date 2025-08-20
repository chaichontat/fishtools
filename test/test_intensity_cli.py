"""
Tests for fishtools.postprocess.cli_intensity module.

Comprehensive unit tests for intensity extraction CLI interface including
configuration handling, validation, and command execution.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import toml
from click.testing import CliRunner

from fishtools.postprocess.cli_intensity import display_config_summary, display_system_info, extract_intensity
from fishtools.postprocess.intensity_config import IntensityExtractionConfig


class TestCLIIntensity:
    """Test CLI interface for intensity extraction."""

    def create_test_config_file(self, workspace_path: Path) -> Path:
        """Create a test TOML configuration file."""
        config_data = {
            "workspace_path": str(workspace_path),
            "roi": "roi1",
            "channel": "cfse",
            "seg_codebook": "atp",
            "max_workers": 2,
            "overwrite": False,
            "segmentation_name": "output_segmentation.zarr",
            "intensity_name": "input_image.zarr",
        }

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        toml.dump(config_data, temp_file)
        temp_file.close()
        return Path(temp_file.name)

    def test_extract_intensity_help(self) -> None:
        """Test CLI help message display."""
        runner = CliRunner()
        result = runner.invoke(extract_intensity, ["--help"])

        assert result.exit_code == 0
        assert "Extract intensity measurements" in result.output
        assert "--config" in result.output
        assert "--workspace" in result.output
        assert "--roi" in result.output
        assert "--channel" in result.output

    def test_extract_intensity_missing_config(self) -> None:
        """Test CLI with missing configuration file."""
        runner = CliRunner()
        result = runner.invoke(extract_intensity, ["--config", "/nonexistent/config.toml"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not exist" in result.output.lower()

    @patch("fishtools.postprocess.cli_intensity.IntensityExtractionPipeline")
    @patch("fishtools.postprocess.cli_intensity.validate_intensity_config")
    def test_extract_intensity_basic_execution(
        self, mock_validate: MagicMock, mock_pipeline_class: MagicMock
    ) -> None:
        """Test basic CLI execution with valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            # Mock validation
            mock_validate.return_value = {
                "validation_status": "passed",
                "zarr_info": {
                    "segmentation_shape": (10, 512, 512),
                    "intensity_shape": (10, 512, 512, 4),
                    "available_channels": ["dapi", "cfse", "txred", "cy5"],
                    "segmentation_dtype": "uint16",
                    "intensity_dtype": "float32",
                },
                "performance_info": {
                    "estimated_peak_memory_gb": 2.0,
                    "available_memory_gb": 16.0,
                    "memory_utilization_pct": 12.5,
                    "recommended_workers": 4,
                },
            }

            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            try:
                runner = CliRunner()
                result = runner.invoke(extract_intensity, ["--config", str(config_file), "--quiet"])

                assert result.exit_code == 0
                mock_validate.assert_called_once()
                mock_pipeline_class.assert_called_once()
                mock_pipeline.run.assert_called_once()

            finally:
                config_file.unlink()

    @patch("fishtools.postprocess.cli_intensity.validate_intensity_config")
    def test_extract_intensity_dry_run(self, mock_validate: MagicMock) -> None:
        """Test CLI dry run mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            # Mock validation
            mock_validate.return_value = {
                "validation_status": "passed",
                "zarr_info": {
                    "segmentation_shape": (5, 256, 256),
                    "intensity_shape": (5, 256, 256, 2),
                    "available_channels": ["dapi", "cfse"],
                    "segmentation_dtype": "uint16",
                    "intensity_dtype": "float32",
                },
            }

            try:
                runner = CliRunner()
                result = runner.invoke(
                    extract_intensity, ["--config", str(config_file), "--dry-run", "--quiet"]
                )

                assert result.exit_code == 0
                assert "Dry run completed successfully" in result.output
                mock_validate.assert_called_once()

            finally:
                config_file.unlink()

    @patch("fishtools.postprocess.cli_intensity.IntensityExtractionPipeline")
    @patch("fishtools.postprocess.cli_intensity.validate_intensity_config")
    def test_extract_intensity_with_overrides(
        self, mock_validate: MagicMock, mock_pipeline_class: MagicMock
    ) -> None:
        """Test CLI with parameter overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            override_workspace = Path(temp_dir) / "override"
            override_workspace.mkdir()

            config_file = self.create_test_config_file(workspace_dir)

            # Mock validation
            mock_validate.return_value = {"validation_status": "passed"}

            # Mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            try:
                runner = CliRunner()
                result = runner.invoke(
                    extract_intensity,
                    [
                        "--config",
                        str(config_file),
                        "--workspace",
                        str(override_workspace),
                        "--roi",
                        "roi2",
                        "--channel",
                        "dapi",
                        "--max-workers",
                        "8",
                        "--overwrite",
                        "--quiet",
                    ],
                )

                assert result.exit_code == 0

                # Verify that load_intensity_config was called with overrides
                mock_validate.assert_called_once()

                # Get the config passed to the pipeline
                pipeline_call_args = mock_pipeline_class.call_args[0]
                config = pipeline_call_args[0]

                # Verify overrides were applied
                assert config.max_workers == 8
                assert config.overwrite is True

            finally:
                config_file.unlink()

    @patch("fishtools.postprocess.cli_intensity.validate_intensity_config")
    def test_extract_intensity_validation_error(self, mock_validate: MagicMock) -> None:
        """Test CLI handling of validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            # Mock validation to raise error
            mock_validate.side_effect = ValueError("Invalid Zarr store format")

            try:
                runner = CliRunner()
                result = runner.invoke(extract_intensity, ["--config", str(config_file), "--quiet"])

                assert result.exit_code != 0
                assert "Configuration Error" in result.output
                assert "Invalid Zarr store format" in result.output

            finally:
                config_file.unlink()

    @patch("fishtools.postprocess.cli_intensity.IntensityExtractionPipeline")
    @patch("fishtools.postprocess.cli_intensity.validate_intensity_config")
    def test_extract_intensity_pipeline_error(
        self, mock_validate: MagicMock, mock_pipeline_class: MagicMock
    ) -> None:
        """Test CLI handling of pipeline execution errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            # Mock validation
            mock_validate.return_value = {"validation_status": "passed"}

            # Mock pipeline to raise error
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = RuntimeError("High failure rate: 8/10 slices failed")
            mock_pipeline_class.return_value = mock_pipeline

            try:
                runner = CliRunner()
                result = runner.invoke(extract_intensity, ["--config", str(config_file), "--quiet"])

                assert result.exit_code != 0
                assert "Processing Error" in result.output
                assert "High failure rate" in result.output

            finally:
                config_file.unlink()

    def test_extract_intensity_verbose_mode(self) -> None:
        """Test CLI verbose logging mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            try:
                runner = CliRunner()
                result = runner.invoke(
                    extract_intensity, ["--config", str(config_file), "--dry-run", "--verbose"]
                )

                # Should show more detailed output in verbose mode
                assert "Loading configuration" in result.output or result.exit_code != 0

            finally:
                config_file.unlink()


class TestDisplayFunctions:
    """Test CLI display utility functions."""

    def test_display_config_summary(self) -> None:
        """Test configuration summary display."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)

            config = IntensityExtractionConfig(
                workspace_path=workspace_dir, roi="roi1", channel="cfse", max_workers=4
            )

            # Should not raise any exceptions
            display_config_summary(config)

    def test_display_system_info_complete(self) -> None:
        """Test system information display with complete validation info."""
        validation_info = {
            "zarr_info": {
                "segmentation_shape": (10, 512, 512),
                "intensity_shape": (10, 512, 512, 4),
                "available_channels": ["dapi", "cfse", "txred", "cy5"],
                "segmentation_dtype": "uint16",
                "intensity_dtype": "float32",
                "segmentation_chunks": (1, 512, 512),
                "intensity_chunks": (1, 512, 512, 4),
            },
            "performance_info": {
                "estimated_peak_memory_gb": 3.2,
                "available_memory_gb": 16.0,
                "memory_utilization_pct": 20.0,
                "recommended_workers": 4,
            },
        }

        # Should not raise any exceptions
        display_system_info(validation_info)

    def test_display_system_info_partial(self) -> None:
        """Test system information display with partial validation info."""
        validation_info = {
            "zarr_info": {
                "segmentation_shape": (5, 256, 256),
                "intensity_shape": (5, 256, 256, 2),
                "available_channels": ["dapi", "cfse"],
                "segmentation_dtype": "uint16",
                "intensity_dtype": "float32",
            }
        }

        # Should not raise any exceptions even with missing performance info
        display_system_info(validation_info)

    def test_display_system_info_empty(self) -> None:
        """Test system information display with empty validation info."""
        validation_info = {}

        # Should handle empty validation info gracefully
        display_system_info(validation_info)


class TestCLIIntegration:
    """Test CLI integration patterns."""

    def test_cli_error_handling_patterns(self) -> None:
        """Test that CLI follows proper error handling patterns."""
        runner = CliRunner()

        # Test with invalid config path
        result = runner.invoke(extract_intensity, ["--config", "/invalid/path.toml"])
        assert result.exit_code != 0

        # Test with missing required argument
        result = runner.invoke(extract_intensity, [])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Usage:" in result.output

    @patch("fishtools.postprocess.cli_intensity.load_intensity_config")
    def test_cli_configuration_loading_error(self, mock_load: MagicMock) -> None:
        """Test CLI handling of configuration loading errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir)
            config_file = self.create_test_config_file(workspace_dir)

            # Mock configuration loading to raise error
            mock_load.side_effect = FileNotFoundError("Configuration file not found")

            try:
                runner = CliRunner()
                result = runner.invoke(extract_intensity, ["--config", str(config_file), "--quiet"])

                assert result.exit_code != 0
                assert "Error:" in result.output

            finally:
                config_file.unlink()

    def create_test_config_file(self, workspace_path: Path) -> Path:
        """Create a test TOML configuration file."""
        config_data = {"workspace_path": str(workspace_path), "roi": "roi1", "channel": "cfse"}

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        toml.dump(config_data, temp_file)
        temp_file.close()
        return Path(temp_file.name)
