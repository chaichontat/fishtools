"""
Tests for fishtools.postprocess CLI functionality.

Tests the CLI interface including configuration loading, validation, and error handling.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import toml
from click.testing import CliRunner

from fishtools.postprocess.cli_concat import concat


class TestConcatCLI:
    """Test concat CLI command functionality."""

    def create_test_config_file(self, config_data: Dict[str, Any]) -> Path:
        """Create a temporary TOML configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        toml.dump(config_data, temp_file)
        temp_file.close()
        return Path(temp_file.name)

    def create_valid_config_data(self) -> Dict[str, Any]:
        """Create valid configuration data for testing."""
        return {
            "workspace_path": "/tmp/test_data/test_experiment",
            "codebooks": ["mousecommon", "zachDE"],
            "seg_codebook": "atp",
            "intensity": {"channel": "cfse", "required": True},
            "quality_control": {"min_counts": 40, "max_counts": 1200, "min_cells": 10, "min_area": 500.0},
            "spatial": {"max_columns": 2, "padding": 100.0},
            "output": {
                "h5ad_path": "concatenated.h5ad",
                "baysor_export": True,
                "baysor_path": "baysor/spots.csv",
            },
        }

    def test_concat_command_missing_config(self) -> None:
        """Test CLI command fails when config file is missing."""
        runner = CliRunner()
        result = runner.invoke(concat, ["--config", "/non/existent/config.toml"])

        assert result.exit_code != 0
        assert "does not" in result.output and "exist" in result.output

    def test_concat_command_invalid_toml(self) -> None:
        """Test CLI command fails with invalid TOML syntax."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        temp_file.write("invalid toml syntax [[[")
        temp_file.close()

        runner = CliRunner()
        try:
            result = runner.invoke(concat, ["--config", temp_file.name])

            assert result.exit_code != 0
            assert "Configuration error" in result.output
        finally:
            Path(temp_file.name).unlink()

    def test_concat_command_dry_run_valid_config(self) -> None:
        """Test CLI dry run with valid configuration."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        # Create temporary workspace directory structure for validation
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            # Update config to use the temporary directory
            config_data["workspace_path"] = str(workspace_dir)
            config_file.unlink()  # Remove old file
            config_file = self.create_test_config_file(config_data)

            runner = CliRunner()
            try:
                result = runner.invoke(concat, ["--config", str(config_file), "--dry-run"])

                assert result.exit_code == 0
                assert "Configuration validation passed" in result.output
                assert "Workspace:" in result.output
                assert "Codebooks: ['mousecommon', 'zachDE']" in result.output

            finally:
                config_file.unlink()

    def test_concat_command_dry_run_missing_workspace(self) -> None:
        """Test CLI dry run fails with missing workspace directory."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        runner = CliRunner()
        try:
            result = runner.invoke(concat, ["--config", str(config_file), "--dry-run"])

            assert result.exit_code != 0
            assert "Workspace directory not found" in result.output

        finally:
            config_file.unlink()

    def test_concat_command_workspace_override(self) -> None:
        """Test CLI workspace override functionality."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        # Create temporary workspace directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            runner = CliRunner()
            try:
                result = runner.invoke(
                    concat, ["--config", str(config_file), "--workspace", str(workspace_dir), "--dry-run"]
                )

                assert result.exit_code == 0
                assert "Configuration validation passed" in result.output
                assert str(workspace_dir) in result.output

            finally:
                config_file.unlink()

    def test_concat_command_output_override(self) -> None:
        """Test CLI output override functionality."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        # Create temporary workspace directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            # Update config to use the temporary directory
            config_data["workspace_path"] = str(workspace_dir)
            config_file.unlink()  # Remove old file
            config_file = self.create_test_config_file(config_data)

            runner = CliRunner()
            try:
                result = runner.invoke(
                    concat, ["--config", str(config_file), "--output", "custom_output.h5ad", "--dry-run"]
                )

                assert result.exit_code == 0
                assert "Configuration validation passed" in result.output
                assert "custom_output.h5ad" in result.output

            finally:
                config_file.unlink()

    def test_concat_command_verbose_levels(self) -> None:
        """Test CLI verbose output levels."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        # Create temporary workspace directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            # Update config to use the temporary directory
            config_data["workspace_path"] = str(workspace_dir)
            config_file.unlink()  # Remove old file
            config_file = self.create_test_config_file(config_data)

            runner = CliRunner()
            try:
                # Test different verbosity levels
                for verbose_flag in ["-v", "-vv", "-vvv"]:
                    result = runner.invoke(concat, ["--config", str(config_file), verbose_flag, "--dry-run"])

                    assert result.exit_code == 0
                    assert "Configuration validation passed" in result.output

            finally:
                config_file.unlink()

    def test_concat_command_help(self) -> None:
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(concat, ["--help"])

        assert result.exit_code == 0
        assert "Concatenate multi-ROI FISH data" in result.output
        assert "--config" in result.output
        assert "--workspace" in result.output
        assert "--output" in result.output
        assert "--dry-run" in result.output
        assert "--verbose" in result.output

    def test_concat_command_pipeline_execution_mock(self) -> None:
        """Test CLI pipeline execution with mocked data processing."""
        config_data = self.create_valid_config_data()
        config_file = self.create_test_config_file(config_data)

        # Create temporary workspace directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            # Update config to use the temporary directory
            config_data["workspace_path"] = str(workspace_dir)
            config_file.unlink()  # Remove old file
            config_file = self.create_test_config_file(config_data)

            runner = CliRunner()
            try:
                # Run without --dry-run to test pipeline execution
                # This will fail due to missing data files, but should show the right error message
                result = runner.invoke(concat, ["--config", str(config_file)])

                # Pipeline should fail due to missing data files
                assert result.exit_code != 0
                assert (
                    "Error:" in result.output
                    or "Configuration error:" in result.output
                    or "No identification files found" in result.output
                )

            finally:
                config_file.unlink()
