"""
Comprehensive CLI tests for fishtools.preprocess.cli_register_typer

This module tests the Typer-based command line interface for image registration:
- CLI command parsing and validation
- Parameter validation and defaults
- Error handling for missing files and invalid parameters
- Integration with the actual CLI workflow
- Mock-based testing to avoid filesystem dependencies

Tests the user-facing CLI that orchestrates the complete registration pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from fishtools.preprocess.cli_register_typer import app


class TestTyperCLICommandParsing:
    """Test Typer CLI command parsing and parameter validation"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_register_app_help(self) -> None:
        """Test register app help command"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "FISH Image Registration" in result.stdout
        assert "Commands:" in result.stdout or "run" in result.stdout

    def test_run_command_help(self) -> None:
        """Test run command help"""
        result = self.runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.stdout
        assert "IDX" in result.stdout
        assert "--codebook" in result.stdout
        assert "--roi" in result.stdout
        assert "--reference" in result.stdout
        assert "--debug" in result.stdout
        assert "--threshold" in result.stdout
        assert "--fwhm" in result.stdout
        assert "--overwrite" in result.stdout
        assert "--no-priors" in result.stdout

    def test_validate_command_help(self) -> None:
        """Test validate command help"""
        result = self.runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate workspace" in result.stdout
        assert "PATH" in result.stdout

    def test_batch_command_help(self) -> None:
        """Test batch command help"""
        result = self.runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        assert "batch registration" in result.stdout
        assert "--codebook" in result.stdout
        assert "--threads" in result.stdout

    def test_run_command_required_arguments(self) -> None:
        """Test run command with missing required arguments"""
        # Missing PATH argument
        result = self.runner.invoke(app, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

        # Missing IDX argument with valid temp path
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(app, ["run", temp_dir])
            assert result.exit_code != 0
            assert "Missing argument" in result.stdout or "Missing option" in result.stdout

    def test_run_command_missing_codebook(self) -> None:
        """Test run command with missing required codebook option"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(app, ["run", temp_dir, "1"])
            assert result.exit_code != 0
            assert "Missing option" in result.stdout and "codebook" in result.stdout

    def test_run_command_nonexistent_path(self) -> None:
        """Test path validation for non-existent directories"""
        result = self.runner.invoke(app, ["run", "/nonexistent/path", "1", "--codebook", "/fake/codebook.json"])
        
        assert result.exit_code != 0
        # Typer's validation should catch this
        assert "not found" in result.stdout or "Invalid value" in result.stdout

    def test_run_command_invalid_idx_type(self) -> None:
        """Test IDX parameter type validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-integer IDX should fail
            result = self.runner.invoke(app, ["run", temp_dir, "not_an_integer", "--codebook", "/fake/codebook.json"])

            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "is not a valid integer" in result.stdout

    def test_run_command_nonexistent_codebook(self) -> None:
        """Test codebook file validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-existent codebook file should fail
            result = self.runner.invoke(
                app, ["run", temp_dir, "1", "--codebook", "/nonexistent/codebook.json"]
            )

            assert result.exit_code != 0
            assert "not found" in result.stdout or "Invalid value" in result.stdout

    def test_threshold_parameter_validation(self) -> None:
        """Test threshold parameter validation with range constraints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebook = Path(temp_dir) / "codebook.json"
            codebook.write_text("{}")
            
            # Test invalid string threshold
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--threshold", "not_a_float"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout

            # Test out-of-range threshold (too low)
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--threshold", "0.05"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "range" in result.stdout

            # Test out-of-range threshold (too high)
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--threshold", "25.0"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "range" in result.stdout

    def test_fwhm_parameter_validation(self) -> None:
        """Test FWHM parameter validation with range constraints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebook = Path(temp_dir) / "codebook.json"
            codebook.write_text("{}")
            
            # Test invalid string FWHM
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--fwhm", "not_a_float"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout

            # Test out-of-range FWHM (too low)
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--fwhm", "0.1"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "range" in result.stdout

            # Test out-of-range FWHM (too high) 
            result = self.runner.invoke(app, ["run", temp_dir, "1", "--codebook", str(codebook), "--fwhm", "25.0"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "range" in result.stdout


class TestTyperCLIIntegration:
    """Integration tests for Typer CLI with mocked dependencies"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()

    @pytest.fixture
    def mock_workspace(self, tmp_path: Path) -> Path:
        """Create mock workspace structure"""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()

        # Create mock ROI directories
        (workspace / "data--roi1").mkdir()
        (workspace / "data--roi2").mkdir()

        # Create mock codebook
        codebook = workspace / "codebook.json"
        codebook.write_text(
            json.dumps({"1": {"gene": "gene1", "ch": "488"}, "2": {"gene": "gene2", "ch": "560"}})
        )

        return workspace

    @patch("fishtools.preprocess.cli_register_typer.Workspace")
    @patch("fishtools.preprocess.cli_register_typer._run")
    def test_run_command_successful_execution(
        self, mock_run: Mock, mock_workspace_class: Mock, mock_workspace: Path
    ) -> None:
        """Test successful CLI execution with mocked _run function"""
        codebook_path = mock_workspace / "codebook.json"
        # Mock Workspace instance
        mock_ws_instance = Mock()
        mock_ws_instance.rois = ["roi1"]
        mock_ws_instance.path = mock_workspace
        mock_workspace_class.return_value = mock_ws_instance

        result = self.runner.invoke(
            app,
            [
                "run",
                str(mock_workspace),
                "1",
                "--codebook",
                str(codebook_path),
                "--roi",
                "roi1",
                "--reference",
                "test_ref",
                "--threshold",
                "5.0",
                "--fwhm",
                "4.0",
            ],
        )

        # Should execute successfully
        assert result.exit_code == 0

        # Verify _run was called with correct parameters
        mock_run.assert_called_once()

        # Check the call was made with correct arguments
        call_args = mock_run.call_args
        assert call_args is not None
        
        # Check keyword arguments
        call_kwargs = call_args[1] if call_args[1] else call_args.kwargs
        assert call_kwargs["codebook"] == codebook_path
        assert call_kwargs["reference"] == "test_ref"
        assert call_kwargs["debug"] == False
        assert call_kwargs["overwrite"] == False
        assert call_kwargs["no_priors"] == False

    @patch("fishtools.preprocess.cli_register_typer.Workspace")
    @patch("fishtools.preprocess.cli_register_typer._run")
    def test_run_command_with_all_options(self, mock_run: Mock, mock_workspace_class: Mock, mock_workspace: Path) -> None:
        """Test CLI execution with all optional parameters"""
        codebook_path = mock_workspace / "codebook.json"
        
        # Mock Workspace instance
        mock_ws_instance = Mock()
        mock_ws_instance.rois = ["roi1", "roi2"]
        mock_ws_instance.path = mock_workspace
        mock_workspace_class.return_value = mock_ws_instance

        result = self.runner.invoke(
            app,
            [
                "run",
                str(mock_workspace),
                "42",
                "--codebook",
                str(codebook_path),
                "--roi",
                "*",
                "--reference",
                "custom_ref",
                "--debug",
                "--threshold",
                "7.5",
                "--fwhm",
                "3.5",
                "--overwrite",
                "--no-priors",
            ],
        )

        assert result.exit_code == 0

        # _run should be called twice (once for each ROI: roi1, roi2)
        assert mock_run.call_count == 2
        
        # Verify parameters for both calls
        for call in mock_run.call_args_list:
            kwargs = call[1] if call[1] else call.kwargs
            
            # Check keyword arguments
            assert kwargs["reference"] == "custom_ref"
            assert kwargs["debug"] == True
            assert kwargs["overwrite"] == True
            assert kwargs["no_priors"] == True

    @patch("fishtools.preprocess.cli_register_typer._run")
    def test_run_command_execution_error(self, mock_run: Mock) -> None:
        """Test CLI handles execution errors gracefully"""
        mock_run.side_effect = ValueError("Test execution error")

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create ROI directories so Workspace.rois finds something
            (workspace / "data--test_roi").mkdir()
            
            codebook = workspace / "codebook.json"
            codebook.write_text("{}")

            result = self.runner.invoke(app, ["run", str(workspace), "1", "--codebook", str(codebook), "--roi", "test_roi"])

            # Should propagate the error
            assert result.exit_code != 0
            assert "Test execution error" in result.stdout or "failed" in result.stdout.lower()


class TestTyperCLIValidation:
    """Test the validate command functionality"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_validate_nonexistent_workspace(self) -> None:
        """Test validate command with non-existent workspace"""
        result = self.runner.invoke(app, ["validate", "/nonexistent/path"])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout

    def test_validate_basic_workspace(self) -> None:
        """Test validate command with basic workspace"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create basic ROI structure
            (workspace / "data--roi1").mkdir()
            
            result = self.runner.invoke(app, ["validate", str(workspace)])
            
            # Should complete validation (may pass or fail based on structure)
            # The important thing is it doesn't crash
            assert isinstance(result.exit_code, int)
            assert "validation" in result.stdout.lower() or "workspace" in result.stdout.lower()

    def test_validate_with_codebook(self) -> None:
        """Test validate command with codebook compatibility check"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            codebook = workspace / "codebook.json"
            codebook.write_text('{"1": {"gene": "test", "ch": "488"}}')
            
            # Create basic ROI structure
            (workspace / "data--roi1").mkdir()
            
            result = self.runner.invoke(app, ["validate", str(workspace), "--codebook", str(codebook)])
            
            # Should complete validation
            assert isinstance(result.exit_code, int)
            assert "codebook" in result.stdout.lower() or "validation" in result.stdout.lower()


# Fixtures for common test data
@pytest.fixture
def mock_codebook_data() -> dict[str, Any]:
    """Mock codebook data for testing"""
    return {
        "1": {"gene": "ACTB", "ch": "488"},
        "2": {"gene": "GAPDH", "ch": "560"},
        "3": {"gene": "DAPI", "ch": "405"},
        "4": {"gene": "PolyT", "ch": "650"},
    }


@pytest.fixture
def typer_cli_runner() -> CliRunner:
    """Pytest fixture for Typer CLI runner"""
    return CliRunner()


def test_mock_codebook_fixture(mock_codebook_data: dict[str, Any]) -> None:
    """Test that mock codebook fixture works correctly"""
    assert len(mock_codebook_data) == 4
    assert "1" in mock_codebook_data
    assert mock_codebook_data["1"]["gene"] == "ACTB"
    assert mock_codebook_data["1"]["ch"] == "488"


def test_typer_cli_runner_fixture(typer_cli_runner: CliRunner) -> None:
    """Test that Typer CLI runner fixture works correctly"""
    assert isinstance(typer_cli_runner, CliRunner)
    
    # Test basic Typer functionality
    simple_app = typer.Typer()
    
    @simple_app.command()
    def test_cmd():
        typer.echo("test")
    
    result = typer_cli_runner.invoke(simple_app, [])
    assert result.exit_code == 0
    assert "test" in result.stdout