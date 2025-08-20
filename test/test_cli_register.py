"""
Comprehensive CLI tests for fishtools.preprocess.cli_register (Typer-based)

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

from fishtools.preprocess.cli_register_migrated import app


class TestTyperCLICommandParsing:
    """Test Typer CLI command parsing and parameter validation"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_register_app_help(self) -> None:
        """Test register app help command"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Register FISH images" in result.stdout
        assert "PATH" in result.stdout
        assert "progressive scoping" in result.stdout

    def test_register_command_help(self) -> None:
        """Test unified register command help"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "PATH" in result.stdout
        assert "ROI" in result.stdout
        assert "IDX" in result.stdout
        assert "--codebook" in result.stdout
        assert "--reference" in result.stdout
        assert "--debug" in result.stdout
        assert "--threshold" in result.stdout
        assert "--fwhm" in result.stdout
        assert "--overwrite" in result.stdout
        assert "--no-priors" in result.stdout

    def test_register_examples_help(self) -> None:
        """Test register command includes usage examples"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Examples:" in result.stdout
        assert "All files in all ROIs" in result.stdout

    def test_register_progressive_scoping_help(self) -> None:
        """Test register command shows progressive scoping pattern"""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "register /workspace" in result.stdout
        assert "register /workspace cortex" in result.stdout
        assert "/workspace cortex 42" in result.stdout

    def test_register_required_arguments(self) -> None:
        """Test register command with missing required arguments"""
        # Missing PATH argument
        result = self.runner.invoke(app, [])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout
        
        # PATH argument alone should be valid (processes all files)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(app, [temp_dir])
            # Should succeed with path alone, but fail due to no files found
            assert "No files found" in result.stdout or result.exit_code != 0

    def test_register_progressive_scoping_behavior(self) -> None:
        """Test register command progressive scoping behavior"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # In progressive scoping, path 42 means roi=42, idx=None
            result = self.runner.invoke(app, [temp_dir, "42"])  # This means roi="42", idx=None
            assert result.exit_code == 0  # Should succeed but find no files
            assert "No files found" in result.stdout

    def test_register_nonexistent_path(self) -> None:
        """Test path validation for non-existent directories"""
        result = self.runner.invoke(app, ["/nonexistent/path"])
        
        assert result.exit_code != 0
        # Typer's validation should catch this
        assert "not found" in result.stdout or "Invalid value" in result.stdout

    def test_register_invalid_idx_type(self) -> None:
        """Test IDX parameter type validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-integer IDX should fail
            result = self.runner.invoke(app, [temp_dir, "cortex", "not_an_integer"])

            assert result.exit_code != 0
            assert "Invalid value" in result.stdout or "is not a valid integer" in result.stdout

    def test_register_nonexistent_codebook(self) -> None:
        """Test codebook file validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Non-existent codebook file should fail
            result = self.runner.invoke(
                app, [temp_dir, "--codebook", "/nonexistent/codebook.json"]
            )

            assert result.exit_code != 0
            assert "not found" in result.stdout or "Invalid value" in result.stdout

    def test_threshold_parameter_validation(self) -> None:
        """Test threshold parameter type validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebook = Path(temp_dir) / "codebook.json"
            codebook.write_text("{}")
            
            # Test invalid string threshold
            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--threshold", "not_a_float"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout

            # Test valid float values (should succeed - no range constraints)
            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--threshold", "0.05"])
            assert result.exit_code == 0  # Should succeed (just find no files)
            assert "No files found" in result.stdout

            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--threshold", "25.0"])
            assert result.exit_code == 0  # Should succeed (just find no files)
            assert "No files found" in result.stdout

    def test_fwhm_parameter_validation(self) -> None:
        """Test FWHM parameter type validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            codebook = Path(temp_dir) / "codebook.json"
            codebook.write_text("{}")
            
            # Test invalid string FWHM
            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--fwhm", "not_a_float"])
            assert result.exit_code != 0
            assert "Invalid value" in result.stdout

            # Test valid float values (should succeed - no range constraints)
            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--fwhm", "0.1"])
            assert result.exit_code == 0  # Should succeed (just find no files)
            assert "No files found" in result.stdout

            result = self.runner.invoke(app, [temp_dir, "--codebook", str(codebook), "--fwhm", "25.0"])
            assert result.exit_code == 0  # Should succeed (just find no files)
            assert "No files found" in result.stdout


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

    @patch("fishtools.preprocess.cli_register_migrated.scan_workspace_once")
    @patch("fishtools.preprocess.cli_register_migrated._run")
    def test_register_single_file_execution(
        self, mock_run: Mock, mock_scan: Mock, mock_workspace: Path
    ) -> None:
        """Test successful CLI execution with single file (path roi idx)"""
        codebook_path = mock_workspace / "codebook.json"
        
        # Mock scan_workspace_once to return available files
        mock_scan.return_value = {"cortex": [42, 43, 44]}

        result = self.runner.invoke(
            app,
            [
                str(mock_workspace),
                "cortex",
                "42",
                "--codebook",
                str(codebook_path),
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

    @patch("fishtools.preprocess.cli_register_migrated.scan_workspace_once")
    @patch("fishtools.preprocess.cli_register_migrated.process_files")
    def test_register_all_files_execution(self, mock_process: Mock, mock_scan: Mock, mock_workspace: Path) -> None:
        """Test CLI execution for all files (path only)"""
        codebook_path = mock_workspace / "codebook.json"
        
        # Mock scan_workspace_once to return available files
        mock_scan.return_value = {"cortex": [42, 43], "striatum": [44, 45]}

        result = self.runner.invoke(
            app,
            [
                str(mock_workspace),
                "--codebook",
                str(codebook_path),
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

        # process_files should be called once with all target files
        mock_process.assert_called_once()
        
        # Verify the target files passed to process_files
        call_args = mock_process.call_args
        target_files = call_args[0][1]  # Second positional argument
        
        # Should process all 4 files from both ROIs
        assert len(target_files) == 4
        assert ("cortex", 42) in target_files
        assert ("cortex", 43) in target_files
        assert ("striatum", 44) in target_files
        assert ("striatum", 45) in target_files

    @patch("fishtools.preprocess.cli_register_migrated.scan_workspace_once")
    @patch("fishtools.preprocess.cli_register_migrated.process_files")
    def test_register_execution_error(self, mock_process: Mock, mock_scan: Mock) -> None:
        """Test CLI handles execution errors gracefully"""
        # Mock scan to return files, but process_files to raise an error
        mock_scan.return_value = {"test_roi": [42]}
        mock_process.side_effect = ValueError("Test execution error")

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            codebook = workspace / "codebook.json"
            codebook.write_text("{}")

            result = self.runner.invoke(app, [str(workspace), "test_roi", "42", "--codebook", str(codebook)])

            # Should propagate the error
            assert result.exit_code != 0
            assert "Test execution error" in str(result.exception) or "failed" in result.stdout.lower()


class TestTyperCLIValidation:
    """Test error handling and validation in unified register command"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_validate_nonexistent_workspace(self) -> None:
        """Test register command with non-existent workspace"""
        result = self.runner.invoke(app, ["/nonexistent/path"])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout

    @patch("fishtools.preprocess.cli_register_migrated.scan_workspace_once")
    def test_validate_basic_workspace(self, mock_scan: Mock) -> None:
        """Test register command with empty workspace (no files found)"""
        mock_scan.return_value = {}  # No files found
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            result = self.runner.invoke(app, [str(workspace)])
            
            # Should handle gracefully when no files are found
            assert result.exit_code == 0
            assert "No files found" in result.stdout

    @patch("fishtools.preprocess.cli_register_migrated.scan_workspace_once")
    def test_validate_with_codebook(self, mock_scan: Mock) -> None:
        """Test register command ROI validation"""
        mock_scan.return_value = {"cortex": [42], "striatum": [43]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            codebook = workspace / "codebook.json"
            codebook.write_text('{"1": {"gene": "test", "ch": "488"}}')
            
            # Test invalid ROI name
            result = self.runner.invoke(app, [str(workspace), "nonexistent_roi", "--codebook", str(codebook)])
            
            assert result.exit_code != 0
            assert "ROI 'nonexistent_roi' not found" in result.stdout
            assert "Available ROIs:" in result.stdout


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