"""
Tests for fishtools.postprocess.concat_config module.

Comprehensive unit tests for configuration loading, validation, and error handling.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import toml
from pydantic import ValidationError

from fishtools.postprocess.concat_config import (
    ConcatConfig,
    IntensityConfig,
    OutputConfig,
    QualityControlConfig,
    SpatialConfig,
    load_concat_config,
    validate_config_workspace,
)


class TestConcatConfigBasics:
    """Test basic ConcatConfig functionality."""

    def test_codebooks_empty_validation(self) -> None:
        """Test validation fails for empty codebooks list."""
        with pytest.raises(ValidationError, match="Codebooks list cannot be empty"):
            ConcatConfig(
                workspace_path=Path("/tmp"),
                codebooks=[],
                seg_codebook="atp",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )


class TestIntensityConfig:
    """Test IntensityConfig functionality."""

    def test_intensity_config_defaults(self) -> None:
        """Test default values for intensity configuration."""
        config = IntensityConfig()
        assert config.channel == "cfse"
        assert config.required is True

    def test_intensity_config_custom(self) -> None:
        """Test custom intensity configuration."""
        config = IntensityConfig(channel="edu", required=False)
        assert config.channel == "edu"
        assert config.required is False


class TestQualityControlConfig:
    """Test QualityControlConfig functionality."""

    def test_qc_config_defaults(self) -> None:
        """Test default values for quality control configuration."""
        config = QualityControlConfig()
        assert config.min_counts == 40
        assert config.max_counts == 1200
        assert config.min_cells == 10
        assert config.min_area == 500.0

    def test_qc_config_custom(self) -> None:
        """Test custom quality control configuration."""
        config = QualityControlConfig(min_counts=50, max_counts=1000, min_cells=5, min_area=300.0)
        assert config.min_counts == 50
        assert config.max_counts == 1000
        assert config.min_cells == 5
        assert config.min_area == 300.0

    def test_qc_config_negative_validation(self) -> None:
        """Test validation fails for negative values."""
        with pytest.raises(ValidationError, match="Count and cell parameters must be positive"):
            QualityControlConfig(min_counts=-1)

        with pytest.raises(ValidationError, match="Count and cell parameters must be positive"):
            QualityControlConfig(max_counts=0)

        with pytest.raises(ValidationError, match="Minimum area must be positive"):
            QualityControlConfig(min_area=-10.0)


class TestSpatialConfig:
    """Test SpatialConfig functionality."""

    def test_spatial_config_defaults(self) -> None:
        """Test default values for spatial configuration."""
        config = SpatialConfig()
        assert config.max_columns == 2
        assert config.padding == 100.0

    def test_spatial_config_custom(self) -> None:
        """Test custom spatial configuration."""
        config = SpatialConfig(max_columns=3, padding=50.0)
        assert config.max_columns == 3
        assert config.padding == 50.0

    def test_spatial_config_validation(self) -> None:
        """Test validation for spatial configuration."""
        with pytest.raises(ValidationError, match="Max columns must be positive"):
            SpatialConfig(max_columns=0)

        with pytest.raises(ValidationError, match="Padding cannot be negative"):
            SpatialConfig(padding=-1.0)


class TestOutputConfig:
    """Test OutputConfig functionality."""

    def test_output_config_defaults(self) -> None:
        """Test default values for output configuration."""
        config = OutputConfig()
        assert config.h5ad_path == Path("concatenated.h5ad")
        assert config.baysor_export is True
        assert config.baysor_path == Path("baysor/spots.csv")

    def test_output_config_custom(self) -> None:
        """Test custom output configuration."""
        config = OutputConfig(
            h5ad_path=Path("custom.h5ad"), baysor_export=False, baysor_path=Path("output/baysor.csv")
        )
        assert config.h5ad_path == Path("custom.h5ad")
        assert config.baysor_export is False
        assert config.baysor_path == Path("output/baysor.csv")


class TestConcatConfig:
    """Test ConcatConfig functionality."""

    def create_valid_config_data(self) -> Dict[str, Any]:
        """Create valid configuration data for testing."""
        return {
            "workspace_path": "/working/data/test_experiment",
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

    def test_concat_config_valid(self) -> None:
        """Test valid complete configuration."""
        config_data = self.create_valid_config_data()
        config = ConcatConfig(**config_data)

        assert config.workspace_path == Path("/working/data/test_experiment")
        assert config.codebooks == ["mousecommon", "zachDE"]
        assert config.seg_codebook == "atp"
        assert config.intensity.channel == "cfse"
        assert config.quality_control.min_counts == 40
        assert config.spatial.max_columns == 2
        assert config.output.h5ad_path == Path("concatenated.h5ad")


class TestConfigLoading:
    """Test configuration loading from TOML files."""

    def create_test_toml_file(self, config_data: Dict[str, Any]) -> Path:
        """Create a temporary TOML file with given configuration data."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        toml.dump(config_data, temp_file)
        temp_file.close()
        return Path(temp_file.name)

    def test_load_concat_config_valid(self) -> None:
        """Test loading valid configuration from TOML file."""
        config_data = {
            "workspace_path": "/test/data/test_exp",
            "codebooks": ["book1", "book2"],
            "seg_codebook": "seg",
            "intensity": {"channel": "test_channel", "required": False},
            "quality_control": {"min_counts": 50, "max_counts": 1500, "min_cells": 15, "min_area": 600.0},
            "spatial": {"max_columns": 3, "padding": 150.0},
            "output": {
                "h5ad_path": "test_output.h5ad",
                "baysor_export": False,
                "baysor_path": "test_baysor.csv",
            },
        }

        config_file = self.create_test_toml_file(config_data)
        try:
            config = load_concat_config(config_file)

            assert config.workspace_path == Path("/test/data/test_exp")
            assert config.codebooks == ["book1", "book2"]
            assert config.seg_codebook == "seg"
            assert config.intensity.channel == "test_channel"
            assert config.intensity.required is False
            assert config.quality_control.min_counts == 50

        finally:
            config_file.unlink()  # Clean up

    def test_load_concat_config_file_not_found(self) -> None:
        """Test error handling for missing configuration file."""
        non_existent_file = Path("/non/existent/config.toml")

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_concat_config(non_existent_file)

    def test_load_concat_config_invalid_toml(self) -> None:
        """Test error handling for invalid TOML syntax."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        temp_file.write("invalid toml syntax [[[")
        temp_file.close()

        try:
            with pytest.raises(ValueError, match="Failed to parse TOML configuration"):
                load_concat_config(Path(temp_file.name))
        finally:
            Path(temp_file.name).unlink()

    def test_load_concat_config_workspace_override(self) -> None:
        """Test workspace override functionality."""
        config_data = {
            "workspace_path": "/original/path/test_exp",
            "codebooks": ["book1"],
            "seg_codebook": "seg",
            "intensity": {"channel": "cfse", "required": True},
            "quality_control": {"min_counts": 40, "max_counts": 1200, "min_cells": 10, "min_area": 500.0},
            "spatial": {"max_columns": 2, "padding": 100.0},
            "output": {"h5ad_path": "test.h5ad", "baysor_export": True, "baysor_path": "baysor/spots.csv"},
        }

        config_file = self.create_test_toml_file(config_data)
        try:
            override_path = Path("/override/path/test_exp")
            config = load_concat_config(config_file, workspace_override=override_path)

            assert config.workspace_path == override_path

        finally:
            config_file.unlink()

    def test_load_concat_config_output_override(self) -> None:
        """Test output override functionality."""
        config_data = {
            "workspace_path": "/test/path/test_exp",
            "codebooks": ["book1"],
            "seg_codebook": "seg",
            "intensity": {"channel": "cfse", "required": True},
            "quality_control": {"min_counts": 40, "max_counts": 1200, "min_cells": 10, "min_area": 500.0},
            "spatial": {"max_columns": 2, "padding": 100.0},
            "output": {
                "h5ad_path": "original.h5ad",
                "baysor_export": True,
                "baysor_path": "baysor/spots.csv",
            },
        }

        config_file = self.create_test_toml_file(config_data)
        try:
            override_output = Path("override_output.h5ad")
            config = load_concat_config(config_file, output_override=override_output)

            assert config.output.h5ad_path == override_output

        finally:
            config_file.unlink()


class TestWorkspaceValidation:
    """Test workspace structure validation."""

    def test_validate_config_workspace_valid(self) -> None:
        """Test validation with valid workspace structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            deconved_dir = workspace_dir / "analysis" / "deconv"

            # Create directory structure
            workspace_dir.mkdir()
            deconved_dir.mkdir(parents=True)

            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            # Should not raise any exception
            validate_config_workspace(config)

    def test_validate_config_workspace_missing_workspace(self) -> None:
        """Test validation fails when workspace directory doesn't exist."""
        config = ConcatConfig(
            workspace_path=Path("/non/existent/workspace"),
            codebooks=["book1"],
            seg_codebook="seg",
            intensity=IntensityConfig(),
            quality_control=QualityControlConfig(),
            spatial=SpatialConfig(),
            output=OutputConfig(),
        )

        with pytest.raises(FileNotFoundError, match="Workspace directory not found"):
            validate_config_workspace(config)

    def test_validate_config_workspace_missing_deconved(self) -> None:
        """Test validation fails when deconved directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_dir = Path(temp_dir) / "test_experiment"
            workspace_dir.mkdir()
            # Note: not creating analysis/deconv directory

            config = ConcatConfig(
                workspace_path=workspace_dir,
                codebooks=["book1"],
                seg_codebook="seg",
                intensity=IntensityConfig(),
                quality_control=QualityControlConfig(),
                spatial=SpatialConfig(),
                output=OutputConfig(),
            )

            with pytest.raises(FileNotFoundError, match="Deconved directory not found"):
                validate_config_workspace(config)
