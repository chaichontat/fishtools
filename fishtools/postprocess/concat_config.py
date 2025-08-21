"""
Configuration management for multi-ROI FISH data concatenation.

This module provides Pydantic models for configuration validation and TOML loading
for the concat pipeline, leveraging the existing Workspace class for path management.
"""

from pathlib import Path
from typing import Dict, List, Optional

import toml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fishtools.utils.io import Workspace


class IntensityConfig(BaseModel):
    """Configuration for intensity channel processing."""

    channel: str = Field(default="cfse", description="Intensity channel identifier")
    required: bool = Field(default=True, description="Whether intensity data is required for the analysis")


class QualityControlConfig(BaseModel):
    """Configuration for quality control filtering parameters."""

    min_counts: int = Field(default=40, description="Minimum counts per cell for filtering")
    max_counts: int = Field(default=1200, description="Maximum counts per cell for filtering")
    min_cells: int = Field(default=10, description="Minimum cells expressing a gene for gene filtering")
    min_area: float = Field(default=500.0, description="Minimum cell area for filtering")

    @field_validator("min_counts", "max_counts", "min_cells")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Count and cell parameters must be positive")
        return v

    @field_validator("min_area")
    @classmethod
    def validate_positive_area(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Minimum area must be positive")
        return v


class SpatialConfig(BaseModel):
    """Configuration for spatial processing and ROI arrangement."""

    max_columns: int = Field(default=2, description="Maximum number of columns in ROI grid arrangement")
    padding: float = Field(default=100.0, description="Padding between ROIs in spatial arrangement")

    @field_validator("max_columns")
    @classmethod
    def validate_positive_columns(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Max columns must be positive")
        return v

    @field_validator("padding")
    @classmethod
    def validate_non_negative_padding(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Padding cannot be negative")
        return v


class OutputConfig(BaseModel):
    """Configuration for output file paths and formats."""

    h5ad_path: Path = Field(
        default=Path("concatenated.h5ad"), description="Output path for AnnData H5AD file"
    )
    baysor_export: bool = Field(default=True, description="Whether to export Baysor-compatible CSV file")
    baysor_path: Path = Field(
        default=Path("baysor/spots.csv"), description="Output path for Baysor spots CSV file"
    )


class ConcatConfig(BaseModel):
    """Main configuration class for the concatenation pipeline."""

    workspace_path: Path = Field(description="Path to the experiment workspace directory")
    codebooks: List[str] = Field(description="List of codebook identifiers for analysis")
    seg_codebook: str = Field(description="Segmentation codebook identifier")
    intensity: IntensityConfig
    quality_control: QualityControlConfig
    spatial: SpatialConfig
    output: OutputConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("codebooks")
    @classmethod
    def validate_codebooks_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Codebooks list cannot be empty")
        return v

    @property
    def workspace(self) -> Workspace:
        """Get the Workspace instance for this configuration."""
        return Workspace(self.workspace_path)

    @property
    def rois(self) -> List[str]:
        """Get the list of ROIs from the workspace."""
        return self.workspace.rois


def load_concat_config(
    config_path: Path, workspace_override: Optional[Path] = None, output_override: Optional[Path] = None
) -> ConcatConfig:
    """
    Load and validate concatenation configuration from TOML file.

    Args:
        config_path: Path to the TOML configuration file
        workspace_override: Optional override for workspace path
        output_override: Optional override for output H5AD path

    Returns:
        Validated ConcatConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration validation fails
        toml.TomlDecodeError: If TOML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load TOML configuration
    try:
        config_data = toml.load(config_path)
    except Exception as e:
        raise ValueError(f"Failed to parse TOML configuration: {e}") from e

    # Apply overrides
    if workspace_override:
        config_data["workspace_path"] = str(workspace_override)

    if output_override:
        if "output" not in config_data:
            config_data["output"] = {}
        config_data["output"]["h5ad_path"] = str(output_override)

    # Validate and create configuration
    try:
        return ConcatConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def validate_config_workspace(config: ConcatConfig) -> None:
    """
    Validate that the configured workspace structure exists.

    Args:
        config: Configuration to validate

    Raises:
        FileNotFoundError: If required directories don't exist
    """
    workspace = config.workspace
    if not workspace.path.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace.path}")

    if not workspace.deconved.exists():
        raise FileNotFoundError(f"Deconved directory not found: {workspace.deconved}")
