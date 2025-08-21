"""
Configuration management for intensity extraction from segmentation masks.

This module provides Pydantic-based configuration classes for managing
intensity extraction parameters, including Zarr file paths, channel specifications,
and processing options.
"""

import warnings
from pathlib import Path
from typing import Any

import psutil
import toml
import zarr
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from fishtools.utils.io import Workspace


class IntensityExtractionConfig(BaseModel):
    """
    Configuration for intensity extraction from segmentation masks.

    This class manages all parameters needed for extracting region properties
    and intensity measurements from segmentation masks using Zarr-format data.
    """

    workspace_path: Path = Field(description="Path to the experiment workspace directory")

    roi: str = Field(description="ROI identifier for processing (e.g., 'roi1', 'roi2')")

    seg_codebook: str = Field(
        default="atp", description="Segmentation codebook identifier used for directory structure"
    )

    segmentation_name: str = Field(
        default="output_segmentation.zarr",
        description="Name of the segmentation Zarr store within the ROI directory",
    )

    intensity_name: str = Field(
        default="input_image.zarr",
        description="Name of the intensity image Zarr store within the ROI directory",
    )

    channel: str = Field(description="Target intensity channel name for extraction")

    max_workers: int = Field(
        default=4, ge=1, le=32, description="Number of parallel processes for Z-slice processing"
    )

    overwrite: bool = Field(default=False, description="Whether to overwrite existing output files")

    @field_validator("workspace_path")
    @classmethod
    def validate_workspace_path(cls, v: Path) -> Path:
        """Validate that workspace path exists and is a directory."""
        if not v.exists():
            raise ValueError(f"Workspace directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Workspace path is not a directory: {v}")
        return v

    @field_validator("roi")
    @classmethod
    def validate_roi_format(cls, v: str) -> str:
        """Validate ROI identifier format."""
        if not v.strip():
            raise ValueError("ROI identifier cannot be empty")
        # Basic validation - could be enhanced based on naming conventions
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"ROI identifier contains invalid characters: {v}")
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel_name(cls, v: str) -> str:
        """Validate channel name format."""
        if not v.strip():
            raise ValueError("Channel name cannot be empty")
        return v.strip()

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers_with_system_info(cls, v: int) -> int:
        """Validate max_workers considering system capabilities."""
        cpu_count = psutil.cpu_count()

        if v > cpu_count * 2:
            warnings.warn(
                f"max_workers ({v}) exceeds 2x CPU count ({cpu_count}), may cause performance issues"
            )

        return v

    @property
    def workspace(self) -> Workspace:
        """Get the Workspace instance for this configuration."""
        return Workspace(self.workspace_path)

    @property
    def roi_directory(self) -> Path:
        """Get the ROI-specific directory path."""
        return self.workspace.path / f"stitch--{self.roi}+{self.seg_codebook}"

    @property
    def segmentation_zarr_path(self) -> Path:
        """Get the full path to the segmentation Zarr store."""
        return self.roi_directory / self.segmentation_name

    @property
    def intensity_zarr_path(self) -> Path:
        """Get the full path to the intensity Zarr store."""
        return self.roi_directory / self.intensity_name

    @property
    def output_directory(self) -> Path:
        """Get the output directory for intensity parquet files."""
        return self.roi_directory / f"intensity_{self.channel}"

    def validate_zarr_files(self) -> None:
        """
        Validate that required Zarr files exist and are accessible.

        Raises:
            FileNotFoundError: If required Zarr files are missing
            ValueError: If Zarr files are not accessible or properly formatted
        """
        if not self.roi_directory.exists():
            raise FileNotFoundError(f"ROI directory not found: {self.roi_directory}")

        if not self.segmentation_zarr_path.exists():
            raise FileNotFoundError(f"Segmentation Zarr not found: {self.segmentation_zarr_path}")

        if not self.intensity_zarr_path.exists():
            raise FileNotFoundError(f"Intensity Zarr not found: {self.intensity_zarr_path}")

    def validate_zarr_stores(self) -> dict[str, Any]:
        """
        Validate Zarr stores are accessible and contain expected data.

        Returns:
            Dictionary containing Zarr store information and validation results

        Raises:
            ValueError: If Zarr stores are invalid or incompatible
        """
        try:
            # Open Zarr stores
            seg_zarr = zarr.open(self.segmentation_zarr_path, mode="r")
            intensity_zarr = zarr.open(self.intensity_zarr_path, mode="r")

            # Validate channel exists
            available_channels = []
            if hasattr(intensity_zarr, "attrs") and "key" in intensity_zarr.attrs:
                available_channels = intensity_zarr.attrs["key"]
            else:
                # Fallback: check array dimensions or subdirectories
                available_channels = (
                    list(range(intensity_zarr.shape[-1])) if len(intensity_zarr.shape) > 3 else ["unknown"]
                )

            if isinstance(available_channels, list) and self.channel not in available_channels:
                raise ValueError(
                    f"Channel '{self.channel}' not found in intensity Zarr. "
                    f"Available channels: {available_channels}"
                )

            # Get array information
            seg_shape = seg_zarr.shape
            intensity_shape = intensity_zarr.shape

            # Validate compatible shapes (intensity may have extra channel dimension)
            if len(intensity_shape) == 4 and len(seg_shape) == 3:
                # Intensity has (Z, Y, X, C) format, segmentation has (Z, Y, X)
                if seg_shape != intensity_shape[:3]:
                    raise ValueError(
                        f"Shape mismatch: segmentation={seg_shape}, intensity spatial={intensity_shape[:3]}"
                    )
            elif seg_shape != intensity_shape:
                raise ValueError(f"Shape mismatch: segmentation={seg_shape}, intensity={intensity_shape}")

            # Return validation information
            return {
                "segmentation_shape": seg_shape,
                "intensity_shape": intensity_shape,
                "available_channels": available_channels,
                "segmentation_chunks": getattr(seg_zarr, "chunks", None),
                "intensity_chunks": getattr(intensity_zarr, "chunks", None),
                "segmentation_dtype": str(seg_zarr.dtype),
                "intensity_dtype": str(intensity_zarr.dtype),
            }

        except (OSError, Exception) as e:
            if "zarr" in str(e).lower():
                raise ValueError(f"Invalid Zarr store format: {e}")
            raise ValueError(f"Error validating Zarr stores: {e}")

    def estimate_memory_requirements(self) -> dict[str, float]:
        """
        Estimate memory requirements for processing.

        Returns:
            Dictionary with memory estimates in GB
        """
        try:
            zarr_info = self.validate_zarr_stores()

            # Estimate memory per slice (segmentation + intensity)
            seg_shape = zarr_info["segmentation_shape"]
            intensity_shape = zarr_info["intensity_shape"]

            # Bytes per pixel (assume uint16 for segmentation, float32 for intensity)
            seg_bytes_per_slice = seg_shape[1] * seg_shape[2] * 2  # uint16
            intensity_bytes_per_slice = intensity_shape[1] * intensity_shape[2] * 4  # float32

            # Total memory per slice (including regionprops overhead)
            bytes_per_slice = (
                seg_bytes_per_slice + intensity_bytes_per_slice
            ) * 3  # 3x for processing overhead

            # Estimate based on worker count
            parallel_memory_gb = (bytes_per_slice * self.max_workers) / (1024**3)
            peak_memory_gb = parallel_memory_gb * 2.0  # Additional overhead for safety

            return {
                "bytes_per_slice": bytes_per_slice,
                "parallel_memory_gb": parallel_memory_gb,
                "peak_memory_gb": peak_memory_gb,
                "recommended_workers": min(
                    self.max_workers, max(1, int(psutil.virtual_memory().available / (1024**3) / 2))
                ),
            }

        except Exception as e:
            logger.warning(f"Could not estimate memory requirements: {e}")
            return {
                "bytes_per_slice": 0,
                "parallel_memory_gb": 2.0,  # Conservative estimate
                "peak_memory_gb": 4.0,
                "recommended_workers": min(self.max_workers, 2),
            }

    def validate_performance_requirements(self) -> dict[str, Any]:
        """
        Validate system can handle processing requirements.

        Returns:
            Dictionary with performance validation results

        Raises:
            ValueError: If system resources are insufficient
        """
        memory_info = self.estimate_memory_requirements()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        performance_info = {
            "estimated_peak_memory_gb": memory_info["peak_memory_gb"],
            "available_memory_gb": available_memory_gb,
            "recommended_workers": memory_info["recommended_workers"],
            "memory_utilization_pct": (memory_info["peak_memory_gb"] / available_memory_gb) * 100,
        }

        # Check memory requirements
        if memory_info["peak_memory_gb"] > available_memory_gb * 0.8:
            raise ValueError(
                f"Insufficient memory: estimated peak usage {memory_info['peak_memory_gb']:.1f}GB "
                f"exceeds 80% of available {available_memory_gb:.1f}GB"
            )

        # Warn about high memory usage
        if memory_info["peak_memory_gb"] > available_memory_gb * 0.6:
            warnings.warn(
                f"High memory usage expected: {memory_info['peak_memory_gb']:.1f}GB "
                f"of {available_memory_gb:.1f}GB available"
            )

        # Recommend worker adjustment
        if self.max_workers > memory_info["recommended_workers"]:
            warnings.warn(
                f"Consider reducing max_workers from {self.max_workers} to "
                f"{memory_info['recommended_workers']} based on available memory"
            )

        return performance_info


def load_intensity_config(
    config_path: Path,
    workspace_override: Path | None = None,
    roi_override: str | None = None,
    channel_override: str | None = None,
) -> IntensityExtractionConfig:
    """
    Load intensity extraction configuration from TOML file with optional overrides.

    Args:
        config_path: Path to TOML configuration file
        workspace_override: Override workspace path from config
        roi_override: Override ROI identifier from config
        channel_override: Override channel name from config

    Returns:
        Validated IntensityExtractionConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
        toml.TomlDecodeError: If TOML syntax is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    try:
        config_data = toml.load(config_path)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML syntax in configuration file: {e}")

    # Apply overrides
    if workspace_override is not None:
        config_data["workspace_path"] = str(workspace_override)

    if roi_override is not None:
        config_data["roi"] = roi_override

    if channel_override is not None:
        config_data["channel"] = channel_override

    # Convert string paths to Path objects
    if "workspace_path" in config_data:
        config_data["workspace_path"] = Path(config_data["workspace_path"])

    return IntensityExtractionConfig(**config_data)


def validate_intensity_config(config: IntensityExtractionConfig) -> dict[str, Any]:
    """
    Validate intensity extraction configuration and required file structure.

    Args:
        config: Configuration instance to validate

    Returns:
        Dictionary containing validation results and system information

    Raises:
        FileNotFoundError: If required directories or files are missing
        ValueError: If configuration is inconsistent or invalid
    """
    # Validate basic file structure
    config.validate_zarr_files()

    # Validate Zarr stores and get information
    zarr_info = config.validate_zarr_stores()

    # Validate performance requirements
    performance_info = config.validate_performance_requirements()

    # Validate output directory can be created
    try:
        config.output_directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise ValueError(f"Cannot create output directory {config.output_directory}: {e}")
    except OSError as e:
        raise ValueError(f"Invalid output directory path {config.output_directory}: {e}")

    # Return comprehensive validation information
    return {
        "zarr_info": zarr_info,
        "performance_info": performance_info,
        "output_directory": str(config.output_directory),
        "validation_status": "passed",
    }
