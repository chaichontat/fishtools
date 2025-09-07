"""
Simple configuration loader for fishtools preprocessing pipeline.

Provides utilities for loading configurations from TOML files with
backward compatibility support.
"""

from pathlib import Path

import toml
from loguru import logger
from pydantic import ValidationError

from .config import (
    BasicConfig,
    Config,
    DeconvolutionConfig,
    Fiducial,
    ImageProcessingConfig,
    ProcessingConfig,
    RegisterConfig,
    SpotAnalysisConfig,
    StitchingConfig,
    SystemConfig,
)


def load_config_from_toml(config_path: Path, data_path: str, **overrides) -> Config:
    """Load configuration from TOML file.

    Args:
        config_path: Path to TOML configuration file
        data_path: Data directory path
        **overrides: Additional configuration overrides

    Returns:
        Loaded and validated Config object
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        config_dict = toml.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise

    # Ensure required fields are present
    config_dict["dataPath"] = data_path

    # Apply any explicit overrides with proper nesting
    if overrides:
        # Map CLI parameters to their nested config locations
        cli_param_mapping = {
            "fwhm": ["registration", "fiducial", "fwhm"],
            "threshold": ["registration", "fiducial", "threshold"],
            "reference": ["registration", "reference"],
            "use_fft": ["registration", "fiducial", "use_fft"],
            "n_fids": ["registration", "fiducial", "n_fids"],
            "crop": ["registration", "crop"],
            "downsample": ["registration", "downsample"],
        }

        for key, value in overrides.items():
            if key in cli_param_mapping:
                # Apply to nested structure
                path = cli_param_mapping[key]
                current = config_dict

                # Navigate to parent of target key
                for part in path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the final value
                current[path[-1]] = value
                logger.debug(f"Applied CLI override: {'.'.join(path)} = {value}")
            else:
                # Unknown parameter, add to top level (for extensibility)
                config_dict[key] = value

    try:
        return Config(**config_dict)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        # Log specific field errors for better debugging
        for error in e.errors():
            field_path = " -> ".join(str(x) for x in error["loc"])
            logger.error(f"  Field '{field_path}': {error['msg']}")
        raise


def load_minimal_config(
    data_path: str = "/working/fishtools/data", reference: str = "4_12_20", **fiducial_overrides
) -> Config:
    """Create a minimal config for backward compatibility.

    This is designed to be a drop-in replacement for existing config creation
    patterns while still providing access to all the new configuration sections.
    """
    fiducial_config = Fiducial(**fiducial_overrides)

    register_config = RegisterConfig(
        fiducial=fiducial_config,
        reference=reference,
        chromatic_shifts={
            # Canonical target wavelengths are 650 and 750
            "650": str(Path(data_path) / "560to650.txt"),
            "750": str(Path(data_path) / "560to750.txt"),
        },
    )

    config_dict = {
        "dataPath": data_path,
        "registration": register_config,  # Pass the object directly, not dumped
    }

    return Config(**config_dict)


def generate_config_template(output_path: Path) -> None:
    """Generate a template TOML configuration file with all defaults."""
    # Create a complete configuration with all required fields
    config_dict = {
        "dataPath": "/path/to/your/data",
        "registration": {
            "reference": "4_12_20",
            "crop": 40,
            "downsample": 1,
            "slices": "slice(None)",  # Simplified slice notation for TOML
            "reduce_bit_depth": 0,
            "split_channels": False,
            "fiducial": Fiducial().model_dump(),
            "chromatic_shifts": {"647": "data/560to647.txt", "750": "data/560to750.txt"},
        },
        "system": SystemConfig().model_dump(),
        "processing": ProcessingConfig().model_dump(),
        "image_processing": ImageProcessingConfig().model_dump(),
        "deconvolution": DeconvolutionConfig().model_dump(),
        "basic_correction": BasicConfig().model_dump(),
        "stitching": StitchingConfig().model_dump(),
        "spot_analysis": SpotAnalysisConfig().model_dump(),
    }

    # Write TOML file with comments using standard toml.dump
    with open(output_path, "w") as f:
        f.write("# Fishtools Preprocessing Configuration\n")
        f.write(
            "# This file contains all configurable parameters for the fishtools preprocessing pipeline.\n\n"
        )

        # Use toml.dump for the entire config to ensure proper formatting
        toml.dump(config_dict, f)

    logger.info(f"Generated configuration template at {output_path}")


# Convenience functions for backward compatibility
def load_config(
    config_path: Path | None = None, data_path: str = "/working/fishtools/data", **overrides
) -> Config:
    """Load configuration from TOML file or create minimal config.

    Args:
        config_path: Path to TOML config file (optional)
        data_path: Data directory path
        **overrides: Additional configuration overrides

    Returns:
        Loaded and validated Config object
    """
    if config_path:
        return load_config_from_toml(config_path, data_path, **overrides)
    else:
        return load_minimal_config(data_path, **overrides)
