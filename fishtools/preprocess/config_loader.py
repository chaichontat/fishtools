import json
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from .config import Config, Fiducial, RegisterConfig


def load_config_from_json(config_path: Path, data_path: str, **overrides) -> Config:
    """Load configuration from JSON file (base format).

    Mirrors load_config_from_toml behavior, including CLI-style overrides for nested keys.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        config_dict = json.loads(config_path.read_text())
        logger.debug(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load JSON config {config_path}: {e}")
        raise ValueError(f"Failed to load config file {config_path}: {e}") from e

    config_dict["dataPath"] = data_path

    if overrides:
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
                path = cli_param_mapping[key]
                current = config_dict
                for part in path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[path[-1]] = value
                logger.debug(f"Applied CLI override: {'.'.join(path)} = {value}")
            else:
                config_dict[key] = value

    try:
        return Config(**config_dict)
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
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
    # Keep minimal shape; rely on default_register_config() from Config for chromatic shifts
    return Config(dataPath=data_path)


def generate_config_template(output_path: Path) -> None:
    """Generate a JSON configuration file with minimal required defaults."""
    config_dict = {
        "dataPath": "/path/to/your/data",
        "registration": {
            "reference": "4_12_20",
            "crop": 40,
            "downsample": 1,
            "slices": "slice(None)",
            "reduce_bit_depth": 0,
            "split_channels": False,
            "fiducial": Fiducial().model_dump(),
            "chromatic_shifts": {"647": "data/560to647.txt", "750": "data/560to750.txt"},
        },
    }
    output_path.write_text(json.dumps(config_dict, indent=2))
    logger.info(f"Generated JSON configuration template at {output_path}")


# Convenience functions for backward compatibility
def load_config(
    config_path: Path | None = None, data_path: str = "/working/fishtools/data", **overrides
) -> Config:
    """Load configuration from JSON file or create minimal config.

    Args:
        config_path: Path to TOML config file (optional)
        data_path: Data directory path
        **overrides: Additional configuration overrides

    Returns:
        Loaded and validated Config object
    """
    if config_path:
        return load_config_from_json(config_path, data_path, **overrides)
    return load_minimal_config(data_path, **overrides)


# ---- Align/Spot JSON config loader (for align_prod.py) ----


def _resolve(base: Path, maybe: str | None) -> str | None:
    if maybe is None:
        return None
    p = Path(maybe)
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


# Note: align_prod consumes the main project Config. No separate loader needed.
