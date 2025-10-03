"""
Processing context for FISH image registration pipeline.

Provides dependency injection pattern for configuration, workspace management,
and scientific parameter caching with validation at initialization.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fishtools.preprocess.config import Config
from fishtools.io.workspace import Workspace


@dataclass
class ProcessingContext:
    """Centralized context for FISH image registration processing.

    Provides dependency injection pattern with configuration validation,
    workspace management, and performance-optimized parameter caching.
    Scientific reproducibility is ensured through explicit versioning
    and comprehensive validation at initialization.

    Args:
        config: Validated configuration object
        workspace: Workspace for path resolution
        validate_on_init: Whether to validate configuration and files at initialization

    Example:
        >>> config = create_configuration(config_file=config_path)
        >>> ctx = ProcessingContext.create(workspace_path, config)
        >>> sigma = ctx.log_sigma  # Cached parameter access
        >>> As, ats = ctx.chromatic_matrices["650"]  # Pre-loaded matrices
    """

    config: Config
    workspace: Workspace
    validate_on_init: bool = True

    # Cached pre-loaded data (populated during initialization)
    _chromatic_matrices: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=dict, init=False
    )
    _parameter_cache: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize context with validation and pre-loading."""
        if self.validate_on_init:
            self._validate_configuration()
            self._load_chromatic_matrices()
            logger.info(f"ProcessingContext initialized for workspace: {self.workspace.path}")
            logger.info(f"Configuration source: {getattr(self.config, '_source', 'minimal config')}")

    @classmethod
    def create(cls, workspace_path: Path, config: Config, validate: bool = True) -> "ProcessingContext":
        """Create processing context with workspace and configuration.

        Args:
            workspace_path: Path to FISH experiment workspace
            config: Validated configuration object
            validate: Whether to validate configuration and files

        Returns:
            Initialized ProcessingContext

        Raises:
            FileNotFoundError: If required chromatic matrices are missing
            ValueError: If configuration validation fails
        """
        workspace = Workspace(workspace_path)
        return cls(config=config, workspace=workspace, validate_on_init=validate)

    def _validate_configuration(self) -> None:
        """Validate configuration and required files exist.

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If configuration is invalid
        """
        logger.debug("Validating processing configuration...")

        # Validate data path exists
        data_path = self.resolve_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        # Validate chromatic shift files exist
        for channel, path_str in self.config.registration.chromatic_shifts.items():
            path = self._resolve_chromatic_path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Chromatic correction file not found for channel {channel}: {path}")

        # Validate scientific parameters are within reasonable bounds
        img_config = self.config.image_processing
        if not (0.1 <= img_config.log_sigma <= 10.0):
            raise ValueError(f"log_sigma {img_config.log_sigma} outside valid range [0.1, 10.0]")

        if len(img_config.percentiles) != 2 or img_config.percentiles[0] >= img_config.percentiles[1]:
            raise ValueError(f"Invalid percentile range: {img_config.percentiles}")

        logger.debug("Configuration validation passed")

    def _resolve_chromatic_path(self, path_str: str) -> Path:
        """Resolve chromatic correction file path.

        Args:
            path_str: Path string from configuration

        Returns:
            Resolved absolute path
        """
        path = Path(path_str)
        if path.is_absolute():
            return path

        # Try relative to data path first
        data_relative = self.resolve_data_path() / path
        if data_relative.exists():
            return data_relative

        # Fall back to workspace relative
        return self.workspace.path / path

    def _load_chromatic_matrices(self) -> None:
        """Pre-load chromatic correction matrices for performance.

        Loads transformation matrices As and translation vectors ats
        for each configured chromatic channel. Matrices are cached
        for repeated access during processing.

        Raises:
            ValueError: If matrix format is invalid
        """
        logger.debug("Loading chromatic correction matrices...")

        for channel, path_str in self.config.registration.chromatic_shifts.items():
            path = self._resolve_chromatic_path(path_str)

            try:
                # Load chromatic correction data (expected format: 6 values)
                a_ = np.loadtxt(path)
                if a_.shape != (6,):
                    raise ValueError(f"Invalid chromatic matrix shape {a_.shape}, expected (6,)")

                # Create transformation matrix A
                A = np.zeros((3, 3), dtype=np.float64)
                A[:2, :2] = a_[:4].reshape(2, 2)  # 2x2 transformation
                A[2] = [0, 0, 1]
                A[:, 2] = [0, 0, 1]

                # Create translation vector t
                t = np.zeros(3, dtype=np.float64)
                t[:2] = a_[-2:]  # translation components
                t[2] = 0

                self._chromatic_matrices[channel] = (A, t)
                logger.debug(f"Loaded chromatic matrix for channel {channel}")

            except Exception as e:
                raise ValueError(f"Failed to load chromatic matrix from {path}: {e}") from e

    def resolve_data_path(self) -> Path:
        """Resolve data directory path with workspace awareness.

        Returns:
            Resolved data directory path
        """
        data_path = Path(self.config.dataPath)
        if data_path.is_absolute():
            return data_path

        # Resolve relative to workspace
        return self.workspace.path / data_path

    # Direct config section access (no excessive method proliferation)

    @property
    def img_config(self):
        """Direct access to image processing configuration."""
        return self.config.image_processing

    @property
    def reg_config(self):
        """Direct access to registration configuration."""
        return self.config.registration

    @property
    def system_config(self):
        """Direct access to system configuration."""
        return self.config.system

    @property
    def stitching_config(self):
        """Direct access to stitching configuration."""
        return self.config.stitching

    @property
    def processing_config(self):
        """Direct access to processing configuration."""
        return self.config.processing

    @property
    def chromatic_matrices(self) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Pre-loaded chromatic correction matrices.

        Returns:
            Dictionary mapping channel names to (transformation_matrix, translation_vector) tuples
        """
        return self._chromatic_matrices

    def get_chromatic_transform(self, channel: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get chromatic transformation for specific channel.

        Args:
            channel: Channel identifier (e.g., "650", "750")

        Returns:
            Tuple of (transformation_matrix, translation_vector)

        Raises:
            KeyError: If channel not configured
        """
        if channel not in self._chromatic_matrices:
            raise KeyError(f"Chromatic correction not configured for channel: {channel}")
        return self._chromatic_matrices[channel]

    def log_processing_parameters(self) -> None:
        """Log all processing parameters for scientific reproducibility."""
        logger.info("=== FISH Processing Parameters ===")
        logger.info(f"Workspace: {self.workspace.path}")
        logger.info(f"Data path: {self.resolve_data_path()}")
        logger.info(f"LoG sigma: {self.img_config.log_sigma}")
        logger.info(f"Percentiles: {self.img_config.percentiles}")
        logger.info(f"Spillover corrections: {self.img_config.spillover_corrections}")
        logger.info(f"Max iterations: {self.img_config.max_iterations}")
        logger.info(f"Residual threshold: {self.img_config.residual_threshold}")
        logger.info(f"Chromatic channels: {list(self.chromatic_matrices.keys())}")
        logger.info("=====================================")


# Backward compatibility helper
def create_legacy_processing_context(
    workspace_path: Path,
    config_file: Path | None = None,
    data_path: str | None = None,
    **legacy_overrides: Any,
) -> ProcessingContext:
    """Create processing context with legacy parameter compatibility.

    Provides backward compatibility for existing code while issuing
    deprecation warnings for legacy parameter usage.

    Args:
        workspace_path: Path to FISH experiment workspace
        config_file: Optional configuration file path
        data_path: Optional data directory path (deprecated)
        **legacy_overrides: Legacy parameter overrides (deprecated)

    Returns:
        ProcessingContext with legacy parameter mapping
    """
    # Issue deprecation warnings for legacy usage
    if data_path is not None:
        warnings.warn(
            "data_path parameter is deprecated. Use workspace-relative paths in config file.",
            DeprecationWarning,
            stacklevel=2,
        )

    if legacy_overrides:
        warnings.warn(
            f"Legacy parameter overrides {list(legacy_overrides.keys())} are deprecated. "
            "Use configuration file or CLI parameters instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Create configuration (this will handle backward compatibility)
    from fishtools.preprocess.cli_register_migrated import create_configuration

    config = create_configuration(
        config_file=config_file, data_path=data_path or "/working/fishtools/data", **legacy_overrides
    )

    return ProcessingContext.create(workspace_path, config)
