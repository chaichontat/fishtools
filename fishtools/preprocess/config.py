from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, PlainSerializer, field_validator

DATA = Path("/working/fishtools/data")


class DeconvolutionOutputMode(str, Enum):
    """Available output encodings for deconvolution pipelines."""

    U16 = "u16"
    F32 = "float32"


class NumpyEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class FiducialDetailedConfig(BaseModel):
    """Comprehensive fiducial spot detection and alignment processing parameters."""

    # Reference spot detection limits
    max_attempts: int = Field(default=6, description="Maximum attempts to find spots on reference image")
    max_spots: int = Field(default=1500, description="Maximum number of spots allowed on reference image")
    min_spots: int = Field(default=6, description="Minimum number of spots required for alignment")

    # Threshold adjustment parameters
    threshold_step: float = Field(
        default=0.5,
        description="Step size for threshold adjustment during spot finding",
    )
    min_threshold_sigma: float = Field(
        default=2.0, description="Minimum threshold sigma before adjusting FWHM"
    )
    min_fwhm: float = Field(default=4.0, description="Minimum FWHM before reducing in exception handling")

    # Image preprocessing
    percentile_clip: float = Field(
        default=50.0,
        description="Percentile value for image clipping during preprocessing",
    )
    background_box_size: tuple[int, int] = Field(
        default=(50, 50), description="Box size for background estimation"
    )
    background_sigma_clip: float = Field(
        default=2.0, description="Sigma clipping value for background estimation"
    )

    # Drift calculation parameters
    max_drift_attempts: int = Field(default=30, description="Maximum attempts for drift calculation")
    max_drift_threshold: float = Field(default=40.0, description="Maximum allowed drift in pixels")
    warning_spots_threshold: int = Field(default=1000, description="Warn if more than this many spots found")
    min_spots_for_mode: int = Field(default=100, description="Minimum spots required to use mode calculation")

    # Alignment quality thresholds
    bin_size: float = Field(default=0.5, description="Bin size for mode calculation in drift estimation")
    correlation_percentile: float = Field(
        default=99.0, description="Percentile for intensity normalization in plotting"
    )


class Fiducial(BaseModel):
    use_fft: bool = Field(
        default=False,
        description="Use FFT to find fiducial spots. Overrides everything else.",
    )
    fwhm: float = Field(
        default=4.0,
        gt=0.1,  # Must be positive
        description="Full width at half maximum for fiducial spot detection. The higher this is, the more spots will be detected.",
    )
    threshold: float = Field(
        default=3.0,
        gt=0.0,  # Must be positive
        description="Threshold for fiducial spot detection in standard deviation above the median.",
    )
    priors: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Shifts to apply before alignment. Name must match round name.",
    )
    overrides: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Overrides for fiducial spot detection. Name must match round name.",
    )
    n_fids: int = Field(default=2, ge=1, description="Number of fiducial frames in each image.")
    detailed: FiducialDetailedConfig = Field(
        default_factory=FiducialDetailedConfig,
        description="Detailed fiducial processing parameters",
    )


class RegisterConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    fiducial: Fiducial
    downsample: int = Field(default=1, description="Downsample factor")
    reduce_bit_depth: int = Field(
        default=0,
        description="Reduce bit depth by n bits. 0 to disable. This is to assist in compression of output intended for visualization.",
    )
    crop: int = Field(
        default=40,
        description="Pixels to crop from each edge. This is to account for translation during alignment.",
    )
    slices: Annotated[
        list[tuple[int | None, int | None]] | slice | str,
        PlainSerializer(lambda x: str(x)),
    ] = Field(default=slice(None), description="Slice range to use for registration")
    split_channels: bool = False
    chromatic_shifts: dict[str, Annotated[str, "path for 560to{channel}.txt"]]
    reference: str = Field(default="4_12_20", description="Reference round to align others to.")
    # Moved from HardwareConfig - threads used specifically for registration
    threads: int = Field(default=15, description="Number of threads for registration operations")
    discards: dict[str, list[str]] | None = Field(
        None,
        description="In case of duplicated key(s), discard the ones from the file in value.",
    )

    @field_validator("slices", mode="before")
    @classmethod
    def parse_slices(cls, v: Any) -> slice | list[tuple[int | None, int | None]]:
        """Parse slice strings into slice objects."""
        if isinstance(v, str):
            if v in ("slice(None)", "slice(None, None, None)", ""):
                return slice(None)
            else:
                # Provide helpful error for unrecognized formats
                raise ValueError(
                    f"Unsupported slice format: '{v}'. Use 'slice(None)' for full range "
                    f"or provide a slice object directly in Python code."
                )
        return v

    # Canonicalize chromatic shift keys to wavelengths {560, 650, 750}
    @field_validator("chromatic_shifts", mode="before")
    @classmethod
    def canonicalize_chromatic(cls, v: dict[str, str] | None) -> dict[str, str]:
        # Fallback to sensible defaults when empty or missing
        if not v:
            return {"650": "data/560to650.txt", "750": "data/560to750.txt"}
        if not isinstance(v, dict):  # type: ignore
            return v  # type: ignore
        synonym_map = {"561": "560", "640": "650", "647": "650"}
        allowed = {"650", "750"}  # chromatic shifts are defined relative to 560
        out: dict[str, str] = {}
        for k, path in v.items():
            ks = str(k)
            kc = synonym_map.get(ks, ks)
            if kc != ks:
                logger.warning(
                    f"Normalized chromatic key '{ks}' -> '{kc}' (canonical wavelengths: 560, 650, 750)"
                )
            out[kc] = path
        # Warn on unexpected keys; keep them for backward-compat
        for k in list(out.keys()):
            if k not in allowed:
                logger.warning(
                    "Chromatic shift provided for '%s'. Expected only target channels {'650','750'} relative to 560.",
                    k,
                )
        return out


def default_register_config() -> "RegisterConfig":
    """Factory for RegisterConfig with built-in chromatic shift defaults.

    Uses canonical target wavelengths (650, 750) relative to the 560 channel.
    Paths are relative and can be resolved by the caller as needed.
    """
    return RegisterConfig(
        chromatic_shifts={
            "650": (DATA / "560to650.txt").resolve().as_posix(),
            "750": (DATA / "560to750.txt").resolve().as_posix(),
        },
        fiducial=Fiducial(),
        downsample=1,
        crop=40,
        slices=slice(None),
        reduce_bit_depth=0,
    )


class ProcessingConfig(BaseModel):
    """Generic processing configuration for cross-cutting concerns."""

    queue_max_sizes: list[int] = Field(
        default=[3, 1], description="Maximum queue sizes for write and image processing"
    )


class BasicConfig(BaseModel):
    """BaSiC correction configuration."""

    max_files: dict[str, int] = Field(
        default={"tiff": 500, "registered": 800, "default": 1000},
        description="Maximum number of files to process for different file types",
    )
    min_file_requirements: dict[str, int] = Field(
        default={"critical": 100, "warning": 500},
        description="Minimum file counts for reliable BaSiC fitting",
    )
    z_slices: list[float] = Field(default=[0.5], description="Z-slice positions to extract (0-1 range)")
    random_sample_limit: int = Field(default=1000, description="Maximum number of files to randomly sample")
    # Moved from HardwareConfig - threads used specifically for BaSiC correction
    threads: int = Field(default=3, description="Number of threads for BaSiC correction operations")


class DeconvolutionConfig(BaseModel):
    """3D deconvolution processing configuration."""

    projector_step: int = Field(default=6, description="PSF projector step size parameter")
    wiener_params: dict[str, float] = Field(
        default={"alpha": 0.02, "beta": 0.02, "n": 10},
        description="Wiener-Butterworth deconvolution parameters",
    )
    percentiles: dict[str, float] = Field(
        default={"min": 0.1, "max": 99.999},
        description="Percentile values for intensity scaling",
    )
    protein_percentiles: dict[str, float] = Field(
        default={"min": 50, "max": 50},
        description="Percentile values specifically for protein channels",
    )
    max_rna_bit: int = Field(default=36, description="Maximum bit number considered RNA (vs protein)")
    # Moved from HardwareConfig - threads used specifically for deconvolution
    threads: int = Field(default=6, description="Number of threads for deconvolution operations")
    # BaSiC correction nested under deconvolution
    basic: BasicConfig = Field(default_factory=BasicConfig, description="BaSiC correction configuration")
    output_mode: DeconvolutionOutputMode = Field(
        default=DeconvolutionOutputMode.U16,
        description="Default output mode for CLI-driven deconvolution (u16 or float32).",
    )


class StitchingConfig(BaseModel):
    """Image stitching and fusion configuration."""

    default_downsample: int = Field(default=2, description="Default downsampling factor for stitching")
    tile_splits: int = Field(default=1, description="Number of splits for very large tile arrays")
    fusion_thresholds: dict[str, float] = Field(
        default={"regression": 0.4, "displacement_max": 1.5, "displacement_abs": 2.5},
        description="ImageJ stitching threshold parameters",
    )
    compression_levels: dict[str, float] = Field(
        default={"low": 0.7, "medium": 0.75, "high": 0.8},
        description="TIFF compression levels for different use cases",
    )
    pixel_scaling: dict[str, float] = Field(
        default={"size": 1024, "reference": 200, "scale": 0.108},
        description="Pixel size and scaling parameters",
    )
    # Moved from HardwareConfig - ImageJ-specific settings used in stitching
    max_memory_mb: int = Field(
        default=102400,
        description="Maximum memory allocation in MB for ImageJ stitching operations",
    )
    parallel_threads: int = Field(
        default=32,
        description="Number of parallel threads for ImageJ stitching operations",
    )
    # Moved from HardwareConfig - threads used specifically for stitching
    threads: int = Field(default=8, description="Number of threads for stitching operations")


class SpotThresholdParams(BaseModel):
    """Flattened, CLI-friendly parameters for spotlook analysis.

    Derived from `SpotAnalysisConfig` and `ImageProcessingConfig` so CLIs don't
    have to know the nested shapes. Keep defaults in sync with SpotAnalysisConfig.
    """

    # Core filtering parameters
    area_min: float = Field(default=10.0, gt=0, description="Minimum spot area in pixels")
    area_max: float = Field(default=200.0, gt=0, description="Maximum spot area in pixels")
    norm_threshold: float = Field(
        default=0.007, gt=0, description="Norm threshold BEFORE contours calculation"
    )
    min_norm: float = Field(default=0, ge=0, description="Norm threshold AFTER contours calculation")
    distance_threshold: float = Field(default=0.3, gt=0, description="Distance threshold")

    # Density analysis parameters
    density_grid_size: int = Field(default=50, gt=0, description="Grid size for density map")
    density_smooth_sigma: float = Field(default=2.0, gt=0, description="Gaussian sigma for density")
    min_spots_per_bin: int = Field(default=50, gt=0, description="Minimum raw spots required per grid bin")
    density_metric: Literal["count", "proportion"] = Field(
        default="proportion",
        description="Density map value: raw blank counts or blank proportion",
    )
    contour_mode: Literal["linear", "log", "sqrt"] = Field(
        default="linear", description="Spacing of density contour levels"
    )
    contour_levels: int = Field(default=50, gt=1, description="Number of contour levels")

    # Reproducibility
    seed: int = Field(default=0, ge=0, description="Random seed")

    # Visualization
    subsample: int = Field(default=200000, gt=0, description="Max spots to display in plots")
    scale_bar_um: float = Field(default=1000.0, gt=0, description="Scale bar size (micrometers)")
    dpi: int = Field(default=200, gt=0, description="Plot DPI")
    figsize_spots: tuple[float, float] = Field(default=(10.0, 10.0), description="Figure size for spots plot")
    figsize_thresh: tuple[float, float] = Field(
        default=(8.0, 6.0), description="Figure size for threshold plot"
    )

    # Imaging meta
    pixel_size_um: float = Field(default=0.108, gt=0, description="Pixel size in micrometers")

    # Note: previously derived from SpotAnalysisConfig; now SpotThresholdParams is the source of truth.


# === Align/Spot decoding config (centralized for align_prod.py) ===


class SpotDecodeConfig(BaseModel):
    """Decoding + prefilter parameters used by align_prod.

    These intentionally mirror semantics from SpotThresholdParams to avoid drift.
    """

    max_distance: float = 0.3  # ↔ SpotThresholdParams.distance_threshold
    min_intensity: float = 0.002  # ↔ SpotThresholdParams.norm_threshold
    min_area: int = 8  # ↔ SpotThresholdParams.area_min
    max_area: int = 200  # ↔ SpotThresholdParams.area_max
    use_correct_direction: bool = True
    sigma: tuple[float, float, float] = (
        2.0,
        2.0,
        2.0,
    )  # (z, y, x) for GaussianHighPass
    threads: int = 12  # CAF/decoding parallelism where applicable
    clip_percentile: float = 40.0  # used by align_prod find_threshold


# Optimize round defaults used by align_prod when --calc-deviations is enabled
OPTIMIZE_DECODE = SpotDecodeConfig(
    min_intensity=0.012,
    max_distance=0.3,
    min_area=15,
    max_area=200,
    sigma=(2.0, 2.0, 2.0),
    use_correct_direction=True,
)


"""Decoding knobs are centralized via SpotDecodeConfig and embedded in Config."""


class SystemConfig(BaseModel):
    """System paths and infrastructure configuration."""

    forbidden_prefixes: list[str] = Field(
        default=["10x", "registered", "shifts", "fids", "analysis", "old", "basic"],
        description="Directory prefixes to exclude from processing",
    )
    default_references: list[str] = Field(
        default=["2_10_18", "7_15_23"],
        description="Default reference round identifiers in priority order",
    )
    available_channels: list[str] = Field(
        default=["405", "488", "560", "650", "750"],
        description="Available laser wavelength channels",
    )


class ImageProcessingConfig(BaseModel):
    """Low-level image processing parameters."""

    image_size: int = Field(default=2048, description="Standard image size in pixels")
    pixel_size_um: float = Field(
        default=0.108,
        description="Pixel size in micrometers for plotting/scale bars (used by spotlook and figures)",
    )
    log_sigma: float = Field(default=3.0, description="Sigma parameter for Laplacian of Gaussian filtering")
    percentiles: list[float] = Field(
        default=[1.0, 99.99],
        description="Percentile values for intensity normalization",
    )
    spillover_corrections: dict[str, float] | None = Field(
        default=None,
        description='Spectral spillover correction factors. Example: {"560_647": 0.22, "647_750": 0.05}',
    )
    # Enhanced fiducial processing parameters
    max_iterations: int = Field(default=5, description="Maximum iterations for fiducial alignment")
    residual_threshold: float = Field(
        default=0.3, description="Residual threshold for fiducial alignment convergence"
    )
    priors_min_count: int = Field(
        default=10, description="Minimum number of existing shifts to use as priors"
    )
    correlation_region: dict[str, int] = Field(
        default={"start": 500, "end": -500, "step": 2},
        description="Region for correlation calculation",
    )


class Config(BaseModel):
    """Master configuration for fishtools preprocessing pipeline."""

    dataPath: str | None = None
    exclude: list[str] | None = Field(None, description="Exclude rounds with these prefixes.")
    system: SystemConfig | None = Field(
        default_factory=SystemConfig,
        description="System paths and infrastructure configuration",
    )
    # RegisterConfig cannot have a safe default (requires chromatic_shifts); leave optional
    registration: RegisterConfig = Field(
        default_factory=default_register_config,
        description="Image registration parameters",
    )
    image_processing: ImageProcessingConfig | None = Field(
        default_factory=ImageProcessingConfig,
        description="Low-level image processing parameters",
    )
    deconvolution: DeconvolutionConfig | None = Field(
        default_factory=DeconvolutionConfig,
        description="3D deconvolution processing configuration",
    )
    stitching: StitchingConfig | None = Field(
        default_factory=StitchingConfig,
        description="Image stitching and fusion configuration",
    )
    spot_decode: SpotDecodeConfig | None = Field(
        default_factory=SpotDecodeConfig, description="Spot decoding parameters"
    )
    spot_threshold: SpotThresholdParams | None = Field(
        default_factory=SpotThresholdParams, description="Threshold parameters"
    )
