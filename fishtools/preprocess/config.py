from json import JSONEncoder
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, PlainSerializer, field_validator


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
        default=0.5, description="Step size for threshold adjustment during spot finding"
    )
    min_threshold_sigma: float = Field(
        default=2.0, description="Minimum threshold sigma before adjusting FWHM"
    )
    min_fwhm: float = Field(default=4.0, description="Minimum FWHM before reducing in exception handling")

    # Image preprocessing
    percentile_clip: float = Field(
        default=50.0, description="Percentile value for image clipping during preprocessing"
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
        default=False, description="Use FFT to find fiducial spots. Overrides everything else."
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
        default_factory=FiducialDetailedConfig, description="Detailed fiducial processing parameters"
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
        default=30,
        description="Pixels to crop from each edge. This is to account for translation during alignment.",
    )
    slices: Annotated[
        list[tuple[int | None, int | None]] | slice | str, PlainSerializer(lambda x: str(x))
    ] = Field(default=slice(None), description="Slice range to use for registration")
    split_channels: bool = False
    chromatic_shifts: dict[str, Annotated[str, "path for 560to{channel}.txt"]]
    reference: str = Field(default="4_12_20", description="Reference round to align others to.")
    # Moved from HardwareConfig - threads used specifically for registration
    threads: int = Field(default=15, description="Number of threads for registration operations")

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


class ChannelConfig(BaseModel):
    discards: dict[str, list[str]] | None = Field(
        None,
        description="In case of duplicated key(s), discard the ones from the file in value.",
    )


class ProcessingConfig(BaseModel):
    """Generic processing configuration for cross-cutting concerns."""

    queue_max_sizes: list[int] = Field(
        default=[3, 1], description="Maximum queue sizes for write and image processing"
    )


class DeconvolutionConfig(BaseModel):
    """3D deconvolution processing configuration."""

    projector_step: int = Field(default=6, description="PSF projector step size parameter")
    wiener_params: dict[str, float] = Field(
        default={"alpha": 0.02, "beta": 0.02, "n": 10},
        description="Wiener-Butterworth deconvolution parameters",
    )
    percentiles: dict[str, float] = Field(
        default={"min": 0.1, "max": 99.999}, description="Percentile values for intensity scaling"
    )
    protein_percentiles: dict[str, float] = Field(
        default={"min": 50, "max": 50}, description="Percentile values specifically for protein channels"
    )
    max_rna_bit: int = Field(default=36, description="Maximum bit number considered RNA (vs protein)")
    # Moved from HardwareConfig - threads used specifically for deconvolution
    threads: int = Field(default=6, description="Number of threads for deconvolution operations")


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
        default=102400, description="Maximum memory allocation in MB for ImageJ stitching operations"
    )
    parallel_threads: int = Field(
        default=32, description="Number of parallel threads for ImageJ stitching operations"
    )
    # Moved from HardwareConfig - threads used specifically for stitching
    threads: int = Field(default=8, description="Number of threads for stitching operations")


class SpotAnalysisConfig(BaseModel):
    """Spot detection and analysis configuration."""

    area_range: list[float] = Field(default=[10.0, 200.0], description="Valid spot area range in pixels")
    norm_threshold: float = Field(default=0.007, description="Normalization threshold for spot filtering")
    distance_threshold: float = Field(default=0.3, description="Distance threshold for spot filtering")
    density: dict[str, float | int] = Field(
        default={"grid_size": 50, "smooth_sigma": 4.0, "min_spots": 100},
        description="Density map calculation parameters",
    )
    seed: int = Field(default=0, description="Random seed for reproducible analysis")
    visualization: dict[str, int | float | list[int | float]] = Field(
        default={
            "subsample": 200000,
            "scale_bar_um": 1000,
            "thumbnail_interval": 8,
            "dpi": 200,
            "figsize_spots": [10, 10],
            "figsize_thresh": [8, 6],
        },
        description="Visualization and output parameters",
    )


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
        default=["405", "488", "560", "650", "750"], description="Available laser wavelength channels"
    )


class ImageProcessingConfig(BaseModel):
    """Low-level image processing parameters."""

    image_size: int = Field(default=2048, description="Standard image size in pixels")
    log_sigma: float = Field(default=3.0, description="Sigma parameter for Laplacian of Gaussian filtering")
    percentiles: list[float] = Field(
        default=[1.0, 99.99], description="Percentile values for intensity normalization"
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
        default={"start": 500, "end": -500, "step": 2}, description="Region for correlation calculation"
    )


class Config(BaseModel):
    """Master configuration for fishtools preprocessing pipeline."""

    # Existing fields for backward compatibility
    dataPath: str
    exclude: list[str] | None = Field(None, description="Exclude rounds with these prefixes.")
    registration: RegisterConfig
    channels: ChannelConfig | None = None
    basic: Literal["per_round", "all_round"] | None = Field(
        None,
        description="Perform BaSiC correction specific to each round or use the same template for all rounds.",
    )

    # New configuration sections (optional for backward compatibility)
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Generic processing configuration"
    )
    deconvolution: DeconvolutionConfig = Field(
        default_factory=DeconvolutionConfig, description="3D deconvolution processing configuration"
    )
    basic_correction: BasicConfig = Field(
        default_factory=BasicConfig, description="BaSiC correction configuration"
    )
    stitching: StitchingConfig = Field(
        default_factory=StitchingConfig, description="Image stitching and fusion configuration"
    )
    spot_analysis: SpotAnalysisConfig = Field(
        default_factory=SpotAnalysisConfig, description="Spot detection and analysis configuration"
    )
    system: SystemConfig = Field(
        default_factory=SystemConfig, description="System paths and infrastructure configuration"
    )
    image_processing: ImageProcessingConfig = Field(
        default_factory=ImageProcessingConfig, description="Low-level image processing parameters"
    )
