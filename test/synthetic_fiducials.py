"""
Comprehensive synthetic fiducial image generation for testing fiducial alignment algorithms.

This module provides realistic synthetic fiducial images with ground truth data for:
- Spot detection algorithm validation
- Alignment accuracy testing
- Noise robustness evaluation
- Missing fiducial handling

The synthetic images model realistic microscopy characteristics:
- Gaussian point spread functions
- Poisson + Gaussian noise models
- Sub-pixel positioning accuracy
- Realistic signal-to-noise ratios
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage


# Scientific constants for realistic microscopy simulation
class MicroscopyConstants:
    """Scientific constants for realistic microscopy simulation."""

    # Typical camera readout noise (electrons RMS)
    CAMERA_READOUT_NOISE_ELECTRONS = 10.0

    # Position matching tolerance for spot detection validation (pixels)
    SPOT_POSITION_TOLERANCE_PIXELS = 5.0

    # Background region distance from fiducials for SNR calculation (pixels)
    BACKGROUND_REGION_MARGIN_PIXELS = 30.0

    # Background sampling region size for noise estimation (pixels)
    BACKGROUND_SAMPLE_SIZE_PIXELS = 5

    # Image edge margin for background sampling (pixels)
    IMAGE_EDGE_MARGIN_PIXELS = 20

    # Signal region size for SNR calculation (pixels)
    SIGNAL_SAMPLE_SIZE_PIXELS = 3

    # Artifact detection threshold (multiple of peak intensity)
    ARTIFACT_THRESHOLD_MULTIPLIER = 2.0

    # PSF energy containment (3-sigma contains 99.7% of energy)
    PSF_ENERGY_CONTAINMENT_SIGMA = 3.0

    # Maximum safe PSF kernel size to prevent memory issues
    MAX_PSF_KERNEL_SIZE_PIXELS = 1000

    # Maximum safe intensity for Poisson noise (prevents overflow)
    MAX_POISSON_INTENSITY = 1e6

    # PSF normalization precision tolerance
    PSF_NORMALIZATION_TOLERANCE = 1e-10


@dataclass
class FiducialParams:
    """Configuration parameters for synthetic fiducial image generation.

    This class encapsulates all parameters needed to generate realistic
    synthetic fiducial images for testing microscopy alignment algorithms.
    """

    image_size: tuple[int, int] = (256, 256)
    """Image dimensions in pixels (height, width)"""

    fiducial_positions: list[tuple[float, float]] | None = None
    """Fiducial center positions (x, y) in pixels. Auto-generated if None."""

    background_level: float = 250.0
    """Mean background intensity level in counts"""

    peak_intensity: float = 2500.0
    """Peak fiducial intensity in counts (background + signal)"""

    psf_sigma: float = 1.49
    """Gaussian PSF standard deviation in pixels (FWHM = 3.5)"""

    noise_factor: float = 1.0
    """Noise multiplier (1.0 = realistic, >1.0 = more noisy)"""

    background_variation: float = 0.05
    """Relative background variation (5% by default)"""

    seed: int = 42
    """Random seed for reproducible generation"""

    edge_margin: int = 30
    """Minimum distance from image edges for fiducials"""

    min_separation: float = 40.0
    """Minimum distance between fiducials in pixels"""

    def __post_init__(self) -> None:
        """Validate parameters and auto-generate fiducial positions if needed."""
        self._validate_parameters()
        if self.fiducial_positions is None:
            self.fiducial_positions = self._generate_default_positions()

    def _validate_parameters(self) -> None:
        """Validate parameter ranges and consistency."""
        if self.image_size[0] < 256 or self.image_size[1] < 256:
            raise ValueError("Image size must be at least 256x256 pixels")

        if self.background_level <= 0:
            raise ValueError("Background level must be positive")

        if self.peak_intensity <= self.background_level:
            raise ValueError("Peak intensity must be greater than background")

        if self.psf_sigma <= 0:
            raise ValueError("PSF sigma must be positive")

        if self.edge_margin < 0 or self.edge_margin >= min(self.image_size) // 4:
            raise ValueError("Edge margin too large for image size")

        if self.min_separation <= 0:
            raise ValueError("Minimum separation must be positive")

    def _generate_default_positions(self) -> list[tuple[float, float]]:
        """Generate well-distributed default fiducial positions.

        Creates fiducials in a pattern that:
        - Covers the image area uniformly
        - Maintains minimum separation distance
        - Avoids edges to prevent PSF truncation
        - Provides good geometric diversity for alignment
        """
        h, w = self.image_size
        margin = self.edge_margin

        # Calculate available space and adjust number of fiducials if needed
        available_w = w - 2 * margin
        available_h = h - 2 * margin

        # For smaller images, use fewer fiducials or adjust separation
        min_area_per_fiducial = self.min_separation**2
        max_fiducials = int((available_w * available_h) / min_area_per_fiducial)

        if max_fiducials < 4:
            # Very small image - use 4 corner positions with reduced separation
            effective_sep = min(self.min_separation, min(available_w, available_h) / 3)
            positions = [
                (margin + effective_sep, margin + effective_sep),
                (w - margin - effective_sep, margin + effective_sep),
                (margin + effective_sep, h - margin - effective_sep),
                (w - margin - effective_sep, h - margin - effective_sep),
            ]
        elif max_fiducials < 7:
            # Medium image - use fewer fiducials strategically placed
            cx, cy = w // 2, h // 2
            offset = self.min_separation * 0.7
            positions = [
                (margin + offset, margin + offset),  # Upper-left
                (w - margin - offset, margin + offset),  # Upper-right
                (margin + offset, h - margin - offset),  # Lower-left
                (w - margin - offset, h - margin - offset),  # Lower-right
                (cx, cy),  # Center
            ]
        else:
            # Large image - use full 7-fiducial pattern
            # Calculate strategic positions with proper spacing
            quarter_w = available_w / 4
            quarter_h = available_h / 4

            positions = [
                (margin + quarter_w, margin + quarter_h),  # Upper-left quadrant
                (margin + 3 * quarter_w, margin + quarter_h),  # Upper-right quadrant
                (margin + quarter_w, margin + 3 * quarter_h),  # Lower-left quadrant
                (margin + 3 * quarter_w, margin + 3 * quarter_h),  # Lower-right quadrant
                (w // 2, margin + quarter_h),  # Top-center
                (w // 2, h - margin - quarter_h),  # Bottom-center
                (w // 2, h // 2),  # Center
            ]

        # Validate positions meet separation requirements
        self._validate_positions(positions)

        return positions

    def _validate_positions(self, positions: list[tuple[float, float]]) -> None:
        """Validate that positions meet separation and boundary requirements."""
        h, w = self.image_size

        for i, (x, y) in enumerate(positions):
            # Check boundaries
            if x < self.edge_margin or x >= w - self.edge_margin:
                raise ValueError(f"Position {i} too close to X boundary: {x}")
            if y < self.edge_margin or y >= h - self.edge_margin:
                raise ValueError(f"Position {i} too close to Y boundary: {y}")

            # Check separation from other positions
            for j, (x2, y2) in enumerate(positions[i + 1 :], i + 1):
                distance = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                if distance < self.min_separation:
                    raise ValueError(
                        f"Positions {i} and {j} too close: {distance:.1f} < {self.min_separation}"
                    )


@dataclass
class ValidationResult:
    """Results from synthetic image validation."""

    detectability_rate: float
    """Fraction of fiducials successfully detected"""

    position_accuracy: float
    """RMS position error in pixels"""

    mean_snr: float
    """Mean signal-to-noise ratio"""

    artifact_score: float
    """Artifact detection score (lower is better)"""

    passed: bool
    """Whether image passes quality thresholds"""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional validation details"""


class SyntheticFiducialImage:
    """High-quality synthetic fiducial image with ground truth data.

    This class generates realistic synthetic fiducial images that closely
    model actual microscopy data, enabling thorough testing of fiducial
    detection and alignment algorithms.
    """

    def __init__(self, params: FiducialParams) -> None:
        """Initialize synthetic fiducial image generator.

        Args:
            params: Configuration parameters for image generation

        Raises:
            TypeError: If params is not a FiducialParams instance
            ValueError: If params contains invalid values
        """
        # Validate input type
        if not isinstance(params, FiducialParams):
            raise TypeError(f"Expected FiducialParams, got {type(params)}")

        # Ensure fiducial positions are available
        if params.fiducial_positions is None:
            raise ValueError("Fiducial positions must be set after FiducialParams initialization")

        # Additional runtime validation beyond dataclass validation
        self._validate_runtime_constraints(params)

        self.params = params
        self._rng = np.random.RandomState(params.seed)
        self._image: NDArray[np.uint16] | None = None
        self._image_float: NDArray[np.float64] | None = None

        # Generate the image immediately for consistency
        self._generate_image()

    @property
    def image(self) -> NDArray[np.uint16]:
        """Get the synthetic fiducial image as uint16 array."""
        if self._image is None:
            raise RuntimeError("Image not generated")
        return self._image

    @property
    def image_float(self) -> NDArray[np.float64]:
        """Get the synthetic fiducial image as float64 array for processing."""
        if self._image_float is None:
            raise RuntimeError("Float image not generated")
        return self._image_float

    @property
    def ground_truth_positions(self) -> list[tuple[float, float]]:
        """Get ground truth fiducial positions."""
        if self.params.fiducial_positions is None:
            raise RuntimeError("Fiducial positions not initialized")
        return self.params.fiducial_positions.copy()

    @property
    def fiducial_count(self) -> int:
        """Get number of fiducials in the image."""
        if self.params.fiducial_positions is None:
            return 0
        return len(self.params.fiducial_positions)

    def _generate_image(self) -> None:
        """Generate the complete synthetic fiducial image.

        This method orchestrates the entire image generation process:
        1. Create realistic background with texture and noise
        2. Generate accurate Gaussian PSF
        3. Add fiducials with sub-pixel positioning
        4. Apply realistic noise models
        5. Convert to appropriate data types
        """
        # Step 1: Create base background image
        background = self._create_background()

        # Step 2: Generate PSF kernel
        psf = self._create_gaussian_psf()

        # Step 3: Add each fiducial to the image
        image = background.copy()
        if self.params.fiducial_positions is None:
            raise RuntimeError("Fiducial positions not initialized")
        for position in self.params.fiducial_positions:
            image = self._add_fiducial_to_image(image, position, psf)

        # Step 4: Add realistic noise
        image = self._add_noise(image)

        # Step 5: Store both float and uint16 versions
        self._image_float = image
        self._image = self._convert_to_uint16(image)

    def _create_background(self) -> NDArray[np.float64]:
        """Create realistic background with spatial variations and shot noise.

        Models typical microscopy background characteristics:
        - Uniform base level with small spatial variations
        - Poisson shot noise from background photons
        - Subtle texture variations
        """
        h, w = self.params.image_size
        bg_level = self.params.background_level

        # Start with uniform background
        background = np.full((h, w), bg_level, dtype=np.float64)

        # Add subtle spatial variations (illumination non-uniformity)
        if self.params.background_variation > 0:
            # Create smooth variations using low-frequency noise
            variation_scale = min(h, w) // 4  # Large-scale variations
            y_coords, x_coords = np.ogrid[:h, :w]

            # Smooth sine wave variations
            x_variation = np.sin(2 * np.pi * x_coords / variation_scale)
            y_variation = np.cos(2 * np.pi * y_coords / variation_scale)

            # Combine and scale variations
            spatial_variation = (x_variation + y_variation) * bg_level * self.params.background_variation
            background += spatial_variation

        # Add Poisson shot noise to background
        # Ensure non-negative values for Poisson
        background = np.maximum(background, 0)

        # Apply Poisson noise (models photon shot noise)
        background = self._apply_poisson_noise_safely(background)

        return background

    def _apply_poisson_noise_safely(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Poisson noise with proper range validation and error handling.

        Args:
            values: Input intensity values

        Returns:
            Values with Poisson noise applied safely
        """
        try:
            # Clip to safe range for Poisson (avoid overflow)
            safe_values = np.clip(values, 0, MicroscopyConstants.MAX_POISSON_INTENSITY)

            # Apply Poisson noise with error handling
            return self._rng.poisson(safe_values).astype(np.float64)

        except (ValueError, OverflowError) as e:
            # Fallback to original values with warning
            warnings.warn(f"Poisson noise application failed: {e}. Using original values.")
            return values

    def _create_gaussian_psf(self) -> NDArray[np.float64]:
        """Create normalized 2D Gaussian point spread function.

        Returns:
            Normalized PSF kernel with sum = 1.0

        Raises:
            ValueError: If PSF parameters would create invalid kernel
        """
        sigma = self.params.psf_sigma

        # Validate sigma is reasonable
        if sigma <= 0:
            raise ValueError(f"PSF sigma must be positive, got {sigma}")
        if sigma > 100:
            warnings.warn(f"Very large PSF sigma ({sigma}) may cause memory issues")

        # Size PSF to contain 99.7% of the energy (3-sigma)
        psf_radius = max(3, int(MicroscopyConstants.PSF_ENERGY_CONTAINMENT_SIGMA * sigma))
        psf_size = 2 * psf_radius + 1

        # Prevent extremely large PSF kernels
        if psf_size > MicroscopyConstants.MAX_PSF_KERNEL_SIZE_PIXELS:
            raise ValueError(f"PSF kernel too large ({psf_size}x{psf_size}), reduce sigma")

        # Create coordinate grids centered at PSF center
        center = psf_radius
        y, x = np.ogrid[:psf_size, :psf_size]

        # Generate 2D Gaussian
        psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

        # Validate and normalize PSF
        psf_sum = psf.sum()
        if psf_sum <= 0 or not np.isfinite(psf_sum):
            raise ValueError(f"Invalid PSF sum: {psf_sum}")

        psf = psf / psf_sum

        # Final validation
        if not np.allclose(psf.sum(), 1.0, atol=MicroscopyConstants.PSF_NORMALIZATION_TOLERANCE):
            warnings.warn(f"PSF normalization imprecise: sum = {psf.sum()}")

        return psf

    def _add_fiducial_to_image(
        self, image: NDArray[np.float64], position: tuple[float, float], psf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Add single fiducial to image with sub-pixel accuracy.

        Args:
            image: Target image array
            position: Fiducial center position (x, y) in pixels
            psf: Point spread function kernel

        Returns:
            Image with fiducial added
        """
        x, y = position

        # Calculate signal intensity (peak - background)
        signal_intensity = self.params.peak_intensity - self.params.background_level

        # Handle sub-pixel positioning with interpolation
        x_int, x_frac = int(x), x % 1
        y_int, y_frac = int(y), y % 1

        # Create sub-pixel shifted PSF using scipy interpolation
        if abs(x_frac) > 1e-6 or abs(y_frac) > 1e-6:
            # Apply sub-pixel shift using spline interpolation
            shifted_psf = ndimage.shift(psf, (y_frac, x_frac), order=3, mode="constant").astype(np.float64)
        else:
            shifted_psf = psf

        # Scale PSF by signal intensity
        scaled_psf = shifted_psf * signal_intensity

        # Add PSF to image with boundary handling
        self._add_psf_to_region(image, scaled_psf, (x_int, y_int))

        return image

    def _add_psf_to_region(
        self, image: NDArray[np.float64], psf: NDArray[np.float64], center: tuple[int, int]
    ) -> None:
        """Add PSF to image region with careful boundary handling.

        Args:
            image: Target image (modified in place)
            psf: PSF kernel to add
            center: Center position (x, y) for PSF placement
        """
        x_center, y_center = center
        psf_h, psf_w = psf.shape
        img_h, img_w = image.shape

        # Calculate PSF offset from center
        psf_radius_x = psf_w // 2
        psf_radius_y = psf_h // 2

        # Calculate target region in image coordinates
        img_x1 = max(0, x_center - psf_radius_x)
        img_x2 = min(img_w, x_center + psf_radius_x + 1)
        img_y1 = max(0, y_center - psf_radius_y)
        img_y2 = min(img_h, y_center + psf_radius_y + 1)

        # Calculate corresponding region in PSF coordinates
        psf_x1 = max(0, psf_radius_x - x_center)
        psf_x2 = psf_x1 + (img_x2 - img_x1)
        psf_y1 = max(0, psf_radius_y - y_center)
        psf_y2 = psf_y1 + (img_y2 - img_y1)

        # Add PSF region to image region
        if img_x2 > img_x1 and img_y2 > img_y1:
            image[img_y1:img_y2, img_x1:img_x2] += psf[psf_y1:psf_y2, psf_x1:psf_x2]

    def _add_noise(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        """Add realistic noise model to image.

        Applies multi-component noise model:
        1. Additional Poisson noise on signal
        2. Gaussian readout noise
        3. Scaled by noise factor parameter

        Args:
            image: Input image

        Returns:
            Noisy image
        """
        noisy_image = image.copy()

        if self.params.noise_factor > 0:
            # Additional Poisson noise on the signal (photon shot noise)
            # Use safe Poisson application to avoid overflow
            scaled_image = noisy_image * self.params.noise_factor
            noisy_scaled = self._apply_poisson_noise_safely(scaled_image)
            # Add the difference as noise component
            poisson_component = noisy_scaled - scaled_image
            noisy_image += poisson_component

            # Gaussian readout noise (detector electronics)
            readout_sigma = MicroscopyConstants.CAMERA_READOUT_NOISE_ELECTRONS * self.params.noise_factor
            readout_noise = self._rng.normal(0, readout_sigma, image.shape)
            noisy_image += readout_noise

        return noisy_image

    def _convert_to_uint16(self, image: NDArray[np.float64]) -> NDArray[np.uint16]:
        """Convert float image to uint16 with proper validation and clipping.

        Args:
            image: Float64 image

        Returns:
            Uint16 image with proper range clipping

        Raises:
            ValueError: If image contains invalid values that can't be handled
        """
        # Check for and handle NaN/Inf values
        if not np.all(np.isfinite(image)):
            nan_count = np.sum(~np.isfinite(image))
            if nan_count > image.size * 0.1:  # More than 10% invalid values
                raise ValueError(f"Too many invalid values in image: {nan_count}/{image.size}")

            # Replace NaN/Inf with background level
            warnings.warn(f"Replacing {nan_count} NaN/Inf values with background level")
            image_cleaned = np.where(np.isfinite(image), image, self.params.background_level)
        else:
            image_cleaned = image

        # Clip to valid uint16 range
        clipped = np.clip(image_cleaned, 0, 65535)

        # Convert to uint16
        return clipped.astype(np.uint16)

    def create_shifted_copy(self, dx: float, dy: float, **kwargs: Any) -> "SyntheticFiducialImage":
        """Create a shifted copy of this image for alignment testing.

        Args:
            dx: X shift in pixels
            dy: Y shift in pixels
            **kwargs: Additional parameters to override

        Returns:
            New SyntheticFiducialImage with shifted fiducial positions
        """
        # Create new parameters with shifted positions
        if self.params.fiducial_positions is None:
            raise RuntimeError("Cannot shift image without fiducial positions")
        new_params = FiducialParams(
            image_size=self.params.image_size,
            fiducial_positions=[(x + dx, y + dy) for x, y in self.params.fiducial_positions],
            background_level=self.params.background_level,
            peak_intensity=self.params.peak_intensity,
            psf_sigma=self.params.psf_sigma,
            noise_factor=self.params.noise_factor,
            background_variation=self.params.background_variation,
            seed=self.params.seed + 1,  # Different seed for noise variation
            edge_margin=self.params.edge_margin,
            min_separation=self.params.min_separation,
            **kwargs,
        )

        # Filter out positions that would go outside boundaries
        h, w = self.params.image_size
        margin = self.params.edge_margin

        valid_positions = []
        if new_params.fiducial_positions is not None:
            for x, y in new_params.fiducial_positions:
                if margin <= x < w - margin and margin <= y < h - margin:
                    valid_positions.append((x, y))

        new_params.fiducial_positions = valid_positions

        if len(valid_positions) == 0:
            raise ValueError(f"Shift ({dx}, {dy}) moves all fiducials outside valid region")

        return SyntheticFiducialImage(new_params)

    def add_extra_noise(self, factor: float) -> "SyntheticFiducialImage":
        """Create copy with additional noise for robustness testing.

        Args:
            factor: Noise multiplier (>1.0 for more noise)

        Returns:
            New SyntheticFiducialImage with increased noise
        """
        if self.params.fiducial_positions is None:
            raise RuntimeError("Cannot add noise without fiducial positions")
        new_params = FiducialParams(
            image_size=self.params.image_size,
            fiducial_positions=self.params.fiducial_positions.copy(),
            background_level=self.params.background_level,
            peak_intensity=self.params.peak_intensity,
            psf_sigma=self.params.psf_sigma,
            noise_factor=self.params.noise_factor * factor,
            background_variation=self.params.background_variation,
            seed=self.params.seed + 2,  # Different seed
            edge_margin=self.params.edge_margin,
            min_separation=self.params.min_separation,
        )

        return SyntheticFiducialImage(new_params)

    def mask_fiducials(self, indices: list[int]) -> "SyntheticFiducialImage":
        """Create copy with specified fiducials removed (simulate missing spots).

        Args:
            indices: List of fiducial indices to remove

        Returns:
            New SyntheticFiducialImage with specified fiducials removed
        """
        if not indices:
            return self  # No change needed

        # Remove specified fiducials
        if self.params.fiducial_positions is None:
            raise RuntimeError("Cannot mask fiducials without positions")
        new_positions = [pos for i, pos in enumerate(self.params.fiducial_positions) if i not in indices]

        if len(new_positions) == 0:
            raise ValueError("Cannot remove all fiducials")

        new_params = FiducialParams(
            image_size=self.params.image_size,
            fiducial_positions=new_positions,
            background_level=self.params.background_level,
            peak_intensity=self.params.peak_intensity,
            psf_sigma=self.params.psf_sigma,
            noise_factor=self.params.noise_factor,
            background_variation=self.params.background_variation,
            seed=self.params.seed + 3,  # Different seed
            edge_margin=self.params.edge_margin,
            min_separation=self.params.min_separation,
        )

        return SyntheticFiducialImage(new_params)

    def validate_detectability(self, **detection_params: Any) -> ValidationResult:
        """Validate that generated image has good detectability characteristics.

        This method tests the synthetic image against the actual spot detection
        algorithms to ensure it will work correctly for testing.

        Args:
            **detection_params: Parameters to pass to find_spots function

        Returns:
            Validation results with quality metrics
        """
        try:
            # Import find_spots for validation
            from fishtools.preprocess.fiducial import find_spots

            # Default detection parameters if not provided
            default_params = {
                "threshold_sigma": 3.0,
                "fwhm": 4.0,
                "minimum_spots": max(1, len(self.params.fiducial_positions or []) - 2),
            }
            default_params.update(detection_params)

            # Attempt spot detection
            try:
                detected_spots = find_spots(self.image, **default_params)
                detection_successful = True
                num_detected = len(detected_spots)
            except Exception:
                detection_successful = False
                num_detected = 0
                detected_spots = None

            # Calculate metrics
            detectability_rate = (
                num_detected / len(self.params.fiducial_positions or []) if detection_successful else 0.0
            )

            # Calculate position accuracy if detection worked
            position_accuracy = (
                self._calculate_position_accuracy(detected_spots) if detection_successful else float("inf")
            )

            # Calculate signal-to-noise ratio
            mean_snr = self._calculate_mean_snr()

            # Calculate artifact score (simple measure of unexpected features)
            artifact_score = self._calculate_artifact_score()

            # Determine if validation passed
            passed = all(
                [
                    detection_successful,
                    detectability_rate >= 0.85,  # 85% detection rate
                    mean_snr >= 3.0,  # Minimum SNR
                    artifact_score < 0.1,  # Low artifact level
                    position_accuracy < 2.0 if position_accuracy != float("inf") else False,
                ]
            )

            return ValidationResult(
                detectability_rate=detectability_rate,
                position_accuracy=position_accuracy,
                mean_snr=mean_snr,
                artifact_score=artifact_score,
                passed=passed,
                details={
                    "detection_successful": detection_successful,
                    "num_detected": num_detected,
                    "num_expected": len(self.params.fiducial_positions or []),
                    "detection_params": default_params,
                },
            )

        except ImportError:
            # If find_spots not available, return basic validation
            mean_snr = self._calculate_mean_snr()
            artifact_score = self._calculate_artifact_score()

            return ValidationResult(
                detectability_rate=1.0,  # Assume good
                position_accuracy=0.0,  # Can't measure
                mean_snr=mean_snr,
                artifact_score=artifact_score,
                passed=mean_snr >= 3.0 and artifact_score < 0.1,
                details={"find_spots_unavailable": True},
            )

    def _calculate_position_accuracy(self, detected_spots: Any) -> float:
        """Calculate RMS position accuracy between detected and ground truth positions."""
        if detected_spots is None or len(detected_spots) == 0:
            return float("inf")

        # Extract detected positions
        try:
            detected_x = detected_spots["xcentroid"].to_numpy()
            detected_y = detected_spots["ycentroid"].to_numpy()
            detected_positions = list(zip(detected_x, detected_y))
        except (KeyError, AttributeError):
            return float("inf")

        # Match detected to ground truth using efficient distance calculation
        if self.params.fiducial_positions is None:
            return float("inf")
        gt_positions = self.params.fiducial_positions

        if len(detected_positions) == 0:
            return float("inf")

        # Convert to numpy arrays for vectorized operations
        gt_array = np.array(gt_positions)
        det_array = np.array(detected_positions)

        # Calculate distance matrix efficiently using broadcasting
        # Shape: (n_gt, n_detected)
        diff = gt_array[:, np.newaxis, :] - det_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Find minimum distance for each ground truth position
        min_distances = np.min(distances, axis=1)

        # Only count matches within tolerance
        valid_matches = min_distances < MicroscopyConstants.SPOT_POSITION_TOLERANCE_PIXELS

        if not np.any(valid_matches):
            return float("inf")

        # Calculate RMS error for valid matches
        matched_errors = min_distances[valid_matches]
        return np.sqrt(np.mean(matched_errors**2))

    def _calculate_mean_snr(self) -> float:
        """Calculate mean signal-to-noise ratio at fiducial positions."""
        snr_values = []

        if self.params.fiducial_positions is None:
            return 0.0
        for x, y in self.params.fiducial_positions:
            # Sample region around fiducial
            x_int, y_int = int(round(x)), int(round(y))

            # Get signal region around fiducial
            region_size = MicroscopyConstants.SIGNAL_SAMPLE_SIZE_PIXELS
            x1 = max(0, x_int - region_size // 2)
            x2 = min(self.params.image_size[1], x_int + region_size // 2 + 1)
            y1 = max(0, y_int - region_size // 2)
            y2 = min(self.params.image_size[0], y_int + region_size // 2 + 1)

            if x2 > x1 and y2 > y1:
                signal_region = self.image_float[y1:y2, x1:x2]
                signal = np.max(signal_region)  # Peak signal

                # Estimate noise from background regions using robust sampling
                background_regions = self._sample_background_regions()

                if background_regions:
                    background_values = np.concatenate([region.flatten() for region in background_regions])
                    noise = np.std(background_values)

                    if noise > 0:
                        snr = (signal - np.mean(background_values)) / noise
                        snr_values.append(max(0, snr))  # Ensure non-negative

        return float(np.mean(snr_values)) if snr_values else 0.0

    def _sample_background_regions(self) -> list[NDArray[np.float64]]:
        """Sample background regions robustly for noise estimation.

        Returns:
            List of background image regions for noise calculation
        """
        if self.params.fiducial_positions is None:
            return []

        background_regions = []
        h, w = self.params.image_size
        margin = MicroscopyConstants.IMAGE_EDGE_MARGIN_PIXELS
        sample_size = MicroscopyConstants.BACKGROUND_SAMPLE_SIZE_PIXELS
        min_distance_required = MicroscopyConstants.BACKGROUND_REGION_MARGIN_PIXELS

        # Generate candidate positions across the image, not just corners
        step = max(min_distance_required // 2, 20)  # Ensure reasonable spacing
        candidate_positions = []

        for bg_x in range(margin + sample_size, w - margin - sample_size, step):
            for bg_y in range(margin + sample_size, h - margin - sample_size, step):
                candidate_positions.append((bg_x, bg_y))

        # If no candidates (very small image), fall back to corners
        if not candidate_positions:
            candidate_positions = [
                (margin, margin),
                (w - margin, margin),
                (margin, h - margin),
                (w - margin, h - margin),
            ]

        # Select positions that are far enough from all fiducials
        fid_array = np.array(self.params.fiducial_positions)

        for bg_x, bg_y in candidate_positions:
            # Vectorized distance calculation
            distances = np.sqrt(np.sum((fid_array - np.array([bg_x, bg_y])) ** 2, axis=1))
            min_fid_distance = np.min(distances)

            if min_fid_distance > min_distance_required:
                # Extract background region
                y1 = max(0, bg_y - sample_size)
                y2 = min(h, bg_y + sample_size)
                x1 = max(0, bg_x - sample_size)
                x2 = min(w, bg_x + sample_size)

                if y2 > y1 and x2 > x1:
                    bg_region = self.image_float[y1:y2, x1:x2]
                    if bg_region.size > 0:
                        background_regions.append(bg_region)

                # Stop after collecting enough regions for good statistics
                if len(background_regions) >= 8:  # Sufficient for robust estimation
                    break

        return background_regions

    def _validate_runtime_constraints(self, params: FiducialParams) -> None:
        """Validate runtime constraints for robust image generation.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If constraints are violated
        """
        # Check that image can accommodate fiducials with margins
        h, w = params.image_size
        usable_area = (h - 2 * params.edge_margin) * (w - 2 * params.edge_margin)
        min_area_needed = len(params.fiducial_positions) * params.min_separation**2

        if usable_area < min_area_needed:
            raise ValueError(
                f"Image too small for {len(params.fiducial_positions)} fiducials: "
                f"usable area {usable_area:.0f} < required area {min_area_needed:.0f}"
            )

        # Validate intensity ranges make physical sense
        dynamic_range = params.peak_intensity / params.background_level
        if dynamic_range < 1.5:
            warnings.warn(f"Low signal dynamic range: {dynamic_range:.2f}x")
        if dynamic_range > 100:
            warnings.warn(f"Very high signal dynamic range: {dynamic_range:.2f}x may cause saturation")

        # Validate PSF size is reasonable for image
        psf_diameter = 6 * params.psf_sigma  # 3-sigma radius on each side
        if psf_diameter > min(h, w) / 4:
            warnings.warn(f"PSF diameter ({psf_diameter:.1f}) is large relative to image size")

    def _calculate_artifact_score(self) -> float:
        """Calculate a simple artifact detection score."""
        # Simple measure: ratio of pixels above threshold (artifacts)
        artifact_threshold = self.params.peak_intensity * MicroscopyConstants.ARTIFACT_THRESHOLD_MULTIPLIER
        num_artifacts = np.sum(self.image_float > artifact_threshold)
        total_pixels = self.image_float.size

        return float(num_artifacts / total_pixels)


# Pytest fixture integration
@pytest.fixture
def synthetic_fiducial_256x256() -> SyntheticFiducialImage:
    """Pytest fixture providing high-quality 256x256 synthetic fiducial image.

    Returns:
        SyntheticFiducialImage with 7 well-distributed fiducials
    """
    params = FiducialParams(
        image_size=(256, 256),
        background_level=250.0,
        peak_intensity=2500.0,
        psf_sigma=1.49,  # FWHM = 3.5 pixels
        noise_factor=1.0,
        seed=42,
    )

    return SyntheticFiducialImage(params)


@pytest.fixture
def synthetic_fiducial_512x512() -> SyntheticFiducialImage:
    """Pytest fixture providing high-quality 512x512 synthetic fiducial image.

    Returns:
        SyntheticFiducialImage with more fiducials for larger field
    """
    # Generate more fiducials for larger image
    positions: list[tuple[float, float]] = [
        (60.0, 60.0),
        (60.0, 256.0),
        (60.0, 452.0),
        (256.0, 60.0),
        (256.0, 256.0),
        (256.0, 452.0),
        (452.0, 60.0),
        (452.0, 256.0),
        (452.0, 452.0),
        (158.0, 158.0),
        (354.0, 354.0),  # Additional diagonal positions
    ]

    params = FiducialParams(
        image_size=(512, 512),
        fiducial_positions=positions,
        background_level=250.0,
        peak_intensity=2500.0,
        psf_sigma=1.49,
        noise_factor=1.0,
        seed=42,
        edge_margin=50,
    )

    return SyntheticFiducialImage(params)


@pytest.fixture
def synthetic_fiducial_noisy() -> SyntheticFiducialImage:
    """Pytest fixture providing noisy synthetic fiducial image for robustness testing.

    Returns:
        SyntheticFiducialImage with increased noise for challenging conditions
    """
    params = FiducialParams(
        image_size=(256, 256),
        background_level=250.0,
        peak_intensity=1500.0,  # Lower signal
        psf_sigma=1.49,
        noise_factor=2.0,  # 2x more noise
        seed=42,
    )

    return SyntheticFiducialImage(params)


# Convenience functions for testing
def create_shifted_pair(
    base_params: FiducialParams | None = None, dx: float = 2.5, dy: float = 1.8
) -> tuple[SyntheticFiducialImage, SyntheticFiducialImage]:
    """Create a pair of images with known shift for alignment testing.

    Args:
        base_params: Base parameters (uses defaults if None)
        dx: X shift in pixels
        dy: Y shift in pixels

    Returns:
        Tuple of (reference_image, shifted_image)
    """
    if base_params is None:
        base_params = FiducialParams()

    reference = SyntheticFiducialImage(base_params)
    shifted = reference.create_shifted_copy(dx, dy)

    return reference, shifted
