import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich
from astropy.stats import SigmaClip, sigma_clipped_stats
from loguru import logger
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from pydantic import BaseModel, TypeAdapter
from scipy.spatial import cKDTree
from skimage import exposure, filters
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform
from tifffile import TiffFile

from fishtools.preprocess.config import FiducialDetailedConfig, RegisterConfig

console = rich.get_console()


class TranslationTransform(AffineTransform):
    def estimate(self, src: np.ndarray, dst: np.ndarray) -> bool:  # type: ignore
        """Estimate the transform parameters from (n×d) points."""
        if src.shape[0] < 1:
            return False

        translation = np.median(dst - src, axis=0)
        self.params = np.eye(3)
        self.params[0:2, 2] = translation
        return True


def imread_page(path: Path | str, page: int):
    with TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


def butterworth(
    image: np.ndarray, cutoff: float = 0.05, squared_butterworth: bool = True, order: int = 3, npad: int = 0
) -> np.ndarray:
    """Apply Butterworth high-pass filter to enhance fiducial spot detection.

    High-pass filtering removes low-frequency background variations while preserving
    high-frequency features like fiducial spots. This preprocessing step improves
    the contrast of fiducial markers against cellular background.

    Args:
        image: Input microscopy image to be filtered
        cutoff: Cutoff frequency ratio (0-1). Lower values remove more background
        squared_butterworth: Whether to use squared Butterworth response for sharper cutoff
        order: Filter order - higher values create sharper frequency transitions
        npad: Number of pixels to pad edges (0 = automatic padding)

    Returns:
        Filtered image with enhanced spot contrast, clipped to non-negative values

    Scientific Context:
        Fiducial spots appear as bright, localized features that need to be distinguished
        from cellular autofluorescence and imaging artifacts. High-pass filtering
        effectively removes slowly varying background while preserving the sharp
        intensity gradients characteristic of diffraction-limited spots.
    """
    res: np.ndarray = filters.butterworth(
        image,
        cutoff_frequency_ratio=cutoff,
        order=order,
        high_pass=True,
        squared_butterworth=squared_butterworth,
        npad=npad,
    )  # type: ignore
    return np.clip(res, 0, None)


def clahe(img: np.ndarray, clip_limit: float = 0.01, bins: int = 200) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve local contrast.

    CLAHE enhances local contrast by applying histogram equalization within small regions
    of the image while preventing over-amplification of noise through clipping.
    This is particularly useful for images with varying illumination conditions.

    Args:
        img: Input microscopy image
        clip_limit: Clipping limit for contrast enhancement (0-1). Higher values allow more contrast
        bins: Number of histogram bins for equalization (unused in current implementation)

    Returns:
        Contrast-enhanced image with improved local feature visibility

    Scientific Context:
        Microscopy images often suffer from uneven illumination due to optical limitations.
        CLAHE improves the visibility of fiducial spots in regions with poor contrast
        without over-amplifying noise in already well-contrasted areas.
    """
    return exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=bins)


def find_spots(
    img: np.ndarray[np.uint16, Any],
    threshold_sigma: float,
    fwhm: float,
    minimum_spots: int = 6,
) -> pl.DataFrame:
    """Detect fiducial spots using Laplacian of Gaussian filtering and peak detection.

    This function identifies sub-pixel accurate spot locations by applying a Laplacian
    of Gaussian (LoG) filter followed by local maxima detection. The LoG filter is
    particularly effective for detecting blob-like features with known approximate size.

    Args:
        img: Input microscopy image containing fiducial spots
        threshold_sigma: Detection threshold in standard deviations above median.
                        Higher values detect only brighter spots, lower values may include noise
        fwhm: Full Width at Half Maximum of expected spots in pixels. Should match the
              point spread function of the imaging system (~2-4 pixels for typical setups)
        minimum_spots: Minimum number of spots required for successful detection.
                      Raises NotEnoughSpots if fewer spots are found

    Returns:
        DataFrame with detected spots sorted by brightness (mag column), containing:
        - xcentroid, ycentroid: Sub-pixel spot coordinates
        - mag: Spot intensity/magnitude
        - Additional photometry measurements

    Raises:
        NotEnoughSpots: If fewer than minimum_spots are detected
        TooManySpots: If an excessive number of spots suggests noise contamination

    Scientific Context:
        Fiducial spots are fluorescent beads embedded in tissue that serve as reference
        points for image registration. Accurate sub-pixel localization is critical for
        achieving nanometer-scale registration precision required in super-resolution
        microscopy and spatial transcriptomics applications.

        The LoG filter approximates the appearance of diffraction-limited spots and
        provides scale-invariant detection when the FWHM parameter matches the actual
        spot size. The sigma parameter controls detection sensitivity - too low includes
        noise, too high misses dim legitimate spots.
    """
    img = img.squeeze()
    if img.ndim != 2:
        raise ValueError(
            f"Reference image must be 2D for DAOStarFinder (is {img.shape}). https://github.com/astropy/photutils/issues/1328"
        )
    if np.sum(img) == 0:
        raise NotEnoughSpots("Reference image has zero sum - no signal available.")

    # min_ = img.min()
    # normalized = clahe((img - min_) / (img.max() - min_))

    _mean, median, std = sigma_clipped_stats(img, sigma=threshold_sigma + 5)
    # You don't want need to subtract the mean here, the median is subtracted in the call three lines below.
    iraffind = DAOStarFinder(
        threshold=threshold_sigma * std, fwhm=fwhm, exclude_border=True, roundhi=0.5, roundlo=-0.5
    )
    try:
        df = pl.DataFrame(iraffind(img - median).to_pandas())
        # Filter out null mag values and non-finite centroids, sort by magnitude (brightest first)
        df = df.filter(
            pl.col("mag").is_not_null()
            & pl.col("xcentroid").is_finite()
            & pl.col("ycentroid").is_finite()
        ).sort("mag").with_row_index("idx")
    except AttributeError:
        df = pl.DataFrame()
    if len(df) < minimum_spots:
        raise NotEnoughSpots
    return df


def _normalize_for_fft(img: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-scale copy for FFT-based alignment."""
    arr = np.asarray(img, dtype=np.float32)
    centered = arr - float(np.median(arr))
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-6:
        max_abs = float(np.max(np.abs(centered)))
        scale = max_abs if max_abs > 0 else 1.0
    return centered / scale


def phase_shift(ref: np.ndarray, img: np.ndarray, precision: int = 2) -> np.ndarray:
    """Calculate sub-pixel image translation using phase cross-correlation.

    Args:
        ref: Reference image (typically fiducial channel from reference round)
        img: Target image to be aligned to reference
        precision: Decimal precision for sub-pixel accuracy. precision=2 gives 0.01 pixel accuracy

    Returns:
        Translation vector [dy, dx] in pixels to align img to ref

    Scientific Context:
        Phase cross-correlation is robust to intensity variations and noise, making it
        ideal for registering fiducial images across imaging rounds. Unlike feature-based
        methods, it works directly on pixel intensities and can detect translations
        even when individual fiducial spots are not clearly visible.

        The upsample_factor parameter determines sub-pixel precision: 100 (10²) gives
        centipixel accuracy, sufficient for most microscopy applications requiring
        nanometer-scale registration precision.
    """
    ref_norm = _normalize_for_fft(ref)
    img_norm = _normalize_for_fft(img)
    return phase_cross_correlation(ref_norm, img_norm, upsample_factor=int(10**precision))[0]


def itk_shift(
    ref: np.ndarray,
    img: np.ndarray,
    max_shift: float = 20.0,
) -> np.ndarray:
    """Calculate image translation using SimpleITK 1+1 evolutionary optimizer (2D translation).

    Uses a correlation metric and the OnePlusOne evolutionary strategy over a small
    multi-resolution pyramid. The optimization is centered on the initial transform
    (zero shift). The final shift is clipped to ``[-max_shift, max_shift]`` per axis
    to guard against implausibly large drifts.

    Args:
        ref: Reference image.
        img: Target image to be aligned to reference.
        max_shift: Maximum absolute translation (in pixels, assuming unit spacing)
            allowed along each axis. Results exceeding this are clipped.

    Returns:
        Translation vector [dy, dx] in pixels to align img to ref.
    """
    import SimpleITK as sitk

    fixed_sitk = sitk.GetImageFromArray(ref.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(img.astype(np.float32))

    registration = sitk.ImageRegistrationMethod()

    # Same-modality metric and linear interpolation
    registration.SetMetricAsMattesMutualInformation()
    registration.SetInterpolator(sitk.sitkLinear)

    # Multi-resolution pyramid for robustness
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([4, 2, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    # Pure 2D translation
    initial_transform = sitk.TranslationTransform(2)
    registration.SetInitialTransform(initial_transform, inPlace=True)

    # 1+1 evolutionary optimizer; scales derived from physical shifts
    registration.SetOptimizerAsOnePlusOneEvolutionary(
        numberOfIterations=200,
        epsilon=1.5e-4,
        initialRadius=1.01,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    final_transform = registration.Execute(fixed_sitk, moving_sitk)
    params = final_transform.GetParameters()

    # Debug: log optimizer diagnostics and final shift
    stop_reason = registration.GetOptimizerStopConditionDescription()
    n_iters = int(registration.GetOptimizerIteration())
    logger.debug(f"ITK optimizer stop: {stop_reason}")
    logger.debug(f"ITK iterations: {n_iters}")
    logger.debug(f"ITK metric: {registration.GetMetricValue()}")
    logger.debug(f"ITK final translation (tx, ty) = ({params[0]:.4f}, {params[1]:.4f})")

    # SITK returns [tx, ty], convert to [dy, dx] = [ty, tx]
    shift = np.array([params[1], params[0]], dtype=float)

    if np.any(np.abs(shift) > max_shift):
        logger.warning(
            f"ITK shift {shift} exceeds max_shift={max_shift:.1f}px; clipping to bounds."
        )
        shift = np.clip(shift, -max_shift, max_shift)

    logger.debug(f"ITK final shift [dy, dx] = ({shift[0]:.4f}, {shift[1]:.4f})")
    return shift, n_iters


def shifts_from_anchor_roi(
    roi_path: Path, reference: str, ordered_keys: list[str]
) -> dict[str, np.ndarray]:
    """Calculate shifts from ImageJ ROI anchor points using channel position.

    Points are matched to rounds by their channel position (c_position) in the ROI.
    Channel indices correspond to the sorted round names in the fids TIFF.
    Points are matched by nearest-neighbor - user doesn't need to mark in same order.

    Args:
        roi_path: Path to ImageJ RoiSet.zip or .roi file
        reference: Name of the reference round
        ordered_keys: Sorted list of round names (channel order in fids TIFF)

    Returns:
        Dict mapping round name to [dx, dy] shift in pixels
    """
    from collections import defaultdict

    import roifile

    rois = roifile.roiread(roi_path)
    if not isinstance(rois, list):
        rois = [rois]

    # Group points by channel (c_position is 1-indexed in ImageJ)
    points_by_channel: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for roi in rois:
        if roi.roitype != roifile.ROI_TYPE.POINT:
            continue
        # c_position is 1-indexed, convert to 0-indexed
        channel = roi.c_position - 1 if roi.c_position > 0 else 0
        for coord in roi.coordinates():
            points_by_channel[channel].append((coord[0], coord[1]))

    # Map channel indices to round names
    points_by_round: dict[str, list[tuple[float, float]]] = {}
    for channel, points in points_by_channel.items():
        if channel < len(ordered_keys):
            points_by_round[ordered_keys[channel]] = points
        else:
            logger.warning(f"Channel {channel} exceeds number of rounds ({len(ordered_keys)})")

    if reference not in points_by_round:
        raise ValueError(f"No anchor points found for reference round '{reference}'")

    ref_points = np.array(points_by_round[reference])
    logger.info(f"Loaded {len(ref_points)} anchor points for reference '{reference}'")

    shifts: dict[str, np.ndarray] = {reference: np.array([0.0, 0.0])}

    for round_name in ordered_keys:
        if round_name == reference:
            continue
        if round_name not in points_by_round:
            logger.warning(f"No anchor points found for round '{round_name}', using zero shift")
            shifts[round_name] = np.array([0.0, 0.0])
            continue

        round_points = np.array(points_by_round[round_name])

        if len(round_points) != len(ref_points):
            raise ValueError(
                f"Round '{round_name}' has {len(round_points)} points but reference has {len(ref_points)}"
            )

        # Nearest-neighbor matching (permutation invariant)
        tree = cKDTree(round_points)
        distances, indices = tree.query(ref_points)
        matched_round_points = round_points[indices]

        # Shift = ref - matched_round (to move round to align with ref)
        # Returns [dx, dy] where dx is column shift, dy is row shift
        shift = np.median(ref_points - matched_round_points, axis=0)
        shifts[round_name] = shift
        logger.debug(f"Anchor shift for {round_name}: {shift}, mean distance: {distances.mean():.2f}")

    return shifts


def background(
    img: np.ndarray, box_size: tuple[int, int] | None = None, sigma_clip: float = 2.0
) -> np.ndarray:
    """Estimate spatially varying background using 2D background fitting.

    This function creates a smooth background model by dividing the image into boxes,
    calculating the median background level in each box while rejecting outliers
    (bright spots), then interpolating between boxes to create a full-frame background map.

    Args:
        img: Input microscopy image for background estimation
        box_size: Size of background estimation boxes (y, x) in pixels.
                 Larger boxes = smoother background, smaller boxes = more local adaptation
                 If None, automatically determines size based on image dimensions
        sigma_clip: Standard deviation threshold for outlier rejection.
                   Higher values include more pixels in background estimation

    Returns:
        2D background image matching input dimensions

    Scientific Context:
        The median estimator is robust to the presence of bright fiducial spots, which
        are treated as outliers and excluded from background calculation. Sigma clipping
        further improves robustness by rejecting pixels that deviate significantly from
        the local median, ensuring the background model represents true background signal.
    """
    if box_size is None:
        # Auto-determine box size based on image dimensions
        # Use boxes that are ~10% of the image size, but at least 10x10
        box_size = (max(10, img.shape[0] // 10), max(10, img.shape[1] // 10))

    sigma_clip_obj = SigmaClip(sigma=sigma_clip)
    bkg = Background2D(
        img, box_size, filter_size=(3, 3), sigma_clip=sigma_clip_obj, bkg_estimator=MedianBackground()
    )
    return bkg.background


def _calculate_drift(
    ref_kd: cKDTree,
    ref_points: pl.DataFrame,
    target_points: pl.DataFrame,
    *,
    initial_drift: np.ndarray | None = None,
    use_brightest: int = 0,
    offset_brightest: int = 0,
    # subtract_background: bool = False,
    plot: bool = False,
    precision: int = 2,
    warning_spots_threshold: int = 400,
    min_spots_for_mode: int = 100,
    max_drift_threshold: float = 40.0,
    bin_size: float = 0.5,
) -> np.ndarray:
    """Calculate drift between reference and target fiducial spots using nearest neighbor matching.

    This function matches detected fiducial spots between reference and target images
    to estimate the global translation drift. It uses k-d tree nearest neighbor search
    for efficient matching, followed by robust drift estimation using mode calculation
    to reject outliers from incorrect matches.

    Args:
        ref_kd: K-d tree built from reference spot coordinates for fast neighbor search
        ref_points: DataFrame of reference fiducial spots with xcentroid, ycentroid columns
        target_points: DataFrame of target fiducial spots to be matched against reference
        initial_drift: Prior estimate of drift to improve matching (typically from previous iteration)
        use_brightest: If >0, use only the N brightest spots for more robust matching
        offset_brightest: Skip the first N brightest spots before applying use_brightest (pagination-style)
        plot: Whether to generate diagnostic histogram plots of drift distributions
        precision: Decimal precision for drift calculation rounding
        warning_spots_threshold: Warn if more spots detected (may indicate noise contamination)
        min_spots_for_mode: Minimum matched spots required for mode-based drift estimation
        max_drift_threshold: Maximum allowed drift magnitude before flagging as suspicious
        bin_size: Bin size for histogram-based mode calculation

    Returns:
        Drift vector [dy, dx] in pixels, rounded to specified precision

    Raises:
        ValueError: If no drift is found or reference image was passed as target

    Note:
        Coordinate ordering follows scipy convention (z, y, x) to match image array indexing.
    """

    if len(target_points) > warning_spots_threshold:
        logger.warning(
            f"WARNING: a lot ({len(target_points)}) of fiducials found. "
            "This may be noise and is slow. "
            "Please reduce FWHM or increase threshold."
        )

    cols = ["xcentroid", "ycentroid"]

    if use_brightest:
        # Sort ascending: most negative (brightest) first
        if offset_brightest < 0:
            raise ValueError("offset_brightest must be >= 0")
        target_points = target_points.sort("mag").slice(offset_brightest, use_brightest)

    # points = moving[cols].to_numpy()
    if initial_drift is None:
        initial_drift = np.zeros(2)

    target_points = target_points.with_columns(
        xcentroid=pl.col("xcentroid") + initial_drift[0],
        ycentroid=pl.col("ycentroid") + initial_drift[1],
    )

    target_points = target_points.filter(
        pl.col("xcentroid").is_finite() & pl.col("ycentroid").is_finite()
    )
    if target_points.is_empty():
        raise NotEnoughSpots("Empty target points (no finite spots) available for drift calculation.")

    dist, idxs = ref_kd.query(target_points[cols].to_numpy(), workers=2)
    mapping = pl.concat(
        [pl.DataFrame(dict(fixed_idx=np.array(idxs, dtype=np.uint32), dist=dist)), target_points],
        how="horizontal",
    )

    # Remove pairs with distance > 50px and duplicate mappings (priority on closest)
    finite = mapping.filter(pl.col("dist") <= 50.0).sort("dist").unique("fixed_idx", keep="first")
    joined = finite.join(
        ref_points[["idx", *cols]], left_on="fixed_idx", right_on="idx", how="left", suffix="_fixed"
    ).with_columns(
        dx=pl.col("xcentroid_fixed") - pl.col("xcentroid"),
        dy=pl.col("ycentroid_fixed") - pl.col("ycentroid"),
    )
    if joined.is_empty():
        raise NotEnoughSpots("No valid spot matches found for drift calculation.")

    def mode(data: np.ndarray):
        bins = np.arange(min(data), max(data) + bin_size, bin_size)
        bin_indices = np.digitize(data, bins)
        bin_counts = np.bincount(bin_indices)
        i = np.argwhere(bin_counts == np.max(bin_counts)).flatten()[0]
        if i == 0 or i == len(bins) - 1:
            return np.median(data)
        return (bins[i] + bins[i + 1]) / 2

    if plot:
        _, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
        axs = axs.flatten()
        axs[0].hist(joined["dx"], bins=100)
        axs[1].hist(joined["dy"], bins=100)

    if np.allclose(initial_drift, np.zeros(2)) and len(joined) > min_spots_for_mode:
        if joined["dx"].sum() == 0 and joined["dy"].sum() == 0:
            raise ValueError("No drift found. Reference passed?")
        res = np.array([mode(joined["dx"]), mode(joined["dy"])])  # type: ignore
        if np.hypot(*res).sum() < max_drift_threshold:
            return np.round(res, precision)

    # if len(joined) < 8:
    #     # use weighted mean
    #     drift = joined.select(dx=pl.col("dx") * pl.col("flux"), dy=pl.col("dy") * pl.col("flux")).sum()
    #     drift = np.array(drift).squeeze() / joined["flux"].sum()
    #     res = initial_drift + drift
    # else:
    drifts = joined[["dx", "dy"]].to_numpy()
    # model: TranslationTransform | None = ransac(
    #     (
    #         joined[["xcentroid", "ycentroid"]].to_numpy(),
    #         joined[["xcentroid_fixed", "ycentroid_fixed"]].to_numpy(),
    #     ),
    #     TranslationTransform,
    #     min_samples=min(len(drifts), 30),
    #     residual_threshold=0.1,
    #     max_trials=200,
    # )[0]  # type: ignore

    # if model is not None:
    #     drift = model.translation
    #     print(f"drift: {drift}")
    # else:
    drift = np.median(drifts, axis=0)

    hypot = np.hypot(*drifts.T)
    cv = np.std(hypot) / np.mean(hypot)
    logger.debug(f"drift: {drift} CV: {cv:04f}.")
    res = initial_drift + drift

    return np.round(res, precision)


class NotEnoughSpots(Exception): ...


class TooManySpots(Exception): ...


class ResidualTooLarge(Exception): ...


class DriftTooLarge(Exception): ...


@dataclass(slots=True)
class FiducialAlignmentStats:
    """Per-image fiducial alignment diagnostics."""

    iterations: int
    final_threshold: float | None
    final_fwhm: float | None
    n_spots: int
    mode: str | None = None
    algorithm: str | None = None


def handle_exception(
    local_σ: float,
    local_fwhm: float,
    *,
    tried: set[tuple[float, float]],
    exc: Exception,
    threshold_step: float = 0.5,
    min_threshold_sigma: float = 2.0,
    min_fwhm: float = 4.0,
) -> tuple[float, float]:
    """Adjust detection parameters when fiducial spot detection fails.

    Args:
        local_σ: Current threshold sigma value
        local_fwhm: Current FWHM parameter
        tried: Previously attempted (sigma, fwhm) combinations
        exc: Exception that triggered adjustment
        threshold_step: Step size for parameter adjustments
        min_threshold_sigma: Minimum threshold before adjusting FWHM
        min_fwhm: Minimum FWHM before reducing

    Returns:
        Adjusted (sigma, fwhm) parameters
    """
    log_message = f"{exc.__class__.__name__}. σ threshold: {local_σ} FWHM: {local_fwhm}"
    if isinstance(exc, NotEnoughSpots):
        logger.debug(log_message)
        if local_σ > min_threshold_sigma:
            local_σ -= threshold_step
        else:
            local_fwhm += threshold_step
    elif isinstance(exc, TooManySpots):
        logger.warning(log_message)
        local_σ += threshold_step
        while (local_σ, local_fwhm) in tried:
            local_fwhm += threshold_step
            logger.warning(f"Trying with larger FWHM. σ threshold: {local_σ}, FWHM: {local_fwhm}")
    elif isinstance(exc, DriftTooLarge) or isinstance(exc, ResidualTooLarge):
        logger.warning(log_message + exc.args[0])
        local_σ += threshold_step
        if (local_σ, local_fwhm) in tried and local_fwhm >= min_fwhm:
            local_fwhm -= threshold_step
            logger.warning(f"Trying with smaller FWHM. σ threshold: {local_σ}, FWHM: {local_fwhm}")
        while (local_σ, local_fwhm) in tried:
            local_fwhm += threshold_step
            logger.warning(f"Trying with smaller FWHM. σ threshold: {local_σ}, FWHM: {local_fwhm}")

    else:
        logger.warning(log_message)
        raise exc
    return local_σ, local_fwhm


def individual_align_fiducial(
    ref: np.ndarray,
    *,
    subtract_background: bool = False,
    debug: bool = False,
    name: str = "",
    threshold_sigma: float = 3,
    threshold_residual: float = 0.3,
    use_brightest: int = 0,
    fwhm: float = 4,
    detailed_config: FiducialDetailedConfig | None = None,
):
    """Create function to align target images to reference fiducials.

    Detects fiducial spots in reference image and returns function that calculates
    drift for target images by matching their fiducials to reference spots.

    Args:
        ref: Reference image containing fiducial spots
        subtract_background: Apply background subtraction before detection
        debug: Enable debug logging and single-threaded execution
        name: Image name for logging
        threshold_sigma: Detection threshold in standard deviations
        threshold_residual: Maximum residual drift for convergence
        use_brightest: Use only N brightest spots (0 = use all)
        fwhm: Expected spot FWHM in pixels
        processing_config: Configuration for detection parameters

    Returns:
        Function that takes target image and returns (drift, residual) tuple
    """
    if detailed_config is None:
        detailed_config = FiducialDetailedConfig()

    if subtract_background:
        ref = ref - background(
            ref, detailed_config.background_box_size, detailed_config.background_sigma_clip
        )

    ref = np.clip(ref, np.percentile(ref, detailed_config.percentile_clip), None)

    _attempt = 0
    thr = threshold_sigma

    while _attempt < detailed_config.max_attempts:
        try:
            fixed = find_spots(ref, threshold_sigma=thr, fwhm=fwhm, minimum_spots=detailed_config.min_spots)
            if len(fixed) > detailed_config.max_spots:
                raise TooManySpots(
                    f"Too many spots ({len(fixed)} > {detailed_config.max_spots}) found on the reference image. Please increase threshold_sigma or reduce FWHM."
                )
        except NotEnoughSpots:
            logger.debug(
                "Not enough spots found on reference image. Trying to find spots with lower threshold."
            )
            thr -= detailed_config.threshold_step
            _attempt += 1
            continue
        except TooManySpots:
            logger.warning(
                "Too many spots found on reference image. Trying to find spots with lower threshold."
            )
            thr += detailed_config.threshold_step
            _attempt += 1
        else:
            # Find steepest slope
            fixed = fixed[: max(np.argmax(np.diff(fixed["mag"])), 10, len(fixed) // 4)]
            break
    else:
        raise NotEnoughSpots(
            f"Could not find spots on reference after {detailed_config.max_attempts} attempts."
        )

    logger.debug(f"{name}: {len(fixed)} peaks found on reference image.")
    kd = cKDTree(fixed[["xcentroid", "ycentroid"]])

    stats: dict[str, FiducialAlignmentStats] = {}
    stats[name] = FiducialAlignmentStats(
        iterations=0,
        final_threshold=float(thr),
        final_fwhm=float(fwhm),
        n_spots=int(len(fixed)),
        mode="spots",
        algorithm=None,
    )

    def inner(img: np.ndarray, *, limit: int = 4, bitname: str = "", local_σ: float = thr):
        # img = np.clip(img, np.percentile(ref, 50), None)
        if subtract_background:
            img = img - background(
                img, detailed_config.background_box_size, detailed_config.background_sigma_clip
            )

        _attempt = 0
        residual = np.inf
        drift = np.array([0, 0])
        local_fwhm = fwhm
        tried: set[tuple[float, float]] = set()
        iterations = 0

        # Iteratively reduce threshold_sigma until we get enough fiducials.
        while _attempt < detailed_config.max_drift_attempts:
            tried.add((local_σ, local_fwhm))
            try:
                moving = find_spots(
                    img,
                    threshold_sigma=local_σ,
                    fwhm=local_fwhm,
                    minimum_spots=max(detailed_config.min_spots, len(fixed) // 2),
                )
                moving = moving[: max(np.argmax(np.diff(moving["mag"])), 10, len(moving) // 4)]

                logger.debug(f"{bitname}: {len(moving)} peaks found on target image.")

                initial_drift = np.zeros(2)
                assert limit > 0

                # Actual loop drift loop.
                for n in range(limit):
                    # Can raise NotEnoughSpots
                    drift = _calculate_drift(
                        kd,
                        fixed,
                        moving,
                        initial_drift=initial_drift,
                        use_brightest=use_brightest,
                        offset_brightest=detailed_config.offset_brightest,
                        warning_spots_threshold=detailed_config.warning_spots_threshold,
                        min_spots_for_mode=detailed_config.min_spots_for_mode,
                        max_drift_threshold=detailed_config.max_drift_threshold,
                        bin_size=detailed_config.bin_size,
                    )
                    residual = np.hypot(*(drift - initial_drift))
                    iterations = n + 1
                    if n == 0:
                        logger.debug(f"{bitname} - attempt {n}: starting drift: {drift}.")
                    else:
                        logger.debug(
                            f"{bitname} - attempt {n}: new drift: {drift} Δ from last {residual:.2f}px."
                        )
                    if residual < threshold_residual:
                        break
                    initial_drift = drift

                if np.max(np.abs(drift)) > detailed_config.max_drift_threshold:
                    if detailed_config.allow_large_shifts:
                        logger.warning(
                            f"{bitname}: drift {np.hypot(*drift):.2f} exceeds threshold "
                            f"{detailed_config.max_drift_threshold}px but allow_large_shifts is enabled; accepting."
                        )
                    else:
                        local_σ += detailed_config.threshold_step
                        raise DriftTooLarge(
                            f"{bitname}: drift very large {np.hypot(*drift):.2f}."
                        )

                if residual > threshold_residual:
                    local_σ += detailed_config.threshold_step
                    raise ResidualTooLarge(
                        f"{bitname}: residual drift too large {residual=:2f}. Please refine parameters."
                    )
            except Exception as e:
                _attempt += 1
                local_σ, local_fwhm = handle_exception(
                    local_σ,
                    local_fwhm,
                    tried=tried,
                    exc=e,
                    threshold_step=detailed_config.threshold_step,
                    min_threshold_sigma=detailed_config.min_threshold_sigma,
                    min_fwhm=detailed_config.min_fwhm,
                )
            else:
                break
        else:
            logger.critical(f"Fiducial matching failed after {_attempt} attempts.")
            raise NotEnoughSpots

        key = bitname or name
        stats[key] = FiducialAlignmentStats(
            iterations=iterations,
            final_threshold=float(local_σ),
            final_fwhm=float(local_fwhm),
            n_spots=int(len(moving)),
            mode="spots",
            algorithm=None,
        )

        return drift, residual

    inner.stats = stats
    return inner


def align_phase(
    fids: dict[str, np.ndarray[Any, Any]], *, reference: str, threads: int = 4, debug: bool = False
):
    keys = list(fids.keys())
    for name in keys:
        if re.search(reference, name):
            ref = name
            break
    else:
        raise ValueError(f"Could not find reference {reference} in {keys}")

    del reference
    with ThreadPoolExecutor(threads if not debug else 1) as exc:
        futs: dict[str, Future] = {}
        for k, img in fids.items():
            if k == ref:
                continue
            futs[k] = exc.submit(phase_shift, fids[ref], img)

            if debug:
                # Force print in order
                logger.debug(f"{k}: {futs[k].result()}")

    return {k: v.result().astype(float) for k, v in futs.items()} | {ref: np.zeros(2)}


def _align_fiducials_internal(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    use_fft: bool = False,
    use_itk: bool = False,
    use_brightest: int = 0,
    threads: int = 4,
    overrides: dict[str, tuple[float, float]] | None = None,
    subtract_background: bool = False,
    debug: bool = False,
    max_iters: int = 4,
    threshold_sigma: float = 3,
    threshold_residual: float = 0.2,
    fwhm: float = 4,
    detailed_config: FiducialDetailedConfig | None = None,
) -> tuple[
    dict[str, np.ndarray[float, Any]],
    dict[str, np.ndarray[float, Any]],
    dict[str, FiducialAlignmentStats | None],
]:
    """Shared implementation for align_fiducials with optional diagnostics."""
    if detailed_config is None:
        detailed_config = FiducialDetailedConfig()

    keys = list(fids.keys())
    for name in keys:
        if re.search(reference, name):
            ref = name
            break
    else:
        raise ValueError(f"Could not find reference {reference} in {keys}")

    del reference
    # returns drift, residual
    corr = individual_align_fiducial(
        fids[ref],
        subtract_background=subtract_background,
        debug=debug,
        name=ref,
        threshold_sigma=threshold_sigma,
        threshold_residual=threshold_residual,
        fwhm=fwhm,
        use_brightest=use_brightest,
        detailed_config=detailed_config,
    )

    # ITK max_shift: use larger limit when allow_large_shifts is enabled
    itk_max_shift = 500.0 if detailed_config.allow_large_shifts else detailed_config.max_drift_threshold

    mode_map: dict[str, str] = {}
    itk_iterations: dict[str, int] = {}

    def _itk_wrapper(img: np.ndarray, bitname: str) -> tuple[np.ndarray, float]:
        """Pure ITK-based registration wrapper for thread pool."""
        shift_vec, n_iters = itk_shift(fids[ref], img, max_shift=itk_max_shift)
        itk_iterations[bitname] = n_iters
        # itk_shift returns [dy, dx], swap to [dx, dy] for consistency.
        return shift_vec[::-1], 0.0

    with ThreadPoolExecutor(threads if not debug else 1) as exc:
        futs: dict[str, Future] = {}
        for k, img in fids.items():
            if k == ref or (overrides is not None and k in overrides):
                continue
            if use_fft:
                mode_map[k] = "fft"
                # phase_shift returns [dy, dx] (row, col) from phase_cross_correlation.
                # Swap to [dx, dy] to match spot-based convention used elsewhere.
                futs[k] = exc.submit(lambda x: (phase_shift(fids[ref], x)[::-1], 0.0), img)
            elif use_itk:
                mode_map[k] = "itk"
                futs[k] = exc.submit(_itk_wrapper, img, k)
            else:
                mode_map[k] = "spots"
                futs[k] = exc.submit(corr, img, bitname=k, limit=max_iters)

            if debug:
                futs[k].result()

        for fut in as_completed(futs.values()):
            fut.result()

    drift_dict: dict[str, np.ndarray[float, Any]] = (
        {k: v.result()[0] for k, v in futs.items()}
        | {ref: np.zeros(2)}
        | ({k: np.array(v) for k, v in overrides.items()} if overrides else {})
    )
    residual_dict: dict[str, np.ndarray[float, Any]] = {k: v.result()[1] for k, v in futs.items()} | {ref: 0.0}

    raw_stats = getattr(corr, "stats", {}) if isinstance(getattr(corr, "stats", None), dict) else {}
    stats_dict: dict[str, FiducialAlignmentStats | None] = {}
    for name in fids.keys():
        base_stat = raw_stats.get(name) if isinstance(raw_stats, dict) else None
        stat: FiducialAlignmentStats | None
        if isinstance(base_stat, FiducialAlignmentStats):
            stat = base_stat
        else:
            stat = None

        mode = mode_map.get(name)
        if mode == "fft":
            stat = FiducialAlignmentStats(
                iterations=0,
                final_threshold=None,
                final_fwhm=None,
                n_spots=0,
                mode="fft",
                algorithm="phase_correlation",
            )
        elif mode == "itk":
            stat = FiducialAlignmentStats(
                iterations=itk_iterations.get(name, 0),
                final_threshold=None,
                final_fwhm=None,
                n_spots=0,
                mode="itk",
                algorithm="OnePlusOneEvo",
            )
        elif stat is not None:
            # Spot-based stats from the fiducial matcher
            stat.mode = "spots"

        stats_dict[name] = stat

    return drift_dict, residual_dict, stats_dict


def align_fiducials(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    use_fft: bool = False,
    use_itk: bool = False,
    use_brightest: int = 0,
    threads: int = 4,
    overrides: dict[str, tuple[float, float]] | None = None,
    subtract_background: bool = False,
    debug: bool = False,
    max_iters: int = 4,
    threshold_sigma: float = 3,
    threshold_residual: float = 0.2,
    fwhm: float = 4,
    detailed_config: FiducialDetailedConfig | None = None,
) -> tuple[dict[str, np.ndarray[float, Any]], dict[str, np.ndarray[float, Any]]]:
    """Calculate drift vectors for all images relative to reference using fiducial alignment.

    This public entry point preserves the original (shifts, residuals) return contract.
    """
    shifts, residuals, _ = _align_fiducials_internal(
        fids,
        reference=reference,
        use_fft=use_fft,
        use_itk=use_itk,
        use_brightest=use_brightest,
        threads=threads,
        overrides=overrides,
        subtract_background=subtract_background,
        debug=debug,
        max_iters=max_iters,
        threshold_sigma=threshold_sigma,
        threshold_residual=threshold_residual,
        fwhm=fwhm,
        detailed_config=detailed_config,
    )
    return shifts, residuals


def align_fiducials_with_stats(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    use_fft: bool = False,
    use_itk: bool = False,
    use_brightest: int = 0,
    threads: int = 4,
    overrides: dict[str, tuple[float, float]] | None = None,
    subtract_background: bool = False,
    debug: bool = False,
    max_iters: int = 4,
    threshold_sigma: float = 3,
    threshold_residual: float = 0.2,
    fwhm: float = 4,
    detailed_config: FiducialDetailedConfig | None = None,
) -> tuple[
    dict[str, np.ndarray[float, Any]],
    dict[str, np.ndarray[float, Any]],
    dict[str, FiducialAlignmentStats | None],
]:
    """Calculate drift vectors and expose per-image alignment diagnostics."""
    return _align_fiducials_internal(
        fids,
        reference=reference,
        use_fft=use_fft,
        use_itk=use_itk,
        use_brightest=use_brightest,
        threads=threads,
        overrides=overrides,
        subtract_background=subtract_background,
        debug=debug,
        max_iters=max_iters,
        threshold_sigma=threshold_sigma,
        threshold_residual=threshold_residual,
        fwhm=fwhm,
        detailed_config=detailed_config,
    )


def plot_alignment(fids: dict[str, np.ndarray[float, Any]], sl: slice = np.s_[500:600]):
    keys = list(fids.keys())
    ns = (len(fids) // 3) + 1
    fig, axs = plt.subplots(ncols=1, nrows=ns, figsize=(3, 9), dpi=200)
    axs = axs.flatten()

    combi = np.stack([fids[k] for k in keys]).astype(np.float64)
    combi /= np.percentile(combi, 99, axis=(1, 2))[:, None, None]
    combi = np.clip(combi, 0, 1)

    for ax, i in zip(axs, range(0, len(fids), 3)):
        if len(fids) - i < 3:
            i = len(fids) - 3
        ax.imshow(np.moveaxis(combi[i : i + 3][:, sl, sl], 0, 2))


class Shift(BaseModel):
    shifts: list[float] | tuple[float, float]
    corr: float
    residual: float
    iterations: int | None = None
    final_threshold: float | None = None
    final_fwhm: float | None = None
    n_spots: int | None = None
    mode: str | None = None
    algorithm: str | None = None


Shifts = TypeAdapter(dict[str, Shift])


class MultipleShifts(BaseModel):
    shift: Shift
    config: RegisterConfig
