from pathlib import Path
from typing import Any

import itk
import numpy as np
import SimpleITK as sitk
from loguru import logger
from matplotlib.axes import Axes


class Affine:
    """Chromatic aberration correction using pre-computed affine transforms.

    Corrects optical aberrations across wavelengths by applying channel-specific
    2D affine transformations to align all channels to a reference wavelength.
    Also applies inter-round registration shifts.
    """

    IGNORE = {"405", "488", "560"}

    def __init__(
        self,
        *,
        ref_img: np.ndarray[np.float32, Any] | None = None,
        As: dict[str, np.ndarray[np.float64, Any]],
        ats: dict[str, np.ndarray[np.float64, Any]],
        # cg: dict[str, np.ndarray[np.float64, Any]],
        ref: str = "560",
    ):
        """Initialize chromatic correction with pre-computed transforms.

        Args:
            ref_img: Reference image for resampling geometry
            As: Channel-specific 2x2 affine matrices
            ats: Channel-specific translation vectors
            ref: Reference channel name (default: "560")
        """
        self.As: dict[str, np.ndarray[np.float64, Any]] = As
        self.ats: dict[str, np.ndarray[np.float64, Any]] = ats
        # self.cg: dict[str, np.ndarray[np.float64, Any]] = cg
        self.ref_channel = ref
        self._ref_image = (
            sitk.Cast(sitk.GetImageFromArray(ref_img), sitk.sitkFloat32) if ref_img is not None else None
        )

    @property
    def ref_image(self):
        """Get reference image in SimpleITK format."""
        return self._ref_image

    @ref_image.setter
    def ref_image(self, img: np.ndarray[np.float32, Any]):
        """Set reference image, converting to SimpleITK format."""
        self._ref_image = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)

    def __call__(
        self,
        img: np.ndarray[np.float32, Any],
        *,
        channel: str,
        shiftpx: np.ndarray | None = None,
        debug: bool = False,
    ):
        """Chromatic and shift correction. Repeated 2D operations of zyx image.

        Args:
            img: Single-bit zyx image.
            channel: channel name. Must be 405, 488, 560, 650, 750.
            shiftpx: Vector of shift in pixels.
            ref: Reference image in sitk format.

        Raises:
            ValueError: Invalid shift vector dimension.

        Returns:
            Corrected image.
        """
        if shiftpx is None:
            shiftpx = np.zeros(2, dtype=np.float64)
        if len(shiftpx) != 2:
            raise ValueError

        # Between imaging-round shifts
        translate = sitk.TranslationTransform(3)
        translate.SetParameters((float(shiftpx[0]), float(shiftpx[1]), 0.0))

        # Don't do chromatic shift if channel is reference
        if channel in self.IGNORE or channel == self.ref_channel:
            return self._st(img, translate)

        if channel not in self.As or channel not in self.ats:
            raise ValueError(f"Chromatic: corrections for channel {channel} not found.")

        # Scale
        affine = sitk.AffineTransform(3)
        matrix = self.As[channel]
        affine.SetMatrix(matrix.flatten())

        # Translate
        translation = self.ats[channel]
        affine.SetTranslation(translation)
        affine.SetCenter([1023.5 + shiftpx[0], 1023.5 + shiftpx[1], 0])

        if debug:
            logger.debug(f"{channel}: affine: {matrix}")
            logger.debug(f"{channel}: translation: {translation}")

        composite = sitk.CompositeTransform(3)
        composite.AddTransform(translate)
        composite.AddTransform(affine)

        return self._st(img, composite)

    def _st(self, img: np.ndarray[np.float32, Any], transform: sitk.Transform):
        """Execute a sitk transform on an image.

        Args:
            ref: Reference image in sitk format.
            img: Image to transform. Must be in float32 format.
            transform: sitk transform to apply.

        Returns:
            Transformed image.
        """
        if self._ref_image is None:
            raise ValueError("Reference image not set. Please set at .ref_image.")
        image = sitk.GetImageFromArray(img)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self._ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(transform)

        return sitk.GetArrayFromImage(resampler.Execute(image))


def overlay(
    ref: np.ndarray[np.float32, Any],
    img: np.ndarray[np.float32, Any],
    img2: np.ndarray[np.float32, Any] | None = None,
    *,
    sl: slice | tuple[slice, ...] | None = np.s_[1600:1800, 1600:1800],
    percentile: tuple[float, float] = (1, 100),
    ax: Axes | None = None,
    title: str | None = None,
):
    """Create RGB overlay visualization comparing reference and corrected images.

    Displays reference in green, target in red, and optional third image in blue
    to visualize registration quality and chromatic correction effectiveness.

    Args:
        ref: Reference image (displayed in green)
        img: Target image (displayed in red)
        img2: Optional third image (displayed in blue)
        sl: Image region to display (default: center 200x200 crop)
        percentile: Intensity normalization range (min, max percentiles)
        ax: Matplotlib axis for plotting (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axis with RGB overlay plot
    """
    import matplotlib.pyplot as plt

    if sl is not None:
        img = img[sl]
        ref = ref[sl]
        img2 = img2[sl] if img2 is not None else None

    # Ensure images are normalized to [0, 1] range
    perc = percentile

    def norm(img: np.ndarray[np.float32, Any]):
        return np.clip(
            (img - np.percentile(img, perc[0])) / (np.percentile(img, perc[1]) - np.percentile(img, perc[0])),
            0,
            1,
        )

    img_norm = norm(img)
    ref_norm = norm(ref)
    if img2 is not None:
        img2_norm = norm(img2)

    # Create RGB image
    rgb_image = np.zeros((*img.shape, 3), dtype=np.float32)
    rgb_image[..., 0] = img_norm  # Red channel
    rgb_image[..., 1] = ref_norm  # Green channel
    if img2 is not None:
        rgb_image[..., 2] = img2_norm  # Blue channel

    # Create figure and axis
    if not ax:
        _, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    assert ax
    ax.set_facecolor("black")

    ax.imshow(rgb_image)
    ax.axis("off")
    ax.set_title(title if title else "Image Comparison (green: ref, red: img)", color="white")

    plt.tight_layout()
    return ax


class FitAffine:
    """Compute affine transform parameters using elastix registration.

    Fits 2D affine transformations between reference and target images to
    characterize chromatic aberration. Uses elastix library for robust
    optimization-based registration.
    """

    def __init__(self):
        """Initialize elastix parameter object with affine registration settings."""
        self.parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = self.parameter_object.GetDefaultParameterMap("affine")
        default_rigid_parameter_map["AutomaticScalesEstimation"] = ["true"]
        self.parameter_object.AddParameterMap(default_rigid_parameter_map)

    def fit(self, ref: np.ndarray, target: np.ndarray):
        """Fit affine transform from reference to target image.

        Args:
            ref: Reference image (fixed)
            target: Target image to be aligned

        Returns:
            Tuple of (affine_matrices, translation_vectors) as lists
        """
        # As is a 2x2 matrix
        # ts is a 2x1 matrix (translation)
        As = []
        ts = []

        result_image, self.parameter_object = itk.elastix_registration_method(
            ref.astype(np.float32),
            target.astype(np.float32),
            parameter_object=self.parameter_object,
            log_to_console=True,
        )

        params = np.array(
            list(
                map(
                    float,
                    self.parameter_object.GetParameter(0, "TransformParameters"),
                )
            )
        )
        As.append(params[:4].reshape(2, 2))
        ts.append(params[-2:])
        return As, ts

    @staticmethod
    def write(
        path: Path | str,
        As: list[np.ndarray[np.float64, Any]],
        ts: list[np.ndarray[np.float64, Any]],
    ):
        """Save median affine parameters to text file.

        Computes median transformation from multiple fits and saves as
        single line with matrix elements followed by translation vector.

        Args:
            path: Output file path
            As: List of 2x2 affine matrices
            ts: List of 2D translation vectors
        """
        affined = np.array([
            *np.median(np.stack(As), axis=0).flatten(),
            *np.median(np.stack(ts), axis=0),
        ])
        Path(path).write_text("\n".join(map(str, affined.flatten())))
