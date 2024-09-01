# %%
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk
from loguru import logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Affine:
    def __init__(
        self,
        *,
        ref_img: np.ndarray[np.float32, Any] | None = None,
        As: dict[str, np.ndarray[np.float64, Any]],
        ats: dict[str, np.ndarray[np.float64, Any]],
        ref: str = "560",
    ):
        self.As: dict[str, np.ndarray[np.float64, Any]] = As
        self.ats: dict[str, np.ndarray[np.float64, Any]] = ats
        self.ref_channel = ref
        self._ref_image = (
            sitk.Cast(sitk.GetImageFromArray(ref_img), sitk.sitkFloat32) if ref_img is not None else None
        )

    @property
    def ref_image(self):
        return self._ref_image

    @ref_image.setter
    def ref_image(self, img: np.ndarray[np.float32, Any]):
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
        if channel == self.ref_channel:
            return self._st(img, translate)

        # Scale
        affine = sitk.AffineTransform(3)
        matrix = self.As.get(channel, np.zeros((3, 3), dtype=np.float64))
        affine.SetMatrix(matrix.flatten())

        # Translate
        translation = self.ats.get(channel, np.zeros(3, dtype=np.float64))
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


# %%

if __name__ == "__main__":
    from pathlib import Path

    import seaborn as sns
    from tifffile import imread

    sns.set_theme()

    img = imread("/mnt/archive/starmap/sagittal-calibration/4_4--cortexRegion/4_4-0000.tif")[:-1].reshape(
        -1, 2, 2048, 2048
    )

    DATA = Path("/home/chaichontat/fishtools/data")

    print(img.max(axis=(0, 2, 3)))
    # %%

    As = {}
    ats = {}
    for λ in ["650", "750"]:
        a_ = np.loadtxt(DATA / f"560to{λ}.txt")
        A = np.zeros((3, 3), dtype=np.float64)
        A[:2, :2] = a_[:4].reshape(2, 2)
        t = np.zeros(3, dtype=np.float64)
        t[:2] = a_[-2:]

        A[2] = [0, 0, 1]
        A[:, 2] = [0, 0, 1]
        t[2] = 0
        As[λ] = A
        ats[λ] = t

    # %%
    affine = Affine(As=As, ats=ats, ref_img=img[[0], 0])
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6), dpi=200, facecolor="black")

    img = imread("/disk/chaichontat/2024/sv101_ACS/2_10_18--noRNAse_big/2_10_18-0012.tif")[:-1].reshape(
        -1, 3, 2048, 2048
    )

    overlay(img[:, 1][5], img[:, 2][5], ax=axs[0], sl=np.s_[1400:1800, 1400:1800])
    overlay(
        img[[0], 1][0],
        affine(img[[0], 2], channel="750", shiftpx=np.array([0, 0.0]))[0],
        ax=axs[1],
        sl=np.s_[1400:1800, 1400:1800],
    )

    # %%
    img = imread("/disk/chaichontat/2024/sv101_ACS/registered/reg-0074.tif")
    # fig, axs = plt.subplots(ncols=2, figsize=(12, 6), dpi=200, facecolor="black")

    overlay(
        img[0, 1],
        img[0, 17],
        # affine(img[:, 24], channel="750", shiftpx=np.array([0, 0.0]))[0],
        # ax=axs[0],
        sl=np.s_[200:450, 0:1000],
        percentile=(1, 99.9),
        title="Green: bit2, Red: bit 18. No chromatic aberration.",
    )

    # %%

    fid = imread("/disk/chaichontat/2024/sv101_ACS/fids_shifted-0075.tif")
    # %%
    overlay(
        fid[1],
        fid[-2],
        # ax=axs[0],
        sl=np.s_[200:450, 800:1000],
        percentile=(1, 100),
        title="Fiducial from the same area. Green: bit2, Red: bit29",
    )
    # %%
    overlay(
        img[0, 1],
        img[0, 24],
        # affine(img[:, 24], channel="750", shiftpx=np.array([0, 0.0]))[0],
        # ax=axs[0],
        sl=np.s_[200:450, 800:1000],
        percentile=(1, 99.9),
        title="Green: bit2, Red: bit 29. Slight shift.",
    )
    # %%
