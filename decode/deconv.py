# %%
from functools import cache
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


def high_pass_filter(img: npt.NDArray[Any], σ: float = 2.0, dtype: npt.DTypeLike = np.float32) -> npt.NDArray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    img = img.astype(dtype)
    window_size = int(2 * np.ceil(2 * σ) + 1)
    lowpass = cv2.GaussianBlur(img, (window_size, window_size), σ, borderType=cv2.BORDER_REPLICATE)
    gauss_highpass = img - lowpass
    gauss_highpass[lowpass > img] = 0
    return gauss_highpass


def _matlab_gauss2D(
    shape: int = 3, σ: float = 0.5, dtype: npt.DTypeLike = np.float32
) -> np.ndarray[np.float64, Any]:
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    x = np.linspace(-(shape // 2), shape // 2, shape)
    x, y = np.meshgrid(x, x)
    h = multivariate_normal([0, 0], σ**2 * np.eye(2)).pdf(np.dstack((x, y)))
    h = h.astype(dtype)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= h.sum()  # normalize
    return h


@cache
def _calculate_projectors(window_size: int, σ_G: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculate forward and backward projectors as described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function

    Returns:
        A list containing the forward and backward projectors to use for
        Lucy-Richardson deconvolution.
    """
    pf = _matlab_gauss2D(shape=window_size, σ=σ_G)
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    # These values are from Guo et al.
    α = β = 0.001
    n = 8

    # This is the cut-off frequency.
    kc = 1.0 / (0.5 * 2.355 * σ_G)

    # FFT frequencies
    kv = np.fft.fftfreq(pfFFT.shape[0])
    kk = np.hypot(*np.meshgrid(kv, kv))

    # Wiener filter
    bWiener = pfFFT / (np.abs(pfFFT) ** 2 + α)
    # Buttersworth filter
    eps = np.sqrt(1.0 / (β**2) - 1)
    bBWorth = 1.0 / np.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth
    # back projector.
    pb = np.real(np.fft.ifft2(pbFFT))

    return pf, pb


EPS = 1e-6
I_MAX = 2**16 - 1


def deconvolve_lucyrichardson_guo(
    img: np.ndarray[np.float32, Any] | np.ndarray[np.float64, Any],
    window_size: int = 9,
    σG: float = 1.4,
    iters: int = 4,
) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:
    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.
    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform
    Returns:
        the deconvolved image
    """

    if img.dtype not in (np.float32, np.float64):
        raise ValueError("Image must be floating point")

    forward_projector, backward_projector = _calculate_projectors(window_size, σG)
    forward_projector = forward_projector.astype(img.dtype)
    backward_projector = backward_projector.astype(img.dtype)

    estimate = np.clip(img.copy(), EPS, None)

    for _ in range(iters):
        filtered_estimate = np.clip(
            cv2.filter2D(estimate, -1, forward_projector, borderType=cv2.BORDER_REPLICATE),
            EPS,
            I_MAX,
        )

        ratio = img / filtered_estimate
        # Correction
        estimate *= cv2.filter2D(ratio, -1, backward_projector, borderType=cv2.BORDER_REPLICATE)
        estimate = np.clip(estimate, EPS, I_MAX)

    return estimate


# %%
