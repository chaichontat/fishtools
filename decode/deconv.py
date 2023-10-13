from typing import Any

import cv2
import numpy as np


def _matlab_gauss2D(shape: tuple[int, int] = (3, 3), sigma: float = 0.5) -> np.ndarray[np.float64, Any]:
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def _calculate_projectors(windowSize: int, σ_G: float) -> tuple[np.ndarray, np.ndarray]:
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
    pf = _matlab_gauss2D(shape=(windowSize, windowSize), sigma=σ_G)
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    #
    # These values are from Guo et al.
    alpha = 0.001
    beta = 0.001
    n = 8

    # This is the cut-off frequency.
    kc = 1.0 / (0.5 * 2.355 * σ_G)

    # FFT frequencies
    kv = np.fft.fftfreq(pfFFT.shape[0])

    kx = np.zeros((kv.size, kv.size))
    for i in range(kv.size):
        kx[i, :] = np.copy(kv)

    ky = np.transpose(kx)
    kk = np.sqrt(kx * kx + ky * ky)

    # Wiener filter
    bWiener = pfFFT / (np.abs(pfFFT) * np.abs(pfFFT) + alpha)

    # Buttersworth filter
    eps = np.sqrt(1.0 / (beta * beta) - 1)

    kkSqr = kk * kk / (kc * kc)
    bBWorth = 1.0 / np.sqrt(1.0 + eps * eps * np.power(kkSqr, n))

    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth

    # back projector.
    pb = np.real(np.fft.ifft2(pbFFT))

    return pf, pb


def deconvolve_lucyrichardson_guo(
    image: np.ndarray, windowSize: int, sigmaG: float, iterationCount: int
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
    eps = 1.0e-6
    i_max = 2**16 - 1

    [pf, pb] = _calculate_projectors(windowSize, sigmaG)
    pf = pf.astype(image.dtype)
    pb = pb.astype(image.dtype)
    ek = np.clip(image.copy(), eps, None)

    for _ in range(iterationCount):
        ekf = cv2.filter2D(ek, -1, pf, borderType=cv2.BORDER_REPLICATE)
        ekf = np.clip(ekf, eps, i_max)

        ek = ek * cv2.filter2D(image / ekf, -1, pb, borderType=cv2.BORDER_REPLICATE)
        ek = np.clip(ek, eps, i_max)

    return ek
