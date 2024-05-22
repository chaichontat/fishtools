# %%

from functools import cache
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
import cupy as cp
import numpy.typing as npt
from cupyx.scipy.ndimage import convolve as cconvolve
from scipy.ndimage import convolve
from scipy.stats import multivariate_normal
from tifffile import imread


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


# @cache
def _calculate_projectors(pf: np.ndarray, σ_G: float) -> tuple[np.ndarray, np.ndarray]:
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
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    # These values are from Guo et al.
    α = 0.05
    β = 1
    n = 10

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


def _calculate_projectors_3d(
    pf: cp.ndarray, σ_G: float, a: float = 0.001, b: float = 0.001, n: int = 8
) -> tuple[cp.ndarray, cp.ndarray]:
    """Calculate forward and backward projectors in 3D as described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        pf: the 3D point spread function
        σ_G: the standard deviation of the Gaussian point spread function

    Returns:
        A tuple containing the forward and backward projectors to use for
        Lucy-Richardson deconvolution in 3D.
    """
    pfFFT = cp.fft.fftn(pf)

    # Wiener-Butterworth back projector.
    # These values are from Guo et al.
    α = a
    β = b

    # This is the cut-off frequency.
    kc = 1.0 / (0.5 * 2.355 * σ_G)

    # FFT frequencies
    kz = cp.fft.fftfreq(pfFFT.shape[0])
    kw = cp.fft.fftfreq(pfFFT.shape[1])
    kk = cp.sqrt((cp.array(cp.meshgrid(kz, kw, kw, indexing="ij")) ** 2).sum())

    # Wiener filter
    bWiener = pfFFT / (cp.abs(pfFFT) ** 2 + α)
    # Butterworth filter
    eps = cp.sqrt(1.0 / (β**2) - 1)
    bBWorth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    # Wiener-Butterworth back projector
    pbFFT = bWiener * bBWorth
    # Back projector
    pb = cp.real(cp.fft.ifftn(pbFFT))

    return pf, pb


EPS = 1e-9
I_MAX = 2**16 - 1


# %%
def deconvolve_lucyrichardson_guo(
    img: np.ndarray[np.float32, Any] | np.ndarray[np.float64, Any],
    projectors: tuple[np.ndarray, np.ndarray],
    # psf: np.ndarray[np.float32, Any] | np.ndarray[np.float64, Any],
    σG: float = 1.7,
    iters: int = 1,
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
    forward_projector, backward_projector = projectors

    # if len(img.shape) == 2:
    #     forward_projector, backward_projector = _calculate_projectors(psf, σG)
    # elif len(img.shape) == 3:
    #     forward_projector, backward_projector = _calculate_projectors_3d(psf, σG)
    # else:
    #     raise ValueError("Image must be 2D or 3D")

    estimate = cp.array(np.clip(img.copy(), EPS, None))

    for _ in range(iters):
        filtered_estimate = cp.clip(
            cconvolve(cp.array(estimate), cp.array(forward_projector), mode="reflect"),
            EPS,
            I_MAX,
        )

        ratio = cp.array(img) / cp.array(filtered_estimate)
        # print(ratio)
        # Correction
        estimate *= cconvolve(cp.array(ratio), cp.array(backward_projector), mode="reflect")
        # print(estimate)
        # estimate = np.clip(estimate, EPS, I_MAX)

    return estimate


# %%
import tifffile

size = 11

psf = np.median(imread("data/psf_560.tif")[:10], axis=0)[1::5, 1:, 1:]

# psf = np.pad(psf, 12, mode="constant", constant_values=0)
psf /= psf.sum()

# # Get the center of the image
center = np.array(psf.shape) // 2
# # Get the window around the center
window = psf[
    :,
    center[1] - size // 2 : center[1] + size // 2 + 1,
    center[1] - size // 2 : center[1] + size // 2 + 1,
]
# # Normalize the window

window = window / window.sum()

# %%
from pathlib import Path

img = imread(path := Path("/fast2/alinatake2/dapi_polyA_28-0204.tif"))[:-1].reshape(-1, 3, 2048, 2048)
# path = Path(path)
# print(path.stem)
# img = imread(path)
# dfs = []
# %%


# %%
img = img.astype(np.float32)
out = np.empty_like(img)[:, 0]


# %%
import pickle

projectors = _calculate_projectors_3d(cp.array(window), 1.7, 0.04, 0.1, 10)
res = deconvolve_lucyrichardson_guo(cp.array(img[:, 2].astype(np.float32)), projectors, iters=1)

# %%
# def run(i: int):
#     out[i] = deconvolve_lucyrichardson_guo(img[i, 0], window, iters=1)


# for i in range(5):
#     run(i)
# from concurrent.futures import ThreadPoolExecutor

# with ThreadPoolExecutor(4) as executor:
#     executor.map(run, range(img.shape[0]))


# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")

axs[0].imshow(img[12, 2, 700:900, 0:300])
axs[1].imshow(res.get()[12, 700:900, 0:300])

# %%
import numpy as np
from astropy import modeling

fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()  # depending on the data you need to give some initial values
# fitted_model = fitter(model, range(11), window[11])

# from clij2fft.richardson_lucy import richardson_lucy_nc
# from tifffile import imread

# img = imread("/fast2/3t3clean/dapi_edu/dapi_edu-0005.tif")[:-1].reshape(-1, 2, 2048, 2048)
# # %%
# import numpy as np

# psf = imread("data/psf_405.tif")
# psf = np.median(psf, axis=0)
# import matplotlib.pyplot as plt

# psf = psf[[5, 10, 15, 20, 25, 30, 35]]
# # %%

# decon_clij2 = richardson_lucy_nc(img[0], psf, 10, 0.0002)

# # %%

# %%
