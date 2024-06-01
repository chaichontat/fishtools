# %%

import json
import queue
import threading
from pathlib import Path
from typing import Any, Iterable

import cupy as cp
import cv2
import numpy as np
import numpy.typing as npt
import rich_click as click
import tifffile
from cupyx.scipy.ndimage import convolve as cconvolve
from loguru import logger
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


def reverse_psf(inPSF: np.ndarray):
    Sx, Sy, Sz = inPSF.shape
    outPSF = np.zeros_like(inPSF)

    i = np.arange(Sx)
    j = np.arange(Sy)
    k = np.arange(Sz)

    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    outPSF = inPSF[Sx - 1 - ii, Sy - 1 - jj, Sz - 1 - kk]

    return outPSF


def _calculate_projectors_3d(
    pf: cp.ndarray, σ_G: float, a: float = 0.001, b: float = 0.001, n: int = 8
) -> tuple[cp.ndarray, cp.ndarray]:
    pfFFT = cp.fft.fftn(pf)
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
    # OTF_butterworth = 1/sqrt(1+ee*(kx/kcx)^pn)
    eps = cp.sqrt(1.0 / (β**2) - 1)
    # https://github.com/eguomin/regDeconProject/blob/3ca800fce083f2e936105f8334bf5ecc6ee8438b/WBDeconvolution/BackProjector.m#L223
    bBWorth = 1.0 / cp.sqrt(1.0 + eps**2 * (kk / kc) ** (2 * n))
    pbFFT = bWiener * bBWorth
    pb = cp.real(cp.fft.ifftn(pbFFT))

    return pf, pb


EPS = 1e-9
I_MAX = 2**16 - 1


def center_index(center: int, nz: int, step: int):
    ns = []
    curr = center
    while curr > 0:
        ns.insert(0, curr)
        curr -= step
    ns.extend(list(range(center, nz, step))[1 : len(ns)])
    return ns


def deconvolve_lucyrichardson_guo(
    img: np.ndarray[np.float32, Any],
    projectors: tuple[cp.ndarray, cp.ndarray],
    iters: int = 1,
) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:
    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.
    """

    if img.dtype not in [np.float32, np.float16]:
        raise ValueError("Image must be float32")
    forward_projector, backward_projector = projectors

    estimate = cp.clip(img, EPS, None)
    # print(estimate.shape, forward_projector.shape, backward_projector.shape)

    for _ in range(iters):
        filtered_estimate = cconvolve(estimate, forward_projector, mode="reflect").clip(
            EPS,
            I_MAX,
        )

        ratio = img / filtered_estimate
        # Correction
        estimate *= cconvolve(ratio, backward_projector, mode="reflect")

    return estimate


def make_projector(path: Path | str, *, step: int = 10, max_z: int = 7, size: int = 31, center: int = 50):
    gen = imread(path := Path(path))
    assert gen.shape[1] == gen.shape[2]

    zs = center_index(center, gen.shape[0], step)
    z_crop = (len(zs) - max_z) // 2
    zs = zs[z_crop:-z_crop] if z_crop > 0 else zs

    crop = (gen.shape[1] - size) // 2
    print(gen.shape, zs, z_crop)
    psf = gen[zs][::-1, crop:-crop, crop:-crop]
    psf /= psf.sum()
    logger.debug(f"PSF shape: {psf.shape}")
    p = [
        x.get()[:, np.newaxis, ...]
        for x in _calculate_projectors_3d(cp.array(psf), σ_G=1.7, a=0.02, b=0.02, n=10)
    ]
    with open(path.with_suffix(".npy"), "wb") as f:
        np.save(f, p)
    return p


def rescale(img: cp.ndarray, scale: float):
    return ((img - img.min()) * scale).get().astype(np.uint16)


# %%
DATA = Path("/home/chaichontat/fishtools/data")
make_projector(Path(DATA / "PSF GL.tif"), step=10, max_z=9)
projectors = [x.astype(cp.float32) for x in cp.load(DATA / "PSF GL.npy")]

# %%
# def run():
# img = (
#     imread(path := Path("/fast2/brainclean/1_9_17/1_9_17-0123.tif"))[:-1]
#     .reshape(-1, 3, 2048, 2048)
#     .astype(np.float32)
# )
# %%

# def run():
#     while True:
#         start, img, fid = q_img.get()
#         logger.info(f"Processing {start.name}")
#         res = deconvolve_lucyrichardson_guo(cp.asarray(img), projectors, iters=1)

#         if normalize:
#             towrite = cp.clip((res - mins) * scale, 0, 65535).astype(np.uint16).get().reshape(-1, 2048, 2048)
#             q_write.put((start, towrite, fid))
#         else:
#             towrite = res.get()
#             q_write.put((start, towrite, None))

#         q_img.task_done()
#         if q_img.empty():
#             break


# with ThreadPoolExecutor(2) as executor:
#     futs = [executor.submit(run) for i in range(2)]
#     [fut.result() for fut in futs]


# %%


# %%

# import matplotlib.pyplot as plt
# import seaborn as sns
# %%
# imgs = [imread(f) for f in out.glob("deconv*.tif")]
# # %%
# i = 1


@click.group()
def main(): ...


def get_scale(imgs: list[np.ndarray], i: int):
    mins = np.array([x[:, i].min() for x in imgs])
    maxs = np.array([x[:, i].max() for x in imgs])
    sub = mins.min()
    scale = np.percentile(60000 / (maxs - sub), 0.1)
    return {"min": float(sub), "scale": float(scale)}


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("name", type=str)
def scale(path: Path, name: str):
    logger.info("Reading images.")
    bits = name.split("_")

    imgs = []
    for f in (path / "analysis" / "deconv" / name).glob(f"float_{name}*.tif"):
        imgs.append(imread(f))
        try:
            assert imgs[-1].shape[1] == len(bits), f"Expected {len(bits)} channels, got {imgs[-1].shape[1]}."
            assert len(imgs[-1].shape) == 4
        except Exception as e:
            logger.error(f"Skipping {f}: {e}")
            imgs.pop()

    scale_file = path / "deconv_scale.json"
    existing = json.loads(scale_file.read_text()) if scale_file.exists() else {}
    for i, bit in enumerate(bits):
        res = get_scale(imgs, int(i))
        logger.debug(f"Bit {bit}: {res}")
        if bit in existing and not np.allclose(existing[bit]["min"], res["min"]):
            raise ValueError(f"Bit {bit} already exists in scale file.")
        existing[bit] = res

    with scale_file.open("w") as f:
        json.dump(existing, f)


# @main.command()
# @click.argument("path", type=click.Path(path_type=Path))
# @click.argument("name", type=str)
# def initital


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("name", type=str)
@click.option("--normalize", is_flag=True)
def run(path: Path, name: str, normalize: bool = False):
    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)

    bits = name.split("_")
    files = [f for f in sorted(path.rglob(f"{name}*.tif")) if not "analysis/deconv" in str(f)]

    if not normalize:
        if len(list(out.glob("float_"))):
            logger.warning("Float files already exist.")
            return

        files = [files[i] for i in np.random.default_rng(0).choice(range(len(files)), size=50, replace=False)]

    if normalize:
        _scaling = json.loads((path / "deconv_scale.json").read_text())
        mins = cp.array([_scaling[x]["min"] for x in bits]).reshape(1, len(bits), 1, 1)
        scale = cp.array([_scaling[x]["scale"] for x in bits]).reshape(1, len(bits), 1, 1)

    q_write = queue.Queue(maxsize=3)
    q_img: queue.Queue[tuple[Path, np.ndarray, Iterable[np.ndarray]]] = queue.Queue(maxsize=1)

    def f_read(files: list[Path]):
        logger.info("Read thread started.")
        # for start in range(0, len(files), 3):
        #     imgs = [imread(file) for file in files[start : start + 3]]
        #     fid = [img[[-1]] for img in imgs]
        #     imgs = np.stack([img[:-1].reshape(-1, 3, 2048, 2048) for img in imgs], axis=0)
        #     q_img.put((start, imgs.astype(np.float16), fid))
        for file in files:
            logger.debug(f"Reading {file.name}")
            img = imread(file)
            fid = img[[-1]]
            nofid = img[:-1].reshape(-1, len(bits), 2048, 2048).astype(np.float32)
            logger.debug(f"Finished reading {file.name}")
            q_img.put((file, nofid, fid))

    def f_write():
        logger.info("Write thread started.")

        while True:
            file, towrite, fid = q_write.get()
            logger.debug(f"Writing {file.name}")
            sub = out / file.name.split("-")[0]
            sub.mkdir(exist_ok=True, parents=True)

            if fid is None:
                tifffile.imwrite(
                    sub / ("float_" + file.name),
                    towrite,
                    compression="zstd",
                    compressionargs={"level": 1},
                )
            else:
                tifffile.imwrite(
                    sub / file.name,
                    np.concatenate([towrite, fid], axis=0),
                    compression=22610,
                    compressionargs={"level": 0.75},
                )
            logger.debug(f"Finished writing {file.name}")
            q_write.task_done()

    thread = threading.Thread(target=f_read, args=(files,), daemon=True)
    thread.start()

    thread_write = threading.Thread(target=f_write, args=(), daemon=True)
    thread_write.start()

    for _ in range(len(files)):
        start, img, fid = q_img.get()
        logger.info(f"Processing {start.name} ({_}/{len(files)})")
        res = deconvolve_lucyrichardson_guo(cp.asarray(img), projectors, iters=1)

        if normalize:
            towrite = cp.clip((res - mins) * scale, 0, 65535).astype(np.uint16).get().reshape(-1, 2048, 2048)
            q_write.put((start, towrite, fid))
        else:
            towrite = res.get()
            q_write.put((start, towrite, None))

        q_img.task_done()


if __name__ == "__main__":
    main()
# %%


# out = {bit: get_scale(imgs, i) for i, bit in enumerate(name.split("_"))}

# %%
# import json


# sns.set_theme()
# fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
# axs = axs.flatten()
# for i, ax in enumerate(axs):
#     ax.axis("off")

# sl = np.s_[200:400, 800:1000]
# sl = np.s_[1200:1500, 700:1000]
# axs[0].imshow(img[11, 0][sl])
# axs[1].imshow(res.get()[11, 0][sl])


# %%


# def run(i: int):
#     out[i] = deconvolve_lucyrichardson_guo(img[i, 0], window, iters=1)


# for i in range(5):
#     run(i)
# from concurrent.futures import ThreadPoolExecutor

# with ThreadPoolExecutor(4) as executor:
#     executor.map(run, range(img.shape[0]))


# %%


# %%
# tifffile.imwrite("ori.tif", img[:, 0].astype(np.uint16), compression=22610)
# # %%
# res = res.get()
# tifffile.imwrite("deconvrl.tif", ((res - res.min()) / res.max() * 65535).astype(np.uint16), compression=22610)

# %%


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
