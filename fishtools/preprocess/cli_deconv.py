import functools
import json
import pickle
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import cupy as cp
import cv2
import numpy as np
import numpy.typing as npt
import pyfiglet
import rich_click as click
import tifffile
from basicpy import BaSiC
from cupyx.scipy.ndimage import convolve as cconvolve
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from scipy.stats import multivariate_normal
from tifffile import TiffFile, imread

from fishtools.utils.pretty_print import progress_bar

DATA = Path(__file__).parent.parent.parent / "data"


def scale_deconv(
    img: np.ndarray,
    idx: int,
    *,
    global_deconv_scaling: np.ndarray,
    metadata: dict[str, Any],
    name: str | None = None,
    debug: bool = False,
):
    """Scale back deconvolved image using global scaling.

    Args:
        img: Original image, e.g. 1_9_17
        idx: Channel index in original image e.g. 0, 1, 2, ...
    """
    global_deconv_scaling = global_deconv_scaling.reshape((2, -1))
    m_ = global_deconv_scaling[0, idx]
    s_ = global_deconv_scaling[1, idx]

    # Same as:
    # scaled = s_ * ((img / self.metadata["deconv_scale"][idx] + self.metadata["deconv_min"][idx]) - m_)
    scale_factor = s_ / metadata["deconv_scale"][idx]
    offset = s_ * (metadata["deconv_min"][idx] - m_)
    scaled = scale_factor * img + offset

    if debug:
        logger.debug(f"Deconvolution scaling: {scale_factor}")
    if name and scaled.max() > 65535:
        logger.debug(f"Scaled image {name} has max > 65535.")

    if np.all(scaled < 0):
        logger.warning(f"Scaled image {name} has all negative values.")

    return np.clip(scaled, 0, 65535)


def _compute_range(path: Path, round_: str, *, perc_min: float = 99.9, perc_scale: float = 0.1):
    """
    Find the min and scale of the deconvolution for all files in a directory.
    The reverse scaling equation is:
                 s_global
        scaled = -------- * img + s_global * (min - m_global)
                  scale
    Hence we want scale_global to be as low as possible
    and min_global to be as high as possible.
    """
    files = sorted(path.glob(f"{round_}*/*.tif"))
    n_c = len(round_.split("_"))
    n = len(files)

    deconv_min = np.zeros((n, n_c))
    deconv_scale = np.zeros((n, n_c))
    logger.info(f"Found {n} files")
    if not files:
        raise FileNotFoundError(f"No files found in {path}")

    with progress_bar(len(files)) as pbar:
        for i, f in enumerate(files):
            try:
                meta = json.loads(Path(f).with_suffix(".deconv.json").read_text())
            except FileNotFoundError as e:
                with TiffFile(f) as tif:
                    try:
                        meta = tif.shaped_metadata[0]
                    except KeyError:
                        raise AttributeError("No deconv metadata found.")

            try:
                deconv_min[i, :] = meta["deconv_min"]
                deconv_scale[i, :] = meta["deconv_scale"]
            except KeyError:
                raise AttributeError("No deconv metadata found.")
            pbar()

    logger.info("Calculating percentiles")
    m_ = np.percentile(deconv_min, perc_min, axis=0)
    s_ = np.percentile(deconv_scale, perc_scale, axis=0)

    if np.any(m_ == 0) or np.any(s_ == 0):
        raise ValueError("Found a channel with min=0. This is not allowed.")

    (path / "deconv_scaling").mkdir(exist_ok=True)
    np.savetxt(path / "deconv_scaling" / f"{round_}.txt", np.vstack([m_, s_]))
    logger.info(f"Saved to {path / 'deconv_scaling' / f'{round_}.txt'}")


logger.remove()
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": "INFO"}])
console = Console()


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
    # print(gen.shape, zs, z_crop)
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
    """Scale from float32 to uint16.

    In order to store as much of the dynamic range as possible,
    while keeping JPEG-XR compression.
    """
    return ((img - img.min()) * scale).get().astype(np.uint16)


@click.group()
def deconv(): ...


@deconv.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--perc_min", type=float, default=0.1, help="Percentile of the min")
@click.option("--perc_scale", type=float, default=0.1, help="Percentile of the scale")
@click.option("--overwrite", is_flag=True)
def compute_range(path: Path, perc_min: float = 0.1, perc_scale: float = 0.1, overwrite: bool = False):
    """
    Find the scaling factors of all images in the children sub-folder of `path`.
    Run this on the entire workspace. See `_compute_range` for more details.
    """
    rounds = sorted({
        p.parent.name.split("--")[0] for p in path.rglob("*.tif") if len(p.parent.name.split("--")) == 2
    })
    print(rounds)
    if "deconv" not in path.resolve().as_posix():
        raise ValueError("This command must be run in the deconvolved folder.")

    for round_ in rounds:
        if not overwrite and (path / "deconv_scaling" / f"{round_}.txt").exists():
            continue
        try:
            logger.info(f"Processing {round_}")
            _compute_range(path, round_, perc_min=perc_min, perc_scale=perc_scale)
        except (AttributeError, FileNotFoundError):
            logger.info("Invalid folder. Skipping.")


# @main.command()
# @click.argument("path", type=click.Path(path_type=Path))
# @click.argument("name", type=str)
# def initital

channels = [560, 650, 750]


@functools.cache
def projectors():
    make_projector(Path(DATA / "PSF GL.tif"), step=6, max_z=7)
    return [x.astype(cp.float32) for x in cp.load(DATA / "PSF GL.npy")]


def _run(
    paths: list[Path],
    out: Path,
    basics: dict[str, list[BaSiC]],
    overwrite: bool,
    n_fids: int,
):
    if not overwrite:
        paths = [f for f in paths if not (out / f.parent.name / f.name).exists()]

    q_write = queue.Queue(maxsize=3)
    q_img: queue.Queue[tuple[Path, np.ndarray, Iterable[np.ndarray], dict]] = queue.Queue(maxsize=1)

    def f_read(files: list[Path]):
        logger.debug("Read thread started.")
        for file in files:
            name = file.name.split("-")[0]
            bits = name.split("_")
            if file.name.startswith("fid"):
                continue
            logger.debug(f"Reading {file.name}")
            try:
                img = imread(file)
                fid = np.atleast_3d(img[-n_fids:])
                nofid = img[:-n_fids].reshape(-1, len(bits), 2048, 2048).astype(np.float32)
            except ValueError as e:
                raise Exception(f"File {file.resolve()} is corrupted. Please check the file.") from e

            with tifffile.TiffFile(file) as tif:
                try:
                    metadata = tif.shaped_metadata[0]  # type: ignore
                except (TypeError, IndexError):
                    metadata = tif.imagej_metadata or {}

            logger.debug(f"Finished reading {file.name}")
            for i, c in enumerate(channels):
                if i < nofid.shape[1]:
                    nofid[:, i] = basics[name][i].transform(np.array(nofid[:, i]))
            q_img.put((file, nofid, fid, metadata))

    def f_write():
        logger.debug("Write thread started.")

        while True:
            gotten = q_write.get()
            if gotten is None:
                break
            file, towrite, fid, metadata = gotten
            logger.debug(f"Writing {file.name}")
            sub = out / file.parent.name
            sub.mkdir(exist_ok=True, parents=True)

            (sub / file.name).with_suffix(".deconv.json").write_text(json.dumps(scaling, indent=2))

            tifffile.imwrite(
                sub / file.name,
                np.concatenate([towrite, fid], axis=0),
                compression=22610,
                compressionargs={"level": 0.75},
                metadata=metadata,
            )
            logger.debug(f"Finished writing {file.name}")
            q_write.task_done()

    try:
        thread = threading.Thread(target=f_read, args=(paths,), daemon=True)
        thread.start()

        thread_write = threading.Thread(target=f_write, args=(), daemon=True)
        thread_write.start()

        with progress_bar(len(paths)) as callback:
            for _ in range(len(paths)):
                start, img, fid, metadata = q_img.get()
                t = time.time()

                res = deconvolve_lucyrichardson_guo(cp.asarray(img), projectors(), iters=1)
                mins = res.min(axis=(0, 2, 3), keepdims=True)
                scale = 65534 / (res.max(axis=(0, 2, 3), keepdims=True) - mins)

                towrite = ((res - mins) * scale).astype(np.uint16).get().reshape(-1, 2048, 2048)
                del res

                scaling = {
                    "deconv_min": list(map(float, mins.get().flatten())),
                    "deconv_scale": list(map(float, scale.get().flatten())),
                }
                logger.info(f"Finished {start.name}: {time.time() - t:.2f}s")

                q_write.put((
                    start,
                    towrite,
                    fid,
                    metadata | scaling,
                ))
                q_img.task_done()
                callback()

    except Exception as e:
        logger.error(f"Error in thread: {e}")
        q_write.put(None)
        raise e

    q_write.put(None)
    thread.join()
    thread_write.join()


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("name", type=str)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--limit", type=int, default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str, default=None)
def run(
    path: Path,
    name: str,
    *,
    ref: Path | None,
    limit: int | None,
    overwrite: bool,
    basic_name: str | None,
    n_fids: int,
):
    """GPU-accelerated very accelerated 3D deconvolution.

    Separate read and write threadsin order to have an image ready for the GPU at all times.
    Outputs in `path/analysis/deconv`.

    Args:
        path: Path to the workspace.
        name: Name of the round to deconvolve. E.g. 1_9_17.
        ref: Path of folder that have the same image indices as those we want to deconvolve.
            Used when skipping blank areas in round >1.
            We don't want to deconvolve the blanks in round 1.
        limit: Limit the total number of images to deconvolve. Mainly for debugging.
        overwrite: Overwrite existing deconvolved images.
        n_fid: Number of fiducial frames.
    """

    console.print(f"[magenta]{pyfiglet.figlet_format('3D Deconv', font='slant')}[/magenta]")

    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)
    files = [
        f
        for f in sorted(path.glob(f"{name}--*/{name}-*.tif"))
        if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
    ]
    if ref is not None:
        ok_idxs = {
            int(f.stem.split("-")[1])
            for f in sorted(path.rglob(f"{ref}*/{ref}-*.tif"))
            if "analysis/deconv" not in str(f)
        }
        logger.info(f"Filtering files to {ref}. Total: {len(ok_idxs)}")
        files = [f for f in files if int(f.stem.split("-")[1]) in ok_idxs]
        if len(files) != len(ok_idxs):
            logger.warning(f"Filtering reduced the number of files to {len(files)} ≠ length of ref.")

    files = files[:limit] if limit is not None else files
    logger.info(
        f"Total: {len(files)} at {path}/{name}*" + (f" Limited to {limit}" if limit is not None else "")
    )

    if not overwrite:
        files = [f for f in files if not (out / f.parent.name / f.name).exists()]

    if files:
        logger.info(f"Running {len(files)} files.")
    else:
        logger.warning("No files found. Skipping.")
        return

    basic_name = basic_name or name
    logger.info(f"Using({path / 'basic'}/{basic_name}-*.pkl for BaSiC")
    basics: dict[str, list[BaSiC]] = {
        name: [
            pickle.loads((path / "basic" / f"{basic_name}-{c}.pkl").read_bytes())
            for c in range(len(name.split("_")))
        ]
    }
    _run(files, path / "analysis" / "deconv", basics, overwrite=overwrite, n_fids=n_fids)


@deconv.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--roi", type=str, default=None)
@click.option("--ref", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--n-fids", type=int, default=2)
@click.option("--basic-name", type=str, default=None)
def batch(
    path: Path,
    *,
    roi: str | None,
    ref: Path | None,
    overwrite: bool,
    n_fids: int,
    basic_name: str | None,
):
    out = path / "analysis" / "deconv"
    out.mkdir(exist_ok=True, parents=True)

    FORBIDDEN = ["10x", "analysis", "shifts", "fid", "registered", "old", "basic"]

    rounds = sorted({p.name.split("--")[0] for p in path.iterdir() if "--" in p.name and p.is_dir()})
    for r in rounds:
        files = [
            f
            for f in sorted(path.glob(f"{r}--{roi or '*'}/{r}-*.tif"))
            if "analysis/deconv" not in str(f.resolve())
            and not any(f.parent.name.endswith(x) for x in FORBIDDEN)
        ]

        if not files:
            logger.info(f"No files found for {r}, skipping. Use --overwrite to reprocess.")
            continue

        if ref is not None:
            ok_idxs = {
                int(f.stem.split("-")[1])
                for f in sorted(path.rglob(f"{ref}--{roi or '*'}/{ref}-*.tif"))
                if "analysis/deconv" not in str(f) and not f.parent.name.endswith("basic")
            }
            logger.info(f"Filtering files to {ref}. Total: {len(ok_idxs)}")
            files = [f for f in files if int(f.stem.split("-")[1]) in ok_idxs]
            if len(files) != len(ok_idxs):
                logger.warning(f"Filtering reduced the number of files to {len(files)} ≠ length of ref.")

        try:
            basics: dict[str, list[BaSiC]] = {
                r: [
                    pickle.loads((path / "basic" / f"{basic_name or r}-{c}.pkl").read_bytes())
                    for c in range(len(r.split("_")))
                ]
            }
        except FileNotFoundError:
            logger.error(
                f"Could not find {path / 'basic' / f'{basic_name or r}-[n].pkl'}. Please run basic first. Skipping to rounds with basic."
            )
        else:
            _run(
                files,
                path / "analysis" / "deconv",
                basics,
                overwrite=overwrite,
                n_fids=n_fids,
            )


if __name__ == "__main__":
    deconv()
