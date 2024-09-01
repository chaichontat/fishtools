import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich
import rich_click as click
from astropy.stats import SigmaClip, sigma_clipped_stats
from loguru import logger
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from tifffile import TiffFile

console = rich.get_console()


def imread_page(path: Path | str, page: int):
    with TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


def find_spots(
    data: np.ndarray[np.uint16, Any],
    threshold_sigma: float,
    fwhm: float,
):
    assert np.sum(data) > 0
    # iraffind = DAOStarFinder(threshold=3.0 * std, fwhm=4, sharplo=0.2, exclude_border=True)
    mean, median, std = sigma_clipped_stats(data, sigma=threshold_sigma + 5)
    # You don't want need to subtract the mean here, the median is subtracted in the call three lines below.
    iraffind = DAOStarFinder(threshold=threshold_sigma * std, fwhm=fwhm, exclude_border=True)
    try:
        return pl.DataFrame(iraffind(data - median).to_pandas()).with_row_count("idx")
    except AttributeError as e:
        return pl.DataFrame()


def phase_shift(ref: np.ndarray, img: np.ndarray, precision: int = 2):
    return phase_cross_correlation(ref, img, upsample_factor=int(10**precision))[0]


def background(img: np.ndarray):
    sigma_clip = SigmaClip(sigma=2.0)
    bkg = Background2D(
        img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=MedianBackground()
    )
    return bkg.background


def _calculate_drift(
    ref_kd: cKDTree,
    ref_points: pl.DataFrame,
    target_points: pl.DataFrame,
    *,
    initial_drift: np.ndarray | None = None,
    # subtract_background: bool = False,
    plot: bool = False,
    precision: int = 2,
):
    """scipy ordering is based on (z, y, x) like the image dimensions."""

    cols = ["xcentroid", "ycentroid"]

    # points = moving[cols].to_numpy()
    if initial_drift is None:
        initial_drift = np.zeros(2)

    target_points = target_points.with_columns(
        xcentroid=pl.col("xcentroid") + initial_drift[0],
        ycentroid=pl.col("ycentroid") + initial_drift[1],
    )

    dist, idxs = ref_kd.query(target_points[cols], workers=2)
    mapping = pl.concat(
        [pl.DataFrame(dict(fixed_idx=np.array(idxs, dtype=np.uint32), dist=dist)), target_points],
        how="horizontal",
    )

    # Remove duplicate mapping, priority on closest
    # thresh = np.percentile(dist, 10), np.percentile(dist, 90)
    finite = mapping.sort("dist").unique("fixed_idx", keep="first")
    joined = finite.join(
        ref_points[["idx", *cols]], left_on="fixed_idx", right_on="idx", how="left", suffix="_fixed"
    ).with_columns(
        dx=pl.col("xcentroid_fixed") - pl.col("xcentroid"),
        dy=pl.col("ycentroid_fixed") - pl.col("ycentroid"),
    )

    if len(joined) > 1000:
        logger.warning(
            f"WARNING: a lot ({len(joined)}) of fiducials found. "
            "This may be noise and is slow. "
            "Please reduce FWHM or increase threshold."
        )

    if len(joined) < 6:
        logger.warning(f"WARNING: not enough fiducials found {len(joined)}. Using initial alignment")
        return np.round([0, 0], precision)

    def mode(data: np.ndarray):
        bin_size = 0.5
        bins = np.arange(min(data), max(data) + bin_size, bin_size)
        bin_indices = np.digitize(data, bins)
        bin_counts = np.bincount(bin_indices)
        i = np.argwhere(bin_counts == np.max(bin_counts)).flatten()[0]
        if i == 0 or i == len(bins) - 1:
            return np.median(data)
        return (bins[i] + bins[i + 1]) / 2

    if plot:
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
        axs = axs.flatten()
        axs[0].hist(joined["dx"], bins=100)
        axs[1].hist(joined["dy"], bins=100)

    if np.allclose(initial_drift, np.zeros(2)):
        if joined["dx"].sum() == 0 and joined["dy"].sum() == 0:
            raise ValueError("No drift found. Reference passed?")
        res = np.array([mode(joined["dx"]), mode(joined["dy"])])
    else:
        drift = joined[["dx", "dy"]].median().to_numpy().squeeze()
        res = initial_drift + drift

    return np.round(res, precision)


def run_fiducial(
    ref: np.ndarray,
    *,
    subtract_background: bool = False,
    debug: bool = False,
    name: str = "",
    threshold_sigma: float = 3,
    threshold_fiducial: float = 0.5,
    fwhm: float = 4,
):
    if subtract_background:
        ref = ref - background(ref)
    try:
        fixed = find_spots(ref, threshold_sigma=threshold_sigma, fwhm=fwhm)
        if not len(fixed):
            raise ValueError("No spots found on reference image.")
    except Exception as e:
        logger.error(f"Cannot find reference spots. {e}")
        return lambda *args, **kwargs: np.zeros(2)

    logger.debug(f"{name}: {len(fixed)} peaks found on reference image.")
    kd = cKDTree(fixed[["xcentroid", "ycentroid"]])

    def inner(img: np.ndarray, *, limit: int = 3, bitname: str = ""):
        if subtract_background:
            img = img - background(img)
        moving = find_spots(img, threshold_sigma=threshold_sigma, fwhm=fwhm)
        if len(moving) < 6:
            logger.warning(f"WARNING: not enough fiducials. Setting zero.")
            return np.round([0, 0])

        logger.debug(f"{bitname}: {len(moving)} peaks found on target image.")

        initial_drift = np.zeros(2)
        assert limit > 0
        for n in range(limit):
            drift = _calculate_drift(kd, fixed, moving, initial_drift=initial_drift)
            residual = np.hypot(*(drift - initial_drift))
            if n == 0:
                logger.debug(f"{bitname} - attempt {n}: starting drift: {drift}.")
            else:
                logger.debug(f"{bitname} - attempt {n}: new drift: {drift} Δ from last {residual:.2f}px.")
            if residual < threshold_fiducial:
                return drift
            initial_drift = drift

        if residual > 0.5:  # type: ignore
            logger.warning(f"{bitname}: residual drift too large {residual=:2f}.")  # type: ignore
        return drift  # type: ignore

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
                # Force single-thread
                logger.debug(f"{k}: {futs[k].result()}")

    return {k: v.result().astype(float) for k, v in futs.items()} | {ref: np.zeros(2)}


def align_fiducials(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    threads: int = 4,
    subtract_background: bool = False,
    debug: bool = False,
    iterations: int = 3,
    threshold_sigma: float = 3,
    threshold_fiducial: float = 0.5,
    fwhm: float = 4,
) -> dict[str, np.ndarray[float, Any]]:
    keys = list(fids.keys())
    for name in keys:
        if re.search(reference, name):
            ref = name
            break
    else:
        raise ValueError(f"Could not find reference {reference} in {keys}")

    del reference
    corr = run_fiducial(
        fids[ref],
        subtract_background=subtract_background,
        debug=debug,
        name=ref,
        threshold_sigma=threshold_sigma,
        threshold_fiducial=threshold_fiducial,
        fwhm=fwhm,
    )
    with ThreadPoolExecutor(threads if not debug else 1) as exc:
        futs: dict[str, Future] = {}
        for k, img in fids.items():
            if k == ref:
                continue
            futs[k] = exc.submit(corr, img, bitname=k, limit=iterations)

            if debug:
                futs[k].result()

    return {k: v.result() for k, v in futs.items()} | {ref: np.zeros(2)}


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


def align_fiducials_from_file(
    folder: Path | str,
    glob: str,
    *,
    reference: str,
    precision: int = 2,
    filter_: Callable[[str], bool] = lambda _: True,
    idx: int = -1,
    threads: int = 4,
) -> dict[str, np.ndarray[float, Any]]:
    return align_fiducials(
        {file.name: imread_page(file, idx) for file in sorted(Path(folder).glob(glob)) if filter_(file.name)},
        reference=reference,
        threads=threads,
    )


@click.command()
@click.argument("folder", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("glob", type=str)
@click.option("--output", "-o", type=click.Path(dir_okay=False, file_okay=True, path_type=Path))
@click.option("--reference", "-r", type=str)
@click.option("--precision", "-p", type=int, default=2)
@click.option("--idx", "-i", type=int, default=-1)
@click.option("--threads", "-t", type=int, default=4)
def main(
    folder: Path | str,
    glob: str,
    output: Path,
    reference: str,
    precision: int = 2,
    idx: int = -1,
    threads: int = 4,
):
    res = align_fiducials_from_file(
        folder, glob, reference=reference, precision=precision, idx=idx, threads=threads
    )
    output.write_text(json.dumps(res))


if __name__ == "__main__":
    main()


# idx = 3
# imgs = {
#     file.name: imread_page(file, -1)
#     for file in sorted(Path("/raid/data/raw/tricyclecells2").glob(f"*-{idx:03d}.tif"))
#     if not file.name.startswith("dapi")
# }


# img = imgs[f"A_1_2-{idx:03d}.tif"]
# no_a = img[:-1].reshape(16, 3, 2048, 2048)[:, [1, 2], ...].reshape(-1, 2048, 2048)
# imgs[f"1_2-{idx:03d}.tif"] = np.concatenate([no_a, imgs[f"A_1_2-{idx:03d}.tif"][None, -1]], axis=0)
# del imgs[f"A_1_2-{idx:03d}.tif"]


# dark = imread("/raid/data/analysis/dark.tif")
# flat = (imread("/raid/data/analysis/flat_647.tif") - dark).astype(np.float32)
# flat /= np.min(flat)  # Prevents overflow
# for name in imgs:
#     img = imgs[name]
#     imgs[name] = ((img - dark).astype(np.float32) / flat).astype(np.uint16)

# fids = {k: v[-1 for k, v in imgs.items()}
# imgs = {k: v[:-1] for k, v in imgs.items()}
# keys = list(imgs.keys())
