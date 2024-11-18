import json
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich
from astropy.stats import SigmaClip, sigma_clipped_stats
from loguru import logger
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree
from skimage import exposure, filters
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform
from tifffile import TiffFile

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
):
    res: np.ndarray = filters.butterworth(
        image,
        cutoff_frequency_ratio=cutoff,
        order=order,
        high_pass=True,
        squared_butterworth=squared_butterworth,
        npad=npad,
    )  # type: ignore
    return np.clip(res, 0, None)


def clahe(img: np.ndarray, clip_limit: float = 0.01, bins: int = 200):
    return exposure.equalize_adapthist(img, clip_limit=0.02)


def find_spots(
    img: np.ndarray[np.uint16, Any],
    threshold_sigma: float,
    fwhm: float,
):
    """Fit Gaussians to the image and find peaks.

    Args:
        data: np.ndarray of an image.
        threshold_sigma: fits with > this std.dev over the median of the image are considered spots.
        fwhm: Full width at half maximum of the Gaussian. Larger values will find more spots.

    Returns:
        pl.DataFrame: Sorted by magnitude (brightest first).
    """
    img = img.squeeze()
    if img.ndim != 2:
        raise ValueError(
            "Reference image must be 2D for DAOStarFinder. https://github.com/astropy/photutils/issues/1328"
        )
    if np.sum(img) == 0:
        raise ValueError("Reference image must have non-zero sum.")

    # min_ = img.min()
    # normalized = clahe((img - min_) / (img.max() - min_))

    _mean, median, std = sigma_clipped_stats(img, sigma=threshold_sigma + 5)
    # You don't want need to subtract the mean here, the median is subtracted in the call three lines below.
    iraffind = DAOStarFinder(threshold=threshold_sigma * std, fwhm=fwhm, exclude_border=True)
    try:
        df = pl.DataFrame(iraffind(img - median).to_pandas()).sort("mag").with_row_count("idx")
    except AttributeError:
        df = pl.DataFrame()
    if len(df) < 6:
        raise NotEnoughSpots
    return df


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
    use_brightest: int = 0,
    # subtract_background: bool = False,
    plot: bool = False,
    precision: int = 2,
):
    """scipy ordering is based on (z, y, x) like the image dimensions."""

    if len(target_points) > 1000:
        logger.warning(
            f"WARNING: a lot ({len(target_points)}) of fiducials found. "
            "This may be noise and is slow. "
            "Please reduce FWHM or increase threshold."
        )

    cols = ["xcentroid", "ycentroid"]

    if use_brightest:
        target_points = target_points.sort("mag", descending=True)[:use_brightest]

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
        _, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
        axs = axs.flatten()
        axs[0].hist(joined["dx"], bins=100)
        axs[1].hist(joined["dy"], bins=100)

    # model: TranslationTransform = ransac(
    #     (
    #         joined[["xcentroid", "ycentroid"]].to_numpy(),
    #         joined[["xcentroid_fixed", "ycentroid_fixed"]].to_numpy(),
    #     ),
    #     TranslationTransform,
    #     min_samples=10,
    #     residual_threshold=0.1,
    #     max_trials=1000,
    # )[0]  # type: ignore
    # print(model.translation)

    if np.allclose(initial_drift, np.zeros(2)) and len(joined) > 100:
        if joined["dx"].sum() == 0 and joined["dy"].sum() == 0:
            raise ValueError("No drift found. Reference passed?")
        res = np.array([mode(joined["dx"]), mode(joined["dy"])])  # type: ignore
        if np.hypot(*res).sum() < 40:
            return np.round(res, precision)

    # if len(joined) < 8:
    #     # use weighted mean
    #     drift = joined.select(dx=pl.col("dx") * pl.col("flux"), dy=pl.col("dy") * pl.col("flux")).sum()
    #     drift = np.array(drift).squeeze() / joined["flux"].sum()
    #     res = initial_drift + drift
    # else:
    drifts = joined[["dx", "dy"]].to_numpy()
    drift = np.median(drifts, axis=0)

    hypot = np.hypot(*drifts.T)
    cv = np.std(hypot) / np.mean(hypot)
    logger.debug(f"drift: {drift} CV: {cv:04f}.")
    res = initial_drift + drift

    return np.round(res, precision)


class NotEnoughSpots(Exception): ...


class TooManySpots(Exception): ...


class ResidualTooLarge(Exception): ...


def run_fiducial(
    ref: np.ndarray,
    *,
    subtract_background: bool = False,
    debug: bool = False,
    name: str = "",
    threshold_sigma: float = 3,
    threshold_residual: float = 0.3,
    use_brightest: int = 100,
    fwhm: float = 4,
):
    if subtract_background:
        ref = ref - background(ref)

    _attempt = 0
    thr = threshold_sigma

    while _attempt < 6:
        try:
            fixed = find_spots(ref, threshold_sigma=thr, fwhm=fwhm)
            if len(fixed) > 1500:
                raise TooManySpots(
                    f"Too many spots ({len(fixed)} > 1500) found on the reference image. Please increase threshold_sigma or reduce FWHM."
                )
        except NotEnoughSpots:
            logger.warning(
                "Not enough spots found on reference image. Trying to find spots with lower threshold."
            )
            thr -= 0.5
            _attempt += 1
            continue
        except TooManySpots:
            logger.warning(
                "Too many spots found on reference image. Trying to find spots with lower threshold."
            )
            thr += 0.5
            _attempt += 1
        else:
            # Find steepest slope
            fixed = fixed[: max(np.argmax(np.diff(fixed["mag"])), 10, len(fixed) // 4)]
            break
    else:
        raise NotEnoughSpots("Could not find spots on reference after 4 attempts.")

    logger.debug(f"{name}: {len(fixed)} peaks found on reference image.")
    kd = cKDTree(fixed[["xcentroid", "ycentroid"]])

    def inner(img: np.ndarray, *, limit: int = 4, bitname: str = "", local_σ: float = thr):
        if subtract_background:
            img = img - background(img)

        _attempt = 0
        residual = np.inf
        drift = np.array([0, 0])

        # Iteratively reduce threshold_sigma until we get enough fiducials.
        while _attempt < 10:
            try:
                moving = find_spots(img, threshold_sigma=local_σ, fwhm=fwhm)
                moving = moving[: max(np.argmax(np.diff(moving["mag"])), 10, len(moving) // 4)]

                logger.debug(f"{bitname}: {len(moving)} peaks found on target image.")

                initial_drift = np.zeros(2)
                assert limit > 0

                # Actual loop drift loop.
                for n in range(limit):
                    # Can raise NotEnoughSpots
                    drift = _calculate_drift(
                        kd, fixed, moving, initial_drift=initial_drift, use_brightest=100
                    )
                    residual = np.hypot(*(drift - initial_drift))
                    if n == 0:
                        logger.debug(f"{bitname} - attempt {n}: starting drift: {drift}.")
                    else:
                        logger.debug(
                            f"{bitname} - attempt {n}: new drift: {drift} Δ from last {residual:.2f}px."
                        )
                    if residual < threshold_residual:
                        break
                    initial_drift = drift

            except NotEnoughSpots:
                _attempt += 1
                local_σ -= 0.5
                logger.warning(f"Not enough spots. Attempt {_attempt}. σ threshold: {local_σ}")
            except TooManySpots:
                _attempt += 1
                local_σ += 0.5
                logger.warning(f"Too many spots. Attempt {_attempt}. σ threshold: {local_σ}")
            else:
                break
        else:
            logger.critical(f"Could not find spots after {_attempt} attempts.")
            raise NotEnoughSpots

        if np.hypot(*drift) > 40:
            logger.warning(f"{bitname}: drift very large {np.hypot(*drift):.2f}.")

        if residual > threshold_residual:
            raise ResidualTooLarge(
                f"{bitname}: residual drift too large {residual=:2f}. Please refine parameters."
            )
        return drift, residual

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


def align_fiducials(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    threads: int = 4,
    overrides: dict[str, tuple[float, float]] | None = None,
    subtract_background: bool = False,
    debug: bool = False,
    max_iters: int = 4,
    threshold_sigma: float = 3,
    threshold_residual: float = 0.3,
    fwhm: float = 4,
) -> tuple[dict[str, np.ndarray[float, Any]], dict[str, np.ndarray[float, Any]]]:
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
        threshold_residual=threshold_residual,
        fwhm=fwhm,
    )
    with ThreadPoolExecutor(threads if not debug else 1) as exc:
        futs: dict[str, Future] = {}
        for k, img in fids.items():
            if k == ref or (overrides is not None and k in overrides):
                continue
            futs[k] = exc.submit(corr, img, bitname=k, limit=max_iters)

            if debug:
                futs[k].result()

        for fut in as_completed(futs.values()):
            fut.result()

    # Shifts and residuals
    return (
        (
            {k: v.result()[0] for k, v in futs.items()}
            | {ref: np.zeros(2)}
            | ({k: np.array(v) for k, v in overrides.items()} if overrides else {})
        ),
        {k: v.result()[1] for k, v in futs.items()},
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


def align_fiducials_from_file(
    folder: Path | str,
    glob: str,
    *,
    reference: str,
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
@click.option("--idx", "-i", type=int, default=-1)
@click.option("--threads", "-t", type=int, default=4)
def main(
    folder: Path | str,
    glob: str,
    output: Path,
    reference: str,
    idx: int = -1,
    threads: int = 4,
):
    res = align_fiducials_from_file(folder, glob, reference=reference, idx=idx, threads=threads)
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
