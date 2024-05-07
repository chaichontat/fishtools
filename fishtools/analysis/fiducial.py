import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import polars as pl
import rich
import rich_click as click
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table.table import QTable
from loguru import logger
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder, IRAFStarFinder
from scipy.ndimage import shift
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from tifffile import TiffFile

console = rich.get_console()


def imread_page(path: Path | str, page: int):
    with TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


def find_spots(data: np.ndarray[np.uint16, Any], threshold_sigma: float = 3, fwhm: float = 4) -> QTable:
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    # iraffind = DAOStarFinder(threshold=3.0 * std, fwhm=4, sharplo=0.2, exclude_border=True)
    iraffind = DAOStarFinder(threshold=threshold_sigma * std, fwhm=fwhm, exclude_border=True)
    return iraffind(data - median)


def calc_shift(ref: np.ndarray, img: np.ndarray, precision: int = 2):
    return phase_cross_correlation(ref, img, upsample_factor=int(10**precision))[0]


def background(img: np.ndarray):
    sigma_clip = SigmaClip(sigma=2.0)
    bkg = Background2D(
        img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=MedianBackground()
    )
    return bkg.background


def gen_corr(ref: np.ndarray, precision: int = 2, fiducial_corr: bool = True):
    cols = ["xcentroid", "ycentroid"]
    ref = ref - background(ref)

    fixed = pl.DataFrame(find_spots(ref).to_pandas()).with_row_count("idx")
    kd = cKDTree(fixed[cols])

    def correct_fiducial(img: np.ndarray):
        """scipy ordering is based on (z, y, x) like the image dimensions."""
        img = img - background(img)

        # Initial alignment
        initial_px = calc_shift(ref, img)
        initial_px = np.array(initial_px)
        logger.debug(f"{initial_px=}")

        if not fiducial_corr:
            return np.round(initial_px, precision)

        t1 = shift(img, initial_px, order=1)
        # Find fiducials
        moving = pl.DataFrame(find_spots(t1)[cols].to_pandas()).with_row_count("idx")
        dist, idxs = kd.query(moving[cols], workers=2)
        mapping = pl.concat(
            [pl.DataFrame(dict(fixed_idx=np.array(idxs, dtype=np.uint32), dist=dist)), moving],
            how="horizontal",
        )

        # Remove duplicate mapping, priority on closest
        thresh = np.percentile(dist, 10), np.percentile(dist, 90)
        finite = (
            mapping.filter(pl.col("dist").gt(thresh[0]) & pl.col("dist").lt(thresh[1]))
            .sort("dist")
            .unique("fixed_idx", keep="first")
        )
        joined = finite.join(
            fixed[["idx", *cols]], left_on="fixed_idx", right_on="idx", how="left", suffix="_fixed"
        ).with_columns(
            dx=pl.col("xcentroid") - pl.col("xcentroid_fixed"),
            dy=pl.col("ycentroid") - pl.col("ycentroid_fixed"),
        )

        if len(joined) < 6:
            logger.warning(f"WARNING: not enough fiducials found {len(joined)}. Using initial alignment")
            return np.round(initial_px, precision)

        drift = joined[["dy", "dx"]].median().to_numpy().squeeze()
        if np.hypot(*drift) > 2:
            logger.warning("WARNING: drift too large", drift)

        final = initial_px - drift
        logger.debug(f"{drift=}")
        return np.round(final, precision)

    return correct_fiducial


def align_fiducials(
    fids: dict[str, np.ndarray[Any, Any]],
    *,
    reference: str,
    precision: int = 2,
    threads: int = 4,
    fiducial_corr: bool = True,
) -> dict[str, np.ndarray[float, Any]]:
    keys = list(fids.keys())
    for name in keys:
        if re.search(reference, name):
            ref = name
            break
    else:
        raise ValueError(f"Could not find reference {reference} in {keys}")

    corr = gen_corr(fids[ref], precision, fiducial_corr=fiducial_corr)
    with ThreadPoolExecutor(threads) as exc:
        futs = {k: exc.submit(corr, img) for k, img in fids.items() if k != ref}

    return {k: v.result() for k, v in futs.items()} | {ref: np.zeros(2)}


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
        precision=precision,
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
