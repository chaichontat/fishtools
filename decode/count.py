# %%
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rich_click as click
import seaborn as sns
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table.table import QTable
from loguru import logger
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder, IRAFStarFinder
from scipy.ndimage import shift
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from tifffile import TiffFile, imread

from fishtools.utils.pretty_print import progress_bar

sns.set_theme()


def imread_page(path: Path | str, page: int):
    with TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


def find_spots(
    data: np.ndarray[np.uint16, Any], threshold: float = 100, fwhm: float = 4, median: int | None = None
) -> QTable | None:
    if median is None:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    iraffind = DAOStarFinder(threshold=threshold, fwhm=fwhm)
    return iraffind(np.clip(data, a_min=median, a_max=65535) - median)


# %%
# %%


def run(path: Path):
    if path.with_suffix(".parquet").exists():
        return
    img = imread(path)
    assert img.shape.__len__() == 3
    out = []
    for i in range(len(img)):
        sp = find_spots(img[i], threshold=50, fwhm=10, median=200)
        if sp is None:
            continue
        df = sp.to_pandas().set_index("id")
        df["channel"] = i
        out.append(df)
    del img
    pl.DataFrame(pd.concat(out)).write_parquet(path.with_suffix(".parquet"))
    # df.to_csv(path.parent / f"{path.stem}_c{c}_spots.csv")


# %%


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
def main(path: Path):
    files = sorted(Path(path).glob("*.tif"))
    with progress_bar(len(files)) as callback, ThreadPoolExecutor(16) as exc:
        futs = [exc.submit(run, f) for f in files]
        for f in as_completed(futs):
            f.result()
            callback()


if __name__ == "__main__":
    main()
# spots = run(img[29], threshold_sigma=2, fwhm=8)
# fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 6), dpi=200)
# sl = np.s_[1100:1400, 1100:1400]
# axs[0].imshow(img[29][sl], zorder=1)
# axs[1].imshow(img[29][sl], zorder=1)
# filtered = spots[spots["xcentroid"].between(1100, 1400) & spots["ycentroid"].between(1100, 1400)]
# axs[1].scatter(filtered["xcentroid"] - 1100, filtered["ycentroid"] - 1100, s=10, c="green")


# %%

# fig, axs = plt.subplots(ncols=5, nrows=6, figsize=(12, 12), dpi=200)
# axs = axs.flatten()
# for i, ax in enumerate(axs):
#     ax.axis("off")
#     ax.imshow(img[i])
#     ax.set_title(i)

# # %%
# with ThreadPoolExecutor(4) as exc:
#     for path in Path("/fast2/synaptosome/registered").rglob("reg-*.tif"):
#         exc.submit(run, path, 2)
# # %%
# from collections import defaultdict

# import pandas as pd

# path = Path("/disk/chaichontat/mer/peg/0peg")
# dfs = []
# for i in range(22, 23):
#     mask = imread(path / (f"dapi_polyA_malat_cux-{i:03d}_max_cp_masks.tif"))
#     img, spots = run(path / f"dapi_polyA_malat_cux-{i:03d}.tif", 2)
#     counts = defaultdict(int)
#     for s in spots.itertuples():
#         x, y = s.xcentroid, s.ycentroid
#         counts[mask[round(y), round(x)]] += 1
#     dfs.append(pd.DataFrame(counts.items(), columns=["img", "count"]))
#     dfs[-1]["img"] = i
# df = pd.concat(dfs)
# # %%

# sns.histplot(data=df, x="count", log_scale=True)

# # %%
# fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
# plt.imshow(mask)
# plt.scatter(spots["xcentroid"], spots["ycentroid"], s=1, c="r", alpha=0.5, marker="x")
# plt.axis("off")
# # %%
# import polars as pl

# for name in ["ctrl", "0peg", "75peg", "15peg"]:
#     df = pl.scan_csv(f"/disk/chaichontat/mer/peg/{name}/*spots.csv").collect()
#     print(name, len(df), df["flux"].sum())


# # %%
# im = img[2, 500:1000, 500:1000]
# sp = find_spots(im, threshold_sigma=10, fwhm=4)
# print(len(sp))
# fig, axs = plt.subplots(figsize=(8, 4), dpi=200, ncols=2)
# axs = axs.flatten()
# axs[0].imshow(np.log(im), cmap="magma")

# axs[1].imshow(im, cmap="magma", vmax=6000)
# axs[1].scatter(sp["xcentroid"], sp["ycentroid"], s=0.5, c="green", alpha=0.5, marker="x")

# # axs[2].imshow(mask[500:1000, 500:1000], cmap="gray")
# # axs[2].scatter(sp["xcentroid"], sp["ycentroid"], s=1, c="magenta", alpha=0.5, marker="x")


# [ax.axis("off") for ax in axs]
# # %%
# sp = find_spots(img[2], threshold_sigma=8, fwhm=4)
# # %%
# plt.hist(sp["peak"], bins=50)
# # %%
# fig, ax = plt.subplots(dpi=200)
# plt.imshow(mask, cmap="gray")
# sp = find_spots(img[2], threshold_sigma=8, fwhm=4)
# plt.scatter(sp["xcentroid"], sp["ycentroid"], s=1, c="magenta", alpha=0.5, marker="x")

# # %%
