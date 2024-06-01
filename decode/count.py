# %%
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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

sns.set_theme()


def imread_page(path: Path | str, page: int):
    with TiffFile(path) as tif:
        if page >= len(tif.pages):
            raise ValueError(f"Page {page} does not exist in {path}")
        return tif.pages[page].asarray()


def find_spots(data: np.ndarray[np.uint16, Any], threshold_sigma: float = 2.5, fwhm: float = 4) -> QTable:
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    iraffind = DAOStarFinder(threshold=threshold_sigma * std, fwhm=fwhm, exclude_border=True)
    return iraffind(np.clip(data, a_min=median, a_max=65535) - median)


# %%


def run(path: Path | str, c: int, threshold_sigma: float = 6, fwhm: float = 4):
    path = Path(path)
    print(path.stem)
    img = imread(path)[:-1].reshape(-1, 4, 2048, 2048).max(axis=0)
    sp = find_spots(img[c], threshold_sigma=threshold_sigma, fwhm=fwhm)
    df = sp.to_pandas().set_index("id")
    df.to_csv(path.parent / f"{path.stem}_c{c}_spots.csv")
    return img, df


# %%
with ThreadPoolExecutor(4) as exc:
    for path in Path("/disk/chaichontat/mer/peg").rglob("*.tif"):
        exc.submit(run, path, 2)
# %%
from collections import defaultdict

import pandas as pd

path = Path("/disk/chaichontat/mer/peg/0peg")
dfs = []
for i in range(22, 23):
    mask = imread(path / (f"dapi_polyA_malat_cux-{i:03d}_max_cp_masks.tif"))
    img, spots = run(path / f"dapi_polyA_malat_cux-{i:03d}.tif", 2)
    counts = defaultdict(int)
    for s in spots.itertuples():
        x, y = s.xcentroid, s.ycentroid
        counts[mask[round(y), round(x)]] += 1
    dfs.append(pd.DataFrame(counts.items(), columns=["img", "count"]))
    dfs[-1]["img"] = i
df = pd.concat(dfs)
# %%

sns.histplot(data=df, x="count", log_scale=True)

# %%
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
plt.imshow(mask)
plt.scatter(spots["xcentroid"], spots["ycentroid"], s=1, c="r", alpha=0.5, marker="x")
plt.axis("off")
# %%
import polars as pl

for name in ["ctrl", "0peg", "75peg", "15peg"]:
    df = pl.scan_csv(f"/disk/chaichontat/mer/peg/{name}/*spots.csv").collect()
    print(name, len(df), df["flux"].sum())


# %%
im = img[2, 500:1000, 500:1000]
sp = find_spots(im, threshold_sigma=10, fwhm=4)
print(len(sp))
fig, axs = plt.subplots(figsize=(8, 4), dpi=200, ncols=2)
axs = axs.flatten()
axs[0].imshow(np.log(im), cmap="magma")

axs[1].imshow(im, cmap="magma", vmax=6000)
axs[1].scatter(sp["xcentroid"], sp["ycentroid"], s=0.5, c="green", alpha=0.5, marker="x")

# axs[2].imshow(mask[500:1000, 500:1000], cmap="gray")
# axs[2].scatter(sp["xcentroid"], sp["ycentroid"], s=1, c="magenta", alpha=0.5, marker="x")


[ax.axis("off") for ax in axs]
# %%
sp = find_spots(img[2], threshold_sigma=8, fwhm=4)
# %%
plt.hist(sp["peak"], bins=50)
# %%
fig, ax = plt.subplots(dpi=200)
plt.imshow(mask, cmap="gray")
sp = find_spots(img[2], threshold_sigma=8, fwhm=4)
plt.scatter(sp["xcentroid"], sp["ycentroid"], s=1, c="magenta", alpha=0.5, marker="x")

# %%
