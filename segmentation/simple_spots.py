# %%
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table.table import QTable
from photutils.detection import DAOStarFinder

from fishtools.analysis.spots import load_spots


def find_spots(
    data: np.ndarray[np.uint16, Any], threshold: float = 100, fwhm: float = 4, median: int | None = None
) -> QTable | None:
    if median is None:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    iraffind = DAOStarFinder(threshold=threshold, fwhm=fwhm)
    return iraffind(np.clip(data, a_min=median, a_max=65535) - median)


# %%
img = tifffile.imread(
    "/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina/reg-0045.tif"
)
maxed = img[-8:].max(axis=0)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.imshow(img[-8:, 5].max(axis=0), zorder=1)
# %%
# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
sl = np.s_[:, :]
spots = find_spots(maxed[13, *sl], threshold=100, fwhm=6, median=1000)
ax.imshow(maxed[13, *sl], cmap="magma", zorder=1, vmax=np.percentile(maxed[2], 99.99))
ax.plot(spots["xcentroid"], spots["ycentroid"], "x", c="green", alpha=0.5)
# %%
import polars as pl

paths = sorted(
    Path("/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina").glob("reg-*.tif")
)
# %%
for p in sorted(paths):
    with tifffile.TiffFile(p) as tif:
        keys = tif.shaped_metadata[0]["key"]
        img = tif.asarray()
    maxed = img[-8:].max(axis=0)
    print(p)
    for c in range(0, maxed.shape[0]):
        spots = find_spots(maxed[c], threshold=100, fwhm=6, median=1000)
        pl.DataFrame(spots.to_pandas()).with_columns(
            channel=pl.lit(c), tile=pl.lit(int(p.stem.split("-")[1]))
        ).write_parquet(p.parent / f"{p.stem}_c{c}_spots.parquet")

# %%
import polars as pl
import json

cb = json.loads(Path(f"/home/chaichontat/fishtools/starwork3/ordered/alina.json").read_text())


def map_idx_bit(x: int):
    return next(k for k, v in cb.items() if int(keys[x]) in v)


# %%

for p in paths:
    if p.suffix != ".tif":
        continue
    print(p)
    df = pl.scan_parquet(
        f"/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina/{p.stem}_c*_spots.parquet"
    ).collect()
    (p.parent / "decoded-alina").mkdir(exist_ok=True, parents=True)
    out = df.with_columns(
        passes_thresholds=pl.lit(True),
        tile=pl.lit(p.stem.split("-")[1]),
        channel=pl.col("channel").cast(pl.Utf8),
        target=pl.col("channel").map_elements(map_idx_bit, return_dtype=pl.Utf8),
    ).rename({
        "xcentroid": "x",
        "ycentroid": "y",
        "flux": "norm",
    })
    out.write_parquet(p.parent / "decoded-alina" / f"{p.stem}.parquet")

# %%

# %%
import pickle

d, y, *_ = pickle.loads(
    Path(
        "/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina/decoded-alina/reg-0019.pkl"
    ).read_bytes()
)
t = pl.DataFrame(y)
# %%
from tifffile import TiffFile, imread

with TiffFile(
    "/archive/starmap/barrel/barrel_thicc1/analysis/deconv/registered--barrel+alina/reg-0019.tif"
) as tif:
    keys = tif.shaped_metadata[0]["key"]
    img = tif.asarray().max(axis=0)
# %%
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), dpi=200)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.axis("off")

# cb = json.loads(Path(f"/home/chaichontat/fishtools/starwork3/ordered/alina.json").read_text())
axs[0].imshow(img[keys.index("13")], zorder=1, vmax=2000)
s = spots.filter(pl.col("target") == "Fos-201").filter(pl.col("tile") == "0045")
axs[1].scatter(s["x"], s["y"], s=1, alpha=0.5)
