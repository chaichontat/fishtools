# %%
"""
Look at blank spots across channels.
"""

import subprocess
from itertools import chain
from pathlib import Path
from shutil import rmtree

import numpy as np
import tifffile

from fishtools.preprocess.fiducial import find_spots

# %%
img = tifffile.imread("/mnt/working/e155_trc/analysis/deconv/registered--right/reg-00100.tif")
center = img.shape[0] // 2


path = Path("foroptimize")
path.mkdir(exist_ok=True)
for i in range(img.shape[1]):
    tifffile.imwrite(
        path / f"foroptimize-{i:02d}.tif",
        img[center - 10 : center + 10, i],
        compression=22610,
        compressionargs={"level": 0.75},
        imagej=True,
        metadata={"axes": "ZYX"},
    )
# subprocess.run(["zip", "-r", "foroptimize.zip", "foroptimize"])
# rmtree(path)
# %%
import skimage

skimage.exposure.equalize_adapthist(img[center - 10 : center + 10, i])


# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
# %%
plt.imshow(img[0, 5][500:1000, 500:1000], zorder=1, vmax=0.03)

# %%
plt.imshow(img[0, 5])
# %%

import pickle

d = pickle.loads(
    Path(
        "/mnt/working/20241113-ZNE172-Zach/analysis/deconv/registered--right/genestar/reg-0056.pkl"
    ).read_bytes()
)
# %%
path = "/mnt/working/20241113-ZNE172-Zach/analysis/deconv/registered--right/reg-0056.tif"
img = tifffile.imread(path)
with tifffile.TiffFile(path) as tif:
    img_keys = tif.shaped_metadata[0]["key"]

bit_mapping = {k: i for i, k in enumerate(img_keys)}
mapping_bit = {v: k for k, v in bit_mapping.items()}

import json

codebook = json.loads((Path.home() / "fishtools/starwork3/ordered/genestar.json").read_text())
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

plt.imshow(img[30, -2], zorder=1)


# %%
oks = d[0][d[0].coords["passes_thresholds"]]
# %%
blanks = oks[oks.coords["target"].str.startswith("Blank")]

# %%


used_bits = list(
    filter(
        lambda x: x is not None,
        map(lambda x: bit_mapping.get(str(x), None), sorted(set(chain.from_iterable(codebook.values())))),
    )
)
# %%
# sorted(((k, sorted(v)) for k,v in codebook.items()), key=lambda x: x[1])

import polars as pl
from shapely import Point, Polygon
from shapely.strtree import STRtree

trees: dict[str, STRtree] = {}
spotss: dict[str, pl.DataFrame] = {}

for (name, i), _ in zip(bit_mapping.items(), range(27)):
    print(name, i)
    spots = find_spots(img[8, i], threshold_sigma=1, fwhm=5)
    points = []
    for spot in spots.iter_rows(named=True):
        points.append(Point((spot["ycentroid"], spot["xcentroid"])))
    spotss[name] = spots
    trees[name] = STRtree(points)
# %%


def plot_blank(img, coords, name, margin=51):
    z, y, x = coords
    bits = sorted(codebook[name])
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(12, 10), dpi=200, facecolor="black")
    axs = axs.flatten()
    windows = []
    for i, (ax, u) in enumerate(zip(axs, used_bits)):
        ax.axis("off")
        windows.append(img[z, u, y - margin : y + margin, x - margin : x + margin])
        ax.imshow(windows[-1], zorder=1)
        ax.axhline(margin, color="red")
        ax.axvline(margin, color="red")
        ax.set_title(mapping_bit[u], color="white")
    plt.tight_layout()
    return windows


# want = blanks[blanks.coords["z"] == 8][4]
want = oks[oks.coords["z"] == 8][90]
z, y, x = want.coords["z"].item(), want.coords["y"].item(), want.coords["x"].item()
windows = plot_blank(
    img,
    (z, y, x),
    want.coords["target"].item(),
    margin=8,
)
print(want.coords["yc"].item(), want.coords["xc"].item(), want.coords["distance"].item())
want_bits = sorted(codebook[want.coords["target"].item()])
print(want.coords["target"].item(), want_bits)

# %%
import pandas as pd

_d = pd.DataFrame({i: w.flatten() for i, w in enumerate(windows)}).corr()
plt.matshow(_d, zorder=1, cmap="bwr_r", vmax=1, vmin=-1)


# %%
def find(name: str, y: float, x: float):
    res = trees[name].query(Point(y, x).buffer(3), predicate="contains")
    if not len(res):
        return None
    print(name, spotss[name][res.astype(int)])


for bit in want_bits:
    find(str(bit), y, x)
# %%
trees["9"].query(Point(y, x).buffer(3), predicate="contains")
# %%
trees["19"].query(Point(y, x).buffer(3), predicate="contains")
# %%
img = tifffile.imread("/mnt/working/20241113-ZNE172-Zach/analysis/deconv/stitch--right/fused.tif")

# %%
maxed = img.max(axis=0)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 8), dpi=200, facecolor="black")
names = ["EdU", "CFSE", "γ-tubulin", "PI", "WGA"]
for i, ax in enumerate(axs.flat):
    ax.imshow(maxed[i], zorder=1, vmin=np.percentile(maxed[i], 50), vmax=np.percentile(maxed[i], 99))
    ax.set_title(names[i], color="white")
    ax.axis("off")

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()

# %%

# %%
