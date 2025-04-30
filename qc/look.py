# %%
"""
Look at blank spots across channels.
"""

import subprocess
from itertools import chain
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fishtools.postprocess import jitter

sns.set_theme()

# %%
import polars as pl
import tifffile

from fishtools.preprocess.fiducial import find_spots

old = pl.read_parquet("/working/20250226_AMH_oct01/analysis/deconv/brain2old.parquet")
new = pl.read_parquet("/working/20250226_AMH_oct01/analysis/deconv/octfull--brain2+octfull.parquet")
# %%
df = pl.DataFrame({"target": old["target"].unique().sort()})
df = df.join(new.group_by("target").len("new"), on="target").join(
    old.group_by("target").len("old"), on="target"
)
# %%
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
ax.scatter(df["old"], df["new"], s=1, alpha=0.5, color=sns.color_palette()[1])
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot([0, df["new"].max()], [0, df["new"].max()], alpha=0.5, color="gray")
ax.set_aspect("equal")
ax.set_title("Number of spots per gene", loc="left")
ax.set_xlabel("Old")
ax.set_ylabel("New")


avg_fold = np.exp(np.log(df["new"] / df["old"]).mean())
ax.text(0.05, 0.95, f"Geometric fold diff: {avg_fold:.2f}x", transform=ax.transAxes, verticalalignment="top")


# %%
img = tifffile.imread(
    "/working/20250317_benchmark_mousecommon/analysis/deconv/registered--center+mousecommon/reg-0041.tif"
)

# %%
with tifffile.TiffFile(
    f"/working/20250317_benchmark_mousecommon/analysis/deconv/registered--center+mousecommon/reg-0041.tif"
) as tif:
    img = tif.asarray()
    metadata = tif.shaped_metadata[0]

keys = metadata["key"]
img560 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and int(k) <= 8]]
img650 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and 8 < int(k) <= 16]]
img750 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and 16 < int(k) <= 24]]


rgb = np.zeros((*img560.shape[2:], 3))
rgb[..., 0] = img560.max(axis=(0, 1)) / img560.max() if img560.size > 0 else 0
# rgb[..., 1] = img650.max(axis=(0,1)) / img650.max() if img650.size > 0 else 0
# rgb[..., 2] = img750.max(axis=(0,1)) / img750.max() if img750.size > 0 else 0
# %%
with tifffile.TiffFile(f"/working/20250317_benchmark_mousecommon/analysis/deconv/reg-0041.tif") as tif:
    img = tif.asarray()
    metadata = tif.shaped_metadata[0]

keys = metadata["key"]
img560 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and int(k) <= 8]]
img650 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and 8 < int(k) <= 16]]
img750 = img[:, [i for i, k in enumerate(keys) if k.isdigit() and 16 < int(k) <= 24]]
rgb[..., 1] = img560.max(axis=(0, 1)) / img560.max() if img560.size > 0 else 0
# rgb[..., 0] = img750.max(axis=(0,1)) / img750.max() if img750.size > 0 else 0

# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
# %%
# %%

import pickle

path = Path("/working/20250411_2957/analysis/deconv")
roi = "hippo"
codebook = "morris"
idx = 55

d = pickle.loads(
    Path(path / f"registered--{roi}+{codebook}/decoded-{codebook}/reg-{idx:04d}-0.pkl").read_bytes()
)

d = pickle.loads(
    Path(
        "/working/20250423_DPE00199_mpeg/analysis/deconv/opt_Dpe/reg-0045--top_left+Dpe_opt01.pkl"
    ).read_bytes()
)

area = np.array(d[1])[d[0].coords["spot_id"].to_numpy()]
# print(len(oks), np.unique(d[0].coords["target"].to_numpy()))

# %%


# %%

# %%

plt.scatter(
    jitter(np.log([x["area"] for x in area]), 0.1),
    np.log(np.linalg.norm(d[0].to_numpy().squeeze(), axis=1)),
    s=3,
    alpha=0.5,
    cmap="bwr",
    c=d[0].coords["target"].str.startswith("Blank"),
)


# %%
import tifffile

with tifffile.TiffFile(path / f"registered--{roi}+{codebook}/reg-{idx:04d}.tif") as tif:
    img = tif.asarray()
    img_keys = tif.shaped_metadata[0]["key"]

with tifffile.TiffFile(
    "/working/20250423_DPE00199_mpeg/analysis/deconv/registered--bottom_right+Dpe/reg-0084.tif"
) as tif:
    img = tif.asarray()
    img_keys = tif.shaped_metadata[0]["key"]

bit_mapping = {k: i for i, k in enumerate(img_keys)}
mapping_bit = {v: k for k, v in bit_mapping.items()}

import json

cb = json.loads((Path.home() / "fishtools/starwork6/Dpe.json").read_text())
# %%
img = tifffile.imread(
    "/working/20250423_DPE00199_mpeg/analysis/deconv/registered--bottom_left+Dpe/_highpassed/reg-0023_Dpe.hp.tif"
)

# %%

# %%


# %%


used_bits = list(
    filter(
        lambda x: x is not None,
        map(lambda x: bit_mapping.get(str(x), None), sorted(set(chain.from_iterable(cb.values())))),
    )
)
# %%
# sorted(((k, sorted(v)) for k,v in codebook.items()), key=lambda x: x[1])

# import polars as pl
# from shapely import Point, Polygon
# from shapely.strtree import STRtree

# trees: dict[str, STRtree] = {}
# spotss: dict[str, pl.DataFrame] = {}

# for (name, i), _ in zip(bit_mapping.items(), range(27)):
#     print(name, i)
#     spots = find_spots(img[8, i], threshold_sigma=1, fwhm=5)
#     points = []
#     for spot in spots.iter_rows(named=True):
#         points.append(Point((spot["ycentroid"], spot["xcentroid"])))
#     spotss[name] = spots
#     trees[name] = STRtree(points)
# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def plot_blank(img, coords, name, margin=51, vmax_percentile=99.999):
    z, y, x = coords
    fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(8, 8), dpi=200, facecolor="black")
    axs = axs.flatten()
    windows = []
    correct = list(map(lambda x: bit_mapping[str(x)], cb[want.coords["target"].item()]))
    for i, (ax, u) in enumerate(zip(axs, used_bits)):
        ax.axis("off")
        windows.append(img[:, u, y - margin : y + margin, x - margin : x + margin].max(axis=0))
        ax.imshow(windows[-1], zorder=1, vmax=np.percentile(img[z, u], vmax_percentile))
        ax.axhline(margin, color="red", alpha=0.3)
        ax.axvline(margin, color="red", alpha=0.3)
        ax.set_title(mapping_bit[u], color="white")
        if u in correct:
            rect = plt.Rectangle(
                (0, 0),
                windows[-1].shape[1] - 1,
                windows[-1].shape[0] - 1,
                fill=False,
                color="green",
                linewidth=1,
                alpha=0.5,
            )
            ax.add_patch(rect)

    for ax in axs.flat:
        if not ax.has_data():
            fig.delaxes(ax)
    plt.tight_layout()
    return windows


idx = 34


oks = d[0][d[0].coords["passes_thresholds"]]
# oks = oks[
#     (np.linalg.norm(d[0].to_numpy().squeeze(), axis=1) > 0.2)
#     & (4 / 3 * np.pi * oks.coords["radius"] ** 3 > 12)
# ]
oks = oks[np.linalg.norm(oks.to_numpy().squeeze(), axis=1).__gt__(0.2)]
blanks = oks[oks.coords["target"].str.startswith("Blank")]
want = oks[idx]
want = blanks[idx]
z, y, x = want.coords["z"].item(), want.coords["y"].item(), want.coords["x"].item()
norm = np.linalg.norm(want) * (1 - want.coords["distance"])
windows = plot_blank(
    img,
    (z, y, x),
    want.coords["target"].item(),
    vmax_percentile=99.999,
    margin=6,
)
print(
    f"norm={norm:.4f}, area={4 / 3 * 3.14 * want.coords['radius'].item() ** 3}, dist={want.coords['distance'].item():.3f}"
)
want_bits = sorted(cb[want.coords["target"].item()])
print(want.coords["target"].item(), want_bits)

# %%
import pandas as pd

_d = pd.DataFrame({i: w.flatten() for i, w in enumerate(windows)}).corr()
plt.matshow(_d, zorder=1, cmap="bwr_r", vmax=1, vmin=-1)
plt.colorbar()


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
import zarr

f = zarr.open_array("/working/20250327_benchmark_coronal2/analysis/deconv/stitch--brain+polyA/half.zarr")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200, facecolor="black")
for ax, chan in zip(axs, range(0, 12, 4)):
    ax.imshow(f[chan, 11000:12000, 5000:5500], zorder=1)
# %%

fi = zarr.open_array(
    "/working/20250327_benchmark_coronal2/analysis/deconv/stitch--brain+polyA/input_image.zarr"
)


plt.imshow(fi[:, 11000:12000, 5000:5500, 1].max(axis=0), zorder=1, vmax=6000)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200, facecolor="black")
for ax, chan in zip(axs, range(0, 12, 4)):
    ax.imshow(fi[chan, 15800:16500, 18500:19500, 1], zorder=1)
# %%
from fishtools.utils.io import Workspace
from tifffile import imread

ws = Workspace("/working/20250411_2957")

u = imread(ws.img("2_10_18", "hippo", 5))

# %%
t = imread(ws.img("x34_polyA", "hippo", 5))

# %%
