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
from fishtools.preprocess.cli_register import run_fiducial
from fishtools.utils.plot import encode_labels_for_colormap, make_rgb, plot_img, tableau20_label_cmap

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

path = Path("/warm/raw/20251129_SV128_Fos-AI-7to9_plate4/analysis/deconv")
roi = "day1"
codebook = "alina_soma.good"
idx = 65

# /warm/raw/20251129_SV128_Fos-AI-7to9_plate4/analysis/deconv/registered--sham+alina_soma.good/decoded-alina_soma.good

d = pickle.loads(
    Path(path / f"registered--{roi}+{codebook}/decoded-{codebook}/reg-{idx:04d}-0.pkl").read_bytes()
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

# with tifffile.TiffFile(
#    path / f"registered--{roi}+{codebook}/reg-{idx:04d}.tif"
# ) as tif:
#     img = tif.asarray()
#     img_keys = tif.shaped_metadata[0]["key"]

bit_mapping = {k: i for i, k in enumerate(img_keys)}
mapping_bit = {v: k for k, v in bit_mapping.items()}
#%%
import json

cb = json.loads((path / "codebooks" / f"{codebook}.json").read_text())
# %%

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


idx = 11


oks = d[0][d[0].coords["passes_thresholds"]]
# oks = oks[
#     (np.linalg.norm(d[0].to_numpy().squeeze(), axis=1) > 0.2)
#     & (4 / 3 * np.pi * oks.coords["radius"] ** 3 > 12)
# ]
oks = oks[np.linalg.norm(oks.to_numpy().squeeze(), axis=1).__gt__(0.05)]
blanks = oks[oks.coords["target"].str.startswith("Blank-2")]
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
from fishtools import Workspace
from fishtools.utils.plot import plot_img

ws = Workspace("/working/20251001_JaxA3_Coro11")
# %%
plot_img(
    "/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2r+pi/fused.zarr",
    np.s_[10, ::4, ::4, 0],
)
# %%
import tifffile

img = tifffile.imread(
    "/working/20250929_JaxA3_Coro4/analysis/deconv/registered--1+cs_base/_sanity_field/canvas_raw.tif"
)
plt.imshow(
    img,
    zorder=1,
)
plt.colorbar()

# %%
img2 = tifffile.imread(
    "/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--1whole+edu/06/01/fused_01-1.tif"
)[::4, ::4]
plt.imshow(img, zorder=1)


# %%
u = make_rgb(
    img2,
    img,
    np.zeros_like(img),
)
# %%
plt.imshow(u / 65535, zorder=1)

# %%
# f = zarr.open_array("/working/20250327_benchmark_coronal2/analysis/deconv/stitch--brain+polyA/half.zarr")
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import zarr

sns.set_theme()
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200, facecolor="black")
for ax, chan in zip(axs, range(0, 12, 4)):
    ax.imshow(f[chan, 11000:12000, 5000:5500], zorder=1)
# %%
sl = np.s_[5, 4500:5000, 4000:5000]

mask = zarr.open_array(
    "/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2r+pi/output_segmentation.zarr"
)
mask = mask[sl]

# %%
img = zarr.open_array("/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2r+edu/fused.zarr")
img = img[*sl, 1]
# %%
plt.rcParams["figure.dpi"] = 300

cmap, norm, lut = tableau20_label_cmap(mask, fill_interiors=True, add_border=True, border_color=(1, 1, 1, 1))
index_img = encode_labels_for_colormap(mask, lut, border_label=-1, connectivity=8)
plt.imshow(index_img, cmap=cmap, norm=norm, zorder=1)
# %%
plt.imshow(img, zorder=1)
plt.imshow(index_img, cmap=cmap, norm=norm, zorder=1, alpha=0.3)

# plt.imshow(mask[5, 8000:9000, 8000:9000], zorder=1)

# %%
# fi *= mask[np.newaxis, :, :, np.newaxis]
# %%
fi = zarr.open_array("/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2l+pi/fused_n4.zarr")
fi2 = zarr.open_array(
    "/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2l+pi/output_segmentation.zarr", mode="r"
)

# %%
plt.imshow()

# %%
plot_img(
    "/working/20250929_JaxA3_Coro4/analysis/deconv/registered--1+cs_base/_sanity_field/canvas_corrected.tif"
)

# fi2[:] = fi[:]
# fi2.flush()
# %%
plt.rcParams["figure.dpi"] = 300
plt.imshow(fi2[15, 8000:10000, 9000:10000], zorder=1)
# %%

# %%

u = unsharp_mask(
    cp.asarray(fi[10, 9000:10000, 9000:10000, 0], dtype=np.float32), radius=2, preserve_range=True
)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200, facecolor="black")
for ax, chan in zip(axs, range(0, 12, 4)):
    ax.imshow(fi[chan, 15800:16500, 18500:19500, 1], zorder=1)
# %%
from tifffile import imread

from fishtools.utils.io import Workspace

ws = Workspace("/working/20250411_2957")

u = imread(ws.img("2_10_18", "hippo", 5))

# %%
t = imread(ws.img("x34_polyA", "hippo", 5))

# %%
from tifffile import imread
img_repaired = imread("/working/20251026_JaxA1_Sag6/analysis/deconv/wga_brdu--4--repaired/wga_brdu-0011.tif")[-2][:512, :512]
ref = imread("/working/20251026_JaxA1_Sag6/analysis/deconv/2_10_18--4/2_10_18-0011.tif")[-2][:512, :512]
img = imread("/working/20251026_JaxA1_Sag6/analysis/deconv/wga_brdu--4/wga_brdu-0011.tif")[-2][:512, :512]
# %%
import matplotlib.pyplot as plt
fig,axs= plt.subplots(1,3,figsize=(12,4),dpi=200)
axs[0].imshow(img, zorder=1, vmin=0, vmax=20000)
axs[0].set_title("Original")
axs[1].imshow(img_repaired, zorder=1, vmin=0, vmax=20000)
axs[1].set_title("Repaired")
axs[2].imshow(ref, zorder=1, vmin=0, vmax=20000)
axs[2].set_title("Reference")
# %%
img=imread("/working/20251026_JaxA1_Sag6/analysis/deconv/stitch--3--shifted-wga_brdu/fid/00/fused_00-1.tif")

# %%
from fishtools.preprocess.fiducial import phase_shift



shift_val = phase_shift(ref, img)

print(f"Fiducial shift = {shift_val}")
# Fiducial shift = [ 50.53 -59.5 ]
# %%
shift_val = phase_shift(ref, img_repaired)
print(f"Fiducial shift = {shift_val}")
# Fiducial shift = [0.03 0.14]
# %%
import pyvista as pv
m = pv.read("/working/20251001_JaxA3_Coro11/analysis/deconv/stitch--2l+pi/output_segmentation-sam_postproc_s1-2-2_v500.zarr/mesh.vtp")

#%%
p = pv.Plotter()
p.add_mesh(m)
p.show_axes()
p.camera.zoom(5.0)
p.show()

# %%
