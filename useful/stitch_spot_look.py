# %%
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyparsing as pp
import seaborn as sns
from loguru import logger
from pydantic import BaseModel, TypeAdapter
from rtree import index
from scipy.stats import ecdf
from shapely import Point, Polygon, contains, intersection
from tifffile import TiffFile, imread

from fishtools.analysis.spots import load_spots
from fishtools.preprocess.tileconfig import TileConfiguration

path = Path("/working/20250131_bigprobenotmangled/analysis/deconv")
codebook = "mousecommon"
roi = f"full+{codebook}"

# Baysor
# spots.filter(
#             pl.col("area").is_between(18, 50) & pl.col("norm").gt(0.03)
#         ).select("x", "y", "z", gene=pl.col("target").str.split("-").list.get(0)).write_csv(path / "leftcortex.csv")

first = next((path / f"registered--{roi}").glob("*.tif"))
with TiffFile(first) as tif:
    names = tif.shaped_metadata[0]["key"]
    mapping = dict(zip(names, range(len(names))))

print(mapping)
# pick = 100
# path = list((path / f"registered--{roi}" / f"decoded-{codebook}").glob("*.pkl"))[pick]
# spots = load_spots(path, idx=0, filter_=True).with_columns(is_blank=pl.col("target").str.starts_with("Blank"))

spots = (
    pl.scan_parquet(
        [
            path / f"registered--{roi}" / f"decoded-{codebook}" / "spots.parquet",
            # path / f"registered--{roi}" / f"decoded-genestar" / "spots.parquet",
        ]
    ).with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
    # .filter(pl.col("x").gt(7000) & pl.col("y").lt(-7500))
    .collect()
)
print(len(spots))
sns.set_theme()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
ax.scatter(spots["x"][::20], spots["y"][::20], s=0.1, alpha=0.3)
ax.set_aspect("equal")
# %%


# %%
# _sp = spots.filter(pl.col("target") == "Blank-9")
# plt.scatter(_sp["x"], _sp["y"], s=0.5, alpha=0.3)


# %%
# %%
def count_by_gene(spots: pl.DataFrame):
    return (
        spots.group_by("target")
        .len("count")
        .sort("count", descending=True)
        .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
        .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
    )


# pergene = count_by_gene(spots)
spots_ = spots.filter(pl.col("area").is_between(13, 100))


# %%
# fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
# rand = np.random.default_rng(0)
# subsample = 1  # max(1, len(spots) // 50000)
# x = (spots_[::subsample]["area"]) ** (1 / 3) + rand.normal(0, 0.05, size=spots_[::subsample].__len__())
# y = np.log10(spots_[::subsample]["norm"])
# ax.hexbin(x, y)
# ax.set_xlabel("Radius")
# ax.set_ylabel("log10(norm)")
# %%

# %%

# spots_ = spots.filter(pl.col("norm").gt(0.01) & pl.col("area").is_between(16, 200))
# fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
# rand = np.random.default_rng(0)
# subsample = 1  # max(1, len(spots) // 50000)
# ax.hexbin(
#     (spots_[::subsample]["area"]) ** (1 / 2) + rand.normal(0, 0.1, size=spots_[::subsample].__len__()),
#     spots_[::subsample]["distance"],
# )
# ax.set_xlabel("Radius")
# ax.set_ylabel("Distance")

# %%
spots_ = spots_.with_columns(
    x_=(pl.col("area") + np.random.uniform(-1, 1)) ** (1 / 3), y_=pl.col("norm").log10()
).filter(pl.col("y_") > -1.9)
bounds = spots_.select(
    [
        pl.col("x_").min().alias("x_min"),
        pl.col("x_").max().alias("x_max"),
        pl.col("y_").min().alias("y_min"),
        pl.col("y_").max().alias("y_max"),
    ]
).row(0, named=True)
# bounds["y_min"] = -2.1
n = 50
x = np.linspace(bounds["x_min"], bounds["x_max"], n)
y = np.linspace(bounds["y_min"], bounds["y_max"], n)
step_x = x[1] - x[0]
step_y = y[1] - y[0]
X, Y = np.meshgrid(x, y)
result = (
    spots_.with_columns(
        [
            ((pl.col("x_") - bounds["x_min"]) / step_x).floor().alias("i"),
            ((pl.col("y_") - bounds["y_min"]) / step_y).floor().alias("j"),
        ]
    )
    .filter((pl.col("i") >= 0) & (pl.col("i") < n - 1) & (pl.col("j") >= 0) & (pl.col("j") < n - 1))
    .group_by(["i", "j"])
    .agg([pl.sum("is_blank").alias("blank_count"), pl.count().alias("total_count")])
    .with_columns((pl.col("blank_count") / (pl.col("total_count") + 1)).alias("proportion"))
)

# Convert to numpy array
Z = np.zeros_like(X)
for row in result.iter_rows(named=True):
    Z[int(row["j"]), int(row["i"])] = row["proportion"]  # * np.log1p(row["total_count"])
from scipy.ndimage import gaussian_filter

Z_smooth = gaussian_filter(Z, sigma=3)
plt.pcolormesh(X, Y, Z)
contours = plt.contour(X, Y, Z_smooth, colors="black", levels=50)  # Add line contours
plt.colorbar()


# %%

from scipy.interpolate import RegularGridInterpolator

interp_func = RegularGridInterpolator((y, x), Z_smooth)
# threshold = contours.levels[5]


# %%
def filter_threshold(thr: int):
    threshold = contours.levels[thr]
    point_densities = interp_func(spots_.select(["y_", "x_"]).to_numpy())
    _sp = spots_.with_columns(point_density=point_densities).with_columns(
        ok=pl.col("point_density") < threshold
    )
    spots_ok = _sp.filter(pl.col("ok"))
    print(len(spots_ok), len(spots_))
    return spots_ok


# %%
x, y = [], []
for i in range(1, 15):
    print(i, end=" ")
    sp = filter_threshold(i)
    x.append(len(sp))
    y.append(sp.filter(pl.col("is_blank")).__len__() / len(sp))

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.plot(list(range(1, 15)), x)
ax2 = ax.twinx()
ax2.plot(list(range(1, 15)), y)
ax.set_ylim(0, None)
# %%

spots_ok = filter_threshold(6)
# spots_ok = spots.filter(pl.col("area").is_between(18, 100) & pl.col("norm").gt(0.02))
# %%
# density_result = (
#     spots_.with_columns([
#         ((pl.col("x_") - bounds["x_min"]) / step_x).floor().alias("i"),
#         ((pl.col("y_") - bounds["y_min"]) / step_y).floor().alias("j"),
#     ])
#     .filter((pl.col("i") >= 0) & (pl.col("i") < 49) & (pl.col("j") >= 0) & (pl.col("j") < 49))
#     .group_by(["i", "j"])
#     .agg([pl.count().alias("density")])
# )

# # Convert to numpy array
# Z_density = np.zeros_like(X)
# for row in density_result.iter_rows(named=True):
#     Z_density[int(row["j"]), int(row["i"])] = row["density"]

# plt.pcolormesh(X, Y, Z_density, norm="linear")
# plt.colorbar()


def plot_scree(pergene: pl.DataFrame, limit: int = 5):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=200)
    print(
        pergene.filter(pl.col("target").str.starts_with("Blank"))["count"].sum() / pergene["count"].sum(),
        pergene["count"].sum(),
    )
    if limit:
        pergene = pergene.filter(pl.col("count") > limit)
    ax.bar(pergene["target"], pergene["count"], color=pergene["color"], width=1, align="edge", linewidth=0)
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Count")
    return fig, ax


plot_scree(count_by_gene(spots_ok))


# %%
if codebook == "tricycleplus":
    genes = ["Ccnd3", "Mcm5", "Top2a", "Pou3f2", "Cenpa", "Hells"]
else:
    genes = ["Eomes", "Tbr1", "Bcl11b", "Fezf2", "Neurod6", "Notch1", "Notch2"]
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 12), dpi=200, facecolor="black")

for ax, gene in zip(axs.flat, genes):
    selected = spots_.filter(pl.col("target").str.starts_with(gene))
    if not len(selected):
        raise ValueError(f"No spots found for {gene}")
    ax.set_aspect("equal")
    ax.set_title(gene, fontsize=16, color="white")
    # ax.hexbin(selected["x"], -selected["y"], gridsize=400, cmap="magma")
    ax.scatter(selected["x"], selected["y"], s=5000 / len(selected), alpha=0.1)
    ax.axis("off")

for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)
# %%


# %%
cb_raw = json.loads(Path(f"/home/chaichontat/fishtools/starwork6/{codebook}.json").read_text())

cb = (
    pl.DataFrame(cb_raw)
    .transpose(include_header=True)
    .with_columns(
        concat_list=pl.concat_list("column_0", "column_1", "column_2").cast(pl.List(pl.UInt8)),
        is_blank=pl.col("column").str.starts_with("Blank"),
    )
    .drop(["column_0", "column_1", "column_2"])
    .rename({"column": "gene", "concat_list": "bits"})
)

joined = spots_ok.join(cb, left_on="target", right_on="gene", how="left")

spots_ok.filter(~pl.col("is_blank")).write_parquet(path / f"{codebook}--{roi}.parquet")
# %%


# %%
plot = False
if plot:
    # spots_ = spots_.filter(pl.col("ok"))
    genes = spots_ok.group_by("target").len().sort("len", descending=True)["target"]
    genes = sorted(set(spots_["target"]))
    dark = False
    nrows = int(np.ceil(len(genes) / 20))
    fig, axs = plt.subplots(
        ncols=20, nrows=nrows, figsize=(72, int(2.4 * nrows)), dpi=160, facecolor="black" if dark else "white"
    )
    axs = axs.flatten()
    for ax, gene in zip(axs, genes):
        selected = spots_ok.filter(pl.col("target") == gene)
        if not len(selected):
            logger.warning(f"No spots found for {gene}")
            continue
        ax.set_aspect("equal")
        ax.set_title(gene, fontsize=16, color="white" if dark else "black")
        ax.axis("off")
        if dark:
            ax.hexbin(selected["x"], selected["y"], gridsize=250, cmap="magma")
        else:
            ax.scatter(selected["x"], selected["y"], s=2000 / len(selected), alpha=0.1)

    for ax in axs.flat:
        if not ax.has_data():
            fig.delaxes(ax)
    plt.tight_layout()

# %%

np.array(np.unique(np.array(cb.filter(pl.col("is_blank"))["bits"].to_list()).flatten(), return_counts=True))
# %% Blank abundance

joined.group_by("target").len().join(cb, left_on="target", right_on="gene", how="left").filter(
    pl.col("is_blank")
).sort("len", descending=True)
# %%


# %%
bits = sorted(set(chain.from_iterable(cb["bits"])))

# %%
# Check for spot distribution in each bit within a frame.
fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(12, 20), dpi=100)
axs = axs.flatten()
for i, (ax, bit) in enumerate(zip(axs, bits)):
    filtered = joined.filter(pl.col("bits").list.contains(bit))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title(f"{bit}")
    ax.hexbin(filtered["x_local"], filtered["y_local"], gridsize=100)
plt.tight_layout()

# %%
# Check for spot distribution in each bit across the whole image.
fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(12, 16), dpi=100)
axs = axs.flatten()
for i, (ax, bit) in enumerate(zip(axs, bits)):
    filtered = joined.filter(pl.col("bits").list.contains(bit))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title(f"{bit}")
    ax.scatter(filtered["x"][::50], filtered["y"][::50], s=0.3, alpha=0.1)
plt.tight_layout()


# %%
# %%
img = imread(path / f"stitch--{roi}" / "fused.tif").max(axis=0)
# %%

fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(8, 4), dpi=300, facecolor="black")
for i, ax in enumerate(axs.flat):
    if i >= 5:
        break
    ax.axis("off")
    ax.imshow(
        img[-(i + 1), ::-4, ::-4].T,
        zorder=1,
        vmin=np.percentile(img[-(i + 1)], 40),
        vmax=np.percentile(img[-(i + 1)], 99.9),
        cmap="inferno",
        origin="lower",
    )
for ax in axs.flat:
    if not ax.has_data():
        fig.delaxes(ax)

plt.tight_layout()

# %%


coords = TileConfiguration.from_file(
    # Path(path.parent / f"fids--{roi}" / "TileConfiguration.registered.txt")
    Path(path / f"stitch--{roi}" / "TileConfiguration.registered.txt")
).df


fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
filtered = joined.filter(pl.col("bits").list.contains(1))
ax.axis("off")
ax.set_aspect("equal")
ax.scatter(filtered["x"][::10], filtered["y"][::10], s=0.3, alpha=0.1)
for c in coords.iter_rows(named=True):
    ax.text(c["x"] + 1788 / 2, c["y"] + 1788 / 2, c["index"], fontsize=6)

# %%
if codebook == "tricycleplus":
    genes = ["Ccnd3", "Mcm5", "Top2a", "Pou3f2", "Cenpa", "Hell"]
else:
    genes = ["Sox9", "Pax6", "Eomes", "Tbr1", "Bcl11b", "Fezf2", "Neurod6", "Notch1", "Notch2"]

genes = pergene["target"]
fig, axs = plt.subplots(ncols=3, nrows=6, figsize=(12, 18), dpi=200, facecolor="white")
axs = axs.flatten()
for ax, gene in zip(axs, genes):
    selected = spots.filter(pl.col("target").str.starts_with(gene))
    if not len(selected):
        raise ValueError(f"No spots found for {gene}")
    ax.set_aspect("equal")
    ax.set_title(gene, fontsize=16, color="black")
    hb = ax.hexbin(selected["x_local"], selected["y_local"], gridsize=50).get_array()
    print(f"{gene}: {np.mean(hb):.2f} CV: {np.std(hb) / np.mean(hb):.4f}")
    # ax.scatter(selected["x_local"], selected["y_local"], s=5000 / len(selected), alpha=0.2)
    ax.axis("off")
# %%
curr = joined.filter(pl.col("bits").list.contains(5))


def combi_value():
    import xarray as xr

    ds = []
    for i in range(100):
        if not (path / "genestar" / f"reg-{i:04d}.pkl").exists():
            continue
        d, y = pickle.loads((path / "genestar" / f"reg-{i:04d}.pkl").read_bytes())
        d = d[d.coords["passes_thresholds"]]
        ds.append(d)

    return xr.concat(ds, dim="features")


# ds = combi_value()


# %%

vals = np.nan_to_num(np.log10(ds[:, :, bits.index(4)]), neginf=0).flatten()
is_blank = ds.coords["target"].str.startswith("Blank")
# %%
plt.hist(vals[~is_blank], bins=100, log=True)
plt.hist(vals[is_blank], bins=100, log=True)

# %%
plt.hist(ds.coords["radius"][~is_blank], bins=100, log=True)
plt.hist(ds.coords["radius"][is_blank], bins=100, log=True)


# %%
perc = np.percentile(vals[is_blank], 50)
# %%
np.sum(np.array(vals[~is_blank]) > perc)
# %%
np.percentile(vals[~is_blank], 50)
# %%
out = []
for p in [1, 10, 50, 90, 99]:
    out.append({"p": p, "val": np.percentile(vals[~is_blank], p), "is_blank": False})
    out.append({"p": p, "val": np.percentile(vals[is_blank], p), "is_blank": True})
out = pl.DataFrame(out)
# %%
sns.barplot(x="p", y="val", hue="is_blank", data=out, log_scale=True)
# %%
vdf = pl.DataFrame(dict(vals=np.log(np.array(vals).flatten()), is_blank=np.array(is_blank)))
sns.histplot(x="vals", hue="is_blank", data=vdf, bins=100)
plt.gca().set_yscale("log")
# %%

# sns.histplot(x="area", hue="is_blank", data=spots[::10])
# plt.gca().set_yscale("log")
# plt.gca().set_xscale("log")
# %%
for i in [8, 10, 15, 20, 30]:
    fed = spots.filter(pl.col("area") >= i)
    print(i, len(fed), fed.filter(pl.col("is_blank")).shape[0] / len(fed))

# %%
plt.plot(
    spots_ok.with_columns(z=pl.col("z").cast(pl.UInt16))
    .group_by("z")
    .agg(blank=pl.col("is_blank").sum() / pl.len())
    .sort("z")["blank"]
)

# %%

files = list(
    Path("/working/20250131_bigprobenotmangled/analysis/deconv/registered--full+zachDE/_highpassed").glob(
        "*.tif"
    )
)

with TiffFile(files[15]) as tif:
    img = tif.asarray()

from tifffile import imwrite

imwrite("out_goof.tif", img.squeeze()[:, 5], compression=22610)
# %%
