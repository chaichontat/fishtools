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
from pydantic import BaseModel, TypeAdapter
from rtree import index
from shapely import Point, Polygon, contains, intersection
from tifffile import imread

from fishtools.preprocess.tileconfig import TileConfiguration

path = Path("/mnt/working/e155_trc/analysis/deconv/registered--right/")
codebook = "genestar"
spots = pl.read_parquet(path / codebook / "spots.parquet").with_columns(
    is_blank=pl.col("target").str.starts_with("Blank")
)
sns.set_theme()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
ax.set_aspect("equal")
ax.scatter(spots["x"][::50], spots["y"][::50], s=0.5, alpha=0.3)

# %%
_sp = spots.filter(pl.col("target") == "Blank-9")
plt.scatter(_sp["x"], _sp["y"], s=0.5, alpha=0.3)


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


pergene = count_by_gene(spots)


# %%
def plot_scree(pergene: pl.DataFrame):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=200)
    ax.bar(pergene["target"], pergene["count"], color=pergene["color"], width=1, align="edge", linewidth=0)
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_xlabel("Gene")
    ax.set_ylabel("Count")
    return fig, ax


plot_scree(count_by_gene(spots.filter(pl.col("area").is_between(20, 50))))
plt.tight_layout()

# %%
spots = spots.filter(pl.col("norm") > 0.003)
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
rand = np.random.default_rng(0)
subsample = 1  # max(1, len(spots) // 50000)
ax.hexbin(
    (spots[::subsample]["area"]) ** (1 / 2) + rand.normal(0, 1, size=spots[::subsample].__len__()),
    np.log10(spots[::subsample]["norm"]),
)
ax.set_xlabel("Radius")
ax.set_ylabel("log10(norm)")

# %%


# %%

spots = spots.filter(pl.col("norm") > 0.003)
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
rand = np.random.default_rng(0)
subsample = 1  # max(1, len(spots) // 50000)
ax.hexbin(
    (spots.filter("is_blank")[::subsample]["area"]) ** (1 / 2)
    + rand.normal(0, 1, size=spots.filter("is_blank")[::subsample].__len__()),
    np.log10(spots.filter("is_blank")[::subsample]["norm"]),
)
ax.set_xlabel("Radius")
ax.set_ylabel("log10(norm)")

# %%
cb_raw = json.loads(Path(f"/home/chaichontat/fishtools/starwork3/ordered/{codebook}.json").read_text())

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

joined = spots.join(cb, left_on="target", right_on="gene", how="left")
# %%
np.array(np.unique(np.array(cb.filter(pl.col("is_blank"))["bits"].to_list()).flatten(), return_counts=True))

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
    ax.scatter(filtered["x"][::10], filtered["y"][::10], s=0.3, alpha=0.1)
plt.tight_layout()

# %%


# %%


coords = TileConfiguration.from_file(
    # Path(path.parent / f"fids--{roi}" / "TileConfiguration.registered.txt")
    Path(path.parent / "fids--left" / "TileConfiguration.registered.txt")
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
