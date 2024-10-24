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

from fishtools.analysis.tileconfig import TileConfiguration

path = Path("/mnt/working/e155trcdeconv/registered--left/")
codebook = "tricycleplus"
spots = pl.read_parquet(path / codebook / "spots.parquet")
sns.set_theme()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
ax.scatter(spots["x"][::50], spots["y"][::50], s=0.5, alpha=0.3)

# %%
pergene = (
    spots.groupby("target")
    .count()
    .sort("count", descending=True)
    .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
    .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
)
# %%

# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=200)
ax.bar(pergene["target"], pergene["count"], color=pergene["color"], width=1, align="edge", linewidth=0)
ax.set_xticks([])
ax.set_yscale("log")
ax.set_xlabel("Gene")
ax.set_ylabel("Count")
plt.tight_layout()

# %%
cb_raw = json.loads(Path(f"/home/chaichontat/fishtools/starwork3/ordered/{codebook}.json").read_text())

cb = (
    pl.DataFrame(cb_raw)
    .transpose(include_header=True)
    .with_columns(concat_list=pl.concat_list("column_0", "column_1", "column_2").cast(pl.List(pl.UInt8)))
    .drop(["column_0", "column_1", "column_2"])
    .rename({"column": "gene", "concat_list": "bits"})
)

joined = spots.join(cb, left_on="target", right_on="gene", how="left")
# %%
bits = set(chain.from_iterable(cb["bits"]))

# %%
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
