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

spots = pl.read_parquet("/mnt/working/e155trcdeconv/registered--right/genestar/spots.parquet")
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
cb = (
    pl.DataFrame(json.loads(Path("/home/chaichontat/fishtools/starwork3/ordered/genestar.json").read_text()))
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
    ax.hexbin(filtered["x_local"], filtered["y_local"], gridsize=150)
plt.tight_layout()

# %%
