# %%
from concurrent.futures import ThreadPoolExecutor
import pickle
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

sns.set_theme()
path = Path("/mnt/working/e155trcdeconv/registered--left/")
codebook = "tricycleplus"
files = sorted(file for file in Path(path / codebook).glob("*.pkl") if "opt" not in file.stem)

coords = TileConfiguration.from_file(Path(path.parent / "fids--left" / "TileConfiguration.registered.txt")).df
assert len(files) == len(coords)
# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
sns.scatterplot(x="x", y="y", data=coords.to_pandas(), ax=ax, s=10, alpha=0.9)
# %%
# %%
# baddies = []
# for file in Path("/fast2/3t3clean/analysis/deconv/shifts").glob("shifts_*.json"):
#     with open(file, "r") as f:
#         shifts = json.load(f)

#     vals = np.array(list(shifts.values()))

#     if len(np.flatnonzero(np.square(vals).sum(axis=1))) < len(shifts) - 1:
#         print(shifts)
#         baddies.append(file)


# %%


def load(i: int):
    d, y = pickle.loads(files[i].read_bytes())
    d = d[d.coords["passes_thresholds"]]
    return (
        pl.DataFrame(y)
        .with_columns(pl.col("centroid").list.to_struct())
        .unnest("centroid")
        .with_columns(
            field_1=pl.col("field_1") + coords[i, "y"],
            field_2=pl.col("field_2") + coords[i, "x"],
            target=pl.Series(list(d.coords["target"].values)),
            distance=pl.Series(list(d.coords["distance"].values)),
            norm=pl.Series(np.linalg.norm(d.values, axis=(1, 2))),
            tile=pl.lit(str(i)),
        )
        .rename({"field_0": "z", "field_1": "y", "field_2": "x"})
        .with_row_count("idx")
        # .join(d,)
    )


v = load(0)
# u = pl.concat([load(i) for i in range(190, 320)])

# %%


w = 1988
assert len(coords) == len(set(coords["index"]))

cells = [
    Polygon([
        (row["x"], row["y"]),
        (row["x"] + w, row["y"]),
        (row["x"] + w, row["y"] + w),
        (row["x"], row["y"] + w),
    ])
    for row in coords.iter_rows(named=True)
]

idx = index.Index()
for pos, cell in enumerate(cells):
    idx.insert(pos, cell.bounds)

# %%
# from collections import defaultdict

crosses = [sorted(idx.intersection(poly.bounds)) for poly in cells]

# %%

out = []


def process(curr: int):
    this_cross = {j: intersection(cells[curr], cells[j]) for j in crosses[curr] if j > curr}
    thrown = 0
    df = load(curr)

    def what(row):
        for name, intersected in this_cross.items():
            if contains(intersected, Point((row["x"], row["y"]))):
                return False
        return True

    df.filter(pl.struct("x", "y").apply(what))

    for row in df.iter_rows(named=True):
        for name, intersected in this_cross.items():
            if contains(intersected, Point((row["x"], row["y"]))):
                thrown += 1
                # thrownover[name].append(row)
                # Prevent double-counting
                break
        else:
            out.append(row)
    print(f"{curr}: thrown {thrown} / {df.__len__()}")


with ThreadPoolExecutor(8) as exc:
    futs = [exc.submit(process, i) for i in range(len(cells[:]))]
    for fut in futs:
        fut.result()
# for curr in range(len(cells[:20])):
#     process(curr)

# del thrownover[curr]
# %%
spots = pl.DataFrame(out)
spots.write_parquet(path / codebook / "spots.parquet")

# %%
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)

# ax.set_aspect("equal")
axs.scatter(spots["x"][::50], spots["y"][::50], s=0.5, alpha=0.3)

# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
selected = spots.filter(pl.col("target") == "Gpm6a-201")
ax.scatter(selected["x"], selected["y"], s=0.5, alpha=0.4)
# spots = pl.read_parquet(path / "genestar" / "spots.parquet")
# %%
pergene = (
    spots.groupby("target")
    .count()
    .sort("count", descending=True)
    .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
    .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
)
# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=200)
ax.bar(pergene["target"], pergene["count"], color=pergene["color"], width=1, align="edge", linewidth=0)
ax.set_xticks([])
ax.set_yscale("log")
ax.set_xlabel("Gene")
ax.set_ylabel("Count")
plt.tight_layout()

# %%

# %%
genes = ["Sox9-201", "Neurod1-201", "Neurog2-201", "Bcl11b-203", "Cldn5-201", "Sst-201"]
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 8), dpi=200)
axs = axs.flatten()
for ax, gene in zip(axs, genes):
    selected = spots.filter(pl.col("target") == gene)
    ax.scatter(selected["x"], selected["y"], s=0.5, alpha=0.2)
    ax.set_title(gene)

    ax.axis("off")
# %%
genes = pergene["target"]
fig, axs = plt.subplots(ncols=8, nrows=12, figsize=(32, 48), dpi=100)
axs = axs.flatten()
for ax, gene in zip(axs, genes):
    selected = spots.filter(pl.col("target") == gene)
    ax.scatter(selected["x"], selected["y"], s=0.1, alpha=0.2)
    ax.set_title(gene, fontsize=16)
    ax.axis("off")


# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 12), dpi=200)
sns.barplot(x="count", y="target", data=pergene[:100], hue="color", width=1, linewidth=0)
# set x axis font size
ax.tick_params(axis="y", labelsize=6)
ax.set_xscale("log")

# %%
# sns.scatterplot(
#     x="x",
#     y="y",
#     data=spots.to_pandas(),
#     hue="tile",
#     s=1,
# )
# plt.axis("equal")
from polars import col as c

spots = pl.read_parquet("/fast2/3t3clean/analysis/spots.parquet")
# %%
what = spots.groupby("target").agg([
    pl.count(),
    pl.mean("distance"),
    pl.quantile("norm", 0.1),
    pl.mean("area"),
])
# %%
sns.scatterplot(x="norm", y="count", data=what.to_pandas(), alpha=0.3, s=10, edgecolor="none")

# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
axs = axs.flatten()
for col, ax in zip(["distance", "norm", "area"], axs):
    sns.histplot(x=col, data=spots.filter(c("target") == "Lgals1-201").to_pandas(), bins=100, ax=ax)
    sns.histplot(x=col, data=spots.filter(c("target").str.starts_with("Blank")).to_pandas(), bins=100, ax=ax)
    sns.histplot(x=col, data=spots.filter(c("target").str.starts_with("Nrxn")).to_pandas(), bins=100, ax=ax)

    ax.set_yscale("log")

# %%
sns.scatterplot(
    x="area", y="norm", data=spots.filter(c("target") == "Lgals1-201").to_pandas(), alpha=0.02, s=10
)
sns.scatterplot(
    x="area", y="norm", data=spots.filter(c("target").str.starts_with("Blank")).to_pandas(), alpha=0.05, s=10
)
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

rf = DecisionTreeClassifier(max_depth=3, random_state=0)

# dt = DecisionTreeClassifier(max_depth=3, random_state=0)

rf.fit(spots[["area", "distance", "norm"]].to_numpy(), ~spots["target"].str.starts_with("Blank"))
# %%
rf.predict(spots[["area", "distance", "norm"]].to_numpy()).sum()
# %%

from sklearn import svm

svm = svm.LinearSVC(class_weight={True: 1, False: 400})
sampled = spots.sample(n=200000, seed=0, shuffle=True)
good = ~(sampled["target"].str.starts_with("Blank") | sampled["target"].str.starts_with("Nrxn"))
svm.fit(sampled[["area", "norm"]].to_numpy(), good)
# %%

spots = spots.with_columns(
    is_blank=pl.col("target").str.starts_with("Blank") | pl.col("target").str.starts_with("Nrxn")
)

# %%
area = (
    spots.filter(c("norm") > -0.3 / 15 * c("area") + 0.25)
    .groupby(["area", "is_blank"])
    .agg(pl.count())
    .sort("area")
)
# spots.filter(c("norm") < -0.3/20 * c("area") + 0.3 )
area.pivot(values="count", index="area", columns="is_blank").with_columns(
    ratio=pl.col("true") / (c("false") + c("true"))
)

# %%


f = spots.filter(c("norm") > -0.3 / 20 * c("area") + 0.25).groupby("target").count()
o = spots.groupby("target").count()

t = f.join(o, on="target").with_columns(ratio=pl.col("count") / (c("count_right")))

sns.scatterplot(x="count", y="ratio", data=t.to_pandas(), alpha=0.3, s=10, edgecolor="none")
# make x log
plt.xscale("log")
# %%
spots.filter(c("norm") > -0.3 / 20 * c("area") + 0.25).write_parquet(
    "/fast2/3t3clean/analysis/spots_filtered.parquet"
)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

plt.scatter(sampled["area"], sampled["norm"], c=good, cmap=plt.cm.Paired, edgecolors="k", s=0.1)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    svm,
    sampled[["area", "norm"]].to_numpy(),
    ax=ax,
    response_method="decision_function",
    plot_method="contour",
    levels=[-1, 0, 1],
    colors=["k", "k", "k"],
    linestyles=["--", "-", "--"],
)
# %%
svm.predict(sampled[["area", "norm"]].to_numpy()).sum()
# %%
svm.decision_function(sampled[["area", "norm"]].to_numpy())
# %%
