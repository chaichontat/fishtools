# %%
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from rtree import index
from shapely import MultiPolygon, Point, Polygon, STRtree, intersection
from shapely.ops import unary_union

from fishtools.analysis.spots import load_spots
from fishtools.preprocess.tileconfig import TileConfiguration

sns.set_theme()
codebook = "genestar"
roi = "right"
split = True
FILENAME = re.compile(r"reg-(\d+)(?:-(\d+))?\.pkl")


def parse_filename(x: str):
    match = FILENAME.match(x)
    if not match:
        raise ValueError(f"Could not parse {x}")
    return {"idx": int(match.group(1)), "split": int(match.group(2)) if match.group(2) else None}


path = Path(f"/mnt/working/20241113-ZNE172-Zach/analysis/deconv/registered--{roi}")
files = sorted(file for file in Path(path / codebook).glob("*.pkl") if "opt" not in file.stem)

coords = TileConfiguration.from_file(
    Path(path.parent / f"stitch--{roi}" / "TileConfiguration.registered.txt")
    # Path(path / "stitch" / "TileConfiguration.registered.txt")
).df
if len(coords) != len(coords.unique(subset=["x", "y"], maintain_order=True)):
    logger.warning("Duplicates found.")
    coords = coords.unique(subset=["x", "y"], maintain_order=True)

idxs = [name[:4] for name in coords["filename"]]
# pickles = list(map(lambda x: x + ".pkl", idxs))
files = (
    [file for file in files if file.name.rsplit("-", 1)[-1] in idxs]
    if not split
    else [file for file in files if file.name.rsplit("-", 2)[1] in idxs]
)

# if len(files) != len(coords) or len(files) != len(coords) * 4:
#     print({int(x.stem.split("-")[-1]) for x in files} - set(coords["index"]))
#     print(set(coords["index"]) - {int(x.stem.split("-")[-1]) for x in files})
#     raise ValueError("Length of files does not match length of coords.")
# assert len(files) == len(coords)
# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
sns.scatterplot(x="x", y="y", data=coords.to_pandas(), ax=ax, s=10, alpha=0.9)
for row in coords.iter_rows(named=True):
    ax.text(row["x"], row["y"], row["index"], fontsize=10)


# %%
def find_precedence_intersections(polygons):
    result = []
    remaining_area = unary_union(polygons)

    for poly in polygons:
        # Intersect with remaining area
        intersection = poly.intersection(remaining_area)
        if not intersection.is_empty:
            result.append(intersection)
        # Remove this polygon from remaining area
        remaining_area = remaining_area.difference(poly)

    return result


find_precedence_intersections()
# %%


# %%


def load_pickle(i: int, filter_: bool = True, *, size: int = 1998, cut: int = 1024):
    idx, split = parse_filename(files[i].name).values()
    c = coords.filter(pl.col("index") == idx).row(0, named=True)

    logger.debug(f"Loading {files[i]} as split {split}")

    if split == 0:  # top-left
        pass
    elif split == 1:  # top-right
        c["x"] += size - cut
    elif split == 2:  # bottom-left
        c["y"] += size - cut  # + to move down
    elif split == 3:  # bottom-right
        c["x"] += size - cut
        c["y"] += size - cut

    return load_spots(files[i], i, filter_=filter_, tile_coords=(c["x"], c["y"]))


# v = load_pickle(0)

# %%


def gen_splits(coords: tuple[float, float], n: int = 4, size: int = 1988, cut: int = 1024):
    if size < 1 or cut < 1:
        raise ValueError("width and offset must be greater than 0")
    x, y = coords
    if n == 0:  # [:cut, :cut] - top-left
        return Polygon([
            (x, y),
            (x + cut, y),
            (x + cut, y + cut),  # + for y to go down
            (x, y + cut),
        ])
    if n == 1:  # [:cut, -cut:] - top-right
        return Polygon([(x + size - cut, y), (x + size, y), (x + size, y + cut), (x + size - cut, y + cut)])
    if n == 2:  # [-cut:, :cut] - bottom-left
        return Polygon([(x, y + size - cut), (x + cut, y + size - cut), (x + cut, y + size), (x, y + size)])
    if n == 3:  # [-cut:, -cut:] - bottom-right
        return Polygon([
            (x + size - cut, y + size - cut),
            (x + size, y + size - cut),
            (x + size, y + size),
            (x + size - cut, y + size),
        ])
    raise ValueError(f"Unknown n={n}")


w = 1988
assert len(coords) == len(set(coords["index"]))


if split:
    cells = {
        f"{row['index']}-{n}": gen_splits((row["x"], row["y"]), n=n)
        for row in coords.iter_rows(named=True)
        for n in range(4)
    }
else:
    cells = {
        row["index"]: Polygon([
            (row["x"], row["y"]),
            (row["x"] + w, row["y"]),
            (row["x"] + w, row["y"] + w),
            (row["x"], row["y"] + w),
        ])
        for row in coords.iter_rows(named=True)
    }


idx = index.Index()
for pos, cell in enumerate(cells.values()):
    idx.insert(pos, cell.bounds)

# Get the indices of the cells that cross each other
crosses = [sorted(idx.intersection(poly.bounds)) for poly in cells.values()]
MultiPolygon([list(cells.values())[idx] for idx in crosses[0]])
_cells = list(cells.values())

# %%

# MultiPolygon(_cells[:2])

MultiPolygon([_cells[i] for i in crosses[0] if i > 0])
# %%


def process(curr: int, filter_: bool = True):
    _cells = list(cells.values())
    current_cell = _cells[curr]

    lower_intersections = {j: intersection(current_cell, _cells[j]) for j in crosses[curr] if j < curr}

    df = load_pickle(curr, filter_=filter_)
    df = df.filter(pl.col("passes_thresholds") & pl.col("area").is_between(15, 80) & pl.col("norm").gt(0.05))

    # Create the "exclusive" area for this cell
    exclusive_area = current_cell
    for intersect in lower_intersections.values():
        exclusive_area = exclusive_area.difference(intersect)

    # Create STRtree with the exclusive area parts
    if isinstance(exclusive_area, MultiPolygon):
        tree_geoms = list(exclusive_area.geoms)
    else:
        tree_geoms = [exclusive_area]
    tree = STRtree(tree_geoms)

    def check_point(x: float, y: float):
        point = Point(x, y)
        return len(tree.query(point, "within")) > 0

    mask = df.select([
        pl.struct(["x", "y"])
        .map_elements(lambda row: check_point(row["x"], row["y"]), return_dtype=pl.Boolean)
        .alias("keep")
    ])

    filtered_df = df.filter(mask["keep"])
    logger.info(f"{files[curr]}: thrown {len(df) - len(filtered_df)} / {df.__len__()}")
    return filtered_df


# %%
#  mp_context=get_context("forkserver")
with ThreadPoolExecutor(8) as exc:
    out = []
    futs = [exc.submit(process, i, filter_=True) for i in range(16, 24)]  # len(spots))]
    for i, fut in enumerate(futs):
        try:
            out.append(fut.result())
        except Exception as e:
            logger.error(f"Error in {files[i]}: {e}")
# for curr in range(len(cells[:20])):
#     process(curr)

# del thrownover[curr]
spots = pl.concat(out)
spots.write_parquet(path / codebook / "spots.parquet")

# %%
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)

# ax.set_aspect("equal")
downsample = 5
axs.scatter(spots["x"][::downsample], spots["y"][::downsample], s=0.5, alpha=0.3)
axs.set_aspect("equal")
axs.axis("off")

# %%
spots = pl.read_parquet(path / codebook / "spots.parquet")
pergene = (
    spots.filter(pl.col("passes_thresholds") & pl.col("area").is_between(15, 50) & pl.col("norm").gt(0.05))
    .group_by("target")
    .len("count")
    .sort("count", descending=True)
    .with_columns(is_blank=pl.col("target").str.starts_with("Blank"))
    .with_columns(color=pl.when(pl.col("is_blank")).then(pl.lit("red")).otherwise(pl.lit("blue")))
)

# print(np.sum(pergene["count"].to_list()))
# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=200)
ax.bar(pergene["target"], pergene["count"], color=pergene["color"], width=1, align="edge", linewidth=0)
ax.set_xticks([])
ax.set_yscale("log")
ax.set_xlabel("Gene")
ax.set_ylabel("Count")
plt.tight_layout()

# %%
# pergene.filter(pl.col("target").str.starts_with("Blank")).join()

# %%
if codebook == "tricycleplus":
    genes = ["Ccnd3", "Mcm5", "Top2a", "Pou3f2", "Cenpa", "Hell"]
else:
    genes = ["Sox9", "Pax6", "Eomes", "Tbr1", "Bcl11b", "Fezf2", "Neurod6", "Notch1", "Notch2"]
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 12), dpi=200, facecolor="black")

for ax, gene in zip(axs.flat, genes):
    selected = spots.filter(pl.col("target").str.starts_with(gene))
    if not len(selected):
        raise ValueError(f"No spots found for {gene}")
    ax.set_aspect("equal")
    ax.set_title(gene, fontsize=16, color="white")
    ax.hexbin(selected["x"], -selected["y"], gridsize=400, cmap="magma")
    # ax.scatter(selected["y"], -selected["x"], s=5000 / len(selected), alpha=0.2)
    ax.axis("off")
# %%
genes = sorted(pergene["target"])
dark = False
fig, axs = plt.subplots(ncols=20, nrows=15, figsize=(72, 36), dpi=160, facecolor="black" if dark else "white")
axs = axs.flatten()
for ax, gene in zip(axs, genes):
    selected = spots.filter(pl.col("target") == gene)
    if not len(selected):
        logger.warning(f"No spots found for {gene}")
        continue
    ax.set_aspect("equal")
    ax.set_title(gene, fontsize=16, color="white" if dark else "black")
    ax.axis("off")
    if dark:
        ax.hexbin(selected["x"], selected["y"], gridsize=250)
    else:
        ax.scatter(selected["x"], selected["y"], s=2000 / len(selected), alpha=0.1)
plt.tight_layout()
# %%


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

# spots = pl.read_parquet("/fast2/3t3clean/analysis/spots.parquet")
# %%
what = spots.group_by("target").agg([
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
