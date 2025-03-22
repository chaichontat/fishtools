# %%
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import BaseModel, TypeAdapter
import pyparsing as pp
import seaborn as sns
from rtree import index
from tifffile import imread
from shapely import Point, Polygon, contains, intersection

sns.set_theme()

files = sorted((path := Path("/fast2/synaptosome/registered")).glob("*.parquet"))


def parse_tileconfig(path: str | Path):
    """Parse TileConfiguration.txt file

    Example:
        file_content =\"""
        fid-0000.tif; ; (0.0, 0.0)
        fid-0001.tif; ; (-1650.901997346868, -155.8836296194227)
        fid-0002.tif; ; (-3294.5706824523218, -392.6388842000409)
        \"""

    Returns:
        pl.DataFrame: ["prefix", "index", "filename", "x", "y"]
    """

    content = Path(path).read_text()

    integer = pp.Word(pp.nums)
    float_number = pp.Combine(
        pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums) + "." + pp.Word(pp.nums)
    ).setParseAction(
        lambda t: float(t[0])  # type: ignore
    )

    point = pp.Suppress("(") + float_number("x") + pp.Suppress(",") + float_number("y") + pp.Suppress(")")
    entry = pp.Group(pp.Regex(r"(\w+-)(\d+)(\.tif)")("filename") + pp.Suppress("; ; ") + point)

    comment = (pp.Literal("#") + pp.restOfLine) | (pp.Literal("dim") + pp.rest_of_line)
    parser = pp.ZeroOrMore(comment.suppress() | entry)

    # Perform this on extracted file name
    # since we want both the raw file name and the index.
    filename = (
        (pp.Word(pp.alphas)("prefix") + "-")
        + integer("index").setParseAction(lambda t: int(t[0]))  # type: ignore
        + pp.Literal(".tif")
    )
    # Parse the content
    out = []
    for x in parser.parseString(content):
        d = x.as_dict()
        out.append(filename.parseString(d["filename"]).as_dict() | d)

    return pl.DataFrame(out)


# %%

# plt.imshow(imread("/fast2/3t3clean/analysis/deconv/registered/reg-0000.tif").max(axis=(0, 1)), zorder=1)
# # plt.imshow(imread("/fast2/3t3clean/analysis/deconv/registered/reg-0056.tif").max(axis=(0,1)),zorder=1 )
# # %%
# plt.imshow(
#     imread("/fast2/3t3clean/analysis/deconv/registered/reg-0003.tif").max(axis=(0, 1))[900:1100, 1788:],
#     zorder=1,
# )
# # %%
# plt.imshow(
#     imread("/fast2/3t3clean/analysis/deconv/registered/reg-0002.tif").max(axis=(0, 1))[950:1150, :200],
#     zorder=1,
# )
# %%

coords = parse_tileconfig(path.parent / "analysis/fid/TileConfiguration.registered.txt")

# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), dpi=200)
sns.scatterplot(x="x", y="y", data=coords.to_pandas(), ax=ax, s=10, alpha=0.9)

# %%
baddies = []
for file in Path(path / "shifts").glob("shifts_*.json"):
    with open(file, "r") as f:
        shifts = json.load(f)

    vals = np.array(list(shifts.values()))

    if len(np.flatnonzero(np.square(vals).sum(axis=1))) < len(shifts) - 1:
        print(shifts)
        baddies.append(file)

# %%


def load(i: int, channel_mapping: dict[int, tuple[str, str]]):
    """Expected columns:
    [xcentroid, ycentroid, sharpness, roundness1, roundness2,
    npix, sky, peak, flux, mag, channel]

    Returns:
        pl.DataFrame: with columns x, y, target
    """
    return (
        pl.read_parquet(files[i])
        .with_columns(
            y=pl.col("ycentroid") + coords[i, "y"],
            x=pl.col("xcentroid") + coords[i, "x"],
            target=pl.col("channel").apply(lambda x: channel_mapping[x][0]),
            channel=pl.col("channel").apply(lambda x: channel_mapping[x][1]),
            tile=pl.lit(str(i)),
        )
        .rename(
            {
                "xcentroid": "xlocal",
                "ycentroid": "ylocal",
            }
        )
    )

    # return (
    #     pl.DataFrame(y)
    #     .with_columns(pl.col("centroid").list.to_struct())
    #     .unnest("centroid")
    #     .with_columns(
    #         field_1=pl.col("field_1") + coords[i, "y"],
    #         field_2=pl.col("field_2") + coords[i, "x"],
    #         target=pl.lit(list(d.coords["target"].values)),
    #         distance=pl.lit(list(d.coords["distance"].values)),
    #         norm=pl.lit(np.linalg.norm(d.values, axis=(1, 2))),
    #         tile=pl.lit(str(i)),
    #     )
    #     .rename({"field_0φ": "z", "field_1": "y", "field_2": "x"})
    #     .with_row_count("idx")
    #     # .join(d,)
    # )


# %%
channel_mapping = {
    i: name
    for i, name in enumerate(
        [
            ("Pfdn2", "560"),
            ("Calm1", "560"),
            ("Gad2", "560"),
            ("Sst", "560"),
            ("Pdgfra", "560"),
            ("Cx3cr1", "560"),
            ("Gfap", "560"),
            ("Cx3cl1", "560"),
            ("Gng7", "650"),
            ("mtCytb", "650"),
            ("Arc", "650"),
            ("Slc17a7", "650"),
            ("Fos", "650"),
            ("Pvalb", "650"),
            ("Ctss", "650"),
            ("16_blank", "650"),
            ("Cox7b", "750"),
            ("mtCo1", "750"),
            ("Plcxd2", "750"),
            ("Cux2", "750"),
            ("Mal", "750"),
            ("Hexb", "750"),
            ("Rorb", "750"),
            ("24_blank", "750"),
            ("Malat1", "560"),
            ("all", "650"),
            ("blank_560", "560"),
            ("blank_650", "650"),
            ("blank_750", "750"),
            ("polyA", "750full"),
            ("wga", "405"),
        ]
    )
}


v = load(0, channel_mapping)

# %%
plt.scatter(v["y"], v["x"], c=np.log10(v["flux"]), s=0.5, alpha=0.3, cmap="viridis")
plt.colorbar()
# set equal aspect ratio
plt.axis("equal")
# %%


w = 1988
assert len(coords) == len(set(coords["index"]))

cells = [
    Polygon(
        [
            (row["x"], row["y"]),
            (row["x"] + w, row["y"]),
            (row["x"] + w, row["y"] + w),
            (row["x"], row["y"] + w),
        ]
    )
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
for i in range(len(cells[:])):
    this_cross = {j: intersection(cells[i], cells[j]) for j in crosses[i] if j > i}
    df = load(i, channel_mapping)

    def what(row: dict[str, float]):
        for name, intersected in this_cross.items():
            if contains(intersected, Point((row["x"], row["y"]))):
                return False
        return True

    filtered = df.filter(pl.struct("x", "y").apply(what))
    thrown = len(df) - len(filtered)
    out.append(filtered)
    # del thrownover[curr]
    print(f"{i}: thrown {thrown} / {len(df)}")
# %%
spots: pl.DataFrame = pl.concat(out)
# %%
spots.write_parquet(path.parent / "spots.parquet")

# %%
# spots = spots.join(
#     pl.DataFrame(list(channel_mapping.values()))
#     .transpose()
#     .rename({"column_0": "target", "column_1": "channel_"}),
#     on="target",
# )
# %%
sns.scatterplot(x="x", y="y", data=spots.to_pandas(), hue="tile", s=10, alpha=1)
# set equal aspect ratio
# plt.axis("equal")
plt.gca().set_xlim(0, 400)
plt.gca().set_ylim(500, 900)

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
fig, axs = plt.subplots(ncols=10, nrows=3, figsize=(18, 6), dpi=200)
cs = ["560", "650", "750"]
ylim = {"560": (1, 100000), "650": (1, 20000), "750": (1, 1000)}
blanks = {"560": "blank_560", "650": "16_blank", "750": "blank_750"}
cumulative = dict(element="step", fill=False, cumulative=True, stat="density", common_norm=False)
for i, (ax_, c) in enumerate(zip(axs, cs)):
    blank = spots.filter(pl.col("target") == blanks[c]).to_pandas()
    for j, (ax, target) in enumerate(zip(ax_, [x[0] for x in channel_mapping.values() if x[1] == c])):
        sns.histplot(
            x="flux",
            data=spots.filter(pl.col("target") == target).to_pandas(),
            bins=50,
            ax=ax,
            log_scale=True,
            # **cumulative,
            # common_norm=False,
        )
        sns.histplot(
            x="flux",
            data=blank,
            bins=50,
            ax=ax,
            log_scale=True,
            # **cumulative,
            # common_norm=False,
        )
        ax.set_title(target)
        ax.set_xlim(1, 3000)
        ax.set_ylim(ylim[c])
        ax.set_yscale("log")
        if i != 2:
            ax.get_xaxis().set_visible(False)
        if j != 0:
            ax.get_yaxis().set_visible(False)


plt.tight_layout()
# %%


# mt = spots.filter(pl.col("target") == "Cux2")


# plt.hist(spots.filter(pl.col("target") == "Calm1")["flux"], bins=100, log=True)
# plt.hist(spots.filter(pl.col("target") == "blank_650")["flux"], bins=100, log=True)


# %%
what = spots.groupby("target").agg(
    [
        pl.count(),
        pl.mean("distance"),
        pl.quantile("norm", 0.1),
        pl.mean("area"),
    ]
)
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
