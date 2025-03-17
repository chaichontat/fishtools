# %%
from itertools import chain
from pathlib import Path

import anndata as ad
import cmocean  # colormap, do not remove
import colorcet as cc  # colormap, do not remove
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spaco
from matplotlib.collections import LineCollection
from scipy.interpolate import splev, splrep
from shapely import MultiPolygon
from tifffile import imread

from fishtools.utils.plot import plot_wheel

sns.set_theme()
path = Path("/working/20241119-ZNE172-Trc/baysor--all")
adata = sc.read_h5ad(path / "pearsonedcortex.h5ad")
# %%
# adata.obs["sample"] = pd.cut(
#     adata.obs["y"],
#     bins=[0, 11576.95489472727, 24987.611969928068, 57788.54119783332],
#     labels=["top", "center", "bottom"],
#     include_lowest=True,
# )
# sc.pl.umap(adata, color="sample")


def scale_bar(ax, *, length=100, y_offset=300, text_pos=0.15, text_offset=400, text="100 μm", fontsize=12):
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[0] + (length / 0.216)],
        [ax.get_ylim()[0] + y_offset, ax.get_ylim()[0] + y_offset],
        c="black",
        alpha=0.5,
    )
    ax.text(
        ax.get_xlim()[0] + (length / 0.216) * text_pos,
        ax.get_ylim()[0] + text_offset,
        text,
        fontsize=fontsize,
    )
    return ax


print(
    "\n".join(
        [
            f"Cluster {c}: "
            + ", ".join(
                [
                    f"{entry['names']}: {np.round(entry['scores'], 2)}"
                    for entry in sc.get.rank_genes_groups_df(adata, group=c)
                    .head(10)[["names", "scores"]]
                    .to_dict(orient="records")
                ]
            )
            for c in adata.obs["leiden"].cat.categories
        ]
    )
)
# %%
from fishtools.utils.plot import plot_wheel

fig, axs = plot_wheel(adata.obsm["tricycle"])
fig.suptitle("Tricycle embedding of the E15.5 cortex", fontsize=18, x=0.118, y=0.96, ha="left")
plt.tight_layout()

# %%
with sns.axes_style("darkgrid"):
    sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, fontsize=11, show=False)
    for ax in plt.gcf().axes:
        for text in ax.texts:
            text.set_rotation(40)

# %%
color_mapping = spaco.colorize(
    cell_coordinates=adata.obsm["spatial"],
    cell_labels=adata.obs["leiden"],
    radius=0.05,
    palette=list(
        map(plt.matplotlib.colors.rgb2hex, plt.cm.tab20(range(len(adata.obs["leiden"].cat.categories))))
    ),
    n_neighbors=20,
    colorblind_type="none",
)

color_mapping = {k: color_mapping[k] for k in adata.obs["leiden"].cat.categories}
palette_spaco = list(color_mapping.values())


# %%
genes = sorted(
    set(
        chain.from_iterable(
            [
                sc.get.rank_genes_groups_df(adata, group=c).head(6)["names"]
                for c in adata.obs["leiden"].cat.categories
            ]
        )
    )
)

# %%

fig, axs = plt.subplots(ncols=2, figsize=(12, 8), dpi=300, facecolor="white")

# sc.pl.umap(adata, color="leiden", palette=palette_spaco, ax=axs[0])
sc.pl.embedding(adata, color="leiden", basis="spatial", ax=axs[1])  # type: ignore
for ax in axs:
    ax.set_aspect("equal")
plt.tight_layout()
# %%


sc.tl.dendrogram(adata, groupby="leiden")
sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", standard_scale="var", n_genes=5)


# %%
fig, axs = plt.subplots(ncols=2, figsize=(15, 4), dpi=300, facecolor="white")
sc.pl.umap(adata, color="leiden", ax=axs[0])
sc.pl.embedding(adata, color="leiden", basis="spatial", ax=axs[1])  # type: ignore
plt.tight_layout()
# %% Highlight clusters
clusters = adata.uns["dendrogram_leiden"]["categories_ordered"]  # adata.obs["leiden"].cat.categories
n_cols = 6
n_rows = int(np.ceil(len(clusters) / n_cols)) * 2
with sns.axes_style("white"):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=300)
    axes = axes.flatten()

    for idx, cluster in enumerate(clusters):
        custom_palette = {c: "#F8F8F8" for c in clusters}
        custom_palette[cluster] = palette_spaco[idx]
        shared = dict(
            color="leiden",
            legend_loc=None,
            frameon=False,
            palette=custom_palette,
            show=False,
            title=f"Cluster {cluster}",
        )
        # sc.pl.umap(adata, ax=axes[idx * 2], **shared)
        sc.pl.embedding(adata, ax=axes[idx * 2], basis="spatial", **shared)  # type: ignore
        axes[idx * 2].set_aspect("equal")
        sc.pl.embedding(adata, ax=axes[idx * 2 + 1], basis="X_straightenedsub", **shared)  # type: ignore

    for ax in fig.axes:
        # ax.set_aspect("equal")
        if not ax.has_data():
            fig.delaxes(ax)
    plt.tight_layout()
    plt.show()

# %%


# %%


trc = pd.read_csv("neuroRef.csv")
shared_genes = sorted(set(trc["symbol"]) & set(adata.var_names))

loadings = trc = (
    trc[trc["symbol"].isin(shared_genes)]
    .set_index("symbol")
    .reindex(shared_genes)
    .reset_index()[["pc1.rot", "pc2.rot"]]
)

pls = adata[:, shared_genes].X @ loadings.to_numpy()
adata.obsm["tricycle"] = pls
adata.obs["tricycle"] = (
    np.arctan2(adata.obsm["tricycle"][:, 1], adata.obsm["tricycle"][:, 0] + 1.0) + np.pi / 2
) % (2 * np.pi) - np.pi
# %%
with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300)
    sc.pl.umap(
        adata,
        palette=palette_spaco,
        color="leiden",
        cmap="cet_colorwheel",
        show=False,
        vmin=-np.pi,
        s=20,
        vmax=np.pi,
        colorbar_loc=None,
        ax=axs[0],
    )
    sc.pl.embedding(
        adata,
        palette=palette_spaco,
        color="leiden",
        basis="spatial",
        cmap="cet_colorwheel",
        show=False,
        s=20,
        vmin=-np.pi,
        vmax=np.pi,
        alpha=1,
        # colorbar_loc=None,
        ax=axs[1],
    )
    for ax in axs:
        ax.set_aspect("equal")

    axs[0].set_title("")
    axs[0].set_title(
        "Leiden community detection recapitulates laminar organization",
        color="black",
        fontsize=24,
        loc="left",
        x=0.025,
        pad=10,
    )
    axs[1].set_title("")
    scale_bar(axs[1])
    for ax in axs:
        ax.spines[:].set_visible(False)

    plt.tight_layout()


# %%
with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300, facecolor="white")
    sc.pl.umap(
        adata,
        title="EdU labels S-phase cells",
        color="intensity_mean",
        cmap="turbo",
        show=False,
        s=20,
        vmin=np.nanpercentile(adata.obs["intensity_mean"], 20),
        vmax=np.nanpercentile(adata.obs["intensity_mean"], 99.9),
        colorbar_loc=None,
        ax=axs[0],
    )
    sc.pl.embedding(
        adata,
        color="intensity_mean",
        title="EdU mean intensity",
        basis="spatial",
        cmap="turbo",
        show=False,
        s=20,
        vmin=np.nanpercentile(adata.obs["intensity_mean"], 20),
        vmax=np.nanpercentile(adata.obs["intensity_mean"], 99.9),
        # colorbar_loc=None,
        ax=axs[1],
    )
    for ax in axs:
        ax.set_aspect("equal")
    scale_bar(axs[1])
    axs[0].set_title("")
    axs[0].set_title("EdU labels S-phase cells", color="black", fontsize=24, loc="left", x=0.025, pad=0)
    axs[1].set_title("")
    for ax in axs:
        ax.spines[:].set_visible(False)
    plt.tight_layout()

# %%
with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300, facecolor="white")
    sc.pl.umap(
        adata,
        title="EdU labels S-phase cells",
        color="tricycle",
        cmap="cet_colorwheel",
        show=False,
        s=20,
        colorbar_loc=None,
        ax=axs[0],
    )
    sc.pl.embedding(
        adata,
        color="tricycle",
        title="cet_colorwheel",
        basis="spatial",
        cmap="cet_colorwheel",
        show=False,
        s=20,
        # colorbar_loc=None,
        ax=axs[1],
    )
    for ax in axs:
        ax.set_aspect("equal")
    scale_bar(axs[1])
    axs[0].set_title("")
    axs[0].set_title("", color="black", fontsize=24, loc="left", x=0.025, pad=0)
    axs[1].set_title("")
    for ax in axs:
        ax.spines[:].set_visible(False)
    plt.tight_layout()

# %%

# Discern IPCs/apical
cluster_mapping = {
    # "0": "Transitional astrocyte",
    "1": "Upper-layer GABAergic interneurons",
    "2": "Maturing deep layer excitatory neurons",
    "3": "More-matured excitatory neurons",
    "4": "G1/S phase neural progenitors",
    "5": "Neural progenitors",
    # "6": "Immature interneurons",
    "7": "Radial glia",
    "8": "Differentiating progenitors",
    "9": "Endothelial cells",
    # "10": "Cortical neuroblasts",
    "11": "Committed excitatory neurons",
    # "12": "Mixed-state radial glia",
    # "13": "G2/M phase neural stem cells",
    # "14": "G2-phase neural stem cells",
    "15": "Immature somatostatin interneurons",
}

# plt.scatter(adata.obs["tricycle"], adata.obs["intensity_mean"], s=1)
# %%
key = "intensity_mean"
with sns.axes_style("whitegrid"):
    fig, axs = plt.subplots(figsize=(15, 3), ncols=2, dpi=300)
    # sc.pl.umap(
    #     adata,
    #     color=key,
    #     title="CFSE mean intensity",
    #     cmap="turbo",
    #     show=False,
    #     s=20,
    #     vmin=np.nanpercentile(adata.obs[key], 30),
    #     vmax=np.nanpercentile(adata.obs[key], 99.9),
    #     colorbar_loc=None,
    #     ax=axs[0],
    # )
    sc.pl.embedding(
        adata,
        color=key,
        title="CFSE mean intensity",
        basis="spatial",
        # frameon=False,
        cmap="turbo",
        show=False,
        s=3,
        vmin=np.nanpercentile(adata.obs[key], 30),
        vmax=np.nanpercentile(adata.obs[key], 99.9),
        # colorbar_loc=None,
        ax=axs[1],
    )
    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)
        ax.set_aspect("equal")
        ax.spines[:].set_visible(False)
    axs[0].set_title("")
    axs[0].set_title("CFSE mean intensity", color="black", fontsize=24, loc="left", x=0.025, pad=0)
    axs[1].set_title("")
    scale_bar(axs[1])
    plt.tight_layout()

# %%
# sns.scatterplot(data=adata.obs, x="tricycle", y="intensity_mean_edu", s=1, alpha=0.2)
# plt.yscale("log")
plt.hexbin(adata.obs["tricycle"], np.log(adata.obs["intensity_mean"]), gridsize=50, bins="log")
# %%
spaces = np.linspace(-np.pi, np.pi, 50)
bin_centers = (spaces[:-1] + spaces[1:]) / 2
adata.obs["tricycle_bin"] = np.digitize(adata.obs["tricycle"], spaces)
adata.obs["tricycle_bin"] = adata.obs["tricycle_bin"].map(lambda x: bin_centers[x - 1])
# %%
adata.obs[["tricycle_bin", "intensity_max"]].groupby("tricycle_bin").agg("mean").plot()
# %%
# sns.kdeplot(
#     data=adata.obs,
#     x="tricycle_bin",  # Use original tricycle values, not bins
#     y="intensity_mean",
#     fill=True,  # Add color between contour lines
#     alpha=0.5,  # Transparency
#     cmap="viridis",  # Color map
# )

sns.lineplot(data=adata.obs, x="tricycle_bin", estimator="median", y="intensity_mean", errorbar=("pi", 50))


# %%
def rainbow_line(x, y, *, ax, linewidth=2):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(segments, cmap="cet_colorwheel", norm=norm)
    lc.set_array(x[:-1])
    line = ax.add_collection(lc)
    lc.set_linewidth(linewidth)
    return ax


fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
sns.lineplot(
    data=adata.obs,
    ax=ax,
    color="gray",
    # alpha=0.2,
    x="tricycle_bin",
    estimator="median",
    y="intensity_mean",
    errorbar=("pi", 50),
)


# Get median values
medians = adata.obs.groupby("tricycle_bin")["intensity_mean"].median()
x = np.array(medians.index)
y = medians.values

# Create colored line segments
rainbow_line(x, y, ax=ax)


ax.set_xlabel("Tricycle θ")
# ax.set_ylabel("Mean EdU intensity", rotation=30, ha='right')
fig.suptitle(
    "EdU intensity correlates with Tricycle θ", fontsize=18, x=0.118, y=0.9525, weight=600, ha="left"
)
ax.set_title("Median over cells, 50 bins, error bar indicates 25-75% percentile", loc="left", fontsize=14)
plt.tight_layout()

# %%
adata.obs["Top2a"] = np.array(adata[:, "Top2a"].X).squeeze()
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
sns.lineplot(
    data=adata.obs,
    ax=ax,
    color="gray",
    # alpha=0.2,
    x="tricycle_bin",
    estimator="median",
    y="intensity_mean",
    errorbar=("pi", 50),
)


# Get median values
medians = adata.obs.groupby("tricycle_bin")["intensity_mean"].median()
x = np.array(medians.index)
y = medians.values

# Create colored line segments
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(x.min(), x.max())
lc = LineCollection(segments, cmap="cet_colorwheel", norm=norm)
lc.set_array(x[:-1])
line = ax.add_collection(lc)
lc.set_linewidth(2)

ax.set_xlabel("Tricycle θ")
# ax.set_ylabel("Mean EdU intensity", rotation=30, ha='right')
fig.suptitle(
    "EdU intensity correlates with Tricycle θ", fontsize=18, x=0.118, y=0.9525, weight=600, ha="left"
)
ax.set_title("Median over cells, 50 bins, error bar indicates 25-75% percentile", loc="left", fontsize=14)
plt.tight_layout()

# %%
g = "Mcm6"
adata.obs[g] = adata.layers["raw"][
    :, np.flatnonzero(adata.var_names == g)[0]
].toarray().squeeze() + np.random.normal(0, 0.1, size=adata.shape[0])

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
sns.scatterplot(
    data=adata.obs,
    ax=ax,
    color="gray",
    # alpha=0.2,
    x="tricycle_bin",
    hue="tricycle_bin",
    y=g,
    palette="cet_colorwheel",
    # errorbar=("pi", 50),
)
# %%
adata.obs["Hes5"] = adata.layers["raw"][:, np.flatnonzero(adata.var_names == "Hes5")[0]].toarray().squeeze()
adata.obs["edu_pos"] = adata.obs["intensity_mean"] > 4000
for i in range(6, 8):
    sns.histplot(data=adata.obs[adata.obs["leiden"] == str(i)], x="Hes5", label=i, bins=np.arange(20))
plt.legend()

# %%

# %%

sc.pl.embedding(adata, basis="spatial", color="intensity_mean", cmap="turbo", show=False)

# %%
# %%
from itertools import repeat

n_rows = int(np.ceil(len(clusters) / n_cols)) * 3
with sns.axes_style("white"):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for idx, cluster in enumerate(clusters):
        print(idx)
        # Mask other clusters by setting their values to nan
        mask = ads.obs["leiden"] != cluster
        ads.obs["tricycle_masked"] = adata.obs["tricycle"].copy()
        ads.obs.loc[mask, "tricycle_masked"] = np.nan

        shared = dict(
            color="tricycle_masked",
            legend_loc=None,
            frameon=False,
            cmap="cet_colorwheel",
            title="",
            show=False,
            colorbar_loc=None,
        )
        sc.pl.umap(ads, ax=axes[idx * 3], **shared)
        sc.pl.embedding(ads, ax=axes[idx * 3 + 1], basis="spatial", **shared)  # type: ignore
        sc.pl.embedding(ads, ax=axes[idx * 3 + 2], basis="X_straightenedscale", **shared)  # type: ignore

    for i, (ax, cluster) in enumerate(
        zip(fig.axes, chain.from_iterable(repeat(cluster, 3) for cluster in clusters))
    ):
        if i % 3 == 0:
            ax.set_title(
                f"Cluster {cluster}\n{cluster_mapping.get(cluster, 'Unknown')}", fontsize=16, loc="left"
            )
        ax.tick_params(colors="gray")
        ax.yaxis.label.set_color("gray")
        if i % 3 != 2:
            ax.set_aspect("equal")
        if not ax.has_data():
            fig.delaxes(ax)
    plt.tight_layout()
    plt.show()

# %%

# %%
import scFates as scf

ads = adata
# μ is curviness, lower is more curvy
# λ is bunching up of spots
tt = scf.tl.curve(ads, Nodes=50, use_rep="spatial", ndims_rep=2, epg_mu=0.5, epg_lambda=0.01)
ads.obsm["X_spatial"] = ads.obsm["spatial"]
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
scf.pl.graph(ads, basis="spatial", ax=ax)
ax.set_aspect("equal")
# %%
nodes = ads.uns["graph"]["F"]
plt.scatter(nodes[0], nodes[1], s=10, alpha=1)

import networkx as nx
import numpy as np

# Convert adjacency matrix to NetworkX graph
G = nx.from_numpy_array(ads.uns["graph"]["B"])

# Find path between endpoints (points with degree 1)
endpoints = [n for n in G.nodes() if G.degree(n) == 1]
path = nx.shortest_path(G, endpoints[0], endpoints[1])

# Get ordered points
ordered_points = ads.uns["graph"]["F"].T[path][::-1]


# %%
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import cdist

# Fit spline to nodes
tck, u = splprep(ordered_points.T, s=0, k=3)

# Generate points along spline
t_samples = np.linspace(0, 1, 2000)
spline_points = np.array(splev(t_samples, tck)).T

# Find closest spline points for each data point
distances = cdist(ads.obsm["spatial"], spline_points)
closest_indices = np.argmin(distances, axis=1)
closest_t = t_samples[closest_indices]
closest_points = spline_points[closest_indices]
# %%
points = ads.obsm["spatial"]


# (positive above curve, negative below)
def get_signed_distances(points: np.ndarray, closest_points: np.ndarray, tck):
    tangents = np.array(splev(closest_t, tck, der=1)).T
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    # Calculate normal vectors (rotate tangent 90 degrees)
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T

    vectors = points - closest_points
    signs = np.sign(np.sum(vectors * normals, axis=1))
    distances = np.linalg.norm(vectors, axis=1) * signs

    return distances


# Calculate signed distances
distances = get_signed_distances(points, closest_points, tck)
# Plot original curve and points
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


# First subplot - original space
ax1.scatter(nodes[0], nodes[1])
ax1.plot(spline_points[:, 0], spline_points[:, 1], markeredgewidth=0, label="Curve")
ax1.scatter(points[:, 0], points[:, 1], c=distances, alpha=0.5, s=0.5, label="Points")
ax1.set_title("Original Sample")
ax1.set_aspect("equal")
ax1.set_xlabel("spatial_1")
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_ylabel("spatial_2")
scale_bar(ax1, fontsize=10)
# ax1.legend()

# Second subplot - straightened version
ax2.scatter(closest_t, distances * 0.216, alpha=0.5, s=1.5, c=ads.obs["tricycle"], cmap="cet_colorwheel")
ax2.axhline(y=0, color="b", linestyle="-", label="Principal curve")
ax2.set_title("Straightened Space")
ax2.set_xlabel("Distance along principal curve (dorsal-ventral)")
ax2.set_ylabel("Distance from principal curve (μm)")
# ax2.set_aspect("equal")
ax2.legend()
ads.obsm["X_straightened"] = np.array([closest_t, distances]).T

fig.tight_layout()
plt.show()


# %%
plt.rcParams["figure.dpi"] = 200
sc.pl.embedding(ads, basis="X_straightened", color="tricycle", cmap="cet_colorwheel", legend_loc="on data")
# %%

plt.scatter(
    ads.obsm["X_straightened"][:, 0],
    ads.obsm["X_straightened"][:, 1],
    c=ads.obs["tricycle"],
    alpha=0.5,
    cmap="cet_colorwheel",
    s=0.5,
)

import numpy as np

num_bins = 75
points = ads.obsm["X_straightened"][:, 0]
bin_edges = np.linspace(min(points), max(points), num_bins)
bin_indices = np.digitize(points, bin_edges) - 1
# %%
bin_mins = np.zeros(num_bins)
bin_maxs = np.zeros(num_bins)
actual_mins = np.zeros(num_bins)
for i in range(num_bins):
    mask = bin_indices == i
    if np.any(mask):
        actual_mins[i] = np.min(ads.obsm["X_straightened"][:, 1][mask])
        bin_mins[i] = np.percentile(ads.obsm["X_straightened"][:, 1][mask], 5)
        bin_maxs[i] = np.percentile(ads.obsm["X_straightened"][:, 1][mask], 99)

lowers = splprep([bin_edges, bin_mins], s=1, k=1)[0]
lowerplot = splev(bin_edges, lowers)
uppers = splprep([bin_edges, bin_maxs], s=1, k=1)[0]
upperplot = splev(bin_edges, uppers)

plt.scatter(*ads.obsm["X_straightened"].T, s=1, label="original")
plt.plot(*np.array(lowerplot), label="lower")
# %% plot spline

result = (
    ads.obsm["X_straightened"][:, 1] - bin_mins[bin_indices] - (np.median(actual_mins) - np.median(bin_mins))
)  # splev(ads.obsm["X_straightened"][:, 0], lowers)[1])
result_scaled = result / (
    bin_maxs[bin_indices] - (bin_mins[bin_indices]) - (np.median(actual_mins) - np.median(bin_mins))
)

result *= 0.216
# %%
ads.obsm["X_straightenedsub"] = np.stack([ads.obsm["X_straightened"][:, 0], result], axis=1)
ads.obsm["X_straightenedscale"] = np.stack([ads.obsm["X_straightened"][:, 0], result_scaled], axis=1)

ads_mask = (ads.obsm["X_straightenedscale"][:, 1] >= 0) & (ads.obsm["X_straightenedscale"][:, 1] <= 1)

# %%
# Subtract minimum value from each point based on its bin
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    ads[ads_mask].obsm["X_straightenedscale"][:, 0],
    ads[ads_mask].obsm["X_straightenedscale"][:, 1],
    # c=ads[:, "Cux2"].X.squeeze(),
    c=ads[ads_mask].obs["tricycle"],
    # cmap="Blues",
    cmap="cet_colorwheel",
    s=2,
)

ax.set_xlabel("Distance along principal curve (D → V)")
ax.set_ylabel("Normalized distance from apical surface")
# ax.set_ylim(0, 250)
# %%
# %%
# Subtract minimum value from each point based on its bin
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    ads.obsm["X_straightenedscale"][:, 0],
    ads.obsm["X_straightenedscale"][:, 1],
    # c=ads[:, "Cux2"].X.squeeze(),
    c=ads.obs["tricycle"],
    # cmap="Blues",
    cmap="cet_colorwheel",
    s=2,
)

ax.set_xlabel("Position on principal curve (D → V)")
ax.set_ylabel("Distance from apical surface (μm)")
# ax.set_ylim(0, 250)

# %%

# %%
# %%
valid = ads[
    ads.obsm["X_straightenedscale"][:, 0].__gt__(0.1)
    & ads.obsm["X_straightenedscale"][:, 0].__lt__(0.7)
    & (ads.obsm["X_straightenedscale"][:, 1] >= 0)
    & (ads.obsm["X_straightenedscale"][:, 1] <= 1)
]
bin_edges = np.linspace(0, 1, 100)
bin_indices = np.digitize(valid.obsm["X_straightenedscale"][:, 1], bin_edges) - 1
out_weights = np.zeros([bin_edges.shape[0], ads.X.shape[1]])
for b in range(bin_edges.shape[0]):
    mask = bin_indices == b
    if np.any(mask):
        out_weights[b] = np.sum(valid.X[mask], axis=0)
# %%
import colorcet as cc
from matplotlib.cm import get_cmap


def set_clustermap(g):
    g.ax_heatmap.set_ylabel("Normalized distance from apical surface", fontsize=16)
    g.ax_heatmap.set_xlabel("")
    g.ax_cbar.set_title("Pearson residuals", loc="left", pad=10)
    g.ax_cbar.set_box_aspect(10)
    return g


out_df = pd.DataFrame(
    out_weights[::-1],
    index=list(reversed(list(map(lambda x: str(np.round(x, 2)), bin_edges)))),
    columns=valid.var_names,
)

pls = (out_df[shared_genes] @ loadings.to_numpy()).to_numpy()
bulked_score = np.arctan2(pls[:, 1], pls[:, 0])

g = sns.clustermap(
    out_df,
    cmap="coolwarm",
    method="complete",  # clustering method
    metric="cosine",  # distance metric
    row_cluster=False,
    # row_colors=get_cmap("cet_colorwheel")((bulked_score + np.pi) / (2 * np.pi)),
    # standard_scale=1,
    figsize=(12, 8),
    dendrogram_ratio=(0.1, 0.1),  # size of dendrograms
    vmin=-50,
    vmax=50,
)
set_clustermap(g)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=30, ha="right", fontsize=10)
# %%
with sns.axes_style("whitegrid"):
    for c in list(cluster_mapping):
        for b in range(bin_edges.shape[0]):
            mask = (bin_indices == b) & (valid.obs["leiden"] == c)
            if np.any(mask):
                out_weights[b] = np.sum(valid.X[mask], axis=0)

        out_df = pd.DataFrame(
            out_weights[::-1],
            index=list(reversed(list(map(lambda x: str(np.round(x, 2)), bin_edges)))),
            columns=valid.var_names,
        )

        _out_df = out_df[sc.get.rank_genes_groups_df(adata, group=c).head(20)["names"]]
        g = sns.clustermap(
            _out_df,
            cmap="coolwarm",
            method="complete",  # clustering method
            metric="correlation",  # distance metric
            row_cluster=False,
            # row_colors=get_cmap("cet_colorwheel")((bulked_score + np.pi) / (2 * np.pi)),
            # standard_scale=1,
            figsize=(8, 6),
            dendrogram_ratio=(0.1, 0.1),  # size of dendrograms
            vmin=-50,
            vmax=50,
        )

        # set title
        g.ax_col_dendrogram.set_title(
            f"Cluster {c}\n{cluster_mapping[c]}", fontsize=30, loc="left", y=2.5, x=-0.065, pad=0
        )
        set_clustermap(g)
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=30, ha="right", fontsize=16)

        # Get current size
        # Extend width (e.g. double it)
        fig = g.figure
        current_width, current_height = fig.get_size_inches()
        fig.set_size_inches(current_width * 2, current_height * 2)

        fig.subplots_adjust(top=0.95, bottom=0.6, left=0, right=0.55)
        axs = [
            fig.add_axes([0.6, 0.58, 0.4, 0.35]),
            fig.add_axes([0.05, 0.1, 0.35, 0.4]),
            fig.add_axes([0.35, 0.1, 0.3, 0.4]),
            fig.add_axes([0.65, 0.122, 0.35, 0.35]),
        ]

        mask = adata.obs["leiden"] != c
        valid.obs["tricycle_masked"] = valid.obs["tricycle"].copy()
        valid.obs.loc[mask, "tricycle_masked"] = np.nan

        shared = dict(
            color="tricycle_masked",
            legend_loc=None,
            # frameon=False,
            cmap="cet_colorwheel",
            title="",
            show=False,
            colorbar_loc=None,
        )
        sc.pl.embedding(valid, ax=axs[0], basis="X_straightenedscale", s=40, **shared)  # type: ignore
        axs[0].set_xlabel("Distance along principal curve D->V")
        axs[0].yaxis.set_label_position("right")
        axs[0].yaxis.labelpad = 0
        axs[0].yaxis.tick_right()
        axs[0].set_ylabel("Normalized distance from apical surface")
        sc.pl.umap(valid, ax=axs[1], s=25, **shared)
        sc.pl.embedding(valid, ax=axs[2], basis="spatial", **shared)  # type: ignore
        axs[1].set_aspect("equal")
        axs[2].plot(
            [axs[2].get_xlim()[0], axs[2].get_xlim()[0] + (100 / 0.216)],
            [axs[2].get_ylim()[0] + 300, axs[2].get_ylim()[0] + 300],
            c="black",
            alpha=0.5,
        )
        axs[2].text(
            axs[2].get_xlim()[0] + (100 / 0.216) * 0.15, axs[2].get_ylim()[0] + 400, "100 μm", fontsize=12
        )
        axs[2].set_aspect("equal")

        cnts = valid.obs[valid.obs["leiden"] == c].groupby("tricycle_bin").size()

        axs[3].plot(cnts.index, cnts.values, alpha=1)
        rainbow_line(cnts.index, cnts.values, ax=axs[3], linewidth=3)
        axs[3].set_xlabel("Tricycle θ")
        axs[3].set_ylabel("Number of cells")

        # xlim, ylim = axs[3].get_xlim(), axs[3].get_ylim()
        # x = np.linspace(xlim[0], xlim[1], 100)
        # y = np.linspace(ylim[0], ylim[1], 100)
        # X, Y = np.meshgrid(x, y)
        # axs[3].pcolormesh(X, Y, X, cmap="cet_colorwheel", alpha=0.5)

        # After plotting, for each axis:
        for ax in axs:
            ax.set_visible(True)
            ax.spines[:].set_visible(False)
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)

        plt.show()


# %%

valid = ads[
    ads.obsm["X_straightenedscale"][:, 0].__gt__(0.1)
    & ads.obsm["X_straightenedscale"][:, 0].__lt__(0.7)
    & (ads.obsm["X_straightenedscale"][:, 1] >= 0)
    & (ads.obsm["X_straightenedscale"][:, 1] <= 1)
]


def phase_shift(theta, shift):
    # Shift the angles
    shifted = theta + shift

    # Wrap back to [-π, π] range
    return np.angle(np.exp(1j * shifted))


valid = valid[valid.obs["leiden"].isin(["5", "7", "8"])]
valid.obs["y_scale"] = np.array(valid.obsm["X_straightenedscale"][:, 1])
valid.obs["_tricycle"] = phase_shift(valid.obs["tricycle"], np.pi / 2)
# %%
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    adata.obs["y_scale"] = np.array(adata.obsm["X_straightenedscale"][:, 1])
    hb = ax.hexbin(
        data=ads[
            ads_mask
            & (ads.obsm["X_straightenedscale"][:, 0] > 0.2)
            & (ads.obsm["X_straightenedscale"][:, 0] < 0.7)
        ].obs,
        x="intensity_mean",
        y="y_scale",
        gridsize=300,
        bins="log",
        cmap="YlGnBu",
        label="Density",
        vmax=5,
        vmin=0.5,
    )
    # cb = plt.colorbar(hb, ax=ax, shrink=0.25, aspect=10, pad=0.02, location="right")
    cb.set_label("Density")
    ax.spines[:].set_visible(False)
    ax.set_xlabel("EdU intensity")
    ax.set_ylabel("Normalized Distance from apical surface")
    plt.tight_layout()
# %%
fig, ax = plt.subplots(
    figsize=(8, 6),
    dpi=200,
)
sns.scatterplot(
    data=valid.obs[valid.obs["intensity_mean"] > 6000], x="tricycle", y="y_scale", ax=ax, label="EdU+"
)
sns.scatterplot(
    data=valid.obs[valid.obs["intensity_mean"] < 6000], x="tricycle", y="y_scale", ax=ax, label="EdU-"
)


# %%
valid.obs["y_idx"] = bin_indices
means = valid.obs[["y_idx", "intensity_mean"]].groupby("y_idx").agg(["mean"])
fig, ax = plt.subplots(figsize=(3, 9), dpi=200)
ax.plot(means["intensity_mean"]["mean"], means.index / 100)
ax.fill_between(means["intensity_mean"]["mean"], means.index / 100, alpha=0.2)
ax.set_xlabel("EdU intensity")
ax.set_ylabel("Normalized Distance from apical surface")
# %%
import scipy

corrss = []

for b in range(100):
    print(b)
    if np.sum(bin_indices == b) < 5:
        corrss.append(np.zeros(ads.n_vars))
        continue
    correlations = pd.Series(
        [
            scipy.stats.pearsonr(
                (chosen := valid[(bin_indices == b)][:, i]).X.toarray().squeeze(),
                np.nan_to_num(chosen.obs["intensity_mean"].values, nan=1000),
            )[0]
            for i in range(ads.n_vars)
        ],
        index=ads.var_names,
    )
    corrss.append(correlations)

# %%
np.array(corrss)
# %%
corr_df = pd.DataFrame(np.array(corrss), columns=ads.var_names)
corr_df = corr_df[10:60][::-1]
corr_df.index = corr_df.index / 100

g = sns.clustermap(
    corr_df,
    row_cluster=False,
    method="complete",
    metric="cosine",
    cmap="coolwarm",
    vmin=-0.4,
    vmax=0.4,
)

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)

# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
sns.barplot(
    data=(df := pd.DataFrame(correlations.sort_values())),
    x=df.index,
    y=0,
    palette="coolwarm",
    hue=0,
    ax=ax,
    width=1,
    linewidth=0,
)

n = 10
ax.set_ylim(-0.5, 0.5)
ax.set_xticks(ax.get_xticks()[::n])
ax.set_xticklabels(df.index[::n], rotation=45, ha="right", fontsize=9)
ax.legend_.remove()
ax.set_xlabel("Gene")
ax.set_ylabel("Correlation with EdU intensity")


# %%

# Sort and display top correlations
correlations_sorted = correlations.sort_values(ascending=True)
print(correlations_sorted.head(20))


# %%
from sklearn.decomposition import PCA

fit = PCA(n_components=2).fit_transform(adata.layers["raw"].toarray().T)

trc = pd.read_csv("neuroRef.csv")
shared = sorted(set(trc["symbol"]) & set(adata.var_names))

loadings = trc = (
    trc[trc["symbol"].isin(shared)].set_index("symbol").reindex(shared).reset_index()[["pc1.rot", "pc2.rot"]]
)


# %%
fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(8, 12), dpi=200)
from scipy import stats

for i in range(4):
    shift = [-0.2 * i, 0]
    wtf = np.log1p(adata[:, shared].layers["raw"].toarray()) @ loadings.to_numpy() + shift

    wtf = adata[:, shared].X.toarray() @ loadings.to_numpy() + shift

    df = pd.DataFrame(
        {
            "theta": (np.arctan2(wtf[:, 1], wtf[:, 0]) + np.pi / 2) % (2 * np.pi) - np.pi,
            "intensity": np.nan_to_num(adata.obs["intensity_mean"], nan=2000),
            "gene": adata[:, "Top2a"].X.toarray().squeeze(),
        }
    )

    print(np.sum(df["theta"] > 0))
    print(np.sum(df["intensity"] > 6000))

    df["idxs"] = np.digitize(df["theta"], np.linspace(-np.pi, np.pi, num_bins))

    u = df.groupby("idxs").size()
    fmt = f"center-shift=({shift[0]:.2f}, {shift[1]:.2f})"
    # sns.lineplot(data=u, x="idxs", y="intensity", ax=axs.flat[i * 2])
    axs.flat[i * 3].set_title(f"Distribution of cells across θ\n{fmt}")
    axs.flat[i * 3].set_xlabel("Tricycle θ")
    axs.flat[i * 3].set_ylabel("Number of cells")
    axs.flat[i * 3 + 1].set_title(f"Top2a expression\n{fmt}")
    axs.flat[i * 3 + 2].set_title(f"EdU intensity\n{fmt}")
    axs.flat[i * 3].plot((np.array(u.index) - (num_bins / 2)) / (num_bins / 2) * np.pi, u.values)

    axs.flat[i * 3 + 1].scatter(df["theta"], df["gene"], s=0.2, alpha=0.5)
    axs.flat[i * 3 + 2].scatter(df["theta"], df["intensity"], s=0.2, alpha=0.5)

plt.tight_layout()
# axs.flat[i * 2 + 1].scatter(df["theta"], adata[:, "Cenpf"].X.toarray().squeeze(), s=0.2)

# axs.flat[i * 2].hexbin(*wtf.T)
# # axs.flat[i * 2 + 1].scatter(
# #     (np.arctan2(wtf[:, 1], wtf[:, 0]) + np.pi / 2) % (2 * np.pi) - np.pi,
# #     np.nan_to_num(adata[:, "Top2a"].X.toarray().squeeze(), nan=2000),
# #     s=0.2,
# # )
# sns.kdeplot(data=df, x="theta", y="gene", hue="intensity", ax=axs.flat[i * 2+1])
# # axs.flat[i*2+1]
# axs.flat[i * 2].set_aspect("equal")
# axs.flat[i * 2].set_xlim(-5, 5)
# axs.flat[i * 2].set_ylim(-5, 5)
# axs.flat[i * 2].hlines(0, 0, 5, color="black", alpha=0.5)
# axs.flat[i * 2].vlines(0, 0, 5, color="black", alpha=0.5)

# %%
bin_mins = np.zeros(num_bins)
bin_maxs = np.zeros(num_bins)

bins = np.histogram(valid.obsm["X_straightened"][:, 0], bins=num_bins)
for b in bins:
    valid.obsm["X_straightened"][:, 0][valid.obsm["X_straightened"][:, 0].__lt__(b[1])]

# np.histogram(ads.obsm["X_straightened"][:, 1], bins=num_bins)

# %%

# %%
sc.pl.embedding(ads, basis="X_straightened", color="Nnat", cmap="Blues", palette=palette_spaco)
# %%
scf.pl.single_trend(ads, basis="spatial", color_exp="k")
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# If displaying in a Jupyter notebook:

# Generate a figure with a polar projection
fg = plt.figure(figsize=(8, 8))
ax = fg.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")

# Define colormap normalization for 0 to 2*pi
norm = mpl.colors.Normalize(-np.pi, np.pi)

# Plot a color mesh on the polar plot
# with the color set by the angle

n = 400  # the number of secants for the mesh
t = np.linspace(-np.pi, np.pi, n)  # theta values
r = np.linspace(0.6, 1, 2)  # radius values change 0.6 to 0 for full circle
rg, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
c = tg  # define color values as theta value
im = ax.pcolormesh(t, r, c.T, norm=norm, cmap="cet_colorwheel")  # plot the colormesh on axis with colormap
ax.set_yticklabels([])  # turn of radial tick labels (yticks)
ax.tick_params(pad=15, labelsize=24)  # cosmetic changes to tick labels
ax.axis("off")  # turn off the axis
ax.spines["polar"].set_visible(False)  # turn off the axis spine.
# %%
