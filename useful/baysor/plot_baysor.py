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
from scipy.interpolate import splev, splrep
from shapely import MultiPolygon

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

print(
    "\n".join([
        f"Cluster {c}: "
        + ", ".join([
            f"{entry['names']}: {np.round(entry['scores'], 2)}"
            for entry in sc.get.rank_genes_groups_df(adata, group=c)
            .head(10)[["names", "scores"]]
            .to_dict(orient="records")
        ])
        for c in adata.obs["leiden"].cat.categories
    ])
)
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
        chain.from_iterable([
            sc.get.rank_genes_groups_df(adata, group=c).head(6)["names"]
            for c in adata.obs["leiden"].cat.categories
        ])
    )
)

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

fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300)
sc.pl.umap(
    adata,
    color="tricycle",
    frameon=False,
    cmap="cet_colorwheel",
    show=False,
    vmin=-np.pi,
    s=10,
    vmax=np.pi,
    colorbar_loc=None,
    ax=axs[0],
)
sc.pl.embedding(
    adata,
    color="tricycle",
    basis="spatial",
    frameon=False,
    cmap="cet_colorwheel",
    show=False,
    s=5,
    vmin=-np.pi,
    vmax=np.pi,
    alpha=1,
    # colorbar_loc=None,
    ax=axs[1],
)
for ax in axs:
    ax.set_aspect("equal")
plt.tight_layout()

# %%

fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300, facecolor="black")
sc.pl.umap(
    adata,
    title="EdU mean intensity",
    color="intensity_mean",
    frameon=False,
    cmap="turbo",
    show=False,
    s=6,
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
    frameon=False,
    cmap="turbo",
    show=False,
    s=1,
    vmin=np.nanpercentile(adata.obs["intensity_mean"], 20),
    vmax=np.nanpercentile(adata.obs["intensity_mean"], 99.9),
    colorbar_loc=None,
    ax=axs[1],
)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_title(ax.get_title(), color="white", fontsize=16)
plt.tight_layout()
# %%
cluster_mapping = {
    "0": "Transitional astrocyte",
    "1": "Upper-layer GABAergic interneurons",
    "2": "Maturing deep layer excitatory neurons",
    "3": "Late-stage differentiating excitatory neurons",
    "4": "G1/S phase neural progenitors",
    "5": "S-phase neural progenitors",
    "6": "Immature interneurons",
    "7": "Neurogenic radial glia",
    "8": "Differentiating intermediate progenitors",
    "9": "Endothelial tip cells",
    "10": "Cortical neuroblasts",
    "11": "Immature excitatory neurons",
    "12": "Mixed-state radial glia",
    "13": "G2/M phase neural stem cells",
    "14": "G2-phase neural stem cells",
    "15": "Immature somatostatin interneurons",
}

# plt.scatter(adata.obs["tricycle"], adata.obs["intensity_mean"], s=1)
# %%

fig, axs = plt.subplots(figsize=(12, 6), ncols=2, dpi=300, facecolor="black")
sc.pl.umap(
    adata,
    color="intensity_mean_cfse",
    title="CFSE mean intensity",
    frameon=False,
    cmap="turbo",
    show=False,
    s=6,
    vmin=np.nanpercentile(adata.obs["intensity_mean_cfse"], 30),
    vmax=np.nanpercentile(adata.obs["intensity_mean_cfse"], 99.9),
    colorbar_loc=None,
    ax=axs[0],
)
sc.pl.embedding(
    adata,
    color="intensity_mean_cfse",
    title="CFSE mean intensity",
    basis="spatial",
    frameon=False,
    cmap="turbo",
    show=False,
    s=2,
    vmin=np.nanpercentile(adata.obs["intensity_mean_cfse"], 30),
    vmax=np.nanpercentile(adata.obs["intensity_mean_cfse"], 99.9),
    colorbar_loc=None,
    ax=axs[1],
)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_title(ax.get_title(), color="white", fontsize=16)
plt.tight_layout()

# %%
# sns.scatterplot(data=adata.obs, x="tricycle", y="intensity_mean_edu", s=1, alpha=0.2)
# plt.yscale("log")
plt.scatter(adata[:, "Top2a"].X.squeeze(), adata.obs["intensity_mean"], s=1)

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
        mask = adata.obs["leiden"] != cluster
        adata.obs["tricycle_masked"] = adata.obs["tricycle"].copy()
        adata.obs.loc[mask, "tricycle_masked"] = np.nan

        shared = dict(
            color="tricycle_masked",
            legend_loc=None,
            frameon=False,
            cmap="cet_colorwheel",
            title="",
            show=False,
            colorbar_loc=None,
        )
        sc.pl.umap(adata, ax=axes[idx * 3], **shared)
        sc.pl.embedding(adata, ax=axes[idx * 3 + 1], basis="spatial", **shared)  # type: ignore
        sc.pl.embedding(adata, ax=axes[idx * 3 + 2], basis="X_straightenedscale", **shared)  # type: ignore

    for i, (ax, cluster) in enumerate(
        zip(fig.axes, chain.from_iterable(repeat(cluster, 3) for cluster in clusters))
    ):
        if i % 3 == 0:
            ax.set_title(f"Cluster {cluster}\n{cluster_mapping[cluster]}", fontsize=16, loc="left")
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
ax1.set_title("Original Space")
ax1.set_aspect("equal")
# ax1.legend()

# Second subplot - straightened version
ax2.scatter(closest_t, distances, alpha=0.5, s=1.5, c=ads.obs["tricycle"], cmap="cet_colorwheel")
ax2.axhline(y=0, color="b", linestyle="-", label="Straightened curve")
ax2.set_title("Straightened Space")
ax2.set_xlabel("t parameter")
ax2.set_ylabel("Signed distance")
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
for i in range(num_bins):
    mask = bin_indices == i
    if np.any(mask):
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
    ads.obsm["X_straightened"][:, 1] - bin_mins[bin_indices]
)  # splev(ads.obsm["X_straightened"][:, 0], lowers)[1])
result_scaled = result / (bin_maxs[bin_indices] - bin_mins[bin_indices])

result *= 0.216
# %%
ads.obsm["X_straightenedsub"] = np.stack([ads.obsm["X_straightened"][:, 0], result], axis=1)
ads.obsm["X_straightenedscale"] = np.stack([ads.obsm["X_straightened"][:, 0], result_scaled], axis=1)

# %%
# Subtract minimum value from each point based on its bin
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    ads.obsm["X_straightened"][:, 0],
    result_scaled,  # _scaled,
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
# Subtract minimum value from each point based on its bin
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    ads.obsm["X_straightenedsub"][:, 0],
    ads.obsm["X_straightenedsub"][:, 1],
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
valid = ads[ads.obsm["X_straightenedsub"][:, 0].__gt__(0.1) & ads.obsm["X_straightenedsub"][:, 0].__lt__(0.7)]
bin_edges = np.linspace(-10, 150, 100)
bin_indices = np.digitize(valid.obsm["X_straightenedsub"][:, 1], bin_edges) - 1
out_weights = np.zeros([bin_edges.shape[0], ads.X.shape[1]])
for b in range(bin_edges.shape[0]):
    mask = bin_indices == b
    if np.any(mask):
        out_weights[b] = np.sum(valid.X[mask], axis=0)
# %%
import colorcet as cc
from matplotlib.cm import get_cmap

out_df = pd.DataFrame(
    out_weights[::-1],
    index=list(reversed(list(map(lambda x: str(np.round(x, 2)), bin_edges)))),
    columns=valid.var_names,
)
from matplotlib.cm import get_cmap

pls = (out_df[shared_genes] @ loadings.to_numpy()).to_numpy()
bulked_score = np.arctan2(pls[:, 1], pls[:, 0])

g = sns.clustermap(
    out_df,
    cmap="coolwarm",
    method="complete",  # clustering method
    metric="correlation",  # distance metric
    row_cluster=False,
    row_colors=get_cmap("cet_colorwheel")((bulked_score + np.pi) / (2 * np.pi)),
    # standard_scale=1,
    figsize=(20, 10),
    dendrogram_ratio=(0.1, 0.1),  # size of dendrograms
    vmin=-50,
    vmax=50,
)

# Rotate x-axis labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=12)

# %%

for c in adata.obs["leiden"].cat.categories:
    _out_df = out_df[sc.get.rank_genes_groups_df(adata, group=c).head(20)["names"]]
    g = sns.clustermap(
        _out_df,
        cmap="coolwarm",
        method="complete",  # clustering method
        metric="correlation",  # distance metric
        row_cluster=False,
        row_colors=get_cmap("cet_colorwheel")((bulked_score + np.pi) / (2 * np.pi)),
        # standard_scale=1,
        figsize=(8, 6),
        dendrogram_ratio=(0.1, 0.1),  # size of dendrograms
        vmin=-50,
        vmax=50,
    )

    # Rotate x-axis labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=20)
    g.ax_heatmap.set_xlabel("")
    # set title
    g.ax_col_dendrogram.set_title(
        f"Cluster {c}\n{cluster_mapping[c]}", fontsize=28, loc="left", y=1.5, x=-0.035
    )
    g.ax_cbar.remove()
    # Get current size
    # Extend width (e.g. double it)
    fig = g.figure
    current_width, current_height = fig.get_size_inches()
    fig.set_size_inches(current_width * 2, current_height * 2)

    # Compress existing elements to the left half
    fig.subplots_adjust(top=0.95, bottom=0.6, left=0, right=1)
    axs = [
        fig.add_axes([0.085, 0.0, 0.35, 0.5]),
        fig.add_axes([0.42, 0.1, 0.3, 0.4]),
        fig.add_axes([0.65, 0.1, 0.35, 0.4]),
    ]

    mask = adata.obs["leiden"] != c
    adata.obs["tricycle_masked"] = adata.obs["tricycle"].copy()
    adata.obs.loc[mask, "tricycle_masked"] = np.nan

    shared = dict(
        color="tricycle_masked",
        legend_loc=None,
        frameon=False,
        cmap="cet_colorwheel",
        title="",
        show=False,
        colorbar_loc=None,
    )
    sc.pl.embedding(adata, ax=axs[0], basis="X_straightenedscale", s=30, **shared)  # type: ignore
    sc.pl.embedding(adata, ax=axs[1], basis="spatial", **shared)  # type: ignore
    sc.pl.umap(adata, ax=axs[2], s=25, **shared)
    axs[1].set_aspect("equal")
    axs[2].set_aspect("equal")

    plt.show()


#         sc.pl.umap(adata, ax=axes[idx * 3], **shared)
#         sc.pl.embedding(adata, ax=axes[idx * 3 + 1], basis="spatial", **shared)  # type: ignore
#         sc.pl.embedding(adata, ax=axes[idx * 3 + 2], basis="X_straightenedscale", **shared)  # type: ignore

#     for i, (ax, cluster) in enumerate(
#         zip(fig.axes, chain.from_iterable(repeat(cluster, 3) for cluster in clusters))
#     ):
#         if i % 3 == 0:
#             ax.set_title(f"Cluster {cluster}\n{cluster_mapping[cluster]}", fontsize=16, loc="left")
#         ax.tick_params(colors="gray")
#         ax.yaxis.label.set_color("gray")
#         if i % 3 != 2:
#             ax.set_aspect("equal")
#         if not ax.has_data():
#             fig.delaxes(ax)


# plt.tight_layout()
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
