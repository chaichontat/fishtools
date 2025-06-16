# %%
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
import tifffile
from shapely import MultiPolygon, Point, Polygon, STRtree

from fishtools.postprocess import normalize_total

sns.set_theme()
plt.rcParams["figure.dpi"] = 200
path = Path("/working/20250612_ebe00219_3/analysis/deconv/baysor10")
# path = Path("/working/20250407_cs3_2/analysis/deconv/baysor--br/point8")
adata = sc.read_loom(path / "segmentation_counts.loom")
adata.var_names = adata.var["Name"]

# adata = adata[~adata.obs["y"].between(24000, 45000) & adata.obs["x"].gt(2000)]
# %%
# %%

# %%
# adata.obs["batch"] = "trc3"
# # %%
# adata2 = sc.read_loom(Path("/working/20241119-ZNE172-Trc/baysor--all") / "segmentation_counts.loom")
# adata2.obs["batch"] = "trc"
# adata2.var_names = adata.var["Name"]
# %%
plt.scatter(adata.obs["y"], adata.obs["x"], s=0.3, alpha=0.4)
# adata = ad.concat([adata, adata2])
# %%

n_genes = adata.shape[1]
sc.pp.calculate_qc_metrics(
    adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
)
sc.pp.filter_cells(adata, min_counts=20)
sc.pp.filter_genes(adata, min_cells=10)
# adata = adata[(adata.obs["y"] < 23189.657075200797) | (adata.obs["y"] > 46211.58630310604)]
adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
adata.write_h5ad(path / "segmentation_counts_filtered.h5ad")
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts"],
    jitter=0.4,
    multi_panel=True,
)
print(np.median(adata.obs["total_counts"]))

# adata.write_h5ad(path / "segmentation_counts.h5ad")


# sc.pp.log1p(adata)
# %%
# %%
normalize_total(adata)
# sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=2000)
# %%

# %%
# sc.pp.scale(adata, max_value=10)
# %%
import rapids_singlecell as rsc

rsc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)

# %%

rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, metric="cosine")
sc.tl.leiden(adata, n_iterations=2, resolution=0.8, flavor="igraph")
rsc.tl.umap(adata, min_dist=0.1, n_components=2)
# sc.tl.umap(adata, min_dist=0.1, n_components=2)
# %%

# sc.pl.embedding(adata, basis="spatial", color="total_counts")

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
# %%
sc.pl.umap(adata, color=["leiden"])
# %%

# %%
for i in range(adata.obs["leiden"].cat.categories.size):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    sc.pl.embedding(adata, color="leiden", basis="spatial", ax=ax, palette=palette_spaco, groups=[str(i)])
    ax.set_aspect("equal")
    plt.show()

# %%
# import rapids_singlecell as rsc

# rsc.gr.spatial_autocorr(adata_gpu, mode="moran", genes=adata.var_names, n_perms=100, use_sparse=False)
# adata_gpu = adata.copy()
# rsc.pp.neighbors(adata_gpu, n_neighbors=20, n_pcs=30)
# %%
sc.pl.umap(
    adata, color=["leiden", "Pax6", "Gad1", "Satb2", "Tbr1", "Lhx6"], frameon=False, cmap="Blues", ncols=2
)

# %%
with sns.axes_style("darkgrid"):
    sc.pl.rank_genes_groups(adata, n_genes=15, sharey=False, fontsize=14, show=False)
    for ax in plt.gcf().axes:
        for text in ax.texts:
            text.set_rotation(40)

# %%

sc.tl.dendrogram(adata, groupby="leiden")
sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", standard_scale="var", n_genes=5)
for ax in plt.gcf().axes:
    for text in ax.texts:
        text.set_rotation(40)
# %%
sc.pl.embedding(
    adata,
    color=["leiden"],
    basis="spatial",
    vmin=2,
    vmax=10,
    # legend_loc="on data",
    frameon=False,
    cmap="Blues",
    show=False,
)
fig = plt.gcf()
fig.set_size_inches(8, 10)
for ax in fig.axes:
    ax.set_aspect("equal")

# %%


# %%
adata.write_h5ad(path / "pearsonedcortex.h5ad")
# %%

plt.scatter(
    adata[:, "Slc17a7"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    adata[:, "Gad1"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    s=0.3,
    alpha=0.4,
)

# %%
plt.scatter(
    adata[:, "Gad1"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    adata[:, "Npy"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    s=0.3,
    alpha=0.4,
)

# %%
plt.scatter(
    adata[:, "Satb2"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    adata[:, "Fezf2"].X.flatten() + np.random.normal(0, 0.5, size=adata.shape[0]),
    s=0.3,
    alpha=0.4,
)
# %%

edu_stats = pd.read_csv(path / "edu_stats.csv")
edu_stats["label"] = [f"{x:.1f}" for x in edu_stats["label"]]
edu_stats = edu_stats.set_index("label")
# %%
cfse_stats = pd.read_csv(path / "cfse_stats.csv")
cfse_stats["label"] = [f"{x:.1f}" for x in cfse_stats["label"]]
cfse_stats = cfse_stats.set_index("label")
# %%
adata.obs = adata.obs.join(edu_stats, rsuffix="_edu", how="left")
adata.obs = adata.obs.join(cfse_stats, rsuffix="_cfse", how="left")

# %%
sc.pl.embedding(adata, basis="spatial", color="intensity_mean_cfse", frameon=False, cmap="magma")
# %%

sc.pl.umap(adata, color=["intensity_mean"], frameon=False, cmap="viridis")
# %%

trc = pd.read_csv("neuroRef.csv")
shared = sorted(set(trc["symbol"]) & set(adata.var_names))

loadings = trc = (
    trc[trc["symbol"].isin(shared)].set_index("symbol").reindex(shared).reset_index()[["pc1.rot", "pc2.rot"]]
)

pls = adata[:, shared].X @ loadings.to_numpy()
adata.obsm["tricycle"] = pls
adata.obs["tricycle"] = (
    np.arctan2(adata.obsm["tricycle"][:, 1], adata.obsm["tricycle"][:, 0] + 1.0) + np.pi / 2
) % (2 * np.pi) - np.pi
sc.pl.embedding(
    adata,
    color=["tricycle"],
    basis="spatial",
    # legend_loc="on data",
    frameon=False,
    # cmap="Blues",
    cmap="cet_colorwheel",
    # palette=palette_spaco,
    show=False,
)
fig = plt.gcf()
fig.set_size_inches(8, 10)
for ax in fig.axes:
    ax.set_aspect("equal")
# %%
sc.pl.umap(adata, color=["leiden", "tricycle"], frameon=False, cmap="cet_colorwheel")


# %%
from itertools import chain

sc.set_figure_params(facecolor="black", frameon=False)
# Set background black and all text elements to white
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "text.color": "gray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

print(
    "\n".join([
        f"Group {c}: " + ", ".join(sc.get.rank_genes_groups_df(adata, group=c).head(5)["names"])
        for c in adata.obs["leiden"].cat.categories
    ])
)
# %%
import spaco

color_mapping = spaco.colorize(
    cell_coordinates=adata.obsm["spatial"],
    cell_labels=adata.obs["leiden"],
    radius=0.1,
    n_neighbors=15,
    colorblind_type="none",
)
# %%

color_mapping = {k: color_mapping[k] for k in adata.obs["leiden"].cat.categories}
palette_spaco = list(color_mapping.values())

# %%
genes = sorted(
    set(
        chain.from_iterable([
            sc.get.rank_genes_groups_df(adata, group=c).head(5)["names"]
            for c in adata.obs["leiden"].cat.categories
        ])
    )
)

p = sc.pl.umap(
    adata,
    color=genes,
    # basis="spatial",
    legend_loc="on data",
    frameon=False,
    cmap="magma",
    return_fig=True,
    ncols=8,
    vmin=[np.percentile(adata[:, gene].X, 1) for gene in genes],
    vmax=[np.percentile(adata[:, gene].X, 99.99) for gene in genes],
)

for ax in p.axes:
    ax.tick_params(colors="gray")
    ax.yaxis.label.set_color("gray")

plt.show()

# %%
genes = ["Pax6"]
p = sc.pl.embedding(
    adata,
    basis="spatial",
    color=genes,
    # basis="spatial",
    frameon=False,
    cmap="magma",
    return_fig=True,
    ncols=8,
    vmin=[np.percentile(adata[:, gene].X, 1) for gene in genes],
    vmax=[np.percentile(adata[:, gene].X, 99.9) for gene in genes],
)

for ax in p.axes:
    ax.tick_params(colors="gray")
    ax.yaxis.label.set_color("gray")

plt.show()


# %%
import polars as pl

genes = ["CD163", "LRRK2"]
coords_umap = adata.obsm["X_umap"]
df = pl.concat(
    [
        pl.DataFrame(adata.to_df()),
        pl.DataFrame(adata.obsm["X_umap"]).rename({"column_0": "X_umap", "column_1": "Y_umap"}),
    ],
    how="horizontal",
)
# %%

fig, axs = plt.subplots(figsize=(8, 6), dpi=200, ncols=2)
for ax, gene in zip(axs, genes):
    # sns.scatterplot(data=df[::4], x="X_umap", y="Y_umap", hue=gene, ax=ax, s=0.5, alpha=0.3)
    ax.scatter(*coords_umap.T, s=0.1, alpha=0.3, c=adata[:, "CD163"], cmap="inferno")

# %%
# sc.pl.umap(
#     adata,
#     color=[
#         "Neurod1",
#         "Neurod6",
#         "Pax6",
#         "Vim",
#         "Eomes",
#         "Tbr1",
#         "Fezf2",
#         "Bcl11b",
#         "Top2a",
#         "Hes5",
#         "Gad2",
#         "Sst",
#     ],
#     cmap="Blues",
# )

# %%
for c in adata.obs["leiden"].cat.categories:
    print(f"Cluster {c}")
    dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=c).head(3)["names"]
    sc.pl.umap(
        adata,
        color=[*dc_cluster_genes],  # , "leiden"],
        legend_loc="on data",
        frameon=False,
        ncols=3,
        # cmap="Blues",
        palette=palette_spaco,
    )
    plt.show()

# %%
for c in adata.obs["leiden"].cat.categories:
    print(f"Cluster {c}")
    dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=c).head(3)["names"]
    sc.pl.embedding(
        adata,
        color=[*dc_cluster_genes],
        basis="spatial",
        legend_loc="on data",
        frameon=False,
        ncols=3,
        show=False,
        # cmap="Blues",
        # palette=palette_spaco,
    )
    fig = plt.gcf()
    for ax in fig.axes:
        ax.set_aspect("equal")
    plt.show()

# %%
from fishtools.utils.plot import plot_wheel

# %%


# %%
t = adata.obsm["tricycle"].copy()
t[:, 0] += 0.5
plot_wheel(t)

# %%
