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
from sklearn.metrics import roc_auc_score
import tifffile
from shapely import MultiPolygon, Point, Polygon, STRtree

from fishtools.postprocess import normalize_total

sns.set_theme()
plt.rcParams["figure.dpi"] = 200
path = Path("/working/20251001_JaxA3_Coro11/analysis/deconv/baysor")
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
sc.tl.leiden(adata, n_iterations=2, resolution=2, flavor="igraph")
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
    sc.pl.embedding(adata, color="leiden", basis="spatial", ax=ax, groups=[str(i)])
    ax.set_aspect("equal")
    plt.show()

# %%
# import rapids_singlecell as rsc

# rsc.gr.spatial_autocorr(adata_gpu, mode="moran", genes=adata.var_names, n_perms=100, use_sparse=False)
# adata_gpu = adata.copy()
# rsc.pp.neighbors(adata_gpu, n_neighbors=20, n_pcs=30)
# %%
sc.pl.umap(
    adata,
    color=["leiden", "Pax6-201", "Gad1-201", "Satb2-202", "Tbr1", "Lhx6"],
    frameon=False,
    cmap="Blues",
    ncols=2,
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
    np.asarray(
        adata[:, [x for x in adata.var_names if x.startswith("Slc17a7")][0]].X.todense().flatten()
        + np.random.normal(0, 0.5, size=adata.shape[0])
    ).flatten(),
    np.asarray(
        adata[:, "Gad1-201"].X.todense().flatten() + np.random.normal(0, 0.5, size=adata.shape[0])
    ).flatten(),
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

def match_gene_names(adata, gene_list):
    """Match gene names from a list to var_names, accounting for isoform suffixes.

    For each gene in gene_list, returns the first matching var_name that starts
    with the gene name followed by a hyphen (e.g., 'Pax6' matches 'Pax6-201').
    Falls back to exact match if no suffix variant is found.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with var_names to search
    gene_list : list[str]
        List of gene names (without suffix)

    Returns
    -------
    list[str]
        List of matched var_names from adata
    """
    matched = []
    for gene in gene_list:
        # First try to find a variant with suffix
        candidates = [v for v in adata.var_names if v.startswith(f"{gene}-")]
        if candidates:
            matched.append(candidates[0])
        elif gene in adata.var_names:
            # Fallback to exact match
            matched.append(gene)
        else:
            # Gene not found, keep original name (will fail in scanpy with clear error)
            matched.append(gene)
    return matched



genes_to_plot = [
    "leiden",
    "Pax6",
    "Sox2",
    "Hes5",
    "Slc17a7",
    "Lhx2",
    "Lhx6",
    "Pdgfra",
    "Gad1",
    "Cux2",
    "Eomes",
    "Foxp2",
]

sc.pl.embedding(
    adata,
    basis="umap",
    color=match_gene_names(adata, genes_to_plot),
    frameon=False,
    cmap="Blues",
    return_fig=True,
)
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

key = "log_brdu_mean"
fig, axs = plot_wheel(
    np.nan_to_num(adata.obsm["tricycle"], nan=0),
    scatter_cmap="RdBu",
    c=adata.obs[key],
    alpha=0.1,
    # c=adata.obs["total_intensity"],
    # scatter_cmap="RdBu_r",
    colorize_background=False,
    fig=fig,
    vmin=np.percentile(adata.obs[key], 80),
    vmax=np.percentile(adata.obs[key], 99),
    colorbar_label="θ",
)
# %%
fig, axs = plot_wheel(
    np.nan_to_num(m.obsm["tricycle"], nan=0),
    # scatter_cmap="Blues",
    alpha=0.1,
    c=m.obs["edu_mean"],
    scatter_cmap="Blues",
    colorize_background=False,
    fig=fig,
    # vmax=np.percentile(adata.obs["total_intensity"], 99),
    colorbar_label="mean EdU intensity",
)


# %%
t = adata.obsm["tricycle"].copy()
t[:, 0] += 0.5
plot_wheel(t)

# %%


def compare_genes(adata, genes, ax=None, jitter=0.02, dark=False, quadrant_thresholds=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200, facecolor="white" if not dark else "black")

    ax.set_aspect("equal")
    rand = np.random.default_rng(0)
    # Ensure gene names are valid and exist in adata
    valid_genes = [gene for gene in genes if gene in adata.var_names]
    if len(valid_genes) < 2:
        raise ValueError(f"At least two valid genes are required. Found: {valid_genes} in {genes}")

    x = adata[:, valid_genes[0]].X.squeeze() + rand.normal(0, jitter, len(adata))
    y = adata[:, valid_genes[1]].X.squeeze() + rand.normal(0, jitter, len(adata))
    print(x)
    if not dark:
        ax.scatter(x, y, s=0.1, alpha=0.1, **kwargs)
    else:
        ax.hexbin(x, y, gridsize=250)
    ax.set_xlabel(valid_genes[0])
    ax.set_ylabel(valid_genes[1])
    ax.set_aspect("equal")
    min_val = min(np.min(x), np.min(y)) - 1
    max_val = max(np.max(x), np.max(y)) + 1
    ax.set_xlim((min_val, max_val))
    ax.set_ylim((min_val, max_val))

    if quadrant_thresholds is not None and len(quadrant_thresholds) == 2:
        x_thresh, y_thresh = quadrant_thresholds
        ax.axhline(y_thresh, color="grey", linestyle="--", lw=1)
        ax.axvline(x_thresh, color="grey", linestyle="--", lw=1)

        total_points = len(x)
        if total_points > 0:
            q_tr = np.sum((x >= x_thresh) & (y >= y_thresh))
            q_tl = np.sum((x < x_thresh) & (y >= y_thresh))
            q_bl = np.sum((x < x_thresh) & (y < y_thresh))
            q_br = np.sum((x >= x_thresh) & (y < y_thresh))

            perc_tr = (q_tr / total_points) * 100
            perc_tl = (q_tl / total_points) * 100
            perc_bl = (q_bl / total_points) * 100
            perc_br = (q_br / total_points) * 100

            # Position text relative to plot limits and thresholds
            text_props = dict(ha="center", va="center", fontsize=12, color="black" if not dark else "white")

            # Adjust text position to be within the plot and quadrant
            x_range = max_val - min_val
            y_range = max_val - min_val

            ax.text(
                x_thresh + 0.5 * (max_val - x_thresh),
                y_thresh + 0.5 * (max_val - y_thresh),
                f"{perc_tr:.1f}%",
                **text_props,
            )  # TR
            ax.text(
                x_thresh - 0.5 * (x_thresh - min_val),
                y_thresh + 0.5 * (max_val - y_thresh),
                f"{perc_tl:.1f}%",
                **text_props,
            )  # TL
            ax.text(
                x_thresh - 0.5 * (x_thresh - min_val),
                y_thresh - 0.5 * (y_thresh - min_val),
                f"{perc_bl:.1f}%",
                **text_props,
            )  # BL
            ax.text(
                x_thresh + 0.5 * (max_val - x_thresh),
                y_thresh - 0.5 * (y_thresh - min_val),
                f"{perc_br:.1f}%",
                **text_props,
            )  # BR
    return ax


adata.X = np.asarray(adata.X.todense())

sns.set_theme()
with sns.axes_style("white"):
    compare_genes(
        adata,
        genes=["Satb2-202", "Gad1-201"],
        jitter=0.02,
        dark=False,
        quadrant_thresholds=(0.1, 0.1),
        color="blue",
    )
# %%
