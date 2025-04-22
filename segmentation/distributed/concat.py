# %%
from pathlib import Path

import polars as pl

path = Path("/working/20250407_cs3_2/analysis/deconv/stitch--br+atp")
df = (
    pl.scan_parquet(
        path / "chunks+*/ident_*.parquet",
        include_file_paths="path",
        allow_missing_columns=True,
    )
    .with_columns(
        z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
        spot_id=pl.col("spot_id").cast(pl.UInt32),
    )
    # .drop("path")
    .sort("z")
    .collect()
)

# %%
from pathlib import Path

# Path("/working/20250327_benchmark_coronal2/analysis/deconv/baysor").mkdir(exist_ok=True, parents=True)

spots = {}
roi = "br"
codebooks = ["mousecommon", "zachDE"]
for codebook in codebooks:
    spots[codebook] = pl.read_parquet(path.parent / f"{roi}+{codebook}.parquet")
    if "index" not in spots[codebook].columns:
        spots[codebook] = spots[codebook].with_row_index()

spots["mousecommon"] = spots["mousecommon"].join(
    df.filter(pl.col("path").str.contains("mousecommon")), left_on="index", right_on="spot_id", how="left"
)
spots["zachDE"] = spots["zachDE"].join(
    df.filter(~pl.col("path").str.contains("mousecommon")), left_on="index", right_on="spot_id", how="left"
)
# %%
spots = pl.concat(spots.values())


# %%

spots.select(
    x="x", y="y", z="z", gene=pl.col("target").str.split("-").list.get(0), cell=pl.col("label").fill_null(0)
).filter(pl.col("gene") != "Blank").write_csv(path.parent / "baysor--br" / "spots.csv")

# %%
polygons = (
    pl.scan_parquet(
        path / "chunks+zachDE/polygons_*.parquet",
        include_file_paths="path",
    )
    .with_columns(z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8))
    .drop("path")
    .sort("z")
    .collect()
)


def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """
    Generate a Polars aggregation expression to take a weighted mean
    https://github.com/pola-rs/polars/issues/7499#issuecomment-2569748864
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(weight_col)
    return weights.dot(values).truediv(weights.sum()).fill_nan(None)


weighted_centroids = (
    polygons.group_by("label")
    .agg(
        x=(pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum(),
        y=(pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum(),
        z=(pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum(),
    )
    .sort("label")
    .to_pandas()
    .set_index("label")
)
weighted_centroids.index = weighted_centroids.index.astype(str)

# %%
# ident = df.join(spots[["index", "target"]], left_on="spot_id", right_on="index", how="left")
# %%
molten = df.group_by(["label", "target"]).agg(pl.len())

# %%
cbg = (
    molten.with_columns(gene_name=pl.col("target").str.split("-").list.get(0))
    .drop("target")
    .pivot("gene_name", index="label", values="len")
    .fill_null(0)
    .sort("label")
    .to_pandas()
    .set_index("label")
)
cbg.index = cbg.index.astype(str)

# %%

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

adata = ad.AnnData(cbg)
adata.obs = weighted_centroids.reindex(adata.obs.index)

# %%
n_genes = adata.shape[1]
sc.pp.calculate_qc_metrics(
    adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
)
sc.pp.filter_cells(adata, min_counts=20)
sc.pp.filter_cells(adata, max_counts=1200)
sc.pp.filter_genes(adata, min_cells=10)
# adata = adata[(adata.obs["y"] < 23189.657075200797) | (adata.obs["y"] > 46211.58630310604)]
adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()

sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts"],
    jitter=0.4,
    multi_panel=True,
)
print(np.median(adata.obs["total_counts"]), len(adata))

# adata.write_h5ad(path / "segmentation_counts.h5ad")
# %%
plt.scatter(adata.obs["x"][::1], adata.obs["y"][::1], s=1, alpha=0.3)


# %%
# %%


def normalize_pearson(adata):
    sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=2000)

    def plot():
        fig, ax = plt.subplots(figsize=(8, 6))

        hvgs = adata.var["highly_variable"]

        ax.scatter(adata.var["means"], adata.var["residual_variances"], s=3, edgecolor="none")
        ax.scatter(
            adata.var["means"][hvgs],
            adata.var["residual_variances"][hvgs],
            c="tab:red",
            label="selected genes",
            s=3,
            edgecolor="none",
        )

        ax.set_xscale("log")
        ax.set_xlabel("mean expression")
        ax.set_yscale("log")
        ax.set_ylabel("residual variance")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        plt.legend()

    adata = adata[:, adata.var["highly_variable"]]

    adata.layers["raw"] = adata.X.copy()
    adata.layers["sqrt_norm"] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])
    sc.experimental.pp.normalize_pearson_residuals(adata)
    return adata, plot


def normalize_total(adata):
    # sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata, None


# adata, plot = normalize_pearson(adata)
# plot()
adata, plot = normalize_pearson(adata)

# %%
# %%


# %%


# %%
# sc.pp.scale(adata, max_value=10)
# %%
sc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)

# %%
import rapids_singlecell as rsc

rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, metric="cosine")
sc.tl.leiden(adata, n_iterations=2, resolution=0.8, flavor="igraph")
rsc.tl.umap(adata, min_dist=0.1, n_components=2)
# sc.tl.umap(adata, min_dist=0.1, n_components=2)
# %%
fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata, color=["leiden"], ax=ax)
ax.set_aspect("equal")

# %%
# sc.pl.embedding(adata, basis="spatial", color="total_counts")

# %%

# %%
fig, ax = plt.subplots(figsize=(8, 6))
gene = "Hes5"
sc.pl.embedding(
    adata,
    color=gene,
    basis="spatial",
    ax=ax,
    cmap="Blues",
    vmax=np.percentile(adata[:, gene].X, 99.9),
    vmin=np.percentile(adata[:, gene].X, 50),
)  # type: ignore
ax.set_aspect("equal")

# %%

sc.pl.umap(
    adata, color=["leiden", "Pax6", "Bcl11b", "Tbr1", "Gad1", "Lhx6", "Satb2", "Fezf2"], cmap="Blues", ncols=2
)

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
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect("equal")
sc.pl.embedding(adata, color="leiden", basis="spatial", ax=ax)

# %%
plt.scatter(adata.obs["x"], adata.obs["y"], s=0.1, alpha=0.3)
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
fig, axs = plt.subplots(figsize=(12, 6), dpi=200, ncols=2)
axs = axs.flatten()
sc.pl.embedding(
    adata,
    color="leiden",
    palette=palette_spaco,
    basis="spatial",
    ax=axs[1],
)  # type: ignore
sc.pl.embedding(
    adata,
    color="leiden",
    palette=palette_spaco,
    basis="umap",
    ax=axs[0],
)  # type: ignore
for ax in axs:
    ax.set_aspect("equal")

# %%
fig, ax = plt.subplots(figsize=(8, 6))

ax.set_aspect("equal")


# %%
def compare_genes(adata, genes, ax=None, dark=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200, facecolor="white" if not dark else "black")

    ax.set_aspect("equal")
    rand = np.random.default_rng(0)
    x = adata[:, genes[0]].X.squeeze() + rand.normal(0, 0.1, len(adata))
    y = adata[:, genes[1]].X.squeeze() + rand.normal(0, 0.1, len(adata))
    print(x)
    if not dark:
        ax.scatter(x, y, s=0.1, alpha=0.1, **kwargs)
    else:
        ax.hexbin(x, y, gridsize=250)
    ax.set_xlabel(genes[0])
    ax.set_ylabel(genes[1])
    ax.set_aspect("equal")
    ax.set_xlim((min(np.min(x), np.min(y)) - 1, max(np.max(x), np.max(y)) + 1))
    ax.set_ylim((min(np.min(x), np.min(y)) - 1, max(np.max(x), np.max(y)) + 1))
    return ax


sns.set_theme()
with sns.axes_style("white"):
    # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(18, 6), dpi=200)
    compare_genes(adata, ["Cux2", "Gad1"], dark=False)
# %%

sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")


# %%
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)


# %%

sns.clustermap(adata.X)

# %%
import pandas as pd
import scipy
import seaborn as sns

# Get gene expression matrix
gene_expr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X

# Create DataFrame with gene names
genes_df = pd.DataFrame(gene_expr, columns=adata.var_names)

# Compute correlation matrix
corr_matrix = genes_df.corr()

# Create clustermap
sns.clustermap(corr_matrix, cmap="RdBu_r", figsize=(12, 12), center=0, vmin=-1, vmax=1)
# %%
