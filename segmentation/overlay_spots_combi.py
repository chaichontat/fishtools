# %%
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

sns.set_theme()
import polars as pl

df = (
    pl.scan_parquet(sorted(Path("chunks").glob("ident_*.parquet")))
    .with_columns(target=pl.col("target").str.split("-").list.get(0), label=pl.col("label").cast(pl.Utf8))
    .group_by(["label", "target"])
    .len()
    .collect()
)
# %%


# %%
polygons = (
    pl.scan_parquet(sorted(Path("chunks").glob("polygons_*.parquet")))
    .filter(pl.col("label").is_not_null())
    .collect()
    .with_columns(centroid=pl.col("centroid").list.to_struct(), label=pl.col("label").cast(pl.Utf8))
    .unnest("centroid")
    .group_by("label")
    .agg(
        area=pl.col("area").sum(),
        centroid_x=((pl.col("field_0") * pl.col("area")).sum() / pl.col("area").sum()),
        centroid_y=((pl.col("field_1") * pl.col("area")).sum() / pl.col("area").sum()),
    )
)
polygons
# %%
# %%
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.hist(polygons["area"], bins=100)
# log scale
ax.set_yscale("log")
ax.set_xscale("log")

# %%


pivoted = (
    df.join(polygons, "label")
    .filter(pl.col("area") > 400)
    .pivot(on="target", index="label", values="len")
    .fill_null(0)
    .sort("label")
)
# %%
unique = df.unique("label")["label"]
final = pivoted.sort("label").to_pandas().set_index("label")

# %%
import anndata as ad
import numpy as np
import scanpy as sc

adata = ad.AnnData(final)


# %%
adata.X = csr_matrix(adata.X)
adata.obs = (
    polygons.filter((pl.col("area") > 400) & pl.col("label").is_in(unique)).to_pandas().set_index("label")
)
adata.write_h5ad("beta.h5ad")

# %%
# adata.write_h5ad("/fast2/3t3clean/analysis/dapi.h5ad")


# %%

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# %%

sc.pp.filter_cells(adata, min_counts=40)
sc.pp.filter_genes(adata, min_cells=1000)  # %%


#
# %%
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts"], jitter=0.8, multi_panel=True, log=True)
# %%


sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)


# %%
sc.pp.highly_variable_genes(adata, n_top_genes=200)
sc.pl.highly_variable_genes(adata)
# %%

sc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)

# %%
sc.pp.neighbors(adata, n_pcs=25)
sc.tl.umap(adata)

# %%

sc.pl.umap(
    adata,
    color="Neurog2",
    # Setting a smaller point size to get prevent overlap
    size=2,
)

# %%
sc.tl.leiden(adata, flavor="igraph", resolution=1)
sc.pl.umap(adata, color=["leiden"])
# %%
# Obtain cluster-specific differentially expressed genes
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")


# %%
sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", standard_scale="var", n_genes=5)
# %%
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
sns.scatterplot(adata.obs, x="centroid_x", y="centroid_y", hue="leiden", s=0.2, ax=ax)
ax.set_aspect("equal")
# disable legend
ax.get_legend().remove()

# %%
sns.FacetGrid(
    adata.obs, hue="leiden", col="leiden", height=4, aspect=1, despine=True, col_wrap=4
).map_dataframe(sns.scatterplot, x="centroid_x", y="centroid_y", s=1)
# disable legend


# %%

sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=200)
# %%

adata.layers["raw"] = adata.X.copy()
adata.layers["sqrt_norm"] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])
sc.experimental.pp.normalize_pearson_residuals(adata)

# %%
sc.tl.pca(adata, n_comps=50)
sc.pl.pca_variance_ratio(adata, log=True)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)
# %%
sc.pl.umap(adata, color="Col1a1", size=2)


# %%
fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

hvgs = adata.var["highly_variable"]
ax.scatter(adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none")
ax.scatter(
    adata.var["mean_counts"][hvgs],
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

# # %%
# adata.layers["raw"] = adata.X.copy()
# adata.layers["sqrt_norm"] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])
# sc.experimental.pp.normalize_pearson_residuals(adata)
# %%
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=20)
sc.pl.pca(adata, color="Col1a1")
# %%
sc.pl.pca_variance_ratio(adata, log=True)

# %%
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)

# %%
sc.tl.umap(adata)
# %%
sc.pl.umap(
    adata,
    color="Col1a1",
    # Setting a smaller point size to get prevent overlap
    size=2,
)

# %%
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.scatter(
    adata.obs["centroid_x"],
    adata.obs["centroid_y"],
    c=np.array(adata[:, "Pax6"].X.todense()),
    s=0.2,
    alpha=0.2,
    cmap="Blues",
)
ax.set_aspect("equal")
# %%
