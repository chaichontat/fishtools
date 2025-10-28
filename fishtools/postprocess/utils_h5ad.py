"""AnnData preprocessing utilities shared across postprocess workflows."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "std_log1p",
    "cluster",
    "qc",
    "normalize_pearson",
    "leiden_umap",
    "normalize_total",
    "filter_leiden",
    "run_tricycle",
    "get_leiden_genes",
    "run_spaco",
]


def std_log1p(adata: ad.AnnData, min_genes: int = 200, min_cells: int = 100) -> ad.AnnData:
    """Standard Scanpy normalization: filter, normalize_total, log1p."""

    import scanpy as sc

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def cluster(
    adata: ad.AnnData,
    min_cells: int = 10,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    use_rep: str = "X_pca",
    metric: str = "cosine",
    leiden_resolution: float = 1.0,
) -> ad.AnnData:
    """Convenience wrapper for RAPIDS PCA → neighbors → UMAP → Leiden."""

    import rapids_singlecell as rsc
    import scanpy as sc

    if use_rep == "X_pca":
        sc.pp.filter_genes(adata, min_cells=min_cells)
        rsc.pp.pca(adata)
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, use_rep=use_rep)  # type: ignore[call-arg]
    rsc.tl.umap(adata)
    rsc.tl.leiden(adata, resolution=leiden_resolution)
    return adata


def qc(adata: ad.AnnData) -> ad.AnnData:
    """Calculate QC metrics with defenses for low gene counts."""

    import scanpy as sc

    n_genes = adata.shape[1]
    sc.pp.calculate_qc_metrics(
        adata,
        inplace=True,
        percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes),
    )
    return adata


def normalize_pearson(adata: ad.AnnData, n_top_genes: int = 2000):
    """Select HVGs via Pearson residuals, then normalize residuals."""

    import scanpy as sc

    sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=n_top_genes)

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


def leiden_umap(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    metric: str = "cosine",
    resolution: float = 0.8,
    min_dist: float = 0.1,
) -> ad.AnnData:
    """Run RAPIDS neighbors + Leiden + UMAP with configurable parameters."""

    import rapids_singlecell as rsc
    import scanpy as sc

    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    sc.tl.leiden(adata, n_iterations=2, resolution=resolution, flavor="igraph")
    rsc.tl.umap(adata, min_dist=min_dist, n_components=2)
    return adata


def normalize_total(adata: ad.AnnData):
    """Return adata.log1p normalized copy and placeholder plot callable."""

    import scanpy as sc

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata, None


def filter_leiden(adata: ad.AnnData, keep: Sequence[int | str]) -> ad.AnnData:
    """Subset AnnData to specified Leiden clusters."""

    return adata[adata.obs["leiden"].isin(keep)]


def run_tricycle(adata: ad.AnnData, trc: pd.DataFrame) -> ad.AnnData:
    """Project AnnData onto tricycle cell-cycle embeddings."""

    shared = sorted(set(trc["symbol"]) & set(adata.var_names))
    loadings = (
        trc[trc["symbol"].isin(shared)]
        .set_index("symbol")
        .reindex(shared)
        .reset_index()[["pc1.rot", "pc2.rot"]]
    )
    pls = adata[:, shared].X @ loadings.to_numpy()
    adata.obsm["tricycle"] = pls
    adata.obs["tricycle"] = (
        np.arctan2(adata.obsm["tricycle"][:, 1], adata.obsm["tricycle"][:, 0] + 1.0) + np.pi / 2
    ) % (2 * np.pi) - np.pi
    return adata


def get_leiden_genes(adata: ad.AnnData, group: int | str | Literal["all"], head: int = 5):
    """Retrieve top marker genes for a Leiden cluster or all clusters."""

    import scanpy as sc

    if group == "all":
        return sorted(
            set(
                chain.from_iterable(
                    [
                        sc.get.rank_genes_groups_df(adata, group=str(c)).head(head)["names"]
                        for c in adata.obs["leiden"].cat.categories
                    ]
                )
            )
        )
    return sc.get.rank_genes_groups_df(adata, group=str(group)).head(head)["names"]


def run_spaco(adata: ad.AnnData, n_neighbors: int = 15, radius: float = 0.1):
    """Generate color palettes via SPAco for Leiden clusters."""

    import spaco

    color_mapping = spaco.colorize(
        cell_coordinates=adata.obsm["spatial"],
        cell_labels=adata.obs["leiden"],
        radius=radius,
        n_neighbors=n_neighbors,
        colorblind_type="none",
    )

    color_mapping = {k: color_mapping[k] for k in adata.obs["leiden"].cat.categories}
    return list(color_mapping.values())
