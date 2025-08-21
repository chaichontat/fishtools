# %%

from collections.abc import Sequence
from itertools import chain
from typing import Literal, cast

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

from fishtools.utils.utils import create_rotation_matrix


def std_log1p(adata: ad.AnnData, min_genes: int = 200, min_cells: int = 100):
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
):
    import rapids_singlecell as rsc

    if use_rep == "X_pca":
        sc.pp.filter_genes(adata, min_cells=min_cells)
        rsc.pp.pca(adata)
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric, use_rep=use_rep)  # type: ignore[call-arg]
    rsc.tl.umap(adata)
    rsc.tl.leiden(adata, resolution=leiden_resolution)
    return adata


def qc(adata: ad.AnnData):
    n_genes = adata.shape[1]
    # Necessary due to a scanpy bug. Need to specify the number of genes when n_genes < 1000.
    sc.pp.calculate_qc_metrics(
        adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
    )
    return adata


def normalize_pearson(adata: ad.AnnData, n_top_genes: int = 2000):
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
):
    import rapids_singlecell as rsc

    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    sc.tl.leiden(adata, n_iterations=2, resolution=resolution, flavor="igraph")
    rsc.tl.umap(adata, min_dist=min_dist, n_components=2)
    return adata


def normalize_total(adata: ad.AnnData):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata, None


def filter_leiden(adata: ad.AnnData, keep: Sequence[int | str]) -> ad.AnnData:
    return adata[adata.obs["leiden"].isin(keep)]


def run_tricycle(adata: ad.AnnData, trc: pd.DataFrame):
    shared = sorted(set(trc["symbol"]) & set(adata.var_names))
    loadings = trc = (
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


def plot_genes(
    adata: ad.AnnData,
    genes: str | Sequence[str],
    basis: str = "spatial",
    n_cols: int = 3,
    perc: tuple[float, float] = (1, 99.99),
    cmap: str = "Blues",
):
    fig = cast(
        Figure,
        sc.pl.embedding(
            adata,
            color=genes,
            basis=basis,
            legend_loc="on data",
            frameon=False,
            cmap=cmap,
            return_fig=True,
            ncols=n_cols,
            vmin=[np.percentile(adata[:, gene].X, perc[0]) for gene in genes],
            vmax=[np.percentile(adata[:, gene].X, perc[1]) for gene in genes],
        ),
    )
    for ax in fig.axes:
        ax.set_aspect("equal")
    return fig


def plot_leiden_genes(adata: ad.AnnData):
    for c in adata.obs["leiden"].cat.categories:
        print(f"Cluster {c}")
        dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=c).head(3)["names"].tolist()
        fig = plot_genes(adata, dc_cluster_genes, basis="spatial", n_cols=3, perc=(1, 99.5), cmap="Blues")
        for ax in fig.axes:
            ax.set_aspect("equal")
        plt.show()


def rotate_label(axes: Sequence[plt.Axes], angle: float = 40):
    for ax in axes:
        for text in ax.texts:
            text.set_rotation(angle)


def plot_ranked_genes(
    adata: ad.AnnData,
    n_genes: int = 15,
    fontsize: int = 14,
):
    with sns.axes_style("darkgrid"):
        sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False, fontsize=fontsize, show=False)
        rotate_label(plt.gcf().axes, angle=40)


def jitter(data: np.ndarray, amount: float = 0.5, seed: int | None = None):
    rand = np.random.default_rng(seed)
    return data + rand.normal(0, amount, size=data.shape[0])


def rotate_rois_in_adata(adata: ad.AnnData, roi_rotation_angles: dict[str, float]) -> ad.AnnData:
    """
    Rotates spatial coordinates in an AnnData object for specified ROIs around their respective centers.

    Args:
        adata: The AnnData object with 'roi' in obs and 'spatial' in obsm.
        roi_rotation_angles: A dictionary where keys are ROI names (matching those in adata.obs['roi'])
                             and values are the rotation angles in degrees.

    Returns:
        A new AnnData object with rotated spatial coordinates.
    """
    spatial_coords = adata.obsm["spatial"].copy()

    for roi_name, angle_degrees in roi_rotation_angles.items():
        # Find indices for the current ROI
        roi_indices = np.where(adata.obs["roi"] == roi_name)[0]

        if len(roi_indices) == 0:
            logger.warning(f"ROI '{roi_name}' not found in adata.obs['roi']. Skipping rotation for this ROI.")

        # Get the spatial coordinates for the current ROI
        roi_spatial_coords = spatial_coords[roi_indices, :]

        if roi_spatial_coords.shape[1] != 2:
            raise ValueError(
                f"Spatial coordinates for ROI '{roi_name}' are not 2D. "
                f"Expected shape (n_cells, 2), got {roi_spatial_coords.shape}"
            )

        # Calculate the center of the ROI
        center_x = np.mean(roi_spatial_coords[:, 0])
        center_y = np.mean(roi_spatial_coords[:, 1])
        center = np.array([center_x, center_y])

        # Translate ROI to origin
        translated_coords = roi_spatial_coords - center
        rotation_matrix = create_rotation_matrix(-angle_degrees)
        rotated_coords_at_origin = (rotation_matrix @ translated_coords.T).T

        # Translate ROI back to its original position
        rotated_coords = rotated_coords_at_origin + center

        spatial_coords[roi_indices, :] = rotated_coords

    adata.obsm["spatial_rot"] = spatial_coords
    return adata
