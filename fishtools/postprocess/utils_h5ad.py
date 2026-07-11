"""AnnData preprocessing utilities shared across postprocess workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import chain
from pathlib import Path
from typing import Literal
import warnings

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
    "coerce_float32_for_concat",
    "prefix_dataset_to_obs_names",
    "write_obs_parquet",
    "read_obs_parquet",
    "write_obsm_h5ad",
    "read_obsm_h5ad",
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


def normalize_pearson(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    *,
    batch_key: str | None = None,
    theta: float = 100,
    clip: float | None = None,
) -> tuple[ad.AnnData, Callable[[], None]]:
    """Select HVGs via Pearson residuals, then normalize Pearson residuals.

    When ``batch_key`` is provided, Pearson residuals are computed separately
    for each batch (similar to per-dataset SCTransform-style normalization).
    """

    import scanpy as sc

    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"Missing batch column in `adata.obs`: {batch_key!r}")

    # sc.experimental.pp.highly_variable_genes(
    #     adata,
    #     flavor="pearson_residuals",
    #     n_top_genes=n_top_genes,
    #     batch_key=batch_key,
    #     theta=theta,
    #     clip=clip,
    # )

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

    # adata = adata[:, adata.var["highly_variable"]].copy()

    adata.layers["raw"] = adata.X.copy()
    adata.layers["sqrt_norm"] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])

    if batch_key is None:
        sc.experimental.pp.normalize_pearson_residuals(adata, theta=theta, clip=clip)
        return adata, plot

    from scipy import sparse

    residuals = np.empty((adata.n_obs, adata.n_vars), dtype=np.float32)
    batch = adata.obs[batch_key]
    for value in batch.unique():
        mask = (batch == value).to_numpy(dtype=bool, copy=False)
        if not bool(mask.any()):
            continue
        sub = adata[mask].copy()
        sc.experimental.pp.normalize_pearson_residuals(sub, theta=theta, clip=clip)
        x = sub.X
        if sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        residuals[mask] = x

    adata.X = residuals
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


def coerce_float32_for_concat(adata: ad.AnnData) -> ad.AnnData:
    """Force float-like payloads to float32 to avoid concat precision mismatches."""

    from scipy import sparse

    def _cast(value: object):
        if sparse.issparse(value):
            if np.issubdtype(value.dtype, np.floating) and value.dtype != np.float32:
                return value.astype(np.float32)
            return value
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating) and value.dtype != np.float32:
            return value.astype(np.float32, copy=False)
        dtype = getattr(value, "dtype", None)
        if dtype is not None and np.issubdtype(dtype, np.floating) and dtype != np.float32 and hasattr(value, "astype"):
            return value.astype(np.float32)
        return value

    adata.X = _cast(adata.X)
    for key in list(adata.layers.keys()):
        adata.layers[key] = _cast(adata.layers[key])
    for key in list(adata.obsm.keys()):
        adata.obsm[key] = _cast(adata.obsm[key])
    for key in list(adata.varm.keys()):
        adata.varm[key] = _cast(adata.varm[key])
    for key in list(adata.obsp.keys()):
        adata.obsp[key] = _cast(adata.obsp[key])

    float_obs_cols = adata.obs.select_dtypes(include=["float"]).columns
    if len(float_obs_cols):
        adata.obs[float_obs_cols] = adata.obs[float_obs_cols].astype(np.float32)
    float_var_cols = adata.var.select_dtypes(include=["float"]).columns
    if len(float_var_cols):
        adata.var[float_var_cols] = adata.var[float_var_cols].astype(np.float32)

    return adata


def prefix_dataset_to_obs_names(
    adata: ad.AnnData,
    *,
    dataset_col: str = "dataset",
    sep: str = ":",
) -> ad.AnnData:
    """Prefix ``obs_names`` with ``obs[dataset_col]`` for concat-safe uniqueness."""

    obs_names = adata.obs_names.astype(str)
    prefix = adata.obs[dataset_col].astype(str)
    first_prefix = f"{prefix.iloc[0]}{sep}" if len(obs_names) else ""
    if not len(obs_names) or not obs_names[0].startswith(first_prefix):
        adata.obs_names = pd.Index(prefix.str.cat(obs_names, sep=str(sep)))
    adata.obs_names.rename("index", inplace=True)
    adata.obs.index.rename("index", inplace=True)
    return adata


def write_obs_parquet(adata: ad.AnnData, path: str | Path) -> None:
    """Write ``adata.obs`` to parquet while preserving ``obs_names`` as the index."""

    adata.obs.copy().to_parquet(path, index=True)


def _validate_one_to_one_obs_join(source_index: pd.Index, target_index: pd.Index, *, source_name: str) -> None:
    """Validate that an obs-index join is one-to-one before alignment."""

    if not source_index.is_unique or not target_index.is_unique:
        warnings.warn(
            "Joining obs metadata is not one-to-one because one or both indices contain duplicates.",
            stacklevel=2,
        )
    if not source_index.is_unique:
        raise ValueError(f"Expected {source_name} index to be unique.")
    if not target_index.is_unique:
        raise ValueError("Expected adata.obs_names to be unique.")


def read_obs_parquet(path: str | Path, *, adata: ad.AnnData | None = None) -> pd.DataFrame | ad.AnnData:
    """Read an ``obs`` parquet, optionally overwriting columns back into ``adata``."""

    obs = pd.read_parquet(path)
    if adata is None:
        return obs

    adata_index = pd.Index(adata.obs_names)
    _validate_one_to_one_obs_join(obs.index, adata_index, source_name="parquet obs")
    missing = adata_index.difference(obs.index)
    if len(missing):
        raise ValueError(f"Obs parquet is missing adata.obs_names entries: {missing.tolist()}.")

    obs = obs.reindex(adata_index)
    for column in obs.columns:
        adata.obs[column] = obs[column]
    return adata


def write_obsm_h5ad(adata: ad.AnnData, path: str | Path) -> None:
    """Write an obsm-only AnnData bundle that preserves obs_names."""

    obsm_only = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names.copy()))
    for key in adata.obsm:
        obsm_only.obsm[key] = adata.obsm[key].copy()
    obsm_only.write_h5ad(path)


def read_obsm_h5ad(path: str | Path, *, adata: ad.AnnData | None = None) -> ad.AnnData:
    """Read an AnnData bundle created by ``write_obsm_h5ad`` and optionally join it back into ``adata``."""

    obsm_adata = ad.read_h5ad(path)
    if obsm_adata.n_vars != 0:
        raise ValueError(f"Expected an obsm-only h5ad with zero vars, got n_vars={obsm_adata.n_vars}.")
    if adata is None:
        return obsm_adata

    adata_index = pd.Index(adata.obs_names)
    _validate_one_to_one_obs_join(obsm_adata.obs_names, adata_index, source_name="obsm h5ad obs")
    missing = adata_index.difference(obsm_adata.obs_names)
    if len(missing):
        raise ValueError(f"Obsm h5ad is missing adata.obs_names entries: {missing.tolist()}.")

    aligned = obsm_adata[adata.obs_names].copy()
    for key in aligned.obsm:
        adata.obsm[key] = aligned.obsm[key].copy()
    return adata


def filter_leiden(adata: ad.AnnData, keep: Sequence[int | str]) -> ad.AnnData:
    """Subset AnnData to specified Leiden clusters."""

    return adata[adata.obs["leiden"].isin(keep)]


def run_tricycle(
    adata: ad.AnnData,
    trc: pd.DataFrame,
    *,
    batch_key: str | None = None,
    layer: str | None = None,
    center_on: Sequence[str] | None = None,
) -> ad.AnnData:
    """Project AnnData onto tricycle cell-cycle embeddings.

    If ``batch_key`` is provided, gene-wise mean centering is performed within
    each batch in ``adata.obs[batch_key]`` before projection. When
    ``center_on`` is provided, only those ``adata.obs_names`` are used to
    estimate the centering mean, while the projection is still computed for all
    cells.
    """

    import scipy.sparse as sp

    shared = sorted(set(trc["symbol"]) & set(adata.var_names))
    if not shared:
        raise ValueError("No shared genes between `trc['symbol']` and `adata.var_names`.")
    loadings = (
        trc[trc["symbol"].isin(shared)]
        .set_index("symbol")
        .reindex(shared)
        .reset_index()[["pc1.rot", "pc2.rot"]]
    )
    if layer is not None:
        x = adata[:, shared].layers[layer]
    else:
        x = adata[:, shared].X

    if sp.issparse(x):
        x = x.toarray()
    else:
        x = np.asarray(x)

    x = x.astype(np.float32, copy=False)
    center_mask = np.ones(adata.n_obs, dtype=bool)
    if center_on is not None:
        center_names = pd.Index(center_on)
        missing = center_names.difference(adata.obs_names)
        if len(missing):
            raise KeyError(f"`center_on` contains unknown obs_names: {missing.tolist()}")
        center_mask = np.asarray(adata.obs_names.isin(center_names))
        if not np.any(center_mask):
            raise ValueError("`center_on` must contain at least one obs_name.")
    if batch_key is None:
        center = np.mean(x[center_mask], axis=0, keepdims=True)
        x_centered = x - center
    else:
        if batch_key not in adata.obs.columns:
            raise KeyError(f"obs.{batch_key} not found")
        batch = adata.obs[batch_key]
        if batch.isna().any():
            raise ValueError(f"obs.{batch_key} contains missing values")
        batch_values = batch.astype(str).to_numpy()
        x_centered = np.empty_like(x)
        for value in np.unique(batch_values):
            batch_mask = batch_values == value
            center_batch_mask = batch_mask & center_mask
            if not np.any(center_batch_mask):
                raise ValueError(f"`center_on` does not include any cells from batch {value!r}.")
            center = np.mean(x[center_batch_mask], axis=0, keepdims=True)
            x_centered[batch_mask] = x[batch_mask] - center
    pls = x_centered @ loadings.to_numpy(dtype=np.float32)

    adata.obsm["tricycle"] = pls
    adata.obsm["X_tricycle"] = pls
    adata.obs["tricycle"] = (np.arctan2(pls[:, 1], pls[:, 0]) + 2 * np.pi) % (2 * np.pi)
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
