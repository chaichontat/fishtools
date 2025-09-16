"""Scanpy-based plotting helpers for AnnData objects.

These wrappers centralize common figure creation patterns used in analysis
notebooks so that plots remain consistent across CLI and interactive contexts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import anndata as ad

__all__ = [
    "plot_embedding",
    "plot_genes",
    "plot_leiden_genes",
    "plot_ranked_genes",
]


def plot_embedding(
    adata: ad.AnnData,
    *,
    color: str | Sequence[str],
    basis: str = "spatial",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = 150,
    show: bool = False,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Create an embedding plot via :func:`scanpy.pl.embedding` with handles."""

    plotting_kwargs = dict(basis=basis, color=color, return_fig=True, show=show)
    plotting_kwargs.update(kwargs)
    fig = cast(Figure, sc.pl.embedding(adata, **plotting_kwargs))

    if figsize is not None:
        fig.set_size_inches(*figsize)
    if dpi is not None:
        fig.set_dpi(dpi)

    axes = [axis for axis in fig.axes if axis.get_label() != "<colorbar>"]
    for axis in axes:
        axis.set_aspect("equal")

    if not show:
        plt.close(fig)

    return fig, axes


def plot_genes(
    adata: ad.AnnData,
    genes: str | Sequence[str],
    basis: str = "spatial",
    n_cols: int = 3,
    perc: tuple[float, float] = (1, 99.99),
    cmap: str = "Blues",
) -> Figure:
    """Plot gene expression panels on a spatial embedding."""

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
    plt.close(fig)
    return fig


def plot_leiden_genes(adata: ad.AnnData) -> None:
    """Print marker genes for each Leiden cluster and render spatial plots."""

    for cluster in adata.obs["leiden"].cat.categories:
        print(f"Cluster {cluster}")
        genes = sc.get.rank_genes_groups_df(adata, group=cluster).head(3)["names"].tolist()
        fig = plot_genes(adata, genes, basis="spatial", n_cols=3, perc=(1, 99.5), cmap="Blues")
        fig.show()


def plot_ranked_genes(
    adata: ad.AnnData,
    n_genes: int = 15,
    fontsize: int = 14,
) -> None:
    """Plot ranked differential expression tables with rotated labels."""

    with plt.rc_context({"axes.grid": True}):
        sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False, fontsize=fontsize, show=False)
        for ax in plt.gcf().axes:
            for text in ax.texts:
                text.set_rotation(40)
        plt.show()
