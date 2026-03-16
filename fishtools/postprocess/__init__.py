"""
fishtools.postprocess

Post-processing utilities for FISH analysis including multi-ROI concatenation,
single-cell data preparation, and analysis pipeline integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from fishtools.utils.utils import make_lazy_getattr

# Lazy re-exports to avoid importing heavy deps (anndata, matplotlib, etc.)
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # io_concat
    "ConcatDataError": ("fishtools.postprocess.io_concat", "ConcatDataError"),
    "ConcatDataSpec": ("fishtools.postprocess.io_concat", "ConcatDataSpec"),
    "arrange_rois": ("fishtools.postprocess.io_concat", "arrange_rois"),
    "compute_weighted_centroids": (
        "fishtools.postprocess.io_concat",
        "compute_weighted_centroids",
    ),
    "join_ident_with_spots": ("fishtools.postprocess.io_concat", "join_ident_with_spots"),
    "load_ident_tables": ("fishtools.postprocess.io_concat", "load_ident_tables"),
    "load_intensity_tables": ("fishtools.postprocess.io_concat", "load_intensity_tables"),
    "load_polygon_tables": ("fishtools.postprocess.io_concat", "load_polygon_tables"),
    "load_spot_tables": ("fishtools.postprocess.io_concat", "load_spot_tables"),
    "merge_polygons_with_intensity": (
        "fishtools.postprocess.io_concat",
        "merge_polygons_with_intensity",
    ),
    # roi_polygons
    "annotate_cells_with_roi": ("fishtools.postprocess.roi_polygons", "annotate_cells_with_roi"),
    "load_roi_polygons": ("fishtools.postprocess.roi_polygons", "load_roi_polygons"),
    # plot_h5ad
    "plot_embedding": ("fishtools.postprocess.plot_h5ad", "plot_embedding"),
    "plot_genes": ("fishtools.postprocess.plot_h5ad", "plot_genes"),
    "plot_leiden_genes": ("fishtools.postprocess.plot_h5ad", "plot_leiden_genes"),
    "plot_ranked_genes": ("fishtools.postprocess.plot_h5ad", "plot_ranked_genes"),
    # layer_boundaries
    "identify_layer_boundaries_knn": (
        "fishtools.postprocess.layer_boundaries",
        "identify_layer_boundaries_knn",
    ),
    # spatial_arrange
    "auto_translate_groups": ("fishtools.postprocess.spatial_arrange", "auto_translate_groups"),
    # utils_h5ad
    "cluster": ("fishtools.postprocess.utils_h5ad", "cluster"),
    "coerce_float32_for_concat": ("fishtools.postprocess.utils_h5ad", "coerce_float32_for_concat"),
    "filter_leiden": ("fishtools.postprocess.utils_h5ad", "filter_leiden"),
    "get_leiden_genes": ("fishtools.postprocess.utils_h5ad", "get_leiden_genes"),
    "leiden_umap": ("fishtools.postprocess.utils_h5ad", "leiden_umap"),
    "normalize_pearson": ("fishtools.postprocess.utils_h5ad", "normalize_pearson"),
    "normalize_total": ("fishtools.postprocess.utils_h5ad", "normalize_total"),
    "prefix_dataset_to_obs_names": ("fishtools.postprocess.utils_h5ad", "prefix_dataset_to_obs_names"),
    "qc": ("fishtools.postprocess.utils_h5ad", "qc"),
    "read_obs_parquet": ("fishtools.postprocess.utils_h5ad", "read_obs_parquet"),
    "read_obsm_h5ad": ("fishtools.postprocess.utils_h5ad", "read_obsm_h5ad"),
    "run_spaco": ("fishtools.postprocess.utils_h5ad", "run_spaco"),
    "run_tricycle": ("fishtools.postprocess.utils_h5ad", "run_tricycle"),
    "std_log1p": ("fishtools.postprocess.utils_h5ad", "std_log1p"),
    "write_obs_parquet": ("fishtools.postprocess.utils_h5ad", "write_obs_parquet"),
    "write_obsm_h5ad": ("fishtools.postprocess.utils_h5ad", "write_obsm_h5ad"),
    # spatial helpers
    "rotate_rois_in_adata": ("fishtools.utils.spatial_transform", "rotate_rois_in_adata"),
    "translate_rois_in_adata": ("fishtools.utils.spatial_transform", "translate_rois_in_adata"),
}

__getattr__, __dir__, __all__ = make_lazy_getattr(
    globals(),
    _LAZY_ATTRS,
    extras=("jitter", "add_scale_bar"),
)

if TYPE_CHECKING:  # pragma: no cover - for editors only
    from anndata import AnnData as AnnData  # noqa: F401
    from matplotlib.axes import Axes as Axes  # noqa: F401

    from fishtools.postprocess.io_concat import (  # noqa: F401
        ConcatDataError as ConcatDataError,
    )
    from fishtools.postprocess.io_concat import (
        ConcatDataSpec as ConcatDataSpec,
    )
    from fishtools.postprocess.io_concat import (
        arrange_rois as arrange_rois,
    )
    from fishtools.postprocess.io_concat import (
        compute_weighted_centroids as compute_weighted_centroids,
    )
    from fishtools.postprocess.io_concat import (
        join_ident_with_spots as join_ident_with_spots,
    )
    from fishtools.postprocess.io_concat import (
        load_ident_tables as load_ident_tables,
    )
    from fishtools.postprocess.io_concat import (
        load_intensity_tables as load_intensity_tables,
    )
    from fishtools.postprocess.io_concat import (
        load_polygon_tables as load_polygon_tables,
    )
    from fishtools.postprocess.io_concat import (
        load_spot_tables as load_spot_tables,
    )
    from fishtools.postprocess.io_concat import (
        merge_polygons_with_intensity as merge_polygons_with_intensity,
    )
    from fishtools.postprocess.roi_polygons import (  # noqa: F401
        annotate_cells_with_roi as annotate_cells_with_roi,
    )
    from fishtools.postprocess.roi_polygons import (  # noqa: F401
        load_roi_polygons as load_roi_polygons,
    )
    from fishtools.postprocess.spatial_arrange import (  # noqa: F401
        auto_translate_groups as auto_translate_groups,
    )
    from fishtools.postprocess.plot_h5ad import (  # noqa: F401
        plot_embedding as plot_embedding,
    )
    from fishtools.postprocess.plot_h5ad import (
        plot_genes as plot_genes,
    )
    from fishtools.postprocess.plot_h5ad import (
        plot_leiden_genes as plot_leiden_genes,
    )
    from fishtools.postprocess.plot_h5ad import (
        plot_ranked_genes as plot_ranked_genes,
    )
    from fishtools.postprocess.layer_boundaries import (  # noqa: F401
        identify_layer_boundaries_knn as identify_layer_boundaries_knn,
    )
    from fishtools.postprocess.utils_h5ad import (  # noqa: F401
        cluster as cluster,
    )
    from fishtools.postprocess.utils_h5ad import (
        coerce_float32_for_concat as coerce_float32_for_concat,
    )
    from fishtools.postprocess.utils_h5ad import (
        filter_leiden as filter_leiden,
    )
    from fishtools.postprocess.utils_h5ad import (
        get_leiden_genes as get_leiden_genes,
    )
    from fishtools.postprocess.utils_h5ad import (
        leiden_umap as leiden_umap,
    )
    from fishtools.postprocess.utils_h5ad import (
        normalize_pearson as normalize_pearson,
    )
    from fishtools.postprocess.utils_h5ad import (
        normalize_total as normalize_total,
    )
    from fishtools.postprocess.utils_h5ad import (
        prefix_dataset_to_obs_names as prefix_dataset_to_obs_names,
    )
    from fishtools.postprocess.utils_h5ad import (
        qc as qc,
    )
    from fishtools.postprocess.utils_h5ad import (
        read_obs_parquet as read_obs_parquet,
    )
    from fishtools.postprocess.utils_h5ad import (
        read_obsm_h5ad as read_obsm_h5ad,
    )
    from fishtools.postprocess.utils_h5ad import (
        run_spaco as run_spaco,
    )
    from fishtools.postprocess.utils_h5ad import (
        run_tricycle as run_tricycle,
    )
    from fishtools.postprocess.utils_h5ad import (
        std_log1p as std_log1p,
    )
    from fishtools.postprocess.utils_h5ad import (
        write_obs_parquet as write_obs_parquet,
    )
    from fishtools.postprocess.utils_h5ad import (
        write_obsm_h5ad as write_obsm_h5ad,
    )


def jitter(data, amount: float = 0.5, seed: int | None = None):
    import numpy as np  # local import to avoid heavy import at module load

    rand = np.random.default_rng(seed)
    return data + rand.normal(0, amount, size=data.shape[0])


def add_scale_bar(
    ax,
    pixel_size: float,
    label: str,
    *,
    color: str = "black",
    linewidth: float = 3.0,
    pad: float = 0.05,
) -> None:
    """Add a simple scale bar to the provided axes."""

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    x_start = x_limits[0] + pad * (x_limits[1] - x_limits[0])
    x_end = x_start + pixel_size
    y_pos = y_limits[0] + pad * (y_limits[1] - y_limits[0])

    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=linewidth)
    ax.text(
        (x_start + x_end) / 2,
        y_pos - pad * (y_limits[1] - y_limits[0]),
        label,
        ha="center",
        va="top",
        color=color,
    )
