"""
fishtools.postprocess

Post-processing utilities for FISH analysis including multi-ROI concatenation,
single-cell data preparation, and analysis pipeline integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from fishtools.utils.utils import create_rotation_matrix, make_lazy_getattr

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
    # plot_h5ad
    "plot_embedding": ("fishtools.postprocess.plot_h5ad", "plot_embedding"),
    "plot_genes": ("fishtools.postprocess.plot_h5ad", "plot_genes"),
    "plot_leiden_genes": ("fishtools.postprocess.plot_h5ad", "plot_leiden_genes"),
    "plot_ranked_genes": ("fishtools.postprocess.plot_h5ad", "plot_ranked_genes"),
    # utils_h5ad
    "cluster": ("fishtools.postprocess.utils_h5ad", "cluster"),
    "filter_leiden": ("fishtools.postprocess.utils_h5ad", "filter_leiden"),
    "get_leiden_genes": ("fishtools.postprocess.utils_h5ad", "get_leiden_genes"),
    "leiden_umap": ("fishtools.postprocess.utils_h5ad", "leiden_umap"),
    "normalize_pearson": ("fishtools.postprocess.utils_h5ad", "normalize_pearson"),
    "normalize_total": ("fishtools.postprocess.utils_h5ad", "normalize_total"),
    "qc": ("fishtools.postprocess.utils_h5ad", "qc"),
    "run_spaco": ("fishtools.postprocess.utils_h5ad", "run_spaco"),
    "run_tricycle": ("fishtools.postprocess.utils_h5ad", "run_tricycle"),
    "std_log1p": ("fishtools.postprocess.utils_h5ad", "std_log1p"),
}

__getattr__, __dir__, __all__ = make_lazy_getattr(
    globals(),
    _LAZY_ATTRS,
    extras=("jitter", "rotate_rois_in_adata", "translate_rois_in_adata", "add_scale_bar"),
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
    from fishtools.postprocess.utils_h5ad import (  # noqa: F401
        cluster as cluster,
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
        qc as qc,
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


def jitter(data, amount: float = 0.5, seed: int | None = None):
    import numpy as np  # local import to avoid heavy import at module load

    rand = np.random.default_rng(seed)
    return data + rand.normal(0, amount, size=data.shape[0])


def rotate_rois_in_adata(adata, roi_rotation_angles: dict[str, float]):
    """
    Rotates spatial coordinates in an AnnData object for specified ROIs around their respective centers.

    Args:
        adata: The AnnData object with 'roi' in obs and 'spatial' in obsm.
        roi_rotation_angles: A dictionary where keys are ROI names (matching those in adata.obs['roi'])
                             and values are the rotation angles in degrees.

    Returns:
        A new AnnData object with rotated spatial coordinates.
    """
    import numpy as np  # local import
    from loguru import logger  # local import to avoid module-level dep

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


def translate_rois_in_adata(adata: "ad.AnnData", roi_translations: dict[str, tuple[float, float]]):
    """Translate ROI spatial coordinates in-place by the provided pixel offsets."""
    import numpy as np  # local import
    from loguru import logger  # local import

    spatial_coords = adata.obsm.get("spatial_rot", adata.obsm["spatial"]).copy()

    for roi_name, shift in roi_translations.items():
        if len(shift) != 2:
            raise ValueError(f"Translation for ROI '{roi_name}' must be a tuple of length 2, got {shift}.")

        dx, dy = shift
        roi_indices = np.where(adata.obs["roi"] == roi_name)[0]

        if len(roi_indices) == 0:
            logger.warning(
                f"ROI '{roi_name}' not found in adata.obs['roi']. Skipping translation for this ROI."
            )
            continue

        spatial_coords[roi_indices, 0] += dx
        spatial_coords[roi_indices, 1] += dy

    adata.obsm["spatial_trans"] = spatial_coords
    return adata


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
