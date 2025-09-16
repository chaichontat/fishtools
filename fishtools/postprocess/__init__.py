"""
fishtools.postprocess

Post-processing utilities for FISH analysis including multi-ROI concatenation,
single-cell data preparation, and analysis pipeline integration.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
from loguru import logger
from matplotlib.axes import Axes

from fishtools.utils.utils import create_rotation_matrix

from .io_concat import (
    ConcatDataError,
    ConcatDataSpec,
    arrange_rois,
    compute_weighted_centroids,
    join_ident_with_spots,
    load_ident_tables,
    load_intensity_tables,
    load_polygon_tables,
    load_spot_tables,
    merge_polygons_with_intensity,
)
from .plot_h5ad import (
    plot_embedding,
    plot_genes,
    plot_leiden_genes,
    plot_ranked_genes,
)
from .utils_h5ad import (
    cluster,
    filter_leiden,
    get_leiden_genes,
    leiden_umap,
    normalize_pearson,
    normalize_total,
    qc,
    run_spaco,
    run_tricycle,
    std_log1p,
)

__all__ = [
    "ConcatDataError",
    "ConcatDataSpec",
    "load_ident_tables",
    "load_intensity_tables",
    "load_spot_tables",
    "load_polygon_tables",
    "join_ident_with_spots",
    "merge_polygons_with_intensity",
    "arrange_rois",
    "compute_weighted_centroids",
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
    "plot_embedding",
    "plot_genes",
    "plot_leiden_genes",
    "plot_ranked_genes",
    "jitter",
    "rotate_rois_in_adata",
    "add_scale_bar",
]


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


def add_scale_bar(
    ax: Axes,
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
