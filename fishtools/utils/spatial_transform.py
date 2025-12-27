from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fishtools.utils.utils import create_rotation_matrix

if TYPE_CHECKING:  # pragma: no cover - for typing only
    import anndata as ad


def rotate_points(
    points: NDArray[np.floating],
    angle_degrees: float,
    *,
    center: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Rotate Nx2 points by angle (degrees) around center (default: centroid)."""
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected points shape (n,2+) for rotation, got {points.shape}.")

    coords = points[:, :2].astype(np.float64, copy=True)
    if center is None:
        pivot = coords.mean(axis=0)
    else:
        pivot = np.asarray(center, dtype=np.float64)
        if pivot.shape != (2,):
            raise ValueError("Rotation center must be a length-2 coordinate.")

    rotation = create_rotation_matrix(angle_degrees)
    rotated = (rotation @ (coords - pivot).T).T + pivot
    if points.shape[1] == 2:
        return rotated
    out = points.astype(np.float64, copy=True)
    out[:, :2] = rotated
    return out


def translate_points(points: NDArray[np.floating], shift: tuple[float, float]) -> NDArray[np.float64]:
    """Translate Nx2 points by (dx, dy)."""
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected points shape (n,2+) for translation, got {points.shape}.")
    if len(shift) != 2:
        raise ValueError(f"Translation shift must be a tuple of length 2, got {shift}.")
    dx, dy = shift
    coords = points.astype(np.float64, copy=True)
    coords[:, 0] += float(dx)
    coords[:, 1] += float(dy)
    return coords


def rotate_rois_in_adata(adata: "ad.AnnData", roi_rotation_angles: dict[str, float]) -> "ad.AnnData":
    """Rotate spatial coordinates for specified ROIs around their respective centers."""
    spatial_coords = adata.obsm["spatial"].copy()

    for roi_name, angle_degrees in roi_rotation_angles.items():
        roi_indices = np.where(adata.obs["roi"] == roi_name)[0]

        if len(roi_indices) == 0:
            logger.warning(f"ROI '{roi_name}' not found in adata.obs['roi']. Skipping rotation for this ROI.")
            continue

        roi_spatial_coords = spatial_coords[roi_indices, :]
        if roi_spatial_coords.shape[1] != 2:
            raise ValueError(
                f"Spatial coordinates for ROI '{roi_name}' are not 2D. "
                f"Expected shape (n_cells, 2), got {roi_spatial_coords.shape}"
            )

        center = roi_spatial_coords.mean(axis=0)
        rotated_coords = rotate_points(roi_spatial_coords, -angle_degrees, center=(center[0], center[1]))
        spatial_coords[roi_indices, :] = rotated_coords

    adata.obsm["spatial_rot"] = spatial_coords
    return adata


def translate_rois_in_adata(
    adata: "ad.AnnData", roi_translations: dict[str, tuple[float, float]]
) -> "ad.AnnData":
    """Translate ROI spatial coordinates in-place by the provided pixel offsets."""
    spatial_coords = adata.obsm.get("spatial_rot", adata.obsm["spatial"]).copy()

    for roi_name, shift in roi_translations.items():
        roi_indices = np.where(adata.obs["roi"] == roi_name)[0]

        if len(roi_indices) == 0:
            logger.warning(
                f"ROI '{roi_name}' not found in adata.obs['roi']. Skipping translation for this ROI."
            )
            continue

        translated = translate_points(spatial_coords[roi_indices, :], shift)
        spatial_coords[roi_indices, :] = translated

    adata.obsm["spatial_trans"] = spatial_coords
    return adata
