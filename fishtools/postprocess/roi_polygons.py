"""Helpers for annotating AnnData using ImageJ ROI polygon files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad

__all__ = ["RoiPolygon", "annotate_cells_with_roi", "load_roi_polygons"]


@dataclass(frozen=True)
class RoiPolygon:
    geometry: object
    name: str


def load_roi_polygons(roi_path: Path, *, scale: float = 8.0) -> list[RoiPolygon]:
    """Return scaled Shapely polygons (with ROI names) from an ImageJ ROI file.

    Parameters
    ----------
    roi_path:
        Path to the ImageJ ``RoiSet.zip`` archive or an individual ``.roi`` file.
    scale:
        Multiplicative factor to map ROI coordinates back into stitched pixel space.
        ImageJ thumbnails are commonly generated at 1/8 scale, so the default compensates
        by multiplying coordinates by 8.
    """

    if scale <= 0:
        raise ValueError("scale must be positive when loading ROI polygons.")

    import numpy as np
    from roifile import ImagejRoi, roiread
    from shapely.geometry import MultiPolygon, Polygon

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    suffix = roi_path.suffix.lower()
    if suffix not in {".zip", ".roi"}:
        raise ValueError(f"ROI path must be a .zip or .roi file, got {roi_path}")

    try:
        rois = roiread(roi_path)
    except Exception as exc:  # pragma: no cover - passthrough to caller with context
        raise ValueError(f"Failed to read ROI {roi_path}: {exc}") from exc

    if isinstance(rois, ImagejRoi):  # roiread can return a singleton ImagejRoi
        rois = [rois]

    polygons: list[RoiPolygon] = []
    offset = np.array([0.0, 0.0], dtype=np.float64)
    for idx, roi in enumerate(rois):
        coords = roi.integer_coordinates.astype(np.float64)
        offset[0] = roi.left
        offset[1] = roi.top
        abs_coords = (coords + offset) * scale
        if abs_coords.shape[0] < 3:
            continue
        try:
            poly = Polygon(abs_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if not isinstance(poly, (Polygon, MultiPolygon)):
                continue
        except Exception as exc:  # pragma: no cover - geometry construction failure
            logger.debug(f"Skipping ROI '{roi.name}' due to geometry error: {exc}")
            continue
        roi_name = str(roi.name) if roi.name else f"roi_{idx}"
        polygons.append(RoiPolygon(geometry=poly, name=roi_name))

    if not polygons:
        raise ValueError(f"No polygons extracted from ROI {roi_path}.")

    logger.info(f"Loaded {len(polygons)} ROI polygon(s) from {roi_path.name} with scale={scale}.")
    return polygons


def annotate_cells_with_roi(
    adata: "ad.AnnData",
    roi_path: Path,
    *,
    spatial_key: str = "spatial",
    column_name: str = "in_roi",
    scale: float = 8.0,
) -> None:
    """Annotate cells whose centroids fall inside polygons from an ImageJ ROI file.

    The ROI coordinates are scaled by ``scale`` (default 8×) to map thumbnail-derived
    outlines back into the stitched pixel reference frame. The resulting label (empty
    string if not inside any polygon) is stored in ``adata.obs[column_name]``.
    """

    import numpy as np
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            "annotate_cells_with_roi requires either adata.obsm['spatial'] or obs columns 'x'/'y'."
        )

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must be a (n_cells, 2+) array of x/y positions.")

    roi_polygons = load_roi_polygons(roi_path, scale=scale)
    geometries = [entry.geometry for entry in roi_polygons]
    tree = STRtree(geometries)

    labels = np.empty(coords.shape[0], dtype=object)
    labels[:] = ""
    for idx, (x_coord, y_coord) in enumerate(coords[:, :2]):
        point = Point(float(x_coord), float(y_coord))
        for poly_index in tree.query(point):
            geometry = geometries[poly_index]
            if geometry.covers(point):
                labels[idx] = roi_polygons[poly_index].name
                break

    adata.obs[column_name] = labels
    logger.info(
        f"Annotated {(labels != '').sum()} of {len(labels)} cells as inside ROI polygons from {roi_path.name}."
    )
