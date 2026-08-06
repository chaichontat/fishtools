"""Spatial layout helpers for concatenated AnnData objects.

These utilities are used in downstream concat/analysis workflows to put multiple
datasets/ROIs into a shared coordinate system for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from loguru import logger

from ccf.landmark import LandmarkRegistrationOutputs
from fishtools.io.workspace import Workspace
from fishtools.utils.utils import create_rotation_matrix

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


__all__ = ["auto_translate_groups"]


class _RowColMapColumns(TypedDict):
    row: list[int]
    col: list[int]
    dataset: list[str]
    roi: list[str]
    x0: list[float]
    y0: list[float]
    x1: list[float]
    y1: list[float]


def auto_translate_groups(
    adata: "ad.AnnData",
    padding: float = 500.0,
    *,
    debug: bool = True,
) -> "ad.AnnData":
    """Translate to layout: datasets stack vertically, ROIs stack horizontally within each.

    Additionally applies a per-dataset pose correction using the CCF landmark outputs
    (rotation from ``p1_similarity.tfm`` and optional x-flip from ``p1_landmarks.json``)
    under ``analysis/output/ccf-transforms/{roi}/`` when available.
    """

    coords = adata.obsm["spatial"].astype(np.float64, copy=True)
    datasets = sorted(adata.obs["dataset"].unique())

    def _try_resolve_workspace_root(dataset_str: str) -> Workspace | None:
        candidates = (Path("/working") / dataset_str, Path.home() / "nvme" / dataset_str)
        for candidate in candidates:
            if candidate.exists():
                return Workspace(candidate)
        return None

    def _read_similarity2d_angle_rad(tfm_path: Path) -> float:
        if not tfm_path.exists():
            raise FileNotFoundError(f"Missing similarity transform: {tfm_path}")
        for line in tfm_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("Parameters:"):
                continue
            parts = line.split(":", 1)[1].strip().split()
            if len(parts) < 2:
                raise ValueError(f"Invalid Similarity2DTransform parameters line in {tfm_path}: {line!r}")
            return float(parts[1])
        raise ValueError(f"Missing 'Parameters:' line in Similarity2DTransform file {tfm_path}")

    def _apply_ccf_pose_for_roi(
        ws: Workspace,
        dataset_str: str,
        roi_str: str,
        mask: object,
    ) -> None:
        mask_np = mask.to_numpy(dtype=bool, copy=False)  # pandas.Series[bool] in practice
        if not bool(mask_np.any()):
            return

        out = LandmarkRegistrationOutputs(ws.ccf_transforms(roi_str))
        if not out.p1_similarity_tfm.exists():
            if debug:
                logger.info(f"[auto_translate_groups] dataset={dataset_str!r} roi={roi_str!r} missing_tfm")
            return

        landmarks = out.try_read_p1_landmarks()
        prior_rotation_deg = 0 if landmarks is None else int(landmarks.prior_rotation_deg)
        prior_flip_x = False if landmarks is None else bool(landmarks.prior_flip_x)

        theta_rad = _read_similarity2d_angle_rad(out.p1_similarity_tfm)
        theta_deg = float(np.rad2deg(theta_rad))
        net_rotation_deg = float(prior_rotation_deg - theta_deg)

        roi_coords_xy = coords[mask_np, :2].astype(np.float64, copy=False)
        roi_center_xy = roi_coords_xy.mean(axis=0)

        if debug:
            logger.info(
                "[auto_translate_groups] "
                f"dataset={dataset_str!r} roi={roi_str!r} tfm={out.p1_similarity_tfm} "
                f"prior_rotation_deg={prior_rotation_deg} prior_flip_x={prior_flip_x} "
                f"tfm_theta_deg={theta_deg:.4f} net_rotation_deg={net_rotation_deg:.4f} "
                f"center=({float(roi_center_xy[0]):.3f},{float(roi_center_xy[1]):.3f}) "
                f"pre_pose x=[{float(roi_coords_xy[:,0].min()):.3f},{float(roi_coords_xy[:,0].max()):.3f}] "
                f"y=[{float(roi_coords_xy[:,1].min()):.3f},{float(roi_coords_xy[:,1].max()):.3f}]"
            )

        if prior_flip_x:
            # Flip around ROI centroid (not around x=0), matching image-style x-flip semantics.
            coords[mask_np, 0] = (2.0 * roi_center_xy[0]) - coords[mask_np, 0]

        if net_rotation_deg != 0:
            rot = create_rotation_matrix(-net_rotation_deg)
            coords_xy = coords[mask_np, :2].astype(np.float64, copy=False)
            coords[mask_np, :2] = ((coords_xy - roi_center_xy) @ rot.T) + roi_center_xy

        if debug:
            c1 = coords[mask_np, :2]
            logger.info(
                "[auto_translate_groups] "
                f"dataset={dataset_str!r} roi={roi_str!r} post_pose "
                f"x=[{float(c1[:,0].min()):.3f},{float(c1[:,0].max()):.3f}] "
                f"y=[{float(c1[:,1].min()):.3f},{float(c1[:,1].max()):.3f}]"
            )

    row_col_map: _RowColMapColumns = {
        "row": [],
        "col": [],
        "dataset": [],
        "roi": [],
        "x0": [],
        "y0": [],
        "x1": [],
        "y1": [],
    }

    curr_y = 0.0
    for row_idx, ds in enumerate(datasets):
        ds_str = str(ds)
        ds_mask = adata.obs["dataset"].astype(str) == ds_str
        if debug:
            logger.info(f"[auto_translate_groups] dataset={ds_str!r} n={int(ds_mask.sum())}")

        ws = _try_resolve_workspace_root(ds_str)
        if ws is None:
            logger.warning(f"Skipping CCF pose for dataset={ds_str!r}: unable to resolve workspace root.")
        elif debug:
            logger.info(f"[auto_translate_groups] dataset={ds_str!r} workspace={ws.path}")

        rois_in_ds = sorted(adata.obs.loc[ds_mask, "roi"].unique())
        row_height = 0.0
        curr_x = 0.0

        for col_idx, roi in enumerate(rois_in_ds):
            mask = ds_mask & (adata.obs["roi"] == roi)
            if ws is not None:
                _apply_ccf_pose_for_roi(ws, ds_str, str(roi), mask)
            mask_np = mask.to_numpy(dtype=bool, copy=False)
            c = coords[mask_np]
            xmin, xmax = c[:, 0].min(), c[:, 0].max()
            ymin, ymax = c[:, 1].min(), c[:, 1].max()
            width = float(xmax - xmin)
            height = float(ymax - ymin)

            coords[mask_np, 0] += curr_x - xmin
            coords[mask_np, 1] += curr_y - ymin
            if debug:
                logger.info(
                    "[auto_translate_groups] "
                    f"dataset={ds_str!r} roi={str(roi)!r} "
                    f"pre_bbox=({float(xmin):.1f},{float(ymin):.1f})-({float(xmax):.1f},{float(ymax):.1f}) "
                    f"size=({width:.1f},{height:.1f}) "
                    f"shift=({float(curr_x - xmin):.1f},{float(curr_y - ymin):.1f})"
                )

            row_col_map["row"].append(row_idx)
            row_col_map["col"].append(col_idx)
            row_col_map["dataset"].append(str(ds))
            row_col_map["roi"].append(str(roi))
            row_col_map["x0"].append(float(curr_x))
            row_col_map["y0"].append(float(curr_y))
            row_col_map["x1"].append(float(curr_x + width))
            row_col_map["y1"].append(float(curr_y + height))
            curr_x += width + padding
            row_height = max(row_height, height)

        curr_y += row_height + padding

    adata.obsm["spatial_trans"] = coords
    adata.uns["row_col_map"] = row_col_map
    return adata
