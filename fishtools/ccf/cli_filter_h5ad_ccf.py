from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import anndata as ad
import ants
import matplotlib as mpl
import numpy as np
import pandas as pd
import rich_click as click
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

# Force a non-interactive backend to avoid GUI/event-loop hangs in headless runs
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from fishtools.ccf.landmark import LandmarkRegistrationOutputs, P1Landmarks
from fishtools.ccf.ndimage_geometry import fused_xy_to_rotated_crop_xy
from fishtools.ccf.ontology import CCFTermKind, mask_ccf_subtree
from fishtools.ccf.sitk_utils import UM_TO_MM, normalize_robust
from fishtools.io.workspace import Workspace
from fishtools.postprocess.roi_polygons import load_roi_polygons
from fishtools.utils.logging import setup_cli_logging

click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""


CombineMode = Literal["any", "all"]
MatchMode = Literal["subtree", "exact"]
CoordUnits = Literal["px", "um", "mm"]
SpatialOrder = Literal["xy", "yx"]
CoordSpace = Literal["crop", "full"]
InputSpace = Literal["crop", "full", "fused"]
T_AXIS_T_CMAP = "viridis"
T_AXIS_OVERLAP_TOL_PX = 1.0


class _SummaryPaths(TypedDict, total=False):
    fixed_nifti: str
    warped_after_nifti: str
    moving_mask_warped_final_nifti: str
    qc_zoom_masked_png: str


class _Summary(TypedDict, total=False):
    fwdtransforms: list[str] | str
    paths: _SummaryPaths


@dataclass(frozen=True, slots=True)
class TAxisOverlay:
    line_xy: np.ndarray
    line_t: np.ndarray
    tick_xy: np.ndarray
    tick_t: np.ndarray
    minor_tick_xy: np.ndarray
    minor_tick_t: np.ndarray
    tick_labels: list[str]


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload in {path}: expected object, got {type(payload).__name__}.")
    return cast(dict[str, object], payload)


def _as_list_of_str(value: object, *, key: str, path: Path) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"Invalid {key} in {path}: expected string or list[str], got {type(value).__name__}.")


def _resolve_paths(paths: list[str], *, base_dir: Path) -> list[str]:
    resolved: list[str] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = base_dir / pp
        resolved.append(str(pp))
    return resolved


def _ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T


def _ants_from_mask_yx(*, mask_yx: np.ndarray, spacing_um: float) -> ants.ANTsImage:
    arr = np.asarray(mask_yx, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape}.")
    sp_mm = float(spacing_um) * UM_TO_MM
    return ants.from_numpy(arr.T, origin=[0.0, 0.0], spacing=[sp_mm, sp_mm])


def _bbox_indices_from_mask(*, mask_xy: np.ndarray, pad_vox: int = 0) -> tuple[list[int], list[int]] | None:
    mask_xy = np.asarray(mask_xy, dtype=bool)
    if mask_xy.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_xy.shape}.")
    if not np.any(mask_xy):
        return None

    idx = np.argwhere(mask_xy)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    pad = int(pad_vox)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad + 1, np.asarray(mask_xy.shape, dtype=int))
    return lo.astype(int).tolist(), hi.astype(int).tolist()


def _compute_t_axis_overlay_for_fixed_view(
    *,
    fixed_view: ants.ANTsImage,
    atlas_name: str,
    atlas_plane: str,
    atlas_slice_idx: int,
    atlas_crop_bbox: tuple[int, int, int, int],
    atlas_voxel_um: float,
    coverage_mask_yx: np.ndarray | None = None,
    side_filter: Literal["both", "primary", "mirrored"] = "both",
    tick_step: float = 0.1,
) -> TAxisOverlay | None:
    expected_atlas_name = "kim_dev_mouse_e15-5_lsfm_20um"
    if atlas_name != expected_atlas_name:
        return None

    atlas_plane_t = str(atlas_plane).lower()
    if atlas_plane_t not in {"coronal", "sagittal"}:
        return None

    outdir = (
        Path(__file__).resolve().parents[2]
        / "ccf"
        / "out"
        / "refextract"
        / "midsurface_neocortex_mesocortex_allocortex_3d"
    )
    u_path = outdir / "halfway_u_3d_ds.npy"
    mask_fit_path = outdir / "cortex_mask_fit_3d_ds.npy"
    mask_clean_path = outdir / "cortex_mask_clean_3d_ds.npy"
    include_path = outdir / "midline_include_neo_meso_3d_ds.npy"
    mask_path = mask_fit_path if mask_fit_path.exists() else mask_clean_path

    missing: list[Path] = []
    if not u_path.exists():
        missing.append(u_path)
    if not mask_path.exists():
        missing.extend([mask_fit_path, mask_clean_path])
    if missing:
        raise FileNotFoundError(f"Missing midsurface artifacts for t-axis overlay: {[str(p) for p in missing]}")

    u_3d = np.load(u_path).astype(np.float32, copy=False)
    cortex_3d = np.load(mask_path).astype(bool)
    if u_3d.shape != cortex_3d.shape:
        raise ValueError(f"u/cortex shape mismatch for t-axis overlay: u={u_3d.shape} cortex={cortex_3d.shape}")

    include_3d: np.ndarray | None = None
    if include_path.exists():
        include_3d = np.load(include_path).astype(bool)
        if include_3d.shape != u_3d.shape:
            raise ValueError(f"midline_include shape mismatch: include={include_3d.shape} u={u_3d.shape}")

    if atlas_plane_t == "coronal":
        if not (0 <= int(atlas_slice_idx) < int(u_3d.shape[0])):
            raise ValueError(f"atlas_slice_idx {atlas_slice_idx} out of bounds for coronal midsurface shape {u_3d.shape}.")
        contour_mask = cortex_3d[int(atlas_slice_idx), :, :]
        if include_3d is not None:
            contour_mask = contour_mask & include_3d[int(atlas_slice_idx), :, :]
        u2 = np.full(u_3d[int(atlas_slice_idx), :, :].shape, np.nan, dtype=np.float32)
        u2[contour_mask] = u_3d[int(atlas_slice_idx), :, :][contour_mask]
    else:
        if not (0 <= int(atlas_slice_idx) < int(u_3d.shape[2])):
            raise ValueError(
                f"atlas_slice_idx {atlas_slice_idx} out of bounds for sagittal midsurface shape {u_3d.shape}."
            )
        contour_mask = cortex_3d[:, :, int(atlas_slice_idx)]
        if include_3d is not None:
            contour_mask = contour_mask & include_3d[:, :, int(atlas_slice_idx)]
        u2 = np.full(u_3d[:, :, int(atlas_slice_idx)].shape, np.nan, dtype=np.float32)
        u2[contour_mask] = u_3d[:, :, int(atlas_slice_idx)][contour_mask]

    if not np.isfinite(u2).any():
        raise ValueError(f"No finite midsurface contour values for atlas_slice_idx={atlas_slice_idx} ({atlas_plane_t}).")

    fig_tmp, ax_tmp = plt.subplots()
    cont = ax_tmp.contour(u2, levels=[0.5], linewidths=0.0, alpha=0.0)
    segs = [np.asarray(seg, dtype=np.float64) for seg in cont.allsegs[0] if np.asarray(seg).shape[0] >= 2]
    plt.close(fig_tmp)
    if not segs:
        raise ValueError(f"No u=0.5 contour segments extracted for atlas_slice_idx={atlas_slice_idx} ({atlas_plane_t}).")
    if side_filter not in {"both", "primary", "mirrored"}:
        raise ValueError(f"Invalid side_filter={side_filter!r}.")
    fixed_shape_xy = tuple(int(v) for v in fixed_view.shape)
    if len(fixed_shape_xy) != 2:
        raise ValueError(f"Expected 2D fixed_view shape, got {fixed_shape_xy}.")
    fixed_width_px = int(fixed_shape_xy[0])

    primary_candidates: list[TAxisOverlay] = []
    mirrored_candidates: list[TAxisOverlay] = []
    for seg in segs:
        if atlas_plane_t == "coronal":
            row = seg[:, 1]
            col = seg[:, 0]
        else:
            row = seg[:, 0]
            col = seg[:, 1]

        primary = _build_t_axis_overlay_from_row_col(
            fixed_view=fixed_view,
            row=row,
            col=col,
            atlas_crop_bbox=atlas_crop_bbox,
            atlas_voxel_um=atlas_voxel_um,
            tick_step=tick_step,
        )
        if primary is not None:
            primary_candidates.append(primary)

        if atlas_plane_t == "coronal":
            mirrored = _mirror_t_axis_overlay_in_fixed_space(
                t_axis_overlay=primary,
                width_px=fixed_width_px,
            )
            if mirrored is not None:
                mirrored_candidates.append(mirrored)

    primary = _pick_best_t_axis_candidate(candidates=primary_candidates, coverage_mask_yx=coverage_mask_yx)
    if atlas_plane_t != "coronal":
        return primary
    if side_filter == "primary":
        return primary

    mirrored = _pick_best_t_axis_candidate(candidates=mirrored_candidates, coverage_mask_yx=coverage_mask_yx)
    if side_filter == "mirrored":
        return mirrored
    if primary is None:
        return mirrored
    if mirrored is None or coverage_mask_yx is None:
        return primary

    return _select_t_axis_overlay_by_coverage(
        primary=primary,
        mirrored=mirrored,
        coverage_mask_yx=coverage_mask_yx,
    )


def _pick_best_t_axis_candidate(
    *,
    candidates: list[TAxisOverlay],
    coverage_mask_yx: np.ndarray | None,
) -> TAxisOverlay | None:
    if not candidates:
        return None
    if coverage_mask_yx is None:
        return max(candidates, key=lambda ov: int(np.asarray(ov.line_xy).shape[0]))

    best: TAxisOverlay | None = None
    best_key: tuple[int, float, int] | None = None
    for ov in candidates:
        score, median_dist = _t_axis_line_coverage_metrics(t_axis_overlay=ov, mask_yx=coverage_mask_yx)
        key = (int(score), float(-median_dist), int(np.asarray(ov.line_xy).shape[0]))
        if best_key is None or key > best_key:
            best = ov
            best_key = key
    return best


def _build_t_axis_overlay_from_row_col(
    *,
    fixed_view: ants.ANTsImage,
    row: np.ndarray,
    col: np.ndarray,
    atlas_crop_bbox: tuple[int, int, int, int],
    atlas_voxel_um: float,
    tick_step: float,
) -> TAxisOverlay | None:
    ar0, _, ac0, _ = (
        int(atlas_crop_bbox[0]),
        int(atlas_crop_bbox[1]),
        int(atlas_crop_bbox[2]),
        int(atlas_crop_bbox[3]),
    )
    spacing_mm = float(atlas_voxel_um) * UM_TO_MM
    origin_x_mm, origin_y_mm = (float(fixed_view.origin[0]), float(fixed_view.origin[1]))
    sp_x_mm, sp_y_mm = (float(fixed_view.spacing[0]), float(fixed_view.spacing[1]))

    row_c = np.asarray(row, dtype=np.float64) - float(ar0)
    col_c = np.asarray(col, dtype=np.float64) - float(ac0)
    x_mm = col_c * spacing_mm
    y_mm = row_c * spacing_mm
    x_idx = (x_mm - origin_x_mm) / sp_x_mm
    y_idx = (y_mm - origin_y_mm) / sp_y_mm

    line_xy = np.column_stack([x_idx, y_idx]).astype(np.float64, copy=False)
    valid = np.isfinite(line_xy).all(axis=1)
    line_xy = line_xy[valid]
    if line_xy.shape[0] < 2:
        return None

    seg_len = np.linalg.norm(np.diff(line_xy, axis=0), axis=1)
    seg_len = np.where(np.isfinite(seg_len), np.maximum(seg_len, 0.0), 0.0)
    cum = np.concatenate(([0.0], np.cumsum(seg_len, dtype=np.float64)))
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0.0:
        line_t = np.linspace(0.0, 1.0, line_xy.shape[0], dtype=np.float64)
    else:
        line_t = cum / total

    ticks = np.arange(0.0, 1.0 + 1.0e-9, float(tick_step), dtype=np.float64)
    tick_idx = np.argmin(np.abs(line_t[None, :] - ticks[:, None]), axis=1)
    tick_xy = line_xy[tick_idx]
    tick_t = line_t[tick_idx]
    tick_labels = [f"t={float(t):.1f}" for t in ticks.tolist()]

    minor_step = float(tick_step) * 0.5
    minor_ticks = np.arange(0.0, 1.0 + 1.0e-9, minor_step, dtype=np.float64)
    is_major = np.isclose(minor_ticks[:, None], ticks[None, :], atol=1.0e-9, rtol=0.0).any(axis=1)
    minor_ticks = minor_ticks[~is_major]
    minor_tick_idx = np.argmin(np.abs(line_t[None, :] - minor_ticks[:, None]), axis=1)
    minor_tick_xy = line_xy[minor_tick_idx]
    minor_tick_t = line_t[minor_tick_idx]

    return TAxisOverlay(
        line_xy=line_xy,
        line_t=line_t,
        tick_xy=tick_xy,
        tick_t=tick_t,
        minor_tick_xy=minor_tick_xy,
        minor_tick_t=minor_tick_t,
        tick_labels=tick_labels,
    )


def _mirror_t_axis_overlay_in_fixed_space(
    *,
    t_axis_overlay: TAxisOverlay | None,
    width_px: int,
) -> TAxisOverlay | None:
    if t_axis_overlay is None:
        return None
    if int(width_px) <= 0:
        raise ValueError(f"Invalid width_px={width_px}.")
    width = float(width_px) - 1.0

    def mirror_xy(xy_in: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_in, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected XY array with shape (N,2), got {xy.shape}.")
        out = xy.copy()
        if out.size:
            out[:, 0] = width - out[:, 0]
        return out

    return TAxisOverlay(
        line_xy=mirror_xy(np.asarray(t_axis_overlay.line_xy, dtype=np.float64)),
        line_t=np.asarray(t_axis_overlay.line_t, dtype=np.float64).copy(),
        tick_xy=mirror_xy(np.asarray(t_axis_overlay.tick_xy, dtype=np.float64)),
        tick_t=np.asarray(t_axis_overlay.tick_t, dtype=np.float64).copy(),
        minor_tick_xy=mirror_xy(np.asarray(t_axis_overlay.minor_tick_xy, dtype=np.float64)),
        minor_tick_t=np.asarray(t_axis_overlay.minor_tick_t, dtype=np.float64).copy(),
        tick_labels=[str(v) for v in t_axis_overlay.tick_labels],
    )


def _t_axis_line_t_and_distances_to_mask(
    *,
    t_axis_overlay: TAxisOverlay,
    mask_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D coverage mask, got shape={mask.shape}.")
    if not np.any(mask):
        return None

    line_xy = np.asarray(t_axis_overlay.line_xy, dtype=np.float64)
    line_t = np.asarray(t_axis_overlay.line_t, dtype=np.float64)
    if line_xy.ndim != 2 or line_xy.shape[1] != 2 or line_t.ndim != 1 or line_t.shape[0] != line_xy.shape[0]:
        return None

    valid = np.isfinite(line_xy).all(axis=1)
    valid &= np.isfinite(line_t)
    if not np.any(valid):
        return None

    line_xy = line_xy[valid]
    line_t = line_t[valid]

    mask_pts_yx = np.argwhere(mask)
    if mask_pts_yx.shape[0] == 0:
        return None
    line_pts_yx = np.column_stack([line_xy[:, 1], line_xy[:, 0]])
    tree = cKDTree(mask_pts_yx.astype(np.float64, copy=False))
    distances, _ = tree.query(line_pts_yx, k=1)
    return (line_t, np.asarray(distances, dtype=np.float64))


def _t_axis_line_coverage_metrics(*, t_axis_overlay: TAxisOverlay, mask_yx: np.ndarray) -> tuple[int, float]:
    pair = _t_axis_line_t_and_distances_to_mask(t_axis_overlay=t_axis_overlay, mask_yx=mask_yx)
    if pair is None:
        return (0, float("inf"))
    _, distances = pair
    if distances.size == 0:
        return (0, float("inf"))
    covered = distances <= float(T_AXIS_OVERLAP_TOL_PX)
    count = int(np.count_nonzero(covered))
    if count > 0:
        return (count, float(np.median(distances[covered])))
    return (0, float(np.min(distances)))


def _select_t_axis_overlay_by_coverage(
    *,
    primary: TAxisOverlay,
    mirrored: TAxisOverlay,
    coverage_mask_yx: np.ndarray,
) -> TAxisOverlay:
    primary_score, primary_median = _t_axis_line_coverage_metrics(t_axis_overlay=primary, mask_yx=coverage_mask_yx)
    mirrored_score, mirrored_median = _t_axis_line_coverage_metrics(t_axis_overlay=mirrored, mask_yx=coverage_mask_yx)
    if mirrored_score > primary_score:
        return mirrored
    if mirrored_score < primary_score:
        return primary
    return mirrored if mirrored_median < primary_median else primary


def _t_axis_range_for_mask(*, t_axis_overlay: TAxisOverlay, mask_yx: np.ndarray) -> tuple[float, float] | None:
    mask = np.asarray(mask_yx, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return None

    line_xy = np.asarray(t_axis_overlay.line_xy, dtype=np.float64)
    line_t = np.asarray(t_axis_overlay.line_t, dtype=np.float64)
    if line_xy.ndim != 2 or line_xy.shape[1] != 2 or line_t.ndim != 1 or line_t.shape[0] != line_xy.shape[0]:
        return None
    valid = np.isfinite(line_xy).all(axis=1) & np.isfinite(line_t)
    if not np.any(valid):
        return None
    line_xy = line_xy[valid]
    line_t = line_t[valid]

    mask_pts_yx = np.argwhere(mask)
    if mask_pts_yx.shape[0] == 0:
        return None
    tree = cKDTree(np.column_stack([line_xy[:, 1], line_xy[:, 0]]))
    distances, nearest_idx = tree.query(mask_pts_yx.astype(np.float64, copy=False), k=1)
    covered = np.isfinite(distances) & (distances <= float(T_AXIS_OVERLAP_TOL_PX))
    if not np.any(covered):
        return None
    nearest_idx = np.asarray(nearest_idx[covered], dtype=np.int64)
    if nearest_idx.size == 0:
        return None
    t_vals = np.sort(np.clip(line_t[nearest_idx], 0.0, 1.0))
    n = int(t_vals.size)
    if n == 0:
        return None
    trim = max(1, int(math.floor(0.01 * float(n))))
    if (2 * trim) >= n:
        trim = max(0, (n - 1) // 2)
    low_idx = int(trim)
    high_idx = int(n - 1 - trim)
    return (float(t_vals[low_idx]), float(t_vals[high_idx]))


def _t_axis_range_for_mask_with_mirroring(
    *,
    primary: TAxisOverlay | None,
    mirrored: TAxisOverlay | None,
    mask_yx: np.ndarray,
) -> tuple[tuple[float, float] | None, Literal["primary", "mirrored"] | None]:
    selected, side = _select_t_axis_overlay_for_mask_with_mirroring(
        primary=primary,
        mirrored=mirrored,
        mask_yx=mask_yx,
    )
    if selected is None:
        return (None, side)
    return (_t_axis_range_for_mask(t_axis_overlay=selected, mask_yx=mask_yx), side)


def _select_t_axis_overlay_for_mask_with_mirroring(
    *,
    primary: TAxisOverlay | None,
    mirrored: TAxisOverlay | None,
    mask_yx: np.ndarray,
) -> tuple[TAxisOverlay | None, Literal["primary", "mirrored"] | None]:
    if primary is None and mirrored is None:
        return (None, None)
    if primary is None:
        return (mirrored, "mirrored")
    if mirrored is None:
        return (primary, "primary")
    selected = _select_t_axis_overlay_by_coverage(
        primary=primary,
        mirrored=mirrored,
        coverage_mask_yx=np.asarray(mask_yx, dtype=bool),
    )
    side: Literal["primary", "mirrored"] = "mirrored" if selected is mirrored else "primary"
    return (selected, side)


def _t_axis_xy_for_t_value(*, t_axis_overlay: TAxisOverlay, t_value: float) -> np.ndarray | None:
    xy = np.asarray(t_axis_overlay.line_xy, dtype=np.float64)
    line_t = np.asarray(t_axis_overlay.line_t, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or line_t.ndim != 1 or line_t.shape[0] != xy.shape[0]:
        return None
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(line_t)
    if not np.any(valid):
        return None
    xy = xy[valid]
    line_t = line_t[valid]
    target = float(np.clip(t_value, 0.0, 1.0))
    idx = int(np.argmin(np.abs(line_t - target)))
    return np.asarray(xy[idx], dtype=np.float64)


def _moving_thumbnail_shape_yx(*, p1: P1Landmarks, target_spacing_um: float) -> tuple[int, int]:
    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216
    sr0, sr1, sc0, sc1 = p1.sample_rotated_crop_bbox
    h_px = int(sr1) - int(sr0)
    w_px = int(sc1) - int(sc0)
    if h_px <= 0 or w_px <= 0:
        raise ValueError(f"Invalid sample_rotated_crop_bbox={p1.sample_rotated_crop_bbox}.")
    scale = float(sample_voxel_um) / float(target_spacing_um)
    return (
        max(1, int(math.ceil(float(h_px) * scale))),
        max(1, int(math.ceil(float(w_px) * scale))),
    )


def _rasterize_imagej_roi_masks(
    *,
    roi_path: Path,
    shape_yx: tuple[int, int],
) -> list[tuple[int, str, np.ndarray]]:
    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    h, w = int(shape_yx[0]), int(shape_yx[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid shape_yx={shape_yx}.")

    roi_polys = load_roi_polygons(roi_path, scale=1.0)
    if not roi_polys:
        raise ValueError(f"No valid ROI polygons found in {roi_path}.")

    masks: list[tuple[int, str, np.ndarray]] = []
    for mask_index, entry in enumerate(roi_polys, start=1):
        mask = np.zeros((h, w), dtype=bool)
        geom = entry.geometry
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:  # pragma: no cover - load_roi_polygons filters unsupported geometries
            continue

        for poly in polys:
            minx, miny, maxx, maxy = poly.bounds
            c0 = max(0, int(math.floor(minx)))
            c1 = min(w - 1, int(math.ceil(maxx)))
            r0 = max(0, int(math.floor(miny)))
            r1 = min(h - 1, int(math.ceil(maxy)))
            if c1 < c0 or r1 < r0:
                continue

            xs = np.arange(c0, c1 + 1, dtype=np.float64) + 0.5
            ys = np.arange(r0, r1 + 1, dtype=np.float64) + 0.5
            xx, yy = np.meshgrid(xs, ys, indexing="xy")
            points = np.column_stack([xx.ravel(), yy.ravel()])

            ext = np.asarray(poly.exterior.coords, dtype=np.float64)
            if ext.ndim != 2 or ext.shape[1] != 2:
                continue
            inside = MplPath(ext, closed=True).contains_points(points, radius=1e-9)
            for hole in poly.interiors:
                hole_xy = np.asarray(hole.coords, dtype=np.float64)
                if hole_xy.ndim == 2 and hole_xy.shape[1] == 2:
                    inside &= ~MplPath(hole_xy, closed=True).contains_points(points, radius=1e-9)
            if np.any(inside):
                mask[r0 : (r1 + 1), c0 : (c1 + 1)] |= inside.reshape((ys.size, xs.size))

        if np.any(mask):
            masks.append((int(mask_index), str(entry.name), mask))

    if not masks:
        raise ValueError(f"Rasterized user ROI masks are empty for {roi_path}.")
    return masks


def _write_syn_zoom_overlay_with_user_mask(
    *,
    fixed_yx: np.ndarray,
    moving_yx: np.ndarray,
    moving_mask_yx: np.ndarray | None,
    user_mask_yx: np.ndarray,
    t_axis_overlay: TAxisOverlay | None,
    t_axis_endpoint_markers: list[dict[str, object]],
    reflect_t_axis_midline: bool,
    out_png: Path,
    axis3_out_png: Path | None,
    title: str,
) -> None:
    fixed = np.asarray(fixed_yx, dtype=np.float32)
    moving = np.asarray(moving_yx, dtype=np.float32)
    user_mask = np.asarray(user_mask_yx, dtype=bool)
    if fixed.shape != moving.shape or fixed.shape != user_mask.shape:
        raise ValueError(f"Overlay shape mismatch: fixed={fixed.shape}, moving={moving.shape}, user={user_mask.shape}.")

    moving_mask = None
    if moving_mask_yx is not None:
        moving_mask = np.asarray(moving_mask_yx, dtype=bool)
        if moving_mask.shape != fixed.shape:
            raise ValueError(f"moving_mask shape mismatch: moving_mask={moving_mask.shape}, fixed={fixed.shape}.")

    union = user_mask.copy()
    if moving_mask is not None:
        union |= moving_mask
    lo_hi = _bbox_indices_from_mask(mask_xy=union, pad_vox=20)
    lo: np.ndarray
    hi: np.ndarray
    if lo_hi is None:
        lo = np.asarray([fixed.shape[0], fixed.shape[1]], dtype=int)
        hi = np.asarray([0, 0], dtype=int)
    else:
        lo = np.asarray(lo_hi[0], dtype=int)
        hi = np.asarray(lo_hi[1], dtype=int)

    if t_axis_overlay is not None and t_axis_overlay.line_xy.size:
        t_xy = np.asarray(t_axis_overlay.line_xy, dtype=np.float64)
        valid = np.isfinite(t_xy).all(axis=1)
        t_xy = t_xy[valid]
        if t_xy.size:
            if reflect_t_axis_midline:
                t_xy_reflected = t_xy.copy()
                t_xy_reflected[:, 0] = (float(fixed.shape[1]) - 1.0) - t_xy_reflected[:, 0]
                t_xy = np.vstack([t_xy, t_xy_reflected])
            pad = 20
            row_min = max(0, int(math.floor(float(np.min(t_xy[:, 1])))) - pad)
            row_max = min(int(fixed.shape[0]), int(math.ceil(float(np.max(t_xy[:, 1])))) + pad + 1)
            col_min = max(0, int(math.floor(float(np.min(t_xy[:, 0])))) - pad)
            col_max = min(int(fixed.shape[1]), int(math.ceil(float(np.max(t_xy[:, 0])))) + pad + 1)
            lo = np.minimum(lo, np.asarray([row_min, col_min], dtype=int))
            hi = np.maximum(hi, np.asarray([row_max, col_max], dtype=int))

    if not ((hi > lo).all()):
        raise ValueError("Union mask and t-axis overlay are empty after warping user ROI mask.")

    row_slice = slice(int(lo[0]), int(hi[0]))
    col_slice = slice(int(lo[1]), int(hi[1]))

    f = normalize_robust(fixed[row_slice, col_slice].astype(np.float32))
    m = normalize_robust(moving[row_slice, col_slice].astype(np.float32))
    if moving_mask is not None:
        m = m * moving_mask[row_slice, col_slice].astype(np.float32)
    user = user_mask[row_slice, col_slice]

    overlay = np.stack([f, m, f], axis=-1)
    height_px = int(max(f.shape[0], m.shape[0]))
    width_px = int(max(f.shape[1], m.shape[1]))
    fig_w_in = max(15.0, (3.0 * float(width_px)) / 220.0)
    fig_h_in = max(5.0, float(height_px) / 220.0)

    fig, axes = plt.subplots(1, 3, figsize=(fig_w_in, fig_h_in))
    axes[0].imshow(f, cmap="gray", interpolation="nearest", resample=False)
    axes[0].set_title("Fixed (atlas crop)")
    axes[0].axis("off")
    axes[1].imshow(m, cmap="gray", interpolation="nearest", resample=False)
    axes[1].set_title("Warped moving")
    axes[1].axis("off")
    _draw_syn_overlay_axis3(
        ax=axes[2],
        fig=fig,
        overlay_rgb=overlay,
        user_mask_yx=user,
        fixed_shape_yx=fixed.shape,
        crop_lo_yx=lo,
        t_axis_overlay=t_axis_overlay,
        t_axis_endpoint_markers=t_axis_endpoint_markers,
        reflect_t_axis_midline=reflect_t_axis_midline,
        panel_title="Overlay + user ROI mask",
    )

    fig.suptitle(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    if axis3_out_png is not None:
        fig_axis3, ax_axis3 = plt.subplots(1, 1, figsize=(fig_w_in / 3.0, fig_h_in))
        _draw_syn_overlay_axis3(
            ax=ax_axis3,
            fig=fig_axis3,
            overlay_rgb=overlay,
            user_mask_yx=user,
            fixed_shape_yx=fixed.shape,
            crop_lo_yx=lo,
            t_axis_overlay=t_axis_overlay,
            t_axis_endpoint_markers=t_axis_endpoint_markers,
            reflect_t_axis_midline=reflect_t_axis_midline,
            panel_title="Overlay + user ROI mask (axis 3)",
        )
        fig_axis3.suptitle(title)
        plt.tight_layout()
        axis3_out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(axis3_out_png, dpi=220)
        plt.close(fig_axis3)


def _draw_syn_overlay_axis3(
    *,
    ax: object,
    fig: object,
    overlay_rgb: np.ndarray,
    user_mask_yx: np.ndarray,
    fixed_shape_yx: tuple[int, int],
    crop_lo_yx: np.ndarray,
    t_axis_overlay: TAxisOverlay | None,
    t_axis_endpoint_markers: list[dict[str, object]],
    reflect_t_axis_midline: bool,
    panel_title: str,
) -> None:
    ax.imshow(overlay_rgb, interpolation="nearest", resample=False)
    ax.set_title(panel_title)
    ax.axis("off")

    user = np.asarray(user_mask_yx, dtype=bool)
    user_alpha = np.where(user, 0.24, 0.0).astype(np.float32)
    user_rgba = np.stack(
        [
            np.ones_like(user_alpha),
            np.full_like(user_alpha, 0.20),
            np.full_like(user_alpha, 0.15),
            user_alpha,
        ],
        axis=-1,
    )
    ax.imshow(user_rgba, interpolation="nearest", resample=False)
    ax.contour(user.astype(np.uint8), levels=[0.5], colors=["#ffd54f"], linewidths=0.8, alpha=0.95)

    if t_axis_overlay is None:
        return

    line_t = np.asarray(t_axis_overlay.line_t, dtype=np.float64)
    norm = Normalize(vmin=0.0, vmax=1.0)
    xy = np.asarray(t_axis_overlay.line_xy, dtype=np.float64)
    minor_ticks_xy = np.asarray(t_axis_overlay.minor_tick_xy, dtype=np.float64)
    minor_ticks_t = np.asarray(t_axis_overlay.minor_tick_t, dtype=np.float64)
    ticks_xy = np.asarray(t_axis_overlay.tick_xy, dtype=np.float64)
    ticks_t = np.asarray(t_axis_overlay.tick_t, dtype=np.float64)
    overlays = [(xy, minor_ticks_xy, ticks_xy, True)]
    if reflect_t_axis_midline:
        mirrored_xy = xy.copy()
        mirrored_xy[:, 0] = (float(fixed_shape_yx[1]) - 1.0) - mirrored_xy[:, 0]
        mirrored_minor = minor_ticks_xy.copy()
        mirrored_minor[:, 0] = (float(fixed_shape_yx[1]) - 1.0) - mirrored_minor[:, 0]
        mirrored_ticks = ticks_xy.copy()
        mirrored_ticks[:, 0] = (float(fixed_shape_yx[1]) - 1.0) - mirrored_ticks[:, 0]
        overlays.append((mirrored_xy, mirrored_minor, mirrored_ticks, False))

    colorbar_added = False
    for line_xy, line_minor_xy, line_ticks_xy, draw_labels in overlays:
        if line_xy.shape[0] >= 2 and line_t.shape[0] == line_xy.shape[0]:
            xy_local = line_xy.copy()
            xy_local[:, 0] -= float(crop_lo_yx[1])
            xy_local[:, 1] -= float(crop_lo_yx[0])
            segments = np.stack([xy_local[:-1], xy_local[1:]], axis=1)
            segment_t = 0.5 * (line_t[:-1] + line_t[1:])
            valid_segments = np.isfinite(segments).all(axis=(1, 2)) & np.isfinite(segment_t)
            if np.any(valid_segments):
                line = LineCollection(
                    segments[valid_segments],
                    cmap=T_AXIS_T_CMAP,
                    norm=norm,
                    linewidths=2.0,
                    alpha=0.85,
                    zorder=10,
                    antialiased=True,
                )
                line.set_array(np.clip(segment_t[valid_segments], 0.0, 1.0))
                ax.add_collection(line)
                if not colorbar_added:
                    cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.02)
                    cbar.set_label("u-curve t")
                    colorbar_added = True

        if line_minor_xy.size:
            line_minor_local = line_minor_xy.copy()
            line_minor_local[:, 0] -= float(crop_lo_yx[1])
            line_minor_local[:, 1] -= float(crop_lo_yx[0])
            clipped_minor_t = np.clip(minor_ticks_t, 0.0, 1.0)
            ax.scatter(
                line_minor_local[:, 0],
                line_minor_local[:, 1],
                s=8,
                c=clipped_minor_t,
                cmap=T_AXIS_T_CMAP,
                norm=norm,
                marker="o",
                linewidths=0.0,
                alpha=0.9,
                zorder=10.5,
            )

        if line_ticks_xy.size:
            line_ticks_local = line_ticks_xy.copy()
            line_ticks_local[:, 0] -= float(crop_lo_yx[1])
            line_ticks_local[:, 1] -= float(crop_lo_yx[0])
            clipped_ticks_t = np.clip(ticks_t, 0.0, 1.0)
            ax.scatter(
                line_ticks_local[:, 0],
                line_ticks_local[:, 1],
                s=18,
                c=clipped_ticks_t,
                cmap=T_AXIS_T_CMAP,
                norm=norm,
                marker="o",
                linewidths=0.0,
                alpha=0.95,
                zorder=11,
            )
            if draw_labels:
                for (x, y), label in zip(line_ticks_local.tolist(), t_axis_overlay.tick_labels, strict=False):
                    ax.text(
                        float(x) + 4.0,
                        float(y) - 4.0,
                        str(label),
                        color="white",
                        fontsize=7,
                        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.4},
                        zorder=12,
                    )

    if not t_axis_endpoint_markers:
        return
    for marker in t_axis_endpoint_markers:
        mask_name = str(marker.get("mask_name", "mask"))
        for endpoint_key, endpoint_label in (("begin", "start"), ("end", "end")):
            xy_marker = marker.get(f"{endpoint_key}_xy")
            t_val = marker.get(f"{endpoint_key}_t")
            if not isinstance(t_val, (float, int)):
                continue
            xy_local = np.asarray(xy_marker, dtype=np.float64).copy()
            if xy_local.shape != (2,):
                continue
            xy_local[0] -= float(crop_lo_yx[1])
            xy_local[1] -= float(crop_lo_yx[0])
            color = plt.get_cmap(T_AXIS_T_CMAP)(float(np.clip(float(t_val), 0.0, 1.0)))
            ax.scatter(
                [xy_local[0]],
                [xy_local[1]],
                s=70,
                c=["black"],
                marker="x",
                linewidths=2.2,
                alpha=0.95,
                zorder=13,
            )
            ax.scatter(
                [xy_local[0]],
                [xy_local[1]],
                s=46,
                c=[color],
                marker="x",
                linewidths=1.7,
                alpha=0.95,
                zorder=14,
            )
            ax.text(
                float(xy_local[0]) + 4.0,
                float(xy_local[1]) + 4.0,
                f"{mask_name} {endpoint_label}",
                color="white",
                fontsize=6,
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.2},
                zorder=15,
            )


def _write_ccf_user_mask_overlay_png(
    *,
    ws: Workspace,
    roi: str,
    run_dirname: str,
    roi_path: Path,
    imagej_target_spacing_um: float,
    overwrite: bool,
    keep_t_mask: bool = False,
) -> dict[str, Path] | None:
    run_dir = ws.ccf_transforms(roi) / str(run_dirname)
    summary_path = run_dir / "similarity_plus_syn_summary.json"
    summary = cast(_Summary, _load_json_object(summary_path))

    if "fwdtransforms" not in summary:
        raise KeyError(f"Missing fwdtransforms in {summary_path}.")
    fwd = _resolve_paths(_as_list_of_str(summary["fwdtransforms"], key="fwdtransforms", path=summary_path), base_dir=run_dir)
    if not fwd:
        raise ValueError(f"No forward transforms listed in {summary_path}.")

    paths = summary.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"Invalid paths payload in {summary_path}: expected object.")

    def path_from_summary(key: str, default_name: str) -> Path:
        raw = paths.get(key, default_name)
        p = Path(str(raw))
        if not p.is_absolute():
            p = run_dir / p
        return p

    fixed_nifti = path_from_summary("fixed_nifti", "fixed_atlas_crop.nii.gz")
    warped_after_nifti = path_from_summary("warped_after_nifti", "moving_warped_similarity_plus_syn.nii.gz")
    moving_mask_nifti = path_from_summary("moving_mask_warped_final_nifti", "moving_mask_warped_final.nii.gz")
    qc_zoom_masked_png = path_from_summary("qc_zoom_masked_png", "similarity_plus_syn_qc_zoom_masked.png")

    out_png = qc_zoom_masked_png.with_name("similarity_plus_syn_qc_zoom_masked_with_user_mask.png")
    out_axis3_png = qc_zoom_masked_png.with_name("similarity_plus_syn_qc_zoom_masked_with_user_mask_axis3.png")
    out_json = qc_zoom_masked_png.with_name("similarity_plus_syn_qc_zoom_masked_with_user_mask_t_axis_endpoints.json")
    skip_overlay_write = bool(out_png.exists() and out_axis3_png.exists() and not overwrite)
    if skip_overlay_write and out_json.exists() and not overwrite:
        return None

    preserve_saved_t_mask = bool(out_json.exists() and (keep_t_mask or not overwrite))
    saved_endpoints_by_mask: dict[tuple[int, str], dict[str, object]] = {}
    if preserve_saved_t_mask:
        saved_payload = _load_json_object(out_json)
        masks_payload = saved_payload.get("masks")
        if isinstance(masks_payload, list):
            for entry in masks_payload:
                if not isinstance(entry, dict):
                    continue
                mask_index_raw = entry.get("mask_index")
                mask_name_raw = entry.get("mask_name")
                if isinstance(mask_index_raw, int) and isinstance(mask_name_raw, str):
                    saved_endpoints_by_mask[(int(mask_index_raw), str(mask_name_raw))] = cast(dict[str, object], entry)

    def _saved_t_range_for_mask(entry: dict[str, object] | None) -> tuple[float, float] | None:
        if entry is None:
            return None
        begin_raw = entry.get("begin")
        end_raw = entry.get("end")
        if not isinstance(begin_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            return None
        begin = float(begin_raw)
        end = float(end_raw)
        if not np.isfinite(begin) or not np.isfinite(end) or begin > end:
            return None
        return begin, end

    for required_path in (fixed_nifti, warped_after_nifti):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file for CCF user-mask overlay: {required_path}")
    for tfm in fwd:
        if not Path(tfm).exists():
            raise FileNotFoundError(f"Missing transform file for CCF user-mask overlay: {tfm}")

    out_contract = LandmarkRegistrationOutputs(ws.ccf_transforms(roi))
    p1 = out_contract.read_p1_landmarks()
    shape_yx = _moving_thumbnail_shape_yx(p1=p1, target_spacing_um=float(imagej_target_spacing_um))
    roi_masks_moving = _rasterize_imagej_roi_masks(roi_path=roi_path, shape_yx=shape_yx)

    fixed_ants = ants.image_read(str(fixed_nifti))
    moving_warped_ants = ants.image_read(str(warped_after_nifti))
    moving_mask_ants = ants.image_read(str(moving_mask_nifti)) if moving_mask_nifti.exists() else None
    atlas_name = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
    atlas_plane = p1.atlas_plane or ("sagittal" if "Sag" in str(ws.path) else "coronal")
    if p1.atlas_slice_idx is None:
        raise ValueError(f"p1_landmarks.json at {out_contract.p1_landmarks_json} is missing atlas_slice_idx.")
    atlas_slice_idx = int(p1.atlas_slice_idx)
    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0

    fixed_yx = _ants_numpy_yx(fixed_ants)
    moving_yx = _ants_numpy_yx(moving_warped_ants)
    moving_mask_yx = _ants_numpy_yx(moving_mask_ants) > 0 if moving_mask_ants is not None else None
    roi_mask_fixed_yx = np.zeros_like(fixed_yx, dtype=bool)
    warped_masks_fixed: list[tuple[int, str, np.ndarray]] = []
    t_axis_endpoint_markers: list[dict[str, object]] = []

    per_mask_payload: list[dict[str, object]] = []
    for mask_index, mask_name, moving_mask_yx_single in roi_masks_moving:
        roi_mask_moving_ants = _ants_from_mask_yx(
            mask_yx=moving_mask_yx_single, spacing_um=float(imagej_target_spacing_um)
        )
        roi_mask_fixed_ants = ants.apply_transforms(
            fixed=fixed_ants,
            moving=roi_mask_moving_ants,
            transformlist=fwd,
            interpolator="nearestNeighbor",
            defaultvalue=0,
        )
        fixed_mask_single = _ants_numpy_yx(roi_mask_fixed_ants) > 0
        roi_mask_fixed_yx |= fixed_mask_single
        warped_masks_fixed.append((int(mask_index), str(mask_name), fixed_mask_single))

    t_axis_overlay = _compute_t_axis_overlay_for_fixed_view(
        fixed_view=fixed_ants,
        atlas_name=atlas_name,
        atlas_plane=atlas_plane,
        atlas_slice_idx=atlas_slice_idx,
        atlas_crop_bbox=p1.atlas_crop_bbox,
        atlas_voxel_um=atlas_voxel_um,
        coverage_mask_yx=roi_mask_fixed_yx,
        side_filter="both",
        tick_step=0.1,
    )

    for mask_index, mask_name, fixed_mask_single in warped_masks_fixed:
        saved_row = saved_endpoints_by_mask.get((int(mask_index), str(mask_name)))
        row: dict[str, object] = {
            "mask_index": int(mask_index),
            "mask_name": str(mask_name),
            "selected_side": None,
            "has_overlap_with_t_axis": False,
            "begin": None,
            "end": None,
        }
        t_axis_overlay_primary_mask = _compute_t_axis_overlay_for_fixed_view(
            fixed_view=fixed_ants,
            atlas_name=atlas_name,
            atlas_plane=atlas_plane,
            atlas_slice_idx=atlas_slice_idx,
            atlas_crop_bbox=p1.atlas_crop_bbox,
            atlas_voxel_um=atlas_voxel_um,
            coverage_mask_yx=fixed_mask_single,
            side_filter="primary",
            tick_step=0.1,
        )
        t_axis_overlay_mirrored_mask: TAxisOverlay | None = None
        if str(atlas_plane).lower() == "coronal":
            t_axis_overlay_mirrored_mask = _compute_t_axis_overlay_for_fixed_view(
                fixed_view=fixed_ants,
                atlas_name=atlas_name,
                atlas_plane=atlas_plane,
                atlas_slice_idx=atlas_slice_idx,
                atlas_crop_bbox=p1.atlas_crop_bbox,
                atlas_voxel_um=atlas_voxel_um,
                coverage_mask_yx=fixed_mask_single,
                side_filter="mirrored",
                tick_step=0.1,
            )
        selected_overlay, selected_side = _select_t_axis_overlay_for_mask_with_mirroring(
            primary=t_axis_overlay_primary_mask,
            mirrored=t_axis_overlay_mirrored_mask,
            mask_yx=fixed_mask_single,
        )
        selected_side_from_saved = saved_row.get("selected_side") if saved_row is not None else None
        if selected_side_from_saved == "primary" and t_axis_overlay_primary_mask is not None:
            selected_overlay = t_axis_overlay_primary_mask
            selected_side = "primary"
        elif selected_side_from_saved == "mirrored" and t_axis_overlay_mirrored_mask is not None:
            selected_overlay = t_axis_overlay_mirrored_mask
            selected_side = "mirrored"
        t_range = _saved_t_range_for_mask(saved_row)
        if t_range is None and selected_overlay is not None:
            t_range = _t_axis_range_for_mask(t_axis_overlay=selected_overlay, mask_yx=fixed_mask_single)
        row["selected_side"] = selected_side
        if t_range is not None:
            row["has_overlap_with_t_axis"] = True
            row["begin"] = float(t_range[0])
            row["end"] = float(t_range[1])
            if selected_overlay is not None:
                begin_xy = _t_axis_xy_for_t_value(t_axis_overlay=selected_overlay, t_value=float(t_range[0]))
                end_xy = _t_axis_xy_for_t_value(t_axis_overlay=selected_overlay, t_value=float(t_range[1]))
                if begin_xy is not None and end_xy is not None:
                    t_axis_endpoint_markers.append(
                        {
                            "mask_index": int(mask_index),
                            "mask_name": str(mask_name),
                            "selected_side": selected_side,
                            "begin_t": float(t_range[0]),
                            "end_t": float(t_range[1]),
                            "begin_xy": begin_xy,
                            "end_xy": end_xy,
                        }
                    )
        per_mask_payload.append(row)

    written: dict[str, Path] = {}
    if not skip_overlay_write:
        _write_syn_zoom_overlay_with_user_mask(
            fixed_yx=fixed_yx,
            moving_yx=moving_yx,
            moving_mask_yx=moving_mask_yx,
            user_mask_yx=roi_mask_fixed_yx,
            t_axis_overlay=t_axis_overlay,
            t_axis_endpoint_markers=t_axis_endpoint_markers,
            reflect_t_axis_midline=str(atlas_plane).lower() == "coronal",
            out_png=out_png,
            axis3_out_png=out_axis3_png,
            title=f"SyN(MI) zoom (masked) + user ROI | roi={roi}",
        )
        written["overlay_png"] = out_png
        written["overlay_axis3_png"] = out_axis3_png

    endpoints_payload: dict[str, object] = {
        "roi": str(roi),
        "value_domain": [0.0, 1.0],
        "available": t_axis_overlay is not None,
        "masks": per_mask_payload,
    }
    if t_axis_overlay is None:
        endpoints_payload["reason"] = "t_axis_overlay_unavailable"
    if not preserve_saved_t_mask and (overwrite or not out_json.exists()):
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(endpoints_payload, indent=2), encoding="utf-8")
        written["t_axis_endpoints_json"] = out_json
    return written


def _write_qc_mask_overlay_plot(
    *,
    coords_xy: np.ndarray,
    keep_mask: np.ndarray,
    output_png: Path,
    title: str,
    units: str,
    max_points: int = 200_000,
) -> None:
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(f"Expected coords_xy with shape (N,2), got {coords_xy.shape}.")

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.ndim != 1 or keep_mask.shape[0] != coords_xy.shape[0]:
        raise ValueError(f"keep_mask shape mismatch: {keep_mask.shape} vs coords {coords_xy.shape}.")

    n = int(coords_xy.shape[0])
    stride = max(1, int(math.ceil(n / int(max_points)))) if max_points > 0 else 1
    idx = np.arange(0, n, stride, dtype=np.int64)

    coords_p = coords_xy[idx]
    keep_p = keep_mask[idx]
    kept = coords_p[keep_p]
    dropped = coords_p[~keep_p]

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5), dpi=180)
    ax.set_aspect("equal")

    s_drop = float(np.clip(2000.0 / max(1, dropped.shape[0]), 0.02, 1.2))
    s_keep = float(np.clip(2000.0 / max(1, kept.shape[0]), 0.05, 3.0))

    if dropped.shape[0] > 0:
        ax.scatter(
            dropped[:, 0],
            dropped[:, 1],
            s=s_drop,
            alpha=0.08,
            linewidths=0,
            color="#666666",
            rasterized=True,
            label=f"dropped ({int((~keep_mask).sum())})",
        )
    if kept.shape[0] > 0:
        ax.scatter(
            kept[:, 0],
            kept[:, 1],
            s=s_keep,
            alpha=0.35,
            linewidths=0,
            color="tab:blue",
            rasterized=True,
            label=f"kept ({int(keep_mask.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel(f"X ({units})")
    ax.set_ylabel(f"Y ({units})")
    ax.invert_yaxis()
    ax.legend(loc="best", markerscale=5)

    fig.tight_layout()
    fig.savefig(output_png.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mask_ccf_exact(adata: ad.AnnData, term: int | str, *, kind: CCFTermKind, obsm_key: str) -> np.ndarray:
    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{obsm_key!r}].")
    table = adata.obsm[obsm_key]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected adata.obsm[{obsm_key!r}] to be a DataFrame, got {type(table).__name__}.")

    kind_norm = cast(CCFTermKind, str(kind).lower())
    if kind_norm not in {"auto", "id", "acronym", "name"}:
        raise ValueError(f"Invalid kind={kind!r}. Expected 'auto'|'id'|'acronym'|'name'.")

    if kind_norm == "auto":
        if isinstance(term, int):
            kind_norm = "id"
        else:
            s = str(term).strip()
            if s.isdigit():
                kind_norm = "id"
            else:
                mask_acr = _mask_ccf_exact(adata, s, kind="acronym", obsm_key=obsm_key)
                mask_name = _mask_ccf_exact(adata, s, kind="name", obsm_key=obsm_key)
                return np.asarray(mask_acr, dtype=bool) | np.asarray(mask_name, dtype=bool)

    if kind_norm == "id":
        if "id" not in table.columns:
            raise KeyError(f"Missing 'id' column in adata.obsm[{obsm_key!r}].")
        q = int(term)
        return table["id"].to_numpy(dtype=np.int64, copy=False) == q

    if kind_norm == "acronym":
        if "acronym" not in table.columns:
            raise KeyError(f"Missing 'acronym' column in adata.obsm[{obsm_key!r}].")
        q = str(term).strip().lower()
        col = table["acronym"].astype("string")
        return col.str.lower().to_numpy() == q

    if "name" not in table.columns:
        raise KeyError(f"Missing 'name' column in adata.obsm[{obsm_key!r}].")
    q = str(term).strip().lower()
    col = table["name"].astype("string")
    return col.str.lower().to_numpy() == q


def _mask_for_term(adata: ad.AnnData, term: str, *, kind: CCFTermKind, match: MatchMode, obsm_key: str) -> np.ndarray:
    if match == "subtree":
        return mask_ccf_subtree(adata, term, kind=kind, obsm_key=obsm_key)
    return _mask_ccf_exact(adata, term, kind=kind, obsm_key=obsm_key)


def _coords_to_um(
    coords: np.ndarray,
    *,
    units: CoordUnits,
    spacing_um: float | None,
) -> np.ndarray:
    coords_f = np.asarray(coords, dtype=np.float64)
    if coords_f.ndim != 2 or coords_f.shape[1] != 2:
        raise ValueError(f"Expected coords with shape (N,2), got {coords_f.shape}.")

    if units == "um":
        return coords_f
    if units == "mm":
        return coords_f * 1000.0
    if units == "px":
        if spacing_um is None:
            raise ValueError("coords spacing_um is required when coords_units='px'.")
        return coords_f * float(spacing_um)
    raise ValueError(f"Unsupported coords units: {units!r}.")


def _coords_to_px(
    coords: np.ndarray,
    *,
    units: CoordUnits,
    spacing_um: float,
) -> np.ndarray:
    coords_f = np.asarray(coords, dtype=np.float64)
    if coords_f.ndim != 2 or coords_f.shape[1] != 2:
        raise ValueError(f"Expected coords with shape (N,2), got {coords_f.shape}.")

    if units == "px":
        return coords_f
    if units == "um":
        return coords_f / float(spacing_um)
    if units == "mm":
        return (coords_f * 1000.0) / float(spacing_um)
    raise ValueError(f"Unsupported coords units: {units!r}.")


def _input_h5ad_candidate(ws: Workspace, *, roi: str, h5ad_name: str | None) -> Path:
    base = ws.output.ccf_transforms / roi
    if h5ad_name is None:
        return base / f"{roi}.syn.h5ad"
    else:
        name = str(h5ad_name).strip()
        if not name:
            raise click.BadParameter("--h5ad-name must be non-empty.")
        if Path(name).name != name:
            raise click.BadParameter("--h5ad-name must be a filename only (no directories).")
        if not name.endswith(".h5ad"):
            name = f"{name}.h5ad"
        return base / name


def _resolve_input_h5ad(ws: Workspace, *, roi: str, h5ad_name: str | None) -> Path:
    candidate = _input_h5ad_candidate(ws, roi=roi, h5ad_name=h5ad_name)
    if not candidate.exists():
        raise click.ClickException(
            f"Missing warped h5ad for roi={roi!r}: {candidate}. "
            "Run `ccf-warp-h5ad-spatial <workspace> <roi> ...` first (or pass --h5ad-name)."
        )
    return candidate


def _default_outputs(
    ws: Workspace,
    *,
    roi: str,
    input_h5ad: Path,
    out_name: str | None,
) -> tuple[str, Path, Path]:
    out_dir = ws.output.ccf_transforms / roi
    if out_name is None:
        out_name = f"{input_h5ad.stem}.annotated.h5ad"
    output_h5ad = out_dir / out_name
    plot_png = output_h5ad.with_suffix(".qc.png")
    return (str(roi), output_h5ad, plot_png)


def _ensure_ccf_obsm(
    adata: ad.AnnData,
    *,
    input_h5ad: Path,
    ccf_obsm_key: str,
    coords_key: str,
    coords_units: CoordUnits,
    coords_space: CoordSpace,
    spatial_order: SpatialOrder,
    roi: str,
    workspace: Path,
) -> None:
    if ccf_obsm_key in adata.obsm:
        return
    if coords_key not in adata.obsm:
        raise click.ClickException(
            f"Missing adata.obsm[{ccf_obsm_key!r}] in {input_h5ad}, and cannot auto-annotate because "
            f"adata.obsm[{coords_key!r}] is missing."
        )

    ws = Workspace(workspace)
    out_root = ws.ccf_transforms(roi)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    atlas_voxel_um = float(p1.atlas_voxel_um) if p1.atlas_voxel_um is not None else 20.0
    atlas_name = p1.atlas_name or "kim_dev_mouse_e15-5_lsfm_20um"
    atlas_plane = p1.atlas_plane or ("sagittal" if "Sag" in str(ws.path) else "coronal")
    if p1.atlas_slice_idx is None:
        raise click.ClickException(f"p1_landmarks.json is missing atlas_slice_idx under {out_root}.")
    atlas_slice_idx = int(p1.atlas_slice_idx)

    ar0, ar1, ac0, ac1 = p1.atlas_crop_bbox
    crop_offset_xy = (float(ac0), float(ar0))

    atlas = BrainGlobeAtlas(atlas_name)
    ann_vol = atlas.annotation if atlas_plane == "coronal" else atlas.annotation.transpose(2, 1, 0)
    ann_slice = np.asarray(ann_vol[atlas_slice_idx, :, :], dtype=np.uint32)
    if ann_slice.ndim != 2:
        raise click.ClickException(f"Expected 2D annotation slice, got shape={ann_slice.shape}.")

    coords = np.asarray(adata.obsm[coords_key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise click.ClickException(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}.")
    coords = coords.astype(np.float64, copy=False)
    if spatial_order == "yx":
        coords = coords[:, ::-1]

    coords_px = _coords_to_px(coords, units=coords_units, spacing_um=atlas_voxel_um)
    x_px = coords_px[:, 0]
    y_px = coords_px[:, 1]

    if coords_space == "crop":
        x_full = x_px + float(crop_offset_xy[0])
        y_full = y_px + float(crop_offset_xy[1])
    else:
        x_full = x_px
        y_full = y_px

    xi = np.round(x_full).astype(np.int64, copy=False)
    yi = np.round(y_full).astype(np.int64, copy=False)

    h, w = ann_slice.shape
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ids = np.zeros(int(adata.n_obs), dtype=np.uint32)
    ids[in_bounds] = ann_slice[yi[in_bounds], xi[in_bounds]]

    structures = atlas.structures
    tree = structures.tree

    uniq, inv = np.unique(ids.astype(np.int64), return_inverse=True)

    def _info_for_id(structure_id: int) -> tuple[str, str, int, str, str, str, bool]:
        if structure_id == 0:
            return ("background", "background", -1, "", "", "", False)
        st = structures[int(structure_id)]
        acronym = str(st["acronym"])
        name = str(st["name"])
        path_ids = [int(v) for v in st["structure_id_path"]]
        parent_id = int(path_ids[-2]) if len(path_ids) >= 2 else int(structure_id)
        path_ids_json = json.dumps(path_ids, separators=(",", ":"))
        path_acronyms = "/".join(str(structures[i]["acronym"]) for i in path_ids)
        path_names = "/".join(str(structures[i]["name"]) for i in path_ids)
        is_leaf = len(tree.children(int(structure_id))) == 0
        return (acronym, name, parent_id, path_ids_json, path_acronyms, path_names, is_leaf)

    uniq_acronym: list[str] = []
    uniq_name: list[str] = []
    uniq_parent: list[int] = []
    uniq_path_ids: list[str] = []
    uniq_path_acr: list[str] = []
    uniq_path_name: list[str] = []
    uniq_is_leaf: list[bool] = []
    for sid in uniq.tolist():
        a, n, pid, pids, pacr, pname, leaf = _info_for_id(int(sid))
        uniq_acronym.append(a)
        uniq_name.append(n)
        uniq_parent.append(pid)
        uniq_path_ids.append(pids)
        uniq_path_acr.append(pacr)
        uniq_path_name.append(pname)
        uniq_is_leaf.append(leaf)

    ccf_df = pd.DataFrame(
        {
            "id": ids.astype(np.int64, copy=False),
            "acronym": pd.Categorical(np.asarray(uniq_acronym, dtype=object)[inv]),
            "name": pd.Categorical(np.asarray(uniq_name, dtype=object)[inv]),
            "parent_id": np.asarray(uniq_parent, dtype=np.int64)[inv],
            "path_ids": pd.Categorical(np.asarray(uniq_path_ids, dtype=object)[inv]),
            "path_acronyms": pd.Categorical(np.asarray(uniq_path_acr, dtype=object)[inv]),
            "path_names": pd.Categorical(np.asarray(uniq_path_name, dtype=object)[inv]),
            "is_leaf": np.asarray(uniq_is_leaf, dtype=bool)[inv],
            "in_bounds": in_bounds,
        },
        index=adata.obs_names,
    )
    adata.obsm[ccf_obsm_key] = ccf_df
    adata.uns.setdefault(f"{ccf_obsm_key}_atlas", {})
    adata.uns[f"{ccf_obsm_key}_atlas"] = {
        "atlas_name": atlas_name,
        "atlas_plane": atlas_plane,
        "atlas_slice_idx": atlas_slice_idx,
        "atlas_voxel_um": atlas_voxel_um,
        "coords_key": coords_key,
        "coords_space": coords_space,
        "coords_units": coords_units,
        "spatial_order": spatial_order,
        "atlas_crop_bbox": [int(ar0), int(ar1), int(ac0), int(ac1)],
    }


def _dilate_mask_by_radius_um(
    coords_um_xy: np.ndarray,
    *,
    base_mask: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    base_mask = np.asarray(base_mask, dtype=bool)
    if base_mask.ndim != 1 or base_mask.shape[0] != coords_um_xy.shape[0]:
        raise ValueError(f"base_mask shape mismatch: {base_mask.shape} vs coords {coords_um_xy.shape}.")

    r = float(radius_um)
    if r <= 0:
        return base_mask

    seeds = coords_um_xy[base_mask]
    if seeds.shape[0] == 0:
        return np.zeros_like(base_mask, dtype=bool)

    tree = cKDTree(seeds)
    dist, _ = tree.query(coords_um_xy, k=1, distance_upper_bound=r, workers=-1)
    return np.isfinite(dist)


def _find_imagej_roi_file(mask_edit_dir: Path) -> Path | None:
    preferred = mask_edit_dir / "RoiSet.zip"
    if preferred.exists():
        return preferred

    rois = sorted(mask_edit_dir.glob("*.roi"))
    if len(rois) == 1:
        return rois[0]
    if len(rois) > 1:
        raise click.ClickException(
            "Multiple ImageJ ROI files found under "
            f"{mask_edit_dir} ({', '.join(p.name for p in rois)}). "
            "Pass --imagej-roi-path to disambiguate."
        )
    return None


def _to_moving_thumbnail_px_xy(
    coords_xy: np.ndarray,
    *,
    ws: Workspace,
    roi: str,
    input_space: InputSpace,
    input_units: CoordUnits,
    spatial_order: SpatialOrder,
    stitch_codebook: str,
    target_spacing_um: float,
) -> np.ndarray:
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords_xy with shape (N,2), got {coords.shape}.")

    if spatial_order == "yx":
        coords = coords[:, ::-1]

    out_root = ws.ccf_transforms(roi)
    out_contract = LandmarkRegistrationOutputs(out_root)
    p1 = out_contract.read_p1_landmarks()

    if input_units != "px":
        raise click.ClickException("--imagej-input-units currently only supports 'px'.")

    sample_voxel_um = float(p1.sample_voxel_xy_um) if p1.sample_voxel_xy_um is not None else 0.216
    crop_bbox = p1.sample_rotated_crop_bbox
    sr0, sr1, sc0, sc1 = (int(crop_bbox[0]), int(crop_bbox[1]), int(crop_bbox[2]), int(crop_bbox[3]))
    if sr1 <= sr0 or sc1 <= sc0:
        raise ValueError(f"Invalid sample_rotated_crop_bbox={crop_bbox}.")

    x_in = coords[:, 0]
    y_in = coords[:, 1]

    if input_space == "crop":
        x_crop = x_in
        y_crop = y_in
    elif input_space == "full":
        x_crop = x_in - float(sc0)
        y_crop = y_in - float(sr0)
    else:
        fused_zarr = ws.stitch(roi, stitch_codebook) / "fused.zarr"
        if not fused_zarr.exists():
            raise FileNotFoundError(f"Missing fused.zarr at {fused_zarr} (needed for --imagej-input-space=fused).")
        import zarr

        arr = zarr.open(str(fused_zarr), mode="r")
        fused_shape_yx = (int(arr.shape[1]), int(arr.shape[2]))
        x_crop, y_crop, _ = fused_xy_to_rotated_crop_xy(
            x_fused=x_in,
            y_fused=y_in,
            fused_shape_yx=fused_shape_yx,
            prior_flip_x=bool(p1.prior_flip_x),
            prior_rotation_deg=float(p1.prior_rotation_deg),
            rotated_crop_bbox=(sr0, sr1, sc0, sc1),
        )

    target_um = float(target_spacing_um)
    if not target_um > 0:
        raise click.ClickException("--imagej-target-spacing-um must be > 0.")
    scale = float(sample_voxel_um) / target_um
    return np.stack([x_crop * scale, y_crop * scale], axis=1)


def _mask_from_imagej_roi(
    *,
    ws: Workspace,
    roi: str,
    adata: ad.AnnData,
    roi_path: Path,
    coords_key: str,
    input_space: InputSpace,
    input_units: CoordUnits,
    spatial_order: SpatialOrder,
    stitch_codebook: str,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords_key not in adata.obsm:
        raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in input h5ad.")

    coords_xy = np.asarray(adata.obsm[coords_key])
    coords_thumb_xy = _to_moving_thumbnail_px_xy(
        coords_xy,
        ws=ws,
        roi=roi,
        input_space=input_space,
        input_units=input_units,
        spatial_order=spatial_order,
        stitch_codebook=stitch_codebook,
        target_spacing_um=target_spacing_um,
    )

    roi_polys = load_roi_polygons(roi_path, scale=1.0)
    geometries = [p.geometry for p in roi_polys]

    from matplotlib.path import Path as MplPath
    from shapely.geometry import MultiPolygon, Polygon

    keep = np.zeros(int(coords_thumb_xy.shape[0]), dtype=bool)
    labels = np.empty(int(coords_thumb_xy.shape[0]), dtype=object)
    labels[:] = ""
    points = np.asarray(coords_thumb_xy, dtype=np.float64, order="C")
    for entry, geom in zip(roi_polys, geometries, strict=True):
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:  # pragma: no cover - load_roi_polygons filters these out
            continue

        for poly in polys:
            ext = np.asarray(poly.exterior.coords, dtype=np.float64)
            if ext.ndim != 2 or ext.shape[1] != 2:
                continue
            # radius>0 includes boundary points (similar to shapely 'covers').
            path = MplPath(ext, closed=True)
            inside = path.contains_points(points, radius=1e-9)
            keep |= inside
            new = inside & (labels == "")
            if np.any(new):
                labels[new] = entry.name

    return keep, coords_thumb_xy, labels


@click.command("filter-h5ad-ccf")
@click.argument(
    "workspace",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument("rois", nargs=-1, type=str)
@click.option(
    "--h5ad-name",
    default=None,
    show_default=False,
    help=(
        "Input h5ad filename under <workspace>/analysis/output/ccf-transforms/<roi>/. "
        "Default: <roi>.syn.h5ad"
    ),
)
@click.option(
    "--out-name",
    default=None,
    show_default=False,
    help=(
        "Output filename (written under <workspace>/analysis/output/ccf-transforms/<roi>/). "
        "Default: <input_stem>.annotated.h5ad."
    ),
)
@click.option(
    "--ccf-col",
    default="ccf",
    show_default=True,
    help="obs column name to write term/atlas-derived selection labels into (empty string if not selected).",
)
@click.option(
    "--ccf-adjusted-col",
    default="ccf_adjusted",
    show_default=True,
    help="obs column name to write ImageJ ROI override labels into (empty string if not selected).",
)
@click.option(
    "--term",
    "terms",
    multiple=True,
    required=False,
    help="Ontology term(s) to select (name/acronym/id). Repeatable.",
)
@click.option(
    "--imagej-roi",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Use ImageJ ROI filtering by auto-detecting a .roi/RoiSet.zip under mask_edit/. "
        "If used, overrides --term."
    ),
)
@click.option(
    "--imagej-roi-path",
    default=None,
    show_default=False,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Explicit ImageJ annotation path (.roi or RoiSet.zip). Overrides --term.",
)
@click.option(
    "--ignore-imagej-roi/--no-ignore-imagej-roi",
    default=False,
    show_default=True,
    help="Ignore any auto-detected ImageJ ROI and fall back to ontology term filtering.",
)
@click.option(
    "--run-dirname",
    default="landmark_syn_mi",
    show_default=True,
    help="Subdirectory under <workspace>/analysis/output/ccf-transforms/<roi>/ containing mask_edit/.",
)
@click.option(
    "--imagej-coords-key",
    default="spatial",
    show_default=True,
    help="obsm key for coordinates to test against the ImageJ ROI (typically the original 'spatial').",
)
@click.option(
    "--imagej-input-space",
    type=click.Choice(["fused", "full", "crop"], case_sensitive=False),
    default="fused",
    show_default=True,
    help=(
        "Coordinate frame for obsm[imagej_coords_key] when using ImageJ ROI filtering. "
        "'fused' = unrotated fused.zarr pixel space; "
        "'full' = rotated full-res slice pixel space; "
        "'crop' = rotated crop-local pixel space."
    ),
)
@click.option(
    "--imagej-input-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
    help="Units for obsm[imagej_coords_key] when using ImageJ ROI filtering.",
)
@click.option(
    "--imagej-target-spacing-um",
    type=float,
    default=2.0,
    show_default=True,
    help="Pixel size (microns) of the moving thumbnail used for ImageJ ROI annotation.",
)
@click.option(
    "--stitch-codebook",
    default="pi",
    show_default=True,
    help="Stitch codebook name used to locate fused.zarr when --imagej-input-space=fused.",
)
@click.option(
    "--kind",
    type=click.Choice(["auto", "id", "acronym", "name"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Interpretation of each --term value.",
)
@click.option(
    "--match",
    type=click.Choice(["subtree", "exact"], case_sensitive=False),
    default="subtree",
    show_default=True,
    help="Whether to match the entire ontology subtree (includes descendants) or exact region labels.",
)
@click.option(
    "--combine",
    type=click.Choice(["any", "all"], case_sensitive=False),
    default="any",
    show_default=True,
    help="How to combine multiple --term masks.",
)
@click.option("--invert/--no-invert", default=False, show_default=True, help="Invert the final selection.")
@click.option("--ccf-obsm-key", default="ccf", show_default=True, help="obsm key containing CCF columns.")
@click.option(
    "--dilate-um",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Optional dilation radius in microns applied to the selected cell mask in atlas space "
        "(keeps any cell within this distance of the selected cells; uses obsm[coords_key])."
    ),
)
@click.option("--coords-key", default="spatial_ccf", show_default=True, help="obsm key containing atlas-space coords.")
@click.option(
    "--coords-units",
    type=click.Choice(["px", "um", "mm"], case_sensitive=False),
    default="px",
    show_default=True,
    help="Units of obsm[coords_key] (used for dilation distance calculations).",
)
@click.option(
    "--coords-space",
    type=click.Choice(["crop", "full"], case_sensitive=False),
    default="crop",
    show_default=True,
    help="Whether obsm[coords_key] is relative to atlas crop bbox (crop) or full atlas slice (full).",
)
@click.option(
    "--coords-spacing-um",
    type=float,
    default=None,
    show_default=False,
    help="Pixel size in microns if --coords-units=px (default: inferred from adata.uns['ccf']['atlas_voxel_um']).",
)
@click.option(
    "--spatial-order",
    type=click.Choice(["xy", "yx"], case_sensitive=False),
    default="xy",
    show_default=True,
    help="Order of columns in obsm[coords_key].",
)
@click.option(
    "--filter-roi/--no-filter-roi",
    default=True,
    show_default=True,
    help="Restrict selection to observations with adata.obs[roi_col] == ROI.",
)
@click.option("--roi-col", default="roi", show_default=True, help="obs column name used for ROI filtering.")
@click.option(
    "--skip-missing/--no-skip-missing",
    default=True,
    show_default=True,
    help="Skip ROIs that are missing the warped input h5ad under ccf-transforms/<roi>/.",
)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Overwrite output_h5ad if it exists.")
@click.option(
    "--keep-t-mask/--no-keep-t-mask",
    default=False,
    show_default=True,
    help="Preserve existing t-axis endpoints JSON begin/end values, even when --overwrite is set.",
)
@click.option("--qc-plot/--no-qc-plot", default=True, show_default=True, help="Write a QC PNG overlay plot.")
@click.option(
    "--qc-plot-png",
    default=None,
    show_default=False,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Optional output PNG path for the QC overlay (default: derived from output_h5ad).",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable verbose logging to <workspace>/analysis/logs/.",
)
def main(  # noqa: PLR0913
    workspace: Path,
    rois: tuple[str, ...],
    *,
    h5ad_name: str | None,
    out_name: str | None,
    ccf_col: str,
    ccf_adjusted_col: str,
    terms: tuple[str, ...],
    imagej_roi: bool,
    imagej_roi_path: Path | None,
    ignore_imagej_roi: bool,
    run_dirname: str,
    imagej_coords_key: str,
    imagej_input_space: str,
    imagej_input_units: str,
    imagej_target_spacing_um: float,
    stitch_codebook: str,
    kind: str,
    match: str,
    combine: str,
    invert: bool,
    ccf_obsm_key: str,
    dilate_um: float,
    coords_key: str,
    coords_units: str,
    coords_space: str,
    coords_spacing_um: float | None,
    spatial_order: str,
    filter_roi: bool,
    roi_col: str,
    skip_missing: bool,
    overwrite: bool,
    keep_t_mask: bool,
    qc_plot: bool,
    qc_plot_png: Path | None,
    debug: bool,
) -> None:
    ws = Workspace(workspace)
    try:
        resolved_rois = ws.resolve_rois(rois)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="rois") from exc

    is_batch = len(resolved_rois) > 1

    if qc_plot_png is not None and len(resolved_rois) > 1:
        raise click.BadParameter("--qc-plot-png cannot be used when running multiple ROIs.")

    ccf_col = str(ccf_col).strip()
    if not ccf_col:
        raise click.BadParameter("--ccf-col must be non-empty.")
    ccf_adjusted_col = str(ccf_adjusted_col).strip()
    if not ccf_adjusted_col:
        raise click.BadParameter("--ccf-adjusted-col must be non-empty.")

    for roi_resolved in resolved_rois:
        try:
            input_h5ad_candidate = _input_h5ad_candidate(ws, roi=str(roi_resolved), h5ad_name=h5ad_name)
            if not input_h5ad_candidate.exists():
                if skip_missing:
                    click.echo(f"Skipping roi={roi_resolved!r} (missing warped h5ad): {input_h5ad_candidate}")
                    continue
                input_h5ad = _resolve_input_h5ad(ws, roi=str(roi_resolved), h5ad_name=h5ad_name)
            else:
                input_h5ad = input_h5ad_candidate
            roi_resolved, output_h5ad, plot_png_default = _default_outputs(
                ws,
                roi=str(roi_resolved),
                input_h5ad=input_h5ad,
                out_name=out_name,
            )
            plot_png = qc_plot_png if qc_plot_png is not None else plot_png_default

            output_h5ad.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite:
                existing: list[Path] = []
                if output_h5ad.exists():
                    existing.append(output_h5ad)
                if qc_plot and plot_png.exists():
                    existing.append(plot_png)
                if existing:
                    click.echo(
                        "Skipping (output already exists; pass --overwrite to replace): " + ", ".join(str(p) for p in existing)
                    )
                    continue

            try:
                setup_cli_logging(
                    workspace,
                    component="ccf.filter_h5ad_ccf",
                    file=f"filter-h5ad-ccf-{roi_resolved}",
                    debug=debug,
                    extra={"roi": str(roi_resolved)},
                )
            except PermissionError as exc:
                click.echo(
                    f"Warning: cannot write logs under {workspace}/analysis/logs (permission denied); continuing without file logging. ({exc})",
                    err=True,
                )

            adata = ad.read_h5ad(input_h5ad)
            n_total = int(adata.n_obs)

            roi_mask: np.ndarray | None = None
            adata_work = adata
            if filter_roi:
                if roi_col not in adata.obs.columns:
                    raise click.ClickException(f"Missing obs column {roi_col!r} in {input_h5ad}.")
                roi_mask = (adata.obs[roi_col].astype(str) == str(roi_resolved)).to_numpy()
                adata_work = adata[roi_mask].copy()
                click.echo(
                    f"Scoped ROI for selection: {adata_work.n_obs}/{n_total} obs (roi_col={roi_col!r}, roi={roi_resolved!r})"
                )

            order_t = cast(SpatialOrder, str(spatial_order).lower())
            qc_units: str
            qc_coords: np.ndarray
            title: str

            roi_path: Path | None = None
            mask_edit_dir = ws.ccf_transforms(str(roi_resolved)) / str(run_dirname) / "mask_edit"
            if not ignore_imagej_roi:
                if imagej_roi_path is not None:
                    roi_path = imagej_roi_path
                elif imagej_roi:
                    if mask_edit_dir.exists():
                        roi_path = _find_imagej_roi_file(mask_edit_dir)
                    if roi_path is None:
                        msg = f"--imagej-roi was requested but no ROI was found under {mask_edit_dir}."
                        if is_batch:
                            click.echo(f"Warning: {msg} Falling back to --term filtering for roi={roi_resolved!r}.", err=True)
                        else:
                            raise click.ClickException(msg)

            imagej_labels: np.ndarray | None = None
            if roi_path is not None:
                click.echo(f"Using ImageJ ROI override: {roi_path}")
                imagej_space_t = cast(InputSpace, str(imagej_input_space).lower())
                imagej_units_t = cast(CoordUnits, str(imagej_input_units).lower())
                mask, qc_coords, imagej_labels = _mask_from_imagej_roi(
                    ws=ws,
                    roi=str(roi_resolved),
                    adata=adata_work,
                    roi_path=roi_path,
                    coords_key=str(imagej_coords_key),
                    input_space=imagej_space_t,
                    input_units=imagej_units_t,
                    spatial_order=order_t,
                    stitch_codebook=str(stitch_codebook),
                    target_spacing_um=float(imagej_target_spacing_um),
                )
                qc_units = "thumb_px"
                title = (
                    f"ImageJ ROI selection overlay | selected={int(mask.sum())}/{int(adata_work.n_obs)} | roi={roi_resolved!r}"
                )
            else:
                if not terms:
                    raise click.ClickException(
                        "At least one --term is required unless an ImageJ ROI is available (or enabled via --imagej-roi/--imagej-roi-path)."
                    )

                coords_units_t = cast(CoordUnits, str(coords_units).lower())
                coords_space_t = cast(CoordSpace, str(coords_space).lower())
                _ensure_ccf_obsm(
                    adata_work,
                    input_h5ad=input_h5ad,
                    ccf_obsm_key=ccf_obsm_key,
                    coords_key=coords_key,
                    coords_units=coords_units_t,
                    coords_space=coords_space_t,
                    spatial_order=order_t,
                    roi=str(roi_resolved),
                    workspace=workspace,
                )

                kind_t = cast(CCFTermKind, str(kind).lower())
                match_t = cast(MatchMode, str(match).lower())
                combine_t = cast(CombineMode, str(combine).lower())

                n_work = int(adata_work.n_obs)
                term_masks: list[np.ndarray] = []
                term_labels = np.empty(n_work, dtype=object)
                term_labels[:] = ""
                if combine_t == "all":
                    for t in terms:
                        term_masks.append(
                            _mask_for_term(adata_work, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)
                        )
                    mask = np.ones(n_work, dtype=bool) if term_masks else np.zeros(n_work, dtype=bool)
                    for m in term_masks:
                        mask &= m
                    if term_masks:
                        joined = "|".join(str(t) for t in terms)
                        term_labels[mask] = joined
                else:
                    for t in terms:
                        m = _mask_for_term(adata_work, str(t), kind=kind_t, match=match_t, obsm_key=ccf_obsm_key)
                        term_masks.append(m)
                        new = m & (term_labels == "")
                        if np.any(new):
                            term_labels[new] = str(t)
                    mask = np.zeros(n_work, dtype=bool)
                    for m in term_masks:
                        mask |= m

                dilate = float(dilate_um)
                if dilate > 0:
                    if coords_key not in adata_work.obsm:
                        raise click.ClickException(
                            f"Missing adata.obsm[{coords_key!r}] in {input_h5ad} (needed for --dilate-um)."
                        )
                    coords = np.asarray(adata_work.obsm[coords_key])
                    if coords.ndim != 2 or coords.shape[1] != 2:
                        raise click.ClickException(
                            f"Expected obsm[{coords_key!r}] to have shape (N,2), got {coords.shape}."
                        )
                    if order_t == "yx":
                        coords = coords[:, ::-1]

                    spacing_um = coords_spacing_um
                    if coords_units_t == "px" and spacing_um is None:
                        ccf_meta = adata_work.uns.get("ccf", {})
                        ccf_atlas_meta = adata_work.uns.get("ccf_atlas", {})
                        if isinstance(ccf_meta, dict) and "atlas_voxel_um" in ccf_meta:
                            spacing_um = float(ccf_meta["atlas_voxel_um"])
                        elif isinstance(ccf_atlas_meta, dict) and "atlas_voxel_um" in ccf_atlas_meta:
                            spacing_um = float(ccf_atlas_meta["atlas_voxel_um"])
                    coords_um = _coords_to_um(coords, units=coords_units_t, spacing_um=spacing_um)
                    mask = _dilate_mask_by_radius_um(coords_um, base_mask=mask, radius_um=dilate)
                    if term_masks:
                        joined = "|".join(str(t) for t in terms)
                        new = mask & (term_labels == "")
                        if np.any(new):
                            term_labels[new] = joined

                if coords_key not in adata_work.obsm:
                    raise click.ClickException(f"Missing adata.obsm[{coords_key!r}] in {input_h5ad} (needed for QC plot).")
                qc_coords = np.asarray(adata_work.obsm[coords_key])
                if qc_coords.ndim != 2 or qc_coords.shape[1] != 2:
                    raise click.ClickException(f"Expected obsm[{coords_key!r}] to have shape (N,2), got {qc_coords.shape}.")
                if order_t == "yx":
                    qc_coords = qc_coords[:, ::-1]

                qc_units = str(coords_units_t)
                title = (
                    f"CCF selection overlay | selected={int(mask.sum())}/{int(adata_work.n_obs)} | "
                    f"match={match_t}, combine={combine_t}, invert={invert}, dilate_um={float(dilate_um)}"
                )

            if invert:
                mask = ~mask

            labels = imagej_labels if imagej_labels is not None else term_labels
            if invert:
                criteria = roi_path.name if roi_path is not None else "|".join(str(t) for t in terms)
                invert_label = f"not({criteria})"
                labels = np.where(mask, np.where(labels != "", labels, invert_label), "")
            else:
                labels = np.where(mask, labels, "")

            labels_full = np.empty(n_total, dtype=object)
            labels_full[:] = ""
            if roi_mask is not None:
                labels_full[roi_mask] = labels
            else:
                labels_full = labels

            if qc_plot:
                _write_qc_mask_overlay_plot(
                    coords_xy=qc_coords,
                    keep_mask=mask,
                    output_png=plot_png,
                    title=title,
                    units=qc_units,
                )
                click.echo(f"Wrote QC plot: {plot_png}")

            t_axis_endpoints_payload: dict[str, object] | None = None
            if roi_path is not None:
                t_axis_json_path: Path | None = None
                try:
                    user_overlay_outputs = _write_ccf_user_mask_overlay_png(
                        ws=ws,
                        roi=str(roi_resolved),
                        run_dirname=str(run_dirname),
                        roi_path=roi_path,
                        imagej_target_spacing_um=float(imagej_target_spacing_um),
                        overwrite=bool(overwrite),
                    )
                except (FileNotFoundError, KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                    click.echo(
                        f"Warning: skipping CCF user-mask overlay for roi={roi_resolved!r}: {exc}",
                        err=True,
                    )
                else:
                    if user_overlay_outputs is not None:
                        overlay_png = user_overlay_outputs.get("overlay_png")
                        if overlay_png is not None:
                            click.echo(f"Wrote CCF user-mask overlay: {overlay_png}")
                        t_axis_json = user_overlay_outputs.get("t_axis_endpoints_json")
                        if t_axis_json is not None:
                            t_axis_json_path = Path(t_axis_json)
                            click.echo(f"Wrote CCF t-axis endpoints JSON: {t_axis_json}")
                if t_axis_json_path is None:
                    candidate = (
                        ws.ccf_transforms(str(roi_resolved))
                        / str(run_dirname)
                        / "similarity_plus_syn_qc_zoom_masked_with_user_mask_t_axis_endpoints.json"
                    )
                    if candidate.exists():
                        t_axis_json_path = candidate
                if t_axis_json_path is not None and t_axis_json_path.exists():
                    t_axis_endpoints_payload = _load_json_object(t_axis_json_path)

            out = adata.copy()
            out.obs[ccf_col] = pd.Series([""] * n_total, index=out.obs_names)
            out.obs[ccf_adjusted_col] = pd.Series([""] * n_total, index=out.obs_names)
            if imagej_labels is not None:
                out.obs[ccf_adjusted_col] = pd.Series(labels_full, index=out.obs_names)
            else:
                out.obs[ccf_col] = pd.Series(labels_full, index=out.obs_names)
            if t_axis_endpoints_payload is not None:
                payload_for_uns = dict(t_axis_endpoints_payload)
                payload_for_uns.pop("masks", None)
                out.uns["t_all_mapping"] = payload_for_uns
            out.write_h5ad(output_h5ad)
            n_labeled = int(np.sum(labels_full != ""))
            target_col = ccf_adjusted_col if imagej_labels is not None else ccf_col
            click.echo(f"Wrote: {output_h5ad} (annotated {n_labeled}/{out.n_obs} obs in {target_col!r})")


        except Exception as exc:
            if is_batch:
                click.echo(f"Warning: error processing roi={roi_resolved!r}: {exc}", err=True)
                continue
            raise

if __name__ == "__main__":
    main()
