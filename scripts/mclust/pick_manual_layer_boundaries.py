#%%
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anndata as ad
os.environ["MATPLOTLIBRC"] = os.devnull
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.ndimage import label as ndi_label
from skimage.color import label2rgb
from skimage.draw import line
from skimage.morphology import disk

# %% [markdown]
# # Pick manual layer boundaries from pooled mclust labels
#
# This script loads `/fast2/cs_outputs/all.h5ad`, joins the STAGATE mclust sweep parquet,
# and walks through each `dataset x roi x ccf_adjusted` job.
#
# The edited interfaces are fixed and ordered:
# `46-2`, `2-3`, `3-7`, `7-5`, `5-1`.
#
# Run cells sequentially. Outputs are written under `/fast2/cs_outputs/all.mclust_manual_boundaries`.

# %%
# Use widget backend for VS Code interactive mode.
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

# %% [markdown]
# ## Config (EDIT THESE)
#
# Update paths and knobs here before running the workflow cells below.

# %%
INPUT_H5AD = Path("/fast2/cs_outputs/all.h5ad")
STAGATE_H5AD = Path("/fast2/cs_outputs/all.stagate.h5ad")
LABELS_PARQUET = Path("/fast2/cs_outputs/all.stagate.mclust_sweep.parquet")
OUTPUT_ROOT = Path("/fast2/cs_outputs/all.mclust_manual_boundaries")

LABEL_KEY = "mclust_k7"
BOUNDARY_ORDER = ("46-2", "2-3", "3-7", "7-5", "5-1")
BOUNDARY_COLORS = {
    "46-2": "#ff33aa",
    "2-3": "#00bcd4",
    "3-7": "#7cb342",
    "7-5": "#ff9800",
    "5-1": "#8e44ad",
}
MANUAL_LAYER_ORDER = ("46", "2", "3", "7", "5", "1")
POOLED_LAYER_NAME = "46"

SKIP_EXISTING_JSON = False
PLOT_MAX_POINTS = 120_000
POINT_SIZE = 4
POINT_ALPHA = 0.55
PIXEL_SIZE = 12.0
SUPPORT_RADIUS = 2
MIN_COMPONENT_PIXELS = 200
SPLINE_MIN_POINTS = 4
CURVE_MAX_SAMPLES = 1_500
UI_FIGSIZE = (16, 12)
QC_DPI = 180
MONTAGE_NCOLS = 4
MONTAGE_MAX_POINTS_PER_UNIT = 30_000
MONTAGE_PANEL_SIZE = (4.2, 4.2)
MONTAGE_DPI = 120
MONTAGE_POINT_SIZE = 1.0
MONTAGE_POINT_ALPHA = 0.55

adata: ad.AnnData | None = None
jobs: list[JobSpec] = []
i = 0


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    roi_group: str
    ccf_adjusted: str
    roi_members: tuple[str, ...]
    n_obs: int


@dataclass(frozen=True)
class AssignmentResult:
    manual_layers: np.ndarray
    point_components: np.ndarray
    component_order: tuple[int, ...]
    component_layer_map: dict[int, str]
    components_raster: np.ndarray
    support_mask: np.ndarray
    barrier_mask: np.ndarray
    raster_origin_xy: tuple[float, float]
    pixel_size: float


def _valid_group_value(value: object) -> bool:
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none"}


def _make_unique_obs_index(obs_index: pd.Index) -> pd.Index:
    values = obs_index.astype(str).tolist()
    counts: dict[str, int] = {}
    unique_values: list[str] = []
    for value in values:
        count = counts.get(value, 0)
        unique_values.append(value if count == 0 else f"{value}-{count}")
        counts[value] = count + 1
    return pd.Index(unique_values, name=obs_index.name)


def _strip_unique_suffix(obs_index: pd.Index) -> pd.Index:
    return pd.Index(obs_index.astype(str).str.replace(r"-\d+$", "", regex=True), name=obs_index.name)


def _label_join_key(obs: pd.DataFrame, *, raw_index: pd.Index | None = None) -> pd.Index:
    if "dataset" not in obs.columns:
        raise KeyError("Expected `dataset` in obs before joining mclust labels.")
    index = obs.index if raw_index is None else raw_index
    return pd.Index(obs["dataset"].astype(str) + "_" + index.astype(str), name="cell_id")


def load_adata_with_labels(
    in_h5ad: Path = INPUT_H5AD,
    stagate_h5ad: Path = STAGATE_H5AD,
    labels_parquet: Path = LABELS_PARQUET,
) -> ad.AnnData:
    adata = ad.read_h5ad(in_h5ad)
    labels = pd.read_parquet(labels_parquet)
    join_key = _label_join_key(adata.obs)
    if not join_key.is_unique:
        raise ValueError("Constructed label join key is not unique.")
    stagate = ad.read_h5ad(stagate_h5ad, backed="r")
    try:
        stagate_obs = stagate.obs[["dataset"]].copy()
    finally:
        stagate.file.close()
    labels.index = _label_join_key(stagate_obs, raw_index=_strip_unique_suffix(stagate_obs.index))
    if not labels.index.is_unique:
        raise ValueError("Constructed parquet join key is not unique.")
    for col in labels.columns:
        labels[col] = labels[col].astype("category")
        adata.obs.drop(columns=[col], inplace=True, errors="ignore")
    adata.obs = adata.obs.join(labels, how="left", on=join_key)
    adata.obs["obs_ix"] = np.arange(adata.n_obs, dtype=np.int64)
    adata.obs["roi_group"] = adata.obs["roi"].astype(str)
    if LABEL_KEY not in adata.obs.columns:
        raise KeyError(f"Missing required obs column after parquet join: {LABEL_KEY}")
    return adata


def build_jobs_from_obs(obs: pd.DataFrame) -> list[JobSpec]:
    if "roi_group" not in obs.columns:
        raise KeyError("Expected `roi_group` in obs. Run load_adata_with_labels first.")

    valid = obs["ccf_adjusted"].map(_valid_group_value)
    grouped = (
        obs.loc[valid, ["dataset", "roi_group", "ccf_adjusted", "roi"]]
        .astype(str)
        .groupby(["dataset", "roi_group", "ccf_adjusted"], observed=True)
    )

    jobs: list[JobSpec] = []
    for (dataset, roi_group, ccf_adjusted), frame in grouped:
        roi_members = tuple(sorted(set(frame["roi"].tolist())))
        jobs.append(
            JobSpec(
                dataset=str(dataset),
                roi_group=str(roi_group),
                ccf_adjusted=str(ccf_adjusted),
                roi_members=roi_members,
                n_obs=int(frame.shape[0]),
            )
        )
    return sorted(jobs, key=lambda job: (job.dataset, job.roi_group, job.ccf_adjusted))


def subset_job_adata(adata: ad.AnnData, job: JobSpec) -> ad.AnnData:
    obs = adata.obs
    mask = (
        (obs["dataset"].astype(str) == job.dataset)
        & (obs["roi_group"].astype(str) == job.roi_group)
        & (obs["ccf_adjusted"].astype(str) == job.ccf_adjusted)
    )
    if not bool(mask.any()):
        raise ValueError(
            "No observations matched "
            f"dataset={job.dataset!r}, roi_group={job.roi_group!r}, ccf_adjusted={job.ccf_adjusted!r}."
        )
    return adata[mask.to_numpy()].copy()


def get_spatial_xy(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"Unexpected adata.obsm['spatial'] shape: {xy.shape}")
        xy = xy[:, :2]
    else:
        if not {"x", "y"}.issubset(adata.obs.columns):
            raise ValueError("Expected either adata.obsm['spatial'] or obs columns 'x' and 'y'.")
        xy = adata.obs[["x", "y"]].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(xy).all():
        raise ValueError("Coordinate array contains non-finite values.")
    return xy


def _job_output_stem(job: JobSpec) -> str:
    return f"{job.dataset}__roi-{job.roi_group}__ccf-{job.ccf_adjusted}"


def _job_paths(output_root: Path, job: JobSpec) -> dict[str, Path]:
    stem = _job_output_stem(job)
    return {
        "json": output_root / f"{stem}.json",
        "assignments": output_root / f"{stem}.assignments.parquet",
        "qc_png": output_root / f"{stem}.qc.png",
    }


def empty_boundaries() -> dict[str, list[tuple[float, float]]]:
    return {name: [] for name in BOUNDARY_ORDER}


def load_boundary_state(path: Path) -> dict[str, list[tuple[float, float]]]:
    if not path.exists():
        return empty_boundaries()

    payload = json.loads(path.read_text())
    boundaries = empty_boundaries()
    raw = payload.get("boundaries", {})
    for name in BOUNDARY_ORDER:
        points = raw.get(name, [])
        boundaries[name] = [(float(x), float(y)) for x, y in points]
    return boundaries


def save_boundary_state(
    *,
    path: Path,
    job: JobSpec,
    boundaries: dict[str, list[tuple[float, float]]],
    completed: bool,
    input_h5ad: Path = INPUT_H5AD,
    labels_parquet: Path = LABELS_PARQUET,
) -> None:
    payload = {
        "source_h5ad": str(input_h5ad),
        "source_parquet": str(labels_parquet),
        "label_key": LABEL_KEY,
        "dataset": job.dataset,
        "roi_group": job.roi_group,
        "roi_members": list(job.roi_members),
        "ccf_adjusted": job.ccf_adjusted,
        "boundary_order": list(BOUNDARY_ORDER),
        "manual_layer_order": list(MANUAL_LAYER_ORDER),
        "pixel_size": float(PIXEL_SIZE),
        "support_radius": int(SUPPORT_RADIUS),
        "min_component_pixels": int(MIN_COMPONENT_PIXELS),
        "boundaries": {name: [[float(x), float(y)] for x, y in boundaries[name]] for name in BOUNDARY_ORDER},
        "completed": bool(completed),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _piecewise_resample(points: np.ndarray, n_samples: int) -> np.ndarray:
    if points.shape[0] == 1:
        return points.astype(np.float32, copy=False)

    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return points.astype(np.float32, copy=False)

    targets = np.linspace(0.0, total, num=int(n_samples), dtype=np.float64)
    out = np.empty((targets.size, 2), dtype=np.float64)
    for i, target in enumerate(targets):
        idx = min(int(np.searchsorted(cumulative, target, side="right") - 1), points.shape[0] - 2)
        start = points[idx]
        end = points[idx + 1]
        length = seg_lengths[idx]
        if length <= 0:
            out[i] = start
            continue
        frac = (target - cumulative[idx]) / length
        out[i] = start + frac * (end - start)
    return out.astype(np.float32, copy=False)


def densify_open_curve(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected point array with shape (n, 2), got {points.shape}")
    if points.shape[0] <= 1:
        return points.astype(np.float32, copy=False)

    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    total = float(seg_lengths.sum())
    sample_count = max(64, min(CURVE_MAX_SAMPLES, int(np.ceil(total / max(PIXEL_SIZE, 1.0))) * 8))

    if points.shape[0] < SPLINE_MIN_POINTS or np.count_nonzero(seg_lengths > 0) < 2:
        return _piecewise_resample(points, n_samples=sample_count)

    u = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if float(u[-1]) <= 0:
        return points.astype(np.float32, copy=False)
    u = u / float(u[-1])
    # Use one shape-preserving spline for both rendering and assignment so the
    # point-in-polygon geometry matches what the user sees after closing a loop.
    x_spline = PchipInterpolator(u, points[:, 0])
    y_spline = PchipInterpolator(u, points[:, 1])
    uu = np.linspace(0.0, 1.0, num=sample_count, dtype=np.float64)
    return np.column_stack([x_spline(uu), y_spline(uu)]).astype(np.float32, copy=False)


def _xy_to_rc_float(xy: np.ndarray, *, x0: float, y0: float, pixel_size: float) -> np.ndarray:
    return np.column_stack(
        [
            (xy[:, 1] - y0) / pixel_size,
            (xy[:, 0] - x0) / pixel_size,
        ]
    ).astype(np.float32, copy=False)


def _xy_to_rc(xy: np.ndarray, *, x0: float, y0: float, pixel_size: float) -> np.ndarray:
    return np.rint(_xy_to_rc_float(xy, x0=x0, y0=y0, pixel_size=pixel_size)).astype(np.int32, copy=False)


def _rasterize_points(
    xy: np.ndarray,
    *,
    pixel_size: float,
    padding_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    x0 = float(xy[:, 0].min()) - float(padding_pixels) * pixel_size
    y0 = float(xy[:, 1].min()) - float(padding_pixels) * pixel_size
    point_rc = _xy_to_rc(xy, x0=x0, y0=y0, pixel_size=pixel_size)
    height = int(point_rc[:, 0].max()) + padding_pixels + 1
    width = int(point_rc[:, 1].max()) + padding_pixels + 1
    occupied = np.zeros((height, width), dtype=bool)
    occupied[point_rc[:, 0], point_rc[:, 1]] = True
    return point_rc, occupied, (x0, y0)


def _make_support_mask(occupied: np.ndarray, radius: int) -> np.ndarray:
    support = binary_fill_holes(occupied)
    if radius > 0:
        support = binary_dilation(support, structure=disk(radius), iterations=4)
    support = binary_fill_holes(support)
    return support.astype(bool, copy=False)


def _ray_to_bbox(point: np.ndarray, direction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    direction = direction.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return point.astype(np.float32, copy=False)
    direction = direction / norm

    candidates: list[float] = []
    max_row = float(shape[0] - 1)
    max_col = float(shape[1] - 1)
    if abs(direction[0]) > 1e-8:
        target_row = max_row if direction[0] > 0 else 0.0
        candidates.append((target_row - float(point[0])) / float(direction[0]))
    if abs(direction[1]) > 1e-8:
        target_col = max_col if direction[1] > 0 else 0.0
        candidates.append((target_col - float(point[1])) / float(direction[1]))

    positive = [value for value in candidates if value > 0]
    if not positive:
        return point.astype(np.float32, copy=False)

    t = min(positive)
    target = point.astype(np.float64, copy=False) + direction * t
    target[0] = np.clip(target[0], 0.0, max_row)
    target[1] = np.clip(target[1], 0.0, max_col)
    return target.astype(np.float32, copy=False)


def _burn_line_pixels(mask: np.ndarray, start_rc: np.ndarray, end_rc: np.ndarray) -> None:
    rr, cc = line(int(round(float(start_rc[0]))), int(round(float(start_rc[1]))), int(round(float(end_rc[0]))), int(round(float(end_rc[1]))))
    keep = (rr >= 0) & (cc >= 0) & (rr < mask.shape[0]) & (cc < mask.shape[1])
    mask[rr[keep], cc[keep]] = True


def _boundary_is_closed_array(points: np.ndarray, *, atol: float = 1e-6) -> bool:
    if points.shape[0] < 4:
        return False
    return bool(np.allclose(points[0], points[-1], atol=atol))


def _boundary_pixel_mask(
    xy_points: np.ndarray,
    *,
    shape: tuple[int, int],
    x0: float,
    y0: float,
    pixel_size: float,
) -> np.ndarray:
    if xy_points.shape[0] < 2:
        raise ValueError("Each boundary needs at least two anchor points.")

    dense_xy = densify_open_curve(xy_points)
    rc = _xy_to_rc_float(dense_xy, x0=x0, y0=y0, pixel_size=pixel_size)
    if rc.shape[0] < 2:
        raise ValueError("Boundary densification produced fewer than two points.")

    if _boundary_is_closed_array(xy_points):
        rc_full = rc.astype(np.float32, copy=False)
    else:
        anchor_rc = _xy_to_rc_float(xy_points, x0=x0, y0=y0, pixel_size=pixel_size)
        start_tangent = anchor_rc[0] - anchor_rc[1]
        end_tangent = anchor_rc[-1] - anchor_rc[-2]
        start_edge = _ray_to_bbox(rc[0], start_tangent, shape=shape)
        end_edge = _ray_to_bbox(rc[-1], end_tangent, shape=shape)
        rc_full = np.vstack([start_edge, rc, end_edge]).astype(np.float32, copy=False)

    mask = np.zeros(shape, dtype=bool)
    for start, end in zip(rc_full[:-1], rc_full[1:], strict=True):
        _burn_line_pixels(mask, start, end)
    return mask


def _extended_boundary_curve_rc(
    xy_points: np.ndarray,
    *,
    shape: tuple[int, int],
    x0: float,
    y0: float,
    pixel_size: float,
) -> np.ndarray:
    if xy_points.shape[0] < 2:
        raise ValueError("Each boundary needs at least two anchor points.")

    dense_xy = densify_open_curve(xy_points)
    rc = _xy_to_rc_float(dense_xy, x0=x0, y0=y0, pixel_size=pixel_size)
    if rc.shape[0] < 2:
        raise ValueError("Boundary densification produced fewer than two points.")
    if _boundary_is_closed_array(xy_points):
        return rc.astype(np.float32, copy=False)

    anchor_rc = _xy_to_rc_float(xy_points, x0=x0, y0=y0, pixel_size=pixel_size)
    start_tangent = anchor_rc[0] - anchor_rc[1]
    end_tangent = anchor_rc[-1] - anchor_rc[-2]
    start_edge = _ray_to_bbox(rc[0], start_tangent, shape=shape)
    end_edge = _ray_to_bbox(rc[-1], end_tangent, shape=shape)
    return np.vstack([start_edge, rc, end_edge]).astype(np.float32, copy=False)


def _relabel_components(mask: np.ndarray) -> np.ndarray:
    relabeled, _ = ndi_label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    return relabeled.astype(np.int32, copy=False)


def _merge_small_components(components: np.ndarray, support: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return _relabel_components(components > 0)

    merged = components.copy()
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    while True:
        component_ids, counts = np.unique(merged[merged > 0], return_counts=True)
        small_ids = [int(component_id) for component_id, count in zip(component_ids, counts, strict=True) if int(count) < int(min_size)]
        if not small_ids:
            break

        changed = False
        for component_id in small_ids:
            component_mask = merged == component_id
            if not bool(component_mask.any()):
                continue
            border = binary_dilation(component_mask, structure=structure) & ~component_mask & support
            neighbors = merged[border]
            neighbors = neighbors[neighbors > 0]
            if neighbors.size == 0:
                continue
            values, neighbor_counts = np.unique(neighbors, return_counts=True)
            merged[component_mask] = int(values[np.argmax(neighbor_counts)])
            changed = True
        if not changed:
            break
        merged = _relabel_components(merged > 0)

    return merged.astype(np.int32, copy=False)


def _nearest_filled_labels(labels: np.ndarray) -> np.ndarray:
    valid = labels > 0
    if not bool(valid.any()):
        raise ValueError("No labeled pixels are available to fill the raster.")
    _, nearest = distance_transform_edt(~valid, return_indices=True)
    return labels[nearest[0], nearest[1]].astype(np.int32, copy=False)


def _sample_point_components(components: np.ndarray, point_rc: np.ndarray) -> np.ndarray:
    filled = _nearest_filled_labels(components)
    rows = np.clip(point_rc[:, 0], 0, components.shape[0] - 1)
    cols = np.clip(point_rc[:, 1], 0, components.shape[1] - 1)
    return filled[rows, cols].astype(np.int32, copy=False)


def _component_majority_score(component_labels: np.ndarray, expected_label: str) -> float:
    if component_labels.size == 0:
        return -np.inf
    if expected_label == POOLED_LAYER_NAME:
        return float(np.mean(np.isin(component_labels, ["4", "6"])))
    return float(np.mean(component_labels == expected_label))


def _collapse_original_labels(labels: np.ndarray) -> np.ndarray:
    collapsed = labels.astype(str, copy=True)
    collapsed[np.isin(collapsed, ["4", "6"])] = POOLED_LAYER_NAME
    return collapsed.astype(object, copy=False)


def _perimeter_t(point_rc: np.ndarray, *, shape: tuple[int, int]) -> float:
    max_row = float(shape[0] - 1)
    max_col = float(shape[1] - 1)
    row = float(np.clip(point_rc[0], 0.0, max_row))
    col = float(np.clip(point_rc[1], 0.0, max_col))
    distances = np.array([abs(row), abs(max_col - col), abs(max_row - row), abs(col)], dtype=np.float64)
    edge = int(np.argmin(distances))
    if edge == 0:
        return col
    if edge == 1:
        return max_col + row
    if edge == 2:
        return max_col + max_row + (max_col - col)
    return max_col + max_row + max_col + (max_row - row)


def _rectangle_path_between(
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    *,
    shape: tuple[int, int],
    clockwise: bool,
) -> np.ndarray:
    if not clockwise:
        return _rectangle_path_between(end_rc, start_rc, shape=shape, clockwise=True)[::-1]

    max_row = float(shape[0] - 1)
    max_col = float(shape[1] - 1)
    total = 2.0 * (max_row + max_col)
    corner_t = np.array([0.0, max_col, max_col + max_row, 2.0 * max_col + max_row], dtype=np.float64)
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, max_col],
            [max_row, max_col],
            [max_row, 0.0],
        ],
        dtype=np.float32,
    )

    t_start = _perimeter_t(start_rc, shape=shape)
    t_end = _perimeter_t(end_rc, shape=shape)
    if t_end < t_start:
        t_end += total

    points = [np.asarray(start_rc, dtype=np.float32)]
    for idx, value in enumerate(corner_t.tolist()):
        adjusted = value
        while adjusted <= t_start:
            adjusted += total
        if adjusted < t_end:
            points.append(corners[idx])
    points.append(np.asarray(end_rc, dtype=np.float32))
    return np.asarray(points, dtype=np.float32)


def _polygon_contains_points(polygon_rc: np.ndarray, point_rc: np.ndarray) -> np.ndarray:
    polygon_xy = np.column_stack([polygon_rc[:, 1], polygon_rc[:, 0]])
    if not np.allclose(polygon_xy[0], polygon_xy[-1]):
        polygon_xy = np.vstack([polygon_xy, polygon_xy[0]])
    points_xy = np.column_stack([point_rc[:, 1], point_rc[:, 0]])
    return MplPath(polygon_xy).contains_points(points_xy, radius=1e-6)


def _boundary_polygon_candidates(curve_rc: np.ndarray, *, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if _boundary_is_closed_array(curve_rc):
        polygon = curve_rc.astype(np.float32, copy=False)
        return polygon, polygon
    clockwise = _rectangle_path_between(curve_rc[-1], curve_rc[0], shape=shape, clockwise=True)
    counter_clockwise = _rectangle_path_between(curve_rc[-1], curve_rc[0], shape=shape, clockwise=False)
    poly_a = np.vstack([curve_rc, clockwise]).astype(np.float32, copy=False)
    poly_b = np.vstack([curve_rc, counter_clockwise]).astype(np.float32, copy=False)
    return poly_a, poly_b


def _select_downstream_polygon(
    *,
    point_rc: np.ndarray,
    collapsed_labels: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    downstream_labels: tuple[str, ...],
) -> np.ndarray:
    best_polygon: np.ndarray | None = None
    best_score: tuple[float, int, int] | None = None
    downstream_mask = np.isin(collapsed_labels, np.asarray(downstream_labels, dtype=object))

    for polygon in candidates:
        inside = _polygon_contains_points(polygon, point_rc)
        inside_count = int(inside.sum())
        downstream_count = int(np.count_nonzero(inside & downstream_mask))
        purity = float(downstream_count / inside_count) if inside_count > 0 else -1.0
        score = (purity, downstream_count, -inside_count)
        if best_score is None or score > best_score:
            best_score = score
            best_polygon = polygon

    assert best_polygon is not None
    return best_polygon.astype(np.float32, copy=False)


def _rasterize_point_layer_codes(point_rc: np.ndarray, codes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    df = pd.DataFrame(
        {
            "row": point_rc[:, 0].astype(np.int32, copy=False),
            "col": point_rc[:, 1].astype(np.int32, copy=False),
            "code": codes.astype(np.int32, copy=False),
        }
    )
    counts = (
        df.groupby(["row", "col", "code"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["row", "col", "count", "code"], ascending=[True, True, False, True])
        .drop_duplicates(["row", "col"], keep="first")
    )
    raster = np.zeros(shape, dtype=np.int32)
    raster[counts["row"].to_numpy(), counts["col"].to_numpy()] = counts["code"].to_numpy(dtype=np.int32)
    return raster


def assign_manual_layers_from_boundaries(
    *,
    xy: np.ndarray,
    original_labels: np.ndarray,
    boundaries: dict[str, list[tuple[float, float]]],
    pixel_size: float = PIXEL_SIZE,
    support_radius: int = SUPPORT_RADIUS,
    min_component_pixels: int = MIN_COMPONENT_PIXELS,
) -> AssignmentResult:
    missing = [name for name in BOUNDARY_ORDER if len(boundaries.get(name, [])) < 2]
    if missing:
        raise ValueError(f"Missing anchors for boundaries: {', '.join(missing)}")

    point_rc, occupied, (x0, y0) = _rasterize_points(xy, pixel_size=pixel_size)
    support = _make_support_mask(occupied, radius=support_radius)
    point_rc_float = point_rc.astype(np.float32, copy=False)

    boundary_curves = {
        name: _extended_boundary_curve_rc(
            np.asarray(boundaries[name], dtype=np.float32),
            shape=support.shape,
            x0=x0,
            y0=y0,
            pixel_size=pixel_size,
        )
        for name in BOUNDARY_ORDER
    }

    barrier_masks = {
        name: _boundary_pixel_mask(
            np.asarray(boundaries[name], dtype=np.float32),
            shape=support.shape,
            x0=x0,
            y0=y0,
            pixel_size=pixel_size,
        )
        for name in BOUNDARY_ORDER
    }
    barrier_mask = np.zeros_like(support, dtype=bool)
    for mask in barrier_masks.values():
        barrier_mask |= mask

    collapsed_labels = _collapse_original_labels(original_labels.astype(str, copy=False))
    manual_layers = np.full(xy.shape[0], POOLED_LAYER_NAME, dtype=object)

    for idx, boundary_name in enumerate(BOUNDARY_ORDER):
        downstream_labels = tuple(MANUAL_LAYER_ORDER[idx + 1 :])
        polygon = _select_downstream_polygon(
            point_rc=point_rc_float,
            collapsed_labels=collapsed_labels,
            candidates=_boundary_polygon_candidates(boundary_curves[boundary_name], shape=support.shape),
            downstream_labels=downstream_labels,
        )
        inside = _polygon_contains_points(polygon, point_rc_float)
        manual_layers[inside] = boundary_name.split("-")[1]

    point_components = (_manual_layer_codes(manual_layers) + 1).astype(np.int32, copy=False)
    components = _rasterize_point_layer_codes(point_rc=point_rc, codes=point_components, shape=support.shape)
    components = _nearest_filled_labels(components)
    components[~support] = 0
    component_order = tuple(range(1, len(MANUAL_LAYER_ORDER) + 1))
    component_layer_map = {int(i + 1): str(layer_name) for i, layer_name in enumerate(MANUAL_LAYER_ORDER)}

    return AssignmentResult(
        manual_layers=manual_layers,
        point_components=point_components.astype(np.int32, copy=False),
        component_order=component_order,
        component_layer_map=component_layer_map,
        components_raster=components.astype(np.int32, copy=False),
        support_mask=support.astype(bool, copy=False),
        barrier_mask=barrier_mask.astype(bool, copy=False),
        raster_origin_xy=(float(x0), float(y0)),
        pixel_size=float(pixel_size),
    )


def _manual_layer_codes(manual_layers: np.ndarray) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(MANUAL_LAYER_ORDER)}
    return np.array([mapping[str(label)] for label in manual_layers], dtype=np.int32)


def write_assignment_outputs(
    *,
    adata_job: ad.AnnData,
    job: JobSpec,
    result: AssignmentResult,
    paths: dict[str, Path],
    boundaries: dict[str, list[tuple[float, float]]],
) -> None:
    paths["assignments"].parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "obs_ix": adata_job.obs["obs_ix"].to_numpy(dtype=np.int64, copy=False),
            "dataset": adata_job.obs["dataset"].astype(str).to_numpy(),
            "roi": adata_job.obs["roi"].astype(str).to_numpy(),
            "roi_group": np.full(adata_job.n_obs, job.roi_group, dtype=object),
            "ccf_adjusted": adata_job.obs["ccf_adjusted"].astype(str).to_numpy(),
            "manual_layer": pd.Categorical(result.manual_layers, categories=list(MANUAL_LAYER_ORDER), ordered=True),
        }
    )
    frame.to_parquet(paths["assignments"], index=False)

    xy = get_spatial_xy(adata_job)
    original_labels = adata_job.obs[LABEL_KEY].astype(str).to_numpy()
    result_codes = _manual_layer_codes(result.manual_layers)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=QC_DPI)
    cmap = ListedColormap(["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#ffcc66", "#76b041"])
    axes[0].scatter(xy[:, 0], xy[:, 1], c=pd.Categorical(original_labels).codes, cmap=cmap, s=1.5, linewidths=0)
    axes[0].set_title("Current mclust_k7")
    axes[1].scatter(xy[:, 0], xy[:, 1], c=result_codes, cmap="tab10", s=1.5, linewidths=0)
    axes[1].set_title("Manual layer assignment")

    axes[2].imshow(
        label2rgb(result.components_raster, bg_label=0),
        origin="upper",
        interpolation="nearest",
    )
    axes[2].imshow(np.where(result.barrier_mask, 1.0, np.nan), origin="upper", cmap="gray", alpha=0.8)
    axes[2].set_title("Raster compartments + barriers")

    for boundary_name in BOUNDARY_ORDER:
        points = boundaries[boundary_name]
        if len(points) < 2:
            continue
        curve = densify_open_curve(np.asarray(points, dtype=np.float32))
        axes[0].plot(curve[:, 0], curve[:, 1], color="white", linewidth=1.2, alpha=0.9)
        axes[1].plot(curve[:, 0], curve[:, 1], color="black", linewidth=1.0, alpha=0.9)

    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.suptitle(f"{job.dataset} | roi={job.roi_group} | ccf_adjusted={job.ccf_adjusted}")
    fig.tight_layout()
    fig.savefig(paths["qc_png"], bbox_inches="tight")
    plt.close(fig)


def combine_existing_assignment_tables(output_root: Path = OUTPUT_ROOT) -> Path | None:
    paths = sorted(output_root.glob("*.assignments.parquet"))
    if not paths:
        return None
    combined = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    out = output_root / "manual_layers.parquet"
    combined.to_parquet(out, index=False)
    return out


def save_job_state_and_assignment(
    *,
    adata_job: ad.AnnData,
    job: JobSpec,
    paths: dict[str, Path],
    boundaries: dict[str, list[tuple[float, float]]],
) -> tuple[bool, str]:
    save_boundary_state(path=paths["json"], job=job, boundaries=boundaries, completed=False)
    try:
        result = assign_manual_layers_from_boundaries(
            xy=get_spatial_xy(adata_job),
            original_labels=adata_job.obs[LABEL_KEY].astype(str).to_numpy(),
            boundaries=boundaries,
            pixel_size=PIXEL_SIZE,
            support_radius=SUPPORT_RADIUS,
            min_component_pixels=MIN_COMPONENT_PIXELS,
        )
    except Exception as exc:
        return False, f"Saved boundary JSON; assignment failed: {exc}"

    save_boundary_state(path=paths["json"], job=job, boundaries=boundaries, completed=True)
    write_assignment_outputs(adata_job=adata_job, job=job, result=result, paths=paths, boundaries=boundaries)
    combine_existing_assignment_tables(paths["json"].parent)
    return True, "Saved boundary JSON, assignments, QC PNG, and combined parquet."


def save_job_boundary_state(
    *,
    job: JobSpec,
    paths: dict[str, Path],
    boundaries: dict[str, list[tuple[float, float]]],
) -> str:
    save_boundary_state(path=paths["json"], job=job, boundaries=boundaries, completed=False)
    return f"Saved boundary JSON: {paths['json'].name}"


def _insertion_index(points: list[tuple[float, float]], x: float, y: float) -> int:
    if len(points) < 2:
        return len(points)
    anchor_xy = np.asarray(points, dtype=np.float64)
    p = np.array([x, y], dtype=np.float64)
    seg_a = anchor_xy[:-1]
    seg_b = anchor_xy[1:]
    v = seg_b - seg_a
    vv = np.sum(v * v, axis=1)
    vv = np.where(vv > 0, vv, 1.0)
    w = p[None, :] - seg_a
    tau = np.clip(np.sum(w * v, axis=1) / vv, 0.0, 1.0)
    proj = seg_a + tau[:, None] * v
    d2 = np.sum((proj - p[None, :]) ** 2, axis=1)
    return int(np.argmin(d2)) + 1


def _anchor_pick_radius(ax: Any) -> float:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = abs(float(xlim[1] - xlim[0]))
    yspan = abs(float(ylim[1] - ylim[0]))
    return max(1.0, 0.015 * float(np.hypot(xspan, yspan)))


def _nearest_anchor_index(points: list[tuple[float, float]], x: float, y: float, radius: float) -> int | None:
    if not points:
        return None
    pts = np.asarray(points, dtype=np.float64)
    d2 = np.sum((pts - np.array([x, y], dtype=np.float64)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    if float(d2[idx]) <= float(radius) ** 2:
        return idx
    return None


def _boundary_is_closed_points(points: list[tuple[float, float]], *, atol: float = 1e-6) -> bool:
    if len(points) < 4:
        return False
    return bool(np.allclose(np.asarray(points[0], dtype=np.float64), np.asarray(points[-1], dtype=np.float64), atol=atol))


def _close_boundary_points(points: list[tuple[float, float]]) -> None:
    if len(points) < 3 or _boundary_is_closed_points(points):
        return
    points.append((float(points[0][0]), float(points[0][1])))


def _set_boundary_anchor(points: list[tuple[float, float]], idx: int, value: tuple[float, float]) -> None:
    if idx < 0 or idx >= len(points):
        return
    points[idx] = (float(value[0]), float(value[1]))
    if _boundary_is_closed_points(points) and idx in {0, len(points) - 1}:
        points[0] = (float(value[0]), float(value[1]))
        points[-1] = (float(value[0]), float(value[1]))


def _remove_boundary_anchor(points: list[tuple[float, float]], idx: int) -> None:
    if idx < 0 or idx >= len(points):
        return
    if _boundary_is_closed_points(points) and idx in {0, len(points) - 1}:
        points.pop()
        return
    points.pop(idx)


def _plot_job_panel(
    ax,
    *,
    adata_job: ad.AnnData,
    title: str,
    boundaries: dict[str, list[tuple[float, float]]] | None = None,
    max_points: int = PLOT_MAX_POINTS,
    point_size: float = POINT_SIZE,
    point_alpha: float = POINT_ALPHA,
    rng_seed: int = 0,
) -> None:
    xy = get_spatial_xy(adata_job)
    labels = adata_job.obs[LABEL_KEY]

    rng = np.random.default_rng(rng_seed)
    plot_idx = np.arange(adata_job.n_obs, dtype=np.int32)
    if plot_idx.size > max_points:
        plot_idx = np.sort(rng.choice(plot_idx, size=max_points, replace=False))

    plot_labels = labels.iloc[plot_idx]
    valid_mask = plot_labels.notna().to_numpy()
    if np.any(~valid_mask):
        missing_idx = plot_idx[~valid_mask]
        ax.scatter(
            xy[missing_idx, 0],
            xy[missing_idx, 1],
            c="#c7c7c7",
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            zorder=1,
        )
    if np.any(valid_mask):
        labeled_idx = plot_idx[valid_mask]
        label_codes = pd.Categorical(labels.iloc[labeled_idx].astype(str)).codes
        ax.scatter(
            xy[labeled_idx, 0],
            xy[labeled_idx, 1],
            c=label_codes,
            cmap=ListedColormap(["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1"]),
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            zorder=2,
        )

    if boundaries is not None:
        for boundary_name in BOUNDARY_ORDER:
            points = boundaries[boundary_name]
            if len(points) < 2:
                continue
            curve = densify_open_curve(np.asarray(points, dtype=np.float32))
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=BOUNDARY_COLORS[boundary_name],
                linewidth=1.2,
                alpha=0.95,
                zorder=3,
            )

    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def review_job(
    *,
    adata_job: ad.AnnData,
    job: JobSpec,
    paths: dict[str, Path],
    initial_boundaries: dict[str, list[tuple[float, float]]],
) -> None:
    fig, ax = plt.subplots(figsize=UI_FIGSIZE)
    fig.subplots_adjust(bottom=0.18)
    _plot_job_panel(
        ax,
        adata_job=adata_job,
        title="",
        boundaries=None,
        max_points=PLOT_MAX_POINTS,
        point_size=POINT_SIZE,
        point_alpha=POINT_ALPHA,
        rng_seed=0,
    )

    boundaries = {name: list(initial_boundaries[name]) for name in BOUNDARY_ORDER}
    active_idx = {"value": 0}
    show_boundaries = {"value": True}
    drag_state: dict[str, Any] = {"index": None, "moved": False}
    curve_artists: dict[str, object] = {}
    anchor_artists: dict[str, object] = {}
    btn_toggle: Button | None = None

    def _active_boundary_name() -> str:
        return BOUNDARY_ORDER[active_idx["value"]]

    def _set_title(message: str | None = None) -> None:
        title = (
            f"{job.dataset} | roi={job.roi_group} | ccf_adjusted={job.ccf_adjusted}\n"
            f"Boundary {_active_boundary_name()} ({active_idx['value'] + 1}/{len(BOUNDARY_ORDER)})"
        )
        if not show_boundaries["value"]:
            title = f"{title} | showing active layer only"
        if message:
            title = f"{title}\n{message}"
        ax.set_title(title)

    def refresh(message: str | None = None) -> None:
        for artist in curve_artists.values():
            try:
                artist.remove()
            except Exception:
                pass
        curve_artists.clear()
        for artist in anchor_artists.values():
            try:
                artist.remove()
            except Exception:
                pass
        anchor_artists.clear()

        visible_boundaries = BOUNDARY_ORDER if show_boundaries["value"] else (_active_boundary_name(),)
        for name in visible_boundaries:
            points = boundaries[name]
            is_active = name == _active_boundary_name()
            color = BOUNDARY_COLORS[name]
            if len(points) >= 2:
                curve = densify_open_curve(np.asarray(points, dtype=np.float32))
                curve_artists[name] = ax.plot(
                    curve[:, 0],
                    curve[:, 1],
                    color=color,
                    linewidth=2.6 if is_active else 1.4,
                    alpha=0.98 if is_active else 0.75,
                    zorder=4 if is_active else 3,
                )[0]
            if points:
                point_xy = np.asarray(points, dtype=np.float32)
                anchor_artists[name] = ax.scatter(
                    point_xy[:, 0],
                    point_xy[:, 1],
                    s=55 if is_active else 26,
                    c=color,
                    edgecolors="black" if is_active else color,
                    linewidths=0.6 if is_active else 0.3,
                    alpha=1.0 if is_active else 0.8,
                    zorder=5 if is_active else 4,
                )

        _set_title(message)
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        name = _active_boundary_name()
        points = boundaries[name]
        x = float(event.xdata)
        y = float(event.ydata)
        key = str(getattr(event, "key", "") or "").lower()
        pick_radius = _anchor_pick_radius(ax)

        if "ctrl" in key or "control" in key:
            idx = _nearest_anchor_index(points, x, y, radius=pick_radius)
            if idx is None:
                return
            _remove_boundary_anchor(points, idx)
            refresh()
            return

        if "shift" in key:
            insert_at = _insertion_index(points, x, y)
            points.insert(insert_at, (x, y))
            refresh()
            return

        idx = _nearest_anchor_index(points, x, y, radius=pick_radius)
        if idx is not None:
            drag_state["index"] = idx
            drag_state["moved"] = False
            return

        points.append((x, y))
        refresh()

    def on_motion(event) -> None:
        idx = drag_state["index"]
        if idx is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        points = boundaries[_active_boundary_name()]
        if idx >= len(points):
            drag_state["index"] = None
            return
        drag_state["moved"] = True
        _set_boundary_anchor(points, idx, (float(event.xdata), float(event.ydata)))
        refresh()

    def on_release(_event) -> None:
        idx = drag_state["index"]
        drag_moved = bool(drag_state["moved"])
        if idx is not None and not drag_moved:
            points = boundaries[_active_boundary_name()]
            if idx == 0 and not _boundary_is_closed_points(points) and len(points) >= 3:
                _close_boundary_points(points)
                refresh(message="Closed loop.")
        drag_state["index"] = None
        drag_state["moved"] = False

    def on_undo(_event) -> None:
        points = boundaries[_active_boundary_name()]
        if points:
            points.pop()
            refresh()

    def on_reset(_event) -> None:
        boundaries[_active_boundary_name()] = []
        refresh()

    def on_prev(_event) -> None:
        active_idx["value"] = (active_idx["value"] - 1) % len(BOUNDARY_ORDER)
        refresh()

    def on_next(_event) -> None:
        active_idx["value"] = (active_idx["value"] + 1) % len(BOUNDARY_ORDER)
        refresh()

    def on_toggle_boundaries(_event) -> None:
        show_boundaries["value"] = not show_boundaries["value"]
        if btn_toggle is not None:
            btn_toggle.label.set_text("Hide boundaries" if show_boundaries["value"] else "Show boundaries")
        refresh()

    def on_save(_event) -> None:
        message = save_job_boundary_state(
            job=job,
            paths=paths,
            boundaries=boundaries,
        )
        refresh(message=message)
        print(message)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    button_specs = [
        (0.03, 0.05, 0.13, 0.055, "Prev boundary", on_prev),
        (0.17, 0.05, 0.13, 0.055, "Next boundary", on_next),
        (0.31, 0.05, 0.15, 0.055, "Hide boundaries", on_toggle_boundaries),
        (0.48, 0.05, 0.10, 0.055, "Undo", on_undo),
        (0.60, 0.05, 0.14, 0.055, "Reset boundary", on_reset),
        (0.78, 0.05, 0.10, 0.055, "Save", on_save),
    ]
    button_handles: list[Button] = []
    for x0, y0, width, height, label, handler in button_specs:
        btn = Button(fig.add_axes([x0, y0, width, height]), label)
        btn.on_clicked(handler)
        button_handles.append(btn)
        if label == "Hide boundaries":
            btn_toggle = btn

    refresh()
    # Keep button widgets alive for `%matplotlib widget` in VS Code/Jupyter.
    fig._pick_manual_layer_boundary_handles = tuple(button_handles)  # type: ignore[attr-defined]
    plt.show()


# %% [markdown]
# ## Phase 0: Setup (run once)
#
# This loads the joined AnnData, builds the job table, and resets the notebook-style
# job pointer `i`. Re-run this cell to restart from the first job.

# %%
if __name__ == "__main__":
    adata = load_adata_with_labels(in_h5ad=INPUT_H5AD, labels_parquet=LABELS_PARQUET)
    jobs = build_jobs_from_obs(adata.obs)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    i = 0

    print(f"Loaded {adata.n_obs} observations from {INPUT_H5AD}")
    print(f"Found {len(jobs)} review jobs.")
    for job in jobs[:10]:
        print(
            f"  - dataset={job.dataset} roi_group={job.roi_group} "
            f"ccf_adjusted={job.ccf_adjusted} n_obs={job.n_obs}"
        )
    if len(jobs) > 10:
        print(f"  ... {len(jobs) - 10} more")


# %% [markdown]
# ## Phase 1: Run next job (re-run this cell)
# %%
if __name__ == "__main__":
    while i < len(jobs):
        job = jobs[i]
        paths = _job_paths(OUTPUT_ROOT, job)
        if SKIP_EXISTING_JSON and paths["json"].exists():
            print(f"Skipping existing job {i + 1}/{len(jobs)}: {paths['json'].name}")
            i += 1
            continue

        adata_job = subset_job_adata(adata, job)
        initial = load_boundary_state(paths["json"])
        print(
            f"Job {i + 1}/{len(jobs)}: dataset={job.dataset} roi_group={job.roi_group} "
            f"ccf_adjusted={job.ccf_adjusted} n_obs={job.n_obs}"
        )
        print(f"  json={paths['json']}")
        if paths["json"].exists():
            print("  loaded existing boundary state")
        review_job(adata_job=adata_job, job=job, paths=paths, initial_boundaries=initial)
        i += 1
        break
    else:
        print(f"Done: i={i} >= n_jobs={len(jobs)}")




# %% [markdown]
# ## Phase 2: Run assignment for all saved jobs
#
# This keeps the interactive save responsive: the figure save only writes boundary JSON,
# and this cell performs manual-layer assignment and QC generation afterward for
# every unit with a saved boundary JSON.

# %%
if __name__ == "__main__":
    if adata is None or not jobs:
        raise RuntimeError("Run the setup cell first.")
    processed = 0
    succeeded = 0
    failed = 0
    skipped_missing_json = 0

    for assign_job in jobs:
        assign_paths = _job_paths(OUTPUT_ROOT, assign_job)
        if not assign_paths["json"].exists():
            skipped_missing_json += 1
            continue

        assign_adata_job = subset_job_adata(adata, assign_job)
        assign_boundaries = load_boundary_state(assign_paths["json"])
        ok, message = save_job_state_and_assignment(
            adata_job=assign_adata_job,
            job=assign_job,
            paths=assign_paths,
            boundaries=assign_boundaries,
        )
        processed += 1
        if ok:
            succeeded += 1
            print(
                f"[ok] dataset={assign_job.dataset} roi={assign_job.roi_group} "
                f"ccf_adjusted={assign_job.ccf_adjusted}"
            )
            print(f"  assignments={assign_paths['assignments']}")
            print(f"  qc_png={assign_paths['qc_png']}")
        else:
            failed += 1
            print(
                f"[failed] dataset={assign_job.dataset} roi={assign_job.roi_group} "
                f"ccf_adjusted={assign_job.ccf_adjusted}: {message}"
            )

    print(
        f"Phase 2 summary: processed={processed} succeeded={succeeded} "
        f"failed={failed} skipped_missing_json={skipped_missing_json}"
    )


# %% [markdown]
# ## Phase 3: Show latest assignment vs earlier labels

# %%
if __name__ == "__main__":
    if adata is None or not jobs:
        raise RuntimeError("Run the setup cell first.")
    if i <= 0:
        raise RuntimeError("Run the `Run next job` cell and save at least one job first.")

    review_idx = min(max(i - 1, 0), len(jobs) - 1)
    review_job_spec = jobs[review_idx]
    review_paths = _job_paths(OUTPUT_ROOT, review_job_spec)
    if not review_paths["assignments"].exists():
        raise FileNotFoundError(
            f"Missing assignments parquet for the latest job: {review_paths['assignments']}. "
            "Save the job successfully first."
        )

    review_adata = subset_job_adata(adata, review_job_spec)
    review_xy = get_spatial_xy(review_adata)
    assignments = pd.read_parquet(review_paths["assignments"]).set_index("obs_ix")
    obs_ix = review_adata.obs["obs_ix"].to_numpy(dtype=np.int64, copy=False)
    manual_layer = assignments.loc[obs_ix, "manual_layer"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=96)
    original_codes = pd.Categorical(review_adata.obs[LABEL_KEY].astype(str)).codes
    manual_codes = pd.Categorical(manual_layer.astype(str), categories=list(MANUAL_LAYER_ORDER), ordered=True).codes

    axes[0].scatter(review_xy[:, 0], review_xy[:, 1], c=original_codes, cmap="tab10", s=1.5, linewidths=0)
    axes[0].set_title("Earlier labels (mclust_k7)")
    axes[1].scatter(review_xy[:, 0], review_xy[:, 1], c=manual_codes, cmap="tab10", s=1.5, linewidths=0)
    axes[1].set_title("Assigned manual_layer")

    saved_boundaries = load_boundary_state(review_paths["json"])
    for boundary_name in BOUNDARY_ORDER:
        points = saved_boundaries[boundary_name]
        if len(points) < 2:
            continue
        curve = densify_open_curve(np.asarray(points, dtype=np.float32))
        axes[0].plot(curve[:, 0], curve[:, 1], color="magenta", linewidth=1.0, alpha=0.9)
        axes[1].plot(curve[:, 0], curve[:, 1], color="magenta", linewidth=1.0, alpha=0.9)

    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{review_job_spec.dataset} | roi={review_job_spec.roi_group} | ccf_adjusted={review_job_spec.ccf_adjusted}"
    )
    fig.tight_layout()
    plt.show()


# %% [markdown]
# ## Phase 3b: Montage all saved units
#
# Build a single figure showing the mclust scatter plus saved spline overlays for
# every unit that already has a boundary JSON.

# %%
if __name__ == "__main__":
    if adata is None or not jobs:
        raise RuntimeError("Run the setup cell first.")

    montage_jobs: list[tuple[JobSpec, dict[str, Path]]] = []
    for montage_job in jobs:
        montage_paths = _job_paths(OUTPUT_ROOT, montage_job)
        if montage_paths["json"].exists():
            montage_jobs.append((montage_job, montage_paths))

    if not montage_jobs:
        raise FileNotFoundError("No saved boundary JSONs found. Save at least one unit first.")

    n_units = len(montage_jobs)
    ncols = min(MONTAGE_NCOLS, n_units)
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(MONTAGE_PANEL_SIZE[0] * ncols, MONTAGE_PANEL_SIZE[1] * nrows),
        dpi=MONTAGE_DPI,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, (montage_job, montage_paths) in zip(axes_arr, montage_jobs, strict=False):
        montage_adata = subset_job_adata(adata, montage_job)
        saved_boundaries = load_boundary_state(montage_paths["json"])
        _plot_job_panel(
            ax,
            adata_job=montage_adata,
            title=f"{montage_job.dataset}\nroi={montage_job.roi_group} | ccf={montage_job.ccf_adjusted}",
            boundaries=saved_boundaries,
            max_points=MONTAGE_MAX_POINTS_PER_UNIT,
            point_size=MONTAGE_POINT_SIZE,
            point_alpha=MONTAGE_POINT_ALPHA,
            rng_seed=0,
        )

    for ax in axes_arr[n_units:]:
        ax.axis("off")

    fig.suptitle("Manual layer boundary montage", fontsize=14)
    fig.tight_layout()
    montage_png = OUTPUT_ROOT / "manual_layer_boundary_montage.png"
    fig.savefig(montage_png, bbox_inches="tight")
    print(f"Wrote montage: {montage_png}")
    plt.show()


# %% [markdown]
# ## Phase 4: Rebuild combined assignment parquet (optional)
#
# Re-run this after any save if you want to refresh the combined table explicitly.


if __name__ == "__main__":
    combined = combine_existing_assignment_tables(OUTPUT_ROOT)
    if combined is None:
        print("No per-job assignment tables found yet.")
    else:
        print(f"Wrote combined assignments: {combined}")

# %%

# %% [markdown]
# ## Next actions
#
# 1. Run `Phase 1` to open the next job.
# 2. Click `Save` in the interactive figure to write boundary JSON.
# 3. Run `Phase 2` to build assignments and QC outputs for the latest saved job.
# 4. Run `Phase 3` to inspect earlier labels versus `manual_layer`.
