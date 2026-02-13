from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import find_contours

MANUAL_CONNECT_CORONAL_SLICE_IS: tuple[int, ...] = (
    183,
    184,
    185,
    230,
    231,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    246,
    247,
    248,
    284,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
)
MANUAL_CONNECT_CORONAL_PATH_TEMPLATE = "manual_coronal_midcurve_override_slice{slice_i}_yx.npy"
MANUAL_CONNECT_SAGITTAL_SLICE_KS: tuple[int, ...] = (212, 213, 214, 215, 216, 217, 218, 219)
MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE = "manual_sagittal_midcurve_override_slice{slice_k}_yx.npy"
CORONAL_T_SMOOTH_WINDOW_SLICES = 9
CORONAL_T_SMOOTH_N_T = 257
SAGITTAL_T_SMOOTH_WINDOW_SLICES = 9
SAGITTAL_T_SMOOTH_N_T = 257
SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS: tuple[int, ...] = (173, 174, 175)
CORONAL_T_ANCHOR_SMOOTH_WINDOW_SLICES = 9
SAGITTAL_T_ANCHOR_SMOOTH_WINDOW_SLICES = 9
JOINT_T_ANCHOR_SHARED_WEIGHT = 0.2
JOINT_XY_SHARED_J_WEIGHT = 0.1
IJK_LUT_KNN_K = 24
IJK_LUT_KNN_SIGMA_SLICE = 2.0
IJK_LUT_KNN_SIGMA_T = 0.05
IJK_LUT_KNN_WEIGHT_EPS = 1e-12


@dataclass(frozen=True)
class CoronalMidlineColumns:
    """Per-slice midline + vent/pia endpoints.

    Coordinates are in the coronal plane (y/x) for a fixed slice_i:
    - (y, x) is a point on the midline curve (u≈0.5 sheet intersection)
    - (vent_y, vent_x) and (pia_y, pia_x) are boundary intersection points along a local normal

    The `t` column here is `t_all` used by `midsurface_coords.py`:
    - It is the full-path along-midline coordinate on that slice.
    - It is normalized arc length on the representative curve (`t_all ∈ [0,1]`).

    Given r01 in [0,1] (0=ventricular/inner, 1=pial), the point in voxel coordinates is:
        p_yx = vent_yx + r01 * (pia_yx - vent_yx)
    and full ijk is (slice_i, p_y, p_x).
    """

    slice_i: int
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    vent_y: np.ndarray
    vent_x: np.ndarray
    pia_y: np.ndarray
    pia_x: np.ndarray
    thickness_um: np.ndarray


def load_coronal_midline_columns(csv_path: Path) -> dict[int, CoronalMidlineColumns]:
    if csv_path.exists():
        return _load_coronal_midline_columns_csv(csv_path)
    return _build_coronal_midline_columns_from_halfway_u(csv_path.parent)


def _interp(t_grid: np.ndarray, values: np.ndarray, t: float) -> float:
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite.")
    return float(np.interp(t, t_grid, values))


def _column_t_affine_bounds(t_values: np.ndarray) -> tuple[float, float]:
    t_arr = np.asarray(t_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(t_arr)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("Column t must contain at least 2 finite values.")
    t_min = float(np.min(t_arr[finite]))
    t_max = float(np.max(t_arr[finite]))
    den = t_max - t_min
    if not np.isfinite(den) or den <= 1.0e-9:
        raise ValueError(f"Column t has invalid affine span: min={t_min}, max={t_max}.")
    return t_min, t_max


def _column_t_to_t_all(*, t: float, t_values: np.ndarray) -> float:
    t_min, t_max = _column_t_affine_bounds(t_values)
    return float((float(t) - t_min) / (t_max - t_min))


def _fill_nan_series(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape={x.shape}.")
    finite = np.isfinite(x)
    if not np.any(finite):
        return x.copy()
    idx = np.arange(x.size, dtype=np.float64)
    out = x.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return out


def _fill_nan_series_interior(values: np.ndarray) -> np.ndarray:
    """Fill interior NaN gaps while keeping out-of-support edges as NaN."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape={x.shape}.")
    finite = np.isfinite(x)
    if not np.any(finite):
        return x.copy()

    idx = np.arange(x.size, dtype=np.float64)
    out = x.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    first = int(np.flatnonzero(finite)[0])
    last = int(np.flatnonzero(finite)[-1])
    if first > 0:
        out[:first] = np.nan
    if last < (x.size - 1):
        out[last + 1 :] = np.nan
    return out


def _interp_series_on_t_grid(*, t: np.ndarray, values: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_vals = np.asarray(t, dtype=np.float64)
    v_vals = np.asarray(values, dtype=np.float64)
    t_out = np.asarray(t_grid, dtype=np.float64)
    if t_vals.ndim != 1 or v_vals.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got t={t_vals.shape} values={v_vals.shape}.")
    if t_vals.shape != v_vals.shape:
        raise ValueError(f"Shape mismatch: t={t_vals.shape} values={v_vals.shape}.")
    if t_out.ndim != 1:
        raise ValueError(f"Expected 1D t_grid, got {t_out.shape}.")

    finite = np.isfinite(t_vals) & np.isfinite(v_vals)
    if int(np.count_nonzero(finite)) < 2:
        return np.full(t_out.shape, np.nan, dtype=np.float64)

    t_f = t_vals[finite]
    v_f = v_vals[finite]
    order = np.argsort(t_f)
    t_f = t_f[order]
    v_f = v_f[order]
    t_unique, unique_idx = np.unique(t_f, return_index=True)
    v_unique = v_f[unique_idx]
    if t_unique.size < 2:
        return np.full(t_out.shape, np.nan, dtype=np.float64)
    return np.interp(t_out, t_unique, v_unique, left=np.nan, right=np.nan).astype(np.float64, copy=False)


def _nanmean_sliding_rows(values: np.ndarray, *, radius: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}.")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    if radius == 0:
        return arr.copy()

    out = np.full(arr.shape, np.nan, dtype=np.float64)
    n_rows = int(arr.shape[0])
    for row in range(n_rows):
        lo = max(0, row - radius)
        hi = min(n_rows, row + radius + 1)
        window = arr[lo:hi]
        finite = np.isfinite(window)
        den = np.sum(finite, axis=0, dtype=np.int32)
        num = np.nansum(window, axis=0, dtype=np.float64)
        valid = den > 0
        out[row, valid] = num[valid] / den[valid]
    return out


def _smooth_coronal_midline_xy_across_slices(
    columns: dict[int, CoronalMidlineColumns],
    *,
    window_slices: int,
    n_t: int,
) -> dict[int, CoronalMidlineColumns]:
    if window_slices <= 1 or len(columns) < 3:
        return columns
    if window_slices % 2 == 0:
        raise ValueError(f"window_slices must be odd, got {window_slices}.")
    if n_t < 8:
        raise ValueError(f"n_t must be >= 8, got {n_t}.")

    keys = sorted(columns.keys())
    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    y_grid = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)
    x_grid = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)

    for idx, key in enumerate(keys):
        c = columns[int(key)]
        y_grid[idx] = _interp_series_on_t_grid(t=c.t, values=c.y, t_grid=t_grid)
        x_grid[idx] = _interp_series_on_t_grid(t=c.t, values=c.x, t_grid=t_grid)

    radius = int(window_slices // 2)
    y_smooth = _nanmean_sliding_rows(y_grid, radius=radius)
    x_smooth = _nanmean_sliding_rows(x_grid, radius=radius)

    out: dict[int, CoronalMidlineColumns] = {}
    for idx, key in enumerate(keys):
        c = columns[int(key)]
        y_new = _interp_series_on_t_grid(t=t_grid, values=y_smooth[idx], t_grid=c.t)
        x_new = _interp_series_on_t_grid(t=t_grid, values=x_smooth[idx], t_grid=c.t)
        if not np.isfinite(y_new).all():
            y_new = np.array(c.y, dtype=np.float64, copy=True)
        if not np.isfinite(x_new).all():
            x_new = np.array(c.x, dtype=np.float64, copy=True)
        out[int(key)] = CoronalMidlineColumns(
            slice_i=int(c.slice_i),
            t=np.array(c.t, dtype=np.float64, copy=True),
            y=y_new,
            x=x_new,
            vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
            vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
            pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
            pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
            thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
        )
    return out


@dataclass(frozen=True)
class SagittalMidlineColumns:
    """Same as CoronalMidlineColumns but for sagittal slices k (axis=2).

    Plane coordinates are (y=i, x=j), with k fixed.
    """

    slice_k: int
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    vent_y: np.ndarray
    vent_x: np.ndarray
    pia_y: np.ndarray
    pia_x: np.ndarray
    thickness_um: np.ndarray


def _smooth_sagittal_midline_xy_across_slices(
    columns: dict[int, SagittalMidlineColumns],
    *,
    window_slices: int,
    n_t: int,
    exclude_slice_keys: set[int] | None = None,
) -> dict[int, SagittalMidlineColumns]:
    if window_slices <= 1 or len(columns) < 3:
        return columns
    if window_slices % 2 == 0:
        raise ValueError(f"window_slices must be odd, got {window_slices}.")
    if n_t < 8:
        raise ValueError(f"n_t must be >= 8, got {n_t}.")

    keys = sorted(columns.keys())
    excluded = set() if exclude_slice_keys is None else {int(k) for k in exclude_slice_keys}
    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    y_grid = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)
    x_grid = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)

    for idx, key in enumerate(keys):
        c = columns[int(key)]
        y_grid[idx] = _interp_series_on_t_grid(t=c.t, values=c.y, t_grid=t_grid)
        x_grid[idx] = _interp_series_on_t_grid(t=c.t, values=c.x, t_grid=t_grid)

    y_smooth_src = y_grid.copy()
    x_smooth_src = x_grid.copy()
    for idx, key in enumerate(keys):
        if int(key) in excluded:
            y_smooth_src[idx, :] = np.nan
            x_smooth_src[idx, :] = np.nan

    radius = int(window_slices // 2)
    y_smooth = _nanmean_sliding_rows(y_smooth_src, radius=radius)
    x_smooth = _nanmean_sliding_rows(x_smooth_src, radius=radius)

    out: dict[int, SagittalMidlineColumns] = {}
    for idx, key in enumerate(keys):
        c = columns[int(key)]
        if int(key) in excluded:
            out[int(key)] = SagittalMidlineColumns(
                slice_k=int(c.slice_k),
                t=np.array(c.t, dtype=np.float64, copy=True),
                y=np.array(c.y, dtype=np.float64, copy=True),
                x=np.array(c.x, dtype=np.float64, copy=True),
                vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
                vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
                pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
                pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
                thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
            )
            continue

        y_new = _interp_series_on_t_grid(t=t_grid, values=y_smooth[idx], t_grid=c.t)
        x_new = _interp_series_on_t_grid(t=t_grid, values=x_smooth[idx], t_grid=c.t)
        if not np.isfinite(y_new).all():
            y_new = np.array(c.y, dtype=np.float64, copy=True)
        if not np.isfinite(x_new).all():
            x_new = np.array(c.x, dtype=np.float64, copy=True)
        out[int(key)] = SagittalMidlineColumns(
            slice_k=int(c.slice_k),
            t=np.array(c.t, dtype=np.float64, copy=True),
            y=y_new,
            x=x_new,
            vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
            vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
            pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
            pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
            thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
        )
    return out


def _smooth_shared_j_coordinate_across_axes(
    coronal_columns: dict[int, CoronalMidlineColumns],
    sagittal_columns: dict[int, SagittalMidlineColumns],
    *,
    window_slices: int,
    n_t: int,
    sagittal_exclude_slice_keys: set[int] | None = None,
    shared_weight: float = JOINT_XY_SHARED_J_WEIGHT,
) -> tuple[dict[int, CoronalMidlineColumns], dict[int, SagittalMidlineColumns]]:
    if window_slices <= 1 or (not coronal_columns and not sagittal_columns):
        return coronal_columns, sagittal_columns
    if window_slices % 2 == 0:
        raise ValueError(f"window_slices must be odd, got {window_slices}.")
    if n_t < 8:
        raise ValueError(f"n_t must be >= 8, got {n_t}.")
    w = float(shared_weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"shared_weight must be in [0,1], got {w}.")
    if w <= 0.0:
        return coronal_columns, sagittal_columns

    coronal_keys = sorted(coronal_columns.keys())
    sagittal_keys = sorted(sagittal_columns.keys())
    n_rows = len(coronal_keys) + len(sagittal_keys)
    if n_rows < 3:
        return coronal_columns, sagittal_columns

    excluded = set() if sagittal_exclude_slice_keys is None else {int(k) for k in sagittal_exclude_slice_keys}
    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    pooled_rows: list[np.ndarray] = []
    for slice_i in coronal_keys:
        c = coronal_columns[int(slice_i)]
        row = _interp_series_on_t_grid(t=c.t, values=c.y, t_grid=t_grid)
        if int(np.count_nonzero(np.isfinite(row))) >= 8:
            pooled_rows.append(row)
    for slice_k in sagittal_keys:
        c = sagittal_columns[int(slice_k)]
        if int(slice_k) in excluded:
            continue
        row = _interp_series_on_t_grid(t=c.t, values=c.x, t_grid=t_grid)
        if int(np.count_nonzero(np.isfinite(row))) >= 8:
            pooled_rows.append(row)

    if len(pooled_rows) < 3:
        return coronal_columns, sagittal_columns

    pooled = np.asarray(pooled_rows, dtype=np.float64)
    shared_j = np.nanmedian(pooled, axis=0)
    finite = np.isfinite(shared_j)
    if int(np.count_nonzero(finite)) < 2:
        return coronal_columns, sagittal_columns
    if not np.all(finite):
        shared_j = _fill_nan_series(shared_j)
    if not np.isfinite(shared_j).all():
        return coronal_columns, sagittal_columns

    coronal_out: dict[int, CoronalMidlineColumns] = {}
    sagittal_out: dict[int, SagittalMidlineColumns] = {}
    for key in coronal_keys:
        c = coronal_columns[int(key)]
        shared_on_t = _interp_series_on_t_grid(t=t_grid, values=shared_j, t_grid=c.t)
        if np.isfinite(shared_on_t).all():
            y_new = (1.0 - w) * np.asarray(c.y, dtype=np.float64) + w * shared_on_t
        else:
            y_new = np.array(c.y, dtype=np.float64, copy=True)
        coronal_out[int(key)] = CoronalMidlineColumns(
            slice_i=int(c.slice_i),
            t=np.array(c.t, dtype=np.float64, copy=True),
            y=y_new,
            x=np.array(c.x, dtype=np.float64, copy=True),
            vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
            vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
            pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
            pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
            thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
        )
    for key in sagittal_keys:
        c = sagittal_columns[int(key)]
        if int(key) in excluded:
            sagittal_out[int(key)] = SagittalMidlineColumns(
                slice_k=int(c.slice_k),
                t=np.array(c.t, dtype=np.float64, copy=True),
                y=np.array(c.y, dtype=np.float64, copy=True),
                x=np.array(c.x, dtype=np.float64, copy=True),
                vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
                vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
                pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
                pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
                thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
            )
            continue
        shared_on_t = _interp_series_on_t_grid(t=t_grid, values=shared_j, t_grid=c.t)
        if np.isfinite(shared_on_t).all():
            x_new = (1.0 - w) * np.asarray(c.x, dtype=np.float64) + w * shared_on_t
        else:
            x_new = np.array(c.x, dtype=np.float64, copy=True)
        sagittal_out[int(key)] = SagittalMidlineColumns(
            slice_k=int(c.slice_k),
            t=np.array(c.t, dtype=np.float64, copy=True),
            y=np.array(c.y, dtype=np.float64, copy=True),
            x=x_new,
            vent_y=np.array(c.vent_y, dtype=np.float64, copy=True),
            vent_x=np.array(c.vent_x, dtype=np.float64, copy=True),
            pia_y=np.array(c.pia_y, dtype=np.float64, copy=True),
            pia_x=np.array(c.pia_x, dtype=np.float64, copy=True),
            thickness_um=np.array(c.thickness_um, dtype=np.float64, copy=True),
        )

    return coronal_out, sagittal_out


def load_sagittal_midline_columns(csv_path: Path) -> dict[int, SagittalMidlineColumns]:
    if csv_path.exists():
        return _load_sagittal_midline_columns_csv(csv_path)
    return _build_sagittal_midline_columns_from_halfway_u(csv_path.parent)


def _load_halfway_u_and_masks(outdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_path = outdir / "halfway_u_3d_ds.npy"
    if not u_path.exists():
        raise FileNotFoundError(
            f"Missing {u_path}. Cannot derive midline columns without halfway_u_3d_ds.npy."
        )
    u_3d = np.load(u_path).astype(np.float32, copy=False)
    if u_3d.ndim != 3:
        raise ValueError(f"Expected halfway_u_3d_ds.npy to be 3D, got shape={u_3d.shape}.")

    fit_mask_path = outdir / "cortex_mask_fit_3d_ds.npy"
    clean_mask_path = outdir / "cortex_mask_clean_3d_ds.npy"
    if fit_mask_path.exists():
        mask_3d = np.load(fit_mask_path).astype(bool, copy=False)
    elif clean_mask_path.exists():
        mask_3d = np.load(clean_mask_path).astype(bool, copy=False)
    else:
        raise FileNotFoundError(
            "Missing cortex mask for midline derivation. Expected one of "
            "{cortex_mask_fit_3d_ds.npy, cortex_mask_clean_3d_ds.npy}."
        )

    include_path = outdir / "midline_include_neo_meso_3d_ds.npy"
    if include_path.exists():
        include_3d = np.load(include_path).astype(bool, copy=False)
    else:
        include_3d = mask_3d

    if mask_3d.shape != u_3d.shape:
        raise ValueError(f"Mask shape mismatch: mask={mask_3d.shape} u={u_3d.shape}.")
    if include_3d.shape != u_3d.shape:
        raise ValueError(f"Include-mask shape mismatch: include={include_3d.shape} u={u_3d.shape}.")
    return u_3d, mask_3d, include_3d


def _nonzero_include_slice_keys(outdir: Path, *, axis: Literal["coronal", "sagittal"]) -> np.ndarray:
    _, _, include_3d = _load_halfway_u_and_masks(outdir)
    if axis == "coronal":
        keys = np.where(include_3d.any(axis=(1, 2)))[0].astype(np.int32, copy=False)
    else:
        keys = np.where(include_3d.any(axis=(0, 1)))[0].astype(np.int32, copy=False)
    if keys.size == 0:
        raise ValueError(f"No non-zero include-mask slices found for axis={axis}.")
    return keys


def _load_neocortex_mesocortex_overlay_mask(outdir: Path, *, fallback_mask: np.ndarray) -> np.ndarray:
    path = outdir / "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy"
    if path.exists():
        overlay = np.load(path).astype(bool, copy=False)
        if overlay.shape != fallback_mask.shape:
            raise ValueError(
                "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy shape mismatch: "
                f"{overlay.shape} vs {fallback_mask.shape}."
            )
        return overlay
    print(
        "[midline] overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy missing; "
        "falling back to include-mask for t remapping."
    )
    return np.asarray(fallback_mask, dtype=bool)


def _polyline_length(path_yx: np.ndarray) -> float:
    if path_yx.ndim != 2 or path_yx.shape[0] < 2 or path_yx.shape[1] != 2:
        return 0.0
    d = np.diff(path_yx.astype(np.float64, copy=False), axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _normalized_arc_t_for_path(path_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yx = _dedupe_consecutive_points(path_yx)
    if yx.shape[0] < 2:
        raise ValueError("Path must have at least 2 unique points.")
    y = yx[:, 0].astype(np.float64, copy=False)
    x = yx[:, 1].astype(np.float64, copy=False)
    seg = np.sqrt(np.sum(np.diff(yx, axis=0) ** 2, axis=1))
    s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Path had invalid arc-length.")
    t = (s / total).astype(np.float64, copy=False)
    return y, x, t


def _t_anchor_range_from_overlap(
    *,
    y: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    overlap_mask_yx: np.ndarray,
) -> tuple[float, float] | None:
    overlap_mask = np.asarray(overlap_mask_yx, dtype=bool)
    if overlap_mask.ndim != 2:
        raise ValueError(f"Expected 2D overlap mask, got shape={overlap_mask.shape}.")
    if y.shape != x.shape or y.shape != t.shape:
        raise ValueError(f"Shape mismatch for y/x/t: {y.shape}, {x.shape}, {t.shape}.")

    yi = np.clip(np.rint(y).astype(np.int64, copy=False), 0, overlap_mask.shape[0] - 1)
    xi = np.clip(np.rint(x).astype(np.int64, copy=False), 0, overlap_mask.shape[1] - 1)
    inside = overlap_mask[yi, xi]
    inside_idx = np.flatnonzero(inside).astype(np.int64, copy=False)
    if inside_idx.size < 2:
        return None
    i0 = int(inside_idx[0])
    i1 = int(inside_idx[-1])
    if i1 <= i0:
        return None
    t0 = float(t[i0])
    t1 = float(t[i1])
    if not np.isfinite(t0) or not np.isfinite(t1) or (t1 - t0) <= 1.0e-9:
        return None
    return (t0, t1)


def _raw_t_anchor_range_for_path(path_yx: np.ndarray, overlap_mask_yx: np.ndarray) -> tuple[float, float] | None:
    try:
        y, x, t = _normalized_arc_t_for_path(path_yx)
    except ValueError:
        return None
    return _t_anchor_range_from_overlap(y=y, x=x, t=t, overlap_mask_yx=overlap_mask_yx)


def _smooth_t_anchor_ranges_by_slice(
    raw_anchors: dict[int, tuple[float, float]],
    *,
    window_slices: int,
    exclude_slice_keys: set[int] | None = None,
) -> dict[int, tuple[float, float]]:
    if not raw_anchors:
        return {}
    if window_slices <= 1 or len(raw_anchors) < 3:
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw_anchors.items()}
    if window_slices % 2 == 0:
        raise ValueError(f"window_slices must be odd, got {window_slices}.")

    keys = np.asarray(sorted(int(k) for k in raw_anchors.keys()), dtype=np.int32)
    excluded = set() if exclude_slice_keys is None else {int(k) for k in exclude_slice_keys}
    vals = np.full((keys.size, 2), np.nan, dtype=np.float64)
    for idx, key in enumerate(keys.tolist()):
        t0, t1 = raw_anchors[int(key)]
        vals[idx, 0] = float(t0)
        vals[idx, 1] = float(t1)

    smooth_src = vals.copy()
    for idx, key in enumerate(keys.tolist()):
        if int(key) in excluded:
            smooth_src[idx, :] = np.nan

    smoothed = _nanmean_sliding_rows(smooth_src, radius=int(window_slices // 2))
    out: dict[int, tuple[float, float]] = {}
    for idx, key in enumerate(keys.tolist()):
        raw_t0, raw_t1 = raw_anchors[int(key)]
        if int(key) in excluded:
            out[int(key)] = (float(raw_t0), float(raw_t1))
            continue
        t0 = float(smoothed[idx, 0])
        t1 = float(smoothed[idx, 1])
        if np.isfinite(t0) and np.isfinite(t1) and (t1 - t0) > 1.0e-9:
            out[int(key)] = (t0, t1)
        else:
            out[int(key)] = (float(raw_t0), float(raw_t1))
    return out


def _is_valid_t_anchor_range(t0: float, t1: float) -> bool:
    return bool(np.isfinite(float(t0)) and np.isfinite(float(t1)) and (float(t1) - float(t0)) > 1.0e-9)


def _smooth_t_anchor_ranges_joint(
    *,
    coronal_raw_anchors: dict[int, tuple[float, float]],
    sagittal_raw_anchors: dict[int, tuple[float, float]],
    coronal_window_slices: int,
    sagittal_window_slices: int,
    sagittal_exclude_slice_keys: set[int] | None = None,
    shared_weight: float = JOINT_T_ANCHOR_SHARED_WEIGHT,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]], tuple[float, float] | None]:
    """Jointly smooth coronal/sagittal anchor ranges with partial pooling.

    The first stage preserves existing behavior (axis-local sliding smoothing). The second stage
    introduces coupling by shrinking both axes toward a shared robust center estimated from all
    non-excluded smoothed ranges. This avoids assuming direct slice-index correspondence across axes.
    """

    w = float(shared_weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"shared_weight must be in [0,1], got {w}.")

    excluded_sagittal = set() if sagittal_exclude_slice_keys is None else {int(k) for k in sagittal_exclude_slice_keys}
    coronal_smoothed = _smooth_t_anchor_ranges_by_slice(
        coronal_raw_anchors,
        window_slices=int(coronal_window_slices),
        exclude_slice_keys=None,
    )
    sagittal_smoothed = _smooth_t_anchor_ranges_by_slice(
        sagittal_raw_anchors,
        window_slices=int(sagittal_window_slices),
        exclude_slice_keys=excluded_sagittal,
    )

    if w <= 0.0:
        return coronal_smoothed, sagittal_smoothed, None

    pooled_t0: list[float] = []
    pooled_t1: list[float] = []
    for t0, t1 in coronal_smoothed.values():
        if _is_valid_t_anchor_range(float(t0), float(t1)):
            pooled_t0.append(float(t0))
            pooled_t1.append(float(t1))
    for key, (t0, t1) in sagittal_smoothed.items():
        if int(key) in excluded_sagittal:
            continue
        if _is_valid_t_anchor_range(float(t0), float(t1)):
            pooled_t0.append(float(t0))
            pooled_t1.append(float(t1))

    shared_anchor: tuple[float, float] | None = None
    if pooled_t0:
        cand_t0 = float(np.median(np.asarray(pooled_t0, dtype=np.float64)))
        cand_t1 = float(np.median(np.asarray(pooled_t1, dtype=np.float64)))
        if _is_valid_t_anchor_range(cand_t0, cand_t1):
            shared_anchor = (cand_t0, cand_t1)

    if shared_anchor is None:
        return coronal_smoothed, sagittal_smoothed, None

    def _blend_with_shared(
        smoothed: dict[int, tuple[float, float]],
        raw: dict[int, tuple[float, float]],
        *,
        excluded: set[int] | None,
    ) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        excluded_keys = set() if excluded is None else {int(k) for k in excluded}
        for key, (s_t0, s_t1) in smoothed.items():
            raw_t0, raw_t1 = raw[int(key)]
            if int(key) in excluded_keys:
                out[int(key)] = (float(raw_t0), float(raw_t1))
                continue
            if not _is_valid_t_anchor_range(float(s_t0), float(s_t1)):
                out[int(key)] = (float(raw_t0), float(raw_t1))
                continue
            b_t0 = (1.0 - w) * float(s_t0) + w * float(shared_anchor[0])
            b_t1 = (1.0 - w) * float(s_t1) + w * float(shared_anchor[1])
            if _is_valid_t_anchor_range(b_t0, b_t1):
                out[int(key)] = (float(b_t0), float(b_t1))
            else:
                out[int(key)] = (float(raw_t0), float(raw_t1))
        return out

    coronal_joint = _blend_with_shared(coronal_smoothed, coronal_raw_anchors, excluded=None)
    sagittal_joint = _blend_with_shared(sagittal_smoothed, sagittal_raw_anchors, excluded=excluded_sagittal)
    return coronal_joint, sagittal_joint, shared_anchor


def _dedupe_consecutive_points(path_yx: np.ndarray) -> np.ndarray:
    if path_yx.ndim != 2 or path_yx.shape[1] != 2:
        raise ValueError(f"Expected path shape (N,2), got {path_yx.shape}.")
    if path_yx.shape[0] <= 1:
        return path_yx.astype(np.float64, copy=False)

    d = np.diff(path_yx.astype(np.float64, copy=False), axis=0)
    keep = np.any(np.abs(d) > 1.0e-9, axis=1)
    return np.concatenate([path_yx[:1], path_yx[1:][keep]], axis=0).astype(np.float64, copy=False)


def _extract_largest_midline_contour(u_yx: np.ndarray, mask_yx: np.ndarray) -> np.ndarray | None:
    if u_yx.ndim != 2 or mask_yx.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got u={u_yx.shape}, mask={mask_yx.shape}.")
    if u_yx.shape != mask_yx.shape:
        raise ValueError(f"Shape mismatch: u={u_yx.shape} mask={mask_yx.shape}.")
    if not np.any(mask_yx):
        return None

    in_vals = u_yx[mask_yx]
    finite = np.isfinite(in_vals)
    if not np.any(finite):
        return None
    min_v = float(np.min(in_vals[finite]))
    max_v = float(np.max(in_vals[finite]))
    if not (min_v <= 0.5 <= max_v):
        return None

    contours = find_contours(u_yx.astype(np.float64, copy=False), level=0.5, mask=mask_yx.astype(bool, copy=False))
    if not contours:
        return None
    path = max(contours, key=_polyline_length)
    path = _dedupe_consecutive_points(np.asarray(path, dtype=np.float64))
    if path.shape[0] < 2:
        return None
    return path


def _load_manual_coronal_override_path(outdir: Path, *, slice_i: int) -> np.ndarray | None:
    path = outdir / MANUAL_CONNECT_CORONAL_PATH_TEMPLATE.format(slice_i=int(slice_i))
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    if not np.isfinite(arr).all():
        return None
    return _dedupe_consecutive_points(arr)


def _load_manual_sagittal_override_path(outdir: Path, *, slice_k: int) -> np.ndarray | None:
    path = outdir / MANUAL_CONNECT_SAGITTAL_PATH_TEMPLATE.format(slice_k=int(slice_k))
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    if not np.isfinite(arr).all():
        return None
    return _dedupe_consecutive_points(arr)


def _sample_mask_linear(mask: np.ndarray, yx: np.ndarray) -> np.ndarray:
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected yx shape (N,2), got {yx.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    return map_coordinates(mask.astype(np.float64, copy=False), [yx[:, 0], yx[:, 1]], order=1, mode="constant", cval=0.0)


def _march_to_boundary_along_normal(
    mask: np.ndarray,
    *,
    p_yx: np.ndarray,
    n_yx: np.ndarray,
    direction: float,
    max_t: float,
    step: float,
    bisect_iters: int = 14,
) -> float:
    if p_yx.shape != (2,) or n_yx.shape != (2,):
        raise ValueError("p_yx and n_yx must be shape (2,).")
    if direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1.0 or +1.0.")
    if max_t <= 0.0 or step <= 0.0:
        raise ValueError("max_t and step must be > 0.")
    if bisect_iters < 1:
        raise ValueError("bisect_iters must be >= 1.")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    h, w = mask.shape
    p0 = p_yx.astype(np.float64, copy=False)
    n = n_yx.astype(np.float64, copy=False)

    def inside_at(t: float) -> bool:
        pt = p0 + (direction * float(t)) * n
        y = float(pt[0])
        x = float(pt[1])
        if y < 0.0 or x < 0.0 or y > float(h - 1) or x > float(w - 1):
            return False
        yi = int(np.clip(int(np.rint(y)), 0, h - 1))
        xi = int(np.clip(int(np.rint(x)), 0, w - 1))
        return bool(mask[yi, xi])

    if not inside_at(0.0):
        return float("nan")

    last_inside_t = 0.0
    for t in np.arange(step, max_t + step, step, dtype=np.float64):
        if inside_at(float(t)):
            last_inside_t = float(t)
            continue

        lo = float(last_inside_t)
        hi = float(t)
        for _ in range(int(bisect_iters)):
            mid = 0.5 * (lo + hi)
            pt = (p0 + (direction * mid) * n)[None, :]
            v = float(_sample_mask_linear(mask, pt)[0])
            if v >= 0.5:
                lo = mid
            else:
                hi = mid
        return float(direction * (0.5 * (lo + hi)))
    return float("nan")


def _build_slice_columns(
    *,
    slice_index: int,
    path_yx: np.ndarray,
    mask_yx: np.ndarray,
    res_y_um: float,
    res_x_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build midline columns for one slice.

    Notes on `t`:
    - `t` is `t_all`: normalized arc length along the full u=0.5 curve on this slice.
    """
    y, x, t = _normalized_arc_t_for_path(path_yx)

    dt = np.gradient(t)
    dy = np.gradient(y)
    dx = np.gradient(x)
    safe = np.abs(dt) > 1.0e-9
    dy_dt = np.divide(dy, dt, out=np.zeros_like(dy), where=safe)
    dx_dt = np.divide(dx, dt, out=np.zeros_like(dx), where=safe)

    ny = -dx_dt
    nx = dy_dt
    n_norm = np.sqrt(ny * ny + nx * nx)
    ok = n_norm > 0.0
    ny = np.divide(ny, n_norm, out=np.zeros_like(ny), where=ok)
    nx = np.divide(nx, n_norm, out=np.zeros_like(nx), where=ok)
    for idx in range(1, y.shape[0]):
        if not np.isfinite(ny[idx - 1]) or not np.isfinite(nx[idx - 1]):
            continue
        if not np.isfinite(ny[idx]) or not np.isfinite(nx[idx]):
            continue
        if (ny[idx - 1] * ny[idx] + nx[idx - 1] * nx[idx]) < 0.0:
            ny[idx] = -ny[idx]
            nx[idx] = -nx[idx]

    t_neg = np.full((y.shape[0],), np.nan, dtype=np.float64)
    t_pos = np.full((y.shape[0],), np.nan, dtype=np.float64)
    max_t = float(np.hypot(*mask_yx.shape)) + 4.0
    for idx in range(y.shape[0]):
        p = np.array([y[idx], x[idx]], dtype=np.float64)
        n = np.array([ny[idx], nx[idx]], dtype=np.float64)
        if not np.isfinite(n).all() or float(np.linalg.norm(n)) <= 0.0:
            continue
        t_pos[idx] = _march_to_boundary_along_normal(mask_yx, p_yx=p, n_yx=n, direction=1.0, max_t=max_t, step=0.5)
        t_neg[idx] = _march_to_boundary_along_normal(mask_yx, p_yx=p, n_yx=n, direction=-1.0, max_t=max_t, step=0.5)

    midline_is_high_x = True
    init_prefer_b0_as_vent: bool | None = None
    valid = np.isfinite(t_neg) & np.isfinite(t_pos) & (t_pos > t_neg) & np.isfinite(nx)
    if np.any(valid):
        b0_x = x[valid] + t_neg[valid] * nx[valid]
        b1_x = x[valid] + t_pos[valid] * nx[valid]
        med_dx = float(np.nanmedian(b0_x - b1_x))
        if np.isfinite(med_dx) and abs(med_dx) >= 0.25:
            init_prefer_b0_as_vent = bool(med_dx >= 0.0) if midline_is_high_x else bool(med_dx < 0.0)

    vent_yx = np.full((y.shape[0], 2), np.nan, dtype=np.float64)
    pia_yx = np.full((y.shape[0], 2), np.nan, dtype=np.float64)
    prev_vent: np.ndarray | None = None
    prev_pia: np.ndarray | None = None

    for idx in range(y.shape[0]):
        if not np.isfinite(t_neg[idx]) or not np.isfinite(t_pos[idx]):
            continue
        if float(t_pos[idx]) <= float(t_neg[idx]):
            continue

        p = np.array([y[idx], x[idx]], dtype=np.float64)
        n = np.array([ny[idx], nx[idx]], dtype=np.float64)
        b0 = p + t_neg[idx] * n
        b1 = p + t_pos[idx] * n

        if prev_vent is None or prev_pia is None:
            prefer_b0 = init_prefer_b0_as_vent
            if prefer_b0 is None:
                prefer_b0 = bool(float(b0[1]) >= float(b1[1])) if midline_is_high_x else bool(float(b0[1]) < float(b1[1]))
            vent, pia = (b0, b1) if prefer_b0 else (b1, b0)
        else:
            c_keep = float(np.sum((b0 - prev_vent) ** 2) + np.sum((b1 - prev_pia) ** 2))
            c_swap = float(np.sum((b1 - prev_vent) ** 2) + np.sum((b0 - prev_pia) ** 2))
            vent, pia = (b0, b1) if c_keep <= c_swap else (b1, b0)

        vent_yx[idx] = vent
        pia_yx[idx] = pia
        prev_vent = vent
        prev_pia = pia

    vent_x = vent_yx[:, 1]
    pia_x = pia_yx[:, 1]
    finite = np.isfinite(vent_x) & np.isfinite(pia_x)
    if int(np.sum(finite)) >= 20:
        med_dx = float(np.nanmedian(vent_x[finite] - pia_x[finite]))
        if np.isfinite(med_dx) and abs(med_dx) >= 0.25:
            wants_positive = midline_is_high_x
            if (med_dx < 0.0) == wants_positive:
                vent_yx, pia_yx = pia_yx, vent_yx

    thickness_um = np.sqrt(
        ((vent_yx[:, 0] - pia_yx[:, 0]) * float(res_y_um)) ** 2 + ((vent_yx[:, 1] - pia_yx[:, 1]) * float(res_x_um)) ** 2
    )
    return t, y, x, vent_yx, pia_yx, thickness_um


def _save_coronal_midline_columns_csv(csv_path: Path, columns: dict[int, CoronalMidlineColumns]) -> None:
    rows: list[np.ndarray] = []
    for slice_i in sorted(columns.keys()):
        c = columns[slice_i]
        rows.append(
            np.column_stack([np.full_like(c.t, slice_i, dtype=np.int32), c.t, c.y, c.x, c.vent_y, c.vent_x, c.pia_y, c.pia_x, c.thickness_um])
        )
    table = np.vstack(rows).astype(np.float64, copy=False)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="slice_i,t,y,x,vent_y,vent_x,pia_y,pia_x,thickness_um",
        comments="",
    )


def _load_coronal_midline_columns_csv(csv_path: Path) -> dict[int, CoronalMidlineColumns]:
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError(f"Unexpected coronal columns CSV shape in {csv_path}: {table.shape}")
    keys = np.asarray(np.rint(table[:, 0]), dtype=np.int32)
    out: dict[int, CoronalMidlineColumns] = {}
    for key in sorted(np.unique(keys).tolist()):
        rows = table[keys == int(key)]
        order = np.argsort(rows[:, 1])
        rows = rows[order]
        out[int(key)] = CoronalMidlineColumns(
            slice_i=int(key),
            t=np.asarray(rows[:, 1], dtype=np.float64),
            y=np.asarray(rows[:, 2], dtype=np.float64),
            x=np.asarray(rows[:, 3], dtype=np.float64),
            vent_y=np.asarray(rows[:, 4], dtype=np.float64),
            vent_x=np.asarray(rows[:, 5], dtype=np.float64),
            pia_y=np.asarray(rows[:, 6], dtype=np.float64),
            pia_x=np.asarray(rows[:, 7], dtype=np.float64),
            thickness_um=np.asarray(rows[:, 8], dtype=np.float64),
        )
    return out


def _save_sagittal_midline_columns_csv(csv_path: Path, columns: dict[int, SagittalMidlineColumns]) -> None:
    rows: list[np.ndarray] = []
    for slice_k in sorted(columns.keys()):
        c = columns[slice_k]
        rows.append(
            np.column_stack([np.full_like(c.t, slice_k, dtype=np.int32), c.t, c.y, c.x, c.vent_y, c.vent_x, c.pia_y, c.pia_x, c.thickness_um])
        )
    table = np.vstack(rows).astype(np.float64, copy=False)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="slice_k,t,y,x,vent_y,vent_x,pia_y,pia_x,thickness_um",
        comments="",
    )


def _load_sagittal_midline_columns_csv(csv_path: Path) -> dict[int, SagittalMidlineColumns]:
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError(f"Unexpected sagittal columns CSV shape in {csv_path}: {table.shape}")
    keys = np.asarray(np.rint(table[:, 0]), dtype=np.int32)
    out: dict[int, SagittalMidlineColumns] = {}
    for key in sorted(np.unique(keys).tolist()):
        rows = table[keys == int(key)]
        order = np.argsort(rows[:, 1])
        rows = rows[order]
        out[int(key)] = SagittalMidlineColumns(
            slice_k=int(key),
            t=np.asarray(rows[:, 1], dtype=np.float64),
            y=np.asarray(rows[:, 2], dtype=np.float64),
            x=np.asarray(rows[:, 3], dtype=np.float64),
            vent_y=np.asarray(rows[:, 4], dtype=np.float64),
            vent_x=np.asarray(rows[:, 5], dtype=np.float64),
            pia_y=np.asarray(rows[:, 6], dtype=np.float64),
            pia_x=np.asarray(rows[:, 7], dtype=np.float64),
            thickness_um=np.asarray(rows[:, 8], dtype=np.float64),
        )
    return out


def _collect_prepared_paths_and_raw_t_anchor_ranges(
    *,
    axis: Literal["coronal", "sagittal"],
    outdir: Path,
    u_3d: np.ndarray,
    mask_3d: np.ndarray,
    include_3d: np.ndarray,
    overlay_3d: np.ndarray,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[int, tuple[float, float]]]:
    prepared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    raw_anchor_ranges: dict[int, tuple[float, float]] = {}
    if axis == "coronal":
        for slice_i in range(u_3d.shape[0]):
            mask_yx = (mask_3d[slice_i, :, :] & include_3d[slice_i, :, :]).astype(bool, copy=False)
            if not np.any(mask_yx):
                continue
            path = _load_manual_coronal_override_path(outdir, slice_i=int(slice_i))
            if path is None:
                path = _extract_largest_midline_contour(u_3d[slice_i, :, :], mask_yx)
            if path is None:
                continue
            overlap_yx = overlay_3d[slice_i, :, :]
            prepared[int(slice_i)] = (
                np.asarray(path, dtype=np.float64),
                np.asarray(mask_yx, dtype=bool),
                np.asarray(overlap_yx, dtype=bool),
            )
            anchor = _raw_t_anchor_range_for_path(path, overlap_yx)
            if anchor is not None:
                raw_anchor_ranges[int(slice_i)] = (float(anchor[0]), float(anchor[1]))
        return prepared, raw_anchor_ranges

    for slice_k in range(u_3d.shape[2]):
        mask_yx = (mask_3d[:, :, slice_k] & include_3d[:, :, slice_k]).astype(bool, copy=False)
        if not np.any(mask_yx):
            continue
        path = _load_manual_sagittal_override_path(outdir, slice_k=int(slice_k))
        if path is None:
            path = _extract_largest_midline_contour(u_3d[:, :, slice_k], mask_yx)
        if path is None:
            continue
        overlap_yx = overlay_3d[:, :, slice_k]
        prepared[int(slice_k)] = (
            np.asarray(path, dtype=np.float64),
            np.asarray(mask_yx, dtype=bool),
            np.asarray(overlap_yx, dtype=bool),
        )
        anchor = _raw_t_anchor_range_for_path(path, overlap_yx)
        if anchor is not None:
            raw_anchor_ranges[int(slice_k)] = (float(anchor[0]), float(anchor[1]))
    return prepared, raw_anchor_ranges


def _build_path_only_coronal_columns_from_prepared(
    prepared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[int, CoronalMidlineColumns]:
    out: dict[int, CoronalMidlineColumns] = {}
    for slice_i in sorted(prepared.keys()):
        path_yx, _mask_yx, _overlap_yx = prepared[int(slice_i)]
        y, x, t = _normalized_arc_t_for_path(path_yx)
        nan_arr = np.full_like(t, np.nan, dtype=np.float64)
        out[int(slice_i)] = CoronalMidlineColumns(
            slice_i=int(slice_i),
            t=np.asarray(t, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            x=np.asarray(x, dtype=np.float64),
            vent_y=np.array(nan_arr, dtype=np.float64, copy=True),
            vent_x=np.array(nan_arr, dtype=np.float64, copy=True),
            pia_y=np.array(nan_arr, dtype=np.float64, copy=True),
            pia_x=np.array(nan_arr, dtype=np.float64, copy=True),
            thickness_um=np.array(nan_arr, dtype=np.float64, copy=True),
        )
    return out


def _build_path_only_sagittal_columns_from_prepared(
    prepared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[int, SagittalMidlineColumns]:
    out: dict[int, SagittalMidlineColumns] = {}
    for slice_k in sorted(prepared.keys()):
        path_yx, _mask_yx, _overlap_yx = prepared[int(slice_k)]
        y, x, t = _normalized_arc_t_for_path(path_yx)
        nan_arr = np.full_like(t, np.nan, dtype=np.float64)
        out[int(slice_k)] = SagittalMidlineColumns(
            slice_k=int(slice_k),
            t=np.asarray(t, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            x=np.asarray(x, dtype=np.float64),
            vent_y=np.array(nan_arr, dtype=np.float64, copy=True),
            vent_x=np.array(nan_arr, dtype=np.float64, copy=True),
            pia_y=np.array(nan_arr, dtype=np.float64, copy=True),
            pia_x=np.array(nan_arr, dtype=np.float64, copy=True),
            thickness_um=np.array(nan_arr, dtype=np.float64, copy=True),
        )
    return out


def _build_coronal_midline_columns_from_halfway_u(outdir: Path) -> dict[int, CoronalMidlineColumns]:
    u_3d, mask_3d, include_3d = _load_halfway_u_and_masks(outdir)
    overlay_3d = _load_neocortex_mesocortex_overlay_mask(outdir, fallback_mask=include_3d)
    res = _load_resolution_ds_ijk_um(outdir) or (1.0, 1.0, 1.0)
    res_j_um = float(res[1])
    res_k_um = float(res[2])
    t0 = time.perf_counter()
    n_slices = int(u_3d.shape[0])
    print(f"[midline] deriving coronal columns from halfway_u: slices={n_slices}")

    prepared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    raw_anchor_ranges: dict[int, tuple[float, float]] = {}
    for slice_i in range(u_3d.shape[0]):
        if slice_i % 50 == 0 or slice_i == (n_slices - 1):
            print(f"[midline] coronal progress {slice_i + 1}/{n_slices}")
        mask_yx = (mask_3d[slice_i, :, :] & include_3d[slice_i, :, :]).astype(bool, copy=False)
        if not np.any(mask_yx):
            continue
        path = _load_manual_coronal_override_path(outdir, slice_i=int(slice_i))
        if path is None:
            path = _extract_largest_midline_contour(u_3d[slice_i, :, :], mask_yx)
        if path is None:
            continue
        overlap_yx = overlay_3d[slice_i, :, :]
        prepared[int(slice_i)] = (
            np.asarray(path, dtype=np.float64),
            np.asarray(mask_yx, dtype=bool),
            np.asarray(overlap_yx, dtype=bool),
        )
        anchor = _raw_t_anchor_range_for_path(path, overlap_yx)
        if anchor is not None:
            raw_anchor_ranges[int(slice_i)] = (float(anchor[0]), float(anchor[1]))

    if not prepared:
        raise ValueError(
            f"Could not derive coronal midline columns from halfway_u_3d_ds.npy in {outdir}. "
            "Expected u=0.5 contours inside the cortex include-mask."
        )

    sagittal_prepared, sagittal_raw_anchor_ranges = _collect_prepared_paths_and_raw_t_anchor_ranges(
        axis="sagittal",
        outdir=outdir,
        u_3d=u_3d,
        mask_3d=mask_3d,
        include_3d=include_3d,
        overlay_3d=overlay_3d,
    )
    anchor_ranges, _sagittal_anchor_ranges_joint, shared_anchor = _smooth_t_anchor_ranges_joint(
        coronal_raw_anchors=raw_anchor_ranges,
        sagittal_raw_anchors=sagittal_raw_anchor_ranges,
        coronal_window_slices=int(CORONAL_T_ANCHOR_SMOOTH_WINDOW_SLICES),
        sagittal_window_slices=int(SAGITTAL_T_ANCHOR_SMOOTH_WINDOW_SLICES),
        sagittal_exclude_slice_keys={int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS},
        shared_weight=float(JOINT_T_ANCHOR_SHARED_WEIGHT),
    )
    out: dict[int, CoronalMidlineColumns] = {}
    for slice_i in sorted(prepared.keys()):
        path_yx, mask_yx, _overlap_yx = prepared[int(slice_i)]
        t, y, x, vent_yx, pia_yx, thickness_um = _build_slice_columns(
            slice_index=int(slice_i),
            path_yx=path_yx,
            mask_yx=mask_yx,
            res_y_um=res_j_um,
            res_x_um=res_k_um,
        )
        out[int(slice_i)] = CoronalMidlineColumns(
            slice_i=int(slice_i),
            t=t,
            y=y,
            x=x,
            vent_y=vent_yx[:, 0],
            vent_x=vent_yx[:, 1],
            pia_y=pia_yx[:, 0],
            pia_x=pia_yx[:, 1],
            thickness_um=thickness_um,
        )

    shared_label = "none" if shared_anchor is None else f"({float(shared_anchor[0]):.4f},{float(shared_anchor[1]):.4f})"
    print(
        f"[midline] coronal overlap-anchor stats (diagnostic only; t_all is not remapped): "
        f"raw_coronal={len(raw_anchor_ranges)} raw_sagittal={len(sagittal_raw_anchor_ranges)} "
        f"smoothed_coronal={len(anchor_ranges)} "
        f"window_coronal={int(CORONAL_T_ANCHOR_SMOOTH_WINDOW_SLICES)} "
        f"window_sagittal={int(SAGITTAL_T_ANCHOR_SMOOTH_WINDOW_SLICES)} "
        f"shared_weight={float(JOINT_T_ANCHOR_SHARED_WEIGHT):.2f} shared={shared_label}"
    )
    if int(CORONAL_T_SMOOTH_WINDOW_SLICES) > 1:
        out = _smooth_coronal_midline_xy_across_slices(
            out,
            window_slices=int(CORONAL_T_SMOOTH_WINDOW_SLICES),
            n_t=int(CORONAL_T_SMOOTH_N_T),
        )
        print(
            f"[midline] coronal cross-slice smoothing applied: "
            f"window={int(CORONAL_T_SMOOTH_WINDOW_SLICES)} n_t={int(CORONAL_T_SMOOTH_N_T)}"
        )
    joint_window = int(max(int(CORONAL_T_SMOOTH_WINDOW_SLICES), int(SAGITTAL_T_SMOOTH_WINDOW_SLICES)))
    joint_n_t = int(max(int(CORONAL_T_SMOOTH_N_T), int(SAGITTAL_T_SMOOTH_N_T)))
    if joint_window > 1 and float(JOINT_XY_SHARED_J_WEIGHT) > 0.0 and sagittal_prepared:
        sagittal_path_columns = _build_path_only_sagittal_columns_from_prepared(sagittal_prepared)
        out, _ = _smooth_shared_j_coordinate_across_axes(
            coronal_columns=out,
            sagittal_columns=sagittal_path_columns,
            window_slices=joint_window,
            n_t=joint_n_t,
            sagittal_exclude_slice_keys={int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS},
            shared_weight=float(JOINT_XY_SHARED_J_WEIGHT),
        )
        print(
            f"[midline] coronal cross-axis shared-j smoothing applied: "
            f"window={joint_window} n_t={joint_n_t} "
            f"peer_rows={len(sagittal_path_columns)} "
            f"weight={float(JOINT_XY_SHARED_J_WEIGHT):.2f}"
        )
    _save_coronal_midline_columns_csv(outdir / "coronal_midline_columns.csv", out)
    dt = time.perf_counter() - t0
    print(f"[midline] coronal done: kept_slices={len(out)} elapsed_s={dt:.1f}")
    return out


def _build_sagittal_midline_columns_from_halfway_u(outdir: Path) -> dict[int, SagittalMidlineColumns]:
    u_3d, mask_3d, include_3d = _load_halfway_u_and_masks(outdir)
    overlay_3d = _load_neocortex_mesocortex_overlay_mask(outdir, fallback_mask=include_3d)
    res = _load_resolution_ds_ijk_um(outdir) or (1.0, 1.0, 1.0)
    res_i_um = float(res[0])
    res_j_um = float(res[1])
    t0 = time.perf_counter()
    n_slices = int(u_3d.shape[2])
    print(f"[midline] deriving sagittal columns from halfway_u: slices={n_slices}")

    prepared: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    raw_anchor_ranges: dict[int, tuple[float, float]] = {}
    for slice_k in range(u_3d.shape[2]):
        if slice_k % 50 == 0 or slice_k == (n_slices - 1):
            print(f"[midline] sagittal progress {slice_k + 1}/{n_slices}")
        mask_yx = (mask_3d[:, :, slice_k] & include_3d[:, :, slice_k]).astype(bool, copy=False)
        if not np.any(mask_yx):
            continue
        path = _load_manual_sagittal_override_path(outdir, slice_k=int(slice_k))
        if path is None:
            path = _extract_largest_midline_contour(u_3d[:, :, slice_k], mask_yx)
        if path is None:
            continue
        overlap_yx = overlay_3d[:, :, slice_k]
        prepared[int(slice_k)] = (
            np.asarray(path, dtype=np.float64),
            np.asarray(mask_yx, dtype=bool),
            np.asarray(overlap_yx, dtype=bool),
        )
        anchor = _raw_t_anchor_range_for_path(path, overlap_yx)
        if anchor is not None:
            raw_anchor_ranges[int(slice_k)] = (float(anchor[0]), float(anchor[1]))

    if not prepared:
        raise ValueError(
            f"Could not derive sagittal midline columns from halfway_u_3d_ds.npy in {outdir}. "
            "Expected u=0.5 contours inside the cortex include-mask."
        )

    coronal_prepared, coronal_raw_anchor_ranges = _collect_prepared_paths_and_raw_t_anchor_ranges(
        axis="coronal",
        outdir=outdir,
        u_3d=u_3d,
        mask_3d=mask_3d,
        include_3d=include_3d,
        overlay_3d=overlay_3d,
    )
    _coronal_anchor_ranges_joint, anchor_ranges, shared_anchor = _smooth_t_anchor_ranges_joint(
        coronal_raw_anchors=coronal_raw_anchor_ranges,
        sagittal_raw_anchors=raw_anchor_ranges,
        coronal_window_slices=int(CORONAL_T_ANCHOR_SMOOTH_WINDOW_SLICES),
        sagittal_window_slices=int(SAGITTAL_T_ANCHOR_SMOOTH_WINDOW_SLICES),
        sagittal_exclude_slice_keys={int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS},
        shared_weight=float(JOINT_T_ANCHOR_SHARED_WEIGHT),
    )
    out: dict[int, SagittalMidlineColumns] = {}
    for slice_k in sorted(prepared.keys()):
        path_yx, mask_yx, _overlap_yx = prepared[int(slice_k)]
        t, y, x, vent_yx, pia_yx, thickness_um = _build_slice_columns(
            slice_index=int(slice_k),
            path_yx=path_yx,
            mask_yx=mask_yx,
            res_y_um=res_i_um,
            res_x_um=res_j_um,
        )
        out[int(slice_k)] = SagittalMidlineColumns(
            slice_k=int(slice_k),
            t=t,
            y=y,
            x=x,
            vent_y=vent_yx[:, 0],
            vent_x=vent_yx[:, 1],
            pia_y=pia_yx[:, 0],
            pia_x=pia_yx[:, 1],
            thickness_um=thickness_um,
        )

    shared_label = "none" if shared_anchor is None else f"({float(shared_anchor[0]):.4f},{float(shared_anchor[1]):.4f})"
    print(
        f"[midline] sagittal overlap-anchor stats (diagnostic only; t_all is not remapped): "
        f"raw_sagittal={len(raw_anchor_ranges)} raw_coronal={len(coronal_raw_anchor_ranges)} "
        f"smoothed_sagittal={len(anchor_ranges)} "
        f"window_sagittal={int(SAGITTAL_T_ANCHOR_SMOOTH_WINDOW_SLICES)} "
        f"window_coronal={int(CORONAL_T_ANCHOR_SMOOTH_WINDOW_SLICES)} "
        f"exclude={tuple(int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS)} "
        f"shared_weight={float(JOINT_T_ANCHOR_SHARED_WEIGHT):.2f} shared={shared_label}"
    )
    if int(SAGITTAL_T_SMOOTH_WINDOW_SLICES) > 1:
        out = _smooth_sagittal_midline_xy_across_slices(
            out,
            window_slices=int(SAGITTAL_T_SMOOTH_WINDOW_SLICES),
            n_t=int(SAGITTAL_T_SMOOTH_N_T),
            exclude_slice_keys={int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS},
        )
        print(
            f"[midline] sagittal cross-slice smoothing applied: "
            f"window={int(SAGITTAL_T_SMOOTH_WINDOW_SLICES)} n_t={int(SAGITTAL_T_SMOOTH_N_T)} "
            f"exclude={tuple(int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS)}"
        )
    joint_window = int(max(int(CORONAL_T_SMOOTH_WINDOW_SLICES), int(SAGITTAL_T_SMOOTH_WINDOW_SLICES)))
    joint_n_t = int(max(int(CORONAL_T_SMOOTH_N_T), int(SAGITTAL_T_SMOOTH_N_T)))
    if joint_window > 1 and float(JOINT_XY_SHARED_J_WEIGHT) > 0.0 and coronal_prepared:
        coronal_path_columns = _build_path_only_coronal_columns_from_prepared(coronal_prepared)
        _, out = _smooth_shared_j_coordinate_across_axes(
            coronal_columns=coronal_path_columns,
            sagittal_columns=out,
            window_slices=joint_window,
            n_t=joint_n_t,
            sagittal_exclude_slice_keys={int(k) for k in SAGITTAL_T_SMOOTH_EXCLUDE_SLICE_KS},
            shared_weight=float(JOINT_XY_SHARED_J_WEIGHT),
        )
        print(
            f"[midline] sagittal cross-axis shared-j smoothing applied: "
            f"window={joint_window} n_t={joint_n_t} "
            f"peer_rows={len(coronal_path_columns)} "
            f"weight={float(JOINT_XY_SHARED_J_WEIGHT):.2f}"
        )
    _save_sagittal_midline_columns_csv(outdir / "sagittal_midline_columns.csv", out)
    dt = time.perf_counter() - t0
    print(f"[midline] sagittal done: kept_slices={len(out)} elapsed_s={dt:.1f}")
    return out


def _load_resolution_ds_ijk_um(outdir: Path) -> tuple[float, float, float] | None:
    path = outdir / "resolution_ds_ijk_um.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.shape != (3,):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class EdtCrop:
    origin_ijk: tuple[int, int, int]
    d_pial_um: np.ndarray
    d_inner_um: np.ndarray


def _load_halfway_distance_fields(outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    pial_path = outdir / "halfway_d_pial_um_crop.npy"
    inner_path = outdir / "halfway_d_inner_um_crop.npy"
    has_direct = pial_path.exists() or inner_path.exists()
    if has_direct:
        if not pial_path.exists() or not inner_path.exists():
            raise FileNotFoundError(
                "Expected both halfway_d_pial_um_crop.npy and halfway_d_inner_um_crop.npy when using direct EDT files."
            )
        d_pial = np.load(pial_path).astype(np.float32, copy=False)
        d_inner = np.load(inner_path).astype(np.float32, copy=False)
        if d_pial.shape != d_inner.shape:
            raise ValueError("halfway_d_pial_um_crop.npy and halfway_d_inner_um_crop.npy must have the same shape.")
        return d_pial, d_inner

    r_um_path = outdir / "halfway_r_um_crop.npy"
    thickness_path = outdir / "halfway_thickness_um_crop.npy"
    has_reconstruct = r_um_path.exists() or thickness_path.exists()
    if has_reconstruct:
        if not r_um_path.exists() or not thickness_path.exists():
            raise FileNotFoundError(
                "Expected both halfway_r_um_crop.npy and halfway_thickness_um_crop.npy to reconstruct EDT distances."
            )
        r_um = np.load(r_um_path).astype(np.float32, copy=False)
        thickness_um = np.load(thickness_path).astype(np.float32, copy=False)
        if r_um.shape != thickness_um.shape:
            raise ValueError("halfway_r_um_crop.npy and halfway_thickness_um_crop.npy must have the same shape.")
        d_inner = 0.5 * (thickness_um + r_um)
        d_pial = 0.5 * (thickness_um - r_um)
        finite = np.isfinite(d_inner) & np.isfinite(d_pial)
        d_inner = np.where(finite, np.maximum(d_inner, 0.0), np.nan).astype(np.float32, copy=False)
        d_pial = np.where(finite, np.maximum(d_pial, 0.0), np.nan).astype(np.float32, copy=False)
        return d_pial, d_inner

    raise FileNotFoundError(
        "Missing EDT crop fields. Need either "
        "{halfway_d_pial_um_crop.npy + halfway_d_inner_um_crop.npy} or "
        "{halfway_r_um_crop.npy + halfway_thickness_um_crop.npy} in outdir."
    )


def load_halfway_edt_crop(outdir: Path) -> EdtCrop:
    origin = np.load(outdir / "halfway_crop_origin_ijk.npy").astype(np.int64, copy=False)
    if origin.shape != (3,):
        raise ValueError("halfway_crop_origin_ijk.npy must be shape (3,).")
    d_pial, d_inner = _load_halfway_distance_fields(outdir)
    return EdtCrop(origin_ijk=(int(origin[0]), int(origin[1]), int(origin[2])), d_pial_um=d_pial, d_inner_um=d_inner)


def _nearest_key(keys: list[int], target: int) -> int:
    if not keys:
        raise ValueError("keys is empty.")
    target = int(target)
    arr = np.asarray(keys, dtype=np.int32)
    return int(arr[np.argmin(np.abs(arr - target))])


def _r01_at_ijk(edt: EdtCrop, ijk: tuple[float, float, float]) -> float:
    i0, j0, k0 = edt.origin_ijk
    ic = float(ijk[0]) - float(i0)
    jc = float(ijk[1]) - float(j0)
    kc = float(ijk[2]) - float(k0)
    si, sj, sk = edt.d_pial_um.shape
    if ic < -0.5 or jc < -0.5 or kc < -0.5 or ic > float(si - 0.5) or jc > float(sj - 0.5) or kc > float(sk - 0.5):
        raise ValueError(f"Point {ijk!r} falls outside EDT crop (origin={edt.origin_ijk}, shape={edt.d_pial_um.shape}).")

    coords = np.asarray([[ic], [jc], [kc]], dtype=np.float64)
    d_pial = float(map_coordinates(edt.d_pial_um, coords, order=1, mode="nearest")[0])
    d_inner = float(map_coordinates(edt.d_inner_um, coords, order=1, mode="nearest")[0])
    den = d_pial + d_inner
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(d_inner / den)


def invert_r01_along_segment(
    edt: EdtCrop,
    *,
    vent_ijk: tuple[float, float, float],
    pia_ijk: tuple[float, float, float],
    r01_target: float,
    max_iters: int = 60,
    tol_r01: float = 2.0e-4,
) -> tuple[float, float, float]:
    r01_target = float(r01_target)
    if not (0.0 <= r01_target <= 1.0):
        raise ValueError("r01_target must be in [0,1].")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if tol_r01 <= 0:
        raise ValueError("tol_r01 must be > 0.")

    v = np.asarray(vent_ijk, dtype=np.float64)
    p = np.asarray(pia_ijk, dtype=np.float64)
    if v.shape != (3,) or p.shape != (3,):
        raise ValueError("vent_ijk and pia_ijk must be 3-tuples.")

    r_v = _r01_at_ijk(edt, (float(v[0]), float(v[1]), float(v[2])))
    r_p = _r01_at_ijk(edt, (float(p[0]), float(p[1]), float(p[2])))
    if not np.isfinite(r_v) or not np.isfinite(r_p):
        raise ValueError("EDT r01 was non-finite at vent/pia endpoints (check crop coverage).")

    # Ensure increasing r01 from vent -> pia.
    if r_v > r_p:
        v, p = p, v
        r_v, r_p = r_p, r_v

    if r01_target <= r_v:
        return (float(v[0]), float(v[1]), float(v[2]))
    if r01_target >= r_p:
        return (float(p[0]), float(p[1]), float(p[2]))

    lo = 0.0
    hi = 1.0
    r_lo = r_v
    r_hi = r_p
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        q = v + mid * (p - v)
        r_mid = _r01_at_ijk(edt, (float(q[0]), float(q[1]), float(q[2])))
        if not np.isfinite(r_mid):
            raise ValueError("EDT r01 became non-finite during inversion.")
        if abs(r_mid - r01_target) <= tol_r01:
            return (float(q[0]), float(q[1]), float(q[2]))
        if r_mid < r01_target:
            lo, r_lo = mid, r_mid
        else:
            hi, r_hi = mid, r_mid
        if abs(r_hi - r_lo) <= tol_r01:
            break

    q = v + (0.5 * (lo + hi)) * (p - v)
    return (float(q[0]), float(q[1]), float(q[2]))


@dataclass(frozen=True)
class InvertedAxisCoordinate:
    axis: Literal["coronal", "sagittal"]
    slice_index: int
    t: float
    r01: float
    residual_vox: float


def _optimize_t_fixed_r01_coronal(
    *,
    c: CoronalMidlineColumns,
    ijk_target: tuple[float, float, float],
    r01_target: float,
    edt: EdtCrop,
) -> tuple[float, float]:
    """Find t minimizing ||x(t, r01_target) - ijk_target|| in the coronal plane."""
    target_jk = np.asarray([float(ijk_target[1]), float(ijk_target[2])], dtype=np.float64)

    def _eval_t(t: float) -> tuple[float, float]:
        vent_y = float(_interp(c.t, c.vent_y, t))
        vent_x = float(_interp(c.t, c.vent_x, t))
        pia_y = float(_interp(c.t, c.pia_y, t))
        pia_x = float(_interp(c.t, c.pia_x, t))
        if not np.isfinite([vent_y, vent_x, pia_y, pia_x]).all():
            return float("nan"), float("inf")
        try:
            ijk = invert_r01_along_segment(
                edt,
                vent_ijk=(float(c.slice_i), vent_y, vent_x),
                pia_ijk=(float(c.slice_i), pia_y, pia_x),
                r01_target=float(r01_target),
                max_iters=32,
                tol_r01=5.0e-4,
            )
        except ValueError:
            return float("nan"), float("inf")
        jk = np.asarray([float(ijk[1]), float(ijk[2])], dtype=np.float64)
        d2 = float(np.sum((jk - target_jk) ** 2))
        return float(t), d2

    best_t = 0.5
    best_d2 = float("inf")
    lo = 0.0
    hi = 1.0
    for n_samples in (161, 121, 121):
        ts = np.linspace(lo, hi, n_samples, dtype=np.float64)
        vals: list[tuple[float, float]] = []
        for t in ts.tolist():
            vals.append(_eval_t(float(t)))
        finite = [(t, d2) for t, d2 in vals if np.isfinite(d2)]
        if not finite:
            break
        best_t, best_d2 = min(finite, key=lambda x: x[1])
        step = float((hi - lo) / max(1, n_samples - 1))
        lo = max(0.0, best_t - 4.0 * step)
        hi = min(1.0, best_t + 4.0 * step)

    return float(best_t), float(np.sqrt(best_d2))


def _optimize_t_fixed_r01_sagittal(
    *,
    c: SagittalMidlineColumns,
    ijk_target: tuple[float, float, float],
    r01_target: float,
    edt: EdtCrop,
) -> tuple[float, float]:
    """Find t minimizing ||x(t, r01_target) - ijk_target|| in the sagittal plane."""
    target_ij = np.asarray([float(ijk_target[0]), float(ijk_target[1])], dtype=np.float64)

    def _eval_t(t: float) -> tuple[float, float]:
        vent_y = float(_interp(c.t, c.vent_y, t))
        vent_x = float(_interp(c.t, c.vent_x, t))
        pia_y = float(_interp(c.t, c.pia_y, t))
        pia_x = float(_interp(c.t, c.pia_x, t))
        if not np.isfinite([vent_y, vent_x, pia_y, pia_x]).all():
            return float("nan"), float("inf")
        try:
            ijk = invert_r01_along_segment(
                edt,
                vent_ijk=(vent_y, vent_x, float(c.slice_k)),
                pia_ijk=(pia_y, pia_x, float(c.slice_k)),
                r01_target=float(r01_target),
                max_iters=32,
                tol_r01=5.0e-4,
            )
        except ValueError:
            return float("nan"), float("inf")
        ij = np.asarray([float(ijk[0]), float(ijk[1])], dtype=np.float64)
        d2 = float(np.sum((ij - target_ij) ** 2))
        return float(t), d2

    best_t = 0.5
    best_d2 = float("inf")
    lo = 0.0
    hi = 1.0
    for n_samples in (161, 121, 121):
        ts = np.linspace(lo, hi, n_samples, dtype=np.float64)
        vals: list[tuple[float, float]] = []
        for t in ts.tolist():
            vals.append(_eval_t(float(t)))
        finite = [(t, d2) for t, d2 in vals if np.isfinite(d2)]
        if not finite:
            break
        best_t, best_d2 = min(finite, key=lambda x: x[1])
        step = float((hi - lo) / max(1, n_samples - 1))
        lo = max(0.0, best_t - 4.0 * step)
        hi = min(1.0, best_t + 4.0 * step)

    return float(best_t), float(np.sqrt(best_d2))


def invert_coronal_from_ijk(
    columns_by_slice: dict[int, CoronalMidlineColumns],
    *,
    ijk: tuple[float, float, float],
    edt: EdtCrop,
) -> InvertedAxisCoordinate:
    if not columns_by_slice:
        raise ValueError("columns_by_slice is empty.")
    i = float(ijk[0])
    slice_i = _nearest_key(sorted(columns_by_slice.keys()), int(np.rint(i)))
    c = columns_by_slice[slice_i]

    r01 = float(np.clip(float(_r01_at_ijk(edt, ijk)), 0.0, 1.0))
    t_opt, residual = _optimize_t_fixed_r01_coronal(c=c, ijk_target=ijk, r01_target=r01, edt=edt)
    return InvertedAxisCoordinate(axis="coronal", slice_index=int(slice_i), t=float(t_opt), r01=r01, residual_vox=float(residual))


def invert_sagittal_from_ijk(
    columns_by_slice: dict[int, SagittalMidlineColumns],
    *,
    ijk: tuple[float, float, float],
    edt: EdtCrop,
) -> InvertedAxisCoordinate:
    if not columns_by_slice:
        raise ValueError("columns_by_slice is empty.")
    k = float(ijk[2])
    slice_k = _nearest_key(sorted(columns_by_slice.keys()), int(np.rint(k)))
    c = columns_by_slice[slice_k]

    r01 = float(np.clip(float(_r01_at_ijk(edt, ijk)), 0.0, 1.0))
    t_opt, residual = _optimize_t_fixed_r01_sagittal(c=c, ijk_target=ijk, r01_target=r01, edt=edt)
    return InvertedAxisCoordinate(axis="sagittal", slice_index=int(slice_k), t=float(t_opt), r01=r01, residual_vox=float(residual))


def _nearest_existing(sorted_keys: np.ndarray, key: int) -> int:
    idx = int(np.searchsorted(sorted_keys, int(key)))
    if idx <= 0:
        return int(sorted_keys[0])
    if idx >= int(sorted_keys.size):
        return int(sorted_keys[-1])
    a = int(sorted_keys[idx - 1])
    b = int(sorted_keys[idx])
    return a if abs(int(key) - a) <= abs(int(key) - b) else b


def _project_t_on_polyline(*, y: np.ndarray, x: np.ndarray, t: np.ndarray, qy: float, qx: float) -> tuple[float, float]:
    y = y.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    t = t.astype(np.float64, copy=False)
    if y.ndim != 1 or x.ndim != 1 or t.ndim != 1 or y.size != x.size or y.size != t.size:
        raise ValueError(f"Polyline arrays must be 1D and same length, got {y.shape} {x.shape} {t.shape}.")
    if y.size < 2:
        raise ValueError("Polyline must contain at least 2 points.")

    p = np.column_stack([y, x])
    p0 = p[:-1]
    p1 = p[1:]
    v = p1 - p0
    q = np.asarray([float(qy), float(qx)], dtype=np.float64).reshape(1, 2)
    w = q - p0

    vv = np.sum(v * v, axis=1)
    vv = np.where(vv > 0.0, vv, 1.0)
    alpha = np.clip(np.sum(w * v, axis=1) / vv, 0.0, 1.0)
    proj = p0 + v * alpha[:, None]
    d2 = np.sum((proj - q) ** 2, axis=1)
    m = int(np.argmin(d2))

    t0 = float(t[m])
    t1 = float(t[m + 1])
    t_proj = t0 + float(alpha[m]) * (t1 - t0)
    return float(t_proj), float(d2[m])


def _p3_to_sagittal(
    *,
    ijk: tuple[float, float, float],
    sagittal: dict[int, SagittalMidlineColumns],
    sagittal_keys: np.ndarray,
    k_window: int,
) -> tuple[int, float, float]:
    i = float(ijk[0])
    j = float(ijk[1])
    k = float(ijk[2])
    k_center = int(np.rint(k))

    cand = sorted({_nearest_existing(sagittal_keys, int(k_center + dk)) for dk in range(-int(k_window), int(k_window) + 1)})
    best_d2: float | None = None
    best_slice = int(cand[0])
    best_t = 0.5
    for k_idx in cand:
        cols = sagittal[int(k_idx)]
        t_proj, d2_plane = _project_t_on_polyline(y=cols.y, x=cols.x, t=cols.t, qy=i, qx=j)
        d2 = float(d2_plane + (k - float(k_idx)) ** 2)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_slice = int(k_idx)
            best_t = float(t_proj)

    return int(best_slice), float(best_t), float(np.sqrt(best_d2)) if best_d2 is not None else float("nan")


def _p3_to_coronal(
    *,
    ijk: tuple[float, float, float],
    coronal: dict[int, CoronalMidlineColumns],
    coronal_keys: np.ndarray,
    i_window: int,
) -> tuple[int, float, float]:
    i = float(ijk[0])
    j = float(ijk[1])
    k = float(ijk[2])
    i_center = int(np.rint(i))

    cand = sorted({_nearest_existing(coronal_keys, int(i_center + di)) for di in range(-int(i_window), int(i_window) + 1)})
    best_d2: float | None = None
    best_slice = int(cand[0])
    best_t = 0.5
    for i_idx in cand:
        cols = coronal[int(i_idx)]
        t_proj, d2_plane = _project_t_on_polyline(y=cols.y, x=cols.x, t=cols.t, qy=j, qx=k)
        d2 = float(d2_plane + (i - float(i_idx)) ** 2)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_slice = int(i_idx)
            best_t = float(t_proj)

    return int(best_slice), float(best_t), float(np.sqrt(best_d2)) if best_d2 is not None else float("nan")


def build_transition_lut_2d(
    *,
    outdir: Path,
    n_t: int = 1024,
    i_window: int = 2,
    k_window: int = 2,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if n_t < 2:
        raise ValueError("n_t must be >= 2.")
    if i_window < 0 or k_window < 0:
        raise ValueError("i_window and k_window must be >= 0.")

    s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    c2s_path = outdir / "chart_map_coronal_to_sagittal_t2d.npz"
    if not overwrite and s2c_path.exists() and c2s_path.exists():
        try:
            s2c = np.load(s2c_path)
            c2s = np.load(c2s_path)
            s2c_nt = int(np.asarray(s2c["t_grid"]).shape[0])
            c2s_nt = int(np.asarray(c2s["t_grid"]).shape[0])
            s2c_iw = int(np.asarray(s2c["i_window"]).reshape(-1)[0])
            s2c_kw = int(np.asarray(s2c["k_window"]).reshape(-1)[0])
            c2s_iw = int(np.asarray(c2s["i_window"]).reshape(-1)[0])
            c2s_kw = int(np.asarray(c2s["k_window"]).reshape(-1)[0])
            s2c_t_domain = str(np.asarray(s2c["t_domain"]).reshape(-1)[0])
            c2s_t_domain = str(np.asarray(c2s["t_domain"]).reshape(-1)[0])
            if (
                s2c_nt == int(n_t)
                and c2s_nt == int(n_t)
                and s2c_iw == int(i_window)
                and c2s_iw == int(i_window)
                and s2c_kw == int(k_window)
                and c2s_kw == int(k_window)
                and s2c_t_domain == "t_all"
                and c2s_t_domain == "t_all"
            ):
                return s2c_path, c2s_path
        except (KeyError, ValueError, IndexError):
            pass

    coronal = load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    sagittal = load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
    coronal_keys_all = np.asarray(sorted(coronal.keys()), dtype=np.int32)
    sagittal_keys_all = np.asarray(sorted(sagittal.keys()), dtype=np.int32)
    coronal_nonzero = _nonzero_include_slice_keys(outdir, axis="coronal")
    sagittal_nonzero = _nonzero_include_slice_keys(outdir, axis="sagittal")
    coronal_keys = np.intersect1d(coronal_keys_all, coronal_nonzero, assume_unique=False)
    sagittal_keys = np.intersect1d(sagittal_keys_all, sagittal_nonzero, assume_unique=False)
    if coronal_keys.size == 0 or sagittal_keys.size == 0:
        raise ValueError("Cannot build LUT with empty coronal or sagittal columns.")
    t0 = time.perf_counter()
    print(
        f"[lut] building 2D chart LUTs: coronal_rows={coronal_keys.size} sagittal_rows={sagittal_keys.size} n_t={int(n_t)}"
    )

    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)

    map_s2c_slice = np.full((sagittal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    map_s2c_t = np.full((sagittal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    map_s2c_err = np.full((sagittal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    for row, slice_k in enumerate(sagittal_keys.tolist()):
        if row % 16 == 0 or row == (sagittal_keys.size - 1):
            print(f"[lut] s2c rows {row + 1}/{sagittal_keys.size}")
        cols = sagittal[int(slice_k)]
        try:
            src_t_min, src_t_max = _column_t_affine_bounds(cols.t)
        except ValueError:
            continue
        src_t_den = src_t_max - src_t_min
        for col, t_val in enumerate(t_grid.tolist()):
            t_col = float(src_t_min + float(t_val) * src_t_den)
            i = float(_interp(cols.t, cols.y, t_col))
            j = float(_interp(cols.t, cols.x, t_col))
            i_idx, t_c, err = _p3_to_coronal(
                ijk=(i, j, float(slice_k)),
                coronal=coronal,
                coronal_keys=coronal_keys,
                i_window=int(i_window),
            )
            map_s2c_slice[row, col] = float(i_idx)
            try:
                map_s2c_t[row, col] = float(_column_t_to_t_all(t=float(t_c), t_values=coronal[int(i_idx)].t))
            except ValueError:
                continue
            map_s2c_err[row, col] = float(err)

    valid_s2c = np.ones((sagittal_keys.size,), dtype=bool)
    for row in range(sagittal_keys.size):
        map_s2c_slice[row] = _fill_nan_series(map_s2c_slice[row]).astype(np.float32, copy=False)
        map_s2c_t[row] = _fill_nan_series(map_s2c_t[row]).astype(np.float32, copy=False)
        map_s2c_err[row] = _fill_nan_series(map_s2c_err[row]).astype(np.float32, copy=False)
        if not (
            np.isfinite(map_s2c_slice[row]).all()
            and np.isfinite(map_s2c_t[row]).all()
            and np.isfinite(map_s2c_err[row]).all()
        ):
            valid_s2c[row] = False
    if not np.all(valid_s2c):
        dropped = int(np.sum(~valid_s2c))
        print(f"[lut] s2c dropping invalid rows={dropped}")
        sagittal_keys = sagittal_keys[valid_s2c]
        map_s2c_slice = map_s2c_slice[valid_s2c]
        map_s2c_t = map_s2c_t[valid_s2c]
        map_s2c_err = map_s2c_err[valid_s2c]
    if sagittal_keys.size == 0:
        raise ValueError("No valid sagittal rows left after NaN sanitization.")

    np.savez_compressed(
        s2c_path,
        source_axis="sagittal",
        target_axis="coronal",
        source_slice_keys=sagittal_keys,
        target_slice_idx=map_s2c_slice,
        target_t=map_s2c_t,
        residual_vox=map_s2c_err,
        t_grid=t_grid.astype(np.float32),
        t_domain=np.asarray(["t_all"]),
        i_window=np.asarray([int(i_window)], dtype=np.int32),
        k_window=np.asarray([int(k_window)], dtype=np.int32),
    )

    map_c2s_slice = np.full((coronal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    map_c2s_t = np.full((coronal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    map_c2s_err = np.full((coronal_keys.size, t_grid.size), np.nan, dtype=np.float32)
    for row, slice_i in enumerate(coronal_keys.tolist()):
        if row % 16 == 0 or row == (coronal_keys.size - 1):
            print(f"[lut] c2s rows {row + 1}/{coronal_keys.size}")
        cols = coronal[int(slice_i)]
        try:
            src_t_min, src_t_max = _column_t_affine_bounds(cols.t)
        except ValueError:
            continue
        src_t_den = src_t_max - src_t_min
        for col, t_val in enumerate(t_grid.tolist()):
            t_col = float(src_t_min + float(t_val) * src_t_den)
            j = float(_interp(cols.t, cols.y, t_col))
            k = float(_interp(cols.t, cols.x, t_col))
            k_idx, t_s, err = _p3_to_sagittal(
                ijk=(float(slice_i), j, k),
                sagittal=sagittal,
                sagittal_keys=sagittal_keys,
                k_window=int(k_window),
            )
            map_c2s_slice[row, col] = float(k_idx)
            try:
                map_c2s_t[row, col] = float(_column_t_to_t_all(t=float(t_s), t_values=sagittal[int(k_idx)].t))
            except ValueError:
                continue
            map_c2s_err[row, col] = float(err)

    valid_c2s = np.ones((coronal_keys.size,), dtype=bool)
    for row in range(coronal_keys.size):
        map_c2s_slice[row] = _fill_nan_series(map_c2s_slice[row]).astype(np.float32, copy=False)
        map_c2s_t[row] = _fill_nan_series(map_c2s_t[row]).astype(np.float32, copy=False)
        map_c2s_err[row] = _fill_nan_series(map_c2s_err[row]).astype(np.float32, copy=False)
        if not (
            np.isfinite(map_c2s_slice[row]).all()
            and np.isfinite(map_c2s_t[row]).all()
            and np.isfinite(map_c2s_err[row]).all()
        ):
            valid_c2s[row] = False
    if not np.all(valid_c2s):
        dropped = int(np.sum(~valid_c2s))
        print(f"[lut] c2s dropping invalid rows={dropped}")
        coronal_keys = coronal_keys[valid_c2s]
        map_c2s_slice = map_c2s_slice[valid_c2s]
        map_c2s_t = map_c2s_t[valid_c2s]
        map_c2s_err = map_c2s_err[valid_c2s]
    if coronal_keys.size == 0:
        raise ValueError("No valid coronal rows left after NaN sanitization.")

    np.savez_compressed(
        c2s_path,
        source_axis="coronal",
        target_axis="sagittal",
        source_slice_keys=coronal_keys,
        target_slice_idx=map_c2s_slice,
        target_t=map_c2s_t,
        residual_vox=map_c2s_err,
        t_grid=t_grid.astype(np.float32),
        t_domain=np.asarray(["t_all"]),
        i_window=np.asarray([int(i_window)], dtype=np.int32),
        k_window=np.asarray([int(k_window)], dtype=np.int32),
    )
    dt = time.perf_counter() - t0
    print(f"[lut] 2D chart LUTs done: elapsed_s={dt:.1f}")
    return s2c_path, c2s_path


def _interp_row_with_nans(*, t_grid: np.ndarray, row: np.ndarray, t: float) -> float:
    finite = np.isfinite(row)
    if not np.any(finite):
        return float("nan")
    t_clip = float(np.clip(float(t), 0.0, 1.0))
    return float(np.interp(t_clip, t_grid[finite], row[finite]))


def transform_with_lut(
    *,
    outdir: Path,
    source_axis: Literal["coronal", "sagittal"],
    source_slice: int,
    t: float,
    r01: float,
    n_t: int = 1024,
    i_window: int = 2,
    k_window: int = 2,
    rebuild_lut: bool = False,
) -> InvertedAxisCoordinate:
    s2c_path, c2s_path = build_transition_lut_2d(
        outdir=outdir,
        n_t=int(n_t),
        i_window=int(i_window),
        k_window=int(k_window),
        overwrite=bool(rebuild_lut),
    )
    lut_path = c2s_path if source_axis == "coronal" else s2c_path
    lut = np.load(lut_path)

    source_keys = np.asarray(lut["source_slice_keys"], dtype=np.int32)
    t_grid = np.asarray(lut["t_grid"], dtype=np.float64)
    target_slice = np.asarray(lut["target_slice_idx"], dtype=np.float64)
    target_t = np.asarray(lut["target_t"], dtype=np.float64)
    residual = np.asarray(lut["residual_vox"], dtype=np.float64)

    row = int(np.argmin(np.abs(source_keys - int(source_slice))))
    slice_float = _interp_row_with_nans(t_grid=t_grid, row=target_slice[row], t=float(t))
    t_out = _interp_row_with_nans(t_grid=t_grid, row=target_t[row], t=float(t))
    residual_out = _interp_row_with_nans(t_grid=t_grid, row=residual[row], t=float(t))
    if not np.isfinite(slice_float) or not np.isfinite(t_out):
        raise ValueError(
            f"LUT lookup failed for source_axis={source_axis}, slice={source_slice}, t={t:.6f}. "
            "Try increasing --lut-nt or --lut-window."
        )

    target_axis: Literal["coronal", "sagittal"] = "sagittal" if source_axis == "coronal" else "coronal"
    return InvertedAxisCoordinate(
        axis=target_axis,
        slice_index=int(np.rint(slice_float)),
        t=float(t_out),
        r01=float(np.clip(float(r01), 0.0, 1.0)),
        residual_vox=float(residual_out) if np.isfinite(residual_out) else float("nan"),
    )


def _build_ijk_segment_lut_for_axis(
    *,
    outdir: Path,
    axis: Literal["coronal", "sagittal"],
    n_t: int,
    overwrite: bool,
) -> Path:
    if axis == "coronal":
        path = outdir / "chart_map_coronal_ijk_from_tr.npz"
    else:
        path = outdir / "chart_map_sagittal_ijk_from_tr.npz"

    if not overwrite and path.exists():
        try:
            d = np.load(path)
            t_domain = str(np.asarray(d["t_domain"]).reshape(-1)[0])
            if int(np.asarray(d["t_grid"]).shape[0]) == int(n_t) and t_domain == "t_all":
                return path
        except (KeyError, ValueError, IndexError):
            pass

    edt = load_halfway_edt_crop(outdir)
    if axis == "coronal":
        cols = load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
        slice_keys_all = np.asarray(sorted(cols.keys()), dtype=np.int32)
    else:
        cols = load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
        slice_keys_all = np.asarray(sorted(cols.keys()), dtype=np.int32)
    slice_keys = np.intersect1d(slice_keys_all, _nonzero_include_slice_keys(outdir, axis=axis), assume_unique=False)

    if slice_keys.size == 0:
        raise ValueError(f"Cannot build {axis} ijk LUT: no slice keys.")
    t0 = time.perf_counter()
    print(f"[lut] building ijk segment LUT axis={axis} rows={slice_keys.size} n_t={int(n_t)}")

    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    low_ijk = np.full((slice_keys.size, t_grid.size, 3), np.nan, dtype=np.float32)
    high_ijk = np.full((slice_keys.size, t_grid.size, 3), np.nan, dtype=np.float32)

    for row, s in enumerate(slice_keys.tolist()):
        if row % 16 == 0 or row == (slice_keys.size - 1):
            print(f"[lut] {axis} ijk rows {row + 1}/{slice_keys.size}")
        c = cols[int(s)]
        try:
            src_t_min, src_t_max = _column_t_affine_bounds(c.t)
        except ValueError:
            continue
        src_t_den = src_t_max - src_t_min
        for col, t_val in enumerate(t_grid.tolist()):
            t_col = float(src_t_min + float(t_val) * src_t_den)
            vent_y = float(_interp(c.t, c.vent_y, t_col))
            vent_x = float(_interp(c.t, c.vent_x, t_col))
            pia_y = float(_interp(c.t, c.pia_y, t_col))
            pia_x = float(_interp(c.t, c.pia_x, t_col))
            if not np.isfinite([vent_y, vent_x, pia_y, pia_x]).all():
                continue

            if axis == "coronal":
                vent = (float(s), vent_y, vent_x)
                pia = (float(s), pia_y, pia_x)
            else:
                vent = (vent_y, vent_x, float(s))
                pia = (pia_y, pia_x, float(s))

            try:
                r_vent = float(_r01_at_ijk(edt, vent))
                r_pia = float(_r01_at_ijk(edt, pia))
            except ValueError:
                continue
            if not np.isfinite(r_vent) or not np.isfinite(r_pia):
                continue

            if r_vent <= r_pia:
                lo, hi = vent, pia
            else:
                lo, hi = pia, vent
            low_ijk[row, col] = np.asarray(lo, dtype=np.float32)
            high_ijk[row, col] = np.asarray(hi, dtype=np.float32)

    valid_rows = np.ones((slice_keys.size,), dtype=bool)
    for row in range(slice_keys.size):
        for dim in range(3):
            low_ijk[row, :, dim] = _fill_nan_series_interior(low_ijk[row, :, dim]).astype(np.float32, copy=False)
            high_ijk[row, :, dim] = _fill_nan_series_interior(high_ijk[row, :, dim]).astype(np.float32, copy=False)
        finite_pairs = np.isfinite(low_ijk[row]).all(axis=1) & np.isfinite(high_ijk[row]).all(axis=1)
        if int(np.count_nonzero(finite_pairs)) == 0:
            valid_rows[row] = False
    if not np.all(valid_rows):
        dropped = int(np.sum(~valid_rows))
        print(f"[lut] {axis} ijk dropping invalid rows={dropped}")
        slice_keys = slice_keys[valid_rows]
        low_ijk = low_ijk[valid_rows]
        high_ijk = high_ijk[valid_rows]
    if slice_keys.size == 0:
        raise ValueError(f"No valid rows left for axis={axis} after NaN sanitization.")

    np.savez_compressed(
        path,
        source_axis=axis,
        source_slice_keys=slice_keys,
        t_grid=t_grid.astype(np.float32),
        t_domain=np.asarray(["t_all"]),
        low_ijk=low_ijk,
        high_ijk=high_ijk,
    )
    dt = time.perf_counter() - t0
    print(f"[lut] ijk segment LUT done axis={axis}: elapsed_s={dt:.1f}")
    return path


def build_ijk_segment_luts(
    *,
    outdir: Path,
    n_t: int = 1024,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if n_t < 2:
        raise ValueError("n_t must be >= 2.")
    coronal = _build_ijk_segment_lut_for_axis(outdir=outdir, axis="coronal", n_t=int(n_t), overwrite=bool(overwrite))
    sagittal = _build_ijk_segment_lut_for_axis(outdir=outdir, axis="sagittal", n_t=int(n_t), overwrite=bool(overwrite))
    return coronal, sagittal


def _lookup_ijk_from_lut_knn(
    *,
    source_slice_keys: np.ndarray,
    t_grid: np.ndarray,
    low_ijk: np.ndarray,
    high_ijk: np.ndarray,
    slice_index: int | float,
    t: float | np.ndarray,
    k: int = IJK_LUT_KNN_K,
    sigma_slice: float = IJK_LUT_KNN_SIGMA_SLICE,
    sigma_t: float = IJK_LUT_KNN_SIGMA_T,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel-regress low/high ijk from LUT samples in (slice, t) space."""
    source_slice = np.asarray(source_slice_keys, dtype=np.int32)
    t_lut = np.asarray(t_grid, dtype=np.float64)
    low = np.asarray(low_ijk, dtype=np.float64)
    high = np.asarray(high_ijk, dtype=np.float64)
    t_query = np.asarray(t, dtype=np.float64).reshape(-1)
    if source_slice.ndim != 1 or t_lut.ndim != 1:
        raise ValueError("source_slice_keys and t_grid must be 1D.")
    if low.shape != high.shape:
        raise ValueError(f"low_ijk and high_ijk shape mismatch: {low.shape} vs {high.shape}.")
    if low.shape != (source_slice.size, t_lut.size, 3):
        raise ValueError(
            "LUT arrays must have shape (n_slices, n_t, 3): "
            f"expected ({source_slice.size}, {t_lut.size}, 3), got {low.shape}."
        )
    if t_query.ndim != 1:
        raise ValueError("t query must be scalar or 1D array.")
    if not np.isfinite(t_query).all():
        raise ValueError("t query contains non-finite values.")
    sigma_slice_f = float(sigma_slice)
    sigma_t_f = float(sigma_t)
    if sigma_slice_f <= 0.0 or sigma_t_f <= 0.0:
        raise ValueError(f"sigma_slice and sigma_t must be > 0. Got {sigma_slice_f}, {sigma_t_f}.")

    slice_flat_full = np.repeat(source_slice.astype(np.float64), t_lut.size)
    t_flat_full = np.tile(t_lut.astype(np.float64), source_slice.size)
    low_flat_full = low.reshape(-1, 3)
    high_flat_full = high.reshape(-1, 3)
    valid = np.isfinite(low_flat_full).all(axis=1) & np.isfinite(high_flat_full).all(axis=1)
    if not np.any(valid):
        raise ValueError("LUT contains no finite low/high ijk samples.")

    slice_flat = slice_flat_full[valid]
    t_flat = t_flat_full[valid]
    low_flat = low_flat_full[valid]
    high_flat = high_flat_full[valid]

    pts = np.column_stack([slice_flat / sigma_slice_f, t_flat / sigma_t_f])
    tree = cKDTree(pts)

    query_slice = np.full_like(t_query, float(slice_index), dtype=np.float64)
    query_t = np.clip(t_query, 0.0, 1.0)
    q = np.column_stack([query_slice / sigma_slice_f, query_t / sigma_t_f])

    k_eff = int(min(max(1, int(k)), pts.shape[0]))
    dists, neigh = tree.query(q, k=k_eff)
    if k_eff == 1:
        dists = np.asarray(dists, dtype=np.float64)[:, None]
        neigh = np.asarray(neigh, dtype=np.int64)[:, None]
    else:
        dists = np.asarray(dists, dtype=np.float64)
        neigh = np.asarray(neigh, dtype=np.int64)

    dist2 = np.square(dists)
    dist2 -= np.min(dist2, axis=1, keepdims=True)
    weights = np.exp(-0.5 * dist2)
    den = np.sum(weights, axis=1, keepdims=True)
    bad = (~np.isfinite(den[:, 0])) | (den[:, 0] <= float(IJK_LUT_KNN_WEIGHT_EPS))
    if np.any(bad):
        weights[bad] = 0.0
        weights[bad, 0] = 1.0
        den = np.sum(weights, axis=1, keepdims=True)

    lo = np.sum(weights[:, :, None] * low_flat[neigh], axis=1) / den
    hi = np.sum(weights[:, :, None] * high_flat[neigh], axis=1) / den

    degenerate = np.linalg.norm(hi - lo, axis=1) <= 1.0e-9
    if np.any(degenerate):
        lo[degenerate] = low_flat[neigh[degenerate, 0]]
        hi[degenerate] = high_flat[neigh[degenerate, 0]]

    source_slice_used = np.rint(slice_flat[neigh[:, 0]]).astype(np.int32, copy=False)
    return lo.astype(np.float64, copy=False), hi.astype(np.float64, copy=False), source_slice_used


def ijk_from_axis_tr_with_lut(
    *,
    outdir: Path,
    axis: Literal["coronal", "sagittal"],
    slice_index: int,
    t: float,
    r01: float,
    n_t: int = 1024,
    rebuild_lut: bool = False,
) -> tuple[tuple[float, float, float], int]:
    coronal_path, sagittal_path = build_ijk_segment_luts(outdir=outdir, n_t=int(n_t), overwrite=bool(rebuild_lut))
    path = coronal_path if axis == "coronal" else sagittal_path
    d = np.load(path)

    source_slice_keys = np.asarray(d["source_slice_keys"], dtype=np.int32)
    t_grid = np.asarray(d["t_grid"], dtype=np.float64)
    low_ijk = np.asarray(d["low_ijk"], dtype=np.float64)
    high_ijk = np.asarray(d["high_ijk"], dtype=np.float64)

    lo_batch, hi_batch, source_slice_used_batch = _lookup_ijk_from_lut_knn(
        source_slice_keys=source_slice_keys,
        t_grid=t_grid,
        low_ijk=low_ijk,
        high_ijk=high_ijk,
        slice_index=int(slice_index),
        t=float(t),
    )
    lo = np.asarray(lo_batch[0], dtype=np.float64)
    hi = np.asarray(hi_batch[0], dtype=np.float64)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError(
            f"IJK LUT lookup failed for axis={axis}, slice={slice_index}, t={float(t):.6f}. "
            "Try increasing --lut-nt."
        )

    r = float(np.clip(float(r01), 0.0, 1.0))
    q = lo + r * (hi - lo)
    return (float(q[0]), float(q[1]), float(q[2])), int(source_slice_used_batch[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="EDT query (slice_i|slice_k, t, r01) -> CCF ijk/um using *midline_columns.csv")
    parser.add_argument("--outdir", type=Path, default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"))
    parser.add_argument("--axis", choices=["coronal", "sagittal"], default="coronal")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help="along-midline t_all on the chosen slice (normalized full-path arc length, typically in [0,1])",
    )
    parser.add_argument("--r01", type=float, default=None, help="depth coordinate in [0,1] (0=inner/vent, 1=pia)")
    parser.add_argument("--transform-to", choices=["coronal", "sagittal"], default=None)
    parser.add_argument("--lut-nt", type=int, default=1024, help="Number of t bins for 2D sagittal<->coronal LUT.")
    parser.add_argument("--lut-window", type=int, default=2, help="Candidate slice window (in voxels) used while building LUT.")
    parser.add_argument("--rebuild-lut", action="store_true", help="Force rebuilding LUT files before transform lookup.")
    parser.add_argument("--build-lut-only", action="store_true", help="Build LUT files and exit.")
    parser.add_argument(
        "--rebuild-ijk-lut",
        action="store_true",
        help="Force rebuilding (axis,slice,t,r01)->ijk segment LUTs before query lookup.",
    )
    parser.add_argument("--build-ijk-lut-only", action="store_true", help="Build ijk segment LUTs and exit.")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    query_mode = not bool(args.build_lut_only) and not bool(args.build_ijk_lut_only)
    if query_mode:
        missing: list[str] = []
        if args.slice is None:
            missing.append("--slice")
        if args.t is None:
            missing.append("--t")
        if args.r01 is None:
            missing.append("--r01")
        if missing:
            parser.error(f"the following arguments are required for query mode: {', '.join(missing)}")

    if args.build_ijk_lut_only:
        coronal_path, sagittal_path = build_ijk_segment_luts(
            outdir=outdir,
            n_t=int(args.lut_nt),
            overwrite=bool(args.rebuild_ijk_lut),
        )
        print(f"ijk_lut_coronal={coronal_path}")
        print(f"ijk_lut_sagittal={sagittal_path}")
        return

    if args.build_lut_only:
        s2c_path, c2s_path = build_transition_lut_2d(
            outdir=outdir,
            n_t=int(args.lut_nt),
            i_window=int(args.lut_window),
            k_window=int(args.lut_window),
            overwrite=bool(args.rebuild_lut),
        )
        print(f"lut_sagittal_to_coronal={s2c_path}")
        print(f"lut_coronal_to_sagittal={c2s_path}")
        return

    ijk, source_slice_used = ijk_from_axis_tr_with_lut(
        outdir=outdir,
        axis=args.axis,
        slice_index=int(args.slice),
        t=float(args.t),
        r01=float(args.r01),
        n_t=int(args.lut_nt),
        rebuild_lut=bool(args.rebuild_ijk_lut),
    )

    print(f"ijk_vox={ijk!r}")

    res = _load_resolution_ds_ijk_um(outdir)
    if res is None:
        print("xyz_um=None (missing resolution_ds_ijk_um.npy)")
        return
    i_um = ijk[0] * res[0]
    j_um = ijk[1] * res[1]
    k_um = ijk[2] * res[2]
    print(f"ijk_um={(i_um, j_um, k_um)!r}")

    if args.transform_to is not None:
        if args.transform_to == args.axis:
            raise ValueError("--transform-to must differ from --axis.")
        if args.transform_to == "coronal":
            inv = transform_with_lut(
                outdir=outdir,
                source_axis=args.axis,
                source_slice=int(source_slice_used),
                t=float(args.t),
                r01=float(args.r01),
                n_t=int(args.lut_nt),
                i_window=int(args.lut_window),
                k_window=int(args.lut_window),
                rebuild_lut=bool(args.rebuild_lut),
            )
            print(
                "transformed_coronal="
                f"{{slice_i:{inv.slice_index}, t:{inv.t:.6f}, r01:{inv.r01:.6f}, residual_vox:{inv.residual_vox:.4f}}}"
            )
        else:
            inv = transform_with_lut(
                outdir=outdir,
                source_axis=args.axis,
                source_slice=int(source_slice_used),
                t=float(args.t),
                r01=float(args.r01),
                n_t=int(args.lut_nt),
                i_window=int(args.lut_window),
                k_window=int(args.lut_window),
                rebuild_lut=bool(args.rebuild_lut),
            )
            print(
                "transformed_sagittal="
                f"{{slice_k:{inv.slice_index}, t:{inv.t:.6f}, r01:{inv.r01:.6f}, residual_vox:{inv.residual_vox:.4f}}}"
            )


if __name__ == "__main__":
    main()
