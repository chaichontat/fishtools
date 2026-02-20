from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COORDS_SCRIPT = Path("ccf/refextract/midsurface_coords.py")


@dataclass(frozen=True)
class SurfaceMappingData:
    vertices_ijk_um: np.ndarray
    faces: np.ndarray
    rgb_u8: np.ndarray
    slice_keys: np.ndarray
    t_grid: np.ndarray
    t_all_at_t: np.ndarray
    path_yx_vox_at_t: np.ndarray
    support_mask_tall: np.ndarray
    neomeso_mask_tall: np.ndarray
    ap_um_by_slice: np.ndarray
    ml_um_at_t: np.ndarray
    ap_norm_by_slice: np.ndarray
    ml_norm_at_t: np.ndarray
    ap_range_um: tuple[float, float]
    ml_range_um: tuple[float, float]


@dataclass(frozen=True)
class SagittalPanel:
    slice_k: int
    ap0_um: float
    img2: np.ndarray
    neomeso2: np.ndarray
    midline_xy_local: np.ndarray
    dot_xy_local: tuple[float, float] | None
    ml_curve: np.ndarray
    ap_curve: np.ndarray
    support_curve: np.ndarray
    neomeso_curve: np.ndarray


def _load_midsurface_coords_module():
    spec = importlib.util.spec_from_file_location("midsurface_coords", COORDS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {COORDS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_resolution_ds_ijk_um(outdir: Path, *, res_ijk_um: tuple[float, float, float] | None) -> tuple[float, float, float]:
    if res_ijk_um is not None:
        ri, rj, rk = (float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2]))
        if not (ri > 0.0 and rj > 0.0 and rk > 0.0):
            raise ValueError(f"--res-ijk-um must be >0, got {(ri, rj, rk)}")
        return (ri, rj, rk)

    path = outdir / "resolution_ds_ijk_um.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Provide --res-ijk-um RI RJ RK (um/voxel), "
            "or write resolution_ds_ijk_um.npy when generating midsurface artifacts."
        )
    arr = np.load(path).astype(np.float64, copy=False).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected {path} shape (3,), got {arr.shape}")
    ri, rj, rk = (float(arr[0]), float(arr[1]), float(arr[2]))
    if not (ri > 0.0 and rj > 0.0 and rk > 0.0):
        raise ValueError(f"Invalid resolution in {path}: {(ri, rj, rk)}")
    return (ri, rj, rk)


def _load_ap_axis_by_slice(outdir: Path) -> dict[int, float]:
    path = outdir / "ap_axis_um_from_strips.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run optimize_ap_axis_from_strips.py first.")
    d = np.load(path)
    slice_keys = np.asarray(d["slice_keys"], dtype=np.int32).reshape(-1)
    ap_um = np.asarray(d["ap_um"], dtype=np.float64).reshape(-1)
    if slice_keys.shape != ap_um.shape:
        raise ValueError(f"slice_keys/ap_um shape mismatch in {path}: {slice_keys.shape} vs {ap_um.shape}")
    return {int(k): float(v) for k, v in zip(slice_keys.tolist(), ap_um.tolist(), strict=True)}


def _load_s2c_t2d(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    source_slice_keys = np.asarray(d["source_slice_keys"], dtype=np.int32).reshape(-1)
    t_grid = np.asarray(d["t_grid"], dtype=np.float64).reshape(-1)
    target_slice_idx = np.asarray(d["target_slice_idx"], dtype=np.float64)
    target_t = np.asarray(d["target_t"], dtype=np.float64)
    return source_slice_keys, t_grid, target_slice_idx, target_t


def _map_sagittal_to_coronal_t2d(
    *,
    source_slice_keys: np.ndarray,
    lut_t_grid: np.ndarray,
    target_slice_idx: np.ndarray,
    target_t: np.ndarray,
    slice_k: int,
    t_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(source_slice_keys, dtype=np.int32).reshape(-1)
    if keys.size == 0:
        raise ValueError("Empty source_slice_keys in sagittal->coronal LUT.")
    row = int(np.argmin(np.abs(keys.astype(np.float64) - float(slice_k))))
    t_grid = np.asarray(lut_t_grid, dtype=np.float64).reshape(-1)
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    t_clip = np.clip(t, 0.0, 1.0)
    slice_row = np.asarray(target_slice_idx[row], dtype=np.float64).reshape(-1)
    t_row = np.asarray(target_t[row], dtype=np.float64).reshape(-1)
    if slice_row.size != t_grid.size or t_row.size != t_grid.size:
        raise ValueError("Unexpected sagittal->coronal LUT row shape.")
    out_slice = np.interp(t_clip, t_grid, slice_row).astype(np.float64, copy=False)
    out_t = np.interp(t_clip, t_grid, t_row).astype(np.float64, copy=False)
    return out_slice, out_t


def _load_overlap_t_ranges_csv(path: Path, *, slice_label: str) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_t_all = ("t_start_all" in fieldnames) and ("t_end_all" in fieldnames)
        has_t = ("t_start" in fieldnames) and ("t_end" in fieldnames)
        if has_t_all:
            t_start_key = "t_start_all"
            t_end_key = "t_end_all"
        elif has_t:
            t_start_key = "t_start"
            t_end_key = "t_end"
        else:
            raise ValueError(f"{path} must contain either ('t_start_all', 't_end_all') or ('t_start', 't_end').")
        for row in reader:
            if slice_label not in row or t_start_key not in row or t_end_key not in row:
                continue
            raw_slice = row[slice_label]
            if raw_slice is None:
                continue
            slice_val = float(raw_slice)
            if not np.isfinite(slice_val):
                continue
            slice_idx = int(np.rint(slice_val))
            t0 = float(row[t_start_key])
            t1 = float(row[t_end_key])
            if not np.isfinite(t0) or not np.isfinite(t1):
                continue
            out.setdefault(slice_idx, []).append((t0, t1))
    for slice_idx in out:
        out[slice_idx].sort(key=lambda x: x[0])
    return out


def _t_extent_on_path_t_for_mask(path_xy: np.ndarray, t_vertices: np.ndarray, mask_yx: np.ndarray) -> tuple[float, float] | None:
    """Match logic from midsurface_neocortex_mesocortex_allocortex_ccf_3d_plot.py."""
    pts = np.asarray(path_xy, dtype=np.float64)
    t_vals = np.asarray(t_vertices, dtype=np.float64)
    mask = np.asarray(mask_yx, dtype=bool)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None
    if t_vals.ndim != 1 or t_vals.shape[0] != pts.shape[0]:
        return None
    if mask.ndim != 2:
        return None
    h, w = mask.shape
    xi = np.clip(np.rint(pts[:, 0]).astype(np.int64, copy=False), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(np.int64, copy=False), 0, h - 1)
    inside = mask[yi, xi]
    if not np.any(inside):
        return None
    t_in = t_vals[inside]
    finite = np.isfinite(t_in)
    if int(np.count_nonzero(finite)) < 2:
        return None
    return (float(np.min(t_in[finite])), float(np.max(t_in[finite])))


def _cum_arclen_um(y: np.ndarray, x: np.ndarray, *, res_y_um: float, res_x_um: float) -> np.ndarray:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    if yv.size < 2:
        return np.zeros((yv.size,), dtype=np.float64)
    dy = np.diff(yv) * float(res_y_um)
    dx = np.diff(xv) * float(res_x_um)
    seg = np.sqrt(dy * dy + dx * dx)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)])


def _resample_polyline_um(
    *,
    y: np.ndarray,
    x: np.ndarray,
    res_y_um: float,
    res_x_um: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if n < 8:
        raise ValueError(f"n must be >= 8, got {n}")
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    if yv.size != xv.size:
        raise ValueError(f"Shape mismatch: y={yv.shape} x={xv.shape}")
    if yv.size < 2:
        raise ValueError("Need at least 2 points to resample")

    s = _cum_arclen_um(yv, xv, res_y_um=float(res_y_um), res_x_um=float(res_x_um))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid physical arc length for polyline")
    s_grid = np.linspace(0.0, total, int(n), dtype=np.float64)
    y_new = np.interp(s_grid, s, yv).astype(np.float64, copy=False)
    x_new = np.interp(s_grid, s, xv).astype(np.float64, copy=False)
    return y_new, x_new, total


def _resample_polyline_with_t_um(
    *,
    y: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    res_y_um: float,
    res_x_um: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    tv = np.asarray(t, dtype=np.float64).reshape(-1)
    if yv.shape != xv.shape or yv.shape != tv.shape:
        raise ValueError(f"Shape mismatch for resampling: y={yv.shape} x={xv.shape} t={tv.shape}")
    y_new, x_new, total = _resample_polyline_um(
        y=yv,
        x=xv,
        res_y_um=float(res_y_um),
        res_x_um=float(res_x_um),
        n=int(n),
    )
    s = _cum_arclen_um(yv, xv, res_y_um=float(res_y_um), res_x_um=float(res_x_um))
    s_grid = np.linspace(0.0, float(total), int(n), dtype=np.float64)
    t_new = np.interp(s_grid, s, tv).astype(np.float64, copy=False)
    return y_new, x_new, t_new, total


def _center_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {pts.shape}")
    center = np.nanmedian(pts, axis=0)
    if not np.isfinite(center).all():
        center = np.nanmean(pts, axis=0)
    return pts - center[None, :]


def _dtw_banded_path(a: np.ndarray, b: np.ndarray, *, band: int) -> list[tuple[int, int]]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Expected same shape for a and b, got {aa.shape} vs {bb.shape}")
    n = int(aa.shape[0])
    if n < 2:
        return []
    band = int(max(0, band))

    inf = np.float64(np.inf)
    dp = np.full((n, n), inf, dtype=np.float64)
    ptr = np.full((n, n), -1, dtype=np.int8)

    def cost(i: int, j: int) -> float:
        d = aa[i] - bb[j]
        return float(d[0] * d[0] + d[1] * d[1])

    for i in range(n):
        j0 = max(0, i - band)
        j1 = min(n - 1, i + band)
        for j in range(j0, j1 + 1):
            c = cost(i, j)
            if i == 0 and j == 0:
                dp[i, j] = c
                ptr[i, j] = -1
                continue
            best = inf
            best_ptr = -1
            if i > 0 and j > 0 and np.isfinite(dp[i - 1, j - 1]):
                best = dp[i - 1, j - 1]
                best_ptr = 0
            if i > 0 and np.isfinite(dp[i - 1, j]) and dp[i - 1, j] < best:
                best = dp[i - 1, j]
                best_ptr = 1
            if j > 0 and np.isfinite(dp[i, j - 1]) and dp[i, j - 1] < best:
                best = dp[i, j - 1]
                best_ptr = 2
            if not np.isfinite(best):
                continue
            dp[i, j] = c + best
            ptr[i, j] = np.int8(best_ptr)

    if not np.isfinite(dp[n - 1, n - 1]):
        raise ValueError("DTW failed to reach end cell; increase band.")

    path: list[tuple[int, int]] = []
    i = n - 1
    j = n - 1
    while True:
        path.append((int(i), int(j)))
        step = int(ptr[i, j])
        if step < 0:
            break
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise RuntimeError(f"Unexpected DTW ptr value {step} at {(i, j)}")
        if i < 0 or j < 0:
            raise RuntimeError("DTW backtrace went out of bounds")
    path.reverse()
    return path


def _path_to_monotone_index_map(path: list[tuple[int, int]], *, n: int) -> np.ndarray:
    buckets: list[list[int]] = [[] for _ in range(int(n))]
    for i, j in path:
        if 0 <= int(i) < int(n):
            buckets[int(i)].append(int(j))
    out = np.full((int(n),), -1, dtype=np.int32)
    for i in range(int(n)):
        if buckets[i]:
            out[i] = int(np.rint(np.median(np.asarray(buckets[i], dtype=np.float64))))

    missing = out < 0
    if np.any(missing):
        idx = np.arange(out.size, dtype=np.float64)
        ok = ~missing
        if int(np.count_nonzero(ok)) < 2:
            out[:] = np.clip(np.arange(out.size, dtype=np.int32), 0, int(n) - 1)
        else:
            out[missing] = np.rint(np.interp(idx[missing], idx[ok], out[ok].astype(np.float64))).astype(np.int32)

    out = np.clip(out, 0, int(n) - 1)
    out = np.maximum.accumulate(out)
    out[-1] = int(n) - 1
    return out.astype(np.int32, copy=False)


def _invert_monotone_index_map_to_fractional_idx(*, idx_map: np.ndarray, ref_idx: int) -> float:
    m = np.asarray(idx_map, dtype=np.int32).reshape(-1)
    n = int(m.size)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    if not np.all(np.diff(m.astype(np.int64, copy=False)) >= 0):
        m = np.maximum.accumulate(m)

    starts = np.concatenate([[0], np.where(m[1:] != m[:-1])[0] + 1]).astype(np.int32, copy=False)
    ends = np.concatenate([starts[1:], [n]]).astype(np.int32, copy=False)
    ref_vals = m[starts].astype(np.float64, copy=False)
    mid = (starts.astype(np.float64) + (ends.astype(np.float64) - 1.0)) / 2.0
    if ref_vals.size == 1:
        return float(mid[0])
    target = float(np.clip(int(ref_idx), float(ref_vals[0]), float(ref_vals[-1])))
    return float(np.interp(target, ref_vals, mid))


def _grid_tri_faces(n_rows: int, n_cols: int) -> np.ndarray:
    if n_rows < 2 or n_cols < 2:
        raise ValueError(f"Need at least 2x2 grid, got {n_rows}x{n_cols}")
    faces: list[list[int]] = []
    for r in range(n_rows - 1):
        base0 = r * n_cols
        base1 = (r + 1) * n_cols
        for c in range(n_cols - 1):
            v00 = base0 + c
            v01 = base0 + c + 1
            v10 = base1 + c
            v11 = base1 + c + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return np.asarray(faces, dtype=np.int32)


def _normalize_range(values: np.ndarray, *, name: str) -> tuple[np.ndarray, tuple[float, float]]:
    x = np.asarray(values, dtype=np.float64)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or not (x_max > x_min):
        raise ValueError(f"Invalid {name} range: [{x_min}, {x_max}]")
    norm = (x - x_min) / (x_max - x_min)
    return np.clip(norm, 0.0, 1.0), (x_min, x_max)


def build_coronal_ap_ml_surface(
    *,
    outdir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    ref_t: float,
    band_frac: float,
    b_const: float,
    res_ijk_um: tuple[float, float, float] | None = None,
) -> SurfaceMappingData:
    outdir = Path(outdir)
    mod = _load_midsurface_coords_module()
    coronal = mod.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    if not coronal:
        raise ValueError(f"No coronal midline columns found under {outdir}")

    n_t_i = int(n_t)
    if n_t_i < 8:
        raise ValueError(f"n_t must be >= 8, got {n_t_i}")
    ref_t_f = float(ref_t)
    if not np.isfinite(ref_t_f) or not (0.0 <= ref_t_f <= 1.0):
        raise ValueError(f"ref_t must be in [0,1], got {ref_t_f}")
    b_const_f = float(np.clip(float(b_const), 0.0, 1.0))

    res_i_um, res_j_um, res_k_um = _load_resolution_ds_ijk_um(outdir, res_ijk_um=res_ijk_um)
    ap_by_slice = _load_ap_axis_by_slice(outdir)
    neomeso_ranges = _load_overlap_t_ranges_csv(
        outdir / "coronal_neocortex_mesocortex_overlap_t_ranges.csv",
        slice_label="slice_i",
    )
    overlay_path = outdir / "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy"
    overlay_3d = np.load(overlay_path).astype(bool, copy=False) if overlay_path.exists() else None

    keys = [
        int(k)
        for k in sorted(coronal.keys())
        if int(slice_i_min) <= int(k) <= int(slice_i_max) and int(k) in ap_by_slice
    ]
    if len(keys) < 2:
        raise ValueError("Need at least 2 coronal slices after filtering by range and AP axis.")

    ap_um_by_slice = np.asarray([ap_by_slice[int(k)] for k in keys], dtype=np.float64)
    order = np.argsort(ap_um_by_slice)
    keys = [keys[i] for i in order.tolist()]
    ap_um_by_slice = ap_um_by_slice[order]
    slice_keys = np.asarray(keys, dtype=np.int32)

    t_grid = np.linspace(0.0, 1.0, n_t_i, dtype=np.float64)
    points_ijk_um = np.full((len(keys), n_t_i, 3), np.nan, dtype=np.float64)
    t_all_at_t = np.full((len(keys), n_t_i), np.nan, dtype=np.float64)
    path_yx_vox_at_t = np.full((len(keys), n_t_i, 2), np.nan, dtype=np.float64)
    support_mask_tall = np.zeros((len(keys), n_t_i), dtype=bool)
    neomeso_mask_tall = np.zeros((len(keys), n_t_i), dtype=bool)
    ml_um_at_t = np.full((len(keys), n_t_i), np.nan, dtype=np.float64)

    ref_slice_i = int(keys[len(keys) // 2])
    ref_cols = coronal[int(ref_slice_i)]
    ref_y_rs, ref_x_rs, _ref_len_um = _resample_polyline_um(
        y=ref_cols.y,
        x=ref_cols.x,
        res_y_um=float(res_j_um),
        res_x_um=float(res_k_um),
        n=n_t_i,
    )
    ref_pts2_um_center = _center_2d(np.column_stack([ref_y_rs * float(res_j_um), ref_x_rs * float(res_k_um)]))
    ref_idx = int(np.clip(int(np.rint(ref_t_f * float(n_t_i - 1))), 0, n_t_i - 1))

    band = int(max(4, int(round(float(band_frac) * float(n_t_i)))))

    for row, key in enumerate(keys):
        cols = coronal[int(key)]
        y_rs, x_rs, t_rs, total_len_um = _resample_polyline_with_t_um(
            y=cols.y,
            x=cols.x,
            t=cols.t,
            res_y_um=float(res_j_um),
            res_x_um=float(res_k_um),
            n=n_t_i,
        )
        pts2_um = np.column_stack([y_rs * float(res_j_um), x_rs * float(res_k_um)]).astype(np.float64, copy=False)
        pts2_center = _center_2d(pts2_um)
        fwd = float(np.nanmean(np.sum((pts2_center - ref_pts2_um_center) ** 2, axis=1)))
        rev_center = _center_2d(pts2_um[::-1])
        rev = float(np.nanmean(np.sum((rev_center - ref_pts2_um_center) ** 2, axis=1)))
        if rev < fwd:
            y_rs = y_rs[::-1]
            x_rs = x_rs[::-1]
            t_rs = t_rs[::-1]
            pts2_center = rev_center

        band_try = int(band)
        for _ in range(3):
            try:
                path = _dtw_banded_path(pts2_center, ref_pts2_um_center, band=band_try)
                break
            except ValueError:
                band_try = int(min(n_t_i - 1, int(round(float(band_try) * 1.75)) + 1))
        else:
            raise ValueError(f"DTW failed for slice {int(key)} vs ref slice {int(ref_slice_i)}; increase band_frac.")

        idx_map = _path_to_monotone_index_map(path, n=n_t_i)
        origin_idx_f = _invert_monotone_index_map_to_fractional_idx(idx_map=idx_map, ref_idx=int(ref_idx))
        origin_t0 = float(origin_idx_f) / float(n_t_i - 1)
        origin_um = float(origin_t0) * float(total_len_um)

        ml_um_at_t[row] = (t_grid * float(total_len_um)) - origin_um
        t_all_at_t[row] = t_rs
        path_yx_vox_at_t[row, :, 0] = y_rs
        path_yx_vox_at_t[row, :, 1] = x_rs
        # Support of `t_all` should reflect where `t_all_at_t` is defined (finite) along this slice,
        # not whether it falls in any particular numeric range.
        support_mask_tall[row] = np.isfinite(t_rs)
        ranges = neomeso_ranges.get(int(key), [])
        if ranges:
            for t0, t1 in ranges:
                # `*_overlap_t_ranges.csv` is produced from normalized arc length along a u=0.5 contour
                # (slice-local full-path coordinate). This matches our slice-local `t_grid`, not the
                # representative-curve `cols.t` we resampled into `t_rs`.
                neomeso_mask_tall[row] |= (t_grid >= float(t0)) & (t_grid <= float(t1))
        elif overlay_3d is not None:
            if overlay_3d.ndim != 3:
                raise ValueError(f"Expected overlay mask to be 3D, got shape={overlay_3d.shape}")
            if not (0 <= int(key) < int(overlay_3d.shape[0])):
                raise ValueError(f"overlay mask does not contain slice_i={int(key)} (shape={overlay_3d.shape})")
            path_xy_raw = np.column_stack([np.asarray(cols.x, dtype=np.float64), np.asarray(cols.y, dtype=np.float64)])
            extent = _t_extent_on_path_t_for_mask(
                path_xy_raw,
                np.asarray(cols.t, dtype=np.float64),
                overlay_3d[int(key), :, :],
            )
            if extent is not None:
                t0, t1 = extent
                neomeso_mask_tall[row] |= np.isfinite(t_rs) & (t_rs >= float(t0)) & (t_rs <= float(t1))

        neomeso_mask_tall[row] &= support_mask_tall[row]
        points_ijk_um[row, :, 0] = float(key) * float(res_i_um)
        points_ijk_um[row, :, 1] = y_rs * float(res_j_um)
        points_ijk_um[row, :, 2] = x_rs * float(res_k_um)

    ap_norm_by_slice, ap_range_um = _normalize_range(ap_um_by_slice, name="AP")
    ml_norm_at_t, ml_range_um = _normalize_range(ml_um_at_t, name="ML")

    ap_norm_grid = np.broadcast_to(ap_norm_by_slice[:, None], ml_norm_at_t.shape)
    rgb = np.empty((int(slice_keys.size), n_t_i, 3), dtype=np.float64)
    rgb[:, :, 0] = ap_norm_grid
    rgb[:, :, 1] = ml_norm_at_t
    rgb[:, :, 2] = b_const_f
    rgb_u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)

    vertices_ijk_um = points_ijk_um.reshape(-1, 3).astype(np.float32, copy=False)
    faces = _grid_tri_faces(int(slice_keys.size), n_t_i)

    return SurfaceMappingData(
        vertices_ijk_um=vertices_ijk_um,
        faces=faces,
        rgb_u8=rgb_u8.reshape(-1, 3),
        slice_keys=slice_keys.astype(np.int32, copy=False),
        t_grid=t_grid.astype(np.float64, copy=False),
        t_all_at_t=t_all_at_t.astype(np.float64, copy=False),
        path_yx_vox_at_t=path_yx_vox_at_t.astype(np.float64, copy=False),
        support_mask_tall=support_mask_tall.astype(bool, copy=False),
        neomeso_mask_tall=neomeso_mask_tall.astype(bool, copy=False),
        ap_um_by_slice=ap_um_by_slice.astype(np.float64, copy=False),
        ml_um_at_t=ml_um_at_t.astype(np.float64, copy=False),
        ap_norm_by_slice=ap_norm_by_slice.astype(np.float64, copy=False),
        ml_norm_at_t=ml_norm_at_t.astype(np.float64, copy=False),
        ap_range_um=ap_range_um,
        ml_range_um=ml_range_um,
    )


def _build_legend_rgb(
    *,
    ap_range_um: tuple[float, float],
    ml_range_um: tuple[float, float],
    b_const: float,
    n_ap: int = 512,
    n_ml: int = 512,
) -> np.ndarray:
    if n_ap < 2 or n_ml < 2:
        raise ValueError(f"legend resolution must be >=2, got {(n_ap, n_ml)}")
    ap_norm = np.linspace(0.0, 1.0, int(n_ap), dtype=np.float64)
    ml_norm = np.linspace(0.0, 1.0, int(n_ml), dtype=np.float64)
    ap_grid, ml_grid = np.meshgrid(ap_norm, ml_norm, indexing="xy")
    rgb = np.empty((int(n_ml), int(n_ap), 3), dtype=np.float64)
    rgb[:, :, 0] = ap_grid
    rgb[:, :, 1] = ml_grid
    rgb[:, :, 2] = float(np.clip(float(b_const), 0.0, 1.0))
    return np.clip(rgb, 0.0, 1.0)


def _build_legend_range_mask(*, data: SurfaceMappingData, row_mask: np.ndarray, n_ap: int, n_ml: int) -> np.ndarray:
    if n_ap < 2 or n_ml < 2:
        raise ValueError(f"legend support resolution must be >=2, got {(n_ap, n_ml)}")
    mask = np.zeros((int(n_ml), int(n_ap)), dtype=bool)
    row_support = np.asarray(row_mask, dtype=bool)
    if row_support.shape != data.ml_norm_at_t.shape:
        raise ValueError(f"row_mask/ml shape mismatch: {row_support.shape} vs {data.ml_norm_at_t.shape}")

    has_support = np.count_nonzero(row_support, axis=1) > 0
    if int(np.count_nonzero(has_support)) == 0:
        return mask

    ap_rows = np.asarray(data.ap_norm_by_slice[has_support], dtype=np.float64)
    ml_lo = np.asarray([float(np.nanmin(data.ml_norm_at_t[i][row_support[i]])) for i in np.flatnonzero(has_support)], dtype=np.float64)
    ml_hi = np.asarray([float(np.nanmax(data.ml_norm_at_t[i][row_support[i]])) for i in np.flatnonzero(has_support)], dtype=np.float64)
    order = np.argsort(ap_rows)
    ap_rows = ap_rows[order]
    ml_lo = ml_lo[order]
    ml_hi = ml_hi[order]

    ap_axis = np.linspace(0.0, 1.0, int(n_ap), dtype=np.float64)
    ml_axis = np.linspace(0.0, 1.0, int(n_ml), dtype=np.float64)

    if ap_rows.size == 1:
        col = int(np.clip(np.rint(float(ap_rows[0]) * float(n_ap - 1)), 0, n_ap - 1))
        lo = float(np.clip(ml_lo[0], 0.0, 1.0))
        hi = float(np.clip(ml_hi[0], 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        mask[:, col] = (ml_axis >= lo) & (ml_axis <= hi)
        return mask

    lo_interp = np.interp(ap_axis, ap_rows, ml_lo, left=np.nan, right=np.nan)
    hi_interp = np.interp(ap_axis, ap_rows, ml_hi, left=np.nan, right=np.nan)
    for col in range(int(n_ap)):
        lo = float(lo_interp[col])
        hi = float(hi_interp[col])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        mask[:, col] = (ml_axis >= lo) & (ml_axis <= hi)
    return mask


def _binary_overlay_slice(mask2: np.ndarray) -> np.ndarray:
    m = np.asarray(mask2, dtype=bool)
    out = np.full(m.shape, np.nan, dtype=np.float32)
    out[m] = 1.0
    return out


def _robust_vmin_vmax(vol_3d: np.ndarray, *, mask_3d: np.ndarray | None = None) -> tuple[float, float]:
    sample = np.asarray(vol_3d, dtype=np.float32)
    if mask_3d is not None:
        m = np.asarray(mask_3d, dtype=bool)
        if m.shape != sample.shape:
            raise ValueError(f"mask_3d shape mismatch: {m.shape} vs {sample.shape}")
        vals = sample[m]
    else:
        vals = sample.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.percentile(vals, [1.0, 99.0]).astype(np.float64)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or not (vmax > vmin):
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or not (vmax > vmin):
        return (0.0, 1.0)
    return (float(vmin), float(vmax))


def _ap_hline_values(*, ap_range_um: tuple[float, float], step_um: float) -> np.ndarray:
    step = float(step_um)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"ap hline step must be >0, got {step_um}")
    ap_min, ap_max = (float(ap_range_um[0]), float(ap_range_um[1]))
    lo = float(np.ceil(ap_min / step) * step)
    vals = np.arange(lo, ap_max + 0.5 * step, step, dtype=np.float64)
    vals = vals[(vals >= ap_min - 1.0e-9) & (vals <= ap_max + 1.0e-9)]
    if vals.size == 0:
        vals = np.asarray([0.5 * (ap_min + ap_max)], dtype=np.float64)
    return vals.astype(np.float64, copy=False)


def _prepare_coronal_slice_panels(
    *,
    outdir: Path,
    data: SurfaceMappingData,
    ap_lines_um: np.ndarray,
    atlas_name: str,
    brainglobe_config_dir: Path,
) -> list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cortex_fit_path = outdir / "cortex_mask_fit_3d_ds.npy"
    cortex_clean_path = outdir / "cortex_mask_clean_3d_ds.npy"
    if cortex_fit_path.exists():
        cortex_3d = np.load(cortex_fit_path).astype(bool, copy=False)
    elif cortex_clean_path.exists():
        cortex_3d = np.load(cortex_clean_path).astype(bool, copy=False)
    else:
        raise FileNotFoundError(f"Missing {cortex_fit_path} and {cortex_clean_path}.")

    brain_mask_path = outdir / "brain_mask_3d_ds.npy"
    if not brain_mask_path.exists():
        raise FileNotFoundError(f"Missing {brain_mask_path}.")
    brain_3d = np.load(brain_mask_path).astype(bool, copy=False)
    if brain_3d.shape != cortex_3d.shape:
        if (
            brain_3d.ndim == 3
            and brain_3d.shape[0] == cortex_3d.shape[0]
            and brain_3d.shape[1] == cortex_3d.shape[1]
            and brain_3d.shape[2] >= cortex_3d.shape[2]
        ):
            brain_3d = brain_3d[:, :, : cortex_3d.shape[2]].astype(bool, copy=False)
        else:
            raise ValueError(f"brain/cortex shape mismatch: {brain_3d.shape} vs {cortex_3d.shape}")

    if brainglobe_config_dir.exists():
        import os

        os.environ["BRAINGLOBE_CONFIG_DIR"] = str(brainglobe_config_dir.resolve())
    from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

    atlas = BrainGlobeAtlas(str(atlas_name))
    ref_full = np.asarray(atlas.reference)
    ds = int(max(1, int(np.rint(float(ref_full.shape[0]) / float(cortex_3d.shape[0])))))
    ref = ref_full[::ds, ::ds, ::ds]
    if ref.shape[0] != cortex_3d.shape[0] or ref.shape[1] != cortex_3d.shape[1]:
        raise ValueError(f"Downsampled reference shape {ref.shape} does not match cortex shape {cortex_3d.shape}; ds={ds}")
    if ref.shape[2] != cortex_3d.shape[2]:
        if ref.shape[2] >= cortex_3d.shape[2]:
            ref = ref[:, :, : cortex_3d.shape[2]]
        else:
            raise ValueError(f"Reference k dim smaller than cortex: ref={ref.shape} cortex={cortex_3d.shape}")
    reference_3d = ref.astype(np.float32, copy=False)
    overlay_path = outdir / "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy"
    overlay_3d: np.ndarray | None = None
    if overlay_path.exists():
        overlay_3d = np.load(overlay_path).astype(bool, copy=False)
        if overlay_3d.ndim != 3:
            raise ValueError(f"Expected overlay mask to be 3D, got shape={overlay_3d.shape}")
        if overlay_3d.shape != cortex_3d.shape:
            if (
                overlay_3d.shape[0] == cortex_3d.shape[0]
                and overlay_3d.shape[1] == cortex_3d.shape[1]
                and overlay_3d.shape[2] >= cortex_3d.shape[2]
            ):
                overlay_3d = overlay_3d[:, :, : cortex_3d.shape[2]].astype(bool, copy=False)
            else:
                raise ValueError(f"overlay/cortex shape mismatch: {overlay_3d.shape} vs {cortex_3d.shape}")

    jk = np.argwhere(np.any(brain_3d, axis=0))
    if jk.size == 0:
        raise ValueError("Brain mask is empty; cannot crop coronal slices.")
    j0 = int(np.min(jk[:, 0]))
    j1 = int(np.max(jk[:, 0])) + 1
    k0 = int(np.min(jk[:, 1]))
    k1 = int(np.max(jk[:, 1])) + 1

    vmin, vmax = _robust_vmin_vmax(reference_3d, mask_3d=brain_3d)
    if not (vmax > vmin):
        vmin, vmax = (0.0, 1.0)

    panels: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    ap_vals = np.asarray(data.ap_um_by_slice, dtype=np.float64)
    slice_vals = np.asarray(data.slice_keys, dtype=np.int32)
    for ap_um in np.asarray(ap_lines_um, dtype=np.float64).tolist():
        idx = int(np.argmin(np.abs(ap_vals - float(ap_um))))
        slice_i = int(slice_vals[idx])
        img = reference_3d[slice_i, j0:j1, k0:k1].astype(np.float32, copy=False)
        brain2 = brain_3d[slice_i, j0:j1, k0:k1]
        cortex = cortex_3d[slice_i, j0:j1, k0:k1]
        img_clip = np.clip((img - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0).astype(np.float32, copy=False)
        img_clip = img_clip.copy()
        img_clip[~brain2] = np.nan
        if overlay_3d is None:
            neomeso_overlay = np.full(cortex.shape, np.nan, dtype=np.float32)
        else:
            neomeso_overlay = _binary_overlay_slice(overlay_3d[slice_i, j0:j1, k0:k1])
        midline_xy_local = np.column_stack(
            [
                np.asarray(data.path_yx_vox_at_t[idx, :, 1], dtype=np.float64) - float(k0),
                np.asarray(data.path_yx_vox_at_t[idx, :, 0], dtype=np.float64) - float(j0),
            ]
        ).astype(np.float32, copy=False)
        panels.append((float(ap_um), int(slice_i), img_clip, _binary_overlay_slice(cortex), neomeso_overlay, midline_xy_local))
    return panels


def _prepare_sagittal_slice_panels(
    *,
    outdir: Path,
    data: SurfaceMappingData,
    atlas_name: str,
    brainglobe_config_dir: Path,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> list[SagittalPanel]:
    outdir = Path(outdir)

    cortex_fit_path = outdir / "cortex_mask_fit_3d_ds.npy"
    cortex_clean_path = outdir / "cortex_mask_clean_3d_ds.npy"
    if cortex_fit_path.exists():
        cortex_3d = np.load(cortex_fit_path).astype(bool, copy=False)
    elif cortex_clean_path.exists():
        cortex_3d = np.load(cortex_clean_path).astype(bool, copy=False)
    else:
        raise FileNotFoundError(f"Missing {cortex_fit_path} and {cortex_clean_path}.")

    brain_mask_path = outdir / "brain_mask_3d_ds.npy"
    if not brain_mask_path.exists():
        raise FileNotFoundError(f"Missing {brain_mask_path}.")
    brain_3d = np.load(brain_mask_path).astype(bool, copy=False)
    if brain_3d.shape != cortex_3d.shape:
        if (
            brain_3d.ndim == 3
            and brain_3d.shape[0] == cortex_3d.shape[0]
            and brain_3d.shape[1] == cortex_3d.shape[1]
            and brain_3d.shape[2] >= cortex_3d.shape[2]
        ):
            brain_3d = brain_3d[:, :, : cortex_3d.shape[2]].astype(bool, copy=False)
        else:
            raise ValueError(f"brain/cortex shape mismatch: {brain_3d.shape} vs {cortex_3d.shape}")

    if brainglobe_config_dir.exists():
        import os

        os.environ["BRAINGLOBE_CONFIG_DIR"] = str(brainglobe_config_dir.resolve())
    from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

    atlas = BrainGlobeAtlas(str(atlas_name))
    ref_full = np.asarray(atlas.reference)
    ds = int(max(1, int(np.rint(float(ref_full.shape[0]) / float(cortex_3d.shape[0])))))
    ref = ref_full[::ds, ::ds, ::ds]
    if ref.shape[0] != cortex_3d.shape[0] or ref.shape[1] != cortex_3d.shape[1]:
        raise ValueError(f"Downsampled reference shape {ref.shape} does not match cortex shape {cortex_3d.shape}; ds={ds}")
    if ref.shape[2] != cortex_3d.shape[2]:
        if ref.shape[2] >= cortex_3d.shape[2]:
            ref = ref[:, :, : cortex_3d.shape[2]]
        else:
            raise ValueError(f"Reference k dim smaller than cortex: ref={ref.shape} cortex={cortex_3d.shape}")
    reference_3d = ref.astype(np.float32, copy=False)

    overlay_path = outdir / "overlay_neocortex_mesocortex_no_allocortex_3d_ds.npy"
    overlay_3d: np.ndarray | None = None
    if overlay_path.exists():
        overlay_3d = np.load(overlay_path).astype(bool, copy=False)
        if overlay_3d.ndim != 3:
            raise ValueError(f"Expected overlay mask to be 3D, got shape={overlay_3d.shape}")
        if overlay_3d.shape != cortex_3d.shape:
            if (
                overlay_3d.shape[0] == cortex_3d.shape[0]
                and overlay_3d.shape[1] == cortex_3d.shape[1]
                and overlay_3d.shape[2] >= cortex_3d.shape[2]
            ):
                overlay_3d = overlay_3d[:, :, : cortex_3d.shape[2]].astype(bool, copy=False)
            else:
                raise ValueError(f"overlay/cortex shape mismatch: {overlay_3d.shape} vs {cortex_3d.shape}")

    ij = np.argwhere(np.any(brain_3d, axis=2))
    if ij.size == 0:
        raise ValueError("Brain mask is empty; cannot crop sagittal slices.")
    i0 = int(np.min(ij[:, 0]))
    i1 = int(np.max(ij[:, 0])) + 1
    j0 = int(np.min(ij[:, 1]))
    j1 = int(np.max(ij[:, 1])) + 1

    vmin, vmax = _robust_vmin_vmax(reference_3d, mask_3d=brain_3d)
    if not (vmax > vmin):
        vmin, vmax = (0.0, 1.0)

    mod = _load_midsurface_coords_module()
    sagittal = mod.load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
    if not sagittal:
        raise ValueError(f"No sagittal midline columns found under {outdir}")

    s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    if not s2c_path.exists():
        raise FileNotFoundError(f"Missing {s2c_path}. Build chart LUTs first with midsurface_coords.py.")
    source_slice_keys, lut_t_grid, target_slice_idx, target_t = _load_s2c_t2d(s2c_path)
    if source_slice_keys.size == 0:
        raise ValueError("No sagittal source_slice_keys in sagittal->coronal LUT.")

    k_step = int(sagittal_k_step)
    if k_step <= 0:
        raise ValueError(f"sagittal_k_step must be >0, got {sagittal_k_step}")

    k_supported_min = int(np.min(source_slice_keys))
    k_supported_max = int(np.max(source_slice_keys))
    requested = list(range(int(k_supported_min), int(k_supported_max) + 1, int(k_step)))
    use_ks = sorted(
        {
            int(source_slice_keys[int(np.argmin(np.abs(source_slice_keys.astype(np.float64) - float(k))))])
            for k in requested
        }
    )
    sag_keys = {int(k) for k in sagittal.keys()}
    use_ks = [int(k) for k in use_ks if int(k) in sag_keys]
    if not use_ks:
        raise ValueError("No sagittal slices available after intersecting LUT keys with sagittal_midline_columns.csv.")

    coronal_keys = np.asarray(data.slice_keys, dtype=np.float64)
    ap_keys = np.asarray(data.slice_keys, dtype=np.float64).reshape(-1)
    ap_vals = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    t_grid = np.asarray(data.t_grid, dtype=np.float64).reshape(-1)
    ml_um_at_t = np.asarray(data.ml_um_at_t, dtype=np.float64)
    support_tall = np.asarray(data.support_mask_tall, dtype=bool)
    neomeso_tall = np.asarray(data.neomeso_mask_tall, dtype=bool)

    n_sample = int(sagittal_n_sample)
    if n_sample < 16:
        raise ValueError(f"sagittal_n_sample must be >=16, got {n_sample}")
    t_s_plot = np.linspace(0.0, 1.0, n_sample, dtype=np.float64)

    panels: list[SagittalPanel] = []
    for k in use_ks:
        cols = sagittal[int(k)]
        t_s = np.asarray(cols.t, dtype=np.float64).reshape(-1)
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_slice_keys,
            lut_t_grid=lut_t_grid,
            target_slice_idx=target_slice_idx,
            target_t=target_t,
            slice_k=int(k),
            t_s=t_s,
        )
        ap_curve_pts = np.interp(cor_slice_f, ap_keys, ap_vals, left=np.nan, right=np.nan)
        row_idx_pts = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f[None, :]), axis=0)
        ml_curve_pts = np.full_like(cor_t, np.nan, dtype=np.float64)
        sup_pts = np.zeros_like(cor_t, dtype=bool)
        keep_pts = np.isfinite(ap_curve_pts) & np.isfinite(cor_t)
        for ridx in np.unique(row_idx_pts[keep_pts]).tolist():
            ridx_i = int(ridx)
            mask = keep_pts & (row_idx_pts == ridx_i)
            ml_curve_pts[mask] = np.interp(cor_t[mask], t_grid, ml_um_at_t[ridx_i], left=np.nan, right=np.nan)
            sup_pts[mask] = (
                np.interp(cor_t[mask], t_grid, support_tall[ridx_i].astype(np.float64), left=0.0, right=0.0) > 0.5
            )
        keep_pts2 = keep_pts & np.isfinite(ml_curve_pts) & sup_pts
        if int(np.count_nonzero(keep_pts2)) < 2:
            continue
        valid_idx = np.flatnonzero(keep_pts2)
        idx0 = int(valid_idx[int(np.argmin(np.abs(ml_curve_pts[valid_idx])))] )
        ap0 = float(ap_curve_pts[idx0])

        cor_slice_f_plot, cor_t_plot = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_slice_keys,
            lut_t_grid=lut_t_grid,
            target_slice_idx=target_slice_idx,
            target_t=target_t,
            slice_k=int(k),
            t_s=t_s_plot,
        )
        ap_curve = np.interp(cor_slice_f_plot, ap_keys, ap_vals, left=np.nan, right=np.nan)
        row_idx = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f_plot[None, :]), axis=0)
        ml_curve = np.full_like(cor_t_plot, np.nan, dtype=np.float64)
        sup_curve = np.zeros_like(cor_t_plot, dtype=bool)
        neo_curve = np.zeros_like(cor_t_plot, dtype=bool)
        keep = np.isfinite(ap_curve) & np.isfinite(cor_t_plot)
        for ridx in np.unique(row_idx[keep]).tolist():
            ridx_i = int(ridx)
            mask = keep & (row_idx == ridx_i)
            ml_curve[mask] = np.interp(cor_t_plot[mask], t_grid, ml_um_at_t[ridx_i], left=np.nan, right=np.nan)
            sup_curve[mask] = np.interp(
                cor_t_plot[mask], t_grid, support_tall[ridx_i].astype(np.float64), left=0.0, right=0.0
            ) > 0.5
            neo_curve[mask] = np.interp(
                cor_t_plot[mask], t_grid, neomeso_tall[ridx_i].astype(np.float64), left=0.0, right=0.0
            ) > 0.5
        keep2 = keep & np.isfinite(ml_curve) & sup_curve
        ml_curve = np.where(keep2, ml_curve, np.nan)
        ap_curve = np.where(keep2, ap_curve, np.nan)
        neo_curve &= keep2

        img = reference_3d[i0:i1, j0:j1, int(k)].astype(np.float32, copy=False)
        brain2 = brain_3d[i0:i1, j0:j1, int(k)]
        img_clip = np.clip((img - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0).astype(np.float32, copy=False)
        img_clip = img_clip.copy()
        img_clip[~brain2] = np.nan
        if overlay_3d is None:
            neomeso2 = np.full(img_clip.shape, np.nan, dtype=np.float32)
        else:
            neomeso2 = _binary_overlay_slice(overlay_3d[i0:i1, j0:j1, int(k)])

        midline_xy_local = np.column_stack(
            [
                np.asarray(cols.x, dtype=np.float64) - float(j0),
                np.asarray(cols.y, dtype=np.float64) - float(i0),
            ]
        ).astype(np.float32, copy=False)
        dot_xy_local: tuple[float, float] | None = None
        if 0 <= idx0 < int(midline_xy_local.shape[0]) and np.isfinite(midline_xy_local[idx0]).all():
            dot_xy_local = (float(midline_xy_local[idx0, 0]), float(midline_xy_local[idx0, 1]))

        panels.append(
            SagittalPanel(
                slice_k=int(k),
                ap0_um=float(ap0),
                img2=img_clip,
                neomeso2=neomeso2,
                midline_xy_local=midline_xy_local,
                dot_xy_local=dot_xy_local,
                ml_curve=ml_curve,
                ap_curve=ap_curve,
                support_curve=sup_curve & keep2,
                neomeso_curve=neo_curve,
            )
        )

    panels = [p for p in panels if np.isfinite(p.ap0_um)]
    return panels


def save_ap_ml_mapping_figure(
    *,
    out_png: Path,
    data: SurfaceMappingData,
    b_const: float,
    outdir: Path,
    ap_hline_step_um: float,
    atlas_name: str,
    brainglobe_config_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.collections import LineCollection  # noqa: E402

    legend_rgb = _build_legend_rgb(ap_range_um=data.ap_range_um, ml_range_um=data.ml_range_um, b_const=float(b_const))
    support_mask = _build_legend_range_mask(
        data=data,
        row_mask=data.support_mask_tall,
        n_ap=int(legend_rgb.shape[1]),
        n_ml=int(legend_rgb.shape[0]),
    )
    neomeso_mask = _build_legend_range_mask(
        data=data,
        row_mask=data.neomeso_mask_tall,
        n_ap=int(legend_rgb.shape[1]),
        n_ml=int(legend_rgb.shape[0]),
    )
    neomeso_mask = neomeso_mask & support_mask
    legend_display = np.ones_like(legend_rgb, dtype=np.float64)
    legend_display[support_mask] = legend_rgb[support_mask]
    ap_lines = _ap_hline_values(ap_range_um=data.ap_range_um, step_um=float(ap_hline_step_um))
    if ap_lines.size > 1:
        ap_lines = ap_lines[1:]
    slice_panels = _prepare_coronal_slice_panels(
        outdir=Path(outdir),
        data=data,
        ap_lines_um=ap_lines,
        atlas_name=str(atlas_name),
        brainglobe_config_dir=Path(brainglobe_config_dir),
    )

    fig_w_in = 8.4
    panel_scale = 1.5
    target_panel_h_in = 1.85
    axes_w_frac = 0.58
    axes_h_frac = 0.88
    if ap_lines.size >= 2:
        step_um = float(np.nanmedian(np.diff(ap_lines.astype(np.float64, copy=False))))
        frac = float(step_um) / (float(data.ap_range_um[1]) - float(data.ap_range_um[0]))
        panel_h_frac = float(axes_h_frac) * float(frac) * 0.98
    else:
        panel_h_frac = float(axes_h_frac) * 0.12
    panel_h_frac = float(np.clip(panel_h_frac, 0.055, 0.22)) * float(panel_scale)
    fig_h_in = float(target_panel_h_in) / float(panel_h_frac)
    fig_h_in = float(max(fig_h_in, fig_w_in * (axes_w_frac / axes_h_frac) * 0.35))
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=200)
    ax0 = fig.add_axes([0.08, 0.08, 0.58, 0.88])

    ap_min, ap_max = data.ap_range_um
    ml_min, ml_max = data.ml_range_um
    ml_supported = np.asarray(data.ml_um_at_t, dtype=np.float64)[np.asarray(data.support_mask_tall, dtype=bool)]
    ml_supported = ml_supported[np.isfinite(ml_supported)]
    if ml_supported.size == 0:
        raise ValueError("No supported (t_all) ML values; cannot set legend x-limits.")
    ml_xlim_min = float(np.min(ml_supported))
    ml_xlim_max = float(np.max(ml_supported))
    pad = 0.03 * float(ml_xlim_max - ml_xlim_min) if ml_xlim_max > ml_xlim_min else 100.0
    ml_xlim_min -= pad
    ml_xlim_max += pad
    ml_xlim_min = float(min(ml_xlim_min, 0.0))
    ml_xlim_max = float(max(ml_xlim_max, 0.0))
    legend_display_t = np.transpose(legend_display, (1, 0, 2))
    tall_alpha = 0.65
    ax0.imshow(
        legend_display_t,
        origin="lower",
        extent=[ml_min, ml_max, ap_min, ap_max],
        aspect="auto",
        alpha=tall_alpha,
    )
    support_overlay = np.transpose(support_mask)
    neomeso_overlay = np.full((legend_rgb.shape[1], legend_rgb.shape[0]), np.nan, dtype=np.float64)
    neomeso_overlay[support_overlay] = 0.0
    neomeso_overlay[np.transpose(neomeso_mask)] = 1.0
    neomeso_im = ax0.imshow(
        neomeso_overlay,
        origin="lower",
        extent=[ml_min, ml_max, ap_min, ap_max],
        aspect="auto",
        cmap="gray",
        alpha=0.35,
        vmin=0.0,
        vmax=1.0,
    )
    neomeso_im.cmap.set_bad(alpha=0.0)
    ax0.axvline(0.0, color="#000000", linestyle=":", linewidth=0.9, alpha=0.9)
    for ap in ap_lines.tolist():
        ax0.axhline(float(ap), color="#000000", linestyle=":", linewidth=0.9, alpha=0.9)
    ap_vals = np.asarray(data.ap_um_by_slice, dtype=np.float64)
    for ap in ap_lines.tolist():
        idx = int(np.argmin(np.abs(ap_vals - float(ap))))
        ml_row = np.asarray(data.ml_um_at_t[idx], dtype=np.float64)
        neo_row = np.asarray(data.neomeso_mask_tall[idx], dtype=bool)
        valid = np.isfinite(ml_row) & neo_row
        if not np.any(valid):
            continue
        run = valid.astype(np.int8, copy=False)
        starts = np.flatnonzero(np.diff(np.concatenate([[0], run])) == 1)
        ends = np.flatnonzero(np.diff(np.concatenate([run, [0]])) == -1) + 1
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            if int(e) - int(s) < 2:
                continue
            x0 = float(np.nanmin(ml_row[int(s) : int(e)]))
            x1 = float(np.nanmax(ml_row[int(s) : int(e)]))
            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                ax0.hlines(float(ap), x0, x1, colors="#ffff00", linewidth=2.0, alpha=0.95)
    ax0.set_xlabel("ML (um)", fontsize=16)
    ax0.set_ylabel("AP (um)", fontsize=16)
    ax0.set_xlim(ml_xlim_min, ml_xlim_max)
    ax0.set_ylim(float(ap_max), float(ap_min))
    ax0.set_yticks(ap_lines.tolist())
    ax0.tick_params(axis="both", labelsize=16)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_anchor("W")
    for spine in ax0.spines.values():
        spine.set_visible(False)

    cmap_neomeso = plt.cm.colors.ListedColormap(["#ff00ff"])
    fig.canvas.draw()
    legend_pos = ax0.get_position()
    legend_pos_orig = ax0.get_position(original=True)
    panel_x0_base = float(legend_pos.x1) - 0.008
    panel_w_base = 0.995 - panel_x0_base
    panel_x0 = float(panel_x0_base - 0.5 * (panel_scale - 1.0) * panel_w_base)
    panel_w = float(panel_w_base * panel_scale)
    if panel_w <= 0.05:
        raise ValueError("Figure layout too narrow for coronal panel column; increase figure width.")
    if ap_lines.size >= 2:
        step_um = float(np.nanmedian(np.diff(ap_lines.astype(np.float64, copy=False))))
        frac = float(step_um) / (float(ap_max) - float(ap_min))
        panel_h = float(legend_pos_orig.height) * frac * 0.98
    else:
        panel_h = float(legend_pos_orig.height) * 0.12
    panel_h = float(np.clip(panel_h, 0.055, 0.22))
    panel_h = float(panel_h * panel_scale)

    from matplotlib.patches import ConnectionPatch  # noqa: E402

    panel_axes: list[tuple[float, "plt.Axes"]] = []
    y0, y1 = (float(ax0.get_ylim()[0]), float(ax0.get_ylim()[1]))
    denom = float(y0 - y1)
    if not np.isfinite(denom) or abs(denom) < 1.0e-9:
        raise ValueError(f"Invalid ax0 ylim after aspect set: {ax0.get_ylim()}")
    for panel in slice_panels:
        ap_um, slice_i, img2, _cortex2, neomeso2, midline_xy = panel
        frac_y = (float(y0) - float(ap_um)) / denom
        center_y = float(legend_pos.y0) + float(legend_pos.height) * float(frac_y)
        panel_y0 = float(np.clip(center_y - 0.5 * panel_h, 0.0, 1.0 - panel_h))
        ax = fig.add_axes([panel_x0, panel_y0, panel_w, panel_h])
        ax.imshow(img2, cmap="gray", interpolation="nearest", origin="upper", vmin=0.0, vmax=1.0, alpha=0.8)
        neomeso_im = ax.imshow(
            neomeso2,
            cmap=cmap_neomeso,
            interpolation="nearest",
            alpha=0.25,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        neomeso_im.cmap.set_bad(alpha=0.0)
        if midline_xy.ndim == 2 and midline_xy.shape[0] >= 2 and midline_xy.shape[1] == 2:
            finite = np.isfinite(midline_xy[:, 0]) & np.isfinite(midline_xy[:, 1])
            if int(np.count_nonzero(finite)) >= 2:
                xy = midline_xy[finite]
                seg = np.stack([xy[:-1], xy[1:]], axis=1)
                lc_all = LineCollection(
                    seg,
                    colors=[(1.0, 1.0, 0.0, 0.5)],
                    linewidths=1.4,
                    zorder=5,
                )
                ax.add_collection(lc_all)
                seg_mid = 0.5 * (seg[:, 0, :] + seg[:, 1, :])
                h2, w2 = int(neomeso2.shape[0]), int(neomeso2.shape[1])
                mx = np.clip(np.rint(seg_mid[:, 0]).astype(np.int64, copy=False), 0, w2 - 1)
                my = np.clip(np.rint(seg_mid[:, 1]).astype(np.int64, copy=False), 0, h2 - 1)
                neo_seg = np.isfinite(neomeso2[my, mx]) & (neomeso2[my, mx] > 0.5)
                if np.any(neo_seg):
                    lc_neo = LineCollection(
                        seg[neo_seg],
                        colors=[(1.0, 1.0, 0.0, 0.95)],
                        linewidths=2.2,
                        zorder=6,
                    )
                    ax.add_collection(lc_neo)

                idxs = np.flatnonzero(np.asarray(data.slice_keys, dtype=np.int32) == int(slice_i))
                if idxs.size == 1:
                    idx = int(idxs[0])
                    mid_idx = int(np.nanargmin(np.abs(np.asarray(data.ml_um_at_t[idx], dtype=np.float64))))
                    if 0 <= mid_idx < int(midline_xy.shape[0]) and np.isfinite(midline_xy[mid_idx]).all():
                        ax.plot(
                            [float(midline_xy[mid_idx, 0])],
                            [float(midline_xy[mid_idx, 1])],
                            marker="o",
                            markersize=5.0,
                            color="#0000ff",
                            alpha=0.95,
                            markeredgecolor="#ffffff",
                            markeredgewidth=0.8,
                            zorder=10,
                        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.text(
            0.02,
            0.06,
            f"AP {ap_um:.0f}um (i={slice_i})",
            transform=ax.transAxes,
            fontsize=7,
            color="w",
            ha="left",
            va="bottom",
        )
        panel_axes.append((float(ap_um), ax))

    x_anchor = float(ax0.get_xlim()[1])
    for ap_um, axp in panel_axes:
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_anchor, float(ap_um)),
                coordsA=ax0.transData,
                xyB=(0.0, 0.5),
                coordsB=axp.transAxes,
                arrowstyle="-",
                linestyle=":",
                linewidth=1.0,
                color="#000000",
                alpha=0.65,
                zorder=30,
                clip_on=False,
            )
        )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_ap_ml_mapping_figure_sagittal(
    *,
    out_png: Path,
    data: SurfaceMappingData,
    b_const: float,
    outdir: Path,
    ap_hline_step_um: float,
    atlas_name: str,
    brainglobe_config_dir: Path,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.collections import LineCollection  # noqa: E402
    from matplotlib.patches import ConnectionPatch  # noqa: E402

    legend_rgb = _build_legend_rgb(ap_range_um=data.ap_range_um, ml_range_um=data.ml_range_um, b_const=float(b_const))
    support_mask = _build_legend_range_mask(
        data=data,
        row_mask=data.support_mask_tall,
        n_ap=int(legend_rgb.shape[1]),
        n_ml=int(legend_rgb.shape[0]),
    )
    neomeso_mask = _build_legend_range_mask(
        data=data,
        row_mask=data.neomeso_mask_tall,
        n_ap=int(legend_rgb.shape[1]),
        n_ml=int(legend_rgb.shape[0]),
    )
    neomeso_mask = neomeso_mask & support_mask
    legend_display = np.ones_like(legend_rgb, dtype=np.float64)
    legend_display[support_mask] = legend_rgb[support_mask]

    ap_lines = _ap_hline_values(ap_range_um=data.ap_range_um, step_um=float(ap_hline_step_um))
    if ap_lines.size > 1:
        ap_lines = ap_lines[1:]
    panels_all = _prepare_sagittal_slice_panels(
        outdir=Path(outdir),
        data=data,
        atlas_name=str(atlas_name),
        brainglobe_config_dir=Path(brainglobe_config_dir),
        sagittal_k_step=int(sagittal_k_step),
        sagittal_n_sample=int(sagittal_n_sample),
    )
    if not panels_all:
        raise ValueError("No sagittal panels available.")

    ks_arr = np.asarray([p.slice_k for p in panels_all], dtype=np.int32)
    ap0_arr = np.asarray([p.ap0_um for p in panels_all], dtype=np.float64)
    selected_ks: list[int] = []
    used: set[int] = set()
    for ap_t in np.asarray(ap_lines, dtype=np.float64).tolist():
        d = np.abs(ap0_arr - float(ap_t))
        order = np.argsort(d)
        picked = None
        for oi in order.tolist():
            k = int(ks_arr[int(oi)])
            if k not in used:
                picked = k
                break
        if picked is not None:
            used.add(int(picked))
            selected_ks.append(int(picked))
    panels_show = [p for p in panels_all if int(p.slice_k) in set(selected_ks)]
    if not panels_show:
        raise ValueError("No sagittal panels selected for display.")
    panels_show.sort(key=lambda p: float(p.ap0_um), reverse=True)

    fig_w_in = 8.4
    # Keep AP/ML 2D axis size exactly matched to coronal figure geometry.
    axes_w_frac = 0.58
    axes_h_frac = 0.88
    coronal_panel_scale = 1.5
    target_panel_h_in = 1.85
    if ap_lines.size >= 2:
        step_um = float(np.nanmedian(np.diff(ap_lines.astype(np.float64, copy=False))))
        frac = float(step_um) / (float(data.ap_range_um[1]) - float(data.ap_range_um[0]))
        panel_h_frac = float(axes_h_frac) * float(frac) * 0.98
    else:
        panel_h_frac = float(axes_h_frac) * 0.12
    panel_h_frac = float(np.clip(panel_h_frac, 0.055, 0.22)) * float(coronal_panel_scale)
    fig_h_coronal_in = float(target_panel_h_in) / float(panel_h_frac)
    fig_h_coronal_in = float(max(fig_h_coronal_in, fig_w_in * (axes_w_frac / axes_h_frac) * 0.35))

    axis_h_in = float(fig_h_coronal_in) * float(axes_h_frac)
    top_margin_in = float(fig_h_coronal_in) * (1.0 - (0.08 + float(axes_h_frac)))
    panel_y0_in = 0.20
    panel_h_in = 2.28
    bottom_band_in = panel_y0_in + panel_h_in + 0.15
    fig_h_in = bottom_band_in + axis_h_in + top_margin_in

    ax0_y0 = bottom_band_in / fig_h_in
    ax0_h_frac = axis_h_in / fig_h_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=200)
    ax0 = fig.add_axes([0.08, ax0_y0, axes_w_frac, ax0_h_frac])

    ap_min, ap_max = data.ap_range_um
    ml_min, ml_max = data.ml_range_um
    ml_supported = np.asarray(data.ml_um_at_t, dtype=np.float64)[np.asarray(data.support_mask_tall, dtype=bool)]
    ml_supported = ml_supported[np.isfinite(ml_supported)]
    if ml_supported.size == 0:
        raise ValueError("No supported (t_all) ML values; cannot set legend x-limits.")
    ml_xlim_min = float(np.min(ml_supported))
    ml_xlim_max = float(np.max(ml_supported))
    pad = 0.03 * float(ml_xlim_max - ml_xlim_min) if ml_xlim_max > ml_xlim_min else 100.0
    ml_xlim_min -= pad
    ml_xlim_max += pad
    ml_xlim_min = float(min(ml_xlim_min, 0.0))
    ml_xlim_max = float(max(ml_xlim_max, 0.0))

    legend_display_t = np.transpose(legend_display, (1, 0, 2))
    ax0.imshow(
        legend_display_t,
        origin="lower",
        extent=[ml_min, ml_max, ap_min, ap_max],
        aspect="auto",
        alpha=0.65,
    )
    support_overlay = np.transpose(support_mask)
    neomeso_overlay = np.full((legend_rgb.shape[1], legend_rgb.shape[0]), np.nan, dtype=np.float64)
    neomeso_overlay[support_overlay] = 0.0
    neomeso_overlay[np.transpose(neomeso_mask)] = 1.0
    neomeso_im = ax0.imshow(
        neomeso_overlay,
        origin="lower",
        extent=[ml_min, ml_max, ap_min, ap_max],
        aspect="auto",
        cmap="gray",
        alpha=0.35,
        vmin=0.0,
        vmax=1.0,
    )
    neomeso_im.cmap.set_bad(alpha=0.0)

    for ap in ap_lines.tolist():
        ax0.axhline(float(ap), color="#000000", linestyle=":", linewidth=0.9, alpha=0.9)

    # Sagittal slice overlays: curves in the AP/ML chart.
    for p in panels_all:
        x = np.asarray(p.ml_curve, dtype=np.float64)
        y = np.asarray(p.ap_curve, dtype=np.float64)
        neo = np.asarray(p.neomeso_curve, dtype=bool)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) < 2:
            continue
        xy = np.column_stack([x[finite], y[finite]]).astype(np.float64, copy=False)
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        # Base curve (t_all support).
        ax0.add_collection(
            LineCollection(
                seg,
                colors=[(1.0, 1.0, 0.0, 0.5)],
                linewidths=1.2,
                zorder=6,
            )
        )
        # Highlight neomeso subset.
        neo_f = neo[finite]
        if neo_f.size >= 2:
            neo_seg = neo_f[:-1] & neo_f[1:]
            if np.any(neo_seg):
                ax0.add_collection(
                    LineCollection(
                        seg[neo_seg],
                        colors=[(1.0, 1.0, 0.0, 0.95)],
                        linewidths=2.0,
                        zorder=7,
                    )
                )

    ax0.set_xlabel("ML (um)", fontsize=16)
    ax0.set_ylabel("AP (um)", fontsize=16)
    ax0.set_xlim(ml_xlim_min, ml_xlim_max)
    ax0.set_ylim(float(ap_max), float(ap_min))
    ax0.set_yticks(ap_lines.tolist())
    ax0.tick_params(axis="both", labelsize=16)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_anchor("W")
    for spine in ax0.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    legend_pos = ax0.get_position()
    legend_pos_orig = ax0.get_position(original=True)
    n_show = int(len(panels_show))
    panel_gap = 0.006
    panel_total_w = float(legend_pos_orig.width)
    panel_scale = 1.9
    panel_w_base = (panel_total_w - float(max(0, n_show - 1)) * panel_gap) / float(max(1, n_show))
    panel_w = float(panel_w_base) * float(panel_scale)
    panel_h = (panel_h_in / fig_h_in) * float(panel_scale)
    panel_y0 = float(max(0.005, float(legend_pos.y0) - panel_h + 0.006))
    if panel_w <= 0.03:
        raise ValueError("Figure layout too narrow for horizontal sagittal panels; reduce panel count.")

    cmap_neomeso = plt.cm.colors.ListedColormap(["#ff00ff"])
    panel_axes: list[tuple[SagittalPanel, "plt.Axes"]] = []
    # Draw right panels first so left panels are rendered on top.
    for idx_panel in range(n_show - 1, -1, -1):
        p = panels_show[idx_panel]
        center_x = float(legend_pos_orig.x0) + float(idx_panel) * (panel_w_base + panel_gap) + 0.5 * float(panel_w_base)
        panel_x0 = center_x - 0.5 * float(panel_w)
        axp = fig.add_axes([panel_x0, panel_y0, panel_w, panel_h])
        img2_rot = np.rot90(np.asarray(p.img2), k=-1)
        neomeso2_rot = np.rot90(np.asarray(p.neomeso2), k=-1)
        h_orig, _w_orig = int(np.asarray(p.img2).shape[0]), int(np.asarray(p.img2).shape[1])
        w_rot = int(img2_rot.shape[1])
        crop_frac = 0.55 if idx_panel >= int(max(0, n_show - 2)) else 0.50
        crop_x0 = int(np.floor(float(crop_frac) * float(w_rot)))
        img2_rot = img2_rot[:, crop_x0:]
        neomeso2_rot = neomeso2_rot[:, crop_x0:]

        axp.imshow(img2_rot, cmap="gray", interpolation="nearest", origin="upper", vmin=0.0, vmax=1.0, alpha=0.8)
        neomeso_im = axp.imshow(
            neomeso2_rot,
            cmap=cmap_neomeso,
            interpolation="nearest",
            alpha=0.25,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        neomeso_im.cmap.set_bad(alpha=0.0)

        xy = np.asarray(p.midline_xy_local, dtype=np.float64)
        if xy.ndim == 2 and xy.shape[0] >= 2 and xy.shape[1] == 2:
            finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            if int(np.count_nonzero(finite)) >= 2:
                xyp_raw = xy[finite]
                xyp = np.empty_like(xyp_raw)
                xyp[:, 0] = (float(h_orig) - 1.0) - xyp_raw[:, 1]
                xyp[:, 1] = xyp_raw[:, 0]
                xyp[:, 0] -= float(crop_x0)
                seg = np.stack([xyp[:-1], xyp[1:]], axis=1)
                axp.add_collection(
                    LineCollection(
                        seg,
                        colors=[(1.0, 1.0, 0.0, 0.5)],
                        linewidths=1.4,
                        zorder=5,
                    )
                )
                seg_mid = 0.5 * (seg[:, 0, :] + seg[:, 1, :])
                h2, w2 = int(neomeso2_rot.shape[0]), int(neomeso2_rot.shape[1])
                mx = np.clip(np.rint(seg_mid[:, 0]).astype(np.int64, copy=False), 0, w2 - 1)
                my = np.clip(np.rint(seg_mid[:, 1]).astype(np.int64, copy=False), 0, h2 - 1)
                neo_seg = np.isfinite(neomeso2_rot[my, mx]) & (neomeso2_rot[my, mx] > 0.5)
                if np.any(neo_seg):
                    axp.add_collection(
                        LineCollection(
                            seg[neo_seg],
                            colors=[(1.0, 1.0, 0.0, 0.95)],
                            linewidths=2.2,
                            zorder=6,
                        )
                    )

        if p.dot_xy_local is not None:
            dot_x = (float(h_orig) - 1.0) - float(p.dot_xy_local[1])
            dot_y = float(p.dot_xy_local[0])
            dot_x -= float(crop_x0)
            axp.plot(
                [dot_x],
                [dot_y],
                marker="o",
                markersize=5.0,
                color="#0000ff",
                alpha=0.95,
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
                zorder=10,
            )

        axp.set_xticks([])
        axp.set_yticks([])
        axp.set_axis_off()
        axp.text(
            0.02,
            0.06,
            f"k={int(p.slice_k)}",
            transform=axp.transAxes,
            fontsize=7,
            color="w",
            ha="left",
            va="bottom",
        )
        panel_axes.append((p, axp))

    # Link each panel to its sagittal curve at AP=3000um.
    ap_anchor_um = 3000.0
    for p, axp in panel_axes:
        x = np.asarray(p.ml_curve, dtype=np.float64)
        y = np.asarray(p.ap_curve, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) < 2:
            continue
        xf = x[finite]
        yf = y[finite]

        # Prefer true AP=3000 intersections; fallback to nearest sampled point.
        x_anchor = None
        y_anchor = None
        crossings = (yf[:-1] - ap_anchor_um) * (yf[1:] - ap_anchor_um) <= 0.0
        if np.any(crossings):
            x_hits: list[float] = []
            for i in np.flatnonzero(crossings).tolist():
                y0 = float(yf[int(i)])
                y1 = float(yf[int(i + 1)])
                x0 = float(xf[int(i)])
                x1 = float(xf[int(i + 1)])
                if abs(y1 - y0) < 1.0e-9:
                    x_hits.append(0.5 * (x0 + x1))
                else:
                    t = (ap_anchor_um - y0) / (y1 - y0)
                    x_hits.append(x0 + t * (x1 - x0))
            if x_hits:
                x_arr = np.asarray(x_hits, dtype=np.float64)
                x_anchor = float(x_arr[int(np.argmin(np.abs(x_arr)))])
                y_anchor = float(ap_anchor_um)
        if x_anchor is None or y_anchor is None:
            idx = int(np.argmin(np.abs(yf - ap_anchor_um)))
            x_anchor = float(xf[idx])
            y_anchor = float(yf[idx])

        fig.add_artist(
            ConnectionPatch(
                xyA=(x_anchor, y_anchor),
                coordsA=ax0.transData,
                xyB=(0.5, 1.0),
                coordsB=axp.transAxes,
                arrowstyle="-",
                linestyle=":",
                linewidth=1.0,
                color="#000000",
                alpha=0.65,
                zorder=30,
                clip_on=False,
            )
        )

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render AP/ML bivariate gradient and compare with coronal atlas panels (brain-cropped)."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Midsurface output directory containing coronal_midline_columns.csv and ap_axis_um_from_strips.npz.",
    )
    p.add_argument("--slice-i-min", type=int, default=161, help="Minimum coronal slice_i to include (inclusive).")
    p.add_argument("--slice-i-max", type=int, default=305, help="Maximum coronal slice_i to include (inclusive).")
    p.add_argument("--n-t", type=int, default=257, help="Number of samples along each coronal midline curve.")
    p.add_argument("--ref-t", type=float, default=0.5, help="Reference t_all in [0,1] used as ML origin anchor.")
    p.add_argument("--band-frac", type=float, default=0.15, help="DTW Sakoe-Chiba band as fraction of n_t.")
    p.add_argument("--b-const", type=float, default=0.25, help="Constant blue channel value in [0,1] for bivariate RGB.")
    p.add_argument(
        "--res-ijk-um",
        type=float,
        nargs=3,
        default=(20.0, 20.0, 20.0),
        metavar=("RI", "RJ", "RK"),
        help="Voxel size in um for (i,j,k); overrides outdir/resolution_ds_ijk_um.npy (default: 20 20 20).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <outdir>/ap_ml_mapping_surface.png).",
    )
    p.add_argument(
        "--output-sagittal",
        type=Path,
        default=None,
        help="Output PNG path for sagittal view (default: <outdir>/ap_ml_mapping_surface_sagittal.png).",
    )
    p.add_argument(
        "--write-sagittal",
        action="store_true",
        help="Also write sagittal view figure (sagittal slices map to curved lines in AP/ML space).",
    )
    p.add_argument(
        "--ap-hline-step-um",
        type=float,
        default=750.0,
        help="Draw AP hlines every N um on the 2D panel and render matching coronal slices.",
    )
    p.add_argument("--sagittal-k-step", type=int, default=10, help="Sagittal slice k step for panel/curve selection.")
    p.add_argument("--sagittal-n-sample", type=int, default=257, help="Number of samples along each sagittal curve (0..1).")
    p.add_argument(
        "--atlas-name",
        type=str,
        default="kim_dev_mouse_e15-5_lsfm_20um",
        help="BrainGlobe atlas name (used to load atlas.reference for panels).",
    )
    p.add_argument(
        "--brainglobe-config-dir",
        type=Path,
        default=Path("ccf/out/atlases/.brainglobe_config"),
        help="BRAINGLOBE_CONFIG_DIR for atlas cache/config.",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    out_png = args.output if args.output is not None else (outdir / "ap_ml_mapping_surface.png")
    out_png_sag = (
        (
            args.output_sagittal
            if args.output_sagittal is not None
            else (outdir / f"ap_ml_mapping_surface_sagittal_step{int(args.sagittal_k_step)}.png")
        )
        if bool(args.write_sagittal)
        else None
    )
    res_ijk_um = (float(args.res_ijk_um[0]), float(args.res_ijk_um[1]), float(args.res_ijk_um[2]))

    data = build_coronal_ap_ml_surface(
        outdir=outdir,
        slice_i_min=int(args.slice_i_min),
        slice_i_max=int(args.slice_i_max),
        n_t=int(args.n_t),
        ref_t=float(args.ref_t),
        band_frac=float(args.band_frac),
        b_const=float(args.b_const),
        res_ijk_um=res_ijk_um,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_ap_ml_mapping_figure(
        out_png=out_png,
        data=data,
        b_const=float(args.b_const),
        outdir=outdir,
        ap_hline_step_um=float(args.ap_hline_step_um),
        atlas_name=str(args.atlas_name),
        brainglobe_config_dir=Path(args.brainglobe_config_dir),
    )
    print(f"Wrote: {out_png}")
    if out_png_sag is not None:
        out_png_sag.parent.mkdir(parents=True, exist_ok=True)
        save_ap_ml_mapping_figure_sagittal(
            out_png=out_png_sag,
            data=data,
            b_const=float(args.b_const),
            outdir=outdir,
            ap_hline_step_um=float(args.ap_hline_step_um),
            atlas_name=str(args.atlas_name),
            brainglobe_config_dir=Path(args.brainglobe_config_dir),
            sagittal_k_step=int(args.sagittal_k_step),
            sagittal_n_sample=int(args.sagittal_n_sample),
        )
        print(f"Wrote: {out_png_sag}")
    print(
        f"Slices={int(data.slice_keys.size)} n_t={int(data.t_grid.size)} "
        f"support={int(np.count_nonzero(data.support_mask_tall))}/{int(data.support_mask_tall.size)} "
        f"AP=[{data.ap_range_um[0]:.1f},{data.ap_range_um[1]:.1f}]um "
        f"ML=[{data.ml_range_um[0]:.1f},{data.ml_range_um[1]:.1f}]um"
    )


if __name__ == "__main__":
    main()
