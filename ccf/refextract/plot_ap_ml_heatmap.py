from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path

import numpy as np


COORDS_SCRIPT = Path(__file__).resolve().with_name("midsurface_coords.py")


@dataclass(frozen=True)
class ApMlNativeGrid:
    ap_um: np.ndarray
    ml_um: np.ndarray
    support_mask: np.ndarray
    coronal_slice_i: np.ndarray
    t_grid: np.ndarray
    ml_um_signed_at_t: np.ndarray


def _load_midsurface_coords_module():
    spec = importlib.util.spec_from_file_location("midsurface_coords", COORDS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {COORDS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_ap_ml_support_mask_native_grid(
    *,
    outdir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    n_ml: int,
    ref_t: float,
    band_frac: float,
    res_ijk_um: tuple[float, float, float] | None,
    restrict_t_neomeso: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an AP/ML grid and a boolean support mask from refextract midsurface artifacts.

    Returns:
      - ap_grid: (N_ap,) AP (um) centers (ordered by AP)
      - ml_grid: (N_ml,) ML (um) centers (anchored arclength along per-slice t)
      - support_mask: (N_ap, N_ml) True where the refextract surface is defined
    """
    details = compute_ap_ml_native_grid_details(
        outdir=outdir,
        slice_i_min=slice_i_min,
        slice_i_max=slice_i_max,
        n_t=n_t,
        n_ml=n_ml,
        ref_t=ref_t,
        band_frac=band_frac,
        res_ijk_um=res_ijk_um,
        restrict_t_neomeso=restrict_t_neomeso,
    )
    return details.ap_um, details.ml_um, details.support_mask


def compute_ap_ml_native_grid_details(
    *,
    outdir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    n_ml: int,
    ref_t: float,
    band_frac: float,
    res_ijk_um: tuple[float, float, float] | None,
    restrict_t_neomeso: bool,
) -> ApMlNativeGrid:
    """Compute AP/ML native-grid details needed for inverse mapping back to atlas ijk.

    Returns:
      - ap_um: (N_ap,) AP (um) centers
      - ml_um: (N_ml,) ML (um) centers
      - support_mask: (N_ap, N_ml) valid AP/ML support mask
      - coronal_slice_i: (N_ap,) corresponding coronal slice index per AP row
      - t_grid: (N_t,) midsurface t grid used for per-slice anchoring
      - ml_um_signed_at_t: (N_ap, N_t) signed anchored ML arclength at each t
    """
    outdir = Path(outdir)
    mod = _load_midsurface_coords_module()
    coronal = mod.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    if not coronal:
        raise ValueError(f"No coronal midline columns found under {outdir}")

    _res_i_um, res_j_um, res_k_um = _load_resolution_ds_ijk_um(outdir, res_ijk_um=res_ijk_um)

    ap_npz = outdir / "ap_axis_um_from_strips.npz"
    ap = np.load(ap_npz)
    slice_keys = np.asarray(ap["slice_keys"], dtype=np.int32).reshape(-1)
    ap_um = np.asarray(ap["ap_um"], dtype=np.float64).reshape(-1)
    if slice_keys.size != ap_um.size:
        raise ValueError(f"slice_keys/ap_um size mismatch in {ap_npz}")
    ap_by_slice = {int(s): float(v) for s, v in zip(slice_keys.tolist(), ap_um.tolist(), strict=True)}

    keys = [
        k
        for k in sorted(coronal.keys())
        if int(slice_i_min) <= int(k) <= int(slice_i_max) and int(k) in ap_by_slice
    ]
    if len(keys) < 2:
        raise ValueError("Not enough coronal slices after filtering by slice range and AP axis.")

    t_ranges: dict[int, list[tuple[float, float]]] = {}
    if restrict_t_neomeso:
        t_ranges_path = outdir / "coronal_neocortex_mesocortex_overlap_t_ranges.csv"
        t_ranges = _load_overlap_t_ranges_csv(t_ranges_path, slice_label="slice_i")
        if not t_ranges:
            raise FileNotFoundError(f"No overlap t ranges found in {t_ranges_path}")

    ref_t = float(ref_t)
    if not np.isfinite(ref_t) or not (0.0 <= ref_t <= 1.0):
        raise ValueError(f"ref_t must be in [0,1], got {ref_t}")

    ap_grid = np.asarray([ap_by_slice[int(k)] for k in keys], dtype=np.float64)
    order = np.argsort(ap_grid)
    keys = [keys[i] for i in order.tolist()]
    ap_grid = ap_grid[order]

    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    thickness_t = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)
    ml_um_signed_at_t = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)

    ref_slice_i = int(np.median(np.asarray(keys, dtype=np.int32)))
    if ref_slice_i not in keys:
        keys_arr = np.asarray(keys, dtype=np.int32)
        ref_slice_i = int(keys_arr[int(np.argmin(np.abs(keys_arr - int(ref_slice_i))))])

    ref_cols = coronal[int(ref_slice_i)]
    ref_y_rs, ref_x_rs, _ref_len_um = _resample_polyline_um(
        y=ref_cols.y,
        x=ref_cols.x,
        res_y_um=float(res_j_um),
        res_x_um=float(res_k_um),
        n=int(n_t),
    )
    ref_pts2_um_center = _center_2d(np.column_stack([ref_y_rs * float(res_j_um), ref_x_rs * float(res_k_um)]))
    ref_idx = int(np.clip(int(np.rint(ref_t * float(n_t - 1))), 0, int(n_t - 1)))
    band = int(max(4, int(round(float(band_frac) * float(n_t)))))

    for row, k in enumerate(keys):
        c = coronal[int(k)]
        thickness_t[row] = mod._interp_series_on_t_grid(t=c.t, values=c.thickness_um, t_grid=t_grid)

        y_rs, x_rs, total_len_um = _resample_polyline_um(
            y=c.y,
            x=c.x,
            res_y_um=float(res_j_um),
            res_x_um=float(res_k_um),
            n=int(n_t),
        )
        pts2_um = np.column_stack([y_rs * float(res_j_um), x_rs * float(res_k_um)]).astype(np.float64, copy=False)
        pts2_center = _center_2d(pts2_um)
        fwd = float(np.nanmean(np.sum((pts2_center - ref_pts2_um_center) ** 2, axis=1)))
        rev_center = _center_2d(pts2_um[::-1])
        rev = float(np.nanmean(np.sum((rev_center - ref_pts2_um_center) ** 2, axis=1)))
        if rev < fwd:
            pts2_center = rev_center

        band_try = int(band)
        for _ in range(3):
            try:
                path = _dtw_banded_path(pts2_center, ref_pts2_um_center, band=band_try)
                break
            except ValueError:
                band_try = int(min(int(n_t - 1), int(round(band_try * 1.75)) + 1))
        else:
            raise ValueError(f"DTW failed for slice {int(k)} vs ref slice {int(ref_slice_i)}; increase band_frac.")

        idx_map = _path_to_monotone_index_map(path, n=int(n_t))
        origin_idx_f = _invert_monotone_index_map_to_fractional_idx(idx_map=idx_map, ref_idx=int(ref_idx))
        origin_t0 = float(origin_idx_f) / float(int(n_t) - 1)
        origin_um = float(origin_t0) * float(total_len_um)
        ml_um_signed_at_t[row] = (t_grid * float(total_len_um)) - origin_um

        if restrict_t_neomeso:
            ranges = t_ranges.get(int(k), [])
            if not ranges:
                thickness_t[row, :] = np.nan
                ml_um_signed_at_t[row, :] = np.nan
            else:
                keep = np.zeros(t_grid.shape, dtype=bool)
                for t0, t1 in ranges:
                    t0f = float(np.clip(t0, 0.0, 1.0))
                    t1f = float(np.clip(t1, 0.0, 1.0))
                    if t1f < t0f:
                        t0f, t1f = t1f, t0f
                    if (t1f - t0f) <= 1.0e-9:
                        continue
                    keep |= (t_grid >= t0f) & (t_grid <= t1f)
                thickness_t[row, ~keep] = np.nan
                ml_um_signed_at_t[row, ~keep] = np.nan

    ml_min = float(np.nanmin(ml_um_signed_at_t))
    ml_max = float(np.nanmax(ml_um_signed_at_t))
    if not np.isfinite(ml_min) or not np.isfinite(ml_max) or not (ml_max > ml_min):
        raise ValueError(f"Invalid ML range computed from refextract: [{ml_min}, {ml_max}]")

    ml_grid = np.linspace(ml_min, ml_max, int(n_ml), dtype=np.float64)
    thickness = np.full((len(keys), ml_grid.size), np.nan, dtype=np.float64)
    for row in range(len(keys)):
        if restrict_t_neomeso:
            thickness[row] = _interp_series_on_grid_piecewise(x=ml_um_signed_at_t[row], y=thickness_t[row], x_grid=ml_grid)
        else:
            thickness[row] = _interp_series_on_grid(x=ml_um_signed_at_t[row], y=thickness_t[row], x_grid=ml_grid)

    support_mask = np.isfinite(thickness)
    return ApMlNativeGrid(
        ap_um=ap_grid.astype(np.float64, copy=False),
        ml_um=ml_grid.astype(np.float64, copy=False),
        support_mask=support_mask.astype(bool, copy=False),
        coronal_slice_i=np.asarray(keys, dtype=np.int32),
        t_grid=t_grid.astype(np.float64, copy=False),
        ml_um_signed_at_t=ml_um_signed_at_t.astype(np.float64, copy=False),
    )


def _invert_ml_um_to_t_piecewise(
    *,
    ml_um_signed_at_t_row: np.ndarray,
    t_grid: np.ndarray,
    ml_grid: np.ndarray,
    support_mask_row: np.ndarray,
    slice_i: int,
) -> np.ndarray:
    ml_vals = np.asarray(ml_um_signed_at_t_row, dtype=np.float64).reshape(-1)
    t_vals = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    ml_q = np.asarray(ml_grid, dtype=np.float64).reshape(-1)
    support = np.asarray(support_mask_row, dtype=bool).reshape(-1)
    if ml_vals.shape != t_vals.shape:
        raise ValueError(f"slice_i={int(slice_i)}: ml/t shape mismatch {ml_vals.shape} vs {t_vals.shape}")
    if support.shape != ml_q.shape:
        raise ValueError(f"slice_i={int(slice_i)}: support/ml_grid shape mismatch {support.shape} vs {ml_q.shape}")

    out = np.full(ml_q.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(ml_vals) & np.isfinite(t_vals)
    idx = np.flatnonzero(finite).astype(np.int64, copy=False)
    if idx.size < 2:
        if np.any(support):
            raise ValueError(f"slice_i={int(slice_i)}: not enough finite (ml,t) points to invert support row")
        return out

    cuts = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, cuts)
    for run in runs:
        if run.size < 2:
            continue
        ml_run = ml_vals[run]
        t_run = t_vals[run]
        d = np.diff(ml_run)
        if np.all(d > 0.0):
            pass
        elif np.all(d < 0.0):
            ml_run = ml_run[::-1]
            t_run = t_run[::-1]
        else:
            raise ValueError(f"slice_i={int(slice_i)}: non-monotone ML(t) segment encountered while inverting to t")

        if np.any(np.diff(ml_run) <= 0.0):
            raise ValueError(f"slice_i={int(slice_i)}: duplicate/non-increasing ML values prevent inversion")

        m = support & (ml_q >= float(ml_run[0])) & (ml_q <= float(ml_run[-1]))
        if not np.any(m):
            continue
        if np.any(np.isfinite(out[m])):
            raise ValueError(f"slice_i={int(slice_i)}: overlapping ML support segments while inverting to t")
        out[m] = np.interp(ml_q[m], ml_run, t_run).astype(np.float64, copy=False)

    missing = support & ~np.isfinite(out)
    if np.any(missing):
        raise ValueError(f"slice_i={int(slice_i)}: failed to invert ML->t for {int(np.count_nonzero(missing))} support bins")
    return out


def _interp_column_values_at_t(
    *,
    t_source: np.ndarray,
    values: np.ndarray,
    t_query: np.ndarray,
    slice_i: int,
    field_name: str,
) -> np.ndarray:
    t_src = np.asarray(t_source, dtype=np.float64).reshape(-1)
    v_src = np.asarray(values, dtype=np.float64).reshape(-1)
    t_q = np.asarray(t_query, dtype=np.float64).reshape(-1)
    if t_src.shape != v_src.shape:
        raise ValueError(f"slice_i={int(slice_i)} field={field_name}: t/value shape mismatch {t_src.shape} vs {v_src.shape}")
    finite = np.isfinite(t_src) & np.isfinite(v_src)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError(f"slice_i={int(slice_i)} field={field_name}: not enough finite samples for interpolation")
    t_f = t_src[finite]
    v_f = v_src[finite]
    order = np.argsort(t_f)
    t_f = t_f[order]
    v_f = v_f[order]
    t_u, uidx = np.unique(t_f, return_index=True)
    v_u = v_f[uidx]
    if t_u.size < 2:
        raise ValueError(f"slice_i={int(slice_i)} field={field_name}: t grid is degenerate")
    out = np.interp(t_q, t_u, v_u, left=np.nan, right=np.nan).astype(np.float64, copy=False)
    if np.any(~np.isfinite(out)):
        raise ValueError(f"slice_i={int(slice_i)} field={field_name}: interpolation produced non-finite values on support")
    return out


def compute_ap_ml_native_ijk_map(
    *,
    outdir: Path,
    grid: ApMlNativeGrid,
    r01: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map AP/ML native-grid support points back to atlas ijk coordinates."""
    outdir = Path(outdir)
    r01f = float(r01)
    if not np.isfinite(r01f) or not (0.0 <= r01f <= 1.0):
        raise ValueError(f"r01 must be in [0,1], got {r01f}")

    mod = _load_midsurface_coords_module()
    coronal = mod.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    if not coronal:
        raise ValueError(f"No coronal midline columns found under {outdir}")

    ap_um = np.asarray(grid.ap_um, dtype=np.float64).reshape(-1)
    ml_um = np.asarray(grid.ml_um, dtype=np.float64).reshape(-1)
    support_mask = np.asarray(grid.support_mask, dtype=bool)
    coronal_slice_i = np.asarray(grid.coronal_slice_i, dtype=np.int32).reshape(-1)
    t_grid = np.asarray(grid.t_grid, dtype=np.float64).reshape(-1)
    ml_um_signed_at_t = np.asarray(grid.ml_um_signed_at_t, dtype=np.float64)

    n_ap = int(ap_um.size)
    n_ml = int(ml_um.size)
    if support_mask.shape != (n_ap, n_ml):
        raise ValueError(f"support_mask shape {support_mask.shape} does not match expected {(n_ap, n_ml)}")
    if coronal_slice_i.shape != (n_ap,):
        raise ValueError(f"coronal_slice_i shape {coronal_slice_i.shape} does not match expected {(n_ap,)}")
    if ml_um_signed_at_t.shape != (n_ap, int(t_grid.size)):
        raise ValueError(
            f"ml_um_signed_at_t shape {ml_um_signed_at_t.shape} does not match expected {(n_ap, int(t_grid.size))}"
        )

    i_map = np.full((n_ap, n_ml), np.nan, dtype=np.float64)
    j_map = np.full((n_ap, n_ml), np.nan, dtype=np.float64)
    k_map = np.full((n_ap, n_ml), np.nan, dtype=np.float64)

    for row in range(n_ap):
        support_row = np.asarray(support_mask[row], dtype=bool)
        if not np.any(support_row):
            continue
        slice_i = int(coronal_slice_i[row])
        cols = coronal.get(slice_i)
        if cols is None:
            raise ValueError(f"slice_i={slice_i} missing from coronal_midline_columns.csv")

        t_at_ml = _invert_ml_um_to_t_piecewise(
            ml_um_signed_at_t_row=ml_um_signed_at_t[row],
            t_grid=t_grid,
            ml_grid=ml_um,
            support_mask_row=support_row,
            slice_i=slice_i,
        )
        t_query = t_at_ml[support_row]

        vent_y = _interp_column_values_at_t(
            t_source=cols.t,
            values=cols.vent_y,
            t_query=t_query,
            slice_i=slice_i,
            field_name="vent_y",
        )
        vent_x = _interp_column_values_at_t(
            t_source=cols.t,
            values=cols.vent_x,
            t_query=t_query,
            slice_i=slice_i,
            field_name="vent_x",
        )
        pia_y = _interp_column_values_at_t(
            t_source=cols.t,
            values=cols.pia_y,
            t_query=t_query,
            slice_i=slice_i,
            field_name="pia_y",
        )
        pia_x = _interp_column_values_at_t(
            t_source=cols.t,
            values=cols.pia_x,
            t_query=t_query,
            slice_i=slice_i,
            field_name="pia_x",
        )

        i_map[row, support_row] = float(slice_i)
        j_map[row, support_row] = vent_y + r01f * (pia_y - vent_y)
        k_map[row, support_row] = vent_x + r01f * (pia_x - vent_x)

    return i_map, j_map, k_map


def compute_ap_ml_support_mask_neomeso_on_native_grid(
    *,
    outdir: Path,
    grid: ApMlNativeGrid,
    t_neomeso_csv: Path | None = None,
) -> np.ndarray:
    """Compute a t_neomeso support mask on an existing native grid.

    This is intended for plotting: use a full `t_all` native grid for geometry, but
    restrict colored values to the neocortex+mesocortex overlap interval(s) by slice.
    """
    outdir = Path(outdir)
    t_ranges_path = (outdir / "coronal_neocortex_mesocortex_overlap_t_ranges.csv") if t_neomeso_csv is None else Path(t_neomeso_csv)
    t_ranges = _load_overlap_t_ranges_csv(t_ranges_path, slice_label="slice_i")
    if not t_ranges:
        raise FileNotFoundError(f"No overlap t ranges found in {t_ranges_path}")

    support_mask = np.asarray(grid.support_mask, dtype=bool)
    coronal_slice_i = np.asarray(grid.coronal_slice_i, dtype=np.int32).reshape(-1)
    t_grid = np.asarray(grid.t_grid, dtype=np.float64).reshape(-1)
    ml_um = np.asarray(grid.ml_um, dtype=np.float64).reshape(-1)
    ml_um_signed_at_t = np.asarray(grid.ml_um_signed_at_t, dtype=np.float64)

    n_ap = int(coronal_slice_i.size)
    n_ml = int(ml_um.size)
    if support_mask.shape != (n_ap, n_ml):
        raise ValueError(f"support_mask shape {support_mask.shape} does not match expected {(n_ap, n_ml)}")
    if ml_um_signed_at_t.shape != (n_ap, int(t_grid.size)):
        raise ValueError(
            f"ml_um_signed_at_t shape {ml_um_signed_at_t.shape} does not match expected {(n_ap, int(t_grid.size))}"
        )

    out = np.zeros((n_ap, n_ml), dtype=bool)
    for row in range(n_ap):
        slice_i = int(coronal_slice_i[row])
        ranges = t_ranges.get(slice_i, [])
        if not ranges:
            continue
        support_row = np.asarray(support_mask[row], dtype=bool)
        if not np.any(support_row):
            continue
        t_at_ml = _invert_ml_um_to_t_piecewise(
            ml_um_signed_at_t_row=ml_um_signed_at_t[row],
            t_grid=t_grid,
            ml_grid=ml_um,
            support_mask_row=support_row,
            slice_i=slice_i,
        )
        for t0, t1 in ranges:
            out[row] |= support_row & (t_at_ml >= float(t0)) & (t_at_ml <= float(t1))
    return out


def _edges_from_centers(x: np.ndarray) -> np.ndarray:
    """Convert monotone centers to edges for pcolormesh."""
    xc = np.asarray(x, dtype=np.float64).reshape(-1)
    if xc.size < 1:
        raise ValueError("Need at least 1 center value.")
    if xc.size == 1:
        dx = 1.0
        return np.asarray([xc[0] - 0.5 * dx, xc[0] + 0.5 * dx], dtype=np.float64)
    if not np.all(np.isfinite(xc)):
        raise ValueError("Non-finite center values.")
    if not np.all(np.diff(xc) > 0.0):
        raise ValueError("Center values must be strictly increasing.")

    mid = 0.5 * (xc[:-1] + xc[1:])
    first = xc[0] - (mid[0] - xc[0])
    last = xc[-1] + (xc[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(np.float64, copy=False)


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
            "or write resolution_ds_ijk_um.npy when generating the midsurface artifacts."
        )
    arr = np.load(path).astype(np.float64, copy=False).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"Expected {path} shape (3,), got {arr.shape}")
    ri, rj, rk = (float(arr[0]), float(arr[1]), float(arr[2]))
    if not (ri > 0.0 and rj > 0.0 and rk > 0.0):
        raise ValueError(f"Invalid resolution in {path}: {(ri, rj, rk)}")
    return (ri, rj, rk)


def _cum_arclen_um(y: np.ndarray, x: np.ndarray, *, res_y_um: float, res_x_um: float) -> np.ndarray:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    if yv.size < 2:
        return np.zeros((yv.size,), dtype=np.float64)
    dy = np.diff(yv) * float(res_y_um)
    dx = np.diff(xv) * float(res_x_um)
    seg = np.sqrt(dy * dy + dx * dx)
    s = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)])
    return s


def _center_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {pts.shape}")
    center = np.nanmedian(pts, axis=0)
    if not np.isfinite(center).all():
        center = np.nanmean(pts, axis=0)
    return pts - center[None, :]


def _resample_polyline_um(
    *,
    y: np.ndarray,
    x: np.ndarray,
    res_y_um: float,
    res_x_um: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample (y,x) polyline to n points using physical arc length in um.

    Returns (y_new, x_new, total_len_um), with coordinates in vox.
    """
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


def _dtw_banded_path(a: np.ndarray, b: np.ndarray, *, band: int) -> list[tuple[int, int]]:
    """Return a DTW path within a Sakoe-Chiba band (a and b are (N,2))."""
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
    ptr = np.full((n, n), -1, dtype=np.int8)  # 0=diag, 1=up, 2=left

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
        if not buckets[i]:
            continue
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
    """Invert a monotone (current_idx -> ref_idx) map to a fractional current index.

    `idx_map` is expected to be non-decreasing and integer-like. We invert by run-length
    encoding `idx_map` and mapping each ref index to the midpoint of its current-index
    run. This is smoother than picking the first matching index (which can jitter by 1
    sample when DTW warping changes slightly).
    """
    m = np.asarray(idx_map, dtype=np.int32).reshape(-1)
    n = int(m.size)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    # Ensure monotone non-decreasing (the DTW map builder should already enforce this).
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


def _interp_series_on_grid(*, x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xg = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if xv.size != yv.size:
        raise ValueError("x/y size mismatch")
    finite = np.isfinite(xv) & np.isfinite(yv)
    if int(np.count_nonzero(finite)) < 2:
        return np.full(xg.shape, np.nan, dtype=np.float64)
    xf = xv[finite]
    yf = yv[finite]
    order = np.argsort(xf)
    xf = xf[order]
    yf = yf[order]
    # De-duplicate x to keep interp stable.
    x_unique, idx = np.unique(xf, return_index=True)
    y_unique = yf[idx]
    if x_unique.size < 2:
        return np.full(xg.shape, np.nan, dtype=np.float64)
    return np.interp(xg, x_unique, y_unique, left=np.nan, right=np.nan).astype(np.float64, copy=False)


def _interp_series_on_grid_piecewise(*, x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Interpolate y(x) onto x_grid without bridging gaps (NaNs split segments)."""
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xg = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if xv.size != yv.size:
        raise ValueError("x/y size mismatch")

    out = np.full(xg.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(xv) & np.isfinite(yv)
    idx = np.flatnonzero(finite).astype(np.int64, copy=False)
    if idx.size < 2:
        return out

    cuts = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, cuts)
    for run in runs:
        if run.size < 2:
            continue
        xf = xv[run]
        yf = yv[run]
        order = np.argsort(xf)
        xf = xf[order]
        yf = yf[order]
        x_unique, uidx = np.unique(xf, return_index=True)
        y_unique = yf[uidx]
        if x_unique.size < 2:
            continue
        m = (xg >= float(x_unique[0])) & (xg <= float(x_unique[-1]))
        if np.any(m):
            out[m] = np.interp(xg[m], x_unique, y_unique).astype(np.float64, copy=False)
    return out


def _load_overlap_t_ranges_csv(path: Path, *, slice_label: str) -> dict[int, list[tuple[float, float]]]:
    """Load slice-local overlap ranges (`t_all`) from `*_overlap_t_ranges.csv`."""
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
            t0 = float(np.clip(t0, 0.0, 1.0))
            t1 = float(np.clip(t1, 0.0, 1.0))
            if t1 < t0:
                t0, t1 = t1, t0
            if (t1 - t0) <= 1.0e-9:
                continue
            out.setdefault(slice_idx, []).append((t0, t1))
    for slice_idx in out:
        ranges = sorted(out[slice_idx], key=lambda x: x[0])
        merged: list[tuple[float, float]] = []
        for t0, t1 in ranges:
            if not merged:
                merged.append((float(t0), float(t1)))
                continue
            p0, p1 = merged[-1]
            if float(t0) <= float(p1) + 1.0e-9:
                merged[-1] = (float(p0), float(max(float(p1), float(t1))))
            else:
                merged.append((float(t0), float(t1)))
        out[slice_idx] = merged
    return out


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


def _compute_sagittal_overlay_curves(
    *,
    outdir: Path,
    ap_slice_keys: np.ndarray,
    ap_um: np.ndarray,
    coronal_slice_keys: list[int],
    t_grid: np.ndarray,
    ml_um_signed_at_t: np.ndarray,
    k_step: int,
    n_sample: int,
    restrict_t_neomeso: bool,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], tuple[int, int]]:
    s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    if not s2c_path.exists():
        raise FileNotFoundError(f"Missing {s2c_path}. Build chart LUTs first with midsurface_coords.py.")

    source_slice_keys, lut_t_grid, target_slice_idx, target_t = _load_s2c_t2d(s2c_path)
    if source_slice_keys.size == 0:
        raise ValueError("No sagittal source_slice_keys in sagittal->coronal LUT.")

    k_step = int(k_step)
    if k_step <= 0:
        raise ValueError(f"k_step must be > 0, got {k_step}")

    k_supported_min = int(np.min(source_slice_keys))
    k_supported_max = int(np.max(source_slice_keys))
    requested = list(range(int(k_supported_min), int(k_supported_max) + 1, int(k_step)))
    use_ks = sorted(
        {
            int(source_slice_keys[int(np.argmin(np.abs(source_slice_keys.astype(np.float64) - float(k))))])
            for k in requested
        }
    )

    coronal_keys = np.asarray(coronal_slice_keys, dtype=np.float64)
    ap_keys = np.asarray(ap_slice_keys, dtype=np.float64).reshape(-1)
    ap_vals = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    t_s = np.linspace(0.0, 1.0, int(n_sample), dtype=np.float64)

    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for k in use_ks:
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_slice_keys,
            lut_t_grid=lut_t_grid,
            target_slice_idx=target_slice_idx,
            target_t=target_t,
            slice_k=int(k),
            t_s=t_s,
        )
        ap_curve = _interp_series_on_grid(x=ap_keys, y=ap_vals, x_grid=cor_slice_f)
        row_idx = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f[None, :]), axis=0)
        ml_curve = np.full_like(cor_t, np.nan, dtype=np.float64)
        for ridx in np.unique(row_idx).tolist():
            ridx_i = int(ridx)
            mask = row_idx == ridx_i
            ml_row = ml_um_signed_at_t[ridx_i]
            interp_fn = _interp_series_on_grid_piecewise if restrict_t_neomeso else _interp_series_on_grid
            ml_curve[mask] = interp_fn(x=t_grid, y=ml_row, x_grid=cor_t[mask])

        keep = np.isfinite(ap_curve) & np.isfinite(ml_curve)
        if int(np.count_nonzero(keep)) >= 2:
            ml_plot = np.where(keep, ml_curve, np.nan)
            ap_plot = np.where(keep, ap_curve, np.nan)
            curves.append((int(k), ml_plot, ap_plot))

    return curves, (k_supported_min, k_supported_max)


def _write_s2c_ml_vs_sagittal_t_per_coronal_slice(
    *,
    outdir: Path,
    coronal_slice_keys: list[int],
    t_grid: np.ndarray,
    ml_um_signed_at_t: np.ndarray,
    k_step: int,
    n_sample: int,
    restrict_t_neomeso: bool,
    slice_i_min: int,
    slice_i_max: int,
    ml_min_um: float,
    ml_max_um: float,
    out_prefix: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    if not s2c_path.exists():
        raise FileNotFoundError(f"Missing {s2c_path}. Build chart LUTs first with midsurface_coords.py.")

    source_slice_keys, lut_t_grid, target_slice_idx, target_t = _load_s2c_t2d(s2c_path)
    if source_slice_keys.size == 0:
        raise ValueError("No sagittal source_slice_keys in sagittal->coronal LUT.")

    k_step = int(k_step)
    if k_step <= 0:
        raise ValueError(f"k_step must be > 0, got {k_step}")

    k_supported_min = int(np.min(source_slice_keys))
    k_supported_max = int(np.max(source_slice_keys))
    requested = list(range(int(k_supported_min), int(k_supported_max) + 1, int(k_step)))
    use_ks = sorted(
        {
            int(source_slice_keys[int(np.argmin(np.abs(source_slice_keys.astype(np.float64) - float(k))))])
            for k in requested
        }
    )

    coronal_keys = np.asarray(coronal_slice_keys, dtype=np.float64)
    t_s = np.linspace(0.0, 1.0, int(n_sample), dtype=np.float64)

    slice_i_min = int(slice_i_min)
    slice_i_max = int(slice_i_max)
    if slice_i_max < slice_i_min:
        raise ValueError(f"slice_i_max must be >= slice_i_min, got {slice_i_min}..{slice_i_max}")
    ml_min_um = float(ml_min_um)
    ml_max_um = float(ml_max_um)
    if not np.isfinite(ml_min_um) or not np.isfinite(ml_max_um) or not (ml_max_um > ml_min_um):
        raise ValueError(f"Invalid ML filter range (um): [{ml_min_um}, {ml_max_um}]")

    interp_fn = _interp_series_on_grid_piecewise if restrict_t_neomeso else _interp_series_on_grid

    curves_by_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in use_ks:
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_slice_keys,
            lut_t_grid=lut_t_grid,
            target_slice_idx=target_slice_idx,
            target_t=target_t,
            slice_k=int(k),
            t_s=t_s,
        )
        row_idx = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f[None, :]), axis=0)
        ml_curve = np.full_like(cor_t, np.nan, dtype=np.float64)
        for ridx in np.unique(row_idx).tolist():
            ridx_i = int(ridx)
            mask = row_idx == ridx_i
            ml_row = ml_um_signed_at_t[ridx_i]
            ml_curve[mask] = interp_fn(x=t_grid, y=ml_row, x_grid=cor_t[mask])

        slice_i_point = np.asarray([int(coronal_slice_keys[int(r)]) for r in row_idx.tolist()], dtype=np.int32)
        keep = (
            np.isfinite(t_s)
            & np.isfinite(ml_curve)
            & (slice_i_point >= int(slice_i_min))
            & (slice_i_point <= int(slice_i_max))
            & (ml_curve >= float(ml_min_um))
            & (ml_curve <= float(ml_max_um))
        )
        if not np.any(keep):
            continue
        curves_by_k[int(k)] = (slice_i_point, np.where(keep, ml_curve, np.nan))

    out_paths: list[Path] = []
    if not curves_by_k:
        return out_paths

    k_arr = np.asarray(sorted(curves_by_k.keys()), dtype=np.int32)
    k_min = int(np.min(k_arr))
    k_max = int(np.max(k_arr))
    if k_min == k_max:
        norm = plt.Normalize(float(k_min) - 0.5, float(k_max) + 0.5)
    else:
        norm = plt.Normalize(float(k_min), float(k_max))
    cmap = plt.get_cmap("turbo")

    for slice_i in range(int(slice_i_min), int(slice_i_max) + 1):
        fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=170)
        n_plotted = 0
        for k in k_arr.tolist():
            slice_i_point, ml_curve = curves_by_k[int(k)]
            y_full = ml_curve
            if int(np.count_nonzero(np.isfinite(y_full))) < 2:
                continue
            color = cmap(norm(float(k)))

            # Draw the full (filtered) curve faintly so interpolation/continuity is visible,
            # then highlight the segment(s) that land on this specific coronal slice.
            ax.plot(t_s, y_full, color="#bdbdbd", linewidth=0.9, alpha=0.18, zorder=1)

            y_sel = np.where(slice_i_point == int(slice_i), y_full, np.nan)
            if int(np.count_nonzero(np.isfinite(y_sel))) < 2:
                continue
            ax.plot(t_s, y_sel, color=color, linewidth=1.05, alpha=0.9, zorder=2)
            n_plotted += 1

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.92, pad=0.02)
        cb.set_label("sagittal slice_k")

        ax.set_title(
            f"S2C overlay: ML vs sagittal t | slice_i={int(slice_i)} | "
            f"{float(ml_min_um):.0f}<=ML<= {float(ml_max_um):.0f} um | curves={int(n_plotted)}"
        )

        ax.set_xlabel("sagittal t (t_s; 0..1)")
        ax.set_ylabel("ML (um; anchored)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(float(ml_min_um), float(ml_max_um))
        ax.grid(True, linewidth=0.5, alpha=0.25)
        out_path = outdir / f"{out_prefix}_s2c_ml_vs_ts_slice{int(slice_i)}_ml{int(round(ml_min_um))}to{int(round(ml_max_um))}_kstep{int(k_step)}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def _write_s2c_ml_vs_sagittal_t_window(
    *,
    outdir: Path,
    coronal_slice_keys: list[int],
    t_grid: np.ndarray,
    ml_um_signed_at_t: np.ndarray,
    k_step: int,
    n_sample: int,
    restrict_t_neomeso: bool,
    slice_i_min: int,
    slice_i_max: int,
    ml_min_um: float,
    ml_max_um: float,
    out_prefix: str,
) -> Path | None:
    import matplotlib.pyplot as plt

    s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    if not s2c_path.exists():
        raise FileNotFoundError(f"Missing {s2c_path}. Build chart LUTs first with midsurface_coords.py.")

    source_slice_keys, lut_t_grid, target_slice_idx, target_t = _load_s2c_t2d(s2c_path)
    if source_slice_keys.size == 0:
        raise ValueError("No sagittal source_slice_keys in sagittal->coronal LUT.")

    k_step = int(k_step)
    if k_step <= 0:
        raise ValueError(f"k_step must be > 0, got {k_step}")

    k_supported_min = int(np.min(source_slice_keys))
    k_supported_max = int(np.max(source_slice_keys))
    requested = list(range(int(k_supported_min), int(k_supported_max) + 1, int(k_step)))
    use_ks = sorted(
        {
            int(source_slice_keys[int(np.argmin(np.abs(source_slice_keys.astype(np.float64) - float(k))))])
            for k in requested
        }
    )

    coronal_keys = np.asarray(coronal_slice_keys, dtype=np.float64)
    t_s = np.linspace(0.0, 1.0, int(n_sample), dtype=np.float64)

    slice_i_min = int(slice_i_min)
    slice_i_max = int(slice_i_max)
    if slice_i_max < slice_i_min:
        raise ValueError(f"slice_i_max must be >= slice_i_min, got {slice_i_min}..{slice_i_max}")
    ml_min_um = float(ml_min_um)
    ml_max_um = float(ml_max_um)
    if not np.isfinite(ml_min_um) or not np.isfinite(ml_max_um) or not (ml_max_um > ml_min_um):
        raise ValueError(f"Invalid ML filter range (um): [{ml_min_um}, {ml_max_um}]")

    interp_fn = _interp_series_on_grid_piecewise if restrict_t_neomeso else _interp_series_on_grid

    curves_by_k: dict[int, np.ndarray] = {}
    for k in use_ks:
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_slice_keys,
            lut_t_grid=lut_t_grid,
            target_slice_idx=target_slice_idx,
            target_t=target_t,
            slice_k=int(k),
            t_s=t_s,
        )
        row_idx = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f[None, :]), axis=0)
        ml_curve = np.full_like(cor_t, np.nan, dtype=np.float64)
        for ridx in np.unique(row_idx).tolist():
            ridx_i = int(ridx)
            mask = row_idx == ridx_i
            ml_row = ml_um_signed_at_t[ridx_i]
            ml_curve[mask] = interp_fn(x=t_grid, y=ml_row, x_grid=cor_t[mask])

        slice_i_point = np.asarray([int(coronal_slice_keys[int(r)]) for r in row_idx.tolist()], dtype=np.int32)
        keep = (
            np.isfinite(t_s)
            & np.isfinite(ml_curve)
            & (slice_i_point >= int(slice_i_min))
            & (slice_i_point <= int(slice_i_max))
            & (ml_curve >= float(ml_min_um))
            & (ml_curve <= float(ml_max_um))
        )
        if int(np.count_nonzero(keep)) < 2:
            continue
        curves_by_k[int(k)] = np.where(keep, ml_curve, np.nan)

    if not curves_by_k:
        return None

    k_arr = np.asarray(sorted(curves_by_k.keys()), dtype=np.int32)
    k_min = int(np.min(k_arr))
    k_max = int(np.max(k_arr))
    if k_min == k_max:
        norm = plt.Normalize(float(k_min) - 0.5, float(k_max) + 0.5)
    else:
        norm = plt.Normalize(float(k_min), float(k_max))
    cmap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=170)
    n_plotted = 0
    for k in k_arr.tolist():
        y = curves_by_k[int(k)]
        if int(np.count_nonzero(np.isfinite(y))) < 2:
            continue
        ax.plot(t_s, y, color=cmap(norm(float(k))), linewidth=1.05, alpha=0.9)
        n_plotted += 1

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.92, pad=0.02)
    cb.set_label("sagittal slice_k")

    ax.set_title(
        f"S2C overlay: ML vs sagittal t | slice_i={int(slice_i_min)}..{int(slice_i_max)} | "
        f"{float(ml_min_um):.0f}<=ML<= {float(ml_max_um):.0f} um | curves={int(n_plotted)}"
    )
    ax.set_xlabel("sagittal t (t_s; 0..1)")
    ax.set_ylabel("ML (um; anchored)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(ml_min_um), float(ml_max_um))
    ax.grid(True, linewidth=0.5, alpha=0.25)

    out_path = (
        outdir
        / f"{out_prefix}_s2c_ml_vs_ts_slice{int(slice_i_min)}to{int(slice_i_max)}_ml{int(round(ml_min_um))}to{int(round(ml_max_um))}_kstep{int(k_step)}.png"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _write_heatmaps(
    *,
    outdir: Path,
    ap_npz: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    n_ml: int,
    ml_max_um: float | None,
    ml_min_um: float | None,
    res_ijk_um: tuple[float, float, float] | None,
    ref_slice_i: int | None,
    ref_t: float,
    band_frac: float,
    out_prefix: str,
    restrict_t_neomeso: bool,
    t_neomeso_csv: Path | None,
    overlay_sagittal_k_step: int,
    overlay_sagittal_n_sample: int,
    write_s2c_ml_vs_sagittal_t_per_slice: bool,
    write_s2c_ml_vs_sagittal_t_window: bool,
    s2c_disp_slice_i_min: int,
    s2c_disp_slice_i_max: int,
    s2c_disp_ml_min_um: float,
    s2c_disp_ml_max_um: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    mod = _load_midsurface_coords_module()
    coronal = mod.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    if not coronal:
        raise ValueError(f"No coronal midline columns found under {outdir}")

    res_i_um, res_j_um, res_k_um = _load_resolution_ds_ijk_um(outdir, res_ijk_um=res_ijk_um)

    ap = np.load(ap_npz)
    slice_keys = np.asarray(ap["slice_keys"], dtype=np.int32).reshape(-1)
    ap_um = np.asarray(ap["ap_um"], dtype=np.float64).reshape(-1)
    if slice_keys.size != ap_um.size:
        raise ValueError(f"slice_keys/ap_um size mismatch in {ap_npz}")

    ap_by_slice = {int(s): float(v) for s, v in zip(slice_keys.tolist(), ap_um.tolist(), strict=True)}

    keys = [k for k in sorted(coronal.keys()) if int(slice_i_min) <= int(k) <= int(slice_i_max) and int(k) in ap_by_slice]
    if len(keys) < 2:
        raise ValueError("Not enough slices after filtering by range and ap axis.")

    t_ranges: dict[int, list[tuple[float, float]]] = {}
    if restrict_t_neomeso:
        t_ranges_path = (outdir / "coronal_neocortex_mesocortex_overlap_t_ranges.csv") if t_neomeso_csv is None else Path(t_neomeso_csv)
        t_ranges = _load_overlap_t_ranges_csv(t_ranges_path, slice_label="slice_i")
        if not t_ranges:
            raise FileNotFoundError(f"No overlap t ranges found in {t_ranges_path}")

    if ref_slice_i is None:
        ref_slice_i = int(np.median(np.asarray(keys, dtype=np.int32)))
    ref_slice_i = int(ref_slice_i)
    if ref_slice_i not in keys:
        # Snap to nearest available key.
        keys_arr = np.asarray(keys, dtype=np.int32)
        ref_slice_i = int(keys_arr[int(np.argmin(np.abs(keys_arr - int(ref_slice_i))))])

    ref_t = float(ref_t)
    if not np.isfinite(ref_t) or not (0.0 <= ref_t <= 1.0):
        raise ValueError(f"--ref-t must be in [0,1], got {ref_t}")

    # Use AP centers ordered by slice key.
    ap_centers = np.asarray([ap_by_slice[int(k)] for k in keys], dtype=np.float64)
    order = np.argsort(ap_centers)
    keys = [keys[i] for i in order.tolist()]
    ap_centers = ap_centers[order]

    t_grid = np.linspace(0.0, 1.0, int(n_t), dtype=np.float64)
    y_edges = _edges_from_centers(ap_centers)

    thickness_t = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)
    ml_len_um = np.full((len(keys),), np.nan, dtype=np.float64)
    ml_um_signed_at_t = np.full((len(keys), t_grid.size), np.nan, dtype=np.float64)

    # Prepare reference slice sequence for DTW anchoring (centered in-plane).
    ref_cols = coronal[int(ref_slice_i)]
    ref_y_rs, ref_x_rs, _ref_len_um = _resample_polyline_um(
        y=ref_cols.y,
        x=ref_cols.x,
        res_y_um=float(res_j_um),
        res_x_um=float(res_k_um),
        n=int(n_t),
    )
    ref_pts2_um_center = _center_2d(np.column_stack([ref_y_rs * float(res_j_um), ref_x_rs * float(res_k_um)]))
    ref_idx = int(np.clip(int(np.rint(ref_t * float(n_t - 1))), 0, int(n_t - 1)))
    band = int(max(4, int(round(float(band_frac) * float(n_t)))))

    for row, k in enumerate(keys):
        c = coronal[int(k)]
        thickness_t[row] = mod._interp_series_on_t_grid(t=c.t, values=c.thickness_um, t_grid=t_grid)
        # Physical ML arclength (um) along this slice's curve is total length; use t_grid * length.
        _y_rs, _x_rs, total_len_um = _resample_polyline_um(
            y=c.y,
            x=c.x,
            res_y_um=float(res_j_um),
            res_x_um=float(res_k_um),
            n=int(n_t),
        )
        ml_len_um[row] = float(total_len_um)

        # Find a stable origin along the curve by DTW-aligning to a constant AP reference slice.
        pts2_um = np.column_stack([_y_rs * float(res_j_um), _x_rs * float(res_k_um)]).astype(np.float64, copy=False)
        pts2_center = _center_2d(pts2_um)
        fwd = float(np.nanmean(np.sum((pts2_center - ref_pts2_um_center) ** 2, axis=1)))
        rev_center = _center_2d(pts2_um[::-1])
        rev = float(np.nanmean(np.sum((rev_center - ref_pts2_um_center) ** 2, axis=1)))
        if rev < fwd:
            pts2_center = rev_center

        band_try = int(band)
        for _ in range(3):
            try:
                path = _dtw_banded_path(pts2_center, ref_pts2_um_center, band=band_try)
                break
            except ValueError:
                band_try = int(min(int(n_t - 1), int(round(band_try * 1.75)) + 1))
        else:
            raise ValueError(f"DTW failed for slice {int(k)} vs ref slice {int(ref_slice_i)}; increase --band-frac.")
        idx_map = _path_to_monotone_index_map(path, n=int(n_t))  # current index -> ref index
        origin_idx_f = _invert_monotone_index_map_to_fractional_idx(idx_map=idx_map, ref_idx=int(ref_idx))
        origin_t0 = float(origin_idx_f) / float(int(n_t) - 1)
        origin_um = float(origin_t0) * float(total_len_um)
        ml_um_signed_at_t[row] = (t_grid * float(total_len_um)) - origin_um

        if restrict_t_neomeso:
            ranges = t_ranges.get(int(k), [])
            if not ranges:
                thickness_t[row, :] = np.nan
                ml_um_signed_at_t[row, :] = np.nan
                ml_len_um[row] = np.nan
            else:
                keep = np.zeros(t_grid.shape, dtype=bool)
                frac = 0.0
                for t0, t1 in ranges:
                    t0f = float(np.clip(t0, 0.0, 1.0))
                    t1f = float(np.clip(t1, 0.0, 1.0))
                    if t1f < t0f:
                        t0f, t1f = t1f, t0f
                    if (t1f - t0f) <= 1.0e-9:
                        continue
                    keep |= (t_grid >= t0f) & (t_grid <= t1f)
                    frac += float(t1f - t0f)
                thickness_t[row, ~keep] = np.nan
                ml_um_signed_at_t[row, ~keep] = np.nan
                ml_len_um[row] = float(total_len_um) * float(frac)

    if ml_max_um is None:
        ml_max = float(np.nanmax(ml_um_signed_at_t))
    else:
        ml_max = float(ml_max_um)
    if ml_min_um is None:
        ml_min = float(np.nanmin(ml_um_signed_at_t))
    else:
        ml_min = float(ml_min_um)
    if not np.isfinite(ml_min):
        raise ValueError("Invalid ML min (um).")
    if not np.isfinite(ml_max):
        raise ValueError(f"Invalid ML max (um): {ml_max}")
    if not (ml_max > ml_min):
        raise ValueError(f"Invalid ML range (um): [{ml_min}, {ml_max}]")

    ml_grid = np.linspace(ml_min, ml_max, int(n_ml), dtype=np.float64)
    x_edges = _edges_from_centers(ml_grid)

    thickness = np.full((len(keys), ml_grid.size), np.nan, dtype=np.float64)
    for row in range(len(keys)):
        if restrict_t_neomeso:
            thickness[row] = _interp_series_on_grid_piecewise(x=ml_um_signed_at_t[row], y=thickness_t[row], x_grid=ml_grid)
        else:
            thickness[row] = _interp_series_on_grid(x=ml_um_signed_at_t[row], y=thickness_t[row], x_grid=ml_grid)
    valid = np.isfinite(thickness).astype(np.float64)
    sagittal_curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    supported_k_range = (0, 0)
    if int(overlay_sagittal_k_step) > 0:
        sagittal_curves, supported_k_range = _compute_sagittal_overlay_curves(
            outdir=outdir,
            ap_slice_keys=slice_keys,
            ap_um=ap_um,
            coronal_slice_keys=keys,
            t_grid=t_grid,
            ml_um_signed_at_t=ml_um_signed_at_t,
            k_step=int(overlay_sagittal_k_step),
            n_sample=int(overlay_sagittal_n_sample),
            restrict_t_neomeso=bool(restrict_t_neomeso),
        )

    out_paths: list[Path] = []
    title_suffix = (
        f"coronal slices {int(slice_i_min)}-{int(slice_i_max)} (n={len(keys)}), n_t={int(n_t)}, "
        f"ref_slice_i={int(ref_slice_i)}, ref_t={float(ref_t):.3f}"
    )
    if int(overlay_sagittal_k_step) > 0 or bool(write_s2c_ml_vs_sagittal_t_per_slice) or bool(write_s2c_ml_vs_sagittal_t_window):
        s2c_path = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
        if s2c_path.exists():
            d = np.load(s2c_path)
            cmin_raw = int(np.asarray(d["coronal_slice_min"]).reshape(-1)[0]) if "coronal_slice_min" in d else -1
            cmax_raw = int(np.asarray(d["coronal_slice_max"]).reshape(-1)[0]) if "coronal_slice_max" in d else -1
            cmin = None if int(cmin_raw) < 0 else int(cmin_raw)
            cmax = None if int(cmax_raw) < 0 else int(cmax_raw)
            if cmin is not None or cmax is not None:
                parts: list[str] = []
                if cmin is not None:
                    parts.append(f">={int(cmin)}")
                if cmax is not None:
                    parts.append(f"<={int(cmax)}")
                title_suffix = title_suffix + f" | LUT coronal slice_i {''.join(parts)}"
    if restrict_t_neomeso:
        title_suffix = title_suffix + " | t restricted to neocortex+mesocortex overlap"

    # Thickness heatmap.
    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=160)
    m = ax.pcolormesh(x_edges, y_edges, thickness, shading="auto", cmap="viridis")
    ax.set_xlabel("mediolateral (um; arclength along t, anchored)")
    ax.set_ylabel("AP (um; optimized)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Thickness heatmap ({title_suffix})")
    ax.grid(False)
    cb = fig.colorbar(m, ax=ax, shrink=0.95, pad=0.02)
    cb.set_label("thickness_um")
    p1 = outdir / f"{out_prefix}_ml_um_anchored_thickness_heatmap.png"
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    out_paths.append(p1)

    # Validity heatmap (where thickness is finite).
    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=160)
    m = ax.pcolormesh(x_edges, y_edges, valid, shading="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_xlabel("mediolateral (um; arclength along t, anchored)")
    ax.set_ylabel("AP (um; optimized)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Coverage heatmap ({title_suffix})")
    ax.grid(False)
    cb = fig.colorbar(m, ax=ax, shrink=0.95, pad=0.02)
    cb.set_label("finite(thickness_um)")
    p2 = outdir / f"{out_prefix}_ml_um_anchored_coverage_heatmap.png"
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    out_paths.append(p2)

    if sagittal_curves:
        fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=160)
        m = ax.pcolormesh(x_edges, y_edges, valid, shading="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
        cmap_k = plt.get_cmap("turbo")
        n_curves = len(sagittal_curves)
        for idx, (_k, ml_curve, ap_curve) in enumerate(sagittal_curves):
            frac = float(idx) / float(max(1, n_curves - 1))
            ax.plot(ml_curve, ap_curve, color=cmap_k(frac), linewidth=0.9, alpha=0.9)

        ax.set_xlabel("mediolateral (um; arclength along t, anchored)")
        ax.set_ylabel("AP (um; optimized)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Coverage + sagittal curves ({title_suffix}) | "
            f"k={int(supported_k_range[0])}..{int(supported_k_range[1])} step={int(overlay_sagittal_k_step)}"
        )
        ax.grid(False)
        cb = fig.colorbar(m, ax=ax, shrink=0.95, pad=0.02)
        cb.set_label("finite(thickness_um)")
        p2_overlay = outdir / f"{out_prefix}_ml_um_anchored_coverage_heatmap_sagittal_step{int(overlay_sagittal_k_step)}.png"
        fig.tight_layout()
        fig.savefig(p2_overlay)
        plt.close(fig)
        out_paths.append(p2_overlay)

    # ML length vs AP (range visualization).
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=160)
    ap = ap_centers
    ax.plot(ap, ml_len_um, linewidth=1.2)
    ax.set_xlabel("AP (um; optimized)")
    ax.set_ylabel("ML total length (um)")
    ax.set_title(f"Per-slice ML length ({title_suffix})")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    p3 = outdir / f"{out_prefix}_ml_um_total_length_vs_ap.png"
    fig.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    out_paths.append(p3)

    if bool(write_s2c_ml_vs_sagittal_t_per_slice):
        if int(overlay_sagittal_k_step) <= 0:
            raise ValueError("--write-s2c-ml-vs-sagittal-t-per-slice requires --overlay-sagittal-k-step > 0")
        out_paths.extend(
            _write_s2c_ml_vs_sagittal_t_per_coronal_slice(
                outdir=outdir,
                coronal_slice_keys=keys,
                t_grid=t_grid,
                ml_um_signed_at_t=ml_um_signed_at_t,
                k_step=int(overlay_sagittal_k_step),
                n_sample=int(overlay_sagittal_n_sample),
                restrict_t_neomeso=bool(restrict_t_neomeso),
                slice_i_min=int(s2c_disp_slice_i_min),
                slice_i_max=int(s2c_disp_slice_i_max),
                ml_min_um=float(s2c_disp_ml_min_um),
                ml_max_um=float(s2c_disp_ml_max_um),
                out_prefix=str(out_prefix),
            )
        )

    if bool(write_s2c_ml_vs_sagittal_t_window):
        if int(overlay_sagittal_k_step) <= 0:
            raise ValueError("--write-s2c-ml-vs-sagittal-t-window requires --overlay-sagittal-k-step > 0")
        pth = _write_s2c_ml_vs_sagittal_t_window(
            outdir=outdir,
            coronal_slice_keys=keys,
            t_grid=t_grid,
            ml_um_signed_at_t=ml_um_signed_at_t,
            k_step=int(overlay_sagittal_k_step),
            n_sample=int(overlay_sagittal_n_sample),
            restrict_t_neomeso=bool(restrict_t_neomeso),
            slice_i_min=int(s2c_disp_slice_i_min),
            slice_i_max=int(s2c_disp_slice_i_max),
            ml_min_um=float(s2c_disp_ml_min_um),
            ml_max_um=float(s2c_disp_ml_max_um),
            out_prefix=str(out_prefix),
        )
        if pth is not None:
            out_paths.append(pth)

    return out_paths


def main() -> None:
    p = argparse.ArgumentParser(description="Plot AP vs mediolateral(t) heatmaps using optimized AP axis + midline columns.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Midsurface output directory containing coronal_midline_columns.csv.",
    )
    p.add_argument(
        "--ap-npz",
        type=Path,
        default=None,
        help="NPZ file containing slice_keys and ap_um (default: <outdir>/ap_axis_um_from_strips.npz).",
    )
    p.add_argument("--slice-i-min", type=int, default=161, help="Minimum coronal slice_i (inclusive).")
    p.add_argument("--slice-i-max", type=int, default=305, help="Maximum coronal slice_i (inclusive).")
    p.add_argument("--n-t", type=int, default=257, help="Number of t grid samples for the heatmap.")
    p.add_argument("--n-ml", type=int, default=512, help="Number of ML (um) grid samples for the heatmap.")
    p.add_argument("--ref-slice-i", type=int, default=None, help="Coronal slice_i to use as the ML anchoring reference.")
    p.add_argument(
        "--ref-t",
        type=float,
        default=0.5,
        help="Reference t (in [0,1]) on ref-slice that defines ML=0 across slices.",
    )
    p.add_argument(
        "--band-frac",
        type=float,
        default=0.15,
        help="DTW Sakoe-Chiba band as a fraction of n-t for anchoring.",
    )
    p.add_argument(
        "--ml-max-um",
        type=float,
        default=None,
        help="Override ML axis max in um (default: max per-slice ML length in range).",
    )
    p.add_argument(
        "--ml-min-um",
        type=float,
        default=None,
        help="Override ML axis min in um (default: min signed ML coordinate in range).",
    )
    p.add_argument(
        "--restrict-t-neomeso",
        action="store_true",
        help="Restrict to neocortex+mesocortex overlap t ranges (uses coronal_neocortex_mesocortex_overlap_t_ranges.csv).",
    )
    p.add_argument(
        "--t-neomeso-csv",
        type=Path,
        default=None,
        help="Override overlap t-ranges CSV path (default: <outdir>/coronal_neocortex_mesocortex_overlap_t_ranges.csv).",
    )
    p.add_argument(
        "--res-ijk-um",
        type=float,
        nargs=3,
        default=None,
        metavar=("RI", "RJ", "RK"),
        help="Voxel size in um for (i,j,k) in the DS grid; overrides outdir/resolution_ds_ijk_um.npy.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="ap_ml",
        help="Output filename prefix written under outdir (e.g. ap_ml_ml_um_thickness_heatmap.png).",
    )
    p.add_argument(
        "--overlay-sagittal-k-step",
        type=int,
        default=0,
        help="If >0, also write a coverage plot with sagittal curves overlaid every k-step in the LUT-supported k range.",
    )
    p.add_argument(
        "--overlay-sagittal-n-sample",
        type=int,
        default=401,
        help="Number of t samples per sagittal curve when overlaying sagittal slices.",
    )
    p.add_argument(
        "--write-s2c-ml-vs-sagittal-t-per-slice",
        action="store_true",
        help=(
            "Also write per-coronal-slice diagnostics plotting ML(um) vs sagittal t for the sagittal->coronal (s2c) LUT overlay "
            "(requires --overlay-sagittal-k-step > 0)."
        ),
    )
    p.add_argument(
        "--write-s2c-ml-vs-sagittal-t-window",
        action="store_true",
        help=(
            "Also write a single diagnostic plot of ML(um) vs sagittal t for each sagittal slice_k, "
            "filtered to the coronal slice window set by --s2c-disp-slice-i-min/max and ML window set by --s2c-disp-ml-min/max "
            "(requires --overlay-sagittal-k-step > 0)."
        ),
    )
    p.add_argument(
        "--write-s2c-displacement-per-slice",
        action="store_true",
        dest="write_s2c_ml_vs_sagittal_t_per_slice",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--s2c-disp-slice-i-min", type=int, default=161, help="Minimum coronal slice_i for s2c displacement plots.")
    p.add_argument(
        "--s2c-disp-slice-i-max",
        type=int,
        default=174,
        help="Maximum coronal slice_i for s2c debug plots (e.g. 174 means 'before 175').",
    )
    p.add_argument("--s2c-disp-ml-min-um", type=float, default=0.0, help="Minimum ML (um) to include in s2c debug plots.")
    p.add_argument("--s2c-disp-ml-max-um", type=float, default=750.0, help="Maximum ML (um) to include in s2c debug plots.")
    args = p.parse_args()

    outdir = args.outdir
    ap_npz = args.ap_npz if args.ap_npz is not None else (outdir / "ap_axis_um_from_strips.npz")
    if not ap_npz.exists():
        raise FileNotFoundError(f"Missing {ap_npz}. Run optimize_ap_axis_from_strips.py first.")

    res_ijk_um = None if args.res_ijk_um is None else (float(args.res_ijk_um[0]), float(args.res_ijk_um[1]), float(args.res_ijk_um[2]))
    out_paths = _write_heatmaps(
        outdir=outdir,
        ap_npz=ap_npz,
        slice_i_min=int(args.slice_i_min),
        slice_i_max=int(args.slice_i_max),
        n_t=int(args.n_t),
        n_ml=int(args.n_ml),
        ml_max_um=args.ml_max_um,
        ml_min_um=args.ml_min_um,
        res_ijk_um=res_ijk_um,
        ref_slice_i=args.ref_slice_i,
        ref_t=float(args.ref_t),
        band_frac=float(args.band_frac),
        out_prefix=str(args.out_prefix),
        restrict_t_neomeso=bool(args.restrict_t_neomeso),
        t_neomeso_csv=args.t_neomeso_csv,
        overlay_sagittal_k_step=int(args.overlay_sagittal_k_step),
        overlay_sagittal_n_sample=int(args.overlay_sagittal_n_sample),
        write_s2c_ml_vs_sagittal_t_per_slice=bool(args.write_s2c_ml_vs_sagittal_t_per_slice),
        write_s2c_ml_vs_sagittal_t_window=bool(args.write_s2c_ml_vs_sagittal_t_window),
        s2c_disp_slice_i_min=int(args.s2c_disp_slice_i_min),
        s2c_disp_slice_i_max=int(args.s2c_disp_slice_i_max),
        s2c_disp_ml_min_um=float(args.s2c_disp_ml_min_um),
        s2c_disp_ml_max_um=float(args.s2c_disp_ml_max_um),
    )
    for pth in out_paths:
        print(f"Wrote: {pth}")


if __name__ == "__main__":
    main()
