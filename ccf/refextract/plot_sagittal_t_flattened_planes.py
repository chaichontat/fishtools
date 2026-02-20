from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import CubicSpline


def _interp_with_linear_extrapolation(*, x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    xqv = np.asarray(xq, dtype=np.float64).reshape(-1)
    if xv.size != yv.size or xv.size < 2:
        raise ValueError("Need x and y arrays with the same length >= 2.")
    if not np.all(np.diff(xv) > 0.0):
        order = np.argsort(xv)
        xv = xv[order]
        yv = yv[order]
    out = np.interp(xqv, xv, yv).astype(np.float64, copy=False)
    left = xqv < float(xv[0])
    right = xqv > float(xv[-1])
    if np.any(left):
        slope = (float(yv[1]) - float(yv[0])) / (float(xv[1]) - float(xv[0]))
        out[left] = float(yv[0]) + slope * (xqv[left] - float(xv[0]))
    if np.any(right):
        slope = (float(yv[-1]) - float(yv[-2])) / (float(xv[-1]) - float(xv[-2]))
        out[right] = float(yv[-1]) + slope * (xqv[right] - float(xv[-1]))
    return out


def _center_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {pts.shape}")
    center = np.nanmedian(pts, axis=0)
    if not np.isfinite(center).all():
        center = np.nanmean(pts, axis=0)
    return pts - center[None, :]


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
    if yv.size != xv.size or yv.size < 2:
        raise ValueError("Need y and x arrays with the same length >= 2.")
    s = _cum_arclen_um(yv, xv, res_y_um=float(res_y_um), res_x_um=float(res_x_um))
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid polyline arc length.")
    s_grid = np.linspace(0.0, total, int(n), dtype=np.float64)
    y_new = np.interp(s_grid, s, yv).astype(np.float64, copy=False)
    x_new = np.interp(s_grid, s, xv).astype(np.float64, copy=False)
    return y_new, x_new, total


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
        raise ValueError("DTW failed; increase band.")

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
            raise RuntimeError(f"Unexpected DTW ptr={step} at {(i, j)}")
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


def _load_midline_paths_csv(path: Path, *, slice_label: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    grouped: dict[int, list[tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if slice_label not in row or "t" not in row or "y" not in row or "x" not in row:
                continue
            slice_val = float(row[slice_label])
            if not np.isfinite(slice_val):
                continue
            slice_idx = int(np.rint(slice_val))
            t_val = float(row["t"])
            y_val = float(row["y"])
            x_val = float(row["x"])
            if not np.isfinite(t_val) or not np.isfinite(y_val) or not np.isfinite(x_val):
                continue
            grouped.setdefault(slice_idx, []).append((t_val, y_val, x_val))

    for slice_idx, vals in grouped.items():
        if len(vals) < 2:
            continue
        vals_sorted = sorted(vals, key=lambda v: v[0])
        t_vals = np.asarray([v[0] for v in vals_sorted], dtype=np.float64)
        path_yx = np.asarray([[v[1], v[2]] for v in vals_sorted], dtype=np.float64)
        out[int(slice_idx)] = (path_yx, t_vals)
    return out


def _build_ml_anchor_for_coronal_slices(
    *,
    coronal_paths: dict[int, tuple[np.ndarray, np.ndarray]],
    slice_keys: np.ndarray,
    ref_slice_i: int,
    ref_t: float,
    n_t: int,
    band_frac: float,
    res_j_um: float,
    res_k_um: float,
) -> tuple[dict[int, float], dict[int, float]]:
    keys = [int(k) for k in np.asarray(slice_keys, dtype=np.int32).reshape(-1).tolist()]
    if int(ref_slice_i) not in coronal_paths:
        avail = np.asarray(sorted(coronal_paths.keys()), dtype=np.int32)
        if avail.size == 0:
            raise ValueError("No coronal midline paths available for ML anchoring.")
        ref_slice_i = int(avail[int(np.argmin(np.abs(avail - int(ref_slice_i))))])
    ref_t = float(ref_t)
    if not np.isfinite(ref_t) or not (0.0 <= ref_t <= 1.0):
        raise ValueError(f"ref_t must be in [0,1], got {ref_t}")
    ref_path, _ref_tvals = coronal_paths[int(ref_slice_i)]
    ref_y_rs, ref_x_rs, _ref_len_um = _resample_polyline_um(
        y=ref_path[:, 0],
        x=ref_path[:, 1],
        res_y_um=float(res_j_um),
        res_x_um=float(res_k_um),
        n=int(n_t),
    )
    ref_pts2 = _center_2d(np.column_stack([ref_y_rs * float(res_j_um), ref_x_rs * float(res_k_um)]))
    ref_idx = int(np.clip(int(np.rint(ref_t * float(n_t - 1))), 0, int(n_t - 1)))
    band = int(max(4, int(round(float(band_frac) * float(n_t)))))

    t0_by_slice: dict[int, float] = {}
    len_by_slice: dict[int, float] = {}
    for s in keys:
        row = coronal_paths.get(int(s))
        if row is None:
            continue
        path_yx, _t_vals = row
        y_rs, x_rs, total_len_um = _resample_polyline_um(
            y=path_yx[:, 0],
            x=path_yx[:, 1],
            res_y_um=float(res_j_um),
            res_x_um=float(res_k_um),
            n=int(n_t),
        )
        pts2 = np.column_stack([y_rs * float(res_j_um), x_rs * float(res_k_um)]).astype(np.float64, copy=False)
        pts2_center = _center_2d(pts2)
        fwd = float(np.nanmean(np.sum((pts2_center - ref_pts2) ** 2, axis=1)))
        rev_center = _center_2d(pts2[::-1])
        rev = float(np.nanmean(np.sum((rev_center - ref_pts2) ** 2, axis=1)))
        if rev < fwd:
            pts2_center = rev_center

        band_try = int(band)
        for _ in range(3):
            try:
                path = _dtw_banded_path(pts2_center, ref_pts2, band=band_try)
                break
            except ValueError:
                band_try = int(min(int(n_t - 1), int(round(band_try * 1.75)) + 1))
        else:
            raise ValueError(f"DTW failed for slice {int(s)}; increase band_frac.")

        idx_map = _path_to_monotone_index_map(path, n=int(n_t))  # current index -> ref index
        origin_idx_f = _invert_monotone_index_map_to_fractional_idx(idx_map=idx_map, ref_idx=int(ref_idx))
        t0_by_slice[int(s)] = float(origin_idx_f) / float(n_t - 1)
        len_by_slice[int(s)] = float(total_len_um)
    return t0_by_slice, len_by_slice


@dataclass(frozen=True)
class S2CLut:
    source_slice_keys: np.ndarray
    t_grid: np.ndarray
    target_slice_idx: np.ndarray
    target_t: np.ndarray


def _load_s2c_t2d(npz_path: Path) -> S2CLut:
    d = np.load(npz_path)
    return S2CLut(
        source_slice_keys=np.asarray(d["source_slice_keys"], dtype=np.int32),
        t_grid=np.asarray(d["t_grid"], dtype=np.float64),
        target_slice_idx=np.asarray(d["target_slice_idx"], dtype=np.float64),
        target_t=np.asarray(d["target_t"], dtype=np.float64),
    )


def _map_sagittal_to_coronal_t2d(*, s2c: S2CLut, slice_k: int, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(s2c.source_slice_keys, dtype=np.int32).reshape(-1)
    if keys.size == 0:
        raise ValueError("Empty source_slice_keys in sagittal->coronal LUT.")
    row = int(np.argmin(np.abs(keys.astype(np.float64) - float(slice_k))))
    t_grid = np.asarray(s2c.t_grid, dtype=np.float64).reshape(-1)
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    t_clip = np.clip(t, 0.0, 1.0)
    slice_row = np.asarray(s2c.target_slice_idx[row], dtype=np.float64).reshape(-1)
    t_row = np.asarray(s2c.target_t[row], dtype=np.float64).reshape(-1)
    if slice_row.size != t_grid.size or t_row.size != t_grid.size:
        raise ValueError("Unexpected sagittal->coronal LUT row shape.")
    out_slice = np.interp(t_clip, t_grid, slice_row).astype(np.float64, copy=False)
    out_t = np.interp(t_clip, t_grid, t_row).astype(np.float64, copy=False)
    return out_slice, out_t


def _t_at_k_on_coronal_midline(
    *, coronal_path_yx: np.ndarray, coronal_t: np.ndarray, k_target: float, prev_t: float | None = None
) -> float:
    path = np.asarray(coronal_path_yx, dtype=np.float64)
    t = np.asarray(coronal_t, dtype=np.float64).reshape(-1)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] != t.shape[0] or t.size < 2:
        raise ValueError(f"Invalid coronal path/t shapes: path={path.shape}, t={t.shape}")
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t)
        t = t[order]
        path = path[order]

    k = path[:, 1]
    d = k - float(k_target)
    finite = np.isfinite(d) & np.isfinite(t)
    if int(np.count_nonzero(finite)) < 2:
        return float("nan")
    t = t[finite]
    k = k[finite]
    d = d[finite]

    hit = np.where(np.isclose(d, 0.0, atol=1.0e-9))[0]
    if hit.size:
        return float(t[int(hit[0])])

    sign = np.sign(d)
    flips = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if flips.size:
        candidates: list[float] = []
        for idx in flips.tolist():
            k0 = float(k[int(idx)])
            k1 = float(k[int(idx) + 1])
            if abs(k1 - k0) <= 1.0e-12:
                candidates.append(float(t[int(idx)]))
                continue
            frac = (float(k_target) - k0) / (k1 - k0)
            candidates.append(float(t[int(idx)] + frac * (t[int(idx) + 1] - t[int(idx)])))
        if prev_t is None:
            return float(candidates[0])
        prev = float(prev_t)
        return float(candidates[int(np.argmin(np.abs(np.asarray(candidates, dtype=np.float64) - prev)))])

    idx = int(np.argmin(np.abs(d)))
    return float(t[idx])


def _build_sagittal_ml_spline_from_coronal_k(
    *,
    slice_k: int,
    coronal_paths: dict[int, tuple[np.ndarray, np.ndarray]],
    ap_slice_keys: np.ndarray,
    ap_um_vals: np.ndarray,
    t0_by_slice_i: dict[int, float],
    len_um_by_slice_i: dict[int, float],
) -> CubicSpline:
    keys = np.asarray(ap_slice_keys, dtype=np.int32).reshape(-1)
    ap = np.asarray(ap_um_vals, dtype=np.float64).reshape(-1)
    if keys.size != ap.size or keys.size < 4:
        raise ValueError("Need >=4 AP axis points to build a sagittal ML spline.")

    ap_out: list[float] = []
    ml_out: list[float] = []
    prev_t: float | None = 0.5
    for slice_i, ap_um in zip(keys.tolist(), ap.tolist(), strict=True):
        row = coronal_paths.get(int(slice_i))
        if row is None:
            continue
        t0 = t0_by_slice_i.get(int(slice_i))
        length = len_um_by_slice_i.get(int(slice_i))
        if t0 is None or length is None:
            continue
        path_yx, t_vals = row
        t_k = _t_at_k_on_coronal_midline(
            coronal_path_yx=path_yx, coronal_t=t_vals, k_target=float(slice_k), prev_t=prev_t
        )
        if not np.isfinite(t_k):
            continue
        prev_t = float(t_k)
        ap_out.append(float(ap_um))
        ml_out.append((float(t_k) - float(t0)) * float(length))

    ap_arr = np.asarray(ap_out, dtype=np.float64)
    ml_arr = np.asarray(ml_out, dtype=np.float64)
    keep = np.isfinite(ap_arr) & np.isfinite(ml_arr)
    ap_arr = ap_arr[keep]
    ml_arr = ml_arr[keep]
    if ap_arr.size < 4:
        raise ValueError(f"Too few valid coronal intersections for slice_k={slice_k} (n={ap_arr.size}).")
    order = np.argsort(ap_arr)
    ap_arr = ap_arr[order]
    ml_arr = ml_arr[order]
    if not np.all(np.diff(ap_arr) > 0.0):
        uniq: list[float] = []
        vals: list[float] = []
        start = 0
        while start < ap_arr.size:
            end = start + 1
            while end < ap_arr.size and abs(float(ap_arr[end]) - float(ap_arr[start])) <= 1.0e-9:
                end += 1
            uniq.append(float(ap_arr[start]))
            vals.append(float(np.mean(ml_arr[start:end])))
            start = end
        ap_arr = np.asarray(uniq, dtype=np.float64)
        ml_arr = np.asarray(vals, dtype=np.float64)
    if ap_arr.size < 4:
        raise ValueError(f"Not enough unique AP points for slice_k={slice_k} after de-dup (n={ap_arr.size}).")
    return CubicSpline(ap_arr, ml_arr, bc_type="natural", extrapolate=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lut-outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
    )
    parser.add_argument("--ref-slice-i", type=int, default=236)
    parser.add_argument("--ref-t", type=float, default=0.5)
    parser.add_argument("--flat-n-t", type=int, default=257)
    parser.add_argument("--dtw-band-frac", type=float, default=0.15)
    parser.add_argument("--k-min", type=int, default=161)
    parser.add_argument("--k-max", type=int, default=235)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--n-sample", type=int, default=401)
    parser.add_argument(
        "--res-jk-um",
        type=float,
        nargs=2,
        default=(20.0, 20.0),
        metavar=("RES_J_UM", "RES_K_UM"),
        help="Fallback voxel resolution (um) for (j,k) when resolution_ds_ijk_um.npy is missing.",
    )
    parser.add_argument(
        "--ml-mode",
        type=str,
        default="s2c_t",
        choices=("s2c_t", "coronal_k_spline"),
        help="How to compute ML for sagittal planes in flattened chart.",
    )
    parser.add_argument("--out", type=Path, default=Path("ccf/refextract/_out/sagittal_planes_t_flattened.png"))
    args = parser.parse_args()

    outdir = Path(args.lut_outdir)
    ap_npz = outdir / "ap_axis_um_from_strips.npz"
    s2c_npz = outdir / "chart_map_sagittal_to_coronal_t2d.npz"
    cor_csv = outdir / "coronal_midline_columns.csv"
    res_path = outdir / "resolution_ds_ijk_um.npy"
    if not ap_npz.exists():
        raise FileNotFoundError(f"Missing: {ap_npz}")
    if not s2c_npz.exists():
        raise FileNotFoundError(f"Missing: {s2c_npz}")
    if not cor_csv.exists():
        raise FileNotFoundError(f"Missing: {cor_csv}")
    if res_path.exists():
        res = np.load(res_path).astype(np.float64, copy=False).reshape(-1)
        if res.shape != (3,):
            raise ValueError(f"Invalid resolution shape: {res.shape}")
        res_j_um = float(res[1])
        res_k_um = float(res[2])
    else:
        res_j_um = float(args.res_jk_um[0])
        res_k_um = float(args.res_jk_um[1])

    ap = np.load(ap_npz)
    ap_slice_keys = np.asarray(ap["slice_keys"], dtype=np.int32).reshape(-1)
    ap_um_vals = np.asarray(ap["ap_um"], dtype=np.float64).reshape(-1)
    if ap_slice_keys.size != ap_um_vals.size:
        raise ValueError("slice_keys/ap_um mismatch in ap axis npz.")

    coronal_paths = _load_midline_paths_csv(cor_csv, slice_label="slice_i")
    t0_by_slice_i, len_um_by_slice_i = _build_ml_anchor_for_coronal_slices(
        coronal_paths=coronal_paths,
        slice_keys=ap_slice_keys,
        ref_slice_i=int(args.ref_slice_i),
        ref_t=float(args.ref_t),
        n_t=int(args.flat_n_t),
        band_frac=float(args.dtw_band_frac),
        res_j_um=float(res_j_um),
        res_k_um=float(res_k_um),
    )
    anchor_keys_i = np.asarray(sorted(t0_by_slice_i.keys()), dtype=np.float64)
    anchor_t0 = np.asarray([t0_by_slice_i[int(k)] for k in anchor_keys_i.tolist()], dtype=np.float64)
    anchor_len = np.asarray([len_um_by_slice_i[int(k)] for k in anchor_keys_i.tolist()], dtype=np.float64)

    s2c = _load_s2c_t2d(s2c_npz)
    lut_keys_k = np.asarray(s2c.source_slice_keys, dtype=np.int32).reshape(-1)

    want_ks = list(range(int(args.k_min), int(args.k_max) + 1, int(args.k_step)))
    use_ks: list[int] = []
    for k in want_ks:
        kk = int(lut_keys_k[int(np.argmin(np.abs(lut_keys_k - int(k))))])
        use_ks.append(kk)
    use_ks = sorted(set(use_ks))

    t_s = np.linspace(0.0, 1.0, int(args.n_sample), dtype=np.float64)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=170)
    norm = plt.Normalize(0.0, 1.0)
    splines: dict[int, CubicSpline] = {}

    for k in use_ks:
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(s2c=s2c, slice_k=int(k), t_s=t_s)
        ap_um = _interp_with_linear_extrapolation(
            x=ap_slice_keys.astype(np.float64), y=ap_um_vals, xq=cor_slice_f
        ).astype(np.float64, copy=False)

        if str(args.ml_mode).strip().lower() == "coronal_k_spline":
            spline = splines.get(int(k))
            if spline is None:
                spline = _build_sagittal_ml_spline_from_coronal_k(
                    slice_k=int(k),
                    coronal_paths=coronal_paths,
                    ap_slice_keys=ap_slice_keys,
                    ap_um_vals=ap_um_vals,
                    t0_by_slice_i=t0_by_slice_i,
                    len_um_by_slice_i=len_um_by_slice_i,
                )
                splines[int(k)] = spline
            ml_um = spline(ap_um).astype(np.float64, copy=False)
        else:
            t0 = _interp_with_linear_extrapolation(x=anchor_keys_i, y=anchor_t0, xq=cor_slice_f)
            length = _interp_with_linear_extrapolation(x=anchor_keys_i, y=anchor_len, xq=cor_slice_f)
            ml_um = (cor_t - t0) * length

        keep = np.isfinite(ml_um) & np.isfinite(ap_um) & np.isfinite(cor_t)
        ml_um = ml_um[keep]
        ap_um = ap_um[keep]
        t_keep = cor_t[keep]
        ax.scatter(
            ml_um,
            ap_um,
            s=6.0,
            c=cmap(norm(t_keep)),
            alpha=0.65,
            linewidths=0.0,
            zorder=2,
        )
        if ml_um.size:
            mid = int(ml_um.size // 2)
            ax.text(
                float(ml_um[mid]),
                float(ap_um[mid]),
                f"k={int(k)}",
                fontsize=7,
                alpha=0.85,
                ha="left",
                va="center",
                zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.6},
            )

    ax.set_xlabel("ML (um; anchored arclength along coronal t)")
    ax.set_ylabel("AP (um; optimized from strip matching)")

    if not np.all(np.diff(ap_um_vals) >= 0.0):
        raise ValueError("ap_um axis must be monotone increasing to create a slice_i secondary axis.")

    def _ap_um_to_slice_i(y: np.ndarray) -> np.ndarray | float:
        yy = np.asarray(y, dtype=np.float64)
        scalar = yy.ndim == 0
        out = _interp_with_linear_extrapolation(
            x=ap_um_vals.astype(np.float64, copy=False),
            y=ap_slice_keys.astype(np.float64, copy=False),
            xq=yy,
        )
        out = out.reshape(yy.shape)
        if scalar:
            return float(out)
        return out

    def _slice_i_to_ap_um(s: np.ndarray) -> np.ndarray | float:
        ss = np.asarray(s, dtype=np.float64)
        scalar = ss.ndim == 0
        out = _interp_with_linear_extrapolation(
            x=ap_slice_keys.astype(np.float64, copy=False),
            y=ap_um_vals.astype(np.float64, copy=False),
            xq=ss,
        )
        out = out.reshape(ss.shape)
        if scalar:
            return float(out)
        return out

    secax = ax.secondary_yaxis("right", functions=(_ap_um_to_slice_i, _slice_i_to_ap_um))
    secax.set_ylabel("coronal slice_i")
    secax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    ax.set_title(
        f"Sagittal planes in flattened chart (every {int(args.k_step)} in k={int(args.k_min)}..{int(args.k_max)}; ml_mode={args.ml_mode})"
    )
    ax.grid(True, linewidth=0.5, alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # Keep extra space on the right because we also add a secondary y-axis there.
    cbar = fig.colorbar(sm, ax=ax, shrink=0.9, pad=0.12)
    cbar.set_label("t_all (coronal t; 0..1)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=240)
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
