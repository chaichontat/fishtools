# %% [markdown]
# # Principal-Curve h5ad -> CCF ijk (LUT) + 3D plot
#
# This workflow converts two `.syn.annotated.princurve.h5ad` files into CCF `ijk` coordinates
# using midsurface LUTs and plots both datasets in one 3D scatter.
#
# Run cells sequentially. Artifacts are written under `OUTDIR`.

# %% [markdown]
# ## Configuration

# %%
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours

# %%
# === EDIT THESE ===

INPUT_H5ADS: list[Path] = [
    Path(
        "/home/chaichontat/fishtools2/working/20250929_JaxA3_Coro4/analysis/output/ccf-transforms/2/2.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20250929_JaxA3_Coro4/analysis/output/ccf-transforms/3/3.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/1/1.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/2/2.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/3/3.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/4/4.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/5/5.syn.annotated.princurve.h5ad"
    ),
    Path(
        "/home/chaichontat/fishtools2/working/20251024_JaxA1_Sag7/analysis/output/ccf-transforms/6/6.syn.annotated.princurve.h5ad"
    ),
]

LUT_OUTDIR = Path(
    "/home/chaichontat/fishtools2/ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"
)
LUT_NT = 1024

R_BIN_COUNT = 1024
R_ROLL_HALF_WINDOW = 24
R_SOURCE_UM_PER_PX = 0.216
R_TARGET_UM_PER_PX = 20.0

T_LOOKUP_MODE = "t_all"  # {"t_all", "t_local_to_t_all"}

MAX_PLOT_POINTS_PER_DATASET = 80_000
PLOT_ALPHA = 0.08
PLOT_SIZE = 1.0
SEED = 0

OUTDIR = Path("/home/chaichontat/fishtools2/results/refextract/princurve_h5ad_ijk_plot")
OUTDIR.mkdir(parents=True, exist_ok=True)
T_TICKS = tuple(float(v) for v in np.linspace(0.0, 1.0, 21))

# Flattened AP/ML plotting (for statistical modeling charts)
PLOT_FLATTENED_AP_ML = True
AP_AXIS_NPZ = LUT_OUTDIR / "ap_axis_um_from_strips.npz"
S2C_T2D_NPZ = LUT_OUTDIR / "chart_map_sagittal_to_coronal_t2d.npz"
FLAT_REF_SLICE_I = 236
FLAT_REF_T = 0.5
FLAT_N_T = 257
FLAT_DTW_BAND_FRAC = 0.15
# How to compute ML for sagittal slices in flattened chart:
# - "s2c_t": use sagittal->coronal (slice_i_float, t_cor) directly, then ml_um=(t_cor-t0)*len.
# - "coronal_k_spline": compute per-sagittal-plane t_cor(i) by intersecting k with coronal midlines,
#   convert to ml_um(i), then use a cubic spline ml(ap_um) to evaluate at each point's AP.
FLAT_SAGITTAL_ML_MODE = "s2c_t"  # {"s2c_t", "coronal_k_spline"}

print(f"LUT_OUTDIR={LUT_OUTDIR}")
print(f"OUTDIR={OUTDIR}")
print(f"LUT_NT={LUT_NT}, R_BIN_COUNT={R_BIN_COUNT}, R_ROLL_HALF_WINDOW={R_ROLL_HALF_WINDOW}")
print(f"T_LOOKUP_MODE={T_LOOKUP_MODE}")
print(f"FLAT_SAGITTAL_ML_MODE={FLAT_SAGITTAL_ML_MODE}")
print(
    f"R_SCALE_PX={R_SOURCE_UM_PER_PX / R_TARGET_UM_PER_PX:.8f} ({R_SOURCE_UM_PER_PX}um/px -> {R_TARGET_UM_PER_PX}um/px)"
)

# %% [markdown]
# ## Phase 0: Helpers (LUT load, rolling normalization, vectorized lookup)

# %%
AXIS = str


@dataclass(frozen=True)
class AxisLut:
    axis: AXIS
    source_slice_keys: np.ndarray
    t_grid: np.ndarray
    low_ijk: np.ndarray
    high_ijk: np.ndarray


@dataclass(frozen=True)
class TFields:
    """Principal-curve `t` variants for one sample, all slice-local.

    For each row:
    - `t_all`: per-slice full arc-length coordinate in atlas slice space.
    - `t_local`: sample coordinate on the same slice-local axis; expected to be a strict subset of `t_all`.

    There is no global cross-slice reference-line coordinate in this workflow.
    """

    t_all: np.ndarray
    t_local: np.ndarray


@dataclass(frozen=True)
class TTypeConverter:
    """Convert between local `t` parameterizations on the same slice-local axis.

    This converter does not define any cross-slice/global mapping.
    """

    local_to_all_affine: tuple[float, float] | None = None  # t_all = slope * t_local + intercept

    @staticmethod
    def fit_local_to_all_affine(*, t_local: np.ndarray, t_all: np.ndarray) -> tuple[float, float]:
        """Fit affine map from `t_local` to slice-local `t_all`: `t_all = slope*t_local + intercept`."""
        x = np.asarray(t_local, dtype=np.float64).reshape(-1)
        y = np.asarray(t_all, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"Cannot fit t_local->t_all affine: shape mismatch {x.shape} vs {y.shape}.")
        keep = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(keep)) < 2:
            raise ValueError("Cannot fit t_local->t_all affine: fewer than 2 finite pairs.")
        xk = x[keep]
        yk = y[keep]
        x_center = xk - float(np.mean(xk))
        var_x = float(np.dot(x_center, x_center))
        if var_x <= 1.0e-12:
            raise ValueError("Cannot fit t_local->t_all affine: degenerate t_local variance.")
        slope = float(np.dot(x_center, yk - float(np.mean(yk))) / var_x)
        intercept = float(np.mean(yk) - slope * np.mean(xk))
        return slope, intercept

    def t_local_to_t_all(self, t_local: np.ndarray) -> np.ndarray:
        """Map sample-local `t_local` to slice-local `t_all` using fitted affine coefficients."""
        affine = self.local_to_all_affine
        if affine is None:
            raise ValueError("Missing local_to_all_affine; cannot convert t_local to t_all.")
        slope = float(affine[0])
        intercept = float(affine[1])
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise ValueError(f"Invalid local_to_all_affine={affine!r}.")
        t = np.asarray(t_local, dtype=np.float64)
        return (slope * t + intercept).astype(np.float64, copy=False)


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    idx = np.arange(y.size, dtype=np.float64)
    ok = np.isfinite(y)
    if not np.any(ok):
        raise ValueError("Cannot fill NaNs: array has no finite values.")
    if np.count_nonzero(ok) == 1:
        y[:] = y[ok][0]
        return y
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


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
    """Resample (y,x) polyline to n points using physical arc length in um.

    Returns (y_new, x_new, total_len_um), coordinates in vox.
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
        raise ValueError("Invalid polyline arc length")
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


def _load_ap_axis_um(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing AP axis file: {npz_path}")
    d = np.load(npz_path)
    slice_keys = np.asarray(d["slice_keys"], dtype=np.int32).reshape(-1)
    ap_um = np.asarray(d["ap_um"], dtype=np.float64).reshape(-1)
    if slice_keys.size != ap_um.size:
        raise ValueError(f"Invalid ap axis arrays in {npz_path}: {slice_keys.shape} vs {ap_um.shape}")
    return slice_keys, ap_um


def _ap_um_for_slice(slice_keys: np.ndarray, ap_um: np.ndarray, slice_i: np.ndarray) -> np.ndarray:
    keys = np.asarray(slice_keys, dtype=np.int32).reshape(-1)
    vals = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    s = np.asarray(slice_i, dtype=np.int32).reshape(-1)
    idx = np.argmin(np.abs(keys[None, :].astype(np.int32) - s[:, None]), axis=1)
    out = vals[idx]
    return out.astype(np.float64, copy=False)


def _ap_um_for_slice_float(slice_keys: np.ndarray, ap_um: np.ndarray, slice_i_float: np.ndarray) -> np.ndarray:
    keys = np.asarray(slice_keys, dtype=np.float64).reshape(-1)
    vals = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    s = np.asarray(slice_i_float, dtype=np.float64).reshape(-1)
    if keys.size != vals.size:
        raise ValueError(f"slice_keys/ap_um size mismatch: {keys.size} vs {vals.size}")
    if keys.size < 2:
        raise ValueError("Need at least 2 AP axis points for interpolation.")
    if not np.all(np.diff(keys) > 0.0):
        order = np.argsort(keys)
        keys = keys[order]
        vals = vals[order]
    s_clip = np.clip(s, float(keys[0]), float(keys[-1]))
    return _interp_with_linear_extrapolation(x=keys, y=vals, xq=s_clip).astype(np.float64, copy=False)


@dataclass(frozen=True)
class S2CLut:
    source_slice_keys: np.ndarray
    t_grid: np.ndarray
    target_slice_idx: np.ndarray
    target_t: np.ndarray


def _load_s2c_t2d(npz_path: Path) -> S2CLut:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing sagittal->coronal chart LUT: {npz_path}. "
            "Build with midsurface_coords.py (it writes chart_map_*_t2d.npz)."
        )
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
    """Return a coronal t where the midline's k crosses k_target.

    This uses segment crossings in the polyline order (increasing t). If multiple crossings
    exist, chooses the one closest to prev_t when provided; otherwise chooses the earliest
    crossing in t. If no crossing exists, falls back to the nearest point (in k).
    """
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
        # Collapse duplicates by averaging.
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
    """Return (slice_i->t0, slice_i->len_um) for ML anchoring in coronal domain."""
    keys = [int(k) for k in np.asarray(slice_keys, dtype=np.int32).reshape(-1).tolist()]
    if int(ref_slice_i) not in coronal_paths:
        # Snap to nearest slice that has a curve.
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
            raise ValueError(f"DTW failed for ML anchoring slice_i={s}; increase FLAT_DTW_BAND_FRAC.")
        idx_map = _path_to_monotone_index_map(path, n=int(n_t))  # current idx -> ref idx
        origin_idx_f = _invert_monotone_index_map_to_fractional_idx(idx_map=idx_map, ref_idx=int(ref_idx))
        t0_by_slice[int(s)] = float(origin_idx_f / float(n_t - 1))
        len_by_slice[int(s)] = float(total_len_um)
    return t0_by_slice, len_by_slice


def _rolling_nan_reduce(x: np.ndarray, half_window: int, *, mode: str) -> np.ndarray:
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported mode: {mode}")
    y = np.asarray(x, dtype=np.float64)
    n = int(y.size)
    out = np.full((n,), np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - int(half_window))
        hi = min(n, i + int(half_window) + 1)
        w = y[lo:hi]
        ok = np.isfinite(w)
        if not np.any(ok):
            continue
        out[i] = float(np.nanmin(w)) if mode == "min" else float(np.nanmax(w))
    return _fill_nan_1d(out)


def _finite_minmax(x: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan"), float("nan")
    vals = arr[finite]
    return float(np.min(vals)), float(np.max(vals))


def _interp_with_linear_extrapolation(*, x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x_in = np.asarray(x, dtype=np.float64).reshape(-1)
    y_in = np.asarray(y, dtype=np.float64).reshape(-1)
    q = np.asarray(xq, dtype=np.float64).reshape(-1)
    if x_in.size != y_in.size:
        raise ValueError(f"x/y size mismatch: {x_in.size} vs {y_in.size}.")
    if x_in.size == 0:
        raise ValueError("Cannot interpolate with empty support.")
    if x_in.size == 1:
        return np.full(q.shape, float(y_in[0]), dtype=np.float64)

    out = np.interp(q, x_in, y_in).astype(np.float64, copy=False)
    left = q < float(x_in[0])
    right = q > float(x_in[-1])

    dx_lo = float(x_in[1] - x_in[0])
    slope_lo = 0.0 if abs(dx_lo) <= 1.0e-12 else float((y_in[1] - y_in[0]) / dx_lo)
    dx_hi = float(x_in[-1] - x_in[-2])
    slope_hi = 0.0 if abs(dx_hi) <= 1.0e-12 else float((y_in[-1] - y_in[-2]) / dx_hi)

    if np.any(left):
        out[left] = float(y_in[0]) + (q[left] - float(x_in[0])) * slope_lo
    if np.any(right):
        out[right] = float(y_in[-1]) + (q[right] - float(x_in[-1])) * slope_hi
    return out


def _load_axis_lut(outdir: Path, axis: AXIS) -> AxisLut:
    """Load IJK LUT for one atlas axis.

    The LUT `t_domain` must be `t_all`, meaning per-slice full-path arc length on that axis.
    """
    if axis == "coronal":
        path = outdir / "chart_map_coronal_ijk_from_tr.npz"
    elif axis == "sagittal":
        path = outdir / "chart_map_sagittal_ijk_from_tr.npz"
    else:
        raise ValueError(f"axis must be 'coronal' or 'sagittal', got {axis!r}")

    d = np.load(path)
    t_grid = np.asarray(d["t_grid"], dtype=np.float64)
    if t_grid.size != int(LUT_NT):
        raise ValueError(f"{path.name} has t_grid size {t_grid.size}, expected LUT_NT={LUT_NT}.")
    try:
        t_domain = str(np.asarray(d["t_domain"]).reshape(-1)[0])
    except (KeyError, ValueError, IndexError):
        t_domain = ""
    if t_domain != "t_all":
        raise ValueError(
            f"{path.name} has unsupported t_domain={t_domain!r}; expected 't_all'. "
            "Rebuild IJK LUTs with midsurface_coords.py --rebuild-ijk-lut."
        )
    return AxisLut(
        axis=axis,
        source_slice_keys=np.asarray(d["source_slice_keys"], dtype=np.int32),
        t_grid=t_grid,
        low_ijk=np.asarray(d["low_ijk"], dtype=np.float64),
        high_ijk=np.asarray(d["high_ijk"], dtype=np.float64),
    )


def _compute_signed_r_um(
    *,
    t_lookup: np.ndarray,
    r_signed: np.ndarray,
    n_bins: int,
    roll_half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert signed principal r to microns using a rolling floor over selected t.

    The floor is the rolling local minimum along t. Radial offsets are converted
    from source px units to real microns.
    """

    t = np.asarray(t_lookup, dtype=np.float64)
    r = np.asarray(r_signed, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError("t_lookup and r_signed must have the same shape.")
    if t.ndim != 1:
        raise ValueError("t_lookup and r_signed must be 1D.")

    bin_idx = np.minimum((t * float(n_bins - 1)).astype(np.int32), int(n_bins - 1))
    r_bin_min = np.full((n_bins,), np.inf, dtype=np.float64)
    r_bin_max = np.full((n_bins,), -np.inf, dtype=np.float64)
    np.minimum.at(r_bin_min, bin_idx, r)
    np.maximum.at(r_bin_max, bin_idx, r)
    r_bin_min[~np.isfinite(r_bin_min)] = np.nan
    r_bin_max[~np.isfinite(r_bin_max)] = np.nan

    r_floor_bin = _rolling_nan_reduce(r_bin_min, half_window=roll_half_window, mode="min")
    r_ceil_bin = _rolling_nan_reduce(r_bin_max, half_window=roll_half_window, mode="max")
    t_bins = np.linspace(0.0, 1.0, num=int(n_bins), dtype=np.float64)

    r_floor = np.interp(t, t_bins, r_floor_bin)
    r_ceil = np.interp(t, t_bins, r_ceil_bin)
    if np.any(~np.isfinite(r_floor)) or np.any(~np.isfinite(r_ceil)):
        raise ValueError("Invalid rolling floor/ceiling while building r_um.")

    r_um = (r - r_floor) * float(R_SOURCE_UM_PER_PX)
    return (
        r_um.astype(np.float64, copy=False),
        r_floor.astype(np.float64, copy=False),
        r_ceil.astype(np.float64, copy=False),
    )


def _nearest_slice_lut_rows(
    *,
    lut: AxisLut,
    atlas_slice_idx: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest source slice and finite per-`t` low/high rows for that slice."""
    source_slice_keys = np.asarray(lut.source_slice_keys, dtype=np.int32)
    if source_slice_keys.size == 0:
        raise ValueError(f"LUT axis={lut.axis} has no source slices.")
    nearest_idx = int(np.argmin(np.abs(source_slice_keys.astype(np.float64) - float(atlas_slice_idx))))
    source_slice_used = int(source_slice_keys[nearest_idx])

    low_row = np.asarray(lut.low_ijk[nearest_idx], dtype=np.float64)
    high_row = np.asarray(lut.high_ijk[nearest_idx], dtype=np.float64)
    if low_row.ndim != 2 or high_row.ndim != 2 or low_row.shape != high_row.shape or low_row.shape[1] != 3:
        raise ValueError(f"Unexpected LUT row shapes: low={low_row.shape}, high={high_row.shape}.")

    valid = np.isfinite(low_row).all(axis=1) & np.isfinite(high_row).all(axis=1)
    if not np.any(valid):
        raise ValueError(f"LUT axis={lut.axis} slice={source_slice_used} has no finite t support.")
    return source_slice_used, np.asarray(lut.t_grid, dtype=np.float64)[valid], low_row[valid], high_row[valid]


def _lookup_ijk_batch_slice_locked(
    *,
    lut: AxisLut,
    atlas_slice_idx: int,
    t_lookup: np.ndarray,
    r_um: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Lookup on nearest source slice using real radial offsets in microns."""
    t = np.asarray(t_lookup, dtype=np.float64)
    r = np.asarray(r_um, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError("t_lookup and r_um must have the same shape.")
    if t.ndim != 1:
        raise ValueError("t_lookup and r_um must be 1D.")

    source_slice_used, t_valid, low_valid, high_valid = _nearest_slice_lut_rows(
        lut=lut,
        atlas_slice_idx=atlas_slice_idx,
    )
    low = np.empty((t.size, 3), dtype=np.float64)
    high = np.empty((t.size, 3), dtype=np.float64)
    for d in range(3):
        low[:, d] = _interp_with_linear_extrapolation(x=t_valid, y=low_valid[:, d], xq=t)
        high[:, d] = _interp_with_linear_extrapolation(x=t_valid, y=high_valid[:, d], xq=t)
    radial_vec = high - low
    radial_norm = np.linalg.norm(radial_vec, axis=1)
    if np.any(~np.isfinite(radial_norm)):
        raise ValueError("Non-finite radial vector norm during slice-locked IJK lookup.")
    if float(np.nanmin(radial_norm)) <= 1.0e-12:
        raise ValueError("Degenerate radial vector norm during slice-locked IJK lookup.")
    radial_unit = radial_vec / radial_norm[:, None]
    r_px_target = r / float(R_TARGET_UM_PER_PX)
    ijk = low + r_px_target[:, None] * radial_unit
    if not np.isfinite(ijk).all():
        raise ValueError("Non-finite values produced during slice-locked IJK lookup.")
    return ijk.astype(np.float64, copy=False), source_slice_used


def _slice_t_support(lut: AxisLut, atlas_slice_idx: int) -> tuple[float, float, int]:
    """Return finite local-`t_all` support for the source slice nearest `atlas_slice_idx`."""
    source_slice_used, t_valid, _low_valid, _high_valid = _nearest_slice_lut_rows(
        lut=lut,
        atlas_slice_idx=atlas_slice_idx,
    )
    return float(np.min(t_valid)), float(np.max(t_valid)), source_slice_used


def _p1_landmarks_path_from_h5ad(h5ad_path: Path) -> Path:
    return h5ad_path.with_name("p1_landmarks.json")


def _read_axis_and_slice(h5ad_path: Path) -> tuple[AXIS, int]:
    p1_path = _p1_landmarks_path_from_h5ad(h5ad_path)
    p1 = json.loads(p1_path.read_text())
    axis = str(p1["atlas_plane"]).lower()
    if axis not in {"coronal", "sagittal"}:
        raise ValueError(f"{p1_path} has unsupported atlas_plane={axis!r}.")
    atlas_slice_idx = int(p1["atlas_slice_idx"])
    return axis, atlas_slice_idx


def _resolve_t_fields(adata: ad.AnnData) -> TFields:
    """Resolve `t_all`/`t_local` from `.h5ad` as slice-local coordinates.

    `t_local` is taken from `obs['t_local']` when present (else from `obsm['principal'][:,0]`).
    `t_all` is taken from `obs['t_all']` when present, else is derived from `uns['t_all_mapping']`
    when available (LUT lookup domain).
    All returned coordinates are local to the current slice; no global `t` is defined.
    """

    def _t_all_from_uns_mapping(adata_in: ad.AnnData, *, t_local_in: np.ndarray) -> np.ndarray | None:
        mapping = adata_in.uns.get("t_all_mapping") if isinstance(adata_in.uns, dict) else None
        if not isinstance(mapping, dict):
            return None

        def _span_from_pair_key(key: str) -> tuple[float, float] | None:
            raw = mapping.get(key)
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                return None
            lo = float(raw[0])
            hi = float(raw[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1.0e-12:
                return None
            return lo, hi

        t_all_span: tuple[float, float] | None = None
        for key in ("t_all_span", "t_all_range", "t_all_domain"):
            t_all_span = _span_from_pair_key(key)
            if t_all_span is not None:
                break
        if t_all_span is None:
            if "t_all_min" in mapping and "t_all_max" in mapping:
                lo = float(mapping["t_all_min"])
                hi = float(mapping["t_all_max"])
                if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1.0e-12:
                    t_all_span = (lo, hi)
        if t_all_span is None:
            if "begin_t" in mapping and "end_t" in mapping:
                lo = float(mapping["begin_t"])
                hi = float(mapping["end_t"])
                if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1.0e-12:
                    t_all_span = (lo, hi)
            elif "begin" in mapping and "end" in mapping:
                lo = float(mapping["begin"])
                hi = float(mapping["end"])
                if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1.0e-12:
                    t_all_span = (lo, hi)
        if t_all_span is None:
            return None

        t_local_span: tuple[float, float] | None = None
        for key in ("t_local_span", "t_local_range", "t_local_domain"):
            t_local_span = _span_from_pair_key(key)
            if t_local_span is not None:
                break
        if t_local_span is None and ("t_local_min" in mapping and "t_local_max" in mapping):
            lo = float(mapping["t_local_min"])
            hi = float(mapping["t_local_max"])
            if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 1.0e-12:
                t_local_span = (lo, hi)
        if t_local_span is None:
            t0, t1 = _finite_minmax(t_local_in)
            if not (np.isfinite(t0) and np.isfinite(t1) and (t1 - t0) > 1.0e-12):
                return None
            t_local_span = (t0, t1)

        t0_local, t1_local = float(t_local_span[0]), float(t_local_span[1])
        t0_all, t1_all = float(t_all_span[0]), float(t_all_span[1])
        slope = (t1_all - t0_all) / (t1_local - t0_local)
        intercept = t0_all - slope * t0_local
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None
        t_local_arr = np.asarray(t_local_in, dtype=np.float64)
        return (slope * t_local_arr + intercept).astype(np.float64, copy=False)

    if "principal" not in adata.obsm:
        raise ValueError("Missing obsm['principal']; cannot derive t_local/t_all.")
    principal = np.asarray(adata.obsm["principal"], dtype=np.float64)
    if principal.ndim != 2 or principal.shape[1] < 1:
        raise ValueError(f"obsm['principal'] has invalid shape {principal.shape}; expected (n, >=1).")

    if "t_local" in adata.obs.columns:
        t_local = np.asarray(adata.obs["t_local"].to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    else:
        t_local = np.asarray(principal[:, 0], dtype=np.float64)
    if "t_all" in adata.obs.columns:
        t_all = np.asarray(adata.obs["t_all"].to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    else:
        t_all = _t_all_from_uns_mapping(adata, t_local_in=t_local)
        if t_all is None:
            keys = sorted(adata.uns.keys()) if isinstance(adata.uns, dict) else []
            raise ValueError(
                f"Missing obs['t_all'] and could not derive t_all from uns['t_all_mapping']. uns_keys={keys}"
            )

    if t_all.shape[0] != adata.n_obs or t_local.shape[0] != adata.n_obs:
        raise ValueError("Resolved t fields are inconsistent with adata.n_obs.")
    return TFields(t_all=t_all, t_local=t_local)


def _resolve_t_lookup_for_lut(t_fields: TFields) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve lookup arrays for LUT sampling in local `t_all` domain.

    Returns:
    - `t_lookup_raw`: source values before conversion.
    - `t_lookup_lut`: values in LUT `t_all` domain.
    - `t_lookup_name`: source coordinate label for reporting.
    """
    mode = str(T_LOOKUP_MODE).strip().lower()
    if mode == "t_all":
        return t_fields.t_all, t_fields.t_all, "t_all"
    if mode == "t_local_to_t_all":
        local_to_all_affine = TTypeConverter.fit_local_to_all_affine(
            t_local=t_fields.t_local,
            t_all=t_fields.t_all,
        )
        converter = TTypeConverter(local_to_all_affine=local_to_all_affine)
        return t_fields.t_local, converter.t_local_to_t_all(t_fields.t_local), "t_local"
    raise ValueError(
        f"Unsupported T_LOOKUP_MODE={T_LOOKUP_MODE!r}; expected one of {{'t_all', 't_local_to_t_all'}}."
    )


def _largest_binary_contour(mask_2d: np.ndarray) -> np.ndarray | None:
    """Return the largest contour from a 2D boolean mask in index coordinates."""
    m = np.asarray(mask_2d, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return None
    contours = find_contours(m.astype(np.float64, copy=False), level=0.5)
    if not contours:
        return None
    largest = max(contours, key=lambda c: float(c.shape[0]))
    return np.asarray(largest, dtype=np.float64)


def _load_midline_paths_csv(path: Path, *, slice_label: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-slice midline curves from `*midline_columns.csv`.

    IMPORTANT: the `t` column in these CSVs is the LUT `t` coordinate:
    - It is the same coordinate system used by `midsurface_coords.py` LUTs.
    - This is per-slice `t_all` (normalized full-path arc length on that section).

    The principal-curve `.h5ad` carries separate coordinates (`t_local`, `t_all`),
    and both are slice-local (there is no global reference-line coordinate in this workflow).
    For a given slice, principal `t_local` is expected to be a strict subset of LUT `t_all`.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out

    grouped: dict[int, list[tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if slice_label not in row or "t" not in row or "y" not in row or "x" not in row:
                continue
            raw_slice = row[slice_label]
            if raw_slice is None:
                continue
            slice_val = float(raw_slice)
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
        if (
            t_vals.ndim != 1
            or path_yx.ndim != 2
            or path_yx.shape[1] != 2
            or path_yx.shape[0] != t_vals.shape[0]
        ):
            continue
        out[int(slice_idx)] = (path_yx, t_vals)
    return out


def _curve_ijk_from_midline_csv(
    *,
    axis: AXIS,
    atlas_slice_idx: int,
    coronal_paths: dict[int, tuple[np.ndarray, np.ndarray]],
    sagittal_paths: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    if axis == "coronal":
        row = coronal_paths.get(int(atlas_slice_idx))
        if row is None:
            return None
        path_yx, t_vals = row
        # coronal columns: y=j, x=k at fixed i.
        j = path_yx[:, 0]
        k = path_yx[:, 1]
        i = np.full(j.shape, float(atlas_slice_idx), dtype=np.float64)
        return np.column_stack([i, j, k]), t_vals
    if axis == "sagittal":
        row = sagittal_paths.get(int(atlas_slice_idx))
        if row is None:
            return None
        path_yx, t_vals = row
        # sagittal columns: y=i, x=j at fixed k.
        i = path_yx[:, 0]
        j = path_yx[:, 1]
        k = np.full(i.shape, float(atlas_slice_idx), dtype=np.float64)
        return np.column_stack([i, j, k]), t_vals
    raise ValueError(f"Unsupported axis={axis!r}")


def _point_on_curve_ijk_by_t(curve_ijk: np.ndarray, t_all: np.ndarray, t_query: float) -> np.ndarray | None:
    """Interpolate one point on a curve using local per-slice `t_all`."""
    curve = np.asarray(curve_ijk, dtype=np.float64)
    t_vals = np.asarray(t_all, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or t_vals.ndim != 1 or t_vals.shape[0] != curve.shape[0]:
        return None
    finite = np.isfinite(t_vals) & np.isfinite(curve).all(axis=1)
    if int(np.count_nonzero(finite)) < 2:
        return None
    t_f = t_vals[finite]
    c_f = curve[finite]
    order = np.argsort(t_f)
    t_f = t_f[order]
    c_f = c_f[order]
    t_unique, uniq_idx = np.unique(t_f, return_index=True)
    if t_unique.size < 2:
        return None
    c_unique = c_f[uniq_idx]
    tq = float(np.clip(float(t_query), float(t_unique[0]), float(t_unique[-1])))
    i = float(np.interp(tq, t_unique, c_unique[:, 0]))
    j = float(np.interp(tq, t_unique, c_unique[:, 1]))
    k = float(np.interp(tq, t_unique, c_unique[:, 2]))
    if not (np.isfinite(i) and np.isfinite(j) and np.isfinite(k)):
        return None
    return np.asarray([i, j, k], dtype=np.float64)


def _plot_t_ticks_with_labels(
    *,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    t_vals: np.ndarray,
    t_ticks: tuple[float, ...],
    color: str,
    label_prefix: str,
    xspan: float,
    yspan: float,
    show_labels: bool = True,
    label_values: tuple[float, ...] | None = None,
) -> None:
    """Draw normalized ticks/labels on a 2D curve projection.

    `t_ticks` are interpreted as fractions in [0, 1] of the provided curve's
    local `t_vals` span, so 0/1 always map to the curve endpoints.
    """
    tick_len = 0.015 * float(max(xspan, yspan))
    label_off = 0.012 * float(max(xspan, yspan))
    if label_values is not None and len(label_values) != len(t_ticks):
        raise ValueError("label_values must be the same length as t_ticks when provided.")
    t_arr = np.asarray(t_vals, dtype=np.float64).reshape(-1)
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if t_arr.size == 0 or x_arr.size != t_arr.size or y_arr.size != t_arr.size:
        return
    finite = np.isfinite(t_arr) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        return
    t_f = t_arr[finite]
    x_f = x_arr[finite]
    y_f = y_arr[finite]
    order = np.argsort(t_f)
    t_f = t_f[order]
    x_f = x_f[order]
    y_f = y_f[order]
    t_unique, uniq_idx = np.unique(t_f, return_index=True)
    if t_unique.size < 2:
        return
    x_unique = x_f[uniq_idx]
    y_unique = y_f[uniq_idx]
    t_min = float(t_unique[0])
    t_max = float(t_unique[-1])
    center = np.asarray([float(np.mean(x_unique)), float(np.mean(y_unique))], dtype=np.float64)
    dt = float(np.median(np.diff(t_unique)))
    dt = float(np.clip(2.0 * dt, 1.0e-3, 0.05))
    for tick_idx, t_tick in enumerate(t_ticks):
        t_frac = float(t_tick)
        if t_frac < 0.0 or t_frac > 1.0:
            continue
        tq = float(t_min + t_frac * (t_max - t_min))
        px = float(np.interp(tq, t_unique, x_unique))
        py = float(np.interp(tq, t_unique, y_unique))
        p = np.asarray([px, py], dtype=np.float64)

        tq0 = float(max(t_min, tq - dt))
        tq1 = float(min(t_max, tq + dt))
        x0 = float(np.interp(tq0, t_unique, x_unique))
        y0 = float(np.interp(tq0, t_unique, y_unique))
        x1 = float(np.interp(tq1, t_unique, x_unique))
        y1 = float(np.interp(tq1, t_unique, y_unique))
        d = np.asarray([x1 - x0, y1 - y0], dtype=np.float64)
        norm = float(np.hypot(d[0], d[1]))
        if norm <= 1.0e-9:
            d = np.asarray([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        n = np.asarray([-d[1], d[0]], dtype=np.float64) / norm
        if float(np.dot(n, p - center)) < 0.0:
            n = -n
        p0 = p - 0.5 * tick_len * n
        p1 = p + 0.5 * tick_len * n
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=1.1, alpha=0.95, zorder=8)
        if show_labels:
            lab = float(label_values[tick_idx]) if label_values is not None else float(t_tick)
            txt = f"{label_prefix}{lab:.2f}"
            pt = p + (0.5 * tick_len + label_off) * n
            ax.text(
                pt[0],
                pt[1],
                txt,
                color=color,
                fontsize=7,
                alpha=0.95,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
                zorder=9,
            )


def _build_plot_payload(
    *,
    results_in: list[dict[str, object]],
    rng_in: np.random.Generator,
) -> list[tuple[dict[str, object], np.ndarray, np.ndarray, str, np.ndarray]]:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    payload: list[tuple[dict[str, object], np.ndarray, np.ndarray, str, np.ndarray]] = []
    for idx, info in enumerate(results_in):
        data = np.load(Path(str(info["out_npz"])))
        ijk = np.asarray(data["ijk"], dtype=np.float64)
        r_signed = np.asarray(data["r_signed"], dtype=np.float64)
        t_lookup = np.asarray(data["t_lookup"], dtype=np.float64)
        if r_signed.shape[0] != ijk.shape[0]:
            raise ValueError(
                f"Mismatched ijk/r_signed rows for {info['out_npz']}: {ijk.shape[0]} vs {r_signed.shape[0]}."
            )
        if t_lookup.shape[0] != ijk.shape[0]:
            raise ValueError(
                f"Mismatched ijk/t_lookup rows for {info['out_npz']}: {ijk.shape[0]} vs {t_lookup.shape[0]}."
            )
        if ijk.shape[0] > int(MAX_PLOT_POINTS_PER_DATASET):
            pick = rng_in.choice(ijk.shape[0], size=int(MAX_PLOT_POINTS_PER_DATASET), replace=False)
            ijk_plot = ijk[pick]
            r_signed_plot = r_signed[pick]
            t_lookup_plot = t_lookup[pick]
        else:
            ijk_plot = ijk
            r_signed_plot = r_signed
            t_lookup_plot = t_lookup
        payload.append((info, ijk_plot, r_signed_plot, colors[idx % len(colors)], t_lookup_plot))
    if not payload:
        raise ValueError("No plotting payload generated; results list is empty.")
    return payload


def _plot_cells_ijk_mpl3d(
    *,
    plot_payload: list[tuple[dict[str, object], np.ndarray, np.ndarray, str, np.ndarray]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(9.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    for info, ijk_plot, _r_signed_plot, color, _t_lookup_plot in plot_payload:
        label = (
            f"{info['name']} ({info['axis']}, slice={info['atlas_slice_idx']}, "
            f"used={info['source_slice_used']}, n={ijk_plot.shape[0]})"
        )
        ax.scatter(
            ijk_plot[:, 0],
            ijk_plot[:, 1],
            ijk_plot[:, 2],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            color=color,
            label=label,
            linewidths=0.0,
        )

    stack = np.vstack([pts for _, pts, _, _, _t in plot_payload])
    span = np.ptp(stack, axis=0)
    span = np.where(span <= 0.0, 1.0, span)
    ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))

    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    ax.set_title("Cells mapped to CCF ijk from (axis, slice, t_lookup, r_um)")
    ax.legend(loc="upper left", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def _plot_cells_ijk_projections(
    *,
    plot_payload: list[tuple[dict[str, object], np.ndarray, np.ndarray, str, np.ndarray]],
    lut_outdir: Path,
    out_path: Path,
) -> None:
    fig2, (ax_ij, ax_ik, ax_jk) = plt.subplots(1, 3, figsize=(20.5, 8.2), constrained_layout=True)
    all_r_signed = np.concatenate([r_signed_plot for _, _, r_signed_plot, _, _t in plot_payload], axis=0)
    if all_r_signed.size == 0 or not np.isfinite(all_r_signed).any():
        raise ValueError("No finite r_signed values available for projection coloring.")
    r_vmin = float(np.nanmin(all_r_signed))
    r_vmax = float(np.nanmax(all_r_signed))
    if abs(r_vmax - r_vmin) <= 1.0e-12:
        r_vmax = r_vmin + 1.0

    scatter_for_colorbar = None
    for info, ijk_plot, r_signed_plot, color, _t_lookup_plot in plot_payload:
        label = (
            f"{info['name']} ({info['axis']}, slice={info['atlas_slice_idx']}, "
            f"used={info['source_slice_used']}, n={ijk_plot.shape[0]})"
        )
        scatter_for_colorbar = ax_ij.scatter(
            ijk_plot[:, 0],
            ijk_plot[:, 1],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=r_signed_plot,
            cmap="coolwarm",
            vmin=r_vmin,
            vmax=r_vmax,
            linewidths=0.0,
            label=label,
            zorder=2,
        )
        ax_ik.scatter(
            ijk_plot[:, 0],
            ijk_plot[:, 2],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=r_signed_plot,
            cmap="coolwarm",
            vmin=r_vmin,
            vmax=r_vmax,
            linewidths=0.0,
            zorder=2,
        )
        ax_jk.scatter(
            ijk_plot[:, 1],
            ijk_plot[:, 2],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=r_signed_plot,
            cmap="coolwarm",
            vmin=r_vmin,
            vmax=r_vmax,
            linewidths=0.0,
            zorder=2,
        )
    if scatter_for_colorbar is None:
        raise ValueError("Failed to create projection scatter for colorbar.")
    cbar = fig2.colorbar(scatter_for_colorbar, ax=[ax_ij, ax_ik, ax_jk], shrink=0.92, pad=0.02)
    cbar.set_label("principal r_signed")

    stack2 = np.vstack([pts for _, pts, _, _, _t in plot_payload])
    i_span = float(max(np.ptp(stack2[:, 0]), 1.0))
    j_span = float(max(np.ptp(stack2[:, 1]), 1.0))
    k_span = float(max(np.ptp(stack2[:, 2]), 1.0))

    cortex_mask_path = lut_outdir / "cortex_mask_3d_ds.npy"
    if not cortex_mask_path.exists():
        raise FileNotFoundError(f"Missing required atlas mask for projection overlays: {cortex_mask_path}")
    cortex_mask = np.asarray(np.load(cortex_mask_path), dtype=bool)

    coronal_midline_paths = _load_midline_paths_csv(
        lut_outdir / "coronal_midline_columns.csv", slice_label="slice_i"
    )
    sagittal_midline_paths = _load_midline_paths_csv(
        lut_outdir / "sagittal_midline_columns.csv", slice_label="slice_k"
    )
    if not coronal_midline_paths:
        raise ValueError(
            f"Missing or empty required midline CSV: {lut_outdir / 'coronal_midline_columns.csv'}"
        )
    if not sagittal_midline_paths:
        raise ValueError(
            f"Missing or empty required midline CSV: {lut_outdir / 'sagittal_midline_columns.csv'}"
        )

    for info, _, _r_signed_plot, color, _t_lookup_plot in plot_payload:
        axis = str(info["axis"])
        atlas_slice_idx = int(info["atlas_slice_idx"])
        if axis == "coronal":
            ax_ij.axvline(atlas_slice_idx, color=color, linewidth=1.2, alpha=0.95, zorder=5)
            ax_ik.axvline(atlas_slice_idx, color=color, linewidth=1.2, alpha=0.95, zorder=5)
            if not (0 <= atlas_slice_idx < cortex_mask.shape[0]):
                raise ValueError(
                    f"Coronal atlas_slice_idx={atlas_slice_idx} out of mask bounds [0,{cortex_mask.shape[0]})."
                )
            contour = _largest_binary_contour(cortex_mask[atlas_slice_idx, :, :])
            if contour is None:
                raise ValueError(f"No coronal contour found in cortex mask at i={atlas_slice_idx}.")
            ax_jk.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linewidth=1.1,
                alpha=0.95,
                zorder=6,
                label=f"Atlas coronal i={atlas_slice_idx} outline",
            )
        elif axis == "sagittal":
            ax_ik.axhline(atlas_slice_idx, color=color, linewidth=1.2, alpha=0.95, zorder=5)
            ax_jk.axhline(atlas_slice_idx, color=color, linewidth=1.2, alpha=0.95, zorder=5)
            if not (0 <= atlas_slice_idx < cortex_mask.shape[2]):
                raise ValueError(
                    f"Sagittal atlas_slice_idx={atlas_slice_idx} out of mask bounds [0,{cortex_mask.shape[2]})."
                )
            contour = _largest_binary_contour(cortex_mask[:, :, atlas_slice_idx])
            if contour is None:
                raise ValueError(f"No sagittal contour found in cortex mask at k={atlas_slice_idx}.")
            ax_ij.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linewidth=1.1,
                alpha=0.95,
                zorder=6,
                label=f"Atlas sagittal k={atlas_slice_idx} outline",
            )
        else:
            raise ValueError(f"Unsupported axis={axis!r}")

    for info, _, _r_signed_plot, color, _t_lookup_plot in plot_payload:
        axis = str(info["axis"])
        atlas_slice_idx = int(info["atlas_slice_idx"])
        path_curve = _curve_ijk_from_midline_csv(
            axis=axis,
            atlas_slice_idx=atlas_slice_idx,
            coronal_paths=coronal_midline_paths,
            sagittal_paths=sagittal_midline_paths,
        )
        if path_curve is None:
            raise ValueError(
                f"Missing required midline curve for axis={axis}, atlas_slice_idx={atlas_slice_idx}."
            )
        curve_ijk, t_all_curve = path_curve

        ax_ij.plot(curve_ijk[:, 0], curve_ijk[:, 1], color=color, linewidth=1.5, alpha=0.9, zorder=7)
        ax_ik.plot(curve_ijk[:, 0], curve_ijk[:, 2], color=color, linewidth=1.5, alpha=0.9, zorder=7)
        ax_jk.plot(curve_ijk[:, 1], curve_ijk[:, 2], color=color, linewidth=1.5, alpha=0.9, zorder=7)

        t_ticks_pos = tuple(float(tt) for tt in T_TICKS)
        label_prefix = "t="
        # Only draw t ticks on the curve's native plane; other projections can be degenerate/misleading.
        if axis == "sagittal":
            _plot_t_ticks_with_labels(
                ax=ax_ij,
                x=curve_ijk[:, 0],
                y=curve_ijk[:, 1],
                t_vals=t_all_curve,
                t_ticks=t_ticks_pos,
                color=color,
                label_prefix=label_prefix,
                xspan=i_span,
                yspan=j_span,
                show_labels=True,
            )
        elif axis == "coronal":
            _plot_t_ticks_with_labels(
                ax=ax_jk,
                x=curve_ijk[:, 1],
                y=curve_ijk[:, 2],
                t_vals=t_all_curve,
                t_ticks=t_ticks_pos,
                color=color,
                label_prefix=label_prefix,
                xspan=j_span,
                yspan=k_span,
                show_labels=True,
            )
        else:
            raise ValueError(f"Unsupported axis={axis!r}")

    ax_ij.set_title("i-j")
    ax_ik.set_title("i-k")
    ax_jk.set_title("j-k")
    ax_ij.set_xlabel("i")
    ax_ij.set_ylabel("j")
    ax_ik.set_xlabel("i")
    ax_ik.set_ylabel("k")
    ax_jk.set_xlabel("j")
    ax_jk.set_ylabel("k")
    ax_ij.set_aspect("equal", adjustable="box")
    ax_ik.set_aspect("equal", adjustable="box")
    ax_jk.set_aspect("equal", adjustable="box")
    ax_ij.invert_yaxis()
    ax_ik.invert_yaxis()
    ax_jk.invert_yaxis()
    ax_ij.legend(loc="upper left", fontsize=8, markerscale=3)
    fig2.savefig(out_path, dpi=240)
    plt.close(fig2)
    print(f"Wrote: {out_path}")


LUTS: dict[AXIS, AxisLut] = {
    "coronal": _load_axis_lut(LUT_OUTDIR, "coronal"),
    "sagittal": _load_axis_lut(LUT_OUTDIR, "sagittal"),
}
print(
    "Loaded LUTs:",
    f"coronal_rows={LUTS['coronal'].source_slice_keys.size}",
    f"sagittal_rows={LUTS['sagittal'].source_slice_keys.size}",
)

# %% [markdown]
# ## Phase 1: Convert each h5ad to ijk using selected t + real r_um

# %%
rng = np.random.default_rng(SEED)
results: list[dict[str, object]] = []

for in_h5ad in INPUT_H5ADS:
    axis, atlas_slice_idx = _read_axis_and_slice(in_h5ad)
    print(f"\nProcessing: {in_h5ad.name}")
    print(f"  axis={axis}, atlas_slice_idx={atlas_slice_idx}")

    adata = ad.read_h5ad(in_h5ad, backed="r")
    if "principal_r_signed" not in adata.obsm:
        raise ValueError(f"Missing obsm['principal_r_signed'] in {in_h5ad}")

    t_fields_all = _resolve_t_fields(adata)
    t_lookup_raw_all, t_lookup_lut_all, t_lookup_name = _resolve_t_lookup_for_lut(t_fields_all)
    r_signed_all = np.asarray(adata.obsm["principal_r_signed"], dtype=np.float64).reshape(-1)
    finite = np.isfinite(t_lookup_lut_all) & np.isfinite(r_signed_all)
    keep_idx = np.flatnonzero(finite)
    if keep_idx.size == 0:
        raise ValueError(f"No finite (t_lookup_mode={T_LOOKUP_MODE}, principal_r_signed) rows in {in_h5ad}")

    t_lookup_raw = np.asarray(t_lookup_raw_all[keep_idx], dtype=np.float64)
    t_lookup = np.asarray(t_lookup_lut_all[keep_idx], dtype=np.float64)
    t_all = np.asarray(t_fields_all.t_all[keep_idx], dtype=np.float64)
    t_local = np.asarray(t_fields_all.t_local[keep_idx], dtype=np.float64)
    r_signed = r_signed_all[keep_idx]

    t_support_lo, t_support_hi, nearest_slice_for_support = _slice_t_support(LUTS[axis], int(atlas_slice_idx))

    r_um, r_floor, r_ceil = _compute_signed_r_um(
        t_lookup=t_lookup,
        r_signed=r_signed,
        n_bins=int(R_BIN_COUNT),
        roll_half_window=int(R_ROLL_HALF_WINDOW),
    )
    outside_support_mask = (t_lookup < float(t_support_lo)) | (t_lookup > float(t_support_hi))
    n_outside_support = int(np.count_nonzero(outside_support_mask))
    outside_support_frac = float(n_outside_support / max(1, int(t_lookup.size)))

    ijk, source_slice_used = _lookup_ijk_batch_slice_locked(
        lut=LUTS[axis],
        atlas_slice_idx=int(atlas_slice_idx),
        t_lookup=t_lookup,
        r_um=r_um,
    )

    # Avoid collisions when different datasets share the same stem (e.g. coronal 2 vs sagittal 2).
    out_npz = OUTDIR / f"{in_h5ad.stem}.{axis}.slice{int(atlas_slice_idx)}.ijk_from_lut.npz"
    np.savez_compressed(
        out_npz,
        h5ad_path=str(in_h5ad),
        axis=axis,
        atlas_slice_idx=np.int32(atlas_slice_idx),
        source_slice_used=np.int32(source_slice_used),
        n_outside_support=np.int32(n_outside_support),
        support_slice_used=np.int32(nearest_slice_for_support),
        t_support_lo=np.float32(t_support_lo),
        t_support_hi=np.float32(t_support_hi),
        t_lookup_raw=t_lookup_raw.astype(np.float32),
        keep_idx=keep_idx.astype(np.int32),
        t_lookup_mode=str(T_LOOKUP_MODE),
        t_lookup_name=str(t_lookup_name),
        t_lookup=t_lookup.astype(np.float32),
        t_all=t_all.astype(np.float32),
        t_local=t_local.astype(np.float32),
        r_signed=r_signed.astype(np.float32),
        r_floor=r_floor.astype(np.float32),
        r_ceil=r_ceil.astype(np.float32),
        r_um=r_um.astype(np.float32),
        ijk=ijk.astype(np.float32),
    )

    t_all_min, t_all_max = _finite_minmax(t_all)
    t_local_min, t_local_max = _finite_minmax(t_local)
    summary = {
        "name": in_h5ad.stem,
        "axis": axis,
        "atlas_slice_idx": int(atlas_slice_idx),
        "source_slice_used": int(source_slice_used),
        "n_outside_support": int(n_outside_support),
        "outside_support_frac": float(outside_support_frac),
        "support_slice_used": int(nearest_slice_for_support),
        "t_support_lo": float(t_support_lo),
        "t_support_hi": float(t_support_hi),
        "lookup_method": "slice_locked",
        "t_lookup_mode": str(T_LOOKUP_MODE),
        "t_lookup_name": str(t_lookup_name),
        "n_total": int(adata.n_obs),
        "n_kept": int(keep_idx.size),
        "t_lookup_min": float(np.min(t_lookup)),
        "t_lookup_max": float(np.max(t_lookup)),
        "t_all_min": t_all_min,
        "t_all_max": t_all_max,
        "t_local_min": t_local_min,
        "t_local_max": t_local_max,
        "r_signed_min": float(np.min(r_signed)),
        "r_signed_max": float(np.max(r_signed)),
        "r_um_min": float(np.min(r_um)),
        "r_um_max": float(np.max(r_um)),
        "out_npz": str(out_npz),
    }
    results.append(summary)
    print(
        f"  kept={summary['n_kept']}/{summary['n_total']}",
        f"t_lookup({summary['t_lookup_name']})=[{summary['t_lookup_min']:.4f},{summary['t_lookup_max']:.4f}]",
        f"support_t=[{summary['t_support_lo']:.4f},{summary['t_support_hi']:.4f}]",
        f"t_all=[{summary['t_all_min']:.4f},{summary['t_all_max']:.4f}]",
        f"t_local=[{summary['t_local_min']:.4f},{summary['t_local_max']:.4f}]",
        f"r_um=[{summary['r_um_min']:.4f},{summary['r_um_max']:.4f}]",
        f"outside_support={summary['n_outside_support']} ({summary['outside_support_frac']:.3%})",
        f"source_slice_used={source_slice_used}",
    )
    adata.file.close()

summary_json = OUTDIR / "phase1_summary.json"
summary_json.write_text(json.dumps(results, indent=2))
print(f"\nWrote: {summary_json}")

# %% [markdown]
# ## Phase 2: 3D ijk scatter plot (mpl3d)

# %%
plot_payload = _build_plot_payload(results_in=results, rng_in=rng)
_plot_cells_ijk_mpl3d(
    plot_payload=plot_payload,
    out_path=OUTDIR / "cells_ijk_mpl3d.png",
)

# %% [markdown]
# ## Phase 2b: 2D projections with atlas sections + t-curve ticks

# %%
_plot_cells_ijk_projections(
    plot_payload=plot_payload,
    lut_outdir=LUT_OUTDIR,
    out_path=OUTDIR / "cells_ijk_projections_2d_with_atlas_sections.png",
)

# %% [markdown]
# ## Phase 2c: Flattened AP/ML plot (anchored ML arclength + optimized AP)

# %%
if PLOT_FLATTENED_AP_ML:
    ap_slice_keys, ap_um_vals = _load_ap_axis_um(AP_AXIS_NPZ)
    s2c = _load_s2c_t2d(S2C_T2D_NPZ)

    coronal_midline_paths = _load_midline_paths_csv(LUT_OUTDIR / "coronal_midline_columns.csv", slice_label="slice_i")
    if not coronal_midline_paths:
        raise ValueError(f"Missing or empty required midline CSV: {LUT_OUTDIR / 'coronal_midline_columns.csv'}")

    ref_slice_i = int(FLAT_REF_SLICE_I)
    if ref_slice_i not in coronal_midline_paths:
        # Snap to a stable slice within AP axis support.
        ref_slice_i = int(ap_slice_keys[int(ap_slice_keys.size // 2)])
    t0_by_slice_i, len_um_by_slice_i = _build_ml_anchor_for_coronal_slices(
        coronal_paths=coronal_midline_paths,
        slice_keys=ap_slice_keys,
        ref_slice_i=ref_slice_i,
        ref_t=float(FLAT_REF_T),
        n_t=int(FLAT_N_T),
        band_frac=float(FLAT_DTW_BAND_FRAC),
        res_j_um=float(R_TARGET_UM_PER_PX),
        res_k_um=float(R_TARGET_UM_PER_PX),
    )

    anchor_keys_i = np.asarray(sorted(t0_by_slice_i.keys()), dtype=np.float64)
    if anchor_keys_i.size < 2:
        raise ValueError("Need at least 2 coronal slices with ML anchors for flattened plotting.")
    anchor_t0 = np.asarray([t0_by_slice_i[int(k)] for k in anchor_keys_i.tolist()], dtype=np.float64)
    anchor_len = np.asarray([len_um_by_slice_i[int(k)] for k in anchor_keys_i.tolist()], dtype=np.float64)
    if anchor_t0.shape != anchor_keys_i.shape or anchor_len.shape != anchor_keys_i.shape:
        raise ValueError("Internal ML anchor shape mismatch.")
    if not np.isfinite(anchor_t0).all() or not np.isfinite(anchor_len).all():
        raise ValueError("Non-finite ML anchor values.")

    flat_payload: list[tuple[dict[str, object], np.ndarray, np.ndarray]] = []
    all_t_cor: list[np.ndarray] = []
    sag_ml_splines: dict[int, CubicSpline] = {}
    for info, _ijk_plot, r_signed_plot, _color, t_lookup_plot in plot_payload:
        axis = str(info["axis"])
        source_slice_used = int(info["source_slice_used"])
        t_lookup_plot = np.asarray(t_lookup_plot, dtype=np.float64).reshape(-1)
        if t_lookup_plot.shape[0] != r_signed_plot.shape[0]:
            raise ValueError(
                f"Flattened plot t_lookup size mismatch for {info['out_npz']}: "
                f"{t_lookup_plot.shape[0]} vs {r_signed_plot.shape[0]}"
            )
        if axis == "coronal":
            cor_slice_f = np.full(t_lookup_plot.shape, float(source_slice_used), dtype=np.float64)
            cor_t = t_lookup_plot.astype(np.float64, copy=False)
        elif axis == "sagittal":
            cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(s2c=s2c, slice_k=int(source_slice_used), t_s=t_lookup_plot)
        else:
            raise ValueError(f"Unsupported axis={axis!r}")

        cor_slice_f = np.asarray(cor_slice_f, dtype=np.float64).reshape(-1)
        cor_slice_f = np.clip(cor_slice_f, float(anchor_keys_i[0]), float(anchor_keys_i[-1]))

        ap_um = _ap_um_for_slice_float(ap_slice_keys, ap_um_vals, cor_slice_f)
        t0 = _interp_with_linear_extrapolation(x=anchor_keys_i, y=anchor_t0, xq=cor_slice_f)
        length = _interp_with_linear_extrapolation(x=anchor_keys_i, y=anchor_len, xq=cor_slice_f)
        if axis == "sagittal" and str(FLAT_SAGITTAL_ML_MODE).strip().lower() == "coronal_k_spline":
            slice_k = int(source_slice_used)
            spline = sag_ml_splines.get(slice_k)
            if spline is None:
                spline = _build_sagittal_ml_spline_from_coronal_k(
                    slice_k=slice_k,
                    coronal_paths=coronal_midline_paths,
                    ap_slice_keys=ap_slice_keys,
                    ap_um_vals=ap_um_vals,
                    t0_by_slice_i=t0_by_slice_i,
                    len_um_by_slice_i=len_um_by_slice_i,
                )
                sag_ml_splines[slice_k] = spline
            ml_um = spline(ap_um).astype(np.float64, copy=False)
        else:
            ml_um = (cor_t - t0) * length

        pts = np.column_stack([ml_um, ap_um]).astype(np.float64, copy=False)
        keep = np.isfinite(pts).all(axis=1) & np.isfinite(cor_t)
        pts = pts[keep]
        t_keep = np.asarray(cor_t, dtype=np.float64)[keep]
        flat_payload.append((info, pts, t_keep))
        all_t_cor.append(t_keep)

    if not all_t_cor:
        raise ValueError("No flattened points available for plotting.")
    t_concat = np.concatenate(all_t_cor, axis=0)
    if t_concat.size == 0 or not np.isfinite(t_concat).any():
        raise ValueError("No finite t_all values for flattened coloring.")
    t_vmin = float(np.nanmin(t_concat))
    t_vmax = float(np.nanmax(t_concat))

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    scatter_for_colorbar = None
    for info, pts, t_vals in flat_payload:
        if pts.shape[0] == 0:
            continue
        label = f"{info['name']} ({info['axis']}, slice={info['atlas_slice_idx']})"
        scatter_for_colorbar = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=t_vals,
            cmap="viridis",
            vmin=t_vmin,
            vmax=t_vmax,
            linewidths=0.0,
            label=label,
        )
    ax.set_xlabel("ML (um; anchored arclength along t)")
    ax.set_ylabel("AP (um; optimized from strip matching)")
    ax.set_title(f"Flattened chart (ref_slice_i={ref_slice_i}, ref_t={float(FLAT_REF_T):.2f})")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, markerscale=3, frameon=False)
    if scatter_for_colorbar is None:
        raise ValueError("Failed to create flattened scatter for colorbar.")
    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("coronal t_all (0..1)")
    out_flat = OUTDIR / "cells_flattened_ap_ml.png"
    fig.tight_layout()
    fig.savefig(out_flat, dpi=240)
    plt.close(fig)
    print(f"Wrote: {out_flat}")

# %% [markdown]
# ## Phase 3: What to check next
#
# - Confirm `phase1_summary.json` has expected source slices and row counts.
# - Inspect `cells_ijk_mpl3d.png` for overlap/separation between coronal and sagittal cohorts.
