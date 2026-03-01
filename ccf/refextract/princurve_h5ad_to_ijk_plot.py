# %% [markdown]
# # Principal-Curve h5ad -> CCF ijk (LUT) + 3D plot
#
# This workflow converts `.syn.annotated*.princurve.h5ad` files into CCF `ijk` coordinates
# using midsurface LUTs and plots both datasets in one 3D scatter.
#
# Run cells sequentially. Artifacts are written under `OUTDIR`.

# %% [markdown]
# ## Configuration

# %%
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from ccf.refextract.midsurface_coords import (
    evaluate_midline_normal_bundle,
    load_coronal_midline_columns,
    load_sagittal_midline_columns,
    nearest_midline_slice_key,
)
from fishtools.ccf.princurve import fit_anchor_curve, project_to_polyline_arclength
from fishtools.utils.io import Workspace
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours

# %%
# === EDIT THESE ===

WORKSPACES: list[Path] = [
    Path("/home/chaichontat/nvme/20251005_JaxA3_Coro2"),
    Path("/home/chaichontat/nvme/20251015_JaxA2_Sag7"),
    Path("/working/20250929_JaxA3_Coro4"),
    Path("/working/20251001_JaxA3_Coro11"),
    Path("/working/20251024_JaxA1_Sag7"),
    Path("/working/20251026_JaxA1_Sag6"),
    Path("/working/20251117_JaxA6_Coro5"),
    Path("/working/20251122_JaxA6_Coro2"),
    Path("/working/20251125_JaxA6_Coro8"),
    Path("/working/20251201_JaxA6_Coro6"),
    Path("/working/20251213_JaxA6_Coro4"),
    Path("/working/20251224_JaxA4_Sag1"),
    Path("/working/20251225_JaxA4_Sag2"),
    Path("/working/20251227_JaxA4_Sag3"),
    Path("/working/20251228_JaxA4_Sag4"),
    Path("/working/20251229_JaxA4_Sag5"),
    Path("/working/20251230_JaxA4_Sag6"),
]

LUT_OUTDIR = Path(
    "/home/chaichontat/fishtools2/ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"
)
LUT_NT = 1024

R_BIN_COUNT = 1024
# Smaller window tracks local thickness changes more tightly (less smoothing).
R_ROLL_HALF_WINDOW = 12
R_SOURCE_UM_PER_PX = 0.216
R_TARGET_UM_PER_PX = 20.0

T_LOOKUP_MODE = "t_all"  # {"t_all", "t_local_to_t_all"}
IJK_MAPPING_MODE = "midline_normal"  # {"lut", "midline_normal"}
REVERSE_SAMPLE_TO_REFERENCE_T = False
RENDER_T_AS_ONE_MINUS_T = REVERSE_SAMPLE_TO_REFERENCE_T
REUSE_PHASE1_FROM_NPZ = False

MAX_PLOT_POINTS_PER_DATASET = 80_000
PLOT_ALPHA = 0.08
PLOT_SIZE = 1.0
PLOT_JITTER_BASE_PX = 0.08
PLOT_JITTER_MAX_PX = 1.0
SEED = 0
REVIEW_CURVE_N_DENSE = 5_000
REVIEW_ANCHOR_SMOOTHING = 0.5
REVIEW_R_SIGN_ENDPOINT_EXTRAPOLATION = 0.25
REVIEW_PANEL_ALPHA = 0.35
REVIEW_PANEL_SIZE = 2.0

OUTDIR = Path("/home/chaichontat/fishtools2/results/refextract/princurve_h5ad_ijk_plot")
OUTDIR.mkdir(parents=True, exist_ok=True)
T_TICKS = tuple(float(v) for v in np.linspace(0.0, 1.0, 11))

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
print(f"IJK_MAPPING_MODE={IJK_MAPPING_MODE}")
print(f"FLAT_SAGITTAL_ML_MODE={FLAT_SAGITTAL_ML_MODE}")
print(
    f"R_SCALE_PX={R_SOURCE_UM_PER_PX / R_TARGET_UM_PER_PX:.8f} ({R_SOURCE_UM_PER_PX}um/px -> {R_TARGET_UM_PER_PX}um/px)"
)


def iter_syn_annotated_h5ads(workspaces: Iterable[Path | str]) -> Iterator[Path]:
    """Yield per-ROI principal-curve files under each workspace CCF transforms directory."""
    for workspace in workspaces:
        ws = Workspace(Path(workspace))
        ccf_dir = ws.ccf_transforms()
        if not ccf_dir.exists():
            print(f"Skipping missing CCF transforms directory: {ccf_dir}")
            continue
        for h5ad in sorted(ccf_dir.glob("*/*.princurve.h5ad")):
            if h5ad.name.endswith("bad.princurve.h5ad"):
                continue
            yield h5ad

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
    return _interp_with_linear_extrapolation(x=keys, y=vals, xq=s).astype(np.float64, copy=False)


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


def _slice_t_support_midline(
    *,
    columns_by_slice: dict[int, object],
    atlas_slice_idx: int,
) -> tuple[float, float, int]:
    source_slice_used = int(nearest_midline_slice_key(columns_by_slice=columns_by_slice, atlas_slice_idx=atlas_slice_idx))
    cols = columns_by_slice[int(source_slice_used)]
    t_vals = np.asarray(getattr(cols, "t"), dtype=np.float64).reshape(-1)
    finite = np.isfinite(t_vals)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError(f"Midline slice={source_slice_used} has insufficient finite t support.")
    return float(np.min(t_vals[finite])), float(np.max(t_vals[finite])), source_slice_used


def _lookup_ijk_batch_midline_normal_slice_locked(
    *,
    axis: AXIS,
    columns_by_slice: dict[int, object],
    atlas_slice_idx: int,
    t_lookup: np.ndarray,
    r_um: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Lookup on nearest midline slice using +pia oriented local normals."""
    t = np.asarray(t_lookup, dtype=np.float64)
    r = np.asarray(r_um, dtype=np.float64)
    if t.shape != r.shape:
        raise ValueError("t_lookup and r_um must have the same shape.")
    if t.ndim != 1:
        raise ValueError("t_lookup and r_um must be 1D.")
    if axis not in {"coronal", "sagittal"}:
        raise ValueError(f"Unsupported axis={axis!r} for midline-normal lookup.")

    source_slice_used = int(nearest_midline_slice_key(columns_by_slice=columns_by_slice, atlas_slice_idx=atlas_slice_idx))
    cols = columns_by_slice[int(source_slice_used)]
    mid_ijk, normal_ijk = evaluate_midline_normal_bundle(columns=cols, axis=axis, t_query=t)
    r_px_target = r / float(R_TARGET_UM_PER_PX)
    ijk = mid_ijk + r_px_target[:, None] * normal_ijk
    if not np.isfinite(ijk).all():
        raise ValueError("Non-finite values produced during midline-normal IJK lookup.")
    return ijk.astype(np.float64, copy=False), source_slice_used


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
    def _sample_to_reference_t(t_values: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t_values, dtype=np.float64)
        if bool(REVERSE_SAMPLE_TO_REFERENCE_T):
            return (1.0 - t_arr).astype(np.float64, copy=False)
        return t_arr.astype(np.float64, copy=False)

    mode = str(T_LOOKUP_MODE).strip().lower()
    if mode == "t_all":
        t_lut = _sample_to_reference_t(t_fields.t_all)
        name = "t_all_rev" if bool(REVERSE_SAMPLE_TO_REFERENCE_T) else "t_all"
        return t_fields.t_all, t_lut, name
    if mode == "t_local_to_t_all":
        local_to_all_affine = TTypeConverter.fit_local_to_all_affine(
            t_local=t_fields.t_local,
            t_all=t_fields.t_all,
        )
        converter = TTypeConverter(local_to_all_affine=local_to_all_affine)
        t_lut = _sample_to_reference_t(converter.t_local_to_t_all(t_fields.t_local))
        name = "t_local_rev" if bool(REVERSE_SAMPLE_TO_REFERENCE_T) else "t_local"
        return t_fields.t_local, t_lut, name
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


def _render_t_values(t_vals: np.ndarray) -> np.ndarray:
    t_arr = np.asarray(t_vals, dtype=np.float64)
    if bool(RENDER_T_AS_ONE_MINUS_T):
        return (1.0 - t_arr).astype(np.float64, copy=False)
    return t_arr.astype(np.float64, copy=False)


def _extract_anchor_ids_from_payload(payload: dict[str, object]) -> list[str]:
    raw = payload.get("anchors")
    if isinstance(raw, list) and raw:
        out: list[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = item.get("cell_id")
            if isinstance(cid, str) and cid != "":
                out.append(cid)
        if len(out) >= 2:
            return out

    start = payload.get("start")
    end = payload.get("end")
    out2: list[str] = []
    for item in (start, end):
        if not isinstance(item, dict):
            continue
        cid = item.get("cell_id")
        if isinstance(cid, str) and cid != "":
            out2.append(cid)
    return out2


def _anchor_json_path_for_h5ad(h5ad_path: Path) -> Path | None:
    stem = str(h5ad_path.stem)
    base_stem = stem[: -len(".princurve")] if stem.endswith(".princurve") else stem

    primary_candidates = [
        h5ad_path.with_name(f"{base_stem}.anchors.json"),
        h5ad_path.with_name(f"{stem}.anchors.json"),
    ]
    for cand in primary_candidates:
        if cand.exists():
            return cand

    glob_patterns = [f"{base_stem}*.anchors.json"]
    if base_stem != stem:
        glob_patterns.append(f"{stem}*.anchors.json")

    candidates: list[Path] = []
    for pattern in glob_patterns:
        candidates.extend(sorted(h5ad_path.parent.glob(pattern)))
    uniq_candidates = sorted(set(candidates))
    if len(uniq_candidates) == 1:
        return uniq_candidates[0]
    return None


def _load_review_panel_data(
    *,
    h5ad_path: Path,
    keep_idx: np.ndarray,
    fallback_r_signed: np.ndarray,
    sample_idx_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str | None]:
    """Load review-space points and signed-r using pick_curve_anchors logic when anchors exist."""
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "spatial" not in adata.obsm:
            raise ValueError(f"Missing obsm['spatial'] in {h5ad_path}.")
        xy_full = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if xy_full.ndim != 2 or xy_full.shape[1] < 2:
            raise ValueError(f"Unexpected spatial shape in {h5ad_path}: {xy_full.shape}.")
        xy_full = xy_full[:, :2]
        cell_ids = adata.obs_names.astype(str).to_numpy()
    finally:
        if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
            adata.file.close()

    keep = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
    if keep.size == 0:
        raise ValueError(f"Empty keep_idx in {h5ad_path}.")
    if int(np.min(keep)) < 0 or int(np.max(keep)) >= int(xy_full.shape[0]):
        raise ValueError(
            f"keep_idx out of bounds for {h5ad_path}: min={int(np.min(keep))}, "
            f"max={int(np.max(keep))}, n_obs={xy_full.shape[0]}."
        )

    sample_idx = np.asarray(sample_idx_keep, dtype=np.int64).reshape(-1)
    if sample_idx.size == 0:
        raise ValueError(f"Empty sample_idx_keep in {h5ad_path}.")
    if int(np.min(sample_idx)) < 0 or int(np.max(sample_idx)) >= int(keep.shape[0]):
        raise ValueError(
            f"sample_idx_keep out of bounds for {h5ad_path}: min={int(np.min(sample_idx))}, "
            f"max={int(np.max(sample_idx))}, n_keep={keep.shape[0]}."
        )

    r_fallback_full = np.asarray(fallback_r_signed, dtype=np.float64).reshape(-1)
    if r_fallback_full.shape[0] != keep.shape[0]:
        raise ValueError(
            f"fallback_r_signed/keep_idx size mismatch for {h5ad_path}: {r_fallback_full.shape[0]} vs {keep.shape[0]}."
        )
    r_fallback = np.asarray(r_fallback_full[sample_idx], dtype=np.float64)

    xy_keep = xy_full[keep]
    xy_plot = np.asarray(xy_keep[sample_idx], dtype=np.float64)
    r_keep = r_fallback.copy()
    curve_xy = np.empty((0, 2), dtype=np.float64)
    anchor_xy = np.empty((0, 2), dtype=np.float64)

    anchor_path = _anchor_json_path_for_h5ad(h5ad_path)
    if anchor_path is None:
        return xy_plot, r_keep, curve_xy, anchor_xy, "missing anchors json"

    payload = json.loads(anchor_path.read_text())
    anchor_ids = _extract_anchor_ids_from_payload(payload)
    if len(anchor_ids) < 2:
        return xy_plot, r_keep, curve_xy, anchor_xy, "anchors json has fewer than 2 anchors"

    cell_to_index = {cid: i for i, cid in enumerate(cell_ids)}
    missing_ids = [cid for cid in anchor_ids if cid not in cell_to_index]
    if missing_ids:
        return (
            xy_plot,
            r_keep,
            curve_xy,
            anchor_xy,
            f"anchors missing in h5ad: {len(missing_ids)}",
        )

    anchor_idx = np.asarray([cell_to_index[cid] for cid in anchor_ids], dtype=np.int64)
    anchor_xy = xy_full[anchor_idx]
    curve_xy = fit_anchor_curve(
        anchor_xy=anchor_xy,
        n_dense=int(REVIEW_CURVE_N_DENSE),
        smoothing=float(REVIEW_ANCHOR_SMOOTHING),
    )
    _t, r_sampled, _proj = project_to_polyline_arclength(
        xy=xy_plot,
        line=curve_xy,
        k=50,
        endpoint_extrapolation=float(REVIEW_R_SIGN_ENDPOINT_EXTRAPOLATION),
    )
    reverse_r_sign = payload.get("reverse_r_sign")
    if reverse_r_sign is None:
        reverse = False
    elif isinstance(reverse_r_sign, bool):
        reverse = bool(reverse_r_sign)
    else:
        raise ValueError(f"Invalid reverse_r_sign in {anchor_path}: expected bool, got {type(reverse_r_sign).__name__}.")
    if reverse:
        r_sampled = -np.asarray(r_sampled, dtype=np.float64)
    else:
        r_sampled = np.asarray(r_sampled, dtype=np.float64)

    r_keep = r_sampled
    bad = ~np.isfinite(r_keep)
    if np.any(bad):
        r_keep[bad] = r_fallback[bad]
    return xy_plot, r_keep, curve_xy, anchor_xy, None


def _native_plane_xy_from_ijk(axis: AXIS, ijk: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    ijk_arr = np.asarray(ijk, dtype=np.float64)
    if ijk_arr.ndim != 2 or ijk_arr.shape[1] != 3:
        raise ValueError(f"Expected ijk shape (N,3), got {ijk_arr.shape}.")
    if axis == "sagittal":
        return ijk_arr[:, 0], ijk_arr[:, 1], "i", "j"
    if axis == "coronal":
        return ijk_arr[:, 1], ijk_arr[:, 2], "j", "k"
    raise ValueError(f"Unsupported axis={axis!r}")


def _info_display_name(info: dict[str, object]) -> str:
    base = str(info["name"])
    dataset = str(info.get("dataset_name", "")).strip()
    if dataset == "":
        return base
    return f"{dataset}/{base}"


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
        r_um = np.asarray(data["r_um"], dtype=np.float64)
        t_lookup = np.asarray(data["t_lookup"], dtype=np.float64)
        if r_signed.shape[0] != ijk.shape[0]:
            raise ValueError(
                f"Mismatched ijk/r_signed rows for {info['out_npz']}: {ijk.shape[0]} vs {r_signed.shape[0]}."
            )
        if r_um.shape[0] != ijk.shape[0]:
            raise ValueError(
                f"Mismatched ijk/r_um rows for {info['out_npz']}: {ijk.shape[0]} vs {r_um.shape[0]}."
            )
        if t_lookup.shape[0] != ijk.shape[0]:
            raise ValueError(
                f"Mismatched ijk/t_lookup rows for {info['out_npz']}: {ijk.shape[0]} vs {t_lookup.shape[0]}."
            )
        if ijk.shape[0] > int(MAX_PLOT_POINTS_PER_DATASET):
            pick = rng_in.choice(ijk.shape[0], size=int(MAX_PLOT_POINTS_PER_DATASET), replace=False)
            ijk_plot = ijk[pick]
            r_signed_plot = r_signed[pick]
            r_um_plot = r_um[pick]
            t_lookup_plot = t_lookup[pick]
        else:
            ijk_plot = ijk
            r_signed_plot = r_signed
            r_um_plot = r_um
            t_lookup_plot = t_lookup
        r_um_plot = np.clip(np.asarray(r_um_plot, dtype=np.float64), 0.0, np.inf)
        r_hi = float(np.nanpercentile(r_um_plot, 99.0))
        if not np.isfinite(r_hi) or r_hi <= 1.0e-12:
            raise ValueError(f"Invalid r_um distribution for jitter in {info['out_npz']}.")
        r_norm = np.clip(r_um_plot / r_hi, 0.0, 1.0)
        jitter_sigma = float(PLOT_JITTER_BASE_PX) + (
            float(PLOT_JITTER_MAX_PX) - float(PLOT_JITTER_BASE_PX)
        ) * r_norm
        jitter = rng_in.normal(loc=0.0, scale=1.0, size=ijk_plot.shape).astype(np.float64, copy=False)
        ijk_plot = ijk_plot + jitter * jitter_sigma[:, None]
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
        disp = _info_display_name(info)
        label = (
            f"{disp} ({info['axis']}, slice={info['atlas_slice_idx']}, "
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
        disp = _info_display_name(info)
        label = (
            f"{disp} ({info['axis']}, slice={info['atlas_slice_idx']}, "
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
        t_curve_render = _render_t_values(t_all_curve)
        # Only draw t ticks on the curve's native plane; other projections can be degenerate/misleading.
        if axis == "sagittal":
            _plot_t_ticks_with_labels(
                ax=ax_ij,
                x=curve_ijk[:, 0],
                y=curve_ijk[:, 1],
                t_vals=t_curve_render,
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
                t_vals=t_curve_render,
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


def _plot_cells_ijk_per_roi_axes(
    *,
    plot_payload: list[tuple[dict[str, object], np.ndarray, np.ndarray, str, np.ndarray]],
    out_path: Path,
) -> None:
    n_items = len(plot_payload)
    if n_items == 0:
        raise ValueError("No plot payload available for per-ROI axes figure.")
    ncols = int(min(8, max(3, int(np.ceil(np.sqrt(n_items))))))
    nrows = int(np.ceil(n_items / ncols))
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axs.ravel()

    all_r_signed = np.concatenate([r_signed_plot for _, _, r_signed_plot, _, _t in plot_payload], axis=0)
    if all_r_signed.size == 0 or not np.isfinite(all_r_signed).any():
        raise ValueError("No finite r_signed values available for per-ROI coloring.")
    r_vmin = float(np.nanmin(all_r_signed))
    r_vmax = float(np.nanmax(all_r_signed))
    if abs(r_vmax - r_vmin) <= 1.0e-12:
        r_vmax = r_vmin + 1.0

    x_mins: list[float] = []
    x_maxs: list[float] = []
    y_mins: list[float] = []
    y_maxs: list[float] = []
    scatter_for_colorbar = None
    for ax, (info, ijk_plot, r_signed_plot, _color, _t_lookup_plot) in zip(axes, plot_payload, strict=False):
        axis = str(info["axis"])
        if axis == "sagittal":
            x = ijk_plot[:, 0]
            y = ijk_plot[:, 1]
            x_label, y_label = "i", "j"
        elif axis == "coronal":
            x = ijk_plot[:, 1]
            y = ijk_plot[:, 2]
            x_label, y_label = "j", "k"
        else:
            raise ValueError(f"Unsupported axis={axis!r}")
        x_mins.append(float(np.min(x)))
        x_maxs.append(float(np.max(x)))
        y_mins.append(float(np.min(y)))
        y_maxs.append(float(np.max(y)))
        scatter_for_colorbar = ax.scatter(
            x,
            y,
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=r_signed_plot,
            cmap="coolwarm",
            vmin=r_vmin,
            vmax=r_vmax,
            linewidths=0.0,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title(
            f"{_info_display_name(info)}\n{axis} slice={int(info['atlas_slice_idx'])}",
            fontsize=8,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if scatter_for_colorbar is None:
        raise ValueError("Failed to create per-ROI scatter for colorbar.")
    for ax in axes[n_items:]:
        ax.axis("off")

    x_min = float(np.min(np.asarray(x_mins, dtype=np.float64)))
    x_max = float(np.max(np.asarray(x_maxs, dtype=np.float64)))
    y_min = float(np.min(np.asarray(y_mins, dtype=np.float64)))
    y_max = float(np.max(np.asarray(y_maxs, dtype=np.float64)))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    for ax in axes[:n_items]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    fig.suptitle("Per-ROI native-plane projections (shared scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def _plot_review_and_native_per_roi_axes_4col(
    *,
    results_in: list[dict[str, object]],
    lut_outdir: Path,
    out_path: Path,
    rng_in: np.random.Generator,
) -> None:
    """Plot per-ROI paired panels: review-space + transformed native-plane in a shared 4-column grid."""
    if not results_in:
        raise ValueError("No results available for review/native paired plotting.")

    coronal_midline_paths = _load_midline_paths_csv(
        lut_outdir / "coronal_midline_columns.csv", slice_label="slice_i"
    )
    sagittal_midline_paths = _load_midline_paths_csv(
        lut_outdir / "sagittal_midline_columns.csv", slice_label="slice_k"
    )

    panels: list[dict[str, object]] = []
    for info in results_in:
        out_npz = Path(str(info["out_npz"]))
        data = np.load(out_npz)
        ijk = np.asarray(data["ijk"], dtype=np.float64)
        r_native = np.asarray(data["r_signed"], dtype=np.float64).reshape(-1)
        keep_idx = np.asarray(data["keep_idx"], dtype=np.int64).reshape(-1)
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise ValueError(f"Unexpected ijk shape in {out_npz}: {ijk.shape}.")
        if r_native.shape[0] != ijk.shape[0]:
            raise ValueError(f"r_signed/ijk mismatch in {out_npz}: {r_native.shape[0]} vs {ijk.shape[0]}.")
        if keep_idx.shape[0] != ijk.shape[0]:
            raise ValueError(f"keep_idx/ijk mismatch in {out_npz}: {keep_idx.shape[0]} vs {ijk.shape[0]}.")

        h5ad_raw = np.asarray(data["h5ad_path"]).reshape(-1)[0]
        h5ad_path = Path(str(h5ad_raw))

        n_pts = ijk.shape[0]
        if n_pts > int(MAX_PLOT_POINTS_PER_DATASET):
            pick = rng_in.choice(n_pts, size=int(MAX_PLOT_POINTS_PER_DATASET), replace=False)
        else:
            pick = np.arange(n_pts, dtype=np.int64)

        ijk_plot = ijk[pick]
        review_xy_plot, r_plot, review_curve_xy, anchor_xy, review_warning = _load_review_panel_data(
            h5ad_path=h5ad_path,
            keep_idx=keep_idx,
            fallback_r_signed=r_native,
            sample_idx_keep=pick,
        )

        axis = str(info["axis"])
        atlas_slice_idx = int(info["atlas_slice_idx"])
        native_x, native_y, native_x_label, native_y_label = _native_plane_xy_from_ijk(axis=axis, ijk=ijk_plot)
        native_xy_plot = np.column_stack([native_x, native_y]).astype(np.float64, copy=False)

        curve_ijk_t = _curve_ijk_from_midline_csv(
            axis=axis,
            atlas_slice_idx=atlas_slice_idx,
            coronal_paths=coronal_midline_paths,
            sagittal_paths=sagittal_midline_paths,
        )
        if curve_ijk_t is None:
            native_curve_xy = np.empty((0, 2), dtype=np.float64)
            native_curve_t = np.empty((0,), dtype=np.float64)
        else:
            curve_ijk, t_curve = curve_ijk_t
            if axis == "sagittal":
                native_curve_xy = np.column_stack([curve_ijk[:, 0], curve_ijk[:, 1]]).astype(np.float64, copy=False)
            elif axis == "coronal":
                native_curve_xy = np.column_stack([curve_ijk[:, 1], curve_ijk[:, 2]]).astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unsupported axis={axis!r}.")
            native_curve_t = np.asarray(t_curve, dtype=np.float64)

        panels.append(
            {
                "info": info,
                "review_xy_plot": review_xy_plot,
                "r_plot": r_plot,
                "review_curve_xy": review_curve_xy,
                "anchor_xy": anchor_xy,
                "review_warning": review_warning,
                "native_xy_plot": native_xy_plot,
                "native_curve_xy": native_curve_xy,
                "native_curve_t": native_curve_t,
                "native_x_label": native_x_label,
                "native_y_label": native_y_label,
            }
        )

    review_xy_parts: list[np.ndarray] = []
    native_xy_parts: list[np.ndarray] = []
    for panel in panels:
        review_xy_parts.append(np.asarray(panel["review_xy_plot"], dtype=np.float64))
        review_curve_xy = np.asarray(panel["review_curve_xy"], dtype=np.float64)
        if review_curve_xy.ndim == 2 and review_curve_xy.shape[0] >= 2:
            review_xy_parts.append(review_curve_xy)
        anchor_xy = np.asarray(panel["anchor_xy"], dtype=np.float64)
        if anchor_xy.ndim == 2 and anchor_xy.shape[0] >= 1:
            review_xy_parts.append(anchor_xy)

        native_xy_parts.append(np.asarray(panel["native_xy_plot"], dtype=np.float64))
        native_curve_xy = np.asarray(panel["native_curve_xy"], dtype=np.float64)
        if native_curve_xy.ndim == 2 and native_curve_xy.shape[0] >= 2:
            native_xy_parts.append(native_curve_xy)

    review_xy_all = np.vstack(review_xy_parts)
    native_xy_all = np.vstack(native_xy_parts)
    if review_xy_all.size == 0 or native_xy_all.size == 0:
        raise ValueError("No review/native points available for paired plotting.")

    review_x_min = float(np.nanmin(review_xy_all[:, 0]))
    review_x_max = float(np.nanmax(review_xy_all[:, 0]))
    review_y_min = float(np.nanmin(review_xy_all[:, 1]))
    review_y_max = float(np.nanmax(review_xy_all[:, 1]))
    native_x_min = float(np.nanmin(native_xy_all[:, 0]))
    native_x_max = float(np.nanmax(native_xy_all[:, 0]))
    native_y_min = float(np.nanmin(native_xy_all[:, 1]))
    native_y_max = float(np.nanmax(native_xy_all[:, 1]))
    if review_x_max <= review_x_min:
        review_x_max = review_x_min + 1.0
    if review_y_max <= review_y_min:
        review_y_max = review_y_min + 1.0
    if native_x_max <= native_x_min:
        native_x_max = native_x_min + 1.0
    if native_y_max <= native_y_min:
        native_y_max = native_y_min + 1.0
    native_x_span = float(max(native_x_max - native_x_min, 1.0))
    native_y_span = float(max(native_y_max - native_y_min, 1.0))

    r_all = np.concatenate([np.asarray(p["r_plot"], dtype=np.float64) for p in panels], axis=0)
    finite_abs = np.abs(r_all[np.isfinite(r_all)])
    r_lim = float(np.quantile(finite_abs, 0.99)) if finite_abs.size else 1.0
    if not np.isfinite(r_lim) or r_lim <= 0.0:
        r_lim = 1.0

    n_items = len(panels)
    nrows = int(np.ceil(n_items / 2))
    fig, axs = plt.subplots(
        nrows,
        4,
        figsize=(16.0, 3.8 * nrows),
        squeeze=False,
    )

    used_axes: set[tuple[int, int]] = set()
    for idx, panel in enumerate(panels):
        row = int(idx // 2)
        review_col = int((idx % 2) * 2)
        native_col = review_col + 1
        used_axes.add((row, review_col))
        used_axes.add((row, native_col))

        info = panel["info"]
        review_xy_plot = np.asarray(panel["review_xy_plot"], dtype=np.float64)
        native_xy_plot = np.asarray(panel["native_xy_plot"], dtype=np.float64)
        r_plot = np.asarray(panel["r_plot"], dtype=np.float64)
        review_curve_xy = np.asarray(panel["review_curve_xy"], dtype=np.float64)
        anchor_xy = np.asarray(panel["anchor_xy"], dtype=np.float64)
        native_curve_xy = np.asarray(panel["native_curve_xy"], dtype=np.float64)
        native_curve_t = np.asarray(panel["native_curve_t"], dtype=np.float64)
        native_x_label = str(panel["native_x_label"])
        native_y_label = str(panel["native_y_label"])
        review_warning = panel["review_warning"]

        ax_review = axs[row, review_col]
        ax_native = axs[row, native_col]

        ax_review.scatter(
            review_xy_plot[:, 0],
            review_xy_plot[:, 1],
            c=r_plot,
            s=float(REVIEW_PANEL_SIZE),
            alpha=float(REVIEW_PANEL_ALPHA),
            cmap="coolwarm",
            vmin=-r_lim,
            vmax=r_lim,
            linewidths=0.0,
            zorder=2,
        )
        if review_curve_xy.shape[0] >= 2:
            ax_review.plot(review_curve_xy[:, 0], review_curve_xy[:, 1], color="black", linewidth=1.3, zorder=3)
        if anchor_xy.shape[0] >= 1:
            ax_review.scatter(
                anchor_xy[:, 0],
                anchor_xy[:, 1],
                c="yellow",
                s=24.0,
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
            )
            ax_review.scatter(
                [anchor_xy[0, 0]],
                [anchor_xy[0, 1]],
                c="lime",
                s=40.0,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            ax_review.scatter(
                [anchor_xy[-1, 0]],
                [anchor_xy[-1, 1]],
                c="red",
                s=40.0,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
        if isinstance(review_warning, str):
            ax_review.text(
                0.01,
                0.99,
                review_warning,
                transform=ax_review.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

        ax_review.set_xlim(review_x_min, review_x_max)
        ax_review.set_ylim(review_y_min, review_y_max)
        ax_review.set_aspect("equal", adjustable="box")
        ax_review.set_xlabel("x")
        ax_review.set_ylabel("y")
        ax_review.set_title(
            f"{_info_display_name(info)} review\n{info['axis']} slice={int(info['atlas_slice_idx'])}",
            fontsize=8,
        )

        ax_native.scatter(
            native_xy_plot[:, 0],
            native_xy_plot[:, 1],
            s=float(PLOT_SIZE),
            alpha=float(PLOT_ALPHA),
            c=r_plot,
            cmap="coolwarm",
            vmin=-r_lim,
            vmax=r_lim,
            linewidths=0.0,
            zorder=2,
        )
        if native_curve_xy.shape[0] >= 2:
            native_curve_t_render = _render_t_values(native_curve_t)
            ax_native.plot(
                native_curve_xy[:, 0],
                native_curve_xy[:, 1],
                color="black",
                linewidth=1.3,
                alpha=0.95,
                zorder=4,
            )
            _plot_t_ticks_with_labels(
                ax=ax_native,
                x=native_curve_xy[:, 0],
                y=native_curve_xy[:, 1],
                t_vals=native_curve_t_render,
                t_ticks=T_TICKS,
                color="black",
                label_prefix="t=",
                xspan=native_x_span,
                yspan=native_y_span,
                show_labels=True,
            )
        ax_native.set_xlim(native_x_min, native_x_max)
        ax_native.set_ylim(native_y_max, native_y_min)
        ax_native.set_aspect("equal", adjustable="box")
        ax_native.set_xlabel(native_x_label)
        ax_native.set_ylabel(native_y_label)
        ax_native.set_title(
            f"{_info_display_name(info)} native\n{info['axis']} slice={int(info['atlas_slice_idx'])}",
            fontsize=8,
        )

    for row in range(nrows):
        for col in range(4):
            if (row, col) not in used_axes:
                axs[row, col].axis("off")

    fig.suptitle("Per-ROI paired panels: review-space and transformed native-plane", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def _load_cached_phase1_summary(
    *,
    out_npz: Path,
    in_h5ad: Path,
    axis: AXIS,
    atlas_slice_idx: int,
    dataset_name: str,
) -> dict[str, object] | None:
    d = np.load(out_npz)
    required = {
        "h5ad_path",
        "axis",
        "atlas_slice_idx",
        "source_slice_used",
        "n_outside_support",
        "support_slice_used",
        "t_support_lo",
        "t_support_hi",
        "t_lookup_mode",
        "t_lookup_name",
        "t_lookup",
        "t_all",
        "t_local",
        "t_neomeso",
        "r_signed",
        "r_um",
        "n_total",
    }
    if not required.issubset(set(d.files)):
        return None

    cached_h5ad = Path(str(np.asarray(d["h5ad_path"]).reshape(-1)[0]))
    cached_axis = str(np.asarray(d["axis"]).reshape(-1)[0])
    cached_slice = int(np.asarray(d["atlas_slice_idx"]).reshape(-1)[0])
    if "ijk_mapping_mode" in d.files:
        cached_mapping_mode = str(np.asarray(d["ijk_mapping_mode"]).reshape(-1)[0]).strip().lower()
    else:
        cached_mapping_mode = "lut"
    expected_mapping_mode = str(IJK_MAPPING_MODE).strip().lower()
    if cached_h5ad != in_h5ad or cached_axis != axis or cached_slice != int(atlas_slice_idx):
        return None
    if cached_mapping_mode != expected_mapping_mode:
        return None

    t_lookup = np.asarray(d["t_lookup"], dtype=np.float64).reshape(-1)
    t_all = np.asarray(d["t_all"], dtype=np.float64).reshape(-1)
    t_local = np.asarray(d["t_local"], dtype=np.float64).reshape(-1)
    t_neomeso = np.asarray(d["t_neomeso"], dtype=np.float64).reshape(-1)
    r_signed = np.asarray(d["r_signed"], dtype=np.float64).reshape(-1)
    r_um = np.asarray(d["r_um"], dtype=np.float64).reshape(-1)
    if not (
        t_lookup.shape == t_all.shape == t_local.shape == t_neomeso.shape == r_signed.shape == r_um.shape
    ):
        return None

    n_kept = int(t_lookup.shape[0])
    if n_kept == 0:
        return None
    n_outside = int(np.asarray(d["n_outside_support"]).reshape(-1)[0])
    outside_frac = float(n_outside / max(1, n_kept))

    t_all_min, t_all_max = _finite_minmax(t_all)
    t_local_min, t_local_max = _finite_minmax(t_local)
    t_neomeso_min, t_neomeso_max = _finite_minmax(t_neomeso)

    return {
        "name": in_h5ad.stem,
        "dataset_name": str(dataset_name),
        "axis": axis,
        "atlas_slice_idx": int(atlas_slice_idx),
        "source_slice_used": int(np.asarray(d["source_slice_used"]).reshape(-1)[0]),
        "n_outside_support": n_outside,
        "outside_support_frac": outside_frac,
        "support_slice_used": int(np.asarray(d["support_slice_used"]).reshape(-1)[0]),
        "t_support_lo": float(np.asarray(d["t_support_lo"]).reshape(-1)[0]),
        "t_support_hi": float(np.asarray(d["t_support_hi"]).reshape(-1)[0]),
        "lookup_method": f"slice_locked_{cached_mapping_mode}",
        "ijk_mapping_mode": str(cached_mapping_mode),
        "t_lookup_mode": str(np.asarray(d["t_lookup_mode"]).reshape(-1)[0]),
        "t_lookup_name": str(np.asarray(d["t_lookup_name"]).reshape(-1)[0]),
        "n_total": int(np.asarray(d["n_total"]).reshape(-1)[0]),
        "n_kept": n_kept,
        "t_lookup_min": float(np.min(t_lookup)),
        "t_lookup_max": float(np.max(t_lookup)),
        "t_all_min": t_all_min,
        "t_all_max": t_all_max,
        "t_local_min": t_local_min,
        "t_local_max": t_local_max,
        "t_neomeso_min": t_neomeso_min,
        "t_neomeso_max": t_neomeso_max,
        "r_signed_min": float(np.min(r_signed)),
        "r_signed_max": float(np.max(r_signed)),
        "r_um_min": float(np.min(r_um)),
        "r_um_max": float(np.max(r_um)),
        "out_npz": str(out_npz),
    }


LUTS: dict[AXIS, AxisLut] = {
    "coronal": _load_axis_lut(LUT_OUTDIR, "coronal"),
    "sagittal": _load_axis_lut(LUT_OUTDIR, "sagittal"),
}
if str(IJK_MAPPING_MODE).strip().lower() not in {"lut", "midline_normal"}:
    raise ValueError(f"Unsupported IJK_MAPPING_MODE={IJK_MAPPING_MODE!r}; expected {{'lut', 'midline_normal'}}.")
MIDLINE_COLUMNS: dict[AXIS, dict[int, object]] = {
    "coronal": load_coronal_midline_columns(LUT_OUTDIR / "coronal_midline_columns.csv"),
    "sagittal": load_sagittal_midline_columns(LUT_OUTDIR / "sagittal_midline_columns.csv"),
}
print(
    "Loaded LUTs:",
    f"coronal_rows={LUTS['coronal'].source_slice_keys.size}",
    f"sagittal_rows={LUTS['sagittal'].source_slice_keys.size}",
)
print(
    "Loaded midline columns:",
    f"coronal_rows={len(MIDLINE_COLUMNS['coronal'])}",
    f"sagittal_rows={len(MIDLINE_COLUMNS['sagittal'])}",
)

# %% [markdown]
# ## Phase 1: Convert each h5ad to ijk using selected t + real r_um

# %%
rng = np.random.default_rng(SEED)
results: list[dict[str, object]] = []
input_h5ads = list(iter_syn_annotated_h5ads(WORKSPACES))
if not input_h5ads:
    workspace_lines = "\n".join(f"  - {ws}" for ws in WORKSPACES)
    raise FileNotFoundError(
        "No principal-curve h5ad files found via workspace discovery. "
        "Expected matches at <workspace>/analysis/output/ccf-transforms/*/*.princurve.h5ad "
        "(excluding *bad.princurve.h5ad).\n"
        f"Workspaces searched:\n{workspace_lines}"
    )
print(f"Discovered {len(input_h5ads)} principal-curve h5ad files across {len(WORKSPACES)} workspaces.")
mapping_mode_tag = str(IJK_MAPPING_MODE).strip().lower()

for in_h5ad in input_h5ads:
    axis, atlas_slice_idx = _read_axis_and_slice(in_h5ad)
    dataset_name = in_h5ad.parents[4].name if len(in_h5ad.parents) >= 5 else in_h5ad.parent.name
    out_npz = OUTDIR / f"{in_h5ad.stem}.{axis}.slice{int(atlas_slice_idx)}.ijk_from_{mapping_mode_tag}.npz"
    print(f"\nProcessing: {in_h5ad.name}")
    print(f"  axis={axis}, atlas_slice_idx={atlas_slice_idx}")
    if bool(REUSE_PHASE1_FROM_NPZ) and out_npz.exists():
        cached = _load_cached_phase1_summary(
            out_npz=out_npz,
            in_h5ad=in_h5ad,
            axis=axis,
            atlas_slice_idx=int(atlas_slice_idx),
            dataset_name=str(dataset_name),
        )
        if cached is not None:
            results.append(cached)
            print(
                f"  reused={cached['n_kept']}/{cached['n_total']}",
                f"t_lookup({cached['t_lookup_name']})=[{cached['t_lookup_min']:.4f},{cached['t_lookup_max']:.4f}]",
                f"support_t=[{cached['t_support_lo']:.4f},{cached['t_support_hi']:.4f}]",
                f"outside_support={cached['n_outside_support']} ({cached['outside_support_frac']:.3%})",
                f"source_slice_used={cached['source_slice_used']}",
            )
            continue

    adata = ad.read_h5ad(in_h5ad, backed="r")
    if "principal_r_signed" not in adata.obsm:
        raise ValueError(f"Missing obsm['principal_r_signed'] in {in_h5ad}")

    t_fields_all = _resolve_t_fields(adata)
    t_lookup_raw_all, t_lookup_lut_all, t_lookup_name = _resolve_t_lookup_for_lut(t_fields_all)
    r_signed_all = np.asarray(adata.obsm["principal_r_signed"], dtype=np.float64).reshape(-1)
    if "t_neomeso" not in adata.obs.columns:
        raise ValueError(f"Missing obs['t_neomeso'] in {in_h5ad}")
    t_neomeso_all = np.asarray(adata.obs["t_neomeso"].to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    in_neomeso = np.isfinite(t_neomeso_all) & (t_neomeso_all >= 0.0) & (t_neomeso_all <= 1.0)
    finite = np.isfinite(t_lookup_lut_all) & np.isfinite(r_signed_all) & in_neomeso
    keep_idx = np.flatnonzero(finite)
    if keep_idx.size == 0:
        raise ValueError(
            f"No rows in t_neomeso range (0..1) with finite "
            f"(t_lookup_mode={T_LOOKUP_MODE}, principal_r_signed) in {in_h5ad}"
        )

    t_lookup_raw = np.asarray(t_lookup_raw_all[keep_idx], dtype=np.float64)
    t_lookup = np.asarray(t_lookup_lut_all[keep_idx], dtype=np.float64)
    t_all = np.asarray(t_fields_all.t_all[keep_idx], dtype=np.float64)
    t_local = np.asarray(t_fields_all.t_local[keep_idx], dtype=np.float64)
    t_neomeso = np.asarray(t_neomeso_all[keep_idx], dtype=np.float64)
    r_signed = r_signed_all[keep_idx]

    if mapping_mode_tag == "lut":
        t_support_lo, t_support_hi, nearest_slice_for_support = _slice_t_support(LUTS[axis], int(atlas_slice_idx))
    else:
        t_support_lo, t_support_hi, nearest_slice_for_support = _slice_t_support_midline(
            columns_by_slice=MIDLINE_COLUMNS[axis], atlas_slice_idx=int(atlas_slice_idx)
        )

    r_um, r_floor, r_ceil = _compute_signed_r_um(
        t_lookup=t_lookup,
        r_signed=r_signed,
        n_bins=int(R_BIN_COUNT),
        roll_half_window=int(R_ROLL_HALF_WINDOW),
    )
    outside_support_mask = (t_lookup < float(t_support_lo)) | (t_lookup > float(t_support_hi))
    n_outside_support = int(np.count_nonzero(outside_support_mask))
    outside_support_frac = float(n_outside_support / max(1, int(t_lookup.size)))

    if mapping_mode_tag == "lut":
        ijk, source_slice_used = _lookup_ijk_batch_slice_locked(
            lut=LUTS[axis],
            atlas_slice_idx=int(atlas_slice_idx),
            t_lookup=t_lookup,
            r_um=r_um,
        )
    else:
        ijk, source_slice_used = _lookup_ijk_batch_midline_normal_slice_locked(
            axis=axis,
            columns_by_slice=MIDLINE_COLUMNS[axis],
            atlas_slice_idx=int(atlas_slice_idx),
            t_lookup=t_lookup,
            r_um=r_um,
        )

    np.savez_compressed(
        out_npz,
        h5ad_path=str(in_h5ad),
        axis=axis,
        atlas_slice_idx=np.int32(atlas_slice_idx),
        ijk_mapping_mode=str(mapping_mode_tag),
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
        t_neomeso=t_neomeso.astype(np.float32),
        n_total=np.int32(adata.n_obs),
        r_signed=r_signed.astype(np.float32),
        r_floor=r_floor.astype(np.float32),
        r_ceil=r_ceil.astype(np.float32),
        r_um=r_um.astype(np.float32),
        ijk=ijk.astype(np.float32),
    )

    t_all_min, t_all_max = _finite_minmax(t_all)
    t_local_min, t_local_max = _finite_minmax(t_local)
    t_neomeso_min, t_neomeso_max = _finite_minmax(t_neomeso)
    summary = {
        "name": in_h5ad.stem,
        "dataset_name": str(dataset_name),
        "axis": axis,
        "atlas_slice_idx": int(atlas_slice_idx),
        "source_slice_used": int(source_slice_used),
        "n_outside_support": int(n_outside_support),
        "outside_support_frac": float(outside_support_frac),
        "support_slice_used": int(nearest_slice_for_support),
        "t_support_lo": float(t_support_lo),
        "t_support_hi": float(t_support_hi),
        "lookup_method": f"slice_locked_{mapping_mode_tag}",
        "ijk_mapping_mode": str(mapping_mode_tag),
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
        "t_neomeso_min": t_neomeso_min,
        "t_neomeso_max": t_neomeso_max,
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
        f"t_neomeso=[{summary['t_neomeso_min']:.4f},{summary['t_neomeso_max']:.4f}]",
        f"r_um=[{summary['r_um_min']:.4f},{summary['r_um_max']:.4f}]",
        f"outside_support={summary['n_outside_support']} ({summary['outside_support_frac']:.3%})",
        f"mapping={mapping_mode_tag}",
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

_plot_cells_ijk_per_roi_axes(
    plot_payload=plot_payload,
    out_path=OUTDIR / "cells_ijk_per_roi_axes.png",
)

_plot_review_and_native_per_roi_axes_4col(
    results_in=results,
    lut_outdir=LUT_OUTDIR,
    out_path=OUTDIR / "cells_review_native_per_roi_axes_4col.png",
    rng_in=rng,
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
        label = f"{_info_display_name(info)} ({info['axis']}, slice={info['atlas_slice_idx']})"
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
