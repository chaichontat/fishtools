from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


COORDS_SCRIPT = Path("ccf/refextract/midsurface_coords.py")


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


def _resample_polyline_um(
    *,
    y: np.ndarray,
    x: np.ndarray,
    res_y_um: float,
    res_x_um: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample (y,x) polyline to n points using physical arc length in um.

    Returns (y_new, x_new) in voxel coordinates.
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
    return y_new, x_new


def _center_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {pts.shape}")
    center = np.nanmedian(pts, axis=0)
    if not np.isfinite(center).all():
        center = np.nanmean(pts, axis=0)
    return pts - center[None, :]


def _coronal_yx_to_ijk_um(
    *,
    slice_i: int,
    y_vox: np.ndarray,
    x_vox: np.ndarray,
    res_ijk_um: tuple[float, float, float],
) -> np.ndarray:
    yv = np.asarray(y_vox, dtype=np.float64).reshape(-1)
    xv = np.asarray(x_vox, dtype=np.float64).reshape(-1)
    if yv.shape != xv.shape:
        raise ValueError(f"y_vox/x_vox shape mismatch: {yv.shape} vs {xv.shape}")
    res_i_um, res_j_um, res_k_um = (float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2]))
    i_um = float(slice_i) * float(res_i_um)
    return np.column_stack(
        [
            np.full((yv.size,), i_um, dtype=np.float64),
            yv * float(res_j_um),
            xv * float(res_k_um),
        ]
    ).astype(np.float64, copy=False)


def _sagittal_yx_to_ijk_um(
    *,
    slice_k: int,
    y_vox: np.ndarray,
    x_vox: np.ndarray,
    res_ijk_um: tuple[float, float, float],
) -> np.ndarray:
    yv = np.asarray(y_vox, dtype=np.float64).reshape(-1)
    xv = np.asarray(x_vox, dtype=np.float64).reshape(-1)
    if yv.shape != xv.shape:
        raise ValueError(f"y_vox/x_vox shape mismatch: {yv.shape} vs {xv.shape}")
    res_i_um, res_j_um, res_k_um = (float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2]))
    k_um = float(slice_k) * float(res_k_um)
    return np.column_stack(
        [
            yv * float(res_i_um),
            xv * float(res_j_um),
            np.full((yv.size,), k_um, dtype=np.float64),
        ]
    ).astype(np.float64, copy=False)


@dataclass(frozen=True)
class PairQC:
    slice_a: int
    slice_b: int
    delta_ap_um_raw: float
    p50_um: float
    p95_um: float
    p95_over_p50: float
    band: int


def _dtw_banded_path(a: np.ndarray, b: np.ndarray, *, band: int) -> list[tuple[int, int]]:
    """Return a DTW path from (0,0) to (N-1,N-1) within a Sakoe-Chiba band.

    a and b are (N,2) arrays.
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"Expected same shape for a and b, got {aa.shape} vs {bb.shape}")
    n = int(aa.shape[0])
    if n < 2:
        return []
    band = int(max(0, band))
    if band < abs(n - n):
        band = abs(n - n)

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

    # Fill gaps by interpolation on index space.
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


def _sliding_median(values: np.ndarray, *, radius: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if radius <= 0 or x.size < 3:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(int(x.size)):
        lo = max(0, i - int(radius))
        hi = min(int(x.size), i + int(radius) + 1)
        out[i] = float(np.nanmedian(x[lo:hi]))
    return out


def optimize_ap_axis_from_strips(
    *,
    outdir: Path,
    n_t: int,
    band_frac: float,
    smooth_window: int,
    res_ijk_um: tuple[float, float, float] | None,
    max_pair_p95_over_p50: float,
    slice_i_min: int | None = None,
    slice_i_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PairQC]]:
    mod = _load_midsurface_coords_module()
    coronal = mod.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    if not coronal:
        raise ValueError(f"No coronal midline columns found under {outdir}")

    res_i_um, res_j_um, res_k_um = _load_resolution_ds_ijk_um(outdir, res_ijk_um=res_ijk_um)
    res_tuple = (float(res_i_um), float(res_j_um), float(res_k_um))
    slice_keys_all = np.asarray(sorted(int(k) for k in coronal.keys()), dtype=np.int32)
    lo = None if slice_i_min is None else int(slice_i_min)
    hi = None if slice_i_max is None else int(slice_i_max)
    if lo is not None:
        slice_keys_all = slice_keys_all[slice_keys_all >= lo]
    if hi is not None:
        slice_keys_all = slice_keys_all[slice_keys_all <= hi]
    slice_keys = slice_keys_all
    if slice_keys.size < 2:
        raise ValueError("Need at least 2 slices to optimize AP axis")

    n_t = int(n_t)
    band = int(max(4, int(round(float(band_frac) * float(n_t)))))
    if smooth_window % 2 == 0:
        raise ValueError(f"--smooth-window must be odd, got {smooth_window}")
    smooth_radius = int(smooth_window // 2)

    # Precompute per-slice resampled 3D points in um and also centered 2D in-plane points for matching.
    pts3_um: dict[int, np.ndarray] = {}
    pts2_inplane_centered: dict[int, np.ndarray] = {}

    prev_pts2: np.ndarray | None = None
    for key in slice_keys.tolist():
        c = coronal[int(key)]
        y_rs, x_rs = _resample_polyline_um(y=c.y, x=c.x, res_y_um=float(res_j_um), res_x_um=float(res_k_um), n=n_t)
        # In-plane points in um (j,k plane).
        pts2 = np.column_stack([y_rs * float(res_j_um), x_rs * float(res_k_um)]).astype(np.float64, copy=False)

        # Enforce consistent direction by minimizing mismatch to previous slice (after centering).
        pts2_center = _center_2d(pts2)
        if prev_pts2 is not None:
            fwd = float(np.nanmean(np.sum((pts2_center - prev_pts2) ** 2, axis=1)))
            rev_pts2_center = _center_2d(pts2[::-1])
            rev = float(np.nanmean(np.sum((rev_pts2_center - prev_pts2) ** 2, axis=1)))
            if rev < fwd:
                y_rs = y_rs[::-1]
                x_rs = x_rs[::-1]
                pts2_center = rev_pts2_center
        prev_pts2 = pts2_center

        pts2_inplane_centered[int(key)] = pts2_center
        # 3D points in um (i,j,k) with i fixed for the slice.
        pts3 = _coronal_yx_to_ijk_um(slice_i=int(key), y_vox=y_rs, x_vox=x_rs, res_ijk_um=res_tuple)
        pts3_um[int(key)] = pts3

    deltas_raw = np.full((slice_keys.size - 1,), np.nan, dtype=np.float64)
    qcs: list[PairQC] = []

    for idx in range(int(slice_keys.size) - 1):
        a_key = int(slice_keys[idx])
        b_key = int(slice_keys[idx + 1])
        a2 = pts2_inplane_centered[a_key]
        b2 = pts2_inplane_centered[b_key]

        # DTW on centered in-plane coords; widen band if needed.
        band_try = int(band)
        for _ in range(3):
            try:
                path = _dtw_banded_path(a2, b2, band=band_try)
                break
            except ValueError:
                band_try = int(min(n_t - 1, int(round(band_try * 1.75)) + 1))
        else:
            raise ValueError(f"DTW failed for pair ({a_key},{b_key}) even after widening band.")

        j_map = _path_to_monotone_index_map(path, n=n_t)
        a3 = pts3_um[a_key]
        b3 = pts3_um[b_key]
        d = np.linalg.norm(a3 - b3[j_map], axis=1).astype(np.float64, copy=False)

        p50 = float(np.nanmedian(d))
        p95 = float(np.nanpercentile(d, 95))
        delta = p50
        ratio = float(p95 / p50) if (np.isfinite(p50) and p50 > 0.0 and np.isfinite(p95)) else float("nan")
        if np.isfinite(ratio) and ratio > float(max_pair_p95_over_p50):
            # Still record QC, but drop delta from optimization (gap will be filled by smoothing).
            deltas_raw[idx] = np.nan
        else:
            deltas_raw[idx] = delta

        qcs.append(
            PairQC(
                slice_a=a_key,
                slice_b=b_key,
                delta_ap_um_raw=float(delta),
                p50_um=float(p50),
                p95_um=float(p95),
                p95_over_p50=float(ratio),
                band=int(band_try),
            )
        )

    # Fill missing deltas by interpolation over index-in-list.
    deltas_filled = deltas_raw.copy()
    missing = ~np.isfinite(deltas_filled)
    if np.any(missing):
        idx = np.arange(deltas_filled.size, dtype=np.float64)
        ok = np.isfinite(deltas_filled)
        if int(np.count_nonzero(ok)) < 2:
            raise ValueError("Too many invalid adjacent deltas; cannot build ap axis.")
        deltas_filled[missing] = np.interp(idx[missing], idx[ok], deltas_filled[ok])

    deltas_smooth = _sliding_median(deltas_filled, radius=int(smooth_radius))
    if not np.isfinite(deltas_smooth).all():
        raise ValueError("Non-finite values after smoothing deltas")
    if np.any(deltas_smooth <= 0.0):
        raise ValueError("Non-positive delta after smoothing; check inputs/QC thresholds.")

    ap_um = np.zeros((slice_keys.size,), dtype=np.float64)
    ap_um[1:] = np.cumsum(deltas_smooth, dtype=np.float64)
    return slice_keys, ap_um, deltas_smooth, qcs


def _write_plots(
    *,
    out_png_prefix: Path,
    slice_keys: np.ndarray,
    ap_um: np.ndarray,
    deltas_um: np.ndarray,
    qcs: list[PairQC],
) -> list[Path]:
    import matplotlib.pyplot as plt

    out: list[Path] = []
    slice_keys = np.asarray(slice_keys, dtype=np.int32).reshape(-1)
    ap_um = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    deltas_um = np.asarray(deltas_um, dtype=np.float64).reshape(-1)

    if slice_keys.size != ap_um.size:
        raise ValueError("slice_keys/ap_um shape mismatch for plotting")
    if slice_keys.size >= 2 and deltas_um.size != (slice_keys.size - 1):
        raise ValueError("deltas_um length mismatch for plotting")

    # 1) AP coordinate vs slice.
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=160)
    ax.plot(slice_keys, ap_um, linewidth=1.5)
    ax.set_xlabel("coronal slice_i (vox)")
    ax.set_ylabel("ap_um (um)")
    ax.set_title("Optimized AP coordinate (cumsum of median strip displacement)")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    p1 = out_png_prefix.with_name(out_png_prefix.name + "_ap_um_vs_slice.png")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    out.append(p1)

    # 2) Delta AP per adjacent slice.
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=160)
    if slice_keys.size >= 2:
        x = 0.5 * (slice_keys[:-1].astype(np.float64) + slice_keys[1:].astype(np.float64))
        ax.plot(x, deltas_um, linewidth=1.2)
    ax.set_xlabel("adjacent slice midpoint (vox)")
    ax.set_ylabel("delta_ap_um (um)")
    ax.set_title("Per-step AP spacing (smoothed)")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    p2 = out_png_prefix.with_name(out_png_prefix.name + "_delta_ap_um.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    out.append(p2)

    # 3) QC: p50/p95 and ratio per pair.
    if qcs:
        a = np.asarray([qc.slice_a for qc in qcs], dtype=np.int32)
        b = np.asarray([qc.slice_b for qc in qcs], dtype=np.int32)
        x = 0.5 * (a.astype(np.float64) + b.astype(np.float64))
        p50 = np.asarray([qc.p50_um for qc in qcs], dtype=np.float64)
        p95 = np.asarray([qc.p95_um for qc in qcs], dtype=np.float64)
        ratio = np.asarray([qc.p95_over_p50 for qc in qcs], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=160)
        ax.plot(x, p50, label="p50 dist (um)", linewidth=1.2)
        ax.plot(x, p95, label="p95 dist (um)", linewidth=1.2)
        ax.set_xlabel("adjacent slice midpoint (vox)")
        ax.set_ylabel("distance (um)")
        ax.set_title("Adjacent-pair strip displacement")
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.legend(frameon=False, fontsize=8)
        p3 = out_png_prefix.with_name(out_png_prefix.name + "_pair_dist_p50_p95.png")
        fig.tight_layout()
        fig.savefig(p3)
        plt.close(fig)
        out.append(p3)

        fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=160)
        ax.plot(x, ratio, linewidth=1.2)
        ax.set_xlabel("adjacent slice midpoint (vox)")
        ax.set_ylabel("p95/p50")
        ax.set_title("Adjacent-pair distortion proxy (higher = worse)")
        ax.grid(True, linewidth=0.5, alpha=0.35)
        p4 = out_png_prefix.with_name(out_png_prefix.name + "_pair_ratio_p95_over_p50.png")
        fig.tight_layout()
        fig.savefig(p4)
        plt.close(fig)
        out.append(p4)

    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Optimize a 1D AP axis (microns) for coronal slices by matching adjacent midline curves ('strips') "
            "and setting per-step AP spacing to the median 3D displacement of corresponding points."
        )
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Midsurface output directory containing coronal_midline_columns.csv.",
    )
    p.add_argument("--n-t", type=int, default=257, help="Number of samples along each slice midline curve.")
    p.add_argument(
        "--band-frac",
        type=float,
        default=0.15,
        help="DTW Sakoe-Chiba band as a fraction of n-t (higher is slower but more flexible).",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd sliding window (in slice steps) for median-smoothing delta AP spacings.",
    )
    p.add_argument(
        "--max-pair-p95-over-p50",
        type=float,
        default=2.5,
        help="Drop adjacent pairs whose (p95 distance / p50 distance) exceeds this threshold (filled by smoothing).",
    )
    p.add_argument("--slice-i-min", type=int, default=None, help="Minimum coronal slice_i to include (inclusive).")
    p.add_argument("--slice-i-max", type=int, default=None, help="Maximum coronal slice_i to include (inclusive).")
    p.add_argument(
        "--res-ijk-um",
        type=float,
        nargs=3,
        default=None,
        metavar=("RI", "RJ", "RK"),
        help="Voxel size in um for (i,j,k) in the DS grid; overrides outdir/resolution_ds_ijk_um.npy.",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Output npz path (default: <outdir>/ap_axis_um_from_strips.npz).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output csv path (default: <outdir>/ap_axis_um_from_strips_qc.csv).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write summary PNG plots to outdir (ap_um, delta, adjacent-pair QC).",
    )
    args = p.parse_args()

    outdir = args.outdir
    out_npz = args.out_npz if args.out_npz is not None else (outdir / "ap_axis_um_from_strips.npz")
    out_csv = args.out_csv if args.out_csv is not None else (outdir / "ap_axis_um_from_strips_qc.csv")
    res_ijk_um = None if args.res_ijk_um is None else (float(args.res_ijk_um[0]), float(args.res_ijk_um[1]), float(args.res_ijk_um[2]))

    slice_keys, ap_um, deltas_um, qcs = optimize_ap_axis_from_strips(
        outdir=outdir,
        n_t=int(args.n_t),
        band_frac=float(args.band_frac),
        smooth_window=int(args.smooth_window),
        res_ijk_um=res_ijk_um,
        max_pair_p95_over_p50=float(args.max_pair_p95_over_p50),
        slice_i_min=args.slice_i_min,
        slice_i_max=args.slice_i_max,
    )

    np.savez_compressed(
        out_npz,
        axis=np.asarray(["coronal"]),
        slice_keys=slice_keys.astype(np.int32, copy=False),
        ap_um=ap_um.astype(np.float32, copy=False),
        delta_ap_um=deltas_um.astype(np.float32, copy=False),
        n_t=np.asarray([int(args.n_t)], dtype=np.int32),
        band_frac=np.asarray([float(args.band_frac)], dtype=np.float32),
        smooth_window=np.asarray([int(args.smooth_window)], dtype=np.int32),
    )

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("slice_a,slice_b,delta_ap_um_raw,p50_um,p95_um,p95_over_p50,dtw_band\n")
        for qc in qcs:
            f.write(
                f"{qc.slice_a},{qc.slice_b},{qc.delta_ap_um_raw:.6f},{qc.p50_um:.6f},"
                f"{qc.p95_um:.6f},{qc.p95_over_p50:.6f},{qc.band}\n"
            )

    print(f"Wrote: {out_npz}")
    print(f"Wrote: {out_csv}")
    print(f"AP axis spans {float(ap_um[-1]):.1f} um across {slice_keys.size} slices (coronal).")
    if args.plot:
        prefix = outdir / "ap_axis_um_from_strips"
        pngs = _write_plots(
            out_png_prefix=prefix,
            slice_keys=slice_keys,
            ap_um=ap_um,
            deltas_um=deltas_um,
            qcs=qcs,
        )
        for pth in pngs:
            print(f"Wrote: {pth}")


if __name__ == "__main__":
    main()
