from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CellsTR:
    cell_id: np.ndarray
    t: np.ndarray
    r: np.ndarray
    theta: np.ndarray


def read_cells_tr_tsv(path: Path) -> CellsTR:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("cells_tr.tsv is missing a header row")
        required = {"cell_id", "t", "r", "theta"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"cells_tr.tsv missing columns: {sorted(missing)}")

        cell_id: list[str] = []
        t: list[float] = []
        r: list[float] = []
        theta: list[float] = []
        for row in reader:
            cell_id.append(row["cell_id"])
            t.append(float(row["t"]))
            r.append(float(row["r"]))
            theta.append(float(row["theta"]))

    return CellsTR(
        cell_id=np.asarray(cell_id, dtype=object),
        t=np.asarray(t, dtype=np.float64),
        r=np.asarray(r, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
    )


def weighted_quantile_1d(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    if x.size == 0:
        return float("nan")
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    s = float(np.sum(ws))
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    cdf = np.cumsum(ws) / s
    return float(np.interp(q, cdf, xs))


def weighted_kmeans_2d(points: np.ndarray, w: np.ndarray, k: int = 2, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be (n,2)")
    if k != 2:
        raise ValueError("only k=2 supported")
    n = points.shape[0]
    if n < 2:
        raise ValueError("need at least 2 points")

    # Init: farthest pair (deterministic).
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    i0, i1 = np.unravel_index(np.argmax(d2), d2.shape)
    cent = np.stack([points[i0], points[i1]], axis=0)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist0 = np.sum((points - cent[0]) ** 2, axis=1)
        dist1 = np.sum((points - cent[1]) ** 2, axis=1)
        new_labels = (dist1 < dist0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for kk in range(2):
            m = labels == kk
            if not np.any(m):
                continue
            ww = w[m]
            s = float(np.sum(ww))
            if s <= 0 or not np.isfinite(s):
                continue
            cent[kk] = np.sum(points[m] * ww[:, None], axis=0) / s

    return labels, cent


def best_single_cut_boundary(r_centers: np.ndarray, labels: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """
    Enforce a single transition along r. Returns (boundary_r, misclass_rate).
    """
    if r_centers.size < 4:
        return float("nan"), float("nan")

    # Assume lower-r region belongs to label of the first bin.
    low = int(labels[0])
    high = 1 - low

    # Precompute cumulative weighted mismatches for all cut positions.
    wsum = float(np.sum(w))
    if wsum <= 0:
        return float("nan"), float("nan")

    mismatch_low = (labels != low).astype(np.float64) * w
    mismatch_high = (labels != high).astype(np.float64) * w

    # cut at c means bins <= c are low, bins > c are high
    c_mis_low = np.cumsum(mismatch_low)
    c_mis_high = np.cumsum(mismatch_high[::-1])[::-1]

    costs = c_mis_low[:-1] + c_mis_high[1:]
    c = int(np.argmin(costs))

    boundary = float(0.5 * (r_centers[c] + r_centers[c + 1]))
    mis = float(costs[c] / wsum)
    return boundary, mis


def segment_boundaries_vs_t(
    t: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    r_min: float,
    r_max: float,
    t_grid_n: int,
    t_bw: float,
    r_bins: int,
    min_cells: int,
    q_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (t_grid, b0, b1, b2). b0/b2 are outer envelopes (quantiles) in [r_min,r_max].
    """
    t_grid = np.linspace(t_min, t_max, t_grid_n, dtype=np.float64)

    b0 = np.full_like(t_grid, np.nan, dtype=np.float64)
    b1 = np.full_like(t_grid, np.nan, dtype=np.float64)
    b2 = np.full_like(t_grid, np.nan, dtype=np.float64)

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    r_edges = np.linspace(r_min, r_max, r_bins + 1, dtype=np.float64)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for j, t0 in enumerate(t_grid):
        w_t = np.exp(-0.5 * ((t - t0) / t_bw) ** 2)
        keep = w_t > np.exp(-0.5 * 3.0**2)
        if int(np.sum(keep)) < min_cells:
            continue

        rt = r[keep]
        wt = w_t[keep]
        if rt.size < min_cells:
            continue

        # Outer envelopes (robust VZ/SVZ slab boundaries).
        b0[j] = weighted_quantile_1d(rt, wt, q_outer)
        b2[j] = weighted_quantile_1d(rt, wt, 1.0 - q_outer)

        # Compute theta contrast vector per r-bin.
        bin_id = np.clip(np.digitize(rt, r_edges) - 1, 0, r_bins - 1)
        wsum = np.bincount(bin_id, weights=wt, minlength=r_bins).astype(np.float64)
        csum = np.bincount(bin_id, weights=wt * cos_th[keep], minlength=r_bins).astype(np.float64)
        ssum = np.bincount(bin_id, weights=wt * sin_th[keep], minlength=r_bins).astype(np.float64)

        ok = wsum >= float(min_cells) / 10.0
        if int(np.sum(ok)) < 6:
            continue

        v = np.column_stack([csum[ok] / wsum[ok], ssum[ok] / wsum[ok]]).astype(np.float64)
        wv = wsum[ok]
        rc = r_centers[ok]

        # If the contrast is extremely weak everywhere, skip.
        if float(np.nanmax(np.linalg.norm(v, axis=1))) < 0.02:
            continue

        labels, _cent = weighted_kmeans_2d(v, wv, k=2)

        # Enforce single cut along r.
        order = np.argsort(rc)
        boundary, mis = best_single_cut_boundary(rc[order], labels[order], wv[order])

        # Reject if kmeans labels are too mixed to be a single cut.
        if not np.isfinite(boundary) or mis > 0.35:
            continue

        b1[j] = boundary

    return t_grid, b0, b1, b2


def smooth_nan_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win < 1 or win % 2 != 1:
        raise ValueError("win must be odd and >= 1")
    half = win // 2
    out = x.copy()
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        w = np.isfinite(x[lo:hi])
        if not np.any(w):
            out[i] = np.nan
        else:
            out[i] = float(np.mean(x[lo:hi][w]))
    return out


def smooth_nan_moving_average_npass(x: np.ndarray, win: int, passes: int) -> np.ndarray:
    if passes < 1:
        raise ValueError("passes must be >= 1")
    y = x
    for _ in range(passes):
        y = smooth_nan_moving_average(y, win)
    return y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cells-tr", type=Path, required=True, help="TSV with columns: cell_id, t, r, theta")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--t-min", type=float, default=0.1)
    p.add_argument("--t-max", type=float, default=0.75)
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--t-grid-n", type=int, default=220)
    p.add_argument("--t-bw", type=float, default=0.015)
    p.add_argument("--r-bins", type=int, default=90)
    p.add_argument("--min-cells", type=int, default=250)
    p.add_argument("--outer-q", type=float, default=0.02)
    p.add_argument("--smooth-win", type=int, default=11)
    p.add_argument("--smooth-passes", type=int, default=2)
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = read_cells_tr_tsv(args.cells_tr)
    m = (
        np.isfinite(cells.t)
        & np.isfinite(cells.r)
        & np.isfinite(cells.theta)
        & (cells.t >= args.t_min)
        & (cells.t <= args.t_max)
        & (cells.r >= args.r_min)
        & (cells.r <= args.r_max)
    )
    if int(np.sum(m)) < 1000:
        raise ValueError(f"too few cells after filtering: n={int(np.sum(m))}")

    t = cells.t[m]
    r = cells.r[m]
    theta = cells.theta[m] % (2.0 * np.pi)
    cell_id = cells.cell_id[m]

    t_grid, b0, b1, b2 = segment_boundaries_vs_t(
        t,
        r,
        theta,
        t_min=args.t_min,
        t_max=args.t_max,
        r_min=args.r_min,
        r_max=args.r_max,
        t_grid_n=args.t_grid_n,
        t_bw=args.t_bw,
        r_bins=args.r_bins,
        min_cells=args.min_cells,
        q_outer=args.outer_q,
    )

    # Smooth the boundary curves in t.
    b0_s = smooth_nan_moving_average_npass(b0, args.smooth_win, args.smooth_passes)
    b1_s = smooth_nan_moving_average_npass(b1, args.smooth_win, args.smooth_passes)
    b2_s = smooth_nan_moving_average_npass(b2, args.smooth_win, args.smooth_passes)

    # Enforce ordering.
    eps = 1e-6
    for j in range(t_grid.size):
        if not np.isfinite(b0_s[j]) or not np.isfinite(b2_s[j]):
            continue
        if not np.isfinite(b1_s[j]):
            # fallback: midpoint of outer envelopes
            b1_s[j] = 0.5 * (b0_s[j] + b2_s[j])
        b1_s[j] = float(np.clip(b1_s[j], b0_s[j] + eps, b2_s[j] - eps))

    # Write boundary curves.
    out_bounds = out_dir / "band_boundaries.tsv"
    with out_bounds.open("w") as f:
        f.write("t\tb0\tb1\tb2\tthick0\tthick1\n")
        for tj, a, b, c in zip(t_grid, b0_s, b1_s, b2_s, strict=True):
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
                f.write(f"{tj}\tNaN\tNaN\tNaN\tNaN\tNaN\n")
                continue
            f.write(f"{tj}\t{a}\t{b}\t{c}\t{b-a}\t{c-b}\n")

    # Annotate each cell (interpolate boundaries to each t).
    b0_i = np.interp(t, t_grid, np.nan_to_num(b0_s, nan=np.nanmedian(b0_s)))
    b1_i = np.interp(t, t_grid, np.nan_to_num(b1_s, nan=np.nanmedian(b1_s)))
    b2_i = np.interp(t, t_grid, np.nan_to_num(b2_s, nan=np.nanmedian(b2_s)))

    # Ensure monotone per-cell too.
    b1_i = np.clip(b1_i, b0_i + eps, b2_i - eps)

    layer = (r >= b1_i).astype(np.int64)  # 0: lower, 1: upper
    lo = np.where(layer == 0, b0_i, b1_i)
    hi = np.where(layer == 0, b1_i, b2_i)
    u = (r - lo) / (hi - lo)
    rho = layer + u  # continuous coordinate in [0,2)

    out_cells = out_dir / "cells_tr_two_bands.tsv"
    with out_cells.open("w") as f:
        f.write("cell_id\tt\tr\ttheta\tlayer\tu\trho\tb0\tb1\tb2\n")
        for row in zip(cell_id, t, r, theta, layer, u, rho, b0_i, b1_i, b2_i, strict=True):
            f.write("\t".join(map(str, row)) + "\n")

    # Plots (no extra deps).
    import matplotlib.pyplot as plt

    # Scatter with boundary overlay.
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(t, r, c=theta, s=3, alpha=0.25, cmap="hsv", linewidths=0)
    ok = np.isfinite(b0_s) & np.isfinite(b1_s) & np.isfinite(b2_s)
    plt.plot(t_grid[ok], b0_s[ok], color="black", lw=1.5, label="b0 (outer)")
    plt.plot(t_grid[ok], b1_s[ok], color="#d62728", lw=2.0, label="b1 (band split)")
    plt.plot(t_grid[ok], b2_s[ok], color="black", lw=1.5, label="b2 (outer)")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.title(
        "Two-band boundaries "
        f"(r in [{args.r_min},{args.r_max}], t in [{args.t_min},{args.t_max}]; "
        f"t_bw={args.t_bw}, smooth_win={args.smooth_win}x{args.smooth_passes})"
    )
    plt.colorbar(sc, label="theta (radians)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "boundaries_overlay.png", dpi=200)
    plt.close()

    # Thickness vs t.
    thick0 = b1_s - b0_s
    thick1 = b2_s - b1_s
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, thick0, color="grey", lw=2, label="thickness lower band")
    plt.plot(t_grid, thick1, color="#d62728", lw=2, label="thickness upper band")
    plt.xlabel("t")
    plt.ylabel("thickness (delta r)")
    plt.title("Band thickness vs t")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "thickness_vs_t.png", dpi=200)
    plt.close()

    # Normalized rho distribution (sanity check).
    plt.figure(figsize=(10, 6))
    plt.hist(rho, bins=80, color="grey", edgecolor="white")
    plt.xlabel("rho (layer + within-layer u)")
    plt.ylabel("count")
    plt.title("Normalized coordinate rho distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "rho_hist.png", dpi=200)
    plt.close()

    print(f"Wrote: {out_bounds}")
    print(f"Wrote: {out_cells}")
    print(f"Wrote: {out_dir / 'boundaries_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
