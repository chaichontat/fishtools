#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrix
from scipy.interpolate import RBFInterpolator, splprep, splev
from scipy.spatial import cKDTree

from fishtools.io.workspace import Workspace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute principal-curve coordinates for an .h5ad.\n\n"
            "Default: fit + project in Python.\n"
            "If --curvefit is provided, apply a precomputed CSV instead."
        )
    )
    p.add_argument(
        "workspace",
        nargs="?",
        help="Workspace root (uses Workspace API to resolve input/output under analysis/output/ccf-transforms).",
    )
    p.add_argument(
        "roi",
        nargs="?",
        help="ROI name (expects output/ccf-transforms/{ROI}/{ROI}.syn.annotated.h5ad).",
    )
    p.add_argument("--input", default=None, help="Path to input .h5ad")
    p.add_argument("--output", default=None, help="Path to output .h5ad")
    p.add_argument(
        "--curvefit",
        default=None,
        help="Optional: apply an existing curvefit CSV (cell_id,t,r,f1,f2) instead of computing one.",
    )
    p.add_argument(
        "--subset-obs-key",
        default=None,
        help="If set, subset AnnData to rows where obs[KEY] == --subset-obs-value before computing/applying.",
    )
    p.add_argument("--subset-obs-value", default=None, help="Used with --subset-obs-key.")
    p.add_argument(
        "--anchors-json",
        default=None,
        help=(
            "Path to anchors JSON produced by scripts/pick_curve_anchors.py "
            "(applies orientation/rescaling to t)."
        ),
    )
    p.add_argument(
        "--anchors-subroi",
        default=None,
        help=(
            "If --anchors-json is not set and multiple anchors JSON candidates exist next to the input, "
            "pick the one matching this subROI suffix (e.g. 'cortexbad' selects "
            "<stem>.cortexbad.anchors.json)."
        ),
    )
    p.add_argument(
        "--anchor-tps-smoothing",
        type=float,
        default=0.0,
        help="TPS warp smoothing (soft constraint) for multi-anchor curve warping (default: 0.0).",
    )
    p.add_argument(
        "--anchor-clamp",
        action="store_true",
        help="If set, clamp anchored t into [0,1] after rescaling.",
    )
    p.add_argument("--pc-fit-points", type=int, default=4000, help="Points used to fit principal curve (default: 4000).")
    p.add_argument("--pc-seed", type=int, default=1, help="RNG seed for principal curve subsampling (default: 1).")
    p.add_argument(
        "--pc-df",
        type=int,
        default=5,
        help="Degrees of freedom for the spline smoother inside the principal-curve fitter (default: 5).",
    )
    p.add_argument(
        "--pc-df-final",
        type=int,
        default=None,
        help="If set (and > --pc-df), linearly increase df across iterations up to this value.",
    )
    p.add_argument(
        "--pc-robust",
        action="store_true",
        help="Use robust reweighting (IRLS) during principal-curve smoothing (default: off).",
    )
    p.add_argument(
        "--pc-robust-quantile",
        type=float,
        default=0.90,
        help="Scale for robust weights: s = quantile(distance, q) (default: 0.90).",
    )
    p.add_argument(
        "--pc-robust-c",
        type=float,
        default=1.345,
        help="Huber cutoff (in units of d/s) when --pc-robust is set (default: 1.345).",
    )
    p.add_argument(
        "--pc-control-points",
        type=int,
        default=600,
        help="Max control points used to spline-interpolate the principal curve (default: 600).",
    )
    p.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Print extra intermediate artifacts next to --output (curvefit CSV path; anchored curvefit CSV when using anchors).",
    )
    return p.parse_args()


def read_curvefit(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit("curvefit CSV is missing a header row.")

        required = {"cell_id", "t", "r", "f1", "f2"}
        missing = required - set(r.fieldnames)
        if missing:
            raise SystemExit(f"curvefit CSV is missing columns: {', '.join(sorted(missing))}.")

        out: dict[str, dict[str, float]] = {}
        for row in r:
            cell_id = row["cell_id"]
            if cell_id in out:
                raise SystemExit(f"Duplicate cell_id in curvefit CSV: {cell_id}")
            out[cell_id] = {
                "t": float(row["t"]),
                "r": float(row["r"]),
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
            }
        return out


def load_anchors(path: Path) -> list[str]:
    data = json.loads(path.read_text())

    anchors = data.get("anchors")
    if isinstance(anchors, list) and anchors:
        ids: list[str] = []
        for a in anchors:
            if not isinstance(a, dict) or "cell_id" not in a:
                raise SystemExit(f"Invalid anchors JSON (anchors entries missing cell_id): {path}")
            cid = a["cell_id"]
            if not isinstance(cid, str) or cid == "":
                raise SystemExit(f"Invalid anchors JSON (empty cell_id): {path}")
            ids.append(cid)

        dedup: list[str] = []
        for cid in ids:
            if not dedup or dedup[-1] != cid:
                dedup.append(cid)
        if len(dedup) < 2:
            raise SystemExit(f"Need at least 2 distinct anchors in: {path}")
        return dedup

    start = data.get("start", {})
    end = data.get("end", {})
    start_id = start.get("cell_id")
    end_id = end.get("cell_id")
    if not isinstance(start_id, str) or not isinstance(end_id, str):
        raise SystemExit(f"Invalid anchors JSON (missing start/end cell_id): {path}")
    if start_id == "" or end_id == "":
        raise SystemExit(f"Invalid anchors JSON (empty cell_id): {path}")
    if start_id == end_id:
        raise SystemExit(f"Anchors JSON start and end are the same cell_id: {start_id}")
    return [start_id, end_id]


def apply_anchor_transform(
    *,
    t: np.ndarray,
    obs_names: np.ndarray,
    start_cell_id: str,
    end_cell_id: str,
    clamp: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    start_matches = np.where(obs_names == start_cell_id)[0]
    end_matches = np.where(obs_names == end_cell_id)[0]
    if start_matches.size != 1:
        raise SystemExit(f"Could not uniquely locate start cell_id in obs_names: {start_cell_id}")
    if end_matches.size != 1:
        raise SystemExit(f"Could not uniquely locate end cell_id in obs_names: {end_cell_id}")

    start_idx = int(start_matches[0])
    end_idx = int(end_matches[0])
    t0 = float(t[start_idx])
    t1 = float(t[end_idx])
    if not np.isfinite(t0) or not np.isfinite(t1):
        raise SystemExit("Anchor t values are not finite.")
    if t0 == t1:
        raise SystemExit("Start and end anchors map to the same t; choose anchors farther apart.")

    flipped = False
    t_out = t.astype(float, copy=True)
    if t0 > t1:
        flipped = True
        t_out = 1.0 - t_out
        t0 = 1.0 - t0
        t1 = 1.0 - t1

    denom = t1 - t0
    if denom == 0:
        raise SystemExit("Invalid anchor rescale (t_end == t_start after orientation).")
    t_out = (t_out - t0) / denom
    if clamp:
        t_out = np.clip(t_out, 0.0, 1.0)

    meta: dict[str, Any] = {
        "start_cell_id": start_cell_id,
        "end_cell_id": end_cell_id,
        "start_index": start_idx,
        "end_index": end_idx,
        "mode": "rescale",
        "clamp": bool(clamp),
        "flipped": bool(flipped),
        "t_start": float(t0),
        "t_end": float(t1),
    }
    return t_out, meta


def rcoord_scale_quantile(x: np.ndarray, q: float) -> np.ndarray:
    x = x.astype(float, copy=False)
    if not np.isfinite(q) or q <= 0:
        return np.full_like(x, 0.5, dtype=float)
    x_clip = np.clip(x / float(q), -1.0, 1.0)
    return 0.5 * (x_clip + 1.0)


def rcoord_scale_q(x: np.ndarray, *, quantile: float = 0.99) -> float:
    x = x.astype(float, copy=False)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 1.0
    q = float(np.quantile(np.abs(finite), float(quantile)))
    if not np.isfinite(q) or q <= 0:
        return 1.0
    return q


def fit_constrained_spline(
    path_xy: np.ndarray, *, u_include: np.ndarray | None = None, n_dense: int = 5000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] < 2:
        raise SystemExit(f"Not enough points to fit a spline (shape={path_xy.shape}).")
    path_xy = path_xy[:, :2]
    if path_xy.shape[0] >= 2:
        d = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        keep = np.concatenate([[True], d > 1e-12])
        path_xy = path_xy[keep]

    seg = np.sqrt(np.sum(np.diff(path_xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if float(s[-1]) == 0:
        u = np.linspace(0.0, 1.0, path_xy.shape[0])
    else:
        u = s / float(s[-1])

    k = min(3, path_xy.shape[0] - 1)
    if k < 1:
        raise SystemExit("Not enough points to fit a spline.")

    tck, _ = splprep([path_xy[:, 0], path_xy[:, 1]], u=u, s=0.0, k=k)

    u_dense = np.linspace(0.0, 1.0, int(n_dense))
    if u_include is not None:
        u_dense = np.concatenate([u_dense, np.asarray(u_include, dtype=float)])
    u_dense = np.unique(np.clip(u_dense, 0.0, 1.0))
    u_dense.sort()
    x_dense, y_dense = splev(u_dense, tck)
    dx_dense, dy_dense = splev(u_dense, tck, der=1)
    curve = np.column_stack([np.asarray(x_dense, float), np.asarray(y_dense, float)])
    deriv = np.column_stack([np.asarray(dx_dense, float), np.asarray(dy_dense, float)])
    return u_dense, curve, deriv, u


def run_princurve_py(
    *,
    xy: np.ndarray,
    max_iter: int = 30,
    rel_improve_tol: float = 1e-5,
    init_curve: np.ndarray | None = None,
    df: int = 5,
    df_final: int | None = None,
    robust: bool = False,
    robust_quantile: float = 0.90,
    robust_c: float = 1.345,
) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise SystemExit(f"Unexpected xy shape: {pts.shape}")
    pts = pts[:, :2]
    n = int(pts.shape[0])
    if n < 5:
        raise SystemExit("Need at least 5 points to fit a principal curve.")

    def project_to_polyline(
        *,
        pts2: np.ndarray,
        poly: np.ndarray,
        stretch: float = 2.0,
        k: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        poly = np.asarray(poly, dtype=float)[:, :2]
        if poly.shape[0] < 2:
            raise ValueError("polyline must have at least 2 points")

        if float(stretch) > 0 and poly.shape[0] >= 2:
            d0 = poly[1] - poly[0]
            d1 = poly[-1] - poly[-2]
            poly = np.vstack([poly[0] - float(stretch) * d0, poly, poly[-1] + float(stretch) * d1])

        seg_a = poly[:-1]
        seg_b = poly[1:]
        seg_v = seg_b - seg_a
        seg_len = np.linalg.norm(seg_v, axis=1)
        ok = seg_len > 1e-12
        seg_a = seg_a[ok]
        seg_b = seg_b[ok]
        seg_v = seg_v[ok]
        seg_len = seg_len[ok]
        if seg_len.size == 0:
            raise ValueError("polyline is degenerate")

        mid = 0.5 * (seg_a + seg_b)
        tree = cKDTree(mid)
        kk = int(min(max(1, k), seg_a.shape[0]))
        _, cand = tree.query(pts2, k=kk)
        if kk == 1:
            cand = cand[:, None]

        best_d2 = np.full((pts2.shape[0],), np.inf, dtype=float)
        best_proj = np.zeros((pts2.shape[0], 2), dtype=float)
        best_seg = np.zeros((pts2.shape[0],), dtype=int)
        best_tau = np.zeros((pts2.shape[0],), dtype=float)

        for j in range(kk):
            si = cand[:, j].astype(int)
            a = seg_a[si]
            v = seg_v[si]
            vv = np.sum(v * v, axis=1)
            w = pts2 - a
            tau = np.sum(w * v, axis=1) / vv
            tau = np.clip(tau, 0.0, 1.0)
            proj = a + tau[:, None] * v
            d2 = np.sum((pts2 - proj) ** 2, axis=1)
            better = d2 < best_d2
            best_d2[better] = d2[better]
            best_proj[better] = proj[better]
            best_seg[better] = si[better]
            best_tau[better] = tau[better]

        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        lam = cum[best_seg] + best_tau * seg_len[best_seg]
        return lam, best_proj, best_d2

    if df < 3:
        raise SystemExit("--pc-df must be >= 3.")
    if df_final is not None and df_final < df:
        raise SystemExit("--pc-df-final must be >= --pc-df.")
    if robust_quantile <= 0 or robust_quantile >= 1:
        raise SystemExit("--pc-robust-quantile must be between 0 and 1.")

    if init_curve is not None:
        curve = np.asarray(init_curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] < 2:
            raise SystemExit(f"Unexpected init_curve shape: {curve.shape}")
        curve = curve[:, :2]
        if curve.shape[0] < 2:
            raise SystemExit("init_curve must have at least 2 points.")
        d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        keep = np.concatenate([[True], d > 1e-12])
        curve = curve[keep]
        if curve.shape[0] < 2:
            raise SystemExit("init_curve is degenerate (all points identical).")
    else:
        mu = np.mean(pts, axis=0, dtype=float)
        centered = pts - mu
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0]
        scores = centered @ pc1
        s0 = float(np.min(scores))
        s1 = float(np.max(scores))
        if s0 == s1:
            raise SystemExit("Degenerate input for principal curve (all points identical along PC1).")

        curve = np.vstack([mu + s0 * pc1, mu + s1 * pc1]).astype(float)
    last_mse = np.inf

    def smooth_spline_fixed_df(
        x_in: np.ndarray,
        y_in: np.ndarray,
        *,
        df_local: int,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        X = np.asarray(dmatrix(f"cr(x, df={int(df_local)}) - 1", {"x": x_in}))
        y = np.asarray(y_in, dtype=float)
        if weights is None:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            return np.asarray(X @ coef, dtype=float)
        w = np.asarray(weights, dtype=float)
        if w.shape != (X.shape[0],):
            raise SystemExit("Internal error: robust weights have the wrong shape.")
        w_sqrt = np.sqrt(np.clip(w, 0.0, np.inf))
        Xw = X * w_sqrt[:, None]
        yw = y * w_sqrt
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return np.asarray(X @ coef, dtype=float)

    for it in range(int(max_iter)):
        lam, _, d2 = project_to_polyline(pts2=pts, poly=curve)
        order = np.argsort(lam, kind="mergesort")
        lam_sorted = lam[order].astype(float, copy=True)
        x_sorted = pts[order, 0]
        y_sorted = pts[order, 1]

        span = float(lam_sorted[-1] - lam_sorted[0]) if lam_sorted.size else 0.0
        eps = max((span if span > 0 else 1.0) / (n * 10), 1e-8)
        for i in range(1, lam_sorted.size):
            if lam_sorted[i] <= lam_sorted[i - 1]:
                lam_sorted[i] = lam_sorted[i - 1] + eps

        df0 = int(df)
        df1 = int(df_final) if df_final is not None else df0
        df1 = max(df0, df1)
        if max_iter <= 1:
            df_local = df0
        else:
            frac = float(it) / float(max_iter - 1)
            df_local = int(round(df0 + frac * (df1 - df0)))
        df_local = max(3, df_local)

        weights_sorted: np.ndarray | None = None
        if robust:
            d = np.sqrt(np.asarray(d2, dtype=float))[order]
            d_finite = d[np.isfinite(d)]
            s = float(np.quantile(d_finite, float(robust_quantile))) if d_finite.size else 0.0
            if not np.isfinite(s) or s <= 0:
                weights_sorted = None
            else:
                u = d / s
                c = float(robust_c)
                weights_sorted = np.where(u <= c, 1.0, c / u)
                weights_sorted = np.where(np.isfinite(weights_sorted), weights_sorted, 0.0)

        x_fit = smooth_spline_fixed_df(lam_sorted, x_sorted, df_local=df_local, weights=weights_sorted)
        y_fit = smooth_spline_fixed_df(lam_sorted, y_sorted, df_local=df_local, weights=weights_sorted)
        curve_new = np.column_stack([x_fit, y_fit]).astype(float)

        _, _, d2_2 = project_to_polyline(pts2=pts, poly=curve_new)
        mse = float(np.mean(np.asarray(d2_2, float)))

        if np.isfinite(last_mse) and last_mse > 0:
            rel = (last_mse - mse) / last_mse
            if rel >= 0 and rel < float(rel_improve_tol):
                curve = curve_new
                break

        curve = curve_new
        last_mse = mse

    if curve.shape[0] < 2:
        raise SystemExit("Principal curve fitting produced a degenerate curve.")

    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-12])
    curve = curve[keep]
    if curve.shape[0] < 2:
        raise SystemExit("Principal curve fitting produced a degenerate curve.")
    return curve


def tps_warp_curve_to_anchors(*, curve: np.ndarray, anchor_xy: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] < 2:
        raise ValueError(f"Unexpected curve shape: {curve.shape}")
    curve = curve[:, :2]

    anchor_xy = np.asarray(anchor_xy, dtype=float)
    if anchor_xy.ndim != 2 or anchor_xy.shape[1] < 2:
        raise ValueError(f"Unexpected anchor_xy shape: {anchor_xy.shape}")
    anchor_xy = anchor_xy[:, :2]
    if anchor_xy.shape[0] < 2:
        raise ValueError("Need at least 2 anchors to warp a curve.")
    if smoothing < 0:
        raise ValueError("TPS smoothing must be >= 0.")

    tree = cKDTree(curve)
    _, nn = tree.query(anchor_xy, k=1)
    nn = nn.astype(int)
    src = curve[nn]

    src_round = src.round(decimals=9)
    if np.unique(src_round, axis=0).shape[0] != src_round.shape[0]:
        raise SystemExit(
            "Multiple anchors project to the same curve location; increase --pc-control-points or move anchors."
        )

    rbf = RBFInterpolator(src, anchor_xy, kernel="thin_plate_spline", smoothing=float(smoothing))
    return np.asarray(rbf(curve), dtype=float)


def similarity_transform_from_two_points(
    *,
    src0: np.ndarray,
    src1: np.ndarray,
    dst0: np.ndarray,
    dst1: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    v_src = (src1 - src0).astype(float)
    v_dst = (dst1 - dst0).astype(float)
    ns = float(np.linalg.norm(v_src))
    nd = float(np.linalg.norm(v_dst))
    if ns == 0 or nd == 0:
        raise SystemExit("Degenerate anchor transform (identical src or dst points).")

    scale = nd / ns
    a = v_src / ns
    b = v_dst / nd
    cos = float(a[0] * b[0] + a[1] * b[1])
    sin = float(a[0] * b[1] - a[1] * b[0])
    R = np.array([[cos, -sin], [sin, cos]], dtype=float)
    t = dst0 - scale * (R @ src0)
    return scale, R, t


def apply_similarity_transform(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (points @ R.T)) + t


def project_to_polyline_arclength(
    *,
    xy: np.ndarray,
    line: np.ndarray,
    k: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(xy, dtype=float)[:, :2]
    poly = np.asarray(line, dtype=float)[:, :2]
    if poly.shape[0] < 2:
        raise SystemExit("Polyline must have at least 2 points.")

    seg_a = poly[:-1]
    seg_b = poly[1:]
    seg_v = seg_b - seg_a
    seg_len = np.linalg.norm(seg_v, axis=1)
    ok = seg_len > 1e-12
    seg_a = seg_a[ok]
    seg_v = seg_v[ok]
    seg_len = seg_len[ok]
    if seg_len.size == 0:
        raise SystemExit("Degenerate polyline (all segments have zero length).")

    mid = seg_a + 0.5 * seg_v
    tree = cKDTree(mid)
    kk = int(min(max(1, k), seg_a.shape[0]))
    _, cand = tree.query(pts, k=kk)
    if kk == 1:
        cand = cand[:, None]

    best_d2 = np.full((pts.shape[0],), np.inf, dtype=float)
    best_proj = np.zeros((pts.shape[0], 2), dtype=float)
    best_seg = np.zeros((pts.shape[0],), dtype=int)
    best_tau = np.zeros((pts.shape[0],), dtype=float)

    for j in range(kk):
        si = cand[:, j].astype(int)
        a = seg_a[si]
        v = seg_v[si]
        vv = np.sum(v * v, axis=1)
        w = pts - a
        tau = np.clip(np.sum(w * v, axis=1) / vv, 0.0, 1.0)
        proj = a + tau[:, None] * v
        d2 = np.sum((pts - proj) ** 2, axis=1)
        better = d2 < best_d2
        best_d2[better] = d2[better]
        best_proj[better] = proj[better]
        best_seg[better] = si[better]
        best_tau[better] = tau[better]

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    lam = cum[best_seg] + best_tau * seg_len[best_seg]
    total = float(cum[-1])
    t = lam / total if total > 0 else np.zeros_like(lam)

    tan = seg_v[best_seg]
    res = pts - best_proj
    cross = tan[:, 0] * res[:, 1] - tan[:, 1] * res[:, 0]
    sign = np.where(cross >= 0, 1.0, -1.0)
    r_signed = sign * np.sqrt(best_d2)
    return t.astype(float), r_signed.astype(float), best_proj.astype(float)


def curve_polyline_from_proj(
    *,
    t: np.ndarray,
    proj_xy: np.ndarray,
    start_proj: np.ndarray,
    end_proj: np.ndarray,
    n_bins: int = 200,
    min_count: int = 30,
) -> np.ndarray:
    t_all = np.asarray(t, dtype=float)
    proj_all = np.asarray(proj_xy, dtype=float)[:, :2]
    keep = np.isfinite(t_all) & np.isfinite(proj_all).all(axis=1)
    keep &= (t_all > 0) & (t_all < 1)
    t_keep = t_all[keep]
    proj_keep = proj_all[keep]
    if t_keep.size < int(min_count) * 3:
        return np.vstack([start_proj, end_proj])

    edges = np.quantile(t_keep, np.linspace(0, 1, int(n_bins) + 1))
    pts: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if not (hi > lo):
            continue
        in_bin = (t_keep >= lo) & (t_keep < hi)
        if int(in_bin.sum()) < int(min_count):
            continue
        pts.append(np.median(proj_keep[in_bin], axis=0))
    if not pts:
        return np.vstack([start_proj, end_proj])
    return np.vstack([start_proj, np.vstack(pts), end_proj])


def clip_polyline_to_interval(line: np.ndarray, t0: float, t1: float) -> np.ndarray:
    poly = np.asarray(line, dtype=float)[:, :2]
    if poly.shape[0] < 2:
        return poly

    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    ok = seg > 1e-12
    if not np.all(ok):
        poly = poly[np.concatenate([[True], ok])]
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    if poly.shape[0] < 2:
        return poly

    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0 or not np.isfinite(total):
        return poly[:2]

    a = float(np.clip(t0, 0.0, 1.0)) * total
    b = float(np.clip(t1, 0.0, 1.0)) * total
    if a > b:
        a, b = b, a

    def interp_at(lam: float) -> tuple[int, np.ndarray]:
        i = int(np.searchsorted(s, lam, side="right") - 1)
        i = int(np.clip(i, 0, poly.shape[0] - 2))
        denom = float(seg[i])
        if denom <= 0:
            return i, poly[i].copy()
        frac = (lam - float(s[i])) / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        return i, poly[i] + frac * (poly[i + 1] - poly[i])

    i0, p0 = interp_at(a)
    i1, p1 = interp_at(b)

    if i0 == i1:
        return np.vstack([p0, p1])

    middle = poly[i0 + 1 : i1 + 1]
    return np.vstack([p0, middle, p1])


def _plot_anchor_qc_on_ax(
    ax: matplotlib.axes.Axes,
    *,
    xy: np.ndarray,
    proj_xy: np.ndarray,
    line: np.ndarray,
    idx: np.ndarray,
    color_values: np.ndarray,
    cmap: str,
    norm: matplotlib.colors.Normalize | None,
    start_index: int,
    end_index: int,
    anchor_indices: list[int],
    title: str,
) -> None:
    ax.scatter(
        xy[idx, 0],
        xy[idx, 1],
        c=color_values[idx],
        s=1,
        alpha=0.8,
        cmap=str(cmap),
        norm=norm,
        linewidths=0,
    )
    if line.shape[0] >= 2:
        ax.plot(line[:, 0], line[:, 1], color="white", linewidth=3, alpha=0.95, zorder=4)
        ax.plot(line[:, 0], line[:, 1], color="black", linewidth=1.25, alpha=0.95, zorder=5)

    for j, aidx in enumerate(anchor_indices):
        if j == 0:
            anchor_color = "lime"
        elif j == len(anchor_indices) - 1:
            anchor_color = "red"
        else:
            anchor_color = "orange"

        x0, y0 = float(xy[aidx, 0]), float(xy[aidx, 1])
        x1, y1 = float(proj_xy[aidx, 0]), float(proj_xy[aidx, 1])
        ax.plot([x0, x1], [y0, y1], color="white", linestyle="--", linewidth=3, alpha=0.9, zorder=5)
        ax.plot([x0, x1], [y0, y1], color="black", linestyle="--", linewidth=1.25, alpha=0.9, zorder=6)
        ax.plot([x0, x1], [y0, y1], color=anchor_color, linestyle="--", linewidth=1.5, alpha=0.9, zorder=7)

        ax.scatter([x0, x1], [y0, y1], s=80, marker="x", c=anchor_color, linewidths=2, zorder=7)
        ax.scatter([x0], [y0], s=90, c=anchor_color, edgecolors="black", linewidths=0.5, zorder=6)
        ax.annotate(
            str(j + 1),
            (x0, y0),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            zorder=8,
        )

    ax.annotate(
        "start (t=0)",
        (xy[int(start_index), 0], xy[int(start_index), 1]),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.annotate(
        "end (t=1)",
        (xy[int(end_index), 0], xy[int(end_index), 1]),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def write_curve_qc_grid_png(
    *,
    out_path: Path,
    xy: np.ndarray,
    proj_xy: np.ndarray,
    t: np.ndarray,
    roi_values: np.ndarray,
    rois: list[str],
    max_points_total: int = 120_000,
    ncols_max: int = 4,
) -> None:
    xy = np.asarray(xy, dtype=float)
    proj_xy = np.asarray(proj_xy, dtype=float)
    t = np.asarray(t, dtype=float)
    roi_values = np.asarray(roi_values)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Unexpected xy shape: {xy.shape}")
    if proj_xy.ndim != 2 or proj_xy.shape[1] < 2:
        raise ValueError(f"Unexpected proj_xy shape: {proj_xy.shape}")
    if t.shape != (xy.shape[0],) or proj_xy.shape[0] != xy.shape[0] or roi_values.shape != (xy.shape[0],):
        raise ValueError("Mismatched lengths for xy/proj_xy/t/roi_values.")

    panels: list[tuple[str, np.ndarray]] = []
    for roi in rois:
        mask = roi_values == roi
        if int(mask.sum()) < 2:
            continue
        panels.append((str(roi), mask))

    if not panels:
        return

    n_panels = len(panels)
    ncols = min(int(ncols_max), int(np.ceil(np.sqrt(n_panels))))
    ncols = max(1, ncols)
    nrows = int(np.ceil(n_panels / ncols))

    max_points_panel = int(min(50_000, max(2_000, int(max_points_total) // n_panels)))
    rng = np.random.default_rng(0)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=False)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")

    for ax, (roi, mask) in zip(axs.flat, panels, strict=False):
        xy_roi = xy[mask]
        proj_roi = proj_xy[mask]
        t_roi = t[mask]

        keep = np.isfinite(t_roi) & np.isfinite(xy_roi[:, :2]).all(axis=1)
        idx = np.flatnonzero(keep)
        if idx.size < 2:
            continue
        if idx.size > max_points_panel:
            idx = rng.choice(idx, size=max_points_panel, replace=False)

        start_i = int(np.flatnonzero(keep)[np.argmin(t_roi[keep])])
        end_i = int(np.flatnonzero(keep)[np.argmax(t_roi[keep])])
        start_proj = proj_roi[start_i, :2].astype(float, copy=True)
        end_proj = proj_roi[end_i, :2].astype(float, copy=True)
        line = curve_polyline_from_proj(t=t_roi, proj_xy=proj_roi, start_proj=start_proj, end_proj=end_proj)

        ax.axis("on")
        _plot_anchor_qc_on_ax(
            ax,
            xy=xy_roi,
            proj_xy=proj_roi,
            line=line,
            idx=idx,
            color_values=t_roi,
            cmap="viridis",
            norm=norm,
            start_index=start_i,
            end_index=end_i,
            anchor_indices=[start_i, end_i],
            title=f"{roi} (n={int(mask.sum())})",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_anchor_qc_png(
    *,
    out_path: Path,
    xy: np.ndarray,
    proj_xy: np.ndarray,
    curve_xy: np.ndarray | None,
    t: np.ndarray,
    c: np.ndarray | None = None,
    cmap: str = "viridis",
    c_label: str = "principal t (anchored)",
    start_index: int,
    end_index: int,
    anchor_indices: list[int] | None,
    title: str,
    max_points: int = 50_000,
    vmin: float | None = None,
    vmax: float | None = None,
    plot_mask: np.ndarray | None = None,
) -> None:
    xy = np.asarray(xy, dtype=float)
    proj_xy = np.asarray(proj_xy, dtype=float)
    t = np.asarray(t, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Unexpected xy shape: {xy.shape}")
    if proj_xy.ndim != 2 or proj_xy.shape[1] < 2:
        raise ValueError(f"Unexpected proj_xy shape: {proj_xy.shape}")
    if t.shape[0] != xy.shape[0] or proj_xy.shape[0] != xy.shape[0]:
        raise ValueError("Mismatched lengths for xy/proj_xy/t.")
    if c is not None:
        c = np.asarray(c, dtype=float)
        if c.shape[0] != xy.shape[0]:
            raise ValueError("Mismatched lengths for xy and c.")

    n = xy.shape[0]
    rng = np.random.default_rng(0)
    if plot_mask is None:
        plot_mask = np.ones((n,), dtype=bool)
    else:
        plot_mask = np.asarray(plot_mask, dtype=bool)
        if plot_mask.shape != (n,):
            raise ValueError("plot_mask must be a 1D boolean mask matching xy length.")

    keep = plot_mask & np.isfinite(xy[:, :2]).all(axis=1)
    color_values = t if c is None else c
    keep &= np.isfinite(color_values)
    idx = np.flatnonzero(keep)
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    if anchor_indices is None:
        anchor_indices = [int(start_index), int(end_index)]
    else:
        anchor_indices = [int(i) for i in anchor_indices]
        if len(anchor_indices) < 2:
            raise ValueError("anchor_indices must contain at least 2 indices.")

    start_proj = proj_xy[int(start_index), :2].astype(float, copy=True)
    end_proj = proj_xy[int(end_index), :2].astype(float, copy=True)
    if curve_xy is not None:
        line = np.asarray(curve_xy, dtype=float)[:, :2]
    else:
        line = curve_polyline_from_proj(t=t, proj_xy=proj_xy, start_proj=start_proj, end_proj=end_proj)

    fig, ax = plt.subplots(figsize=(7, 6))
    norm: matplotlib.colors.Normalize | None = None
    if vmin is not None and vmax is not None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    _plot_anchor_qc_on_ax(
        ax,
        xy=xy,
        proj_xy=proj_xy,
        line=line,
        idx=idx,
        color_values=color_values,
        cmap=str(cmap),
        norm=norm,
        start_index=int(start_index),
        end_index=int(end_index),
        anchor_indices=anchor_indices,
        title=title,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def signed_distance_to_polyline(*, xy: np.ndarray, line: np.ndarray) -> np.ndarray:
    _, r_signed, _ = project_to_polyline_arclength(xy=xy, line=line)
    return r_signed


def resolve_workspace_input_path(ws: Workspace, roi: str) -> Path:
    roi_clean = str(roi).strip()
    roi_dir = ws.ccf_transforms(roi_clean)
    expected = roi_dir / f"{roi_clean}.syn.annotated.h5ad"
    if expected.exists():
        return expected

    if not roi_dir.exists():
        raise SystemExit(f"Input not found: {expected}")

    candidates = sorted(p for p in roi_dir.glob("*.syn.annotated.h5ad") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = "\n".join([f"  - {p.name}" for p in candidates])
        raise SystemExit(
            f"Input not found: {expected}\n"
            "Found multiple candidate inputs in ROI directory; please pass --input explicitly:\n"
            f"{names}"
        )
    raise SystemExit(f"Input not found: {expected}")


def infer_anchors_json(
    *,
    in_path: Path,
    anchors_json: str | None,
    anchors_subroi: str | None,
    auto_subroi: str | None = None,
) -> str | None:
    if anchors_json is not None:
        return anchors_json

    base = in_path.with_suffix("")
    candidates = sorted(in_path.parent.glob(f"{in_path.stem}*.anchors.json"))

    if anchors_subroi is not None or auto_subroi is not None:
        subroi = str(anchors_subroi) if anchors_subroi is not None else str(auto_subroi)
        preferred = Path(f"{base}.{subroi}.anchors.json")
        if preferred.exists():
            return str(preferred)

        matches = [p for p in candidates if p.name.endswith(f".{subroi}.anchors.json")]
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) == 0 and anchors_subroi is not None:
            names = "\n".join([f"  - {p}" for p in candidates]) if candidates else "  (none found)"
            raise SystemExit(f"--anchors-subroi={subroi!r} but no matching anchors JSON found. Candidates:\n{names}")
        if len(matches) > 1 and anchors_subroi is not None:
            names = "\n".join([f"  - {p}" for p in matches])
            raise SystemExit(f"--anchors-subroi={subroi!r} matched multiple anchors JSON files:\n{names}")

    if len(candidates) == 1:
        return str(candidates[0])

    preferred_default = Path(f"{base}.anchors.json")
    if preferred_default in candidates:
        return str(preferred_default)

    if candidates:
        names = "\n".join([f"  - {p}" for p in candidates])
        print(
            "Note: multiple anchors JSON files found next to the input.\n"
            "Pass --anchors-json (full path) or --anchors-subroi (e.g. 'cortexbad') to choose one:\n"
            f"{names}"
        )

    return None


def main() -> None:
    args = parse_args()

    if args.workspace is not None or args.roi is not None:
        if args.input is not None or args.output is not None:
            raise SystemExit("Use either WORKSPACE ROI positional args or --input/--output (not both).")
        if args.workspace is None or args.roi is None:
            raise SystemExit("Provide both WORKSPACE and ROI.")
        ws = Workspace(str(args.workspace).strip())
        roi = str(args.roi).strip()
        in_path = resolve_workspace_input_path(ws, roi)
        out_path = in_path.with_name(f"{in_path.stem}.princurve.h5ad")
    else:
        if args.input is None or args.output is None:
            raise SystemExit("Provide either --workspace/--roi or --input/--output.")
        in_path = Path(args.input)
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_prefix = out_path.with_suffix("")

    auto_subroi = None
    if args.subset_obs_key == "ccf_adjusted" and args.subset_obs_value is not None:
        auto_subroi = str(args.subset_obs_value)
    resolved_anchors_json = infer_anchors_json(
        in_path=in_path,
        anchors_json=args.anchors_json,
        anchors_subroi=args.anchors_subroi,
        auto_subroi=auto_subroi,
    )
    if args.anchors_json is None and resolved_anchors_json is not None:
        print(f"Using inferred anchors JSON: {resolved_anchors_json}")
    args.anchors_json = resolved_anchors_json

    adata = ad.read_h5ad(in_path)
    if (args.subset_obs_key is None) != (args.subset_obs_value is None):
        raise SystemExit("Use --subset-obs-key and --subset-obs-value together.")
    if args.subset_obs_key is not None:
        if args.subset_obs_key not in adata.obs.columns:
            raise SystemExit(f"obs column not found: {args.subset_obs_key}")
        mask = adata.obs[args.subset_obs_key].astype(str) == str(args.subset_obs_value)
        n_selected = int(mask.sum())
        if n_selected == 0:
            raise SystemExit(f"No rows match obs[{args.subset_obs_key!r}] == {args.subset_obs_value!r}.")
        adata = adata[mask].copy()
        print(f"Subsetting: kept {n_selected} cells where obs[{args.subset_obs_key}] == {args.subset_obs_value}")

    if "spatial" not in adata.obsm:
        raise SystemExit("Input .h5ad is missing obsm['spatial'].")
    xy = np.asarray(adata.obsm["spatial"])
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit("obsm['spatial'] must be a 2D array with >=2 columns.")
    if xy.shape[1] > 2:
        xy = xy[:, :2]
    xy = xy.astype(float, copy=False)

    obs_names = adata.obs_names.astype(str).to_numpy()
    if xy.shape[0] != obs_names.shape[0]:
        raise SystemExit(f"Mismatch: spatial rows={xy.shape[0]} obs_names={obs_names.shape[0]}.")

    curvefit_path: Path | None = Path(args.curvefit) if args.curvefit is not None else None
    roi_obs_key = "ccf_adjusted"
    rois_to_process: list[str] | None = None
    roi_values: np.ndarray | None = None
    if (
        args.curvefit is None
        and args.subset_obs_key is None
        and args.subset_obs_value is None
        and args.anchors_json is None
        and roi_obs_key in adata.obs.columns
    ):
        roi_series = adata.obs[roi_obs_key].astype(str)
        if int(roi_series.nunique(dropna=False)) > 1:
            roi_values = roi_series.to_numpy()
            roi_unique = sorted(set(roi_values.tolist()) - {"", "nan", "none", "NaN", "None"})
            rois_to_process = [str(v) for v in roi_unique]

    anchor_meta: dict[str, Any] | None = None
    curve_xy_for_qc: np.ndarray | None = None
    r_signed_store: np.ndarray | None = None
    r_scale_info: dict[str, Any] | None = None
    anchor_ids: list[str] | None = load_anchors(Path(args.anchors_json)) if args.anchors_json is not None else None
    anchor_indices: list[int] | None = None
    anchor_roi_value: str | None = None
    if anchor_ids is not None:
        anchor_indices = []
        for cid in anchor_ids:
            m = np.where(obs_names == cid)[0]
            if m.size != 1:
                raise SystemExit(f"Could not uniquely locate anchor cell_id in obs_names: {cid}")
            anchor_indices.append(int(m[0]))
        if roi_obs_key in adata.obs.columns:
            roi_arr = adata.obs[roi_obs_key].astype(str).to_numpy()
            uniq = sorted(set(roi_arr[np.asarray(anchor_indices, dtype=int)].tolist()) - {"", "nan", "none", "NaN", "None"})
            if len(uniq) != 1:
                raise SystemExit(
                    "Anchors must be within a single obs['ccf_adjusted'] value when multiple subROIs exist; "
                    f"got: {uniq}"
                )
            anchor_roi_value = str(uniq[0])

    if args.curvefit is None:
        if curvefit_path is not None:
            raise SystemExit("Internal error: curvefit_path is set while args.curvefit is None.")
        curvefit_path = Path(f"{out_prefix}.curvefit.csv")
        r_scale_abs_quantile = 0.99

        def compute_principal_curve_for_xy(xy_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = int(xy_in.shape[0])
            rng = np.random.default_rng(args.pc_seed)
            fit_n = min(int(args.pc_fit_points), n)
            fit_idx = rng.choice(np.arange(n), size=fit_n, replace=False) if fit_n < n else np.arange(n)
            xy_fit = xy_in[fit_idx]

            curve_pts = run_princurve_py(
                xy=xy_fit,
                df=int(args.pc_df),
                df_final=int(args.pc_df_final) if args.pc_df_final is not None else None,
                robust=bool(args.pc_robust),
                robust_quantile=float(args.pc_robust_quantile),
                robust_c=float(args.pc_robust_c),
            )
            if curve_pts.shape[0] > args.pc_control_points:
                pick = np.linspace(0, curve_pts.shape[0] - 1, args.pc_control_points).round().astype(int)
                curve_pts = curve_pts[pick]

            _, curve, _deriv, _ = fit_constrained_spline(curve_pts, n_dense=5_000)
            t, r_signed, proj = project_to_polyline_arclength(xy=xy_in, line=curve)
            q = rcoord_scale_q(r_signed, quantile=r_scale_abs_quantile)
            return t, rcoord_scale_quantile(r_signed, q), proj, r_signed, q

        if rois_to_process is not None:
            if args.anchors_json is not None:
                raise SystemExit("Cannot use --anchors-json when splitting by ccf_adjusted ROI.")
            if roi_values is None:
                raise SystemExit("Internal error: expected roi_values when splitting by ccf_adjusted ROI.")

            t = np.full((xy.shape[0],), np.nan, dtype=float)
            r_ = np.full((xy.shape[0],), np.nan, dtype=float)
            proj = np.full((xy.shape[0], 2), np.nan, dtype=float)
            r_signed_all = np.full((xy.shape[0],), np.nan, dtype=float)
            r_q_by_roi: dict[str, float] = {}

            for roi in rois_to_process:
                mask = roi_values == roi
                n_roi = int(mask.sum())
                if n_roi == 0:
                    continue
                if n_roi < 5:
                    print(f"Skipping principal curve: roi={roi} (n={n_roi} < 5)")
                    continue
                print(f"Principal curve: roi={roi} n={n_roi}")
                t_roi, r_roi, proj_roi, r_signed_roi, r_q = compute_principal_curve_for_xy(xy[mask])
                t[mask] = t_roi
                r_[mask] = r_roi
                proj[mask] = proj_roi
                r_signed_all[mask] = r_signed_roi
                r_q_by_roi[roi] = float(r_q)
            r_signed_store = r_signed_all
            r_scale_info = {"abs_quantile": float(r_scale_abs_quantile), "q_by_roi": r_q_by_roi}
        else:
            n_dense = 20_000 if args.anchors_json is not None else 5_000

            fit_mask = np.ones((xy.shape[0],), dtype=bool)
            if roi_obs_key in adata.obs.columns:
                roi = adata.obs[roi_obs_key].astype(str).to_numpy()
                if args.anchors_json is not None and anchor_roi_value is not None:
                    fit_mask = roi == anchor_roi_value
                else:
                    cortex_mask = roi == "cortex"
                    if np.any(cortex_mask):
                        fit_mask = cortex_mask

            used_idx = np.flatnonzero(fit_mask)
            if used_idx.size == 0:
                raise SystemExit("No cells selected for fitting (ccf_adjusted=='cortex' is empty).")

            xy_used = xy[used_idx]
            full_to_used = np.full((xy.shape[0],), -1, dtype=int)
            full_to_used[used_idx] = np.arange(used_idx.size, dtype=int)

            n = int(xy_used.shape[0])
            rng = np.random.default_rng(args.pc_seed)
            fit_n = min(int(args.pc_fit_points), n)
            init_curve: np.ndarray | None = None

            required_idx: np.ndarray | None = None
            if args.anchors_json is not None:
                if anchor_indices is None or anchor_ids is None:
                    raise SystemExit("Internal error: anchors are not available after parsing --anchors-json.")
                required_idx_full = np.asarray(anchor_indices, dtype=int)
                required_idx = full_to_used[required_idx_full]
                if np.any(required_idx < 0):
                    if anchor_roi_value is None:
                        raise SystemExit("Anchors must be within the cells used for fitting.")
                    raise SystemExit(f"Anchors must be within obs['ccf_adjusted']=={anchor_roi_value!r} for fitting.")
                fit_n = max(fit_n, max(5, int(required_idx.size)))
                init_curve = xy_used[required_idx, :2].astype(float, copy=False)

            if fit_n >= n:
                fit_idx = np.arange(n)
            elif required_idx is None:
                fit_idx = rng.choice(np.arange(n), size=fit_n, replace=False)
            else:
                required_mask = np.zeros((n,), dtype=bool)
                required_mask[required_idx] = True
                available = np.flatnonzero(~required_mask)
                extra_n = int(fit_n - required_idx.size)
                if extra_n <= 0:
                    fit_idx = required_idx
                else:
                    extra = (
                        rng.choice(available, size=extra_n, replace=False)
                        if extra_n < available.size
                        else available
                    )
                    fit_idx = np.concatenate([required_idx, extra])
            xy_fit = xy_used[fit_idx]

            curve_pts = run_princurve_py(
                xy=xy_fit,
                init_curve=init_curve,
                df=int(args.pc_df),
                df_final=int(args.pc_df_final) if args.pc_df_final is not None else None,
                robust=bool(args.pc_robust),
                robust_quantile=float(args.pc_robust_quantile),
                robust_c=float(args.pc_robust_c),
            )
            if curve_pts.shape[0] > args.pc_control_points:
                pick = np.linspace(0, curve_pts.shape[0] - 1, args.pc_control_points).round().astype(int)
                curve_pts = curve_pts[pick]

            _u_dense, curve, _deriv, _ = fit_constrained_spline(curve_pts, n_dense=n_dense)

            curve_for_proj = curve
            curve_xy_for_qc = curve

            if args.anchors_json is not None:
                if anchor_indices is None or anchor_ids is None:
                    raise SystemExit("Internal error: anchors are not available after parsing --anchors-json.")
                assert required_idx is not None
                anchor_xy = xy_used[np.asarray(required_idx, dtype=int), :2].astype(float, copy=False)

                if len(anchor_ids) == 2:
                    start_i, end_i = int(required_idx[0]), int(required_idx[-1])
                    tree0 = cKDTree(curve)
                    _, nn0 = tree0.query(xy_used, k=1)
                    nn0 = nn0.astype(int)
                    src0 = curve[nn0[start_i]].copy()
                    src1 = curve[nn0[end_i]].copy()
                    dst0 = xy_used[start_i].copy()
                    dst1 = xy_used[end_i].copy()
                    scale, R, tt = similarity_transform_from_two_points(src0=src0, src1=src1, dst0=dst0, dst1=dst1)
                    curve_for_proj = apply_similarity_transform(curve, scale, R, tt)
                else:
                    curve_for_proj = tps_warp_curve_to_anchors(
                        curve=curve,
                        anchor_xy=anchor_xy,
                        smoothing=float(args.anchor_tps_smoothing),
                    )

                    t_anchor, _, _ = project_to_polyline_arclength(
                        xy=anchor_xy,
                        line=curve_for_proj,
                    )
                    if np.any(~np.isfinite(t_anchor)):
                        raise SystemExit("Non-finite anchor t after TPS warping.")
                    dt = np.diff(t_anchor)
                    if not (np.all(dt > 0) or np.all(dt < 0)):
                        raise SystemExit(
                            "Anchors are not monotone along the fitted curve (order inversion); "
                            "click anchors in path order or use fewer anchors."
                        )

                curve_xy_for_qc = curve_for_proj

            t_used, r_signed_used, proj_used = project_to_polyline_arclength(xy=xy_used, line=curve_for_proj)
            r_q = rcoord_scale_q(r_signed_used, quantile=r_scale_abs_quantile)
            r_used = rcoord_scale_quantile(r_signed_used, r_q)

            t = np.full((xy.shape[0],), np.nan, dtype=float)
            r_ = np.full((xy.shape[0],), np.nan, dtype=float)
            proj = np.full((xy.shape[0], 2), np.nan, dtype=float)
            r_signed_store = np.full((xy.shape[0],), np.nan, dtype=float)

            t[fit_mask] = t_used
            r_[fit_mask] = r_used
            proj[fit_mask] = proj_used
            r_signed_store[fit_mask] = r_signed_used
            r_scale_info = {
                "abs_quantile": float(r_scale_abs_quantile),
                "q": float(r_q),
                "subset": {"obs_key": "ccf_adjusted", "obs_value": "cortex", "n_used": int(used_idx.size)},
            }

        with curvefit_path.open("w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["cell_id", "x", "y", "t", "r", "f1", "f2"])
            for cell_id, (x0, y0), ti, ri, (f1, f2) in zip(obs_names, xy, t, r_, proj, strict=True):
                w.writerow([cell_id, float(x0), float(y0), float(ti), float(ri), float(f1), float(f2)])
        if args.keep_intermediates:
            print(f"Wrote: {curvefit_path}")
    else:
        if curvefit_path is None:
            raise SystemExit("Provide --curvefit, or omit it to compute the principal curve.")
        curvefit = read_curvefit(curvefit_path)

        t = np.empty((obs_names.shape[0],), dtype=float)
        r_ = np.empty((obs_names.shape[0],), dtype=float)
        proj = np.empty((obs_names.shape[0], 2), dtype=float)

        missing: list[str] = []
        for i, cell_id in enumerate(obs_names):
            row = curvefit.get(cell_id)
            if row is None:
                missing.append(cell_id)
                continue
            t[i] = row["t"]
            r_[i] = row["r"]
            proj[i, 0] = row["f1"]
            proj[i, 1] = row["f2"]

        if missing:
            preview = ", ".join(missing[:10])
            extra = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            raise SystemExit(f"curvefit CSV missing {len(missing)} cell_ids: {preview}{extra}")

    if args.anchors_json is not None:
        if anchor_ids is None:
            raise SystemExit("Internal error: anchor_ids are not available after parsing --anchors-json.")
        start_id = anchor_ids[0]
        end_id = anchor_ids[-1]
        t, anchor_meta = apply_anchor_transform(
            t=t,
            obs_names=obs_names,
            start_cell_id=start_id,
            end_cell_id=end_id,
            clamp=args.anchor_clamp,
        )
        print(
            "Anchors applied:",
            f"start={start_id}",
            f"end={end_id}",
            "mode=rescale",
            f"clamp={args.anchor_clamp}",
            f"flipped={anchor_meta['flipped']}",
        )

    adata.obsm["principal"] = np.column_stack([t, r_]).astype(float, copy=False)
    adata.obsm["principal_curve_proj_xy"] = proj
    if r_signed_store is not None:
        adata.obsm["principal_r_signed"] = np.asarray(r_signed_store, dtype=float)

    uns: dict[str, Any] = dict(adata.uns) if adata.uns is not None else {}
    fit_meta: dict[str, Any] = {"spatial_key": "spatial", "curvefit_csv": str(curvefit_path)}
    if rois_to_process is not None:
        fit_meta["roi_obs_key"] = roi_obs_key
        fit_meta["rois"] = list(rois_to_process)
    if r_scale_info is not None:
        fit_meta["r_scale"] = r_scale_info
    uns["principal_curve_fit"] = fit_meta
    if anchor_meta is not None:
        uns["principal_anchors"] = anchor_meta
    adata.uns = uns

    if rois_to_process is not None and roi_values is not None:
        qc_path = Path(f"{out_prefix}.curve.qc.png")
        write_curve_qc_grid_png(
            out_path=qc_path,
            xy=xy,
            proj_xy=proj,
            t=t,
            roi_values=roi_values,
            rois=list(rois_to_process),
        )
        print(f"Wrote: {qc_path}")
    else:
        finite = np.isfinite(t) & np.isfinite(xy[:, :2]).all(axis=1)
        if np.any(finite):
            idx = np.flatnonzero(finite)
            start_i = int(idx[np.argmin(t[idx])])
            end_i = int(idx[np.argmax(t[idx])])
            qc_path = Path(f"{out_prefix}.curve.qc.png")
            write_anchor_qc_png(
                out_path=qc_path,
                xy=xy,
                proj_xy=proj,
                curve_xy=curve_xy_for_qc,
                t=t,
                start_index=start_i,
                end_index=end_i,
                anchor_indices=None,
                title=f"Principal curve QC (n={int(finite.sum())})",
            )
            print(f"Wrote: {qc_path}")

    if args.anchors_json is not None:
        anchored_qc_path = Path(f"{out_prefix}.anchors.qc.png")
        anchored_rsigned_qc_path = Path(f"{out_prefix}.anchors.r_signed.qc.png")

        if anchor_meta is None:
            raise SystemExit("Internal error: expected anchor_meta when --anchors-json is provided.")
        if anchor_indices is None:
            raise SystemExit("Internal error: expected anchor_indices when --anchors-json is provided.")

        plot_mask = np.isfinite(t)
        if roi_obs_key in adata.obs.columns and anchor_roi_value is not None:
            roi = adata.obs[roi_obs_key].astype(str).to_numpy()
            plot_mask &= roi == anchor_roi_value

        curve_xy_plot = curve_xy_for_qc
        if curve_xy_plot is not None:
            start_proj = proj[int(anchor_meta["start_index"]), :2].astype(float, copy=False)
            end_proj = proj[int(anchor_meta["end_index"]), :2].astype(float, copy=False)
            t_ends, _, _ = project_to_polyline_arclength(xy=np.vstack([start_proj, end_proj]), line=curve_xy_plot)
            curve_xy_plot = clip_polyline_to_interval(curve_xy_plot, float(t_ends[0]), float(t_ends[1]))

        write_anchor_qc_png(
            out_path=anchored_qc_path,
            xy=xy,
            proj_xy=proj,
            curve_xy=curve_xy_plot,
            t=t,
            start_index=int(anchor_meta["start_index"]),
            end_index=int(anchor_meta["end_index"]),
            anchor_indices=anchor_indices,
            title=f"Anchored princurve t (n={xy.shape[0]})",
            vmin=0.0,
            vmax=1.0,
            plot_mask=plot_mask,
        )
        print(f"Wrote: {anchored_qc_path}")

        if curve_xy_plot is not None:
            line = np.asarray(curve_xy_plot, dtype=float)[:, :2]
        else:
            start_proj = proj[int(anchor_meta["start_index"]), :2].astype(float, copy=True)
            end_proj = proj[int(anchor_meta["end_index"]), :2].astype(float, copy=True)
            line = curve_polyline_from_proj(t=t, proj_xy=proj, start_proj=start_proj, end_proj=end_proj)

        r_signed = signed_distance_to_polyline(xy=xy, line=line)
        lim = float(np.quantile(np.abs(r_signed[np.isfinite(r_signed)]), 0.99)) if r_signed.size else 1.0
        if not np.isfinite(lim) or lim <= 0:
            lim = 1.0
        write_anchor_qc_png(
            out_path=anchored_rsigned_qc_path,
            xy=xy,
            proj_xy=proj,
            curve_xy=line,
            t=t,
            c=r_signed,
            cmap="RdBu_r",
            c_label="signed distance to curve",
            start_index=int(anchor_meta["start_index"]),
            end_index=int(anchor_meta["end_index"]),
            anchor_indices=anchor_indices,
            title=f"Signed distance QC (n={xy.shape[0]})",
            vmin=-lim,
            vmax=lim,
            plot_mask=plot_mask,
        )
        print(f"Wrote: {anchored_rsigned_qc_path}")

        if args.keep_intermediates:
            anchored_curvefit_path = Path(f"{out_prefix}.anchors.curvefit.csv")
            with anchored_curvefit_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cell_id", "x", "y", "t", "r", "f1", "f2"])
                for cell_id, (x0, y0), ti, ri, (f1, f2) in zip(obs_names, xy, t, r_, proj, strict=True):
                    w.writerow([cell_id, float(x0), float(y0), float(ti), float(ri), float(f1), float(f2)])
            print(f"Wrote: {anchored_curvefit_path}")

    adata.write_h5ad(out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
