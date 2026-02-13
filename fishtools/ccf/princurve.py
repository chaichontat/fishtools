from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.spatial import cKDTree


def _fit_dense_spline_through_points(path_xy: np.ndarray, *, n_dense: int) -> np.ndarray:
    points = np.asarray(path_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise SystemExit(f"Not enough points to fit a spline (shape={points.shape}).")
    points = points[:, :2]
    if points.shape[0] >= 2:
        d = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate([[True], d > 1e-12])
        points = points[keep]
    if points.shape[0] < 2:
        raise SystemExit("Anchors are degenerate (all identical after deduplication).")

    seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if float(s[-1]) == 0:
        u = np.linspace(0.0, 1.0, points.shape[0])
    else:
        u = s / float(s[-1])

    u_dense = np.linspace(0.0, 1.0, int(max(2, n_dense)))
    x_spline = CubicSpline(u, points[:, 0], bc_type="natural")
    y_spline = CubicSpline(u, points[:, 1], bc_type="natural")
    x_dense = x_spline(u_dense)
    y_dense = y_spline(u_dense)
    curve = np.column_stack([np.asarray(x_dense, float), np.asarray(y_dense, float)])
    curve[0, :] = points[0, :]
    curve[-1, :] = points[-1, :]
    return curve.astype(float, copy=False)


def fit_anchor_curve(
    *,
    anchor_xy: np.ndarray,
    n_dense: int,
    smoothing: float,
    fit_constrained_spline_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Fit an anchor-driven spline with scale-aware smoothing.

    `splprep` smoothing is expressed in squared coordinate units, so raw values are
    not comparable across datasets with different spatial scales. We normalize anchor
    coordinates by the median anchor-to-anchor spacing and interpret `smoothing` as a
    dimensionless knob where larger values allow larger RMS residuals (relative to that
    spacing) before re-scaling back to original coordinates.
    """
    points = np.asarray(anchor_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise SystemExit(f"Unexpected anchor shape: {points.shape}")
    points = points[:, :2]
    if points.shape[0] < 2:
        raise SystemExit("Need at least 2 anchors to build curve.")
    if smoothing < 0:
        raise SystemExit("--anchor-smoothing must be >= 0.")

    if points.shape[0] >= 2:
        d = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate([[True], d > 1e-12])
        points = points[keep]
    if points.shape[0] < 2:
        raise SystemExit("Anchors are degenerate (all identical after deduplication).")

    if smoothing <= 0:
        if fit_constrained_spline_fn is None:
            return _fit_dense_spline_through_points(points, n_dense=int(n_dense))
        _u_dense, curve, _deriv, _ = fit_constrained_spline_fn(points, n_dense=int(n_dense))
        return np.asarray(curve, dtype=float)

    seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, 1.0, points.shape[0]) if float(s[-1]) == 0 else s / float(s[-1])

    k = min(3, points.shape[0] - 1)
    if k < 1:
        raise SystemExit("Not enough anchors to fit a spline.")

    # Normalize smoothing to anchor spacing so the knob behaves similarly across ROIs.
    finite_seg = seg[np.isfinite(seg) & (seg > 0)]
    spacing = float(np.median(finite_seg)) if finite_seg.size > 0 else 1.0
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0

    origin = points[0, :].astype(float, copy=True)
    points_norm = (points - origin[None, :]) / spacing

    # In normalized space, treat smoothing as ~0.1 * anchor-spacing RMS residual per point.
    smooth_sigma = 0.1 * float(smoothing)
    smooth_s = float(points.shape[0]) * float(smooth_sigma**2)
    tck, _ = splprep([points_norm[:, 0], points_norm[:, 1]], u=u, s=smooth_s, k=k)

    u_dense = np.linspace(0.0, 1.0, int(max(2, n_dense)))
    x_dense_norm, y_dense_norm = splev(u_dense, tck)
    curve_norm = np.column_stack([np.asarray(x_dense_norm, float), np.asarray(y_dense_norm, float)])
    curve = curve_norm * spacing + origin[None, :]
    curve[0, :] = points[0, :]
    curve[-1, :] = points[-1, :]
    return curve


def _polyline_with_optional_smooth_extrapolation(
    *,
    line: np.ndarray,
    endpoint_extrapolation: float,
) -> tuple[np.ndarray, np.ndarray]:
    poly = np.asarray(line, dtype=float)[:, :2]
    if poly.shape[0] < 2:
        raise SystemExit("Polyline must have at least 2 points.")
    if endpoint_extrapolation < 0:
        raise SystemExit("--endpoint-extrapolation must be >= 0.")

    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    poly = poly[keep]
    if poly.shape[0] < 2:
        raise SystemExit("Degenerate polyline (all segments have zero length).")

    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0:
        raise SystemExit("Degenerate polyline arc-length.")
    u = cum / total

    ext = float(endpoint_extrapolation)
    if ext == 0.0:
        return poly.astype(float, copy=False), u.astype(float, copy=False)

    u_eval = np.linspace(-ext, 1.0 + ext, int(max(3, poly.shape[0])))
    if poly.shape[0] == 2:
        d = poly[1] - poly[0]
        poly_ext = np.column_stack(
            [
                poly[0, 0] + (u_eval * d[0]),
                poly[0, 1] + (u_eval * d[1]),
            ]
        ).astype(float, copy=False)
        return poly_ext, u_eval.astype(float, copy=False)

    x_spline = CubicSpline(u, poly[:, 0], bc_type="natural")
    y_spline = CubicSpline(u, poly[:, 1], bc_type="natural")
    poly_ext = np.column_stack([x_spline(u_eval), y_spline(u_eval)]).astype(float, copy=False)
    return poly_ext, u_eval.astype(float, copy=False)


def project_to_polyline_arclength(
    *,
    xy: np.ndarray,
    line: np.ndarray,
    k: int = 50,
    endpoint_extrapolation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(xy, dtype=float)[:, :2]
    poly, poly_u = _polyline_with_optional_smooth_extrapolation(
        line=line,
        endpoint_extrapolation=float(endpoint_extrapolation),
    )

    seg_a = poly[:-1]
    seg_b = poly[1:]
    seg_u0 = poly_u[:-1]
    seg_u1 = poly_u[1:]
    seg_v = seg_b - seg_a
    seg_len = np.linalg.norm(seg_v, axis=1)
    ok = seg_len > 1e-12
    seg_a = seg_a[ok]
    seg_v = seg_v[ok]
    seg_u0 = seg_u0[ok]
    seg_u1 = seg_u1[ok]
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

    t = seg_u0[best_seg] + best_tau * (seg_u1[best_seg] - seg_u0[best_seg])

    tan = seg_v[best_seg]
    res = pts - best_proj
    cross = tan[:, 0] * res[:, 1] - tan[:, 1] * res[:, 0]
    sign = np.where(cross >= 0, 1.0, -1.0)
    r_signed = sign * np.sqrt(best_d2)
    return t.astype(float), r_signed.astype(float), best_proj.astype(float)


def signed_distance_to_polyline(
    *,
    xy: np.ndarray,
    line: np.ndarray,
    endpoint_extrapolation: float = 0.0,
) -> np.ndarray:
    _, r_signed, _ = project_to_polyline_arclength(
        xy=xy,
        line=line,
        endpoint_extrapolation=float(endpoint_extrapolation),
    )
    return r_signed
