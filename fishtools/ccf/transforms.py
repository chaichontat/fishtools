from __future__ import annotations

from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import UnivariateSpline

from ccf.refextract.plot_ap_ml_mapping_surface_3d import build_coronal_ap_ml_surface


def project_xyz_to_2d_ortho_data(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    elev_deg: float,
    azim_deg: float,
    roll_deg: float,
    vertical_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic camera projection in data units, matching mpl3d view conventions."""
    x_vals = np.asarray(x, dtype=np.float64).reshape(-1)
    y_vals = np.asarray(y, dtype=np.float64).reshape(-1)
    z_vals = np.asarray(z, dtype=np.float64).reshape(-1)
    if x_vals.shape != y_vals.shape or x_vals.shape != z_vals.shape:
        raise ValueError(f"x/y/z shape mismatch: {x_vals.shape} {y_vals.shape} {z_vals.shape}")
    if int(vertical_axis) not in (0, 1, 2):
        raise ValueError(f"vertical_axis must be 0,1,2; got {vertical_axis}")

    elev_rad = np.deg2rad(float(art3d._norm_angle(float(elev_deg))))
    azim_rad = np.deg2rad(float(art3d._norm_angle(float(azim_deg))))
    roll_rad = np.deg2rad(float(art3d._norm_angle(float(roll_deg))))

    p0 = float(np.cos(elev_rad) * np.cos(azim_rad))
    p1 = float(np.cos(elev_rad) * np.sin(azim_rad))
    p2 = float(np.sin(elev_rad))
    ps = np.roll(np.array([p0, p1, p2], dtype=np.float64), int(vertical_axis) - 2)
    w = ps / float(np.linalg.norm(ps))

    V = np.zeros(3, dtype=np.float64)
    V[int(vertical_axis)] = -1.0 if abs(float(elev_rad)) > (np.pi / 2.0) else 1.0
    u = np.cross(V, w)
    u = u / float(np.linalg.norm(u))
    v = np.cross(w, u)
    if roll_rad != 0:
        rot_fn = getattr(proj3d, "rotation_about_vector", None)
        if rot_fn is None:
            rot_fn = getattr(proj3d, "_rotation_about_vector", None)
        if rot_fn is None:
            raise AttributeError("mpl_toolkits.mplot3d.proj3d is missing rotation_about_vector/_rotation_about_vector")
        Rroll = rot_fn(w, -float(roll_rad))
        u = Rroll @ u
        v = Rroll @ v

    x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    z_min, z_max = float(np.nanmin(z_vals)), float(np.nanmax(z_vals))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    dx = x_vals - x_mid
    dy = y_vals - y_mid
    dz = z_vals - z_mid
    proj_x = (dx * u[0]) + (dy * u[1]) + (dz * u[2])
    proj_y = (dx * v[0]) + (dy * v[1]) + (dz * v[2])
    proj_z = (dx * w[0]) + (dy * w[1]) + (dz * w[2])
    return proj_x, proj_y, proj_z


def prepare_ordered_surface_geometry(
    *,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    faces: np.ndarray,
    tri_neomeso: np.ndarray,
    keep_tri: np.ndarray,
) -> dict[str, np.ndarray]:
    """Precompute painter-order triangle geometry and normals for native projections."""
    x2 = np.asarray(x2d, dtype=np.float64).reshape(-1)
    y2 = np.asarray(y2d, dtype=np.float64).reshape(-1)
    z2 = np.asarray(z2d, dtype=np.float64).reshape(-1)
    x3 = np.asarray(x3d, dtype=np.float64).reshape(-1)
    y3 = np.asarray(y3d, dtype=np.float64).reshape(-1)
    z3 = np.asarray(z3d, dtype=np.float64).reshape(-1)
    tris = np.asarray(faces, dtype=np.int32)
    neo = np.asarray(tri_neomeso, dtype=bool).reshape(-1)
    keep = np.asarray(keep_tri, dtype=bool).reshape(-1)
    if x2.shape != y2.shape or x2.shape != z2.shape:
        raise ValueError("x2d/y2d/z2d shape mismatch in ordered geometry prep")
    if x3.shape != x2.shape or y3.shape != x2.shape or z3.shape != x2.shape:
        raise ValueError("x3d/y3d/z3d shape mismatch in ordered geometry prep")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must be (F,3) in ordered geometry prep")
    if neo.shape != (tris.shape[0],) or keep.shape != (tris.shape[0],):
        raise ValueError("triangle mask shape mismatch in ordered geometry prep")

    tri_idx = np.flatnonzero(keep).astype(np.int32, copy=False)
    if tri_idx.size == 0:
        raise ValueError("No triangles selected in ordered geometry prep")
    tris_k = tris[tri_idx]
    tri_depth = np.nanmean(z2[tris_k], axis=1)
    order = np.argsort(tri_depth)
    tris_o = tris_k[order]
    tri_neo_o = neo[tri_idx][order]
    polys2d = np.stack([x2[tris_o], y2[tris_o]], axis=2).astype(np.float64, copy=False)

    p0 = np.column_stack([x3[tris_o[:, 0]], y3[tris_o[:, 0]], z3[tris_o[:, 0]]]).astype(np.float64, copy=False)
    p1 = np.column_stack([x3[tris_o[:, 1]], y3[tris_o[:, 1]], z3[tris_o[:, 1]]]).astype(np.float64, copy=False)
    p2 = np.column_stack([x3[tris_o[:, 2]], y3[tris_o[:, 2]], z3[tris_o[:, 2]]]).astype(np.float64, copy=False)
    normals = np.cross(p1 - p0, p2 - p0)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normal_norm) & (normal_norm > 0)
    normals_unit = np.zeros_like(normals)
    normals_unit[valid] = normals[valid] / normal_norm[valid, None]
    return {
        "tris": tris_o.astype(np.int32, copy=False),
        "tri_neomeso": tri_neo_o.astype(bool, copy=False),
        "polys2d": polys2d,
        "normals_unit": normals_unit.astype(np.float64, copy=False),
        "normal_valid": valid.astype(bool, copy=False),
    }


def _clip_polygon_against_signed_distance(
    poly_param: np.ndarray,
    poly_proj: np.ndarray,
    poly_xyz: np.ndarray,
    *,
    signed_distance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if poly_param.shape[0] == 0:
        return poly_param, poly_proj, poly_xyz
    out_param: list[np.ndarray] = []
    out_proj: list[np.ndarray] = []
    out_xyz: list[np.ndarray] = []
    prev_param = np.asarray(poly_param[-1], dtype=np.float64)
    prev_proj = np.asarray(poly_proj[-1], dtype=np.float64)
    prev_xyz = np.asarray(poly_xyz[-1], dtype=np.float64)
    prev_dist = float(signed_distance(prev_param))
    prev_inside = np.isfinite(prev_dist) and prev_dist >= 0.0
    for idx in range(poly_param.shape[0]):
        curr_param = np.asarray(poly_param[idx], dtype=np.float64)
        curr_proj = np.asarray(poly_proj[idx], dtype=np.float64)
        curr_xyz = np.asarray(poly_xyz[idx], dtype=np.float64)
        curr_dist = float(signed_distance(curr_param))
        curr_inside = np.isfinite(curr_dist) and curr_dist >= 0.0
        if prev_inside != curr_inside and np.isfinite(prev_dist) and np.isfinite(curr_dist) and prev_dist != curr_dist:
            t = float(np.clip(prev_dist / (prev_dist - curr_dist), 0.0, 1.0))
            out_param.append(prev_param + t * (curr_param - prev_param))
            out_proj.append(prev_proj + t * (curr_proj - prev_proj))
            out_xyz.append(prev_xyz + t * (curr_xyz - prev_xyz))
        if curr_inside:
            out_param.append(curr_param)
            out_proj.append(curr_proj)
            out_xyz.append(curr_xyz)
        prev_param = curr_param
        prev_proj = curr_proj
        prev_xyz = curr_xyz
        prev_dist = curr_dist
        prev_inside = curr_inside
    if len(out_param) < 3:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(out_param, dtype=np.float64),
        np.asarray(out_proj, dtype=np.float64),
        np.asarray(out_xyz, dtype=np.float64),
    )


def prepare_ordered_surface_geometry_clipped_to_parametric_band(
    *,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    faces: np.ndarray,
    tri_support: np.ndarray,
    n_cols: int,
    start_fit: np.ndarray | None,
    end_fit: np.ndarray | None,
) -> dict[str, object]:
    """Clip support triangles to a continuous row->t band and keep painter order."""
    if start_fit is None or end_fit is None:
        raise ValueError("start_fit/end_fit are required for clipped ordered geometry")
    x2 = np.asarray(x2d, dtype=np.float64).reshape(-1)
    y2 = np.asarray(y2d, dtype=np.float64).reshape(-1)
    z2 = np.asarray(z2d, dtype=np.float64).reshape(-1)
    x3 = np.asarray(x3d, dtype=np.float64).reshape(-1)
    y3 = np.asarray(y3d, dtype=np.float64).reshape(-1)
    z3 = np.asarray(z3d, dtype=np.float64).reshape(-1)
    tris = np.asarray(faces, dtype=np.int32)
    tri_support_b = np.asarray(tri_support, dtype=bool).reshape(-1)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must be (F,3) in clipped ordered geometry prep")
    if tri_support_b.shape != (tris.shape[0],):
        raise ValueError("tri_support shape mismatch in clipped ordered geometry prep")
    start_arr = np.asarray(start_fit, dtype=np.float64).reshape(-1)
    end_arr = np.asarray(end_fit, dtype=np.float64).reshape(-1)
    if start_arr.shape != end_arr.shape:
        raise ValueError("start_fit/end_fit shape mismatch in clipped ordered geometry prep")
    valid_rows = np.isfinite(start_arr) & np.isfinite(end_arr)
    valid_idx = np.flatnonzero(valid_rows)
    if valid_idx.size < 2:
        raise ValueError("Need at least two valid rows for clipped ordered geometry prep")

    rows_valid = valid_idx.astype(np.float64)
    start_valid = start_arr[valid_rows]
    end_valid = end_arr[valid_rows]
    row_lo = float(rows_valid[0])
    row_hi = float(rows_valid[-1])

    def interp_start(row: float) -> float:
        return float(np.interp(row, rows_valid, start_valid))

    def interp_end(row: float) -> float:
        return float(np.interp(row, rows_valid, end_valid))

    polys2d: list[np.ndarray] = []
    polys3d: list[np.ndarray] = []
    source_tris: list[np.ndarray] = []
    depths: list[float] = []
    normals: list[np.ndarray] = []
    normal_valid: list[bool] = []

    for tri_i in np.flatnonzero(tri_support_b):
        tri = tris[int(tri_i)]
        tri_rows = (tri // int(n_cols)).astype(np.float64)
        tri_cols = (tri % int(n_cols)).astype(np.float64)
        poly_param = np.column_stack([tri_rows, tri_cols]).astype(np.float64, copy=False)
        poly_proj = np.column_stack([x2[tri], y2[tri], z2[tri]]).astype(np.float64, copy=False)
        poly_xyz = np.column_stack([x3[tri], y3[tri], z3[tri]]).astype(np.float64, copy=False)
        poly_param, poly_proj, poly_xyz = _clip_polygon_against_signed_distance(
            poly_param,
            poly_proj,
            poly_xyz,
            signed_distance=lambda p: p[0] - row_lo,
        )
        poly_param, poly_proj, poly_xyz = _clip_polygon_against_signed_distance(
            poly_param,
            poly_proj,
            poly_xyz,
            signed_distance=lambda p: row_hi - p[0],
        )
        poly_param, poly_proj, poly_xyz = _clip_polygon_against_signed_distance(
            poly_param,
            poly_proj,
            poly_xyz,
            signed_distance=lambda p: p[1] - interp_start(float(p[0])),
        )
        poly_param, poly_proj, poly_xyz = _clip_polygon_against_signed_distance(
            poly_param,
            poly_proj,
            poly_xyz,
            signed_distance=lambda p: interp_end(float(p[0])) - p[1],
        )
        if poly_param.shape[0] < 3:
            continue
        polys2d.append(poly_proj[:, :2].astype(np.float64, copy=False))
        polys3d.append(poly_xyz.astype(np.float64, copy=False))
        source_tris.append(tri.astype(np.int32, copy=False))
        depths.append(float(np.nanmean(poly_proj[:, 2])))
        normal = np.cross(poly_xyz[1] - poly_xyz[0], poly_xyz[2] - poly_xyz[0])
        norm = float(np.linalg.norm(normal))
        valid = np.isfinite(norm) and norm > 0.0
        normal_valid.append(valid)
        normals.append((normal / norm) if valid else np.zeros((3,), dtype=np.float64))

    if not polys2d:
        raise ValueError("No triangles remain after clipping ordered geometry to neomeso band")

    order = np.argsort(np.asarray(depths, dtype=np.float64))
    return {
        "tris": np.asarray([source_tris[int(i)] for i in order], dtype=np.int32),
        "tri_neomeso": np.ones((len(order),), dtype=bool),
        "polys2d": [polys2d[int(i)] for i in order],
        "polys3d": [polys3d[int(i)] for i in order],
        "normals_unit": np.asarray([normals[int(i)] for i in order], dtype=np.float64),
        "normal_valid": np.asarray([normal_valid[int(i)] for i in order], dtype=bool),
    }


def smooth_neomeso_mask_tall_parametric(
    *,
    support_mask_tall: np.ndarray,
    neomeso_mask_tall: np.ndarray,
    smoothing: float = 6.0,
) -> np.ndarray:
    """Smooth row-wise neomeso t-bounds across AP before native triangulation."""
    support = np.asarray(support_mask_tall, dtype=bool)
    start_fit, end_fit = _smooth_neomeso_t_bounds(
        support_mask_tall=support,
        neomeso_mask_tall=neomeso_mask_tall,
        smoothing=smoothing,
    )
    if start_fit is None or end_fit is None:
        return support & np.asarray(neomeso_mask_tall, dtype=bool)
    n_rows, n_cols = support.shape
    cols = np.arange(n_cols, dtype=np.float64)
    out = np.zeros_like(support, dtype=bool)
    valid_rows = np.isfinite(start_fit) & np.isfinite(end_fit)
    row_idx = np.flatnonzero(valid_rows)
    if row_idx.size == 0:
        return out
    row_start = int(row_idx[0])
    row_stop = int(row_idx[-1]) + 1
    for row in range(row_start, row_stop):
        out[row] = support[row] & (cols >= start_fit[row]) & (cols <= end_fit[row])
    return out


def _smooth_neomeso_t_bounds(
    *,
    support_mask_tall: np.ndarray,
    neomeso_mask_tall: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    support = np.asarray(support_mask_tall, dtype=bool)
    neomeso = np.asarray(neomeso_mask_tall, dtype=bool)
    if support.shape != neomeso.shape or support.ndim != 2:
        raise ValueError("support_mask_tall and neomeso_mask_tall must be matching 2D arrays.")
    n_rows, n_cols = support.shape
    start = np.full((n_rows,), np.nan, dtype=np.float64)
    end = np.full((n_rows,), np.nan, dtype=np.float64)
    for row in range(n_rows):
        idx = np.flatnonzero(support[row] & neomeso[row])
        if idx.size == 0:
            continue
        start[row] = float(idx[0])
        end[row] = float(idx[-1])
    valid = np.isfinite(start) & np.isfinite(end)
    if np.count_nonzero(valid) < 4:
        return None, None

    row_idx = np.flatnonzero(valid).astype(np.float64)
    smooth_s = float(smoothing) * float(row_idx.size)
    if not np.isfinite(smooth_s) or smooth_s < 0.0:
        raise ValueError(f"smoothing must be finite and >= 0, got {smoothing!r}")
    k = min(3, int(row_idx.size) - 1)
    start_spline = UnivariateSpline(row_idx, start[valid], k=k, s=smooth_s)
    end_spline = UnivariateSpline(row_idx, end[valid], k=k, s=smooth_s)
    eval_rows = np.arange(n_rows, dtype=np.float64)
    start_fit = np.clip(start_spline(eval_rows), 0.0, float(n_cols - 1))
    end_fit = np.clip(end_spline(eval_rows), 0.0, float(n_cols - 1))
    mid = 0.5 * (start_fit + end_fit)
    half = 0.5 * np.maximum(end_fit - start_fit, 0.0)
    start_fit = np.clip(mid - half, 0.0, float(n_cols - 1))
    end_fit = np.clip(mid + half, 0.0, float(n_cols - 1))
    outside = ~valid
    start_fit[outside] = np.nan
    end_fit[outside] = np.nan
    return start_fit, end_fit


def smooth_neomeso_tri_mask_parametric(
    *,
    faces: np.ndarray,
    tri_support: np.ndarray,
    n_cols: int,
    start_fit: np.ndarray | None,
    end_fit: np.ndarray | None,
) -> np.ndarray:
    """Evaluate the smoothed neomeso boundary at triangle centroids."""
    tris = np.asarray(faces, dtype=np.int32)
    tri_support_b = np.asarray(tri_support, dtype=bool).reshape(-1)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {tris.shape}")
    if tri_support_b.shape != (tris.shape[0],):
        raise ValueError("tri_support shape mismatch")
    if start_fit is None or end_fit is None:
        return tri_support_b.copy()
    start_arr = np.asarray(start_fit, dtype=np.float64).reshape(-1)
    end_arr = np.asarray(end_fit, dtype=np.float64).reshape(-1)
    if start_arr.shape != end_arr.shape:
        raise ValueError("start_fit/end_fit shape mismatch")
    rows = np.arange(start_arr.size, dtype=np.float64)
    row_cent = np.mean((tris // int(n_cols)).astype(np.float64), axis=1)
    col_cent = np.mean((tris % int(n_cols)).astype(np.float64), axis=1)
    valid_rows = np.isfinite(start_arr) & np.isfinite(end_arr)
    valid_idx = np.flatnonzero(valid_rows)
    if valid_idx.size < 2:
        return tri_support_b.copy()
    start_cent = np.interp(row_cent, rows[valid_rows], start_arr[valid_rows], left=np.nan, right=np.nan)
    end_cent = np.interp(row_cent, rows[valid_rows], end_arr[valid_rows], left=np.nan, right=np.nan)
    inside = np.isfinite(start_cent) & np.isfinite(end_cent) & (col_cent >= start_cent) & (col_cent <= end_cent)
    return tri_support_b & inside


def build_apml_native_surface_projection_context(
    *,
    outdir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    ref_t: float,
    band_frac: float,
    res_ijk_um: tuple[float, float, float],
    elev_deg: float,
    azim_deg: float,
    roll_deg: float,
) -> dict[str, object]:
    """Build flattened native-projection context for AP/ML surface plotting."""
    data = build_coronal_ap_ml_surface(
        outdir=outdir,
        slice_i_min=int(slice_i_min),
        slice_i_max=int(slice_i_max),
        n_t=int(n_t),
        ref_t=float(ref_t),
        band_frac=float(band_frac),
        b_const=0.25,
        res_ijk_um=(float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2])),
    )
    slice_keys = np.asarray(data.slice_keys, dtype=np.int32).reshape(-1)
    t_grid = np.asarray(data.t_grid, dtype=np.float64).reshape(-1)
    n_rows = int(slice_keys.size)
    n_cols = int(t_grid.size)
    vertices_um = np.asarray(data.vertices_ijk_um, dtype=np.float64).reshape(n_rows, n_cols, 3)
    verts_flat = vertices_um.reshape(-1, 3)
    # mpl3d uses x,y,z; match notebook axes: x=k (ML), y=j (DV), z=i (AP)
    x3 = verts_flat[:, 2]
    y3 = verts_flat[:, 1]
    z3 = verts_flat[:, 0]
    x2d, y2d, z2d = project_xyz_to_2d_ortho_data(
        x=x3,
        y=y3,
        z=z3,
        elev_deg=float(elev_deg),
        azim_deg=float(azim_deg),
        roll_deg=float(roll_deg),
    )

    faces = np.asarray(data.faces, dtype=np.int32)
    support_mask_tall = np.asarray(data.support_mask_tall, dtype=bool).reshape(n_rows, n_cols)
    raw_neomeso_mask_tall = np.asarray(data.neomeso_mask_tall, dtype=bool).reshape(n_rows, n_cols)
    start_fit, end_fit = _smooth_neomeso_t_bounds(
        support_mask_tall=support_mask_tall,
        neomeso_mask_tall=raw_neomeso_mask_tall,
        smoothing=6.0,
    )
    neomeso_mask_tall = smooth_neomeso_mask_tall_parametric(
        support_mask_tall=support_mask_tall,
        neomeso_mask_tall=raw_neomeso_mask_tall,
        smoothing=6.0,
    )
    support_flat = support_mask_tall.reshape(-1)
    neomeso_flat = neomeso_mask_tall.reshape(-1)
    tri_support = np.all(support_flat[faces], axis=1)
    tri_neomeso = smooth_neomeso_tri_mask_parametric(
        faces=faces,
        tri_support=tri_support,
        n_cols=n_cols,
        start_fit=start_fit,
        end_fit=end_fit,
    )
    ordered_geom_support = prepare_ordered_surface_geometry(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3,
        y3d=y3,
        z3d=z3,
        faces=faces,
        tri_neomeso=tri_neomeso,
        keep_tri=tri_support,
    )
    ordered_geom_neomeso = prepare_ordered_surface_geometry_clipped_to_parametric_band(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3,
        y3d=y3,
        z3d=z3,
        faces=faces,
        tri_support=tri_support,
        n_cols=n_cols,
        start_fit=start_fit,
        end_fit=end_fit,
    )

    ap_um_by_slice = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    ml_um_at_t = np.asarray(data.ml_um_at_t, dtype=np.float64).reshape(n_rows, n_cols)
    ap_um_flat = np.broadcast_to(ap_um_by_slice[:, None], (n_rows, n_cols)).reshape(-1).astype(np.float64, copy=False)
    ml_um_flat = ml_um_at_t.reshape(-1).astype(np.float64, copy=False)

    return {
        "x2d": x2d.astype(np.float64, copy=False),
        "y2d": y2d.astype(np.float64, copy=False),
        "z2d": z2d.astype(np.float64, copy=False),
        "x3": x3.astype(np.float64, copy=False),
        "y3": y3.astype(np.float64, copy=False),
        "z3": z3.astype(np.float64, copy=False),
        "triangles": faces.astype(np.int32, copy=False),
        "tri_support": tri_support.astype(bool, copy=False),
        "tri_neomeso": tri_neomeso.astype(bool, copy=False),
        "neomeso_start_fit": start_fit.astype(np.float64, copy=False) if start_fit is not None else None,
        "neomeso_end_fit": end_fit.astype(np.float64, copy=False) if end_fit is not None else None,
        "ordered_geom_support": ordered_geom_support,
        "ordered_geom_neomeso": ordered_geom_neomeso,
        "support_flat": support_flat.astype(bool, copy=False),
        "neomeso_flat": neomeso_flat.astype(bool, copy=False),
        "support_mask_tall": support_mask_tall.astype(bool, copy=False),
        "ap_um_by_slice": ap_um_by_slice.astype(np.float64, copy=False),
        "ml_um_at_t": ml_um_at_t.astype(np.float64, copy=False),
        "faces": faces.astype(np.int32, copy=False),
        "ap_um_flat": ap_um_flat,
        "ml_um_flat": ml_um_flat,
        "n_rows": n_rows,
        "n_cols": n_cols,
    }
