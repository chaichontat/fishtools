from __future__ import annotations

from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d

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
    support_flat = np.asarray(data.support_mask_tall, dtype=bool).reshape(-1)
    neomeso_flat = np.asarray(data.neomeso_mask_tall, dtype=bool).reshape(-1)
    tri_support = np.all(support_flat[faces], axis=1)
    tri_neomeso = np.all(neomeso_flat[faces], axis=1)
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
    ordered_geom_neomeso = prepare_ordered_surface_geometry(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3,
        y3d=y3,
        z3d=z3,
        faces=faces,
        tri_neomeso=tri_neomeso,
        keep_tri=tri_support & tri_neomeso,
    )

    ap_um_by_slice = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    ml_um_at_t = np.asarray(data.ml_um_at_t, dtype=np.float64).reshape(n_rows, n_cols)
    support_mask_tall = np.asarray(data.support_mask_tall, dtype=bool).reshape(n_rows, n_cols)
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
