from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ccf.refextract.plot_ap_ml_mapping_surface_3d import SurfaceMappingData, build_coronal_ap_ml_surface


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"Expected triangle faces shape (F,3), got {tri.shape}.")
    return np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel(order="C")


def _ijk_um_to_xyz_um(verts_ijk_um: np.ndarray) -> np.ndarray:
    arr = np.asarray(verts_ijk_um, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N,3), got {arr.shape}.")
    return arr[:, [2, 1, 0]]


def _flip_xyz_about_xz_plane(points_xyz: np.ndarray, *, y0: float) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {pts.shape}.")
    if not np.isfinite(float(y0)):
        raise ValueError(f"y0 must be finite, got {y0}.")
    out = pts.copy()
    out[:, 1] = 2.0 * float(y0) - out[:, 1]
    return out.astype(np.float32, copy=False)


def _build_target_vertices_ijk_um(data: SurfaceMappingData) -> np.ndarray:
    """
    Build the flattened target surface in physical ijk_um coordinates.

    Target chart is:
    - i := source-anchored AP displacement from optimized per-slice AP axis
    - k := source-anchored ML displacement from per-vertex ML arclength coordinate
    - j := constant DV plane (median source j) to avoid an arbitrary global shift
    """
    n_rows = int(data.slice_keys.size)
    n_cols = int(data.t_grid.size)
    n_verts = int(n_rows * n_cols)
    src = np.asarray(data.vertices_ijk_um, dtype=np.float64)
    if src.shape != (n_verts, 3):
        raise ValueError(f"Source vertex shape mismatch: expected {(n_verts, 3)}, got {src.shape}.")

    ap_per_row = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    ml_grid = np.asarray(data.ml_um_at_t, dtype=np.float64)
    if ap_per_row.shape != (n_rows,):
        raise ValueError(f"ap_um_by_slice shape mismatch: expected {(n_rows,)}, got {ap_per_row.shape}.")
    if ml_grid.shape != (n_rows, n_cols):
        raise ValueError(f"ml_um_at_t shape mismatch: expected {(n_rows, n_cols)}, got {ml_grid.shape}.")

    src_grid = src.reshape(n_rows, n_cols, 3)
    target_k = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for row in range(n_rows):
        ml_row = np.asarray(ml_grid[row], dtype=np.float64)
        src_k_row = np.asarray(src_grid[row, :, 2], dtype=np.float64)
        finite = np.isfinite(ml_row) & np.isfinite(src_k_row)
        if int(np.count_nonzero(finite)) == 0:
            raise ValueError(f"Row {row} has no finite ML/source-k values for anchored unfolding.")
        valid_idx = np.flatnonzero(finite)
        anchor_local = int(np.argmin(np.abs(ml_row[finite])))
        anchor_idx = int(valid_idx[anchor_local])
        ml_anchor = float(ml_row[anchor_idx])
        src_anchor_k = float(src_k_row[anchor_idx])
        target_k[row] = src_anchor_k + (ml_row - ml_anchor)

    anchor_row = int(n_rows // 2)
    ap_anchor = float(ap_per_row[anchor_row])
    src_anchor_i = float(np.nanmedian(src_grid[anchor_row, :, 0]))
    if not np.isfinite(src_anchor_i):
        raise ValueError("Could not determine finite AP anchor position from source vertices.")

    ap_target_i_by_row = src_anchor_i + (ap_per_row - ap_anchor)

    target = np.empty((n_verts, 3), dtype=np.float64)
    target[:, 0] = np.repeat(ap_target_i_by_row, n_cols)
    target[:, 2] = target_k.reshape(-1)

    j_plane = float(np.nanmedian(src[:, 1]))
    if not np.isfinite(j_plane):
        raise ValueError("Could not determine finite DV collapse plane from source vertices.")
    target[:, 1] = j_plane

    if not np.isfinite(target).all():
        raise ValueError("Target vertices contain non-finite values.")
    return target.astype(np.float32, copy=False)


def _phase_layout(*, n_frames: int, phase1_frac: float) -> tuple[int, int]:
    n = int(n_frames)
    if n < 2:
        raise ValueError(f"--n-frames must be >=2, got {n_frames}.")
    p = float(phase1_frac)
    if not (0.0 < p < 1.0):
        raise ValueError(f"--phase1-frac must be in (0,1), got {phase1_frac}.")
    n1 = int(np.clip(np.rint(p * n), 1, n - 1))
    n2 = n - n1
    return n1, n2


def _phase_alphas(*, frame: int, n_frames: int, n_phase1: int) -> tuple[float, float]:
    idx = int(frame)
    n = int(n_frames)
    n1 = int(n_phase1)
    if not (0 <= idx < n):
        raise ValueError(f"frame index out of bounds: frame={idx} n_frames={n}.")
    if not (1 <= n1 <= n - 1):
        raise ValueError(f"Invalid phase split: n_phase1={n1} n_frames={n}.")

    if idx < n1:
        den = max(1, n1 - 1)
        return (float(idx) / float(den), 0.0)
    den = max(1, (n - n1) - 1)
    beta = float(idx - n1) / float(den)
    return (1.0, beta)


def _interpolate_vertices_ijk_um(
    *,
    source_ijk_um: np.ndarray,
    target_ijk_um: np.ndarray,
    alpha_ml: float,
    beta_ap: float,
) -> np.ndarray:
    """
    Two-stage morph:
    - Stage 1 (alpha_ml): unfold ML by moving k only.
    - Stage 2 (beta_ap): unfold AP and collapse DV by moving i and j.
    """
    src = np.asarray(source_ijk_um, dtype=np.float64)
    dst = np.asarray(target_ijk_um, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching (N,3) source/target arrays, got {src.shape} and {dst.shape}.")

    a = float(np.clip(alpha_ml, 0.0, 1.0))
    b = float(np.clip(beta_ap, 0.0, 1.0))
    out = src.copy()
    out[:, 2] = src[:, 2] + a * (dst[:, 2] - src[:, 2])
    out[:, 0] = src[:, 0] + b * (dst[:, 0] - src[:, 0])
    out[:, 1] = src[:, 1] + b * (dst[:, 1] - src[:, 1])
    return out.astype(np.float32, copy=False)


def _prepare_ml_unroll_rows_jk(
    verts_ijk_um: np.ndarray, *, n_rows: int, n_cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute per-row segment lengths and segment angles in the (k,j) plane.

    For each row (fixed i/AP slice), we treat the polyline along columns t as a 2D curve in
    (k, j) = (ML, DV). ML-unrolling is done by preserving segment lengths while shrinking
    local turn angles toward a straight line.
    """
    v = np.asarray(verts_ijk_um, dtype=np.float64)
    if v.shape != (int(n_rows * n_cols), 3):
        raise ValueError(f"Expected vertices shape {(int(n_rows * n_cols), 3)}, got {v.shape}.")
    if not np.isfinite(v).all():
        raise ValueError("Non-finite vertices in source mesh.")

    grid = v.reshape(int(n_rows), int(n_cols), 3)
    j = grid[:, :, 1]
    k = grid[:, :, 2]
    dj = np.diff(j, axis=1)
    dk = np.diff(k, axis=1)
    seg_len = np.hypot(dk, dj).astype(np.float64, copy=False)
    theta = np.arctan2(dj, dk).astype(np.float64, copy=False)
    return grid.astype(np.float32, copy=False), seg_len, theta


def _wrap_angle_rad(theta: np.ndarray) -> np.ndarray:
    x = np.asarray(theta, dtype=np.float64)
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float64, copy=False)


def _apply_ml_unroll(
    *,
    source_grid_ijk_um: np.ndarray,
    seg_len: np.ndarray,
    theta: np.ndarray,
    anchor_col: int,
    alpha: float,
) -> np.ndarray:
    """
    Unroll each row's (k,j) polyline by preserving segment lengths and shrinking curvature.

    - Anchor point at column `anchor_col` stays fixed.
    - Forward side (increasing column) straightens toward +k axis (angle -> 0).
    - Backward side straightens toward -k axis (angle -> pi).
    """
    grid = np.asarray(source_grid_ijk_um, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"Expected grid shape (n_rows,n_cols,3), got {grid.shape}.")
    n_rows, n_cols, _ = (int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2]))
    if not (0 <= int(anchor_col) < n_cols):
        raise ValueError(f"anchor_col out of bounds: {anchor_col} for n_cols={n_cols}.")

    lens = np.asarray(seg_len, dtype=np.float64)
    th0 = np.asarray(theta, dtype=np.float64)
    if lens.shape != (n_rows, n_cols - 1) or th0.shape != (n_rows, n_cols - 1):
        raise ValueError(f"seg_len/theta shape mismatch: {lens.shape} / {theta.shape} vs {(n_rows, n_cols - 1)}.")
    if not np.isfinite(lens).all() or not np.isfinite(th0).all():
        raise ValueError("Non-finite segment lengths/angles in ML unroll precompute.")

    a = float(np.clip(alpha, 0.0, 1.0))
    out = grid.copy()

    j0 = out[:, int(anchor_col), 1].astype(np.float64, copy=False)
    k0 = out[:, int(anchor_col), 2].astype(np.float64, copy=False)

    zeros = np.zeros((n_rows, 1), dtype=np.float64)

    # Forward reconstruction: points anchor..end.
    if int(anchor_col) < (n_cols - 1):
        th_f0 = th0[:, int(anchor_col) :].astype(np.float64, copy=False)
        dth_f = _wrap_angle_rad(th_f0[:, 1:] - th_f0[:, :-1])
        th_f = np.empty_like(th_f0, dtype=np.float64)
        th_f[:, 0] = (1.0 - a) * _wrap_angle_rad(th_f0[:, 0] - 0.0) + 0.0
        if th_f.shape[1] > 1:
            th_f[:, 1:] = th_f[:, 0:1] + np.cumsum((1.0 - a) * dth_f, axis=1)
        dk_f = lens[:, int(anchor_col) :] * np.cos(th_f)
        dj_f = lens[:, int(anchor_col) :] * np.sin(th_f)
        k_f = k0[:, None] + np.cumsum(np.concatenate([zeros, dk_f], axis=1), axis=1)
        j_f = j0[:, None] + np.cumsum(np.concatenate([zeros, dj_f], axis=1), axis=1)
        out[:, int(anchor_col) :, 2] = k_f
        out[:, int(anchor_col) :, 1] = j_f

    # Backward reconstruction: points anchor..0 (filled in reverse).
    if int(anchor_col) > 0:
        th_b0 = _wrap_angle_rad(th0[:, : int(anchor_col)] + np.pi)[:, ::-1].astype(np.float64, copy=False)
        ln_b = lens[:, : int(anchor_col)][:, ::-1].astype(np.float64, copy=False)
        dth_b = _wrap_angle_rad(th_b0[:, 1:] - th_b0[:, :-1])
        th_b = np.empty_like(th_b0, dtype=np.float64)
        th_b[:, 0] = (1.0 - a) * _wrap_angle_rad(th_b0[:, 0] - np.pi) + np.pi
        if th_b.shape[1] > 1:
            th_b[:, 1:] = th_b[:, 0:1] + np.cumsum((1.0 - a) * dth_b, axis=1)
        dk_rev = ln_b * np.cos(th_b)
        dj_rev = ln_b * np.sin(th_b)
        k_rev = k0[:, None] + np.cumsum(np.concatenate([zeros, dk_rev], axis=1), axis=1)
        j_rev = j0[:, None] + np.cumsum(np.concatenate([zeros, dj_rev], axis=1), axis=1)
        out[:, : int(anchor_col) + 1, 2] = k_rev[:, ::-1]
        out[:, : int(anchor_col) + 1, 1] = j_rev[:, ::-1]

    return out.astype(np.float32, copy=False)


def _fixed_camera_from_points_xyz(
    points_xyz: np.ndarray,
    *,
    azim_deg: float,
    elev_deg: float,
    dist_scale: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {pts.shape}.")
    center = np.nanmean(pts, axis=0)
    if not np.isfinite(center).all():
        raise ValueError("Non-finite center in source mesh.")
    span = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    diag = float(np.linalg.norm(span))
    if not np.isfinite(diag) or diag <= 0.0:
        raise ValueError("Invalid mesh spatial extent for camera setup.")

    dist = float(dist_scale) * diag
    if not np.isfinite(dist) or dist <= 0.0:
        raise ValueError(f"Invalid camera dist_scale={dist_scale}.")

    az = np.deg2rad(float(azim_deg))
    el = np.deg2rad(float(elev_deg))
    base = np.asarray(
        [
            np.cos(el) * np.cos(az),
            np.sin(el),
            np.cos(el) * np.sin(az),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(base)) <= 0.0:
        raise ValueError("Invalid camera direction.")
    base = base / float(np.linalg.norm(base))
    pos = center + base * dist
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(center[0]), float(center[1]), float(center[2])),
        (0.0, 1.0, 0.0),
    )


def animate_unfold_ap_ml_surface(
    *,
    outdir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    ref_t: float,
    band_frac: float,
    b_const: float,
    res_ijk_um: tuple[float, float, float],
    output_mp4: Path,
    fps: int,
    n_frames: int,
    phase1_frac: float,
    ml_only: bool,
    window_size: tuple[int, int],
    show_axes: bool,
    show_text: bool,
    cam_azim_deg: float,
    cam_elev_deg: float,
    cam_dist_scale: float,
    cam_view_angle_deg: float,
) -> tuple[Path, int]:
    import imageio.v2 as imageio
    import pyvista as pv

    w, h = (int(window_size[0]), int(window_size[1]))
    if w <= 0 or h <= 0:
        raise ValueError(f"--window-size must be positive, got {(w, h)}.")
    fps_i = int(fps)
    if fps_i <= 0:
        raise ValueError(f"--fps must be >0, got {fps}.")

    n_frames_i = int(n_frames)
    if n_frames_i < 2:
        raise ValueError(f"--n-frames must be >=2, got {n_frames}.")
    if bool(ml_only):
        n1 = n_frames_i
    else:
        n1, _n2 = _phase_layout(n_frames=n_frames_i, phase1_frac=float(phase1_frac))

    data = build_coronal_ap_ml_surface(
        outdir=Path(outdir),
        slice_i_min=int(slice_i_min),
        slice_i_max=int(slice_i_max),
        n_t=int(n_t),
        ref_t=float(ref_t),
        band_frac=float(band_frac),
        b_const=float(b_const),
        res_ijk_um=(float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2])),
    )

    source_ijk = np.asarray(data.vertices_ijk_um, dtype=np.float32)
    colors = np.asarray(data.rgb_u8, dtype=np.uint8)
    if colors.shape != source_ijk.shape:
        raise ValueError(f"Vertex color shape mismatch: expected {source_ijk.shape}, got {colors.shape}.")

    source_xyz = _ijk_um_to_xyz_um(source_ijk).astype(np.float32, copy=False)
    flip_y0 = float(np.nanmean(source_xyz[:, 1].astype(np.float64, copy=False)))
    if not np.isfinite(flip_y0):
        raise ValueError("Could not determine finite y0 for xz-plane flip.")
    source_xyz = _flip_xyz_about_xz_plane(source_xyz, y0=flip_y0)
    faces_pv = _faces_to_pyvista(data.faces)
    mesh = pv.PolyData(source_xyz, faces_pv)
    mesh.point_data["rgb"] = colors

    plotter = pv.Plotter(off_screen=True, window_size=(w, h))
    plotter.set_background("white")
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, show_scalar_bar=False, smooth_shading=False)
    if bool(show_axes):
        plotter.add_axes()
    plotter.camera_position = _fixed_camera_from_points_xyz(
        source_xyz,
        azim_deg=float(cam_azim_deg),
        elev_deg=float(cam_elev_deg),
        dist_scale=float(cam_dist_scale),
    )
    plotter.camera.view_angle = float(cam_view_angle_deg)
    plotter.reset_camera_clipping_range()

    n_rows = int(data.slice_keys.size)
    n_cols = int(data.t_grid.size)
    anchor_col = int(np.clip(np.rint(float(ref_t) * float(n_cols - 1)), 0, n_cols - 1))
    source_grid, seg_len, theta = _prepare_ml_unroll_rows_jk(source_ijk, n_rows=n_rows, n_cols=n_cols)
    ap_per_row = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    if ap_per_row.shape != (n_rows,):
        raise ValueError(f"ap_um_by_slice shape mismatch: expected {(n_rows,)}, got {ap_per_row.shape}.")
    anchor_row = int(n_rows // 2)
    ap_anchor = float(ap_per_row[anchor_row])
    src_i_by_row = np.asarray(source_grid[:, 0, 0], dtype=np.float64).reshape(-1)
    src_anchor_i = float(src_i_by_row[anchor_row])
    ap_target_i_by_row = src_anchor_i + (ap_per_row - ap_anchor)
    j_plane = float(np.nanmedian(source_grid[:, :, 1].astype(np.float64, copy=False)))
    if not np.isfinite(j_plane):
        raise ValueError("Could not determine finite DV collapse plane from source vertices.")

    output = Path(output_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    writer = imageio.get_writer(
        str(output),
        format="FFMPEG",
        mode="I",
        fps=fps_i,
        codec="libx264",
        pixelformat="yuv420p",
    )
    try:
        for frame_idx in range(n_frames_i):
            if bool(ml_only):
                alpha_den = max(1, n_frames_i - 1)
                alpha_ml = float(frame_idx) / float(alpha_den)
                beta_ap = 0.0
            else:
                alpha_ml, beta_ap = _phase_alphas(frame=frame_idx, n_frames=n_frames_i, n_phase1=n1)

            cur_grid = _apply_ml_unroll(
                source_grid_ijk_um=source_grid,
                seg_len=seg_len,
                theta=theta,
                anchor_col=anchor_col,
                alpha=float(alpha_ml),
            )
            if float(beta_ap) > 0.0:
                b = float(np.clip(beta_ap, 0.0, 1.0))
                cur_i = (1.0 - b) * np.asarray(cur_grid[:, 0, 0], dtype=np.float64) + b * ap_target_i_by_row
                cur_grid[:, :, 0] = cur_i[:, None]
                cur_grid[:, :, 1] = (1.0 - b) * np.asarray(cur_grid[:, :, 1], dtype=np.float64) + b * j_plane

            cur_xyz = _ijk_um_to_xyz_um(cur_grid.reshape(-1, 3)).astype(np.float32, copy=False)
            cur_xyz = _flip_xyz_about_xz_plane(cur_xyz, y0=flip_y0)
            mesh.points = cur_xyz

            if bool(show_text):
                if bool(ml_only):
                    phase_label = "ML unfolding"
                else:
                    phase_label = "ML unfolding" if frame_idx < n1 else "AP reparameterization"
                plotter.add_text(phase_label, name="phase_label", position="upper_left", font_size=13, color="black")

            frame = plotter.screenshot(return_img=True)
            if frame is None:
                raise RuntimeError("PyVista returned no frame image from screenshot().")
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
        plotter.close()

    return output, n_frames_i


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate unfolding of the coronal AP/ML mapping surface into AP/ML chart coordinates (ML first)."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Midsurface output directory containing coronal_midline_columns.csv and AP-axis artifacts.",
    )
    parser.add_argument("--slice-i-min", type=int, default=161, help="Minimum coronal slice_i to include (inclusive).")
    parser.add_argument("--slice-i-max", type=int, default=305, help="Maximum coronal slice_i to include (inclusive).")
    parser.add_argument("--n-t", type=int, default=257, help="Number of samples along each coronal midline curve.")
    parser.add_argument("--ref-t", type=float, default=0.5, help="Reference t_all in [0,1] used as ML origin anchor.")
    parser.add_argument("--band-frac", type=float, default=0.15, help="DTW Sakoe-Chiba band as fraction of n_t.")
    parser.add_argument("--b-const", type=float, default=0.25, help="Constant blue channel value in [0,1] for bivariate RGB.")
    parser.add_argument(
        "--res-ijk-um",
        type=float,
        nargs=3,
        default=(20.0, 20.0, 20.0),
        metavar=("RI", "RJ", "RK"),
        help="Voxel size in um for (i,j,k); overrides outdir/resolution_ds_ijk_um.npy.",
    )
    parser.add_argument(
        "--output-mp4",
        type=Path,
        default=None,
        help="Output MP4 path (default: <outdir>/ap_ml_unfold.mp4).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    parser.add_argument("--n-frames", type=int, default=180, help="Total number of animation frames.")
    parser.add_argument("--phase1-frac", type=float, default=0.5, help="Fraction of frames used for ML-only unfolding.")
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Render only ML unfolding across all frames (no AP flattening stage).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(1600, 1200),
        metavar=("W", "H"),
        help="Off-screen render size in pixels.",
    )
    parser.add_argument("--no-axes", action="store_true", help="Hide orientation axes in the render.")
    parser.add_argument("--no-text", action="store_true", help="Hide phase text labels in the render.")
    parser.add_argument("--cam-azim-deg", type=float, default=35.0, help="Camera azimuth in degrees (about +y).")
    parser.add_argument("--cam-elev-deg", type=float, default=15.0, help="Camera elevation in degrees (+y is up).")
    parser.add_argument("--cam-dist-scale", type=float, default=1.55, help="Camera distance multiplier on bbox diagonal.")
    parser.add_argument("--cam-view-angle-deg", type=float, default=40.0, help="Perspective view angle in degrees (larger = more depth).")
    args = parser.parse_args()

    out_mp4 = Path(args.output_mp4) if args.output_mp4 is not None else (Path(args.outdir) / "ap_ml_unfold.mp4")
    out_path, n_frames_written = animate_unfold_ap_ml_surface(
        outdir=Path(args.outdir),
        slice_i_min=int(args.slice_i_min),
        slice_i_max=int(args.slice_i_max),
        n_t=int(args.n_t),
        ref_t=float(args.ref_t),
        band_frac=float(args.band_frac),
        b_const=float(args.b_const),
        res_ijk_um=(float(args.res_ijk_um[0]), float(args.res_ijk_um[1]), float(args.res_ijk_um[2])),
        output_mp4=out_mp4,
        fps=int(args.fps),
        n_frames=int(args.n_frames),
        phase1_frac=float(args.phase1_frac),
        ml_only=bool(args.ml_only),
        window_size=(int(args.window_size[0]), int(args.window_size[1])),
        show_axes=not bool(args.no_axes),
        show_text=not bool(args.no_text),
        cam_azim_deg=float(args.cam_azim_deg),
        cam_elev_deg=float(args.cam_elev_deg),
        cam_dist_scale=float(args.cam_dist_scale),
        cam_view_angle_deg=float(args.cam_view_angle_deg),
    )
    print(f"Wrote: {out_path}")
    print(
        f"frames={n_frames_written} fps={int(args.fps)} "
        f"ml_only={bool(args.ml_only)} "
        f"phase1_frac={float(args.phase1_frac):.3f} "
        f"cam_azim_deg={float(args.cam_azim_deg):.1f} cam_elev_deg={float(args.cam_elev_deg):.1f} "
        f"cam_view_angle_deg={float(args.cam_view_angle_deg):.1f} "
        f"window={int(args.window_size[0])}x{int(args.window_size[1])}"
    )


if __name__ == "__main__":
    main()
