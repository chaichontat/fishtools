from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
import numpy as np

from fishtools.utils.plot import add_scale_bar


@dataclass(frozen=True)
class CoronalSurfaceProjectionParams:
    x2d: np.ndarray
    y2d: np.ndarray
    z2d: np.ndarray
    x3d: np.ndarray | None
    y3d: np.ndarray | None
    z3d: np.ndarray | None
    faces: np.ndarray
    tri_support: np.ndarray
    tri_neomeso: np.ndarray
    restrict_t_neomeso: bool
    gray_context: bool
    latlon: bool
    graticule: str
    lat_stride: int
    lon_stride: int
    max_lat_lines: int
    max_lon_lines: int
    vertex_support: np.ndarray | None
    vertex_neomeso: np.ndarray | None
    vertex_ap_um: np.ndarray | None
    vertex_ml_um: np.ndarray | None
    n_rows: int | None
    n_cols: int | None
    shade: bool
    shade_strength: float
    shade_elev_deg: float
    shade_azim_deg: float
    ordered_geometry: dict[str, np.ndarray] | None
    tri_alpha: np.ndarray | None = None


def native_proj_png_name(png_name: str) -> str:
    if png_name.endswith(".png"):
        return f"{png_name.removesuffix('.png')}_native_proj.png"
    return f"{png_name}_native_proj"


def _with_png_suffix(png_name: str, suffix: str) -> str:
    if not suffix:
        return png_name
    path = Path(png_name)
    if path.suffix == ".png":
        return str(path.with_name(f"{path.stem}{suffix}.png"))
    return str(path.with_name(f"{path.name}{suffix}"))


def _transform_mu_for_display(values: np.ndarray, *, scale: str, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape:
            raise ValueError(f"mu display mask shape {m.shape} does not match values shape {x.shape}")
        x[~m] = np.nan
    if str(scale) == "linear":
        return x
    if str(scale) == "log1p":
        finite = np.isfinite(x)
        x[finite] = np.log1p(np.maximum(x[finite], 0.0))
        return x
    if str(scale) == "log":
        finite = np.isfinite(x)
        if np.any(x[finite] <= 0):
            vmin = float(np.nanmin(x[finite]))
            raise ValueError(
                f"mu display scale='log' requires mu>0 on plotted support; found min={vmin:.3g}. "
                "Use --apml-mu-scale log1p instead."
            )
        x[finite] = np.log(x[finite])
        return x
    raise ValueError(f"Unknown mu display scale: {scale!r}")


def _mu_display_label(*, scale: str) -> str:
    if str(scale) == "linear":
        return "μ (counts)"
    if str(scale) == "log1p":
        return "log1p(μ) (counts)"
    if str(scale) == "log":
        return "log(μ) (counts)"
    raise ValueError(f"Unknown mu display scale: {scale!r}")


def _compute_mu_color_limits(
    values: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
    clip_percentiles: tuple[float, float],
) -> tuple[float | None, float | None]:
    if (vmin is None) != (vmax is None):
        raise ValueError("Provide both vmin and vmax, or neither.")
    if vmin is not None and vmax is not None:
        vmin_f = float(vmin)
        vmax_f = float(vmax)
        if not np.isfinite(vmin_f) or not np.isfinite(vmax_f) or vmin_f >= vmax_f:
            raise ValueError(f"Invalid mu vmin/vmax: {vmin} {vmax}")
        return vmin_f, vmax_f

    p_lo, p_hi = (float(clip_percentiles[0]), float(clip_percentiles[1]))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError(f"Invalid mu clip percentiles: {clip_percentiles!r} (expected 0 <= lo < hi <= 100)")
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    if not np.any(finite):
        return None, None
    lo = float(np.nanpercentile(flat[finite], p_lo))
    hi = float(np.nanpercentile(flat[finite], p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None, None
    return lo, hi


def _compute_pair_mu_limits(
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    *,
    support_mask: np.ndarray | None,
    vmin: float | None,
    vmax: float | None,
    clip_percentiles: tuple[float, float],
) -> tuple[float | None, float | None]:
    mu_lo_m = apply_support_mask_for_imshow(mu_lo, support_mask)
    mu_hi_m = apply_support_mask_for_imshow(mu_hi, support_mask)
    return _compute_mu_color_limits(
        np.stack([mu_lo_m, mu_hi_m], axis=0),
        vmin=vmin,
        vmax=vmax,
        clip_percentiles=clip_percentiles,
    )


def _fold_change_colorbar_log2_spec(*, ratio_min: float, ratio_max: float) -> tuple[list[float], list[str], float, float]:
    rmin = float(ratio_min)
    rmax = float(ratio_max)
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmax <= 0 or rmin >= rmax:
        raise ValueError(f"Invalid fold-change range: {ratio_min} {ratio_max} (expected 0 < min < max)")
    vmin = float(np.log2(rmin))
    vmax = float(np.log2(rmax))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Invalid fold-change range after log2: {ratio_min} {ratio_max}")

    i_lo = int(np.ceil(vmin))
    i_hi = int(np.floor(vmax))
    ticks = [float(t) for t in range(i_lo, i_hi + 1)]
    if not ticks:
        ticks = [vmin, 0.0, vmax] if (vmin < 0.0 < vmax) else [vmin, vmax]
    if vmin < ticks[0] - 1e-9:
        ticks = [vmin] + ticks
    if vmax > ticks[-1] + 1e-9:
        ticks = ticks + [vmax]
    labels = [f"{(2.0 ** t):.3g}×" for t in ticks]
    return ticks, labels, vmin, vmax


def _draw_coronal_surface_projection(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    params: CoronalSurfaceProjectionParams,
    cmap,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
    show_title: bool,
    show_scale_bar: bool,
    latlon_visibility: bool,
) -> mpl_colors.Normalize:
    x = np.asarray(params.x2d, dtype=np.float64).reshape(-1)
    y = np.asarray(params.y2d, dtype=np.float64).reshape(-1)
    z = np.asarray(params.z2d, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    tris = np.asarray(params.faces, dtype=np.int32)
    tri_support_b = np.asarray(params.tri_support, dtype=bool).reshape(-1)
    tri_neomeso_b = np.asarray(params.tri_neomeso, dtype=bool).reshape(-1)
    restrict_t_neomeso = bool(params.restrict_t_neomeso)
    gray_context = bool(params.gray_context)
    latlon = bool(params.latlon)
    graticule = str(params.graticule)
    lat_stride = int(params.lat_stride)
    lon_stride = int(params.lon_stride)
    max_lat_lines = int(params.max_lat_lines)
    max_lon_lines = int(params.max_lon_lines)
    vertex_support = params.vertex_support
    vertex_neomeso = params.vertex_neomeso
    vertex_ap_um = params.vertex_ap_um
    vertex_ml_um = params.vertex_ml_um
    n_rows = params.n_rows
    n_cols = params.n_cols
    shade = bool(params.shade)
    shade_strength = float(params.shade_strength)
    shade_elev_deg = float(params.shade_elev_deg)
    shade_azim_deg = float(params.shade_azim_deg)
    ordered_geometry = params.ordered_geometry
    tri_alpha = params.tri_alpha
    if x.shape != y.shape or x.shape != vals.shape:
        raise ValueError(f"x2d/y2d/values shape mismatch: {x.shape} {y.shape} {vals.shape}")
    if z.shape != x.shape:
        raise ValueError(f"z2d shape {z.shape} does not match x2d shape {x.shape}")
    x3 = None if params.x3d is None else np.asarray(params.x3d, dtype=np.float64).reshape(-1)
    y3 = None if params.y3d is None else np.asarray(params.y3d, dtype=np.float64).reshape(-1)
    z3 = None if params.z3d is None else np.asarray(params.z3d, dtype=np.float64).reshape(-1)
    if shade:
        if x3 is None or y3 is None or z3 is None:
            raise ValueError("shade=True requires x3d/y3d/z3d")
        if x3.shape != x.shape or y3.shape != x.shape or z3.shape != x.shape:
            raise ValueError("x3d/y3d/z3d shape mismatch with x2d/y2d/z2d")
        if not np.isfinite(shade_strength) or not (0.0 <= shade_strength <= 1.0):
            raise ValueError(f"shade_strength must be in [0,1], got {shade_strength}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {tris.shape}")
    if tri_support_b.shape != (tris.shape[0],) or tri_neomeso_b.shape != (tris.shape[0],):
        raise ValueError("triangle mask shape mismatch")
    if latlon:
        if n_rows is None or n_cols is None:
            raise ValueError("latlon overlay requires n_rows and n_cols")
        if vertex_support is None or vertex_neomeso is None:
            raise ValueError("latlon overlay requires vertex_support and vertex_neomeso masks")
        if int(n_rows) <= 0 or int(n_cols) <= 0:
            raise ValueError(f"Invalid n_rows/n_cols for latlon overlay: {n_rows} {n_cols}")
        if int(n_rows) * int(n_cols) != x.shape[0]:
            raise ValueError(
                f"latlon overlay expects n_rows*n_cols == n_vertices; got {n_rows}*{n_cols} != {x.shape[0]}"
            )
        if int(lat_stride) <= 0 or int(lon_stride) <= 0:
            raise ValueError(f"lat_stride and lon_stride must be >0, got {lat_stride} {lon_stride}")
        if int(max_lat_lines) < 2 or int(max_lon_lines) < 2:
            raise ValueError(f"max_lat_lines/max_lon_lines must be >=2, got {max_lat_lines} {max_lon_lines}")
        if str(graticule) not in {"apml", "param", "ijk"}:
            raise ValueError(f"graticule must be 'apml', 'param', or 'ijk', got {graticule!r}")
        if str(graticule) == "apml" and (vertex_ap_um is None or vertex_ml_um is None):
            raise ValueError("apml graticule requires vertex_ap_um and vertex_ml_um arrays")
        if str(graticule) == "ijk" and (x3 is None or z3 is None):
            raise ValueError("ijk graticule requires x3d (k_um) and z3d (i_um) coordinates")

    fig = ax.figure

    keep_tri = tri_support_b if (not restrict_t_neomeso or gray_context) else (tri_support_b & tri_neomeso_b)
    tri_idx = np.flatnonzero(keep_tri).astype(np.int32, copy=False)
    if tri_idx.size == 0:
        raise ValueError("No support triangles to plot")

    if ordered_geometry is None:
        tris_k = tris[tri_idx]
        tri_depth = np.nanmean(z[tris_k], axis=1)
        # Draw far-to-near so nearer triangles occlude. With our view basis, proj_z
        # increases towards the camera, so sort ascending (furthest first).
        order = np.argsort(tri_depth)
        tris_k = tris_k[order]
        tri_neomeso_k = tri_neomeso_b[tri_idx][order]
        polys = np.stack([x[tris_k], y[tris_k]], axis=2).astype(np.float64, copy=False)
        normals_unit = None
        normal_valid = None
        if tri_alpha is not None:
            raise ValueError("tri_alpha requires ordered_geometry (stable far-to-near rendering)")
    else:
        tris_k = np.asarray(ordered_geometry["tris"], dtype=np.int32)
        tri_neomeso_k = np.asarray(ordered_geometry["tri_neomeso"], dtype=bool).reshape(-1)
        polys = np.asarray(ordered_geometry["polys2d"], dtype=np.float64)
        normals_unit = np.asarray(ordered_geometry["normals_unit"], dtype=np.float64) if shade else None
        normal_valid = np.asarray(ordered_geometry["normal_valid"], dtype=bool).reshape(-1) if shade else None
        if tris_k.ndim != 2 or tris_k.shape[1] != 3:
            raise ValueError("ordered geometry tris must be (F,3)")
        if tri_neomeso_k.shape != (tris_k.shape[0],):
            raise ValueError("ordered geometry tri_neomeso shape mismatch")
        if polys.shape != (tris_k.shape[0], 3, 2):
            raise ValueError("ordered geometry polys2d shape mismatch")
        if shade and (normals_unit is None or normal_valid is None):
            raise ValueError("ordered geometry missing normals for shading")
        if shade and (normals_unit.shape != (tris_k.shape[0], 3) or normal_valid.shape != (tris_k.shape[0],)):
            raise ValueError("ordered geometry normals shape mismatch")
        if tri_alpha is not None:
            tri_alpha = np.asarray(tri_alpha, dtype=np.float64).reshape(-1)
            if tri_alpha.shape != (tris_k.shape[0],):
                raise ValueError(f"tri_alpha shape {tri_alpha.shape} does not match n_tris={tris_k.shape[0]}")
            if not np.all(np.isfinite(tri_alpha)) or np.any(tri_alpha < 0.0) or np.any(tri_alpha > 1.0):
                raise ValueError("tri_alpha must be finite and in [0,1]")

    if vmin is None or vmax is None:
        scale_keep = tri_support_b & (tri_neomeso_b if restrict_t_neomeso else True)
        scale_idx = np.flatnonzero(scale_keep).astype(np.int32, copy=False)
        if scale_idx.size == 0:
            scale_idx = tri_idx
        face_vals_scale = np.nanmean(vals[tris[scale_idx]], axis=1)
        finite = np.isfinite(face_vals_scale)
        if not np.any(finite):
            raise ValueError("No finite values available to autoscale coronal surface projection")
        vmin0 = float(np.nanmin(face_vals_scale[finite]))
        vmax0 = float(np.nanmax(face_vals_scale[finite]))
        if not np.isfinite(vmin0) or not np.isfinite(vmax0) or vmin0 >= vmax0:
            mid = float(np.nanmean(face_vals_scale[finite]))
            if not np.isfinite(mid):
                mid = 0.0
            vmin0 = mid - 1.0
            vmax0 = mid + 1.0
        vmin = vmin0
        vmax = vmax0

    norm = mpl_colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=False)

    if shade:
        face_vals = np.nanmean(vals[tris_k], axis=1)
        facecolors = np.asarray(cmap(norm(face_vals)), dtype=np.float64)
        if facecolors.ndim != 2 or facecolors.shape[1] != 4:
            raise ValueError(f"Colormap did not return RGBA array, got {facecolors.shape}")
        invalid_vals = ~np.isfinite(face_vals)
        if np.any(invalid_vals):
            facecolors[invalid_vals, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if restrict_t_neomeso and gray_context:
            gray_mask = ~tri_neomeso_k
            if np.any(gray_mask):
                facecolors[gray_mask, :] = np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float64)
        if tri_alpha is not None:
            # Blend towards gray outside the hull (keep extrapolated values; just make them look uncertain).
            w = np.clip(np.asarray(tri_alpha, dtype=np.float64).reshape(-1), 0.0, 1.0)[:, None]
            base_rgb = np.array([0.65, 0.65, 0.65], dtype=np.float64)[None, :]
            facecolors[:, :3] = np.clip(base_rgb * (1.0 - w) + facecolors[:, :3] * w, 0.0, 1.0)
            facecolors[:, 3] = 1.0

        elev = np.deg2rad(float(shade_elev_deg))
        azim = np.deg2rad(float(shade_azim_deg))
        cam_pos = np.array(
            [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
            dtype=np.float64,
        )
        view_dir = -cam_pos
        view_norm = float(np.linalg.norm(view_dir))
        if not np.isfinite(view_norm) or view_norm <= 0:
            raise ValueError("Failed to compute valid view direction for shading")
        view_dir = view_dir / view_norm
        if normals_unit is None or normal_valid is None:
            if x3 is None or y3 is None or z3 is None:
                raise ValueError("shade=True requires x3d/y3d/z3d")
            p0 = np.column_stack([x3[tris_k[:, 0]], y3[tris_k[:, 0]], z3[tris_k[:, 0]]]).astype(np.float64, copy=False)
            p1 = np.column_stack([x3[tris_k[:, 1]], y3[tris_k[:, 1]], z3[tris_k[:, 1]]]).astype(np.float64, copy=False)
            p2 = np.column_stack([x3[tris_k[:, 2]], y3[tris_k[:, 2]], z3[tris_k[:, 2]]]).astype(np.float64, copy=False)
            normals = np.cross(p1 - p0, p2 - p0)
            normal_norm = np.linalg.norm(normals, axis=1)
            normal_valid = np.isfinite(normal_norm) & (normal_norm > 0)
            normals_unit = np.zeros_like(normals)
            normals_unit[normal_valid] = normals[normal_valid] / normal_norm[normal_valid, None]
        d = np.clip(normals_unit @ view_dir, 0.0, 1.0)
        intensity = np.ones((tris_k.shape[0],), dtype=np.float64)
        intensity[normal_valid] = (1.0 - float(shade_strength)) + float(shade_strength) * d[normal_valid]
        facecolors[:, :3] = np.clip(facecolors[:, :3] * intensity[:, None], 0.0, 1.0)
        coll = PolyCollection(polys, facecolors=facecolors, edgecolors="none", linewidths=0.0, antialiased=False)
        ax.add_collection(coll)
    else:
        # In the native projection, triangles can overlap in 2D. tripcolor does not
        # guarantee a stable painter's order, which can produce z-fighting artifacts.
        # When ordered geometry is available, render with PolyCollection in far-to-near order.
        if ordered_geometry is None:
            if restrict_t_neomeso and gray_context:
                base_fc = np.tile(np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float64), (polys.shape[0], 1))
                base_coll = PolyCollection(polys, facecolors=base_fc, edgecolors="none", linewidths=0.0, antialiased=False)
                ax.add_collection(base_coll)
                tris_draw = tris_k[tri_neomeso_k]
            else:
                tris_draw = tris_k

            vals_v = np.ma.masked_invalid(vals)
            tri_invalid = np.any(~np.isfinite(vals[tris_draw]), axis=1)
            triang = mtri.Triangulation(x, y, tris_draw)
            triang.set_mask(tri_invalid)
            ax.tripcolor(
                triang,
                vals_v,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
                edgecolors="none",
                linewidth=0.0,
                antialiased=True,
                zorder=2,
            )
        else:
            face_vals = np.nanmean(vals[tris_k], axis=1)
            facecolors = np.asarray(cmap(norm(face_vals)), dtype=np.float64)
            tri_invalid = np.any(~np.isfinite(vals[tris_k]), axis=1) | (~np.isfinite(face_vals))
            if np.any(tri_invalid):
                facecolors[tri_invalid, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if restrict_t_neomeso and gray_context:
                base_fc = np.tile(np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float64), (polys.shape[0], 1))
                base_coll = PolyCollection(polys, facecolors=base_fc, edgecolors="none", linewidths=0.0, antialiased=False)
                ax.add_collection(base_coll)
                keep = np.asarray(tri_neomeso_k, dtype=bool).reshape(-1)
                if tri_alpha is not None:
                    w = np.clip(np.asarray(tri_alpha, dtype=np.float64).reshape(-1), 0.0, 1.0)[:, None]
                    base_rgb = np.array([0.65, 0.65, 0.65], dtype=np.float64)[None, :]
                    facecolors[:, :3] = np.clip(base_rgb * (1.0 - w) + facecolors[:, :3] * w, 0.0, 1.0)
                    facecolors[:, 3] = 1.0
                coll = PolyCollection(
                    polys[keep],
                    facecolors=facecolors[keep],
                    edgecolors="none",
                    linewidths=0.0,
                    antialiased=False,
                )
                ax.add_collection(coll)
            else:
                if tri_alpha is not None:
                    w = np.clip(np.asarray(tri_alpha, dtype=np.float64).reshape(-1), 0.0, 1.0)[:, None]
                    base_rgb = np.array([0.65, 0.65, 0.65], dtype=np.float64)[None, :]
                    facecolors[:, :3] = np.clip(base_rgb * (1.0 - w) + facecolors[:, :3] * w, 0.0, 1.0)
                    facecolors[:, 3] = 1.0
                coll = PolyCollection(polys, facecolors=facecolors, edgecolors="none", linewidths=0.0, antialiased=False)
                ax.add_collection(coll)

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()

    if latlon:
        n_r = int(n_rows)
        n_c = int(n_cols)
        xg = x.reshape(n_r, n_c)
        yg = y.reshape(n_r, n_c)
        support_g = np.asarray(vertex_support, dtype=bool).reshape(n_r, n_c)
        neomeso_g = np.asarray(vertex_neomeso, dtype=bool).reshape(n_r, n_c)
        keep_g = support_g & (neomeso_g if restrict_t_neomeso else True)
        trace_g = keep_g
        ml_um_g = None if vertex_ml_um is None else np.asarray(vertex_ml_um, dtype=np.float64).reshape(n_r, n_c)

        zg = z.reshape(n_r, n_c)
        if latlon_visibility:
            face_depth = np.nanmean(z[tris_k], axis=1)
            finite_fd = np.isfinite(face_depth)
            depth_vmin = float(np.nanmin(face_depth[finite_fd])) if np.any(finite_fd) else 0.0
            depth_vmax = float(np.nanmax(face_depth[finite_fd])) if np.any(finite_fd) else 1.0
            if not np.isfinite(depth_vmin) or not np.isfinite(depth_vmax) or depth_vmin >= depth_vmax:
                depth_vmin, depth_vmax = 0.0, 1.0

            vis_fig, vis_ax = plt.subplots(figsize=fig.get_size_inches(), dpi=fig.dpi)
            vis_ax.set_aspect("equal", adjustable="box")
            vis_ax.set_xlim(ax.get_xlim())
            vis_ax.set_ylim(ax.get_ylim())
            vis_ax.set_axis_off()
            vis_fig.patch.set_facecolor("black")
            vis_ax.set_facecolor("black")
            depth_norm = np.clip((face_depth - depth_vmin) / (depth_vmax - depth_vmin), 0.0, 1.0)
            depth_colors = np.zeros((face_depth.size, 4), dtype=np.float64)
            depth_colors[:, 0] = 0.01 + 0.98 * depth_norm
            depth_colors[:, 3] = 1.0
            vis_coll = PolyCollection(polys, facecolors=depth_colors, edgecolors="none", linewidths=0.0, antialiased=False)
            vis_ax.add_collection(vis_coll)
            vis_fig.canvas.draw()
            depth_img = np.asarray(vis_fig.canvas.buffer_rgba(), dtype=np.uint8)[..., 0].astype(np.float64) / 255.0
            depth_h, depth_w = depth_img.shape
            vis_trans = vis_ax.transData
            plt.close(vis_fig)

        def _smooth_xyzm(
            x_seg: np.ndarray, y_seg: np.ndarray, z_seg: np.ndarray, draw_seg: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            n = int(x_seg.size)
            if n < 3:
                return x_seg, y_seg, z_seg, np.asarray(draw_seg, dtype=bool)
            t = np.linspace(0.0, 1.0, n, dtype=np.float64)
            n_dense = max(n, 32 * n)
            t_dense = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)
            x_dense = np.interp(t_dense, t, x_seg)
            y_dense = np.interp(t_dense, t, y_seg)
            z_dense = np.interp(t_dense, t, z_seg)
            draw_dense = np.interp(t_dense, t, np.asarray(draw_seg, dtype=np.float64)) >= 0.5
            window = int(min(101, n_dense if (n_dense % 2 == 1) else (n_dense - 1)))
            if window < 3:
                return x_dense, y_dense, z_dense, draw_dense
            kernel = np.ones(window, dtype=np.float64) / float(window)
            left = window // 2
            right = window - 1 - left
            x_pad = np.pad(x_dense, (left, right), mode="edge")
            y_pad = np.pad(y_dense, (left, right), mode="edge")
            z_pad = np.pad(z_dense, (left, right), mode="edge")
            x_sm = np.convolve(x_pad, kernel, mode="valid")
            y_sm = np.convolve(y_pad, kernel, mode="valid")
            z_sm = np.convolve(z_pad, kernel, mode="valid")
            return x_sm, y_sm, z_sm, draw_dense

        def _plot_draw_segments(xline: np.ndarray, yline: np.ndarray, draw_mask: np.ndarray) -> None:
            draw_b = np.asarray(draw_mask, dtype=bool)
            if not np.any(draw_b):
                return
            idx = np.flatnonzero(draw_b)
            splits = np.flatnonzero(np.diff(idx) != 1)
            start = 0
            for split in splits:
                seg = idx[start : split + 1]
                if seg.size >= 2:
                    ax.plot(
                        xline[seg],
                        yline[seg],
                        color="k",
                        alpha=0.20,
                        linewidth=0.6,
                        solid_capstyle="round",
                        zorder=10,
                    )
                start = split + 1
            seg = idx[start:]
            if seg.size >= 2:
                ax.plot(
                    xline[seg],
                    yline[seg],
                    color="k",
                    alpha=0.20,
                    linewidth=0.6,
                    solid_capstyle="round",
                    zorder=10,
                )

        def _plot_masked_polyline(
            xline: np.ndarray,
            yline: np.ndarray,
            zline: np.ndarray,
            trace_mask: np.ndarray,
            draw_mask: np.ndarray,
        ) -> None:
            trace_b = np.asarray(trace_mask, dtype=bool)
            draw_b = np.asarray(draw_mask, dtype=bool)
            finite = np.isfinite(xline) & np.isfinite(yline) & np.isfinite(zline) & trace_b
            if not np.any(finite):
                return
            idx = np.flatnonzero(finite)
            splits = np.flatnonzero(np.diff(idx) != 1)
            start = 0
            for split in splits:
                seg = idx[start : split + 1]
                if seg.size >= 2:
                    x_seg, y_seg, z_seg, draw_seg = _smooth_xyzm(xline[seg], yline[seg], zline[seg], draw_b[seg])
                    if latlon_visibility:
                        pts = np.column_stack([x_seg, y_seg])
                        disp = vis_trans.transform(pts)
                        col = np.rint(disp[:, 0]).astype(np.int32)
                        row_from_bottom = np.rint(disp[:, 1]).astype(np.int32)
                        row = (depth_h - 1) - row_from_bottom
                        in_bounds = (col >= 0) & (col < depth_w) & (row >= 0) & (row < depth_h)
                        pix_depth = np.zeros((x_seg.size,), dtype=np.float64)
                        pix_depth[in_bounds] = depth_img[row[in_bounds], col[in_bounds]]
                        has_surface = pix_depth > 0.0
                        z_norm = np.clip((z_seg - depth_vmin) / (depth_vmax - depth_vmin), 0.0, 1.0)
                        z_vis = 0.01 + 0.98 * z_norm
                        tol = 0.01
                        vis = in_bounds & has_surface & (z_vis >= (pix_depth - tol)) & draw_seg
                        _plot_draw_segments(x_seg, y_seg, vis)
                    else:
                        _plot_draw_segments(x_seg, y_seg, draw_seg)
                start = split + 1
            seg = idx[start:]
            if seg.size >= 2:
                x_seg, y_seg, z_seg, draw_seg = _smooth_xyzm(xline[seg], yline[seg], zline[seg], draw_b[seg])
                if latlon_visibility:
                    pts = np.column_stack([x_seg, y_seg])
                    disp = vis_trans.transform(pts)
                    col = np.rint(disp[:, 0]).astype(np.int32)
                    row_from_bottom = np.rint(disp[:, 1]).astype(np.int32)
                    row = (depth_h - 1) - row_from_bottom
                    in_bounds = (col >= 0) & (col < depth_w) & (row >= 0) & (row < depth_h)
                    pix_depth = np.zeros((x_seg.size,), dtype=np.float64)
                    pix_depth[in_bounds] = depth_img[row[in_bounds], col[in_bounds]]
                    has_surface = pix_depth > 0.0
                    z_norm = np.clip((z_seg - depth_vmin) / (depth_vmax - depth_vmin), 0.0, 1.0)
                    z_vis = 0.01 + 0.98 * z_norm
                    tol = 0.01
                    vis = in_bounds & has_surface & (z_vis >= (pix_depth - tol)) & draw_seg
                    _plot_draw_segments(x_seg, y_seg, vis)
                else:
                    _plot_draw_segments(x_seg, y_seg, draw_seg)

        if str(graticule) == "param":
            # Parameter-space graticule: constant slice index and constant t-index.
            for rr in range(0, n_r, int(lat_stride)):
                _plot_masked_polyline(xg[rr, :], yg[rr, :], zg[rr, :], trace_g[rr, :], keep_g[rr, :])
            if (n_r - 1) % int(lat_stride) != 0:
                _plot_masked_polyline(
                    xg[n_r - 1, :], yg[n_r - 1, :], zg[n_r - 1, :], trace_g[n_r - 1, :], keep_g[n_r - 1, :]
                )
            for cc in range(0, n_c, int(lon_stride)):
                _plot_masked_polyline(xg[:, cc], yg[:, cc], zg[:, cc], trace_g[:, cc], keep_g[:, cc])
            if (n_c - 1) % int(lon_stride) != 0:
                _plot_masked_polyline(
                    xg[:, n_c - 1], yg[:, n_c - 1], zg[:, n_c - 1], trace_g[:, n_c - 1], keep_g[:, n_c - 1]
                )
        elif str(graticule) == "ijk":
            k_um_g = np.asarray(x3, dtype=np.float64).reshape(n_r, n_c)
            i_um_g = np.asarray(z3, dtype=np.float64).reshape(n_r, n_c)

            i_trace = i_um_g[trace_g]
            finite_i = np.isfinite(i_trace)
            if not np.any(finite_i):
                raise ValueError("ijk graticule: no finite i_um values on trace mask")
            i_min = float(np.nanmin(i_trace[finite_i]))
            i_max = float(np.nanmax(i_trace[finite_i]))
            n_lat = max(2, int(np.ceil(n_r / float(lat_stride))))
            n_lat = min(n_lat, int(max_lat_lines))
            i_levels = np.linspace(i_min, i_max, n_lat, dtype=np.float64)
            if not np.all(np.diff(i_levels) > 0):
                raise ValueError("ijk graticule: i_levels must be strictly increasing")

            k_trace = k_um_g[trace_g]
            finite_k = np.isfinite(k_trace)
            if not np.any(finite_k):
                raise ValueError("ijk graticule: no finite k_um values on trace mask")
            k_min = float(np.nanmin(k_trace[finite_k]))
            k_max = float(np.nanmax(k_trace[finite_k]))
            n_lon = max(2, int(np.ceil(n_c / float(lon_stride))))
            n_lon = min(n_lon, int(max_lon_lines))
            k_levels = np.linspace(k_min, k_max, n_lon, dtype=np.float64)
            if not np.all(np.diff(k_levels) > 0):
                raise ValueError("ijk graticule: k_levels must be strictly increasing")

            def _bilinear_sample(grid: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
                g = np.asarray(grid, dtype=np.float64)
                rr_f = np.asarray(rr, dtype=np.float64)
                cc_f = np.asarray(cc, dtype=np.float64)
                r0 = np.floor(rr_f).astype(np.int32)
                c0 = np.floor(cc_f).astype(np.int32)
                r1 = r0 + 1
                c1 = c0 + 1
                inb = (r0 >= 0) & (c0 >= 0) & (r1 < n_r) & (c1 < n_c)
                out = np.full(rr_f.shape, np.nan, dtype=np.float64)
                if not np.any(inb):
                    return out
                wr = rr_f[inb] - r0[inb].astype(np.float64)
                wc = cc_f[inb] - c0[inb].astype(np.float64)
                v00 = g[r0[inb], c0[inb]]
                v10 = g[r1[inb], c0[inb]]
                v01 = g[r0[inb], c1[inb]]
                v11 = g[r1[inb], c1[inb]]
                finite = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
                if not np.any(finite):
                    return out
                wr = wr[finite]
                wc = wc[finite]
                v00 = v00[finite]
                v10 = v10[finite]
                v01 = v01[finite]
                v11 = v11[finite]
                v0 = (1.0 - wr) * v00 + wr * v10
                v1 = (1.0 - wr) * v01 + wr * v11
                out_idx = np.flatnonzero(inb)[finite]
                out[out_idx] = (1.0 - wc) * v0 + wc * v1
                return out

            def _contour_segments(field: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
                tmp_fig, tmp_ax = plt.subplots(figsize=(1.0, 1.0), dpi=50)
                try:
                    cs = tmp_ax.contour(field, levels=levels)
                    segs: list[np.ndarray] = []
                    for level_segs in cs.allsegs:
                        for s in level_segs:
                            if s is None:
                                continue
                            s = np.asarray(s, dtype=np.float64)
                            if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] < 2:
                                continue
                            segs.append(s)
                    return segs
                finally:
                    plt.close(tmp_fig)

            i_field = np.where(trace_g, i_um_g, np.nan)
            k_field = np.where(trace_g, k_um_g, np.nan)
            for seg in _contour_segments(i_field, i_levels):
                rr = seg[:, 1]
                cc = seg[:, 0]
                xs = _bilinear_sample(xg, rr, cc)
                ys = _bilinear_sample(yg, rr, cc)
                zs = _bilinear_sample(zg, rr, cc)
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                _plot_masked_polyline(xs, ys, zs, finite, finite)
            for seg in _contour_segments(k_field, k_levels):
                rr = seg[:, 1]
                cc = seg[:, 0]
                xs = _bilinear_sample(xg, rr, cc)
                ys = _bilinear_sample(yg, rr, cc)
                zs = _bilinear_sample(zg, rr, cc)
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                _plot_masked_polyline(xs, ys, zs, finite, finite)
        else:
            # AP/ML-space graticule draped on the surface: iso-AP rows and iso-ML curves.
            if ml_um_g is None:
                raise ValueError("apml graticule requires vertex_ml_um array")
            for rr in range(0, n_r, int(lat_stride)):
                _plot_masked_polyline(xg[rr, :], yg[rr, :], zg[rr, :], trace_g[rr, :], keep_g[rr, :])
            if (n_r - 1) % int(lat_stride) != 0:
                _plot_masked_polyline(
                    xg[n_r - 1, :], yg[n_r - 1, :], zg[n_r - 1, :], trace_g[n_r - 1, :], keep_g[n_r - 1, :]
                )

            ml_trace = ml_um_g[trace_g]
            finite_ml = np.isfinite(ml_trace)
            if not np.any(finite_ml):
                raise ValueError("latlon graticule: no finite ML_um values on trace mask")
            ml_min = float(np.nanmin(ml_trace[finite_ml]))
            ml_max = float(np.nanmax(ml_trace[finite_ml]))
            n_lon = max(2, int(np.ceil(n_c / float(lon_stride))))
            ml_levels = np.linspace(ml_min, ml_max, n_lon, dtype=np.float64)
            ml_step = float(ml_levels[1] - ml_levels[0]) if n_lon >= 2 else float(ml_max - ml_min)
            ml_tol = 0.75 * ml_step if np.isfinite(ml_step) and ml_step > 0 else 50.0

            for ml0 in ml_levels:
                xline = np.full(n_r, np.nan, dtype=np.float64)
                yline = np.full(n_r, np.nan, dtype=np.float64)
                zline = np.full(n_r, np.nan, dtype=np.float64)
                tline = np.zeros(n_r, dtype=bool)
                dline = np.zeros(n_r, dtype=bool)
                for rr in range(n_r):
                    km = trace_g[rr, :]
                    if not np.any(km):
                        continue
                    d = np.abs(ml_um_g[rr, :] - float(ml0))
                    d = np.where(km & np.isfinite(d), d, np.inf)
                    cc = int(np.argmin(d))
                    if not np.isfinite(d[cc]) or float(d[cc]) > ml_tol:
                        continue
                    xline[rr] = xg[rr, cc]
                    yline[rr] = yg[rr, cc]
                    zline[rr] = zg[rr, cc]
                    tline[rr] = True
                    dline[rr] = bool(keep_g[rr, cc])
                _plot_masked_polyline(xline, yline, zline, tline, dline)

    if show_scale_bar:
        add_scale_bar(
            ax,
            500.0,
            "500 μm",
            location="lower left",
            pad=0.2,
            borderpad=0.3,
            sep=4,
            bar_thickness=3,
            font_size=13,
            color="black",
        )
    if show_title and title is not None:
        ax.set_title(title, loc="center", fontsize=17, pad=2)
    ax.set_axis_off()
    return norm


def plot_coronal_surface_projection(
    values: np.ndarray,
    *,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray | None,
    y3d: np.ndarray | None,
    z3d: np.ndarray | None,
    faces: np.ndarray,
    tri_support: np.ndarray,
    tri_neomeso: np.ndarray,
    restrict_t_neomeso: bool,
    gray_context: bool,
    latlon: bool,
    graticule: str,
    lat_stride: int,
    lon_stride: int,
    max_lat_lines: int,
    max_lon_lines: int,
    vertex_support: np.ndarray | None,
    vertex_neomeso: np.ndarray | None,
    vertex_ap_um: np.ndarray | None,
    vertex_ml_um: np.ndarray | None,
    n_rows: int | None,
    n_cols: int | None,
    shade: bool,
    shade_strength: float,
    shade_elev_deg: float,
    shade_azim_deg: float,
    ordered_geometry: dict[str, np.ndarray] | None,
    tri_alpha: np.ndarray | None = None,
    out_png: Path,
    title: str,
    cmap,
    cbar_label: str,
    cbar_ticks: list[float] | None,
    cbar_ticklabels: list[str] | None,
    vmin: float | None,
    vmax: float | None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    draw_params = CoronalSurfaceProjectionParams(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_support,
        tri_neomeso=tri_neomeso,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        latlon=latlon,
        graticule=graticule,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        max_lat_lines=max_lat_lines,
        max_lon_lines=max_lon_lines,
        vertex_support=vertex_support,
        vertex_neomeso=vertex_neomeso,
        vertex_ap_um=vertex_ap_um,
        vertex_ml_um=vertex_ml_um,
        n_rows=n_rows,
        n_cols=n_cols,
        shade=shade,
        shade_strength=shade_strength,
        shade_elev_deg=shade_elev_deg,
        shade_azim_deg=shade_azim_deg,
        ordered_geometry=ordered_geometry,
        tri_alpha=tri_alpha,
    )
    norm = _draw_coronal_surface_projection(
        ax,
        values,
        params=draw_params,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=str(title),
        show_title=True,
        show_scale_bar=True,
        latlon_visibility=True,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=str(cbar_label), fraction=0.03, pad=0.02, shrink=0.72)
    if cbar_ticks is not None:
        cbar.set_ticks([float(x) for x in cbar_ticks])
    if cbar_ticklabels is not None:
        cbar.set_ticklabels([str(x) for x in cbar_ticklabels])
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_coronal_surface_projection_triptych(
    values: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    panel_titles: tuple[str, str, str],
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray | None,
    y3d: np.ndarray | None,
    z3d: np.ndarray | None,
    faces: np.ndarray,
    tri_support: np.ndarray,
    tri_neomeso: np.ndarray,
    restrict_t_neomeso: bool,
    gray_context: bool,
    latlon: bool,
    graticule: str,
    lat_stride: int,
    lon_stride: int,
    max_lat_lines: int,
    max_lon_lines: int,
    vertex_support: np.ndarray | None,
    vertex_neomeso: np.ndarray | None,
    vertex_ap_um: np.ndarray | None,
    vertex_ml_um: np.ndarray | None,
    n_rows: int | None,
    n_cols: int | None,
    shade: bool,
    shade_strength: float,
    shade_elev_deg: float,
    shade_azim_deg: float,
    ordered_geometry: dict[str, np.ndarray] | None,
    out_png: Path,
    title: str,
    cmap,
    cbar_label: str,
    cbar_ticks: list[float] | None,
    cbar_ticklabels: list[str] | None,
    vmin: float | None,
    vmax: float | None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), dpi=200, constrained_layout=True)
    draw_params = CoronalSurfaceProjectionParams(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_support,
        tri_neomeso=tri_neomeso,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        latlon=latlon,
        graticule=graticule,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        max_lat_lines=max_lat_lines,
        max_lon_lines=max_lon_lines,
        vertex_support=vertex_support,
        vertex_neomeso=vertex_neomeso,
        vertex_ap_um=vertex_ap_um,
        vertex_ml_um=vertex_ml_um,
        n_rows=n_rows,
        n_cols=n_cols,
        shade=shade,
        shade_strength=shade_strength,
        shade_elev_deg=shade_elev_deg,
        shade_azim_deg=shade_azim_deg,
        ordered_geometry=ordered_geometry,
    )
    norms = []
    for ax, vals, t in zip(axes, values, panel_titles, strict=True):
        ax.axis("off")
        norm = _draw_coronal_surface_projection(
            ax,
            vals,
            params=draw_params,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=str(t),
            show_title=True,
            show_scale_bar=True,
            latlon_visibility=True,
        )
        norms.append(norm)

    fig.suptitle(str(title))
    sm = plt.cm.ScalarMappable(norm=norms[0], cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=str(cbar_label), fraction=0.03, pad=0.02, shrink=0.72)
    if cbar_ticks is not None:
        cbar.set_ticks([float(x) for x in cbar_ticks])
    if cbar_ticklabels is not None:
        cbar.set_ticklabels([str(x) for x in cbar_ticklabels])
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_apml_native_proj_montage(
    *,
    gene_values: list[tuple[str, np.ndarray]],
    out_png: Path,
    ncols: int,
    suptitle: str | None,
    scale_bar: str,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    faces: np.ndarray,
    tri_support: np.ndarray,
    tri_neomeso: np.ndarray,
    restrict_t_neomeso: bool,
    gray_context: bool,
    latlon: bool,
    graticule: str,
    lat_stride: int,
    lon_stride: int,
    max_lat_lines: int,
    max_lon_lines: int,
    vertex_support: np.ndarray,
    vertex_neomeso: np.ndarray,
    vertex_ap_um: np.ndarray,
    vertex_ml_um: np.ndarray,
    n_rows: int,
    n_cols: int,
    shade: bool,
    shade_strength: float,
    shade_elev_deg: float,
    shade_azim_deg: float,
    ordered_geometry: dict[str, np.ndarray] | None,
    tri_alpha: np.ndarray | None,
    cmap,
    cbar_label: str,
    cbar_ticks: list[float] | None,
    cbar_ticklabels: list[str] | None,
    vmin: float,
    vmax: float,
) -> None:
    if not gene_values:
        raise ValueError("No gene values provided for montage.")
    if str(scale_bar) not in {"first", "all", "none"}:
        raise ValueError(f"Invalid scale_bar mode: {scale_bar!r} (expected 'first', 'all', or 'none')")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(gene_values)
    ncols_i = max(1, int(ncols))
    ncols_i = min(ncols_i, n)
    nrows = (n + ncols_i - 1) // ncols_i

    panel_scale = 0.65
    fig_w = panel_scale * 6.4 * ncols_i
    fig_h = panel_scale * 4.8 * nrows + 0.6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols_i, figsize=(fig_w, fig_h), dpi=200)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    draw_params = CoronalSurfaceProjectionParams(
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_support,
        tri_neomeso=tri_neomeso,
        restrict_t_neomeso=bool(restrict_t_neomeso),
        gray_context=bool(gray_context),
        latlon=bool(latlon),
        graticule=str(graticule),
        lat_stride=int(lat_stride),
        lon_stride=int(lon_stride),
        max_lat_lines=int(max_lat_lines),
        max_lon_lines=int(max_lon_lines),
        vertex_support=vertex_support,
        vertex_neomeso=vertex_neomeso,
        vertex_ap_um=vertex_ap_um,
        vertex_ml_um=vertex_ml_um,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        shade=bool(shade),
        shade_strength=float(shade_strength),
        shade_elev_deg=float(shade_elev_deg),
        shade_azim_deg=float(shade_azim_deg),
        ordered_geometry=ordered_geometry,
        tri_alpha=tri_alpha,
    )

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= n:
            continue
        gene, vals_all = gene_values[i]
        show_sb = (scale_bar == "all") or (scale_bar == "first" and i == 0)
        _draw_coronal_surface_projection(
            ax,
            vals_all,
            params=draw_params,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            title=str(gene),
            show_title=True,
            show_scale_bar=bool(show_sb),
            latlon_visibility=True,
        )

    norm = mpl_colors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if suptitle is not None:
        fig.suptitle(str(suptitle), fontsize=20, y=0.985)
        rect_top = 0.965
    else:
        rect_top = 0.985

    fig.tight_layout(rect=(0.0, 0.0, 0.90, rect_top))
    cax = fig.add_axes([0.915, 0.14, 0.02, 0.72])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(str(cbar_label), fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    if cbar_ticks is not None:
        cbar.set_ticks([float(x) for x in cbar_ticks])
    if cbar_ticklabels is not None:
        cbar.set_ticklabels([str(x) for x in cbar_ticklabels])

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_apml_native_proj_montage(
    *,
    gene_values: list[tuple[str, np.ndarray]],
    out_png: Path,
    ncols: int,
    suptitle: str | None,
    scale_bar: str,
    x2d: np.ndarray,
    y2d: np.ndarray,
    z2d: np.ndarray,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    faces: np.ndarray,
    tri_support: np.ndarray,
    tri_neomeso: np.ndarray,
    restrict_t_neomeso: bool,
    gray_context: bool,
    latlon: bool,
    graticule: str,
    lat_stride: int,
    lon_stride: int,
    max_lat_lines: int,
    max_lon_lines: int,
    vertex_support: np.ndarray,
    vertex_neomeso: np.ndarray,
    vertex_ap_um: np.ndarray,
    vertex_ml_um: np.ndarray,
    n_rows: int,
    n_cols: int,
    shade: bool,
    shade_strength: float,
    shade_elev_deg: float,
    shade_azim_deg: float,
    ordered_geometry: dict[str, np.ndarray] | None,
    tri_alpha: np.ndarray | None = None,
    cmap,
    cbar_label: str,
    cbar_ticks: list[float] | None,
    cbar_ticklabels: list[str] | None,
    vmin: float,
    vmax: float,
) -> None:
    """Public wrapper for writing native-projection montages with shared colorbar."""
    _write_apml_native_proj_montage(
        gene_values=gene_values,
        out_png=out_png,
        ncols=ncols,
        suptitle=suptitle,
        scale_bar=scale_bar,
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_support,
        tri_neomeso=tri_neomeso,
        restrict_t_neomeso=restrict_t_neomeso,
        gray_context=gray_context,
        latlon=latlon,
        graticule=graticule,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        max_lat_lines=max_lat_lines,
        max_lon_lines=max_lon_lines,
        vertex_support=vertex_support,
        vertex_neomeso=vertex_neomeso,
        vertex_ap_um=vertex_ap_um,
        vertex_ml_um=vertex_ml_um,
        n_rows=n_rows,
        n_cols=n_cols,
        shade=shade,
        shade_strength=shade_strength,
        shade_elev_deg=shade_elev_deg,
        shade_azim_deg=shade_azim_deg,
        ordered_geometry=ordered_geometry,
        tri_alpha=tri_alpha,
        cmap=cmap,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        cbar_ticklabels=cbar_ticklabels,
        vmin=vmin,
        vmax=vmax,
    )


def apply_support_mask_for_imshow(matrix: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    if support_mask.shape != matrix.shape:
        raise ValueError(f"support_mask shape {support_mask.shape} does not match matrix shape {matrix.shape}")
    out = matrix.copy()
    out[~support_mask] = np.nan
    return out


def _mask_r_ap_ml_pair_by_support(
    *,
    mu_r_ap: np.ndarray,
    mu_r_ml: np.ndarray,
    ap_grid: np.ndarray,
    ml_grid: np.ndarray,
    apml_support_mask: np.ndarray,
    ap0: float,
    ml0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask r×AP and r×ML matrices using the AP/ML support mask at fixed ML0 / AP0.

    For the r×AP panel (ML fixed at ML0), we keep only AP columns whose (AP, nearest(ML0))
    position is supported. For the r×ML panel (AP fixed at AP0), we keep only ML columns
    whose (nearest(AP0), ML) position is supported.
    """
    mu_r_ap_m = np.asarray(mu_r_ap, dtype=float).copy()
    mu_r_ml_m = np.asarray(mu_r_ml, dtype=float).copy()
    ap = np.asarray(ap_grid, dtype=float).reshape(-1)
    ml = np.asarray(ml_grid, dtype=float).reshape(-1)
    support = np.asarray(apml_support_mask, dtype=bool)

    if support.shape != (ap.size, ml.size):
        raise ValueError(
            f"apml_support_mask shape {support.shape} does not match (len(ap_grid), len(ml_grid))={(ap.size, ml.size)}"
        )
    if mu_r_ap_m.ndim != 2 or mu_r_ap_m.shape[1] != ap.size:
        raise ValueError(f"mu_r_ap must be (n_r, len(ap_grid)); got {mu_r_ap_m.shape}")
    if mu_r_ml_m.ndim != 2 or mu_r_ml_m.shape[1] != ml.size:
        raise ValueError(f"mu_r_ml must be (n_r, len(ml_grid)); got {mu_r_ml_m.shape}")

    j_ml0 = int(np.nanargmin(np.abs(ml - float(ml0))))
    i_ap0 = int(np.nanargmin(np.abs(ap - float(ap0))))

    keep_ap = support[:, j_ml0].astype(bool, copy=False)
    keep_ml = support[i_ap0, :].astype(bool, copy=False)

    mu_r_ap_m[:, ~keep_ap] = np.nan
    mu_r_ml_m[:, ~keep_ml] = np.nan
    return mu_r_ap_m, mu_r_ml_m


def _want_png(only: set[str] | None, png_name: str) -> bool:
    return only is None or png_name in only


def _as_float_or_nan(value: object) -> float:
    if value is None:
        return np.nan
    try:
        out = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(out):
        return np.nan
    return out
