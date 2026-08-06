from __future__ import annotations

import argparse
from concurrent.futures import Future
import importlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.transforms import IdentityTransform

from gam.native_surface_plotting import plot_coronal_surface_projection


def test_plot_coronal_surface_projection_perspective_smoke(tmp_path: Path) -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    x2d = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    y2d = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    z2d = np.asarray([0.0, 0.0, 0.2, 0.2], dtype=np.float64)
    x3d = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    y3d = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    z3d = np.asarray([0.0, 0.1, 0.2, 0.1], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    tri_mask = np.asarray([True, True], dtype=bool)
    support = np.asarray([True, True, True, True], dtype=bool)

    out_png = tmp_path / "perspective.png"
    plot_coronal_surface_projection(
        values,
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_mask,
        tri_neomeso=tri_mask,
        restrict_t_neomeso=True,
        gray_context=False,
        latlon=False,
        graticule="ijk",
        lat_stride=2,
        lon_stride=2,
        max_lat_lines=2,
        max_lon_lines=2,
        vertex_support=support,
        vertex_neomeso=support,
        vertex_ap_um=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        vertex_ml_um=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        n_rows=2,
        n_cols=2,
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=0.0,
        shade_azim_deg=-95.0,
        camera_elev_deg=0.0,
        camera_azim_deg=-95.0,
        camera_roll_deg=180.0,
        proj_type="persp",
        focal_length=1.0,
        ordered_geometry=None,
        out_png=out_png,
        title="perspective",
        cmap=importlib.import_module("matplotlib.pyplot").get_cmap("viridis"),
        cbar_label="value",
        cbar_ticks=None,
        cbar_ticklabels=None,
        vmin=0.0,
        vmax=3.0,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_coronal_surface_projection_perspective_with_latlon_smoke(tmp_path: Path) -> None:
    values = np.arange(9, dtype=np.float64)
    x2d = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y2d = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    z2d = np.asarray([0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4], dtype=np.float64)
    x3d = x2d.copy()
    y3d = y2d.copy()
    z3d = np.asarray([0.0, 0.1, 0.2, 0.1, 0.3, 0.4, 0.2, 0.3, 0.5], dtype=np.float64)
    faces = np.asarray(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]],
        dtype=np.int32,
    )
    tri_mask = np.asarray([True] * len(faces), dtype=bool)
    support = np.asarray([True] * 9, dtype=bool)
    neomeso = np.asarray([True, True, False, True, True, True, False, True, True], dtype=bool)

    out_png = tmp_path / "perspective_latlon.png"
    plot_coronal_surface_projection(
        values,
        x2d=x2d,
        y2d=y2d,
        z2d=z2d,
        x3d=x3d,
        y3d=y3d,
        z3d=z3d,
        faces=faces,
        tri_support=tri_mask,
        tri_neomeso=tri_mask,
        restrict_t_neomeso=True,
        gray_context=False,
        latlon=True,
        graticule="ijk",
        lat_stride=1,
        lon_stride=1,
        max_lat_lines=2,
        max_lon_lines=2,
        vertex_support=support,
        vertex_neomeso=neomeso,
        vertex_ap_um=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64),
        vertex_ml_um=np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64),
        n_rows=3,
        n_cols=3,
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=0.0,
        shade_azim_deg=-95.0,
        camera_elev_deg=0.0,
        camera_azim_deg=-95.0,
        camera_roll_deg=180.0,
        proj_type="persp",
        focal_length=1.0,
        ordered_geometry=None,
        out_png=out_png,
        title="perspective-latlon",
        cmap=importlib.import_module("matplotlib.pyplot").get_cmap("viridis"),
        cbar_label="value",
        cbar_ticks=None,
        cbar_ticklabels=None,
        vmin=0.0,
        vmax=3.0,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_perspective_tri_alpha_uses_ordered_geometry_order(monkeypatch) -> None:
    mod = importlib.import_module("gam.native_surface_plotting")

    captured: list[np.ndarray] = []

    class _FakePoly3DCollection:
        def __init__(self, *_args, **kwargs):
            self._alpha = float(kwargs.get("alpha", 1.0))

        def set_facecolor(self, colors):
            if isinstance(colors, str):
                rgba = importlib.import_module("matplotlib.colors").to_rgba(colors)
                arr = np.asarray(rgba, dtype=np.float64)
                arr[3] = self._alpha
                captured.append(arr)
                return None
            captured.append(np.asarray(colors, dtype=np.float64))

    class _FakeAx:
        def add_collection3d(self, *_args, **_kwargs):
            return None

        def set_xlim(self, *_args, **_kwargs):
            return None

        def set_ylim(self, *_args, **_kwargs):
            return None

        def set_zlim(self, *_args, **_kwargs):
            return None

        def set_box_aspect(self, *_args, **_kwargs):
            return None

        def view_init(self, *_args, **_kwargs):
            return None

        def set_proj_type(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        def plot(self, *_args, **_kwargs):
            return None

        def text(self, *_args, **_kwargs):
            return None

        def set_axis_off(self):
            return None

    monkeypatch.setattr(mod, "Poly3DCollection", _FakePoly3DCollection)

    params = mod.CoronalSurfaceProjectionParams(
        x2d=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        y2d=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        z2d=np.asarray([0.0, 0.0, 0.2, 0.2], dtype=np.float64),
        x3d=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        y3d=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        z3d=np.asarray([0.0, 0.1, 0.2, 0.1], dtype=np.float64),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        tri_support=np.asarray([True, True], dtype=bool),
        tri_neomeso=np.asarray([True, True], dtype=bool),
        restrict_t_neomeso=False,
        gray_context=False,
        latlon=False,
        graticule="ijk",
        lat_stride=1,
        lon_stride=1,
        max_lat_lines=2,
        max_lon_lines=2,
        vertex_support=np.asarray([True, True, True, True], dtype=bool),
        vertex_neomeso=np.asarray([True, True, True, True], dtype=bool),
        vertex_ap_um=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        vertex_ml_um=np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        n_rows=2,
        n_cols=2,
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=0.0,
        shade_azim_deg=0.0,
        camera_elev_deg=0.0,
        camera_azim_deg=-95.0,
        camera_roll_deg=180.0,
        proj_type="persp",
        focal_length=1.0,
        ordered_geometry={
            "tris": np.asarray([[0, 2, 3], [0, 1, 2]], dtype=np.int32),
            "tri_neomeso": np.asarray([True, True], dtype=bool),
        },
        tri_alpha=np.asarray([1.0, 0.0], dtype=np.float64),
    )

    mod._draw_coronal_surface_projection_perspective(
        _FakeAx(),
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        params=params,
        cmap=importlib.import_module("matplotlib.pyplot").get_cmap("viridis"),
        vmin=0.0,
        vmax=3.0,
        title=None,
        show_title=False,
        show_scale_bar=False,
    )

    assert captured
    facecolors = captured[-1]
    np.testing.assert_allclose(facecolors[1, :3], np.asarray([0.65, 0.65, 0.65]), atol=1e-6)
    assert not np.allclose(facecolors[0, :3], np.asarray([0.65, 0.65, 0.65]))


def test_perspective_renderer_reapplies_surface_limits_after_latlon(monkeypatch) -> None:
    mod = importlib.import_module("gam.native_surface_plotting")

    class _FakePoly3DCollection:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_facecolor(self, _colors):
            return None

    class _FakeAx:
        def __init__(self) -> None:
            self._xlim = (-999.0, 999.0)
            self._ylim = (-999.0, 999.0)
            self._zlim = (-999.0, 999.0)

        def add_collection3d(self, *_args, **_kwargs):
            return None

        def set_xlim(self, lo, hi):
            self._xlim = (float(lo), float(hi))

        def set_ylim(self, lo, hi):
            self._ylim = (float(lo), float(hi))

        def set_zlim(self, lo, hi):
            self._zlim = (float(lo), float(hi))

        def set_box_aspect(self, *_args, **_kwargs):
            return None

        def view_init(self, *_args, **_kwargs):
            return None

        def set_proj_type(self, *_args, **_kwargs):
            return None

        def get_proj(self):
            return np.eye(4, dtype=np.float64)

        def plot(self, *_args, **_kwargs):
            self._xlim = (-5.0, 5.0)
            self._ylim = (-6.0, 6.0)
            self._zlim = (-7.0, 7.0)
            return None

        def text(self, *_args, **_kwargs):
            return None

        def set_axis_off(self):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

        @property
        def figure(self):
            class _FakeCanvas:
                def draw(self):
                    return None

                def buffer_rgba(self):
                    out = np.zeros((10, 10, 4), dtype=np.uint8)
                    out[..., 0] = 255
                    out[..., 1] = 255
                    out[..., 2] = 255
                    out[..., 3] = 255
                    return out

            class _FakeFigure:
                dpi = 100.0

                canvas = _FakeCanvas()

                def get_size_inches(self):
                    return np.asarray([2.0, 2.0], dtype=np.float64)

            return _FakeFigure()

        def get_zorder(self):
            return 1

    monkeypatch.setattr(mod, "Poly3DCollection", _FakePoly3DCollection)
    monkeypatch.setattr(mod, "_rasterize_projected_face_depth", lambda **_kwargs: (np.full((10, 10), -1.0), IdentityTransform()))
    monkeypatch.setattr(mod, "_visible_projected_segments", lambda line3d, **_kwargs: [np.asarray(line3d, dtype=np.float64)])
    monkeypatch.setattr(mod, "_offset_lines_towards_camera", lambda lines, **_kwargs: lines)

    params = mod.CoronalSurfaceProjectionParams(
        x2d=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        y2d=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        z2d=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        x3d=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        y3d=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        z3d=np.asarray([0.0, 0.0, 0.2, 0.2], dtype=np.float64),
        faces=np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
        tri_support=np.asarray([True, True], dtype=bool),
        tri_neomeso=np.asarray([True, True], dtype=bool),
        restrict_t_neomeso=False,
        gray_context=False,
        latlon=True,
        graticule="param",
        lat_stride=1,
        lon_stride=1,
        max_lat_lines=2,
        max_lon_lines=2,
        vertex_support=np.asarray([True, True, True, True], dtype=bool),
        vertex_neomeso=np.asarray([True, True, True, True], dtype=bool),
        vertex_ap_um=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        vertex_ml_um=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        n_rows=2,
        n_cols=2,
        shade=False,
        shade_strength=0.75,
        shade_elev_deg=0.0,
        shade_azim_deg=0.0,
        camera_elev_deg=0.0,
        camera_azim_deg=-95.0,
        camera_roll_deg=180.0,
        proj_type="persp",
        focal_length=1.0,
        ordered_geometry=None,
    )

    ax = _FakeAx()
    mod._draw_coronal_surface_projection_perspective(
        ax,
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        params=params,
        cmap=importlib.import_module("matplotlib.pyplot").get_cmap("viridis"),
        vmin=0.0,
        vmax=3.0,
        title=None,
        show_title=False,
        show_scale_bar=False,
    )

    assert ax._xlim != (-5.0, 5.0)
    assert ax._ylim != (-6.0, 6.0)
    assert ax._zlim != (-7.0, 7.0)


def test_rasterize_projected_face_depth_preserves_projected_depth_scale() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    depth_img, _vis_trans = mod._rasterize_projected_face_depth(
        face_polys2d=np.asarray([[[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]]], dtype=np.float64),
        face_depth=np.asarray([-1.25], dtype=np.float64),
        xlim=(0.0, 6.0),
        ylim=(0.0, 6.0),
        fig_size_inches=np.asarray([2.0, 2.0], dtype=np.float64),
        dpi=50.0,
    )

    finite = np.isfinite(depth_img)
    assert np.any(finite)
    assert float(np.nanmin(depth_img[finite])) < -1.0
    np.testing.assert_allclose(float(np.nanmedian(depth_img[finite])), -1.25, atol=0.02)


def test_rasterize_projected_face_depth_prefers_nearest_overlapping_triangle() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    depth_img, _vis_trans = mod._rasterize_projected_face_depth(
        face_polys2d=np.asarray(
            [
                [[1.0, 1.0], [6.0, 1.0], [1.0, 6.0]],
                [[1.0, 1.0], [6.0, 1.0], [1.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        face_depth=np.asarray([-1.20, -0.80], dtype=np.float64),
        xlim=(0.0, 7.0),
        ylim=(0.0, 7.0),
        fig_size_inches=np.asarray([2.0, 2.0], dtype=np.float64),
        dpi=50.0,
    )

    finite = np.isfinite(depth_img)
    assert np.any(finite)
    np.testing.assert_allclose(float(np.nanmedian(depth_img[finite])), -1.20, atol=0.02)


def test_perspective_visible_mask_prefers_more_negative_depth() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    mask = mod._perspective_visible_mask(
        np.asarray([-1.03, -0.97, -1.01], dtype=np.float64),
        np.asarray([-1.00, -1.00, -1.00], dtype=np.float64),
        tol=0.01,
    )
    np.testing.assert_array_equal(mask, np.asarray([True, False, True], dtype=bool))


def test_visible_projected_segments_split_around_hidden_interval() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    depth_img = np.full((8, 8), np.nan, dtype=np.float64)
    row = (depth_img.shape[0] - 1) - 2
    depth_img[row, 0:6] = -1.0
    line3d = np.asarray(
        [
            [0.0, 2.0, -1.05],
            [1.0, 2.0, -1.05],
            [2.0, 2.0, -0.95],
            [3.0, 2.0, -0.95],
            [4.0, 2.0, -1.05],
            [5.0, 2.0, -1.05],
        ],
        dtype=np.float64,
    )

    segs = mod._visible_projected_segments(
        line3d,
        M=np.eye(4, dtype=np.float64),
        vis_trans=IdentityTransform(),
        depth_img=depth_img,
        tol=0.01,
    )

    assert len(segs) == 2
    np.testing.assert_allclose(segs[0][:, 0], np.asarray([0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(segs[1][:, 0], np.asarray([4.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(segs[0][:, 1], np.asarray([2.0, 2.0], dtype=np.float64))


def test_build_surface_graticule_segments_preserves_mask_gap() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 0.4],
            [3.0, 0.0, 0.6],
            [4.0, 0.0, 0.8],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.2],
            [2.0, 1.0, 0.4],
            [3.0, 1.0, 0.6],
            [4.0, 1.0, 0.8],
        ],
        dtype=np.float64,
    )
    support = np.asarray([True, True, False, True, True, True, True, False, True, True], dtype=bool)

    segs = mod._build_surface_graticule_segments_3d(
        xyz=xyz,
        x3=xyz[:, 0],
        z3=xyz[:, 2],
        vertex_support=support,
        vertex_neomeso=np.ones_like(support, dtype=bool),
        vertex_ml_um=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        n_rows=2,
        n_cols=5,
        restrict_t_neomeso=False,
        graticule="param",
        lat_stride=10,
        lon_stride=10,
        max_lat_lines=2,
        max_lon_lines=2,
    )

    assert len(segs) == 6
    row0 = [seg for seg in segs if np.allclose(seg[:, 1], 0.0, atol=0.1)]
    assert len(row0) == 2
    assert any(float(np.max(seg[:, 0])) < 1.1 for seg in row0)
    assert any(float(np.min(seg[:, 0])) > 2.9 for seg in row0)


def test_compute_apml_percentile_bounds_and_tri_alpha() -> None:
    module = runpy.run_path(str(Path("scripts/gam/plot_significant_gams.py")))
    compute_apml_percentile_bounds = module["compute_apml_percentile_bounds"]
    build_native_tri_alpha = module["build_native_tri_alpha"]

    bounds = compute_apml_percentile_bounds(
        ap_um=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        ml_um=np.asarray([100.0, 110.0, 120.0, 130.0, 140.0], dtype=np.float64),
        percentile_range=(20.0, 80.0),
    )
    assert bounds == (8.0, 32.0, 108.0, 132.0)

    tri_alpha = build_native_tri_alpha(
        ordered_geom={"tris": np.asarray([[0, 1, 2], [3, 4, 4]], dtype=np.int32)},
        ap_um_flat=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        ml_um_flat=np.asarray([100.0, 110.0, 120.0, 130.0, 140.0], dtype=np.float64),
        percentile_bounds=bounds,
    )

    np.testing.assert_array_equal(tri_alpha, np.asarray([1.0, 0.0], dtype=np.float64))


def test_smooth_neomeso_mask_tall_parametric_reduces_rowwise_t_range_kink() -> None:
    mod = importlib.import_module("ccf.transforms")
    orig = np.asarray(
        [
            [False, True, True, True, True, False, False],
            [False, False, True, True, True, True, False],
            [False, True, True, True, True, False, False],
            [False, False, True, True, True, True, False],
            [False, True, True, True, True, False, False],
        ],
        dtype=bool,
    )
    smoothed = mod.smooth_neomeso_mask_tall_parametric(
        support_mask_tall=np.ones_like(orig, dtype=bool),
        neomeso_mask_tall=orig,
        smoothing=2.0,
    )

    def _start_idx(mask: np.ndarray) -> np.ndarray:
        out = np.full((mask.shape[0],), np.nan, dtype=np.float64)
        for row in range(mask.shape[0]):
            idx = np.flatnonzero(mask[row])
            if idx.size:
                out[row] = float(idx[0])
        return out

    start_orig = _start_idx(orig)
    start_smooth = _start_idx(smoothed)
    tv_orig = float(np.nansum(np.abs(np.diff(start_orig))))
    tv_smooth = float(np.nansum(np.abs(np.diff(start_smooth))))

    assert tv_smooth < tv_orig
    assert bool(np.all(smoothed[:, 2]))


def test_smooth_neomeso_tri_mask_parametric_uses_centroid_boundary() -> None:
    mod = importlib.import_module("ccf.transforms")
    tri_mask = mod.smooth_neomeso_tri_mask_parametric(
        faces=np.asarray([[0, 1, 4], [1, 5, 4]], dtype=np.int32),
        tri_support=np.asarray([True, True], dtype=bool),
        n_cols=3,
        start_fit=np.asarray([0.6, 0.6], dtype=np.float64),
        end_fit=np.asarray([1.1, 1.1], dtype=np.float64),
    )

    np.testing.assert_array_equal(tri_mask, np.asarray([True, False], dtype=bool))


def test_prepare_ordered_surface_geometry_clipped_to_parametric_band_creates_boundary_vertices() -> None:
    mod = importlib.import_module("ccf.transforms")
    geom = mod.prepare_ordered_surface_geometry_clipped_to_parametric_band(
        x2d=np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64),
        y2d=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
        z2d=np.zeros((6,), dtype=np.float64),
        x3d=np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64),
        y3d=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
        z3d=np.zeros((6,), dtype=np.float64),
        faces=np.asarray([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32),
        tri_support=np.asarray([True, True, True, True], dtype=bool),
        n_cols=3,
        start_fit=np.asarray([0.6, 0.6], dtype=np.float64),
        end_fit=np.asarray([1.4, 1.4], dtype=np.float64),
    )

    polys2d = [np.asarray(poly, dtype=np.float64) for poly in geom["polys2d"]]
    assert any(poly.shape[0] > 3 for poly in polys2d)
    xs = np.concatenate([poly[:, 0] for poly in polys2d])
    assert np.any(np.isclose(xs, 0.6, atol=1e-6))
    assert np.any(np.isclose(xs, 1.4, atol=1e-6))


def test_build_surface_graticule_segments_3d_uses_continuous_neomeso_boundary() -> None:
    mod = importlib.import_module("gam.native_surface_plotting")
    xyz = np.column_stack(
        [
            np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.zeros((6,), dtype=np.float64),
        ]
    )
    segments = mod._build_surface_graticule_segments_3d(
        xyz=xyz,
        x3=xyz[:, 0],
        z3=xyz[:, 2],
        vertex_support=np.ones((6,), dtype=bool),
        vertex_neomeso=np.asarray([False, True, False, False, True, False], dtype=bool),
        vertex_ml_um=np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float64),
        n_rows=2,
        n_cols=3,
        restrict_t_neomeso=True,
        graticule="param",
        lat_stride=1,
        lon_stride=1,
        max_lat_lines=4,
        max_lon_lines=4,
        neomeso_start_fit=np.asarray([0.6, 0.6], dtype=np.float64),
        neomeso_end_fit=np.asarray([1.4, 1.4], dtype=np.float64),
    )

    assert segments
    xs = np.concatenate([seg[:, 0] for seg in segments])
    assert np.any(np.isclose(xs, 0.6, atol=0.05))
    assert np.any(np.isclose(xs, 1.4, atol=0.05))


class _FakePredictor:
    def predict_link(self, fit, newdata: pd.DataFrame) -> np.ndarray:
        animal = str(newdata["animal"].iloc[0])
        ab = str(newdata["ab"].iloc[0])
        animal_eff = 1.0 if animal == "A" else 3.0
        ab_eff = 100.0 if ab == "A.batch1" else 200.0
        return np.full((len(newdata),), 2.0 + animal_eff + ab_eff, dtype=np.float64)

    def predict_terms_se(self, fit, newdata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        animal = str(newdata["animal"].iloc[0])
        ab = str(newdata["ab"].iloc[0])
        animal_eff = 1.0 if animal == "A" else 3.0
        ab_eff = 100.0 if ab == "A.batch1" else 200.0
        terms = np.tile(np.asarray([[animal_eff, ab_eff, 2.0]], dtype=np.float64), (len(newdata), 1))
        se = np.ones_like(terms, dtype=np.float64)
        return terms, se, ["s(animal)", "s(ab)", "s(AP_um)"]


class _OverflowPredictor:
    def predict_link(self, fit, newdata: pd.DataFrame) -> np.ndarray:
        animal = str(newdata["animal"].iloc[0])
        animal_eff = 710.4 if animal == "A" else -1000.0
        ab_eff = 0.0 if str(newdata["ab"].iloc[0]).startswith(f"{animal}.") else 100.0
        return np.full((len(newdata),), animal_eff + ab_eff, dtype=np.float64)

    def predict_terms_se(self, fit, newdata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        animal = str(newdata["animal"].iloc[0])
        animal_eff = 710.4 if animal == "A" else -1000.0
        ab_eff = 0.0 if str(newdata["ab"].iloc[0]).startswith(f"{animal}.") else 100.0
        terms = np.tile(np.asarray([[animal_eff, ab_eff]], dtype=np.float64), (len(newdata), 1))
        se = np.ones_like(terms, dtype=np.float64)
        return terms, se, ["s(animal)", "s(ab)"]


def test_predict_response_shrunk_marginalizing_animal_excludes_ab() -> None:
    module = runpy.run_path(str(Path("scripts/gam/plot_significant_gams.py")))
    predict_response_shrunk_marginalizing_animal = module["predict_response_shrunk_marginalizing_animal"]

    newdata = pd.DataFrame(
        {
            "animal": pd.Categorical(["A"], categories=["A", "B"]),
            "ab": pd.Categorical(["A.batch1"], categories=["A.batch1", "B.batch1"]),
        }
    )
    out = predict_response_shrunk_marginalizing_animal(
        predictor=_FakePredictor(),
        fit=None,
        newdata=newdata,
        animal_levels=["A", "B"],
        ab_levels=["A.batch1", "B.batch1"],
        batch_ref="batch1",
        shrink="none",
        hard_z=1.96,
    )

    np.testing.assert_allclose(out, np.asarray([(np.exp(3.0) + np.exp(5.0)) / 2.0], dtype=np.float64))


def test_predict_response_shrunk_marginalizing_animal_is_stable_for_large_link_values() -> None:
    module = runpy.run_path(str(Path("scripts/gam/plot_significant_gams.py")))
    predict_response_shrunk_marginalizing_animal = module["predict_response_shrunk_marginalizing_animal"]

    newdata = pd.DataFrame(
        {
            "animal": pd.Categorical(["A"], categories=["A", "B"]),
            "ab": pd.Categorical(["A.batch1"], categories=["A.batch1", "B.batch1"]),
        }
    )
    with np.errstate(over="raise", invalid="raise"):
        out = predict_response_shrunk_marginalizing_animal(
            predictor=_OverflowPredictor(),
            fit=None,
            newdata=newdata,
            animal_levels=["A", "B"],
            ab_levels=["A.batch1", "B.batch1"],
            batch_ref="batch1",
            shrink="none",
            hard_z=1.96,
        )

    assert np.isfinite(out[0])


def test_run_plot_jobs_uses_process_pool_chunks(monkeypatch) -> None:
    mod = importlib.import_module("scripts.gam.plot_significant_gams")
    calls: list[list[str]] = []

    class _ImmediateExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            return future

    def _fake_plot_selected_genes(args, sig_genes):
        calls.append(list(sig_genes))
        return len(sig_genes)

    monkeypatch.setattr(mod, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(mod, "get_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "plot_selected_genes", _fake_plot_selected_genes)

    mod.run_plot_jobs(argparse.Namespace(workers=2), ["G1", "G2", "G3"])

    assert calls == [["G1", "G2"], ["G3"]]
