from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fishtools.postprocess.utils_h5ad import write_obsm_h5ad


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_brdu_edu_proportions_by_ap_ml.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_brdu_edu_proportions_by_ap_ml", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_coronal_midline_columns_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["slice_i", "t", "y", "x", "vent_y", "vent_x", "pia_y", "pia_x", "thickness_um"],
        )
        writer.writeheader()
        ts = np.linspace(0.0, 1.0, 25, dtype=np.float64)
        for slice_i, y_shift in [(10, 0.0), (11, 3.0), (12, 6.0)]:
            y = 100.0 * ts + float(y_shift)
            x = 40.0 * ts + 0.5 * float(slice_i - 10)
            vent_y = y - 1.5
            vent_x = x - 1.0
            pia_y = y + 1.5
            pia_x = x + 1.0
            thickness = np.full(ts.shape, 30.0 + float(slice_i - 10), dtype=np.float64)
            for t, yy, xx, vy, vx, py, px, th in zip(
                ts, y, x, vent_y, vent_x, pia_y, pia_x, thickness, strict=True
            ):
                writer.writerow(
                    {
                        "slice_i": int(slice_i),
                        "t": float(t),
                        "y": float(yy),
                        "x": float(xx),
                        "vent_y": float(vy),
                        "vent_x": float(vx),
                        "pia_y": float(py),
                        "pia_x": float(px),
                        "thickness_um": float(th),
                    }
                )


def _write_overlap_t_ranges_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slice_i", "t_start_all", "t_end_all"])
        writer.writeheader()
        for slice_i in [10, 11, 12]:
            writer.writerow({"slice_i": slice_i, "t_start_all": 0.2, "t_end_all": 0.8})


def _make_refextract_fixture(tmp_path: Path) -> Path:
    outdir = tmp_path / "refextract"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_coronal_midline_columns_csv(outdir / "coronal_midline_columns.csv")
    _write_overlap_t_ranges_csv(outdir / "coronal_neocortex_mesocortex_overlap_t_ranges.csv")
    np.save(outdir / "resolution_ds_ijk_um.npy", np.asarray([20.0, 10.0, 5.0], dtype=np.float64))
    np.savez(
        outdir / "ap_axis_um_from_strips.npz",
        slice_keys=np.asarray([10, 11, 12], dtype=np.int32),
        ap_um=np.asarray([0.0, 100.0, 200.0], dtype=np.float64),
    )
    return outdir


def test_predict_ts_hours_2d_matches_fraction_identity() -> None:
    mod = _load_module()

    coef = np.asarray([0.0, 0.0, 0.0], dtype=float)
    ts = mod._predict_ts_hours_2d(
        coef=coef,
        ap_um=np.asarray([0.0, 100.0], dtype=float),
        ml_um=np.asarray([-50.0, 50.0], dtype=float),
        delta_t_hours=1.5,
    )

    np.testing.assert_allclose(ts, np.asarray([1.5, 1.5], dtype=float))


def test_read_gene_layer_values_reads_selected_rows() -> None:
    mod = _load_module()
    adata = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["Eomes", "Btg2"]),
        layers={"raw": np.asarray([[0.0, 5.0], [2.0, 6.0], [4.0, 7.0]], dtype=np.float32)},
    )

    values = mod._read_gene_layer_values(adata, "Eomes", layer="raw", obs_idx=np.asarray([0, 2], dtype=int))

    np.testing.assert_allclose(values, np.asarray([0.0, 4.0], dtype=float))


def test_filename_filter_tag_omits_leiden_clusters() -> None:
    mod = _load_module()

    assert mod._filename_filter_tag(manual_layer="5", eomes_gt=None) == "manual_layer_5"
    assert mod._filename_filter_tag(manual_layer="5", eomes_gt=1.0) == "manual_layer_5__Eomes_gt_1"
    assert mod._filename_filter_tag(manual_layer=None, eomes_gt=None) == "all_cells"


def test_load_adata_with_external_obsm_overwrites_embedded_obsm(tmp_path: Path) -> None:
    mod = _load_module()

    obs = pd.DataFrame(index=pd.Index(["cell_a", "cell_b"], name="index"))
    base = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32), obs=obs.copy())
    base.obsm["AP_ML_um"] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    h5ad_path = tmp_path / "base.h5ad"
    base.write_h5ad(h5ad_path)

    external = ad.AnnData(X=np.zeros((2, 0), dtype=np.float32), obs=obs.copy())
    external.obsm["AP_ML_um"] = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    obsm_path = tmp_path / "obsm_only.h5ad"
    write_obsm_h5ad(external, obsm_path)

    loaded = mod._load_adata_with_external_obsm(h5ad_path=h5ad_path, obsm_h5ad_path=obsm_path)

    np.testing.assert_allclose(loaded.obsm["AP_ML_um"], external.obsm["AP_ML_um"])


def test_tc_hours_obs_from_counts_uses_dual_fraction() -> None:
    mod = _load_module()

    tc = mod._tc_hours_obs_from_counts(
        n_total=np.asarray([50.0], dtype=float),
        n_brdu_only=np.asarray([10.0], dtype=float),
        n_edu_only=np.asarray([5.0], dtype=float),
        n_dual=np.asarray([15.0], dtype=float),
        delta_t_hours=1.5,
    )

    np.testing.assert_allclose(tc, np.asarray([7.5], dtype=float))


def test_predict_tc_hours_2d_matches_dual_fraction_identity() -> None:
    mod = _load_module()

    coef_ts = np.asarray([0.0, 0.0, 0.0], dtype=float)
    coef_dual = np.asarray([0.0, 0.0, 0.0], dtype=float)
    tc = mod._predict_tc_hours_2d(
        coef_ts=coef_ts,
        coef_dual=coef_dual,
        ap_um=np.asarray([0.0, 100.0], dtype=float),
        ml_um=np.asarray([-50.0, 50.0], dtype=float),
        delta_t_hours=1.5,
    )

    np.testing.assert_allclose(tc, np.asarray([3.0, 3.0], dtype=float))


def test_predict_tc_hours_1d_matches_dual_fraction_identity() -> None:
    mod = _load_module()

    tc = mod._predict_tc_hours_1d(
        intercept_ts=0.0,
        slope_ts=0.0,
        intercept_dual=0.0,
        slope_dual=0.0,
        coord_um=np.asarray([0.0, 100.0], dtype=float),
        delta_t_hours=1.5,
    )

    np.testing.assert_allclose(tc, np.asarray([3.0, 3.0], dtype=float))


def test_predict_tc_hours_1d_distinguishes_dual_from_edu_pos_fraction() -> None:
    mod = _load_module()

    tc = mod._predict_tc_hours_1d(
        intercept_ts=0.0,
        slope_ts=0.0,
        intercept_dual=np.log(0.25 / 0.75),
        slope_dual=0.0,
        coord_um=np.asarray([0.0], dtype=float),
        delta_t_hours=1.5,
    )

    np.testing.assert_allclose(tc, np.asarray([6.0], dtype=float))


def test_fit_binomial_logit_2d_with_animal_effects_reduces_spurious_ap_slope() -> None:
    mod = _load_module()

    ap_um = np.asarray([-300.0, -200.0, -100.0, 0.0, 0.0, 100.0, 200.0, 300.0], dtype=float)
    ml_um = np.asarray([-50.0, 50.0, -50.0, 50.0, -50.0, 50.0, -50.0, 50.0], dtype=float)
    n = np.full(ap_um.shape, 1000.0, dtype=float)
    k = np.asarray([200.0, 200.0, 200.0, 200.0, 800.0, 800.0, 800.0, 800.0], dtype=float)
    animal = np.asarray(["JaxA1", "JaxA1", "JaxA1", "JaxA1", "JaxA2", "JaxA2", "JaxA2", "JaxA2"], dtype=object)

    coef_pooled, _cov_pooled = mod._fit_binomial_logit_2d(ap_um, ml_um, k, n)
    coef_adjusted, _cov_adjusted = mod._fit_binomial_logit_2d(ap_um, ml_um, k, n, animal=animal)

    assert coef_pooled[1] > 0.0
    assert abs(coef_adjusted[1]) < (0.1 * abs(coef_pooled[1]))


def test_percentile_support_bounds_accept_percent_inputs() -> None:
    mod = _load_module()

    lo, hi = mod._percentile_support_bounds(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float))

    assert 0.0 <= lo < hi <= 4.0


def test_plot_2d_plane_ts_hours_smoke(tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "ts_2d.png"
    plane = {
        "ap_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ml_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ap_centers": np.asarray([50.0, 150.0], dtype=float),
        "ml_centers": np.asarray([50.0, 150.0], dtype=float),
        "n_brdu_only": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        "n_dual": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    }

    mod._plot_2d_plane_ts_hours(
        plane,
        coef_ts=np.asarray([0.0, 0.0, 0.0], dtype=float),
        delta_t_hours=1.5,
        title="Ts 2D smoke",
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_2d_plane_ts_hours_inverts_both_axes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()

    class FakeAxis:
        def __init__(self) -> None:
            self.x_inverted = False
            self.y_inverted = False

        def pcolormesh(self, *_args, **_kwargs):
            return object()

        def set_title(self, *_args, **_kwargs) -> None:
            return None

        def set_xlabel(self, *_args, **_kwargs) -> None:
            return None

        def set_ylabel(self, *_args, **_kwargs) -> None:
            return None

        def invert_xaxis(self) -> None:
            self.x_inverted = True

        def invert_yaxis(self) -> None:
            self.y_inverted = True

    class FakeFigure:
        def colorbar(self, *_args, **_kwargs) -> None:
            return None

        def suptitle(self, *_args, **_kwargs) -> None:
            return None

        def savefig(self, path, **_kwargs) -> None:
            Path(path).write_bytes(b"png")

    fake_axes = np.asarray([FakeAxis(), FakeAxis()], dtype=object)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (FakeFigure(), fake_axes))
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plane = {
        "ap_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ml_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ap_centers": np.asarray([50.0, 150.0], dtype=float),
        "ml_centers": np.asarray([50.0, 150.0], dtype=float),
        "n_brdu_only": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        "n_dual": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    }

    mod._plot_2d_plane_ts_hours(
        plane,
        coef_ts=np.asarray([0.0, 0.0, 0.0], dtype=float),
        delta_t_hours=1.5,
        title="Ts 2D inverted",
        out_png=tmp_path / "ts_2d_inverted.png",
    )

    assert all(ax.x_inverted for ax in fake_axes)
    assert all(ax.y_inverted for ax in fake_axes)


def test_plot_2d_plane_ts_hours_uses_turbo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    captured_cmaps: list[object] = []

    class FakeAxis:
        def pcolormesh(self, *_args, **kwargs):
            captured_cmaps.append(kwargs["cmap"])
            return object()

        def set_title(self, *_args, **_kwargs) -> None:
            return None

        def set_xlabel(self, *_args, **_kwargs) -> None:
            return None

        def set_ylabel(self, *_args, **_kwargs) -> None:
            return None

        def invert_xaxis(self) -> None:
            return None

        def invert_yaxis(self) -> None:
            return None

    class FakeFigure:
        def colorbar(self, *_args, **_kwargs) -> None:
            return None

        def suptitle(self, *_args, **_kwargs) -> None:
            return None

        def savefig(self, path, **_kwargs) -> None:
            Path(path).write_bytes(b"png")

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (FakeFigure(), np.asarray([FakeAxis(), FakeAxis()], dtype=object)))
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plane = {
        "ap_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ml_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ap_centers": np.asarray([50.0, 150.0], dtype=float),
        "ml_centers": np.asarray([50.0, 150.0], dtype=float),
        "n_brdu_only": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        "n_dual": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
    }

    mod._plot_2d_plane_ts_hours(
        plane,
        coef_ts=np.asarray([0.0, 0.0, 0.0], dtype=float),
        delta_t_hours=1.5,
        title="Ts 2D turbo",
        out_png=tmp_path / "ts_2d_turbo.png",
    )

    assert captured_cmaps == ["turbo", "turbo"]


def test_plot_2d_plane_tc_hours_uses_turbo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    captured_cmaps: list[object] = []

    class FakeAxis:
        def pcolormesh(self, *_args, **kwargs):
            captured_cmaps.append(kwargs["cmap"])
            return object()

        def set_title(self, *_args, **_kwargs) -> None:
            return None

        def set_xlabel(self, *_args, **_kwargs) -> None:
            return None

        def set_ylabel(self, *_args, **_kwargs) -> None:
            return None

        def invert_xaxis(self) -> None:
            return None

        def invert_yaxis(self) -> None:
            return None

    class FakeFigure:
        def colorbar(self, *_args, **_kwargs) -> None:
            return None

        def suptitle(self, *_args, **_kwargs) -> None:
            return None

        def savefig(self, path, **_kwargs) -> None:
            Path(path).write_bytes(b"png")

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (FakeFigure(), np.asarray([FakeAxis(), FakeAxis()], dtype=object)))
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    plane = {
        "ap_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ml_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ap_centers": np.asarray([50.0, 150.0], dtype=float),
        "ml_centers": np.asarray([50.0, 150.0], dtype=float),
        "n_total": np.asarray([[40.0, 50.0], [60.0, 70.0]], dtype=float),
        "n_brdu_only": np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float),
        "n_edu_only": np.asarray([[5.0, 10.0], [15.0, 20.0]], dtype=float),
        "n_dual": np.asarray([[5.0, 10.0], [15.0, 20.0]], dtype=float),
    }

    mod._plot_2d_plane_tc_hours(
        plane,
        coef_ts=np.asarray([0.0, 0.0, 0.0], dtype=float),
        coef_dual=np.asarray([0.0, 0.0, 0.0], dtype=float),
        delta_t_hours=1.5,
        title="Tc 2D turbo",
        out_png=tmp_path / "tc_2d_turbo.png",
    )

    assert captured_cmaps == ["turbo", "turbo"]


def test_plot_2d_plane_edu_pos_fraction_smoke(tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "edu_pos_2d.png"
    plane = {
        "ap_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ml_edges": np.asarray([0.0, 100.0, 200.0], dtype=float),
        "ap_centers": np.asarray([50.0, 150.0], dtype=float),
        "ml_centers": np.asarray([50.0, 150.0], dtype=float),
        "n_total": np.asarray([[40.0, 50.0], [60.0, 70.0]], dtype=float),
        "n_edu_only": np.asarray([[5.0, 10.0], [15.0, 20.0]], dtype=float),
        "n_dual": np.asarray([[5.0, 10.0], [15.0, 20.0]], dtype=float),
    }

    mod._plot_2d_plane_edu_pos_fraction(
        plane,
        coef_edu_pos=np.asarray([0.0, 0.0, 0.0], dtype=float),
        title="EdU+ 2D smoke",
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_tc_hours_1d_smoke(tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "tc_line.png"
    df = pd.DataFrame(
        {
            "bin_center_um": np.asarray([50.0, 150.0, 250.0], dtype=float),
            "tc_hours_obs": np.asarray([4.0, 5.0, 6.0], dtype=float),
            "tc_hours_fit": np.asarray([4.5, 5.5, 6.5], dtype=float),
        }
    )

    mod._plot_tc_hours_1d(
        df,
        axis_label="AP (um, binned)",
        title="Tc line smoke",
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_native_ts_surface_smoke(tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "ts_native.png"

    def fake_build_context(**_kwargs):
        return {
            "x2d": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y2d": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z2d": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "x3": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y3": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z3": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "tri_support": np.asarray([True], dtype=bool),
            "tri_neomeso": np.asarray([True], dtype=bool),
            "neomeso_start_fit": None,
            "neomeso_end_fit": None,
            "ordered_geom_support": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "ordered_geom_neomeso": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "support_flat": np.asarray([True, True, True], dtype=bool),
            "neomeso_flat": np.asarray([True, True, True], dtype=bool),
            "ap_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "ml_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "n_rows": 1,
            "n_cols": 3,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "build_apml_native_surface_projection_context", fake_build_context)
    monkeypatch.setattr(mod, "plot_coronal_surface_projection", lambda *_args, **kwargs: Path(kwargs["out_png"]).write_bytes(b"png"))

    try:
        mod._plot_native_ts_surface(
            coef=np.asarray([0.0, 0.0, 0.0], dtype=float),
            delta_t_hours=1.5,
            out_png=out_png,
            refextract_outdir=tmp_path,
            refextract_slice_i_min=9,
            refextract_slice_i_max=13,
            refextract_n_t=33,
            refextract_ref_t=0.5,
            refextract_band_frac=0.2,
            refextract_res_ijk_um=(20.0, 10.0, 5.0),
            native_elev_deg=-10.0,
            native_azim_deg=-110.0,
            native_roll_deg=180.0,
            native_latlon=True,
            native_graticule="ijk",
            native_lat_stride=4,
            native_lon_stride=4,
            native_max_lat_lines=4,
            native_max_lon_lines=4,
            native_proj_type="persp",
            native_focal_length=0.4,
            ap_support_bounds_um=None,
            ml_support_bounds_um=None,
            restrict_t_neomeso=True,
            gray_context=True,
            title="Ts smoke",
        )
    finally:
        monkeypatch.undo()

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_native_ts_surface_uses_gam_native_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "ts_native_gam.png"
    captured: dict[str, object] = {}

    def fake_build_context(**_kwargs):
        return {
            "x2d": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y2d": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z2d": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "x3": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y3": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z3": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "tri_support": np.asarray([True], dtype=bool),
            "tri_neomeso": np.asarray([True], dtype=bool),
            "neomeso_start_fit": None,
            "neomeso_end_fit": None,
            "ordered_geom_support": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "ordered_geom_neomeso": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "support_flat": np.asarray([True, True, True], dtype=bool),
            "neomeso_flat": np.asarray([True, True, True], dtype=bool),
            "ap_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "ml_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "n_rows": 1,
            "n_cols": 3,
        }

    def fake_plot(values, **kwargs):
        captured["values"] = np.asarray(values, dtype=np.float64)
        captured.update(kwargs)
        Path(kwargs["out_png"]).write_bytes(b"png")

    monkeypatch.setattr(mod, "build_apml_native_surface_projection_context", fake_build_context)
    monkeypatch.setattr(mod, "plot_coronal_surface_projection", fake_plot)

    mod._plot_native_ts_surface(
        coef=np.asarray([0.0, 0.0, 0.0], dtype=float),
        delta_t_hours=1.5,
        out_png=out_png,
        refextract_outdir=tmp_path,
        refextract_slice_i_min=9,
        refextract_slice_i_max=13,
        refextract_n_t=33,
        refextract_ref_t=0.5,
        refextract_band_frac=0.2,
        refextract_res_ijk_um=(20.0, 10.0, 5.0),
        native_elev_deg=-5.0,
        native_azim_deg=-95.0,
        native_roll_deg=180.0,
        native_latlon=True,
        native_graticule="ijk",
        native_lat_stride=4,
        native_lon_stride=4,
        native_max_lat_lines=4,
        native_max_lon_lines=4,
        native_proj_type="persp",
        native_focal_length=0.4,
        ap_support_bounds_um=(0.0, 20.0),
        ml_support_bounds_um=(0.0, 20.0),
        restrict_t_neomeso=True,
        gray_context=False,
        title="Ts GAM renderer",
    )

    assert out_png.exists()
    assert captured["graticule"] == "ijk"
    assert captured["proj_type"] == "persp"
    assert captured["focal_length"] == 0.4
    np.testing.assert_allclose(captured["tri_alpha"], np.asarray([1.0], dtype=np.float64))
    np.testing.assert_allclose(captured["values"], np.asarray([1.5, 1.5, 1.5], dtype=np.float64))
    assert captured["cmap"].name == "turbo"


def test_plot_native_brdu_only_fraction_surface_uses_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "brdu_only_native.png"
    captured: dict[str, object] = {}

    def fake_build_context(**_kwargs):
        return {
            "x2d": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y2d": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z2d": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "x3": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y3": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z3": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "tri_support": np.asarray([True], dtype=bool),
            "tri_neomeso": np.asarray([True], dtype=bool),
            "neomeso_start_fit": None,
            "neomeso_end_fit": None,
            "ordered_geom_support": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "ordered_geom_neomeso": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "support_flat": np.asarray([True, True, True], dtype=bool),
            "neomeso_flat": np.asarray([True, True, True], dtype=bool),
            "ap_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "ml_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "n_rows": 1,
            "n_cols": 3,
        }

    def fake_plot(values, **kwargs):
        captured["values"] = np.asarray(values, dtype=np.float64)
        captured.update(kwargs)
        Path(kwargs["out_png"]).write_bytes(b"png")

    monkeypatch.setattr(mod, "build_apml_native_surface_projection_context", fake_build_context)
    monkeypatch.setattr(mod, "plot_coronal_surface_projection", fake_plot)

    mod._plot_native_brdu_only_fraction_surface(
        coef=np.asarray([0.0, 0.0, 0.0], dtype=float),
        out_png=out_png,
        refextract_outdir=tmp_path,
        refextract_slice_i_min=9,
        refextract_slice_i_max=13,
        refextract_n_t=33,
        refextract_ref_t=0.5,
        refextract_band_frac=0.2,
        refextract_res_ijk_um=(20.0, 10.0, 5.0),
        native_elev_deg=-5.0,
        native_azim_deg=-95.0,
        native_roll_deg=180.0,
        native_latlon=True,
        native_graticule="ijk",
        native_lat_stride=4,
        native_lon_stride=4,
        native_max_lat_lines=4,
        native_max_lon_lines=4,
        native_proj_type="persp",
        native_focal_length=0.4,
        ap_support_bounds_um=(0.0, 20.0),
        ml_support_bounds_um=(0.0, 20.0),
        restrict_t_neomeso=True,
        gray_context=False,
        title="BrdU-only native renderer",
    )

    assert out_png.exists()
    np.testing.assert_allclose(captured["values"], np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
    assert captured["cmap"].name == "turbo"


def test_plot_native_edu_pos_fraction_surface_uses_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out_png = tmp_path / "edu_pos_native.png"
    captured: dict[str, object] = {}

    def fake_build_context(**_kwargs):
        return {
            "x2d": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y2d": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z2d": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "x3": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "y3": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z3": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "tri_support": np.asarray([True], dtype=bool),
            "tri_neomeso": np.asarray([True], dtype=bool),
            "neomeso_start_fit": None,
            "neomeso_end_fit": None,
            "ordered_geom_support": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "ordered_geom_neomeso": {"tris": np.asarray([[0, 1, 2]], dtype=np.int32)},
            "support_flat": np.asarray([True, True, True], dtype=bool),
            "neomeso_flat": np.asarray([True, True, True], dtype=bool),
            "ap_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "ml_um_flat": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "n_rows": 1,
            "n_cols": 3,
        }

    def fake_plot(values, **kwargs):
        captured["values"] = np.asarray(values, dtype=np.float64)
        captured.update(kwargs)
        Path(kwargs["out_png"]).write_bytes(b"png")

    monkeypatch.setattr(mod, "build_apml_native_surface_projection_context", fake_build_context)
    monkeypatch.setattr(mod, "plot_coronal_surface_projection", fake_plot)

    mod._plot_native_edu_pos_fraction_surface(
        coef_edu_pos=np.asarray([0.0, 0.0, 0.0], dtype=float),
        out_png=out_png,
        refextract_outdir=tmp_path,
        refextract_slice_i_min=9,
        refextract_slice_i_max=13,
        refextract_n_t=33,
        refextract_ref_t=0.5,
        refextract_band_frac=0.2,
        refextract_res_ijk_um=(20.0, 10.0, 5.0),
        native_elev_deg=-5.0,
        native_azim_deg=-95.0,
        native_roll_deg=180.0,
        native_latlon=True,
        native_graticule="ijk",
        native_lat_stride=4,
        native_lon_stride=4,
        native_max_lat_lines=4,
        native_max_lon_lines=4,
        native_proj_type="persp",
        native_focal_length=0.4,
        ap_support_bounds_um=(0.0, 20.0),
        ml_support_bounds_um=(0.0, 20.0),
        restrict_t_neomeso=True,
        gray_context=False,
        title="EdU+ native renderer",
    )

    assert out_png.exists()
    np.testing.assert_allclose(captured["values"], np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
    assert captured["cmap"].name == "turbo"
