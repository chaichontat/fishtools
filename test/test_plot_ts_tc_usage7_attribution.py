from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_ts_tc_usage7_attribution.py"
FIT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_apml_multinomial_animal_meta.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bins_for(values: np.ndarray, bin_width_um: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    start = float(bin_width_um) * np.floor(np.nanmin(arr) / float(bin_width_um))
    end = float(bin_width_um) * (np.floor(np.nanmax(arr) / float(bin_width_um)) + 1.0)
    edges = np.arange(start, end + float(bin_width_um), float(bin_width_um), dtype=float)
    centers = edges[:-1] + (float(bin_width_um) / 2.0)
    return edges, centers


def _softmax_probs(ap_mm: float, ml_mm: float, usage7: float, animal_offset: float) -> np.ndarray:
    logits = np.array(
        [
            0.0,
            -0.7 - 0.10 * ap_mm + 0.12 * ml_mm + 0.55 * usage7 + animal_offset,
            -1.0 + 0.06 * ap_mm - 0.10 * ml_mm - 0.20 * usage7 - 0.4 * animal_offset,
            -1.4 + 0.14 * ap_mm - 0.18 * ml_mm + 0.85 * usage7 + 0.2 * animal_offset,
        ],
        dtype=float,
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    return probs / probs.sum()


def _make_usage7_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    animal_specs = (
        ("JaxA1", "sag", -0.2),
        ("JaxA3", "coro", -0.05),
        ("JaxA4", "sag", 0.05),
        ("JaxA6", "coro", 0.15),
    )
    for animal, orientation, animal_offset in animal_specs:
        for _ in range(320):
            ap_mm = rng.uniform(0.0, 3.0)
            ml_mm = rng.uniform(-0.8, 1.2)
            usage7 = float(np.clip(0.12 + 0.10 * ap_mm - 0.07 * ml_mm + 0.25 * animal_offset, 0.0, 1.0))
            probs = _softmax_probs(ap_mm, ml_mm, usage7, animal_offset)
            state = int(rng.choice(4, p=probs))
            rows.append(
                {
                    "subset": "manual_layer_1",
                    "ap_mm": ap_mm,
                    "ml_mm": ml_mm,
                    "usage7": usage7,
                    "brdu_pos": int(state in (1, 3)),
                    "edu_pos": int(state in (2, 3)),
                    "animal": animal,
                    "orientation": orientation,
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


def test_build_usage7_bin_summary_computes_mean_usage7() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    helper_mod = SimpleNamespace(_bins_for=_bins_for)
    df = pd.DataFrame(
        {
            "ap_mm": [0.05, 0.15, 0.25],
            "ml_mm": [0.05, 0.05, 0.25],
            "usage7": [0.1, 0.3, 0.7],
        }
    )

    summary = mod._build_usage7_bin_summary(helper_mod=helper_mod, df=df, bin_width_um=200.0)
    table = pd.DataFrame(summary["table"])

    first_bin = table.loc[(table["ap_center_um"] == 100.0) & (table["ml_center_um"] == 100.0)].iloc[0]
    second_bin = table.loc[(table["ap_center_um"] == 300.0) & (table["ml_center_um"] == 300.0)].iloc[0]
    assert int(first_bin["n_total"]) == 2
    assert float(first_bin["mean_usage7"]) == 0.2
    assert int(second_bin["n_total"]) == 1
    assert float(second_bin["mean_usage7"]) == 0.7


def test_lookup_binned_usage7_returns_nan_for_empty_bins() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    helper_mod = SimpleNamespace(_bins_for=_bins_for)
    df = pd.DataFrame(
        {
            "ap_mm": [0.05, 0.15, 0.25],
            "ml_mm": [0.05, 0.05, 0.25],
            "usage7": [0.1, 0.3, 0.7],
        }
    )

    summary = mod._build_usage7_bin_summary(helper_mod=helper_mod, df=df, bin_width_um=200.0)
    usage7, counts = mod._lookup_binned_usage7(
        ap_um=np.asarray([100.0, 300.0, 100.0], dtype=float),
        ml_um=np.asarray([100.0, 100.0, 300.0], dtype=float),
        bin_summary=summary,
    )

    np.testing.assert_allclose(usage7[:2], np.asarray([0.2, np.nan]), equal_nan=True)
    np.testing.assert_allclose(usage7[2:], np.asarray([np.nan]), equal_nan=True)
    np.testing.assert_array_equal(counts, np.asarray([2, 0, 0]))


def test_build_summary_rows_emits_variance_and_derivative_sections() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    fit_mod = _load_module(FIT_SCRIPT_PATH, "fit_apml_multinomial_animal_meta")
    helper_mod = SimpleNamespace(_bins_for=_bins_for)
    df = _make_usage7_df()
    fit = fit_mod._fit_multinomial_logit_2d(df, usage7_col="usage7")
    bin_summary = mod._build_usage7_bin_summary(helper_mod=helper_mod, df=df, bin_width_um=200.0)

    summary = mod._build_summary_rows(
        meta_mod=fit_mod,
        fit=fit,
        df=df,
        subset="manual_layer_1",
        bin_summary=bin_summary,
        delta_t_hours=1.5,
        n_draws=128,
        seed=0,
        usage7_baseline=0.0,
        usage7_col="mean_usage7",
    )

    assert {"variance", "variance_reduction", "local_derivative", "derivative_attenuation"} <= set(summary["summary_type"])
    assert {"Ts_hours", "Tc_hours"} <= set(summary["metric"])
    assert {"AP", "ML", "surface"} <= set(summary["axis"])
    assert np.isfinite(
        summary.loc[summary["summary_type"] == "variance_reduction", "estimate"].to_numpy(dtype=float)
    ).all()


def test_baseline_usage7_quantile_value_uses_populated_bins_only() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    bin_summary = {
        "table": pd.DataFrame(
            {
                "n_total": [3, 0, 5, 2],
                "gam_usage7": [0.10, np.nan, 0.30, 0.50],
            }
        )
    }

    value = mod._baseline_usage7_quantile_value(bin_summary=bin_summary, usage7_col="gam_usage7", quantile=0.05)

    np.testing.assert_allclose(value, np.quantile(np.asarray([0.10, 0.30, 0.50]), 0.05))


def test_output_stem_supports_quantile_baseline() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")

    stem = mod._output_stem(usage7_baseline_quantile=0.05)

    assert stem == "manual_layer_1__Usage7_vs_baseline_q05"


def test_circular_theta_marginal_values_stay_on_circle() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")

    theta = np.asarray([6.10, 6.20, 0.05, 0.10, 0.20], dtype=float)
    values = mod._circular_theta_marginal_values(theta)

    assert values.shape == (5,)
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0)
    assert np.all(values <= 2.0 * np.pi)


def test_load_or_predict_usage7_gam_uses_cache(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    calls = {"n": 0}

    def _fake_predict(*, bundle, ap_um, ml_um):
        calls["n"] += 1
        return np.asarray(ap_um, dtype=float) * 0.01 + np.asarray(ml_um, dtype=float) * 0.001

    monkeypatch.setattr(mod, "_predict_usage7_gam", _fake_predict)
    cache_path = tmp_path / "usage7_cache.npz"
    ap = np.asarray([100.0, 200.0], dtype=float)
    ml = np.asarray([10.0, 20.0], dtype=float)
    bundle = {
        "marginal_animal_levels": ["JaxA1", "JaxA3"],
        "r_values": np.asarray([10.0, 20.0], dtype=float),
        "theta_values": np.asarray([0.1, 0.2], dtype=float),
        "topic_index": 7,
    }

    first = mod._load_or_predict_usage7_gam(cache_path=cache_path, bundle=bundle, ap_um=ap, ml_um=ml)
    second = mod._load_or_predict_usage7_gam(cache_path=cache_path, bundle=bundle, ap_um=ap, ml_um=ml)

    np.testing.assert_allclose(first, second)
    assert calls["n"] == 1


def test_load_or_predict_usage7_gam_invalidates_cache_when_signature_changes(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    calls = {"n": 0}

    def _fake_predict(*, bundle, ap_um, ml_um):
        calls["n"] += 1
        scale = float(len(bundle["marginal_animal_levels"]))
        return scale + (np.asarray(ap_um, dtype=float) * 0.0) + (np.asarray(ml_um, dtype=float) * 0.0)

    monkeypatch.setattr(mod, "_predict_usage7_gam", _fake_predict)
    cache_path = tmp_path / "usage7_cache.npz"
    ap = np.asarray([100.0, 200.0], dtype=float)
    ml = np.asarray([10.0, 20.0], dtype=float)
    bundle_a = {
        "marginal_animal_levels": ["JaxA1", "JaxA3"],
        "r_values": np.asarray([10.0, 20.0], dtype=float),
        "theta_values": np.asarray([0.1, 0.2], dtype=float),
        "topic_index": 7,
    }
    bundle_b = {
        "marginal_animal_levels": ["JaxA1"],
        "r_values": np.asarray([10.0, 20.0], dtype=float),
        "theta_values": np.asarray([0.1, 0.2], dtype=float),
        "topic_index": 7,
    }

    first = mod._load_or_predict_usage7_gam(cache_path=cache_path, bundle=bundle_a, ap_um=ap, ml_um=ml)
    second = mod._load_or_predict_usage7_gam(cache_path=cache_path, bundle=bundle_b, ap_um=ap, ml_um=ml)

    np.testing.assert_allclose(first, np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(second, np.asarray([1.0, 1.0]))
    assert calls["n"] == 2


def test_interpolate_usage7_field_uses_linear_apml_interpolation() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    interp = mod._build_usage7_interpolator(
        ap_um=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float),
        ml_um=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float),
        usage7=np.asarray([0.0, 1.0, 1.0, 2.0], dtype=float),
    )
    values = interp(
        ap_um=np.asarray([0.25, 0.75, 1.5], dtype=float),
        ml_um=np.asarray([0.25, 0.75, 0.5], dtype=float),
    )

    np.testing.assert_allclose(values[:2], np.asarray([0.5, 1.5]), atol=1e-8)
    assert np.isnan(values[2])


def test_build_line_delta_figure_writes_png(tmp_path: Path) -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    curve = pd.DataFrame(
        {
            "AP_um": np.linspace(200.0, 1200.0, 6),
            "ML_um": np.linspace(100.0, 300.0, 6),
            "t_s": np.linspace(0.3, 0.7, 6),
            "Tc_delta_mean": np.linspace(-1.0, 1.0, 6),
            "Tc_delta_ci_low": np.linspace(-1.2, 0.8, 6),
            "Tc_delta_ci_high": np.linspace(-0.8, 1.2, 6),
            "Ts_delta_mean": np.linspace(-0.4, 0.6, 6),
            "Ts_delta_ci_low": np.linspace(-0.5, 0.5, 6),
            "Ts_delta_ci_high": np.linspace(-0.3, 0.7, 6),
        }
    )
    fig = mod._build_line_delta_figure(
        {160: curve, 184: curve.assign(Tc_delta_mean=curve["Tc_delta_mean"] * 0.5)},
        title="usage7 attribution",
        ap_support_bounds_um=(100.0, 1300.0),
        ml_support_bounds_um=(50.0, 350.0),
        line_t_min=0.25,
        line_t_max=0.75,
    )

    out_png = tmp_path / "line_delta.png"
    fig.savefig(out_png, dpi=100)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_collect_line_y_limits_uses_factual_intervals() -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    curve = pd.DataFrame(
        {
            "AP_um": np.linspace(200.0, 1200.0, 6),
            "ML_um": np.linspace(100.0, 300.0, 6),
            "t_s": np.linspace(0.3, 0.7, 6),
            "Tc_factual_ci_low": np.linspace(10.0, 15.0, 6),
            "Tc_factual_ci_high": np.linspace(20.0, 25.0, 6),
            "Ts_factual_ci_low": np.linspace(1.0, 3.0, 6),
            "Ts_factual_ci_high": np.linspace(4.0, 6.0, 6),
        }
    )

    limits = mod._collect_line_y_limits(
        {160: curve, 184: curve.assign(Tc_factual_ci_low=curve["Tc_factual_ci_low"] - 2.0)},
        ap_support_bounds_um=(100.0, 1300.0),
        ml_support_bounds_um=(50.0, 350.0),
        line_t_min=0.25,
        line_t_max=0.75,
        scenario="factual",
    )

    assert limits["Tc"] == (8.0, 25.0)
    assert limits["Ts"] == (1.0, 6.0)


def test_save_native_counterfactual_figure_uses_control_vmin_vmax(tmp_path: Path) -> None:
    mod = _load_module(SCRIPT_PATH, "plot_ts_tc_usage7_attribution")
    calls: list[dict[str, float]] = []

    class _FakeNativeMod:
        @staticmethod
        def _draw_native_scalar_surface_panel(**kwargs):
            calls.append({"vmin": float(kwargs["vmin"]), "vmax": float(kwargs["vmax"])})

    mod._save_native_counterfactual_figure(
        native_mod=_FakeNativeMod(),
        helper_mod=SimpleNamespace(),
        context={},
        support_hull=None,
        ts_values=np.asarray([100.0, 101.0], dtype=float),
        tc_values=np.asarray([200.0, 201.0], dtype=float),
        ts_control_values=np.asarray([1.0, 3.0], dtype=float),
        tc_control_values=np.asarray([10.0, 40.0], dtype=float),
        title_suffix="q05 baseline",
        out_png=tmp_path / "native_counterfactual.png",
    )

    assert calls == [{"vmin": 10.0, "vmax": 40.0}, {"vmin": 1.0, "vmax": 3.0}]
