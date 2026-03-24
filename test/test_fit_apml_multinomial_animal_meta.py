from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_apml_multinomial_animal_meta.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fit_apml_multinomial_animal_meta", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _softmax_probs(ap_mm: float, ml_mm: float, animal_offset: float) -> np.ndarray:
    logits = np.array(
        [
            0.0,
            -0.9 - 0.15 * ap_mm + 0.25 * ml_mm + animal_offset,
            -1.2 + 0.10 * ap_mm - 0.08 * ml_mm - 0.5 * animal_offset,
            -1.5 + 0.18 * ap_mm - 0.22 * ml_mm + 0.3 * animal_offset,
        ],
        dtype=float,
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    return probs / probs.sum()


def _make_synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    animal_specs = (
        ("JaxA1", "sag", -0.2),
        ("JaxA3", "coro", -0.05),
        ("JaxA4", "sag", 0.05),
        ("JaxA6", "coro", 0.15),
    )
    for animal, orientation, animal_offset in animal_specs:
        for _ in range(420):
            ap_mm = rng.uniform(0.0, 3.0)
            ml_mm = rng.uniform(-0.8, 1.2)
            probs = _softmax_probs(ap_mm, ml_mm, animal_offset)
            state = int(rng.choice(4, p=probs))
            rows.append(
                {
                    "subset": "manual_layer_1",
                    "ap_mm": ap_mm,
                    "ml_mm": ml_mm,
                    "usage7": float(np.clip(0.18 + (0.08 * ap_mm) - (0.06 * ml_mm) + (0.30 * animal_offset), 0.0, 1.0)),
                    "brdu_pos": int(state in (1, 3)),
                    "edu_pos": int(state in (2, 3)),
                    "animal": animal,
                    "orientation": orientation,
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_subset_multinomial_emits_animal_and_meta_rows() -> None:
    mod = _load_module()

    animal_df, meta_df = mod._summarize_subset_multinomial(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=0,
    )

    assert animal_df.shape[0] == 40
    assert meta_df.shape[0] == 10
    assert set(animal_df["animal"]) == {"JaxA1", "JaxA3", "JaxA4", "JaxA6"}
    assert set(animal_df["orientation"]) == {"sag", "coro"}
    assert set(animal_df["metric"]) == set(mod.METRIC_ORDER)
    assert set(meta_df["metric"]) == set(mod.METRIC_ORDER)
    assert set(meta_df["axis"]) == {"AP", "ML"}
    assert (animal_df["span_mm"] > 0).all()


def test_multinomial_meta_summary_is_finite_and_tracks_expected_signs() -> None:
    mod = _load_module()

    _animal_df, meta_df = mod._summarize_subset_multinomial(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=1,
    )

    assert np.isfinite(meta_df["estimate"]).all()
    assert np.isfinite(meta_df["se"]).all()
    dual_ap = meta_df.loc[(meta_df["metric"] == "dual_frac") & (meta_df["axis"] == "AP"), "estimate"].iloc[0]
    dual_ml = meta_df.loc[(meta_df["metric"] == "dual_frac") & (meta_df["axis"] == "ML"), "estimate"].iloc[0]
    assert dual_ap > 0
    assert dual_ml < 0


def test_summarize_subset_pooled_multinomial_emits_expected_rows() -> None:
    mod = _load_module()

    pooled_df = mod._summarize_subset_pooled_multinomial(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=2,
    )

    assert pooled_df.shape[0] == 10
    assert set(pooled_df["metric"]) == set(mod.METRIC_ORDER)
    assert set(pooled_df["axis"]) == {"AP", "ML"}
    assert (pooled_df["span_mm"] > 0).all()


def test_summarize_subset_pooled_local_derivatives_emits_expected_rows() -> None:
    mod = _load_module()

    pooled_df = mod._summarize_subset_pooled_local_derivatives(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=3,
        step_mm=0.001,
    )

    assert pooled_df.shape[0] == 10
    assert set(pooled_df["metric"]) == set(mod.METRIC_ORDER)
    assert set(pooled_df["axis"]) == {"AP", "ML"}
    assert (pooled_df["step_mm"] > 0).all()


def test_evaluate_pooled_multinomial_at_queries_preserves_query_order() -> None:
    mod = _load_module()

    query_ap = np.asarray([0.25, 1.15, 2.35], dtype=float)
    query_ml = np.asarray([-0.30, 0.10, 0.75], dtype=float)
    evaluated = mod._evaluate_pooled_multinomial_at_queries(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=4,
        ap_mm=query_ap,
        ml_mm=query_ml,
    )

    assert evaluated.shape[0] == 3
    np.testing.assert_allclose(evaluated["ap_mm"].to_numpy(dtype=float), query_ap)
    np.testing.assert_allclose(evaluated["ml_mm"].to_numpy(dtype=float), query_ml)
    for col in (
        "dual_frac_mean",
        "dual_frac_ci_low",
        "dual_frac_ci_high",
        "Ts_hours_mean",
        "Ts_hours_ci_low",
        "Ts_hours_ci_high",
        "Tc_hours_mean",
        "Tc_hours_ci_low",
        "Tc_hours_ci_high",
    ):
        assert col in evaluated.columns
        assert np.isfinite(evaluated[col].to_numpy(dtype=float)).all()


def test_evaluate_pooled_multinomial_at_queries_supports_usage7_covariate() -> None:
    mod = _load_module()

    query_ap = np.asarray([0.25, 1.15, 2.35], dtype=float)
    query_ml = np.asarray([-0.30, 0.10, 0.75], dtype=float)
    query_usage7 = np.asarray([0.05, 0.15, 0.30], dtype=float)
    evaluated = mod._evaluate_pooled_multinomial_at_queries(
        _make_synthetic_df(),
        subset="manual_layer_1",
        delta_t_hours=1.5,
        n_draws=128,
        seed=5,
        ap_mm=query_ap,
        ml_mm=query_ml,
        usage7=query_usage7,
        usage7_col="usage7",
    )

    assert evaluated.shape[0] == 3
    np.testing.assert_allclose(evaluated["ap_mm"].to_numpy(dtype=float), query_ap)
    np.testing.assert_allclose(evaluated["ml_mm"].to_numpy(dtype=float), query_ml)
    assert np.isfinite(evaluated["Ts_hours_mean"].to_numpy(dtype=float)).all()
    assert np.isfinite(evaluated["Tc_hours_mean"].to_numpy(dtype=float)).all()


def test_build_model_df_applies_strict_eomes_lt_filter() -> None:
    mod = _load_module()

    obs = pd.DataFrame(
        {
            "leiden": ["0", "0", "0", "0"],
            "manual_layer": ["1", "1", "1", "5"],
            "brdu_pos": [1, 0, 1, 1],
            "edu_pos": [0, 1, 1, 0],
            "dataset": ["JaxA1_Sag", "JaxA1_Sag", "JaxA3_Coro", "JaxA4_Sag"],
        }
    )
    adata = SimpleNamespace(
        obs=obs,
        obsm={"AP_ML_um": np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 100.0], [300.0, 150.0]], dtype=float)},
    )
    helper_mod = SimpleNamespace(
        _read_gene_layer_values=lambda _adata, _gene, layer, obs_idx: np.array([0.2, 1.0, 1.4], dtype=float),
        _dataset_animal=lambda dataset: dataset.split("_")[0],
        _dataset_orientation=lambda dataset: dataset.split("_")[1].lower(),
        _cell_state_codes=lambda *, brdu_pos, edu_pos: np.asarray(brdu_pos, dtype=int) + 2 * np.asarray(edu_pos, dtype=int),
    )

    df = mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters={"0"},
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=1.0,
        label="manual_layer_1__Eomes_lt_1",
        exclude_animals=set(),
    )

    assert df.shape[0] == 1
    assert df["animal"].tolist() == ["JaxA1"]
    assert df["orientation"].tolist() == ["sag"]


def test_build_model_df_applies_usage7_lte_filter() -> None:
    mod = _load_module()

    obs = pd.DataFrame(
        {
            "leiden": ["0", "0", "0", "0"],
            "manual_layer": ["1", "1", "1", "1"],
            "brdu_pos": [1, 0, 1, 1],
            "edu_pos": [0, 1, 1, 0],
            "dataset": ["JaxA1_Sag", "JaxA1_Sag", "JaxA3_Coro", "JaxA4_Sag"],
            "Usage_7": [0.05, 0.20, 0.21, np.nan],
        }
    )
    adata = SimpleNamespace(
        obs=obs,
        obsm={"AP_ML_um": np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 100.0], [300.0, 150.0]], dtype=float)},
    )
    helper_mod = SimpleNamespace(
        _read_gene_layer_values=lambda _adata, _gene, layer, obs_idx: np.array([], dtype=float),
        _dataset_animal=lambda dataset: dataset.split("_")[0],
        _dataset_orientation=lambda dataset: dataset.split("_")[1].lower(),
        _cell_state_codes=lambda *, brdu_pos, edu_pos: np.asarray(brdu_pos, dtype=int) + 2 * np.asarray(edu_pos, dtype=int),
    )

    df = mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters={"0"},
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=None,
        usage7_lte=0.2,
        label="manual_layer_1__Usage7_lte_0p2",
        exclude_animals=set(),
    )

    assert df.shape[0] == 2
    assert df["animal"].tolist() == ["JaxA1", "JaxA1"]
    assert df["orientation"].tolist() == ["sag", "sag"]


def test_build_model_df_applies_usage7_fraction_lte_filter() -> None:
    mod = _load_module()

    obs = pd.DataFrame(
        {
            "leiden": ["0", "0", "0", "0"],
            "manual_layer": ["1", "1", "1", "1"],
            "brdu_pos": [1, 0, 1, 1],
            "edu_pos": [0, 1, 1, 0],
            "dataset": ["JaxA1_Sag", "JaxA1_Sag", "JaxA3_Coro", "JaxA4_Sag"],
            "Usage_1": [0.8, 0.2, 0.3, 0.2],
            "Usage_3": [0.0, 0.0, 0.1, 0.0],
            "Usage_4": [0.0, 0.0, 0.1, 0.0],
            "Usage_6": [0.0, 0.0, 0.1, 0.0],
            "Usage_7": [0.2, 0.8, 0.1, np.nan],
        }
    )
    adata = SimpleNamespace(
        obs=obs,
        obsm={"AP_ML_um": np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 100.0], [300.0, 150.0]], dtype=float)},
    )
    helper_mod = SimpleNamespace(
        _read_gene_layer_values=lambda _adata, _gene, layer, obs_idx: np.array([], dtype=float),
        _dataset_animal=lambda dataset: dataset.split("_")[0],
        _dataset_orientation=lambda dataset: dataset.split("_")[1].lower(),
        _cell_state_codes=lambda *, brdu_pos, edu_pos: np.asarray(brdu_pos, dtype=int) + 2 * np.asarray(edu_pos, dtype=int),
    )

    df = mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters={"0"},
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=None,
        usage7_fraction_lte=0.25,
        label="manual_layer_1__Usage7frac_lte_0p25",
        exclude_animals=set(),
    )

    assert df.shape[0] == 2
    assert df["animal"].tolist() == ["JaxA1", "JaxA3"]
    assert df["orientation"].tolist() == ["sag", "coro"]
