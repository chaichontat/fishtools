#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.meta_analysis import combine_effects


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PLOT_APML_PATH = SCRIPT_DIR / "plot_brdu_edu_proportions_by_ap_ml.py"
DEFAULT_H5AD = pathlib.Path("~/nvme/all_excit.h5ad")
DEFAULT_OBSM_H5AD = pathlib.Path("~/nvme/obsm.h5ad")
DEFAULT_OUTDIR = pathlib.Path("scripts/_out/apml_multinomial_animal_meta")
DEFAULT_SUBSETS = (
    {
        "label": "manual_layer_1",
        "manual_layer": "1",
        "eomes_gt": None,
        "eomes_lt": None,
        "title": "manual_layer=1",
    },
    {
        "label": "manual_layer_5__Eomes_gt_1",
        "manual_layer": "5",
        "eomes_gt": 1.0,
        "eomes_lt": None,
        "title": "manual_layer=5, Eomes>1",
    },
)
METRIC_ORDER = (
    "brdu_only_frac",
    "edu_only_frac",
    "dual_frac",
    "Ts_hours",
    "Tc_hours",
)
AXIS_ORDER = ("AP", "ML")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-animal 2D multinomial AP/ML models, derive on-support contrasts, "
            "and combine them across animals with random-effects meta-analysis."
        )
    )
    parser.add_argument("--h5ad", type=pathlib.Path, default=DEFAULT_H5AD)
    parser.add_argument("--obsm-h5ad", type=pathlib.Path, default=DEFAULT_OBSM_H5AD)
    parser.add_argument("--clusters", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--exclude-animals", type=str, default="JaxA2")
    parser.add_argument("--delta-t-hours", type=float, default=1.5)
    parser.add_argument("--n-draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def _load_plot_apml_module() -> Any:
    spec = importlib.util.spec_from_file_location("plot_brdu_edu_proportions_by_ap_ml", PLOT_APML_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {PLOT_APML_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _usage7_fraction_13467(obs: pd.DataFrame) -> np.ndarray:
    usage_cols = ["Usage_1", "Usage_3", "Usage_4", "Usage_6", "Usage_7"]
    missing = [col for col in usage_cols if col not in obs.columns]
    if missing:
        raise KeyError(f"Missing Usage columns for Usage_7 fraction: {missing}")
    values = {
        col: pd.to_numeric(obs[col], errors="coerce").to_numpy(dtype=float, copy=False) for col in usage_cols
    }
    denom = values["Usage_1"] + values["Usage_3"] + values["Usage_4"] + values["Usage_6"] + values["Usage_7"]
    out = np.full(denom.shape, np.nan, dtype=float)
    finite = np.isfinite(denom) & (denom > 0.0) & np.isfinite(values["Usage_7"])
    out[finite] = values["Usage_7"][finite] / denom[finite]
    return out


def _build_model_df(
    *,
    helper_mod: Any,
    adata: Any,
    clusters: set[str],
    manual_layer: str,
    eomes_gt: float | None,
    eomes_lt: float | None,
    usage7_lte: float | None = None,
    usage7_fraction_lte: float | None = None,
    label: str,
    exclude_animals: set[str],
) -> pd.DataFrame:
    obs = adata.obs
    mask = obs["leiden"].astype(str).isin(clusters).to_numpy()
    mask &= obs["manual_layer"].astype(str).to_numpy() == str(manual_layer)
    if usage7_lte is not None and usage7_fraction_lte is not None:
        raise ValueError("Specify at most one of usage7_lte or usage7_fraction_lte.")
    if usage7_lte is not None:
        usage7_values = obs["Usage_7"].to_numpy(dtype=float)
        mask &= np.isfinite(usage7_values)
        mask &= usage7_values <= float(usage7_lte)
    if usage7_fraction_lte is not None:
        usage7_fraction = _usage7_fraction_13467(obs)
        mask &= np.isfinite(usage7_fraction)
        mask &= usage7_fraction <= float(usage7_fraction_lte)
    idx = np.flatnonzero(mask)
    if eomes_gt is not None or eomes_lt is not None:
        eomes_values = helper_mod._read_gene_layer_values(adata, "Eomes", layer="raw", obs_idx=idx)
        if eomes_gt is not None:
            idx = idx[eomes_values > float(eomes_gt)]
            eomes_values = eomes_values[eomes_values > float(eomes_gt)]
        if eomes_lt is not None:
            idx = idx[eomes_values < float(eomes_lt)]
    coords = np.asarray(adata.obsm["AP_ML_um"][idx], dtype=float)
    brdu = obs.iloc[idx]["brdu_pos"].to_numpy(dtype=bool)
    edu = obs.iloc[idx]["edu_pos"].to_numpy(dtype=bool)
    dataset = obs.iloc[idx]["dataset"].astype(str).to_numpy(dtype=object)
    animal = np.asarray([helper_mod._dataset_animal(d) for d in dataset], dtype=object)
    orientation = np.asarray([helper_mod._dataset_orientation(d) for d in dataset], dtype=object)
    finite = np.isfinite(coords).all(axis=1)
    keep = finite & (~np.isin(animal, list(exclude_animals)))
    if not np.any(keep):
        raise ValueError(f"No cells remain for subset={label}")
    df = pd.DataFrame(
        {
            "subset": label,
            "ap_mm": coords[keep, 0] / 1000.0,
            "ml_mm": coords[keep, 1] / 1000.0,
            "brdu_pos": brdu[keep].astype(int),
            "edu_pos": edu[keep].astype(int),
            "animal": animal[keep].astype(str),
            "orientation": orientation[keep].astype(str),
        }
    )
    df["state"] = helper_mod._cell_state_codes(
        brdu_pos=df["brdu_pos"].to_numpy(dtype=int),
        edu_pos=df["edu_pos"].to_numpy(dtype=int),
    )
    if "Usage_7" in obs.columns:
        usage7 = pd.to_numeric(obs.iloc[idx]["Usage_7"], errors="coerce").to_numpy(dtype=float, copy=False)
        df["usage7"] = usage7[keep]
    return df


def _flatten_multinomial_params(params: np.ndarray) -> np.ndarray:
    return np.asarray(params, dtype=float).reshape(-1, order="F")


def _unflatten_multinomial_params(vec: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape(shape, order="F")


def _multinomial_exog(
    *,
    ap_mm: np.ndarray,
    ml_mm: np.ndarray,
    usage7: np.ndarray | None = None,
) -> np.ndarray:
    ap_vals = np.asarray(ap_mm, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_mm, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching AP/ML shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    cols = [
        np.ones_like(ap_vals, dtype=float),
        ap_vals,
        ml_vals,
    ]
    if usage7 is not None:
        usage_vals = np.asarray(usage7, dtype=float).reshape(-1)
        if usage_vals.shape != ap_vals.shape:
            raise ValueError(f"Expected usage7 shape {ap_vals.shape}, got {usage_vals.shape}")
        cols.append(usage_vals)
    return np.column_stack(cols)


def _fit_multinomial_logit_2d(df: pd.DataFrame, *, usage7_col: str | None = None) -> dict[str, Any]:
    state_arr = df["state"].to_numpy(dtype=int)
    unique = np.unique(state_arr)
    expected = np.asarray([0, 1, 2, 3], dtype=int)
    if unique.shape != expected.shape or not np.array_equal(unique, expected):
        raise ValueError(f"Expected all cell states {expected.tolist()}, got {unique.tolist()}")
    usage7 = None
    if usage7_col is not None:
        if usage7_col not in df.columns:
            raise KeyError(f"{usage7_col!r} not found in dataframe columns.")
        usage7 = df[usage7_col].to_numpy(dtype=float)
        if not np.isfinite(usage7).all():
            raise ValueError(f"{usage7_col!r} contains non-finite values.")
    exog = _multinomial_exog(
        ap_mm=df["ap_mm"].to_numpy(dtype=float),
        ml_mm=df["ml_mm"].to_numpy(dtype=float),
        usage7=usage7,
    )
    result = MNLogit(state_arr, exog).fit(disp=0, maxiter=200)
    return {
        "params": np.asarray(result.params, dtype=float),
        "cov": np.asarray(result.cov_params(), dtype=float),
        "usage7_col": usage7_col,
    }


def _predict_multinomial_probs_2d(
    *,
    params: np.ndarray,
    ap_mm: np.ndarray,
    ml_mm: np.ndarray,
    usage7: np.ndarray | None = None,
) -> np.ndarray:
    params_arr = np.asarray(params, dtype=float)
    needs_usage7 = int(params_arr.shape[0]) == 4
    if needs_usage7 and usage7 is None:
        raise ValueError("Model params require usage7 values for prediction.")
    if (not needs_usage7) and usage7 is not None:
        raise ValueError("usage7 was provided for a model that was fit without it.")
    exog = _multinomial_exog(ap_mm=ap_mm, ml_mm=ml_mm, usage7=usage7)
    eta = exog @ params_arr
    exp_eta = np.exp(np.clip(eta, -60.0, 60.0))
    denom = 1.0 + np.sum(exp_eta, axis=1, keepdims=True)
    return np.concatenate([1.0 / denom, exp_eta / denom], axis=1)


def _simulate_multinomial_probs_2d(
    *,
    fit: dict[str, Any],
    ap_mm: np.ndarray,
    ml_mm: np.ndarray,
    usage7: np.ndarray | None = None,
    seed: int,
    n_draws: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = np.asarray(fit["params"], dtype=float)
    draws = rng.multivariate_normal(
        mean=_flatten_multinomial_params(params),
        cov=np.asarray(fit["cov"], dtype=float),
        size=int(n_draws),
        check_valid="ignore",
    )
    draw_params = np.stack(
        [_unflatten_multinomial_params(draw, shape=params.shape) for draw in draws],
        axis=0,
    )
    needs_usage7 = int(params.shape[0]) == 4
    if needs_usage7 and usage7 is None:
        raise ValueError("Model params require usage7 values for simulation.")
    if (not needs_usage7) and usage7 is not None:
        raise ValueError("usage7 was provided for a model that was fit without it.")
    exog = _multinomial_exog(ap_mm=ap_mm, ml_mm=ml_mm, usage7=usage7)
    eta = np.einsum("nk,dkm->dnm", exog, draw_params)
    exp_eta = np.exp(np.clip(eta, -60.0, 60.0))
    denom = 1.0 + np.sum(exp_eta, axis=2, keepdims=True)
    return np.concatenate([1.0 / denom, exp_eta / denom], axis=2)


def _metrics_from_probs(probs: np.ndarray, *, delta_t_hours: float) -> dict[str, np.ndarray]:
    brdu_only = probs[..., 1]
    edu_only = probs[..., 2]
    dual = probs[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = float(delta_t_hours) * dual / brdu_only
        tc = ts / dual
    return {
        "brdu_only_frac": brdu_only,
        "edu_only_frac": edu_only,
        "dual_frac": dual,
        "Ts_hours": ts,
        "Tc_hours": tc,
    }


def _animal_support_positions(df: pd.DataFrame) -> dict[str, float]:
    return {
        "ap_q10": float(df["ap_mm"].quantile(0.10)),
        "ap_q90": float(df["ap_mm"].quantile(0.90)),
        "ap_med": float(df["ap_mm"].median()),
        "ml_q10": float(df["ml_mm"].quantile(0.10)),
        "ml_q90": float(df["ml_mm"].quantile(0.90)),
        "ml_med": float(df["ml_mm"].median()),
    }


def _slope_rows_from_fit(
    *,
    fit: dict[str, Any],
    support: dict[str, float],
    subset: str,
    grouping: str,
    animal: str,
    orientation: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    n_cells: int,
    usage7_eval: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    contrast_specs = (
        (
            "AP",
            np.array([support["ap_q10"], support["ap_q90"]]),
            np.array([support["ml_med"], support["ml_med"]]),
            float(support["ap_q90"] - support["ap_q10"]),
        ),
        (
            "ML",
            np.array([support["ap_med"], support["ap_med"]]),
            np.array([support["ml_q10"], support["ml_q90"]]),
            float(support["ml_q90"] - support["ml_q10"]),
        ),
    )
    for axis, ap_eval, ml_eval, span_mm in contrast_specs:
        if span_mm <= 0.0:
            raise ValueError(f"Expected positive q10-q90 span for animal={animal}, axis={axis}")
        point_probs = _predict_multinomial_probs_2d(
            params=np.asarray(fit["params"], dtype=float),
            ap_mm=ap_eval,
            ml_mm=ml_eval,
            usage7=None if usage7_eval is None else np.asarray(usage7_eval, dtype=float),
        )
        draw_probs = _simulate_multinomial_probs_2d(
            fit=fit,
            ap_mm=ap_eval,
            ml_mm=ml_eval,
            usage7=None if usage7_eval is None else np.asarray(usage7_eval, dtype=float),
            seed=seed + (0 if axis == "AP" else 1),
            n_draws=n_draws,
        )
        point_metrics = _metrics_from_probs(point_probs, delta_t_hours=delta_t_hours)
        draw_metrics = _metrics_from_probs(draw_probs, delta_t_hours=delta_t_hours)
        for metric in METRIC_ORDER:
            slope_point = float((point_metrics[metric][1] - point_metrics[metric][0]) / span_mm)
            slope_draws = np.asarray(
                (draw_metrics[metric][:, 1] - draw_metrics[metric][:, 0]) / span_mm,
                dtype=float,
            )
            rows.append(
                {
                    "subset": subset,
                    "grouping": grouping,
                    "animal": animal,
                    "orientation": orientation,
                    "axis": axis,
                    "metric": metric,
                    "estimate": slope_point,
                    "se": float(np.std(slope_draws, ddof=1)),
                    "ci_low": float(np.quantile(slope_draws, 0.025)),
                    "ci_high": float(np.quantile(slope_draws, 0.975)),
                    "span_mm": span_mm,
                    "q10_mm": float(support["ap_q10"] if axis == "AP" else support["ml_q10"]),
                    "q90_mm": float(support["ap_q90"] if axis == "AP" else support["ml_q90"]),
                    "fixed_other_mm": float(support["ml_med"] if axis == "AP" else support["ap_med"]),
                    "n_cells": int(n_cells),
                }
            )
    return rows


def _animal_contrast_rows(
    df: pd.DataFrame,
    *,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
) -> list[dict[str, object]]:
    animal_names = sorted(df["animal"].astype(str).unique().tolist())
    rows: list[dict[str, object]] = []
    for animal_idx, animal_name in enumerate(animal_names):
        dfa = df.loc[df["animal"] == animal_name].copy()
        orientations = sorted(dfa["orientation"].astype(str).unique().tolist())
        if len(orientations) != 1:
            raise ValueError(f"Expected one orientation for animal={animal_name}, got {orientations}")
        fit = _fit_multinomial_logit_2d(dfa)
        support = _animal_support_positions(dfa)
        rows.extend(
            _slope_rows_from_fit(
                fit=fit,
                support=support,
                subset=subset,
                grouping="animal",
                animal=animal_name,
                orientation=str(orientations[0]),
                delta_t_hours=delta_t_hours,
                n_draws=n_draws,
                seed=seed + animal_idx * 10,
                n_cells=int(dfa.shape[0]),
            )
        )
    return rows


def _meta_summary_rows(animal_df: pd.DataFrame, *, subset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in METRIC_ORDER:
        for axis in AXIS_ORDER:
            sub = animal_df.loc[(animal_df["metric"] == metric) & (animal_df["axis"] == axis)].copy()
            if sub.empty:
                continue
            meta = combine_effects(
                sub["estimate"].to_numpy(dtype=float),
                sub["se"].to_numpy(dtype=float) ** 2,
                method_re="iterated",
            )
            mean = float(meta.mean_effect_re)
            se = float(meta.sd_eff_w_re)
            rows.append(
                {
                    "subset": subset,
                    "grouping": "meta_random",
                    "animal": "all",
                    "orientation": "pooled",
                    "axis": axis,
                    "metric": metric,
                    "estimate": mean,
                    "se": se,
                    "ci_low": float(mean - 1.96 * se),
                    "ci_high": float(mean + 1.96 * se),
                    "tau2": float(meta.tau2),
                    "i2": float(max(0.0, meta.i2)),
                    "n_animals": int(sub.shape[0]),
                    "n_positive": int(np.sum(sub["estimate"].to_numpy(dtype=float) > 0)),
                    "n_negative": int(np.sum(sub["estimate"].to_numpy(dtype=float) < 0)),
                }
            )
    return rows


def _summarize_subset_multinomial(
    df: pd.DataFrame,
    *,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    animal_rows = pd.DataFrame(
        _animal_contrast_rows(
            df,
            subset=subset,
            delta_t_hours=delta_t_hours,
            n_draws=n_draws,
            seed=seed,
        )
    )
    meta_rows = pd.DataFrame(_meta_summary_rows(animal_rows, subset=subset))
    return animal_rows, meta_rows


def _summarize_subset_pooled_multinomial(
    df: pd.DataFrame,
    *,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    fit = _fit_multinomial_logit_2d(df)
    support = _animal_support_positions(df)
    return pd.DataFrame(
        _slope_rows_from_fit(
            fit=fit,
            support=support,
            subset=subset,
            grouping="pooled",
            animal="all",
            orientation="pooled",
            delta_t_hours=delta_t_hours,
            n_draws=n_draws,
            seed=seed,
            n_cells=int(df.shape[0]),
        )
    )


def _derivative_rows_from_fit(
    *,
    fit: dict[str, Any],
    subset: str,
    grouping: str,
    animal: str,
    orientation: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    n_cells: int,
    ap_mm: float,
    ml_mm: float,
    step_mm: float,
    usage7_value: float | None = None,
) -> list[dict[str, object]]:
    if step_mm <= 0.0:
        raise ValueError(f"Expected positive step_mm, got {step_mm}")
    rows: list[dict[str, object]] = []
    deriv_specs = (
        ("AP", np.array([ap_mm - step_mm, ap_mm + step_mm]), np.array([ml_mm, ml_mm])),
        ("ML", np.array([ap_mm, ap_mm]), np.array([ml_mm - step_mm, ml_mm + step_mm])),
    )
    for axis, ap_eval, ml_eval in deriv_specs:
        point_probs = _predict_multinomial_probs_2d(
            params=np.asarray(fit["params"], dtype=float),
            ap_mm=ap_eval,
            ml_mm=ml_eval,
            usage7=None if usage7_value is None else np.full(ap_eval.shape, float(usage7_value), dtype=float),
        )
        draw_probs = _simulate_multinomial_probs_2d(
            fit=fit,
            ap_mm=ap_eval,
            ml_mm=ml_eval,
            usage7=None if usage7_value is None else np.full(ap_eval.shape, float(usage7_value), dtype=float),
            seed=seed + (0 if axis == "AP" else 1),
            n_draws=n_draws,
        )
        point_metrics = _metrics_from_probs(point_probs, delta_t_hours=delta_t_hours)
        draw_metrics = _metrics_from_probs(draw_probs, delta_t_hours=delta_t_hours)
        for metric in METRIC_ORDER:
            deriv_point = float((point_metrics[metric][1] - point_metrics[metric][0]) / (2.0 * step_mm))
            deriv_draws = np.asarray(
                (draw_metrics[metric][:, 1] - draw_metrics[metric][:, 0]) / (2.0 * step_mm),
                dtype=float,
            )
            rows.append(
                {
                    "subset": subset,
                    "grouping": grouping,
                    "animal": animal,
                    "orientation": orientation,
                    "axis": axis,
                    "metric": metric,
                    "estimate": deriv_point,
                    "se": float(np.std(deriv_draws, ddof=1)),
                    "ci_low": float(np.quantile(deriv_draws, 0.025)),
                    "ci_high": float(np.quantile(deriv_draws, 0.975)),
                    "ap_eval_mm": float(ap_mm),
                    "ml_eval_mm": float(ml_mm),
                    "step_mm": float(step_mm),
                    "n_cells": int(n_cells),
                }
            )
    return rows


def _summarize_subset_pooled_local_derivatives(
    df: pd.DataFrame,
    *,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    step_mm: float = 0.001,
) -> pd.DataFrame:
    fit = _fit_multinomial_logit_2d(df)
    ap_med = float(df["ap_mm"].median())
    ml_med = float(df["ml_mm"].median())
    return pd.DataFrame(
        _derivative_rows_from_fit(
            fit=fit,
            subset=subset,
            grouping="pooled_local",
            animal="all",
            orientation="pooled",
            delta_t_hours=delta_t_hours,
            n_draws=n_draws,
            seed=seed,
            n_cells=int(df.shape[0]),
            ap_mm=ap_med,
            ml_mm=ml_med,
            step_mm=float(step_mm),
        )
    )


def _evaluate_pooled_multinomial_at_queries(
    df: pd.DataFrame,
    *,
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    ap_mm: np.ndarray,
    ml_mm: np.ndarray,
    usage7: np.ndarray | None = None,
    usage7_col: str | None = None,
) -> pd.DataFrame:
    """Evaluate the pooled multinomial fit at arbitrary AP/ML query points."""
    ap_vals = np.asarray(ap_mm, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_mm, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching query shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    usage7_vals = None
    if usage7 is not None:
        usage7_vals = np.asarray(usage7, dtype=float).reshape(-1)
        if usage7_vals.shape != ap_vals.shape:
            raise ValueError(f"Expected usage7 query shape {ap_vals.shape}, got {usage7_vals.shape}")

    fit = _fit_multinomial_logit_2d(df, usage7_col=usage7_col)
    return _evaluate_multinomial_fit_at_queries(
        fit=fit,
        subset=subset,
        delta_t_hours=delta_t_hours,
        n_draws=n_draws,
        seed=seed,
        ap_mm=ap_vals,
        ml_mm=ml_vals,
        usage7=usage7_vals,
    )


def _evaluate_multinomial_fit_at_queries(
    *,
    fit: dict[str, Any],
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    ap_mm: np.ndarray,
    ml_mm: np.ndarray,
    usage7: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate an already-fit pooled multinomial model at arbitrary query points."""
    ap_vals = np.asarray(ap_mm, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_mm, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching query shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    usage7_vals = None
    if usage7 is not None:
        usage7_vals = np.asarray(usage7, dtype=float).reshape(-1)
        if usage7_vals.shape != ap_vals.shape:
            raise ValueError(f"Expected usage7 query shape {ap_vals.shape}, got {usage7_vals.shape}")

    point_probs = _predict_multinomial_probs_2d(
        params=np.asarray(fit["params"], dtype=float),
        ap_mm=ap_vals,
        ml_mm=ml_vals,
        usage7=usage7_vals,
    )
    draw_probs = _simulate_multinomial_probs_2d(
        fit=fit,
        ap_mm=ap_vals,
        ml_mm=ml_vals,
        usage7=usage7_vals,
        seed=seed,
        n_draws=n_draws,
    )
    point_metrics = _metrics_from_probs(point_probs, delta_t_hours=delta_t_hours)
    draw_metrics = _metrics_from_probs(draw_probs, delta_t_hours=delta_t_hours)

    out: dict[str, np.ndarray | list[str]] = {
        "subset": [subset] * int(ap_vals.size),
        "ap_mm": ap_vals,
        "ml_mm": ml_vals,
    }
    metric_map = (
        ("dual_frac", "dual_frac"),
        ("Ts_hours", "Ts_hours"),
        ("Tc_hours", "Tc_hours"),
    )
    for metric_key, prefix in metric_map:
        point = np.asarray(point_metrics[metric_key], dtype=float).reshape(-1)
        draws = np.asarray(draw_metrics[metric_key], dtype=float)
        out[f"{prefix}_mean"] = point
        out[f"{prefix}_ci_low"] = np.quantile(draws, 0.025, axis=0)
        out[f"{prefix}_ci_high"] = np.quantile(draws, 0.975, axis=0)
    return pd.DataFrame(out)


def _write_summary_text(meta_df: pd.DataFrame, *, subset_title: str, out_txt: pathlib.Path) -> None:
    lines = [subset_title]
    for metric in METRIC_ORDER:
        lines.append(f"\n{metric}")
        sub = meta_df.loc[meta_df["metric"] == metric].copy()
        for axis in AXIS_ORDER:
            row = sub.loc[sub["axis"] == axis]
            if row.empty:
                continue
            rec = row.iloc[0]
            lines.append(
                (
                    f"  {axis}: slope={rec['estimate']:.4g}/mm "
                    f"[{rec['ci_low']:.4g}, {rec['ci_high']:.4g}], "
                    f"tau2={rec['tau2']:.4g}, I2={rec['i2']:.3f}, "
                    f"agree={int(rec['n_positive'])}/{int(rec['n_animals'])} positive"
                )
            )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    helper_mod = _load_plot_apml_module()
    clusters = {c.strip() for c in str(args.clusters).split(",") if c.strip()}
    exclude_animals = {a.strip() for a in str(args.exclude_animals).split(",") if a.strip()}
    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    adata = helper_mod._load_adata_with_external_obsm(
        h5ad_path=args.h5ad.expanduser(),
        obsm_h5ad_path=args.obsm_h5ad.expanduser(),
    )

    for spec in DEFAULT_SUBSETS:
        subset = str(spec["label"])
        df = _build_model_df(
            helper_mod=helper_mod,
            adata=adata,
            clusters=clusters,
            manual_layer=str(spec["manual_layer"]),
            eomes_gt=spec["eomes_gt"],
            eomes_lt=spec["eomes_lt"],
            label=subset,
            exclude_animals=exclude_animals,
        )
        animal_df, meta_df = _summarize_subset_multinomial(
            df,
            subset=subset,
            delta_t_hours=float(args.delta_t_hours),
            n_draws=int(args.n_draws),
            seed=int(args.seed),
        )
        stem = f"multinomial_animal_meta_{subset}"
        animal_csv = outdir / f"{stem}__animal.csv"
        meta_csv = outdir / f"{stem}__meta.csv"
        summary_txt = outdir / f"{stem}.txt"
        animal_df.to_csv(animal_csv, index=False)
        meta_df.to_csv(meta_csv, index=False)
        _write_summary_text(meta_df, subset_title=str(spec["title"]), out_txt=summary_txt)
        print(f"wrote {animal_csv}")
        print(f"wrote {meta_csv}")
        print(f"wrote {summary_txt}")


if __name__ == "__main__":
    main()
