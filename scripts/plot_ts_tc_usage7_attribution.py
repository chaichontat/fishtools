#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from ccf.transforms import DEFAULT_NATIVE_CAMERA_AZIM_DEG
from ccf.transforms import DEFAULT_NATIVE_CAMERA_ELEV_DEG
from ccf.transforms import DEFAULT_NATIVE_CAMERA_ROLL_DEG
from ccf.transforms import build_apml_native_surface_projection_context


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MULTINOMIAL_PATH = SCRIPT_DIR / "fit_apml_multinomial_animal_meta.py"
NATIVE_PATH = SCRIPT_DIR / "plot_ts_tc_native_with_sagittal_line.py"
LINE_PATH = SCRIPT_DIR / "plot_ts_tc_vs_sagittal_t_pooled.py"
SIMPLEX_NATIVE_PATH = SCRIPT_DIR / "gam" / "plot_simplex_native_proj.py"
DEFAULT_H5AD = pathlib.Path("~/nvme/all_excit.h5ad")
DEFAULT_OBSM_H5AD = pathlib.Path("~/nvme/obsm.h5ad")
DEFAULT_REFEXTRACT_OUTDIR = pathlib.Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d")
DEFAULT_OUTDIR = pathlib.Path("scripts/_out/apml_usage7_attribution")
DEFAULT_USAGE7_GAM_DIR = pathlib.Path("scripts/_out/gam/vzsvz_manual_layer1_simplex_20260322_142901")
DEFAULT_SUBSET = "manual_layer_1"
DEFAULT_BIN_WIDTH_UM = 200.0
DEFAULT_USAGE7_COL = "usage7"
DEFAULT_USAGE7_BASELINE_QUANTILE = 0.05
DEFAULT_SLICE_KS = (160, 184, 208)
DEFAULT_AP_CLIP_LOW_PCT = 2.5
DEFAULT_AP_CLIP_HIGH_PCT = 97.5
DEFAULT_LINE_T_MIN = 0.25
DEFAULT_LINE_T_MAX = 0.75
DEFAULT_REFEXTRACT_SLICE_I_MIN = 161
DEFAULT_REFEXTRACT_SLICE_I_MAX = 305
DEFAULT_REFEXTRACT_N_T = 257
DEFAULT_REFEXTRACT_REF_T = 0.5
DEFAULT_REFEXTRACT_BAND_FRAC = 0.15
DEFAULT_REFEXTRACT_RES_IJK_UM = (20.0, 20.0, 20.0)

sns.set_theme()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify layer-1 Ts/Tc attribution from Usage_7 by fitting pooled "
            "state ~ AP + ML + Usage_7 and comparing factual bin-mean Usage_7 "
            "against a low-quantile baseline."
        )
    )
    parser.add_argument("--h5ad", type=pathlib.Path, default=DEFAULT_H5AD)
    parser.add_argument("--obsm-h5ad", type=pathlib.Path, default=DEFAULT_OBSM_H5AD)
    parser.add_argument("--clusters", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--exclude-animals", type=str, default="JaxA2")
    parser.add_argument("--delta-t-hours", type=float, default=1.5)
    parser.add_argument("--n-draws", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bin-width-um", type=float, default=DEFAULT_BIN_WIDTH_UM)
    parser.add_argument("--usage7-baseline-quantile", type=float, default=DEFAULT_USAGE7_BASELINE_QUANTILE)
    parser.add_argument("--usage7-gam-dir", type=pathlib.Path, default=DEFAULT_USAGE7_GAM_DIR)
    parser.add_argument("--slice-ks", type=str, default="160,184,208")
    parser.add_argument("--line-t-min", type=float, default=DEFAULT_LINE_T_MIN)
    parser.add_argument("--line-t-max", type=float, default=DEFAULT_LINE_T_MAX)
    parser.add_argument("--refextract-outdir", type=pathlib.Path, default=DEFAULT_REFEXTRACT_OUTDIR)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_slice_ks(slice_ks: str) -> list[int]:
    parsed: list[int] = []
    seen: set[int] = set()
    for part in str(slice_ks).split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value in seen:
            continue
        parsed.append(value)
        seen.add(value)
    if not parsed:
        raise ValueError("--slice-ks must contain at least one integer slice index.")
    return parsed


def _output_stem(*, usage7_baseline: float | None = None, usage7_baseline_quantile: float | None = None) -> str:
    if usage7_baseline_quantile is not None:
        pct = int(round(float(usage7_baseline_quantile) * 100.0))
        return f"{DEFAULT_SUBSET}__Usage7_vs_baseline_q{pct:02d}"
    if usage7_baseline is None:
        raise ValueError("Expected either usage7_baseline or usage7_baseline_quantile.")
    baseline_tag = str(float(usage7_baseline)).replace(".", "p")
    return f"{DEFAULT_SUBSET}__Usage7_vs_baseline_{baseline_tag}"


def _resolve_usage7_gam_panel_dir(path: pathlib.Path) -> pathlib.Path:
    candidate = path.expanduser()
    panel_dir = candidate / "panel"
    return panel_dir if panel_dir.is_dir() else candidate


def _circular_theta_marginal_values(theta: np.ndarray) -> np.ndarray:
    theta_vals = np.asarray(theta, dtype=float).reshape(-1)
    theta_vals = theta_vals[np.isfinite(theta_vals)]
    if theta_vals.size == 0:
        return np.asarray([0.0], dtype=float)
    mean_angle = float(np.angle(np.mean(np.exp(1j * theta_vals))))
    offsets = np.angle(np.exp(1j * (theta_vals - mean_angle)))
    q = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
    values = mean_angle + np.quantile(offsets, q)
    return np.mod(values, 2.0 * np.pi).astype(float, copy=False)


def _load_usage7_gam_bundle(*, gam_dir: pathlib.Path, exclude_animals: set[str]) -> dict[str, Any]:
    simplex_mod = _load_module("plot_simplex_native_proj", SIMPLEX_NATIVE_PATH)
    panel_dir = _resolve_usage7_gam_panel_dir(gam_dir)
    meta_path = simplex_mod._default_meta_path(panel_dir)
    meta = simplex_mod._load_meta(meta_path)
    topic_ids = [int(x) for x in meta["topic_ids"]]
    if 7 not in set(topic_ids):
        raise ValueError(f"Topic 7 not found in {meta_path}")
    transform = str(meta.get("transform") or "alr")
    if transform != "ilr":
        raise ValueError(f"Expected ILR simplex GAM for Usage_7, got transform={transform!r}")
    coord_names = meta.get("coord_names")
    if not isinstance(coord_names, list) or not coord_names:
        raise ValueError(f"{meta_path}: missing/invalid coord_names for ILR")
    fits_dir = pathlib.Path(str(meta["fits_dir"])).expanduser()
    if not fits_dir.is_dir():
        raise FileNotFoundError(f"fits_dir not found: {fits_dir}")
    cells = pd.read_csv(panel_dir / "cells.tsv", sep="\t")
    required = {"r_um", "AP_um", "ML_um"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"{panel_dir / 'cells.tsv'}: missing required columns: {missing}")
    r_um = pd.to_numeric(cells["r_um"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(r_um)):
        raise ValueError("cells.tsv r_um must be finite")
    theta = (
        pd.to_numeric(cells["theta"], errors="coerce").to_numpy(dtype=float)
        if "theta" in cells.columns
        else np.zeros(cells.shape[0], dtype=float)
    )
    if not np.all(np.isfinite(theta)):
        raise ValueError("cells.tsv theta must be finite")
    predictor = simplex_mod.RPredictor()
    fit_infos: list[tuple[object, list[str], str | None]] = []
    batch_series = pd.Series(cells["batch"]) if "batch" in cells.columns else pd.Series(dtype=object)
    animal_ref_meta = meta.get("animal_ref")
    animal_ref_meta_str = None if animal_ref_meta is None else str(animal_ref_meta)
    for fit_name in [str(x) for x in coord_names]:
        fit = predictor.read_fit(fits_dir / f"{fit_name}.gam.rds")
        animal_levels = predictor.animal_levels(fit)
        animal_ref_default = animal_ref_meta_str
        if animal_levels and (animal_ref_default is None or animal_ref_default not in animal_levels):
            animal_ref_default = simplex_mod._infer_animal_ref_from_batch_values(batch_series, animal_levels)
        if animal_levels and animal_ref_default is None:
            animal_ref_default = animal_levels[0]
        fit_infos.append((fit, animal_levels, animal_ref_default))
    first_levels = fit_infos[0][1] if fit_infos else []
    marginal_animal_levels = [level for level in first_levels if level not in exclude_animals]
    if first_levels and not marginal_animal_levels:
        raise ValueError("No animal levels remain for Usage_7 GAM after exclusion filter.")
    return {
        "simplex_mod": simplex_mod,
        "predictor": predictor,
        "fit_infos": fit_infos,
        "topic_ids": topic_ids,
        "topic_index": int(topic_ids.index(7)),
        "inverse_fn": lambda z: simplex_mod.inv_ilr(z, V=simplex_mod.ilr_basis_pivot(len(topic_ids))),
        "batch_ref": None if meta.get("batch_ref") is None else str(meta.get("batch_ref")),
        "r_values": simplex_mod._native_r_marginal_values(r_um=r_um, fit_r_max=meta.get("r_max")),
        "theta_values": _circular_theta_marginal_values(theta),
        "marginal_animal_levels": marginal_animal_levels,
    }


def _predict_usage7_gam(
    *,
    bundle: dict[str, Any],
    ap_um: np.ndarray,
    ml_um: np.ndarray,
) -> np.ndarray:
    ap_vals = np.asarray(ap_um, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_um, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching AP/ML shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    out = np.full(ap_vals.shape, np.nan, dtype=float)
    finite = np.isfinite(ap_vals) & np.isfinite(ml_vals)
    if not np.any(finite):
        return out

    predictor = bundle["predictor"]
    simplex_mod = bundle["simplex_mod"]
    fit_infos = bundle["fit_infos"]
    ap_query = ap_vals[finite]
    ml_query = ml_vals[finite]
    theta_values = np.asarray(bundle["theta_values"], dtype=float)
    r_values = np.asarray(bundle["r_values"], dtype=float)
    animal_levels = list(bundle["marginal_animal_levels"]) if bundle["marginal_animal_levels"] else [None]
    n_topics = int(len(bundle["topic_ids"]))
    u_acc = np.zeros((ap_query.size, n_topics), dtype=float)
    n_states = 0
    for animal_level in animal_levels:
        for r_val in r_values:
            for theta_val in theta_values:
                z_hat = np.zeros((ap_query.size, len(fit_infos)), dtype=float)
                for j, (fit, levels, animal_ref_default) in enumerate(fit_infos):
                    animal_ref = str(animal_level) if animal_level is not None else animal_ref_default
                    config = simplex_mod.GAMPredictorConfig(
                        mean_log_sf=None,
                        include_pos=False,
                        include_batch=False,
                        batch_ref=bundle["batch_ref"],
                        batch_levels=[],
                        include_animal=bool(levels),
                        animal_ref=animal_ref,
                        animal_levels=levels,
                        include_log_sf_c=False,
                    )
                    nd = simplex_mod.make_newdata_for_fit(
                        predictor=predictor,
                        fit=fit,
                        n=int(ap_query.size),
                        r_um=float(r_val),
                        theta=float(theta_val),
                        ap_um=ap_query,
                        ml_um=ml_query,
                        config=config,
                    )
                    z_hat[:, j] = simplex_mod._predict_link_excluding_terms(
                        predictor=predictor,
                        fit=fit,
                        newdata=nd,
                        exclude_terms=("s(ab)",),
                    )
                u_acc += bundle["inverse_fn"](z_hat)
                n_states += 1
    out[np.flatnonzero(finite)] = u_acc[:, int(bundle["topic_index"])] / float(n_states)
    return out


def _usage7_gam_cache_signature(bundle: dict[str, Any]) -> str:
    payload = {
        "marginal_animal_levels": list(bundle["marginal_animal_levels"]),
        "r_values": np.asarray(bundle["r_values"], dtype=float).round(6).tolist(),
        "theta_values": np.asarray(bundle["theta_values"], dtype=float).round(6).tolist(),
        "topic_index": int(bundle["topic_index"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_or_predict_usage7_gam(
    *,
    cache_path: pathlib.Path,
    bundle: dict[str, Any],
    ap_um: np.ndarray,
    ml_um: np.ndarray,
) -> np.ndarray:
    ap_vals = np.asarray(ap_um, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_um, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching AP/ML shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    cache_path = cache_path.expanduser()
    cache_signature = _usage7_gam_cache_signature(bundle)
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_ap = np.asarray(cached["AP_um"], dtype=float).reshape(-1)
        cached_ml = np.asarray(cached["ML_um"], dtype=float).reshape(-1)
        cached_usage7 = np.asarray(cached["usage7_gam"], dtype=float).reshape(-1)
        cached_signature = str(np.asarray(cached["cache_signature"]).reshape(-1)[0]) if "cache_signature" in cached else ""
        if cached_ap.shape == ap_vals.shape and cached_ml.shape == ml_vals.shape:
            if (
                cached_signature == cache_signature
                and np.allclose(cached_ap, ap_vals, equal_nan=True)
                and np.allclose(cached_ml, ml_vals, equal_nan=True)
            ):
                return cached_usage7
    usage7_gam = _predict_usage7_gam(bundle=bundle, ap_um=ap_vals, ml_um=ml_vals)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        AP_um=ap_vals,
        ML_um=ml_vals,
        usage7_gam=usage7_gam,
        cache_signature=np.asarray([cache_signature]),
    )
    return usage7_gam


def _build_usage7_interpolator(
    *,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    usage7: np.ndarray,
):
    ap_all = np.asarray(ap_um, dtype=np.float64).reshape(-1)
    ml_all = np.asarray(ml_um, dtype=np.float64).reshape(-1)
    usage_all = np.asarray(usage7, dtype=np.float64).reshape(-1)
    if not (ap_all.shape == ml_all.shape == usage_all.shape):
        raise ValueError("AP, ML, and Usage_7 arrays must share the same shape.")
    finite = np.isfinite(ap_all) & np.isfinite(ml_all) & np.isfinite(usage_all)
    ap_all = ap_all[finite]
    ml_all = ml_all[finite]
    usage_all = usage_all[finite]
    if ap_all.size < 3:
        raise ValueError("Need at least three finite AP/ML support points for Usage_7 interpolation.")
    triang = mtri.Triangulation(ap_all, ml_all)
    interp = mtri.LinearTriInterpolator(triang, usage_all)

    def _evaluate(*, ap_um: np.ndarray, ml_um: np.ndarray) -> np.ndarray:
        ap_query = np.asarray(ap_um, dtype=np.float64)
        ml_query = np.asarray(ml_um, dtype=np.float64)
        if ap_query.shape != ml_query.shape:
            raise ValueError(f"Expected matching AP/ML shapes, got {ap_query.shape} vs {ml_query.shape}")
        out_shape = ap_query.shape
        vals = interp(ap_query.reshape(-1), ml_query.reshape(-1))
        if np.ma.isMaskedArray(vals):
            flat = np.asarray(vals.filled(np.nan), dtype=float)
        else:
            flat = np.asarray(vals, dtype=float)
        return flat.reshape(out_shape)

    return _evaluate


def _build_usage7_bin_summary(
    *,
    helper_mod: Any,
    df: pd.DataFrame,
    bin_width_um: float,
) -> dict[str, Any]:
    if DEFAULT_USAGE7_COL not in df.columns:
        raise KeyError(f"{DEFAULT_USAGE7_COL!r} column is required for Usage_7 attribution.")
    ap_um = df["ap_mm"].to_numpy(dtype=float) * 1000.0
    ml_um = df["ml_mm"].to_numpy(dtype=float) * 1000.0
    usage7 = df[DEFAULT_USAGE7_COL].to_numpy(dtype=float)
    finite = np.isfinite(ap_um) & np.isfinite(ml_um) & np.isfinite(usage7)
    if not np.any(finite):
        raise ValueError("No finite AP/ML/Usage_7 rows remain for binning.")
    ap_um = ap_um[finite]
    ml_um = ml_um[finite]
    usage7 = usage7[finite]

    ap_edges, ap_centers = helper_mod._bins_for(ap_um, bin_width_um=float(bin_width_um))
    ml_edges, ml_centers = helper_mod._bins_for(ml_um, bin_width_um=float(bin_width_um))
    ap_idx = np.digitize(ap_um, ap_edges, right=False) - 1
    ml_idx = np.digitize(ml_um, ml_edges, right=False) - 1
    n_ap = ap_edges.size - 1
    n_ml = ml_edges.size - 1
    in_range = (ap_idx >= 0) & (ap_idx < n_ap) & (ml_idx >= 0) & (ml_idx < n_ml)
    ap_idx = ap_idx[in_range]
    ml_idx = ml_idx[in_range]
    usage7 = usage7[in_range]
    flat = (ap_idx * n_ml) + ml_idx
    n_total = np.bincount(flat, minlength=n_ap * n_ml).reshape(n_ap, n_ml)
    usage7_sum = np.bincount(flat, weights=usage7, minlength=n_ap * n_ml).reshape(n_ap, n_ml)
    mean_usage7 = np.full((n_ap, n_ml), np.nan, dtype=float)
    np.divide(usage7_sum, n_total, out=mean_usage7, where=n_total > 0)
    ap_grid, ml_grid = np.meshgrid(ap_centers, ml_centers, indexing="ij")
    table = pd.DataFrame(
        {
            "ap_center_um": ap_grid.ravel(),
            "ml_center_um": ml_grid.ravel(),
            "n_total": n_total.ravel(),
            "mean_usage7": mean_usage7.ravel(),
        }
    )
    return {
        "ap_edges": ap_edges,
        "ml_edges": ml_edges,
        "ap_centers": ap_centers,
        "ml_centers": ml_centers,
        "n_total": n_total,
        "mean_usage7": mean_usage7,
        "table": table,
    }


def _lookup_binned_usage7(
    *,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    bin_summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    ap_vals = np.asarray(ap_um, dtype=float).reshape(-1)
    ml_vals = np.asarray(ml_um, dtype=float).reshape(-1)
    if ap_vals.shape != ml_vals.shape:
        raise ValueError(f"Expected matching AP/ML shapes, got {ap_vals.shape} vs {ml_vals.shape}")
    ap_edges = np.asarray(bin_summary["ap_edges"], dtype=float)
    ml_edges = np.asarray(bin_summary["ml_edges"], dtype=float)
    n_total = np.asarray(bin_summary["n_total"], dtype=float)
    mean_usage7 = np.asarray(bin_summary["mean_usage7"], dtype=float)
    ap_idx = np.digitize(ap_vals, ap_edges, right=False) - 1
    ml_idx = np.digitize(ml_vals, ml_edges, right=False) - 1
    usage7 = np.full(ap_vals.shape, np.nan, dtype=float)
    counts = np.zeros(ap_vals.shape, dtype=int)
    in_range = (ap_idx >= 0) & (ap_idx < n_total.shape[0]) & (ml_idx >= 0) & (ml_idx < n_total.shape[1])
    if np.any(in_range):
        row = ap_idx[in_range]
        col = ml_idx[in_range]
        counts_in = n_total[row, col].astype(int, copy=False)
        usage7_in = mean_usage7[row, col]
        valid = counts_in > 0
        usage7[np.flatnonzero(in_range)[valid]] = usage7_in[valid]
        counts[np.flatnonzero(in_range)] = counts_in
    return usage7, counts


def _nearest_populated_usage7(
    *,
    ap_um: float,
    ml_um: float,
    bin_summary: dict[str, Any],
    usage7_col: str,
) -> float:
    table = pd.DataFrame(bin_summary["table"]).copy()
    table = table.loc[(table["n_total"] > 0) & np.isfinite(table[usage7_col])].copy()
    if table.empty:
        raise ValueError("No populated Usage_7 bins available for reference lookup.")
    delta_ap = table["ap_center_um"].to_numpy(dtype=float) - float(ap_um)
    delta_ml = table["ml_center_um"].to_numpy(dtype=float) - float(ml_um)
    nearest = int(np.argmin((delta_ap * delta_ap) + (delta_ml * delta_ml)))
    return float(table.iloc[nearest][usage7_col])


def _weighted_variance(*, values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(finite):
        raise ValueError("Need at least one finite positive-weight value for weighted variance.")
    vals = vals[finite]
    w = w[finite]
    mean = float(np.average(vals, weights=w))
    return float(np.average((vals - mean) ** 2, weights=w))


def _evaluate_factual_zero_delta(
    *,
    meta_mod: Any,
    fit: dict[str, Any],
    subset: str,
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    usage7_mean: np.ndarray,
    usage7_baseline: float,
) -> pd.DataFrame:
    ap_vals_um = np.asarray(ap_um, dtype=float).reshape(-1)
    ml_vals_um = np.asarray(ml_um, dtype=float).reshape(-1)
    usage_vals = np.asarray(usage7_mean, dtype=float).reshape(-1)
    if not (ap_vals_um.shape == ml_vals_um.shape == usage_vals.shape):
        raise ValueError("AP, ML, and usage7_mean must share the same shape.")
    out = pd.DataFrame(
        {
            "subset": [subset] * int(ap_vals_um.size),
            "AP_um": ap_vals_um,
            "ML_um": ml_vals_um,
            "usage7_mean": usage_vals,
        }
    )
    valid = np.isfinite(ap_vals_um) & np.isfinite(ml_vals_um) & np.isfinite(usage_vals)
    metrics = ("Ts_hours", "Tc_hours")
    for metric in metrics:
        prefix = metric.replace("_hours", "")
        out[f"{prefix}_factual_mean"] = np.nan
        out[f"{prefix}_factual_ci_low"] = np.nan
        out[f"{prefix}_factual_ci_high"] = np.nan
        out[f"{prefix}_zero_mean"] = np.nan
        out[f"{prefix}_zero_ci_low"] = np.nan
        out[f"{prefix}_zero_ci_high"] = np.nan
        out[f"{prefix}_delta_mean"] = np.nan
        out[f"{prefix}_delta_ci_low"] = np.nan
        out[f"{prefix}_delta_ci_high"] = np.nan
    if not np.any(valid):
        return out

    ap_vals_mm = ap_vals_um[valid] / 1000.0
    ml_vals_mm = ml_vals_um[valid] / 1000.0
    usage_fact = usage_vals[valid]
    usage_zero = np.full(usage_fact.shape, float(usage7_baseline), dtype=float)
    point_probs_factual = meta_mod._predict_multinomial_probs_2d(
        params=np.asarray(fit["params"], dtype=float),
        ap_mm=ap_vals_mm,
        ml_mm=ml_vals_mm,
        usage7=usage_fact,
    )
    point_probs_zero = meta_mod._predict_multinomial_probs_2d(
        params=np.asarray(fit["params"], dtype=float),
        ap_mm=ap_vals_mm,
        ml_mm=ml_vals_mm,
        usage7=usage_zero,
    )
    draw_probs_factual = meta_mod._simulate_multinomial_probs_2d(
        fit=fit,
        ap_mm=ap_vals_mm,
        ml_mm=ml_vals_mm,
        usage7=usage_fact,
        seed=int(seed),
        n_draws=int(n_draws),
    )
    draw_probs_zero = meta_mod._simulate_multinomial_probs_2d(
        fit=fit,
        ap_mm=ap_vals_mm,
        ml_mm=ml_vals_mm,
        usage7=usage_zero,
        seed=int(seed),
        n_draws=int(n_draws),
    )
    point_metrics_factual = meta_mod._metrics_from_probs(point_probs_factual, delta_t_hours=float(delta_t_hours))
    point_metrics_zero = meta_mod._metrics_from_probs(point_probs_zero, delta_t_hours=float(delta_t_hours))
    draw_metrics_factual = meta_mod._metrics_from_probs(draw_probs_factual, delta_t_hours=float(delta_t_hours))
    draw_metrics_zero = meta_mod._metrics_from_probs(draw_probs_zero, delta_t_hours=float(delta_t_hours))
    valid_idx = np.flatnonzero(valid)
    for metric in metrics:
        prefix = metric.replace("_hours", "")
        point_fact = np.asarray(point_metrics_factual[metric], dtype=float).reshape(-1)
        point_zero = np.asarray(point_metrics_zero[metric], dtype=float).reshape(-1)
        draw_fact = np.asarray(draw_metrics_factual[metric], dtype=float)
        draw_zero = np.asarray(draw_metrics_zero[metric], dtype=float)
        draw_delta = draw_fact - draw_zero
        out.loc[valid_idx, f"{prefix}_factual_mean"] = point_fact
        out.loc[valid_idx, f"{prefix}_factual_ci_low"] = np.quantile(draw_fact, 0.025, axis=0)
        out.loc[valid_idx, f"{prefix}_factual_ci_high"] = np.quantile(draw_fact, 0.975, axis=0)
        out.loc[valid_idx, f"{prefix}_zero_mean"] = point_zero
        out.loc[valid_idx, f"{prefix}_zero_ci_low"] = np.quantile(draw_zero, 0.025, axis=0)
        out.loc[valid_idx, f"{prefix}_zero_ci_high"] = np.quantile(draw_zero, 0.975, axis=0)
        out.loc[valid_idx, f"{prefix}_delta_mean"] = point_fact - point_zero
        out.loc[valid_idx, f"{prefix}_delta_ci_low"] = np.quantile(draw_delta, 0.025, axis=0)
        out.loc[valid_idx, f"{prefix}_delta_ci_high"] = np.quantile(draw_delta, 0.975, axis=0)
    return out


def _reference_usage7_value(*, df: pd.DataFrame, bin_summary: dict[str, Any], usage7_col: str) -> float:
    ap_ref_um = float(df["ap_mm"].median() * 1000.0)
    ml_ref_um = float(df["ml_mm"].median() * 1000.0)
    table = pd.DataFrame(bin_summary["table"]).copy()
    ap_edges = np.asarray(bin_summary["ap_edges"], dtype=float)
    ml_edges = np.asarray(bin_summary["ml_edges"], dtype=float)
    ap_idx = int(np.digitize(np.asarray([ap_ref_um]), ap_edges, right=False)[0] - 1)
    ml_idx = int(np.digitize(np.asarray([ml_ref_um]), ml_edges, right=False)[0] - 1)
    n_ap = int(len(ap_edges) - 1)
    n_ml = int(len(ml_edges) - 1)
    if 0 <= ap_idx < n_ap and 0 <= ml_idx < n_ml:
        row = table.loc[
            (table["ap_center_um"] == float(bin_summary["ap_centers"][ap_idx]))
            & (table["ml_center_um"] == float(bin_summary["ml_centers"][ml_idx]))
        ]
        if not row.empty and int(row.iloc[0]["n_total"]) > 0 and np.isfinite(float(row.iloc[0][usage7_col])):
            return float(row.iloc[0][usage7_col])
    return _nearest_populated_usage7(ap_um=ap_ref_um, ml_um=ml_ref_um, bin_summary=bin_summary, usage7_col=usage7_col)


def _baseline_usage7_quantile_value(*, bin_summary: dict[str, Any], usage7_col: str, quantile: float) -> float:
    if not (0.0 <= float(quantile) <= 1.0):
        raise ValueError(f"Expected quantile in [0,1], got {quantile}")
    table = pd.DataFrame(bin_summary["table"]).copy()
    table = table.loc[(table["n_total"] > 0) & np.isfinite(table[usage7_col])].copy()
    if table.empty:
        raise ValueError("No populated Usage_7 bins available for baseline lookup.")
    return float(np.quantile(table[usage7_col].to_numpy(dtype=float), float(quantile)))


def _build_summary_rows(
    *,
    meta_mod: Any,
    fit: dict[str, Any],
    df: pd.DataFrame,
    subset: str,
    bin_summary: dict[str, Any],
    delta_t_hours: float,
    n_draws: int,
    seed: int,
    usage7_baseline: float,
    usage7_col: str,
) -> pd.DataFrame:
    table = pd.DataFrame(bin_summary["table"]).copy()
    table = table.loc[(table["n_total"] > 0) & np.isfinite(table[usage7_col])].copy()
    surface_eval = _evaluate_factual_zero_delta(
        meta_mod=meta_mod,
        fit=fit,
        subset=subset,
        delta_t_hours=delta_t_hours,
        n_draws=n_draws,
        seed=seed,
        ap_um=table["ap_center_um"].to_numpy(dtype=float),
        ml_um=table["ml_center_um"].to_numpy(dtype=float),
        usage7_mean=table[usage7_col].to_numpy(dtype=float),
        usage7_baseline=usage7_baseline,
    )
    rows: list[dict[str, object]] = []
    weights = table["n_total"].to_numpy(dtype=float)
    for prefix in ("Ts", "Tc"):
        factual = surface_eval[f"{prefix}_factual_mean"].to_numpy(dtype=float)
        zero = surface_eval[f"{prefix}_zero_mean"].to_numpy(dtype=float)
        var_factual = _weighted_variance(values=factual, weights=weights)
        var_zero = _weighted_variance(values=zero, weights=weights)
        frac_removed = np.nan if var_factual <= 0 else float(1.0 - (var_zero / var_factual))
        rows.extend(
            [
                {
                    "summary_type": "variance",
                    "metric": f"{prefix}_hours",
                    "axis": "surface",
                    "scenario": "factual",
                    "estimate": var_factual,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_bins": int(table.shape[0]),
                    "weight_total": float(np.sum(weights)),
                    "reference_usage7": np.nan,
                },
                {
                    "summary_type": "variance",
                    "metric": f"{prefix}_hours",
                    "axis": "surface",
                    "scenario": "zero_baseline",
                    "estimate": var_zero,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_bins": int(table.shape[0]),
                    "weight_total": float(np.sum(weights)),
                    "reference_usage7": np.nan,
                },
                {
                    "summary_type": "variance_reduction",
                    "metric": f"{prefix}_hours",
                    "axis": "surface",
                    "scenario": "factual_vs_zero",
                    "estimate": frac_removed,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_bins": int(table.shape[0]),
                    "weight_total": float(np.sum(weights)),
                    "reference_usage7": np.nan,
                },
            ]
        )

    usage7_ref = _reference_usage7_value(df=df, bin_summary=bin_summary, usage7_col=usage7_col)
    derivative_factual = pd.DataFrame(
        meta_mod._derivative_rows_from_fit(
            fit=fit,
            subset=subset,
            grouping="pooled_usage7_factual",
            animal="all",
            orientation="pooled",
            delta_t_hours=float(delta_t_hours),
            n_draws=int(n_draws),
            seed=int(seed),
            n_cells=int(df.shape[0]),
            ap_mm=float(df["ap_mm"].median()),
            ml_mm=float(df["ml_mm"].median()),
            step_mm=0.001,
            usage7_value=float(usage7_ref),
        )
    )
    derivative_zero = pd.DataFrame(
        meta_mod._derivative_rows_from_fit(
            fit=fit,
            subset=subset,
            grouping="pooled_usage7_zero",
            animal="all",
            orientation="pooled",
            delta_t_hours=float(delta_t_hours),
            n_draws=int(n_draws),
            seed=int(seed),
            n_cells=int(df.shape[0]),
            ap_mm=float(df["ap_mm"].median()),
            ml_mm=float(df["ml_mm"].median()),
            step_mm=0.001,
            usage7_value=float(usage7_baseline),
        )
    )
    for prefix in ("Ts", "Tc"):
        metric = f"{prefix}_hours"
        for axis in ("AP", "ML"):
            factual_row = derivative_factual.loc[
                (derivative_factual["metric"] == metric) & (derivative_factual["axis"] == axis)
            ].iloc[0]
            zero_row = derivative_zero.loc[
                (derivative_zero["metric"] == metric) & (derivative_zero["axis"] == axis)
            ].iloc[0]
            factual_est = float(factual_row["estimate"])
            zero_est = float(zero_row["estimate"])
            attenuation = np.nan if factual_est == 0.0 else float(1.0 - (abs(zero_est) / abs(factual_est)))
            rows.extend(
                [
                    {
                        "summary_type": "local_derivative",
                        "metric": metric,
                        "axis": axis,
                        "scenario": "factual",
                        "estimate": factual_est,
                        "ci_low": float(factual_row["ci_low"]),
                        "ci_high": float(factual_row["ci_high"]),
                        "n_bins": np.nan,
                        "weight_total": np.nan,
                        "reference_usage7": float(usage7_ref),
                    },
                    {
                        "summary_type": "local_derivative",
                        "metric": metric,
                        "axis": axis,
                        "scenario": "zero_baseline",
                        "estimate": zero_est,
                        "ci_low": float(zero_row["ci_low"]),
                        "ci_high": float(zero_row["ci_high"]),
                        "n_bins": np.nan,
                        "weight_total": np.nan,
                        "reference_usage7": float(usage7_ref),
                    },
                    {
                        "summary_type": "derivative_attenuation",
                        "metric": metric,
                        "axis": axis,
                        "scenario": "factual_vs_zero",
                        "estimate": attenuation,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_bins": np.nan,
                        "weight_total": np.nan,
                        "reference_usage7": float(usage7_ref),
                    },
                ]
            )
    return pd.DataFrame(rows)


def _symmetric_vlim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def _save_native_delta_figure(
    *,
    native_mod: Any,
    helper_mod: Any,
    context: dict[str, object],
    support_hull: Any,
    ts_delta_values: np.ndarray,
    tc_delta_values: np.ndarray,
    title_suffix: str,
    out_png: pathlib.Path,
) -> None:
    fig = plt.figure(figsize=(12.6, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    panels = (
        ("Tc delta", np.asarray(tc_delta_values, dtype=float), "Tc factual - baseline (hours)"),
        ("Ts delta", np.asarray(ts_delta_values, dtype=float), "Ts factual - baseline (hours)"),
    )
    for ax, (title, values, cbar_label) in zip(axes, panels, strict=True):
        ax.set_facecolor("white")
        vmin, vmax = _symmetric_vlim(values)
        native_mod._draw_native_scalar_surface_panel(
            fig=fig,
            ax=ax,
            helper_mod=helper_mod,
            context=context,
            values=values,
            support_hull=support_hull,
            panel_title=title,
            cbar_label=cbar_label,
            native_elev_deg=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
            native_azim_deg=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
            native_roll_deg=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
            cmap_name="coolwarm",
            vmin=vmin,
            vmax=vmax,
    )
    fig.suptitle(f"Layer 1 Usage_7 attribution: factual minus {title_suffix}", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
    plt.close(fig)


def _save_native_counterfactual_figure(
    *,
    native_mod: Any,
    helper_mod: Any,
    context: dict[str, object],
    support_hull: Any,
    ts_values: np.ndarray,
    tc_values: np.ndarray,
    ts_control_values: np.ndarray,
    tc_control_values: np.ndarray,
    title_suffix: str,
    out_png: pathlib.Path,
) -> None:
    fig = plt.figure(figsize=(12.6, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    panels = (
        (
            "Tc counterfactual",
            np.asarray(tc_values, dtype=float),
            np.asarray(tc_control_values, dtype=float),
            "Tc counterfactual (hours)",
        ),
        (
            "Ts counterfactual",
            np.asarray(ts_values, dtype=float),
            np.asarray(ts_control_values, dtype=float),
            "Ts counterfactual (hours)",
        ),
    )
    for ax, (title, values, control_values, cbar_label) in zip(axes, panels, strict=True):
        ax.set_facecolor("white")
        finite = np.asarray(control_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin == vmax:
                vmax = vmin + 1.0
        native_mod._draw_native_scalar_surface_panel(
            fig=fig,
            ax=ax,
            helper_mod=helper_mod,
            context=context,
            values=values,
            support_hull=support_hull,
            panel_title=title,
            cbar_label=cbar_label,
            native_elev_deg=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
            native_azim_deg=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
            native_roll_deg=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
            cmap_name="turbo",
            vmin=vmin,
            vmax=vmax,
        )
    fig.suptitle(f"Layer 1 Usage_7 counterfactual: {title_suffix}", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, facecolor="white", edgecolor="none")
    plt.close(fig)


def _build_line_delta_figure(
    curve_dfs: dict[int, pd.DataFrame],
    *,
    title: str,
    ap_support_bounds_um: tuple[float, float],
    ml_support_bounds_um: tuple[float, float],
    line_t_min: float,
    line_t_max: float,
) -> plt.Figure:
    fig, axes_arr = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    fig.patch.set_facecolor("white")
    axes = {"Tc": axes_arr[0], "Ts": axes_arr[1]}
    ap_low_um, ap_high_um = ap_support_bounds_um
    ml_low_um, ml_high_um = ml_support_bounds_um
    palette = sns.color_palette(n_colors=len(curve_dfs))
    for ax in axes.values():
        ax.set_facecolor("white")
        ax.grid(True, color="#d0d0d0", linewidth=0.8)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    for color, slice_k in zip(palette, sorted(curve_dfs), strict=True):
        curve_df = curve_dfs[slice_k]
        ap_um = curve_df["AP_um"].to_numpy(dtype=float)
        ml_um = curve_df["ML_um"].to_numpy(dtype=float)
        t_s = curve_df["t_s"].to_numpy(dtype=float)
        x = ap_um / 1000.0
        finite = np.isfinite(ap_um) & np.isfinite(ml_um) & np.isfinite(t_s)
        finite &= ap_um >= ap_low_um
        finite &= ap_um <= ap_high_um
        finite &= ml_um >= ml_low_um
        finite &= ml_um <= ml_high_um
        finite &= t_s > float(line_t_min)
        finite &= t_s < float(line_t_max)
        for prefix, ax in axes.items():
            mean = curve_df[f"{prefix}_delta_mean"].to_numpy(dtype=float)
            low = curve_df[f"{prefix}_delta_ci_low"].to_numpy(dtype=float)
            high = curve_df[f"{prefix}_delta_ci_high"].to_numpy(dtype=float)
            mask = finite & np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
            mean_plot = np.where(mask, mean, np.nan)
            low_plot = np.where(mask, low, np.nan)
            high_plot = np.where(mask, high, np.nan)
            ax.fill_between(x, low_plot, high_plot, where=np.isfinite(low_plot) & np.isfinite(high_plot), color=color, alpha=0.10, linewidth=0.0)
            ax.plot(x, mean_plot, color=color, linewidth=1.8, label=f"k={int(slice_k)}")
    axes["Tc"].set_title("Tc attribution", fontsize=11)
    axes["Ts"].set_title("Ts attribution", fontsize=11)
    axes["Tc"].set_ylabel("Factual - baseline (hours)")
    axes["Tc"].set_xlabel("AP (mm)")
    axes["Ts"].set_xlabel("AP (mm)")
    axes["Tc"].legend(title="Sagittal slice", frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def _build_line_counterfactual_figure(
    curve_dfs: dict[int, pd.DataFrame],
    *,
    title: str,
    ap_support_bounds_um: tuple[float, float],
    ml_support_bounds_um: tuple[float, float],
    line_t_min: float,
    line_t_max: float,
    y_limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    fig, axes_arr = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    fig.patch.set_facecolor("white")
    axes = {"Tc": axes_arr[0], "Ts": axes_arr[1]}
    ap_low_um, ap_high_um = ap_support_bounds_um
    ml_low_um, ml_high_um = ml_support_bounds_um
    palette = sns.color_palette(n_colors=len(curve_dfs))
    for ax in axes.values():
        ax.set_facecolor("white")
        ax.grid(True, color="#d0d0d0", linewidth=0.8)
    for color, slice_k in zip(palette, sorted(curve_dfs), strict=True):
        curve_df = curve_dfs[slice_k]
        ap_um = curve_df["AP_um"].to_numpy(dtype=float)
        ml_um = curve_df["ML_um"].to_numpy(dtype=float)
        t_s = curve_df["t_s"].to_numpy(dtype=float)
        x = ap_um / 1000.0
        finite = np.isfinite(ap_um) & np.isfinite(ml_um) & np.isfinite(t_s)
        finite &= ap_um >= ap_low_um
        finite &= ap_um <= ap_high_um
        finite &= ml_um >= ml_low_um
        finite &= ml_um <= ml_high_um
        finite &= t_s > float(line_t_min)
        finite &= t_s < float(line_t_max)
        for prefix, ax in axes.items():
            mean = curve_df[f"{prefix}_zero_mean"].to_numpy(dtype=float)
            low = curve_df[f"{prefix}_zero_ci_low"].to_numpy(dtype=float)
            high = curve_df[f"{prefix}_zero_ci_high"].to_numpy(dtype=float)
            mask = finite & np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
            mean_plot = np.where(mask, mean, np.nan)
            low_plot = np.where(mask, low, np.nan)
            high_plot = np.where(mask, high, np.nan)
            ax.fill_between(
                x,
                low_plot,
                high_plot,
                where=np.isfinite(low_plot) & np.isfinite(high_plot),
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
            ax.plot(x, mean_plot, color=color, linewidth=1.8, label=f"k={int(slice_k)}")
    axes["Tc"].set_title("Tc counterfactual", fontsize=11)
    axes["Ts"].set_title("Ts counterfactual", fontsize=11)
    axes["Tc"].set_ylabel("Counterfactual (hours)")
    axes["Tc"].set_xlabel("AP (mm)")
    axes["Ts"].set_xlabel("AP (mm)")
    if y_limits is not None:
        for prefix, ax in axes.items():
            if prefix in y_limits:
                ax.set_ylim(*y_limits[prefix])
    axes["Tc"].legend(title="Sagittal slice", frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def _collect_line_y_limits(
    curve_dfs: dict[int, pd.DataFrame],
    *,
    ap_support_bounds_um: tuple[float, float],
    ml_support_bounds_um: tuple[float, float],
    line_t_min: float,
    line_t_max: float,
    scenario: str,
) -> dict[str, tuple[float, float]]:
    ap_low_um, ap_high_um = ap_support_bounds_um
    ml_low_um, ml_high_um = ml_support_bounds_um
    out: dict[str, tuple[float, float]] = {}
    for prefix in ("Tc", "Ts"):
        low_parts: list[np.ndarray] = []
        high_parts: list[np.ndarray] = []
        for curve_df in curve_dfs.values():
            ap_um = curve_df["AP_um"].to_numpy(dtype=float)
            ml_um = curve_df["ML_um"].to_numpy(dtype=float)
            t_s = curve_df["t_s"].to_numpy(dtype=float)
            finite = np.isfinite(ap_um) & np.isfinite(ml_um) & np.isfinite(t_s)
            finite &= ap_um >= ap_low_um
            finite &= ap_um <= ap_high_um
            finite &= ml_um >= ml_low_um
            finite &= ml_um <= ml_high_um
            finite &= t_s > float(line_t_min)
            finite &= t_s < float(line_t_max)
            low = curve_df[f"{prefix}_{scenario}_ci_low"].to_numpy(dtype=float)
            high = curve_df[f"{prefix}_{scenario}_ci_high"].to_numpy(dtype=float)
            mask = finite & np.isfinite(low) & np.isfinite(high)
            if np.any(mask):
                low_parts.append(low[mask])
                high_parts.append(high[mask])
        if not low_parts:
            continue
        lower = float(np.nanmin(np.concatenate(low_parts)))
        upper = float(np.nanmax(np.concatenate(high_parts)))
        if lower == upper:
            upper = lower + 1.0
        out[prefix] = (lower, upper)
    return out


def main() -> None:
    args = _parse_args()
    meta_mod = _load_module("fit_apml_multinomial_animal_meta", MULTINOMIAL_PATH)
    native_mod = _load_module("plot_ts_tc_native_with_sagittal_line", NATIVE_PATH)
    line_mod = _load_module("plot_ts_tc_vs_sagittal_t_pooled", LINE_PATH)
    helper_mod = meta_mod._load_plot_apml_module()
    clusters = {c.strip() for c in str(args.clusters).split(",") if c.strip()}
    exclude_animals = {a.strip() for a in str(args.exclude_animals).split(",") if a.strip()}
    usage7_gam_bundle = _load_usage7_gam_bundle(gam_dir=args.usage7_gam_dir, exclude_animals=exclude_animals)
    slice_ks = _parse_slice_ks(str(args.slice_ks))
    outdir = args.outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    adata = helper_mod._load_adata_with_external_obsm(
        h5ad_path=args.h5ad.expanduser(),
        obsm_h5ad_path=args.obsm_h5ad.expanduser(),
    )
    df = meta_mod._build_model_df(
        helper_mod=helper_mod,
        adata=adata,
        clusters=clusters,
        manual_layer="1",
        eomes_gt=None,
        eomes_lt=None,
        label=DEFAULT_SUBSET,
        exclude_animals=exclude_animals,
    )
    if DEFAULT_USAGE7_COL not in df.columns:
        raise KeyError(f"Expected {DEFAULT_USAGE7_COL!r} in model dataframe.")
    df = df.loc[np.isfinite(df[DEFAULT_USAGE7_COL].to_numpy(dtype=float))].copy()
    if df.empty:
        raise ValueError("No finite Usage_7 rows remain after filtering.")
    fit = meta_mod._fit_multinomial_logit_2d(df, usage7_col=DEFAULT_USAGE7_COL)
    bin_summary = _build_usage7_bin_summary(helper_mod=helper_mod, df=df, bin_width_um=float(args.bin_width_um))
    stem = _output_stem(usage7_baseline_quantile=float(args.usage7_baseline_quantile))
    bin_table = pd.DataFrame(bin_summary["table"]).copy()
    bin_table["gam_usage7"] = _load_or_predict_usage7_gam(
        cache_path=outdir / f"{stem}__gam_usage7_bins.npz",
        bundle=usage7_gam_bundle,
        ap_um=bin_table["ap_center_um"].to_numpy(dtype=float),
        ml_um=bin_table["ml_center_um"].to_numpy(dtype=float),
    )
    bin_summary["table"] = bin_table
    usage7_baseline = _baseline_usage7_quantile_value(
        bin_summary=bin_summary,
        usage7_col="gam_usage7",
        quantile=float(args.usage7_baseline_quantile),
    )
    baseline_pct = int(round(float(args.usage7_baseline_quantile) * 100.0))
    baseline_title_suffix = f"q{baseline_pct:02d} baseline"
    usage7_interp = _build_usage7_interpolator(
        ap_um=bin_table.loc[bin_table["n_total"] > 0, "ap_center_um"].to_numpy(dtype=float),
        ml_um=bin_table.loc[bin_table["n_total"] > 0, "ml_center_um"].to_numpy(dtype=float),
        usage7=bin_table.loc[bin_table["n_total"] > 0, "gam_usage7"].to_numpy(dtype=float),
    )
    summary_df = _build_summary_rows(
        meta_mod=meta_mod,
        fit=fit,
        df=df,
        subset=DEFAULT_SUBSET,
        bin_summary=bin_summary,
        delta_t_hours=float(args.delta_t_hours),
        n_draws=int(args.n_draws),
        seed=int(args.seed),
        usage7_baseline=float(usage7_baseline),
        usage7_col="gam_usage7",
    )
    summary_df.to_csv(outdir / f"{stem}__summary.csv", index=False)
    pd.DataFrame(bin_summary["table"]).to_csv(outdir / f"{stem}__bin_usage7.csv", index=False)
    context = build_apml_native_surface_projection_context(
        outdir=args.refextract_outdir.expanduser(),
        slice_i_min=DEFAULT_REFEXTRACT_SLICE_I_MIN,
        slice_i_max=DEFAULT_REFEXTRACT_SLICE_I_MAX,
        n_t=DEFAULT_REFEXTRACT_N_T,
        ref_t=DEFAULT_REFEXTRACT_REF_T,
        band_frac=DEFAULT_REFEXTRACT_BAND_FRAC,
        res_ijk_um=DEFAULT_REFEXTRACT_RES_IJK_UM,
        elev_deg=DEFAULT_NATIVE_CAMERA_ELEV_DEG,
        azim_deg=DEFAULT_NATIVE_CAMERA_AZIM_DEG,
        roll_deg=DEFAULT_NATIVE_CAMERA_ROLL_DEG,
    )
    ap_support_bounds_um, ml_support_bounds_um, support_hull = native_mod._build_subset_support_hull(
        ap_um=df["ap_mm"].to_numpy(dtype=float) * 1000.0,
        ml_um=df["ml_mm"].to_numpy(dtype=float) * 1000.0,
    )
    surface_ap_um = np.asarray(context["ap_um_flat"], dtype=float)
    surface_ml_um = np.asarray(context["ml_um_flat"], dtype=float)
    display_mask = np.asarray(context["support_flat"], dtype=bool)
    display_mask &= native_mod._points_inside_support_hull(
        ap_um=surface_ap_um,
        ml_um=surface_ml_um,
        support_hull=support_hull,
    )
    surface_usage7 = usage7_interp(ap_um=surface_ap_um, ml_um=surface_ml_um)
    surface_usage7[~display_mask] = np.nan
    surface_eval = _evaluate_factual_zero_delta(
        meta_mod=meta_mod,
        fit=fit,
        subset=DEFAULT_SUBSET,
        delta_t_hours=float(args.delta_t_hours),
        n_draws=int(args.n_draws),
        seed=int(args.seed),
        ap_um=surface_ap_um,
        ml_um=surface_ml_um,
        usage7_mean=surface_usage7,
        usage7_baseline=float(usage7_baseline),
    )
    ts_delta = surface_eval["Ts_delta_mean"].to_numpy(dtype=float)
    tc_delta = surface_eval["Tc_delta_mean"].to_numpy(dtype=float)
    ts_fact = surface_eval["Ts_factual_mean"].to_numpy(dtype=float)
    tc_fact = surface_eval["Tc_factual_mean"].to_numpy(dtype=float)
    ts_zero = surface_eval["Ts_zero_mean"].to_numpy(dtype=float)
    tc_zero = surface_eval["Tc_zero_mean"].to_numpy(dtype=float)
    support_mask = np.asarray(context["support_flat"], dtype=bool)
    ts_delta[~support_mask] = np.nan
    tc_delta[~support_mask] = np.nan
    ts_fact[~support_mask] = np.nan
    tc_fact[~support_mask] = np.nan
    ts_zero[~support_mask] = np.nan
    tc_zero[~support_mask] = np.nan
    _save_native_delta_figure(
        native_mod=native_mod,
        helper_mod=helper_mod,
        context=context,
        support_hull=support_hull,
        ts_delta_values=ts_delta,
        tc_delta_values=tc_delta,
        title_suffix=baseline_title_suffix,
        out_png=outdir / f"{stem}__native_delta.png",
    )
    _save_native_counterfactual_figure(
        native_mod=native_mod,
        helper_mod=helper_mod,
        context=context,
        support_hull=support_hull,
        ts_values=ts_zero,
        tc_values=tc_zero,
        ts_control_values=ts_fact,
        tc_control_values=tc_fact,
        title_suffix=baseline_title_suffix,
        out_png=outdir / f"{stem}__native_counterfactual.png",
    )

    curve_dfs: dict[int, pd.DataFrame] = {}
    curve_rows: list[pd.DataFrame] = []
    for slice_k in slice_ks:
        query_df = line_mod._build_sagittal_query_df(
            lut_outdir=args.refextract_outdir.expanduser(),
            slice_k=int(slice_k),
            native_context=context,
            res_ijk_um=DEFAULT_REFEXTRACT_RES_IJK_UM,
        )
        usage7_line = usage7_interp(
            ap_um=query_df["AP_um"].to_numpy(dtype=float),
            ml_um=query_df["ML_um"].to_numpy(dtype=float),
        )
        curve_df = pd.concat(
            [
                query_df.reset_index(drop=True),
                _evaluate_factual_zero_delta(
                    meta_mod=meta_mod,
                    fit=fit,
                    subset=DEFAULT_SUBSET,
                    delta_t_hours=float(args.delta_t_hours),
                    n_draws=int(args.n_draws),
                    seed=int(args.seed),
                    ap_um=query_df["AP_um"].to_numpy(dtype=float),
                    ml_um=query_df["ML_um"].to_numpy(dtype=float),
                    usage7_mean=usage7_line,
                    usage7_baseline=float(usage7_baseline),
                ).drop(columns=["subset", "AP_um", "ML_um", "usage7_mean"]),
            ],
            axis=1,
        )
        curve_df["slice_k"] = int(slice_k)
        curve_df["usage7_mean"] = usage7_line
        _counts = _lookup_binned_usage7(
            ap_um=query_df["AP_um"].to_numpy(dtype=float),
            ml_um=query_df["ML_um"].to_numpy(dtype=float),
            bin_summary=bin_summary,
        )[1]
        curve_df["bin_n_total"] = _counts
        curve_dfs[int(slice_k)] = curve_df
        curve_rows.append(curve_df)
    pd.concat(curve_rows, ignore_index=True).to_csv(outdir / f"{stem}__line_delta.csv", index=False)
    factual_line_y_limits = _collect_line_y_limits(
        curve_dfs,
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        line_t_min=float(args.line_t_min),
        line_t_max=float(args.line_t_max),
        scenario="factual",
    )
    line_fig = _build_line_delta_figure(
        curve_dfs,
        title=f"Layer 1 Usage_7 attribution along sagittal slices: factual minus {baseline_title_suffix}",
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        line_t_min=float(args.line_t_min),
        line_t_max=float(args.line_t_max),
    )
    line_fig.savefig(
        outdir / f"{stem}__line_delta.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(line_fig)
    line_counterfactual_fig = _build_line_counterfactual_figure(
        curve_dfs,
        title=f"Layer 1 Usage_7 counterfactual along sagittal slices: {baseline_title_suffix}",
        ap_support_bounds_um=ap_support_bounds_um,
        ml_support_bounds_um=ml_support_bounds_um,
        line_t_min=float(args.line_t_min),
        line_t_max=float(args.line_t_max),
        y_limits=factual_line_y_limits,
    )
    line_counterfactual_fig.savefig(
        outdir / f"{stem}__line_counterfactual.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(line_counterfactual_fig)

    print(f"wrote {outdir / f'{stem}__summary.csv'}")
    print(f"wrote {outdir / f'{stem}__bin_usage7.csv'}")
    print(f"wrote {outdir / f'{stem}__native_delta.png'}")
    print(f"wrote {outdir / f'{stem}__native_counterfactual.png'}")
    print(f"wrote {outdir / f'{stem}__line_delta.csv'}")
    print(f"wrote {outdir / f'{stem}__line_delta.png'}")
    print(f"wrote {outdir / f'{stem}__line_counterfactual.png'}")


if __name__ == "__main__":
    main()
