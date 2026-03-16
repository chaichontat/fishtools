from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fishtools.gam.mgcv_predict import RPredictor
from fishtools.gam.mgcv_predict import make_newdata_base
from fishtools.gam.mgcv_predict import select_s_apml
from fishtools.gam.mgcv_predict import select_ti_apml_r
from fishtools.gam.mgcv_predict import shrink_hard
from fishtools.gam.mgcv_predict import shrink_soft


@dataclass(frozen=True)
class GAMPredictorConfig:
    mean_log_sf: float | None
    include_pos: bool
    include_batch: bool
    batch_ref: str | None
    batch_levels: list[str]
    include_animal: bool
    animal_ref: str | None
    animal_levels: list[str]
    include_log_sf_c: bool


def predict_term_effect_or_z_shrunk(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    idx: list[int],
    kind: str,
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    terms, se_terms, _term_names = predictor.predict_terms_se(fit, newdata)
    if not idx:
        return np.zeros((terms.shape[0],), dtype=float)
    eff = terms[:, idx]
    se = se_terms[:, idx]
    if shrink == "none":
        eff_shrunk = eff
    elif shrink == "hard":
        eff_shrunk = shrink_hard(eff, se, z=float(hard_z))
    elif shrink == "soft":
        eff_shrunk = shrink_soft(eff, se)
    else:
        raise ValueError(f"Unknown shrink mode: {shrink}")
    effect = np.sum(eff_shrunk, axis=1)
    if kind == "effect":
        return effect
    if kind == "z":
        se_sum = np.sqrt(np.sum(np.square(se), axis=1))
        if not np.all(np.isfinite(se_sum)):
            raise ValueError("Non-finite SE encountered while computing z-surface.")
        if np.any(se_sum <= 0):
            raise ValueError("Non-positive SE encountered while computing z-surface.")
        return effect / se_sum
    raise ValueError(f"Unknown surface kind: {kind}")


def predict_term_effect_shrunk(
    *,
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    idx: list[int],
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    return predict_term_effect_or_z_shrunk(
        predictor=predictor,
        fit=fit,
        newdata=newdata,
        idx=idx,
        kind="effect",
        shrink=shrink,
        hard_z=hard_z,
    )


def make_newdata_for_fit(
    *,
    predictor: RPredictor,
    fit,
    n: int,
    r_um: float | np.ndarray,
    theta: float | np.ndarray,
    ap_um: float | np.ndarray,
    ml_um: float | np.ndarray,
    config: GAMPredictorConfig,
) -> pd.DataFrame:
    nd = make_newdata_base(
        n,
        r_um=r_um,
        theta=theta,
        ap_um=ap_um,
        ml_um=ml_um,
        mean_log_sf=config.mean_log_sf,
        include_pos=bool(config.include_pos),
        include_batch=bool(config.include_batch),
        batch_ref=config.batch_ref,
        batch_levels=config.batch_levels,
        include_animal=bool(config.include_animal),
        animal_ref=config.animal_ref,
        animal_levels=config.animal_levels,
        include_log_sf_c=bool(config.include_log_sf_c),
    )
    out = nd.copy()
    ab_levels = predictor.factor_levels(fit, "ab")
    if len(ab_levels) > 0 and "ab" not in out.columns:
        ab_ref: str | None = None
        if ("animal" in out.columns) and ("batch" in out.columns):
            animal_value = str(out["animal"].iloc[0])
            batch_value = str(out["batch"].iloc[0])
            candidate = f"{animal_value}.{batch_value}"
            if candidate in ab_levels:
                ab_ref = candidate
        if ab_ref is None and (config.animal_ref is not None) and (config.batch_ref is not None):
            candidate = f"{config.animal_ref}.{config.batch_ref}"
            if candidate in ab_levels:
                ab_ref = candidate
        if ab_ref is None and config.animal_ref is not None:
            for level in ab_levels:
                if str(level).startswith(f"{config.animal_ref}."):
                    ab_ref = str(level)
                    break
        if ab_ref is None and config.batch_ref is not None:
            for level in ab_levels:
                if str(level).endswith(f".{config.batch_ref}"):
                    ab_ref = str(level)
                    break
        if ab_ref is None:
            ab_ref = str(ab_levels[0])
        out["ab"] = pd.Categorical([ab_ref] * out.shape[0], categories=ab_levels)

    fit_xlevels = predictor.xlevels(fit)
    for col, levels in fit_xlevels.items():
        if col in out.columns:
            continue
        if len(levels) <= 0:
            continue
        out[col] = pd.Categorical([levels[0]] * out.shape[0], categories=levels)
    return out


def predict_apml_surface_values_on_coronal_surface(
    *,
    predictor: RPredictor,
    fit,
    ap_um_flat: np.ndarray,
    ml_um_flat: np.ndarray,
    support_flat: np.ndarray,
    predict_mask: np.ndarray | None,
    r0: float,
    theta0: float,
    config: GAMPredictorConfig,
    apml_surface: str,
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    ap = np.asarray(ap_um_flat, dtype=np.float64).reshape(-1)
    ml = np.asarray(ml_um_flat, dtype=np.float64).reshape(-1)
    support = np.asarray(support_flat, dtype=bool).reshape(-1)
    if ap.shape != ml.shape or ap.shape != support.shape:
        raise ValueError("ap/ml/support shape mismatch for coronal surface export")
    if not np.any(support):
        raise ValueError("Empty coronal surface support mask")
    if predict_mask is None:
        pred = support
    else:
        pred = np.asarray(predict_mask, dtype=bool).reshape(-1)
        if pred.shape != support.shape:
            raise ValueError("predict_mask shape mismatch for coronal surface export")
        if np.any(pred & ~support):
            raise ValueError("predict_mask must be a subset of support mask")
        if not np.any(pred):
            raise ValueError("Empty coronal surface predict mask")

    ap_s = ap[pred]
    ml_s = ml[pred]
    nd = make_newdata_for_fit(
        predictor=predictor,
        fit=fit,
        n=int(ap_s.size),
        r_um=float(r0),
        theta=float(theta0),
        ap_um=ap_s,
        ml_um=ml_s,
        config=config,
    )
    if config.include_log_sf_c and config.mean_log_sf is not None:
        nd["sf"] = float(np.exp(float(config.mean_log_sf)))
        nd["log_sf_c"] = 0.0

    if str(apml_surface) == "mu":
        vals_s = predictor.predict_response(fit, nd).astype(np.float64, copy=False)
    elif str(apml_surface) == "eta":
        vals_s = predictor.predict_link(fit, nd).astype(np.float64, copy=False)
    else:
        terms, _se_terms, term_names = predictor.predict_terms_se(fit, nd)
        idx_s = select_s_apml(term_names)
        idx_ti = select_ti_apml_r(term_names)
        if not idx_s:
            raise ValueError("No s(AP_um,ML_um) term found for coronal surface AP/ML plotting")
        idx = idx_s + idx_ti
        vals_s = predict_term_effect_or_z_shrunk(
            predictor=predictor,
            fit=fit,
            newdata=nd,
            idx=idx,
            kind=str(apml_surface),
            shrink=str(shrink),
            hard_z=float(hard_z),
        ).astype(np.float64, copy=False)

    out = np.full(ap.shape, np.nan, dtype=np.float64)
    out[pred] = vals_s
    return out
