from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


class RPredictor:
    def __init__(self) -> None:
        self.base = importr("base")
        self.r_predict = ro.r["predict"]
        self.r_names = ro.r["names"]
        self.r_levels = ro.r["levels"]
        self.r_colnames = ro.r["colnames"]

    def read_fit(self, fit_path: Path):
        return self.base.readRDS(str(fit_path))

    def coefficient_names(self, fit) -> list[str]:
        coef = fit.rx2("coefficients")
        return [str(x) for x in self.r_names(coef)]

    def mean_log_sf(self, fit) -> float:
        names = [str(x) for x in self.r_names(fit)]
        if "mean_log_sf" not in names:
            raise ValueError("Fit is missing mean_log_sf (required for log_sf_c). Refit with updated scripts/gam/inm_gam.R.")
        value = float(fit.rx2("mean_log_sf")[0])
        if not np.isfinite(value):
            raise ValueError("mean_log_sf is not finite in the fitted object.")
        return value

    def xlevels(self, fit) -> dict[str, list[str]]:
        fit_names = [str(x) for x in self.r_names(fit)]
        if "xlevels" not in fit_names:
            return {}
        xlevels = fit.rx2("xlevels")
        names = [str(x) for x in self.r_names(xlevels)]
        out: dict[str, list[str]] = {}
        for name in names:
            levels = xlevels.rx2(name)
            out[str(name)] = [str(x) for x in levels]
        return out

    def factor_levels(self, fit, name: str) -> list[str]:
        xlevels = self.xlevels(fit)
        if name in xlevels:
            return xlevels[name]
        fit_names = [str(x) for x in self.r_names(fit)]
        if "var.summary" in fit_names:
            var_summary = fit.rx2("var.summary")
            vs_names = [str(x) for x in self.r_names(var_summary)]
            if name in vs_names:
                value = var_summary.rx2(name)
                if bool(self.base.is_factor(value)[0]):
                    levels = self.r_levels(value)
                    return [str(x) for x in levels]
        return []

    def batch_levels(self, fit) -> list[str]:
        return self.factor_levels(fit, "batch")

    def animal_levels(self, fit) -> list[str]:
        return self.factor_levels(fit, "animal")

    def predict_response(self, fit, newdata: pd.DataFrame) -> np.ndarray:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_newdata = ro.conversion.py2rpy(newdata)
        pred = self.r_predict(fit, newdata=r_newdata, type="response")
        return np.asarray(pred, dtype=float)

    def predict_link(self, fit, newdata: pd.DataFrame) -> np.ndarray:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_newdata = ro.conversion.py2rpy(newdata)
        pred = self.r_predict(fit, newdata=r_newdata, type="link")
        return np.asarray(pred, dtype=float)

    def predict_terms_se(self, fit, newdata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_newdata = ro.conversion.py2rpy(newdata)
        pred = self.r_predict(fit, newdata=r_newdata, type="terms", **{"se.fit": True})
        fit_mat = pred.rx2("fit")
        se_mat = pred.rx2("se.fit")
        colnames = [str(x) for x in self.r_colnames(fit_mat)]
        return np.asarray(fit_mat, dtype=float), np.asarray(se_mat, dtype=float), colnames


def shrink_soft(effect: np.ndarray, se: np.ndarray) -> np.ndarray:
    effect = np.asarray(effect, dtype=float)
    se = np.asarray(se, dtype=float)
    return effect / (1.0 + se)


def shrink_hard(effect: np.ndarray, se: np.ndarray, *, z: float) -> np.ndarray:
    effect = np.asarray(effect, dtype=float)
    se = np.asarray(se, dtype=float)
    out = effect.copy()
    out[np.abs(effect) <= (float(z) * se)] = 0.0
    return out


def is_smooth_term(term_name: str) -> bool:
    return term_name.startswith(("s(", "ti(", "te(", "t2("))


def predict_response_shrunk(
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    *,
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    return np.exp(predict_link_shrunk(predictor, fit, newdata, shrink=shrink, hard_z=hard_z))


def predict_link_shrunk(
    predictor: RPredictor,
    fit,
    newdata: pd.DataFrame,
    *,
    shrink: str,
    hard_z: float,
) -> np.ndarray:
    eta = predictor.predict_link(fit, newdata)
    terms, se_terms, term_names = predictor.predict_terms_se(fit, newdata)
    smooth_idx = [i for i, name in enumerate(term_names) if is_smooth_term(name)]
    if not smooth_idx:
        return eta
    smooth = terms[:, smooth_idx]
    smooth_se = se_terms[:, smooth_idx]
    if shrink == "none":
        smooth_shrunk = smooth
    elif shrink == "hard":
        smooth_shrunk = shrink_hard(smooth, smooth_se, z=hard_z)
    elif shrink == "soft":
        smooth_shrunk = shrink_soft(smooth, smooth_se)
    else:
        raise ValueError(f"Unknown shrink mode: {shrink}")
    return eta - np.sum(smooth, axis=1) + np.sum(smooth_shrunk, axis=1)


def make_newdata_base(
    size: int,
    *,
    r_um: np.ndarray | float,
    theta: np.ndarray | float,
    ap_um: np.ndarray | float,
    ml_um: np.ndarray | float,
    mean_log_sf: float | None,
    include_pos: bool,
    include_batch: bool,
    batch_ref: str | None,
    batch_levels: list[str],
    include_animal: bool,
    animal_ref: str | None,
    animal_levels: list[str],
    include_log_sf_c: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "r_um": np.full(size, float(r_um)) if np.isscalar(r_um) else np.asarray(r_um, dtype=float),
            "theta": np.full(size, float(theta)) if np.isscalar(theta) else np.asarray(theta, dtype=float),
            "AP_um": np.full(size, float(ap_um)) if np.isscalar(ap_um) else np.asarray(ap_um, dtype=float),
            "ML_um": np.full(size, float(ml_um)) if np.isscalar(ml_um) else np.asarray(ml_um, dtype=float),
            "sf": np.full(size, 1.0, dtype=float),
        }
    )
    if include_log_sf_c:
        if mean_log_sf is None or not np.isfinite(mean_log_sf):
            raise ValueError("Model uses log_sf_c but mean_log_sf was not provided.")
        frame["log_sf_c"] = np.log(frame["sf"].to_numpy(dtype=float)) - float(mean_log_sf)
    if include_pos:
        frame["brdu_pos"] = np.zeros(size, dtype=int)
        frame["edu_pos"] = np.zeros(size, dtype=int)
    if include_batch:
        if batch_ref is None:
            raise ValueError("Model contains batch terms, but no batch reference was provided.")
        if batch_levels:
            frame["batch"] = pd.Categorical([batch_ref] * size, categories=batch_levels)
        else:
            frame["batch"] = pd.Categorical([batch_ref] * size)
    if include_animal:
        if animal_ref is None:
            raise ValueError("Model contains animal terms, but no animal reference was provided.")
        if animal_levels:
            frame["animal"] = pd.Categorical([animal_ref] * size, categories=animal_levels)
        else:
            frame["animal"] = pd.Categorical([animal_ref] * size)
    return frame


def compact_term_name(term_name: str) -> str:
    return "".join(str(term_name).split())


def select_ti_apml_r(term_names: list[str]) -> list[int]:
    idx: list[int] = []
    for i, name in enumerate(term_names):
        c = compact_term_name(name)
        if not c.startswith("ti("):
            continue
        if ("AP_um" in c) and ("ML_um" in c) and ("r_um" in c) and ("theta" not in c):
            idx.append(i)
    return idx


def select_s_apml(term_names: list[str]) -> list[int]:
    idx: list[int] = []
    for i, name in enumerate(term_names):
        c = compact_term_name(name)
        if not c.startswith("s("):
            continue
        if ("AP_um" in c) and ("ML_um" in c) and ("r_um" not in c) and ("theta" not in c):
            idx.append(i)
    return idx
