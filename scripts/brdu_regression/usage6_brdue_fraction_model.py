#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import shlex
import sys
import time

import numpy as np
import pandas as pd


def _load_tricycle_ref(path: pathlib.Path) -> pd.DataFrame:
    trc = pd.read_csv(path)
    required = {"symbol", "pc1.rot", "pc2.rot"}
    missing = required - set(trc.columns)
    if missing:
        raise ValueError(f"tricycle ref CSV missing columns: {sorted(missing)}")
    trc = trc.loc[:, ["symbol", "pc1.rot", "pc2.rot"]].copy()
    trc["symbol"] = trc["symbol"].astype(str)
    return trc


def _animal_from_dataset(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r} (expected contains 'JaxA#').")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _normalize_animal_label(animal: str) -> str:
    m = re.fullmatch(r"\s*(jaxa\d+)\s*", str(animal), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Invalid animal label {animal!r}; expected values like 'JaxA2'.")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _stable_hash_u64(dataset: np.ndarray, obs_name: np.ndarray) -> np.ndarray:
    ds = dataset.astype(str)
    on = obs_name.astype(str)
    s = pd.Series(
        [f"{d}|{o}|{i}" for i, (d, o) in enumerate(zip(ds.tolist(), on.tolist(), strict=True))],
        copy=False,
    )
    return pd.util.hash_pandas_object(s, index=False).to_numpy(np.uint64, copy=False)


def _load_threshold_map_from_uns(
    *,
    uns: dict[str, object],
    uns_key: str = "brdu_edu_thresholds_by_dataset_roi_ccf_adjusted",
) -> dict[tuple[str, str, str], tuple[float, float]]:
    raw = uns.get(str(uns_key))
    if raw is None:
        raise KeyError(f"Missing adata.uns[{uns_key!r}] with per-unit thresholds.")
    if not isinstance(raw, dict):
        raise ValueError(f"adata.uns[{uns_key!r}] must be a nested dict.")

    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for dataset, roi_map in raw.items():
        if not isinstance(roi_map, dict):
            raise ValueError(f"adata.uns[{uns_key!r}][{dataset!r}] must be a dict.")
        for roi, ccf_map in roi_map.items():
            if not isinstance(ccf_map, dict):
                raise ValueError(f"adata.uns[{uns_key!r}][{dataset!r}][{roi!r}] must be a dict.")
            for ccf, thr in ccf_map.items():
                if not isinstance(thr, dict):
                    raise ValueError(
                        f"adata.uns[{uns_key!r}][{dataset!r}][{roi!r}][{ccf!r}] must be a dict."
                    )
                brdu_thr = thr.get("log_brdu_mean")
                edu_thr = thr.get("log_edu_mean")
                if brdu_thr is None or edu_thr is None:
                    raise ValueError(
                        f"Missing log_brdu_mean/log_edu_mean for "
                        f"adata.uns[{uns_key!r}][{dataset!r}][{roi!r}][{ccf!r}]"
                    )
                b = float(brdu_thr)
                e = float(edu_thr)
                if not (math.isfinite(b) and math.isfinite(e)):
                    raise ValueError(
                        f"Non-finite threshold values for "
                        f"adata.uns[{uns_key!r}][{dataset!r}][{roi!r}][{ccf!r}]"
                    )
                key = (str(dataset), str(roi), str(ccf))
                if key in out:
                    raise ValueError(f"Duplicate threshold key in uns map: {key}")
                out[key] = (b, e)
    return out


def _load_threshold_delta_map_from_csv(
    *,
    path: pathlib.Path,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    required = {"dataset", "roi", "ccf_adjusted", "delta_log_brdu", "delta_log_edu"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    dataset = df["dataset"].astype(str)
    roi = df["roi"].astype(str)
    ccf = df["ccf_adjusted"].astype(str)
    d_b = pd.to_numeric(df["delta_log_brdu"], errors="raise").to_numpy(dtype=np.float64, copy=False)
    d_e = pd.to_numeric(df["delta_log_edu"], errors="raise").to_numpy(dtype=np.float64, copy=False)
    if np.any(~np.isfinite(d_b)) or np.any(~np.isfinite(d_e)):
        raise ValueError(f"{path} contains non-finite delta_log_brdu/delta_log_edu values.")

    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for i, key in enumerate(zip(dataset.tolist(), roi.tolist(), ccf.tolist(), strict=True)):
        if key in out:
            raise ValueError(f"{path} contains duplicate threshold-delta key: {key}")
        out[key] = (float(d_b[i]), float(d_e[i]))
    return out


def _soft_logistic_probabilities(
    x: np.ndarray,
    threshold: np.ndarray,
    *,
    span: float,
    p_hi: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    thr_arr = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if x_arr.shape != thr_arr.shape:
        raise ValueError("x and threshold must have the same shape.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(thr_arr)):
        raise ValueError("x and threshold must be finite.")
    if not (0.5 < float(p_hi) < 1.0):
        raise ValueError("p_hi must be in (0.5, 1.0).")
    if not (float(span) > 0.0):
        raise ValueError("span must be > 0.")
    from scipy.special import expit

    soft_k = math.log(float(p_hi) / (1.0 - float(p_hi))) / float(span)
    return expit(soft_k * (x_arr - thr_arr)).astype(np.float64, copy=False)


def _kish_effective_sample_size(weight_sum: np.ndarray, weight_sq_sum: np.ndarray) -> np.ndarray:
    weight_sum_arr = np.asarray(weight_sum, dtype=np.float64)
    weight_sq_sum_arr = np.asarray(weight_sq_sum, dtype=np.float64)
    if weight_sum_arr.shape != weight_sq_sum_arr.shape:
        raise ValueError("weight_sum and weight_sq_sum must have the same shape.")
    if np.any(~np.isfinite(weight_sum_arr)) or np.any(~np.isfinite(weight_sq_sum_arr)):
        raise ValueError("Kish effective sample size requires finite inputs.")
    if np.any(weight_sum_arr < 0.0) or np.any(weight_sq_sum_arr < 0.0):
        raise ValueError("Kish effective sample size requires non-negative inputs.")

    out = np.full(weight_sum_arr.shape, np.nan, dtype=np.float64)
    ok = (weight_sum_arr > 0.0) & (weight_sq_sum_arr > 0.0)
    out[ok] = (weight_sum_arr[ok] ** 2) / weight_sq_sum_arr[ok]
    bad = (weight_sum_arr > 0.0) & ~(weight_sq_sum_arr > 0.0)
    if np.any(bad):
        raise ValueError("Positive weight_sum requires positive weight_sq_sum.")
    return out


def _renormalize_usage_excluding(
    usage: np.ndarray,
    excluded: np.ndarray,
    *,
    excluded_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    usage_arr = np.asarray(usage, dtype=np.float64).reshape(-1)
    ex = np.asarray(excluded, dtype=np.float64)
    if ex.ndim != 2:
        raise ValueError("excluded must be a 2D array.")
    if ex.shape[0] != usage_arr.shape[0]:
        raise ValueError("usage and excluded must have the same number of rows.")
    if ex.shape[1] != len(excluded_cols):
        raise ValueError("excluded column count must match excluded_cols length.")
    if np.any(~np.isfinite(usage_arr)) or np.any(~np.isfinite(ex)):
        raise ValueError("Usage renormalization requires finite usage and excluded program values.")

    denom = 1.0 - np.sum(ex, axis=1, dtype=np.float64)
    bad = ~np.isfinite(denom) | (denom <= 0.0)
    if np.any(bad):
        idx = np.flatnonzero(bad)[:5]
        preview = [
            {
                "row": int(i),
                "usage": float(usage_arr[i]),
                "denom": float(denom[i]) if np.isfinite(denom[i]) else float("nan"),
                "excluded_sum": float(np.sum(ex[i, :], dtype=np.float64)),
            }
            for i in idx.tolist()
        ]
        raise ValueError(
            "Usage renormalization denominator must be finite and > 0 for all rows. "
            f"excluded_cols={excluded_cols!r} n_bad={int(np.sum(bad))}; first_bad_rows={preview}"
        )
    return (usage_arr / denom).astype(np.float64, copy=False), denom.astype(np.float64, copy=False)


def _compute_tricycle_theta(adata, *, trc: pd.DataFrame, dataset: np.ndarray, row_idx: np.ndarray | None = None) -> np.ndarray:
    import scipy.sparse as sp

    ref_syms = np.asarray(trc["symbol"], dtype=object)
    var_syms = np.asarray(adata.var_names, dtype=object)
    ref_mask = np.isin(var_syms, ref_syms)
    if not np.any(ref_mask):
        raise ValueError("No shared genes between tricycle ref and adata.var_names.")

    sym_to_ref = {s: i for i, s in enumerate(ref_syms)}
    ref_rows = np.array([sym_to_ref[s] for s in var_syms[ref_mask]], dtype=int)
    w1 = trc["pc1.rot"].to_numpy(float)[ref_rows]
    w2 = trc["pc2.rot"].to_numpy(float)[ref_rows]

    ds = np.asarray(dataset, dtype=object).reshape(-1)
    theta = np.empty(ds.shape[0], dtype=np.float32)
    if row_idx is not None:
        row_idx = np.asarray(row_idx, dtype=np.int64).reshape(-1)
        if row_idx.shape[0] != ds.shape[0]:
            raise ValueError("row_idx must have the same length as dataset.")

    for d in np.unique(ds):
        m = ds == d
        sub_rows = np.flatnonzero(m)
        if sub_rows.size == 0:
            continue
        rows = sub_rows if row_idx is None else row_idx[sub_rows]
        # Avoid boolean indexing into backed arrays; this can be extremely slow.
        Xd = adata.X[rows][:, ref_mask]
        if sp.issparse(Xd):
            mu = np.asarray(Xd.mean(axis=0)).reshape(-1)
            pc1 = (Xd @ w1) - float(mu @ w1)
            pc2 = (Xd @ w2) - float(mu @ w2)
            pc1 = np.asarray(pc1).reshape(-1)
            pc2 = np.asarray(pc2).reshape(-1)
        else:
            Xd = np.asarray(Xd, dtype=float)
            mu = Xd.mean(axis=0)
            pc1 = (Xd @ w1) - float(mu @ w1)
            pc2 = (Xd @ w2) - float(mu @ w2)
        theta[sub_rows] = np.arctan2(pc2, pc1).astype(np.float32, copy=False)
    return theta


def _theta_edges_by_dataset_leiden(
    theta: np.ndarray, dataset_code: np.ndarray, leiden_code: np.ndarray, *, n_bins: int
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    n_bins = int(n_bins)
    for ds in np.unique(dataset_code):
        for lei in np.unique(leiden_code):
            m = (dataset_code == ds) & (leiden_code == lei)
            if not np.any(m):
                continue
            t = theta[m].astype(float, copy=False)
            edges = np.quantile(t, q=np.linspace(0.0, 1.0, n_bins + 1))
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = np.nextafter(edges[i - 1], edges[i - 1] + 1.0)
            out[(int(ds), int(lei))] = edges
    return out


def _assign_theta_bins_from_edges(
    theta: np.ndarray,
    dataset_code: np.ndarray,
    leiden_code: np.ndarray,
    edges_by_group: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    out = np.empty(theta.shape[0], dtype=np.int16)
    for (ds, lei), edges in edges_by_group.items():
        m = (dataset_code == ds) & (leiden_code == lei)
        if not np.any(m):
            continue
        out[m] = np.digitize(theta[m].astype(float, copy=False), edges[1:-1], right=False).astype(np.int16)
    return out


def _theta_bins(theta: np.ndarray, ds_code: np.ndarray, lei_code: np.ndarray, *, n_bins: int, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    n_bins_i = int(n_bins)
    if mode == "quantile":
        edges = _theta_edges_by_dataset_leiden(theta, ds_code, lei_code, n_bins=n_bins_i)
        return _assign_theta_bins_from_edges(theta, ds_code, lei_code, edges)
    if mode == "angle":
        t = (theta.astype(np.float64, copy=False) + (2.0 * np.pi)) % (2.0 * np.pi)
        w = (2.0 * np.pi) / float(n_bins_i)
        return np.clip(np.floor(t / w), 0, n_bins_i - 1).astype(np.int16, copy=False)
    raise ValueError(f"Unknown theta bin mode={mode!r} (expected 'quantile' or 'angle').")


def _residualize_against_theta_within_dataset(
    x: np.ndarray, *, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    sin_t = np.asarray(sin_t, dtype=np.float64).reshape(-1)
    cos_t = np.asarray(cos_t, dtype=np.float64).reshape(-1)
    ds = np.asarray(dataset, dtype=object).reshape(-1)
    if not (x.shape == sin_t.shape == cos_t.shape == ds.shape):
        raise ValueError("x/sin_t/cos_t/dataset length mismatch")

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    for d in np.unique(ds):
        m = ds == d
        if not np.any(m):
            continue
        xm = x[m]
        ok = np.isfinite(xm)
        if not np.any(ok):
            continue
        Z = np.column_stack([np.ones(int(np.sum(ok)), dtype=np.float64), sin_t[m][ok], cos_t[m][ok]])
        coef, *_ = np.linalg.lstsq(Z, xm[ok], rcond=None)
        resid = xm[ok] - (Z @ coef)
        out[np.flatnonzero(m)[ok]] = resid
    return out


def _assign_quantile_bins_global_exact(x: np.ndarray, *, n_bins: int, jitter: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    jitter = np.asarray(jitter, dtype=np.float64).reshape(-1)
    if x.shape != jitter.shape:
        raise ValueError("x/jitter length mismatch")

    n_bins = int(n_bins)
    out = np.full(x.shape[0], -1, dtype=np.int16)
    eps = 1e-9
    ok = np.isfinite(x)
    if not np.any(ok):
        return out
    score = x[ok] + eps * jitter[ok]
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.int64)
    ranks[order] = np.arange(order.shape[0], dtype=np.int64)
    out[np.flatnonzero(ok)] = np.clip((ranks * n_bins) // int(order.shape[0]), 0, n_bins - 1).astype(
        np.int16, copy=False
    )
    return out


def _fit_stratified_glm_binomial(*, y_succ: np.ndarray, y_tot: np.ndarray, strata: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str | None, int, int]:
    import statsmodels.api as sm

    y_succ = np.asarray(y_succ, dtype=float).reshape(-1)
    y_tot = np.asarray(y_tot, dtype=float).reshape(-1)
    strata = np.asarray(strata, dtype=int).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not (y_succ.shape == y_tot.shape == (X.shape[0],) and strata.shape == (X.shape[0],)):
        raise ValueError("Bad shapes for GLM inputs.")

    keep = y_tot > 0
    if not np.any(keep):
        p = int(X.shape[1])
        return (
            np.full(p, np.nan),
            np.full((p, p), np.nan),
            np.full(0, np.nan),
            np.full(0, np.nan),
            "no_rows",
            0,
            0,
        )

    y_succ = y_succ[keep]
    y_tot = y_tot[keep]
    strata = strata[keep]
    X = X[keep]

    n_strata0 = int(strata.max()) + 1
    tot_by_s = np.bincount(strata, weights=y_tot, minlength=n_strata0)
    succ_by_s = np.bincount(strata, weights=y_succ, minlength=n_strata0)
    keep_s = (succ_by_s > 0) & (succ_by_s < tot_by_s)
    keep2 = keep_s[strata]
    if not np.any(keep2):
        p = int(X.shape[1])
        return (
            np.full(p, np.nan),
            np.full((p, p), np.nan),
            np.full(0, np.nan),
            np.full(0, np.nan),
            "no_variation",
            int(y_tot.size),
            int(np.unique(strata).size),
        )

    y_succ = y_succ[keep2]
    y_tot = y_tot[keep2]
    strata = strata[keep2]
    X = X[keep2]

    strata_levels, strata_code = np.unique(strata, return_inverse=True)
    n_s = int(strata_levels.size)
    n_rows = int(strata_code.size)
    p = int(X.shape[1])

    if n_s <= 2000:
        Z = np.zeros((n_rows, n_s), dtype=float)
        Z[np.arange(n_rows), strata_code] = 1.0
        exog = np.concatenate([Z, X], axis=1)
        exog_is_sparse = False
    else:
        from scipy import sparse

        Z = sparse.csr_matrix(
            (np.ones(n_rows, dtype=float), (np.arange(n_rows, dtype=int), strata_code.astype(int))),
            shape=(n_rows, n_s),
        )
        exog = sparse.hstack([Z, sparse.csr_matrix(X)], format="csr")
        exog_is_sparse = True

    y_prop = y_succ / y_tot
    try:
        mod = sm.GLM(y_prop, exog, family=sm.families.Binomial(), freq_weights=y_tot)
        res = mod.fit(maxiter=200, disp=0)
    except Exception as e:
        return (
            np.full(p, np.nan),
            np.full((p, p), np.nan),
            np.full(0, np.nan),
            strata_levels.astype(int, copy=False),
            f"fit_failed:{type(e).__name__}",
            int(n_rows),
            int(n_s),
        )

    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov_params(), dtype=float)
    if params.shape[0] != n_s + p:
        return (
            np.full(p, np.nan),
            np.full((p, p), np.nan),
            np.full(0, np.nan),
            strata_levels.astype(int, copy=False),
            "bad_param_shape",
            int(n_rows),
            int(n_s),
        )

    alpha = params[:n_s]
    beta = params[n_s:]
    if exog_is_sparse:
        # cov is returned dense by statsmodels; keep only slope block for output.
        pass
    cov_beta = cov[n_s:, n_s:]
    return (
        beta.astype(float, copy=False),
        cov_beta.astype(float, copy=False),
        alpha.astype(float, copy=False),
        strata_levels.astype(int, copy=False),
        None,
        int(n_rows),
        int(n_s),
    )


def _t_summary(x: np.ndarray) -> dict[str, float]:
    import scipy.stats as st

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n >= 2 else float("nan")
    if n < 2 or not (sd > 0.0):
        return {
            "n": float(n),
            "mean": float(mean),
            "sd": float(sd),
            "t": float("nan"),
            "p": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    se = sd / math.sqrt(float(n))
    t = mean / se
    df = float(n - 1)
    p = float(2.0 * st.t.sf(abs(t), df=df))
    q = float(st.t.ppf(0.975, df=df))
    return {
        "n": float(n),
        "mean": float(mean),
        "sd": float(sd),
        "t": float(t),
        "p": float(p),
        "ci_low": float(mean - q * se),
        "ci_high": float(mean + q * se),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Model BrdU+EdU+/BrdU+ as a function of CNMF Usage_6, within matched dataset×theta_bin strata, "
            "for specified Leiden clusters."
        )
    )
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument(
        "--usage-parquet",
        type=pathlib.Path,
        default=pathlib.Path("~/nvme/cnmf_all_progenitors/usage_norm.k9.dt0.1.parquet"),
    )
    ap.add_argument("--usage-col", type=str, default="Usage_6")
    ap.add_argument(
        "--usage-renorm-exclude-cols",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional program columns to exclude from denominator when using usage weights. "
            "If set, usage becomes usage_col / (1 - sum(excluded_cols)) row-wise."
        ),
    )
    ap.add_argument(
        "--brdu-col",
        type=str,
        default="brdu_pos",
        help="Column in adata.obs containing the BrdU binary call (0/1 or bool). Ignored if --brdu-threshold is set.",
    )
    ap.add_argument(
        "--edu-col",
        type=str,
        default="edu_pos",
        help="Column in adata.obs containing the EdU binary call (0/1 or bool). Ignored if --edu-threshold is set.",
    )
    ap.add_argument(
        "--brdu-intensity-col",
        type=str,
        default="brdu_mean",
        help="Column in adata.obs containing a BrdU intensity to threshold if --brdu-threshold is set.",
    )
    ap.add_argument(
        "--edu-intensity-col",
        type=str,
        default="edu_mean",
        help="Column in adata.obs containing an EdU intensity to threshold if --edu-threshold is set.",
    )
    ap.add_argument(
        "--brdu-threshold",
        type=float,
        default=None,
        help="If set, define BrdU+ as (obs[brdu_intensity_col] >= threshold) instead of using --brdu-col.",
    )
    ap.add_argument(
        "--edu-threshold",
        type=float,
        default=None,
        help="If set, define EdU+ as (obs[edu_intensity_col] >= threshold) instead of using --edu-col.",
    )
    ap.add_argument("--pulse-call-mode", type=str, choices=["hard", "soft_logistic"], default="hard")
    ap.add_argument(
        "--soft-log-brdu-col",
        type=str,
        default="log_brdu_mean",
        help="BrdU log-intensity column used for soft logistic calls (soft mode only).",
    )
    ap.add_argument(
        "--soft-log-edu-col",
        type=str,
        default="log_edu_mean",
        help="EdU log-intensity column used for soft logistic calls (soft mode only).",
    )
    ap.add_argument(
        "--soft-span",
        type=float,
        default=1.0,
        help="Distance from threshold where p reaches soft-p-hi (and 1-soft-p-hi on the lower side).",
    )
    ap.add_argument(
        "--soft-p-hi",
        type=float,
        default=0.99,
        help="Target probability at threshold+soft-span in soft logistic mode.",
    )
    ap.add_argument(
        "--soft-threshold-uns-key",
        type=str,
        default="brdu_edu_thresholds_by_dataset_roi_ccf_adjusted",
        help="adata.uns key containing per-unit thresholds for soft logistic mode.",
    )
    ap.add_argument(
        "--soft-threshold-delta-csv",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional CSV with columns dataset,roi,ccf_adjusted,delta_log_brdu,delta_log_edu; "
            "applied as shifted thresholds thr'=thr+delta in soft logistic mode."
        ),
    )
    ap.add_argument("--include-leiden", type=str, nargs="+", default=["7", "8", "9", "10"])
    ap.add_argument(
        "--exclude-animals",
        type=str,
        nargs="*",
        default=[],
        help="Optional animal IDs to exclude from analysis (e.g., JaxA2).",
    )
    ap.add_argument(
        "--pool-leiden",
        action="store_true",
        help=(
            "Pool all included Leiden clusters into one analysis group (one curve per animal). "
            "Stratification still includes the original leiden via dataset×theta_bin×leiden."
        ),
    )
    ap.add_argument(
        "--write-by-unit",
        action="store_true",
        help="Also write usage6_by_unit.csv with unit=(dataset×roi×ccf_adjusted) for weighted error bars/diagnostics.",
    )
    ap.add_argument("--theta-bin-mode", type=str, choices=["quantile", "angle"], default="quantile")
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument("--usage-bins", type=int, default=10)
    ap.add_argument(
        "--no-residualize-usage-theta",
        action="store_true",
        help="If set, do not residualize Usage against sin/cos(theta) before binning.",
    )
    ap.add_argument("--outdir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import anndata as ad
    from scipy.special import expit

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log_fp = open(outdir / "run.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        dt = time.time() - t0
        line = f"[{dt:7.1f}s] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log(f"PID={os.getpid()} writing logs to {outdir/'run.log'}")
    log(f"argv={shlex.join(sys.argv)}")

    def _binary_call_from_obs(
        *,
        obs: pd.DataFrame,
        mask: np.ndarray,
        col: str,
        intensity_col: str,
        threshold: float | None,
        name: str,
    ) -> np.ndarray:
        if threshold is None:
            if col not in obs.columns:
                raise ValueError(f"adata.obs missing required column for {name}: {col!r}")
            s = obs.loc[mask, col]
            if pd.api.types.is_bool_dtype(s):
                if bool(pd.isna(s).any()):
                    raise ValueError(f"{name} column {col!r} contains NA; expected a fully-defined binary call.")
                return s.astype(bool).to_numpy(dtype=np.int8, copy=False)
            x = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.int64, copy=False)
            u = np.unique(x)
            if not np.all(np.isin(u, [0, 1])):
                raise ValueError(f"{name} column {col!r} must be binary (0/1 or bool), got unique={u[:10]!r}")
            return x.astype(np.int8, copy=False)

        if intensity_col not in obs.columns:
            raise ValueError(f"adata.obs missing required intensity column for {name}: {intensity_col!r}")
        v = pd.to_numeric(obs.loc[mask, intensity_col], errors="raise").to_numpy(dtype=np.float64, copy=False)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} intensity column {intensity_col!r} contains non-finite values in analysis subset.")
        return (v >= float(threshold)).astype(np.int8, copy=False)

    leiden_keep = {str(x) for x in args.include_leiden}
    exclude_animals = {_normalize_animal_label(x) for x in args.exclude_animals if str(x).strip()}
    usage_col = str(args.usage_col).strip()
    if not usage_col:
        raise ValueError("Empty --usage-col")
    usage_renorm_exclude_cols = [str(c).strip() for c in args.usage_renorm_exclude_cols if str(c).strip()]
    if usage_col in usage_renorm_exclude_cols:
        raise ValueError("--usage-renorm-exclude-cols must not include --usage-col itself.")

    log("Loading h5ad (backed='r')...")
    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    if "dataset" not in obs.columns:
        raise ValueError("adata.obs must contain 'dataset'.")
    for col in ["leiden", "roi", "ccf_adjusted"]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing required column: {col!r}")

    leiden_all = obs["leiden"].astype(str)
    m_lei = leiden_all.isin(sorted(leiden_keep)).to_numpy()
    if not bool(np.any(m_lei)):
        raise ValueError(f"No cells found for leiden in {sorted(leiden_keep)}")

    # Exclude known-bad units (poor staining).
    ds_all = obs["dataset"].astype(str)
    roi_all = obs["roi"].astype(str)
    m_excl_animal = np.zeros(obs.shape[0], dtype=bool)
    if exclude_animals:
        idx_lei = np.flatnonzero(m_lei)
        ds_lei = ds_all.iloc[idx_lei].to_numpy(dtype=object, copy=False)
        animals_lei = np.array([_animal_from_dataset(d) for d in ds_lei.tolist()], dtype=object)
        present_animals = set(animals_lei.tolist())
        missing_animals = sorted(exclude_animals - present_animals)
        if missing_animals:
            raise ValueError(
                f"--exclude-animals contains values absent from selected leiden subset: {missing_animals}"
            )
        m_excl_animal_lei = np.isin(animals_lei, sorted(exclude_animals))
        m_excl_animal[idx_lei] = m_excl_animal_lei
        log(
            f"Excluding animals in selected leiden subset: {sorted(exclude_animals)}; "
            f"n_excluded={int(np.sum(m_excl_animal_lei))}"
        )
    m_excl = ds_all.str.contains("20251005", regex=False) & roi_all.isin(["1", "3"])
    n_excl = int(np.sum(m_lei & m_excl.to_numpy()))
    if n_excl > 0:
        log(f"Excluding poor-stain cells: dataset contains '20251005' and roi in {{1,3}}; n_excluded={n_excl}")
    m = m_lei & (~m_excl.to_numpy()) & (~m_excl_animal)
    if not bool(np.any(m)):
        raise ValueError(
            "All cells were excluded after applying poor-stain and animal exclusion filters."
        )

    row_idx = np.flatnonzero(m).astype(np.int64, copy=False)
    obs_name_raw = adata.obs_names.to_numpy(dtype=object, copy=False)
    obs_name = obs_name_raw[row_idx]
    dataset = obs.loc[m, "dataset"].astype(str).to_numpy().astype(object)
    roi = obs.loc[m, "roi"].astype(str).to_numpy().astype(object)
    ccf_adjusted = obs.loc[m, "ccf_adjusted"].astype(str).to_numpy().astype(object)
    leiden_orig = obs.loc[m, "leiden"].astype(str).to_numpy().astype(object)
    if bool(args.pool_leiden):
        leiden = np.full(leiden_orig.shape[0], "_pooled", dtype=object)
    else:
        leiden = leiden_orig
    B_hard = _binary_call_from_obs(
        obs=obs,
        mask=m,
        col=str(args.brdu_col),
        intensity_col=str(args.brdu_intensity_col),
        threshold=args.brdu_threshold,
        name="BrdU call",
    )
    E_hard = _binary_call_from_obs(
        obs=obs,
        mask=m,
        col=str(args.edu_col),
        intensity_col=str(args.edu_intensity_col),
        threshold=args.edu_threshold,
        name="EdU call",
    )
    b_soft = B_hard.astype(np.float64, copy=False)
    e_soft = E_hard.astype(np.float64, copy=False)
    pi_11 = b_soft * e_soft
    pulse_mode = str(args.pulse_call_mode)
    if pulse_mode == "soft_logistic":
        if args.brdu_threshold is not None or args.edu_threshold is not None:
            raise ValueError("Soft logistic mode does not support --brdu-threshold / --edu-threshold.")
        if not (0.5 < float(args.soft_p_hi) < 1.0):
            raise ValueError("--soft-p-hi must be in (0.5, 1.0).")
        if not (float(args.soft_span) > 0.0):
            raise ValueError("--soft-span must be > 0.")
        for col in [str(args.soft_log_brdu_col), str(args.soft_log_edu_col)]:
            if col not in obs.columns:
                raise ValueError(f"Soft logistic mode missing required obs column: {col!r}")

        uns_key = str(args.soft_threshold_uns_key)
        threshold_map = _load_threshold_map_from_uns(uns=dict(adata.uns), uns_key=uns_key)
        log(f"Soft logistic: loaded threshold units from adata.uns[{uns_key!r}] n_units={len(threshold_map)}")
        threshold_delta_map: dict[tuple[str, str, str], tuple[float, float]] | None = None
        if args.soft_threshold_delta_csv is not None:
            delta_csv = args.soft_threshold_delta_csv.expanduser()
            threshold_delta_map = _load_threshold_delta_map_from_csv(path=delta_csv)
            log(
                f"Soft logistic: loaded threshold deltas from {delta_csv} "
                f"n_units={len(threshold_delta_map)}"
            )

        brdu_thr = np.full(row_idx.shape[0], np.nan, dtype=np.float64)
        edu_thr = np.full(row_idx.shape[0], np.nan, dtype=np.float64)
        brdu_delta = np.zeros(row_idx.shape[0], dtype=np.float64)
        edu_delta = np.zeros(row_idx.shape[0], dtype=np.float64)
        used_units: set[tuple[str, str, str]] = set()
        for i, key in enumerate(zip(dataset.tolist(), roi.tolist(), ccf_adjusted.tolist(), strict=True)):
            key3 = (str(key[0]), str(key[1]), str(key[2]))
            value = threshold_map.get(key3)
            if value is None:
                raise KeyError(f"No adata.uns threshold match for unit={key}")
            d_b = 0.0
            d_e = 0.0
            if threshold_delta_map is not None:
                delta_value = threshold_delta_map.get(key3)
                if delta_value is None:
                    raise KeyError(f"No adata.uns threshold-delta match for unit={key}")
                d_b = float(delta_value[0])
                d_e = float(delta_value[1])
            brdu_delta[i] = d_b
            edu_delta[i] = d_e
            brdu_thr[i] = value[0] + d_b
            edu_thr[i] = value[1] + d_e
            used_units.add(key3)
        if np.any(~np.isfinite(brdu_thr)) or np.any(~np.isfinite(edu_thr)):
            raise ValueError("Non-finite thresholds found after mapping in soft logistic mode.")
        if threshold_delta_map is not None:
            extra_units = len(set(threshold_delta_map) - used_units)
            log(
                "Soft logistic threshold delta summary: "
                f"delta_brdu[min/med/max]={float(np.min(brdu_delta)):.4f}/{float(np.median(brdu_delta)):.4f}/{float(np.max(brdu_delta)):.4f} "
                f"delta_edu[min/med/max]={float(np.min(edu_delta)):.4f}/{float(np.median(edu_delta)):.4f}/{float(np.max(edu_delta)):.4f} "
                f"thr_brdu[min/med/max]={float(np.min(brdu_thr)):.4f}/{float(np.median(brdu_thr)):.4f}/{float(np.max(brdu_thr)):.4f} "
                f"thr_edu[min/med/max]={float(np.min(edu_thr)):.4f}/{float(np.median(edu_thr)):.4f}/{float(np.max(edu_thr)):.4f} "
                f"extra_delta_units_not_used={extra_units}"
            )

        log_brdu = pd.to_numeric(obs.loc[m, str(args.soft_log_brdu_col)], errors="raise").to_numpy(dtype=np.float64, copy=False)
        log_edu = pd.to_numeric(obs.loc[m, str(args.soft_log_edu_col)], errors="raise").to_numpy(dtype=np.float64, copy=False)
        valid_soft = np.isfinite(log_brdu) & np.isfinite(log_edu)
        n_drop_soft = int(np.sum(~valid_soft))
        if n_drop_soft > 0:
            bad_idx = np.flatnonzero(~valid_soft)[:5]
            preview = [
                (
                    str(dataset[i]),
                    str(roi[i]),
                    str(ccf_adjusted[i]),
                    str(obs_name[i]),
                    float(log_brdu[i]) if np.isfinite(log_brdu[i]) else float("nan"),
                    float(log_edu[i]) if np.isfinite(log_edu[i]) else float("nan"),
                )
                for i in bad_idx.tolist()
            ]
            raise ValueError(
                "Soft logistic mode requires finite log BrdU/EdU intensity values. "
                f"Found n_bad={n_drop_soft}; first_bad_rows={preview}"
            )

        b_soft = _soft_logistic_probabilities(
            log_brdu,
            brdu_thr,
            span=float(args.soft_span),
            p_hi=float(args.soft_p_hi),
        )
        e_soft = _soft_logistic_probabilities(
            log_edu,
            edu_thr,
            span=float(args.soft_span),
            p_hi=float(args.soft_p_hi),
        )
        pi_11 = (b_soft * e_soft).astype(np.float64, copy=False)
        log(
            f"Calls: pulse_call_mode=soft_logistic (p@thr=0.5; p@thr±{float(args.soft_span):g}="
            f"{1.0-float(args.soft_p_hi):.3f}/{float(args.soft_p_hi):.3f}) using "
            f"{args.soft_log_brdu_col!r}/{args.soft_log_edu_col!r} and adata.uns thresholds"
            + (
                f" + deltas from {args.soft_threshold_delta_csv}"
                if args.soft_threshold_delta_csv is not None
                else ""
            )
        )
    else:
        log(
            "Calls: pulse_call_mode=hard; "
            + (
                f"B from {args.brdu_col!r}"
                if args.brdu_threshold is None
                else f"B from ({args.brdu_intensity_col!r} >= {args.brdu_threshold:g})"
            )
            + "; "
            + (
                f"E from {args.edu_col!r}"
                if args.edu_threshold is None
                else f"E from ({args.edu_intensity_col!r} >= {args.edu_threshold:g})"
            )
        )

    animals = np.array([_animal_from_dataset(d) for d in dataset.tolist()], dtype=object)
    log(
        f"Subset: n_cells={int(row_idx.size)} n_datasets={int(pd.Series(dataset).nunique())} "
        f"animals={dict(pd.Series(animals).value_counts())}"
    )
    log(
        "Call prevalence in subset: "
        f"hard_brdu_rate={float(np.mean(B_hard)):.4f} hard_edu_rate={float(np.mean(E_hard)):.4f} "
        f"soft_brdu_rate={float(np.mean(b_soft)):.4f} soft_edu_rate={float(np.mean(e_soft)):.4f}"
    )

    brdu_int: np.ndarray | None = None
    edu_int: np.ndarray | None = None
    if str(args.brdu_intensity_col) in obs.columns:
        brdu_int = pd.to_numeric(obs.loc[m, str(args.brdu_intensity_col)], errors="raise").to_numpy(
            dtype=np.float64, copy=False
        )
    if str(args.edu_intensity_col) in obs.columns:
        edu_int = pd.to_numeric(obs.loc[m, str(args.edu_intensity_col)], errors="raise").to_numpy(
            dtype=np.float64, copy=False
        )

    audit_rows: list[dict[str, object]] = []
    for ds in pd.unique(pd.Series(dataset)):
        m = dataset == ds
        audit_rows.append(
            {
                "dataset": str(ds),
                "animal": _animal_from_dataset(str(ds)),
                "n_cells": int(np.sum(m)),
                "brdu_rate": float(np.mean(B_hard[m])),
                "edu_rate": float(np.mean(E_hard[m])),
                "brdu_soft_rate": float(np.mean(b_soft[m])),
                "edu_soft_rate": float(np.mean(e_soft[m])),
                "brdu_int_median": float(np.nanmedian(brdu_int[m])) if brdu_int is not None else float("nan"),
                "brdu_int_p90": float(np.nanquantile(brdu_int[m], 0.9)) if brdu_int is not None else float("nan"),
                "edu_int_median": float(np.nanmedian(edu_int[m])) if edu_int is not None else float("nan"),
                "edu_int_p90": float(np.nanquantile(edu_int[m], 0.9)) if edu_int is not None else float("nan"),
            }
        )
    pd.DataFrame(audit_rows).sort_values(["animal", "dataset"], kind="mergesort").to_csv(
        outdir / "call_audit_by_dataset.csv", index=False
    )

    log("Loading Usage parquet...")
    usage_cols_for_merge = [usage_col, *usage_renorm_exclude_cols]
    u = pd.read_parquet(args.usage_parquet.expanduser(), columns=["index", "dataset", *usage_cols_for_merge])
    u = u.rename(columns={"index": "obs_name"}).copy()
    u["dataset"] = u["dataset"].astype(str)
    u["obs_name"] = u["obs_name"].astype(str)
    # The parquet `obs_name` column is globally unique (suffixes like "-1" may be present).
    # In the h5ad, obs_names are unique *within* each dataset but can repeat across datasets.
    # We therefore merge on (dataset, obs_name_base) where obs_name_base strips a trailing "-<int>" suffix.
    u["obs_name_base"] = u["obs_name"].str.replace(r"-\d+$", "", regex=True)
    if u.duplicated(["dataset", "obs_name_base"]).any():
        raise ValueError("Usage parquet has duplicate (dataset, obs_name_base) keys; cannot perform a 1:1 merge.")

    # Sanity: within the analysis subset, obs_name should be unique within each dataset.
    obs_name_base_left = (
        pd.Series(obs_name.astype(str, copy=False), copy=False)
        .str.split(":", n=1)
        .str[-1]
        .str.replace(r"-\d+$", "", regex=True)
    )
    _key = pd.Series(dataset.astype(str, copy=False), copy=False) + "|" + obs_name_base_left
    if bool(_key.duplicated().any()):
        raise ValueError("Within the analysis subset, obs_name is not unique within dataset (unexpected).")

    left = pd.DataFrame(
        {
            "dataset": dataset.astype(str, copy=False),
            "obs_name_base": obs_name_base_left.to_numpy(dtype=object, copy=False),
            "pos": np.arange(row_idx.size),
        },
        copy=False,
    )
    merged = left.merge(
        u.loc[:, ["dataset", "obs_name_base", *usage_cols_for_merge]],
        on=["dataset", "obs_name_base"],
        how="left",
        sort=False,
    )
    merged = merged.sort_values("pos", kind="mergesort")
    usage_raw = pd.to_numeric(merged[usage_col], errors="raise").to_numpy(dtype=np.float64, copy=False)
    if usage_renorm_exclude_cols:
        excluded = np.column_stack(
            [
                pd.to_numeric(merged[c], errors="raise").to_numpy(dtype=np.float64, copy=False)
                for c in usage_renorm_exclude_cols
            ]
        )
        usage, usage_denom = _renormalize_usage_excluding(
            usage_raw,
            excluded,
            excluded_cols=usage_renorm_exclude_cols,
        )
        log(
            "Usage renormalization applied: "
            f"usage_col={usage_col!r} excluded_cols={usage_renorm_exclude_cols!r} "
            f"denom[min/med/max]={float(np.min(usage_denom)):.4f}/{float(np.median(usage_denom)):.4f}/{float(np.max(usage_denom)):.4f}"
        )
    else:
        usage = usage_raw
    miss = float(np.mean(~np.isfinite(usage)))
    log(f"Usage join: missing_frac={miss:.6f}")
    if miss > 0.01:
        raise ValueError(f"Usage join missing_frac too high ({miss:.3f}); check obs_name/dataset alignment.")

    ds_code, ds_uniq = pd.factorize(dataset, sort=True)
    # Always compute theta bins within dataset×original_leiden.
    lei_code, lei_uniq = pd.factorize(leiden_orig, sort=True)

    log("Computing theta (tricycle) for subset...")
    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    theta = _compute_tricycle_theta(adata, trc=trc, dataset=dataset, row_idx=row_idx)
    theta_bin = _theta_bins(theta, ds_code, lei_code, n_bins=int(args.theta_bins), mode=str(args.theta_bin_mode))

    if bool(args.no_residualize_usage_theta):
        log("Skipping residualization: using raw Usage for binning.")
        usage_for_bins = usage
    else:
        log("Residualizing Usage against sin/cos(theta) within dataset...")
        sin_t = np.sin(theta).astype(np.float64, copy=False)
        cos_t = np.cos(theta).astype(np.float64, copy=False)
        usage_for_bins = _residualize_against_theta_within_dataset(usage, sin_t=sin_t, cos_t=cos_t, dataset=dataset)

    log("Assigning Usage quantile bins (exact, deterministic; global over all included cells)...")
    h = _stable_hash_u64(dataset, obs_name_base_left.to_numpy(dtype=object, copy=False))
    jitter = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)) - 0.5
    usage_bin = _assign_quantile_bins_global_exact(usage_for_bins, n_bins=int(args.usage_bins), jitter=jitter)
    if int(np.min(usage_bin)) < 0:
        raise ValueError("Some cells did not receive a usage_bin (unexpected).")

    # Fit per animal × leiden; retention axis uses hard BrdU+ slice in hard mode and soft BrdU mass in soft mode.
    out_rows: list[dict[str, object]] = []
    by_unit_rows: list[dict[str, object]] = []
    usage_bins = int(args.usage_bins)

    log("Fitting per animal × leiden models...")
    for animal in np.unique(animals):
        m_a = animals == animal
        for lei in np.unique(leiden[m_a]):
            m_b1 = m_a & (leiden == lei) & (B_hard == 1)
            m_all = m_a & (leiden == lei)
            n_b1 = int(np.sum(m_b1))
            n_all = int(np.sum(m_all))
            eff_b_mass = float(np.sum(b_soft[m_all])) if n_all > 0 else float("nan")
            eff_double_mass = float(np.sum(pi_11[m_all])) if n_all > 0 else float("nan")
            raw_f = float(eff_double_mass / eff_b_mass) if (np.isfinite(eff_b_mass) and eff_b_mass > 0.0) else float("nan")
            raw_pE = float(np.mean(e_soft[m_all])) if n_all > 0 else float("nan")
            log(
                f"[block] animal={animal} leiden={lei} n_b1_hard={n_b1} raw_f={raw_f:.4f} "
                f"n_all={n_all} raw_pE={raw_pE:.4f} eff_b_mass={eff_b_mass:.1f}"
            )
            if n_all == 0 or not (eff_b_mass > 0.0):
                continue

            m_ret = m_b1 if pulse_mode == "hard" else m_all
            ds_code_b1, _ = pd.factorize(dataset[m_ret], sort=True)
            if bool(args.pool_leiden):
                lei_code_b1, _ = pd.factorize(leiden_orig[m_ret], sort=True)
                n_lei_b1 = int(lei_code_b1.max()) + 1
                stratum_idx_b1 = (
                    ds_code_b1.astype(np.int64) * int(args.theta_bins) * n_lei_b1
                    + theta_bin[m_ret].astype(np.int64, copy=False) * n_lei_b1
                    + lei_code_b1.astype(np.int64, copy=False)
                )
            else:
                stratum_idx_b1 = ds_code_b1.astype(np.int64) * int(args.theta_bins) + theta_bin[m_ret].astype(
                    np.int64, copy=False
                )
            n_strata0_b1 = int(stratum_idx_b1.max()) + 1

            # Aggregate to (stratum, usage_bin) grouped-binomial rows.
            ub_b1 = usage_bin[m_ret].astype(np.int64, copy=False)
            group_b1 = stratum_idx_b1 * usage_bins + ub_b1
            ret_trial_mass = (
                np.ones(int(np.sum(m_ret)), dtype=np.float64)
                if pulse_mode == "hard"
                else b_soft[m_ret].astype(np.float64, copy=False)
            )
            y_tot = np.bincount(
                group_b1,
                weights=ret_trial_mass,
                minlength=int(n_strata0_b1) * usage_bins,
            ).astype(np.float64, copy=False)
            y_succ = np.bincount(
                group_b1,
                weights=pi_11[m_ret].astype(np.float64, copy=False),
                minlength=int(n_strata0_b1) * usage_bins,
            ).astype(
                np.float64, copy=False
            )

            keep = y_tot > 0
            if not np.any(keep):
                continue
            group_ids = np.flatnonzero(keep).astype(np.int64, copy=False)
            strata_row = (group_ids // usage_bins).astype(np.int64, copy=False)
            ub_row = (group_ids % usage_bins).astype(np.int64, copy=False)
            y_tot_row = y_tot[keep]
            y_succ_row = y_succ[keep]

            # One-hot for usage_bin (baseline bin 0 omitted).
            X = np.zeros((int(group_ids.size), usage_bins - 1), dtype=np.float64)
            for i in range(1, usage_bins):
                X[:, i - 1] = (ub_row == i).astype(np.float64, copy=False)

            beta, cov_beta, alpha, strata_levels, fail, n_rows, n_strata_fit = _fit_stratified_glm_binomial(
                y_succ=y_succ_row, y_tot=y_tot_row, strata=strata_row, X=X
            )
            tot_by_s = np.bincount(stratum_idx_b1, weights=ret_trial_mass, minlength=n_strata0_b1).astype(np.float64, copy=False)

            # g-computation over fitted strata: weights are total BrdU+ cells per stratum (within this animal×leiden).
            w = tot_by_s[strata_levels.astype(int, copy=False)] if strata_levels.size else np.zeros(0, dtype=float)
            w_sum = float(np.sum(w))
            f_hat = np.full(usage_bins, np.nan, dtype=np.float64)
            inv_f_hat = np.full(usage_bins, np.nan, dtype=np.float64)
            ts_over_dt = np.full(usage_bins, np.nan, dtype=np.float64)
            if fail is None and w_sum > 0 and alpha.size:
                for t in range(usage_bins):
                    gamma_t = 0.0 if t == 0 else float(beta[t - 1])
                    p = expit(alpha + gamma_t)
                    f = float(np.sum(w * p) / w_sum)
                    f_hat[t] = f
                    inv_f_hat[t] = float(1.0 / f) if f > 0 else float("nan")
                    ts_over_dt[t] = float(1.0 / (1.0 - f)) if (f > 0.0 and f < 1.0) else float("nan")

            ret_units: dict[tuple[str, str, str], dict[str, object]] = {}
            if bool(args.write_by_unit) and fail is None and alpha.size:
                alpha_full = np.full(n_strata0_b1, np.nan, dtype=np.float64)
                alpha_full[strata_levels.astype(int, copy=False)] = alpha.astype(np.float64, copy=False)
                in_fit = np.isfinite(alpha_full)

                mi = pd.MultiIndex.from_arrays(
                    [
                        dataset[m_ret].astype(str, copy=False),
                        roi[m_ret].astype(str, copy=False),
                        ccf_adjusted[m_ret].astype(str, copy=False),
                    ],
                    names=["dataset", "roi", "ccf_adjusted"],
                )
                unit_code, unit_levels = pd.factorize(mi, sort=True)
                n_units_total = int(unit_levels.size)
                if int(np.max(unit_code)) >= 2**32:
                    raise ValueError("Too many units for bitpacking (unexpected).")
                if int(n_strata0_b1) >= 2**32:
                    raise ValueError("Too many strata for bitpacking (unexpected).")

                # Raw counts per unit×usage_bin (total and kept strata only), for reviewability and boundary audits.
                unit_bin = unit_code.astype(np.int64, copy=False) * usage_bins + ub_b1
                n_b1_total = np.bincount(unit_bin, minlength=n_units_total * usage_bins).reshape(n_units_total, usage_bins)
                eff_n_b1_total = np.bincount(
                    unit_bin,
                    weights=ret_trial_mass,
                    minlength=n_units_total * usage_bins,
                ).reshape(n_units_total, usage_bins)
                k_b1e_total = np.bincount(
                    unit_bin,
                    weights=pi_11[m_ret].astype(np.float64, copy=False),
                    minlength=n_units_total * usage_bins,
                ).reshape(n_units_total, usage_bins)

                cell_keep_b1 = in_fit[stratum_idx_b1.astype(np.int64, copy=False)]
                unit_bin_kept = unit_code[cell_keep_b1].astype(np.int64, copy=False) * usage_bins + ub_b1[cell_keep_b1]
                n_b1_kept = np.bincount(unit_bin_kept, minlength=n_units_total * usage_bins).reshape(n_units_total, usage_bins)
                eff_n_b1_kept = np.bincount(
                    unit_bin_kept,
                    weights=ret_trial_mass[cell_keep_b1],
                    minlength=n_units_total * usage_bins,
                ).reshape(n_units_total, usage_bins)
                k_b1e_kept = np.bincount(
                    unit_bin_kept,
                    weights=pi_11[m_ret][cell_keep_b1].astype(np.float64, copy=False),
                    minlength=n_units_total * usage_bins,
                ).reshape(n_units_total, usage_bins)

                pair = (unit_code.astype(np.uint64) << np.uint64(32)) | stratum_idx_b1.astype(np.uint64, copy=False)
                uniq_pair, inv_pair = np.unique(pair, return_inverse=True)
                pair_mass = np.bincount(inv_pair, weights=ret_trial_mass, minlength=uniq_pair.size).astype(
                    np.float64, copy=False
                )
                unit_of_pair = (uniq_pair >> np.uint64(32)).astype(np.int64, copy=False)
                stratum_of_pair = (uniq_pair & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
                w_sum_total_by_unit = np.bincount(
                    unit_of_pair,
                    weights=pair_mass,
                    minlength=n_units_total,
                ).astype(np.float64, copy=False)
                keep_pair = in_fit[stratum_of_pair]
                if np.any(keep_pair):
                    unit_k = unit_of_pair[keep_pair]
                    w_pair = pair_mass[keep_pair].astype(np.float64, copy=False)
                    s_pair = stratum_of_pair[keep_pair]
                    # uniq_pair is sorted, so unit_k is already grouped.
                    idx0 = np.flatnonzero(np.r_[True, unit_k[1:] != unit_k[:-1]])
                    unit_ids = unit_k[idx0]
                    w_sum_u = np.add.reduceat(w_pair, idx0)
                    if not bool(np.all(w_sum_u > 0)):
                        raise ValueError("Found unit with zero weight (unexpected).")

                    unit_weight_sq_total = np.bincount(
                        unit_code.astype(np.int64, copy=False),
                        weights=np.square(ret_trial_mass),
                        minlength=n_units_total,
                    ).astype(np.float64, copy=False)
                    unit_weight_sq_kept = np.bincount(
                        unit_code[cell_keep_b1].astype(np.int64, copy=False),
                        weights=np.square(ret_trial_mass[cell_keep_b1]),
                        minlength=n_units_total,
                    ).astype(np.float64, copy=False)
                    w_sum_total_u = _kish_effective_sample_size(
                        w_sum_total_by_unit[unit_ids],
                        unit_weight_sq_total[unit_ids],
                    )
                    w_sum_u = _kish_effective_sample_size(
                        w_sum_u,
                        unit_weight_sq_kept[unit_ids],
                    )
                    if not bool(np.all(np.isfinite(w_sum_u) & (w_sum_u > 0.0))):
                        raise ValueError("Found unit with invalid Kish effective sample size (unexpected).")

                    ds_u = unit_levels.get_level_values(0).to_numpy(dtype=object, copy=False)[unit_ids]
                    roi_u = unit_levels.get_level_values(1).to_numpy(dtype=object, copy=False)[unit_ids]
                    ccf_u = unit_levels.get_level_values(2).to_numpy(dtype=object, copy=False)[unit_ids]

                    kept_frac = float(np.sum(w_sum_u) / np.sum(w_sum_total_u)) if float(np.sum(w_sum_total_u)) > 0 else float("nan")
                    log(
                        f"[units_b1] animal={animal} leiden={lei} n_units_kept={int(unit_ids.size)}/{n_units_total} "
                        f"b1_weight_kept={float(np.sum(w_sum_u)):.0f} b1_weight_total={float(np.sum(w_sum_total_u)):.0f} "
                        f"kept_frac={kept_frac:.3f}"
                    )

                    for t in range(usage_bins):
                        gamma_t = 0.0 if t == 0 else float(beta[t - 1])
                        p_pair = expit(alpha_full[s_pair] + gamma_t)
                        num_u = np.add.reduceat(w_pair * p_pair, idx0)
                        f_u = num_u / w_sum_u
                        if np.any(f_u <= 0) or np.any(f_u >= 1):
                            raise ValueError("Some unit-level f_hat are outside (0,1); cannot compute T_S/Δt.")
                        inv_f_u = 1.0 / f_u
                        ts_over_dt_u = 1.0 / (1.0 - f_u)
                        for j in range(int(unit_ids.size)):
                            key = (str(ds_u[j]), str(roi_u[j]), str(ccf_u[j]))
                            cur = ret_units.get(key)
                            if cur is None:
                                cur = {
                                    "animal": str(animal),
                                    "leiden": str(lei),
                                    "dataset": str(ds_u[j]),
                                    "roi": str(roi_u[j]),
                                    "ccf_adjusted": str(ccf_u[j]),
                                    "unit_weight_b1": float(w_sum_u[j]),
                                    "unit_weight_b1_total": float(w_sum_total_u[j]),
                                    "unit_weight_b1_frac_kept": float(w_sum_u[j] / w_sum_total_u[j]) if w_sum_total_u[j] > 0 else float("nan"),
                                    "f_hat": np.full(usage_bins, np.nan, dtype=np.float64),
                                    "inv_f_hat": np.full(usage_bins, np.nan, dtype=np.float64),
                                    "ts_over_dt": np.full(usage_bins, np.nan, dtype=np.float64),
                                    "n_b1_total_bin": np.asarray(n_b1_total[unit_ids[j]], dtype=np.int64).copy(),
                                    "eff_n_b1_total_bin": np.asarray(eff_n_b1_total[unit_ids[j]], dtype=np.float64).copy(),
                                    "k_b1e_total_bin": np.asarray(k_b1e_total[unit_ids[j]], dtype=np.float64).copy(),
                                    "n_b1_kept_bin": np.asarray(n_b1_kept[unit_ids[j]], dtype=np.int64).copy(),
                                    "eff_n_b1_kept_bin": np.asarray(eff_n_b1_kept[unit_ids[j]], dtype=np.float64).copy(),
                                    "k_b1e_kept_bin": np.asarray(k_b1e_kept[unit_ids[j]], dtype=np.float64).copy(),
                                }
                                ret_units[key] = cur
                            cur["f_hat"][t] = float(f_u[j])
                            cur["inv_f_hat"][t] = float(inv_f_u[j])
                            cur["ts_over_dt"][t] = float(ts_over_dt_u[j])

            # Fit marginal EdU labeling index pE_hat(t) on all cells (same strata logic).
            ds_code_all, _ = pd.factorize(dataset[m_all], sort=True)
            if bool(args.pool_leiden):
                lei_code_all, _ = pd.factorize(leiden_orig[m_all], sort=True)
                n_lei_all = int(lei_code_all.max()) + 1
                stratum_idx_all = (
                    ds_code_all.astype(np.int64) * int(args.theta_bins) * n_lei_all
                    + theta_bin[m_all].astype(np.int64, copy=False) * n_lei_all
                    + lei_code_all.astype(np.int64, copy=False)
                )
            else:
                stratum_idx_all = ds_code_all.astype(np.int64) * int(args.theta_bins) + theta_bin[m_all].astype(
                    np.int64, copy=False
                )
            n_strata0_all = int(stratum_idx_all.max()) + 1

            ub_all = usage_bin[m_all].astype(np.int64, copy=False)
            group_all = stratum_idx_all * usage_bins + ub_all
            y_tot_all = np.bincount(group_all, minlength=int(n_strata0_all) * usage_bins).astype(np.float64, copy=False)
            y_succ_all = np.bincount(
                group_all,
                weights=e_soft[m_all].astype(np.float64, copy=False),
                minlength=int(n_strata0_all) * usage_bins,
            ).astype(np.float64, copy=False)
            keep_all = y_tot_all > 0
            if not np.any(keep_all):
                continue
            group_ids_all = np.flatnonzero(keep_all).astype(np.int64, copy=False)
            strata_row_all = (group_ids_all // usage_bins).astype(np.int64, copy=False)
            ub_row_all = (group_ids_all % usage_bins).astype(np.int64, copy=False)
            y_tot_row_all = y_tot_all[keep_all]
            y_succ_row_all = y_succ_all[keep_all]

            X_all = np.zeros((int(group_ids_all.size), usage_bins - 1), dtype=np.float64)
            for i in range(1, usage_bins):
                X_all[:, i - 1] = (ub_row_all == i).astype(np.float64, copy=False)

            beta_all, cov_beta_all, alpha_all, strata_levels_all, fail_all, n_rows_all, n_strata_fit_all = _fit_stratified_glm_binomial(
                y_succ=y_succ_row_all,
                y_tot=y_tot_row_all,
                strata=strata_row_all,
                X=X_all,
            )
            tot_by_s_all = np.bincount(stratum_idx_all, minlength=n_strata0_all).astype(np.float64, copy=False)
            w_all = tot_by_s_all[strata_levels_all.astype(int, copy=False)] if strata_levels_all.size else np.zeros(0, dtype=float)
            w_sum_all = float(np.sum(w_all))
            pE_hat = np.full(usage_bins, np.nan, dtype=np.float64)
            if fail_all is None and w_sum_all > 0 and alpha_all.size:
                for t in range(usage_bins):
                    gamma_t = 0.0 if t == 0 else float(beta_all[t - 1])
                    p = expit(alpha_all + gamma_t)
                    pe = float(np.sum(w_all * p) / w_sum_all)
                    pE_hat[t] = pe

            if bool(args.write_by_unit) and fail_all is None and alpha_all.size and ret_units:
                alpha_full_all = np.full(n_strata0_all, np.nan, dtype=np.float64)
                alpha_full_all[strata_levels_all.astype(int, copy=False)] = alpha_all.astype(np.float64, copy=False)
                in_fit_all = np.isfinite(alpha_full_all)

                mi_all = pd.MultiIndex.from_arrays(
                    [
                        dataset[m_all].astype(str, copy=False),
                        roi[m_all].astype(str, copy=False),
                        ccf_adjusted[m_all].astype(str, copy=False),
                    ],
                    names=["dataset", "roi", "ccf_adjusted"],
                )
                unit_code_all, unit_levels_all = pd.factorize(mi_all, sort=True)
                n_units_total_all = int(unit_levels_all.size)
                if int(np.max(unit_code_all)) >= 2**32:
                    raise ValueError("Too many units for bitpacking (unexpected).")
                if int(n_strata0_all) >= 2**32:
                    raise ValueError("Too many strata for bitpacking (unexpected).")

                unit_bin_all = unit_code_all.astype(np.int64, copy=False) * usage_bins + ub_all
                n_all_total = np.bincount(unit_bin_all, minlength=n_units_total_all * usage_bins).reshape(
                    n_units_total_all, usage_bins
                )
                k_e_total = np.bincount(
                    unit_bin_all,
                    weights=e_soft[m_all].astype(np.float64, copy=False),
                    minlength=n_units_total_all * usage_bins,
                ).reshape(n_units_total_all, usage_bins)

                cell_keep_all = in_fit_all[stratum_idx_all.astype(np.int64, copy=False)]
                unit_bin_all_kept = (
                    unit_code_all[cell_keep_all].astype(np.int64, copy=False) * usage_bins + ub_all[cell_keep_all]
                )
                n_all_kept = np.bincount(unit_bin_all_kept, minlength=n_units_total_all * usage_bins).reshape(
                    n_units_total_all, usage_bins
                )
                k_e_kept = np.bincount(
                    unit_bin_all_kept,
                    weights=e_soft[m_all][cell_keep_all].astype(np.float64, copy=False),
                    minlength=n_units_total_all * usage_bins,
                ).reshape(n_units_total_all, usage_bins)

                pair_all = (unit_code_all.astype(np.uint64) << np.uint64(32)) | stratum_idx_all.astype(np.uint64, copy=False)
                uniq_pair_all, counts_all = np.unique(pair_all, return_counts=True)
                unit_of_pair_all = (uniq_pair_all >> np.uint64(32)).astype(np.int64, copy=False)
                stratum_of_pair_all = (uniq_pair_all & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
                w_sum_total_by_unit_all = np.bincount(
                    unit_of_pair_all,
                    weights=counts_all.astype(np.float64, copy=False),
                    minlength=n_units_total_all,
                ).astype(np.float64, copy=False)
                keep_pair_all = in_fit_all[stratum_of_pair_all]
                if np.any(keep_pair_all):
                    unit_k_all = unit_of_pair_all[keep_pair_all]
                    w_pair_all = counts_all[keep_pair_all].astype(np.float64, copy=False)
                    s_pair_all = stratum_of_pair_all[keep_pair_all]
                    idx0_all = np.flatnonzero(np.r_[True, unit_k_all[1:] != unit_k_all[:-1]])
                    unit_ids_all = unit_k_all[idx0_all]
                    w_sum_u_all = np.add.reduceat(w_pair_all, idx0_all)
                    w_sum_total_u_all = w_sum_total_by_unit_all[unit_ids_all]

                    ds_u_all = unit_levels_all.get_level_values(0).to_numpy(dtype=object, copy=False)[unit_ids_all]
                    roi_u_all = unit_levels_all.get_level_values(1).to_numpy(dtype=object, copy=False)[unit_ids_all]
                    ccf_u_all = unit_levels_all.get_level_values(2).to_numpy(dtype=object, copy=False)[unit_ids_all]

                    key_to_idx = {(str(ds_u_all[j]), str(roi_u_all[j]), str(ccf_u_all[j])): j for j in range(int(unit_ids_all.size))}
                    kept_keys = [k for k in ret_units.keys() if k in key_to_idx]
                    if len(kept_keys) != len(ret_units):
                        missing = len(ret_units) - len(kept_keys)
                        raise ValueError(f"Some retention units are missing from all-cells units (unexpected); missing={missing}")

                    idx_sel = np.array([key_to_idx[k] for k in kept_keys], dtype=np.int64)
                    kept_frac_all = (
                        float(np.sum(w_sum_u_all[idx_sel]) / np.sum(w_sum_total_u_all[idx_sel]))
                        if float(np.sum(w_sum_total_u_all[idx_sel])) > 0
                        else float("nan")
                    )
                    log(
                        f"[units_all] animal={animal} leiden={lei} n_units_kept={int(idx_sel.size)}/{n_units_total_all} "
                        f"all_weight_kept={float(np.sum(w_sum_u_all[idx_sel])):.0f} all_weight_total={float(np.sum(w_sum_total_u_all[idx_sel])):.0f} "
                        f"kept_frac={kept_frac_all:.3f}"
                    )

                    # Compute pE per selected unit × usage_bin.
                    for t in range(usage_bins):
                        gamma_t = 0.0 if t == 0 else float(beta_all[t - 1])
                        p_pair = expit(alpha_full_all[s_pair_all] + gamma_t)
                        num_u = np.add.reduceat(w_pair_all * p_pair, idx0_all)
                        pE_u_all = num_u / w_sum_u_all
                        for k, j in zip(kept_keys, idx_sel, strict=True):
                            ru = ret_units[k]
                            unit_code_val = int(unit_ids_all[j])
                            pe = float(pE_u_all[j])
                            ts = float(ru["ts_over_dt"][t])
                            odds = float(pe / (1.0 - pe)) if (pe > 0.0 and pe < 1.0) else float("nan")
                            tc = float(ts / pe) if (np.isfinite(ts) and pe > 0.0) else float("nan")
                            by_unit_rows.append(
                                {
                                    "animal": str(animal),
                                    "leiden": str(lei),
                                    "dataset": str(ru["dataset"]),
                                    "roi": str(ru["roi"]),
                                    "ccf_adjusted": str(ru["ccf_adjusted"]),
                                    "usage_bin": int(t),
                                    "f_hat": float(ru["f_hat"][t]),
                                    "inv_f_hat": float(ru["inv_f_hat"][t]),
                                    "ts_over_dt": float(ru["ts_over_dt"][t]),
                                    "pE_hat": float(pe),
                                    "pE_odds_hat": float(odds),
                                    "tc_over_dt": float(tc),
                                    "unit_weight_b1": float(ru["unit_weight_b1"]),
                                    "unit_weight_b1_total": float(ru["unit_weight_b1_total"]),
                                    "unit_weight_b1_frac_kept": float(ru["unit_weight_b1_frac_kept"]),
                                    "unit_weight_all": float(w_sum_u_all[j]),
                                    "unit_weight_all_total": float(w_sum_total_u_all[j]),
                                    "unit_weight_all_frac_kept": float(w_sum_u_all[j] / w_sum_total_u_all[j]) if w_sum_total_u_all[j] > 0 else float("nan"),
                                    "n_b1_bin_total": int(ru["n_b1_total_bin"][t]),
                                    "eff_n_b1_bin_total": float(ru["eff_n_b1_total_bin"][t]),
                                    "k_b1e_bin_total": float(ru["k_b1e_total_bin"][t]),
                                    "n_b1_bin_kept": int(ru["n_b1_kept_bin"][t]),
                                    "eff_n_b1_bin_kept": float(ru["eff_n_b1_kept_bin"][t]),
                                    "k_b1e_bin_kept": float(ru["k_b1e_kept_bin"][t]),
                                    "n_all_bin_total": int(n_all_total[unit_code_val, t]),
                                    "k_e_bin_total": float(k_e_total[unit_code_val, t]),
                                    "n_all_bin_kept": int(n_all_kept[unit_code_val, t]),
                                    "k_e_bin_kept": float(k_e_kept[unit_code_val, t]),
                                }
                            )

            for t in range(usage_bins):
                pe = float(pE_hat[t])
                odds = float(pe / (1.0 - pe)) if (pe > 0.0 and pe < 1.0) else float("nan")
                tc_over_dt = float(ts_over_dt[t] / pe) if (np.isfinite(ts_over_dt[t]) and pe > 0.0) else float("nan")
                out_rows.append(
                    {
                        "animal": str(animal),
                        "leiden": str(lei),
                        "usage_bin": int(t),
                        "f_hat": float(f_hat[t]),
                        "inv_f_hat": float(inv_f_hat[t]),
                        "ts_over_dt": float(ts_over_dt[t]),
                        "pE_hat": float(pE_hat[t]),
                        "pE_odds_hat": float(odds),
                        "tc_over_dt": float(tc_over_dt),
                        "raw_f": float(raw_f),
                        "raw_pE": float(raw_pE),
                        "eff_b_mass_block": float(eff_b_mass),
                        "eff_double_mass_block": float(eff_double_mass),
                        "n_cells_b1": int(n_b1),
                        "n_cells_all": int(n_all),
                        "n_rows": int(n_rows),
                        "n_strata_fit": int(n_strata_fit),
                        "fail_reason": str(fail) if fail is not None else "",
                    }
                )

    by_animal = pd.DataFrame(out_rows)
    if by_animal.empty:
        raise ValueError("No results produced (empty by_animal table).")

    by_animal.to_csv(outdir / "usage6_by_animal.csv", index=False)
    log(f"Wrote {outdir/'usage6_by_animal.csv'} rows={int(by_animal.shape[0])}")

    if bool(args.write_by_unit):
        by_unit = pd.DataFrame(by_unit_rows)
        if by_unit.empty:
            raise ValueError("--write-by-unit requested but no by-unit rows were produced.")
        by_unit.to_csv(outdir / "usage6_by_unit.csv", index=False)
        log(f"Wrote {outdir/'usage6_by_unit.csv'} rows={int(by_unit.shape[0])}")

    # Across-animal summaries.
    meta_rows: list[dict[str, object]] = []
    for lei in sorted(by_animal["leiden"].unique().tolist(), key=lambda s: int(s) if str(s).isdigit() else str(s)):
        for t in range(usage_bins):
            d = by_animal[(by_animal["leiden"] == lei) & (by_animal["usage_bin"] == t)]
            s_f = _t_summary(d["f_hat"].to_numpy(float, copy=False))
            s_inv = _t_summary(d["inv_f_hat"].to_numpy(float, copy=False))
            s_pE = _t_summary(d["pE_hat"].to_numpy(float, copy=False))
            s_ts = _t_summary(d["ts_over_dt"].to_numpy(float, copy=False))
            s_tc = _t_summary(d["tc_over_dt"].to_numpy(float, copy=False))
            meta_rows.append(
                {
                    "leiden": str(lei),
                    "usage_bin": int(t),
                    "n_animals_used_f_hat": int(s_f["n"]),
                    "animal_mean_f_hat": float(s_f["mean"]),
                    "animal_sd_f_hat": float(s_f["sd"]),
                    "animal_t_f_hat": float(s_f["t"]),
                    "animal_p_f_hat": float(s_f["p"]),
                    "animal_ci_low_f_hat": float(s_f["ci_low"]),
                    "animal_ci_high_f_hat": float(s_f["ci_high"]),
                    "n_animals_used_inv_f_hat": int(s_inv["n"]),
                    "animal_mean_inv_f_hat": float(s_inv["mean"]),
                    "animal_sd_inv_f_hat": float(s_inv["sd"]),
                    "animal_t_inv_f_hat": float(s_inv["t"]),
                    "animal_p_inv_f_hat": float(s_inv["p"]),
                    "animal_ci_low_inv_f_hat": float(s_inv["ci_low"]),
                    "animal_ci_high_inv_f_hat": float(s_inv["ci_high"]),
                    "n_animals_used_pE_hat": int(s_pE["n"]),
                    "animal_mean_pE_hat": float(s_pE["mean"]),
                    "animal_sd_pE_hat": float(s_pE["sd"]),
                    "animal_t_pE_hat": float(s_pE["t"]),
                    "animal_p_pE_hat": float(s_pE["p"]),
                    "animal_ci_low_pE_hat": float(s_pE["ci_low"]),
                    "animal_ci_high_pE_hat": float(s_pE["ci_high"]),
                    "n_animals_used_ts_over_dt": int(s_ts["n"]),
                    "animal_mean_ts_over_dt": float(s_ts["mean"]),
                    "animal_sd_ts_over_dt": float(s_ts["sd"]),
                    "animal_t_ts_over_dt": float(s_ts["t"]),
                    "animal_p_ts_over_dt": float(s_ts["p"]),
                    "animal_ci_low_ts_over_dt": float(s_ts["ci_low"]),
                    "animal_ci_high_ts_over_dt": float(s_ts["ci_high"]),
                    "n_animals_used_tc_over_dt": int(s_tc["n"]),
                    "animal_mean_tc_over_dt": float(s_tc["mean"]),
                    "animal_sd_tc_over_dt": float(s_tc["sd"]),
                    "animal_t_tc_over_dt": float(s_tc["t"]),
                    "animal_p_tc_over_dt": float(s_tc["p"]),
                    "animal_ci_low_tc_over_dt": float(s_tc["ci_low"]),
                    "animal_ci_high_tc_over_dt": float(s_tc["ci_high"]),
                }
            )

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(outdir / "usage6_meta.csv", index=False)
    log(f"Wrote {outdir/'usage6_meta.csv'} rows={int(meta.shape[0])}")

    defs = """# usage6_* outputs: column definitions (Usage_6 → BrdU/EdU dual-pulse proxies)

This outdir contains stage-1 matched-strata predictions and derived proxies.

## Key columns (per animal×usage_bin or per unit×usage_bin)

- `f_hat`: estimated retention fraction `P(EdU+ | BrdU+, usage_bin, matched strata)`.
- `ts_over_dt`: S-phase time proxy `T_S/Δt ≈ 1/(1 - f_hat)` (valid when `0 < f_hat < 1`).
- `pE_hat`: estimated labeling fraction `P(EdU+ | usage_bin, matched strata)`.
- `tc_over_dt`: **cell-cycle proxy** `T_C/Δt ≈ (T_S/Δt) / pE_hat`. This equals literal `T_C/Δt` only if the effective growth fraction is ~1; otherwise it behaves like `T_C/(GF·Δt)`.

## Unit support / auditing columns (usage6_by_unit.csv)

- `unit_weight_b1`: Kish effective BrdU+ support size for the strata kept by the retention fit. `unit_weight_all`: all-cell support weight for strata kept by the marginal EdU fit.
- `n_*` / `k_*` columns: raw counts per `unit×usage_bin` (both total and restricted to kept strata), for boundary/leverage diagnostics.
- `eff_n_b1_*` columns: effective BrdU+ trial mass per `unit×usage_bin` under soft logistic calls (equals integer counts in hard mode).
"""
    (outdir / "column_definitions.md").write_text(defs, encoding="utf-8")
    log(f"Wrote {outdir/'column_definitions.md'}")

    log("Done.")


if __name__ == "__main__":
    main()
