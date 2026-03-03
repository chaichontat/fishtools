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


def _stable_hash_u64(dataset: np.ndarray, obs_name: np.ndarray) -> np.ndarray:
    ds = dataset.astype(str)
    on = obs_name.astype(str)
    s = pd.Series(
        [f"{d}|{o}|{i}" for i, (d, o) in enumerate(zip(ds.tolist(), on.tolist(), strict=True))],
        copy=False,
    )
    return pd.util.hash_pandas_object(s, index=False).to_numpy(np.uint64, copy=False)


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
    ap.add_argument("--include-leiden", type=str, nargs="+", default=["7", "8", "9", "10"])
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

    leiden_keep = {str(x) for x in args.include_leiden}
    usage_col = str(args.usage_col).strip()
    if not usage_col:
        raise ValueError("Empty --usage-col")

    log("Loading h5ad (backed='r')...")
    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    if "dataset" not in obs.columns:
        raise ValueError("adata.obs must contain 'dataset'.")
    for col in ["brdu_pos", "edu_pos", "leiden"]:
        if col not in obs.columns:
            raise ValueError(f"adata.obs missing required column: {col!r}")

    leiden_all = obs["leiden"].astype(str)
    m_lei = leiden_all.isin(sorted(leiden_keep)).to_numpy()
    if not bool(np.any(m_lei)):
        raise ValueError(f"No cells found for leiden in {sorted(leiden_keep)}")

    row_idx = np.flatnonzero(m_lei).astype(np.int64, copy=False)
    obs_name_raw = adata.obs_names.to_numpy(dtype=object, copy=False)
    obs_name = obs_name_raw[row_idx]
    dataset = obs.loc[m_lei, "dataset"].astype(str).to_numpy().astype(object)
    roi = obs.loc[m_lei, "roi"].astype(str).to_numpy().astype(object)
    ccf_adjusted = obs.loc[m_lei, "ccf_adjusted"].astype(str).to_numpy().astype(object)
    leiden_orig = obs.loc[m_lei, "leiden"].astype(str).to_numpy().astype(object)
    if bool(args.pool_leiden):
        leiden = np.full(leiden_orig.shape[0], "_pooled", dtype=object)
    else:
        leiden = leiden_orig
    B = obs.loc[m_lei, "brdu_pos"].to_numpy().astype(int, copy=False)
    E = obs.loc[m_lei, "edu_pos"].to_numpy().astype(int, copy=False)

    animals = np.array([_animal_from_dataset(d) for d in dataset.tolist()], dtype=object)
    log(
        f"Subset: n_cells={int(row_idx.size)} n_datasets={int(pd.Series(dataset).nunique())} "
        f"animals={dict(pd.Series(animals).value_counts())}"
    )

    log("Loading Usage parquet...")
    u = pd.read_parquet(args.usage_parquet.expanduser(), columns=["index", "dataset", usage_col])
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
    _key = pd.Series(dataset.astype(str, copy=False), copy=False) + "|" + pd.Series(obs_name.astype(str, copy=False), copy=False)
    if bool(_key.duplicated().any()):
        raise ValueError("Within the analysis subset, obs_name is not unique within dataset (unexpected).")

    left = pd.DataFrame(
        {
            "dataset": dataset.astype(str, copy=False),
            "obs_name_base": obs_name.astype(str, copy=False),
            "pos": np.arange(row_idx.size),
        },
        copy=False,
    )
    merged = left.merge(u.loc[:, ["dataset", "obs_name_base", usage_col]], on=["dataset", "obs_name_base"], how="left", sort=False)
    merged = merged.sort_values("pos", kind="mergesort")
    usage = merged[usage_col].to_numpy(dtype=np.float64, copy=False)
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

    log("Residualizing Usage against sin/cos(theta) within dataset...")
    sin_t = np.sin(theta).astype(np.float64, copy=False)
    cos_t = np.cos(theta).astype(np.float64, copy=False)
    usage_resid = _residualize_against_theta_within_dataset(usage, sin_t=sin_t, cos_t=cos_t, dataset=dataset)

    log("Assigning Usage quantile bins (exact, deterministic; global over all included cells)...")
    h = _stable_hash_u64(dataset, obs_name)
    jitter = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)) - 0.5
    usage_bin = _assign_quantile_bins_global_exact(usage_resid, n_bins=int(args.usage_bins), jitter=jitter)
    if int(np.min(usage_bin)) < 0:
        raise ValueError("Some cells did not receive a usage_bin (unexpected).")

    # Fit per animal × leiden, restricting outcome to BrdU+ cells.
    out_rows: list[dict[str, object]] = []
    by_unit_rows: list[dict[str, object]] = []
    usage_bins = int(args.usage_bins)

    log("Fitting per animal × leiden models (BrdU+ only)...")
    for animal in np.unique(animals):
        m_a = animals == animal
        for lei in np.unique(leiden[m_a]):
            m = m_a & (leiden == lei) & (B == 1)
            n_b1 = int(np.sum(m))
            raw_f = float(np.mean(E[m])) if n_b1 > 0 else float("nan")
            log(f"[block] animal={animal} leiden={lei} n_b1={n_b1} raw_f={raw_f:.4f}")
            if n_b1 == 0:
                continue

            ds_code_m, ds_uniq_m = pd.factorize(dataset[m], sort=True)
            if bool(args.pool_leiden):
                lei_code_m, _ = pd.factorize(leiden_orig[m], sort=True)
                n_lei = int(lei_code_m.max()) + 1
                stratum_idx = (
                    ds_code_m.astype(np.int64) * int(args.theta_bins) * n_lei
                    + theta_bin[m].astype(np.int64, copy=False) * n_lei
                    + lei_code_m.astype(np.int64, copy=False)
                )
            else:
                stratum_idx = ds_code_m.astype(np.int64) * int(args.theta_bins) + theta_bin[m].astype(
                    np.int64, copy=False
                )
            n_strata0 = int(stratum_idx.max()) + 1

            # Aggregate to (stratum, usage_bin) grouped-binomial rows.
            ub = usage_bin[m].astype(np.int64, copy=False)
            group = stratum_idx * usage_bins + ub
            y_tot = np.bincount(group, minlength=int(n_strata0) * usage_bins).astype(np.float64, copy=False)
            y_succ = np.bincount(group, weights=E[m].astype(np.float64, copy=False), minlength=int(n_strata0) * usage_bins).astype(
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
            tot_by_s = np.bincount(stratum_idx, minlength=n_strata0).astype(np.float64, copy=False)

            # g-computation over fitted strata: weights are total BrdU+ cells per stratum (within this animal×leiden).
            w = tot_by_s[strata_levels.astype(int, copy=False)] if strata_levels.size else np.zeros(0, dtype=float)
            w_sum = float(np.sum(w))
            f_hat = np.full(usage_bins, np.nan, dtype=np.float64)
            inv_f_hat = np.full(usage_bins, np.nan, dtype=np.float64)
            if fail is None and w_sum > 0 and alpha.size:
                for t in range(usage_bins):
                    gamma_t = 0.0 if t == 0 else float(beta[t - 1])
                    p = expit(alpha + gamma_t)
                    f = float(np.sum(w * p) / w_sum)
                    f_hat[t] = f
                    inv_f_hat[t] = float(1.0 / f) if f > 0 else float("nan")

            if bool(args.write_by_unit) and fail is None and alpha.size:
                alpha_full = np.full(n_strata0, np.nan, dtype=np.float64)
                alpha_full[strata_levels.astype(int, copy=False)] = alpha.astype(np.float64, copy=False)
                in_fit = np.isfinite(alpha_full)

                mi = pd.MultiIndex.from_arrays(
                    [
                        dataset[m].astype(str, copy=False),
                        roi[m].astype(str, copy=False),
                        ccf_adjusted[m].astype(str, copy=False),
                    ],
                    names=["dataset", "roi", "ccf_adjusted"],
                )
                unit_code, unit_levels = pd.factorize(mi, sort=True)
                if int(np.max(unit_code)) >= 2**32:
                    raise ValueError("Too many units for bitpacking (unexpected).")
                if int(n_strata0) >= 2**32:
                    raise ValueError("Too many strata for bitpacking (unexpected).")

                pair = (unit_code.astype(np.uint64) << np.uint64(32)) | stratum_idx.astype(np.uint64, copy=False)
                uniq_pair, counts = np.unique(pair, return_counts=True)
                unit_of_pair = (uniq_pair >> np.uint64(32)).astype(np.int64, copy=False)
                stratum_of_pair = (uniq_pair & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
                keep_pair = in_fit[stratum_of_pair]
                if np.any(keep_pair):
                    unit_k = unit_of_pair[keep_pair]
                    w_pair = counts[keep_pair].astype(np.float64, copy=False)
                    s_pair = stratum_of_pair[keep_pair]
                    # uniq_pair is sorted, so unit_k is already grouped.
                    idx0 = np.flatnonzero(np.r_[True, unit_k[1:] != unit_k[:-1]])
                    unit_ids = unit_k[idx0]
                    w_sum_u = np.add.reduceat(w_pair, idx0)
                    if not bool(np.all(w_sum_u > 0)):
                        raise ValueError("Found unit with zero weight (unexpected).")

                    ds_u = unit_levels.get_level_values(0).to_numpy(dtype=object, copy=False)[unit_ids]
                    roi_u = unit_levels.get_level_values(1).to_numpy(dtype=object, copy=False)[unit_ids]
                    ccf_u = unit_levels.get_level_values(2).to_numpy(dtype=object, copy=False)[unit_ids]

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
                            by_unit_rows.append(
                                {
                                    "animal": str(animal),
                                    "leiden": str(lei),
                                    "dataset": str(ds_u[j]),
                                    "roi": str(roi_u[j]),
                                    "ccf_adjusted": str(ccf_u[j]),
                                    "unit_weight_b1": float(w_sum_u[j]),
                                    "usage_bin": int(t),
                                    "f_hat": float(f_u[j]),
                                    "inv_f_hat": float(inv_f_u[j]),
                                    "ts_over_dt": float(ts_over_dt_u[j]),
                                }
                            )

            for t in range(usage_bins):
                out_rows.append(
                    {
                        "animal": str(animal),
                        "leiden": str(lei),
                        "usage_bin": int(t),
                        "f_hat": float(f_hat[t]),
                        "inv_f_hat": float(inv_f_hat[t]),
                        "raw_f": float(raw_f),
                        "n_cells_b1": int(n_b1),
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
                }
            )

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(outdir / "usage6_meta.csv", index=False)
    log(f"Wrote {outdir/'usage6_meta.csv'} rows={int(meta.shape[0])}")

    log("Done.")


if __name__ == "__main__":
    main()
