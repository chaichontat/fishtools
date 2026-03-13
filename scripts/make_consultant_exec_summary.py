#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _dataset_animal(dataset: str) -> str:
    import re

    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r} (expected contains 'JaxA#').")
    s = m.group(1)
    return s[0].upper() + s[1:]


@dataclass(frozen=True)
class MhDsContrib:
    """MH log-OR decomposed into per-dataset contributions.

    For a given set of strata, MH uses:
      delta = log(sum_s r_s / sum_s s_s)
    where r_s = (a_s d_s)/n_s and s_s = (b_s c_s)/n_s in each stratum.

    We aggregate r_s and s_s to datasets to enable dataset bootstrap with shared
    resampling (preserving covariance between multiple endpoints).
    """

    delta: float
    R_by_ds: np.ndarray
    S_by_ds: np.ndarray


@dataclass(frozen=True)
class RdDsContrib:
    """Risk-difference style contrasts pooled by dataset.

    Define (B=brdu_pos, E=edu_pos, G=gate):
      rd_B1 = P(E=1|G=1,B=1) - P(E=1|G=0,B=1)
      rd_B0 = P(E=1|G=1,B=0) - P(E=1|G=0,B=0)
      rd_int = rd_B1 - rd_B0

    Pooling is trial-weighted within each dataset; bootstrap resamples datasets.
    """

    rd_B1: float
    rd_B0: float
    rd_int: float
    num_B1_by_ds: np.ndarray
    den_B1_by_ds: np.ndarray
    num_B0_by_ds: np.ndarray
    den_B0_by_ds: np.ndarray


def _load_tricycle_ref(path: pathlib.Path) -> pd.DataFrame:
    trc = pd.read_csv(path)
    required = {"symbol", "pc1.rot", "pc2.rot"}
    missing = required - set(trc.columns)
    if missing:
        raise ValueError(f"tricycle ref CSV missing columns: {sorted(missing)}")
    trc = trc.loc[:, ["symbol", "pc1.rot", "pc2.rot"]].copy()
    trc["symbol"] = trc["symbol"].astype(str)
    return trc


def _compute_tricycle_theta(adata, trc: pd.DataFrame, dataset: np.ndarray) -> np.ndarray:
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

    ds = np.asarray(dataset, dtype=object)
    theta = np.empty(ds.shape[0], dtype=np.float32)
    for d in np.unique(ds):
        m = ds == d
        Xd = adata.X[m][:, ref_mask]
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
        theta[m] = np.arctan2(pc2, pc1).astype(np.float32, copy=False)
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
    *,
    n_bins: int,
) -> np.ndarray:
    n_bins = int(n_bins)
    out = np.empty(theta.shape[0], dtype=np.int16)
    for (ds, lei), edges in edges_by_group.items():
        m = (dataset_code == ds) & (leiden_code == lei)
        if not np.any(m):
            continue
        out[m] = np.digitize(theta[m].astype(float, copy=False), edges[1:-1], right=False).astype(np.int16)
    if np.any(out < 0):
        raise RuntimeError("Some theta bins were not assigned (missing dataset×leiden edges).")
    if np.max(out) >= n_bins:
        raise RuntimeError("Theta bin assignment out of range.")
    return out


def _theta_bins(
    theta: np.ndarray,
    dataset_code: np.ndarray,
    leiden_code: np.ndarray,
    *,
    n_bins: int,
    mode: str,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "quantile":
        edges = _theta_edges_by_dataset_leiden(theta, dataset_code, leiden_code, n_bins=int(n_bins))
        return _assign_theta_bins_from_edges(theta, dataset_code, leiden_code, edges, n_bins=int(n_bins))
    if mode == "angle":
        # Fixed absolute angular bins: ensure the same geometric phase intervals across datasets.
        # Assumes theta lies on a shared (-pi, pi] circle.
        t = (theta.astype(np.float64, copy=False) + (2.0 * np.pi)) % (2.0 * np.pi)
        w = (2.0 * np.pi) / float(n_bins)
        out = np.floor(t / w).astype(np.int16, copy=False)
        return np.clip(out, 0, int(n_bins) - 1).astype(np.int16, copy=False)
    raise ValueError(f"Unknown theta bin mode={mode!r} (expected 'quantile' or 'angle').")


def _raw_counts_for_genes(adata, genes: list[str]) -> np.ndarray:
    var = np.asarray(adata.var_names, dtype=object)
    gene_to_col = {g: i for i, g in enumerate(var.tolist())}
    cols = []
    for g in genes:
        if g not in gene_to_col:
            raise ValueError(f"Gene not found in adata.var_names: {g}")
        cols.append(int(gene_to_col[g]))
    cols = np.asarray(cols, dtype=int)

    if "raw" in adata.layers:
        X = adata.layers["raw"][:, cols]
    else:
        X = adata.X[:, cols]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _pearson_residuals_by_dataset(
    X_raw: np.ndarray,
    total_counts: np.ndarray,
    dataset_codes: np.ndarray,
    n_datasets: int,
    *,
    theta: float,
    clip: float | None,
    block_size: int,
) -> np.ndarray:
    if not (float(theta) > 0.0):
        raise ValueError(f"pearson theta must be > 0, got {theta!r}")
    X_raw = np.asarray(X_raw, dtype=np.float32)
    total_counts = np.asarray(total_counts, dtype=np.float32)
    dataset_codes = np.asarray(dataset_codes, dtype=int)
    if X_raw.ndim != 2:
        raise ValueError("X_raw must be 2D")
    if total_counts.ndim != 1 or total_counts.shape[0] != X_raw.shape[0]:
        raise ValueError("total_counts must be 1D with length n_cells")
    if dataset_codes.ndim != 1 or dataset_codes.shape[0] != X_raw.shape[0]:
        raise ValueError("dataset_codes must be 1D with length n_cells")

    n_genes = int(X_raw.shape[1])
    out = np.empty((X_raw.shape[0], n_genes), dtype=np.float32)

    for d in range(int(n_datasets)):
        sel = dataset_codes == d
        if not bool(np.any(sel)):
            continue

        tt = total_counts[sel].astype(np.float32, copy=False)
        denom = float(tt.sum(dtype=np.float64))
        if not (denom > 0.0):
            out[sel, :] = 0.0
            continue

        Xd = X_raw[sel, :]
        sum_x = Xd.sum(axis=0, dtype=np.float64)
        p = (sum_x / denom).astype(np.float32, copy=False)

        for start in range(0, n_genes, int(block_size)):
            end = min(start + int(block_size), n_genes)
            pp = p[start:end].astype(np.float32, copy=False)
            mu = tt[:, None] * pp[None, :]
            var = mu + (mu * mu) / float(theta)
            var = np.maximum(var, 1e-12, dtype=np.float32)
            r = (Xd[:, start:end] - mu) / np.sqrt(var)
            if clip is not None:
                r = np.clip(r, -float(clip), float(clip))
            out[sel, start:end] = r.astype(np.float32, copy=False)

    return out


def _residualize_matrix_within_dataset(
    X: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray
) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    R = np.empty_like(X, dtype=np.float32)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m), dtype=np.float32), sin_t[m], cos_t[m]]).astype(np.float32, copy=False)
        coef, *_ = np.linalg.lstsq(Z, X[m], rcond=None)  # (3, n_genes)
        R[m] = X[m] - (Z @ coef).astype(np.float32, copy=False)
    return R


def _stable_hash_u64(dataset: np.ndarray, obs_name: np.ndarray) -> np.ndarray:
    ds = dataset.astype(str)
    on = obs_name.astype(str)
    # Include a stable position to avoid collisions when obs_names are not unique (can happen in the AnnData).
    s = pd.Series(
        [f"{d}|{o}|{i}" for i, (d, o) in enumerate(zip(ds.tolist(), on.tolist(), strict=True))],
        copy=False,
    )
    return pd.util.hash_pandas_object(s, index=False).to_numpy(np.uint64, copy=False)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return out
    pv = np.clip(p[m], 0.0, 1.0)
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    q = ranked * n / (np.arange(1, n + 1, dtype=float))
    # monotone
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[m] = tmp
    return out


def _z_to_p_two_sided(z: float) -> float:
    if not math.isfinite(z):
        return float("nan")
    # Two-sided p-value for standard normal using erfc for numerical stability.
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _mh_log_or(
    *,
    stratum_idx: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    n_strata: int,
    cc: float,
) -> float:
    g = G.astype(float, copy=False)
    y1 = y.astype(float, copy=False)
    y0 = 1.0 - y1
    a = np.bincount(stratum_idx, weights=g * y1, minlength=int(n_strata)).astype(np.float64, copy=False) + float(cc)
    b = np.bincount(stratum_idx, weights=g * y0, minlength=int(n_strata)).astype(np.float64, copy=False) + float(cc)
    c = (
        np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=int(n_strata)).astype(np.float64, copy=False)
        + float(cc)
    )
    d = (
        np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=int(n_strata)).astype(np.float64, copy=False)
        + float(cc)
    )
    n = a + b + c + d
    r = (a * d) / n
    s = (b * c) / n
    R = float(r.sum())
    S = float(s.sum())
    return float(math.log(R / S)) if (R > 0.0 and S > 0.0) else float("nan")


def _make_gates_by_dataset_quantile_exact(
    R: np.ndarray,
    dataset: np.ndarray,
    q_by_gene: np.ndarray,
    *,
    jitter: np.ndarray,
    direction: str,
) -> np.ndarray:
    direction = str(direction).strip().lower()
    if direction not in {"high", "low"}:
        raise ValueError(f"Unknown gate direction={direction!r} (expected 'high' or 'low').")
    ds = np.asarray(dataset, dtype=object)
    G = np.zeros(R.shape, dtype=bool)
    uniq_q = np.unique(q_by_gene)
    for d in np.unique(ds):
        m = ds == d
        rows = np.where(m)[0]
        Rd = R[m]
        jd = jitter[m].astype(np.float64, copy=False)
        eps = 1e-9
        for q in uniq_q:
            j = np.where(q_by_gene == q)[0]
            if j.size == 0:
                continue

            n = int(Rd.shape[0])
            k = int(math.ceil((1.0 - float(q)) * n))
            if k <= 0:
                continue
            if k >= n:
                G[np.ix_(rows, j)] = True
                continue

            # Exact-size top-k gating with deterministic tie-breaking via stable jitter.
            # This avoids prevalence explosions in zero-inflated panels when many values tie at the quantile.
            for col in j.tolist():
                vals = Rd[:, int(col)].astype(np.float64, copy=False) + eps * jd
                if direction == "high":
                    # Select the k largest values.
                    cut = np.argpartition(vals, n - k)[n - k :]
                else:
                    # Select the k smallest values.
                    cut = np.argpartition(vals, k - 1)[:k]
                G[rows[cut], int(col)] = True
    return G


@dataclass(frozen=True)
class GlmCoef:
    beta: np.ndarray  # (p,)
    cov: np.ndarray  # (p,p)
    fail_reason: str | None
    n_rows: int
    n_strata: int


def _fit_stratified_glm_binomial(
    *,
    y_succ: np.ndarray,
    y_tot: np.ndarray,
    strata: np.ndarray,
    X: np.ndarray,
) -> GlmCoef:
    """Fit binomial GLM with stratum fixed effects on aggregated rows.

    Model: logit(p) = alpha_stratum + X * beta.
    """
    import statsmodels.api as sm

    y_succ = np.asarray(y_succ, dtype=float)
    y_tot = np.asarray(y_tot, dtype=float)
    strata = np.asarray(strata, dtype=int)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if not (y_succ.shape == y_tot.shape == (X.shape[0],) and strata.shape == (X.shape[0],)):
        raise ValueError("Bad shapes for GLM inputs.")

    keep = y_tot > 0
    if not np.any(keep):
        p = int(X.shape[1])
        return GlmCoef(
            beta=np.full(p, np.nan),
            cov=np.full((p, p), np.nan),
            fail_reason="no_rows",
            n_rows=0,
            n_strata=0,
        )

    y_succ = y_succ[keep]
    y_tot = y_tot[keep]
    strata = strata[keep]
    X = X[keep]

    # Drop strata with no variation in outcome across all rows.
    n_strata0 = int(strata.max()) + 1
    tot_by_s = np.bincount(strata, weights=y_tot, minlength=n_strata0)
    succ_by_s = np.bincount(strata, weights=y_succ, minlength=n_strata0)
    keep_s = (succ_by_s > 0) & (succ_by_s < tot_by_s)
    keep2 = keep_s[strata]
    if not np.any(keep2):
        p = int(X.shape[1])
        return GlmCoef(
            beta=np.full(p, np.nan),
            cov=np.full((p, p), np.nan),
            fail_reason="no_variation",
            n_rows=int(y_tot.size),
            n_strata=int(np.unique(strata).size),
        )

    y_succ = y_succ[keep2]
    y_tot = y_tot[keep2]
    strata = strata[keep2]
    X = X[keep2]

    # Build one-hot strata dummies (no intercept); switch to sparse if large.
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
        return GlmCoef(
            beta=np.full(p, np.nan),
            cov=np.full((p, p), np.nan),
            fail_reason=f"{type(e).__name__}: {e}",
            n_rows=int(y_tot.size),
            n_strata=int(n_s),
        )

    params = np.asarray(res.params, dtype=float)[-p:]
    covp = res.cov_params()
    if exog_is_sparse and hasattr(covp, "toarray"):
        covp = covp.toarray()
    cov = np.asarray(covp, dtype=float)[-p:, -p:]
    return GlmCoef(beta=params, cov=cov, fail_reason=None, n_rows=int(y_tot.size), n_strata=int(n_s))


def _mh_ds_contrib(
    *,
    stratum_idx: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    n_strata: int,
    n_datasets: int,
    n_leiden: int,
    theta_bins: int,
    cc: float,
    keep_strata: np.ndarray | None = None,
) -> MhDsContrib:
    g = G.astype(float, copy=False)
    y1 = y.astype(float, copy=False)
    y0 = 1.0 - y1
    a0 = np.bincount(stratum_idx, weights=g * y1, minlength=int(n_strata)).astype(np.float64, copy=False)
    b0 = np.bincount(stratum_idx, weights=g * y0, minlength=int(n_strata)).astype(np.float64, copy=False)
    c0 = np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=int(n_strata)).astype(np.float64, copy=False)
    d0 = np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=int(n_strata)).astype(np.float64, copy=False)
    n0 = a0 + b0 + c0 + d0
    keep = n0 > 0
    if keep_strata is not None:
        keep_strata = np.asarray(keep_strata, dtype=bool)
        if keep_strata.shape != (int(n_strata),):
            raise ValueError(f"keep_strata must have shape (n_strata,), got {keep_strata.shape}")
        keep = keep & keep_strata
    if not np.any(keep):
        return MhDsContrib(delta=float("nan"), R_by_ds=np.zeros(int(n_datasets)), S_by_ds=np.zeros(int(n_datasets)))

    a = a0[keep] + float(cc)
    b = b0[keep] + float(cc)
    c = c0[keep] + float(cc)
    d = d0[keep] + float(cc)
    n = a + b + c + d
    r = (a * d) / n
    s = (b * c) / n
    R_total = float(r.sum())
    S_total = float(s.sum())
    if not (R_total > 0.0 and S_total > 0.0):
        return MhDsContrib(delta=float("nan"), R_by_ds=np.zeros(int(n_datasets)), S_by_ds=np.zeros(int(n_datasets)))

    delta = float(math.log(R_total / S_total))
    stratum_ids = np.arange(int(n_strata), dtype=int)[keep]
    ds_code = (stratum_ids // (int(theta_bins) * int(n_leiden))).astype(int, copy=False)
    R_by_ds = np.bincount(ds_code, weights=r, minlength=int(n_datasets)).astype(np.float64, copy=False)
    S_by_ds = np.bincount(ds_code, weights=s, minlength=int(n_datasets)).astype(np.float64, copy=False)
    return MhDsContrib(delta=delta, R_by_ds=R_by_ds, S_by_ds=S_by_ds)


def _bootstrap_ci_for_delta_diff(
    c1: MhDsContrib,
    c2: MhDsContrib,
    *,
    rng: np.random.Generator,
    n_boot: int,
    group_of_ds: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if group_of_ds is not None:
        group_of_ds = np.asarray(group_of_ds, dtype=int)
        if group_of_ds.shape != c1.R_by_ds.shape:
            raise ValueError("group_of_ds must have shape (n_datasets,).")
        n_groups = int(group_of_ds.max()) + 1
        R1 = np.bincount(group_of_ds, weights=c1.R_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        S1 = np.bincount(group_of_ds, weights=c1.S_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        R2 = np.bincount(group_of_ds, weights=c2.R_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        S2 = np.bincount(group_of_ds, weights=c2.S_by_ds, minlength=n_groups).astype(np.float64, copy=False)
    else:
        R1, S1, R2, S2 = c1.R_by_ds, c1.S_by_ds, c2.R_by_ds, c2.S_by_ds

    present = (R1 + S1 + R2 + S2) > 0
    groups_present = np.flatnonzero(present)
    if groups_present.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    d_point = float(c1.delta - c2.delta)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(groups_present, size=groups_present.size, replace=True)
        d1 = float(math.log(float(R1[samp].sum()) / float(S1[samp].sum())))
        d2 = float(math.log(float(R2[samp].sum()) / float(S2[samp].sum())))
        boots[i] = d1 - d2
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return (d_point, float(lo), float(hi))


def _bootstrap_ci_from_contrib(
    c: MhDsContrib,
    *,
    rng: np.random.Generator,
    n_boot: int,
    group_of_ds: np.ndarray | None = None,
) -> tuple[float, float]:
    if group_of_ds is not None:
        group_of_ds = np.asarray(group_of_ds, dtype=int)
        if group_of_ds.shape != c.R_by_ds.shape:
            raise ValueError("group_of_ds must have shape (n_datasets,).")
        n_groups = int(group_of_ds.max()) + 1
        R = np.bincount(group_of_ds, weights=c.R_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        S = np.bincount(group_of_ds, weights=c.S_by_ds, minlength=n_groups).astype(np.float64, copy=False)
    else:
        R, S = c.R_by_ds, c.S_by_ds

    present = (R + S) > 0
    groups_present = np.flatnonzero(present)
    if groups_present.size == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(groups_present, size=groups_present.size, replace=True)
        Rb = float(R[samp].sum())
        Sb = float(S[samp].sum())
        boots[i] = float(math.log(Rb / Sb))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return (float(lo), float(hi))


def _bootstrap_ci_for_int_diff(
    c1: MhDsContrib,
    c2: MhDsContrib,
    c3: MhDsContrib,
    c4: MhDsContrib,
    *,
    rng: np.random.Generator,
    n_boot: int,
    group_of_ds: np.ndarray | None = None,
) -> tuple[float, float, float]:
    # int_diff := (d1 - d3) - (d2 - d4) = d1 - d3 - d2 + d4
    if group_of_ds is not None:
        group_of_ds = np.asarray(group_of_ds, dtype=int)
        if group_of_ds.shape != c1.R_by_ds.shape:
            raise ValueError("group_of_ds must have shape (n_datasets,).")
        n_groups = int(group_of_ds.max()) + 1
        def _agg(x: np.ndarray) -> np.ndarray:
            return np.bincount(group_of_ds, weights=x, minlength=n_groups).astype(np.float64, copy=False)
        R1, S1, R2, S2, R3, S3, R4, S4 = map(_agg, [c1.R_by_ds, c1.S_by_ds, c2.R_by_ds, c2.S_by_ds, c3.R_by_ds, c3.S_by_ds, c4.R_by_ds, c4.S_by_ds])
    else:
        R1, S1, R2, S2, R3, S3, R4, S4 = c1.R_by_ds, c1.S_by_ds, c2.R_by_ds, c2.S_by_ds, c3.R_by_ds, c3.S_by_ds, c4.R_by_ds, c4.S_by_ds

    present = (R1 + S1 + R2 + S2 + R3 + S3 + R4 + S4) > 0
    groups_present = np.flatnonzero(present)
    if groups_present.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    d_point = float((c1.delta - c3.delta) - (c2.delta - c4.delta))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(groups_present, size=groups_present.size, replace=True)
        d1 = float(math.log(float(R1[samp].sum()) / float(S1[samp].sum())))
        d2 = float(math.log(float(R2[samp].sum()) / float(S2[samp].sum())))
        d3 = float(math.log(float(R3[samp].sum()) / float(S3[samp].sum())))
        d4 = float(math.log(float(R4[samp].sum()) / float(S4[samp].sum())))
        boots[i] = (d1 - d3) - (d2 - d4)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return (d_point, float(lo), float(hi))


def _rd_ds_contrib(
    *,
    stratum_idx: np.ndarray,
    ds_of_stratum: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    E: np.ndarray,
    n_strata: int,
    n_datasets: int,
    cc: float,
) -> RdDsContrib:
    sidx = np.asarray(stratum_idx, dtype=int)
    g = np.asarray(G, dtype=np.int8, order="C")
    b = np.asarray(B, dtype=np.int8, order="C")
    e = np.asarray(E, dtype=np.int8, order="C")
    if not (sidx.shape == g.shape == b.shape == e.shape):
        raise ValueError("Bad shapes for RD inputs.")

    if sidx.size == 0:
        z = np.zeros(int(n_datasets), dtype=np.float64)
        return RdDsContrib(
            rd_B1=float("nan"),
            rd_B0=float("nan"),
            rd_int=float("nan"),
            num_B1_by_ds=z,
            den_B1_by_ds=z,
            num_B0_by_ds=z,
            den_B0_by_ds=z,
        )

    # Count totals and successes for E within each (stratum, B, G).
    group = sidx * 4 + b * 2 + g  # (stratum, B, G)
    tot = np.bincount(group, minlength=int(n_strata) * 4).astype(np.float64, copy=False)
    succ = np.bincount(group, weights=e.astype(np.float64, copy=False), minlength=int(n_strata) * 4).astype(
        np.float64, copy=False
    )
    tot = tot.reshape(int(n_strata), 2, 2)
    succ = succ.reshape(int(n_strata), 2, 2)

    # Smoothed within-stratum probabilities; strata without within-stratum gate contrast get weight 0.
    p = (succ + float(cc)) / (tot + 2.0 * float(cc))

    tot_B1_G0 = tot[:, 1, 0]
    tot_B1_G1 = tot[:, 1, 1]
    tot_B0_G0 = tot[:, 0, 0]
    tot_B0_G1 = tot[:, 0, 1]

    ok_B1 = (tot_B1_G0 > 0) & (tot_B1_G1 > 0)
    ok_B0 = (tot_B0_G0 > 0) & (tot_B0_G1 > 0)

    rd_B1_s = (p[:, 1, 1] - p[:, 1, 0]).astype(np.float64, copy=False)
    rd_B0_s = (p[:, 0, 1] - p[:, 0, 0]).astype(np.float64, copy=False)
    w_B1_s = (tot_B1_G0 + tot_B1_G1).astype(np.float64, copy=False) * ok_B1.astype(np.float64, copy=False)
    w_B0_s = (tot_B0_G0 + tot_B0_G1).astype(np.float64, copy=False) * ok_B0.astype(np.float64, copy=False)

    num_B1_s = w_B1_s * rd_B1_s
    num_B0_s = w_B0_s * rd_B0_s

    num_B1_by_ds = np.bincount(ds_of_stratum, weights=num_B1_s, minlength=int(n_datasets)).astype(np.float64, copy=False)
    den_B1_by_ds = np.bincount(ds_of_stratum, weights=w_B1_s, minlength=int(n_datasets)).astype(np.float64, copy=False)
    num_B0_by_ds = np.bincount(ds_of_stratum, weights=num_B0_s, minlength=int(n_datasets)).astype(np.float64, copy=False)
    den_B0_by_ds = np.bincount(ds_of_stratum, weights=w_B0_s, minlength=int(n_datasets)).astype(np.float64, copy=False)

    def _safe_ratio(num: np.ndarray, den: np.ndarray) -> float:
        dn = float(den.sum())
        return float(num.sum() / dn) if dn > 0.0 else float("nan")

    rd_B1 = _safe_ratio(num_B1_by_ds, den_B1_by_ds)
    rd_B0 = _safe_ratio(num_B0_by_ds, den_B0_by_ds)
    rd_int = float(rd_B1 - rd_B0) if (np.isfinite(rd_B1) and np.isfinite(rd_B0)) else float("nan")
    return RdDsContrib(
        rd_B1=rd_B1,
        rd_B0=rd_B0,
        rd_int=rd_int,
        num_B1_by_ds=num_B1_by_ds,
        den_B1_by_ds=den_B1_by_ds,
        num_B0_by_ds=num_B0_by_ds,
        den_B0_by_ds=den_B0_by_ds,
    )


def _bootstrap_ci_for_rd(
    c: RdDsContrib,
    *,
    rng: np.random.Generator,
    n_boot: int,
    group_of_ds: np.ndarray | None = None,
) -> dict[str, tuple[float, float, float]]:
    if group_of_ds is not None:
        group_of_ds = np.asarray(group_of_ds, dtype=int)
        if group_of_ds.shape != c.den_B1_by_ds.shape:
            raise ValueError("group_of_ds must have shape (n_datasets,).")
        n_groups = int(group_of_ds.max()) + 1
        num_B1 = np.bincount(group_of_ds, weights=c.num_B1_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        den_B1 = np.bincount(group_of_ds, weights=c.den_B1_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        num_B0 = np.bincount(group_of_ds, weights=c.num_B0_by_ds, minlength=n_groups).astype(np.float64, copy=False)
        den_B0 = np.bincount(group_of_ds, weights=c.den_B0_by_ds, minlength=n_groups).astype(np.float64, copy=False)
    else:
        num_B1, den_B1, num_B0, den_B0 = c.num_B1_by_ds, c.den_B1_by_ds, c.num_B0_by_ds, c.den_B0_by_ds

    present = (den_B1 + den_B0) > 0
    groups_present = np.flatnonzero(present)
    if groups_present.size == 0:
        return {
            "rd_B1": (float("nan"), float("nan"), float("nan")),
            "rd_B0": (float("nan"), float("nan"), float("nan")),
            "rd_int": (float("nan"), float("nan"), float("nan")),
        }

    def _ratio(num: np.ndarray, den: np.ndarray, samp: np.ndarray) -> float:
        dn = float(den[samp].sum())
        return float(num[samp].sum() / dn) if dn > 0.0 else float("nan")

    boots_B1 = np.empty(int(n_boot), dtype=np.float64)
    boots_B0 = np.empty(int(n_boot), dtype=np.float64)
    boots_int = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(groups_present, size=groups_present.size, replace=True)
        b1 = _ratio(num_B1, den_B1, samp)
        b0 = _ratio(num_B0, den_B0, samp)
        boots_B1[i] = b1
        boots_B0[i] = b0
        boots_int[i] = b1 - b0

    lo1, hi1 = np.nanquantile(boots_B1, [0.025, 0.975])
    lo0, hi0 = np.nanquantile(boots_B0, [0.025, 0.975])
    loi, hii = np.nanquantile(boots_int, [0.025, 0.975])
    return {
        "rd_B1": (float(c.rd_B1), float(lo1), float(hi1)),
        "rd_B0": (float(c.rd_B0), float(lo0), float(hi0)),
        "rd_int": (float(c.rd_int), float(loi), float(hii)),
    }


def _sparse_strata_fraction(
    *, stratum_idx: np.ndarray, G: np.ndarray, y: np.ndarray, n_strata: int, keep_strata: np.ndarray | None = None
) -> float:
    # Fraction of non-empty strata where any of (a,b,c,d) is exactly 0 before continuity correction.
    g = G.astype(float, copy=False)
    y1 = y.astype(float, copy=False)
    y0 = 1.0 - y1
    a0 = np.bincount(stratum_idx, weights=g * y1, minlength=int(n_strata)).astype(np.float64, copy=False)
    b0 = np.bincount(stratum_idx, weights=g * y0, minlength=int(n_strata)).astype(np.float64, copy=False)
    c0 = np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=int(n_strata)).astype(np.float64, copy=False)
    d0 = np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=int(n_strata)).astype(np.float64, copy=False)
    n0 = a0 + b0 + c0 + d0
    keep = n0 > 0
    if keep_strata is not None:
        keep_strata = np.asarray(keep_strata, dtype=bool)
        if keep_strata.shape != (int(n_strata),):
            raise ValueError(f"keep_strata must have shape (n_strata,), got {keep_strata.shape}")
        keep = keep & keep_strata
    if not np.any(keep):
        return float("nan")
    sparse = (a0 == 0) | (b0 == 0) | (c0 == 0) | (d0 == 0)
    return float(np.mean(sparse[keep]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Create consultant-facing executive summary table for quadrant validation.")
    ap.add_argument(
        "--h5ad",
        type=pathlib.Path,
        default=pathlib.Path("~/nvme/all_progenitors.h5ad"),
    )
    ap.add_argument(
        "--quadrant-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2/quadrant_validation.csv"
        ),
    )
    ap.add_argument(
        "--scorecard-csv",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a_phase_matched/gene_validation_scorecard_minimal_no_cnksr2.csv"),
    )
    ap.add_argument(
        "--crossfit-bestq-by-gene-csv",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional CSV produced by `scripts/brdu_regression/crossfit_bestq_intE_mh.py` "
            "(the *_by_gene.csv output). If provided, merge cross-fit selection diagnostics by gene."
        ),
    )
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument(
        "--theta-exclude-genes",
        type=str,
        nargs="+",
        default=[],
        help="Exclude these genes from the tricycle reference set when computing theta (theta-leakage sensitivity).",
    )
    ap.add_argument(
        "--theta-exclude-tested-genes",
        action="store_true",
        help="Exclude all tested gate genes (from --quadrant-csv) from the tricycle reference when computing theta.",
    )
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument(
        "--theta-bin-mode",
        type=str,
        choices=["quantile", "angle"],
        default="quantile",
        help="Theta binning: quantile within (dataset×leiden) vs fixed absolute angular bins.",
    )
    ap.add_argument(
        "--no-theta-matching",
        dest="theta_matching",
        action="store_false",
        help="Ablation: disable theta-bin matching (strata become dataset×leiden only).",
    )
    ap.set_defaults(theta_matching=True)
    ap.add_argument("--n-bootstrap", type=int, default=300)
    ap.add_argument(
        "--bootstrap-unit",
        type=str,
        choices=["dataset", "animal"],
        default="dataset",
        help="Resampling unit for MH/RD uncertainty: dataset bootstrap (default) vs animal bootstrap (more conservative if datasets nest within animals).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cc", type=float, default=0.5)
    ap.add_argument("--pearson-theta", type=float, default=100.0, help="Pearson residual theta (NB overdispersion).")
    ap.add_argument(
        "--pearson-clip",
        type=str,
        default="10.0",
        help="Clip Pearson residuals to +/-clip; use 'none' to disable clipping.",
    )
    ap.add_argument("--pearson-block-size", type=int, default=64)
    ap.add_argument(
        "--no-gate-theta-residualize",
        dest="gate_theta_residualize",
        action="store_false",
        help="Ablation: do not residualize gate scores against sin/cos(theta) within dataset before thresholding.",
    )
    ap.set_defaults(gate_theta_residualize=True)
    ap.add_argument(
        "--gate-direction",
        type=str,
        choices=["high", "low"],
        default="high",
        help="Define gates as top-k ('high', default) or bottom-k ('low') within each dataset by the gate score.",
    )
    ap.add_argument(
        "--glm-mh-divergence-abs",
        type=float,
        default=0.2,
        help="Flag GLM-vs-MH(overlap) divergence when |beta_GxB - delta_int_E_overlap| exceeds this threshold.",
    )
    ap.add_argument(
        "--glm-mh-divergence-z",
        type=float,
        default=5.0,
        help=(
            "Flag large GLM-vs-MH(overlap) divergence when |beta_GxB - delta_int_E_overlap| / "
            "sqrt(se_glm^2 + se_mh^2) exceeds this threshold. (Uses mh SE approximated from its CI.)"
        ),
    )
    ap.add_argument(
        "--glm-min-usable-strata",
        type=int,
        default=50,
        help="Hard-flag genes when the GLM interaction is supported by fewer than this many usable strata.",
    )
    ap.add_argument(
        "--no-support-filter",
        dest="support_filter",
        action="store_false",
        help=(
            "Disable gene×dataset support filtering. By default, for each gene we exclude datasets where the "
            "gate selects zero detected (raw>0) gate+ cells, to avoid manufacturing gate+ cells for all-zero genes."
        ),
    )
    ap.set_defaults(support_filter=True)
    ap.add_argument(
        "--support-min-detected",
        type=int,
        default=1,
        help="Per gene×dataset: minimum number of detected (raw>0) cells required to include a dataset.",
    )
    ap.add_argument(
        "--support-min-detected-gatepos",
        type=int,
        default=1,
        help="Per gene×dataset: minimum number of detected (raw>0) cells among gate+ required to include a dataset.",
    )
    ap.add_argument(
        "--no-glm",
        dest="glm",
        action="store_false",
        help="Skip stratum-FE grouped-binomial GLM fit (faster; GLM columns set to NaN).",
    )
    ap.set_defaults(glm=True)
    ap.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a_phase_matched/consultant_exec_summary_no_cnksr2"),
    )
    args = ap.parse_args()

    import anndata as ad

    rng = np.random.default_rng(int(args.seed))
    args.outdir.mkdir(parents=True, exist_ok=True)

    clip_s = str(args.pearson_clip).strip().lower()
    pearson_clip: float | None = None if clip_s in {"none", "nan"} else float(args.pearson_clip)

    qdf = pd.read_csv(args.quadrant_csv)
    if "gene" not in qdf.columns or "q" not in qdf.columns:
        raise ValueError("--quadrant-csv must include columns: gene,q")
    genes = qdf["gene"].astype(str).tolist()
    q_by_gene = qdf["q"].astype(float).to_numpy()

    # Load AnnData and core covariates.
    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    dataset = obs["dataset"].astype(str).to_numpy().astype(object)
    obs_name = adata.obs_names.to_numpy(dtype=object, copy=False)
    leiden = obs["leiden"].astype(str).to_numpy().astype(object)
    brdu = obs["brdu_pos"].to_numpy().astype(int)
    edu = obs["edu_pos"].to_numpy().astype(int)
    if "total_counts" not in obs.columns:
        raise ValueError("adata.obs must contain 'total_counts' for Pearson residual gating.")
    total_counts = obs["total_counts"].to_numpy()

    ds_code, ds_uniq = pd.factorize(dataset, sort=True)
    lei_code, lei_uniq = pd.factorize(leiden, sort=True)
    n_datasets = int(ds_uniq.shape[0])
    n_leiden = int(lei_uniq.shape[0])
    theta_bins_used = int(args.theta_bins) if bool(args.theta_matching) else 1
    n_strata = n_datasets * int(theta_bins_used) * n_leiden

    need_theta = bool(args.theta_matching) or bool(args.gate_theta_residualize)
    if need_theta:
        trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
        exclude = {str(x) for x in (args.theta_exclude_genes or []) if str(x).strip()}
        if bool(args.theta_exclude_tested_genes):
            exclude |= {str(g) for g in genes}
        if exclude:
            trc = trc.loc[~trc["symbol"].isin(sorted(exclude))].copy()
        theta = _compute_tricycle_theta(adata, trc=trc, dataset=dataset)
    else:
        theta = np.zeros(dataset.shape[0], dtype=np.float32)

    if bool(args.theta_matching):
        theta_bin = _theta_bins(
            theta,
            ds_code,
            lei_code,
            n_bins=int(args.theta_bins),
            mode=str(args.theta_bin_mode),
        )
    else:
        theta_bin = np.zeros(dataset.shape[0], dtype=np.int16)

    stratum_idx_all = (ds_code * int(theta_bins_used) + theta_bin.astype(int)) * n_leiden + lei_code
    ds_of_stratum = (np.arange(int(n_strata), dtype=int) // (int(theta_bins_used) * int(n_leiden))).astype(int, copy=False)

    animal_by_ds = np.array([_dataset_animal(d) for d in ds_uniq.tolist()], dtype=object)
    animal_code_by_ds, animal_uniq = pd.factorize(animal_by_ds, sort=True)
    if str(args.bootstrap_unit) == "animal":
        bootstrap_group_of_ds = animal_code_by_ds.astype(np.int32, copy=False)
        bootstrap_unit = "animal"
    else:
        bootstrap_group_of_ds = None
        bootstrap_unit = "dataset"

    # Overlap support mask for interaction (shared strata for B=1 and B=0).
    present_B1 = np.bincount(stratum_idx_all[brdu == 1], minlength=int(n_strata)).astype(np.int64, copy=False)
    present_B0 = np.bincount(stratum_idx_all[brdu == 0], minlength=int(n_strata)).astype(np.int64, copy=False)
    keep_overlap = (present_B1 > 0) & (present_B0 > 0)

    # Gates.
    X_raw = _raw_counts_for_genes(adata, genes=genes)
    X = _pearson_residuals_by_dataset(
        X_raw,
        total_counts=total_counts,
        dataset_codes=ds_code,
        n_datasets=n_datasets,
        theta=float(args.pearson_theta),
        clip=pearson_clip,
        block_size=int(args.pearson_block_size),
    )
    if bool(args.gate_theta_residualize):
        sin_t = np.sin(theta).astype(np.float32, copy=False)
        cos_t = np.cos(theta).astype(np.float32, copy=False)
        R = _residualize_matrix_within_dataset(X, sin_t=sin_t, cos_t=cos_t, dataset=dataset)
    else:
        R = X
    h = _stable_hash_u64(dataset, obs_name)
    jitter = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)) - 0.5
    Gmat = _make_gates_by_dataset_quantile_exact(
        R, dataset=dataset, q_by_gene=q_by_gene, jitter=jitter, direction=args.gate_direction
    )

    # Support table (gene × dataset): counts and achieved gate fractions.
    support_rows: list[dict[str, object]] = []
    n_cells_by_ds = np.bincount(ds_code, minlength=int(n_datasets)).astype(np.int64, copy=False)
    n_B1_by_ds = np.bincount(ds_code, weights=brdu.astype(np.int64, copy=False), minlength=int(n_datasets)).astype(
        np.int64, copy=False
    )
    n_E1_by_ds = np.bincount(ds_code, weights=edu.astype(np.int64, copy=False), minlength=int(n_datasets)).astype(
        np.int64, copy=False
    )

    present_any = np.bincount(stratum_idx_all, minlength=int(n_strata)).astype(np.int64, copy=False) > 0
    ds_by_stratum = (np.arange(int(n_strata), dtype=np.int64) // (int(theta_bins_used) * int(n_leiden))).astype(
        np.int32, copy=False
    )
    n_present_strata_by_ds = np.bincount(ds_by_stratum[present_any], minlength=int(n_datasets)).astype(
        np.int64, copy=False
    )
    n_overlap_strata_by_ds = np.bincount(
        ds_by_stratum[present_any & keep_overlap], minlength=int(n_datasets)
    ).astype(np.int64, copy=False)

    # Endpoint definitions.
    endpoints: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("BrdUplus_Edu", brdu == 1, edu),
        ("Eduplus_BrdU", edu == 1, brdu),
        ("BrdUminus_Edu", brdu == 0, edu),
        ("EdUminus_BrdU", edu == 0, brdu),
    ]

    rows: list[dict[str, object]] = []
    hetero_rows: list[dict[str, object]] = []
    for j, g in enumerate(genes):
        G = Gmat[:, j]
        row: dict[str, object] = {"gene": g, "q": float(q_by_gene[j])}
        row["bootstrap_unit"] = bootstrap_unit
        row["theta_matching_enabled"] = bool(args.theta_matching)
        row["theta_bin_mode"] = str(args.theta_bin_mode)
        row["theta_bins_used"] = int(theta_bins_used)
        row["gate_theta_residualize_enabled"] = bool(args.gate_theta_residualize)
        row["gate_direction"] = str(args.gate_direction)
        row["strata_overlap_frac_all"] = float(np.mean(keep_overlap))

        xg = X_raw[:, j]
        detected = (xg > 0).astype(np.int64, copy=False)
        n_detect_by_ds = np.bincount(ds_code, weights=detected, minlength=int(n_datasets)).astype(np.int64, copy=False)
        n_detect_gatepos_by_ds = np.bincount(ds_code, weights=detected * G.astype(np.int64, copy=False), minlength=int(n_datasets)).astype(
            np.int64, copy=False
        )
        if bool(args.support_filter):
            ds_keep = (n_detect_by_ds >= int(args.support_min_detected)) & (
                n_detect_gatepos_by_ds >= int(args.support_min_detected_gatepos)
            )
        else:
            ds_keep = np.ones(int(n_datasets), dtype=bool)
        row["support_filter_enabled"] = bool(args.support_filter)
        row["support_min_detected"] = int(args.support_min_detected)
        row["support_min_detected_gatepos"] = int(args.support_min_detected_gatepos)
        row["support_n_datasets_kept"] = int(np.sum(ds_keep))
        row["support_n_datasets_dropped"] = int(int(n_datasets) - int(np.sum(ds_keep)))
        keep_strata_gene = ds_keep[ds_of_stratum]
        keep_overlap_gene_bonly = keep_overlap & keep_strata_gene
        keep_cells = ds_keep[ds_code]

        # Per-gene overlap mask requiring gate variation within each B level.
        # This is stricter than B-only overlap and prevents E1/E3 being pooled over strata
        # where the gate contrast is absent for B=1 or B=0.
        stratum_keep = stratum_idx_all[keep_cells].astype(np.int64, copy=False)
        B_keep = brdu[keep_cells].astype(np.int8, copy=False)
        G_keep = G[keep_cells].astype(np.int8, copy=False)
        key_gb = stratum_keep * 4 + G_keep.astype(np.int64, copy=False) * 2 + B_keep.astype(np.int64, copy=False)
        n_gb = np.bincount(key_gb, minlength=int(n_strata) * 4).astype(np.int64, copy=False).reshape(int(n_strata), 4)
        # combo index = G*2 + B, so B=1 requires combos 1 and 3; B=0 requires combos 0 and 2.
        keep_overlap_gene_bg = (n_gb[:, 1] > 0) & (n_gb[:, 3] > 0) & (n_gb[:, 0] > 0) & (n_gb[:, 2] > 0)
        keep_overlap_gene = keep_overlap_gene_bg & keep_strata_gene

        present_any_gene = np.bincount(stratum_keep, minlength=int(n_strata)).astype(np.int64, copy=False) > 0
        row["strata_overlap_frac_gene_bonly"] = (
            float(np.mean(keep_overlap_gene_bonly[present_any_gene])) if np.any(present_any_gene) else float("nan")
        )
        row["strata_overlap_frac_gene_bg"] = (
            float(np.mean(keep_overlap_gene_bg[present_any_gene])) if np.any(present_any_gene) else float("nan")
        )
        row["glm_n_strata_total_present"] = int(np.sum(present_any_gene))
        # Stratum-level variation diagnostics (within kept datasets for this gene).
        present_B1_gene = np.bincount(stratum_keep[B_keep == 1], minlength=int(n_strata)).astype(np.int64, copy=False) > 0
        present_B0_gene = np.bincount(stratum_keep[B_keep == 0], minlength=int(n_strata)).astype(np.int64, copy=False) > 0
        B_ok = present_B1_gene & present_B0_gene
        present_G1_gene = (n_gb[:, 2] + n_gb[:, 3]) > 0
        present_G0_gene = (n_gb[:, 0] + n_gb[:, 1]) > 0
        G_ok = present_G1_gene & present_G0_gene
        # For the GLM outcome E=edu.
        E_keep = edu[keep_cells].astype(np.int8, copy=False)
        present_E1_gene = np.bincount(stratum_keep[E_keep == 1], minlength=int(n_strata)).astype(np.int64, copy=False) > 0
        present_E0_gene = np.bincount(stratum_keep[E_keep == 0], minlength=int(n_strata)).astype(np.int64, copy=False) > 0
        E_ok = present_E1_gene & present_E0_gene
        # Interaction identifiability requires some mixed (G,B) support within a stratum.
        mixed_ok = (n_gb[:, 1] > 0) | (n_gb[:, 2] > 0)
        usable_for_beta_gxb = present_any_gene & B_ok & G_ok & E_ok & mixed_ok
        row["glm_n_strata_drop_B_const"] = int(np.sum(present_any_gene & ~B_ok))
        row["glm_n_strata_drop_G_const"] = int(np.sum(present_any_gene & ~G_ok))
        row["glm_n_strata_drop_E_const"] = int(np.sum(present_any_gene & ~E_ok))
        row["glm_n_strata_drop_no_mixed_GB"] = int(np.sum(present_any_gene & ~mixed_ok))
        row["glm_n_strata_used_beta_GxB"] = int(np.sum(usable_for_beta_gxb))
        row["glm_n_datasets_with_used_strata"] = int(np.unique(ds_of_stratum[usable_for_beta_gxb]).size) if np.any(usable_for_beta_gxb) else 0
        row["glm_low_info_flag"] = bool(row["glm_n_strata_used_beta_GxB"] < int(args.glm_min_usable_strata))

        # Per-dataset heterogeneity (MH within dataset; strata = theta_bin × leiden).
        n_strata_ds = int(theta_bins_used) * int(n_leiden)
        intE_by_ds: list[float] = []
        for ds_i, ds_name in enumerate(ds_uniq.tolist()):
            m_ds = ds_code == int(ds_i)
            if not np.any(m_ds):
                continue
            if not bool(ds_keep[int(ds_i)]):
                intE_by_ds.append(float("nan"))
                hetero_rows.append({"gene": str(g), "q": float(q_by_gene[j]), "dataset": str(ds_name), "delta_int_E_ds": float("nan")})
                continue
            st_ds = (theta_bin[m_ds].astype(int) * int(n_leiden) + lei_code[m_ds].astype(int)).astype(
                np.int32, copy=False
            )
            if int(np.sum(m_ds & (brdu == 1))) == 0 or int(np.sum(m_ds & (brdu == 0))) == 0:
                intE_by_ds.append(float("nan"))
                continue
            d1 = _mh_log_or(
                stratum_idx=st_ds[brdu[m_ds] == 1],
                G=G[m_ds][brdu[m_ds] == 1],
                y=edu[m_ds][brdu[m_ds] == 1],
                n_strata=n_strata_ds,
                cc=float(args.cc),
            )
            d0 = _mh_log_or(
                stratum_idx=st_ds[brdu[m_ds] == 0],
                G=G[m_ds][brdu[m_ds] == 0],
                y=edu[m_ds][brdu[m_ds] == 0],
                n_strata=n_strata_ds,
                cc=float(args.cc),
            )
            intE = float(d1 - d0) if (math.isfinite(d1) and math.isfinite(d0)) else float("nan")
            intE_by_ds.append(intE)
            hetero_rows.append({"gene": str(g), "q": float(q_by_gene[j]), "dataset": str(ds_name), "delta_int_E_ds": intE})

        intE_by_ds_arr = np.asarray(intE_by_ds, dtype=float)
        finite = np.isfinite(intE_by_ds_arr)
        row["intE_n_datasets"] = int(np.sum(finite))
        row["intE_sd_datasets"] = float(np.nanstd(intE_by_ds_arr, ddof=1)) if int(np.sum(finite)) >= 2 else float("nan")
        row["intE_sign_consistency_datasets"] = (
            float(np.mean(np.sign(intE_by_ds_arr[finite]) == np.sign(np.nanmean(intE_by_ds_arr[finite]))))
            if int(np.sum(finite)) >= 3
            else float("nan")
        )

        # GLM audit estimator (grouped binomial with stratum fixed effects).
        if bool(args.glm):
            key = (
                stratum_idx_all[keep_cells].astype(np.int64, copy=False) * 4
                + G[keep_cells].astype(np.int64, copy=False) * 2
                + brdu[keep_cells].astype(np.int64, copy=False)
            )
            n_tot = np.bincount(key, minlength=int(n_strata) * 4).astype(np.float64, copy=False)
            n_succ = np.bincount(
                key, weights=edu[keep_cells].astype(np.float64, copy=False), minlength=int(n_strata) * 4
            ).astype(
                np.float64, copy=False
            )
            nz = np.flatnonzero(n_tot > 0)
            strata = (nz // 4).astype(np.int32, copy=False)
            combo = (nz % 4).astype(np.int8, copy=False)
            g_row = (combo // 2).astype(np.float64, copy=False)
            b_row = (combo % 2).astype(np.float64, copy=False)
            X_glm = np.column_stack([g_row, b_row, g_row * b_row]).astype(np.float64, copy=False)
            coef = _fit_stratified_glm_binomial(y_succ=n_succ[nz], y_tot=n_tot[nz], strata=strata, X=X_glm)
            row["glm_fail_reason"] = coef.fail_reason
            row["glm_n_rows"] = int(coef.n_rows)
            row["glm_n_strata"] = int(coef.n_strata)
            row["glm_beta_G"] = float(coef.beta[0]) if coef.beta.size >= 1 else float("nan")
            row["glm_beta_B"] = float(coef.beta[1]) if coef.beta.size >= 2 else float("nan")
            row["glm_beta_GxB"] = float(coef.beta[2]) if coef.beta.size >= 3 else float("nan")
            row["glm_se_G"] = float(math.sqrt(coef.cov[0, 0])) if coef.cov.shape == (3, 3) else float("nan")
            row["glm_se_B"] = float(math.sqrt(coef.cov[1, 1])) if coef.cov.shape == (3, 3) else float("nan")
            row["glm_se_GxB"] = float(math.sqrt(coef.cov[2, 2])) if coef.cov.shape == (3, 3) else float("nan")
            row["glm_cov_G_GxB"] = float(coef.cov[0, 2]) if coef.cov.shape == (3, 3) else float("nan")
            z = float(row["glm_beta_GxB"] / row["glm_se_GxB"]) if float(row["glm_se_GxB"]) > 0 else float("nan")
            p = _z_to_p_two_sided(z)
            row["glm_z_GxB"] = z
            row["glm_p_GxB"] = p
        else:
            row["glm_fail_reason"] = "skipped"
            row["glm_n_rows"] = 0
            row["glm_n_strata"] = 0
            row["glm_beta_G"] = float("nan")
            row["glm_beta_B"] = float("nan")
            row["glm_beta_GxB"] = float("nan")
            row["glm_se_G"] = float("nan")
            row["glm_se_B"] = float("nan")
            row["glm_se_GxB"] = float("nan")
            row["glm_cov_G_GxB"] = float("nan")
            row["glm_z_GxB"] = float("nan")
            row["glm_p_GxB"] = float("nan")

        contribs: dict[str, MhDsContrib] = {}
        for name, cohort, y_all in endpoints:
            sidx = stratum_idx_all[cohort]
            c = _mh_ds_contrib(
                stratum_idx=sidx,
                G=G[cohort],
                y=y_all[cohort],
                n_strata=n_strata,
                n_datasets=n_datasets,
                n_leiden=n_leiden,
                theta_bins=int(theta_bins_used),
                cc=float(args.cc),
                keep_strata=keep_strata_gene,
            )
            contribs[name] = c
            lo, hi = _bootstrap_ci_from_contrib(
                c, rng=rng, n_boot=int(args.n_bootstrap), group_of_ds=bootstrap_group_of_ds
            )
            row[f"delta_{name}"] = float(c.delta)
            row[f"ci_low_{name}"] = float(lo)
            row[f"ci_high_{name}"] = float(hi)
            row[f"Ts_fold_{name}"] = float(math.exp(c.delta))
            row[f"Ts_ci_low_{name}"] = float(math.exp(lo))
            row[f"Ts_ci_high_{name}"] = float(math.exp(hi))
            # Sparse stratum diagnostics (pre-cc).
            row[f"sparse_strata_frac_{name}"] = _sparse_strata_fraction(
                stratum_idx=sidx, G=G[cohort], y=y_all[cohort], n_strata=n_strata, keep_strata=keep_strata_gene
            )

        # Overlap-restricted variants (shared support for B=1 and B=0 strata).
        contribs_ov: dict[str, MhDsContrib] = {}
        for name, cohort, y_all in endpoints:
            sidx = stratum_idx_all[cohort]
            c = _mh_ds_contrib(
                stratum_idx=sidx,
                G=G[cohort],
                y=y_all[cohort],
                n_strata=n_strata,
                n_datasets=n_datasets,
                n_leiden=n_leiden,
                theta_bins=int(theta_bins_used),
                cc=float(args.cc),
                keep_strata=keep_overlap_gene,
            )
            contribs_ov[name] = c
            lo, hi = _bootstrap_ci_from_contrib(
                c, rng=rng, n_boot=int(args.n_bootstrap), group_of_ds=bootstrap_group_of_ds
            )
            row[f"delta_{name}_overlap"] = float(c.delta)
            row[f"ci_low_{name}_overlap"] = float(lo)
            row[f"ci_high_{name}_overlap"] = float(hi)
            row[f"Ts_fold_{name}_overlap"] = float(math.exp(c.delta)) if math.isfinite(c.delta) else float("nan")
            row[f"Ts_ci_low_{name}_overlap"] = float(math.exp(lo)) if math.isfinite(lo) else float("nan")
            row[f"Ts_ci_high_{name}_overlap"] = float(math.exp(hi)) if math.isfinite(hi) else float("nan")
            row[f"sparse_strata_frac_{name}_overlap"] = _sparse_strata_fraction(
                stratum_idx=sidx,
                G=G[cohort],
                y=y_all[cohort],
                n_strata=n_strata,
                keep_strata=keep_overlap_gene,
            )

        # Mirror ratio: delta1 - delta2 (BrdU+ conditional minus EdU+ conditional).
        mr, mr_lo, mr_hi = _bootstrap_ci_for_delta_diff(
            contribs["BrdUplus_Edu"], contribs["Eduplus_BrdU"], rng=rng, n_boot=int(args.n_bootstrap)
        )
        row["mirror_ratio_delta"] = float(mr)
        row["mirror_ratio_ci_low"] = float(mr_lo)
        row["mirror_ratio_ci_high"] = float(mr_hi)

        # Interaction terms (same as mh_quadrant_validation.py, but also keep for downstream diffs).
        intE, intE_lo, intE_hi = _bootstrap_ci_for_delta_diff(
            contribs["BrdUplus_Edu"],
            contribs["BrdUminus_Edu"],
            rng=rng,
            n_boot=int(args.n_bootstrap),
            group_of_ds=bootstrap_group_of_ds,
        )
        intB, intB_lo, intB_hi = _bootstrap_ci_for_delta_diff(
            contribs["Eduplus_BrdU"],
            contribs["EdUminus_BrdU"],
            rng=rng,
            n_boot=int(args.n_bootstrap),
            group_of_ds=bootstrap_group_of_ds,
        )
        row["delta_int_E"] = float(intE)
        row["ci_low_int_E"] = float(intE_lo)
        row["ci_high_int_E"] = float(intE_hi)
        row["Ts_fold_int_E"] = float(math.exp(intE))
        row["Ts_ci_low_int_E"] = float(math.exp(intE_lo))
        row["Ts_ci_high_int_E"] = float(math.exp(intE_hi))
        row["delta_int_B"] = float(intB)
        row["ci_low_int_B"] = float(intB_lo)
        row["ci_high_int_B"] = float(intB_hi)
        row["Ts_fold_int_B"] = float(math.exp(intB))
        row["Ts_ci_low_int_B"] = float(math.exp(intB_lo))
        row["Ts_ci_high_int_B"] = float(math.exp(intB_hi))

        intE_ov, intE_ov_lo, intE_ov_hi = _bootstrap_ci_for_delta_diff(
            contribs_ov["BrdUplus_Edu"],
            contribs_ov["BrdUminus_Edu"],
            rng=rng,
            n_boot=int(args.n_bootstrap),
            group_of_ds=bootstrap_group_of_ds,
        )
        row["delta_int_E_overlap"] = float(intE_ov)
        row["ci_low_int_E_overlap"] = float(intE_ov_lo)
        row["ci_high_int_E_overlap"] = float(intE_ov_hi)
        row["Ts_fold_int_E_overlap"] = float(math.exp(intE_ov)) if math.isfinite(intE_ov) else float("nan")
        row["Ts_ci_low_int_E_overlap"] = float(math.exp(intE_ov_lo)) if math.isfinite(intE_ov_lo) else float("nan")
        row["Ts_ci_high_int_E_overlap"] = float(math.exp(intE_ov_hi)) if math.isfinite(intE_ov_hi) else float("nan")
        row["delta_int_E_overlap_minus_all"] = float(intE_ov - intE) if (math.isfinite(intE_ov) and math.isfinite(intE)) else float("nan")
        row["intE_sign_flip_overlap_vs_all"] = (
            bool(np.sign(intE_ov) != np.sign(intE)) if (math.isfinite(intE_ov) and math.isfinite(intE) and intE != 0.0) else False
        )

        # GLM-as-primary (final shortlist stage) + explicit MH-vs-GLM divergence QC.
        if bool(args.glm):
            glm_beta_gxb = float(row["glm_beta_GxB"])
            glm_se_gxb = float(row["glm_se_GxB"])
            row["intE_primary_source"] = "glm_beta_GxB"
            row["intE_primary"] = glm_beta_gxb if math.isfinite(glm_beta_gxb) else float("nan")
            row["intE_primary_se"] = glm_se_gxb if math.isfinite(glm_se_gxb) else float("nan")
            row["intE_primary_p"] = float(row["glm_p_GxB"]) if math.isfinite(float(row["glm_p_GxB"])) else float("nan")
            row["intE_primary_fail_reason"] = str(row["glm_fail_reason"]) if row["glm_fail_reason"] is not None else ""
            row["IOR_primary"] = float(math.exp(glm_beta_gxb)) if math.isfinite(glm_beta_gxb) else float("nan")
            if math.isfinite(glm_beta_gxb) and glm_se_gxb > 0.0:
                lo = glm_beta_gxb - 1.96 * glm_se_gxb
                hi = glm_beta_gxb + 1.96 * glm_se_gxb
                row["IOR_primary_ci_low"] = float(math.exp(lo))
                row["IOR_primary_ci_high"] = float(math.exp(hi))
            else:
                row["IOR_primary_ci_low"] = float("nan")
                row["IOR_primary_ci_high"] = float("nan")

            row["glm_minus_mh_intE_overlap"] = (
                float(glm_beta_gxb - intE_ov) if (math.isfinite(glm_beta_gxb) and math.isfinite(intE_ov)) else float("nan")
            )
            row["abs_glm_minus_mh_intE_overlap"] = (
                float(abs(glm_beta_gxb - intE_ov))
                if (math.isfinite(glm_beta_gxb) and math.isfinite(intE_ov))
                else float("nan")
            )
            row["glm_mh_intE_overlap_sign_mismatch"] = (
                bool(np.sign(glm_beta_gxb) != np.sign(intE_ov))
                if (math.isfinite(glm_beta_gxb) and math.isfinite(intE_ov) and float(intE_ov) != 0.0)
                else False
            )
            row["glm_outside_mh_intE_overlap_ci"] = (
                bool((glm_beta_gxb < float(intE_ov_lo)) or (glm_beta_gxb > float(intE_ov_hi)))
                if (math.isfinite(glm_beta_gxb) and math.isfinite(intE_ov_lo) and math.isfinite(intE_ov_hi))
                else False
            )
            mh_se = (
                float((float(intE_ov_hi) - float(intE_ov_lo)) / (2.0 * 1.96))
                if (math.isfinite(intE_ov_lo) and math.isfinite(intE_ov_hi) and float(intE_ov_hi) > float(intE_ov_lo))
                else float("nan")
            )
            row["mh_intE_overlap_se_approx"] = mh_se
            diff_se = (
                float(math.sqrt((glm_se_gxb**2) + (mh_se**2)))
                if (math.isfinite(glm_se_gxb) and glm_se_gxb > 0.0 and math.isfinite(mh_se) and mh_se > 0.0)
                else float("nan")
            )
            row["glm_mh_intE_overlap_diff_z"] = (
                float(abs(glm_beta_gxb - intE_ov) / diff_se)
                if (math.isfinite(glm_beta_gxb) and math.isfinite(intE_ov) and math.isfinite(diff_se) and diff_se > 0.0)
                else float("nan")
            )
            row["flag_glm_vs_mh_overlap_divergent_z"] = bool(
                math.isfinite(float(row["glm_mh_intE_overlap_diff_z"]))
                and float(row["glm_mh_intE_overlap_diff_z"]) > float(args.glm_mh_divergence_z)
            )
            row["flag_glm_vs_mh_overlap_divergent"] = bool(
                row["glm_mh_intE_overlap_sign_mismatch"]
                or (
                    math.isfinite(float(row["abs_glm_minus_mh_intE_overlap"]))
                    and float(row["abs_glm_minus_mh_intE_overlap"]) > float(args.glm_mh_divergence_abs)
                )
            )
        else:
            row["intE_primary_source"] = "mh_intE_overlap"
            row["intE_primary"] = float(intE_ov)
            row["intE_primary_se"] = float("nan")
            row["intE_primary_p"] = float("nan")
            row["intE_primary_fail_reason"] = "glm_skipped"
            row["IOR_primary"] = float(row["Ts_fold_int_E_overlap"])
            row["IOR_primary_ci_low"] = float(row["Ts_ci_low_int_E_overlap"])
            row["IOR_primary_ci_high"] = float(row["Ts_ci_high_int_E_overlap"])
            row["glm_minus_mh_intE_overlap"] = float("nan")
            row["abs_glm_minus_mh_intE_overlap"] = float("nan")
            row["glm_mh_intE_overlap_sign_mismatch"] = False
            row["glm_outside_mh_intE_overlap_ci"] = False
            row["mh_intE_overlap_se_approx"] = float("nan")
            row["glm_mh_intE_overlap_diff_z"] = float("nan")
            row["flag_glm_vs_mh_overlap_divergent_z"] = False
            row["flag_glm_vs_mh_overlap_divergent"] = False

        # int_diff = int_E - int_B with shared dataset bootstrap.
        idiff, idiff_lo, idiff_hi = _bootstrap_ci_for_int_diff(
            contribs["BrdUplus_Edu"],
            contribs["Eduplus_BrdU"],
            contribs["BrdUminus_Edu"],
            contribs["EdUminus_BrdU"],
            rng=rng,
            n_boot=int(args.n_bootstrap),
            group_of_ds=bootstrap_group_of_ds,
        )
        row["delta_int_diff_E_minus_B"] = float(idiff)
        row["ci_low_int_diff_E_minus_B"] = float(idiff_lo)
        row["ci_high_int_diff_E_minus_B"] = float(idiff_hi)

        row["sig_int_E"] = bool((intE_lo > 0) or (intE_hi < 0))
        row["sig_int_E_overlap"] = bool((intE_ov_lo > 0) or (intE_ov_hi < 0))
        row["sig_int_B"] = bool((intB_lo > 0) or (intB_hi < 0))
        row["int_sign_match"] = bool(np.sign(intE) == np.sign(intB))

        s1 = math.copysign(1.0, contribs["BrdUplus_Edu"].delta) if math.isfinite(contribs["BrdUplus_Edu"].delta) else 0.0
        s2 = math.copysign(1.0, contribs["Eduplus_BrdU"].delta) if math.isfinite(contribs["Eduplus_BrdU"].delta) else 0.0
        row["mirror_sign_match"] = bool(s1 == s2) if (s1 != 0.0 and s2 != 0.0) else False

        # (Fix 5) Probability-scale interaction metrics (risk differences) to make saturation explicit.
        rd = _rd_ds_contrib(
            stratum_idx=stratum_idx_all[keep_cells],
            ds_of_stratum=ds_of_stratum,
            G=G[keep_cells].astype(int, copy=False),
            B=brdu[keep_cells].astype(int, copy=False),
            E=edu[keep_cells].astype(int, copy=False),
            n_strata=n_strata,
            n_datasets=n_datasets,
            cc=float(args.cc),
        )
        rd_ci = _bootstrap_ci_for_rd(rd, rng=rng, n_boot=int(args.n_bootstrap), group_of_ds=bootstrap_group_of_ds)
        row["rd_B1"] = float(rd_ci["rd_B1"][0])
        row["rd_ci_low_B1"] = float(rd_ci["rd_B1"][1])
        row["rd_ci_high_B1"] = float(rd_ci["rd_B1"][2])
        row["rd_B0"] = float(rd_ci["rd_B0"][0])
        row["rd_ci_low_B0"] = float(rd_ci["rd_B0"][1])
        row["rd_ci_high_B0"] = float(rd_ci["rd_B0"][2])
        row["rd_int"] = float(rd_ci["rd_int"][0])
        row["rd_ci_low_int"] = float(rd_ci["rd_int"][1])
        row["rd_ci_high_int"] = float(rd_ci["rd_int"][2])

        # Gate counts within BrdU+ by Leiden (consultant asked for "not one cluster" and support).
        for lei in ["6", "14", "15"]:
            m = (brdu == 1) & (leiden.astype(str) == lei)
            row[f"n_brdUpos_leiden_{lei}"] = int(np.sum(m))
            row[f"n_gatepos_brdUpos_leiden_{lei}"] = int(np.sum(m & G))
            row[f"frac_gatepos_brdUpos_leiden_{lei}"] = float(np.mean(G[m])) if np.any(m) else float("nan")

        rows.append(row)

        # Gene × dataset support rows.
        g1 = G.astype(np.int64, copy=False)
        n_G1_by_ds = np.bincount(ds_code, weights=g1, minlength=int(n_datasets)).astype(np.int64, copy=False)
        n_G1_B1_by_ds = np.bincount(ds_code, weights=g1 * brdu.astype(np.int64, copy=False), minlength=int(n_datasets)).astype(
            np.int64, copy=False
        )
        n_G1_B0_by_ds = n_G1_by_ds - n_G1_B1_by_ds
        n_G0_by_ds = n_cells_by_ds - n_G1_by_ds
        n_G0_B1_by_ds = n_B1_by_ds - n_G1_B1_by_ds
        n_G0_B0_by_ds = n_G0_by_ds - n_G0_B1_by_ds
        # Gene-specific BG-overlap stratum counts per dataset.
        n_overlap_bg_strata_by_ds = np.bincount(
            ds_of_stratum[present_any_gene & keep_overlap_gene_bg], minlength=int(n_datasets)
        ).astype(np.int64, copy=False)
        for ds_i, ds_name in enumerate(ds_uniq.tolist()):
            n_cells = int(n_cells_by_ds[ds_i])
            if n_cells == 0:
                continue
            support_ok = bool(ds_keep[int(ds_i)])
            reason = ""
            if not support_ok:
                if int(n_detect_by_ds[ds_i]) < int(args.support_min_detected):
                    reason = "min_detected"
                elif int(n_detect_gatepos_by_ds[ds_i]) < int(args.support_min_detected_gatepos):
                    reason = "min_detected_gatepos"
                else:
                    reason = "filtered"
            support_rows.append(
                {
                    "gene": str(g),
                    "q": float(q_by_gene[j]),
                    "dataset": str(ds_name),
                    "n_cells": n_cells,
                    "n_B1": int(n_B1_by_ds[ds_i]),
                    "n_B0": int(n_cells_by_ds[ds_i] - n_B1_by_ds[ds_i]),
                    "n_E1": int(n_E1_by_ds[ds_i]),
                    "n_E0": int(n_cells_by_ds[ds_i] - n_E1_by_ds[ds_i]),
                    "n_G1": int(n_G1_by_ds[ds_i]),
                    "gate_frac": float(n_G1_by_ds[ds_i] / n_cells_by_ds[ds_i]),
                    "n_detect": int(n_detect_by_ds[ds_i]),
                    "det_rate": float(n_detect_by_ds[ds_i] / n_cells_by_ds[ds_i]),
                    "n_detect_gatepos": int(n_detect_gatepos_by_ds[ds_i]),
                    "det_gatepos_rate": float(n_detect_gatepos_by_ds[ds_i] / max(int(n_G1_by_ds[ds_i]), 1)),
                    "support_ok": support_ok,
                    "support_fail_reason": str(reason),
                    "n_G1_B1": int(n_G1_B1_by_ds[ds_i]),
                    "n_G1_B0": int(n_G1_B0_by_ds[ds_i]),
                    "n_G0_B1": int(n_G0_B1_by_ds[ds_i]),
                    "n_G0_B0": int(n_G0_B0_by_ds[ds_i]),
                    "n_present_strata_dataset_theta_leiden": int(n_present_strata_by_ds[ds_i]),
                    "n_overlap_strata_dataset_theta_leiden": int(n_overlap_strata_by_ds[ds_i]),
                    "n_overlap_bg_strata_dataset_theta_leiden": int(n_overlap_bg_strata_by_ds[ds_i]),
                }
            )

    out = pd.DataFrame(rows)
    out["glm_fdr_GxB"] = _bh_fdr(out["glm_p_GxB"].to_numpy(float))
    out["intE_primary_fdr"] = out["glm_fdr_GxB"] if bool(args.glm) else float("nan")

    # Merge in earlier scorecard fields (animal robustness + confound flags + Leiden confounding metrics).
    sc = pd.read_csv(args.scorecard_csv)
    keep_sc = [
        "gene",
        "primary_pass",
        "passes_not_one_animal",
        "strong_same_animals",
        "mh_delta_all",
        "mh_delta_all_ci_low",
        "mh_delta_all_ci_high",
        "mh_Ts_fold_all",
        "mh_Ts_fold_ci_low_all",
        "mh_Ts_fold_ci_high_all",
        "mh_delta_leiden_6",
        "mh_delta_leiden_14",
        "mh_delta_leiden_15",
        "n_robust_leidens",
        "delta_naive_minus_mh",
        "naive_mh_sign_flip",
        "cramers_v_G_vs_leiden",
        "delta_logloss_LODO",
        "flag_sex_axis",
        "flag_stress_axis",
    ]
    missing = [c for c in keep_sc if c not in sc.columns]
    if missing:
        raise ValueError(f"--scorecard-csv missing columns: {missing}")
    out = out.merge(sc.loc[:, keep_sc], on="gene", how="left", validate="one_to_one")

    if args.crossfit_bestq_by_gene_csv is not None:
        cf = pd.read_csv(args.crossfit_bestq_by_gene_csv)
        needed = {
            "gene",
            "n_folds",
            "q_selected_mode",
            "test_intE_selected_mean",
            "test_intE_selected_sd",
            "test_intE_baseline_mean",
            "test_intE_baseline_sd",
        }
        missing = needed - set(cf.columns)
        if missing:
            raise ValueError(f"--crossfit-bestq-by-gene-csv missing columns: {sorted(missing)}")
        cf = cf.loc[:, sorted(needed)].copy()
        cf = cf.rename(
            columns={
                "n_folds": "crossfit_n_folds",
                "q_selected_mode": "crossfit_q_selected_mode",
                "test_intE_selected_mean": "crossfit_test_intE_selected_mean",
                "test_intE_selected_sd": "crossfit_test_intE_selected_sd",
                "test_intE_baseline_mean": "crossfit_test_intE_baseline_mean",
                "test_intE_baseline_sd": "crossfit_test_intE_baseline_sd",
            }
        )
        out = out.merge(cf, on="gene", how="left", validate="one_to_one")
    else:
        out["crossfit_n_folds"] = float("nan")
        out["crossfit_q_selected_mode"] = float("nan")
        out["crossfit_test_intE_selected_mean"] = float("nan")
        out["crossfit_test_intE_selected_sd"] = float("nan")
        out["crossfit_test_intE_baseline_mean"] = float("nan")
        out["crossfit_test_intE_baseline_sd"] = float("nan")

    if bool(args.glm):
        out["abs_intE_primary"] = out["intE_primary"].abs()
        glm_ok = out["glm_fail_reason"].isna() | (out["glm_fail_reason"] == "")
        out = out.assign(glm_ok=glm_ok)
        out = out.sort_values(["glm_ok", "intE_primary_fdr", "abs_intE_primary"], ascending=[False, True, False])
        out = out.drop(columns=["glm_ok"])
    else:
        # Sort by interaction magnitude using overlap-restricted IntE (shared support for B=1 and B=0).
        out["abs_delta_int_E_overlap"] = out["delta_int_E_overlap"].abs()
        out = out.sort_values(["sig_int_E_overlap", "abs_delta_int_E_overlap"], ascending=[False, False])

    out_path = args.outdir / "consultant_exec_summary.csv"
    out.to_csv(out_path, index=False)
    pd.DataFrame(support_rows).to_csv(args.outdir / "support_table_gene_dataset.csv", index=False)
    pd.DataFrame(hetero_rows).to_csv(args.outdir / "heterogeneity_intE_by_dataset.csv", index=False)

    # Also write a compact TSV for easy paste into email.
    tsv_cols = [
        "gene",
        "q",
        "Ts_fold_BrdUplus_Edu",
        "delta_Eduplus_BrdU",
        "Ts_fold_Eduplus_BrdU",
        "Ts_fold_BrdUminus_Edu",
        "Ts_fold_EdUminus_BrdU",
        "delta_int_E",
        "Ts_fold_int_E",
        "Ts_ci_low_int_E",
        "Ts_ci_high_int_E",
        "delta_int_E_overlap",
        "Ts_fold_int_E_overlap",
        "Ts_ci_low_int_E_overlap",
        "Ts_ci_high_int_E_overlap",
        "intE_primary_source",
        "intE_primary",
        "intE_primary_se",
        "intE_primary_p",
        "intE_primary_fdr",
        "IOR_primary",
        "IOR_primary_ci_low",
        "IOR_primary_ci_high",
        "abs_glm_minus_mh_intE_overlap",
        "glm_mh_intE_overlap_diff_z",
        "flag_glm_vs_mh_overlap_divergent",
        "glm_beta_GxB",
        "glm_se_GxB",
        "glm_fail_reason",
        "glm_fdr_GxB",
        "crossfit_q_selected_mode",
        "crossfit_test_intE_selected_mean",
        "crossfit_test_intE_baseline_mean",
        "rd_B1",
        "rd_ci_low_B1",
        "rd_ci_high_B1",
        "rd_B0",
        "rd_ci_low_B0",
        "rd_ci_high_B0",
        "rd_int",
        "rd_ci_low_int",
        "rd_ci_high_int",
        "delta_int_B",
        "Ts_fold_int_B",
        "Ts_ci_low_int_B",
        "Ts_ci_high_int_B",
        "int_sign_match",
        "delta_int_diff_E_minus_B",
        "mirror_ratio_delta",
        "mirror_sign_match",
        "passes_not_one_animal",
        "strong_same_animals",
        "flag_sex_axis",
        "flag_stress_axis",
    ]
    out.loc[:, tsv_cols].to_csv(args.outdir / "consultant_exec_summary.compact.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
