#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    s = pd.Series([f"{d}|{o}" for d, o in zip(ds.tolist(), on.tolist(), strict=True)], copy=False)
    return pd.util.hash_pandas_object(s, index=False).to_numpy(np.uint64, copy=False)


def _make_gates_by_dataset_quantile_exact(
    R: np.ndarray,
    dataset: np.ndarray,
    q_by_gene: np.ndarray,
    *,
    jitter: np.ndarray,
) -> np.ndarray:
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
                # Select the k largest values.
                cut = np.argpartition(vals, n - k)[n - k :]
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
    c1: MhDsContrib, c2: MhDsContrib, *, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    present = (c1.R_by_ds + c1.S_by_ds + c2.R_by_ds + c2.S_by_ds) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    d_point = float(c1.delta - c2.delta)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        d1 = float(math.log(float(c1.R_by_ds[samp].sum()) / float(c1.S_by_ds[samp].sum())))
        d2 = float(math.log(float(c2.R_by_ds[samp].sum()) / float(c2.S_by_ds[samp].sum())))
        boots[i] = d1 - d2
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return (d_point, float(lo), float(hi))


def _bootstrap_ci_from_contrib(
    c: MhDsContrib, *, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    present = (c.R_by_ds + c.S_by_ds) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        Rb = float(c.R_by_ds[samp].sum())
        Sb = float(c.S_by_ds[samp].sum())
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
) -> tuple[float, float, float]:
    # int_diff := (d1 - d3) - (d2 - d4) = d1 - d3 - d2 + d4
    present = (
        c1.R_by_ds
        + c1.S_by_ds
        + c2.R_by_ds
        + c2.S_by_ds
        + c3.R_by_ds
        + c3.S_by_ds
        + c4.R_by_ds
        + c4.S_by_ds
    ) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    d_point = float((c1.delta - c3.delta) - (c2.delta - c4.delta))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        d1 = float(math.log(float(c1.R_by_ds[samp].sum()) / float(c1.S_by_ds[samp].sum())))
        d2 = float(math.log(float(c2.R_by_ds[samp].sum()) / float(c2.S_by_ds[samp].sum())))
        d3 = float(math.log(float(c3.R_by_ds[samp].sum()) / float(c3.S_by_ds[samp].sum())))
        d4 = float(math.log(float(c4.R_by_ds[samp].sum()) / float(c4.S_by_ds[samp].sum())))
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
    c: RdDsContrib, *, rng: np.random.Generator, n_boot: int
) -> dict[str, tuple[float, float, float]]:
    present = (c.den_B1_by_ds + c.den_B0_by_ds) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
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
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        b1 = _ratio(c.num_B1_by_ds, c.den_B1_by_ds, samp)
        b0 = _ratio(c.num_B0_by_ds, c.den_B0_by_ds, samp)
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
    *, stratum_idx: np.ndarray, G: np.ndarray, y: np.ndarray, n_strata: int
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
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument(
        "--theta-bin-mode",
        type=str,
        choices=["quantile", "angle"],
        default="quantile",
        help="Theta binning: quantile within (dataset×leiden) vs fixed absolute angular bins.",
    )
    ap.add_argument("--n-bootstrap", type=int, default=300)
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
    n_strata = n_datasets * int(args.theta_bins) * n_leiden

    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    theta = _compute_tricycle_theta(adata, trc=trc, dataset=dataset)
    theta_bin = _theta_bins(
        theta,
        ds_code,
        lei_code,
        n_bins=int(args.theta_bins),
        mode=str(args.theta_bin_mode),
    )
    stratum_idx_all = (ds_code * int(args.theta_bins) + theta_bin.astype(int)) * n_leiden + lei_code
    ds_of_stratum = (np.arange(int(n_strata), dtype=int) // (int(args.theta_bins) * int(n_leiden))).astype(
        int, copy=False
    )

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
    sin_t = np.sin(theta).astype(np.float32, copy=False)
    cos_t = np.cos(theta).astype(np.float32, copy=False)
    R = _residualize_matrix_within_dataset(X, sin_t=sin_t, cos_t=cos_t, dataset=dataset)
    h = _stable_hash_u64(dataset, obs_name)
    jitter = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)) - 0.5
    Gmat = _make_gates_by_dataset_quantile_exact(R, dataset=dataset, q_by_gene=q_by_gene, jitter=jitter)

    # Endpoint definitions.
    endpoints: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("BrdUplus_Edu", brdu == 1, edu),
        ("Eduplus_BrdU", edu == 1, brdu),
        ("BrdUminus_Edu", brdu == 0, edu),
        ("EdUminus_BrdU", edu == 0, brdu),
    ]

    rows: list[dict[str, object]] = []
    for j, g in enumerate(genes):
        G = Gmat[:, j]
        row: dict[str, object] = {"gene": g, "q": float(q_by_gene[j])}

        # GLM audit estimator (grouped binomial with stratum fixed effects).
        if bool(args.glm):
            key = stratum_idx_all.astype(np.int64, copy=False) * 4 + G.astype(np.int64, copy=False) * 2 + brdu.astype(
                np.int64, copy=False
            )
            n_tot = np.bincount(key, minlength=int(n_strata) * 4).astype(np.float64, copy=False)
            n_succ = np.bincount(key, weights=edu.astype(np.float64, copy=False), minlength=int(n_strata) * 4).astype(
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
                theta_bins=int(args.theta_bins),
                cc=float(args.cc),
            )
            contribs[name] = c
            lo, hi = _bootstrap_ci_from_contrib(c, rng=rng, n_boot=int(args.n_bootstrap))
            row[f"delta_{name}"] = float(c.delta)
            row[f"ci_low_{name}"] = float(lo)
            row[f"ci_high_{name}"] = float(hi)
            row[f"Ts_fold_{name}"] = float(math.exp(c.delta))
            row[f"Ts_ci_low_{name}"] = float(math.exp(lo))
            row[f"Ts_ci_high_{name}"] = float(math.exp(hi))
            # Sparse stratum diagnostics (pre-cc).
            row[f"sparse_strata_frac_{name}"] = _sparse_strata_fraction(
                stratum_idx=sidx, G=G[cohort], y=y_all[cohort], n_strata=n_strata
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
            contribs["BrdUplus_Edu"], contribs["BrdUminus_Edu"], rng=rng, n_boot=int(args.n_bootstrap)
        )
        intB, intB_lo, intB_hi = _bootstrap_ci_for_delta_diff(
            contribs["Eduplus_BrdU"], contribs["EdUminus_BrdU"], rng=rng, n_boot=int(args.n_bootstrap)
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

        # int_diff = int_E - int_B with shared dataset bootstrap.
        idiff, idiff_lo, idiff_hi = _bootstrap_ci_for_int_diff(
            contribs["BrdUplus_Edu"],
            contribs["Eduplus_BrdU"],
            contribs["BrdUminus_Edu"],
            contribs["EdUminus_BrdU"],
            rng=rng,
            n_boot=int(args.n_bootstrap),
        )
        row["delta_int_diff_E_minus_B"] = float(idiff)
        row["ci_low_int_diff_E_minus_B"] = float(idiff_lo)
        row["ci_high_int_diff_E_minus_B"] = float(idiff_hi)

        row["sig_int_E"] = bool((intE_lo > 0) or (intE_hi < 0))
        row["sig_int_B"] = bool((intB_lo > 0) or (intB_hi < 0))
        row["int_sign_match"] = bool(np.sign(intE) == np.sign(intB))

        s1 = math.copysign(1.0, contribs["BrdUplus_Edu"].delta) if math.isfinite(contribs["BrdUplus_Edu"].delta) else 0.0
        s2 = math.copysign(1.0, contribs["Eduplus_BrdU"].delta) if math.isfinite(contribs["Eduplus_BrdU"].delta) else 0.0
        row["mirror_sign_match"] = bool(s1 == s2) if (s1 != 0.0 and s2 != 0.0) else False

        # (Fix 5) Probability-scale interaction metrics (risk differences) to make saturation explicit.
        rd = _rd_ds_contrib(
            stratum_idx=stratum_idx_all,
            ds_of_stratum=ds_of_stratum,
            G=G.astype(int, copy=False),
            B=brdu.astype(int, copy=False),
            E=edu.astype(int, copy=False),
            n_strata=n_strata,
            n_datasets=n_datasets,
            cc=float(args.cc),
        )
        rd_ci = _bootstrap_ci_for_rd(rd, rng=rng, n_boot=int(args.n_bootstrap))
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

    out = pd.DataFrame(rows)

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

    # Sort by interaction magnitude (primary review criterion at this stage).
    out["abs_delta_int_E"] = out["delta_int_E"].abs()
    out = out.sort_values(["sig_int_E", "abs_delta_int_E"], ascending=[False, False])

    out_path = args.outdir / "consultant_exec_summary.csv"
    out.to_csv(out_path, index=False)

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
        "glm_beta_GxB",
        "glm_se_GxB",
        "glm_fail_reason",
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
