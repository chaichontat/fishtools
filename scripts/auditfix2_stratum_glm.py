#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import re
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd


def _dataset_animal(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r} (expected contains 'JaxA#').")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _load_tricycle_ref(path: pathlib.Path) -> pd.DataFrame:
    trc = pd.read_csv(path)
    required = {"symbol", "pc1.rot", "pc2.rot"}
    missing = required - set(trc.columns)
    if missing:
        raise ValueError(f"tricycle ref CSV missing columns: {sorted(missing)}")
    trc = trc.loc[:, ["symbol", "pc1.rot", "pc2.rot"]].copy()
    trc["symbol"] = trc["symbol"].astype(str)
    return trc


def _compute_tricycle_theta(adata, idx: np.ndarray, trc: pd.DataFrame, dataset_for_idx: np.ndarray) -> np.ndarray:
    # Within-dataset mean-centering on tricycle ref genes, then projection and atan2.
    ref_syms = np.asarray(trc["symbol"], dtype=object)
    var_syms = np.asarray(adata.var_names, dtype=object)
    ref_mask = np.isin(var_syms, ref_syms)
    if not np.any(ref_mask):
        raise ValueError("No shared genes between tricycle ref and adata.var_names.")

    sym_to_ref = {s: i for i, s in enumerate(ref_syms)}
    ref_rows = np.array([sym_to_ref[s] for s in var_syms[ref_mask]], dtype=int)
    w1 = trc["pc1.rot"].to_numpy(float)[ref_rows]
    w2 = trc["pc2.rot"].to_numpy(float)[ref_rows]

    X = adata.X[idx][:, ref_mask]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    ds = np.asarray(dataset_for_idx, dtype=object)
    theta = np.empty(X.shape[0], dtype=float)
    for d in np.unique(ds):
        m = ds == d
        Xd = X[m]
        Xd = Xd - Xd.mean(axis=0, keepdims=True)
        pc1 = Xd @ w1
        pc2 = Xd @ w2
        theta[m] = np.arctan2(pc2, pc1)
    return theta


def _theta_bins_within_dataset(theta: np.ndarray, dataset: np.ndarray, n_bins: int) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    out = np.empty(theta.shape[0], dtype=np.int16)
    for d in np.unique(ds):
        m = ds == d
        t = theta[m].astype(float)
        edges = np.quantile(t, q=np.linspace(0.0, 1.0, int(n_bins) + 1))
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], edges[i - 1] + 1.0)
        out[m] = np.digitize(t, edges[1:-1], right=False).astype(np.int16)
    return out


def _log1p_raw_for_gene(adata, idx: np.ndarray, gene: str) -> np.ndarray:
    var = np.asarray(adata.var_names, dtype=object)
    where = np.where(var == gene)[0]
    if where.size == 0:
        raise ValueError(f"Gene not found in adata.var_names: {gene}")
    col = int(where[0])
    if "raw" in adata.layers:
        X = adata.layers["raw"][idx][:, col]
    else:
        X = adata.X[idx][:, col]
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = np.asarray(X, dtype=float).reshape(-1)
    return np.log1p(x)


def _raw_count_for_gene(adata, idx: np.ndarray, gene: str) -> np.ndarray:
    """Return raw counts (not log1p) for a gene."""
    var = np.asarray(adata.var_names, dtype=object)
    where = np.where(var == gene)[0]
    if where.size == 0:
        raise ValueError(f"Gene not found in adata.var_names: {gene}")
    col = int(where[0])
    if "raw" in adata.layers:
        X = adata.layers["raw"][idx][:, col]
    else:
        X = adata.X[idx][:, col]
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = np.asarray(X, dtype=float).reshape(-1)
    return x


def _pearson_residualize_by_dataset(
    x_raw: np.ndarray,
    total_counts: np.ndarray,
    dataset: np.ndarray,
    *,
    theta: float = 100.0,
    clip: float | None = 10.0,
) -> np.ndarray:
    """Pearson residuals for raw counts within dataset.

    This matches the per-batch Pearson residual logic used by
    `scanpy.experimental.pp.normalize_pearson_residuals`, but computed for a
    single gene vector `x_raw` using per-cell `total_counts`.
    """
    if not (float(theta) > 0.0):
        raise ValueError(f"theta must be > 0, got {theta!r}")
    ds = np.asarray(dataset, dtype=object)
    x_raw = np.asarray(x_raw, dtype=float)
    total = np.asarray(total_counts, dtype=float)

    out = np.empty(x_raw.shape[0], dtype=float)
    for d in np.unique(ds):
        md = ds == d
        if not bool(np.any(md)):
            continue

        xt = x_raw[md]
        tt = total[md]

        denom = float(np.sum(tt))
        if not (denom > 0.0):
            out[md] = 0.0
            continue

        p = float(np.sum(xt) / denom)
        if not (p > 0.0):
            out[md] = 0.0
            continue

        mu = tt * p
        var = mu + (mu * mu) / float(theta)
        var = np.maximum(var, 1e-12)
        r = (xt - mu) / np.sqrt(var)
        if clip is not None:
            c = float(clip)
            r = np.clip(r, -c, c)
        out[md] = r
    return out


def _depth_residualize_by_dataset_leiden(
    x: np.ndarray, log_total: np.ndarray, dataset: np.ndarray, leiden: np.ndarray
) -> np.ndarray:
    """Residualize x ~ 1 + log_total within (dataset, leiden)."""
    ds = np.asarray(dataset, dtype=object)
    lei = np.asarray(leiden, dtype=int)
    resid = np.empty(x.shape[0], dtype=float)
    for d in np.unique(ds):
        md = ds == d
        for lei_val in np.unique(lei[md]):
            m = md & (lei == int(lei_val))
            xt = x[m].astype(float, copy=False)
            lt = log_total[m].astype(float, copy=False)
            if xt.size == 0:
                continue
            v = float(np.var(lt))
            if not (v > 0.0):
                resid[m] = xt - float(np.mean(xt))
                continue
            b = float(np.cov(xt, lt, ddof=0)[0, 1] / v)
            a = float(np.mean(xt) - b * np.mean(lt))
            resid[m] = xt - (a + b * lt)
    return resid


def _stable_hash_u64(dataset: np.ndarray, obs_name: np.ndarray) -> np.ndarray:
    ds = dataset.astype(str)
    on = obs_name.astype(str)
    s = pd.Series([f"{d}|{o}" for d, o in zip(ds.tolist(), on.tolist(), strict=True)], copy=False)
    return pd.util.hash_pandas_object(s, index=False).to_numpy(np.uint64, copy=False)


def _gate_top_fraction_within_strata(score: np.ndarray, stratum: np.ndarray, q: float, x: np.ndarray | None = None) -> np.ndarray:
    s = np.asarray(stratum)
    out = np.zeros(score.shape[0], dtype=bool)
    for st in np.unique(s):
        idx = np.flatnonzero(s == st)
        n = int(idx.size)
        if n == 0:
            continue
        k = int(math.ceil((1.0 - float(q)) * n))
        if k <= 0:
            continue
        st_scores = score[idx]
        order = np.argsort(st_scores, kind="mergesort")
        if k >= n:
            if x is not None:
                out[idx] = x[idx] > 0
            else:
                out[idx] = True
            continue
        cutoff = st_scores[order[-k]]
        gate_idx = idx[st_scores >= cutoff]
        if x is not None:
            gate_idx = gate_idx[x[gate_idx] > 0]
        out[gate_idx] = True
    return out


@dataclass(frozen=True)
class Coef:
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
) -> Coef:
    """Fit binomial GLM with stratum fixed effects using aggregated rows.

    Model: logit(p) = alpha_stratum + X * beta, where alpha is a full set of stratum dummies.
    This is equivalent to individual-level logistic with stratum intercepts, but much faster.
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
        return Coef(
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
    n_strata = int(strata.max()) + 1
    tot_by_s = np.bincount(strata, weights=y_tot, minlength=n_strata)
    succ_by_s = np.bincount(strata, weights=y_succ, minlength=n_strata)
    keep_s = (succ_by_s > 0) & (succ_by_s < tot_by_s)
    keep2 = keep_s[strata]
    if not np.any(keep2):
        p = int(X.shape[1])
        return Coef(
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

    # Build one-hot for strata (no intercept). For large numbers of strata (e.g. when adding
    # spatial bins), dense one-hot explodes memory/time, so switch to sparse CSR.
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
        return Coef(
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
    return Coef(beta=params, cov=cov, fail_reason=None, n_rows=int(y_tot.size), n_strata=int(n_s))


def _meta_fixed(effect: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    if not np.any(m):
        return float("nan"), float("nan")
    w = 1.0 / (se[m] ** 2)
    mu = float(np.sum(w * effect[m]) / np.sum(w))
    se_mu = float(np.sqrt(1.0 / np.sum(w)))
    return mu, se_mu


def _meta_random_dl(effect: np.ndarray, se: np.ndarray) -> tuple[float, float, float]:
    m = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    if np.sum(m) < 2:
        mu, se_mu = _meta_fixed(effect, se)
        return mu, se_mu, 0.0
    e = effect[m]
    v = se[m] ** 2
    w = 1.0 / v
    mu_fixed = float(np.sum(w * e) / np.sum(w))
    Q = float(np.sum(w * (e - mu_fixed) ** 2))
    df = float(e.size - 1)
    C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w2 = 1.0 / (v + tau2)
    mu = float(np.sum(w2 * e) / np.sum(w2))
    se_mu = float(np.sqrt(1.0 / np.sum(w2)))
    return mu, se_mu, float(tau2)


def _meta_i2(effect: np.ndarray, se: np.ndarray) -> float:
    """I^2 heterogeneity estimate for fixed-effect meta-analysis."""
    m = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    if np.sum(m) < 2:
        return float("nan")
    e = effect[m]
    v = se[m] ** 2
    w = 1.0 / v
    mu = float(np.sum(w * e) / np.sum(w))
    Q = float(np.sum(w * (e - mu) ** 2))
    if Q <= 0.0:
        return 0.0
    df = float(e.size - 1)
    return float(max(0.0, (Q - df) / Q))


def _bootstrap_animals_mean(values_by_animal: dict[str, float], *, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    vals = np.array(list(values_by_animal.values()), dtype=float)
    m = np.isfinite(vals)
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")
    vals = vals[m]
    point = float(np.mean(vals))
    boots = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        samp = rng.choice(vals, size=vals.size, replace=True)
        boots[i] = float(np.mean(samp))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit-fix v2 (fast): stratum-fixed-effects binomial GLM on aggregated counts.\n"
            "Avoids per-stratum continuity correction by fitting E~G+B+G*B with stratum intercepts (dataset×theta×leiden)."
        )
    )
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    ap.add_argument(
        "--genes-csv",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/brdu_persistence_auditfix_v1/auditfix_effects_sorted.csv"),
    )
    ap.add_argument("--n-genes", type=int, default=30)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--pearson-theta", type=float, default=100.0, help="Pearson residual theta (NB overdispersion).")
    ap.add_argument(
        "--pearson-clip",
        type=str,
        default="10.0",
        help="Clip Pearson residuals to +/-clip; use 'none' to disable clipping.",
    )
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument("--leiden", type=str, default="6,14,15")
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument("--exclude-genes", type=str, default="Cnksr2")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=0, help="If >0, print progress every N genes.")
    ap.add_argument(
        "--spatial-bin-um",
        type=float,
        default=0.0,
        help=(
            "If >0, include an additional per-dataset spatial bin in strata for the interaction model "
            "(dataset×theta×leiden×spatial_bin), using AP for Sag datasets and ML for Coro datasets "
            "under the existing AP/ML swap convention."
        ),
    )
    ap.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/brdu_persistence_auditfix_v2_stratum_glm"),
    )
    args = ap.parse_args()

    import anndata as ad

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    leiden_keep = {int(x) for x in str(args.leiden).split(",") if str(x).strip()}

    genes_df = pd.read_csv(args.genes_csv)
    genes = genes_df["gene"].astype(str).tolist()[: int(args.n_genes)]
    exclude = {g.strip() for g in str(args.exclude_genes).split(",") if g.strip()}
    genes = [g for g in genes if g not in exclude]

    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs

    lei_all = obs["leiden"].to_numpy().astype(int, copy=False)
    idx = np.isin(lei_all, np.array(sorted(leiden_keep), dtype=int))
    if not np.any(idx):
        raise ValueError(f"No cells found for leiden in {sorted(leiden_keep)}")

    obs_name = adata.obs_names.to_numpy(dtype=object, copy=False)[idx]
    dataset = obs.loc[idx, "dataset"].astype(str).to_numpy()
    animal = np.array([_dataset_animal(d) for d in dataset], dtype=object)

    B = obs.loc[idx, "brdu_pos"].to_numpy().astype(int, copy=False)
    E = obs.loc[idx, "edu_pos"].to_numpy().astype(int, copy=False)
    leiden = lei_all[idx]

    if "total_counts" not in obs.columns:
        raise ValueError("obs['total_counts'] not found; required for depth residualization.")
    total_counts = obs.loc[idx, "total_counts"].to_numpy().astype(float, copy=False)
    log_total = np.log1p(total_counts)

    # Codes.
    ds_levels = np.unique(dataset)
    ds_to_code = {d: i for i, d in enumerate(ds_levels.tolist())}
    ds_code = np.array([ds_to_code[d] for d in dataset], dtype=int)

    lei_levels = np.array(sorted(leiden_keep), dtype=int)
    lei_to_code = {int(lei): i for i, lei in enumerate(lei_levels.tolist())}
    lei_code = np.array([lei_to_code[int(lei)] for lei in leiden], dtype=int)

    trc = _load_tricycle_ref(args.tricycle_ref_csv)
    theta = _compute_tricycle_theta(adata, np.flatnonzero(idx), trc, dataset)
    theta_bin = _theta_bins_within_dataset(theta, dataset, int(args.theta_bins)).astype(int, copy=False)

    n_theta = int(args.theta_bins)
    # Gate strata are dataset-only (no theta/leiden): we do NOT force a fixed gate fraction within each
    # dataset×theta_bin×leiden slice, because that manufactures positives in bins where the gene is off
    # (a real artifact for zero-inflated targeted panels). Phase/type control belongs in model strata.
    stratum_gate = ds_code

    if float(args.spatial_bin_um) > 0.0:
        if "AP_ML_um" not in adata.obsm:
            raise ValueError("Expected obsm['AP_ML_um'] for spatial binning.")

        xy = np.asarray(adata.obsm["AP_ML_um"][idx], dtype=float)
        ap_um = xy[:, 0]
        ml_um = xy[:, 1]

        ds_lower = np.char.lower(dataset.astype(str))
        is_sag = np.char.find(ds_lower, "sag") >= 0
        is_coro = np.char.find(ds_lower, "coro") >= 0
        if not (np.all(is_sag | is_coro)):
            bad = dataset[~(is_sag | is_coro)]
            raise ValueError(f"Cannot infer orientation for spatial binning from dataset strings: e.g. {bad[:3]}")

        spatial_axis = np.where(is_sag, ap_um, ml_um)
        spatial_bin = np.floor(spatial_axis / float(args.spatial_bin_um)).astype(int)
        keys = pd.Series(
            [
                f"{d}|{t}|{lei}|{sb}"
                for d, t, lei, sb in zip(
                    ds_code.tolist(), theta_bin.tolist(), lei_code.tolist(), spatial_bin.tolist(), strict=True
                )
            ],
            copy=False,
        )
        stratum_model = pd.factorize(keys, sort=True)[0].astype(int)
    else:
        stratum_model = stratum_gate

    # Stable tie-breaker jitter.
    h = _stable_hash_u64(dataset, obs_name)
    jitter = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)) - 0.5
    eps = 1e-9

    animal_levels = np.unique(animal)
    animal_mask = {a: (animal == a) for a in animal_levels.tolist()}

    rows: list[dict[str, object]] = []
    per_animal_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    idx_i = np.flatnonzero(idx)
    t0 = perf_counter()
    for gene in genes:
        if int(args.progress_every) > 0 and (len(rows) % int(args.progress_every) == 0):
            dt = perf_counter() - t0
            print(f"[progress] {len(rows)}/{len(genes)} genes done in {dt:.1f}s; next={gene}", flush=True)

        x_raw = _raw_count_for_gene(adata, idx_i, gene)
        x_log1p = np.log1p(x_raw)
        clip = str(args.pearson_clip).strip().lower()
        x_resid = _pearson_residualize_by_dataset(
            x_raw,
            total_counts,
            dataset,
            theta=float(args.pearson_theta),
            clip=None if clip in {"none", "nan"} else float(args.pearson_clip),
        )
        score = x_resid + eps * jitter
        G = _gate_top_fraction_within_strata(score, stratum_gate, q=float(args.q))

        # Depth diagnostic within strata: mean log_total difference between gate+ and gate- per stratum.
        n_strata_global = int(stratum_gate.max()) + 1
        g1 = G.astype(float, copy=False)
        g0 = 1.0 - g1
        lt = log_total.astype(float, copy=False)
        sum_lt_g1 = np.bincount(stratum_gate, weights=lt * g1, minlength=n_strata_global)
        sum_lt_g0 = np.bincount(stratum_gate, weights=lt * g0, minlength=n_strata_global)
        n_g1 = np.bincount(stratum_gate, weights=g1, minlength=n_strata_global)
        n_g0 = np.bincount(stratum_gate, weights=g0, minlength=n_strata_global)
        keep_str = (n_g1 > 0) & (n_g0 > 0)
        diff = (sum_lt_g1[keep_str] / n_g1[keep_str]) - (sum_lt_g0[keep_str] / n_g0[keep_str])

        diag_rows.append(
            {
                "gene": gene,
                "q": float(args.q),
                "detect_frac_all": float(np.mean(x_log1p > 0.0)),
                "gate_frac_all": float(np.mean(G)),
                "frac_gate_raw0_all": float(np.mean(x_log1p[G] == 0.0)) if np.any(G) else float("nan"),
                "frac_gate_raw0_B1": float(np.mean(x_log1p[G & (B == 1)] == 0.0))
                if np.any(G & (B == 1))
                else float("nan"),
                "frac_gate_raw0_B0": float(np.mean(x_log1p[G & (B == 0)] == 0.0))
                if np.any(G & (B == 0))
                else float("nan"),
                "corr_gate_logtotal_all": float(np.corrcoef(G.astype(float), log_total.astype(float))[0, 1])
                if (np.std(G) > 0 and np.std(log_total) > 0)
                else float("nan"),
                "mean_logtotal_gate1_minus_gate0_by_stratum": float(np.mean(diff)) if diff.size else float("nan"),
                "median_logtotal_gate1_minus_gate0_by_stratum": float(np.median(diff)) if diff.size else float("nan"),
            }
        )

        # Per animal: fit interaction model via aggregated GLM.
        logIOR_by_a: dict[str, float] = {}
        se_logIOR_by_a: dict[str, float] = {}
        logOR_B1_by_a: dict[str, float] = {}
        se_logOR_B1_by_a: dict[str, float] = {}
        logOR_B0_by_a: dict[str, float] = {}
        se_logOR_B0_by_a: dict[str, float] = {}
        support_by_a: dict[str, dict[str, float]] = {}

        for a in animal_levels.tolist():
            m = animal_mask[a]

            # Aggregate over stratum×g×b counts of E.
            g = G[m].astype(int)
            b = B[m].astype(int)
            e = E[m].astype(int)
            s = stratum_model[m].astype(int)

            # Table index: (stratum, g, b) -> counts of success and total.
            key = s * 4 + g * 2 + b  # 0..(n_strata*4-1)
            n_s_max = int(stratum_model.max() + 1)
            n_tot = np.bincount(key, minlength=n_s_max * 4).astype(np.float64, copy=False)
            n_succ = np.bincount(key, weights=e, minlength=n_s_max * 4).astype(np.float64, copy=False)

            nt = n_tot.reshape((-1, 4))
            n_b0 = nt[:, 0] + nt[:, 2]
            n_b1 = nt[:, 1] + nt[:, 3]
            n_b0g0 = nt[:, 0]
            n_b0g1 = nt[:, 2]
            n_b1g0 = nt[:, 1]
            n_b1g1 = nt[:, 3]
            strata_any = (n_b0 + n_b1) > 0
            strata_bothB = (n_b0 > 0) & (n_b1 > 0)
            strata_all_BG = (n_b0g0 > 0) & (n_b0g1 > 0) & (n_b1g0 > 0) & (n_b1g1 > 0)
            support_by_a[str(a)] = {
                "n_strata_any": float(np.sum(strata_any)),
                "n_strata_bothB": float(np.sum(strata_bothB)),
                "n_strata_all_BG": float(np.sum(strata_all_BG)),
                "n_B0": float(n_b0.sum()),
                "n_B1": float(n_b1.sum()),
                "n_B0G0": float(n_b0g0.sum()),
                "n_B0G1": float(n_b0g1.sum()),
                "n_B1G0": float(n_b1g0.sum()),
                "n_B1G1": float(n_b1g1.sum()),
            }

            # Expand to rows for GLM.
            nonzero = n_tot > 0
            if not np.any(nonzero):
                per_animal_rows.append(
                    {
                        "gene": gene,
                        "q": float(args.q),
                        "animal": str(a),
                        "model": "interaction",
                        "fail_reason": "no_rows",
                        "n_rows": 0,
                        "n_strata": 0,
                    }
                )
                continue

            kk = np.flatnonzero(nonzero)
            strata_row = (kk // 4).astype(int)
            gb = (kk % 4).astype(int)
            g_row = (gb // 2).astype(int)
            b_row = (gb % 2).astype(int)

            Xpred = np.column_stack([g_row.astype(float), b_row.astype(float), (g_row * b_row).astype(float)])
            c = _fit_stratified_glm_binomial(
                y_succ=n_succ[kk],
                y_tot=n_tot[kk],
                strata=strata_row,
                X=Xpred,
            )

            if c.fail_reason is None and c.beta.shape == (3,) and c.cov.shape == (3, 3):
                beta_g = float(c.beta[0])
                beta_gb = float(c.beta[2])
                se_gb = float(np.sqrt(c.cov[2, 2]))
                se_b1 = float(np.sqrt(c.cov[0, 0] + c.cov[2, 2] + 2.0 * c.cov[0, 2]))
                se_b0 = float(np.sqrt(c.cov[0, 0]))

                logIOR_by_a[str(a)] = beta_gb
                se_logIOR_by_a[str(a)] = se_gb
                logOR_B1_by_a[str(a)] = beta_g + beta_gb
                se_logOR_B1_by_a[str(a)] = se_b1
                logOR_B0_by_a[str(a)] = beta_g
                se_logOR_B0_by_a[str(a)] = se_b0
            else:
                logIOR_by_a[str(a)] = float("nan")
                se_logIOR_by_a[str(a)] = float("nan")
                logOR_B1_by_a[str(a)] = float("nan")
                se_logOR_B1_by_a[str(a)] = float("nan")
                logOR_B0_by_a[str(a)] = float("nan")
                se_logOR_B0_by_a[str(a)] = float("nan")

            per_animal_rows.append(
                {
                    "gene": gene,
                    "q": float(args.q),
                    "animal": str(a),
                    "model": "interaction",
                    "fail_reason": c.fail_reason,
                    "logIOR": logIOR_by_a[str(a)],
                    "se_logIOR": se_logIOR_by_a[str(a)],
                    "logOR_GE_B1": logOR_B1_by_a[str(a)],
                    "se_logOR_GE_B1": se_logOR_B1_by_a[str(a)],
                    "logOR_GE_B0": logOR_B0_by_a[str(a)],
                    "se_logOR_GE_B0": se_logOR_B0_by_a[str(a)],
                    **support_by_a.get(str(a), {}),
                    "n_rows": int(c.n_rows),
                    "n_strata": int(c.n_strata),
                }
            )

        # Meta across animals.
        ea_logior, ea_lo, ea_hi = _bootstrap_animals_mean(logIOR_by_a, n_boot=int(args.n_bootstrap), rng=rng)
        ea_logor_b1, ea_lo_b1, ea_hi_b1 = _bootstrap_animals_mean(logOR_B1_by_a, n_boot=int(args.n_bootstrap), rng=rng)
        ea_logor_b0, ea_lo_b0, ea_hi_b0 = _bootstrap_animals_mean(logOR_B0_by_a, n_boot=int(args.n_bootstrap), rng=rng)

        iv_logior, iv_se_logior = _meta_fixed(np.array(list(logIOR_by_a.values())), np.array(list(se_logIOR_by_a.values())))
        iv_logor_b1, iv_se_logor_b1 = _meta_fixed(np.array(list(logOR_B1_by_a.values())), np.array(list(se_logOR_B1_by_a.values())))
        iv_logor_b0, iv_se_logor_b0 = _meta_fixed(np.array(list(logOR_B0_by_a.values())), np.array(list(se_logOR_B0_by_a.values())))
        re_logior, re_se_logior, re_tau2 = _meta_random_dl(np.array(list(logIOR_by_a.values())), np.array(list(se_logIOR_by_a.values())))
        re_logor_b1, re_se_logor_b1, re_tau2_b1 = _meta_random_dl(np.array(list(logOR_B1_by_a.values())), np.array(list(se_logOR_B1_by_a.values())))
        re_logor_b0, re_se_logor_b0, re_tau2_b0 = _meta_random_dl(np.array(list(logOR_B0_by_a.values())), np.array(list(se_logOR_B0_by_a.values())))

        n_animals_logior = int(np.sum(np.isfinite(np.array(list(logIOR_by_a.values()), dtype=float))))
        i2_logior = _meta_i2(np.array(list(logIOR_by_a.values())), np.array(list(se_logIOR_by_a.values())))

        row: dict[str, object] = {
            "gene": gene,
            "q": float(args.q),
            "n_animals_logIOR": n_animals_logior,
            "logOR_GE_B1_EA": ea_logor_b1,
            "logOR_GE_B1_EA_ci_low": ea_lo_b1,
            "logOR_GE_B1_EA_ci_high": ea_hi_b1,
            "OR_GE_B1_EA": float(math.exp(ea_logor_b1)) if math.isfinite(ea_logor_b1) else float("nan"),
            "OR_GE_B1_EA_ci_low": float(math.exp(ea_lo_b1)) if math.isfinite(ea_lo_b1) else float("nan"),
            "OR_GE_B1_EA_ci_high": float(math.exp(ea_hi_b1)) if math.isfinite(ea_hi_b1) else float("nan"),
            "logOR_GE_B0_EA": ea_logor_b0,
            "logOR_GE_B0_EA_ci_low": ea_lo_b0,
            "logOR_GE_B0_EA_ci_high": ea_hi_b0,
            "OR_GE_B0_EA": float(math.exp(ea_logor_b0)) if math.isfinite(ea_logor_b0) else float("nan"),
            "OR_GE_B0_EA_ci_low": float(math.exp(ea_lo_b0)) if math.isfinite(ea_lo_b0) else float("nan"),
            "OR_GE_B0_EA_ci_high": float(math.exp(ea_hi_b0)) if math.isfinite(ea_hi_b0) else float("nan"),
            "logIOR_EA": ea_logior,
            "logIOR_EA_ci_low": ea_lo,
            "logIOR_EA_ci_high": ea_hi,
            "IOR_EA": float(math.exp(ea_logior)) if math.isfinite(ea_logior) else float("nan"),
            "IOR_EA_ci_low": float(math.exp(ea_lo)) if math.isfinite(ea_lo) else float("nan"),
            "IOR_EA_ci_high": float(math.exp(ea_hi)) if math.isfinite(ea_hi) else float("nan"),
            "logOR_GE_B1_IV": iv_logor_b1,
            "se_logOR_GE_B1_IV": iv_se_logor_b1,
            "logOR_GE_B0_IV": iv_logor_b0,
            "se_logOR_GE_B0_IV": iv_se_logor_b0,
            "logIOR_IV": iv_logior,
            "se_logIOR_IV": iv_se_logior,
            "logOR_GE_B1_RE": re_logor_b1,
            "se_logOR_GE_B1_RE": re_se_logor_b1,
            "tau2_logOR_GE_B1_RE": re_tau2_b1,
            "logOR_GE_B0_RE": re_logor_b0,
            "se_logOR_GE_B0_RE": re_se_logor_b0,
            "tau2_logOR_GE_B0_RE": re_tau2_b0,
            "logIOR_RE": re_logior,
            "se_logIOR_RE": re_se_logior,
            "tau2_logIOR_RE": re_tau2,
            "I2_logIOR_IV": i2_logior,
            "delta_EA_minus_IV_logIOR": float(ea_logior - iv_logior) if (math.isfinite(ea_logior) and math.isfinite(iv_logior)) else float("nan"),
        }

        # Per-leiden BrdU+ OR: E~G with strata(dataset×theta) (within each animal), then animal-meta.
        per_lei_animal: dict[int, dict[str, float]] = {int(lei): {} for lei in lei_levels.tolist()}
        per_lei_animal_se: dict[int, dict[str, float]] = {int(lei): {} for lei in lei_levels.tolist()}
        for lei in lei_levels.tolist():
            lei = int(lei)
            for a in animal_levels.tolist():
                m = animal_mask[a] & (B == 1) & (leiden == lei)
                if not np.any(m):
                    per_lei_animal[lei][str(a)] = float("nan")
                    per_lei_animal_se[lei][str(a)] = float("nan")
                    continue
                st2 = ds_code[m] * n_theta + theta_bin[m]
                # Aggregate (st2, g) -> counts of E
                key2 = st2.astype(int) * 2 + G[m].astype(int)
                n_tot2 = np.bincount(key2, minlength=int(st2.max() + 1) * 2).astype(np.float64, copy=False)
                n_succ2 = np.bincount(key2, weights=E[m], minlength=int(st2.max() + 1) * 2).astype(np.float64, copy=False)
                nonzero = n_tot2 > 0
                if not np.any(nonzero):
                    per_lei_animal[lei][str(a)] = float("nan")
                    per_lei_animal_se[lei][str(a)] = float("nan")
                    continue
                kk = np.flatnonzero(nonzero)
                strata_row = (kk // 2).astype(int)
                g_row = (kk % 2).astype(float)
                c = _fit_stratified_glm_binomial(y_succ=n_succ2[kk], y_tot=n_tot2[kk], strata=strata_row, X=g_row)
                if c.fail_reason is None:
                    per_lei_animal[lei][str(a)] = float(c.beta[0])
                    per_lei_animal_se[lei][str(a)] = float(np.sqrt(c.cov[0, 0]))
                else:
                    per_lei_animal[lei][str(a)] = float("nan")
                    per_lei_animal_se[lei][str(a)] = float("nan")

            ea, lo, hi = _bootstrap_animals_mean(per_lei_animal[lei], n_boot=int(args.n_bootstrap), rng=rng)
            iv, iv_se = _meta_fixed(np.array(list(per_lei_animal[lei].values())), np.array(list(per_lei_animal_se[lei].values())))
            row[f"logOR_B1_L{lei}_EA"] = ea
            row[f"logOR_B1_L{lei}_EA_ci_low"] = lo
            row[f"logOR_B1_L{lei}_EA_ci_high"] = hi
            row[f"OR_B1_L{lei}_EA"] = float(math.exp(ea)) if math.isfinite(ea) else float("nan")
            row[f"OR_B1_L{lei}_EA_ci_low"] = float(math.exp(lo)) if math.isfinite(lo) else float("nan")
            row[f"OR_B1_L{lei}_EA_ci_high"] = float(math.exp(hi)) if math.isfinite(hi) else float("nan")
            row[f"logOR_B1_L{lei}_IV"] = iv
            row[f"se_logOR_B1_L{lei}_IV"] = iv_se

        # Per-leiden IOR: E~G+B+G*B within leiden (strata dataset×theta), then animal-meta.
        for lei in lei_levels.tolist():
            lei = int(lei)
            logior_by_a = {}
            se_by_a = {}
            for a in animal_levels.tolist():
                m = animal_mask[a] & (leiden == lei)
                if not np.any(m):
                    logior_by_a[str(a)] = float("nan")
                    se_by_a[str(a)] = float("nan")
                    continue
                st2 = ds_code[m] * n_theta + theta_bin[m]
                key2 = st2.astype(int) * 4 + G[m].astype(int) * 2 + B[m].astype(int)
                n_tot2 = np.bincount(key2, minlength=int(st2.max() + 1) * 4).astype(np.float64, copy=False)
                n_succ2 = np.bincount(key2, weights=E[m], minlength=int(st2.max() + 1) * 4).astype(np.float64, copy=False)
                nonzero = n_tot2 > 0
                if not np.any(nonzero):
                    logior_by_a[str(a)] = float("nan")
                    se_by_a[str(a)] = float("nan")
                    continue
                kk = np.flatnonzero(nonzero)
                strata_row = (kk // 4).astype(int)
                gb = (kk % 4).astype(int)
                g_row = (gb // 2).astype(int)
                b_row = (gb % 2).astype(int)
                Xpred = np.column_stack([g_row.astype(float), b_row.astype(float), (g_row * b_row).astype(float)])
                c = _fit_stratified_glm_binomial(y_succ=n_succ2[kk], y_tot=n_tot2[kk], strata=strata_row, X=Xpred)
                if c.fail_reason is None and c.beta.shape == (3,) and c.cov.shape == (3, 3):
                    beta_gb = float(c.beta[2])
                    se_gb = float(np.sqrt(c.cov[2, 2]))
                    logior_by_a[str(a)] = beta_gb
                    se_by_a[str(a)] = se_gb
                else:
                    logior_by_a[str(a)] = float("nan")
                    se_by_a[str(a)] = float("nan")

            ea, lo, hi = _bootstrap_animals_mean(logior_by_a, n_boot=int(args.n_bootstrap), rng=rng)
            row[f"logIOR_L{lei}_EA"] = ea
            row[f"logIOR_L{lei}_EA_ci_low"] = lo
            row[f"logIOR_L{lei}_EA_ci_high"] = hi
            row[f"IOR_L{lei}_EA"] = float(math.exp(ea)) if math.isfinite(ea) else float("nan")
            row[f"IOR_L{lei}_EA_ci_low"] = float(math.exp(lo)) if math.isfinite(lo) else float("nan")
            row[f"IOR_L{lei}_EA_ci_high"] = float(math.exp(hi)) if math.isfinite(hi) else float("nan")

        # Equal-leiden IOR summary (geometric mean of per-leiden IORs on log scale).
        l6 = float(row.get("logIOR_L6_EA", float("nan")))
        l14 = float(row.get("logIOR_L14_EA", float("nan")))
        l15 = float(row.get("logIOR_L15_EA", float("nan")))
        if all(math.isfinite(v) for v in [l6, l14, l15]):
            row["logIOR_equal_leiden_EA"] = float(np.mean([l6, l14, l15]))
            row["IOR_equal_leiden_EA"] = float(math.exp(row["logIOR_equal_leiden_EA"]))
        else:
            row["logIOR_equal_leiden_EA"] = float("nan")
            row["IOR_equal_leiden_EA"] = float("nan")

        # Equal-leiden summary (computed on the EA per-leiden logORs, with animal bootstrap below).
        l6 = float(row.get("logOR_B1_L6_EA", float("nan")))
        l14 = float(row.get("logOR_B1_L14_EA", float("nan")))
        l15 = float(row.get("logOR_B1_L15_EA", float("nan")))
        if all(math.isfinite(v) for v in [l6, l14, l15]):
            row["logOR_B1_equal_leiden_EA"] = float(np.mean([l6, l14, l15]))
            row["OR_B1_equal_leiden_EA"] = float(math.exp(row["logOR_B1_equal_leiden_EA"]))
        else:
            row["logOR_B1_equal_leiden_EA"] = float("nan")
            row["OR_B1_equal_leiden_EA"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "auditfix2_effects_stratum_glm.csv", index=False)
    pd.DataFrame(per_animal_rows).to_csv(outdir / "auditfix2_per_animal_effects_stratum_glm.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(outdir / "auditfix2_gate_diagnostics.csv", index=False)

    out2 = out.copy()
    out2["abs_logIOR_EA"] = out2["logIOR_EA"].abs()
    out2 = out2.sort_values(["abs_logIOR_EA", "gene"], ascending=[False, True]).drop(columns=["abs_logIOR_EA"])
    out2.to_csv(outdir / "auditfix2_effects_stratum_glm_sorted.csv", index=False)


if __name__ == "__main__":
    main()
