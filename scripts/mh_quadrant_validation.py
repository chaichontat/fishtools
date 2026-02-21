#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

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


def _compute_tricycle_theta(adata, trc: pd.DataFrame, dataset: np.ndarray) -> np.ndarray:
    """Compute theta for all rows using sparse-safe centering.

    theta = atan2(pc2, pc1) where pc1/pc2 are projections onto tricycle ref loadings,
    after within-dataset mean-centering on the ref genes.
    """
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


def _theta_bins(theta: np.ndarray, dataset: np.ndarray, *, n_bins: int, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "quantile":
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
    if mode == "angle":
        # Fixed absolute angular bins across datasets (theta on (-pi, pi]).
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
    """Compute Pearson residuals per gene within each dataset.

    Per-dataset, blockwise adaptation of Scanpy's `normalize_pearson_residuals`
    for NB counts (theta overdispersion).
    """
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
        p = (sum_x / denom).astype(np.float32, copy=False)  # (n_genes,)

        for start in range(0, n_genes, int(block_size)):
            end = min(start + int(block_size), n_genes)
            pp = p[start:end].astype(np.float32, copy=False)
            mu = tt[:, None] * pp[None, :]  # (n, b)
            var = mu + (mu * mu) / float(theta)
            var = np.maximum(var, 1e-12, dtype=np.float32)
            r = (Xd[:, start:end] - mu) / np.sqrt(var)
            if clip is not None:
                r = np.clip(r, -float(clip), float(clip))
            out[sel, start:end] = r.astype(np.float32, copy=False)

    return out


def _residualize_matrix_within_dataset(X: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    R = np.empty_like(X, dtype=np.float32)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m), dtype=np.float32), sin_t[m], cos_t[m]]).astype(np.float32, copy=False)
        # Solve for all genes at once.
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

            for col in j.tolist():
                vals = Rd[:, int(col)].astype(np.float64, copy=False) + eps * jd
                cut = np.argpartition(vals, n - k)[n - k :]
                G[rows[cut], int(col)] = True
    return G


@dataclass(frozen=True)
class Mh:
    delta: float
    lo: float
    hi: float

    @property
    def Ts_fold(self) -> float:
        return float(math.exp(self.delta)) if math.isfinite(self.delta) else float("nan")

    @property
    def Ts_lo(self) -> float:
        return float(math.exp(self.lo)) if math.isfinite(self.lo) else float("nan")

    @property
    def Ts_hi(self) -> float:
        return float(math.exp(self.hi)) if math.isfinite(self.hi) else float("nan")

@dataclass(frozen=True)
class MhDsContrib:
    delta: float
    R_by_ds: np.ndarray
    S_by_ds: np.ndarray


def _mh_bootstrap(
    stratum_idx: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    *,
    n_strata: int,
    n_datasets: int,
    n_leiden: int,
    theta_bins: int,
    cc: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Mh:
    # a,b,c,d per stratum (no cc)
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
        return Mh(delta=float("nan"), lo=float("nan"), hi=float("nan"))

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
        return Mh(delta=float("nan"), lo=float("nan"), hi=float("nan"))

    delta = float(math.log(R_total / S_total))

    stratum_ids = np.arange(int(n_strata), dtype=int)[keep]
    ds_code = (stratum_ids // (int(theta_bins) * int(n_leiden))).astype(int, copy=False)
    R_by_ds = np.bincount(ds_code, weights=r, minlength=int(n_datasets)).astype(np.float64, copy=False)
    S_by_ds = np.bincount(ds_code, weights=s, minlength=int(n_datasets)).astype(np.float64, copy=False)

    boots = np.empty(int(n_boot), dtype=np.float64)
    present = (R_by_ds + S_by_ds) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
        return Mh(delta=float("nan"), lo=float("nan"), hi=float("nan"))
    for i in range(int(n_boot)):
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        Rb = float(R_by_ds[samp].sum())
        Sb = float(S_by_ds[samp].sum())
        boots[i] = float(math.log(Rb / Sb))

    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return Mh(delta=delta, lo=float(lo), hi=float(hi))


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
    # a,b,c,d per stratum (no cc)
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


def _bootstrap_ci_from_ds_contrib(
    contrib: MhDsContrib, *, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    present = (contrib.R_by_ds + contrib.S_by_ds) > 0
    ds_present = np.flatnonzero(present)
    if ds_present.size == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(ds_present, size=ds_present.size, replace=True)
        Rb = float(contrib.R_by_ds[samp].sum())
        Sb = float(contrib.S_by_ds[samp].sum())
        boots[i] = float(math.log(Rb / Sb))
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return (float(lo), float(hi))


def _bootstrap_ci_for_delta_diff(
    c1: MhDsContrib, c2: MhDsContrib, *, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    # Bootstrap datasets with shared resamples to preserve covariance between the two deltas.
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
    # Any missing group would be a bug (dataset/leiden code combo should exist in edges).
    if np.any(out < 0):
        raise RuntimeError("Some theta bins were not assigned (missing dataset×leiden edges).")
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description="MH validation using all four BrdU/EdU quadrants.")
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    ap.add_argument(
        "--genes-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2/mh_phase_consistency_top_genes.csv"
        ),
        help="CSV with columns gene,q.",
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
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a_phase_matched/quadrant_validation_no_cnksr2"),
    )
    args = ap.parse_args()

    import anndata as ad
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(args.seed))
    args.outdir.mkdir(parents=True, exist_ok=True)

    clip_s = str(args.pearson_clip).strip().lower()
    pearson_clip: float | None = None if clip_s in {"none", "nan"} else float(args.pearson_clip)

    genes_df = pd.read_csv(args.genes_csv)
    if "gene" not in genes_df.columns or "q" not in genes_df.columns:
        raise ValueError("--genes-csv must have columns: gene,q")

    genes = genes_df["gene"].astype(str).tolist()
    q_by_gene = genes_df["q"].astype(float).to_numpy()

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
    n_datasets = int(ds_uniq.size)
    n_leiden = int(lei_uniq.size)

    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    theta = _compute_tricycle_theta(adata, trc=trc, dataset=dataset)
    n_strata = n_datasets * int(args.theta_bins) * n_leiden

    if str(args.theta_bin_mode) == "quantile":
        # Shared theta-bin edges across all endpoints (per dataset×leiden), matching legacy behavior.
        edges = _theta_edges_by_dataset_leiden(theta, ds_code, lei_code, n_bins=int(args.theta_bins))
        theta_bin_shared = _assign_theta_bins_from_edges(theta, ds_code, lei_code, edges, n_bins=int(args.theta_bins))
    else:
        # Fixed absolute angular bins: ensure "theta_bin=k" refers to the same geometric interval across datasets.
        t = (theta.astype(np.float64, copy=False) + (2.0 * np.pi)) % (2.0 * np.pi)
        w = (2.0 * np.pi) / float(int(args.theta_bins))
        theta_bin_shared = np.clip(np.floor(t / w), 0, int(args.theta_bins) - 1).astype(np.int16, copy=False)

    stratum_idx_all = (ds_code * int(args.theta_bins) + theta_bin_shared.astype(int)) * n_leiden + lei_code

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
    endpoints = [
        ("BrdUplus_Edu", brdu == 1, edu),  # A vs B within BrdU+
        ("Eduplus_BrdU", edu == 1, brdu),  # A vs C within EdU+
        ("BrdUminus_Edu", brdu == 0, edu),  # C vs D within BrdU-
        ("EdUminus_BrdU", edu == 0, brdu),  # B vs D within EdU-
    ]

    rows: list[dict[str, object]] = []
    for j, g in enumerate(genes):
        G = Gmat[:, j]
        row: dict[str, object] = {"gene": g, "q": float(q_by_gene[j])}
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
            lo, hi = _bootstrap_ci_from_ds_contrib(c, rng=rng, n_boot=int(args.n_bootstrap))
            mh = Mh(delta=float(c.delta), lo=float(lo), hi=float(hi))
            contribs[name] = c
            row[f"delta_{name}"] = float(mh.delta)
            row[f"ci_low_{name}"] = float(mh.lo)
            row[f"ci_high_{name}"] = float(mh.hi)
            row[f"Ts_fold_{name}"] = float(mh.Ts_fold)
            row[f"Ts_ci_low_{name}"] = float(mh.Ts_lo)
            row[f"Ts_ci_high_{name}"] = float(mh.Ts_hi)

        # Three-way interaction (ratio of odds ratios) estimands.
        # δ_int,E = δ(EdU | BrdU+) - δ(EdU | BrdU-)
        d_int_E, loE, hiE = _bootstrap_ci_for_delta_diff(
            contribs["BrdUplus_Edu"], contribs["BrdUminus_Edu"], rng=rng, n_boot=int(args.n_bootstrap)
        )
        row["delta_int_E"] = float(d_int_E)
        row["ci_low_int_E"] = float(loE)
        row["ci_high_int_E"] = float(hiE)
        row["Ts_fold_int_E"] = float(math.exp(d_int_E))
        row["Ts_ci_low_int_E"] = float(math.exp(loE))
        row["Ts_ci_high_int_E"] = float(math.exp(hiE))

        # δ_int,B = δ(BrdU | EdU+) - δ(BrdU | EdU-)
        d_int_B, loB, hiB = _bootstrap_ci_for_delta_diff(
            contribs["Eduplus_BrdU"], contribs["EdUminus_BrdU"], rng=rng, n_boot=int(args.n_bootstrap)
        )
        row["delta_int_B"] = float(d_int_B)
        row["ci_low_int_B"] = float(loB)
        row["ci_high_int_B"] = float(hiB)
        row["Ts_fold_int_B"] = float(math.exp(d_int_B))
        row["Ts_ci_low_int_B"] = float(math.exp(loB))
        row["Ts_ci_high_int_B"] = float(math.exp(hiB))

        # Concordance between the two persistence-like conditionals.
        s1 = math.copysign(1.0, row["delta_BrdUplus_Edu"]) if math.isfinite(row["delta_BrdUplus_Edu"]) else 0.0
        s2 = math.copysign(1.0, row["delta_Eduplus_BrdU"]) if math.isfinite(row["delta_Eduplus_BrdU"]) else 0.0
        row["mirror_sign_match"] = bool(s1 == s2) if (s1 != 0.0 and s2 != 0.0) else False
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("Ts_fold_BrdUplus_Edu", ascending=False)
    out.to_csv(args.outdir / "quadrant_validation.csv", index=False)

    # Entry vs exit balance (not gene-specific): EdU-only / BrdU-only by dataset x leiden.
    df_bal = pd.DataFrame({"dataset": dataset, "leiden": leiden, "brdu": brdu, "edu": edu})
    df_bal["is_entry"] = (df_bal["brdu"] == 0) & (df_bal["edu"] == 1)  # EdU-only
    df_bal["is_exit"] = (df_bal["brdu"] == 1) & (df_bal["edu"] == 0)  # BrdU-only

    bal = (
        df_bal.groupby(["dataset", "leiden"], observed=True)
        .agg(n_entry=("is_entry", "sum"), n_exit=("is_exit", "sum"), n_total=("brdu", "size"))
        .reset_index()
    )
    bal["entry_exit_ratio"] = bal["n_entry"] / bal["n_exit"].replace(0, np.nan)
    bal.to_csv(args.outdir / "entry_exit_balance_by_dataset_leiden.csv", index=False)

    # Overview plot: 4 panels of Ts_fold with 95% CI.
    fig, axs = plt.subplots(2, 3, figsize=(18.0, 11.0), dpi=170, sharey=True)
    axs = axs.reshape(-1)
    panels = [
        ("BrdU+ : EdU+ vs EdU-", "BrdUplus_Edu"),
        ("EdU+ : BrdU+ vs BrdU-", "Eduplus_BrdU"),
        ("BrdU- : EdU+ vs EdU-", "BrdUminus_Edu"),
        ("EdU- : BrdU+ vs BrdU-", "EdUminus_BrdU"),
        ("Interaction: δ1 − δ3 (EdU | BrdU)", "int_E"),
        ("Interaction: δ2 − δ4 (BrdU | EdU)", "int_B"),
    ]

    ytick = np.arange(out.shape[0])
    for ax, (title, key) in zip(axs, panels, strict=True):
        x = out[f"Ts_fold_{key}"].to_numpy(float)
        lo = out[f"Ts_ci_low_{key}"].to_numpy(float)
        hi = out[f"Ts_ci_high_{key}"].to_numpy(float)
        ax.hlines(ytick, lo, hi, color="black", lw=1.0)
        ax.plot(x, ytick, "o", color="black", ms=3)
        ax.axvline(1.0, color="gray", lw=1.0, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Ts_fold (OR scale)")

    axs[0].set_yticks(ytick)
    axs[0].set_yticklabels(out["gene"].astype(str).tolist())
    axs[0].invert_yaxis()

    fig.tight_layout()
    fig.savefig(args.outdir / "quadrant_overview.png")


if __name__ == "__main__":
    main()
