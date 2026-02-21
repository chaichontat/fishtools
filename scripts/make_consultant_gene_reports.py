#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


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

    Matches scripts/mh_quadrant_validation.py.
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
    return out


def _log1p_raw_for_genes(adata, genes: list[str]) -> np.ndarray:
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
    X = np.asarray(X, dtype=np.float32)
    return np.log1p(X)

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


def _residualize_matrix_within_dataset(X: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    R = np.empty_like(X, dtype=np.float32)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m), dtype=np.float32), sin_t[m], cos_t[m]]).astype(np.float32, copy=False)
        coef, *_ = np.linalg.lstsq(Z, X[m], rcond=None)  # (3, n_genes)
        R[m] = X[m] - (Z @ coef).astype(np.float32, copy=False)
    return R


def _make_gates_by_dataset_quantile(R: np.ndarray, dataset: np.ndarray, q_by_gene: np.ndarray) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    G = np.zeros(R.shape, dtype=bool)
    uniq_q = np.unique(q_by_gene)
    for d in np.unique(ds):
        m = ds == d
        rows = np.where(m)[0]
        Rd = R[m]
        for q in uniq_q:
            j = np.where(q_by_gene == q)[0]
            thr = np.quantile(Rd[:, j], q=float(q), axis=0)
            G[np.ix_(rows, j)] = Rd[:, j] >= thr
    return G


@dataclass(frozen=True)
class MhDsContrib:
    delta: float
    R_by_ds: np.ndarray
    S_by_ds: np.ndarray


def _mh_ds_contrib(*, stratum_idx: np.ndarray, G: np.ndarray, y: np.ndarray, n_strata: int, n_datasets: int, n_leiden: int, theta_bins: int, cc: float) -> MhDsContrib:
    # per-stratum counts (no cc)
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
    # This assumes stratum coding: (ds*theta_bins + theta_bin)*n_leiden + lei
    ds_code = (stratum_ids // (int(theta_bins) * int(n_leiden))).astype(int, copy=False)
    R_by_ds = np.bincount(ds_code, weights=r, minlength=int(n_datasets)).astype(np.float64, copy=False)
    S_by_ds = np.bincount(ds_code, weights=s, minlength=int(n_datasets)).astype(np.float64, copy=False)
    return MhDsContrib(delta=delta, R_by_ds=R_by_ds, S_by_ds=S_by_ds)


def _bootstrap_ci_from_ds_contrib(c: MhDsContrib, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
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


def _bootstrap_ci_for_diff(c1: MhDsContrib, c2: MhDsContrib, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    # Returns (point, lo, hi) for delta1 - delta2, resampling datasets.
    present = ((c1.R_by_ds + c1.S_by_ds) > 0) & ((c2.R_by_ds + c2.S_by_ds) > 0)
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

def _sparse_strata_fraction(*, stratum_idx: np.ndarray, G: np.ndarray, y: np.ndarray, n_strata: int) -> float:
    """Fraction of non-empty strata with any zero cell in the 2x2 table (pre-cc)."""
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


def _module_score_within_dataset_leiden_z(
    X_log1p: np.ndarray,
    dataset_code: np.ndarray,
    leiden_code: np.ndarray,
) -> np.ndarray:
    """Z-score each column within dataset×leiden; return same shape."""

    out = np.empty_like(X_log1p, dtype=np.float32)
    for ds in np.unique(dataset_code):
        for lei in np.unique(leiden_code):
            m = (dataset_code == ds) & (leiden_code == lei)
            if not np.any(m):
                continue
            Xm = X_log1p[m]
            mu = Xm.mean(axis=0)
            sd = Xm.std(axis=0, ddof=0)
            sd[sd == 0.0] = 1.0
            out[m] = ((Xm - mu) / sd).astype(np.float32, copy=False)
    return out


def _safe_qcut_bins(x: np.ndarray, q: int) -> np.ndarray:
    # qcut within group, but robust to degenerate distributions.
    # Returns bin codes 0..q-1.
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.zeros(x.shape[0], dtype=np.int16)
    if np.nanstd(x) == 0.0:
        return np.zeros(x.shape[0], dtype=np.int16)
    edges = np.nanquantile(x, q=np.linspace(0.0, 1.0, q + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], edges[i - 1] + 1.0)
    return np.digitize(x, edges[1:-1], right=False).astype(np.int16)


def _module_bins_within_dataset_leiden(module: np.ndarray, dataset_code: np.ndarray, leiden_code: np.ndarray, *, q: int) -> np.ndarray:
    out = np.empty(module.shape[0], dtype=np.int16)
    for ds in np.unique(dataset_code):
        for lei in np.unique(leiden_code):
            m = (dataset_code == ds) & (leiden_code == lei)
            if not np.any(m):
                continue
            out[m] = _safe_qcut_bins(module[m], q=q)
    return out


def _sparse_log1p_layer_raw(adata):
    import scipy.sparse as sp

    if "raw" in adata.layers:
        X = adata.layers["raw"]
    else:
        X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    X = X.tocsr(copy=True)
    X.data = np.log1p(X.data).astype(np.float32, copy=False)
    return X


def _pseudobulk_de_within_leiden(
    X_log1p_csr,
    dataset: np.ndarray,
    leiden: np.ndarray,
    gate: np.ndarray,
    *,
    leiden_value: str,
    min_cells_each: int = 200,
) -> pd.DataFrame:
    """Per-gene pseudobulk DE: gate+ vs gate- within (dataset, leiden).

    Returns meta across datasets: mean_diff, t_stat, n_datasets_used.
    """

    ds = np.asarray(dataset, dtype=object)
    lei = np.asarray(leiden, dtype=object)

    diffs = []
    ds_used = []

    for d in np.unique(ds):
        m = (ds == d) & (lei == leiden_value)
        if not np.any(m):
            continue
        idx = np.where(m)[0]
        g = gate[idx]
        if int(g.sum()) < int(min_cells_each) or int((~g).sum()) < int(min_cells_each):
            continue
        Xd = X_log1p_csr[idx]
        X_pos = Xd[g]
        X_neg = Xd[~g]
        mu_pos = np.asarray(X_pos.mean(axis=0)).reshape(-1)
        mu_neg = np.asarray(X_neg.mean(axis=0)).reshape(-1)
        diffs.append(mu_pos - mu_neg)
        ds_used.append(d)

    if not diffs:
        raise RuntimeError(f"No datasets passed min_cells_each={min_cells_each} for leiden={leiden_value}")

    D = np.vstack(diffs).astype(np.float64, copy=False)  # (n_ds, n_genes)
    mean_diff = D.mean(axis=0)
    # One-sample t-stat across datasets
    sd = D.std(axis=0, ddof=1)
    sd[sd == 0.0] = np.nan
    t_stat = mean_diff / (sd / math.sqrt(D.shape[0]))

    return pd.DataFrame(
        {
            "mean_diff": mean_diff,
            "t_stat": t_stat,
            "n_datasets_used": int(D.shape[0]),
        }
    )


def _plot_gene_summary_panel(out_png: pathlib.Path, *, gene: str, summary: dict[str, Mh], intE: Mh, intB: Mh, power: dict[str, object], module_assoc: pd.DataFrame, mediation: dict[str, Mh]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    # Endpoint effects
    names = ["E1 BrdU+->EdU", "E2 EdU+->BrdU", "E3 BrdU- ->EdU", "E4 EdU- ->BrdU", "IntE", "IntB"]
    deltas = [
        summary["BrdUplus_Edu"].delta,
        summary["Eduplus_BrdU"].delta,
        summary["BrdUminus_Edu"].delta,
        summary["EdUminus_BrdU"].delta,
        intE.delta,
        intB.delta,
    ]
    lo = [
        summary["BrdUplus_Edu"].lo,
        summary["Eduplus_BrdU"].lo,
        summary["BrdUminus_Edu"].lo,
        summary["EdUminus_BrdU"].lo,
        intE.lo,
        intB.lo,
    ]
    hi = [
        summary["BrdUplus_Edu"].hi,
        summary["Eduplus_BrdU"].hi,
        summary["BrdUminus_Edu"].hi,
        summary["EdUminus_BrdU"].hi,
        intE.hi,
        intB.hi,
    ]

    y = np.arange(len(names))
    ax0.errorbar(deltas, y, xerr=[np.array(deltas) - np.array(lo), np.array(hi) - np.array(deltas)], fmt="o", color="black")
    ax0.axvline(0.0, color="0.7", lw=1)
    ax0.set_yticks(y)
    ax0.set_yticklabels(names, fontsize=9)
    ax0.set_xlabel("log OR (delta)")
    ax0.set_title(f"{gene}: quadrant + interaction effects")

    # Power / support
    ax1.axis("off")
    txt = [
        f"Datasets: {power['n_datasets']}",
        f"BrdU+ cells: {power['n_brdUpos']}",
        f"EdU+ cells: {power['n_edUpos']}",
        f"Gate+ all: {power['n_gatepos_all']} ({power['frac_gatepos_all']:.3f})",
        f"Gate+ among BrdU+: {power['n_gatepos_brdUpos']} ({power['frac_gatepos_brdUpos']:.3f})",
        f"Gate+ BrdU+ by Leiden: 6={power['n_gatepos_brdUpos_L6']}, 14={power['n_gatepos_brdUpos_L14']}, 15={power['n_gatepos_brdUpos_L15']}",
        f"Sparse strata frac E1/E3: {power['sparse_E1']:.3f} / {power['sparse_E3']:.3f}",
    ]
    if power.get("passes_not_one_animal") is not None:
        txt.append(f"Not-one-animal pass: {power['passes_not_one_animal']}")
    ax1.text(0.0, 1.0, "\n".join(txt), va="top", family="monospace", fontsize=9)
    ax1.set_title("Support / power")

    # Module association bar
    ax2.set_title("Module association (mean z diff, gate+ - gate-)")
    top = module_assoc.sort_values("abs_diff", ascending=False).head(8)
    ax2.barh(top["module"], top["diff"], color=["#4C78A8" if v > 0 else "#F58518" for v in top["diff"]])
    ax2.axvline(0.0, color="0.7", lw=1)
    ax2.set_xlabel("Delta module score")
    ax2.invert_yaxis()

    # Mediation summary
    ax3.errorbar(
        ["baseline"] + list(mediation.keys()),
        [intE.delta] + [mediation[k].delta for k in mediation.keys()],
        yerr=[
            [intE.delta - intE.lo] + [mediation[k].delta - mediation[k].lo for k in mediation.keys()],
            [intE.hi - intE.delta] + [mediation[k].hi - mediation[k].delta for k in mediation.keys()],
        ],
        fmt="o",
        color="black",
    )
    ax3.axhline(0.0, color="0.7", lw=1)
    ax3.set_ylabel("delta_int,E")
    ax3.set_title("Mediation-style stratification: delta_int,E after adding module-bin to MH strata")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_spatial(out_png: pathlib.Path, *, gene: str, bins: np.ndarray, prev: np.ndarray, prev_lo: np.ndarray, prev_hi: np.ndarray, de: np.ndarray, de_lo: np.ndarray, de_hi: np.ndarray, axis_label: str, datasets_per_bin: np.ndarray):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True, constrained_layout=True)

    ax0.fill_between(bins, prev_lo, prev_hi, color="#4C78A8", alpha=0.25)
    ax0.plot(bins, prev, color="#4C78A8", lw=2)
    ax0.set_ylabel("Gate+ prevalence")
    ax0.set_title(f"{gene}: prevalence and delta_int,E vs {axis_label}")

    # Bootstrap percentile CIs can occasionally miss the point estimate; keep plotting stable.
    lo = np.minimum(de_lo, de_hi)
    hi = np.maximum(de_lo, de_hi)
    yerr_low = np.maximum(0.0, de - lo)
    yerr_high = np.maximum(0.0, hi - de)
    ax1.errorbar(bins, de, yerr=[yerr_low, yerr_high], fmt="o-", color="black", ms=3)
    ax1.axhline(0.0, color="0.7", lw=1)
    ax1.set_ylabel("delta_int,E")
    ax1.set_xlabel(axis_label)

    ax2 = ax1.twinx()
    ax2.plot(bins, datasets_per_bin, color="0.5", lw=1, linestyle="--")
    ax2.set_ylabel("#datasets/bin", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def _spatial_summary_text(df: pd.DataFrame) -> str:
    """Summarize spatial prevalence and delta_int,E patterns for text-only reporting."""

    if df.empty:
        return "No spatial bins passed minimum cell count."
    df = df.loc[np.isfinite(df["delta_intE"]) & np.isfinite(df["prev"])].copy()
    if df.empty:
        return "No finite spatial effect estimates."

    # Prefer bins with at least modest dataset support.
    df_supp = df.loc[df["n_datasets"] >= 5].copy()
    if df_supp.empty:
        df_supp = df

    def _fmt(v: float) -> str:
        return f"{v:.3f}"

    out: list[str] = []
    out.append(
        f"Bins used (n)={int(df_supp.shape[0])}; n_datasets/bin median={int(np.median(df_supp['n_datasets']))}."
    )
    out.append(f"Prevalence range={_fmt(float(df_supp['prev'].min()))} to {_fmt(float(df_supp['prev'].max()))}.")
    out.append(
        "delta_int,E range="
        + _fmt(float(df_supp["delta_intE"].min()))
        + " to "
        + _fmt(float(df_supp["delta_intE"].max()))
        + " (Ts_fold range="
        + _fmt(float(np.exp(df_supp["delta_intE"].min())))
        + " to "
        + _fmt(float(np.exp(df_supp["delta_intE"].max())))
        + ")."
    )

    # Coarse monotonicity check (Spearman) for quick narrative; not a formal test.
    try:
        import scipy.stats as st

        rho, p = st.spearmanr(
            df_supp["bin_center"].to_numpy(), df_supp["delta_intE"].to_numpy(), nan_policy="omit"
        )
        if np.isfinite(rho):
            out.append(f"Spearman(bin_center, delta_int,E) rho={rho:+.2f}, p={p:.3g}.")
    except Exception:
        pass

    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate consultant-requested per-gene reports (6 genes) with module associations and mediation tests.")
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("/home/chaichontat/nvme/all_progenitors.h5ad"))
    ap.add_argument(
        "--q-csv",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a_phase_matched/ts_scan_gate_mode_a_combined_bestq.csv"),
        help="CSV with columns gene,q used to define per-dataset quantile gates for the focus genes.",
    )
    ap.add_argument(
        "--animal-csv",
        type=pathlib.Path,
        default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a_phase_matched/non_neuroref_by_animal_effects.csv"),
        help="CSV with columns gene,passes_not_one_animal,strong_same_animals used for robustness flags.",
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
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("scripts/_out/consultant_gene_reports"))
    ap.add_argument(
        "--out-md",
        type=pathlib.Path,
        default=None,
        help="Optional markdown output path. Default: <outdir>/consultant_gene_reports.md",
    )
    args = ap.parse_args()

    import anndata as ad

    clip_s = str(args.pearson_clip).strip().lower()
    pearson_clip: float | None = None if clip_s in {"none", "nan"} else float(args.pearson_clip)

    outdir = args.outdir
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    genes_focus = ["Hsp90b1", "Gm3764", "Trim9", "Myt1", "Clk2", "Atp1b1"]
    qdf = pd.read_csv(args.q_csv)
    if "gene" not in qdf.columns:
        raise ValueError("--q-csv must include column: gene")
    q_col = "q" if "q" in qdf.columns else ("best_q" if "best_q" in qdf.columns else None)
    if q_col is None:
        raise ValueError("--q-csv must include a quantile column named 'q' or 'best_q'")
    qdf = qdf.set_index(qdf["gene"].astype(str))
    missing_q = [g for g in genes_focus if g not in qdf.index]
    if missing_q:
        raise ValueError(f"Missing focus genes in --q-csv: {missing_q}")

    genes = list(genes_focus)
    q_by_gene = np.array([float(qdf.loc[g, q_col]) for g in genes], dtype=float)

    anim = pd.read_csv(args.animal_csv)
    if "gene" not in anim.columns:
        raise ValueError("--animal-csv must include column: gene")
    anim = anim.set_index(anim["gene"].astype(str))
    passes_not_one_animal_by_gene: dict[str, bool | None] = {}
    strong_same_animals_by_gene: dict[str, bool | None] = {}
    for g in genes:
        if g in anim.index:
            p = anim.loc[g, "passes_not_one_animal"] if "passes_not_one_animal" in anim.columns else np.nan
            s = anim.loc[g, "strong_same_animals"] if "strong_same_animals" in anim.columns else np.nan
            passes_not_one_animal_by_gene[g] = None if pd.isna(p) else bool(p)
            strong_same_animals_by_gene[g] = None if pd.isna(s) else bool(s)
        else:
            passes_not_one_animal_by_gene[g] = None
            strong_same_animals_by_gene[g] = None

    # Modules (panel-limited)
    modules: dict[str, list[str]] = {
        "UPR_proteostasis": ["Ddit3", "Pdia6", "Tmbim6", "Calr", "Cryab", "Hsph1", "Bnip3"],
        "RG_AP_like": ["Slc1a3", "Ptprz1", "Igfbp2", "Qki", "Nes", "Vim", "Pea15a", "Tnc", "Aqp4", "Gja1", "Gfap"],
        "Neurogenic_commit": ["Eomes", "Neurog2", "Ascl1", "Dcx", "Btg2", "Gadd45g", "Neurod1", "Neurod2", "Neurod6", "Mllt11", "Tubb3"],
        "Notch_maint": ["Notch1", "Notch2", "Hes1", "Hes5", "Hes6", "Dll1", "Dll3", "Dll4", "Hey1", "Sox2", "Sox3", "Id2", "Id3", "Id4"],
        "Splicing_RNA": ["Clk2", "Srsf6", "Alyref", "Prpf4", "Sf3b3", "Son", "Hnrnpa2b1", "Hnrnpdl", "Hnrnpab", "Hnrnpm", "Nudt21"],
        "miR9_proxy": [
            "Gm3764",
            "Nr2e1",
            "Hes1",
            "Fgf8",
            "Wnt7a",
            "Wnt7b",
            "Lrp6",
            "Fzd1",
            "Fzd2",
            "Fzd3",
            "Fzd8",
            "Fzd9",
            "Ccnd1",
            "Ccnd2",
            "Ccnd3",
            "Cdkn1b",
            "Foxg1",
            "Nr2f1",
            "Nr2f2",
        ],
        "Adhesion_ECM": ["Atp1b1", "Itga6", "Itgb1", "Itga7", "Cadm1", "Cadm2", "Ctnnb1", "Fn1", "Col4a1", "Tnc", "Bcan", "Bsg"],
        "Guidance_migr": ["Robo2", "Robo3", "Unc5d", "Sema3c", "Sema5a", "Sema6d", "Dcx", "Tubb3", "Rnd2"],
        "AKT_FOXO_p27": ["Akt2", "Foxo3", "Cdkn1b", "Mki67"],
    }

    # What module to use for mediation per gene (as per consultant-genes.md)
    mediation_modules_by_gene: dict[str, list[str]] = {
        "Hsp90b1": ["UPR_proteostasis", "Adhesion_ECM"],
        "Gm3764": ["miR9_proxy", "Notch_maint"],
        "Trim9": ["Guidance_migr", "Neurogenic_commit"],
        "Myt1": ["Notch_maint", "Neurogenic_commit"],
        "Clk2": ["Splicing_RNA", "AKT_FOXO_p27"],
        "Atp1b1": ["Adhesion_ECM", "RG_AP_like"],
    }

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs

    dataset = obs["dataset"].astype(str).to_numpy().astype(object)
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

    trc = _load_tricycle_ref(args.tricycle_ref_csv)
    theta = _compute_tricycle_theta(adata, trc=trc, dataset=dataset)

    if str(args.theta_bin_mode) == "quantile":
        edges = _theta_edges_by_dataset_leiden(theta, ds_code, lei_code, n_bins=int(args.theta_bins))
        theta_bin = _assign_theta_bins_from_edges(theta, ds_code, lei_code, edges, n_bins=int(args.theta_bins))
    else:
        t = (theta.astype(np.float64, copy=False) + (2.0 * np.pi)) % (2.0 * np.pi)
        w = (2.0 * np.pi) / float(int(args.theta_bins))
        theta_bin = np.clip(np.floor(t / w), 0, int(args.theta_bins) - 1).astype(np.int16, copy=False)

    n_strata = n_datasets * int(args.theta_bins) * n_leiden
    stratum_idx_base = (ds_code * int(args.theta_bins) + theta_bin.astype(int)) * n_leiden + lei_code

    # Gates for focus genes
    X_focus_raw = _raw_counts_for_genes(adata, genes=genes)
    X_focus = _pearson_residuals_by_dataset(
        X_focus_raw,
        total_counts=total_counts,
        dataset_codes=ds_code,
        n_datasets=n_datasets,
        theta=float(args.pearson_theta),
        clip=pearson_clip,
        block_size=int(args.pearson_block_size),
    )
    sin_t = np.sin(theta).astype(np.float32, copy=False)
    cos_t = np.cos(theta).astype(np.float32, copy=False)
    R_focus = _residualize_matrix_within_dataset(X_focus, sin_t=sin_t, cos_t=cos_t, dataset=dataset)
    G_focus = _make_gates_by_dataset_quantile(R_focus, dataset=dataset, q_by_gene=q_by_gene)

    # Load sparse log1p raw for DE (all genes)
    X_all_log1p = _sparse_log1p_layer_raw(adata)
    var_names = adata.var_names.astype(str).to_numpy()

    # Load module gene log1p (dense) for module scoring and mediation
    module_gene_set = sorted({g for glist in modules.values() for g in glist if g in set(var_names.tolist())})
    X_mod = _log1p_raw_for_genes(adata, genes=module_gene_set)
    X_mod_z = _module_score_within_dataset_leiden_z(X_mod, ds_code, lei_code)
    gene_to_modcol = {g: j for j, g in enumerate(module_gene_set)}

    module_scores: dict[str, np.ndarray] = {}
    for mname, glist in modules.items():
        use = [g for g in glist if g in gene_to_modcol]
        if not use:
            continue
        cols = np.array([gene_to_modcol[g] for g in use], dtype=int)
        module_scores[mname] = X_mod_z[:, cols].mean(axis=1).astype(np.float32, copy=False)

    def _module_score_leave_one_out(module_name: str, *, gene_under_test: str) -> np.ndarray | None:
        """Return module score excluding gene_under_test if it is part of the module.

        This prevents 'mechanical' attenuation/amplification when stratifying on a score
        that includes the predictor itself (e.g., Clk2 in Splicing_RNA; Atp1b1 in Adhesion_ECM).
        """

        if module_name not in modules:
            return None
        glist = [g for g in modules[module_name] if g in gene_to_modcol and g != gene_under_test]
        if not glist:
            return None
        cols = np.array([gene_to_modcol[g] for g in glist], dtype=int)
        return X_mod_z[:, cols].mean(axis=1).astype(np.float32, copy=False)

    rng = np.random.default_rng(int(args.seed))

    # Spatial coordinate extraction
    XY = np.asarray(adata.obsm["AP_ML_um"], dtype=np.float32)
    AP_um = XY[:, 0]
    ML_um = XY[:, 1]
    orient = np.array(["Sag" if "Sag" in d else "Coro" for d in dataset], dtype=object)

    def _spatial_bins_for_orientation(orient_value: str, bin_um: float = 200.0):
        if orient_value == "Sag":
            x = AP_um
            label = "AP (um) [col0]"
        else:
            x = ML_um
            label = "ML (um) [col1]"
        m = orient == orient_value
        xmin, xmax = float(np.nanmin(x[m])), float(np.nanmax(x[m]))
        edges = np.arange(math.floor(xmin / bin_um) * bin_um, math.ceil(xmax / bin_um) * bin_um + bin_um, bin_um)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return x, m, edges, centers, label

    # Prepare report markdown
    md_lines: list[str] = []
    md_lines.append("# Consultant Gene Reports (Phase-Matched, 4-Quadrant, Interaction Phenotype)\n")
    md_lines.append("This document implements the per-gene report structure requested in `consultant-genes.md` for: **Hsp90b1, Gm3764, Trim9, Myt1, Clk2, Atp1b1**.\n")
    md_lines.append("All MH estimates are phase-matched via strict stratification by **(dataset × theta-bin × leiden)** with **shared theta-bin edges per dataset×leiden** across all endpoints (E1–E4). Gates are built from **dataset-wise Pearson residuals** (NB theta, clip) computed from raw counts using `total_counts`, then residualized within dataset on **[1, sin(theta), cos(theta)]** and thresholded at per-dataset quantile `q` (from the exec summary).\n")
    md_lines.append("**Module definitions (panel-limited; leave-one-out is used for mediation stratification when the tested gene is part of the module):**\n")
    for mname, glist in modules.items():
        use = [x for x in glist if x in set(var_names.tolist())]
        md_lines.append(f"- {mname}: " + ", ".join(use))
    md_lines.append("")

    for j, g in enumerate(genes):
        gate = G_focus[:, j]

        # Endpoint deltas
        endpoints = [
            ("BrdUplus_Edu", brdu == 1, edu),
            ("Eduplus_BrdU", edu == 1, brdu),
            ("BrdUminus_Edu", brdu == 0, edu),
            ("EdUminus_BrdU", edu == 0, brdu),
        ]

        summary: dict[str, Mh] = {}
        contribs: dict[str, MhDsContrib] = {}

        for name, cohort, y_all in endpoints:
            c = _mh_ds_contrib(
                stratum_idx=stratum_idx_base[cohort],
                G=gate[cohort],
                y=y_all[cohort],
                n_strata=n_strata,
                n_datasets=n_datasets,
                n_leiden=n_leiden,
                theta_bins=int(args.theta_bins),
                cc=float(args.cc),
            )
            lo, hi = _bootstrap_ci_from_ds_contrib(c, rng=rng, n_boot=int(args.n_bootstrap))
            summary[name] = Mh(delta=float(c.delta), lo=float(lo), hi=float(hi))
            contribs[name] = c

        d_intE, lo_intE, hi_intE = _bootstrap_ci_for_diff(contribs["BrdUplus_Edu"], contribs["BrdUminus_Edu"], rng=rng, n_boot=int(args.n_bootstrap))
        d_intB, lo_intB, hi_intB = _bootstrap_ci_for_diff(contribs["Eduplus_BrdU"], contribs["EdUminus_BrdU"], rng=rng, n_boot=int(args.n_bootstrap))
        intE = Mh(delta=float(d_intE), lo=float(lo_intE), hi=float(hi_intE))
        intB = Mh(delta=float(d_intB), lo=float(lo_intB), hi=float(hi_intB))

        # Power/support
        n_brd = int((brdu == 1).sum())
        n_edu = int((edu == 1).sum())
        n_gate_all = int(gate.sum())
        n_gate_brd = int((gate & (brdu == 1)).sum())

        def _count_gate_brd_leiden(val: str) -> int:
            return int((gate & (brdu == 1) & (leiden == val)).sum())

        sparse_E1 = _sparse_strata_fraction(
            stratum_idx=stratum_idx_base[brdu == 1], G=gate[brdu == 1], y=edu[brdu == 1], n_strata=n_strata
        )
        sparse_E3 = _sparse_strata_fraction(
            stratum_idx=stratum_idx_base[brdu == 0], G=gate[brdu == 0], y=edu[brdu == 0], n_strata=n_strata
        )

        passes_not_one_animal = passes_not_one_animal_by_gene.get(g)

        power = {
            "n_datasets": n_datasets,
            "n_brdUpos": n_brd,
            "n_edUpos": n_edu,
            "n_gatepos_all": n_gate_all,
            "frac_gatepos_all": n_gate_all / float(gate.size),
            "n_gatepos_brdUpos": n_gate_brd,
            "frac_gatepos_brdUpos": n_gate_brd / float(n_brd) if n_brd else float("nan"),
            "n_gatepos_brdUpos_L6": _count_gate_brd_leiden("6"),
            "n_gatepos_brdUpos_L14": _count_gate_brd_leiden("14"),
            "n_gatepos_brdUpos_L15": _count_gate_brd_leiden("15"),
            "sparse_E1": sparse_E1,
            "sparse_E3": sparse_E3,
            "passes_not_one_animal": passes_not_one_animal,
            "strong_same_animals": strong_same_animals_by_gene.get(g),
        }

        # Module association: mean diff gate+ - gate- within each leiden, averaged across datasets (equal weight)
        mod_rows = []
        for mname, score in module_scores.items():
            diffs = []
            for ds in np.unique(ds_code):
                for lei in np.unique(lei_code):
                    m = (ds_code == ds) & (lei_code == lei)
                    if not np.any(m):
                        continue
                    gsel = gate[m]
                    if int(gsel.sum()) < 50 or int((~gsel).sum()) < 50:
                        continue
                    diffs.append(float(score[m][gsel].mean() - score[m][~gsel].mean()))
            if diffs:
                d = float(np.mean(diffs))
                mod_rows.append({"module": mname, "diff": d, "abs_diff": abs(d)})
        module_assoc = pd.DataFrame(mod_rows).sort_values("abs_diff", ascending=False)

        # Mediation-style stratification: add module-bin to strata and recompute intE
        mediation_res: dict[str, Mh] = {}
        for mname in mediation_modules_by_gene[g]:
            score = _module_score_leave_one_out(mname, gene_under_test=g)
            if score is None:
                # Fall back to the full module score if the leave-one-out is empty (should be rare).
                score = module_scores.get(mname)
            if score is None:
                continue
            mbin = _module_bins_within_dataset_leiden(score, ds_code, lei_code, q=3)
            # strata: dataset×theta×leiden×mbin
            n_mbin = 3
            n_strata_m = n_datasets * int(args.theta_bins) * n_leiden * n_mbin
            stratum_idx_m = (((ds_code * int(args.theta_bins) + theta_bin.astype(int)) * n_leiden + lei_code) * n_mbin + mbin.astype(int))

            c1 = _mh_ds_contrib(
                stratum_idx=stratum_idx_m[brdu == 1],
                G=gate[brdu == 1],
                y=edu[brdu == 1],
                n_strata=n_strata_m,
                n_datasets=n_datasets,
                n_leiden=n_leiden * n_mbin,
                theta_bins=int(args.theta_bins),
                cc=float(args.cc),
            )
            c3 = _mh_ds_contrib(
                stratum_idx=stratum_idx_m[brdu == 0],
                G=gate[brdu == 0],
                y=edu[brdu == 0],
                n_strata=n_strata_m,
                n_datasets=n_datasets,
                n_leiden=n_leiden * n_mbin,
                theta_bins=int(args.theta_bins),
                cc=float(args.cc),
            )
            d, lo, hi = _bootstrap_ci_for_diff(c1, c3, rng=rng, n_boot=int(args.n_bootstrap))
            mediation_res[mname] = Mh(delta=float(d), lo=float(lo), hi=float(hi))

        # Pseudobulk DE for L14 and L15
        de_tables = {}
        for L in ["14", "15"]:
            de = _pseudobulk_de_within_leiden(X_all_log1p, dataset=dataset, leiden=leiden, gate=gate, leiden_value=L)
            de["gene"] = var_names
            de = de.loc[:, ["gene", "mean_diff", "t_stat", "n_datasets_used"]].sort_values("t_stat", ascending=False)
            de_tables[L] = de
            de.to_csv(tabdir / f"{g}_DE_leiden{L}.csv", index=False)

        # Save plots
        summary_png = figdir / f"{g}_panel.png"
        _plot_gene_summary_panel(summary_png, gene=g, summary=summary, intE=intE, intB=intB, power=power, module_assoc=module_assoc, mediation=mediation_res)

        # Spatial plots (Sag(AP) and Coro(ML))
        spatial_pngs = {}
        spatial_summaries: dict[str, str] = {}
        for orient_value in ["Sag", "Coro"]:
            x, m_or, edges_sp, centers_sp, axis_label = _spatial_bins_for_orientation(orient_value)
            # Compute prevalence and delta_int,E per spatial bin using dataset bootstrap.
            prev = np.full(centers_sp.shape[0], np.nan)
            prev_lo = np.full(centers_sp.shape[0], np.nan)
            prev_hi = np.full(centers_sp.shape[0], np.nan)
            dE = np.full(centers_sp.shape[0], np.nan)
            dE_lo = np.full(centers_sp.shape[0], np.nan)
            dE_hi = np.full(centers_sp.shape[0], np.nan)
            ds_per_bin = np.zeros(centers_sp.shape[0], dtype=int)
            n_cells_bin = np.zeros(centers_sp.shape[0], dtype=int)
            n_gatepos_bin = np.zeros(centers_sp.shape[0], dtype=int)
            n_brdUpos_bin = np.zeros(centers_sp.shape[0], dtype=int)
            n_brdUneg_bin = np.zeros(centers_sp.shape[0], dtype=int)

            # Precompute helpful codes for ds bootstrap
            for bi in range(centers_sp.shape[0]):
                loe, hie = float(edges_sp[bi]), float(edges_sp[bi + 1])
                m_bin = m_or & (x >= loe) & (x < hie)
                if int(m_bin.sum()) < 500:
                    continue
                n_cells_bin[bi] = int(m_bin.sum())
                n_gatepos_bin[bi] = int(gate[m_bin].sum())
                n_brdUpos_bin[bi] = int(((brdu == 1) & m_bin).sum())
                n_brdUneg_bin[bi] = int(((brdu == 0) & m_bin).sum())

                prev[bi] = float(gate[m_bin].mean())
                # Bootstrap prevalence by dataset
                ds_here = ds_code[m_bin]
                ds_unique = np.unique(ds_here)
                ds_per_bin[bi] = int(ds_unique.size)
                boots = []
                for _ in range(200):
                    samp = rng.choice(ds_unique, size=ds_unique.size, replace=True)
                    mb = np.isin(ds_here, samp)
                    boots.append(float(gate[m_bin][mb].mean()))
                prev_lo[bi], prev_hi[bi] = np.quantile(boots, [0.025, 0.975])

                # Effect: compute E1 and E3 within spatial bin
                c1 = _mh_ds_contrib(
                    stratum_idx=stratum_idx_base[m_bin & (brdu == 1)],
                    G=gate[m_bin & (brdu == 1)],
                    y=edu[m_bin & (brdu == 1)],
                    n_strata=n_strata,
                    n_datasets=n_datasets,
                    n_leiden=n_leiden,
                    theta_bins=int(args.theta_bins),
                    cc=float(args.cc),
                )
                c3 = _mh_ds_contrib(
                    stratum_idx=stratum_idx_base[m_bin & (brdu == 0)],
                    G=gate[m_bin & (brdu == 0)],
                    y=edu[m_bin & (brdu == 0)],
                    n_strata=n_strata,
                    n_datasets=n_datasets,
                    n_leiden=n_leiden,
                    theta_bins=int(args.theta_bins),
                    cc=float(args.cc),
                )
                d, lo2, hi2 = _bootstrap_ci_for_diff(c1, c3, rng=rng, n_boot=200)
                dE[bi], dE_lo[bi], dE_hi[bi] = float(d), float(lo2), float(hi2)

            ok = np.isfinite(prev) & np.isfinite(dE)
            if np.any(ok):
                df_sp = pd.DataFrame(
                    {
                        "bin_center": centers_sp[ok],
                        "n_cells": n_cells_bin[ok],
                        "n_gatepos": n_gatepos_bin[ok],
                        "n_brdUpos": n_brdUpos_bin[ok],
                        "n_brdUneg": n_brdUneg_bin[ok],
                        "prev": prev[ok],
                        "prev_ci_low": prev_lo[ok],
                        "prev_ci_high": prev_hi[ok],
                        "delta_intE": dE[ok],
                        "delta_intE_ci_low": dE_lo[ok],
                        "delta_intE_ci_high": dE_hi[ok],
                        "n_datasets": ds_per_bin[ok],
                    }
                )
                df_sp.to_csv(tabdir / f"{g}_spatial_{orient_value}.csv", index=False)
                spatial_summaries[orient_value] = _spatial_summary_text(df_sp)

                sp_png = figdir / f"{g}_spatial_{orient_value}.png"
                _plot_spatial(
                    sp_png,
                    gene=g,
                    bins=centers_sp[ok],
                    prev=prev[ok],
                    prev_lo=prev_lo[ok],
                    prev_hi=prev_hi[ok],
                    de=dE[ok],
                    de_lo=dE_lo[ok],
                    de_hi=dE_hi[ok],
                    axis_label=f"{orient_value} {axis_label}",
                    datasets_per_bin=ds_per_bin[ok],
                )
                spatial_pngs[orient_value] = sp_png

        # Markdown section
        md_lines.append(f"\n## {g}\n")
        md_lines.append(
            f"Gate definition: dataset-wise Pearson residuals (NB theta={float(args.pearson_theta):.1f}, clip={str(args.pearson_clip)}) residualized within-dataset on [1,sin(theta),cos(theta)], threshold q={float(q_by_gene[j]):.2f} per dataset.\n"
        )

        md_lines.append("**Quadrant/interaction summary (MH; strata dataset×theta_bin×leiden):**\n")
        for k, label in [
            ("BrdUplus_Edu", "E1: BrdU+ cohort, outcome EdU+"),
            ("Eduplus_BrdU", "E2: EdU+ cohort, outcome BrdU+"),
            ("BrdUminus_Edu", "E3: BrdU- cohort, outcome EdU+"),
            ("EdUminus_BrdU", "E4: EdU- cohort, outcome BrdU+"),
        ]:
            mh = summary[k]
            md_lines.append(
                f"- {label}: delta={mh.delta:.3f} (CI {mh.lo:.3f},{mh.hi:.3f}); Ts_fold={mh.Ts_fold:.3f} (CI {mh.Ts_lo:.3f},{mh.Ts_hi:.3f})"
            )
        md_lines.append(
            f"- IntE (delta_int,E = E1-E3): delta={intE.delta:.3f} (CI {intE.lo:.3f},{intE.hi:.3f}); Ts_fold={intE.Ts_fold:.3f} (CI {intE.Ts_lo:.3f},{intE.Ts_hi:.3f})"
        )
        md_lines.append(
            f"- IntB (delta_int,B = E2-E4): delta={intB.delta:.3f} (CI {intB.lo:.3f},{intB.hi:.3f}); Ts_fold={intB.Ts_fold:.3f} (CI {intB.Ts_lo:.3f},{intB.Ts_hi:.3f})\n"
        )

        md_lines.append("**Power / support (phase-matched):**\n")
        md_lines.append(
            f"- gate+ all={power['n_gatepos_all']} ({power['frac_gatepos_all']:.3f}); gate+ among BrdU+={power['n_gatepos_brdUpos']} ({power['frac_gatepos_brdUpos']:.3f})"
        )
        md_lines.append(
            f"- gate+ among BrdU+ by Leiden: L6={power['n_gatepos_brdUpos_L6']}, L14={power['n_gatepos_brdUpos_L14']}, L15={power['n_gatepos_brdUpos_L15']}"
        )
        md_lines.append(f"- sparse strata fraction (zeros in 2x2 before cc): E1={power['sparse_E1']:.3f}, E3={power['sparse_E3']:.3f}")
        md_lines.append(f"- not-one-animal filter pass: {power['passes_not_one_animal']}\n")

        if not module_assoc.empty:
            md_lines.append("**Module associations (panel-limited; within dataset×leiden z-scored log1p):**\n")
            for _, r in module_assoc.head(6).iterrows():
                md_lines.append(f"- {r['module']}: mean z-diff gate+ - gate- = {r['diff']:+.3f}")
            md_lines.append("")

        if mediation_res:
            md_lines.append("**Mediation-style stratification (delta_int,E after adding module-bin to MH strata):**\n")
            for mname, mh in mediation_res.items():
                md_lines.append(
                    f"- +{mname} bins: delta_int,E={mh.delta:.3f} (CI {mh.lo:.3f},{mh.hi:.3f}); Ts_fold={mh.Ts_fold:.3f}"
                )
            md_lines.append("")

        md_lines.append("**Within-lineage pseudobulk DE (gate+ vs gate-; log1p(raw); per-dataset means; meta t across datasets):**\n")
        for L, label in [("14", "Leiden 14 (RG-like)"), ("15", "Leiden 15 (IPC/neurogenic)")]:
            de = de_tables[L]
            md_lines.append(f"- {label}: top up genes (mean_diff, t_stat):")
            top_up = de.head(8)
            for _, r in top_up.iterrows():
                md_lines.append(f"  - {r['gene']}: {r['mean_diff']:+.3f}, t={r['t_stat']:.2f}")
            md_lines.append(f"- {label}: top down genes (mean_diff, t_stat):")
            top_dn = de.tail(8).iloc[::-1]
            for _, r in top_dn.iterrows():
                md_lines.append(f"  - {r['gene']}: {r['mean_diff']:+.3f}, t={r['t_stat']:.2f}")
        md_lines.append("")

        md_lines.append("**Figures:**\n")
        md_lines.append(f"- Summary panel: `{summary_png}`")
        for orient_value, sp_png in spatial_pngs.items():
            md_lines.append(f"- Spatial (prevalence + delta_int,E): `{sp_png}`")
        for orient_value, txt in spatial_summaries.items():
            md_lines.append(f"- Spatial summary ({orient_value}): {txt}")

    out_md = args.out_md if args.out_md is not None else (outdir / "consultant_gene_reports.md")
    out_md.write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
