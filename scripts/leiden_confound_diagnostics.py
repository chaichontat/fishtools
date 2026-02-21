#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib

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


def _compute_tricycle_theta(adata, idx: np.ndarray, trc: pd.DataFrame, dataset_for_idx: np.ndarray) -> np.ndarray:
    ref_syms = np.asarray(trc["symbol"], dtype=object)
    ref_mask = np.isin(np.asarray(adata.var_names, dtype=object), ref_syms)
    if not np.any(ref_mask):
        raise ValueError("No shared genes between tricycle ref and adata.var_names.")

    var_syms = np.asarray(adata.var_names, dtype=object)
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
        Xd = X[m] - X[m].mean(axis=0, keepdims=True)
        pc1 = Xd @ w1
        pc2 = Xd @ w2
        theta[m] = np.arctan2(pc2, pc1)
    return theta


def _theta_bins(theta: np.ndarray, dataset: np.ndarray, *, n_bins: int, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "quantile":
        ds = np.asarray(dataset, dtype=object)
        out = np.empty(theta.shape[0], dtype=int)
        for d in np.unique(ds):
            m = ds == d
            t = theta[m]
            edges = np.quantile(t, q=np.linspace(0, 1, int(n_bins) + 1))
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = np.nextafter(edges[i - 1], edges[i - 1] + 1.0)
            out[m] = np.digitize(t, edges[1:-1], right=False)
        return out
    if mode == "angle":
        t = (theta.astype(np.float64, copy=False) + (2.0 * np.pi)) % (2.0 * np.pi)
        w = (2.0 * np.pi) / float(int(n_bins))
        return np.clip(np.floor(t / w), 0, int(n_bins) - 1).astype(int, copy=False)
    raise ValueError(f"Unknown theta bin mode={mode!r} (expected 'quantile' or 'angle').")


def _raw_counts_for_genes(adata, idx: np.ndarray, genes: list[str]) -> np.ndarray:
    var = np.asarray(adata.var_names, dtype=object)
    gene_to_col = {g: i for i, g in enumerate(var)}
    cols = []
    for g in genes:
        if g not in gene_to_col:
            raise ValueError(f"Gene not found in adata.var_names: {g}")
        cols.append(gene_to_col[g])
    cols = np.asarray(cols, dtype=int)
    if "raw" in adata.layers:
        X = adata.layers["raw"][idx][:, cols]
    else:
        X = adata.X[idx][:, cols]
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


def _residualize_against_tricycle_within_dataset(
    X: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray
) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    R = np.empty_like(X, dtype=float)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m)), sin_t[m], cos_t[m]])
        coef, *_ = np.linalg.lstsq(Z, X[m], rcond=None)
        R[m] = X[m] - (Z @ coef)
    return R


def _make_gates_by_dataset_quantile(R: np.ndarray, dataset: np.ndarray, q_by_gene: np.ndarray) -> np.ndarray:
    # R: (n_cells, n_genes), q_by_gene: (n_genes,)
    ds = np.asarray(dataset, dtype=object)
    G = np.zeros(R.shape, dtype=bool)
    uniq_q = np.unique(q_by_gene)
    for d in np.unique(ds):
        m = ds == d
        rows = np.where(m)[0]
        Rd = R[m]
        for q in uniq_q:
            j = np.where(q_by_gene == q)[0]
            thr = np.quantile(Rd[:, j], q=q, axis=0)
            G[np.ix_(rows, j)] = Rd[:, j] >= thr
    return G


def _naive_log_or(G: np.ndarray, y: np.ndarray) -> float:
    # Haldane-Anscombe correction
    a = float(np.sum(G & (y == 1))) + 0.5
    b = float(np.sum(G & (y == 0))) + 0.5
    c = float(np.sum((~G) & (y == 1))) + 0.5
    d = float(np.sum((~G) & (y == 0))) + 0.5
    return math.log((a * d) / (b * c))


def _mh_log_or(stratum_idx: np.ndarray, G: np.ndarray, y: np.ndarray, n_strata: int) -> float:
    g = G.astype(float)
    y1 = y.astype(float)
    y0 = 1.0 - y1

    a = np.bincount(stratum_idx, weights=g * y1, minlength=n_strata)
    b = np.bincount(stratum_idx, weights=g * y0, minlength=n_strata)
    c = np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=n_strata)
    d = np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=n_strata)

    # Haldane-Anscombe correction per stratum
    a = a + 0.5
    b = b + 0.5
    c = c + 0.5
    d = d + 0.5
    n = a + b + c + d

    R = np.sum((a * d) / n)
    S = np.sum((b * c) / n)
    return float(math.log(R / S))


def _cramers_v(G: np.ndarray, leiden_code: np.ndarray, n_leiden: int) -> float:
    # 2 x k table
    g1 = np.bincount(leiden_code, weights=G.astype(float), minlength=n_leiden)
    n = np.bincount(leiden_code, minlength=n_leiden).astype(float)
    g0 = n - g1

    obs = np.vstack([g0, g1])  # (2, k)
    tot = float(obs.sum())
    if tot <= 0:
        return float("nan")
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    exp = row @ col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    phi2 = chi2 / tot
    denom = min(1, n_leiden - 1)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(max(phi2, 0.0) / denom))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose confounding by Leiden: naive OR vs MH within (dataset×theta×leiden), plus G~Leiden association."
    )
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    ap.add_argument(
        "--genes-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_no_cnksr2/mh_phase_consistency_top_genes.csv"
        ),
    )
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument(
        "--theta-bin-mode",
        type=str,
        choices=["quantile", "angle"],
        default="quantile",
        help="Theta binning: quantile within dataset vs fixed absolute angular bins.",
    )
    ap.add_argument("--pearson-theta", type=float, default=100.0, help="Pearson residual theta (NB overdispersion).")
    ap.add_argument(
        "--pearson-clip",
        type=str,
        default="10.0",
        help="Clip Pearson residuals to +/-clip; use 'none' to disable clipping.",
    )
    ap.add_argument("--pearson-block-size", type=int, default=64)
    ap.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/leiden_confound_diagnostics_no_cnksr2.csv"
        ),
    )
    ap.add_argument(
        "--out-png",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/leiden_confound_diagnostics_no_cnksr2.png"
        ),
    )
    args = ap.parse_args()

    import anndata as ad
    import matplotlib.pyplot as plt

    genes_df = pd.read_csv(args.genes_csv)
    genes = genes_df["gene"].astype(str).tolist()
    q_by_gene = genes_df["q"].astype(float).to_numpy()

    clip_s = str(args.pearson_clip).strip().lower()
    pearson_clip: float | None = None if clip_s in {"none", "nan"} else float(args.pearson_clip)

    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    idx = np.asarray(obs["brdu_pos"].to_numpy() == 1, dtype=bool).nonzero()[0]
    if "total_counts" not in obs.columns:
        raise ValueError("adata.obs must contain 'total_counts' for Pearson residual gating.")

    dataset = obs["dataset"].astype(str).to_numpy()[idx].astype(object)
    leiden = obs["leiden"].astype(str).to_numpy()[idx].astype(object)
    y = obs["edu_pos"].to_numpy()[idx].astype(int)
    total_counts = obs["total_counts"].to_numpy()[idx]

    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    theta = _compute_tricycle_theta(adata, idx, trc=trc, dataset_for_idx=dataset)
    theta_bin = _theta_bins(theta, dataset=dataset, n_bins=int(args.theta_bins), mode=str(args.theta_bin_mode))

    # Encode leiden and dataset for stratum IDs
    ds_code, ds_uniq = pd.factorize(dataset, sort=True)
    lei_code, lei_uniq = pd.factorize(leiden, sort=True)
    n_strata = int((ds_uniq.size) * args.theta_bins * (lei_uniq.size))
    stratum_idx = (ds_code * args.theta_bins + theta_bin) * (lei_uniq.size) + lei_code

    X_raw = _raw_counts_for_genes(adata, idx, genes=genes)
    X = _pearson_residuals_by_dataset(
        X_raw,
        total_counts=total_counts,
        dataset_codes=ds_code,
        n_datasets=int(ds_uniq.size),
        theta=float(args.pearson_theta),
        clip=pearson_clip,
        block_size=int(args.pearson_block_size),
    )
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    R = _residualize_against_tricycle_within_dataset(X, sin_t=sin_t, cos_t=cos_t, dataset=dataset)
    Gmat = _make_gates_by_dataset_quantile(R, dataset=dataset, q_by_gene=q_by_gene)

    rows = []
    for j, g in enumerate(genes):
        G = Gmat[:, j]
        d_naive = _naive_log_or(G, y=y)
        d_mh = _mh_log_or(stratum_idx=stratum_idx, G=G, y=y, n_strata=n_strata)
        v = _cramers_v(G=G, leiden_code=lei_code, n_leiden=lei_uniq.size)

        # Gate prevalence by leiden (cell-weighted)
        nL = np.bincount(lei_code, minlength=lei_uniq.size).astype(float)
        g1 = np.bincount(lei_code, weights=G.astype(float), minlength=lei_uniq.size)
        piL = np.divide(g1, nL, out=np.full_like(nL, np.nan), where=nL > 0)

        row = {
            "gene": g,
            "q": float(q_by_gene[j]),
            "naive_delta": float(d_naive),
            "naive_Ts_fold": float(math.exp(d_naive)),
            "mh_delta_ds_theta_leiden": float(d_mh),
            "mh_Ts_fold_ds_theta_leiden": float(math.exp(d_mh)),
            "delta_naive_minus_mh": float(d_naive - d_mh),
            "cramers_v_G_vs_leiden": float(v),
        }
        for k, name in enumerate(lei_uniq.tolist()):
            row[f"pi_G1_given_leiden_{name}"] = float(piL[k])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(genes_df[["gene", "delta_logloss_LODO"]], on="gene", how="left")
    out = out.sort_values("delta_logloss_LODO", ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Figure: naive vs MH Ts_fold, colored by Cramer's V
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 8.0), dpi=160)

    x = out["naive_Ts_fold"].to_numpy(float)
    yv = out["mh_Ts_fold_ds_theta_leiden"].to_numpy(float)
    c = out["cramers_v_G_vs_leiden"].to_numpy(float)

    ax1.plot([0.5, 3.5], [0.5, 3.5], color="black", lw=1.0, alpha=0.5)
    sc = ax1.scatter(x, yv, c=c, cmap="viridis", s=55, edgecolor="black", linewidths=0.3)
    for _, r in out.iterrows():
        ax1.text(
            float(r["naive_Ts_fold"]) * 1.01,
            float(r["mh_Ts_fold_ds_theta_leiden"]) * 1.01,
            str(r["gene"]),
            fontsize=7,
            alpha=0.9,
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.6, 3.5)
    ax1.set_ylim(0.6, 3.5)
    ax1.set_xlabel("Naive Ts_fold (unstratified)")
    ax1.set_ylabel("MH Ts_fold (strata dataset×θ×leiden)")
    ax1.set_title("Confounding check: naive vs within-Leiden phase-matched effect")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Cramer's V (G vs Leiden)")

    # Panel 2: delta difference (naive - MH)
    dd = out["delta_naive_minus_mh"].to_numpy(float)
    order = np.argsort(dd)
    ax2.axvline(0.0, color="black", lw=1.0, alpha=0.5)
    ax2.barh(np.arange(len(order)), dd[order], color="#888888", alpha=0.9)
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels(out["gene"].to_numpy()[order])
    ax2.set_xlabel("delta_naive - delta_MH (log OR)")
    ax2.set_title("Leiden confounding index (positive = naive inflated)")

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png)
    plt.close(fig)


if __name__ == "__main__":
    main()
