#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import re
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


def _dataset_animal(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r}.")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _compute_tricycle_theta(adata, idx: np.ndarray, trc: pd.DataFrame, dataset_for_idx: np.ndarray) -> np.ndarray:
    # Matches mh_phase_consistency_mode_a.py: within-dataset mean-centering on tricycle ref genes, then projection.
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
        Xd = X[m]
        Xd = Xd - Xd.mean(axis=0, keepdims=True)
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
            edges = np.quantile(t, q=np.linspace(0.0, 1.0, int(n_bins) + 1))
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


def _log1p_raw_for_gene(adata, idx: np.ndarray, gene: str) -> np.ndarray:
    var = np.asarray(adata.var_names, dtype=object)
    try:
        col = int(np.where(var == gene)[0][0])
    except IndexError as e:
        raise ValueError(f"Gene not found in adata.var_names: {gene}") from e
    if "raw" in adata.layers:
        X = adata.layers["raw"][idx][:, col]
    else:
        X = adata.X[idx][:, col]
    if hasattr(X, "toarray"):
        X = X.toarray()
    x = np.asarray(X, dtype=float).reshape(-1)
    return np.log1p(x)


def _log1p_raw_for_genes(adata, idx: np.ndarray, genes: list[str]) -> np.ndarray:
    var = np.asarray(adata.var_names, dtype=object)
    gene_to_col = {g: i for i, g in enumerate(var.tolist())}
    cols = []
    for g in genes:
        if g not in gene_to_col:
            raise ValueError(f"Gene not found in adata.var_names: {g}")
        cols.append(int(gene_to_col[g]))
    cols = np.asarray(cols, dtype=int)

    if "raw" in adata.layers:
        X = adata.layers["raw"][idx][:, cols]
    else:
        X = adata.X[idx][:, cols]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    return np.log1p(X)


def _residualize_within_dataset(values: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    resid = np.empty(values.shape[0], dtype=float)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m)), sin_t[m], cos_t[m]])
        coef, *_ = np.linalg.lstsq(Z, values[m].reshape(-1, 1), rcond=None)  # (3,1)
        resid[m] = values[m] - (Z @ coef).reshape(-1)
    return resid


def _residualize_matrix_within_dataset(
    X: np.ndarray, sin_t: np.ndarray, cos_t: np.ndarray, dataset: np.ndarray
) -> np.ndarray:
    # Residualize each column of X against 1 + sin(theta) + cos(theta), within dataset.
    ds = np.asarray(dataset, dtype=object)
    R = np.empty_like(X, dtype=float)
    for d in np.unique(ds):
        m = ds == d
        Z = np.column_stack([np.ones(np.sum(m)), sin_t[m], cos_t[m]])
        coef, *_ = np.linalg.lstsq(Z, X[m], rcond=None)  # (3, n_genes)
        R[m] = X[m] - (Z @ coef)
    return R


def _gate_by_dataset_quantile(resid: np.ndarray, dataset: np.ndarray, q: float) -> np.ndarray:
    ds = np.asarray(dataset, dtype=object)
    G = np.zeros(resid.shape[0], dtype=bool)
    for d in np.unique(ds):
        m = ds == d
        thr = float(np.quantile(resid[m], q=float(q)))
        G[m] = resid[m] >= thr
    return G


def _gates_for_qs_by_dataset(R: np.ndarray, dataset: np.ndarray, qs: list[float]) -> dict[float, np.ndarray]:
    # Return gates per q as bool matrix (n_cells, n_genes).
    ds = np.asarray(dataset, dtype=object)
    qs = [float(q) for q in qs]
    out = {float(q): np.zeros(R.shape, dtype=bool) for q in qs}
    for d in np.unique(ds):
        m = ds == d
        Rd = R[m]
        for q in qs:
            thr = np.quantile(Rd, q=float(q), axis=0)
            out[float(q)][m] = Rd >= thr
    return out


@dataclass(frozen=True)
class MhResult:
    delta: float
    ci_low: float
    ci_high: float
    Ts_fold: float
    Ts_ci_low: float
    Ts_ci_high: float


def _mh_components_by_stratum(
    stratum_idx: np.ndarray,
    G: np.ndarray,
    y: np.ndarray,
    *,
    n_strata: int,
    cc: float,
    drop_zero_strata: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Return arrays per stratum: a,b,c,d,n with optional per-stratum continuity correction.
    g = G.astype(float, copy=False)
    y1 = y.astype(float, copy=False)
    y0 = 1.0 - y1

    a0 = np.bincount(stratum_idx, weights=g * y1, minlength=int(n_strata))
    b0 = np.bincount(stratum_idx, weights=g * y0, minlength=int(n_strata))
    c0 = np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=int(n_strata))
    d0 = np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=int(n_strata))

    if drop_zero_strata:
        keep = (a0 > 0) & (b0 > 0) & (c0 > 0) & (d0 > 0)
    else:
        keep = np.ones_like(a0, dtype=bool)

    a = a0[keep] + float(cc)
    b = b0[keep] + float(cc)
    c = c0[keep] + float(cc)
    d = d0[keep] + float(cc)
    n = a + b + c + d
    return a, b, c, d, n


def _mh_delta_from_components(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, n: np.ndarray) -> float:
    R = float(np.sum((a * d) / n))
    S = float(np.sum((b * c) / n))
    if not (R > 0.0 and S > 0.0):
        return float("nan")
    return float(math.log(R / S))


def _mh_bootstrap_ci_by_dataset(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    n: np.ndarray,
    stratum_to_dataset_code: np.ndarray,
    *,
    n_datasets: int,
    rng: np.random.Generator,
    n_boot: int,
) -> MhResult:
    # Compute per-stratum contributions then bootstrap over datasets.
    r = (a * d) / n
    s = (b * c) / n

    R_total = float(r.sum())
    S_total = float(s.sum())
    delta = float(math.log(R_total / S_total))

    R_by_ds = np.bincount(stratum_to_dataset_code, weights=r, minlength=int(n_datasets)).astype(float, copy=False)
    S_by_ds = np.bincount(stratum_to_dataset_code, weights=s, minlength=int(n_datasets)).astype(float, copy=False)
    ds = np.arange(int(n_datasets), dtype=int)

    boots = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        samp = rng.choice(ds, size=ds.size, replace=True)
        Rb = float(R_by_ds[samp].sum())
        Sb = float(S_by_ds[samp].sum())
        boots[i] = float(math.log(Rb / Sb))

    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    Ts = float(math.exp(delta))
    return MhResult(
        delta=delta,
        ci_low=float(lo),
        ci_high=float(hi),
        Ts_fold=Ts,
        Ts_ci_low=float(math.exp(lo)),
        Ts_ci_high=float(math.exp(hi)),
    )


def _naive_log_or(G: np.ndarray, y: np.ndarray, *, cc: float) -> float:
    a = float(np.sum(G & (y == 1))) + float(cc)
    b = float(np.sum(G & (y == 0))) + float(cc)
    c = float(np.sum((~G) & (y == 1))) + float(cc)
    d = float(np.sum((~G) & (y == 0))) + float(cc)
    return float(math.log((a * d) / (b * c)))


def _perm_pvalue_stratified_hypergeom(
    a0: np.ndarray,
    b0: np.ndarray,
    c0: np.ndarray,
    d0: np.ndarray,
    *,
    cc: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    # Permute y within each stratum: keep margins fixed (g1=a+b, y1=a+c, n).
    # Under null, a ~ Hypergeom(ngood=y1, nbad=n-y1, nsample=g1).
    g1 = a0 + b0
    y1 = a0 + c0
    n = a0 + b0 + c0 + d0

    # Drop strata with no variation or insufficient counts; they contribute negligible or undefined info.
    ok = (g1 > 0) & (g1 < n) & (y1 > 0) & (y1 < n) & (n > 0)
    if not np.any(ok):
        return float("nan")

    g1 = g1[ok].astype(int, copy=False)
    y1 = y1[ok].astype(int, copy=False)
    n = n[ok].astype(int, copy=False)
    a0 = a0[ok]
    b0 = b0[ok]
    c0 = c0[ok]
    d0 = d0[ok]

    # Observed delta (with continuity correction) using ok strata only.
    a_obs = a0 + float(cc)
    b_obs = b0 + float(cc)
    c_obs = c0 + float(cc)
    d_obs = d0 + float(cc)
    n_obs = a_obs + b_obs + c_obs + d_obs
    delta_obs = _mh_delta_from_components(a_obs, b_obs, c_obs, d_obs, n_obs)
    if not math.isfinite(delta_obs):
        return float("nan")

    extreme = 0
    for _ in range(int(n_perm)):
        a = rng.hypergeometric(ngood=y1, nbad=n - y1, nsample=g1).astype(float, copy=False) + float(cc)
        b = (g1.astype(float, copy=False) - (a - float(cc))) + float(cc)
        c = (y1.astype(float, copy=False) - (a - float(cc))) + float(cc)
        d = (n.astype(float, copy=False) - (a - float(cc)) - (b - float(cc)) - (c - float(cc))) + float(cc)
        nn = a + b + c + d
        delta = _mh_delta_from_components(a, b, c, d, nn)
        if abs(delta) >= abs(delta_obs):
            extreme += 1
    return float((extreme + 1) / (int(n_perm) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-gene validation suite for BrdU->EdU persistence (MH-based).")
    ap.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    ap.add_argument(
        "--genes-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "scripts/_out/ts_scan_gate_mode_a_phase_matched/mh_phase_consistency_leiden_no_cnksr2/mh_phase_consistency_top_genes.csv"
        ),
        help="CSV with at least columns: gene,q. (Defaults to current top-30 list excluding Cnksr2.)",
    )
    ap.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    ap.add_argument("--theta-bins", type=int, default=12)
    ap.add_argument("--theta-bins-grid", type=int, nargs="*", default=[8, 12, 16])
    ap.add_argument(
        "--theta-bin-mode",
        type=str,
        choices=["quantile", "angle"],
        default="quantile",
        help="Theta binning: quantile within dataset vs fixed absolute angular bins.",
    )
    ap.add_argument("--q-grid", type=float, nargs="*", default=[0.8, 0.9, 0.95])
    ap.add_argument("--n-bootstrap", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cc", type=float, default=0.5, help="Haldane-Anscombe continuity correction per stratum cell.")
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("scripts/_out/gene_validation_suite_no_cnksr2"))
    ap.add_argument(
        "--genes",
        type=str,
        default="",
        help="Optional comma-separated subset of genes to run (overrides --genes-csv gene list).",
    )
    ap.add_argument("--skip-perm", action="store_true", help="Skip stratified permutation p-value (faster).")
    args = ap.parse_args()

    import anndata as ad
    import matplotlib.pyplot as plt

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    genes_df = pd.read_csv(args.genes_csv)
    if "gene" not in genes_df.columns or "q" not in genes_df.columns:
        raise ValueError("--genes-csv must have columns: gene,q")
    genes_all = genes_df["gene"].astype(str).tolist()
    q_primary_all = genes_df["q"].astype(float).to_numpy()

    if args.genes.strip():
        want = [g.strip() for g in args.genes.split(",") if g.strip()]
        keep = np.isin(np.asarray(genes_all, dtype=object), np.asarray(want, dtype=object))
        if not np.any(keep):
            raise ValueError(f"--genes did not match any genes in --genes-csv: {want}")
        genes = [g for g, k in zip(genes_all, keep, strict=True) if k]
        q_primary = q_primary_all[keep]
        genes_df = genes_df.loc[keep].copy()
    else:
        genes = genes_all
        q_primary = q_primary_all

    def qkey(q: float) -> float:
        # Avoid float-key surprises when joining q grids coming from CSV parsing.
        return float(np.round(float(q), 6))

    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    idx = np.asarray(obs["brdu_pos"].to_numpy() == 1, dtype=bool).nonzero()[0]
    if idx.size == 0:
        raise ValueError("No BrdU+ cells found (obs['brdu_pos'] == 1).")

    dataset = obs["dataset"].astype(str).to_numpy()[idx].astype(object)
    leiden = obs["leiden"].astype(str).to_numpy()[idx].astype(object)
    y = obs["edu_pos"].to_numpy()[idx].astype(int)

    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    theta = _compute_tricycle_theta(adata, idx, trc=trc, dataset_for_idx=dataset)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    X_all = _log1p_raw_for_genes(adata, idx, genes=genes)
    R_all = _residualize_matrix_within_dataset(X_all, sin_t=sin_t, cos_t=cos_t, dataset=dataset)

    q_all = sorted({qkey(q) for q in list(q_primary) + [float(x) for x in args.q_grid]})
    gates_by_q = _gates_for_qs_by_dataset(R_all, dataset=dataset, qs=q_all)

    # Encode categorical covariates once.
    ds_code, ds_uniq = pd.factorize(dataset, sort=True)
    lei_code, lei_uniq = pd.factorize(leiden, sort=True)

    theta_bins_map: dict[int, np.ndarray] = {}
    for nb in sorted(set([int(args.theta_bins)] + [int(x) for x in args.theta_bins_grid])):
        theta_bins_map[nb] = _theta_bins(theta, dataset=dataset, n_bins=int(nb), mode=str(args.theta_bin_mode))

    # Precompute global stratum index and dataset mapping (for bootstrap/LOO).
    tb = theta_bins_map[int(args.theta_bins)]
    n_strata = int(ds_uniq.size) * int(args.theta_bins) * int(lei_uniq.size)
    stratum_idx = (ds_code * int(args.theta_bins) + tb) * int(lei_uniq.size) + lei_code
    stratum_to_ds_code = (np.arange(n_strata, dtype=int) // (int(args.theta_bins) * int(lei_uniq.size))).astype(int, copy=False)

    # Precompute within-Leiden stratum indices (dataset x theta) and mappings.
    leiden_pre: dict[str, dict[str, object]] = {}
    for lei in lei_uniq.tolist():
        sel = leiden == lei
        if not np.any(sel):
            continue
        ds_sub_code, ds_sub_uniq = pd.factorize(dataset[sel], sort=True)
        tb_sub = _theta_bins(theta[sel], dataset=dataset[sel], n_bins=int(args.theta_bins), mode=str(args.theta_bin_mode))
        n_str = int(ds_sub_uniq.size) * int(args.theta_bins)
        sidx = ds_sub_code * int(args.theta_bins) + tb_sub
        s_to_ds = (np.arange(n_str, dtype=int) // int(args.theta_bins)).astype(int, copy=False)
        leiden_pre[str(lei)] = {
            "sel": sel,
            "n_str": n_str,
            "sidx": sidx,
            "s_to_ds": s_to_ds,
            "n_datasets": int(ds_sub_uniq.size),
        }

    # Main scorecard rows.
    rows: list[dict[str, object]] = []
    tmp_csv = args.outdir / "gene_validation_scorecard.partial.csv"
    if tmp_csv.exists():
        tmp_csv.unlink()

    for j, gene in enumerate(genes):
        if (j % 5) == 0:
            print(f"[validate] gene {j+1}/{len(genes)}: {gene}", flush=True)
        qprim = qkey(float(q_primary[j]))
        if qprim not in gates_by_q:
            raise ValueError(f"Primary q={qprim} for gene={gene} not present in q grid: {sorted(gates_by_q)}")
        Gp = gates_by_q[qprim][:, j]

        # Primary MH within (dataset x theta_bin x leiden) at theta_bins=args.theta_bins.
        a, b, c, d, n = _mh_components_by_stratum(
            stratum_idx=stratum_idx,
            G=Gp,
            y=y,
            n_strata=n_strata,
            cc=float(args.cc),
            drop_zero_strata=False,
        )
        g = Gp.astype(float, copy=False)
        y1 = y.astype(float, copy=False)
        y0 = 1.0 - y1
        a0 = np.bincount(stratum_idx, weights=g * y1, minlength=n_strata)
        b0 = np.bincount(stratum_idx, weights=g * y0, minlength=n_strata)
        c0 = np.bincount(stratum_idx, weights=(1.0 - g) * y1, minlength=n_strata)
        d0 = np.bincount(stratum_idx, weights=(1.0 - g) * y0, minlength=n_strata)
        mh_primary = _mh_bootstrap_ci_by_dataset(
            a=a,
            b=b,
            c=c,
            d=d,
            n=n,
            stratum_to_dataset_code=stratum_to_ds_code,
            n_datasets=int(ds_uniq.size),
            rng=rng,
            n_boot=int(args.n_bootstrap),
        )

        # Naive (unstratified) effect for context.
        naive_delta = _naive_log_or(Gp, y=y, cc=float(args.cc))

        # Per-Leiden effects: MH within each Leiden, strata=(dataset x theta_bin) at theta_bins=args.theta_bins.
        per_leiden: dict[str, MhResult] = {}
        for lei, pre in leiden_pre.items():
            sel = pre["sel"]
            sidx = pre["sidx"]
            n_str = int(pre["n_str"])
            aL, bL, cL, dL, nL = _mh_components_by_stratum(
                stratum_idx=sidx,
                G=Gp[sel],
                y=y[sel],
                n_strata=n_str,
                cc=float(args.cc),
                drop_zero_strata=False,
            )
            str_to_ds = pre["s_to_ds"]
            per_leiden[str(lei)] = _mh_bootstrap_ci_by_dataset(
                a=aL,
                b=bL,
                c=cL,
                d=dL,
                n=nL,
                stratum_to_dataset_code=str_to_ds,
                n_datasets=int(pre["n_datasets"]),
                rng=rng,
                n_boot=int(args.n_bootstrap),
            )

        # LOO dataset influence using MH components (r,s) decomposed by dataset.
        r = (a * d) / n
        s = (b * c) / n
        R_total = float(r.sum())
        S_total = float(s.sum())
        R_by_ds = np.bincount(stratum_to_ds_code, weights=r, minlength=int(ds_uniq.size)).astype(float, copy=False)
        S_by_ds = np.bincount(stratum_to_ds_code, weights=s, minlength=int(ds_uniq.size)).astype(float, copy=False)

        delta_loo = np.empty(int(ds_uniq.size), dtype=float)
        for di in range(int(ds_uniq.size)):
            Rl = R_total - float(R_by_ds[di])
            Sl = S_total - float(S_by_ds[di])
            delta_loo[di] = float(math.log(Rl / Sl))
        loo_max_abs_dev = float(np.max(np.abs(delta_loo - mh_primary.delta)))
        loo_any_sign_flip = bool(np.any(np.sign(delta_loo) != np.sign(mh_primary.delta)))

        # Per-animal deltas pooled across datasets belonging to each animal.
        an_of_ds = np.array([_dataset_animal(d) for d in ds_uniq.tolist()], dtype=object)
        an_ds_code, an_ds_uniq = pd.factorize(an_of_ds, sort=True)
        R_by_an = np.bincount(an_ds_code, weights=R_by_ds, minlength=int(an_ds_uniq.size)).astype(float, copy=False)
        S_by_an = np.bincount(an_ds_code, weights=S_by_ds, minlength=int(an_ds_uniq.size)).astype(float, copy=False)
        delta_by_an = np.log(R_by_an / S_by_an)
        an_sign_rate = float(np.mean(np.sign(delta_by_an) == np.sign(mh_primary.delta)))

        # Sensitivity: theta bin grid.
        theta_grid_deltas: dict[int, float] = {}
        for nb in sorted(set(int(x) for x in args.theta_bins_grid)):
            tb2 = theta_bins_map[int(nb)]
            n_str2 = int(ds_uniq.size) * int(nb) * int(lei_uniq.size)
            sidx2 = (ds_code * int(nb) + tb2) * int(lei_uniq.size) + lei_code
            a2, b2, c2, d2, n2 = _mh_components_by_stratum(
                stratum_idx=sidx2,
                G=Gp,
                y=y,
                n_strata=n_str2,
                cc=float(args.cc),
                drop_zero_strata=False,
            )
            theta_grid_deltas[int(nb)] = _mh_delta_from_components(a2, b2, c2, d2, n2)
        theta_sign_stable = bool(
            np.all([np.sign(v) == np.sign(mh_primary.delta) for v in theta_grid_deltas.values() if math.isfinite(v)])
        )

        # Sensitivity: q-grid effects at primary theta bins.
        q_grid_deltas: dict[float, float] = {}
        for q in [qkey(float(x)) for x in args.q_grid]:
            Gq = gates_by_q[q][:, j]
            aq, bq, cq, dq, nq = _mh_components_by_stratum(
                stratum_idx=stratum_idx,
                G=Gq,
                y=y,
                n_strata=n_strata,
                cc=float(args.cc),
                drop_zero_strata=False,
            )
            q_grid_deltas[float(q)] = _mh_delta_from_components(aq, bq, cq, dq, nq)
        q_sign_stable = bool(
            np.all([np.sign(v) == np.sign(mh_primary.delta) for v in q_grid_deltas.values() if math.isfinite(v)])
        )

        # Continuity correction robustness (same strata).
        drop_a, drop_b, drop_c, drop_d, drop_n = _mh_components_by_stratum(
            stratum_idx=stratum_idx,
            G=Gp,
            y=y,
            n_strata=n_strata,
            cc=0.0,
            drop_zero_strata=True,
        )
        mh_drop_zero = _mh_delta_from_components(drop_a, drop_b, drop_c, drop_d, drop_n)

        # Permutation p-value (within dataset x theta x leiden strata), hypergeometric null.
        if args.skip_perm:
            p_perm = float("nan")
        else:
            p_perm = _perm_pvalue_stratified_hypergeom(
                a0=a0,
                b0=b0,
                c0=c0,
                d0=d0,
                cc=float(args.cc),
                n_perm=int(args.n_perm),
                rng=rng,
            )

        row: dict[str, object] = {
            "gene": gene,
            "q_primary": float(q_primary[j]),
            "naive_delta": float(naive_delta),
            "naive_Ts_fold": float(math.exp(naive_delta)),
            "mh_delta_ds_theta_leiden": float(mh_primary.delta),
            "mh_delta_ci_low": float(mh_primary.ci_low),
            "mh_delta_ci_high": float(mh_primary.ci_high),
            "mh_Ts_fold": float(mh_primary.Ts_fold),
            "mh_Ts_ci_low": float(mh_primary.Ts_ci_low),
            "mh_Ts_ci_high": float(mh_primary.Ts_ci_high),
            "loo_max_abs_dev": float(loo_max_abs_dev),
            "loo_any_sign_flip": bool(loo_any_sign_flip),
            "animal_sign_rate": float(an_sign_rate),
            "theta_sign_stable": bool(theta_sign_stable),
            "q_sign_stable": bool(q_sign_stable),
            "mh_delta_drop_zero_strata": float(mh_drop_zero),
            "perm_p_two_sided": float(p_perm),
        }

        for lei, res in per_leiden.items():
            row[f"mh_Ts_fold_L{lei}"] = float(res.Ts_fold)
            row[f"mh_Ts_ci_low_L{lei}"] = float(res.Ts_ci_low)
            row[f"mh_Ts_ci_high_L{lei}"] = float(res.Ts_ci_high)
            row[f"mh_delta_L{lei}"] = float(res.delta)

        for nb, dlt in theta_grid_deltas.items():
            row[f"mh_delta_thetaBins{nb}"] = float(dlt)

        for q, dlt in q_grid_deltas.items():
            q_label = str(q).replace(".", "p")
            row[f"mh_delta_q{q_label}"] = float(dlt)

        rows.append(row)
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)

    out = pd.DataFrame(rows)
    out = out.merge(genes_df.drop(columns=["q"], errors="ignore").rename(columns={"q": "q_primary"}), on="gene", how="left")
    out = out.sort_values("mh_Ts_fold", ascending=False)
    out.to_csv(args.outdir / "gene_validation_scorecard.csv", index=False)

    # Overview figure: within-Leiden MH Ts_fold with CIs, plus naive Ts_fold as points.
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 10.5), dpi=170)
    ytick = np.arange(out.shape[0])
    ax.hlines(ytick, out["mh_Ts_ci_low"], out["mh_Ts_ci_high"], color="black", lw=1.2, alpha=0.8)
    ax.plot(out["mh_Ts_fold"], ytick, "o", color="black", ms=4, label="MH (dataset x theta x leiden)")
    ax.plot(out["naive_Ts_fold"], ytick, "o", color="#cc3d3d", ms=3, alpha=0.75, label="Naive (unstratified)")
    ax.axvline(1.0, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_yticks(ytick)
    ax.set_yticklabels(out["gene"].astype(str).tolist())
    ax.set_xlabel("Ts_fold (odds ratio scale)")
    ax.set_title("BrdU+ EdU odds: naive vs within-Leiden matched MH (with 95% CI)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.outdir / "overview_forest.png")


if __name__ == "__main__":
    main()
