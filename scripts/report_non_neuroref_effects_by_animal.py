from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _dataset_orientation(dataset: str) -> str:
    s = str(dataset).lower()
    if "sag" in s:
        return "sag"
    if "coro" in s:
        return "coro"
    raise ValueError(f"Cannot infer orientation from dataset={dataset!r}.")


def _dataset_animal(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r}.")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _fit_delta_gate_offset(eta0_pos: np.ndarray, y_pos: np.ndarray, *, max_iter: int = 50, tol: float = 1e-8) -> float:
    """
    Fit delta in: logit(p) = eta0 + delta * G.
    Only G==1 rows affect delta, so we pass eta0_pos/y_pos for the G==1 subset.
    """
    eta0_pos = np.asarray(eta0_pos, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)
    if eta0_pos.size == 0:
        return 0.0

    delta = 0.0
    for _ in range(int(max_iter)):
        p = _expit(eta0_pos + delta)
        w = p * (1.0 - p)
        h = float(w.sum())
        if h <= 0.0:
            break
        g = float((y_pos - p).sum())
        step = g / h
        delta_new = delta + step
        delta = delta_new
        if abs(step) < float(tol):
            break
    return float(delta)


def _compute_tricycle_theta(
    adata: ad.AnnData,
    idx: np.ndarray,
    *,
    trc: pd.DataFrame,
    dataset_for_idx: np.ndarray,
) -> np.ndarray:
    shared = sorted(set(trc["symbol"].astype(str)) & set(adata.var_names.astype(str)))
    if not shared:
        raise ValueError("No shared genes between tricycle ref and adata.var_names.")

    loadings = (
        trc[trc["symbol"].isin(shared)]
        .set_index("symbol")
        .reindex(shared)
        .reset_index()[["pc1.rot", "pc2.rot"]]
        .to_numpy(dtype=np.float32)
    )

    X = adata[idx, shared].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # within-dataset centering
    Xc = np.empty_like(X)
    for d in np.unique(dataset_for_idx):
        sel = dataset_for_idx == d
        Xc[sel] = X[sel] - np.mean(X[sel], axis=0, keepdims=True)

    pls = Xc @ loadings
    theta = (np.arctan2(pls[:, 1], pls[:, 0]) + 2 * np.pi) % (2 * np.pi)
    return theta.astype(np.float32, copy=False)


def _build_X0_combined(
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    orientation: np.ndarray,
    dataset: np.ndarray,
    animal: np.ndarray,
    leiden: np.ndarray,
    theta: np.ndarray,
    *,
    spline_n_knots: int,
    spline_degree: int,
) -> sp.csr_matrix:
    ap_um = np.asarray(ap_um, dtype=float).reshape(-1, 1)
    ml_um = np.asarray(ml_um, dtype=float).reshape(-1, 1)
    ori = np.asarray(orientation, dtype=object)
    is_sag = (ori == "sag").astype(float).reshape(-1, 1)
    is_coro = (ori == "coro").astype(float).reshape(-1, 1)

    spline_ap = SplineTransformer(
        n_knots=int(spline_n_knots), degree=int(spline_degree), include_bias=False, sparse_output=True
    )
    spline_ml = SplineTransformer(
        n_knots=int(spline_n_knots), degree=int(spline_degree), include_bias=False, sparse_output=True
    )
    X_ap = spline_ap.fit_transform(ap_um).multiply(is_sag)
    X_ml = spline_ml.fit_transform(ml_um).multiply(is_coro)
    X_spline = sp.hstack([X_ap, X_ml], format="csr")

    X_dataset = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(dataset.reshape(-1, 1))
    X_animal = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(animal.reshape(-1, 1))
    X_leiden = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(leiden.reshape(-1, 1))
    X_ori = sp.csr_matrix(is_coro.astype(float))

    th = np.asarray(theta, dtype=float)
    X_tri = sp.csr_matrix(np.column_stack([np.sin(th), np.cos(th)]).astype(float, copy=False))

    return sp.hstack([X_spline, X_dataset, X_animal, X_leiden, X_ori, X_tri], format="csr")


def _gate_raw_quantile(values: np.ndarray, ds_codes: np.ndarray, n_datasets: int, *, q: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ds_codes = np.asarray(ds_codes, dtype=int)
    thr = np.zeros(n_datasets, dtype=np.float64)
    for d in range(int(n_datasets)):
        sel = ds_codes == d
        thr[d] = float(np.quantile(values[sel], float(q)))
    return values >= thr[ds_codes]


def _gate_residual_quantile(
    values: np.ndarray,
    theta: np.ndarray,
    ds_codes: np.ndarray,
    n_datasets: int,
    *,
    q: float,
) -> np.ndarray:
    """
    Within each dataset, residualize values on (1, sin(theta), cos(theta)) then quantile gate on residuals.
    """
    values = np.asarray(values, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    ds_codes = np.asarray(ds_codes, dtype=int)
    resid = np.empty_like(values)

    s = np.sin(theta)
    c = np.cos(theta)
    for d in range(int(n_datasets)):
        sel = ds_codes == d
        M = np.column_stack([np.ones(int(sel.sum())), s[sel], c[sel]]).astype(np.float64, copy=False)
        inv = np.linalg.inv(M.T @ M)
        y = values[sel].reshape(-1, 1)
        b = inv @ (M.T @ y)  # (3,1)
        yhat = (M @ b).reshape(-1)
        resid[sel] = values[sel] - yhat

    return _gate_raw_quantile(resid, ds_codes, n_datasets, q=q)


def _gate_phase_stratified(
    values: np.ndarray,
    theta: np.ndarray,
    ds_codes: np.ndarray,
    n_datasets: int,
    *,
    q: float,
    phase_bins: int,
) -> np.ndarray:
    """
    Within each dataset, bin theta into quantile bins, then select the top (1-q) fraction by values within each bin.
    This makes gene+ and gene- have matched theta distributions by construction (within each dataset).
    """
    values = np.asarray(values, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    ds_codes = np.asarray(ds_codes, dtype=int)
    phase_bins = int(phase_bins)
    if phase_bins < 2:
        raise ValueError("phase_bins must be >=2")

    edges = np.linspace(0.0, 1.0, phase_bins + 1, dtype=np.float64)
    phase_bin = np.zeros(values.shape[0], dtype=np.int16)
    for d in range(int(n_datasets)):
        sel = ds_codes == d
        th = theta[sel]
        qs = np.quantile(th, edges)
        qs[0] = -np.inf
        qs[-1] = np.inf
        phase_bin[sel] = np.digitize(th, qs[1:-1], right=False).astype(np.int16)

    G = np.zeros(values.shape[0], dtype=bool)
    for d in range(int(n_datasets)):
        ds_sel = ds_codes == d
        for b in range(phase_bins):
            sel = ds_sel & (phase_bin == b)
            n = int(sel.sum())
            if n == 0:
                continue
            k = int(np.ceil((1.0 - float(q)) * n))
            if k <= 0:
                continue
            if k >= n:
                G[sel] = True
                continue
            vals = values[sel]
            # deterministic tie-breaking: epsilon by within-bin order
            eps = (np.arange(n, dtype=np.float64) / float(n)) * 1e-6
            adj = vals + eps
            loc = np.flatnonzero(sel)
            pos_loc = loc[np.argpartition(adj, n - k)[n - k :]]
            G[pos_loc] = True
    return G


def _pearson_residuals_by_dataset(
    X_raw: np.ndarray,
    total_counts: np.ndarray,
    dataset_codes: np.ndarray,
    n_datasets: int,
    *,
    theta: float,
    clip: float | None,
    block_size: int = 256,
) -> np.ndarray:
    if not (float(theta) > 0.0):
        raise ValueError(f"theta must be > 0, got {theta!r}")
    X_raw = np.asarray(X_raw, dtype=np.float32)
    total_counts = np.asarray(total_counts, dtype=np.float32)
    dataset_codes = np.asarray(dataset_codes, dtype=int)
    n_cells, n_genes = X_raw.shape
    out = np.empty((n_cells, n_genes), dtype=np.float32)

    for d in range(int(n_datasets)):
        sel = dataset_codes == d
        if not bool(np.any(sel)):
            continue
        tt = total_counts[sel]
        denom = float(tt.sum(dtype=np.float64))
        if not (denom > 0.0):
            out[sel, :] = 0.0
            continue
        Xd = X_raw[sel, :]
        p = (Xd.sum(axis=0, dtype=np.float64) / denom).astype(np.float32, copy=False)
        for start in range(0, n_genes, int(block_size)):
            end = min(start + int(block_size), n_genes)
            pp = p[start:end]
            mu = tt[:, None] * pp[None, :]
            var = mu + (mu * mu) / float(theta)
            var = np.maximum(var, 1e-12, dtype=np.float32)
            r = (Xd[:, start:end] - mu) / np.sqrt(var)
            if clip is not None:
                r = np.clip(r, -float(clip), float(clip))
            out[sel, start:end] = r.astype(np.float32, copy=False)
    return out


@dataclass(frozen=True)
class Criteria:
    min_animals: int
    min_pos_per_animal: int
    max_opposite_abs_delta: float
    strong_abs_delta: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List non-neuroRef genes with effect sizes and animal replication checks.")
    p.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    p.add_argument("--tricycle-ref-csv", type=pathlib.Path, default=pathlib.Path("neuroRef.csv"))
    p.add_argument("--bestq-csv", type=pathlib.Path, required=True)
    p.add_argument("--out-csv", type=pathlib.Path, required=True)
    # NOTE: we intentionally do NOT support phase-stratified gating (top-(1-q) per dataset×theta_bin),
    # because it forces uniform positives in theta bins where the gene should be off, which is a real
    # failure mode in zero-inflated targeted panels. Phase control belongs in the model strata, not
    # in the gate definition.
    p.add_argument("--gate-method", type=str, choices=["raw", "residual"], required=True)
    p.add_argument("--top-non-neuroref", type=int, default=50)
    p.add_argument("--min-animals", type=int, default=3)
    p.add_argument("--min-pos-per-animal", type=int, default=200)
    p.add_argument("--strong-abs-delta", type=float, default=0.1)
    p.add_argument("--max-opposite-abs-delta", type=float, default=0.1)
    p.add_argument(
        "--exclude-genes",
        type=str,
        nargs="*",
        default=[],
        help="Optional list of genes to exclude from the candidate list (e.g., known artifacts).",
    )
    p.add_argument("--max-iter-baseline", type=int, default=800, help="Max iterations for baseline LogisticRegression.")
    p.add_argument("--pearson-theta", type=float, default=100.0, help="Pearson residual theta (NB overdispersion).")
    p.add_argument(
        "--pearson-clip",
        type=str,
        default="10.0",
        help="Clip Pearson residuals to +/-clip; use 'none' to disable clipping.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    crit = Criteria(
        min_animals=int(args.min_animals),
        min_pos_per_animal=int(args.min_pos_per_animal),
        max_opposite_abs_delta=float(args.max_opposite_abs_delta),
        strong_abs_delta=float(args.strong_abs_delta),
    )

    trc = pd.read_csv(args.tricycle_ref_csv.expanduser())
    neuroref = set(trc["symbol"].astype(str))

    bestq = pd.read_csv(args.bestq_csv)
    if "gene" not in bestq.columns or "best_q" not in bestq.columns:
        raise ValueError("bestq-csv must contain columns: gene, best_q")
    if "in_tricycle_ref" in bestq.columns:
        non = bestq[bestq["in_tricycle_ref"] == False].copy()  # noqa: E712
    else:
        non = bestq[~bestq["gene"].astype(str).isin(neuroref)].copy()

    non = non.sort_values("delta_logloss_LODO", ascending=False, ignore_index=True).head(int(args.top_non_neuroref))
    exclude = {str(g) for g in (args.exclude_genes or [])}
    non = non[~non["gene"].astype(str).isin(exclude)].copy()
    genes = non["gene"].astype(str).to_list()
    q_map = dict(zip(non["gene"].astype(str), non["best_q"].astype(float)))

    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    obs = adata.obs
    dataset_all = obs["dataset"].astype(str).to_numpy()
    ori_all = np.array([_dataset_orientation(d) for d in dataset_all], dtype=object)
    animal_all = np.array([_dataset_animal(d) for d in dataset_all], dtype=object)
    leiden_all = obs["leiden"].astype(str).to_numpy()
    brdu = obs["brdu_pos"].to_numpy(bool)
    y_all = obs["edu_pos"].to_numpy(bool).astype(int)

    coords = np.asarray(adata.obsm["AP_ML_um"], dtype=float)
    ap_all = coords[:, 0]  # NOTE: AP/ML labels swapped in file; treat col0 as AP.
    ml_all = coords[:, 1]
    mask = brdu & np.isin(ori_all, ["sag", "coro"]) & np.isfinite(ap_all) & np.isfinite(ml_all)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No BrdU+ cells found after filtering.")

    dataset = dataset_all[idx]
    ori = ori_all[idx]
    animal = animal_all[idx]
    leiden = leiden_all[idx]
    y = y_all[idx]
    ap = ap_all[idx]
    ml = ml_all[idx]

    theta = _compute_tricycle_theta(adata, idx, trc=trc, dataset_for_idx=dataset)

    X0 = _build_X0_combined(
        ap,
        ml,
        ori,
        dataset,
        animal,
        leiden,
        theta,
        spline_n_knots=8,
        spline_degree=3,
    )
    clf0 = LogisticRegression(solver="saga", C=1.0, max_iter=int(args.max_iter_baseline), random_state=0)
    clf0.fit(X0, y)
    eta0 = np.asarray(clf0.decision_function(X0), dtype=np.float64)

    codes, ds_uniques = pd.factorize(dataset, sort=True)
    n_datasets = int(ds_uniques.size)
    animals = sorted(set(animal))

    if "total_counts" not in obs.columns:
        raise ValueError("obs['total_counts'] is required for Pearson residual gating.")
    total_counts = obs["total_counts"].to_numpy(dtype=np.float32, copy=False)[idx]

    var_names = np.asarray(adata.var_names, dtype=str)
    col_idx = np.array([np.where(var_names == g)[0][0] for g in genes], dtype=int)
    X_raw = adata.layers["raw"][idx, :][:, col_idx]
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)
    clip_s = str(args.pearson_clip).strip().lower()
    clip = None if clip_s in {"none", "nan"} else float(args.pearson_clip)
    X_val = _pearson_residuals_by_dataset(
        X_raw,
        total_counts,
        codes,
        n_datasets,
        theta=float(args.pearson_theta),
        clip=clip,
    ).astype(np.float64, copy=False)

    rows: list[dict[str, object]] = []
    for j, g in enumerate(genes):
        q = float(q_map[g])
        vals = X_val[:, j]

        if args.gate_method == "raw":
            G = _gate_raw_quantile(vals, codes, n_datasets, q=q)
        elif args.gate_method == "residual":
            G = _gate_residual_quantile(vals, theta, codes, n_datasets, q=q)
        else:
            raise RuntimeError(f"Unknown gate_method={args.gate_method!r}")

        delta = _fit_delta_gate_offset(eta0[G], y[G])
        ts_fold = float(np.exp(delta))
        pooled_sign = 0 if delta == 0.0 else (1 if delta > 0.0 else -1)

        per_an_delta: dict[str, float] = {}
        per_an_npos: dict[str, int] = {}
        has_opposite = False
        strong_same = 0
        for an in animals:
            sel = animal == an
            n_pos = int((sel & G).sum())
            per_an_npos[an] = n_pos
            d_an = _fit_delta_gate_offset(eta0[sel & G], y[sel & G])
            per_an_delta[an] = float(d_an)

            if n_pos >= crit.min_pos_per_animal and pooled_sign != 0 and d_an != 0.0:
                an_sign = 1 if d_an > 0.0 else -1
                if an_sign != pooled_sign and abs(d_an) >= crit.max_opposite_abs_delta:
                    has_opposite = True
                if an_sign == pooled_sign and abs(d_an) >= crit.strong_abs_delta:
                    strong_same += 1

        passes = (not has_opposite) and (strong_same >= crit.min_animals)

        row: dict[str, object] = {
            "gene": g,
            "best_q": q,
            "gate_method": str(args.gate_method),
            "phase_bins": None,
            "delta": float(delta),
            "Ts_fold": float(ts_fold),
            "passes_not_one_animal": bool(passes),
            "strong_same_animals": int(strong_same),
        }
        for an in animals:
            row[f"delta_{an}"] = per_an_delta[an]
            row[f"n_pos_{an}"] = per_an_npos[an]
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Ts_fold", ascending=False, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    passing = out[out["passes_not_one_animal"] == True].copy()  # noqa: E712
    print(f"animals={animals}")
    print(f"gate_method={args.gate_method}  non_neuroref_genes_tested={len(genes)}  passing={passing.shape[0]}")

    if not passing.empty:
        show_cols = ["gene", "best_q", "delta", "Ts_fold", "strong_same_animals"] + [f"delta_{an}" for an in animals]
        show = passing.sort_values("delta", key=lambda s: s.abs(), ascending=False)[show_cols].head(25)
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
