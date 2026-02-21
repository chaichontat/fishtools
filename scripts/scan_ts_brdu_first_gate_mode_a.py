from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


def _expit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _log_loss_sum(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).sum())


def _dataset_orientation(dataset: str) -> str:
    s = str(dataset).lower()
    if "sag" in s:
        return "sag"
    if "coro" in s:
        return "coro"
    raise ValueError(f"Cannot infer orientation from dataset={dataset!r} (expected contains 'Sag' or 'Coro').")


def _dataset_animal(dataset: str) -> str:
    m = re.search(r"(jaxa\d+)", str(dataset), flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer animal from dataset={dataset!r} (expected contains 'JaxA#').")
    s = m.group(1)
    return s[0].upper() + s[1:]


def _fit_delta_gate_offset(
    eta0_pos: np.ndarray,
    y_pos: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """
    Fit delta in: logit(p) = eta0 + delta * G, where G is binary and we pass only G==1 rows.

    Returns (delta, se_delta) with se from observed information: I = sum p(1-p) over G==1 rows.
    """
    eta0_pos = np.asarray(eta0_pos, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)
    if eta0_pos.ndim != 1 or y_pos.ndim != 1:
        raise ValueError("eta0_pos and y_pos must be 1D")
    if eta0_pos.shape[0] != y_pos.shape[0]:
        raise ValueError("eta0_pos/y_pos length mismatch")
    if eta0_pos.size == 0:
        return 0.0, float("inf")

    delta = 0.0
    for _ in range(max_iter):
        p = _expit(eta0_pos + delta)
        w = p * (1.0 - p)
        h = float(w.sum())
        if h <= 0.0:
            break
        g = float((y_pos - p).sum())
        step = g / h
        delta_new = delta + step
        if abs(step) < tol:
            delta = delta_new
            break
        delta = delta_new

    # Observed information at optimum.
    p = _expit(eta0_pos + delta)
    info = float((p * (1.0 - p)).sum())
    se = float(1.0 / np.sqrt(info)) if info > 0.0 else float("inf")
    return float(delta), se


@dataclass(frozen=True)
class Config:
    delta_t_min: float
    n_splits: int
    spline_n_knots: int
    spline_degree: int
    C: float
    max_iter_baseline: int
    max_iter_delta: int
    quantiles: tuple[float, ...]
    min_pos_cells: int
    min_neg_cells: int
    min_datasets: int
    phase_matched: bool
    phase_matched_method: str
    phase_bins: int
    residual_block_size: int
    pearson_theta: float
    pearson_clip: float | None
    random_state: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mode A (single-gene gating) discovery for BrdU-first assay.\n"
            "On BrdU+ cells: define per-dataset quantile gates on log1p(raw) and evaluate gate effect on EdU positivity\n"
            "using a baseline offset model and GroupKFold held-out datasets."
        )
    )
    p.add_argument("--h5ad", type=pathlib.Path, default=pathlib.Path("~/nvme/all_progenitors.h5ad"))
    p.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("scripts/_out/ts_scan_gate_mode_a"))
    p.add_argument("--delta-t-min", type=float, default=90.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--spline-n-knots", type=int, default=8)
    p.add_argument("--spline-degree", type=int, default=3)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter-baseline", type=int, default=300)
    p.add_argument("--max-iter-delta", type=int, default=50)
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    p.add_argument("--min-pos-cells", type=int, default=200)
    p.add_argument("--min-neg-cells", type=int, default=200)
    p.add_argument("--min-datasets", type=int, default=8)
    p.add_argument(
        "--phase-matched",
        action="store_true",
        help=(
            "If set, define the gene gate on residualized expression after regressing log1p(raw) on "
            "sin(theta),cos(theta) within each dataset (tricycle-phase matched gating)."
        ),
    )
    p.add_argument(
        "--phase-matched-method",
        type=str,
        choices=["residual", "stratified"],
        default="residual",
        help=(
            "Only used when --phase-matched is set. "
            "residual: gate on residualized expression (log1p(raw) residualized on sin/cos(theta)). "
            "stratified: (DEPRECATED) gate within theta bins per dataset (forces uniform positives per bin). "
            "This is not appropriate for zero-inflated targeted panels and is disabled in this pipeline."
        ),
    )
    p.add_argument(
        "--phase-bins",
        type=int,
        default=12,
        help="Only used for --phase-matched-method=stratified (deprecated). Number of within-dataset theta bins.",
    )
    p.add_argument("--residual-block-size", type=int, default=64)
    p.add_argument("--pearson-theta", type=float, default=100.0, help="Pearson residual theta (NB overdispersion).")
    p.add_argument(
        "--pearson-clip",
        type=str,
        default="10.0",
        help="Clip Pearson residuals to +/-clip; use 'none' to disable clipping.",
    )
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument(
        "--tricycle-ref-csv",
        type=pathlib.Path,
        default=pathlib.Path("neuroRef.csv"),
        help="Path to tricycle reference CSV (symbols + pc1.rot/pc2.rot).",
    )
    p.add_argument("--top-plot-n", type=int, default=30)
    p.add_argument("--top-forest-n", type=int, default=5)
    return p.parse_args()


def _load_tricycle_ref(path: pathlib.Path) -> pd.DataFrame:
    trc = pd.read_csv(path)
    needed = {"symbol", "pc1.rot", "pc2.rot"}
    missing = needed - set(trc.columns)
    if missing:
        raise ValueError(f"tricycle ref CSV missing columns: {sorted(missing)}")
    return trc


def _compute_tricycle_theta(
    adata: ad.AnnData,
    idx: np.ndarray,
    *,
    trc: pd.DataFrame,
    dataset_for_idx: np.ndarray,
) -> np.ndarray:
    shared = sorted(set(trc["symbol"]) & set(adata.var_names))
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

    batch_values = np.asarray(dataset_for_idx, dtype=object)
    X_centered = np.empty_like(X)
    for value in np.unique(batch_values):
        sel = batch_values == value
        X_sel = X[sel]
        X_centered[sel] = X_sel - np.mean(X_sel, axis=0, keepdims=True)

    pls = X_centered @ loadings
    theta = (np.arctan2(pls[:, 1], pls[:, 0]) + 2 * np.pi) % (2 * np.pi)
    return theta.astype(np.float32, copy=False)


def _build_X0_combined(
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    orientation: np.ndarray,
    dataset: np.ndarray,
    animal: np.ndarray,
    leiden: np.ndarray,
    tricycle_theta: np.ndarray,
    cfg: Config,
) -> sp.csr_matrix:
    ap_um = np.asarray(ap_um, dtype=float).reshape(-1, 1)
    ml_um = np.asarray(ml_um, dtype=float).reshape(-1, 1)
    ori = np.asarray(orientation, dtype=object)
    is_sag = (ori == "sag").astype(float).reshape(-1, 1)
    is_coro = (ori == "coro").astype(float).reshape(-1, 1)

    spline_ap = SplineTransformer(
        n_knots=cfg.spline_n_knots,
        degree=cfg.spline_degree,
        include_bias=False,
        sparse_output=True,
    )
    spline_ml = SplineTransformer(
        n_knots=cfg.spline_n_knots,
        degree=cfg.spline_degree,
        include_bias=False,
        sparse_output=True,
    )
    X_ap = spline_ap.fit_transform(ap_um).multiply(is_sag)
    X_ml = spline_ml.fit_transform(ml_um).multiply(is_coro)
    X_spline = sp.hstack([X_ap, X_ml], format="csr")

    X_dataset = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(dataset.reshape(-1, 1))
    X_animal = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(animal.reshape(-1, 1))
    X_leiden = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float).fit_transform(leiden.reshape(-1, 1))

    X_ori = sp.csr_matrix(is_coro.astype(float))

    theta = np.asarray(tricycle_theta, dtype=float)
    X_tri = np.column_stack([np.sin(theta), np.cos(theta)]).astype(float, copy=False)
    X_tri = sp.csr_matrix(X_tri)

    return sp.hstack([X_spline, X_dataset, X_animal, X_leiden, X_ori, X_tri], format="csr")


def _fit_baseline(X0: sp.csr_matrix, y: np.ndarray, *, C: float, max_iter: int, random_state: int) -> LogisticRegression:
    clf = LogisticRegression(solver="saga", C=C, max_iter=max_iter, random_state=random_state)
    clf.fit(X0, y)
    return clf


def _residualize_against_tricycle_within_dataset(
    X: np.ndarray,
    theta: np.ndarray,
    dataset_codes: np.ndarray,
    n_datasets: int,
    *,
    block_size: int,
) -> np.ndarray:
    """
    Residualize each gene against tricycle phase within each dataset:
      X ~ b0 + b1*sin(theta) + b2*cos(theta)
    Returns residuals with same shape as X.
    """
    X = np.asarray(X, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float64)
    dataset_codes = np.asarray(dataset_codes, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if theta.ndim != 1 or theta.shape[0] != X.shape[0]:
        raise ValueError("theta must be 1D with length n_cells")
    if dataset_codes.ndim != 1 or dataset_codes.shape[0] != X.shape[0]:
        raise ValueError("dataset_codes must be 1D with length n_cells")

    s = np.sin(theta)
    c = np.cos(theta)
    out = np.empty_like(X, dtype=np.float32)
    n_genes = int(X.shape[1])

    for d in range(int(n_datasets)):
        sel = dataset_codes == d
        n = int(sel.sum())
        if n < 10:
            raise ValueError(f"Dataset code {d} has too few cells for residualization (n={n}).")

        M = np.column_stack([np.ones(n, dtype=np.float64), s[sel], c[sel]]).astype(np.float64, copy=False)
        MtM = M.T @ M
        inv = np.linalg.inv(MtM)

        for start in range(0, n_genes, int(block_size)):
            end = min(start + int(block_size), n_genes)
            Y = X[sel, start:end].astype(np.float64, copy=False)
            B = inv @ (M.T @ Y)  # (3,b)
            Yhat = M @ B  # (n,b)
            out[sel, start:end] = (Y - Yhat).astype(np.float32, copy=False)

    return out


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

    `X_raw` is raw counts (n_cells, n_genes). This is a per-batch version of
    Scanpy's `normalize_pearson_residuals` computed in blocks to limit peak memory.
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


def _plot_top_hits(df: pd.DataFrame, *, out_png: pathlib.Path, top_n: int, title: str) -> None:
    sub = df.head(int(top_n)).copy()
    sub = sub.iloc[::-1]
    colors = np.where(sub["delta_hat"].to_numpy(float) >= 0, "#c0392b", "#2980b9")

    fig_h = max(4.0, 0.25 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), constrained_layout=True)
    ax.barh(sub["gene"], sub["delta_logloss_LODO"], color=colors, alpha=0.9)
    ax.set_xlabel("Held-out delta log loss (baseline - with gate)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_forest(
    effects: pd.DataFrame,
    *,
    out_png: pathlib.Path,
    title: str,
    group_col: str,
) -> None:
    df = effects.copy()
    df = df.sort_values("delta_hat", ascending=True, ignore_index=True)
    y = np.arange(df.shape[0], dtype=float)
    x = df["delta_hat"].to_numpy(float)
    se = df["se_hat"].to_numpy(float)
    lo = x - 1.96 * se
    hi = x + 1.96 * se

    fig_h = max(3.8, 0.28 * df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), constrained_layout=True)
    ax.hlines(y, lo, hi, color="#444444", linewidth=1.2)
    ax.plot(x, y, "o", color="#111111", markersize=4)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df[group_col].astype(str).to_list())
    ax.set_xlabel("Gate effect delta (log-odds shift among BrdU+; exp(delta)=Ts fold-change)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _run_mode_a(
    adata: ad.AnnData,
    *,
    cfg: Config,
    outdir: pathlib.Path,
    trc: pd.DataFrame,
    top_plot_n: int,
    top_forest_n: int,
) -> None:
    obs = adata.obs
    dataset_all = obs["dataset"].astype(str).to_numpy()
    ori_all = np.array([_dataset_orientation(d) for d in dataset_all], dtype=object)
    animal_all = np.array([_dataset_animal(d) for d in dataset_all], dtype=object)
    leiden_all = obs["leiden"].astype(str).to_numpy()
    brdu = obs["brdu_pos"].to_numpy(bool)
    edu = obs["edu_pos"].to_numpy(bool).astype(int)

    coords = np.asarray(adata.obsm["AP_ML_um"], dtype=float)
    ap_all = coords[:, 0]  # NOTE: AP/ML labels are swapped in file; treat col0 as AP.
    ml_all = coords[:, 1]

    mask = brdu & np.isin(ori_all, ["sag", "coro"]) & np.isfinite(ap_all) & np.isfinite(ml_all)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No BrdU+ cells for combined Sag+Coro")

    ap = ap_all[idx]
    ml = ml_all[idx]
    ori = ori_all[idx]
    dataset = dataset_all[idx]
    animal = animal_all[idx]
    leiden = leiden_all[idx]
    y = edu[idx]

    ds_codes, ds_uniques = pd.factorize(dataset, sort=True)
    n_datasets = int(ds_uniques.size)
    ds_sizes = np.bincount(ds_codes, minlength=n_datasets).astype(int)

    tri_theta = _compute_tricycle_theta(adata, idx, trc=trc, dataset_for_idx=dataset)

    if "raw" not in adata.layers:
        raise ValueError("Expected adata.layers['raw'] for raw counts.")
    X_raw = adata.layers["raw"][idx, :]
    X0 = _build_X0_combined(ap, ml, ori, dataset, animal, leiden, tri_theta, cfg)

    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)

    if "total_counts" not in obs.columns:
        raise ValueError("obs['total_counts'] is required for Pearson residual gating.")
    total_counts = obs["total_counts"].to_numpy(dtype=np.float32, copy=False)[idx]

    X_pearson = _pearson_residuals_by_dataset(
        X_raw,
        total_counts,
        ds_codes,
        n_datasets,
        theta=float(cfg.pearson_theta),
        clip=cfg.pearson_clip,
        block_size=int(cfg.residual_block_size),
    )
    del X_raw
    var_names = np.asarray(adata.var_names, dtype=object)

    in_tricycle_ref = np.isin(var_names, np.asarray(trc["symbol"].astype(str), dtype=object))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ts_scan_gate_mode_a_metadata.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8"
    )

    # Evaluate each quantile separately because small datasets can make very-high quantiles unstable.
    all_best_rows: list[pd.DataFrame] = []
    for q in cfg.quantiles:
        q = float(q)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Invalid quantile q={q}")

        # Dataset support based on expected group sizes at this q.
        min_n_for_pos = int(np.ceil(cfg.min_pos_cells / (1.0 - q)))
        min_n_for_neg = int(np.ceil(cfg.min_neg_cells / q))
        min_n = max(min_n_for_pos, min_n_for_neg)
        ds_ok = ds_sizes >= min_n
        ok_datasets = ds_uniques[ds_ok]
        if int(ok_datasets.size) < cfg.min_datasets:
            print(
                f"q={q}: only {int(ok_datasets.size)} datasets meet min size >= {min_n} "
                f"(min_pos={cfg.min_pos_cells}, min_neg={cfg.min_neg_cells}); skipping"
            )
            continue

        mask_q = np.isin(dataset, ok_datasets)
        if not bool(mask_q.any()):
            continue

        X0_q = X0[mask_q]
        X_pearson_q = X_pearson[mask_q]
        y_q = y[mask_q]
        dataset_q = dataset[mask_q]
        ds_codes_q, ds_uniques_q = pd.factorize(dataset_q, sort=True)
        n_datasets_q = int(ds_uniques_q.size)
        tri_theta_q = tri_theta[mask_q]

        n_splits_eff = min(cfg.n_splits, n_datasets_q)
        if n_splits_eff < 2:
            continue
        gkf = GroupKFold(n_splits=n_splits_eff)
        splits = list(gkf.split(X0_q, y_q, groups=dataset_q))

        # Pre-fit baseline per fold and cache logits.
        fold_eta0: list[np.ndarray] = []
        fold_tr: list[np.ndarray] = []
        fold_te: list[np.ndarray] = []
        nll_base_sum = 0.0
        n_total = 0

        print(f"q={q}: n={int(mask_q.sum()):,} datasets={n_datasets_q} genes={var_names.size}")
        for fold_idx, (tr, te) in enumerate(splits):
            clf0 = _fit_baseline(X0_q[tr], y_q[tr], C=cfg.C, max_iter=cfg.max_iter_baseline, random_state=cfg.random_state)
            eta0_all = clf0.decision_function(X0_q)
            fold_eta0.append(np.asarray(eta0_all, dtype=np.float64))
            fold_tr.append(tr)
            fold_te.append(te)

            eta0_te = eta0_all[te]
            nll_base_sum += _log_loss_sum(y_q[te], _expit(eta0_te))
            n_total += int(te.size)
            print(f"q={q}: baseline fold {fold_idx + 1}/{n_splits_eff} done")

        base_logloss = float(nll_base_sum / n_total)

        if cfg.phase_matched:
            if cfg.phase_matched_method == "residual":
                X_gate_q = _residualize_against_tricycle_within_dataset(
                    X_pearson_q,
                    tri_theta_q,
                    ds_codes_q,
                    n_datasets_q,
                    block_size=cfg.residual_block_size,
                )
                gate_source = "resid_pearson_on_sin_cos_theta_within_dataset"
            elif cfg.phase_matched_method == "stratified":
                raise ValueError(
                    "phase_matched_method='stratified' is disabled: it gates within each dataset×theta_bin, "
                    "forcing uniform positives in bins where the gene is off (a real artifact in zero-inflated panels). "
                    "Use phase_matched_method='residual' instead."
                )
            else:
                raise ValueError(f"Unknown phase_matched_method={cfg.phase_matched_method!r}")
        else:
            X_gate_q = X_pearson_q
            gate_source = "pearson"

        # Precompute per-dataset thresholds for this q (used in all non-stratified cases).
        thr = None
        if not (cfg.phase_matched and cfg.phase_matched_method == "stratified"):
            thr = np.zeros((n_datasets_q, var_names.size), dtype=np.float32)
            for d in range(n_datasets_q):
                sel = ds_codes_q == d
                thr[d, :] = np.quantile(X_gate_q[sel, :], q, axis=0).astype(np.float32, copy=False)

        # Precompute within-dataset theta bins for stratified gating.
        phase_bin = None
        if cfg.phase_matched and cfg.phase_matched_method == "stratified":
            if int(cfg.phase_bins) < 2:
                raise ValueError("phase_bins must be >= 2")
            phase_bin = np.zeros(tri_theta_q.shape[0], dtype=np.int16)
            edges = np.linspace(0.0, 1.0, int(cfg.phase_bins) + 1, dtype=np.float64)
            for d in range(n_datasets_q):
                sel = ds_codes_q == d
                theta_d = np.asarray(tri_theta_q[sel], dtype=np.float64)
                qs = np.quantile(theta_d, edges)
                qs[0] = -np.inf
                qs[-1] = np.inf
                phase_bin[sel] = np.digitize(theta_d, qs[1:-1], right=False).astype(np.int16)

        nll_gate_sum = np.zeros(var_names.size, dtype=np.float64)
        delta_pooled = np.zeros(var_names.size, dtype=np.float64)
        se_pooled = np.zeros(var_names.size, dtype=np.float64)
        valid = np.ones(var_names.size, dtype=bool)

        # Pooled baseline for effect-size reporting.
        clf0_all = _fit_baseline(X0_q, y_q, C=cfg.C, max_iter=cfg.max_iter_baseline, random_state=cfg.random_state)
        eta0_q_all = np.asarray(clf0_all.decision_function(X0_q), dtype=np.float64)

        for j in range(var_names.size):
            if cfg.phase_matched and cfg.phase_matched_method == "stratified":
                # Build a phase-stratified gate: within each dataset and theta-bin, take the top (1-q) fraction.
                G = np.zeros(X_gate_q.shape[0], dtype=bool)
                for d in range(n_datasets_q):
                    ds_sel = ds_codes_q == d
                    for b in range(int(cfg.phase_bins)):
                        sel = ds_sel & (phase_bin == b)
                        n = int(sel.sum())
                        if n == 0:
                            continue
                        k = int(np.ceil((1.0 - q) * n))
                        if k <= 0:
                            continue
                        if k >= n:
                            G[sel] = True
                            continue
                        vals = np.asarray(X_gate_q[sel, j], dtype=np.float64)
                        # Deterministic tie-breaking: add tiny increasing epsilon by local order.
                        eps = (np.arange(n, dtype=np.float64) / float(n)) * 1e-6
                        adj = vals + eps
                        loc = np.flatnonzero(sel)
                        pos_loc = loc[np.argpartition(adj, n - k)[n - k :]]
                        G[pos_loc] = True
            else:
                if thr is None:
                    raise RuntimeError("Internal error: thr is None in non-stratified gating.")
                G = X_gate_q[:, j] >= thr[ds_codes_q, j]

            # Enforce per-dataset min pos/neg (ties can break exact quantile fractions).
            ok = True
            for d in range(n_datasets_q):
                sel = ds_codes_q == d
                pos = int(G[sel].sum())
                neg = int(sel.sum() - pos)
                if pos < cfg.min_pos_cells or neg < cfg.min_neg_cells:
                    ok = False
                    break
            if not ok:
                valid[j] = False
                continue

            # CV evaluation: fit delta per fold and score on held-out datasets.
            nll_j = 0.0
            for eta0_all_fold, tr, te in zip(fold_eta0, fold_tr, fold_te, strict=True):
                eta0_tr = eta0_all_fold[tr]
                eta0_te = eta0_all_fold[te]
                y_tr = y_q[tr]
                y_te = y_q[te]
                G_tr = G[tr]
                G_te = G[te]

                delta_hat, _ = _fit_delta_gate_offset(
                    eta0_tr[G_tr],
                    y_tr[G_tr],
                    max_iter=cfg.max_iter_delta,
                )
                p_te = _expit(eta0_te + delta_hat * G_te.astype(float))
                nll_j += _log_loss_sum(y_te, p_te)
            nll_gate_sum[j] = nll_j

            delta_hat, se_hat = _fit_delta_gate_offset(
                eta0_q_all[G],
                y_q[G],
                max_iter=cfg.max_iter_delta,
            )
            delta_pooled[j] = delta_hat
            se_pooled[j] = se_hat

        gate_logloss = (nll_gate_sum / n_total).astype(float)
        delta_logloss = base_logloss - gate_logloss

        df_q = pd.DataFrame(
            {
                "quantile_q": q,
                "gate_source": gate_source,
                "phase_matched": bool(cfg.phase_matched),
                "phase_matched_method": str(cfg.phase_matched_method),
                "gene": var_names,
                "in_tricycle_ref": in_tricycle_ref,
                "base_logloss_LODO": base_logloss,
                "logloss_LODO_with_gate": gate_logloss,
                "delta_logloss_LODO": delta_logloss,
                "delta_hat": delta_pooled.astype(float),
                "se_hat": se_pooled.astype(float),
                "Ts_fold_change_exp_delta": np.exp(delta_pooled.astype(float)),
                "n_cells": int(mask_q.sum()),
                "n_datasets": n_datasets_q,
                "valid_gate_all_datasets": valid,
            }
        )
        df_q = df_q[df_q["valid_gate_all_datasets"]].sort_values("delta_logloss_LODO", ascending=False, ignore_index=True)

        out_csv = outdir / f"ts_scan_gate_mode_a_combined_q{str(q).replace('.', 'p')}.csv"
        df_q.to_csv(out_csv, index=False)
        print(f"q={q}: wrote {out_csv}")

        title = (
            f"Mode A gate scan (combined Sag+Coro)  q={q}  n={int(mask_q.sum()):,}  "
            f"datasets={n_datasets_q}  base logloss={base_logloss:.4f}"
        )
        out_png = outdir / f"ts_scan_gate_mode_a_combined_q{str(q).replace('.', 'p')}_top{int(top_plot_n)}.png"
        _plot_top_hits(df_q, out_png=out_png, top_n=top_plot_n, title=title)
        print(f"q={q}: wrote {out_png}")

        # Per-dataset and per-animal forest plots for top hits (pooled baseline offsets).
        for rank in range(min(int(top_forest_n), df_q.shape[0])):
            gene = str(df_q.loc[rank, "gene"])
            j = int(np.where(var_names == gene)[0][0])
            if cfg.phase_matched and cfg.phase_matched_method == "stratified":
                # Recompute G for this gene for diagnostics.
                G = np.zeros(X_gate_q.shape[0], dtype=bool)
                for d in range(n_datasets_q):
                    ds_sel = ds_codes_q == d
                    for b in range(int(cfg.phase_bins)):
                        sel = ds_sel & (phase_bin == b)
                        n = int(sel.sum())
                        if n == 0:
                            continue
                        k = int(np.ceil((1.0 - q) * n))
                        if k <= 0:
                            continue
                        if k >= n:
                            G[sel] = True
                            continue
                        vals = np.asarray(X_gate_q[sel, j], dtype=np.float64)
                        eps = (np.arange(n, dtype=np.float64) / float(n)) * 1e-6
                        adj = vals + eps
                        loc = np.flatnonzero(sel)
                        pos_loc = loc[np.argpartition(adj, n - k)[n - k :]]
                        G[pos_loc] = True
            else:
                if thr is None:
                    raise RuntimeError("Internal error: thr is None in non-stratified diagnostics.")
                G = X_gate_q[:, j] >= thr[ds_codes_q, j]

            per_ds_rows: list[dict[str, object]] = []
            for d, ds_name in enumerate(ds_uniques_q):
                sel = ds_codes_q == d
                delta_hat, se_hat = _fit_delta_gate_offset(
                    eta0_q_all[sel & G],
                    y_q[sel & G],
                    max_iter=cfg.max_iter_delta,
                )
                per_ds_rows.append(
                    {
                        "dataset": str(ds_name),
                        "delta_hat": float(delta_hat),
                        "se_hat": float(se_hat),
                        "n_pos": int((sel & G).sum()),
                        "n_total": int(sel.sum()),
                    }
                )
            per_ds = pd.DataFrame(per_ds_rows)
            out_ds_csv = outdir / f"ts_scan_gate_mode_a_{gene}_q{str(q).replace('.', 'p')}_per_dataset.csv"
            per_ds.to_csv(out_ds_csv, index=False)

            out_ds_png = outdir / f"ts_scan_gate_mode_a_{gene}_q{str(q).replace('.', 'p')}_per_dataset_forest.png"
            _plot_forest(per_ds, out_png=out_ds_png, title=f"{gene} gate effect by dataset (q={q})", group_col="dataset")

            per_an_rows: list[dict[str, object]] = []
            for an in sorted(set(animal[mask_q])):
                sel = (animal[mask_q] == an).astype(bool)
                delta_hat, se_hat = _fit_delta_gate_offset(
                    eta0_q_all[sel & G],
                    y_q[sel & G],
                    max_iter=cfg.max_iter_delta,
                )
                per_an_rows.append(
                    {
                        "animal": str(an),
                        "delta_hat": float(delta_hat),
                        "se_hat": float(se_hat),
                        "n_pos": int((sel & G).sum()),
                        "n_total": int(sel.sum()),
                    }
                )
            per_an = pd.DataFrame(per_an_rows)
            out_an_csv = outdir / f"ts_scan_gate_mode_a_{gene}_q{str(q).replace('.', 'p')}_per_animal.csv"
            per_an.to_csv(out_an_csv, index=False)
            out_an_png = outdir / f"ts_scan_gate_mode_a_{gene}_q{str(q).replace('.', 'p')}_per_animal_forest.png"
            _plot_forest(per_an, out_png=out_an_png, title=f"{gene} gate effect by animal (q={q})", group_col="animal")

        # Best-q summary per gene (within this q run).
        all_best_rows.append(df_q.assign(best_q=q))

    if not all_best_rows:
        raise RuntimeError("No quantile runs produced results; check dataset sizes / min constraints.")

    # Choose best-q per gene by maximizing delta_logloss among the runs that were valid.
    all_q = pd.concat(all_best_rows, ignore_index=True)
    all_q = all_q.sort_values(["gene", "delta_logloss_LODO"], ascending=[True, False], ignore_index=True)
    best = all_q.drop_duplicates(subset=["gene"], keep="first").sort_values("delta_logloss_LODO", ascending=False, ignore_index=True)

    out_best_csv = outdir / "ts_scan_gate_mode_a_combined_bestq.csv"
    best.to_csv(out_best_csv, index=False)
    print(f"wrote {out_best_csv}")

    out_best_png = outdir / f"ts_scan_gate_mode_a_combined_bestq_top{int(top_plot_n)}.png"
    title = f"Mode A gate scan (combined Sag+Coro) best-q per gene  n_genes={best.shape[0]}"
    _plot_top_hits(best, out_png=out_best_png, top_n=top_plot_n, title=title)
    print(f"wrote {out_best_png}")


def main() -> None:
    args = _parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    pearson_clip: float | None
    clip_s = str(args.pearson_clip).strip().lower()
    pearson_clip = None if clip_s in {"none", "nan"} else float(args.pearson_clip)

    cfg = Config(
        delta_t_min=float(args.delta_t_min),
        n_splits=int(args.n_splits),
        spline_n_knots=int(args.spline_n_knots),
        spline_degree=int(args.spline_degree),
        C=float(args.C),
        max_iter_baseline=int(args.max_iter_baseline),
        max_iter_delta=int(args.max_iter_delta),
        quantiles=tuple(float(x) for x in args.quantiles),
        min_pos_cells=int(args.min_pos_cells),
        min_neg_cells=int(args.min_neg_cells),
        min_datasets=int(args.min_datasets),
        phase_matched=bool(args.phase_matched),
        phase_matched_method=str(args.phase_matched_method),
        phase_bins=int(args.phase_bins),
        residual_block_size=int(args.residual_block_size),
        pearson_theta=float(args.pearson_theta),
        pearson_clip=pearson_clip,
        random_state=int(args.random_state),
    )

    adata = ad.read_h5ad(args.h5ad.expanduser(), backed="r")
    trc = _load_tricycle_ref(args.tricycle_ref_csv.expanduser())
    _run_mode_a(adata, cfg=cfg, outdir=outdir, trc=trc, top_plot_n=int(args.top_plot_n), top_forest_n=int(args.top_forest_n))


if __name__ == "__main__":
    main()
