from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Export a scripts/gam panel dir where cNMF programs are treated as if they were genes.\n"
            "Writes cells.tsv and counts.tsv where counts columns are P1..Pk (or Usage_1..Usage_k) pseudo-counts.\n"
            "Model is unchanged: fit_inm_panel.R will fit the same per-'gene' NB GAM, now per program."
        )
    )
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--usage-tsv", type=Path, required=True, help="usage_norm.*.tsv from cNMF (index/cell_id in first col).")
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--x-col", type=str, default="x")
    p.add_argument("--r-col", type=str, default="r_um")
    p.add_argument("--ap-col", type=str, default="ap")
    p.add_argument("--ml-col", type=str, default="ml")
    p.add_argument("--theta-col", type=str, default="tricycle")
    p.add_argument("--s-col", type=str, default="total_counts")
    p.add_argument("--batch-col", type=str, default="dataset")

    p.add_argument(
        "--program-prefix",
        type=str,
        default="P",
        help="Counts.tsv program column prefix (e.g. P => P1..Pk).",
    )
    p.add_argument(
        "--rounding",
        choices=["round", "floor", "ceil", "none"],
        default="round",
        help="How to convert pseudo-counts to integers. 'none' writes float32 counts.",
    )
    p.add_argument(
        "--count-mode",
        choices=["total_counts", "scale_only"],
        default="total_counts",
        help="Pseudo-count definition: usage * total_counts, or usage * scale.",
    )
    p.add_argument("--scale", type=float, default=1000.0, help="Used when --count-mode=scale_only.")
    p.add_argument(
        "--drop-nonfinite-apml",
        action="store_true",
        help="Drop cells with non-finite AP/ML before writing (recommended if ap/ml contain NaNs).",
    )
    args = p.parse_args()

    import anndata as ad

    adata = ad.read_h5ad(args.h5ad, backed="r")
    for col in [args.x_col, args.r_col, args.ap_col, args.ml_col, args.theta_col, args.s_col, args.batch_col]:
        if col not in adata.obs.columns:
            raise KeyError(f"{args.h5ad}: obs.{col} not found")

    usage_raw = pd.read_csv(args.usage_tsv, sep="\t")
    if usage_raw.shape[1] < 2:
        raise ValueError(f"{args.usage_tsv}: expected at least 2 columns, got {usage_raw.shape[1]}")
    cell_col = str(usage_raw.columns[0])
    usage_raw[cell_col] = usage_raw[cell_col].astype(str)
    usage = usage_raw.set_index(cell_col, drop=True)
    usage_cols = [c for c in usage.columns if str(c).startswith("Usage_")]
    if not usage_cols:
        raise ValueError(f"{args.usage_tsv}: no columns starting with 'Usage_'")
    usage = usage[usage_cols]
    usage = usage.apply(pd.to_numeric, errors="raise")
    usage_vals = usage.to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(usage_vals)):
        raise ValueError("usage has non-finite values")
    if np.any(usage_vals < 0):
        raise ValueError("usage has negative values")

    obs_names = adata.obs_names.astype(str)
    # Align to h5ad obs order.
    usage = usage.reindex(obs_names)
    if usage.isna().any(axis=None):
        missing = usage.index[usage.isna().any(axis=1)].to_list()
        raise KeyError(f"{args.usage_tsv}: missing usage rows for {len(missing)} cells; e.g. {missing[:5]}")

    x = adata.obs[args.x_col].to_numpy(dtype=np.float64, copy=False)
    r_um = adata.obs[args.r_col].to_numpy(dtype=np.float64, copy=False)
    ap = adata.obs[args.ap_col].to_numpy(dtype=np.float64, copy=False)
    ml = adata.obs[args.ml_col].to_numpy(dtype=np.float64, copy=False)
    theta = adata.obs[args.theta_col].to_numpy(dtype=np.float64, copy=False)
    s = adata.obs[args.s_col].to_numpy(dtype=np.float64, copy=False)
    s = np.clip(s, 1.0, None)
    batch = adata.obs[args.batch_col].astype(str).to_numpy()

    keep = np.isfinite(x) & np.isfinite(r_um) & np.isfinite(theta) & np.isfinite(s)
    keep_apml = np.isfinite(ap) & np.isfinite(ml)
    if bool(args.drop_nonfinite_apml):
        keep = keep & keep_apml
    else:
        if not np.all(keep_apml):
            raise ValueError("AP/ML contain non-finite values; pass --drop-nonfinite-apml to drop them.")
        keep = keep & keep_apml

    n_keep = int(np.sum(keep))
    if n_keep <= 0:
        raise ValueError("No cells remain after filtering finite covariates.")
    if n_keep < int(adata.n_obs):
        print(f"Filtering: keeping {n_keep}/{int(adata.n_obs)} cells")

    cell_id = obs_names.to_numpy()[keep]
    x = x[keep].astype(np.float32, copy=False)
    r_um = r_um[keep].astype(np.float32, copy=False)
    ap = ap[keep].astype(np.float32, copy=False)
    ml = ml[keep].astype(np.float32, copy=False)
    theta = theta[keep].astype(np.float32, copy=False)
    s = s[keep].astype(np.float32, copy=False)
    batch = batch[keep]
    usage = usage.iloc[np.flatnonzero(keep)].copy()
    usage_arr = usage.to_numpy(dtype=np.float64, copy=False)

    if args.count_mode == "total_counts":
        pseudo = usage_arr * s.astype(np.float64, copy=False)[:, None]
    else:
        scale = float(args.scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("--scale must be finite and > 0")
        pseudo = usage_arr * scale
    pseudo = np.clip(pseudo, 0.0, None)

    if args.rounding == "round":
        counts = np.rint(pseudo)
    elif args.rounding == "floor":
        counts = np.floor(pseudo)
    elif args.rounding == "ceil":
        counts = np.ceil(pseudo)
    else:
        counts = pseudo

    if args.rounding != "none":
        counts = counts.astype(np.int32, copy=False)
    else:
        counts = counts.astype(np.float32, copy=False)

    k = int(counts.shape[1])
    if k <= 0:
        raise ValueError("No programs found in usage TSV.")
    prog_cols = [f"{args.program_prefix}{i}" for i in range(1, k + 1)]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cells.tsv").write_text("")  # fail fast if unwritable
    (out_dir / "counts.tsv").write_text("")

    cells_df = pd.DataFrame(
        {
            "cell_id": cell_id,
            "x": x,
            "r_um": r_um,
            "AP_um": ap,
            "ML_um": ml,
            "theta": theta,
            "s": s,
            "batch": batch,
        }
    )
    cells_df.to_csv(out_dir / "cells.tsv", sep="\t", index=False)

    counts_df = pd.DataFrame(counts, columns=prog_cols)
    counts_df.insert(0, "cell_id", cell_id)
    counts_df.to_csv(out_dir / "counts.tsv", sep="\t", index=False)

    meta = {
        "h5ad": str(args.h5ad),
        "usage_tsv": str(args.usage_tsv),
        "n_cells": int(cell_id.size),
        "n_programs": int(k),
        "program_cols": prog_cols,
        "pseudo_count_mode": str(args.count_mode),
        "rounding": str(args.rounding),
        "scale": None if args.count_mode != "scale_only" else float(args.scale),
        "obs_cols": {
            "x": args.x_col,
            "r_um": args.r_col,
            "AP": args.ap_col,
            "ML": args.ml_col,
            "theta": args.theta_col,
            "s": args.s_col,
            "batch": args.batch_col,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

