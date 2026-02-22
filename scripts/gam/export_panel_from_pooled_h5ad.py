from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


MIN_GENE_EXPR_FRAC = 0.001  # >0.1% cells expressing


def _as_dense_f32(x) -> np.ndarray:
    import scipy.sparse as sp

    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _filter_genes_by_expr_fraction(*, counts: np.ndarray, genes: list[str], min_frac: float) -> tuple[np.ndarray, list[str], int]:
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape={counts.shape}")
    n_cells = int(counts.shape[0])
    if n_cells <= 0:
        raise ValueError("counts must have at least one row")
    if int(counts.shape[1]) != int(len(genes)):
        raise ValueError(f"counts columns != len(genes): {counts.shape[1]} != {len(genes)}")
    if not np.isfinite(min_frac) or float(min_frac) < 0:
        raise ValueError(f"min_frac must be finite and >=0, got {min_frac}")

    nnz = np.count_nonzero(counts > 0, axis=0).astype(np.int64, copy=False)
    frac = nnz.astype(np.float64) / float(n_cells)
    keep = frac > float(min_frac)
    n_before = int(len(genes))
    n_keep = int(np.sum(keep))
    if n_keep <= 0:
        raise ValueError(f"no genes remain after expression filter: frac_expressing > {min_frac} (n_cells={n_cells})")
    if n_keep == n_before:
        return counts, genes, n_before
    counts_f = counts[:, keep]
    genes_f = [g for g, k in zip(genes, keep, strict=True) if bool(k)]
    return counts_f, genes_f, n_before


def calc_rmaxnorm_strict(t: np.ndarray, r: np.ndarray, *, n_grid: int) -> np.ndarray:
    """Compute x = r / r_max(t) using a strict binned upper-envelope interpolation."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if t.size != r.size:
        raise ValueError("t and r must have same length")
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")

    ok = np.isfinite(t) & np.isfinite(r)
    if int(np.sum(ok)) < 2:
        raise ValueError("need at least 2 finite points for rmaxnorm")

    t_ok = t[ok]
    r_ok = r[ok]
    t_grid = np.linspace(float(np.min(t_ok)), float(np.max(t_ok)), int(n_grid))
    dt = float(t_grid[1] - t_grid[0])

    edges = np.empty((t_grid.size + 1,), dtype=np.float64)
    edges[1:-1] = 0.5 * (t_grid[:-1] + t_grid[1:])
    edges[0] = t_grid[0] - 0.5 * dt
    edges[-1] = t_grid[-1] + 0.5 * dt

    bin_idx = np.digitize(t_ok, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, t_grid.size - 1)

    r_tmax = np.full((t_grid.size,), np.nan, dtype=np.float64)
    for j in range(t_grid.size):
        m = bin_idx == j
        if np.any(m):
            r_tmax[j] = float(np.max(r_ok[m]))

    good = np.isfinite(r_tmax)
    if int(np.sum(good)) < 2:
        raise ValueError("could not estimate r_max(t) from data")

    r_tmax_i = np.interp(t, t_grid[good], r_tmax[good], left=float(r_tmax[good][0]), right=float(r_tmax[good][-1]))
    return (r / np.clip(r_tmax_i, 1e-9, None)).astype(np.float32)


def _reset_out_dir(out_dir: Path) -> None:
    """Recreate output directory from scratch to avoid stale mixed artifacts."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"--out-dir exists and is not a directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _parse_list(raw: str) -> list[str]:
    out = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("expected a non-empty comma-separated list")
    return out


def _parse_overrides(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"override must be PATTERN=VALUE, got: {raw!r}")
        pat, val = s.split("=", 1)
        pat = pat.strip()
        val = val.strip()
        if not pat:
            raise ValueError(f"override pattern is empty in: {raw!r}")
        fv = float(val)
        if not np.isfinite(fv):
            raise ValueError(f"override value must be finite, got: {raw!r}")
        out[pat] = fv
    return out


def _match_override(haystack: str, overrides: dict[str, float]) -> float | None:
    if not overrides:
        return None
    hits = [(k, v) for k, v in overrides.items() if k in haystack]
    if len(hits) > 1:
        raise ValueError(f"multiple overrides matched for {haystack}: {[k for k, _ in hits]}")
    if len(hits) == 1:
        return float(hits[0][1])
    return None


def _write_cells_tsv(
    out_path: Path,
    *,
    cell_id: np.ndarray,
    x: np.ndarray,
    r_um: np.ndarray,
    ap_um: np.ndarray,
    ml_um: np.ndarray,
    theta: np.ndarray | None,
    s: np.ndarray,
    source: np.ndarray,
    batch: np.ndarray,
    log_brdu: np.ndarray | None,
    log_edu: np.ndarray | None,
    brdu_pos: np.ndarray | None,
    edu_pos: np.ndarray | None,
) -> None:
    with out_path.open("w") as f:
        extra_cols = "\tsource\tbatch"
        if brdu_pos is not None and edu_pos is not None and log_brdu is not None and log_edu is not None:
            extra_cols += "\tlog_brdu_mean\tlog_edu_mean\tbrdu_pos\tedu_pos"
        if theta is None:
            f.write("cell_id\tx\tr_um\tAP_um\tML_um\ts" + extra_cols + "\n")
        else:
            f.write("cell_id\tx\tr_um\tAP_um\tML_um\ttheta\ts" + extra_cols + "\n")

        if theta is None and "log_brdu_mean" in extra_cols:
            assert log_brdu is not None and log_edu is not None and brdu_pos is not None and edu_pos is not None
            for cid, xx, r0, ap0, ml0, sf, src, b, lb, le, bp, ep in zip(
                cell_id,
                x,
                r_um,
                ap_um,
                ml_um,
                s,
                source,
                batch,
                log_brdu,
                log_edu,
                brdu_pos,
                edu_pos,
                strict=True,
            ):
                f.write(f"{cid}\t{float(xx)}\t{float(r0)}\t{float(ap0)}\t{float(ml0)}\t{float(sf)}\t{src}\t{b}\t{float(lb)}\t{float(le)}\t{int(bp)}\t{int(ep)}\n")
        elif theta is None:
            for cid, xx, r0, ap0, ml0, sf, src, b in zip(
                cell_id, x, r_um, ap_um, ml_um, s, source, batch, strict=True
            ):
                f.write(f"{cid}\t{float(xx)}\t{float(r0)}\t{float(ap0)}\t{float(ml0)}\t{float(sf)}\t{src}\t{b}\n")
        elif "log_brdu_mean" in extra_cols:
            assert log_brdu is not None and log_edu is not None and brdu_pos is not None and edu_pos is not None
            for cid, xx, r0, ap0, ml0, th, sf, src, b, lb, le, bp, ep in zip(
                cell_id,
                x,
                r_um,
                ap_um,
                ml_um,
                theta,
                s,
                source,
                batch,
                log_brdu,
                log_edu,
                brdu_pos,
                edu_pos,
                strict=True,
            ):
                f.write(
                    f"{cid}\t{float(xx)}\t{float(r0)}\t{float(ap0)}\t{float(ml0)}\t{float(th)}\t{float(sf)}\t{src}\t{b}\t{float(lb)}\t{float(le)}\t{int(bp)}\t{int(ep)}\n"
                )
        else:
            for cid, xx, r0, ap0, ml0, th, sf, src, b in zip(
                cell_id, x, r_um, ap_um, ml_um, theta, s, source, batch, strict=True
            ):
                f.write(f"{cid}\t{float(xx)}\t{float(r0)}\t{float(ap0)}\t{float(ml0)}\t{float(th)}\t{float(sf)}\t{src}\t{b}\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Export a GAM panel from one pooled h5ad. "
            "Intended for files like ~/nvme/wip.h5ad where rows already include dataset/roi metadata."
        )
    )
    p.add_argument("h5ad", type=Path, help="Single pooled h5ad (e.g. /home/chaichontat/nvme/wip.h5ad).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--region-col", type=str, default="ccf_adjusted")
    p.add_argument(
        "--region",
        type=str,
        default="cortex,cortex2",
        help="Comma-separated allowed values in --region-col.",
    )
    p.add_argument("--dataset-col", type=str, default="dataset")
    p.add_argument("--roi-col", type=str, default="roi")
    p.add_argument("--theta-col", type=str, default="tricycle")
    p.add_argument(
        "--subset-col",
        type=str,
        default=None,
        help="Optional obs column name for subsetting rows before region+tr filtering (e.g. leiden).",
    )
    p.add_argument(
        "--subset-values",
        type=str,
        default=None,
        help="Comma-separated allowed values in --subset-col (compared as strings).",
    )
    p.add_argument(
        "--theta-obsm",
        type=str,
        default=None,
        help=(
            "If set, compute theta (radians) from obsm[KEY] via atan2(y, x). "
            "KEY must refer to an (n_cells, 2) array. Overrides --theta-col."
        ),
    )
    p.add_argument(
        "--no-theta",
        action="store_true",
        help="Do not require or export theta values. Use when downstream fit is run with --no-theta.",
    )
    p.add_argument("--t-col", type=str, default="t_local")
    p.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Optional: filter to t >= t-min (t is from --t-col). If omitted, no t-min filtering is applied.",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional: filter to t <= t-max (t is from --t-col). If omitted, no t-max filtering is applied.",
    )
    p.add_argument(
        "--drop-nans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop rows with non-finite t/r/theta/AP_ML values after region filtering. "
            "Enabled by default; pass --no-drop-nans to fail instead."
        ),
    )
    p.add_argument(
        "--t-min-override",
        action="append",
        default=[],
        help="Override t-min for matching sources: PATTERN=VALUE (pattern matched against dataset/roi source label).",
    )
    p.add_argument(
        "--t-max-override",
        action="append",
        default=[],
        help="Override t-max for matching sources: PATTERN=VALUE (pattern matched against dataset/roi source label).",
    )
    p.add_argument(
        "--r-min",
        type=float,
        default=None,
        help=(
            "Optional: filter to x > r-min where x = r / r_max(t). "
            "If omitted, no x filtering is applied."
        ),
    )
    p.add_argument(
        "--flip-x-for",
        action="append",
        default=[],
        help=(
            "Apply x := 1 - x for matching source labels (or dataset labels). "
            "Repeatable; each value is treated as a substring pattern."
        ),
    )
    p.add_argument(
        "--genes",
        type=str,
        default="all",
        help="Comma-separated gene list for counts.tsv, or 'all' to export all genes in var_names.",
    )
    p.add_argument(
        "--counts-layer",
        type=str,
        default="raw",
        help="Layer to read counts from (default: raw). Use 'X' to read from adata.X.",
    )
    p.add_argument("--rmaxnorm-grid", type=int, default=240)
    p.add_argument("--log-brdu-threshold", type=float, default=None, help="If set with --log-edu-threshold, add brdu_pos.")
    p.add_argument("--log-edu-threshold", type=float, default=None, help="If set with --log-brdu-threshold, add edu_pos.")
    args = p.parse_args()

    import anndata as ad
    import pandas as pd

    if (args.log_brdu_threshold is None) != (args.log_edu_threshold is None):
        raise ValueError("Provide both --log-brdu-threshold and --log-edu-threshold, or neither.")
    if (args.subset_col is None) != (args.subset_values is None):
        raise ValueError("Provide both --subset-col and --subset-values, or neither.")

    out_dir: Path = args.out_dir
    _reset_out_dir(out_dir)

    h5ad = Path(args.h5ad).expanduser()
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)

    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        required_obs_cols = [args.region_col, args.dataset_col, args.roi_col, args.t_col, "r_um"]
        if (not args.no_theta) and args.theta_obsm is None:
            required_obs_cols.append(args.theta_col)

        for col in required_obs_cols:
            if col not in adata.obs.columns:
                raise KeyError(f"{h5ad}: obs.{col} not found")
        if "AP_ML_um" not in adata.obsm:
            raise KeyError(f"{h5ad}: obsm['AP_ML_um'] not found")
        if args.theta_obsm is not None and args.theta_obsm not in adata.obsm:
            raise KeyError(f"{h5ad}: obsm['{args.theta_obsm}'] not found")
        if args.no_theta and args.theta_obsm is not None:
            raise ValueError("--no-theta cannot be combined with --theta-obsm")

        region_vals = adata.obs[args.region_col].astype(str).to_numpy()
        regions_keep = set(_parse_list(args.region))
        base_mask = np.isin(region_vals, list(regions_keep))
        if args.subset_col is not None:
            if args.subset_col not in adata.obs.columns:
                raise KeyError(f"{h5ad}: obs.{args.subset_col} not found")
            subset_vals = adata.obs[args.subset_col].astype(str).to_numpy()
            subset_keep = set(_parse_list(args.subset_values))
            base_mask = base_mask & np.isin(subset_vals, list(subset_keep))
        idx_region = np.where(base_mask)[0]
        if idx_region.size == 0:
            msg = f"{h5ad}: no cells found for obs.{args.region_col} in {sorted(regions_keep)}"
            if args.subset_col is not None:
                msg += f" after subsetting obs.{args.subset_col} in {sorted(subset_keep)}"
            raise ValueError(msg)

        t_region = adata.obs.iloc[idx_region][args.t_col].to_numpy(dtype=np.float64, copy=False)
        r_region = adata.obs.iloc[idx_region]["r_um"].to_numpy(dtype=np.float64, copy=False)
        apml_region = np.asarray(adata.obsm["AP_ML_um"][idx_region, :], dtype=np.float64)
        if apml_region.ndim != 2 or apml_region.shape != (idx_region.size, 2):
            raise ValueError(f"{h5ad}: unexpected AP_ML_um shape={apml_region.shape}")
        theta_region: np.ndarray | None
        if args.no_theta:
            theta_region = None
        elif args.theta_obsm is None:
            theta_region = adata.obs.iloc[idx_region][args.theta_col].to_numpy(dtype=np.float64, copy=False)
        else:
            th_xy = np.asarray(adata.obsm[args.theta_obsm][idx_region, :], dtype=np.float64)
            if th_xy.ndim != 2 or th_xy.shape != (idx_region.size, 2):
                raise ValueError(f"{h5ad}: unexpected obsm['{args.theta_obsm}'] shape={th_xy.shape}; expected (n,2)")
            theta_region = np.arctan2(th_xy[:, 1], th_xy[:, 0]).astype(np.float64, copy=False)
        dataset_region = adata.obs.iloc[idx_region][args.dataset_col].astype(str).to_numpy()
        roi_region = adata.obs.iloc[idx_region][args.roi_col].astype(str).to_numpy()

        finite_mask = np.isfinite(t_region) & np.isfinite(r_region) & np.isfinite(apml_region[:, 0]) & np.isfinite(apml_region[:, 1])
        if theta_region is not None:
            finite_mask = finite_mask & np.isfinite(theta_region)
        n_rows_dropped_nonfinite = int(np.sum(~finite_mask))
        if args.drop_nans:
            idx_region = idx_region[finite_mask]
            t_region = t_region[finite_mask]
            r_region = r_region[finite_mask]
            apml_region = apml_region[finite_mask, :]
            if theta_region is not None:
                theta_region = theta_region[finite_mask]
            dataset_region = dataset_region[finite_mask]
            roi_region = roi_region[finite_mask]
            if idx_region.size == 0:
                raise ValueError(f"{h5ad}: no cells remain after dropping non-finite rows")
        else:
            if not np.all(np.isfinite(t_region)):
                raise ValueError(f"{h5ad}: obs.{args.t_col} must be finite for selected rows")
            if not np.all(np.isfinite(r_region)):
                raise ValueError(f"{h5ad}: obs.r_um must be finite for selected rows")
            if theta_region is not None and not np.all(np.isfinite(theta_region)):
                raise ValueError(f"{h5ad}: obs.{args.theta_col} must be finite for selected rows")
            if not np.all(np.isfinite(apml_region)):
                raise ValueError(f"{h5ad}: obsm['AP_ML_um'] must be finite for selected rows")

        source_region = np.array([f"{d}.{r}" for d, r in zip(dataset_region, roi_region, strict=True)], dtype=object)

        t_min_default: float | None = None if args.t_min is None else float(args.t_min)
        t_max_default: float | None = None if args.t_max is None else float(args.t_max)
        r_min: float | None = None if args.r_min is None else float(args.r_min)
        t_min_overrides = _parse_overrides(list(args.t_min_override))
        t_max_overrides = _parse_overrides(list(args.t_max_override))
        flip_patterns = list(args.flip_x_for)

        x_region = np.full((idx_region.size,), np.nan, dtype=np.float32)
        keep_region = np.zeros((idx_region.size,), dtype=bool)
        per_source_filter: dict[str, dict[str, float | bool | int]] = {}

        for src in np.unique(source_region):
            src_mask = source_region == src
            t_src = t_region[src_mask]
            r_src = r_region[src_mask]
            x_src = calc_rmaxnorm_strict(t=t_src, r=r_src, n_grid=int(args.rmaxnorm_grid))
            dataset_src = str(dataset_region[np.where(src_mask)[0][0]])

            hay = f"{h5ad} {src} {dataset_src}"
            t_min_src = _match_override(hay, t_min_overrides)
            t_max_src = _match_override(hay, t_max_overrides)
            if t_min_src is None:
                t_min_src = t_min_default
            if t_max_src is None:
                t_max_src = t_max_default
            if (t_min_src is not None) and (t_max_src is not None) and (float(t_min_src) > float(t_max_src)):
                raise ValueError(f"{h5ad}: for source={src}, t-min ({t_min_src}) must be <= t-max ({t_max_src})")

            flip_x = any((pat in src) or (pat in dataset_src) for pat in flip_patterns)
            if flip_x:
                x_src = (1.0 - x_src).astype(np.float32, copy=False)

            keep_src = np.isfinite(t_src) & np.isfinite(x_src)
            if t_min_src is not None:
                keep_src = keep_src & (t_src >= float(t_min_src))
            if t_max_src is not None:
                keep_src = keep_src & (t_src <= float(t_max_src))
            if r_min is not None:
                keep_src = keep_src & (x_src > float(r_min))
            x_region[src_mask] = x_src
            keep_region[src_mask] = keep_src

            per_source_filter[str(src)] = {
                "n_source_rows": int(np.sum(src_mask)),
                "n_kept_rows": int(np.sum(keep_src)),
                "t_min": None if t_min_src is None else float(t_min_src),
                "t_max": None if t_max_src is None else float(t_max_src),
                "x_min": None if r_min is None else float(r_min),
                "x_flip_applied": bool(flip_x),
            }

        if not np.any(keep_region):
            raise ValueError(f"{h5ad}: no cells after region+tr filtering")

        idx = idx_region[keep_region]
        x = x_region[keep_region].astype(np.float32, copy=False)
        r_um = r_region[keep_region].astype(np.float32, copy=False)
        ap_um = apml_region[keep_region, 0].astype(np.float32, copy=False)
        ml_um = apml_region[keep_region, 1].astype(np.float32, copy=False)
        theta = None if theta_region is None else theta_region[keep_region].astype(np.float32, copy=False)
        batch = dataset_region[keep_region].astype(object, copy=False)
        source = source_region[keep_region].astype(object, copy=False)

        genes_arg = str(args.genes).strip()
        if genes_arg.lower() in {"all", "*"}:
            genes = list(map(str, adata.var_names))
        else:
            genes = [g.strip() for g in genes_arg.split(",") if g.strip()]
            if not genes:
                raise ValueError("no genes provided")
            var_set = set(map(str, adata.var_names))
            missing = [g for g in genes if g not in var_set]
            if missing:
                raise KeyError(f"genes not found in var_names: {missing}")

        counts_view = adata[idx, genes]
        if args.counts_layer == "X":
            counts = _as_dense_f32(counts_view.X)
        else:
            if args.counts_layer not in adata.layers:
                raise KeyError(f"{h5ad}: layers['{args.counts_layer}'] not found")
            counts = _as_dense_f32(counts_view.layers[args.counts_layer])
        if float(np.nanmin(counts)) < 0:
            raise ValueError("counts matrix has negative values")

        counts, genes, n_genes_before_expr = _filter_genes_by_expr_fraction(
            counts=counts, genes=genes, min_frac=MIN_GENE_EXPR_FRAC
        )
        if len(genes) != n_genes_before_expr:
            print(
                f"Gene expression filter: kept {len(genes)}/{n_genes_before_expr} genes with frac_expressing > {MIN_GENE_EXPR_FRAC:g}",
                flush=True,
            )

        s_source = "sum(selected genes)"
        if "total_counts" in adata.obs.columns:
            s = adata.obs.iloc[idx]["total_counts"].to_numpy(dtype=np.float64, copy=False)
            if not np.all(np.isfinite(s)):
                raise ValueError("obs.total_counts is not finite for all selected cells")
            s_source = "obs.total_counts (clipped to >=1)"
        else:
            s = np.sum(counts, axis=1).astype(np.float64)
        s = np.clip(s, 1.0, None)

        obs_name = adata.obs_names.astype(str).to_numpy()[idx]
        row_idx = idx.astype(np.int64, copy=False)
        cell_id = np.array(
            [f"{d}|{r}|{c}|{i}" for d, r, c, i in zip(batch, source, obs_name, row_idx, strict=True)],
            dtype=object,
        )
        if int(np.unique(cell_id).size) != int(cell_id.size):
            raise ValueError("constructed cell_id is not unique; check dataset/roi/source key choices")

        log_brdu = None
        log_edu = None
        brdu_pos = None
        edu_pos = None
        if args.log_brdu_threshold is not None and args.log_edu_threshold is not None:
            if "log_brdu_mean" in adata.obs.columns and "log_edu_mean" in adata.obs.columns:
                log_brdu = adata.obs.iloc[idx]["log_brdu_mean"].to_numpy(dtype=np.float64, copy=False).astype(np.float32)
                log_edu = adata.obs.iloc[idx]["log_edu_mean"].to_numpy(dtype=np.float64, copy=False).astype(np.float32)
            else:
                if "brdu_mean" not in adata.obs.columns or "edu_mean" not in adata.obs.columns:
                    raise KeyError("Need obs.log_brdu_mean/log_edu_mean or obs.brdu_mean/edu_mean for positivity.")
                brdu = adata.obs.iloc[idx]["brdu_mean"].to_numpy(dtype=np.float64, copy=False)
                edu = adata.obs.iloc[idx]["edu_mean"].to_numpy(dtype=np.float64, copy=False)
                if np.any(brdu < 0) or np.any(edu < 0):
                    raise ValueError("brdu_mean/edu_mean has negative values; cannot log1p safely.")
                log_brdu = np.log1p(brdu).astype(np.float32)
                log_edu = np.log1p(edu).astype(np.float32)

            if not np.all(np.isfinite(log_brdu)) or not np.all(np.isfinite(log_edu)):
                raise ValueError("log_brdu/log_edu contains non-finite values")
            brdu_pos = (log_brdu >= float(args.log_brdu_threshold)).astype(np.int8)
            edu_pos = (log_edu >= float(args.log_edu_threshold)).astype(np.int8)

        cells_path = out_dir / "cells.tsv"
        counts_path = out_dir / "counts.tsv"
        meta_path = out_dir / "panel_meta.json"

        _write_cells_tsv(
            cells_path,
            cell_id=cell_id,
            x=x,
            r_um=r_um,
            ap_um=ap_um,
            ml_um=ml_um,
            theta=theta,
            s=s,
            source=source,
            batch=batch,
            log_brdu=log_brdu,
            log_edu=log_edu,
            brdu_pos=brdu_pos,
            edu_pos=edu_pos,
        )

        frac = counts.astype(np.float64) - np.rint(counts.astype(np.float64))
        max_abs_frac = float(np.max(np.abs(frac)))
        if max_abs_frac <= 1e-6 and float(np.nanmax(counts)) <= float(np.iinfo(np.int16).max):
            counts_out = np.rint(counts).astype(np.int16, copy=False)
            counts_dtype = "int16"
        else:
            counts_out = counts.astype(np.float32, copy=False)
            counts_dtype = "float32"

        df = pd.DataFrame(counts_out, columns=genes)
        df.insert(0, "cell_id", cell_id)
        df.to_csv(counts_path, sep="\t", index=False)

        meta = {
            "h5ad": str(h5ad),
            "region_col": args.region_col,
            "region_allowed": sorted(regions_keep),
            "dataset_col": args.dataset_col,
            "roi_col": args.roi_col,
            "subset_col": args.subset_col,
            "subset_allowed": None if args.subset_values is None else _parse_list(args.subset_values),
            "theta_col": None if args.no_theta else args.theta_col,
            "theta_exported": not args.no_theta,
            "t_col": args.t_col,
            "counts_layer": args.counts_layer,
            "n_cells_region": int(idx_region.size),
            "n_cells": int(cell_id.size),
            "n_unique_cell_id": int(np.unique(cell_id).size),
            "drop_nans": bool(args.drop_nans),
            "n_rows_dropped_nonfinite": n_rows_dropped_nonfinite,
            "tr_filter": {"t_min_default": t_min_default, "t_max_default": t_max_default, "x_min": r_min},
            "tr_filter_overrides": {"t_min": t_min_overrides, "t_max": t_max_overrides},
            "per_source_filter": per_source_filter,
            "genes": genes,
            "gene_expr_min_frac": float(MIN_GENE_EXPR_FRAC),
            "n_genes_before_expr_filter": int(n_genes_before_expr),
            "counts_dtype": counts_dtype,
            "max_abs_frac_from_integer": max_abs_frac,
            "x_note": "x is r / r_max(t) (strict binned upper-envelope interpolation), computed per source",
            "r_um_note": "r_um is obs['r_um'] (um) after region + optional filtering",
            "ap_ml_note": "AP_um/ML_um are obsm['AP_ML_um'] (um) after region + optional filtering",
            "source_col": "cells.tsv.source (dataset.roi)",
            "batch_col": "cells.tsv.batch (dataset)",
            "rmaxnorm_grid": int(args.rmaxnorm_grid),
            "size_factor_source": s_source,
            "positivity_thresholds": (
                None
                if args.log_brdu_threshold is None
                else {"log_brdu_mean": float(args.log_brdu_threshold), "log_edu_mean": float(args.log_edu_threshold)}
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        print(f"Wrote {cells_path}")
        print(f"Wrote {counts_path}")
        print(f"Wrote {meta_path}")
        return 0
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


if __name__ == "__main__":
    raise SystemExit(main())
