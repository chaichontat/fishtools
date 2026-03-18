# ruff: noqa: E402
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

for key, value in {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ[key] = value

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")

H5AD_PATH = Path("~/nvme/vz.h5ad").expanduser()
REPO_ROOT = Path("/home/chaichontat/fishtools2")
FIT_SCRIPT = REPO_ROOT / "scripts/gam/fit_pseudotime_panel.R"
PREDICT_SCRIPT = REPO_ROOT / "scripts/gam/predict_reference_grid_from_rds.R"
OUTDIR = Path("/home/chaichontat/fishtools2/scripts/_out/vz_usage7_over_usage1plus7_k12_excl_u3_u4_u6_reml_results")
WORKDIR = Path("/home/chaichontat/fishtools2/scripts/_out/vz_usage7_over_usage1plus7_k12_excl_u3_u4_u6_reml_fits")

LAYER = "raw_sct_corrected"
USAGE1 = "Usage_1"
USAGE3 = "Usage_3"
USAGE4 = "Usage_4"
USAGE6 = "Usage_6"
USAGE7 = "Usage_7"
MIN_USAGE = 0.2
SPLINE_K = 12
FDR_CUT = 1e-2
AMPLITUDE_CUT = 0.2
ORD_THRESH = 0.7
GRID_SIZE = 400
HISTOGRAM_BINS = 100
MAX_WORKERS = 30


@dataclass(frozen=True)
class GeneTask:
    gene_idx: int
    gene: str
    panel_dir: Path
    out_tsv: Path
    fit_path: Path


def _to_dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _extract_animal(dataset: str) -> str:
    match = re.search(r"JaxA\d+", dataset)
    if match is None:
        raise ValueError(f"Could not parse animal from dataset={dataset!r}")
    return match.group(0)


def _prepare_subset(adata: ad.AnnData) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    obs = adata.obs.copy()
    u1 = pd.to_numeric(obs[USAGE1], errors="coerce").to_numpy(dtype=np.float32)
    u3 = pd.to_numeric(obs[USAGE3], errors="coerce").to_numpy(dtype=np.float32)
    u4 = pd.to_numeric(obs[USAGE4], errors="coerce").to_numpy(dtype=np.float32)
    u6 = pd.to_numeric(obs[USAGE6], errors="coerce").to_numpy(dtype=np.float32)
    u7 = pd.to_numeric(obs[USAGE7], errors="coerce").to_numpy(dtype=np.float32)
    keep = (
        np.isfinite(u1)
        & np.isfinite(u3)
        & np.isfinite(u4)
        & np.isfinite(u6)
        & np.isfinite(u7)
        & ((u1 >= MIN_USAGE) | (u7 >= MIN_USAGE))
        & (u3 <= MIN_USAGE)
        & (u4 <= MIN_USAGE)
        & (u6 <= MIN_USAGE)
    )
    if not np.any(keep):
        raise ValueError("No cells passed the Usage_1/Usage_7 inclusion and Usage_3/4/6 exclusion thresholds.")

    idx = np.flatnonzero(keep)
    denom = u1[keep] + u7[keep]
    valid = denom > 0
    if not np.any(valid):
        raise ValueError("No selected cells had positive Usage_1 + Usage_7.")

    idx = idx[valid]
    pseudotime = (u7[keep][valid] / denom[valid]).astype(np.float32)
    order = np.argsort(pseudotime, kind="mergesort")
    idx = idx[order]
    pseudotime = pseudotime[order]
    obs_sub = obs.iloc[idx].copy()

    cells = pd.DataFrame(
        {
            "cell_id": obs_sub.index.astype(str),
            "s": np.ones(obs_sub.shape[0], dtype=np.float32),
            "batch": obs_sub["roi"].astype(str).to_numpy(),
            "animal": obs_sub["dataset"].astype(str).map(_extract_animal).to_numpy(),
            "t": pseudotime,
            "seg": np.repeat("1", obs_sub.shape[0]),
        }
    )
    return cells, idx, pseudotime


def _write_gene_tasks(cells: pd.DataFrame, x: np.ndarray, genes: np.ndarray) -> list[GeneTask]:
    WORKDIR.mkdir(parents=True, exist_ok=True)

    cells_path = WORKDIR / "cells.tsv"
    if cells_path.exists():
        existing_cells = pd.read_csv(cells_path, sep="\t")
        same_columns = existing_cells.columns.tolist() == cells.columns.tolist()
        same_ids = same_columns and existing_cells["cell_id"].astype(str).equals(cells["cell_id"].astype(str))
        same_t = same_columns and np.allclose(
            existing_cells["t"].to_numpy(dtype=np.float64),
            cells["t"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )
        same_s = same_columns and np.allclose(
            existing_cells["s"].to_numpy(dtype=np.float64),
            cells["s"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        )
        same_batch = same_columns and existing_cells["batch"].astype(str).equals(cells["batch"].astype(str))
        same_animal = same_columns and existing_cells["animal"].astype(str).equals(cells["animal"].astype(str))
        same_seg = same_columns and existing_cells["seg"].astype(str).equals(cells["seg"].astype(str))
        if not (same_columns and same_ids and same_t and same_s and same_batch and same_animal and same_seg):
            raise ValueError(f"Existing {cells_path} does not match the current cell subset.")
    else:
        cells.to_csv(cells_path, sep="\t", index=False)

    tasks: list[GeneTask] = []
    for gene_idx, gene in enumerate(genes.tolist()):
        panel_dir = WORKDIR / f"gene_{gene_idx:04d}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel_cells = panel_dir / "cells.tsv"
        if not panel_cells.exists():
            os.symlink(cells_path, panel_cells)

        counts_path = panel_dir / "counts.tsv"
        if counts_path.exists():
            counts = pd.read_csv(counts_path, sep="\t", nrows=5)
            expected_cols = ["cell_id", gene]
            if counts.columns.tolist() != expected_cols:
                raise ValueError(f"Existing {counts_path} has columns {counts.columns.tolist()}, expected {expected_cols}.")
        else:
            counts = pd.DataFrame(x[:, [gene_idx]], columns=[gene])
            counts.insert(0, "cell_id", cells["cell_id"].to_numpy())
            counts.to_csv(counts_path, sep="\t", index=False)
        task = GeneTask(
            gene_idx=gene_idx,
            gene=gene,
            panel_dir=panel_dir,
            out_tsv=panel_dir / "pseudotime_fit_results.tsv",
            fit_path=panel_dir / "fits_rds__pseudotime_fit_results" / f"0001_{re.sub(r'[^A-Za-z0-9._-]+', '_', gene)}.gam.rds",
        )
        tasks.append(task)
        if task.out_tsv.exists() and task.out_tsv.with_suffix(".fitted.tsv").exists():
            print(f"found completed gene {gene_idx + 1}/{len(genes)}: {gene}", flush=True)
        else:
            print(f"prepared gene {gene_idx + 1}/{len(genes)}: {gene}", flush=True)
    return tasks


def _run_gene(task: GeneTask) -> pd.DataFrame:
    fitted_path = task.out_tsv.with_suffix(".fitted.tsv")
    if task.out_tsv.exists() and fitted_path.exists():
        print(f"reusing gene {task.gene_idx + 1}: {task.gene}", flush=True)
        stats_df = pd.read_csv(task.out_tsv, sep="\t")
        if stats_df.shape[0] != 1 or stats_df.loc[0, "gene"] != task.gene:
            raise ValueError(f"Unexpected stats rows for cached {task.gene}: {stats_df['gene'].tolist()}")
        return stats_df

    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_THREAD_LIMIT"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MKL_DYNAMIC"] = "FALSE"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    cmd = [
        shutil.which("Rscript") or "Rscript",
        str(FIT_SCRIPT),
        str(task.panel_dir),
        str(task.out_tsv),
        "--threads",
        "1",
        "--bam-threads",
        "1",
        "--k-pseudotime",
        str(SPLINE_K),
        "--a-cut",
        str(AMPLITUDE_CUT),
        "--fdr-cut",
        str(FDR_CUT),
        "--heartbeat-sec",
        "30",
        "--no-diagnostics",
    ]
    print(f"launching gene {task.gene_idx + 1}: {task.gene}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Gene {task.gene} failed with code {result.returncode}")

    stats_df = pd.read_csv(task.out_tsv, sep="\t")
    if stats_df.shape[0] != 1 or stats_df.loc[0, "gene"] != task.gene:
        raise ValueError(f"Unexpected stats rows for {task.gene}: {stats_df['gene'].tolist()}")
    print(f"finished gene {task.gene_idx + 1}: {task.gene}", flush=True)
    return stats_df


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    row_min = x.min(axis=1, keepdims=True)
    row_max = x.max(axis=1, keepdims=True)
    span = row_max - row_min
    span[span == 0] = 1.0
    return (x - row_min) / span


def _peak_time_order(norm_fitted: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = norm_fitted > ORD_THRESH
    peak_time = np.empty(norm_fitted.shape[0], dtype=np.float32)
    for i in range(norm_fitted.shape[0]):
        if np.any(mask[i]):
            peak_time[i] = float(grid[mask[i]].mean())
        else:
            peak_time[i] = float(grid[np.argmax(norm_fitted[i])])
    return peak_time


def _predict_reference_grid(signi_genes: list[str], task_by_gene: dict[str, GeneTask], pseudotime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(float(pseudotime.min()), float(pseudotime.max()), num=GRID_SIZE, dtype=np.float32)
    manifest_path = OUTDIR / "_reference_prediction_manifest.tsv"
    grid_path = OUTDIR / "_reference_prediction_grid.tsv"
    out_path = OUTDIR / "_reference_prediction_matrix.tsv"

    pd.DataFrame(
        {
            "gene": signi_genes,
            "fit_path": [str(task_by_gene[gene].fit_path) for gene in signi_genes],
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    pd.DataFrame({"t": grid}).to_csv(grid_path, sep="\t", index=False)

    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_THREAD_LIMIT"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MKL_DYNAMIC"] = "FALSE"
    env["BLIS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    cmd = [
        shutil.which("Rscript") or "Rscript",
        str(PREDICT_SCRIPT),
        str(manifest_path),
        str(grid_path),
        str(out_path),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Reference-grid prediction failed with code {result.returncode}")

    pred_df = pd.read_csv(out_path, sep="\t")
    if pred_df["gene"].astype(str).tolist() != signi_genes:
        raise ValueError("Reference-grid prediction gene order mismatch.")

    matrix = pred_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    center_rank = np.searchsorted(pseudotime, grid, side="left")
    center_rank = np.clip(center_rank, 0, pseudotime.size - 1)
    return matrix, grid, center_rank


def _count_cells_on_grid(pseudotime: np.ndarray, grid: np.ndarray) -> np.ndarray:
    edges = np.empty(grid.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    edges[0] = float(pseudotime.min())
    edges[-1] = float(pseudotime.max())
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    counts, _ = np.histogram(pseudotime, bins=edges)
    return counts.astype(np.int32)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(H5AD_PATH)
    if LAYER not in adata.layers:
        raise ValueError(f"Expected layer {LAYER!r}; found {list(adata.layers.keys())}")

    cells, idx, pseudotime = _prepare_subset(adata)
    x = _to_dense(adata.layers[LAYER][idx, :]).astype(np.float32)
    genes = adata.var_names.to_numpy(dtype=str)

    finite_gene_mask = np.isfinite(x).all(axis=0)
    variable_gene_mask = np.nanstd(x, axis=0) > 0
    gene_mask = finite_gene_mask & variable_gene_mask
    x = x[:, gene_mask]
    genes = genes[gene_mask]

    tasks = _write_gene_tasks(cells, x, genes)
    task_by_gene = {task.gene: task for task in tasks}
    print(
        f"running {len(tasks)} gene fits with max_workers={MAX_WORKERS}, spline_k={SPLINE_K}, grid_size={GRID_SIZE}",
        flush=True,
    )
    stats_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_gene, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            print(f"completed gene {task.gene_idx + 1}/{len(tasks)}: {task.gene}", flush=True)
            stats_df = future.result()
            stats_parts.append(stats_df)

    stats_df = pd.concat(stats_parts, ignore_index=True)
    stats_df = stats_df.drop_duplicates(subset=["gene"]).set_index("gene").reindex(genes).reset_index()

    p_ok = np.isfinite(stats_df["p_val"].to_numpy(dtype=float))
    fdr = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    if p_ok.any():
        fdr[p_ok] = multipletests(stats_df.loc[p_ok, "p_val"].to_numpy(dtype=float), method="fdr_bh")[1]
    stats_df["fdr"] = fdr
    stats_df["st"] = np.where(
        np.isfinite(stats_df["fdr"]) & np.isfinite(stats_df["A"]) & (stats_df["fdr"] < FDR_CUT) & (stats_df["A"] > AMPLITUDE_CUT),
        1.0,
        0.0,
    )
    stats_df["signi"] = stats_df["st"] > 0.8

    signi_genes = stats_df.loc[stats_df["signi"], "gene"].astype(str).tolist()
    if not signi_genes:
        raise ValueError("No genes passed the mgcv significance + amplitude filters.")

    predicted_grid, bin_pt, center_rank = _predict_reference_grid(signi_genes, task_by_gene, pseudotime)
    hist_pt = np.linspace(float(pseudotime.min()), float(pseudotime.max()), num=HISTOGRAM_BINS, dtype=np.float32)
    cell_counts = _count_cells_on_grid(pseudotime, hist_pt)
    norm_binned = _normalize_rows(predicted_grid)
    peak_time = _peak_time_order(norm_binned, bin_pt)
    row_order = np.argsort(peak_time, kind="mergesort")

    genes_ordered = np.asarray(signi_genes, dtype=str)[row_order]
    heatmap_df = pd.DataFrame(
        norm_binned[row_order],
        index=genes_ordered,
        columns=[f"{value:.4f}" for value in bin_pt],
    )

    peak_map = {gene: float(pt) for gene, pt in zip(signi_genes, peak_time, strict=True)}
    rank_map = {gene: int(rank) for rank, gene in enumerate(genes_ordered)}
    stats_df["peak_pseudotime"] = stats_df["gene"].map(peak_map)
    stats_df["heatmap_rank"] = stats_df["gene"].map(rank_map)
    stats_df = stats_df.sort_values(["signi", "heatmap_rank", "A"], ascending=[False, True, False])

    sns.set_theme(style="white")
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.8, 20, 2.2, 1.0],
        height_ratios=[0.8, 15],
        wspace=0.02,
        hspace=0.02,
    )
    ax_blank_hist = fig.add_subplot(gs[0, 0])
    ax_blank_hist.axis("off")
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_blank_hist_labels = fig.add_subplot(gs[0, 2])
    ax_blank_hist_labels.axis("off")
    ax_blank_hist_cbar = fig.add_subplot(gs[0, 3])
    ax_blank_hist_cbar.axis("off")
    ax_dend = fig.add_subplot(gs[1, 0])
    ax_heatmap = fig.add_subplot(gs[1, 1])
    ax_blank_labels = fig.add_subplot(gs[1, 2])
    ax_blank_labels.axis("off")
    ax_cbar = fig.add_subplot(gs[1, 3])
    hist_edges = np.linspace(0, bin_pt.size, num=HISTOGRAM_BINS + 1, dtype=np.float32)
    ax_hist.bar(
        hist_edges[:-1],
        cell_counts,
        width=np.diff(hist_edges),
        align="edge",
        color="#4C72B0",
        linewidth=0,
    )
    ax_hist.set_xlim(0, bin_pt.size)
    ax_hist.set_xticks([])
    ax_hist.set_ylabel("cells", fontsize=8)
    ax_hist.set_title("Cell density along pseudotime", fontsize=10, loc="left")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    sns.heatmap(
        heatmap_df,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        cbar_ax=ax_cbar,
        cbar_kws={"label": "normalized fitted expression"},
        xticklabels=False,
        yticklabels=True,
        rasterized=True,
        ax=ax_heatmap,
    )
    ax_dend.axis("off")
    ax_hist.set_xlim(ax_heatmap.get_xlim())
    ax_heatmap.set_xlabel("Usage_7 / (Usage_1 + Usage_7) pseudotime")
    ax_heatmap.set_ylabel("")
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.yaxis.set_label_position("right")
    ax_heatmap.tick_params(axis="y", labelright=True, labelleft=False, labelsize=12, pad=2, length=0)
    for label in ax_heatmap.get_yticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("left")
        label.set_verticalalignment("center")
    ax_heatmap.set_title(
        "mgcv::bam genes with pseudotime structure, ordered by peak pseudotime",
        pad=18,
    )
    fig_path = OUTDIR / "all_genes_clustermap_rolling_average.png"
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"gene": genes_ordered}).to_csv(OUTDIR / "gene_order.csv", index=False)
    heatmap_df.to_csv(OUTDIR / "all_gene_rolling_zscores.csv.gz", compression="gzip")
    pd.DataFrame(
        {
            "window_index": np.arange(GRID_SIZE, dtype=int),
            "center_cell_rank": center_rank,
            "usage7_over_usage1plus7": bin_pt,
        }
    ).to_csv(OUTDIR / "rolling_windows.csv", index=False)
    stats_df.to_csv(OUTDIR / "gene_structure_stats.csv", index=False)

    summary = {
        "h5ad_path": str(H5AD_PATH),
        "layer": LAYER,
        "backend": "mgcv_bam_threadpool_subprocess_reml",
        "family": "nb",
        "fit_method": "REML",
        "n_selected_cells": int(idx.size),
        "n_genes_tested": int(genes.size),
        "n_genes_significant": int(len(signi_genes)),
        "spline_k": SPLINE_K,
        "fdr_cut": FDR_CUT,
        "amplitude_cut": AMPLITUDE_CUT,
        "p_adjust_method": "fdr_bh",
        "grid_size": GRID_SIZE,
        "histogram_bins": HISTOGRAM_BINS,
        "display_grid": "uniform_pseudotime_reference_prediction",
        "cell_density_track": True,
        "row_ordering": "peak_pseudotime",
        "max_workers": MAX_WORKERS,
        "n_gene_tasks": len(tasks),
        "figure_path": str(fig_path),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
