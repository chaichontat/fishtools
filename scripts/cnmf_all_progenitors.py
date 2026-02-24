# %% [markdown]
# # Run cNMF on `all_progenitors.h5ad`
#
# This is a pypercent (cell-based) translation of the official cNMF PBMC tutorial notebook:
# `references/cnmf/analyze_pbmc_example_data.ipynb`.
#
# Run cells sequentially. Each phase writes outputs to `OUTDIR` for inspection and fast reruns.

# %%
from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from cnmf import cNMF
from IPython import get_ipython

ip = get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

# %%
# === EDIT THESE ===

# Input AnnData. This should contain *raw counts* in `.X`, a counts layer, or `.raw`.
IN_H5AD = Path("/fast2/cs_outputs/all_progenitors.h5ad")

# All artifacts (filtered counts, cNMF outputs, plots) go here.
OUTDIR = Path("/fast2/cs_outputs/fishtools2/_out/cnmf_all_progenitors")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Choose where counts live in the input h5ad (fail fast if the selection doesn't look count-like).
USE_RAW_IF_AVAILABLE = False
COUNTS_LAYER: str | None = "raw"  # e.g. "counts"

# Filtering (tutorial defaults), but disabled by request.
DO_FILTER = False
MIN_GENES_PER_CELL = 200
MIN_COUNTS_PER_CELL = 200  # mainly to populate `n_counts` like the tutorial
MIN_CELLS_PER_GENE = 3

# Subsample for quicker iteration (set to `None` to use all cells).
SUBSAMPLE_N_CELLS: int | None = 50_000

# Large datasets can be huge on disk if stored dense. This keeps the same values but stores `.X` as CSR.
CONVERT_DENSE_TO_SPARSE = True

# cNMF run parameters.
RUN_NAME = "all_progenitors_cNMF"
SEED = 14
N_ITER = 50  # tutorial uses 20; for real runs consider ~200+
NUM_HIGHVAR_GENES = 2000
K_MIN = 5
K_MAX = 10
TOTAL_WORKERS = 16

# If rerunning with the same RUN_NAME, delete the previous cNMF run directory so
# factorization jobs don't get treated as already-completed.
OVERWRITE_CNMf_RUN = True

# Consensus parameters (see tutorial notes on using >=2.0 to disable outlier filtering).
SELECTED_K = 7
# Use a large value to effectively disable outlier filtering (tutorial notes: >=2.0).
DENSITY_THRESHOLD = 20.0

# Postprocess / plotting.
UMAP_N_NEIGHBORS = 50
UMAP_N_PCS = 15
MAX_PROGRAMS_TO_PLOT = 12

# %%
print(f"IN_H5AD={IN_H5AD}")
print(f"OUTDIR={OUTDIR}")

counts_h5ad = OUTDIR / "counts.filtered_for_cnmf.h5ad"
cnmf_output_dir = OUTDIR / "cNMF"
cnmf_output_dir.mkdir(parents=True, exist_ok=True)

components = np.arange(K_MIN, K_MAX + 1)
print(f"K components={components.tolist()}")

params = {
    "IN_H5AD": str(IN_H5AD),
    "OUTDIR": str(OUTDIR),
    "USE_RAW_IF_AVAILABLE": USE_RAW_IF_AVAILABLE,
    "COUNTS_LAYER": COUNTS_LAYER,
    "DO_FILTER": DO_FILTER,
    "MIN_GENES_PER_CELL": MIN_GENES_PER_CELL,
    "MIN_COUNTS_PER_CELL": MIN_COUNTS_PER_CELL,
    "MIN_CELLS_PER_GENE": MIN_CELLS_PER_GENE,
    "SUBSAMPLE_N_CELLS": SUBSAMPLE_N_CELLS,
    "CONVERT_DENSE_TO_SPARSE": CONVERT_DENSE_TO_SPARSE,
    "RUN_NAME": RUN_NAME,
    "SEED": SEED,
    "N_ITER": N_ITER,
    "NUM_HIGHVAR_GENES": NUM_HIGHVAR_GENES,
    "K_MIN": K_MIN,
    "K_MAX": K_MAX,
    "TOTAL_WORKERS": TOTAL_WORKERS,
    "OVERWRITE_CNMf_RUN": OVERWRITE_CNMf_RUN,
    "SELECTED_K": SELECTED_K,
    "DENSITY_THRESHOLD": DENSITY_THRESHOLD,
}
pd.Series(params).to_json(OUTDIR / "params.json", indent=2)

# %% [markdown]
# ## Phase 0: Load and validate counts

# %%
adata_in = sc.read_h5ad(IN_H5AD, backed="r")
print(f"Loaded: n_obs={adata_in.n_obs:,}, n_vars={adata_in.n_vars:,}")

if SUBSAMPLE_N_CELLS is None or SUBSAMPLE_N_CELLS >= int(adata_in.n_obs):
    cell_pos = np.arange(int(adata_in.n_obs), dtype=int)
else:
    rng = np.random.default_rng(SEED)
    cell_pos = np.sort(
        rng.choice(int(adata_in.n_obs), size=int(SUBSAMPLE_N_CELLS), replace=False)
    )

subsample_idx_path = OUTDIR / "subsample_cell_positions.npy"
np.save(subsample_idx_path, cell_pos)
print(f"Using cells: n={cell_pos.size:,} (saved positions to {subsample_idx_path})")

if USE_RAW_IF_AVAILABLE and adata_in.raw is not None:
    X_sub = np.asarray(adata_in.raw.X[cell_pos, :])
    var = adata_in.raw.var.copy()
    print("Using counts from `adata.raw.X`.")
elif COUNTS_LAYER is not None:
    if COUNTS_LAYER not in adata_in.layers:
        raise KeyError(
            f"COUNTS_LAYER={COUNTS_LAYER!r} not found. Available: {sorted(adata_in.layers.keys())}"
        )
    X_sub = np.asarray(adata_in.layers[COUNTS_LAYER][cell_pos, :])
    var = adata_in.var.copy()
    print(f"Using counts from `adata.layers[{COUNTS_LAYER!r}]`.")
else:
    X_sub = np.asarray(adata_in.X[cell_pos, :])
    var = adata_in.var.copy()
    print("Using counts from `adata.X`.")

obs = adata_in.obs.iloc[cell_pos].copy()
adata_counts = sc.AnnData(X=X_sub, obs=obs, var=var)
adata_counts.obs_names_make_unique()
adata_counts.var_names_make_unique()

X = adata_counts.X
if sp.issparse(X):
    sample = X.data[: min(X.data.size, 50_000)]
else:
    n_sample_rows = min(int(X.shape[0]), 2000)
    x_sample = np.asarray(X[:n_sample_rows, :])
    sample = x_sample.ravel()[: min(x_sample.size, 50_000)]

if sample.size == 0:
    raise ValueError("Counts matrix has no nonzero entries (or is empty).")
if np.min(sample) < 0:
    raise ValueError("Counts matrix contains negative values; cNMF expects nonnegative counts.")

max_abs_frac = float(np.max(np.abs(sample - np.round(sample))))
print(f"Counts sanity: max(|x-round(x)|) on sample = {max_abs_frac:g}")
if max_abs_frac > 1e-6:
    raise ValueError(
        "Counts do not look integer-like. Set `COUNTS_LAYER` (or ensure `.raw` contains counts) "
        "and rerun."
    )

if sp.issparse(X):
    print(f"X is sparse: shape={X.shape}, nnz={X.nnz:,}, dtype={X.dtype}")
else:
    print(f"X is dense: shape={X.shape}, dtype={np.asarray(X).dtype}")

if CONVERT_DENSE_TO_SPARSE and not sp.issparse(adata_counts.X):
    adata_counts.X = sp.csr_matrix(adata_counts.X)
    print(f"Converted X to CSR: nnz={adata_counts.X.nnz:,}, dtype={adata_counts.X.dtype}")

# %% [markdown]
# ## Phase 1: Filter and write counts `.h5ad` for cNMF


# %% [markdown]
# ## Phase 2: Run cNMF (prepare + factorize)
#
# Factorization is parallelized by launching `cnmf factorize` for each worker index via subprocess.

# %%
cnmf_run_dir = cnmf_output_dir / RUN_NAME
if OVERWRITE_CNMf_RUN and cnmf_run_dir.exists():
    shutil.rmtree(cnmf_run_dir)

cnmf_obj = cNMF(output_dir=str(cnmf_output_dir), name=RUN_NAME)

cnmf_obj.prepare(
    counts_fn=str(counts_h5ad),
    components=components,
    n_iter=N_ITER,
    seed=SEED,
    num_highvar_genes=NUM_HIGHVAR_GENES,
)

factorize_logs_dir = OUTDIR / "factorize_logs"
factorize_logs_dir.mkdir(parents=True, exist_ok=True)


def _run_factorize_worker(worker_i: int) -> None:
    cmd = [
        "cnmf",
        "factorize",
        "--output-dir",
        str(cnmf_output_dir),
        "--name",
        RUN_NAME,
        "--worker-index",
        str(worker_i),
        "--total-workers",
        str(TOTAL_WORKERS),
    ]
    log_path = factorize_logs_dir / f"worker_{worker_i:02d}.log"
    with log_path.open("w") as f:
        f.write(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


if TOTAL_WORKERS == 1:
    _run_factorize_worker(0)
else:
    with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as ex:
        futures = {ex.submit(_run_factorize_worker, i): i for i in range(TOTAL_WORKERS)}
        for fut in as_completed(futures):
            fut.result()

# %% [markdown]
# ## Phase 3: Combine + K selection plot

# %%
cnmf_obj.combine()
cnmf_obj.k_selection_plot(close_fig=False)
print(f"K-selection plot: {cnmf_obj.paths['k_selection_plot']}")

# %% [markdown]
# ## Phase 4: Consensus (selected K)
#
# - `density_threshold >= 2.0` disables outlier filtering (tutorial trick).

# %%
print(f"Consensus: k={SELECTED_K}, density_threshold={DENSITY_THRESHOLD}")
cnmf_obj.consensus(
    k=SELECTED_K,
    density_threshold=float(DENSITY_THRESHOLD),
    show_clustering=True,
    close_clustergram_fig=False,
)

# %% [markdown]
# ## Phase 5: Load results + save tables

# %%
dt = float(DENSITY_THRESHOLD)
usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.load_results(
    K=SELECTED_K, density_threshold=dt
)

usage_norm.columns = [f"Usage_{i}" for i in usage_norm.columns]
print("Loaded:")
print(f"- usage_norm: {usage_norm.shape}")
print(f"- gep_scores: {gep_scores.shape}")
print(f"- gep_tpm: {gep_tpm.shape}")
print(f"- topgenes: {topgenes.shape}")

usage_norm.to_csv(OUTDIR / f"usage_norm.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
gep_scores.to_csv(OUTDIR / f"gep_scores.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
gep_tpm.to_csv(OUTDIR / f"gep_tpm.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
topgenes.to_csv(OUTDIR / f"topgenes.k{SELECTED_K}.dt{dt}.tsv", sep="\t")

# %% [markdown]
# ## Phase 6: UMAP + visualize usages (optional)
#
# Mirrors tutorial steps:
# TPT normalize → set `.raw` to log1p copy → subset to cNMF HVGs → scale → PCA → neighbors → UMAP.

# %%
# adata = sc.read_h5ad(counts_h5ad)

# overdispersed_genes_txt = (
#     cnmf_output_dir / RUN_NAME / f"{RUN_NAME}.overdispersed_genes.txt"
# )
# hvgs = [g for g in overdispersed_genes_txt.read_text().splitlines() if g]
# print(f"HVGs from cNMF: {len(hvgs):,} (file: {overdispersed_genes_txt})")

# sc.pp.normalize_per_cell(adata, counts_per_cell_after=10**4)
# adata.raw = sc.pp.log1p(adata.copy(), copy=True)

# adata = adata[:, hvgs]
# sc.pp.scale(adata)
# sc.pp.pca(adata)
# sc.pp.neighbors(adata, n_neighbors=UMAP_N_NEIGHBORS, n_pcs=UMAP_N_PCS)
# sc.tl.umap(adata)

# adata.obs = pd.merge(
#     left=adata.obs, right=usage_norm, how="left", left_index=True, right_index=True
# )

# usage_cols = list(usage_norm.columns[:MAX_PROGRAMS_TO_PLOT])
# ax = sc.pl.umap(
#     adata,
#     color=usage_cols,
#     use_raw=True,
#     ncols=3,
#     vmin=0,
#     vmax=1,
#     show=False,
# )
# fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
# fig.savefig(OUTDIR / f"umap_usages.k{SELECTED_K}.dt{dt}.png", dpi=200)
# plt.show()

# adata.write_h5ad(OUTDIR / f"umap_with_usages.k{SELECTED_K}.dt{dt}.h5ad")
