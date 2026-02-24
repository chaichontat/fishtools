# %% [markdown]
# # Run cNMF on `all_progenitors.h5ad`
#
# Minimal pypercent workflow for preprocessed data:
# - load counts from `adata.layers["raw"]`
# - subsample cells
# - run cNMF prepare/factorize/combine/consensus
# - save usage and GEP tables

# %%
from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from cnmf import cNMF

# %%
# === EDIT THESE ===

IN_H5AD = Path("/fast2/cs_outputs/all_progenitors.h5ad")
OUTDIR = Path("/fast2/cs_outputs/fishtools2/_out/cnmf_all_progenitors")
COUNTS_LAYER = "raw"
SUBSAMPLE_N_CELLS = 50_000

RUN_NAME = "all_progenitors_cNMF"
SEED = 14
N_ITER = 50
NUM_HIGHVAR_GENES = 2000
K_MIN = 5
K_MAX = 10
TOTAL_WORKERS = 16
OVERWRITE_CNMF_RUN = True

SELECTED_K = 7
DENSITY_THRESHOLD = 20.0

# %%
OUTDIR.mkdir(parents=True, exist_ok=True)
counts_h5ad = OUTDIR / "counts.filtered_for_cnmf.h5ad"
cnmf_output_dir = OUTDIR / "cNMF"
cnmf_output_dir.mkdir(parents=True, exist_ok=True)
components = np.arange(K_MIN, K_MAX + 1)

params = {
    "IN_H5AD": str(IN_H5AD),
    "OUTDIR": str(OUTDIR),
    "COUNTS_LAYER": COUNTS_LAYER,
    "SUBSAMPLE_N_CELLS": SUBSAMPLE_N_CELLS,
    "RUN_NAME": RUN_NAME,
    "SEED": SEED,
    "N_ITER": N_ITER,
    "NUM_HIGHVAR_GENES": NUM_HIGHVAR_GENES,
    "K_MIN": K_MIN,
    "K_MAX": K_MAX,
    "TOTAL_WORKERS": TOTAL_WORKERS,
    "OVERWRITE_CNMF_RUN": OVERWRITE_CNMF_RUN,
    "SELECTED_K": SELECTED_K,
    "DENSITY_THRESHOLD": DENSITY_THRESHOLD,
}
pd.Series(params).to_json(OUTDIR / "params.json", indent=2)

# %% [markdown]
# ## Phase 0: Load counts and subsample

# %%
adata_in = sc.read_h5ad(IN_H5AD, backed="r")
print(f"Loaded: n_obs={adata_in.n_obs:,}, n_vars={adata_in.n_vars:,}")

if COUNTS_LAYER not in adata_in.layers:
    raise KeyError(
        f"COUNTS_LAYER={COUNTS_LAYER!r} not found. Available: {sorted(adata_in.layers.keys())}"
    )

if SUBSAMPLE_N_CELLS is None or SUBSAMPLE_N_CELLS >= int(adata_in.n_obs):
    cell_pos = np.arange(int(adata_in.n_obs), dtype=int)
else:
    rng = np.random.default_rng(SEED)
    cell_pos = np.sort(
        rng.choice(int(adata_in.n_obs), size=int(SUBSAMPLE_N_CELLS), replace=False)
    )

np.save(OUTDIR / "subsample_cell_positions.npy", cell_pos)
print(f"Using cells: n={cell_pos.size:,}")

X_sub = adata_in.layers[COUNTS_LAYER][cell_pos, :]
if sp.issparse(X_sub):
    X_sub = X_sub.tocsr()
else:
    X_sub = sp.csr_matrix(np.asarray(X_sub))

obs = adata_in.obs.iloc[cell_pos].copy()
var = adata_in.var.copy()
adata_counts = sc.AnnData(X=X_sub, obs=obs, var=var)
if "X_umap" in adata_in.obsm:
    adata_counts.obsm["X_umap"] = np.asarray(adata_in.obsm["X_umap"][cell_pos, :])
adata_counts.obs_names_make_unique()
adata_counts.var_names_make_unique()
print(
    f"Prepared counts: shape={adata_counts.shape}, nnz={adata_counts.X.nnz:,}, dtype={adata_counts.X.dtype}"
)

adata_counts.write_h5ad(counts_h5ad)
print(f"Wrote: {counts_h5ad}")

# %% [markdown]
# ## Phase 1: cNMF prepare + factorize

# %%
cnmf_run_dir = cnmf_output_dir / RUN_NAME
if OVERWRITE_CNMF_RUN and cnmf_run_dir.exists():
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
    with (factorize_logs_dir / f"worker_{worker_i:02d}.log").open("w") as f:
        f.write(" ".join(cmd) + "\n")
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as ex:
    futures = [ex.submit(_run_factorize_worker, i) for i in range(TOTAL_WORKERS)]
    for fut in as_completed(futures):
        fut.result()

# %% [markdown]
# ## Phase 2: combine + consensus + export tables

# %%
cnmf_obj.combine()
cnmf_obj.consensus(
    k=SELECTED_K,
    density_threshold=float(DENSITY_THRESHOLD),
    show_clustering=False,
    close_clustergram_fig=True,
)

dt = float(DENSITY_THRESHOLD)
usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.load_results(
    K=SELECTED_K, density_threshold=dt
)
usage_norm.columns = [f"Usage_{i}" for i in usage_norm.columns]

usage_norm.to_csv(OUTDIR / f"usage_norm.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
gep_scores.to_csv(OUTDIR / f"gep_scores.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
gep_tpm.to_csv(OUTDIR / f"gep_tpm.k{SELECTED_K}.dt{dt}.tsv", sep="\t")
topgenes.to_csv(OUTDIR / f"topgenes.k{SELECTED_K}.dt{dt}.tsv", sep="\t")

print("Saved:")
print(f"- usage_norm: {usage_norm.shape}")
print(f"- gep_scores: {gep_scores.shape}")
print(f"- gep_tpm: {gep_tpm.shape}")
print(f"- topgenes: {topgenes.shape}")
