# %% [markdown]
# # Analyze cNMF outputs for `all_progenitors`
#
# This pypercent script generates plots from an existing cNMF run:
# - K selection plot
# - consensus + exports for a sweep of `K` values
# - UMAP colored by cNMF usage programs per K

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
from cnmf import cNMF

# %%
# === EDIT THESE ===

OUTDIR = Path("~/nvme/cnmf_all_progenitors").expanduser()
RUN_NAME = "all_progenitors_cNMF"
K_VALUES = list(range(8, 12))
DENSITY_THRESHOLD = 0.1
# UMAP parameters for usage visualization
UMAP_N_NEIGHBORS = 25
UMAP_N_PCS = 15
# Plot all programs by default; set an int to cap the number of Usage_* panels.
MAX_PROGRAMS_TO_PLOT: int | None = None
WRITE_UMAP_H5AD = True

# %%
cnmf_output_dir = OUTDIR / "cNMF"
cnmf_run_dir = cnmf_output_dir / RUN_NAME
counts_h5ad = OUTDIR / "counts.filtered_for_cnmf.h5ad"
overdispersed_genes_txt = cnmf_run_dir / f"{RUN_NAME}.overdispersed_genes.txt"

for required in [cnmf_run_dir, counts_h5ad, overdispersed_genes_txt]:
    if not required.exists():
        raise FileNotFoundError(f"Required path not found: {required}")

cnmf_obj = cNMF(output_dir=str(cnmf_output_dir), name=RUN_NAME)

# %% [markdown]
# ## Phase 1: Plot K selection

# %%
cnmf_obj.k_selection_plot()
print(f"K selection plot: {cnmf_obj.paths['k_selection_plot']}")

# %% [markdown]
# ## Phase 2: Consensus + export for K sweep

# %%
adata = sc.read_h5ad(counts_h5ad)
if "dataset" not in adata.obs.columns:
    raise KeyError("Expected 'dataset' column in counts.filtered_for_cnmf.h5ad obs.")

dataset_by_index = adata.obs["dataset"].copy()
dataset_by_index.index = dataset_by_index.index.astype(str)
base_obs = adata.obs.copy()

for k in K_VALUES:
    print(f"=== k={k}, dt={DENSITY_THRESHOLD} ===")
    cnmf_obj.consensus(
        k=int(k),
        density_threshold=float(DENSITY_THRESHOLD),
        show_clustering=False,
    )

    usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.load_results(
        K=int(k), density_threshold=float(DENSITY_THRESHOLD)
    )
    usage_norm.columns = [f"Usage_{i}" for i in usage_norm.columns]

    usage_norm.to_csv(OUTDIR / f"usage_norm.k{k}.dt{DENSITY_THRESHOLD}.tsv", sep="\t")
    gep_scores.to_csv(OUTDIR / f"gep_scores.k{k}.dt{DENSITY_THRESHOLD}.tsv", sep="\t")
    gep_tpm.to_csv(OUTDIR / f"gep_tpm.k{k}.dt{DENSITY_THRESHOLD}.tsv", sep="\t")
    topgenes.to_csv(OUTDIR / f"topgenes.k{k}.dt{DENSITY_THRESHOLD}.tsv", sep="\t")

    usage_table = usage_norm.copy()
    usage_table.index = usage_table.index.astype(str)
    usage_dataset = dataset_by_index.reindex(usage_table.index)
    if usage_dataset.isna().any():
        missing = usage_dataset[usage_dataset.isna()].index[:5].tolist()
        raise ValueError(
            f"Found {int(usage_dataset.isna().sum())} usage rows missing dataset metadata; "
            f"example index values: {missing}"
        )

    usage_export = usage_table.copy()
    usage_export.insert(0, "dataset", usage_dataset.to_numpy())
    usage_export.insert(0, "index", usage_export.index)
    usage_parquet = OUTDIR / f"usage_norm.k{k}.dt{DENSITY_THRESHOLD}.parquet"
    usage_export.to_parquet(usage_parquet, index=False)
    print(f"Wrote usage parquet: {usage_parquet} ({usage_export.shape[0]:,} rows)")

    adata.obs = base_obs.join(usage_norm, how="left")
    if MAX_PROGRAMS_TO_PLOT is None:
        usage_cols = list(usage_norm.columns)
    else:
        usage_cols = list(usage_norm.columns[:MAX_PROGRAMS_TO_PLOT])

    umap_path = OUTDIR / f"umap_usages.k{k}.dt{DENSITY_THRESHOLD}.png"
    ax = sc.pl.umap(
        adata,
        color=[*usage_cols, "tricycle"],
        ncols=3,
        vmin=0,
        vmax=1,
        cmap="CMRmap_r",
        show=False,
    )
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
    fig.savefig(umap_path, dpi=200)
    plt.close(fig)
    print(f"UMAP usage plot: {umap_path}")

    if WRITE_UMAP_H5AD:
        adata.write_h5ad(OUTDIR / f"umap_with_usages.k{k}.dt{DENSITY_THRESHOLD}.h5ad")

# %%
