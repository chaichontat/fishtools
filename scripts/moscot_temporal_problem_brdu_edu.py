# %% [markdown]
# # Moscot temporal problem on BrdU/EdU-labeled cells
#
# This script adapts the moscot tutorial (TemporalProblem) to `~/nvme/all_r300_excit_progenitors.h5ad`.
#
# Run cells sequentially. Each phase writes outputs to `OUTDIR` for inspection and fast reruns.
#
# Time ordering (as requested):
# - `edu_pos` only: oldest
# - `brdu_pos` + `edu_pos`: intermediate
# - `brdu_pos` only: youngest

# %% [markdown]
# ## Configuration

# %%
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, read_h5ad
from fishtools.brdu.ot import TemporalProblemConfig, fit_temporal_problem, make_brdu_pair_cost_builder
from fishtools.brdu.temporal_order import assign_temporal_order_from_brdu_edu
from matplotlib.collections import LineCollection
from moscot.problems.time import TemporalProblem
from scipy.sparse.csgraph import connected_components

try:
    import moscot.plotting as mpl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import `moscot.plotting`. If you're running headless, this is still required for saving figures."
    ) from e

# %%
# === EDIT THESE ===

INFILE = Path("~/nvme/vz.h5ad").expanduser()
OUTDIR = Path("results/moscot_temporal_problem_brdu_edu_n5000")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0

# Dataset-specific keys
BRDU_KEY = "brdu_pos"
EDU_KEY = "edu_pos"
TIME_KEY = "time"

# Sampling: the full dataset is very large; default to a deterministic subset per timepoint
SAMPLE_N_PER_TIME = 5000

# Representation used by moscot + neighbors/leiden
JOINT_ATTR = "X_pca"
UMAP_BASIS = "umap"
TRICYCLE_KEY = "tricycle"
AP_KEY = "ap"
ML_KEY = "ml"
BACKWARD_TRICYCLE_PENALTY = 30.0
AP_ML_DISPLACEMENT_PENALTY = 10.0

# Optional: use graph-based (geodesic) cost like in the tutorial
USE_GRAPH_COST = False
PAIR_N_NEIGHBORS = 50
ASSERT_PAIR_GRAPH_CONNECTED = True
GRAPH_HEAT_T = 100.0

# Leiden clustering (used as a "cell type" proxy for transition summaries)
LEIDEN_KEY = "leiden"

# OT solver knobs (keep small enough to run on the sampled dataset)
EPSILON = 1e-3
TAU_A = 0.95
TAU_B = 0.95
MAX_ITERATIONS = 200_000

# Expression used for driver gene/TF correlation.
# - None: use `tp.adata.X` as-is (no normalization/log transform done in this script).
# - str: use `tp.adata.layers[EXPR_LAYER]`.
EXPR_LAYER: str | None = None

# %%
rng = np.random.default_rng(RANDOM_SEED)
print(f"INFILE={INFILE}")
print(f"OUTDIR={OUTDIR}")


# %% [markdown]
# ## Phase 0: Load + subset labeled cells

# %%
def _sample_indices(mask: np.ndarray, *, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if n >= len(idx):
        return idx
    return rng.choice(idx, size=n, replace=False)


def _barycentric_pull(subproblem: object, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each source cell to the transport-weighted centroid of target UMAP coordinates."""
    numer = np.asarray(subproblem.pull(data=target_xy, normalize=False, scale_by_marginals=False), dtype=float)
    denom = np.asarray(
        subproblem.pull(
            data=np.ones((target_xy.shape[0], 1), dtype=float), normalize=False, scale_by_marginals=False
        ),
        dtype=float,
    )
    if numer.ndim != 2 or numer.shape[1] < 2:
        raise ValueError(f"Expected pulled coordinates with shape (n_src, >=2), found {numer.shape}.")
    if denom.ndim == 1:
        denom = denom[:, None]
    valid = np.isfinite(denom[:, 0]) & (denom[:, 0] > 0)
    mapped = np.full_like(numer[:, :2], fill_value=np.nan, dtype=float)
    mapped[valid] = numer[valid, :2] / denom[valid]
    return mapped, valid


def _plot_umap_barycentric_transition(
    *,
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    mapped_xy: np.ndarray,
    valid: np.ndarray,
    source: int,
    target: int,
    out_png: Path,
) -> None:
    segments = np.stack([src_xy[valid], mapped_xy[valid]], axis=1)
    displacement = np.linalg.norm(mapped_xy[valid] - src_xy[valid], axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(tgt_xy[:, 0], tgt_xy[:, 1], s=4, c="#d9d9d9", alpha=0.5, linewidths=0, rasterized=True)

    lc = LineCollection(
        segments,
        cmap="viridis",
        array=displacement,
        linewidths=0.25,
        alpha=0.2,
        rasterized=True,
    )
    ax.add_collection(lc)
    scatt = ax.scatter(
        src_xy[valid, 0],
        src_xy[valid, 1],
        c=displacement,
        s=4,
        cmap="viridis",
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        mapped_xy[valid, 0],
        mapped_xy[valid, 1],
        s=3,
        c="#b2182b",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    cbar = fig.colorbar(scatt, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Barycentric UMAP displacement")
    ax.set_title(f"UMAP barycentric transport: time {source} -> {target}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


adata_b = read_h5ad(INFILE, backed="r")
obs = adata_b.obs

spatial_ok = np.isfinite(obs[AP_KEY].to_numpy(dtype=np.float32)) & np.isfinite(obs[ML_KEY].to_numpy(dtype=np.float32))
mask_old = spatial_ok & ~obs[BRDU_KEY].to_numpy(dtype=bool) & obs[EDU_KEY].to_numpy(dtype=bool)
mask_mid = spatial_ok & obs[BRDU_KEY].to_numpy(dtype=bool) & obs[EDU_KEY].to_numpy(dtype=bool)
mask_young = spatial_ok & obs[BRDU_KEY].to_numpy(dtype=bool) & ~obs[EDU_KEY].to_numpy(dtype=bool)

idx_old = _sample_indices(mask_old, n=SAMPLE_N_PER_TIME, rng=rng)
idx_mid = _sample_indices(mask_mid, n=SAMPLE_N_PER_TIME, rng=rng)
idx_young = _sample_indices(mask_young, n=SAMPLE_N_PER_TIME, rng=rng)

if len(idx_old) == 0 or len(idx_mid) == 0 or len(idx_young) == 0:
    raise ValueError(
        "One of the BrdU/EdU time groups has 0 eligible cells after requiring finite AP/ML coordinates. "
        f"Counts: old={len(idx_old)}, mid={len(idx_mid)}, young={len(idx_young)}."
    )

sel = np.sort(np.concatenate([idx_old, idx_mid, idx_young]))
adata: AnnData = adata_b[sel].to_memory()
adata.obs_names_make_unique()

time = assign_temporal_order_from_brdu_edu(
    brdu_pos=adata.obs[BRDU_KEY].to_numpy(dtype=bool),
    edu_pos=adata.obs[EDU_KEY].to_numpy(dtype=bool),
)
if np.any(time < 0):
    raise ValueError("Found cells that don't match any of the (brdu_pos, edu_pos) combinations used for time ordering.")

adata.obs[TIME_KEY] = pd.Categorical(time, categories=[0, 1, 2], ordered=True)
adata = adata[adata.obs.sort_values(TIME_KEY).index].copy()

counts = adata.obs[TIME_KEY].value_counts().sort_index()
print("subset shape:", adata.shape)
print("counts per time:\n", counts.to_string())
print("obsm keys:", list(adata.obsm.keys()))

subset_path = OUTDIR / "subset.h5ad"
adata.write_h5ad(subset_path)
print(f"wrote {subset_path}")


# %% [markdown]
# ## Phase 1: Validate existing Leiden clusters (for transition summaries)

# %%
if JOINT_ATTR not in adata.obsm:
    raise KeyError(f"Expected `{JOINT_ATTR}` in `adata.obsm`, found: {list(adata.obsm.keys())}")

umap_key = f"X_{UMAP_BASIS}"
if umap_key not in adata.obsm:
    raise KeyError(f"Expected `{umap_key}` in `adata.obsm`, found: {list(adata.obsm.keys())}")

if TRICYCLE_KEY not in adata.obs:
    raise KeyError(f"Expected `{TRICYCLE_KEY}` in `adata.obs`, found columns: {list(adata.obs.columns)}")

if LEIDEN_KEY not in adata.obs:
    raise KeyError(
        f"Expected `{LEIDEN_KEY}` in `adata.obs`, found columns: {list(adata.obs.columns)}. "
        "You said Leiden clusters already exist in the data; if the key differs, update `LEIDEN_KEY`."
    )

for key in (AP_KEY, ML_KEY):
    if key not in adata.obs:
        raise KeyError(f"Expected `{key}` in `adata.obs`, found columns: {list(adata.obs.columns)}")

ap_ml = adata.obs[[AP_KEY, ML_KEY]].to_numpy(dtype=np.float32)
if not np.isfinite(ap_ml).all():
    raise ValueError(f"Found non-finite values in `{AP_KEY}`/`{ML_KEY}`; expected finite AP/ML coordinates.")

cluster_counts = adata.obs[LEIDEN_KEY].value_counts()
print(f"`{LEIDEN_KEY}` clusters: n={len(cluster_counts)}")
print(cluster_counts.head(20).to_string())


# %% [markdown]
# ## Phase 2: Prepare TemporalProblem

# %%
if USE_GRAPH_COST:
    if BACKWARD_TRICYCLE_PENALTY > 0 or AP_ML_DISPLACEMENT_PENALTY > 0:
        raise ValueError("`USE_GRAPH_COST` cannot be combined with explicit linear AP/ML or tricycle penalties.")
    tp = TemporalProblem(adata=adata).prepare(time_key=TIME_KEY, joint_attr=JOINT_ATTR)
    time_points = sorted({t for pair in tp.problems for t in pair})
    print("prepared TemporalProblem with timepoints:", time_points)
    for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
        sub = tp[t_src, t_tgt]
        pair_obs_names = list(sub.adata_src.obs_names) + list(sub.adata_tgt.obs_names)

        adata_pair = adata[pair_obs_names].copy()
        sc.pp.neighbors(adata_pair, use_rep=JOINT_ATTR, n_neighbors=PAIR_N_NEIGHBORS)
        conn = adata_pair.obsp["connectivities"].tocsr().astype(float)

        if ASSERT_PAIR_GRAPH_CONNECTED:
            n_components, _ = connected_components(conn, directed=False, return_labels=True)
            if n_components != 1:
                raise ValueError(
                    f"Pair graph for ({t_src} -> {t_tgt}) has {n_components} connected components. "
                    f"Try increasing `PAIR_N_NEIGHBORS` or reducing `SAMPLE_N_PER_TIME`."
                )

        idx = adata_pair.obs_names.to_series()
        sub.set_graph_xy((conn, idx, idx), t=GRAPH_HEAT_T)

    print("set graph-based costs for all adjacent time pairs")
    tp = tp.solve(
        epsilon=EPSILON,
        tau_a=TAU_A,
        tau_b=TAU_B,
        scale_cost="mean",
        max_iterations=MAX_ITERATIONS,
    )
    print("solved TemporalProblem")
else:
    cfg = TemporalProblemConfig(
        time_key=TIME_KEY,
        joint_attr=JOINT_ATTR,
        policy="sequential",
        cost="sq_euclidean",
        epsilon=EPSILON,
        tau_a=TAU_A,
        tau_b=TAU_B,
        scale_cost="mean",
        max_iterations=MAX_ITERATIONS,
    )
    tp, time_points = fit_temporal_problem(
        adata,
        cfg,
        pair_cost_builder=make_brdu_pair_cost_builder(
            joint_attr=JOINT_ATTR,
            tricycle_key=TRICYCLE_KEY,
            ap_key=AP_KEY,
            ml_key=ML_KEY,
            backward_tricycle_penalty=BACKWARD_TRICYCLE_PENALTY,
            ap_ml_penalty=AP_ML_DISPLACEMENT_PENALTY,
        ),
    )
    print("prepared TemporalProblem with timepoints:", time_points)
    print(
        "set custom pairwise costs with "
        f"backward tricycle penalty={BACKWARD_TRICYCLE_PENALTY} and "
        f"AP/ML displacement penalty={AP_ML_DISPLACEMENT_PENALTY}"
    )
    print("solved TemporalProblem")


# %% [markdown]
# ## Phase 5: UMAP barycentric transition maps

# %%
umap_outdir = OUTDIR / "umap_transitions"
umap_outdir.mkdir(parents=True, exist_ok=True)

for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
    sub = tp[t_src, t_tgt]
    src_xy = np.asarray(sub.adata_src.obsm[umap_key], dtype=float)[:, :2]
    tgt_xy = np.asarray(sub.adata_tgt.obsm[umap_key], dtype=float)[:, :2]
    mapped_xy, valid = _barycentric_pull(sub, tgt_xy)

    mapped_csv = umap_outdir / f"barycentric_umap_{t_src}_to_{t_tgt}.csv"
    pd.DataFrame(
        {
            "obs_name": sub.adata_src.obs_names.to_numpy(),
            "time_source": np.full(sub.adata_src.n_obs, t_src, dtype=int),
            "time_target": np.full(sub.adata_src.n_obs, t_tgt, dtype=int),
            "umap1_src": src_xy[:, 0],
            "umap2_src": src_xy[:, 1],
            "umap1_mapped": mapped_xy[:, 0],
            "umap2_mapped": mapped_xy[:, 1],
            "mapping_valid": valid,
            "umap_displacement": np.linalg.norm(mapped_xy - src_xy, axis=1),
        }
    ).to_csv(mapped_csv, index=False)
    print(f"wrote {mapped_csv}")

    out_png = umap_outdir / f"barycentric_umap_{t_src}_to_{t_tgt}.png"
    _plot_umap_barycentric_transition(
        src_xy=src_xy,
        tgt_xy=tgt_xy,
        mapped_xy=mapped_xy,
        valid=valid,
        source=t_src,
        target=t_tgt,
        out_png=out_png,
    )
    print(f"wrote {out_png}")


# %% [markdown]
# ## Phase 6: Cell transition summaries + plots

# %%
plt.rcParams["figure.dpi"] = 120

time_points = sorted({t for pair in tp.problems for t in pair})
clusters = sorted(adata.obs[LEIDEN_KEY].unique().tolist())

for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
    df = tp.cell_transition(
        t_src,
        t_tgt,
        {LEIDEN_KEY: clusters},
        {LEIDEN_KEY: clusters},
        forward=True,
        key_added=f"{LEIDEN_KEY}_{t_src}_to_{t_tgt}",
    )
    out_csv = OUTDIR / f"cell_transition_{LEIDEN_KEY}_{t_src}_to_{t_tgt}.csv"
    df.to_csv(out_csv)
    print(f"wrote {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 7))
    mpl.cell_transition(tp, key=f"{LEIDEN_KEY}_{t_src}_to_{t_tgt}", ax=ax)
    fig.tight_layout()
    out_png = OUTDIR / f"cell_transition_{LEIDEN_KEY}_{t_src}_to_{t_tgt}.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


# %% [markdown]
# ## Phase 7: Transcriptomic drivers for adjacent transitions
#
# For each adjacent time pair:
# - push forward each source Leiden population onto the next timepoint and correlate those descendant
#   probabilities with target-time transcriptomes
# - pull back each target Leiden population onto the previous timepoint and correlate those ancestor
#   probabilities with source-time transcriptomes
#
# This uses `compute_feature_correlation()` directly on OT-derived push/pull probabilities rather than
# only on endpoint-style fates.

# %%
drivers_dir = OUTDIR / "drivers"
drivers_dir.mkdir(parents=True, exist_ok=True)

if EXPR_LAYER is not None and EXPR_LAYER not in tp.adata.layers:
    raise KeyError(f"Expected `EXPR_LAYER={EXPR_LAYER!r}` in `tp.adata.layers`, found: {list(tp.adata.layers.keys())}")


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _present_clusters_at_timepoint(*, adata: AnnData, timepoint: int, key: str) -> list[str]:
    obs = adata.obs.loc[adata.obs[TIME_KEY].astype(int) == timepoint, key]
    return sorted(obs.astype(str).unique().tolist())


adjacent_dir = drivers_dir / "adjacent"
adjacent_dir.mkdir(parents=True, exist_ok=True)

adjacent_summary_rows: list[dict[str, object]] = []

for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
    src_clusters = _present_clusters_at_timepoint(adata=adata, timepoint=t_src, key=LEIDEN_KEY)
    tgt_clusters = _present_clusters_at_timepoint(adata=adata, timepoint=t_tgt, key=LEIDEN_KEY)

    for leiden in src_clusters:
        leiden_id = _safe_id(leiden)
        obs_key = f"{LEIDEN_KEY}_{leiden_id}_push_{t_src}to{t_tgt}"
        n_cells = int(
            ((adata.obs[TIME_KEY].astype(int) == t_src) & (adata.obs[LEIDEN_KEY].astype(str) == leiden)).sum()
        )

        tp.push(t_src, t_tgt, data=LEIDEN_KEY, subset=leiden, key_added=obs_key, scale_by_marginals=True)
        drivers_genes = tp.compute_feature_correlation(obs_key=obs_key, annotation={TIME_KEY: [t_tgt]}, layer=EXPR_LAYER)
        drivers_tfs = tp.compute_feature_correlation(
            obs_key=obs_key,
            features="mouse",
            annotation={TIME_KEY: [t_tgt]},
            layer=EXPR_LAYER,
        )

        out_genes = adjacent_dir / f"drivers_genes_push_{LEIDEN_KEY}_{leiden_id}_{t_src}_to_{t_tgt}.csv"
        out_tfs = adjacent_dir / f"drivers_tfs_push_{LEIDEN_KEY}_{leiden_id}_{t_src}_to_{t_tgt}.csv"
        drivers_genes.to_csv(out_genes)
        drivers_tfs.to_csv(out_tfs)

        adjacent_summary_rows.append(
            {
                "transition": f"{t_src}->{t_tgt}",
                "direction": "push",
                "leiden": leiden,
                "expression_timepoint": t_tgt,
                "n_group_cells": n_cells,
                "top_tfs": ";".join(drivers_tfs.head(10).index.tolist()),
            }
        )

        print(f"[drivers][push] transition={t_src}->{t_tgt} leiden={leiden} wrote {out_tfs}")
        print(drivers_tfs.head(10).to_string())
        print()

    for leiden in tgt_clusters:
        leiden_id = _safe_id(leiden)
        obs_key = f"{LEIDEN_KEY}_{leiden_id}_pull_{t_src}to{t_tgt}"
        n_cells = int(
            ((adata.obs[TIME_KEY].astype(int) == t_tgt) & (adata.obs[LEIDEN_KEY].astype(str) == leiden)).sum()
        )

        tp.pull(t_src, t_tgt, data=LEIDEN_KEY, subset=leiden, key_added=obs_key, scale_by_marginals=True)
        drivers_genes = tp.compute_feature_correlation(obs_key=obs_key, annotation={TIME_KEY: [t_src]}, layer=EXPR_LAYER)
        drivers_tfs = tp.compute_feature_correlation(
            obs_key=obs_key,
            features="mouse",
            annotation={TIME_KEY: [t_src]},
            layer=EXPR_LAYER,
        )

        out_genes = adjacent_dir / f"drivers_genes_pull_{LEIDEN_KEY}_{leiden_id}_{t_src}_to_{t_tgt}.csv"
        out_tfs = adjacent_dir / f"drivers_tfs_pull_{LEIDEN_KEY}_{leiden_id}_{t_src}_to_{t_tgt}.csv"
        drivers_genes.to_csv(out_genes)
        drivers_tfs.to_csv(out_tfs)

        adjacent_summary_rows.append(
            {
                "transition": f"{t_src}->{t_tgt}",
                "direction": "pull",
                "leiden": leiden,
                "expression_timepoint": t_src,
                "n_group_cells": n_cells,
                "top_tfs": ";".join(drivers_tfs.head(10).index.tolist()),
            }
        )

        print(f"[drivers][pull] transition={t_src}->{t_tgt} leiden={leiden} wrote {out_tfs}")
        print(drivers_tfs.head(10).to_string())
        print()

adjacent_summary_df = pd.DataFrame(adjacent_summary_rows).sort_values(["transition", "direction", "leiden"])
adjacent_summary_path = adjacent_dir / "drivers_tfs_summary_adjacent.csv"
adjacent_summary_df.to_csv(adjacent_summary_path, index=False)
print(f"wrote {adjacent_summary_path}")


# %% [markdown]
# ## Phase 8: Driver genes and transcription factors (per endpoint-like Leiden fate)
#
# For each Leiden cluster, compute a pull-back distribution (ancestors) and correlate it with gene expression.
# We mimic the tutorial by combining pull distributions across adjacent time pairs and restricting correlation
# to timepoints 0 and 1 (the timepoints where ancestors live for a 3-timepoint setup).

# %%
driver_timepoints = [0, 1]
leiden_categories = adata.obs[LEIDEN_KEY].cat.categories.tolist()

summary_rows: list[dict[str, object]] = []

for leiden in leiden_categories:
    key_early = f"{LEIDEN_KEY}_{leiden}_pull_0to1"
    key_late = f"{LEIDEN_KEY}_{leiden}_pull_1to2"
    key = f"{LEIDEN_KEY}_{leiden}_pull"

    tp.pull(0, 1, data=LEIDEN_KEY, subset=leiden, key_added=key_early, normalize=False)
    tp.pull(1, 2, data=LEIDEN_KEY, subset=leiden, key_added=key_late, normalize=False)
    tp.adata.obs[key] = tp.adata.obs[key_early].fillna(0) + tp.adata.obs[key_late].fillna(0)

    drivers_genes = tp.compute_feature_correlation(obs_key=key, annotation={TIME_KEY: driver_timepoints}, layer=EXPR_LAYER)
    drivers_tfs = tp.compute_feature_correlation(
        obs_key=key, features="mouse", annotation={TIME_KEY: driver_timepoints}, layer=EXPR_LAYER
    )

    out_genes = drivers_dir / f"drivers_genes_{LEIDEN_KEY}_{leiden}.csv"
    out_tfs = drivers_dir / f"drivers_tfs_{LEIDEN_KEY}_{leiden}.csv"
    drivers_genes.to_csv(out_genes)
    drivers_tfs.to_csv(out_tfs)

    top_tfs = drivers_tfs.head(5).index.tolist()
    summary_rows.append({"leiden": leiden, "top_tfs": ";".join(top_tfs)})

    print(f"[drivers] leiden={leiden} wrote {out_tfs} and {out_genes}")
    print(drivers_tfs.head(10).to_string())
    print()

summary_df = pd.DataFrame(summary_rows).set_index("leiden").sort_index()
summary_path = drivers_dir / "drivers_tfs_summary.csv"
summary_df.to_csv(summary_path)
print(f"wrote {summary_path}")

print("done")
