# %% [markdown]
# # BrdU/EdU temporal sweep with AP/ML displacement penalty

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_h5ad

from fishtools.brdu.temporal_order import assign_temporal_order_from_brdu_edu
from fishtools.brdu.transport_cost import pairwise_sqeuclidean_with_backward_tricycle_penalty
from moscot.problems.time import TemporalProblem

# %%
INFILE = Path("~/nvme/vz.h5ad").expanduser()
OUTDIR = Path("results/moscot_temporal_problem_brdu_edu_20k_ap_ml_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0

BRDU_KEY = "brdu_pos"
EDU_KEY = "edu_pos"
TIME_KEY = "time"
TRICYCLE_KEY = "tricycle"
LEIDEN_KEY = "leiden"
JOINT_ATTR = "X_pca"
AP_KEY = "ap"
ML_KEY = "ml"

BACKWARD_TRICYCLE_PENALTY = 30.0
AP_ML_DISPLACEMENT_PENALTIES = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
PCA_DIMS = 20
BATCH_SIZE = 2048
EPSILON = 1e-3
TAU_A = 0.95
TAU_B = 0.95
MAX_ITERATIONS = 200_000
TOTAL_SAMPLE_N = 20_000


def _sample_indices(mask: np.ndarray, *, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if n >= len(idx):
        return idx
    return rng.choice(idx, size=n, replace=False)


def _allocate_counts(*, total: int, n_groups: int) -> list[int]:
    base = total // n_groups
    rem = total % n_groups
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def _barycentric_pull(subproblem: object, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numer = np.asarray(subproblem.pull(data=target_xy, normalize=False, scale_by_marginals=False), dtype=np.float32)
    denom = np.asarray(
        subproblem.pull(
            data=np.ones((target_xy.shape[0], 1), dtype=np.float32), normalize=False, scale_by_marginals=False
        ),
        dtype=np.float32,
    )
    if denom.ndim == 1:
        denom = denom[:, None]
    valid = np.isfinite(denom[:, 0]) & (denom[:, 0] > 0)
    mapped = np.full_like(numer, fill_value=np.nan, dtype=np.float32)
    mapped[valid] = numer[valid] / denom[valid]
    return mapped, valid


def _expected_sq_distance_from_plan(plan: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    mass = float(plan.sum())
    if mass <= 0:
        return float("nan")

    row = plan.sum(axis=1)
    col = plan.sum(axis=0)

    x2 = np.sum(x * x, axis=1)
    y2 = np.sum(y * y, axis=1)

    ex2 = float(np.sum(row * x2)) / mass
    ey2 = float(np.sum(col * y2)) / mass
    cross = float(np.sum(x * (plan @ y))) / mass
    return float(max(ex2 + ey2 - 2.0 * cross, 0.0))


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.nanquantile(x, q))


def _same_label_mass_frac(plan: np.ndarray, labels_x: np.ndarray, labels_y: np.ndarray) -> float:
    mass = float(plan.sum())
    if mass <= 0:
        return float("nan")

    out = 0.0
    labs = sorted(set(labels_x.tolist()) | set(labels_y.tolist()))
    for lab in labs:
        mx = labels_x == lab
        my = labels_y == lab
        if not mx.any() or not my.any():
            continue
        out += float(plan[np.ix_(mx, my)].sum())
    return out / mass


rng = np.random.default_rng(RANDOM_SEED)
adata_b = read_h5ad(INFILE, backed="r")
obs = adata_b.obs

spatial_ok = np.isfinite(obs[AP_KEY].to_numpy(dtype=np.float32)) & np.isfinite(obs[ML_KEY].to_numpy(dtype=np.float32))
mask_t0 = spatial_ok & ~obs[BRDU_KEY].to_numpy(dtype=bool) & obs[EDU_KEY].to_numpy(dtype=bool)
mask_t1 = spatial_ok & obs[BRDU_KEY].to_numpy(dtype=bool) & obs[EDU_KEY].to_numpy(dtype=bool)
mask_t2 = spatial_ok & obs[BRDU_KEY].to_numpy(dtype=bool) & ~obs[EDU_KEY].to_numpy(dtype=bool)
sample_counts = _allocate_counts(total=TOTAL_SAMPLE_N, n_groups=3)
if not mask_t0.any() or not mask_t1.any() or not mask_t2.any():
    raise ValueError(
        "One of the BrdU/EdU time groups has 0 eligible cells after requiring finite AP/ML coordinates."
    )
sel = np.sort(
    np.concatenate(
        [
            _sample_indices(mask_t0, n=sample_counts[0], rng=rng),
            _sample_indices(mask_t1, n=sample_counts[1], rng=rng),
            _sample_indices(mask_t2, n=sample_counts[2], rng=rng),
        ]
    )
)

adata = adata_b[sel].to_memory()
adata.obs_names_make_unique()

time = assign_temporal_order_from_brdu_edu(
    brdu_pos=adata.obs[BRDU_KEY].to_numpy(dtype=bool),
    edu_pos=adata.obs[EDU_KEY].to_numpy(dtype=bool),
)
if np.any(time < 0):
    raise ValueError("Found cells that do not match the three labeled BrdU/EdU states.")

adata.obs[TIME_KEY] = pd.Categorical(time, categories=[0, 1, 2], ordered=True)
adata = adata[adata.obs.sort_values(TIME_KEY).index].copy()

for key, location in ((JOINT_ATTR, "obsm"), (TRICYCLE_KEY, "obs"), (LEIDEN_KEY, "obs"), (AP_KEY, "obs"), (ML_KEY, "obs")):
    container = adata.obsm if location == "obsm" else adata.obs
    if key not in container:
        raise KeyError(f"Expected `{key}` in `adata.{location}`.")

effective_pca_dims = min(PCA_DIMS, adata.obsm[JOINT_ATTR].shape[1])

ap_ml = adata.obs[[AP_KEY, ML_KEY]].to_numpy(dtype=np.float32)
if not np.isfinite(ap_ml).all():
    raise ValueError(f"Found non-finite values in `{AP_KEY}`/`{ML_KEY}`; expected finite AP/ML coordinates.")

subset_path = OUTDIR / "subset_20k.h5ad"
adata.write_h5ad(subset_path)
print(f"wrote {subset_path}")
print("counts per time:")
print(adata.obs[TIME_KEY].value_counts().sort_index().to_string())

clusters = sorted(adata.obs[LEIDEN_KEY].unique().tolist())
summary_rows: list[dict[str, float | int]] = []

for penalty in AP_ML_DISPLACEMENT_PENALTIES:
    penalty_label = f"ap_ml_penalty_{str(penalty).replace('.', 'p')}"
    penalty_dir = OUTDIR / penalty_label
    penalty_dir.mkdir(parents=True, exist_ok=True)

    tp = TemporalProblem(adata=adata).prepare(time_key=TIME_KEY, joint_attr=JOINT_ATTR)
    time_points = sorted({t for pair in tp.problems for t in pair})
    for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
        sub = tp[t_src, t_tgt]
        cost = pairwise_sqeuclidean_with_backward_tricycle_penalty(
            src_features=np.asarray(sub.adata_src.obsm[JOINT_ATTR], dtype=np.float32),
            tgt_features=np.asarray(sub.adata_tgt.obsm[JOINT_ATTR], dtype=np.float32),
            src_tricycle=sub.adata_src.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32),
            tgt_tricycle=sub.adata_tgt.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32),
            backward_penalty_weight=BACKWARD_TRICYCLE_PENALTY,
            src_ap=sub.adata_src.obs[AP_KEY].to_numpy(dtype=np.float32),
            tgt_ap=sub.adata_tgt.obs[AP_KEY].to_numpy(dtype=np.float32),
            src_ml=sub.adata_src.obs[ML_KEY].to_numpy(dtype=np.float32),
            tgt_ml=sub.adata_tgt.obs[ML_KEY].to_numpy(dtype=np.float32),
            ap_ml_penalty_weight=penalty,
        )
        sub.set_xy(pd.DataFrame(cost, index=sub.adata_src.obs_names, columns=sub.adata_tgt.obs_names), tag="cost_matrix")
        del cost

    tp = tp.solve(
        epsilon=EPSILON,
        tau_a=TAU_A,
        tau_b=TAU_B,
        scale_cost="mean",
        max_iterations=MAX_ITERATIONS,
        batch_size=BATCH_SIZE,
    )
    print(f"solved backward_penalty={BACKWARD_TRICYCLE_PENALTY}, ap_ml_penalty={penalty}")

    for t_src, t_tgt in zip(time_points[:-1], time_points[1:], strict=True):
        sub = tp[t_src, t_tgt]
        plan = np.asarray(sub.solution.transport_matrix, dtype=np.float64)
        cell_transition = tp.cell_transition(
            t_src,
            t_tgt,
            {LEIDEN_KEY: clusters},
            {LEIDEN_KEY: clusters},
            forward=True,
            key_added=None,
        )
        cell_transition.to_csv(penalty_dir / f"cell_transition_{LEIDEN_KEY}_{t_src}_to_{t_tgt}.csv")

        tgt_theta = sub.adata_tgt.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32)
        mapped_feat, valid = _barycentric_pull(
            sub,
            np.column_stack([np.cos(tgt_theta), np.sin(tgt_theta)]).astype(np.float32),
        )
        src_ap_ml = sub.adata_src.obs[[AP_KEY, ML_KEY]].to_numpy(dtype=np.float32)
        tgt_ap_ml = sub.adata_tgt.obs[[AP_KEY, ML_KEY]].to_numpy(dtype=np.float32)
        src_pca = np.asarray(sub.adata_src.obsm[JOINT_ATTR][:, :effective_pca_dims], dtype=np.float32)
        tgt_pca = np.asarray(sub.adata_tgt.obsm[JOINT_ATTR][:, :effective_pca_dims], dtype=np.float32)
        src_leiden = sub.adata_src.obs[LEIDEN_KEY].astype(str).to_numpy()
        tgt_leiden = sub.adata_tgt.obs[LEIDEN_KEY].astype(str).to_numpy()
        mapped_ap_ml, spatial_valid = _barycentric_pull(sub, tgt_ap_ml)
        src_theta = sub.adata_src.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32)
        mapped_theta = np.mod(np.arctan2(mapped_feat[:, 1], mapped_feat[:, 0]), 2.0 * np.pi)
        signed_delta = (mapped_theta - src_theta + np.pi) % (2.0 * np.pi) - np.pi
        backward = np.clip(-signed_delta[valid], a_min=0.0, a_max=None)
        ap_ml_displacement = np.linalg.norm(mapped_ap_ml - src_ap_ml, axis=1)
        mean_plan_ap_ml_dist = np.sqrt(_expected_sq_distance_from_plan(plan, src_ap_ml.astype(float), tgt_ap_ml.astype(float)))
        mean_plan_pca_dist = np.sqrt(_expected_sq_distance_from_plan(plan, src_pca.astype(float), tgt_pca.astype(float)))
        same_leiden_frac = _same_label_mass_frac(plan, src_leiden, tgt_leiden)
        valid_signed_delta = signed_delta[valid]
        valid_ap_ml_displacement = ap_ml_displacement[spatial_valid]

        summary_rows.append(
            {
                "backward_penalty": float(BACKWARD_TRICYCLE_PENALTY),
                "ap_ml_penalty": float(penalty),
                "source_time": int(t_src),
                "target_time": int(t_tgt),
                "n_source": int(sub.adata_src.n_obs),
                "n_target": int(sub.adata_tgt.n_obs),
                "valid_fraction": float(valid.mean()),
                "spatial_valid_fraction": float(spatial_valid.mean()),
                "mean_signed_delta": float(np.nanmean(signed_delta)),
                "backward_fraction": float(np.mean(valid_signed_delta < 0)) if len(valid_signed_delta) else float("nan"),
                "mean_backward_magnitude": float(backward.mean()) if len(backward) else 0.0,
                "p95_abs_delta": _safe_quantile(np.abs(valid_signed_delta), 0.95),
                "mean_barycentric_ap_ml_displacement_um": _safe_mean(valid_ap_ml_displacement),
                "p95_barycentric_ap_ml_displacement_um": _safe_quantile(valid_ap_ml_displacement, 0.95),
                "mean_plan_ap_ml_dist_um": float(mean_plan_ap_ml_dist),
                "pca_dims_used": int(effective_pca_dims),
                "mean_plan_pca_dist": float(mean_plan_pca_dist),
                "same_leiden_frac": float(same_leiden_frac),
            }
        )

summary = pd.DataFrame(summary_rows).sort_values(["ap_ml_penalty", "source_time"])
summary_path = OUTDIR / "ap_ml_sweep_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"wrote {summary_path}")

aggregate = (
    summary.groupby("ap_ml_penalty", as_index=False)
    .agg(
        mean_backward_fraction=("backward_fraction", "mean"),
        mean_backward_magnitude=("mean_backward_magnitude", "mean"),
        mean_plan_ap_ml_dist_um=("mean_plan_ap_ml_dist_um", "mean"),
        pca_dims_used=("pca_dims_used", "max"),
        mean_plan_pca_dist=("mean_plan_pca_dist", "mean"),
        mean_same_leiden_frac=("same_leiden_frac", "mean"),
        mean_barycentric_ap_ml_displacement_um=("mean_barycentric_ap_ml_displacement_um", "mean"),
        p95_barycentric_ap_ml_displacement_um=("p95_barycentric_ap_ml_displacement_um", "mean"),
        mean_valid_fraction=("valid_fraction", "mean"),
    )
    .sort_values("ap_ml_penalty")
)

def _normalize_low_is_good(x: pd.Series) -> pd.Series:
    xmin = float(x.min())
    xmax = float(x.max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - xmin) / (xmax - xmin)


aggregate["spatial_score"] = _normalize_low_is_good(aggregate["mean_plan_ap_ml_dist_um"])
aggregate["backward_score"] = _normalize_low_is_good(aggregate["mean_backward_fraction"])
aggregate["pca_score"] = _normalize_low_is_good(aggregate["mean_plan_pca_dist"])
aggregate["leiden_score"] = _normalize_low_is_good(aggregate["mean_same_leiden_frac"].max() - aggregate["mean_same_leiden_frac"])
aggregate["selection_score"] = np.sqrt(
    aggregate["spatial_score"] ** 2
    + aggregate["backward_score"] ** 2
    + aggregate["pca_score"] ** 2
    + aggregate["leiden_score"] ** 2
)
best_idx = aggregate["selection_score"].fillna(float("inf")).idxmin()
best_row = aggregate.loc[[best_idx]].copy()

aggregate_path = OUTDIR / "ap_ml_sweep_aggregate.csv"
aggregate.to_csv(aggregate_path, index=False)
print(f"wrote {aggregate_path}")

best_path = OUTDIR / "ap_ml_penalty_recommendation.csv"
best_row.to_csv(best_path, index=False)
print(f"wrote {best_path}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for (src, tgt), subdf in summary.groupby(["source_time", "target_time"]):
    ax.plot(subdf["ap_ml_penalty"], subdf["backward_fraction"], marker="o", label=f"{src} -> {tgt}")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel("AP/ML displacement penalty")
ax.set_ylabel("Backward mapped fraction")
ax.set_title("Backward mapped fraction across AP/ML penalty sweep")
ax.legend()
fig.tight_layout()
plot_path = OUTDIR / "ap_ml_vs_backward_fraction.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"wrote {plot_path}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for (src, tgt), subdf in summary.groupby(["source_time", "target_time"]):
    ax.plot(subdf["ap_ml_penalty"], subdf["mean_plan_ap_ml_dist_um"], marker="o", label=f"{src} -> {tgt}")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel("AP/ML displacement penalty")
ax.set_ylabel("Mean AP/ML transport distance (um)")
ax.set_title("Mean AP/ML transport distance across penalty sweep")
ax.legend()
fig.tight_layout()
plot_path = OUTDIR / "ap_ml_vs_mean_plan_distance.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"wrote {plot_path}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for (src, tgt), subdf in summary.groupby(["source_time", "target_time"]):
    ax.plot(subdf["ap_ml_penalty"], subdf["mean_plan_pca_dist"], marker="o", label=f"{src} -> {tgt}")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel("AP/ML displacement penalty")
ax.set_ylabel(f"Mean PCA transport distance ({effective_pca_dims}D)")
ax.set_title("Mean PCA transport distance across penalty sweep")
ax.legend()
fig.tight_layout()
plot_path = OUTDIR / f"ap_ml_vs_mean_pca_distance_{effective_pca_dims}d.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"wrote {plot_path}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for (src, tgt), subdf in summary.groupby(["source_time", "target_time"]):
    ax.plot(subdf["ap_ml_penalty"], subdf["same_leiden_frac"], marker="o", label=f"{src} -> {tgt}")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel("AP/ML displacement penalty")
ax.set_ylabel("Same-Leiden mass fraction")
ax.set_title("Same-Leiden mass fraction across penalty sweep")
ax.legend()
fig.tight_layout()
plot_path = OUTDIR / "ap_ml_vs_same_leiden_frac.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"wrote {plot_path}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(aggregate["ap_ml_penalty"], aggregate["selection_score"], marker="o", color="#222222")
ax.axvline(float(best_row["ap_ml_penalty"].iloc[0]), color="#b2182b", linestyle="--", linewidth=1.5)
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel("AP/ML displacement penalty")
ax.set_ylabel("Selection score")
ax.set_title("AP/ML penalty selection score")
fig.tight_layout()
plot_path = OUTDIR / "ap_ml_selection_score.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"wrote {plot_path}")
