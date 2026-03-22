from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from fishtools.brdu.rendering import (
    barycentric_density_image,
    density_image,
    heatmap_intensity,
    nearest_display_positions,
    subsample_indices,
)
from fishtools.brdu.sampling import proportional_sample_sizes
from fishtools.brdu.temporal_order import assign_temporal_order_from_brdu_edu
from fishtools.brdu.transport_cost import pairwise_sqeuclidean_with_backward_tricycle_penalty
from fishtools.brdu.vector_field import (
    kernel_mode_endpoint_field,
    support_adaptive_quiver_mask,
    wrapped_theta_difference,
)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch


BRDU_KEY = "brdu_pos"
EDU_KEY = "edu_pos"
TIME_KEY = "time"
JOINT_ATTR = "X_pca"
TRICYCLE_KEY = "tricycle"
RVZ_KEY = "r_vz"
AP_KEY = "ap"
ML_KEY = "ml"

BACKWARD_TRICYCLE_PENALTY = 30.0
AP_ML_DISPLACEMENT_PENALTY = 10.0
EPSILON = 1e-3
TAU_A = 0.95
TAU_B = 0.95
MAX_ITERATIONS = 200_000
FRAMES_PER_STAGE = 24
FORWARD_DISPLAY_RATIO = 3.0
THETA_BINS = 144
RVZ_BINS = 120
VECTOR_FIELD_THETA_BINS = 48
VECTOR_FIELD_RVZ_BINS = 40

DEFAULT_RESULTS_DIR = Path("/home/chaichontat/fishtools2/results/moscot_temporal_problem_brdu_edu_n5000")
DEFAULT_VZ_H5AD = Path("/home/chaichontat/nvme/vz.h5ad")
RANDOM_SEED = 0


def forward_delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.mod(target - source, 2.0 * np.pi)


def display_delta(target: np.ndarray, source: np.ndarray, *, forward_ratio: float = FORWARD_DISPLAY_RATIO) -> np.ndarray:
    """Prefer the wrapped forward arc when it is not much longer than the backward arc."""

    forward = forward_delta(target, source)
    backward = np.mod(source - target, 2.0 * np.pi)
    use_forward = forward <= forward_ratio * backward
    return np.where(use_forward, forward, -backward)

def barycentric_pull_from_transport(transport: np.ndarray, target_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numer = np.asarray(transport @ target_feat, dtype=np.float32)
    denom = np.asarray(transport.sum(axis=1, keepdims=True), dtype=np.float32)
    valid = np.isfinite(denom[:, 0]) & (denom[:, 0] > 0)
    mapped = np.full_like(numer, np.nan, dtype=np.float32)
    mapped[valid] = numer[valid] / denom[valid]
    return mapped, valid


def pair_cache_path(*, results_dir: Path, t_src: int, t_tgt: int) -> Path:
    return results_dir / "tricycle_r_vz_transitions" / f"ot_pair_{t_src}_to_{t_tgt}.npz"


def _sample_per_timepoint(adata: AnnData, *, sample_n_per_time: int, random_seed: int) -> AnnData:
    rng = np.random.default_rng(random_seed)
    chosen = []
    for timepoint in (0, 1, 2):
        idx = np.flatnonzero(adata.obs[TIME_KEY].to_numpy() == timepoint)
        take = min(int(sample_n_per_time), int(idx.size))
        chosen.append(idx if take == idx.size else rng.choice(idx, size=take, replace=False))
    return adata[np.sort(np.concatenate(chosen))].copy()


def _sample_total_preserving_ratios(adata: AnnData, *, sample_total: int, random_seed: int) -> AnnData:
    rng = np.random.default_rng(random_seed)
    time = adata.obs[TIME_KEY].to_numpy(dtype=int)
    counts = np.array([(time == t).sum() for t in (0, 1, 2)], dtype=int)
    alloc = proportional_sample_sizes(counts=counts, total=sample_total)

    chosen = []
    for timepoint, take in zip((0, 1, 2), alloc, strict=True):
        idx = np.flatnonzero(time == timepoint)
        chosen.append(idx if take == idx.size else rng.choice(idx, size=int(take), replace=False))
    return adata[np.sort(np.concatenate(chosen))].copy()


def _filter_finite_ap_ml(adata: AnnData) -> AnnData:
    finite = (
        np.isfinite(adata.obs[AP_KEY].to_numpy(dtype=float))
        & np.isfinite(adata.obs[ML_KEY].to_numpy(dtype=float))
    )
    removed = int((~finite).sum())
    if removed:
        print(f"Filtering out {removed} labeled cells with non-finite `{AP_KEY}`/`{ML_KEY}` before AP/ML-penalized OT.")
    return adata[finite].copy()


def load_all_labeled_cells(
    *,
    infile: Path,
    sample_n_per_time: int | None = None,
    sample_total: int | None = None,
    require_finite_ap_ml: bool = False,
) -> AnnData:
    adata_b = read_h5ad(infile, backed="r")
    obs = adata_b.obs
    brdu = obs[BRDU_KEY].to_numpy(dtype=bool)
    edu = obs[EDU_KEY].to_numpy(dtype=bool)
    labeled = brdu | edu

    adata = adata_b[np.flatnonzero(labeled)].to_memory()
    adata.obs_names_make_unique()

    time = assign_temporal_order_from_brdu_edu(
        brdu_pos=adata.obs[BRDU_KEY].to_numpy(dtype=bool),
        edu_pos=adata.obs[EDU_KEY].to_numpy(dtype=bool),
    )
    if np.any(time < 0):
        raise ValueError("Found double-negative cells after labeled-cell filtering.")

    adata.obs[TIME_KEY] = pd.Categorical(time, categories=[0, 1, 2], ordered=True)
    adata = adata[adata.obs.sort_values(TIME_KEY).index].copy()
    if require_finite_ap_ml:
        adata = _filter_finite_ap_ml(adata)
    if sample_total is not None:
        adata = _sample_total_preserving_ratios(adata, sample_total=sample_total, random_seed=RANDOM_SEED)
    if sample_n_per_time is not None:
        adata = _sample_per_timepoint(adata, sample_n_per_time=sample_n_per_time, random_seed=RANDOM_SEED)
    return adata


def solve_pair_all_cells(
    adata: AnnData,
    *,
    t_src: int,
    t_tgt: int,
    backward_tricycle_penalty_weight: float,
    ap_ml_penalty_weight: float,
    tau_a: float,
    tau_b: float,
) -> dict[str, np.ndarray]:
    from moscot.problems.generic import SinkhornProblem

    pair = adata[adata.obs[TIME_KEY].isin([t_src, t_tgt])].copy()
    pair.obs[TIME_KEY] = pair.obs[TIME_KEY].cat.remove_unused_categories()

    problem = SinkhornProblem(pair).prepare(key=TIME_KEY, joint_attr=JOINT_ATTR, policy="sequential")
    sub = problem[t_src, t_tgt]
    cost = pairwise_sqeuclidean_with_backward_tricycle_penalty(
        src_features=np.asarray(sub.adata_src.obsm[JOINT_ATTR], dtype=np.float32),
        tgt_features=np.asarray(sub.adata_tgt.obsm[JOINT_ATTR], dtype=np.float32),
        src_tricycle=sub.adata_src.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32),
        tgt_tricycle=sub.adata_tgt.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32),
        backward_penalty_weight=backward_tricycle_penalty_weight,
        src_ap=sub.adata_src.obs[AP_KEY].to_numpy(dtype=np.float32),
        tgt_ap=sub.adata_tgt.obs[AP_KEY].to_numpy(dtype=np.float32),
        src_ml=sub.adata_src.obs[ML_KEY].to_numpy(dtype=np.float32),
        tgt_ml=sub.adata_tgt.obs[ML_KEY].to_numpy(dtype=np.float32),
        ap_ml_penalty_weight=ap_ml_penalty_weight,
    )
    sub.set_xy(
        pd.DataFrame(cost, index=sub.adata_src.obs_names, columns=sub.adata_tgt.obs_names, copy=False),
        tag="cost_matrix",
    )
    del cost
    gc.collect()

    problem = problem.solve(
        epsilon=EPSILON,
        tau_a=tau_a,
        tau_b=tau_b,
        scale_cost="mean",
        max_iterations=MAX_ITERATIONS,
    )
    sub = problem[t_src, t_tgt]
    transport = np.asarray(sub.solution.transport_matrix, dtype=np.float32)
    src_mass = np.asarray(transport.sum(axis=1), dtype=np.float32)
    tgt_mass = np.asarray(transport.sum(axis=0), dtype=np.float32)

    src_theta = sub.adata_src.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32)
    src_rvz = sub.adata_src.obs[RVZ_KEY].to_numpy(dtype=np.float32)
    tgt_theta = sub.adata_tgt.obs[TRICYCLE_KEY].to_numpy(dtype=np.float32)
    tgt_rvz = sub.adata_tgt.obs[RVZ_KEY].to_numpy(dtype=np.float32)
    target_feat = np.column_stack([np.cos(tgt_theta), np.sin(tgt_theta), tgt_rvz]).astype(np.float32, copy=False)
    mapped_feat, valid = barycentric_pull_from_transport(transport, target_feat)
    mapped_theta = np.mod(np.arctan2(mapped_feat[:, 1], mapped_feat[:, 0]), 2.0 * np.pi).astype(np.float32, copy=False)
    mapped_theta_display = (src_theta + display_delta(mapped_theta, src_theta)).astype(np.float32, copy=False)
    mapped_rvz = mapped_feat[:, 2].astype(np.float32, copy=False)

    del transport, problem, sub, pair
    gc.collect()
    return {
        "src_theta": src_theta,
        "src_rvz": src_rvz,
        "src_mass": src_mass,
        "tgt_theta": tgt_theta,
        "tgt_rvz": tgt_rvz,
        "tgt_mass": tgt_mass,
        "mapped_theta_display": mapped_theta_display,
        "mapped_rvz": mapped_rvz,
        "mapping_valid": valid.astype(bool, copy=False),
    }


def solve_cached_pairs(
    *,
    infile: Path,
    results_dir: Path,
    sample_n_per_time: int | None,
    sample_total: int | None,
    backward_tricycle_penalty_weight: float,
    ap_ml_penalty_weight: float,
    tau_a: float,
    tau_b: float,
) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)

    adata = load_all_labeled_cells(
        infile=infile,
        sample_n_per_time=sample_n_per_time,
        sample_total=sample_total,
        require_finite_ap_ml=ap_ml_penalty_weight > 0,
    )
    print(f"all_labeled_shape={adata.shape}")
    print(adata.obs[TIME_KEY].value_counts().sort_index().to_string())
    print(
        "using penalties: "
        f"backward_tricycle={backward_tricycle_penalty_weight}, "
        f"ap_ml={ap_ml_penalty_weight}; "
        f"taus=({tau_a}, {tau_b})"
    )

    for t_src, t_tgt in ((0, 1), (1, 2)):
        solved = solve_pair_all_cells(
            adata,
            t_src=t_src,
            t_tgt=t_tgt,
            backward_tricycle_penalty_weight=backward_tricycle_penalty_weight,
            ap_ml_penalty_weight=ap_ml_penalty_weight,
            tau_a=tau_a,
            tau_b=tau_b,
        )
        cache_path = pair_cache_path(results_dir=results_dir, t_src=t_src, t_tgt=t_tgt)
        np.savez_compressed(
            cache_path,
            backward_tricycle_penalty_weight=np.float32(backward_tricycle_penalty_weight),
            ap_ml_penalty_weight=np.float32(ap_ml_penalty_weight),
            tau_a=np.float32(tau_a),
            tau_b=np.float32(tau_b),
            **solved,
        )
        print(f"wrote {cache_path}")


def load_cached_pairs(
    *,
    results_dir: Path,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    solved_pairs = []
    for t_src, t_tgt in ((0, 1), (1, 2)):
        cache_path = pair_cache_path(results_dir=results_dir, t_src=t_src, t_tgt=t_tgt)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing cached OT solve at {cache_path}. Run `--mode solve` first."
            )
        with np.load(cache_path) as data:
            solved_pairs.append((f"{t_src} -> {t_tgt}", {k: data[k] for k in data.files}))
    return solved_pairs


def load_path_sim_tricycle_paths(*, results_dir: Path) -> dict[str, np.ndarray]:
    solved_pairs = load_cached_pairs(results_dir=results_dir)
    if len(solved_pairs) != 2:
        raise ValueError(f"Expected exactly 2 cached adjacent pairs, found {len(solved_pairs)}.")

    solved_0_1 = solved_pairs[0][1]
    solved_1_2 = solved_pairs[1][1]
    if not np.allclose(solved_0_1["tgt_theta"], solved_1_2["src_theta"], equal_nan=True):
        raise ValueError("Cached 0->1 target cells do not align with cached 1->2 source cells.")
    if not np.allclose(solved_0_1["tgt_rvz"], solved_1_2["src_rvz"], equal_nan=True):
        raise ValueError("Cached 0->1 target r_vz values do not align with cached 1->2 source cells.")

    valid_0_1 = solved_0_1["mapping_valid"].astype(bool)
    target_n = int(valid_0_1.sum())
    keep_0_1 = subsample_indices(n_obs=target_n, target_n=target_n, random_seed=RANDOM_SEED)

    src0_theta = solved_0_1["src_theta"][valid_0_1][keep_0_1]
    src0_rvz = solved_0_1["src_rvz"][valid_0_1][keep_0_1]
    mapped01_theta = solved_0_1["mapped_theta_display"][valid_0_1][keep_0_1]
    mapped01_rvz = solved_0_1["mapped_rvz"][valid_0_1][keep_0_1]
    mid_idx, mid_theta, mid_rvz = nearest_display_positions(
        query_theta=mapped01_theta,
        query_rvz=mapped01_rvz,
        target_theta=solved_0_1["tgt_theta"],
        target_rvz=solved_0_1["tgt_rvz"],
    )

    mapped12_theta = solved_1_2["mapped_theta_display"][mid_idx]
    mapped12_rvz = solved_1_2["mapped_rvz"][mid_idx]
    shift_mid = mid_theta - solved_1_2["src_theta"][mid_idx]
    mapped12_theta = mapped12_theta + shift_mid

    final_idx, final_theta, final_rvz = nearest_display_positions(
        query_theta=mapped12_theta,
        query_rvz=mapped12_rvz,
        target_theta=solved_1_2["tgt_theta"],
        target_rvz=solved_1_2["tgt_rvz"],
    )
    return {
        "src0_theta": src0_theta,
        "src0_rvz": src0_rvz,
        "mid_theta": mid_theta,
        "mid_rvz": mid_rvz,
        "final_theta": final_theta,
        "final_rvz": final_rvz,
        "mid_idx": mid_idx,
        "final_idx": final_idx,
        "valid_0_1": valid_0_1,
        "keep_0_1": keep_0_1,
        "ap_ml_penalty_weight": solved_0_1["ap_ml_penalty_weight"],
    }


def render_barycentric_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    solved_pairs = load_cached_pairs(results_dir=results_dir)
    target_n = int(solved_pairs[0][1]["mapping_valid"].astype(bool).sum())

    all_plot_xy = []
    for pair_index, (_, solved) in enumerate(solved_pairs):
        valid = solved["mapping_valid"].astype(bool)
        keep = subsample_indices(n_obs=int(valid.sum()), target_n=target_n, random_seed=RANDOM_SEED + pair_index)
        all_plot_xy.extend(
            [
                np.column_stack([solved["src_theta"][valid][keep], solved["src_rvz"][valid][keep]]),
                np.column_stack([solved["mapped_theta_display"][valid][keep], solved["mapped_rvz"][valid][keep]]),
            ]
        )

    stack = np.vstack(all_plot_xy)
    ymin = float(np.nanmin(stack[:, 1]))
    ymax = float(np.nanmax(stack[:, 1]))
    ypad = 0.05 * max(1e-6, ymax - ymin)
    ymin -= ypad
    ymax += ypad

    frame_path = outdir / "_barycentric_frame.png"
    frames: list[Image.Image] = []

    for pair_index, (label, solved) in enumerate(solved_pairs):
        valid = solved["mapping_valid"].astype(bool)
        keep = subsample_indices(n_obs=int(valid.sum()), target_n=target_n, random_seed=RANDOM_SEED + pair_index)
        src_theta = solved["src_theta"][valid][keep].astype(float, copy=False)
        src_rvz = solved["src_rvz"][valid][keep].astype(float, copy=False)
        mapped_theta = solved["mapped_theta_display"][valid][keep].astype(float, copy=False)
        mapped_rvz = solved["mapped_rvz"][valid][keep].astype(float, copy=False)
        theta_delta = mapped_theta - src_theta
        rvz_delta = mapped_rvz - src_rvz
        displacement = np.sqrt(theta_delta * theta_delta + rvz_delta * rvz_delta)
        vmax = float(np.quantile(displacement, 0.98)) if len(displacement) else 1.0

        for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
            pos_theta = src_theta + alpha * theta_delta
            pos_rvz = src_rvz + alpha * rvz_delta

            fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
            ax.scatter(mapped_theta, mapped_rvz, s=1, c="#d9d9d9", alpha=0.12, linewidths=0, rasterized=True)
            ax.scatter(
                pos_theta,
                pos_rvz,
                c=displacement,
                s=2,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
            )
            sc = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="viridis")
            ax.set_xlim(0.0, 2.0 * np.pi)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("tricycle")
            ax.set_ylabel("r_vz")
            ax.set_title(f"Barycentric OT transition in tricycle/r_vz {label}  t={alpha:0.2f}")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Tricycle/r_vz displacement")
            fig.tight_layout()
            fig.savefig(frame_path)
            plt.close(fig)
            with Image.open(frame_path) as image:
                frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        raise RuntimeError("No barycentric frames generated.")

    gif_path = outdir / "barycentric_tricycle_r_vz_transitions.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    frame_path.unlink(missing_ok=True)
    print(f"wrote {gif_path}")


def render_barycentric_heatmap_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    solved_pairs = load_cached_pairs(results_dir=results_dir)
    target_n = int(solved_pairs[0][1]["mapping_valid"].astype(bool).sum())

    rvz_values: list[np.ndarray] = []
    for pair_index, (_, solved) in enumerate(solved_pairs):
        valid = solved["mapping_valid"].astype(bool)
        keep = subsample_indices(n_obs=int(valid.sum()), target_n=target_n, random_seed=RANDOM_SEED + pair_index)
        rvz_values.extend([solved["src_rvz"][valid][keep], solved["mapped_rvz"][valid][keep]])

    all_rvz = np.concatenate(rvz_values)
    rvz_min = float(np.nanmin(all_rvz))
    rvz_max = float(np.nanmax(all_rvz))
    pad = 0.05 * max(rvz_max - rvz_min, 1e-6)
    rvz_edges = np.linspace(rvz_min - pad, rvz_max + pad, RVZ_BINS + 1)

    frame_path = outdir / "_barycentric_heatmap_frame.png"
    frames: list[Image.Image] = []

    for pair_index, (label, solved) in enumerate(solved_pairs):
        valid = solved["mapping_valid"].astype(bool)
        keep = subsample_indices(n_obs=int(valid.sum()), target_n=target_n, random_seed=RANDOM_SEED + pair_index)
        src_theta = solved["src_theta"][valid][keep]
        src_rvz = solved["src_rvz"][valid][keep]
        mapped_theta = solved["mapped_theta_display"][valid][keep]
        mapped_rvz = solved["mapped_rvz"][valid][keep]

        src_density = density_image(
            src_theta,
            src_rvz,
            rvz_edges=rvz_edges,
            theta_bins=THETA_BINS,
            wrap_theta=False,
        )
        mapped_density = density_image(
            mapped_theta,
            mapped_rvz,
            rvz_edges=rvz_edges,
            theta_bins=THETA_BINS,
            wrap_theta=False,
        )

        panel_stack = np.concatenate([src_density.ravel(), mapped_density.ravel()])
        log_vmax = float(np.quantile(np.log1p(panel_stack), 0.995))
        log_vmax = max(log_vmax, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        panels = [
            ("Source cells", src_density),
            ("Barycentric-mapped cells", mapped_density),
        ]
        for ax, (title, image) in zip(axes[:3], panels, strict=True):
            im = ax.imshow(
                np.log1p(image),
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
                cmap="magma",
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xlabel("tricycle")
        axes[0].set_ylabel("r_vz")
        fig.suptitle(f"Barycentric OT density summary: {label}")
        cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("log1p(cell count)")
        fig.tight_layout()
        png_path = outdir / f"barycentric_heatmap_tricycle_r_vz_{label.replace(' -> ', '_to_')}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"wrote {png_path}")

        for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
            blended = barycentric_density_image(
                src_theta=src_theta,
                src_rvz=src_rvz,
                mapped_theta=mapped_theta,
                mapped_rvz=mapped_rvz,
                alpha=float(alpha),
                rvz_edges=rvz_edges,
                theta_bins=THETA_BINS,
                wrap_theta=False,
            )
            fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
            im = ax.imshow(
                np.log1p(blended),
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
                cmap="magma",
                interpolation="nearest",
                vmin=0.0,
                vmax=log_vmax,
            )
            ax.set_xlabel("tricycle")
            ax.set_ylabel("r_vz")
            ax.set_title(f"Barycentric OT heatmap in tricycle/r_vz: {label}  t={alpha:0.2f}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("log1p(cell count)")
            fig.tight_layout()
            fig.savefig(frame_path)
            plt.close(fig)
            with Image.open(frame_path) as image:
                frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        raise RuntimeError("No barycentric heatmap frames generated.")

    gif_path = outdir / "barycentric_heatmap_tricycle_r_vz_transitions.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    frame_path.unlink(missing_ok=True)
    print(f"wrote {gif_path}")


def _mass_to_equivalent_cell_count(weights: np.ndarray, *, total_cells: float) -> np.ndarray:
    """Scale OT mass weights into a chosen total-cell normalization for readable heatmaps."""

    weights_arr = np.asarray(weights, dtype=np.float32)
    total_mass = float(weights_arr.sum())
    if total_mass <= 0:
        raise ValueError("Expected strictly positive OT mass when rendering density summaries.")
    if total_cells <= 0:
        raise ValueError(f"`total_cells` must be positive, found {total_cells}.")
    return weights_arr * np.float32(total_cells / total_mass)


def render_ot_heatmap_summary_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    solved_pairs = load_cached_pairs(results_dir=results_dir)

    rvz_values = []
    for _, solved in solved_pairs:
        rvz_values.extend([solved["src_rvz"], solved["tgt_rvz"]])
    all_rvz = np.concatenate(rvz_values)
    rvz_min = float(np.nanmin(all_rvz))
    rvz_max = float(np.nanmax(all_rvz))
    rvz_pad = 0.05 * max(rvz_max - rvz_min, 1e-6)
    rvz_edges = np.linspace(rvz_min - rvz_pad, rvz_max + rvz_pad, RVZ_BINS + 1)

    summaries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    density_values = []
    delta_values = []
    for label, solved in solved_pairs:
        normalized_total = float(max(solved["src_theta"].shape[0], solved["tgt_theta"].shape[0]))
        src_density = density_image(
            solved["src_theta"],
            solved["src_rvz"],
            rvz_edges=rvz_edges,
            theta_bins=THETA_BINS,
            weights=_mass_to_equivalent_cell_count(solved["src_mass"], total_cells=normalized_total),
            wrap_theta=True,
        )
        transported_density = density_image(
            solved["tgt_theta"],
            solved["tgt_rvz"],
            rvz_edges=rvz_edges,
            theta_bins=THETA_BINS,
            weights=_mass_to_equivalent_cell_count(solved["tgt_mass"], total_cells=normalized_total),
            wrap_theta=True,
        )
        delta_density = transported_density - src_density
        summaries.append((label, src_density, transported_density, delta_density))
        density_values.extend([src_density.ravel(), transported_density.ravel()])
        delta_values.append(delta_density.ravel())

    density_vmax = float(np.quantile(np.concatenate(density_values), 0.995))
    density_vmax = max(density_vmax, 1.0)
    delta_absmax = float(np.quantile(np.abs(np.concatenate(delta_values)), 0.995))
    delta_absmax = max(delta_absmax, 1e-3)

    for label, src_density, transported_density, delta_density in summaries:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True)
        density_panels = [("Source", src_density), ("Transported", transported_density)]
        density_im = None
        for ax, (title, image) in zip(axes[:2], density_panels, strict=True):
            density_im = ax.imshow(
                heatmap_intensity(image, log_scale=False),
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
                cmap="magma",
                interpolation="nearest",
                vmin=0.0,
                vmax=density_vmax,
            )
            ax.set_title(title)
            ax.set_xlabel("tricycle")

        delta_im = axes[2].imshow(
            delta_density,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-delta_absmax,
            vmax=delta_absmax,
        )
        axes[2].set_title("Transported - source")
        axes[2].set_xlabel("tricycle")
        axes[0].set_ylabel("r_vz")
        fig.suptitle(f"OT density summary: {label}")
        fig.colorbar(density_im, ax=axes[:2], fraction=0.03, pad=0.02, label="normalized cell count")
        fig.colorbar(delta_im, ax=axes[2], fraction=0.046, pad=0.04, label="normalized cell count delta")
        png_path = outdir / f"heatmap_tricycle_r_vz_{label.replace(' -> ', '_to_')}.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"wrote {png_path}")

    if len(summaries) != 2:
        raise ValueError(f"Expected exactly 2 adjacent OT summaries, found {len(summaries)}.")

    overlay_path = outdir / "heatmap_tricycle_r_vz_stage_overlay.png"
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    delta_0_to_1 = summaries[0][3]
    delta_1_to_2 = summaries[1][3]
    transported_0_to_1_neg = np.clip(-delta_0_to_1, 0.0, None)
    transported_0_to_1_pos = np.clip(delta_0_to_1, 0.0, None)
    transported_1_to_2_neg = np.clip(-delta_1_to_2, 0.0, None)
    transported_1_to_2_pos = np.clip(delta_1_to_2, 0.0, None)
    def _suppress_weak_signal(image: np.ndarray, *, keep_quantile: float) -> np.ndarray:
        positive = image[image > 0]
        if positive.size == 0:
            return image
        threshold = float(np.quantile(positive, keep_quantile))
        return np.where(image >= threshold, image, 0.0)

    transported_0_to_1_neg = _suppress_weak_signal(transported_0_to_1_neg, keep_quantile=0.65)
    transported_0_to_1_pos = _suppress_weak_signal(transported_0_to_1_pos, keep_quantile=0.65)
    transported_1_to_2_neg = _suppress_weak_signal(transported_1_to_2_neg, keep_quantile=0.65)
    transported_1_to_2_pos = _suppress_weak_signal(transported_1_to_2_pos, keep_quantile=0.65)
    vmax_0_to_1_neg = max(float(np.quantile(transported_0_to_1_neg, 0.995)), 1e-6)
    vmax_0_to_1_pos = max(float(np.quantile(transported_0_to_1_pos, 0.995)), 1e-6)
    vmax_1_to_2_neg = max(float(np.quantile(transported_1_to_2_neg, 0.995)), 1e-6)
    vmax_1_to_2_pos = max(float(np.quantile(transported_1_to_2_pos, 0.995)), 1e-6)
    alpha_0_to_1_neg = 0.85 * np.power(np.clip(transported_0_to_1_neg / vmax_0_to_1_neg, 0.0, 1.0), 0.8)
    alpha_0_to_1_pos = 0.85 * np.power(np.clip(transported_0_to_1_pos / vmax_0_to_1_pos, 0.0, 1.0), 0.8)
    alpha_1_to_2_neg = 0.85 * np.power(np.clip(transported_1_to_2_neg / vmax_1_to_2_neg, 0.0, 1.0), 0.8)
    alpha_1_to_2_pos = 0.85 * np.power(np.clip(transported_1_to_2_pos / vmax_1_to_2_pos, 0.0, 1.0), 0.8)
    ax.imshow(
        transported_0_to_1_neg,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax_0_to_1_neg,
        alpha=alpha_0_to_1_neg,
    )
    ax.imshow(
        transported_0_to_1_pos,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
        interpolation="nearest",
        cmap="Greens",
        vmin=0.0,
        vmax=vmax_0_to_1_pos,
        alpha=alpha_0_to_1_pos,
    )
    ax.imshow(
        transported_1_to_2_neg,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
        interpolation="nearest",
        cmap="Greens",
        vmin=0.0,
        vmax=vmax_1_to_2_neg,
        alpha=alpha_1_to_2_neg,
    )
    ax.imshow(
        transported_1_to_2_pos,
        origin="lower",
        aspect="auto",
        extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
        interpolation="nearest",
        cmap="Reds",
        vmin=0.0,
        vmax=vmax_1_to_2_pos,
        alpha=alpha_1_to_2_pos,
    )
    ax.set_xlabel("tricycle")
    ax.set_ylabel("r_vz")
    ax.set_title("OT delta overlay")
    ax.legend(
        handles=[
            Patch(facecolor=plt.get_cmap("Blues")(0.8), edgecolor="none", label="Blues: -1 * (0 -> 1 delta < 0)"),
            Patch(facecolor=plt.get_cmap("Greens")(0.8), edgecolor="none", label="Greens: 0 -> 1 delta > 0 and 1 -> 2 delta < 0"),
            Patch(facecolor=plt.get_cmap("Reds")(0.8), edgecolor="none", label="Reds: 1 -> 2 delta > 0"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)
    print(f"wrote {overlay_path}")

    polar_overlay_path = outdir / "heatmap_tricycle_r_vz_stage_overlay_polar.png"
    theta_edges = np.linspace(0.0, 2.0 * np.pi, THETA_BINS + 1)
    radial_edges = rvz_edges - rvz_edges[0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for image, cmap, vmax in [
        (transported_0_to_1_neg, "Blues", vmax_0_to_1_neg),
        (transported_0_to_1_pos, "Greens", vmax_0_to_1_pos),
        (transported_1_to_2_neg, "Greens", vmax_1_to_2_neg),
        (transported_1_to_2_pos, "Reds", vmax_1_to_2_pos),
    ]:
        ax.pcolormesh(
            theta_edges,
            radial_edges,
            np.ma.masked_less_equal(image, 0.0),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            shading="flat",
            alpha=0.85,
        )
    tick_values = np.linspace(rvz_edges[0], rvz_edges[-1], 5)
    ax.set_rticks(tick_values - rvz_edges[0])
    ax.set_yticklabels([f"{tick:0.0f}" for tick in tick_values])
    ax.set_rlabel_position(112.5)
    ax.set_title("OT delta overlay (radial)")
    ax.legend(
        handles=[
            Patch(facecolor=plt.get_cmap("Blues")(0.8), edgecolor="none", label="Blues: -1 * (0 -> 1 delta < 0)"),
            Patch(facecolor=plt.get_cmap("Greens")(0.8), edgecolor="none", label="Greens: 0 -> 1 delta > 0 and 1 -> 2 delta < 0"),
            Patch(facecolor=plt.get_cmap("Reds")(0.8), edgecolor="none", label="Reds: 1 -> 2 delta > 0"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.25, 1.10),
        frameon=True,
    )
    fig.savefig(polar_overlay_path, dpi=200)
    plt.close(fig)
    print(f"wrote {polar_overlay_path}")

    path_data = load_path_sim_tricycle_paths(results_dir=results_dir)
    src0_theta = np.mod(path_data["src0_theta"], 2.0 * np.pi)
    src0_rvz = path_data["src0_rvz"]
    mid_theta = np.mod(path_data["mid_theta"], 2.0 * np.pi)
    mid_rvz = path_data["mid_rvz"]
    final_theta = np.mod(path_data["final_theta"], 2.0 * np.pi)
    final_rvz = path_data["final_rvz"]
    theta_grid = np.linspace(0.0, 2.0 * np.pi, VECTOR_FIELD_THETA_BINS, endpoint=False)
    rvz_grid = np.linspace(rvz_edges[0], rvz_edges[-1], VECTOR_FIELD_RVZ_BINS)
    theta_bandwidth = 2.0 * np.pi / VECTOR_FIELD_THETA_BINS * 1.5
    rvz_bandwidth = max((rvz_grid[-1] - rvz_grid[0]) / VECTOR_FIELD_RVZ_BINS * 1.5, 1e-3)
    theta_mesh, rvz_mesh = np.meshgrid(theta_grid, rvz_grid, indexing="xy")
    src_density_0_to_1 = summaries[0][1]
    transported_density_0_to_1 = summaries[0][2]
    src_density_1_to_2 = summaries[1][1]
    transported_density_1_to_2 = summaries[1][2]

    def _density_lookup(image: np.ndarray, theta: np.ndarray, rvz: np.ndarray) -> np.ndarray:
        theta_idx = np.floor(np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi) * THETA_BINS).astype(int)
        theta_idx = np.clip(theta_idx, 0, THETA_BINS - 1)
        rvz_idx = np.searchsorted(rvz_edges, rvz, side="right") - 1
        rvz_idx = np.clip(rvz_idx, 0, RVZ_BINS - 1)
        return image[rvz_idx, theta_idx]

    def _filtered_quiver_mask(
        *,
        field_weight: np.ndarray,
        src_density: np.ndarray,
        transported_density: np.ndarray,
        field_end_theta: np.ndarray,
        field_end_rvz: np.ndarray,
    ) -> np.ndarray:
        mask_local = support_adaptive_quiver_mask(support=field_weight, min_quantile=0.35, random_seed=RANDOM_SEED)
        src_mass = _density_lookup(src_density, theta_mesh, rvz_mesh)
        end_mass = _density_lookup(transported_density, field_end_theta, field_end_rvz)
        src_positive = src_density[src_density > 0]
        end_positive = transported_density[transported_density > 0]
        src_threshold = float(np.quantile(src_positive, 0.9)) if src_positive.size else 0.0
        end_threshold = float(np.quantile(end_positive, 0.9)) if end_positive.size else 0.0
        mask_local &= src_mass >= src_threshold
        mask_local &= end_mass >= end_threshold
        row_idx, col_idx = np.indices(mask_local.shape)
        mask_local &= ((row_idx + col_idx) % 3) == 0
        return mask_local

    def _render_overlay_quiver(
        *,
        out_path: Path,
        title: str,
        quivers: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]],
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        for image, cmap, vmax in [
            (transported_0_to_1_neg, "Blues", vmax_0_to_1_neg),
            (transported_0_to_1_pos, "Greens", vmax_0_to_1_pos),
            (transported_1_to_2_neg, "Greens", vmax_1_to_2_neg),
            (transported_1_to_2_pos, "Reds", vmax_1_to_2_pos),
        ]:
            ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                alpha=0.85 * np.power(np.clip(image / max(vmax, 1e-6), 0.0, 1.0), 0.8),
            )
        legend_handles = [
            Patch(facecolor=plt.get_cmap("Blues")(0.8), edgecolor="none", label="Blues: -1 * (0 -> 1 delta < 0)"),
            Patch(facecolor=plt.get_cmap("Greens")(0.8), edgecolor="none", label="Greens: 0 -> 1 delta > 0 and 1 -> 2 delta < 0"),
            Patch(facecolor=plt.get_cmap("Reds")(0.8), edgecolor="none", label="Reds: 1 -> 2 delta > 0"),
        ]
        for field_dtheta, field_drvz, mask, color, label in quivers:
            q = ax.quiver(
                theta_mesh[mask],
                rvz_mesh[mask],
                field_dtheta[mask],
                field_drvz[mask],
                color=color,
                alpha=0.6,
                angles="xy",
                scale_units="xy",
                scale=4.0,
                width=0.005,
            )
            q.set_path_effects([pe.Stroke(linewidth=1.0, foreground="white"), pe.Normal()])
            legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_ylim(0.0, 80.0)
        ax.set_title(title)
        ax.legend(handles=legend_handles, loc="upper right", frameon=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"wrote {out_path}")

    field_end_theta, field_end_rvz, field_weight = kernel_mode_endpoint_field(
        src_theta=src0_theta,
        src_rvz=src0_rvz,
        end_theta=mid_theta,
        end_rvz=mid_rvz,
        theta_grid=theta_grid,
        rvz_grid=rvz_grid,
        theta_bandwidth=theta_bandwidth,
        rvz_bandwidth=rvz_bandwidth,
        endpoint_theta_bandwidth=theta_bandwidth,
        endpoint_rvz_bandwidth=rvz_bandwidth,
    )
    field_dtheta = wrapped_theta_difference(source_theta=theta_mesh, target_theta=field_end_theta)
    field_drvz = field_end_rvz - rvz_mesh
    mask = _filtered_quiver_mask(
        field_weight=field_weight,
        src_density=src_density_0_to_1,
        transported_density=transported_density_0_to_1,
        field_end_theta=field_end_theta,
        field_end_rvz=field_end_rvz,
    )
    _render_overlay_quiver(
        out_path=overlay_path,
        title="OT delta overlay + quivers",
        quivers=[(field_dtheta, field_drvz, mask, "#c026d3", "0 -> 1 quiver")],
    )

    field_end_theta_1_to_2, field_end_rvz_1_to_2, field_weight_1_to_2 = kernel_mode_endpoint_field(
        src_theta=mid_theta,
        src_rvz=mid_rvz,
        end_theta=final_theta,
        end_rvz=final_rvz,
        theta_grid=theta_grid,
        rvz_grid=rvz_grid,
        theta_bandwidth=theta_bandwidth,
        rvz_bandwidth=rvz_bandwidth,
        endpoint_theta_bandwidth=theta_bandwidth,
        endpoint_rvz_bandwidth=rvz_bandwidth,
    )
    field_dtheta_1_to_2 = wrapped_theta_difference(source_theta=theta_mesh, target_theta=field_end_theta_1_to_2)
    field_drvz_1_to_2 = field_end_rvz_1_to_2 - rvz_mesh
    mask_1_to_2 = _filtered_quiver_mask(
        field_weight=field_weight_1_to_2,
        src_density=src_density_1_to_2,
        transported_density=transported_density_1_to_2,
        field_end_theta=field_end_theta_1_to_2,
        field_end_rvz=field_end_rvz_1_to_2,
    )
    _render_overlay_quiver(
        out_path=overlay_path,
        title="OT delta overlay + quivers",
        quivers=[
            (field_dtheta, field_drvz, mask, "#c026d3", "0 -> 1 quiver"),
            (field_dtheta_1_to_2, field_drvz_1_to_2, mask_1_to_2, "#ea580c", "1 -> 2 quiver"),
        ],
    )

    polar_quiver_path = outdir / "heatmap_tricycle_r_vz_stage_overlay_polar_quiver.png"
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for image, cmap, vmax in [
        (transported_0_to_1_neg, "Blues", vmax_0_to_1_neg),
        (transported_0_to_1_pos, "Greens", vmax_0_to_1_pos),
        (transported_1_to_2_neg, "Greens", vmax_1_to_2_neg),
        (transported_1_to_2_pos, "Reds", vmax_1_to_2_pos),
    ]:
        ax.pcolormesh(
            theta_edges,
            radial_edges,
            np.ma.masked_less_equal(image, 0.0),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            shading="flat",
            alpha=0.75,
        )
    radial_mesh = rvz_mesh - rvz_edges[0]
    ax.quiver(
        theta_mesh[mask],
        radial_mesh[mask],
        field_dtheta[mask],
        field_drvz[mask],
        color="#111111",
        alpha=0.55,
        scale=20.0,
        width=0.0035,
        headwidth=3.5,
        headlength=4.5,
    )
    ax.set_rticks(tick_values - rvz_edges[0])
    ax.set_yticklabels([f"{tick:0.0f}" for tick in tick_values])
    ax.set_rlabel_position(112.5)
    ax.set_title("OT delta overlay (radial + quiver)")
    ax.legend(
        handles=[
            Patch(facecolor=plt.get_cmap("Blues")(0.8), edgecolor="none", label="Blues: -1 * (0 -> 1 delta < 0)"),
            Patch(facecolor=plt.get_cmap("Greens")(0.8), edgecolor="none", label="Greens: 0 -> 1 delta > 0 and 1 -> 2 delta < 0"),
            Patch(facecolor=plt.get_cmap("Reds")(0.8), edgecolor="none", label="Reds: 1 -> 2 delta > 0"),
            Patch(facecolor="#111111", edgecolor="none", label="Combined OT quiver"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.32, 1.10),
        frameon=True,
    )
    fig.savefig(polar_quiver_path, dpi=200)
    plt.close(fig)
    print(f"wrote {polar_quiver_path}")


def render_path_sim_heatmap_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    path_data = load_path_sim_tricycle_paths(results_dir=results_dir)
    src0_theta = path_data["src0_theta"]
    src0_rvz = path_data["src0_rvz"]
    mid_theta = path_data["mid_theta"]
    mid_rvz = path_data["mid_rvz"]
    final_theta = path_data["final_theta"]
    final_rvz = path_data["final_rvz"]

    all_rvz = np.concatenate([src0_rvz, mid_rvz, final_rvz])
    rvz_min = float(np.nanmin(all_rvz))
    rvz_max = float(np.nanmax(all_rvz))
    rvz_pad = 0.05 * max(rvz_max - rvz_min, 1e-6)
    rvz_edges = np.linspace(rvz_min - rvz_pad, rvz_max + rvz_pad, RVZ_BINS + 1)

    def density(theta: np.ndarray, rvz: np.ndarray) -> np.ndarray:
        return density_image(theta, rvz, rvz_edges=rvz_edges, theta_bins=THETA_BINS, wrap_theta=True)

    src_density = density(src0_theta, src0_rvz)
    mid_density = density(mid_theta, mid_rvz)
    final_density = density(final_theta, final_rvz)
    panel_stack = np.concatenate([src_density.ravel(), mid_density.ravel(), final_density.ravel()])
    vmax = float(np.quantile(panel_stack, 0.995))
    vmax = max(vmax, 1.0)

    for png_path, left_title, left_image, right_title, right_image in [
        (
            outdir / "path_sim_heatmap_tricycle_r_vz_0_to_1.png",
            "Stage 0",
            src_density,
            "Matched stage 1",
            mid_density,
        ),
        (
            outdir / "path_sim_heatmap_tricycle_r_vz_1_to_2.png",
            "Matched stage 1",
            mid_density,
            "Matched stage 2",
            final_density,
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
        for ax, (title, image) in zip(axes, [(left_title, left_image), (right_title, right_image)], strict=True):
            im = ax.imshow(
                heatmap_intensity(image, log_scale=False),
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
                cmap="magma",
                interpolation="nearest",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.set_xlabel("tricycle")
        axes[0].set_ylabel("r_vz")
        cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("cell count")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"wrote {png_path}")

    summary_path = outdir / "path_sim_heatmap_tricycle_r_vz_summary.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (title, image) in zip(
        axes,
        [("Stage 0", src_density), ("Matched stage 1", mid_density), ("Matched stage 2", final_density)],
        strict=True,
    ):
        im = ax.imshow(
            heatmap_intensity(image, log_scale=False),
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("tricycle")
    axes[0].set_ylabel("r_vz")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("cell count")
    fig.tight_layout()
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f"wrote {summary_path}")

    frame_path = outdir / "_path_sim_heatmap_frame.png"
    frames: list[Image.Image] = []
    for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
        stage01 = density(
            src0_theta + alpha * (mid_theta - src0_theta),
            src0_rvz + alpha * (mid_rvz - src0_rvz),
        )
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        im = ax.imshow(
            heatmap_intensity(stage01, log_scale=False),
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_title(f"Path-simulated OT heatmap in tricycle/r_vz: 0 -> 1  t={alpha:0.2f}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cell count")
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        with Image.open(frame_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
        stage12 = density(
            mid_theta + alpha * (final_theta - mid_theta),
            mid_rvz + alpha * (final_rvz - mid_rvz),
        )
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        im = ax.imshow(
            heatmap_intensity(stage12, log_scale=False),
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_title(f"Path-simulated OT heatmap in tricycle/r_vz: 1 -> 2  t={alpha:0.2f}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cell count")
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        with Image.open(frame_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        raise RuntimeError("No path-simulation heatmap frames generated.")

    gif_path = outdir / "path_sim_heatmap_tricycle_r_vz_transitions.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    frame_path.unlink(missing_ok=True)
    print(f"wrote {gif_path}")


def render_path_sim_umap_from_cache(*, results_dir: Path, infile: Path) -> None:
    outdir = results_dir / "umap_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    path_data = load_path_sim_tricycle_paths(results_dir=results_dir)
    require_finite_ap_ml = float(path_data["ap_ml_penalty_weight"]) > 0
    adata = load_all_labeled_cells(infile=infile, require_finite_ap_ml=require_finite_ap_ml)
    umap_key = "X_umap"
    if umap_key not in adata.obsm:
        raise KeyError(f"Expected `{umap_key}` in `adata.obsm`, found: {list(adata.obsm.keys())}")

    time = adata.obs[TIME_KEY].to_numpy(dtype=int)
    src0_umap = np.asarray(adata.obsm[umap_key][time == 0], dtype=np.float32)[:, :2]
    mid_umap = np.asarray(adata.obsm[umap_key][time == 1], dtype=np.float32)[:, :2]
    final_umap = np.asarray(adata.obsm[umap_key][time == 2], dtype=np.float32)[:, :2]
    if src0_umap.shape[0] != path_data["valid_0_1"].shape[0]:
        raise ValueError("Loaded time-0 UMAP rows do not align with cached 0->1 source cells.")
    if mid_umap.shape[0] <= int(np.max(path_data["mid_idx"])) or final_umap.shape[0] <= int(np.max(path_data["final_idx"])):
        raise ValueError("Loaded UMAP rows do not align with cached matched path indices.")

    src0_xy = src0_umap[path_data["valid_0_1"]][path_data["keep_0_1"]]
    mid_xy = mid_umap[path_data["mid_idx"]]
    final_xy = final_umap[path_data["final_idx"]]

    stack = np.vstack([src0_xy, mid_xy, final_xy])
    xmin = float(np.nanmin(stack[:, 0]))
    xmax = float(np.nanmax(stack[:, 0]))
    ymin = float(np.nanmin(stack[:, 1]))
    ymax = float(np.nanmax(stack[:, 1]))
    xpad = 0.05 * max(xmax - xmin, 1e-6)
    ypad = 0.05 * max(ymax - ymin, 1e-6)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad
    total_displacement = np.linalg.norm(final_xy - src0_xy, axis=1)
    vmax = float(np.quantile(total_displacement, 0.98)) if len(total_displacement) else 1.0

    summary_path = outdir / "path_sim_umap_summary.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (title, xy) in zip(
        axes,
        [("Stage 0", src0_xy), ("Matched stage 1", mid_xy), ("Matched stage 2", final_xy)],
        strict=True,
    ):
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c="#1f1f1f", alpha=0.2, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f"wrote {summary_path}")

    frame_path = outdir / "_path_sim_umap_frame.png"
    frames: list[Image.Image] = []
    for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
        pos_xy = src0_xy + alpha * (mid_xy - src0_xy)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        ax.scatter(mid_xy[:, 0], mid_xy[:, 1], s=1, c="#d9d9d9", alpha=0.1, linewidths=0, rasterized=True)
        ax.scatter(
            pos_xy[:, 0],
            pos_xy[:, 1],
            c=total_displacement,
            s=2,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        sc = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="viridis")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"Path-simulated OT in UMAP: 0 -> 1  t={alpha:0.2f}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Total UMAP displacement")
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        with Image.open(frame_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    for alpha in np.linspace(0.0, 1.0, FRAMES_PER_STAGE):
        pos_xy = mid_xy + alpha * (final_xy - mid_xy)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        ax.scatter(final_xy[:, 0], final_xy[:, 1], s=1, c="#d9d9d9", alpha=0.1, linewidths=0, rasterized=True)
        ax.scatter(
            pos_xy[:, 0],
            pos_xy[:, 1],
            c=total_displacement,
            s=2,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        sc = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="viridis")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"Path-simulated OT in UMAP: 1 -> 2  t={alpha:0.2f}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Total UMAP displacement")
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        with Image.open(frame_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        raise RuntimeError("No path-simulation UMAP frames generated.")

    gif_path = outdir / "path_sim_umap_transitions.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    frame_path.unlink(missing_ok=True)
    print(f"wrote {gif_path}")


def render_path_sim_vector_field_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    path_data = load_path_sim_tricycle_paths(results_dir=results_dir)

    src0_theta = np.mod(path_data["src0_theta"], 2.0 * np.pi)
    src0_rvz = path_data["src0_rvz"]
    mid_theta = np.mod(path_data["mid_theta"], 2.0 * np.pi)
    mid_rvz = path_data["mid_rvz"]
    final_theta = np.mod(path_data["final_theta"], 2.0 * np.pi)
    final_rvz = path_data["final_rvz"]

    rvz_all = np.concatenate([src0_rvz, mid_rvz, final_rvz])
    rvz_min = float(np.nanmin(rvz_all))
    rvz_max = float(np.nanmax(rvz_all))
    rvz_pad = 0.05 * max(rvz_max - rvz_min, 1e-6)
    rvz_grid = np.linspace(rvz_min - rvz_pad, rvz_max + rvz_pad, VECTOR_FIELD_RVZ_BINS)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, VECTOR_FIELD_THETA_BINS, endpoint=False)
    theta_bandwidth = 2.0 * np.pi / VECTOR_FIELD_THETA_BINS * 1.5
    rvz_bandwidth = max((rvz_grid[-1] - rvz_grid[0]) / VECTOR_FIELD_RVZ_BINS * 1.5, 1e-3)

    for label, start_theta, start_rvz, end_theta, end_rvz in [
        ("0_to_1", src0_theta, src0_rvz, mid_theta, mid_rvz),
        ("1_to_2", mid_theta, mid_rvz, final_theta, final_rvz),
        (
            "combined",
            np.concatenate([src0_theta, mid_theta]),
            np.concatenate([src0_rvz, mid_rvz]),
            np.concatenate([mid_theta, final_theta]),
            np.concatenate([mid_rvz, final_rvz]),
        ),
    ]:
        field_end_theta, field_end_rvz, field_weight = kernel_mode_endpoint_field(
            src_theta=start_theta,
            src_rvz=start_rvz,
            end_theta=end_theta,
            end_rvz=end_rvz,
            theta_grid=theta_grid,
            rvz_grid=rvz_grid,
            theta_bandwidth=theta_bandwidth,
            rvz_bandwidth=rvz_bandwidth,
            endpoint_theta_bandwidth=theta_bandwidth,
            endpoint_rvz_bandwidth=rvz_bandwidth,
        )
        theta_mesh, rvz_mesh = np.meshgrid(theta_grid, rvz_grid, indexing="xy")
        field_dtheta = wrapped_theta_difference(source_theta=theta_mesh, target_theta=field_end_theta)
        field_drvz = field_end_rvz - rvz_mesh
        csv_path = outdir / f"path_sim_vector_field_{label}.csv"
        pd.DataFrame(
            {
                "theta": theta_mesh.ravel(),
                "r_vz": rvz_mesh.ravel(),
                "endpoint_theta": field_end_theta.ravel(),
                "endpoint_r_vz": field_end_rvz.ravel(),
                "dtheta": field_dtheta.ravel(),
                "drvz": field_drvz.ravel(),
                "kernel_weight": field_weight.ravel(),
            }
        ).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")

        speed = np.hypot(field_dtheta, field_drvz)
        mask = support_adaptive_quiver_mask(support=field_weight, min_quantile=0.1, random_seed=RANDOM_SEED)

        quiver_path = outdir / f"path_sim_vector_field_{label}_quiver.png"
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        bg = ax.imshow(
            field_weight,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_grid[0], rvz_grid[-1]],
            cmap="Greys",
            interpolation="nearest",
        )
        q = ax.quiver(
            theta_mesh[mask],
            rvz_mesh[mask],
            field_dtheta[mask],
            field_drvz[mask],
            speed[mask],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.003,
        )
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_title(f"Path-sim modal-endpoint field: {label.replace('_', ' -> ')}")
        fig.colorbar(bg, ax=ax, fraction=0.046, pad=0.04, label="kernel weight")
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.10, label="vector speed")
        fig.tight_layout()
        fig.savefig(quiver_path)
        plt.close(fig)
        print(f"wrote {quiver_path}")

        stream_path = outdir / f"path_sim_vector_field_{label}_streamplot.png"
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        bg = ax.imshow(
            field_weight,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_grid[0], rvz_grid[-1]],
            cmap="Greys",
            interpolation="nearest",
        )
        stream = ax.streamplot(theta_grid, rvz_grid, field_dtheta, field_drvz, color=speed, cmap="viridis", density=1.2)
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_title(f"Path-sim modal-endpoint streamplot: {label.replace('_', ' -> ')}")
        fig.colorbar(bg, ax=ax, fraction=0.046, pad=0.04, label="kernel weight")
        fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.10, label="vector speed")
        fig.tight_layout()
        fig.savefig(stream_path)
        plt.close(fig)
        print(f"wrote {stream_path}")


def render_path_sim_difference_quiver_from_cache(*, results_dir: Path) -> None:
    outdir = results_dir / "tricycle_r_vz_transitions"
    outdir.mkdir(parents=True, exist_ok=True)
    path_data = load_path_sim_tricycle_paths(results_dir=results_dir)

    src0_theta = np.mod(path_data["src0_theta"], 2.0 * np.pi)
    src0_rvz = path_data["src0_rvz"]
    mid_theta = np.mod(path_data["mid_theta"], 2.0 * np.pi)
    mid_rvz = path_data["mid_rvz"]
    final_theta = np.mod(path_data["final_theta"], 2.0 * np.pi)
    final_rvz = path_data["final_rvz"]

    rvz_all = np.concatenate([src0_rvz, mid_rvz, final_rvz])
    rvz_min = float(np.nanmin(rvz_all))
    rvz_max = float(np.nanmax(rvz_all))
    rvz_pad = 0.05 * max(rvz_max - rvz_min, 1e-6)
    rvz_edges = np.linspace(rvz_min - rvz_pad, rvz_max + rvz_pad, RVZ_BINS + 1)
    rvz_grid = np.linspace(rvz_edges[0], rvz_edges[-1], VECTOR_FIELD_RVZ_BINS)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, VECTOR_FIELD_THETA_BINS, endpoint=False)
    theta_bandwidth = 2.0 * np.pi / VECTOR_FIELD_THETA_BINS * 1.5
    rvz_bandwidth = max((rvz_grid[-1] - rvz_grid[0]) / VECTOR_FIELD_RVZ_BINS * 1.5, 1e-3)
    theta_mesh, rvz_mesh = np.meshgrid(theta_grid, rvz_grid, indexing="xy")

    for label, start_theta, start_rvz, end_theta, end_rvz in [
        ("0_to_1", src0_theta, src0_rvz, mid_theta, mid_rvz),
        ("1_to_2", mid_theta, mid_rvz, final_theta, final_rvz),
    ]:
        src_density = density_image(start_theta, start_rvz, rvz_edges=rvz_edges, theta_bins=THETA_BINS, wrap_theta=True)
        tgt_density = density_image(end_theta, end_rvz, rvz_edges=rvz_edges, theta_bins=THETA_BINS, wrap_theta=True)
        diff_density = tgt_density - src_density
        diff_abs_vmax = float(np.quantile(np.abs(diff_density).ravel(), 0.995))
        diff_abs_vmax = max(diff_abs_vmax, 1.0)

        field_end_theta, field_end_rvz, field_weight = kernel_mode_endpoint_field(
            src_theta=start_theta,
            src_rvz=start_rvz,
            end_theta=end_theta,
            end_rvz=end_rvz,
            theta_grid=theta_grid,
            rvz_grid=rvz_grid,
            theta_bandwidth=theta_bandwidth,
            rvz_bandwidth=rvz_bandwidth,
            endpoint_theta_bandwidth=theta_bandwidth,
            endpoint_rvz_bandwidth=rvz_bandwidth,
        )
        field_dtheta = wrapped_theta_difference(source_theta=theta_mesh, target_theta=field_end_theta)
        field_drvz = field_end_rvz - rvz_mesh
        speed = np.hypot(field_dtheta, field_drvz)
        mask = support_adaptive_quiver_mask(support=field_weight, min_quantile=0.1, random_seed=RANDOM_SEED)

        out_png = outdir / f"path_sim_difference_quiver_{label}.png"
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        im = ax.imshow(
            diff_density,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * np.pi, rvz_edges[0], rvz_edges[-1]],
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-diff_abs_vmax,
            vmax=diff_abs_vmax,
        )
        q = ax.quiver(
            theta_mesh[mask],
            rvz_mesh[mask],
            field_dtheta[mask],
            field_drvz[mask],
            speed[mask],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.003,
        )
        ax.set_xlabel("tricycle")
        ax.set_ylabel("r_vz")
        ax.set_title(f"Path-sim target-source difference + modal-endpoint field: {label.replace('_', ' -> ')}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="target - source cell count")
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.10, label="endpoint-field speed")
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        print(f"wrote {out_png}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tricycle/r_vz OT heatmaps.")
    parser.add_argument(
        "--mode",
        choices=(
            "solve",
            "ot-heatmap-plot",
            "barycentric-plot",
            "barycentric-heatmap-plot",
            "path-sim-heatmap-plot",
            "path-sim-umap-plot",
            "path-sim-vector-field-plot",
            "path-sim-difference-quiver-plot",
            "barycentric",
        ),
        default="barycentric",
        help="`solve` caches the all-cell AP/ML-penalized OT solve for all barycentric plotters. `barycentric-plot` and `barycentric-heatmap-plot` render from that cache only. `barycentric` does solve+scatter-animation, while `barycentric-heatmap` should be run as `solve` followed by `barycentric-heatmap-plot`.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Results directory containing `subset.h5ad` and the `tricycle_r_vz_transitions` output folder.",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        default=DEFAULT_VZ_H5AD,
        help="Input `.h5ad` used for solve modes.",
    )
    parser.add_argument(
        "--sample-n-per-time",
        type=int,
        default=None,
        help="Optional deterministic per-timepoint cap used only by solve modes.",
    )
    parser.add_argument(
        "--sample-total",
        type=int,
        default=None,
        help="Optional deterministic total sample size that preserves observed stage ratios. Used only by solve modes.",
    )
    parser.add_argument(
        "--backward-tricycle-penalty",
        type=float,
        default=BACKWARD_TRICYCLE_PENALTY,
        help="Backward tricycle penalty used only by solve modes.",
    )
    parser.add_argument(
        "--ap-ml-penalty",
        type=float,
        default=AP_ML_DISPLACEMENT_PENALTY,
        help="AP/ML displacement penalty used only by solve modes.",
    )
    parser.add_argument("--tau-a", type=float, default=TAU_A, help="Source marginal relaxation used only by solve modes.")
    parser.add_argument("--tau-b", type=float, default=TAU_B, help="Target marginal relaxation used only by solve modes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_total is not None and args.sample_n_per_time is not None:
        raise ValueError("Use only one of `--sample-total` or `--sample-n-per-time`.")
    if args.mode == "solve":
        solve_cached_pairs(
            infile=args.infile,
            results_dir=args.results_dir,
            sample_n_per_time=args.sample_n_per_time,
            sample_total=args.sample_total,
            backward_tricycle_penalty_weight=float(args.backward_tricycle_penalty),
            ap_ml_penalty_weight=float(args.ap_ml_penalty),
            tau_a=float(args.tau_a),
            tau_b=float(args.tau_b),
        )
        return
    if args.mode == "barycentric-plot":
        render_barycentric_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "ot-heatmap-plot":
        render_ot_heatmap_summary_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "barycentric-heatmap-plot":
        render_barycentric_heatmap_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "path-sim-heatmap-plot":
        render_path_sim_heatmap_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "path-sim-umap-plot":
        render_path_sim_umap_from_cache(results_dir=args.results_dir, infile=args.infile)
        return
    if args.mode == "path-sim-vector-field-plot":
        render_path_sim_vector_field_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "path-sim-difference-quiver-plot":
        render_path_sim_difference_quiver_from_cache(results_dir=args.results_dir)
        return
    if args.mode == "barycentric":
        solve_cached_pairs(
            infile=args.infile,
            results_dir=args.results_dir,
            sample_n_per_time=args.sample_n_per_time,
            sample_total=args.sample_total,
            backward_tricycle_penalty_weight=float(args.backward_tricycle_penalty),
            ap_ml_penalty_weight=float(args.ap_ml_penalty),
            tau_a=float(args.tau_a),
            tau_b=float(args.tau_b),
        )
        render_barycentric_from_cache(results_dir=args.results_dir)
        return
    raise ValueError(f"Unsupported mode {args.mode!r}.")


if __name__ == "__main__":
    main()
