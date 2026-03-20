from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from moscot.problems.time import TemporalProblem

from fishtools.brdu.transport_cost import pairwise_sqeuclidean_with_backward_tricycle_penalty

PairCostBuilder = Callable[[Any], pd.DataFrame | np.ndarray]


@dataclass
class TemporalProblemConfig:
    time_key: str = "time"
    joint_attr: Optional[str] = "X_scvi"
    policy: str = "sequential"
    cost: str = "sq_euclidean"
    epsilon: float = 1e-3
    tau_a: float = 0.95
    tau_b: float = 0.95
    rank: int = -1
    scale_cost: str = "mean"
    batch_size: Optional[int] = None
    threshold: float = 1e-3
    max_iterations: Optional[int] = None
    estimate_marginals: bool = False
    gene_set_proliferation: Optional[str] = None
    gene_set_apoptosis: Optional[str] = None
    marginal_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _sorted_times(adata: AnnData, time_key: str) -> list[Any]:
    vals = pd.Index(pd.unique(adata.obs[time_key]))
    try:
        vals = vals.sort_values()
    except TypeError:
        vals = pd.Index(sorted(vals.tolist()))
    return vals.tolist()


def validate_adata(adata: AnnData, cfg: TemporalProblemConfig) -> list[Any]:
    if cfg.time_key not in adata.obs:
        raise KeyError(f"`adata.obs[{cfg.time_key!r}]` is missing.")

    if cfg.joint_attr is not None and cfg.joint_attr not in adata.obsm:
        raise KeyError(
            f"`adata.obsm[{cfg.joint_attr!r}]` is missing. "
            "Populate it or set `joint_attr=None` to let moscot compute PCA."
        )

    times = _sorted_times(adata, cfg.time_key)
    if len(times) < 2:
        raise ValueError("Need at least 2 time points.")
    return times


def _normalize_pair_cost(pair_cost: pd.DataFrame | np.ndarray, subproblem: Any) -> pd.DataFrame:
    if isinstance(pair_cost, pd.DataFrame):
        return pair_cost

    arr = np.asarray(pair_cost, dtype=np.float32)
    expected_shape = (subproblem.adata_src.n_obs, subproblem.adata_tgt.n_obs)
    if arr.shape != expected_shape:
        raise ValueError(f"Custom pair cost must have shape {expected_shape}, found {arr.shape}.")
    return pd.DataFrame(arr, index=subproblem.adata_src.obs_names, columns=subproblem.adata_tgt.obs_names, copy=False)


def fit_temporal_problem(
    adata: AnnData,
    cfg: TemporalProblemConfig,
    *,
    pair_cost_builder: PairCostBuilder | None = None,
) -> tuple[TemporalProblem, list[Any]]:
    """Fit a `TemporalProblem`, optionally replacing each prepared pair's linear cost."""

    times = validate_adata(adata, cfg)
    tp = TemporalProblem(adata)

    if cfg.estimate_marginals:
        if cfg.gene_set_proliferation is None or cfg.gene_set_apoptosis is None:
            raise ValueError(
                "If `estimate_marginals=True`, provide both `gene_set_proliferation` and `gene_set_apoptosis`."
            )
        tp = tp.score_genes_for_marginals(
            gene_set_proliferation=cfg.gene_set_proliferation,
            gene_set_apoptosis=cfg.gene_set_apoptosis,
        )

    tp = tp.prepare(
        time_key=cfg.time_key,
        joint_attr=cfg.joint_attr,
        policy=cfg.policy,
        cost=cfg.cost,
        a=cfg.estimate_marginals,
        b=cfg.estimate_marginals,
        marginal_kwargs=dict(cfg.marginal_kwargs),
    )

    if pair_cost_builder is not None:
        for t_src, t_tgt in tp.problems:
            subproblem = tp[t_src, t_tgt]
            subproblem.set_xy(_normalize_pair_cost(pair_cost_builder(subproblem), subproblem), tag="cost_matrix")

    tp = tp.solve(
        epsilon=cfg.epsilon,
        tau_a=cfg.tau_a,
        tau_b=cfg.tau_b,
        rank=cfg.rank,
        scale_cost=cfg.scale_cost,
        batch_size=cfg.batch_size,
        threshold=cfg.threshold,
        max_iterations=cfg.max_iterations,
    )
    return tp, times


def grouped_transitions(
    tp: TemporalProblem,
    source: Any,
    target: Any,
    state_key: str,
) -> pd.DataFrame:
    if state_key not in tp.adata.obs:
        raise KeyError(f"`adata.obs[{state_key!r}]` is missing.")

    trans = tp.cell_transition(
        source=source,
        target=target,
        source_groups=state_key,
        target_groups=state_key,
        forward=True,
        aggregation_mode="annotation",
        normalize=True,
        key_added=None,
    )
    if trans is None:
        raise RuntimeError("`cell_transition(..., key_added=None)` unexpectedly returned `None`.")
    return trans


def push_named_subset(
    tp: TemporalProblem,
    source: Any,
    target: Any,
    label_key: str,
    subset: Any,
    obs_key: str,
) -> pd.Series:
    if label_key not in tp.adata.obs:
        raise KeyError(f"`adata.obs[{label_key!r}]` is missing.")

    tp.push(source=source, target=target, data=label_key, subset=subset, key_added=obs_key)
    return tp.adata.obs[obs_key].copy()


def gaussian_source_kernel(source_coords: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    source_coords_arr = np.asarray(source_coords, dtype=float)
    center_arr = np.asarray(center, dtype=float)
    if source_coords_arr.ndim != 2:
        raise ValueError("`source_coords` must be 2D.")
    if center_arr.shape != (source_coords_arr.shape[1],):
        raise ValueError(f"`center` must have shape ({source_coords_arr.shape[1]},), got {center_arr.shape}.")
    if sigma <= 0:
        raise ValueError("`sigma` must be > 0.")

    d2 = np.sum((source_coords_arr - center_arr[None, :]) ** 2, axis=1)
    weights = np.exp(-0.5 * d2 / (sigma**2))
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Kernel weights are invalid.")
    return weights / weights.sum()


def push_soft_source_region(
    tp: TemporalProblem,
    source: Any,
    target: Any,
    source_weights: np.ndarray,
    obs_key: str,
) -> pd.Series:
    time_mask = tp.adata.obs[tp.temporal_key].to_numpy() == source
    n_source = int(time_mask.sum())
    source_weights_arr = np.asarray(source_weights, dtype=float).ravel()
    if source_weights_arr.shape[0] != n_source:
        raise ValueError(f"`source_weights` must have length {n_source}, got {source_weights_arr.shape[0]}.")

    tp.push(source=source, target=target, data=source_weights_arr, key_added=obs_key)
    return tp.adata.obs[obs_key].copy()


def correlate_target_genes(
    tp: TemporalProblem,
    obs_key: str,
    target: Any,
    layer: Optional[str] = None,
    features: Optional[Sequence[str] | str] = None,
    corr_method: str = "spearman",
    significance_method: str = "fisher",
) -> pd.DataFrame:
    if obs_key not in tp.adata.obs:
        raise KeyError(f"`adata.obs[{obs_key!r}]` is missing.")

    res = tp.compute_feature_correlation(
        obs_key=obs_key,
        corr_method=corr_method,
        significance_method=significance_method,
        annotation={tp.temporal_key: [target]},
        layer=layer,
        features=features,
    )
    return res.sort_values(["qval", "corr"], ascending=[True, False])


def sweep_source_anchors(
    tp: TemporalProblem,
    source: Any,
    target: Any,
    source_coords: np.ndarray,
    anchors: np.ndarray,
    sigma: float,
    layer: Optional[str] = None,
    features: Optional[Sequence[str] | str] = None,
    corr_method: str = "spearman",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source_coords_arr = np.asarray(source_coords, dtype=float)
    anchors_arr = np.asarray(anchors, dtype=float)
    if source_coords_arr.ndim != 2:
        raise ValueError("`source_coords` must be 2D.")
    if anchors_arr.ndim != 2:
        raise ValueError("`anchors` must be 2D.")
    if anchors_arr.shape[1] != source_coords_arr.shape[1]:
        raise ValueError(f"`anchors` must have {source_coords_arr.shape[1]} columns, got {anchors_arr.shape[1]}.")

    corr_cols: dict[str, pd.Series] = {}
    full_tables: dict[str, pd.DataFrame] = {}
    for idx, center in enumerate(anchors_arr):
        obs_key = f"push_{source}_to_{target}_anchor_{idx:03d}"
        weights = gaussian_source_kernel(source_coords_arr, center=center, sigma=sigma)
        push_soft_source_region(tp=tp, source=source, target=target, source_weights=weights, obs_key=obs_key)
        tab = correlate_target_genes(
            tp=tp,
            obs_key=obs_key,
            target=target,
            layer=layer,
            features=features,
            corr_method=corr_method,
        )
        corr_cols[obs_key] = tab["corr"]
        full_tables[obs_key] = tab

    return pd.DataFrame(corr_cols), full_tables


def make_brdu_pair_cost_builder(
    *,
    joint_attr: str = "X_pca",
    tricycle_key: str = "tricycle",
    ap_key: str = "ap",
    ml_key: str = "ml",
    backward_tricycle_penalty: float = 30.0,
    ap_ml_penalty: float = 0.0,
) -> PairCostBuilder:
    """Return a per-pair cost builder that matches the current BrdU/EdU transport geometry."""

    def _build(subproblem: Any) -> pd.DataFrame:
        cost = pairwise_sqeuclidean_with_backward_tricycle_penalty(
            src_features=np.asarray(subproblem.adata_src.obsm[joint_attr], dtype=np.float32),
            tgt_features=np.asarray(subproblem.adata_tgt.obsm[joint_attr], dtype=np.float32),
            src_tricycle=subproblem.adata_src.obs[tricycle_key].to_numpy(dtype=np.float32),
            tgt_tricycle=subproblem.adata_tgt.obs[tricycle_key].to_numpy(dtype=np.float32),
            backward_penalty_weight=backward_tricycle_penalty,
            src_ap=subproblem.adata_src.obs[ap_key].to_numpy(dtype=np.float32),
            tgt_ap=subproblem.adata_tgt.obs[ap_key].to_numpy(dtype=np.float32),
            src_ml=subproblem.adata_src.obs[ml_key].to_numpy(dtype=np.float32),
            tgt_ml=subproblem.adata_tgt.obs[ml_key].to_numpy(dtype=np.float32),
            ap_ml_penalty_weight=ap_ml_penalty,
        )
        return pd.DataFrame(
            cost,
            index=subproblem.adata_src.obs_names,
            columns=subproblem.adata_tgt.obs_names,
            copy=False,
        )

    return _build
