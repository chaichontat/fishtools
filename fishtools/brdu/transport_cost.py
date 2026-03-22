from __future__ import annotations

import numpy as np


def wrapped_tricycle_delta(*, source_theta: np.ndarray, target_theta: np.ndarray) -> np.ndarray:
    """Return the shortest signed target-minus-source tricycle delta in ``[-pi, pi)``."""

    src = np.asarray(source_theta, dtype=np.float32)
    tgt = np.asarray(target_theta, dtype=np.float32)
    return (tgt[None, :] - src[:, None] + np.pi) % (2.0 * np.pi) - np.pi


def _safe_scale(src: np.ndarray, tgt: np.ndarray) -> np.float32:
    pooled = np.concatenate([np.asarray(src, dtype=np.float32), np.asarray(tgt, dtype=np.float32)])
    scale = np.float32(np.std(pooled, dtype=np.float32))
    if not np.isfinite(scale) or scale <= 0:
        return np.float32(1.0)
    return scale


def pairwise_sqeuclidean_with_backward_tricycle_penalty(
    *,
    src_features: np.ndarray,
    tgt_features: np.ndarray,
    src_tricycle: np.ndarray,
    tgt_tricycle: np.ndarray,
    backward_penalty_weight: float,
    src_ap: np.ndarray | None = None,
    tgt_ap: np.ndarray | None = None,
    src_ml: np.ndarray | None = None,
    tgt_ml: np.ndarray | None = None,
    ap_ml_penalty_weight: float = 0.0,
) -> np.ndarray:
    """Build a pairwise linear OT cost with tricycle and optional AP/ML displacement penalties.

    The base term is squared Euclidean distance in ``src_features``/``tgt_features``.
    The regularizer adds ``weight * backward_delta^2`` where ``backward_delta`` is
    the negative part of the shortest signed tricycle difference.
    If ``ap_ml_penalty_weight > 0``, an additional normalized squared AP/ML displacement
    term is added using the pooled per-pair standard deviation on each axis.
    """

    src = np.asarray(src_features, dtype=np.float32)
    tgt = np.asarray(tgt_features, dtype=np.float32)
    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, found {src.shape} and {tgt.shape}.")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(f"Feature dimensions must match, found {src.shape[1]} and {tgt.shape[1]}.")

    x_sq = np.sum(src * src, axis=1, dtype=np.float32)
    y_sq = np.sum(tgt * tgt, axis=1, dtype=np.float32)
    cost = x_sq[:, None] + y_sq[None, :] - 2.0 * (src @ tgt.T)
    np.maximum(cost, 0.0, out=cost)

    if backward_penalty_weight > 0:
        delta = wrapped_tricycle_delta(source_theta=src_tricycle, target_theta=tgt_tricycle)
        backward = np.clip(-delta, a_min=0.0, a_max=None)
        cost += np.float32(backward_penalty_weight) * backward * backward

    if ap_ml_penalty_weight > 0:
        if src_ap is None or tgt_ap is None or src_ml is None or tgt_ml is None:
            raise ValueError("AP/ML penalty requested but one or more AP/ML coordinate arrays were not provided.")

        src_ap_arr = np.asarray(src_ap, dtype=np.float32).reshape(-1)
        tgt_ap_arr = np.asarray(tgt_ap, dtype=np.float32).reshape(-1)
        src_ml_arr = np.asarray(src_ml, dtype=np.float32).reshape(-1)
        tgt_ml_arr = np.asarray(tgt_ml, dtype=np.float32).reshape(-1)
        if len(src_ap_arr) != src.shape[0] or len(src_ml_arr) != src.shape[0]:
            raise ValueError("Source AP/ML arrays must have length equal to the number of source cells.")
        if len(tgt_ap_arr) != tgt.shape[0] or len(tgt_ml_arr) != tgt.shape[0]:
            raise ValueError("Target AP/ML arrays must have length equal to the number of target cells.")
        if not np.isfinite(src_ap_arr).all() or not np.isfinite(tgt_ap_arr).all():
            raise ValueError("AP/ML penalty requires finite AP coordinates for all source and target cells.")
        if not np.isfinite(src_ml_arr).all() or not np.isfinite(tgt_ml_arr).all():
            raise ValueError("AP/ML penalty requires finite ML coordinates for all source and target cells.")

        ap_scale = _safe_scale(src_ap_arr, tgt_ap_arr)
        ml_scale = _safe_scale(src_ml_arr, tgt_ml_arr)
        ap_delta = (src_ap_arr[:, None] - tgt_ap_arr[None, :]) / ap_scale
        ml_delta = (src_ml_arr[:, None] - tgt_ml_arr[None, :]) / ml_scale
        cost += np.float32(ap_ml_penalty_weight) * (ap_delta * ap_delta + ml_delta * ml_delta)

    return cost


def endpoint_tricycle_unary_penalties(
    *,
    q00_tricycle: np.ndarray,
    q01_tricycle: np.ndarray,
    q10_tricycle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unary tricycle penalties for q00 cells as pre- vs post-endpoints.

    The pre penalty is large when a q00 cell is late relative to the q01 reference.
    The post penalty is large when a q00 cell is early relative to the q10 reference.
    """

    q00 = np.asarray(q00_tricycle, dtype=np.float32).reshape(-1)
    q01 = np.asarray(q01_tricycle, dtype=np.float32).reshape(-1)
    q10 = np.asarray(q10_tricycle, dtype=np.float32).reshape(-1)
    if q00.ndim != 1 or q01.ndim != 1 or q10.ndim != 1:
        raise ValueError("q00_tricycle, q01_tricycle, and q10_tricycle must be 1D.")
    if len(q00) == 0 or len(q01) == 0 or len(q10) == 0:
        raise ValueError("q00_tricycle, q01_tricycle, and q10_tricycle must be non-empty.")
    if not np.isfinite(q00).all() or not np.isfinite(q01).all() or not np.isfinite(q10).all():
        raise ValueError("Tricycle arrays must be finite.")

    q01_ref = np.float32(np.nanmedian(q01))
    q10_ref = np.float32(np.nanmedian(q10))
    pre_delta = wrapped_tricycle_delta(source_theta=q00, target_theta=np.array([q01_ref], dtype=np.float32))[:, 0]
    post_delta = wrapped_tricycle_delta(source_theta=np.array([q10_ref], dtype=np.float32), target_theta=q00)[0]
    pre_penalty = np.clip(-pre_delta, a_min=0.0, a_max=None)
    post_penalty = np.clip(-post_delta, a_min=0.0, a_max=None)
    return pre_penalty.astype(np.float32, copy=False), post_penalty.astype(np.float32, copy=False)
