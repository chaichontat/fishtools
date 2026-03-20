from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def subsample_indices(*, n_obs: int, target_n: int, random_seed: int) -> np.ndarray:
    """Return deterministic indices of length ``min(n_obs, target_n)`` without replacement."""

    if target_n <= 0:
        raise ValueError(f"`target_n` must be positive, found {target_n}.")
    if n_obs <= target_n:
        return np.arange(n_obs, dtype=int)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(n_obs, size=target_n, replace=False))


def density_image(
    theta: np.ndarray,
    rvz: np.ndarray,
    *,
    rvz_edges: np.ndarray,
    theta_bins: int,
    weights: np.ndarray | None = None,
    wrap_theta: bool = True,
) -> np.ndarray:
    """Bin tricycle/r_vz coordinates into a 2D density image."""

    theta_values = np.mod(theta, 2.0 * np.pi) if wrap_theta else np.asarray(theta, dtype=np.float32)
    hist, _, _ = np.histogram2d(
        theta_values,
        rvz,
        bins=[theta_bins, rvz_edges],
        range=[[0.0, 2.0 * np.pi], [rvz_edges[0], rvz_edges[-1]]],
        weights=weights,
    )
    return hist.T


def heatmap_intensity(image: np.ndarray, *, log_scale: bool) -> np.ndarray:
    """Return either raw or log-scaled heatmap intensity values."""

    image_arr = np.asarray(image, dtype=np.float32)
    return np.log1p(image_arr) if log_scale else image_arr


def barycentric_density_image(
    *,
    src_theta: np.ndarray,
    src_rvz: np.ndarray,
    mapped_theta: np.ndarray,
    mapped_rvz: np.ndarray,
    alpha: float,
    rvz_edges: np.ndarray,
    theta_bins: int,
    weights: np.ndarray | None = None,
    wrap_theta: bool = True,
) -> np.ndarray:
    """Bin the per-cell barycentric interpolation at time ``alpha`` into tricycle/r_vz density space."""

    src_theta_arr = np.asarray(src_theta, dtype=np.float32)
    src_rvz_arr = np.asarray(src_rvz, dtype=np.float32)
    mapped_theta_arr = np.asarray(mapped_theta, dtype=np.float32)
    mapped_rvz_arr = np.asarray(mapped_rvz, dtype=np.float32)
    pos_theta = src_theta_arr + np.float32(alpha) * (mapped_theta_arr - src_theta_arr)
    pos_rvz = src_rvz_arr + np.float32(alpha) * (mapped_rvz_arr - src_rvz_arr)
    return density_image(
        pos_theta,
        pos_rvz,
        rvz_edges=rvz_edges,
        theta_bins=theta_bins,
        weights=weights,
        wrap_theta=wrap_theta,
    )


def nearest_display_positions(
    *,
    query_theta: np.ndarray,
    query_rvz: np.ndarray,
    target_theta: np.ndarray,
    target_rvz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each display-space query to the nearest target cell, allowing +/- 2pi theta shifts."""

    target_theta_arr = np.asarray(target_theta, dtype=np.float32)
    target_rvz_arr = np.asarray(target_rvz, dtype=np.float32)
    query_theta_arr = np.asarray(query_theta, dtype=np.float32)
    query_rvz_arr = np.asarray(query_rvz, dtype=np.float32)
    if target_theta_arr.shape != target_rvz_arr.shape:
        raise ValueError("`target_theta` and `target_rvz` must have matching shapes.")
    if query_theta_arr.shape != query_rvz_arr.shape:
        raise ValueError("`query_theta` and `query_rvz` must have matching shapes.")

    theta_shifts = np.array([-2.0 * np.pi, 0.0, 2.0 * np.pi], dtype=np.float32)
    tiled_theta = np.concatenate([target_theta_arr + shift for shift in theta_shifts])
    tiled_rvz = np.tile(target_rvz_arr, 3)
    tree = cKDTree(np.column_stack([tiled_theta, tiled_rvz]))
    _, tiled_idx = tree.query(np.column_stack([query_theta_arr, query_rvz_arr]), k=1)

    n_target = int(target_theta_arr.shape[0])
    base_idx = np.asarray(tiled_idx % n_target, dtype=int)
    shift_idx = np.asarray(tiled_idx // n_target, dtype=int)
    matched_theta = target_theta_arr[base_idx] + theta_shifts[shift_idx]
    matched_rvz = target_rvz_arr[base_idx]
    return base_idx, matched_theta.astype(np.float32, copy=False), matched_rvz.astype(np.float32, copy=False)
