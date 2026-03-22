from __future__ import annotations

import numpy as np


def wrapped_theta_difference(*, source_theta: np.ndarray, target_theta: np.ndarray) -> np.ndarray:
    """Return the shortest signed target-minus-source theta difference in ``[-pi, pi)``."""

    src = np.asarray(source_theta, dtype=np.float32)
    tgt = np.asarray(target_theta, dtype=np.float32)
    return (tgt - src + np.pi) % (2.0 * np.pi) - np.pi


def kernel_regressed_vector_field(
    *,
    src_theta: np.ndarray,
    src_rvz: np.ndarray,
    dtheta: np.ndarray,
    drvz: np.ndarray,
    theta_grid: np.ndarray,
    rvz_grid: np.ndarray,
    theta_bandwidth: float,
    rvz_bandwidth: float,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a Gaussian-kernel vector field on a regular grid in tricycle/r_vz space."""

    src_theta_arr = np.asarray(src_theta, dtype=np.float32).reshape(-1)
    src_rvz_arr = np.asarray(src_rvz, dtype=np.float32).reshape(-1)
    dtheta_arr = np.asarray(dtheta, dtype=np.float32).reshape(-1)
    drvz_arr = np.asarray(drvz, dtype=np.float32).reshape(-1)
    theta_grid_arr = np.asarray(theta_grid, dtype=np.float32)
    rvz_grid_arr = np.asarray(rvz_grid, dtype=np.float32)
    if not (src_theta_arr.shape == src_rvz_arr.shape == dtheta_arr.shape == drvz_arr.shape):
        raise ValueError("Source positions and displacement arrays must all have the same length.")
    if theta_bandwidth <= 0 or rvz_bandwidth <= 0:
        raise ValueError("Kernel bandwidths must be positive.")

    theta_mesh, rvz_mesh = np.meshgrid(theta_grid_arr, rvz_grid_arr, indexing="xy")
    query_theta = theta_mesh.ravel()
    query_rvz = rvz_mesh.ravel()
    out_dtheta = np.empty_like(query_theta, dtype=np.float32)
    out_drvz = np.empty_like(query_theta, dtype=np.float32)
    out_weight = np.empty_like(query_theta, dtype=np.float32)

    for start in range(0, query_theta.shape[0], chunk_size):
        stop = min(start + chunk_size, query_theta.shape[0])
        theta_delta = wrapped_theta_difference(
            source_theta=np.broadcast_to(query_theta[start:stop, None], (stop - start, src_theta_arr.shape[0])),
            target_theta=np.broadcast_to(src_theta_arr[None, :], (stop - start, src_theta_arr.shape[0])),
        )
        rvz_delta = src_rvz_arr[None, :] - query_rvz[start:stop, None]
        weight = np.exp(
            -0.5 * (theta_delta / np.float32(theta_bandwidth)) ** 2
            -0.5 * (rvz_delta / np.float32(rvz_bandwidth)) ** 2
        ).astype(np.float32, copy=False)
        weight_sum = weight.sum(axis=1, dtype=np.float32)
        out_weight[start:stop] = weight_sum

        numer_theta = weight @ dtheta_arr
        numer_rvz = weight @ drvz_arr
        valid = weight_sum > 0
        out_dtheta[start:stop] = np.nan
        out_drvz[start:stop] = np.nan
        out_dtheta[start:stop][valid] = numer_theta[valid] / weight_sum[valid]
        out_drvz[start:stop][valid] = numer_rvz[valid] / weight_sum[valid]

    return (
        out_dtheta.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_drvz.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_weight.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
    )


def kernel_regressed_endpoint_field(
    *,
    src_theta: np.ndarray,
    src_rvz: np.ndarray,
    end_theta: np.ndarray,
    end_rvz: np.ndarray,
    theta_grid: np.ndarray,
    rvz_grid: np.ndarray,
    theta_bandwidth: float,
    rvz_bandwidth: float,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a Gaussian-kernel expected-endpoint field on a regular grid in tricycle/r_vz space."""

    src_theta_arr = np.asarray(src_theta, dtype=np.float32).reshape(-1)
    src_rvz_arr = np.asarray(src_rvz, dtype=np.float32).reshape(-1)
    end_theta_arr = np.asarray(end_theta, dtype=np.float32).reshape(-1)
    end_rvz_arr = np.asarray(end_rvz, dtype=np.float32).reshape(-1)
    theta_grid_arr = np.asarray(theta_grid, dtype=np.float32)
    rvz_grid_arr = np.asarray(rvz_grid, dtype=np.float32)
    if not (src_theta_arr.shape == src_rvz_arr.shape == end_theta_arr.shape == end_rvz_arr.shape):
        raise ValueError("Source positions and endpoint arrays must all have the same length.")
    if theta_bandwidth <= 0 or rvz_bandwidth <= 0:
        raise ValueError("Kernel bandwidths must be positive.")

    theta_mesh, rvz_mesh = np.meshgrid(theta_grid_arr, rvz_grid_arr, indexing="xy")
    query_theta = theta_mesh.ravel()
    query_rvz = rvz_mesh.ravel()
    out_theta = np.empty_like(query_theta, dtype=np.float32)
    out_rvz = np.empty_like(query_theta, dtype=np.float32)
    out_weight = np.empty_like(query_theta, dtype=np.float32)
    end_cos = np.cos(end_theta_arr).astype(np.float32, copy=False)
    end_sin = np.sin(end_theta_arr).astype(np.float32, copy=False)

    for start in range(0, query_theta.shape[0], chunk_size):
        stop = min(start + chunk_size, query_theta.shape[0])
        theta_delta = wrapped_theta_difference(
            source_theta=np.broadcast_to(query_theta[start:stop, None], (stop - start, src_theta_arr.shape[0])),
            target_theta=np.broadcast_to(src_theta_arr[None, :], (stop - start, src_theta_arr.shape[0])),
        )
        rvz_delta = src_rvz_arr[None, :] - query_rvz[start:stop, None]
        weight = np.exp(
            -0.5 * (theta_delta / np.float32(theta_bandwidth)) ** 2
            -0.5 * (rvz_delta / np.float32(rvz_bandwidth)) ** 2
        ).astype(np.float32, copy=False)
        weight_sum = weight.sum(axis=1, dtype=np.float32)
        out_weight[start:stop] = weight_sum

        valid = weight_sum > 0
        out_theta[start:stop] = np.nan
        out_rvz[start:stop] = np.nan
        mean_cos = weight @ end_cos
        mean_sin = weight @ end_sin
        mean_rvz = weight @ end_rvz_arr
        out_theta[start:stop][valid] = np.mod(np.arctan2(mean_sin[valid], mean_cos[valid]), 2.0 * np.pi)
        out_rvz[start:stop][valid] = mean_rvz[valid] / weight_sum[valid]

    return (
        out_theta.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_rvz.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_weight.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
    )


def kernel_mode_endpoint_field(
    *,
    src_theta: np.ndarray,
    src_rvz: np.ndarray,
    end_theta: np.ndarray,
    end_rvz: np.ndarray,
    theta_grid: np.ndarray,
    rvz_grid: np.ndarray,
    theta_bandwidth: float,
    rvz_bandwidth: float,
    endpoint_theta_bandwidth: float,
    endpoint_rvz_bandwidth: float,
    top_k: int = 128,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a local modal-endpoint field on a regular grid in tricycle/r_vz space."""

    src_theta_arr = np.asarray(src_theta, dtype=np.float32).reshape(-1)
    src_rvz_arr = np.asarray(src_rvz, dtype=np.float32).reshape(-1)
    end_theta_arr = np.asarray(end_theta, dtype=np.float32).reshape(-1)
    end_rvz_arr = np.asarray(end_rvz, dtype=np.float32).reshape(-1)
    theta_grid_arr = np.asarray(theta_grid, dtype=np.float32)
    rvz_grid_arr = np.asarray(rvz_grid, dtype=np.float32)
    if not (src_theta_arr.shape == src_rvz_arr.shape == end_theta_arr.shape == end_rvz_arr.shape):
        raise ValueError("Source positions and endpoint arrays must all have the same length.")
    if theta_bandwidth <= 0 or rvz_bandwidth <= 0 or endpoint_theta_bandwidth <= 0 or endpoint_rvz_bandwidth <= 0:
        raise ValueError("All kernel bandwidths must be positive.")

    theta_mesh, rvz_mesh = np.meshgrid(theta_grid_arr, rvz_grid_arr, indexing="xy")
    query_theta = theta_mesh.ravel()
    query_rvz = rvz_mesh.ravel()
    out_theta = np.empty_like(query_theta, dtype=np.float32)
    out_rvz = np.empty_like(query_theta, dtype=np.float32)
    out_weight = np.empty_like(query_theta, dtype=np.float32)
    use_k = min(int(top_k), int(src_theta_arr.shape[0]))

    for start in range(0, query_theta.shape[0], chunk_size):
        stop = min(start + chunk_size, query_theta.shape[0])
        theta_delta = wrapped_theta_difference(
            source_theta=np.broadcast_to(query_theta[start:stop, None], (stop - start, src_theta_arr.shape[0])),
            target_theta=np.broadcast_to(src_theta_arr[None, :], (stop - start, src_theta_arr.shape[0])),
        )
        rvz_delta = src_rvz_arr[None, :] - query_rvz[start:stop, None]
        weight = np.exp(
            -0.5 * (theta_delta / np.float32(theta_bandwidth)) ** 2
            -0.5 * (rvz_delta / np.float32(rvz_bandwidth)) ** 2
        ).astype(np.float32, copy=False)
        weight_sum = weight.sum(axis=1, dtype=np.float32)
        out_weight[start:stop] = weight_sum

        top_idx = np.argpartition(-weight, kth=use_k - 1, axis=1)[:, :use_k]
        local_weight = np.take_along_axis(weight, top_idx, axis=1)
        cand_theta = end_theta_arr[top_idx]
        cand_rvz = end_rvz_arr[top_idx]
        theta_pair = wrapped_theta_difference(
            source_theta=cand_theta[:, :, None],
            target_theta=cand_theta[:, None, :],
        )
        rvz_pair = cand_rvz[:, :, None] - cand_rvz[:, None, :]
        endpoint_kernel = np.exp(
            -0.5 * (theta_pair / np.float32(endpoint_theta_bandwidth)) ** 2
            -0.5 * (rvz_pair / np.float32(endpoint_rvz_bandwidth)) ** 2
        ).astype(np.float32, copy=False)
        score = endpoint_kernel @ local_weight[:, :, None]
        best = np.argmax(score[:, :, 0], axis=1)

        out_theta[start:stop] = cand_theta[np.arange(stop - start), best]
        out_rvz[start:stop] = cand_rvz[np.arange(stop - start), best]

    return (
        out_theta.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_rvz.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
        out_weight.reshape(rvz_grid_arr.shape[0], theta_grid_arr.shape[0]),
    )


def support_adaptive_quiver_mask(
    *,
    support: np.ndarray,
    min_quantile: float = 0.1,
    random_seed: int = 0,
) -> np.ndarray:
    """Keep more arrows in high-support regions via deterministic support-weighted thinning."""

    support_arr = np.asarray(support, dtype=np.float32)
    finite = np.isfinite(support_arr)
    if not np.any(finite):
        return np.zeros_like(support_arr, dtype=bool)

    floor = float(np.quantile(support_arr[finite], min_quantile))
    clipped = np.clip(support_arr - np.float32(floor), a_min=0.0, a_max=None)
    max_clipped = float(np.max(clipped[finite]))
    if max_clipped <= 0:
        return finite

    keep_prob = np.zeros_like(support_arr, dtype=np.float32)
    keep_prob[finite] = clipped[finite] / np.float32(max_clipped)
    row_idx, col_idx = np.indices(support_arr.shape, dtype=np.float32)
    phase = row_idx * np.float32(12.9898 + 0.17 * random_seed) + col_idx * np.float32(78.233 + 0.11 * random_seed)
    pseudo = np.mod(np.sin(phase) * np.float32(43758.5453), 1.0)
    pseudo = np.abs(pseudo).astype(np.float32, copy=False)
    return finite & (keep_prob > pseudo)
