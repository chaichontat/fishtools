#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


EPS = 1e-12


def logsumexp(a: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)


def entropy(p: np.ndarray, axis: int = 1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(p * np.log(p + EPS), axis=axis)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_prob_power_by_batch(
    *,
    p: np.ndarray,
    hard: np.ndarray,
    batch_idx: np.ndarray,
    n_batches: int,
    max_pow: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monotone calibration: for each batch b, choose a power s_b so that
        mean_i p_i**s_b  ~=  mean_i hard_i
    within that batch.

    Returns (p_cal, pow_by_batch).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    hard = np.asarray(hard, dtype=np.int8).reshape(-1)
    bidx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
    if not (p.shape == hard.shape == bidx.shape):
        raise ValueError("p, hard, and batch_idx must have the same shape.")
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive.")
    if not np.isin(hard, [0, 1]).all():
        raise ValueError("hard must be binary.")

    out = np.empty_like(p)
    pow_b = np.full(int(n_batches), np.nan, dtype=np.float64)

    for b in range(int(n_batches)):
        idx = np.where(bidx == b)[0]
        if idx.size == 0:
            continue
        pb = np.clip(p[idx], 0.0, 1.0)
        fb = float(np.mean(hard[idx]))

        if fb <= 0.0 + 1e-12:
            s = float(max_pow)
            out[idx] = pb**s
            pow_b[b] = s
            continue
        if fb >= 1.0 - 1e-12:
            s = 1e-6
            out[idx] = pb**s
            pow_b[b] = s
            continue

        # Feasible range is [0, frac(pb>0)] as s goes [inf, 0].
        f0 = float(np.mean(pb > 0))
        if fb > f0:
            s = 1e-6
            out[idx] = pb**s
            pow_b[b] = s
            continue

        lo = 1e-6
        hi = float(max_pow)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fmid = float(np.mean(pb**mid))
            if fmid > fb:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        out[idx] = pb**s
        pow_b[b] = s

    if not np.isfinite(out).all():
        raise RuntimeError("Internal error: non-finite calibrated probabilities.")
    return out, pow_b


def _calibrate_prob_logit_shift_by_batch(
    *,
    p: np.ndarray,
    hard: np.ndarray,
    batch_idx: np.ndarray,
    n_batches: int,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
    max_shift: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monotone calibration: for each batch b, choose a logit shift c_b so that
        mean_i sigmoid(logit(p_i) + c_b) ~= mean_i hard_i
    within that batch.

    Compared to power calibration, this can also meaningfully down-shift
    probabilities that are very close to 1, because p=1 is clipped to (1-eps).

    Returns (p_cal, shift_by_batch).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    hard = np.asarray(hard, dtype=np.int8).reshape(-1)
    bidx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
    if not (p.shape == hard.shape == bidx.shape):
        raise ValueError("p, hard, and batch_idx must have the same shape.")
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive.")
    if not np.isin(hard, [0, 1]).all():
        raise ValueError("hard must be binary.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != p.shape:
            raise ValueError("mask must have the same shape as p.")
        if not np.any(mask):
            raise ValueError("mask selects zero rows.")
    if not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must satisfy 0 < eps < 0.5.")
    if float(max_shift) <= 0:
        raise ValueError("max_shift must be positive.")

    p = np.clip(p, float(eps), 1.0 - float(eps))
    logit_p = np.log(p) - np.log1p(-p)

    out = p.copy()
    shift_b = np.full(int(n_batches), np.nan, dtype=np.float64)

    for b in range(int(n_batches)):
        if mask is None:
            idx = np.where(bidx == b)[0]
        else:
            idx = np.where((bidx == b) & mask)[0]
        if idx.size == 0:
            continue
        z = logit_p[idx]
        fb = float(np.mean(hard[idx]))

        if fb <= 0.0 + 1e-12:
            c = -float(max_shift)
            out[idx] = sigmoid(z + c)
            shift_b[b] = c
            continue
        if fb >= 1.0 - 1e-12:
            c = float(max_shift)
            out[idx] = sigmoid(z + c)
            shift_b[b] = c
            continue

        lo = -float(max_shift)
        hi = float(max_shift)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fmid = float(np.mean(sigmoid(z + mid)))
            if fmid < fb:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        out[idx] = sigmoid(z + c)
        shift_b[b] = c

    if not np.isfinite(out).all():
        raise RuntimeError("Internal error: non-finite calibrated probabilities.")
    return out, shift_b


def _calibrate_prob_logit_shift_to_xmin_by_batch(
    *,
    p: np.ndarray,
    x: np.ndarray,
    batch_idx: np.ndarray,
    n_batches: int,
    x_min: float,
    nearest_k: int = 2000,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-batch monotone calibration that only raises the implied intensity threshold.

    We apply a per-batch additive shift to logit(p):
      p' = sigmoid(logit(p) + shift_b)

    shift_b is estimated from cells near a target intensity x_min, and is clamped
    to <= 0 so that probabilities only decrease (pushing the p=0.5 boundary to
    higher intensities in batches where it was too low).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    bidx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
    if not (p.shape == x.shape == bidx.shape):
        raise ValueError("p, x, and batch_idx must have the same shape.")
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive.")
    if not math.isfinite(float(x_min)):
        raise ValueError("x_min must be finite.")
    k = int(nearest_k)
    if k <= 0:
        raise ValueError("nearest_k must be positive.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != p.shape:
            raise ValueError("mask must have the same shape as p.")
        if not np.any(mask):
            raise ValueError("mask selects zero rows.")
    if not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must satisfy 0 < eps < 0.5.")

    p = np.clip(p, float(eps), 1.0 - float(eps))
    logit_p = np.log(p) - np.log1p(-p)

    out = p.copy()
    shift_b = np.full(int(n_batches), np.nan, dtype=np.float64)

    x_min_f = float(x_min)
    for b in range(int(n_batches)):
        if mask is None:
            idx = np.where(bidx == b)[0]
        else:
            idx = np.where((bidx == b) & mask)[0]
        if idx.size == 0:
            continue
        xb = x[idx]
        zb = logit_p[idx]
        ok = np.isfinite(xb) & np.isfinite(zb)
        if not np.any(ok):
            continue
        xb = xb[ok]
        zb = zb[ok]
        if xb.size == 0:
            continue

        order = np.argsort(np.abs(xb - x_min_f))
        take = order[: min(k, int(order.size))]
        shift = -float(np.median(zb[take]))
        shift = min(0.0, shift)
        shift_b[b] = shift
        if shift == 0.0:
            continue
        out[idx] = sigmoid(logit_p[idx] + shift)

    if not np.isfinite(out).all():
        raise RuntimeError("Internal error: non-finite calibrated probabilities.")
    return out, shift_b


def _calibrate_prob_logit_shift_to_xrange_by_batch(
    *,
    p: np.ndarray,
    x: np.ndarray,
    batch_idx: np.ndarray,
    n_batches: int,
    x_min: float,
    x_max: float,
    nearest_k: int = 2000,
    mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-batch monotone calibration that constrains the implied p=0.5 boundary to lie within [x_min, x_max].

    If the current boundary is below x_min (i.e. p(x_min) > 0.5), we apply a negative logit shift so that
    p'(x_min) ≈ 0.5 (raises threshold).

    If the current boundary is above x_max (i.e. p(x_max) < 0.5), we apply a positive logit shift so that
    p'(x_max) ≈ 0.5 (lowers threshold).

    Otherwise, shift=0.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    bidx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
    if not (p.shape == x.shape == bidx.shape):
        raise ValueError("p, x, and batch_idx must have the same shape.")
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive.")
    if not math.isfinite(float(x_min)) or not math.isfinite(float(x_max)):
        raise ValueError("x_min and x_max must be finite.")
    if float(x_min) >= float(x_max):
        raise ValueError("x_min must be < x_max.")
    k = int(nearest_k)
    if k <= 0:
        raise ValueError("nearest_k must be positive.")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != p.shape:
            raise ValueError("mask must have the same shape as p.")
        if not np.any(mask):
            raise ValueError("mask selects zero rows.")
    if not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must satisfy 0 < eps < 0.5.")

    p = np.clip(p, float(eps), 1.0 - float(eps))
    logit_p = np.log(p) - np.log1p(-p)

    out = p.copy()
    shift_b = np.full(int(n_batches), np.nan, dtype=np.float64)

    x_min_f = float(x_min)
    x_max_f = float(x_max)
    for b in range(int(n_batches)):
        if mask is None:
            idx = np.where(bidx == b)[0]
        else:
            idx = np.where((bidx == b) & mask)[0]
        if idx.size == 0:
            continue
        xb = x[idx]
        zb = logit_p[idx]
        ok = np.isfinite(xb) & np.isfinite(zb)
        if not np.any(ok):
            continue
        xb = xb[ok]
        zb = zb[ok]
        if xb.size == 0:
            continue

        # Median logit among nearest-to-xmin / nearest-to-xmax cells.
        order_min = np.argsort(np.abs(xb - x_min_f))
        take_min = order_min[: min(k, int(order_min.size))]
        m_min = float(np.median(zb[take_min]))

        order_max = np.argsort(np.abs(xb - x_max_f))
        take_max = order_max[: min(k, int(order_max.size))]
        m_max = float(np.median(zb[take_max]))

        # Decide direction: if p(xmin)>0.5 -> boundary<xmin -> decrease probs (negative shift).
        # If p(xmax)<0.5 -> boundary>xmax -> increase probs (positive shift).
        shift = 0.0
        if m_min > 0.0:
            shift = -m_min
        elif m_max < 0.0:
            shift = -m_max

        shift_b[b] = float(shift)
        if shift == 0.0:
            continue
        out[idx] = sigmoid(logit_p[idx] + float(shift))

    if not np.isfinite(out).all():
        raise RuntimeError("Internal error: non-finite calibrated probabilities.")
    return out, shift_b


def _sample_indices(n: int, max_n: int | None, *, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=max_n, replace=False)).astype(np.int64, copy=False)


def _parse_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _write_summary_tsv(path: Path, items: dict[str, object]) -> None:
    lines = ["key\tvalue"]
    lines.extend(f"{key}\t{value}" for key, value in items.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _select_pos_components_1d(
    *,
    mu: np.ndarray,
    pi: np.ndarray,
    mode: str,
    topk: int,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    if mu.shape != pi.shape:
        raise ValueError("mu and pi must have same shape.")
    K = int(mu.size)
    if K < 2:
        raise ValueError("expected K>=2 components.")

    if mode == "topk":
        k = int(topk)
        if not (1 <= k <= K):
            raise ValueError(f"topk must satisfy 1..K (got {k} with K={K})")
        order = np.argsort(mu)
        return np.sort(order[-k:]).astype(int)
    if mode == "not_main":
        main = int(np.argmax(pi))
        return np.asarray([k for k in range(K) if k != main], dtype=int)

    raise ValueError(f"Unknown pos mode: {mode!r}")


def _select_pos_components_1d_by_manual(
    *,
    resp: np.ndarray,
    hard: np.ndarray,
) -> np.ndarray:
    """
    Select which mixture components should count as "positive", using manual hard calls
    as a mapping layer only (no supervision in fitting).

    We compute per-component manual-positive rates:
        theta_k = sum_i resp[i,k] * hard[i] / sum_i resp[i,k]
    Then select the number of positive components m=1..K that makes the implied
    positive mass closest to the manual positive fraction:
        mean_i sum_{k in top-m(theta)} resp[i,k]  ~  mean_i hard[i]
    """
    resp = np.asarray(resp, dtype=np.float64)
    if resp.ndim != 2:
        raise ValueError("resp must be (N,K)")
    hard = np.asarray(hard, dtype=np.int8).reshape(-1)
    if resp.shape[0] != hard.shape[0]:
        raise ValueError("resp and hard must have matching N")
    if not np.isin(hard, [0, 1]).all():
        raise ValueError("hard must be binary (0/1)")

    Nk = resp.sum(axis=0) + 1e-12
    theta = (resp.T @ hard.astype(np.float64, copy=False)) / Nk
    order = np.argsort(theta)  # ascending
    target = float(np.mean(hard))

    best_m = 1
    best_err = float("inf")
    for m in range(1, int(resp.shape[1]) + 1):
        pos = order[-m:]
        frac = float(np.mean(resp[:, pos].sum(axis=1)))
        err = abs(frac - target)
        if err < best_err:
            best_err = err
            best_m = m

    return np.sort(order[-best_m:]).astype(int, copy=False)


class BatchAwareDiagGMM2D:
    """
    Batch-aware 2D diagonal Gaussian mixture with 4 components (00, 10, 01, 11)
    and explicit staining-run nuisance parameters.

    Observed x_i (2D: [EdU_feature, BrdU_feature]) in batch b is modeled as:
        x = alpha_b + beta_b * u
        u | class=k, batch=b ~ N(mu_k, gamma_b * diag(var_k))

    Identifiability:
      A reference batch is fixed to alpha=0, beta=1, gamma=1.
    """

    def __init__(
        self,
        *,
        covariance: str = "diag",
        reg_covar: float = 1e-3,
        max_iter: int = 200,
        tol: float = 1e-5,
        random_state: int = 0,
        min_beta: float = 1e-3,
        max_beta: float = 5.0,
        min_gamma: float = 1e-3,
        max_gamma: float = 50.0,
        label_weight: float = 0.0,
        theta_eps: float = 1e-3,
        manual_eps: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.K = 4
        covariance = str(covariance).strip().lower()
        if covariance not in {"diag", "full"}:
            raise ValueError("covariance must be 'diag' or 'full'")
        self.covariance = covariance
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)
        self.label_weight = float(label_weight)
        self.theta_eps = float(theta_eps)
        self.manual_eps = float(manual_eps)
        self.verbose = bool(verbose)

    def transform_to_u(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        alpha = self.alpha_[batch_idx]
        beta = self.beta_[batch_idx]
        return (X - alpha) / beta

    def _e_step(
        self,
        U: np.ndarray,
        batch_idx: np.ndarray,
        *,
        y_edu: np.ndarray | None = None,
        y_brdu: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Returns (resp, log_norm_per_obs)
        gamma_obs = self.gamma_[batch_idx]  # (N,)
        diff = U[:, None, :] - self.mu_[None, :, :]  # (N,K,2)

        if self.covariance == "diag":
            var = self.var_[None, :, :]  # (1,K,2)
            g = gamma_obs[:, None, None]  # (N,1,1)

            log_det = np.sum(np.log(2.0 * np.pi * g * var), axis=2)  # (N,K)
            quad = np.sum((diff**2) / (g * var), axis=2)  # (N,K)
            log_p = -0.5 * (log_det + quad) + np.log(self.pi_[None, :])  # (N,K)
        else:
            cov = self.cov_[None, :, :, :]  # (1,K,2,2)
            a = cov[:, :, 0, 0]
            b = cov[:, :, 0, 1]
            c = cov[:, :, 1, 1]
            det = a * c - b * b
            det = np.clip(det, self.reg_covar * self.reg_covar, np.inf)

            inv_00 = c / det
            inv_01 = -b / det
            inv_11 = a / det

            dx = diff[:, :, 0]
            dy = diff[:, :, 1]
            mahal = inv_00 * dx * dx + 2.0 * inv_01 * dx * dy + inv_11 * dy * dy  # (N,K)
            quad = mahal / gamma_obs[:, None]  # (N,K)
            log_det = (2.0 * math.log(2.0 * math.pi)) + (2.0 * np.log(gamma_obs)[:, None]) + np.log(det)  # (N,K)
            log_p = -0.5 * (log_det + quad) + np.log(self.pi_[None, :])  # (N,K)

        if self.label_weight > 0 and y_edu is not None and y_brdu is not None:
            y_edu = np.asarray(y_edu, dtype=np.float64).reshape(-1, 1)
            y_brdu = np.asarray(y_brdu, dtype=np.float64).reshape(-1, 1)
            if y_edu.shape[0] != log_p.shape[0] or y_brdu.shape[0] != log_p.shape[0]:
                raise ValueError("label arrays must have same length as U")

            te = np.clip(self.theta_edu_, self.theta_eps, 1.0 - self.theta_eps)[None, :]
            tb = np.clip(self.theta_brdu_, self.theta_eps, 1.0 - self.theta_eps)[None, :]
            log_lab = (
                y_edu * np.log(te)
                + (1.0 - y_edu) * np.log(1.0 - te)
                + y_brdu * np.log(tb)
                + (1.0 - y_brdu) * np.log(1.0 - tb)
            )
            log_p = log_p + float(self.label_weight) * log_lab

        log_norm = logsumexp(log_p, axis=1, keepdims=True)  # (N,1)
        resp = np.exp(log_p - log_norm)  # (N,K)
        return resp, log_norm[:, 0]

    def score_samples(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        # log p(x_i) in observed space, including Jacobian.
        X = np.asarray(X, dtype=np.float64)
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        U = self.transform_to_u(X, batch_idx)
        _, log_norm_u = self._e_step(U, batch_idx)  # log p(u_i)
        jac = -np.sum(np.log(self.beta_[batch_idx]), axis=1)
        return log_norm_u + jac

    def predict_proba(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        U = self.transform_to_u(X, batch_idx)
        resp, _ = self._e_step(U, batch_idx)
        return resp

    def component_params_in_x_space(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (m, var_x or cov_x):
          m[b,k,2] = alpha_b + beta_b * mu_k
          diag case: var_x[b,k,2] = beta_b^2 * (gamma_b * var_k)
          full case: cov_x[b,k,2,2] = diag(beta_b) @ (gamma_b * cov_k) @ diag(beta_b)
        """
        B = int(self.alpha_.shape[0])
        m = self.alpha_[:, None, :] + self.beta_[:, None, :] * self.mu_[None, :, :]
        if self.covariance == "diag":
            var_x = (self.beta_[:, None, :] ** 2) * (self.gamma_[:, None, None] * self.var_[None, :, :])
            if m.shape != (B, self.K, 2) or var_x.shape != (B, self.K, 2):
                raise ValueError("unexpected component params shapes")
            return m, var_x

        cov = self.cov_[None, :, :, :]  # (1,K,2,2)
        gamma = self.gamma_[:, None, None, None]  # (B,1,1,1)
        beta = self.beta_[:, None, :]  # (B,1,2)
        scale = beta[:, :, :, None] * beta[:, :, None, :]  # (B,1,2,2)
        cov_x = scale * (gamma * cov)  # (B,K,2,2)
        if m.shape != (B, self.K, 2) or cov_x.shape != (B, self.K, 2, 2):
            raise ValueError("unexpected component params shapes")
        return m, cov_x

    def map_components_to_classes(self) -> dict[str, list[int]]:
        """
        Deterministic mapping of the 4 mixture components to:
          neg, EdU_only, BrdU_only, double

        Uses component means only (no thresholds, no controls):
          - neg:    smallest (mu_EdU + mu_BrdU)
          - double: largest  (mu_EdU + mu_BrdU)
          - remaining two: the one with larger mu_EdU is EdU_only; the other BrdU_only
        """
        if bool(getattr(self, "freeze_theta_", False)):
            return {"neg": [0], "EdU_only": [1], "BrdU_only": [2], "double": [3]}

        if hasattr(self, "theta_edu_") and hasattr(self, "theta_brdu_") and self.label_weight > 0:
            score = self.theta_edu_ + self.theta_brdu_
            neg = int(np.argmin(score))
            dbl = int(np.argmax(score))
            remaining = [k for k in range(self.K) if k not in (neg, dbl)]
            edu_only = int(remaining[int(np.argmax(self.theta_edu_[remaining]))])
            brdu_only = int([k for k in remaining if k != edu_only][0])
            return {"neg": [neg], "EdU_only": [edu_only], "BrdU_only": [brdu_only], "double": [dbl]}

        mu = self.mu_
        score = mu.sum(axis=1)
        neg = int(np.argmin(score))
        dbl = int(np.argmax(score))
        remaining = [k for k in range(self.K) if k not in (neg, dbl)]
        edu_only = int(remaining[int(np.argmax(mu[remaining, 0]))])
        brdu_only = int([k for k in remaining if k != edu_only][0])
        return {"neg": [neg], "EdU_only": [edu_only], "BrdU_only": [brdu_only], "double": [dbl]}

    def fit(
        self,
        X: np.ndarray,
        batch_idx: np.ndarray,
        *,
        ref_batch: int | None = None,
        y_edu: np.ndarray | None = None,
        y_brdu: np.ndarray | None = None,
        freeze_theta: bool = False,
        learn_affine: bool = False,
        affine_init_mask: np.ndarray | None = None,
    ) -> BatchAwareDiagGMM2D:
        X = np.asarray(X, dtype=np.float64)
        batch_idx = np.asarray(batch_idx, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must be (N,2)")
        if batch_idx.shape != (X.shape[0],):
            raise ValueError("batch_idx must be (N,)")

        N = int(X.shape[0])
        B = int(batch_idx.max()) + 1
        if B < 1:
            raise ValueError("no batches found")

        if ref_batch is None:
            counts = np.bincount(batch_idx, minlength=B)
            ref_batch = int(np.argmax(counts))
        self.ref_batch_ = int(ref_batch)

        use_labels = self.label_weight > 0 and y_edu is not None and y_brdu is not None
        if use_labels:
            y_edu = np.asarray(y_edu, dtype=np.float64).reshape(-1)
            y_brdu = np.asarray(y_brdu, dtype=np.float64).reshape(-1)
            if y_edu.shape != (N,) or y_brdu.shape != (N,):
                raise ValueError("y_edu/y_brdu must be (N,) and match X rows")
            if np.any((y_edu < -1e-6) | (y_edu > 1 + 1e-6)) or np.any((y_brdu < -1e-6) | (y_brdu > 1 + 1e-6)):
                raise ValueError("y_edu/y_brdu values must be in [0,1]")

        if affine_init_mask is not None:
            affine_init_mask = np.asarray(affine_init_mask, dtype=bool).reshape(-1)
            if affine_init_mask.shape != (N,):
                raise ValueError("affine_init_mask must be (N,) and match X rows")
            if not np.any(affine_init_mask):
                raise ValueError("affine_init_mask selects zero rows")

        def mad(a: np.ndarray) -> np.ndarray:
            med = np.median(a, axis=0)
            return np.median(np.abs(a - med), axis=0) + 1e-9

        alpha = np.zeros((B, 2), dtype=np.float64)
        beta = np.ones((B, 2), dtype=np.float64)
        gamma = np.ones((B,), dtype=np.float64)

        ref_sel = batch_idx == self.ref_batch_
        if affine_init_mask is not None:
            ref_sel = ref_sel & affine_init_mask
        X_ref = X[ref_sel]
        if X_ref.shape[0] < 50:
            raise ValueError(
                f"too few rows for affine init in reference batch={self.ref_batch_} (n={X_ref.shape[0]}); "
                "use a broader affine init mask or disable masked affine init"
            )
        med_ref = np.median(X_ref, axis=0)
        mad_ref = mad(X_ref)

        for b in range(B):
            if b == self.ref_batch_:
                continue
            sel = batch_idx == b
            if affine_init_mask is not None:
                sel = sel & affine_init_mask
            Xb = X[sel]
            if Xb.shape[0] == 0:
                continue
            if Xb.shape[0] < 50:
                raise ValueError(
                    f"too few rows for affine init in batch={b} (n={Xb.shape[0]}); "
                    "use a broader affine init mask or disable masked affine init"
                )
            med_b = np.median(Xb, axis=0)
            mad_b = mad(Xb)
            beta[b] = np.clip(mad_b / mad_ref, self.min_beta, self.max_beta)
            alpha[b] = med_b - beta[b] * med_ref

        alpha[self.ref_batch_] = 0.0
        beta[self.ref_batch_] = 1.0
        gamma[self.ref_batch_] = 1.0
        self.alpha_ = alpha
        self.beta_ = beta
        self.gamma_ = gamma

        U0 = self.transform_to_u(X, batch_idx)
        init_from_manual = bool(use_labels and freeze_theta)
        if init_from_manual:
            edu_pos = np.asarray(y_edu >= 0.5, dtype=np.int8)
            brdu_pos = np.asarray(y_brdu >= 0.5, dtype=np.int8)
            cls = edu_pos + 2 * brdu_pos  # 0=neg, 1=EdU_only, 2=BrdU_only, 3=double
            counts = np.bincount(cls, minlength=self.K).astype(np.int64, copy=False)
            if np.any(counts <= 0):
                raise ValueError(f"manual init requires all 4 classes to be present; counts={counts.tolist()}")

            self.pi_ = counts.astype(np.float64) / float(counts.sum())
            self.mu_ = np.empty((self.K, 2), dtype=np.float64)
            for k in range(self.K):
                Uk = U0[cls == k]
                self.mu_[k] = np.mean(Uk, axis=0)
            if self.covariance == "diag":
                self.var_ = np.empty((self.K, 2), dtype=np.float64)
                for k in range(self.K):
                    Uk = U0[cls == k]
                    self.var_[k] = np.var(Uk, axis=0) + self.reg_covar
                self.var_ = np.clip(self.var_, self.reg_covar, np.inf)
            else:
                self.cov_ = np.empty((self.K, 2, 2), dtype=np.float64)
                for k in range(self.K):
                    Uk = U0[cls == k]
                    cov_k = np.cov(Uk.T, bias=True)
                    cov_k = np.asarray(cov_k, dtype=np.float64)
                    cov_k[0, 0] += self.reg_covar
                    cov_k[1, 1] += self.reg_covar
                    self.cov_[k] = cov_k
        else:
            qlo = np.quantile(U0, 0.2, axis=0)
            qhi = np.quantile(U0, 0.8, axis=0)
            means_init = np.array(
                [[qlo[0], qlo[1]], [qhi[0], qlo[1]], [qlo[0], qhi[1]], [qhi[0], qhi[1]]],
                dtype=np.float64,
            )

            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(
                    n_components=self.K,
                    covariance_type="diag" if self.covariance == "diag" else "full",
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    means_init=means_init,
                    n_init=3,
                    max_iter=200,
                )
                gmm.fit(U0)
                self.pi_ = np.asarray(gmm.weights_, dtype=np.float64).copy()
                self.mu_ = np.asarray(gmm.means_, dtype=np.float64).copy()
                if self.covariance == "diag":
                    self.var_ = np.clip(np.asarray(gmm.covariances_, dtype=np.float64).copy(), self.reg_covar, np.inf)
                else:
                    self.cov_ = np.asarray(gmm.covariances_, dtype=np.float64).copy()
                    self.cov_[:, 0, 0] += self.reg_covar
                    self.cov_[:, 1, 1] += self.reg_covar
            except Exception:
                self.pi_ = np.ones((self.K,), dtype=np.float64) / float(self.K)
                self.mu_ = means_init
                if self.covariance == "diag":
                    self.var_ = np.tile(np.var(U0, axis=0) + self.reg_covar, (self.K, 1))
                else:
                    base = np.cov(U0.T, bias=True)
                    base[0, 0] += self.reg_covar
                    base[1, 1] += self.reg_covar
                    self.cov_ = np.tile(base[None, :, :], (self.K, 1, 1))

        prev_ll = -np.inf

        if use_labels:
            self.freeze_theta_ = bool(freeze_theta)
            if self.freeze_theta_:
                eps = float(np.clip(self.manual_eps, self.theta_eps, 0.5))
                self.theta_edu_ = np.array([eps, 1.0 - eps, eps, 1.0 - eps], dtype=np.float64)
                self.theta_brdu_ = np.array([eps, eps, 1.0 - eps, 1.0 - eps], dtype=np.float64)
            else:
                self.theta_edu_ = np.full((self.K,), 0.5, dtype=np.float64)
                self.theta_brdu_ = np.full((self.K,), 0.5, dtype=np.float64)

        ll_trace: list[float] = []
        for it in range(self.max_iter):
            U = self.transform_to_u(X, batch_idx)
            resp, _ = self._e_step(U, batch_idx, y_edu=y_edu, y_brdu=y_brdu)

            if learn_affine:
                if self.covariance != "diag":
                    raise ValueError("learn_affine is only supported for covariance='diag'")
                # (A) update alpha/beta per batch
                for b in range(B):
                    if b == self.ref_batch_:
                        continue
                    idx = np.where(batch_idx == b)[0]
                    nb = idx.size
                    if nb == 0:
                        continue
                    g_b = float(np.clip(self.gamma_[b], self.min_gamma, np.inf))

                    for d in range(2):
                        xbd = X[idx, d]
                        rbd = resp[idx, :]
                        mu_d = self.mu_[:, d]
                        var_d = np.clip(self.var_[:, d], self.reg_covar, np.inf)

                        w = rbd / (g_b * var_d[None, :])
                        W = float(w.sum())
                        if (not np.isfinite(W)) or W < 1e-12:
                            continue

                        A = float((w * xbd[:, None]).sum())
                        w_sum_k = w.sum(axis=0)
                        Bmu = float((w_sum_k * mu_d).sum())

                        a0 = A / W
                        b0 = Bmu / W

                        c = xbd - a0
                        s_i = w.sum(axis=1)
                        C2 = float((s_i * (c**2)).sum())

                        Dk = b0 - mu_d
                        t_i = (w * Dk[None, :]).sum(axis=1)
                        C1 = float((c * t_i).sum())

                        disc = C1 * C1 + 4.0 * nb * C2
                        beta_bd = (C1 + math.sqrt(max(disc, 0.0))) / (2.0 * nb)
                        beta_bd = float(np.clip(beta_bd, self.min_beta, self.max_beta))
                        alpha_bd = float(a0 - beta_bd * b0)

                        self.beta_[b, d] = beta_bd
                        self.alpha_[b, d] = alpha_bd

                self.alpha_[self.ref_batch_] = 0.0
                self.beta_[self.ref_batch_] = 1.0

            # (B) update mixture params in u-space
            U = self.transform_to_u(X, batch_idx)
            gamma_obs = np.clip(self.gamma_[batch_idx], self.min_gamma, np.inf)
            inv_gamma = 1.0 / gamma_obs

            Nk = resp.sum(axis=0) + 1e-12
            self.pi_ = Nk / float(N)

            w_rg = resp * inv_gamma[:, None]
            denom = w_rg.sum(axis=0) + 1e-12
            self.mu_ = (w_rg.T @ U) / denom[:, None]

            diff = U[:, None, :] - self.mu_[None, :, :]  # (N,K,2)
            if self.covariance == "diag":
                self.var_ = (w_rg[:, :, None] * (diff**2)).sum(axis=0) / Nk[:, None]
                self.var_ = np.clip(self.var_ + self.reg_covar, self.reg_covar, np.inf)
            else:
                # cov_k = (1/Nk) sum_i (r_ik/gamma_i) (u-mu)(u-mu)^T
                self.cov_ = np.empty((self.K, 2, 2), dtype=np.float64)
                for k in range(self.K):
                    w = w_rg[:, k]
                    outer = diff[:, k, :, None] * diff[:, k, None, :]
                    cov_k = outer * w[:, None, None]
                    cov_k = np.sum(cov_k, axis=0) / float(Nk[k])
                    cov_k[0, 0] += self.reg_covar
                    cov_k[1, 1] += self.reg_covar
                    self.cov_[k] = cov_k

            if use_labels and (not self.freeze_theta_):
                nk = Nk
                self.theta_edu_ = (resp.T @ y_edu) / nk
                self.theta_brdu_ = (resp.T @ y_brdu) / nk
                self.theta_edu_ = np.clip(self.theta_edu_, self.theta_eps, 1.0 - self.theta_eps)
                self.theta_brdu_ = np.clip(self.theta_brdu_, self.theta_eps, 1.0 - self.theta_eps)

            # (C) update gamma per batch
            if self.covariance == "diag":
                mahal = np.sum((diff**2) / self.var_[None, :, :], axis=2)
            else:
                cov = self.cov_[None, :, :, :]  # (1,K,2,2)
                a = cov[:, :, 0, 0]
                b = cov[:, :, 0, 1]
                c = cov[:, :, 1, 1]
                det = a * c - b * b
                det = np.clip(det, self.reg_covar * self.reg_covar, np.inf)
                inv_00 = c / det
                inv_01 = -b / det
                inv_11 = a / det
                dx = diff[:, :, 0]
                dy = diff[:, :, 1]
                mahal = inv_00 * dx * dx + 2.0 * inv_01 * dx * dy + inv_11 * dy * dy  # (N,K)
            a = np.sum(resp * mahal, axis=1)
            for b in range(B):
                if b == self.ref_batch_:
                    continue
                idx = np.where(batch_idx == b)[0]
                nb = idx.size
                if nb == 0:
                    continue
                gamma_b = float(a[idx].sum() / (2.0 * nb))
                self.gamma_[b] = float(np.clip(gamma_b, self.min_gamma, self.max_gamma))
            self.gamma_[self.ref_batch_] = 1.0

            U = self.transform_to_u(X, batch_idx)
            _, log_norm_u = self._e_step(U, batch_idx, y_edu=y_edu, y_brdu=y_brdu)
            jac = -np.sum(np.log(self.beta_[batch_idx]), axis=1)
            ll = float(np.sum(log_norm_u + jac))
            ll_trace.append(ll)
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"iter {it:3d}  loglik={ll:.3f}")

            if np.isfinite(prev_ll) and abs(ll - prev_ll) <= self.tol * (1.0 + abs(prev_ll)):
                break
            prev_ll = ll

        self.ll_trace_ = ll_trace
        self.resp_ = self.predict_proba(X, batch_idx)
        return self


class BatchAwareGMM1D:
    """
    Batch-aware 1D Gaussian mixture with K components and per-batch nuisance parameters.

    Observed x_i (scalar) in batch b is modeled as:
        x = alpha_b + beta_b * u
        u | class=k, batch=b ~ N(mu_k, gamma_b * var_k)

    Mixing proportions are shared globally across batches (pi_k).

    Identifiability:
      A reference batch is fixed to alpha=0, beta=1, gamma=1.
    """

    def __init__(
        self,
        *,
        n_components: int,
        pi_prior: float = 1e-2,
        affine: str = "shift",
        variance: str = "component",
        reg_covar: float = 1e-3,
        max_iter: int = 200,
        tol: float = 1e-5,
        random_state: int = 0,
        min_beta: float = 1e-3,
        max_beta: float = 5.0,
        min_gamma: float = 1e-3,
        max_gamma: float = 50.0,
        verbose: bool = False,
    ) -> None:
        K = int(n_components)
        if K < 2:
            raise ValueError("n_components must be >= 2.")
        self.K = K
        self.pi_prior = float(pi_prior)
        if self.pi_prior < 0:
            raise ValueError("pi_prior must be >= 0.")
        affine = str(affine).strip().lower()
        if affine not in {"shift", "shift_scale"}:
            raise ValueError("affine must be 'shift' or 'shift_scale'.")
        self.affine = affine
        variance = str(variance).strip().lower()
        if variance not in {"component", "tied"}:
            raise ValueError("variance must be 'component' or 'tied'.")
        self.variance = variance
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)
        self.verbose = bool(verbose)

    def transform_to_u(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1)
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        alpha = self.alpha_[batch_idx]
        beta = self.beta_[batch_idx]
        return (X - alpha) / beta

    def _e_step(self, U: np.ndarray, batch_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Returns (resp, log_norm_per_obs) where log_norm is log p(u_i).
        U = np.asarray(U, dtype=np.float64).reshape(-1)
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        gamma_obs = self.gamma_[batch_idx]  # (N,)
        diff = U[:, None] - self.mu_[None, :]  # (N,K)
        g = gamma_obs[:, None]  # (N,1)
        var = self.var_[None, :]  # (1,K)
        pi = self.pi_[None, :]  # (1,K)

        log_det = np.log(2.0 * np.pi * g * var)  # (N,K)
        quad = (diff**2) / (g * var)  # (N,K)
        log_p = -0.5 * (log_det + quad) + np.log(pi + 1e-300)  # (N,K)

        log_norm = logsumexp(log_p, axis=1, keepdims=True)  # (N,1)
        resp = np.exp(log_p - log_norm)  # (N,K)
        return resp, log_norm[:, 0]

    def score_samples(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        # log p(x_i) in observed space, including Jacobian.
        X = np.asarray(X, dtype=np.float64).reshape(-1)
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        U = self.transform_to_u(X, batch_idx)
        _, log_norm_u = self._e_step(U, batch_idx)  # log p(u_i)
        jac = -np.log(self.beta_[batch_idx])
        return log_norm_u + jac

    def predict_proba(self, X: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1)
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        U = self.transform_to_u(X, batch_idx)
        resp, _ = self._e_step(U, batch_idx)
        return resp

    def component_params_in_x_space(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (m, var_x):
          m[b,k] = alpha_b + beta_b * mu_k
          var_x[b,k] = beta_b^2 * (gamma_b * var_k)
        """
        B = int(self.alpha_.shape[0])
        m = self.alpha_[:, None] + self.beta_[:, None] * self.mu_[None, :]
        var_x = (self.beta_[:, None] ** 2) * (self.gamma_[:, None] * self.var_[None, :])
        if m.shape != (B, self.K) or var_x.shape != (B, self.K):
            raise ValueError("unexpected component params shapes")
        return m, var_x

    def fit(
        self,
        X: np.ndarray,
        batch_idx: np.ndarray,
        *,
        ref_batch: int | None = None,
        init: str = "quantile",
        hard: np.ndarray | None = None,
        learn_affine: bool = False,
        affine_init_mask: np.ndarray | None = None,
        fit_mask: np.ndarray | None = None,
    ) -> BatchAwareGMM1D:
        X_full = np.asarray(X, dtype=np.float64).reshape(-1)
        batch_idx_full = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        if batch_idx_full.shape != (X_full.shape[0],):
            raise ValueError("batch_idx must be (N,) and match X length.")

        B = int(batch_idx_full.max()) + 1
        if B < 1:
            raise ValueError("no batches found")

        hard_full: np.ndarray | None
        if hard is None:
            hard_full = None
        else:
            hard_full = np.asarray(hard, dtype=bool).reshape(-1)
            if hard_full.shape != (X_full.shape[0],):
                raise ValueError("hard must be (N,) and match X length.")

        if fit_mask is None:
            X = X_full
            batch_idx = batch_idx_full
            hard = hard_full
        else:
            fit_mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
            if fit_mask.shape != (X_full.shape[0],):
                raise ValueError("fit_mask must be (N,) and match X length.")
            if not np.any(fit_mask):
                raise ValueError("fit_mask selects zero rows.")
            X = X_full[fit_mask]
            batch_idx = batch_idx_full[fit_mask]
            hard = None if hard_full is None else hard_full[fit_mask]
            counts_fit = np.bincount(batch_idx, minlength=B)
            bad = np.where(counts_fit < 50)[0]
            if bad.size:
                examples = ", ".join(str(int(i)) for i in bad[:8])
                raise ValueError(
                    f"too few rows after fit_mask in some batches (need >=50). Example batch_idx: {examples}"
                )

        N = int(X.shape[0])

        if ref_batch is None:
            counts = np.bincount(batch_idx, minlength=B)
            ref_batch = int(np.argmax(counts))
        self.ref_batch_ = int(ref_batch)

        if affine_init_mask is not None:
            affine_init_mask = np.asarray(affine_init_mask, dtype=bool).reshape(-1)
            if affine_init_mask.shape != (N,):
                raise ValueError("affine_init_mask must be (N,) and match X length.")
            if not np.any(affine_init_mask):
                raise ValueError("affine_init_mask selects zero rows.")

        alpha = np.zeros((B,), dtype=np.float64)
        beta = np.ones((B,), dtype=np.float64)
        gamma = np.ones((B,), dtype=np.float64)

        ref_sel = batch_idx == self.ref_batch_
        if affine_init_mask is not None:
            ref_sel = ref_sel & affine_init_mask
        X_ref = X[ref_sel]
        if X_ref.size < 50:
            raise ValueError(
                f"too few rows for affine init in reference batch={self.ref_batch_} (n={int(X_ref.size)}); "
                "use a broader affine init mask or disable masked affine init"
            )
        med_ref = float(np.median(X_ref))
        if self.affine == "shift_scale":
            mad_ref = float(np.median(np.abs(X_ref - med_ref)) + 1e-9)

        for b in range(B):
            if b == self.ref_batch_:
                continue
            sel = batch_idx == b
            if affine_init_mask is not None:
                sel = sel & affine_init_mask
            Xb = X[sel]
            if Xb.size == 0:
                continue
            if Xb.size < 50:
                raise ValueError(
                    f"too few rows for affine init in batch={b} (n={int(Xb.size)}); "
                    "use a broader affine init mask or disable masked affine init"
                )
            med_b = float(np.median(Xb))
            if self.affine == "shift_scale":
                mad_b = float(np.median(np.abs(Xb - med_b)) + 1e-9)
                beta[b] = float(np.clip(mad_b / mad_ref, self.min_beta, self.max_beta))
                alpha[b] = float(med_b - beta[b] * med_ref)
            else:
                beta[b] = 1.0
                alpha[b] = float(med_b - med_ref)

        alpha[self.ref_batch_] = 0.0
        beta[self.ref_batch_] = 1.0
        gamma[self.ref_batch_] = 1.0
        self.alpha_ = alpha
        self.beta_ = beta
        self.gamma_ = gamma

        U0 = self.transform_to_u(X, batch_idx)
        init = str(init).strip().lower()
        if init not in {"quantile", "tail_quantile", "manual_seed"}:
            raise ValueError(f"Unknown init mode: {init!r}")
        if init == "manual_seed" and hard is None:
            raise ValueError("init='manual_seed' requires hard labels to be provided.")

        if init == "manual_seed":
            hard = np.asarray(hard, dtype=bool).reshape(-1)
            n_pos = int(np.sum(hard))
            n_neg = int(np.sum(~hard))
            if n_pos < 50 or n_neg < 50:
                raise ValueError(f"manual_seed requires >=50 pos and >=50 neg (got pos={n_pos}, neg={n_neg}).")
            U_neg = U0[~hard]
            U_pos = U0[hard]

            q = np.linspace(0.15, 0.95, num=max(self.K - 1, 1))
            pos_means = np.quantile(U_pos, q).reshape(-1)
            mu = np.concatenate([[float(np.mean(U_neg))], pos_means]).astype(np.float64, copy=False)
            if mu.size != self.K:
                raise RuntimeError("internal error: unexpected manual_seed mu size")
            self.mu_ = mu

            var_neg = float(np.var(U_neg) + self.reg_covar)
            var_pos = float(np.var(U_pos) + self.reg_covar)
            var = np.concatenate([[var_neg], np.full(self.K - 1, var_pos, dtype=np.float64)]).astype(
                np.float64, copy=False
            )
            if self.variance == "tied":
                var = np.full((self.K,), float(np.mean(var)), dtype=np.float64)
            self.var_ = np.clip(var, self.reg_covar, np.inf)

            frac_pos = float(n_pos) / float(n_pos + n_neg)
            pi = np.concatenate(
                [[1.0 - frac_pos], np.full(self.K - 1, frac_pos / float(self.K - 1), dtype=np.float64)],
            ).astype(np.float64, copy=False)
            self.pi_ = np.clip(pi, 1e-12, np.inf)
            self.pi_ = self.pi_ / float(np.sum(self.pi_))
        else:
            if init == "quantile":
                q = np.linspace(0.15, 0.85, num=self.K)
            else:
                t = np.linspace(0.0, 1.0, num=self.K)
                q = 0.15 + (0.995 - 0.15) * (t**2)
            means_init = np.quantile(U0, q, axis=0).reshape(-1, 1)
            try:
                from sklearn.mixture import GaussianMixture

                gmm = GaussianMixture(
                    n_components=self.K,
                    covariance_type="full",
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    means_init=means_init,
                    n_init=3,
                    max_iter=200,
                )
                gmm.fit(U0.reshape(-1, 1))
                pi0 = np.asarray(gmm.weights_, dtype=np.float64).copy()
                self.pi_ = pi0
                self.mu_ = np.asarray(gmm.means_, dtype=np.float64).reshape(self.K).copy()
                cov = np.asarray(gmm.covariances_, dtype=np.float64)
                if cov.ndim == 3:
                    var = cov[:, 0, 0].reshape(self.K)
                else:
                    var = cov.reshape(self.K)
                var = np.clip(var + self.reg_covar, self.reg_covar, np.inf).astype(np.float64, copy=False)
                if self.variance == "tied":
                    var = np.full((self.K,), float(np.mean(var)), dtype=np.float64)
                self.var_ = var
            except Exception:
                self.pi_ = np.ones((self.K,), dtype=np.float64) / float(self.K)
                self.mu_ = means_init.reshape(self.K)
                var0 = float(np.var(U0) + self.reg_covar)
                self.var_ = np.full((self.K,), var0, dtype=np.float64)

        prev_ll = -np.inf
        ll_trace: list[float] = []
        for it in range(self.max_iter):
            U = self.transform_to_u(X, batch_idx)
            resp, log_norm_u = self._e_step(U, batch_idx)

            if learn_affine:
                for b in range(B):
                    if b == self.ref_batch_:
                        continue
                    idx = np.where(batch_idx == b)[0]
                    nb = idx.size
                    if nb == 0:
                        continue

                    g_b = float(np.clip(self.gamma_[b], self.min_gamma, np.inf))
                    xb = X[idx]
                    rb = resp[idx, :]
                    mu = self.mu_
                    var = np.clip(self.var_, self.reg_covar, np.inf)

                    w = rb / (g_b * var[None, :])
                    W = float(w.sum())
                    if (not np.isfinite(W)) or W < 1e-12:
                        continue

                    if self.affine == "shift":
                        alpha_b = float((w * (xb[:, None] - mu[None, :])).sum() / W)
                        self.alpha_[b] = alpha_b
                        self.beta_[b] = 1.0
                    else:
                        A = float((w * xb[:, None]).sum())
                        w_sum_k = w.sum(axis=0)
                        Bmu = float((w_sum_k * mu).sum())

                        a0 = A / W
                        b0 = Bmu / W

                        c = xb - a0
                        s_i = w.sum(axis=1)
                        C2 = float((s_i * (c**2)).sum())

                        Dk = b0 - mu
                        t_i = (w * Dk[None, :]).sum(axis=1)
                        C1 = float((c * t_i).sum())

                        disc = C1 * C1 + 4.0 * nb * C2
                        beta_b = (C1 + math.sqrt(max(disc, 0.0))) / (2.0 * nb)
                        beta_b = float(np.clip(beta_b, self.min_beta, self.max_beta))
                        alpha_b = float(a0 - beta_b * b0)

                        self.beta_[b] = beta_b
                        self.alpha_[b] = alpha_b

                self.alpha_[self.ref_batch_] = 0.0
                self.beta_[self.ref_batch_] = 1.0

            # (B) update mixture params in u-space
            U = self.transform_to_u(X, batch_idx)
            gamma_obs = np.clip(self.gamma_[batch_idx], self.min_gamma, np.inf)
            inv_gamma = 1.0 / gamma_obs
            Nk = resp.sum(axis=0) + 1e-12  # (K,)
            w_rg = resp * inv_gamma[:, None]
            denom = w_rg.sum(axis=0) + 1e-12
            self.mu_ = (w_rg.T @ U) / denom
            diff = U[:, None] - self.mu_[None, :]
            if self.variance == "component":
                self.var_ = (w_rg * (diff**2)).sum(axis=0) / Nk
                self.var_ = np.clip(self.var_ + self.reg_covar, self.reg_covar, np.inf)
            else:
                var = float((w_rg * (diff**2)).sum() / float(N))
                var = float(np.clip(var + self.reg_covar, self.reg_covar, np.inf))
                self.var_ = np.full((self.K,), var, dtype=np.float64)

            # Update global mixing proportions pi_k
            Nk_all = resp.sum(axis=0) + float(self.pi_prior)
            self.pi_ = Nk_all / float(np.sum(Nk_all))

            # (C) update gamma per batch
            mahal = (diff**2) / self.var_[None, :]
            a = np.sum(resp * mahal, axis=1)
            for b in range(B):
                if b == self.ref_batch_:
                    continue
                idx = np.where(batch_idx == b)[0]
                nb = idx.size
                if nb == 0:
                    continue
                gamma_b = float(a[idx].sum() / (1.0 * nb))
                self.gamma_[b] = float(np.clip(gamma_b, self.min_gamma, self.max_gamma))
            self.gamma_[self.ref_batch_] = 1.0

            jac = -np.log(self.beta_[batch_idx])
            ll = float(np.sum(log_norm_u + jac))
            ll_trace.append(ll)
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"iter {it:3d}  loglik={ll:.3f}")

            if np.isfinite(prev_ll) and abs(ll - prev_ll) <= self.tol * (1.0 + abs(prev_ll)):
                break
            prev_ll = ll

        self.ll_trace_ = ll_trace
        return self


@dataclass(frozen=True)
class QcInputs:
    X: np.ndarray  # (N,2)
    u: np.ndarray  # (N,2)
    batch_idx: np.ndarray  # (N,)
    batch_levels: list[str]
    classes: dict[str, np.ndarray]  # name -> (N,)
    class_names: list[str]
    p_max: np.ndarray  # (N,)
    entropy: np.ndarray  # (N,)
    ll_x: np.ndarray  # (N,)


@dataclass(frozen=True)
class Gmm1DByBatchFit:
    p_pos: np.ndarray  # (N,)
    ll_x: np.ndarray  # (N,)
    u_zneg: np.ndarray  # (N,) z-score vs negative component in x-space
    params_by_batch: pd.DataFrame


def _fit_1d_gmm_by_batch(
    *,
    x: np.ndarray,
    batch_idx: np.ndarray,
    batch_levels: list[str],
    channel: str,
    n_components: int,
    pos_top_k: int,
    reg_covar: float,
    max_iter: int,
    seed: int,
) -> Gmm1DByBatchFit:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as e:
        raise RuntimeError("scikit-learn is required for the 1D GMM fit.") from e

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    bidx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
    if x.shape != bidx.shape:
        raise ValueError("x and batch_idx must have the same shape.")
    if not np.isfinite(x).all():
        raise ValueError("x contains non-finite values; filter or impute before fitting.")
    K = int(n_components)
    if K < 2:
        raise ValueError("n_components must be >= 2.")
    top_k = int(pos_top_k)
    if top_k < 1 or top_k >= K:
        raise ValueError("pos_top_k must satisfy 1 <= pos_top_k < n_components.")

    B = int(len(batch_levels))
    n = int(x.size)
    p_pos = np.full(n, np.nan, dtype=np.float64)
    ll_x = np.full(n, np.nan, dtype=np.float64)
    u_zneg = np.full(n, np.nan, dtype=np.float64)

    rows: list[dict[str, object]] = []
    for b in range(B):
        idx_b = np.where(bidx == b)[0]
        if idx_b.size < 50:
            raise ValueError(f"Batch {batch_levels[b]!r} too small for 1D GMM (n={int(idx_b.size)}).")
        xb = x[idx_b]
        if float(np.nanstd(xb)) < 1e-8:
            raise ValueError(f"Batch {batch_levels[b]!r} has near-zero variance in {channel}.")

        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            reg_covar=float(reg_covar),
            random_state=int(seed),
            n_init=3,
            max_iter=int(max_iter),
        )
        gmm.fit(xb.reshape(-1, 1))

        means = np.asarray(gmm.means_, dtype=np.float64).reshape(K)
        cov = np.asarray(gmm.covariances_, dtype=np.float64)
        if cov.ndim == 3:
            var = cov[:, 0, 0].reshape(K)
        else:
            var = cov.reshape(K)
        weights = np.asarray(gmm.weights_, dtype=np.float64).reshape(K)

        comp_pos = int(np.argmax(means))
        comp_neg = int(np.argmin(means))
        order = np.argsort(means)
        pos_comps = np.sort(order[-top_k:]).astype(int).tolist()

        resp = gmm.predict_proba(xb.reshape(-1, 1)).astype(np.float64, copy=False)
        p_pos[idx_b] = resp[:, pos_comps].sum(axis=1)
        ll_x[idx_b] = gmm.score_samples(xb.reshape(-1, 1)).astype(np.float64, copy=False)

        denom = float(np.sqrt(max(float(var[comp_neg]), 1e-12)))
        u_zneg[idx_b] = (xb - float(means[comp_neg])) / denom

        rows.append(
            {
                "batch_idx": b,
                "batch_level": batch_levels[b],
                "channel": channel,
                "n_components": K,
                "pos_top_k": top_k,
                "n_cells": int(idx_b.size),
                **{f"weight{k}": float(weights[k]) for k in range(K)},
                **{f"mean{k}": float(means[k]) for k in range(K)},
                **{f"var{k}": float(var[k]) for k in range(K)},
                "comp_pos": comp_pos,
                "comp_neg": comp_neg,
                "pos_comps": ",".join(str(k) for k in pos_comps),
                "converged": bool(getattr(gmm, "converged_", True)),
                "n_iter": int(getattr(gmm, "n_iter_", -1)),
            }
        )

    params_by_batch = pd.DataFrame(rows)
    if not np.isfinite(p_pos).all() or not np.isfinite(ll_x).all() or not np.isfinite(u_zneg).all():
        raise RuntimeError("Internal error: non-finite 1D GMM outputs.")

    return Gmm1DByBatchFit(p_pos=p_pos, ll_x=ll_x, u_zneg=u_zneg, params_by_batch=params_by_batch)


def compute_qc_inputs(
    *,
    X: np.ndarray,
    batch_idx: np.ndarray,
    batch_levels: list[str],
    model: BatchAwareDiagGMM2D,
) -> QcInputs:
    resp = model.predict_proba(X, batch_idx)
    mapping = model.map_components_to_classes()

    classes: dict[str, np.ndarray] = {}
    for cls in ["neg", "EdU_only", "BrdU_only", "double"]:
        comps = mapping[cls]
        classes[cls] = resp[:, comps].sum(axis=1).astype(np.float64, copy=False)

    class_names = ["neg", "EdU_only", "BrdU_only", "double"]
    P = np.stack([classes[c] for c in class_names], axis=1)
    p_max = np.max(P, axis=1)
    H = entropy(P, axis=1)
    u = model.transform_to_u(X, batch_idx)
    ll_x = model.score_samples(X, batch_idx)

    return QcInputs(
        X=X,
        u=u,
        batch_idx=batch_idx,
        batch_levels=batch_levels,
        classes=classes,
        class_names=class_names,
        p_max=p_max,
        entropy=H,
        ll_x=ll_x,
    )


def _ellipse_xy(*, mean: np.ndarray, cov: np.ndarray, n: int = 200, nsig: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError("cov must be (2,2)")
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, np.inf)
    r = float(nsig) * np.sqrt(w)
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2,n)
    pts = mean.reshape(2, 1) + (v @ (r.reshape(2, 1) * circle))
    return pts[0], pts[1]


def _class_colors() -> dict[str, str]:
    return {
        "neg": "#bdbdbd",
        "EdU_only": "#1f77b4",
        "BrdU_only": "#ff7f0e",
        "double": "#d62728",
    }


def plot_global_hexbin_x_calls(
    *,
    out_png: Path,
    qc: QcInputs,
    batch_levels: list[str],
    max_scatter: int,
    seed: int,
    edu_map_thr: float,
    brdu_map_thr: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.X.shape[0], max_scatter, rng=rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200, constrained_layout=True)

    colors = _class_colors()
    pE = (qc.classes["EdU_only"] + qc.classes["double"])[idx]
    pB = (qc.classes["BrdU_only"] + qc.classes["double"])[idx]
    e_pos = pE >= float(edu_map_thr)
    b_pos = pB >= float(brdu_map_thr)
    quad = e_pos.astype(np.int8) + 2 * b_pos.astype(np.int8)  # 0=neg,1=edu_only,2=brdu_only,3=double
    cls_name = np.asarray(qc.class_names, dtype=object)[quad]

    for name in qc.class_names:
        m = cls_name == name
        if not np.any(m):
            continue
        ax.scatter(
            qc.X[idx[m], 0],
            qc.X[idx[m], 1],
            s=0.2,
            c=colors[name],
            linewidths=0.0,
            alpha=1.0,
            label=name,
        )

    ax.set_xlabel("edu_x (observed)")
    ax.set_ylabel("brdu_x (observed)")
    ax.set_title("Observed space (x): MAP quadrant assignments")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", ms=6, lw=0, markerfacecolor="none", markeredgecolor=colors[name], label=name)
        for name in qc.class_names
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_png)
    plt.close(fig)


def plot_group_facets_x_calls(
    *,
    out_png: Path,
    qc: QcInputs,
    group_codes: np.ndarray,
    group_levels: list[str],
    max_groups: int,
    max_cells_per_group: int,
    seed: int,
    edu_map_thr: float,
    brdu_map_thr: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_codes = np.asarray(group_codes, dtype=np.int64).reshape(-1)
    if group_codes.shape[0] != qc.X.shape[0]:
        raise ValueError("group_codes must have same length as qc.X")

    rng = np.random.default_rng(seed)
    counts = np.bincount(group_codes, minlength=int(len(group_levels))).astype(np.int64, copy=False)
    order = np.argsort(-counts)  # descending by N
    order = order[counts[order] > 0]
    if order.size == 0:
        raise ValueError("No non-empty groups to plot.")
    order = order[: int(max_groups)]

    colors = _class_colors()
    ncols = 4
    nrows = int(math.ceil(int(order.size) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=200)
    axs2 = np.asarray(axs).reshape(-1)

    for j in range(nrows * ncols):
        ax = axs2[j]
        if j >= int(order.size):
            ax.axis("off")
            continue

        g = int(order[j])
        idx_g = np.where(group_codes == g)[0]
        if idx_g.size == 0:
            ax.axis("off")
            continue

        idx_g = idx_g[_sample_indices(idx_g.size, max_cells_per_group, rng=rng)]
        pts = qc.X[idx_g]

        pE = (qc.classes["EdU_only"] + qc.classes["double"])[idx_g]
        pB = (qc.classes["BrdU_only"] + qc.classes["double"])[idx_g]
        e_pos = pE >= float(edu_map_thr)
        b_pos = pB >= float(brdu_map_thr)
        quad = e_pos.astype(np.int8) + 2 * b_pos.astype(np.int8)
        cls_name = np.asarray(qc.class_names, dtype=object)[quad]
        for name in qc.class_names:
            m = cls_name == name
            if not np.any(m):
                continue
            ax.scatter(
                pts[m, 0],
                pts[m, 1],
                s=0.2,
                c=colors[name],
                linewidths=0.0,
                alpha=1.0,
            )

        ax.set_title(f"{group_levels[g]}\n(n={int(idx_g.size):,})", fontsize=9)
        ax.set_xlabel("edu_x")
        ax.set_ylabel("brdu_x")

    fig.suptitle("Observed space (x) faceted by group (MAP quadrant assignments)", fontsize=12)
    fig.savefig(out_png)
    plt.close(fig)


def plot_global_hexbin_x(
    *,
    out_png: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
    batch_show: int,
    max_scatter: int,
    seed: int,
    draw_contours: bool = True,
    gridsize: int = 200,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.X.shape[0], max_scatter, rng=rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200, constrained_layout=True)
    hb = ax.hexbin(qc.X[:, 0], qc.X[:, 1], gridsize=gridsize, bins="log", mincnt=1, cmap="Greys")
    fig.colorbar(hb, ax=ax, pad=0.01, label="log10(count)")

    colors = _class_colors()
    P = np.stack([qc.classes[c] for c in qc.class_names], axis=1)
    cls_idx = np.argmax(P[idx], axis=1)
    cls_name = np.asarray(qc.class_names, dtype=object)[cls_idx]
    pmax = qc.p_max[idx].astype(np.float64, copy=False)
    norm = plt.Normalize(0.5, 1.0)
    cmap = plt.get_cmap("viridis")
    for name in qc.class_names:
        m = cls_name == name
        if not np.any(m):
            continue
        ax.scatter(
            qc.X[idx[m], 0],
            qc.X[idx[m], 1],
            s=0.2,
            c=cmap(norm(pmax[m])),
            edgecolors=colors[name],
            linewidths=0.25,
            alpha=0.2,
            label=name,
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.01, label="p_max (max posterior)")

    m_x, var_x = model.component_params_in_x_space()
    means = m_x[batch_show]
    v = var_x[batch_show]
    mapping = model.map_components_to_classes()
    comp_to_cls = {}
    for cls, comps in mapping.items():
        for k in comps:
            comp_to_cls[int(k)] = cls
    for k in range(model.K):
        cls = comp_to_cls.get(int(k), "neg")
        ax.scatter(means[k, 0], means[k, 1], s=60, c=colors[cls], edgecolors="white", linewidths=0.8, zorder=10)
        if v.shape == (model.K, 2):
            cov = np.diag(v[k].astype(np.float64, copy=False))
        else:
            cov = v[k].astype(np.float64, copy=False)
        ex, ey = _ellipse_xy(mean=means[k], cov=cov)
        ax.plot(ex, ey, lw=1.2, color=colors[cls], alpha=0.9)

    if draw_contours:
        # Diagnostic: iso-posterior contours (per quadrant/class) at P(class|x,batch)=0.5.
        # Useful to visualize the model's joint decision boundaries, but can look like a
        # "curved threshold" (because it is a joint classifier, not two independent cutoffs).
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        gx = np.linspace(float(xlim[0]), float(xlim[1]), 250, endpoint=True)
        gy = np.linspace(float(ylim[0]), float(ylim[1]), 250, endpoint=True)
        XX, YY = np.meshgrid(gx, gy)
        grid = np.column_stack([XX.reshape(-1), YY.reshape(-1)]).astype(np.float64, copy=False)
        batch = np.full(grid.shape[0], int(batch_show), dtype=np.int64)
        resp = model.predict_proba(grid, batch)
        mapping = model.map_components_to_classes()
        for name in qc.class_names:
            comps = mapping.get(name, [])
            if not comps:
                continue
            z = resp[:, comps].sum(axis=1).reshape(YY.shape)
            z_min = float(np.min(z))
            z_max = float(np.max(z))
            if not (z_min <= 0.5 <= z_max):
                continue
            ax.contour(
                XX,
                YY,
                z,
                levels=[0.5],
                colors=[colors[name]],
                linewidths=1.6,
                linestyles="--",
                alpha=0.9,
                zorder=9,
            )

    ax.set_xlabel("edu_x (observed)")
    ax.set_ylabel("brdu_x (observed)")
    ax.set_title(
        "Observed space (x): hexbin + subsample (edge=class, fill=p_max) "
        f"(batch={qc.batch_levels[batch_show]})"
    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", ms=6, lw=0, markerfacecolor="none", markeredgecolor=colors[name], label=name)
        for name in qc.class_names
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_png)
    plt.close(fig)


def plot_global_hexbin_u(
    *,
    out_png: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
    max_scatter: int,
    seed: int,
    gridsize: int = 200,
    use_gamma: bool = True,
    batch_show: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.u.shape[0], max_scatter, rng=rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200, constrained_layout=True)
    hb = ax.hexbin(qc.u[:, 0], qc.u[:, 1], gridsize=gridsize, bins="log", mincnt=1, cmap="Greys")
    fig.colorbar(hb, ax=ax, pad=0.01, label="log10(count)")

    colors = _class_colors()
    P = np.stack([qc.classes[c] for c in qc.class_names], axis=1)
    cls_idx = np.argmax(P[idx], axis=1)
    cls_name = np.asarray(qc.class_names, dtype=object)[cls_idx]
    for name in qc.class_names:
        m = cls_name == name
        if not np.any(m):
            continue
        ax.scatter(qc.u[idx[m], 0], qc.u[idx[m], 1], s=3, alpha=0.35, c=colors[name], label=name, linewidths=0)

    mapping = model.map_components_to_classes()
    comp_to_cls = {}
    for cls, comps in mapping.items():
        for k in comps:
            comp_to_cls[int(k)] = cls

    gamma = 1.0
    if use_gamma and batch_show is not None:
        gamma = float(model.gamma_[int(batch_show)])
    for k in range(model.K):
        cls = comp_to_cls.get(int(k), "neg")
        ax.scatter(model.mu_[k, 0], model.mu_[k, 1], s=60, c=colors[cls], edgecolors="white", linewidths=0.8, zorder=10)
        if model.covariance == "diag":
            cov = np.diag((gamma * model.var_[k]).astype(np.float64, copy=False))
        else:
            cov = gamma * model.cov_[k]
        ex, ey = _ellipse_xy(mean=model.mu_[k], cov=cov)
        ax.plot(ex, ey, lw=1.2, color=colors[cls], alpha=0.9)

    # Diagnostic: iso-posterior contours at P(class|u,batch)=0.5. The model is defined on x;
    # we visualize in u by mapping u->x for a chosen batch.
    if batch_show is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        gx = np.linspace(float(xlim[0]), float(xlim[1]), 250, endpoint=True)
        gy = np.linspace(float(ylim[0]), float(ylim[1]), 250, endpoint=True)
        UU, VV = np.meshgrid(gx, gy)
        u_grid = np.column_stack([UU.reshape(-1), VV.reshape(-1)]).astype(np.float64, copy=False)
        b = int(batch_show)
        x_grid = model.alpha_[b].reshape(1, 2) + model.beta_[b].reshape(1, 2) * u_grid
        batch = np.full(x_grid.shape[0], b, dtype=np.int64)
        resp = model.predict_proba(x_grid, batch)
        mapping = model.map_components_to_classes()
        for name in qc.class_names:
            comps = mapping.get(name, [])
            if not comps:
                continue
            z = resp[:, comps].sum(axis=1).reshape(VV.shape)
            z_min = float(np.min(z))
            z_max = float(np.max(z))
            if not (z_min <= 0.5 <= z_max):
                continue
            ax.contour(
                UU,
                VV,
                z,
                levels=[0.5],
                colors=[colors[name]],
                linewidths=1.6,
                linestyles="--",
                alpha=0.9,
                zorder=9,
            )

    extra = ""
    if use_gamma and batch_show is not None:
        extra = f" (gamma for batch={qc.batch_levels[int(batch_show)]})"
    ax.set_xlabel("u_EdU (latent, batch-corrected)")
    ax.set_ylabel("u_BrdU (latent, batch-corrected)")
    ax.set_title(f"Latent space (u): hexbin + subsample colored by argmax posterior{extra}")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_png)
    plt.close(fig)


def plot_batch_facets(
    *,
    out_png: Path,
    qc_xy: np.ndarray,
    batch_idx: np.ndarray,
    batch_levels: list[str],
    ell_means: np.ndarray,
    ell_vars: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    gridsize: int = 120,
    max_cells_per_batch: int = 200_000,
    seed: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    B = int(len(batch_levels))
    ncols = 4
    nrows = int(math.ceil(B / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=200)
    axs2 = np.asarray(axs).reshape(-1)

    for b in range(nrows * ncols):
        ax = axs2[b]
        if b >= B:
            ax.axis("off")
            continue
        idx_b = np.where(batch_idx == b)[0]
        if idx_b.size == 0:
            ax.axis("off")
            continue
        idx_b = idx_b[_sample_indices(idx_b.size, max_cells_per_batch, rng=rng)]
        pts = qc_xy[idx_b]
        ax.hexbin(pts[:, 0], pts[:, 1], gridsize=gridsize, bins="log", mincnt=1, cmap="Greys")
        ax.set_title(f"{batch_levels[b]}\n(n={int(idx_b.size)})", fontsize=9)

        for k in range(ell_means.shape[1]):
            # Assume k->class mapping already applied by caller if needed; use consistent colors by k ordering.
            mean = ell_means[b, k]
            v = ell_vars[b, k]
            if v.shape == (2,):
                cov = np.diag(v.astype(np.float64, copy=False))
            else:
                cov = v.astype(np.float64, copy=False)
            ex, ey = _ellipse_xy(mean=mean, cov=cov)
            ax.plot(ex, ey, lw=1.2, color="black", alpha=0.65)
            ax.scatter(mean[0], mean[1], s=25, c="black", edgecolors="white", linewidths=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png)
    plt.close(fig)


def plot_group_facets_x(
    *,
    out_png: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
    group_codes: np.ndarray,
    group_levels: list[str],
    max_groups: int,
    max_cells_per_group: int,
    seed: int,
    draw_contours: bool = True,
    gridsize: int = 120,
    boundary_grid: int = 120,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_codes = np.asarray(group_codes, dtype=np.int64).reshape(-1)
    if group_codes.shape[0] != qc.X.shape[0]:
        raise ValueError("group_codes must have same length as qc.X")

    rng = np.random.default_rng(seed)

    counts = np.bincount(group_codes, minlength=int(len(group_levels))).astype(np.int64, copy=False)
    order = np.argsort(-counts)  # descending by N
    order = order[counts[order] > 0]
    if order.size == 0:
        raise ValueError("No non-empty groups to plot.")
    order = order[: int(max_groups)]

    colors = _class_colors()
    mapping = model.map_components_to_classes()
    comp_to_cls: dict[int, str] = {}
    for cls, comps in mapping.items():
        for k in comps:
            comp_to_cls[int(k)] = cls

    m_x, var_x = model.component_params_in_x_space()

    ncols = 4
    nrows = int(math.ceil(int(order.size) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=200)
    axs2 = np.asarray(axs).reshape(-1)

    for j in range(nrows * ncols):
        ax = axs2[j]
        if j >= int(order.size):
            ax.axis("off")
            continue

        g = int(order[j])
        idx_g = np.where(group_codes == g)[0]
        if idx_g.size == 0:
            ax.axis("off")
            continue

        batch_vals = np.unique(qc.batch_idx[idx_g])
        if batch_vals.size != 1:
            raise ValueError(
                f"Group {group_levels[g]!r} spans multiple batches ({batch_vals.tolist()}). "
                "For diagnostic faceting with per-batch contours, include the batch column (e.g. dataset) in the group key."
            )
        b = int(batch_vals[0])

        idx_g = idx_g[_sample_indices(idx_g.size, max_cells_per_group, rng=rng)]
        pts = qc.X[idx_g]
        ax.hexbin(pts[:, 0], pts[:, 1], gridsize=gridsize, bins="log", mincnt=1, cmap="Greys")

        P = np.stack([qc.classes[c] for c in qc.class_names], axis=1)
        cls_idx = np.argmax(P[idx_g], axis=1)
        cls_name = np.asarray(qc.class_names, dtype=object)[cls_idx]
        pmax = qc.p_max[idx_g].astype(np.float64, copy=False)
        norm = plt.Normalize(0.5, 1.0)
        cmap = plt.get_cmap("viridis")
        for name in qc.class_names:
            m = cls_name == name
            if not np.any(m):
                continue
            ax.scatter(
                pts[m, 0],
                pts[m, 1],
                s=6,
                c=cmap(norm(pmax[m])),
                edgecolors=colors[name],
                linewidths=0.22,
                alpha=0.9,
            )

        means = m_x[b]
        v = var_x[b]
        for k in range(model.K):
            cls = comp_to_cls.get(int(k), "neg")
            ax.scatter(means[k, 0], means[k, 1], s=25, c=colors[cls], edgecolors="white", linewidths=0.6, zorder=10)
            if v.shape == (model.K, 2):
                cov = np.diag(v[k].astype(np.float64, copy=False))
            else:
                cov = v[k].astype(np.float64, copy=False)
            ex, ey = _ellipse_xy(mean=means[k], cov=cov)
            ax.plot(ex, ey, lw=1.0, color=colors[cls], alpha=0.85)

        if draw_contours:
            # P(class|x,batch)=0.5 contours per class
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            gx = np.linspace(float(xlim[0]), float(xlim[1]), int(boundary_grid), endpoint=True)
            gy = np.linspace(float(ylim[0]), float(ylim[1]), int(boundary_grid), endpoint=True)
            XX, YY = np.meshgrid(gx, gy)
            grid = np.column_stack([XX.reshape(-1), YY.reshape(-1)]).astype(np.float64, copy=False)
            batch = np.full(grid.shape[0], b, dtype=np.int64)
            resp = model.predict_proba(grid, batch)
            for name in qc.class_names:
                comps = mapping.get(name, [])
                if not comps:
                    continue
                z = resp[:, comps].sum(axis=1).reshape(YY.shape)
                z_min = float(np.min(z))
                z_max = float(np.max(z))
                if not (z_min <= 0.5 <= z_max):
                    continue
                ax.contour(
                    XX,
                    YY,
                    z,
                    levels=[0.5],
                    colors=[colors[name]],
                    linewidths=1.2,
                    linestyles="--",
                    alpha=0.9,
                    zorder=9,
                )

        ax.set_title(f"{group_levels[g]}\n(batch={qc.batch_levels[b]}, n={int(idx_g.size):,})", fontsize=9)
        ax.set_xlabel("edu_x")
        ax.set_ylabel("brdu_x")

    fig.suptitle("Observed space (x) faceted by group (with 0.5 class contours)", fontsize=12)
    fig.savefig(out_png)
    plt.close(fig)


def plot_nuisance_params(*, out_dir: Path, model: BatchAwareDiagGMM2D, batch_levels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(batch_levels))
    labels = batch_levels

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(labels)), 4.5), dpi=200, constrained_layout=True)
    ax.plot(x, model.alpha_[:, 0], marker="o", ms=3, lw=1.2, label="alpha_EdU")
    ax.plot(x, model.alpha_[:, 1], marker="o", ms=3, lw=1.2, label="alpha_BrdU")
    ax.axhline(0.0, color="grey", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("alpha (shift)")
    ax.set_title("Per-batch additive shift (alpha_b)")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(out_dir / "nuisance_alpha.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(labels)), 4.5), dpi=200, constrained_layout=True)
    ax.plot(x, model.beta_[:, 0], marker="o", ms=3, lw=1.2, label="beta_EdU")
    ax.plot(x, model.beta_[:, 1], marker="o", ms=3, lw=1.2, label="beta_BrdU")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("beta (scale)")
    ax.set_title("Per-batch multiplicative scale (beta_b)")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(out_dir / "nuisance_beta.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(labels)), 4.5), dpi=200, constrained_layout=True)
    ax.bar(x, model.gamma_, color="#666666")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("gamma (variance inflation)")
    ax.set_title("Per-batch variance inflation (gamma_b)")
    fig.savefig(out_dir / "nuisance_gamma.png")
    plt.close(fig)


def plot_uncertainty_hists(*, out_dir: Path, qc: QcInputs, bins: int = 100) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200, constrained_layout=True)
    ax.hist(qc.p_max, bins=bins, color="#333333", alpha=0.9)
    ax.set_xlabel("p_max (max class posterior)")
    ax.set_ylabel("count")
    ax.set_title("Posterior certainty: p_max histogram")
    fig.savefig(out_dir / "hist_pmax.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200, constrained_layout=True)
    ax.hist(qc.entropy, bins=bins, color="#333333", alpha=0.9)
    ax.set_xlabel("entropy H")
    ax.set_ylabel("count")
    ax.set_title("Posterior uncertainty: entropy histogram")
    fig.savefig(out_dir / "hist_entropy.png")
    plt.close(fig)


def plot_uncertainty_scatter_u(
    *,
    out_png: Path,
    qc: QcInputs,
    max_scatter: int,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.u.shape[0], max_scatter, rng=rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=200, constrained_layout=True)
    sc = ax.scatter(
        qc.u[idx, 0],
        qc.u[idx, 1],
        s=4,
        c=qc.entropy[idx],
        cmap="viridis",
        alpha=np.clip(qc.p_max[idx], 0.05, 1.0),
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, pad=0.01, label="entropy H")
    ax.set_xlabel("u_EdU")
    ax.set_ylabel("u_BrdU")
    ax.set_title("Uncertainty in u-space (alpha=p_max, color=entropy)")
    fig.savefig(out_png)
    plt.close(fig)


def plot_loglik_vs_intensity(
    *,
    out_png: Path,
    qc: QcInputs,
    max_scatter: int,
    seed: int,
    worst_frac: float = 0.005,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.X.shape[0], max_scatter, rng=rng)
    s = qc.X[idx, 0] + qc.X[idx, 1]
    ll = qc.ll_x[idx]

    thr = float(np.quantile(qc.ll_x, worst_frac))
    bad = qc.ll_x <= thr
    bad_idx = idx[bad[idx]]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200, constrained_layout=True)
    ax.scatter(s, ll, s=4, alpha=0.25, c="#1f77b4", linewidths=0)
    if bad_idx.size:
        ax.scatter(qc.X[bad_idx, 0] + qc.X[bad_idx, 1], qc.ll_x[bad_idx], s=8, alpha=0.6, c="#d62728", linewidths=0)
    ax.set_xlabel("edu_x + brdu_x")
    ax.set_ylabel("log p(x_i)")
    ax.set_title(f"Log-likelihood vs intensity (highlight worst {100*worst_frac:.1f}%)")
    fig.savefig(out_png)
    plt.close(fig)


def plot_component_health(
    *,
    out_dir: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 4A: global pi_k and expected per-batch weights
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=200, constrained_layout=True)
    ax.bar(np.arange(model.K), model.pi_, color="#666666")
    ax.set_xlabel("component k")
    ax.set_ylabel("pi_k")
    ax.set_title("Mixture weights (global)")
    fig.savefig(out_dir / "mixture_weights_pi.png")
    plt.close(fig)

    resp = model.predict_proba(qc.X, qc.batch_idx)
    per_batch = []
    for b, name in enumerate(qc.batch_levels):
        idx = np.where(qc.batch_idx == b)[0]
        if idx.size == 0:
            continue
        w = resp[idx].mean(axis=0)
        per_batch.append({"batch": name, **{f"k{k}": float(w[k]) for k in range(model.K)}})
    per_batch_df = pd.DataFrame(per_batch)
    _write_tsv(out_dir / "expected_component_weights_by_batch.tsv", per_batch_df)

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(qc.batch_levels)), 4.5), dpi=200, constrained_layout=True)
    x = np.arange(len(per_batch_df))
    bottom = np.zeros_like(x, dtype=np.float64)
    for k in range(model.K):
        y = per_batch_df[f"k{k}"].to_numpy(dtype=np.float64)
        ax.bar(x, y, bottom=bottom, label=f"k{k}")
        bottom += y
    ax.set_xticks(x)
    ax.set_xticklabels(per_batch_df["batch"].tolist(), rotation=90, fontsize=7)
    ax.set_ylabel("E[r_ik] per batch")
    ax.set_title("Expected component weights per batch (stacked)")
    ax.legend(frameon=True, fontsize=7, ncols=4)
    fig.savefig(out_dir / "expected_component_weights_by_batch.png")
    plt.close(fig)

    # 4B: component means/vars in u-space
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=200, constrained_layout=True)
    colors = ["#111111", "#1f77b4", "#ff7f0e", "#d62728"]
    for k in range(model.K):
        ax.scatter(model.mu_[k, 0], model.mu_[k, 1], s=70, c=colors[k], edgecolors="white", linewidths=0.8, label=f"k{k}")
        if model.covariance == "diag":
            cov = np.diag(model.var_[k].astype(np.float64, copy=False))
        else:
            cov = model.cov_[k]
        ex, ey = _ellipse_xy(mean=model.mu_[k], cov=cov)
        ax.plot(ex, ey, lw=1.2, color=colors[k], alpha=0.9)
    ax.set_xlabel("mu_EdU (u-space)")
    ax.set_ylabel("mu_BrdU (u-space)")
    ax.set_title("Component means + 1-sigma ellipses in latent u-space")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(out_dir / "components_u_space.png")
    plt.close(fig)


def plot_residual_normality(
    *,
    out_png: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
    max_points: int = 400_000,
    seed: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = _sample_indices(qc.u.shape[0], max_points, rng=rng)
    U = qc.u[idx]
    batch = qc.batch_idx[idx]
    resp = model.predict_proba(qc.X[idx], batch)

    # Posterior-weighted standardized residuals for each component and dimension.
    gamma = np.clip(model.gamma_[batch], model.min_gamma, np.inf)  # (n,)
    fig, axs = plt.subplots(nrows=model.K, ncols=4, figsize=(16, 3.2 * model.K), dpi=200, constrained_layout=True)
    axs2 = np.asarray(axs)
    for k in range(model.K):
        if model.covariance == "diag":
            z_all = (U - model.mu_[k]) / np.sqrt(gamma[:, None] * model.var_[k][None, :])
        else:
            cov = model.cov_[k].astype(np.float64, copy=False)
            # z = L^{-1} (u-mu) / sqrt(gamma), where cov = L L^T
            L = np.linalg.cholesky(cov)
            d = (U - model.mu_[k][None, :]) / np.sqrt(gamma[:, None])
            z_all = np.linalg.solve(L, d.T).T

        for d in range(2):
            z = z_all[:, d]
            w = resp[:, k]
            # Subsample again for plotting speed by effective weight.
            eff = float(np.sum(w))
            if eff <= 1e-6:
                continue

            # Weighted histogram (approx: sample proportional to weights).
            p = w / (np.sum(w) + 1e-300)
            n_samp = min(120_000, z.size)
            nz = np.flatnonzero(p > 0)
            if nz.size == 0:
                continue
            n_samp2 = int(min(n_samp, nz.size))
            p_nz = p[nz]
            p_nz = p_nz / (np.sum(p_nz) + 1e-300)
            samp = rng.choice(nz, size=n_samp2, replace=False, p=p_nz)
            z_s = z[samp]

            axh = axs2[k, 2 * d + 0]
            axq = axs2[k, 2 * d + 1]

            axh.hist(z_s, bins=100, density=True, color="#333333", alpha=0.85)
            xs = np.linspace(-4, 4, 400)
            axh.plot(xs, (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xs * xs), color="#d62728", lw=1.2)
            axh.set_xlim(-6, 6)
            axh.set_title(f"k{k} dim={d} z hist")

            # QQ plot vs standard normal
            z_sorted = np.sort(z_s)
            n = z_sorted.size
            qs = (np.arange(1, n + 1) - 0.5) / n
            from scipy.stats import norm

            theo = norm.ppf(qs).astype(np.float64, copy=False)
            axq.scatter(theo, z_sorted, s=2, alpha=0.35, c="#1f77b4", linewidths=0)
            lo = float(np.min([theo.min(), z_sorted.min()]))
            hi = float(np.max([theo.max(), z_sorted.max()]))
            axq.plot([lo, hi], [lo, hi], color="#333333", lw=1.2)
            axq.set_title(f"k{k} dim={d} QQ")
            axq.set_xlabel("theoretical")
            axq.set_ylabel("observed")

    fig.suptitle("Posterior-weighted standardized residual normality checks", fontsize=14)
    fig.savefig(out_png)
    plt.close(fig)


def plot_simulated_vs_observed_marginals(
    *,
    out_dir: Path,
    qc: QcInputs,
    model: BatchAwareDiagGMM2D,
    n_sim_per_batch: int = 50_000,
    seed: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    B = len(qc.batch_levels)
    m_x, var_x = model.component_params_in_x_space()

    # Global: pool simulation across batches, proportional to batch size.
    counts = np.bincount(qc.batch_idx, minlength=B).astype(np.float64)
    p_batch = counts / (counts.sum() + 1e-300)
    n_total = int(min(500_000, qc.X.shape[0]))
    batch_s = rng.choice(B, size=n_total, replace=True, p=p_batch)

    # Sample component, then sample x ~ N(m_bk, diag(var_x_bk)) with diagonal covariance.
    comp_s = rng.choice(model.K, size=n_total, replace=True, p=model.pi_)
    mean_s = m_x[batch_s, comp_s]
    if model.covariance == "diag":
        std_s = np.sqrt(var_x[batch_s, comp_s])
        sim = mean_s + std_s * rng.standard_normal(size=mean_s.shape)
    else:
        cov_s = var_x[batch_s, comp_s]  # (n,2,2)
        a = cov_s[:, 0, 0]
        b = cov_s[:, 0, 1]
        c = cov_s[:, 1, 1]
        a = np.clip(a, 0.0, np.inf)
        c = np.clip(c, 0.0, np.inf)
        l00 = np.sqrt(a + 1e-30)
        l10 = b / np.clip(l00, 1e-30, np.inf)
        l11_sq = c - l10 * l10
        l11 = np.sqrt(np.clip(l11_sq, 1e-30, np.inf))
        eps = rng.standard_normal(size=mean_s.shape)
        sim0 = mean_s[:, 0] + l00 * eps[:, 0]
        sim1 = mean_s[:, 1] + l10 * eps[:, 0] + l11 * eps[:, 1]
        sim = np.stack([sim0, sim1], axis=1)

    obs_idx = _sample_indices(qc.X.shape[0], n_total, rng=rng)
    obs = qc.X[obs_idx]

    for d, name in enumerate(["edu_x", "brdu_x"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200, constrained_layout=True)
        ax.hist(obs[:, d], bins=200, density=True, alpha=0.55, label="observed", color="#1f77b4")
        ax.hist(sim[:, d], bins=200, density=True, alpha=0.55, label="simulated", color="#ff7f0e")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"Posterior predictive check (global): {name} marginals")
        ax.legend(frameon=True, fontsize=8)
        fig.savefig(out_dir / f"ppc_global_{name}.png")
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fit a mixture model for BrdU/EdU pulses and write QC caches + diagnostic plots. "
            "Default is two independent 1D mixtures (one per channel), which avoids joint 2D decision boundaries."
        )
    )
    p.add_argument("h5ad", type=Path, nargs="?", default=None, help="Input h5ad (optional if --obs-parquet is provided).")
    p.add_argument(
        "--obs-parquet",
        type=Path,
        default=None,
        help="Optional parquet containing the required obs columns (index must be cell_id/obs_name).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--model",
        type=str,
        default="independent1d",
        choices=["independent1d", "joint2d"],
        help="Mixture model: independent 1D mixtures per channel, or the legacy joint 2D model.",
    )
    p.add_argument("--edu-col", type=str, default="log_edu_mean", help="obs column used as edu_x")
    p.add_argument("--brdu-col", type=str, default="log_brdu_mean", help="obs column used as brdu_x")
    p.add_argument(
        "--n-components-1d",
        type=int,
        default=3,
        help="Number of mixture components per channel when --model independent1d (>=2).",
    )
    p.add_argument(
        "--batchaware1d-affine",
        type=str,
        default="shift",
        choices=["shift", "shift_scale"],
        help="For --model independent1d: per-batch affine nuisance transform (default: shift).",
    )
    p.add_argument(
        "--batchaware1d-variance",
        type=str,
        default="component",
        choices=["component", "tied"],
        help="For --model independent1d: variance in u-space (per-component or tied across components).",
    )
    p.add_argument(
        "--edu-init",
        type=str,
        default="quantile",
        choices=["quantile", "tail_quantile", "manual_seed"],
        help=(
            "For --model independent1d: how to initialize the 1D EdU mixture. "
            "'manual_seed' uses edu_pos as initialization only (then runs standard unsupervised EM)."
        ),
    )
    p.add_argument(
        "--brdu-init",
        type=str,
        default="quantile",
        choices=["quantile", "tail_quantile", "manual_seed"],
        help=(
            "For --model independent1d: how to initialize the 1D BrdU mixture. "
            "'manual_seed' uses brdu_pos as initialization only (then runs standard unsupervised EM)."
        ),
    )
    p.add_argument(
        "--edu-zero-policy",
        type=str,
        default="include",
        choices=["include", "exclude"],
        help="For --model independent1d: whether to exclude edu_x==0 cells from fitting (still inferred).",
    )
    p.add_argument(
        "--brdu-zero-policy",
        type=str,
        default="include",
        choices=["include", "exclude"],
        help="For --model independent1d: whether to exclude brdu_x==0 cells from fitting (still inferred).",
    )
    p.add_argument(
        "--edu-pos-topk",
        type=int,
        default=1,
        help="When --model independent1d, define EdU-positive as the top-k highest-mean components (default 1).",
    )
    p.add_argument(
        "--edu-pos-mode",
        type=str,
        default="topk",
        choices=["topk", "not_main", "manual_match"],
        help=(
            "When --model independent1d, how to define EdU-positive components. "
            "'topk' uses --edu-pos-topk highest-mean components; 'not_main' treats all but the highest-weight "
            "component as EdU-positive; 'manual_match' selects components to match edu_pos fraction (mapping only)."
        ),
    )
    p.add_argument(
        "--brdu-pos-topk",
        type=int,
        default=1,
        help="When --model independent1d, define BrdU-positive as the top-k highest-mean components (default 1).",
    )
    p.add_argument(
        "--brdu-pos-mode",
        type=str,
        default="topk",
        choices=["topk", "not_main", "manual_match"],
        help=(
            "When --model independent1d, how to define BrdU-positive components. "
            "'topk' uses --brdu-pos-topk highest-mean components; 'not_main' treats all but the highest-weight "
            "component as BrdU-positive; 'manual_match' selects components to match brdu_pos fraction (mapping only)."
        ),
    )
    p.add_argument(
        "--edu-map-thr",
        type=float,
        default=0.5,
        help="For assignment plots/benchmarks: call EdU+ if P(E=1) >= this threshold.",
    )
    p.add_argument(
        "--brdu-map-thr",
        type=float,
        default=0.3,
        help="For assignment plots/benchmarks: call BrdU+ if P(B=1) >= this threshold.",
    )
    p.add_argument(
        "--covariance",
        type=str,
        default="diag",
        choices=["diag", "full"],
        help="Per-component covariance in latent u-space (diag assumes conditional independence within component).",
    )
    p.add_argument(
        "--batch-col",
        type=str,
        default="dataset",
        help="obs column used as batch/run, or comma-separated columns to form a composite batch key",
    )
    p.add_argument("--ref-batch", type=str, default=None, help="Optional reference batch level name")
    p.add_argument("--region-col", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument(
        "--affine-init",
        type=str,
        default="all",
        choices=["all", "manual_neg"],
        help=(
            "How to initialize per-batch alpha/beta before EM. "
            "'all' uses all cells; 'manual_neg' uses manual-negative cells and requires edu_pos/brdu_pos "
            "(and for --model joint2d also requires --manual-supervision)."
        ),
    )

    p.add_argument(
        "--manual-supervision",
        action="store_true",
        help=(
            "Use manual BrdU/EdU threshold calls as weak supervision during EM. "
            "Posteriors are still reported from x only (no thresholds at inference)."
        ),
    )
    p.add_argument("--edu-pos-col", type=str, default="edu_pos", help="obs column for manual EdU+ calls (bool/0/1)")
    p.add_argument("--brdu-pos-col", type=str, default="brdu_pos", help="obs column for manual BrdU+ calls (bool/0/1)")
    p.add_argument(
        "--no-benchmark-manual",
        action="store_true",
        help="Skip benchmark vs obs[edu_pos_col]/obs[brdu_pos_col] hard calls.",
    )
    p.add_argument(
        "--calibrate-to-manual",
        action="store_true",
        help=(
            "For --model independent1d: enable per-batch monotone calibration of P(E=1) and P(B=1) "
            "to match manual edu_pos/brdu_pos fractions. Calibration requires manual columns and is applied "
            "before forming joint quadrant probabilities."
        ),
    )
    p.add_argument(
        "--calibrate-to-xmin",
        action="store_true",
        help=(
            "For --model independent1d: enable per-batch monotone calibration that only raises the implied "
            "p=0.5 intensity boundary in batches where it is too low. Uses --edu-xmin / --brdu-xmin."
        ),
    )
    p.add_argument(
        "--calibrate-to-xrange",
        action="store_true",
        help=(
            "For --model independent1d: enable per-batch monotone calibration that constrains the implied "
            "p=0.5 intensity boundary to lie within [xmin, xmax]. Uses --*-xmin/--*-xmax."
        ),
    )
    p.add_argument(
        "--edu-xmin",
        type=float,
        default=float("nan"),
        help="When --calibrate-to-xmin is set: minimum desired EdU x-space p=0.5 boundary (floor).",
    )
    p.add_argument(
        "--edu-xmax",
        type=float,
        default=float("nan"),
        help="When --calibrate-to-xrange is set: maximum desired EdU x-space p=0.5 boundary (ceiling).",
    )
    p.add_argument(
        "--brdu-xmin",
        type=float,
        default=float("nan"),
        help="When --calibrate-to-xmin is set: minimum desired BrdU x-space p=0.5 boundary (floor).",
    )
    p.add_argument(
        "--brdu-xmax",
        type=float,
        default=float("nan"),
        help="When --calibrate-to-xrange is set: maximum desired BrdU x-space p=0.5 boundary (ceiling).",
    )
    p.add_argument(
        "--xmin-nearest-k",
        type=int,
        default=2000,
        help="When --calibrate-to-xmin is set: number of nearest-to-xmin cells used per batch to estimate the logit shift.",
    )
    p.add_argument(
        "--manual-group-cols",
        type=str,
        default="ccf_adjusted,roi,dataset",
        help="Comma-separated obs columns defining the manual-threshold groups used to derive soft labels",
    )
    p.add_argument(
        "--manual-sigma-edu",
        type=float,
        default=0.05,
        help="Sigmoid softness for EdU threshold supervision (smaller = closer to hard threshold).",
    )
    p.add_argument(
        "--manual-sigma-brdu",
        type=float,
        default=0.05,
        help="Sigmoid softness for BrdU threshold supervision (smaller = closer to hard threshold).",
    )
    p.add_argument("--label-weight", type=float, default=0.6, help="Weight of manual supervision term in EM")
    p.add_argument(
        "--manual-eps",
        type=float,
        default=0.05,
        help="Noise rate for manual supervision when theta is frozen (0<eps<0.5).",
    )
    p.add_argument(
        "--learn-theta",
        action="store_true",
        help="Learn per-component manual label probabilities instead of fixing them to (neg, edu_only, brdu_only, double).",
    )

    p.add_argument("--max-fit-cells", type=int, default=None, help="Optional subsample size for fitting (inference still runs on all)")
    p.add_argument("--max-scatter", type=int, default=80_000)
    p.add_argument("--max-cells-per-batch", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--facet-group-cols",
        type=str,
        default=None,
        help=(
            "Optional comma-separated obs columns to form a diagnostic facet key (e.g. 'ccf_adjusted,roi,dataset'). "
            "Generates an x-space facet grid with per-class 0.5 contours for the batch implied by each group."
        ),
    )
    p.add_argument("--facet-max-groups", type=int, default=24)
    p.add_argument("--facet-max-cells-per-group", type=int, default=200_000)
    p.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable P(class|x,batch)=0.5 contour overlays (keeps density + p_max-colored calls).",
    )

    p.add_argument(
        "--reg-covar",
        type=float,
        default=1e-3,
        help="Diagonal covariance floor in latent u-space (prevents variance collapse / overfitting).",
    )
    p.add_argument("--min-beta", type=float, default=1e-3, help="Lower bound for per-batch beta.")
    p.add_argument("--max-beta", type=float, default=5.0, help="Upper bound for per-batch beta (prevents degeneracy).")
    p.add_argument("--min-gamma", type=float, default=1e-3, help="Lower bound for per-batch gamma.")
    p.add_argument("--max-gamma", type=float, default=50.0, help="Upper bound for per-batch gamma.")
    p.add_argument(
        "--learn-affine",
        action="store_true",
        help="Learn per-batch (alpha,beta) updates during EM (can overfit; default is frozen after robust init).",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.h5ad is None and args.obs_parquet is None:
        raise ValueError("Provide either an input h5ad or --obs-parquet.")
    h5ad: Path | None = None
    if args.obs_parquet is None:
        if args.h5ad is None:
            raise ValueError("Input h5ad is required when --obs-parquet is not set.")
        h5ad = args.h5ad.expanduser()
        if not h5ad.exists():
            raise FileNotFoundError(h5ad)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    plot_dir = out_dir / "plots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batch_cols = _parse_list(args.batch_col)
    if not batch_cols:
        raise ValueError("--batch-col must be non-empty")

    manual_group_cols = _parse_list(args.manual_group_cols) if args.manual_supervision else []
    if args.manual_supervision:
        if args.label_weight <= 0:
            raise ValueError("--label-weight must be > 0 when --manual-supervision is set")
        if args.manual_sigma_edu <= 0 or args.manual_sigma_brdu <= 0:
            raise ValueError("--manual-sigma-edu/--manual-sigma-brdu must be > 0")
        if not (0.0 < float(args.manual_eps) < 0.5):
            raise ValueError("--manual-eps must satisfy 0 < eps < 0.5")
        if not manual_group_cols:
            raise ValueError("--manual-group-cols must be non-empty when --manual-supervision is set")

    cols = [args.edu_col, args.brdu_col, *batch_cols]
    if args.region_col is not None:
        cols.append(args.region_col)
    if args.manual_supervision:
        cols.extend(manual_group_cols)
    if (not bool(args.no_benchmark_manual)) or args.manual_supervision:
        cols.append(args.edu_pos_col)
        cols.append(args.brdu_pos_col)
    if args.facet_group_cols is not None:
        cols.extend(_parse_list(args.facet_group_cols))
    cols = _ordered_unique(cols)

    if args.obs_parquet is not None:
        pq = Path(args.obs_parquet).expanduser()
        if not pq.exists():
            raise FileNotFoundError(pq)
        df = pd.read_parquet(pq).copy()
        if df.index is None:
            raise ValueError("Parquet input must have a non-null index (cell_id/obs_name).")
        missing = sorted(set(cols) - set(df.columns))
        if missing:
            raise KeyError(f"Missing columns in parquet {pq}: {missing}")
        df = df.loc[:, cols].copy()
    else:
        if h5ad is None:
            raise RuntimeError("Internal error: expected h5ad to be set.")
        adata = sc.read_h5ad(h5ad, backed="r")
        try:
            obs = adata.obs
            missing = sorted(set(cols) - set(obs.columns))
            if missing:
                raise KeyError(f"Missing obs columns: {missing}")
            df = obs.loc[:, cols].copy()
            df.index = pd.Index(adata.obs_names.astype(str), name="cell_id")
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    df["_row_idx"] = np.arange(df.shape[0], dtype=np.int64)

    edu_x = pd.to_numeric(df[args.edu_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    brdu_x = pd.to_numeric(df[args.brdu_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    keep = np.isfinite(edu_x) & np.isfinite(brdu_x)
    for c in batch_cols:
        keep &= df[c].notna().to_numpy()

    if args.region_col is not None or args.region is not None:
        if args.region_col is None or args.region is None:
            raise ValueError("Set both --region-col and --region to enable region filtering")
        allowed = set(_parse_list(args.region))
        keep &= df[args.region_col].astype(str).isin(allowed).to_numpy()

    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("No cells remain after filtering.")

    X = np.column_stack(
        [
            pd.to_numeric(df[args.edu_col], errors="raise").to_numpy(dtype=np.float64, copy=False),
            pd.to_numeric(df[args.brdu_col], errors="raise").to_numpy(dtype=np.float64, copy=False),
        ]
    )

    y_edu: np.ndarray | None = None
    y_brdu: np.ndarray | None = None
    manual_thr_tsv: Path | None = None
    codes_g: np.ndarray | None = None
    levels_g: pd.Index | None = None
    edu_pos: np.ndarray | None = None
    brdu_pos: np.ndarray | None = None

    # Optional benchmark hard calls (also used for manual-negative affine init).
    edu_hard: np.ndarray | None = None
    brdu_hard: np.ndarray | None = None
    if not bool(args.no_benchmark_manual):
        edu_pos_raw = pd.to_numeric(df[args.edu_pos_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        brdu_pos_raw = pd.to_numeric(df[args.brdu_pos_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(edu_pos_raw).all() or not np.isfinite(brdu_pos_raw).all():
            raise ValueError(f"{args.edu_pos_col}/{args.brdu_pos_col} must be finite (bool/0/1) for all kept rows.")
        edu_hard = edu_pos_raw > 0.5
        brdu_hard = brdu_pos_raw > 0.5

    if args.model != "joint2d" and args.manual_supervision:
        raise ValueError("--manual-supervision is only supported with --model joint2d.")
    if args.model == "joint2d" and args.manual_supervision:
        if edu_hard is None or brdu_hard is None:
            raise ValueError("--manual-supervision requires edu_pos/brdu_pos (do not set --no-benchmark-manual).")
        edu_pos = edu_hard
        brdu_pos = brdu_hard

        group_key = df[manual_group_cols[0]].astype(str).to_numpy()
        for c in manual_group_cols[1:]:
            group_key = group_key + "|" + df[c].astype(str).to_numpy()

        tmp = pd.DataFrame(
            {
                "key": group_key,
                "edu_x": X[:, 0],
                "brdu_x": X[:, 1],
                "edu_pos": edu_pos,
                "brdu_pos": brdu_pos,
            }
        )
        edu_max_neg = tmp.loc[~tmp["edu_pos"]].groupby("key")["edu_x"].max()
        edu_min_pos = tmp.loc[tmp["edu_pos"]].groupby("key")["edu_x"].min()
        brdu_max_neg = tmp.loc[~tmp["brdu_pos"]].groupby("key")["brdu_x"].max()
        brdu_min_pos = tmp.loc[tmp["brdu_pos"]].groupby("key")["brdu_x"].min()

        all_keys = set(tmp["key"].unique())
        ok_edu = set(edu_max_neg.index) & set(edu_min_pos.index)
        ok_brdu = set(brdu_max_neg.index) & set(brdu_min_pos.index)
        missing_edu = sorted(all_keys - ok_edu)
        missing_brdu = sorted(all_keys - ok_brdu)
        if missing_edu:
            head = missing_edu[:8]
            tail = f" (+{len(missing_edu) - 8} more)" if len(missing_edu) > 8 else ""
            raise ValueError(f"cannot derive EdU thresholds; groups missing pos/neg: {head}{tail}")
        if missing_brdu:
            head = missing_brdu[:8]
            tail = f" (+{len(missing_brdu) - 8} more)" if len(missing_brdu) > 8 else ""
            raise ValueError(f"cannot derive BrdU thresholds; groups missing pos/neg: {head}{tail}")

        edu_gap = edu_min_pos - edu_max_neg
        brdu_gap = brdu_min_pos - brdu_max_neg
        if not (edu_gap > 0).all():
            bad = edu_gap[edu_gap <= 0].index.astype(str).tolist()[:5]
            raise ValueError(f"EdU groups with non-positive threshold gap (example keys): {bad}")
        if not (brdu_gap > 0).all():
            bad = brdu_gap[brdu_gap <= 0].index.astype(str).tolist()[:5]
            raise ValueError(f"BrdU groups with non-positive threshold gap (example keys): {bad}")

        edu_thr = 0.5 * (edu_max_neg + edu_min_pos)
        brdu_thr = 0.5 * (brdu_max_neg + brdu_min_pos)

        codes_g, levels_g = pd.factorize(tmp["key"], sort=False)
        edu_thr_by_level = edu_thr.reindex(levels_g).to_numpy(dtype=np.float64)
        brdu_thr_by_level = brdu_thr.reindex(levels_g).to_numpy(dtype=np.float64)
        if not np.isfinite(edu_thr_by_level).all() or not np.isfinite(brdu_thr_by_level).all():
            raise ValueError("non-finite derived thresholds found")

        y_edu = sigmoid((X[:, 0] - edu_thr_by_level[codes_g]) / float(args.manual_sigma_edu))
        y_brdu = sigmoid((X[:, 1] - brdu_thr_by_level[codes_g]) / float(args.manual_sigma_brdu))

        manual_thr = pd.DataFrame(
            {
                "key": levels_g.astype(str),
                "edu_thr": edu_thr_by_level,
                "brdu_thr": brdu_thr_by_level,
                "edu_gap": edu_gap.reindex(levels_g).to_numpy(dtype=np.float64),
                "brdu_gap": brdu_gap.reindex(levels_g).to_numpy(dtype=np.float64),
                "n_cells": np.bincount(codes_g, minlength=int(levels_g.size)).astype(np.int64),
            }
        )
        manual_thr_tsv = cache_dir / "manual_thresholds_by_group.tsv"
        manual_thr.to_csv(manual_thr_tsv, sep="\t", index=False)

    affine_init_mask: np.ndarray | None = None
    affine_init_mask_edu: np.ndarray | None = None
    affine_init_mask_brdu: np.ndarray | None = None
    if args.affine_init == "manual_neg":
        if args.model == "joint2d":
            if not args.manual_supervision:
                raise ValueError("--affine-init manual_neg requires --manual-supervision for --model joint2d.")
            if edu_pos is None or brdu_pos is None:
                raise RuntimeError("internal error: edu_pos/brdu_pos not available")
            affine_init_mask = (~edu_pos) & (~brdu_pos)
        else:
            if edu_hard is None or brdu_hard is None:
                raise ValueError("--affine-init manual_neg requires edu_pos/brdu_pos (do not set --no-benchmark-manual).")
            affine_init_mask_edu = ~edu_hard
            affine_init_mask_brdu = ~brdu_hard

    # Build a composite batch key if needed (e.g. ccf_adjusted,roi,dataset).
    batch_key = df[batch_cols[0]].astype(str).to_numpy()
    for c in batch_cols[1:]:
        batch_key = batch_key + "|" + df[c].astype(str).to_numpy()

    batch_codes, batch_levels = pd.factorize(batch_key, sort=True)
    batch_levels_str = [str(x) for x in batch_levels]

    ref_idx: int | None = None
    if args.ref_batch is not None:
        level_to_idx = {name: i for i, name in enumerate(batch_levels_str)}
        if args.ref_batch not in level_to_idx:
            raise ValueError(f"--ref-batch not found in levels: {batch_levels_str}")
        ref_idx = int(level_to_idx[args.ref_batch])

    rng = np.random.default_rng(args.seed)
    fit_idx = np.array([], dtype=np.int64)
    model_kind = str(args.model)
    model: BatchAwareDiagGMM2D | None = None
    params_1d: pd.DataFrame | None = None

    if model_kind == "independent1d":
        edu_fit_mask: np.ndarray | None = None
        brdu_fit_mask: np.ndarray | None = None
        if str(args.edu_zero_policy) == "exclude":
            edu_fit_mask = X[:, 0] > 0
        if str(args.brdu_zero_policy) == "exclude":
            brdu_fit_mask = X[:, 1] > 0

        edu_affine_init_mask = affine_init_mask_edu
        if edu_affine_init_mask is not None and edu_fit_mask is not None:
            edu_affine_init_mask = edu_affine_init_mask[edu_fit_mask]
        brdu_affine_init_mask = affine_init_mask_brdu
        if brdu_affine_init_mask is not None and brdu_fit_mask is not None:
            brdu_affine_init_mask = brdu_affine_init_mask[brdu_fit_mask]

        edu_model = BatchAwareGMM1D(
            n_components=int(args.n_components_1d),
            affine=str(args.batchaware1d_affine),
            variance=str(args.batchaware1d_variance),
            reg_covar=float(args.reg_covar),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            random_state=int(args.seed),
            min_beta=float(args.min_beta),
            max_beta=float(args.max_beta),
            min_gamma=float(args.min_gamma),
            max_gamma=float(args.max_gamma),
            verbose=bool(args.verbose),
        ).fit(
            X[:, 0],
            batch_codes.astype(np.int64, copy=False),
            ref_batch=ref_idx,
            init=str(args.edu_init),
            hard=edu_hard,
            learn_affine=bool(args.learn_affine),
            affine_init_mask=edu_affine_init_mask,
            fit_mask=edu_fit_mask,
        )
        brdu_model = BatchAwareGMM1D(
            n_components=int(args.n_components_1d),
            affine=str(args.batchaware1d_affine),
            variance=str(args.batchaware1d_variance),
            reg_covar=float(args.reg_covar),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            random_state=int(args.seed),
            min_beta=float(args.min_beta),
            max_beta=float(args.max_beta),
            min_gamma=float(args.min_gamma),
            max_gamma=float(args.max_gamma),
            verbose=bool(args.verbose),
        ).fit(
            X[:, 1],
            batch_codes.astype(np.int64, copy=False),
            ref_batch=ref_idx,
            init=str(args.brdu_init),
            hard=brdu_hard,
            learn_affine=bool(args.learn_affine),
            affine_init_mask=brdu_affine_init_mask,
            fit_mask=brdu_fit_mask,
        )

        resp_edu = edu_model.predict_proba(X[:, 0], batch_codes.astype(np.int64, copy=False))
        resp_brdu = brdu_model.predict_proba(X[:, 1], batch_codes.astype(np.int64, copy=False))

        if str(args.edu_pos_mode) == "manual_match":
            if edu_hard is None:
                raise ValueError("--edu-pos-mode manual_match requires edu_pos (do not set --no-benchmark-manual).")
            edu_pos_comps = _select_pos_components_1d_by_manual(resp=resp_edu, hard=edu_hard.astype(np.int8, copy=False))
        else:
            edu_pos_comps = _select_pos_components_1d(
                mu=edu_model.mu_,
                pi=edu_model.pi_,
                mode=str(args.edu_pos_mode),
                topk=int(args.edu_pos_topk),
            )

        if str(args.brdu_pos_mode) == "manual_match":
            if brdu_hard is None:
                raise ValueError("--brdu-pos-mode manual_match requires brdu_pos (do not set --no-benchmark-manual).")
            brdu_pos_comps = _select_pos_components_1d_by_manual(
                resp=resp_brdu, hard=brdu_hard.astype(np.int8, copy=False)
            )
        else:
            brdu_pos_comps = _select_pos_components_1d(
                mu=brdu_model.mu_,
                pi=brdu_model.pi_,
                mode=str(args.brdu_pos_mode),
                topk=int(args.brdu_pos_topk),
            )

        pE_raw = resp_edu[:, edu_pos_comps].sum(axis=1).astype(np.float64, copy=False)
        pB_raw = resp_brdu[:, brdu_pos_comps].sum(axis=1).astype(np.float64, copy=False)

        if bool(args.calibrate_to_manual) and (bool(args.calibrate_to_xmin) or bool(args.calibrate_to_xrange)):
            raise ValueError("--calibrate-to-manual cannot be combined with x-anchored calibration flags.")
        if bool(args.calibrate_to_xmin) and bool(args.calibrate_to_xrange):
            raise ValueError("--calibrate-to-xmin and --calibrate-to-xrange are mutually exclusive.")

        if bool(args.calibrate_to_manual):
            if bool(args.no_benchmark_manual) or edu_hard is None or brdu_hard is None:
                raise ValueError(
                    "--no-benchmark-manual cannot be used when --calibrate-to-manual is set, "
                    "because calibration requires edu_pos/brdu_pos."
                )

            edu_mask = edu_fit_mask if edu_fit_mask is not None else None
            brdu_mask = brdu_fit_mask if brdu_fit_mask is not None else None

            pE_in = pE_raw
            if edu_mask is not None:
                pE_in = np.asarray(pE_in, dtype=np.float64).copy()
                pE_in[~edu_mask] = 0.0
            pB_in = pB_raw
            if brdu_mask is not None:
                pB_in = np.asarray(pB_in, dtype=np.float64).copy()
                pB_in[~brdu_mask] = 0.0

            pE, shiftE = _calibrate_prob_logit_shift_by_batch(
                p=pE_in,
                hard=edu_hard.astype(np.int8, copy=False),
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                mask=edu_mask,
            )
            pB, shiftB = _calibrate_prob_logit_shift_by_batch(
                p=pB_in,
                hard=brdu_hard.astype(np.int8, copy=False),
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                mask=brdu_mask,
            )

            calib_rows: list[dict[str, object]] = []
            for b, name in enumerate(batch_levels_str):
                idx_b = batch_codes == b
                if not np.any(idx_b):
                    continue
                calib_rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "edu",
                        "method": "logit_shift",
                        "hard_frac": float(np.mean(edu_hard[idx_b])),
                        "raw_mean": float(np.mean(pE_raw[idx_b])),
                        "cal_mean": float(np.mean(pE[idx_b])),
                        "param": float(shiftE[b]),
                    }
                )
                calib_rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "brdu",
                        "method": "logit_shift",
                        "hard_frac": float(np.mean(brdu_hard[idx_b])),
                        "raw_mean": float(np.mean(pB_raw[idx_b])),
                        "cal_mean": float(np.mean(pB[idx_b])),
                        "param": float(shiftB[b]),
                    }
                )
            _write_tsv(cache_dir / "batchaware_1d_calibration_by_batch.tsv", pd.DataFrame(calib_rows))
        elif bool(args.calibrate_to_xmin):
            if not math.isfinite(float(args.edu_xmin)) or not math.isfinite(float(args.brdu_xmin)):
                raise ValueError("--edu-xmin and --brdu-xmin must be finite when --calibrate-to-xmin is set.")

            edu_mask = edu_fit_mask if edu_fit_mask is not None else None
            brdu_mask = brdu_fit_mask if brdu_fit_mask is not None else None

            pE, shiftE = _calibrate_prob_logit_shift_to_xmin_by_batch(
                p=pE_raw,
                x=X[:, 0],
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                x_min=float(args.edu_xmin),
                nearest_k=int(args.xmin_nearest_k),
                mask=edu_mask,
            )
            pB, shiftB = _calibrate_prob_logit_shift_to_xmin_by_batch(
                p=pB_raw,
                x=X[:, 1],
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                x_min=float(args.brdu_xmin),
                nearest_k=int(args.xmin_nearest_k),
                mask=brdu_mask,
            )

            rows: list[dict[str, object]] = []
            for b, name in enumerate(batch_levels_str):
                idx_b = batch_codes == b
                if not np.any(idx_b):
                    continue
                rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "edu",
                        "method": "logit_shift_floor_xmin",
                        "xmin": float(args.edu_xmin),
                        "raw_mean": float(np.mean(pE_raw[idx_b])),
                        "cal_mean": float(np.mean(pE[idx_b])),
                        "param": float(shiftE[b]),
                    }
                )
                rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "brdu",
                        "method": "logit_shift_floor_xmin",
                        "xmin": float(args.brdu_xmin),
                        "raw_mean": float(np.mean(pB_raw[idx_b])),
                        "cal_mean": float(np.mean(pB[idx_b])),
                        "param": float(shiftB[b]),
                    }
                )
            _write_tsv(cache_dir / "batchaware_1d_xmin_calibration_by_batch.tsv", pd.DataFrame(rows))
        elif bool(args.calibrate_to_xrange):
            if not math.isfinite(float(args.edu_xmin)) or not math.isfinite(float(args.edu_xmax)):
                raise ValueError("--edu-xmin and --edu-xmax must be finite when --calibrate-to-xrange is set.")
            if not math.isfinite(float(args.brdu_xmin)) or not math.isfinite(float(args.brdu_xmax)):
                raise ValueError("--brdu-xmin and --brdu-xmax must be finite when --calibrate-to-xrange is set.")

            edu_mask = edu_fit_mask if edu_fit_mask is not None else None
            brdu_mask = brdu_fit_mask if brdu_fit_mask is not None else None

            pE, shiftE = _calibrate_prob_logit_shift_to_xrange_by_batch(
                p=pE_raw,
                x=X[:, 0],
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                x_min=float(args.edu_xmin),
                x_max=float(args.edu_xmax),
                nearest_k=int(args.xmin_nearest_k),
                mask=edu_mask,
            )
            pB, shiftB = _calibrate_prob_logit_shift_to_xrange_by_batch(
                p=pB_raw,
                x=X[:, 1],
                batch_idx=batch_codes.astype(np.int64, copy=False),
                n_batches=len(batch_levels_str),
                x_min=float(args.brdu_xmin),
                x_max=float(args.brdu_xmax),
                nearest_k=int(args.xmin_nearest_k),
                mask=brdu_mask,
            )

            rows: list[dict[str, object]] = []
            for b, name in enumerate(batch_levels_str):
                idx_b = batch_codes == b
                if not np.any(idx_b):
                    continue
                rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "edu",
                        "method": "logit_shift_xrange",
                        "xmin": float(args.edu_xmin),
                        "xmax": float(args.edu_xmax),
                        "raw_mean": float(np.mean(pE_raw[idx_b])),
                        "cal_mean": float(np.mean(pE[idx_b])),
                        "param": float(shiftE[b]),
                    }
                )
                rows.append(
                    {
                        "batch_idx": int(b),
                        "batch": str(name),
                        "channel": "brdu",
                        "method": "logit_shift_xrange",
                        "xmin": float(args.brdu_xmin),
                        "xmax": float(args.brdu_xmax),
                        "raw_mean": float(np.mean(pB_raw[idx_b])),
                        "cal_mean": float(np.mean(pB[idx_b])),
                        "param": float(shiftB[b]),
                    }
                )
            _write_tsv(cache_dir / "batchaware_1d_xrange_calibration_by_batch.tsv", pd.DataFrame(rows))
        else:
            pE = pE_raw
            pB = pB_raw

        if edu_fit_mask is not None:
            pE = np.asarray(pE, dtype=np.float64).copy()
            pE[~edu_fit_mask] = 0.0
        if brdu_fit_mask is not None:
            pB = np.asarray(pB, dtype=np.float64).copy()
            pB[~brdu_fit_mask] = 0.0

        pi_11 = (pB * pE).astype(np.float64, copy=False)
        pi_10 = (pB * (1.0 - pE)).astype(np.float64, copy=False)
        pi_01 = ((1.0 - pB) * pE).astype(np.float64, copy=False)
        pi_00 = ((1.0 - pB) * (1.0 - pE)).astype(np.float64, copy=False)
        P = np.stack([pi_00, pi_01, pi_10, pi_11], axis=1)
        p_max = np.max(P, axis=1)
        H = entropy(P, axis=1)

        u_edu = edu_model.transform_to_u(X[:, 0], batch_codes.astype(np.int64, copy=False))
        u_brdu = brdu_model.transform_to_u(X[:, 1], batch_codes.astype(np.int64, copy=False))
        ll_x = edu_model.score_samples(X[:, 0], batch_codes.astype(np.int64, copy=False)) + brdu_model.score_samples(
            X[:, 1], batch_codes.astype(np.int64, copy=False)
        )

        qc = QcInputs(
            X=X,
            u=np.column_stack([u_edu, u_brdu]),
            batch_idx=batch_codes.astype(np.int64, copy=False),
            batch_levels=batch_levels_str,
            classes={"neg": pi_00, "EdU_only": pi_01, "BrdU_only": pi_10, "double": pi_11},
            class_names=["neg", "EdU_only", "BrdU_only", "double"],
            p_max=p_max,
            entropy=H,
            ll_x=ll_x.astype(np.float64, copy=False),
        )

        params_1d = pd.concat(
            [
                pd.DataFrame(
                    {
                        "batch_idx": np.arange(len(batch_levels_str), dtype=np.int64),
                        "batch_level": batch_levels_str,
                        "channel": "edu",
                        "alpha": edu_model.alpha_,
                        "beta": edu_model.beta_,
                        "gamma": edu_model.gamma_,
                    }
                ),
                pd.DataFrame(
                    {
                        "batch_idx": np.arange(len(batch_levels_str), dtype=np.int64),
                        "batch_level": batch_levels_str,
                        "channel": "brdu",
                        "alpha": brdu_model.alpha_,
                        "beta": brdu_model.beta_,
                        "gamma": brdu_model.gamma_,
                    }
                ),
            ],
            ignore_index=True,
        )

        comp_1d = pd.concat(
            [
                pd.DataFrame(
                    {
                        "channel": "edu",
                        "k": np.arange(edu_model.K, dtype=np.int64),
                        "pi_mean": edu_model.pi_,
                        "pi_min": edu_model.pi_,
                        "pi_max": edu_model.pi_,
                        "mu": edu_model.mu_,
                        "var": edu_model.var_,
                        "pos": np.isin(np.arange(edu_model.K), edu_pos_comps),
                    }
                ),
                pd.DataFrame(
                    {
                        "channel": "brdu",
                        "k": np.arange(brdu_model.K, dtype=np.int64),
                        "pi_mean": brdu_model.pi_,
                        "pi_min": brdu_model.pi_,
                        "pi_max": brdu_model.pi_,
                        "mu": brdu_model.mu_,
                        "var": brdu_model.var_,
                        "pos": np.isin(np.arange(brdu_model.K), brdu_pos_comps),
                    }
                ),
            ],
            ignore_index=True,
        )
        _write_tsv(cache_dir / "batchaware_1d_components.tsv", comp_1d)
        ref_batch = int(edu_model.ref_batch_)
    elif model_kind == "joint2d":
        fit_idx = np.arange(X.shape[0], dtype=np.int64)
        if args.max_fit_cells is not None and X.shape[0] > args.max_fit_cells:
            fit_idx = np.sort(rng.choice(X.shape[0], size=int(args.max_fit_cells), replace=False)).astype(np.int64, copy=False)

        model = BatchAwareDiagGMM2D(
            covariance=str(args.covariance),
            reg_covar=float(args.reg_covar),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            random_state=int(args.seed),
            min_beta=float(args.min_beta),
            max_beta=float(args.max_beta),
            min_gamma=float(args.min_gamma),
            max_gamma=float(args.max_gamma),
            label_weight=float(args.label_weight) if args.manual_supervision else 0.0,
            manual_eps=float(args.manual_eps),
            verbose=bool(args.verbose),
        )
        model.fit(
            X[fit_idx],
            batch_codes[fit_idx].astype(np.int64, copy=False),
            ref_batch=ref_idx,
            y_edu=None if y_edu is None else y_edu[fit_idx],
            y_brdu=None if y_brdu is None else y_brdu[fit_idx],
            freeze_theta=bool(args.manual_supervision and (not args.learn_theta)),
            learn_affine=bool(args.learn_affine),
            affine_init_mask=None if affine_init_mask is None else affine_init_mask[fit_idx],
        )

        # Derived per-cell QC quantities (computed on all cells, no thresholds).
        qc = compute_qc_inputs(
            X=X, batch_idx=batch_codes.astype(np.int64, copy=False), batch_levels=batch_levels_str, model=model
        )
        ref_batch = int(model.ref_batch_)
    else:
        raise ValueError(f"Unknown --model {model_kind!r}")

    qc_meta = {
        "input_h5ad": "" if h5ad is None else str(h5ad),
        "input_obs_parquet": "" if args.obs_parquet is None else str(Path(args.obs_parquet).expanduser()),
        "model": model_kind,
        "n_obs_total": int(df.shape[0]),
        "edu_col": args.edu_col,
        "brdu_col": args.brdu_col,
        "batch_col": args.batch_col,
        "n_components_1d": int(args.n_components_1d) if model_kind == "independent1d" else 0,
        "edu_pos_topk": int(args.edu_pos_topk) if model_kind == "independent1d" else 0,
        "brdu_pos_topk": int(args.brdu_pos_topk) if model_kind == "independent1d" else 0,
        "edu_pos_mode": str(args.edu_pos_mode) if model_kind == "independent1d" else "",
        "brdu_pos_mode": str(args.brdu_pos_mode) if model_kind == "independent1d" else "",
        "batchaware1d_affine": str(args.batchaware1d_affine) if model_kind == "independent1d" else "",
        "batchaware1d_variance": str(args.batchaware1d_variance) if model_kind == "independent1d" else "",
        "edu_zero_policy": str(args.edu_zero_policy) if model_kind == "independent1d" else "",
        "brdu_zero_policy": str(args.brdu_zero_policy) if model_kind == "independent1d" else "",
        "edu_init": str(args.edu_init) if model_kind == "independent1d" else "",
        "brdu_init": str(args.brdu_init) if model_kind == "independent1d" else "",
        "pi_mode_1d": "global" if model_kind == "independent1d" else "",
        "covariance": str(args.covariance),
        "manual_supervision": bool(args.manual_supervision),
        "calibrate_to_manual": bool(model_kind == "independent1d" and bool(args.calibrate_to_manual)),
        "calibrate_to_xmin": bool(model_kind == "independent1d" and bool(args.calibrate_to_xmin)),
        "calibrate_to_xrange": bool(model_kind == "independent1d" and bool(args.calibrate_to_xrange)),
        "edu_xmin": float(args.edu_xmin) if (model_kind == "independent1d" and bool(args.calibrate_to_xmin)) else float("nan"),
        "brdu_xmin": float(args.brdu_xmin) if (model_kind == "independent1d" and bool(args.calibrate_to_xmin)) else float("nan"),
        "edu_xmax": float(args.edu_xmax) if (model_kind == "independent1d" and bool(args.calibrate_to_xrange)) else float("nan"),
        "brdu_xmax": float(args.brdu_xmax) if (model_kind == "independent1d" and bool(args.calibrate_to_xrange)) else float("nan"),
        "xmin_nearest_k": int(args.xmin_nearest_k)
        if (model_kind == "independent1d" and (bool(args.calibrate_to_xmin) or bool(args.calibrate_to_xrange)))
        else 0,
        "manual_group_cols": args.manual_group_cols if args.manual_supervision else "",
        "edu_pos_col": args.edu_pos_col if args.manual_supervision else "",
        "brdu_pos_col": args.brdu_pos_col if args.manual_supervision else "",
        "manual_sigma_edu": float(args.manual_sigma_edu) if args.manual_supervision else 0.0,
        "manual_sigma_brdu": float(args.manual_sigma_brdu) if args.manual_supervision else 0.0,
        "label_weight": 0.0 if model is None else float(model.label_weight),
        "manual_eps": float(args.manual_eps) if args.manual_supervision else 0.0,
        "learn_theta": bool(args.learn_theta) if args.manual_supervision else False,
        "manual_thresholds_tsv": str(manual_thr_tsv) if manual_thr_tsv is not None else "",
        "n_batches": int(len(batch_levels_str)),
        "ref_batch_idx": int(ref_batch),
        "ref_batch_level": "" if not batch_levels_str else batch_levels_str[int(ref_batch)],
        "fit_rows": 0 if model_kind == "independent1d" else int(fit_idx.size),
        "reg_covar": float(args.reg_covar),
        "min_beta": float(args.min_beta),
        "max_beta": float(args.max_beta),
        "min_gamma": float(args.min_gamma),
        "max_gamma": float(args.max_gamma),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "seed": int(args.seed),
        "learn_affine": bool(args.learn_affine),
        "affine_init": str(args.affine_init),
    }
    if model_kind == "joint2d":
        if model is None:
            raise RuntimeError("internal error: joint2d model missing")
        qc_meta["component_mapping"] = model.map_components_to_classes()
        qc_meta["pi"] = [float(x) for x in model.pi_]
        qc_meta["mu"] = [[float(a), float(b)] for a, b in model.mu_]
        if model.covariance == "diag":
            qc_meta["var"] = [[float(a), float(b)] for a, b in model.var_]
        else:
            qc_meta["cov"] = [[[float(x) for x in row] for row in model.cov_[k]] for k in range(model.K)]
        if hasattr(model, "ll_trace_"):
            qc_meta["ll_trace"] = [float(x) for x in model.ll_trace_]
        if hasattr(model, "theta_edu_") and hasattr(model, "theta_brdu_"):
            qc_meta["theta_edu"] = [float(x) for x in model.theta_edu_]
            qc_meta["theta_brdu"] = [float(x) for x in model.theta_brdu_]
    (cache_dir / "model_params.json").write_text(json.dumps(qc_meta, indent=2) + "\n", encoding="utf-8")

    if model_kind == "independent1d":
        if params_1d is None:
            raise RuntimeError("internal error: missing 1D params.")
        _write_tsv(cache_dir / "batchaware_1d_nuisance_by_batch.tsv", params_1d)
    else:
        if model is None:
            raise RuntimeError("internal error: missing joint2d model.")
        batch_params = pd.DataFrame(
            {
                "batch_idx": np.arange(len(batch_levels_str), dtype=np.int64),
                "batch_level": batch_levels_str,
                "N_cells": np.bincount(batch_codes, minlength=len(batch_levels_str)).astype(np.int64),
                "alpha_edu": model.alpha_[:, 0],
                "alpha_brdu": model.alpha_[:, 1],
                "beta_edu": model.beta_[:, 0],
                "beta_brdu": model.beta_[:, 1],
                "gamma": model.gamma_,
            }
        )
        _write_tsv(cache_dir / "batch_parameters.tsv", batch_params)

    if "dataset" not in df.columns:
        raise KeyError("Expected obs['dataset'] to exist so per-cell QC can be keyed by dataset-prefixed cell IDs.")
    ds = df["dataset"].astype(str).to_numpy(dtype=str, copy=False)
    idx = df.index.astype(str).to_numpy(dtype=str, copy=False)
    prefix = np.char.add(ds, ":")
    already = np.char.startswith(idx, prefix)
    cell_id = np.where(already, idx, np.char.add(prefix, idx))

    # Cache derived quantities in a compact npz (numeric only).
    np.savez_compressed(
        cache_dir / "per_cell_qc.npz",
        row_idx=df["_row_idx"].to_numpy(dtype=np.int64, copy=False),
        cell_id=cell_id,
        batch_idx=qc.batch_idx.astype(np.int16, copy=False),
        edu_x=qc.X[:, 0].astype(np.float32, copy=False),
        brdu_x=qc.X[:, 1].astype(np.float32, copy=False),
        u_edu=qc.u[:, 0].astype(np.float32, copy=False),
        u_brdu=qc.u[:, 1].astype(np.float32, copy=False),
        p_neg=qc.classes["neg"].astype(np.float32, copy=False),
        p_EdU_only=qc.classes["EdU_only"].astype(np.float32, copy=False),
        p_BrdU_only=qc.classes["BrdU_only"].astype(np.float32, copy=False),
        p_double=qc.classes["double"].astype(np.float32, copy=False),
        p_max=qc.p_max.astype(np.float32, copy=False),
        entropy=qc.entropy.astype(np.float32, copy=False),
        ll_x=qc.ll_x.astype(np.float32, copy=False),
    )
    (cache_dir / "batch_levels.json").write_text(json.dumps(batch_levels_str, indent=2) + "\n", encoding="utf-8")

    # Summaries by batch (expected proportions).
    per_batch_props = []
    for b, name in enumerate(batch_levels_str):
        idx_b = qc.batch_idx == b
        n_b = int(np.sum(idx_b))
        if n_b == 0:
            continue
        per_batch_props.append(
            {
                "batch": name,
                "N_cells": n_b,
                "prop_neg": float(np.mean(qc.classes["neg"][idx_b])),
                "prop_EdU_only": float(np.mean(qc.classes["EdU_only"][idx_b])),
                "prop_BrdU_only": float(np.mean(qc.classes["BrdU_only"][idx_b])),
                "prop_double": float(np.mean(qc.classes["double"][idx_b])),
                "mean_pmax": float(np.mean(qc.p_max[idx_b])),
                "mean_entropy": float(np.mean(qc.entropy[idx_b])),
                "mean_ll": float(np.mean(qc.ll_x[idx_b])),
            }
        )
    _write_tsv(cache_dir / "expected_props_by_batch.tsv", pd.DataFrame(per_batch_props))

    # Diagnostics: where does the model put p=0.5 (approximately) in observed intensity space?
    # We summarize the observed x among cells with 0.45 <= p <= 0.55, per batch.
    p05_rows = []
    pE_all = (qc.classes["EdU_only"] + qc.classes["double"]).astype(np.float64, copy=False)
    pB_all = (qc.classes["BrdU_only"] + qc.classes["double"]).astype(np.float64, copy=False)
    for b, name in enumerate(batch_levels_str):
        idx_b = qc.batch_idx == b
        if not np.any(idx_b):
            continue
        edu_mid = idx_b & (pE_all >= 0.45) & (pE_all <= 0.55)
        brdu_mid = idx_b & (pB_all >= 0.45) & (pB_all <= 0.55)

        row = {"batch": str(name), "N_cells": int(np.sum(idx_b))}
        if int(np.sum(edu_mid)) >= 200:
            x = qc.X[edu_mid, 0].astype(np.float64, copy=False)
            row.update(
                {
                    "N_mid_edu": int(np.sum(edu_mid)),
                    "edu_x_p05_med": float(np.median(x)),
                    "edu_x_p05_q25": float(np.quantile(x, 0.25)),
                    "edu_x_p05_q75": float(np.quantile(x, 0.75)),
                }
            )
        else:
            row.update({"N_mid_edu": int(np.sum(edu_mid)), "edu_x_p05_med": np.nan, "edu_x_p05_q25": np.nan, "edu_x_p05_q75": np.nan})

        if int(np.sum(brdu_mid)) >= 200:
            x = qc.X[brdu_mid, 1].astype(np.float64, copy=False)
            row.update(
                {
                    "N_mid_brdu": int(np.sum(brdu_mid)),
                    "brdu_x_p05_med": float(np.median(x)),
                    "brdu_x_p05_q25": float(np.quantile(x, 0.25)),
                    "brdu_x_p05_q75": float(np.quantile(x, 0.75)),
                }
            )
        else:
            row.update(
                {
                    "N_mid_brdu": int(np.sum(brdu_mid)),
                    "brdu_x_p05_med": np.nan,
                    "brdu_x_p05_q25": np.nan,
                    "brdu_x_p05_q75": np.nan,
                }
            )
        p05_rows.append(row)
    _write_tsv(cache_dir / "p05_x_by_batch.tsv", pd.DataFrame(p05_rows))

    # Benchmark soft calls against existing hard calls (edu_pos/brdu_pos) if available.
    benchmark_manual_tsv: Path | None = None
    if not bool(args.no_benchmark_manual):
        if edu_hard is None or brdu_hard is None:
            raise RuntimeError("internal error: expected edu_hard/brdu_hard to be set when benchmarking is enabled.")
        edu_hard = edu_hard.astype(np.int8, copy=False)
        brdu_hard = brdu_hard.astype(np.int8, copy=False)

        pE = (qc.classes["EdU_only"] + qc.classes["double"]).astype(np.float64, copy=False)
        pB = (qc.classes["BrdU_only"] + qc.classes["double"]).astype(np.float64, copy=False)
        P_quad = np.stack([qc.classes["neg"], qc.classes["EdU_only"], qc.classes["BrdU_only"], qc.classes["double"]], axis=1)
        quad_map = np.argmax(P_quad, axis=1).astype(np.int8, copy=False)
        pE_map = np.isin(quad_map, [1, 3]).astype(np.int8, copy=False)  # EdU_only or double
        pB_map = np.isin(quad_map, [2, 3]).astype(np.int8, copy=False)  # BrdU_only or double

        hard_quad = edu_hard + 2 * brdu_hard
        soft_quad_mass = np.stack(
            [qc.classes["neg"], qc.classes["EdU_only"], qc.classes["BrdU_only"], qc.classes["double"]], axis=1
        )
        soft_quad_global = soft_quad_mass.mean(axis=0)
        hard_quad_global = np.bincount(hard_quad, minlength=4).astype(np.float64) / float(hard_quad.size)

        rows = [
            {
                "scope": "global",
                "batch": "",
                "n_cells": int(hard_quad.size),
                "hard_edu_frac": float(np.mean(edu_hard)),
                "soft_edu_mass": float(np.mean(pE)),
                "soft_edu_map_frac": float(np.mean(pE_map)),
                "hard_brdu_frac": float(np.mean(brdu_hard)),
                "soft_brdu_mass": float(np.mean(pB)),
                "soft_brdu_map_frac": float(np.mean(pB_map)),
                "hard_neg_frac": float(hard_quad_global[0]),
                "hard_edu_only_frac": float(hard_quad_global[1]),
                "hard_brdu_only_frac": float(hard_quad_global[2]),
                "hard_double_frac": float(hard_quad_global[3]),
                "soft_neg_mass": float(soft_quad_global[0]),
                "soft_edu_only_mass": float(soft_quad_global[1]),
                "soft_brdu_only_mass": float(soft_quad_global[2]),
                "soft_double_mass": float(soft_quad_global[3]),
            }
        ]

        for b, name in enumerate(batch_levels_str):
            idx_b = qc.batch_idx == b
            n_b = int(np.sum(idx_b))
            if n_b == 0:
                continue
            hard_b = hard_quad[idx_b]
            hard_frac = np.bincount(hard_b, minlength=4).astype(np.float64) / float(n_b)
            soft_mass = soft_quad_mass[idx_b].mean(axis=0)
            rows.append(
                {
                    "scope": "batch",
                    "batch": name,
                    "n_cells": n_b,
                    "hard_edu_frac": float(np.mean(edu_hard[idx_b])),
                    "soft_edu_mass": float(np.mean(pE[idx_b])),
                    "soft_edu_map_frac": float(np.mean(pE_map[idx_b])),
                    "hard_brdu_frac": float(np.mean(brdu_hard[idx_b])),
                    "soft_brdu_mass": float(np.mean(pB[idx_b])),
                    "soft_brdu_map_frac": float(np.mean(pB_map[idx_b])),
                    "hard_neg_frac": float(hard_frac[0]),
                    "hard_edu_only_frac": float(hard_frac[1]),
                    "hard_brdu_only_frac": float(hard_frac[2]),
                    "hard_double_frac": float(hard_frac[3]),
                    "soft_neg_mass": float(soft_mass[0]),
                    "soft_edu_only_mass": float(soft_mass[1]),
                    "soft_brdu_only_mass": float(soft_mass[2]),
                    "soft_double_mass": float(soft_mass[3]),
                }
            )

        benchmark_manual_tsv = cache_dir / "benchmark_vs_manual.tsv"
        _write_tsv(benchmark_manual_tsv, pd.DataFrame(rows))
        print("Benchmark vs manual (global):")
        print(
            f"  EdU hard={np.mean(edu_hard):.3f}  soft_mass={np.mean(pE):.3f}  soft_map={np.mean(pE_map):.3f}"
        )
        print(
            f"  BrdU hard={np.mean(brdu_hard):.3f} soft_mass={np.mean(pB):.3f} soft_map={np.mean(pB_map):.3f}"
        )

    manual_vs_model_tsv: Path | None = None
    manual_vs_model_weighted_mae: float | None = None
    manual_global_props: dict[str, float] | None = None
    model_global_props: dict[str, float] | None = None
    if args.manual_supervision:
        if codes_g is None or levels_g is None or edu_pos is None or brdu_pos is None:
            raise RuntimeError("internal error: manual supervision enabled but manual group codes/labels not available")

        edu_pos_i = edu_pos.astype(np.int8, copy=False)
        brdu_pos_i = brdu_pos.astype(np.int8, copy=False)
        manual_cls = edu_pos_i + 2 * brdu_pos_i  # 0=neg,1=edu_only,2=brdu_only,3=double
        n_groups = int(levels_g.size)
        counts_g = np.bincount(codes_g, minlength=n_groups).astype(np.int64, copy=False)
        denom = np.clip(counts_g.astype(np.float64), 1.0, np.inf)

        manual_props = np.zeros((n_groups, 4), dtype=np.float64)
        for k in range(4):
            manual_props[:, k] = np.bincount(codes_g, weights=(manual_cls == k).astype(np.float64), minlength=n_groups) / denom

        model_props = np.zeros((n_groups, 4), dtype=np.float64)
        model_props[:, 0] = np.bincount(codes_g, weights=qc.classes["neg"], minlength=n_groups) / denom
        model_props[:, 1] = np.bincount(codes_g, weights=qc.classes["EdU_only"], minlength=n_groups) / denom
        model_props[:, 2] = np.bincount(codes_g, weights=qc.classes["BrdU_only"], minlength=n_groups) / denom
        model_props[:, 3] = np.bincount(codes_g, weights=qc.classes["double"], minlength=n_groups) / denom

        key_parts = pd.Series(levels_g.astype(str)).str.split("|", expand=True)
        if key_parts.shape[1] != len(manual_group_cols):
            raise ValueError(
                f"manual group key split produced {key_parts.shape[1]} columns; expected {len(manual_group_cols)}"
            )
        key_parts.columns = list(manual_group_cols)

        out = key_parts.copy()
        out["N"] = counts_g
        for k, name in enumerate(["neg", "edu_only", "brdu_only", "double"]):
            out[f"manual_{name}"] = manual_props[:, k]
            out[f"model_{name}"] = model_props[:, k]
            out[f"delta_{name}"] = out[f"model_{name}"] - out[f"manual_{name}"]
        out["mae"] = out[[f"delta_{n}" for n in ["neg", "edu_only", "brdu_only", "double"]]].abs().mean(axis=1)

        group_name = "_".join(manual_group_cols)
        manual_vs_model_tsv = cache_dir / f"manual_vs_model_by_{group_name}.tsv"
        out.to_csv(manual_vs_model_tsv, sep="\t", index=False)

        manual_vs_model_weighted_mae = float(np.average(out["mae"], weights=out["N"]))

        manual_global = manual_props.T @ counts_g.astype(np.float64) / float(counts_g.sum())
        model_global = model_props.T @ counts_g.astype(np.float64) / float(counts_g.sum())
        manual_global_props = {k: float(v) for k, v in zip(["neg", "edu_only", "brdu_only", "double"], manual_global)}
        model_global_props = {k: float(v) for k, v in zip(["neg", "edu_only", "brdu_only", "double"], model_global)}

    # Core plots.
    if model_kind == "independent1d":
        plot_global_hexbin_x_calls(
            out_png=plot_dir / "global_x_hexbin_scatter.png",
            qc=qc,
            batch_levels=batch_levels_str,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
            edu_map_thr=float(args.edu_map_thr),
            brdu_map_thr=float(args.brdu_map_thr),
        )
        if args.facet_group_cols is not None:
            facet_cols = _parse_list(args.facet_group_cols)
            if not facet_cols:
                raise ValueError("--facet-group-cols must be non-empty when set")
            missing = sorted(set(facet_cols) - set(df.columns))
            if missing:
                raise KeyError(f"Missing obs columns requested for faceting: {missing}")

            key = df[facet_cols[0]].astype(str).to_numpy()
            for c in facet_cols[1:]:
                key = key + "|" + df[c].astype(str).to_numpy()
            codes, levels = pd.factorize(key, sort=False)
            levels_str = [str(x) for x in levels]
            plot_group_facets_x_calls(
                out_png=plot_dir / f"facets_x_space_by_{'_'.join(facet_cols)}.png",
                qc=qc,
                group_codes=codes,
                group_levels=levels_str,
                max_groups=int(args.facet_max_groups),
                max_cells_per_group=int(args.facet_max_cells_per_group),
                seed=int(args.seed),
                edu_map_thr=float(args.edu_map_thr),
                brdu_map_thr=float(args.brdu_map_thr),
            )

        plot_uncertainty_hists(out_dir=plot_dir, qc=qc)
        plot_uncertainty_scatter_u(
            out_png=plot_dir / "uncertainty_scatter_u.png",
            qc=qc,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
        )
        plot_loglik_vs_intensity(
            out_png=plot_dir / "loglik_vs_intensity.png",
            qc=qc,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
        )
    else:
        if model is None:
            raise RuntimeError("internal error: missing joint2d model.")
        m_x, var_x = model.component_params_in_x_space()
        plot_global_hexbin_x(
            out_png=plot_dir / "global_x_hexbin_scatter.png",
            qc=qc,
            model=model,
            batch_show=ref_batch,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
            draw_contours=not bool(args.no_contours),
        )
        plot_global_hexbin_u(
            out_png=plot_dir / "global_u_hexbin_scatter.png",
            qc=qc,
            model=model,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
            use_gamma=True,
            batch_show=ref_batch,
        )

        # Per-batch facets: x-space with per-batch ellipses, u-space with shared ellipses (+gamma).
        plot_batch_facets(
            out_png=plot_dir / "facets_x_space.png",
            qc_xy=qc.X,
            batch_idx=qc.batch_idx,
            batch_levels=batch_levels_str,
            ell_means=m_x,
            ell_vars=var_x,
            title="Per-batch observed space (x) with fitted 1-sigma ellipses",
            xlabel="edu_x",
            ylabel="brdu_x",
            max_cells_per_batch=int(args.max_cells_per_batch),
            seed=int(args.seed),
        )

        # For u-space facets, use shared mu/var and optionally gamma scaling per batch.
        ell_means_u = np.tile(model.mu_[None, :, :], (len(batch_levels_str), 1, 1))
        if model.covariance == "diag":
            ell_vars_u = np.empty_like(ell_means_u)
            for b in range(len(batch_levels_str)):
                ell_vars_u[b] = model.gamma_[b] * model.var_
        else:
            ell_vars_u = np.empty((len(batch_levels_str), model.K, 2, 2), dtype=np.float64)
            for b in range(len(batch_levels_str)):
                ell_vars_u[b] = model.gamma_[b] * model.cov_

        plot_batch_facets(
            out_png=plot_dir / "facets_u_space.png",
            qc_xy=qc.u,
            batch_idx=qc.batch_idx,
            batch_levels=batch_levels_str,
            ell_means=ell_means_u,
            ell_vars=ell_vars_u,
            title="Per-batch latent space (u) with shared ellipses (scaled by gamma_b)",
            xlabel="u_edu",
            ylabel="u_brdu",
            max_cells_per_batch=int(args.max_cells_per_batch),
            seed=int(args.seed),
        )

        if args.facet_group_cols is not None:
            facet_cols = _parse_list(args.facet_group_cols)
            if not facet_cols:
                raise ValueError("--facet-group-cols must be non-empty when set")
            missing = sorted(set(facet_cols) - set(df.columns))
            if missing:
                raise KeyError(f"Missing obs columns requested for faceting: {missing}")

            key = df[facet_cols[0]].astype(str).to_numpy()
            for c in facet_cols[1:]:
                key = key + "|" + df[c].astype(str).to_numpy()
            codes, levels = pd.factorize(key, sort=False)
            levels_str = [str(x) for x in levels]
            plot_group_facets_x(
                out_png=plot_dir / f"facets_x_space_by_{'_'.join(facet_cols)}.png",
                qc=qc,
                model=model,
                group_codes=codes,
                group_levels=levels_str,
                max_groups=int(args.facet_max_groups),
                max_cells_per_group=int(args.facet_max_cells_per_group),
                seed=int(args.seed),
                draw_contours=not bool(args.no_contours),
            )

        plot_nuisance_params(out_dir=plot_dir, model=model, batch_levels=batch_levels_str)
        plot_uncertainty_hists(out_dir=plot_dir, qc=qc)
        plot_uncertainty_scatter_u(
            out_png=plot_dir / "uncertainty_scatter_u.png",
            qc=qc,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
        )
        plot_loglik_vs_intensity(
            out_png=plot_dir / "loglik_vs_intensity.png",
            qc=qc,
            max_scatter=int(args.max_scatter),
            seed=int(args.seed),
        )
        plot_component_health(out_dir=plot_dir, qc=qc, model=model)
        plot_residual_normality(out_png=plot_dir / "residual_normality.png", qc=qc, model=model, seed=int(args.seed))
        plot_simulated_vs_observed_marginals(out_dir=plot_dir, qc=qc, model=model, seed=int(args.seed))

    summary = {
        "input_h5ad": "" if h5ad is None else str(h5ad),
        "input_obs_parquet": "" if args.obs_parquet is None else str(Path(args.obs_parquet).expanduser()),
        "model": model_kind,
        "n_obs_total": int(df.shape[0]),
        "fit_rows": int(fit_idx.size),
        "n_batches": int(len(batch_levels_str)),
        "ref_batch_level": batch_levels_str[ref_batch],
        "mean_pmax": float(np.mean(qc.p_max)),
        "mean_entropy": float(np.mean(qc.entropy)),
        "mean_ll": float(np.mean(qc.ll_x)),
        "out_dir": str(out_dir),
        "cache_model_params": str(cache_dir / "model_params.json"),
        "cache_per_cell_qc": str(cache_dir / "per_cell_qc.npz"),
        "plots_dir": str(plot_dir),
    }
    if benchmark_manual_tsv is not None:
        summary["benchmark_manual_tsv"] = str(benchmark_manual_tsv)
    if manual_vs_model_tsv is not None:
        summary["manual_vs_model_tsv"] = str(manual_vs_model_tsv)
    if manual_vs_model_weighted_mae is not None:
        summary["manual_vs_model_weighted_mae"] = float(manual_vs_model_weighted_mae)
    if manual_global_props is not None:
        for k, v in manual_global_props.items():
            summary[f"manual_global_{k}"] = float(v)
    if model_global_props is not None:
        for k, v in model_global_props.items():
            summary[f"model_global_{k}"] = float(v)
    _write_summary_tsv(out_dir / "summary.tsv", summary)

    print(f"Wrote: {out_dir / 'summary.tsv'}")
    print(f"Wrote cache: {cache_dir / 'per_cell_qc.npz'}")
    print(f"Wrote plots under: {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
