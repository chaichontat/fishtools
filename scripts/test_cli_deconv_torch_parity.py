"""Torch vs CuPy Lucy-Richardson parity on production tile/PSF."""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import tifffile as tiff
import torch
from cupyx.scipy import ndimage as cnd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fishtools.preprocess.deconv.torch_lr import deconvolve_lucyrichardson_guo_torch


def deconvolve_lucyrichardson_guo_cupy(
    img: cp.ndarray,
    projectors: tuple[cp.ndarray, cp.ndarray],
    iters: int = 1,
    *,
    eps: float = 1e-6,
    i_max: float | None = None,
) -> cp.ndarray:
    assert iters == 1, "Only single-iter LR is implemented here."
    fwd, bwd = projectors
    estimate = cp.clip(img, eps, None)
    filtered = cnd.convolve(estimate, fwd, mode="reflect")
    filtered = cp.clip(filtered, eps, i_max) if i_max is not None else cp.maximum(filtered, eps)
    ratio = img / filtered
    correction = cnd.convolve(ratio, bwd, mode="reflect")
    return estimate * correction


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse == 0:
        return float("inf")
    data_range = max(float(np.max(a) - np.min(a)), 1e-12)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


IMG_PATH = Path("/raid/chaichontat/20250919_Coro2/2_10_18--1/2_10_18-0025.tif")
PSF_PATH = Path("/home/chaichontat/fishtools/data/PSF GL.tif")
IMG_SLICE = np.s_[:-2:3, :512, :512]
IMG_Z_STEP_UM = 0.4
PSF_Z_STEP_UM = 0.1
PSF_HALF_PLANES = 6


@lru_cache(maxsize=1)
def _load_structured_tile() -> np.ndarray:
    raw = tiff.imread(IMG_PATH)
    cropped = raw[IMG_SLICE].astype(np.float32)
    return cropped / 65535.0


@lru_cache(maxsize=1)
def _load_z_matched_psf() -> tuple[np.ndarray, np.ndarray]:
    psf = tiff.imread(PSF_PATH).astype(np.float32)
    center = psf.shape[0] // 2
    stride = max(1, int(round(IMG_Z_STEP_UM / PSF_Z_STEP_UM)))
    max_half = min(center // stride, (psf.shape[0] - center - 1) // stride)
    half = min(PSF_HALF_PLANES, max_half)
    start = center - half * stride
    stop = center + half * stride + 1
    psf_down = psf[start:stop:stride].copy()
    psf_down /= psf_down.sum()
    bwd = psf_down[::-1, ::-1, ::-1].copy()
    return psf_down, bwd


@pytest.mark.cuda
def test_lr_single_iter_parity_torch_fp32_vs_cupy_real_tile():
    img_np = _load_structured_tile()
    psf_np, bwd_np = _load_z_matched_psf()

    img_cp = cp.asarray(img_np, dtype=cp.float32)
    fwd_cp = cp.asarray(psf_np, dtype=cp.float32)
    bwd_cp = cp.asarray(bwd_np, dtype=cp.float32)
    out_cp = deconvolve_lucyrichardson_guo_cupy(img_cp, (fwd_cp, bwd_cp), iters=1, eps=1e-6)
    out_ref = cp.asnumpy(out_cp)

    device = torch.device("cuda")
    img_t = torch.from_numpy(img_np).to(device=device, dtype=torch.float32)
    fwd_t = torch.from_numpy(psf_np).to(device=device, dtype=torch.float32)
    bwd_t = torch.from_numpy(bwd_np).to(device=device, dtype=torch.float32)
    out_t = deconvolve_lucyrichardson_guo_torch(
        img_t,
        (fwd_t, bwd_t),
        iters=1,
        eps=1e-6,
        i_max=None,
        enable_tf32=False,
        flip_kernels=True,
        debug=False,
    )
    out_t_np = out_t.detach().cpu().numpy()

    rel = float(np.linalg.norm(out_ref - out_t_np) / (np.linalg.norm(out_ref) + 1e-12))
    peak = psnr(out_ref, out_t_np)
    max_abs = float(np.max(np.abs(out_ref - out_t_np)))

    assert rel < 1.5e-2
    assert peak >= 42.0
    assert max_abs < 4.0e-2
