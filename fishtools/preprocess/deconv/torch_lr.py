from __future__ import annotations

from contextlib import contextmanager

import torch

try:
    profile
except NameError:
    profile = lambda x: x


def _flip_for_convolution(kernel: torch.Tensor) -> torch.Tensor:
    return kernel.flip(dims=tuple(range(kernel.ndim)))


def _pad_tuple_for_reflect(kernel_shape: tuple[int, ...]) -> tuple[int, ...]:
    pads = []
    for k in reversed(kernel_shape):
        left = k // 2
        right = k - 1 - left
        pads.extend([left, right])
    return tuple(pads)


def _maybe_squeeze_channel(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0], True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}; expected (Z,Y,X) or (Z,1,Y,X).")


def _maybe_unsqueeze_channel(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return x.unsqueeze(1) if squeezed else x


def _convn_reflect(x: torch.Tensor, kernel: torch.Tensor, *, flip_kernel: bool) -> torch.Tensor:
    if x.ndim not in (2, 3):
        raise ValueError("_convn_reflect expects 2D or 3D tensors after squeezing channel axis.")
    if kernel.ndim != x.ndim:
        raise ValueError("Kernel dimensionality must match input dimensionality.")

    kernel = kernel.to(device=x.device, dtype=torch.float32)
    if flip_kernel:
        kernel = _flip_for_convolution(kernel)

    x4or5 = x.to(torch.float32).unsqueeze(0).unsqueeze(0)
    w = kernel.view(1, 1, *kernel.shape)
    pads = _pad_tuple_for_reflect(kernel.shape)
    x_pad = torch.nn.functional.pad(x4or5, pads, mode="reflect")
    if x.ndim == 2:
        y = torch.nn.functional.conv2d(x_pad, w)
    else:
        y = torch.nn.functional.conv3d(x_pad, w)
    return y[0, 0]


@contextmanager
def _tf32_context(enabled: bool):
    if not (enabled and torch.cuda.is_available()):
        yield
        return

    prev_matmul_allow = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    prev_cudnn_precision = torch.backends.cudnn.fp32_precision  # type: ignore[attr-defined]
    torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_allow
        torch.backends.cudnn.fp32_precision = prev_cudnn_precision  # type: ignore[attr-defined]


@torch.inference_mode()
@profile
def deconvolve_lucyrichardson_guo_torch(
    img: torch.Tensor,
    projectors: tuple[torch.Tensor, torch.Tensor],
    *,
    iters: int = 1,
    eps: float = 1e-6,
    i_max: float | None = None,
    enable_tf32: bool = False,
    flip_kernels: bool = True,
    debug: bool = False,
) -> torch.Tensor:
    if iters != 1:
        raise NotImplementedError("Only iters=1 is implemented for the Torch LR kernel.")

    fwd, bwd = projectors
    device = img.device
    fwd = fwd.to(device=device, dtype=torch.float32)
    bwd = bwd.to(device=device, dtype=torch.float32)
    img = img.to(device=device, dtype=torch.float32)

    squeezed_img, squeezed = _maybe_squeeze_channel(img)
    squeezed_fwd, _ = _maybe_squeeze_channel(fwd)
    squeezed_bwd, _ = _maybe_squeeze_channel(bwd)

    estimate = torch.clamp(squeezed_img, min=eps)
    with _tf32_context(enable_tf32 and estimate.is_cuda):
        filtered = _convn_reflect(estimate, squeezed_fwd, flip_kernel=flip_kernels)

    filtered = torch.clamp(filtered, min=eps) if i_max is None else torch.clamp(filtered, min=eps, max=i_max)
    ratio = squeezed_img / filtered

    with _tf32_context(enable_tf32 and estimate.is_cuda):
        correction = _convn_reflect(ratio, squeezed_bwd, flip_kernel=flip_kernels)

    if debug:
        print(
            f"[Torch LR] estimate: min={estimate.min().item():.6f} max={estimate.max().item():.6f} mean={estimate.mean().item():.6f}"
        )
        print(
            f"[Torch LR] filtered: min={filtered.min().item():.6f} max={filtered.max().item():.6f} mean={filtered.mean().item():.6f}"
        )
        print(
            f"[Torch LR] ratio: min={ratio.min().item():.6f} max={ratio.max().item():.6f} mean={ratio.mean().item():.6f}"
        )
        print(
            f"[Torch LR] correction: min={correction.min().item():.6f} max={correction.max().item():.6f} mean={correction.mean().item():.6f}"
        )

    result = estimate * correction
    return _maybe_unsqueeze_channel(result, squeezed)
