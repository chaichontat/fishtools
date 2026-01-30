# %% [markdown]
# # SyN morph: Image #2 → Image #1 (ANTsPy, first channel)
#
# Runs a SyN registration to morph a *moving* 2D image onto a *fixed* 2D image using ANTsPy.
# This is designed for same-modality images with minor sample differences (robust normalization + masks + CC).
#
# Run cells sequentially. Each phase writes artifacts to `OUTDIR` for inspection and fast reruns.

# %%
from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import ants
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from roifile import ImagejRoi, ROI_TYPE, roiread, roiwrite
from scipy import ndimage as ndi

# %%
# === EDIT THESE ===
#
# Inputs: save the two chat images to disk, then point these paths at them.
#
# Convention for this workflow:
# - FIXED = Image #1 (target)
# - MOVING = Image #2 (to be morphed onto fixed)
FIXED_PATH = Path(
    "/working/20251125_JaxA6_Coro8/analysis/output/ccf-transforms/1/landmark_syn_mi/mask_edit/warped_moving_z5_ds8_target2um.png"
)
MOVING_PATH = Path(
    "/working/20251125_JaxA6_Coro8/analysis/output/ccf-transforms/6/landmark_syn_mi/mask_edit/warped_moving_z5_ds8_target2um.png"
)

# Use the "first channel" only (RGB -> channel 0). If your images are grayscale, this is ignored.
CHANNEL = 0

# Outputs
OUTDIR = Path("/working/20251125_JaxA6_Coro8/analysis/output/ccf-transforms/6/warp_to_ours")

# Export a stable “output format” bundle (transforms + warped + QC + manifest).
# If `EXPORT_BASENAME` is None, it is inferred from the input paths.
EXPORT_ROOT = OUTDIR / "exports"
EXPORT_BASENAME: str | None = None

# Controls (safe defaults; adjust as needed)
RANDOM_SEED = 0
N_THREADS = 8
DOWNSAMPLE_FOR_REG = 1  # 1 = full-res; 2–4 recommended for faster iteration

# Optional COM (mask) translation initialization before affine.
USE_COM_TRANSLATION_INIT = True

USE_AFFINE_INITIALIZER = False
AFFINE_INIT_SEARCH_FACTOR = 10
AFFINE_INIT_RADIAN_FRACTION = 0.5
AFFINE_INIT_USE_PRINCIPAL_AXIS = True
AFFINE_INIT_LOCAL_SEARCH_ITERS = 100

# Masks: read `mask_edit/full_mask.tif` next to each input image.
FULLMASK_FILENAME = "full_mask.tif"

# ROI morphing: read moving-space polygons from `RoiSet.zip` next to MOVING_PATH and write a morphed copy.
ROISET_IN_FILENAME = "RoiSet.zip"
ROISET_MORPHED_FILENAME = "RoiSetMorphed.zip"
ROI_MORPHED_SUFFIX = "_morphed"

# QC / metrics
CHECKER_TILE_PX = 64
MI_BINS = 64
EDGE_SIGMA_PX = 2.0

# SyN profile (same-modality -> CC)
SYN_SPEC: dict[str, Any] = {
    "name": "syn_cc_r4_g0p20",
    "type_of_transform": "SyN",
    "syn_metric": "cc",
    "syn_sampling": 4,  # CC radius
    "grad_step": 0.20,
    "flow_sigma": 3.0,
    "total_sigma": 0.0,
    "reg_iterations": (200, 120, 60, 20),
}


os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(int(N_THREADS)))
os.environ.setdefault("OMP_NUM_THREADS", str(int(N_THREADS)))

# Allow overriding key config via env vars (for batch runs).
_fixed_env = os.environ.get("ANTS_SYN_FIXED_PATH")
if _fixed_env:
    FIXED_PATH = Path(_fixed_env)
_moving_env = os.environ.get("ANTS_SYN_MOVING_PATH")
if _moving_env:
    MOVING_PATH = Path(_moving_env)
_channel_env = os.environ.get("ANTS_SYN_CHANNEL")
if _channel_env is not None:
    CHANNEL = int(_channel_env)
_outdir_env = os.environ.get("ANTS_SYN_OUTDIR")
if _outdir_env:
    OUTDIR = Path(_outdir_env)
_export_root_env = os.environ.get("ANTS_SYN_EXPORT_ROOT")
if _export_root_env:
    EXPORT_ROOT = Path(_export_root_env)
else:
    EXPORT_ROOT = OUTDIR
_export_basename_env = os.environ.get("ANTS_SYN_EXPORT_BASENAME")
if _export_basename_env:
    EXPORT_BASENAME = str(_export_basename_env)

_use_affine_init_env = os.environ.get("ANTS_SYN_USE_AFFINE_INITIALIZER")
if _use_affine_init_env:
    USE_AFFINE_INITIALIZER = str(_use_affine_init_env).strip().lower() in {"1", "true", "yes", "y"}

OUTDIR.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return v


_ENV_KEYS = [
    "ANTS_SYN_FIXED_PATH",
    "ANTS_SYN_MOVING_PATH",
    "ANTS_SYN_CHANNEL",
    "ANTS_SYN_OUTDIR",
    "ANTS_SYN_EXPORT_ROOT",
    "ANTS_SYN_EXPORT_BASENAME",
    "ANTS_SYN_USE_AFFINE_INITIALIZER",
    "ANTS_SYN_ROISET_IN",
    "ANTS_SYN_ROISET_OUT",
]

def _infer_roi_token(path: Path) -> str | None:
    parts = path.resolve().parts
    try:
        i = parts.index("ccf-transforms")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    return str(parts[i + 1])


RUN_CONFIG_PATH = OUTDIR / "run_config.json"
RUN_CONFIG_PATH.write_text(
    json.dumps(
        {
            "reference": {
                "role": "fixed",
                "fixed_path": str(FIXED_PATH),
                "moving_path": str(MOVING_PATH),
                "fixed_roi": _infer_roi_token(FIXED_PATH),
                "moving_roi": _infer_roi_token(MOVING_PATH),
            },
            "paths": {
                "fixed_path": str(FIXED_PATH),
                "moving_path": str(MOVING_PATH),
                "fixed_fullmask_path": str(FIXED_PATH.with_name(FULLMASK_FILENAME)),
                "moving_fullmask_path": str(MOVING_PATH.with_name(FULLMASK_FILENAME)),
                "roiset_in_path": os.environ.get("ANTS_SYN_ROISET_IN")
                or str(FIXED_PATH.with_name(ROISET_IN_FILENAME)),
                "roiset_out_path": os.environ.get("ANTS_SYN_ROISET_OUT")
                or str(MOVING_PATH.with_name(ROISET_MORPHED_FILENAME)),
                "outdir": str(OUTDIR.resolve()),
                "export_root": str(EXPORT_ROOT.resolve()),
                "export_basename": EXPORT_BASENAME,
            },
            "runtime": {
                "random_seed": int(RANDOM_SEED),
                "n_threads": int(N_THREADS),
                "downsample_for_reg": int(DOWNSAMPLE_FOR_REG),
                "use_com_translation_init": bool(USE_COM_TRANSLATION_INIT),
                "use_affine_initializer": bool(USE_AFFINE_INITIALIZER),
                "affine_initializer": {
                    "search_factor": int(AFFINE_INIT_SEARCH_FACTOR),
                    "radian_fraction": float(AFFINE_INIT_RADIAN_FRACTION),
                    "use_principal_axis": bool(AFFINE_INIT_USE_PRINCIPAL_AXIS),
                    "local_search_iters": int(AFFINE_INIT_LOCAL_SEARCH_ITERS),
                },
            },
            "inputs": {
                "channel": int(CHANNEL),
                "mask_filename": str(FULLMASK_FILENAME),
            },
            "roi_morph": {
                "direction": "fixed_to_moving",
                "roiset_in_filename": str(ROISET_IN_FILENAME),
                "roiset_morphed_filename": str(ROISET_MORPHED_FILENAME),
                "suffix": str(ROI_MORPHED_SUFFIX),
            },
            "qc": {
                "checker_tile_px": int(CHECKER_TILE_PX),
                "mi_bins": int(MI_BINS),
                "edge_sigma_px": float(EDGE_SIGMA_PX),
            },
            "syn": _jsonable(SYN_SPEC),
            "versions": {
                "ants": getattr(ants, "__version__", None),
                "python": sys.version,
            },
            "env_overrides": {k: os.environ.get(k) for k in _ENV_KEYS if os.environ.get(k) is not None},
        },
        indent=2,
    ),
    encoding="utf-8",
)


# %%
def load_first_channel(path: Path, *, channel: int) -> np.ndarray:
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3:
        if not (0 <= channel < arr.shape[-1]):
            raise ValueError(f"Requested channel={channel}, but image has shape={arr.shape}.")
        out = arr[..., channel]
    else:
        raise ValueError(f"Unsupported image ndim={arr.ndim} for path={path}.")

    out_f = out.astype(np.float32, copy=False)
    if not np.isfinite(out_f).all():
        raise ValueError(f"Non-finite values in {path}.")
    return out_f


def robust_normalize01(x: np.ndarray, *, q_lo: float = 0.01, q_hi: float = 0.99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.quantile(x, (q_lo, q_hi)).tolist()
    lo = float(lo)
    hi = float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"Bad quantiles: lo={lo}, hi={hi}.")
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def infer_roi_from_ccf_path(path: Path) -> str | None:
    parts = list(path.resolve().parts)
    if "ccf-transforms" not in parts:
        return None
    i = parts.index("ccf-transforms")
    if i + 1 >= len(parts):
        return None
    return str(parts[i + 1])


def load_mask_from_fullmask_tif(*, image_path: Path, shape_yx: tuple[int, int]) -> np.ndarray:
    mask_path = image_path.with_name(FULLMASK_FILENAME)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask file {mask_path} (expected next to {image_path.name}).")

    arr = tifffile.imread(mask_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask tif at {mask_path}, got shape={arr.shape}.")
    if tuple(arr.shape) != tuple(shape_yx):
        raise ValueError(f"Mask shape mismatch: mask={arr.shape}, image={shape_yx} at {mask_path}.")
    return (np.asarray(arr) > 0).astype(bool)


def sanitize_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._-") or "export"


def infer_export_basename(*, fixed_path: Path, moving_path: Path, channel: int) -> str:
    fixed_roi = infer_roi_from_ccf_path(fixed_path) or "fixed"
    moving_roi = infer_roi_from_ccf_path(moving_path) or "moving"

    stem = moving_path.stem
    if stem.startswith("warped_moving_"):
        stem = stem.removeprefix("warped_moving_")

    return sanitize_slug(f"syn_{stem}_{moving_roi}_to_{fixed_roi}_ch{int(channel)}")


def downsample(
    img: np.ndarray, *, factor: int, order: int
) -> np.ndarray:
    factor = int(factor)
    if factor <= 1:
        return np.asarray(img)
    zoom = 1.0 / float(factor)
    return ndi.zoom(img, zoom=zoom, order=int(order))


def ncc(a: np.ndarray, b: np.ndarray, *, mask: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError(f"NCC shape mismatch: a={a.shape}, b={b.shape}, mask={m.shape}.")
    if int(m.sum()) < 256:
        return float("nan")
    aa = a[m]
    bb = b[m]
    aa = aa - float(aa.mean())
    bb = bb - float(bb.mean())
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))) + 1e-12
    return float(np.sum(aa * bb) / denom)


def masked_mi_hist2d(a: np.ndarray, b: np.ndarray, *, mask: np.ndarray, bins: int) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or a.shape != m.shape:
        raise ValueError(f"MI shape mismatch: a={a.shape}, b={b.shape}, mask={m.shape}.")
    if int(m.sum()) < 256:
        return float("nan")

    aa = a[m]
    bb = b[m]
    ok = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[ok]
    bb = bb[ok]
    if aa.size < 256:
        return float("nan")

    lo_a, hi_a = np.quantile(aa, (0.005, 0.995)).tolist()
    lo_b, hi_b = np.quantile(bb, (0.005, 0.995)).tolist()
    if float(hi_a) <= float(lo_a) or float(hi_b) <= float(lo_b):
        return float("nan")

    h, _, _ = np.histogram2d(
        aa,
        bb,
        bins=int(bins),
        range=((float(lo_a), float(hi_a)), (float(lo_b), float(hi_b))),
    )
    s = float(h.sum())
    if s <= 0:
        return float("nan")
    pxy = h / s
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))


def checkerboard(a01: np.ndarray, b01: np.ndarray, *, tile_px: int) -> np.ndarray:
    a = np.asarray(a01, dtype=np.float32)
    b = np.asarray(b01, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Checkerboard shape mismatch: a={a.shape}, b={b.shape}.")
    tile_px = max(4, int(tile_px))
    yy, xx = np.indices(a.shape)
    use_a = ((yy // tile_px) + (xx // tile_px)) % 2 == 0
    out = np.where(use_a, a, b)
    return out


def center_of_mass_xy(mask: np.ndarray) -> tuple[float, float]:
    m = np.asarray(mask, dtype=bool)
    if int(m.sum()) == 0:
        return 0.0, 0.0
    yy, xx = np.nonzero(m)
    return float(xx.mean()), float(yy.mean())


def save_overlay_qc(*, fixed01: np.ndarray, warped01: np.ndarray, out_png: Path, title: str) -> None:
    overlay = np.stack([fixed01, warped01, fixed01], axis=-1)
    cb = checkerboard(fixed01, warped01, tile_px=int(CHECKER_TILE_PX))
    diff = np.abs(fixed01 - warped01)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(fixed01, cmap="gray")
    axes[0].set_title("Fixed")
    axes[1].imshow(warped01, cmap="gray")
    axes[1].set_title("Warped moving")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (magenta=fixed, green=moving)")
    axes[3].imshow(cb, cmap="gray")
    axes[3].set_title("Checkerboard")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    fig2, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(diff, cmap="magma")
    ax.set_title("abs(fixed - warped)")
    ax.axis("off")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(out_png.with_name(out_png.stem + "_diff.png"), dpi=220)
    plt.close(fig2)


def save_side_by_side_qc(
    *,
    fixed01: np.ndarray,
    moving01: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(fixed01, cmap="gray")
    axes[0].set_title(f"Fixed ({fixed01.shape[0]}x{fixed01.shape[1]})")
    axes[1].imshow(moving01, cmap="gray")
    axes[1].set_title(f"Moving ({moving01.shape[0]}x{moving01.shape[1]})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_edge_qc(*, fixed01: np.ndarray, warped01: np.ndarray, out_png: Path, title: str) -> None:
    fixed_edge = gradmag01(fixed01, sigma_px=float(EDGE_SIGMA_PX))
    warped_edge = gradmag01(warped01, sigma_px=float(EDGE_SIGMA_PX))
    save_overlay_qc(
        fixed01=fixed_edge,
        warped01=warped_edge,
        out_png=out_png,
        title=title,
    )


def save_mask_overlay(*, img01: np.ndarray, mask: np.ndarray, out_png: Path, title: str) -> None:
    img = np.asarray(img01, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if img.shape != m.shape:
        raise ValueError(f"Mask overlay shape mismatch: img={img.shape}, mask={m.shape}.")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(m.astype(np.float32), cmap="Reds", alpha=0.30, vmin=0.0, vmax=1.0)
    ax.contour(m.astype(np.uint8), levels=[0.5], colors=["cyan"], linewidths=0.8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_ants_diagnostic_metric_values(path: Path) -> tuple[list[int], list[float]]:
    iters: list[int] = []
    metric_values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "DIAGNOSTIC" not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or parts[1] in {"Iteration", ""}:
                continue
            try:
                it = int(parts[1])
                mv = float(parts[2])
            except ValueError:
                continue
            iters.append(it)
            metric_values.append(mv)
    return iters, metric_values


@contextmanager
def redirect_fds(*, stdout_path: Path, stderr_path: Path | None = None) -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.open(str(stdout_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    if stderr_path is None:
        stderr_fd = stdout_fd
    else:
        stderr_fd = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        if stderr_path is not None:
            os.close(stderr_fd)
        os.close(stdout_fd)


def gradmag01(img01: np.ndarray, *, sigma_px: float) -> np.ndarray:
    sm = ndi.gaussian_filter(img01, sigma=float(sigma_px))
    gx = ndi.sobel(sm, axis=1)
    gy = ndi.sobel(sm, axis=0)
    g = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return robust_normalize01(g, q_lo=0.01, q_hi=0.999)


@dataclass(frozen=True)
class RunResult:
    name: str
    outprefix: str
    metric_ncc: float
    metric_mi: float
    fwdtransforms: list[str]
    invtransforms: list[str]


# %% [markdown]
# ## Phase 0: Load images (first channel)

# %%
print(f"FIXED_PATH={FIXED_PATH}")
print(f"MOVING_PATH={MOVING_PATH}")
print(f"OUTDIR={OUTDIR.resolve()}")

if not FIXED_PATH.exists():
    raise FileNotFoundError(f"FIXED_PATH does not exist: {FIXED_PATH}")
if not MOVING_PATH.exists():
    raise FileNotFoundError(f"MOVING_PATH does not exist: {MOVING_PATH}")

fixed_raw = load_first_channel(FIXED_PATH, channel=int(CHANNEL))
moving_raw = load_first_channel(MOVING_PATH, channel=int(CHANNEL))

print(f"fixed_raw: shape={fixed_raw.shape}, dtype={fixed_raw.dtype}, min/max={fixed_raw.min():.3g}/{fixed_raw.max():.3g}")
print(f"moving_raw: shape={moving_raw.shape}, dtype={moving_raw.dtype}, min/max={moving_raw.min():.3g}/{moving_raw.max():.3g}")

np.save(OUTDIR / "fixed_ch0_raw.npy", fixed_raw)
np.save(OUTDIR / "moving_ch0_raw.npy", moving_raw)

fixed01 = robust_normalize01(fixed_raw)
moving01 = robust_normalize01(moving_raw)
iio.imwrite(OUTDIR / "fixed_ch0_norm.png", (fixed01 * 255).astype(np.uint8))
iio.imwrite(OUTDIR / "moving_ch0_norm.png", (moving01 * 255).astype(np.uint8))

if fixed01.shape == moving01.shape:
    save_edge_qc(
        fixed01=fixed01,
        warped01=moving01,
        out_png=OUTDIR / "edges_before_anything_qc.png",
        title=f"Edges (before) | sigma={EDGE_SIGMA_PX}px",
    )
else:
    save_side_by_side_qc(
        fixed01=gradmag01(fixed01, sigma_px=float(EDGE_SIGMA_PX)),
        moving01=gradmag01(moving01, sigma_px=float(EDGE_SIGMA_PX)),
        out_png=OUTDIR / "edges_before_anything_side_by_side.png",
        title=f"Edges (before; no overlay due to shape mismatch) | sigma={EDGE_SIGMA_PX}px",
    )

# %% [markdown]
# ## Phase 1: Masks (from fullmask.tif), to stabilize SyN on black backgrounds

# %%
fixed_roi = infer_roi_from_ccf_path(FIXED_PATH)
moving_roi = infer_roi_from_ccf_path(MOVING_PATH)
fixed_mask = load_mask_from_fullmask_tif(image_path=FIXED_PATH, shape_yx=fixed01.shape)
moving_mask = load_mask_from_fullmask_tif(image_path=MOVING_PATH, shape_yx=moving01.shape)

print(f"fixed_mask coverage={fixed_mask.mean():.3f}")
print(f"moving_mask coverage={moving_mask.mean():.3f}")

iio.imwrite(OUTDIR / "fixed_mask.png", (fixed_mask * 255).astype(np.uint8))
iio.imwrite(OUTDIR / "moving_mask.png", (moving_mask * 255).astype(np.uint8))

save_mask_overlay(
    img01=fixed01,
    mask=fixed_mask,
    out_png=OUTDIR / "fixed_mask_overlay.png",
    title=f"Fixed mask overlay | {FULLMASK_FILENAME} | roi={fixed_roi}",
)
save_mask_overlay(
    img01=moving01,
    mask=moving_mask,
    out_png=OUTDIR / "moving_mask_overlay.png",
    title=f"Moving mask overlay | {FULLMASK_FILENAME} | roi={moving_roi}",
)

# %% [markdown]
# ## Phase 2: Build ANTs images (optional downsample for faster iteration)

# %%
ds = int(DOWNSAMPLE_FOR_REG)
fixed01_ds = downsample(fixed01, factor=ds, order=1)
moving01_ds = downsample(moving01, factor=ds, order=1)
fixed_mask_ds = downsample(fixed_mask.astype(np.uint8), factor=ds, order=0) > 0
moving_mask_ds = downsample(moving_mask.astype(np.uint8), factor=ds, order=0) > 0

spacing_full = (1.0, 1.0)
spacing_ds = (float(ds), float(ds))

fixed_ants_full = ants.from_numpy(fixed01.astype(np.float32), spacing=spacing_full)
moving_ants_full = ants.from_numpy(moving01.astype(np.float32), spacing=spacing_full)
fixed_mask_ants_full = ants.from_numpy(fixed_mask.astype(np.uint8), spacing=spacing_full)
moving_mask_ants_full = ants.from_numpy(moving_mask.astype(np.uint8), spacing=spacing_full)

fixed_ants = ants.from_numpy(fixed01_ds.astype(np.float32), spacing=spacing_ds)
moving_ants = ants.from_numpy(moving01_ds.astype(np.float32), spacing=spacing_ds)
fixed_mask_ants = ants.from_numpy(fixed_mask_ds.astype(np.uint8), spacing=spacing_ds)
moving_mask_ants = ants.from_numpy(moving_mask_ds.astype(np.uint8), spacing=spacing_ds)

print(f"Downsample factor={ds} -> fixed reg shape={fixed01_ds.shape}, moving reg shape={moving01_ds.shape}")

# %% [markdown]
# ## Phase 3: Affine initialization (once), then SyN sweep

# %%
aff_prefix = str(OUTDIR / "affine_init_")
aff_stdout = OUTDIR / "affine_init_stdout.txt"

aff_initial: str | None = None
if bool(USE_AFFINE_INITIALIZER):
    tx_init = ants.affine_initializer(
        fixed_image=fixed_ants,
        moving_image=moving_ants,
        search_factor=int(AFFINE_INIT_SEARCH_FACTOR),
        radian_fraction=float(AFFINE_INIT_RADIAN_FRACTION),
        use_principal_axis=bool(AFFINE_INIT_USE_PRINCIPAL_AXIS),
        local_search_iterations=int(AFFINE_INIT_LOCAL_SEARCH_ITERS),
        mask=fixed_mask_ants,
    )
    aff_initial = str(OUTDIR / "affine_initializer.tfm")
    if isinstance(tx_init, str):
        tx_obj = ants.read_transform(tx_init)
        ants.write_transform(tx_obj, aff_initial)
    else:
        ants.write_transform(tx_init, aff_initial)
    (OUTDIR / "affine_initializer.json").write_text(
        json.dumps(
            {
                "use": True,
                "search_factor": int(AFFINE_INIT_SEARCH_FACTOR),
                "radian_fraction": float(AFFINE_INIT_RADIAN_FRACTION),
                "use_principal_axis": bool(AFFINE_INIT_USE_PRINCIPAL_AXIS),
                "local_search_iterations": int(AFFINE_INIT_LOCAL_SEARCH_ITERS),
                "transform_path": aff_initial,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
elif bool(USE_COM_TRANSLATION_INIT):
    fx, fy = center_of_mass_xy(fixed_mask_ds)
    mx, my = center_of_mass_xy(moving_mask_ds)
    dx = fx - mx
    dy = fy - my
    tx = ants.create_ants_transform(
        transform_type="Euler2DTransform",
        dimension=2,
        center=(0.0, 0.0),
        translation=(float(dx * spacing_ds[0]), float(dy * spacing_ds[1])),
    )
    aff_initial = str(OUTDIR / "com_init.tfm")
    ants.write_transform(tx, aff_initial)
    (OUTDIR / "com_init.json").write_text(
        json.dumps(
            {
                "use": True,
                "fixed_com_xy_ds": [fx, fy],
                "moving_com_xy_ds": [mx, my],
                "delta_xy_ds": [dx, dy],
                "translation_physical": [float(dx * spacing_ds[0]), float(dy * spacing_ds[1])],
                "spacing_ds": list(spacing_ds),
                "transform_path": aff_initial,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

with redirect_fds(stdout_path=aff_stdout):
    tx_aff = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="Affine",
        initial_transform=aff_initial if aff_initial is not None else "Identity",
        mask=fixed_mask_ants,
        moving_mask=moving_mask_ants,
        mask_all_stages=True,
        random_seed=int(RANDOM_SEED),
        write_composite_transform=True,
        verbose=True,
        outprefix=aff_prefix,
    )

print(f"Wrote: {aff_stdout}")


def _as_transform_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [v]
    raise TypeError(f"Unexpected transform spec type: {type(v)}")


def export_output_bundle(
    *,
    outdir: Path,
    export_dir: Path,
    basename: str,
    fixed_path: Path,
    moving_path: Path,
    channel: int,
    run: "RunResult",
    run_row: dict[str, Any],
    run_spec: dict[str, Any],
    affine_fwd: list[str],
    affine_inv: list[str],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    def cp(src: Path, dst: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"Missing expected artifact: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    transforms_dir = export_dir / "transforms"
    warped_dir = export_dir / "warped"
    qc_dir = export_dir / "qc"

    # Transforms (forward = moving→fixed; inverse = fixed→moving)
    cp(outdir / f"{run.name}_Composite.h5", transforms_dir / "syn_fwd_Composite.h5")
    cp(outdir / f"{run.name}_InverseComposite.h5", transforms_dir / "syn_inv_Composite.h5")
    cp(outdir / "affine_init_Composite.h5", transforms_dir / "affine_fwd_Composite.h5")
    cp(outdir / "affine_init_InverseComposite.h5", transforms_dir / "affine_inv_Composite.h5")

    (transforms_dir / "syn_fwd_transformlist.txt").write_text(
        "\n".join(run.fwdtransforms) + "\n", encoding="utf-8"
    )
    (transforms_dir / "syn_inv_transformlist.txt").write_text(
        "\n".join(run.invtransforms) + "\n", encoding="utf-8"
    )
    (transforms_dir / "affine_fwd_transformlist.txt").write_text(
        "\n".join(affine_fwd) + "\n", encoding="utf-8"
    )
    (transforms_dir / "affine_inv_transformlist.txt").write_text(
        "\n".join(affine_inv) + "\n", encoding="utf-8"
    )

    # Warped outputs (moving→fixed) in fixed grid
    cp(outdir / f"{run.name}_warped.nii.gz", warped_dir / "moving_to_fixed.nii.gz")
    cp(outdir / f"{run.name}_warped.png", warped_dir / "moving_to_fixed.png")
    cp(outdir / f"{run.name}_warped_mask.png", warped_dir / "moving_to_fixed_mask.png")

    # QC
    for suffix in [
        "qc.png",
        "qc_diff.png",
        "edges_qc.png",
        "edges_qc_diff.png",
        "loss.png",
        "stdout.txt",
        "summary.json",
    ]:
        p = outdir / f"{run.name}_{suffix}"
        if p.exists():
            cp(p, qc_dir / suffix)

    # Reproducibility helpers
    for p in [
        outdir / "run_config.json",
        outdir / "roi_morph.json",
        outdir / ROISET_MORPHED_FILENAME,
        outdir / "fixed_ch0_norm.png",
        outdir / "moving_ch0_norm.png",
        outdir / "fixed_mask.png",
        outdir / "moving_mask.png",
        outdir / "fixed_mask_overlay.png",
        outdir / "moving_mask_overlay.png",
        outdir / "affine_init_qc.png",
        outdir / "affine_init_edges_qc.png",
        outdir / "affine_init_summary.json",
        outdir / "run.json",
    ]:
        if p.exists():
            cp(p, qc_dir / p.name)

    manifest = {
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "channel": int(channel),
        "ants_version": getattr(ants, "__version__", None),
        "random_seed": int(RANDOM_SEED),
        "downsample_for_reg": int(DOWNSAMPLE_FOR_REG),
        "run_config_json": str((export_dir / "qc" / "run_config.json").resolve()),
        "roi_morph_json": str((export_dir / "qc" / "roi_morph.json").resolve()),
        "roiset_morphed_zip": str((export_dir / "qc" / ROISET_MORPHED_FILENAME).resolve()),
        "mask": {
            "fixed_fullmask_path": str(Path(fixed_path).with_name(str(FULLMASK_FILENAME))),
            "moving_fullmask_path": str(Path(moving_path).with_name(str(FULLMASK_FILENAME))),
            "fullmask_filename": str(FULLMASK_FILENAME),
        },
        "run": {
            "name": run.name,
            "spec": run_spec,
            "metrics": {
                "ncc": float(run.metric_ncc),
                "mi": float(run.metric_mi),
                "edge_ncc": float(run_row["edge_ncc"]),
                "overlap_frac": float(run_row["overlap_frac"]),
            },
            "fwdtransforms": run.fwdtransforms,
            "invtransforms": run.invtransforms,
        },
        "export": {
            "basename": str(basename),
            "bundle_dir": str(export_dir.resolve()),
        },
        "outputs": {
            "transforms/syn_fwd_Composite.h5": "moving→fixed composite transform",
            "transforms/syn_inv_Composite.h5": "fixed→moving composite transform",
            "transforms/affine_fwd_Composite.h5": "moving→fixed affine-only composite",
            "transforms/affine_inv_Composite.h5": "fixed→moving affine-only composite",
            "warped/moving_to_fixed.png": "moving warped into fixed grid (png)",
            "warped/moving_to_fixed.nii.gz": "moving warped into fixed grid (nii.gz)",
            "warped/moving_to_fixed_mask.png": "moving mask warped into fixed grid (png)",
            "qc/qc.png": "QC overlay",
            "qc/edges_qc.png": "QC edge overlay",
        },
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


aff_fwd = _as_transform_list(tx_aff["fwdtransforms"])
aff_inv = _as_transform_list(tx_aff["invtransforms"])

warped_aff = ants.apply_transforms(
    fixed=fixed_ants_full,
    moving=moving_ants_full,
    transformlist=aff_fwd,
    interpolator="linear",
)
warped_aff_mask = ants.apply_transforms(
    fixed=fixed_mask_ants_full,
    moving=moving_mask_ants_full,
    transformlist=aff_fwd,
    interpolator="nearestNeighbor",
)

warped_aff01 = np.asarray(warped_aff.numpy()).astype(np.float32)
warped_aff01n = robust_normalize01(warped_aff01)
warped_aff_mask_np = np.asarray(warped_aff_mask.numpy()) > 0
overlap_aff = fixed_mask & warped_aff_mask_np
ncc_aff = ncc(fixed01, warped_aff01, mask=overlap_aff)
mi_aff = masked_mi_hist2d(fixed01, warped_aff01, mask=overlap_aff, bins=int(MI_BINS))
edge_ncc_aff = ncc(
    gradmag01(fixed01, sigma_px=float(EDGE_SIGMA_PX)),
    gradmag01(warped_aff01n, sigma_px=float(EDGE_SIGMA_PX)),
    mask=overlap_aff,
)
save_overlay_qc(
    fixed01=fixed01,
    warped01=warped_aff01n,
    out_png=OUTDIR / "affine_init_qc.png",
    title=(
        f"Affine init | NCC={ncc_aff:.4f} | MI={mi_aff:.4f} | "
        f"EdgeNCC={edge_ncc_aff:.4f} | overlap={overlap_aff.mean():.3f}"
    ),
)
save_edge_qc(
    fixed01=fixed01,
    warped01=warped_aff01n,
    out_png=OUTDIR / "affine_init_edges_qc.png",
    title=f"Affine init edges | EdgeNCC={edge_ncc_aff:.4f} | sigma={EDGE_SIGMA_PX}px",
)

ants.image_write(warped_aff, str(OUTDIR / "affine_init_warped.nii.gz"))
iio.imwrite(OUTDIR / "affine_init_warped.png", (warped_aff01n * 255).astype(np.uint8))

aff_summary = {
    "type": "Affine",
    "ncc": float(ncc_aff),
    "mi": float(mi_aff),
    "edge_ncc": float(edge_ncc_aff),
    "overlap_frac": float(overlap_aff.mean()),
    "fwdtransforms": aff_fwd,
    "invtransforms": aff_inv,
    "stdout": str(aff_stdout),
}
(OUTDIR / "affine_init_summary.json").write_text(json.dumps(aff_summary, indent=2))


def run_syn(spec: dict[str, Any]) -> RunResult:
    name = str(spec["name"])
    prefix = str(OUTDIR / f"{name}_")
    stdout_path = OUTDIR / f"{name}_stdout.txt"
    loss_png = OUTDIR / f"{name}_loss.png"

    with redirect_fds(stdout_path=stdout_path):
        tx = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform=str(spec["type_of_transform"]),
            initial_transform=aff_fwd,
            grad_step=float(spec["grad_step"]),
            flow_sigma=float(spec["flow_sigma"]),
            total_sigma=float(spec["total_sigma"]),
            syn_metric=str(spec["syn_metric"]),
            syn_sampling=int(spec["syn_sampling"]),
            reg_iterations=tuple(int(x) for x in spec["reg_iterations"]),
            mask=fixed_mask_ants,
            moving_mask=moving_mask_ants,
            mask_all_stages=True,
            random_seed=int(RANDOM_SEED),
            write_composite_transform=True,
            verbose=True,
            outprefix=prefix,
        )

    iters, metric_values = parse_ants_diagnostic_metric_values(stdout_path)
    if metric_values:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.asarray(iters, dtype=np.int32), np.asarray(metric_values, dtype=np.float64), lw=1)
        ax.set_xlabel("Iteration (global; all levels)")
        ax.set_ylabel("Metric value")
        ax.set_title(f"{name} diagnostics")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(loss_png, dpi=180)
        plt.close(fig)

    warped = ants.apply_transforms(
        fixed=fixed_ants_full,
        moving=moving_ants_full,
        transformlist=tx["fwdtransforms"],
        interpolator="linear",
    )
    warped_mask = ants.apply_transforms(
        fixed=fixed_mask_ants_full,
        moving=moving_mask_ants_full,
        transformlist=tx["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    warped_np = np.asarray(warped.numpy()).astype(np.float32)
    warped01n = robust_normalize01(warped_np)
    warped_mask_np = np.asarray(warped_mask.numpy()) > 0
    overlap = fixed_mask & warped_mask_np
    metric_ncc = ncc(fixed01, warped_np, mask=overlap)
    metric_mi = masked_mi_hist2d(fixed01, warped_np, mask=overlap, bins=int(MI_BINS))
    metric_edge_ncc = ncc(
        gradmag01(fixed01, sigma_px=float(EDGE_SIGMA_PX)),
        gradmag01(warped01n, sigma_px=float(EDGE_SIGMA_PX)),
        mask=overlap,
    )

    ants.image_write(warped, str(OUTDIR / f"{name}_warped.nii.gz"))
    iio.imwrite(OUTDIR / f"{name}_warped.png", (warped01n * 255).astype(np.uint8))
    iio.imwrite(OUTDIR / f"{name}_warped_mask.png", (warped_mask_np * 255).astype(np.uint8))

    save_overlay_qc(
        fixed01=fixed01,
        warped01=warped01n,
        out_png=OUTDIR / f"{name}_qc.png",
        title=(
            f"{name} | NCC={metric_ncc:.4f} | MI={metric_mi:.4f} | "
            f"EdgeNCC={metric_edge_ncc:.4f} | overlap={overlap.mean():.3f}"
        ),
    )
    save_edge_qc(
        fixed01=fixed01,
        warped01=warped01n,
        out_png=OUTDIR / f"{name}_edges_qc.png",
        title=f"{name} edges | EdgeNCC={metric_edge_ncc:.4f} | sigma={EDGE_SIGMA_PX}px",
    )

    summary = {
        "spec": spec,
        "ncc": metric_ncc,
        "mi": metric_mi,
        "edge_ncc": metric_edge_ncc,
        "overlap_frac": float(overlap.mean()),
        "fwdtransforms": _as_transform_list(tx["fwdtransforms"]),
        "invtransforms": _as_transform_list(tx["invtransforms"]),
        "stdout": str(stdout_path),
    }
    (OUTDIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"{name}: NCC={metric_ncc:.4f}, MI={metric_mi:.4f}, EdgeNCC={metric_edge_ncc:.4f}, "
        f"overlap={overlap.mean():.3f} (stdout={stdout_path.name})"
    )

    return RunResult(
        name=name,
        outprefix=prefix,
        metric_ncc=float(metric_ncc),
        metric_mi=float(metric_mi),
        fwdtransforms=_as_transform_list(tx["fwdtransforms"]),
        invtransforms=_as_transform_list(tx["invtransforms"]),
    )


run = run_syn(SYN_SPEC)
run_summary = json.loads((OUTDIR / f"{run.name}_summary.json").read_text(encoding="utf-8"))

roiset_in_path = Path(os.environ.get("ANTS_SYN_ROISET_IN") or FIXED_PATH.with_name(ROISET_IN_FILENAME))
roiset_out_path = Path(os.environ.get("ANTS_SYN_ROISET_OUT") or MOVING_PATH.with_name(ROISET_MORPHED_FILENAME))

rois = roiread(roiset_in_path)
if isinstance(rois, ImagejRoi):
    rois = [rois]

roi_entries: list[dict[str, Any]] = []
morphed_rois: list[ImagejRoi] = []
for roi in rois:
    if roi.roitype not in {ROI_TYPE.POLYGON, ROI_TYPE.TRACED, ROI_TYPE.FREEHAND}:
        raise ValueError(
            f"Expected area ROI (polygon/traced/freehand), got roitype={roi.roitype} for name={roi.name!r}."
        )

    coords_abs = None
    coords_fn = getattr(roi, "coordinates", None)
    if callable(coords_fn):
        coords_abs = coords_fn()
    if coords_abs is None:
        coords_raw = roi.integer_coordinates
        if coords_raw is None:
            coords_raw = roi.subpixel_coordinates
        if coords_raw is None:
            raise ValueError(f"ROI {roi.name!r} has no coordinates.")
        offset = np.array([float(roi.left), float(roi.top)], dtype=np.float64)
        coords_abs = np.asarray(coords_raw, dtype=np.float64) + offset
    else:
        coords_abs = np.asarray(coords_abs, dtype=np.float64)

    if coords_abs.ndim != 2 or coords_abs.shape[1] != 2 or coords_abs.shape[0] < 3:
        raise ValueError(f"Invalid polygon coords for roi={roi.name!r}: shape={coords_abs.shape}.")

    df = pd.DataFrame(
        {
            "x": coords_abs[:, 0],
            "y": coords_abs[:, 1],
            "z": 0.0,
            "t": 0.0,
        }
    )
    out = ants.apply_transforms_to_points(dim=2, points=df, transformlist=run.invtransforms)
    x_out = out["x"].to_numpy(dtype=np.float64, copy=False)
    y_out = out["y"].to_numpy(dtype=np.float64, copy=False)
    pts_out = list(zip(x_out.tolist(), y_out.tolist(), strict=True))

    in_name = str(roi.name).strip() if roi.name else "roi"
    out_name = f"{in_name}{ROI_MORPHED_SUFFIX}"
    morphed_rois.append(ImagejRoi.frompoints(pts_out, name=out_name))
    roi_entries.append({"in_name": in_name, "out_name": out_name, "n_points": int(coords_abs.shape[0])})

roiwrite(roiset_out_path, morphed_rois, mode="w")
morphed_zip_copy = OUTDIR / ROISET_MORPHED_FILENAME
shutil.copy2(roiset_out_path, morphed_zip_copy)

roi_morph = {
    "direction": "fixed_to_moving",
    "roiset_in_path": str(roiset_in_path),
    "roiset_out_path": str(roiset_out_path),
    "roiset_out_copy_path": str(morphed_zip_copy),
    "suffix": str(ROI_MORPHED_SUFFIX),
    "count": int(len(morphed_rois)),
    "entries": roi_entries,
    "transformlist": list(run.invtransforms),
}
(OUTDIR / "roi_morph.json").write_text(json.dumps(roi_morph, indent=2), encoding="utf-8")

(OUTDIR / "run.json").write_text(
    json.dumps(
        {
            "reference": {
                "role": "fixed",
                "fixed_path": str(FIXED_PATH),
                "moving_path": str(MOVING_PATH),
                "fixed_roi": fixed_roi,
                "moving_roi": moving_roi,
            },
            "name": run.name,
            "ncc": run.metric_ncc,
            "mi": run.metric_mi,
            "edge_ncc": float(run_summary.get("edge_ncc", float("nan"))),
            "overlap_frac": float(run_summary.get("overlap_frac", float("nan"))),
            "fwdtransforms": run.fwdtransforms,
            "invtransforms": run.invtransforms,
            "spec": SYN_SPEC,
            "run_config_path": str(RUN_CONFIG_PATH),
            "roi_morph_path": str(OUTDIR / "roi_morph.json"),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(
    f"Run: {run.name} (EdgeNCC={float(run_summary.get('edge_ncc', float('nan'))):.4f}, "
    f"NCC={run.metric_ncc:.4f}, MI={run.metric_mi:.4f})"
)

# %% [markdown]
# ## Phase 5: Export output bundle (stable filenames)
#
# Writes a small, stable “output format” under `EXPORT_DIR/`:
# - `transforms/` (Composite.h5 + transform lists)
# - `warped/` (moving→fixed)
# - `qc/` (QC images + JSONs)
# - `manifest.json` (metadata and parameters)

# %%
run_spec = run_summary.get("spec", {})

export_basename = str(EXPORT_BASENAME) if EXPORT_BASENAME is not None else infer_export_basename(
    fixed_path=FIXED_PATH,
    moving_path=MOVING_PATH,
    channel=int(CHANNEL),
)
export_dir = EXPORT_ROOT / export_basename

export_output_bundle(
    outdir=OUTDIR,
    export_dir=export_dir,
    basename=export_basename,
    fixed_path=FIXED_PATH,
    moving_path=MOVING_PATH,
    channel=int(CHANNEL),
    run=run,
    run_row={
        "edge_ncc": float(run_summary.get("edge_ncc", float("nan"))),
        "overlap_frac": float(run_summary.get("overlap_frac", float("nan"))),
    },
    run_spec=run_spec if isinstance(run_spec, dict) else {"spec": run_spec},
    affine_fwd=aff_fwd,
    affine_inv=aff_inv,
)
print(f"Exported output bundle to: {export_dir.resolve()}")

# %%
