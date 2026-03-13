# %%
# You need itk-elastix installed to run this.
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import tifffile
from scipy.linalg import expm, logm
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation

# %%

_RUN_T0 = 0.0


def _log(msg: str) -> None:
    dt = time.perf_counter() - _RUN_T0
    print(f"[{dt:9.3f}s] {msg}", flush=True)


@contextmanager
def _log_step(name: str):
    t0 = time.perf_counter()
    _log(f"{name} | start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _log(f"{name} | done ({dt:.3f}s)")


def _patch_centered(img: np.ndarray, *, y: int, x: int, radius: int) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be > 0")
    y0 = y - radius
    x0 = x - radius
    size = radius * 2 + 1
    return _block_from(img, y0=y0, x0=x0, size=size)


def _fill_nans_nearest_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    if not np.isnan(x).any():
        return x
    valid = np.isfinite(x)
    if not np.any(valid):
        raise ValueError("All values are NaN/inf; cannot fill.")
    _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
    filled = x.copy()
    filled[~valid] = filled[iy[~valid], ix[~valid]]
    return filled


def _masked_gaussian_smooth_2d(
    x: np.ndarray,
    w: np.ndarray,
    *,
    sigma: float | tuple[float, float],
    mode: str = "nearest",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalized convolution: gaussian(x*w) / gaussian(w).

    Missing values should have weight 0. This is useful for smoothing/inpainting sparse
    displacement lattices without a nearest-neighbor "plateau" prior.
    """
    if x.shape != w.shape:
        raise ValueError("x and w must have same shape")
    if isinstance(sigma, tuple):
        if len(sigma) != 2:
            raise ValueError("sigma tuple must have length 2")
        if float(sigma[0]) < 0.0 or float(sigma[1]) < 0.0:
            raise ValueError("sigma must be >= 0")
        sigma = (float(sigma[0]), float(sigma[1]))
    else:
        if float(sigma) < 0.0:
            raise ValueError("sigma must be >= 0")
        sigma = float(sigma)
    if float(eps) <= 0.0:
        raise ValueError("eps must be > 0")

    x0 = np.where(np.isfinite(x), x, 0.0)
    w0 = np.where(np.isfinite(x), w, 0.0)
    num = ndi.gaussian_filter(x0 * w0, sigma=sigma, mode=str(mode))
    den = ndi.gaussian_filter(w0, sigma=sigma, mode=str(mode))
    return num / np.maximum(den, float(eps))


def build_bspline_displacement_field(
    *,
    height: int,
    width: int,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    dx0: np.ndarray,
    dy0: np.ndarray,
    control_grid_spacing_x_px: int,
    control_grid_spacing_y_px: int,
    weights0: np.ndarray | None = None,
    smooth_sigma_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if xgrid.shape != ygrid.shape or xgrid.shape != dx0.shape or xgrid.shape != dy0.shape:
        raise ValueError("xgrid/ygrid/dx0/dy0 must have the same shape.")
    h = int(height)
    w = int(width)
    spacing_x = int(control_grid_spacing_x_px)
    spacing_y = int(control_grid_spacing_y_px)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be > 0")
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError("control_grid_spacing_x_px/control_grid_spacing_y_px must be > 0")
    if smooth_sigma_px < 0:
        raise ValueError("smooth_sigma_px must be >= 0")

    if weights0 is None:
        w0 = np.where(np.isfinite(dx0) & np.isfinite(dy0), 1.0, 0.0).astype(np.float64, copy=False)
    else:
        w0 = np.asarray(weights0, dtype=np.float64)
        if w0.shape != xgrid.shape:
            raise ValueError(f"weights0 must have same shape as xgrid, got {w0.shape}")
        if np.any(w0 < 0):
            raise ValueError("weights0 must be nonnegative")
        w0 = np.where(np.isfinite(dx0) & np.isfinite(dy0), w0, 0.0).astype(np.float64, copy=False)

    x_min = float(np.min(xgrid))
    y_min = float(np.min(ygrid))
    x_max = float(np.max(xgrid))
    y_max = float(np.max(ygrid))

    # Extend the lattice so the output domain [0..w-1]x[0..h-1] is covered; fill any
    # missing control points by nearest-neighbor before B-spline interpolation.
    # We also add a few extra lattice steps of padding so B-spline interpolation near the
    # image edges doesn't depend on out-of-bounds lattice samples (which would otherwise
    # bias towards the default pixel value).
    pad_steps = 4
    kx = int(np.ceil(max(0.0, x_min) / float(spacing_x)))
    ky = int(np.ceil(max(0.0, y_min) / float(spacing_y)))
    origin_x = x_min - float((kx + pad_steps) * spacing_x)
    origin_y = y_min - float((ky + pad_steps) * spacing_y)
    size_x = int(np.ceil((float(w - 1) - origin_x) / float(spacing_x))) + 1 + pad_steps
    size_y = int(np.ceil((float(h - 1) - origin_y) / float(spacing_y))) + 1 + pad_steps
    if size_x <= 1 or size_y <= 1:
        raise ValueError(f"Invalid lattice size: {(size_y, size_x)} for image {(h, w)}")
    if x_max < origin_x or y_max < origin_y:
        raise ValueError("Control grid lies outside lattice origin (unexpected).")

    dx_lat = np.full((size_y, size_x), np.nan, dtype=np.float64)
    dy_lat = np.full((size_y, size_x), np.nan, dtype=np.float64)
    w_lat = np.zeros((size_y, size_x), dtype=np.float64)
    for x, y, dx, dy, ww in zip(xgrid.tolist(), ygrid.tolist(), dx0.tolist(), dy0.tolist(), w0.tolist(), strict=True):
        ix = int(round((float(x) - origin_x) / float(spacing_x)))
        iy = int(round((float(y) - origin_y) / float(spacing_y)))
        if ix < 0 or iy < 0 or ix >= size_x or iy >= size_y:
            continue
        dx_lat[iy, ix] = float(dx)
        dy_lat[iy, ix] = float(dy)
        w_lat[iy, ix] = float(ww)

    if float(smooth_sigma_px) > 0.0:
        # Note: smoothing is applied on the coarse displacement lattice, so the *effective*
        # sigma in lattice steps is (smooth_sigma_px / control_grid_spacing_{x,y}_px).
        sigma_y = float(smooth_sigma_px) / float(spacing_y)
        sigma_x = float(smooth_sigma_px) / float(spacing_x)
        dx_lat = _masked_gaussian_smooth_2d(dx_lat, w_lat, sigma=(sigma_y, sigma_x), mode="nearest")
        dy_lat = _masked_gaussian_smooth_2d(dy_lat, w_lat, sigma=(sigma_y, sigma_x), mode="nearest")
    else:
        dx_lat = _fill_nans_nearest_2d(dx_lat)
        dy_lat = _fill_nans_nearest_2d(dy_lat)

    field_lat = np.stack([dx_lat, dy_lat], axis=-1).astype(np.float32, copy=False)
    field_img = sitk.GetImageFromArray(field_lat, isVector=True)
    field_img.SetOrigin((float(origin_x), float(origin_y)))
    field_img.SetSpacing((float(spacing_x), float(spacing_y)))

    ref_img = sitk.Image(int(w), int(h), sitk.sitkFloat32)
    ref_img.SetOrigin((0.0, 0.0))
    ref_img.SetSpacing((1.0, 1.0))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
    dense_img = resampler.Execute(field_img)
    dense = sitk.GetArrayFromImage(dense_img)
    if dense.shape != (h, w, 2):
        raise ValueError(f"Unexpected dense field shape: {dense.shape}, expected {(h, w, 2)}")
    dx_dense = dense[..., 0].astype(np.float32, copy=False)
    dy_dense = dense[..., 1].astype(np.float32, copy=False)
    return dx_dense, dy_dense


def build_bspline_scalar_field(
    *,
    height: int,
    width: int,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    v0: np.ndarray,
    control_grid_spacing_x_px: int,
    control_grid_spacing_y_px: int,
    weights0: np.ndarray | None = None,
    smooth_sigma_px: float = 0.0,
) -> np.ndarray:
    if xgrid.shape != ygrid.shape or xgrid.shape != v0.shape:
        raise ValueError("xgrid/ygrid/v0 must have the same shape.")
    h = int(height)
    w = int(width)
    spacing_x = int(control_grid_spacing_x_px)
    spacing_y = int(control_grid_spacing_y_px)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be > 0")
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError("control_grid_spacing_x_px/control_grid_spacing_y_px must be > 0")
    if smooth_sigma_px < 0:
        raise ValueError("smooth_sigma_px must be >= 0")

    if weights0 is None:
        w0 = np.where(np.isfinite(v0), 1.0, 0.0).astype(np.float64, copy=False)
    else:
        w0 = np.asarray(weights0, dtype=np.float64)
        if w0.shape != xgrid.shape:
            raise ValueError(f"weights0 must have same shape as xgrid, got {w0.shape}")
        if np.any(w0 < 0):
            raise ValueError("weights0 must be nonnegative")
        w0 = np.where(np.isfinite(v0), w0, 0.0).astype(np.float64, copy=False)

    x_min = float(np.min(xgrid))
    y_min = float(np.min(ygrid))
    x_max = float(np.max(xgrid))
    y_max = float(np.max(ygrid))

    pad_steps = 4
    kx = int(np.ceil(max(0.0, x_min) / float(spacing_x)))
    ky = int(np.ceil(max(0.0, y_min) / float(spacing_y)))
    origin_x = x_min - float((kx + pad_steps) * spacing_x)
    origin_y = y_min - float((ky + pad_steps) * spacing_y)
    size_x = int(np.ceil((float(w - 1) - origin_x) / float(spacing_x))) + 1 + pad_steps
    size_y = int(np.ceil((float(h - 1) - origin_y) / float(spacing_y))) + 1 + pad_steps
    if size_x <= 1 or size_y <= 1:
        raise ValueError(f"Invalid lattice size: {(size_y, size_x)} for image {(h, w)}")
    if x_max < origin_x or y_max < origin_y:
        raise ValueError("Control grid lies outside lattice origin (unexpected).")

    v_lat = np.full((size_y, size_x), np.nan, dtype=np.float64)
    w_lat = np.zeros((size_y, size_x), dtype=np.float64)
    for x, y, vv, ww in zip(xgrid.tolist(), ygrid.tolist(), v0.tolist(), w0.tolist(), strict=True):
        ix = int(round((float(x) - origin_x) / float(spacing_x)))
        iy = int(round((float(y) - origin_y) / float(spacing_y)))
        if ix < 0 or iy < 0 or ix >= size_x or iy >= size_y:
            continue
        v_lat[iy, ix] = float(vv)
        w_lat[iy, ix] = float(ww)

    if float(smooth_sigma_px) > 0.0:
        sigma_y = float(smooth_sigma_px) / float(spacing_y)
        sigma_x = float(smooth_sigma_px) / float(spacing_x)
        v_lat = _masked_gaussian_smooth_2d(v_lat, w_lat, sigma=(sigma_y, sigma_x), mode="nearest")
    else:
        v_lat = _fill_nans_nearest_2d(v_lat)

    field_img = sitk.GetImageFromArray(v_lat.astype(np.float32, copy=False))
    field_img.SetOrigin((float(origin_x), float(origin_y)))
    field_img.SetSpacing((float(spacing_x), float(spacing_y)))

    ref_img = sitk.Image(int(w), int(h), sitk.sitkFloat32)
    ref_img.SetOrigin((0.0, 0.0))
    ref_img.SetSpacing((1.0, 1.0))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
    dense_img = resampler.Execute(field_img)
    dense = sitk.GetArrayFromImage(dense_img)
    if dense.shape != (h, w):
        raise ValueError(f"Unexpected dense field shape: {dense.shape}, expected {(h, w)}")
    return dense.astype(np.float32, copy=False)


def _bspline_field_cache_path(
    *,
    cache_dir: Path,
    target: str,
    z: int,
    fit_pool_hash: str,
    tps_n_files: int,
    affine_hash: str,
    tps_point_method: str,
    grid_n: int,
    spacing_x_px: int,
    spacing_y_px: int,
    patch_radius_px: int,
    upsample_factor: int,
    max_shift_px: float,
    smooth_sigma_px: float,
    deaffine: bool,
) -> Path:
    smooth_tag = f"{float(smooth_sigma_px):g}"
    max_tag = f"{float(max_shift_px):g}"
    return cache_dir / (
        "bspline_field_"
        f"{target}_z{int(z)}"
        f"_pool{fit_pool_hash}_n{int(tps_n_files)}"
        f"_aff{affine_hash}"
        f"_m{tps_point_method}"
        f"_gridn{int(grid_n)}_sx{int(spacing_x_px)}_sy{int(spacing_y_px)}_r{int(patch_radius_px)}"
        f"_u{int(upsample_factor)}_max{max_tag}"
        f"_smooth{smooth_tag}_deaff{int(bool(deaffine))}"
        ".npz"
    )


def _write_bspline_field_cache(
    *,
    path: Path,
    meta: dict[str, object],
    dx_dense: np.ndarray,
    dy_dense: np.ndarray,
) -> None:
    dx_dense = np.asarray(dx_dense, dtype=np.float32)
    dy_dense = np.asarray(dy_dense, dtype=np.float32)
    if dx_dense.shape != dy_dense.shape:
        raise ValueError(f"dx_dense and dy_dense must have same shape, got {dx_dense.shape} vs {dy_dense.shape}")
    if dx_dense.ndim != 2:
        raise ValueError(f"dx_dense/dy_dense must be 2D arrays, got shape {dx_dense.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(
        path,
        meta_json=np.asarray(meta_json),
        dx_dense=dx_dense,
        dy_dense=dy_dense,
    )


def apply_displacement_field(
    moving: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    warp_block_rows: int = 256,
    mode: str = "constant",
    cval: float = 0.0,
    clamp: bool = False,
) -> np.ndarray:
    h, w = moving.shape
    if dx.shape != moving.shape or dy.shape != moving.shape:
        raise ValueError(f"Displacement shape mismatch: moving={moving.shape}, dx={dx.shape}, dy={dy.shape}")
    block_rows = int(warp_block_rows)
    if block_rows <= 0:
        raise ValueError("warp_block_rows must be > 0")
    if mode not in {"constant", "nearest", "reflect", "mirror", "wrap"}:
        raise ValueError(f"Unsupported mode={mode!r} for map_coordinates.")

    warped = np.empty_like(moving, dtype=np.float32)
    xs_full = np.arange(w, dtype=np.float64)
    for y0i in range(0, h, block_rows):
        y1i = min(h, y0i + block_rows)
        ys_full = np.arange(y0i, y1i, dtype=np.float64)
        yy, xx = np.meshgrid(ys_full, xs_full, indexing="ij")
        dx_blk = dx[y0i:y1i].astype(np.float64, copy=False)
        dy_blk = dy[y0i:y1i].astype(np.float64, copy=False)
        sample_y = yy + dy_blk
        sample_x = xx + dx_blk
        if clamp:
            sample_y = np.clip(sample_y, 0.0, float(h - 1))
            sample_x = np.clip(sample_x, 0.0, float(w - 1))
        warped[y0i:y1i] = ndi.map_coordinates(
            moving,
            [sample_y, sample_x],
            order=1,
            mode=mode,
            cval=float(cval),
        ).astype(np.float32, copy=False)

    return warped


def generate_control_grid(
    *,
    height: int,
    width: int,
    control_grid_n: int,
    patch_radius_px: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n = int(control_grid_n)
    radius = int(patch_radius_px)
    if n < 2:
        raise ValueError("control_grid_n must be >= 2")
    if radius <= 0:
        raise ValueError("patch_radius_px must be > 0")

    if height <= 0 or width <= 0:
        raise ValueError("height/width must be > 0")

    # Keep patches fully in-bounds with a small additional offset from edges.
    # (No patch padding is used in phase correlation.)
    edge_offset_px = 5
    margin = radius + int(edge_offset_px)
    x_min = margin
    y_min = margin
    x_max = int(width) - 1 - margin
    y_max = int(height) - 1 - margin
    if x_max < x_min or y_max < y_min:
        raise ValueError("Empty TPS control grid (check spacing/radius vs image size).")

    # Build a regular integer lattice with exactly n points per axis, ensuring all points
    # lie on that lattice (no "append x_max" misalignment).
    span_x = int(x_max) - int(x_min)
    span_y = int(y_max) - int(y_min)
    if span_x < (n - 1) or span_y < (n - 1):
        raise ValueError(
            f"Image too small for control_grid_n={n} with margin={margin}: "
            f"x_span={span_x}, y_span={span_y}"
        )

    spacing_x = span_x // (n - 1)
    spacing_y = span_y // (n - 1)
    if spacing_x <= 0 or spacing_y <= 0:
        raise ValueError(
            f"Invalid derived control grid spacing: spacing_x={spacing_x}, spacing_y={spacing_y} "
            f"for control_grid_n={n}"
        )

    start_x = int(x_max) - spacing_x * (n - 1)
    start_y = int(y_max) - spacing_y * (n - 1)
    if start_x < int(x_min) or start_y < int(y_min):
        raise ValueError("Internal error: derived control grid start is out of bounds.")

    xs_int = [start_x + i * spacing_x for i in range(n)]
    ys_int = [start_y + i * spacing_y for i in range(n)]

    xs: list[float] = []
    ys: list[float] = []
    for y in ys_int:
        for x in xs_int:
            xs.append(float(x))
            ys.append(float(y))

    x0 = np.asarray(xs, dtype=np.float64)
    y0 = np.asarray(ys, dtype=np.float64)
    if x0.size == 0:
        raise ValueError("Empty TPS control grid (check spacing/radius vs image size).")
    return x0, y0, int(spacing_x), int(spacing_y)


def estimate_displacements_on_grid(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    x0: np.ndarray,
    y0: np.ndarray,
    patch_radius_px: int,
    upsample_factor: int,
    max_shift_px: float,
    method: str = "sitk",
) -> tuple[np.ndarray, np.ndarray]:
    if fixed.shape != moving.shape:
        raise ValueError(f"Shape mismatch: fixed={fixed.shape}, moving={moving.shape}")
    if x0.shape != y0.shape:
        raise ValueError("x0 and y0 must have the same shape.")
    method = str(method)
    if method not in {"sitk", "phase"}:
        raise ValueError(f"Unknown method={method!r}; expected 'sitk' or 'phase'.")

    radius = int(patch_radius_px)
    if radius <= 0:
        raise ValueError("patch_radius_px must be > 0")
    if upsample_factor <= 0:
        raise ValueError("upsample_factor must be > 0")
    if max_shift_px <= 0:
        raise ValueError("max_shift_px must be > 0")

    dx0 = np.full_like(x0, np.nan, dtype=np.float64)
    dy0 = np.full_like(y0, np.nan, dtype=np.float64)

    phase_window: np.ndarray | None = None
    if method == "phase":
        size = radius * 2 + 1
        wy = np.hanning(size).astype(np.float32, copy=False)
        wx = np.hanning(size).astype(np.float32, copy=False)
        phase_window = (wy[:, None] * wx[None, :]).astype(np.float32, copy=False)

    for i in range(int(x0.size)):
        x = int(x0[i])
        y = int(y0[i])
        fixed_patch = _patch_centered(fixed, y=y, x=x, radius=radius)
        moving_patch = _patch_centered(moving, y=y, x=x, radius=radius)

        # Skip flat/no-signal patches which can confuse the optimizer.
        if float(np.std(fixed_patch)) < 1.0e-6 or float(np.std(moving_patch)) < 1.0e-6:
            continue

        if method == "phase":
            # phase_cross_correlation returns the shift (dy, dx) to apply to `moving`
            # to align it to `fixed` (moving->fixed). We store dx/dy as a fixed->moving
            # mapping because warping samples `moving` at (y + dy, x + dx).
            fixed_patch_f = fixed_patch.astype(np.float32, copy=False)
            moving_patch_f = moving_patch.astype(np.float32, copy=False)
            fixed_patch_f = fixed_patch_f - float(np.median(fixed_patch_f))
            moving_patch_f = moving_patch_f - float(np.median(moving_patch_f))
            fixed_scale = float(np.std(fixed_patch_f))
            moving_scale = float(np.std(moving_patch_f))
            if np.isfinite(fixed_scale) and fixed_scale > 1.0e-6:
                fixed_patch_f = fixed_patch_f / fixed_scale
            if np.isfinite(moving_scale) and moving_scale > 1.0e-6:
                moving_patch_f = moving_patch_f / moving_scale
            assert phase_window is not None
            fixed_patch_f = fixed_patch_f * phase_window
            moving_patch_f = moving_patch_f * phase_window
            shift_dy_dx, _, _ = phase_cross_correlation(
                fixed_patch_f,
                moving_patch_f,
                upsample_factor=int(upsample_factor),
            )
            dy = -float(shift_dy_dx[0])
            dx = -float(shift_dy_dx[1])
        else:
            fixed_img = sitk.GetImageFromArray(fixed_patch.astype(np.float32, copy=False))
            moving_img = sitk.GetImageFromArray(moving_patch.astype(np.float32, copy=False))

            n_iter = max(10, int(5 * int(upsample_factor)))
            reg = sitk.ImageRegistrationMethod()
            reg.SetMetricAsMeanSquares()
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsRegularStepGradientDescent(
                learningRate=1.0,
                minStep=0.01,
                numberOfIterations=int(n_iter),
                relaxationFactor=0.5,
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel([2, 1])
            reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

            tfm0 = sitk.TranslationTransform(2)
            reg.SetInitialTransform(tfm0, inPlace=False)
            tfm = reg.Execute(fixed_img, moving_img)
            tx, ty = tfm.GetParameters()  # fixed->moving (x,y) in SimpleITK's convention

            dx = float(tx)  # fixed->moving
            dy = float(ty)
        if float(np.hypot(dx, dy)) > float(max_shift_px):
            continue

        dx0[i] = dx
        dy0[i] = dy

    return dx0, dy0


# Fit quality metric
# ------------------
# We want an objective score so we can compare optimizers/iteration budgets.
# Fit score: Pearson correlation coefficient on a contrast-normalized high-pass
# image (Gaussian sigma=2px) to reduce intensity-scale / background sensitivity.
def _block_from(img: np.ndarray, *, y0: int, x0: int, size: int) -> np.ndarray:
    if y0 < 0 or x0 < 0 or size <= 0:
        raise ValueError("y0/x0 must be >= 0 and size must be > 0.")
    y1 = y0 + size
    x1 = x0 + size
    if y1 > img.shape[0] or x1 > img.shape[1]:
        raise ValueError(
            f"Requested block [{y0}:{y1}, {x0}:{x1}] out of bounds for image shape {img.shape}."
        )
    return img[y0:y1, x0:x1]


def _norm01_percentile(img: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi) <= float(lo):
        raise ValueError("Invalid percentiles for normalization.")
    x = (x - float(lo)) / float(hi - lo)
    return np.clip(x, 0.0, 1.0)


def _highpass_gaussian(img: np.ndarray, *, sigma_px: float = 3.0) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    lp = gaussian(x, sigma=float(sigma_px), preserve_range=True)
    return (x - lp).astype(np.float32, copy=False)


def _prep_for_fit(img: np.ndarray) -> np.ndarray:
    return _norm01_percentile(_highpass_gaussian(img, sigma_px=3.0))


def _fmt_mat(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=6, floatmode="fixed", suppress_small=False)


def _split_train_val(
    items: list[Path],
    *,
    train_frac: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not items:
        return [], []
    f = float(train_frac)
    if f <= 0.0 or f > 1.0:
        raise ValueError("train_frac must be in (0, 1].")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(items))
    n_train = max(1, int(np.floor(f * float(len(items)))))
    train_idx = set(order[:n_train].tolist())
    train = [p for i, p in enumerate(items) if i in train_idx]
    val = [p for i, p in enumerate(items) if i not in train_idx]
    return train, val


def _fit_weighted_plane(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    w: np.ndarray,
) -> np.ndarray:
    """
    Fit z ~= b0 + b1*x + b2*y with weighted least squares.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.shape != y.shape or x.shape != z.shape or x.shape != w.shape:
        raise ValueError("x/y/z/w must have same shape")
    if x.ndim != 1:
        raise ValueError("x/y/z/w must be 1D")
    if np.any(w < 0):
        raise ValueError("w must be nonnegative")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    if int(np.sum(mask)) < 3:
        raise ValueError("Need at least 3 weighted points to fit a plane.")

    X = np.stack([np.ones(int(np.sum(mask))), x[mask], y[mask]], axis=1)
    sqrtw = np.sqrt(w[mask])
    beta, *_ = np.linalg.lstsq(X * sqrtw[:, None], z[mask] * sqrtw, rcond=None)
    return beta.astype(np.float64, copy=False)


def _eval_plane(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64).reshape(3)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return beta[0] + beta[1] * x + beta[2] * y


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a.astype(np.float32, copy=False).ravel()
    b0 = b.astype(np.float32, copy=False).ravel()
    a0 = a0 - float(a0.mean())
    b0 = b0 - float(b0.mean())
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def _params_centered_to_hom(A: np.ndarray, t: np.ndarray, *, center_xy: np.ndarray) -> np.ndarray:
    """
    Convert center-based affine params to homogeneous matrix about the origin.

    Our warp applies (x,y) in pixel coordinates as:
        x' = A @ (x - c) + c + t
    where c is the image center in (x,y) order.

    This is equivalent to:
        x' = A @ x + t_eff,  where t_eff = t + c - A@c
    so we average H = [[A, t_eff],[0,0,1]] in log-Euclidean space.
    """
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if A.shape != (2, 2):
        raise ValueError(f"A must be (2,2), got {A.shape}")
    t_eff = t + c - (A @ c)
    H = np.eye(3, dtype=np.float64)
    H[:2, :2] = A
    H[:2, 2] = t_eff
    return H


def _hom_to_params_centered(H: np.ndarray, *, center_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be (3,3), got {H.shape}")
    A = H[:2, :2].copy()
    t_eff = H[:2, 2].copy()
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    t = t_eff - c + (A @ c)
    return A, t


def _real_if_close(M: np.ndarray, *, tol: float) -> np.ndarray:
    if np.iscomplexobj(M):
        imag_max = float(np.max(np.abs(np.imag(M))))
        if imag_max > float(tol):
            raise ValueError(
                f"Matrix log/exp produced significant imaginary part (max |Im|={imag_max:g}). "
                "This can happen if some transforms have negative determinant or are far apart."
            )
        return np.real(M)
    return M


def average_affines_logeuclid(
    As: np.ndarray,
    ts: np.ndarray,
    *,
    center_xy: np.ndarray,
    weights: np.ndarray | None = None,
    imag_tol: float = 1e-8,
    project_last_row: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    As = np.asarray(As, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    if As.ndim != 3 or As.shape[1:] != (2, 2):
        raise ValueError(f"As must be (N,2,2), got {As.shape}")
    if ts.shape != (As.shape[0], 2):
        raise ValueError(f"ts must be (N,2), got {ts.shape}")

    n = int(As.shape[0])
    if n <= 0:
        raise ValueError("Need at least one transform to average.")

    if weights is None:
        w = np.full(n, 1.0 / float(n), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(n)
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must sum to a positive number")
        w = w / s

    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    L_bar = np.zeros((3, 3), dtype=np.complex128)
    for i in range(n):
        H = _params_centered_to_hom(As[i], ts[i], center_xy=c)
        detA = float(np.linalg.det(As[i]))
        if detA <= 0.0:
            raise ValueError(f"Non-positive det(A) at i={i}: det={detA:g}")
        L_bar += float(w[i]) * logm(H)

    L_bar = _real_if_close(L_bar, tol=float(imag_tol))
    H_bar = expm(L_bar)
    H_bar = _real_if_close(H_bar, tol=float(imag_tol))

    if project_last_row:
        H_bar = np.asarray(H_bar, dtype=np.float64)
        H_bar[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return _hom_to_params_centered(H_bar, center_xy=c)


def _roi_slices(
    *,
    height: int,
    width: int,
    size: int,
    margin: int,
) -> dict[str, tuple[slice, slice]]:
    h = int(height)
    w = int(width)
    s = int(size)
    m = int(margin)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be > 0")
    if s <= 0:
        raise ValueError("size must be > 0")
    if m < 0:
        raise ValueError("margin must be >= 0")
    if s + 2 * m > h or s + 2 * m > w:
        raise ValueError(f"ROI size too large for image: size={s}, margin={m}, shape={(h, w)}")

    tl = (m, m)
    tr = (m, w - m - s)
    bl = (h - m - s, m)
    br = (h - m - s, w - m - s)
    cy = (h - s) // 2
    cx = (w - s) // 2

    coords = {
        "tl": tl,
        "tr": tr,
        "bl": bl,
        "br": br,
        "center": (cy, cx),
    }
    return {k: (slice(y, y + s), slice(x, x + s)) for k, (y, x) in coords.items()}


def control_point_drift_rms(
    dxs: np.ndarray,
    dys: np.ndarray,
    *,
    dx0: np.ndarray,
    dy0: np.ndarray,
) -> np.ndarray:
    """
    Per-point drift as RMS residual magnitude around the per-point median displacement.

    This uses SimpleITK's vector magnitude to avoid x/y component-order mistakes.
    All values are in pixel units.
    """
    dxs = np.asarray(dxs, dtype=np.float64)
    dys = np.asarray(dys, dtype=np.float64)
    dx0 = np.asarray(dx0, dtype=np.float64)
    dy0 = np.asarray(dy0, dtype=np.float64)
    if dxs.shape != dys.shape or dxs.ndim != 2:
        raise ValueError(f"dxs/dys must be (N_files, N_points), got dxs={dxs.shape}, dys={dys.shape}")
    if dx0.shape != (dxs.shape[1],) or dy0.shape != (dxs.shape[1],):
        raise ValueError(f"dx0/dy0 must be (N_points,), got dx0={dx0.shape}, dy0={dy0.shape}")

    res_dx = dxs - dx0[None, :]
    res_dy = dys - dy0[None, :]
    res_vec = np.stack([res_dx, res_dy], axis=-1).astype(np.float32, copy=False)
    res_img = sitk.GetImageFromArray(res_vec, isVector=True)
    mag = sitk.GetArrayFromImage(sitk.VectorMagnitude(res_img)).astype(np.float64, copy=False)
    return np.sqrt(np.nanmean(mag**2, axis=0))


def alignment_metrics(
    ref: np.ndarray,
    moving: np.ndarray,
    *,
    size: int = 512,
    margin: int = 32,
) -> dict[str, float]:
    if ref.shape != moving.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, moving={moving.shape}")

    rois = _roi_slices(height=int(ref.shape[0]), width=int(ref.shape[1]), size=int(size), margin=int(margin))
    corners = ("tl", "tr", "bl", "br")
    corrs = [_corrcoef(ref[rois[k]], moving[rois[k]]) for k in corners]
    return {"corrcoef": float(np.mean(corrs))}


def _warp_affine_sitk(
    moving: np.ndarray,
    fixed_like: np.ndarray,
    *,
    A: np.ndarray,
    t: np.ndarray,
    inverse: bool,
) -> np.ndarray:
    if moving.shape != fixed_like.shape:
        raise ValueError(f"Shape mismatch: moving={moving.shape}, fixed_like={fixed_like.shape}")
    if A.shape != (2, 2):
        raise ValueError(f"A must be (2,2), got {A.shape}")
    if t.shape != (2,):
        raise ValueError(f"t must be (2,), got {t.shape}")

    moving_img = sitk.GetImageFromArray(moving.astype(np.float32, copy=False))
    fixed_img = sitk.GetImageFromArray(fixed_like.astype(np.float32, copy=False))

    tfm = sitk.AffineTransform(2)
    tfm.SetMatrix([float(x) for x in A.reshape(-1)])
    tfm.SetTranslation([float(x) for x in t.reshape(-1)])
    height, width = fixed_like.shape
    tfm.SetCenter([(float(width) - 1.0) / 2.0, (float(height) - 1.0) / 2.0])
    if inverse:
        tfm = tfm.GetInverse()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(tfm)

    warped = sitk.GetArrayFromImage(resampler.Execute(moving_img))
    return warped.astype(np.float32, copy=False)


def _effective_affine_params_for_resample(
    A: np.ndarray,
    t: np.ndarray,
    *,
    height: int,
    width: int,
    inverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (A,t,inverse) into an equivalent (A_eff,t_eff) that can be applied with inverse=False.

    This matters because the inverse of an affine transform depends on the center of rotation.
    """
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    if A.shape != (2, 2):
        raise ValueError(f"A must be (2,2), got {A.shape}")
    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be > 0")

    tfm = sitk.AffineTransform(2)
    tfm.SetMatrix([float(x) for x in A.reshape(-1)])
    tfm.SetTranslation([float(x) for x in t.reshape(-1)])
    tfm.SetCenter([(float(w) - 1.0) / 2.0, (float(h) - 1.0) / 2.0])
    if inverse:
        tfm = tfm.GetInverse()

    A_eff = np.asarray(tfm.GetMatrix(), dtype=np.float64).reshape(2, 2)
    t_eff = np.asarray(tfm.GetTranslation(), dtype=np.float64).reshape(2)
    return A_eff, t_eff


def _write_cli_register_chromatic_txt(
    *,
    path: Path,
    A: np.ndarray,
    t: np.ndarray,
) -> None:
    """
    Write chromatic correction in the format expected by `preprocess register --chromatic`.

    File format: 6 newline-separated floats:
      a00, a01, a10, a11, tx, ty
    """
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    if A.shape != (2, 2):
        raise ValueError(f"A must be (2,2), got {A.shape}")
    vals = np.array([*A.flatten(), *t], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, vals.tolist())) + "\n")


def _write_cli_register_chromatic_field_npz(
    *,
    path: Path,
    meta: dict[str, object],
    dx_dense: np.ndarray,
    dy_dense: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    dx_control: np.ndarray,
    dy_control: np.ndarray,
    keep_mask: np.ndarray,
    valid_counts: np.ndarray,
    files: list[Path],
) -> None:
    dx_dense = np.asarray(dx_dense, dtype=np.float32)
    dy_dense = np.asarray(dy_dense, dtype=np.float32)
    if dx_dense.shape != dy_dense.shape or dx_dense.ndim != 2:
        raise ValueError(f"dx_dense/dy_dense must be 2D and same shape, got {dx_dense.shape} vs {dy_dense.shape}")

    xgrid = np.asarray(xgrid, dtype=np.float64).reshape(-1)
    ygrid = np.asarray(ygrid, dtype=np.float64).reshape(-1)
    dx_control = np.asarray(dx_control, dtype=np.float32).reshape(-1)
    dy_control = np.asarray(dy_control, dtype=np.float32).reshape(-1)
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
    valid_counts = np.asarray(valid_counts, dtype=np.int32).reshape(-1)
    if not (xgrid.shape == ygrid.shape == dx_control.shape == dy_control.shape == keep_mask.shape):
        raise ValueError(
            "xgrid/ygrid/dx_control/dy_control/keep_mask must have the same flat shape, "
            f"got {xgrid.shape}, {ygrid.shape}, {dx_control.shape}, {dy_control.shape}, {keep_mask.shape}"
        )
    if valid_counts.shape != keep_mask.shape:
        raise ValueError(f"valid_counts must have shape {keep_mask.shape}, got {valid_counts.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(
        path,
        meta_json=np.asarray(meta_json),
        dx_dense=dx_dense,
        dy_dense=dy_dense,
        xgrid=xgrid,
        ygrid=ygrid,
        dx_control=dx_control,
        dy_control=dy_control,
        keep_mask=keep_mask.astype(np.uint8, copy=False),
        valid_counts=valid_counts,
        files=np.asarray(list(map(str, files)), dtype=np.str_),
    )


def _affine_hash(A: np.ndarray, t: np.ndarray, *, inverse: bool) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(A, dtype=np.float64).tobytes())
    h.update(np.asarray(t, dtype=np.float64).tobytes())
    h.update(b"inv1" if inverse else b"inv0")
    return h.hexdigest()[:12]


def _overlay_rg01(r: np.ndarray, g: np.ndarray) -> np.ndarray:
    rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = np.clip(r, 0.0, 1.0)
    rgb[..., 1] = np.clip(g, 0.0, 1.0)
    return rgb


def _prep_for_viz(img2d: np.ndarray) -> np.ndarray:
    hp = _highpass_gaussian(img2d.astype(np.float32, copy=False), sigma_px=2.0)
    scale = float(np.percentile(np.abs(hp), 99.5))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(hp, dtype=np.float32)
    hp = np.clip(hp, -scale, scale) / scale
    return ((hp + 1.0) / 2.0).astype(np.float32, copy=False)


def run_register_genmaster(
    *,
    target: str = "750",
    data_dir: Path | None = None,
    max_files: int = 0,
    file_index: int = 1,
    z: int = 4,
    affine_sample_n: int = 10,
    affine_sample_seed: int = 0,
    cv_train_frac: float = 1.0,
    cv_seed: int = 0,
    cv_eval_n: int = 0,
    do_nonrigid_refinement: bool = True,
    nonrigid_tps_grid_n: int = 6,
    nonrigid_tps_patch_radius_px: int = 64,
    nonrigid_tps_upsample_factor: int = 10,
    nonrigid_tps_max_shift_px: float = 10.0,
    nonrigid_tps_point_method: str = "sitk",
    nonrigid_tps_smooth: float = 0.0,
    nonrigid_tps_deaffine: bool = False,
    nonrigid_tps_min_valid_per_point: int = 5,
    nonrigid_tps_warp_block_rows: int = 256,
    nonrigid_tps_warp_mode: str = "constant",
    nonrigid_tps_warp_cval: float = 0.0,
    nonrigid_tps_warp_clamp: bool = False,
    save_overlay_qc: bool = True,
    qc_dir: Path | None = None,
    use_tps_cache: bool = True,
    cache_dir: Path = Path("/working/20260103_chromatic/register_genmaster_cache"),
    use_affine_cache: bool = True,
    save_quiver_qc: bool = True,
    quiver_stride: int = 64,
    quiver_magnify: float = 200.0,
    print_max_displacement_table: bool = False,
) -> None:
    # Somehow you need to run this, stop this cell and run it again.
    # The first run can stall (first elastix call / ITK init); interrupt and rerun.
    global _RUN_T0
    _RUN_T0 = time.perf_counter()

    if data_dir is None:
        data_dir = Path(f"/working/20260103_chromatic/560_{target}--0002-6149-4605")

    _log(
        "config: "
        f"target={target}, data_dir={data_dir}, file_index={file_index}, max_files={max_files}, z={z}; "
        f"affine_sample_n={affine_sample_n}, affine_sample_seed={affine_sample_seed}; "
        f"tps(grid_n={nonrigid_tps_grid_n}, r={nonrigid_tps_patch_radius_px}, "
        f"u={nonrigid_tps_upsample_factor}, max={nonrigid_tps_max_shift_px}, method={str(nonrigid_tps_point_method)}, "
        f"smooth_px={nonrigid_tps_smooth:g}, "
        f"deaffine={bool(nonrigid_tps_deaffine)}); "
        f"tps_warp(mode={nonrigid_tps_warp_mode}, cval={nonrigid_tps_warp_cval:g}, clamp={nonrigid_tps_warp_clamp}); "
        f"use_affine_cache={use_affine_cache}, use_tps_cache={use_tps_cache}; "
        f"save_overlay_qc={save_overlay_qc}, save_quiver_qc={save_quiver_qc}"
    )

    with _log_step("discover files"):
        files = sorted(data_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    max_files = int(max_files)
    file_index = int(file_index)
    if max_files < 0:
        raise ValueError("max_files must be >= 0 (use 0 for all files).")

    if max_files == 0:
        selected_files = files
    else:
        if file_index < 0 or file_index >= len(files):
            raise IndexError(f"file_index={file_index} out of range for {len(files)} files.")
        selected_files = files[file_index : file_index + max_files]
        if not selected_files:
            raise ValueError("No files selected (check file_index/max_files).")

    _log(
        f"selected_files: n={len(selected_files)} first={selected_files[0].name} last={selected_files[-1].name} "
        f"(max_files={max_files}, file_index={file_index})"
    )

    cache_dir = Path(cache_dir)
    if qc_dir is None:
        qc_dir = cache_dir / "qc"
    qc_dir = Path(qc_dir)

    # Nonrigid fit uses bead files in data-dir. For hyperparameter sweeps, you can enable
    # a simple cross-validation split (fit on train subset; report metric on val subset).
    tps_files = files
    tps_pool_hash = hashlib.sha256("\n".join(map(str, tps_files)).encode("utf-8")).hexdigest()[:12]
    _log(f"nonrigid pool: n={len(tps_files)} pool_hash={tps_pool_hash} (uses all images in data_dir)")

    cv_train_frac = float(cv_train_frac)
    cv_seed = int(cv_seed)
    cv_eval_n = int(cv_eval_n)
    if cv_train_frac < 1.0:
        tps_fit_files, tps_val_files = _split_train_val(tps_files, train_frac=cv_train_frac, seed=cv_seed)
        fit_pool_hash = hashlib.sha256("\n".join(map(str, tps_fit_files)).encode("utf-8")).hexdigest()[:12]
        _log(
            f"CV split: train_frac={cv_train_frac:g}, seed={cv_seed} -> "
            f"train_n={len(tps_fit_files)} val_n={len(tps_val_files)} fit_pool_hash={fit_pool_hash}"
        )
    else:
        tps_fit_files = tps_files
        tps_val_files = []
        fit_pool_hash = tps_pool_hash

    with _log_step("infer image shape"):
        img0 = tifffile.imread(tps_files[0])[:-2]
        img0 = img0.reshape(-1, 2, img0.shape[-2], img0.shape[-1])
        height0, width0 = int(img0.shape[-2]), int(img0.shape[-1])
        shape_hw0 = (height0, width0)
        center_xy0 = np.asarray([(float(width0) - 1.0) / 2.0, (float(height0) - 1.0) / 2.0], dtype=np.float64)

    affine_avg_method = "logeuclid_centered"
    affine_cache_path = cache_dir / (
        f"affine_shared_{target}_z{int(z)}_pool{tps_pool_hash}_n{int(affine_sample_n)}_seed{int(affine_sample_seed)}"
        f"_avg{affine_avg_method}.npz"
    )

    A_med: np.ndarray
    t_med: np.ndarray
    inputs_inverted: bool
    used_affine_cache = False
    if use_affine_cache and affine_cache_path.exists():
        _log(f"affine cache: candidate={affine_cache_path}")
        with _log_step("affine cache: load npz"):
            loaded = np.load(affine_cache_path, allow_pickle=False)
            meta = json.loads(str(loaded["meta_json"].item()))
        expected_sample_n = min(int(affine_sample_n), len(tps_files))
        params_ok = (
            meta.get("target") == target
            and str(meta.get("data_dir")) == str(data_dir)
            and int(meta.get("z")) == int(z)
            and int(meta.get("sample_n")) == int(expected_sample_n)
            and int(meta.get("seed")) == int(affine_sample_seed)
            and str(meta.get("tps_pool_hash")) == str(tps_pool_hash)
            and str(meta.get("avg_method")) == str(affine_avg_method)
            and meta.get("shape_hw") == [int(shape_hw0[0]), int(shape_hw0[1])]
            and meta.get("center_xy") == [float(center_xy0[0]), float(center_xy0[1])]
            and isinstance(meta.get("inputs_inverted"), bool)
        )
        if params_ok:
            A_med = loaded["A_med"].astype(np.float64, copy=False)
            t_med = loaded["t_med"].astype(np.float64, copy=False)
            inputs_inverted = bool(meta.get("inputs_inverted"))
            used_affine_cache = True
            print(
                f"affine cache: loaded {affine_cache_path} (inputs_inverted={inputs_inverted}, hash={_affine_hash(A_med, t_med, inverse=False)})",
                flush=True,
            )
        else:
            print(f"affine cache: ignoring {affine_cache_path} (meta mismatch)", flush=True)

    if not used_affine_cache:
        rng = np.random.default_rng(int(affine_sample_seed))
        sample_n = min(int(affine_sample_n), len(tps_files))
        sample_indices = np.sort(rng.choice(len(tps_files), size=sample_n, replace=False))
        sample_files = [tps_files[int(i)] for i in sample_indices.tolist()]
        print(
            f"affine sampling: n={sample_n}/{len(tps_files)} files (seed={affine_sample_seed}); indices={sample_indices.tolist()}",
            flush=True,
        )

        print("affine sampling: importing itk/elastix (FitAffine)...", flush=True)
        from fishtools.preprocess.chromatic import FitAffine

        affine_As: list[np.ndarray] = []
        affine_ts: list[np.ndarray] = []
        mse_direct = 0.0
        mse_inv = 0.0
        mse_direct_by_roi: dict[str, list[float]] = {}
        mse_inv_by_roi: dict[str, list[float]] = {}
        rois: dict[str, tuple[slice, slice]] | None = None

        for k, file in enumerate(sample_files, start=1):
            fit = FitAffine(log_to_console=False)
            with _log_step(f"affine fit {k}/{sample_n}: read {file.name}"):
                img = tifffile.imread(file)[:-2]
                img = img.reshape(-1, 2, img.shape[-2], img.shape[-1])

            ref_raw = img[int(z), 0].astype(np.float32, copy=False)
            mov_raw = img[int(z), 1].astype(np.float32, copy=False)
            with _log_step(f"affine fit {k}/{sample_n}: preprocess"):
                ref = _prep_for_fit(ref_raw)
                mov = _prep_for_fit(mov_raw)
            before = alignment_metrics(ref, mov)
            if rois is None:
                roi_size = int(min(512, ref.shape[0] - 64, ref.shape[1] - 64))
                if roi_size <= 0:
                    raise ValueError(f"Image too small for ROI evaluation: shape={ref.shape}")
                rois = _roi_slices(height=int(ref.shape[0]), width=int(ref.shape[1]), size=roi_size, margin=32)

            print(f"affine fit {k}/{sample_n}: {file.name} | starting elastix", flush=True)
            with _log_step(f"affine fit {k}/{sample_n}: elastix optimize"):
                A, t, warped_elx = fit.fit(ref, mov)
            after_elx = alignment_metrics(ref, warped_elx)
            print(
                f"affine fit {k}/{sample_n}: {file.name} | corrcoef(prep,crop): before={before['corrcoef']:.6f}, elastix={after_elx['corrcoef']:.6f}",
                flush=True,
            )

            with _log_step(f"affine fit {k}/{sample_n}: determine inverse-vs-direct"):
                warped_direct = _warp_affine_sitk(mov, ref, A=A, t=t, inverse=False)
                warped_inv = _warp_affine_sitk(mov, ref, A=A, t=t, inverse=True)
            assert rois is not None
            mse_direct_rois: list[float] = []
            mse_inv_rois: list[float] = []
            for name, sl in rois.items():
                md = float(np.mean((warped_direct[sl] - warped_elx[sl]) ** 2))
                mi = float(np.mean((warped_inv[sl] - warped_elx[sl]) ** 2))
                mse_direct_by_roi.setdefault(name, []).append(md)
                mse_inv_by_roi.setdefault(name, []).append(mi)
                mse_direct_rois.append(md)
                mse_inv_rois.append(mi)
            mse_direct += float(np.mean(mse_direct_rois))
            mse_inv += float(np.mean(mse_inv_rois))

            affine_As.append(A.astype(np.float64, copy=False))
            affine_ts.append(t.astype(np.float64, copy=False))

        inputs_inverted = bool(mse_inv < mse_direct)
        if inputs_inverted:
            As_eff: list[np.ndarray] = []
            ts_eff: list[np.ndarray] = []
            for A_i, t_i in zip(affine_As, affine_ts, strict=True):
                H_i = _params_centered_to_hom(A_i, t_i, center_xy=center_xy0)
                H_i_inv = np.linalg.inv(H_i)
                A_i_eff, t_i_eff = _hom_to_params_centered(H_i_inv, center_xy=center_xy0)
                As_eff.append(A_i_eff)
                ts_eff.append(t_i_eff)
            affine_As = As_eff
            affine_ts = ts_eff

        A_med, t_med = average_affines_logeuclid(
            np.stack(affine_As),
            np.stack(affine_ts),
            center_xy=center_xy0,
        )
        affine_hash = _affine_hash(A_med, t_med, inverse=False)
        print(
            "affine sampling: "
            f"mse_vs_elastix(direct)={mse_direct:.6g}, mse_vs_elastix(inverse)={mse_inv:.6g} (avg over ROIs) -> inputs_inverted={inputs_inverted}",
            flush=True,
        )
        if mse_direct_by_roi:
            print("affine sampling: mse_vs_elastix per ROI (median/max over samples):", flush=True)
            for name in sorted(mse_direct_by_roi.keys()):
                md = float(np.median(mse_direct_by_roi[name]))
                mx = float(np.max(mse_direct_by_roi[name]))
                mi_md = float(np.median(mse_inv_by_roi[name]))
                mi_mx = float(np.max(mse_inv_by_roi[name]))
                print(
                    f"  {name:>6s}: direct(med/max)={md:.6g}/{mx:.6g}  inverse(med/max)={mi_md:.6g}/{mi_mx:.6g}",
                    flush=True,
                )
        print("affine sampling: avg(logeuclid) A=\n" + _fmt_mat(A_med), flush=True)
        print("affine sampling: avg(logeuclid) t=" + _fmt_mat(t_med) + "  # (tx, ty)", flush=True)
        print(f"affine sampling: hash={affine_hash}", flush=True)

        if use_affine_cache:
            with _log_step("affine cache: write npz"):
                cache_dir.mkdir(parents=True, exist_ok=True)
                meta_json = json.dumps(
                    dict(
                        target=target,
                        data_dir=str(data_dir),
                        tps_pool_hash=str(tps_pool_hash),
                        z=int(z),
                        sample_n=int(sample_n),
                        seed=int(affine_sample_seed),
                        avg_method=str(affine_avg_method),
                        shape_hw=[int(shape_hw0[0]), int(shape_hw0[1])],
                        center_xy=[float(center_xy0[0]), float(center_xy0[1])],
                        inputs_inverted=bool(inputs_inverted),
                        sample_indices=sample_indices.tolist(),
                        sample_files=list(map(str, sample_files)),
                    )
                )
                np.savez_compressed(
                    affine_cache_path,
                    meta_json=np.asarray(meta_json),
                    A_med=A_med.astype(np.float64, copy=False),
                    t_med=t_med.astype(np.float64, copy=False),
                    A_samples=np.stack(affine_As).astype(np.float64, copy=False),
                    t_samples=np.stack(affine_ts).astype(np.float64, copy=False),
                )
            print(f"affine cache: wrote {affine_cache_path}", flush=True)

    affine_hash = _affine_hash(A_med, t_med, inverse=False)
    print(f"affine: using shared median (inputs_inverted={inputs_inverted}, hash={affine_hash})", flush=True)

    with _log_step("affine: write cli_register chromatic file"):
        # cli_register expects a directory with 560to650.txt and/or 560to750.txt.
        # We write the *effective* (A,t) for resampling (i.e., already in the correct direction).
        first_file = selected_files[0]
        img0 = tifffile.imread(first_file)[:-2]
        img0 = img0.reshape(-1, 2, img0.shape[-2], img0.shape[-1])
        height0, width0 = int(img0.shape[-2]), int(img0.shape[-1])
        A_eff, t_eff = _effective_affine_params_for_resample(
            A_med,
            t_med,
            height=height0,
            width=width0,
            inverse=False,
        )
        chromatic_dir = cache_dir / "chromatic"
        chromatic_path = chromatic_dir / f"560to{target}.txt"
        _write_cli_register_chromatic_txt(path=chromatic_path, A=A_eff, t=t_eff)
        print(f"affine: wrote chromatic file: {chromatic_path}", flush=True)

    records: list[dict[str, object]] = []
    for i, file in enumerate(selected_files, start=1):
        with _log_step(f"affine apply {i}/{len(selected_files)}: read {file.name}"):
            img = tifffile.imread(file)[:-2]
            img = img.reshape(-1, 2, img.shape[-2], img.shape[-1])

        ref_raw = img[int(z), 0].astype(np.float32, copy=False)
        mov_raw = img[int(z), 1].astype(np.float32, copy=False)
        with _log_step(f"affine apply {i}/{len(selected_files)}: preprocess"):
            ref = _prep_for_fit(ref_raw)
            mov = _prep_for_fit(mov_raw)
        before = alignment_metrics(ref, mov)

        with _log_step(f"affine apply {i}/{len(selected_files)}: warp(shared affine)"):
            warped_aff = _warp_affine_sitk(mov, ref, A=A_med, t=t_med, inverse=False)
        after_aff = alignment_metrics(ref, warped_aff)

        print(
            f"affine apply {i}/{len(selected_files)}: {file.name} | corrcoef(prep,crop): before={before['corrcoef']:.6f}, affine(shared)={after_aff['corrcoef']:.6f}",
            flush=True,
        )

        records.append(
            dict(
                file=file,
                z=int(z),
                ref_raw=ref_raw,
                mov_raw=mov_raw,
                ref=ref,
                mov=mov,
                warped_aff=warped_aff,
                before=before,
                after_aff=after_aff,
            )
        )

    if not do_nonrigid_refinement:
        return

    nonrigid_gains: list[float] = []
    first = records[0]
    ref0 = first["ref"]
    assert isinstance(ref0, np.ndarray)

    with _log_step("avg TPS: generate control grid"):
        xgrid, ygrid, spacing_x_px, spacing_y_px = generate_control_grid(
            height=int(ref0.shape[0]),
            width=int(ref0.shape[1]),
            control_grid_n=int(nonrigid_tps_grid_n),
            patch_radius_px=int(nonrigid_tps_patch_radius_px),
        )
        _log(
            f"avg TPS: grid_n={int(nonrigid_tps_grid_n)} -> spacing_x_px={int(spacing_x_px)}, spacing_y_px={int(spacing_y_px)}"
        )

    tps_point_method = str(nonrigid_tps_point_method)
    if tps_point_method not in {"sitk", "phase"}:
        raise ValueError(f"nonrigid_tps_point_method must be 'sitk' or 'phase', got {tps_point_method!r}")
    tps_point_method_tag = "sitk_reg_translation_v1" if tps_point_method == "sitk" else "phase_cross_correlation_v1"
    cache_path = cache_dir / (
        "tps_raw_"
        f"{target}_z{int(first['z'])}"
        f"_pool{fit_pool_hash}_n{len(tps_fit_files)}"
        f"_aff{affine_hash}"
        f"_m{tps_point_method_tag}"
        f"_gridn{int(nonrigid_tps_grid_n)}_sx{int(spacing_x_px)}_sy{int(spacing_y_px)}_r{int(nonrigid_tps_patch_radius_px)}"
        f"_u{int(nonrigid_tps_upsample_factor)}_max{float(nonrigid_tps_max_shift_px)}.npz"
    )

    dxs: np.ndarray
    dys: np.ndarray
    used_cache = False
    if use_tps_cache and cache_path.exists():
        _log(f"avg TPS cache: candidate={cache_path}")
        with _log_step("avg TPS cache: load npz"):
            loaded = np.load(cache_path, allow_pickle=False)
            files_arr = loaded["files"]

        meta = json.loads(str(loaded["meta_json"].item()))
        cached_files = [str(x) for x in files_arr.tolist()]
        expected_files = list(map(str, tps_fit_files))
        params_ok = (
            meta.get("target") == target
            and str(meta.get("data_dir")) == str(data_dir)
            and int(meta.get("z")) == int(first["z"])
            and str(meta.get("affine_hash")) == str(affine_hash)
            and str(meta.get("tps_pool_hash")) == str(fit_pool_hash)
            and int(meta.get("tps_n_files")) == int(len(tps_fit_files))
            and str(meta.get("tps_point_method")) == str(tps_point_method_tag)
            and int(meta.get("grid_n")) == int(nonrigid_tps_grid_n)
            and int(meta.get("spacing_x_px")) == int(spacing_x_px)
            and int(meta.get("spacing_y_px")) == int(spacing_y_px)
            and int(meta.get("patch_radius_px")) == int(nonrigid_tps_patch_radius_px)
            and int(meta.get("upsample_factor")) == int(nonrigid_tps_upsample_factor)
            and float(meta.get("max_shift_px")) == float(nonrigid_tps_max_shift_px)
        )
        files_ok = cached_files == expected_files
        grids_ok = np.array_equal(loaded["xgrid"], xgrid) and np.array_equal(loaded["ygrid"], ygrid)
        if params_ok and files_ok and grids_ok:
            dxs = loaded["dxs"]
            dys = loaded["dys"]
            if dxs.shape != (len(tps_fit_files), xgrid.size) or dys.shape != (len(tps_fit_files), xgrid.size):
                raise ValueError(
                    f"Cache shape mismatch: dxs={dxs.shape}, dys={dys.shape}, expected={(len(tps_fit_files), xgrid.size)}"
                )
            used_cache = True
            print(f"avg TPS cache: loaded {cache_path}", flush=True)
        else:
            print(f"avg TPS cache: ignoring {cache_path} (meta/files/grid mismatch)", flush=True)

    if not used_cache:
        dxs = np.full((len(tps_fit_files), xgrid.size), np.nan, dtype=np.float64)
        dys = np.full((len(tps_fit_files), xgrid.size), np.nan, dtype=np.float64)

        for i, filei in enumerate(tps_fit_files, start=1):
            with _log_step(f"avg TPS sampling {i}/{len(tps_fit_files)}: {Path(filei).name}"):
                img = tifffile.imread(filei)[:-2]
                img = img.reshape(-1, 2, img.shape[-2], img.shape[-1])
                ref_raw = img[int(z), 0].astype(np.float32, copy=False)
                mov_raw = img[int(z), 1].astype(np.float32, copy=False)
                refi = _prep_for_fit(ref_raw)
                movi = _prep_for_fit(mov_raw)
                affi = _warp_affine_sitk(movi, refi, A=A_med, t=t_med, inverse=False)

                dx_i, dy_i = estimate_displacements_on_grid(
                    refi,
                    affi,
                    x0=xgrid,
                    y0=ygrid,
                    patch_radius_px=int(nonrigid_tps_patch_radius_px),
                    upsample_factor=int(nonrigid_tps_upsample_factor),
                    max_shift_px=float(nonrigid_tps_max_shift_px),
                    method=tps_point_method,
                )
            dxs[i - 1] = dx_i
            dys[i - 1] = dy_i
            print(f"avg TPS sampling {i}/{len(tps_fit_files)}: {Path(filei).name} done", flush=True)

        if use_tps_cache:
            with _log_step("avg TPS cache: write npz"):
                cache_dir.mkdir(parents=True, exist_ok=True)
                meta_json = json.dumps(
                    dict(
                        target=target,
                        data_dir=str(data_dir),
                        tps_pool_hash=str(fit_pool_hash),
                        tps_n_files=int(len(tps_fit_files)),
                        tps_point_method=str(tps_point_method_tag),
                        cv_train_frac=float(cv_train_frac),
                        cv_seed=int(cv_seed),
                        z=int(first["z"]),
                        affine_hash=str(affine_hash),
                        grid_n=int(nonrigid_tps_grid_n),
                        spacing_x_px=int(spacing_x_px),
                        spacing_y_px=int(spacing_y_px),
                        patch_radius_px=int(nonrigid_tps_patch_radius_px),
                        upsample_factor=int(nonrigid_tps_upsample_factor),
                        max_shift_px=float(nonrigid_tps_max_shift_px),
                    )
                )
                np.savez_compressed(
                    cache_path,
                    meta_json=np.asarray(meta_json),
                    files=np.asarray(list(map(str, tps_fit_files)), dtype=np.str_),
                    xgrid=xgrid,
                    ygrid=ygrid,
                    dxs=dxs,
                    dys=dys,
                )
            print(f"avg TPS cache: wrote {cache_path}", flush=True)

    valid_counts = np.sum(np.isfinite(dxs) & np.isfinite(dys), axis=0)
    keep = valid_counts >= int(nonrigid_tps_min_valid_per_point)
    keep_n = int(np.sum(keep))

    disp_mag = np.hypot(dxs, dys)
    max_disp_per_point = np.nanmax(disp_mag, axis=0)
    max_disp_kept = max_disp_per_point[keep]

    dx0 = np.nanmedian(dxs[:, keep], axis=0)
    dy0 = np.nanmedian(dys[:, keep], axis=0)
    drift_rms = control_point_drift_rms(dxs[:, keep], dys[:, keep], dx0=dx0, dy0=dy0)

    print(
        "avg TPS sampling: "
        f"n_files={len(tps_fit_files)}, grid_points={int(xgrid.size)}, kept_points={keep_n} "
        f"({(100.0 * keep_n / float(xgrid.size)):.1f}%), "
        f"valid_counts(min/median/max)={int(np.min(valid_counts))}/{int(np.median(valid_counts))}/{int(np.max(valid_counts))}",
        flush=True,
    )
    print(
        "avg TPS sampling: "
        f"per-point drift_rms (median/p95)={float(np.median(drift_rms)):.3f}/{float(np.percentile(drift_rms, 95.0)):.3f} px; "
        f"per-point max_disp (median/p95/max)={float(np.median(max_disp_kept)):.3f}/"
        f"{float(np.percentile(max_disp_kept, 95.0)):.3f}/{float(np.max(max_disp_kept)):.3f} px",
        flush=True,
    )

    if print_max_displacement_table:
        print("avg TPS sampling: max displacement per point (kept):", flush=True)
        order = np.argsort(max_disp_kept)[::-1]
        x_keep = xgrid[keep]
        y_keep = ygrid[keep]
        for idx in order:
            x = float(x_keep[idx])
            y = float(y_keep[idx])
            n = int(valid_counts[keep][idx])
            md = float(max_disp_kept[idx])
            dr = float(drift_rms[idx])
            print(f"  (x={x:.0f}, y={y:.0f}) max_disp={md:.3f}px drift_rms={dr:.3f}px (n={n})", flush=True)

    if keep_n < 6:
        raise ValueError(
            f"Too few TPS control points after multi-image filtering: {keep_n} "
            f"(min_valid_per_point={int(nonrigid_tps_min_valid_per_point)})."
        )

    x0 = xgrid[keep]
    y0 = ygrid[keep]

    dx_all = np.nanmedian(dxs, axis=0)
    dy_all = np.nanmedian(dys, axis=0)
    dx_all[~keep] = np.nan
    dy_all[~keep] = np.nan
    w_all = valid_counts.astype(np.float64, copy=False)
    w_all[~keep] = 0.0

    drift_all = np.full_like(xgrid, np.nan, dtype=np.float64)
    drift_all[keep] = drift_rms.astype(np.float64, copy=False)

    beta_dx: np.ndarray | None = None
    beta_dy: np.ndarray | None = None
    if bool(nonrigid_tps_deaffine):
        with _log_step("avg BSpline: remove best-fit affine from control displacements"):
            xk = xgrid[keep]
            yk = ygrid[keep]
            wk = valid_counts[keep].astype(np.float64, copy=False)
            beta_dx = _fit_weighted_plane(xk, yk, dx0, w=wk)
            beta_dy = _fit_weighted_plane(xk, yk, dy0, w=wk)
            pred_all_dx = _eval_plane(beta_dx, xgrid, ygrid)
            pred_all_dy = _eval_plane(beta_dy, xgrid, ygrid)

            dx_all = dx_all - pred_all_dx
            dy_all = dy_all - pred_all_dy

            # Use the residual field for control-point QC visualization too.
            dx0 = dx0 - _eval_plane(beta_dx, xk, yk)
            dy0 = dy0 - _eval_plane(beta_dy, xk, yk)

            print(
                "avg BSpline: removed affine displacement model "
                f"dx=b0+b1*x+b2*y with b={_fmt_mat(beta_dx)}; "
                f"dy=c0+c1*x+c2*y with c={_fmt_mat(beta_dy)}",
                flush=True,
            )

    with _log_step("avg BSpline: build dense displacement field"):
        dx_dense, dy_dense = build_bspline_displacement_field(
            height=int(ref0.shape[0]),
            width=int(ref0.shape[1]),
            xgrid=xgrid,
            ygrid=ygrid,
            dx0=dx_all,
            dy0=dy_all,
            control_grid_spacing_x_px=int(spacing_x_px),
            control_grid_spacing_y_px=int(spacing_y_px),
            weights0=w_all,
            smooth_sigma_px=float(nonrigid_tps_smooth),
        )

    with _log_step("avg BSpline: build dense drift RMS field (QC background)"):
        drift_dense = build_bspline_scalar_field(
            height=int(ref0.shape[0]),
            width=int(ref0.shape[1]),
            xgrid=xgrid,
            ygrid=ygrid,
            v0=drift_all,
            control_grid_spacing_x_px=int(spacing_x_px),
            control_grid_spacing_y_px=int(spacing_y_px),
            weights0=w_all,
            smooth_sigma_px=float(nonrigid_tps_smooth),
        )

    with _log_step("avg BSpline: write displacement field cache"):
        field_cache_path = _bspline_field_cache_path(
            cache_dir=cache_dir,
            target=target,
            z=int(z),
            fit_pool_hash=str(fit_pool_hash),
            tps_n_files=int(len(tps_fit_files)),
            affine_hash=str(affine_hash),
            tps_point_method=str(tps_point_method_tag),
            grid_n=int(nonrigid_tps_grid_n),
            spacing_x_px=int(spacing_x_px),
            spacing_y_px=int(spacing_y_px),
            patch_radius_px=int(nonrigid_tps_patch_radius_px),
            upsample_factor=int(nonrigid_tps_upsample_factor),
            max_shift_px=float(nonrigid_tps_max_shift_px),
            smooth_sigma_px=float(nonrigid_tps_smooth),
            deaffine=bool(nonrigid_tps_deaffine),
        )
        _write_bspline_field_cache(
            path=field_cache_path,
            meta=dict(
                kind="bspline_dense_displacement_dxdy_v1",
                target=target,
                data_dir=str(data_dir),
                z=int(z),
                shape_hw=[int(ref0.shape[0]), int(ref0.shape[1])],
                tps_pool_hash=str(fit_pool_hash),
                tps_n_files=int(len(tps_fit_files)),
                affine_hash=str(affine_hash),
                tps_point_method=str(tps_point_method_tag),
                grid_n=int(nonrigid_tps_grid_n),
                spacing_x_px=int(spacing_x_px),
                spacing_y_px=int(spacing_y_px),
                patch_radius_px=int(nonrigid_tps_patch_radius_px),
                upsample_factor=int(nonrigid_tps_upsample_factor),
                max_shift_px=float(nonrigid_tps_max_shift_px),
                smooth_sigma_px=float(nonrigid_tps_smooth),
                deaffine=bool(nonrigid_tps_deaffine),
            ),
            dx_dense=dx_dense,
            dy_dense=dy_dense,
        )
        print(f"avg BSpline: wrote dense field cache: {field_cache_path}", flush=True)

    with _log_step("avg BSpline: write cli_register chromatic field"):
        chromatic_field_path = chromatic_dir / f"560to{target}_field.npz"
        field_meta: dict[str, object] = {
            "kind": "chromatic_bspline_dense_displacement_dxdy_v1",
            "target": str(target),
            "data_dir": str(data_dir),
            "z": int(z),
            "shape_hw": [int(ref0.shape[0]), int(ref0.shape[1])],
            "selection": {
                "max_files": int(max_files),
                "cv_train_frac": float(cv_train_frac),
                "cv_seed": int(cv_seed),
                "cv_eval_n": int(cv_eval_n),
            },
            "prep": {
                "highpass_sigma_px": 3.0,
                "norm_percentiles": [1.0, 99.0],
            },
            "affine": {
                "hash": str(affine_hash),
                "inputs_inverted": bool(inputs_inverted),
                "sample_n": int(affine_sample_n),
                "sample_seed": int(affine_sample_seed),
                "avg_method": str(affine_avg_method),
                "A_med": np.asarray(A_med, dtype=np.float64).tolist(),
                "t_med": np.asarray(t_med, dtype=np.float64).reshape(2).tolist(),
                "A_effective_for_cli_register": np.asarray(A_eff, dtype=np.float64).tolist(),
                "t_effective_for_cli_register": np.asarray(t_eff, dtype=np.float64).reshape(2).tolist(),
            },
	            "tps": {
	                "pool_hash": str(fit_pool_hash),
	                "n_files": int(len(tps_fit_files)),
	                "point_method": str(tps_point_method_tag),
	                "grid_n": int(nonrigid_tps_grid_n),
	                "spacing_x_px": int(spacing_x_px),
	                "spacing_y_px": int(spacing_y_px),
	                "patch_radius_px": int(nonrigid_tps_patch_radius_px),
	                "edge_offset_px": 5,
	                "upsample_factor": int(nonrigid_tps_upsample_factor),
	                "max_shift_px": float(nonrigid_tps_max_shift_px),
                "min_valid_per_point": int(nonrigid_tps_min_valid_per_point),
                "smooth_sigma_px": float(nonrigid_tps_smooth),
                "deaffine": bool(nonrigid_tps_deaffine),
                "deaffine_plane": None
                if not bool(nonrigid_tps_deaffine)
                else {
                    "beta_dx": np.asarray(beta_dx, dtype=np.float64).tolist() if beta_dx is not None else None,
                    "beta_dy": np.asarray(beta_dy, dtype=np.float64).tolist() if beta_dy is not None else None,
                },
                "lattice_pad_steps": 4,
            },
        }
        _write_cli_register_chromatic_field_npz(
            path=chromatic_field_path,
            meta=field_meta,
            dx_dense=dx_dense,
            dy_dense=dy_dense,
            xgrid=xgrid,
            ygrid=ygrid,
            dx_control=dx_all,
            dy_control=dy_all,
            keep_mask=keep,
            valid_counts=valid_counts,
            files=tps_fit_files,
        )
        print(f"avg BSpline: wrote chromatic field file: {chromatic_field_path}", flush=True)

    for rec in records:
        file = rec["file"]
        ref = rec["ref"]
        warped_aff = rec["warped_aff"]
        before = rec["before"]
        after_aff = rec["after_aff"]
        assert isinstance(file, Path)
        assert isinstance(ref, np.ndarray)
        assert isinstance(warped_aff, np.ndarray)
        assert isinstance(before, dict)
        assert isinstance(after_aff, dict)

        with _log_step(f"apply avg BSpline field: {file.name}"):
            warped_nr = apply_displacement_field(
                warped_aff,
                dx_dense,
                dy_dense,
                warp_block_rows=int(nonrigid_tps_warp_block_rows),
                mode=str(nonrigid_tps_warp_mode),
                cval=float(nonrigid_tps_warp_cval),
                clamp=bool(nonrigid_tps_warp_clamp),
            )
        after_nr = alignment_metrics(ref, warped_nr)
        gain_over_affine = after_nr["corrcoef"] - float(after_aff["corrcoef"])
        gain_total = after_nr["corrcoef"] - float(before["corrcoef"])
        nonrigid_gains.append(float(gain_over_affine))

        print(
            f"{file.name} | corrcoef(prep,crop): nonrigid(avgBSpline)={after_nr['corrcoef']:.6f} "
            f"(gain_over_affine={gain_over_affine:+.6f}, gain_total={gain_total:+.6f})",
            flush=True,
        )
        rec["warped_nr"] = warped_nr

    if save_overlay_qc:
        with _log_step("QC overlay: write first-file panel"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            y0c, x0c, size = 32, 32, 512
            sl = np.s_[y0c : y0c + size, x0c : x0c + size]
            rec0 = records[0]
            file0 = rec0["file"]
            ref = rec0["ref"]
            mov = rec0["mov"]
            warped_aff0 = rec0["warped_aff"]
            warped_nr0 = rec0.get("warped_nr")
            assert isinstance(file0, Path)
            assert isinstance(ref, np.ndarray)
            assert isinstance(mov, np.ndarray)
            assert isinstance(warped_aff0, np.ndarray)

            ref_v = _prep_for_viz(ref[sl])
            mov_v = _prep_for_viz(mov[sl])
            aff_v = _prep_for_viz(warped_aff0[sl])
            fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4), dpi=200)
            for ax in axs:
                ax.axis("off")
            axs[0].imshow(_overlay_rg01(ref_v, mov_v))
            axs[0].set_title("Before (ref=R, moving=G)")
            axs[1].imshow(_overlay_rg01(ref_v, aff_v))
            axs[1].set_title("After affine (ref=R, warped=G)")
            if isinstance(warped_nr0, np.ndarray):
                nr_v = _prep_for_viz(warped_nr0[sl])
                axs[2].imshow(_overlay_rg01(ref_v, nr_v))
                axs[2].set_title("After avg BSpline (ref=R, warped=G)")
            else:
                axs[2].imshow(_overlay_rg01(ref_v, aff_v))
                axs[2].set_title("After affine (ref=R, warped=G)")
            out_path = qc_dir / f"{file0.stem}_z{int(z)}_qc.png"
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            print(f"{file0.name} | wrote QC overlay: {out_path}", flush=True)

    if save_quiver_qc:
        with _log_step("QC quiver: write whole-image deformation field (background=drift RMS)"):
            qc_dir.mkdir(parents=True, exist_ok=True)
            stride = int(quiver_stride)
            if stride <= 0:
                raise ValueError("quiver_stride must be > 0")

            height, width = ref0.shape
            ys = np.arange(0, height, stride, dtype=np.int32)
            xs = np.arange(0, width, stride, dtype=np.int32)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            u = (dx_dense[yy, xx] * float(quiver_magnify)).astype(np.float32, copy=False)
            v = (dy_dense[yy, xx] * float(quiver_magnify)).astype(np.float32, copy=False)

            bg = drift_dense
            vmax = float(np.percentile(bg, 99.0)) if np.isfinite(bg).any() else 1.0
            vmax = max(vmax, 1.0e-6)
            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            im = ax.imshow(bg, cmap="magma", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="control-point drift RMS (px)")

            ax.quiver(
                xx,
                yy,
                u,
                v,
                color="cyan",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
            )
            ax.quiver(
                x0,
                y0,
                (dx0 * float(quiver_magnify)).astype(np.float32, copy=False),
                (dy0 * float(quiver_magnify)).astype(np.float32, copy=False),
                color="magenta",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
            )
            ax.set_title(
                "avg BSpline field (cyan=dense, magenta=control points)\n"
                f"bg=drift_rms px, stride={stride}px, magnify={quiver_magnify:g}x"
            )
            ax.axis("off")
            out_path = qc_dir / (
                "avg_bspline_quiver_rms_"
                f"{target}_z{int(z)}"
                f"_gridn{int(nonrigid_tps_grid_n)}_sx{int(spacing_x_px)}_sy{int(spacing_y_px)}"
                f"_r{int(nonrigid_tps_patch_radius_px)}_u{int(nonrigid_tps_upsample_factor)}"
                f"_max{float(nonrigid_tps_max_shift_px):g}_smooth{float(nonrigid_tps_smooth):g}"
                f"_m{tps_point_method_tag}_deaff{int(bool(nonrigid_tps_deaffine))}"
                f"_stride{int(stride)}_mag{float(quiver_magnify):g}"
                ".png"
            )
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            print(f"avg BSpline | wrote QC quiver: {out_path}", flush=True)

    if nonrigid_gains:
        print(
            "NONRIGID summary: "
            f"mean_gain_over_affine={float(np.mean(nonrigid_gains)):+.6f}, "
            f"median_gain_over_affine={float(np.median(nonrigid_gains)):+.6f}",
            flush=True,
        )

    if tps_val_files:
        val_files = tps_val_files
        if cv_eval_n > 0 and cv_eval_n < len(val_files):
            rng = np.random.default_rng(int(cv_seed) + 1)
            take = np.sort(rng.choice(len(val_files), size=int(cv_eval_n), replace=False))
            val_files = [val_files[int(i)] for i in take.tolist()]

        cv_gains: list[float] = []
        for file in val_files:
            with _log_step(f"CV eval: read {file.name}"):
                img = tifffile.imread(file)[:-2]
                img = img.reshape(-1, 2, img.shape[-2], img.shape[-1])

            ref_raw = img[int(z), 0].astype(np.float32, copy=False)
            mov_raw = img[int(z), 1].astype(np.float32, copy=False)
            ref = _prep_for_fit(ref_raw)
            mov = _prep_for_fit(mov_raw)
            warped_aff = _warp_affine_sitk(mov, ref, A=A_med, t=t_med, inverse=False)
            warped_nr = apply_displacement_field(
                warped_aff,
                dx_dense,
                dy_dense,
                warp_block_rows=int(nonrigid_tps_warp_block_rows),
                mode=str(nonrigid_tps_warp_mode),
                cval=float(nonrigid_tps_warp_cval),
                clamp=bool(nonrigid_tps_warp_clamp),
            )
            after_aff = alignment_metrics(ref, warped_aff)["corrcoef"]
            after_nr = alignment_metrics(ref, warped_nr)["corrcoef"]
            cv_gains.append(float(after_nr - after_aff))

        if cv_gains:
            print(
                "CV NONRIGID summary: "
                f"n_val={len(val_files)} mean_gain_over_affine={float(np.mean(cv_gains)):+.6f} "
                f"median_gain_over_affine={float(np.median(cv_gains)):+.6f}",
                flush=True,
            )


def main() -> None:
    import rich_click as click

    click.rich_click.USE_RICH_MARKUP = True
    click.rich_click.SHOW_METAVARS_COLUMN = True
    click.rich_click.MAX_WIDTH = 120

    @click.command()
    @click.option("--target", type=str, default="750", show_default=True, help="Target channel label (e.g. 750).")
    @click.option(
        "--data-dir",
        type=click.Path(path_type=Path, dir_okay=True, file_okay=False, exists=True),
        default=None,
        help="Directory containing the *.tif files (defaults to the hardcoded /working path for the target).",
    )
    @click.option(
        "--max-files",
        type=int,
        default=0,
        show_default=True,
        help=(
            "Number of files to APPLY the shared correction to / report metrics for. "
            "Use 0 to apply to all files (TPS fitting uses all images in data-dir)."
        ),
    )
    @click.option("--file-index", type=int, default=1, show_default=True, help="Starting index into sorted file list.")
    @click.option(
        "--z",
        "zs",
        multiple=True,
        type=int,
        default=(4,),
        show_default=True,
        help="Z plane(s) to use. Pass multiple times (e.g. `--z 1 --z 6`).",
    )
    @click.option(
        "--affine-sample-n",
        type=int,
        default=10,
        show_default=True,
        help="Number of randomly-sampled files used to estimate the shared affine.",
    )
    @click.option(
        "--affine-sample-seed",
        type=int,
        default=0,
        show_default=True,
        help="RNG seed for selecting affine-sample files.",
    )
    @click.option(
        "--cv-train-frac",
        type=float,
        default=1.0,
        show_default=True,
        help="If <1, fit nonrigid field using this fraction of bead files and report holdout (val) metric.",
    )
    @click.option("--cv-seed", type=int, default=0, show_default=True, help="RNG seed for the CV train/val split.")
    @click.option(
        "--cv-eval-n",
        type=int,
        default=0,
        show_default=True,
        help="If >0, evaluate CV metric on a random subset of this many val files.",
    )
    @click.option("--nonrigid/--no-nonrigid", default=True, show_default=True, help="Enable shared TPS nonrigid.")
    @click.option(
        "--tps-grid",
        "nonrigid_tps_grid_n",
        type=int,
        default=6,
        show_default=True,
        help="Number of control points per axis (NxN).",
    )
    @click.option(
        "--tps-radius",
        "nonrigid_tps_patch_radius_px",
        type=int,
        default=64,
        show_default=True,
        help="Patch radius in px (patch size is 2*radius+1).",
    )
    @click.option(
        "--tps-upsample",
        "nonrigid_tps_upsample_factor",
        type=int,
        default=10,
        show_default=True,
        help="Patch shift upsample/iteration parameter (SITK scales n_iter; phase uses it as upsample_factor).",
    )
    @click.option(
        "--tps-point-method",
        "nonrigid_tps_point_method",
        type=click.Choice(["sitk", "phase"], case_sensitive=False),
        default="sitk",
        show_default=True,
        help="Control-point (patch) shift estimator: 'sitk' translation optimizer or 'phase' cross-correlation.",
    )
    @click.option("--tps-max-shift", "nonrigid_tps_max_shift_px", type=float, default=10.0, show_default=True)
    @click.option(
        "--tps-smooth",
        "nonrigid_tps_smooth",
        type=float,
        default=0.0,
        show_default=True,
        help=(
            "Gaussian smoothing sigma in px applied to the *coarse displacement lattice* before B-spline interpolation. "
            "Effective sigma in lattice steps is approximately (tps_smooth / step_px); "
            "a good sweep range is ~0.5–2× step_px."
        ),
    )
    @click.option(
        "--tps-deaffine/--no-tps-deaffine",
        "nonrigid_tps_deaffine",
        default=False,
        show_default=True,
        help="Subtract best-fit affine (translation + linear ramp) from control displacements before building the spline.",
    )
    @click.option(
        "--tps-min-valid",
        "nonrigid_tps_min_valid_per_point",
        type=int,
        default=5,
        show_default=True,
        help="Minimum number of per-image valid shifts required to keep a control point.",
    )
    @click.option("--tps-warp-block-rows", "nonrigid_tps_warp_block_rows", type=int, default=256, show_default=True)
    @click.option(
        "--tps-warp-mode",
        "nonrigid_tps_warp_mode",
        type=click.Choice(["constant", "nearest", "reflect", "mirror", "wrap"], case_sensitive=False),
        default="constant",
        show_default=True,
        help="Out-of-bounds handling for TPS warping (scipy.ndimage.map_coordinates).",
    )
    @click.option(
        "--tps-warp-cval",
        "nonrigid_tps_warp_cval",
        type=float,
        default=0.0,
        show_default=True,
        help="Constant fill value when --tps-warp-mode=constant.",
    )
    @click.option(
        "--tps-warp-clamp/--no-tps-warp-clamp",
        "nonrigid_tps_warp_clamp",
        default=False,
        show_default=True,
        help="Clamp sample coordinates into bounds before sampling (prevents black borders, can hide extrapolation).",
    )
    @click.option("--qc-dir", type=click.Path(path_type=Path), default=None, show_default=True)
    @click.option("--overlay/--no-overlay", default=True, show_default=True, help="Write the first-file overlay panel.")
    @click.option("--quiver/--no-quiver", default=True, show_default=True, help="Write the quiver deformation field.")
    @click.option("--quiver-stride", type=int, default=64, show_default=True)
    @click.option("--quiver-magnify", type=float, default=200.0, show_default=True)
    @click.option(
        "--cache-dir",
        type=click.Path(path_type=Path),
        default=Path("/working/20260103_chromatic/register_genmaster_cache"),
        show_default=True,
        help="Directory to store affine/TPS caches.",
    )
    @click.option("--affine-cache/--no-affine-cache", default=True, show_default=True)
    @click.option("--tps-cache/--no-tps-cache", default=True, show_default=True)
    @click.option("--print-max-displacement-table", is_flag=True, default=False, show_default=True)
    def _cli(**kwargs) -> None:
        zs = tuple(int(z) for z in kwargs["zs"])
        if not zs:
            raise ValueError("At least one --z must be provided.")
        for z in zs:
            run_register_genmaster(
                target=kwargs["target"],
                data_dir=kwargs["data_dir"],
                max_files=kwargs["max_files"],
                file_index=kwargs["file_index"],
                z=int(z),
                affine_sample_n=kwargs["affine_sample_n"],
                affine_sample_seed=kwargs["affine_sample_seed"],
                cv_train_frac=kwargs["cv_train_frac"],
                cv_seed=kwargs["cv_seed"],
                cv_eval_n=kwargs["cv_eval_n"],
                do_nonrigid_refinement=kwargs["nonrigid"],
                nonrigid_tps_grid_n=kwargs["nonrigid_tps_grid_n"],
                nonrigid_tps_patch_radius_px=kwargs["nonrigid_tps_patch_radius_px"],
                nonrigid_tps_upsample_factor=kwargs["nonrigid_tps_upsample_factor"],
                nonrigid_tps_max_shift_px=kwargs["nonrigid_tps_max_shift_px"],
                nonrigid_tps_point_method=kwargs["nonrigid_tps_point_method"],
                nonrigid_tps_smooth=kwargs["nonrigid_tps_smooth"],
                nonrigid_tps_deaffine=kwargs["nonrigid_tps_deaffine"],
                nonrigid_tps_min_valid_per_point=kwargs["nonrigid_tps_min_valid_per_point"],
                nonrigid_tps_warp_block_rows=kwargs["nonrigid_tps_warp_block_rows"],
                nonrigid_tps_warp_mode=kwargs["nonrigid_tps_warp_mode"],
                nonrigid_tps_warp_cval=kwargs["nonrigid_tps_warp_cval"],
                nonrigid_tps_warp_clamp=kwargs["nonrigid_tps_warp_clamp"],
                save_overlay_qc=kwargs["overlay"],
                qc_dir=kwargs["qc_dir"],
                use_tps_cache=kwargs["tps_cache"],
                cache_dir=kwargs["cache_dir"],
                use_affine_cache=kwargs["affine_cache"],
                save_quiver_qc=kwargs["quiver"],
                quiver_stride=kwargs["quiver_stride"],
                quiver_magnify=kwargs["quiver_magnify"],
                print_max_displacement_table=kwargs["print_max_displacement_table"],
            )

    _cli()


if __name__ == "__main__":
    main()
