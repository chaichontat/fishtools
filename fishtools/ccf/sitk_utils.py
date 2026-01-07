from __future__ import annotations

import numpy as np
import SimpleITK as sitk

UM_TO_MM = 1e-3


def normalize_robust(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Normalize a numeric array to [0,1] using robust percentiles."""

    a = arr.astype(np.float32)
    nz = a[np.isfinite(a)]
    if nz.size == 0:
        return a
    p_lo, p_hi = np.percentile(nz, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros_like(a)
    a = np.clip(a, p_lo, p_hi)
    a = (a - p_lo) / (p_hi - p_lo)
    return a


def sitk_from_numpy_2d(arr_yx: np.ndarray, spacing_um: float) -> sitk.Image:
    """Create a 2D SimpleITK image with spacing expressed in mm (ANTs/NIfTI convention)."""

    img = sitk.GetImageFromArray(arr_yx.astype(np.float32))
    spacing_mm = float(spacing_um) * UM_TO_MM
    img.SetSpacing((spacing_mm, spacing_mm))
    img.SetOrigin((0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 1.0))
    return img


def resample_sitk_to_spacing(img: sitk.Image, target_spacing_um: float, interp: int) -> sitk.Image:
    target_spacing_mm = float(target_spacing_um) * UM_TO_MM
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()

    out_size = [int(np.round(in_size[d] * in_spacing[d] / target_spacing_mm)) for d in range(2)]
    out_size = [max(1, s) for s in out_size]

    return sitk.Resample(
        img,
        out_size,
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        (target_spacing_mm, target_spacing_mm),
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )


def largest_cc(mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.Cast(relabel == 1, sitk.sitkUInt8)


def make_moving_mask_sitk(moving_img: sitk.Image) -> sitk.Image:
    """Compute a conservative binary mask for a moving image (partial tissue)."""

    sm = sitk.SmoothingRecursiveGaussian(sitk.Cast(moving_img, sitk.sitkFloat32), sigma=0.05)  # 50 µm
    m = sitk.OtsuThreshold(sm, 0, 1)
    frac = float(sitk.GetArrayFromImage(m).mean())
    if frac > 0.5:
        m = sitk.InvertIntensity(m, maximum=1)
    m = sitk.BinaryMorphologicalClosing(m, [3, 3])
    m = sitk.BinaryFillhole(m)
    m = largest_cc(sitk.Cast(m, sitk.sitkUInt8))
    return m


def erode_by_um(mask: sitk.Image, guard_um: float) -> sitk.Image:
    sp = float(mask.GetSpacing()[0])  # mm
    guard_mm = float(guard_um) * UM_TO_MM
    rad_px = int(np.round(guard_mm / sp))
    rad_px = max(0, rad_px)
    if rad_px == 0:
        return sitk.Cast(mask, sitk.sitkUInt8)
    return sitk.BinaryErode(sitk.Cast(mask, sitk.sitkUInt8), [rad_px, rad_px])


def gradmag_feature(img: sitk.Image, sigma_um: float) -> sitk.Image:
    sigma_mm = float(sigma_um) * UM_TO_MM
    g = sitk.GradientMagnitudeRecursiveGaussian(sitk.Cast(img, sitk.sitkFloat32), sigma=sigma_mm)

    a = sitk.GetArrayFromImage(g)
    a = normalize_robust(a, 1, 99)
    out = sitk.GetImageFromArray(a.astype(np.float32))
    out.CopyInformation(img)
    return out


def signed_distance(mask: sitk.Image) -> sitk.Image:
    dt = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    a = sitk.GetArrayFromImage(dt)
    p1, p99 = np.percentile(a, (1, 99))
    a = np.clip(a, p1, p99)
    a = a / (np.max(np.abs(a)) + 1e-8)
    out = sitk.GetImageFromArray(a.astype(np.float32))
    out.CopyInformation(mask)
    return out


def pixels_to_physical_um(points_xy_px: list[tuple[float, float]], voxel_size_um: float) -> list[tuple[float, float]]:
    """Convert pixel coordinates to physical coordinates in µm."""

    v = float(voxel_size_um)
    return [(float(x) * v, float(y) * v) for x, y in points_xy_px]


def compute_similarity2d_from_landmarks(
    fixed_points_xy: list[tuple[float, float]],
    moving_points_xy: list[tuple[float, float]],
) -> sitk.Similarity2DTransform:
    """Compute a Similarity2D transform from corresponding landmark pairs.

    Returns a transform T such that `T(fixed_point) ≈ moving_point` (fixed→moving point mapping).
    """

    if len(fixed_points_xy) < 3:
        raise ValueError("Need at least 3 landmarks.")
    if len(fixed_points_xy) != len(moving_points_xy):
        raise ValueError("Landmark counts must match.")

    fixed_arr = np.asarray(fixed_points_xy, dtype=np.float64)
    moving_arr = np.asarray(moving_points_xy, dtype=np.float64)

    fixed_centroid = fixed_arr.mean(axis=0)
    moving_centroid = moving_arr.mean(axis=0)

    fixed_centered = fixed_arr - fixed_centroid
    moving_centered = moving_arr - moving_centroid

    fixed_var = float(np.sum(fixed_centered**2))
    if fixed_var < 1e-10:
        raise ValueError("Landmarks are degenerate (zero variance).")

    if np.linalg.matrix_rank(fixed_centered, tol=1e-6) < 2:
        raise ValueError("Landmarks are collinear.")

    h = fixed_centered.T @ moving_centered
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if float(np.linalg.det(r)) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    scale = float(np.sum(s) / fixed_var)
    if scale <= 0:
        raise ValueError(f"Invalid scale: {scale}")

    angle = float(np.arctan2(r[1, 0], r[0, 0]))

    transform = sitk.Similarity2DTransform()
    cx, cy = fixed_centroid.tolist()
    transform.SetCenter((float(cx), float(cy)))
    transform.SetAngle(angle)
    transform.SetScale(scale)
    dx, dy = (moving_centroid - fixed_centroid).tolist()
    transform.SetTranslation((float(dx), float(dy)))
    return transform
