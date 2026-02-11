from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class CoronalMidlineColumns:
    """Per-slice midline + vent/pia endpoints.

    Coordinates are in the coronal plane (y/x) for a fixed slice_i:
    - (y, x) is a point on the midline curve (u≈0.5 sheet intersection)
    - (vent_y, vent_x) and (pia_y, pia_x) are boundary intersection points along a local normal

    Given r01 in [0,1] (0=ventricular/inner, 1=pial), the point in voxel coordinates is:
        p_yx = vent_yx + r01 * (pia_yx - vent_yx)
    and full ijk is (slice_i, p_y, p_x).
    """

    slice_i: int
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    vent_y: np.ndarray
    vent_x: np.ndarray
    pia_y: np.ndarray
    pia_x: np.ndarray
    thickness_um: np.ndarray


def load_coronal_midline_columns(csv_path: Path) -> dict[int, CoronalMidlineColumns]:
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if table.ndim == 1:
        table = table[None, :]
    if table.shape[1] != 9:
        raise ValueError(f"Unexpected columns in {csv_path} (shape={table.shape})")

    slice_i = table[:, 0].astype(np.int32)
    t = table[:, 1].astype(np.float64)
    y = table[:, 2].astype(np.float64)
    x = table[:, 3].astype(np.float64)
    vent_y = table[:, 4].astype(np.float64)
    vent_x = table[:, 5].astype(np.float64)
    pia_y = table[:, 6].astype(np.float64)
    pia_x = table[:, 7].astype(np.float64)
    thickness_um = table[:, 8].astype(np.float64)

    out: dict[int, CoronalMidlineColumns] = {}
    for s in np.unique(slice_i).tolist():
        keep = slice_i == int(s)
        out[int(s)] = CoronalMidlineColumns(
            slice_i=int(s),
            t=t[keep],
            y=y[keep],
            x=x[keep],
            vent_y=vent_y[keep],
            vent_x=vent_x[keep],
            pia_y=pia_y[keep],
            pia_x=pia_x[keep],
            thickness_um=thickness_um[keep],
        )
    return out


def _interp(t_grid: np.ndarray, values: np.ndarray, t: float) -> float:
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite.")
    return float(np.interp(t, t_grid, values))


@dataclass(frozen=True)
class SagittalMidlineColumns:
    """Same as CoronalMidlineColumns but for sagittal slices k (axis=2).

    Plane coordinates are (y=i, x=j), with k fixed.
    """

    slice_k: int
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    vent_y: np.ndarray
    vent_x: np.ndarray
    pia_y: np.ndarray
    pia_x: np.ndarray
    thickness_um: np.ndarray


def load_sagittal_midline_columns(csv_path: Path) -> dict[int, SagittalMidlineColumns]:
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if table.ndim == 1:
        table = table[None, :]
    if table.shape[1] != 9:
        raise ValueError(f"Unexpected columns in {csv_path} (shape={table.shape})")

    slice_k = table[:, 0].astype(np.int32)
    t = table[:, 1].astype(np.float64)
    y = table[:, 2].astype(np.float64)
    x = table[:, 3].astype(np.float64)
    vent_y = table[:, 4].astype(np.float64)
    vent_x = table[:, 5].astype(np.float64)
    pia_y = table[:, 6].astype(np.float64)
    pia_x = table[:, 7].astype(np.float64)
    thickness_um = table[:, 8].astype(np.float64)

    out: dict[int, SagittalMidlineColumns] = {}
    for s in np.unique(slice_k).tolist():
        keep = slice_k == int(s)
        out[int(s)] = SagittalMidlineColumns(
            slice_k=int(s),
            t=t[keep],
            y=y[keep],
            x=x[keep],
            vent_y=vent_y[keep],
            vent_x=vent_x[keep],
            pia_y=pia_y[keep],
            pia_x=pia_x[keep],
            thickness_um=thickness_um[keep],
        )
    return out


def _load_resolution_ds_ijk_um(outdir: Path) -> tuple[float, float, float] | None:
    path = outdir / "resolution_ds_ijk_um.npy"
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.shape != (3,):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class EdtCrop:
    origin_ijk: tuple[int, int, int]
    d_pial_um: np.ndarray
    d_inner_um: np.ndarray


def _load_halfway_distance_fields(outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    pial_path = outdir / "halfway_d_pial_um_crop.npy"
    inner_path = outdir / "halfway_d_inner_um_crop.npy"
    has_direct = pial_path.exists() or inner_path.exists()
    if has_direct:
        if not pial_path.exists() or not inner_path.exists():
            raise FileNotFoundError(
                "Expected both halfway_d_pial_um_crop.npy and halfway_d_inner_um_crop.npy when using direct EDT files."
            )
        d_pial = np.load(pial_path).astype(np.float32, copy=False)
        d_inner = np.load(inner_path).astype(np.float32, copy=False)
        if d_pial.shape != d_inner.shape:
            raise ValueError("halfway_d_pial_um_crop.npy and halfway_d_inner_um_crop.npy must have the same shape.")
        return d_pial, d_inner

    r_um_path = outdir / "halfway_r_um_crop.npy"
    thickness_path = outdir / "halfway_thickness_um_crop.npy"
    has_reconstruct = r_um_path.exists() or thickness_path.exists()
    if has_reconstruct:
        if not r_um_path.exists() or not thickness_path.exists():
            raise FileNotFoundError(
                "Expected both halfway_r_um_crop.npy and halfway_thickness_um_crop.npy to reconstruct EDT distances."
            )
        r_um = np.load(r_um_path).astype(np.float32, copy=False)
        thickness_um = np.load(thickness_path).astype(np.float32, copy=False)
        if r_um.shape != thickness_um.shape:
            raise ValueError("halfway_r_um_crop.npy and halfway_thickness_um_crop.npy must have the same shape.")
        d_inner = 0.5 * (thickness_um + r_um)
        d_pial = 0.5 * (thickness_um - r_um)
        finite = np.isfinite(d_inner) & np.isfinite(d_pial)
        d_inner = np.where(finite, np.maximum(d_inner, 0.0), np.nan).astype(np.float32, copy=False)
        d_pial = np.where(finite, np.maximum(d_pial, 0.0), np.nan).astype(np.float32, copy=False)
        return d_pial, d_inner

    raise FileNotFoundError(
        "Missing EDT crop fields. Need either "
        "{halfway_d_pial_um_crop.npy + halfway_d_inner_um_crop.npy} or "
        "{halfway_r_um_crop.npy + halfway_thickness_um_crop.npy} in outdir."
    )


def load_halfway_edt_crop(outdir: Path) -> EdtCrop:
    origin = np.load(outdir / "halfway_crop_origin_ijk.npy").astype(np.int64, copy=False)
    if origin.shape != (3,):
        raise ValueError("halfway_crop_origin_ijk.npy must be shape (3,).")
    d_pial, d_inner = _load_halfway_distance_fields(outdir)
    return EdtCrop(origin_ijk=(int(origin[0]), int(origin[1]), int(origin[2])), d_pial_um=d_pial, d_inner_um=d_inner)


def _nearest_key(keys: list[int], target: int) -> int:
    if not keys:
        raise ValueError("keys is empty.")
    target = int(target)
    arr = np.asarray(keys, dtype=np.int32)
    return int(arr[np.argmin(np.abs(arr - target))])


def _r01_at_ijk(edt: EdtCrop, ijk: tuple[float, float, float]) -> float:
    i0, j0, k0 = edt.origin_ijk
    ic = float(ijk[0]) - float(i0)
    jc = float(ijk[1]) - float(j0)
    kc = float(ijk[2]) - float(k0)
    si, sj, sk = edt.d_pial_um.shape
    if ic < -0.5 or jc < -0.5 or kc < -0.5 or ic > float(si - 0.5) or jc > float(sj - 0.5) or kc > float(sk - 0.5):
        raise ValueError(f"Point {ijk!r} falls outside EDT crop (origin={edt.origin_ijk}, shape={edt.d_pial_um.shape}).")

    coords = np.asarray([[ic], [jc], [kc]], dtype=np.float64)
    d_pial = float(map_coordinates(edt.d_pial_um, coords, order=1, mode="nearest")[0])
    d_inner = float(map_coordinates(edt.d_inner_um, coords, order=1, mode="nearest")[0])
    den = d_pial + d_inner
    if not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(d_inner / den)


def invert_r01_along_segment(
    edt: EdtCrop,
    *,
    vent_ijk: tuple[float, float, float],
    pia_ijk: tuple[float, float, float],
    r01_target: float,
    max_iters: int = 60,
    tol_r01: float = 2.0e-4,
) -> tuple[float, float, float]:
    r01_target = float(r01_target)
    if not (0.0 <= r01_target <= 1.0):
        raise ValueError("r01_target must be in [0,1].")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")
    if tol_r01 <= 0:
        raise ValueError("tol_r01 must be > 0.")

    v = np.asarray(vent_ijk, dtype=np.float64)
    p = np.asarray(pia_ijk, dtype=np.float64)
    if v.shape != (3,) or p.shape != (3,):
        raise ValueError("vent_ijk and pia_ijk must be 3-tuples.")

    r_v = _r01_at_ijk(edt, (float(v[0]), float(v[1]), float(v[2])))
    r_p = _r01_at_ijk(edt, (float(p[0]), float(p[1]), float(p[2])))
    if not np.isfinite(r_v) or not np.isfinite(r_p):
        raise ValueError("EDT r01 was non-finite at vent/pia endpoints (check crop coverage).")

    # Ensure increasing r01 from vent -> pia.
    if r_v > r_p:
        v, p = p, v
        r_v, r_p = r_p, r_v

    if r01_target <= r_v:
        return (float(v[0]), float(v[1]), float(v[2]))
    if r01_target >= r_p:
        return (float(p[0]), float(p[1]), float(p[2]))

    lo = 0.0
    hi = 1.0
    r_lo = r_v
    r_hi = r_p
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        q = v + mid * (p - v)
        r_mid = _r01_at_ijk(edt, (float(q[0]), float(q[1]), float(q[2])))
        if not np.isfinite(r_mid):
            raise ValueError("EDT r01 became non-finite during inversion.")
        if abs(r_mid - r01_target) <= tol_r01:
            return (float(q[0]), float(q[1]), float(q[2]))
        if r_mid < r01_target:
            lo, r_lo = mid, r_mid
        else:
            hi, r_hi = mid, r_mid
        if abs(r_hi - r_lo) <= tol_r01:
            break

    q = v + (0.5 * (lo + hi)) * (p - v)
    return (float(q[0]), float(q[1]), float(q[2]))


@dataclass(frozen=True)
class InvertedAxisCoordinate:
    axis: Literal["coronal", "sagittal"]
    slice_index: int
    t: float
    r01: float
    residual_vox: float


def _optimize_t_fixed_r01_coronal(
    *,
    c: CoronalMidlineColumns,
    ijk_target: tuple[float, float, float],
    r01_target: float,
    edt: EdtCrop,
) -> tuple[float, float]:
    """Find t minimizing ||x(t, r01_target) - ijk_target|| in the coronal plane."""
    target_jk = np.asarray([float(ijk_target[1]), float(ijk_target[2])], dtype=np.float64)

    def _eval_t(t: float) -> tuple[float, float]:
        vent_y = float(_interp(c.t, c.vent_y, t))
        vent_x = float(_interp(c.t, c.vent_x, t))
        pia_y = float(_interp(c.t, c.pia_y, t))
        pia_x = float(_interp(c.t, c.pia_x, t))
        if not np.isfinite([vent_y, vent_x, pia_y, pia_x]).all():
            return float("nan"), float("inf")
        try:
            ijk = invert_r01_along_segment(
                edt,
                vent_ijk=(float(c.slice_i), vent_y, vent_x),
                pia_ijk=(float(c.slice_i), pia_y, pia_x),
                r01_target=float(r01_target),
                max_iters=32,
                tol_r01=5.0e-4,
            )
        except ValueError:
            return float("nan"), float("inf")
        jk = np.asarray([float(ijk[1]), float(ijk[2])], dtype=np.float64)
        d2 = float(np.sum((jk - target_jk) ** 2))
        return float(t), d2

    best_t = 0.5
    best_d2 = float("inf")
    lo = 0.0
    hi = 1.0
    for n_samples in (161, 121, 121):
        ts = np.linspace(lo, hi, n_samples, dtype=np.float64)
        vals: list[tuple[float, float]] = []
        for t in ts.tolist():
            vals.append(_eval_t(float(t)))
        finite = [(t, d2) for t, d2 in vals if np.isfinite(d2)]
        if not finite:
            break
        best_t, best_d2 = min(finite, key=lambda x: x[1])
        step = float((hi - lo) / max(1, n_samples - 1))
        lo = max(0.0, best_t - 4.0 * step)
        hi = min(1.0, best_t + 4.0 * step)

    return float(best_t), float(np.sqrt(best_d2))


def _optimize_t_fixed_r01_sagittal(
    *,
    c: SagittalMidlineColumns,
    ijk_target: tuple[float, float, float],
    r01_target: float,
    edt: EdtCrop,
) -> tuple[float, float]:
    """Find t minimizing ||x(t, r01_target) - ijk_target|| in the sagittal plane."""
    target_ij = np.asarray([float(ijk_target[0]), float(ijk_target[1])], dtype=np.float64)

    def _eval_t(t: float) -> tuple[float, float]:
        vent_y = float(_interp(c.t, c.vent_y, t))
        vent_x = float(_interp(c.t, c.vent_x, t))
        pia_y = float(_interp(c.t, c.pia_y, t))
        pia_x = float(_interp(c.t, c.pia_x, t))
        if not np.isfinite([vent_y, vent_x, pia_y, pia_x]).all():
            return float("nan"), float("inf")
        try:
            ijk = invert_r01_along_segment(
                edt,
                vent_ijk=(vent_y, vent_x, float(c.slice_k)),
                pia_ijk=(pia_y, pia_x, float(c.slice_k)),
                r01_target=float(r01_target),
                max_iters=32,
                tol_r01=5.0e-4,
            )
        except ValueError:
            return float("nan"), float("inf")
        ij = np.asarray([float(ijk[0]), float(ijk[1])], dtype=np.float64)
        d2 = float(np.sum((ij - target_ij) ** 2))
        return float(t), d2

    best_t = 0.5
    best_d2 = float("inf")
    lo = 0.0
    hi = 1.0
    for n_samples in (161, 121, 121):
        ts = np.linspace(lo, hi, n_samples, dtype=np.float64)
        vals: list[tuple[float, float]] = []
        for t in ts.tolist():
            vals.append(_eval_t(float(t)))
        finite = [(t, d2) for t, d2 in vals if np.isfinite(d2)]
        if not finite:
            break
        best_t, best_d2 = min(finite, key=lambda x: x[1])
        step = float((hi - lo) / max(1, n_samples - 1))
        lo = max(0.0, best_t - 4.0 * step)
        hi = min(1.0, best_t + 4.0 * step)

    return float(best_t), float(np.sqrt(best_d2))


def invert_coronal_from_ijk(
    columns_by_slice: dict[int, CoronalMidlineColumns],
    *,
    ijk: tuple[float, float, float],
    edt: EdtCrop,
) -> InvertedAxisCoordinate:
    if not columns_by_slice:
        raise ValueError("columns_by_slice is empty.")
    i = float(ijk[0])
    slice_i = _nearest_key(sorted(columns_by_slice.keys()), int(np.rint(i)))
    c = columns_by_slice[slice_i]

    r01 = float(np.clip(float(_r01_at_ijk(edt, ijk)), 0.0, 1.0))
    t_opt, residual = _optimize_t_fixed_r01_coronal(c=c, ijk_target=ijk, r01_target=r01, edt=edt)
    return InvertedAxisCoordinate(axis="coronal", slice_index=int(slice_i), t=float(t_opt), r01=r01, residual_vox=float(residual))


def invert_sagittal_from_ijk(
    columns_by_slice: dict[int, SagittalMidlineColumns],
    *,
    ijk: tuple[float, float, float],
    edt: EdtCrop,
) -> InvertedAxisCoordinate:
    if not columns_by_slice:
        raise ValueError("columns_by_slice is empty.")
    k = float(ijk[2])
    slice_k = _nearest_key(sorted(columns_by_slice.keys()), int(np.rint(k)))
    c = columns_by_slice[slice_k]

    r01 = float(np.clip(float(_r01_at_ijk(edt, ijk)), 0.0, 1.0))
    t_opt, residual = _optimize_t_fixed_r01_sagittal(c=c, ijk_target=ijk, r01_target=r01, edt=edt)
    return InvertedAxisCoordinate(axis="sagittal", slice_index=int(slice_k), t=float(t_opt), r01=r01, residual_vox=float(residual))


def main() -> None:
    parser = argparse.ArgumentParser(description="EDT query (slice_i|slice_k, t, r01) -> CCF ijk/um using *midline_columns.csv")
    parser.add_argument("--outdir", type=Path, default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"))
    parser.add_argument("--axis", choices=["coronal", "sagittal"], default="coronal")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--t", type=float, required=True, help="along-midline coordinate in [0,1] on the chosen slice")
    parser.add_argument("--r01", type=float, required=True, help="depth coordinate in [0,1] (0=inner/vent, 1=pia)")
    parser.add_argument("--tol-r01", type=float, default=2.0e-4)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--transform-to", choices=["coronal", "sagittal"], default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    edt = load_halfway_edt_crop(outdir)
    if args.axis == "coronal":
        cols = load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
        slice_i = int(args.slice)
        if slice_i not in cols:
            slice_i = _nearest_key(sorted(cols.keys()), slice_i)
        c = cols[slice_i]
        vent_y = _interp(c.t, c.vent_y, float(args.t))
        vent_x = _interp(c.t, c.vent_x, float(args.t))
        pia_y = _interp(c.t, c.pia_y, float(args.t))
        pia_x = _interp(c.t, c.pia_x, float(args.t))
        ijk = invert_r01_along_segment(
            edt,
            vent_ijk=(float(c.slice_i), float(vent_y), float(vent_x)),
            pia_ijk=(float(c.slice_i), float(pia_y), float(pia_x)),
            r01_target=float(args.r01),
            max_iters=int(args.max_iters),
            tol_r01=float(args.tol_r01),
        )
    else:
        cols = load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
        slice_k = int(args.slice)
        if slice_k not in cols:
            slice_k = _nearest_key(sorted(cols.keys()), slice_k)
        c = cols[slice_k]
        vent_y = _interp(c.t, c.vent_y, float(args.t))
        vent_x = _interp(c.t, c.vent_x, float(args.t))
        pia_y = _interp(c.t, c.pia_y, float(args.t))
        pia_x = _interp(c.t, c.pia_x, float(args.t))
        ijk = invert_r01_along_segment(
            edt,
            vent_ijk=(float(vent_y), float(vent_x), float(c.slice_k)),
            pia_ijk=(float(pia_y), float(pia_x), float(c.slice_k)),
            r01_target=float(args.r01),
            max_iters=int(args.max_iters),
            tol_r01=float(args.tol_r01),
        )

    print(f"ijk_vox={ijk!r}")

    res = _load_resolution_ds_ijk_um(outdir)
    if res is None:
        print("xyz_um=None (missing resolution_ds_ijk_um.npy)")
        return
    i_um = ijk[0] * res[0]
    j_um = ijk[1] * res[1]
    k_um = ijk[2] * res[2]
    print(f"ijk_um={(i_um, j_um, k_um)!r}")

    if args.transform_to is not None:
        if args.transform_to == args.axis:
            raise ValueError("--transform-to must differ from --axis.")
        if args.transform_to == "coronal":
            coronal = load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
            inv = invert_coronal_from_ijk(coronal, ijk=ijk, edt=edt)
            print(
                "transformed_coronal="
                f"{{slice_i:{inv.slice_index}, t:{inv.t:.6f}, r01:{inv.r01:.6f}, residual_vox:{inv.residual_vox:.4f}}}"
            )
        else:
            sagittal = load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
            inv = invert_sagittal_from_ijk(sagittal, ijk=ijk, edt=edt)
            print(
                "transformed_sagittal="
                f"{{slice_k:{inv.slice_index}, t:{inv.t:.6f}, r01:{inv.r01:.6f}, residual_vox:{inv.residual_vox:.4f}}}"
            )


if __name__ == "__main__":
    main()
