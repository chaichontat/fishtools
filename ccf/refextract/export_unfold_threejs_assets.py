from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccf.refextract.plot_ap_ml_mapping_surface_3d import (
    SurfaceMappingData,
    _ap_hline_values,
    _load_midsurface_coords_module,
    _load_s2c_t2d,
    _map_sagittal_to_coronal_t2d,
    build_coronal_ap_ml_surface,
)


MANIFEST_VERSION = 2
CORONAL_LINE_KIND = 0
SAGITTAL_LINE_KIND = 1
DEFAULT_AP_HLINE_STEP_UM = 750.0
DEFAULT_SAGITTAL_K_STEP = 12
DEFAULT_SAGITTAL_N_SAMPLE = 257


def _ijk_um_to_xyz_um(verts_ijk_um: np.ndarray) -> np.ndarray:
    arr = np.asarray(verts_ijk_um, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N,3), got {arr.shape}.")
    return arr[:, [2, 1, 0]]


def _prepare_ml_unroll_rows_jk(verts_ijk_um: np.ndarray, *, n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(verts_ijk_um, dtype=np.float64)
    if verts.shape != (int(n_rows * n_cols), 3):
        raise ValueError(f"Expected vertices shape {(int(n_rows * n_cols), 3)}, got {verts.shape}.")
    grid = verts.reshape(int(n_rows), int(n_cols), 3)
    j = grid[:, :, 1]
    k = grid[:, :, 2]
    dj = np.diff(j, axis=1)
    dk = np.diff(k, axis=1)
    seg_len = np.hypot(dk, dj).astype(np.float32, copy=False)
    theta = np.arctan2(dj, dk).astype(np.float32, copy=False)
    if seg_len.shape != (n_rows, n_cols - 1) or theta.shape != (n_rows, n_cols - 1):
        raise ValueError(f"Unexpected segment arrays: seg_len={seg_len.shape} theta={theta.shape}.")
    return seg_len, theta


def _write_bin(path: Path, array: np.ndarray, *, dtype: np.dtype) -> None:
    arr = np.asarray(array, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def _append_line_runs(
    *,
    row_f: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    kind: int,
    points_acc: list[np.ndarray],
    ranges_acc: list[np.ndarray],
    n_points: int,
) -> int:
    row_vals = np.asarray(row_f, dtype=np.float64).reshape(-1)
    t_vals = np.asarray(t, dtype=np.float64).reshape(-1)
    keep = np.asarray(valid, dtype=bool).reshape(-1)
    if row_vals.shape != t_vals.shape or row_vals.shape != keep.shape:
        raise ValueError(
            f"line shape mismatch: row={row_vals.shape} t={t_vals.shape} valid={keep.shape}"
        )

    run = keep.astype(np.int8, copy=False)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], run])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([run, [0]])) == -1) + 1
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if int(end) - int(start) < 2:
            continue
        pts = np.column_stack([row_vals[int(start) : int(end)], t_vals[int(start) : int(end)]])
        pts = pts.astype(np.float32, copy=False)
        points_acc.append(pts)
        ranges_acc.append(
            np.asarray([int(n_points), int(pts.shape[0]), int(kind), 0], dtype=np.uint32)
        )
        n_points += int(pts.shape[0])
    return n_points


def _build_coronal_line_params(
    *,
    data: SurfaceMappingData,
    ap_hline_step_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    ap_lines = _ap_hline_values(ap_range_um=data.ap_range_um, step_um=float(ap_hline_step_um))
    if ap_lines.size > 1:
        ap_lines = ap_lines[1:]

    ap_vals = np.asarray(data.ap_um_by_slice, dtype=np.float64)
    t_grid = np.asarray(data.t_grid, dtype=np.float64)
    support = np.asarray(data.support_mask_tall, dtype=bool)
    if support.shape != (int(ap_vals.size), int(t_grid.size)):
        raise ValueError(f"support_mask_tall shape mismatch: {support.shape}")

    points_acc: list[np.ndarray] = []
    ranges_acc: list[np.ndarray] = []
    n_points = 0
    for ap_um in np.asarray(ap_lines, dtype=np.float64).tolist():
        row_idx = int(np.argmin(np.abs(ap_vals - float(ap_um))))
        row_f = np.full(t_grid.shape, float(row_idx), dtype=np.float64)
        valid = np.isfinite(t_grid) & support[row_idx]
        n_points = _append_line_runs(
            row_f=row_f,
            t=t_grid,
            valid=valid,
            kind=CORONAL_LINE_KIND,
            points_acc=points_acc,
            ranges_acc=ranges_acc,
            n_points=n_points,
        )

    if not points_acc:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.uint32),
        )
    return (
        np.vstack(points_acc).astype(np.float32, copy=False),
        np.vstack(ranges_acc).astype(np.uint32, copy=False),
    )


def _build_sagittal_line_params(
    *,
    data: SurfaceMappingData,
    available_sagittal_keys: set[int],
    source_slice_keys: np.ndarray,
    lut_t_grid: np.ndarray,
    target_slice_idx: np.ndarray,
    target_t: np.ndarray,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_keys = np.asarray(source_slice_keys, dtype=np.int32).reshape(-1)
    if source_keys.size == 0:
        raise ValueError("No sagittal source_slice_keys available for line export.")
    k_step = int(sagittal_k_step)
    if k_step <= 0:
        raise ValueError(f"sagittal_k_step must be >0, got {sagittal_k_step}")
    n_sample = int(sagittal_n_sample)
    if n_sample < 16:
        raise ValueError(f"sagittal_n_sample must be >=16, got {sagittal_n_sample}")

    requested = list(range(int(np.min(source_keys)), int(np.max(source_keys)) + 1, k_step))
    use_ks = sorted(
        {
            int(source_keys[int(np.argmin(np.abs(source_keys.astype(np.float64) - float(k))))])
            for k in requested
        }
    )
    use_ks = [int(k) for k in use_ks if int(k) in available_sagittal_keys]
    if not use_ks:
        raise ValueError("No sagittal slices available after intersecting LUT keys with sagittal_midline_columns.csv.")

    coronal_keys = np.asarray(data.slice_keys, dtype=np.float64).reshape(-1)
    row_axis = np.arange(coronal_keys.size, dtype=np.float64)
    t_grid = np.asarray(data.t_grid, dtype=np.float64).reshape(-1)
    support_tall = np.asarray(data.support_mask_tall, dtype=bool)
    if support_tall.shape != (int(coronal_keys.size), int(t_grid.size)):
        raise ValueError(f"support_mask_tall shape mismatch: {support_tall.shape}")

    t_s_plot = np.linspace(0.0, 1.0, n_sample, dtype=np.float64)
    points_acc: list[np.ndarray] = []
    ranges_acc: list[np.ndarray] = []
    n_points = 0
    for slice_k in use_ks:
        cor_slice_f, cor_t = _map_sagittal_to_coronal_t2d(
            source_slice_keys=source_keys,
            lut_t_grid=np.asarray(lut_t_grid, dtype=np.float64),
            target_slice_idx=np.asarray(target_slice_idx, dtype=np.float64),
            target_t=np.asarray(target_t, dtype=np.float64),
            slice_k=int(slice_k),
            t_s=t_s_plot,
        )
        row_idx = np.argmin(np.abs(coronal_keys[:, None] - cor_slice_f[None, :]), axis=0)
        row_f = np.interp(cor_slice_f, coronal_keys, row_axis, left=np.nan, right=np.nan)
        valid = np.isfinite(row_f) & np.isfinite(cor_t)
        support_curve = np.zeros(cor_t.shape, dtype=bool)
        for ridx in np.unique(row_idx[valid]).tolist():
            ridx_i = int(ridx)
            mask = valid & (row_idx == ridx_i)
            support_curve[mask] = (
                np.interp(
                    cor_t[mask],
                    t_grid,
                    support_tall[ridx_i].astype(np.float64),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
        n_points = _append_line_runs(
            row_f=row_f,
            t=cor_t,
            valid=valid & support_curve,
            kind=SAGITTAL_LINE_KIND,
            points_acc=points_acc,
            ranges_acc=ranges_acc,
            n_points=n_points,
        )

    if not points_acc:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.uint32),
        )
    return (
        np.vstack(points_acc).astype(np.float32, copy=False),
        np.vstack(ranges_acc).astype(np.uint32, copy=False),
    )


def _build_reference_line_assets(
    *,
    data: SurfaceMappingData,
    outdir: Path,
    ap_hline_step_um: float,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    coronal_points, coronal_ranges = _build_coronal_line_params(
        data=data,
        ap_hline_step_um=float(ap_hline_step_um),
    )

    mod = _load_midsurface_coords_module()
    sagittal = mod.load_sagittal_midline_columns(Path(outdir) / "sagittal_midline_columns.csv")
    if not sagittal:
        raise ValueError(f"No sagittal midline columns found under {outdir}")
    s2c_path = Path(outdir) / "chart_map_sagittal_to_coronal_t2d.npz"
    if not s2c_path.exists():
        raise FileNotFoundError(f"Missing {s2c_path}. Build chart LUTs first with midsurface_coords.py.")
    source_slice_keys, lut_t_grid, target_slice_idx, target_t = _load_s2c_t2d(s2c_path)
    sagittal_points, sagittal_ranges = _build_sagittal_line_params(
        data=data,
        available_sagittal_keys={int(k) for k in sagittal.keys()},
        source_slice_keys=source_slice_keys,
        lut_t_grid=lut_t_grid,
        target_slice_idx=target_slice_idx,
        target_t=target_t,
        sagittal_k_step=int(sagittal_k_step),
        sagittal_n_sample=int(sagittal_n_sample),
    )

    if coronal_points.size == 0 and sagittal_points.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.uint32),
        )
    if coronal_points.size == 0:
        return sagittal_points, sagittal_ranges
    if sagittal_points.size == 0:
        return coronal_points, coronal_ranges

    offset = int(coronal_points.shape[0])
    sagittal_ranges_shifted = sagittal_ranges.copy()
    sagittal_ranges_shifted[:, 0] += np.uint32(offset)
    return (
        np.vstack([coronal_points, sagittal_points]).astype(np.float32, copy=False),
        np.vstack([coronal_ranges, sagittal_ranges_shifted]).astype(np.uint32, copy=False),
    )


def _build_manifest(
    *,
    data: SurfaceMappingData,
    output_dir: Path,
    ref_t: float,
    phase1_frac_default: float,
    line_points: np.ndarray,
    line_ranges: np.ndarray,
    ap_hline_step_um: float,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> dict[str, object]:
    n_rows = int(data.slice_keys.size)
    n_cols = int(data.t_grid.size)
    n_vertices = int(n_rows * n_cols)
    verts_ijk = np.asarray(data.vertices_ijk_um, dtype=np.float32)
    if verts_ijk.shape != (n_vertices, 3):
        raise ValueError(f"Unexpected vertex shape: {verts_ijk.shape}.")

    anchor_col = int(np.clip(np.rint(float(ref_t) * float(n_cols - 1)), 0, n_cols - 1))
    anchor_row = int(n_rows // 2)

    ap = np.asarray(data.ap_um_by_slice, dtype=np.float64).reshape(-1)
    if ap.shape != (n_rows,):
        raise ValueError(f"Unexpected ap_um_by_slice shape {ap.shape}; expected {(n_rows,)}.")
    ap_anchor = float(ap[anchor_row])

    src_grid = verts_ijk.reshape(n_rows, n_cols, 3).astype(np.float64, copy=False)
    src_anchor_i = float(np.nanmedian(src_grid[anchor_row, :, 0]))
    j_plane = float(np.nanmedian(src_grid[:, :, 1]))
    if not np.isfinite(src_anchor_i) or not np.isfinite(j_plane):
        raise ValueError("Non-finite source AP/DV anchors.")

    verts_xyz = _ijk_um_to_xyz_um(verts_ijk).astype(np.float32, copy=False)
    flip_y0 = float(np.nanmean(verts_xyz[:, 1].astype(np.float64, copy=False)))
    if not np.isfinite(flip_y0):
        raise ValueError("Non-finite flip_y0.")

    faces = np.asarray(data.faces, dtype=np.uint32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape {faces.shape}; expected (F,3).")
    colors = np.asarray(data.rgb_u8, dtype=np.uint8)
    if colors.shape != verts_xyz.shape:
        raise ValueError(f"Unexpected colors shape {colors.shape}; expected {verts_xyz.shape}.")
    neo_t_support = np.asarray(data.neomeso_mask_tall, dtype=np.uint8)
    if neo_t_support.shape != (n_rows, n_cols):
        raise ValueError(
            f"Unexpected neomeso_mask_tall shape {neo_t_support.shape}; expected {(n_rows, n_cols)}."
        )

    seg_len, theta = _prepare_ml_unroll_rows_jk(verts_ijk, n_rows=n_rows, n_cols=n_cols)

    files = {
        "positions_f32": "positions_f32.bin",
        "faces_u32": "faces_u32.bin",
        "colors_u8": "colors_u8.bin",
        "seglen_f32": "seglen_f32.bin",
        "theta_f32": "theta_f32.bin",
        "ap_um_f32": "ap_um_f32.bin",
        "neo_t_support_u8": "neo_t_support_u8.bin",
        "line_points_f32": "line_points_f32.bin",
        "line_ranges_u32": "line_ranges_u32.bin",
    }

    _write_bin(output_dir / files["positions_f32"], verts_xyz.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["faces_u32"], faces.reshape(-1), dtype=np.uint32)
    _write_bin(output_dir / files["colors_u8"], colors.reshape(-1), dtype=np.uint8)
    _write_bin(output_dir / files["seglen_f32"], seg_len.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["theta_f32"], theta.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["ap_um_f32"], ap.astype(np.float32, copy=False), dtype=np.float32)
    _write_bin(output_dir / files["neo_t_support_u8"], neo_t_support.reshape(-1), dtype=np.uint8)
    _write_bin(output_dir / files["line_points_f32"], line_points.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["line_ranges_u32"], line_ranges.reshape(-1), dtype=np.uint32)

    return {
        "version": MANIFEST_VERSION,
        "source": "build_coronal_ap_ml_surface",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_vertices": n_vertices,
        "n_faces": int(faces.shape[0]),
        "anchor_col": anchor_col,
        "anchor_row": anchor_row,
        "phase1_frac_default": float(phase1_frac_default),
        "ap_anchor": ap_anchor,
        "src_anchor_i": src_anchor_i,
        "j_plane": j_plane,
        "flip_y0": flip_y0,
        "reference_lines": {
            "n_points": int(line_points.shape[0]),
            "n_ranges": int(line_ranges.shape[0]),
            "kinds": {"coronal": CORONAL_LINE_KIND, "sagittal": SAGITTAL_LINE_KIND},
            "defaults": {
                "ap_hline_step_um": float(ap_hline_step_um),
                "sagittal_k_step": int(sagittal_k_step),
                "sagittal_n_sample": int(sagittal_n_sample),
            },
        },
        "files": files,
    }


def export_threejs_assets(
    *,
    outdir: Path,
    output_dir: Path,
    slice_i_min: int,
    slice_i_max: int,
    n_t: int,
    ref_t: float,
    band_frac: float,
    b_const: float,
    res_ijk_um: tuple[float, float, float],
    phase1_frac_default: float,
    ap_hline_step_um: float,
    sagittal_k_step: int,
    sagittal_n_sample: int,
) -> Path:
    outdir = Path(outdir)
    output_dir = Path(output_dir)
    data = build_coronal_ap_ml_surface(
        outdir=outdir,
        slice_i_min=int(slice_i_min),
        slice_i_max=int(slice_i_max),
        n_t=int(n_t),
        ref_t=float(ref_t),
        band_frac=float(band_frac),
        b_const=float(b_const),
        res_ijk_um=(float(res_ijk_um[0]), float(res_ijk_um[1]), float(res_ijk_um[2])),
    )
    line_points, line_ranges = _build_reference_line_assets(
        data=data,
        outdir=outdir,
        ap_hline_step_um=float(ap_hline_step_um),
        sagittal_k_step=int(sagittal_k_step),
        sagittal_n_sample=int(sagittal_n_sample),
    )
    manifest = _build_manifest(
        data=data,
        output_dir=output_dir,
        ref_t=float(ref_t),
        phase1_frac_default=float(phase1_frac_default),
        line_points=line_points,
        line_ranges=line_ranges,
        ap_hline_step_um=float(ap_hline_step_um),
        sagittal_k_step=int(sagittal_k_step),
        sagittal_n_sample=int(sagittal_n_sample),
    )
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AP/ML unfolding mesh assets for a Three.js interactive viewer.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("ccf/out/refextract/midsurface_neocortex_mesocortex_allocortex_3d"),
        help="Midsurface output directory used to build the coronal AP/ML surface.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory for Three.js assets (default: <outdir>/threejs_unfold).",
    )
    parser.add_argument("--slice-i-min", type=int, default=161, help="Minimum coronal slice_i to include (inclusive).")
    parser.add_argument("--slice-i-max", type=int, default=305, help="Maximum coronal slice_i to include (inclusive).")
    parser.add_argument("--n-t", type=int, default=257, help="Number of samples along each coronal midline curve.")
    parser.add_argument("--ref-t", type=float, default=0.5, help="Reference t_all in [0,1] used as ML anchor column.")
    parser.add_argument("--band-frac", type=float, default=0.15, help="DTW Sakoe-Chiba band as fraction of n_t.")
    parser.add_argument("--b-const", type=float, default=0.25, help="Constant blue channel value in [0,1] for RGB.")
    parser.add_argument(
        "--res-ijk-um",
        type=float,
        nargs=3,
        default=(20.0, 20.0, 20.0),
        metavar=("RI", "RJ", "RK"),
        help="Voxel size in um for (i,j,k).",
    )
    parser.add_argument(
        "--phase1-frac-default",
        type=float,
        default=0.65,
        help="Default ML-phase fraction suggested to the web viewer UI.",
    )
    parser.add_argument(
        "--ap-hline-step-um",
        type=float,
        default=DEFAULT_AP_HLINE_STEP_UM,
        help="Coronal AP spacing in um for exported manifold reference lines.",
    )
    parser.add_argument(
        "--sagittal-k-step",
        type=int,
        default=DEFAULT_SAGITTAL_K_STEP,
        help="Sagittal slice step for exported manifold reference lines.",
    )
    parser.add_argument(
        "--sagittal-n-sample",
        type=int,
        default=DEFAULT_SAGITTAL_N_SAMPLE,
        help="Samples per sagittal manifold line.",
    )
    args = parser.parse_args()

    if not (0.0 < float(args.phase1_frac_default) < 1.0):
        raise ValueError(f"--phase1-frac-default must be in (0,1), got {args.phase1_frac_default}.")

    output_dir = Path(args.output_dir) if args.output_dir is not None else (Path(args.outdir) / "threejs_unfold")
    manifest_path = export_threejs_assets(
        outdir=Path(args.outdir),
        output_dir=output_dir,
        slice_i_min=int(args.slice_i_min),
        slice_i_max=int(args.slice_i_max),
        n_t=int(args.n_t),
        ref_t=float(args.ref_t),
        band_frac=float(args.band_frac),
        b_const=float(args.b_const),
        res_ijk_um=(float(args.res_ijk_um[0]), float(args.res_ijk_um[1]), float(args.res_ijk_um[2])),
        phase1_frac_default=float(args.phase1_frac_default),
        ap_hline_step_um=float(args.ap_hline_step_um),
        sagittal_k_step=int(args.sagittal_k_step),
        sagittal_n_sample=int(args.sagittal_n_sample),
    )
    print(f"Wrote: {manifest_path}")
    print(f"Assets dir: {output_dir}")


if __name__ == "__main__":
    main()
