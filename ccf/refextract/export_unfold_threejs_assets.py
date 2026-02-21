from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccf.refextract.plot_ap_ml_mapping_surface_3d import SurfaceMappingData, build_coronal_ap_ml_surface


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


def _build_manifest(
    *,
    data: SurfaceMappingData,
    output_dir: Path,
    ref_t: float,
    phase1_frac_default: float,
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
    }

    _write_bin(output_dir / files["positions_f32"], verts_xyz.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["faces_u32"], faces.reshape(-1), dtype=np.uint32)
    _write_bin(output_dir / files["colors_u8"], colors.reshape(-1), dtype=np.uint8)
    _write_bin(output_dir / files["seglen_f32"], seg_len.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["theta_f32"], theta.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["ap_um_f32"], ap.astype(np.float32, copy=False), dtype=np.float32)
    _write_bin(output_dir / files["neo_t_support_u8"], neo_t_support.reshape(-1), dtype=np.uint8)

    return {
        "version": 1,
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
    manifest = _build_manifest(
        data=data,
        output_dir=output_dir,
        ref_t=float(ref_t),
        phase1_frac_default=float(phase1_frac_default),
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
    )
    print(f"Wrote: {manifest_path}")
    print(f"Assets dir: {output_dir}")


if __name__ == "__main__":
    main()
