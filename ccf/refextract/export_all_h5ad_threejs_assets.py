from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


R_SOURCE_UM_PER_PX = 0.216


def _write_bin(path: Path, values: np.ndarray, *, dtype: np.dtype) -> None:
    arr = np.asarray(values, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def export_all_h5ad_assets(
    *,
    h5ad_path: Path,
    output_dir: Path,
    max_points: int,
    seed: int,
) -> Path:
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad: {h5ad_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    if "ijk" not in adata.obsm:
        raise ValueError(f"Missing obsm['ijk'] in {h5ad_path}")
    if "leiden" not in adata.obs.columns:
        raise ValueError(f"Missing obs['leiden'] in {h5ad_path}")
    if "t_all" not in adata.obs.columns:
        raise ValueError(f"Missing obs['t_all'] in {h5ad_path}")
    if "principal_r_signed" not in adata.obsm:
        raise ValueError(f"Missing obsm['principal_r_signed'] in {h5ad_path}")

    n_total = int(adata.n_obs)
    if n_total <= 0:
        raise ValueError(f"Unexpected n_obs={n_total} in {h5ad_path}")

    max_points_i = int(max_points)
    if max_points_i <= 0:
        raise ValueError(f"max_points must be positive, got {max_points_i}")

    rng = np.random.default_rng(int(seed))
    if n_total <= max_points_i:
        keep = np.arange(n_total, dtype=np.int64)
    else:
        keep = np.sort(rng.choice(np.arange(n_total, dtype=np.int64), size=max_points_i, replace=False))

    ijk = np.asarray(adata.obsm["ijk"][keep, :], dtype=np.float32)
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected obsm['ijk'] shape (N,3), got {ijk.shape} in {h5ad_path}")
    if not np.isfinite(ijk).all():
        raise ValueError(f"Non-finite ijk values in {h5ad_path}")

    # Viewer uses xyz; atlas arrays are ijk, so xyz=(k,j,i).
    positions_xyz = np.column_stack([ijk[:, 2], ijk[:, 1], ijk[:, 0]]).astype(np.float32, copy=False)
    if not np.isfinite(positions_xyz).all():
        raise ValueError(f"Non-finite xyz positions derived from ijk in {h5ad_path}")

    t_all = np.asarray(adata.obs["t_all"].to_numpy(dtype=np.float32, copy=False), dtype=np.float32).reshape(-1)
    r_signed = np.asarray(adata.obsm["principal_r_signed"], dtype=np.float32).reshape(-1)
    if t_all.shape[0] != n_total or r_signed.shape[0] != n_total:
        raise ValueError(f"Length mismatch in {h5ad_path}: t_all={t_all.shape[0]} r_signed={r_signed.shape[0]}")
    t_lookup = t_all[keep].astype(np.float32, copy=False)
    r_um = (r_signed[keep] * float(R_SOURCE_UM_PER_PX)).astype(np.float32, copy=False)
    if not np.isfinite(t_lookup).all():
        raise ValueError(f"Non-finite t_all values in {h5ad_path}")
    if not np.isfinite(r_um).all():
        raise ValueError(f"Non-finite principal_r_signed-derived r_um values in {h5ad_path}")

    leiden = adata.obs["leiden"]
    try:
        categories = [str(v) for v in list(leiden.cat.categories)]
        codes_all = leiden.cat.codes.to_numpy(dtype=np.int32, copy=False)
    except Exception as e:
        raise ValueError(f"Expected obs['leiden'] to be a pandas Categorical in {h5ad_path}") from e
    if codes_all.shape[0] != n_total:
        raise ValueError(f"leiden codes length mismatch in {h5ad_path}: {codes_all.shape[0]} vs {n_total}")
    codes = codes_all[keep]
    if np.any(codes < 0):
        raise ValueError(f"Found missing leiden assignments (code -1) in kept rows from {h5ad_path}")
    leiden_u16 = np.asarray(codes, dtype=np.uint16)

    # This dataset is already in 3D ijk, so "axis" is not meaningful; keep for viewer compatibility.
    axis_u8 = np.zeros((positions_xyz.shape[0],), dtype=np.uint8)

    files = {
        "positions_f32": "positions_f32.bin",
        "r_um_f32": "r_um_f32.bin",
        "t_lookup_f32": "t_lookup_f32.bin",
        "axis_u8": "axis_u8.bin",
        "leiden_u16": "leiden_u16.bin",
        "leiden_categories_json": "leiden_categories.json",
    }

    _write_bin(output_dir / files["positions_f32"], positions_xyz.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["r_um_f32"], r_um, dtype=np.float32)
    _write_bin(output_dir / files["t_lookup_f32"], t_lookup, dtype=np.float32)
    _write_bin(output_dir / files["axis_u8"], axis_u8, dtype=np.uint8)
    _write_bin(output_dir / files["leiden_u16"], leiden_u16, dtype=np.uint16)
    (output_dir / files["leiden_categories_json"]).write_text(json.dumps(categories, indent=2))

    n_points = int(positions_xyz.shape[0])
    bbox_min = [float(v) for v in np.min(positions_xyz, axis=0)]
    bbox_max = [float(v) for v in np.max(positions_xyz, axis=0)]
    counts: dict[str, int] = {}
    for code in range(len(categories)):
        counts[str(categories[code])] = int(np.count_nonzero(leiden_u16 == code))

    manifest = {
        "version": 1,
        "source": "all.h5ad",
        "coord_space": "ijk",
        "position_order": "xyz=kji",
        "n_points": n_points,
        "files": files,
        "stats": {
            "r_um_min": float(np.min(r_um)),
            "r_um_max": float(np.max(r_um)),
            "t_lookup_min": float(np.min(t_lookup)),
            "t_lookup_max": float(np.max(t_lookup)),
            "bbox_xyz_min": bbox_min,
            "bbox_xyz_max": bbox_max,
        },
        "axis_counts": {
            "coronal": int(np.count_nonzero(axis_u8 == 0)),
            "sagittal": int(np.count_nonzero(axis_u8 == 1)),
        },
        "leiden": {
            "n_categories": int(len(categories)),
            "counts": counts,
        },
        "export": {
            "h5ad_path": str(h5ad_path),
            "n_total": n_total,
            "max_points": int(max_points),
            "seed": int(seed),
            "r_source_um_per_px": float(R_SOURCE_UM_PER_PX),
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def main() -> None:
    default_h5ad = Path.home() / "nvme" / "all.h5ad"
    default_out = Path("results/refextract/all_h5ad_threejs_ijk_assets")

    parser = argparse.ArgumentParser(description="Export ~/nvme/all.h5ad into Three.js viewer assets.")
    parser.add_argument("--h5ad", type=Path, default=default_h5ad, help="Input .h5ad (default: ~/nvme/all.h5ad)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Output directory for manifest/bin files.",
    )
    parser.add_argument("--max-points", type=int, default=1_500_000, help="Downsample cap for rendering.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for downsampling.")
    args = parser.parse_args()

    manifest = export_all_h5ad_assets(
        h5ad_path=Path(args.h5ad),
        output_dir=Path(args.output_dir),
        max_points=int(args.max_points),
        seed=int(args.seed),
    )
    print(f"Wrote: {manifest}")


if __name__ == "__main__":
    main()

