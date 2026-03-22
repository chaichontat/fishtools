from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SUMMARY_JSON = Path("results/refextract/princurve_h5ad_ijk_plot/phase1_summary.json")
DEFAULT_MAX_POINTS = 1_500_000


def _load_summary(summary_json: Path) -> list[dict[str, object]]:
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_json}")
    payload = json.loads(summary_json.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected summary JSON list, got {type(payload).__name__}.")
    out: list[dict[str, object]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Summary row {idx} is {type(row).__name__}, expected object.")
        out.append(row)
    return out


def _is_midline_normal_entry(row: dict[str, object]) -> bool:
    mapping_mode = row.get("ijk_mapping_mode")
    if isinstance(mapping_mode, str) and mapping_mode.strip().lower() == "midline_normal":
        return True
    out_npz = row.get("out_npz")
    if isinstance(out_npz, str):
        return out_npz.endswith("ijk_from_midline_normal.npz")
    return False


def _axis_to_code(axis: str) -> int:
    axis_clean = axis.strip().lower()
    if axis_clean == "coronal":
        return 0
    if axis_clean == "sagittal":
        return 1
    raise ValueError(f"Unsupported axis={axis!r}; expected 'coronal' or 'sagittal'.")


def _load_npz_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    d = np.load(npz_path)
    required = {"ijk", "r_um", "t_lookup", "axis"}
    if not required.issubset(set(d.files)):
        missing = sorted(required - set(d.files))
        raise ValueError(f"NPZ missing required keys {missing}: {npz_path}")

    ijk = np.asarray(d["ijk"], dtype=np.float32)
    r_um = np.asarray(d["r_um"], dtype=np.float32).reshape(-1)
    t_lookup = np.asarray(d["t_lookup"], dtype=np.float32).reshape(-1)
    axis_scalar = str(np.asarray(d["axis"]).reshape(-1)[0])

    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected ijk shape (N,3), got {ijk.shape} in {npz_path}")
    n = int(ijk.shape[0])
    if r_um.shape[0] != n or t_lookup.shape[0] != n:
        raise ValueError(
            f"Array size mismatch in {npz_path}: ijk={n}, r_um={r_um.shape[0]}, t_lookup={t_lookup.shape[0]}"
        )
    if not np.isfinite(ijk).all():
        raise ValueError(f"Non-finite ijk values in {npz_path}")
    if not np.isfinite(r_um).all():
        raise ValueError(f"Non-finite r_um values in {npz_path}")
    if not np.isfinite(t_lookup).all():
        raise ValueError(f"Non-finite t_lookup values in {npz_path}")

    # Viewer uses xyz; atlas arrays are ijk, so xyz=(k,j,i).
    positions_xyz = np.column_stack([ijk[:, 2], ijk[:, 1], ijk[:, 0]]).astype(np.float32, copy=False)
    axis_code = _axis_to_code(axis_scalar)
    axis_u8 = np.full((n,), axis_code, dtype=np.uint8)
    return positions_xyz, r_um.astype(np.float32, copy=False), t_lookup.astype(np.float32, copy=False), axis_u8


def _write_bin(path: Path, values: np.ndarray, *, dtype: np.dtype) -> None:
    arr = np.asarray(values, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def _downsample_all(
    *,
    positions: np.ndarray,
    r_um: np.ndarray,
    t_lookup: np.ndarray,
    axis_u8: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(positions.shape[0])
    if n <= int(max_points):
        return positions, r_um, t_lookup, axis_u8
    rng = np.random.default_rng(int(seed))
    keep = np.sort(rng.choice(np.arange(n, dtype=np.int64), size=int(max_points), replace=False))
    return positions[keep], r_um[keep], t_lookup[keep], axis_u8[keep]


def export_ijk_assets(
    *,
    summary_json: Path,
    output_dir: Path,
    max_points: int,
    seed: int,
) -> Path:
    rows = _load_summary(summary_json)
    selected = [row for row in rows if _is_midline_normal_entry(row)]
    if not selected:
        raise ValueError(
            f"No midline-normal entries found in {summary_json}. Run princurve_h5ad_to_ijk_plot.py with IJK_MAPPING_MODE='midline_normal'."
        )

    pos_parts: list[np.ndarray] = []
    rum_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []

    for row in selected:
        out_npz_raw = row.get("out_npz")
        if not isinstance(out_npz_raw, str) or out_npz_raw.strip() == "":
            raise ValueError(f"Invalid out_npz entry in summary row: {row}")
        npz_path = Path(out_npz_raw)
        positions_xyz, r_um, t_lookup, axis_u8 = _load_npz_arrays(npz_path)
        pos_parts.append(positions_xyz)
        rum_parts.append(r_um)
        t_parts.append(t_lookup)
        axis_parts.append(axis_u8)

    positions = np.concatenate(pos_parts, axis=0).astype(np.float32, copy=False)
    r_um = np.concatenate(rum_parts, axis=0).astype(np.float32, copy=False)
    t_lookup = np.concatenate(t_parts, axis=0).astype(np.float32, copy=False)
    axis_u8 = np.concatenate(axis_parts, axis=0).astype(np.uint8, copy=False)

    positions, r_um, t_lookup, axis_u8 = _downsample_all(
        positions=positions,
        r_um=r_um,
        t_lookup=t_lookup,
        axis_u8=axis_u8,
        max_points=int(max_points),
        seed=int(seed),
    )

    n_points = int(positions.shape[0])
    if n_points == 0:
        raise ValueError("No points available after filtering/downsampling.")

    files = {
        "positions_f32": "positions_f32.bin",
        "r_um_f32": "r_um_f32.bin",
        "t_lookup_f32": "t_lookup_f32.bin",
        "axis_u8": "axis_u8.bin",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_bin(output_dir / files["positions_f32"], positions.reshape(-1), dtype=np.float32)
    _write_bin(output_dir / files["r_um_f32"], r_um, dtype=np.float32)
    _write_bin(output_dir / files["t_lookup_f32"], t_lookup, dtype=np.float32)
    _write_bin(output_dir / files["axis_u8"], axis_u8, dtype=np.uint8)

    axis_counts = {
        "coronal": int(np.count_nonzero(axis_u8 == 0)),
        "sagittal": int(np.count_nonzero(axis_u8 == 1)),
    }
    bbox_min = [float(v) for v in np.min(positions, axis=0)]
    bbox_max = [float(v) for v in np.max(positions, axis=0)]

    manifest = {
        "version": 1,
        "source": "princurve_h5ad_to_ijk_plot",
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
        "axis_counts": axis_counts,
        "export": {
            "summary_json": str(summary_json),
            "rows_selected": int(len(selected)),
            "max_points": int(max_points),
            "seed": int(seed),
            "mapping_mode": "midline_normal",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export global ijk point-cloud assets (midline-normal mapping) for a Vite + Three.js viewer."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Phase1 summary JSON from princurve_h5ad_to_ijk_plot.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifest/bin files (default: <summary_dir>/threejs_ijk_assets)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of points to export (deterministic downsample if exceeded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic downsampling.",
    )
    args = parser.parse_args()

    if int(args.max_points) <= 0:
        raise ValueError(f"--max-points must be > 0, got {args.max_points}")

    summary_json = Path(args.summary_json)
    output_dir = Path(args.output_dir) if args.output_dir is not None else (summary_json.parent / "threejs_ijk_assets")

    manifest_path = export_ijk_assets(
        summary_json=summary_json,
        output_dir=output_dir,
        max_points=int(args.max_points),
        seed=int(args.seed),
    )
    print(f"Wrote: {manifest_path}")
    print(f"Assets dir: {output_dir}")


if __name__ == "__main__":
    main()
