from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_find_princurve_module():
    script_path = Path(__file__).parents[1] / "scripts/princurve/find_princurve.py"
    spec = importlib.util.spec_from_file_location("find_princurve", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load find_princurve script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_coronal_midline_csv(path: Path, *, slice_is: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["slice_i", "t", "y", "x"])
        w.writeheader()
        for slice_i in slice_is:
            for t in t_vals:
                w.writerow({"slice_i": int(slice_i), "t": float(t), "y": 0.0, "x": float(t) * 10.0})


def test_compute_ap_ml_um_from_refextract_coronal_and_sagittal(tmp_path: Path) -> None:
    module = _load_find_princurve_module()
    compute = getattr(module, "compute_ap_ml_um_from_refextract")

    outdir = tmp_path / "lut"
    outdir.mkdir(parents=True, exist_ok=True)

    slice_keys = np.asarray([10, 11, 12], dtype=np.int32)
    ap_um = np.asarray([0.0, 100.0, 200.0], dtype=np.float64)
    np.savez(outdir / "ap_axis_um_from_strips.npz", slice_keys=slice_keys, ap_um=ap_um)

    np.save(outdir / "resolution_ds_ijk_um.npy", np.asarray([20.0, 1.0, 1.0], dtype=np.float64))
    _write_coronal_midline_csv(outdir / "coronal_midline_columns.csv", slice_is=[10, 11, 12])

    source_slice_keys = np.asarray([20], dtype=np.int32)
    t_grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    target_slice_idx = np.asarray([[10.0, 11.0, 12.0]], dtype=np.float64)
    target_t = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float64)
    np.savez(
        outdir / "chart_map_sagittal_to_coronal_t2d.npz",
        source_slice_keys=source_slice_keys,
        t_grid=t_grid,
        target_slice_idx=target_slice_idx,
        target_t=target_t,
    )

    t_all = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    cor = compute(
        lut_outdir=outdir,
        axis="coronal",
        atlas_slice_idx=11,
        t_all=t_all,
        ref_slice_i=11,
        ref_t=0.5,
        n_t=33,
        dtw_band_frac=0.2,
    )
    assert cor.shape == (3, 2)
    assert np.allclose(cor[:, 0], np.asarray([100.0, 100.0, 100.0]))
    assert np.allclose(cor[:, 1], np.asarray([-5.0, 0.0, 5.0]))

    sag = compute(
        lut_outdir=outdir,
        axis="sagittal",
        atlas_slice_idx=20,
        t_all=t_all,
        ref_slice_i=11,
        ref_t=0.5,
        n_t=33,
        dtw_band_frac=0.2,
    )
    assert sag.shape == (3, 2)
    assert np.allclose(sag[:, 0], np.asarray([0.0, 100.0, 200.0]))
    assert np.allclose(sag[:, 1], np.asarray([-5.0, 0.0, 5.0]))


def test_compute_ap_ml_um_from_refextract_raises_on_missing_artifacts(tmp_path: Path) -> None:
    module = _load_find_princurve_module()
    compute = getattr(module, "compute_ap_ml_um_from_refextract")

    with pytest.raises(FileNotFoundError, match="Missing required refextract artifacts"):
        compute(lut_outdir=tmp_path, axis="coronal", atlas_slice_idx=1, t_all=np.asarray([0.5], dtype=np.float64))

