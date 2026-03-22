from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


EXPORT_SCRIPT = Path("ccf/refextract/export_unfold_threejs_assets.py")
PLOT_SCRIPT = Path("ccf/refextract/plot_ap_ml_mapping_surface_3d.py")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_surface_data(plot_mod):
    n_rows = 3
    n_cols = 5
    slice_keys = np.asarray([10, 20, 30], dtype=np.int32)
    t_grid = np.linspace(0.0, 1.0, n_cols, dtype=np.float64)
    ap_um = np.asarray([100.0, 800.0, 1500.0], dtype=np.float64)
    ml_um = np.asarray(
        [
            [-200.0, -100.0, 0.0, 100.0, 200.0],
            [-180.0, -80.0, 20.0, 120.0, 220.0],
            [-160.0, -60.0, 40.0, 140.0, 240.0],
        ],
        dtype=np.float64,
    )
    support = np.asarray(
        [
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        dtype=bool,
    )
    vertices: list[list[float]] = []
    for row, slice_i in enumerate(slice_keys.tolist()):
        for col in range(n_cols):
            vertices.append([float(slice_i) * 20.0, 100.0 + row, 200.0 + col])
    faces = np.asarray(
        [
            [0, 1, 5],
            [1, 6, 5],
            [1, 2, 6],
            [2, 7, 6],
        ],
        dtype=np.uint32,
    )
    colors = np.full((n_rows * n_cols, 3), 127, dtype=np.uint8)
    return plot_mod.SurfaceMappingData(
        vertices_ijk_um=np.asarray(vertices, dtype=np.float32),
        faces=faces,
        rgb_u8=colors,
        slice_keys=slice_keys,
        t_grid=t_grid,
        t_all_at_t=np.broadcast_to(t_grid, (n_rows, n_cols)).astype(np.float64, copy=False),
        path_yx_vox_at_t=np.zeros((n_rows, n_cols, 2), dtype=np.float64),
        support_mask_tall=support,
        neomeso_mask_tall=support.copy(),
        ap_um_by_slice=ap_um,
        ml_um_at_t=ml_um,
        ap_norm_by_slice=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        ml_norm_at_t=np.asarray(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.05, 0.3, 0.55, 0.8, 1.0],
                [0.1, 0.35, 0.6, 0.85, 1.0],
            ],
            dtype=np.float64,
        ),
        ap_range_um=(0.0, 2000.0),
        ml_range_um=(-200.0, 240.0),
    )


def test_build_line_params_for_coronal_and_sagittal_defaults() -> None:
    plot_mod = _load_module(PLOT_SCRIPT, "plot_ap_ml_mapping_surface_3d_test_export")
    export_mod = _load_module(EXPORT_SCRIPT, "export_unfold_threejs_assets_test_lines")
    data = _make_surface_data(plot_mod)

    coronal_points, coronal_ranges = export_mod._build_coronal_line_params(
        data=data,
        ap_hline_step_um=750.0,
    )
    assert coronal_points.shape[1] == 2
    assert coronal_ranges.shape[1] == 4
    assert coronal_ranges[:, 2].tolist() == [export_mod.CORONAL_LINE_KIND] * int(coronal_ranges.shape[0])
    assert coronal_ranges[:, 1].tolist() == [2, 2, 5]
    assert np.allclose(coronal_points[:2, 0], 1.0)
    assert np.allclose(coronal_points[-5:, 0], 2.0)

    source_slice_keys = np.asarray([5, 17], dtype=np.int32)
    lut_t_grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    target_slice_idx = np.asarray(
        [
            [10.0, 15.0, 20.0, 25.0, 30.0],
            [10.0, 15.0, 20.0, 25.0, 30.0],
        ],
        dtype=np.float64,
    )
    target_t = np.broadcast_to(lut_t_grid, target_slice_idx.shape).astype(np.float64, copy=False)
    sagittal_points, sagittal_ranges = export_mod._build_sagittal_line_params(
        data=data,
        available_sagittal_keys={5, 17},
        source_slice_keys=source_slice_keys,
        lut_t_grid=lut_t_grid,
        target_slice_idx=target_slice_idx,
        target_t=target_t,
        sagittal_k_step=12,
        sagittal_n_sample=17,
    )
    assert sagittal_points.shape[1] == 2
    assert sagittal_ranges.shape == (4, 4)
    assert sagittal_ranges[:, 2].tolist() == [export_mod.SAGITTAL_LINE_KIND] * 4
    assert np.all(sagittal_ranges[:, 1] >= 2)
    assert int(np.sum(sagittal_ranges[:, 1].astype(np.int64, copy=False))) == int(
        sagittal_points.shape[0]
    )
    assert np.all(np.diff(sagittal_ranges[:, 0].astype(np.int64, copy=False)) > 0)
    assert np.all(np.isfinite(sagittal_points))
    assert float(np.min(sagittal_points[:, 0])) >= 0.0
    assert float(np.max(sagittal_points[:, 0])) <= 2.0
    assert float(np.min(sagittal_points[:, 1])) >= 0.0
    assert float(np.max(sagittal_points[:, 1])) <= 1.0


def test_export_threejs_assets_writes_reference_line_manifest_and_bins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plot_mod = _load_module(PLOT_SCRIPT, "plot_ap_ml_mapping_surface_3d_test_manifest")
    export_mod = _load_module(EXPORT_SCRIPT, "export_unfold_threejs_assets_test_manifest")
    data = _make_surface_data(plot_mod)

    monkeypatch.setattr(export_mod, "build_coronal_ap_ml_surface", lambda **_kwargs: data)
    monkeypatch.setattr(
        export_mod,
        "_build_reference_line_assets",
        lambda **_kwargs: (
            np.asarray([[1.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32),
            np.asarray(
                [
                    [0, 2, export_mod.CORONAL_LINE_KIND, 0],
                    [2, 1, export_mod.SAGITTAL_LINE_KIND, 0],
                ],
                dtype=np.uint32,
            ),
        ),
    )

    output_dir = tmp_path / "threejs_unfold"
    manifest_path = export_mod.export_threejs_assets(
        outdir=tmp_path / "out",
        output_dir=output_dir,
        slice_i_min=0,
        slice_i_max=100,
        n_t=5,
        ref_t=0.5,
        band_frac=0.2,
        b_const=0.25,
        res_ijk_um=(20.0, 20.0, 20.0),
        phase1_frac_default=0.65,
        ap_hline_step_um=750.0,
        sagittal_k_step=12,
        sagittal_n_sample=257,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == export_mod.MANIFEST_VERSION
    assert manifest["reference_lines"]["n_points"] == 3
    assert manifest["reference_lines"]["n_ranges"] == 2
    assert manifest["reference_lines"]["defaults"]["ap_hline_step_um"] == 750.0
    assert manifest["reference_lines"]["defaults"]["sagittal_k_step"] == 12
    assert manifest["reference_lines"]["defaults"]["sagittal_n_sample"] == 257

    files = manifest["files"]
    assert (output_dir / files["line_points_f32"]).exists()
    assert (output_dir / files["line_ranges_u32"]).exists()
    line_points = np.fromfile(output_dir / files["line_points_f32"], dtype=np.float32)
    line_ranges = np.fromfile(output_dir / files["line_ranges_u32"], dtype=np.uint32)
    assert line_points.size == 6
    assert line_ranges.size == 8
