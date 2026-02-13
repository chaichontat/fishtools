from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path("ccf/refextract/optimize_ap_axis_from_strips.py")
COORDS_SCRIPT = Path("ccf/refextract/midsurface_coords.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("optimize_ap_axis_from_strips", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_coords_module():
    spec = importlib.util.spec_from_file_location("midsurface_coords", COORDS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {COORDS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_coronal_midline_columns_csv(*, csv_path: Path, slice_keys: list[int], t: np.ndarray, y: np.ndarray, x: np.ndarray) -> None:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if t.shape != y.shape or t.shape != x.shape:
        raise ValueError(f"Shape mismatch: t={t.shape} y={y.shape} x={x.shape}")

    rows: list[np.ndarray] = []
    for key in slice_keys:
        vent_y = y
        vent_x = x
        pia_y = y + 1.0
        pia_x = x + 1.0
        thickness_um = np.full_like(t, 100.0, dtype=np.float64)
        rows.append(
            np.column_stack(
                [
                    np.full_like(t, float(key), dtype=np.float64),
                    t,
                    y,
                    x,
                    vent_y,
                    vent_x,
                    pia_y,
                    pia_x,
                    thickness_um,
                ]
            )
        )

    table = np.vstack(rows).astype(np.float64, copy=False)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="slice_i,t,y,x,vent_y,vent_x,pia_y,pia_x,thickness_um",
        comments="",
    )


def test_coronal_coords_transform_to_constant_ap_line() -> None:
    mod = _load_module()
    res_ijk_um = (20.0, 10.0, 5.0)
    slice_i = 12
    y_vox = np.linspace(3.0, 9.0, 17, dtype=np.float64)
    x_vox = np.linspace(20.0, 50.0, 17, dtype=np.float64)

    pts = mod._coronal_yx_to_ijk_um(slice_i=slice_i, y_vox=y_vox, x_vox=x_vox, res_ijk_um=res_ijk_um)
    assert pts.shape == (y_vox.size, 3)
    assert np.allclose(pts[:, 0], float(slice_i) * float(res_ijk_um[0]))
    assert np.allclose(pts[:, 1], y_vox * float(res_ijk_um[1]))
    assert np.allclose(pts[:, 2], x_vox * float(res_ijk_um[2]))


def test_sagittal_coords_transform_to_constant_ml_line() -> None:
    mod = _load_module()
    res_ijk_um = (20.0, 10.0, 5.0)
    slice_k = 30
    y_vox = np.linspace(0.0, 12.0, 23, dtype=np.float64)
    x_vox = np.linspace(7.0, 11.0, 23, dtype=np.float64)

    pts = mod._sagittal_yx_to_ijk_um(slice_k=slice_k, y_vox=y_vox, x_vox=x_vox, res_ijk_um=res_ijk_um)
    assert pts.shape == (y_vox.size, 3)
    assert np.allclose(pts[:, 2], float(slice_k) * float(res_ijk_um[2]))
    assert np.allclose(pts[:, 0], y_vox * float(res_ijk_um[0]))
    assert np.allclose(pts[:, 1], x_vox * float(res_ijk_um[1]))


def test_optimize_ap_axis_from_strips_uses_fixed_coronal_ap_coordinate(tmp_path: Path) -> None:
    mod = _load_module()
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "resolution_ds_ijk_um.npy", np.asarray([20.0, 20.0, 20.0], dtype=np.float64))

    n = 33
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.linspace(10.0, 50.0, n, dtype=np.float64)
    x = np.linspace(20.0, 60.0, n, dtype=np.float64)
    _write_coronal_midline_columns_csv(csv_path=outdir / "coronal_midline_columns.csv", slice_keys=[1, 2], t=t, y=y, x=x)

    slice_keys, ap_um, deltas_um, _qcs = mod.optimize_ap_axis_from_strips(
        outdir=outdir,
        n_t=n,
        band_frac=0.2,
        smooth_window=9,
        res_ijk_um=None,
        max_pair_p95_over_p50=3.0,
    )

    assert slice_keys.tolist() == [1, 2]
    assert deltas_um.shape == (1,)
    assert np.allclose(deltas_um[0], 20.0)
    assert ap_um.shape == (2,)
    assert np.allclose(ap_um, np.asarray([0.0, 20.0], dtype=np.float64))


def _write_synthetic_midline_outdir(outdir: Path) -> None:
    shape = (24, 40, 40)
    mask = np.zeros(shape, dtype=bool)
    i0, i1 = 2, 21
    j0, j1 = 5, 34
    k0, k1 = 8, 31
    mask[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] = True

    u = np.zeros(shape, dtype=np.float32)
    span_i = float(i1 - i0)
    span_j = float(j1 - j0)
    span_k = float(k1 - k0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if not mask[i, j, k]:
                    continue
                val_i = (float(i) - float(i0)) / span_i
                val_j = (float(j) - float(j0)) / span_j
                val_k = (float(k) - float(k0)) / span_k
                u[i, j, k] = (val_i + val_j + val_k) / 3.0

    np.save(outdir / "halfway_u_3d_ds.npy", u)
    np.save(outdir / "cortex_mask_fit_3d_ds.npy", mask.astype(np.bool_))
    np.save(outdir / "midline_include_neo_meso_3d_ds.npy", mask.astype(np.bool_))
    np.save(outdir / "resolution_ds_ijk_um.npy", np.asarray([20.0, 20.0, 20.0], dtype=np.float64))


def test_coronal_to_sagittal_transform_yields_constant_ap_midline(tmp_path: Path) -> None:
    coords = _load_coords_module()
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_midline_outdir(outdir)

    coronal = coords.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    sagittal = coords.load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
    assert coronal
    assert sagittal

    slice_i = sorted(coronal.keys())[len(coronal) // 2]
    ts = np.linspace(0.1, 0.9, 9, dtype=np.float64)

    errs: list[float] = []
    target_i: list[float] = []
    for idx, t in enumerate(ts.tolist()):
        inv = coords.transform_with_lut(
            outdir=outdir,
            source_axis="coronal",
            source_slice=int(slice_i),
            t=float(t),
            r01=0.5,
            n_t=128,
            i_window=2,
            k_window=2,
            rebuild_lut=bool(idx == 0),
        )
        c = coronal[int(slice_i)]
        s = sagittal[int(inv.slice_index)]
        p_cor = np.asarray(
            [
                float(c.slice_i),
                float(coords._interp(c.t, c.y, float(t))),
                float(coords._interp(c.t, c.x, float(t))),
            ],
            dtype=np.float64,
        )
        p_sag = np.asarray(
            [
                float(coords._interp(s.t, s.y, float(inv.t))),
                float(coords._interp(s.t, s.x, float(inv.t))),
                float(s.slice_k),
            ],
            dtype=np.float64,
        )
        errs.append(float(np.linalg.norm(p_cor - p_sag)))
        target_i.append(float(p_sag[0]))

    assert float(np.percentile(np.asarray(errs, dtype=np.float64), 95)) <= 0.5
    assert float(np.ptp(np.asarray(target_i, dtype=np.float64))) <= 1.0


def test_sagittal_to_coronal_transform_yields_constant_ml_midline(tmp_path: Path) -> None:
    coords = _load_coords_module()
    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_midline_outdir(outdir)

    coronal = coords.load_coronal_midline_columns(outdir / "coronal_midline_columns.csv")
    sagittal = coords.load_sagittal_midline_columns(outdir / "sagittal_midline_columns.csv")
    assert coronal
    assert sagittal

    slice_k = sorted(sagittal.keys())[len(sagittal) // 2]
    ts = np.linspace(0.1, 0.9, 9, dtype=np.float64)

    errs: list[float] = []
    target_k: list[float] = []
    coronal_slices: list[int] = []
    for idx, t in enumerate(ts.tolist()):
        inv = coords.transform_with_lut(
            outdir=outdir,
            source_axis="sagittal",
            source_slice=int(slice_k),
            t=float(t),
            r01=0.5,
            n_t=128,
            i_window=2,
            k_window=2,
            rebuild_lut=bool(idx == 0),
        )
        s = sagittal[int(slice_k)]
        c = coronal[int(inv.slice_index)]
        p_sag = np.asarray(
            [
                float(coords._interp(s.t, s.y, float(t))),
                float(coords._interp(s.t, s.x, float(t))),
                float(s.slice_k),
            ],
            dtype=np.float64,
        )
        p_cor = np.asarray(
            [
                float(c.slice_i),
                float(coords._interp(c.t, c.y, float(inv.t))),
                float(coords._interp(c.t, c.x, float(inv.t))),
            ],
            dtype=np.float64,
        )
        errs.append(float(np.linalg.norm(p_sag - p_cor)))
        target_k.append(float(p_cor[2]))
        coronal_slices.append(int(inv.slice_index))

    assert len(set(coronal_slices)) >= 3
    assert float(np.percentile(np.asarray(errs, dtype=np.float64), 95)) <= 1.0
    assert float(np.ptp(np.asarray(target_k, dtype=np.float64))) <= 1.5
