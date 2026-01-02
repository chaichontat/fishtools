from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner

from _cellpose_stub import ensure_cellpose_stub
from fishtools.utils.zarr_utils import default_zarr_codecs


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "analysis/deconv").mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def _write_segmentation_zarr(ws: Path, roi: str, seg_cb: str, data: np.ndarray) -> Path:
    seg_dir = ws / "analysis/deconv" / f"stitch--{roi}+{seg_cb}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / "output_segmentation.zarr"
    arr = zarr.open_array(
        str(out),
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=data.shape,
        codecs=default_zarr_codecs(data.dtype),
    )
    arr[:] = data
    return out


def _read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header.")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii")
        vertex_count: int | None = None
        for line in header.splitlines():
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise RuntimeError("Missing 'element vertex' in PLY header.")
        verts = np.fromfile(f, dtype="<f4", count=vertex_count * 3).reshape(vertex_count, 3)
        return verts


def test_segment_export_mesh_help() -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-mesh", "--help"])
    assert result.exit_code == 0
    assert "--seg-codebook" in result.output
    assert "--spacing" in result.output
    assert "--origin" in result.output


def test_segment_export_mesh_parallel_help() -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-mesh-parallel", "--help"])
    assert result.exit_code == 0
    assert "--workers" in result.output
    assert "--batch-size" in result.output


def test_segment_export_mesh_writes_watertight_box_extents(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    ws = _make_workspace(tmp_path)
    roi = "roi1"
    seg_cb = "seg"

    vol = np.zeros((10, 20, 30), dtype=np.int32)
    vol[2:5, 3:7, 4:9] = 1
    _write_segmentation_zarr(ws, roi, seg_cb, vol)

    spacing_zyx = (2.0, 3.0, 4.0)
    origin_zyx = (10.0, 20.0, 30.0)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-mesh",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--spacing",
            "2,3,4",
            "--origin",
            "10,20,30",
        ],
    )
    assert result.exit_code == 0, result.output

    mesh_path = (
        ws
        / "analysis/deconv"
        / f"stitch--{roi}+{seg_cb}"
        / "output_segmentation.zarr"
        / "mesh.ply"
    )
    assert mesh_path.exists()
    assert mesh_path.with_suffix(".vtp").exists()

    verts = _read_ply_vertices(mesh_path)
    assert verts.shape[1] == 3

    x_start, x_end = 4, 9
    y_start, y_end = 3, 7
    z_start, z_end = 2, 5

    sz, sy, sx = spacing_zyx
    oz, oy, ox = origin_zyx
    expected_min = np.array([ox + x_start * sx, oy + y_start * sy, oz + z_start * sz])
    expected_max = np.array([ox + x_end * sx, oy + y_end * sy, oz + z_end * sz])

    got_min = verts.min(axis=0)
    got_max = verts.max(axis=0)
    assert np.allclose(got_min, expected_min, atol=1e-3)
    assert np.allclose(got_max, expected_max, atol=1e-3)


def test_segment_export_mesh_parallel_writes_watertight_box_extents(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    roi = "roi1"
    seg_cb = "seg"

    vol = np.zeros((10, 20, 30), dtype=np.int32)
    vol[2:5, 3:7, 4:9] = 1
    ws = _make_workspace(tmp_path)
    seg_zarr = _write_segmentation_zarr(ws, roi, seg_cb, vol)

    spacing_xyz = (2.0, 3.0, 4.0)
    origin_xyz = (10.0, 20.0, 30.0)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-mesh-parallel",
            str(seg_zarr),
            "--spacing",
            "2,3,4",
            "--origin",
            "10,20,30",
        ],
    )
    assert result.exit_code == 0, result.output

    mesh_path = seg_zarr / "mesh_parallel.ply"
    assert mesh_path.exists()

    verts = _read_ply_vertices(mesh_path)
    assert verts.shape[1] == 3

    x_start, x_end = 4, 9
    y_start, y_end = 3, 7
    z_start, z_end = 2, 5

    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    expected_min = np.array([ox + (x_start - 0.5) * sx, oy + (y_start - 0.5) * sy, oz + (z_start - 0.5) * sz])
    expected_max = np.array([ox + (x_end - 0.5) * sx, oy + (y_end - 0.5) * sy, oz + (z_end - 0.5) * sz])

    got_min = verts.min(axis=0)
    got_max = verts.max(axis=0)
    assert np.allclose(got_min, expected_min, atol=1e-3)
    assert np.allclose(got_max, expected_max, atol=1e-3)


def test_segment_export_mesh_respects_label_filter(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    ws = _make_workspace(tmp_path)
    roi = "roi1"
    seg_cb = "seg"

    vol = np.zeros((6, 6, 6), dtype=np.int32)
    vol[1:4, 1:4, 1:4] = 1
    vol[2:5, 2:5, 2:5] = 2
    _write_segmentation_zarr(ws, roi, seg_cb, vol)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-mesh",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--labels",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    mesh_path = (
        ws
        / "analysis/deconv"
        / f"stitch--{roi}+{seg_cb}"
        / "output_segmentation.zarr"
        / "mesh.ply"
    )
    assert mesh_path.exists()
    assert mesh_path.with_suffix(".vtp").exists()


def test_segment_export_mesh_downsample_smoke(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    ws = _make_workspace(tmp_path)
    roi = "roi1"
    seg_cb = "seg"

    vol = np.zeros((10, 20, 30), dtype=np.int32)
    vol[2:5, 3:7, 4:9] = 1
    _write_segmentation_zarr(ws, roi, seg_cb, vol)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-mesh",
            str(ws),
            roi,
            "--seg-codebook",
            seg_cb,
            "--downsample",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    mesh_path = (
        ws
        / "analysis/deconv"
        / f"stitch--{roi}+{seg_cb}"
        / "output_segmentation.zarr"
        / "mesh.ply"
    )
    assert mesh_path.exists()
    assert mesh_path.with_suffix(".vtp").exists()


def test_segment_export_mesh_parallel_respects_label_filter(tmp_path: Path) -> None:
    ensure_cellpose_stub()
    from fishtools.segment import app

    roi = "roi1"
    seg_cb = "seg"

    vol = np.zeros((6, 6, 6), dtype=np.int32)
    vol[1:4, 1:4, 1:4] = 1
    vol[2:5, 2:5, 2:5] = 2
    ws = _make_workspace(tmp_path)
    seg_zarr = _write_segmentation_zarr(ws, roi, seg_cb, vol)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-mesh-parallel",
            str(seg_zarr),
            "--labels",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    mesh_path = seg_zarr / "mesh_parallel.ply"
    assert mesh_path.exists()
