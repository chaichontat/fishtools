from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner

from fishtools.segment import app as segment_app


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def _write_fused(ws: Path, *, roi: str, codebook: str, name: str, shape: tuple[int, int, int, int]) -> Path:
    out_dir = ws / "analysis/deconv" / f"stitch--{roi}+{codebook}"
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / name
    arr = zarr.open_array(store_path, mode="w", shape=shape, chunks=shape, dtype=np.uint16)
    arr[...] = np.arange(arr.size, dtype=np.uint16).reshape(shape)
    return store_path


def test_segment_thumbnail_respects_z_range_and_overwrite(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(5, 4, 4, 2))

    thumb_dir = ws / "analysis/output/thumbnails" / "roi+cb1"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = thumb_dir / "thumbnail_z001.png"
    sentinel_path.write_bytes(b"sentinel")

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-stride",
            "1",
            "--z-range",
            "1:3",
            "--downsample",
            "1",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    assert sentinel_path.read_bytes() == b"sentinel"
    created = thumb_dir / "thumbnail_z002.png"
    assert created.exists()
    assert created.read_bytes() != b"sentinel"

    res_overwrite = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-stride",
            "1",
            "--z-range",
            "1:3",
            "--downsample",
            "1",
            "--overwrite",
        ],
        prog_name="segment",
    )
    assert res_overwrite.exit_code == 0, res_overwrite.output
    assert sentinel_path.read_bytes() != b"sentinel"


def test_segment_thumbnail_include_n4(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(2, 4, 4, 1))
    _write_fused(ws, roi="roi", codebook="cb1", name="fused_n4.zarr", shape=(2, 4, 4, 1))

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--include-n4",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    thumb_dir = ws / "analysis/output/thumbnails" / "roi+cb1"
    assert (thumb_dir / "thumbnail_z000.png").exists()
    assert (thumb_dir / "thumbnail_n4_z000.png").exists()

