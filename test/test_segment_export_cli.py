from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner

from _cellpose_stub import ensure_cellpose_stub


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "analysis/deconv").mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def test_segment_export_runs_thumbnail_and_exports_base_and_postproc(
    tmp_path: Path, monkeypatch
) -> None:
    ensure_cellpose_stub()
    import fishtools.segment.export as segment_export
    from fishtools.segment import app

    ws = _make_workspace(tmp_path)
    roi = "roi1"
    seg_cb = "seg"

    stitch_dir = ws / "analysis/deconv" / f"stitch--{roi}+{seg_cb}"
    (stitch_dir / "output_segmentation-sam.zarr").mkdir(parents=True, exist_ok=True)
    (stitch_dir / "output_segmentation-sam_postproc_s1-2-2_v500.zarr").mkdir(parents=True, exist_ok=True)

    thumbnail_cmds: list[list[str]] = []
    export_calls: list[tuple[Path, str | None, str, str]] = []

    def _fake_run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        thumbnail_cmds.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    def _fake_export_cmd(
        *, path: Path, roi: str | None, seg_codebook: str, segmentation_name: str, **_: object
    ) -> None:
        export_calls.append((path, roi, seg_codebook, segmentation_name))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(segment_export, "export_cmd", _fake_export_cmd)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "export",
            str(ws),
            "--seg-codebook",
            seg_cb,
            "--codebook",
            "gene",
        ],
    )
    assert res.exit_code == 0, res.output

    assert thumbnail_cmds == [["segment", "thumbnail", str(ws), "--codebook", seg_cb]]
    assert sorted(export_calls) == sorted(
        [
            (ws, None, seg_cb, "output_segmentation-sam.zarr"),
            (ws, None, seg_cb, "output_segmentation-sam_postproc_s1-2-2_v500.zarr"),
        ]
    )
