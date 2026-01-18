from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from fishtools.io.workspace import Workspace
from fishtools.segment import app as segment_app


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def _touch_segmentation(ws: Path, *, roi: str, seg_codebook: str, name: str) -> Path:
    workspace = Workspace(ws)
    seg_root = workspace.stitch(roi, seg_codebook) / name
    seg_root.mkdir(parents=True, exist_ok=True)
    return seg_root


def test_segment_export_prefers_postproc_over_base(tmp_path: Path, monkeypatch) -> None:
    ws = _make_workspace(tmp_path)
    _touch_segmentation(ws, roi="1", seg_codebook="pi", name="output_segmentation-sam.zarr")
    _touch_segmentation(ws, roi="1", seg_codebook="pi", name="output_segmentation-sam_postproc_s1-2-2_v500.zarr")

    calls: list[str] = []

    import fishtools.segment.export as export_mod

    def _fake_export_cmd(
        *,
        path: Path,
        roi: str | None,
        seg_codebook: str,
        codebooks: tuple[str, ...],
        segmentation_name: str,
        channels: str,
        thumbnail_scale: float,
        diag: bool,
    ) -> None:
        calls.append(segmentation_name)

    monkeypatch.setattr(export_mod, "export_cmd", _fake_export_cmd)

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "export",
            str(ws),
            "--seg-codebook",
            "pi",
            "--codebook",
            "cs_base",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output
    assert calls == ["output_segmentation-sam_postproc_s1-2-2_v500.zarr"]


def test_segment_export_errors_when_postproc_missing(tmp_path: Path, monkeypatch) -> None:
    ws = _make_workspace(tmp_path)
    _touch_segmentation(ws, roi="1", seg_codebook="pi", name="output_segmentation-sam.zarr")

    calls: list[str] = []

    import fishtools.segment.export as export_mod

    def _fake_export_cmd(
        *,
        path: Path,
        roi: str | None,
        seg_codebook: str,
        codebooks: tuple[str, ...],
        segmentation_name: str,
        channels: str,
        thumbnail_scale: float,
        diag: bool,
    ) -> None:
        calls.append(segmentation_name)

    monkeypatch.setattr(export_mod, "export_cmd", _fake_export_cmd)

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "export",
            str(ws),
            "--seg-codebook",
            "pi",
            "--codebook",
            "cs_base",
        ],
        prog_name="segment",
    )
    assert res.exit_code != 0
    assert not calls


def test_segment_export_can_use_base_when_requested(tmp_path: Path, monkeypatch) -> None:
    ws = _make_workspace(tmp_path)
    _touch_segmentation(ws, roi="1", seg_codebook="pi", name="output_segmentation-sam.zarr")

    calls: list[str] = []

    import fishtools.segment.export as export_mod

    def _fake_export_cmd(
        *,
        path: Path,
        roi: str | None,
        seg_codebook: str,
        codebooks: tuple[str, ...],
        segmentation_name: str,
        channels: str,
        thumbnail_scale: float,
        diag: bool,
    ) -> None:
        calls.append(segmentation_name)

    monkeypatch.setattr(export_mod, "export_cmd", _fake_export_cmd)

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "export",
            str(ws),
            "--seg-codebook",
            "pi",
            "--codebook",
            "cs_base",
            "--segmentation-name",
            "output_segmentation-sam.zarr",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output
    assert calls == ["output_segmentation-sam.zarr"]
