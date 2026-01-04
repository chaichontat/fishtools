from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from fishtools.segment import app as segment_app


def test_segment_batch_help() -> None:
    runner = CliRunner()
    res = runner.invoke(segment_app, ["batch", "--help"], prog_name="segment")
    assert res.exit_code == 0, res.output


def test_segment_batch_calls_distributed_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    from fishtools.segmentation.distributed import distributed_segmentation as ds

    called: dict[str, object] = {}

    def _fake_callback(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(ds.run, "callback", _fake_callback)

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "batch",
            str(ws),
            "roi1",
            "--codebook",
            "cb1",
            "--workers-per-gpu",
            "2",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    assert called["workspace"] == ws
    assert called["roi"] == "roi1"
    assert called["codebook"] == "cb1"
    assert called["workers_per_gpu"] == 2
