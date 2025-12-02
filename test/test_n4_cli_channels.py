from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from fishtools.preprocess import n4
from fishtools.utils.zarr_utils import default_zarr_codecs


pytestmark = pytest.mark.timeout(30)


def _make_fused(workspace: Path, roi: str, codebook: str, shape=(1, 8, 8, 3), names=None) -> Path:
    stitch_dir = workspace / f"analysis/deconv/stitch--{roi}+{codebook}"
    stitch_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open_array(
        stitch_dir / "fused.zarr",
        mode="w",
        shape=shape,
        chunks=(1, shape[1], shape[2], 1),
        dtype=np.uint16,
        codecs=default_zarr_codecs(),
    )
    store[...] = 100
    if names is not None:
        store.attrs["key"] = list(names)
    return stitch_dir


def test_cli_channels_default_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _make_fused(ws, "roi", "cb", shape=(1, 8, 8, 3), names=["polyA", "reddot", "wga"])

    captured: dict = {}

    def _fake_run(cfg: n4.N4RuntimeConfig) -> List[n4.N4Result]:
        captured["channels"] = cfg.channels
        # return minimal fake result list
        return [n4.N4Result(field_path=ws / "f.tif", corrected_path=None)] * len(cfg.channels or [])

    monkeypatch.setattr(n4, "compute_fields_from_workspace", _fake_run)

    runner = CliRunner()
    res = runner.invoke(n4.app, [str(ws), "roi", "--codebook", "cb", "--z-index", "0", "--field-only"])
    assert res.exit_code == 0, res.output
    assert captured["channels"] == (0, 1, 2)


def test_cli_channels_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _make_fused(ws, "roi", "cb", shape=(1, 8, 8, 3), names=["polyA", "reddot", "wga"])

    captured: dict = {}

    def _fake_run(cfg: n4.N4RuntimeConfig) -> List[n4.N4Result]:
        captured["channels"] = cfg.channels
        return [n4.N4Result(field_path=ws / "f.tif", corrected_path=None)] * len(cfg.channels or [])

    monkeypatch.setattr(n4, "compute_fields_from_workspace", _fake_run)

    runner = CliRunner()
    res = runner.invoke(
        n4.app,
        [str(ws), "roi", "--codebook", "cb", "--channels", "polyA,reddot", "--z-index", "0", "--field-only"],
    )
    assert res.exit_code == 0, res.output
    assert captured["channels"] == (0, 1)


def test_cli_channels_numeric_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _make_fused(ws, "roi", "cb", shape=(1, 8, 8, 3), names=None)  # no names, numeric allowed

    captured: dict = {}

    def _fake_run(cfg: n4.N4RuntimeConfig) -> List[n4.N4Result]:
        captured["channels"] = cfg.channels
        return [n4.N4Result(field_path=ws / "f.tif", corrected_path=None)] * len(cfg.channels or [])

    monkeypatch.setattr(n4, "compute_fields_from_workspace", _fake_run)

    runner = CliRunner()
    res = runner.invoke(
        n4.app,
        [str(ws), "roi", "--codebook", "cb", "--channels", "2,0", "--z-index", "0", "--field-only"],
    )
    assert res.exit_code == 0, res.output
    assert captured["channels"] == (2, 0)
