from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from fishtools.preprocess.cli_stitch import extract_channel, run_imagej
from fishtools.preprocess.config import StitchingConfig


def test_run_imagej_uses_stitching_config(monkeypatch: Any, tmp_path: Path) -> None:
    # Create fake ImageJ binary and set IMAGEJ_PATH env var
    imagej = tmp_path / "ImageJ-linux64"
    imagej.write_text("")

    monkeypatch.setenv("IMAGEJ_PATH", str(imagej))

    # Capture subprocess invocation and read the macro file path
    recorded: dict[str, str] = {}

    class _FakeProc:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(cmd, **kwargs):  # noqa: ANN001
        macro_path = Path(cmd[cmd.index("-macro") + 1])
        recorded["macro"] = macro_path.read_text(encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr("fishtools.preprocess.imagej.subprocess.Popen", fake_popen)
    monkeypatch.setattr("fishtools.preprocess.imagej.register_subprocess", lambda proc: None)
    monkeypatch.setattr("fishtools.preprocess.imagej.unregister_subprocess", lambda proc: None)
    monkeypatch.setattr("fishtools.preprocess.imagej.get_cancel_event", lambda: threading.Event())

    sc = StitchingConfig(
        max_memory_mb=2048,
        parallel_threads=16,
        fusion_thresholds={"regression": 0.33, "displacement_max": 1.2, "displacement_abs": 2.2},
    )

    (tmp_path / "TileConfiguration.registered.txt").write_text("")

    run_imagej(
        tmp_path,
        compute_overlap=True,
        fuse=True,
        threshold=None,  # should fall back to sc.fusion_thresholds["regression"]
        name="TileConfiguration",
        capture_output=False,
        sc=sc,
    )

    macro = recorded["macro"]
    assert "maximum=2048 parallel=16" in macro
    assert "regression_threshold=0.33" in macro
    assert "max/avg_displacement_threshold=1.2" in macro
    assert "absolute_displacement_threshold=2.2" in macro


def test_extract_channel_uses_compression_level(monkeypatch: Any, tmp_path: Path) -> None:
    # Fake TiffFile with one page returning a 3D array so we take the [idx] slice path
    class _Page:
        def asarray(self):
            return np.zeros((3, 10, 10), dtype=np.uint16)

    class _Tiff:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    recorded: dict[str, Any] = {}

    def fake_tifffile(path: Path):  # type: ignore[no-untyped-def]
        return _Tiff()

    def fake_safe_imwrite(out: Path, img: np.ndarray, *, compression: int, metadata: dict, compressionargs: dict):  # type: ignore[no-untyped-def]
        recorded["compressionargs"] = compressionargs
        recorded["out"] = out

    monkeypatch.setattr("fishtools.preprocess.cli_stitch.TiffFile", fake_tifffile)
    monkeypatch.setattr("fishtools.preprocess.cli_stitch.safe_imwrite", fake_safe_imwrite)

    sc = StitchingConfig(compression_levels={"low": 0.61, "medium": 0.71, "high": 0.81})

    infile = tmp_path / "reg-0001.tif"
    infile.write_text("")
    outfile = tmp_path / "out.tif"

    extract_channel(infile, outfile, idx=0, sc=sc)

    assert recorded["compressionargs"]["level"] == 0.61
    assert recorded["out"] == outfile
