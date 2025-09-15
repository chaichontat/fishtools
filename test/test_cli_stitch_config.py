from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fishtools.preprocess.cli_stitch import extract_channel, run_imagej
from fishtools.preprocess.config import StitchingConfig


def test_run_imagej_uses_stitching_config(monkeypatch: Any, tmp_path: Path) -> None:
    # Create fake ImageJ binary under a temp $HOME
    home = tmp_path / "home"
    imagej = home / "Fiji.app" / "ImageJ-linux64"
    imagej.parent.mkdir(parents=True)
    imagej.write_text("")

    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    # Capture subprocess invocation and read the macro file path
    recorded: dict[str, str] = {}

    def fake_run(cmd: str, capture_output: bool, check: bool, shell: bool):  # type: ignore[no-untyped-def]
        assert shell is True
        # Extract macro file path
        assert "-macro" in cmd
        macro_path = cmd.split("-macro ", 1)[1].strip()
        with open(macro_path, "r") as f:
            macro = f.read()
        recorded["macro"] = macro

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)

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

    def fake_imwrite(out: Path, img: np.ndarray, compression: int, metadata: dict, compressionargs: dict):  # type: ignore[no-untyped-def]
        recorded["compressionargs"] = compressionargs
        recorded["out"] = out

    monkeypatch.setattr("fishtools.preprocess.cli_stitch.TiffFile", fake_tifffile)
    monkeypatch.setattr("fishtools.preprocess.cli_stitch.imwrite", fake_imwrite)

    sc = StitchingConfig(compression_levels={"low": 0.61, "medium": 0.71, "high": 0.81})

    infile = tmp_path / "reg-0001.tif"
    infile.write_text("")
    outfile = tmp_path / "out.tif"

    extract_channel(infile, outfile, idx=0, sc=sc)

    assert recorded["compressionargs"]["level"] == 0.61
    assert recorded["out"] == outfile
