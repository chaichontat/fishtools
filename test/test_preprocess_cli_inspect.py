from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner
from tifffile import imwrite

from fishtools.preprocess.cli_inspect import inspect_cli


def _write_tif(path: Path, *, with_metadata: bool) -> None:
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    metadata = {"axes": "YX", "name": "demo"} if with_metadata else None
    imwrite(path, data, metadata=metadata)


def test_inspect_cli_reports_core_fields(tmp_path: Path) -> None:
    tif_path = tmp_path / "with_meta.tif"
    _write_tif(tif_path, with_metadata=True)

    runner = CliRunner()
    result = runner.invoke(inspect_cli, [str(tif_path)])

    assert result.exit_code == 0, result.output
    assert "TIFF Overview" in result.output
    assert "shape      (2, 3)" in result.output
    assert "dtype      uint16" in result.output
    assert "Shaped Metadata" in result.output
    assert '"axes": "YX"' in result.output


def test_inspect_cli_handles_missing_metadata(tmp_path: Path) -> None:
    tif_path = tmp_path / "without_meta.tif"
    _write_tif(tif_path, with_metadata=False)

    runner = CliRunner()
    result = runner.invoke(inspect_cli, [str(tif_path)])

    assert result.exit_code == 0, result.output
    assert "No shaped metadata present." in result.output


def test_inspect_cli_parses_string_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    tif_path = tmp_path / "string_meta.tif"
    tif_path.write_bytes(b"II*")

    class _Series:
        shape = (2, 3)
        dtype = "uint16"

    class _FakeTiff:
        def __init__(self) -> None:
            self.series = [_Series()]
            self.shaped_metadata = ('{"axes": "YX", "shape": [2, 3]}',)

        def __enter__(self) -> "_FakeTiff":
            return self

        def __exit__(self, *_: Any) -> None:  # type: ignore[override]
            return None

    monkeypatch.setattr("fishtools.preprocess.cli_inspect.TiffFile", lambda *_: _FakeTiff())

    runner = CliRunner()
    result = runner.invoke(inspect_cli, [str(tif_path)])

    assert result.exit_code == 0, result.output
    assert '"axes": "YX"' in result.output
