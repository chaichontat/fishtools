from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from fishtools.preprocess.cli_verify import verify


def _write_good_tiff(path: Path) -> None:
    data = np.zeros((8, 8), dtype=np.uint16)
    tifffile.imwrite(path, data, compression="zlib")


def _write_bad_tiff(path: Path) -> None:
    # Intentionally write invalid bytes so tifffile fails to read
    path.write_bytes(b"NOT_A_TIFF")


def test_verify_path_directory_mixed_files_deletes_by_default(tmp_path: Path) -> None:
    d = tmp_path
    _write_good_tiff(d / "ok.tif")
    (d / "nested").mkdir()
    bad = d / "nested" / "bad.tiff"
    _write_bad_tiff(bad)

    runner = CliRunner()
    result = runner.invoke(verify, ["path", str(d)])

    # One file is corrupted → non-zero exit and deleted by default
    assert result.exit_code == 1
    assert not bad.exists()


def test_verify_path_single_file_ok(tmp_path: Path) -> None:
    f = tmp_path / "single_ok.tif"
    _write_good_tiff(f)

    runner = CliRunner()
    result = runner.invoke(verify, ["path", str(f)])

    assert result.exit_code == 0
    # No stdout expected in quiet run
    assert result.output.strip() == ""


def test_verify_path_no_tiffs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.txt").write_text("hello")

    runner = CliRunner()
    result = runner.invoke(verify, ["path", str(tmp_path)])

    assert result.exit_code != 0
    # Error printed to stderr; just ensure failure


def test_verify_path_no_delete_flag_keeps_files(tmp_path: Path) -> None:
    d = tmp_path
    (d / "dir").mkdir()
    bad = d / "dir" / "bad.tif"
    _write_bad_tiff(bad)

    runner = CliRunner()
    result = runner.invoke(verify, ["path", str(d), "--no-delete"])

    assert result.exit_code == 1
    # File should remain on disk
    assert bad.exists()


def test_verify_path_variadic_multiple_roots(tmp_path: Path) -> None:
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()

    _write_good_tiff(d1 / "ok1.tif")
    _write_bad_tiff(d1 / "bad1.tif")
    _write_good_tiff(d2 / "ok2.tiff")
    _write_bad_tiff(d2 / "bad2.tiff")

    runner = CliRunner()
    result = runner.invoke(verify, ["path", str(d1), str(d2)])

    assert result.exit_code == 1
    assert not (d1 / "bad1.tif").exists()
    assert not (d2 / "bad2.tiff").exists()
