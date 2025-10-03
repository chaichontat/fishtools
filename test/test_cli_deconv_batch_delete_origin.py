from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner
import numpy as np

from fishtools.preprocess.cli_deconv import deconv as deconv_cli


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_batch_delete_origin_removes_source_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    """--delete-origin removes each {round}--{roi} folder when outputs exist for all inputs.

    We stub the heavy deconvolution pipeline by monkeypatching _run to create
    matching outputs and metadata files. We also stub get_channels to avoid
    reading TIFF metadata and provide a lightweight 'basic' pickle.
    """

    # Workspace layout
    # Create two ROIs with two input files each
    for roi in ("roiA", "roiB"):
        for idx in (1, 2):
            _touch(tmp_path / f"1_2_3--{roi}" / f"1_2_3-{idx:04d}.tif")

    # Minimal 'basic' dependency
    _touch(tmp_path / "basic" / "all-c0.pkl", b"DUMMY")

    # Stubs
    monkeypatch.setattr("fishtools.preprocess.cli_deconv.get_channels", lambda _f: ["c0"])  # single channel
    monkeypatch.setattr("fishtools.preprocess.cli_deconv.get_metadata", lambda _f: {})

    # Don't depend on pickle content; return an object with 'basic' key
    class _FakeBasic:
        def __init__(self) -> None:
            self.darkfield = np.zeros((2048, 2048), dtype=np.float32)
            self.flatfield = np.ones((2048, 2048), dtype=np.float32)

    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv.pickle.loads",
        lambda _b: {"basic": _FakeBasic()},
    )

    # Replace the heavy _run with a stub that writes outputs for every input file
    def _fake_run(files, out, basics, overwrite, n_fids, step=6, debug=False):  # type: ignore[no-untyped-def]
        for f in files:
            dst_dir = out / f.parent.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            _touch(dst_dir / f.name)  # output image
            (dst_dir / f.with_suffix(".deconv.json").name).write_text("{}")

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, [
        "batch",
        str(tmp_path),
        "--basic-name","all",
        "--delete-origin",
    ])

    assert result.exit_code == 0, result.output
    # Input folders should be removed
    assert not (tmp_path / "1_2_3--roiA").exists()
    assert not (tmp_path / "1_2_3--roiB").exists()
    # Outputs retained
    assert (tmp_path / "analysis" / "deconv" / "1_2_3--roiA").exists()
    assert (tmp_path / "analysis" / "deconv" / "1_2_3--roiB").exists()


def test_batch_no_delete_without_flag(monkeypatch: Any, tmp_path: Path) -> None:
    """Without --delete-origin, source directories remain intact."""

    _touch(tmp_path / "1_2_3--roi" / "1_2_3-0001.tif")
    _touch(tmp_path / "basic" / "all-c0.pkl", b"DUMMY")

    monkeypatch.setattr("fishtools.preprocess.cli_deconv.get_channels", lambda _f: ["c0"])  # single channel
    monkeypatch.setattr("fishtools.preprocess.cli_deconv.get_metadata", lambda _f: {})
    monkeypatch.setattr(
        "fishtools.preprocess.cli_deconv.pickle.loads",
        lambda _b: {"basic": object()},
    )

    def _fake_run(files, out, basics, overwrite, n_fids, step=6, debug=False):  # type: ignore[no-untyped-def]
        for f in files:
            dst_dir = out / f.parent.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            _touch(dst_dir / f.name)
            (dst_dir / f.with_suffix(".deconv.json").name).write_text("{}")

    monkeypatch.setattr("fishtools.preprocess.cli_deconv._run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(deconv_cli, [
        "batch",
        str(tmp_path),
        "--basic-name","all",
    ])

    assert result.exit_code == 0, result.output
    # Source should still exist because flag not provided
    assert (tmp_path / "1_2_3--roi").exists()
