from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess


def test_deconv_prepare_accepts_roi_positional(tmp_path: Path) -> None:
    # Empty workspace; using wildcard ROI should be accepted and no-op
    runner = CliRunner()
    result = runner.invoke(
        preprocess,
        [
            "deconv",
            "prepare",
            str(tmp_path),
            "*",
            "-n",
            "1",
            "--percent",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output
