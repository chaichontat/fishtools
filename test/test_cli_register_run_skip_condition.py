from pathlib import Path
from typing import Any

import pytest

from click.testing import CliRunner
import fishtools.preprocess.cli_register as cli_register_module


def test_run_does_not_skip_when_only_shifts_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange a minimal workspace under analysis/deconv
    deconv_path = tmp_path / "analysis" / "deconv"
    round_name = "2_10_18"
    roi = "4"
    idx = 55
    (deconv_path / f"{round_name}--{roi}").mkdir(parents=True)
    (deconv_path / f"{round_name}--{roi}" / f"{round_name}-{idx:04d}.tif").touch()

    # Pre-create a shifts JSON (simulating an interrupted earlier run)
    cb_path = tmp_path / "cb.json"
    cb_path.write_text('{"gene": [1]}')
    shifts_dir = deconv_path / f"shifts--{roi}+{cb_path.stem}"
    shifts_dir.mkdir(parents=True, exist_ok=True)
    (shifts_dir / f"shifts-{idx:04d}.json").write_text("{}")

    # Stub the heavy _run implementation; we only need to verify that `run`
    # decides whether to call it based on the presence of the final reg output.
    calls = {"count": 0}

    def _run_stub(*args: Any, **kwargs: Any) -> None:  # signature intentionally broad
        calls["count"] += 1

    monkeypatch.setattr(cli_register_module, "_run", _run_stub)

    # Act: run should not skip, despite pre-existing shifts JSON
    runner = CliRunner()
    result = runner.invoke(
        cli_register_module.register,
        [
            "run",
            str(deconv_path),
            str(idx),
            f"--codebook={cb_path}",
            f"--roi={roi}",
            "--reference",
            round_name,
            "--threshold",
            "3.0",
            "--fwhm",
            "3.0",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # Assert: _run is invoked exactly once
    assert calls["count"] == 1, "Expected run() to invoke _run when only shifts exist"

    # And when the final registered output already exists, run should skip and not call _run
    calls["count"] = 0
    reg_dir = deconv_path / f"registered--{roi}+{cb_path.stem}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / f"reg-{idx:04d}.tif").touch()
    result2 = runner.invoke(
        cli_register_module.register,
        [
            "run",
            str(deconv_path),
            str(idx),
            f"--codebook={cb_path}",
            f"--roi={roi}",
            "--reference",
            round_name,
        ],
        catch_exceptions=False,
    )
    assert result2.exit_code == 0, result2.output
    assert calls["count"] == 0, "Expected run() to skip when reg output already exists"
