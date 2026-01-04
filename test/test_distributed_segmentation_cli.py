from pathlib import Path

import pytest
from click.testing import CliRunner

from fishtools.segmentation.distributed.distributed_segmentation import _collect_input_paths
from fishtools.segmentation.distributed.distributed_segmentation import _run_inputs_with_roi_retries
from fishtools.segmentation.distributed.distributed_segmentation import cli


def test_collect_input_paths_uses_codebook_and_fused_n4(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "OK.DONE").write_text("ok\n")

    stitch_dir = workspace / "analysis" / "deconv" / "stitch--roi3+cb1"
    stitch_dir.mkdir(parents=True)
    (stitch_dir / "fused_n4.zarr").mkdir()

    inputs = _collect_input_paths(workspace, "*", "fused_n4.zarr", "cb1")

    assert inputs == [stitch_dir / "fused_n4.zarr"]


def test_run_inputs_with_roi_retries_retries_per_roi(tmp_path: Path) -> None:
    paths = [
        tmp_path / "analysis" / "deconv" / "stitch--roi1+cb1" / "fused_n4.zarr",
        tmp_path / "analysis" / "deconv" / "stitch--roi2+cb1" / "fused_n4.zarr",
    ]

    attempts: dict[Path, int] = {}

    def run_one(path: Path) -> None:
        attempts[path] = attempts.get(path, 0) + 1
        if attempts[path] == 1:
            raise RuntimeError("boom")

    _run_inputs_with_roi_retries(
        input_paths=paths,
        run_one=run_one,
        roi_retries=1,
        roi_retry_delay_s=0,
    )

    assert attempts == {paths[0]: 2, paths[1]: 2}


def test_run_inputs_with_roi_retries_raises_after_exhausted(tmp_path: Path) -> None:
    paths = [tmp_path / "analysis" / "deconv" / "stitch--roi1+cb1" / "fused_n4.zarr"]

    def run_one(_: Path) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_inputs_with_roi_retries(
            input_paths=paths,
            run_one=run_one,
            roi_retries=1,
            roi_retry_delay_s=0,
        )


def test_distributed_segmentation_click_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
