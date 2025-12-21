from pathlib import Path

from fishtools.segmentation.distributed.distributed_segmentation import _collect_input_paths


def test_collect_input_paths_uses_codebook_and_fused_n4(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "OK.DONE").write_text("ok\n")

    stitch_dir = workspace / "analysis" / "deconv" / "stitch--roi3+cb1"
    stitch_dir.mkdir(parents=True)
    (stitch_dir / "fused_n4.zarr").mkdir()

    inputs = _collect_input_paths(workspace, "*", "fused_n4.zarr", "cb1")

    assert inputs == [stitch_dir / "fused_n4.zarr"]
