from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_stitch import stitch


def test_stitch_highpass_writes_fused_highpassed_uint16(tmp_path: Path) -> None:
    (tmp_path / "workspace.DONE").write_text("")
    deconv_dir = tmp_path / "analysis" / "deconv"
    deconv_dir.mkdir(parents=True, exist_ok=True)

    ws = Workspace(deconv_dir)
    stitch_dir = ws.stitch("roi_a", "cb1")
    stitch_dir.mkdir(parents=True, exist_ok=True)

    src_path = stitch_dir / "fused.zarr"
    src = zarr.open_array(
        str(src_path),
        mode="w",
        shape=(2, 32, 32, 2),
        chunks=(1, 16, 16, 1),
        dtype=np.uint16,
    )
    src.attrs["axes"] = "ZYXC"
    src.attrs["key"] = ["ch0", "ch1"]

    # Smooth background + a bright spot -> highpass should keep the spot and suppress background.
    yy, xx = np.mgrid[:32, :32]
    background = (1000 + 3 * yy + 2 * xx).astype(np.uint16)
    img = np.stack([background, background], axis=-1)  # YXC
    stack = np.stack([img, img], axis=0)  # ZYXC
    stack[0, 16, 16, 0] = 60000
    stack[1, 10, 22, 1] = 50000
    src[:] = stack

    runner = CliRunner()
    result = runner.invoke(
        stitch,
        [
            "highpass",
            str(deconv_dir),
            "roi_a",
            "--codebook",
            "cb1",
            "--highpass-px",
            "2",
            "--anisotropy",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    out_path = stitch_dir / "fused_highpassed.zarr"
    assert out_path.exists()
    out = zarr.open_array(str(out_path), mode="r")
    assert out.shape == src.shape
    assert out.dtype == np.uint16
    assert out.attrs["axes"] == "ZYXC"
    assert "highpass" in out.attrs
    assert int(out[:].max()) > 0
    assert (stitch_dir / "thumbnails" / "thumbnail_highpass_z000.png").exists()
