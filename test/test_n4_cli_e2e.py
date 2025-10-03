from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tifffile import imread
from typer.testing import CliRunner

from fishtools.preprocess import n4


def test_cli_field_only_writes_field_quick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: workspace layout and a placeholder fused.zarr directory so path checks pass
    ws = tmp_path / "ws"
    stitch_dir = ws / "analysis/deconv/stitch--roi+cb"
    stitch_dir.mkdir(parents=True)
    (stitch_dir / "fused.zarr").mkdir()

    # Avoid heavy N4 run; return neutral field quickly
    monkeypatch.setattr(n4, "DEFAULT_ITERATIONS", (1, 1, 1, 1))
    monkeypatch.setattr(n4, "compute_correction_field", lambda *a, **k: np.ones((12, 10), dtype=np.float32))

    # Provide a tiny dummy fused store via monkeypatched zarr.open_array
    class Dummy:
        shape = (1, 12, 10, 1)
        dtype = np.uint16

        @property
        def ndim(self) -> int:  # pragma: no cover - shape implies ndim
            return 4

        def __getitem__(self, idx):
            z, y, x, c = idx
            return np.full((12, 10), 100, dtype=np.float32)

        @property
        def attrs(self):
            return {"key": ["ch0"]}

    monkeypatch.setattr(n4.zarr, "open_array", lambda *a, **k: Dummy())

    runner = CliRunner()
    res = runner.invoke(
        n4.app,
        [str(ws), "roi", "--codebook", "cb", "--z-index", "0", "--field-only", "--shrink", "2", "--spline-lowres-px", "16"],
    )

    # Assert: CLI succeeded and wrote default field path
    assert res.exit_code == 0, res.output
    field_path = stitch_dir / "n4-fields/n4_correction_field.tif"
    assert field_path.exists()
    field = imread(field_path)
    # CYX or YX depending on writer collapsing C=1; both accepted
    if field.ndim == 3:
        assert field.shape[0] == 1 and field.shape[1:] == (12, 10)
    else:
        assert field.shape == (12, 10)

