import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner


class DummyBasic:
    def __init__(self, darkfield: np.ndarray, flatfield: np.ndarray) -> None:
        self.darkfield = darkfield
        self.flatfield = flatfield


def _write_basic_profiles(base: Path, *, channels: int, shape: tuple[int, int]) -> list[Path]:
    (base / "basic").mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(channels):
        profile = DummyBasic(
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
        )
        path = base / "basic" / f"all-{idx:03d}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"basic": profile}, fh)
        # Also create channel-suffixed copies expected by resolve_basic_paths
        ch = f"{idx:03d}"
        (base / "basic" / f"all-{ch}.pkl").write_bytes(path.read_bytes())
        (base / "basic" / f"1_9_17-{ch}.pkl").write_bytes(path.read_bytes())
        paths.append(path)
    return paths


@pytest.fixture(autouse=True)
def stub_cupy_cuda(monkeypatch):
    import cupy as cp

    class _Runtime:
        def getDeviceCount(self):
            return 1

        class CUDARuntimeError(RuntimeError):
            pass

    cuda = type("_Cuda", (), {"runtime": _Runtime()})
    monkeypatch.setattr(cp, "cuda", cuda, raising=False)


def _patch_deconv_stubs(monkeypatch, channels: int, height: int, width: int) -> None:
    import cupy as cp

    monkeypatch.setattr(
        "fishtools.preprocess.deconv.core.load_projectors_cached",
        lambda step: (
            cp.ones((channels, height, width), dtype=cp.float32),
            cp.ones((channels, height, width), dtype=cp.float32),
        ),
    )
    monkeypatch.setattr(
        "fishtools.preprocess.deconv.core.deconvolve_lucyrichardson_guo",
        lambda payload, projectors, iters: payload,
    )


def test_cli_prepare_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create workspace with one tile and BaSiC profiles
    workspace = tmp_path / "workspace"
    round_name = "1_9_17"
    roi = "roiA"
    tile_dir = workspace / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True)
    H, W = 128, 192
    C = 2
    payload = (np.arange(C * 2 * H * W, dtype=np.uint16)).reshape(C * 2, H, W)
    metadata = {
        "waveform": json.dumps({"params": {"step": 0.6, "powers": ["000", "001"]}}),
    }
    tifffile.imwrite(tile_dir / f"{round_name}-0000.tif", payload, metadata=metadata)
    _write_basic_profiles(workspace, channels=C, shape=(H, W))
    _patch_deconv_stubs(monkeypatch, channels=C, height=H, width=W)

    # Run CLI via Click runner using positional rounds
    from fishtools.preprocess.cli_deconv import deconv as multi_deconv

    runner = CliRunner()
    result = runner.invoke(
        multi_deconv,
        [
            "prepare",
            str(workspace),
            round_name,
            "--devices",
            "0",
            "--n-fids",
            "0",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output

    # Float32 artifacts should exist
    f32 = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}" / f"{round_name}-0000.tif"
    assert f32.exists()
