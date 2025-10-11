from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fishtools.preprocess.cli_register as cli_register_module
from fishtools.preprocess.config import Config, Fiducial, RegisterConfig


def test_cli_register_calls_gpu_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange: workspace structure
    deconv_path = tmp_path / "analysis" / "deconv"
    round_name = "1_2"
    roi = "roiA"
    (deconv_path / f"{round_name}--{roi}").mkdir(parents=True)
    # Tile presence required for Workspace.img(...).exists()
    (deconv_path / f"{round_name}--{roi}" / f"{round_name}-0002.tif").touch()

    # Minimal codebook with one bit matching our fake Image.bits
    codebook_path = tmp_path / "cb.json"
    codebook_path.write_text("{""gene"": [1]}")

    # Fake Image loader
    def fake_from_file(*args: Any, **kwargs: Any) -> SimpleNamespace:
        bits = ["1"]
        return SimpleNamespace(
            name=round_name,
            idx=2,
            nofid=np.zeros((1, len(bits), 8, 8), dtype=np.float32),
            fid=np.zeros((8, 8), dtype=np.float32),
            fid_raw=np.zeros((8, 8), dtype=np.float32),
            bits=bits,
            powers=[560],  # channel metadata -> "560"
            metadata={"prenormalized": False},
            global_deconv_scaling=np.ones((2, len(bits)), dtype=np.float32),
            basic=lambda: None,
        )

    # Fake fiducial alignment to avoid invoking heavier logic
    def fake_run_fiducial(path: Path, fids: dict[str, np.ndarray], *args: Any, **kwargs: Any):
        return {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}

    # Lightweight Affine stub
    class DummyAffine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ref_image = None

        def __call__(self, img: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:  # type: ignore[override]
            return img

    # Count GPU cleanup invocations
    calls = {"count": 0}

    def _release() -> None:
        calls["count"] += 1

    # Bypass actual GPU work; return a dtype/shape compatible array
    def _gpu_downsample_xy(arr: np.ndarray, **_: Any) -> np.ndarray:
        return np.clip(arr, 0, 65534).astype(np.uint16)

    monkeypatch.setattr(cli_register_module.Image, "from_file", staticmethod(fake_from_file))
    monkeypatch.setattr(cli_register_module, "run_fiducial", fake_run_fiducial)
    monkeypatch.setattr(cli_register_module, "Affine", DummyAffine)
    monkeypatch.setattr(cli_register_module.tifffile, "imwrite", lambda *a, **k: None)
    monkeypatch.setattr(cli_register_module, "gpu_release_all", _release)
    monkeypatch.setattr(cli_register_module, "gpu_downsample_xy", _gpu_downsample_xy)

    # Downsample>1 to trigger GPU path
    config = Config(
        dataPath=str(tmp_path),
        registration=RegisterConfig(
            chromatic_shifts={"650": "dummy", "750": "dummy"},
            fiducial=Fiducial(use_fft=False, fwhm=3.0, threshold=3.0, n_fids=1),
            downsample=2,
            crop=0,
            slices=slice(None),
            reduce_bit_depth=0,
        ),
    )

    # Act
    cli_register_module._run(
        deconv_path,
        roi,
        2,
        codebook=codebook_path,
        reference=round_name,
        config=config,
        debug=False,
        overwrite=True,
        no_priors=True,
    )

    # Assert
    assert calls["count"] >= 1, "Expected GPU cleanup to be invoked at least once"

