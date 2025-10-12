from __future__ import annotations

import sys
import types
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from click.testing import CliRunner
from skimage import filters as sk_filters

# Ensure cucim.skimage modules exist and proxy to skimage for filters
if "cucim.skimage.filters" not in sys.modules:
    cucim_module = sys.modules.setdefault("cucim", types.ModuleType("cucim"))
    cucim_skimage = sys.modules.setdefault("cucim.skimage", types.ModuleType("cucim.skimage"))
    cucim_filters = types.ModuleType("cucim.skimage.filters")
    cucim_filters.unsharp_mask = sk_filters.unsharp_mask
    cucim_skimage.filters = cucim_filters
    cucim_module.skimage = getattr(cucim_module, "skimage", cucim_skimage)
    sys.modules["cucim.skimage.filters"] = cucim_filters

from fishtools.preprocess import n4
from fishtools.preprocess.cli_stitch import stitch

if not hasattr(cp, "ElementwiseKernel"):
    cp.ElementwiseKernel = lambda *args, **kwargs: None  # type: ignore[assignment]

import cupyx.scipy.ndimage as cupyx_ndimage

try:
    from cucim.skimage import transform as cucim_transform  # type: ignore[import-not-found]
except ModuleNotFoundError:
    cucim_module = types.ModuleType("cucim")
    cucim_skimage = types.ModuleType("cucim.skimage")
    cucim_transform = types.ModuleType("cucim.skimage.transform")
    cucim_module.skimage = cucim_skimage
    cucim_skimage.transform = cucim_transform
    sys.modules["cucim"] = cucim_module
    sys.modules["cucim.skimage"] = cucim_skimage
    sys.modules["cucim.skimage.transform"] = cucim_transform
else:
    cucim_module = sys.modules.setdefault("cucim", types.ModuleType("cucim"))
    cucim_skimage = sys.modules.setdefault("cucim.skimage", types.ModuleType("cucim.skimage"))
    setattr(cucim_module, "skimage", cucim_skimage)
    setattr(cucim_skimage, "transform", cucim_transform)

if not hasattr(cupyx_ndimage, "rank_filter"):
    cupyx_ndimage.rank_filter = lambda image, *args, **kwargs: np.asarray(image)  # type: ignore[assignment]

if not hasattr(cupyx_ndimage, "uniform_filter"):
    cupyx_ndimage.uniform_filter = lambda image, *args, **kwargs: np.asarray(image)  # type: ignore[assignment]

cupyx_module = sys.modules.get("cupyx")
cupyx_scipy_module = sys.modules.get("cupyx.scipy")
if cupyx_module is not None and cupyx_scipy_module is None:
    cupyx_scipy_module = types.ModuleType("cupyx.scipy")
    sys.modules["cupyx.scipy"] = cupyx_scipy_module
    setattr(cupyx_module, "scipy", cupyx_scipy_module)

if cupyx_scipy_module is not None:
    if not hasattr(cupyx_scipy_module, "fft"):
        cupyx_scipy_module.fft = types.SimpleNamespace(
            fft=lambda *args, **kwargs: np.fft.fft(*args, **kwargs)
        )
    if not hasattr(cupyx_scipy_module, "ndimage"):
        cupyx_scipy_module.ndimage = cupyx_ndimage


def _downscale_local_mean(image: np.ndarray, factors: tuple[int, ...], **_: object) -> np.ndarray:
    return np.asarray(image)


setattr(cucim_transform, "downscale_local_mean", _downscale_local_mean)


pytestmark = pytest.mark.timeout(30)


def test_stitch_n4_subcommand_invokes_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def fake_run_cli_workflow(**kwargs: object) -> list[n4.N4Result]:
        captured.update(kwargs)
        return [n4.N4Result(field_path=workspace / "field.tif", corrected_path=None)]

    monkeypatch.setattr("fishtools.preprocess.cli_stitch.run_cli_workflow", fake_run_cli_workflow)

    runner = CliRunner()
    result = runner.invoke(
        stitch,
        [
            "n4",
            str(workspace),
            "roi_a",
            "--codebook",
            "cb1",
            "--z-index",
            "0",
            "--field-only",
            "--unsharp-mask",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == workspace
    assert captured["roi"] == "roi_a"
    assert captured["codebook"] == "cb1"
    assert captured["apply_correction"] is False
    assert captured["use_unsharp_mask"] is True
    assert "Correction field saved to" in result.output
    assert "Corrected imagery saved to" not in result.output


def test_stitch_n4_subcommand_wraps_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def fake_run_cli_workflow(**kwargs: object) -> list[n4.N4Result]:  # pragma: no cover - error path
        raise ValueError("missing fused.zarr")

    monkeypatch.setattr("fishtools.preprocess.cli_stitch.run_cli_workflow", fake_run_cli_workflow)

    runner = CliRunner()
    result = runner.invoke(
        stitch,
        [
            "n4",
            str(workspace),
            "roi_a",
            "--codebook",
            "cb1",
            "--z-index",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "missing fused.zarr" in result.output
