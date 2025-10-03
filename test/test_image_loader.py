from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fishtools.preprocess.image_loader import DEFAULT_CHANNELS, ImageComponents, load_image_components


def _build_waveform(powers: dict[str, float]) -> dict[str, Any]:
    waveform: dict[str, Any] = {}
    for channel in DEFAULT_CHANNELS:
        power = powers.get(channel[3:], 0.0)
        waveform[channel] = {"sequence": [1 if power else 0], "power": power}
    return waveform


def test_load_image_components_basic(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "analysis" / "deconv" / "round--roi"
    workspace.mkdir(parents=True)
    tiff_path = workspace / "geneA_geneB-0000.tif"

    img = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
    metadata = {"waveform": json.dumps(_build_waveform({"560": 2.0, "650": 3.0}))}

    class DummyTiff:
        def __init__(self, path: Path) -> None:
            assert path == tiff_path

        def __enter__(self) -> "DummyTiff":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def asarray(self) -> np.ndarray:
            return img

        @property
        def shaped_metadata(self) -> list[dict[str, Any]]:
            return [metadata]

        @property
        def imagej_metadata(self) -> dict[str, Any]:
            return metadata

    monkeypatch.setattr("fishtools.preprocess.image_loader.TiffFile", DummyTiff)

    scaling_dir = workspace.parent / "deconv_scaling"
    scaling_dir.mkdir(parents=True)
    np.savetxt(scaling_dir / "geneA_geneB.txt", np.ones((2, 2)))

    components = load_image_components(tiff_path, image_size=4)

    assert isinstance(components, ImageComponents)
    assert components.name == "geneA_geneB"
    assert components.idx == 0
    assert components.nofid.shape == (1, 2, 4, 4)
    assert list(components.bits) == ["geneA", "geneB"]
    assert components.powers == {"560": 2.0, "650": 3.0}
    assert components.basic_loader() is None


def test_load_image_components_discards(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "analysis" / "deconv" / "round--roi"
    workspace.mkdir(parents=True)
    tiff_path = workspace / "geneA_geneB-0000.tif"

    img = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
    metadata = {"waveform": json.dumps(_build_waveform({"560": 2.0, "650": 3.0}))}

    class DummyTiff:
        def __init__(self, path: Path) -> None:
            assert path == tiff_path

        def __enter__(self) -> "DummyTiff":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def asarray(self) -> np.ndarray:
            return img

        @property
        def shaped_metadata(self) -> list[dict[str, Any]]:
            return [metadata]

        @property
        def imagej_metadata(self) -> dict[str, Any]:
            return metadata

    monkeypatch.setattr("fishtools.preprocess.image_loader.TiffFile", DummyTiff)

    scaling_dir = workspace.parent / "deconv_scaling"
    scaling_dir.mkdir(parents=True)
    np.savetxt(scaling_dir / "geneA_geneB.txt", np.ones((2, 2)))

    components = load_image_components(
        tiff_path,
        discards={"geneA": ["geneA_geneB"]},
        image_size=4,
    )

    assert components.nofid.shape == (1, 1, 4, 4)
    assert list(components.bits) == ["geneB"]
    assert components.global_deconv_scaling.shape == (2, 1)
