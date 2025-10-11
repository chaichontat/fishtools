from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

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


def _build_prenorm_waveform(z_planes: int, n_fids: int) -> dict[str, Any]:
    waveform: dict[str, Any] = {
        "params": {
            "powers": {
                "ilm560": 1.1,
                "ilm650": 2.2,
                "ilm750": 3.3,
            }
        }
    }
    for channel in DEFAULT_CHANNELS:
        if channel == "ilm560" or channel == "ilm650" or channel == "ilm750":
            count = z_planes
            power = waveform["params"]["powers"][channel]
        elif channel == "ilm405":
            count = n_fids
            power = 5.0
        else:
            count = 0
            power = 0.0
        waveform[channel] = {"sequence": [count], "power": power}
    return waveform


def _write_prenorm_tile(
    path: Path,
    *,
    z_planes: int,
    n_fids: int,
    image_size: int,
    metadata: dict[str, Any],
) -> None:
    payload = np.arange(
        (z_planes * 3 + n_fids) * image_size * image_size,
        dtype=np.uint16,
    ).reshape(z_planes * 3 + n_fids, image_size, image_size)
    tifffile.imwrite(path, payload, metadata=metadata)


def test_load_image_components_prenormalized_uses_metadata_scaling(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "analysis" / "deconv" / "round--roi"
    workspace.mkdir(parents=True)
    tiff_path = workspace / "560_650_750-0000.tif"

    z_planes = 2
    n_fids = 2
    image_size = 4

    metadata = {
        "waveform": json.dumps(_build_prenorm_waveform(z_planes, n_fids)),
        "prenormalized": True,
        "deconv_min": [0.1, 0.2, 0.3],
        "deconv_scale": [1.1, 1.2, 1.3],
    }

    _write_prenorm_tile(
        tiff_path,
        z_planes=z_planes,
        n_fids=n_fids,
        image_size=image_size,
        metadata=metadata,
    )

    components = load_image_components(tiff_path, n_fids=n_fids, image_size=image_size)

    assert components.metadata["prenormalized"] is True
    assert components.nofid.shape == (z_planes, 3, image_size, image_size)
    expected = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float32)
    np.testing.assert_allclose(components.global_deconv_scaling, expected)


def test_load_image_components_prenormalized_missing_metadata_defaults(tmp_path: Path) -> None:
    workspace = tmp_path / "ws" / "analysis" / "deconv" / "round--roi"
    workspace.mkdir(parents=True)
    tiff_path = workspace / "560_650_750-0001.tif"

    z_planes = 2
    n_fids = 2
    image_size = 4

    metadata = {
        "waveform": json.dumps(_build_prenorm_waveform(z_planes, n_fids)),
        "prenormalized": True,
    }

    _write_prenorm_tile(
        tiff_path,
        z_planes=z_planes,
        n_fids=n_fids,
        image_size=image_size,
        metadata=metadata,
    )

    components = load_image_components(tiff_path, n_fids=n_fids, image_size=image_size)

    assert components.metadata["prenormalized"] is True
    np.testing.assert_allclose(
        components.global_deconv_scaling,
        np.array([
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ]),
    )
