from __future__ import annotations

import json
from pathlib import Path

import pytest

from fishtools.preprocess.config_loader import (
    load_config_from_json,
    load_config,
)


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _minimal_registration(data_path: Path) -> dict:
    return {
        # dataPath is injected by the loader; include chromatic_path for determinism
        "registration": {
            "chromatic_path": str(data_path),
        }
    }


def test_load_config_from_json_valid(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    data = _minimal_registration(tmp_path)
    _write_json(cfg_path, data)

    cfg = load_config_from_json(cfg_path, data_path=str(tmp_path))

    assert cfg.dataPath == str(tmp_path)
    assert cfg.registration.reference == "2_10_18"  # default
    assert cfg.registration.chromatic_path == tmp_path


def test_load_config_from_json_with_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    data = _minimal_registration(tmp_path)
    _write_json(cfg_path, data)

    cfg = load_config_from_json(cfg_path, data_path=str(tmp_path), fwhm=5.0, threshold=4.0)
    assert cfg.registration.fiducial.fwhm == 5.0
    assert cfg.registration.fiducial.threshold == 4.0


def test_load_config_autodetect_json(tmp_path: Path) -> None:
    # load_config should choose JSON by suffix
    cfg_path = tmp_path / "project.json"
    _write_json(cfg_path, _minimal_registration(tmp_path))
    cfg = load_config(cfg_path, data_path=str(tmp_path))
    assert cfg.registration.reference == "2_10_18"


def test_load_config_json_errors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "broken.json"
    cfg_path.write_text("{ invalid json")

    with pytest.raises(ValueError):
        load_config_from_json(cfg_path, data_path=str(tmp_path))

    with pytest.raises(Exception):
        load_config(cfg_path, data_path=str(tmp_path))  # autodetect should error as well
