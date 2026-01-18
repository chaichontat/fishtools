from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rich_click as click
import tifffile

from fishtools.preprocess.config import SpotDecodeConfig
from fishtools.preprocess.spots import align_prod


def test_spots_batch_forwards_config_to_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "workspace.DONE").write_text("ok\n", encoding="utf-8")
    reg_dir = tmp_path / "analysis/deconv/registered--roiA+cb"
    reg_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(reg_dir / "reg-0000.tif", np.zeros((1, 1, 4, 4), dtype=np.uint16))

    cb = tmp_path / "cb.json"
    cb.write_text("{}", encoding="utf-8")

    cfg = tmp_path / "project.json"
    cfg.write_text("{}", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def fake_batch(paths: list[Path], mode: str, args: list[str], **kwargs: object) -> None:
        calls.append({"paths": paths, "mode": mode, "args": args, "kwargs": kwargs})

    monkeypatch.setattr(align_prod, "_batch", fake_batch)

    align_prod.batch.callback(
        path=tmp_path,
        roi="roiA",
        codebook_path=cb,
        threads=1,
        overwrite=False,
        overwrite_stale=False,
        simple=False,
        split=False,
        since=None,
        delete_corrupted=False,
        local_opt=False,
        blank=None,
        json_config=cfg,
        stagger=0.0,
        stagger_jitter=0.0,
        field_correct=False,
    )

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, list)
    assert "--config" in args
    assert cfg.as_posix() in args


def test_spots_run_uses_spot_decode_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "project.json"
    cfg.write_text(
        json.dumps(
            {
                "spot_decode": {
                    "min_intensity": 0.123,
                    "max_distance": 0.9,
                    "min_area": 9,
                    "max_area": 111,
                    "threads": 3,
                    "sigma": [3.0, 4.0, 5.0],
                }
            }
        ),
        encoding="utf-8",
    )

    decode = align_prod._resolve_spot_decode_config(None, json_config=cfg)
    assert decode.min_intensity == pytest.approx(0.123)
    assert decode.max_distance == pytest.approx(0.9)
    assert decode.min_area == 9
    assert decode.max_area == 111
    assert decode.threads == 3
    assert decode.sigma == (3.0, 4.0, 5.0)

    override = SpotDecodeConfig(min_intensity=0.5)
    assert align_prod._resolve_spot_decode_config(override, json_config=cfg) is override


def test_spots_run_invalid_config_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "project.json"
    cfg.write_text(json.dumps({"spot_decode": {"min_area": "nope"}}), encoding="utf-8")

    with pytest.raises(click.ClickException):
        align_prod._resolve_spot_decode_config(None, json_config=cfg)
