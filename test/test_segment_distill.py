from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.segment import distill as distill_module


def _write_images(base: Path, stem: str, count: int) -> list[Path]:
    paths: list[Path] = []
    for idx in range(count):
        array = np.full((4, 4), idx + 1, dtype=np.uint16)
        path = base / f"{stem}_{idx}.tif"
        tifffile.imwrite(path, array)
        paths.append(path)
    return paths


def test_segment_distill_cli_samples_and_copies_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path
    models_dir = workspace / "models"
    models_dir.mkdir()

    training_root = workspace / "training"
    sample_a = training_root / "sample_a"
    sample_b = training_root / "sample_b"
    sample_a.mkdir(parents=True)
    sample_b.mkdir(parents=True)

    _write_images(sample_a, "a", 3)
    _write_images(sample_b, "b", 1)

    config = {
        "outdir": "example",
        "model_name": "distill-model",
        "training_paths": ["training"],
        "samples_per_directory": 2,
        "seed": 42,
        "channels": [0, 0],
    }
    config_path = models_dir / "example.json"
    config_path.write_text(json.dumps(config))

    model_path = models_dir / "distill-model"
    model_path.write_text("fake-model")

    teacher_config = {
        "training_paths": ["training/sample_a"],
    }
    teacher_path = models_dir / "distill-model.json"
    teacher_path.write_text(json.dumps(teacher_config))

    (sample_a / "a_0_seg.npy").write_text("mask")

    class DummyModel:
        def __init__(self, *args, **kwargs):  # noqa: D401
            pass

        def eval(self, image, **kwargs):  # type: ignore[no-untyped-def]
            masks = np.ones_like(image, dtype=np.int32)
            return masks, None, None

    def fake_model_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        return DummyModel()

    plan_path = model_path.with_name(f"{model_path.name}.plan")
    plan_path.write_text("plan")

    monkeypatch.setattr(distill_module, "_find_trt_plan", lambda _: (plan_path, "cuda:test"))
    monkeypatch.setattr("fishtools.segment.distill.PackedCellposeModelTRT", fake_model_loader)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "distill",
            str(workspace),
            "example",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output
    assert "sample_b" in result.output.lower()

    output_root = workspace / "example"
    sample_a_outputs = sorted(output_root.glob("training/sample_a/*.tif"))
    sample_b_outputs = sorted(output_root.glob("training/sample_b/*.tif"))

    assert len(sample_a_outputs) == 4  # 2 images + 2 mask files
    assert len(sample_b_outputs) == 2  # 1 image + 1 mask file

    sampled_images = [path for path in sample_a_outputs if not path.name.endswith("_masks.tif")]
    assert len(sampled_images) == 2
    assert all("a_0" not in path.name for path in sampled_images)

    for image_path in sampled_images:
        mask_path = image_path.with_name(f"{image_path.stem}_masks.tif")
        assert mask_path.exists()
        masks = tifffile.imread(mask_path)
        assert np.all(masks == 1)


def test_segment_distill_cli_recurses_nested_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path
    models_dir = workspace / "models"
    models_dir.mkdir()

    training_root = workspace / "training"
    deep_root = training_root / "roi_a" / "deep"
    nested_a = deep_root / "sub1"
    nested_b = deep_root / "sub2"
    nested_c = training_root / "roi_b"
    deep_root.mkdir(parents=True)
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    nested_c.mkdir(parents=True)

    _write_images(deep_root, "root", 2)
    _write_images(nested_a, "na", 3)
    _write_images(nested_b, "nb", 2)
    _write_images(nested_c, "nc", 1)

    config = {
        "outdir": "example",
        "model_name": "nested-model",
        "training_paths": ["training"],
        "samples_per_directory": 3,
        "seed": 7,
        "channels": [0, 0],
    }
    config_path = models_dir / "example.json"
    config_path.write_text(json.dumps(config))

    model_path = models_dir / "nested-model"
    model_path.write_text("fake-model")

    teacher_config = {
        "training_paths": ["training/roi_a", "training/roi_b"],
    }
    teacher_path = models_dir / "nested-model.json"
    teacher_path.write_text(json.dumps(teacher_config))

    (deep_root / "root_0_seg.npy").write_text("mask")
    (nested_b / "nb_0_seg.npy").write_text("mask")

    class DummyModel:
        def eval(self, image, **kwargs):  # type: ignore[no-untyped-def]
            masks = np.ones_like(image, dtype=np.int32)
            return masks, None, None

    plan_path = model_path.with_name(f"{model_path.name}.plan")
    plan_path.write_text("plan")

    monkeypatch.setattr(distill_module, "_find_trt_plan", lambda _: (plan_path, "cuda:test"))
    monkeypatch.setattr(
        "fishtools.segment.distill.PackedCellposeModelTRT",
        lambda *_, **__: DummyModel(),
    )

    recorded_paths: list[Path] = []
    original_process = distill_module._process_image

    def _recording_process(*, image_path: Path, **kwargs):  # type: ignore[no-untyped-def]
        recorded_paths.append(image_path)
        return original_process(image_path=image_path, **kwargs)

    monkeypatch.setattr(distill_module, "_process_image", _recording_process)

    runner = CliRunner()
    result = runner.invoke(
        segment_app,
        [
            "distill",
            str(workspace),
            "example",
        ],
        prog_name="segment",
    )

    assert result.exit_code == 0, result.output

    output_root = workspace / "example"
    source_counts = {
        deep_root: 2,
        nested_a: 3,
        nested_b: 2,
        nested_c: 1,
    }
    teacher_removed = {
        deep_root: 1,
        nested_b: 1,
    }

    for source_dir, available in source_counts.items():
        relative_dir = source_dir.relative_to(workspace)
        directory = output_root / relative_dir
        images = sorted(path for path in directory.glob("*.tif") if not path.name.endswith("_masks.tif"))
        remaining = max(0, available - teacher_removed.get(source_dir, 0))
        expected = min(config["samples_per_directory"], remaining)
        assert len(images) == expected
        for image in images:
            mask = image.with_name(f"{image.stem}_masks.tif")
            assert mask.exists()

    counts = Counter(recorded_paths)
    assert all(value == 1 for value in counts.values())
