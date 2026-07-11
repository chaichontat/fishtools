from importlib import import_module
from types import SimpleNamespace
from pathlib import Path

from click.testing import CliRunner
import numpy as np
import pytest

from fishtools.segment import app as segment_app
from fishtools.segment.train import TrainConfig

segment_train = import_module("fishtools.segment.train")


def test_run_train_accepts_non_workspace_training_root(tmp_path, monkeypatch):
    training_root = tmp_path / "cellpose-training"
    (training_root / "models").mkdir(parents=True)

    def fake_discover_training_dirs(path: Path, training_paths: list[str]) -> list[Path]:
        assert path == training_root
        assert training_paths == ["sample"]
        return [Path("sample")]

    def fake_concat_output(
        path: Path,
        samples: list[str],
        mask_filter: list[str] | str = "_seg.npy",
        look_one_level_down: bool = False,
    ) -> tuple[list, ...]:
        assert path == training_root
        assert samples == ["sample"]
        return (
            [
                np.zeros((1, 4, 4), dtype=np.float32),
                np.zeros((2, 4, 4), dtype=np.float32),
                np.zeros((3, 4, 4), dtype=np.float32),
            ],
            [
                np.zeros((4, 4), dtype=np.uint16),
                np.zeros((4, 4), dtype=np.uint16),
                np.zeros((4, 4), dtype=np.uint16),
            ],
            [Path("sample/single.tif"), Path("sample/two_channel.tif"), Path("sample/three_channel.tif")],
            None,
            None,
            None,
        )

    def fake_train(out: tuple[list, ...], path: Path, name: str, train_config: TrainConfig):
        assert path == training_root
        train_images = out[0]
        assert [image.shape for image in train_images] == [(4, 4), (2, 4, 4), (2, 4, 4)]
        model_path = path / "models" / name
        model_path.write_bytes(b"trained model")
        return model_path, [0.3, 0.2], None

    monkeypatch.setattr(segment_train, "_discover_training_dirs", fake_discover_training_dirs)
    monkeypatch.setattr(segment_train, "concat_output", fake_concat_output)
    monkeypatch.setattr(segment_train, "_train", fake_train)

    updated = segment_train.run_train(
        "embryonicsheet",
        training_root,
        TrainConfig(
            name="embryonicsheet",
            base_model=None,
            channels=(1, 2),
            training_paths=["sample"],
        ),
    )

    assert updated.train_losses == [0.3, 0.2]
    assert updated.model_md5 is not None
    assert not (training_root / "analysis").exists()


def test_train_cli_skip_trt_sets_config(tmp_path, monkeypatch):
    training_root = tmp_path / "cellpose-training"
    models_path = training_root / "models"
    models_path.mkdir(parents=True)
    (models_path / "embryonicsheet.json").write_text(
        TrainConfig(
            name="embryonicsheet",
            base_model=None,
            channels=(1, 2),
            training_paths=["sample"],
        ).model_dump_json()
    )

    seen: dict[str, bool] = {}

    def fake_run_train(name: str, path: Path, train_config: TrainConfig) -> TrainConfig:
        assert name == "embryonicsheet"
        assert path == training_root
        seen["skip_trt"] = train_config.skip_trt
        return train_config

    monkeypatch.setattr(segment_train, "run_train", fake_run_train)

    result = CliRunner().invoke(
        segment_app,
        ["train", str(training_root), "embryonicsheet", "--skip-trt"],
        prog_name="segment",
    )

    assert result.exit_code == 0
    assert seen == {"skip_trt": True}


def test_train_skips_trt_build_when_configured(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "embryonicsheet"
    model_path.parent.mkdir()
    model_path.write_bytes(b"trained model")

    monkeypatch.setattr(segment_train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(segment_train.torch, "device", lambda name: name)
    monkeypatch.setattr(
        segment_train,
        "CellposeModel",
        lambda **_: SimpleNamespace(net=SimpleNamespace(diam_mean=np.array(30.0))),
    )
    monkeypatch.setattr(
        segment_train,
        "train_seg_transformer",
        lambda *_args, **_kwargs: (model_path, [0.3], None),
    )
    monkeypatch.setattr(
        segment_train,
        "_cleanup_model_artifacts",
        lambda *_args, **_kwargs: pytest.fail("cleanup should not run when skip_trt=True"),
    )
    monkeypatch.setattr(
        segment_train,
        "build_trt_engine",
        lambda *_args, **_kwargs: pytest.fail("TRT build should not run when skip_trt=True"),
    )

    returned_model_path, train_losses, test_losses = segment_train._train(
        (
            [np.zeros((4, 4), dtype=np.float32)],
            [np.zeros((4, 4), dtype=np.uint16)],
            [Path("sample/image.tif")],
            None,
            None,
            None,
        ),
        tmp_path,
        "embryonicsheet",
        TrainConfig(
            name="embryonicsheet",
            base_model="cpsam",
            channels=(1, 2),
            training_paths=["sample"],
            skip_trt=True,
        ),
    )

    assert returned_model_path == model_path
    assert train_losses == [0.3]
    assert test_losses is None
