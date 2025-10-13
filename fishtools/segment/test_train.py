import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from fishtools.segment import app
from fishtools.segment.train import TrainConfig, concat_output, run_train

PATCH_LOAD_TRAIN_TEST_DATA = "cellpose.io.load_train_test_data"


@pytest.fixture
def sample_train_config_data() -> dict:
    return {
        "name": "test_model",
        "channels": (0, 1),
        "training_paths": ["sample_data1", "sample_data2"],
        "include": [],
        "exclude": [],
        "n_epochs": 10,
        "learning_rate": 0.01,
        "batch_size": 8,
        "bsize": 128,
        "weight_decay": 1e-6,
        "SGD": True,
        "normalization_percs": (0.5, 99.0),
        "train_losses": [],
    }


@pytest.fixture
def sample_train_config(sample_train_config_data: dict) -> TrainConfig:
    return TrainConfig(**sample_train_config_data)


@pytest.fixture(autouse=True)
def stub_cellpose_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    cellpose_module = types.ModuleType("cellpose")
    cellpose_io = types.ModuleType("cellpose.io")
    cellpose_train = types.ModuleType("cellpose.train")
    cellpose_models = types.ModuleType("cellpose.models")

    def _noop(*_: object, **__: object) -> None:
        raise RuntimeError("cellpose stub should be mocked in tests")

    cellpose_io.load_train_test_data = _noop  # type: ignore[attr-defined]
    cellpose_train.train_seg = _noop  # type: ignore[attr-defined]

    class _StubModel:
        net: object = object()

        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - trivial stub
            self.args = args
            self.kwargs = kwargs

    cellpose_models.CellposeModel = _StubModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cellpose", cellpose_module)
    monkeypatch.setitem(sys.modules, "cellpose.io", cellpose_io)
    monkeypatch.setitem(sys.modules, "cellpose.train", cellpose_train)
    monkeypatch.setitem(sys.modules, "cellpose.models", cellpose_models)


class TestConcatOutput:
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_concat_output_multiple_samples_str_mask(self, mock_load_data: MagicMock, tmp_path: Path):
        base_path = tmp_path / "train_data"
        base_path.mkdir()
        samples = ["s1", "s2"]

        mock_load_data.side_effect = [
            (["img_s1"], ["lbl_s1"], ["name_s1"], ["test_img_s1"], ["test_lbl_s1"], ["test_name_s1"]),
            (["img_s2"], ["lbl_s2"], ["name_s2"], ["test_img_s2"], ["test_lbl_s2"], ["test_name_s2"]),
        ]

        result = concat_output(base_path, samples, mask_filter="_seg.npy", one_level_down=False)

        assert result[0] == ["img_s1", "img_s2"]
        assert result[1] == ["lbl_s1", "lbl_s2"]
        assert result[2] == ["name_s1", "name_s2"]
        assert result[3] == ["test_img_s1", "test_img_s2"]
        assert result[4] == ["test_lbl_s1", "test_lbl_s2"]
        assert result[5] == ["test_name_s1", "test_name_s2"]

        expected_calls = [
            call(str(base_path / "s1"), mask_filter="_seg.npy", look_one_level_down=False),
            call(str(base_path / "s2"), mask_filter="_seg.npy", look_one_level_down=False),
        ]
        mock_load_data.assert_has_calls(expected_calls)

    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_concat_output_list_mask_one_level_down(self, mock_load_data: MagicMock, tmp_path: Path):
        base_path = tmp_path / "train_data"
        base_path.mkdir()
        samples = ["s1", "s2"]
        mask_filters = ["_s1_seg.npy", "_s2_seg.npy"]

        mock_load_data.side_effect = [
            (["img_s1"], ["lbl_s1"], ["name_s1"], None, None, None),
            (["img_s2"], ["lbl_s2"], ["name_s2"], None, None, None),
        ]

        result = concat_output(base_path, samples, mask_filter=mask_filters, one_level_down=True)

        assert result[0] == ["img_s1", "img_s2"]
        assert result[1] == ["lbl_s1", "lbl_s2"]

        expected_calls = [
            call(str(base_path / "s1"), mask_filter="_s1_seg.npy", look_one_level_down=True),
            call(str(base_path / "s2"), mask_filter="_s2_seg.npy", look_one_level_down=True),
        ]
        mock_load_data.assert_has_calls(expected_calls)

    def test_concat_output_mask_filter_list_len_mismatch(self, tmp_path: Path):
        base_path = tmp_path / "train_data"
        samples = ["s1", "s2"]
        mask_filters = ["_s1_seg.npy"]
        with pytest.raises(ValueError, match="Number of samples must match number of mask filters."):
            concat_output(base_path, samples, mask_filter=mask_filters)

    @patch(PATCH_LOAD_TRAIN_TEST_DATA)  # Still need to mock for the first call
    def test_concat_output_empty_samples_list(self, mock_load_data: MagicMock, tmp_path: Path):
        with pytest.raises(IndexError):  # Accesses samples[0] before load_train_test_data
            concat_output(tmp_path, [])

    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_concat_output_first_sample_has_nones(self, mock_load_data: MagicMock, tmp_path: Path):
        base_path = tmp_path / "train_data"
        base_path.mkdir()
        samples = ["s1", "s2"]

        mock_load_data.side_effect = [
            (["img_s1"], ["lbl_s1"], ["name_s1"], None, None, None),
            (["img_s2"], ["lbl_s2"], ["name_s2"], ["test_img_s2"], ["test_lbl_s2"], ["test_name_s2"]),
        ]
        result = concat_output(base_path, samples)

        assert result[0] == ["img_s1", "img_s2"]
        assert result[1] == ["lbl_s1", "lbl_s2"]
        assert result[2] == ["name_s1", "name_s2"]
        assert result[3] is None
        assert result[4] is None
        assert result[5] is None


class TestTrainFunction:
    @patch("fishtools.segment.train._train")  # Mock the internal heavy _train
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)  # Mock load_train_test_data used by concat_output
    def test_train_flow_updates_config_and_returns_json(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        # Setup mock for load_train_test_data (called by concat_output)
        # concat_output will call it twice due to sample_train_config.training_paths
        mock_load_cellpose_data.side_effect = [
            (["img_s1"], ["lbl_s1"], ["name_s1"], None, None, None),
            (["img_s2"], ["lbl_s2"], ["name_s2"], None, None, None),
        ]

        # Expected data after concat_output processes the mocked load_train_test_data returns
        expected_concatenated_data = (
            ["img_s1", "img_s2"],
            ["lbl_s1", "lbl_s2"],
            ["name_s1", "name_s2"],
            None,
            None,
            None,
        )

        mock_model_path_str = str(main_train_path / "models" / model_name)
        mock_returned_train_losses = [0.5, 0.4, 0.3]
        mock_test_losses = [0.6, 0.5, 0.4]  # Though not used in config update in current code
        mock_internal_heavy_train.return_value = (
            mock_model_path_str,
            mock_returned_train_losses,
            mock_test_losses,
        )

        returned_json = run_train(model_name, main_train_path, sample_train_config)

        # Assert that load_train_test_data was called correctly by concat_output
        expected_load_calls = [
            call(
                str(main_train_path / sample_train_config.training_paths[0]),
                mask_filter="_seg.npy",  # Hardcoded in train -> concat_output
                look_one_level_down=True,  # Hardcoded in train -> concat_output
            ),
            call(
                str(main_train_path / sample_train_config.training_paths[1]),
                mask_filter="_seg.npy",
                look_one_level_down=True,
            ),
        ]
        mock_load_cellpose_data.assert_has_calls(expected_load_calls)

        # Assert that _train was called with the concatenated data
        mock_internal_heavy_train.assert_called_once_with(
            expected_concatenated_data, main_train_path, train_config=sample_train_config, name=model_name
        )

        expected_config_data = sample_train_config.model_dump()
        expected_config_data["train_losses"] = mock_returned_train_losses

        assert returned_json == TrainConfig.model_validate(expected_config_data)

    @patch("fishtools.segment.train._train")
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_train_flow_applies_include_exclude_filters(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        filtered_config = sample_train_config.model_copy(
            update=dict(
                training_paths=["roi1", "roi2", "roi3"],
                include=[r".*/roi[12]/"],
                exclude=[r".*/roi2/"],
            )
        )

        mock_load_cellpose_data.side_effect = [
            (
                ["img_roi1"],
                ["lbl_roi1"],
                ["name_roi1"],
                ["test_img_roi1"],
                ["test_lbl_roi1"],
                ["test_name_roi1"],
            ),
            (
                ["img_roi2"],
                ["lbl_roi2"],
                ["name_roi2"],
                ["test_img_roi2"],
                ["test_lbl_roi2"],
                ["test_name_roi2"],
            ),
            (
                ["img_roi3"],
                ["lbl_roi3"],
                ["name_roi3"],
                ["test_img_roi3"],
                ["test_lbl_roi3"],
                ["test_name_roi3"],
            ),
        ]

        mock_internal_heavy_train.return_value = (
            str(main_train_path / "models" / model_name),
            [0.2],
            [0.3],
        )

        # Use absolute image names so regexes match full resolved paths.
        def _mk_abs(sample: str, name: str) -> str:
            return str(main_train_path / sample / name)

        # Adjust side effects to return absolute image names
        mock_load_cellpose_data.side_effect = [
            (
                ["img_roi1"],
                ["lbl_roi1"],
                [_mk_abs("roi1", "name_roi1")],
                ["test_img_roi1"],
                ["test_lbl_roi1"],
                ["test_name_roi1"],
            ),
            (
                ["img_roi2"],
                ["lbl_roi2"],
                [_mk_abs("roi2", "name_roi2")],
                ["test_img_roi2"],
                ["test_lbl_roi2"],
                ["test_name_roi2"],
            ),
            (
                ["img_roi3"],
                ["lbl_roi3"],
                [_mk_abs("roi3", "name_roi3")],
                ["test_img_roi3"],
                ["test_lbl_roi3"],
                ["test_name_roi3"],
            ),
        ]

        returned_config = run_train(model_name, main_train_path, filtered_config)

        expected_calls = [
            call(
                str(main_train_path / "roi1"),
                mask_filter="_seg.npy",
                look_one_level_down=True,
            ),
            call(
                str(main_train_path / "roi2"),
                mask_filter="_seg.npy",
                look_one_level_down=True,
            ),
            call(
                str(main_train_path / "roi3"),
                mask_filter="_seg.npy",
                look_one_level_down=True,
            ),
        ]
        assert mock_load_cellpose_data.call_args_list == expected_calls

        expected_concat_data = (
            ["img_roi1"],
            ["lbl_roi1"],
            [str(main_train_path / "roi1" / "name_roi1")],
            ["test_img_roi1", "test_img_roi2", "test_img_roi3"],
            ["test_lbl_roi1", "test_lbl_roi2", "test_lbl_roi3"],
            ["test_name_roi1", "test_name_roi2", "test_name_roi3"],
        )

        mock_internal_heavy_train.assert_called_once_with(
            expected_concat_data, main_train_path, train_config=filtered_config, name=model_name
        )

        # Config no longer mutates training_paths; only train_losses are updated
        assert returned_config.training_paths == ["roi1", "roi2", "roi3"]
        assert returned_config.include == [r".*/roi[12]/"]
        assert returned_config.exclude == [r".*/roi2/"]

    @patch("fishtools.segment.train._train")
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_train_flow_filters_windows_style_paths(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        filtered_config = sample_train_config.model_copy(
            update=dict(
                training_paths=["roi1", "roi2"],
                include=[r".*/roi1/"],
                exclude=[],
            )
        )

        def _windows_path(sample: str, name: str) -> str:
            return (main_train_path / sample / name).as_posix().replace("/", "\\")

        mock_load_cellpose_data.side_effect = [
            (
                ["img_roi1"],
                ["lbl_roi1"],
                [_windows_path("roi1", "name_roi1.tif")],
                None,
                None,
                None,
            ),
            (
                ["img_roi2"],
                ["lbl_roi2"],
                [_windows_path("roi2", "name_roi2.tif")],
                None,
                None,
                None,
            ),
        ]

        mock_internal_heavy_train.return_value = (
            str(main_train_path / "models" / model_name),
            [0.1],
            [0.2],
        )

        run_train(model_name, main_train_path, filtered_config)

        expected_filtered = (
            ["img_roi1"],
            ["lbl_roi1"],
            [_windows_path("roi1", "name_roi1.tif")],
            None,
            None,
            None,
        )

        mock_internal_heavy_train.assert_called_once_with(
            expected_filtered,
            main_train_path,
            train_config=filtered_config,
            name=model_name,
        )

    @patch("fishtools.segment.train._train")
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    @pytest.mark.parametrize(
        ("include_patterns", "exclude_patterns", "expected_samples"),
        [
            pytest.param([], [], ["roi1", "roi2", "roi3"], id="no_filters"),
            pytest.param([r".*/roi1/"], [], ["roi1"], id="include_only"),
            pytest.param([], [r".*/roi2/"], ["roi1", "roi3"], id="exclude_only"),
            pytest.param([r".*/roi[13]/"], [r".*/roi3/"], ["roi1"], id="include_then_exclude"),
            pytest.param([r".*name_roi[12]\.tif$"], [r".*/roi1/"], ["roi2"], id="filename_regex"),
        ],
    )
    def test_train_flow_parametrized_filters(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
        include_patterns: list[str],
        exclude_patterns: list[str],
        expected_samples: list[str],
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        parametrized_config = sample_train_config.model_copy(
            update=dict(
                training_paths=["roi1", "roi2", "roi3"],
                include=include_patterns,
                exclude=exclude_patterns,
            )
        )

        def _original_path(sample: str) -> str:
            raw_path = str(main_train_path / sample / f"name_{sample}.tif")
            if sample == "roi2":
                return raw_path.replace("/", "\\")  # simulate Windows-style entry
            return raw_path

        records: list[dict[str, str]] = []
        side_effects: list[tuple[list[str], ...]] = []

        for sample in ["roi1", "roi2", "roi3"]:
            original_path = _original_path(sample)
            normalized_path = original_path.replace("\\", "/")
            record = {
                "sample": sample,
                "image": f"img_{sample}",
                "label": f"lbl_{sample}",
                "original": original_path,
                "normalized": normalized_path,
            }
            records.append(record)
            side_effects.append(
                (
                    [record["image"]],
                    [record["label"]],
                    [record["original"]],
                    [f"test_img_{sample}"],
                    [f"test_lbl_{sample}"],
                    [f"test_name_{sample}"],
                )
            )

        mock_load_cellpose_data.side_effect = side_effects
        mock_internal_heavy_train.return_value = (
            str(main_train_path / "models" / model_name),
            [0.123],
            [0.456],
        )

        run_train(model_name, main_train_path, parametrized_config)

        mock_internal_heavy_train.assert_called_once()
        filtered_payload = mock_internal_heavy_train.call_args.args[0]
        filtered_images, filtered_labels, filtered_names = filtered_payload[:3]

        assert filtered_names is not None

        expected_records = [record for record in records if record["sample"] in expected_samples]

        assert filtered_images == [record["image"] for record in expected_records]
        assert filtered_labels == [record["label"] for record in expected_records]

        normalized_filtered_names = [
            (name.as_posix() if isinstance(name, Path) else str(name)).replace("\\", "/")
            for name in filtered_names
        ]
        assert normalized_filtered_names == [record["normalized"] for record in expected_records]

    @patch("fishtools.segment.train._train")
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_train_flow_invalid_regex_raises_error(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        invalid_config = sample_train_config.model_copy(
            update=dict(
                training_paths=["roi1"],
                include=["["],
                exclude=[],
            )
        )

        mock_load_cellpose_data.side_effect = [
            (
                ["img_roi1"],
                ["lbl_roi1"],
                [str(main_train_path / "roi1" / "name_roi1.tif")],
                None,
                None,
                None,
            ),
        ]

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            run_train(model_name, main_train_path, invalid_config)

        mock_internal_heavy_train.assert_not_called()

    @patch("fishtools.segment.train._train")
    @patch(PATCH_LOAD_TRAIN_TEST_DATA)
    def test_train_flow_filters_remove_all_paths_error(
        self,
        mock_load_cellpose_data: MagicMock,
        mock_internal_heavy_train: MagicMock,
        tmp_path: Path,
        sample_train_config: TrainConfig,
    ):
        model_name = sample_train_config.name
        main_train_path = tmp_path / "training_root"
        main_train_path.mkdir()

        dropping_config = sample_train_config.model_copy(
            update=dict(
                training_paths=["roi1"],
                include=[r".*/nomatch/"],
                exclude=[],
            )
        )

        mock_load_cellpose_data.side_effect = [
            (
                ["img_roi1"],
                ["lbl_roi1"],
                [str(main_train_path / "roi1" / "name_roi1.tif")],
                None,
                None,
                None,
            ),
        ]

        with pytest.raises(ValueError, match="No training images remain after applying include/exclude filters"):
            run_train(model_name, main_train_path, dropping_config)

        mock_internal_heavy_train.assert_not_called()


class TestMainCommand:
    runner = CliRunner()

    @patch("fishtools.segment.run_train")
    def test_main_success(
        self, mock_train_orchestrator_func: MagicMock, tmp_path: Path, sample_train_config_data: dict
    ):
        train_dir = tmp_path / "my_train_project"
        train_dir.mkdir()
        models_dir = train_dir / "models"
        models_dir.mkdir()

        model_name = sample_train_config_data["name"]
        config_path = models_dir / f"{model_name}.json"
        with open(config_path, "w") as f:
            json.dump(sample_train_config_data, f)

        mock_run_train_return_object = MagicMock()
        mock_run_train_return_object.model_dump_json.return_value = "{}"
        mock_train_orchestrator_func.return_value = mock_run_train_return_object

        result = self.runner.invoke(app, ["train", str(train_dir), model_name])

        if result.exception:
            raise result.exception

        assert result.exit_code == 0

        assert mock_train_orchestrator_func.call_count == 1
        args, kwargs = mock_train_orchestrator_func.call_args

        assert args[0] == model_name  # name
        assert args[1] == train_dir.resolve()  # path
        loaded_config_arg = args[2]  # train_config
        assert isinstance(loaded_config_arg, TrainConfig)
        assert loaded_config_arg.name == model_name
        assert loaded_config_arg.n_epochs == sample_train_config_data["n_epochs"]

    def test_main_models_path_not_exist(self, tmp_path: Path):
        train_dir = tmp_path / "my_train_project"
        train_dir.mkdir()
        # models_dir is NOT created

        model_name = "test_model"
        result = self.runner.invoke(app, ["train", str(train_dir), model_name])

        assert result.exit_code != 0
        assert "Models path" in str(result.exception)

    def test_main_config_file_not_exist(self, tmp_path: Path):
        train_dir = tmp_path / "my_train_project"
        train_dir.mkdir()
        models_dir = train_dir / "models"
        models_dir.mkdir()
        # Config file is NOT created

        model_name = "non_existent_model"
        result = self.runner.invoke(app, ["train", str(train_dir), model_name])

        assert result.exit_code != 0
        assert f"Config file {model_name}.json not found" in str(result.exception)
