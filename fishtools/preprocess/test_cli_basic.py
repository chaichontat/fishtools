from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture
from tifffile import imwrite

from fishtools.preprocess.cli_basic import (
    extract_data_from_registered,
    extract_data_from_tiff,
    run_with_extractor,
)

IMG_HEIGHT = 2048
IMG_WIDTH = 2048


@pytest.fixture
def mock_basic_model(mocker: MockerFixture) -> MagicMock:
    mock_model = mocker.MagicMock(name="BaSiCModelInstance")
    mock_model.fit.return_value = None
    mock_model.transform.return_value = np.zeros((1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    mock_model.flatfield = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    mock_model.darkfield = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    return mock_model


@pytest.fixture
def mock_basic_module_components(mocker: MockerFixture, mock_basic_model: MagicMock) -> MagicMock:
    mocker.patch("fishtools.preprocess.cli_basic.fit_basic", return_value=mock_basic_model)
    mocker.patch("fishtools.preprocess.cli_basic.plot_basic")
    return mock_basic_model


@pytest.fixture
def mock_get_channels(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("fishtools.preprocess.cli_basic.get_channels", return_value=["C1", "C2"])


@pytest.fixture
def temp_tiff_files(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    files_info: list[dict[str, Any]] = []
    # Structure: tmp_path / round--pos / file.tif
    # For extract_data_from_tiff
    round_dir1 = tmp_path / "R1--P1"
    round_dir1.mkdir()
    file1_path = round_dir1 / "img1.tif"

    data_file1 = []
    for z in range(5):
        for c in range(2):
            img_data = np.full((IMG_HEIGHT, IMG_WIDTH), (z + 1) * 10 + c, dtype=np.uint16)
            data_file1.append(img_data)
    imwrite(file1_path, np.stack(data_file1))
    files_info.append({"path": file1_path, "nc": 2, "nz_total_pages": 10, "nz_per_channel": 5})

    round_dir2 = tmp_path / "R1--P2"
    round_dir2.mkdir()
    file2_path = round_dir2 / "img2.tif"
    # 2 channels, 5 z-slices per channel
    data_file2 = []
    for z in range(5):
        for c in range(2):
            img_data = np.full((IMG_HEIGHT, IMG_WIDTH), (z + 1) * 100 + c, dtype=np.uint16)
            data_file2.append(img_data)
    imwrite(file2_path, np.stack(data_file2))
    files_info.append({"path": file2_path, "nc": 2, "nz_total_pages": 10, "nz_per_channel": 5})

    reg_dir = tmp_path / "registered--P1"
    reg_dir.mkdir()
    file_reg_path = reg_dir / "reg_img1.tif"

    data_reg_for_max_proj = np.arange(5 * 2 * IMG_HEIGHT * IMG_WIDTH, dtype=np.uint16).reshape((
        5,
        2,
        IMG_HEIGHT,
        IMG_WIDTH,
    ))
    # Make values distinct for max projection
    for z_val in range(5):
        data_reg_for_max_proj[z_val, :, :, :] += z_val * 1000
    imwrite(file_reg_path, data_reg_for_max_proj, imagej=True)  # ZCTYX order
    files_info.append({"path": file_reg_path, "type": "registered", "shape": data_reg_for_max_proj.shape})

    return tmp_path, files_info


class TestExtractDataFromTiff:
    def test_extract_basic(self, temp_tiff_files: tuple[Path, list[dict[str, Any]]]) -> None:
        tmp_p, files_info = temp_tiff_files
        tiff_files = [fi["path"] for fi in files_info if fi.get("type") != "registered"]

        nc = 2
        zs = [0.5]  # Middle z-slice (index 2 for 0-4 range)

        result = extract_data_from_tiff(tiff_files, zs=zs, nc=nc, max_files=2)

        # Expected shape: (n_files * len(zs), nc, height, width)
        assert result.shape == (2, nc, IMG_HEIGHT, IMG_WIDTH)

        # Check values for file1, z=0.5 (actual index 2), channel 0
        # Original data: (z_idx+1)*10 + c. For z_idx=2, c=0 -> (2+1)*10 + 0 = 30
        assert np.all(result[0, 0] == 30)
        # Check values for file1, z=0.5 (actual index 2), channel 1
        # Original data: (z_idx+1)*10 + c. For z_idx=2, c=1 -> (2+1)*10 + 1 = 31
        assert np.all(result[0, 1] == 31)

        # Check values for file2, z=0.5 (actual index 2), channel 0
        # Original data: (z_idx+1)*100 + c. For z_idx=2, c=0 -> (2+1)*100 + 0 = 300
        assert np.all(result[1, 0] == 300)

    def test_extract_multiple_zs(self, temp_tiff_files: tuple[Path, list[dict[str, Any]]]) -> None:
        tmp_p, files_info = temp_tiff_files
        tiff_files = [fi["path"] for fi in files_info if fi.get("type") != "registered"][:1]  # Use one file

        nc = 2
        zs = [0.0, 0.99]  # z-slice 0 and z-slice 4 (for 5 slices 0..4)

        result = extract_data_from_tiff(tiff_files, zs=zs, nc=nc, max_files=1)

        assert result.shape == (1 * len(zs), nc, IMG_HEIGHT, IMG_WIDTH)
        # File1, z=0.0 (index 0), channel 0: (0+1)*10 + 0 = 10
        assert np.all(result[0, 0] == 10)
        # File1, z=0.99 (index 4), channel 1: (4+1)*10 + 1 = 51
        assert np.all(result[1, 1] == 51)

    def test_nc_must_be_provided(self, temp_tiff_files: tuple[Path, list[dict[str, Any]]]) -> None:
        tmp_p, files_info = temp_tiff_files
        tiff_files = [fi["path"] for fi in files_info if fi.get("type") != "registered"]
        with pytest.raises(ValueError, match="nc must be provided"):
            extract_data_from_tiff(tiff_files, zs=[0.5], nc=None)

    def test_deconv_meta_usage(
        self, temp_tiff_files: tuple[Path, list[dict[str, Any]]], mocker: MockerFixture
    ) -> None:
        tmp_p, files_info = temp_tiff_files
        tiff_files = [fi["path"] for fi in files_info if fi.get("type") != "registered"][:1]

        mock_scale_deconv = mocker.patch(
            "fishtools.preprocess.cli_basic.scale_deconv", side_effect=lambda img, *args, **kwargs: img * 2
        )

        deconv_meta_arr = np.array([1.0, 1.0])  # Dummy
        result = extract_data_from_tiff(tiff_files, zs=[0.5], nc=2, deconv_meta=deconv_meta_arr)

        assert mock_scale_deconv.call_count == 2  # 1 file * 1 z-slice * 2 channels
        # Value for file1, z=0.5 (idx 2), channel 0 was 30. Scaled: 30*2 = 60
        assert np.all(result[0, 0] == 60)


class TestExtractDataFromRegistered:
    def test_extract_basic_registered(
        self, temp_tiff_files: tuple[Path, list[dict[str, Any]]], mocker: MockerFixture
    ) -> None:
        tmp_p, files_info = temp_tiff_files
        reg_file_info = next(fi for fi in files_info if fi.get("type") == "registered")
        reg_files = [reg_file_info["path"]]

        # Mock imread to return the specific data we saved for this test
        # The data was (5, 2, H, W) with z_val * 1000 added
        # Max projection over axis 0
        original_data = np.arange(5 * 2 * IMG_HEIGHT * IMG_WIDTH, dtype=np.uint16).reshape((
            5,
            2,
            IMG_HEIGHT,
            IMG_WIDTH,
        ))
        for z_val in range(5):
            original_data[z_val, :, :, :] += z_val * 1000

        mocker.patch("fishtools.preprocess.cli_basic.imread", return_value=original_data)

        result = extract_data_from_registered(reg_files, zs=[0.5], nc=None, max_files=1)

        expected_nc = original_data.shape[1]  # Should be 2
        assert result.shape == (1, expected_nc, IMG_HEIGHT, IMG_WIDTH)

        # Max over Z (axis 0)
        expected_max_proj_data = np.max(original_data, axis=0)
        assert np.array_equal(result[0], expected_max_proj_data)


class TestRunWithExtractor:
    @pytest.fixture
    def mock_extractor(self, mocker: MockerFixture) -> MagicMock:
        # Returns (samples, nc, H, W)
        return mocker.patch(
            "fishtools.preprocess.cli_basic.DataExtractor.__call__",  # Mocking via protocol might be tricky
            return_value=np.zeros((5, 2, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32),
        )

    def test_run_basic_flow(
        self,
        tmp_path: Path,
        mock_get_channels: MagicMock,
        mock_basic_module_components: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        round_name = "R1Test"
        for i in range(100):
            p = tmp_path / f"{round_name}--pos{i}"
            p.mkdir()
            (p / "img.tif").touch()

        # Mock the extractor function that will be passed
        mock_actual_extractor_func = mocker.MagicMock(
            return_value=np.zeros((10, 2, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  # 10 files, 2 channels
        )

        mock_fit_save = mocker.patch("fishtools.preprocess.cli_basic.fit_and_save_basic")
        mocker.patch("numpy.loadtxt", return_value=None)  # No deconv_meta

        # To avoid "Not enough files" error, we need to ensure enough files are "found"
        # The globbing is `path.glob(f"{round_}--*/*.tif")`
        # We need to make sure `get_channels` is happy too.
        mock_get_channels.return_value = ["chA", "chB"]

        # Patch random.sample
        mocker.patch("random.sample", side_effect=lambda x, k: list(x)[:k])

        run_with_extractor(
            tmp_path, round_=round_name, extractor_func=mock_actual_extractor_func, plot=False, zs=(0.5,)
        )

        mock_actual_extractor_func.assert_called_once()
        args, kwargs = mock_actual_extractor_func.call_args
        assert len(args[0]) >= 100  # files list, updated from previous fix
        assert args[1] == (0.5,)  # zs
        assert kwargs["nc"] == 2  # from get_channels

        mock_fit_save.assert_called_once()
        fit_args, fit_kwargs = mock_fit_save.call_args
        # data (pos 0), output_dir (pos 1), round_ (pos 2), channels (pos 3)
        assert fit_args[2] == round_name
        assert fit_args[3] == ["chA", "chB"]

    def test_run_not_enough_files_error(
        self, tmp_path: Path, mock_get_channels: MagicMock, mocker: MockerFixture
    ) -> None:
        mock_actual_extractor_func = mocker.MagicMock()
        mocker.patch("fishtools.preprocess.cli_basic.fit_and_save_basic")

        # Create one file so that files[0] is valid, but len(files) < 100
        round_name = "R1Small"
        p_dir = tmp_path / f"{round_name}--pos0"
        p_dir.mkdir(exist_ok=True, parents=True)
        (p_dir / "img.tif").touch()

        # Ensure get_channels returns a valid list even if called with a dummy file
        mock_get_channels.return_value = ["chA", "chB"]

        with pytest.raises(ValueError, match="Not enough files"):
            run_with_extractor(
                tmp_path, round_=round_name, extractor_func=mock_actual_extractor_func, plot=False, zs=(0.5,)
            )
