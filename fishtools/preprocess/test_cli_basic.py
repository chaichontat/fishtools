import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture
from tifffile import TiffFile, imwrite

from fishtools.preprocess.cli_basic import (
    basic,
    extract_data_from_registered,
    extract_data_from_tiff,
    run_with_extractor,
    sample_canonical_unique_tiles,
)
from fishtools.preprocess.tileconfig import tiles_at_least_n_steps_from_edges


class StubBasicModel:
    def __init__(self, darkfield: np.ndarray, flatfield: np.ndarray) -> None:
        self.darkfield = darkfield
        self.flatfield = flatfield


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

    def test_sampling_filters_to_interior_tiles_when_csv_present(
        self,
        tmp_path: Path,
        mock_get_channels: MagicMock,
        mock_basic_module_components: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # Create 3 ROIs, each a 12x10 grid (120 tiles). Interior tiles with n=2 → (12-4)*(10-4)=8*6=48.
        round_name = "R2"
        rois = ["roiA", "roiB", "roiC"]

        # Prepare CSVs with consistent grid coordinates and files with numeric indices
        for roi in rois:
            coords = []
            for y in range(10):
                for x in range(12):
                    coords.append((float(x), float(y)))
            # Write CSV as x,y pairs; order maps to index i = y*12 + x
            (tmp_path / f"{roi}.csv").write_text("\n".join(f"{y},{x}" for x, y in coords))

            d = tmp_path / f"{round_name}--{roi}"
            d.mkdir(parents=True, exist_ok=True)

            # Set file sizes: edges small (0B), interior large (~1KB) so that
            # size >= 60th percentile selects interiors; OR keeps interiors anyway.
            # 12x10 => interior count (n=2) = 48, edges = 72 per ROI
            def is_interior(i: int) -> bool:
                y, x = divmod(i, 12)
                return (x >= 2 and x <= 9) and (y >= 2 and y <= 7)

            for idx in range(12 * 10):
                p = d / f"{round_name}-{idx:04d}.tif"
                if is_interior(idx):
                    p.write_bytes(b"1" * 1024)  # large file
                else:
                    p.touch()  # 0B file

        mock_get_channels.return_value = ["chA", "chB"]

        # Collector to introspect which files were passed into the extractor
        captured_files = {}

        def fake_extractor(files, zs, deconv_meta=None, max_files=800, nc=None):
            captured_files["files"] = list(files)
            # Return shape (n_samples, nc, H, W)
            return np.zeros((len(files), 2, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        # Make sampling deterministic: identity sample
        mocker.patch("random.sample", side_effect=lambda x, k: list(x)[:k])
        mocker.patch("numpy.loadtxt", return_value=None)

        # Avoid pickling real/basic mocks by bypassing fit_and_save_basic
        mocker.patch("fishtools.preprocess.cli_basic.fit_and_save_basic", return_value=[])
        run_with_extractor(tmp_path, round_=round_name, extractor_func=fake_extractor, plot=False, zs=(0.5,))

        assert "files" in captured_files
        sampled = captured_files["files"]
        # Expect exactly 3 * 48 = 144 interior tiles kept by OR logic and 60th percentile sizing
        assert len(sampled) == 3 * 48

        # Verify no edge indices slipped through for a representative ROI
        # Recompute allowed for roiA
        import pandas as pd

        df = pd.read_csv(tmp_path / "roiA.csv", header=None)
        allowed = set(tiles_at_least_n_steps_from_edges(df.iloc[:, :2].to_numpy(), n=2).tolist())

        roiA_files = [p for p in sampled if p.parent.name == f"{round_name}--roiA"]
        roiA_indices = {int(p.stem.split("-")[-1]) for p in roiA_files}
        assert roiA_indices.issubset(allowed)


def test_sample_canonical_unique_tiles_deduplicates_and_filters(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Build one ROI grid 12x10 → interior (n=2) = 48 indices
    roi = "roiA"
    coords = []
    for y in range(10):
        for x in range(12):
            coords.append((float(x), float(y)))
    (tmp_path / f"{roi}.csv").write_text("\n".join(f"{y},{x}" for x, y in coords))

    rounds = ["1_2_3", "4_5_6"]  # canonical numeric rounds
    weird = "wga_brdu_dapi"  # non-canonical

    def is_interior(i: int) -> bool:
        y, x = divmod(i, 12)
        return (x >= 2 and x <= 9) and (y >= 2 and y <= 7)

    for r in rounds:
        d = tmp_path / f"{r}--{roi}"
        d.mkdir(parents=True, exist_ok=True)
        for idx in range(12 * 10):
            p = d / f"{r}-{idx:04d}.tif"
            if is_interior(idx):
                p.write_bytes(b"1")
            else:
                p.touch()

    # Add a non-canonical round with a few tiles
    wd = tmp_path / f"{weird}--{roi}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / f"{weird}-0000.tif").write_bytes(b"1")

    # Mock get_channels depending on round token derived from parent dir
    def fake_get_channels(sample_file: Path) -> list[str]:
        parent = sample_file.parent.name
        round_token = parent.split("--", 1)[0]
        if round_token in rounds:
            return ["560", "650", "750"]
        return ["wga", "brdu", "dapi"]

    mocker.patch("fishtools.preprocess.cli_basic.get_channels", side_effect=fake_get_channels)

    files, extra = sample_canonical_unique_tiles(tmp_path)

    # Non-canonical should be reported
    assert weird in extra

    # Dedup: only 48 unique interior indices across 2 canonical rounds
    assert len(files) == 48

    # All returned files belong to canonical rounds and are interior indices
    df = pd.read_csv(tmp_path / f"{roi}.csv", header=None)
    allowed = set(tiles_at_least_n_steps_from_edges(df.iloc[:, :2].to_numpy(), n=2).tolist())
    seen = set()
    for p in files:
        assert any(p.as_posix().startswith((tmp_path / f"{r}--{roi}").as_posix()) for r in rounds)
        idx = int(p.stem.split("-")[-1])
        assert idx in allowed
        key = (roi, idx)
        assert key not in seen
        seen.add(key)


def test_run_writes_sampling_json(tmp_path: Path, mocker: MockerFixture) -> None:
    # One ROI grid 12x10 → 48 interior indices
    roi = "roiA"
    round_name = "R3"
    coords = []
    for y in range(10):
        for x in range(12):
            coords.append((float(x), float(y)))
    (tmp_path / f"{roi}.csv").write_text("\n".join(f"{y},{x}" for x, y in coords))

    d = tmp_path / f"{round_name}--{roi}"
    d.mkdir(parents=True, exist_ok=True)

    def is_interior(i: int) -> bool:
        y, x = divmod(i, 12)
        return (x >= 2 and x <= 9) and (y >= 2 and y <= 7)

    for idx in range(12 * 10):
        p = d / f"{round_name}-{idx:04d}.tif"
        if is_interior(idx):
            p.write_bytes(b"1")
        else:
            p.touch()

    # Channels for this round (3-channel canonical)
    mocker.patch("fishtools.preprocess.cli_basic.get_channels", return_value=["560", "650", "750"])

    # Deterministic sampling
    seed = 7

    # Mock extractor and fit to avoid heavy work
    mocker.patch(
        "fishtools.preprocess.cli_basic.fit_and_save_basic",
        return_value=[],
    )

    def fake_extractor(files, zs, deconv_meta=None, max_files=800, nc=None):
        # Return array with correct nc
        n = len(files)
        c = nc or 3
        return np.zeros((n, c, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    mocker.patch("numpy.loadtxt", return_value=None)

    run_with_extractor(tmp_path, round_name, fake_extractor, plot=False, zs=(0.5,), seed=seed)

    manifest = tmp_path / "basic" / f"{round_name}-sampling.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["round"] == round_name
    assert payload["seed"] == seed
    assert payload["channels"] == ["560", "650", "750"]
    assert isinstance(payload["sampled_tiles"], list)
    # Exactly the interior count
    assert len(payload["sampled_tiles"]) == 48


def test_run_all_canonical_deduplicates_indices(tmp_path: Path, mocker: MockerFixture) -> None:
    # Workspace with one ROI and two canonical rounds; ensure 'all' sampling dedups indices across rounds
    roi = "roiA"
    rounds = ["1_2_3", "4_5_6"]

    # ROI grid 16x14 → interior(n=2) = (16-4)*(14-4)=12*10=120 indices (meets >=100 threshold)
    coords = []
    for y in range(14):
        for x in range(16):
            coords.append((float(x), float(y)))
    (tmp_path / f"{roi}.csv").write_text("\n".join(f"{y},{x}" for x, y in coords))

    def is_interior(i: int) -> bool:
        y, x = divmod(i, 16)
        return (x >= 2 and x <= 13) and (y >= 2 and y <= 11)

    for r in rounds:
        d = tmp_path / f"{r}--{roi}"
        d.mkdir(parents=True, exist_ok=True)
        for idx in range(16 * 14):
            p = d / f"{r}-{idx:04d}.tif"
            if is_interior(idx):
                p.write_bytes(b"1")
            else:
                p.touch()

    # Channels for canonical rounds
    mocker.patch("fishtools.preprocess.cli_basic.get_channels", return_value=["560", "650", "750"])

    captured = {}

    def fake_extractor(files, zs, deconv_meta=None, max_files=800, nc=None):
        captured["files"] = list(files)
        n = len(files)
        c = nc or 3
        return np.zeros((n, c, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    # Make sampling deterministic and avoid heavy work
    mocker.patch("random.sample", side_effect=lambda x, k: list(x)[:k])
    mocker.patch("numpy.loadtxt", return_value=None)
    mocker.patch("fishtools.preprocess.cli_basic.fit_and_save_basic", return_value=[])

    # Run in 'all' mode → round_=None
    run_with_extractor(tmp_path, round_=None, extractor_func=fake_extractor, plot=False, zs=(0.5,))

    assert "files" in captured
    sampled = captured["files"]
    # Expect exactly 120 unique interior indices overall (dedup across rounds)
    assert len(sampled) == 120
    # Ensure we sampled from both canonical rounds
    sampled_rounds = {p.parent.name.split("--", 1)[0] for p in sampled}
    assert sampled_rounds.issuperset(set(rounds))
    # And ensure no two paths share the same (roi, index)
    seen: set[tuple[str, int]] = set()
    for p in sampled:
        roi_dir = p.parent.name
        roi_base = roi_dir.split("--", 1)[-1]
        idx = int(p.stem.split("-")[-1])
        key = (roi_base, idx)
        assert key not in seen
        seen.add(key)


class TestTransformCommand:
    def _write_basic_profile(self, target: Path, dark: float, flat: float, *, size: int = 4) -> None:
        darkfield = np.full((size, size), dark, dtype=np.float32)
        flatfield = np.full((size, size), flat, dtype=np.float32)
        payload = {"basic": StubBasicModel(darkfield, flatfield)}
        target.write_bytes(pickle.dumps(payload))

    def test_transform_corrects_tile_and_writes_to_default_location(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "basic").mkdir()
        (workspace / "analysis").mkdir()
        (workspace / "workspace.DONE").touch()

        mocker.patch("fishtools.preprocess.cli_basic.IMWRITE_KWARGS", {"compression": None})

        channels = ["560", "650"]
        for idx, channel in enumerate(channels):
            self._write_basic_profile(workspace / "basic" / f"roundA-{channel}.pkl", dark=5.0 * idx, flat=2.0)

        roi_dir = workspace / "roundA--roi1"
        roi_dir.mkdir()
        tile = np.stack(
            [
                np.full((4, 4), 20, dtype=np.uint16),  # channel 560
                np.full((4, 4), 200, dtype=np.uint16),  # channel 650
            ],
            axis=0,
        )
        input_tiff = roi_dir / "roundA-0001.tif"
        imwrite(input_tiff, tile)

        runner = CliRunner()
        result = runner.invoke(
            basic,
            [
                "transform",
                str(workspace),
                "roi1",
                "1",
                "--round",
                "roundA",
                "--n-fids",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        output_path = workspace / "analysis" / "basic_transform" / "roundA--roi1" / input_tiff.name
        assert output_path.exists()

        with TiffFile(output_path) as tif:
            corrected = tif.asarray()
            metadata = tif.shaped_metadata[0]

        expected_ch0 = np.full((4, 4), 10, dtype=np.uint16)  # (20 - 0) / 2
        expected_ch1 = np.full((4, 4), 98, dtype=np.uint16)  # (200 - 5*1) / 2 → 97.5 -> 98 after rounding

        assert corrected.dtype == np.uint16
        assert corrected.shape == tile.shape
        np.testing.assert_array_equal(corrected[0], expected_ch0)
        np.testing.assert_array_equal(corrected[1], expected_ch1)
        assert metadata["basic_corrected"] is True
        assert metadata["basic_channels"] == channels

    def test_transform_respects_overwrite_flag(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "basic").mkdir()
        (workspace / "analysis").mkdir()
        (workspace / "workspace.DONE").touch()

        mocker.patch("fishtools.preprocess.cli_basic.IMWRITE_KWARGS", {"compression": None})

        self._write_basic_profile(workspace / "basic" / "roundA-560.pkl", dark=0.0, flat=1.0)

        roi_dir = workspace / "roundA--roi1"
        roi_dir.mkdir()
        tile1 = np.full((1, 4, 4), 50, dtype=np.uint16)
        tile2 = np.full((1, 4, 4), 80, dtype=np.uint16)
        input_tiff1 = roi_dir / "roundA-0001.tif"
        input_tiff2 = roi_dir / "roundA-0002.tif"
        imwrite(input_tiff1, tile1)
        imwrite(input_tiff2, tile2)

        runner = CliRunner()
        first = runner.invoke(
            basic,
            [
                "transform",
                str(workspace),
                "roi1",
                "--round",
                "roundA",
                "--n-fids",
                "0",
            ],
        )
        assert first.exit_code == 0, first.output

        output_dir = workspace / "analysis" / "basic_transform" / "roundA--roi1"
        output_path1 = output_dir / input_tiff1.name
        output_path2 = output_dir / input_tiff2.name
        assert output_path1.exists()
        assert output_path2.exists()

        with TiffFile(output_path1) as tif:
            initial = tif.asarray().copy()

        # Modify the first output file manually to detect changes
        imwrite(output_path1, np.full_like(tile1, 7, dtype=np.uint16))

        second = runner.invoke(
            basic,
            [
                "transform",
                str(workspace),
                "roi1",
                "--round",
                "roundA",
                "--n-fids",
                "0",
            ],
        )
        assert second.exit_code == 0, second.output

        with TiffFile(output_path1) as tif:
            skipped = tif.asarray()
        # Should remain the manual value because overwrite was not requested
        np.testing.assert_array_equal(skipped, np.full_like(tile1, 7, dtype=np.uint16))

        third = runner.invoke(
            basic,
            [
                "transform",
                str(workspace),
                "roi1",
                "--round",
                "roundA",
                "--n-fids",
                "0",
                "--overwrite",
            ],
        )
        assert third.exit_code == 0, third.output

        with TiffFile(output_path1) as tif:
            final = tif.asarray()
        np.testing.assert_array_equal(final, initial)
