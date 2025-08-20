import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import BaseModel
from tifffile import imwrite

# Import the module to test
# Note: Adjust these imports to match your actual module structure
from fishtools.preprocess.cli_basic import (
    basic,
    extract_data_from_registered,
    extract_data_from_tiff,
    fit_and_save_basic,
    run,
    run_with_extractor,
)


# Define a Pydantic model for the test data directory
class DataDir(BaseModel):
    """Pydantic model for test data directory structure and metadata"""

    dir: Path
    round_name: str
    n: int
    nc: int  # number of channels
    nz: int  # number of z-slices
    img_size: int
    roi_name: str

    class Config:
        arbitrary_types_allowed = True  # Allow Path type


# Create test directory structure and TIFF files
@pytest.fixture
def test_data_dir() -> Generator[DataDir, None, None]:
    """Create a test directory with TIFF files for testing."""
    # Define constants

    ROUND_NAME = "round1_round2_round3"
    ROI = "brain1"
    N = 200
    NC = 3  # number of channels
    NZ = 4  # number of z-slices
    IMG_SIZE = 32  # small image size for faster tests

    with TemporaryDirectory() as tmp_path:
        # Create main directories
        data_dir = Path(tmp_path) / "data"
        data_dir.mkdir()
        basic_dir = data_dir / "basic"
        basic_dir.mkdir()
        deconv_scaling_dir = data_dir / "deconv_scaling"
        deconv_scaling_dir.mkdir()

        # Create deconv scaling file
        np.savetxt(deconv_scaling_dir / f"{ROUND_NAME}.txt", np.ones((NC, 2)))

        site_dir = data_dir / f"{ROUND_NAME}--{ROI}"
        site_dir.mkdir()

        # Create site directories and TIFF files
        for tile in range(N):
            # Create multi-channel TIFF file
            tiff_path = site_dir / f"{ROUND_NAME}-{tile:04d}.tif"

            rand = np.random.default_rng(tile)
            img_data = np.zeros((NZ, NC, IMG_SIZE, IMG_SIZE), dtype=np.uint16)
            assert NC < 64
            for c in range(NC):
                img_data[:, c] = rand.integers(
                    1000 * c, 1000 * (c + 1), (NZ, IMG_SIZE, IMG_SIZE), dtype=np.uint16
                )

            img_data = np.concatenate(
                [
                    img_data.reshape(-1, IMG_SIZE, IMG_SIZE),
                    rand.integers(60000, 65535, (2, IMG_SIZE, IMG_SIZE), dtype=np.uint16),
                ],
                axis=0,
            )

            # Write TIFF file
            imwrite(tiff_path, img_data, compression=22610)

        yield DataDir(
            dir=data_dir,
            round_name=ROUND_NAME,
            n=N,
            nc=NC,
            nz=NZ,
            img_size=IMG_SIZE,
            roi_name=ROI,
        )


def test_fixture(test_data_dir: DataDir) -> None:
    """Test the test_data_dir fixture."""
    path = test_data_dir.dir
    assert (round_path := (path / f"{test_data_dir.round_name}--{test_data_dir.roi_name}")).exists()
    assert len(list(round_path.glob("*.tif"))) == test_data_dir.n

    assert len(list(round_path.glob("*.tif"))) == test_data_dir.n

    regex = re.compile(rf"{test_data_dir.round_name}-(\d{4}).tif")
    for f in round_path.glob("*.tif"):
        assert regex.match(f.name)


# # Test extractors with actual TIFF files
# def test_extract_data_from_tiff(test_data_dir: TestDataDir) -> None:
#     """Test the extract_data_from_tiff function with real TIFF files."""
#     # Get all TIFF files for the round
#     files = list(test_data_dir.dir.glob(f"{test_data_dir.round_name}--*/*.tif"))

#     # Test extraction with different z values
#     zs = [0.25, 0.75]

#     # Extract data
#     data = extract_data_from_tiff(files, zs, None, max_files=10, nc=test_data_dir.nc)

#     # Verify shape and content
#     expected_shape = (len(files) * len(zs), test_data_dir.nc, test_data_dir.img_size, test_data_dir.img_size)
#     assert data.shape == expected_shape
#     assert np.all(data > 0)  # Make sure we have non-zero data


# def test_extract_data_from_registered(test_data_dir: TestDataDir) -> None:
#     """Test the extract_data_from_registered function with real TIFF files."""
#     # Get registered TIFF files
#     files: list[Path] = list(test_data_dir.dir.glob("registered--*/*.tif"))

#     # Test extraction
#     zs = [0.5]  # Doesn't matter for registered files

#     # Extract data
#     data = extract_data_from_registered(files, zs, None, max_files=10)

#     # Verify shape and content
#     expected_shape = (len(files), test_data_dir.nc, test_data_dir.img_size, test_data_dir.img_size)
#     assert data.shape == expected_shape
#     assert np.all(data > 0)  # Make sure we have non-zero data


# # Test the fit_and_save_basic function with real data
# def test_fit_and_save_basic(test_data_dir: TestDataDir, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Test the fit_and_save_basic function with synthetic data."""
#     # Create a simple dataset for testing
#     # We'll use random data since we're not testing the BaSiC algorithm itself
#     test_data = np.random.rand(10, test_data_dir.nc, test_data_dir.img_size, test_data_dir.img_size).astype(
#         np.float32
#     )

#     # Define a mock BaSiC class and fit_basic function for faster testing
#     class MockBaSiC:
#         def __init__(self) -> None:
#             self.flatfield = np.ones((test_data_dir.img_size, test_data_dir.img_size))
#             self.darkfield = np.zeros((test_data_dir.img_size, test_data_dir.img_size))

#     def mock_fit_basic(data: np.ndarray, channel: int) -> MockBaSiC:
#         return MockBaSiC()

#     # Patch the fit_basic function to use our mock
#     monkeypatch.setattr("fishtools.preprocess.cli_basic.fit_basic", mock_fit_basic)

#     # Need to patch plot_basic too
#     def mock_plot_basic(basic: MockBaSiC) -> None:
#         pass

#     monkeypatch.setattr("fishtools.preprocess.cli_basic.plot_basic", mock_plot_basic)

#     # Run fit_and_save_basic
#     output_dir = test_data_dir.dir / "basic"
#     results = fit_and_save_basic(test_data, output_dir, test_data_dir.round_name, plot=False)

#     # Check results
#     assert len(results) == test_data_dir.nc

#     # Verify files were created
#     for c in range(test_data_dir.nc):
#         assert (output_dir / f"{test_data_dir.round_name}-{c}.pkl").exists()
#         assert (output_dir / f"{test_data_dir.round_name}-{c}.png").exists()


# # Test the main workflow with real files
# def test_run_with_extractor(test_data_dir: TestDataDir, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Test the main workflow with real files and mocked BaSiC."""

#     # Define a mock BaSiC class and fit_basic function for faster testing
#     class MockBaSiC:
#         def __init__(self) -> None:
#             self.flatfield = np.ones((test_data_dir.img_size, test_data_dir.img_size))
#             self.darkfield = np.zeros((test_data_dir.img_size, test_data_dir.img_size))

#     def mock_fit_basic(data: np.ndarray, channel: int) -> MockBaSiC:
#         return MockBaSiC()

#     # Patch the fit_basic function to use our mock
#     monkeypatch.setattr("fishtools.preprocess.basic.fit_basic", mock_fit_basic)

#     # Need to patch plot_basic too
#     def mock_plot_basic(basic: MockBaSiC) -> None:
#         pass

#     monkeypatch.setattr("fishtools.preprocess.basic.plot_basic", mock_plot_basic)

#     # Run the workflow
#     results = run_with_extractor(
#         test_data_dir.dir, test_data_dir.round_name, extract_data_from_tiff, plot=False, zs=[0.5]
#     )

#     # Check results
#     assert len(results) == test_data_dir.nc

#     # Verify output files
#     for c in range(test_data_dir.nc):
#         assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.pkl").exists()
#         assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.png").exists()


# # Test CLI run command using CliRunner
# def test_cli_run_command(test_data_dir: TestDataDir, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Test CLI run command with CliRunner."""

#     # Define a mock BaSiC class and fit_basic function for faster testing
#     class MockBaSiC:
#         def __init__(self) -> None:
#             self.flatfield = np.ones((test_data_dir.img_size, test_data_dir.img_size))
#             self.darkfield = np.zeros((test_data_dir.img_size, test_data_dir.img_size))

#     def mock_fit_basic(data: np.ndarray, channel: int) -> MockBaSiC:
#         return MockBaSiC()

#     # Patch the fit_basic function to use our mock
#     monkeypatch.setattr("fishtools.preprocess.cli_basic.fit_basic", mock_fit_basic)

#     # Need to patch plot_basic too
#     def mock_plot_basic(basic: MockBaSiC) -> None:
#         pass

#     monkeypatch.setattr("fishtools.preprocess.cli_basic.plot_basic", mock_plot_basic)

#     # Create a CliRunner for testing
#     runner = CliRunner()

#     # Run the CLI command with mocking
#     try:
#         # Override the minimum file requirement
#         with patch("fishtools.preprocess.cli_basic.len") as mock_len:
#             # Make it think we have plenty of files
#             mock_len.return_value = 150

#             # Run the command using the CliRunner
#             result = runner.invoke(
#                 basic, ["run", str(test_data_dir.dir), test_data_dir.round_name, "--overwrite", "--zs", "0.5"]
#             )

#             # Check exit code
#             assert result.exit_code == 0

#             # Verify expected output messages in result.output
#             assert f"Running {test_data_dir.round_name}" in result.output

#             # Verify output files
#             for c in range(test_data_dir.nc):
#                 assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.pkl").exists()
#                 assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.png").exists()

#     except ValueError as e:
#         if "Not enough files" in str(e):
#             pytest.skip("Skip test due to not enough files - this is expected in some cases")
#         else:
#             raise


# # Test CLI batch command using CliRunner
# def test_cli_batch_command(test_data_dir: TestDataDir, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Test CLI batch command with CliRunner."""

#     # Define a mock BaSiC class and fit_basic function for faster testing
#     class MockBaSiC:
#         def __init__(self) -> None:
#             self.flatfield = np.ones((test_data_dir.img_size, test_data_dir.img_size))
#             self.darkfield = np.zeros((test_data_dir.img_size, test_data_dir.img_size))

#     def mock_fit_basic(data: np.ndarray, channel: int) -> MockBaSiC:
#         return MockBaSiC()

#     # Patch the fit_basic function to use our mock
#     monkeypatch.setattr("fishtools.preprocess.cli_basic.fit_basic", mock_fit_basic)

#     # Need to patch plot_basic too
#     def mock_plot_basic(basic: MockBaSiC) -> None:
#         pass

#     monkeypatch.setattr("fishtools.preprocess.cli_basic.plot_basic", mock_plot_basic)

#     # Create a CliRunner for testing
#     runner = CliRunner()

#     # Run the CLI command with mocking
#     try:
#         # Override the minimum file requirement
#         with (
#             patch("fishtools.preprocess.cli_basic.len") as mock_len,
#             patch("fishtools.preprocess.basic.len") as basic_mock_len,
#         ):
#             # Make it think we have plenty of files
#             mock_len.return_value = 150
#             basic_mock_len.return_value = 150

#             # Run the command using the CliRunner
#             result = runner.invoke(
#                 basic, ["batch", str(test_data_dir.dir), "--overwrite", "--threads", "1", "--zs", "0.5"]
#             )

#             # Check exit code
#             assert result.exit_code == 0

#             # Verify output files
#             # Since batch processes all rounds, we should at least have the main round files
#             for c in range(test_data_dir.nc):
#                 assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.pkl").exists()
#                 assert (test_data_dir.dir / "basic" / f"{test_data_dir.round_name}-{c}.png").exists()

#     except ValueError as e:
#         if "Not enough files" in str(e):
#             pytest.skip("Skip test due to not enough files - this is expected in some cases")
#         else:
#             raise


# # Test CLI run command error handling
# def test_cli_run_invalid_input(test_data_dir: TestDataDir) -> None:
#     """Test CLI run command handles invalid inputs gracefully."""
#     runner = CliRunner()

#     # Test with non-existent directory
#     result = runner.invoke(basic, ["run", "/path/does/not/exist", "some_round"])
#     assert result.exit_code != 0

#     # Test with invalid z values
#     result = runner.invoke(basic, ["run", str(test_data_dir.dir), test_data_dir.round_name, "--zs", "2.0"])
#     # Should fail with ValueError about zs being between 0 and 1
#     assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main(["-xvs"])
