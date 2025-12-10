"""Tests for the salvage workflow (high-shift round recovery)."""

from pathlib import Path

import numpy as np
import pytest
from tifffile import imread, imwrite

from fishtools.preprocess.cli_stitch import extract


class TestExtractWithFiducials:
    """Tests for extract function with fiducial handling."""

    def test_fiducials_saved_to_correct_structure(self, tmp_path: Path):
        """Fiducials should be saved as fid/00/, fid/01/ structure."""
        # Create [ZC]YX image with 2 fiducial frames
        # 3 channels, 4 Z slices = 12 frames + 2 fiducials = 14 frames
        n_channels = 3
        n_z = 4
        n_fids = 2
        h, w = 64, 64

        img = np.random.randint(0, 1000, (n_z * n_channels + n_fids, h, w), dtype=np.uint16)

        # Mark fiducial frames with distinct values for verification
        img[-2, :, :] = 5000  # Fiducial 0
        img[-1, :, :] = 6000  # Fiducial 1

        input_file = tmp_path / "input" / "1_9_17-0001.tif"
        input_file.parent.mkdir(parents=True)
        imwrite(input_file, img)

        out_path = tmp_path / "output"

        extract(
            path=input_file,
            out_path=out_path,
            n_channels_reshape=n_channels,
            n_fids=n_fids,
            include_fiducials=True,
            downsample=1,
            trim=0,
        )

        # Check fiducial structure exists
        assert (out_path / "fid" / "00").exists(), "fid/00/ should exist"
        assert (out_path / "fid" / "01").exists(), "fid/01/ should exist"

        # Check fiducial tiles exist
        fid0_tile = out_path / "fid" / "00" / "0001.tif"
        fid1_tile = out_path / "fid" / "01" / "0001.tif"
        assert fid0_tile.exists(), "Fiducial 0 tile should exist"
        assert fid1_tile.exists(), "Fiducial 1 tile should exist"

        # Verify fiducial values are correct
        fid0_data = imread(fid0_tile)
        fid1_data = imread(fid1_tile)
        assert fid0_data.mean() > 4000, "Fiducial 0 should have high values"
        assert fid1_data.mean() > 5000, "Fiducial 1 should have high values"

    def test_main_channels_reshaped_correctly(self, tmp_path: Path):
        """Main channels should be reshaped from [ZC]YX to ZCYX."""
        n_channels = 3
        n_z = 4
        n_fids = 2
        h, w = 64, 64

        # Create [ZC]YX image - interleaved as Z0C0, Z0C1, Z0C2, Z1C0, ...
        img = np.zeros((n_z * n_channels + n_fids, h, w), dtype=np.uint16)

        # Mark each channel with distinct value for verification
        for z in range(n_z):
            for c in range(n_channels):
                frame_idx = z * n_channels + c
                img[frame_idx, :, :] = (c + 1) * 1000  # C0=1000, C1=2000, C2=3000

        input_file = tmp_path / "input" / "1_9_17-0001.tif"
        input_file.parent.mkdir(parents=True)
        imwrite(input_file, img)

        out_path = tmp_path / "output"

        extract(
            path=input_file,
            out_path=out_path,
            n_channels_reshape=n_channels,
            n_fids=n_fids,
            include_fiducials=False,
            downsample=1,
            trim=0,
        )

        # Check main channel structure: ZZ/CC/tile.tif
        for z in range(n_z):
            for c in range(n_channels):
                tile_path = out_path / f"{z:02d}" / f"{c:02d}" / "0001.tif"
                assert tile_path.exists(), f"Tile Z={z} C={c} should exist"

                tile_data = imread(tile_path)
                expected_val = (c + 1) * 1000
                assert tile_data.mean() > expected_val - 100, f"Channel {c} should have value ~{expected_val}"

    def test_no_fiducials_when_disabled(self, tmp_path: Path):
        """Fiducials should not be saved when include_fiducials=False."""
        n_channels = 3
        n_z = 2
        n_fids = 2
        h, w = 32, 32

        img = np.random.randint(0, 1000, (n_z * n_channels + n_fids, h, w), dtype=np.uint16)

        input_file = tmp_path / "input" / "1_9_17-0001.tif"
        input_file.parent.mkdir(parents=True)
        imwrite(input_file, img)

        out_path = tmp_path / "output"

        extract(
            path=input_file,
            out_path=out_path,
            n_channels_reshape=n_channels,
            n_fids=n_fids,
            include_fiducials=False,
            downsample=1,
            trim=0,
        )

        # Fiducial folder should not exist
        assert not (out_path / "fid").exists(), "fid/ should not exist when disabled"

    def test_reshape_error_on_indivisible_frames(self, tmp_path: Path):
        """Should raise error if frame count not divisible by channel count."""
        n_channels = 3
        n_fids = 2
        h, w = 32, 32

        # Create image with wrong number of frames (not divisible by 3)
        img = np.random.randint(0, 1000, (10 + n_fids, h, w), dtype=np.uint16)  # 10 not divisible by 3

        input_file = tmp_path / "input" / "1_9_17-0001.tif"
        input_file.parent.mkdir(parents=True)
        imwrite(input_file, img)

        out_path = tmp_path / "output"

        with pytest.raises(ValueError, match="not divisible"):
            extract(
                path=input_file,
                out_path=out_path,
                n_channels_reshape=n_channels,
                n_fids=n_fids,
                include_fiducials=False,
                downsample=1,
                trim=0,
            )
