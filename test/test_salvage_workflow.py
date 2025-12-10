"""Tests for the salvage workflow (high-shift round recovery)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr
from tifffile import imread, imwrite

from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_stitch import extract, slice_mosaic
from fishtools.preprocess.tileconfig import TileConfiguration


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


class TestSliceMosaicIntegration:
    """Integration-style tests for slice_mosaic coordinate logic."""

    def test_slice_mosaic_recovers_shifted_tile(self, tmp_path: Path) -> None:
        """slice_mosaic should reconstruct tiles from shifted mosaic using original positions."""
        workspace = tmp_path / "ws"
        deconv_root = workspace / "analysis" / "deconv"
        deconv_root.mkdir(parents=True)
        (workspace / "workspace.DONE").write_text("ok")

        roi = "roi1"
        round_name = "1_9_17"
        tile_size = 8

        ws = Workspace(deconv_root)
        stitch_dir = ws.deconved / f"stitch--{roi}--shifted-{round_name}"
        stitch_dir.mkdir(parents=True)

        # Synthetic mosaic: one Z-plane, one channel
        mosaic_h, mosaic_w = 32, 32
        zarr_path = stitch_dir / "fused.zarr"
        arr = zarr.open_array(zarr_path, mode="w", shape=(1, mosaic_h, mosaic_w, 1), dtype=np.uint16)
        arr[:] = 0

        # Original vs shifted tile positions (world coordinates)
        x_orig, y_orig = 10.0, 20.0
        dx, dy = 5.0, 7.0  # coarse shift (content moved right/down)
        x_shifted = x_orig - dx
        y_shifted = y_orig - dy

        # According to slice_mosaic: origin = min(shifted), slice = original - origin
        slice_x = int(round(x_orig - x_shifted))
        slice_y = int(round(y_orig - y_shifted))

        # Place a distinguishable tile payload at the expected slice location
        tile_payload = np.arange(tile_size * tile_size, dtype=np.uint16).reshape(tile_size, tile_size)
        arr[0, slice_y : slice_y + tile_size, slice_x : slice_x + tile_size, 0] = tile_payload

        # Write ORIGINAL TileConfiguration.registered.txt
        original_tc_path = ws.tileconfig_dir(roi) / "TileConfiguration.registered.txt"
        original_tc_path.parent.mkdir(parents=True, exist_ok=True)
        df_orig = pl.DataFrame(
            {"index": [1], "x": [x_orig], "y": [y_orig]},
            schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32}),
        )
        TileConfiguration(df_orig).write(original_tc_path)

        # Write SHIFTED TileConfiguration.shifted.txt
        shifted_tc_path = stitch_dir / "TileConfiguration.shifted.txt"
        df_shift = pl.DataFrame(
            {"index": [1], "x": [x_shifted], "y": [y_shifted]},
            schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32}),
        )
        TileConfiguration(df_shift).write(shifted_tc_path)

        # Create empty source round dir so slice_mosaic can resolve metadata (optional)
        (ws.deconved / f"{round_name}--{roi}").mkdir(parents=True, exist_ok=True)

        # Run slice_mosaic against the synthetic workspace
        slice_mosaic(
            path=deconv_root,
            roi=roi,
            round_name=round_name,
            tile_size=tile_size,
            overwrite=True,
        )

        repaired_dir = ws.deconved / f"{round_name}--{roi}--repaired"
        out_path = repaired_dir / f"{round_name}-0001.tif"
        assert out_path.exists()

        out = imread(out_path)
        assert out.shape == (1, tile_size, tile_size)
        np.testing.assert_array_equal(out[0], tile_payload)
