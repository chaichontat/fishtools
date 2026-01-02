"""Tests for quantize command deletion verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess


def _make_done_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.DONE").touch()
    return workspace


def _setup_workspace_for_quantize(
    workspace: Path,
    round_name: str,
    roi: str,
    tile_names: list[str],
    n_fids: int = 2,
) -> None:
    """Set up a workspace with float32 tiles ready for quantization."""
    (workspace / "workspace.DONE").touch()

    # Create raw directory with fiducial metadata
    raw_dir = workspace / f"{round_name}--{roi}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create deconv32 directory with float32 tiles
    deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
    deconv32_dir.mkdir(parents=True, exist_ok=True)

    # Create scaling file
    scaling_dir = workspace / "analysis" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)
    m_glob = np.array([10.0, 100.0], dtype=np.float32)
    s_glob = np.array([1000.0, 500.0], dtype=np.float32)
    np.savetxt(scaling_dir / f"{round_name}.txt", np.vstack([m_glob, s_glob]))

    z_slices = 1
    channels = 2
    height = width = 2

    for tile_name in tile_names:
        # Float32 deconv tile
        float32_payload = np.zeros((z_slices * channels, height, width), dtype=np.float32)
        tifffile.imwrite(deconv32_dir / tile_name, float32_payload)

        # Raw tile with fiducial info
        fid_planes = np.zeros((n_fids, height, width), dtype=np.uint16)
        raw_payload = np.concatenate(
            [np.zeros_like(float32_payload, dtype=np.uint16), fid_planes], axis=0
        )
        raw_metadata = {
            "waveform": json.dumps({"params": {"powers": ["bit001", "bit002"]}}),
        }
        tifffile.imwrite(raw_dir / tile_name, raw_payload, metadata=raw_metadata)


class TestQuantizeDeletionVerification:
    """Test that deconv32 directories are only deleted when all tiles are quantized."""

    def test_deletes_deconv32_when_all_tiles_quantized(self, tmp_path: Path) -> None:
        """When all tiles are quantized successfully, deconv32 should be deleted."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        round_name = "1_2"
        roi = "cortex"
        tiles = [f"{round_name}-0001.tif", f"{round_name}-0002.tif"]

        _setup_workspace_for_quantize(workspace, round_name, roi, tiles)

        deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
        assert deconv32_dir.exists()

        runner = CliRunner()
        result = runner.invoke(
            preprocess,
            ["deconv", "quantize", str(workspace), round_name, "--n-fids", "2"],
        )
        assert result.exit_code == 0, result.output

        deconv_dir = workspace / "analysis" / "deconv" / f"{round_name}--{roi}"
        produced = {p.name for p in deconv_dir.glob("*.tif")}
        assert set(tiles).issubset(produced)

        assert not deconv32_dir.exists(), "deconv32 should be deleted after successful quantization"


class TestDeleteVerifiedDeconv32:
    """Unit tests for _delete_verified_deconv32 function."""

    def test_deletes_when_all_tiles_quantized(self, tmp_path: Path) -> None:
        """Deletes deconv32 when all float32 tiles have quantized counterparts."""
        from fishtools.io.workspace import Workspace
        from fishtools.preprocess.cli_deconv import _delete_verified_deconv32

        workspace = _make_done_workspace(tmp_path)
        round_name = "1_2"
        roi = "cortex"

        # Create deconv32 with 2 tiles
        deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
        deconv32_dir.mkdir(parents=True)
        (deconv32_dir / f"{round_name}-0001.tif").touch()
        (deconv32_dir / f"{round_name}-0002.tif").touch()

        # Create deconv with same 2 tiles
        deconv_dir = workspace / "analysis" / "deconv" / f"{round_name}--{roi}"
        deconv_dir.mkdir(parents=True)
        (deconv_dir / f"{round_name}-0001.tif").touch()
        (deconv_dir / f"{round_name}-0002.tif").touch()

        ws = Workspace(workspace)
        _delete_verified_deconv32(ws, round_name)

        assert not deconv32_dir.exists(), "deconv32 should be deleted when all tiles verified"

    def test_keeps_when_tiles_missing(self, tmp_path: Path) -> None:
        """Keeps deconv32 when some float32 tiles are not quantized."""
        from fishtools.io.workspace import Workspace
        from fishtools.preprocess.cli_deconv import _delete_verified_deconv32

        workspace = _make_done_workspace(tmp_path)
        round_name = "1_2"
        roi = "cortex"

        # Create deconv32 with 2 tiles
        deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
        deconv32_dir.mkdir(parents=True)
        (deconv32_dir / f"{round_name}-0001.tif").touch()
        (deconv32_dir / f"{round_name}-0002.tif").touch()

        # Create deconv with only 1 tile (tile 2 is missing)
        deconv_dir = workspace / "analysis" / "deconv" / f"{round_name}--{roi}"
        deconv_dir.mkdir(parents=True)
        (deconv_dir / f"{round_name}-0001.tif").touch()

        ws = Workspace(workspace)
        _delete_verified_deconv32(ws, round_name)

        assert deconv32_dir.exists(), "deconv32 should be kept when tiles are missing"
        assert (deconv32_dir / f"{round_name}-0002.tif").exists()

    def test_keeps_when_output_dir_missing(self, tmp_path: Path) -> None:
        """Keeps deconv32 when quantized output directory doesn't exist."""
        from fishtools.io.workspace import Workspace
        from fishtools.preprocess.cli_deconv import _delete_verified_deconv32

        workspace = _make_done_workspace(tmp_path)
        round_name = "1_2"
        roi = "cortex"

        # Create deconv32 with tiles
        deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
        deconv32_dir.mkdir(parents=True)
        (deconv32_dir / f"{round_name}-0001.tif").touch()

        # Do NOT create deconv directory

        ws = Workspace(workspace)
        _delete_verified_deconv32(ws, round_name)

        assert deconv32_dir.exists(), "deconv32 should be kept when output dir missing"

    def test_handles_multiple_rois_independently(self, tmp_path: Path) -> None:
        """Each ROI is verified independently."""
        from fishtools.io.workspace import Workspace
        from fishtools.preprocess.cli_deconv import _delete_verified_deconv32

        workspace = _make_done_workspace(tmp_path)
        round_name = "1_2"

        # ROI 1: complete (should be deleted)
        deconv32_roi1 = workspace / "analysis" / "deconv32" / f"{round_name}--roi1"
        deconv32_roi1.mkdir(parents=True)
        (deconv32_roi1 / f"{round_name}-0001.tif").touch()
        deconv_roi1 = workspace / "analysis" / "deconv" / f"{round_name}--roi1"
        deconv_roi1.mkdir(parents=True)
        (deconv_roi1 / f"{round_name}-0001.tif").touch()

        # ROI 2: incomplete (should be kept)
        deconv32_roi2 = workspace / "analysis" / "deconv32" / f"{round_name}--roi2"
        deconv32_roi2.mkdir(parents=True)
        (deconv32_roi2 / f"{round_name}-0001.tif").touch()
        (deconv32_roi2 / f"{round_name}-0002.tif").touch()
        deconv_roi2 = workspace / "analysis" / "deconv" / f"{round_name}--roi2"
        deconv_roi2.mkdir(parents=True)
        (deconv_roi2 / f"{round_name}-0001.tif").touch()  # Missing 0002

        ws = Workspace(workspace)
        _delete_verified_deconv32(ws, round_name)

        assert not deconv32_roi1.exists(), "Complete ROI should be deleted"
        assert deconv32_roi2.exists(), "Incomplete ROI should be kept"
