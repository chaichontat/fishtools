import json
import pickle
from pathlib import Path

import numpy as np
import tifffile


class _DummyBasic:
    def __init__(self, h: int, w: int) -> None:
        self.darkfield = np.zeros((h, w), dtype=np.float32)
        self.flatfield = np.ones((h, w), dtype=np.float32)


def _write_basic_profiles(base: Path, wavelengths: list[str], shape: tuple[int, int]) -> None:
    """Write BaSiC profiles under base/basic for the given wavelengths."""
    basic_dir = base / "basic"
    basic_dir.mkdir(parents=True, exist_ok=True)
    for wl in wavelengths:
        payload = _DummyBasic(*shape)
        with open(basic_dir / f"all-{wl}.pkl", "wb") as fh:
            pickle.dump({"basic": payload}, fh)


def _write_raw_tile(base: Path, round_name: str, roi: str, *, wavelengths: list[str]) -> Path:
    """Write a minimal raw tile embedding waveform metadata with given wavelengths."""
    tile_dir = base / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    # One plane per wavelength, plus two fid planes to exercise indexing logic in helpers.
    planes = len(wavelengths)
    payload = np.zeros((planes + 2, 8, 8), dtype=np.uint16)
    meta = {"waveform": json.dumps({"params": {"step": 0.6, "powers": wavelengths}})}
    tifffile.imwrite(tile_dir / f"{round_name}-0000.tif", payload, metadata=meta)
    return tile_dir / f"{round_name}-0000.tif"


def _write_deconv32_tile_with_names(base: Path, round_name: str, roi: str, names: list[str]) -> None:
    """Create a float32 staging tile carrying channel names (non-wavelength)."""
    tile_dir = base / "analysis" / "deconv32" / f"{round_name}--{roi}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    # Two payload planes; metadata encodes human-readable names.
    payload = np.zeros((len(names), 8, 8), dtype=np.float32)
    metadata = {"axes": "CYX", "key": names}
    tifffile.imwrite(tile_dir / f"{round_name}-0000.tif", payload, metadata=metadata)


def test_cli_deconv_prefers_wavelengths_for_basic_lookup(tmp_path: Path) -> None:
    """
    Regression test: cli_deconv should resolve BaSiC profiles using wavelengths,
    even when channel names are present in the deconv32 metadata.

    Before the fix, channel names like ["wga", "brdu"] were used, causing
    lookups for files like all-wga.pkl which do not exist. We assert that with
    wavelengths present in raw metadata, lookup succeeds against all-<wl>.pkl.
    """
    from fishtools.preprocess.cli_deconv import _prepare_round_plan, DeconvolutionOutputMode

    workspace = tmp_path / "ws"
    # Mark as a valid workspace per Workspace auto-detection (looks for *.DONE files)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "ROOT.DONE").write_text("ok")
    round_name = "r1"
    roi = "roi"
    wavelengths = ["405", "560"]

    raw_tile = _write_raw_tile(workspace, round_name, roi, wavelengths=wavelengths)
    _write_deconv32_tile_with_names(workspace, round_name, roi, names=["wga", "brdu"])
    _write_basic_profiles(workspace, wavelengths=wavelengths, shape=(8, 8))

    # Sanity: Only wavelength-named profiles exist.
    assert (workspace / "basic" / "all-wga.pkl").exists() is False
    assert (workspace / "basic" / "all-405.pkl").exists() is True
    assert (workspace / "basic" / "all-560.pkl").exists() is True

    # Prepare should succeed by using wavelengths (from raw metadata) for lookup.
    plan = _prepare_round_plan(
        path=workspace,
        round_name=round_name,
        files=[raw_tile],
        out_dir=workspace / "analysis" / "deconv",
        basic_name="all",
        n_fids=2,
        histogram_bins=16,
        load_scaling=False,
        overwrite=True,
        debug=False,
        label=None,
        mode=DeconvolutionOutputMode.F32,
    )

    assert plan is not None, "Plan construction should succeed when wavelengths are available."
