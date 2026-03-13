"""
Unit tests for selected utilities in align_prod.py that do not require heavy
dependencies. These focus on small, deterministic behaviors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
import xarray as xr

from fishtools.preprocess.spots.align_prod import (
    Deviation,
    Deviations,
    InitialScale,
    NormFieldCorrection,
    _gain_crop,
    _slice_bounds,
    append_json,
    batch,
    create_opt_path,
    generate_subtraction_matrix,
    get_blank_channel_info,
)
from fishtools.utils.io import CorruptedTiffError, Workspace


def test_get_blank_channel_info_mappings() -> None:
    # Basic mappings across the three wavelength groups
    assert get_blank_channel_info("1") == (560, 0)
    assert get_blank_channel_info("9") == (650, 1)
    assert get_blank_channel_info("17") == (750, 2)

    with pytest.raises(ValueError):
        get_blank_channel_info("999")


def test_generate_subtraction_matrix_shapes_and_values() -> None:
    # blanks dims: (r, c, z, y, x)
    r, c, z, y, x = 1, 3, 2, 4, 4
    blanks = xr.DataArray(
        np.linspace(0, 1, r * c * z * y * x, dtype=np.float32).reshape(r, c, z, y, x),
        dims=["r", "c", "z", "y", "x"],
    )

    # Channel keys we want to produce; they map to source blank indices [0,1,2,0,1,2]
    keys = ["1", "9", "17", "25", "26", "27"]

    # Provide coefficients for all required numeric channel_key values
    coefs = pl.DataFrame({
        "channel_key": [1, 9, 17, 25, 26, 27],
        "slope": [1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        "intercept": [0.0, 0.0, 0.0, 10.0, -5.0, 0.0],
    })

    out = generate_subtraction_matrix(blanks, coefs, keys)
    assert tuple(out.dims) == ("r", "c", "z", "y", "x")
    assert out.shape == (1, len(keys), 2, 4, 4)

    # Returned matrix is negated and floored at zero before negation -> values <= 0
    out_np = np.asarray(out.data)
    np.testing.assert_array_less(out_np, 1e-6)  # strictly negative or ~0

    blanks_np = np.asarray(blanks.data)
    # Channel 0 (key "1") should equal the negated blank values for slope=1, intercept=0
    np.testing.assert_allclose(np.asarray(out.sel(c=0).data), -blanks_np[:, 0], rtol=1e-6, atol=1e-6)

    # Channel 3 (key "25") maps back to blank index 0 with a positive intercept of 10
    intercept_offset = 10.0 / 65535.0
    expected_channel3 = -(blanks_np[:, 0] + intercept_offset)
    np.testing.assert_allclose(np.asarray(out.sel(c=3).data), expected_channel3, rtol=1e-6, atol=1e-6)

    # Missing parameters should raise a clear error for the specific key
    with pytest.raises(ValueError, match="channel key '999'"):
        generate_subtraction_matrix(blanks, coefs, keys + ["999"])  # type: ignore[arg-type]


def test_create_opt_path_variants(tmp_path: Path) -> None:
    # Build a minimal on-disk layout that matches expectations
    reg_dir = tmp_path / "registered--r1+cb1"
    reg_dir.mkdir(parents=True)
    tif_path = reg_dir / "reg0001.tif"
    tif_path.write_bytes(b"\x00")

    codebook_path = tmp_path / "cb1.json"
    codebook_path.write_text(json.dumps({"GeneA": [1], "Blank-1": [2]}))

    # JSON path
    p_json = create_opt_path(
        path_img=tif_path,
        codebook_path=codebook_path,
        mode="json",
        round_num=0,
        roi="roi1",
    )
    assert p_json.name.endswith(".json")
    assert p_json.parent.name == "opt_cb1+roi1"

    # PKL path (includes round suffix)
    p_pkl = create_opt_path(
        path_img=tif_path,
        codebook_path=codebook_path,
        mode="pkl",
        round_num=2,
        roi="roi1",
    )
    assert p_pkl.name.endswith("_opt02.pkl")
    assert p_pkl.parent.name == "opt_cb1+roi1"

    # Folder variant using path_folder
    p_folder = create_opt_path(
        path_folder=tmp_path,
        codebook_path=codebook_path,
        mode="folder",
        roi="roi1",
    )
    assert p_folder.name == "opt_cb1+roi1"


def test_norm_field_gain_crop_and_slice_bounds() -> None:
    # 10x10 full-res tile with a 2x2 gain grid.
    # grid bins: x [0..4]->0, [5..9]->1; same for y.
    gain = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    field = NormFieldCorrection(gain=gain, full_w=10, full_h=10, grid_w=2, grid_h=2)

    # Negative slice bounds should map to full-res indices.
    assert _slice_bounds(slice(-5, None), 10) == (5, 10)
    assert _slice_bounds(slice(None, -5), 10) == (0, 5)

    # Top-right quadrant crop: y [0..4], x [5..9] -> gain bin (0,1) == 2.0
    g_tr = _gain_crop(field, y0=0, y1=5, x0=5, x1=10)
    assert g_tr.shape == (5, 5)
    assert float(np.unique(g_tr)[0]) == pytest.approx(2.0)

    # Bottom-left quadrant crop: y [5..9], x [0..4] -> gain bin (1,0) == 3.0
    g_bl = _gain_crop(field, y0=5, y1=10, x0=0, x1=5)
    assert g_bl.shape == (5, 5)
    assert float(np.unique(g_bl)[0]) == pytest.approx(3.0)


def test_create_opt_path_invalid_argument_combinations(tmp_path: Path) -> None:
    codebook_path = tmp_path / "cb1.json"
    codebook_path.write_text("{}", encoding="utf-8")
    tif_path = tmp_path / "reg0001.tif"
    tif_path.write_bytes(b"\x00")

    with pytest.raises(ValueError):
        create_opt_path(codebook_path=codebook_path, mode="json")  # neither provided

    with pytest.raises(ValueError):
        create_opt_path(
            path_img=tif_path,
            path_folder=tmp_path,
            codebook_path=codebook_path,
            mode="json",
        )


def test_deviations_append_and_read(tmp_path: Path) -> None:
    jpath = tmp_path / "dev.json"

    # Round 0: initial scale + mins
    init_scale = np.array([1.0, 2.0, 3.0])
    mins = np.array([0.1, 0.2, 0.3])
    append_json(jpath, 0, initial_scale=init_scale, mins=mins)

    # Round 1: deviation + n
    dev = np.array([0.9, 1.1, 1.0])
    append_json(jpath, 1, deviation=dev, n=42, percent_blanks=0.2)

    objects = Deviations.validate_json(jpath.read_text())
    assert len(objects) == 2

    # Validate types and fields
    assert isinstance(objects[0], InitialScale)
    assert objects[0].round_num == 0
    assert np.allclose(objects[0].initial_scale, init_scale)
    assert np.allclose(objects[0].mins, mins)

    assert isinstance(objects[1], Deviation)
    assert objects[1].round_num == 1
    assert objects[1].n == 42
    assert np.allclose(objects[1].deviation, dev)


def test_batch_delete_corrupted_uses_workspace_helpers(monkeypatch, tmp_path: Path) -> None:
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
    (workspace_root / "workspace.DONE").touch()
    registered = workspace_root / "analysis" / "deconv" / "registered--cortex+cb"
    registered.mkdir(parents=True)

    valid = registered / "reg-0000.tif"
    tifffile.imwrite(valid, np.zeros((1, 1, 2, 2), dtype=np.uint16))

    corrupted = registered / "reg-0001.tif"
    corrupted.write_bytes(b"not-tiff")

    opt_dir = workspace_root / "analysis" / "deconv" / "opt_cb"
    opt_dir.mkdir(parents=True)
    np.savetxt(opt_dir / "global_scale.txt", np.ones((2, 2)))

    codebook_path = tmp_path / "cb.json"
    codebook_path.write_text("{}")

    checked: list[Path] = []

    real_ensure = Workspace.ensure_tiff_readable

    def fake_ensure(path: Path) -> None:
        checked.append(path)
        if path == corrupted:
            raise CorruptedTiffError(path, ValueError("corrupted"))
        real_ensure(path)

    monkeypatch.setattr(Workspace, "ensure_tiff_readable", staticmethod(fake_ensure))

    captured: list[list[Path]] = []

    def fake_batch(paths: list[Path], command: str, args: list[str], **kwargs) -> None:  # type: ignore[override]
        captured.append(list(paths))

    monkeypatch.setattr("fishtools.preprocess.spots.align_prod._batch", fake_batch)

    batch.callback(
        path=workspace_root,
        roi="cortex",
        codebook_path=codebook_path,
        threads=1,
        overwrite=False,
        simple=False,
        split=False,
        since=None,
        delete_corrupted=True,
        local_opt=False,
        blank=None,
        json_config=None,
        stagger=0.0,
        stagger_jitter=0.0,
    )

    assert captured, "_batch should be invoked with files to process"
    assert captured[0] == [valid]
    assert not corrupted.exists()
    assert set(checked) == {valid, corrupted}
