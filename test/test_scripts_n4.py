from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from tifffile import imread

from fishtools.preprocess import n4


def _create_fused_store(path: Path, data: np.ndarray) -> None:
    _, y_dim, x_dim, _ = data.shape
    store = zarr.open_array(
        path / "fused.zarr",
        mode="w",
        shape=data.shape,
        chunks=(1, y_dim, x_dim, 1),
        dtype=data.dtype,
    )
    store[...] = data


def test_compute_field_from_workspace_creates_correction_field(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    data = np.zeros((2, 8, 8, 1), dtype=np.uint16)
    data[0, 2:6, 2:6, 0] = 100
    data[1, 3:7, 3:7, 0] = 400
    _create_fused_store(fused_dir, data)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=2,
        spline_lowres_px=32.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=None,
        apply_correction=False,
        overwrite=True,
    )

    result = n4.compute_field_from_workspace(config)

    assert result.field_path.exists()
    assert result.corrected_path is None

    field = imread(result.field_path)
    assert field.shape[-2:] == (8, 8)
    assert field.dtype == np.float32
    assert np.isfinite(field).all()
    assert np.isfinite(field).all()
    assert field.mean() > 0


def test_compute_field_from_workspace_applies_correction(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    field = np.ones((8, 8), dtype=np.float32)
    field[2:6, 2:6] = 2.0

    biased = np.zeros((3, 8, 8, 1), dtype=np.float32)
    for z in range(3):
        biased[z, :, :, 0] = (z + 1) * field
    _create_fused_store(fused_dir, biased.astype(np.uint16))

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=16.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=fused_dir / "corrected.zarr",
        apply_correction=True,
        overwrite=True,
        debug=True,
    )

    result = n4.compute_field_from_workspace(config)
    assert result.corrected_path is not None

    corrected = zarr.open_array(result.corrected_path, mode="r")
    assert corrected.shape == (3, 8, 8, 1)
    wf = imread(result.field_path).astype(np.float32)
    written_field = wf[0] if wf.ndim == 3 else wf

    fused = zarr.open_array(fused_dir / "fused.zarr", mode="r")
    corrected_float = []
    for z in range(3):
        plane = np.asarray(fused[z, :, :, 0], dtype=np.float32)
        corrected_float.append(plane / written_field)

    quant = dict(corrected.attrs["quantization"])
    channel_meta = dict(quant["channels"][0])
    lower = float(channel_meta["lower"])
    scale = float(channel_meta["scale"])
    dtype_info = np.iinfo(np.uint16)
    corrected_float_arr = np.stack(corrected_float, axis=0)
    stored_uint16 = np.asarray(corrected, dtype=np.uint16).squeeze(axis=-1)
    recovered = stored_uint16.astype(np.float32) / scale + lower
    diff = np.abs(recovered - corrected_float_arr)
    inside_mask = (stored_uint16 > 0) & (stored_uint16 < dtype_info.max)
    if np.any(inside_mask):
        assert diff[inside_mask].max() <= 1e-2
    below_mask = stored_uint16 == 0
    if np.any(below_mask):
        assert np.all(corrected_float_arr[below_mask] <= lower + 2e-2)
    above_mask = stored_uint16 == dtype_info.max
    upper = float(channel_meta["upper"])
    if np.any(above_mask):
        assert np.all(corrected_float_arr[above_mask] >= upper - 2e-2)
    total_samples = quant["sampling"]["total_samples"]
    assert quant["lower_percentile"] == pytest.approx(n4.QUANT_LOWER_PERCENTILE)
    if total_samples < n4.QUANT_MIN_TOTAL_SAMPLES_FOR_HIGH_PERCENTILE:
        assert quant["upper_percentile"] == pytest.approx(n4.QUANT_FALLBACK_UPPER_PERCENTILE)
    else:
        assert quant["upper_percentile"] == pytest.approx(n4.QUANT_UPPER_PERCENTILE)

    float_store = zarr.open_array(
        result.corrected_path.with_name(f"{result.corrected_path.stem}_float32{result.corrected_path.suffix}"),
        mode="r",
    )
    assert float_store.shape == corrected.shape
    np.testing.assert_allclose(
        np.asarray(float_store).squeeze(axis=-1),
        corrected_float_arr,
        rtol=1e-4,
        atol=5e-2,
    )


def test_compute_correction_field_requires_positive_mask() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Mask is empty after thresholding >0"):
        n4.compute_correction_field(
            image,
            shrink=1,
            spline_lowres_px=16.0,
        )


def test_apply_correction_field_divides_by_field() -> None:
    image = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    field = np.array([[2.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    corrected = n4.apply_correction_field(image, field)

    expected = np.array([[1.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-6)


def test_apply_correction_field_rejects_non_positive_values() -> None:
    image = np.ones((2, 2), dtype=np.float32)
    field = np.ones((2, 2), dtype=np.float32)
    field[0, 0] = 0.0

    with pytest.raises(ValueError, match="Field must be strictly positive"):
        n4.apply_correction_field(image, field)


def test_apply_correction_to_store_outputs_corrected_zarr(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    base = np.stack(
        [
            np.full((8, 8), 100.0, dtype=np.float32),
            np.full((8, 8), 50.0, dtype=np.float32),
        ]
    )
    field = np.linspace(1.0, 2.0, 64, dtype=np.float32).reshape(8, 8)
    biased = base * field
    data = np.stack([biased, biased], axis=-1)  # duplicate channel for unrelated data
    _create_fused_store(fused_dir, data.astype(np.uint16))

    output = fused_dir / "n4_corrected.zarr"
    n4.apply_correction_to_store(
        fused_dir / "fused.zarr",
        channel_index=0,
        field=field,
        output_path=output,
        overwrite=True,
        debug=True,
    )

    corrected = zarr.open_array(output, mode="r")
    assert corrected.shape == (2, 8, 8, 1)
    assert corrected.dtype == np.uint16

    fused = zarr.open_array(fused_dir / "fused.zarr", mode="r")
    corrected_float = [np.asarray(fused[idx, :, :, 0], dtype=np.float32) / field for idx in range(2)]
    quant = dict(corrected.attrs["quantization"])
    lower = float(quant["lower"])
    scale = float(quant["scale"])
    dtype_info = np.iinfo(np.uint16)
    corrected_float_arr = np.stack(corrected_float, axis=0)
    expected_uint16 = np.clip(
        np.round(((corrected_float_arr[..., None]) - lower) * scale),
        0.0,
        dtype_info.max,
    ).astype(np.uint16)

    np.testing.assert_array_equal(np.asarray(corrected), expected_uint16)

    float_store = zarr.open_array(output.with_name(f"{output.stem}_float32{output.suffix}"), mode="r")
    np.testing.assert_allclose(
        np.asarray(float_store).squeeze(axis=-1),
        corrected_float_arr,
        rtol=1e-4,
        atol=5e-2,
    )


def test_apply_correction_to_store_float_debug_preserves_outlier(tmp_path: Path) -> None:
    fused_path = tmp_path / "fused.zarr"
    fused = zarr.open_array(
        fused_path,
        mode="w",
        shape=(1, 8, 8, 1),
        chunks=(1, 8, 8, 1),
        dtype=np.float32,
    )
    fused[:] = 10.0
    fused[0, 1, 1, 0] = 1000.0  # outlier not captured by stride-based sampling

    field = np.ones((8, 8), dtype=np.float32)
    output = tmp_path / "corrected.zarr"
    result = n4.apply_correction_to_store(
        fused_path,
        channel_index=0,
        field=field,
        output_path=output,
        overwrite=True,
        debug=True,
    )

    quant = zarr.open_array(result, mode="r")
    float_store = zarr.open_array(result.with_name(f"{result.stem}_float32{result.suffix}"), mode="r")
    dtype_info = np.iinfo(np.uint16)

    assert quant[0, 1, 1, 0] == dtype_info.max  # clipped in uint16
    assert float_store[0, 1, 1, 0] == pytest.approx(1000.0, rel=1e-6)
    assert quant.attrs["quantization"]["float_store"] == result.with_name(
        f"{result.stem}_float32{result.suffix}"
    ).name
