from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from skimage import filters as sk_filters
from tifffile import imwrite as tif_imwrite

# Provide a CuCIM stand-in backed by skimage filters so imports succeed in tests.
if "cucim.skimage.filters" not in sys.modules:
    cucim_module = sys.modules.setdefault("cucim", types.ModuleType("cucim"))
    cucim_skimage = sys.modules.setdefault("cucim.skimage", types.ModuleType("cucim.skimage"))
    cucim_filters = types.ModuleType("cucim.skimage.filters")
    cucim_filters.unsharp_mask = sk_filters.unsharp_mask
    cucim_skimage.filters = cucim_filters
    cucim_module.skimage = cucim_skimage
    sys.modules["cucim.skimage.filters"] = cucim_filters

from fishtools.preprocess import n4
from fishtools.utils.zarr_utils import default_zarr_codecs

pytestmark = pytest.mark.timeout(30)


def _create_fused_store(path: Path, data: np.ndarray) -> None:
    _, y_dim, x_dim, _ = data.shape
    store = zarr.open_array(
        path / "fused.zarr",
        mode="w",
        shape=data.shape,
        chunks=(1, y_dim, x_dim, 1),
        dtype=data.dtype,
        codecs=default_zarr_codecs(),
    )
    store[...] = data


def _fake_cp() -> types.SimpleNamespace:
    def _asarray(arr: np.ndarray, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(arr, dtype=dtype)

    ns = types.SimpleNamespace()
    ns.asarray = _asarray
    ns.float32 = np.float32
    ns.bool_ = np.bool_
    ns.ndarray = np.ndarray
    ns.where = np.where
    ns.asnumpy = lambda arr: np.asarray(arr)
    ns.empty = lambda shape, dtype=None: np.empty(shape, dtype=dtype if dtype is not None else np.float32)
    ns.nan_to_num = np.nan_to_num
    return ns


def test_compute_correction_field_requires_positive_mask() -> None:
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Mask is empty after thresholding >0"):
        n4.compute_correction_field(
            image,
            shrink=1,
            spline_lowres_px=16.0,
        )


def test_compute_correction_field_numeric_threshold_empty_mask() -> None:
    image = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=r"> 10.0; check the selected channel"):
        n4.compute_correction_field(
            image,
            shrink=1,
            spline_lowres_px=4.0,
            threshold=10.0,
        )


def test_compute_correction_field_method_threshold_empty_mask() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"skimage\.filters\.threshold_otsu"):
        n4.compute_correction_field(
            image,
            shrink=1,
            spline_lowres_px=8.0,
            threshold="otsu",
        )


def test_unsharp_mask_helper_executes_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cp = _fake_cp()
    monkeypatch.setattr(n4, "cp", fake_cp)
    monkeypatch.setattr(
        n4,
        "cucim_filters",
        # Accept radius kwarg to mirror production call signature
        types.SimpleNamespace(unsharp_mask=lambda arr, radius=3, preserve_range=True, **kwargs: arr * 2.0),
    )

    image = np.ones((3, 3), dtype=np.float32)
    mask = np.zeros_like(image, dtype=bool)
    mask[1, 1] = True

    result = n4._apply_unsharp_mask_if_enabled(image, mask=mask, enabled=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
    assert np.isclose(result[1, 1], 2.0)
    assert np.allclose(result[mask == 0], 1.0)


def test_unsharp_mask_preprocesses_n4_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    data = np.zeros((1, 6, 6, 1), dtype=np.uint16)
    data[0, 2:4, 2:4, 0] = 100
    _create_fused_store(fused_dir, data)

    fake_cp = _fake_cp()
    fake_cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setattr(n4, "cp", fake_cp)
    monkeypatch.setattr(
        n4,
        "cucim_filters",
        # Accept radius kwarg to mirror production call signature
        types.SimpleNamespace(unsharp_mask=lambda arr, radius=3, preserve_range=True, **kwargs: arr + 5.0),
    )

    observed: dict[str, np.ndarray] = {}

    def fake_compute(image: np.ndarray, **_: Any) -> np.ndarray:
        observed["input"] = np.asarray(image, dtype=np.float32)
        return np.ones_like(image, dtype=np.float32)

    monkeypatch.setattr(n4, "compute_correction_field", fake_compute)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=None,
        apply_correction=False,
        overwrite=True,
        use_unsharp_mask=True,
    )

    n4.compute_fields_from_workspace(config)

    assert "input" in observed
    mask = data[0, :, :, 0] > 0
    original_plane = data[0, :, :, 0].astype(np.float32)
    expected = np.where(mask, original_plane + 5.0, original_plane)
    np.testing.assert_allclose(observed["input"], expected)


def test_unsharp_mask_applies_during_correction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    data = np.zeros((1, 4, 4, 1), dtype=np.uint16)
    data[0, 1:3, 1:3, 0] = 50
    _create_fused_store(fused_dir, data)

    # Simplify GPU helpers
    monkeypatch.setattr(n4, "_prepare_gpu_field", lambda field: np.asarray(field, dtype=np.float32))

    masks_seen: list[np.ndarray] = []

    def fake_correct_plane_gpu(
        plane: np.ndarray,
        *,
        field_gpu: np.ndarray,
        use_unsharp_mask: bool,
        mask_cpu: np.ndarray | None = None,
    ) -> np.ndarray:
        if mask_cpu is not None:
            masks_seen.append(np.asarray(mask_cpu, dtype=bool))
        return np.asarray(plane, dtype=np.float32) / np.asarray(field_gpu, dtype=np.float32)

    monkeypatch.setattr(n4, "_correct_plane_gpu", fake_correct_plane_gpu)
    monkeypatch.setattr(
        n4, "compute_correction_field", lambda *args, **kwargs: np.ones((4, 4), dtype=np.float32)
    )
    fake_cp = _fake_cp()
    fake_cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setattr(n4, "cp", fake_cp)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=fused_dir / "corrected.zarr",
        apply_correction=True,
        overwrite=True,
        use_unsharp_mask=True,
    )

    n4.compute_fields_from_workspace(config)

    assert len(masks_seen) == 2  # reference plane + correction loop
    expected_mask = data[0, :, :, 0] > 0
    for mask in masks_seen:
        assert mask.shape == expected_mask.shape
        assert np.array_equal(mask, expected_mask)


def test_compute_fields_from_workspace_records_threshold_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    data = np.zeros((1, 6, 6, 1), dtype=np.uint16)
    data[..., 0] = 50
    _create_fused_store(fused_dir, data)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=None,
        apply_correction=False,
        overwrite=True,
        threshold="otsu",
    )

    monkeypatch.setattr(
        n4,
        "compute_correction_field",
        lambda *args, **kwargs: np.ones((6, 6), dtype=np.float32),
    )

    captured_meta: dict[str, Any] = {}

    def _spy_imwrite(path: Path, data: Any, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            captured_meta.update(metadata)
        cleaned = {k: v for k, v in kwargs.items() if k != "metadata"}
        tif_imwrite(path, data, **cleaned)

    monkeypatch.setattr(n4, "imwrite", _spy_imwrite)

    result = n4.compute_fields_from_workspace(config)
    assert result[0].field_path.exists()

    assert "n4" in captured_meta
    threshold_meta = captured_meta["n4"].get("threshold")
    assert threshold_meta == {"kind": "method", "function": "threshold_otsu"}


def test_compute_fields_from_workspace_corrects_zarr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    base_planes = np.stack([
        np.full((8, 8), 100.0, dtype=np.float32),
        np.full((8, 8), 50.0, dtype=np.float32),
    ])
    field = np.linspace(1.0, 2.0, 64, dtype=np.float32).reshape(8, 8)
    biased = base_planes * field
    data = np.stack([biased, biased], axis=-1)
    _create_fused_store(fused_dir, data.astype(np.uint16))

    monkeypatch.setattr(n4, "compute_correction_field", lambda *args, **kwargs: field.copy())
    monkeypatch.setattr(n4, "_prepare_gpu_field", lambda arr: np.asarray(arr, dtype=np.float32))

    def fake_correct_plane_gpu(
        plane: np.ndarray,
        *,
        field_gpu: np.ndarray,
        use_unsharp_mask: bool,
        mask_cpu: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.asarray(plane, dtype=np.float32) / np.asarray(field_gpu, dtype=np.float32)

    monkeypatch.setattr(n4, "_correct_plane_gpu", fake_correct_plane_gpu)
    fake_cp = _fake_cp()
    fake_cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setattr(n4, "cp", fake_cp)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=fused_dir / "corrected.zarr",
        apply_correction=True,
        overwrite=True,
        debug=True,
        use_unsharp_mask=False,
    )

    result = n4.compute_fields_from_workspace(config)[0]
    assert result.corrected_path is not None

    corrected = zarr.open_array(result.corrected_path, mode="r")
    assert corrected.shape == (2, 8, 8, 1)
    fused = zarr.open_array(fused_dir / "fused.zarr", mode="r")
    corrected_float = [np.asarray(fused[idx, :, :, 0], dtype=np.float32) / field for idx in range(2)]

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

    float_store = zarr.open_array(
        result.corrected_path.with_name(
            f"{result.corrected_path.stem}_float32{result.corrected_path.suffix}"
        ),
        mode="r",
    )
    np.testing.assert_allclose(
        np.asarray(float_store).squeeze(axis=-1),
        corrected_float_arr,
        rtol=1e-4,
        atol=5e-2,
    )


def test_normalize_field_sets_background_to_one() -> None:
    # Foreground: a 2x2 block with positive values; background: zeros
    image = np.zeros((6, 6), dtype=np.float32)
    image[2:4, 2:4] = 10.0

    # Field: vary across image; background deliberately set far from 1.0
    yy, xx = np.mgrid[0:6, 0:6].astype(np.float32)
    field = 0.5 + 0.1 * (yy + xx)  # ranges from 0.5 .. 1.6

    # Normalize so median ≈ 1 in foreground and exactly 1 outside
    norm, scale, fg_mask = n4._normalize_field_to_unity(field, image, threshold=None)

    assert norm.shape == field.shape
    assert scale > 0
    assert np.any(fg_mask)

    # Background must be exactly 1.0
    bg = ~fg_mask
    if np.any(bg):
        assert np.allclose(norm[bg], 1.0)

    # Foreground median ~ 1 (robust within a small tolerance)
    fg_vals = norm[fg_mask]
    med = float(np.median(fg_vals)) if fg_vals.size else 1.0
    assert abs(med - 1.0) < 1e-3


def test_quantization_dynamic_guard_spans_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthetic plane with broad dynamic range and a few bright spikes
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    base = rng.uniform(10.0, 200.0, size=(1, 128, 128, 1)).astype(np.float32)
    # Insert bright spikes (~0.05%)
    spike_coords = [(5, 5), (64, 64), (100, 100)]
    for y, x in spike_coords:
        base[0, y, x, 0] = 2000.0
    _create_fused_store(fused_dir, base.astype(np.uint16))

    # Identity field; no sharpening during correction
    monkeypatch.setattr(
        n4, "compute_correction_field", lambda *args, **kwargs: np.ones((128, 128), dtype=np.float32)
    )
    monkeypatch.setattr(n4, "_prepare_gpu_field", lambda arr: np.asarray(arr, dtype=np.float32))
    monkeypatch.setattr(
        n4,
        "_correct_plane_gpu",
        lambda plane, field_gpu, use_unsharp_mask, mask_cpu=None: np.asarray(plane, dtype=np.float32)
        / field_gpu,
    )
    fake_cp = _fake_cp()
    fake_cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setattr(n4, "cp", fake_cp)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=fused_dir / "corrected.zarr",
        apply_correction=True,
        overwrite=True,
        debug=False,
        use_unsharp_mask=False,
    )
    result = n4.compute_fields_from_workspace(config)[0]
    corrected = zarr.open_array(result.corrected_path, mode="r")
    u16 = np.asarray(corrected, dtype=np.uint16).squeeze(axis=-1)
    assert u16.max() <= np.iinfo(np.uint16).max
    # Expect near-full usage of dynamic range due to adaptive guard
    assert u16.max() >= 60000


def test_quantization_with_unsharp_spans_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    fused_dir = workspace / "analysis/deconv/stitch--roi+cb"
    fused_dir.mkdir(parents=True)

    # Base image with moderate values; unsharp will boost edges
    img = np.zeros((1, 64, 64, 1), dtype=np.float32)
    img[0, 16:48, 16:48, 0] = 100.0
    _create_fused_store(fused_dir, img.astype(np.uint16))

    monkeypatch.setattr(
        n4, "compute_correction_field", lambda *args, **kwargs: np.ones((64, 64), dtype=np.float32)
    )
    monkeypatch.setattr(n4, "_prepare_gpu_field", lambda arr: np.asarray(arr, dtype=np.float32))

    # Simulate unsharp by inflating intensities inside the mask by 1.5x
    def _fake_correct(
        plane: np.ndarray,
        *,
        field_gpu: np.ndarray,
        use_unsharp_mask: bool,
        mask_cpu: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.asarray(plane, dtype=np.float32) / np.asarray(field_gpu, dtype=np.float32)
        if use_unsharp_mask and mask_cpu is not None:
            out = np.where(mask_cpu, out * 1.5, out)
        return out

    monkeypatch.setattr(n4, "_correct_plane_gpu", _fake_correct)
    fake_cp = _fake_cp()
    fake_cp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1))
    monkeypatch.setattr(n4, "cp", fake_cp)

    config = n4.N4RuntimeConfig(
        workspace=workspace,
        roi="roi",
        codebook="cb",
        channel=0,
        shrink=1,
        spline_lowres_px=8.0,
        z_index=0,
        field_output=fused_dir / "field.tif",
        corrected_output=fused_dir / "corrected.zarr",
        apply_correction=True,
        overwrite=True,
        debug=False,
        use_unsharp_mask=True,
    )

    result = n4.compute_fields_from_workspace(config)[0]
    corrected = zarr.open_array(result.corrected_path, mode="r")
    u16 = np.asarray(corrected, dtype=np.uint16).squeeze(axis=-1)
    # No overflow and near-full usage of the range
    assert u16.max() <= np.iinfo(np.uint16).max
    assert u16.max() >= 60000
