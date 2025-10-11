from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from PIL import Image

from fishtools.preprocess import n4


def test_normalize_to_uint8_by_percentile_handles_nans_and_clamps(tmp_path: Path) -> None:
    # Arrange: include NaN/Inf and a wide range
    arr = np.array(
        [
            [np.nan, 0.0, 1.0, 2.0],
            [3.0, 4.0, np.inf, 5.0],
        ],
        dtype=np.float32,
    )

    # Act
    out = n4._normalize_to_uint8_by_percentile(arr, 1.0, 99.0)

    # Assert
    assert out.dtype == np.uint8
    assert out.shape == arr.shape
    assert out.min() >= 0 and out.max() <= 255
    # Not all equal (real scaling happened despite NaNs/Infs)
    assert len(np.unique(out)) > 2


def test_write_png_normalized_resizes_long_edge(tmp_path: Path) -> None:
    # Arrange: create a tall image so resize triggers
    img = np.linspace(0, 1, 2048 * 512, dtype=np.float32).reshape(2048, 512)
    out_png = tmp_path / "field.png"

    # Act
    n4._write_png_normalized(img, out_png, long_edge_max=512)

    # Assert
    assert out_png.exists()
    with Image.open(out_png) as im:
        w, h = im.size
        assert max(w, h) == 512
        assert im.mode == "L"


def test_zarr_attrs_write_update_then_item_then_put(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange three attribute mocks to exercise all branches
    class A:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def update(self, m: dict[str, Any]) -> None:
            self.store.update(m)

    class B:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def __setitem__(self, k: str, v: Any) -> None:
            self.store[k] = v

    class C:
        def __init__(self) -> None:
            self.store: dict[str, Any] = {}

        def put(self, k: str, v: Any) -> None:
            self.store[k] = v

    mapping = {"a": 1, "b": 2}
    a, b, c = A(), B(), C()

    # Act
    n4._zarr_attrs_write(a, mapping)
    n4._zarr_attrs_write(b, mapping)
    n4._zarr_attrs_write(c, mapping)

    # Assert
    assert a.store == mapping
    assert b.store == mapping
    assert c.store == mapping


def test_compute_quantization_params_small_samples_uses_fallback_upper() -> None:
    # Arrange: tiny planes so total_samples < QUANT_MIN_TOTAL_SAMPLES_FOR_HIGH_PERCENTILE
    planes = [
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
    ]

    def supplier():
        for p in planes:
            yield p

    # Act
    params = n4._compute_quantization_params(supplier)

    # Assert
    assert params.sample_count == 8
    assert params.lower_percentile == pytest.approx(n4.QUANT_LOWER_PERCENTILE)
    assert params.upper_percentile == pytest.approx(n4.QUANT_FALLBACK_UPPER_PERCENTILE)
    assert params.upper > params.lower


def test_quantize_to_uint16_headroom_prevents_full_saturation() -> None:
    # Arrange: Map [0, 1] with headroom so value at exactly 1 does not hit 65535
    params = n4.QuantizationParams(
        lower=0.0,
        upper=1.0,
        observed_min=0.0,
        observed_max=1.0,
        lower_percentile=1.0,
        upper_percentile=99.0,
        sample_count=100,
    )
    data = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    # Act
    q = n4._quantize_to_uint16(data, params)

    # Assert
    assert q.dtype == np.uint16
    assert q[0] == 0
    assert q[-1] < np.iinfo(np.uint16).max  # headroom keeps top value under max


def test_read_channel_names_from_zarr_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: stub zarr.open_array to avoid real IO/event loop issues
    class DummyArr:
        def __init__(self) -> None:
            self.attrs: dict[str, Any] = {}

    dummy = DummyArr()

    def fake_open_array(store: Path, mode: str = "r") -> DummyArr:  # type: ignore[override]
        return dummy

    monkeypatch.setattr(n4.zarr, "open_array", fake_open_array)

    for key_name, value in (
        ("key", ["a", "b"]),
        ("channel_names", ["c", "d"]),
        ("channels", ["e", "f"]),
        ("labels", ["g", "h"]),
    ):
        dummy.attrs.clear()
        dummy.attrs[key_name] = value
        names = n4._read_channel_names_from_zarr(Path("ignored"))
        assert names == [str(v) for v in value]


def test_slugify_sanitizes_names() -> None:
    assert n4._slugify("  Poly A  ") == "poly-a"
    assert n4._slugify("A--B??C") == "a-b-c"
    assert n4._slugify("___") == "unnamed"


def test_ensure_float32_contiguous() -> None:
    a = np.asfortranarray(np.ones((4, 3), dtype=np.float64))
    b = n4._ensure_float32(a)
    assert b.dtype == np.float32
    # Either C or F contiguous is acceptable when casting; ensure one is true
    assert b.flags.c_contiguous or b.flags.f_contiguous
