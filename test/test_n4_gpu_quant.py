from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fishtools.preprocess import n4

pytestmark = pytest.mark.timeout(30)


def test_apply_correction_field_with_mocked_cupy_numpy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.array([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32)
    field = np.array([[2.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expected = np.array([[2.0, 4.0], [4.0, 4.0]], dtype=np.float32)

    calls: dict[str, int] = {"asarray": 0, "asnumpy": 0}

    class _DummyCuPy:
        float32 = np.float32
        ndarray = np.ndarray

        def __init__(self) -> None:
            self.cuda = type(
                "_Cuda", (), {"runtime": type("_Rt", (), {"getDeviceCount": staticmethod(lambda: 1)})}
            )

        def asarray(self, arr: np.ndarray, dtype: Any | None = None) -> np.ndarray:
            calls["asarray"] += 1
            return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

        def asnumpy(self, arr: np.ndarray) -> np.ndarray:
            calls["asnumpy"] += 1
            return np.asarray(arr)

    dummy_cp = _DummyCuPy()
    # Force the GPU branch without real CUDA
    n4._cupy_available.cache_clear()
    monkeypatch.setattr(n4, "cp", dummy_cp, raising=True)
    monkeypatch.setattr(n4, "_cupy_available", lambda: True, raising=True)

    corrected = n4.apply_correction_field(image, field)
    np.testing.assert_allclose(corrected, expected, rtol=1e-6, atol=1e-6)
    assert calls["asarray"] >= 2  # field + image transfers
    assert calls["asnumpy"] >= 1


def test_quantization_unit_mapping_roundtrip() -> None:
    # Build synthetic planes with a known dynamic range and a few outliers
    rng = np.random.default_rng(0)
    base = rng.normal(loc=100.0, scale=5.0, size=(64, 64)).astype(np.float32)
    base[0, 0] = 5.0  # low outlier
    base[1, 1] = 300.0  # high outlier

    def supplier():
        # Two planes with slight variation
        yield base
        yield (base * 1.02).astype(np.float32)

    params = n4._compute_quantization_params(supplier)
    assert np.isfinite(params.lower)
    assert np.isfinite(params.upper)
    assert params.upper > params.lower

    # Quantize and dequantize and ensure interior values round-trip reasonably
    q = n4._quantize_to_uint16(base, params)
    scale = np.iinfo(np.uint16).max / max(params.upper - params.lower, n4.QUANT_MIN_RANGE)
    recovered = q.astype(np.float32) / scale + params.lower

    inside = (q > 0) & (q < np.iinfo(np.uint16).max)
    if np.any(inside):
        err = np.abs(recovered - base)
        # Allow a small absolute error due to quantization granularity and added headroom
        assert float(np.nanmax(err[inside])) <= 3.0
