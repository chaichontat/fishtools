from __future__ import annotations

import numpy as np
import pytest

from scripts.gam import plot_simplex_native_proj as mod


def test_inv_alr_outputs_simplex() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(100, 4)).astype(float)  # K=5
    u = mod.inv_alr(z, ref_index=2)
    assert u.shape == (100, 5)
    assert np.all(np.isfinite(u))
    assert np.all(u >= 0.0)
    assert np.all(u <= 1.0)
    assert np.allclose(np.sum(u, axis=1), 1.0, atol=1e-6)


def test_inv_alr_zero_logits_is_uniform() -> None:
    z = np.zeros((3, 2), dtype=float)  # K=3
    u = mod.inv_alr(z, ref_index=0)
    assert np.allclose(u, 1.0 / 3.0)


def test_inv_alr_ref_index_validation() -> None:
    z = np.zeros((1, 2), dtype=float)  # K=3
    with pytest.raises(ValueError):
        mod.inv_alr(z, ref_index=-1)
    with pytest.raises(ValueError):
        mod.inv_alr(z, ref_index=3)


def test_inv_alr_large_positive_coordinate_dominates() -> None:
    # K=3 with ref_index=0 => z columns correspond to topics 1 and 2.
    z = np.array([[10.0, -10.0]], dtype=float)
    u = mod.inv_alr(z, ref_index=0)
    assert u[0, 1] > 0.999
    assert u[0, 2] < 1e-6
