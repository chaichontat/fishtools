from __future__ import annotations

import runpy
from pathlib import Path

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


def test_plot_simplex_native_proj_script_loads() -> None:
    loaded = runpy.run_path(str(Path("scripts/gam/plot_simplex_native_proj.py")))
    assert "inv_alr" in loaded


def test_default_meta_path_prefers_ilr(tmp_path: Path) -> None:
    alr_meta = tmp_path / "simplex_meta.json"
    ilr_meta = tmp_path / "simplex_meta_ilr.json"
    alr_meta.write_text("{}", encoding="utf-8")
    ilr_meta.write_text("{}", encoding="utf-8")

    assert mod._default_meta_path(tmp_path) == ilr_meta


def test_default_meta_path_falls_back_to_alr(tmp_path: Path) -> None:
    alr_meta = tmp_path / "simplex_meta.json"
    alr_meta.write_text("{}", encoding="utf-8")

    assert mod._default_meta_path(tmp_path) == alr_meta


def test_compute_apml_percentile_bounds_applies_independent_ranges() -> None:
    ap = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)
    ml = np.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype=float)
    keep = np.array([True, True, True, True, False])

    ap_lo, ap_hi, ml_lo, ml_hi = mod._compute_apml_percentile_bounds(
        ap=ap,
        ml=ml,
        valid_mask=keep,
        percentile_range=(25.0, 75.0),
    )

    assert ap_lo == pytest.approx(7.5)
    assert ap_hi == pytest.approx(22.5)
    assert ml_lo == pytest.approx(107.5)
    assert ml_hi == pytest.approx(122.5)


def test_feather_box_alpha_softens_box_edges() -> None:
    alpha = mod._feather_box_alpha(
        ap=np.array([5.0, 7.5, 10.0, 22.5, 25.0], dtype=float),
        ml=np.array([115.0, 115.0, 115.0, 115.0, 115.0], dtype=float),
        bounds=(5.0, 25.0, 110.0, 120.0),
        feather_fraction=0.25,
    )

    assert alpha[0] == pytest.approx(0.0)
    assert 0.0 < alpha[1] < 1.0
    assert alpha[2] == pytest.approx(1.0)
    assert 0.0 < alpha[3] < 1.0
    assert alpha[4] == pytest.approx(0.0)


def test_transform_display_values_zscore_is_centered() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    out, vmin, vmax, label, _cmap = mod._transform_display_values(
        vals,
        mode="zscore",
        percentile_range=(1.0, 99.0),
        z_limit=2.0,
    )

    assert np.mean(out) == pytest.approx(0.0)
    assert np.std(out) == pytest.approx(1.0)
    assert (vmin, vmax) == (-2.0, 2.0)
    assert label == "per-topic z-score"


def test_transform_display_values_percentile_scales_to_unit_interval() -> None:
    vals = np.array([0.0, 1.0, 2.0, 100.0], dtype=float)
    out, vmin, vmax, label, _cmap = mod._transform_display_values(
        vals,
        mode="percentile",
        percentile_range=(25.0, 75.0),
        z_limit=2.5,
    )

    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert (vmin, vmax) == (0.0, 1.0)
    assert "25-75%" in label
