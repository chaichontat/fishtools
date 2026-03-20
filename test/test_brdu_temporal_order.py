from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

import fishtools.brdu.ot as brdu_ot
from fishtools.brdu.rendering import (
    barycentric_density_image,
    density_image,
    heatmap_intensity,
    nearest_display_positions,
    subsample_indices,
)
from fishtools.brdu.temporal_order import assign_temporal_order_from_brdu_edu
from fishtools.brdu.sampling import proportional_sample_sizes
from fishtools.brdu.transport_cost import (
    endpoint_tricycle_unary_penalties,
    pairwise_sqeuclidean_with_backward_tricycle_penalty,
    wrapped_tricycle_delta,
)
from fishtools.brdu.vector_field import (
    kernel_mode_endpoint_field,
    kernel_regressed_vector_field,
    support_adaptive_quiver_mask,
)
from fishtools.brdu.ot import TemporalProblemConfig, fit_temporal_problem, make_brdu_pair_cost_builder


def test_assign_temporal_order_from_brdu_edu_uses_edu_to_brdu_sequence() -> None:
    brdu = np.array([False, True, True, False], dtype=bool)
    edu = np.array([True, True, False, False], dtype=bool)

    result = assign_temporal_order_from_brdu_edu(brdu_pos=brdu, edu_pos=edu)

    np.testing.assert_array_equal(result, np.array([0, 1, 2, -1], dtype=int))


def test_pairwise_cost_penalizes_backward_tricycle_moves_more_than_forward_moves() -> None:
    src_features = np.zeros((1, 2), dtype=np.float32)
    tgt_features = np.zeros((2, 2), dtype=np.float32)
    src_theta = np.array([1.0], dtype=np.float32)
    tgt_theta = np.array([1.5, 0.5], dtype=np.float32)

    cost = pairwise_sqeuclidean_with_backward_tricycle_penalty(
        src_features=src_features,
        tgt_features=tgt_features,
        src_tricycle=src_theta,
        tgt_tricycle=tgt_theta,
        backward_penalty_weight=100.0,
    )

    assert cost.shape == (1, 2)
    assert cost[0, 0] < cost[0, 1]


def test_pairwise_cost_penalizes_large_ap_ml_displacements() -> None:
    src_features = np.zeros((1, 2), dtype=np.float32)
    tgt_features = np.zeros((2, 2), dtype=np.float32)
    src_theta = np.array([1.0], dtype=np.float32)
    tgt_theta = np.array([1.0, 1.0], dtype=np.float32)

    cost = pairwise_sqeuclidean_with_backward_tricycle_penalty(
        src_features=src_features,
        tgt_features=tgt_features,
        src_tricycle=src_theta,
        tgt_tricycle=tgt_theta,
        backward_penalty_weight=0.0,
        src_ap=np.array([0.0], dtype=np.float32),
        tgt_ap=np.array([0.0, 1000.0], dtype=np.float32),
        src_ml=np.array([0.0], dtype=np.float32),
        tgt_ml=np.array([0.0, 1000.0], dtype=np.float32),
        ap_ml_penalty_weight=10.0,
    )

    assert cost.shape == (1, 2)
    assert cost[0, 0] < cost[0, 1]


def test_pairwise_cost_rejects_nonfinite_ap_ml_inputs() -> None:
    src_features = np.zeros((1, 2), dtype=np.float32)
    tgt_features = np.zeros((1, 2), dtype=np.float32)
    src_theta = np.array([1.0], dtype=np.float32)
    tgt_theta = np.array([1.0], dtype=np.float32)

    try:
        pairwise_sqeuclidean_with_backward_tricycle_penalty(
            src_features=src_features,
            tgt_features=tgt_features,
            src_tricycle=src_theta,
            tgt_tricycle=tgt_theta,
            backward_penalty_weight=0.0,
            src_ap=np.array([np.nan], dtype=np.float32),
            tgt_ap=np.array([0.0], dtype=np.float32),
            src_ml=np.array([0.0], dtype=np.float32),
            tgt_ml=np.array([0.0], dtype=np.float32),
            ap_ml_penalty_weight=10.0,
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite AP/ML inputs to raise ValueError.")


def test_proportional_sample_sizes_preserves_ratios_with_exact_total() -> None:
    counts = np.array([17291, 51267, 31551], dtype=int)

    result = proportional_sample_sizes(counts=counts, total=20_000)

    assert int(result.sum()) == 20_000
    assert np.all(result <= counts)
    expected = counts.astype(float) * (20_000.0 / float(counts.sum()))
    assert float(np.max(np.abs(result - expected))) < 1.0


def test_wrapped_tricycle_delta_treats_wraparound_as_forward_motion() -> None:
    src_theta = np.array([2.0 * np.pi - 0.1], dtype=np.float32)
    tgt_theta = np.array([0.2], dtype=np.float32)

    delta = wrapped_tricycle_delta(source_theta=src_theta, target_theta=tgt_theta)

    np.testing.assert_allclose(delta, np.array([[0.3]], dtype=np.float32), atol=1.0e-5)


def test_endpoint_tricycle_unary_penalties_favor_early_cells_for_pre_and_late_cells_for_post() -> None:
    q00 = np.array([2.8, 3.0, 3.8, 4.0], dtype=np.float32)
    q01 = np.array([2.9, 3.0, 3.1], dtype=np.float32)
    q10 = np.array([3.7, 3.8, 3.9], dtype=np.float32)

    pre_penalty, post_penalty = endpoint_tricycle_unary_penalties(
        q00_tricycle=q00,
        q01_tricycle=q01,
        q10_tricycle=q10,
    )

    assert pre_penalty[0] < pre_penalty[-1]
    assert post_penalty[-1] < post_penalty[0]


def test_barycentric_density_bins_interpolated_positions_instead_of_blending_endpoints() -> None:
    rvz_edges = np.array([0.0, 0.5, 1.5], dtype=np.float32)
    src_theta = np.array([0.1], dtype=np.float32)
    src_rvz = np.array([0.1], dtype=np.float32)
    mapped_theta = np.array([0.1], dtype=np.float32)
    mapped_rvz = np.array([1.1], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)

    midpoint = barycentric_density_image(
        src_theta=src_theta,
        src_rvz=src_rvz,
        mapped_theta=mapped_theta,
        mapped_rvz=mapped_rvz,
        alpha=0.5,
        rvz_edges=rvz_edges,
        theta_bins=1,
        weights=weights,
    )
    endpoint_blend = 0.5 * density_image(src_theta, src_rvz, rvz_edges=rvz_edges, theta_bins=1, weights=weights) + 0.5 * density_image(
        mapped_theta,
        mapped_rvz,
        rvz_edges=rvz_edges,
        theta_bins=1,
        weights=weights,
    )

    np.testing.assert_array_equal(midpoint, np.array([[0.0], [1.0]], dtype=np.float64))
    np.testing.assert_array_equal(endpoint_blend, np.array([[0.5], [0.5]], dtype=np.float64))


def test_subsample_indices_caps_to_target_size_deterministically() -> None:
    first = subsample_indices(n_obs=10, target_n=4, random_seed=0)
    second = subsample_indices(n_obs=10, target_n=4, random_seed=0)

    assert len(first) == 4
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:-1] <= first[1:])


def test_density_image_can_bin_display_space_without_wrapping_theta() -> None:
    wrapped = density_image(
        np.array([2.0 * np.pi + 0.1], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        rvz_edges=np.array([0.0, 0.5], dtype=np.float32),
        theta_bins=1,
        wrap_theta=True,
    )
    display_space = density_image(
        np.array([2.0 * np.pi + 0.1], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        rvz_edges=np.array([0.0, 0.5], dtype=np.float32),
        theta_bins=1,
        wrap_theta=False,
    )

    np.testing.assert_array_equal(wrapped, np.array([[1.0]], dtype=np.float64))
    np.testing.assert_array_equal(display_space, np.array([[0.0]], dtype=np.float64))


def test_nearest_display_positions_matches_across_tricycle_wrap() -> None:
    idx, theta, rvz = nearest_display_positions(
        query_theta=np.array([2.0 * np.pi + 0.05], dtype=np.float32),
        query_rvz=np.array([1.0], dtype=np.float32),
        target_theta=np.array([0.05, 1.5], dtype=np.float32),
        target_rvz=np.array([1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(idx, np.array([0], dtype=int))
    np.testing.assert_allclose(theta, np.array([2.0 * np.pi + 0.05], dtype=np.float32))
    np.testing.assert_allclose(rvz, np.array([1.0], dtype=np.float32))


def test_heatmap_intensity_can_stay_linear() -> None:
    image = np.array([[0.0, 1.0], [9.0, 99.0]], dtype=np.float32)

    np.testing.assert_array_equal(heatmap_intensity(image, log_scale=False), image)
    np.testing.assert_allclose(heatmap_intensity(image, log_scale=True), np.log1p(image))


def test_kernel_regressed_vector_field_preserves_constant_displacements() -> None:
    field_dtheta, field_drvz, field_weight = kernel_regressed_vector_field(
        src_theta=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        src_rvz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        dtheta=np.array([0.05, 0.05, 0.05], dtype=np.float32),
        drvz=np.array([0.02, 0.02, 0.02], dtype=np.float32),
        theta_grid=np.array([0.2], dtype=np.float32),
        rvz_grid=np.array([0.2], dtype=np.float32),
        theta_bandwidth=0.5,
        rvz_bandwidth=0.5,
    )

    np.testing.assert_allclose(field_dtheta, np.array([[0.05]], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(field_drvz, np.array([[0.02]], dtype=np.float32), atol=1e-5)
    assert float(field_weight[0, 0]) > 0.0


def test_kernel_mode_endpoint_field_preserves_constant_endpoints() -> None:
    field_theta, field_rvz, field_weight = kernel_mode_endpoint_field(
        src_theta=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        src_rvz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        end_theta=np.array([0.4, 0.4, 0.4], dtype=np.float32),
        end_rvz=np.array([0.05, 0.05, 0.05], dtype=np.float32),
        theta_grid=np.array([0.2], dtype=np.float32),
        rvz_grid=np.array([0.2], dtype=np.float32),
        theta_bandwidth=0.5,
        rvz_bandwidth=0.5,
        endpoint_theta_bandwidth=0.5,
        endpoint_rvz_bandwidth=0.5,
    )

    np.testing.assert_allclose(field_theta, np.array([[0.4]], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(field_rvz, np.array([[0.05]], dtype=np.float32), atol=1e-5)
    assert float(field_weight[0, 0]) > 0.0


def test_support_adaptive_quiver_mask_keeps_more_high_support_arrows() -> None:
    mask = support_adaptive_quiver_mask(
        support=np.array([[0.0, 0.2, 0.5, 1.0]], dtype=np.float32),
        min_quantile=0.0,
        random_seed=0,
    )

    assert not bool(mask[0, 0])
    assert bool(mask[0, -1])


def test_make_brdu_pair_cost_builder_returns_pairwise_cost_dataframe() -> None:
    adata_src = AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "tricycle": [1.0],
                "ap": [0.0],
                "ml": [0.0],
            },
            index=["src0"],
        ),
        obsm={"X_pca": np.zeros((1, 2), dtype=np.float32)},
    )
    adata_tgt = AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "tricycle": [1.5, 0.5],
                "ap": [0.0, 0.0],
                "ml": [0.0, 0.0],
            },
            index=["tgt0", "tgt1"],
        ),
        obsm={"X_pca": np.zeros((2, 2), dtype=np.float32)},
    )
    subproblem = type("DummySubproblem", (), {"adata_src": adata_src, "adata_tgt": adata_tgt})()

    cost = make_brdu_pair_cost_builder(ap_ml_penalty=0.0, backward_tricycle_penalty=100.0)(subproblem)

    assert list(cost.index) == ["src0"]
    assert list(cost.columns) == ["tgt0", "tgt1"]
    assert cost.shape == (1, 2)
    assert float(cost.iloc[0, 0]) < float(cost.iloc[0, 1])


def test_fit_temporal_problem_applies_custom_pair_cost_builder() -> None:
    calls: dict[str, object] = {}

    class FakeSubproblem:
        def __init__(self, adata_src: AnnData, adata_tgt: AnnData) -> None:
            self.adata_src = adata_src
            self.adata_tgt = adata_tgt
            self.set_xy_calls: list[tuple[pd.DataFrame, str]] = []

        def set_xy(self, xy: pd.DataFrame, tag: str) -> None:
            self.set_xy_calls.append((xy, tag))

    class FakeTemporalProblem:
        def __init__(self, adata: AnnData) -> None:
            self.adata = adata
            self.problems = [(0, 1)]
            src = adata[adata.obs["time"] == 0].copy()
            tgt = adata[adata.obs["time"] == 1].copy()
            self._subproblem = FakeSubproblem(src, tgt)

        def score_genes_for_marginals(self, *, gene_set_proliferation: str, gene_set_apoptosis: str) -> "FakeTemporalProblem":
            calls["marginals"] = (gene_set_proliferation, gene_set_apoptosis)
            return self

        def prepare(self, **kwargs: object) -> "FakeTemporalProblem":
            calls["prepare"] = kwargs
            return self

        def __getitem__(self, key: tuple[int, int]) -> FakeSubproblem:
            assert key == (0, 1)
            return self._subproblem

        def solve(self, **kwargs: object) -> "FakeTemporalProblem":
            calls["solve"] = kwargs
            return self

    adata = AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"time": [0, 0, 1, 1]}, index=["c0", "c1", "c2", "c3"]),
        obsm={"X_scvi": np.zeros((4, 2), dtype=np.float32)},
    )
    cfg = TemporalProblemConfig(time_key="time", joint_attr="X_scvi", estimate_marginals=False, max_iterations=123)

    original_temporal_problem = brdu_ot.TemporalProblem
    brdu_ot.TemporalProblem = FakeTemporalProblem
    try:
        tp, times = fit_temporal_problem(
            adata,
            cfg,
            pair_cost_builder=lambda sub: np.zeros((sub.adata_src.n_obs, sub.adata_tgt.n_obs), dtype=np.float32),
        )
    finally:
        brdu_ot.TemporalProblem = original_temporal_problem

    assert isinstance(tp, FakeTemporalProblem)
    assert times == [0, 1]
    assert calls["prepare"] == {
        "time_key": "time",
        "joint_attr": "X_scvi",
        "policy": "sequential",
        "cost": "sq_euclidean",
        "a": False,
        "b": False,
        "marginal_kwargs": {},
    }
    assert calls["solve"] == {
        "epsilon": 1e-3,
        "tau_a": 0.95,
        "tau_b": 0.95,
        "rank": -1,
        "scale_cost": "mean",
        "batch_size": None,
        "threshold": 1e-3,
        "max_iterations": 123,
    }
    assert len(tp._subproblem.set_xy_calls) == 1
    xy, tag = tp._subproblem.set_xy_calls[0]
    assert tag == "cost_matrix"
    assert list(xy.index) == ["c0", "c1"]
    assert list(xy.columns) == ["c2", "c3"]
