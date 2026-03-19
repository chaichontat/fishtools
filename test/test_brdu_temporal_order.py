from __future__ import annotations

import numpy as np

from fishtools.brdu.temporal_order import assign_temporal_order_from_brdu_edu
from fishtools.brdu.transport_cost import (
    pairwise_sqeuclidean_with_backward_tricycle_penalty,
    wrapped_tricycle_delta,
)


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


def test_wrapped_tricycle_delta_treats_wraparound_as_forward_motion() -> None:
    src_theta = np.array([2.0 * np.pi - 0.1], dtype=np.float32)
    tgt_theta = np.array([0.2], dtype=np.float32)

    delta = wrapped_tricycle_delta(source_theta=src_theta, target_theta=tgt_theta)

    np.testing.assert_allclose(delta, np.array([[0.3]], dtype=np.float32), atol=1.0e-5)
