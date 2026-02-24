"""Extra tests for puzzle_reconstruction/utils/metrics.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.metrics import (
    ReconstructionMetrics,
    placement_iou,
    order_kendall_tau,
    permutation_distance,
    assembly_precision_recall,
    fragment_placement_accuracy,
    compute_reconstruction_metrics,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _metrics(**kwargs) -> ReconstructionMetrics:
    defaults = dict(precision=0.8, recall=0.7, f1=0.75,
                    placement_accuracy=0.9, order_accuracy=0.85,
                    permutation_dist=0.1, n_correct=9, n_total=10)
    defaults.update(kwargs)
    return ReconstructionMetrics(**defaults)


# ─── ReconstructionMetrics ────────────────────────────────────────────────────

class TestReconstructionMetricsExtra:
    def test_stores_precision_recall_f1(self):
        m = _metrics(precision=0.6, recall=0.8, f1=0.686)
        assert m.precision == pytest.approx(0.6)
        assert m.recall == pytest.approx(0.8)

    def test_repr_contains_f1(self):
        m = _metrics()
        assert "f1" in repr(m).lower()

    def test_extra_dict_default_empty(self):
        m = _metrics()
        assert m.extra == {}

    def test_n_correct_n_total(self):
        m = _metrics(n_correct=7, n_total=10)
        assert m.n_correct == 7 and m.n_total == 10


# ─── placement_iou ────────────────────────────────────────────────────────────

class TestPlacementIouExtra:
    def test_identical_boxes_is_one(self):
        assert placement_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        assert placement_iou((0, 0, 5, 5), (10, 10, 5, 5)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = placement_iou((0, 0, 10, 10), (5, 0, 10, 10))
        assert 0.0 < score < 1.0

    def test_zero_area_box(self):
        score = placement_iou((0, 0, 0, 0), (0, 0, 5, 5))
        assert score == pytest.approx(0.0)

    def test_in_range(self):
        s = placement_iou((1, 1, 8, 8), (3, 3, 8, 8))
        assert 0.0 <= s <= 1.0


# ─── order_kendall_tau ────────────────────────────────────────────────────────

class TestOrderKendallTauExtra:
    def test_identical_is_one(self):
        assert order_kendall_tau([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_reversed_is_zero(self):
        tau = order_kendall_tau([2, 1, 0], [0, 1, 2])
        assert tau == pytest.approx(0.0)

    def test_empty_is_one(self):
        assert order_kendall_tau([], []) == pytest.approx(1.0)

    def test_single_element_is_one(self):
        assert order_kendall_tau([5], [5]) == pytest.approx(1.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            order_kendall_tau([0, 1], [0, 1, 2])

    def test_in_range(self):
        tau = order_kendall_tau([0, 2, 1], [0, 1, 2])
        assert 0.0 <= tau <= 1.0


# ─── permutation_distance ─────────────────────────────────────────────────────

class TestPermutationDistanceExtra:
    def test_identical_is_zero(self):
        assert permutation_distance([0, 1, 2], [0, 1, 2]) == pytest.approx(0.0)

    def test_all_different_is_one(self):
        assert permutation_distance([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert permutation_distance([], []) == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            permutation_distance([0], [0, 1])

    def test_partial_mismatch(self):
        d = permutation_distance([0, 1, 2, 3], [0, 2, 1, 3])
        assert d == pytest.approx(0.5)


# ─── assembly_precision_recall ────────────────────────────────────────────────

class TestAssemblyPrecisionRecallExtra:
    def test_empty_both_is_one_one(self):
        p, r = assembly_precision_recall([], [])
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_perfect_match(self):
        pairs = [(0, 1), (1, 2)]
        p, r = assembly_precision_recall(pairs, pairs)
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_reversed_pair_counts(self):
        p, r = assembly_precision_recall([(1, 0)], [(0, 1)])
        assert p == pytest.approx(1.0) and r == pytest.approx(1.0)

    def test_no_match(self):
        p, r = assembly_precision_recall([(0, 1)], [(2, 3)])
        assert p == pytest.approx(0.0) and r == pytest.approx(0.0)

    def test_partial_recall(self):
        _, r = assembly_precision_recall([(0, 1)], [(0, 1), (1, 2)])
        assert r == pytest.approx(0.5)


# ─── fragment_placement_accuracy ──────────────────────────────────────────────

class TestFragmentPlacementAccuracyExtra:
    def test_empty_true_positions_is_one(self):
        assert fragment_placement_accuracy({}, {}) == pytest.approx(1.0)

    def test_exact_match(self):
        pred = {0: (10.0, 20.0), 1: (30.0, 40.0)}
        true = {0: (10.0, 20.0), 1: (30.0, 40.0)}
        assert fragment_placement_accuracy(pred, true) == pytest.approx(1.0)

    def test_all_wrong(self):
        pred = {0: (100.0, 100.0)}
        true = {0: (0.0, 0.0)}
        acc = fragment_placement_accuracy(pred, true, tolerance=10.0)
        assert acc == pytest.approx(0.0)

    def test_missing_fragment_not_counted(self):
        pred = {}
        true = {0: (0.0, 0.0)}
        acc = fragment_placement_accuracy(pred, true)
        assert acc == pytest.approx(0.0)


# ─── compute_reconstruction_metrics ──────────────────────────────────────────

class TestComputeReconstructionMetricsExtra:
    def test_returns_metrics(self):
        m = compute_reconstruction_metrics()
        assert isinstance(m, ReconstructionMetrics)

    def test_no_inputs_neutral_values(self):
        m = compute_reconstruction_metrics()
        assert m.precision == pytest.approx(0.0)
        assert m.order_accuracy == pytest.approx(0.5)

    def test_with_pairs(self):
        pairs = [(0, 1), (1, 2)]
        m = compute_reconstruction_metrics(pred_pairs=pairs, true_pairs=pairs)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_with_order(self):
        m = compute_reconstruction_metrics(pred_order=[0, 1, 2],
                                            true_order=[0, 1, 2])
        assert m.order_accuracy == pytest.approx(1.0)
        assert m.permutation_dist == pytest.approx(0.0)

    def test_with_positions(self):
        pos = {0: (0.0, 0.0), 1: (10.0, 10.0)}
        m = compute_reconstruction_metrics(pred_positions=pos,
                                            true_positions=pos)
        assert m.placement_accuracy == pytest.approx(1.0)
        assert m.n_total == 2
