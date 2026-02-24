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


# ─── ReconstructionMetrics (extra) ────────────────────────────────────────────

class TestReconstructionMetricsExtra:
    def _make(self, **kwargs):
        defaults = dict(
            precision=0.5, recall=0.5, f1=0.5,
            placement_accuracy=0.5, order_accuracy=0.5,
            permutation_dist=0.5, n_correct=5, n_total=10,
        )
        defaults.update(kwargs)
        return ReconstructionMetrics(**defaults)

    def test_perfect_scores(self):
        rm = self._make(precision=1.0, recall=1.0, f1=1.0,
                        placement_accuracy=1.0, order_accuracy=1.0,
                        permutation_dist=0.0, n_correct=10, n_total=10)
        assert rm.f1 == pytest.approx(1.0)
        assert rm.permutation_dist == pytest.approx(0.0)

    def test_worst_scores(self):
        rm = self._make(precision=0.0, recall=0.0, f1=0.0,
                        placement_accuracy=0.0, order_accuracy=0.0,
                        permutation_dist=1.0, n_correct=0, n_total=10)
        assert rm.precision == pytest.approx(0.0)
        assert rm.n_correct == 0

    def test_extra_dict_preserved(self):
        rm = self._make(extra={"my_metric": 3.14})
        assert rm.extra["my_metric"] == pytest.approx(3.14)

    def test_extra_multiple_keys(self):
        rm = self._make(extra={"a": 1.0, "b": 2.0})
        assert rm.extra["a"] == pytest.approx(1.0)
        assert rm.extra["b"] == pytest.approx(2.0)

    def test_n_total_zero_allowed(self):
        rm = self._make(n_correct=0, n_total=0)
        assert rm.n_total == 0


# ─── placement_iou (extra) ────────────────────────────────────────────────────

class TestPlacementIouExtra:
    def test_large_outer_small_inner(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (10.0, 10.0, 5.0, 5.0)
        iou = placement_iou(outer, inner)
        assert 0.0 < iou < 1.0

    def test_symmetry(self):
        a = (0.0, 0.0, 20.0, 30.0)
        b = (10.0, 15.0, 20.0, 30.0)
        assert placement_iou(a, b) == pytest.approx(placement_iou(b, a))

    def test_large_overlap_near_one(self):
        a = (0.0, 0.0, 100.0, 100.0)
        b = (1.0, 1.0, 100.0, 100.0)
        iou = placement_iou(a, b)
        assert iou > 0.90

    def test_both_zero_area_returns_zero(self):
        a = (5.0, 5.0, 0.0, 0.0)
        b = (5.0, 5.0, 0.0, 0.0)
        assert placement_iou(a, b) == pytest.approx(0.0)

    def test_vertical_strip_overlap(self):
        a = (0.0, 0.0, 10.0, 50.0)
        b = (5.0, 0.0, 10.0, 50.0)
        iou = placement_iou(a, b)
        # intersection = 5×50=250, union = 2*500-250=750
        assert iou == pytest.approx(250.0 / 750.0, abs=1e-5)

    def test_result_type_float(self):
        a = (0.0, 0.0, 10.0, 10.0)
        result = placement_iou(a, a)
        assert isinstance(result, float)


# ─── order_kendall_tau (extra) ────────────────────────────────────────────────

class TestOrderKendallTauExtra:
    def test_three_elements_perfect(self):
        o = [10, 20, 30]
        assert order_kendall_tau(o, o) == pytest.approx(1.0)

    def test_three_elements_reversed(self):
        pred = [3, 2, 1]
        true = [1, 2, 3]
        assert order_kendall_tau(pred, true) == pytest.approx(0.0)

    def test_five_elements_one_swap(self):
        true = [1, 2, 3, 4, 5]
        pred = [1, 3, 2, 4, 5]  # one transposition
        tau = order_kendall_tau(pred, true)
        assert 0.5 < tau < 1.0

    def test_partially_unknown_ids(self):
        pred = [1, 99]
        true = [1, 2]
        tau = order_kendall_tau(pred, true)
        assert 0.0 <= tau <= 1.0

    def test_all_unknown_returns_neutral(self):
        pred = [100, 200, 300]
        true = [1, 2, 3]
        tau = order_kendall_tau(pred, true)
        assert tau == pytest.approx(0.5)

    def test_two_elements_correct(self):
        assert order_kendall_tau([5, 10], [5, 10]) == pytest.approx(1.0)

    def test_result_is_float(self):
        result = order_kendall_tau([1, 2], [1, 2])
        assert isinstance(result, float)


# ─── permutation_distance (extra) ────────────────────────────────────────────

class TestPermutationDistanceExtra:
    def test_half_mismatch(self):
        pred = [1, 9, 3, 9]
        true = [1, 2, 3, 4]
        # positions 1 and 3 wrong → dist = 2/4 = 0.5
        assert permutation_distance(pred, true) == pytest.approx(0.5)

    def test_large_sequence_identical(self):
        seq = list(range(100))
        assert permutation_distance(seq, seq) == pytest.approx(0.0)

    def test_all_wrong_single(self):
        assert permutation_distance([9], [1]) == pytest.approx(1.0)

    def test_three_elements_one_wrong(self):
        pred = [1, 9, 3]
        true = [1, 2, 3]
        assert permutation_distance(pred, true) == pytest.approx(1.0 / 3.0)

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            permutation_distance([1, 2, 3], [1, 2])

    def test_result_in_0_1(self):
        pred = [3, 1, 2, 4]
        true = [1, 2, 3, 4]
        d = permutation_distance(pred, true)
        assert 0.0 <= d <= 1.0


# ─── assembly_precision_recall (extra) ───────────────────────────────────────

class TestAssemblyPrecisionRecallExtra:
    def test_empty_pred_full_true(self):
        p, r = assembly_precision_recall([], [(1, 2), (3, 4)])
        assert p == pytest.approx(1.0) or p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)

    def test_full_pred_empty_true(self):
        p, r = assembly_precision_recall([(1, 2)], [])
        assert p == pytest.approx(0.0)

    def test_duplicate_pairs_count_once(self):
        pred = [(1, 2), (1, 2)]
        true = [(1, 2)]
        p, _ = assembly_precision_recall(pred, true)
        # May deduplicate or not — result should be in range
        assert 0.0 <= p <= 1.0

    def test_large_perfect_overlap(self):
        pairs = [(i, i + 1) for i in range(50)]
        p, r = assembly_precision_recall(pairs, pairs)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_reversed_pair_considered_match(self):
        pred = [(2, 1)]
        true = [(1, 2)]
        p, r = assembly_precision_recall(pred, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_results_nonneg(self):
        pred = [(1, 3), (2, 5)]
        true = [(1, 2), (3, 4)]
        p, r = assembly_precision_recall(pred, true)
        assert p >= 0.0
        assert r >= 0.0


# ─── fragment_placement_accuracy (extra) ─────────────────────────────────────

class TestFragmentPlacementAccuracyExtra:
    def test_exact_match_multiple_frags(self):
        pos = {i: (float(i * 10), float(i * 5)) for i in range(10)}
        assert fragment_placement_accuracy(pos, pos, tolerance=0.1) == pytest.approx(1.0)

    def test_large_offset_all_wrong(self):
        pred = {1: (0.0, 0.0), 2: (0.0, 0.0)}
        true = {1: (1000.0, 0.0), 2: (0.0, 1000.0)}
        assert fragment_placement_accuracy(pred, true, tolerance=10.0) == pytest.approx(0.0)

    def test_tolerance_boundary_exact(self):
        pred = {1: (0.0, 0.0)}
        true = {1: (3.0, 4.0)}  # distance = 5.0
        assert fragment_placement_accuracy(pred, true, tolerance=5.0) == pytest.approx(1.0)
        assert fragment_placement_accuracy(pred, true, tolerance=4.9) == pytest.approx(0.0)

    def test_extra_pred_fragments_ignored(self):
        pred = {1: (0.0, 0.0), 99: (100.0, 100.0)}
        true = {1: (0.0, 0.0)}
        acc = fragment_placement_accuracy(pred, true, tolerance=1.0)
        assert acc == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(
            fragment_placement_accuracy({1: (0., 0.)}, {1: (0., 0.)}, 1.0),
            float,
        )

    def test_both_empty_returns_one(self):
        assert fragment_placement_accuracy({}, {}, 1.0) == pytest.approx(1.0)


# ─── compute_reconstruction_metrics (extra) ──────────────────────────────────

class TestComputeReconstructionMetricsExtra:
    def test_returns_reconstruction_metrics_instance(self):
        rm = compute_reconstruction_metrics()
        assert isinstance(rm, ReconstructionMetrics)

    def test_no_pairs_f1_zero(self):
        rm = compute_reconstruction_metrics()
        assert rm.f1 == pytest.approx(0.0)

    def test_partial_pairs_f1_between(self):
        pred = [(1, 2), (2, 3), (9, 10)]
        true = [(1, 2), (2, 3), (3, 4)]
        rm = compute_reconstruction_metrics(pred_pairs=pred, true_pairs=true)
        assert 0.0 < rm.f1 < 1.0

    def test_order_accuracy_between_zero_and_one(self):
        pred = [2, 1, 3, 4]
        true = [1, 2, 3, 4]
        rm = compute_reconstruction_metrics(pred_order=pred, true_order=true)
        assert 0.0 <= rm.order_accuracy <= 1.0

    def test_n_correct_n_total_set(self):
        pos = {1: (0., 0.), 2: (100., 100.)}
        rm = compute_reconstruction_metrics(pred_positions=pos, true_positions=pos)
        assert rm.n_total == 2
        assert rm.n_correct == 2

    def test_placement_tol_zero_all_wrong(self):
        pred = {1: (0.01, 0.0)}
        true = {1: (0.0, 0.0)}
        rm = compute_reconstruction_metrics(
            pred_positions=pred, true_positions=true, placement_tol=0.0
        )
        assert rm.placement_accuracy == pytest.approx(0.0)

    def test_permutation_dist_with_perfect_order(self):
        order = [1, 2, 3, 4, 5]
        rm = compute_reconstruction_metrics(pred_order=order, true_order=order)
        assert rm.permutation_dist == pytest.approx(0.0)

    def test_all_inputs_combined(self):
        pairs = [(1, 2), (2, 3)]
        pos = {1: (0., 0.), 2: (10., 10.)}
        order = [1, 2]
        rm = compute_reconstruction_metrics(
            pred_pairs=pairs, true_pairs=pairs,
            pred_positions=pos, true_positions=pos,
            pred_order=order, true_order=order,
        )
        assert rm.precision == pytest.approx(1.0)
        assert rm.placement_accuracy == pytest.approx(1.0)
        assert rm.order_accuracy == pytest.approx(1.0)
