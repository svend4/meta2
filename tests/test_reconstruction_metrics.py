"""
Тесты для puzzle_reconstruction/utils/metrics.py

Покрывает:
    ReconstructionMetrics      — repr, поля, extra dict
    placement_iou              — identical→1, non-overlapping→0, partial,
                                 zero-area, fully inside
    order_kendall_tau          — perfect→1, reversed→0, n≤1→1, unknown IDs→0.5,
                                 ValueError на разные длины
    permutation_distance       — same→0, all-diff→1, partial, empty→0,
                                 ValueError на разные длины
    assembly_precision_recall  — perfect→(1,1), empty both→(1,1),
                                 no overlap→(0,0), symmetric pairs, partial
    fragment_placement_accuracy — all correct→1, none correct→0, partial,
                                  empty true→1, missing pred frag, tolerance
    compute_reconstruction_metrics — все None→нейтральные значения,
                                     пары, порядок, позиции, f1 формула
"""
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


# ─── ReconstructionMetrics ────────────────────────────────────────────────────

class TestReconstructionMetrics:
    def test_fields(self):
        rm = ReconstructionMetrics(
            precision=0.8, recall=0.7, f1=0.747,
            placement_accuracy=0.9, order_accuracy=0.85,
            permutation_dist=0.1, n_correct=9, n_total=10,
        )
        assert rm.precision          == pytest.approx(0.8)
        assert rm.recall             == pytest.approx(0.7)
        assert rm.f1                 == pytest.approx(0.747)
        assert rm.placement_accuracy == pytest.approx(0.9)
        assert rm.order_accuracy     == pytest.approx(0.85)
        assert rm.permutation_dist   == pytest.approx(0.1)
        assert rm.n_correct == 9
        assert rm.n_total   == 10

    def test_extra_default_empty(self):
        rm = ReconstructionMetrics(
            precision=0.0, recall=0.0, f1=0.0,
            placement_accuracy=0.0, order_accuracy=0.5,
            permutation_dist=0.0, n_correct=0, n_total=0,
        )
        assert rm.extra == {}

    def test_extra_custom(self):
        rm = ReconstructionMetrics(
            precision=1.0, recall=1.0, f1=1.0,
            placement_accuracy=1.0, order_accuracy=1.0,
            permutation_dist=0.0, n_correct=5, n_total=5,
            extra={"iou_mean": 0.9},
        )
        assert rm.extra["iou_mean"] == pytest.approx(0.9)

    def test_repr_fields(self):
        rm = ReconstructionMetrics(
            precision=0.8, recall=0.6, f1=0.686,
            placement_accuracy=0.7, order_accuracy=0.8,
            permutation_dist=0.2, n_correct=7, n_total=10,
        )
        r = repr(rm)
        assert "ReconstructionMetrics" in r
        assert "f1=" in r
        assert "placement=" in r
        assert "n=" in r


# ─── placement_iou ────────────────────────────────────────────────────────────

class TestPlacementIou:
    def test_identical_returns_one(self):
        box = (10.0, 10.0, 50.0, 30.0)
        assert placement_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_returns_zero(self):
        box1 = (0.0,  0.0,  10.0, 10.0)
        box2 = (20.0, 0.0,  10.0, 10.0)
        assert placement_iou(box1, box2) == pytest.approx(0.0)

    def test_adjacent_no_overlap(self):
        box1 = (0.0,  0.0, 10.0, 10.0)
        box2 = (10.0, 0.0, 10.0, 10.0)
        assert placement_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (5.0, 0.0, 10.0, 10.0)
        iou = placement_iou(box1, box2)
        assert iou == pytest.approx(50.0 / 150.0, abs=1e-5)

    def test_result_in_range(self):
        box1 = (0.0, 0.0, 20.0, 20.0)
        box2 = (5.0, 5.0, 20.0, 20.0)
        assert 0.0 <= placement_iou(box1, box2) <= 1.0

    def test_zero_area_returns_zero(self):
        box1 = (0.0, 0.0, 0.0, 10.0)
        box2 = (0.0, 0.0, 10.0, 10.0)
        assert placement_iou(box1, box2) == pytest.approx(0.0)

    def test_fully_inside(self):
        outer = (0.0, 0.0, 20.0, 20.0)
        inner = (5.0, 5.0, 5.0, 5.0)
        iou = placement_iou(outer, inner)
        assert iou == pytest.approx(25.0 / 400.0, abs=1e-5)

    def test_returns_float(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert isinstance(placement_iou(box, box), float)


# ─── order_kendall_tau ────────────────────────────────────────────────────────

class TestOrderKendallTau:
    def test_perfect_order_returns_one(self):
        order = [1, 2, 3, 4, 5]
        assert order_kendall_tau(order, order) == pytest.approx(1.0)

    def test_reversed_order_returns_zero(self):
        true = [1, 2, 3, 4]
        pred = [4, 3, 2, 1]
        assert order_kendall_tau(pred, true) == pytest.approx(0.0)

    def test_single_element_returns_one(self):
        assert order_kendall_tau([7], [7]) == pytest.approx(1.0)

    def test_empty_returns_one(self):
        assert order_kendall_tau([], []) == pytest.approx(1.0)

    def test_result_in_range(self):
        pred = [3, 1, 4, 2]
        true = [1, 2, 3, 4]
        tau  = order_kendall_tau(pred, true)
        assert 0.0 <= tau <= 1.0

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="Длины"):
            order_kendall_tau([1, 2], [1, 2, 3])

    def test_unknown_ids_returns_neutral(self):
        pred = [99, 100]
        true = [1, 2]
        tau  = order_kendall_tau(pred, true)
        assert tau == pytest.approx(0.5)

    def test_partially_correct_between_zero_and_one(self):
        true = [1, 2, 3, 4, 5]
        pred = [1, 2, 4, 3, 5]  # one swap
        tau  = order_kendall_tau(pred, true)
        assert 0.5 < tau < 1.0

    def test_two_elements_same_as_perfect(self):
        assert order_kendall_tau([1, 2], [1, 2]) == pytest.approx(1.0)

    def test_two_elements_swapped_returns_zero(self):
        assert order_kendall_tau([2, 1], [1, 2]) == pytest.approx(0.0)


# ─── permutation_distance ─────────────────────────────────────────────────────

class TestPermutationDistance:
    def test_identical_returns_zero(self):
        p = [1, 2, 3, 4]
        assert permutation_distance(p, p) == pytest.approx(0.0)

    def test_all_different_returns_one(self):
        pred = [1, 2, 3, 4]
        true = [5, 6, 7, 8]
        assert permutation_distance(pred, true) == pytest.approx(1.0)

    def test_partial_mismatch(self):
        pred = [1, 2, 9, 4]
        true = [1, 2, 3, 4]
        assert permutation_distance(pred, true) == pytest.approx(0.25)

    def test_empty_returns_zero(self):
        assert permutation_distance([], []) == pytest.approx(0.0)

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="Длины"):
            permutation_distance([1, 2], [1, 2, 3])

    def test_result_in_range(self):
        pred = [2, 1, 4, 3]
        true = [1, 2, 3, 4]
        d = permutation_distance(pred, true)
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        assert isinstance(permutation_distance([1], [1]), float)

    def test_single_match(self):
        assert permutation_distance([42], [42]) == pytest.approx(0.0)

    def test_single_mismatch(self):
        assert permutation_distance([1], [2]) == pytest.approx(1.0)


# ─── assembly_precision_recall ────────────────────────────────────────────────

class TestAssemblyPrecisionRecall:
    def test_perfect_returns_one_one(self):
        pairs = [(1, 2), (2, 3), (3, 4)]
        p, r  = assembly_precision_recall(pairs, pairs)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_empty_both_returns_one_one(self):
        p, r = assembly_precision_recall([], [])
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = [(1, 2), (3, 4)]
        true = [(5, 6), (7, 8)]
        p, r = assembly_precision_recall(pred, true)
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)

    def test_symmetric_pairs_match(self):
        pred = [(2, 1)]
        true = [(1, 2)]
        p, r = assembly_precision_recall(pred, true)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)

    def test_partial_overlap(self):
        pred = [(1, 2), (2, 3), (9, 10)]
        true = [(1, 2), (2, 3), (3, 4)]
        p, r = assembly_precision_recall(pred, true)
        assert p == pytest.approx(2.0 / 3.0, abs=1e-5)
        assert r == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_empty_pred_zero_recall(self):
        _, r = assembly_precision_recall([], [(1, 2)])
        assert r == pytest.approx(0.0)

    def test_empty_true_zero_precision(self):
        p, _ = assembly_precision_recall([(1, 2)], [])
        assert p == pytest.approx(0.0)

    def test_results_in_range(self):
        pred = [(1, 2), (3, 4)]
        true = [(1, 2), (2, 3)]
        p, r = assembly_precision_recall(pred, true)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0


# ─── fragment_placement_accuracy ──────────────────────────────────────────────

class TestFragmentPlacementAccuracy:
    def test_all_correct_returns_one(self):
        pos = {1: (10.0, 10.0), 2: (50.0, 50.0)}
        assert fragment_placement_accuracy(pos, pos, tolerance=5.0) == pytest.approx(1.0)

    def test_none_correct_returns_zero(self):
        pred = {1: (0.0, 0.0)}
        true = {1: (500.0, 500.0)}
        assert fragment_placement_accuracy(pred, true, tolerance=5.0) == pytest.approx(0.0)

    def test_partial_correct(self):
        pred = {1: (10.0, 10.0), 2: (500.0, 500.0)}
        true = {1: (10.0, 10.0), 2: (50.0,  50.0)}
        assert fragment_placement_accuracy(pred, true, tolerance=5.0) == pytest.approx(0.5)

    def test_empty_true_returns_one(self):
        assert fragment_placement_accuracy({1: (0., 0.)}, {}, 10.0) == pytest.approx(1.0)

    def test_missing_frag_in_pred(self):
        pred = {}
        true = {1: (10.0, 10.0), 2: (20.0, 20.0)}
        assert fragment_placement_accuracy(pred, true, tolerance=5.0) == pytest.approx(0.0)

    def test_tolerance_matters(self):
        pred = {1: (10.0, 10.0)}
        true = {1: (18.0, 10.0)}
        assert fragment_placement_accuracy(pred, true, tolerance=5.0)  == pytest.approx(0.0)
        assert fragment_placement_accuracy(pred, true, tolerance=10.0) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(
            fragment_placement_accuracy({1: (0., 0.)}, {1: (0., 0.)}, 1.0),
            float,
        )

    def test_border_tolerance(self):
        """Расстояние точно равно tolerance → правильно."""
        pred = {1: (0., 0.)}
        true = {1: (5., 0.)}
        assert fragment_placement_accuracy(pred, true, tolerance=5.0) == pytest.approx(1.0)


# ─── compute_reconstruction_metrics ──────────────────────────────────────────

class TestComputeReconstructionMetrics:
    def test_all_none_returns_metrics(self):
        rm = compute_reconstruction_metrics()
        assert isinstance(rm, ReconstructionMetrics)

    def test_all_none_neutral_values(self):
        rm = compute_reconstruction_metrics()
        assert rm.precision == pytest.approx(0.0)
        assert rm.recall    == pytest.approx(0.0)
        assert rm.f1        == pytest.approx(0.0)
        assert rm.order_accuracy == pytest.approx(0.5)

    def test_with_perfect_pairs(self):
        pairs = [(1, 2), (2, 3)]
        rm = compute_reconstruction_metrics(
            pred_pairs=pairs, true_pairs=pairs
        )
        assert rm.precision == pytest.approx(1.0)
        assert rm.recall    == pytest.approx(1.0)
        assert rm.f1        == pytest.approx(1.0)

    def test_with_perfect_order(self):
        order = [1, 2, 3, 4]
        rm = compute_reconstruction_metrics(
            pred_order=order, true_order=order
        )
        assert rm.order_accuracy   == pytest.approx(1.0)
        assert rm.permutation_dist == pytest.approx(0.0)

    def test_with_perfect_positions(self):
        pos = {1: (10., 10.), 2: (30., 30.)}
        rm = compute_reconstruction_metrics(
            pred_positions=pos, true_positions=pos
        )
        assert rm.placement_accuracy == pytest.approx(1.0)
        assert rm.n_correct == 2
        assert rm.n_total   == 2

    def test_f1_formula(self):
        pred = [(1, 2), (2, 3), (9, 9)]
        true = [(1, 2), (2, 3), (3, 4)]
        rm = compute_reconstruction_metrics(
            pred_pairs=pred, true_pairs=true
        )
        ep = 2.0 / 3.0
        er = 2.0 / 3.0
        assert rm.f1 == pytest.approx(2 * ep * er / (ep + er), abs=1e-4)

    def test_placement_tol_forwarded(self):
        pred = {1: (0., 0.)}
        true = {1: (8., 0.)}
        rm_tight = compute_reconstruction_metrics(
            pred_positions=pred, true_positions=true, placement_tol=5.0
        )
        rm_loose = compute_reconstruction_metrics(
            pred_positions=pred, true_positions=true, placement_tol=10.0
        )
        assert rm_tight.placement_accuracy == pytest.approx(0.0)
        assert rm_loose.placement_accuracy == pytest.approx(1.0)
