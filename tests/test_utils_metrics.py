"""Tests for puzzle_reconstruction.utils.metrics"""
import numpy as np
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


# ── placement_iou ─────────────────────────────────────────────────────────────

def test_placement_iou_identical_boxes():
    assert placement_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_placement_iou_no_overlap():
    assert placement_iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)


def test_placement_iou_partial_overlap():
    iou = placement_iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.0 < iou < 1.0


def test_placement_iou_range():
    iou = placement_iou((0, 0, 5, 5), (2, 2, 5, 5))
    assert 0.0 <= iou <= 1.0


def test_placement_iou_zero_area():
    # zero area box
    assert placement_iou((0, 0, 0, 0), (0, 0, 10, 10)) == pytest.approx(0.0)


# ── order_kendall_tau ─────────────────────────────────────────────────────────

def test_order_kendall_tau_perfect():
    order = [0, 1, 2, 3, 4]
    assert order_kendall_tau(order, order) == pytest.approx(1.0)


def test_order_kendall_tau_reversed():
    order = [0, 1, 2, 3, 4]
    rev = list(reversed(order))
    tau = order_kendall_tau(rev, order)
    # reversed order → tau = -1, normalised = 0
    assert tau == pytest.approx(0.0, abs=1e-9)


def test_order_kendall_tau_single_element():
    assert order_kendall_tau([5], [5]) == pytest.approx(1.0)


def test_order_kendall_tau_length_mismatch_raises():
    with pytest.raises(ValueError):
        order_kendall_tau([0, 1, 2], [0, 1])


def test_order_kendall_tau_range():
    import random
    random.seed(42)
    perm = list(range(10))
    random.shuffle(perm)
    tau = order_kendall_tau(perm, list(range(10)))
    assert 0.0 <= tau <= 1.0


def test_order_kendall_tau_unknown_ids():
    # IDs not in true_order → returns 0.5
    assert order_kendall_tau([99, 100], [0, 1]) == pytest.approx(0.5)


# ── permutation_distance ──────────────────────────────────────────────────────

def test_permutation_distance_identical():
    assert permutation_distance([0, 1, 2], [0, 1, 2]) == pytest.approx(0.0)


def test_permutation_distance_all_different():
    assert permutation_distance([1, 2, 0], [0, 1, 2]) == pytest.approx(1.0)


def test_permutation_distance_partial():
    d = permutation_distance([0, 2, 1], [0, 1, 2])
    assert abs(d - 2/3) < 1e-9


def test_permutation_distance_empty():
    assert permutation_distance([], []) == pytest.approx(0.0)


def test_permutation_distance_length_mismatch_raises():
    with pytest.raises(ValueError):
        permutation_distance([0, 1], [0, 1, 2])


def test_permutation_distance_range():
    d = permutation_distance([2, 0, 1], [0, 1, 2])
    assert 0.0 <= d <= 1.0


# ── assembly_precision_recall ─────────────────────────────────────────────────

def test_assembly_precision_recall_perfect():
    pairs = [(0, 1), (1, 2), (2, 3)]
    p, r = assembly_precision_recall(pairs, pairs)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_assembly_precision_recall_empty_both():
    p, r = assembly_precision_recall([], [])
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_assembly_precision_recall_no_overlap():
    pred = [(0, 1)]
    true = [(2, 3)]
    p, r = assembly_precision_recall(pred, true)
    assert p == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_assembly_precision_recall_symmetric_pairs():
    pred = [(1, 0)]  # reversed
    true = [(0, 1)]  # canonical
    p, r = assembly_precision_recall(pred, true)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_assembly_precision_recall_partial():
    pred = [(0, 1), (1, 2), (3, 4)]
    true = [(0, 1), (1, 2)]
    p, r = assembly_precision_recall(pred, true)
    assert p == pytest.approx(2/3)
    assert r == pytest.approx(1.0)


# ── fragment_placement_accuracy ───────────────────────────────────────────────

def test_fragment_placement_accuracy_perfect():
    pos = {0: (0.0, 0.0), 1: (10.0, 10.0)}
    acc = fragment_placement_accuracy(pos, pos, tolerance=5.0)
    assert acc == pytest.approx(1.0)


def test_fragment_placement_accuracy_none_correct():
    pred = {0: (100.0, 100.0)}
    true = {0: (0.0, 0.0)}
    acc = fragment_placement_accuracy(pred, true, tolerance=5.0)
    assert acc == pytest.approx(0.0)


def test_fragment_placement_accuracy_empty_true():
    acc = fragment_placement_accuracy({}, {})
    assert acc == pytest.approx(1.0)


def test_fragment_placement_accuracy_partial():
    pred = {0: (0.0, 0.0), 1: (999.0, 999.0)}
    true = {0: (0.0, 0.0), 1: (10.0, 10.0)}
    acc = fragment_placement_accuracy(pred, true, tolerance=5.0)
    assert acc == pytest.approx(0.5)


def test_fragment_placement_accuracy_missing_fragment():
    pred = {}  # missing fragment 0
    true = {0: (0.0, 0.0)}
    acc = fragment_placement_accuracy(pred, true, tolerance=5.0)
    assert acc == pytest.approx(0.0)


# ── compute_reconstruction_metrics ────────────────────────────────────────────

def test_compute_reconstruction_metrics_returns_dataclass():
    m = compute_reconstruction_metrics()
    assert isinstance(m, ReconstructionMetrics)


def test_compute_reconstruction_metrics_all_defaults():
    m = compute_reconstruction_metrics()
    assert m.order_accuracy == pytest.approx(0.5)
    assert m.precision == pytest.approx(0.0)
    assert m.recall == pytest.approx(0.0)


def test_compute_reconstruction_metrics_with_pairs():
    pred = [(0, 1), (1, 2)]
    true = [(0, 1), (1, 2)]
    m = compute_reconstruction_metrics(pred_pairs=pred, true_pairs=true)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_compute_reconstruction_metrics_f1_range():
    pred = [(0, 1)]
    true = [(0, 1), (1, 2)]
    m = compute_reconstruction_metrics(pred_pairs=pred, true_pairs=true)
    assert 0.0 <= m.f1 <= 1.0


def test_compute_reconstruction_metrics_placement():
    pred_pos = {0: (0.0, 0.0), 1: (10.0, 10.0)}
    true_pos = {0: (0.0, 0.0), 1: (10.0, 10.0)}
    m = compute_reconstruction_metrics(pred_positions=pred_pos,
                                        true_positions=true_pos,
                                        placement_tol=5.0)
    assert m.placement_accuracy == pytest.approx(1.0)


def test_compute_reconstruction_metrics_order():
    order = [0, 1, 2, 3]
    m = compute_reconstruction_metrics(pred_order=order, true_order=order)
    assert m.order_accuracy == pytest.approx(1.0)
    assert m.permutation_dist == pytest.approx(0.0)


def test_reconstruction_metrics_repr_contains_f1():
    m = ReconstructionMetrics(
        precision=0.8, recall=0.7, f1=0.747,
        placement_accuracy=0.9, order_accuracy=0.8,
        permutation_dist=0.1, n_correct=9, n_total=10,
    )
    r = repr(m)
    assert "f1" in r.lower()
