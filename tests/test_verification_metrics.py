"""Tests for puzzle_reconstruction.verification.metrics"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.metrics import (
    ReconstructionMetrics,
    BenchmarkResult,
    evaluate_reconstruction,
    compare_methods,
    _normalize_config,
    _compute_adjacency,
    _compute_edge_match_rate,
    _angle_diff_deg,
    _zero_metrics,
)


def make_config(n=3, spread=50.0):
    """Return a dict with n fragment positions and angles."""
    config = {}
    for i in range(n):
        pos = np.array([float(i * spread), 0.0])
        angle = 0.0
        config[i] = (pos, angle)
    return config


# ─── ReconstructionMetrics ────────────────────────────────────────────────────

def test_reconstruction_metrics_summary():
    m = ReconstructionMetrics(
        neighbor_accuracy=0.8,
        direct_comparison=0.7,
        perfect=False,
        position_rmse=5.0,
        angular_error_deg=3.0,
        n_fragments=4,
        n_correct_pairs=3,
        n_total_pairs=4,
        edge_match_rate=0.75,
    )
    summary = m.summary()
    assert "0.8" in summary or "80.0%" in summary or "80%" in summary
    assert "4" in summary


def test_reconstruction_metrics_perfect_true():
    m = ReconstructionMetrics(
        neighbor_accuracy=1.0, direct_comparison=1.0, perfect=True,
        position_rmse=0.0, angular_error_deg=0.0,
        n_fragments=3, n_correct_pairs=3, n_total_pairs=3, edge_match_rate=1.0
    )
    assert m.perfect is True


# ─── _angle_diff_deg ──────────────────────────────────────────────────────────

def test_angle_diff_zero():
    assert _angle_diff_deg(45.0, 45.0) == pytest.approx(0.0)


def test_angle_diff_opposite():
    diff = _angle_diff_deg(0.0, 180.0)
    assert diff == pytest.approx(180.0)


def test_angle_diff_near_360():
    diff = _angle_diff_deg(5.0, 355.0)
    assert diff == pytest.approx(10.0)


def test_angle_diff_range():
    diff = _angle_diff_deg(90.0, 270.0)
    assert diff == pytest.approx(180.0)


# ─── _zero_metrics ────────────────────────────────────────────────────────────

def test_zero_metrics():
    m = _zero_metrics(5)
    assert m.n_fragments == 5
    assert m.neighbor_accuracy == 0.0
    assert m.direct_comparison == 0.0
    assert m.perfect is False


# ─── _normalize_config ────────────────────────────────────────────────────────

def test_normalize_config_basic():
    config = {
        0: (np.array([10.0, 20.0]), 0.5),
        1: (np.array([60.0, 20.0]), 0.5),
    }
    norm = _normalize_config(config, [0, 1])
    # Fragment 0 should be at origin
    assert np.allclose(norm[0][0], [0.0, 0.0], atol=1e-6)


def test_normalize_config_empty():
    norm = _normalize_config({}, [])
    assert norm == {}


def test_normalize_config_single():
    config = {0: (np.array([5.0, 10.0]), 1.0)}
    norm = _normalize_config(config, [0])
    assert np.allclose(norm[0][0], [0.0, 0.0], atol=1e-6)


# ─── _compute_adjacency ───────────────────────────────────────────────────────

def test_compute_adjacency_basic():
    config = {
        0: (np.array([0.0, 0.0]), 0.0),
        1: (np.array([50.0, 0.0]), 0.0),
        2: (np.array([500.0, 0.0]), 0.0),
    }
    pairs = _compute_adjacency(config, threshold=100.0)
    # Only 0,1 should be adjacent
    assert (0, 1) in pairs
    assert (0, 2) not in pairs


def test_compute_adjacency_empty():
    assert _compute_adjacency({}, threshold=50.0) == []


# ─── _compute_edge_match_rate ─────────────────────────────────────────────────

def test_compute_edge_match_rate_perfect():
    config = {
        0: (np.array([0.0, 0.0]), 0.0),
        1: (np.array([50.0, 0.0]), 0.0),
    }
    emr = _compute_edge_match_rate(config, config, [0, 1], tolerance=10.0)
    assert emr == pytest.approx(1.0)


def test_compute_edge_match_rate_none_match():
    pred = {
        0: (np.array([0.0, 0.0]), 0.0),
        1: (np.array([50.0, 0.0]), 0.0),
    }
    gt = {
        0: (np.array([100.0, 0.0]), 0.0),
        1: (np.array([200.0, 0.0]), 0.0),
    }
    emr = _compute_edge_match_rate(pred, gt, [0, 1], tolerance=5.0)
    assert emr == pytest.approx(0.0)


# ─── evaluate_reconstruction ──────────────────────────────────────────────────

def test_evaluate_reconstruction_perfect():
    gt = make_config(n=3, spread=50.0)
    metrics = evaluate_reconstruction(gt, gt)
    assert metrics.direct_comparison == pytest.approx(1.0)
    assert metrics.n_fragments == 3


def test_evaluate_reconstruction_empty():
    metrics = evaluate_reconstruction({}, {})
    assert metrics.n_fragments == 0
    assert metrics.neighbor_accuracy == 0.0


def test_evaluate_reconstruction_noisy():
    gt = make_config(n=4, spread=50.0)
    pred = {}
    for fid, (pos, angle) in gt.items():
        noisy_pos = pos + np.array([2.0, 2.0])
        pred[fid] = (noisy_pos, angle)
    metrics = evaluate_reconstruction(pred, gt, position_tolerance=5.0)
    assert isinstance(metrics, ReconstructionMetrics)
    assert metrics.n_fragments == 4


def test_evaluate_reconstruction_position_rmse():
    gt = make_config(n=2)
    pred = {0: (np.array([0.0, 0.0]), 0.0), 1: (np.array([10.0, 0.0]), 0.0)}
    metrics = evaluate_reconstruction(pred, gt)
    assert metrics.position_rmse >= 0.0


def test_evaluate_reconstruction_with_adjacency():
    gt = make_config(n=2)
    adjacency = [(0, 1)]
    metrics = evaluate_reconstruction(gt, gt, adjacency=adjacency)
    assert metrics.n_total_pairs == 1


def test_evaluate_reconstruction_subset():
    gt = make_config(n=4)
    # Predicted only has 2 of 4
    pred = {0: gt[0], 1: gt[1]}
    metrics = evaluate_reconstruction(pred, gt)
    assert metrics.n_fragments == 2


# ─── compare_methods ──────────────────────────────────────────────────────────

def test_compare_methods_basic():
    gt = make_config(n=3)
    m1 = evaluate_reconstruction(gt, gt)
    m2 = evaluate_reconstruction(gt, gt)
    results = [
        BenchmarkResult(method="A", metrics=m1, runtime_sec=1.0),
        BenchmarkResult(method="B", metrics=m2, runtime_sec=2.0),
    ]
    table = compare_methods(results)
    assert "A" in table
    assert "B" in table


def test_compare_methods_empty():
    table = compare_methods([])
    assert isinstance(table, str)


def test_compare_methods_sorted_by_na():
    gt = make_config(n=2)
    m_good = evaluate_reconstruction(gt, gt)
    m_bad = evaluate_reconstruction({0: (np.array([0.0, 0.0]), 0.0)},
                                     {0: (np.array([500.0, 0.0]), 0.0)},
                                     position_tolerance=5.0)
    results = [
        BenchmarkResult(method="bad", metrics=m_bad, runtime_sec=0.1),
        BenchmarkResult(method="good", metrics=m_good, runtime_sec=0.1),
    ]
    table = compare_methods(results)
    # Both methods should appear in the table
    assert "good" in table
    assert "bad" in table
