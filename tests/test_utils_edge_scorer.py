"""Tests for puzzle_reconstruction.utils.edge_scorer"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.edge_scorer import (
    EdgeScoreConfig,
    EdgeScoreResult,
    score_edge_overlap,
    score_edge_curvature,
    score_edge_length,
    score_edge_endpoints,
    aggregate_edge_scores,
    rank_edge_pairs,
    batch_score_edges,
)

np.random.seed(42)


# ─── EdgeScoreConfig ─────────────────────────────────────────────────────────

def test_edge_score_config_defaults():
    cfg = EdgeScoreConfig()
    assert cfg.n_samples == 64
    assert cfg.length_tol == pytest.approx(0.5)
    assert cfg.endpoint_sigma == pytest.approx(10.0)


def test_edge_score_config_invalid_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        EdgeScoreConfig(n_samples=1)


def test_edge_score_config_invalid_length_tol():
    with pytest.raises(ValueError, match="length_tol"):
        EdgeScoreConfig(length_tol=-0.1)


def test_edge_score_config_invalid_endpoint_sigma():
    with pytest.raises(ValueError, match="endpoint_sigma"):
        EdgeScoreConfig(endpoint_sigma=0.0)


def test_edge_score_config_invalid_weight():
    with pytest.raises(ValueError, match="weight"):
        EdgeScoreConfig(weights={"overlap": -1.0, "curvature": 0.3,
                                  "length": 0.15, "endpoints": 0.15})


def test_edge_score_config_normalized_weights_sum():
    cfg = EdgeScoreConfig()
    nw = cfg.normalized_weights
    assert sum(nw.values()) == pytest.approx(1.0)


def test_edge_score_config_zero_weights_equal_split():
    cfg = EdgeScoreConfig(weights={"overlap": 0.0, "curvature": 0.0,
                                    "length": 0.0, "endpoints": 0.0})
    nw = cfg.normalized_weights
    assert all(abs(v - 0.25) < 1e-9 for v in nw.values())


# ─── EdgeScoreResult ─────────────────────────────────────────────────────────

def test_edge_score_result_to_dict():
    r = EdgeScoreResult(overlap=0.8, curvature=0.7, length=0.9,
                        endpoints=0.6, total=0.75)
    d = r.to_dict()
    assert d["overlap"] == pytest.approx(0.8)
    assert d["total"] == pytest.approx(0.75)
    assert set(d.keys()) == {"overlap", "curvature", "length", "endpoints", "total"}


def test_edge_score_result_defaults():
    r = EdgeScoreResult()
    assert r.total == 0.0


# ─── score_edge_overlap ──────────────────────────────────────────────────────

def _straight_curve(n=20, length=100.0):
    return np.column_stack([np.linspace(0, length, n), np.zeros(n)])


def test_score_edge_overlap_identical():
    curve = _straight_curve()
    score = score_edge_overlap(curve, curve)
    assert 0.0 <= score <= 1.0


def test_score_edge_overlap_range():
    a = _straight_curve(20, 100)
    b = _straight_curve(20, 100) + np.array([0, 50])
    score = score_edge_overlap(a, b)
    assert 0.0 <= score <= 1.0


def test_score_edge_overlap_invalid_shape():
    with pytest.raises(ValueError):
        score_edge_overlap(np.random.rand(5, 3), np.random.rand(5, 2))


def test_score_edge_overlap_empty_curve():
    a = np.zeros((0, 2))
    b = _straight_curve()
    score = score_edge_overlap(a, b)
    assert score == pytest.approx(0.0)


# ─── score_edge_curvature ────────────────────────────────────────────────────

def test_score_edge_curvature_same_curve():
    curve = _straight_curve()
    score = score_edge_curvature(curve, curve)
    assert 0.0 <= score <= 1.0


def test_score_edge_curvature_few_points():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [1.0, 1.0]])
    score = score_edge_curvature(a, b)
    assert score == pytest.approx(0.5)


def test_score_edge_curvature_range():
    a = np.random.rand(15, 2)
    b = np.random.rand(15, 2)
    score = score_edge_curvature(a, b)
    assert 0.0 <= score <= 1.0


# ─── score_edge_length ───────────────────────────────────────────────────────

def test_score_edge_length_equal():
    curve = _straight_curve(20, 100)
    score = score_edge_length(curve, curve)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_score_edge_length_zero_both():
    a = np.array([[5.0, 5.0]])
    b = np.array([[5.0, 5.0]])
    score = score_edge_length(a, b)
    assert score == pytest.approx(1.0)


def test_score_edge_length_one_zero():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[5.0, 5.0]])
    score = score_edge_length(a, b)
    assert score == pytest.approx(0.0)


def test_score_edge_length_range():
    a = _straight_curve(20, 100)
    b = _straight_curve(20, 200)
    score = score_edge_length(a, b)
    assert 0.0 <= score <= 1.0


# ─── score_edge_endpoints ────────────────────────────────────────────────────

def test_score_edge_endpoints_perfect_join():
    # a[-1] = b[0], a[0] = b[-1]
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    b = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    score = score_edge_endpoints(a, b)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_score_edge_endpoints_far_apart():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[1000.0, 1000.0], [2000.0, 2000.0]])
    score = score_edge_endpoints(a, b)
    assert score < 0.1


def test_score_edge_endpoints_empty():
    a = np.zeros((0, 2))
    b = _straight_curve()
    score = score_edge_endpoints(a, b)
    assert score == pytest.approx(0.0)


def test_score_edge_endpoints_range():
    a = np.random.rand(10, 2) * 10
    b = np.random.rand(10, 2) * 10
    score = score_edge_endpoints(a, b)
    assert 0.0 <= score <= 1.0


# ─── aggregate_edge_scores ───────────────────────────────────────────────────

def test_aggregate_edge_scores_all_one():
    cfg = EdgeScoreConfig()
    score = aggregate_edge_scores(1.0, 1.0, 1.0, 1.0, cfg)
    assert score == pytest.approx(1.0)


def test_aggregate_edge_scores_all_zero():
    cfg = EdgeScoreConfig()
    score = aggregate_edge_scores(0.0, 0.0, 0.0, 0.0, cfg)
    assert score == pytest.approx(0.0)


def test_aggregate_edge_scores_clipped():
    cfg = EdgeScoreConfig(weights={"overlap": 1.0, "curvature": 1.0,
                                    "length": 1.0, "endpoints": 1.0})
    score = aggregate_edge_scores(1.0, 1.0, 1.0, 1.0, cfg)
    assert score <= 1.0


def test_aggregate_edge_scores_weighted():
    cfg = EdgeScoreConfig(weights={"overlap": 1.0, "curvature": 0.0,
                                    "length": 0.0, "endpoints": 0.0})
    score = aggregate_edge_scores(0.5, 0.0, 0.0, 0.0, cfg)
    assert score == pytest.approx(0.5)


# ─── rank_edge_pairs ─────────────────────────────────────────────────────────

def test_rank_edge_pairs_order():
    pairs = [
        (0, 1, EdgeScoreResult(total=0.3)),
        (2, 3, EdgeScoreResult(total=0.9)),
        (4, 5, EdgeScoreResult(total=0.6)),
    ]
    ranked = rank_edge_pairs(pairs)
    assert ranked[0][2].total == pytest.approx(0.9)
    assert ranked[-1][2].total == pytest.approx(0.3)


def test_rank_edge_pairs_empty():
    ranked = rank_edge_pairs([])
    assert ranked == []


# ─── batch_score_edges ───────────────────────────────────────────────────────

def test_batch_score_edges_basic():
    curves_a = [_straight_curve(10), _straight_curve(10)]
    curves_b = [_straight_curve(10), _straight_curve(10) + np.array([0, 5])]
    results = batch_score_edges(curves_a, curves_b)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, EdgeScoreResult)
        assert 0.0 <= r.total <= 1.0


def test_batch_score_edges_length_mismatch():
    with pytest.raises(ValueError):
        batch_score_edges([_straight_curve()], [_straight_curve(), _straight_curve()])


def test_batch_score_edges_empty():
    results = batch_score_edges([], [])
    assert results == []


def test_batch_score_edges_result_fields():
    curves_a = [np.random.rand(10, 2)]
    curves_b = [np.random.rand(10, 2)]
    results = batch_score_edges(curves_a, curves_b)
    r = results[0]
    assert hasattr(r, "overlap")
    assert hasattr(r, "curvature")
    assert hasattr(r, "length")
    assert hasattr(r, "endpoints")
    assert hasattr(r, "total")
