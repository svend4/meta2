"""Tests for puzzle_reconstruction.utils.distance_shape_utils."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.distance_shape_utils import (
    DistanceMatrixRecord,
    SimilarityPair,
    DistanceBatchResult,
    ContourMatchRecord,
    ShapeContextBatchSummary,
    MetricsRunRecord,
    EvidenceAggregationRecord,
    make_distance_record,
    make_contour_match,
)

np.random.seed(21)


def _square_matrix(n=5):
    m = np.abs(np.random.randn(n, n))
    m = (m + m.T) / 2  # symmetric
    np.fill_diagonal(m, 0.0)
    return m


# ── 1. DistanceMatrixRecord basic ────────────────────────────────────────────
def test_distance_matrix_record_basic():
    m = _square_matrix(4)
    r = DistanceMatrixRecord(label="test", metric="euclidean", matrix=m)
    assert r.n == 4
    assert r.label == "test"
    assert r.metric == "euclidean"
    assert r.normalized is False


# ── 2. DistanceMatrixRecord n and max_value ───────────────────────────────────
def test_distance_matrix_max_value():
    m = np.array([[0.0, 2.0], [2.0, 0.0]])
    r = DistanceMatrixRecord("x", "l2", m)
    assert r.max_value == 2.0


# ── 3. DistanceMatrixRecord min_offdiag ──────────────────────────────────────
def test_distance_matrix_min_offdiag():
    m = np.array([[0.0, 3.0, 1.0],
                  [3.0, 0.0, 2.0],
                  [1.0, 2.0, 0.0]])
    r = DistanceMatrixRecord("x", "l1", m)
    assert r.min_offdiag == 1.0


# ── 4. DistanceMatrixRecord non-square raises ─────────────────────────────────
def test_distance_matrix_non_square():
    with pytest.raises(ValueError):
        DistanceMatrixRecord("x", "l2", np.ones((3, 4)))


# ── 5. make_distance_record ──────────────────────────────────────────────────
def test_make_distance_record():
    m = _square_matrix(3)
    r = make_distance_record("label", "cosine", m, normalized=True)
    assert r.label == "label"
    assert r.metric == "cosine"
    assert r.normalized is True


# ── 6. SimilarityPair basic ──────────────────────────────────────────────────
def test_similarity_pair_basic():
    sp = SimilarityPair(0, 1, 0.8)
    assert sp.i == 0
    assert sp.j == 1
    assert sp.similarity == 0.8
    assert sp.is_high is True


def test_similarity_pair_low():
    sp = SimilarityPair(2, 3, 0.3)
    assert sp.is_high is False


# ── 7. SimilarityPair validation ─────────────────────────────────────────────
def test_similarity_pair_invalid_index():
    with pytest.raises(ValueError):
        SimilarityPair(-1, 0, 0.5)


def test_similarity_pair_invalid_similarity():
    with pytest.raises(ValueError):
        SimilarityPair(0, 1, 1.5)


# ── 9. DistanceBatchResult best_pair ─────────────────────────────────────────
def test_distance_batch_best_pair():
    r = DistanceBatchResult(
        n_queries=3, metric="euclidean",
        top_pairs=[(0, 1, 0.9), (1, 2, 0.7)],
    )
    assert r.best_pair == (0, 1, 0.9)


def test_distance_batch_empty():
    r = DistanceBatchResult(n_queries=0, metric="l2")
    assert r.best_pair is None


# ── 11. ContourMatchRecord basic ─────────────────────────────────────────────
def test_contour_match_basic():
    r = ContourMatchRecord(0, 1, cost=2.5, n_correspondences=10, similarity=0.8)
    assert r.contour_id_a == 0
    assert r.contour_id_b == 1
    assert r.cost == 2.5
    assert r.n_correspondences == 10
    assert r.is_match is True


def test_contour_match_not_match():
    r = ContourMatchRecord(0, 1, cost=1.0, n_correspondences=5, similarity=0.3)
    assert r.is_match is False


# ── 12. ContourMatchRecord invalid cost ──────────────────────────────────────
def test_contour_match_negative_cost():
    with pytest.raises(ValueError):
        ContourMatchRecord(0, 1, cost=-1.0, n_correspondences=5, similarity=0.5)


# ── 13. ContourMatchRecord invalid similarity ─────────────────────────────────
def test_contour_match_invalid_similarity():
    with pytest.raises(ValueError):
        ContourMatchRecord(0, 1, cost=1.0, n_correspondences=5, similarity=1.5)


# ── 14. make_contour_match ───────────────────────────────────────────────────
def test_make_contour_match():
    r = make_contour_match(2, 3, 1.5, 8, 0.75)
    assert r.contour_id_a == 2
    assert r.contour_id_b == 3
    assert r.cost == 1.5
    assert r.similarity == 0.75


# ── 15. ShapeContextBatchSummary ─────────────────────────────────────────────
def test_shape_context_batch_summary():
    s = ShapeContextBatchSummary(n_contours=5, mean_similarity=0.7,
                                  best_pair=(0, 1), worst_pair=(3, 4))
    assert s.n_contours == 5
    assert s.is_valid is True


def test_shape_context_batch_invalid():
    s = ShapeContextBatchSummary(n_contours=0, mean_similarity=0.5)
    assert s.is_valid is False


# ── 16. MetricsRunRecord basic ───────────────────────────────────────────────
def test_metrics_run_record():
    r = MetricsRunRecord(run_id="run1", precision=0.9, recall=0.8,
                          f1=0.85, n_fragments=10)
    assert r.run_id == "run1"
    assert r.is_perfect is False


def test_metrics_run_perfect():
    r = MetricsRunRecord("run0", 1.0, 1.0, 1.0, 5)
    assert r.is_perfect is True


def test_metrics_run_invalid():
    with pytest.raises(ValueError):
        MetricsRunRecord("bad", 1.2, 0.9, 0.9, 5)


# ── 17. EvidenceAggregationRecord ────────────────────────────────────────────
def test_evidence_aggregation_record():
    r = EvidenceAggregationRecord(step=1, pair_id=(0, 1), n_channels=3,
                                   confidence=0.8, dominant_channel="color")
    assert r.is_confident is True


def test_evidence_aggregation_not_confident():
    r = EvidenceAggregationRecord(0, (1, 2), 2, 0.3)
    assert r.is_confident is False


def test_evidence_aggregation_invalid_confidence():
    with pytest.raises(ValueError):
        EvidenceAggregationRecord(0, (0,1), 1, -0.1)
