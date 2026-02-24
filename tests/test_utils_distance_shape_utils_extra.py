"""Extra tests for puzzle_reconstruction/utils/distance_shape_utils.py"""
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


# ─── DistanceMatrixRecord ─────────────────────────────────────────────────────

class TestDistanceMatrixRecord:
    def _make_3x3(self):
        return np.array([[0.0, 1.0, 2.0],
                         [1.0, 0.0, 1.5],
                         [2.0, 1.5, 0.0]])

    def test_basic_creation(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="test", metric="l2", matrix=mat)
        assert rec.label == "test"
        assert rec.metric == "l2"

    def test_n_property(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat)
        assert rec.n == 3

    def test_max_value_property(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat)
        assert rec.max_value == pytest.approx(2.0)

    def test_min_offdiag_property(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat)
        assert rec.min_offdiag == pytest.approx(1.0)

    def test_non_square_raises(self):
        mat = np.ones((2, 3))
        with pytest.raises(ValueError):
            DistanceMatrixRecord(label="bad", metric="l2", matrix=mat)

    def test_non_2d_raises(self):
        mat = np.ones((3,))
        with pytest.raises(ValueError):
            DistanceMatrixRecord(label="bad", metric="l2", matrix=mat)

    def test_normalized_default_false(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat)
        assert rec.normalized is False

    def test_normalized_custom(self):
        mat = self._make_3x3()
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat, normalized=True)
        assert rec.normalized is True

    def test_1x1_matrix(self):
        mat = np.array([[0.0]])
        rec = DistanceMatrixRecord(label="x", metric="l2", matrix=mat)
        assert rec.n == 1
        assert rec.max_value == pytest.approx(0.0)


# ─── SimilarityPair ───────────────────────────────────────────────────────────

class TestSimilarityPair:
    def test_basic_creation(self):
        sp = SimilarityPair(i=0, j=1, similarity=0.7)
        assert sp.i == 0
        assert sp.j == 1
        assert sp.similarity == pytest.approx(0.7)

    def test_is_high_true(self):
        sp = SimilarityPair(i=0, j=1, similarity=0.8)
        assert sp.is_high is True

    def test_is_high_at_boundary(self):
        sp = SimilarityPair(i=0, j=1, similarity=0.5)
        assert sp.is_high is True

    def test_is_high_false(self):
        sp = SimilarityPair(i=0, j=1, similarity=0.3)
        assert sp.is_high is False

    def test_negative_i_raises(self):
        with pytest.raises(ValueError):
            SimilarityPair(i=-1, j=0, similarity=0.5)

    def test_negative_j_raises(self):
        with pytest.raises(ValueError):
            SimilarityPair(i=0, j=-1, similarity=0.5)

    def test_similarity_above_one_raises(self):
        with pytest.raises(ValueError):
            SimilarityPair(i=0, j=1, similarity=1.1)

    def test_similarity_below_zero_raises(self):
        with pytest.raises(ValueError):
            SimilarityPair(i=0, j=1, similarity=-0.1)

    def test_similarity_zero_is_ok(self):
        sp = SimilarityPair(i=0, j=1, similarity=0.0)
        assert sp.similarity == 0.0
        assert sp.is_high is False

    def test_similarity_one_is_ok(self):
        sp = SimilarityPair(i=2, j=3, similarity=1.0)
        assert sp.similarity == 1.0
        assert sp.is_high is True


# ─── DistanceBatchResult ──────────────────────────────────────────────────────

class TestDistanceBatchResult:
    def test_basic_creation(self):
        dbr = DistanceBatchResult(n_queries=5, metric="l2")
        assert dbr.n_queries == 5
        assert dbr.metric == "l2"

    def test_top_pairs_default_empty(self):
        dbr = DistanceBatchResult(n_queries=3, metric="cosine")
        assert dbr.top_pairs == []

    def test_best_pair_none_when_empty(self):
        dbr = DistanceBatchResult(n_queries=3, metric="cosine")
        assert dbr.best_pair is None

    def test_best_pair_returns_first(self):
        pairs = [(0, 1, 0.2), (2, 3, 0.5)]
        dbr = DistanceBatchResult(n_queries=4, metric="l2", top_pairs=pairs)
        assert dbr.best_pair == (0, 1, 0.2)

    def test_top_pairs_stored(self):
        pairs = [(1, 2, 0.3)]
        dbr = DistanceBatchResult(n_queries=2, metric="l2", top_pairs=pairs)
        assert len(dbr.top_pairs) == 1

    def test_n_queries_stored(self):
        dbr = DistanceBatchResult(n_queries=10, metric="l2")
        assert dbr.n_queries == 10


# ─── ContourMatchRecord ───────────────────────────────────────────────────────

class TestContourMatchRecord:
    def test_basic_creation(self):
        rec = ContourMatchRecord(
            contour_id_a=0, contour_id_b=1, cost=0.5, n_correspondences=10
        )
        assert rec.contour_id_a == 0
        assert rec.contour_id_b == 1
        assert rec.cost == pytest.approx(0.5)
        assert rec.n_correspondences == 10

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            ContourMatchRecord(
                contour_id_a=0, contour_id_b=1, cost=-0.1, n_correspondences=5
            )

    def test_similarity_default(self):
        rec = ContourMatchRecord(
            contour_id_a=0, contour_id_b=1, cost=0.0, n_correspondences=0
        )
        assert rec.similarity == 0.0

    def test_similarity_above_one_raises(self):
        with pytest.raises(ValueError):
            ContourMatchRecord(
                contour_id_a=0, contour_id_b=1, cost=0.0, n_correspondences=5,
                similarity=1.5
            )

    def test_is_match_true(self):
        rec = ContourMatchRecord(
            contour_id_a=0, contour_id_b=1, cost=0.2, n_correspondences=8,
            similarity=0.7
        )
        assert rec.is_match is True

    def test_is_match_false(self):
        rec = ContourMatchRecord(
            contour_id_a=0, contour_id_b=1, cost=0.8, n_correspondences=3,
            similarity=0.3
        )
        assert rec.is_match is False

    def test_zero_cost_ok(self):
        rec = ContourMatchRecord(
            contour_id_a=2, contour_id_b=3, cost=0.0, n_correspondences=15
        )
        assert rec.cost == 0.0


# ─── ShapeContextBatchSummary ─────────────────────────────────────────────────

class TestShapeContextBatchSummary:
    def test_basic_creation(self):
        s = ShapeContextBatchSummary(n_contours=5, mean_similarity=0.6)
        assert s.n_contours == 5
        assert s.mean_similarity == pytest.approx(0.6)

    def test_is_valid_true(self):
        s = ShapeContextBatchSummary(n_contours=3, mean_similarity=0.5)
        assert s.is_valid is True

    def test_is_valid_false_zero_contours(self):
        s = ShapeContextBatchSummary(n_contours=0, mean_similarity=0.5)
        assert s.is_valid is False

    def test_best_pair_default_none(self):
        s = ShapeContextBatchSummary(n_contours=2, mean_similarity=0.4)
        assert s.best_pair is None

    def test_worst_pair_stored(self):
        s = ShapeContextBatchSummary(n_contours=4, mean_similarity=0.3,
                                      worst_pair=(2, 3))
        assert s.worst_pair == (2, 3)

    def test_is_valid_false_invalid_similarity(self):
        s = ShapeContextBatchSummary(n_contours=2, mean_similarity=1.5)
        assert s.is_valid is False


# ─── MetricsRunRecord ─────────────────────────────────────────────────────────

class TestMetricsRunRecord:
    def test_basic_creation(self):
        rec = MetricsRunRecord(
            run_id="run1", precision=0.8, recall=0.7, f1=0.75, n_fragments=10
        )
        assert rec.run_id == "run1"
        assert rec.precision == pytest.approx(0.8)

    def test_is_perfect_false(self):
        rec = MetricsRunRecord(
            run_id="r", precision=0.9, recall=1.0, f1=1.0, n_fragments=5
        )
        assert rec.is_perfect is False

    def test_is_perfect_true(self):
        rec = MetricsRunRecord(
            run_id="r", precision=1.0, recall=1.0, f1=1.0, n_fragments=5
        )
        assert rec.is_perfect is True

    def test_precision_above_one_raises(self):
        with pytest.raises(ValueError):
            MetricsRunRecord(
                run_id="r", precision=1.1, recall=0.9, f1=0.9, n_fragments=5
            )

    def test_recall_below_zero_raises(self):
        with pytest.raises(ValueError):
            MetricsRunRecord(
                run_id="r", precision=0.9, recall=-0.1, f1=0.9, n_fragments=5
            )

    def test_extra_dict_default_empty(self):
        rec = MetricsRunRecord(
            run_id="r", precision=0.5, recall=0.5, f1=0.5, n_fragments=3
        )
        assert rec.extra == {}


# ─── EvidenceAggregationRecord ────────────────────────────────────────────────

class TestEvidenceAggregationRecord:
    def test_basic_creation(self):
        rec = EvidenceAggregationRecord(
            step=0, pair_id=(1, 2), n_channels=3, confidence=0.7
        )
        assert rec.step == 0
        assert rec.pair_id == (1, 2)
        assert rec.n_channels == 3

    def test_is_confident_true(self):
        rec = EvidenceAggregationRecord(
            step=1, pair_id=(0, 1), n_channels=2, confidence=0.8
        )
        assert rec.is_confident is True

    def test_is_confident_false(self):
        rec = EvidenceAggregationRecord(
            step=1, pair_id=(0, 1), n_channels=2, confidence=0.3
        )
        assert rec.is_confident is False

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError):
            EvidenceAggregationRecord(
                step=0, pair_id=(0, 1), n_channels=1, confidence=1.5
            )

    def test_dominant_channel_default_none(self):
        rec = EvidenceAggregationRecord(
            step=0, pair_id=(0, 1), n_channels=2, confidence=0.5
        )
        assert rec.dominant_channel is None

    def test_dominant_channel_stored(self):
        rec = EvidenceAggregationRecord(
            step=2, pair_id=(0, 1), n_channels=3, confidence=0.6,
            dominant_channel="color"
        )
        assert rec.dominant_channel == "color"


# ─── make_distance_record ─────────────────────────────────────────────────────

class TestMakeDistanceRecord:
    def test_returns_distance_matrix_record(self):
        mat = np.eye(3)
        rec = make_distance_record("test", "l2", mat)
        assert isinstance(rec, DistanceMatrixRecord)

    def test_label_and_metric(self):
        mat = np.eye(2)
        rec = make_distance_record("my_label", "cosine", mat)
        assert rec.label == "my_label"
        assert rec.metric == "cosine"

    def test_non_square_raises(self):
        mat = np.ones((2, 3))
        with pytest.raises(ValueError):
            make_distance_record("bad", "l2", mat)

    def test_normalized_flag(self):
        mat = np.eye(4)
        rec = make_distance_record("x", "l2", mat, normalized=True)
        assert rec.normalized is True


# ─── make_contour_match ───────────────────────────────────────────────────────

class TestMakeContourMatch:
    def test_returns_contour_match_record(self):
        rec = make_contour_match(0, 1, 0.5, 10)
        assert isinstance(rec, ContourMatchRecord)

    def test_fields_stored(self):
        rec = make_contour_match(2, 3, 1.0, 20, similarity=0.8)
        assert rec.contour_id_a == 2
        assert rec.contour_id_b == 3
        assert rec.cost == pytest.approx(1.0)
        assert rec.n_correspondences == 20
        assert rec.similarity == pytest.approx(0.8)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            make_contour_match(0, 1, -1.0, 5)

    def test_default_similarity_zero(self):
        rec = make_contour_match(0, 1, 0.0, 0)
        assert rec.similarity == 0.0
