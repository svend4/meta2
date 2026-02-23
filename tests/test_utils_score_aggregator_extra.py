"""Extra tests for puzzle_reconstruction.utils.score_aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.score_aggregator import (
    AggregationMethod,
    AggregationResult,
    ScoreVector,
    aggregate_matrix,
    aggregate_pair,
    batch_aggregate,
    geometric_mean,
    harmonic_mean,
    top_k_pairs,
    weighted_sum,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sv(a=0, b=1, channels=None):
    if channels is None:
        channels = {"color": 0.6, "shape": 0.8}
    return ScoreVector(idx_a=a, idx_b=b, channels=channels)


def _result(scores=None):
    if scores is None:
        scores = {(0, 1): 0.7, (0, 2): 0.5, (1, 2): 0.3}
    n = len(scores)
    top = max(scores, key=scores.__getitem__) if scores else None
    m = sum(scores.values()) / n if n else 0.0
    return AggregationResult(scores=scores, method="weighted",
                              n_pairs=n, mean=m, top_pair=top)


# ─── TestAggregationMethodExtra ───────────────────────────────────────────────

class TestAggregationMethodExtra:
    def test_min_value(self):
        assert AggregationMethod.MIN.value == "min"

    def test_max_value(self):
        assert AggregationMethod.MAX.value == "max"

    def test_all_five_members(self):
        assert len(list(AggregationMethod)) == 5

    def test_from_value(self):
        m = AggregationMethod("geometric")
        assert m == AggregationMethod.GEOMETRIC


# ─── TestScoreVectorExtra ─────────────────────────────────────────────────────

class TestScoreVectorExtra:
    def test_boundary_zero_ok(self):
        sv = ScoreVector(0, 1, channels={"a": 0.0, "b": 1.0})
        assert sv.min_score == pytest.approx(0.0)
        assert sv.max_score == pytest.approx(1.0)

    def test_pair_key_equal_indices(self):
        sv = ScoreVector(3, 3, channels={"c": 0.5})
        assert sv.pair_key == (3, 3)

    def test_n_channels_zero(self):
        sv = ScoreVector(0, 1, channels={})
        assert sv.n_channels == 0

    def test_weights_partial(self):
        # weights for only one of two channels — no error
        sv = ScoreVector(0, 1, channels={"a": 0.4, "b": 0.6},
                         weights={"a": 2.0})
        assert sv.n_channels == 2

    def test_single_channel_mean_equals_value(self):
        sv = ScoreVector(0, 1, channels={"only": 0.77})
        assert sv.mean_score == pytest.approx(0.77)

    def test_idx_a_b_stored(self):
        sv = ScoreVector(4, 7, channels={"c": 0.5})
        assert sv.idx_a == 4
        assert sv.idx_b == 7


# ─── TestAggregationResultExtra ───────────────────────────────────────────────

class TestAggregationResultExtra:
    def test_top_pair_is_max(self):
        r = _result({(0, 1): 0.9, (0, 2): 0.3})
        assert r.top_pair == (0, 1)

    def test_get_score_reversed_canonical(self):
        r = _result({(0, 1): 0.8})
        assert r.get_score(1, 0) == pytest.approx(0.8)

    def test_get_score_missing_returns_none(self):
        r = _result()
        assert r.get_score(9, 10) is None

    def test_method_stored(self):
        r = _result()
        assert r.method == "weighted"

    def test_scores_dict_accessible(self):
        r = _result({(0, 1): 0.5})
        assert (0, 1) in r.scores

    def test_zero_n_pairs_mean_zero(self):
        r = AggregationResult(scores={}, method="weighted",
                               n_pairs=0, mean=0.0, top_pair=None)
        assert r.mean == pytest.approx(0.0)


# ─── TestWeightedSumExtra ─────────────────────────────────────────────────────

class TestWeightedSumExtra:
    def test_single_channel(self):
        result = weighted_sum({"only": 0.65})
        assert result == pytest.approx(0.65)

    def test_custom_weights_dominant(self):
        # channel 'a' has weight=10, 'b' has weight=0 → result ~= value of 'a'
        result = weighted_sum({"a": 0.9, "b": 0.1}, weights={"a": 10.0, "b": 0.0})
        assert result == pytest.approx(0.9, abs=0.02)

    def test_all_zero_values(self):
        result = weighted_sum({"a": 0.0, "b": 0.0})
        assert result == pytest.approx(0.0)

    def test_all_one_values(self):
        result = weighted_sum({"a": 1.0, "b": 1.0})
        assert result == pytest.approx(1.0)

    def test_three_equal_channels(self):
        result = weighted_sum({"a": 0.5, "b": 0.5, "c": 0.5})
        assert result == pytest.approx(0.5)


# ─── TestHarmonicMeanExtra ────────────────────────────────────────────────────

class TestHarmonicMeanExtra:
    def test_single_channel(self):
        result = harmonic_mean({"only": 0.7})
        assert result == pytest.approx(0.7)

    def test_all_zeros_result_near_zero(self):
        # implementation uses epsilon to avoid division by zero
        result = harmonic_mean({"a": 0.0, "b": 0.0})
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_result_nonneg(self):
        result = harmonic_mean({"a": 0.3, "b": 0.7, "c": 0.5})
        assert result >= 0.0

    def test_leq_geometric(self):
        channels = {"a": 0.2, "b": 0.8}
        h = harmonic_mean(channels)
        g = geometric_mean(channels)
        assert h <= g + 1e-6


# ─── TestGeometricMeanExtra ───────────────────────────────────────────────────

class TestGeometricMeanExtra:
    def test_single_channel(self):
        result = geometric_mean({"only": 0.64})
        assert result == pytest.approx(0.64)

    def test_two_channels_known(self):
        result = geometric_mean({"a": 0.25, "b": 1.0})
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_all_zeros_result_near_zero(self):
        # implementation uses epsilon to avoid log(0)
        result = geometric_mean({"a": 0.0, "b": 0.0})
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_result_nonneg(self):
        result = geometric_mean({"a": 0.4, "b": 0.6, "c": 0.9})
        assert result >= 0.0


# ─── TestAggregatePairExtra ───────────────────────────────────────────────────

class TestAggregatePairExtra:
    def test_min_returns_minimum(self):
        sv = _sv(channels={"a": 0.2, "b": 0.9})
        assert aggregate_pair(sv, AggregationMethod.MIN) == pytest.approx(0.2)

    def test_max_returns_maximum(self):
        sv = _sv(channels={"a": 0.2, "b": 0.9})
        assert aggregate_pair(sv, AggregationMethod.MAX) == pytest.approx(0.9)

    def test_single_channel_all_methods_same(self):
        sv = _sv(channels={"only": 0.6})
        for m in AggregationMethod:
            result = aggregate_pair(sv, m)
            assert result == pytest.approx(0.6, abs=0.01)

    def test_weighted_with_equal_weights(self):
        sv = _sv(channels={"a": 0.4, "b": 0.6})
        result = aggregate_pair(sv, AggregationMethod.WEIGHTED)
        assert result == pytest.approx(0.5)

    def test_default_method_is_weighted(self):
        sv = _sv(channels={"a": 0.4, "b": 0.6})
        assert aggregate_pair(sv) == pytest.approx(
            aggregate_pair(sv, AggregationMethod.WEIGHTED)
        )


# ─── TestAggregateMatrixExtra ─────────────────────────────────────────────────

class TestAggregateMatrixExtra:
    def test_diagonal_zero(self):
        vectors = [_sv(0, 1), _sv(0, 2), _sv(1, 2)]
        result = aggregate_matrix(vectors, n_fragments=3)
        np.testing.assert_array_equal(np.diag(result), np.zeros(3, dtype=np.float32))

    def test_values_filled(self):
        sv = ScoreVector(0, 1, channels={"c": 0.75})
        result = aggregate_matrix([sv], n_fragments=2)
        assert result[0, 1] == pytest.approx(0.75, abs=0.01)
        assert result[1, 0] == pytest.approx(0.75, abs=0.01)

    def test_large_n_fragments(self):
        result = aggregate_matrix([], n_fragments=10)
        assert result.shape == (10, 10)

    def test_dtype_float32_with_data(self):
        vectors = [_sv(0, 1)]
        result = aggregate_matrix(vectors, n_fragments=3)
        assert result.dtype == np.float32

    def test_all_pairs_3x3(self):
        vectors = [_sv(i, j, channels={"c": 0.5})
                   for i in range(3) for j in range(i + 1, 3)]
        result = aggregate_matrix(vectors, n_fragments=3)
        # off-diagonal should all be 0.5
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert result[i, j] == pytest.approx(0.5, abs=0.01)


# ─── TestTopKPairsExtra ───────────────────────────────────────────────────────

class TestTopKPairsExtra:
    def test_pairs_are_tuples(self):
        r = _result()
        for pair, _ in top_k_pairs(r, k=2):
            assert isinstance(pair, tuple)

    def test_scores_in_0_1(self):
        r = _result()
        for _, s in top_k_pairs(r, k=3):
            assert 0.0 <= s <= 1.0

    def test_k_one_returns_top(self):
        scores = {(0, 1): 0.9, (0, 2): 0.5, (1, 2): 0.1}
        r = _result(scores)
        result = top_k_pairs(r, k=1)
        assert result[0][0] == (0, 1)

    def test_empty_result_k_exceeds(self):
        r = AggregationResult(scores={}, method="weighted",
                               n_pairs=0, mean=0.0, top_pair=None)
        assert top_k_pairs(r, k=5) == []


# ─── TestBatchAggregateExtra ──────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def test_single_vector(self):
        sv = ScoreVector(0, 1, channels={"c": 0.5})
        result = batch_aggregate([sv])
        assert result.n_pairs == 1
        assert result.mean == pytest.approx(0.5, abs=0.1)

    def test_geometric_method(self):
        sv = ScoreVector(0, 1, channels={"c": 0.8})
        result = batch_aggregate([sv], method=AggregationMethod.GEOMETRIC)
        assert result.method == "geometric"

    def test_min_method(self):
        sv = ScoreVector(0, 1, channels={"a": 0.3, "b": 0.9})
        result = batch_aggregate([sv], method=AggregationMethod.MIN)
        assert result.scores[(0, 1)] == pytest.approx(0.3)

    def test_dedup_pair_kept_once(self):
        sv1 = ScoreVector(0, 1, channels={"c": 0.4})
        sv2 = ScoreVector(0, 1, channels={"c": 0.4})
        result = batch_aggregate([sv1, sv2])
        assert result.n_pairs == 1

    def test_top_pair_is_best(self):
        sv1 = ScoreVector(0, 1, channels={"c": 0.9})
        sv2 = ScoreVector(1, 2, channels={"c": 0.3})
        result = batch_aggregate([sv1, sv2])
        assert result.top_pair == (0, 1)
