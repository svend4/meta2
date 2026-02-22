"""Тесты для puzzle_reconstruction.algorithms.score_aggregator."""
import pytest
import numpy as np
from puzzle_reconstruction.algorithms.score_aggregator import (
    AggregationResult,
    weighted_avg,
    harmonic_mean,
    aggregate_scores,
    threshold_filter,
    top_k_pairs,
    batch_aggregate,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scores(**kv) -> dict:
    return {k: float(v) for k, v in kv.items()}


def _make_result(score=0.5) -> AggregationResult:
    return AggregationResult(
        score=score,
        scores={"a": score},
        weights={"a": 1.0},
        method="weighted_avg",
    )


# ─── TestAggregationResult ────────────────────────────────────────────────────

class TestAggregationResult:
    def test_construction(self):
        r = _make_result(0.7)
        assert r.score == pytest.approx(0.7)

    def test_method_stored(self):
        r = AggregationResult(
            score=0.5, scores={}, weights={}, method="harmonic"
        )
        assert r.method == "harmonic"

    def test_params_default_empty(self):
        r = _make_result()
        assert r.params == {}

    def test_scores_dict_stored(self):
        r = AggregationResult(
            score=0.6,
            scores={"color": 0.6, "texture": 0.8},
            weights={"color": 0.5, "texture": 0.5},
            method="weighted_avg",
        )
        assert r.scores["color"] == pytest.approx(0.6)
        assert r.scores["texture"] == pytest.approx(0.8)


# ─── TestWeightedAvg ──────────────────────────────────────────────────────────

class TestWeightedAvg:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_avg({})

    def test_equal_weights_average(self):
        scores = _scores(a=0.4, b=0.8)
        result = weighted_avg(scores)
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_custom_weights(self):
        scores = _scores(a=1.0, b=0.0)
        weights = {"a": 1.0, "b": 0.0}
        result = weighted_avg(scores, weights=weights)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_zero_total_weight_raises(self):
        scores = _scores(a=0.5)
        weights = {"a": 0.0}
        with pytest.raises(ValueError):
            weighted_avg(scores, weights=weights)

    def test_result_in_range(self):
        scores = _scores(a=0.3, b=0.7, c=0.5)
        result = weighted_avg(scores)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        result = weighted_avg(_scores(a=0.5))
        assert isinstance(result, float)

    def test_single_channel(self):
        result = weighted_avg(_scores(x=0.75))
        assert result == pytest.approx(0.75, abs=1e-6)

    def test_missing_weight_defaults_to_one(self):
        scores = _scores(a=0.4, b=0.8)
        weights = {"a": 1.0}  # b not in weights → default 1.0
        result = weighted_avg(scores, weights=weights)
        assert result == pytest.approx(0.6, abs=1e-6)


# ─── TestHarmonicMean ─────────────────────────────────────────────────────────

class TestHarmonicMean:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            harmonic_mean({})

    def test_zero_channel_returns_zero(self):
        scores = _scores(a=0.5, b=0.0)
        assert harmonic_mean(scores) == pytest.approx(0.0)

    def test_identical_values(self):
        scores = _scores(a=0.5, b=0.5)
        result = harmonic_mean(scores)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_penalizes_weak_channel(self):
        # arithmetic avg = 0.75, harmonic < 0.75
        scores_equal = _scores(a=0.75, b=0.75)
        scores_unequal = _scores(a=1.0, b=0.5)
        h_equal = harmonic_mean(scores_equal)
        h_unequal = harmonic_mean(scores_unequal)
        assert h_unequal < h_equal

    def test_result_in_range(self):
        scores = _scores(a=0.4, b=0.8)
        result = harmonic_mean(scores)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        result = harmonic_mean(_scores(a=0.5))
        assert isinstance(result, float)

    def test_single_channel(self):
        result = harmonic_mean(_scores(only=0.6))
        assert result == pytest.approx(0.6, abs=1e-6)


# ─── TestAggregateScores ──────────────────────────────────────────────────────

class TestAggregateScores:
    def test_weighted_avg_method(self):
        r = aggregate_scores(_scores(a=0.4, b=0.8), method="weighted_avg")
        assert r.score == pytest.approx(0.6, abs=1e-6)
        assert r.method == "weighted_avg"

    def test_harmonic_method(self):
        r = aggregate_scores(_scores(a=0.5, b=0.5), method="harmonic")
        assert r.method == "harmonic"
        assert r.score == pytest.approx(0.5, abs=1e-6)

    def test_min_method(self):
        r = aggregate_scores(_scores(a=0.3, b=0.9), method="min")
        assert r.score == pytest.approx(0.3, abs=1e-6)

    def test_max_method(self):
        r = aggregate_scores(_scores(a=0.3, b=0.9), method="max")
        assert r.score == pytest.approx(0.9, abs=1e-6)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores(_scores(a=0.5), method="geometric")

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores({})

    def test_returns_aggregation_result(self):
        r = aggregate_scores(_scores(a=0.5))
        assert isinstance(r, AggregationResult)

    def test_scores_preserved_in_result(self):
        s = _scores(color=0.6, texture=0.8)
        r = aggregate_scores(s)
        assert r.scores["color"] == pytest.approx(0.6)
        assert r.scores["texture"] == pytest.approx(0.8)

    def test_custom_weights_stored(self):
        s = _scores(a=0.5, b=0.5)
        w = {"a": 2.0, "b": 1.0}
        r = aggregate_scores(s, weights=w)
        assert r.weights["a"] == pytest.approx(2.0)


# ─── TestThresholdFilter ──────────────────────────────────────────────────────

class TestThresholdFilter:
    def test_empty_returns_empty(self):
        assert threshold_filter([]) == []

    def test_above_threshold_true(self):
        results = [_make_result(0.8)]
        mask = threshold_filter(results, threshold=0.5)
        assert mask == [True]

    def test_below_threshold_false(self):
        results = [_make_result(0.3)]
        mask = threshold_filter(results, threshold=0.5)
        assert mask == [False]

    def test_equal_threshold_false(self):
        results = [_make_result(0.5)]
        mask = threshold_filter(results, threshold=0.5)
        # score > threshold (strict), 0.5 > 0.5 is False
        assert mask == [False]

    def test_mixed_results(self):
        results = [_make_result(0.2), _make_result(0.6), _make_result(0.9)]
        mask = threshold_filter(results, threshold=0.5)
        assert mask == [False, True, True]

    def test_length_matches(self):
        results = [_make_result(0.5)] * 5
        mask = threshold_filter(results)
        assert len(mask) == 5


# ─── TestTopKPairs ────────────────────────────────────────────────────────────

class TestTopKPairs:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            top_k_pairs([(0, 1)], [_make_result(), _make_result()], k=1)

    def test_returns_top_k(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        results = [_make_result(0.3), _make_result(0.9), _make_result(0.6)]
        top = top_k_pairs(pairs, results, k=2)
        assert len(top) == 2
        assert top[0] == (1, 2)  # highest score

    def test_k_larger_than_list(self):
        pairs = [(0, 1)]
        results = [_make_result(0.7)]
        top = top_k_pairs(pairs, results, k=10)
        assert len(top) == 1

    def test_k_zero_empty(self):
        pairs = [(0, 1), (1, 2)]
        results = [_make_result(0.5), _make_result(0.8)]
        top = top_k_pairs(pairs, results, k=0)
        assert top == []

    def test_empty_inputs(self):
        top = top_k_pairs([], [], k=3)
        assert top == []

    def test_sorted_descending(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        scores_vals = [0.4, 0.9, 0.6]
        results = [_make_result(s) for s in scores_vals]
        top = top_k_pairs(pairs, results, k=3)
        top_scores = [r.score for p in top
                      for i, r in enumerate(results) if pairs[i] == p]
        assert top_scores == sorted(top_scores, reverse=True)


# ─── TestBatchAggregate ───────────────────────────────────────────────────────

class TestBatchAggregate:
    def test_returns_ndarray(self):
        mat = np.array([[0.5, 0.7], [0.3, 0.8]], dtype=np.float32)
        out = batch_aggregate(mat)
        assert isinstance(out, np.ndarray)

    def test_output_length(self):
        mat = np.random.rand(10, 3).astype(np.float32)
        out = batch_aggregate(mat)
        assert len(out) == 10

    def test_dtype_float32(self):
        mat = np.array([[0.5, 0.7]], dtype=np.float32)
        out = batch_aggregate(mat)
        assert out.dtype == np.float32

    def test_values_in_range(self):
        mat = np.random.rand(5, 4).astype(np.float32)
        out = batch_aggregate(mat)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            batch_aggregate(np.ones(10))

    def test_zero_channels_raises(self):
        with pytest.raises(ValueError):
            batch_aggregate(np.ones((5, 0), dtype=np.float32))

    def test_custom_channel_names(self):
        mat = np.array([[0.4, 0.8]], dtype=np.float32)
        out = batch_aggregate(mat, channel_names=["color", "texture"])
        assert len(out) == 1

    def test_custom_weights(self):
        mat = np.array([[1.0, 0.0]], dtype=np.float32)
        # weight all on ch_0 → score should be close to 1.0
        out = batch_aggregate(mat, weights=[1.0, 0.0])
        assert out[0] == pytest.approx(1.0, abs=1e-5)

    def test_harmonic_method(self):
        mat = np.array([[0.5, 0.0]], dtype=np.float32)
        out = batch_aggregate(mat, method="harmonic")
        assert out[0] == pytest.approx(0.0, abs=1e-5)

    def test_min_method(self):
        mat = np.array([[0.3, 0.9, 0.6]], dtype=np.float32)
        out = batch_aggregate(mat, method="min")
        assert out[0] == pytest.approx(0.3, abs=1e-5)

    def test_max_method(self):
        mat = np.array([[0.3, 0.9, 0.6]], dtype=np.float32)
        out = batch_aggregate(mat, method="max")
        assert out[0] == pytest.approx(0.9, abs=1e-5)
