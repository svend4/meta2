"""Extra tests for puzzle_reconstruction.algorithms.score_aggregator."""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.score_aggregator import (
    AggregationResult,
    aggregate_scores,
    batch_aggregate,
    harmonic_mean,
    threshold_filter,
    top_k_pairs,
    weighted_avg,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scores(**kv):
    return {k: float(v) for k, v in kv.items()}


def _result(score=0.5, method="weighted_avg"):
    return AggregationResult(
        score=score,
        scores={"x": score},
        weights={"x": 1.0},
        method=method,
    )


# ─── AggregationResult extras ─────────────────────────────────────────────────

class TestAggregationResultExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(_result()), str)

    def test_scores_empty_dict(self):
        r = AggregationResult(score=0.0, scores={}, weights={},
                              method="min")
        assert r.scores == {}

    def test_weights_empty_dict(self):
        r = AggregationResult(score=0.0, scores={}, weights={},
                              method="max")
        assert r.weights == {}

    def test_params_stored(self):
        r = AggregationResult(score=0.5, scores={}, weights={},
                              method="weighted_avg",
                              params={"custom_key": 42})
        assert r.params["custom_key"] == 42

    def test_score_one_valid(self):
        r = _result(score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_score_zero_valid(self):
        r = _result(score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_method_min_stored(self):
        r = _result(method="min")
        assert r.method == "min"

    def test_method_max_stored(self):
        r = _result(method="max")
        assert r.method == "max"


# ─── weighted_avg extras ──────────────────────────────────────────────────────

class TestWeightedAvgExtra:
    def test_three_equal_weight_mean(self):
        result = weighted_avg(_scores(a=0.2, b=0.5, c=0.8))
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_all_ones(self):
        result = weighted_avg(_scores(a=1.0, b=1.0, c=1.0))
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_all_zeros(self):
        result = weighted_avg(_scores(a=0.0, b=0.0))
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_result_float_type(self):
        result = weighted_avg(_scores(a=0.5))
        assert isinstance(result, float)

    def test_heavy_weight_on_zero(self):
        result = weighted_avg(_scores(a=0.0, b=1.0),
                              weights={"a": 100.0, "b": 1.0})
        assert result < 0.05

    def test_heavy_weight_on_one(self):
        result = weighted_avg(_scores(a=1.0, b=0.0),
                              weights={"a": 100.0, "b": 1.0})
        assert result > 0.95

    def test_five_channels_equal_weight(self):
        result = weighted_avg(_scores(a=0.0, b=0.25, c=0.5, d=0.75, e=1.0))
        assert result == pytest.approx(0.5, abs=1e-6)


# ─── harmonic_mean extras ─────────────────────────────────────────────────────

class TestHarmonicMeanExtra:
    def test_three_identical(self):
        result = harmonic_mean(_scores(a=0.6, b=0.6, c=0.6))
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_all_ones_returns_one(self):
        result = harmonic_mean(_scores(a=1.0, b=1.0))
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_single_value(self):
        result = harmonic_mean(_scores(x=0.4))
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_harmonic_le_arithmetic(self):
        scores = _scores(a=0.3, b=0.9)
        arithmetic = (0.3 + 0.9) / 2
        harmonic = harmonic_mean(scores)
        assert harmonic <= arithmetic + 1e-9

    def test_zero_dominates(self):
        result = harmonic_mean(_scores(a=1.0, b=0.0, c=0.8))
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(harmonic_mean(_scores(a=0.5, b=0.5)), float)

    def test_five_channels(self):
        result = harmonic_mean(_scores(a=1.0, b=1.0, c=1.0, d=1.0, e=1.0))
        assert result == pytest.approx(1.0, abs=1e-6)


# ─── aggregate_scores extras ──────────────────────────────────────────────────

class TestAggregateScoresExtra:
    def test_min_single_channel(self):
        r = aggregate_scores(_scores(x=0.4), method="min")
        assert r.score == pytest.approx(0.4, abs=1e-6)

    def test_max_single_channel(self):
        r = aggregate_scores(_scores(x=0.7), method="max")
        assert r.score == pytest.approx(0.7, abs=1e-6)

    def test_harmonic_zero_channel(self):
        r = aggregate_scores(_scores(a=1.0, b=0.0), method="harmonic")
        assert r.score == pytest.approx(0.0, abs=1e-6)

    def test_weights_stored_in_result(self):
        s = _scores(a=0.5, b=0.5)
        w = {"a": 3.0, "b": 1.0}
        r = aggregate_scores(s, weights=w, method="weighted_avg")
        assert r.weights["a"] == pytest.approx(3.0)

    def test_scores_preserved_three_channels(self):
        s = _scores(x=0.1, y=0.5, z=0.9)
        r = aggregate_scores(s)
        assert set(r.scores.keys()) == {"x", "y", "z"}

    def test_min_three_channels(self):
        r = aggregate_scores(_scores(a=0.2, b=0.5, c=0.8), method="min")
        assert r.score == pytest.approx(0.2, abs=1e-6)

    def test_max_three_channels(self):
        r = aggregate_scores(_scores(a=0.2, b=0.5, c=0.8), method="max")
        assert r.score == pytest.approx(0.8, abs=1e-6)

    def test_method_stored_harmonic(self):
        r = aggregate_scores(_scores(a=0.5), method="harmonic")
        assert r.method == "harmonic"

    def test_method_stored_min(self):
        r = aggregate_scores(_scores(a=0.5), method="min")
        assert r.method == "min"


# ─── threshold_filter extras ──────────────────────────────────────────────────

class TestThresholdFilterExtra:
    def test_threshold_zero_all_pass(self):
        results = [_result(s) for s in (0.0, 0.5, 1.0)]
        mask = threshold_filter(results, threshold=0.0)
        # score > 0.0: 0.5 and 1.0 pass, 0.0 does not
        assert mask[1] is True
        assert mask[2] is True

    def test_threshold_one_none_pass(self):
        results = [_result(0.9), _result(1.0)]
        mask = threshold_filter(results, threshold=1.0)
        assert all(m is False for m in mask)

    def test_single_passing(self):
        assert threshold_filter([_result(0.8)], threshold=0.5) == [True]

    def test_single_failing(self):
        assert threshold_filter([_result(0.2)], threshold=0.5) == [False]

    def test_length_preserved(self):
        results = [_result(0.6)] * 7
        assert len(threshold_filter(results, threshold=0.5)) == 7

    def test_all_above_threshold(self):
        results = [_result(0.9) for _ in range(5)]
        mask = threshold_filter(results, threshold=0.5)
        assert all(m is True for m in mask)

    def test_all_below_threshold(self):
        results = [_result(0.1) for _ in range(4)]
        mask = threshold_filter(results, threshold=0.5)
        assert all(m is False for m in mask)


# ─── top_k_pairs extras ───────────────────────────────────────────────────────

class TestTopKPairsExtra:
    def test_k_one_returns_best(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        results = [_result(0.3), _result(0.9), _result(0.6)]
        top = top_k_pairs(pairs, results, k=1)
        assert len(top) == 1
        assert top[0] == (1, 2)

    def test_single_pair_k_one(self):
        top = top_k_pairs([(0, 1)], [_result(0.5)], k=1)
        assert len(top) == 1

    def test_all_same_score_k_all(self):
        pairs = [(i, i + 1) for i in range(5)]
        results = [_result(0.5) for _ in range(5)]
        top = top_k_pairs(pairs, results, k=5)
        assert len(top) == 5

    def test_k_equals_n_returns_all(self):
        pairs = [(0, 1), (1, 2)]
        results = [_result(0.5), _result(0.8)]
        top = top_k_pairs(pairs, results, k=2)
        assert len(top) == 2

    def test_five_pairs_top_three_descending(self):
        scores_vals = [0.1, 0.5, 0.3, 0.9, 0.7]
        pairs = [(i, i + 1) for i in range(5)]
        results = [_result(s) for s in scores_vals]
        top = top_k_pairs(pairs, results, k=3)
        assert len(top) == 3

    def test_returns_list_of_pairs(self):
        pairs = [(0, 1), (2, 3)]
        results = [_result(0.4), _result(0.6)]
        top = top_k_pairs(pairs, results, k=2)
        assert all(isinstance(p, tuple) for p in top)


# ─── batch_aggregate extras ───────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def test_single_row_single_channel(self):
        mat = np.array([[0.7]], dtype=np.float32)
        out = batch_aggregate(mat)
        assert len(out) == 1
        assert out[0] == pytest.approx(0.7, abs=1e-5)

    def test_large_matrix_20x5(self):
        mat = np.random.default_rng(0).random((20, 5)).astype(np.float32)
        out = batch_aggregate(mat)
        assert len(out) == 20
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_one_channel_output_equals_input(self):
        vals = np.array([0.2, 0.5, 0.8], dtype=np.float32).reshape(3, 1)
        out = batch_aggregate(vals)
        np.testing.assert_allclose(out, [0.2, 0.5, 0.8], atol=1e-5)

    def test_min_method_row(self):
        mat = np.array([[0.1, 0.6, 0.4]], dtype=np.float32)
        out = batch_aggregate(mat, method="min")
        assert out[0] == pytest.approx(0.1, abs=1e-5)

    def test_max_method_row(self):
        mat = np.array([[0.1, 0.6, 0.4]], dtype=np.float32)
        out = batch_aggregate(mat, method="max")
        assert out[0] == pytest.approx(0.6, abs=1e-5)

    def test_harmonic_all_ones(self):
        mat = np.ones((3, 4), dtype=np.float32)
        out = batch_aggregate(mat, method="harmonic")
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0], atol=1e-5)

    def test_weighted_avg_custom_weights(self):
        # Single row: [1.0, 0.0], weights [9, 1] → 0.9
        mat = np.array([[1.0, 0.0]], dtype=np.float32)
        out = batch_aggregate(mat, weights=[9.0, 1.0])
        assert out[0] == pytest.approx(0.9, abs=1e-5)

    def test_output_dtype_float32(self):
        mat = np.random.rand(4, 3).astype(np.float32)
        out = batch_aggregate(mat)
        assert out.dtype == np.float32
