"""Extra tests for puzzle_reconstruction/algorithms/score_aggregator.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.score_aggregator import (
    AggregationResult,
    weighted_avg,
    harmonic_mean,
    aggregate_scores,
    threshold_filter,
    top_k_pairs,
    batch_aggregate,
)


def _s(**kw):
    return {k: float(v) for k, v in kw.items()}


def _result(score=0.7, method="weighted_avg"):
    return AggregationResult(
        score=score,
        scores={"a": 0.8, "b": 0.6},
        weights={"a": 1.0, "b": 1.0},
        method=method,
    )


# ─── AggregationResult (extra) ────────────────────────────────────────────────

class TestAggregationResultExtra:
    def test_score_zero_valid(self):
        r = AggregationResult(score=0.0, scores={}, weights={}, method="min")
        assert r.score == pytest.approx(0.0)

    def test_score_one_valid(self):
        r = AggregationResult(score=1.0, scores={}, weights={}, method="max")
        assert r.score == pytest.approx(1.0)

    def test_scores_stored(self):
        r = _result()
        assert "a" in r.scores and "b" in r.scores

    def test_weights_stored(self):
        r = _result()
        assert r.weights["a"] == pytest.approx(1.0)

    def test_method_name_stored(self):
        for m in ("weighted_avg", "harmonic", "min", "max"):
            r = AggregationResult(score=0.5, scores={}, weights={}, method=m)
            assert r.method == m

    def test_params_stored(self):
        r = AggregationResult(score=0.5, scores={}, weights={},
                               method="min", params={"k": 3})
        assert r.params["k"] == 3

    def test_repr_not_empty(self):
        r = _result()
        assert len(repr(r)) > 0


# ─── weighted_avg (extra) ─────────────────────────────────────────────────────

class TestWeightedAvgExtra:
    def test_all_zero_returns_zero(self):
        v = weighted_avg(_s(a=0.0, b=0.0))
        assert v == pytest.approx(0.0)

    def test_high_weight_dominates(self):
        v = weighted_avg({"a": 1.0, "b": 0.0}, weights={"a": 100.0, "b": 1.0})
        assert v > 0.98

    def test_symmetric_keys(self):
        v1 = weighted_avg({"a": 0.3, "b": 0.7})
        v2 = weighted_avg({"b": 0.7, "a": 0.3})
        assert v1 == pytest.approx(v2)

    def test_three_equal_channels(self):
        v = weighted_avg(_s(a=0.5, b=0.5, c=0.5))
        assert v == pytest.approx(0.5, abs=1e-6)

    def test_result_type_float(self):
        v = weighted_avg(_s(x=0.7))
        assert isinstance(v, float)

    def test_weights_all_equal_same_as_mean(self):
        scores = {"a": 0.2, "b": 0.4, "c": 0.6}
        expected = (0.2 + 0.4 + 0.6) / 3
        v = weighted_avg(scores, weights={"a": 1.0, "b": 1.0, "c": 1.0})
        assert v == pytest.approx(expected, abs=1e-6)

    def test_clip_high_value(self):
        v = weighted_avg({"a": 2.0})
        assert v <= 1.0

    def test_clip_low_value(self):
        v = weighted_avg({"a": -1.0})
        assert v >= 0.0


# ─── harmonic_mean (extra) ────────────────────────────────────────────────────

class TestHarmonicMeanExtra:
    def test_all_zeros_returns_zero(self):
        assert harmonic_mean(_s(a=0.0, b=0.0)) == pytest.approx(0.0)

    def test_all_ones_returns_one(self):
        assert harmonic_mean(_s(a=1.0, b=1.0, c=1.0)) == pytest.approx(1.0, abs=1e-6)

    def test_three_equal_channels(self):
        v = harmonic_mean(_s(a=0.8, b=0.8, c=0.8))
        assert v == pytest.approx(0.8, abs=1e-5)

    def test_result_leq_arithmetic_mean(self):
        s = _s(a=0.2, b=0.9, c=0.6)
        h = harmonic_mean(s)
        a = sum(s.values()) / len(s)
        assert h <= a + 1e-9

    def test_single_channel_returns_value(self):
        v = harmonic_mean({"only": 0.42})
        assert v == pytest.approx(0.42, abs=1e-6)

    def test_very_small_channel_low_result(self):
        h = harmonic_mean({"a": 0.999, "b": 0.001})
        assert h < 0.01

    def test_result_nonneg(self):
        h = harmonic_mean(_s(a=0.4, b=0.6))
        assert h >= 0.0


# ─── aggregate_scores (extra) ────────────────────────────────────────────────

class TestAggregateScoresExtra:
    def test_weighted_avg_three_channels(self):
        r = aggregate_scores(_s(a=0.3, b=0.6, c=0.9), method="weighted_avg")
        assert r.score == pytest.approx(0.6, abs=1e-5)

    def test_harmonic_two_equal(self):
        r = aggregate_scores(_s(a=0.5, b=0.5), method="harmonic")
        assert r.score == pytest.approx(0.5, abs=1e-5)

    def test_min_returns_minimum(self):
        r = aggregate_scores(_s(a=0.1, b=0.9), method="min")
        assert r.score == pytest.approx(0.1, abs=1e-5)

    def test_max_returns_maximum(self):
        r = aggregate_scores(_s(a=0.1, b=0.9), method="max")
        assert r.score == pytest.approx(0.9, abs=1e-5)

    def test_n_channels_in_params(self):
        r = aggregate_scores(_s(a=0.5, b=0.6, c=0.7))
        assert r.params.get("n_channels") == 3

    def test_scores_stored_in_result(self):
        s = _s(x=0.3, y=0.7)
        r = aggregate_scores(s)
        assert r.scores == s

    def test_custom_weights_stored(self):
        w = {"a": 2.0, "b": 0.5}
        r = aggregate_scores(_s(a=0.5, b=0.5), weights=w)
        assert r.weights == w

    def test_result_is_aggregation_result(self):
        r = aggregate_scores(_s(a=0.5))
        assert isinstance(r, AggregationResult)

    def test_score_in_0_1(self):
        r = aggregate_scores(_s(a=0.3, b=0.8, c=0.5))
        assert 0.0 <= r.score <= 1.0


# ─── threshold_filter (extra) ────────────────────────────────────────────────

class TestThresholdFilterExtra:
    def _results(self, scores):
        return [_result(s) for s in scores]

    def test_threshold_zero_all_above(self):
        r = self._results([0.1, 0.2, 0.3])
        mask = threshold_filter(r, threshold=0.0)
        # Strict >0, so anything >0 passes
        assert all(mask)

    def test_threshold_one_nothing_above(self):
        r = self._results([0.5, 0.9, 1.0])
        mask = threshold_filter(r, threshold=1.0)
        assert not any(mask)

    def test_single_result_above(self):
        r = self._results([0.8])
        assert threshold_filter(r, threshold=0.5) == [True]

    def test_single_result_below(self):
        r = self._results([0.3])
        assert threshold_filter(r, threshold=0.5) == [False]

    def test_all_at_boundary(self):
        r = self._results([0.5, 0.5])
        mask = threshold_filter(r, threshold=0.5)
        # Strict comparison: 0.5 > 0.5 is False
        assert mask == [False, False]

    def test_result_is_list_of_bools(self):
        r = self._results([0.4, 0.6])
        mask = threshold_filter(r)
        for v in mask:
            assert isinstance(v, bool)

    def test_five_results_mixed(self):
        r = self._results([0.1, 0.4, 0.6, 0.8, 0.9])
        mask = threshold_filter(r, threshold=0.5)
        assert mask == [False, False, True, True, True]


# ─── top_k_pairs (extra) ─────────────────────────────────────────────────────

class TestTopKPairsExtra:
    def _make(self, scores):
        results = [_result(s) for s in scores]
        pairs = [(i, i + 1) for i in range(len(scores))]
        return pairs, results

    def test_k_1_best_pair(self):
        pairs, results = self._make([0.2, 0.9, 0.5])
        r = top_k_pairs(pairs, results, k=1)
        assert len(r) == 1
        assert r[0] == (1, 2)

    def test_k_all_returns_all(self):
        pairs, results = self._make([0.1, 0.3, 0.7])
        r = top_k_pairs(pairs, results, k=3)
        assert len(r) == 3

    def test_sorted_descending(self):
        pairs, results = self._make([0.4, 0.9, 0.6, 0.1])
        r = top_k_pairs(pairs, results, k=4)
        # Check order: 0.9, 0.6, 0.4, 0.1
        assert r[0] == (1, 2)
        assert r[1] == (2, 3)

    def test_returns_correct_type(self):
        pairs, results = self._make([0.5])
        r = top_k_pairs(pairs, results, k=1)
        assert isinstance(r[0], tuple)
        assert len(r[0]) == 2

    def test_empty_input(self):
        r = top_k_pairs([], [], k=5)
        assert r == []

    def test_k_zero_returns_empty(self):
        pairs, results = self._make([0.5, 0.8])
        r = top_k_pairs(pairs, results, k=0)
        assert r == []


# ─── batch_aggregate (extra) ─────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def _mat(self, n=4, m=3, seed=0):
        return np.random.default_rng(seed).uniform(0, 1, (n, m)).astype(np.float32)

    def test_shape_n_from_rows(self):
        r = batch_aggregate(self._mat(n=7, m=3))
        assert r.shape == (7,)

    def test_dtype_float32(self):
        r = batch_aggregate(self._mat())
        assert r.dtype == np.float32

    def test_values_in_0_1(self):
        r = batch_aggregate(self._mat())
        assert r.min() >= 0.0
        assert r.max() <= 1.0

    def test_all_ones_returns_ones(self):
        m = np.ones((5, 3), dtype=np.float32)
        r = batch_aggregate(m)
        np.testing.assert_allclose(r, np.ones(5, dtype=np.float32), atol=1e-5)

    def test_all_zeros_returns_zeros(self):
        m = np.zeros((4, 2), dtype=np.float32)
        r = batch_aggregate(m)
        np.testing.assert_allclose(r, np.zeros(4, dtype=np.float32), atol=1e-5)

    def test_single_column_identity(self):
        m = np.array([[0.3], [0.7], [0.5]], dtype=np.float32)
        r = batch_aggregate(m)
        np.testing.assert_allclose(r, m[:, 0], atol=1e-5)

    def test_harmonic_method(self):
        m = self._mat()
        r = batch_aggregate(m, method="harmonic")
        assert r.shape == (4,)
        assert r.dtype == np.float32

    def test_min_method(self):
        m = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        r = batch_aggregate(m, method="min")
        np.testing.assert_allclose(r, [0.1, 0.2], atol=1e-5)

    def test_max_method(self):
        m = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        r = batch_aggregate(m, method="max")
        np.testing.assert_allclose(r, [0.9, 0.8], atol=1e-5)

    def test_weights_zero_second_col_ignored(self):
        m = np.array([[0.4, 0.9], [0.6, 0.1]], dtype=np.float32)
        r = batch_aggregate(m, weights=[1.0, 0.0])
        np.testing.assert_allclose(r, m[:, 0], atol=1e-5)

    def test_large_matrix(self):
        m = self._mat(n=100, m=10, seed=7)
        r = batch_aggregate(m)
        assert r.shape == (100,)
