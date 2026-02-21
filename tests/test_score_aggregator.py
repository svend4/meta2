"""Тесты для puzzle_reconstruction/algorithms/score_aggregator.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _scores(**kw):
    """Словарь оценок ∈ [0,1]."""
    return {k: float(v) for k, v in kw.items()}


def _make_result(score=0.7, method="weighted_avg"):
    return AggregationResult(
        score=score,
        scores={"a": 0.8, "b": 0.6},
        weights={"a": 1.0, "b": 1.0},
        method=method,
    )


# ─── AggregationResult ────────────────────────────────────────────────────────

class TestAggregationResult:
    def test_fields(self):
        r = _make_result()
        assert r.score == pytest.approx(0.7)
        assert r.scores == {"a": 0.8, "b": 0.6}
        assert r.weights == {"a": 1.0, "b": 1.0}
        assert r.method == "weighted_avg"

    def test_params_default_empty(self):
        r = _make_result()
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = AggregationResult(score=0.5, scores={}, weights={},
                               method="min", params={"n_channels": 3})
        assert r.params["n_channels"] == 3

    def test_repr_contains_class(self):
        assert "AggregationResult" in repr(_make_result())

    def test_repr_contains_score(self):
        r = _make_result(score=0.1234)
        s = repr(r)
        assert "0.12" in s or "score" in s.lower()

    def test_repr_contains_method(self):
        r = _make_result(method="harmonic")
        assert "harmonic" in repr(r)

    def test_scores_dict(self):
        r = _make_result()
        assert isinstance(r.scores, dict)

    def test_weights_dict(self):
        r = _make_result()
        assert isinstance(r.weights, dict)

    def test_score_float(self):
        r = _make_result(score=0.95)
        assert isinstance(r.score, float)


# ─── weighted_avg ─────────────────────────────────────────────────────────────

class TestWeightedAvg:
    def test_returns_float(self):
        assert isinstance(weighted_avg(_scores(a=0.5, b=0.7)), float)

    def test_equal_weights_mean(self):
        v = weighted_avg(_scores(a=0.4, b=0.6))
        assert v == pytest.approx(0.5, abs=1e-6)

    def test_custom_weights(self):
        v = weighted_avg({"a": 0.0, "b": 1.0}, weights={"a": 0.0, "b": 1.0})
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_single_channel(self):
        v = weighted_avg({"x": 0.75})
        assert v == pytest.approx(0.75, abs=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_avg({})

    def test_zero_weight_sum_raises(self):
        with pytest.raises(ValueError):
            weighted_avg({"a": 0.5}, weights={"a": 0.0})

    def test_clips_to_0_1_high(self):
        v = weighted_avg({"a": 1.5})
        assert v <= 1.0

    def test_clips_to_0_1_low(self):
        v = weighted_avg({"a": -0.5})
        assert v >= 0.0

    def test_three_channels_equal(self):
        v = weighted_avg(_scores(a=0.2, b=0.4, c=0.9))
        assert v == pytest.approx((0.2 + 0.4 + 0.9) / 3.0, abs=1e-6)

    def test_missing_key_in_weights_uses_default(self):
        # Вес по умолчанию = 1.0 для ключей, отсутствующих в weights
        v = weighted_avg({"a": 0.6, "b": 0.6}, weights={"a": 1.0})
        assert v == pytest.approx(0.6, abs=1e-5)

    def test_order_independent(self):
        s1 = {"a": 0.3, "b": 0.7}
        s2 = {"b": 0.7, "a": 0.3}
        assert weighted_avg(s1) == pytest.approx(weighted_avg(s2), abs=1e-9)

    def test_all_one(self):
        v = weighted_avg(_scores(a=1.0, b=1.0, c=1.0))
        assert v == pytest.approx(1.0, abs=1e-6)


# ─── harmonic_mean ────────────────────────────────────────────────────────────

class TestHarmonicMean:
    def test_returns_float(self):
        assert isinstance(harmonic_mean(_scores(a=0.5, b=0.8)), float)

    def test_zero_channel_returns_zero(self):
        assert harmonic_mean({"a": 0.5, "b": 0.0}) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            harmonic_mean({})

    def test_single_channel(self):
        v = harmonic_mean({"x": 0.75})
        assert v == pytest.approx(0.75, abs=1e-6)

    def test_all_equal(self):
        v = harmonic_mean(_scores(a=0.5, b=0.5, c=0.5))
        assert v == pytest.approx(0.5, abs=1e-5)

    def test_in_range(self):
        v = harmonic_mean(_scores(a=0.2, b=0.8))
        assert 0.0 <= v <= 1.0

    def test_harmonic_leq_arithmetic(self):
        s = _scores(a=0.3, b=0.9)
        h = harmonic_mean(s)
        a = (0.3 + 0.9) / 2.0
        assert h <= a + 1e-9

    def test_penalizes_low_channels(self):
        # Один слабый канал тянет вниз сильнее, чем среднее
        h = harmonic_mean({"strong": 0.9, "weak": 0.1})
        avg = (0.9 + 0.1) / 2.0  # 0.5
        assert h < avg

    def test_all_one(self):
        assert harmonic_mean(_scores(a=1.0, b=1.0)) == pytest.approx(1.0, abs=1e-6)

    def test_near_zero_channel_dominates(self):
        h = harmonic_mean({"a": 0.99, "b": 0.001, "c": 0.99})
        assert h < 0.01


# ─── aggregate_scores ─────────────────────────────────────────────────────────

class TestAggregateScores:
    def test_returns_result(self):
        r = aggregate_scores(_scores(a=0.5, b=0.7))
        assert isinstance(r, AggregationResult)

    def test_score_in_range(self):
        r = aggregate_scores(_scores(a=0.3, b=0.9))
        assert 0.0 <= r.score <= 1.0

    def test_method_weighted_avg(self):
        r = aggregate_scores(_scores(a=0.6, b=0.4), method="weighted_avg")
        assert r.method == "weighted_avg"
        assert r.score == pytest.approx(0.5, abs=1e-5)

    def test_method_harmonic(self):
        r = aggregate_scores(_scores(a=0.5, b=0.5), method="harmonic")
        assert r.method == "harmonic"
        assert r.score == pytest.approx(0.5, abs=1e-5)

    def test_method_min(self):
        r = aggregate_scores(_scores(a=0.3, b=0.9), method="min")
        assert r.method == "min"
        assert r.score == pytest.approx(0.3, abs=1e-5)

    def test_method_max(self):
        r = aggregate_scores(_scores(a=0.3, b=0.9), method="max")
        assert r.method == "max"
        assert r.score == pytest.approx(0.9, abs=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores(_scores(a=0.5), method="super_fusion_xyz")

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores({})

    def test_params_n_channels(self):
        r = aggregate_scores(_scores(a=0.5, b=0.5, c=0.5))
        assert r.params.get("n_channels") == 3

    def test_scores_stored(self):
        s = _scores(color=0.7, texture=0.4)
        r = aggregate_scores(s)
        assert r.scores == s

    def test_weights_stored_when_provided(self):
        w = {"a": 2.0, "b": 1.0}
        r = aggregate_scores(_scores(a=0.5, b=0.5), weights=w)
        assert r.weights == w

    def test_weights_none_equal(self):
        r = aggregate_scores(_scores(a=0.6, b=0.6))
        for v in r.weights.values():
            assert v == pytest.approx(1.0)

    def test_custom_weights_affect_score(self):
        scores = {"a": 1.0, "b": 0.0}
        r1 = aggregate_scores(scores, weights={"a": 1.0, "b": 0.0}, method="weighted_avg")
        r2 = aggregate_scores(scores, weights={"a": 0.0, "b": 1.0}, method="weighted_avg")
        assert r1.score > r2.score


# ─── threshold_filter ─────────────────────────────────────────────────────────

class TestThresholdFilter:
    def _results(self, scores):
        return [_make_result(score=s) for s in scores]

    def test_returns_list(self):
        assert isinstance(threshold_filter(self._results([0.5])), list)

    def test_same_length(self):
        r = self._results([0.3, 0.7, 0.9])
        assert len(threshold_filter(r)) == 3

    def test_all_above_threshold(self):
        r = self._results([0.8, 0.9, 1.0])
        assert all(threshold_filter(r, threshold=0.5))

    def test_all_below_threshold(self):
        r = self._results([0.1, 0.2, 0.3])
        assert not any(threshold_filter(r, threshold=0.5))

    def test_empty_results(self):
        assert threshold_filter([]) == []

    def test_mixed(self):
        r = self._results([0.3, 0.8])
        mask = threshold_filter(r, threshold=0.5)
        assert mask == [False, True]

    def test_boundary_exclusive(self):
        r = self._results([0.5])
        # score > threshold (strict), 0.5 is NOT above 0.5
        assert threshold_filter(r, threshold=0.5) == [False]

    def test_all_bool(self):
        r = self._results([0.4, 0.6])
        for v in threshold_filter(r):
            assert isinstance(v, bool)


# ─── top_k_pairs ──────────────────────────────────────────────────────────────

class TestTopKPairs:
    def _make(self, scores):
        results = [_make_result(score=s) for s in scores]
        pairs   = [(i, i + 1) for i in range(len(scores))]
        return pairs, results

    def test_returns_list(self):
        pairs, results = self._make([0.5, 0.8])
        assert isinstance(top_k_pairs(pairs, results, k=1), list)

    def test_mismatched_len_raises(self):
        with pytest.raises(ValueError):
            top_k_pairs([(0, 1)], [_make_result(), _make_result()], k=1)

    def test_k_greater_than_n_clips(self):
        pairs, results = self._make([0.5, 0.8])
        r = top_k_pairs(pairs, results, k=100)
        assert len(r) == 2

    def test_k_zero_empty(self):
        pairs, results = self._make([0.5, 0.8])
        assert top_k_pairs(pairs, results, k=0) == []

    def test_sorted_desc(self):
        pairs, results = self._make([0.3, 0.9, 0.6])
        r = top_k_pairs(pairs, results, k=3)
        assert r[0] == (1, 2)   # score=0.9

    def test_single_pair(self):
        pairs, results = self._make([0.7])
        r = top_k_pairs(pairs, results, k=1)
        assert r == [(0, 1)]

    def test_top_1_is_best(self):
        scores = [0.1, 0.9, 0.5, 0.7]
        pairs, results = self._make(scores)
        r = top_k_pairs(pairs, results, k=1)
        # Пара с score=0.9 — пара (1,2)
        assert r[0] == (1, 2)

    def test_top_2_correct_order(self):
        pairs, results = self._make([0.2, 0.9, 0.7])
        r = top_k_pairs(pairs, results, k=2)
        assert r[0] == (1, 2)   # 0.9
        assert r[1] == (2, 3)   # 0.7


# ─── batch_aggregate ──────────────────────────────────────────────────────────

class TestBatchAggregate:
    def _matrix(self, n=4, m=3, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 1, (n, m)).astype(np.float32)

    def test_returns_ndarray(self):
        assert isinstance(batch_aggregate(self._matrix()), np.ndarray)

    def test_dtype_float32(self):
        assert batch_aggregate(self._matrix()).dtype == np.float32

    def test_shape_n(self):
        r = batch_aggregate(self._matrix(n=7, m=3))
        assert r.shape == (7,)

    def test_values_in_0_1(self):
        r = batch_aggregate(self._matrix())
        assert r.min() >= 0.0
        assert r.max() <= 1.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            batch_aggregate(np.array([0.1, 0.2]))

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            batch_aggregate(np.empty((4, 0)))

    @pytest.mark.parametrize("method", ["weighted_avg", "harmonic", "min", "max"])
    def test_all_methods(self, method):
        r = batch_aggregate(self._matrix(), method=method)
        assert r.shape == (4,)
        assert r.dtype == np.float32

    def test_weights_forwarded(self):
        m = self._matrix(n=2, m=2)
        r = batch_aggregate(m, weights=[1.0, 0.0])
        # Только первый канал → оценки из первого столбца
        np.testing.assert_allclose(r, m[:, 0], atol=1e-5)

    def test_channel_names_forwarded(self):
        m  = self._matrix(n=3, m=2)
        r  = batch_aggregate(m, channel_names=["color", "texture"])
        assert r.shape == (3,)

    def test_single_column(self):
        m = np.array([[0.4], [0.6], [0.8]], dtype=np.float32)
        r = batch_aggregate(m)
        np.testing.assert_allclose(r, m[:, 0], atol=1e-5)

    def test_all_ones_matrix(self):
        m = np.ones((5, 3), dtype=np.float32)
        r = batch_aggregate(m)
        np.testing.assert_allclose(r, np.ones(5, dtype=np.float32), atol=1e-5)
