"""Тесты для puzzle_reconstruction/utils/score_aggregator.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.score_aggregator import (
    AggregationMethod,
    ScoreVector,
    AggregationResult,
    weighted_sum,
    harmonic_mean,
    geometric_mean,
    aggregate_pair,
    aggregate_matrix,
    top_k_pairs,
    batch_aggregate,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_sv(idx_a=0, idx_b=1, channels=None, weights=None):
    if channels is None:
        channels = {"color": 0.6, "shape": 0.8}
    return ScoreVector(idx_a=idx_a, idx_b=idx_b, channels=channels,
                       weights=weights or {})


def make_result(scores=None, method="weighted", mean=0.5):
    if scores is None:
        scores = {(0, 1): 0.8, (0, 2): 0.6, (1, 2): 0.4}
    n = len(scores)
    top_pair = max(scores, key=scores.__getitem__) if scores else None
    return AggregationResult(scores=scores, method=method, n_pairs=n,
                              mean=mean, top_pair=top_pair)


# ─── AggregationMethod ────────────────────────────────────────────────────────

class TestAggregationMethod:
    def test_values_exist(self):
        assert AggregationMethod.WEIGHTED.value == "weighted"
        assert AggregationMethod.HARMONIC.value == "harmonic"
        assert AggregationMethod.GEOMETRIC.value == "geometric"
        assert AggregationMethod.MIN.value == "min"
        assert AggregationMethod.MAX.value == "max"

    def test_is_string_enum(self):
        assert isinstance(AggregationMethod.WEIGHTED, str)


# ─── ScoreVector ──────────────────────────────────────────────────────────────

class TestScoreVector:
    def test_creation(self):
        sv = make_sv()
        assert sv.idx_a == 0
        assert sv.idx_b == 1
        assert "color" in sv.channels

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(0, 1, channels={"c": 1.5})

    def test_negative_channel_value_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(0, 1, channels={"c": -0.1})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(0, 1, channels={"c": 0.5}, weights={"c": -1.0})

    def test_n_channels_property(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.7, "c": 0.5})
        assert sv.n_channels == 3

    def test_pair_key_canonical(self):
        sv = ScoreVector(3, 1, channels={"c": 0.5})
        assert sv.pair_key == (1, 3)

    def test_pair_key_already_ordered(self):
        sv = ScoreVector(1, 3, channels={"c": 0.5})
        assert sv.pair_key == (1, 3)

    def test_mean_score(self):
        sv = make_sv(channels={"a": 0.4, "b": 0.6})
        assert sv.mean_score == pytest.approx(0.5)

    def test_max_score(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.9})
        assert sv.max_score == pytest.approx(0.9)

    def test_min_score(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.9})
        assert sv.min_score == pytest.approx(0.3)

    def test_empty_channels_mean_zero(self):
        sv = ScoreVector(0, 1, channels={})
        assert sv.mean_score == pytest.approx(0.0)
        assert sv.max_score == pytest.approx(0.0)
        assert sv.min_score == pytest.approx(0.0)


# ─── AggregationResult ────────────────────────────────────────────────────────

class TestAggregationResult:
    def test_creation(self):
        r = make_result()
        assert r.n_pairs == 3
        assert 0.0 <= r.mean <= 1.0
        assert r.top_pair is not None

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError, match="n_pairs"):
            AggregationResult(scores={}, method="weighted",
                               n_pairs=-1, mean=0.5, top_pair=None)

    def test_mean_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mean"):
            AggregationResult(scores={}, method="weighted",
                               n_pairs=0, mean=1.5, top_pair=None)

    def test_get_score_found(self):
        r = make_result()
        assert r.get_score(0, 1) == pytest.approx(0.8)

    def test_get_score_canonical_form(self):
        r = make_result()
        assert r.get_score(1, 0) == pytest.approx(0.8)  # reversed pair

    def test_get_score_not_found_returns_none(self):
        r = make_result()
        assert r.get_score(5, 6) is None

    def test_empty_scores_top_pair_none(self):
        r = AggregationResult(scores={}, method="weighted",
                               n_pairs=0, mean=0.0, top_pair=None)
        assert r.top_pair is None


# ─── weighted_sum ─────────────────────────────────────────────────────────────

class TestWeightedSum:
    def test_returns_float(self):
        result = weighted_sum({"a": 0.5, "b": 0.7})
        assert isinstance(result, float)

    def test_equal_weights_is_mean(self):
        result = weighted_sum({"a": 0.4, "b": 0.6})
        assert result == pytest.approx(0.5)

    def test_custom_weights(self):
        result = weighted_sum({"a": 1.0, "b": 0.0}, weights={"a": 1.0, "b": 0.0})
        # Only channel 'a' contributes
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            weighted_sum({})

    def test_in_0_1(self):
        result = weighted_sum({"a": 0.8, "b": 0.3, "c": 0.6})
        assert 0.0 <= result <= 1.0


# ─── harmonic_mean ────────────────────────────────────────────────────────────

class TestHarmonicMean:
    def test_returns_float(self):
        result = harmonic_mean({"a": 0.5, "b": 0.5})
        assert isinstance(result, float)

    def test_equal_values(self):
        result = harmonic_mean({"a": 0.6, "b": 0.6})
        assert result == pytest.approx(0.6, abs=1e-4)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            harmonic_mean({})

    def test_zero_value_handled(self):
        result = harmonic_mean({"a": 0.0, "b": 0.5})
        assert 0.0 <= result <= 1.0

    def test_harmonic_leq_arithmetic(self):
        channels = {"a": 0.2, "b": 0.8}
        h = harmonic_mean(channels)
        a = (0.2 + 0.8) / 2
        assert h <= a + 1e-6


# ─── geometric_mean ───────────────────────────────────────────────────────────

class TestGeometricMean:
    def test_returns_float(self):
        result = geometric_mean({"a": 0.5, "b": 0.5})
        assert isinstance(result, float)

    def test_equal_values(self):
        result = geometric_mean({"a": 0.7, "b": 0.7})
        assert result == pytest.approx(0.7, abs=1e-4)

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            geometric_mean({})

    def test_zero_value_handled(self):
        result = geometric_mean({"a": 0.0, "b": 0.5})
        assert 0.0 <= result <= 1.0

    def test_in_0_1(self):
        result = geometric_mean({"a": 0.3, "b": 0.9, "c": 0.5})
        assert 0.0 <= result <= 1.0


# ─── aggregate_pair ───────────────────────────────────────────────────────────

class TestAggregatePair:
    def test_weighted_method(self):
        sv = make_sv(channels={"a": 0.4, "b": 0.6})
        result = aggregate_pair(sv, AggregationMethod.WEIGHTED)
        assert result == pytest.approx(0.5)

    def test_harmonic_method(self):
        sv = make_sv(channels={"a": 0.5, "b": 0.5})
        result = aggregate_pair(sv, AggregationMethod.HARMONIC)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_geometric_method(self):
        sv = make_sv(channels={"a": 0.5, "b": 0.5})
        result = aggregate_pair(sv, AggregationMethod.GEOMETRIC)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_min_method(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.8})
        result = aggregate_pair(sv, AggregationMethod.MIN)
        assert result == pytest.approx(0.3)

    def test_max_method(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.8})
        result = aggregate_pair(sv, AggregationMethod.MAX)
        assert result == pytest.approx(0.8)

    def test_empty_channels_raises(self):
        sv = ScoreVector(0, 1, channels={})
        with pytest.raises(ValueError):
            aggregate_pair(sv)

    def test_returns_float(self):
        sv = make_sv()
        result = aggregate_pair(sv)
        assert isinstance(result, float)

    def test_in_0_1(self):
        sv = make_sv(channels={"a": 0.3, "b": 0.7})
        for m in AggregationMethod:
            result = aggregate_pair(sv, m)
            assert 0.0 <= result <= 1.0


# ─── aggregate_matrix ─────────────────────────────────────────────────────────

class TestAggregateMatrix:
    def test_shape(self):
        vectors = [make_sv(0, 1), make_sv(0, 2), make_sv(1, 2)]
        result = aggregate_matrix(vectors, n_fragments=3)
        assert result.shape == (3, 3)

    def test_symmetric(self):
        vectors = [make_sv(0, 1), make_sv(0, 2)]
        result = aggregate_matrix(vectors, n_fragments=3)
        np.testing.assert_allclose(result, result.T, atol=1e-6)

    def test_dtype_float32(self):
        vectors = [make_sv()]
        result = aggregate_matrix(vectors, n_fragments=3)
        assert result.dtype == np.float32

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            aggregate_matrix([], n_fragments=0)

    def test_empty_vectors_zero_matrix(self):
        result = aggregate_matrix([], n_fragments=3)
        np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=np.float32))

    def test_out_of_bounds_index_skipped(self):
        sv = ScoreVector(0, 10, channels={"c": 0.5})
        result = aggregate_matrix([sv], n_fragments=3)
        assert result.max() == pytest.approx(0.0)


# ─── top_k_pairs ──────────────────────────────────────────────────────────────

class TestTopKPairs:
    def test_returns_list(self):
        r = make_result()
        assert isinstance(top_k_pairs(r, k=2), list)

    def test_length_k(self):
        r = make_result()
        result = top_k_pairs(r, k=2)
        assert len(result) == 2

    def test_sorted_descending(self):
        r = make_result()
        result = top_k_pairs(r, k=3)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_zero_raises(self):
        r = make_result()
        with pytest.raises(ValueError):
            top_k_pairs(r, k=0)

    def test_k_exceeds_n_pairs(self):
        r = make_result()  # 3 pairs
        result = top_k_pairs(r, k=100)
        assert len(result) == 3


# ─── batch_aggregate ──────────────────────────────────────────────────────────

class TestBatchAggregate:
    def test_returns_aggregation_result(self):
        vectors = [make_sv(0, 1), make_sv(1, 2)]
        result = batch_aggregate(vectors)
        assert isinstance(result, AggregationResult)

    def test_n_pairs_correct(self):
        vectors = [make_sv(0, 1), make_sv(1, 2), make_sv(0, 2)]
        result = batch_aggregate(vectors)
        assert result.n_pairs == 3

    def test_empty_vectors_zero_mean(self):
        result = batch_aggregate([])
        assert result.mean == pytest.approx(0.0)
        assert result.n_pairs == 0
        assert result.top_pair is None

    def test_deduplication_keeps_best(self):
        sv_low = ScoreVector(0, 1, channels={"c": 0.3})
        sv_high = ScoreVector(0, 1, channels={"c": 0.9})
        result = batch_aggregate([sv_low, sv_high])
        assert result.n_pairs == 1
        assert result.scores[(0, 1)] == pytest.approx(0.9)

    def test_method_stored(self):
        result = batch_aggregate([make_sv()], method=AggregationMethod.HARMONIC)
        assert result.method == "harmonic"

    def test_mean_in_0_1(self):
        vectors = [make_sv(0, 1), make_sv(1, 2)]
        result = batch_aggregate(vectors)
        assert 0.0 <= result.mean <= 1.0
