"""Tests for puzzle_reconstruction/matching/score_normalizer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.score_normalizer import (
    ScoreNormResult,
    normalize_minmax,
    normalize_zscore,
    normalize_rank,
    calibrate_scores,
    combine_scores,
    normalize_score_matrix,
    batch_normalize_scores,
)


# ─── ScoreNormResult ──────────────────────────────────────────────────────────

class TestScoreNormResult:
    def test_basic_creation(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = ScoreNormResult(scores=scores, method="minmax",
                            original_min=0.1, original_max=0.9)
        assert r.method == "minmax"
        assert r.original_min == pytest.approx(0.1)
        assert r.original_max == pytest.approx(0.9)

    def test_params_default_empty(self):
        r = ScoreNormResult(scores=np.array([0.5]), method="rank",
                            original_min=0.5, original_max=0.5)
        assert r.params == {}

    def test_params_stored(self):
        r = ScoreNormResult(scores=np.array([0.5]), method="minmax",
                            original_min=0.0, original_max=1.0,
                            params={"feature_range": (0.0, 1.0)})
        assert r.params["feature_range"] == (0.0, 1.0)


# ─── normalize_minmax ─────────────────────────────────────────────────────────

class TestNormalizeMinmax:
    def test_returns_score_norm_result(self):
        r = normalize_minmax(np.array([1.0, 2.0, 3.0]))
        assert isinstance(r, ScoreNormResult)
        assert r.method == "minmax"

    def test_output_range_01(self):
        r = normalize_minmax(np.array([1.0, 3.0, 2.0]))
        assert float(r.scores.min()) == pytest.approx(0.0)
        assert float(r.scores.max()) == pytest.approx(1.0)

    def test_custom_feature_range(self):
        r = normalize_minmax(np.array([0.0, 5.0, 10.0]),
                             feature_range=(2.0, 4.0))
        assert float(r.scores.min()) == pytest.approx(2.0)
        assert float(r.scores.max()) == pytest.approx(4.0)

    def test_constant_array_returns_low(self):
        r = normalize_minmax(np.array([5.0, 5.0, 5.0]),
                             feature_range=(0.0, 1.0))
        assert np.all(r.scores == pytest.approx(0.0))

    def test_original_min_max_preserved(self):
        arr = np.array([2.0, 5.0, 8.0])
        r = normalize_minmax(arr)
        assert r.original_min == pytest.approx(2.0)
        assert r.original_max == pytest.approx(8.0)

    def test_output_length_matches_input(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = normalize_minmax(arr)
        assert len(r.scores) == len(arr)

    def test_dtype_float64(self):
        r = normalize_minmax(np.array([1, 2, 3]))
        assert r.scores.dtype == np.float64

    def test_monotone_preserved(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        r = normalize_minmax(arr)
        assert list(r.scores) == sorted(r.scores)


# ─── normalize_zscore ─────────────────────────────────────────────────────────

class TestNormalizeZscore:
    def test_returns_score_norm_result(self):
        r = normalize_zscore(np.array([1.0, 2.0, 3.0]))
        assert isinstance(r, ScoreNormResult)
        assert r.method == "zscore"

    def test_output_in_01(self):
        r = normalize_zscore(np.array([1.0, 2.0, 3.0, 10.0, -5.0]))
        assert float(r.scores.min()) >= 0.0
        assert float(r.scores.max()) <= 1.0

    def test_constant_array_returns_half(self):
        r = normalize_zscore(np.array([3.0, 3.0, 3.0]))
        assert np.all(r.scores == pytest.approx(0.5))

    def test_original_min_max_stored(self):
        arr = np.array([1.0, 5.0, 9.0])
        r = normalize_zscore(arr)
        assert r.original_min == pytest.approx(1.0)
        assert r.original_max == pytest.approx(9.0)

    def test_length_preserved(self):
        arr = np.arange(10, dtype=float)
        r = normalize_zscore(arr)
        assert len(r.scores) == len(arr)

    def test_clip_std_param_stored(self):
        r = normalize_zscore(np.array([1.0, 2.0, 3.0]), clip_std=2.0)
        assert r.params["clip_std"] == pytest.approx(2.0)


# ─── normalize_rank ───────────────────────────────────────────────────────────

class TestNormalizeRank:
    def test_returns_score_norm_result(self):
        r = normalize_rank(np.array([3.0, 1.0, 2.0]))
        assert isinstance(r, ScoreNormResult)
        assert r.method == "rank"

    def test_single_element_returns_zero(self):
        r = normalize_rank(np.array([5.0]))
        assert float(r.scores[0]) == pytest.approx(0.0)

    def test_sorted_order_reflected(self):
        arr = np.array([10.0, 30.0, 20.0])
        r = normalize_rank(arr)
        # index of smallest → 0.0, index of largest → 1.0
        order = np.argsort(arr)
        assert float(r.scores[order[0]]) == pytest.approx(0.0)
        assert float(r.scores[order[-1]]) == pytest.approx(1.0)

    def test_output_range_01(self):
        arr = np.array([5.0, 3.0, 8.0, 1.0, 9.0])
        r = normalize_rank(arr)
        assert float(r.scores.min()) >= 0.0
        assert float(r.scores.max()) <= 1.0

    def test_two_elements(self):
        r = normalize_rank(np.array([1.0, 2.0]))
        scores = sorted(r.scores)
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(1.0)

    def test_length_preserved(self):
        arr = np.linspace(0, 1, 10)
        r = normalize_rank(arr)
        assert len(r.scores) == 10


# ─── calibrate_scores ─────────────────────────────────────────────────────────

class TestCalibrateScores:
    def test_returns_score_norm_result(self):
        s   = np.array([0.1, 0.5, 0.9])
        ref = np.array([0.2, 0.6, 0.8])
        r = calibrate_scores(s, ref)
        assert isinstance(r, ScoreNormResult)
        assert r.method == "calibrated"

    def test_empty_reference_returns_unchanged(self):
        s = np.array([0.1, 0.5, 0.9])
        r = calibrate_scores(s, np.array([]))
        assert len(r.scores) == len(s)

    def test_output_length_matches_input(self):
        s   = np.linspace(0, 1, 20)
        ref = np.linspace(0, 1, 10)
        r = calibrate_scores(s, ref)
        assert len(r.scores) == 20

    def test_original_min_max_stored(self):
        s = np.array([0.2, 0.8])
        r = calibrate_scores(s, s)
        assert r.original_min == pytest.approx(0.2)
        assert r.original_max == pytest.approx(0.8)


# ─── combine_scores ───────────────────────────────────────────────────────────

class TestCombineScores:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            combine_scores([])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            combine_scores([np.array([0.5])], method="unknown")

    def test_length_mismatch_raises(self):
        a = np.array([0.5, 0.3])
        b = np.array([0.5])
        with pytest.raises(ValueError):
            combine_scores([a, b])

    def test_weighted_equal_weights(self):
        a = np.array([0.2, 0.4])
        b = np.array([0.6, 0.8])
        r = combine_scores([a, b])
        np.testing.assert_allclose(r, [0.4, 0.6])

    def test_weighted_custom_weights(self):
        a = np.array([0.0, 1.0])
        b = np.array([1.0, 0.0])
        r = combine_scores([a, b], weights=[0.0, 1.0])
        np.testing.assert_allclose(r, b)

    def test_weights_zero_sum_raises(self):
        with pytest.raises(ValueError):
            combine_scores([np.array([0.5])], weights=[0.0])

    def test_method_min(self):
        a = np.array([0.3, 0.7])
        b = np.array([0.5, 0.4])
        r = combine_scores([a, b], method="min")
        np.testing.assert_allclose(r, [0.3, 0.4])

    def test_method_max(self):
        a = np.array([0.3, 0.7])
        b = np.array([0.5, 0.4])
        r = combine_scores([a, b], method="max")
        np.testing.assert_allclose(r, [0.5, 0.7])

    def test_method_product(self):
        a = np.array([0.5, 0.4])
        b = np.array([0.5, 0.2])
        r = combine_scores([a, b], method="product")
        np.testing.assert_allclose(r, [0.25, 0.08])

    def test_output_float64(self):
        r = combine_scores([np.array([0.5, 0.3])])
        assert r.dtype == np.float64


# ─── normalize_score_matrix ───────────────────────────────────────────────────

class TestNormalizeScoreMatrix:
    def make_matrix(self, n=4):
        rng = np.random.default_rng(42)
        m = rng.random((n, n)).astype(np.float64)
        return m

    def test_non_square_raises(self):
        m = np.ones((3, 4), dtype=np.float64)
        with pytest.raises(ValueError):
            normalize_score_matrix(m)

    def test_unknown_method_raises(self):
        m = self.make_matrix()
        with pytest.raises(ValueError):
            normalize_score_matrix(m, method="unknown")

    def test_diagonal_preserved_by_default(self):
        m = self.make_matrix()
        diag_before = np.diag(m).copy()
        result = normalize_score_matrix(m, method="minmax", keep_diagonal=True)
        np.testing.assert_allclose(np.diag(result), diag_before)

    def test_shape_preserved(self):
        m = self.make_matrix(5)
        result = normalize_score_matrix(m)
        assert result.shape == (5, 5)

    def test_off_diagonal_in_range_minmax(self):
        m = self.make_matrix()
        result = normalize_score_matrix(m, method="minmax")
        mask = ~np.eye(m.shape[0], dtype=bool)
        assert float(result[mask].min()) >= 0.0
        assert float(result[mask].max()) <= 1.0

    def test_method_zscore(self):
        m = self.make_matrix()
        result = normalize_score_matrix(m, method="zscore")
        assert result.shape == m.shape

    def test_method_rank(self):
        m = self.make_matrix()
        result = normalize_score_matrix(m, method="rank")
        assert result.shape == m.shape

    def test_output_dtype_float64(self):
        m = self.make_matrix()
        result = normalize_score_matrix(m)
        assert result.dtype == np.float64


# ─── batch_normalize_scores ───────────────────────────────────────────────────

class TestBatchNormalizeScores:
    def test_returns_list(self):
        arrays = [np.array([0.1, 0.5, 0.9]) for _ in range(3)]
        results = batch_normalize_scores(arrays)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_score_norm_results(self):
        arrays = [np.array([0.1, 0.5, 0.9]) for _ in range(3)]
        results = batch_normalize_scores(arrays)
        assert all(isinstance(r, ScoreNormResult) for r in results)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_normalize_scores([np.array([0.5])], method="unknown")

    def test_empty_list(self):
        results = batch_normalize_scores([])
        assert results == []

    def test_method_zscore(self):
        arrays = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])]
        results = batch_normalize_scores(arrays, method="zscore")
        assert all(r.method == "zscore" for r in results)

    def test_method_rank(self):
        arrays = [np.array([3.0, 1.0, 2.0])]
        results = batch_normalize_scores(arrays, method="rank")
        assert results[0].method == "rank"

    def test_each_normalized_independently(self):
        """Two arrays normalized independently, not jointly."""
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([10.0, 20.0, 30.0])
        r1, r2 = batch_normalize_scores([a1, a2], method="minmax")
        # Each should span [0, 1]
        assert float(r1.scores.max()) == pytest.approx(1.0)
        assert float(r2.scores.max()) == pytest.approx(1.0)
