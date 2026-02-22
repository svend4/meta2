"""Extra tests for puzzle_reconstruction.matching.score_normalizer."""
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


# ─── TestScoreNormResultExtra ────────────────────────────────────────────────

class TestScoreNormResultExtra:
    def test_scores_stored(self):
        s = np.array([0.1, 0.5])
        r = ScoreNormResult(scores=s, method="minmax",
                            original_min=0.1, original_max=0.5)
        np.testing.assert_array_equal(r.scores, s)

    def test_method_stored(self):
        r = ScoreNormResult(scores=np.array([1.0]), method="zscore",
                            original_min=1.0, original_max=1.0)
        assert r.method == "zscore"

    def test_original_min_max(self):
        r = ScoreNormResult(scores=np.array([0.0, 1.0]), method="rank",
                            original_min=-5.0, original_max=10.0)
        assert r.original_min == pytest.approx(-5.0)
        assert r.original_max == pytest.approx(10.0)

    def test_params_custom(self):
        r = ScoreNormResult(scores=np.array([0.5]), method="minmax",
                            original_min=0.0, original_max=1.0,
                            params={"clip": True})
        assert r.params["clip"] is True

    def test_scores_ndim(self):
        r = ScoreNormResult(scores=np.array([0.1, 0.9]), method="minmax",
                            original_min=0.1, original_max=0.9)
        assert r.scores.ndim == 1


# ─── TestNormalizeMinmaxExtra ────────────────────────────────────────────────

class TestNormalizeMinmaxExtra:
    def test_two_elements(self):
        r = normalize_minmax(np.array([3.0, 7.0]))
        assert float(r.scores[0]) == pytest.approx(0.0)
        assert float(r.scores[1]) == pytest.approx(1.0)

    def test_single_element(self):
        r = normalize_minmax(np.array([5.0]))
        assert float(r.scores[0]) == pytest.approx(0.0)

    def test_negative_values(self):
        r = normalize_minmax(np.array([-10.0, 0.0, 10.0]))
        assert float(r.scores.min()) == pytest.approx(0.0)
        assert float(r.scores.max()) == pytest.approx(1.0)

    def test_already_01(self):
        arr = np.array([0.0, 0.5, 1.0])
        r = normalize_minmax(arr)
        np.testing.assert_allclose(r.scores, arr)

    def test_reverse_order(self):
        r = normalize_minmax(np.array([10.0, 5.0, 0.0]))
        assert float(r.scores[0]) == pytest.approx(1.0)
        assert float(r.scores[2]) == pytest.approx(0.0)

    def test_feature_range_negative(self):
        r = normalize_minmax(np.array([0.0, 10.0]),
                             feature_range=(-1.0, 1.0))
        assert float(r.scores.min()) == pytest.approx(-1.0)
        assert float(r.scores.max()) == pytest.approx(1.0)

    def test_large_array(self):
        arr = np.arange(1000, dtype=float)
        r = normalize_minmax(arr)
        assert float(r.scores.min()) == pytest.approx(0.0)
        assert float(r.scores.max()) == pytest.approx(1.0)

    def test_original_min_max(self):
        r = normalize_minmax(np.array([3.0, 7.0, 5.0]))
        assert r.original_min == pytest.approx(3.0)
        assert r.original_max == pytest.approx(7.0)


# ─── TestNormalizeZscoreExtra ────────────────────────────────────────────────

class TestNormalizeZscoreExtra:
    def test_two_elements(self):
        r = normalize_zscore(np.array([0.0, 10.0]))
        assert len(r.scores) == 2

    def test_single_element(self):
        r = normalize_zscore(np.array([5.0]))
        assert float(r.scores[0]) == pytest.approx(0.5)

    def test_clip_std_applied(self):
        r = normalize_zscore(np.array([0.0, 5.0, 100.0]), clip_std=1.0)
        assert float(r.scores.min()) >= 0.0
        assert float(r.scores.max()) <= 1.0

    def test_method_is_zscore(self):
        r = normalize_zscore(np.array([1.0, 2.0, 3.0]))
        assert r.method == "zscore"

    def test_original_stored(self):
        r = normalize_zscore(np.array([-5.0, 5.0]))
        assert r.original_min == pytest.approx(-5.0)
        assert r.original_max == pytest.approx(5.0)

    def test_symmetric_distribution(self):
        arr = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        r = normalize_zscore(arr)
        assert float(r.scores[2]) == pytest.approx(0.5, abs=0.15)


# ─── TestNormalizeRankExtra ──────────────────────────────────────────────────

class TestNormalizeRankExtra:
    def test_three_elements(self):
        r = normalize_rank(np.array([10.0, 30.0, 20.0]))
        assert len(r.scores) == 3

    def test_all_same_values(self):
        r = normalize_rank(np.array([5.0, 5.0, 5.0]))
        assert len(r.scores) == 3

    def test_descending_input(self):
        r = normalize_rank(np.array([3.0, 2.0, 1.0]))
        assert float(r.scores[0]) == pytest.approx(1.0)
        assert float(r.scores[2]) == pytest.approx(0.0)

    def test_ascending_input(self):
        r = normalize_rank(np.array([1.0, 2.0, 3.0]))
        assert float(r.scores[0]) == pytest.approx(0.0)
        assert float(r.scores[2]) == pytest.approx(1.0)

    def test_method_is_rank(self):
        r = normalize_rank(np.array([1.0, 2.0]))
        assert r.method == "rank"

    def test_original_min_max(self):
        r = normalize_rank(np.array([3.0, 1.0, 7.0]))
        assert r.original_min == pytest.approx(1.0)
        assert r.original_max == pytest.approx(7.0)

    def test_large_array(self):
        arr = np.arange(100, dtype=float)
        r = normalize_rank(arr)
        assert float(r.scores.min()) == pytest.approx(0.0)
        assert float(r.scores.max()) == pytest.approx(1.0)


# ─── TestCalibrateScoresExtra ────────────────────────────────────────────────

class TestCalibrateScoresExtra:
    def test_method_is_calibrated(self):
        r = calibrate_scores(np.array([0.5]), np.array([0.3, 0.7]))
        assert r.method == "calibrated"

    def test_single_score(self):
        r = calibrate_scores(np.array([0.5]), np.array([0.1, 0.9]))
        assert len(r.scores) == 1

    def test_identical_ref(self):
        s = np.array([0.2, 0.5, 0.8])
        r = calibrate_scores(s, s)
        assert len(r.scores) == 3

    def test_scores_dtype(self):
        r = calibrate_scores(np.array([0.5, 0.8]),
                             np.array([0.1, 0.9]))
        assert r.scores.dtype in (np.float32, np.float64)

    def test_large_reference(self):
        s = np.array([0.5])
        ref = np.linspace(0, 1, 1000)
        r = calibrate_scores(s, ref)
        assert len(r.scores) == 1


# ─── TestCombineScoresExtra ──────────────────────────────────────────────────

class TestCombineScoresExtra:
    def test_single_array_returned(self):
        a = np.array([0.2, 0.8])
        r = combine_scores([a])
        np.testing.assert_allclose(r, a)

    def test_three_arrays(self):
        a = np.array([0.1, 0.5])
        b = np.array([0.3, 0.7])
        c = np.array([0.5, 0.9])
        r = combine_scores([a, b, c])
        assert len(r) == 2

    def test_weighted_sum_dominance(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        r = combine_scores([a, b], weights=[100.0, 1.0])
        assert r[0] > r[1]

    def test_product_zeros(self):
        a = np.array([0.0, 0.5])
        b = np.array([0.5, 0.5])
        r = combine_scores([a, b], method="product")
        assert float(r[0]) == pytest.approx(0.0)

    def test_min_selects_min(self):
        a = np.array([0.3, 0.9])
        b = np.array([0.7, 0.1])
        r = combine_scores([a, b], method="min")
        assert float(r[0]) == pytest.approx(0.3)
        assert float(r[1]) == pytest.approx(0.1)

    def test_max_selects_max(self):
        a = np.array([0.3, 0.9])
        b = np.array([0.7, 0.1])
        r = combine_scores([a, b], method="max")
        assert float(r[0]) == pytest.approx(0.7)
        assert float(r[1]) == pytest.approx(0.9)

    def test_output_length(self):
        a = np.array([0.1, 0.2, 0.3])
        r = combine_scores([a, a])
        assert len(r) == 3


# ─── TestNormalizeScoreMatrixExtra ──────────────────────────────────────────

class TestNormalizeScoreMatrixExtra:
    def _mat(self, n=4, seed=42):
        return np.random.default_rng(seed).random((n, n))

    def test_2x2_shape(self):
        m = self._mat(2)
        result = normalize_score_matrix(m)
        assert result.shape == (2, 2)

    def test_2x2(self):
        m = self._mat(2)
        result = normalize_score_matrix(m)
        assert result.shape == (2, 2)

    def test_keep_diagonal_false(self):
        m = self._mat(3)
        result = normalize_score_matrix(m, method="minmax",
                                        keep_diagonal=False)
        assert result.shape == (3, 3)

    def test_symmetric_input(self):
        m = self._mat(4)
        m = (m + m.T) / 2
        result = normalize_score_matrix(m)
        assert result.shape == (4, 4)

    def test_large_matrix(self):
        m = self._mat(20)
        result = normalize_score_matrix(m)
        assert result.shape == (20, 20)

    def test_rank_output_in_range(self):
        m = self._mat(5)
        result = normalize_score_matrix(m, method="rank")
        mask = ~np.eye(5, dtype=bool)
        assert float(result[mask].min()) >= -1e-9
        assert float(result[mask].max()) <= 1.0 + 1e-9

    def test_zscore_output_shape(self):
        m = self._mat(6)
        result = normalize_score_matrix(m, method="zscore")
        assert result.shape == (6, 6)


# ─── TestBatchNormalizeScoresExtra ──────────────────────────────────────────

class TestBatchNormalizeScoresExtra:
    def test_single_array(self):
        results = batch_normalize_scores([np.array([1.0, 2.0, 3.0])])
        assert len(results) == 1
        assert isinstance(results[0], ScoreNormResult)

    def test_method_minmax_default(self):
        results = batch_normalize_scores([np.array([1.0, 2.0])])
        assert results[0].method == "minmax"

    def test_each_independent_zscore(self):
        a1 = np.array([1.0, 2.0, 3.0])
        a2 = np.array([100.0, 200.0, 300.0])
        r1, r2 = batch_normalize_scores([a1, a2], method="zscore")
        assert r1.original_min == pytest.approx(1.0)
        assert r2.original_min == pytest.approx(100.0)

    def test_rank_method(self):
        results = batch_normalize_scores(
            [np.array([3.0, 1.0, 2.0])], method="rank")
        assert results[0].method == "rank"

    def test_length_preserved(self):
        arrays = [np.arange(5, dtype=float) for _ in range(7)]
        results = batch_normalize_scores(arrays)
        assert len(results) == 7

    def test_each_result_scores_length(self):
        a1 = np.array([1.0, 2.0])
        a2 = np.array([1.0, 2.0, 3.0, 4.0])
        r1, r2 = batch_normalize_scores([a1, a2])
        assert len(r1.scores) == 2
        assert len(r2.scores) == 4

    def test_minmax_each_spans_01(self):
        arrays = [np.array([float(i), float(i + 10)]) for i in range(5)]
        results = batch_normalize_scores(arrays, method="minmax")
        for r in results:
            assert float(r.scores.min()) == pytest.approx(0.0)
            assert float(r.scores.max()) == pytest.approx(1.0)
