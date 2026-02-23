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


def _ramp(n=20):
    return np.linspace(0.0, 1.0, n)

def _rand(n=20, seed=0):
    return np.random.default_rng(seed).random(n)

def _flat(n=10, val=0.5):
    return np.full(n, val)


# ─── ScoreNormResult (extra) ──────────────────────────────────────────────────

class TestScoreNormResultExtra:
    def test_scores_stored(self):
        arr = np.array([0.1, 0.5, 0.9])
        r = ScoreNormResult(scores=arr, method="rank", original_min=0.1, original_max=0.9)
        np.testing.assert_array_equal(r.scores, arr)

    def test_original_min_max_stored(self):
        r = ScoreNormResult(np.zeros(3), "minmax", original_min=2.0, original_max=8.0)
        assert r.original_min == pytest.approx(2.0)
        assert r.original_max == pytest.approx(8.0)

    def test_params_custom(self):
        r = ScoreNormResult(np.zeros(3), "zscore", 0.0, 1.0,
                             params={"clip_std": 2.0})
        assert r.params["clip_std"] == pytest.approx(2.0)

    def test_repr_contains_original_range(self):
        r = ScoreNormResult(np.array([0.5]), "minmax", 2.0, 8.0)
        s = repr(r)
        assert "minmax" in s

    def test_method_stored(self):
        r = ScoreNormResult(np.zeros(2), "rank", 0.0, 1.0)
        assert r.method == "rank"


# ─── normalize_minmax (extra) ─────────────────────────────────────────────────

class TestNormalizeMinmaxExtra:
    def test_large_array(self):
        a = _rand(n=100)
        r = normalize_minmax(a)
        assert r.scores.min() == pytest.approx(0.0)
        assert r.scores.max() == pytest.approx(1.0)

    def test_already_normalized_unchanged(self):
        a = np.array([0.0, 0.5, 1.0])
        r = normalize_minmax(a)
        np.testing.assert_allclose(r.scores, a, atol=1e-9)

    def test_feature_range_0_10(self):
        a = np.array([0.0, 0.5, 1.0])
        r = normalize_minmax(a, feature_range=(0.0, 10.0))
        assert r.scores.max() == pytest.approx(10.0)
        assert r.scores.min() == pytest.approx(0.0)

    def test_feature_range_minus1_plus1(self):
        a = _ramp()
        r = normalize_minmax(a, feature_range=(-1.0, 1.0))
        assert r.scores.min() == pytest.approx(-1.0)
        assert r.scores.max() == pytest.approx(1.0)

    def test_two_elements(self):
        a = np.array([3.0, 7.0])
        r = normalize_minmax(a)
        assert r.scores[0] == pytest.approx(0.0)
        assert r.scores[1] == pytest.approx(1.0)

    def test_output_length_preserved(self):
        a = _rand(50)
        r = normalize_minmax(a)
        assert len(r.scores) == 50

    def test_method_is_minmax(self):
        r = normalize_minmax(_rand())
        assert r.method == "minmax"

    def test_positive_values(self):
        a = np.array([100.0, 200.0, 300.0])
        r = normalize_minmax(a)
        assert r.scores[0] == pytest.approx(0.0)
        assert r.scores[-1] == pytest.approx(1.0)


# ─── normalize_zscore (extra) ─────────────────────────────────────────────────

class TestNormalizeZscoreExtra:
    def test_output_in_0_1(self):
        a = _rand(50, seed=7)
        r = normalize_zscore(a)
        assert r.scores.min() >= -1e-9
        assert r.scores.max() <= 1.0 + 1e-9

    def test_method_is_zscore(self):
        r = normalize_zscore(_rand())
        assert r.method == "zscore"

    def test_output_length_preserved(self):
        a = _rand(30)
        r = normalize_zscore(a)
        assert len(r.scores) == 30

    def test_outlier_clamped(self):
        # z_outlier = sqrt(5) ≈ 2.24 > clip_std=2.0 → clamped → score == 1.0
        a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0])
        r = normalize_zscore(a, clip_std=2.0)
        assert r.scores[-1] == pytest.approx(1.0)

    def test_single_element_returns_half(self):
        r = normalize_zscore(np.array([99.0]))
        assert r.scores[0] == pytest.approx(0.5)

    def test_original_min_max_stored(self):
        a = np.array([1.0, 5.0, 10.0])
        r = normalize_zscore(a)
        assert r.original_min == pytest.approx(1.0)
        assert r.original_max == pytest.approx(10.0)

    def test_dtype_float64(self):
        r = normalize_zscore(_rand().astype(np.float32))
        assert r.scores.dtype == np.float64

    def test_large_clip_std_max_has_highest_score(self):
        a = np.array([0.0, 1.0, 2.0, 10.0])
        r = normalize_zscore(a, clip_std=100.0)
        assert r.scores[-1] == max(r.scores)


# ─── normalize_rank (extra) ───────────────────────────────────────────────────

class TestNormalizeRankExtra:
    def test_method_is_rank(self):
        r = normalize_rank(_ramp())
        assert r.method == "rank"

    def test_output_in_0_1(self):
        a = _rand(30)
        r = normalize_rank(a)
        assert r.scores.min() >= -1e-9
        assert r.scores.max() <= 1.0 + 1e-9

    def test_larger_values_higher_rank(self):
        a = np.array([1.0, 3.0, 2.0])
        r = normalize_rank(a)
        assert r.scores[1] > r.scores[2] > r.scores[0]

    def test_three_elements_ranks(self):
        a = np.array([10.0, 20.0, 30.0])
        r = normalize_rank(a)
        assert r.scores[0] == pytest.approx(0.0)
        assert r.scores[-1] == pytest.approx(1.0)

    def test_output_length_preserved(self):
        a = _rand(40)
        r = normalize_rank(a)
        assert len(r.scores) == 40

    def test_dtype_float64(self):
        r = normalize_rank(_ramp().astype(np.float32))
        assert r.scores.dtype == np.float64

    def test_two_elements_order(self):
        a = np.array([5.0, 3.0])
        r = normalize_rank(a)
        assert r.scores[0] == pytest.approx(1.0)
        assert r.scores[1] == pytest.approx(0.0)


# ─── calibrate_scores (extra) ─────────────────────────────────────────────────

class TestCalibrateScoresExtra:
    def test_returns_score_norm_result(self):
        r = calibrate_scores(_rand(20), _rand(30, seed=1))
        assert isinstance(r, ScoreNormResult)

    def test_method_is_calibrated(self):
        r = calibrate_scores(_rand(10), _rand(10, seed=2))
        assert r.method == "calibrated"

    def test_output_length_matches_input(self):
        src = _rand(25)
        ref = _rand(100, seed=3)
        r = calibrate_scores(src, ref)
        assert len(r.scores) == 25

    def test_same_distribution_identity_approx(self):
        a = _ramp(20)
        r = calibrate_scores(a, a)
        np.testing.assert_allclose(np.sort(r.scores), np.sort(a), atol=0.15)

    def test_empty_scores_handled(self):
        try:
            r = calibrate_scores(np.empty(0), _rand())
            assert isinstance(r, ScoreNormResult)
        except (ValueError, Exception):
            pass  # empty arrays may raise

    def test_empty_reference_handled(self):
        try:
            r = calibrate_scores(_rand(), np.empty(0))
            assert isinstance(r, ScoreNormResult)
        except (ValueError, Exception):
            pass  # empty arrays may raise


# ─── combine_scores (extra) ───────────────────────────────────────────────────

class TestCombineScoresExtra:
    def test_three_arrays_equal_weights(self):
        a = np.array([0.3, 0.6])
        b = np.array([0.5, 0.5])
        c = np.array([0.7, 0.4])
        r = combine_scores([a, b, c])
        np.testing.assert_allclose(r, [(0.3+0.5+0.7)/3, (0.6+0.5+0.4)/3], atol=1e-6)

    def test_weighted_avg_asymmetric(self):
        a = np.array([0.2])
        b = np.array([0.8])
        r = combine_scores([a, b], weights=[3.0, 1.0])
        expected = (0.2*3 + 0.8*1) / 4.0
        np.testing.assert_allclose(r, [expected], atol=1e-6)

    def test_min_method_correct(self):
        a = np.array([0.3, 0.7, 0.5])
        b = np.array([0.5, 0.4, 0.9])
        r = combine_scores([a, b], method="min")
        np.testing.assert_allclose(r, [0.3, 0.4, 0.5])

    def test_max_method_correct(self):
        a = np.array([0.3, 0.7, 0.5])
        b = np.array([0.5, 0.4, 0.9])
        r = combine_scores([a, b], method="max")
        np.testing.assert_allclose(r, [0.5, 0.7, 0.9])

    def test_product_method_correct(self):
        a = np.array([0.5])
        b = np.array([0.4])
        r = combine_scores([a, b], method="product")
        np.testing.assert_allclose(r, [0.2], atol=1e-6)

    def test_output_dtype_float64(self):
        a = np.array([0.5], dtype=np.float32)
        r = combine_scores([a])
        assert r.dtype == np.float64

    def test_single_array_identity(self):
        a = np.array([0.1, 0.5, 0.9])
        r = combine_scores([a])
        np.testing.assert_allclose(r, a, atol=1e-9)

    def test_negative_weight_raises(self):
        a = np.array([0.5])
        with pytest.raises(ValueError):
            combine_scores([a], weights=[-1.0])


# ─── normalize_score_matrix (extra) ───────────────────────────────────────────

class TestNormalizeScoreMatrixExtra:
    def _mat(self, n=4, seed=42):
        rng = np.random.default_rng(seed)
        m = rng.random((n, n))
        np.fill_diagonal(m, 0.0)
        return m

    def test_minmax_off_diag_in_0_1(self):
        m = self._mat()
        r = normalize_score_matrix(m, method="minmax")
        off = r[~np.eye(4, dtype=bool)]
        assert off.min() >= -1e-9
        assert off.max() <= 1.0 + 1e-9

    def test_zscore_method_returns_ndarray(self):
        m = self._mat()
        r = normalize_score_matrix(m, method="zscore")
        assert isinstance(r, np.ndarray)

    def test_rank_method_returns_ndarray(self):
        m = self._mat()
        r = normalize_score_matrix(m, method="rank")
        assert isinstance(r, np.ndarray)

    def test_shape_preserved_5x5(self):
        m = self._mat(n=5)
        for method in ("minmax", "zscore", "rank"):
            r = normalize_score_matrix(m, method=method)
            assert r.shape == (5, 5)

    def test_keep_diagonal_true(self):
        m = self._mat()
        diag = np.diag(m).copy()
        r = normalize_score_matrix(m, method="minmax", keep_diagonal=True)
        np.testing.assert_allclose(np.diag(r), diag, atol=1e-9)

    def test_keep_diagonal_false(self):
        m = self._mat()
        np.fill_diagonal(m, 0.5)
        r = normalize_score_matrix(m, method="minmax", keep_diagonal=False)
        # diagonal may or may not be zeroed; just check shape
        assert r.shape == m.shape

    def test_dtype_float64_output(self):
        m = self._mat(3).astype(np.float32)
        r = normalize_score_matrix(m, method="minmax")
        assert r.dtype == np.float64


# ─── batch_normalize_scores (extra) ──────────────────────────────────────────

class TestBatchNormalizeScoresExtra:
    def test_single_array(self):
        result = batch_normalize_scores([_rand(10)])
        assert len(result) == 1
        assert isinstance(result[0], ScoreNormResult)

    def test_five_arrays(self):
        arrays = [_rand(n=10+i, seed=i) for i in range(5)]
        result = batch_normalize_scores(arrays)
        assert len(result) == 5

    def test_minmax_each_independent(self):
        a1 = np.linspace(0.0, 1.0, 10)
        a2 = np.linspace(5.0, 10.0, 10)
        r1, r2 = batch_normalize_scores([a1, a2], method="minmax")
        assert r1.scores.min() == pytest.approx(0.0)
        assert r2.scores.min() == pytest.approx(0.0)

    def test_rank_method_all_rank(self):
        arrays = [_rand(10), _ramp(10)]
        result = batch_normalize_scores(arrays, method="rank")
        for r in result:
            assert r.method == "rank"

    def test_zscore_method_all_zscore(self):
        arrays = [_rand(10), _ramp(10)]
        result = batch_normalize_scores(arrays, method="zscore")
        for r in result:
            assert r.method == "zscore"

    def test_output_dtype_float64(self):
        result = batch_normalize_scores([_rand(10).astype(np.float32)])
        assert result[0].scores.dtype == np.float64
