"""
Тесты для puzzle_reconstruction.matching.score_normalizer.
"""
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


# ─── Вспомогательные данные ───────────────────────────────────────────────────

def _ramp(n: int = 20) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _random(n: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)


def _uniform(n: int = 10, val: float = 0.5) -> np.ndarray:
    return np.full(n, val)


# ─── ScoreNormResult ──────────────────────────────────────────────────────────

class TestScoreNormResult:
    def test_fields(self):
        r = ScoreNormResult(
            scores=np.array([0.0, 0.5, 1.0]),
            method="minmax",
            original_min=0.0,
            original_max=1.0,
        )
        assert r.method == "minmax"
        assert r.original_min == pytest.approx(0.0)
        assert r.original_max == pytest.approx(1.0)
        assert len(r.scores) == 3

    def test_default_params_empty(self):
        r = ScoreNormResult(np.zeros(3), "test", 0.0, 1.0)
        assert r.params == {}

    def test_repr_contains_method(self):
        r = ScoreNormResult(np.array([0.5]), "minmax", 0.0, 1.0)
        assert "minmax" in repr(r)

    def test_repr_contains_n(self):
        r = ScoreNormResult(np.arange(7, dtype=float), "rank", 0.0, 6.0)
        assert "7" in repr(r)


# ─── normalize_minmax ─────────────────────────────────────────────────────────

class TestNormalizeMinmax:
    def test_returns_score_norm_result(self):
        assert isinstance(normalize_minmax(_ramp()), ScoreNormResult)

    def test_method_name(self):
        assert normalize_minmax(_ramp()).method == "minmax"

    def test_min_zero_max_one(self):
        r = normalize_minmax(_ramp())
        assert r.scores.min() == pytest.approx(0.0)
        assert r.scores.max() == pytest.approx(1.0)

    def test_custom_range(self):
        r = normalize_minmax(_ramp(), feature_range=(-1.0, 1.0))
        assert r.scores.min() == pytest.approx(-1.0)
        assert r.scores.max() == pytest.approx(1.0)

    def test_flat_array_returns_low(self):
        r = normalize_minmax(_uniform(val=3.7), feature_range=(0.0, 1.0))
        assert np.all(r.scores == pytest.approx(0.0))

    def test_original_min_max_stored(self):
        a = np.array([2.0, 5.0, 8.0])
        r = normalize_minmax(a)
        assert r.original_min == pytest.approx(2.0)
        assert r.original_max == pytest.approx(8.0)

    def test_single_element(self):
        r = normalize_minmax(np.array([42.0]))
        assert r.scores[0] == pytest.approx(0.0)

    def test_preserves_order(self):
        a = _random()
        r = normalize_minmax(a)
        assert np.all(np.diff(np.argsort(a)) == np.diff(np.argsort(r.scores)))

    def test_output_dtype_float64(self):
        r = normalize_minmax(_ramp().astype(np.float32))
        assert r.scores.dtype == np.float64

    def test_negative_values(self):
        a = np.array([-5.0, 0.0, 5.0])
        r = normalize_minmax(a)
        assert r.scores[0] == pytest.approx(0.0)
        assert r.scores[-1] == pytest.approx(1.0)


# ─── normalize_zscore ─────────────────────────────────────────────────────────

class TestNormalizeZscore:
    def test_returns_score_norm_result(self):
        assert isinstance(normalize_zscore(_random()), ScoreNormResult)

    def test_method_name(self):
        assert normalize_zscore(_random()).method == "zscore"

    def test_output_in_zero_one(self):
        r = normalize_zscore(_random())
        assert r.scores.min() >= -1e-9
        assert r.scores.max() <= 1.0 + 1e-9

    def test_flat_returns_half(self):
        r = normalize_zscore(_uniform())
        assert np.all(r.scores == pytest.approx(0.5))

    def test_clip_std_affects_result(self):
        a = np.array([0.0, 0.0, 0.0, 100.0])  # outlier
        r1 = normalize_zscore(a, clip_std=1.0)
        r2 = normalize_zscore(a, clip_std=3.0)
        # Outlier at 1σ clip → normalized to exactly 1.0
        assert r1.scores[-1] == pytest.approx(1.0)
        # At 3σ clip, outlier (~1.73σ) not clipped → still the max but < 1.0
        assert r2.scores[-1] == max(r2.scores)
        assert r2.scores[-1] <= 1.0

    def test_original_min_max_stored(self):
        a = np.array([1.0, 2.0, 3.0])
        r = normalize_zscore(a)
        assert r.original_min == pytest.approx(1.0)
        assert r.original_max == pytest.approx(3.0)

    def test_symmetric_data_median_near_half(self):
        a = np.array([-1.0, 0.0, 1.0])
        r = normalize_zscore(a)
        assert r.scores[1] == pytest.approx(0.5)

    def test_single_element_half(self):
        r = normalize_zscore(np.array([5.0]))
        assert r.scores[0] == pytest.approx(0.5)


# ─── normalize_rank ───────────────────────────────────────────────────────────

class TestNormalizeRank:
    def test_returns_score_norm_result(self):
        assert isinstance(normalize_rank(_ramp()), ScoreNormResult)

    def test_method_name(self):
        assert normalize_rank(_ramp()).method == "rank"

    def test_min_zero_max_one(self):
        r = normalize_rank(_random())
        assert r.scores.min() == pytest.approx(0.0)
        assert r.scores.max() == pytest.approx(1.0)

    def test_preserves_order(self):
        a = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
        r = normalize_rank(a)
        assert np.all(np.argsort(a) == np.argsort(r.scores))

    def test_n_one_returns_zero(self):
        r = normalize_rank(np.array([7.0]))
        assert r.scores[0] == pytest.approx(0.0)

    def test_uniform_array(self):
        r = normalize_rank(_uniform())
        # Все ранги равны
        assert np.all(r.scores == r.scores[0])

    def test_dtype_float64(self):
        r = normalize_rank(_ramp().astype(np.float32))
        assert r.scores.dtype == np.float64

    def test_two_elements(self):
        r = normalize_rank(np.array([10.0, 20.0]))
        assert r.scores[0] == pytest.approx(0.0)
        assert r.scores[1] == pytest.approx(1.0)


# ─── calibrate_scores ─────────────────────────────────────────────────────────

class TestCalibrateScores:
    def test_returns_score_norm_result(self):
        r = calibrate_scores(_random(), _random(seed=7))
        assert isinstance(r, ScoreNormResult)

    def test_method_name(self):
        r = calibrate_scores(_random(), _random(seed=7))
        assert r.method == "calibrated"

    def test_same_distribution_preserved(self):
        a = _ramp(20)
        r = calibrate_scores(a, a)
        # Собственная калибровка не должна слишком сильно менять значения
        assert np.allclose(np.sort(r.scores), np.sort(a), atol=0.1)

    def test_empty_scores_no_error(self):
        r = calibrate_scores(np.empty(0), _random())
        assert isinstance(r, ScoreNormResult)

    def test_empty_reference_no_error(self):
        r = calibrate_scores(_random(), np.empty(0))
        assert isinstance(r, ScoreNormResult)

    def test_output_length_equals_input(self):
        n = 15
        r = calibrate_scores(np.random.rand(n), np.random.rand(30))
        assert len(r.scores) == n

    def test_calibrated_range_near_reference_range(self):
        # Источник [0,1], эталон [10,20]
        src = np.linspace(0.0, 1.0, 20)
        ref = np.linspace(10.0, 20.0, 50)
        r   = calibrate_scores(src, ref)
        assert r.scores.min() >= 9.0
        assert r.scores.max() <= 21.0


# ─── combine_scores ───────────────────────────────────────────────────────────

class TestCombineScores:
    def test_weighted_equal_weights(self):
        a = np.array([0.2, 0.4, 0.6])
        b = np.array([0.4, 0.6, 0.8])
        r = combine_scores([a, b])
        np.testing.assert_array_almost_equal(r, [0.3, 0.5, 0.7])

    def test_weighted_custom_weights(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        r = combine_scores([a, b], weights=[1.0, 3.0])
        np.testing.assert_array_almost_equal(r, [0.75, 0.75])

    def test_min_method(self):
        a = np.array([0.5, 0.8])
        b = np.array([0.3, 0.9])
        r = combine_scores([a, b], method="min")
        np.testing.assert_array_almost_equal(r, [0.3, 0.8])

    def test_max_method(self):
        a = np.array([0.5, 0.8])
        b = np.array([0.3, 0.9])
        r = combine_scores([a, b], method="max")
        np.testing.assert_array_almost_equal(r, [0.5, 0.9])

    def test_product_method(self):
        a = np.array([0.5, 0.4])
        b = np.array([0.5, 0.5])
        r = combine_scores([a, b], method="product")
        np.testing.assert_array_almost_equal(r, [0.25, 0.2])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            combine_scores([])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            combine_scores([np.array([1.0, 2.0]), np.array([1.0])])

    def test_zero_weight_sum_raises(self):
        with pytest.raises(ValueError):
            combine_scores([np.array([1.0])], weights=[0.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            combine_scores([np.array([1.0])], method="mean")

    def test_single_array_unchanged(self):
        a = np.array([0.1, 0.5, 0.9])
        r = combine_scores([a])
        np.testing.assert_array_almost_equal(r, a)

    def test_output_dtype_float64(self):
        r = combine_scores([np.array([0.5], dtype=np.float32)])
        assert r.dtype == np.float64


# ─── normalize_score_matrix ───────────────────────────────────────────────────

class TestNormalizeScoreMatrix:
    def _make_matrix(self, n=4, seed=42):
        rng = np.random.default_rng(seed)
        m   = rng.random((n, n)).astype(np.float64)
        np.fill_diagonal(m, 0.5)
        return m

    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_returns_ndarray(self, method):
        m = self._make_matrix()
        r = normalize_score_matrix(m, method=method)
        assert isinstance(r, np.ndarray)

    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_shape_preserved(self, method):
        m = self._make_matrix(5)
        r = normalize_score_matrix(m, method=method)
        assert r.shape == (5, 5)

    def test_diagonal_preserved(self):
        m = self._make_matrix()
        diag = np.diag(m).copy()
        r    = normalize_score_matrix(m, method="minmax", keep_diagonal=True)
        np.testing.assert_array_almost_equal(np.diag(r), diag)

    def test_off_diagonal_minmax_in_zero_one(self):
        m = self._make_matrix(5)
        r = normalize_score_matrix(m, "minmax")
        n = r.shape[0]
        off = r[~np.eye(n, dtype=bool)]
        assert off.min() >= -1e-9
        assert off.max() <= 1.0 + 1e-9

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            normalize_score_matrix(np.ones((3, 4)))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_score_matrix(self._make_matrix(), method="bad")

    def test_dtype_float64(self):
        m = self._make_matrix(3).astype(np.float32)
        r = normalize_score_matrix(m)
        assert r.dtype == np.float64


# ─── batch_normalize_scores ───────────────────────────────────────────────────

class TestBatchNormalizeScores:
    def test_length_preserved(self):
        arrays = [_random(n) for n in [5, 10, 15]]
        result = batch_normalize_scores(arrays)
        assert len(result) == 3

    def test_all_score_norm_results(self):
        arrays = [_random() for _ in range(4)]
        result = batch_normalize_scores(arrays)
        for r in result:
            assert isinstance(r, ScoreNormResult)

    def test_empty_list_returns_empty(self):
        assert batch_normalize_scores([]) == []

    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_method_applied(self, method):
        arrays = [_random(), _ramp()]
        result = batch_normalize_scores(arrays, method=method)
        for r in result:
            assert r.method == method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_normalize_scores([_random()], method="bad")

    def test_independent_normalization(self):
        # Первый массив [0..1], второй [10..20]
        a1 = np.linspace(0.0, 1.0, 10)
        a2 = np.linspace(10.0, 20.0, 10)
        r1, r2 = batch_normalize_scores([a1, a2], method="minmax")
        # Каждый нормализуется независимо
        assert r1.scores.min() == pytest.approx(0.0)
        assert r2.scores.min() == pytest.approx(0.0)
        assert r1.scores.max() == pytest.approx(1.0)
        assert r2.scores.max() == pytest.approx(1.0)

    def test_kwargs_forwarded_to_zscore(self):
        arrays = [_random()]
        result = batch_normalize_scores(arrays, method="zscore", clip_std=1.0)
        assert result[0].params.get("clip_std") == pytest.approx(1.0)
