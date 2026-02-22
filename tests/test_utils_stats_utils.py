"""Расширенные тесты для puzzle_reconstruction/utils/stats_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.stats_utils import (
    StatsConfig,
    batch_describe,
    describe,
    iqr,
    outlier_mask,
    percentile_rank,
    running_stats,
    weighted_mean,
    weighted_std,
    winsorize,
    zscore_array,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _arr(n: int = 10) -> np.ndarray:
    return np.arange(1.0, n + 1.0)


def _const(n: int = 10, val: float = 5.0) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


# ─── TestStatsConfig ──────────────────────────────────────────────────────────

class TestStatsConfig:
    def test_defaults(self):
        c = StatsConfig()
        assert c.outlier_iqr_k == pytest.approx(1.5)
        assert c.winsor_low    == pytest.approx(0.05)
        assert c.winsor_high   == pytest.approx(0.95)
        assert c.ddof == 0

    def test_outlier_iqr_k_zero_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=0.0)

    def test_outlier_iqr_k_negative_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=-1.0)

    def test_winsor_low_ge_high_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.6, winsor_high=0.5)

    def test_winsor_equal_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.5, winsor_high=0.5)

    def test_winsor_low_negative_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=-0.1, winsor_high=0.9)

    def test_winsor_high_gt_1_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.05, winsor_high=1.1)

    def test_ddof_2_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(ddof=2)

    def test_ddof_1_ok(self):
        c = StatsConfig(ddof=1)
        assert c.ddof == 1

    def test_custom_values(self):
        c = StatsConfig(outlier_iqr_k=3.0, winsor_low=0.01, winsor_high=0.99, ddof=1)
        assert c.outlier_iqr_k == pytest.approx(3.0)
        assert c.winsor_low    == pytest.approx(0.01)
        assert c.winsor_high   == pytest.approx(0.99)
        assert c.ddof == 1


# ─── TestDescribe ─────────────────────────────────────────────────────────────

class TestDescribe:
    def test_returns_dict(self):
        assert isinstance(describe(_arr()), dict)

    def test_has_required_keys(self):
        d = describe(_arr())
        for key in ("min", "max", "mean", "std", "median", "q25", "q75", "iqr"):
            assert key in d

    def test_all_values_float(self):
        for v in describe(_arr()).values():
            assert isinstance(v, float)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            describe(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            describe(np.array([]))

    def test_min_le_max(self):
        d = describe(_arr())
        assert d["min"] <= d["max"]

    def test_mean_in_range(self):
        d = describe(_arr())
        assert d["min"] <= d["mean"] <= d["max"]

    def test_median_in_range(self):
        d = describe(_arr())
        assert d["min"] <= d["median"] <= d["max"]

    def test_iqr_nonneg(self):
        assert describe(_arr())["iqr"] >= 0.0

    def test_constant_std_zero(self):
        d = describe(_const())
        assert d["std"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_iqr_zero(self):
        d = describe(_const())
        assert d["iqr"] == pytest.approx(0.0, abs=1e-9)

    def test_ddof1_larger_std(self):
        cfg0 = StatsConfig(ddof=0)
        cfg1 = StatsConfig(ddof=1)
        a = _arr(20)
        assert describe(a, cfg1)["std"] >= describe(a, cfg0)["std"]

    def test_single_element(self):
        d = describe(np.array([42.0]))
        assert d["min"] == pytest.approx(42.0)
        assert d["max"] == pytest.approx(42.0)
        assert d["mean"] == pytest.approx(42.0)

    def test_q25_le_median_le_q75(self):
        d = describe(_arr(20))
        assert d["q25"] <= d["median"] <= d["q75"]


# ─── TestZscoreArray ──────────────────────────────────────────────────────────

class TestZscoreArray:
    def test_returns_ndarray(self):
        assert isinstance(zscore_array(_arr()), np.ndarray)

    def test_float64(self):
        assert zscore_array(_arr()).dtype == np.float64

    def test_same_length(self):
        a = _arr(20)
        assert len(zscore_array(a)) == 20

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.array([]))

    def test_constant_returns_zeros(self):
        result = zscore_array(_const(10))
        assert np.allclose(result, 0.0)

    def test_mean_near_zero(self):
        result = zscore_array(_arr(20))
        assert abs(result.mean()) < 1e-9

    def test_std_near_one(self):
        result = zscore_array(_arr(20))
        assert abs(result.std() - 1.0) < 1e-6

    def test_single_element_returns_zero(self):
        result = zscore_array(np.array([5.0]))
        assert result[0] == pytest.approx(0.0)

    def test_accepts_int_array(self):
        result = zscore_array(np.array([1, 2, 3, 4, 5]))
        assert result.dtype == np.float64


# ─── TestIqr ──────────────────────────────────────────────────────────────────

class TestIqr:
    def test_returns_float(self):
        assert isinstance(iqr(_arr()), float)

    def test_nonneg(self):
        assert iqr(_arr()) >= 0.0

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            iqr(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iqr(np.array([]))

    def test_constant_zero(self):
        assert iqr(_const(10)) == pytest.approx(0.0)

    def test_ramp_positive(self):
        assert iqr(_arr(20)) > 0.0

    def test_matches_describe(self):
        a = _arr(20)
        assert iqr(a) == pytest.approx(describe(a)["iqr"])


# ─── TestWinsorize ────────────────────────────────────────────────────────────

class TestWinsorize:
    def test_returns_ndarray(self):
        assert isinstance(winsorize(_arr()), np.ndarray)

    def test_float64(self):
        assert winsorize(_arr()).dtype == np.float64

    def test_same_length(self):
        a = _arr(20)
        assert len(winsorize(a)) == 20

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            winsorize(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            winsorize(np.array([]))

    def test_values_within_bounds(self):
        a = np.concatenate([np.array([-1000.0, 1000.0]), np.ones(18)])
        cfg = StatsConfig(winsor_low=0.1, winsor_high=0.9)
        result = winsorize(a, cfg)
        lo = np.percentile(a, 10)
        hi = np.percentile(a, 90)
        assert np.all(result >= lo - 1e-9)
        assert np.all(result <= hi + 1e-9)

    def test_constant_unchanged(self):
        a = _const(10, 3.0)
        result = winsorize(a)
        assert np.allclose(result, 3.0)

    def test_reduces_extreme_values(self):
        a = np.array([0.0] * 9 + [1000.0])
        cfg = StatsConfig(winsor_low=0.05, winsor_high=0.9)
        result = winsorize(a, cfg)
        assert result.max() < 1000.0


# ─── TestPercentileRank ───────────────────────────────────────────────────────

class TestPercentileRank:
    def test_returns_float(self):
        assert isinstance(percentile_rank(_arr(), 5.0), float)

    def test_in_0_100(self):
        val = percentile_rank(_arr(10), 5.0)
        assert 0.0 <= val <= 100.0

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(np.ones((5, 2)), 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(np.array([]), 1.0)

    def test_below_all_is_zero(self):
        a = _arr(10)  # [1..10]
        assert percentile_rank(a, 0.0) == pytest.approx(0.0)

    def test_above_all_is_100(self):
        a = _arr(10)  # [1..10]
        assert percentile_rank(a, 11.0) == pytest.approx(100.0)

    def test_midpoint(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        rank = percentile_rank(a, 5.5)
        assert rank == pytest.approx(50.0)

    def test_monotone_with_value(self):
        a = _arr(10)
        r1 = percentile_rank(a, 3.0)
        r2 = percentile_rank(a, 7.0)
        assert r1 < r2


# ─── TestOutlierMask ──────────────────────────────────────────────────────────

class TestOutlierMask:
    def test_returns_ndarray(self):
        result = outlier_mask(_arr())
        assert isinstance(result, np.ndarray)

    def test_bool_dtype(self):
        assert outlier_mask(_arr()).dtype == bool

    def test_same_length(self):
        a = _arr(20)
        assert len(outlier_mask(a)) == 20

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            outlier_mask(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            outlier_mask(np.array([]))

    def test_constant_all_false(self):
        result = outlier_mask(_const(20))
        assert not np.any(result)

    def test_extreme_outlier_detected(self):
        a = np.concatenate([np.ones(18), np.array([1000.0, -1000.0])])
        result = outlier_mask(a)
        assert result[-1] or result[-2]

    def test_high_k_fewer_outliers(self):
        a = np.concatenate([np.ones(18), np.array([100.0, 200.0])])
        cfg_low  = StatsConfig(outlier_iqr_k=0.5)
        cfg_high = StatsConfig(outlier_iqr_k=10.0)
        assert np.sum(outlier_mask(a, cfg_high)) <= np.sum(outlier_mask(a, cfg_low))


# ─── TestRunningStats ─────────────────────────────────────────────────────────

class TestRunningStats:
    def test_returns_dict(self):
        assert isinstance(running_stats(_arr()), dict)

    def test_has_keys(self):
        d = running_stats(_arr())
        for key in ("cumsum", "cummax", "cummin", "cummean"):
            assert key in d

    def test_all_ndarrays(self):
        for v in running_stats(_arr()).values():
            assert isinstance(v, np.ndarray)

    def test_same_length(self):
        a = _arr(15)
        for v in running_stats(a).values():
            assert len(v) == 15

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            running_stats(np.ones((5, 2)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            running_stats(np.array([]))

    def test_cumsum_final_equals_sum(self):
        a = _arr(10)
        d = running_stats(a)
        assert d["cumsum"][-1] == pytest.approx(float(np.sum(a)))

    def test_cummax_nondecreasing(self):
        a = _arr(10)
        d = running_stats(a)
        assert np.all(np.diff(d["cummax"]) >= 0)

    def test_cummin_nonincreasing(self):
        a = _arr(10)[::-1].copy()  # descending
        d = running_stats(a)
        assert np.all(np.diff(d["cummin"]) <= 0)

    def test_cummean_final_near_mean(self):
        a = _arr(20)
        d = running_stats(a)
        assert d["cummean"][-1] == pytest.approx(float(np.mean(a)))

    def test_float64_output(self):
        d = running_stats(_arr())
        for v in d.values():
            assert v.dtype == np.float64


# ─── TestWeightedMean ─────────────────────────────────────────────────────────

class TestWeightedMean:
    def test_returns_float(self):
        a = _arr(5)
        w = np.ones(5)
        assert isinstance(weighted_mean(a, w), float)

    def test_equal_weights_matches_mean(self):
        a = _arr(10)
        w = np.ones(10)
        assert weighted_mean(a, w) == pytest.approx(float(np.mean(a)))

    def test_2d_arr_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.ones((5, 2)), np.ones(5))

    def test_2d_weights_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.ones(5), np.ones((5, 1)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(_arr(5), np.ones(3))

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(_arr(5), np.zeros(5))

    def test_weighted_toward_high(self):
        a = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 100.0])
        result = weighted_mean(a, w)
        assert result > 2.5

    def test_single_element(self):
        assert weighted_mean(np.array([7.0]), np.array([1.0])) == pytest.approx(7.0)


# ─── TestWeightedStd ──────────────────────────────────────────────────────────

class TestWeightedStd:
    def test_returns_float(self):
        assert isinstance(weighted_std(_arr(5), np.ones(5)), float)

    def test_nonneg(self):
        assert weighted_std(_arr(10), np.ones(10)) >= 0.0

    def test_constant_zero(self):
        a = _const(10, 3.0)
        w = np.ones(10)
        assert weighted_std(a, w) == pytest.approx(0.0, abs=1e-9)

    def test_equal_weights_matches_std(self):
        a = _arr(10)
        w = np.ones(10)
        assert weighted_std(a, w) == pytest.approx(float(np.std(a)), rel=1e-5)

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            weighted_std(_arr(5), np.zeros(5))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_std(_arr(5), np.ones(3))


# ─── TestBatchDescribe ────────────────────────────────────────────────────────

class TestBatchDescribe:
    def test_returns_list(self):
        assert isinstance(batch_describe([_arr(), _arr()]), list)

    def test_same_length(self):
        arrs = [_arr(5), _arr(10), _arr(15)]
        assert len(batch_describe(arrs)) == 3

    def test_each_is_dict(self):
        for d in batch_describe([_arr(), _const()]):
            assert isinstance(d, dict)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_describe([])

    def test_cfg_passed_through(self):
        cfg = StatsConfig(ddof=1)
        results = batch_describe([_arr(10)], cfg)
        assert len(results) == 1
        assert "std" in results[0]

    def test_single_element_list(self):
        result = batch_describe([_arr(5)])
        assert len(result) == 1
        assert isinstance(result[0], dict)
