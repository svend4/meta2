"""Extra tests for puzzle_reconstruction/utils/stats_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.stats_utils import (
    StatsConfig,
    describe,
    zscore_array,
    iqr,
    winsorize,
    percentile_rank,
    outlier_mask,
    running_stats,
    weighted_mean,
    weighted_std,
    batch_describe,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _arr(n=10, val=1.0) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n=10) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def _with_outlier() -> np.ndarray:
    a = np.zeros(20)
    a[-1] = 1000.0
    return a


# ─── StatsConfig ──────────────────────────────────────────────────────────────

class TestStatsConfigExtra:
    def test_default_outlier_iqr_k(self):
        assert StatsConfig().outlier_iqr_k == pytest.approx(1.5)

    def test_default_winsor_low(self):
        assert StatsConfig().winsor_low == pytest.approx(0.05)

    def test_default_winsor_high(self):
        assert StatsConfig().winsor_high == pytest.approx(0.95)

    def test_default_ddof(self):
        assert StatsConfig().ddof == 0

    def test_iqr_k_zero_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=0.0)

    def test_iqr_k_negative_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=-1.0)

    def test_winsor_low_ge_high_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=0.8, winsor_high=0.2)

    def test_ddof_invalid_raises(self):
        with pytest.raises(ValueError):
            StatsConfig(ddof=2)

    def test_valid_ddof_one(self):
        cfg = StatsConfig(ddof=1)
        assert cfg.ddof == 1


# ─── describe ─────────────────────────────────────────────────────────────────

class TestDescribeExtra:
    def test_returns_dict(self):
        assert isinstance(describe(_ramp()), dict)

    def test_required_keys(self):
        keys = describe(_ramp()).keys()
        for k in ("min", "max", "mean", "std", "median", "q25", "q75", "iqr"):
            assert k in keys

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            describe(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            describe(np.array([]))

    def test_constant_std_zero(self):
        d = describe(_arr(5, 3.0))
        assert d["std"] == pytest.approx(0.0)

    def test_min_max_correct(self):
        a = np.array([2.0, 5.0, 1.0, 8.0])
        d = describe(a)
        assert d["min"] == pytest.approx(1.0)
        assert d["max"] == pytest.approx(8.0)

    def test_none_cfg(self):
        d = describe(_ramp(5), cfg=None)
        assert "mean" in d

    def test_single_element(self):
        d = describe(np.array([42.0]))
        assert d["min"] == d["max"] == pytest.approx(42.0)


# ─── zscore_array ─────────────────────────────────────────────────────────────

class TestZscoreArrayExtra:
    def test_returns_ndarray(self):
        assert isinstance(zscore_array(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert zscore_array(_ramp()).dtype == np.float64

    def test_length_preserved(self):
        assert len(zscore_array(_ramp(8))) == 8

    def test_constant_returns_zeros(self):
        out = zscore_array(_arr(5, 3.0))
        assert np.allclose(out, 0.0)

    def test_mean_near_zero(self):
        out = zscore_array(_ramp(10))
        assert abs(out.mean()) < 1e-10

    def test_std_near_one(self):
        out = zscore_array(_ramp(10))
        assert abs(out.std() - 1.0) < 0.01

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.array([]))


# ─── iqr ──────────────────────────────────────────────────────────────────────

class TestIqrExtra:
    def test_returns_float(self):
        assert isinstance(iqr(_ramp()), float)

    def test_nonneg(self):
        assert iqr(_ramp(10)) >= 0.0

    def test_constant_is_zero(self):
        assert iqr(_arr(5, 1.0)) == pytest.approx(0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            iqr(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iqr(np.array([]))

    def test_known_value(self):
        # Uniform 0..3: q25=0.75, q75=2.25, IQR=1.5
        a = np.array([0.0, 1.0, 2.0, 3.0])
        assert iqr(a) == pytest.approx(1.5)


# ─── winsorize ────────────────────────────────────────────────────────────────

class TestWinsorizeExtra:
    def test_returns_ndarray(self):
        assert isinstance(winsorize(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert winsorize(_ramp()).dtype == np.float64

    def test_length_preserved(self):
        assert len(winsorize(_ramp(12))) == 12

    def test_clips_outlier(self):
        out = winsorize(_with_outlier())
        assert out.max() < 1000.0

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            winsorize(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            winsorize(np.array([]))

    def test_constant_unchanged(self):
        out = winsorize(_arr(5, 3.0))
        assert np.allclose(out, 3.0)

    def test_none_cfg(self):
        out = winsorize(_ramp(10), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── percentile_rank ──────────────────────────────────────────────────────────

class TestPercentileRankExtra:
    def test_returns_float(self):
        assert isinstance(percentile_rank(_ramp(), 5.0), float)

    def test_in_range(self):
        r = percentile_rank(_ramp(10), 5.0)
        assert 0.0 <= r <= 100.0

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(np.zeros((2, 3)), 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(np.array([]), 1.0)

    def test_min_value_low_rank(self):
        a = _ramp(10)
        r = percentile_rank(a, a.min())
        assert r <= 10.0

    def test_max_value_high_rank(self):
        a = _ramp(10)
        r = percentile_rank(a, a.max())
        assert r >= 90.0


# ─── outlier_mask ─────────────────────────────────────────────────────────────

class TestOutlierMaskExtra:
    def test_returns_bool_ndarray(self):
        out = outlier_mask(_ramp())
        assert out.dtype == bool

    def test_shape_preserved(self):
        a = _ramp(8)
        assert outlier_mask(a).shape == (8,)

    def test_constant_no_outliers(self):
        out = outlier_mask(_arr(10, 1.0))
        assert not out.any()

    def test_extreme_outlier_detected(self):
        out = outlier_mask(_with_outlier())
        assert out[-1] is np.True_

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            outlier_mask(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            outlier_mask(np.array([]))

    def test_none_cfg(self):
        out = outlier_mask(_ramp(10), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── running_stats ────────────────────────────────────────────────────────────

class TestRunningStatsExtra:
    def test_returns_dict(self):
        assert isinstance(running_stats(_ramp()), dict)

    def test_keys_present(self):
        rs = running_stats(_ramp())
        for k in ("cumsum", "cummax", "cummin", "cummean"):
            assert k in rs

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            running_stats(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            running_stats(np.array([]))

    def test_cumsum_final(self):
        a = _ramp(5)
        rs = running_stats(a)
        assert rs["cumsum"][-1] == pytest.approx(a.sum())

    def test_cummax_nondecreasing(self):
        a = _ramp(8)
        cm = running_stats(a)["cummax"]
        assert np.all(cm[1:] >= cm[:-1])

    def test_cummin_nonincreasing(self):
        a = _ramp(8)
        cm = running_stats(a)["cummin"]
        # cummin is non-increasing only if series is decreasing; for ramp it stays at 0
        assert np.all(cm[1:] >= cm[:-1] - 1e-10)

    def test_cummean_dtype(self):
        assert running_stats(_ramp())["cummean"].dtype == np.float64


# ─── weighted_mean ────────────────────────────────────────────────────────────

class TestWeightedMeanExtra:
    def test_returns_float(self):
        a = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        assert isinstance(weighted_mean(a, w), float)

    def test_equal_weights_matches_mean(self):
        a = _ramp(8)
        w = np.ones(8)
        assert weighted_mean(a, w) == pytest.approx(a.mean())

    def test_known_value(self):
        a = np.array([0.0, 1.0])
        w = np.array([1.0, 3.0])
        assert weighted_mean(a, w) == pytest.approx(0.75)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.zeros((2, 3)), np.ones(6))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([1.0, 2.0]), np.ones(3))

    def test_zero_weight_sum_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([1.0]), np.array([0.0]))


# ─── weighted_std ─────────────────────────────────────────────────────────────

class TestWeightedStdExtra:
    def test_returns_float(self):
        a = _ramp(5)
        w = np.ones(5)
        assert isinstance(weighted_std(a, w), float)

    def test_nonneg(self):
        a = _ramp(8)
        w = np.ones(8)
        assert weighted_std(a, w) >= 0.0

    def test_constant_is_zero(self):
        a = _arr(5, 3.0)
        w = np.ones(5)
        assert weighted_std(a, w) == pytest.approx(0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            weighted_std(np.zeros((2, 3)), np.ones(6))


# ─── batch_describe ───────────────────────────────────────────────────────────

class TestBatchDescribeExtra:
    def test_returns_list(self):
        assert isinstance(batch_describe([_ramp()]), list)

    def test_length_matches(self):
        result = batch_describe([_ramp(5), _ramp(8)])
        assert len(result) == 2

    def test_each_element_dict(self):
        for d in batch_describe([_ramp(5), _ramp(8)]):
            assert isinstance(d, dict)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_describe([])

    def test_none_cfg(self):
        result = batch_describe([_ramp(5)], cfg=None)
        assert len(result) == 1

    def test_keys_correct(self):
        for d in batch_describe([_ramp(5), _ramp(10)]):
            assert "mean" in d and "std" in d
