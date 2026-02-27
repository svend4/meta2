"""
Property-based tests for puzzle_reconstruction.utils.stats_utils.

Verifies mathematical invariants:
- StatsConfig:       valid parameter ranges, ValueError on bad inputs
- describe:          min ≤ mean ≤ max, q25 ≤ q75, iqr = q75 - q25, range check
- zscore_array:      mean ≈ 0 and std ≈ 1 for non-constant input; zeros for const
- iqr:               ≥ 0; equal elements → 0; sorted vs unsorted same result
- winsorize:         output ∈ [lo_quantile, hi_quantile], same length as input
- percentile_rank:   ∈ [0, 100]; value below all → 0; ordering preserved
- outlier_mask:      boolean mask, same length; constant array → no outliers
- running_stats:     cumsum[-1] = sum; cummax non-decreasing; cummin non-incr
- weighted_mean:     equal weights → arithmetic mean; single element = that elem
- weighted_std:      ≥ 0; constant array → 0
- batch_describe:    same length as input; each entry = describe(a)
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

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

RNG = np.random.default_rng(42)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_arr(n: int = 20, low: float = -10.0, high: float = 10.0) -> np.ndarray:
    return RNG.uniform(low, high, size=n)


def _rand_pos_weights(n: int = 20) -> np.ndarray:
    w = RNG.uniform(0.1, 1.0, size=n)
    return w


# ─── StatsConfig ──────────────────────────────────────────────────────────────

class TestStatsConfig:
    def test_default_valid(self):
        cfg = StatsConfig()
        assert cfg.outlier_iqr_k > 0
        assert 0.0 <= cfg.winsor_low < cfg.winsor_high <= 1.0
        assert cfg.ddof in (0, 1)

    def test_custom_valid(self):
        cfg = StatsConfig(outlier_iqr_k=3.0, winsor_low=0.01, winsor_high=0.99, ddof=1)
        assert cfg.outlier_iqr_k == 3.0
        assert cfg.ddof == 1

    @pytest.mark.parametrize("k", [0.0, -1.0])
    def test_invalid_outlier_iqr_k(self, k):
        with pytest.raises(ValueError):
            StatsConfig(outlier_iqr_k=k)

    @pytest.mark.parametrize("lo, hi", [(0.5, 0.1), (-0.1, 0.9), (0.0, 0.0), (1.1, 0.9)])
    def test_invalid_winsor_range(self, lo, hi):
        with pytest.raises(ValueError):
            StatsConfig(winsor_low=lo, winsor_high=hi)

    def test_invalid_ddof(self):
        with pytest.raises(ValueError):
            StatsConfig(ddof=2)


# ─── describe ─────────────────────────────────────────────────────────────────

class TestDescribe:
    def test_keys_present(self):
        arr = _rand_arr()
        d = describe(arr)
        for key in ("min", "max", "mean", "std", "median", "q25", "q75", "iqr"):
            assert key in d

    def test_min_le_mean_le_max(self):
        for _ in range(20):
            arr = _rand_arr()
            d = describe(arr)
            assert d["min"] <= d["mean"] <= d["max"] + 1e-12

    def test_q25_le_median_le_q75(self):
        for _ in range(20):
            arr = _rand_arr()
            d = describe(arr)
            assert d["q25"] <= d["median"] + 1e-12
            assert d["median"] <= d["q75"] + 1e-12

    def test_iqr_equals_q75_minus_q25(self):
        for _ in range(20):
            arr = _rand_arr()
            d = describe(arr)
            assert abs(d["iqr"] - (d["q75"] - d["q25"])) < 1e-10

    def test_std_nonneg(self):
        for _ in range(20):
            arr = _rand_arr()
            d = describe(arr)
            assert d["std"] >= 0.0

    def test_single_element(self):
        d = describe(np.array([5.0]))
        assert d["min"] == 5.0
        assert d["max"] == 5.0
        assert d["mean"] == 5.0
        assert d["std"] == 0.0
        assert d["iqr"] == 0.0

    def test_constant_array(self):
        arr = np.full(10, 3.14)
        d = describe(arr)
        assert d["min"] == pytest.approx(3.14)
        assert d["max"] == pytest.approx(3.14)
        assert d["std"] == 0.0
        assert d["iqr"] == 0.0

    def test_min_is_actual_min(self):
        arr = _rand_arr()
        d = describe(arr)
        assert d["min"] == pytest.approx(float(arr.min()))

    def test_max_is_actual_max(self):
        arr = _rand_arr()
        d = describe(arr)
        assert d["max"] == pytest.approx(float(arr.max()))

    def test_mean_is_actual_mean(self):
        arr = _rand_arr()
        d = describe(arr)
        assert d["mean"] == pytest.approx(float(arr.mean()), rel=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            describe(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            describe(np.ones((3, 3)))


# ─── zscore_array ─────────────────────────────────────────────────────────────

class TestZscoreArray:
    def test_mean_near_zero(self):
        for _ in range(20):
            arr = _rand_arr(50)
            z = zscore_array(arr)
            assert abs(z.mean()) < 1e-10

    def test_std_near_one(self):
        for _ in range(20):
            arr = _rand_arr(50)
            if arr.std() < 1e-6:
                continue
            z = zscore_array(arr)
            assert abs(z.std() - 1.0) < 1e-8

    def test_constant_returns_zeros(self):
        arr = np.full(10, 7.0)
        z = zscore_array(arr)
        np.testing.assert_array_equal(z, np.zeros(10))

    def test_same_length(self):
        arr = _rand_arr(30)
        z = zscore_array(arr)
        assert len(z) == 30

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            zscore_array(np.ones((3, 3)))

    def test_ordering_preserved(self):
        """Zscore preserves relative ordering."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore_array(arr)
        # strictly increasing
        assert all(z[i] < z[i + 1] for i in range(len(z) - 1))

    def test_ddof_respected(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg_0 = StatsConfig(ddof=0)
        cfg_1 = StatsConfig(ddof=1)
        z0 = zscore_array(arr, cfg_0)
        z1 = zscore_array(arr, cfg_1)
        # ddof=0 → smaller std → larger z values than ddof=1
        assert abs(z0.max()) >= abs(z1.max()) - 1e-12


# ─── iqr ──────────────────────────────────────────────────────────────────────

class TestIqr:
    def test_nonneg(self):
        for _ in range(30):
            assert iqr(_rand_arr()) >= 0.0

    def test_constant_array_zero(self):
        assert iqr(np.full(10, 5.0)) == 0.0

    def test_sorted_vs_unsorted_same(self):
        arr = _rand_arr(40)
        shuffled = arr.copy()
        RNG.shuffle(shuffled)
        assert abs(iqr(arr) - iqr(shuffled)) < 1e-10

    def test_scale_linear(self):
        arr = _rand_arr(30)
        k = 3.0
        assert abs(iqr(arr * k) - k * iqr(arr)) < 1e-8

    def test_shift_invariant(self):
        arr = _rand_arr(30)
        assert abs(iqr(arr + 100.0) - iqr(arr)) < 1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            iqr(np.array([]))

    def test_two_distinct_elements(self):
        # IQR of [0, 10] should be >= 0
        assert iqr(np.array([0.0, 10.0])) >= 0.0


# ─── winsorize ────────────────────────────────────────────────────────────────

class TestWinsorize:
    def test_same_length(self):
        arr = _rand_arr(30)
        w = winsorize(arr)
        assert len(w) == 30

    def test_values_in_range(self):
        for _ in range(20):
            arr = _rand_arr(50)
            cfg = StatsConfig(winsor_low=0.05, winsor_high=0.95)
            lo = float(np.percentile(arr, 5))
            hi = float(np.percentile(arr, 95))
            w = winsorize(arr, cfg)
            assert w.min() >= lo - 1e-10
            assert w.max() <= hi + 1e-10

    def test_constant_array_unchanged(self):
        arr = np.full(20, 4.0)
        w = winsorize(arr)
        np.testing.assert_array_almost_equal(w, arr)

    def test_sorted_output_monotone(self):
        arr = np.sort(_rand_arr(50))
        w = winsorize(arr)
        # After winsorization, array should still be non-decreasing
        assert np.all(np.diff(w) >= -1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            winsorize(np.array([]))


# ─── percentile_rank ──────────────────────────────────────────────────────────

class TestPercentileRank:
    def test_in_range_0_100(self):
        for _ in range(30):
            arr = _rand_arr(30)
            value = float(RNG.uniform(-20, 20))
            pr = percentile_rank(arr, value)
            assert 0.0 <= pr <= 100.0

    def test_below_min_returns_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(arr, 0.0) == 0.0

    def test_above_all_near_100(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pr = percentile_rank(arr, 10.0)
        assert pr == 100.0

    def test_ordering(self):
        arr = _rand_arr(40)
        pr_low = percentile_rank(arr, -100.0)
        pr_high = percentile_rank(arr, 100.0)
        assert pr_low <= pr_high

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(np.array([]), 0.0)


# ─── outlier_mask ─────────────────────────────────────────────────────────────

class TestOutlierMask:
    def test_boolean_mask(self):
        arr = _rand_arr(30)
        mask = outlier_mask(arr)
        assert mask.dtype == bool

    def test_same_length(self):
        arr = _rand_arr(30)
        mask = outlier_mask(arr)
        assert len(mask) == 30

    def test_constant_array_no_outliers(self):
        arr = np.full(20, 3.0)
        mask = outlier_mask(arr)
        assert not mask.any()

    def test_extreme_outlier_detected(self):
        arr = np.concatenate([np.zeros(20), [1000.0]])
        mask = outlier_mask(arr)
        assert mask[-1]  # last element should be outlier

    def test_symmetric_k_effect(self):
        """Larger k → fewer outliers."""
        arr = np.concatenate([np.zeros(18), [-50.0, 50.0]])
        cfg_tight = StatsConfig(outlier_iqr_k=1.0)
        cfg_loose = StatsConfig(outlier_iqr_k=5.0)
        assert outlier_mask(arr, cfg_tight).sum() >= outlier_mask(arr, cfg_loose).sum()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            outlier_mask(np.array([]))


# ─── running_stats ────────────────────────────────────────────────────────────

class TestRunningStats:
    def test_cumsum_last_equals_sum(self):
        for _ in range(20):
            arr = _rand_arr(30)
            rs = running_stats(arr)
            assert abs(rs["cumsum"][-1] - arr.sum()) < 1e-8

    def test_cummax_nondecreasing(self):
        for _ in range(20):
            arr = _rand_arr(30)
            rs = running_stats(arr)
            assert np.all(np.diff(rs["cummax"]) >= -1e-12)

    def test_cummin_nonincreasing(self):
        for _ in range(20):
            arr = _rand_arr(30)
            rs = running_stats(arr)
            assert np.all(np.diff(rs["cummin"]) <= 1e-12)

    def test_cummean_last_equals_mean(self):
        arr = _rand_arr(30)
        rs = running_stats(arr)
        assert abs(rs["cummean"][-1] - arr.mean()) < 1e-8

    def test_all_same_length(self):
        arr = _rand_arr(25)
        rs = running_stats(arr)
        for key in ("cumsum", "cummax", "cummin", "cummean"):
            assert len(rs[key]) == 25

    def test_cummax_starts_at_first(self):
        arr = _rand_arr(10)
        rs = running_stats(arr)
        assert rs["cummax"][0] == pytest.approx(float(arr[0]))

    def test_cummin_starts_at_first(self):
        arr = _rand_arr(10)
        rs = running_stats(arr)
        assert rs["cummin"][0] == pytest.approx(float(arr[0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            running_stats(np.array([]))

    def test_single_element(self):
        arr = np.array([7.0])
        rs = running_stats(arr)
        assert rs["cumsum"][0] == pytest.approx(7.0)
        assert rs["cummean"][0] == pytest.approx(7.0)


# ─── weighted_mean ────────────────────────────────────────────────────────────

class TestWeightedMean:
    def test_equal_weights_equals_mean(self):
        for _ in range(20):
            arr = _rand_arr(20)
            w = np.ones(20)
            assert abs(weighted_mean(arr, w) - arr.mean()) < 1e-10

    def test_single_element(self):
        arr = np.array([42.0])
        w = np.array([1.0])
        assert weighted_mean(arr, w) == pytest.approx(42.0)

    def test_weight_concentration(self):
        """Mean should be close to element with highest weight."""
        arr = np.array([0.0, 1.0, 100.0])
        w = np.array([0.001, 0.001, 100.0])
        wm = weighted_mean(arr, w)
        assert wm > 90.0

    def test_zero_weight_sum_raises(self):
        arr = np.array([1.0, 2.0])
        w = np.zeros(2)
        with pytest.raises(ValueError):
            weighted_mean(arr, w)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.ones(3), np.ones(4))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_mean(np.array([]), np.array([]))

    def test_scaling_weights_same_result(self):
        arr = _rand_arr(20)
        w = _rand_pos_weights(20)
        wm1 = weighted_mean(arr, w)
        wm2 = weighted_mean(arr, w * 5.0)
        assert abs(wm1 - wm2) < 1e-8

    def test_in_range_of_values(self):
        for _ in range(20):
            arr = _rand_arr(20)
            w = _rand_pos_weights(20)
            wm = weighted_mean(arr, w)
            assert arr.min() - 1e-10 <= wm <= arr.max() + 1e-10


# ─── weighted_std ─────────────────────────────────────────────────────────────

class TestWeightedStd:
    def test_nonneg(self):
        for _ in range(30):
            arr = _rand_arr(20)
            w = _rand_pos_weights(20)
            assert weighted_std(arr, w) >= 0.0

    def test_constant_array_zero(self):
        arr = np.full(10, 5.0)
        w = _rand_pos_weights(10)
        assert weighted_std(arr, w) < 1e-10

    def test_single_element_zero(self):
        arr = np.array([7.0])
        w = np.array([1.0])
        assert weighted_std(arr, w) == pytest.approx(0.0)

    def test_zero_weight_sum_raises(self):
        arr = np.array([1.0, 2.0])
        w = np.zeros(2)
        with pytest.raises(ValueError):
            weighted_std(arr, w)

    def test_scaling_values_scales_std(self):
        arr = _rand_arr(20)
        w = _rand_pos_weights(20)
        k = 3.0
        std1 = weighted_std(arr, w)
        std2 = weighted_std(arr * k, w)
        assert abs(std2 - k * std1) < 1e-6


# ─── batch_describe ───────────────────────────────────────────────────────────

class TestBatchDescribe:
    def test_same_length(self):
        arrays = [_rand_arr(n) for n in [5, 10, 20]]
        result = batch_describe(arrays)
        assert len(result) == 3

    def test_each_matches_describe(self):
        arrays = [_rand_arr(n) for n in [5, 10, 20]]
        result = batch_describe(arrays)
        for arr, d in zip(arrays, result):
            expected = describe(arr)
            for key in expected:
                assert abs(d[key] - expected[key]) < 1e-10

    def test_single_array(self):
        arr = _rand_arr(10)
        result = batch_describe([arr])
        assert len(result) == 1
        assert result[0]["mean"] == pytest.approx(float(arr.mean()), rel=1e-9)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_describe([])

    def test_order_preserved(self):
        arrays = [np.array([float(i)]) for i in range(5)]
        result = batch_describe(arrays)
        means = [d["mean"] for d in result]
        assert means == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
