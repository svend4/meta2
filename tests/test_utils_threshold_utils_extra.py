"""Extra tests for puzzle_reconstruction/utils/threshold_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.threshold_utils import (
    ThresholdConfig,
    apply_threshold,
    binarize,
    adaptive_threshold,
    soft_threshold,
    threshold_matrix,
    hysteresis_threshold,
    otsu_threshold,
    count_above,
    fraction_above,
    batch_threshold,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _arr(n=10, val=0.5) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n=10) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


# ─── ThresholdConfig ──────────────────────────────────────────────────────────

class TestThresholdConfigExtra:
    def test_default_low(self):
        assert ThresholdConfig().low == pytest.approx(0.3)

    def test_default_high(self):
        assert ThresholdConfig().high == pytest.approx(0.7)

    def test_default_invert(self):
        assert ThresholdConfig().invert is False

    def test_default_mode(self):
        assert ThresholdConfig().mode == "hard"

    def test_low_gt_high_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(low=0.8, high=0.2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(mode="adaptive")

    def test_soft_mode_valid(self):
        cfg = ThresholdConfig(mode="soft")
        assert cfg.mode == "soft"

    def test_low_equals_high_valid(self):
        cfg = ThresholdConfig(low=0.5, high=0.5)
        assert cfg.low == cfg.high


# ─── apply_threshold ──────────────────────────────────────────────────────────

class TestApplyThresholdExtra:
    def test_returns_bool_array(self):
        out = apply_threshold(_ramp(), 0.5)
        assert out.dtype == bool

    def test_shape_preserved(self):
        a = _ramp(8)
        assert apply_threshold(a, 0.5).shape == (8,)

    def test_at_threshold_is_true(self):
        a = np.array([0.5, 0.6, 0.4])
        out = apply_threshold(a, 0.5)
        assert out[0] is np.True_

    def test_invert_flips(self):
        a = np.array([0.3, 0.7])
        out = apply_threshold(a, 0.5, invert=True)
        assert out[0] is np.True_
        assert out[1] is np.False_

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)

    def test_2d_works(self):
        M = np.array([[0.3, 0.7], [0.8, 0.1]])
        out = apply_threshold(M, 0.5)
        assert out.shape == (2, 2)
        assert out.dtype == bool

    def test_all_above(self):
        out = apply_threshold(np.ones(5), 0.5)
        assert out.all()

    def test_all_below(self):
        out = apply_threshold(np.zeros(5), 0.5)
        assert not out.any()


# ─── binarize ─────────────────────────────────────────────────────────────────

class TestBinarizeExtra:
    def test_returns_float64(self):
        assert binarize(_ramp(), 0.5).dtype == np.float64

    def test_only_zero_or_one(self):
        out = binarize(_ramp(10), 0.5)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_shape_preserved(self):
        assert binarize(_ramp(8), 0.5).shape == (8,)

    def test_invert(self):
        a = np.array([0.3, 0.7])
        out = binarize(a, 0.5, invert=True)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            binarize(np.array([]), 0.5)

    def test_2d_works(self):
        M = np.ones((3, 3))
        out = binarize(M, 0.5)
        assert out.shape == (3, 3)
        assert np.all(out == 1.0)


# ─── adaptive_threshold ───────────────────────────────────────────────────────

class TestAdaptiveThresholdExtra:
    def test_returns_bool_array(self):
        out = adaptive_threshold(_ramp())
        assert out.dtype == bool

    def test_same_length(self):
        assert len(adaptive_threshold(_ramp(12))) == 12

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([]))

    def test_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(_ramp(), window=0)

    def test_all_same_all_true(self):
        out = adaptive_threshold(_arr(10, 1.0))
        assert out.all()

    def test_positive_offset_makes_stricter(self):
        s = _ramp(10)
        out0 = adaptive_threshold(s, offset=0.0)
        out_pos = adaptive_threshold(s, offset=0.5)
        assert out_pos.sum() <= out0.sum()


# ─── soft_threshold ───────────────────────────────────────────────────────────

class TestSoftThresholdExtra:
    def test_returns_ndarray(self):
        assert isinstance(soft_threshold(_ramp(), 0.2), np.ndarray)

    def test_dtype_float64(self):
        assert soft_threshold(_ramp(), 0.2).dtype == np.float64

    def test_zero_value_identity(self):
        s = _ramp(10)
        np.testing.assert_allclose(soft_threshold(s, 0.0), s)

    def test_below_threshold_becomes_zero(self):
        s = np.array([0.1, 0.5, 1.0])
        out = soft_threshold(s, 0.3)
        assert out[0] == pytest.approx(0.0)

    def test_shrinks_above_threshold(self):
        s = np.array([1.0])
        out = soft_threshold(s, 0.3)
        assert out[0] == pytest.approx(0.7)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.zeros((2, 3)), 0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([]), 0.1)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(_ramp(), -0.1)


# ─── threshold_matrix ─────────────────────────────────────────────────────────

class TestThresholdMatrixExtra:
    def test_returns_float64(self):
        M = np.ones((3, 3))
        assert threshold_matrix(M, 0.5).dtype == np.float64

    def test_shape_preserved(self):
        M = np.zeros((4, 5))
        assert threshold_matrix(M, 0.0).shape == (4, 5)

    def test_fills_below_threshold(self):
        M = np.array([[0.2, 0.8], [0.5, 0.3]])
        out = threshold_matrix(M, 0.5, fill=0.0)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[1, 0] == pytest.approx(0.5)  # exactly at threshold → kept

    def test_custom_fill(self):
        M = np.ones((3, 3)) * 0.2
        out = threshold_matrix(M, 0.5, fill=-1.0)
        assert np.all(out == -1.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.zeros(5), 0.5)

    def test_original_not_modified(self):
        M = np.ones((3, 3))
        original = M.copy()
        threshold_matrix(M, 0.5)
        np.testing.assert_array_equal(M, original)


# ─── hysteresis_threshold ─────────────────────────────────────────────────────

class TestHysteresisThresholdExtra:
    def test_returns_bool_array(self):
        out = hysteresis_threshold(_ramp(), 0.3, 0.7)
        assert out.dtype == bool

    def test_length_preserved(self):
        assert len(hysteresis_threshold(_ramp(10), 0.3, 0.7)) == 10

    def test_strong_is_true(self):
        s = np.array([0.8])
        out = hysteresis_threshold(s, 0.3, 0.7)
        assert out[0] is np.True_

    def test_below_low_is_false(self):
        s = np.array([0.1])
        out = hysteresis_threshold(s, 0.3, 0.7)
        assert out[0] is np.False_

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.zeros((2, 3)), 0.3, 0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([]), 0.3, 0.7)

    def test_low_gt_high_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(_ramp(), 0.8, 0.2)

    def test_weak_adjacent_to_strong_true(self):
        s = np.array([0.0, 0.4, 0.8])  # weak=0.4, strong=0.8
        out = hysteresis_threshold(s, 0.3, 0.7)
        assert out[1] is np.True_


# ─── otsu_threshold ───────────────────────────────────────────────────────────

class TestOtsuThresholdExtra:
    def test_returns_float(self):
        a = np.array([0.0, 0.0, 1.0, 1.0])
        assert isinstance(otsu_threshold(a), float)

    def test_bimodal_separates(self):
        a = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)])
        t = otsu_threshold(a)
        assert 0.1 <= t <= 0.9

    def test_in_data_range(self):
        a = _ramp(10)
        t = otsu_threshold(a)
        assert a.min() <= t <= a.max()

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.zeros((2, 3)))

    def test_less_than_2_elements_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.array([0.5]))

    def test_constant_signal(self):
        a = _arr(5, 0.7)
        t = otsu_threshold(a)
        assert isinstance(t, float)


# ─── count_above ──────────────────────────────────────────────────────────────

class TestCountAboveExtra:
    def test_returns_int(self):
        assert isinstance(count_above(_ramp(), 0.5), int)

    def test_all_above(self):
        assert count_above(np.ones(5), 0.5) == 5

    def test_none_above(self):
        assert count_above(np.zeros(5), 0.5) == 0

    def test_at_threshold_counted(self):
        a = np.array([0.5, 0.6, 0.4])
        assert count_above(a, 0.5) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            count_above(np.array([]), 0.5)

    def test_2d_works(self):
        M = np.array([[0.3, 0.7], [0.8, 0.1]])
        assert count_above(M, 0.5) == 2


# ─── fraction_above ───────────────────────────────────────────────────────────

class TestFractionAboveExtra:
    def test_returns_float(self):
        assert isinstance(fraction_above(_ramp(), 0.5), float)

    def test_in_range(self):
        f = fraction_above(_ramp(10), 0.5)
        assert 0.0 <= f <= 1.0

    def test_all_above_is_one(self):
        assert fraction_above(np.ones(5), 0.5) == pytest.approx(1.0)

    def test_none_above_is_zero(self):
        assert fraction_above(np.zeros(5), 0.5) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fraction_above(np.array([]), 0.5)

    def test_2d_works(self):
        M = np.ones((4, 4))
        assert fraction_above(M, 0.5) == pytest.approx(1.0)


# ─── batch_threshold ──────────────────────────────────────────────────────────

class TestBatchThresholdExtra:
    def test_returns_list(self):
        result = batch_threshold([_ramp()], 0.5)
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_threshold([_ramp(5), _ramp(8)], 0.5)
        assert len(result) == 2

    def test_each_element_bool(self):
        for out in batch_threshold([_ramp(5), _ramp(8)], 0.5):
            assert out.dtype == bool

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_threshold([], 0.5)

    def test_invert(self):
        arrays = [np.array([0.3, 0.7])]
        result = batch_threshold(arrays, 0.5, invert=True)
        assert result[0][0] is np.True_
        assert result[0][1] is np.False_

    def test_consistent_with_apply_threshold(self):
        a = _ramp(8)
        expected = apply_threshold(a, 0.5)
        result = batch_threshold([a], 0.5)[0]
        np.testing.assert_array_equal(result, expected)
