"""Tests for puzzle_reconstruction/utils/threshold_utils.py."""
import numpy as np
import pytest

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


# ─── TestThresholdConfig ──────────────────────────────────────────────────────

class TestThresholdConfig:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.low == pytest.approx(0.3)
        assert cfg.high == pytest.approx(0.7)
        assert cfg.invert is False
        assert cfg.mode == "hard"

    def test_valid_low_equals_high(self):
        cfg = ThresholdConfig(low=0.5, high=0.5)
        assert cfg.low == cfg.high

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(low=0.8, high=0.2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(mode="unknown")

    def test_mode_hard(self):
        cfg = ThresholdConfig(mode="hard")
        assert cfg.mode == "hard"

    def test_mode_soft(self):
        cfg = ThresholdConfig(mode="soft")
        assert cfg.mode == "soft"

    def test_invert_true(self):
        cfg = ThresholdConfig(invert=True)
        assert cfg.invert is True

    def test_low_zero_high_one(self):
        cfg = ThresholdConfig(low=0.0, high=1.0)
        assert cfg.low == 0.0 and cfg.high == 1.0


# ─── TestApplyThreshold ───────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_returns_bool_array(self):
        result = apply_threshold(np.array([0.1, 0.5, 0.9]), 0.5)
        assert result.dtype == bool

    def test_same_shape(self):
        arr = np.array([1, 2, 3, 4])
        result = apply_threshold(arr, 2.5)
        assert result.shape == arr.shape

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)

    def test_all_above_threshold_all_true(self):
        arr = np.array([0.6, 0.7, 0.8])
        result = apply_threshold(arr, 0.5)
        assert result.all()

    def test_all_below_threshold_all_false(self):
        arr = np.array([0.1, 0.2, 0.3])
        result = apply_threshold(arr, 0.5)
        assert not result.any()

    def test_at_threshold_is_true(self):
        arr = np.array([0.5])
        result = apply_threshold(arr, 0.5)
        assert bool(result[0]) is True

    def test_invert_false_below_false(self):
        arr = np.array([0.1, 0.9])
        result = apply_threshold(arr, 0.5, invert=False)
        assert not result[0] and result[1]

    def test_invert_true_below_true(self):
        arr = np.array([0.1, 0.9])
        result = apply_threshold(arr, 0.5, invert=True)
        assert result[0] and not result[1]

    def test_2d_array(self):
        arr = np.array([[0.2, 0.8], [0.4, 0.6]])
        result = apply_threshold(arr, 0.5)
        assert result.shape == (2, 2)

    def test_mixed_values(self):
        arr = np.array([0.3, 0.5, 0.7])
        result = apply_threshold(arr, 0.5)
        assert not result[0] and result[1] and result[2]

    def test_scalar_array(self):
        result = apply_threshold(np.array([1.0]), 0.5)
        assert result[0] is np.bool_(True)


# ─── TestBinarize ─────────────────────────────────────────────────────────────

class TestBinarize:
    def test_returns_float64(self):
        result = binarize(np.array([0.3, 0.7]), 0.5)
        assert result.dtype == np.float64

    def test_values_zero_or_one(self):
        result = binarize(np.array([0.1, 0.5, 0.9]), 0.5)
        assert set(result.tolist()).issubset({0.0, 1.0})

    def test_same_shape(self):
        arr = np.array([1, 2, 3])
        assert binarize(arr, 1.5).shape == arr.shape

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            binarize(np.array([]), 0.5)

    def test_above_is_one(self):
        result = binarize(np.array([0.8, 0.9]), 0.5)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_below_is_zero(self):
        result = binarize(np.array([0.1, 0.2]), 0.5)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_invert(self):
        result = binarize(np.array([0.1, 0.9]), 0.5, invert=True)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_2d_array(self):
        arr = np.array([[0.2, 0.8]])
        result = binarize(arr, 0.5)
        assert result.shape == (1, 2)


# ─── TestAdaptiveThreshold ────────────────────────────────────────────────────

class TestAdaptiveThreshold:
    def test_returns_bool(self):
        result = adaptive_threshold(np.array([0.1, 0.5, 0.9]))
        assert result.dtype == bool

    def test_same_length(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert len(adaptive_threshold(arr)) == len(arr)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([[1.0, 2.0]]))

    def test_window_less_than_1_raises(self):
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([1.0, 2.0]), window=0)

    def test_all_same_values_all_true(self):
        # All values equal local mean → all >= thresh → all True
        result = adaptive_threshold(np.ones(5))
        assert result.all()

    def test_invert_all_same_all_false(self):
        result = adaptive_threshold(np.ones(5), invert=True)
        assert not result.any()

    def test_window_1(self):
        arr = np.array([0.1, 0.9, 0.1])
        result = adaptive_threshold(arr, window=1)
        assert isinstance(result, np.ndarray)

    def test_offset_positive_fewer_true(self):
        arr = np.linspace(0, 1, 10)
        r_no_offset = adaptive_threshold(arr, offset=0.0)
        r_with_offset = adaptive_threshold(arr, offset=0.5)
        assert r_with_offset.sum() <= r_no_offset.sum()

    def test_offset_negative_more_true(self):
        arr = np.linspace(0, 1, 10)
        r_no_offset = adaptive_threshold(arr, offset=0.0)
        r_with_offset = adaptive_threshold(arr, offset=-0.5)
        assert r_with_offset.sum() >= r_no_offset.sum()


# ─── TestSoftThreshold ────────────────────────────────────────────────────────

class TestSoftThreshold:
    def test_returns_float64(self):
        result = soft_threshold(np.array([1.0, -1.0]), 0.5)
        assert result.dtype == np.float64

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([]), 0.5)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([[1.0, 2.0]]), 0.5)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            soft_threshold(np.array([1.0, 2.0]), -0.5)

    def test_zero_value_identity(self):
        arr = np.array([1.0, -2.0, 0.5])
        result = soft_threshold(arr, 0.0)
        np.testing.assert_allclose(result, arr)

    def test_shrinks_positive(self):
        result = soft_threshold(np.array([1.0]), 0.3)
        assert result[0] == pytest.approx(0.7)

    def test_shrinks_negative(self):
        result = soft_threshold(np.array([-1.0]), 0.3)
        assert result[0] == pytest.approx(-0.7)

    def test_below_threshold_becomes_zero(self):
        result = soft_threshold(np.array([0.2]), 0.5)
        assert result[0] == pytest.approx(0.0)

    def test_at_threshold_becomes_zero(self):
        result = soft_threshold(np.array([0.5]), 0.5)
        assert result[0] == pytest.approx(0.0)

    def test_preserves_sign(self):
        arr = np.array([2.0, -2.0])
        result = soft_threshold(arr, 1.0)
        assert result[0] > 0 and result[1] < 0

    def test_same_length(self):
        arr = np.array([0.1, 0.5, 0.9])
        assert len(soft_threshold(arr, 0.3)) == 3


# ─── TestThresholdMatrix ──────────────────────────────────────────────────────

class TestThresholdMatrix:
    def test_returns_float64(self):
        m = np.ones((3, 3))
        result = threshold_matrix(m, 0.5)
        assert result.dtype == np.float64

    def test_same_shape(self):
        m = np.ones((4, 5))
        assert threshold_matrix(m, 0.5).shape == (4, 5)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.array([1.0, 2.0]), 0.5)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.ones((2, 2, 2)), 0.5)

    def test_below_value_replaced_with_fill(self):
        m = np.array([[0.3, 0.7], [0.1, 0.9]])
        result = threshold_matrix(m, 0.5)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 0] == pytest.approx(0.0)

    def test_above_value_unchanged(self):
        m = np.array([[0.3, 0.7], [0.1, 0.9]])
        result = threshold_matrix(m, 0.5)
        assert result[0, 1] == pytest.approx(0.7)
        assert result[1, 1] == pytest.approx(0.9)

    def test_custom_fill(self):
        m = np.array([[0.2, 0.8]])
        result = threshold_matrix(m, 0.5, fill=-1.0)
        assert result[0, 0] == pytest.approx(-1.0)

    def test_all_zeros_stays_zeros(self):
        m = np.zeros((3, 3))
        result = threshold_matrix(m, 0.5)
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_does_not_modify_input(self):
        m = np.array([[0.2, 0.8]])
        original = m.copy()
        threshold_matrix(m, 0.5)
        np.testing.assert_array_equal(m, original)


# ─── TestHysteresisThreshold ──────────────────────────────────────────────────

class TestHysteresisThreshold:
    def test_returns_bool(self):
        result = hysteresis_threshold(np.array([0.1, 0.5, 0.9]), 0.3, 0.7)
        assert result.dtype == bool

    def test_same_length(self):
        arr = np.array([0.1, 0.5, 0.9])
        assert len(hysteresis_threshold(arr, 0.3, 0.7)) == len(arr)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([]), 0.3, 0.7)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([[0.5]]), 0.3, 0.7)

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([0.5]), 0.8, 0.2)

    def test_strong_always_true(self):
        arr = np.array([0.9, 0.95, 0.8])
        result = hysteresis_threshold(arr, 0.5, 0.7)
        assert result.all()

    def test_below_low_always_false_isolated(self):
        arr = np.array([0.1, 0.2, 0.1])
        result = hysteresis_threshold(arr, 0.3, 0.7)
        assert not result.any()

    def test_weak_connected_to_strong_becomes_true(self):
        # strong | weak | below
        arr = np.array([0.9, 0.5, 0.1])
        result = hysteresis_threshold(arr, 0.3, 0.7)
        assert bool(result[0]) is True
        assert bool(result[1]) is True
        assert bool(result[2]) is False

    def test_weak_not_connected_stays_false(self):
        arr = np.array([0.1, 0.5, 0.1])
        result = hysteresis_threshold(arr, 0.3, 0.7)
        assert not result.any()

    def test_low_equals_high(self):
        # Weak range is empty; values >= high are strong
        arr = np.array([0.5, 0.6, 0.5])
        result = hysteresis_threshold(arr, 0.5, 0.5)
        assert result.all()


# ─── TestOtsuThreshold ────────────────────────────────────────────────────────

class TestOtsuThreshold:
    def test_returns_float(self):
        result = otsu_threshold(np.array([0.1, 0.9]))
        assert isinstance(result, float)

    def test_single_element_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.array([0.5]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            otsu_threshold(np.array([[0.5, 0.6]]))

    def test_bimodal_separates_clusters(self):
        low = np.linspace(0, 0.1, 50)
        high = np.linspace(0.9, 1.0, 50)
        arr = np.concatenate([low, high])
        t = otsu_threshold(arr)
        # Any threshold in [0.1, 0.9] correctly separates the clusters
        assert 0.09 < t <= 0.9

    def test_constant_array_returns_value(self):
        result = otsu_threshold(np.array([0.5, 0.5, 0.5]))
        assert result == pytest.approx(0.5)

    def test_two_element_array(self):
        result = otsu_threshold(np.array([0.0, 1.0]))
        assert isinstance(result, float)

    def test_result_in_data_range(self):
        arr = np.array([0.1, 0.3, 0.7, 0.9])
        t = otsu_threshold(arr)
        assert arr.min() <= t <= arr.max()


# ─── TestCountAbove ───────────────────────────────────────────────────────────

class TestCountAbove:
    def test_returns_int(self):
        result = count_above(np.array([1, 2, 3]), 2)
        assert isinstance(result, int)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            count_above(np.array([]), 0.5)

    def test_all_above(self):
        assert count_above(np.array([1.0, 2.0, 3.0]), 0.0) == 3

    def test_none_above(self):
        assert count_above(np.array([0.1, 0.2, 0.3]), 0.5) == 0

    def test_at_threshold_counted(self):
        assert count_above(np.array([0.5, 0.5]), 0.5) == 2

    def test_partial(self):
        assert count_above(np.array([0.3, 0.6, 0.9]), 0.5) == 2

    def test_2d_array(self):
        result = count_above(np.array([[0.2, 0.8], [0.4, 0.6]]), 0.5)
        assert result == 2

    def test_negative_threshold(self):
        assert count_above(np.array([-1.0, 0.0, 1.0]), -0.5) == 2


# ─── TestFractionAbove ────────────────────────────────────────────────────────

class TestFractionAbove:
    def test_returns_float(self):
        result = fraction_above(np.array([0.1, 0.9]), 0.5)
        assert isinstance(result, float)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fraction_above(np.array([]), 0.5)

    def test_all_above(self):
        assert fraction_above(np.array([1.0, 2.0]), 0.5) == pytest.approx(1.0)

    def test_none_above(self):
        assert fraction_above(np.array([0.1, 0.2]), 0.5) == pytest.approx(0.0)

    def test_half_above(self):
        assert fraction_above(np.array([0.3, 0.7]), 0.5) == pytest.approx(0.5)

    def test_in_zero_one(self):
        result = fraction_above(np.array([0.1, 0.5, 0.9]), 0.5)
        assert 0.0 <= result <= 1.0

    def test_2d_array(self):
        result = fraction_above(np.array([[0.2, 0.8], [0.4, 0.6]]), 0.5)
        assert result == pytest.approx(0.5)


# ─── TestBatchThreshold ───────────────────────────────────────────────────────

class TestBatchThreshold:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_threshold([], 0.5)

    def test_returns_list(self):
        arrays = [np.array([0.3, 0.7])]
        result = batch_threshold(arrays, 0.5)
        assert isinstance(result, list)

    def test_same_length_as_input(self):
        arrays = [np.array([0.3]), np.array([0.7]), np.array([0.5])]
        result = batch_threshold(arrays, 0.5)
        assert len(result) == 3

    def test_each_is_bool_array(self):
        arrays = [np.array([0.3, 0.7]), np.array([0.1, 0.9])]
        for r in batch_threshold(arrays, 0.5):
            assert r.dtype == bool

    def test_invert_applied(self):
        arrays = [np.array([0.3, 0.7])]
        result = batch_threshold(arrays, 0.5, invert=True)
        assert bool(result[0][0]) is True
        assert bool(result[0][1]) is False

    def test_consistent_with_apply_threshold(self):
        arr = np.array([0.2, 0.5, 0.8])
        expected = apply_threshold(arr, 0.5)
        result = batch_threshold([arr], 0.5)[0]
        np.testing.assert_array_equal(result, expected)

    def test_single_array(self):
        result = batch_threshold([np.array([0.6])], 0.5)
        assert len(result) == 1
        assert bool(result[0][0]) is True
