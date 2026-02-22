"""Tests for puzzle_reconstruction/utils/interpolation_utils.py"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.interpolation_utils import (
    InterpolationConfig,
    lerp,
    lerp_array,
    bilinear_interpolate,
    resample_1d,
    fill_missing,
    interpolate_scores,
    smooth_interpolate,
    batch_resample,
)


# ─── InterpolationConfig ──────────────────────────────────────────────────────

class TestInterpolationConfig:
    def test_defaults(self):
        cfg = InterpolationConfig()
        assert cfg.method == "linear"
        assert cfg.clamp is True
        assert cfg.fill_val == 0.0

    def test_nearest_valid(self):
        cfg = InterpolationConfig(method="nearest")
        assert cfg.method == "nearest"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            InterpolationConfig(method="cubic")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="")

    def test_clamp_false(self):
        cfg = InterpolationConfig(clamp=False)
        assert cfg.clamp is False

    def test_fill_val_custom(self):
        cfg = InterpolationConfig(fill_val=99.0)
        assert cfg.fill_val == 99.0

    def test_all_fields_set(self):
        cfg = InterpolationConfig(method="nearest", clamp=False, fill_val=1.0)
        assert cfg.method == "nearest"
        assert cfg.clamp is False
        assert cfg.fill_val == 1.0


# ─── lerp ─────────────────────────────────────────────────────────────────────

class TestLerp:
    def test_midpoint(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)

    def test_t_zero_returns_a(self):
        assert lerp(3.0, 7.0, 0.0) == pytest.approx(3.0)

    def test_t_one_returns_b(self):
        assert lerp(3.0, 7.0, 1.0) == pytest.approx(7.0)

    def test_negative_a(self):
        assert lerp(-10.0, 10.0, 0.5) == pytest.approx(0.0)

    def test_returns_float(self):
        result = lerp(1, 2, 0.5)
        assert isinstance(result, float)

    def test_t_below_zero_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, -0.1)

    def test_t_above_one_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, 1.1)

    def test_quarter(self):
        assert lerp(0.0, 8.0, 0.25) == pytest.approx(2.0)

    def test_same_endpoints(self):
        assert lerp(5.0, 5.0, 0.7) == pytest.approx(5.0)

    def test_formula_correctness(self):
        a, b, t = 2.0, 8.0, 0.75
        expected = a + t * (b - a)
        assert lerp(a, b, t) == pytest.approx(expected)


# ─── lerp_array ───────────────────────────────────────────────────────────────

class TestLerpArray:
    def test_basic(self):
        a = np.array([0.0, 0.0])
        b = np.array([10.0, 20.0])
        result = lerp_array(a, b, 0.5)
        np.testing.assert_allclose(result, [5.0, 10.0])

    def test_t_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(lerp_array(a, b, 0.0), a)

    def test_t_one(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        np.testing.assert_allclose(lerp_array(a, b, 1.0), b)

    def test_dtype_float64(self):
        result = lerp_array(np.array([0]), np.array([10]), 0.5)
        assert result.dtype == np.float64

    def test_shape_preserved_2d(self):
        a = np.zeros((3, 4))
        b = np.ones((3, 4))
        assert lerp_array(a, b, 0.5).shape == (3, 4)

    def test_t_below_zero_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.array([0.0]), np.array([1.0]), -0.01)

    def test_t_above_one_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.array([0.0]), np.array([1.0]), 1.01)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Формы"):
            lerp_array(np.array([1.0, 2.0]), np.array([1.0]), 0.5)

    def test_integer_arrays_coerced(self):
        result = lerp_array(np.array([0, 4]), np.array([4, 0]), 0.25)
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_3d_shape(self):
        a = np.zeros((2, 3, 4))
        b = np.ones((2, 3, 4))
        result = lerp_array(a, b, 0.5)
        assert result.shape == (2, 3, 4)
        assert np.allclose(result, 0.5)


# ─── bilinear_interpolate ─────────────────────────────────────────────────────

class TestBilinearInterpolate:
    def setup_method(self):
        self.grid = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_top_left_corner(self):
        assert bilinear_interpolate(self.grid, 0.0, 0.0) == pytest.approx(1.0)

    def test_top_right_corner(self):
        assert bilinear_interpolate(self.grid, 1.0, 0.0) == pytest.approx(2.0)

    def test_bottom_left_corner(self):
        assert bilinear_interpolate(self.grid, 0.0, 1.0) == pytest.approx(3.0)

    def test_bottom_right_corner(self):
        assert bilinear_interpolate(self.grid, 1.0, 1.0) == pytest.approx(4.0)

    def test_center(self):
        # Average of all four corners
        assert bilinear_interpolate(self.grid, 0.5, 0.5) == pytest.approx(2.5)

    def test_returns_float(self):
        result = bilinear_interpolate(self.grid, 0.5, 0.5)
        assert isinstance(result, float)

    def test_1d_grid_raises(self):
        with pytest.raises(ValueError, match="2D"):
            bilinear_interpolate(np.array([1.0, 2.0]), 0.5, 0.0)

    def test_x_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(self.grid, 2.0, 0.0)

    def test_y_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(self.grid, 0.0, 2.0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(self.grid, -0.1, 0.0)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(self.grid, 0.0, -0.1)

    def test_larger_uniform_grid(self):
        g = np.ones((5, 5))
        result = bilinear_interpolate(g, 2.0, 2.0)
        assert result == pytest.approx(1.0)


# ─── resample_1d ──────────────────────────────────────────────────────────────

class TestResample1D:
    def test_same_length_returns_copy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = resample_1d(arr, 3)
        np.testing.assert_allclose(result, arr)

    def test_upsample_length(self):
        arr = np.array([0.0, 10.0])
        result = resample_1d(arr, 5)
        assert len(result) == 5

    def test_upsample_endpoints(self):
        arr = np.array([0.0, 10.0])
        result = resample_1d(arr, 5)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(10.0)

    def test_downsample(self):
        arr = np.linspace(0.0, 1.0, 100)
        result = resample_1d(arr, 10)
        assert len(result) == 10

    def test_to_length_1(self):
        arr = np.array([5.0, 10.0, 15.0])
        result = resample_1d(arr, 1)
        assert len(result) == 1

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = resample_1d(arr, 5)
        assert result.dtype == np.float64

    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            resample_1d(np.zeros((3, 3)), 5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            resample_1d(np.array([]), 5)

    def test_target_len_zero_raises(self):
        with pytest.raises(ValueError):
            resample_1d(np.array([1.0, 2.0]), 0)

    def test_target_len_negative_raises(self):
        with pytest.raises(ValueError):
            resample_1d(np.array([1.0, 2.0]), -1)

    def test_nearest_method_length(self):
        cfg = InterpolationConfig(method="nearest")
        arr = np.array([0.0, 1.0])
        result = resample_1d(arr, 3, cfg)
        assert len(result) == 3

    def test_linear_monotone_upscale(self):
        arr = np.linspace(0.0, 1.0, 10)
        result = resample_1d(arr, 20)
        assert np.all(np.diff(result) >= -1e-10)


# ─── fill_missing ─────────────────────────────────────────────────────────────

class TestFillMissing:
    def test_no_nan(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = fill_missing(arr)
        np.testing.assert_allclose(result, arr)

    def test_single_nan_middle(self):
        arr = np.array([0.0, np.nan, 2.0])
        result = fill_missing(arr)
        assert not np.any(np.isnan(result))
        assert result[1] == pytest.approx(1.0)

    def test_all_nan_returns_zeros(self):
        arr = np.array([np.nan, np.nan, np.nan])
        result = fill_missing(arr)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_nan_at_start(self):
        arr = np.array([np.nan, np.nan, 3.0])
        result = fill_missing(arr)
        assert not np.any(np.isnan(result))

    def test_nan_at_end(self):
        arr = np.array([1.0, 2.0, np.nan])
        result = fill_missing(arr)
        assert not np.any(np.isnan(result))

    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            fill_missing(np.zeros((3, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            fill_missing(np.array([]))

    def test_dtype_float64(self):
        result = fill_missing(np.array([1.0, np.nan, 3.0]))
        assert result.dtype == np.float64

    def test_length_preserved(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = fill_missing(arr)
        assert len(result) == len(arr)
        assert not np.any(np.isnan(result))


# ─── interpolate_scores ───────────────────────────────────────────────────────

class TestInterpolateScores:
    def test_symmetric_result(self):
        m = np.array([[0.0, 0.8], [0.2, 0.0]])
        result = interpolate_scores(m, alpha=0.5)
        np.testing.assert_allclose(result, result.T)

    def test_alpha_one_returns_original(self):
        m = np.array([[0.0, 1.0], [0.5, 0.0]])
        result = interpolate_scores(m, alpha=1.0)
        np.testing.assert_allclose(result, m)

    def test_alpha_zero_returns_transpose(self):
        m = np.array([[0.0, 1.0], [0.5, 0.0]])
        result = interpolate_scores(m, alpha=0.0)
        np.testing.assert_allclose(result, m.T)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            interpolate_scores(np.array([1.0, 2.0]))

    def test_not_square_raises(self):
        with pytest.raises(ValueError, match="квадратн"):
            interpolate_scores(np.zeros((2, 3)))

    def test_alpha_below_zero_raises(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.eye(2), alpha=-0.1)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.eye(2), alpha=1.1)

    def test_dtype_float64(self):
        result = interpolate_scores(np.eye(3))
        assert result.dtype == np.float64

    def test_3x3_symmetric(self):
        m = np.arange(9, dtype=float).reshape(3, 3)
        result = interpolate_scores(m, alpha=0.5)
        np.testing.assert_allclose(result, result.T)

    def test_default_alpha_half(self):
        m = np.array([[0.0, 0.8], [0.2, 0.0]])
        result = interpolate_scores(m)
        assert result[0, 1] == pytest.approx(0.5)
        assert result[1, 0] == pytest.approx(0.5)


# ─── smooth_interpolate ───────────────────────────────────────────────────────

class TestSmoothInterpolate:
    def test_window_1_unchanged(self):
        arr = np.array([1.0, 3.0, 5.0, 7.0])
        result = smooth_interpolate(arr, window=1)
        np.testing.assert_allclose(result, arr)

    def test_single_element(self):
        arr = np.array([42.0])
        result = smooth_interpolate(arr, window=5)
        np.testing.assert_allclose(result, [42.0])

    def test_smoothing_applied(self):
        arr = np.array([0.0, 100.0, 0.0, 100.0, 0.0])
        result = smooth_interpolate(arr, window=3)
        # Middle element should be smoothed toward average
        assert result[1] < 100.0
        assert result[3] < 100.0

    def test_length_preserved(self):
        arr = np.linspace(0.0, 1.0, 20)
        result = smooth_interpolate(arr, window=5)
        assert len(result) == 20

    def test_dtype_float64(self):
        result = smooth_interpolate(np.array([1, 2, 3]))
        assert result.dtype == np.float64

    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            smooth_interpolate(np.zeros((3, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            smooth_interpolate(np.array([]))

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_interpolate(np.array([1.0, 2.0]), window=0)

    def test_constant_array_unchanged(self):
        arr = np.full(10, 5.0)
        result = smooth_interpolate(arr, window=3)
        np.testing.assert_allclose(result, arr)


# ─── batch_resample ───────────────────────────────────────────────────────────

class TestBatchResample:
    def test_basic_two_arrays(self):
        arrays = [np.array([0.0, 1.0]), np.array([0.0, 2.0, 4.0])]
        results = batch_resample(arrays, 5)
        assert len(results) == 2
        assert all(len(r) == 5 for r in results)

    def test_single_array(self):
        results = batch_resample([np.array([1.0, 2.0, 3.0])], 10)
        assert len(results) == 1
        assert len(results[0]) == 10

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            batch_resample([], 5)

    def test_target_len_zero_raises(self):
        with pytest.raises(ValueError):
            batch_resample([np.array([1.0, 2.0])], 0)

    def test_all_float64(self):
        arrays = [np.array([1, 2, 3])]
        results = batch_resample(arrays, 5)
        assert results[0].dtype == np.float64

    def test_with_nearest_cfg(self):
        cfg = InterpolationConfig(method="nearest")
        results = batch_resample([np.array([0.0, 1.0])], 3, cfg)
        assert len(results[0]) == 3

    def test_same_target_len(self):
        arrays = [np.linspace(0, i, 10) for i in range(1, 5)]
        results = batch_resample(arrays, 7)
        for r in results:
            assert len(r) == 7
