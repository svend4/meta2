"""Extra tests for puzzle_reconstruction/utils/interpolation_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sig(n=20) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _grid(h=8, w=8) -> np.ndarray:
    return np.arange(h * w, dtype=np.float64).reshape(h, w)


# ─── InterpolationConfig ──────────────────────────────────────────────────────

class TestInterpolationConfigExtra:
    def test_default_method(self):
        assert InterpolationConfig().method == "linear"

    def test_default_clamp(self):
        assert InterpolationConfig().clamp is True

    def test_default_fill_val(self):
        assert InterpolationConfig().fill_val == pytest.approx(0.0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="cubic")

    def test_valid_nearest(self):
        cfg = InterpolationConfig(method="nearest")
        assert cfg.method == "nearest"

    def test_valid_linear(self):
        cfg = InterpolationConfig(method="linear")
        assert cfg.method == "linear"


# ─── lerp ─────────────────────────────────────────────────────────────────────

class TestLerpExtra:
    def test_returns_float(self):
        assert isinstance(lerp(0.0, 1.0, 0.5), float)

    def test_t_zero_returns_a(self):
        assert lerp(3.0, 7.0, 0.0) == pytest.approx(3.0)

    def test_t_one_returns_b(self):
        assert lerp(3.0, 7.0, 1.0) == pytest.approx(7.0)

    def test_t_half(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)

    def test_t_negative_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, -0.1)

    def test_t_gt_one_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, 1.1)

    def test_negative_values(self):
        assert lerp(-10.0, 10.0, 0.5) == pytest.approx(0.0)

    def test_same_a_b(self):
        assert lerp(5.0, 5.0, 0.7) == pytest.approx(5.0)


# ─── lerp_array ───────────────────────────────────────────────────────────────

class TestLerpArrayExtra:
    def test_returns_ndarray(self):
        a = np.zeros(5)
        b = np.ones(5)
        assert isinstance(lerp_array(a, b, 0.5), np.ndarray)

    def test_t_zero_returns_a(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(lerp_array(a, b, 0.0), a)

    def test_t_one_returns_b(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        np.testing.assert_allclose(lerp_array(a, b, 1.0), b)

    def test_t_half(self):
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 4.0])
        np.testing.assert_allclose(lerp_array(a, b, 0.5), [1.0, 2.0])

    def test_t_negative_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.zeros(3), np.ones(3), -0.1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            lerp_array(np.zeros(3), np.zeros(5), 0.5)

    def test_dtype_float64(self):
        out = lerp_array(np.zeros(3, dtype=np.uint8), np.ones(3, dtype=np.uint8), 0.5)
        assert out.dtype == np.float64

    def test_2d_arrays(self):
        a = np.zeros((2, 2))
        b = np.ones((2, 2))
        out = lerp_array(a, b, 0.5)
        assert out.shape == (2, 2)
        assert np.allclose(out, 0.5)


# ─── bilinear_interpolate ─────────────────────────────────────────────────────

class TestBilinearInterpolateExtra:
    def test_returns_float(self):
        g = _grid(4, 4)
        assert isinstance(bilinear_interpolate(g, 0.0, 0.0), float)

    def test_integer_coord_exact(self):
        g = _grid(4, 4)
        assert bilinear_interpolate(g, 0.0, 0.0) == pytest.approx(float(g[0, 0]))

    def test_integer_coord_corner(self):
        g = _grid(4, 4)
        assert bilinear_interpolate(g, 3.0, 3.0) == pytest.approx(float(g[3, 3]))

    def test_midpoint_average(self):
        g = np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float64)
        # row=1 col=0.5 → midpoint between g[1,0]=0 and g[1,1]=4
        val = bilinear_interpolate(g, 1.0, 0.5)
        assert val == pytest.approx(2.0)

    def test_out_of_bounds_raises(self):
        g = _grid(4, 4)
        with pytest.raises(ValueError):
            bilinear_interpolate(g, 5.0, 0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            bilinear_interpolate(np.zeros((2, 2, 2)), 0.0, 0.0)

    def test_negative_coord_raises(self):
        g = _grid(4, 4)
        with pytest.raises(ValueError):
            bilinear_interpolate(g, -1.0, 0.0)


# ─── resample_1d ──────────────────────────────────────────────────────────────

class TestResample1dExtra:
    def test_returns_ndarray(self):
        assert isinstance(resample_1d(_sig(), 10), np.ndarray)

    def test_length_equals_target(self):
        assert len(resample_1d(_sig(20), 10)) == 10

    def test_dtype_float64(self):
        assert resample_1d(_sig(), 10).dtype == np.float64

    def test_same_length_identity(self):
        s = _sig(10)
        out = resample_1d(s, 10)
        np.testing.assert_allclose(out, s)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            resample_1d(np.zeros((2, 3)), 5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_1d(np.array([]), 5)

    def test_target_lt_1_raises(self):
        with pytest.raises(ValueError):
            resample_1d(_sig(), 0)

    def test_nearest_method(self):
        cfg = InterpolationConfig(method="nearest")
        out = resample_1d(np.array([0.0, 1.0, 2.0]), 5, cfg)
        assert len(out) == 5

    def test_upsample_linear(self):
        s = np.array([0.0, 10.0])
        out = resample_1d(s, 5)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(10.0)


# ─── fill_missing ─────────────────────────────────────────────────────────────

class TestFillMissingExtra:
    def test_returns_ndarray(self):
        assert isinstance(fill_missing(np.array([1.0, np.nan, 3.0])), np.ndarray)

    def test_no_nan_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(fill_missing(a), a)

    def test_fills_middle_nan(self):
        a = np.array([0.0, np.nan, 4.0])
        out = fill_missing(a)
        assert out[1] == pytest.approx(2.0)

    def test_no_nan_in_result(self):
        a = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
        out = fill_missing(a)
        assert not np.any(np.isnan(out))

    def test_all_nan_returns_zeros(self):
        a = np.array([np.nan, np.nan])
        out = fill_missing(a)
        assert np.allclose(out, 0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            fill_missing(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fill_missing(np.array([]))

    def test_dtype_float64(self):
        a = np.array([1.0, np.nan, 3.0])
        assert fill_missing(a).dtype == np.float64


# ─── interpolate_scores ───────────────────────────────────────────────────────

class TestInterpolateScoresExtra:
    def test_returns_ndarray(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(interpolate_scores(M), np.ndarray)

    def test_alpha_one_is_original(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(interpolate_scores(M, alpha=1.0), M)

    def test_alpha_zero_is_transpose(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(interpolate_scores(M, alpha=0.0), M.T)

    def test_alpha_half_is_symmetric(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = interpolate_scores(M, alpha=0.5)
        np.testing.assert_allclose(out, out.T)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.zeros((2, 3)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            interpolate_scores(np.zeros((2, 2, 2)))

    def test_alpha_out_of_range_raises(self):
        M = np.eye(2)
        with pytest.raises(ValueError):
            interpolate_scores(M, alpha=1.5)

    def test_shape_preserved(self):
        M = np.eye(4)
        out = interpolate_scores(M)
        assert out.shape == (4, 4)


# ─── smooth_interpolate ───────────────────────────────────────────────────────

class TestSmoothInterpolateExtra:
    def test_returns_ndarray(self):
        assert isinstance(smooth_interpolate(_sig()), np.ndarray)

    def test_length_preserved(self):
        s = _sig(15)
        out = smooth_interpolate(s, window=3)
        assert len(out) == 15

    def test_dtype_float64(self):
        assert smooth_interpolate(_sig()).dtype == np.float64

    def test_window_one_identity(self):
        s = _sig(10)
        np.testing.assert_allclose(smooth_interpolate(s, window=1), s)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            smooth_interpolate(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_interpolate(np.array([]))

    def test_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            smooth_interpolate(_sig(), window=0)

    def test_smoothed_less_extreme(self):
        a = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        out = smooth_interpolate(a, window=3)
        assert out.max() <= a.max()


# ─── batch_resample ───────────────────────────────────────────────────────────

class TestBatchResampleExtra:
    def test_returns_list(self):
        result = batch_resample([_sig()], 10)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        result = batch_resample([_sig(), _sig(15)], 8)
        assert len(result) == 2

    def test_each_element_length(self):
        result = batch_resample([_sig(5), _sig(20)], 10)
        for out in result:
            assert len(out) == 10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_resample([], 5)

    def test_target_lt_1_raises(self):
        with pytest.raises(ValueError):
            batch_resample([_sig()], 0)

    def test_none_cfg(self):
        result = batch_resample([_sig()], 5, cfg=None)
        assert len(result) == 1

    def test_nearest_method(self):
        cfg = InterpolationConfig(method="nearest")
        result = batch_resample([np.array([0.0, 1.0, 2.0])], 5, cfg)
        assert len(result[0]) == 5
