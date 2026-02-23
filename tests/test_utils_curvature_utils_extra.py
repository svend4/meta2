"""Extra tests for puzzle_reconstruction/utils/curvature_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.curvature_utils import (
    CurvatureConfig,
    batch_curvature,
    compute_curvature,
    compute_total_curvature,
    compute_turning_angle,
    corner_score,
    find_corners,
    find_inflection_points,
    smooth_curvature,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 10.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n: int = 32) -> np.ndarray:
    xs = np.linspace(0.0, 10.0, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _s_curve(n: int = 64) -> np.ndarray:
    t = np.linspace(-np.pi, np.pi, n)
    return np.stack([t, np.sin(t)], axis=1)


def _triangle() -> np.ndarray:
    pts = []
    s = 20
    side = 10.0
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side - i * side / s * 0.5, i * side / s])
    for i in range(s):
        pts.append([side * 0.5 - i * side / s * 0.5, side - i * side / s])
    return np.array(pts, dtype=np.float64)


def _half_circle(n: int = 64) -> np.ndarray:
    t = np.linspace(0, np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


# ─── CurvatureConfig (extra) ──────────────────────────────────────────────────

class TestCurvatureConfigExtra:
    def test_large_smooth_sigma_ok(self):
        cfg = CurvatureConfig(smooth_sigma=50.0)
        assert cfg.smooth_sigma == pytest.approx(50.0)

    def test_small_corner_threshold_positive_ok(self):
        cfg = CurvatureConfig(corner_threshold=1e-10)
        assert cfg.corner_threshold == pytest.approx(1e-10)

    def test_large_min_distance_ok(self):
        cfg = CurvatureConfig(min_distance=100)
        assert cfg.min_distance == 100

    def test_independent_instances(self):
        c1 = CurvatureConfig(smooth_sigma=1.0)
        c2 = CurvatureConfig(smooth_sigma=5.0)
        assert c1.smooth_sigma != c2.smooth_sigma

    def test_corner_threshold_1_ok(self):
        cfg = CurvatureConfig(corner_threshold=1.0)
        assert cfg.corner_threshold == pytest.approx(1.0)

    def test_corner_threshold_large_ok(self):
        cfg = CurvatureConfig(corner_threshold=100.0)
        assert cfg.corner_threshold == pytest.approx(100.0)

    def test_min_distance_large_int(self):
        cfg = CurvatureConfig(min_distance=50)
        assert cfg.min_distance == 50


# ─── compute_curvature (extra) ────────────────────────────────────────────────

class TestComputeCurvatureExtra:
    def test_output_finite(self):
        result = compute_curvature(_circle(n=64))
        assert np.all(np.isfinite(result))

    def test_s_curve_has_positive_and_negative(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        kappa = compute_curvature(_s_curve(n=128), cfg)
        assert np.any(kappa > 0.0)
        assert np.any(kappa < 0.0)

    def test_larger_circle_nonzero(self):
        result = compute_curvature(_circle(n=64, r=100.0))
        assert not np.allclose(result, 0.0)

    def test_larger_sigma_produces_float64(self):
        cfg = CurvatureConfig(smooth_sigma=2.0)
        result = compute_curvature(_circle(n=64), cfg)
        assert result.dtype == np.float64

    def test_shape_matches_input_n(self):
        for n in (10, 32, 100):
            c = _circle(n=n)
            assert compute_curvature(c).shape == (n,)

    def test_half_circle_nonzero(self):
        result = compute_curvature(_half_circle(n=64))
        assert not np.allclose(result, 0.0)

    def test_circle_curvature_near_constant(self):
        # Circle curvature should be approximately uniform
        cfg = CurvatureConfig(smooth_sigma=0.0)
        kappa = compute_curvature(_circle(n=128), cfg)
        abs_kappa = np.abs(kappa)
        # std/mean should be small relative to mean
        assert abs_kappa.std() < abs_kappa.mean() * 0.1 + 1e-9


# ─── compute_total_curvature (extra) ──────────────────────────────────────────

class TestComputeTotalCurvatureExtra:
    def test_s_curve_positive(self):
        result = compute_total_curvature(_s_curve(n=128))
        assert result > 0.0

    def test_triangle_positive(self):
        result = compute_total_curvature(_triangle())
        assert result > 0.0

    def test_finite_value(self):
        result = compute_total_curvature(_circle(n=64))
        assert np.isfinite(result)

    def test_half_circle_positive(self):
        result = compute_total_curvature(_half_circle(n=64))
        assert result > 0.0

    def test_cfg_smooth_sigma_zero_ok(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        result = compute_total_curvature(_circle(n=32), cfg)
        assert isinstance(result, float)

    def test_large_circle_positive(self):
        result = compute_total_curvature(_circle(n=64, r=1000.0))
        assert result > 0.0


# ─── compute_turning_angle (extra) ────────────────────────────────────────────

class TestComputeTurningAngleExtra:
    def test_result_finite(self):
        result = compute_turning_angle(_circle(n=64))
        assert np.isfinite(result)

    def test_half_circle_near_pi(self):
        result = compute_turning_angle(_half_circle(n=128))
        assert abs(abs(result) - np.pi) < 0.5

    def test_result_float_type(self):
        result = compute_turning_angle(_s_curve())
        assert isinstance(result, float)

    def test_many_points_circle_accurate(self):
        result = compute_turning_angle(_circle(n=512))
        assert abs(abs(result) - 2 * np.pi) < 0.1

    def test_quarter_circle_near_half_pi(self):
        t = np.linspace(0, np.pi / 2, 64)
        pts = np.stack([np.cos(t), np.sin(t)], axis=1)
        result = compute_turning_angle(pts)
        assert abs(abs(result) - np.pi / 2) < 0.5

    def test_triangle_positive(self):
        result = compute_turning_angle(_triangle())
        assert isinstance(result, float)


# ─── smooth_curvature (extra) ─────────────────────────────────────────────────

class TestSmoothCurvatureExtra:
    def test_larger_sigma_smaller_variance(self):
        kappa = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                         dtype=float)
        r1 = smooth_curvature(kappa, sigma=0.5)
        r2 = smooth_curvature(kappa, sigma=2.0)
        assert r2.var() <= r1.var() + 1e-9

    def test_large_sigma_ok(self):
        kappa = np.ones(20, dtype=float)
        result = smooth_curvature(kappa, sigma=10.0)
        assert result.shape == (20,)

    def test_input_not_modified(self):
        kappa = np.array([1.0, 2.0, 5.0, 2.0, 1.0])
        original = kappa.copy()
        smooth_curvature(kappa, sigma=1.0)
        np.testing.assert_array_equal(kappa, original)

    def test_spike_reduced(self):
        kappa = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], dtype=float)
        result = smooth_curvature(kappa, sigma=1.0)
        assert result.max() < 100.0

    def test_output_finite(self):
        kappa = np.random.randn(30)
        result = smooth_curvature(kappa, sigma=1.0)
        assert np.all(np.isfinite(result))

    def test_small_sigma_close_to_input(self):
        kappa = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        result = smooth_curvature(kappa, sigma=0.01)
        np.testing.assert_allclose(result, kappa, atol=0.1)


# ─── corner_score (extra) ─────────────────────────────────────────────────────

class TestCornerScoreExtra:
    def test_s_curve_nonneg_all(self):
        result = corner_score(_s_curve(n=64))
        assert np.all(result >= 0.0)

    def test_s_curve_le_one(self):
        result = corner_score(_s_curve(n=64))
        assert np.all(result <= 1.0)

    def test_finite_values(self):
        result = corner_score(_circle(n=64))
        assert np.all(np.isfinite(result))

    def test_shape_matches_n(self):
        for n in (10, 32, 64):
            c = _circle(n=n)
            assert corner_score(c).shape == (n,)

    def test_float64_output(self):
        result = corner_score(_circle(n=32))
        assert result.dtype == np.float64

    def test_triangle_scores_sum_positive(self):
        result = corner_score(_triangle())
        assert result.sum() > 0.0


# ─── find_corners (extra) ─────────────────────────────────────────────────────

class TestFindCornersExtra:
    def test_sorted_indices(self):
        cfg = CurvatureConfig(corner_threshold=1e-6)
        c = _circle(n=64)
        result = find_corners(c, cfg)
        if len(result) > 1:
            assert np.all(np.diff(result) >= 0)

    def test_higher_threshold_fewer_corners(self):
        c = _triangle()
        cfg_low = CurvatureConfig(corner_threshold=0.01)
        cfg_high = CurvatureConfig(corner_threshold=10.0)
        n_low = len(find_corners(c, cfg_low))
        n_high = len(find_corners(c, cfg_high))
        assert n_low >= n_high

    def test_line_no_corners(self):
        cfg = CurvatureConfig(corner_threshold=1.0)
        result = find_corners(_line(n=32), cfg)
        assert len(result) == 0

    def test_indices_nonneg(self):
        cfg = CurvatureConfig(corner_threshold=1e-6)
        result = find_corners(_circle(n=64), cfg)
        assert np.all(result >= 0)

    def test_indices_lt_n(self):
        n = 64
        cfg = CurvatureConfig(corner_threshold=1e-6)
        result = find_corners(_circle(n=n), cfg)
        assert np.all(result < n)

    def test_result_dtype_int64(self):
        result = find_corners(_circle(n=64))
        assert result.dtype == np.int64


# ─── find_inflection_points (extra) ───────────────────────────────────────────

class TestFindInflectionPointsExtra:
    def test_sorted_indices(self):
        c = _s_curve(n=128)
        result = find_inflection_points(c)
        if len(result) > 1:
            assert np.all(np.diff(result) >= 0)

    def test_line_few_inflections(self):
        result = find_inflection_points(_line(n=64))
        assert len(result) < 5

    def test_large_n_ok(self):
        c = _s_curve(n=256)
        result = find_inflection_points(c)
        assert isinstance(result, np.ndarray)

    def test_indices_nonneg(self):
        result = find_inflection_points(_s_curve(n=64))
        assert np.all(result >= 0)

    def test_indices_lt_n(self):
        n = 64
        result = find_inflection_points(_s_curve(n=n))
        assert np.all(result < n)

    def test_finite_length_result(self):
        result = find_inflection_points(_circle(n=64))
        assert len(result) < 64


# ─── batch_curvature (extra) ──────────────────────────────────────────────────

class TestBatchCurvatureExtra:
    def test_single_curve_list(self):
        result = batch_curvature([_circle(n=32)])
        assert len(result) == 1

    def test_consistent_with_individual(self):
        c = _circle(n=32)
        batch = batch_curvature([c])
        individual = compute_curvature(c)
        np.testing.assert_allclose(batch[0], individual)

    def test_different_lengths(self):
        curves = [_circle(n=10), _circle(n=64), _line(n=32)]
        results = batch_curvature(curves)
        assert results[0].shape == (10,)
        assert results[1].shape == (64,)
        assert results[2].shape == (32,)

    def test_large_batch(self):
        curves = [_circle(n=32) for _ in range(10)]
        result = batch_curvature(curves)
        assert len(result) == 10

    def test_all_output_finite(self):
        curves = [_circle(n=32), _line(n=32), _s_curve(n=32)]
        for r in batch_curvature(curves):
            assert np.all(np.isfinite(r))

    def test_mixed_curves(self):
        result = batch_curvature([_circle(n=16), _s_curve(n=16), _line(n=16)])
        assert len(result) == 3
        for r in result:
            assert isinstance(r, np.ndarray)
