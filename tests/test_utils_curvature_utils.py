"""Расширенные тесты для puzzle_reconstruction/utils/curvature_utils.py."""
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
    """S-shaped curve: curvature changes sign at least once."""
    t = np.linspace(-np.pi, np.pi, n)
    x = t
    y = np.sin(t)
    return np.stack([x, y], axis=1)


def _triangle(side: float = 10.0) -> np.ndarray:
    pts = []
    s = 20
    for i in range(s):
        pts.append([i * side / s, 0.0])
    for i in range(s):
        pts.append([side - i * side / s * 0.5, i * side / s])
    for i in range(s):
        pts.append([side * 0.5 - i * side / s * 0.5, side - i * side / s])
    return np.array(pts)


# ─── TestCurvatureConfig ──────────────────────────────────────────────────────

class TestCurvatureConfig:
    def test_defaults(self):
        c = CurvatureConfig()
        assert c.smooth_sigma == pytest.approx(1.0)
        assert c.corner_threshold == pytest.approx(0.1)
        assert c.min_distance == 3

    def test_corner_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            CurvatureConfig(corner_threshold=0.0)

    def test_corner_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            CurvatureConfig(corner_threshold=-0.5)

    def test_min_distance_zero_raises(self):
        with pytest.raises(ValueError):
            CurvatureConfig(min_distance=0)

    def test_min_distance_negative_raises(self):
        with pytest.raises(ValueError):
            CurvatureConfig(min_distance=-1)

    def test_smooth_sigma_zero_ok(self):
        c = CurvatureConfig(smooth_sigma=0.0)
        assert c.smooth_sigma == pytest.approx(0.0)

    def test_custom_values(self):
        c = CurvatureConfig(smooth_sigma=2.0, corner_threshold=0.5, min_distance=5)
        assert c.smooth_sigma == pytest.approx(2.0)
        assert c.corner_threshold == pytest.approx(0.5)
        assert c.min_distance == 5

    def test_min_distance_1_ok(self):
        c = CurvatureConfig(min_distance=1)
        assert c.min_distance == 1


# ─── TestComputeCurvature ─────────────────────────────────────────────────────

class TestComputeCurvature:
    def test_returns_ndarray(self):
        result = compute_curvature(_circle())
        assert isinstance(result, np.ndarray)

    def test_shape_n(self):
        c = _circle(n=64)
        result = compute_curvature(c)
        assert result.shape == (64,)

    def test_dtype_float64(self):
        result = compute_curvature(_circle())
        assert result.dtype == np.float64

    def test_circle_nonzero(self):
        kappa = compute_curvature(_circle(n=64))
        assert not np.allclose(kappa, 0.0)

    def test_line_near_zero(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        kappa = compute_curvature(_line(n=32), cfg)
        # A straight line has zero curvature
        assert np.allclose(np.abs(kappa), 0.0, atol=1e-6)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_curvature(np.ones((10, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            compute_curvature(np.ones(10))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_curvature(np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_accepts_float32(self):
        c = _circle(n=32).astype(np.float32)
        result = compute_curvature(c)
        assert result.dtype == np.float64

    def test_deterministic(self):
        c = _circle(n=64)
        r1 = compute_curvature(c)
        r2 = compute_curvature(c)
        assert np.allclose(r1, r2)

    def test_no_smoothing_config(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        kappa = compute_curvature(_circle(n=64), cfg)
        assert kappa.shape == (64,)

    def test_circle_sign_consistent(self):
        kappa = compute_curvature(_circle(n=64))
        # Circle (counterclockwise) → kappa has consistent sign
        # majority should be positive or negative
        pos_count = np.sum(kappa > 0)
        neg_count = np.sum(kappa < 0)
        assert pos_count > 50 or neg_count > 50  # one sign dominates


# ─── TestComputeTotalCurvature ────────────────────────────────────────────────

class TestComputeTotalCurvature:
    def test_returns_float(self):
        assert isinstance(compute_total_curvature(_circle()), float)

    def test_nonneg(self):
        assert compute_total_curvature(_circle()) >= 0.0

    def test_line_near_zero(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        result = compute_total_curvature(_line(n=32), cfg)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_circle_larger_than_line(self):
        assert compute_total_curvature(_circle()) > compute_total_curvature(_line())

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_total_curvature(np.ones((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_total_curvature(np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_positive_for_closed_curve(self):
        assert compute_total_curvature(_circle(n=128)) > 0.0

    def test_float_type(self):
        result = compute_total_curvature(_circle())
        assert isinstance(result, float)


# ─── TestFindInflectionPoints ─────────────────────────────────────────────────

class TestFindInflectionPoints:
    def test_returns_ndarray(self):
        result = find_inflection_points(_s_curve())
        assert isinstance(result, np.ndarray)

    def test_int64_dtype(self):
        result = find_inflection_points(_s_curve())
        assert result.dtype == np.int64

    def test_s_curve_has_inflections(self):
        result = find_inflection_points(_s_curve(n=128))
        assert len(result) >= 1

    def test_circle_few_inflections(self):
        # A perfect circle has very few or no inflection points
        result = find_inflection_points(_circle(n=128))
        assert len(result) < 10

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            find_inflection_points(np.ones((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            find_inflection_points(np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_indices_in_range(self):
        c = _s_curve(n=64)
        idx = find_inflection_points(c)
        assert np.all(idx >= 0)
        assert np.all(idx < len(c))

    def test_nonneg_length(self):
        result = find_inflection_points(_circle(n=64))
        assert len(result) >= 0


# ─── TestComputeTurningAngle ──────────────────────────────────────────────────

class TestComputeTurningAngle:
    def test_returns_float(self):
        assert isinstance(compute_turning_angle(_circle()), float)

    def test_closed_circle_near_two_pi(self):
        # Full circle turning angle ≈ ±2π
        result = compute_turning_angle(_circle(n=256))
        assert abs(abs(result) - 2 * np.pi) < 0.5

    def test_straight_line_near_zero(self):
        line = _line(n=20)
        result = compute_turning_angle(line)
        assert abs(result) < 0.5

    def test_two_points_ok(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = compute_turning_angle(pts)
        assert isinstance(result, float)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            compute_turning_angle(np.array([[0.0, 0.0]]))

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            compute_turning_angle(np.ones((5, 3)))

    def test_three_points_ok(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = compute_turning_angle(pts)
        assert isinstance(result, float)

    def test_half_circle_near_pi(self):
        t = np.linspace(0, np.pi, 64)
        pts = np.stack([np.cos(t), np.sin(t)], axis=1)
        result = compute_turning_angle(pts)
        assert abs(abs(result) - np.pi) < 0.5


# ─── TestSmoothCurvature ──────────────────────────────────────────────────────

class TestSmoothCurvature:
    def test_returns_ndarray(self):
        kappa = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
        assert isinstance(smooth_curvature(kappa), np.ndarray)

    def test_same_length(self):
        kappa = np.random.randn(20)
        result = smooth_curvature(kappa)
        assert len(result) == len(kappa)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_curvature(np.ones(10), sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            smooth_curvature(np.ones(10), sigma=-1.0)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            smooth_curvature(np.ones((5, 2)))

    def test_float64_output(self):
        kappa = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        assert smooth_curvature(kappa).dtype == np.float64

    def test_smoothing_reduces_max(self):
        kappa = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=float)
        result = smooth_curvature(kappa, sigma=1.0)
        assert result.max() < kappa.max()

    def test_small_sigma_preserves_shape(self):
        kappa = np.ones(10, dtype=float)
        result = smooth_curvature(kappa, sigma=0.1)
        assert np.allclose(result, 1.0, atol=0.01)


# ─── TestCornerScore ──────────────────────────────────────────────────────────

class TestCornerScore:
    def test_returns_ndarray(self):
        assert isinstance(corner_score(_circle()), np.ndarray)

    def test_shape_n(self):
        c = _circle(n=64)
        result = corner_score(c)
        assert result.shape == (64,)

    def test_range_0_to_1(self):
        result = corner_score(_circle(n=64))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_max_is_1(self):
        result = corner_score(_circle(n=64))
        assert result.max() == pytest.approx(1.0, abs=1e-9)

    def test_zero_curvature_all_zeros(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        result = corner_score(_line(n=32), cfg)
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            corner_score(np.ones((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            corner_score(np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_nonneg_all_elements(self):
        result = corner_score(_s_curve())
        assert np.all(result >= 0.0)


# ─── TestFindCorners ──────────────────────────────────────────────────────────

class TestFindCorners:
    def test_returns_ndarray(self):
        assert isinstance(find_corners(_circle()), np.ndarray)

    def test_int64_dtype(self):
        result = find_corners(_circle())
        assert result.dtype == np.int64

    def test_very_high_threshold_empty(self):
        cfg = CurvatureConfig(corner_threshold=1000.0)
        result = find_corners(_circle(n=64), cfg)
        assert len(result) == 0

    def test_low_threshold_finds_corners(self):
        cfg = CurvatureConfig(corner_threshold=1e-6)
        result = find_corners(_circle(n=64), cfg)
        assert len(result) > 0

    def test_indices_in_range(self):
        c = _circle(n=64)
        result = find_corners(c)
        assert np.all(result >= 0)
        assert np.all(result < len(c))

    def test_min_distance_enforced(self):
        cfg = CurvatureConfig(corner_threshold=1e-6, min_distance=5)
        c = _circle(n=64)
        result = find_corners(c, cfg)
        if len(result) > 1:
            diffs = np.diff(result)
            assert np.all(diffs >= 5)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            find_corners(np.ones((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            find_corners(np.array([[0.0, 0.0], [1.0, 0.0]]))


# ─── TestBatchCurvature ───────────────────────────────────────────────────────

class TestBatchCurvature:
    def test_returns_list(self):
        curves = [_circle(), _line()]
        assert isinstance(batch_curvature(curves), list)

    def test_length_matches(self):
        curves = [_circle(), _line(), _s_curve()]
        assert len(batch_curvature(curves)) == 3

    def test_each_is_ndarray(self):
        curves = [_circle(n=32), _line(n=32)]
        for result in batch_curvature(curves):
            assert isinstance(result, np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_curvature([])

    def test_shapes_match_inputs(self):
        c1 = _circle(n=32)
        c2 = _circle(n=64)
        results = batch_curvature([c1, c2])
        assert results[0].shape == (32,)
        assert results[1].shape == (64,)

    def test_all_float64(self):
        for r in batch_curvature([_circle(), _line()]):
            assert r.dtype == np.float64

    def test_cfg_passed_through(self):
        cfg = CurvatureConfig(smooth_sigma=0.0)
        results = batch_curvature([_line(n=32)], cfg)
        assert np.allclose(np.abs(results[0]), 0.0, atol=1e-6)
