"""Tests for puzzle_reconstruction.algorithms.contour_smoother."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.contour_smoother import (
    SmoothedContour,
    SmootherConfig,
    align_contours,
    batch_smooth,
    compute_arc_length,
    contour_similarity,
    resample_contour,
    smooth_and_resample,
    smooth_gaussian,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _line(n=10, length=100.0):
    """Horizontal line from (0,0) to (length,0)."""
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _circle(n=32, r=50.0):
    """Discrete circle points."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


# ─── TestSmootherConfig ───────────────────────────────────────────────────────

class TestSmootherConfig:
    def test_defaults(self):
        cfg = SmootherConfig()
        assert cfg.sigma == pytest.approx(1.0)
        assert cfg.n_points == 64
        assert cfg.closed is False

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SmootherConfig(sigma=-0.1)

    def test_n_points_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            SmootherConfig(n_points=1)

    def test_sigma_zero_allowed(self):
        cfg = SmootherConfig(sigma=0.0)
        assert cfg.sigma == pytest.approx(0.0)

    def test_closed_flag(self):
        cfg = SmootherConfig(closed=True)
        assert cfg.closed is True


# ─── TestSmoothedContour ──────────────────────────────────────────────────────

class TestSmoothedContour:
    def test_basic_creation(self):
        pts = _line(10)
        sc = SmoothedContour(points=pts, original_n=20, method="gaussian")
        assert sc.original_n == 20
        assert sc.method == "gaussian"

    def test_negative_original_n_raises(self):
        pts = _line(5)
        with pytest.raises(ValueError, match="original_n"):
            SmoothedContour(points=pts, original_n=-1, method="none")

    def test_n_points_property(self):
        pts = _line(15)
        sc = SmoothedContour(points=pts, original_n=15, method="none")
        assert sc.n_points == 15

    def test_length_property(self):
        pts = _line(n=5, length=100.0)
        sc = SmoothedContour(points=pts, original_n=5, method="none",
                             params={"closed": False})
        assert sc.length == pytest.approx(100.0, rel=0.01)

    def test_is_closed_property(self):
        pts = _line(5)
        sc = SmoothedContour(points=pts, original_n=5, method="none",
                             params={"closed": True})
        assert sc.is_closed is True

    def test_params_default_empty(self):
        pts = _line(5)
        sc = SmoothedContour(points=pts, original_n=5, method="none")
        assert sc.params == {}


# ─── TestComputeArcLength ─────────────────────────────────────────────────────

class TestComputeArcLength:
    def test_single_point_returns_zero(self):
        pts = np.array([[0.0, 0.0]])
        assert compute_arc_length(pts) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_arc_length(np.zeros((0, 2)))

    def test_line_length(self):
        pts = _line(n=5, length=100.0)
        assert compute_arc_length(pts) == pytest.approx(100.0, rel=1e-6)

    def test_closed_adds_segment(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        open_len = compute_arc_length(pts, closed=False)
        closed_len = compute_arc_length(pts, closed=True)
        # Closing segment: from [10,10] to [0,0] = sqrt(200)
        assert closed_len > open_len

    def test_nx1x2_shape_accepted(self):
        pts = _line(5).reshape(-1, 1, 2)
        assert compute_arc_length(pts) > 0.0

    def test_circle_perimeter(self):
        n = 360
        r = 50.0
        pts = _circle(n, r)
        expected = 2 * np.pi * r
        # Fine approximation for many points
        assert compute_arc_length(pts, closed=True) == pytest.approx(
            expected, rel=0.02
        )


# ─── TestSmoothGaussian ───────────────────────────────────────────────────────

class TestSmoothGaussian:
    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            smooth_gaussian(_line(5), sigma=-1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_gaussian(np.zeros((0, 2)), sigma=1.0)

    def test_zero_sigma_returns_copy(self):
        pts = _line(10)
        result = smooth_gaussian(pts, sigma=0.0)
        np.testing.assert_array_almost_equal(result, pts)

    def test_output_shape_preserved(self):
        pts = _line(20)
        result = smooth_gaussian(pts, sigma=2.0)
        assert result.shape == pts.shape

    def test_output_dtype_float64(self):
        pts = _line(10)
        result = smooth_gaussian(pts, sigma=1.0)
        assert result.dtype == np.float64

    def test_smoothing_reduces_variance(self):
        """Adding noise then smoothing should reduce point-to-point variation."""
        rng = np.random.default_rng(0)
        pts = _circle(64) + rng.normal(0, 5.0, (64, 2))
        smooth = smooth_gaussian(pts, sigma=3.0)
        orig_var = np.var(np.diff(pts, axis=0))
        smooth_var = np.var(np.diff(smooth, axis=0))
        assert smooth_var < orig_var


# ─── TestResampleContour ──────────────────────────────────────────────────────

class TestResampleContour:
    def test_n_points_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            resample_contour(_line(5), n_points=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.zeros((0, 2)), n_points=4)

    def test_single_point_tiles(self):
        pts = np.array([[3.0, 7.0]])
        result = resample_contour(pts, n_points=5)
        assert result.shape == (5, 2)
        np.testing.assert_array_almost_equal(result[0], [3.0, 7.0])

    def test_output_n_points(self):
        pts = _line(20, 100.0)
        result = resample_contour(pts, n_points=10)
        assert result.shape == (10, 2)

    def test_line_endpoints_preserved(self):
        pts = _line(20, 100.0)
        result = resample_contour(pts, n_points=8)
        assert result[0, 0] == pytest.approx(pts[0, 0], abs=0.01)
        assert result[-1, 0] == pytest.approx(pts[-1, 0], abs=0.01)

    def test_output_dtype_float64(self):
        result = resample_contour(_line(10), n_points=8)
        assert result.dtype == np.float64


# ─── TestSmoothAndResample ────────────────────────────────────────────────────

class TestSmoothAndResample:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_and_resample(np.zeros((0, 2)))

    def test_returns_smoothed_contour(self):
        pts = _line(20)
        result = smooth_and_resample(pts)
        assert isinstance(result, SmoothedContour)

    def test_output_n_points_from_config(self):
        pts = _line(20)
        cfg = SmootherConfig(n_points=32)
        result = smooth_and_resample(pts, cfg)
        assert result.n_points == 32

    def test_method_gaussian_when_sigma_positive(self):
        pts = _line(20)
        cfg = SmootherConfig(sigma=2.0)
        result = smooth_and_resample(pts, cfg)
        assert result.method == "gaussian"

    def test_method_none_when_sigma_zero(self):
        pts = _line(20)
        cfg = SmootherConfig(sigma=0.0)
        result = smooth_and_resample(pts, cfg)
        assert result.method == "none"

    def test_original_n_stored(self):
        pts = _line(15)
        result = smooth_and_resample(pts)
        assert result.original_n == 15


# ─── TestAlignContours ────────────────────────────────────────────────────────

class TestAlignContours:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            align_contours(np.zeros((0, 2)), np.zeros((0, 2)))

    def test_length_mismatch_raises(self):
        c1 = _circle(10)
        c2 = _circle(8)
        with pytest.raises(ValueError):
            align_contours(c1, c2)

    def test_returns_tuple_of_two(self):
        c1 = _circle(8)
        c2 = _circle(8)
        result = align_contours(c1, c2)
        assert len(result) == 2

    def test_identical_contours_zero_shift(self):
        c1 = _circle(8)
        _, c2_aligned = align_contours(c1, c1.copy())
        # Already aligned, so should be same
        np.testing.assert_array_almost_equal(c2_aligned, c1, decimal=10)

    def test_shifted_contour_aligned(self):
        c1 = _circle(16)
        shift = 4
        c2 = np.roll(c1, shift, axis=0)
        _, c2_aligned = align_contours(c1, c2)
        # After alignment, should match well
        residual = float(np.mean(np.linalg.norm(c1 - c2_aligned, axis=1)))
        assert residual < 5.0  # small after alignment


# ─── TestContourSimilarity ────────────────────────────────────────────────────

class TestContourSimilarity:
    def test_unknown_metric_raises(self):
        c = _line(10)
        with pytest.raises(ValueError, match="metric"):
            contour_similarity(c, c, metric="unknown")

    def test_empty_raises(self):
        c = _line(5)
        with pytest.raises(ValueError):
            contour_similarity(np.zeros((0, 2)), c)

    def test_identical_contours_high_similarity_l2(self):
        c = _line(20)
        sim = contour_similarity(c, c.copy(), metric="l2")
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_identical_contours_high_similarity_hausdorff(self):
        c = _line(20)
        sim = contour_similarity(c, c.copy(), metric="hausdorff")
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_returns_float_in_0_1(self):
        c1 = _circle(20)
        c2 = _line(20) + 100.0
        sim = contour_similarity(c1, c2)
        assert 0.0 <= sim <= 1.0

    def test_different_lengths_accepted(self):
        c1 = _line(10)
        c2 = _line(20)
        sim = contour_similarity(c1, c2, metric="l2")
        assert 0.0 <= sim <= 1.0


# ─── TestBatchSmooth ──────────────────────────────────────────────────────────

class TestBatchSmooth:
    def test_returns_list_of_smoothed_contours(self):
        pts_list = [_line(15), _circle(20), _line(10)]
        result = batch_smooth(pts_list)
        assert len(result) == 3
        assert all(isinstance(sc, SmoothedContour) for sc in result)

    def test_empty_list(self):
        assert batch_smooth([]) == []

    def test_config_applied(self):
        pts_list = [_line(20), _circle(32)]
        cfg = SmootherConfig(n_points=16)
        result = batch_smooth(pts_list, cfg)
        for sc in result:
            assert sc.n_points == 16
