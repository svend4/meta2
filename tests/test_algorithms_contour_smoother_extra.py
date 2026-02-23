"""Extra tests for puzzle_reconstruction.algorithms.contour_smoother."""
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


def _line(n=10, length=100.0):
    xs = np.linspace(0.0, length, n)
    return np.stack([xs, np.zeros(n)], axis=1)


def _circle(n=32, r=50.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


# ─── TestSmootherConfigExtra ──────────────────────────────────────────────────

class TestSmootherConfigExtra:
    def test_defaults(self):
        c = SmootherConfig()
        assert c.sigma == pytest.approx(1.0)
        assert c.n_points == 64
        assert c.closed is False

    def test_neg_sigma_raises(self):
        with pytest.raises(ValueError):
            SmootherConfig(sigma=-0.1)

    def test_n_points_1_raises(self):
        with pytest.raises(ValueError):
            SmootherConfig(n_points=1)

    def test_sigma_zero(self):
        assert SmootherConfig(sigma=0.0).sigma == pytest.approx(0.0)

    def test_closed_true(self):
        assert SmootherConfig(closed=True).closed is True

    def test_large_sigma(self):
        assert SmootherConfig(sigma=100.0).sigma == pytest.approx(100.0)

    def test_n_points_2(self):
        assert SmootherConfig(n_points=2).n_points == 2

    def test_n_points_1000(self):
        assert SmootherConfig(n_points=1000).n_points == 1000


# ─── TestSmoothedContourExtra ─────────────────────────────────────────────────

class TestSmoothedContourExtra:
    def test_creation(self):
        sc = SmoothedContour(points=_line(10), original_n=20, method="gaussian")
        assert sc.original_n == 20

    def test_neg_original_n_raises(self):
        with pytest.raises(ValueError):
            SmoothedContour(points=_line(5), original_n=-1, method="none")

    def test_n_points_prop(self):
        sc = SmoothedContour(points=_line(15), original_n=15, method="none")
        assert sc.n_points == 15

    def test_length_prop(self):
        sc = SmoothedContour(points=_line(5, 100.0), original_n=5,
                             method="none", params={"closed": False})
        assert sc.length == pytest.approx(100.0, rel=0.01)

    def test_is_closed_prop(self):
        sc = SmoothedContour(points=_line(5), original_n=5,
                             method="none", params={"closed": True})
        assert sc.is_closed is True

    def test_params_default(self):
        sc = SmoothedContour(points=_line(5), original_n=5, method="none")
        assert sc.params == {}

    def test_method_stored(self):
        sc = SmoothedContour(points=_line(5), original_n=5, method="gaussian")
        assert sc.method == "gaussian"


# ─── TestComputeArcLengthExtra ────────────────────────────────────────────────

class TestComputeArcLengthExtra:
    def test_single_point_zero(self):
        assert compute_arc_length(np.array([[0.0, 0.0]])) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_arc_length(np.zeros((0, 2)))

    def test_line_length(self):
        assert compute_arc_length(_line(5, 100.0)) == pytest.approx(100.0, rel=1e-6)

    def test_closed_gt_open(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        assert compute_arc_length(pts, closed=True) > compute_arc_length(pts, closed=False)

    def test_nx1x2_accepted(self):
        assert compute_arc_length(_line(5).reshape(-1, 1, 2)) > 0.0

    def test_circle_perimeter(self):
        n, r = 360, 50.0
        expected = 2 * np.pi * r
        assert compute_arc_length(_circle(n, r), closed=True) == pytest.approx(
            expected, rel=0.02)

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert compute_arc_length(pts) == pytest.approx(5.0)


# ─── TestSmoothGaussianExtra ─────────────────────────────────────────────────

class TestSmoothGaussianExtra:
    def test_neg_sigma_raises(self):
        with pytest.raises(ValueError):
            smooth_gaussian(_line(5), sigma=-1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_gaussian(np.zeros((0, 2)), sigma=1.0)

    def test_zero_sigma_copy(self):
        pts = _line(10)
        np.testing.assert_array_almost_equal(smooth_gaussian(pts, sigma=0.0), pts)

    def test_shape_preserved(self):
        pts = _line(20)
        assert smooth_gaussian(pts, sigma=2.0).shape == pts.shape

    def test_dtype(self):
        assert smooth_gaussian(_line(10), sigma=1.0).dtype == np.float64

    def test_reduces_variance(self):
        rng = np.random.default_rng(0)
        pts = _circle(64) + rng.normal(0, 5.0, (64, 2))
        smooth = smooth_gaussian(pts, sigma=3.0)
        assert np.var(np.diff(smooth, axis=0)) < np.var(np.diff(pts, axis=0))

    def test_large_sigma(self):
        result = smooth_gaussian(_line(20), sigma=50.0)
        assert result.shape == (20, 2)


# ─── TestResampleContourExtra ─────────────────────────────────────────────────

class TestResampleContourExtra:
    def test_n_1_raises(self):
        with pytest.raises(ValueError):
            resample_contour(_line(5), n_points=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.zeros((0, 2)), n_points=4)

    def test_single_point_tiles(self):
        pts = np.array([[3.0, 7.0]])
        result = resample_contour(pts, n_points=5)
        assert result.shape == (5, 2)

    def test_output_n(self):
        assert resample_contour(_line(20, 100.0), n_points=10).shape == (10, 2)

    def test_endpoints_preserved(self):
        pts = _line(20, 100.0)
        result = resample_contour(pts, n_points=8)
        assert result[0, 0] == pytest.approx(pts[0, 0], abs=0.01)
        assert result[-1, 0] == pytest.approx(pts[-1, 0], abs=0.01)

    def test_dtype(self):
        assert resample_contour(_line(10), n_points=8).dtype == np.float64

    def test_upsample(self):
        assert resample_contour(_line(5), n_points=20).shape == (20, 2)


# ─── TestSmoothAndResampleExtra ───────────────────────────────────────────────

class TestSmoothAndResampleExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_and_resample(np.zeros((0, 2)))

    def test_returns_smoothed_contour(self):
        assert isinstance(smooth_and_resample(_line(20)), SmoothedContour)

    def test_n_points_from_config(self):
        cfg = SmootherConfig(n_points=32)
        assert smooth_and_resample(_line(20), cfg).n_points == 32

    def test_method_gaussian(self):
        cfg = SmootherConfig(sigma=2.0)
        assert smooth_and_resample(_line(20), cfg).method == "gaussian"

    def test_method_none_sigma_zero(self):
        cfg = SmootherConfig(sigma=0.0)
        assert smooth_and_resample(_line(20), cfg).method == "none"

    def test_original_n(self):
        assert smooth_and_resample(_line(15)).original_n == 15


# ─── TestAlignContoursExtra ───────────────────────────────────────────────────

class TestAlignContoursExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            align_contours(np.zeros((0, 2)), np.zeros((0, 2)))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            align_contours(_circle(10), _circle(8))

    def test_returns_two(self):
        assert len(align_contours(_circle(8), _circle(8))) == 2

    def test_identical_zero_shift(self):
        c = _circle(8)
        _, aligned = align_contours(c, c.copy())
        np.testing.assert_array_almost_equal(aligned, c)

    def test_shifted_aligned(self):
        c1 = _circle(16)
        c2 = np.roll(c1, 4, axis=0)
        _, aligned = align_contours(c1, c2)
        residual = float(np.mean(np.linalg.norm(c1 - aligned, axis=1)))
        assert residual < 5.0


# ─── TestContourSimilarityExtra ───────────────────────────────────────────────

class TestContourSimilarityExtra:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            contour_similarity(_line(10), _line(10), metric="unknown")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            contour_similarity(np.zeros((0, 2)), _line(5))

    def test_identical_l2_one(self):
        c = _line(20)
        assert contour_similarity(c, c.copy(), metric="l2") == pytest.approx(1.0, abs=1e-6)

    def test_identical_hausdorff_one(self):
        c = _line(20)
        assert contour_similarity(c, c.copy(), metric="hausdorff") == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        assert 0.0 <= contour_similarity(_circle(20), _line(20) + 100) <= 1.0

    def test_different_lengths(self):
        assert 0.0 <= contour_similarity(_line(10), _line(20), metric="l2") <= 1.0


# ─── TestBatchSmoothExtra ────────────────────────────────────────────────────

class TestBatchSmoothExtra:
    def test_returns_list(self):
        result = batch_smooth([_line(15), _circle(20)])
        assert len(result) == 2
        assert all(isinstance(sc, SmoothedContour) for sc in result)

    def test_empty(self):
        assert batch_smooth([]) == []

    def test_config_applied(self):
        cfg = SmootherConfig(n_points=16)
        for sc in batch_smooth([_line(20), _circle(32)], cfg):
            assert sc.n_points == 16

    def test_single_contour(self):
        result = batch_smooth([_line(10)])
        assert len(result) == 1
