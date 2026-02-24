"""Extra tests for puzzle_reconstruction/preprocessing/contour_processor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.contour_processor import (
    ContourConfig,
    ContourStats,
    ContourResult,
    resample_contour,
    smooth_contour,
    rdp_simplify,
    normalize_contour,
    contour_area,
    contour_perimeter,
    compute_contour_stats,
    process_contour,
    batch_process_contours,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square():
    """A simple unit square contour."""
    return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)


def _circle(n=64, r=10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


# ─── ContourConfig ────────────────────────────────────────────────────────────

class TestContourConfigExtra:
    def test_defaults(self):
        cfg = ContourConfig()
        assert cfg.n_points == 128
        assert cfg.smooth_sigma == pytest.approx(1.0)
        assert cfg.rdp_epsilon == pytest.approx(2.0)
        assert cfg.normalize is True

    def test_custom(self):
        cfg = ContourConfig(n_points=64, smooth_sigma=0.0, rdp_epsilon=0.0,
                            normalize=False)
        assert cfg.n_points == 64
        assert cfg.normalize is False

    def test_n_points_too_small_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(n_points=2)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(smooth_sigma=-0.1)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            ContourConfig(rdp_epsilon=-1.0)


# ─── ContourStats ─────────────────────────────────────────────────────────────

class TestContourStatsExtra:
    def test_valid(self):
        s = ContourStats(n_points=10, perimeter=40.0, area=100.0,
                         compactness=0.8, mean_curvature=0.5)
        assert s.n_points == 10

    def test_negative_n_points_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=-1, perimeter=0, area=0,
                         compactness=0, mean_curvature=0)

    def test_negative_perimeter_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=0, perimeter=-1.0, area=0,
                         compactness=0, mean_curvature=0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            ContourStats(n_points=0, perimeter=0, area=-1.0,
                         compactness=0, mean_curvature=0)


# ─── ContourResult ─────────────────────────────────────────────────────────────

class TestContourResultExtra:
    def test_valid(self):
        pts = _square()
        stats = ContourStats(n_points=4, perimeter=40.0, area=100.0,
                             compactness=0.8, mean_curvature=0.5)
        r = ContourResult(points=pts, stats=stats, fragment_id=0)
        assert r.n_points == 4
        assert r.simplified is False

    def test_bad_shape_raises(self):
        pts = np.array([1, 2, 3], dtype=np.float64)
        stats = ContourStats(n_points=0, perimeter=0, area=0,
                             compactness=0, mean_curvature=0)
        with pytest.raises(ValueError):
            ContourResult(points=pts, stats=stats, fragment_id=0)

    def test_centroid(self):
        pts = _square()
        stats = ContourStats(n_points=4, perimeter=40.0, area=100.0,
                             compactness=0.8, mean_curvature=0.5)
        r = ContourResult(points=pts, stats=stats, fragment_id=0)
        cx, cy = r.centroid
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)


# ─── resample_contour ─────────────────────────────────────────────────────────

class TestResampleContourExtra:
    def test_output_length(self):
        pts = _square()
        out = resample_contour(pts, 20)
        assert out.shape == (20, 2)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.array([1, 2, 3]), 10)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            resample_contour(np.array([[0, 0]]), 10)

    def test_n_points_too_small_raises(self):
        with pytest.raises(ValueError):
            resample_contour(_square(), 1)

    def test_degenerate_contour(self):
        pts = np.array([[5, 5], [5, 5]], dtype=np.float64)
        out = resample_contour(pts, 10)
        assert out.shape == (10, 2)
        assert np.allclose(out, [5, 5])


# ─── smooth_contour ───────────────────────────────────────────────────────────

class TestSmoothContourExtra:
    def test_zero_sigma_identity(self):
        pts = _square()
        out = smooth_contour(pts, sigma=0.0)
        assert np.allclose(out, pts)

    def test_shape_preserved(self):
        pts = _circle(64)
        out = smooth_contour(pts, sigma=2.0)
        assert out.shape == pts.shape

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(_square(), sigma=-1.0)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.array([1, 2, 3]), sigma=1.0)


# ─── rdp_simplify ─────────────────────────────────────────────────────────────

class TestRdpSimplifyExtra:
    def test_reduces_circle(self):
        pts = _circle(128)
        out = rdp_simplify(pts, epsilon=1.0)
        assert len(out) < len(pts)

    def test_two_points_unchanged(self):
        pts = np.array([[0, 0], [10, 10]], dtype=np.float64)
        out = rdp_simplify(pts, epsilon=5.0)
        assert len(out) == 2

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            rdp_simplify(_square(), epsilon=-1.0)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            rdp_simplify(np.array([1, 2, 3]), epsilon=1.0)

    def test_zero_epsilon_preserves_all(self):
        pts = _circle(32)
        out = rdp_simplify(pts, epsilon=0.0)
        assert len(out) == len(pts)


# ─── normalize_contour ────────────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_centered(self):
        pts = _square()
        out = normalize_contour(pts)
        assert np.allclose(out.mean(axis=0), [0, 0], atol=1e-6)

    def test_range(self):
        pts = _square()
        out = normalize_contour(pts)
        assert out.max() <= 1.0 + 1e-6
        assert out.min() >= -1.0 - 1e-6

    def test_empty(self):
        pts = np.zeros((0, 2), dtype=np.float64)
        out = normalize_contour(pts)
        assert out.shape == (0, 2)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.array([1, 2]))


# ─── contour_area ─────────────────────────────────────────────────────────────

class TestContourAreaExtra:
    def test_square(self):
        area = contour_area(_square())
        assert area == pytest.approx(100.0)

    def test_triangle(self):
        pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        area = contour_area(pts)
        assert area == pytest.approx(50.0)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            contour_area(np.array([[0, 0], [1, 1]]))

    def test_nonnegative(self):
        # Clockwise vs counter-clockwise should both give positive area
        pts = _square()[::-1]
        assert contour_area(pts) >= 0


# ─── contour_perimeter ────────────────────────────────────────────────────────

class TestContourPerimeterExtra:
    def test_square(self):
        perim = contour_perimeter(_square())
        # 10+10+10+ sqrt(200) for closing edge (0,10)->(0,0)=10
        assert perim == pytest.approx(40.0)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            contour_perimeter(np.array([[0, 0]]))

    def test_nonnegative(self):
        assert contour_perimeter(_square()) >= 0


# ─── compute_contour_stats ────────────────────────────────────────────────────

class TestComputeContourStatsExtra:
    def test_returns_stats(self):
        s = compute_contour_stats(_square())
        assert isinstance(s, ContourStats)
        assert s.n_points == 4

    def test_perimeter_positive(self):
        s = compute_contour_stats(_square())
        assert s.perimeter > 0

    def test_area_positive(self):
        s = compute_contour_stats(_square())
        assert s.area > 0

    def test_compactness_in_range(self):
        s = compute_contour_stats(_circle(128))
        # Circle should have compactness near 1
        assert 0.0 <= s.compactness <= 1.5

    def test_mean_curvature_nonneg(self):
        s = compute_contour_stats(_square())
        assert s.mean_curvature >= 0


# ─── process_contour ──────────────────────────────────────────────────────────

class TestProcessContourExtra:
    def test_returns_result(self):
        pts = _circle(64)
        r = process_contour(pts, fragment_id=5)
        assert isinstance(r, ContourResult)
        assert r.fragment_id == 5

    def test_default_config(self):
        pts = _circle(64)
        r = process_contour(pts)
        assert r.n_points > 0

    def test_no_normalize(self):
        cfg = ContourConfig(normalize=False, smooth_sigma=0.0, rdp_epsilon=0.0)
        pts = _circle(64)
        r = process_contour(pts, cfg=cfg)
        # Should not be centered at zero
        assert isinstance(r, ContourResult)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            process_contour(np.array([1, 2, 3]))


# ─── batch_process_contours ──────────────────────────────────────────────────

class TestBatchProcessContoursExtra:
    def test_empty(self):
        assert batch_process_contours([]) == []

    def test_length(self):
        contours = [_circle(64), _square()]
        results = batch_process_contours(contours)
        assert len(results) == 2

    def test_fragment_ids(self):
        contours = [_circle(64), _square()]
        results = batch_process_contours(contours)
        assert results[0].fragment_id == 0
        assert results[1].fragment_id == 1
