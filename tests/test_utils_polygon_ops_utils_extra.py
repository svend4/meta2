"""Extra tests for puzzle_reconstruction/utils/polygon_ops_utils.py."""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.polygon_ops_utils import (
    PolygonOpsConfig,
    PolygonOverlapResult,
    PolygonStats,
    signed_area,
    polygon_area,
    polygon_perimeter,
    polygon_centroid,
    polygon_bounding_box,
    polygon_stats,
    point_in_polygon,
    polygon_overlap,
    remove_collinear,
    ensure_ccw,
    ensure_cw,
    polygon_similarity,
    batch_polygon_stats,
    batch_polygon_overlap,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(side=10.0) -> np.ndarray:
    """CCW square."""
    return np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=float)


def _triangle() -> np.ndarray:
    return np.array([[0, 0], [4, 0], [2, 3]], dtype=float)


# ─── PolygonOpsConfig ─────────────────────────────────────────────────────────

class TestPolygonOpsConfigExtra:
    def test_defaults(self):
        cfg = PolygonOpsConfig()
        assert cfg.n_samples == 64

    def test_negative_clip_epsilon_raises(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(clip_epsilon=-0.1)

    def test_zero_n_samples_raises(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(n_samples=0)

    def test_negative_simplify_epsilon_raises(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(simplify_epsilon=-1e-5)


# ─── PolygonOverlapResult ─────────────────────────────────────────────────────

class TestPolygonOverlapResultExtra:
    def test_to_dict_keys(self):
        r = PolygonOverlapResult(overlap=True, iou=0.5,
                                  intersection_area=25.0, union_area=50.0)
        d = r.to_dict()
        for k in ("overlap", "iou", "intersection_area", "union_area"):
            assert k in d


# ─── signed_area / polygon_area ───────────────────────────────────────────────

class TestSignedAreaExtra:
    def test_square_ccw_positive(self):
        assert signed_area(_square(10.0)) > 0

    def test_square_cw_negative(self):
        sq = _square(10.0)[::-1]
        assert signed_area(sq) < 0

    def test_degenerate_empty(self):
        assert signed_area(np.zeros((0, 2))) == pytest.approx(0.0)

    def test_area_unit_square(self):
        assert polygon_area(_square(1.0)) == pytest.approx(1.0)

    def test_area_10x10_square(self):
        assert polygon_area(_square(10.0)) == pytest.approx(100.0)

    def test_area_triangle(self):
        assert polygon_area(_triangle()) == pytest.approx(6.0)


# ─── polygon_perimeter ────────────────────────────────────────────────────────

class TestPolygonPerimeterExtra:
    def test_square_perimeter(self):
        assert polygon_perimeter(_square(5.0)) == pytest.approx(20.0)

    def test_empty_polygon_zero(self):
        assert polygon_perimeter(np.zeros((0, 2))) == pytest.approx(0.0)

    def test_single_point_zero(self):
        assert polygon_perimeter(np.array([[1.0, 2.0]])) == pytest.approx(0.0)


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroidExtra:
    def test_square_centroid(self):
        c = polygon_centroid(_square(10.0))
        assert c == pytest.approx([5.0, 5.0], abs=1e-5)

    def test_empty_polygon(self):
        c = polygon_centroid(np.zeros((0, 2)))
        assert c.shape == (2,)

    def test_two_points_mean(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0]])
        c = polygon_centroid(pts)
        assert c == pytest.approx([2.0, 0.0])


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

class TestPolygonBoundingBoxExtra:
    def test_square_bbox(self):
        bbox = polygon_bounding_box(_square(10.0))
        assert bbox == pytest.approx((0.0, 0.0, 10.0, 10.0))

    def test_empty_returns_zeros(self):
        assert polygon_bounding_box(np.zeros((0, 2))) == (0.0, 0.0, 0.0, 0.0)


# ─── polygon_stats ────────────────────────────────────────────────────────────

class TestPolygonStatsExtra:
    def test_returns_stats(self):
        s = polygon_stats(_square(10.0))
        assert isinstance(s, PolygonStats)

    def test_n_vertices(self):
        s = polygon_stats(_square(10.0))
        assert s.n_vertices == 4

    def test_area_correct(self):
        s = polygon_stats(_square(4.0))
        assert s.area == pytest.approx(16.0)

    def test_compactness_square_near_pi_over_4(self):
        s = polygon_stats(_square(10.0))
        # square compactness = pi/4
        assert s.compactness == pytest.approx(math.pi / 4, rel=0.01)

    def test_to_dict_has_keys(self):
        d = polygon_stats(_square(10.0)).to_dict()
        for k in ("n_vertices", "area", "perimeter", "centroid", "bounding_box"):
            assert k in d


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygonExtra:
    def test_inside(self):
        assert point_in_polygon([5.0, 5.0], _square(10.0)) is True

    def test_outside(self):
        assert point_in_polygon([15.0, 15.0], _square(10.0)) is False

    def test_degenerate_polygon(self):
        # < 3 vertices => always False
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert point_in_polygon([0.5, 0.5], pts) is False


# ─── polygon_overlap ──────────────────────────────────────────────────────────

class TestPolygonOverlapExtra:
    def test_identical_squares_iou_one(self):
        r = polygon_overlap(_square(10.0), _square(10.0))
        assert r.iou == pytest.approx(1.0)

    def test_non_overlapping_zero(self):
        sq_a = _square(5.0)
        sq_b = sq_a + np.array([100.0, 0.0])
        r = polygon_overlap(sq_a, sq_b)
        assert r.iou == pytest.approx(0.0)
        assert r.overlap is False

    def test_partial_overlap_in_range(self):
        sq_a = _square(10.0)
        sq_b = sq_a + np.array([5.0, 0.0])
        r = polygon_overlap(sq_a, sq_b)
        assert 0.0 < r.iou < 1.0


# ─── remove_collinear ─────────────────────────────────────────────────────────

class TestRemoveCollinearExtra:
    def test_square_unchanged(self):
        # Square corners are not collinear
        result = remove_collinear(_square(10.0))
        assert result.shape[0] == 4

    def test_degenerate_too_few(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = remove_collinear(pts)
        assert result.shape[0] == 2


# ─── ensure_ccw / ensure_cw ───────────────────────────────────────────────────

class TestEnsureOrientationExtra:
    def test_ccw_already_ccw(self):
        sq = _square(10.0)  # CCW
        result = ensure_ccw(sq)
        assert signed_area(result) > 0

    def test_cw_flips_cw(self):
        sq = _square(10.0)
        result = ensure_cw(sq)
        assert signed_area(result) < 0

    def test_ccw_from_cw_input(self):
        sq_cw = _square(10.0)[::-1]
        result = ensure_ccw(sq_cw)
        assert signed_area(result) > 0


# ─── polygon_similarity ───────────────────────────────────────────────────────

class TestPolygonSimilarityExtra:
    def test_identical_is_one(self):
        sq = _square(10.0)
        assert polygon_similarity(sq, sq) == pytest.approx(1.0)

    def test_non_overlapping_is_zero(self):
        sq_a = _square(5.0)
        sq_b = sq_a + np.array([100.0, 0.0])
        assert polygon_similarity(sq_a, sq_b) == pytest.approx(0.0)


# ─── batch helpers ────────────────────────────────────────────────────────────

class TestBatchPolygonExtra:
    def test_batch_stats_length(self):
        result = batch_polygon_stats([_square(5.0), _square(10.0)])
        assert len(result) == 2

    def test_batch_overlap_length(self):
        result = batch_polygon_overlap([_square(5.0)], [_square(5.0)])
        assert len(result) == 1

    def test_batch_overlap_mismatched_raises(self):
        with pytest.raises(ValueError):
            batch_polygon_overlap([_square(5.0), _square(10.0)], [_square(5.0)])
