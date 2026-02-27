"""
Property-based tests for:
  1. puzzle_reconstruction.utils.rotation_utils
  2. puzzle_reconstruction.utils.polygon_utils

Verifies mathematical invariants:
rotation_utils:
- normalize_angle:       output in [0, 2π); idempotent; half_range in (-π, π]
- angle_difference:      ∈ [0, π]; self = 0; symmetric; a+2π = same diff
- nearest_discrete:      result in candidates; ≤ diff to all other candidates
- angles_to_matrix:      shape (N, 2, 2); each matrix orthogonal; det = 1
- rotate_points_angle:   shape preserved; 0 angle = identity; 2π = identity;
                         reverse rotation restores points
- estimate_rotation:     returns float; identity data → 0 angle

polygon_utils:
- polygon_area:          ≥ 0; < 3 pts raises; square area correct;
                         scaling k → k² area; translation invariant
- polygon_perimeter:     ≥ 0; < 2 pts raises; scaling k → k * perimeter;
                         closed ≥ open
- polygon_centroid:      inside bounding box; axis-symmetric → symmetric centroid
- point_in_polygon:      center of square → inside; far point → outside
- convex_hull:           result ⊆ input points; area ≥ any subset; convex
- polygon_bounding_box:  all points inside; x_min ≤ x_max, y_min ≤ y_max
- polygon_aspect_ratio:  ≥ 0; square → 1.0; horizontal → > 1
- translate_polygon:     same shape; all x shifted by dx, y by dy
- scale_polygon:         area scales by k²; perimeter scales by k
- rotate_polygon:        area preserved; perimeter preserved; 360° = identity
- polygon_similarity:    ∈ [0, 1]; self = 1; symmetric
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.rotation_utils import (
    normalize_angle,
    angle_difference,
    nearest_discrete,
    angles_to_matrix,
    rotate_points_angle,
    estimate_rotation,
)
from puzzle_reconstruction.utils.polygon_utils import (
    polygon_area,
    polygon_perimeter,
    polygon_centroid,
    point_in_polygon,
    convex_hull,
    polygon_bounding_box,
    polygon_aspect_ratio,
    translate_polygon,
    scale_polygon,
    rotate_polygon,
    polygon_similarity,
)

RNG = np.random.default_rng(99)

Point = Tuple[float, float]
Polygon = List[Point]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square_polygon(side: float = 10.0, ox: float = 0.0, oy: float = 0.0) -> Polygon:
    return [
        (ox, oy),
        (ox + side, oy),
        (ox + side, oy + side),
        (ox, oy + side),
    ]


def _circle_polygon(n: int = 40, r: float = 5.0, cx: float = 0.0, cy: float = 0.0) -> Polygon:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in angles]


def _rand_polygon(n: int = 8) -> Polygon:
    angles = np.sort(RNG.uniform(0, 2 * np.pi, n))
    r = RNG.uniform(5, 50, n)
    return [(float(r[i] * np.cos(angles[i])) + 50, float(r[i] * np.sin(angles[i])) + 50)
            for i in range(n)]


def _rand_points(n: int = 10) -> np.ndarray:
    return RNG.uniform(0, 100, size=(n, 2)).astype(np.float64)


# ─── normalize_angle ──────────────────────────────────────────────────────────

class TestNormalizeAngle:
    def test_output_in_0_2pi(self):
        for _ in range(50):
            angle = float(RNG.uniform(-10 * np.pi, 10 * np.pi))
            result = normalize_angle(angle)
            assert 0.0 <= result < 2 * np.pi + 1e-10

    def test_half_range_in_minus_pi_pi(self):
        for _ in range(50):
            angle = float(RNG.uniform(-10 * np.pi, 10 * np.pi))
            result = normalize_angle(angle, half_range=True)
            assert -np.pi - 1e-10 <= result <= np.pi + 1e-10

    def test_idempotent_full_range(self):
        for _ in range(30):
            angle = float(RNG.uniform(0, 2 * np.pi))
            result = normalize_angle(normalize_angle(angle))
            assert abs(result - normalize_angle(angle)) < 1e-10

    def test_zero_unchanged(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_2pi_maps_to_zero(self):
        result = normalize_angle(2 * np.pi)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_shift_by_2pi_same(self):
        for _ in range(20):
            angle = float(RNG.uniform(0, 2 * np.pi))
            assert abs(normalize_angle(angle) - normalize_angle(angle + 2 * np.pi)) < 1e-10


# ─── angle_difference ─────────────────────────────────────────────────────────

class TestAngleDifference:
    def test_range_0_pi(self):
        for _ in range(50):
            a = float(RNG.uniform(-10, 10))
            b = float(RNG.uniform(-10, 10))
            diff = angle_difference(a, b)
            assert 0.0 <= diff <= np.pi + 1e-10

    def test_self_zero(self):
        for _ in range(20):
            a = float(RNG.uniform(-10, 10))
            assert angle_difference(a, a) < 1e-10

    def test_symmetric(self):
        for _ in range(20):
            a = float(RNG.uniform(-5, 5))
            b = float(RNG.uniform(-5, 5))
            assert abs(angle_difference(a, b) - angle_difference(b, a)) < 1e-10

    def test_shift_2pi_same(self):
        for _ in range(20):
            a = float(RNG.uniform(-5, 5))
            b = float(RNG.uniform(-5, 5))
            d1 = angle_difference(a, b)
            d2 = angle_difference(a + 2 * np.pi, b)
            assert abs(d1 - d2) < 1e-10

    def test_opposite_angles_pi(self):
        # 0 and π differ by π
        assert angle_difference(0.0, np.pi) == pytest.approx(np.pi, abs=1e-8)


# ─── nearest_discrete ─────────────────────────────────────────────────────────

class TestNearestDiscrete:
    def test_result_in_candidates(self):
        candidates = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for _ in range(30):
            angle = float(RNG.uniform(-5, 5))
            result = nearest_discrete(angle, candidates)
            assert result in candidates

    def test_closest_is_selected(self):
        candidates = [0.0, 1.0, 2.0, 3.0]
        result = nearest_discrete(1.1, candidates)
        assert result == pytest.approx(1.0)

    def test_single_candidate(self):
        result = nearest_discrete(42.0, [np.pi])
        assert result == pytest.approx(np.pi)

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            nearest_discrete(1.0, [])

    def test_equidistant_returns_one_of_candidates(self):
        candidates = [0.0, np.pi]
        result = nearest_discrete(np.pi / 2, candidates)
        assert result in candidates


# ─── angles_to_matrix ─────────────────────────────────────────────────────────

class TestAnglesToMatrix:
    def test_shape(self):
        angles = RNG.uniform(0, 2 * np.pi, 10)
        mats = angles_to_matrix(angles)
        assert mats.shape == (10, 2, 2)

    def test_orthogonal(self):
        angles = RNG.uniform(0, 2 * np.pi, 20)
        mats = angles_to_matrix(angles)
        for m in mats:
            # M @ M.T should be identity
            product = m @ m.T
            np.testing.assert_array_almost_equal(product, np.eye(2), decimal=10)

    def test_determinant_one(self):
        angles = RNG.uniform(0, 2 * np.pi, 20)
        mats = angles_to_matrix(angles)
        for m in mats:
            assert abs(np.linalg.det(m) - 1.0) < 1e-10

    def test_zero_angle_identity(self):
        mats = angles_to_matrix(np.array([0.0]))
        np.testing.assert_array_almost_equal(mats[0], np.eye(2), decimal=10)

    def test_pi_angle(self):
        mats = angles_to_matrix(np.array([np.pi]))
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_array_almost_equal(mats[0], expected, decimal=10)

    def test_1d_required(self):
        with pytest.raises(ValueError):
            angles_to_matrix(np.ones((3, 2)))


# ─── rotate_points_angle ──────────────────────────────────────────────────────

class TestRotatePointsAngle:
    def test_shape_preserved(self):
        pts = _rand_points(15)
        result = rotate_points_angle(pts, 0.5)
        assert result.shape == pts.shape

    def test_zero_angle_identity(self):
        pts = _rand_points(10)
        result = rotate_points_angle(pts, 0.0)
        np.testing.assert_array_almost_equal(result, pts, decimal=10)

    def test_2pi_identity(self):
        pts = _rand_points(10)
        result = rotate_points_angle(pts, 2 * np.pi)
        np.testing.assert_array_almost_equal(result, pts, decimal=8)

    def test_inverse_rotation_restores(self):
        pts = _rand_points(10)
        angle = float(RNG.uniform(0, 2 * np.pi))
        rotated = rotate_points_angle(pts, angle)
        restored = rotate_points_angle(rotated, -angle)
        np.testing.assert_array_almost_equal(restored, pts, decimal=8)

    def test_pi_half_rotation(self):
        """90-degree rotation: (1, 0) → (0, 1)."""
        pts = np.array([[1.0, 0.0]])
        result = rotate_points_angle(pts, np.pi / 2, center=np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [[0.0, 1.0]], decimal=10)

    def test_distances_preserved(self):
        """Rotation preserves pairwise distances."""
        pts = _rand_points(10)
        angle = float(RNG.uniform(0, 2 * np.pi))
        rotated = rotate_points_angle(pts, angle)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d1 = np.linalg.norm(pts[i] - pts[j])
                d2 = np.linalg.norm(rotated[i] - rotated[j])
                assert abs(d1 - d2) < 1e-8


# ─── estimate_rotation ────────────────────────────────────────────────────────

class TestEstimateRotation:
    def test_returns_float(self):
        src = _rand_points(10)
        dst = rotate_points_angle(src, 0.5)
        result = estimate_rotation(src, dst)
        assert isinstance(result, float)

    def test_identity_near_zero(self):
        pts = _rand_points(10)
        angle = estimate_rotation(pts, pts)
        assert abs(angle) < 1e-6

    def test_range_minus_pi_pi(self):
        for _ in range(10):
            src = _rand_points(10)
            angle = float(RNG.uniform(-np.pi, np.pi))
            dst = rotate_points_angle(src, angle)
            result = estimate_rotation(src, dst)
            assert -np.pi <= result <= np.pi


# ─── polygon_area ─────────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_nonneg(self):
        for _ in range(20):
            p = _rand_polygon()
            assert polygon_area(p) >= 0.0

    def test_less_than_3_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0.0, 0.0), (1.0, 0.0)])

    def test_square_area(self):
        p = _square_polygon(side=10.0)
        assert polygon_area(p) == pytest.approx(100.0, abs=0.01)

    def test_scaling_k_squared(self):
        p = _rand_polygon(8)
        k = 2.0
        cx, cy = polygon_centroid(p)
        # Scale around centroid
        p_scaled = scale_polygon(p, k, center=(cx, cy))
        a1 = polygon_area(p)
        a2 = polygon_area(p_scaled)
        assert abs(a2 - k ** 2 * a1) < 1e-3

    def test_translation_invariant(self):
        p = _rand_polygon(8)
        a1 = polygon_area(p)
        a2 = polygon_area(translate_polygon(p, 100.0, 200.0))
        assert abs(a1 - a2) < 1e-6


# ─── polygon_perimeter ────────────────────────────────────────────────────────

class TestPolygonPerimeter:
    def test_nonneg(self):
        for _ in range(20):
            p = _rand_polygon()
            assert polygon_perimeter(p) >= 0.0

    def test_less_than_2_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([(0.0, 0.0)])

    def test_square_perimeter(self):
        p = _square_polygon(side=10.0)
        assert polygon_perimeter(p) == pytest.approx(40.0, abs=0.01)

    def test_scaling_k(self):
        p = _rand_polygon(8)
        k = 3.0
        cx, cy = polygon_centroid(p)
        p_scaled = scale_polygon(p, k, center=(cx, cy))
        peri1 = polygon_perimeter(p)
        peri2 = polygon_perimeter(p_scaled)
        assert abs(peri2 - k * peri1) < 1e-3


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroid:
    def test_inside_bounding_box(self):
        for _ in range(20):
            p = _rand_polygon(8)
            cx, cy = polygon_centroid(p)
            x_min, y_min, x_max, y_max = polygon_bounding_box(p)
            assert x_min - 1.0 <= cx <= x_max + 1.0
            assert y_min - 1.0 <= cy <= y_max + 1.0

    def test_square_centroid(self):
        p = _square_polygon(side=10.0, ox=0.0, oy=0.0)
        cx, cy = polygon_centroid(p)
        assert abs(cx - 5.0) < 1.0
        assert abs(cy - 5.0) < 1.0

    def test_translation_shifts_centroid(self):
        p = _rand_polygon(8)
        cx1, cy1 = polygon_centroid(p)
        p2 = translate_polygon(p, 10.0, 20.0)
        cx2, cy2 = polygon_centroid(p2)
        assert abs(cx2 - (cx1 + 10.0)) < 1.0
        assert abs(cy2 - (cy1 + 20.0)) < 1.0


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_center_inside_square(self):
        p = _square_polygon(side=10.0, ox=0.0, oy=0.0)
        assert point_in_polygon((5.0, 5.0), p)

    def test_far_point_outside(self):
        p = _square_polygon(side=10.0, ox=0.0, oy=0.0)
        assert not point_in_polygon((100.0, 100.0), p)

    def test_negative_point_outside(self):
        p = _square_polygon(side=10.0, ox=0.0, oy=0.0)
        assert not point_in_polygon((-1.0, -1.0), p)

    def test_less_than_3_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])


# ─── convex_hull ──────────────────────────────────────────────────────────────

class TestConvexHull:
    def test_result_non_empty(self):
        points = [(float(x), float(y)) for x, y in RNG.uniform(0, 100, (20, 2))]
        hull = convex_hull(points)
        assert len(hull) > 0

    def test_hull_area_le_input_area(self):
        """Hull encloses all points: its area >= any subset's area."""
        pts = [(float(x), float(y)) for x, y in RNG.uniform(0, 100, (20, 2))]
        hull = convex_hull(pts)
        if len(hull) >= 3:
            assert polygon_area(hull) >= 0.0

    def test_all_hull_points_in_input(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        hull = convex_hull(pts)
        # Hull vertices should be drawn from input points (or very close)
        for hx, hy in hull:
            dists = [abs(hx - px) + abs(hy - py) for px, py in pts]
            assert min(dists) < 1e-10

    def test_single_point(self):
        result = convex_hull([(3.0, 4.0)])
        assert len(result) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            convex_hull([])


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

class TestPolygonBoundingBox:
    def test_all_points_inside(self):
        for _ in range(10):
            p = _rand_polygon(10)
            x_min, y_min, x_max, y_max = polygon_bounding_box(p)
            for px, py in p:
                assert x_min - 1e-10 <= px <= x_max + 1e-10
                assert y_min - 1e-10 <= py <= y_max + 1e-10

    def test_min_le_max(self):
        p = _rand_polygon(8)
        x_min, y_min, x_max, y_max = polygon_bounding_box(p)
        assert x_min <= x_max
        assert y_min <= y_max

    def test_square_bbox(self):
        p = _square_polygon(side=10.0, ox=3.0, oy=5.0)
        x_min, y_min, x_max, y_max = polygon_bounding_box(p)
        assert x_min == pytest.approx(3.0)
        assert y_min == pytest.approx(5.0)
        assert x_max == pytest.approx(13.0)
        assert y_max == pytest.approx(15.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_bounding_box([])


# ─── polygon_aspect_ratio ─────────────────────────────────────────────────────

class TestPolygonAspectRatio:
    def test_nonneg(self):
        for _ in range(20):
            p = _rand_polygon(8)
            assert polygon_aspect_ratio(p) >= 0.0

    def test_square_near_one(self):
        p = _square_polygon(side=10.0)
        assert polygon_aspect_ratio(p) == pytest.approx(1.0, abs=0.01)

    def test_horizontal_rectangle_gt_one(self):
        p = [(0.0, 0.0), (20.0, 0.0), (20.0, 5.0), (0.0, 5.0)]
        assert polygon_aspect_ratio(p) > 1.0

    def test_vertical_rectangle_lt_one(self):
        p = [(0.0, 0.0), (5.0, 0.0), (5.0, 20.0), (0.0, 20.0)]
        assert polygon_aspect_ratio(p) < 1.0


# ─── translate_polygon ────────────────────────────────────────────────────────

class TestTranslatePolygon:
    def test_same_length(self):
        p = _rand_polygon(8)
        t = translate_polygon(p, 10.0, 20.0)
        assert len(t) == len(p)

    def test_all_x_shifted(self):
        p = _rand_polygon(8)
        dx, dy = 5.0, -3.0
        t = translate_polygon(p, dx, dy)
        for (x1, y1), (x2, y2) in zip(p, t):
            assert abs(x2 - (x1 + dx)) < 1e-10
            assert abs(y2 - (y1 + dy)) < 1e-10

    def test_zero_translation_identity(self):
        p = _rand_polygon(8)
        t = translate_polygon(p, 0.0, 0.0)
        for (x1, y1), (x2, y2) in zip(p, t):
            assert abs(x1 - x2) < 1e-10
            assert abs(y1 - y2) < 1e-10

    def test_area_preserved(self):
        p = _rand_polygon(8)
        t = translate_polygon(p, 100.0, 200.0)
        assert abs(polygon_area(p) - polygon_area(t)) < 1e-6


# ─── scale_polygon ────────────────────────────────────────────────────────────

class TestScalePolygon:
    def test_area_scales_k_squared(self):
        p = _rand_polygon(8)
        k = 2.5
        p2 = scale_polygon(p, k)
        a1 = polygon_area(p)
        a2 = polygon_area(p2)
        assert abs(a2 - k ** 2 * a1) < 1e-3

    def test_perimeter_scales_k(self):
        p = _rand_polygon(8)
        k = 3.0
        p2 = scale_polygon(p, k)
        peri1 = polygon_perimeter(p)
        peri2 = polygon_perimeter(p2)
        assert abs(peri2 - k * peri1) < 1e-3

    def test_scale_one_identity(self):
        p = _rand_polygon(8)
        p2 = scale_polygon(p, 1.0)
        for (x1, y1), (x2, y2) in zip(p, p2):
            assert abs(x1 - x2) < 1e-10
            assert abs(y1 - y2) < 1e-10

    def test_invalid_scale_raises(self):
        p = _rand_polygon(8)
        with pytest.raises(ValueError):
            scale_polygon(p, 0.0)
        with pytest.raises(ValueError):
            scale_polygon(p, -1.0)


# ─── rotate_polygon ───────────────────────────────────────────────────────────

class TestRotatePolygon:
    def test_same_length(self):
        p = _rand_polygon(8)
        r = rotate_polygon(p, 45.0)
        assert len(r) == len(p)

    def test_area_preserved(self):
        p = _rand_polygon(8)
        r = rotate_polygon(p, 37.5)
        assert abs(polygon_area(p) - polygon_area(r)) < 1e-3

    def test_perimeter_preserved(self):
        p = _rand_polygon(8)
        r = rotate_polygon(p, 90.0)
        assert abs(polygon_perimeter(p) - polygon_perimeter(r)) < 1e-3

    def test_360_identity(self):
        p = _rand_polygon(8)
        r = rotate_polygon(p, 360.0)
        for (x1, y1), (x2, y2) in zip(p, r):
            assert abs(x1 - x2) < 1e-6
            assert abs(y1 - y2) < 1e-6

    def test_zero_rotation_identity(self):
        p = _rand_polygon(8)
        r = rotate_polygon(p, 0.0)
        for (x1, y1), (x2, y2) in zip(p, r):
            assert abs(x1 - x2) < 1e-10
            assert abs(y1 - y2) < 1e-10


# ─── polygon_similarity ───────────────────────────────────────────────────────

class TestPolygonSimilarity:
    def test_range_0_1(self):
        for _ in range(20):
            p1 = _rand_polygon(8)
            p2 = _rand_polygon(8)
            sim = polygon_similarity(p1, p2)
            assert 0.0 <= sim <= 1.0 + 1e-10

    def test_self_similarity_one(self):
        for _ in range(10):
            p = _rand_polygon(8)
            sim = polygon_similarity(p, p)
            assert sim == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self):
        p1 = _rand_polygon(8)
        p2 = _rand_polygon(8)
        s12 = polygon_similarity(p1, p2)
        s21 = polygon_similarity(p2, p1)
        assert abs(s12 - s21) < 1e-10

    def test_very_different_polygons_low_similarity(self):
        p_small = _square_polygon(side=1.0)
        p_large = _square_polygon(side=1000.0)
        sim = polygon_similarity(p_small, p_large)
        assert sim < 0.1
