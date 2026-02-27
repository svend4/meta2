"""
Property-based tests for puzzle_reconstruction.utils.geometry.

Verifies mathematical invariants:
- rotation_matrix_2d: orthogonality, determinant = 1, angle composition
- rotate_points: zero-angle identity, 2π identity, inverse rotation
- polygon_area: sign (CCW positive), quadratic scaling, known shapes
- polygon_centroid: axis-symmetric polygon centroid, collinear fallback
- bbox_from_points: all points inside, tight bounds
- resample_curve: exact n points, arc-length preservation
- align_centroids: centroids match after alignment
- poly_iou: range [0,1], self-IoU = 1, symmetry, disjoint = 0
- point_in_polygon: centroid inside, distant point outside
- normalize_contour: centroid at origin, diagonal = 1
- smooth_contour: length preserved, endpoints approximately kept
- curvature: circle → constant κ, straight line → κ ≈ 0
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.geometry import (
    rotation_matrix_2d,
    rotate_points,
    polygon_area,
    polygon_centroid,
    bbox_from_points,
    resample_curve,
    align_centroids,
    poly_iou,
    point_in_polygon,
    normalize_contour,
    smooth_contour,
    curvature,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _circle_pts(n: int = 64, r: float = 1.0) -> np.ndarray:
    """CCW circle with *n* points and radius *r*."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square_ccw(side: float = 10.0) -> np.ndarray:
    """Unit square CCW: bottom-left → bottom-right → top-right → top-left."""
    s = side
    return np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)


def _straight_line(n: int = 64) -> np.ndarray:
    return np.column_stack([np.linspace(0, 10, n), np.zeros(n)])


# ── 1. rotation_matrix_2d ──────────────────────────────────────────────────────

class TestRotationMatrix2D:
    """Mathematical properties of the 2×2 rotation matrix."""

    def test_shape_is_2x2(self):
        R = rotation_matrix_2d(0.5)
        assert R.shape == (2, 2)

    def test_dtype_is_float64(self):
        R = rotation_matrix_2d(1.0)
        assert R.dtype == np.float64

    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, math.pi / 4, math.pi, 2 * math.pi])
    def test_orthogonality(self, angle):
        """R @ R.T must equal the identity matrix."""
        R = rotation_matrix_2d(angle)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)

    @pytest.mark.parametrize("angle", [0.0, 0.3, -1.2, math.pi])
    def test_determinant_equals_one(self, angle):
        """det(R) = 1 for all rotation angles."""
        R = rotation_matrix_2d(angle)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_zero_angle_is_identity(self):
        R = rotation_matrix_2d(0.0)
        assert np.allclose(R, np.eye(2), atol=1e-14)

    def test_pi_rotates_180(self):
        R = rotation_matrix_2d(math.pi)
        # (1, 0) → (−1, 0) approximately
        v = np.array([1.0, 0.0])
        result = R @ v
        assert np.allclose(result, [-1.0, 0.0], atol=1e-14)

    def test_angle_composition(self):
        """R(a) @ R(b) ≈ R(a+b)."""
        a, b = 0.7, 1.3
        assert np.allclose(rotation_matrix_2d(a) @ rotation_matrix_2d(b),
                           rotation_matrix_2d(a + b), atol=1e-13)

    def test_inverse_is_transpose(self):
        """R(-θ) = R(θ).T for any θ."""
        angle = 1.23
        R = rotation_matrix_2d(angle)
        Rinv = rotation_matrix_2d(-angle)
        assert np.allclose(Rinv, R.T, atol=1e-14)


# ── 2. rotate_points ──────────────────────────────────────────────────────────

class TestRotatePoints:
    """Properties of point rotation."""

    def test_zero_angle_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = rotate_points(pts, 0.0)
        assert np.allclose(result, pts, atol=1e-14)

    def test_two_pi_identity(self):
        pts = _circle_pts(32)
        result = rotate_points(pts, 2 * math.pi)
        assert np.allclose(result, pts, atol=1e-13)

    def test_inverse_rotation_restores_original(self):
        pts = _circle_pts(16)
        angle = 1.234
        rotated = rotate_points(pts, angle)
        restored = rotate_points(rotated, -angle)
        assert np.allclose(restored, pts, atol=1e-12)

    def test_rotation_preserves_distances(self):
        """Distances between points are invariant under rotation."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        angle = 0.7
        rotated = rotate_points(pts, angle)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                orig_dist = np.linalg.norm(pts[i] - pts[j])
                rot_dist  = np.linalg.norm(rotated[i] - rotated[j])
                assert abs(orig_dist - rot_dist) < 1e-13

    def test_rotation_around_center_keeps_center(self):
        """Center point is fixed when rotating around it."""
        center = np.array([5.0, 3.0])
        pts = _circle_pts(8)
        rotated = rotate_points(pts, math.pi / 3, center=center)
        # center itself should not move
        center_rot = rotate_points(center[np.newaxis, :], math.pi / 3, center=center)
        assert np.allclose(center_rot, center[np.newaxis, :], atol=1e-12)

    def test_output_shape_matches_input(self):
        pts = np.random.default_rng(0).random((20, 2))
        result = rotate_points(pts, 0.5)
        assert result.shape == pts.shape


# ── 3. polygon_area ───────────────────────────────────────────────────────────

class TestPolygonArea:
    """Shoelace area formula invariants."""

    def test_square_area(self):
        sq = _square_ccw(10.0)
        assert abs(polygon_area(sq) - 100.0) < 1e-10

    def test_ccw_is_positive(self):
        sq = _square_ccw(5.0)
        assert polygon_area(sq) > 0

    def test_cw_is_negative(self):
        sq = _square_ccw(5.0)[::-1]   # reverse → CW
        assert polygon_area(sq) < 0

    def test_reversing_changes_sign_only(self):
        """area(CCW) == -area(CW)."""
        sq = _square_ccw(7.0)
        assert abs(polygon_area(sq) + polygon_area(sq[::-1])) < 1e-10

    def test_quadratic_scaling(self):
        """Area scales as scale²."""
        for scale in [1.0, 2.0, 3.0, 0.5]:
            area_scaled = polygon_area(_square_ccw(scale * 10.0))
            area_orig   = polygon_area(_square_ccw(10.0))
            assert abs(area_scaled - scale ** 2 * area_orig) < 1e-10

    def test_triangle_area(self):
        """Right triangle with legs 3, 4 has area 6."""
        tri = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
        assert abs(polygon_area(tri) - 6.0) < 1e-10

    def test_degenerate_less_than_3_points(self):
        assert polygon_area(np.array([[0, 0], [1, 1]])) == 0.0

    def test_translation_invariant(self):
        sq = _square_ccw(10.0)
        offset = np.array([100.0, 200.0])
        assert abs(polygon_area(sq) - polygon_area(sq + offset)) < 1e-8


# ── 4. polygon_centroid ───────────────────────────────────────────────────────

class TestPolygonCentroid:
    """Centroid invariants."""

    def test_square_centroid(self):
        sq = _square_ccw(10.0)
        c = polygon_centroid(sq)
        assert np.allclose(c, [5.0, 5.0], atol=1e-10)

    def test_centroid_inside_polygon(self):
        """Centroid of a convex polygon must be inside it."""
        sq = _square_ccw(10.0)
        c = polygon_centroid(sq)
        assert point_in_polygon(c, sq)

    def test_centroid_translation_equivariant(self):
        """Centroid shifts by the same offset as the polygon."""
        sq = _square_ccw(10.0)
        offset = np.array([15.0, -7.0])
        c1 = polygon_centroid(sq)
        c2 = polygon_centroid(sq + offset)
        assert np.allclose(c2, c1 + offset, atol=1e-10)

    def test_centroid_symmetric_polygon_at_origin(self):
        """Centroid of a symmetric polygon centered at origin ≈ (0,0)."""
        circle = _circle_pts(128)
        c = polygon_centroid(circle)
        assert np.allclose(c, [0.0, 0.0], atol=1e-10)

    def test_collinear_fallback(self):
        """Degenerate polygon (< 3 pts) returns mean."""
        pts = np.array([[0.0, 0.0], [2.0, 2.0]])
        c = polygon_centroid(pts)
        assert np.allclose(c, [1.0, 1.0], atol=1e-12)


# ── 5. bbox_from_points ───────────────────────────────────────────────────────

class TestBboxFromPoints:
    """Bounding-box properties."""

    def test_all_points_inside_bbox(self):
        rng = np.random.default_rng(7)
        pts = rng.uniform(-10, 10, (50, 2))
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert np.all(pts[:, 0] >= x0 - 1e-12)
        assert np.all(pts[:, 1] >= y0 - 1e-12)
        assert np.all(pts[:, 0] <= x1 + 1e-12)
        assert np.all(pts[:, 1] <= y1 + 1e-12)

    def test_tight_bbox_single_point_on_boundary(self):
        pts = np.array([[0.0, 0.0], [5.0, 3.0]])
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 == 0.0 and y0 == 0.0 and x1 == 5.0 and y1 == 3.0

    def test_x_min_leq_x_max(self):
        pts = _circle_pts(32)
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 <= x1 and y0 <= y1

    def test_empty_returns_zeros(self):
        result = bbox_from_points(np.empty((0, 2)))
        assert result == (0.0, 0.0, 0.0, 0.0)


# ── 6. resample_curve ─────────────────────────────────────────────────────────

class TestResampleCurve:
    """Resampled curve invariants."""

    @pytest.mark.parametrize("n", [8, 16, 32, 100])
    def test_exact_n_points(self, n):
        pts = _circle_pts(64)
        result = resample_curve(pts, n)
        assert result.shape == (n, 2)

    def test_circle_arc_length_preserved(self):
        """Resampling a circle should preserve total arc length approximately."""
        orig = _circle_pts(64, r=1.0)
        resampled = resample_curve(orig, 128)
        def arc_len(p):
            d = np.diff(p, axis=0)
            return float(np.hypot(d[:, 0], d[:, 1]).sum())
        # Allow ±5% tolerance
        L_orig  = arc_len(orig)
        L_resam = arc_len(resampled)
        assert abs(L_resam - L_orig) / L_orig < 0.05

    def test_resamples_straight_line_uniformly(self):
        """Points on a straight line should stay collinear after resampling."""
        pts = _straight_line(32)
        result = resample_curve(pts, 16)
        # All y-values should be near 0
        assert np.allclose(result[:, 1], 0.0, atol=1e-12)

    def test_first_and_last_points_approx_preserved(self):
        pts = _straight_line(64)
        result = resample_curve(pts, 32)
        assert np.allclose(result[0], pts[0], atol=1e-12)
        assert np.allclose(result[-1], pts[-1], atol=1e-10)


# ── 7. align_centroids ────────────────────────────────────────────────────────

class TestAlignCentroids:
    """Centroid alignment invariants."""

    def test_centroids_match_after_alignment(self):
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        tgt = np.array([[10.0, 5.0], [12.0, 5.0], [11.0, 7.0]])
        aligned = align_centroids(src, tgt)
        assert np.allclose(aligned.mean(axis=0), tgt.mean(axis=0), atol=1e-12)

    def test_shape_unchanged(self):
        src = np.random.default_rng(3).random((15, 2))
        tgt = np.random.default_rng(4).random((10, 2))
        result = align_centroids(src, tgt)
        assert result.shape == src.shape

    def test_same_centroid_means_no_movement(self):
        src = _circle_pts(32)
        tgt = _circle_pts(16)   # same centroid (0, 0)
        aligned = align_centroids(src, tgt)
        assert np.allclose(aligned, src, atol=1e-12)

    def test_relative_distances_preserved(self):
        """Rigid translation must not change pairwise distances."""
        src = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        tgt = np.array([[100.0, 0.0]])
        aligned = align_centroids(src, tgt)
        orig_d = np.linalg.norm(src[0] - src[2])
        aln_d  = np.linalg.norm(aligned[0] - aligned[2])
        assert abs(orig_d - aln_d) < 1e-12


# ── 8. poly_iou ───────────────────────────────────────────────────────────────

class TestPolyIoU:
    """Intersection-over-Union properties."""

    def test_self_iou_equals_one(self):
        sq = _square_ccw(10.0)
        assert abs(poly_iou(sq, sq) - 1.0) < 1e-6

    def test_iou_in_range_zero_one(self):
        sq = _square_ccw(10.0)
        sq2 = sq + np.array([5.0, 0.0])
        iou = poly_iou(sq, sq2)
        assert 0.0 <= iou <= 1.0 + 1e-9

    def test_symmetry(self):
        sq1 = _square_ccw(10.0)
        sq2 = sq1 + np.array([3.0, 3.0])
        assert abs(poly_iou(sq1, sq2) - poly_iou(sq2, sq1)) < 1e-10

    def test_disjoint_polygons_iou_zero(self):
        sq1 = _square_ccw(5.0)
        sq2 = _square_ccw(5.0) + np.array([100.0, 0.0])
        assert poly_iou(sq1, sq2) < 1e-9

    def test_contained_iou_less_than_one(self):
        """Smaller polygon inside larger → IoU < 1."""
        big   = _square_ccw(10.0)
        small = _square_ccw(4.0) + np.array([3.0, 3.0])
        iou = poly_iou(big, small)
        assert iou < 1.0

    def test_overlap_half_iou_roughly_one_third(self):
        """Two 10×10 squares sharing 5×10 strip: IoU = 50/(100+100-50) = 1/3."""
        sq1 = _square_ccw(10.0)
        sq2 = _square_ccw(10.0) + np.array([5.0, 0.0])
        iou = poly_iou(sq1, sq2)
        assert abs(iou - 1.0 / 3.0) < 0.01


# ── 9. point_in_polygon ───────────────────────────────────────────────────────

class TestPointInPolygon:
    """Winding-number inside/outside tests."""

    def test_centroid_inside_square(self):
        sq = _square_ccw(10.0)
        center = np.array([5.0, 5.0])
        assert point_in_polygon(center, sq) is True

    def test_far_point_outside_square(self):
        sq = _square_ccw(10.0)
        far = np.array([100.0, 100.0])
        assert point_in_polygon(far, sq) is False

    def test_centroid_inside_circle_approx(self):
        circle = _circle_pts(64, r=5.0)
        assert point_in_polygon(np.array([0.0, 0.0]), circle) is True

    def test_outside_circle_approx(self):
        circle = _circle_pts(64, r=5.0)
        assert point_in_polygon(np.array([10.0, 10.0]), circle) is False

    def test_collinear_polygon_returns_false(self):
        """Degenerate polygon with < 3 points."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert point_in_polygon(np.array([0.5, 0.5]), pts) is False


# ── 10. normalize_contour ─────────────────────────────────────────────────────

class TestNormalizeContour:
    """Normalization invariants."""

    def test_centroid_at_origin(self):
        sq = _square_ccw(20.0)
        normed = normalize_contour(sq)
        assert np.allclose(normed.mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_bbox_diagonal_equals_target_scale(self):
        sq = _square_ccw(20.0)
        normed = normalize_contour(sq, target_scale=1.0)
        x0, y0, x1, y1 = bbox_from_points(normed)
        diag = np.hypot(x1 - x0, y1 - y0)
        assert abs(diag - 1.0) < 1e-12

    def test_custom_target_scale(self):
        sq = _square_ccw(10.0)
        for scale in [0.5, 2.0, 5.0]:
            normed = normalize_contour(sq, target_scale=scale)
            x0, y0, x1, y1 = bbox_from_points(normed)
            diag = np.hypot(x1 - x0, y1 - y0)
            assert abs(diag - scale) < 1e-12

    def test_shape_preserved(self):
        pts = _circle_pts(32)
        normed = normalize_contour(pts)
        assert normed.shape == pts.shape

    def test_normalize_twice_idempotent(self):
        """Normalizing an already-normalized contour returns the same."""
        sq = _square_ccw(10.0)
        once  = normalize_contour(sq)
        twice = normalize_contour(once)
        assert np.allclose(once, twice, atol=1e-12)


# ── 11. smooth_contour ────────────────────────────────────────────────────────

class TestSmoothContour:
    """Smoothing invariants."""

    def test_output_same_shape(self):
        pts = _circle_pts(64)
        smoothed = smooth_contour(pts, window=5)
        assert smoothed.shape == pts.shape

    def test_circle_center_preserved(self):
        """Mean of a circle should stay at origin after smoothing."""
        pts = _circle_pts(128)
        smoothed = smooth_contour(pts, window=7)
        assert np.allclose(smoothed.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_constant_signal_unchanged(self):
        """Smoothing a constant signal (all same point) gives same value."""
        pts = np.ones((32, 2)) * 3.0
        smoothed = smooth_contour(pts, window=5)
        assert np.allclose(smoothed, pts, atol=1e-12)

    def test_smoothing_reduces_high_frequency_noise(self):
        """After smoothing, the max deviation from mean should decrease."""
        rng = np.random.default_rng(99)
        pts = _circle_pts(64)
        noisy = pts + rng.normal(0, 0.5, pts.shape)
        smoothed = smooth_contour(noisy, window=11)
        # Smoothed should be closer to original circle than noisy version
        err_noisy = np.abs(np.linalg.norm(noisy, axis=1) - 1.0).mean()
        err_smooth = np.abs(np.linalg.norm(smoothed, axis=1) - 1.0).mean()
        assert err_smooth < err_noisy


# ── 12. curvature ─────────────────────────────────────────────────────────────

class TestCurvature:
    """Curvature invariants."""

    def test_output_length_equals_input(self):
        pts = _circle_pts(64)
        kappa = curvature(pts)
        assert len(kappa) == len(pts)

    def test_straight_line_near_zero_curvature(self):
        """A straight line has κ ≈ 0 everywhere (interior)."""
        pts = _straight_line(64)
        kappa = curvature(pts)
        # Interior points (exclude boundaries where gradient is less accurate)
        interior = kappa[5:-5]
        assert np.all(interior < 1e-6)

    def test_circle_curvature_approximately_constant(self):
        """Circle with radius r should have κ ≈ 1/r (constant)."""
        r = 5.0
        pts = _circle_pts(256, r=r)
        kappa = curvature(pts)
        # Mean curvature should be close to 1/r
        mean_kappa = float(kappa.mean())
        expected = 1.0 / r
        assert abs(mean_kappa - expected) / expected < 0.05   # 5% tolerance

    def test_curvature_nonnegative(self):
        """Curvature is defined as |κ|, so always ≥ 0."""
        pts = _circle_pts(64)
        kappa = curvature(pts)
        assert np.all(kappa >= 0)

    def test_smaller_circle_has_larger_curvature(self):
        """κ = 1/r → smaller r → larger κ."""
        kappa_big   = curvature(_circle_pts(128, r=10.0)).mean()
        kappa_small = curvature(_circle_pts(128, r=2.0)).mean()
        assert kappa_small > kappa_big

    def test_degenerate_less_than_3_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        kappa = curvature(pts)
        assert np.all(kappa == 0.0)
