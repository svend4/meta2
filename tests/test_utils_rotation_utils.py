"""Tests for puzzle_reconstruction/utils/rotation_utils.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.rotation_utils import (
    RotationConfig,
    rotate_image_angle,
    rotate_points_angle,
    normalize_angle,
    angle_difference,
    nearest_discrete,
    angles_to_matrix,
    batch_rotate_images,
    estimate_rotation,
)
import cv2


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=20, w=20, val=200):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=20, w=20, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _square_pts(side=1.0):
    return np.array([
        [0, 0], [side, 0], [side, side], [0, side]
    ], dtype=np.float64)


# ─── TestRotationConfig ───────────────────────────────────────────────────────

class TestRotationConfig:
    def test_defaults(self):
        cfg = RotationConfig()
        assert cfg.border_mode == cv2.BORDER_CONSTANT
        assert cfg.interpolation == cv2.INTER_LINEAR
        assert cfg.expand is True

    def test_invalid_border_mode_raises(self):
        with pytest.raises(ValueError):
            RotationConfig(border_mode=999)

    def test_invalid_interpolation_raises(self):
        with pytest.raises(ValueError):
            RotationConfig(interpolation=999)

    def test_valid_border_reflect(self):
        cfg = RotationConfig(border_mode=cv2.BORDER_REFLECT)
        assert cfg.border_mode == cv2.BORDER_REFLECT

    def test_valid_inter_nearest(self):
        cfg = RotationConfig(interpolation=cv2.INTER_NEAREST)
        assert cfg.interpolation == cv2.INTER_NEAREST

    def test_expand_false(self):
        cfg = RotationConfig(expand=False)
        assert cfg.expand is False


# ─── TestRotateImageAngle ─────────────────────────────────────────────────────

class TestRotateImageAngle:
    def test_returns_ndarray(self):
        result = rotate_image_angle(_bgr(), 30.0)
        assert isinstance(result, np.ndarray)

    def test_zero_rotation_same_size(self):
        img = _bgr(h=20, w=30)
        result = rotate_image_angle(img, 0.0, RotationConfig(expand=False))
        assert result.shape == img.shape

    def test_grayscale_ok(self):
        result = rotate_image_angle(_gray(), 45.0)
        assert result.ndim == 2

    def test_4d_raises(self):
        img = np.zeros((5, 5, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            rotate_image_angle(img, 30.0)

    def test_expand_increases_size(self):
        img = _bgr(h=20, w=20)
        result_expand = rotate_image_angle(img, 45.0, RotationConfig(expand=True))
        result_fixed = rotate_image_angle(img, 45.0, RotationConfig(expand=False))
        assert (result_expand.shape[0] >= result_fixed.shape[0] or
                result_expand.shape[1] >= result_fixed.shape[1])

    def test_360_rotation_restores_shape(self):
        img = _bgr(h=20, w=20)
        result = rotate_image_angle(img, 360.0, RotationConfig(expand=False))
        assert result.shape == img.shape

    def test_none_cfg_uses_default(self):
        result = rotate_image_angle(_bgr(), 30.0, None)
        assert result.ndim == 3

    def test_output_same_channels(self):
        img = _bgr(h=15, w=15)
        result = rotate_image_angle(img, 45.0)
        assert result.shape[2] == 3


# ─── TestRotatePointsAngle ────────────────────────────────────────────────────

class TestRotatePointsAngle:
    def test_returns_float64(self):
        pts = _square_pts()
        result = rotate_points_angle(pts, 0.0)
        assert result.dtype == np.float64

    def test_zero_rotation_identity(self):
        pts = _square_pts(2.0)
        result = rotate_points_angle(pts, 0.0)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_shape_preserved(self):
        pts = _square_pts()
        result = rotate_points_angle(pts, math.pi / 4)
        assert result.shape == pts.shape

    def test_360_rotation_returns_to_start(self):
        pts = _square_pts(2.0)
        result = rotate_points_angle(pts, 2 * math.pi)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            rotate_points_angle(np.array([[1, 2, 3]]), 0.5)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            rotate_points_angle(np.array([1, 2]), 0.5)

    def test_centroid_unchanged_by_rotation(self):
        pts = _square_pts(2.0)
        center = pts.mean(axis=0)
        result = rotate_points_angle(pts, math.pi / 3)
        new_center = result.mean(axis=0)
        np.testing.assert_allclose(new_center, center, atol=1e-10)

    def test_explicit_center(self):
        pts = np.array([[1.0, 0.0]], dtype=np.float64)
        center = np.array([0.0, 0.0])
        result = rotate_points_angle(pts, math.pi / 2, center=center)
        np.testing.assert_allclose(result[0], [0.0, 1.0], atol=1e-10)

    def test_distances_preserved(self):
        pts = _square_pts(3.0)
        center = pts.mean(axis=0)
        orig_dists = np.linalg.norm(pts - center, axis=1)
        result = rotate_points_angle(pts, math.pi / 5)
        new_dists = np.linalg.norm(result - center, axis=1)
        np.testing.assert_allclose(new_dists, orig_dists, atol=1e-10)


# ─── TestNormalizeAngle ───────────────────────────────────────────────────────

class TestNormalizeAngle:
    def test_zero_stays_zero(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_two_pi_becomes_zero(self):
        assert normalize_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_negative_angle_normalized(self):
        v = normalize_angle(-math.pi / 2)
        assert 0.0 <= v < 2 * math.pi

    def test_half_range_negative_pi(self):
        v = normalize_angle(-math.pi / 4, half_range=True)
        assert -math.pi < v <= math.pi

    def test_half_range_pi_stays(self):
        v = normalize_angle(math.pi, half_range=True)
        assert v == pytest.approx(math.pi) or v == pytest.approx(-math.pi)

    def test_returns_float(self):
        assert isinstance(normalize_angle(1.0), float)

    def test_full_range_in_0_2pi(self):
        for a in [-5.0, -math.pi, 0.0, math.pi, 3 * math.pi]:
            v = normalize_angle(a)
            assert 0.0 <= v < 2 * math.pi + 1e-10


# ─── TestAngleDifference ──────────────────────────────────────────────────────

class TestAngleDifference:
    def test_same_angle_is_zero(self):
        assert angle_difference(1.0, 1.0) == pytest.approx(0.0)

    def test_opposite_angles_is_pi(self):
        assert angle_difference(0.0, math.pi) == pytest.approx(math.pi)

    def test_90_degrees(self):
        assert angle_difference(0.0, math.pi / 2) == pytest.approx(math.pi / 2)

    def test_symmetric(self):
        a, b = 0.3, 2.1
        assert angle_difference(a, b) == pytest.approx(angle_difference(b, a))

    def test_result_in_0_pi(self):
        for a in [0.0, 1.0, math.pi, 4.0, 6.0]:
            for b in [0.0, 0.5, math.pi, 5.0]:
                d = angle_difference(a, b)
                assert 0.0 <= d <= math.pi + 1e-10

    def test_returns_float(self):
        assert isinstance(angle_difference(0.0, 1.0), float)

    def test_wrap_around(self):
        # 0.1 and 2*pi - 0.1 should be ~0.2 apart
        assert angle_difference(0.1, 2 * math.pi - 0.1) == pytest.approx(0.2, abs=1e-6)


# ─── TestNearestDiscrete ──────────────────────────────────────────────────────

class TestNearestDiscrete:
    def test_exact_match(self):
        cands = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        assert nearest_discrete(0.0, cands) == pytest.approx(0.0)

    def test_nearest_candidate(self):
        cands = [0.0, math.pi / 2]
        # 0.1 is closer to 0 than to pi/2
        result = nearest_discrete(0.1, cands)
        assert result == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nearest_discrete(1.0, [])

    def test_single_candidate(self):
        result = nearest_discrete(99.0, [1.0])
        assert result == pytest.approx(1.0)

    def test_returns_float(self):
        result = nearest_discrete(0.5, [0.0, 1.0])
        assert isinstance(result, float)

    def test_standard_orientations(self):
        """Nearest among 0, 90, 180, 270 degrees."""
        cands = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        assert nearest_discrete(0.1, cands) == pytest.approx(0.0)
        assert nearest_discrete(math.pi / 2 - 0.1, cands) == pytest.approx(math.pi / 2)


# ─── TestAnglesToMatrix ───────────────────────────────────────────────────────

class TestAnglesToMatrix:
    def test_shape(self):
        angles = np.array([0.0, math.pi / 2, math.pi])
        result = angles_to_matrix(angles)
        assert result.shape == (3, 2, 2)

    def test_dtype_float64(self):
        result = angles_to_matrix(np.array([0.0]))
        assert result.dtype == np.float64

    def test_zero_angle_identity(self):
        result = angles_to_matrix(np.array([0.0]))
        np.testing.assert_allclose(result[0], np.eye(2), atol=1e-10)

    def test_90_degrees(self):
        result = angles_to_matrix(np.array([math.pi / 2]))
        expected = np.array([[0, -1], [1, 0]], dtype=np.float64)
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_empty_array(self):
        result = angles_to_matrix(np.array([]))
        assert result.shape == (0, 2, 2)

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            angles_to_matrix(np.array([[0.0, 1.0]]))

    def test_rotation_matrices_orthogonal(self):
        angles = np.linspace(0, 2 * math.pi, 10)
        mats = angles_to_matrix(angles)
        for R in mats:
            np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)


# ─── TestBatchRotateImages ────────────────────────────────────────────────────

class TestBatchRotateImages:
    def test_returns_list(self):
        imgs = [_bgr(), _bgr()]
        result = batch_rotate_images(imgs, [30.0, 45.0])
        assert isinstance(result, list)

    def test_length_preserved(self):
        imgs = [_bgr()] * 4
        result = batch_rotate_images(imgs, [0.0, 90.0, 180.0, 270.0])
        assert len(result) == 4

    def test_empty_input(self):
        result = batch_rotate_images([], [])
        assert result == []

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_rotate_images([_bgr(), _bgr()], [30.0])

    def test_results_are_ndarrays(self):
        imgs = [_bgr()]
        result = batch_rotate_images(imgs, [45.0])
        assert isinstance(result[0], np.ndarray)


# ─── TestEstimateRotation ─────────────────────────────────────────────────────

class TestEstimateRotation:
    def test_identity_rotation(self):
        pts = _square_pts(2.0)
        angle = estimate_rotation(pts, pts)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_known_rotation(self):
        pts = _square_pts(2.0)
        theta = math.pi / 4  # 45 degrees
        rotated = rotate_points_angle(pts, theta)
        angle = estimate_rotation(pts, rotated)
        assert abs(angle - theta) == pytest.approx(0.0, abs=1e-4)

    def test_returns_float(self):
        pts = _square_pts()
        assert isinstance(estimate_rotation(pts, pts), float)

    def test_result_in_neg_pi_pi(self):
        pts = _square_pts()
        rotated = rotate_points_angle(pts, 1.0)
        angle = estimate_rotation(pts, rotated)
        assert -math.pi <= angle <= math.pi

    def test_invalid_shape_raises(self):
        pts = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            estimate_rotation(pts, pts)

    def test_single_point_raises(self):
        pts = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            estimate_rotation(pts, pts)

    def test_various_angles(self):
        pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
        for theta in [0.3, -0.7, math.pi / 3, -math.pi / 4]:
            rotated = rotate_points_angle(pts, theta)
            estimated = estimate_rotation(pts, rotated)
            assert abs(estimated - theta) == pytest.approx(0.0, abs=1e-4)
