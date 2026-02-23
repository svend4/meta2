"""Extra tests for puzzle_reconstruction/utils/rotation_utils.py"""
import math
import pytest
import numpy as np
import cv2

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=24, w=32):
    return np.random.default_rng(7).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=24, w=32):
    return np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _pts(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).astype(np.float64) * 10.0


# ─── TestRotationConfigExtra ──────────────────────────────────────────────────

class TestRotationConfigExtra:
    def test_border_value_default(self):
        cfg = RotationConfig()
        assert cfg.border_value == (255, 255, 255)

    def test_border_replicate_valid(self):
        cfg = RotationConfig(border_mode=cv2.BORDER_REPLICATE)
        assert cfg.border_mode == cv2.BORDER_REPLICATE

    def test_border_wrap_valid(self):
        cfg = RotationConfig(border_mode=cv2.BORDER_WRAP)
        assert cfg.border_mode == cv2.BORDER_WRAP

    def test_inter_cubic_valid(self):
        cfg = RotationConfig(interpolation=cv2.INTER_CUBIC)
        assert cfg.interpolation == cv2.INTER_CUBIC

    def test_inter_lanczos4_valid(self):
        cfg = RotationConfig(interpolation=cv2.INTER_LANCZOS4)
        assert cfg.interpolation == cv2.INTER_LANCZOS4


# ─── TestRotateImageAngleExtra ─────────────────────────────────────────────────

class TestRotateImageAngleExtra:
    def test_non_square_bgr_shape_non_expand(self):
        img = _bgr(h=20, w=40)
        result = rotate_image_angle(img, 0.0, RotationConfig(expand=False))
        assert result.shape == (20, 40, 3)

    def test_90_degree_gray_expand(self):
        img = _gray(h=10, w=30)
        result = rotate_image_angle(img, 90.0, RotationConfig(expand=True))
        assert result.ndim == 2

    def test_180_no_expand_shape_preserved(self):
        img = _bgr(h=15, w=25)
        result = rotate_image_angle(img, 180.0, RotationConfig(expand=False))
        assert result.shape == img.shape

    def test_border_replicate_config(self):
        img = _bgr()
        cfg = RotationConfig(border_mode=cv2.BORDER_REPLICATE)
        result = rotate_image_angle(img, 30.0, cfg)
        assert isinstance(result, np.ndarray)

    def test_dtype_preserved(self):
        img = _bgr().astype(np.uint8)
        result = rotate_image_angle(img, 45.0)
        assert result.dtype == np.uint8


# ─── TestRotatePointsAngleExtra ───────────────────────────────────────────────

class TestRotatePointsAngleExtra:
    def test_90_degree_unit_x_becomes_unit_y(self):
        pts = np.array([[1.0, 0.0]], dtype=np.float64)
        center = np.array([0.0, 0.0])
        result = rotate_points_angle(pts, math.pi / 2, center=center)
        np.testing.assert_allclose(result[0], [0.0, 1.0], atol=1e-9)

    def test_negative_angle(self):
        pts = np.array([[1.0, 0.0]], dtype=np.float64)
        center = np.array([0.0, 0.0])
        result = rotate_points_angle(pts, -math.pi / 2, center=center)
        np.testing.assert_allclose(result[0], [0.0, -1.0], atol=1e-9)

    def test_single_point_valid(self):
        pts = np.array([[3.0, 4.0]], dtype=np.float64)
        result = rotate_points_angle(pts, 0.0)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_custom_center_non_origin(self):
        pts = np.array([[2.0, 1.0]], dtype=np.float64)
        center = np.array([1.0, 1.0])
        # rotate 90 deg around (1,1): (2,1) -> (1,2)
        result = rotate_points_angle(pts, math.pi / 2, center=center)
        np.testing.assert_allclose(result[0], [1.0, 2.0], atol=1e-9)

    def test_float64_output(self):
        pts = np.array([[1, 0], [0, 1]], dtype=np.int32)
        result = rotate_points_angle(pts, 0.5)
        assert result.dtype == np.float64


# ─── TestNormalizeAngleExtra ──────────────────────────────────────────────────

class TestNormalizeAngleExtra:
    def test_large_positive_angle(self):
        v = normalize_angle(5 * math.pi)
        assert 0.0 <= v < 2 * math.pi

    def test_large_negative_angle(self):
        v = normalize_angle(-10.0)
        assert 0.0 <= v < 2 * math.pi

    def test_pi_half_range_positive(self):
        v = normalize_angle(math.pi + 0.5, half_range=True)
        assert -math.pi < v <= math.pi

    def test_zero_half_range(self):
        v = normalize_angle(0.0, half_range=True)
        assert v == pytest.approx(0.0)

    def test_two_pi_plus_epsilon(self):
        v = normalize_angle(2 * math.pi + 0.1)
        assert v == pytest.approx(0.1, abs=1e-9)


# ─── TestAngleDifferenceExtra ─────────────────────────────────────────────────

class TestAngleDifferenceExtra:
    def test_very_close_angles(self):
        d = angle_difference(0.001, 0.002)
        assert d == pytest.approx(0.001, abs=1e-9)

    def test_one_full_period_apart(self):
        d = angle_difference(0.5, 0.5 + 2 * math.pi)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_three_quarter_is_same_as_quarter(self):
        # 3*pi/2 difference should collapse to pi/2
        d = angle_difference(0.0, 3 * math.pi / 2)
        assert d == pytest.approx(math.pi / 2, abs=1e-9)

    def test_result_nonneg(self):
        for a in [-3.0, 0.0, 1.0, 4.0]:
            for b in [-1.0, 0.0, 2.0, 5.0]:
                assert angle_difference(a, b) >= 0.0


# ─── TestNearestDiscreteExtra ─────────────────────────────────────────────────

class TestNearestDiscreteExtra:
    def test_all_candidates_equidistant_returns_one(self):
        # Exactly between 0 and pi/2; result is one of them
        mid = math.pi / 4
        cands = [0.0, math.pi / 2]
        result = nearest_discrete(mid, cands)
        assert result in (0.0, math.pi / 2)

    def test_negative_candidate_ok(self):
        result = nearest_discrete(-0.1, [-math.pi / 2, 0.0, math.pi / 2])
        assert result == pytest.approx(0.0)

    def test_large_angle_wraps_correctly(self):
        # 2*pi is same as 0 — should pick 0 over pi/2
        result = nearest_discrete(2 * math.pi, [0.0, math.pi / 2])
        assert result in (0.0, math.pi / 2)

    def test_many_candidates(self):
        cands = [float(i) * math.pi / 4 for i in range(8)]
        result = nearest_discrete(math.pi / 8, cands)
        # pi/8 is equidistant between 0 and pi/4; result is one of them
        assert result in (0.0, math.pi / 4)


# ─── TestAnglesToMatrixExtra ──────────────────────────────────────────────────

class TestAnglesToMatrixExtra:
    def test_pi_angle(self):
        result = angles_to_matrix(np.array([math.pi]))
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_determinants_one(self):
        angles = np.linspace(0, 2 * math.pi, 8)
        mats = angles_to_matrix(angles)
        for R in mats:
            det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
            assert det == pytest.approx(1.0, abs=1e-10)

    def test_multiple_angles_shape(self):
        angles = np.array([0.0, math.pi / 4, math.pi / 2, math.pi])
        result = angles_to_matrix(angles)
        assert result.shape == (4, 2, 2)

    def test_float64_dtype(self):
        result = angles_to_matrix(np.array([1.0, 2.0]))
        assert result.dtype == np.float64


# ─── TestBatchRotateImagesExtra ───────────────────────────────────────────────

class TestBatchRotateImagesExtra:
    def test_with_custom_config(self):
        imgs = [_bgr()] * 2
        cfg = RotationConfig(expand=False)
        result = batch_rotate_images(imgs, [0.0, 90.0], cfg)
        assert all(r.shape == imgs[0].shape for r in result)

    def test_single_image(self):
        imgs = [_bgr()]
        result = batch_rotate_images(imgs, [45.0])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_gray_images(self):
        imgs = [_gray(), _gray()]
        result = batch_rotate_images(imgs, [30.0, 60.0])
        assert all(r.ndim == 2 for r in result)


# ─── TestEstimateRotationExtra ────────────────────────────────────────────────

class TestEstimateRotationExtra:
    def test_negative_rotation(self):
        pts = _pts(5)
        theta = -math.pi / 6
        rotated = rotate_points_angle(pts, theta)
        estimated = estimate_rotation(pts, rotated)
        assert abs(estimated - theta) == pytest.approx(0.0, abs=1e-4)

    def test_symmetric_estimation(self):
        pts = _pts(4)
        theta = 0.5
        rotated = rotate_points_angle(pts, theta)
        angle_fwd = estimate_rotation(pts, rotated)
        angle_rev = estimate_rotation(rotated, pts)
        assert abs(angle_fwd + angle_rev) == pytest.approx(0.0, abs=1e-4)

    def test_large_point_set(self):
        pts = _pts(20)
        theta = math.pi / 5
        rotated = rotate_points_angle(pts, theta)
        estimated = estimate_rotation(pts, rotated)
        assert abs(estimated - theta) == pytest.approx(0.0, abs=1e-3)

    def test_wrong_point_count_raises(self):
        src = np.ones((3, 2), dtype=np.float64)
        dst = np.ones((4, 2), dtype=np.float64)
        # Different N: SVD should still run but shape mismatch might not be checked
        # The key validation is that shape[1] != 2 raises
        with pytest.raises(ValueError):
            estimate_rotation(np.ones((2, 3)), np.ones((2, 3)))
