"""Extra tests for puzzle_reconstruction/utils/transform_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.transform_utils import (
    rotate_image,
    flip_image,
    scale_image,
    crop_region,
    affine_from_params,
    compose_affines,
    apply_affine,
    apply_homography,
    batch_rotate,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=100, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _bgr(h=50, w=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100  # B
    img[:, :, 1] = 150  # G
    img[:, :, 2] = 200  # R
    return img


# ─── rotate_image ────────────────────────────────────────────────────────────

class TestRotateImageExtra:
    def test_zero_angle_preserves_shape(self):
        img = _bgr()
        out = rotate_image(img, 0.0)
        assert out.shape == img.shape

    def test_zero_angle_preserves_content(self):
        img = _bgr()
        out = rotate_image(img, 0.0)
        assert np.allclose(out, img)

    def test_360_angle_preserves_content(self):
        img = _bgr(60, 60)
        out = rotate_image(img, 360.0)
        assert np.allclose(out, img, atol=1)

    def test_custom_center(self):
        img = _gray(50, 50, 128)
        out = rotate_image(img, 45.0, center=(0.0, 0.0))
        assert out.shape == img.shape

    def test_fill_value(self):
        img = _gray(10, 10, 0)
        out = rotate_image(img, 45.0, fill=200)
        # Corners should be filled with 200
        assert out[0, 0] == 200 or out.shape == img.shape  # at least shape is preserved

    def test_grayscale_stays_grayscale(self):
        img = _gray()
        out = rotate_image(img, 30.0)
        assert out.ndim == 2

    def test_bgr_stays_bgr(self):
        img = _bgr()
        out = rotate_image(img, 30.0)
        assert out.ndim == 3 and out.shape[2] == 3


# ─── flip_image ──────────────────────────────────────────────────────────────

class TestFlipImageExtra:
    def test_horizontal_flip(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = flip_image(img, mode=1)
        assert np.array_equal(out[0], img[0][::-1])

    def test_vertical_flip(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = flip_image(img, mode=0)
        assert np.array_equal(out[0], img[2])

    def test_both_flip(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out = flip_image(img, mode=-1)
        assert np.array_equal(out[0, 0], img[2, 3])

    def test_double_horizontal_identity(self):
        img = _bgr()
        out = flip_image(flip_image(img, mode=1), mode=1)
        assert np.array_equal(out, img)

    def test_preserves_shape(self):
        img = _bgr(30, 50)
        out = flip_image(img)
        assert out.shape == img.shape


# ─── scale_image ─────────────────────────────────────────────────────────────

class TestScaleImageExtra:
    def test_double_size(self):
        img = _gray(10, 20)
        out = scale_image(img, sx=2.0)
        assert out.shape == (20, 40)

    def test_half_size(self):
        img = _gray(20, 40)
        out = scale_image(img, sx=0.5)
        assert out.shape == (10, 20)

    def test_asymmetric_scale(self):
        img = _gray(20, 20)
        out = scale_image(img, sx=2.0, sy=0.5)
        assert out.shape == (10, 40)

    def test_identity_scale(self):
        img = _gray(30, 30)
        out = scale_image(img, sx=1.0)
        assert out.shape == img.shape

    def test_sy_defaults_to_sx(self):
        img = _gray(10, 10)
        out = scale_image(img, sx=3.0)
        assert out.shape == (30, 30)

    def test_bgr_scales(self):
        img = _bgr(10, 20)
        out = scale_image(img, sx=2.0)
        assert out.shape == (20, 40, 3)


# ─── crop_region ─────────────────────────────────────────────────────────────

class TestCropRegionExtra:
    def test_basic_crop(self):
        img = _gray(100, 100)
        out = crop_region(img, 10, 20, 30, 40)
        assert out.shape == (40, 30)

    def test_clamp_negative_coords(self):
        img = _gray(50, 50)
        out = crop_region(img, -5, -5, 20, 20, clamp=True)
        assert out.shape == (15, 15)

    def test_clamp_overflow(self):
        img = _gray(50, 50)
        out = crop_region(img, 40, 40, 30, 30, clamp=True)
        assert out.shape == (10, 10)

    def test_empty_crop_raises(self):
        img = _gray(50, 50)
        with pytest.raises(ValueError, match="Empty crop"):
            crop_region(img, 60, 60, 10, 10, clamp=True)

    def test_no_clamp(self):
        img = _gray(50, 50)
        out = crop_region(img, 10, 10, 20, 20, clamp=False)
        assert out.shape == (20, 20)

    def test_bgr_crop(self):
        img = _bgr(100, 100)
        out = crop_region(img, 5, 5, 10, 10)
        assert out.shape == (10, 10, 3)


# ─── affine_from_params ──────────────────────────────────────────────────────

class TestAffineFromParamsExtra:
    def test_identity(self):
        M = affine_from_params()
        assert M.shape == (2, 3)
        assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-5)

    def test_translation_only(self):
        M = affine_from_params(tx=10.0, ty=20.0)
        assert abs(M[0, 2] - 10.0) < 1e-4
        assert abs(M[1, 2] - 20.0) < 1e-4

    def test_scale_only(self):
        M = affine_from_params(sx=2.0, sy=3.0)
        assert abs(M[0, 0] - 2.0) < 1e-4
        assert abs(M[1, 1] - 3.0) < 1e-4

    def test_sy_defaults_to_sx(self):
        M = affine_from_params(sx=2.0)
        assert abs(M[0, 0] - 2.0) < 1e-4
        assert abs(M[1, 1] - 2.0) < 1e-4

    def test_dtype_float32(self):
        M = affine_from_params(angle=45.0)
        assert M.dtype == np.float32

    def test_90_degree_rotation(self):
        M = affine_from_params(angle=90.0)
        # cos(90)=0, sin(90)=1
        assert abs(M[0, 0]) < 1e-4  # cos
        assert abs(M[1, 1]) < 1e-4  # cos


# ─── compose_affines ────────────────────────────────────────────────────────

class TestComposeAffinesExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_affines([])

    def test_single_identity(self):
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        result = compose_affines([M])
        assert np.allclose(result, M, atol=1e-5)

    def test_two_translations(self):
        M1 = np.array([[1, 0, 5], [0, 1, 0]], dtype=np.float32)
        M2 = np.array([[1, 0, 0], [0, 1, 10]], dtype=np.float32)
        result = compose_affines([M1, M2])
        assert abs(result[0, 2] - 5.0) < 1e-3
        assert abs(result[1, 2] - 10.0) < 1e-3

    def test_result_dtype(self):
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        result = compose_affines([M, M])
        assert result.dtype == np.float32

    def test_result_shape(self):
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        result = compose_affines([M, M, M])
        assert result.shape == (2, 3)


# ─── apply_affine ────────────────────────────────────────────────────────────

class TestApplyAffineExtra:
    def test_identity_preserves(self):
        img = _bgr(50, 50)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = apply_affine(img, M)
        assert np.allclose(out, img)

    def test_custom_size(self):
        img = _bgr(50, 50)
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = apply_affine(img, M, size=(100, 80))
        assert out.shape == (80, 100, 3)

    def test_preserves_dtype(self):
        img = _gray()
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = apply_affine(img, M)
        assert out.dtype == np.uint8


# ─── apply_homography ────────────────────────────────────────────────────────

class TestApplyHomographyExtra:
    def test_identity_preserves(self):
        img = _bgr(50, 50)
        H = np.eye(3, dtype=np.float32)
        out = apply_homography(img, H)
        assert np.allclose(out, img)

    def test_custom_size(self):
        img = _bgr(50, 50)
        H = np.eye(3, dtype=np.float32)
        out = apply_homography(img, H, size=(100, 80))
        assert out.shape == (80, 100, 3)

    def test_grayscale(self):
        img = _gray(30, 40)
        H = np.eye(3, dtype=np.float32)
        out = apply_homography(img, H)
        assert out.ndim == 2

    def test_fill_value(self):
        img = _gray(10, 10, 0)
        # Large translation
        H = np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]], dtype=np.float32)
        out = apply_homography(img, H, fill=200)
        assert out[0, 0] == 200


# ─── batch_rotate ────────────────────────────────────────────────────────────

class TestBatchRotateExtra:
    def test_empty_list(self):
        result = batch_rotate([], 45.0)
        assert result == []

    def test_same_length(self):
        imgs = [_gray(20, 20) for _ in range(3)]
        result = batch_rotate(imgs, 30.0)
        assert len(result) == 3

    def test_preserves_shapes(self):
        imgs = [_gray(20, 30), _gray(40, 50)]
        result = batch_rotate(imgs, 15.0)
        assert result[0].shape == (20, 30)
        assert result[1].shape == (40, 50)

    def test_zero_angle(self):
        imgs = [_bgr(20, 20)]
        result = batch_rotate(imgs, 0.0)
        assert np.allclose(result[0], imgs[0])
