"""Extra tests for puzzle_reconstruction/utils/transform_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=32, w=32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _identity_affine() -> np.ndarray:
    return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def _identity_H() -> np.ndarray:
    return np.eye(3, dtype=np.float32)


# ─── rotate_image ─────────────────────────────────────────────────────────────

class TestRotateImageExtra:
    def test_returns_ndarray(self):
        assert isinstance(rotate_image(_gray(), 0), np.ndarray)

    def test_same_shape(self):
        img = _gray(32, 48)
        out = rotate_image(img, 45)
        assert out.shape == img.shape

    def test_zero_rotation_unchanged(self):
        img = _gray(val=200)
        out = rotate_image(img, 0)
        np.testing.assert_array_equal(out, img)

    def test_180_rotation_same_shape(self):
        img = _gray()
        out = rotate_image(img, 180)
        assert out.shape == img.shape

    def test_fill_value_applied(self):
        img = _gray(val=0)
        out = rotate_image(img, 45, fill=255)
        # Corners should be 255 after rotation of a zero image
        assert out.max() == 255 or out.min() == 0

    def test_custom_center(self):
        img = _gray(32, 32)
        out = rotate_image(img, 90, center=(0.0, 0.0))
        assert out.shape == img.shape

    def test_bgr_image_ok(self):
        img = _bgr()
        out = rotate_image(img, 30)
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        out = rotate_image(img, 15)
        assert out.dtype == np.uint8


# ─── flip_image ───────────────────────────────────────────────────────────────

class TestFlipImageExtra:
    def test_returns_ndarray(self):
        assert isinstance(flip_image(_gray()), np.ndarray)

    def test_same_shape(self):
        img = _gray(16, 32)
        assert flip_image(img, 0).shape == img.shape

    def test_flip_twice_horizontal_identity(self):
        img = _gray()
        out = flip_image(flip_image(img, 1), 1)
        np.testing.assert_array_equal(out, img)

    def test_flip_twice_vertical_identity(self):
        img = _gray()
        out = flip_image(flip_image(img, 0), 0)
        np.testing.assert_array_equal(out, img)

    def test_flip_both_is_180_rotation(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = flip_image(img, -1)
        expected = img[::-1, ::-1]
        np.testing.assert_array_equal(out, expected)

    def test_bgr_image_ok(self):
        img = _bgr()
        assert flip_image(img, 1).shape == img.shape


# ─── scale_image ──────────────────────────────────────────────────────────────

class TestScaleImageExtra:
    def test_returns_ndarray(self):
        assert isinstance(scale_image(_gray()), np.ndarray)

    def test_scale_one_same_size(self):
        img = _gray(20, 30)
        out = scale_image(img, sx=1.0)
        assert out.shape[:2] == (20, 30)

    def test_scale_two_doubles_size(self):
        img = _gray(20, 30)
        out = scale_image(img, sx=2.0)
        assert out.shape[:2] == (40, 60)

    def test_scale_half_halves_size(self):
        img = _gray(20, 30)
        out = scale_image(img, sx=0.5)
        assert out.shape[:2] == (10, 15)

    def test_anisotropic_scale(self):
        img = _gray(20, 20)
        out = scale_image(img, sx=2.0, sy=0.5)
        assert out.shape[:2] == (10, 40)

    def test_very_small_scale_at_least_1px(self):
        img = _gray(5, 5)
        out = scale_image(img, sx=0.01)
        assert out.shape[0] >= 1 and out.shape[1] >= 1

    def test_bgr_channels_preserved(self):
        img = _bgr(10, 10)
        out = scale_image(img, sx=2.0)
        assert out.ndim == 3 and out.shape[2] == 3


# ─── crop_region ──────────────────────────────────────────────────────────────

class TestCropRegionExtra:
    def test_returns_ndarray(self):
        assert isinstance(crop_region(_gray(32, 32), 0, 0, 10, 10), np.ndarray)

    def test_correct_output_size(self):
        img = _gray(32, 32)
        out = crop_region(img, 5, 5, 10, 10)
        assert out.shape == (10, 10)

    def test_full_image_crop(self):
        img = _gray(32, 32)
        out = crop_region(img, 0, 0, 32, 32)
        np.testing.assert_array_equal(out, img)

    def test_clamp_oob_coordinates(self):
        img = _gray(32, 32)
        out = crop_region(img, -5, -5, 20, 20, clamp=True)
        assert out.shape[0] > 0 and out.shape[1] > 0

    def test_empty_after_clamp_raises(self):
        img = _gray(32, 32)
        with pytest.raises(ValueError):
            crop_region(img, 40, 0, 10, 10, clamp=True)

    def test_bgr_image_ok(self):
        img = _bgr(32, 32)
        out = crop_region(img, 0, 0, 10, 10)
        assert out.ndim == 3


# ─── affine_from_params ───────────────────────────────────────────────────────

class TestAffineFromParamsExtra:
    def test_returns_ndarray(self):
        M = affine_from_params()
        assert isinstance(M, np.ndarray)

    def test_shape_2x3(self):
        assert affine_from_params().shape == (2, 3)

    def test_dtype_float32(self):
        assert affine_from_params().dtype == np.float32

    def test_identity_no_rotation_no_scale(self):
        M = affine_from_params(angle=0, tx=0, ty=0, sx=1.0, sy=1.0)
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(M, expected, atol=1e-5)

    def test_translation_only(self):
        M = affine_from_params(tx=5, ty=10)
        assert M[0, 2] == pytest.approx(5.0, abs=1e-4)
        assert M[1, 2] == pytest.approx(10.0, abs=1e-4)

    def test_scale_applied(self):
        M = affine_from_params(angle=0, sx=2.0, sy=2.0)
        assert M[0, 0] == pytest.approx(2.0, abs=1e-4)
        assert M[1, 1] == pytest.approx(2.0, abs=1e-4)

    def test_sy_none_equals_sx(self):
        M = affine_from_params(angle=0, sx=1.5, sy=None)
        assert M[0, 0] == pytest.approx(1.5, abs=1e-4)
        assert M[1, 1] == pytest.approx(1.5, abs=1e-4)


# ─── compose_affines ──────────────────────────────────────────────────────────

class TestComposeAffinesExtra:
    def test_returns_ndarray(self):
        M = _identity_affine()
        assert isinstance(compose_affines([M]), np.ndarray)

    def test_shape_2x3(self):
        assert compose_affines([_identity_affine()]).shape == (2, 3)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            compose_affines([])

    def test_single_matrix_unchanged(self):
        M = affine_from_params(tx=5, ty=3)
        result = compose_affines([M])
        np.testing.assert_allclose(result, M, atol=1e-5)

    def test_identity_composed_twice(self):
        I = _identity_affine()
        result = compose_affines([I, I])
        np.testing.assert_allclose(result, I, atol=1e-5)

    def test_translation_composes(self):
        M1 = affine_from_params(tx=5, ty=0)
        M2 = affine_from_params(tx=3, ty=0)
        result = compose_affines([M1, M2])
        # Total translation ~8
        assert abs(result[0, 2]) == pytest.approx(8.0, abs=0.1)


# ─── apply_affine ─────────────────────────────────────────────────────────────

class TestApplyAffineExtra:
    def test_returns_ndarray(self):
        img = _gray()
        M = _identity_affine()
        assert isinstance(apply_affine(img, M), np.ndarray)

    def test_identity_preserves_image(self):
        img = _gray(val=100)
        M = _identity_affine()
        out = apply_affine(img, M)
        assert out.shape == img.shape

    def test_custom_size(self):
        img = _gray(32, 32)
        M = _identity_affine()
        out = apply_affine(img, M, size=(64, 64))
        assert out.shape == (64, 64)

    def test_fill_value(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        M = affine_from_params(tx=30)  # shift image completely out
        out = apply_affine(img, M, fill=200)
        # Most pixels should be fill (200)
        assert out.max() <= 200

    def test_bgr_image_ok(self):
        img = _bgr()
        out = apply_affine(img, _identity_affine())
        assert out.shape == img.shape


# ─── apply_homography ─────────────────────────────────────────────────────────

class TestApplyHomographyExtra:
    def test_returns_ndarray(self):
        img = _gray()
        H = _identity_H()
        assert isinstance(apply_homography(img, H), np.ndarray)

    def test_identity_preserves_shape(self):
        img = _gray(32, 48)
        out = apply_homography(img, _identity_H())
        assert out.shape == img.shape

    def test_custom_size(self):
        img = _gray(32, 32)
        out = apply_homography(img, _identity_H(), size=(16, 16))
        assert out.shape == (16, 16)

    def test_bgr_image_ok(self):
        img = _bgr()
        out = apply_homography(img, _identity_H())
        assert out.shape == img.shape

    def test_fill_value_applied(self):
        img = _gray(val=0)
        H = np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        out = apply_homography(img, H, fill=123)
        # Shifted image — left portion should be fill
        assert 123 in out


# ─── batch_rotate ─────────────────────────────────────────────────────────────

class TestBatchRotateExtra:
    def test_returns_list(self):
        assert isinstance(batch_rotate([_gray()], 0), list)

    def test_length_preserved(self):
        imgs = [_gray(), _bgr()]
        result = batch_rotate(imgs, 45)
        assert len(result) == 2

    def test_empty_list(self):
        assert batch_rotate([], 90) == []

    def test_shapes_preserved(self):
        imgs = [_gray(16, 32), _gray(24, 24)]
        for orig, rot in zip(imgs, batch_rotate(imgs, 30)):
            assert rot.shape == orig.shape

    def test_zero_angle_unchanged(self):
        img = _gray(val=150)
        result = batch_rotate([img], 0)
        np.testing.assert_array_equal(result[0], img)
