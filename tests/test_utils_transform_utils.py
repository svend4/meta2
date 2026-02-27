"""Tests for puzzle_reconstruction.utils.transform_utils.

Note: transform_utils uses cv2 internally but tests use only numpy for
creating synthetic images. We verify output shapes, dtypes, and geometry.
"""
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

np.random.seed(42)


def _gray(h=64, w=64):
    """Create a reproducible grayscale uint8 image."""
    rng = np.random.default_rng(0)
    return (rng.random((h, w)) * 255).astype(np.uint8)


def _color(h=64, w=64):
    """Create a reproducible BGR uint8 image."""
    rng = np.random.default_rng(1)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# ─── rotate_image ─────────────────────────────────────────────────────────────

def test_rotate_preserves_shape_gray():
    img = _gray()
    result = rotate_image(img, 45)
    assert result.shape == img.shape


def test_rotate_preserves_shape_color():
    img = _color()
    result = rotate_image(img, 90)
    assert result.shape == img.shape


def test_rotate_preserves_dtype():
    img = _gray()
    result = rotate_image(img, 30)
    assert result.dtype == np.uint8


def test_rotate_zero_angle_close():
    img = _gray()
    result = rotate_image(img, 0)
    assert result.shape == img.shape


def test_rotate_360_close():
    img = _gray()
    result = rotate_image(img, 360)
    # Should look very similar to the original (modulo interpolation artifacts)
    assert result.shape == img.shape


def test_rotate_custom_center():
    img = _gray()
    result = rotate_image(img, 45, center=(10.0, 10.0))
    assert result.shape == img.shape


def test_rotate_fill_value():
    img = np.zeros((64, 64), dtype=np.uint8)
    result = rotate_image(img, 45, fill=128)
    # Some pixels should be 128 (the fill)
    assert result.max() <= 255
    assert result.dtype == np.uint8


# ─── flip_image ───────────────────────────────────────────────────────────────

def test_flip_horizontal():
    img = _gray()
    result = flip_image(img, mode=1)
    assert result.shape == img.shape
    # First column of flipped == last column of original
    np.testing.assert_array_equal(result[:, 0], img[:, -1])


def test_flip_vertical():
    img = _gray()
    result = flip_image(img, mode=0)
    assert result.shape == img.shape
    np.testing.assert_array_equal(result[0, :], img[-1, :])


def test_flip_both():
    img = _gray()
    result = flip_image(img, mode=-1)
    assert result.shape == img.shape


def test_flip_dtype_preserved():
    img = _gray()
    assert flip_image(img, 1).dtype == np.uint8


# ─── scale_image ─────────────────────────────────────────────────────────────

def test_scale_up():
    img = _gray(32, 32)
    result = scale_image(img, sx=2.0)
    assert result.shape == (64, 64)


def test_scale_down():
    img = _gray(64, 64)
    result = scale_image(img, sx=0.5)
    assert result.shape == (32, 32)


def test_scale_non_uniform():
    img = _gray(64, 32)
    result = scale_image(img, sx=2.0, sy=0.5)
    assert result.shape == (32, 64)


def test_scale_sx_only():
    img = _gray(64, 64)
    result = scale_image(img, sx=1.5)
    assert result.shape == (96, 96)


def test_scale_dtype_preserved():
    img = _gray()
    result = scale_image(img, sx=1.0)
    assert result.dtype == np.uint8


# ─── crop_region ─────────────────────────────────────────────────────────────

def test_crop_basic():
    img = _gray(100, 100)
    result = crop_region(img, x=10, y=10, w=20, h=30)
    assert result.shape == (30, 20)


def test_crop_clamp():
    img = _gray(64, 64)
    result = crop_region(img, x=-5, y=-5, w=30, h=30, clamp=True)
    assert result.shape[0] > 0
    assert result.shape[1] > 0


def test_crop_outside_raises():
    img = _gray(64, 64)
    with pytest.raises(ValueError):
        crop_region(img, x=70, y=70, w=10, h=10, clamp=True)


def test_crop_returns_correct_content():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    result = crop_region(img, x=2, y=3, w=3, h=2)
    np.testing.assert_array_equal(result, img[3:5, 2:5])


# ─── affine_from_params ───────────────────────────────────────────────────────

def test_affine_from_params_shape():
    M = affine_from_params()
    assert M.shape == (2, 3)


def test_affine_from_params_dtype():
    M = affine_from_params()
    assert M.dtype == np.float32


def test_affine_from_params_identity():
    M = affine_from_params(angle=0.0, tx=0.0, ty=0.0, sx=1.0, sy=1.0)
    # Identity: [[1,0,0],[0,1,0]]
    np.testing.assert_allclose(M[:, :2], np.eye(2), atol=1e-5)
    np.testing.assert_allclose(M[:, 2], [0.0, 0.0], atol=1e-5)


def test_affine_from_params_translation():
    M = affine_from_params(tx=5.0, ty=10.0)
    assert M[0, 2] == pytest.approx(5.0, abs=1e-4)
    assert M[1, 2] == pytest.approx(10.0, abs=1e-4)


def test_affine_from_params_scale():
    M = affine_from_params(sx=2.0, sy=3.0)
    assert M[0, 0] == pytest.approx(2.0, abs=1e-4)
    assert M[1, 1] == pytest.approx(3.0, abs=1e-4)


# ─── compose_affines ─────────────────────────────────────────────────────────

def test_compose_affines_shape():
    M1 = affine_from_params(tx=5.0)
    M2 = affine_from_params(ty=3.0)
    result = compose_affines([M1, M2])
    assert result.shape == (2, 3)


def test_compose_affines_empty_raises():
    with pytest.raises(ValueError):
        compose_affines([])


def test_compose_affines_single():
    M = affine_from_params(tx=10.0, ty=5.0)
    result = compose_affines([M])
    np.testing.assert_allclose(result, M, atol=1e-5)


def test_compose_affines_dtype():
    M1 = affine_from_params()
    M2 = affine_from_params()
    result = compose_affines([M1, M2])
    assert result.dtype == np.float32


# ─── apply_affine ─────────────────────────────────────────────────────────────

def test_apply_affine_shape_preserved():
    img = _gray()
    M = affine_from_params()
    result = apply_affine(img, M)
    assert result.shape == img.shape


def test_apply_affine_custom_size():
    img = _gray(64, 64)
    M = affine_from_params()
    result = apply_affine(img, M, size=(128, 128))
    assert result.shape == (128, 128)


def test_apply_affine_dtype_preserved():
    img = _gray()
    M = affine_from_params()
    result = apply_affine(img, M)
    assert result.dtype == np.uint8


# ─── apply_homography ─────────────────────────────────────────────────────────

def test_apply_homography_shape_preserved():
    img = _gray()
    H = np.eye(3, dtype=np.float32)
    result = apply_homography(img, H)
    assert result.shape == img.shape


def test_apply_homography_custom_size():
    img = _gray(64, 64)
    H = np.eye(3, dtype=np.float32)
    result = apply_homography(img, H, size=(128, 128))
    assert result.shape == (128, 128)


def test_apply_homography_dtype():
    img = _gray()
    H = np.eye(3, dtype=np.float64)
    result = apply_homography(img, H)
    assert result.dtype == np.uint8


# ─── batch_rotate ─────────────────────────────────────────────────────────────

def test_batch_rotate_length():
    images = [_gray(32, 32) for _ in range(4)]
    result = batch_rotate(images, 45)
    assert len(result) == 4


def test_batch_rotate_shapes():
    images = [_gray(32, 32) for _ in range(3)]
    result = batch_rotate(images, 90)
    for orig, rot in zip(images, result):
        assert rot.shape == orig.shape


def test_batch_rotate_empty():
    result = batch_rotate([], 45)
    assert result == []


def test_batch_rotate_dtypes():
    images = [_gray(), _color()]
    result = batch_rotate(images, 30)
    for img, rot in zip(images, result):
        assert rot.dtype == img.dtype
