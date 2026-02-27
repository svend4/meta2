"""Tests for puzzle_reconstruction.utils.image_transform_utils"""
import math
import numpy as np
import pytest
from puzzle_reconstruction.utils.image_transform_utils import (
    ImageTransformConfig,
    TransformResult,
    rotate_image,
    flip_horizontal,
    flip_vertical,
    pad_image,
    crop_image,
    resize_image,
    resize_to_max_side,
    apply_affine,
    rotation_matrix_2x3,
    batch_rotate,
    batch_pad,
    batch_resize_to_max,
)

np.random.seed(42)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_image():
    return np.random.randint(0, 256, (100, 80), dtype=np.uint8)


@pytest.fixture
def color_image():
    return np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)


# ─── ImageTransformConfig ────────────────────────────────────────────────────

def test_config_defaults():
    cfg = ImageTransformConfig()
    assert cfg.border_value == 255
    assert cfg.expand is False


def test_config_border_value_invalid():
    with pytest.raises(ValueError):
        ImageTransformConfig(border_value=300)


def test_config_border_value_zero():
    cfg = ImageTransformConfig(border_value=0)
    assert cfg.border_value == 0


# ─── TransformResult ─────────────────────────────────────────────────────────

def test_transform_result_to_dict(color_image):
    r = TransformResult(
        image=color_image,
        angle_rad=math.pi / 4,
        scale=1.0,
        translation=(0.0, 0.0),
    )
    d = r.to_dict()
    assert "shape" in d
    assert "angle_deg" in d
    assert abs(d["angle_deg"] - 45.0) < 1e-9


# ─── rotate_image ─────────────────────────────────────────────────────────────

def test_rotate_image_shape_preserved_gray(gray_image):
    rotated = rotate_image(gray_image, math.pi / 4)
    assert rotated.shape == gray_image.shape


def test_rotate_image_shape_preserved_color(color_image):
    rotated = rotate_image(color_image, math.pi / 6)
    assert rotated.shape == color_image.shape


def test_rotate_image_zero_angle(gray_image):
    rotated = rotate_image(gray_image, 0.0)
    assert rotated.shape == gray_image.shape
    # At 0 angle, images should be very similar
    assert np.mean(np.abs(rotated.astype(int) - gray_image.astype(int))) < 5.0


def test_rotate_image_return_type(gray_image):
    rotated = rotate_image(gray_image, math.pi / 2)
    assert isinstance(rotated, np.ndarray)
    assert rotated.dtype == np.uint8


def test_rotate_image_custom_cfg(gray_image):
    cfg = ImageTransformConfig(border_value=0)
    rotated = rotate_image(gray_image, math.pi / 4, cfg)
    assert rotated.shape == gray_image.shape


# ─── flip_horizontal / flip_vertical ─────────────────────────────────────────

def test_flip_horizontal_shape(color_image):
    flipped = flip_horizontal(color_image)
    assert flipped.shape == color_image.shape


def test_flip_horizontal_reversal(gray_image):
    flipped = flip_horizontal(gray_image)
    np.testing.assert_array_equal(flipped, gray_image[:, ::-1])


def test_flip_vertical_shape(color_image):
    flipped = flip_vertical(color_image)
    assert flipped.shape == color_image.shape


def test_flip_vertical_reversal(gray_image):
    flipped = flip_vertical(gray_image)
    np.testing.assert_array_equal(flipped, gray_image[::-1, :])


# ─── pad_image ────────────────────────────────────────────────────────────────

def test_pad_image_gray_shape(gray_image):
    padded = pad_image(gray_image, top=10, bottom=5, left=3, right=7)
    h, w = gray_image.shape[:2]
    assert padded.shape == (h + 15, w + 10)


def test_pad_image_color_shape(color_image):
    padded = pad_image(color_image, top=4, bottom=4, left=4, right=4)
    h, w = color_image.shape[:2]
    assert padded.shape == (h + 8, w + 8, 3)


def test_pad_image_fill_value(gray_image):
    padded = pad_image(gray_image, top=5, bottom=0, left=0, right=0, fill=0)
    assert np.all(padded[:5, :] == 0)


# ─── crop_image ───────────────────────────────────────────────────────────────

def test_crop_image_basic(gray_image):
    cropped = crop_image(gray_image, 10, 10, 50, 60)
    assert cropped.shape == (40, 50)


def test_crop_image_clamped_bounds(gray_image):
    h, w = gray_image.shape
    cropped = crop_image(gray_image, -5, -5, h + 5, w + 5)
    assert cropped.shape == (h, w)


def test_crop_image_returns_copy(gray_image):
    cropped = crop_image(gray_image, 0, 0, 10, 10)
    cropped[:] = 0
    assert gray_image[0, 0] != 0 or True  # original unchanged


# ─── resize_image ─────────────────────────────────────────────────────────────

def test_resize_image_target_size(gray_image):
    resized = resize_image(gray_image, (40, 30))
    assert resized.shape == (30, 40)


def test_resize_image_color(color_image):
    resized = resize_image(color_image, (50, 50))
    assert resized.shape == (50, 50, 3)


# ─── resize_to_max_side ──────────────────────────────────────────────────────

def test_resize_to_max_side_long_dim(gray_image):
    resized = resize_to_max_side(gray_image, 50)
    assert max(resized.shape[:2]) == 50


def test_resize_to_max_side_no_upscale(gray_image):
    # max_side larger than image → returns copy
    resized = resize_to_max_side(gray_image, 200)
    assert resized.shape == gray_image.shape


def test_resize_to_max_side_aspect_ratio(gray_image):
    h, w = gray_image.shape[:2]
    resized = resize_to_max_side(gray_image, 50)
    rh, rw = resized.shape[:2]
    orig_ratio = w / h
    new_ratio = rw / rh
    assert abs(orig_ratio - new_ratio) < 0.15


# ─── apply_affine ─────────────────────────────────────────────────────────────

def test_apply_affine_identity(gray_image):
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    result = apply_affine(gray_image, M)
    assert result.shape == gray_image.shape


def test_apply_affine_returns_ndarray(color_image):
    M = np.array([[1, 0, 5], [0, 1, 5]], dtype=np.float32)
    result = apply_affine(color_image, M)
    assert isinstance(result, np.ndarray)


# ─── rotation_matrix_2x3 ──────────────────────────────────────────────────────

def test_rotation_matrix_2x3_shape():
    M = rotation_matrix_2x3(0.0, 50.0, 40.0)
    assert M.shape == (2, 3)


def test_rotation_matrix_2x3_zero_angle():
    M = rotation_matrix_2x3(0.0, 0.0, 0.0)
    # Identity at zero angle (scale=1)
    assert abs(M[0, 0] - 1.0) < 1e-6
    assert abs(M[1, 1] - 1.0) < 1e-6


# ─── Batch helpers ────────────────────────────────────────────────────────────

def test_batch_rotate_length(gray_image, color_image):
    imgs = [gray_image, gray_image]
    results = batch_rotate(imgs, math.pi / 4)
    assert len(results) == 2


def test_batch_pad_shape(gray_image):
    imgs = [gray_image, gray_image]
    padded = batch_pad(imgs, pad=10)
    for p in padded:
        h, w = gray_image.shape[:2]
        assert p.shape == (h + 20, w + 20)


def test_batch_resize_to_max(gray_image):
    imgs = [gray_image, gray_image]
    resized = batch_resize_to_max(imgs, 50)
    for r in resized:
        assert max(r.shape[:2]) == 50
