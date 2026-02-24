"""Extra tests for puzzle_reconstruction/utils/image_transform_utils.py."""
from __future__ import annotations

import math
import pytest
import numpy as np
import cv2

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=32, w=32, c=3) -> np.ndarray:
    return np.zeros((h, w, c), dtype=np.uint8)


def _gray(h=32, w=32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ─── ImageTransformConfig ─────────────────────────────────────────────────────

class TestImageTransformConfigExtra:
    def test_default_border_value(self):
        assert ImageTransformConfig().border_value == 255

    def test_border_value_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ImageTransformConfig(border_value=300)

    def test_border_value_negative_raises(self):
        with pytest.raises(ValueError):
            ImageTransformConfig(border_value=-1)

    def test_custom_border_value(self):
        cfg = ImageTransformConfig(border_value=0)
        assert cfg.border_value == 0


# ─── TransformResult ──────────────────────────────────────────────────────────

class TestTransformResultExtra:
    def test_to_dict_keys(self):
        r = TransformResult(image=_img(), angle_rad=0.1, scale=1.0,
                             translation=(0.0, 0.0))
        d = r.to_dict()
        for k in ("shape", "angle_deg", "scale", "translation"):
            assert k in d

    def test_angle_deg_conversion(self):
        r = TransformResult(image=_img(), angle_rad=math.pi / 2,
                             scale=1.0, translation=(0.0, 0.0))
        assert r.to_dict()["angle_deg"] == pytest.approx(90.0)


# ─── rotate_image ─────────────────────────────────────────────────────────────

class TestRotateImageExtra:
    def test_returns_same_shape(self):
        img = _img(32, 32)
        result = rotate_image(img, 0.0)
        assert result.shape == img.shape

    def test_zero_rotation_preserved(self):
        img = _img(16, 16)
        result = rotate_image(img, 0.0)
        assert result.shape == img.shape

    def test_works_with_gray(self):
        img = _gray(32, 32)
        result = rotate_image(img, math.pi / 4)
        assert result.shape == img.shape


# ─── flip_horizontal / flip_vertical ──────────────────────────────────────────

class TestFlipExtra:
    def test_flip_h_changes_image(self):
        img = np.zeros((4, 8, 3), dtype=np.uint8)
        img[:, :4] = 128
        flipped = flip_horizontal(img)
        assert not np.array_equal(flipped, img)

    def test_flip_h_same_shape(self):
        img = _img()
        assert flip_horizontal(img).shape == img.shape

    def test_flip_v_same_shape(self):
        img = _img()
        assert flip_vertical(img).shape == img.shape

    def test_flip_twice_identity(self):
        img = _img(16, 16)
        img[0, 0] = 100
        assert np.array_equal(flip_horizontal(flip_horizontal(img)), img)


# ─── pad_image ────────────────────────────────────────────────────────────────

class TestPadImageExtra:
    def test_size_increases(self):
        img = _img(16, 16)
        padded = pad_image(img, top=5, bottom=5, left=5, right=5)
        assert padded.shape[0] == 26 and padded.shape[1] == 26

    def test_no_pad_unchanged(self):
        img = _img(16, 16)
        assert pad_image(img, 0, 0, 0, 0).shape == img.shape

    def test_gray_image(self):
        img = _gray(8, 8)
        padded = pad_image(img, top=2, bottom=2, left=2, right=2)
        assert padded.shape == (12, 12)


# ─── crop_image ───────────────────────────────────────────────────────────────

class TestCropImageExtra:
    def test_returns_region(self):
        img = _img(32, 32)
        crop = crop_image(img, 0, 0, 16, 16)
        assert crop.shape[:2] == (16, 16)

    def test_clamps_to_bounds(self):
        img = _img(16, 16)
        crop = crop_image(img, -5, -5, 100, 100)
        assert crop.shape[:2] == (16, 16)


# ─── resize_image ─────────────────────────────────────────────────────────────

class TestResizeImageExtra:
    def test_target_size(self):
        img = _img(64, 32)
        result = resize_image(img, (24, 48))
        assert result.shape[:2] == (48, 24)

    def test_returns_array(self):
        assert isinstance(resize_image(_img(), (16, 16)), np.ndarray)


# ─── resize_to_max_side ───────────────────────────────────────────────────────

class TestResizeToMaxSideExtra:
    def test_small_unchanged(self):
        img = _img(32, 32)
        result = resize_to_max_side(img, 100)
        assert result.shape[:2] == (32, 32)

    def test_large_resized(self):
        img = _img(256, 256)
        result = resize_to_max_side(img, 64)
        assert max(result.shape[:2]) <= 64

    def test_preserves_aspect(self):
        img = _img(200, 100)
        result = resize_to_max_side(img, 100)
        h, w = result.shape[:2]
        assert abs(h / w - 2.0) < 0.1


# ─── rotation_matrix_2x3 ──────────────────────────────────────────────────────

class TestRotationMatrix2x3Extra:
    def test_returns_correct_shape(self):
        M = rotation_matrix_2x3(0.0, 16.0, 16.0)
        assert M.shape == (2, 3)

    def test_zero_angle_identity_translation(self):
        M = rotation_matrix_2x3(0.0, 0.0, 0.0, scale=1.0)
        # At zero rotation, [0,0]=1 [1,1]=1 off-diag=0
        assert M[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert M[1, 1] == pytest.approx(1.0, abs=1e-5)


# ─── batch helpers ────────────────────────────────────────────────────────────

class TestBatchTransformExtra:
    def test_batch_rotate_length(self):
        imgs = [_img(), _img()]
        result = batch_rotate(imgs, 0.1)
        assert len(result) == 2

    def test_batch_pad_length(self):
        imgs = [_img(), _img()]
        result = batch_pad(imgs, pad=4)
        assert len(result) == 2

    def test_batch_pad_size_increases(self):
        imgs = [_img(16, 16)]
        result = batch_pad(imgs, pad=4)
        assert result[0].shape[0] == 24

    def test_batch_resize_to_max_length(self):
        imgs = [_img(200, 200), _img(100, 100)]
        result = batch_resize_to_max(imgs, 64)
        assert len(result) == 2

    def test_batch_empty_input(self):
        assert batch_rotate([], 0.0) == []
        assert batch_pad([], 4) == []
        assert batch_resize_to_max([], 64) == []
