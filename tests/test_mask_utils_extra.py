"""Extra tests for puzzle_reconstruction/utils/mask_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.mask_utils import (
    apply_mask,
    combine_masks,
    create_alpha_mask,
    crop_to_mask,
    dilate_mask,
    erode_mask,
    invert_mask,
    mask_from_contour,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _black(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _half(h: int = 32, w: int = 32) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[:, : w // 2] = 255
    return m


def _gray_img(h: int = 32, w: int = 32, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _bgr_img(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _rect_contour(x: int = 5, y: int = 5, w: int = 20, h: int = 20) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


# ─── create_alpha_mask (extra) ────────────────────────────────────────────────

class TestCreateAlphaMaskExtra:
    def test_returns_uint8(self):
        m = create_alpha_mask(32, 32)
        assert m.dtype == np.uint8

    def test_shape_correct(self):
        m = create_alpha_mask(16, 24)
        assert m.shape == (16, 24)

    def test_default_fill_255(self):
        m = create_alpha_mask(8, 8)
        assert m.min() == 255 and m.max() == 255

    def test_fill_zero(self):
        m = create_alpha_mask(8, 8, fill=0)
        assert m.max() == 0

    def test_fill_128(self):
        m = create_alpha_mask(8, 8, fill=128)
        assert m[0, 0] == 128

    def test_fill_clipped_above_255(self):
        m = create_alpha_mask(4, 4, fill=300)
        assert m[0, 0] == 255

    def test_fill_clipped_below_0(self):
        m = create_alpha_mask(4, 4, fill=-10)
        assert m[0, 0] == 0

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(0, 10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(10, 0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(-1, 10)

    def test_large_mask(self):
        m = create_alpha_mask(256, 256)
        assert m.shape == (256, 256)


# ─── apply_mask (extra) ───────────────────────────────────────────────────────

class TestApplyMaskExtra:
    def test_returns_same_shape(self):
        img = _gray_img()
        mask = _white()
        result = apply_mask(img, mask)
        assert result.shape == img.shape

    def test_dtype_preserved(self):
        result = apply_mask(_gray_img(), _white())
        assert result.dtype == np.uint8

    def test_white_mask_preserves_image(self):
        img = _gray_img(val=100)
        result = apply_mask(img, _white())
        assert (result == img).all()

    def test_black_mask_fills_all(self):
        img = _gray_img(val=200)
        result = apply_mask(img, _black(), fill=0)
        assert result.max() == 0

    def test_half_mask(self):
        img = _gray_img(val=150)
        result = apply_mask(img, _half())
        assert result[0, 0] == 150         # left side preserved
        assert result[0, 31] == 0          # right side filled

    def test_custom_fill_value(self):
        img = _gray_img(val=150)
        result = apply_mask(img, _black(), fill=77)
        assert result[0, 0] == 77

    def test_shape_mismatch_raises(self):
        img = _gray_img(32, 32)
        mask = _white(16, 16)
        with pytest.raises(ValueError):
            apply_mask(img, mask)

    def test_bgr_image(self):
        img = _bgr_img()
        mask = _white()
        result = apply_mask(img, mask)
        assert result.shape == img.shape

    def test_bgr_black_mask_fills(self):
        img = _bgr_img()
        result = apply_mask(img, _black(), fill=0)
        assert result.max() == 0


# ─── erode_mask (extra) ───────────────────────────────────────────────────────

class TestErodeMaskExtra:
    def test_returns_uint8(self):
        result = erode_mask(_white())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        m = _white(16, 24)
        result = erode_mask(m)
        assert result.shape == (16, 24)

    def test_white_mask_erodes(self):
        m = _white(32, 32)
        result = erode_mask(m, ksize=5)
        # Border pixels should be eroded
        assert result[0, 0] == 0

    def test_black_mask_stays_black(self):
        result = erode_mask(_black())
        assert result.max() == 0

    def test_ksize_1_no_change(self):
        m = _white()
        result = erode_mask(m, ksize=1)
        assert (result == m).all()

    def test_multiple_iterations(self):
        m = _white()
        r1 = erode_mask(m, ksize=3, iterations=1)
        r2 = erode_mask(m, ksize=3, iterations=3)
        # More iterations → more erosion (fewer non-zero pixels)
        assert r2.sum() <= r1.sum()


# ─── dilate_mask (extra) ──────────────────────────────────────────────────────

class TestDilateMaskExtra:
    def test_returns_uint8(self):
        result = dilate_mask(_half())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        m = _half(16, 24)
        result = dilate_mask(m)
        assert result.shape == (16, 24)

    def test_black_mask_stays_black(self):
        result = dilate_mask(_black())
        assert result.max() == 0

    def test_white_mask_stays_white(self):
        result = dilate_mask(_white())
        assert result.min() == 255

    def test_half_mask_expands(self):
        m = _half()
        result = dilate_mask(m, ksize=5)
        # Right half should gain some white pixels
        assert result[16, 17] > 0

    def test_multiple_iterations_expand_more(self):
        m = _half()
        r1 = dilate_mask(m, ksize=3, iterations=1)
        r3 = dilate_mask(m, ksize=3, iterations=3)
        assert r3.sum() >= r1.sum()


# ─── mask_from_contour (extra) ────────────────────────────────────────────────

class TestMaskFromContourExtra:
    def test_returns_uint8(self):
        m = mask_from_contour(_rect_contour(), 32, 32)
        assert m.dtype == np.uint8

    def test_shape_correct(self):
        m = mask_from_contour(_rect_contour(), 32, 32)
        assert m.shape == (32, 32)

    def test_interior_is_255(self):
        m = mask_from_contour(_rect_contour(5, 5, 20, 20), 32, 32)
        assert m[10, 10] == 255

    def test_exterior_is_0(self):
        m = mask_from_contour(_rect_contour(5, 5, 10, 10), 32, 32)
        assert m[30, 30] == 0

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour(_rect_contour(), 0, 32)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour(_rect_contour(), 32, 0)

    def test_n1_2_contour_format(self):
        pts = _rect_contour().reshape(-1, 1, 2)
        m = mask_from_contour(pts, 32, 32)
        assert m.shape == (32, 32)

    def test_triangle_contour(self):
        pts = np.array([[16, 2], [2, 30], [30, 30]], dtype=np.int32)
        m = mask_from_contour(pts, 32, 32)
        assert m[20, 16] == 255   # center of triangle
        assert m[0, 0] == 0       # corner outside


# ─── combine_masks (extra) ────────────────────────────────────────────────────

class TestCombineMasksExtra:
    def test_and_white_white(self):
        result = combine_masks(_white(), _white(), mode="and")
        assert result.min() == 255

    def test_and_white_black(self):
        result = combine_masks(_white(), _black(), mode="and")
        assert result.max() == 0

    def test_or_black_black(self):
        result = combine_masks(_black(), _black(), mode="or")
        assert result.max() == 0

    def test_or_half_inverted_half(self):
        h = _half()
        inv = invert_mask(h)
        result = combine_masks(h, inv, mode="or")
        assert result.min() == 255

    def test_xor_same_masks_black(self):
        m = _white()
        result = combine_masks(m, m, mode="xor")
        assert result.max() == 0

    def test_xor_different_masks(self):
        result = combine_masks(_white(), _black(), mode="xor")
        assert result.min() == 255

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            combine_masks(_white(16, 16), _white(32, 32))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            combine_masks(_white(), _white(), mode="nand")

    def test_returns_uint8(self):
        result = combine_masks(_half(), _half())
        assert result.dtype == np.uint8

    def test_and_half_half(self):
        m = _half()
        result = combine_masks(m, m, mode="and")
        assert (result == m).all()


# ─── crop_to_mask (extra) ─────────────────────────────────────────────────────

class TestCropToMaskExtra:
    def test_returns_tuple(self):
        result = crop_to_mask(_gray_img(), _white())
        assert isinstance(result, tuple) and len(result) == 2

    def test_empty_mask_returns_full_image(self):
        img = _gray_img()
        cropped, bbox = crop_to_mask(img, _black())
        assert cropped.shape == img.shape

    def test_empty_mask_bbox_full(self):
        img = _gray_img(32, 32)
        _, bbox = crop_to_mask(img, _black())
        assert bbox == (0, 0, 32, 32)

    def test_white_mask_crops_all(self):
        img = _gray_img()
        cropped, _ = crop_to_mask(img, _white())
        assert cropped.shape[:2] == img.shape[:2]

    def test_small_roi_mask(self):
        img = _gray_img(32, 32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 10:20] = 255
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.shape[0] == 10
        assert cropped.shape[1] == 10

    def test_bbox_correct(self):
        img = _gray_img(32, 32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:10, 8:15] = 255
        _, bbox = crop_to_mask(img, mask)
        x, y, w, h = bbox
        assert x == 8
        assert y == 5
        assert w == 7
        assert h == 5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            crop_to_mask(_gray_img(32, 32), _white(16, 16))

    def test_bgr_image_cropped(self):
        img = _bgr_img(32, 32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:20, 5:20] = 255
        cropped, _ = crop_to_mask(img, mask)
        assert cropped.ndim == 3


# ─── invert_mask (extra) ──────────────────────────────────────────────────────

class TestInvertMaskExtra:
    def test_returns_uint8(self):
        result = invert_mask(_white())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        m = _white(16, 24)
        result = invert_mask(m)
        assert result.shape == (16, 24)

    def test_white_becomes_black(self):
        result = invert_mask(_white())
        assert result.max() == 0

    def test_black_becomes_white(self):
        result = invert_mask(_black())
        assert result.min() == 255

    def test_double_invert_identity(self):
        m = _half()
        result = invert_mask(invert_mask(m))
        assert (result == m).all()

    def test_half_mask_inverted(self):
        m = _half()
        inv = invert_mask(m)
        # Left half (255) should become 0
        assert inv[0, 0] == 0
        # Right half (0) should become 255
        assert inv[0, 31] == 255
