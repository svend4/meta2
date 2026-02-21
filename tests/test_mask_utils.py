"""Тесты для puzzle_reconstruction/utils/mask_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.mask_utils import (
    create_alpha_mask,
    apply_mask,
    erode_mask,
    dilate_mask,
    mask_from_contour,
    combine_masks,
    crop_to_mask,
    invert_mask,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _white_mask(h=32, w=32):
    return np.full((h, w), 255, dtype=np.uint8)


def _black_mask(h=32, w=32):
    return np.zeros((h, w), dtype=np.uint8)


def _circle_mask(h=64, w=64):
    """Маска с белым кругом в центре."""
    import cv2
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (w // 2, h // 2), min(h, w) // 3, 255, -1)
    return m


def _gray_img(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr_img(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _rect_contour(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                     dtype=np.int32)


# ─── create_alpha_mask ────────────────────────────────────────────────────────

class TestCreateAlphaMask:
    def test_returns_ndarray(self):
        assert isinstance(create_alpha_mask(32, 32), np.ndarray)

    def test_shape(self):
        r = create_alpha_mask(24, 48)
        assert r.shape == (24, 48)

    def test_dtype_uint8(self):
        assert create_alpha_mask(16, 16).dtype == np.uint8

    def test_fill_255_default(self):
        r = create_alpha_mask(10, 10)
        assert r.min() == 255
        assert r.max() == 255

    def test_fill_0(self):
        r = create_alpha_mask(10, 10, fill=0)
        assert r.max() == 0

    def test_fill_128(self):
        r = create_alpha_mask(8, 8, fill=128)
        assert r[0, 0] == 128

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(0, 10)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(10, 0)

    def test_negative_h_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(-5, 10)

    def test_negative_w_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(10, -5)

    def test_1x1(self):
        r = create_alpha_mask(1, 1)
        assert r.shape == (1, 1)


# ─── apply_mask ───────────────────────────────────────────────────────────────

class TestApplyMask:
    def test_returns_ndarray(self):
        assert isinstance(apply_mask(_gray_img(), _white_mask()), np.ndarray)

    def test_same_shape_gray(self):
        r = apply_mask(_gray_img(24, 48), _white_mask(24, 48))
        assert r.shape == (24, 48)

    def test_same_shape_bgr(self):
        r = apply_mask(_bgr_img(24, 48), _white_mask(24, 48))
        assert r.shape == (24, 48, 3)

    def test_dtype_uint8(self):
        assert apply_mask(_gray_img(), _white_mask()).dtype == np.uint8

    def test_white_mask_preserves_image(self):
        img = _gray_img(val=77)
        r   = apply_mask(img, _white_mask())
        np.testing.assert_array_equal(r, img)

    def test_black_mask_fills_with_fill(self):
        img = _gray_img(val=100)
        r   = apply_mask(img, _black_mask(), fill=0)
        assert r.max() == 0

    def test_fill_default_0(self):
        img = _gray_img(val=200)
        r   = apply_mask(img, _black_mask())
        assert r.max() == 0

    def test_fill_value_applied(self):
        img  = _gray_img(val=100)
        mask = _black_mask()
        r    = apply_mask(img, mask, fill=42)
        assert r[0, 0] == 42

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_mask(_gray_img(32, 32), _white_mask(16, 16))

    def test_gray_input(self):
        r = apply_mask(_gray_img(), _white_mask())
        assert r.ndim == 2

    def test_bgr_input(self):
        r = apply_mask(_bgr_img(), _white_mask())
        assert r.ndim == 3

    def test_partial_mask(self):
        img  = _gray_img(val=100)
        mask = _black_mask()
        mask[10:20, 10:20] = 255   # маленькая белая область
        r    = apply_mask(img, mask, fill=0)
        # Внутри маски — исходные значения
        assert r[15, 15] == 100
        # Снаружи — fill
        assert r[0, 0] == 0


# ─── erode_mask ───────────────────────────────────────────────────────────────

class TestErodeMask:
    def test_returns_ndarray(self):
        assert isinstance(erode_mask(_circle_mask()), np.ndarray)

    def test_same_shape(self):
        m = _circle_mask()
        assert erode_mask(m).shape == m.shape

    def test_dtype_uint8(self):
        assert erode_mask(_circle_mask()).dtype == np.uint8

    def test_reduces_nonzero_area(self):
        m   = _circle_mask()
        r   = erode_mask(m, ksize=5)
        assert r.sum() <= m.sum()

    def test_white_mask_shrinks(self):
        m = _white_mask(32, 32)
        r = erode_mask(m, ksize=3)
        # Углы должны стать чёрными
        assert r[0, 0] == 0

    def test_black_mask_unchanged(self):
        m = _black_mask()
        r = erode_mask(m, ksize=3)
        assert r.max() == 0

    def test_ksize_param(self):
        m  = _circle_mask()
        r1 = erode_mask(m, ksize=3)
        r2 = erode_mask(m, ksize=7)
        assert r1.sum() >= r2.sum()   # больший ksize → меньше

    def test_iterations_param(self):
        m  = _circle_mask()
        r1 = erode_mask(m, ksize=3, iterations=1)
        r2 = erode_mask(m, ksize=3, iterations=3)
        assert r1.sum() >= r2.sum()


# ─── dilate_mask ──────────────────────────────────────────────────────────────

class TestDilateMask:
    def test_returns_ndarray(self):
        assert isinstance(dilate_mask(_circle_mask()), np.ndarray)

    def test_same_shape(self):
        m = _circle_mask()
        assert dilate_mask(m).shape == m.shape

    def test_dtype_uint8(self):
        assert dilate_mask(_circle_mask()).dtype == np.uint8

    def test_expands_nonzero_area(self):
        m = _circle_mask()
        r = dilate_mask(m, ksize=5)
        assert r.sum() >= m.sum()

    def test_black_mask_expands(self):
        m = _black_mask(32, 32)
        m[16, 16] = 255   # одна белая точка
        r = dilate_mask(m, ksize=3)
        assert r.sum() > m.sum()

    def test_white_mask_unchanged(self):
        m = _white_mask()
        r = dilate_mask(m, ksize=3)
        assert r.min() == 255   # белая маска — дилатация ничего не меняет

    def test_ksize_param(self):
        m  = _circle_mask()
        r1 = dilate_mask(m, ksize=3)
        r2 = dilate_mask(m, ksize=7)
        assert r2.sum() >= r1.sum()

    def test_iterations_param(self):
        m  = _circle_mask()
        r1 = dilate_mask(m, ksize=3, iterations=1)
        r2 = dilate_mask(m, ksize=3, iterations=3)
        assert r2.sum() >= r1.sum()


# ─── mask_from_contour ────────────────────────────────────────────────────────

class TestMaskFromContour:
    def test_returns_ndarray(self):
        c = _rect_contour(5, 5, 20, 20)
        assert isinstance(mask_from_contour(c, 64, 64), np.ndarray)

    def test_shape(self):
        c = _rect_contour(5, 5, 20, 20)
        r = mask_from_contour(c, 48, 80)
        assert r.shape == (48, 80)

    def test_dtype_uint8(self):
        c = _rect_contour(5, 5, 20, 20)
        assert mask_from_contour(c, 64, 64).dtype == np.uint8

    def test_inside_is_255(self):
        c = _rect_contour(10, 10, 20, 20)
        r = mask_from_contour(c, 64, 64)
        assert r[15, 15] == 255

    def test_outside_is_0(self):
        c = _rect_contour(10, 10, 20, 20)
        r = mask_from_contour(c, 64, 64)
        assert r[0, 0] == 0

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour(_rect_contour(0, 0, 5, 5), 0, 10)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour(_rect_contour(0, 0, 5, 5), 10, 0)

    def test_n1_2_input(self):
        c = _rect_contour(5, 5, 20, 20).reshape(-1, 1, 2)
        r = mask_from_contour(c, 64, 64)
        assert r[15, 15] == 255

    def test_float_contour(self):
        c = _rect_contour(5, 5, 20, 20).astype(np.float32)
        r = mask_from_contour(c, 64, 64)
        assert r.dtype == np.uint8


# ─── combine_masks ────────────────────────────────────────────────────────────

class TestCombineMasks:
    def test_returns_ndarray(self):
        assert isinstance(combine_masks(_white_mask(), _black_mask()), np.ndarray)

    def test_shape_preserved(self):
        r = combine_masks(_white_mask(24, 48), _black_mask(24, 48))
        assert r.shape == (24, 48)

    def test_dtype_uint8(self):
        assert combine_masks(_white_mask(), _black_mask()).dtype == np.uint8

    def test_and_white_black_is_zero(self):
        r = combine_masks(_white_mask(), _black_mask(), mode="and")
        assert r.max() == 0

    def test_and_white_white_is_white(self):
        r = combine_masks(_white_mask(), _white_mask(), mode="and")
        assert r.min() == 255

    def test_or_white_black_is_white(self):
        r = combine_masks(_white_mask(), _black_mask(), mode="or")
        assert r.min() == 255

    def test_or_black_black_is_black(self):
        r = combine_masks(_black_mask(), _black_mask(), mode="or")
        assert r.max() == 0

    def test_xor_same_is_zero(self):
        r = combine_masks(_white_mask(), _white_mask(), mode="xor")
        assert r.max() == 0

    def test_xor_different_is_nonzero(self):
        r = combine_masks(_white_mask(), _black_mask(), mode="xor")
        assert r.min() == 255

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            combine_masks(_white_mask(16, 16), _black_mask(32, 32))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            combine_masks(_white_mask(), _black_mask(), mode="nand")


# ─── crop_to_mask ─────────────────────────────────────────────────────────────

class TestCropToMask:
    def test_returns_tuple(self):
        mask = _circle_mask()
        img  = _gray_img(64, 64)
        r    = crop_to_mask(img, mask)
        assert isinstance(r, tuple)
        assert len(r) == 2

    def test_cropped_ndarray(self):
        cropped, _ = crop_to_mask(_gray_img(64, 64), _circle_mask())
        assert isinstance(cropped, np.ndarray)

    def test_bbox_4_tuple(self):
        _, bbox = crop_to_mask(_gray_img(64, 64), _circle_mask())
        assert len(bbox) == 4

    def test_cropped_smaller_than_full(self):
        mask = _circle_mask(64, 64)
        img  = _gray_img(64, 64)
        cropped, _ = crop_to_mask(img, mask)
        assert cropped.shape[0] <= 64
        assert cropped.shape[1] <= 64

    def test_empty_mask_returns_full(self):
        img  = _gray_img(32, 48)
        mask = _black_mask(32, 48)
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.shape == (32, 48)
        assert bbox == (0, 0, 48, 32)

    def test_bbox_within_image(self):
        img  = _gray_img(64, 64)
        mask = _circle_mask(64, 64)
        _, (x, y, bw, bh) = crop_to_mask(img, mask)
        assert x >= 0
        assert y >= 0
        assert x + bw <= 64
        assert y + bh <= 64

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            crop_to_mask(_gray_img(32, 32), _white_mask(16, 16))

    def test_gray_input(self):
        cropped, _ = crop_to_mask(_gray_img(64, 64), _circle_mask(64, 64))
        assert cropped.ndim == 2

    def test_bgr_input(self):
        cropped, _ = crop_to_mask(_bgr_img(64, 64), _circle_mask(64, 64))
        assert cropped.ndim == 3

    def test_full_mask_returns_same_size(self):
        img  = _gray_img(32, 48)
        mask = _white_mask(32, 48)
        cropped, _ = crop_to_mask(img, mask)
        assert cropped.shape == (32, 48)


# ─── invert_mask ──────────────────────────────────────────────────────────────

class TestInvertMask:
    def test_returns_ndarray(self):
        assert isinstance(invert_mask(_white_mask()), np.ndarray)

    def test_same_shape(self):
        r = invert_mask(_white_mask(24, 48))
        assert r.shape == (24, 48)

    def test_dtype_uint8(self):
        assert invert_mask(_white_mask()).dtype == np.uint8

    def test_255_becomes_0(self):
        r = invert_mask(_white_mask())
        assert r.max() == 0

    def test_0_becomes_255(self):
        r = invert_mask(_black_mask())
        assert r.min() == 255

    def test_double_invert_restores(self):
        m = _circle_mask()
        np.testing.assert_array_equal(invert_mask(invert_mask(m)), m)

    def test_partial_mask(self):
        m = _black_mask(10, 10)
        m[5, 5] = 255
        r = invert_mask(m)
        assert r[5, 5] == 0
        assert r[0, 0] == 255
