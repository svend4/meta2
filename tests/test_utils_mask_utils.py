"""Tests for puzzle_reconstruction/utils/mask_utils.py"""
import pytest
import numpy as np

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


# ─── create_alpha_mask ───────────────────────────────────────────────────────

class TestCreateAlphaMask:
    def test_shape(self):
        mask = create_alpha_mask(50, 80)
        assert mask.shape == (50, 80)

    def test_dtype(self):
        mask = create_alpha_mask(10, 10)
        assert mask.dtype == np.uint8

    def test_default_fill_255(self):
        mask = create_alpha_mask(5, 5)
        assert (mask == 255).all()

    def test_fill_zero(self):
        mask = create_alpha_mask(5, 5, fill=0)
        assert (mask == 0).all()

    def test_custom_fill(self):
        mask = create_alpha_mask(5, 5, fill=128)
        assert (mask == 128).all()

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(0, 10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(10, 0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(-1, 10)

    def test_fill_clipped_to_255(self):
        mask = create_alpha_mask(5, 5, fill=300)
        assert (mask == 255).all()

    def test_fill_clipped_to_zero(self):
        mask = create_alpha_mask(5, 5, fill=-10)
        assert (mask == 0).all()


# ─── apply_mask ──────────────────────────────────────────────────────────────

class TestApplyMask:
    def _make_img_mask(self, h=50, w=50, ndim=2):
        if ndim == 2:
            img = np.ones((h, w), dtype=np.uint8) * 200
        else:
            img = np.ones((h, w, 3), dtype=np.uint8) * 200
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        return img, mask

    def test_basic_grayscale(self):
        img, mask = self._make_img_mask(ndim=2)
        result = apply_mask(img, mask, fill=0)
        assert result[0, 0] == 0     # outside mask → fill
        assert result[25, 25] == 200  # inside mask → preserved

    def test_basic_bgr(self):
        img, mask = self._make_img_mask(ndim=3)
        result = apply_mask(img, mask, fill=0)
        assert (result[0, 0] == 0).all()     # outside
        assert (result[25, 25] == 200).all()  # inside

    def test_fill_value_applied(self):
        img, mask = self._make_img_mask(ndim=2)
        result = apply_mask(img, mask, fill=99)
        assert result[0, 0] == 99

    def test_shape_mismatch_raises(self):
        img = np.ones((50, 50), dtype=np.uint8)
        mask = np.zeros((60, 60), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_mask(img, mask)

    def test_full_mask_unchanged(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = apply_mask(img, mask, fill=0)
        np.testing.assert_array_equal(result, img)

    def test_zero_mask_all_fill(self):
        img = np.full((10, 10), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = apply_mask(img, mask, fill=0)
        assert (result == 0).all()

    def test_output_same_shape(self):
        img = np.ones((30, 40, 3), dtype=np.uint8) * 150
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5:25, 5:35] = 255
        result = apply_mask(img, mask)
        assert result.shape == img.shape


# ─── erode_mask ──────────────────────────────────────────────────────────────

class TestErodeMask:
    def _make_full_mask(self, h=50, w=50):
        return np.full((h, w), 255, dtype=np.uint8)

    def test_erode_reduces_area(self):
        """Erode a mask that has a non-255 border, so edge pixels get eroded."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:45, 5:45] = 255  # inner square, surrounded by zeros
        eroded = erode_mask(mask, ksize=5, iterations=1)
        assert eroded.sum() < mask.sum()

    def test_output_shape(self):
        mask = self._make_full_mask(40, 60)
        eroded = erode_mask(mask)
        assert eroded.shape == (40, 60)

    def test_dtype_uint8(self):
        mask = self._make_full_mask()
        eroded = erode_mask(mask)
        assert eroded.dtype == np.uint8

    def test_zero_mask_unchanged(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        eroded = erode_mask(mask)
        assert (eroded == 0).all()

    def test_multiple_iterations_more_erosion(self):
        mask = self._make_full_mask(60, 60)
        e1 = erode_mask(mask, ksize=3, iterations=1)
        e3 = erode_mask(mask, ksize=3, iterations=3)
        assert e3.sum() <= e1.sum()

    def test_small_ksize(self):
        mask = self._make_full_mask()
        eroded = erode_mask(mask, ksize=1)
        assert eroded.shape == mask.shape


# ─── dilate_mask ─────────────────────────────────────────────────────────────

class TestDilateMask:
    def _make_small_mask(self, h=50, w=50):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        return mask

    def test_dilate_increases_area(self):
        mask = self._make_small_mask()
        dilated = dilate_mask(mask, ksize=5, iterations=1)
        assert dilated.sum() > mask.sum()

    def test_output_shape(self):
        mask = self._make_small_mask(40, 60)
        dilated = dilate_mask(mask)
        assert dilated.shape == (40, 60)

    def test_dtype_uint8(self):
        mask = self._make_small_mask()
        dilated = dilate_mask(mask)
        assert dilated.dtype == np.uint8

    def test_full_mask_unchanged(self):
        mask = np.full((30, 30), 255, dtype=np.uint8)
        dilated = dilate_mask(mask)
        assert (dilated == 255).all()

    def test_multiple_iterations_more_dilation(self):
        mask = self._make_small_mask()
        d1 = dilate_mask(mask, ksize=3, iterations=1)
        d3 = dilate_mask(mask, ksize=3, iterations=3)
        assert d3.sum() >= d1.sum()

    def test_erode_then_dilate_keeps_center(self):
        """Eroding then dilating a full-center should preserve center pixels."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        eroded = erode_mask(mask, ksize=3, iterations=2)
        dilated = dilate_mask(eroded, ksize=3, iterations=2)
        # Center should still be active
        assert dilated[25, 25] == 255


# ─── mask_from_contour ───────────────────────────────────────────────────────

class TestMaskFromContour:
    def test_square_contour(self):
        contour = [(10, 10), (40, 10), (40, 40), (10, 40)]
        mask = mask_from_contour(contour, h=50, w=50)
        assert mask.shape == (50, 50)
        assert mask[25, 25] == 255  # inside
        assert mask[5, 5] == 0      # outside

    def test_dtype_uint8(self):
        contour = [(5, 5), (20, 5), (20, 20), (5, 20)]
        mask = mask_from_contour(contour, h=30, w=30)
        assert mask.dtype == np.uint8

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour([(0, 0), (10, 0), (10, 10)], h=0, w=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            mask_from_contour([(0, 0), (10, 0), (10, 10)], h=10, w=0)

    def test_numpy_contour_input(self):
        contour = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
        mask = mask_from_contour(contour, h=50, w=50)
        assert mask[25, 25] == 255

    def test_n1_2_contour(self):
        """(N, 1, 2) contour format."""
        contour = np.array([[[10, 10]], [[40, 10]], [[40, 40]], [[10, 40]]])
        mask = mask_from_contour(contour, h=50, w=50)
        assert mask.shape == (50, 50)

    def test_filled_region_nonzero(self):
        contour = [(0, 0), (49, 0), (49, 49), (0, 49)]
        mask = mask_from_contour(contour, h=50, w=50)
        assert (mask > 0).sum() > 100


# ─── combine_masks ───────────────────────────────────────────────────────────

class TestCombineMasks:
    def _make_pair(self, h=30, w=30):
        m1 = np.zeros((h, w), dtype=np.uint8)
        m1[:15, :] = 255  # top half
        m2 = np.zeros((h, w), dtype=np.uint8)
        m2[10:, :] = 255  # bottom 20 rows
        return m1, m2

    def test_and_mode(self):
        m1, m2 = self._make_pair()
        result = combine_masks(m1, m2, mode="and")
        # Overlap: rows 10-14
        assert result[12, 15] == 255  # in both
        assert result[5, 15] == 0     # only in m1
        assert result[20, 15] == 0    # only in m2

    def test_or_mode(self):
        m1, m2 = self._make_pair()
        result = combine_masks(m1, m2, mode="or")
        assert result[5, 15] == 255   # in m1 only
        assert result[20, 15] == 255  # in m2 only

    def test_xor_mode(self):
        m1, m2 = self._make_pair()
        result = combine_masks(m1, m2, mode="xor")
        # Overlap zone → 0 (both have 255)
        assert result[12, 15] == 0

    def test_unknown_mode_raises(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            combine_masks(m, m, mode="nand")

    def test_shape_mismatch_raises(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            combine_masks(m1, m2)

    def test_and_same_mask(self):
        m = np.full((10, 10), 255, dtype=np.uint8)
        result = combine_masks(m, m, mode="and")
        assert (result == 255).all()

    def test_xor_same_mask(self):
        m = np.full((10, 10), 255, dtype=np.uint8)
        result = combine_masks(m, m, mode="xor")
        assert (result == 0).all()

    def test_output_dtype_uint8(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        result = combine_masks(m, m, mode="or")
        assert result.dtype == np.uint8


# ─── crop_to_mask ────────────────────────────────────────────────────────────

class TestCropToMask:
    def test_basic_crop(self):
        img = np.ones((50, 50), dtype=np.uint8) * 100
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 15:40] = 255
        cropped, bbox = crop_to_mask(img, mask)
        x, y, bw, bh = bbox
        assert bw == 25
        assert bh == 20
        assert cropped.shape == (bh, bw)

    def test_empty_mask_returns_full_img(self):
        img = np.ones((30, 40), dtype=np.uint8)
        mask = np.zeros((30, 40), dtype=np.uint8)
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.shape == img.shape
        x, y, bw, bh = bbox
        assert x == 0 and y == 0 and bw == 40 and bh == 30

    def test_shape_mismatch_raises(self):
        img = np.ones((30, 40), dtype=np.uint8)
        mask = np.zeros((50, 40), dtype=np.uint8)
        with pytest.raises(ValueError):
            crop_to_mask(img, mask)

    def test_bgr_crop(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.ndim == 3
        assert cropped.shape[2] == 3

    def test_full_mask_returns_same_size(self):
        img = np.ones((30, 40), dtype=np.uint8)
        mask = np.full((30, 40), 255, dtype=np.uint8)
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.shape == (30, 40)

    def test_bbox_values_correct(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255  # y=20-59, x=30-69
        _, bbox = crop_to_mask(img, mask)
        x, y, bw, bh = bbox
        assert x == 30
        assert y == 20
        assert bw == 40
        assert bh == 40


# ─── invert_mask ─────────────────────────────────────────────────────────────

class TestInvertMask:
    def test_all_zero_becomes_255(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        inv = invert_mask(mask)
        assert (inv == 255).all()

    def test_all_255_becomes_zero(self):
        mask = np.full((10, 10), 255, dtype=np.uint8)
        inv = invert_mask(mask)
        assert (inv == 0).all()

    def test_double_invert_is_identity(self):
        mask = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        inv_inv = invert_mask(invert_mask(mask))
        np.testing.assert_array_equal(inv_inv, mask)

    def test_shape_preserved(self):
        mask = np.zeros((30, 40), dtype=np.uint8)
        inv = invert_mask(mask)
        assert inv.shape == (30, 40)

    def test_dtype_uint8(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        inv = invert_mask(mask)
        assert inv.dtype == np.uint8

    def test_partial_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        inv = invert_mask(mask)
        assert inv[0, 0] == 255
        assert inv[5, 5] == 0
