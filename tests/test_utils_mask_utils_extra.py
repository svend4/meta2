"""Extra tests for puzzle_reconstruction/utils/mask_utils.py."""
from __future__ import annotations

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


# ─── create_alpha_mask (extra) ────────────────────────────────────────────────

class TestCreateAlphaMaskExtra:
    def test_rectangular_shape(self):
        mask = create_alpha_mask(20, 40)
        assert mask.shape == (20, 40)

    def test_fill_128(self):
        mask = create_alpha_mask(8, 8, fill=128)
        assert (mask == 128).all()

    def test_fill_1_stored(self):
        mask = create_alpha_mask(4, 4, fill=1)
        assert (mask == 1).all()

    def test_large_dimensions(self):
        mask = create_alpha_mask(512, 512)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8

    def test_1x1_valid(self):
        mask = create_alpha_mask(1, 1)
        assert mask.shape == (1, 1)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            create_alpha_mask(10, -1)

    def test_fill_boundary_254(self):
        mask = create_alpha_mask(3, 3, fill=254)
        assert (mask == 254).all()


# ─── apply_mask (extra) ───────────────────────────────────────────────────────

class TestApplyMaskExtra:
    def test_fill_255_outside(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        result = apply_mask(img, mask, fill=255)
        assert result[0, 0] == 255

    def test_inside_preserved(self):
        img = np.full((20, 20), 77, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        result = apply_mask(img, mask, fill=0)
        assert result[10, 10] == 77

    def test_bgr_fill_applied(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = apply_mask(img, mask, fill=0)
        assert (result == 0).all()

    def test_output_dtype_uint8(self):
        img = np.ones((10, 10), dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = apply_mask(img, mask)
        assert result.dtype == np.uint8

    def test_partial_mask_bgr(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10:, :] = 255
        result = apply_mask(img, mask, fill=0)
        assert (result[:10] == 0).all()
        assert (result[10:] == 100).all()

    def test_output_shape_matches_input(self):
        img = np.ones((30, 50, 3), dtype=np.uint8)
        mask = np.full((30, 50), 255, dtype=np.uint8)
        result = apply_mask(img, mask)
        assert result.shape == (30, 50, 3)


# ─── erode_mask (extra) ───────────────────────────────────────────────────────

class TestErodeMaskExtra:
    def test_values_binary_after_erosion(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:25] = 255
        eroded = erode_mask(mask, ksize=3, iterations=1)
        assert set(np.unique(eroded)).issubset({0, 255})

    def test_large_ksize_erases_small_blob(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[14:16, 14:16] = 255  # 2×2 blob
        eroded = erode_mask(mask, ksize=7, iterations=1)
        assert eroded.sum() == 0

    def test_shape_preserved(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[5:15, 10:30] = 255
        eroded = erode_mask(mask, ksize=3)
        assert eroded.shape == (20, 40)

    def test_iterations_monotone(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:45, 5:45] = 255
        e1 = erode_mask(mask, ksize=3, iterations=1).sum()
        e2 = erode_mask(mask, ksize=3, iterations=2).sum()
        e4 = erode_mask(mask, ksize=3, iterations=4).sum()
        assert e1 >= e2 >= e4

    def test_full_mask_center_preserved(self):
        mask = np.full((20, 20), 255, dtype=np.uint8)
        eroded = erode_mask(mask, ksize=3, iterations=1)
        # Center should still be 255 after mild erosion
        assert eroded[10, 10] == 255


# ─── dilate_mask (extra) ──────────────────────────────────────────────────────

class TestDilateMaskExtra:
    def test_values_binary_after_dilation(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[14:16, 14:16] = 255
        dilated = dilate_mask(mask, ksize=3, iterations=1)
        assert set(np.unique(dilated)).issubset({0, 255})

    def test_zero_mask_stays_zero(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        dilated = dilate_mask(mask, ksize=5, iterations=2)
        assert (dilated == 0).all()

    def test_shape_preserved(self):
        mask = np.zeros((25, 35), dtype=np.uint8)
        mask[10:15, 15:20] = 255
        dilated = dilate_mask(mask)
        assert dilated.shape == (25, 35)

    def test_dilation_monotone_with_iterations(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[18:22, 18:22] = 255
        d1 = dilate_mask(mask, ksize=3, iterations=1).sum()
        d2 = dilate_mask(mask, ksize=3, iterations=2).sum()
        d3 = dilate_mask(mask, ksize=3, iterations=3).sum()
        assert d1 <= d2 <= d3

    def test_dilate_then_erode_subset(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        dilated = dilate_mask(mask, ksize=3, iterations=1)
        eroded = erode_mask(dilated, ksize=3, iterations=1)
        # Original should be subset of re-eroded
        assert (eroded[mask == 255] == 255).all()


# ─── mask_from_contour (extra) ────────────────────────────────────────────────

class TestMaskFromContourExtra:
    def test_triangle_contour(self):
        contour = [(10, 10), (40, 10), (25, 40)]
        mask = mask_from_contour(contour, h=50, w=50)
        assert mask.shape == (50, 50)
        assert (mask > 0).sum() > 0

    def test_larger_canvas(self):
        contour = [(5, 5), (20, 5), (20, 20), (5, 20)]
        mask = mask_from_contour(contour, h=100, w=100)
        assert mask.shape == (100, 100)
        assert mask[12, 12] == 255

    def test_outside_is_zero(self):
        contour = [(10, 10), (30, 10), (30, 30), (10, 30)]
        mask = mask_from_contour(contour, h=50, w=50)
        assert mask[0, 0] == 0
        assert mask[49, 49] == 0

    def test_dtype_is_uint8(self):
        contour = [(0, 0), (10, 0), (10, 10), (0, 10)]
        mask = mask_from_contour(contour, h=20, w=20)
        assert mask.dtype == np.uint8

    def test_numpy_array_contour(self):
        contour = np.array([[5, 5], [25, 5], [25, 25], [5, 25]])
        mask = mask_from_contour(contour, h=30, w=30)
        assert mask[15, 15] == 255


# ─── combine_masks (extra) ────────────────────────────────────────────────────

class TestCombineMasksExtra:
    def test_or_union_full(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[:5, :] = 255
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[5:, :] = 255
        result = combine_masks(m1, m2, mode="or")
        assert (result == 255).all()

    def test_and_disjoint_all_zero(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[:5, :] = 255
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[5:, :] = 255
        result = combine_masks(m1, m2, mode="and")
        assert (result == 0).all()

    def test_xor_identical_all_zero(self):
        m = np.full((8, 8), 255, dtype=np.uint8)
        result = combine_masks(m, m, mode="xor")
        assert (result == 0).all()

    def test_or_with_zero_mask_identity(self):
        m = np.full((10, 10), 255, dtype=np.uint8)
        zeros = np.zeros((10, 10), dtype=np.uint8)
        result = combine_masks(m, zeros, mode="or")
        assert (result == 255).all()

    def test_and_with_full_mask_identity(self):
        m = np.full((10, 10), 255, dtype=np.uint8)
        full = np.full((10, 10), 255, dtype=np.uint8)
        result = combine_masks(m, full, mode="and")
        assert (result == 255).all()

    def test_output_shape_preserved(self):
        m = np.zeros((15, 25), dtype=np.uint8)
        result = combine_masks(m, m, mode="or")
        assert result.shape == (15, 25)


# ─── crop_to_mask (extra) ─────────────────────────────────────────────────────

class TestCropToMaskExtra:
    def test_bgr_crop_shape(self):
        img = np.ones((40, 60, 3), dtype=np.uint8) * 150
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 20:50] = 255
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.ndim == 3
        x, y, bw, bh = bbox
        assert bw == 30
        assert bh == 20

    def test_bbox_x_y_correct(self):
        img = np.zeros((50, 80), dtype=np.uint8)
        mask = np.zeros((50, 80), dtype=np.uint8)
        mask[15:35, 25:65] = 255
        _, bbox = crop_to_mask(img, mask)
        x, y, bw, bh = bbox
        assert x == 25
        assert y == 15

    def test_single_pixel_mask(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255
        cropped, bbox = crop_to_mask(img, mask)
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] >= 1

    def test_output_values_in_crop(self):
        img = np.full((30, 30), 99, dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        cropped, _ = crop_to_mask(img, mask)
        assert (cropped == 99).all()

    def test_full_mask_bbox_covers_all(self):
        img = np.zeros((20, 30), dtype=np.uint8)
        mask = np.full((20, 30), 255, dtype=np.uint8)
        _, bbox = crop_to_mask(img, mask)
        x, y, bw, bh = bbox
        assert bw == 30
        assert bh == 20


# ─── invert_mask (extra) ──────────────────────────────────────────────────────

class TestInvertMaskExtra:
    def test_arbitrary_value_inverted(self):
        mask = np.array([[100, 200], [50, 0]], dtype=np.uint8)
        inv = invert_mask(mask)
        np.testing.assert_array_equal(inv, 255 - mask)

    def test_invert_rectangular(self):
        mask = np.zeros((10, 20), dtype=np.uint8)
        inv = invert_mask(mask)
        assert inv.shape == (10, 20)
        assert (inv == 255).all()

    def test_invert_single_pixel(self):
        mask = np.array([[0]], dtype=np.uint8)
        inv = invert_mask(mask)
        assert inv[0, 0] == 255

    def test_invert_sum_plus_original_equals_255(self):
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        inv = invert_mask(mask)
        np.testing.assert_array_equal(mask.astype(int) + inv.astype(int),
                                       np.full_like(mask, 255, dtype=int))

    def test_double_invert_non_binary(self):
        mask = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(invert_mask(invert_mask(mask)), mask)
