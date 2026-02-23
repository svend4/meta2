"""Extra tests for puzzle_reconstruction/utils/blend_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.blend_utils import (
    BlendConfig,
    alpha_blend,
    weighted_blend,
    feather_mask,
    paste_with_mask,
    horizontal_blend,
    vertical_blend,
    batch_blend,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img(h=10, w=10, fill=128, channels=3):
    if channels == 1:
        return np.full((h, w), fill, dtype=np.uint8)
    return np.full((h, w, channels), fill, dtype=np.uint8)


# ─── BlendConfig (extra) ──────────────────────────────────────────────────────

class TestBlendConfigExtra:
    def test_large_feather_ok(self):
        cfg = BlendConfig(feather_px=100)
        assert cfg.feather_px == 100

    def test_gamma_1_is_linear(self):
        cfg = BlendConfig(gamma=1.0)
        assert cfg.gamma == pytest.approx(1.0)

    def test_gamma_2_ok(self):
        cfg = BlendConfig(gamma=2.0)
        assert cfg.gamma == pytest.approx(2.0)

    def test_clip_output_true(self):
        cfg = BlendConfig(clip_output=True)
        assert cfg.clip_output is True

    def test_clip_output_false(self):
        cfg = BlendConfig(clip_output=False)
        assert cfg.clip_output is False

    def test_defaults_immutable(self):
        c1 = BlendConfig()
        c2 = BlendConfig()
        assert c1.feather_px == c2.feather_px
        assert c1.gamma == c2.gamma

    def test_large_gamma_ok(self):
        cfg = BlendConfig(gamma=10.0)
        assert cfg.gamma == pytest.approx(10.0)


# ─── alpha_blend (extra) ──────────────────────────────────────────────────────

class TestAlphaBlendExtra:
    def test_alpha_quarter_near_src_weighted(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 0, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.25)
        np.testing.assert_array_equal(result, np.full((4, 4, 3), 50, dtype=np.uint8))

    def test_alpha_three_quarter(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 0, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.75)
        np.testing.assert_array_equal(result, np.full((4, 4, 3), 150, dtype=np.uint8))

    def test_grayscale_blended_correct(self):
        src = np.full((4, 4), 100, dtype=np.uint8)
        dst = np.full((4, 4), 200, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert np.all(result == 150)

    def test_shape_mismatch_channels_raises(self):
        src = _img(fill=100, channels=3)
        dst = _img(fill=50, channels=1)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=0.5)

    def test_result_shape_matches_input(self):
        src = _img(h=5, w=7, channels=3)
        dst = _img(h=5, w=7, channels=3)
        result = alpha_blend(src, dst, alpha=0.3)
        assert result.shape == (5, 7, 3)

    def test_output_within_uint8_range(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.7)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_alpha_zero_returns_dst(self):
        src = _img(fill=200)
        dst = _img(fill=0)
        result = alpha_blend(src, dst, alpha=0.0)
        np.testing.assert_array_equal(result, dst)

    def test_symmetric_at_half_alpha(self):
        src = np.full((4, 4, 3), 100, dtype=np.uint8)
        dst = np.full((4, 4, 3), 200, dtype=np.uint8)
        r1 = alpha_blend(src, dst, alpha=0.5)
        r2 = alpha_blend(dst, src, alpha=0.5)
        np.testing.assert_array_equal(r1, r2)


# ─── weighted_blend (extra) ───────────────────────────────────────────────────

class TestWeightedBlendExtra:
    def test_three_images_equal_weights(self):
        a = np.full((4, 4, 3), 30, dtype=np.uint8)
        b = np.full((4, 4, 3), 60, dtype=np.uint8)
        c = np.full((4, 4, 3), 90, dtype=np.uint8)
        result = weighted_blend([a, b, c], weights=[1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, np.full((4, 4, 3), 60, dtype=np.uint8))

    def test_one_image_full_weight(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[1.0, 0.0])
        np.testing.assert_array_equal(result, a)

    def test_grayscale_weighted_blend(self):
        a = np.full((4, 4), 0, dtype=np.uint8)
        b = np.full((4, 4), 200, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[1.0, 3.0])
        # b weight = 0.75, a weight = 0.25 → 150
        np.testing.assert_array_equal(result, np.full((4, 4), 150, dtype=np.uint8))

    def test_output_clipped(self):
        a = np.full((4, 4, 3), 255, dtype=np.uint8)
        b = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = weighted_blend([a, b])
        assert result.max() <= 255

    def test_uniform_weights_equal_simple_average(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[1.0, 1.0])
        assert result.dtype == np.uint8

    def test_five_images(self):
        imgs = [np.full((4, 4, 3), 50, dtype=np.uint8) for _ in range(5)]
        result = weighted_blend(imgs)
        np.testing.assert_array_equal(result, imgs[0])


# ─── feather_mask (extra) ─────────────────────────────────────────────────────

class TestFeatherMaskExtra:
    def test_symmetric_horizontally(self):
        mask = feather_mask(10, 20, feather_px=4)
        # Left-right symmetry
        np.testing.assert_allclose(mask, mask[:, ::-1], atol=1e-6)

    def test_symmetric_vertically(self):
        mask = feather_mask(20, 10, feather_px=4)
        np.testing.assert_allclose(mask, mask[::-1, :], atol=1e-6)

    def test_square_center_max(self):
        mask = feather_mask(20, 20, feather_px=5)
        center = mask[10, 10]
        assert center == pytest.approx(mask.max())

    def test_large_mask(self):
        mask = feather_mask(512, 512, feather_px=20)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32

    def test_feather_10_smaller_at_edges(self):
        mask = feather_mask(20, 20, feather_px=10)
        assert mask[0, 0] < mask[10, 10]
        assert mask[19, 19] < mask[10, 10]

    def test_all_values_between_0_and_1(self):
        mask = feather_mask(30, 40, feather_px=6)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


# ─── paste_with_mask (extra) ──────────────────────────────────────────────────

class TestPasteWithMaskExtra:
    def test_half_mask_blended(self):
        canvas = np.full((10, 10, 3), 0, dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.full((4, 4), 0.5, dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert np.all(result[:4, :4] == 100)

    def test_negative_offset_clamped(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        # Negative y should be handled (clamped or clipped)
        result = paste_with_mask(canvas, patch, mask, y=-2, x=0)
        assert result.shape == canvas.shape

    def test_center_paste(self):
        canvas = np.zeros((20, 20, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 255, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=8, x=8)
        assert np.all(result[8:12, 8:12] == 255)

    def test_grayscale_paste_correct(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        patch = np.full((4, 4), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert np.all(result[:4, :4] == 200)

    def test_canvas_unchanged_outside_paste(self):
        canvas = np.full((10, 10, 3), 50, dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        np.testing.assert_array_equal(result[4:, :], canvas[4:, :])
        np.testing.assert_array_equal(result[:, 4:], canvas[:, 4:])


# ─── horizontal_blend (extra) ─────────────────────────────────────────────────

class TestHorizontalBlendExtra:
    def test_overlap_zero_output_is_concatenation(self):
        left = np.full((4, 6, 3), 100, dtype=np.uint8)
        right = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape == (4, 10, 3)

    def test_output_dtype_uint8(self):
        left = _img(h=4, w=8)
        right = _img(h=4, w=6)
        result = horizontal_blend(left, right, overlap=2)
        assert result.dtype == np.uint8

    def test_overlap_larger_than_right_raises(self):
        left = _img(h=4, w=10)
        right = _img(h=4, w=4)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=5)

    def test_channels_preserved(self):
        left = _img(h=4, w=8, channels=3)
        right = _img(h=4, w=6, channels=3)
        result = horizontal_blend(left, right, overlap=0)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_grayscale_result_2d(self):
        left = _img(h=4, w=8, channels=1)
        right = _img(h=4, w=6, channels=1)
        result = horizontal_blend(left, right, overlap=0)
        assert result.ndim == 2

    def test_different_widths_no_overlap(self):
        left = _img(h=5, w=12)
        right = _img(h=5, w=3)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape[1] == 15


# ─── vertical_blend (extra) ───────────────────────────────────────────────────

class TestVerticalBlendExtra:
    def test_overlap_zero_concatenation(self):
        top = np.full((4, 5, 3), 100, dtype=np.uint8)
        bottom = np.full((6, 5, 3), 200, dtype=np.uint8)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.shape == (10, 5, 3)

    def test_output_dtype_uint8(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        result = vertical_blend(top, bottom, overlap=2)
        assert result.dtype == np.uint8

    def test_overlap_larger_than_bottom_raises(self):
        top = _img(h=10, w=5)
        bottom = _img(h=3, w=5)
        with pytest.raises(ValueError):
            vertical_blend(top, bottom, overlap=4)

    def test_grayscale_result_2d(self):
        top = _img(h=6, w=5, channels=1)
        bottom = _img(h=4, w=5, channels=1)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.ndim == 2

    def test_different_heights_no_overlap(self):
        top = _img(h=12, w=5)
        bottom = _img(h=3, w=5)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.shape[0] == 15


# ─── batch_blend (extra) ──────────────────────────────────────────────────────

class TestBatchBlendExtra:
    def test_single_pair_alpha_zero(self):
        src = _img(fill=200)
        dst = _img(fill=50)
        result = batch_blend([(src, dst)], alpha=0.0)
        np.testing.assert_array_equal(result[0], dst)

    def test_single_pair_alpha_one(self):
        src = _img(fill=200)
        dst = _img(fill=50)
        result = batch_blend([(src, dst)], alpha=1.0)
        np.testing.assert_array_equal(result[0], src)

    def test_five_pairs(self):
        pairs = [(_img(fill=100), _img(fill=200)) for _ in range(5)]
        result = batch_blend(pairs, alpha=0.5)
        assert len(result) == 5

    def test_all_results_uint8(self):
        pairs = [(_img(), _img()) for _ in range(3)]
        for r in batch_blend(pairs, alpha=0.3):
            assert r.dtype == np.uint8

    def test_grayscale_pairs(self):
        pairs = [(_img(channels=1, fill=100), _img(channels=1, fill=200))]
        result = batch_blend(pairs, alpha=0.5)
        assert result[0].ndim == 2
        assert result[0].dtype == np.uint8

    def test_shapes_preserved_per_pair(self):
        pairs = [(_img(h=4, w=6), _img(h=4, w=6)),
                 (_img(h=8, w=10), _img(h=8, w=10))]
        results = batch_blend(pairs, alpha=0.5)
        assert results[0].shape == (4, 6, 3)
        assert results[1].shape == (8, 10, 3)
