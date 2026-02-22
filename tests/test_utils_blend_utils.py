"""Tests for puzzle_reconstruction/utils/blend_utils.py"""
import pytest
import numpy as np

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


# ─── BlendConfig ──────────────────────────────────────────────────────────────

class TestBlendConfig:
    def test_defaults(self):
        cfg = BlendConfig()
        assert cfg.feather_px == 8
        assert cfg.gamma == 1.0
        assert cfg.clip_output is True

    def test_negative_feather_raises(self):
        with pytest.raises(ValueError, match="feather_px"):
            BlendConfig(feather_px=-1)

    def test_zero_feather_ok(self):
        cfg = BlendConfig(feather_px=0)
        assert cfg.feather_px == 0

    def test_positive_feather_ok(self):
        cfg = BlendConfig(feather_px=16)
        assert cfg.feather_px == 16

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            BlendConfig(gamma=0.0)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            BlendConfig(gamma=-1.0)

    def test_positive_gamma_ok(self):
        cfg = BlendConfig(gamma=2.5)
        assert cfg.gamma == 2.5

    def test_clip_output_false(self):
        cfg = BlendConfig(clip_output=False)
        assert cfg.clip_output is False


# ─── alpha_blend ──────────────────────────────────────────────────────────────

class TestAlphaBlend:
    def test_alpha_one_returns_src(self):
        src = _img(fill=200)
        dst = _img(fill=0)
        result = alpha_blend(src, dst, alpha=1.0)
        np.testing.assert_array_equal(result, src)

    def test_alpha_zero_returns_dst(self):
        src = _img(fill=200)
        dst = _img(fill=50)
        result = alpha_blend(src, dst, alpha=0.0)
        np.testing.assert_array_equal(result, dst)

    def test_alpha_half_midpoint(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 0, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert np.all(result == 100)

    def test_output_dtype_uint8(self):
        src = _img(fill=100)
        dst = _img(fill=50)
        result = alpha_blend(src, dst, alpha=0.5)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        src = _img(h=8, w=12)
        dst = _img(h=8, w=12)
        result = alpha_blend(src, dst, alpha=0.3)
        assert result.shape == src.shape

    def test_alpha_negative_raises(self):
        src = _img()
        dst = _img()
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=-0.1)

    def test_alpha_gt_one_raises(self):
        src = _img()
        dst = _img()
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=1.1)

    def test_shape_mismatch_raises(self):
        src = _img(h=4, w=4)
        dst = _img(h=8, w=8)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=0.5)

    def test_grayscale_2d(self):
        src = _img(fill=200, channels=1)
        dst = _img(fill=0, channels=1)
        result = alpha_blend(src, dst, alpha=0.5)
        assert result.shape == src.shape
        assert result.dtype == np.uint8

    def test_output_clipped_to_uint8_range(self):
        src = np.full((4, 4, 3), 255, dtype=np.uint8)
        dst = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = alpha_blend(src, dst, alpha=0.5)
        assert np.all(result <= 255)
        assert np.all(result >= 0)

    def test_alpha_boundary_zero(self):
        src = _img(fill=100)
        dst = _img(fill=200)
        result = alpha_blend(src, dst, alpha=0.0)
        assert np.all(result == 200)

    def test_alpha_boundary_one(self):
        src = _img(fill=100)
        dst = _img(fill=200)
        result = alpha_blend(src, dst, alpha=1.0)
        assert np.all(result == 100)

    def test_ndim_1_raises(self):
        src = np.array([100, 200, 150], dtype=np.uint8)
        dst = np.array([50, 100, 75], dtype=np.uint8)
        with pytest.raises(ValueError):
            alpha_blend(src, dst, alpha=0.5)


# ─── weighted_blend ───────────────────────────────────────────────────────────

class TestWeightedBlend:
    def test_single_image(self):
        im = _img(fill=150)
        result = weighted_blend([im])
        np.testing.assert_array_equal(result, im)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            weighted_blend([])

    def test_wrong_weights_length_raises(self):
        im = _img()
        with pytest.raises(ValueError):
            weighted_blend([im, im], weights=[1.0])

    def test_equal_weights_is_mean(self):
        a = np.full((4, 4, 3), 100, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = weighted_blend([a, b])
        assert np.all(result == 150)

    def test_custom_weights_normalized(self):
        a = np.full((4, 4, 3), 0, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        # weights [1, 3] → b gets 0.75 weight → 0*0.25 + 200*0.75 = 150
        result = weighted_blend([a, b], weights=[1.0, 3.0])
        assert np.all(result == 150)

    def test_all_zero_weights_fallback_equal(self):
        a = np.full((4, 4, 3), 0, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[0.0, 0.0])
        # Should fall back to equal blend → mean = 100
        assert np.all(result == 100)

    def test_output_uint8(self):
        a = _img(fill=50)
        b = _img(fill=150)
        result = weighted_blend([a, b])
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        a = _img(h=6, w=8)
        b = _img(h=6, w=8)
        result = weighted_blend([a, b])
        assert result.shape == a.shape

    def test_shape_mismatch_raises(self):
        a = _img(h=4, w=4)
        b = _img(h=8, w=8)
        with pytest.raises(ValueError):
            weighted_blend([a, b])

    def test_three_equal_images(self):
        a = np.full((4, 4, 3), 60, dtype=np.uint8)
        b = np.full((4, 4, 3), 60, dtype=np.uint8)
        c = np.full((4, 4, 3), 60, dtype=np.uint8)
        result = weighted_blend([a, b, c])
        assert np.all(result == 60)

    def test_weight_sum_one(self):
        """Weights summing to 1 act as direct blend coefficients."""
        a = np.full((4, 4, 3), 0, dtype=np.uint8)
        b = np.full((4, 4, 3), 100, dtype=np.uint8)
        result = weighted_blend([a, b], weights=[0.5, 0.5])
        assert np.all(result == 50)


# ─── feather_mask ─────────────────────────────────────────────────────────────

class TestFeatherMask:
    def test_shape(self):
        mask = feather_mask(10, 20)
        assert mask.shape == (10, 20)

    def test_dtype_float32(self):
        mask = feather_mask(10, 20)
        assert mask.dtype == np.float32

    def test_values_in_range(self):
        mask = feather_mask(20, 30, feather_px=4)
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_zero_feather_all_ones(self):
        mask = feather_mask(10, 10, feather_px=0)
        np.testing.assert_array_equal(mask, np.ones((10, 10), dtype=np.float32))

    def test_center_is_one(self):
        h, w, fp = 20, 20, 4
        mask = feather_mask(h, w, feather_px=fp)
        cy, cx = h // 2, w // 2
        assert mask[cy, cx] == pytest.approx(1.0)

    def test_corners_less_than_center_with_feather(self):
        h, w, fp = 20, 20, 4
        mask = feather_mask(h, w, feather_px=fp)
        assert mask[0, 0] < mask[h // 2, w // 2]

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            feather_mask(0, 10)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            feather_mask(10, 0)

    def test_negative_feather_raises(self):
        with pytest.raises(ValueError):
            feather_mask(10, 10, feather_px=-1)

    def test_large_feather_clamped(self):
        """feather_px larger than half dimension should not crash."""
        mask = feather_mask(10, 10, feather_px=100)
        assert mask.shape == (10, 10)
        assert np.all(mask >= 0.0)

    def test_1x1_mask(self):
        mask = feather_mask(1, 1, feather_px=0)
        assert mask.shape == (1, 1)
        assert mask[0, 0] == pytest.approx(1.0)

    def test_rectangular_shape(self):
        mask = feather_mask(5, 15, feather_px=2)
        assert mask.shape == (5, 15)


# ─── paste_with_mask ──────────────────────────────────────────────────────────

class TestPasteWithMask:
    def test_basic_paste_replaces(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert np.all(result[:4, :4] == 200)

    def test_returns_copy(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert result is not canvas

    def test_output_dtype_uint8(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert result.dtype == np.uint8

    def test_zero_mask_unchanged_region(self):
        canvas = np.full((10, 10, 3), 50, dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        # mask=0 → canvas stays unchanged
        np.testing.assert_array_equal(result[:4, :4], canvas[:4, :4])

    def test_out_of_bounds_paste_no_crash(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=8, x=8)
        assert result.shape == canvas.shape

    def test_2d_canvas(self):
        canvas = np.zeros((10, 10), dtype=np.uint8)
        patch = np.full((4, 4), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=0, x=0)
        assert result.shape == (10, 10)

    def test_paste_at_offset(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        patch = np.full((2, 2, 3), 255, dtype=np.uint8)
        mask = np.ones((2, 2), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=5, x=5)
        assert np.all(result[5:7, 5:7] == 255)

    def test_fully_outside_returns_copy(self):
        canvas = np.zeros((5, 5, 3), dtype=np.uint8)
        patch = np.full((3, 3, 3), 200, dtype=np.uint8)
        mask = np.ones((3, 3), dtype=np.float32)
        result = paste_with_mask(canvas, patch, mask, y=10, x=10)
        np.testing.assert_array_equal(result, canvas)


# ─── horizontal_blend ─────────────────────────────────────────────────────────

class TestHorizontalBlend:
    def test_output_width_no_overlap(self):
        left = _img(h=5, w=8)
        right = _img(h=5, w=6)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape[1] == 14

    def test_output_width_with_overlap(self):
        left = _img(h=5, w=8)
        right = _img(h=5, w=6)
        result = horizontal_blend(left, right, overlap=3)
        assert result.shape[1] == 8 + 6 - 3

    def test_height_preserved(self):
        left = _img(h=5, w=8)
        right = _img(h=5, w=6)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape[0] == 5

    def test_height_mismatch_raises(self):
        left = _img(h=5, w=8)
        right = _img(h=7, w=6)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=0)

    def test_negative_overlap_raises(self):
        left = _img(h=5, w=8)
        right = _img(h=5, w=6)
        with pytest.raises(ValueError):
            horizontal_blend(left, right, overlap=-1)

    def test_grayscale(self):
        left = _img(h=5, w=8, channels=1)
        right = _img(h=5, w=6, channels=1)
        result = horizontal_blend(left, right, overlap=0)
        assert result.shape == (5, 14)

    def test_output_dtype_uint8(self):
        left = _img(h=5, w=8)
        right = _img(h=5, w=6)
        result = horizontal_blend(left, right, overlap=0)
        assert result.dtype == np.uint8

    def test_left_portion_preserved_no_overlap(self):
        """With overlap=0, left portion should equal left image."""
        left = np.full((4, 6, 3), 100, dtype=np.uint8)
        right = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = horizontal_blend(left, right, overlap=0)
        np.testing.assert_array_equal(result[:, :6], left)

    def test_right_portion_preserved_no_overlap(self):
        """With overlap=0, right portion should equal right image."""
        left = np.full((4, 6, 3), 100, dtype=np.uint8)
        right = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = horizontal_blend(left, right, overlap=0)
        np.testing.assert_array_equal(result[:, 6:], right)

    def test_with_config(self):
        cfg = BlendConfig(feather_px=0)
        left = _img(h=4, w=8)
        right = _img(h=4, w=6)
        result = horizontal_blend(left, right, overlap=2, cfg=cfg)
        assert result.shape[1] == 12


# ─── vertical_blend ───────────────────────────────────────────────────────────

class TestVerticalBlend:
    def test_output_height_no_overlap(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.shape[0] == 10

    def test_output_height_with_overlap(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        result = vertical_blend(top, bottom, overlap=2)
        assert result.shape[0] == 6 + 4 - 2

    def test_width_preserved(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.shape[1] == 5

    def test_width_mismatch_raises(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=7)
        with pytest.raises(ValueError):
            vertical_blend(top, bottom, overlap=0)

    def test_negative_overlap_raises(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        with pytest.raises(ValueError):
            vertical_blend(top, bottom, overlap=-1)

    def test_output_dtype_uint8(self):
        top = _img(h=6, w=5)
        bottom = _img(h=4, w=5)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.dtype == np.uint8

    def test_grayscale(self):
        top = _img(h=6, w=5, channels=1)
        bottom = _img(h=4, w=5, channels=1)
        result = vertical_blend(top, bottom, overlap=0)
        assert result.shape == (10, 5)

    def test_top_portion_preserved_no_overlap(self):
        top = np.full((4, 5, 3), 100, dtype=np.uint8)
        bottom = np.full((4, 5, 3), 200, dtype=np.uint8)
        result = vertical_blend(top, bottom, overlap=0)
        np.testing.assert_array_equal(result[:4], top)

    def test_bottom_portion_preserved_no_overlap(self):
        top = np.full((4, 5, 3), 100, dtype=np.uint8)
        bottom = np.full((4, 5, 3), 200, dtype=np.uint8)
        result = vertical_blend(top, bottom, overlap=0)
        np.testing.assert_array_equal(result[4:], bottom)


# ─── batch_blend ──────────────────────────────────────────────────────────────

class TestBatchBlend:
    def test_empty_returns_empty(self):
        result = batch_blend([])
        assert result == []

    def test_length_preserved(self):
        pairs = [(_img(fill=100), _img(fill=200)) for _ in range(3)]
        result = batch_blend(pairs, alpha=0.5)
        assert len(result) == 3

    def test_each_result_uint8(self):
        pairs = [(_img(fill=100), _img(fill=200))]
        result = batch_blend(pairs, alpha=0.5)
        assert result[0].dtype == np.uint8

    def test_applies_alpha(self):
        src = np.full((4, 4, 3), 200, dtype=np.uint8)
        dst = np.full((4, 4, 3), 0, dtype=np.uint8)
        result = batch_blend([(src, dst)], alpha=0.5)
        assert np.all(result[0] == 100)

    def test_alpha_one_all_src(self):
        src = np.full((4, 4, 3), 255, dtype=np.uint8)
        dst = np.full((4, 4, 3), 0, dtype=np.uint8)
        result = batch_blend([(src, dst)], alpha=1.0)
        np.testing.assert_array_equal(result[0], src)

    def test_multiple_pairs_independent(self):
        pairs = [
            (np.full((4, 4, 3), 100, dtype=np.uint8),
             np.full((4, 4, 3), 200, dtype=np.uint8)),
            (np.full((4, 4, 3), 0, dtype=np.uint8),
             np.full((4, 4, 3), 50, dtype=np.uint8)),
        ]
        result = batch_blend(pairs, alpha=0.5)
        assert len(result) == 2
        assert np.all(result[0] == 150)
        assert np.all(result[1] == 25)
