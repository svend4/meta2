"""
Tests for puzzle_reconstruction.preprocessing.multiscale_segmenter
"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.multiscale_segmenter import (
    MultiscaleConfig,
    MultiscaleSegmentationResult,
    MultiscaleSegmenter,
    segment_multiscale,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_gray(h=64, w=64, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _make_gradient(h=64, w=64):
    """Image with a clear dark foreground region on light background."""
    img = np.full((h, w), 220, dtype=np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 30
    return img


def _make_noisy(h=64, w=64):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _make_rgb(h=64, w=64):
    rng = np.random.default_rng(7)
    return rng.integers(50, 200, (h, w, 3), dtype=np.uint8)


# ─── Config tests ─────────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = MultiscaleConfig()
    assert cfg.scales == [1.0, 0.5, 0.25]
    assert cfg.vote_threshold == 0.5
    assert cfg.method == "otsu"
    assert cfg.min_area == 100
    assert cfg.smooth_final is True


def test_config_invalid_method():
    with pytest.raises(ValueError):
        MultiscaleConfig(method="grabcut")


def test_config_empty_scales():
    with pytest.raises(ValueError):
        MultiscaleConfig(scales=[])


def test_config_invalid_vote_threshold_low():
    with pytest.raises(ValueError):
        MultiscaleConfig(vote_threshold=-0.1)


def test_config_invalid_vote_threshold_high():
    with pytest.raises(ValueError):
        MultiscaleConfig(vote_threshold=1.1)


def test_config_invalid_min_area():
    with pytest.raises(ValueError):
        MultiscaleConfig(min_area=-1)


# ─── Result field tests ───────────────────────────────────────────────────────

def test_result_fields():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert hasattr(result, "mask")
    assert hasattr(result, "confidence_map")
    assert hasattr(result, "n_scales_used")
    assert hasattr(result, "scales")


def test_segment_returns_result_type():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_mask_is_bool_dtype():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert result.mask.dtype == bool


def test_mask_same_spatial_shape_as_input():
    img = _make_gradient(64, 80)
    result = segment_multiscale(img)
    assert result.mask.shape == (64, 80)


def test_confidence_map_values_in_01():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert result.confidence_map.min() >= 0.0
    assert result.confidence_map.max() <= 1.0


def test_n_scales_used_positive():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert result.n_scales_used > 0


def test_scales_list_matches_config():
    scales = [1.0, 0.5]
    cfg = MultiscaleConfig(scales=scales)
    img = _make_gradient()
    result = segment_multiscale(img, config=cfg)
    assert result.scales == scales


def test_uniform_white_image_returns_mask():
    img = np.full((64, 64), 255, dtype=np.uint8)
    cfg = MultiscaleConfig(min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    # Uniform image: mask is valid bool array (either all-True or all-False)
    assert result.mask.dtype == bool
    assert result.mask.shape == (64, 64)


def test_clear_foreground_nonempty_mask():
    img = _make_gradient(64, 64)
    cfg = MultiscaleConfig(min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.any()


def test_method_otsu_works():
    img = _make_gradient()
    cfg = MultiscaleConfig(method="otsu")
    result = segment_multiscale(img, config=cfg)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_method_triangle_works():
    img = _make_gradient()
    cfg = MultiscaleConfig(method="triangle")
    result = segment_multiscale(img, config=cfg)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_method_adaptive_works():
    img = _make_gradient()
    cfg = MultiscaleConfig(method="adaptive")
    result = segment_multiscale(img, config=cfg)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_grayscale_input_works():
    img = _make_gradient()
    assert img.ndim == 2
    result = segment_multiscale(img)
    assert result.mask.shape == img.shape


def test_rgb_input_works():
    img = _make_rgb()
    result = segment_multiscale(img)
    assert result.mask.shape == img.shape[:2]


def test_small_image_16x16():
    img = _make_gradient(h=16, w=16)
    cfg = MultiscaleConfig(scales=[1.0], min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.shape == (16, 16)


def test_single_scale():
    img = _make_gradient()
    cfg = MultiscaleConfig(scales=[1.0])
    result = segment_multiscale(img, config=cfg)
    assert result.n_scales_used == 1


def test_vote_threshold_zero_large_mask():
    img = _make_gradient()
    cfg = MultiscaleConfig(vote_threshold=0.0, min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    # At threshold 0, any pixel voted by at least 0% of scales is foreground
    # => nearly full mask expected
    assert result.mask.sum() >= img.size * 0.5


def test_vote_threshold_one_sparse_mask():
    img = _make_gradient()
    cfg = MultiscaleConfig(vote_threshold=1.0, min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    # Threshold=1.0 requires all scales to agree; may be empty or sparse
    assert result.mask.sum() <= img.size


def test_module_level_segment_multiscale():
    img = _make_gradient()
    result = segment_multiscale(img)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_segmenter_class_direct():
    img = _make_gradient()
    segmenter = MultiscaleSegmenter()
    result = segmenter.segment(img)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_confidence_map_shape_matches_image():
    img = _make_gradient(48, 64)
    result = segment_multiscale(img)
    assert result.confidence_map.shape == (48, 64)


def test_n_scales_used_matches_config():
    img = _make_gradient()
    cfg = MultiscaleConfig(scales=[1.0, 0.5, 0.25])
    result = segment_multiscale(img, config=cfg)
    assert result.n_scales_used == 3


def test_otsu_pure_numpy_threshold():
    img = _make_gradient()
    segmenter = MultiscaleSegmenter()
    mask = segmenter._threshold_otsu(img)
    assert mask.dtype == bool
    assert mask.shape == img.shape


def test_triangle_threshold_returns_bool_mask():
    img = _make_gradient()
    segmenter = MultiscaleSegmenter()
    mask = segmenter._threshold_triangle(img)
    assert mask.dtype == bool
    assert mask.shape == img.shape


def test_to_gray_rgb():
    img = _make_rgb()
    segmenter = MultiscaleSegmenter()
    gray = segmenter._to_gray(img)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_to_gray_gray():
    img = _make_gray()
    segmenter = MultiscaleSegmenter()
    gray = segmenter._to_gray(img)
    assert gray.ndim == 2


def test_resize_mask():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    segmenter = MultiscaleSegmenter()
    resized = segmenter._resize_mask(mask, (64, 64))
    assert resized.shape == (64, 64)
    assert resized.any()
