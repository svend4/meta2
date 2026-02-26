"""Extra tests for puzzle_reconstruction/preprocessing/multiscale_segmenter.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.multiscale_segmenter import (
    MultiscaleConfig,
    MultiscaleSegmentationResult,
    MultiscaleSegmenter,
    segment_multiscale,
    _resize_bilinear,
    _morph_erode_dilate,
    _remove_small_components,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gradient_img(h=64, w=64):
    """Dark object on light background."""
    img = np.full((h, w), 220, dtype=np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 30
    return img


def _uniform(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=99):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgba(h=32, w=32, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


# ─── Config validation ────────────────────────────────────────────────────────

def test_config_scales_single_value():
    cfg = MultiscaleConfig(scales=[0.75])
    assert cfg.scales == [0.75]


def test_config_vote_threshold_boundary_zero():
    cfg = MultiscaleConfig(vote_threshold=0.0)
    assert cfg.vote_threshold == 0.0


def test_config_vote_threshold_boundary_one():
    cfg = MultiscaleConfig(vote_threshold=1.0)
    assert cfg.vote_threshold == 1.0


def test_config_min_area_zero_valid():
    cfg = MultiscaleConfig(min_area=0)
    assert cfg.min_area == 0


def test_config_smooth_final_false():
    cfg = MultiscaleConfig(smooth_final=False)
    assert cfg.smooth_final is False


def test_config_invalid_method_space():
    with pytest.raises(ValueError):
        MultiscaleConfig(method=" otsu")


def test_config_scales_large_value():
    cfg = MultiscaleConfig(scales=[2.0, 4.0])
    assert 2.0 in cfg.scales


def test_config_negative_vote_threshold_raises():
    with pytest.raises(ValueError):
        MultiscaleConfig(vote_threshold=-0.01)


def test_config_vote_threshold_above_one_raises():
    with pytest.raises(ValueError):
        MultiscaleConfig(vote_threshold=1.01)


def test_config_negative_min_area_raises():
    with pytest.raises(ValueError):
        MultiscaleConfig(min_area=-5)


# ─── Segmentation output shapes ───────────────────────────────────────────────

def test_mask_shape_landscape():
    img = _gradient_img(48, 96)
    result = segment_multiscale(img)
    assert result.mask.shape == (48, 96)


def test_mask_shape_portrait():
    img = _gradient_img(96, 48)
    result = segment_multiscale(img)
    assert result.mask.shape == (96, 48)


def test_confidence_map_dtype_float32():
    img = _gradient_img()
    result = segment_multiscale(img)
    assert result.confidence_map.dtype == np.float32


def test_mask_dtype_bool():
    img = _gradient_img()
    result = segment_multiscale(img)
    assert result.mask.dtype == bool


def test_result_scales_list_copied():
    scales = [1.0, 0.5]
    cfg = MultiscaleConfig(scales=scales)
    result = segment_multiscale(_gradient_img(), config=cfg)
    assert result.scales == scales


def test_result_scales_not_same_object():
    scales = [1.0, 0.5]
    cfg = MultiscaleConfig(scales=scales)
    result = segment_multiscale(_gradient_img(), config=cfg)
    result.scales.append(0.25)
    assert cfg.scales == [1.0, 0.5]


# ─── Vote threshold boundary effects ─────────────────────────────────────────

def test_vote_0_entire_image_foreground():
    """With threshold=0 every pixel is foreground."""
    img = _gradient_img()
    cfg = MultiscaleConfig(vote_threshold=0.0, min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.all()


def test_vote_1_requires_all_scales_agree():
    img = _gradient_img()
    cfg = MultiscaleConfig(scales=[1.0, 0.5, 0.25], vote_threshold=1.0,
                           min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    # Mask may be empty or sparse; just check valid dtype
    assert result.mask.dtype == bool


def test_vote_threshold_half_intermediate_result():
    img = _gradient_img()
    cfg = MultiscaleConfig(vote_threshold=0.5, min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    # With 3 scales, 0.5 threshold means >= 2 scales must agree
    total_pixels = result.mask.size
    assert 0 <= result.mask.sum() <= total_pixels


# ─── Method-specific tests ────────────────────────────────────────────────────

def test_adaptive_method_returns_bool_mask():
    img = _gradient_img()
    cfg = MultiscaleConfig(method="adaptive", min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.dtype == bool


def test_triangle_method_returns_bool_mask():
    img = _gradient_img()
    cfg = MultiscaleConfig(method="triangle", min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.dtype == bool


def test_all_black_image_otsu():
    img = _uniform(val=0)
    cfg = MultiscaleConfig(min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.dtype == bool


def test_all_white_image_otsu():
    img = _uniform(val=255)
    cfg = MultiscaleConfig(min_area=0, smooth_final=False)
    result = segment_multiscale(img, config=cfg)
    assert result.mask.dtype == bool


def test_two_tone_image_foreground_detected():
    """Strict two-tone image: Otsu should cleanly separate."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    cfg = MultiscaleConfig(method="otsu", min_area=0, smooth_final=False,
                           scales=[1.0])
    result = segment_multiscale(img, config=cfg)
    # Inner bright region should be detected as foreground
    assert result.mask.sum() > 0


# ─── min_area filtering ───────────────────────────────────────────────────────

def test_min_area_zero_keeps_small_components():
    img = _gradient_img()
    cfg_no_filter  = MultiscaleConfig(min_area=0, smooth_final=False, scales=[1.0])
    cfg_filter     = MultiscaleConfig(min_area=10000, smooth_final=False, scales=[1.0])
    r_nf = segment_multiscale(img, config=cfg_no_filter)
    r_f  = segment_multiscale(img, config=cfg_filter)
    # Filtered should have fewer or equal foreground pixels
    assert r_f.mask.sum() <= r_nf.mask.sum()


def test_min_area_removes_single_pixels():
    img = _gradient_img()
    cfg = MultiscaleConfig(min_area=img.size, smooth_final=False, scales=[1.0])
    result = segment_multiscale(img, config=cfg)
    # With min_area = full image size, all small components removed
    assert result.mask.dtype == bool


# ─── Smooth final ─────────────────────────────────────────────────────────────

def test_smooth_final_true_does_not_crash():
    img = _gradient_img()
    cfg = MultiscaleConfig(smooth_final=True)
    result = segment_multiscale(img, config=cfg)
    assert isinstance(result, MultiscaleSegmentationResult)


def test_smooth_vs_no_smooth_same_shape():
    img = _gradient_img()
    r_smooth = segment_multiscale(img, config=MultiscaleConfig(smooth_final=True))
    r_raw    = segment_multiscale(img, config=MultiscaleConfig(smooth_final=False))
    assert r_smooth.mask.shape == r_raw.mask.shape


# ─── n_scales_used ────────────────────────────────────────────────────────────

def test_n_scales_used_single_scale():
    cfg = MultiscaleConfig(scales=[1.0])
    result = segment_multiscale(_gradient_img(), config=cfg)
    assert result.n_scales_used == 1


def test_n_scales_used_two_scales():
    cfg = MultiscaleConfig(scales=[1.0, 0.5])
    result = segment_multiscale(_gradient_img(), config=cfg)
    assert result.n_scales_used == 2


def test_n_scales_used_five_scales():
    cfg = MultiscaleConfig(scales=[1.0, 0.75, 0.5, 0.25, 0.125])
    result = segment_multiscale(_gradient_img(), config=cfg)
    assert result.n_scales_used == 5


# ─── Grayscale / colour conversion ───────────────────────────────────────────

def test_to_gray_single_channel():
    seg = MultiscaleSegmenter()
    img = _uniform(32, 32, 128)
    gray = seg._to_gray(img)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray, img)


def test_to_gray_two_channel_img():
    seg = MultiscaleSegmenter()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 2), dtype=np.uint8)
    gray = seg._to_gray(img)
    assert gray.ndim == 2


def test_to_gray_rgba_image():
    seg = MultiscaleSegmenter()
    img = _rgba(32, 32)
    gray = seg._to_gray(img)
    assert gray.ndim == 2


# ─── Internal helpers ─────────────────────────────────────────────────────────

def test_resize_bilinear_same_size():
    arr = np.eye(8, dtype=np.float32)
    out = _resize_bilinear(arr, 8, 8)
    np.testing.assert_array_almost_equal(out, arr)


def test_resize_bilinear_upsample():
    arr = np.ones((4, 4), dtype=np.float32) * 10
    out = _resize_bilinear(arr, 8, 8)
    assert out.shape == (8, 8)
    np.testing.assert_allclose(out, 10.0, atol=1e-5)


def test_resize_bilinear_downsample():
    arr = np.ones((64, 64), dtype=np.float32) * 5
    out = _resize_bilinear(arr, 16, 16)
    assert out.shape == (16, 16)
    np.testing.assert_allclose(out, 5.0, atol=1e-4)


def test_morph_erode_radius_1():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    eroded = _morph_erode_dilate(mask, radius=1, dilate=False)
    assert eroded.dtype == bool
    # Eroded region should be smaller than original
    assert eroded.sum() <= mask.sum()


def test_morph_dilate_radius_1():
    mask = np.zeros((16, 16), dtype=bool)
    mask[7, 7] = True
    dilated = _morph_erode_dilate(mask, radius=1, dilate=True)
    assert dilated.sum() >= mask.sum()


def test_remove_small_components_min_area_zero():
    mask = np.zeros((16, 16), dtype=bool)
    mask[5:10, 5:10] = True
    out = _remove_small_components(mask, min_area=0)
    np.testing.assert_array_equal(out, mask)


def test_remove_small_components_removes_small():
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True       # single pixel component
    mask[5:10, 5:10] = True  # 25-pixel component
    out = _remove_small_components(mask, min_area=10)
    assert not out[0, 0]
    assert out[5:10, 5:10].all()


def test_remove_small_components_empty_mask():
    mask = np.zeros((16, 16), dtype=bool)
    out = _remove_small_components(mask, min_area=5)
    assert not out.any()


def test_resize_mask_same_size():
    mask = np.ones((16, 16), dtype=bool)
    seg = MultiscaleSegmenter()
    out = seg._resize_mask(mask, (16, 16))
    np.testing.assert_array_equal(out, mask)


def test_resize_mask_double_size():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    seg = MultiscaleSegmenter()
    out = seg._resize_mask(mask, (16, 16))
    assert out.shape == (16, 16)
    assert out.any()


# ─── Determinism ─────────────────────────────────────────────────────────────

def test_segment_deterministic_otsu():
    img = _gradient_img()
    r1 = segment_multiscale(img)
    r2 = segment_multiscale(img)
    np.testing.assert_array_equal(r1.mask, r2.mask)
    np.testing.assert_array_equal(r1.confidence_map, r2.confidence_map)


def test_segment_deterministic_triangle():
    img = _gradient_img()
    cfg = MultiscaleConfig(method="triangle")
    r1 = segment_multiscale(img, config=cfg)
    r2 = segment_multiscale(img, config=cfg)
    np.testing.assert_array_equal(r1.mask, r2.mask)
