"""
Tests for puzzle_reconstruction.preprocessing.tear_enhancer
"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.tear_enhancer import (
    TearEnhancerConfig,
    TearEnhancerResult,
    TearEdgeEnhancer,
    enhance_torn_edge,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_gray(h=60, w=60):
    rng = np.random.default_rng(0)
    return (rng.integers(50, 200, (h, w))).astype(np.uint8)


def _make_rgb(h=60, w=60):
    rng = np.random.default_rng(1)
    return (rng.integers(50, 200, (h, w, 3))).astype(np.uint8)


def _make_contour(n=20, h=60, w=60):
    """A simple rectangular contour with n points along the perimeter."""
    pts = []
    side = n // 4
    for i in range(side):
        pts.append([int(w * 0.2 + i * (w * 0.6 / max(side - 1, 1))), int(h * 0.2)])
    for i in range(side):
        pts.append([int(w * 0.8), int(h * 0.2 + i * (h * 0.6 / max(side - 1, 1)))])
    for i in range(side):
        pts.append([int(w * 0.8 - i * (w * 0.6 / max(side - 1, 1))), int(h * 0.8)])
    for i in range(side):
        pts.append([int(w * 0.2), int(h * 0.8 - i * (h * 0.6 / max(side - 1, 1)))])
    return np.array(pts, dtype=np.float32)


# ─── Config tests ─────────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = TearEnhancerConfig()
    assert cfg.supersample_factor == 2
    assert cfg.denoise_radius == 3
    assert cfg.contrast_alpha == 1.5
    assert cfg.method == "gaussian"
    assert cfg.edge_band_width == 5


def test_config_invalid_supersample_factor():
    with pytest.raises(ValueError):
        TearEnhancerConfig(supersample_factor=0)


def test_config_invalid_supersample_factor_high():
    with pytest.raises(ValueError):
        TearEnhancerConfig(supersample_factor=5)


def test_config_invalid_denoise_radius():
    with pytest.raises(ValueError):
        TearEnhancerConfig(denoise_radius=-1)


def test_config_invalid_contrast_alpha_low():
    with pytest.raises(ValueError):
        TearEnhancerConfig(contrast_alpha=0.5)


def test_config_invalid_contrast_alpha_high():
    with pytest.raises(ValueError):
        TearEnhancerConfig(contrast_alpha=3.5)


def test_config_invalid_method():
    with pytest.raises(ValueError):
        TearEnhancerConfig(method="unknown")


def test_config_invalid_edge_band_width():
    with pytest.raises(ValueError):
        TearEnhancerConfig(edge_band_width=0)


# ─── Result field tests ───────────────────────────────────────────────────────

def test_result_fields():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert hasattr(result, "enhanced_image")
    assert hasattr(result, "enhanced_contour")
    assert hasattr(result, "sharpness_before")
    assert hasattr(result, "sharpness_after")


def test_enhance_returns_result_type():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert isinstance(result, TearEnhancerResult)


def test_enhanced_image_same_shape_gray():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_enhanced_image_same_shape_rgb():
    img = _make_rgb()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_sharpness_before_is_float():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert isinstance(result.sharpness_before, float)


def test_sharpness_after_is_float():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert isinstance(result.sharpness_after, float)


def test_sharpness_after_nonnegative():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert result.sharpness_after >= 0.0


def test_sharpness_before_nonnegative():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert result.sharpness_before >= 0.0


def test_method_none_returns_nearly_unchanged():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(method="none", contrast_alpha=1.0)
    result = enhance_torn_edge(img, contour, config=cfg)
    # method="none" with alpha=1.0 should leave image essentially unchanged
    diff = np.abs(result.enhanced_image.astype(float) - img.astype(float)).max()
    assert diff < 1e-3


def test_method_gaussian_works():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(method="gaussian")
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.shape == img.shape


def test_method_bilateral_works():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(method="bilateral")
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.shape == img.shape


def test_supersample_factor_1():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(supersample_factor=1)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_contour.shape[1] == 2


def test_supersample_factor_4():
    img = _make_gray()
    contour = _make_contour(n=8)
    cfg = TearEnhancerConfig(supersample_factor=4)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_contour.shape[0] >= len(contour)
    assert result.enhanced_contour.shape[1] == 2


def test_works_with_grayscale_image():
    img = _make_gray(30, 30)
    contour = _make_contour(n=12, h=30, w=30)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.ndim == 2


def test_works_with_rgb_image():
    img = _make_rgb(30, 30)
    contour = _make_contour(n=12, h=30, w=30)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.ndim == 3
    assert result.enhanced_image.shape[2] == 3


def test_small_image():
    img = _make_gray(20, 20)
    contour = _make_contour(n=8, h=20, w=20)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_large_image():
    img = _make_gray(200, 200)
    contour = _make_contour(n=40, h=200, w=200)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_contour_few_points():
    img = _make_gray()
    contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.shape[1] == 2


def test_contour_many_points():
    img = _make_gray()
    contour = _make_contour(n=100)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.shape[1] == 2


def test_module_level_enhance_torn_edge():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert isinstance(result, TearEnhancerResult)


def test_deterministic():
    img = _make_gray()
    contour = _make_contour()
    result1 = enhance_torn_edge(img, contour)
    result2 = enhance_torn_edge(img, contour)
    np.testing.assert_array_equal(result1.enhanced_image, result2.enhanced_image)
    assert result1.sharpness_before == result2.sharpness_before


def test_contrast_alpha_1_no_change_in_band():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(contrast_alpha=1.0, method="none")
    result = enhance_torn_edge(img, contour, config=cfg)
    diff = np.abs(result.enhanced_image.astype(float) - img.astype(float)).max()
    # alpha=1.0 means no contrast change; method=none means no denoise
    assert diff < 1e-3


def test_denoise_radius_0_works():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(denoise_radius=0)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.shape == img.shape


def test_denoise_radius_1_works():
    img = _make_gray()
    contour = _make_contour()
    cfg = TearEnhancerConfig(denoise_radius=1)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.shape == img.shape


def test_enhanced_contour_shape():
    img = _make_gray()
    contour = _make_contour(n=20)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.ndim == 2
    assert result.enhanced_contour.shape[1] == 2


def test_enhanced_contour_dtype():
    img = _make_gray()
    contour = _make_contour()
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.dtype in (np.float32, np.float64)


def test_enhancer_class_direct_usage():
    img = _make_gray()
    contour = _make_contour()
    enhancer = TearEdgeEnhancer()
    result = enhancer.enhance(img, contour)
    assert isinstance(result, TearEnhancerResult)


def test_estimate_sharpness_returns_float():
    img = _make_gray()
    enhancer = TearEdgeEnhancer()
    val = enhancer._estimate_sharpness(img)
    assert isinstance(val, float)
    assert val >= 0.0


def test_estimate_sharpness_flat_image():
    img = np.full((40, 40), 128, dtype=np.uint8)
    enhancer = TearEdgeEnhancer()
    val = enhancer._estimate_sharpness(img)
    # Flat image should have zero or near-zero sharpness
    assert val < 1.0
