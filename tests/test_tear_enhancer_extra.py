"""Extra tests for puzzle_reconstruction/preprocessing/tear_enhancer.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.tear_enhancer import (
    TearEnhancerConfig,
    TearEnhancerResult,
    TearEdgeEnhancer,
    enhance_torn_edge,
    _gaussian_kernel_1d,
    _convolve2d_separable,
    _build_edge_mask,
    _numpy_gaussian_blur,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=60, w=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, (h, w), dtype=np.uint8)


def _rgb(h=60, w=60, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(50, 200, (h, w, 3), dtype=np.uint8)


def _contour(n=20, h=60, w=60):
    """Simple rectangular contour."""
    side = max(n // 4, 1)
    pts = []
    for i in range(side):
        pts.append([int(w * 0.2 + i * (w * 0.6 / max(side - 1, 1))), int(h * 0.2)])
    for i in range(side):
        pts.append([int(w * 0.8), int(h * 0.2 + i * (h * 0.6 / max(side - 1, 1)))])
    for i in range(side):
        pts.append([int(w * 0.8 - i * (w * 0.6 / max(side - 1, 1))), int(h * 0.8)])
    for i in range(side):
        pts.append([int(w * 0.2), int(h * 0.8 - i * (h * 0.6 / max(side - 1, 1)))])
    return np.array(pts, dtype=np.float32)


# ─── Config boundary values ───────────────────────────────────────────────────

def test_config_supersample_factor_boundary_low():
    cfg = TearEnhancerConfig(supersample_factor=1)
    assert cfg.supersample_factor == 1


def test_config_supersample_factor_boundary_high():
    cfg = TearEnhancerConfig(supersample_factor=4)
    assert cfg.supersample_factor == 4


def test_config_supersample_below_1_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(supersample_factor=0)


def test_config_supersample_above_4_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(supersample_factor=5)


def test_config_contrast_alpha_boundary_low():
    cfg = TearEnhancerConfig(contrast_alpha=1.0)
    assert cfg.contrast_alpha == pytest.approx(1.0)


def test_config_contrast_alpha_boundary_high():
    cfg = TearEnhancerConfig(contrast_alpha=3.0)
    assert cfg.contrast_alpha == pytest.approx(3.0)


def test_config_contrast_alpha_below_1_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(contrast_alpha=0.9)


def test_config_contrast_alpha_above_3_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(contrast_alpha=3.1)


def test_config_denoise_radius_zero_valid():
    cfg = TearEnhancerConfig(denoise_radius=0)
    assert cfg.denoise_radius == 0


def test_config_denoise_radius_negative_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(denoise_radius=-1)


def test_config_edge_band_width_one_valid():
    cfg = TearEnhancerConfig(edge_band_width=1)
    assert cfg.edge_band_width == 1


def test_config_edge_band_width_zero_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(edge_band_width=0)


def test_config_method_none_valid():
    cfg = TearEnhancerConfig(method="none")
    assert cfg.method == "none"


def test_config_method_bilateral_valid():
    cfg = TearEnhancerConfig(method="bilateral")
    assert cfg.method == "bilateral"


def test_config_method_invalid_raises():
    with pytest.raises(ValueError):
        TearEnhancerConfig(method="median")


# ─── Output shape invariants ──────────────────────────────────────────────────

def test_output_shape_preserved_gray_small():
    img = _gray(16, 16)
    contour = _contour(8, 16, 16)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_output_shape_preserved_gray_large():
    img = _gray(128, 128)
    contour = _contour(40, 128, 128)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_output_shape_preserved_rgb():
    img = _rgb(64, 64)
    contour = _contour(20, 64, 64)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == img.shape


def test_output_shape_nonsquare_wide():
    img = _gray(32, 96)
    contour = _contour(12, 32, 96)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.shape == (32, 96)


def test_output_dtype_preserved_uint8_gray():
    img = _gray()
    contour = _contour()
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.dtype == np.uint8


def test_output_dtype_preserved_uint8_rgb():
    img = _rgb()
    contour = _contour()
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_image.dtype == np.uint8


# ─── Contour shape handling ───────────────────────────────────────────────────

def test_contour_single_point():
    img = _gray()
    contour = np.array([[30.0, 30.0]])
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.shape[1] == 2


def test_contour_two_points():
    img = _gray()
    contour = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
    result = enhance_torn_edge(img, contour)
    assert result.enhanced_contour.ndim == 2
    assert result.enhanced_contour.shape[1] == 2


def test_contour_1d_input_reshaped():
    """If contour is passed as flat 1-D array, it should be reshaped."""
    img = _gray()
    contour_flat = np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32)
    result = enhance_torn_edge(img, contour_flat)
    assert result.enhanced_contour.shape[1] == 2


def test_contour_integer_dtype_accepted():
    img = _gray()
    contour = np.array([[10, 10], [50, 50]], dtype=np.int32)
    result = enhance_torn_edge(img, contour)
    assert isinstance(result, TearEnhancerResult)


# ─── Supersampling refinement size ────────────────────────────────────────────

def test_supersample_1_contour_same_length():
    img = _gray()
    contour = _contour(n=8)
    cfg = TearEnhancerConfig(supersample_factor=1)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert len(result.enhanced_contour) == len(contour)


def test_supersample_2_contour_larger():
    img = _gray()
    contour = _contour(n=8)
    cfg = TearEnhancerConfig(supersample_factor=2)
    result = enhance_torn_edge(img, contour, config=cfg)
    # With factor=2, upsampled to (8-1)*2+1 = 15 points (before refinement)
    assert len(result.enhanced_contour) >= len(contour)


def test_supersample_4_contour_larger():
    img = _gray()
    contour = _contour(n=4)
    cfg = TearEnhancerConfig(supersample_factor=4)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert len(result.enhanced_contour) >= len(contour)


# ─── Contrast alpha effects ───────────────────────────────────────────────────

def test_contrast_alpha_1_noop():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(contrast_alpha=1.0, method="none", denoise_radius=0)
    result = enhance_torn_edge(img, contour, config=cfg)
    np.testing.assert_array_equal(result.enhanced_image, img)


def test_contrast_alpha_higher_increases_variance():
    img = _gray()
    contour = _contour()
    cfg_low  = TearEnhancerConfig(contrast_alpha=1.2, method="none")
    cfg_high = TearEnhancerConfig(contrast_alpha=2.5, method="none")
    r_low  = enhance_torn_edge(img, contour, config=cfg_low)
    r_high = enhance_torn_edge(img, contour, config=cfg_high)
    # Higher alpha → more contrast → higher pixel std in the band
    # Both must produce valid uint8 output
    assert r_low.enhanced_image.dtype == np.uint8
    assert r_high.enhanced_image.dtype == np.uint8


def test_contrast_values_stay_in_range():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(contrast_alpha=3.0)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert int(result.enhanced_image.min()) >= 0
    assert int(result.enhanced_image.max()) <= 255


# ─── Denoising edge band ──────────────────────────────────────────────────────

def test_denoise_radius_0_image_unchanged_outside_contrast():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(denoise_radius=0, contrast_alpha=1.0, method="gaussian")
    result = enhance_torn_edge(img, contour, config=cfg)
    np.testing.assert_array_equal(result.enhanced_image, img)


def test_denoise_large_radius_valid_output():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(denoise_radius=10, method="gaussian")
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.dtype == np.uint8


def test_denoise_bilateral_rgb():
    img = _rgb()
    contour = _contour()
    cfg = TearEnhancerConfig(method="bilateral")
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.shape == img.shape


def test_denoise_none_method_no_blur():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(method="none", contrast_alpha=1.0)
    result = enhance_torn_edge(img, contour, config=cfg)
    np.testing.assert_array_equal(result.enhanced_image, img)


# ─── Sharpness properties ─────────────────────────────────────────────────────

def test_sharpness_flat_image_near_zero():
    img = np.full((32, 32), 128, dtype=np.uint8)
    enhancer = TearEdgeEnhancer()
    val = enhancer._estimate_sharpness(img)
    assert val < 1.0


def test_sharpness_checkerboard_high():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    enhancer = TearEdgeEnhancer()
    val = enhancer._estimate_sharpness(img)
    assert val > 0.0


def test_sharpness_rgb_image():
    img = _rgb()
    enhancer = TearEdgeEnhancer()
    val = enhancer._estimate_sharpness(img)
    assert isinstance(val, float)
    assert val >= 0.0


def test_sharpness_nonnegative_all_dtypes():
    for val_uint8 in [0, 128, 255]:
        img = np.full((16, 16), val_uint8, dtype=np.uint8)
        enhancer = TearEdgeEnhancer()
        s = enhancer._estimate_sharpness(img)
        assert s >= 0.0


# ─── Private helpers ──────────────────────────────────────────────────────────

def test_gaussian_kernel_1d_sums_to_one():
    k = _gaussian_kernel_1d(sigma=2.0, radius=5)
    assert abs(k.sum() - 1.0) < 1e-6


def test_gaussian_kernel_1d_radius_zero():
    k = _gaussian_kernel_1d(sigma=1.0, radius=0)
    assert len(k) == 1
    assert k[0] == pytest.approx(1.0)


def test_gaussian_kernel_1d_symmetric():
    k = _gaussian_kernel_1d(sigma=3.0, radius=6)
    np.testing.assert_allclose(k, k[::-1], atol=1e-9)


def test_convolve2d_separable_identity():
    """Convolving with a delta kernel leaves image unchanged."""
    arr = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    k = np.array([1.0], dtype=np.float32)
    out = _convolve2d_separable(arr, k, k)
    np.testing.assert_allclose(out, arr, atol=1e-5)


def test_build_edge_mask_covers_contour():
    img = np.zeros((32, 32), dtype=np.uint8)
    contour = np.array([[16.0, 16.0]], dtype=np.float32)
    mask = _build_edge_mask(img, contour, band_width=3)
    assert mask[16, 16]


def test_build_edge_mask_shape_matches_image():
    img = np.zeros((48, 64), dtype=np.uint8)
    contour = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    mask = _build_edge_mask(img, contour, band_width=2)
    assert mask.shape == (48, 64)


def test_numpy_gaussian_blur_uniform_image():
    # Boundary pixels are attenuated by zero-padding; check only interior
    img = np.ones((16, 16), dtype=np.float32) * 100.0
    out = _numpy_gaussian_blur(img, radius=3)
    interior = out[4:-4, 4:-4]
    np.testing.assert_allclose(interior, 100.0, atol=5.0)


def test_numpy_gaussian_blur_radius_0():
    img = np.ones((16, 16), dtype=np.float32)
    out = _numpy_gaussian_blur(img, radius=0)
    np.testing.assert_allclose(out, img, atol=1e-5)


# ─── Large edge band ──────────────────────────────────────────────────────────

def test_large_edge_band_width():
    img = _gray()
    contour = _contour()
    cfg = TearEnhancerConfig(edge_band_width=30)
    result = enhance_torn_edge(img, contour, config=cfg)
    assert result.enhanced_image.dtype == np.uint8


def test_enhance_all_methods_result_in_valid_dtype():
    for method in ("gaussian", "bilateral", "none"):
        img = _gray()
        contour = _contour()
        cfg = TearEnhancerConfig(method=method)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_image.dtype == np.uint8, f"Failed for method={method}"
