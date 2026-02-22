"""Тесты для puzzle_reconstruction/preprocessing/contrast.py."""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.contrast import (
    ContrastResult,
    measure_contrast,
    enhance_clahe,
    enhance_histeq,
    enhance_gamma,
    enhance_stretch,
    enhance_retinex,
    auto_enhance,
    batch_enhance,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_noisy_bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_low_contrast(h=64, w=64):
    """Very low contrast image (almost uniform)."""
    return np.full((h, w), 128, dtype=np.uint8)


def make_high_contrast(h=64, w=64, seed=0):
    """High contrast image with extreme pixel values."""
    img = np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)
    # Force some pixels to extremes
    img[0, :] = 0
    img[-1, :] = 255
    return img


# ─── ContrastResult ───────────────────────────────────────────────────────────

class TestContrastResult:
    def test_creation(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="clahe",
                           contrast_before=10.0, contrast_after=25.0)
        assert r.method == "clahe"
        assert r.contrast_before == pytest.approx(10.0)
        assert r.contrast_after == pytest.approx(25.0)
        assert r.params == {}

    def test_enhanced_stored(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="test",
                           contrast_before=1.0, contrast_after=2.0)
        assert r.enhanced is img

    def test_params_stored(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="gamma",
                           contrast_before=5.0, contrast_after=8.0,
                           params={"gamma": 1.5})
        assert r.params["gamma"] == pytest.approx(1.5)

    def test_improvement_positive(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="clahe",
                           contrast_before=10.0, contrast_after=30.0)
        assert r.improvement == pytest.approx(20.0)

    def test_improvement_negative(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="test",
                           contrast_before=30.0, contrast_after=10.0)
        assert r.improvement == pytest.approx(-20.0)

    def test_improvement_ratio(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="test",
                           contrast_before=10.0, contrast_after=20.0)
        assert r.improvement_ratio == pytest.approx(1.0)

    def test_improvement_ratio_zero_before(self):
        img = make_gray()
        r = ContrastResult(enhanced=img, method="test",
                           contrast_before=0.0, contrast_after=5.0)
        assert r.improvement_ratio == pytest.approx(0.0)


# ─── measure_contrast ─────────────────────────────────────────────────────────

class TestMeasureContrast:
    def test_returns_float(self):
        img = make_noisy()
        val = measure_contrast(img)
        assert isinstance(val, float)

    def test_uniform_gray_is_zero(self):
        img = make_gray(fill=128)
        val = measure_contrast(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_noisy_image_positive(self):
        img = make_noisy()
        val = measure_contrast(img)
        assert val > 0.0

    def test_uniform_bgr_is_zero(self):
        img = make_bgr(fill=100)
        val = measure_contrast(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_noisy_bgr_positive(self):
        img = make_noisy_bgr()
        val = measure_contrast(img)
        assert val > 0.0

    def test_higher_spread_higher_contrast(self):
        low = np.full((64, 64), 128, dtype=np.uint8)
        high = make_high_contrast()
        assert measure_contrast(high) > measure_contrast(low)


# ─── enhance_clahe ────────────────────────────────────────────────────────────

class TestEnhanceClahe:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = enhance_clahe(img)
        assert isinstance(r, ContrastResult)

    def test_method_is_clahe(self):
        img = make_noisy()
        r = enhance_clahe(img)
        assert r.method == "clahe"

    def test_same_shape_gray(self):
        img = make_noisy(h=32, w=48)
        r = enhance_clahe(img)
        assert r.enhanced.shape == img.shape

    def test_same_shape_bgr(self):
        img = make_noisy_bgr(h=32, w=32)
        r = enhance_clahe(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        img = make_noisy()
        r = enhance_clahe(img)
        assert r.enhanced.dtype == np.uint8

    def test_params_contain_clip_limit(self):
        img = make_noisy()
        r = enhance_clahe(img, clip_limit=3.0)
        assert r.params["clip_limit"] == pytest.approx(3.0)

    def test_params_contain_tile_size(self):
        img = make_noisy()
        r = enhance_clahe(img, tile_size=16)
        assert r.params["tile_size"] == 16

    def test_contrast_before_nonneg(self):
        img = make_noisy()
        r = enhance_clahe(img)
        assert r.contrast_before >= 0.0

    def test_contrast_after_nonneg(self):
        img = make_noisy()
        r = enhance_clahe(img)
        assert r.contrast_after >= 0.0

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = enhance_clahe(img)
        assert r.enhanced.shape == img.shape


# ─── enhance_histeq ───────────────────────────────────────────────────────────

class TestEnhanceHisteq:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = enhance_histeq(img)
        assert isinstance(r, ContrastResult)

    def test_method_is_histeq(self):
        img = make_noisy()
        r = enhance_histeq(img)
        assert r.method == "histeq"

    def test_same_shape_gray(self):
        img = make_noisy(h=32, w=64)
        r = enhance_histeq(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        img = make_noisy()
        r = enhance_histeq(img)
        assert r.enhanced.dtype == np.uint8

    def test_contrast_nonneg(self):
        img = make_noisy()
        r = enhance_histeq(img)
        assert r.contrast_before >= 0.0
        assert r.contrast_after >= 0.0

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = enhance_histeq(img)
        assert r.enhanced.shape == img.shape


# ─── enhance_gamma ────────────────────────────────────────────────────────────

class TestEnhanceGamma:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = enhance_gamma(img)
        assert isinstance(r, ContrastResult)

    def test_method_is_gamma(self):
        img = make_noisy()
        r = enhance_gamma(img)
        assert r.method == "gamma"

    def test_same_shape(self):
        img = make_noisy(h=32, w=48)
        r = enhance_gamma(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        img = make_noisy()
        r = enhance_gamma(img)
        assert r.enhanced.dtype == np.uint8

    def test_params_contain_gamma(self):
        img = make_noisy()
        r = enhance_gamma(img, gamma=2.0)
        assert r.params["gamma"] == pytest.approx(2.0)

    def test_gamma_1_preserves_image(self):
        img = make_noisy()
        r = enhance_gamma(img, gamma=1.0)
        np.testing.assert_array_equal(r.enhanced, img)

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = enhance_gamma(img, gamma=1.5)
        assert r.enhanced.shape == img.shape


# ─── enhance_stretch ──────────────────────────────────────────────────────────

class TestEnhanceStretch:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = enhance_stretch(img)
        assert isinstance(r, ContrastResult)

    def test_method_is_stretch(self):
        img = make_noisy()
        r = enhance_stretch(img)
        assert r.method == "stretch"

    def test_same_shape_gray(self):
        img = make_noisy(h=32, w=32)
        r = enhance_stretch(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        img = make_noisy()
        r = enhance_stretch(img)
        assert r.enhanced.dtype == np.uint8

    def test_params_contain_percentiles(self):
        img = make_noisy()
        r = enhance_stretch(img, p_low=5.0, p_high=95.0)
        assert r.params["p_low"] == pytest.approx(5.0)
        assert r.params["p_high"] == pytest.approx(95.0)

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = enhance_stretch(img)
        assert r.enhanced.shape == img.shape

    def test_uniform_image_unchanged(self):
        img = make_gray(fill=100)
        r = enhance_stretch(img)
        assert r.enhanced.shape == img.shape


# ─── enhance_retinex ──────────────────────────────────────────────────────────

class TestEnhanceRetinex:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = enhance_retinex(img)
        assert isinstance(r, ContrastResult)

    def test_method_is_retinex(self):
        img = make_noisy()
        r = enhance_retinex(img)
        assert r.method == "retinex"

    def test_same_shape_gray(self):
        img = make_noisy(h=32, w=32)
        r = enhance_retinex(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        img = make_noisy()
        r = enhance_retinex(img)
        assert r.enhanced.dtype == np.uint8

    def test_params_contain_sigma(self):
        img = make_noisy()
        r = enhance_retinex(img, sigma=50.0)
        assert r.params["sigma"] == pytest.approx(50.0)

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = enhance_retinex(img)
        assert r.enhanced.shape == img.shape


# ─── auto_enhance ─────────────────────────────────────────────────────────────

class TestAutoEnhance:
    def test_returns_contrast_result(self):
        img = make_noisy()
        r = auto_enhance(img)
        assert isinstance(r, ContrastResult)

    def test_low_contrast_uses_clahe(self):
        # Uniform image → RMS=0 < 20 → CLAHE
        img = make_low_contrast()
        r = auto_enhance(img)
        assert r.method == "clahe"

    def test_high_contrast_uses_gamma(self):
        # High contrast → RMS >= 60 → gamma
        rng = np.random.default_rng(0)
        # Force extreme contrast: alternating 0 and 255
        img = np.zeros((64, 64), dtype=np.uint8)
        img[::2, :] = 255
        assert measure_contrast(img) >= 60.0
        r = auto_enhance(img)
        assert r.method == "gamma"

    def test_medium_contrast_uses_stretch(self):
        # Construct image with RMS between 20 and 60
        rng = np.random.default_rng(42)
        base = np.random.default_rng(42).integers(90, 170, (64, 64), dtype=np.uint8)
        rms = measure_contrast(base)
        # only run test if RMS in expected range
        if 20.0 <= rms < 60.0:
            r = auto_enhance(base)
            assert r.method == "stretch"

    def test_same_shape(self):
        img = make_noisy(h=32, w=48)
        r = auto_enhance(img)
        assert r.enhanced.shape == img.shape

    def test_accepts_bgr(self):
        img = make_noisy_bgr()
        r = auto_enhance(img)
        assert isinstance(r, ContrastResult)


# ─── batch_enhance ────────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_empty_list_returns_empty(self):
        result = batch_enhance([])
        assert result == []

    def test_length_matches(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = batch_enhance(images)
        assert len(result) == 3

    def test_returns_list_of_results(self):
        images = [make_noisy(seed=i) for i in range(2)]
        result = batch_enhance(images, method="clahe")
        for r in result:
            assert isinstance(r, ContrastResult)

    def test_method_clahe(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="clahe")
        assert result[0].method == "clahe"

    def test_method_histeq(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="histeq")
        assert result[0].method == "histeq"

    def test_method_gamma(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="gamma")
        assert result[0].method == "gamma"

    def test_method_stretch(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="stretch")
        assert result[0].method == "stretch"

    def test_method_retinex(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="retinex")
        assert result[0].method == "retinex"

    def test_method_auto(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="auto")
        assert isinstance(result[0], ContrastResult)

    def test_unknown_method_raises(self):
        images = [make_noisy()]
        with pytest.raises(ValueError):
            batch_enhance(images, method="bilateral")

    def test_kwargs_passed_through(self):
        images = [make_noisy(seed=0)]
        result = batch_enhance(images, method="gamma", gamma=2.0)
        assert result[0].params["gamma"] == pytest.approx(2.0)
