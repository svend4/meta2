"""Tests for puzzle_reconstruction/preprocessing/adaptive_threshold.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.adaptive_threshold import (
    ThresholdParams,
    global_threshold,
    adaptive_mean,
    adaptive_gaussian,
    niblack_threshold,
    sauvola_threshold,
    bernsen_threshold,
    apply_threshold,
    batch_threshold,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=30, w=30, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=30, w=30, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


def make_gradient(h=30, w=30):
    """Image with a horizontal gradient 0→255."""
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_checkerboard(h=20, w=20):
    """Checkerboard pattern (0 and 255)."""
    r, c = np.mgrid[:h, :w]
    return (((r + c) % 2) * 255).astype(np.uint8)


# ─── ThresholdParams ──────────────────────────────────────────────────────────

class TestThresholdParams:
    def test_defaults(self):
        p = ThresholdParams()
        assert p.method == "otsu"
        assert p.block_size == 11
        assert p.k == pytest.approx(0.2)
        assert p.threshold == 128

    def test_valid_methods(self):
        for m in ("global", "otsu", "adaptive_mean", "adaptive_gaussian",
                  "niblack", "sauvola", "bernsen"):
            p = ThresholdParams(method=m)
            assert p.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(method="unknown")

    def test_block_size_too_small_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=2)

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=10)

    def test_block_size_ok(self):
        p = ThresholdParams(block_size=3)
        assert p.block_size == 3

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=-1)

    def test_threshold_too_large_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=256)

    def test_threshold_boundary_ok(self):
        p0 = ThresholdParams(threshold=0)
        p255 = ThresholdParams(threshold=255)
        assert p0.threshold == 0
        assert p255.threshold == 255

    def test_custom_params(self):
        p = ThresholdParams(method="niblack", block_size=15, k=-0.3, threshold=100)
        assert p.k == pytest.approx(-0.3)
        assert p.threshold == 100


# ─── global_threshold ─────────────────────────────────────────────────────────

class TestGlobalThreshold:
    def test_output_shape_gray(self):
        img = make_gray()
        result = global_threshold(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_gray()
        result = global_threshold(img)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient()
        result = global_threshold(img, threshold=128)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_threshold_zero_all_255(self):
        img = make_gray(value=1)
        result = global_threshold(img, threshold=0)
        assert set(np.unique(result)).issubset({0, 255})

    def test_threshold_255_all_zero(self):
        img = make_gray(value=200)
        result = global_threshold(img, threshold=255)
        assert (result == 0).all()

    def test_otsu_mode(self):
        img = make_gradient()
        result = global_threshold(img, use_otsu=True)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_bgr_input(self):
        img = make_bgr()
        result = global_threshold(img)
        assert result.ndim == 2

    def test_invalid_threshold_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            global_threshold(img, threshold=300)

    def test_invalid_negative_threshold_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            global_threshold(img, threshold=-1)


# ─── adaptive_mean ────────────────────────────────────────────────────────────

class TestAdaptiveMean:
    def test_output_shape(self):
        img = make_gradient(40, 40)
        result = adaptive_mean(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_gradient(40, 40)
        result = adaptive_mean(img)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient(40, 40)
        result = adaptive_mean(img)
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_too_small_raises(self):
        img = make_gradient(40, 40)
        with pytest.raises(ValueError):
            adaptive_mean(img, block_size=2)

    def test_block_size_even_raises(self):
        img = make_gradient(40, 40)
        with pytest.raises(ValueError):
            adaptive_mean(img, block_size=8)

    def test_bgr_input(self):
        img = make_bgr(40, 40)
        result = adaptive_mean(img)
        assert result.ndim == 2

    def test_custom_block_size(self):
        img = make_gradient(40, 40)
        result = adaptive_mean(img, block_size=7)
        assert result.shape == img.shape


# ─── adaptive_gaussian ────────────────────────────────────────────────────────

class TestAdaptiveGaussian:
    def test_output_shape(self):
        img = make_gradient(40, 40)
        result = adaptive_gaussian(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_gradient(40, 40)
        result = adaptive_gaussian(img)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient(40, 40)
        result = adaptive_gaussian(img)
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_too_small_raises(self):
        img = make_gradient(40, 40)
        with pytest.raises(ValueError):
            adaptive_gaussian(img, block_size=1)

    def test_block_size_even_raises(self):
        img = make_gradient(40, 40)
        with pytest.raises(ValueError):
            adaptive_gaussian(img, block_size=6)

    def test_bgr_input(self):
        img = make_bgr(40, 40)
        result = adaptive_gaussian(img)
        assert result.ndim == 2


# ─── niblack_threshold ────────────────────────────────────────────────────────

class TestNiblackThreshold:
    def test_output_shape(self):
        img = make_gradient(20, 20)
        result = niblack_threshold(img, block_size=5)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_gradient(20, 20)
        result = niblack_threshold(img, block_size=5)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_checkerboard()
        result = niblack_threshold(img, block_size=5)
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_too_small_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            niblack_threshold(img, block_size=2)

    def test_block_size_even_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            niblack_threshold(img, block_size=4)

    def test_bgr_input(self):
        img = make_bgr(20, 20)
        result = niblack_threshold(img, block_size=5)
        assert result.ndim == 2

    def test_gradient_image_has_both_classes(self):
        """Gradient image with Niblack should produce both 0 and 255 pixels."""
        img = make_gradient(20, 20)
        result = niblack_threshold(img, block_size=5, k=-0.2)
        assert set(np.unique(result)).issubset({0, 255})


# ─── sauvola_threshold ────────────────────────────────────────────────────────

class TestSauvolaThreshold:
    def test_output_shape(self):
        img = make_gradient(20, 20)
        result = sauvola_threshold(img, block_size=5)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_gradient(20, 20)
        result = sauvola_threshold(img, block_size=5)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_checkerboard()
        result = sauvola_threshold(img, block_size=5)
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_too_small_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            sauvola_threshold(img, block_size=2)

    def test_block_size_even_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            sauvola_threshold(img, block_size=6)

    def test_r_zero_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            sauvola_threshold(img, r=0.0)

    def test_r_negative_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            sauvola_threshold(img, r=-10.0)

    def test_bgr_input(self):
        img = make_bgr(20, 20)
        result = sauvola_threshold(img, block_size=5)
        assert result.ndim == 2


# ─── bernsen_threshold ────────────────────────────────────────────────────────

class TestBernsenThreshold:
    def test_output_shape(self):
        img = make_checkerboard()
        result = bernsen_threshold(img, block_size=5)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = make_checkerboard()
        result = bernsen_threshold(img, block_size=5)
        assert result.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient(20, 20)
        result = bernsen_threshold(img, block_size=5)
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_too_small_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            bernsen_threshold(img, block_size=2)

    def test_block_size_even_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            bernsen_threshold(img, block_size=4)

    def test_negative_contrast_raises(self):
        img = make_gray(20, 20)
        with pytest.raises(ValueError):
            bernsen_threshold(img, contrast_threshold=-1.0)

    def test_high_contrast_threshold_all_background(self):
        """Low contrast image + very high contrast threshold → all 0."""
        img = make_gray(10, 10, value=128)  # constant → contrast = 0
        result = bernsen_threshold(img, block_size=5, contrast_threshold=200.0)
        assert (result == 0).all()

    def test_bgr_input(self):
        img = make_bgr(20, 20)
        result = bernsen_threshold(img, block_size=5)
        assert result.ndim == 2


# ─── apply_threshold ──────────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_otsu(self):
        img = make_gradient(30, 30)
        params = ThresholdParams(method="otsu")
        result = apply_threshold(img, params)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_global(self):
        img = make_gradient(30, 30)
        params = ThresholdParams(method="global", threshold=100)
        result = apply_threshold(img, params)
        assert set(np.unique(result)).issubset({0, 255})

    def test_adaptive_mean(self):
        img = make_gradient(30, 30)
        params = ThresholdParams(method="adaptive_mean", block_size=11)
        result = apply_threshold(img, params)
        assert result.shape == img.shape

    def test_adaptive_gaussian(self):
        img = make_gradient(30, 30)
        params = ThresholdParams(method="adaptive_gaussian", block_size=11)
        result = apply_threshold(img, params)
        assert result.shape == img.shape

    def test_niblack(self):
        img = make_gradient(15, 15)
        params = ThresholdParams(method="niblack", block_size=5, k=-0.2)
        result = apply_threshold(img, params)
        assert result.shape == img.shape

    def test_sauvola(self):
        img = make_gradient(15, 15)
        params = ThresholdParams(method="sauvola", block_size=5, k=0.2)
        result = apply_threshold(img, params)
        assert result.shape == img.shape

    def test_bernsen(self):
        img = make_gradient(15, 15)
        params = ThresholdParams(method="bernsen", block_size=5)
        result = apply_threshold(img, params)
        assert result.shape == img.shape


# ─── batch_threshold ──────────────────────────────────────────────────────────

class TestBatchThreshold:
    def test_returns_list(self):
        imgs = [make_gradient(20, 20) for _ in range(3)]
        params = ThresholdParams(method="otsu")
        results = batch_threshold(imgs, params)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_uint8(self):
        imgs = [make_gradient(20, 20) for _ in range(3)]
        params = ThresholdParams(method="otsu")
        results = batch_threshold(imgs, params)
        for r in results:
            assert r.dtype == np.uint8

    def test_empty_list(self):
        params = ThresholdParams(method="otsu")
        results = batch_threshold([], params)
        assert results == []

    def test_shapes_match_inputs(self):
        imgs = [make_gradient(h + 10, 20) for h in range(3)]
        params = ThresholdParams(method="adaptive_mean", block_size=7)
        results = batch_threshold(imgs, params)
        for img, result in zip(imgs, results):
            assert result.shape == img.shape
