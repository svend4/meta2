"""Тесты для puzzle_reconstruction.preprocessing.image_enhancer."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.image_enhancer import (
    EnhanceConfig,
    EnhanceResult,
    batch_enhance,
    denoise_image,
    enhance_contrast,
    enhance_image,
    sharpen_image,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w), dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


# ─── TestEnhanceConfig ────────────────────────────────────────────────────────

class TestEnhanceConfig:
    def test_defaults(self):
        cfg = EnhanceConfig()
        assert cfg.sharpness == "none"
        assert cfg.denoise == "none"
        assert cfg.contrast == "none"
        assert cfg.kernel_size == 3

    def test_invalid_sharpness(self):
        with pytest.raises(ValueError):
            EnhanceConfig(sharpness="extreme")

    def test_invalid_denoise(self):
        with pytest.raises(ValueError):
            EnhanceConfig(denoise="nlm")

    def test_invalid_contrast(self):
        with pytest.raises(ValueError):
            EnhanceConfig(contrast="histeq")

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=4)

    def test_small_kernel_size_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=1)

    def test_valid_full_config(self):
        cfg = EnhanceConfig(sharpness="mild", denoise="gaussian",
                            contrast="clahe", kernel_size=5)
        assert cfg.sharpness == "mild"
        assert cfg.denoise == "gaussian"
        assert cfg.contrast == "clahe"
        assert cfg.kernel_size == 5


# ─── TestEnhanceResult ────────────────────────────────────────────────────────

class TestEnhanceResult:
    def _make(self, mean_before=100.0, mean_after=110.0):
        return EnhanceResult(
            image=_gray(),
            operations=["denoise:gaussian"],
            mean_before=mean_before,
            mean_after=mean_after,
        )

    def test_delta_mean(self):
        r = self._make(100.0, 120.0)
        assert abs(r.delta_mean - 20.0) < 1e-10

    def test_delta_mean_negative(self):
        r = self._make(120.0, 100.0)
        assert abs(r.delta_mean - (-20.0)) < 1e-10

    def test_negative_mean_before_raises(self):
        with pytest.raises(ValueError):
            EnhanceResult(image=_gray(), operations=[],
                          mean_before=-1.0, mean_after=0.0)

    def test_negative_mean_after_raises(self):
        with pytest.raises(ValueError):
            EnhanceResult(image=_gray(), operations=[],
                          mean_before=0.0, mean_after=-1.0)

    def test_operations_list(self):
        r = self._make()
        assert isinstance(r.operations, list)


# ─── TestSharpenImage ─────────────────────────────────────────────────────────

class TestSharpenImage:
    def test_output_shape_gray(self):
        img = _gray()
        result = sharpen_image(img, mode="mild")
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        result = sharpen_image(_gray(), mode="mild")
        assert result.dtype == np.uint8

    def test_mild_and_strong_differ(self):
        img = _gray()
        mild = sharpen_image(img, mode="mild")
        strong = sharpen_image(img, mode="strong")
        assert not np.array_equal(mild, strong)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            sharpen_image(_gray(), mode="extreme")

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            sharpen_image(_gray(), kernel_size=4)

    def test_output_in_valid_range(self):
        result = sharpen_image(_gray(), mode="strong")
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_shape_preserved(self):
        img = _bgr()
        result = sharpen_image(img, mode="mild")
        assert result.shape == img.shape


# ─── TestDenoiseImage ─────────────────────────────────────────────────────────

class TestDenoiseImage:
    def test_gaussian_gray(self):
        img = _gray()
        result = denoise_image(img, mode="gaussian")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_median_gray(self):
        img = _gray()
        result = denoise_image(img, mode="median")
        assert result.shape == img.shape

    def test_bilateral_gray(self):
        img = _gray()
        result = denoise_image(img, mode="bilateral")
        assert result.shape == img.shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            denoise_image(_gray(), mode="nlm")

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            denoise_image(_gray(), kernel_size=4)

    def test_gaussian_bgr(self):
        img = _bgr()
        result = denoise_image(img, mode="gaussian")
        assert result.shape == img.shape


# ─── TestEnhanceContrast ──────────────────────────────────────────────────────

class TestEnhanceContrast:
    def test_stretch_gray(self):
        img = _gray()
        result = enhance_contrast(img, mode="stretch")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_clahe_gray(self):
        img = _gray()
        result = enhance_contrast(img, mode="clahe")
        assert result.shape == img.shape

    def test_clahe_bgr(self):
        img = _bgr()
        result = enhance_contrast(img, mode="clahe")
        assert result.shape == img.shape

    def test_stretch_bgr(self):
        img = _bgr()
        result = enhance_contrast(img, mode="stretch")
        assert result.shape == img.shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(_gray(), mode="histeq")

    def test_output_in_valid_range(self):
        img = _gray()
        result = enhance_contrast(img, mode="stretch")
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_uniform_image_no_crash(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = enhance_contrast(img, mode="stretch")
        assert result.shape == img.shape


# ─── TestEnhanceImage ─────────────────────────────────────────────────────────

class TestEnhanceImage:
    def test_returns_enhance_result(self):
        result = enhance_image(_gray())
        assert isinstance(result, EnhanceResult)

    def test_no_ops_default(self):
        result = enhance_image(_gray())
        assert result.operations == []

    def test_single_op_denoise(self):
        cfg = EnhanceConfig(denoise="gaussian")
        result = enhance_image(_gray(), cfg)
        assert any("denoise" in op for op in result.operations)

    def test_single_op_sharpen(self):
        cfg = EnhanceConfig(sharpness="mild")
        result = enhance_image(_gray(), cfg)
        assert any("sharpen" in op for op in result.operations)

    def test_single_op_contrast(self):
        cfg = EnhanceConfig(contrast="stretch")
        result = enhance_image(_gray(), cfg)
        assert any("contrast" in op for op in result.operations)

    def test_all_ops(self):
        cfg = EnhanceConfig(sharpness="mild", denoise="gaussian", contrast="stretch")
        result = enhance_image(_gray(), cfg)
        assert len(result.operations) == 3

    def test_mean_before_recorded(self):
        img = _gray()
        result = enhance_image(img)
        assert abs(result.mean_before - float(img.mean())) < 1e-4

    def test_output_uint8(self):
        result = enhance_image(_bgr(), EnhanceConfig(denoise="gaussian"))
        assert result.image.dtype == np.uint8


# ─── TestBatchEnhance ─────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_empty_batch(self):
        result = batch_enhance([])
        assert result == []

    def test_single_image(self):
        result = batch_enhance([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], EnhanceResult)

    def test_multiple_images(self):
        images = [_gray(seed=i) for i in range(4)]
        result = batch_enhance(images)
        assert len(result) == 4

    def test_custom_cfg(self):
        cfg = EnhanceConfig(sharpness="strong")
        result = batch_enhance([_bgr(), _bgr(seed=1)], cfg)
        for r in result:
            assert any("sharpen" in op for op in r.operations)

    def test_returns_list(self):
        result = batch_enhance([_gray()])
        assert isinstance(result, list)
