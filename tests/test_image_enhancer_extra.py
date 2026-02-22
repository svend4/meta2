"""Additional tests for puzzle_reconstruction.preprocessing.image_enhancer."""
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
    return np.random.default_rng(seed).integers(30, 220, (h, w), dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(30, 220, (h, w, 3), dtype=np.uint8)


def _const(val=128, h=32, w=32):
    return np.full((h, w), val, dtype=np.uint8)


# ─── TestEnhanceConfigExtra ───────────────────────────────────────────────────

class TestEnhanceConfigExtra:
    def test_kernel_size_3_ok(self):
        cfg = EnhanceConfig(kernel_size=3)
        assert cfg.kernel_size == 3

    def test_kernel_size_7_ok(self):
        cfg = EnhanceConfig(kernel_size=7)
        assert cfg.kernel_size == 7

    def test_sharpness_none_ok(self):
        cfg = EnhanceConfig(sharpness="none")
        assert cfg.sharpness == "none"

    def test_sharpness_strong_ok(self):
        cfg = EnhanceConfig(sharpness="strong")
        assert cfg.sharpness == "strong"

    def test_denoise_none_ok(self):
        cfg = EnhanceConfig(denoise="none")
        assert cfg.denoise == "none"

    def test_denoise_gaussian_ok(self):
        cfg = EnhanceConfig(denoise="gaussian")
        assert cfg.denoise == "gaussian"

    def test_denoise_median_ok(self):
        cfg = EnhanceConfig(denoise="median")
        assert cfg.denoise == "median"

    def test_denoise_bilateral_ok(self):
        cfg = EnhanceConfig(denoise="bilateral")
        assert cfg.denoise == "bilateral"

    def test_contrast_none_ok(self):
        cfg = EnhanceConfig(contrast="none")
        assert cfg.contrast == "none"

    def test_contrast_clahe_ok(self):
        cfg = EnhanceConfig(contrast="clahe")
        assert cfg.contrast == "clahe"

    def test_contrast_stretch_ok(self):
        cfg = EnhanceConfig(contrast="stretch")
        assert cfg.contrast == "stretch"

    def test_kernel_size_9_ok(self):
        cfg = EnhanceConfig(kernel_size=9)
        assert cfg.kernel_size == 9


# ─── TestEnhanceResultExtra ───────────────────────────────────────────────────

class TestEnhanceResultExtra:
    def test_operations_empty_list_ok(self):
        r = EnhanceResult(image=_gray(), operations=[],
                          mean_before=100.0, mean_after=100.0)
        assert r.operations == []

    def test_mean_before_zero_ok(self):
        r = EnhanceResult(image=_gray(), operations=[],
                          mean_before=0.0, mean_after=0.0)
        assert r.mean_before == pytest.approx(0.0)

    def test_mean_after_255_ok(self):
        r = EnhanceResult(image=_gray(), operations=[],
                          mean_before=100.0, mean_after=255.0)
        assert r.mean_after == pytest.approx(255.0)

    def test_multiple_operations_stored(self):
        ops = ["denoise:gaussian", "sharpen:mild", "contrast:clahe"]
        r = EnhanceResult(image=_gray(), operations=ops,
                          mean_before=100.0, mean_after=110.0)
        assert len(r.operations) == 3

    def test_image_dtype_preserved(self):
        img = _bgr()
        r = EnhanceResult(image=img, operations=[],
                          mean_before=100.0, mean_after=100.0)
        assert r.image.dtype == np.uint8

    def test_delta_mean_zero_when_equal(self):
        r = EnhanceResult(image=_gray(), operations=[],
                          mean_before=128.0, mean_after=128.0)
        assert r.delta_mean == pytest.approx(0.0)


# ─── TestSharpenImageExtra ────────────────────────────────────────────────────

class TestSharpenImageExtra:
    def test_kernel_size_3_gray(self):
        img = _gray()
        result = sharpen_image(img, mode="mild", kernel_size=3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_kernel_size_5_gray(self):
        img = _gray()
        result = sharpen_image(img, mode="strong", kernel_size=5)
        assert result.shape == img.shape

    def test_kernel_size_7_gray(self):
        img = _gray()
        result = sharpen_image(img, mode="mild", kernel_size=7)
        assert result.shape == img.shape

    def test_bgr_strong_mode(self):
        img = _bgr()
        result = sharpen_image(img, mode="strong")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_non_square_image(self):
        img = _gray(h=16, w=48)
        result = sharpen_image(img, mode="mild")
        assert result.shape == (16, 48)

    def test_constant_image_unchanged(self):
        img = _const(200)
        result = sharpen_image(img, mode="mild")
        assert result.shape == img.shape

    def test_output_values_in_range(self):
        img = _bgr(seed=3)
        result = sharpen_image(img, mode="strong")
        assert result.min() >= 0 and result.max() <= 255


# ─── TestDenoiseImageExtra ────────────────────────────────────────────────────

class TestDenoiseImageExtra:
    def test_median_bgr(self):
        img = _bgr()
        result = denoise_image(img, mode="median")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_bilateral_bgr(self):
        img = _bgr()
        result = denoise_image(img, mode="bilateral")
        assert result.shape == img.shape

    def test_kernel_size_5_gaussian(self):
        img = _gray()
        result = denoise_image(img, mode="gaussian", kernel_size=5)
        assert result.shape == img.shape

    def test_kernel_size_7_median(self):
        img = _gray()
        result = denoise_image(img, mode="median", kernel_size=7)
        assert result.shape == img.shape

    def test_constant_image_gaussian(self):
        img = _const(128)
        result = denoise_image(img, mode="gaussian")
        assert result.shape == img.shape

    def test_non_square_gaussian(self):
        img = _gray(h=16, w=48)
        result = denoise_image(img, mode="gaussian")
        assert result.shape == (16, 48)

    def test_output_uint8_median(self):
        img = _bgr(seed=5)
        result = denoise_image(img, mode="median")
        assert result.dtype == np.uint8


# ─── TestEnhanceContrastExtra ─────────────────────────────────────────────────

class TestEnhanceContrastExtra:
    def test_constant_bgr_clahe(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = enhance_contrast(img, mode="clahe")
        assert result.shape == img.shape

    def test_constant_bgr_stretch(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        result = enhance_contrast(img, mode="stretch")
        assert result.shape == img.shape

    def test_non_square_stretch_gray(self):
        img = _gray(h=16, w=64)
        result = enhance_contrast(img, mode="stretch")
        assert result.shape == (16, 64)

    def test_non_square_clahe_bgr(self):
        img = _bgr(h=16, w=64)
        result = enhance_contrast(img, mode="clahe")
        assert result.shape == (16, 64, 3)

    def test_output_dtype_uint8_clahe(self):
        result = enhance_contrast(_bgr(seed=2), mode="clahe")
        assert result.dtype == np.uint8

    def test_output_dtype_uint8_stretch(self):
        result = enhance_contrast(_gray(seed=3), mode="stretch")
        assert result.dtype == np.uint8


# ─── TestEnhanceImageExtra ────────────────────────────────────────────────────

class TestEnhanceImageExtra:
    def test_bgr_all_ops(self):
        cfg = EnhanceConfig(sharpness="mild", denoise="gaussian", contrast="clahe")
        result = enhance_image(_bgr(), cfg)
        assert isinstance(result, EnhanceResult)
        assert len(result.operations) == 3

    def test_mean_after_recorded(self):
        img = _gray()
        result = enhance_image(img, EnhanceConfig(denoise="gaussian"))
        assert result.mean_after >= 0.0

    def test_output_shape_preserved_gray(self):
        img = _gray(h=16, w=48)
        result = enhance_image(img)
        assert result.image.shape == (16, 48)

    def test_output_shape_preserved_bgr(self):
        img = _bgr(h=16, w=48)
        result = enhance_image(img)
        assert result.image.shape == (16, 48, 3)

    def test_contrast_only_one_op(self):
        cfg = EnhanceConfig(contrast="clahe")
        result = enhance_image(_gray(), cfg)
        assert len(result.operations) == 1

    def test_sharpen_only_one_op(self):
        cfg = EnhanceConfig(sharpness="strong")
        result = enhance_image(_gray(), cfg)
        assert len(result.operations) == 1

    def test_result_image_uint8(self):
        result = enhance_image(_bgr(), EnhanceConfig(contrast="stretch"))
        assert result.image.dtype == np.uint8


# ─── TestBatchEnhanceExtra ────────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_mixed_gray_and_bgr(self):
        images = [_gray(), _bgr(), _gray(seed=2)]
        cfg = EnhanceConfig(denoise="gaussian")
        result = batch_enhance(images, cfg)
        assert len(result) == 3

    def test_large_batch(self):
        images = [_gray(seed=i) for i in range(8)]
        result = batch_enhance(images)
        assert len(result) == 8

    def test_all_mean_before_nonneg(self):
        images = [_gray(seed=i) for i in range(3)]
        for r in batch_enhance(images):
            assert r.mean_before >= 0.0

    def test_all_output_uint8(self):
        images = [_bgr(seed=i) for i in range(3)]
        cfg = EnhanceConfig(sharpness="mild")
        for r in batch_enhance(images, cfg):
            assert r.image.dtype == np.uint8

    def test_denoise_median_batch(self):
        images = [_gray(seed=i) for i in range(3)]
        cfg = EnhanceConfig(denoise="median")
        result = batch_enhance(images, cfg)
        for r in result:
            assert any("denoise" in op for op in r.operations)
