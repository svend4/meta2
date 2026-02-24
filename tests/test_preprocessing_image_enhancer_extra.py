"""Extra tests for puzzle_reconstruction/preprocessing/image_enhancer.py"""
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

def _bgr(h=32, w=32):
    return np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h=32, w=32):
    return np.random.default_rng(42).integers(0, 256, (h, w), dtype=np.uint8)


def _white_bgr(h=32, w=32):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _dark_bgr(h=32, w=32):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─── TestEnhanceConfigExtra ───────────────────────────────────────────────────

class TestEnhanceConfigExtra:
    def test_kernel_size_9(self):
        c = EnhanceConfig(kernel_size=9)
        assert c.kernel_size == 9

    def test_kernel_size_11(self):
        c = EnhanceConfig(kernel_size=11)
        assert c.kernel_size == 11

    def test_all_none_defaults(self):
        c = EnhanceConfig()
        assert c.sharpness == "none"
        assert c.denoise == "none"
        assert c.contrast == "none"

    def test_strong_sharpness_bilateral_clahe(self):
        c = EnhanceConfig(sharpness="strong", denoise="bilateral", contrast="clahe")
        assert c.sharpness == "strong"
        assert c.denoise == "bilateral"
        assert c.contrast == "clahe"

    def test_median_denoise(self):
        c = EnhanceConfig(denoise="median")
        assert c.denoise == "median"

    def test_stretch_contrast(self):
        c = EnhanceConfig(contrast="stretch")
        assert c.contrast == "stretch"

    def test_all_invalid_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(sharpness="invalid_x")
        with pytest.raises(ValueError):
            EnhanceConfig(denoise="invalid_y")
        with pytest.raises(ValueError):
            EnhanceConfig(contrast="invalid_z")

    def test_kernel_2_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=2)

    def test_kernel_6_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=6)


# ─── TestEnhanceResultExtra ───────────────────────────────────────────────────

class TestEnhanceResultExtra:
    def test_multiple_operations(self):
        r = EnhanceResult(image=_bgr(), operations=["denoise:gaussian",
                           "sharpen:mild", "contrast:clahe"],
                          mean_before=100.0, mean_after=110.0)
        assert len(r.operations) == 3

    def test_delta_mean_zero(self):
        r = EnhanceResult(image=_bgr(), operations=[],
                          mean_before=128.0, mean_after=128.0)
        assert r.delta_mean == pytest.approx(0.0)

    def test_large_delta_mean(self):
        r = EnhanceResult(image=_bgr(), operations=[],
                          mean_before=0.0, mean_after=255.0)
        assert r.delta_mean == pytest.approx(255.0)

    def test_gray_image_shape(self):
        r = EnhanceResult(image=_gray(40, 50), operations=[],
                          mean_before=100.0, mean_after=100.0)
        assert r.image.shape == (40, 50)

    def test_operations_empty_list(self):
        r = EnhanceResult(image=_bgr(), operations=[],
                          mean_before=100.0, mean_after=100.0)
        assert r.operations == []

    def test_mean_before_zero_valid(self):
        r = EnhanceResult(image=_bgr(), operations=[],
                          mean_before=0.0, mean_after=0.0)
        assert r.mean_before == pytest.approx(0.0)


# ─── TestSharpenImageExtra ────────────────────────────────────────────────────

class TestSharpenImageExtra:
    def test_non_square_bgr(self):
        img = _bgr(h=40, w=80)
        result = sharpen_image(img)
        assert result.shape == (40, 80, 3)

    def test_non_square_gray(self):
        img = _gray(h=40, w=80)
        result = sharpen_image(img)
        assert result.shape == (40, 80)

    def test_all_modes_no_crash(self):
        img = _bgr()
        for mode in ("mild", "strong"):
            result = sharpen_image(img, mode=mode)
            assert result.dtype == np.uint8

    def test_none_mode_raises(self):
        img = _bgr()
        with pytest.raises(ValueError):
            sharpen_image(img, mode="none")

    def test_kernel_5(self):
        img = _bgr()
        result = sharpen_image(img, kernel_size=5, mode="mild")
        assert result.dtype == np.uint8

    def test_kernel_7(self):
        img = _bgr()
        result = sharpen_image(img, kernel_size=7, mode="strong")
        assert result.dtype == np.uint8

    def test_white_image_mild(self):
        img = _white_bgr()
        result = sharpen_image(img, mode="mild")
        assert result.dtype == np.uint8

    def test_dark_image_strong(self):
        img = _dark_bgr()
        result = sharpen_image(img, mode="strong")
        assert result.dtype == np.uint8

    def test_output_range(self):
        img = _bgr()
        for mode in ("mild", "strong"):
            result = sharpen_image(img, mode=mode)
            assert int(result.min()) >= 0
            assert int(result.max()) <= 255


# ─── TestDenoiseImageExtra ────────────────────────────────────────────────────

class TestDenoiseImageExtra:
    def test_non_square_bgr(self):
        img = _bgr(h=40, w=80)
        result = denoise_image(img)
        assert result.shape == (40, 80, 3)

    def test_non_square_gray(self):
        img = _gray(h=40, w=80)
        result = denoise_image(img)
        assert result.shape == (40, 80)

    def test_all_modes_no_crash(self):
        img = _bgr()
        for mode in ("gaussian", "median", "bilateral"):
            result = denoise_image(img, mode=mode)
            assert result.dtype == np.uint8

    def test_none_mode_raises(self):
        img = _bgr()
        with pytest.raises(ValueError):
            denoise_image(img, mode="none")

    def test_kernel_7_gaussian(self):
        img = _bgr()
        result = denoise_image(img, mode="gaussian", kernel_size=7)
        assert result.dtype == np.uint8

    def test_kernel_9_median(self):
        img = _bgr()
        result = denoise_image(img, mode="median", kernel_size=9)
        assert result.dtype == np.uint8

    def test_output_range(self):
        img = _bgr()
        for mode in ("gaussian", "median", "bilateral"):
            result = denoise_image(img, mode=mode)
            assert int(result.min()) >= 0
            assert int(result.max()) <= 255

    def test_gray_bilateral(self):
        img = _gray()
        result = denoise_image(img, mode="bilateral")
        assert result.dtype == np.uint8
        assert result.shape == img.shape


# ─── TestEnhanceContrastExtra ─────────────────────────────────────────────────

class TestEnhanceContrastExtra:
    def test_non_square_bgr(self):
        img = _bgr(h=40, w=80)
        result = enhance_contrast(img)
        assert result.shape == (40, 80, 3)

    def test_non_square_gray(self):
        img = _gray(h=40, w=80)
        result = enhance_contrast(img)
        assert result.shape == (40, 80)

    def test_all_modes_no_crash(self):
        img = _bgr()
        for mode in ("stretch", "clahe"):
            result = enhance_contrast(img, mode=mode)
            assert result.dtype == np.uint8

    def test_none_mode_raises(self):
        img = _bgr()
        with pytest.raises(ValueError):
            enhance_contrast(img, mode="none")

    def test_stretch_gray(self):
        img = _gray()
        result = enhance_contrast(img, mode="stretch")
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_clahe_gray(self):
        img = _gray()
        result = enhance_contrast(img, mode="clahe")
        assert result.dtype == np.uint8

    def test_output_range_stretch(self):
        img = _bgr()
        result = enhance_contrast(img, mode="stretch")
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_output_range_clahe(self):
        img = _bgr()
        result = enhance_contrast(img, mode="clahe")
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_white_image_no_crash(self):
        img = _white_bgr()
        result = enhance_contrast(img, mode="stretch")
        assert result.dtype == np.uint8

    def test_dark_image_no_crash(self):
        img = _dark_bgr()
        result = enhance_contrast(img, mode="stretch")
        assert result.dtype == np.uint8


# ─── TestEnhanceImageExtra ────────────────────────────────────────────────────

class TestEnhanceImageExtra:
    def test_non_square(self):
        img = _bgr(h=40, w=80)
        r = enhance_image(img)
        assert r.image.shape == (40, 80, 3)

    def test_gray_input(self):
        img = _gray()
        r = enhance_image(img)
        assert r.image.shape == img.shape
        assert r.image.dtype == np.uint8

    def test_default_no_ops(self):
        r = enhance_image(_bgr())
        assert r.operations == []

    def test_denoise_median_op(self):
        cfg = EnhanceConfig(denoise="median")
        r = enhance_image(_bgr(), cfg)
        assert any("denoise" in op for op in r.operations)

    def test_denoise_bilateral_op(self):
        cfg = EnhanceConfig(denoise="bilateral")
        r = enhance_image(_bgr(), cfg)
        assert any("denoise" in op for op in r.operations)

    def test_contrast_clahe_op(self):
        cfg = EnhanceConfig(contrast="clahe")
        r = enhance_image(_bgr(), cfg)
        assert any("contrast" in op for op in r.operations)

    def test_contrast_stretch_op(self):
        cfg = EnhanceConfig(contrast="stretch")
        r = enhance_image(_bgr(), cfg)
        assert any("contrast" in op for op in r.operations)

    def test_sharpen_strong_op(self):
        cfg = EnhanceConfig(sharpness="strong")
        r = enhance_image(_bgr(), cfg)
        assert any("sharpen" in op for op in r.operations)

    def test_two_ops_two_entries(self):
        cfg = EnhanceConfig(sharpness="mild", denoise="gaussian")
        r = enhance_image(_bgr(), cfg)
        assert len(r.operations) == 2

    def test_output_range(self):
        cfg = EnhanceConfig(sharpness="strong", denoise="median", contrast="clahe")
        r = enhance_image(_bgr(), cfg)
        assert int(r.image.min()) >= 0
        assert int(r.image.max()) <= 255


# ─── TestBatchEnhanceExtra ────────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_ten_images(self):
        imgs = [_bgr() for _ in range(10)]
        results = batch_enhance(imgs)
        assert len(results) == 10

    def test_mixed_sizes(self):
        imgs = [_bgr(h=30, w=30), _bgr(h=40, w=60), _bgr(h=64, w=32)]
        results = batch_enhance(imgs)
        assert results[0].image.shape == (30, 30, 3)
        assert results[1].image.shape == (40, 60, 3)
        assert results[2].image.shape == (64, 32, 3)

    def test_sharpen_batch(self):
        cfg = EnhanceConfig(sharpness="mild")
        imgs = [_bgr() for _ in range(4)]
        results = batch_enhance(imgs, cfg)
        assert all(any("sharpen" in op for op in r.operations) for r in results)

    def test_denoise_batch(self):
        cfg = EnhanceConfig(denoise="gaussian")
        imgs = [_bgr() for _ in range(3)]
        results = batch_enhance(imgs, cfg)
        assert all(any("denoise" in op for op in r.operations) for r in results)

    def test_all_uint8(self):
        imgs = [_bgr() for _ in range(5)]
        results = batch_enhance(imgs)
        assert all(r.image.dtype == np.uint8 for r in results)

    def test_gray_in_batch(self):
        imgs = [_gray(), _gray(40, 50)]
        results = batch_enhance(imgs)
        assert results[0].image.shape == (32, 32)
        assert results[1].image.shape == (40, 50)

    def test_single_image(self):
        results = batch_enhance([_bgr()])
        assert len(results) == 1
        assert isinstance(results[0], EnhanceResult)
