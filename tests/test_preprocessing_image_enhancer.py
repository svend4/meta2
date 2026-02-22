"""Расширенные тесты для puzzle_reconstruction/preprocessing/image_enhancer.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    return (np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))


def _gray(h: int = 32, w: int = 32) -> np.ndarray:
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _white_bgr(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _dark_bgr(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mixed_bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[:h//2, :, :] = 50
    img[h//2:, :, :] = 200
    return img


# ─── TestEnhanceConfig ────────────────────────────────────────────────────────

class TestEnhanceConfig:
    def test_defaults(self):
        c = EnhanceConfig()
        assert c.sharpness   == "none"
        assert c.denoise     == "none"
        assert c.contrast    == "none"
        assert c.kernel_size == 3

    def test_invalid_sharpness_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(sharpness="super")

    def test_invalid_denoise_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(denoise="nlmeans")

    def test_invalid_contrast_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(contrast="histogram")

    def test_kernel_size_lt3_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=2)

    def test_kernel_size_even_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(kernel_size=4)

    def test_valid_sharpness_modes(self):
        for mode in ("none", "mild", "strong"):
            c = EnhanceConfig(sharpness=mode)
            assert c.sharpness == mode

    def test_valid_denoise_modes(self):
        for mode in ("none", "gaussian", "median", "bilateral"):
            c = EnhanceConfig(denoise=mode)
            assert c.denoise == mode

    def test_valid_contrast_modes(self):
        for mode in ("none", "stretch", "clahe"):
            c = EnhanceConfig(contrast=mode)
            assert c.contrast == mode

    def test_kernel_size_5_ok(self):
        c = EnhanceConfig(kernel_size=5)
        assert c.kernel_size == 5

    def test_kernel_size_7_ok(self):
        c = EnhanceConfig(kernel_size=7)
        assert c.kernel_size == 7

    def test_custom_values(self):
        c = EnhanceConfig(sharpness="mild", denoise="gaussian",
                           contrast="clahe", kernel_size=5)
        assert c.sharpness == "mild"
        assert c.denoise   == "gaussian"
        assert c.contrast  == "clahe"


# ─── TestEnhanceResult ────────────────────────────────────────────────────────

class TestEnhanceResult:
    def _make(self, mean_before=100.0, mean_after=110.0):
        return EnhanceResult(
            image=_bgr(),
            operations=["denoise:gaussian"],
            mean_before=mean_before,
            mean_after=mean_after,
        )

    def test_stores_image(self):
        r = self._make()
        assert isinstance(r.image, np.ndarray)

    def test_stores_operations(self):
        r = self._make()
        assert isinstance(r.operations, list)

    def test_stores_mean_before(self):
        r = self._make(mean_before=50.0)
        assert r.mean_before == pytest.approx(50.0)

    def test_stores_mean_after(self):
        r = self._make(mean_after=60.0)
        assert r.mean_after == pytest.approx(60.0)

    def test_delta_mean(self):
        r = self._make(mean_before=100.0, mean_after=110.0)
        assert r.delta_mean == pytest.approx(10.0)

    def test_delta_mean_negative(self):
        r = self._make(mean_before=110.0, mean_after=90.0)
        assert r.delta_mean == pytest.approx(-20.0)

    def test_mean_before_negative_raises(self):
        with pytest.raises(ValueError):
            EnhanceResult(image=_bgr(), operations=[], mean_before=-1.0, mean_after=0.0)

    def test_mean_after_negative_raises(self):
        with pytest.raises(ValueError):
            EnhanceResult(image=_bgr(), operations=[], mean_before=0.0, mean_after=-1.0)


# ─── TestSharpenImage ─────────────────────────────────────────────────────────

class TestSharpenImage:
    def test_returns_ndarray(self):
        assert isinstance(sharpen_image(_bgr()), np.ndarray)

    def test_uint8_dtype(self):
        assert sharpen_image(_bgr()).dtype == np.uint8

    def test_same_shape_bgr(self):
        img = _bgr(48, 64)
        result = sharpen_image(img)
        assert result.shape == img.shape

    def test_same_shape_gray(self):
        img = _gray(32, 32)
        result = sharpen_image(img)
        assert result.shape == img.shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            sharpen_image(_bgr(), mode="ultra")

    def test_kernel_size_lt3_raises(self):
        with pytest.raises(ValueError):
            sharpen_image(_bgr(), kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            sharpen_image(_bgr(), kernel_size=4)

    def test_mild_mode(self):
        result = sharpen_image(_bgr(), mode="mild")
        assert result.dtype == np.uint8

    def test_strong_mode(self):
        result = sharpen_image(_bgr(), mode="strong")
        assert result.dtype == np.uint8

    def test_values_in_range(self):
        result = sharpen_image(_bgr())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_kernel_size_5(self):
        result = sharpen_image(_bgr(), kernel_size=5)
        assert result.dtype == np.uint8


# ─── TestDenoiseImage ─────────────────────────────────────────────────────────

class TestDenoiseImage:
    def test_returns_ndarray(self):
        assert isinstance(denoise_image(_bgr()), np.ndarray)

    def test_uint8_dtype(self):
        assert denoise_image(_bgr()).dtype == np.uint8

    def test_same_shape(self):
        img = _bgr(48, 48)
        assert denoise_image(img).shape == img.shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            denoise_image(_bgr(), mode="wavelet")

    def test_kernel_lt3_raises(self):
        with pytest.raises(ValueError):
            denoise_image(_bgr(), kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            denoise_image(_bgr(), kernel_size=4)

    def test_gaussian_mode(self):
        result = denoise_image(_bgr(), mode="gaussian")
        assert result.dtype == np.uint8

    def test_median_mode(self):
        result = denoise_image(_bgr(), mode="median")
        assert result.dtype == np.uint8

    def test_bilateral_mode(self):
        result = denoise_image(_bgr(), mode="bilateral")
        assert result.dtype == np.uint8

    def test_values_in_range(self):
        result = denoise_image(_bgr())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_gaussian_smooths(self):
        noisy = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = denoise_image(noisy, mode="gaussian", kernel_size=9)
        # Std should be lower after heavy smoothing
        assert float(result.std()) <= float(noisy.std()) + 10


# ─── TestEnhanceContrast ──────────────────────────────────────────────────────

class TestEnhanceContrast:
    def test_returns_ndarray(self):
        assert isinstance(enhance_contrast(_bgr()), np.ndarray)

    def test_uint8_dtype(self):
        assert enhance_contrast(_bgr()).dtype == np.uint8

    def test_same_shape_bgr(self):
        img = _bgr(32, 32)
        assert enhance_contrast(img, mode="stretch").shape == img.shape

    def test_same_shape_gray(self):
        img = _gray(32, 32)
        assert enhance_contrast(img, mode="stretch").shape == img.shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(_bgr(), mode="equalize")

    def test_kernel_lt3_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(_bgr(), kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(_bgr(), kernel_size=4)

    def test_stretch_mode(self):
        result = enhance_contrast(_mixed_bgr(), mode="stretch")
        assert result.dtype == np.uint8

    def test_clahe_mode_bgr(self):
        result = enhance_contrast(_bgr(), mode="clahe")
        assert result.dtype == np.uint8
        assert result.shape == _bgr().shape

    def test_clahe_mode_gray(self):
        result = enhance_contrast(_gray(), mode="clahe")
        assert result.dtype == np.uint8

    def test_constant_stretch_returns_copy(self):
        img = _white_bgr()
        result = enhance_contrast(img, mode="stretch")
        assert result.dtype == np.uint8

    def test_values_in_range(self):
        result = enhance_contrast(_bgr(), mode="stretch")
        assert result.min() >= 0
        assert result.max() <= 255


# ─── TestEnhanceImage ─────────────────────────────────────────────────────────

class TestEnhanceImage:
    def test_returns_enhance_result(self):
        assert isinstance(enhance_image(_bgr()), EnhanceResult)

    def test_default_no_operations(self):
        result = enhance_image(_bgr())
        assert result.operations == []

    def test_image_uint8(self):
        result = enhance_image(_bgr())
        assert result.image.dtype == np.uint8

    def test_same_shape_default(self):
        img = _bgr(48, 64)
        result = enhance_image(img)
        assert result.image.shape == img.shape

    def test_mean_before_nonneg(self):
        result = enhance_image(_bgr())
        assert result.mean_before >= 0.0

    def test_mean_after_nonneg(self):
        result = enhance_image(_bgr())
        assert result.mean_after >= 0.0

    def test_denoise_cfg_adds_op(self):
        cfg = EnhanceConfig(denoise="gaussian")
        result = enhance_image(_bgr(), cfg)
        assert any("denoise" in op for op in result.operations)

    def test_sharpen_cfg_adds_op(self):
        cfg = EnhanceConfig(sharpness="mild")
        result = enhance_image(_bgr(), cfg)
        assert any("sharpen" in op for op in result.operations)

    def test_contrast_cfg_adds_op(self):
        cfg = EnhanceConfig(contrast="stretch")
        result = enhance_image(_bgr(), cfg)
        assert any("contrast" in op for op in result.operations)

    def test_all_ops_combined(self):
        cfg = EnhanceConfig(sharpness="strong", denoise="median",
                             contrast="clahe", kernel_size=3)
        result = enhance_image(_bgr(), cfg)
        assert len(result.operations) == 3

    def test_grayscale_input(self):
        result = enhance_image(_gray())
        assert isinstance(result, EnhanceResult)


# ─── TestBatchEnhance ─────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_returns_list(self):
        assert isinstance(batch_enhance([_bgr(), _bgr()]), list)

    def test_same_length(self):
        imgs = [_bgr(), _bgr(16, 16), _bgr(48, 48)]
        assert len(batch_enhance(imgs)) == 3

    def test_each_is_enhance_result(self):
        for r in batch_enhance([_bgr(), _bgr()]):
            assert isinstance(r, EnhanceResult)

    def test_empty_list(self):
        assert batch_enhance([]) == []

    def test_cfg_applied_to_all(self):
        cfg = EnhanceConfig(denoise="gaussian")
        results = batch_enhance([_bgr(), _bgr()], cfg)
        for r in results:
            assert any("denoise" in op for op in r.operations)

    def test_images_uint8(self):
        for r in batch_enhance([_bgr(), _gray()]):
            assert r.image.dtype == np.uint8

    def test_preserves_shapes(self):
        imgs = [_bgr(32, 32), _bgr(48, 64)]
        results = batch_enhance(imgs)
        for img, res in zip(imgs, results):
            assert res.image.shape == img.shape
