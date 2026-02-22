"""
Тесты для puzzle_reconstruction/preprocessing/contrast.py

Покрывает:
    ContrastResult   — improvement, improvement_ratio, repr, поля
    measure_contrast — float, константа≈0, контрастное > константного
    enhance_clahe    — ContrastResult, method, shape, dtype, tile_size,
                       BGR(LAB) и gray-ветки
    enhance_histeq   — ContrastResult, method, shape, dtype, BGR(HSV) и gray
    enhance_gamma    — ContrastResult, method, gamma stored, LUT effect,
                       кэш (одинаковый LUT для одной γ)
    enhance_stretch  — ContrastResult, method, p_low/p_high stored,
                       одноцветный канал не падает, gray и BGR
    enhance_retinex  — ContrastResult, method, sigma stored, shape, dtype
    auto_enhance     — возвращает ContrastResult, очень низкий RMS → clahe,
                       средний → stretch, высокий → gamma
    batch_enhance    — список ContrastResult, пустой список, ValueError,
                       все 6 методов parametrize
"""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_low_contrast():
    """Grayscale изображение с очень низким контрастом (значения 120-135)."""
    rng = np.random.default_rng(0)
    return (rng.random((60, 60)) * 15 + 120).astype(np.uint8)


@pytest.fixture
def gray_high_contrast():
    """Grayscale изображение с высоким контрастом."""
    img = np.zeros((60, 60), dtype=np.uint8)
    img[:30, :] = 20
    img[30:, :] = 220
    return img


@pytest.fixture
def bgr_image():
    rng = np.random.default_rng(1)
    return (rng.random((60, 60, 3)) * 200 + 30).astype(np.uint8)


@pytest.fixture
def constant_gray():
    return np.full((40, 40), 128, dtype=np.uint8)


@pytest.fixture
def gradient_gray():
    """Линейный градиент — умеренный контраст."""
    img = np.zeros((60, 60), dtype=np.uint8)
    for i in range(60):
        img[i, :] = i * 4
    return img


# ─── ContrastResult ───────────────────────────────────────────────────────────

class TestContrastResult:
    def test_improvement_positive(self):
        res = ContrastResult(
            enhanced=np.zeros((10, 10), dtype=np.uint8),
            method="clahe",
            contrast_before=20.0,
            contrast_after=45.0,
        )
        assert res.improvement == pytest.approx(25.0)

    def test_improvement_negative(self):
        res = ContrastResult(
            enhanced=np.zeros((10, 10), dtype=np.uint8),
            method="x",
            contrast_before=50.0,
            contrast_after=30.0,
        )
        assert res.improvement == pytest.approx(-20.0)

    def test_improvement_ratio_positive(self):
        res = ContrastResult(
            enhanced=np.zeros((10, 10), dtype=np.uint8),
            method="x",
            contrast_before=40.0,
            contrast_after=60.0,
        )
        assert res.improvement_ratio == pytest.approx(0.5)

    def test_improvement_ratio_zero_before(self):
        res = ContrastResult(
            enhanced=np.zeros((10, 10), dtype=np.uint8),
            method="x",
            contrast_before=0.0,
            contrast_after=10.0,
        )
        assert res.improvement_ratio == pytest.approx(0.0)

    def test_repr_contains_fields(self):
        res = ContrastResult(
            enhanced=np.zeros((10, 10), dtype=np.uint8),
            method="gamma",
            contrast_before=30.0,
            contrast_after=45.0,
        )
        r = repr(res)
        assert "ContrastResult" in r
        assert "gamma" in r
        assert "→" in r
        assert "Δ=" in r

    def test_default_params(self):
        res = ContrastResult(
            enhanced=np.zeros((5, 5), dtype=np.uint8),
            method="x",
            contrast_before=0.0,
            contrast_after=0.0,
        )
        assert res.params == {}


# ─── measure_contrast ─────────────────────────────────────────────────────────

class TestMeasureContrast:
    def test_returns_float(self, gray_low_contrast):
        val = measure_contrast(gray_low_contrast)
        assert isinstance(val, float)

    def test_nonnegative(self, constant_gray):
        assert measure_contrast(constant_gray) >= 0.0

    def test_constant_is_zero(self, constant_gray):
        assert measure_contrast(constant_gray) == pytest.approx(0.0)

    def test_high_contrast_greater_than_low(self, gray_low_contrast,
                                             gray_high_contrast):
        assert measure_contrast(gray_high_contrast) > measure_contrast(gray_low_contrast)

    def test_bgr_accepted(self, bgr_image):
        val = measure_contrast(bgr_image)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_gradient_intermediate(self, gradient_gray, constant_gray):
        assert measure_contrast(gradient_gray) > measure_contrast(constant_gray)


# ─── enhance_clahe ────────────────────────────────────────────────────────────

class TestEnhanceClahe:
    def test_returns_contrast_result(self, gray_low_contrast):
        res = enhance_clahe(gray_low_contrast)
        assert isinstance(res, ContrastResult)

    def test_method_field(self, gray_low_contrast):
        assert enhance_clahe(gray_low_contrast).method == "clahe"

    def test_gray_shape_preserved(self, gray_low_contrast):
        res = enhance_clahe(gray_low_contrast)
        assert res.enhanced.shape == gray_low_contrast.shape

    def test_gray_dtype_uint8(self, gray_low_contrast):
        assert enhance_clahe(gray_low_contrast).enhanced.dtype == np.uint8

    def test_bgr_shape_preserved(self, bgr_image):
        res = enhance_clahe(bgr_image)
        assert res.enhanced.shape == bgr_image.shape

    def test_bgr_dtype_uint8(self, bgr_image):
        assert enhance_clahe(bgr_image).enhanced.dtype == np.uint8

    def test_clip_limit_stored(self, gray_low_contrast):
        res = enhance_clahe(gray_low_contrast, clip_limit=3.0)
        assert res.params["clip_limit"] == pytest.approx(3.0)

    def test_tile_size_stored(self, gray_low_contrast):
        res = enhance_clahe(gray_low_contrast, tile_size=4)
        assert res.params["tile_size"] == 4

    def test_does_not_modify_input(self, gray_low_contrast):
        orig = gray_low_contrast.copy()
        enhance_clahe(gray_low_contrast)
        np.testing.assert_array_equal(gray_low_contrast, orig)

    def test_low_contrast_improves(self, gray_low_contrast):
        res = enhance_clahe(gray_low_contrast, clip_limit=4.0)
        # Контраст должен вырасти или остаться
        assert res.contrast_after >= res.contrast_before - 1.0


# ─── enhance_histeq ───────────────────────────────────────────────────────────

class TestEnhanceHisteq:
    def test_returns_contrast_result(self, gray_low_contrast):
        res = enhance_histeq(gray_low_contrast)
        assert isinstance(res, ContrastResult)

    def test_method_field(self, gray_low_contrast):
        assert enhance_histeq(gray_low_contrast).method == "histeq"

    def test_gray_shape_preserved(self, gray_low_contrast):
        res = enhance_histeq(gray_low_contrast)
        assert res.enhanced.shape == gray_low_contrast.shape

    def test_gray_dtype_uint8(self, gray_low_contrast):
        assert enhance_histeq(gray_low_contrast).enhanced.dtype == np.uint8

    def test_bgr_shape_preserved(self, bgr_image):
        res = enhance_histeq(bgr_image)
        assert res.enhanced.shape == bgr_image.shape

    def test_bgr_dtype_uint8(self, bgr_image):
        assert enhance_histeq(bgr_image).enhanced.dtype == np.uint8

    def test_low_contrast_improves(self, gray_low_contrast):
        res = enhance_histeq(gray_low_contrast)
        assert res.contrast_after >= res.contrast_before

    def test_does_not_modify_input(self, gray_low_contrast):
        orig = gray_low_contrast.copy()
        enhance_histeq(gray_low_contrast)
        np.testing.assert_array_equal(gray_low_contrast, orig)


# ─── enhance_gamma ────────────────────────────────────────────────────────────

class TestEnhanceGamma:
    def test_returns_contrast_result(self, gradient_gray):
        res = enhance_gamma(gradient_gray)
        assert isinstance(res, ContrastResult)

    def test_method_field(self, gradient_gray):
        assert enhance_gamma(gradient_gray).method == "gamma"

    def test_gamma_stored(self, gradient_gray):
        res = enhance_gamma(gradient_gray, gamma=2.0)
        assert res.params["gamma"] == pytest.approx(2.0)

    def test_shape_preserved_gray(self, gradient_gray):
        assert enhance_gamma(gradient_gray).enhanced.shape == gradient_gray.shape

    def test_dtype_uint8(self, gradient_gray):
        assert enhance_gamma(gradient_gray).enhanced.dtype == np.uint8

    def test_shape_preserved_bgr(self, bgr_image):
        assert enhance_gamma(bgr_image).enhanced.shape == bgr_image.shape

    def test_gamma_one_identity(self, gradient_gray):
        """γ = 1 → LUT ≈ identity → изображение практически не меняется."""
        res = enhance_gamma(gradient_gray, gamma=1.0)
        # Допускаем небольшое отклонение из-за целочисленного округления
        diff = np.abs(res.enhanced.astype(int) - gradient_gray.astype(int))
        assert diff.max() <= 2

    def test_gamma_large_brightens(self, gradient_gray):
        """Большая γ осветляет изображение (output = input^(1/γ))."""
        res = enhance_gamma(gradient_gray, gamma=3.0)
        # Среднее яркости должно вырасти
        assert float(res.enhanced.mean()) >= float(gradient_gray.mean())

    def test_same_gamma_same_lut(self, gradient_gray):
        """Повторный вызов с той же γ использует кэш."""
        r1 = enhance_gamma(gradient_gray, gamma=2.2)
        r2 = enhance_gamma(gradient_gray, gamma=2.2)
        np.testing.assert_array_equal(r1.enhanced, r2.enhanced)


# ─── enhance_stretch ──────────────────────────────────────────────────────────

class TestEnhanceStretch:
    def test_returns_contrast_result(self, gray_low_contrast):
        res = enhance_stretch(gray_low_contrast)
        assert isinstance(res, ContrastResult)

    def test_method_field(self, gray_low_contrast):
        assert enhance_stretch(gray_low_contrast).method == "stretch"

    def test_p_low_p_high_stored(self, gray_low_contrast):
        res = enhance_stretch(gray_low_contrast, p_low=5.0, p_high=95.0)
        assert res.params["p_low"] == pytest.approx(5.0)
        assert res.params["p_high"] == pytest.approx(95.0)

    def test_shape_preserved_gray(self, gray_low_contrast):
        assert enhance_stretch(gray_low_contrast).enhanced.shape == gray_low_contrast.shape

    def test_dtype_uint8(self, gray_low_contrast):
        assert enhance_stretch(gray_low_contrast).enhanced.dtype == np.uint8

    def test_shape_preserved_bgr(self, bgr_image):
        assert enhance_stretch(bgr_image).enhanced.shape == bgr_image.shape

    def test_constant_channel_no_crash(self, constant_gray):
        """Одноцветный канал (hi==lo) не должен вызывать деление на ноль."""
        res = enhance_stretch(constant_gray)
        assert isinstance(res, ContrastResult)

    def test_improves_low_contrast(self, gray_low_contrast):
        res = enhance_stretch(gray_low_contrast)
        assert res.contrast_after >= res.contrast_before

    def test_does_not_modify_input(self, gray_low_contrast):
        orig = gray_low_contrast.copy()
        enhance_stretch(gray_low_contrast)
        np.testing.assert_array_equal(gray_low_contrast, orig)


# ─── enhance_retinex ──────────────────────────────────────────────────────────

class TestEnhanceRetinex:
    def test_returns_contrast_result(self, gradient_gray):
        res = enhance_retinex(gradient_gray)
        assert isinstance(res, ContrastResult)

    def test_method_field(self, gradient_gray):
        assert enhance_retinex(gradient_gray).method == "retinex"

    def test_sigma_stored(self, gradient_gray):
        res = enhance_retinex(gradient_gray, sigma=20.0)
        assert res.params["sigma"] == pytest.approx(20.0)

    def test_gray_shape_preserved(self, gradient_gray):
        assert enhance_retinex(gradient_gray).enhanced.shape == gradient_gray.shape

    def test_gray_dtype_uint8(self, gradient_gray):
        assert enhance_retinex(gradient_gray).enhanced.dtype == np.uint8

    def test_bgr_shape_preserved(self, bgr_image):
        assert enhance_retinex(bgr_image).enhanced.shape == bgr_image.shape

    def test_constant_image_no_crash(self, constant_gray):
        """Константный канал → retinex = 0, не должно падать."""
        res = enhance_retinex(constant_gray)
        assert isinstance(res, ContrastResult)

    def test_output_in_uint8_range(self, gradient_gray):
        res = enhance_retinex(gradient_gray)
        assert res.enhanced.min() >= 0
        assert res.enhanced.max() <= 255


# ─── auto_enhance ─────────────────────────────────────────────────────────────

class TestAutoEnhance:
    def test_returns_contrast_result(self, gradient_gray):
        res = auto_enhance(gradient_gray)
        assert isinstance(res, ContrastResult)

    def test_very_low_contrast_uses_clahe(self):
        """RMS < 20 → CLAHE."""
        img = np.full((60, 60), 128, dtype=np.uint8)
        rng = np.random.default_rng(5)
        img = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255
                      ).astype(np.uint8)
        res = auto_enhance(img)
        assert res.method == "clahe"

    def test_medium_contrast_uses_stretch(self, gray_low_contrast):
        """RMS ~ 4-15 → stretch (если выше нижнего порога)."""
        # Создаём изображение с RMS ≈ 30-55
        rng = np.random.default_rng(6)
        img = np.clip(rng.integers(80, 160, (60, 60)), 0, 255).astype(np.uint8)
        from puzzle_reconstruction.preprocessing.contrast import measure_contrast, _AUTO_LOW, _AUTO_HIGH
        rms = measure_contrast(img)
        res = auto_enhance(img)
        if _AUTO_LOW < rms < _AUTO_HIGH:
            assert res.method == "stretch"

    def test_high_contrast_uses_gamma(self, gray_high_contrast):
        """RMS ≥ 60 → gamma."""
        from puzzle_reconstruction.preprocessing.contrast import measure_contrast, _AUTO_HIGH
        rms = measure_contrast(gray_high_contrast)
        if rms >= _AUTO_HIGH:
            res = auto_enhance(gray_high_contrast)
            assert res.method == "gamma"

    def test_shape_preserved(self, gradient_gray):
        assert auto_enhance(gradient_gray).enhanced.shape == gradient_gray.shape

    def test_dtype_uint8(self, gradient_gray):
        assert auto_enhance(gradient_gray).enhanced.dtype == np.uint8


# ─── batch_enhance ────────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_returns_list(self, gradient_gray):
        results = batch_enhance([gradient_gray, gradient_gray], method="gamma")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_contrast_result(self, gradient_gray):
        results = batch_enhance([gradient_gray], method="clahe")
        assert isinstance(results[0], ContrastResult)

    def test_empty_list(self):
        assert batch_enhance([], method="gamma") == []

    def test_invalid_method_raises(self, gradient_gray):
        with pytest.raises(ValueError, match="Неизвестный"):
            batch_enhance([gradient_gray], method="unknown_xyz")

    @pytest.mark.parametrize("method", [
        "auto", "clahe", "histeq", "gamma", "stretch", "retinex"
    ])
    def test_all_methods_work(self, gradient_gray, method):
        results = batch_enhance([gradient_gray], method=method)
        assert len(results) == 1
        assert isinstance(results[0], ContrastResult)

    def test_kwargs_forwarded(self, gradient_gray):
        results = batch_enhance([gradient_gray], method="gamma", gamma=2.5)
        assert results[0].params["gamma"] == pytest.approx(2.5)

    def test_shapes_preserved(self, gradient_gray, bgr_image):
        results = batch_enhance([gradient_gray, bgr_image], method="histeq")
        assert results[0].enhanced.shape == gradient_gray.shape
        assert results[1].enhanced.shape == bgr_image.shape
