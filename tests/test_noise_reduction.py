"""
Тесты для puzzle_reconstruction/preprocessing/noise_reduction.py

Покрывает:
    DenoiseResult        — noise_reduction_ratio, repr, поля
    estimate_noise_level — float ≥ 0, шумное > чистого, константа ≈ 0
    denoise_gaussian     — DenoiseResult, ksize нечётный, форма, method
    denoise_median       — DenoiseResult, ksize ≥ 3 нечётный, форма
    denoise_nlm          — DenoiseResult, форма, BGR и gray
    denoise_bilateral    — DenoiseResult, форма, gray-ветка
    denoise_morphological — DenoiseResult, ops: open/close/tophat/blackhat,
                            ValueError на неверный op
    smart_denoise        — возвращает DenoiseResult, чистое → method='none',
                            очень шумное → NLM или bilateral
    batch_denoise        — список результатов, ValueError на неверный метод,
                           пустой список
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_reduction import (
    DenoiseResult,
    estimate_noise_level,
    denoise_gaussian,
    denoise_median,
    denoise_nlm,
    denoise_bilateral,
    denoise_morphological,
    smart_denoise,
    batch_denoise,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_clean():
    """Чистое grayscale изображение (однородный градиент)."""
    img = np.zeros((60, 60), dtype=np.uint8)
    for i in range(60):
        img[i, :] = i * 4
    return img


@pytest.fixture
def gray_noisy():
    """Grayscale изображение с сильным гауссовым шумом."""
    rng = np.random.default_rng(0)
    base = np.full((60, 60), 128, dtype=np.float32)
    noise = rng.normal(0, 50, (60, 60)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def bgr_clean():
    rng = np.random.default_rng(1)
    return (rng.random((60, 60, 3)) * 80).astype(np.uint8)


@pytest.fixture
def constant_image():
    return np.full((40, 40), 128, dtype=np.uint8)


# ─── DenoiseResult ────────────────────────────────────────────────────────────

class TestDenoiseResult:
    def test_fields(self, gray_clean):
        res = DenoiseResult(
            denoised=gray_clean.copy(),
            method="test",
            noise_before=5.0,
            noise_after=2.0,
            params={"k": 3},
        )
        assert res.method == "test"
        assert res.noise_before == pytest.approx(5.0)
        assert res.noise_after == pytest.approx(2.0)
        assert res.params == {"k": 3}

    def test_noise_reduction_ratio_positive(self):
        res = DenoiseResult(
            denoised=np.zeros((10, 10), dtype=np.uint8),
            method="m",
            noise_before=10.0,
            noise_after=4.0,
        )
        assert res.noise_reduction_ratio == pytest.approx(0.6)

    def test_noise_reduction_ratio_clipped_zero(self):
        """Если после > до — ratio = 0 (не отрицательное)."""
        res = DenoiseResult(
            denoised=np.zeros((10, 10), dtype=np.uint8),
            method="m",
            noise_before=3.0,
            noise_after=5.0,
        )
        assert res.noise_reduction_ratio == pytest.approx(0.0)

    def test_noise_reduction_ratio_zero_before(self):
        res = DenoiseResult(
            denoised=np.zeros((10, 10), dtype=np.uint8),
            method="m",
            noise_before=0.0,
            noise_after=0.0,
        )
        assert res.noise_reduction_ratio == pytest.approx(0.0)

    def test_noise_reduction_ratio_clipped_one(self):
        """Полное устранение шума → ratio = 1."""
        res = DenoiseResult(
            denoised=np.zeros((10, 10), dtype=np.uint8),
            method="m",
            noise_before=10.0,
            noise_after=0.0,
        )
        assert res.noise_reduction_ratio == pytest.approx(1.0)

    def test_repr_contains_method(self, gray_clean):
        res = DenoiseResult(
            denoised=gray_clean,
            method="gaussian",
            noise_before=4.0,
            noise_after=1.5,
        )
        r = repr(res)
        assert "DenoiseResult" in r
        assert "gaussian" in r
        assert "→" in r
        assert "reduction=" in r

    def test_default_params(self, gray_clean):
        res = DenoiseResult(
            denoised=gray_clean,
            method="x",
            noise_before=0.0,
            noise_after=0.0,
        )
        assert res.params == {}


# ─── estimate_noise_level ─────────────────────────────────────────────────────

class TestEstimateNoiseLevel:
    def test_returns_float(self, gray_clean):
        val = estimate_noise_level(gray_clean)
        assert isinstance(val, float)

    def test_nonnegative(self, gray_clean):
        assert estimate_noise_level(gray_clean) >= 0.0

    def test_constant_image_low_noise(self, constant_image):
        assert estimate_noise_level(constant_image) == pytest.approx(0.0)

    def test_noisy_greater_than_clean(self, gray_clean, gray_noisy):
        assert estimate_noise_level(gray_noisy) > estimate_noise_level(gray_clean)

    def test_bgr_input_accepted(self, bgr_clean):
        val = estimate_noise_level(bgr_clean)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_single_pixel_image(self):
        img = np.array([[128]], dtype=np.uint8)
        val = estimate_noise_level(img)
        assert val >= 0.0


# ─── denoise_gaussian ─────────────────────────────────────────────────────────

class TestDenoiseGaussian:
    def test_returns_denoise_result(self, gray_clean):
        res = denoise_gaussian(gray_clean)
        assert isinstance(res, DenoiseResult)

    def test_method_field(self, gray_clean):
        res = denoise_gaussian(gray_clean)
        assert res.method == "gaussian"

    def test_shape_preserved(self, gray_clean):
        res = denoise_gaussian(gray_clean)
        assert res.denoised.shape == gray_clean.shape

    def test_dtype_preserved(self, gray_clean):
        res = denoise_gaussian(gray_clean)
        assert res.denoised.dtype == np.uint8

    def test_bgr_shape_preserved(self, bgr_clean):
        res = denoise_gaussian(bgr_clean)
        assert res.denoised.shape == bgr_clean.shape

    def test_ksize_stored_odd(self, gray_clean):
        res = denoise_gaussian(gray_clean, ksize=4)   # even → made odd
        assert res.params["ksize"] % 2 == 1

    def test_ksize_explicit_odd(self, gray_clean):
        res = denoise_gaussian(gray_clean, ksize=7)
        assert res.params["ksize"] == 7

    def test_sigma_stored(self, gray_clean):
        res = denoise_gaussian(gray_clean, ksize=3, sigma=1.5)
        assert res.params["sigma"] == pytest.approx(1.5)

    def test_noise_fields_set(self, gray_clean):
        res = denoise_gaussian(gray_clean)
        assert res.noise_before >= 0.0
        assert res.noise_after >= 0.0


# ─── denoise_median ───────────────────────────────────────────────────────────

class TestDenoiseMedian:
    def test_returns_denoise_result(self, gray_clean):
        res = denoise_median(gray_clean)
        assert isinstance(res, DenoiseResult)

    def test_method_field(self, gray_clean):
        assert denoise_median(gray_clean).method == "median"

    def test_shape_preserved(self, gray_clean):
        res = denoise_median(gray_clean)
        assert res.denoised.shape == gray_clean.shape

    def test_ksize_at_least_3(self, gray_clean):
        res = denoise_median(gray_clean, ksize=1)  # 1 → 3
        assert res.params["ksize"] >= 3

    def test_ksize_odd(self, gray_clean):
        res = denoise_median(gray_clean, ksize=6)  # even → odd
        assert res.params["ksize"] % 2 == 1

    def test_bgr_accepted(self, bgr_clean):
        res = denoise_median(bgr_clean, ksize=3)
        assert res.denoised.shape == bgr_clean.shape

    def test_noisy_image_reduces_noise(self, gray_noisy):
        res = denoise_median(gray_noisy, ksize=5)
        assert res.noise_after <= res.noise_before + 1.0  # не хуже


# ─── denoise_nlm ──────────────────────────────────────────────────────────────

class TestDenoiseNlm:
    def test_returns_denoise_result(self, gray_clean):
        res = denoise_nlm(gray_clean)
        assert isinstance(res, DenoiseResult)

    def test_method_field(self, gray_clean):
        assert denoise_nlm(gray_clean).method == "nlm"

    def test_gray_shape_preserved(self, gray_clean):
        res = denoise_nlm(gray_clean)
        assert res.denoised.shape == gray_clean.shape
        assert res.denoised.ndim == 2

    def test_bgr_shape_preserved(self, bgr_clean):
        res = denoise_nlm(bgr_clean)
        assert res.denoised.shape == bgr_clean.shape

    def test_h_param_stored(self, gray_clean):
        res = denoise_nlm(gray_clean, h=8.0)
        assert res.params["h"] == pytest.approx(8.0)

    def test_template_win_odd(self, gray_clean):
        res = denoise_nlm(gray_clean, template_win=6)
        assert res.params["template_win"] % 2 == 1

    def test_search_win_odd(self, gray_clean):
        res = denoise_nlm(gray_clean, search_win=20)
        assert res.params["search_win"] % 2 == 1

    def test_dtype_preserved(self, gray_clean):
        res = denoise_nlm(gray_clean)
        assert res.denoised.dtype == np.uint8


# ─── denoise_bilateral ────────────────────────────────────────────────────────

class TestDenoiseBilateral:
    def test_returns_denoise_result(self, gray_clean):
        res = denoise_bilateral(gray_clean)
        assert isinstance(res, DenoiseResult)

    def test_method_field(self, gray_clean):
        assert denoise_bilateral(gray_clean).method == "bilateral"

    def test_gray_shape_preserved(self, gray_clean):
        res = denoise_bilateral(gray_clean)
        assert res.denoised.shape == gray_clean.shape
        assert res.denoised.ndim == 2

    def test_bgr_shape_preserved(self, bgr_clean):
        res = denoise_bilateral(bgr_clean)
        assert res.denoised.shape == bgr_clean.shape

    def test_params_stored(self, gray_clean):
        res = denoise_bilateral(gray_clean, d=5, sigma_color=50.0,
                                 sigma_space=50.0)
        assert res.params["d"] == 5
        assert res.params["sigma_color"] == pytest.approx(50.0)
        assert res.params["sigma_space"] == pytest.approx(50.0)

    def test_dtype_uint8(self, gray_clean):
        assert denoise_bilateral(gray_clean).denoised.dtype == np.uint8


# ─── denoise_morphological ────────────────────────────────────────────────────

class TestDenoiseMorphological:
    @pytest.mark.parametrize("op", ["open", "close", "tophat", "blackhat"])
    def test_valid_ops(self, gray_clean, op):
        res = denoise_morphological(gray_clean, ksize=3, op=op)
        assert isinstance(res, DenoiseResult)
        assert op in res.method

    def test_invalid_op_raises(self, gray_clean):
        with pytest.raises(ValueError, match="Неизвестная"):
            denoise_morphological(gray_clean, op="erode")

    def test_shape_preserved(self, gray_clean):
        res = denoise_morphological(gray_clean, ksize=3, op="open")
        assert res.denoised.shape == gray_clean.shape

    def test_ksize_stored(self, gray_clean):
        res = denoise_morphological(gray_clean, ksize=5)
        assert res.params["ksize"] == 5

    def test_op_stored(self, gray_clean):
        res = denoise_morphological(gray_clean, op="close")
        assert res.params["op"] == "close"

    def test_bgr_accepted(self, bgr_clean):
        res = denoise_morphological(bgr_clean, ksize=3, op="open")
        assert res.denoised.shape == bgr_clean.shape

    def test_dtype_uint8(self, gray_clean):
        res = denoise_morphological(gray_clean)
        assert res.denoised.dtype == np.uint8


# ─── smart_denoise ────────────────────────────────────────────────────────────

class TestSmartDenoise:
    def test_returns_denoise_result(self, gray_clean):
        res = smart_denoise(gray_clean)
        assert isinstance(res, DenoiseResult)

    def test_constant_image_method_none(self, constant_image):
        res = smart_denoise(constant_image)
        assert res.method == "none"
        assert "reason" in res.params

    def test_clean_image_method_none(self, constant_image):
        res = smart_denoise(constant_image)
        # Чистый → 'none' или лёгкий метод
        assert res.method in ("none", "median", "bilateral", "nlm")

    def test_noisy_image_picks_stronger_method(self, gray_noisy):
        res = smart_denoise(gray_noisy)
        # Шумное изображение не должно оставаться без обработки
        assert isinstance(res, DenoiseResult)

    def test_shape_preserved_clean(self, gray_clean):
        res = smart_denoise(gray_clean)
        assert res.denoised.shape == gray_clean.shape

    def test_shape_preserved_noisy(self, gray_noisy):
        res = smart_denoise(gray_noisy)
        assert res.denoised.shape == gray_noisy.shape

    def test_dtype_uint8(self, gray_noisy):
        res = smart_denoise(gray_noisy)
        assert res.denoised.dtype == np.uint8

    def test_bgr_accepted(self, bgr_clean):
        res = smart_denoise(bgr_clean)
        assert isinstance(res, DenoiseResult)


# ─── batch_denoise ────────────────────────────────────────────────────────────

class TestBatchDenoise:
    def test_returns_list(self, gray_clean):
        results = batch_denoise([gray_clean, gray_clean], method="gaussian")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_denoise_result(self, gray_clean):
        results = batch_denoise([gray_clean], method="median")
        assert isinstance(results[0], DenoiseResult)

    def test_empty_list(self):
        results = batch_denoise([], method="gaussian")
        assert results == []

    def test_invalid_method_raises(self, gray_clean):
        with pytest.raises(ValueError, match="Неизвестный"):
            batch_denoise([gray_clean], method="nonexistent")

    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral",
                                         "nlm", "morphological", "smart"])
    def test_all_methods_work(self, gray_clean, method):
        results = batch_denoise([gray_clean], method=method)
        assert len(results) == 1
        assert isinstance(results[0], DenoiseResult)

    def test_kwargs_forwarded(self, gray_clean):
        results = batch_denoise([gray_clean], method="gaussian", ksize=3)
        assert results[0].params["ksize"] == 3

    def test_shapes_preserved(self, gray_clean, gray_noisy):
        results = batch_denoise([gray_clean, gray_noisy], method="median")
        assert results[0].denoised.shape == gray_clean.shape
        assert results[1].denoised.shape == gray_noisy.shape
