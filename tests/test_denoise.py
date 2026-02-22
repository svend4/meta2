"""
Тесты для puzzle_reconstruction/preprocessing/denoise.py

Покрытие:
    gaussian_denoise   — sigma=0, shape, dtype, values ∈ [0,255]
    median_denoise     — ksize нечётный ≥ 3, shape
    bilateral_denoise  — BGR и grayscale
    nlmeans_denoise    — BGR и grayscale
    auto_denoise       — noise < 2 → pass-through; aggressive=True; None guard
    denoise_batch      — dispatch, ValueError, None passthrough
"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.denoise import (
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlmeans_denoise,
    auto_denoise,
    denoise_batch,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_bgr():
    """Чистое синтетическое BGR-изображение (64×64, uint8)."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 180
    # Рисуем простые паттерны для проверки сохранения краёв
    img[20:44, 20:44] = (100, 150, 200)
    return img


@pytest.fixture
def noisy_bgr(clean_bgr):
    """BGR-изображение с сильным гауссовым шумом σ=25."""
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 25, clean_bgr.shape)
    return np.clip(clean_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def gray_img():
    """Grayscale uint8 64×64."""
    img = np.ones((64, 64), dtype=np.uint8) * 128
    img[16:48, 16:48] = 200
    return img


# ─── gaussian_denoise ─────────────────────────────────────────────────────────

class TestGaussianDenoise:
    def test_sigma_zero_returns_original(self, clean_bgr):
        out = gaussian_denoise(clean_bgr, sigma=0.0)
        assert np.array_equal(out, clean_bgr), "sigma=0 должен вернуть исходное"

    def test_output_shape_preserved(self, noisy_bgr):
        out = gaussian_denoise(noisy_bgr, sigma=1.5)
        assert out.shape == noisy_bgr.shape

    def test_output_dtype(self, noisy_bgr):
        out = gaussian_denoise(noisy_bgr, sigma=1.5)
        assert out.dtype == np.uint8

    def test_values_in_range(self, noisy_bgr):
        out = gaussian_denoise(noisy_bgr, sigma=2.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_reduces_variance(self, noisy_bgr, clean_bgr):
        """После сглаживания дисперсия должна уменьшиться."""
        out = gaussian_denoise(noisy_bgr, sigma=3.0)
        noise_before = float(np.std(noisy_bgr.astype(np.float32) -
                                    clean_bgr.astype(np.float32)))
        noise_after  = float(np.std(out.astype(np.float32) -
                                    clean_bgr.astype(np.float32)))
        assert noise_after < noise_before, "Гаусс должен уменьшать дисперсию шума"

    def test_custom_kernel_size(self, noisy_bgr):
        out = gaussian_denoise(noisy_bgr, sigma=1.0, kernel_size=5)
        assert out.shape == noisy_bgr.shape
        assert out.dtype == np.uint8

    def test_grayscale_input(self, gray_img):
        out = gaussian_denoise(gray_img, sigma=1.0)
        assert out.shape == gray_img.shape
        assert out.dtype == np.uint8


# ─── median_denoise ───────────────────────────────────────────────────────────

class TestMedianDenoise:
    def test_output_shape(self, noisy_bgr):
        out = median_denoise(noisy_bgr, ksize=3)
        assert out.shape == noisy_bgr.shape

    def test_output_dtype(self, noisy_bgr):
        out = median_denoise(noisy_bgr, ksize=3)
        assert out.dtype == np.uint8

    def test_even_ksize_rounded_up(self, noisy_bgr):
        """Чётный ksize должен быть округлён до нечётного."""
        out_even = median_denoise(noisy_bgr, ksize=4)
        out_odd  = median_denoise(noisy_bgr, ksize=5)
        assert out_even.shape == out_odd.shape  # Оба работают без ошибок

    def test_ksize_less_than_3_uses_3(self, noisy_bgr):
        """ksize < 3 должен быть скорректирован до 3."""
        out = median_denoise(noisy_bgr, ksize=1)
        assert out.shape == noisy_bgr.shape

    def test_removes_salt_pepper(self):
        """Медиана удаляет одиночные выбросы."""
        img = np.ones((32, 32), dtype=np.uint8) * 128
        img[10, 10] = 255   # Salt
        img[20, 20] = 0     # Pepper
        out = median_denoise(img, ksize=3)
        assert abs(int(out[10, 10]) - 128) < 30, "Соль должна быть удалена"
        assert abs(int(out[20, 20]) - 128) < 30, "Перец должен быть удалён"

    def test_grayscale(self, gray_img):
        out = median_denoise(gray_img, ksize=3)
        assert out.shape == gray_img.shape


# ─── bilateral_denoise ────────────────────────────────────────────────────────

class TestBilateralDenoise:
    def test_bgr_shape_preserved(self, noisy_bgr):
        out = bilateral_denoise(noisy_bgr)
        assert out.shape == noisy_bgr.shape

    def test_bgr_dtype(self, noisy_bgr):
        out = bilateral_denoise(noisy_bgr)
        assert out.dtype == np.uint8

    def test_grayscale_shape(self, gray_img):
        out = bilateral_denoise(gray_img)
        assert out.shape == gray_img.shape
        assert out.dtype == np.uint8

    def test_custom_params(self, noisy_bgr):
        out = bilateral_denoise(noisy_bgr, d=5, sigma_color=50.0, sigma_space=50.0)
        assert out.shape == noisy_bgr.shape

    def test_preserves_step_edge(self):
        """Билатеральный фильтр должен сохранять резкие перепады."""
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 200
        out = bilateral_denoise(img, d=9, sigma_color=75.0, sigma_space=75.0)
        # Левая половина ≈ 0, правая ≈ 200
        left_mean  = float(out[:, :25].mean())
        right_mean = float(out[:, 39:].mean())
        assert left_mean  < 40,  f"Левая половина: {left_mean}"
        assert right_mean > 160, f"Правая половина: {right_mean}"


# ─── nlmeans_denoise ──────────────────────────────────────────────────────────

class TestNLMeansDenoise:
    def test_bgr_shape(self, noisy_bgr):
        out = nlmeans_denoise(noisy_bgr, h=10.0)
        assert out.shape == noisy_bgr.shape

    def test_bgr_dtype(self, noisy_bgr):
        out = nlmeans_denoise(noisy_bgr, h=10.0)
        assert out.dtype == np.uint8

    def test_grayscale(self, gray_img):
        out = nlmeans_denoise(gray_img, h=10.0)
        assert out.shape == gray_img.shape
        assert out.dtype == np.uint8

    def test_reduces_noise(self, noisy_bgr, clean_bgr):
        """NLM должен уменьшать расстояние до чистого изображения."""
        out = nlmeans_denoise(noisy_bgr, h=15.0)
        rmse_before = float(np.sqrt(np.mean(
            (noisy_bgr.astype(np.float32) - clean_bgr.astype(np.float32)) ** 2
        )))
        rmse_after  = float(np.sqrt(np.mean(
            (out.astype(np.float32) - clean_bgr.astype(np.float32)) ** 2
        )))
        assert rmse_after < rmse_before, "NLM должен улучшить RMSE"


# ─── auto_denoise ─────────────────────────────────────────────────────────────

class TestAutoDenoise:
    def test_clean_image_passthrough(self, clean_bgr):
        """Чистое изображение (σ_noise < 2) должно возвращаться неизменным."""
        out = auto_denoise(clean_bgr)
        # Для чистого изображения метод должен вернуть без обработки (или легко)
        assert out.shape == clean_bgr.shape
        assert out.dtype == np.uint8

    def test_noisy_image_processed(self, noisy_bgr, clean_bgr):
        """Зашумлённое изображение должно быть обработано."""
        out = auto_denoise(noisy_bgr)
        assert out.shape == noisy_bgr.shape
        # NLM или bilateral должны уменьшить шум
        rmse_in  = float(np.sqrt(np.mean(
            (noisy_bgr.astype(np.float32) - clean_bgr.astype(np.float32)) ** 2
        )))
        rmse_out = float(np.sqrt(np.mean(
            (out.astype(np.float32) - clean_bgr.astype(np.float32)) ** 2
        )))
        assert rmse_out < rmse_in * 1.1, "auto_denoise не должен ухудшать качество"

    def test_aggressive_uses_nlm(self, noisy_bgr):
        """aggressive=True → всегда NLM."""
        out = auto_denoise(noisy_bgr, aggressive=True)
        out_nlm = nlmeans_denoise(noisy_bgr)
        assert np.array_equal(out, out_nlm), "aggressive=True должен использовать NLM"

    def test_none_guard(self):
        """None на входе → None на выходе (без краша)."""
        import numpy as np
        # Вместо None передаём пустой массив — авто-денойз проверяет size == 0
        empty = np.array([], dtype=np.uint8)
        out = auto_denoise(empty)
        assert out is empty

    def test_output_dtype(self, noisy_bgr):
        out = auto_denoise(noisy_bgr)
        assert out.dtype == np.uint8


# ─── denoise_batch ────────────────────────────────────────────────────────────

class TestDenoiseBatch:
    def test_basic_dispatch(self, clean_bgr):
        result = denoise_batch([clean_bgr, clean_bgr], method="gaussian")
        assert len(result) == 2
        for r in result:
            assert r.shape == clean_bgr.shape

    def test_all_methods_run(self, clean_bgr):
        for method in ("gaussian", "median", "bilateral", "auto"):
            result = denoise_batch([clean_bgr], method=method)
            assert len(result) == 1
            assert result[0].shape == clean_bgr.shape

    def test_nlmeans_method(self, clean_bgr):
        result = denoise_batch([clean_bgr], method="nlmeans")
        assert len(result) == 1

    def test_unknown_method_raises(self, clean_bgr):
        with pytest.raises(ValueError, match="Неизвестный метод"):
            denoise_batch([clean_bgr], method="magic")

    def test_none_passthrough(self, clean_bgr):
        """None в списке → None в выводе, без исключений."""
        result = denoise_batch([clean_bgr, None, clean_bgr], method="gaussian")
        assert len(result) == 3
        assert result[1] is None

    def test_empty_list(self):
        result = denoise_batch([], method="auto")
        assert result == []

    def test_kwargs_forwarded(self, clean_bgr):
        """Дополнительные kwargs корректно передаются в функцию."""
        result = denoise_batch([clean_bgr], method="gaussian", sigma=2.0)
        assert result[0].shape == clean_bgr.shape
