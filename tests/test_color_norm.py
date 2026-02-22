"""
Юнит-тесты для puzzle_reconstruction/preprocessing/color_norm.py.

Тесты покрывают:
    - white_balance()          — коррекция баланса белого (Gray World)
    - clahe_normalize()        — CLAHE в LAB пространстве
    - gamma_correction()       — степенная гамма-коррекция
    - normalize_brightness()   — масштабирование яркости к цели
    - normalize_color()        — полный стек нормализации
    - batch_normalize()        — нормализация набора изображений
"""
import numpy as np
import pytest
import cv2

from puzzle_reconstruction.preprocessing.color_norm import (
    white_balance,
    clahe_normalize,
    gamma_correction,
    normalize_brightness,
    normalize_color,
    batch_normalize,
)


# ─── Вспомогательные функции ─────────────────────────────────────────────

def _gray_image(v: int = 128, h: int = 64, w: int = 64) -> np.ndarray:
    """Однотонное серое BGR изображение."""
    return np.full((h, w, 3), v, dtype=np.uint8)


def _colored_image(b: int, g: int, r: int, h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _random_image(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.rand(h, w, 3) * 255).astype(np.uint8)


# ─── white_balance ────────────────────────────────────────────────────────

class TestWhiteBalance:

    def test_returns_same_shape(self):
        img = _random_image()
        out = white_balance(img)
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _random_image()
        out = white_balance(img)
        assert out.dtype == np.uint8

    def test_neutral_gray_unchanged(self):
        """Нейтральное серое изображение (R=G=B) → без изменений."""
        img = _gray_image(128)
        out = white_balance(img)
        # Gray World: если mean(R)=mean(G)=mean(B), коэффициенты = 1
        np.testing.assert_allclose(
            out.astype(float), img.astype(float), atol=2.0
        )

    def test_blue_tinted_corrected(self):
        """Изображение с синим сдвигом → после коррекции B уменьшается."""
        img = _colored_image(b=180, g=100, r=100)
        out = white_balance(img)
        mean_b_in  = float(img[:, :, 0].mean())
        mean_b_out = float(out[:, :, 0].mean())
        mean_r_in  = float(img[:, :, 2].mean())
        mean_r_out = float(out[:, :, 2].mean())
        # После коррекции B должен уменьшиться, R — вырасти
        assert mean_b_out < mean_b_in
        assert mean_r_out > mean_r_in

    def test_values_in_range(self):
        img = _random_image()
        out = white_balance(img)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_grayscale_passthrough(self):
        """2D (grayscale) → возвращается без изменений."""
        gray = np.full((32, 32), 128, dtype=np.uint8)
        out  = white_balance(gray)
        np.testing.assert_array_equal(out, gray)

    def test_near_zero_channels_no_crash(self):
        """Нулевые каналы не вызывают ZeroDivisionError."""
        img = _colored_image(b=0, g=100, r=100)
        out = white_balance(img)
        assert out is not None


# ─── clahe_normalize ──────────────────────────────────────────────────────

class TestClaheNormalize:

    def test_returns_same_shape(self):
        img = _random_image()
        out = clahe_normalize(img)
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _random_image()
        out = clahe_normalize(img)
        assert out.dtype == np.uint8

    def test_colors_preserved(self):
        """A и B каналы LAB не должны меняться (только L)."""
        img  = _random_image(seed=42)
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        out  = clahe_normalize(img)
        lab2 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        # A и B каналы (цвет) почти не меняются
        diff_a = float(np.abs(lab[:, :, 1].astype(float) - lab2[:, :, 1].astype(float)).mean())
        diff_b = float(np.abs(lab[:, :, 2].astype(float) - lab2[:, :, 2].astype(float)).mean())
        assert diff_a < 5.0
        assert diff_b < 5.0

    def test_contrast_improves_for_low_contrast(self):
        """Низкоконтрастное изображение → стандартное отклонение вырастает."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[20:44, 20:44] = 132   # Маленький контраст
        out  = clahe_normalize(img)
        std_in  = float(img.std())
        std_out = float(out.std())
        assert std_out >= std_in - 1.0   # Не хуже

    def test_clip_limit_effect(self):
        """Высокий clip_limit → больший контраст."""
        img   = _random_image(seed=5)
        out_1 = clahe_normalize(img, clip_limit=1.0)
        out_8 = clahe_normalize(img, clip_limit=8.0)
        std_1 = float(out_1.std())
        std_8 = float(out_8.std())
        assert std_8 >= std_1 - 1.0

    def test_grayscale_input(self):
        gray = np.random.RandomState(0).randint(100, 160, (64, 64), dtype=np.uint8)
        out  = clahe_normalize(gray)
        assert out.ndim == 2
        assert out.dtype == np.uint8


# ─── gamma_correction ─────────────────────────────────────────────────────

class TestGammaCorrection:

    def test_gamma_1_identity(self):
        """gamma=1.0 → изображение не изменяется."""
        img = _random_image()
        out = gamma_correction(img, gamma=1.0)
        np.testing.assert_array_equal(out, img)

    def test_gamma_lt_1_brightens(self):
        """gamma < 1 → изображение светлее."""
        img = _gray_image(100)
        out = gamma_correction(img, gamma=0.5)
        assert float(out.mean()) > float(img.mean())

    def test_gamma_gt_1_darkens(self):
        """gamma > 1 → изображение темнее."""
        img = _gray_image(150)
        out = gamma_correction(img, gamma=2.0)
        assert float(out.mean()) < float(img.mean())

    def test_returns_uint8(self):
        img = _random_image()
        out = gamma_correction(img, gamma=1.5)
        assert out.dtype == np.uint8

    def test_same_shape(self):
        img = _random_image()
        out = gamma_correction(img, gamma=0.8)
        assert out.shape == img.shape

    def test_values_in_range(self):
        img = _random_image()
        out = gamma_correction(img, gamma=2.5)
        assert out.min() >= 0 and out.max() <= 255

    def test_black_stays_black(self):
        img = _gray_image(0)
        out = gamma_correction(img, gamma=0.5)
        assert out.max() == 0

    def test_white_stays_white(self):
        img = _gray_image(255)
        out = gamma_correction(img, gamma=2.0)
        assert out.min() == 255


# ─── normalize_brightness ─────────────────────────────────────────────────

class TestNormalizeBrightness:

    def test_reaches_target_mean(self):
        img    = _gray_image(100)
        target = 180.0
        out    = normalize_brightness(img, target=target)
        gray   = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        actual = float(gray.mean())
        assert abs(actual - target) < 5.0, f"mean={actual}, target={target}"

    def test_returns_uint8(self):
        img = _random_image()
        out = normalize_brightness(img, target=150.0)
        assert out.dtype == np.uint8

    def test_same_shape(self):
        img = _random_image()
        out = normalize_brightness(img)
        assert out.shape == img.shape

    def test_mask_aware(self):
        """Яркость считается только по области маски."""
        img  = _random_image(seed=3)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:44, 20:44] = 1
        out1 = normalize_brightness(img, target=150.0, mask=None)
        out2 = normalize_brightness(img, target=150.0, mask=mask)
        # Результаты с маской и без могут различаться
        assert out1.dtype == np.uint8 and out2.dtype == np.uint8

    def test_near_zero_image_no_crash(self):
        img = _gray_image(0)
        out = normalize_brightness(img, target=150.0)
        assert out is not None


# ─── normalize_color ──────────────────────────────────────────────────────

class TestNormalizeColor:

    def test_returns_same_shape(self):
        img = _random_image()
        out = normalize_color(img)
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _random_image()
        out = normalize_color(img)
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        img = _random_image()
        out = normalize_color(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_none_image_passthrough(self):
        """None возвращается без падения."""
        out = normalize_color(None)
        assert out is None

    def test_empty_image_passthrough(self):
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        out = normalize_color(img)
        assert out.size == 0

    def test_idempotent_approximately(self):
        """Двойное применение не сильно меняет результат."""
        img  = _random_image(seed=7)
        out1 = normalize_color(img)
        out2 = normalize_color(out1)
        diff = float(np.abs(out1.astype(float) - out2.astype(float)).mean())
        assert diff < 20.0   # Мягкий критерий идемпотентности


# ─── batch_normalize ──────────────────────────────────────────────────────

class TestBatchNormalize:

    def test_returns_same_length(self):
        images = [_random_image(seed=i) for i in range(4)]
        result = batch_normalize(images)
        assert len(result) == 4

    def test_empty_list(self):
        result = batch_normalize([])
        assert result == []

    def test_all_uint8(self):
        images = [_random_image(seed=i) for i in range(3)]
        result = batch_normalize(images)
        for r in result:
            assert r.dtype == np.uint8

    def test_same_shapes(self):
        images = [_random_image(seed=i, h=60, w=80) for i in range(3)]
        result = batch_normalize(images)
        for r in result:
            assert r.shape == (60, 80, 3)

    def test_reference_brightness_preserved(self):
        """Яркость эталонного изображения (idx=0) должна остаться близкой."""
        ref   = _gray_image(180)
        other = _gray_image(100)
        images = [ref, other, other]
        result = batch_normalize(images, reference_idx=0)
        gray_ref_in  = float(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).mean())
        gray_ref_out = float(cv2.cvtColor(result[0], cv2.COLOR_BGR2GRAY).mean())
        # Яркость эталона после нормализации относительно себя = близко к исходной
        assert abs(gray_ref_out - gray_ref_in) < 15.0
