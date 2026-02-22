"""Тесты для puzzle_reconstruction.preprocessing.orientation."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.orientation import (
    estimate_orientation,
    rotate_to_upright,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h: int = 64, w: int = 64, fill: int = 255) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _rgb(h: int = 64, w: int = 64, fill: int = 200) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _horizontal_lines(h: int = 64, w: int = 64,
                      n_lines: int = 5) -> np.ndarray:
    """Изображение с горизонтальными чёрными линиями на белом фоне."""
    img = np.full((h, w), 255, dtype=np.uint8)
    step = h // (n_lines + 1)
    for i in range(1, n_lines + 1):
        row = i * step
        img[row, :] = 0
    return img


def _noisy(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _mask(h: int = 64, w: int = 64, margin: int = 8) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[margin:h - margin, margin:w - margin] = 255
    return m


# ─── TestEstimateOrientation ──────────────────────────────────────────────────

class TestEstimateOrientation:

    # ── Возвращаемый тип ──────────────────────────────────────────────────────

    def test_returns_float(self):
        angle = estimate_orientation(_horizontal_lines())
        assert isinstance(angle, float)

    # ── Белое/пустое изображение ──────────────────────────────────────────────

    def test_blank_image_returns_float(self):
        angle = estimate_orientation(_blank())
        assert isinstance(angle, float)

    def test_blank_image_angle_finite(self):
        angle = estimate_orientation(_blank())
        assert np.isfinite(angle)

    # ── Горизонтальные линии → угол ≈ 0 ──────────────────────────────────────

    def test_horizontal_lines_near_zero(self):
        img = _horizontal_lines()
        angle = estimate_orientation(img)
        assert abs(angle) < np.pi / 4  # в пределах 45°

    def test_horizontal_lines_within_pi(self):
        img = _horizontal_lines()
        angle = estimate_orientation(img)
        assert -np.pi <= angle <= np.pi

    # ── Маска ─────────────────────────────────────────────────────────────────

    def test_with_mask_returns_float(self):
        img = _horizontal_lines()
        mask = _mask()
        angle = estimate_orientation(img, mask=mask)
        assert isinstance(angle, float)

    def test_with_mask_finite(self):
        angle = estimate_orientation(_horizontal_lines(), mask=_mask())
        assert np.isfinite(angle)

    def test_zero_mask_returns_float(self):
        img = _horizontal_lines()
        zero_mask = np.zeros((64, 64), dtype=np.uint8)
        angle = estimate_orientation(img, mask=zero_mask)
        assert isinstance(angle, float)

    # ── RGB изображение ───────────────────────────────────────────────────────

    def test_rgb_image_ok(self):
        angle = estimate_orientation(_rgb())
        assert isinstance(angle, float)

    def test_rgb_horizontal_lines(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :] = 255
        for row in [10, 20, 30, 40, 50]:
            img[row, :] = 0
        angle = estimate_orientation(img)
        assert np.isfinite(angle)

    # ── Шумное изображение ────────────────────────────────────────────────────

    def test_noisy_image_returns_float(self):
        angle = estimate_orientation(_noisy())
        assert isinstance(angle, float)

    def test_noisy_image_finite(self):
        angle = estimate_orientation(_noisy())
        assert np.isfinite(angle)

    # ── Малые изображения ─────────────────────────────────────────────────────

    def test_small_image_ok(self):
        img = np.full((16, 16), 200, dtype=np.uint8)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_non_square_image(self):
        img = _horizontal_lines(48, 80)
        angle = estimate_orientation(img)
        assert np.isfinite(angle)

    # ── Повторяемость ─────────────────────────────────────────────────────────

    def test_deterministic(self):
        img = _horizontal_lines()
        a1 = estimate_orientation(img)
        a2 = estimate_orientation(img)
        assert a1 == pytest.approx(a2)


# ─── TestRotateToUpright ──────────────────────────────────────────────────────

class TestRotateToUpright:

    # ── Базовые свойства ──────────────────────────────────────────────────────

    def test_returns_ndarray(self):
        img = _rgb()
        rotated = rotate_to_upright(img, 0.0)
        assert isinstance(rotated, np.ndarray)

    def test_shape_preserved(self):
        img = _rgb(64, 64)
        rotated = rotate_to_upright(img, 0.1)
        assert rotated.shape == img.shape

    def test_dtype_uint8(self):
        img = _rgb()
        rotated = rotate_to_upright(img, 0.0)
        assert rotated.dtype == np.uint8

    # ── Нулевой угол → идентичность ───────────────────────────────────────────

    def test_zero_angle_nearly_identical(self):
        img = _rgb()
        rotated = rotate_to_upright(img, 0.0)
        assert np.allclose(rotated.astype(float), img.astype(float), atol=2.0)

    # ── Различные углы ────────────────────────────────────────────────────────

    def test_small_angle_ok(self):
        img = _rgb()
        rotated = rotate_to_upright(img, 0.05)
        assert rotated.shape == img.shape

    def test_quarter_turn_ok(self):
        img = _rgb()
        rotated = rotate_to_upright(img, np.pi / 2)
        assert rotated.shape == img.shape

    def test_negative_angle_ok(self):
        img = _rgb()
        rotated = rotate_to_upright(img, -0.2)
        assert rotated.shape == img.shape

    def test_pi_angle_ok(self):
        img = _rgb()
        rotated = rotate_to_upright(img, np.pi)
        assert rotated.shape == img.shape

    # ── Серое изображение ─────────────────────────────────────────────────────

    def test_gray_image_ok(self):
        img = _blank()
        rotated = rotate_to_upright(img, 0.1)
        assert rotated.shape == img.shape

    def test_gray_dtype_preserved(self):
        img = _blank()
        rotated = rotate_to_upright(img, 0.0)
        assert rotated.dtype == np.uint8

    # ── Нестандартные размеры ─────────────────────────────────────────────────

    def test_non_square_image(self):
        img = np.full((48, 80, 3), 200, dtype=np.uint8)
        rotated = rotate_to_upright(img, 0.1)
        assert rotated.shape == (48, 80, 3)

    def test_small_image_ok(self):
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        rotated = rotate_to_upright(img, 0.2)
        assert rotated.shape == (16, 16, 3)

    # ── Белый фон заполнения ──────────────────────────────────────────────────

    def test_border_fill_white_for_color(self):
        """Угловые пиксели при повороте должны быть белыми."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        rotated = rotate_to_upright(img, np.pi / 6)
        # Угловые пиксели заполняются белым (255, 255, 255)
        corner = rotated[0, 0]
        assert corner.tolist() == [255, 255, 255]

    # ── Полный цикл estimate + rotate ────────────────────────────────────────

    def test_pipeline_horizontal_stays_horizontal(self):
        img = _horizontal_lines()
        angle = estimate_orientation(img)
        rotated = rotate_to_upright(img, angle)
        assert rotated.shape == img.shape
        assert rotated.dtype == np.uint8
