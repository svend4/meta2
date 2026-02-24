"""Тесты для puzzle_reconstruction.preprocessing.segmentation."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.segmentation import (
    segment_fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white_on_black(h: int = 64, w: int = 64, margin: int = 10) -> np.ndarray:
    """Белый прямоугольник на чёрном фоне — простой случай для Otsu."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[margin:h - margin, margin:w - margin] = 200
    return img


def _black_on_white(h: int = 64, w: int = 64, margin: int = 10) -> np.ndarray:
    """Чёрный прямоугольник на белом фоне."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[margin:h - margin, margin:w - margin] = 20
    return img


def _gray_img(h: int = 64, w: int = 64) -> np.ndarray:
    """Серое полутоновое изображение (2D)."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:54, 10:54] = 180
    return img


def _rgb_random(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestSegmentFragment ──────────────────────────────────────────────────────

class TestSegmentFragment:

    # ── Базовые возвращаемые значения ─────────────────────────────────────────

    def test_returns_ndarray(self):
        mask = segment_fragment(_white_on_black())
        assert isinstance(mask, np.ndarray)

    def test_shape_matches_input(self):
        img = _white_on_black(64, 64)
        mask = segment_fragment(img)
        assert mask.shape == (64, 64)

    def test_dtype_uint8(self):
        mask = segment_fragment(_white_on_black())
        assert mask.dtype == np.uint8

    def test_values_binary(self):
        mask = segment_fragment(_white_on_black())
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    # ── Различные методы ──────────────────────────────────────────────────────

    def test_otsu_method_ok(self):
        mask = segment_fragment(_white_on_black(), method="otsu")
        assert mask.dtype == np.uint8

    def test_adaptive_method_ok(self):
        mask = segment_fragment(_white_on_black(), method="adaptive")
        assert mask.shape == (64, 64)

    def test_grabcut_method_ok(self):
        mask = segment_fragment(_white_on_black(), method="grabcut")
        assert mask.dtype == np.uint8

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            segment_fragment(_white_on_black(), method="unknown")

    def test_empty_string_method_raises(self):
        with pytest.raises(ValueError):
            segment_fragment(_white_on_black(), method="")

    # ── Типы входных изображений ──────────────────────────────────────────────

    def test_grayscale_input_ok(self):
        mask = segment_fragment(_gray_img())
        assert mask.shape == (64, 64)

    def test_rgb_input_ok(self):
        mask = segment_fragment(_white_on_black())
        assert mask is not None

    # ── Содержимое маски ──────────────────────────────────────────────────────

    def test_mask_not_empty_for_nonempty_image(self):
        mask = segment_fragment(_white_on_black())
        assert mask.max() == 255

    def test_mask_not_all_foreground(self):
        # Маска не должна покрывать весь кадр
        mask = segment_fragment(_white_on_black())
        assert mask.min() == 0

    def test_fragment_region_detected(self):
        """Центр тёмного прямоугольника (фрагмента) должен быть внутри маски."""
        # segment_fragment uses THRESH_BINARY_INV: dark regions → 255 in mask
        img = _black_on_white(64, 64, margin=12)
        mask = segment_fragment(img, method="otsu")
        # Центральная область должна быть большей частью покрыта
        center_region = mask[20:44, 20:44]
        coverage = (center_region == 255).mean()
        assert coverage > 0.3  # минимальная покрытость

    # ── Параметр morph_kernel ─────────────────────────────────────────────────

    def test_morph_kernel_three_ok(self):
        mask = segment_fragment(_white_on_black(), morph_kernel=3)
        assert mask.shape == (64, 64)

    def test_morph_kernel_five_ok(self):
        mask = segment_fragment(_white_on_black(), morph_kernel=5)
        assert mask.shape == (64, 64)

    def test_morph_kernel_one_ok(self):
        mask = segment_fragment(_white_on_black(), morph_kernel=1)
        assert mask.dtype == np.uint8

    # ── Инвариантность к одинаковому входу ────────────────────────────────────

    def test_deterministic_otsu(self):
        img = _white_on_black()
        mask1 = segment_fragment(img, method="otsu")
        mask2 = segment_fragment(img, method="otsu")
        assert np.array_equal(mask1, mask2)

    def test_deterministic_adaptive(self):
        img = _white_on_black()
        m1 = segment_fragment(img, method="adaptive")
        m2 = segment_fragment(img, method="adaptive")
        assert np.array_equal(m1, m2)

    # ── Размеры ───────────────────────────────────────────────────────────────

    def test_non_square_image(self):
        img = _white_on_black(48, 80, margin=8)
        mask = segment_fragment(img)
        assert mask.shape == (48, 80)

    def test_small_image_ok(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[4:16, 4:16] = 200
        mask = segment_fragment(img)
        assert mask.shape == (20, 20)
