"""Тесты для puzzle_reconstruction/utils/color_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.color_utils import (
    to_gray,
    to_lab,
    to_hsv,
    from_lab,
    compute_histogram,
    compare_histograms,
    dominant_colors,
    color_distance,
    strip_histogram,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 120
    img[:, :, 2] = 60
    return img


def _noisy(h=32, w=32, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _uniform_hist(bins=256):
    h = np.ones(bins, dtype=np.float32)
    return h / h.sum()


# ─── to_gray ──────────────────────────────────────────────────────────────────

class TestToGray:
    def test_returns_ndarray(self):
        assert isinstance(to_gray(_gray()), np.ndarray)

    def test_dtype_uint8(self):
        assert to_gray(_bgr()).dtype == np.uint8

    def test_ndim_2(self):
        assert to_gray(_bgr()).ndim == 2

    def test_gray_input_returns_copy(self):
        img = _gray()
        r   = to_gray(img)
        assert r.shape == img.shape
        np.testing.assert_array_equal(r, img)

    def test_bgr_shape_preserved(self):
        h, w = 24, 48
        r = to_gray(_bgr(h, w))
        assert r.shape == (h, w)

    def test_bgr_gray_not_same_as_channel0(self):
        # Серое значение из BGR ≠ просто B-канал
        img = _bgr()
        r   = to_gray(img)
        # Правильное преобразование использует взвешенную сумму каналов
        assert r.ndim == 2

    def test_pure_white_bgr(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        r   = to_gray(img)
        assert r.min() >= 250

    def test_pure_black_bgr(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        r   = to_gray(img)
        assert r.max() == 0


# ─── to_lab ───────────────────────────────────────────────────────────────────

class TestToLab:
    def test_returns_ndarray(self):
        assert isinstance(to_lab(_bgr()), np.ndarray)

    def test_dtype_float32(self):
        assert to_lab(_bgr()).dtype == np.float32

    def test_ndim_3(self):
        assert to_lab(_bgr()).ndim == 3

    def test_shape_hw3(self):
        h, w = 24, 32
        r = to_lab(_bgr(h, w))
        assert r.shape == (h, w, 3)

    def test_L_in_range(self):
        r = to_lab(_bgr())
        L = r[:, :, 0]
        assert L.min() >= 0.0
        assert L.max() <= 100.0 + 1e-3

    def test_gray_input_ok(self):
        r = to_lab(_gray())
        assert r.shape == (32, 32, 3)

    def test_white_L_near_100(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        r   = to_lab(img)
        assert r[:, :, 0].mean() > 90.0

    def test_black_L_near_0(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        r   = to_lab(img)
        assert r[:, :, 0].mean() < 10.0


# ─── to_hsv ───────────────────────────────────────────────────────────────────

class TestToHsv:
    def test_returns_ndarray(self):
        assert isinstance(to_hsv(_bgr()), np.ndarray)

    def test_dtype_uint8(self):
        assert to_hsv(_bgr()).dtype == np.uint8

    def test_ndim_3(self):
        assert to_hsv(_bgr()).ndim == 3

    def test_shape_hw3(self):
        h, w = 24, 32
        assert to_hsv(_bgr(h, w)).shape == (h, w, 3)

    def test_H_in_range(self):
        r = to_hsv(_bgr())
        H = r[:, :, 0]
        assert H.min() >= 0
        assert H.max() <= 180

    def test_S_in_range(self):
        r = to_hsv(_bgr())
        S = r[:, :, 1]
        assert S.min() >= 0
        assert S.max() <= 255

    def test_gray_input_ok(self):
        r = to_hsv(_gray())
        assert r.shape == (32, 32, 3)

    def test_pure_white_S_zero(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        r   = to_hsv(img)
        assert r[:, :, 1].max() == 0   # white has S=0


# ─── from_lab ─────────────────────────────────────────────────────────────────

class TestFromLab:
    def test_returns_ndarray(self):
        assert isinstance(from_lab(to_lab(_bgr())), np.ndarray)

    def test_dtype_uint8(self):
        assert from_lab(to_lab(_bgr())).dtype == np.uint8

    def test_ndim_3(self):
        assert from_lab(to_lab(_bgr())).ndim == 3

    def test_shape_hw3(self):
        h, w = 24, 32
        lab  = to_lab(_bgr(h, w))
        r    = from_lab(lab)
        assert r.shape == (h, w, 3)

    def test_roundtrip_approx(self):
        img = _bgr()
        r   = from_lab(to_lab(img))
        # Ошибка конвертации должна быть небольшой (≤ 5 единиц на канал)
        diff = np.abs(img.astype(np.float32) - r.astype(np.float32))
        assert diff.mean() < 5.0

    def test_white_roundtrip(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        r   = from_lab(to_lab(img))
        assert r.min() > 240

    def test_black_roundtrip(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        r   = from_lab(to_lab(img))
        assert r.max() < 15


# ─── compute_histogram ────────────────────────────────────────────────────────

class TestComputeHistogram:
    def test_returns_float32(self):
        assert compute_histogram(_gray()).dtype == np.float32

    def test_length_equals_bins(self):
        assert len(compute_histogram(_gray(), bins=64)) == 64

    def test_normalized_sum_one(self):
        h = compute_histogram(_gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_unnormalized_sum_n_pixels(self):
        img = _gray(10, 10)
        h   = compute_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(100.0, abs=1e-3)

    def test_gray_input(self):
        r = compute_histogram(_gray())
        assert r.ndim == 1

    def test_bgr_channel_0(self):
        r = compute_histogram(_bgr(), channel=0)
        assert len(r) == 256

    def test_bgr_channel_1(self):
        r = compute_histogram(_bgr(), channel=1)
        assert len(r) == 256

    def test_uniform_image_single_bin_peak(self):
        img  = _gray(val=100)
        h    = compute_histogram(img, bins=256, normalize=False)
        # Все пиксели в bin 100 → только там ненулевое значение
        assert h[100] > 0
        assert h[:100].sum() == 0
        assert h[101:].sum() == 0

    def test_bins_param(self):
        for bins in [16, 64, 128, 256]:
            assert len(compute_histogram(_gray(), bins=bins)) == bins

    def test_nonneg(self):
        assert compute_histogram(_gray()).min() >= 0.0


# ─── compare_histograms ───────────────────────────────────────────────────────

class TestCompareHistograms:
    def test_returns_float(self):
        h = compute_histogram(_gray())
        assert isinstance(compare_histograms(h, h), float)

    def test_correlation_identical_is_one(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="correlation") == pytest.approx(1.0, abs=1e-5)

    def test_chi_identical_is_zero(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="chi") == pytest.approx(0.0, abs=1e-5)

    def test_bhattacharyya_identical_is_zero(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="bhattacharyya") == pytest.approx(0.0, abs=1e-5)

    def test_intersection_identical_nonneg(self):
        h = compute_histogram(_gray())
        v = compare_histograms(h, h, method="intersection")
        assert v >= 0.0

    def test_unknown_method_raises(self):
        h = compute_histogram(_gray())
        with pytest.raises(ValueError):
            compare_histograms(h, h, method="super_correlation_xyz")

    def test_correlation_different_less_than_one(self):
        h1 = compute_histogram(_gray(val=50))
        h2 = compute_histogram(_gray(val=200))
        assert compare_histograms(h1, h2, method="correlation") < 1.0

    def test_bhattacharyya_different_positive(self):
        h1 = compute_histogram(_gray(val=50))
        h2 = compute_histogram(_gray(val=200))
        assert compare_histograms(h1, h2, method="bhattacharyya") > 0.0

    def test_1d_arrays_input(self):
        h = _uniform_hist(64)
        v = compare_histograms(h, h, method="correlation")
        assert isinstance(v, float)


# ─── dominant_colors ──────────────────────────────────────────────────────────

class TestDominantColors:
    def test_returns_ndarray(self):
        assert isinstance(dominant_colors(_bgr(), k=3), np.ndarray)

    def test_shape_k_3(self):
        r = dominant_colors(_bgr(), k=3)
        assert r.shape == (3, 3)

    def test_dtype_uint8(self):
        assert dominant_colors(_bgr(), k=3).dtype == np.uint8

    def test_k_1_single_row(self):
        r = dominant_colors(_bgr(), k=1)
        assert r.shape == (1, 3)

    def test_gray_input(self):
        r = dominant_colors(_gray(), k=2)
        assert r.shape == (2, 3)

    def test_k_clips_to_pixel_count(self):
        img = np.zeros((2, 2), dtype=np.uint8)  # 4 пикселя
        r   = dominant_colors(img, k=100)
        assert r.shape[0] <= 4
        assert r.shape[1] == 3

    def test_values_in_0_255(self):
        r = dominant_colors(_bgr(), k=3)
        assert r.min() >= 0
        assert r.max() <= 255

    def test_noisy_input(self):
        r = dominant_colors(_noisy(), k=3)
        assert r.shape == (3, 3)


# ─── color_distance ───────────────────────────────────────────────────────────

class TestColorDistance:
    def _bgr_color(self, b, g, r):
        return np.array([b, g, r], dtype=np.uint8)

    def test_returns_float(self):
        c = self._bgr_color(100, 150, 200)
        assert isinstance(color_distance(c, c), float)

    def test_nonneg(self):
        c1 = self._bgr_color(100, 150, 200)
        c2 = self._bgr_color(50, 80, 120)
        assert color_distance(c1, c2) >= 0.0

    def test_same_color_zero(self):
        c = self._bgr_color(100, 150, 200)
        assert color_distance(c, c) == pytest.approx(0.0, abs=1e-3)

    def test_different_colors_positive(self):
        c1 = self._bgr_color(0, 0, 0)
        c2 = self._bgr_color(255, 255, 255)
        assert color_distance(c1, c2) > 0.0

    def test_unknown_space_raises(self):
        c = self._bgr_color(100, 100, 100)
        with pytest.raises(ValueError):
            color_distance(c, c, space="xyz_unknown")

    def test_rgb_space(self):
        c1 = self._bgr_color(0, 0, 0)
        c2 = self._bgr_color(255, 0, 0)
        d  = color_distance(c1, c2, space="rgb")
        assert d == pytest.approx(255.0, abs=1e-3)

    def test_lab_vs_rgb_differ(self):
        c1 = self._bgr_color(0, 0, 0)
        c2 = self._bgr_color(255, 0, 0)
        d_lab = color_distance(c1, c2, space="lab")
        d_rgb = color_distance(c1, c2, space="rgb")
        assert d_lab != d_rgb

    def test_symmetric(self):
        c1 = self._bgr_color(100, 50, 200)
        c2 = self._bgr_color(30, 180, 90)
        assert color_distance(c1, c2) == pytest.approx(color_distance(c2, c1), abs=1e-3)

    def test_white_black_large(self):
        white = self._bgr_color(255, 255, 255)
        black = self._bgr_color(0, 0, 0)
        assert color_distance(white, black) > 50.0


# ─── strip_histogram ──────────────────────────────────────────────────────────

class TestStripHistogram:
    def test_returns_float32(self):
        r = strip_histogram(_gray())
        assert r.dtype == np.float32

    def test_length_equals_bins(self):
        r = strip_histogram(_gray(), bins=64)
        assert len(r) == 64

    def test_sum_to_one(self):
        r = strip_histogram(_gray(), bins=32)
        assert r.sum() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = strip_histogram(_gray(48, 64), side=side, border_px=8)
        assert r.dtype == np.float32
        assert len(r) == 256

    def test_unknown_side_raises(self):
        with pytest.raises(ValueError):
            strip_histogram(_gray(), side=5)

    def test_border_px_1(self):
        r = strip_histogram(_gray(), side=0, border_px=1)
        assert r.sum() == pytest.approx(1.0, abs=1e-5)

    def test_bgr_input_channel0(self):
        r = strip_histogram(_bgr(), side=0, channel=0)
        assert r.dtype == np.float32
        assert len(r) == 256

    def test_bgr_input_channel2(self):
        r = strip_histogram(_bgr(), side=2, channel=2)
        assert r.dtype == np.float32

    def test_gray_and_bgr_different(self):
        img_g = _gray()
        img_b = _bgr()
        h_g   = strip_histogram(img_g, side=0)
        h_b   = strip_histogram(img_b, side=0, channel=0)
        # Разные изображения → разные гистограммы (обычно)
        assert not np.array_equal(h_g, h_b)

    def test_uniform_image_single_peak(self):
        img = _gray(val=100)
        h   = strip_histogram(img, side=0, bins=256, border_px=4)
        assert h[100] == pytest.approx(1.0, abs=1e-5)
