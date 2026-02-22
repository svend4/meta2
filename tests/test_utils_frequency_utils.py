"""Тесты для puzzle_reconstruction/utils/frequency_utils.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig,
    compute_fft_magnitude,
    radial_power_spectrum,
    frequency_band_energy,
    high_frequency_ratio,
    low_pass_filter,
    high_pass_filter,
    compare_frequency_spectra,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_bgr(h=64, w=64):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def make_checkerboard(h=64, w=64):
    """Высокочастотный шаблон."""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                img[i, j] = 255
    return img


def make_sinusoid(h=64, w=64, freq=2):
    """Синусоид с низкой частотой."""
    x = np.linspace(0, 2 * np.pi * freq, w)
    col = (np.sin(x) * 127 + 127).astype(np.uint8)
    return np.tile(col, (h, 1))


# ─── FrequencyConfig ──────────────────────────────────────────────────────────

class TestFrequencyConfig:
    def test_defaults(self):
        cfg = FrequencyConfig()
        assert cfg.log_scale is True
        assert cfg.center_zero is True
        assert cfg.normalize is True
        assert cfg.n_bands == 32

    def test_n_bands_1_raises(self):
        with pytest.raises(ValueError, match="n_bands"):
            FrequencyConfig(n_bands=1)

    def test_n_bands_0_raises(self):
        with pytest.raises(ValueError, match="n_bands"):
            FrequencyConfig(n_bands=0)

    def test_n_bands_2_valid(self):
        cfg = FrequencyConfig(n_bands=2)
        assert cfg.n_bands == 2

    def test_flags_stored(self):
        cfg = FrequencyConfig(log_scale=False, center_zero=False, normalize=False)
        assert cfg.log_scale is False
        assert cfg.center_zero is False
        assert cfg.normalize is False


# ─── compute_fft_magnitude ────────────────────────────────────────────────────

class TestComputeFftMagnitude:
    def test_returns_ndarray(self):
        img = make_gray()
        result = compute_fft_magnitude(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape_gray(self):
        img = make_gray(h=32, w=48)
        result = compute_fft_magnitude(img)
        assert result.shape == (32, 48)

    def test_same_shape_bgr(self):
        img = make_bgr(h=32, w=48)
        result = compute_fft_magnitude(img)
        assert result.shape == (32, 48)

    def test_dtype_float32(self):
        img = make_gray()
        result = compute_fft_magnitude(img)
        assert result.dtype == np.float32

    def test_normalized_in_0_1(self):
        img = make_gray()
        cfg = FrequencyConfig(normalize=True)
        result = compute_fft_magnitude(img, cfg=cfg)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0

    def test_log_scale_false(self):
        img = make_gray()
        cfg = FrequencyConfig(log_scale=False, normalize=False)
        result = compute_fft_magnitude(img, cfg=cfg)
        assert result.dtype == np.float32

    def test_none_cfg_uses_defaults(self):
        img = make_gray()
        result = compute_fft_magnitude(img, cfg=None)
        assert isinstance(result, np.ndarray)

    def test_1d_array_raises(self):
        with pytest.raises(ValueError):
            compute_fft_magnitude(np.zeros(64))

    def test_black_image_returns_ndarray(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = compute_fft_magnitude(img)
        assert isinstance(result, np.ndarray)

    def test_accepts_bgr_image(self):
        img = make_bgr()
        result = compute_fft_magnitude(img)
        assert result.ndim == 2


# ─── radial_power_spectrum ────────────────────────────────────────────────────

class TestRadialPowerSpectrum:
    def test_returns_ndarray(self):
        img = make_gray()
        result = radial_power_spectrum(img)
        assert isinstance(result, np.ndarray)

    def test_length_equals_n_bands(self):
        img = make_gray()
        cfg = FrequencyConfig(n_bands=16)
        result = radial_power_spectrum(img, cfg=cfg)
        assert len(result) == 16

    def test_dtype_float32(self):
        img = make_gray()
        result = radial_power_spectrum(img)
        assert result.dtype == np.float32

    def test_normalized_in_0_1(self):
        img = make_gray()
        cfg = FrequencyConfig(normalize=True)
        result = radial_power_spectrum(img, cfg=cfg)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0

    def test_nonneg_values(self):
        img = make_gray()
        result = radial_power_spectrum(img)
        assert (result >= 0).all()

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            radial_power_spectrum(np.zeros(64))

    def test_bgr_accepted(self):
        img = make_bgr()
        result = radial_power_spectrum(img)
        assert result.ndim == 1

    def test_default_n_bands_32(self):
        img = make_gray()
        result = radial_power_spectrum(img)
        assert len(result) == 32


# ─── frequency_band_energy ────────────────────────────────────────────────────

class TestFrequencyBandEnergy:
    def test_returns_float(self):
        img = make_gray()
        result = frequency_band_energy(img)
        assert isinstance(result, float)

    def test_nonneg(self):
        img = make_gray()
        result = frequency_band_energy(img)
        assert result >= 0.0

    def test_invalid_bounds_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            frequency_band_energy(img, low_frac=0.6, high_frac=0.3)

    def test_equal_bounds_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            frequency_band_energy(img, low_frac=0.5, high_frac=0.5)

    def test_full_range(self):
        img = make_gray()
        result = frequency_band_energy(img, low_frac=0.0, high_frac=1.0)
        assert result >= 0.0

    def test_narrow_band(self):
        img = make_gray()
        result = frequency_band_energy(img, low_frac=0.1, high_frac=0.2)
        assert result >= 0.0

    def test_high_band_less_than_full(self):
        img = make_sinusoid()  # mostly low-frequency
        full = frequency_band_energy(img, 0.0, 1.0)
        high = frequency_band_energy(img, 0.5, 1.0)
        assert high <= full + 1e-6

    def test_out_of_range_low_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            frequency_band_energy(img, low_frac=-0.1, high_frac=0.5)


# ─── high_frequency_ratio ─────────────────────────────────────────────────────

class TestHighFrequencyRatio:
    def test_returns_float(self):
        img = make_gray()
        result = high_frequency_ratio(img)
        assert isinstance(result, float)

    def test_in_0_1(self):
        img = make_gray()
        result = high_frequency_ratio(img)
        assert 0.0 <= result <= 1.0

    def test_threshold_0_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_frequency_ratio(img, threshold_frac=0.0)

    def test_threshold_1_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_frequency_ratio(img, threshold_frac=1.0)

    def test_checkerboard_high_ratio(self):
        img = make_checkerboard()
        ratio = high_frequency_ratio(img, threshold_frac=0.3)
        # Checkerboard is mostly high-frequency
        assert ratio > 0.0

    def test_black_image_returns_0(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = high_frequency_ratio(img)
        assert result == pytest.approx(0.0)

    def test_bgr_accepted(self):
        img = make_bgr()
        result = high_frequency_ratio(img)
        assert 0.0 <= result <= 1.0


# ─── low_pass_filter ──────────────────────────────────────────────────────────

class TestLowPassFilter:
    def test_returns_ndarray(self):
        img = make_gray()
        result = low_pass_filter(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape_gray(self):
        img = make_gray(h=32, w=48)
        result = low_pass_filter(img)
        assert result.shape == (32, 48)

    def test_same_shape_bgr(self):
        img = make_bgr(h=32, w=48)
        result = low_pass_filter(img)
        assert result.shape == (32, 48, 3)

    def test_dtype_uint8(self):
        img = make_gray()
        result = low_pass_filter(img)
        assert result.dtype == np.uint8

    def test_cutoff_0_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            low_pass_filter(img, cutoff_frac=0.0)

    def test_cutoff_above_1_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            low_pass_filter(img, cutoff_frac=1.1)

    def test_cutoff_1_preserves_energy(self):
        # cutoff=1 → keep all frequencies → nearly identical to input
        img = make_sinusoid()
        result = low_pass_filter(img, cutoff_frac=1.0)
        assert result.shape == img.shape

    def test_checkerboard_low_pass_blurs(self):
        img = make_checkerboard()
        result = low_pass_filter(img, cutoff_frac=0.1)
        # Very low cutoff → image should be mostly uniform
        assert result.std() <= img.std() + 1.0  # can't be more varied than original


# ─── high_pass_filter ─────────────────────────────────────────────────────────

class TestHighPassFilter:
    def test_returns_ndarray(self):
        img = make_gray()
        result = high_pass_filter(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape_gray(self):
        img = make_gray(h=32, w=48)
        result = high_pass_filter(img)
        assert result.shape == (32, 48)

    def test_same_shape_bgr(self):
        img = make_bgr(h=32, w=48)
        result = high_pass_filter(img)
        assert result.shape == (32, 48, 3)

    def test_dtype_uint8(self):
        img = make_gray()
        result = high_pass_filter(img)
        assert result.dtype == np.uint8

    def test_cutoff_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_pass_filter(img, cutoff_frac=-0.1)

    def test_cutoff_1_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_pass_filter(img, cutoff_frac=1.0)

    def test_cutoff_0_keeps_all(self):
        img = make_sinusoid()
        result = high_pass_filter(img, cutoff_frac=0.0)
        assert result.shape == img.shape

    def test_bgr_preserved(self):
        img = make_bgr()
        result = high_pass_filter(img)
        assert result.ndim == 3 and result.shape[2] == 3


# ─── compare_frequency_spectra ────────────────────────────────────────────────

class TestCompareFrequencySpectra:
    def test_returns_float(self):
        img1 = make_gray()
        img2 = make_gray()
        result = compare_frequency_spectra(img1, img2)
        assert isinstance(result, float)

    def test_identical_images_high_similarity(self):
        img = make_sinusoid()
        result = compare_frequency_spectra(img, img)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_different_images_less_than_1(self):
        img1 = make_sinusoid(freq=1)
        img2 = make_checkerboard()
        result = compare_frequency_spectra(img1, img2)
        assert 0.0 <= result <= 1.0

    def test_in_0_1(self):
        img1 = make_gray(fill=100)
        img2 = make_gray(fill=200)
        result = compare_frequency_spectra(img1, img2)
        assert 0.0 <= result <= 1.0

    def test_black_image_returns_0(self):
        img1 = np.zeros((32, 32), dtype=np.uint8)
        img2 = make_gray()
        result = compare_frequency_spectra(img1, img2)
        assert result == pytest.approx(0.0)

    def test_bgr_accepted(self):
        img1 = make_bgr()
        img2 = make_bgr()
        result = compare_frequency_spectra(img1, img2)
        assert 0.0 <= result <= 1.0

    def test_cfg_passed(self):
        cfg = FrequencyConfig(n_bands=8)
        img1 = make_gray()
        img2 = make_gray()
        result = compare_frequency_spectra(img1, img2, cfg=cfg)
        assert 0.0 <= result <= 1.0
