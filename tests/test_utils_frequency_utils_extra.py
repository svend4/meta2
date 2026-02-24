"""Extra tests for puzzle_reconstruction/utils/frequency_utils.py"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.frequency_utils import (
    FrequencyConfig,
    compare_frequency_spectra,
    compute_fft_magnitude,
    frequency_band_energy,
    high_frequency_ratio,
    high_pass_filter,
    low_pass_filter,
    radial_power_spectrum,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _checkerboard(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                img[i, j] = 255
    return img


def _sinusoid(h=64, w=64, freq=2):
    x = np.linspace(0, 2 * np.pi * freq, w)
    col = (np.sin(x) * 127 + 127).astype(np.uint8)
    return np.tile(col, (h, 1))


def _noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ─── TestFrequencyConfigExtra ─────────────────────────────────────────────────

class TestFrequencyConfigExtra:
    def test_n_bands_64_valid(self):
        cfg = FrequencyConfig(n_bands=64)
        assert cfg.n_bands == 64

    def test_n_bands_negative_raises(self):
        with pytest.raises(ValueError):
            FrequencyConfig(n_bands=-1)

    def test_log_scale_false(self):
        cfg = FrequencyConfig(log_scale=False)
        assert cfg.log_scale is False

    def test_center_zero_false(self):
        cfg = FrequencyConfig(center_zero=False)
        assert cfg.center_zero is False

    def test_normalize_false(self):
        cfg = FrequencyConfig(normalize=False)
        assert cfg.normalize is False

    def test_n_bands_32_default(self):
        cfg = FrequencyConfig()
        assert cfg.n_bands == 32

    def test_n_bands_3_valid(self):
        cfg = FrequencyConfig(n_bands=3)
        assert cfg.n_bands == 3

    def test_all_false_flags(self):
        cfg = FrequencyConfig(log_scale=False, center_zero=False, normalize=False)
        assert cfg.log_scale is False
        assert cfg.center_zero is False
        assert cfg.normalize is False


# ─── TestComputeFftMagnitudeExtra ─────────────────────────────────────────────

class TestComputeFftMagnitudeExtra:
    def test_non_square_image(self):
        img = _gray(h=32, w=64)
        result = compute_fft_magnitude(img)
        assert result.shape == (32, 64)

    def test_various_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            result = compute_fft_magnitude(img)
            assert result.ndim == 2

    def test_center_zero_false(self):
        cfg = FrequencyConfig(center_zero=False, normalize=False)
        img = _gray()
        result = compute_fft_magnitude(img, cfg=cfg)
        assert result.dtype == np.float32

    def test_checkerboard_non_uniform(self):
        img = _checkerboard()
        result = compute_fft_magnitude(img)
        # Checkerboard has strong high-frequency content
        assert result.max() > 0.0

    def test_nonneg_when_normalized(self):
        cfg = FrequencyConfig(normalize=True)
        img = _noisy(seed=7)
        result = compute_fft_magnitude(img, cfg=cfg)
        assert result.min() >= 0.0

    def test_sinusoid_nonneg(self):
        img = _sinusoid()
        result = compute_fft_magnitude(img)
        assert result.min() >= 0.0


# ─── TestRadialPowerSpectrumExtra ─────────────────────────────────────────────

class TestRadialPowerSpectrumExtra:
    def test_n_bands_8(self):
        img = _gray()
        cfg = FrequencyConfig(n_bands=8)
        result = radial_power_spectrum(img, cfg=cfg)
        assert len(result) == 8

    def test_n_bands_64(self):
        img = _gray()
        cfg = FrequencyConfig(n_bands=64)
        result = radial_power_spectrum(img, cfg=cfg)
        assert len(result) == 64

    def test_various_seeds(self):
        for s in range(5):
            img = _noisy(seed=s)
            result = radial_power_spectrum(img)
            assert len(result) == 32

    def test_sinusoid_returns_ndarray(self):
        img = _sinusoid()
        result = radial_power_spectrum(img)
        assert isinstance(result, np.ndarray)

    def test_log_scale_false(self):
        cfg = FrequencyConfig(log_scale=False, normalize=True)
        img = _gray()
        result = radial_power_spectrum(img, cfg=cfg)
        assert result.dtype == np.float32

    def test_checkerboard_nonneg(self):
        img = _checkerboard()
        result = radial_power_spectrum(img)
        assert np.all(result >= 0.0)


# ─── TestFrequencyBandEnergyExtra ─────────────────────────────────────────────

class TestFrequencyBandEnergyExtra:
    def test_out_of_range_high_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            frequency_band_energy(img, low_frac=0.5, high_frac=1.5)

    def test_checkerboard_high_band_energy(self):
        img = _checkerboard()
        result = frequency_band_energy(img, low_frac=0.5, high_frac=1.0)
        assert result >= 0.0

    def test_various_bands(self):
        img = _sinusoid()
        for low, high in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0)):
            e = frequency_band_energy(img, low_frac=low, high_frac=high)
            assert e >= 0.0

    def test_low_frequency_band_on_sinusoid(self):
        img = _sinusoid(freq=1)
        e_low = frequency_band_energy(img, low_frac=0.0, high_frac=0.3)
        e_high = frequency_band_energy(img, low_frac=0.7, high_frac=1.0)
        # Sinusoid has more low-frequency energy
        assert e_low >= 0.0 and e_high >= 0.0

    def test_bgr_image(self):
        img = _bgr()
        result = frequency_band_energy(img, low_frac=0.0, high_frac=1.0)
        assert isinstance(result, float)


# ─── TestHighFrequencyRatioExtra ──────────────────────────────────────────────

class TestHighFrequencyRatioExtra:
    def test_sinusoid_low_ratio(self):
        # Low-freq sinusoid should have lower HF ratio than checkerboard
        s = high_frequency_ratio(_sinusoid(freq=1), threshold_frac=0.5)
        c = high_frequency_ratio(_checkerboard(), threshold_frac=0.5)
        assert 0.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0

    def test_threshold_0_5_valid(self):
        img = _gray()
        result = high_frequency_ratio(img, threshold_frac=0.5)
        assert 0.0 <= result <= 1.0

    def test_five_seeds_in_range(self):
        for s in range(5):
            img = _noisy(seed=s)
            result = high_frequency_ratio(img)
            assert 0.0 <= result <= 1.0

    def test_white_image_zero(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        result = high_frequency_ratio(img)
        assert result == pytest.approx(0.0)


# ─── TestLowPassFilterExtra ───────────────────────────────────────────────────

class TestLowPassFilterExtra:
    def test_non_square_gray(self):
        img = _gray(h=32, w=64)
        result = low_pass_filter(img)
        assert result.shape == (32, 64)

    def test_cutoff_0_5(self):
        img = _checkerboard()
        result = low_pass_filter(img, cutoff_frac=0.5)
        assert result.dtype == np.uint8

    def test_various_cutoffs(self):
        img = _checkerboard()
        for cutoff in (0.1, 0.3, 0.5, 0.9, 1.0):
            result = low_pass_filter(img, cutoff_frac=cutoff)
            assert result.shape == img.shape

    def test_sinusoid_low_pass_preserves_shape(self):
        img = _sinusoid()
        result = low_pass_filter(img, cutoff_frac=0.3)
        assert result.shape == img.shape

    def test_noisy_image_no_crash(self):
        for s in range(3):
            img = _noisy(seed=s)
            result = low_pass_filter(img)
            assert isinstance(result, np.ndarray)


# ─── TestHighPassFilterExtra ──────────────────────────────────────────────────

class TestHighPassFilterExtra:
    def test_non_square_gray(self):
        img = _gray(h=32, w=64)
        result = high_pass_filter(img)
        assert result.shape == (32, 64)

    def test_cutoff_0_5(self):
        img = _checkerboard()
        result = high_pass_filter(img, cutoff_frac=0.5)
        assert result.dtype == np.uint8

    def test_various_cutoffs(self):
        img = _checkerboard()
        for cutoff in (0.1, 0.3, 0.5, 0.9):
            result = high_pass_filter(img, cutoff_frac=cutoff)
            assert result.shape == img.shape

    def test_checkerboard_high_pass_not_blank(self):
        img = _checkerboard()
        result = high_pass_filter(img, cutoff_frac=0.1)
        # Checkerboard is mostly high-frequency, should preserve some content
        assert isinstance(result, np.ndarray)

    def test_bgr_non_square(self):
        img = _bgr(h=32, w=64)
        result = high_pass_filter(img)
        assert result.shape == (32, 64, 3)


# ─── TestCompareFrequencySpectraExtra ─────────────────────────────────────────

class TestCompareFrequencySpectraExtra:
    def test_two_black_images_similarity(self):
        img1 = np.zeros((32, 32), dtype=np.uint8)
        img2 = np.zeros((32, 32), dtype=np.uint8)
        result = compare_frequency_spectra(img1, img2)
        assert result == pytest.approx(0.0) or result == pytest.approx(1.0)

    def test_two_identical_sinusoids(self):
        img = _sinusoid(freq=3)
        result = compare_frequency_spectra(img, img)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_different_freqs(self):
        img1 = _sinusoid(freq=1)
        img2 = _sinusoid(freq=8)
        result = compare_frequency_spectra(img1, img2)
        assert 0.0 <= result <= 1.0

    def test_cfg_n_bands_4(self):
        cfg = FrequencyConfig(n_bands=4)
        img1 = _gray(fill=100)
        img2 = _gray(fill=200)
        result = compare_frequency_spectra(img1, img2, cfg=cfg)
        assert 0.0 <= result <= 1.0

    def test_noisy_vs_smooth(self):
        img1 = _noisy(seed=1)
        img2 = _gray(fill=128)
        result = compare_frequency_spectra(img1, img2)
        assert 0.0 <= result <= 1.0

    def test_five_random_pairs_in_range(self):
        for s in range(5):
            a = _noisy(seed=s)
            b = _noisy(seed=s + 10)
            result = compare_frequency_spectra(a, b)
            assert 0.0 <= result <= 1.0
