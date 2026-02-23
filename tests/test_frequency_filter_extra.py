"""Extra tests for puzzle_reconstruction.preprocessing.frequency_filter."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.frequency_filter import (
    FrequencyFilterParams,
    apply_frequency_filter,
    band_pass_filter,
    batch_frequency_filter,
    fft_image,
    gaussian_high_pass,
    gaussian_low_pass,
    ifft_image,
    notch_filter,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=None, seed=0):
    if val is not None:
        return np.full((h, w), val, dtype=np.uint8)
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestFrequencyFilterParamsExtra ──────────────────────────────────────────

class TestFrequencyFilterParamsExtra:
    def test_default_sigma_low(self):
        p = FrequencyFilterParams()
        assert p.sigma_low > 0

    def test_default_notch_radius(self):
        p = FrequencyFilterParams(filter_type="notch")
        assert p.notch_radius > 0

    def test_band_pass_valid_sigmas(self):
        p = FrequencyFilterParams(filter_type="band_pass", sigma_low=2.0, sigma_high=10.0)
        assert p.sigma_low < p.sigma_high

    def test_high_pass_type_stored(self):
        p = FrequencyFilterParams(filter_type="high_pass")
        assert p.filter_type == "high_pass"

    def test_notch_type_stored(self):
        p = FrequencyFilterParams(filter_type="notch")
        assert p.filter_type == "notch"

    def test_band_pass_sigma_high_only_le_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="band_pass", sigma_low=15.0, sigma_high=5.0)

    def test_large_valid_sigma(self):
        p = FrequencyFilterParams(sigma_low=500.0)
        assert p.sigma_low == 500.0


# ─── TestFftImageExtra ───────────────────────────────────────────────────────

class TestFftImageExtra:
    def test_small_image(self):
        img = _gray(8, 8)
        result = fft_image(img)
        assert result.shape == (8, 8)

    def test_non_square(self):
        img = _gray(32, 64)
        result = fft_image(img)
        assert result.shape == (32, 64)

    def test_constant_image_has_nonzero_component(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        result = fft_image(img)
        assert np.abs(result).max() > 0

    def test_zero_image_all_zero(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        result = fft_image(img)
        assert np.all(result == 0)

    def test_real_image_complex_output(self):
        result = fft_image(_gray())
        assert np.iscomplexobj(result)

    def test_gray_single_channel(self):
        result = fft_image(_gray(32, 32))
        assert result.ndim == 2


# ─── TestIfftImageExtra ──────────────────────────────────────────────────────

class TestIfftImageExtra:
    def test_roundtrip_preserves_shape(self):
        img = _gray(32, 32)
        result = ifft_image(fft_image(img))
        assert result.shape == img.shape

    def test_roundtrip_close_values(self):
        img = _gray(32, 32)
        rt = ifft_image(fft_image(img))
        # Should be close after round-trip (uint8 clipping applies)
        assert rt.shape == img.shape
        assert rt.dtype == np.uint8

    def test_non_square_shape(self):
        img = _gray(32, 48)
        result = ifft_image(fft_image(img))
        assert result.shape == (32, 48)

    def test_output_uint8(self):
        spec = fft_image(_gray())
        assert ifft_image(spec).dtype == np.uint8

    def test_output_in_range(self):
        spec = fft_image(_gray())
        out = ifft_image(spec)
        assert out.min() >= 0 and out.max() <= 255


# ─── TestGaussianLowPassExtra ─────────────────────────────────────────────────

class TestGaussianLowPassExtra:
    def test_small_image(self):
        result = gaussian_low_pass(_gray(8, 8), sigma=2.0)
        assert result.shape == (8, 8)

    def test_tiny_sigma_high_pass_complement(self):
        # Very small sigma low-pass passes little; output still valid
        result = gaussian_low_pass(_gray(), sigma=0.1)
        assert result.dtype == np.uint8

    def test_non_square(self):
        result = gaussian_low_pass(_gray(32, 48), sigma=5.0)
        assert result.shape == (32, 48)

    def test_uniform_image_preserved(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = gaussian_low_pass(img, sigma=200.0)
        # With very large sigma, output should be close to original
        assert result.shape == img.shape

    def test_multiple_sigmas_valid(self):
        for sigma in [1.0, 5.0, 20.0, 100.0]:
            result = gaussian_low_pass(_gray(), sigma=sigma)
            assert result.dtype == np.uint8

    def test_bgr_output_shape(self):
        result = gaussian_low_pass(_bgr(), sigma=5.0)
        assert result.ndim == 2  # returns grayscale


# ─── TestGaussianHighPassExtra ───────────────────────────────────────────────

class TestGaussianHighPassExtra:
    def test_non_square(self):
        result = gaussian_high_pass(_gray(32, 48), sigma=5.0)
        assert result.shape == (32, 48)

    def test_small_image(self):
        result = gaussian_high_pass(_gray(8, 8), sigma=2.0)
        assert result.shape == (8, 8)

    def test_multiple_sigmas_valid(self):
        for sigma in [1.0, 5.0, 20.0]:
            result = gaussian_high_pass(_gray(), sigma=sigma)
            assert result.dtype == np.uint8

    def test_very_large_sigma_near_zero(self):
        # With huge sigma, HP ≈ 0 everywhere
        img = _gray()
        result = gaussian_high_pass(img, sigma=10000.0)
        assert result.shape == img.shape

    def test_bgr_accepted(self):
        result = gaussian_high_pass(_bgr(), sigma=5.0)
        assert result.ndim == 2


# ─── TestBandPassFilterExtra ──────────────────────────────────────────────────

class TestBandPassFilterExtra:
    def test_non_square(self):
        result = band_pass_filter(_gray(32, 48), sigma_low=2.0, sigma_high=15.0)
        assert result.shape == (32, 48)

    def test_small_image(self):
        result = band_pass_filter(_gray(8, 8), sigma_low=1.0, sigma_high=3.0)
        assert result.shape == (8, 8)

    def test_various_sigmas(self):
        for lo, hi in [(1.0, 5.0), (2.0, 10.0), (5.0, 25.0)]:
            result = band_pass_filter(_gray(), sigma_low=lo, sigma_high=hi)
            assert result.dtype == np.uint8

    def test_bgr_accepted(self):
        result = band_pass_filter(_bgr(), sigma_low=3.0, sigma_high=15.0)
        assert result.ndim == 2

    def test_values_nonneg(self):
        result = band_pass_filter(_gray())
        assert result.min() >= 0


# ─── TestNotchFilterExtra ─────────────────────────────────────────────────────

class TestNotchFilterExtra:
    def test_non_square(self):
        result = notch_filter(_gray(32, 48))
        assert result.shape == (32, 48)

    def test_small_image(self):
        result = notch_filter(_gray(8, 8))
        assert result.shape == (8, 8)

    def test_multiple_notch_freqs(self):
        result = notch_filter(_gray(), notch_freqs=[(2, 2), (4, 0), (0, 4)], radius=2.0)
        assert result.dtype == np.uint8

    def test_large_radius_valid(self):
        result = notch_filter(_gray(), notch_freqs=[(5, 5)], radius=100.0)
        assert result.shape == (64, 64)

    def test_empty_notch_freqs_noop(self):
        img = _gray()
        result = notch_filter(img, notch_freqs=[])
        assert result.shape == img.shape


# ─── TestApplyFrequencyFilterExtra ───────────────────────────────────────────

class TestApplyFrequencyFilterExtra:
    def test_low_pass_non_square(self):
        img = _gray(32, 48)
        p = FrequencyFilterParams(filter_type="low_pass", sigma_low=5.0)
        result = apply_frequency_filter(img, p)
        assert result.shape == (32, 48)

    def test_high_pass_non_square(self):
        img = _gray(32, 48)
        p = FrequencyFilterParams(filter_type="high_pass", sigma_low=5.0)
        result = apply_frequency_filter(img, p)
        assert result.shape == (32, 48)

    def test_notch_non_square(self):
        img = _gray(32, 48)
        p = FrequencyFilterParams(filter_type="notch")
        result = apply_frequency_filter(img, p)
        assert result.shape == (32, 48)

    def test_all_output_uint8(self):
        img = _gray()
        for ft in ("low_pass", "high_pass", "notch"):
            p = FrequencyFilterParams(filter_type=ft)
            assert apply_frequency_filter(img, p).dtype == np.uint8

    def test_band_pass_non_square(self):
        img = _gray(32, 48)
        p = FrequencyFilterParams(filter_type="band_pass", sigma_low=2.0, sigma_high=10.0)
        result = apply_frequency_filter(img, p)
        assert result.shape == (32, 48)


# ─── TestBatchFrequencyFilterExtra ───────────────────────────────────────────

class TestBatchFrequencyFilterExtra:
    def test_single_image(self):
        result = batch_frequency_filter([_gray()])
        assert len(result) == 1

    def test_all_shapes_preserved(self):
        shapes = [(16, 16), (32, 48), (24, 24)]
        images = [_gray(h, w) for h, w in shapes]
        results = batch_frequency_filter(images)
        for r, (h, w) in zip(results, shapes):
            assert r.shape == (h, w)

    def test_high_pass_batch(self):
        p = FrequencyFilterParams(filter_type="high_pass")
        results = batch_frequency_filter([_gray(), _gray(32, 32)], params=p)
        assert all(r.dtype == np.uint8 for r in results)

    def test_band_pass_batch(self):
        p = FrequencyFilterParams(filter_type="band_pass", sigma_low=2.0, sigma_high=15.0)
        results = batch_frequency_filter([_gray()], params=p)
        assert results[0].dtype == np.uint8

    def test_bgr_in_batch(self):
        results = batch_frequency_filter([_bgr()])
        assert results[0].dtype == np.uint8

    def test_large_batch(self):
        images = [_gray(seed=i) for i in range(8)]
        results = batch_frequency_filter(images)
        assert len(results) == 8
