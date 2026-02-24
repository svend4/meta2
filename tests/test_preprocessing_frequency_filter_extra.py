"""Extra tests for puzzle_reconstruction/preprocessing/frequency_filter.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.frequency_filter import (
    FrequencyFilterParams,
    fft_image,
    ifft_image,
    gaussian_low_pass,
    gaussian_high_pass,
    band_pass_filter,
    notch_filter,
    apply_frequency_filter,
    batch_frequency_filter,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _edge_image(h=64, w=64):
    """Image with a sharp vertical edge."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2:] = 255
    return img


# ─── FrequencyFilterParams ────────────────────────────────────────────────────

class TestFrequencyFilterParamsExtra:
    def test_defaults(self):
        p = FrequencyFilterParams()
        assert p.filter_type == "low_pass"
        assert p.sigma_low == pytest.approx(10.0)

    def test_valid_types(self):
        for t in ("low_pass", "high_pass", "band_pass", "notch"):
            FrequencyFilterParams(filter_type=t)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="invalid")

    def test_zero_sigma_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(sigma_low=0)

    def test_band_pass_bad_sigmas_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=20, sigma_high=10)

    def test_zero_notch_radius_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(notch_radius=0)


# ─── fft_image ────────────────────────────────────────────────────────────────

class TestFftImageExtra:
    def test_returns_complex(self):
        out = fft_image(_gray())
        assert np.iscomplexobj(out)

    def test_shape_preserved(self):
        img = _gray(32, 48)
        out = fft_image(img)
        assert out.shape == (32, 48)

    def test_bgr_input(self):
        out = fft_image(_bgr())
        assert out.ndim == 2  # Converted to gray

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            fft_image(np.array([1, 2, 3], dtype=np.uint8))


# ─── ifft_image ───────────────────────────────────────────────────────────────

class TestIfftImageExtra:
    def test_returns_uint8(self):
        spec = fft_image(_gray())
        out = ifft_image(spec)
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(32, 48)
        spec = fft_image(img)
        out = ifft_image(spec)
        assert out.shape == (32, 48)

    def test_roundtrip(self):
        img = _edge_image()
        spec = fft_image(img)
        out = ifft_image(spec)
        # Should be close to original after normalization
        assert out.shape == img.shape


# ─── gaussian_low_pass ────────────────────────────────────────────────────────

class TestGaussianLowPassExtra:
    def test_shape(self):
        out = gaussian_low_pass(_gray(), sigma=10.0)
        assert out.shape == (64, 64)

    def test_dtype_uint8(self):
        out = gaussian_low_pass(_gray(), sigma=10.0)
        assert out.dtype == np.uint8

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_low_pass(_gray(), sigma=0)

    def test_bgr_input(self):
        out = gaussian_low_pass(_bgr(), sigma=5.0)
        assert out.ndim == 2


# ─── gaussian_high_pass ───────────────────────────────────────────────────────

class TestGaussianHighPassExtra:
    def test_shape(self):
        out = gaussian_high_pass(_gray(), sigma=10.0)
        assert out.shape == (64, 64)

    def test_dtype_uint8(self):
        out = gaussian_high_pass(_gray(), sigma=5.0)
        assert out.dtype == np.uint8

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_high_pass(_gray(), sigma=0)


# ─── band_pass_filter ─────────────────────────────────────────────────────────

class TestBandPassFilterExtra:
    def test_shape(self):
        out = band_pass_filter(_gray(), sigma_low=5.0, sigma_high=20.0)
        assert out.shape == (64, 64)

    def test_dtype_uint8(self):
        out = band_pass_filter(_gray())
        assert out.dtype == np.uint8

    def test_bad_sigmas_raises(self):
        with pytest.raises(ValueError):
            band_pass_filter(_gray(), sigma_low=20, sigma_high=10)

    def test_zero_sigma_low_raises(self):
        with pytest.raises(ValueError):
            band_pass_filter(_gray(), sigma_low=0)


# ─── notch_filter ─────────────────────────────────────────────────────────────

class TestNotchFilterExtra:
    def test_no_freqs(self):
        out = notch_filter(_gray(), notch_freqs=None)
        assert out.shape == (64, 64)

    def test_with_freqs(self):
        out = notch_filter(_edge_image(), notch_freqs=[(10, 0)], radius=3.0)
        assert out.shape == (64, 64)
        assert out.dtype == np.uint8

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            notch_filter(_gray(), radius=0)


# ─── apply_frequency_filter ──────────────────────────────────────────────────

class TestApplyFrequencyFilterExtra:
    def test_low_pass(self):
        p = FrequencyFilterParams(filter_type="low_pass")
        out = apply_frequency_filter(_gray(), p)
        assert out.dtype == np.uint8

    def test_high_pass(self):
        p = FrequencyFilterParams(filter_type="high_pass")
        out = apply_frequency_filter(_gray(), p)
        assert out.dtype == np.uint8

    def test_band_pass(self):
        p = FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=5, sigma_high=20)
        out = apply_frequency_filter(_gray(), p)
        assert out.dtype == np.uint8

    def test_notch(self):
        p = FrequencyFilterParams(filter_type="notch")
        out = apply_frequency_filter(_gray(), p)
        assert out.dtype == np.uint8


# ─── batch_frequency_filter ──────────────────────────────────────────────────

class TestBatchFrequencyFilterExtra:
    def test_empty(self):
        assert batch_frequency_filter([]) == []

    def test_length(self):
        results = batch_frequency_filter([_gray(), _gray()])
        assert len(results) == 2

    def test_default_params(self):
        results = batch_frequency_filter([_gray()])
        assert results[0].dtype == np.uint8
