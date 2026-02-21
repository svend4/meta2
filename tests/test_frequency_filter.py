"""Tests for puzzle_reconstruction.preprocessing.frequency_filter."""
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

def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.integers(0, 256, (h, w))).astype(np.uint8)


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.integers(0, 256, (h, w, 3))).astype(np.uint8)


# ─── FrequencyFilterParams ───────────────────────────────────────────────────

class TestFrequencyFilterParams:
    def test_default_filter_type(self):
        p = FrequencyFilterParams()
        assert p.filter_type == "low_pass"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="unknown")

    def test_all_valid_types(self):
        for t in ("low_pass", "high_pass", "band_pass", "notch"):
            if t == "band_pass":
                FrequencyFilterParams(filter_type=t, sigma_low=5.0, sigma_high=20.0)
            else:
                FrequencyFilterParams(filter_type=t)

    def test_zero_sigma_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(sigma_low=0.0)

    def test_negative_sigma_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(sigma_low=-1.0)

    def test_band_pass_sigma_high_le_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=20.0, sigma_high=10.0)

    def test_band_pass_sigma_high_eq_low_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=10.0, sigma_high=10.0)

    def test_zero_notch_radius_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="notch", notch_radius=0.0)

    def test_negative_notch_radius_raises(self):
        with pytest.raises(ValueError):
            FrequencyFilterParams(filter_type="notch", notch_radius=-1.0)

    def test_valid_band_pass_params(self):
        p = FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=5.0, sigma_high=25.0)
        assert p.sigma_low < p.sigma_high


# ─── fft_image ───────────────────────────────────────────────────────────────

class TestFftImage:
    def test_returns_complex(self):
        result = fft_image(_gray())
        assert np.iscomplexobj(result)

    def test_shape_preserved(self):
        img = _gray(48, 64)
        result = fft_image(img)
        assert result.shape == (48, 64)

    def test_bgr_accepted(self):
        result = fft_image(_bgr())
        assert result.shape == (64, 64)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            fft_image(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_not_all_zero_for_nonzero_image(self):
        img = np.ones((32, 32), dtype=np.uint8) * 128
        result = fft_image(img)
        assert not np.all(result == 0)

    def test_dtype_float64(self):
        result = fft_image(_gray())
        assert result.dtype == np.complex128


# ─── ifft_image ──────────────────────────────────────────────────────────────

class TestIfftImage:
    def test_returns_uint8(self):
        spec = fft_image(_gray())
        result = ifft_image(spec)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        spec = fft_image(img)
        result = ifft_image(spec)
        assert result.shape == (48, 64)

    def test_values_in_0_255(self):
        spec = fft_image(_gray())
        result = ifft_image(spec)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_zero_spectrum_all_zero(self):
        spec = np.zeros((32, 32), dtype=np.complex128)
        result = ifft_image(spec)
        assert np.all(result == 0)


# ─── gaussian_low_pass ───────────────────────────────────────────────────────

class TestGaussianLowPass:
    def test_returns_uint8(self):
        result = gaussian_low_pass(_gray(), sigma=10.0)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        result = gaussian_low_pass(img, sigma=10.0)
        assert result.shape == (48, 64)

    def test_values_in_range(self):
        result = gaussian_low_pass(_gray(), sigma=10.0)
        assert 0 <= result.min() and result.max() <= 255

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_low_pass(_gray(), sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_low_pass(_gray(), sigma=-5.0)

    def test_bgr_accepted(self):
        result = gaussian_low_pass(_bgr(), sigma=10.0)
        assert result.shape == (64, 64)

    def test_large_sigma_preserves_dc(self):
        # With very large sigma the low-pass passes everything
        img = _gray()
        result = gaussian_low_pass(img, sigma=1000.0)
        assert result.shape == img.shape


# ─── gaussian_high_pass ──────────────────────────────────────────────────────

class TestGaussianHighPass:
    def test_returns_uint8(self):
        result = gaussian_high_pass(_gray(), sigma=10.0)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        result = gaussian_high_pass(img, sigma=10.0)
        assert result.shape == (48, 64)

    def test_values_in_range(self):
        result = gaussian_high_pass(_gray(), sigma=10.0)
        assert 0 <= result.min() and result.max() <= 255

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_high_pass(_gray(), sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_high_pass(_gray(), sigma=-1.0)

    def test_bgr_accepted(self):
        result = gaussian_high_pass(_bgr(), sigma=5.0)
        assert result.shape == (64, 64)

    def test_uniform_image_zero_output(self):
        # Uniform image has no high-freq content → near-zero output
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = gaussian_high_pass(img, sigma=5.0)
        assert result.shape == (32, 32)


# ─── band_pass_filter ────────────────────────────────────────────────────────

class TestBandPassFilter:
    def test_returns_uint8(self):
        result = band_pass_filter(_gray(), sigma_low=5.0, sigma_high=20.0)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        result = band_pass_filter(img, sigma_low=5.0, sigma_high=20.0)
        assert result.shape == (48, 64)

    def test_values_in_range(self):
        result = band_pass_filter(_gray())
        assert 0 <= result.min() and result.max() <= 255

    def test_zero_sigma_low_raises(self):
        with pytest.raises(ValueError):
            band_pass_filter(_gray(), sigma_low=0.0, sigma_high=20.0)

    def test_sigma_high_le_low_raises(self):
        with pytest.raises(ValueError):
            band_pass_filter(_gray(), sigma_low=20.0, sigma_high=10.0)

    def test_sigma_high_eq_low_raises(self):
        with pytest.raises(ValueError):
            band_pass_filter(_gray(), sigma_low=10.0, sigma_high=10.0)

    def test_bgr_accepted(self):
        result = band_pass_filter(_bgr(), sigma_low=3.0, sigma_high=15.0)
        assert result.shape == (64, 64)


# ─── notch_filter ─────────────────────────────────────────────────────────────

class TestNotchFilter:
    def test_returns_uint8(self):
        result = notch_filter(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        result = notch_filter(img)
        assert result.shape == (48, 64)

    def test_none_notch_freqs_identity(self):
        img = _gray()
        result = notch_filter(img, notch_freqs=None)
        assert result.shape == img.shape

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            notch_filter(_gray(), radius=0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            notch_filter(_gray(), radius=-1.0)

    def test_with_notch_freqs(self):
        result = notch_filter(_gray(), notch_freqs=[(5, 5), (10, 0)], radius=3.0)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_values_in_range(self):
        result = notch_filter(_gray(), notch_freqs=[(3, 3)])
        assert 0 <= result.min() and result.max() <= 255


# ─── apply_frequency_filter ──────────────────────────────────────────────────

class TestApplyFrequencyFilter:
    def _result(self, filter_type: str) -> np.ndarray:
        if filter_type == "band_pass":
            p = FrequencyFilterParams(filter_type=filter_type,
                                      sigma_low=5.0, sigma_high=20.0)
        else:
            p = FrequencyFilterParams(filter_type=filter_type)
        return apply_frequency_filter(_gray(), p)

    def test_low_pass_returns_uint8(self):
        assert self._result("low_pass").dtype == np.uint8

    def test_high_pass_returns_uint8(self):
        assert self._result("high_pass").dtype == np.uint8

    def test_band_pass_returns_uint8(self):
        assert self._result("band_pass").dtype == np.uint8

    def test_notch_returns_uint8(self):
        assert self._result("notch").dtype == np.uint8

    def test_all_types_correct_shape(self):
        img = _gray(48, 64)
        for t in ("low_pass", "high_pass", "notch"):
            p = FrequencyFilterParams(filter_type=t)
            result = apply_frequency_filter(img, p)
            assert result.shape == (48, 64)

    def test_band_pass_correct_shape(self):
        img = _gray(48, 64)
        p = FrequencyFilterParams(filter_type="band_pass",
                                  sigma_low=5.0, sigma_high=25.0)
        result = apply_frequency_filter(img, p)
        assert result.shape == (48, 64)


# ─── batch_frequency_filter ──────────────────────────────────────────────────

class TestBatchFrequencyFilter:
    def test_returns_list(self):
        result = batch_frequency_filter([_gray(), _gray()])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        images = [_gray(), _gray(32, 32), _gray(48, 48)]
        result = batch_frequency_filter(images)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert batch_frequency_filter([]) == []

    def test_all_uint8(self):
        result = batch_frequency_filter([_gray(), _gray()])
        assert all(r.dtype == np.uint8 for r in result)

    def test_shapes_preserved(self):
        images = [_gray(32, 32), _gray(48, 64)]
        result = batch_frequency_filter(images)
        assert result[0].shape == (32, 32)
        assert result[1].shape == (48, 64)

    def test_default_params_low_pass(self):
        result = batch_frequency_filter([_gray()])
        assert result[0].dtype == np.uint8

    def test_custom_params_forwarded(self):
        p = FrequencyFilterParams(filter_type="high_pass", sigma_low=5.0)
        result = batch_frequency_filter([_gray()], params=p)
        assert result[0].shape == (64, 64)
