"""Extra tests for puzzle_reconstruction/preprocessing/noise_filter.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_filter import (
    NoiseFilterParams,
    average_filter,
    gaussian_filter,
    median_filter,
    bilateral_filter,
    nlm_filter,
    apply_noise_filter,
    batch_noise_filter,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── NoiseFilterParams ────────────────────────────────────────────────────────

class TestNoiseFilterParamsExtra:
    def test_defaults(self):
        p = NoiseFilterParams()
        assert p.method == "gaussian"
        assert p.kernel_size == 5

    def test_valid_methods(self):
        for m in ("average", "gaussian", "median", "bilateral", "nlm"):
            NoiseFilterParams(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(method="bad")

    def test_small_kernel_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(kernel_size=4)

    def test_zero_sigma_color_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(sigma_color=0)

    def test_zero_sigma_space_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(sigma_space=0)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(h=0)

    def test_even_template_window_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(template_window=4)

    def test_even_search_window_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(search_window=4)


# ─── average_filter ───────────────────────────────────────────────────────────

class TestAverageFilterExtra:
    def test_shape_preserved(self):
        out = average_filter(_gray())
        assert out.shape == (50, 50)

    def test_bgr(self):
        out = average_filter(_bgr())
        assert out.shape == (50, 50, 3)

    def test_dtype_uint8(self):
        assert average_filter(_gray()).dtype == np.uint8

    def test_bad_ksize_raises(self):
        with pytest.raises(ValueError):
            average_filter(_gray(), kernel_size=2)


# ─── gaussian_filter ──────────────────────────────────────────────────────────

class TestGaussianFilterExtra:
    def test_shape_preserved(self):
        out = gaussian_filter(_gray())
        assert out.shape == (50, 50)

    def test_bgr(self):
        out = gaussian_filter(_bgr())
        assert out.shape == (50, 50, 3)

    def test_bad_ksize_raises(self):
        with pytest.raises(ValueError):
            gaussian_filter(_gray(), kernel_size=2)


# ─── median_filter ────────────────────────────────────────────────────────────

class TestMedianFilterExtra:
    def test_shape_preserved(self):
        out = median_filter(_gray())
        assert out.shape == (50, 50)

    def test_bgr(self):
        out = median_filter(_bgr())
        assert out.shape == (50, 50, 3)

    def test_bad_ksize_raises(self):
        with pytest.raises(ValueError):
            median_filter(_gray(), kernel_size=2)


# ─── bilateral_filter ─────────────────────────────────────────────────────────

class TestBilateralFilterExtra:
    def test_shape_preserved(self):
        out = bilateral_filter(_gray(), kernel_size=5)
        assert out.shape == (50, 50)

    def test_bgr(self):
        out = bilateral_filter(_bgr(), kernel_size=5)
        assert out.shape == (50, 50, 3)

    def test_zero_sigma_color_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(_gray(), sigma_color=0)

    def test_zero_sigma_space_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(_gray(), sigma_space=0)


# ─── nlm_filter ───────────────────────────────────────────────────────────────

class TestNlmFilterExtra:
    def test_shape_gray(self):
        out = nlm_filter(_gray())
        assert out.shape == (50, 50)

    def test_shape_bgr(self):
        out = nlm_filter(_bgr())
        assert out.shape == (50, 50, 3)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(_gray(), h=0)

    def test_even_template_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(_gray(), template_window=4)

    def test_even_search_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(_gray(), search_window=4)


# ─── apply_noise_filter ──────────────────────────────────────────────────────

class TestApplyNoiseFilterExtra:
    def test_all_methods(self):
        for m in ("average", "gaussian", "median", "bilateral", "nlm"):
            p = NoiseFilterParams(method=m)
            out = apply_noise_filter(_gray(), p)
            assert out.dtype == np.uint8


# ─── batch_noise_filter ──────────────────────────────────────────────────────

class TestBatchNoiseFilterExtra:
    def test_empty(self):
        p = NoiseFilterParams()
        assert batch_noise_filter([], p) == []

    def test_length(self):
        p = NoiseFilterParams()
        results = batch_noise_filter([_gray(), _gray()], p)
        assert len(results) == 2
