"""Extra tests for puzzle_reconstruction/preprocessing/illumination_corrector.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_corrector import (
    IlluminationParams,
    estimate_background,
    subtract_background,
    correct_by_homomorph,
    correct_by_retinex,
    correct_illumination,
    batch_correct,
    estimate_uniformity,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── IlluminationParams ───────────────────────────────────────────────────────

class TestIlluminationParamsExtra:
    def test_defaults(self):
        p = IlluminationParams()
        assert p.method == "background"
        assert p.blur_ksize == 51

    def test_valid_methods(self):
        for m in ("background", "homomorph", "retinex", "none"):
            IlluminationParams(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(method="bad")

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(blur_ksize=50)

    def test_small_ksize_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(blur_ksize=1)


# ─── estimate_background ──────────────────────────────────────────────────────

class TestEstimateBackgroundExtra:
    def test_returns_float32(self):
        bg = estimate_background(_gray())
        assert bg.dtype == np.float32

    def test_shape(self):
        bg = estimate_background(_gray(30, 40))
        assert bg.shape == (30, 40)

    def test_bgr_input(self):
        bg = estimate_background(_bgr())
        assert bg.ndim == 2

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            estimate_background(_gray(), ksize=50)

    def test_small_ksize_raises(self):
        with pytest.raises(ValueError):
            estimate_background(_gray(), ksize=1)


# ─── subtract_background ──────────────────────────────────────────────────────

class TestSubtractBackgroundExtra:
    def test_shape_gray(self):
        out = subtract_background(_gray())
        assert out.ndim == 2

    def test_shape_bgr(self):
        out = subtract_background(_bgr())
        assert out.ndim == 3

    def test_dtype_uint8(self):
        out = subtract_background(_gray())
        assert out.dtype == np.uint8

    def test_with_background(self):
        bg = estimate_background(_gray())
        out = subtract_background(_gray(), background=bg)
        assert out.dtype == np.uint8


# ─── correct_by_homomorph ─────────────────────────────────────────────────────

class TestCorrectByHomomorphExtra:
    def test_shape_gray(self):
        out = correct_by_homomorph(_gray())
        assert out.ndim == 2

    def test_shape_bgr(self):
        out = correct_by_homomorph(_bgr())
        assert out.ndim == 3

    def test_dtype_uint8(self):
        out = correct_by_homomorph(_gray())
        assert out.dtype == np.uint8

    def test_zero_d0_raises(self):
        with pytest.raises(ValueError):
            correct_by_homomorph(_gray(), d0=0)


# ─── correct_by_retinex ───────────────────────────────────────────────────────

class TestCorrectByRetinexExtra:
    def test_shape_gray(self):
        out = correct_by_retinex(_gray())
        assert out.ndim == 2

    def test_shape_bgr(self):
        out = correct_by_retinex(_bgr())
        assert out.ndim == 3

    def test_dtype_uint8(self):
        out = correct_by_retinex(_gray())
        assert out.dtype == np.uint8

    def test_empty_scales_raises(self):
        with pytest.raises(ValueError):
            correct_by_retinex(_gray(), scales=[])


# ─── correct_illumination ────────────────────────────────────────────────────

class TestCorrectIlluminationExtra:
    def test_default(self):
        out = correct_illumination(_gray())
        assert out.dtype == np.uint8

    def test_none_method(self):
        p = IlluminationParams(method="none")
        out = correct_illumination(_gray(), params=p)
        assert np.array_equal(out, _gray())

    def test_homomorph(self):
        p = IlluminationParams(method="homomorph")
        out = correct_illumination(_gray(), params=p)
        assert out.dtype == np.uint8

    def test_retinex(self):
        p = IlluminationParams(method="retinex")
        out = correct_illumination(_gray(), params=p)
        assert out.dtype == np.uint8


# ─── batch_correct ────────────────────────────────────────────────────────────

class TestBatchCorrectExtra:
    def test_empty(self):
        assert batch_correct([]) == []

    def test_length(self):
        results = batch_correct([_gray(), _gray()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_correct([_gray()])
        assert results[0].dtype == np.uint8


# ─── estimate_uniformity ──────────────────────────────────────────────────────

class TestEstimateUniformityExtra:
    def test_uniform_image(self):
        val = estimate_uniformity(_gray())
        assert 0.0 <= val <= 1.0

    def test_high_for_uniform(self):
        val = estimate_uniformity(_gray(val=128))
        assert val > 0.8

    def test_bgr_input(self):
        val = estimate_uniformity(_bgr())
        assert isinstance(val, float)
