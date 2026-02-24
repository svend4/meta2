"""Extra tests for puzzle_reconstruction/preprocessing/color_normalizer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.color_normalizer import (
    NormConfig,
    NormResult,
    gamma_correction,
    equalize_histogram,
    apply_clahe,
    grey_world_balance,
    max_rgb_balance,
    minmax_normalize,
    normalize_image,
    batch_normalize,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── NormConfig ─────────────────────────────────────────────────────────────

class TestNormConfigExtra:
    def test_defaults(self):
        cfg = NormConfig()
        assert cfg.method == "clahe"
        assert cfg.gamma == pytest.approx(1.0)
        assert cfg.clip_limit == pytest.approx(2.0)

    def test_valid_methods(self):
        for m in ("gamma", "equalize", "clahe", "grey_world", "max_rgb", "minmax"):
            NormConfig(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NormConfig(method="invalid")

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            NormConfig(gamma=0.0)

    def test_zero_clip_limit_raises(self):
        with pytest.raises(ValueError):
            NormConfig(clip_limit=0.0)

    def test_small_tile_raises(self):
        with pytest.raises(ValueError):
            NormConfig(tile_size=1)

    def test_target_mean_out_of_range_raises(self):
        with pytest.raises(ValueError):
            NormConfig(target_mean=300.0)


# ─── NormResult ─────────────────────────────────────────────────────────────

class TestNormResultExtra:
    def test_delta_mean(self):
        r = NormResult(image=_bgr(), method="clahe",
                       mean_before=100.0, mean_after=150.0)
        assert r.delta_mean == pytest.approx(50.0)

    def test_negative_mean_before_raises(self):
        with pytest.raises(ValueError):
            NormResult(image=_bgr(), method="clahe",
                       mean_before=-1.0, mean_after=100.0)

    def test_negative_mean_after_raises(self):
        with pytest.raises(ValueError):
            NormResult(image=_bgr(), method="clahe",
                       mean_before=100.0, mean_after=-1.0)


# ─── gamma_correction ───────────────────────────────────────────────────────

class TestGammaCorrectionNormExtra:
    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            gamma_correction(_bgr(), gamma=0.0)

    def test_shape_preserved(self):
        img = _bgr()
        out = gamma_correction(img, gamma=0.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


# ─── equalize_histogram ─────────────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_bgr(self):
        img = _bgr()
        out = equalize_histogram(img)
        assert out.shape == img.shape

    def test_grayscale(self):
        img = _gray()
        out = equalize_histogram(img)
        assert out.ndim == 2


# ─── apply_clahe ────────────────────────────────────────────────────────────

class TestApplyClaheExtra:
    def test_bgr(self):
        img = _bgr()
        out = apply_clahe(img)
        assert out.shape == img.shape

    def test_grayscale(self):
        img = _gray()
        out = apply_clahe(img)
        assert out.ndim == 2

    def test_zero_clip_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_bgr(), clip_limit=0.0)

    def test_small_tile_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_bgr(), tile_size=1)


# ─── grey_world_balance ─────────────────────────────────────────────────────

class TestGreyWorldBalanceExtra:
    def test_neutral(self):
        img = _bgr(val=128)
        out = grey_world_balance(img)
        assert np.allclose(out.astype(float), img.astype(float), atol=2)

    def test_non_3channel_raises(self):
        with pytest.raises(ValueError):
            grey_world_balance(_gray())

    def test_dtype_uint8(self):
        img = _bgr()
        out = grey_world_balance(img)
        assert out.dtype == np.uint8


# ─── max_rgb_balance ────────────────────────────────────────────────────────

class TestMaxRgbBalanceExtra:
    def test_shape_preserved(self):
        img = _bgr(val=100)
        out = max_rgb_balance(img)
        assert out.shape == img.shape

    def test_non_3channel_raises(self):
        with pytest.raises(ValueError):
            max_rgb_balance(_gray())

    def test_uniform_image(self):
        img = _bgr(val=100)
        out = max_rgb_balance(img)
        # Uniform channels should be scaled near 255 (rounding may differ by 1)
        assert out.max() >= 254


# ─── minmax_normalize ───────────────────────────────────────────────────────

class TestMinmaxNormalizeExtra:
    def test_range(self):
        img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        out = minmax_normalize(img)
        assert out.min() == 0 and out.max() == 255

    def test_constant_image(self):
        img = _gray(val=128)
        out = minmax_normalize(img)
        assert np.all(out == 0)

    def test_dtype_uint8(self):
        img = _bgr()
        out = minmax_normalize(img)
        assert out.dtype == np.uint8


# ─── normalize_image ────────────────────────────────────────────────────────

class TestNormalizeImageExtra:
    def test_default_returns_result(self):
        img = _bgr()
        r = normalize_image(img)
        assert isinstance(r, NormResult)
        assert r.method == "clahe"

    def test_gamma_method(self):
        cfg = NormConfig(method="gamma", gamma=0.5)
        r = normalize_image(_bgr(), cfg)
        assert r.method == "gamma"

    def test_equalize_method(self):
        cfg = NormConfig(method="equalize")
        r = normalize_image(_bgr(), cfg)
        assert r.method == "equalize"

    def test_minmax_method(self):
        cfg = NormConfig(method="minmax")
        r = normalize_image(_bgr(), cfg)
        assert r.method == "minmax"


# ─── batch_normalize ────────────────────────────────────────────────────────

class TestBatchNormalizeNormExtra:
    def test_empty(self):
        assert batch_normalize([]) == []

    def test_length(self):
        imgs = [_bgr(), _bgr()]
        result = batch_normalize(imgs)
        assert len(result) == 2

    def test_result_type(self):
        result = batch_normalize([_bgr()])
        assert isinstance(result[0], NormResult)
