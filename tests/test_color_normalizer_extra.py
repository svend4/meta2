"""Extra tests for puzzle_reconstruction.preprocessing.color_normalizer."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.color_normalizer import (
    NormConfig,
    NormResult,
    apply_clahe,
    batch_normalize,
    equalize_histogram,
    gamma_correction,
    grey_world_balance,
    max_rgb_balance,
    minmax_normalize,
    normalize_image,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestNormConfigExtra ─────────────────────────────────────────────────────

class TestNormConfigExtra:
    def test_default_method_clahe(self):
        assert NormConfig().method == "clahe"

    def test_default_gamma_1(self):
        assert NormConfig().gamma == pytest.approx(1.0)

    def test_default_clip_limit(self):
        assert NormConfig().clip_limit == pytest.approx(2.0)

    def test_default_tile_size(self):
        assert NormConfig().tile_size == 8

    @pytest.mark.parametrize("method", ["gamma", "equalize", "clahe",
                                         "grey_world", "max_rgb", "minmax"])
    def test_all_methods_valid(self, method):
        assert NormConfig(method=method).method == method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            NormConfig(method="unknown")

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            NormConfig(gamma=0.0)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError):
            NormConfig(gamma=-0.5)

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            NormConfig(clip_limit=0.0)

    def test_tile_size_1_raises(self):
        with pytest.raises(ValueError):
            NormConfig(tile_size=1)

    def test_target_mean_negative_raises(self):
        with pytest.raises(ValueError):
            NormConfig(target_mean=-1.0)

    def test_target_mean_256_raises(self):
        with pytest.raises(ValueError):
            NormConfig(target_mean=256.0)


# ─── TestNormResultExtra ─────────────────────────────────────────────────────

class TestNormResultExtra:
    def test_delta_mean_positive(self):
        nr = NormResult(image=_gray(), method="clahe",
                        mean_before=100.0, mean_after=130.0)
        assert nr.delta_mean == pytest.approx(30.0)

    def test_delta_mean_zero(self):
        nr = NormResult(image=_gray(), method="gamma",
                        mean_before=80.0, mean_after=80.0)
        assert nr.delta_mean == pytest.approx(0.0)

    def test_method_stored(self):
        nr = NormResult(image=_gray(), method="equalize",
                        mean_before=50.0, mean_after=80.0)
        assert nr.method == "equalize"

    def test_image_reference(self):
        img = _gray()
        nr = NormResult(image=img, method="gamma",
                        mean_before=50.0, mean_after=80.0)
        assert nr.image is img

    def test_negative_mean_before_raises(self):
        with pytest.raises(ValueError):
            NormResult(image=_gray(), method="clahe",
                       mean_before=-1.0, mean_after=100.0)

    def test_negative_mean_after_raises(self):
        with pytest.raises(ValueError):
            NormResult(image=_gray(), method="clahe",
                       mean_before=100.0, mean_after=-1.0)


# ─── TestGammaCorrectionExtra ─────────────────────────────────────────────────

class TestGammaCorrectionExtra:
    def test_returns_uint8(self):
        assert gamma_correction(_gray(), gamma=1.0).dtype == np.uint8

    def test_gamma_1_identity(self):
        img = _gray()
        out = gamma_correction(img, gamma=1.0)
        assert np.allclose(out.astype(float), img.astype(float), atol=1)

    def test_gamma_gt_1_brighter(self):
        # Standard gamma: (x/255)^gamma*255; gamma>1 → exponent>1 → darker
        img = np.full((10, 10), 200, dtype=np.uint8)
        assert gamma_correction(img, gamma=2.0).mean() < img.mean()

    def test_gamma_lt_1_darker(self):
        # Standard gamma: (x/255)^gamma*255; gamma<1 → exponent<1 → brighter
        img = np.full((10, 10), 100, dtype=np.uint8)
        assert gamma_correction(img, gamma=0.5).mean() > img.mean()

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            gamma_correction(_gray(), gamma=0.0)

    def test_bgr_uint8(self):
        assert gamma_correction(_bgr(), gamma=1.5).dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(32, 48)
        assert gamma_correction(img, gamma=1.2).shape == img.shape


# ─── TestEqualizeHistogramExtra ───────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_gray_uint8(self):
        assert equalize_histogram(_gray()).dtype == np.uint8

    def test_gray_shape(self):
        img = _gray(48, 64)
        assert equalize_histogram(img).shape == img.shape

    def test_bgr_uint8(self):
        assert equalize_histogram(_bgr()).dtype == np.uint8

    def test_bgr_shape(self):
        img = _bgr(48, 64)
        assert equalize_histogram(img).shape == img.shape

    def test_uniform_no_crash(self):
        assert equalize_histogram(np.full((32, 32), 128, dtype=np.uint8)).dtype == np.uint8


# ─── TestApplyClaheExtra ─────────────────────────────────────────────────────

class TestApplyClaheExtra:
    def test_gray_uint8(self):
        assert apply_clahe(_gray()).dtype == np.uint8

    def test_gray_shape(self):
        img = _gray(48, 64)
        assert apply_clahe(img).shape == img.shape

    def test_bgr_uint8(self):
        assert apply_clahe(_bgr()).dtype == np.uint8

    def test_bgr_shape(self):
        img = _bgr(48, 64)
        assert apply_clahe(img).shape == img.shape

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=0.0)

    def test_tile_size_1_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_size=1)


# ─── TestGreyWorldBalanceExtra ────────────────────────────────────────────────

class TestGreyWorldBalanceExtra:
    def test_returns_uint8(self):
        assert grey_world_balance(_bgr()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr(48, 64)
        assert grey_world_balance(img).shape == img.shape

    def test_gray_raises(self):
        with pytest.raises(ValueError):
            grey_world_balance(_gray())

    def test_neutral_stays_near_128(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = grey_world_balance(img)
        assert np.allclose(out.astype(float), 128.0, atol=2)


# ─── TestMaxRgbBalanceExtra ───────────────────────────────────────────────────

class TestMaxRgbBalanceExtra:
    def test_returns_uint8(self):
        assert max_rgb_balance(_bgr()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr(48, 64)
        assert max_rgb_balance(img).shape == img.shape

    def test_gray_raises(self):
        with pytest.raises(ValueError):
            max_rgb_balance(_gray())

    def test_all_channels_max_255(self):
        out = max_rgb_balance(_bgr())
        ch_max = out.astype(float).max(axis=(0, 1))
        assert np.all(ch_max == pytest.approx(255.0, abs=1))


# ─── TestMinmaxNormalizeExtra ─────────────────────────────────────────────────

class TestMinmaxNormalizeExtra:
    def test_returns_uint8(self):
        assert minmax_normalize(_gray()).dtype == np.uint8

    def test_range_0_255(self):
        out = minmax_normalize(_gray())
        assert out.min() == 0 and out.max() == 255

    def test_uniform_all_zeros(self):
        out = minmax_normalize(np.full((10, 10), 128, dtype=np.uint8))
        assert np.all(out == 0)

    def test_bgr_uint8(self):
        assert minmax_normalize(_bgr()).dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        assert minmax_normalize(img).shape == img.shape


# ─── TestNormalizeImageExtra ─────────────────────────────────────────────────

class TestNormalizeImageExtra:
    def test_returns_norm_result(self):
        assert isinstance(normalize_image(_gray()), NormResult)

    def test_image_uint8(self):
        assert normalize_image(_gray()).image.dtype == np.uint8

    def test_method_stored(self):
        cfg = NormConfig(method="gamma", gamma=1.5)
        assert normalize_image(_gray(), cfg).method == "gamma"

    @pytest.mark.parametrize("method", ["gamma", "equalize", "clahe", "minmax"])
    def test_gray_methods(self, method):
        nr = normalize_image(_gray(), NormConfig(method=method))
        assert nr.image.dtype == np.uint8

    @pytest.mark.parametrize("method", ["gamma", "equalize", "clahe",
                                          "grey_world", "max_rgb", "minmax"])
    def test_bgr_methods(self, method):
        nr = normalize_image(_bgr(), NormConfig(method=method))
        assert nr.image.dtype == np.uint8

    def test_default_config_ok(self):
        assert isinstance(normalize_image(_gray(), None), NormResult)

    def test_mean_before_nonneg(self):
        assert normalize_image(_gray()).mean_before >= 0.0

    def test_mean_after_nonneg(self):
        assert normalize_image(_gray()).mean_after >= 0.0


# ─── TestBatchNormalizeExtra ─────────────────────────────────────────────────

class TestBatchNormalizeExtra:
    def test_returns_list(self):
        assert isinstance(batch_normalize([_gray() for _ in range(3)]), list)

    def test_length_matches(self):
        assert len(batch_normalize([_gray() for _ in range(4)])) == 4

    def test_all_norm_results(self):
        results = batch_normalize([_gray() for _ in range(3)])
        assert all(isinstance(r, NormResult) for r in results)

    def test_empty_list(self):
        assert batch_normalize([]) == []

    def test_custom_config_method(self):
        cfg = NormConfig(method="equalize")
        results = batch_normalize([_gray() for _ in range(2)], cfg)
        assert all(r.method == "equalize" for r in results)

    def test_single_image(self):
        results = batch_normalize([_gray()])
        assert len(results) == 1 and isinstance(results[0], NormResult)
