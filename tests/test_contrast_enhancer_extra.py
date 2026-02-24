"""Extra tests for puzzle_reconstruction/preprocessing/contrast_enhancer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.contrast_enhancer import (
    EnhanceConfig,
    EnhanceResult,
    equalize_histogram,
    stretch_contrast,
    apply_gamma,
    clahe_enhance,
    enhance_contrast,
    batch_enhance,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=16, w=16, low=50, high=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, (h, w), dtype=np.uint8)


def _rgb(h=16, w=16, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat(val=128):
    return np.full((8, 8), val, dtype=np.uint8)


# ─── EnhanceConfig (extra) ───────────────────────────────────────────────────

class TestEnhanceConfigExtra:
    def test_default_method(self):
        assert EnhanceConfig().method == "equalize"

    def test_default_gamma(self):
        assert EnhanceConfig().gamma == pytest.approx(1.0)

    def test_default_clip_limit(self):
        assert EnhanceConfig().clip_limit == pytest.approx(0.03)

    def test_default_tile_size(self):
        assert EnhanceConfig().tile_size == 8

    def test_valid_method_stretch(self):
        cfg = EnhanceConfig(method="stretch")
        assert cfg.method == "stretch"

    def test_valid_method_gamma(self):
        cfg = EnhanceConfig(method="gamma")
        assert cfg.method == "gamma"

    def test_valid_method_clahe(self):
        cfg = EnhanceConfig(method="clahe")
        assert cfg.method == "clahe"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="histogram")

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=0.0)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=-1.0)

    def test_clip_limit_zero_ok(self):
        cfg = EnhanceConfig(clip_limit=0.0)
        assert cfg.clip_limit == pytest.approx(0.0)

    def test_clip_limit_negative_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=-0.01)

    def test_clip_limit_above_one_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=1.1)

    def test_tile_size_one_ok(self):
        cfg = EnhanceConfig(tile_size=1)
        assert cfg.tile_size == 1

    def test_tile_size_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(tile_size=0)

    def test_output_range_equal_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(1.0, 1.0))

    def test_output_range_inverted_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(2.0, 1.0))

    def test_output_range_valid(self):
        cfg = EnhanceConfig(output_range=(0.0, 1.0))
        assert cfg.output_range == (0.0, 1.0)


# ─── EnhanceResult (extra) ───────────────────────────────────────────────────

class TestEnhanceResultExtra:
    def _make(self, in_std=50.0, out_std=100.0, in_mean=100.0, out_mean=120.0):
        return EnhanceResult(
            image=_gray(), method="equalize",
            input_mean=in_mean, output_mean=out_mean,
            input_std=in_std, output_std=out_std,
        )

    def test_contrast_gain(self):
        r = self._make(in_std=50.0, out_std=75.0)
        assert r.contrast_gain == pytest.approx(1.5)

    def test_contrast_gain_zero_in_std(self):
        r = self._make(in_std=0.0, out_std=10.0)
        assert r.contrast_gain == pytest.approx(0.0)

    def test_mean_shift_positive(self):
        r = self._make(in_mean=100.0, out_mean=130.0)
        assert r.mean_shift == pytest.approx(30.0)

    def test_mean_shift_negative(self):
        r = self._make(in_mean=150.0, out_mean=100.0)
        assert r.mean_shift == pytest.approx(-50.0)

    def test_method_stored(self):
        r = self._make()
        assert r.method == "equalize"

    def test_image_shape_preserved(self):
        img = _gray(20, 24)
        r = EnhanceResult(image=img, method="gamma",
                          input_mean=0.0, output_mean=0.0,
                          input_std=0.0, output_std=0.0)
        assert r.image.shape == (20, 24)


# ─── equalize_histogram (extra) ──────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_shape_preserved_2d(self):
        img = _gray()
        assert equalize_histogram(img).shape == img.shape

    def test_shape_preserved_3d(self):
        img = _rgb()
        assert equalize_histogram(img).shape == img.shape

    def test_output_in_range(self):
        out = equalize_histogram(_gray()).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_dtype_uint8(self):
        assert equalize_histogram(_gray()).dtype == np.uint8

    def test_flat_image_ok(self):
        out = equalize_histogram(_flat(128)).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_3d_three_channels(self):
        out = equalize_histogram(_rgb())
        assert out.shape[2] == 3


# ─── stretch_contrast (extra) ────────────────────────────────────────────────

class TestStretchContrastExtra:
    def test_shape_preserved_2d(self):
        img = _gray()
        assert stretch_contrast(img).shape == img.shape

    def test_shape_preserved_3d(self):
        img = _rgb()
        assert stretch_contrast(img).shape == img.shape

    def test_output_in_range(self):
        out = stretch_contrast(_gray()).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_dtype_uint8(self):
        assert stretch_contrast(_gray()).dtype == np.uint8

    def test_flat_image_ok(self):
        out = stretch_contrast(_flat(100)).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_custom_output_range(self):
        cfg = EnhanceConfig(method="stretch", output_range=(0.0, 1.0))
        out = stretch_contrast(_gray(), cfg).astype(float)
        assert float(out.max()) <= 1.0


# ─── apply_gamma (extra) ─────────────────────────────────────────────────────

class TestApplyGammaExtra:
    def test_shape_2d_preserved(self):
        img = _gray()
        assert apply_gamma(img).shape == img.shape

    def test_shape_3d_preserved(self):
        img = _rgb()
        assert apply_gamma(img).shape == img.shape

    def test_dtype_uint8(self):
        cfg = EnhanceConfig(method="gamma", gamma=1.5)
        assert apply_gamma(_gray(), cfg).dtype == np.uint8

    def test_output_in_range(self):
        cfg = EnhanceConfig(method="gamma", gamma=2.0)
        out = apply_gamma(_gray(), cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_gamma_one_near_identity(self):
        img = _gray()
        cfg = EnhanceConfig(method="gamma", gamma=1.0)
        out = apply_gamma(img, cfg).astype(float)
        diff = np.abs(out - img.astype(float)).max()
        assert diff <= 2.0

    def test_gamma_gt_1_darkens_midtones(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        cfg_hi = EnhanceConfig(method="gamma", gamma=2.0)
        cfg_lo = EnhanceConfig(method="gamma", gamma=0.5)
        mean_hi = apply_gamma(img, cfg_hi).astype(float).mean()
        mean_lo = apply_gamma(img, cfg_lo).astype(float).mean()
        assert mean_hi < mean_lo


# ─── clahe_enhance (extra) ───────────────────────────────────────────────────

class TestClaheEnhanceExtra:
    def test_shape_2d_preserved(self):
        img = _gray(16, 16)
        assert clahe_enhance(img).shape == img.shape

    def test_shape_3d_preserved(self):
        img = _rgb(16, 16)
        assert clahe_enhance(img).shape == img.shape

    def test_output_in_range(self):
        out = clahe_enhance(_gray()).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_dtype_uint8(self):
        assert clahe_enhance(_gray()).dtype == np.uint8

    def test_tile_size_one(self):
        cfg = EnhanceConfig(method="clahe", tile_size=1)
        out = clahe_enhance(_gray(4, 4), cfg)
        assert out.shape == (4, 4)

    def test_flat_ok(self):
        cfg = EnhanceConfig(method="clahe", tile_size=4)
        out = clahe_enhance(_flat(100), cfg).astype(float)
        assert float(out.min()) >= 0.0


# ─── enhance_contrast (extra) ────────────────────────────────────────────────

class TestEnhanceContrastExtra:
    def test_returns_enhance_result(self):
        assert isinstance(enhance_contrast(_gray()), EnhanceResult)

    def test_method_equalize_default(self):
        assert enhance_contrast(_gray()).method == "equalize"

    def test_method_stretch(self):
        cfg = EnhanceConfig(method="stretch")
        assert enhance_contrast(_gray(), cfg).method == "stretch"

    def test_image_shape_preserved(self):
        img = _gray(20, 24)
        assert enhance_contrast(img).image.shape == (20, 24)

    def test_input_mean_correct(self):
        img = _gray()
        r = enhance_contrast(img)
        assert r.input_mean == pytest.approx(float(img.astype(float).mean()))

    def test_contrast_gain_gte_zero(self):
        assert enhance_contrast(_gray()).contrast_gain >= 0.0

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.array([1, 2, 3], dtype=np.uint8))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.zeros((2, 2, 2, 2), dtype=np.uint8))

    def test_3d_rgb_ok(self):
        img = _rgb()
        r = enhance_contrast(img)
        assert r.image.shape == img.shape


# ─── batch_enhance (extra) ───────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_returns_list(self):
        imgs = [_gray(seed=i) for i in range(3)]
        assert isinstance(batch_enhance(imgs), list)

    def test_length_matches(self):
        imgs = [_gray(seed=i) for i in range(3)]
        assert len(batch_enhance(imgs)) == 3

    def test_all_enhance_result(self):
        for r in batch_enhance([_gray(seed=i) for i in range(2)]):
            assert isinstance(r, EnhanceResult)

    def test_empty_list(self):
        assert batch_enhance([]) == []

    def test_custom_config_applied(self):
        cfg = EnhanceConfig(method="gamma", gamma=2.0)
        imgs = [_gray(seed=i) for i in range(3)]
        for r in batch_enhance(imgs, cfg):
            assert r.method == "gamma"

    def test_shapes_preserved(self):
        shapes = [(8, 8), (12, 16)]
        imgs = [np.zeros(s, dtype=np.uint8) for s in shapes]
        for shape, r in zip(shapes, batch_enhance(imgs)):
            assert r.image.shape == shape
