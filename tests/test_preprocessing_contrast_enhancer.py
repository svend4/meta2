"""Тесты для puzzle_reconstruction.preprocessing.contrast_enhancer."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(val: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _cfg(method="equalize", **kw) -> EnhanceConfig:
    return EnhanceConfig(method=method, **kw)


# ─── TestEnhanceConfig ────────────────────────────────────────────────────────

class TestEnhanceConfig:
    def test_defaults(self):
        cfg = EnhanceConfig()
        assert cfg.method == "equalize"
        assert cfg.gamma == pytest.approx(1.0)
        assert cfg.clip_limit == pytest.approx(0.03)
        assert cfg.tile_size == 8
        assert cfg.output_range == (0.0, 255.0)

    def test_valid_methods(self):
        for m in ("equalize", "stretch", "gamma", "clahe"):
            cfg = EnhanceConfig(method=m)
            assert cfg.method == m

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="unknown")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="")

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=0.0)

    def test_gamma_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=-1.0)

    def test_gamma_small_positive_ok(self):
        cfg = EnhanceConfig(method="gamma", gamma=0.01)
        assert cfg.gamma == pytest.approx(0.01)

    def test_clip_limit_zero_ok(self):
        cfg = EnhanceConfig(clip_limit=0.0)
        assert cfg.clip_limit == 0.0

    def test_clip_limit_one_ok(self):
        cfg = EnhanceConfig(clip_limit=1.0)
        assert cfg.clip_limit == 1.0

    def test_clip_limit_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=-0.01)

    def test_clip_limit_above_one_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=1.01)

    def test_tile_size_one_ok(self):
        cfg = EnhanceConfig(tile_size=1)
        assert cfg.tile_size == 1

    def test_tile_size_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(tile_size=0)

    def test_output_range_min_eq_max_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(128.0, 128.0))

    def test_output_range_min_gt_max_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(200.0, 100.0))

    def test_output_range_custom_ok(self):
        cfg = EnhanceConfig(output_range=(0.0, 1.0))
        assert cfg.output_range == (0.0, 1.0)


# ─── TestEnhanceResult ────────────────────────────────────────────────────────

class TestEnhanceResult:
    def _make(self) -> EnhanceResult:
        img = _gray()
        return EnhanceResult(image=img, method="equalize",
                             input_mean=100.0, output_mean=128.0,
                             input_std=30.0, output_std=50.0)

    def test_contrast_gain(self):
        r = self._make()
        assert r.contrast_gain == pytest.approx(50.0 / 30.0)

    def test_contrast_gain_zero_input_std(self):
        r = EnhanceResult(image=_gray(), method="equalize",
                          input_mean=128.0, output_mean=128.0,
                          input_std=0.0, output_std=10.0)
        assert r.contrast_gain == pytest.approx(0.0)

    def test_mean_shift(self):
        r = self._make()
        assert r.mean_shift == pytest.approx(28.0)

    def test_mean_shift_negative(self):
        r = EnhanceResult(image=_gray(), method="equalize",
                          input_mean=200.0, output_mean=128.0,
                          input_std=30.0, output_std=30.0)
        assert r.mean_shift == pytest.approx(-72.0)


# ─── TestEqualizeHistogram ────────────────────────────────────────────────────

class TestEqualizeHistogram:
    def test_returns_ndarray(self):
        assert isinstance(equalize_histogram(_gray()), np.ndarray)

    def test_shape_preserved_2d(self):
        img = _gray(48, 64)
        assert equalize_histogram(img).shape == (48, 64)

    def test_shape_preserved_3d(self):
        img = _rgb()
        assert equalize_histogram(img).shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        assert equalize_histogram(img).dtype == img.dtype

    def test_constant_image_ok(self):
        img = _const(100)
        result = equalize_histogram(img)
        assert result.shape == img.shape

    def test_varied_image_output_in_range(self):
        img = _gray()
        result = equalize_histogram(img)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0

    def test_custom_output_range(self):
        cfg = _cfg(output_range=(0.0, 1.0))
        result = equalize_histogram(_gray(), cfg)
        assert result.max() <= 1.5  # float dtype — допуск


# ─── TestStretchContrast ──────────────────────────────────────────────────────

class TestStretchContrast:
    def test_returns_ndarray(self):
        assert isinstance(stretch_contrast(_gray()), np.ndarray)

    def test_shape_preserved_2d(self):
        img = _gray(48, 64)
        assert stretch_contrast(img).shape == (48, 64)

    def test_shape_preserved_3d(self):
        img = _rgb()
        assert stretch_contrast(img).shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        assert stretch_contrast(img).dtype == img.dtype

    def test_constant_image_ok(self):
        img = _const(100)
        result = stretch_contrast(img)
        assert result.shape == img.shape

    def test_output_within_range(self):
        cfg = _cfg(method="stretch")
        result = stretch_contrast(_gray(), cfg)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0

    def test_full_range_clip(self):
        cfg = _cfg(method="stretch", clip_limit=0.0)
        img = _gray()
        result = stretch_contrast(img, cfg)
        assert result.shape == img.shape


# ─── TestApplyGamma ───────────────────────────────────────────────────────────

class TestApplyGamma:
    def test_returns_ndarray(self):
        assert isinstance(apply_gamma(_gray()), np.ndarray)

    def test_shape_preserved(self):
        img = _gray(48, 64)
        assert apply_gamma(img).shape == (48, 64)

    def test_dtype_preserved(self):
        img = _gray()
        assert apply_gamma(img).dtype == img.dtype

    def test_gamma_one_near_identity(self):
        img = _gray()
        cfg = _cfg(method="gamma", gamma=1.0)
        result = apply_gamma(img, cfg)
        assert np.allclose(result.astype(float), img.astype(float), atol=2.0)

    def test_gamma_two_darkens(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        cfg = _cfg(method="gamma", gamma=2.0)
        result = apply_gamma(img, cfg)
        assert result.mean() < img.mean()

    def test_gamma_half_brightens(self):
        img = np.full((32, 32), 50, dtype=np.uint8)
        cfg = _cfg(method="gamma", gamma=0.5)
        result = apply_gamma(img, cfg)
        assert result.mean() > img.mean()

    def test_output_in_range(self):
        cfg = _cfg(method="gamma", gamma=2.0)
        result = apply_gamma(_gray(), cfg)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0


# ─── TestClaheEnhance ─────────────────────────────────────────────────────────

class TestClaheEnhance:
    def test_returns_ndarray(self):
        assert isinstance(clahe_enhance(_gray()), np.ndarray)

    def test_shape_preserved_2d(self):
        img = _gray(48, 64)
        assert clahe_enhance(img).shape == (48, 64)

    def test_shape_preserved_3d(self):
        img = _rgb()
        assert clahe_enhance(img).shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        assert clahe_enhance(img).dtype == img.dtype

    def test_constant_image_ok(self):
        img = _const(100)
        result = clahe_enhance(img)
        assert result.shape == img.shape

    def test_output_in_range(self):
        result = clahe_enhance(_gray())
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0

    def test_custom_tile_size(self):
        cfg = _cfg(method="clahe", tile_size=4)
        result = clahe_enhance(_gray(), cfg)
        assert result.shape == (32, 32)


# ─── TestEnhanceContrast ──────────────────────────────────────────────────────

class TestEnhanceContrast:
    def test_returns_enhance_result(self):
        assert isinstance(enhance_contrast(_gray()), EnhanceResult)

    def test_equalize_method(self):
        r = enhance_contrast(_gray(), _cfg("equalize"))
        assert r.method == "equalize"

    def test_stretch_method(self):
        r = enhance_contrast(_gray(), _cfg("stretch"))
        assert r.method == "stretch"

    def test_gamma_method(self):
        r = enhance_contrast(_gray(), _cfg("gamma"))
        assert r.method == "gamma"

    def test_clahe_method(self):
        r = enhance_contrast(_gray(), _cfg("clahe"))
        assert r.method == "clahe"

    def test_image_ndarray(self):
        r = enhance_contrast(_gray())
        assert isinstance(r.image, np.ndarray)

    def test_shape_preserved(self):
        img = _gray(48, 64)
        r = enhance_contrast(img)
        assert r.image.shape == (48, 64)

    def test_rgb_ok(self):
        r = enhance_contrast(_rgb())
        assert r.image.shape == _rgb().shape

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.zeros(32))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.zeros((4, 4, 3, 2)))

    def test_input_mean_stored(self):
        img = _const(100)
        r = enhance_contrast(img, _cfg("gamma"))
        assert r.input_mean == pytest.approx(100.0, abs=1.0)

    def test_input_std_stored(self):
        img = _const(100)
        r = enhance_contrast(img, _cfg("gamma"))
        assert r.input_std == pytest.approx(0.0, abs=1.0)

    def test_output_mean_stored(self):
        r = enhance_contrast(_gray())
        assert isinstance(r.output_mean, float)

    def test_default_config(self):
        r = enhance_contrast(_gray())
        assert r.method == "equalize"


# ─── TestBatchEnhance ─────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_returns_list(self):
        images = [_gray(seed=i) for i in range(3)]
        assert isinstance(batch_enhance(images), list)

    def test_length_matches(self):
        images = [_gray(seed=i) for i in range(5)]
        assert len(batch_enhance(images)) == 5

    def test_empty_list(self):
        assert batch_enhance([]) == []

    def test_all_enhance_results(self):
        images = [_gray(seed=i) for i in range(3)]
        for r in batch_enhance(images):
            assert isinstance(r, EnhanceResult)

    def test_custom_config(self):
        cfg = _cfg("gamma", gamma=2.0)
        images = [_gray(seed=i) for i in range(2)]
        for r in batch_enhance(images, cfg):
            assert r.method == "gamma"

    def test_rgb_images(self):
        images = [_rgb(seed=i) for i in range(2)]
        for r in batch_enhance(images):
            assert r.image.ndim == 3
