"""Extra tests for puzzle_reconstruction/preprocessing/contrast_enhancer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.contrast_enhancer import (
    EnhanceConfig,
    EnhanceResult,
    apply_gamma,
    batch_enhance,
    clahe_enhance,
    enhance_contrast,
    equalize_histogram,
    stretch_contrast,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(val=128, h=32, w=32):
    return np.full((h, w), val, dtype=np.uint8)


def _cfg(method="equalize", **kw):
    return EnhanceConfig(method=method, **kw)


# ─── TestEnhanceConfigExtra ───────────────────────────────────────────────────

class TestEnhanceConfigExtra:
    def test_gamma_large_valid(self):
        cfg = EnhanceConfig(method="gamma", gamma=10.0)
        assert cfg.gamma == pytest.approx(10.0)

    def test_tile_size_large(self):
        cfg = EnhanceConfig(tile_size=64)
        assert cfg.tile_size == 64

    def test_tile_size_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(tile_size=-1)

    def test_output_range_0_to_1(self):
        cfg = EnhanceConfig(output_range=(0.0, 1.0))
        assert cfg.output_range == (0.0, 1.0)

    def test_output_range_0_to_100(self):
        cfg = EnhanceConfig(output_range=(0.0, 100.0))
        assert cfg.output_range == (0.0, 100.0)

    def test_gamma_one_ok(self):
        cfg = EnhanceConfig(method="gamma", gamma=1.0)
        assert cfg.gamma == pytest.approx(1.0)

    def test_clip_limit_boundary_zero(self):
        cfg = EnhanceConfig(clip_limit=0.0)
        assert cfg.clip_limit == pytest.approx(0.0)

    def test_clip_limit_boundary_one(self):
        cfg = EnhanceConfig(clip_limit=1.0)
        assert cfg.clip_limit == pytest.approx(1.0)


# ─── TestEnhanceResultExtra ───────────────────────────────────────────────────

class TestEnhanceResultExtra:
    def _make(self, input_mean=100.0, output_mean=128.0,
              input_std=30.0, output_std=50.0, method="equalize"):
        return EnhanceResult(
            image=_gray(),
            method=method,
            input_mean=input_mean,
            output_mean=output_mean,
            input_std=input_std,
            output_std=output_std,
        )

    def test_method_stored(self):
        r = self._make(method="clahe")
        assert r.method == "clahe"

    def test_image_ndarray(self):
        r = self._make()
        assert isinstance(r.image, np.ndarray)

    def test_image_shape(self):
        r = self._make()
        assert r.image.shape == (32, 32)

    def test_contrast_gain_no_std_zero(self):
        r = self._make(input_std=0.0, output_std=0.0)
        assert r.contrast_gain == pytest.approx(0.0)

    def test_mean_shift_positive(self):
        r = self._make(input_mean=50.0, output_mean=100.0)
        assert r.mean_shift == pytest.approx(50.0)

    def test_mean_shift_zero(self):
        r = self._make(input_mean=128.0, output_mean=128.0)
        assert r.mean_shift == pytest.approx(0.0)

    def test_contrast_gain_equal_stds(self):
        r = self._make(input_std=20.0, output_std=20.0)
        assert r.contrast_gain == pytest.approx(1.0)


# ─── TestEqualizeHistogramExtra ───────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_non_square_gray(self):
        img = _gray(h=16, w=48)
        result = equalize_histogram(img)
        assert result.shape == (16, 48)

    def test_large_gray(self):
        img = _gray(h=128, w=128, seed=3)
        result = equalize_histogram(img)
        assert result.shape == (128, 128)

    def test_all_black_no_crash(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = equalize_histogram(img)
        assert result.shape == (32, 32)

    def test_all_white_no_crash(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        result = equalize_histogram(img)
        assert result.shape == (32, 32)

    def test_non_square_rgb(self):
        img = _rgb(h=20, w=40)
        result = equalize_histogram(img)
        assert result.shape == (20, 40, 3)

    def test_output_dtype_uint8(self):
        result = equalize_histogram(_gray())
        assert result.dtype == np.uint8


# ─── TestStretchContrastExtra ─────────────────────────────────────────────────

class TestStretchContrastExtra:
    def test_non_square_gray(self):
        img = _gray(h=20, w=60)
        result = stretch_contrast(img)
        assert result.shape == (20, 60)

    def test_all_same_value(self):
        img = _const(50)
        result = stretch_contrast(img)
        assert result.shape == (32, 32)

    def test_clip_limit_half(self):
        cfg = _cfg(method="stretch", clip_limit=0.5)
        result = stretch_contrast(_gray(), cfg)
        assert result.shape == (32, 32)

    def test_large_rgb(self):
        img = _rgb(h=64, w=64, seed=5)
        result = stretch_contrast(img)
        assert result.shape == (64, 64, 3)

    def test_output_in_0_255_range(self):
        result = stretch_contrast(_gray(seed=7))
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0

    def test_dtype_preserved_rgb(self):
        img = _rgb()
        assert stretch_contrast(img).dtype == img.dtype


# ─── TestApplyGammaExtra ─────────────────────────────────────────────────────

class TestApplyGammaExtra:
    def test_non_square_gray(self):
        img = _gray(h=16, w=48)
        result = apply_gamma(img)
        assert result.shape == (16, 48)

    def test_rgb_shape_preserved(self):
        img = _rgb(h=32, w=48)
        result = apply_gamma(img)
        assert result.shape == (32, 48, 3)

    def test_gamma_1_exact_identity(self):
        img = _const(200)
        cfg = _cfg(method="gamma", gamma=1.0)
        result = apply_gamma(img, cfg)
        assert np.allclose(result.astype(float), img.astype(float), atol=1.0)

    def test_gamma_large_darkens_more(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        cfg1 = _cfg(method="gamma", gamma=2.0)
        cfg2 = _cfg(method="gamma", gamma=4.0)
        r1 = apply_gamma(img, cfg1)
        r2 = apply_gamma(img, cfg2)
        assert r2.mean() <= r1.mean()

    def test_output_all_black_stays_black(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        cfg = _cfg(method="gamma", gamma=2.0)
        result = apply_gamma(img, cfg)
        assert result.max() == 0

    def test_output_in_range_rgb(self):
        cfg = _cfg(method="gamma", gamma=0.5)
        result = apply_gamma(_rgb(), cfg)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 255.0


# ─── TestClaheEnhanceExtra ────────────────────────────────────────────────────

class TestClaheEnhanceExtra:
    def test_non_square_gray(self):
        img = _gray(h=24, w=48)
        result = clahe_enhance(img)
        assert result.shape == (24, 48)

    def test_large_image(self):
        img = _gray(h=128, w=128, seed=4)
        result = clahe_enhance(img)
        assert result.shape == (128, 128)

    def test_non_square_rgb(self):
        img = _rgb(h=32, w=64)
        result = clahe_enhance(img)
        assert result.shape == (32, 64, 3)

    def test_tile_size_1(self):
        cfg = _cfg(method="clahe", tile_size=1)
        result = clahe_enhance(_gray(), cfg)
        assert result.shape == (32, 32)

    def test_output_dtype_uint8_rgb(self):
        result = clahe_enhance(_rgb())
        assert result.dtype == np.uint8

    def test_all_black_no_crash(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = clahe_enhance(img)
        assert result.shape == (32, 32)


# ─── TestEnhanceContrastExtra ─────────────────────────────────────────────────

class TestEnhanceContrastExtra:
    def test_non_square_image(self):
        img = _gray(h=24, w=48)
        r = enhance_contrast(img)
        assert r.image.shape == (24, 48)

    def test_large_gray_image(self):
        img = _gray(h=128, w=128, seed=6)
        r = enhance_contrast(img)
        assert r.image.shape == (128, 128)

    def test_all_methods_produce_result(self):
        img = _gray()
        for method in ("equalize", "stretch", "gamma", "clahe"):
            r = enhance_contrast(img, _cfg(method))
            assert isinstance(r, EnhanceResult)
            assert r.method == method

    def test_output_std_nonneg(self):
        r = enhance_contrast(_gray())
        assert r.output_std >= 0.0

    def test_input_output_mean_floats(self):
        r = enhance_contrast(_gray())
        assert isinstance(r.input_mean, float)
        assert isinstance(r.output_mean, float)

    def test_gamma_2_darkens_bright(self):
        img = np.full((32, 32), 200, dtype=np.uint8)
        r = enhance_contrast(img, _cfg("gamma", gamma=2.0))
        assert r.output_mean < r.input_mean

    def test_rgb_large(self):
        img = _rgb(h=64, w=64, seed=2)
        r = enhance_contrast(img)
        assert r.image.shape == (64, 64, 3)


# ─── TestBatchEnhanceExtra ────────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_ten_images(self):
        imgs = [_gray(seed=i) for i in range(10)]
        result = batch_enhance(imgs)
        assert len(result) == 10

    def test_all_methods_batch(self):
        imgs = [_gray(seed=i) for i in range(3)]
        for method in ("equalize", "stretch", "gamma", "clahe"):
            results = batch_enhance(imgs, _cfg(method))
            assert all(r.method == method for r in results)

    def test_mixed_sizes(self):
        imgs = [_gray(h=16, w=16), _gray(h=32, w=32), _gray(h=64, w=64)]
        results = batch_enhance(imgs)
        assert results[0].image.shape == (16, 16)
        assert results[1].image.shape == (32, 32)
        assert results[2].image.shape == (64, 64)

    def test_rgb_images(self):
        imgs = [_rgb(seed=i) for i in range(4)]
        results = batch_enhance(imgs)
        assert all(r.image.ndim == 3 for r in results)

    def test_single_image(self):
        result = batch_enhance([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], EnhanceResult)

    def test_output_mean_in_range(self):
        imgs = [_gray(seed=i) for i in range(5)]
        for r in batch_enhance(imgs):
            assert 0.0 <= r.output_mean <= 255.0
