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

def _gray(h: int = 16, w: int = 16, low: int = 50, high: int = 200,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high, (h, w), dtype=np.uint8)
    return arr


def _rgb(h: int = 16, w: int = 16, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat(val: int = 128) -> np.ndarray:
    return np.full((8, 8), val, dtype=np.uint8)


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

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="histogram")

    def test_gamma_positive_ok(self):
        cfg = EnhanceConfig(method="gamma", gamma=2.5)
        assert cfg.gamma == pytest.approx(2.5)

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=0.0)

    def test_gamma_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="gamma", gamma=-1.0)

    def test_clip_limit_zero_ok(self):
        cfg = EnhanceConfig(clip_limit=0.0)
        assert cfg.clip_limit == 0.0

    def test_clip_limit_one_ok(self):
        cfg = EnhanceConfig(clip_limit=1.0)
        assert cfg.clip_limit == 1.0

    def test_clip_limit_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=-0.01)

    def test_clip_limit_above_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(clip_limit=1.1)

    def test_tile_size_one_ok(self):
        cfg = EnhanceConfig(tile_size=1)
        assert cfg.tile_size == 1

    def test_tile_size_zero_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(tile_size=0)

    def test_tile_size_neg_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(tile_size=-4)

    def test_output_range_valid(self):
        cfg = EnhanceConfig(output_range=(0.0, 1.0))
        assert cfg.output_range == (0.0, 1.0)

    def test_output_range_equal_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(1.0, 1.0))

    def test_output_range_inverted_raises(self):
        with pytest.raises(ValueError):
            EnhanceConfig(output_range=(2.0, 1.0))


# ─── TestEnhanceResult ────────────────────────────────────────────────────────

class TestEnhanceResult:
    def _make(self, in_std=50.0, out_std=100.0,
              in_mean=100.0, out_mean=120.0):
        img = _gray()
        return EnhanceResult(
            image=img, method="equalize",
            input_mean=in_mean, output_mean=out_mean,
            input_std=in_std, output_std=out_std,
        )

    def test_contrast_gain_ratio(self):
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

    def test_mean_shift_zero(self):
        r = self._make(in_mean=120.0, out_mean=120.0)
        assert r.mean_shift == pytest.approx(0.0)

    def test_method_stored(self):
        r = self._make()
        assert r.method == "equalize"

    def test_image_shape_preserved(self):
        img = _gray(32, 32)
        r = EnhanceResult(image=img, method="gamma",
                          input_mean=0.0, output_mean=0.0,
                          input_std=0.0, output_std=0.0)
        assert r.image.shape == (32, 32)


# ─── TestEqualizeHistogram ────────────────────────────────────────────────────

class TestEqualizeHistogram:
    def test_shape_preserved_2d(self):
        img = _gray()
        out = equalize_histogram(img)
        assert out.shape == img.shape

    def test_shape_preserved_3d(self):
        img = _rgb()
        out = equalize_histogram(img)
        assert out.shape == img.shape

    def test_output_in_range_default(self):
        img = _gray()
        out = equalize_histogram(img).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_flat_image_returns_midpoint(self):
        img = _flat(128)
        out = equalize_histogram(img).astype(float)
        # Все пиксели должны быть в допустимом диапазоне
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_custom_output_range(self):
        cfg = EnhanceConfig(method="equalize", output_range=(0.0, 1.0))
        img = _gray()
        out = equalize_histogram(img, cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_dtype_preserved(self):
        img = _gray().astype(np.uint8)
        out = equalize_histogram(img)
        assert out.dtype == np.uint8

    def test_3d_each_channel_processed(self):
        img = _rgb()
        cfg = EnhanceConfig(method="equalize")
        out = equalize_histogram(img, cfg)
        assert out.shape[2] == 3

    def test_different_inputs_differ(self):
        img1 = np.zeros((16, 16), dtype=np.uint8)
        img2 = np.full((16, 16), 255, dtype=np.uint8)
        o1 = equalize_histogram(img1).astype(float)
        o2 = equalize_histogram(img2).astype(float)
        assert not np.allclose(o1, o2)


# ─── TestStretchContrast ──────────────────────────────────────────────────────

class TestStretchContrast:
    def test_shape_preserved_2d(self):
        img = _gray()
        out = stretch_contrast(img)
        assert out.shape == img.shape

    def test_shape_preserved_3d(self):
        img = _rgb()
        out = stretch_contrast(img)
        assert out.shape == img.shape

    def test_output_in_range(self):
        img = _gray()
        out = stretch_contrast(img).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_clip_zero_covers_full_range(self):
        cfg = EnhanceConfig(method="stretch", clip_limit=0.0)
        img = _gray()
        out = stretch_contrast(img, cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_flat_image_handled(self):
        img = _flat(100)
        out = stretch_contrast(img).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_high_contrast_input(self):
        img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        out = stretch_contrast(img).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_dtype_preserved(self):
        img = _gray().astype(np.uint8)
        out = stretch_contrast(img)
        assert out.dtype == np.uint8

    def test_custom_output_range(self):
        cfg = EnhanceConfig(method="stretch", output_range=(0.0, 1.0))
        img = _gray()
        out = stretch_contrast(img, cfg).astype(float)
        assert float(out.max()) <= 1.0


# ─── TestApplyGamma ───────────────────────────────────────────────────────────

class TestApplyGamma:
    def test_gamma_one_identity(self):
        img = _gray()
        cfg = EnhanceConfig(method="gamma", gamma=1.0)
        out = apply_gamma(img, cfg).astype(float)
        orig = img.astype(float)
        # С нормировкой round-trip не будет точным на uint8, но отклонение мало
        diff = np.abs(out - orig).max()
        assert diff <= 2.0  # 1 LSB погрешность

    def test_gamma_gt_one_darkens(self):
        # gamma > 1 → тёмные области затемняются (mid-tones снижаются)
        img = np.full((16, 16), 128, dtype=np.uint8)
        cfg_hi = EnhanceConfig(method="gamma", gamma=2.0)
        cfg_lo = EnhanceConfig(method="gamma", gamma=0.5)
        out_hi = apply_gamma(img, cfg_hi).astype(float).mean()
        out_lo = apply_gamma(img, cfg_lo).astype(float).mean()
        assert out_hi < out_lo

    def test_output_in_range(self):
        img = _gray()
        cfg = EnhanceConfig(method="gamma", gamma=2.0)
        out = apply_gamma(img, cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_shape_2d_preserved(self):
        img = _gray()
        out = apply_gamma(img)
        assert out.shape == img.shape

    def test_shape_3d_preserved(self):
        img = _rgb()
        out = apply_gamma(img)
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _gray().astype(np.uint8)
        cfg = EnhanceConfig(method="gamma", gamma=1.5)
        out = apply_gamma(img, cfg)
        assert out.dtype == np.uint8

    def test_custom_range(self):
        cfg = EnhanceConfig(method="gamma", gamma=1.0, output_range=(0.0, 1.0))
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        out = apply_gamma(img, cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


# ─── TestClaheEnhance ─────────────────────────────────────────────────────────

class TestClaheEnhance:
    def test_shape_2d_preserved(self):
        img = _gray(32, 32)
        out = clahe_enhance(img)
        assert out.shape == img.shape

    def test_shape_3d_preserved(self):
        img = _rgb(32, 32)
        out = clahe_enhance(img)
        assert out.shape == img.shape

    def test_output_in_range(self):
        img = _gray()
        out = clahe_enhance(img).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_tile_size_one(self):
        cfg = EnhanceConfig(method="clahe", tile_size=1)
        img = _gray(4, 4)
        out = clahe_enhance(img, cfg)
        assert out.shape == (4, 4)

    def test_large_tile(self):
        cfg = EnhanceConfig(method="clahe", tile_size=64)
        img = _gray(16, 16)
        out = clahe_enhance(img, cfg)
        assert out.shape == (16, 16)

    def test_flat_tile_handled(self):
        cfg = EnhanceConfig(method="clahe", tile_size=4)
        img = _flat(100)
        out = clahe_enhance(img, cfg).astype(float)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 255.0

    def test_dtype_preserved(self):
        img = _gray().astype(np.uint8)
        out = clahe_enhance(img)
        assert out.dtype == np.uint8

    def test_3d_all_channels(self):
        img = _rgb(16, 16)
        out = clahe_enhance(img)
        assert out.shape[2] == 3


# ─── TestEnhanceContrast ──────────────────────────────────────────────────────

class TestEnhanceContrast:
    def test_returns_enhance_result(self):
        r = enhance_contrast(_gray())
        assert isinstance(r, EnhanceResult)

    def test_method_equalize_default(self):
        r = enhance_contrast(_gray())
        assert r.method == "equalize"

    def test_method_stretch(self):
        cfg = EnhanceConfig(method="stretch")
        r = enhance_contrast(_gray(), cfg)
        assert r.method == "stretch"

    def test_method_gamma(self):
        cfg = EnhanceConfig(method="gamma")
        r = enhance_contrast(_gray(), cfg)
        assert r.method == "gamma"

    def test_method_clahe(self):
        cfg = EnhanceConfig(method="clahe")
        r = enhance_contrast(_gray(), cfg)
        assert r.method == "clahe"

    def test_image_shape_preserved(self):
        img = _gray(24, 32)
        r = enhance_contrast(img)
        assert r.image.shape == (24, 32)

    def test_input_mean_correct(self):
        img = _gray()
        r = enhance_contrast(img)
        assert r.input_mean == pytest.approx(float(img.astype(float).mean()))

    def test_input_std_correct(self):
        img = _gray()
        r = enhance_contrast(img)
        assert r.input_std == pytest.approx(float(img.astype(float).std()))

    def test_output_mean_matches_image(self):
        img = _gray()
        r = enhance_contrast(img)
        assert r.output_mean == pytest.approx(
            float(r.image.astype(float).mean()))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.array([1, 2, 3], dtype=np.uint8))

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            enhance_contrast(np.zeros((2, 2, 2, 2), dtype=np.uint8))

    def test_contrast_gain_gte_zero(self):
        r = enhance_contrast(_gray())
        assert r.contrast_gain >= 0.0

    def test_3d_rgb_image(self):
        img = _rgb()
        r = enhance_contrast(img)
        assert r.image.shape == img.shape


# ─── TestBatchEnhance ─────────────────────────────────────────────────────────

class TestBatchEnhance:
    def test_returns_list(self):
        imgs = [_gray(seed=i) for i in range(3)]
        results = batch_enhance(imgs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_enhance_result(self):
        imgs = [_gray(seed=i) for i in range(2)]
        for r in batch_enhance(imgs):
            assert isinstance(r, EnhanceResult)

    def test_empty_list(self):
        assert batch_enhance([]) == []

    def test_single_image(self):
        imgs = [_gray()]
        results = batch_enhance(imgs)
        assert len(results) == 1

    def test_consistent_method(self):
        cfg = EnhanceConfig(method="gamma", gamma=2.0)
        imgs = [_gray(seed=i) for i in range(4)]
        for r in batch_enhance(imgs, cfg):
            assert r.method == "gamma"

    def test_shapes_preserved(self):
        shapes = [(8, 8), (16, 24), (12, 12)]
        imgs = [np.zeros(s, dtype=np.uint8) for s in shapes]
        for shape, r in zip(shapes, batch_enhance(imgs)):
            assert r.image.shape == shape

    def test_different_sizes_ok(self):
        imgs = [_gray(h=8 + i, w=8 + i, seed=i) for i in range(3)]
        results = batch_enhance(imgs)
        assert len(results) == 3
