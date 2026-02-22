"""Тесты для puzzle_reconstruction.preprocessing.color_normalizer."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestNormConfig ───────────────────────────────────────────────────────────

class TestNormConfig:
    def test_defaults(self):
        cfg = NormConfig()
        assert cfg.method == "clahe"
        assert cfg.gamma == pytest.approx(1.0)
        assert cfg.clip_limit == pytest.approx(2.0)
        assert cfg.tile_size == 8

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NormConfig(method="unknown")

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError):
            NormConfig(gamma=0.0)

    def test_gamma_negative_raises(self):
        with pytest.raises(ValueError):
            NormConfig(gamma=-1.0)

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            NormConfig(clip_limit=0.0)

    def test_tile_size_below_2_raises(self):
        with pytest.raises(ValueError):
            NormConfig(tile_size=1)

    def test_target_mean_negative_raises(self):
        with pytest.raises(ValueError):
            NormConfig(target_mean=-1.0)

    def test_target_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            NormConfig(target_mean=256.0)

    def test_valid_all_methods(self):
        for m in ("gamma", "equalize", "clahe", "grey_world", "max_rgb", "minmax"):
            cfg = NormConfig(method=m)
            assert cfg.method == m


# ─── TestNormResult ───────────────────────────────────────────────────────────

class TestNormResult:
    def test_delta_mean(self):
        img = _gray()
        nr = NormResult(image=img, method="clahe",
                        mean_before=100.0, mean_after=120.0)
        assert nr.delta_mean == pytest.approx(20.0)

    def test_negative_mean_before_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            NormResult(image=img, method="clahe",
                       mean_before=-1.0, mean_after=100.0)

    def test_negative_mean_after_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            NormResult(image=img, method="clahe",
                       mean_before=100.0, mean_after=-1.0)

    def test_fields_accessible(self):
        img = _gray()
        nr = NormResult(image=img, method="gamma",
                        mean_before=50.0, mean_after=80.0)
        assert nr.method == "gamma"
        assert nr.image is img


# ─── TestGammaCorrection ──────────────────────────────────────────────────────

class TestGammaCorrection:
    def test_returns_uint8(self):
        img = _gray()
        out = gamma_correction(img, gamma=1.0)
        assert out.dtype == np.uint8

    def test_gamma_one_identity(self):
        img = _gray()
        out = gamma_correction(img, gamma=1.0)
        assert np.allclose(out.astype(float), img.astype(float), atol=1)

    def test_gamma_gt_one_darker(self):
        img = np.full((10, 10), 200, dtype=np.uint8)
        out = gamma_correction(img, gamma=2.0)
        assert out.mean() < img.mean()

    def test_gamma_lt_one_brighter(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        out = gamma_correction(img, gamma=0.5)
        assert out.mean() > img.mean()

    def test_zero_gamma_raises(self):
        with pytest.raises(ValueError):
            gamma_correction(_gray(), gamma=0.0)

    def test_bgr_input(self):
        img = _bgr()
        out = gamma_correction(img, gamma=1.5)
        assert out.dtype == np.uint8


# ─── TestEqualizeHistogram ────────────────────────────────────────────────────

class TestEqualizeHistogram:
    def test_grayscale_returns_uint8(self):
        img = _gray()
        out = equalize_histogram(img)
        assert out.dtype == np.uint8

    def test_grayscale_shape_preserved(self):
        img = _gray(48, 64)
        out = equalize_histogram(img)
        assert out.shape == img.shape

    def test_bgr_returns_uint8(self):
        img = _bgr()
        out = equalize_histogram(img)
        assert out.dtype == np.uint8

    def test_bgr_shape_preserved(self):
        img = _bgr(48, 64)
        out = equalize_histogram(img)
        assert out.shape == img.shape

    def test_uniform_image(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        out = equalize_histogram(img)
        assert out.dtype == np.uint8


# ─── TestApplyClahe ───────────────────────────────────────────────────────────

class TestApplyClahe:
    def test_grayscale_returns_uint8(self):
        img = _gray()
        out = apply_clahe(img)
        assert out.dtype == np.uint8

    def test_grayscale_shape_preserved(self):
        img = _gray(48, 64)
        out = apply_clahe(img)
        assert out.shape == img.shape

    def test_bgr_returns_uint8(self):
        img = _bgr()
        out = apply_clahe(img)
        assert out.dtype == np.uint8

    def test_bgr_shape_preserved(self):
        img = _bgr(48, 64)
        out = apply_clahe(img)
        assert out.shape == img.shape

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=0.0)

    def test_tile_size_below_2_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_size=1)


# ─── TestGreyWorldBalance ─────────────────────────────────────────────────────

class TestGreyWorldBalance:
    def test_returns_uint8(self):
        img = _bgr()
        out = grey_world_balance(img)
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr(48, 64)
        out = grey_world_balance(img)
        assert out.shape == img.shape

    def test_grayscale_raises(self):
        with pytest.raises(ValueError):
            grey_world_balance(_gray())

    def test_neutral_image_unchanged(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = grey_world_balance(img)
        assert np.allclose(out.astype(float), 128.0, atol=2)

    def test_channel_means_equalised(self):
        # After grey world, channel means should be more equal
        img = _bgr()
        out = grey_world_balance(img)
        means_out = out.astype(float).mean(axis=(0, 1))
        spread_out = means_out.max() - means_out.min()
        means_in = img.astype(float).mean(axis=(0, 1))
        spread_in = means_in.max() - means_in.min()
        assert spread_out <= spread_in + 5.0  # tolerance for rounding


# ─── TestMaxRgbBalance ────────────────────────────────────────────────────────

class TestMaxRgbBalance:
    def test_returns_uint8(self):
        img = _bgr()
        out = max_rgb_balance(img)
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr(48, 64)
        out = max_rgb_balance(img)
        assert out.shape == img.shape

    def test_grayscale_raises(self):
        with pytest.raises(ValueError):
            max_rgb_balance(_gray())

    def test_all_channels_max_255(self):
        img = _bgr()
        out = max_rgb_balance(img)
        ch_max = out.astype(float).max(axis=(0, 1))
        assert np.all(ch_max == pytest.approx(255.0, abs=1))


# ─── TestMinmaxNormalize ──────────────────────────────────────────────────────

class TestMinmaxNormalize:
    def test_returns_uint8(self):
        img = _gray()
        out = minmax_normalize(img)
        assert out.dtype == np.uint8

    def test_range_0_255(self):
        img = _gray()
        out = minmax_normalize(img)
        assert out.min() == 0
        assert out.max() == 255

    def test_uniform_returns_zeros(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        out = minmax_normalize(img)
        assert np.all(out == 0)

    def test_bgr_input(self):
        img = _bgr()
        out = minmax_normalize(img)
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(48, 64)
        out = minmax_normalize(img)
        assert out.shape == img.shape


# ─── TestNormalizeImage ───────────────────────────────────────────────────────

class TestNormalizeImage:
    def test_returns_norm_result(self):
        nr = normalize_image(_gray())
        assert isinstance(nr, NormResult)

    def test_image_is_uint8(self):
        nr = normalize_image(_gray())
        assert nr.image.dtype == np.uint8

    def test_method_stored(self):
        cfg = NormConfig(method="gamma", gamma=1.5)
        nr = normalize_image(_gray(), cfg)
        assert nr.method == "gamma"

    def test_all_methods_work_gray(self):
        img = _gray()
        for m in ("gamma", "equalize", "clahe", "minmax"):
            cfg = NormConfig(method=m)
            nr = normalize_image(img, cfg)
            assert nr.image.dtype == np.uint8

    def test_all_methods_work_bgr(self):
        img = _bgr()
        for m in ("gamma", "equalize", "clahe", "grey_world", "max_rgb", "minmax"):
            cfg = NormConfig(method=m)
            nr = normalize_image(img, cfg)
            assert nr.image.dtype == np.uint8

    def test_default_config(self):
        nr = normalize_image(_gray(), None)
        assert isinstance(nr, NormResult)

    def test_mean_before_nonneg(self):
        nr = normalize_image(_gray())
        assert nr.mean_before >= 0.0

    def test_mean_after_nonneg(self):
        nr = normalize_image(_gray())
        assert nr.mean_after >= 0.0


# ─── TestBatchNormalize ───────────────────────────────────────────────────────

class TestBatchNormalize:
    def test_returns_list(self):
        imgs = [_gray() for _ in range(3)]
        result = batch_normalize(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gray() for _ in range(4)]
        result = batch_normalize(imgs)
        assert len(result) == 4

    def test_all_norm_results(self):
        imgs = [_gray() for _ in range(2)]
        result = batch_normalize(imgs)
        assert all(isinstance(r, NormResult) for r in result)

    def test_empty_list(self):
        result = batch_normalize([])
        assert result == []

    def test_custom_config(self):
        cfg = NormConfig(method="equalize")
        imgs = [_gray() for _ in range(2)]
        result = batch_normalize(imgs, cfg)
        assert all(r.method == "equalize" for r in result)
