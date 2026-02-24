"""Extra tests for puzzle_reconstruction/preprocessing/patch_normalizer.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.patch_normalizer import (
    NormalizationParams,
    batch_normalize,
    compute_normalization_stats,
    equalize_histogram,
    normalize_patch,
    standardize_patch,
    stretch_contrast,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient(h: int = 64, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _noisy(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ─── TestNormalizationParamsExtra ────────────────────────────────────────────

class TestNormalizationParamsExtra:
    def test_clip_limit_small(self):
        p = NormalizationParams(clip_limit=0.1)
        assert p.clip_limit == pytest.approx(0.1)

    def test_tile_grid_size_4x4(self):
        p = NormalizationParams(tile_grid_size=(4, 4))
        assert p.tile_grid_size == (4, 4)

    def test_target_mean_0(self):
        p = NormalizationParams(target_mean=0.0)
        assert p.target_mean == pytest.approx(0.0)

    def test_target_std_100(self):
        p = NormalizationParams(target_std=100.0)
        assert p.target_std == pytest.approx(100.0)

    def test_low_pct_0(self):
        p = NormalizationParams(low_pct=0.0)
        assert p.low_pct == pytest.approx(0.0)

    def test_high_pct_100(self):
        p = NormalizationParams(high_pct=100.0)
        assert p.high_pct == pytest.approx(100.0)

    def test_none_method_valid(self):
        p = NormalizationParams(method="none")
        assert p.method == "none"

    def test_stretch_method_valid(self):
        p = NormalizationParams(method="stretch")
        assert p.method == "stretch"


# ─── TestEqualizeHistogramExtra ───────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_various_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = equalize_histogram(img, method="global")
            assert r.dtype == np.uint8

    def test_small_image(self):
        img = _gradient(h=8, w=8)
        r = equalize_histogram(img, method="global")
        assert r.shape == (8, 8)

    def test_clahe_large_clip(self):
        img = _noisy(seed=3)
        r = equalize_histogram(img, method="clahe", clip_limit=10.0)
        assert r.dtype == np.uint8

    def test_global_black_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        r = equalize_histogram(img, method="global")
        assert r.dtype == np.uint8

    def test_clahe_non_square(self):
        img = _gradient(h=32, w=64)
        r = equalize_histogram(img, method="clahe")
        assert r.shape == (32, 64)

    def test_clahe_small_tile_grid(self):
        img = _noisy(seed=2)
        r = equalize_histogram(img, method="clahe", tile_grid_size=(2, 2))
        assert r.shape == img.shape


# ─── TestStretchContrastExtra ─────────────────────────────────────────────────

class TestStretchContrastExtra:
    def test_various_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = stretch_contrast(img)
            assert r.dtype == np.uint8

    def test_narrow_range(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        img[0, 0] = 90
        img[-1, -1] = 110
        r = stretch_contrast(img, low_pct=0.0, high_pct=100.0)
        assert r.dtype == np.uint8

    def test_percentile_5_95(self):
        img = _gradient()
        r = stretch_contrast(img, low_pct=5.0, high_pct=95.0)
        assert r.shape == img.shape

    def test_non_square(self):
        img = _gradient(h=32, w=64)
        r = stretch_contrast(img)
        assert r.shape == (32, 64)

    def test_bgr_non_square(self):
        img = _solid_bgr(150, h=32, w=64)
        r = stretch_contrast(img)
        assert r.ndim == 2


# ─── TestStandardizePatchExtra ────────────────────────────────────────────────

class TestStandardizePatchExtra:
    def test_five_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = standardize_patch(img)
            assert r.dtype == np.uint8

    def test_non_square_input(self):
        img = _gradient(h=32, w=64)
        r = standardize_patch(img)
        assert r.shape == (32, 64)

    def test_target_mean_50(self):
        img = _noisy(seed=0)
        r = standardize_patch(img, target_mean=50.0, target_std=30.0)
        assert r.dtype == np.uint8

    def test_values_in_0_255_range(self):
        img = _noisy(seed=7)
        r = standardize_patch(img, target_mean=128.0, target_std=50.0)
        assert int(r.min()) >= 0
        assert int(r.max()) <= 255

    def test_target_mean_200(self):
        img = _solid_gray(100)
        r = standardize_patch(img, target_mean=200.0, target_std=30.0)
        assert r.dtype == np.uint8


# ─── TestNormalizePatchExtra ──────────────────────────────────────────────────

class TestNormalizePatchExtra:
    def test_non_square_input(self):
        img = _gradient(h=32, w=64)
        r = normalize_patch(img)
        assert r.shape == (32, 64)

    def test_small_image(self):
        img = _noisy(seed=1, h=8, w=8)
        r = normalize_patch(img)
        assert r.dtype == np.uint8

    def test_five_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = normalize_patch(img)
            assert r.dtype == np.uint8

    def test_bgr_non_square(self):
        img = _solid_bgr(150, h=32, w=64)
        r = normalize_patch(img)
        assert r.ndim == 2

    def test_all_methods_output_2d(self):
        img = _gradient(h=32, w=32)
        for method in ("equalize", "clahe", "stretch", "standardize", "none"):
            r = normalize_patch(img, NormalizationParams(method=method))
            assert r.ndim == 2

    def test_all_methods_uint8(self):
        img = _noisy(seed=3)
        for method in ("equalize", "clahe", "stretch", "standardize", "none"):
            r = normalize_patch(img, NormalizationParams(method=method))
            assert r.dtype == np.uint8


# ─── TestBatchNormalizeExtra ──────────────────────────────────────────────────

class TestBatchNormalizeExtra:
    def test_ten_images(self):
        imgs = [_noisy(seed=i) for i in range(10)]
        r = batch_normalize(imgs)
        assert len(r) == 10

    def test_all_images_2d(self):
        imgs = [_gradient() for _ in range(3)]
        r = batch_normalize(imgs)
        for img in r:
            assert img.ndim == 2

    def test_stretch_method(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        params = NormalizationParams(method="stretch")
        r = batch_normalize(imgs, params)
        assert len(r) == 3

    def test_bgr_images_normalized(self):
        imgs = [_solid_bgr(v) for v in (100, 150, 200)]
        r = batch_normalize(imgs)
        for img in r:
            assert img.dtype == np.uint8

    def test_single_image_batch(self):
        imgs = [_gradient()]
        r = batch_normalize(imgs)
        assert len(r) == 1

    def test_standardize_method(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        params = NormalizationParams(method="standardize")
        r = batch_normalize(imgs, params)
        assert len(r) == 3


# ─── TestComputeNormalizationStatsExtra ───────────────────────────────────────

class TestComputeNormalizationStatsExtra:
    def test_single_gradient_image(self):
        imgs = [_gradient()]
        r = compute_normalization_stats(imgs)
        assert r["n_images"] == 1

    def test_ten_images(self):
        imgs = [_noisy(seed=i) for i in range(10)]
        r = compute_normalization_stats(imgs)
        assert r["n_images"] == 10

    def test_min_lte_mean(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        r = compute_normalization_stats(imgs)
        assert r["min"] <= r["mean"] + 1e-9

    def test_mean_lte_max(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        r = compute_normalization_stats(imgs)
        assert r["mean"] <= r["max"] + 1e-9

    def test_std_nonneg(self):
        imgs = [_gradient(), _noisy(seed=1)]
        r = compute_normalization_stats(imgs)
        assert r["std"] >= 0.0

    def test_all_keys_present(self):
        imgs = [_gradient()]
        r = compute_normalization_stats(imgs)
        for key in ("mean", "std", "min", "max", "p2", "p98", "n_images"):
            assert key in r

    def test_mixed_types(self):
        imgs = [_gradient(), _solid_gray(200), _noisy(seed=5)]
        r = compute_normalization_stats(imgs)
        assert r["n_images"] == 3
