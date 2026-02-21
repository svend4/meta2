"""
Тесты для puzzle_reconstruction.preprocessing.patch_normalizer.
"""
import pytest
import numpy as np
import cv2

from puzzle_reconstruction.preprocessing.patch_normalizer import (
    NormalizationParams,
    equalize_histogram,
    stretch_contrast,
    standardize_patch,
    normalize_patch,
    batch_normalize,
    compute_normalization_stats,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_gray(h: int = 64, w: int = 64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _noisy_gray(seed: int = 42, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ─── NormalizationParams ──────────────────────────────────────────────────────

class TestNormalizationParams:
    def test_default_method_clahe(self):
        p = NormalizationParams()
        assert p.method == "clahe"

    def test_default_clip_limit(self):
        p = NormalizationParams()
        assert p.clip_limit == pytest.approx(2.0)

    def test_default_tile_grid(self):
        p = NormalizationParams()
        assert p.tile_grid_size == (8, 8)

    def test_default_target_mean(self):
        p = NormalizationParams()
        assert p.target_mean == pytest.approx(128.0)

    def test_default_target_std(self):
        p = NormalizationParams()
        assert p.target_std == pytest.approx(50.0)

    def test_default_percentiles(self):
        p = NormalizationParams()
        assert p.low_pct  == pytest.approx(2.0)
        assert p.high_pct == pytest.approx(98.0)

    def test_valid_methods_accepted(self):
        for m in ("equalize", "clahe", "stretch", "standardize", "none"):
            p = NormalizationParams(method=m)
            assert p.method == m

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            NormalizationParams(method="magic")

    def test_custom_params(self):
        p = NormalizationParams(
            method="clahe",
            clip_limit=4.0,
            tile_grid_size=(4, 4),
            target_mean=100.0,
            target_std=30.0,
            low_pct=5.0,
            high_pct=95.0,
        )
        assert p.clip_limit      == pytest.approx(4.0)
        assert p.tile_grid_size  == (4, 4)
        assert p.target_mean     == pytest.approx(100.0)
        assert p.target_std      == pytest.approx(30.0)
        assert p.low_pct         == pytest.approx(5.0)
        assert p.high_pct        == pytest.approx(95.0)


# ─── equalize_histogram ───────────────────────────────────────────────────────

class TestEqualizeHistogram:
    def test_global_returns_uint8(self):
        img = _gradient_gray()
        r   = equalize_histogram(img, method="global")
        assert r.dtype == np.uint8

    def test_global_grayscale_output(self):
        img = _gradient_gray()
        r   = equalize_histogram(img, method="global")
        assert r.ndim == 2

    def test_global_bgr_input(self):
        img = _solid_bgr(150)
        r   = equalize_histogram(img, method="global")
        assert r.dtype == np.uint8
        assert r.ndim  == 2

    def test_clahe_returns_uint8(self):
        img = _noisy_gray()
        r   = equalize_histogram(img, method="clahe")
        assert r.dtype == np.uint8

    def test_clahe_same_shape(self):
        img = _gradient_gray(h=48, w=80)
        r   = equalize_histogram(img, method="clahe")
        assert r.shape == (48, 80)

    def test_global_same_shape(self):
        img = _gradient_gray(h=48, w=80)
        r   = equalize_histogram(img, method="global")
        assert r.shape == (48, 80)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            equalize_histogram(_solid_gray(100), method="unknown")

    def test_uniform_image_equalized(self):
        img = _solid_gray(128)
        r   = equalize_histogram(img, method="global")
        assert r.dtype == np.uint8

    def test_clahe_custom_params(self):
        img = _noisy_gray(seed=1)
        r   = equalize_histogram(img, method="clahe",
                                  clip_limit=4.0, tile_grid_size=(4, 4))
        assert r.shape == img.shape
        assert r.dtype == np.uint8

    def test_gradient_equalized_non_constant(self):
        img = _gradient_gray()
        r   = equalize_histogram(img, method="global")
        # Результат не должен быть однородным
        assert r.max() > r.min()


# ─── stretch_contrast ─────────────────────────────────────────────────────────

class TestStretchContrast:
    def test_returns_uint8(self):
        img = _gradient_gray()
        r   = stretch_contrast(img)
        assert r.dtype == np.uint8

    def test_grayscale_output(self):
        img = _gradient_gray()
        r   = stretch_contrast(img)
        assert r.ndim == 2

    def test_bgr_input(self):
        img = _solid_bgr(150)
        r   = stretch_contrast(img)
        assert r.dtype == np.uint8
        assert r.ndim  == 2

    def test_same_shape(self):
        img = _gradient_gray(h=48, w=80)
        r   = stretch_contrast(img)
        assert r.shape == (48, 80)

    def test_low_ge_high_raises(self):
        with pytest.raises(ValueError):
            stretch_contrast(_solid_gray(100), low_pct=50.0, high_pct=50.0)

    def test_low_gt_high_raises(self):
        with pytest.raises(ValueError):
            stretch_contrast(_solid_gray(100), low_pct=60.0, high_pct=40.0)

    def test_full_range_image_max_255(self):
        img = _gradient_gray()
        r   = stretch_contrast(img, low_pct=0.0, high_pct=100.0)
        # Градиент занимает полный диапазон
        assert int(r.max()) == 255

    def test_uniform_image_no_crash(self):
        img = _solid_gray(128)
        r   = stretch_contrast(img)
        assert r.dtype == np.uint8

    def test_stretched_max_ge_original_max(self):
        img = np.full((64, 64), 100, dtype=np.uint8)
        # Добавим небольшой диапазон
        img[0, 0] = 80; img[-1, -1] = 120
        r = stretch_contrast(img, low_pct=1.0, high_pct=99.0)
        assert int(r.max()) >= 100


# ─── standardize_patch ────────────────────────────────────────────────────────

class TestStandardizePatch:
    def test_returns_uint8(self):
        img = _gradient_gray()
        r   = standardize_patch(img)
        assert r.dtype == np.uint8

    def test_grayscale_output(self):
        img = _gradient_gray()
        r   = standardize_patch(img)
        assert r.ndim == 2

    def test_bgr_input(self):
        img = _solid_bgr(100)
        r   = standardize_patch(img)
        assert r.dtype == np.uint8
        assert r.ndim  == 2

    def test_same_shape(self):
        img = _gradient_gray(h=32, w=64)
        r   = standardize_patch(img)
        assert r.shape == (32, 64)

    def test_uniform_image_returns_target_mean(self):
        img = _solid_gray(200)
        r   = standardize_patch(img, target_mean=100.0, target_std=30.0)
        # Все пиксели должны быть ~100
        assert np.abs(float(r.mean()) - 100.0) < 2.0

    def test_output_mean_near_target(self):
        img = _noisy_gray(seed=7)
        r   = standardize_patch(img, target_mean=128.0, target_std=40.0)
        assert abs(float(r.astype(np.float64).mean()) - 128.0) < 10.0

    def test_values_clipped_to_0_255(self):
        img = _gradient_gray()
        r   = standardize_patch(img, target_mean=128.0, target_std=200.0)
        assert r.min() >= 0
        assert r.max() <= 255

    def test_custom_target_mean(self):
        img = _noisy_gray(seed=1)
        r   = standardize_patch(img, target_mean=50.0, target_std=20.0)
        assert r.dtype == np.uint8


# ─── normalize_patch ──────────────────────────────────────────────────────────

class TestNormalizePatch:
    def test_default_params_clahe(self):
        img = _gradient_gray()
        r   = normalize_patch(img)
        assert r.dtype == np.uint8
        assert r.ndim  == 2

    def test_none_method_returns_uint8_gray(self):
        img = _gradient_gray()
        r   = normalize_patch(img, NormalizationParams(method="none"))
        assert r.dtype == np.uint8

    def test_equalize_method(self):
        img = _gradient_gray()
        r   = normalize_patch(img, NormalizationParams(method="equalize"))
        assert r.dtype == np.uint8

    def test_clahe_method(self):
        img = _gradient_gray()
        r   = normalize_patch(img, NormalizationParams(method="clahe"))
        assert r.dtype == np.uint8

    def test_stretch_method(self):
        img = _gradient_gray()
        r   = normalize_patch(img, NormalizationParams(method="stretch"))
        assert r.dtype == np.uint8

    def test_standardize_method(self):
        img = _gradient_gray()
        r   = normalize_patch(img, NormalizationParams(method="standardize"))
        assert r.dtype == np.uint8

    @pytest.mark.parametrize("method", ["equalize", "clahe", "stretch", "standardize", "none"])
    def test_shape_preserved(self, method):
        img = _gradient_gray(h=48, w=64)
        r   = normalize_patch(img, NormalizationParams(method=method))
        assert r.shape == (48, 64)

    def test_bgr_input_all_methods(self):
        img = _solid_bgr(150)
        for method in ("equalize", "clahe", "stretch", "standardize", "none"):
            r = normalize_patch(img, NormalizationParams(method=method))
            assert r.ndim  == 2
            assert r.dtype == np.uint8


# ─── batch_normalize ──────────────────────────────────────────────────────────

class TestBatchNormalize:
    def test_length_preserved(self):
        imgs = [_gradient_gray() for _ in range(5)]
        r    = batch_normalize(imgs)
        assert len(r) == 5

    def test_all_uint8(self):
        imgs = [_noisy_gray(seed=i) for i in range(3)]
        r    = batch_normalize(imgs)
        for img in r:
            assert img.dtype == np.uint8

    def test_empty_list_returns_empty(self):
        r = batch_normalize([])
        assert r == []

    def test_custom_params(self):
        imgs   = [_gradient_gray() for _ in range(3)]
        params = NormalizationParams(method="stretch", low_pct=5.0, high_pct=95.0)
        r      = batch_normalize(imgs, params)
        assert len(r) == 3

    def test_shapes_preserved(self):
        imgs = [_gradient_gray(h=32, w=64) for _ in range(3)]
        r    = batch_normalize(imgs)
        for img in r:
            assert img.shape == (32, 64)


# ─── compute_normalization_stats ──────────────────────────────────────────────

class TestComputeNormalizationStats:
    def test_returns_dict_with_keys(self):
        imgs = [_gradient_gray(), _noisy_gray()]
        r    = compute_normalization_stats(imgs)
        for key in ("mean", "std", "min", "max", "p2", "p98", "n_images"):
            assert key in r

    def test_n_images_correct(self):
        imgs = [_gradient_gray() for _ in range(5)]
        r    = compute_normalization_stats(imgs)
        assert r["n_images"] == 5

    def test_min_le_p2_le_mean_le_p98_le_max(self):
        imgs = [_gradient_gray(), _noisy_gray(seed=3)]
        r    = compute_normalization_stats(imgs)
        assert r["min"]  <= r["p2"]   + 1e-9
        assert r["p2"]   <= r["mean"] + 1e-9
        assert r["mean"] <= r["p98"]  + 1e-9
        assert r["p98"]  <= r["max"]  + 1e-9

    def test_uniform_image_zero_std(self):
        imgs = [_solid_gray(100), _solid_gray(100)]
        r    = compute_normalization_stats(imgs)
        assert r["std"] == pytest.approx(0.0, abs=1e-6)

    def test_uniform_image_mean_equals_value(self):
        imgs = [_solid_gray(200)]
        r    = compute_normalization_stats(imgs)
        assert r["mean"] == pytest.approx(200.0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            compute_normalization_stats([])

    def test_bgr_images_accepted(self):
        imgs = [_solid_bgr(100), _solid_bgr(200)]
        r    = compute_normalization_stats(imgs)
        assert r["n_images"] == 2

    def test_std_positive_for_gradient(self):
        imgs = [_gradient_gray()]
        r    = compute_normalization_stats(imgs)
        assert r["std"] > 0.0
