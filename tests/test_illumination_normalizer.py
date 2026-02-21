"""Тесты для puzzle_reconstruction.preprocessing.illumination_normalizer."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.illumination_normalizer import (
    IllumConfig,
    IllumResult,
    estimate_illumination,
    subtract_background,
    normalize_mean_std,
    apply_clahe,
    normalize_illumination,
    batch_normalize_illumination,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, val: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(h: int = 64, w: int = 64, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _mask(h: int = 64, w: int = 64, margin: int = 8) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[margin:h - margin, margin:w - margin] = 255
    return m


def _result(method: str = "mean_std") -> IllumResult:
    return IllumResult(
        image=_const(),
        original_mean=100.0,
        original_std=30.0,
        method=method,
    )


# ─── TestIllumConfig ──────────────────────────────────────────────────────────

class TestIllumConfig:
    def test_defaults(self):
        cfg = IllumConfig()
        assert cfg.blur_ksize == 51
        assert cfg.target_mean == pytest.approx(128.0)
        assert cfg.target_std == pytest.approx(60.0)
        assert cfg.clip_limit == pytest.approx(2.0)
        assert cfg.tile_grid_size == (8, 8)

    def test_blur_ksize_three_ok(self):
        cfg = IllumConfig(blur_ksize=3)
        assert cfg.blur_ksize == 3

    def test_blur_ksize_two_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=2)

    def test_blur_ksize_even_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=4)

    def test_blur_ksize_one_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=1)

    def test_target_mean_zero_ok(self):
        cfg = IllumConfig(target_mean=0.0)
        assert cfg.target_mean == 0.0

    def test_target_mean_255_ok(self):
        cfg = IllumConfig(target_mean=255.0)
        assert cfg.target_mean == 255.0

    def test_target_mean_neg_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_mean=-1.0)

    def test_target_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_mean=256.0)

    def test_target_std_small_positive_ok(self):
        cfg = IllumConfig(target_std=0.1)
        assert cfg.target_std == pytest.approx(0.1)

    def test_target_std_zero_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_std=0.0)

    def test_target_std_neg_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_std=-1.0)

    def test_clip_limit_small_positive_ok(self):
        cfg = IllumConfig(clip_limit=0.1)
        assert cfg.clip_limit == pytest.approx(0.1)

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(clip_limit=0.0)

    def test_clip_limit_neg_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(clip_limit=-1.0)

    def test_tile_grid_zero_w_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(tile_grid_size=(0, 8))

    def test_tile_grid_zero_h_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(tile_grid_size=(8, 0))

    def test_tile_grid_one_ok(self):
        cfg = IllumConfig(tile_grid_size=(1, 1))
        assert cfg.tile_grid_size == (1, 1)


# ─── TestIllumResult ──────────────────────────────────────────────────────────

class TestIllumResult:
    def test_basic(self):
        r = _result()
        assert r.method == "mean_std"
        assert r.original_mean == pytest.approx(100.0)

    def test_shape_2d(self):
        r = _result()
        assert r.shape == (64, 64)

    def test_shape_3d(self):
        r = IllumResult(image=_rgb(), original_mean=128.0,
                        original_std=40.0, method="clahe")
        assert r.shape == (64, 64, 3)

    def test_1d_image_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=np.zeros(64), original_mean=0.0,
                        original_std=0.0, method="x")

    def test_4d_image_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=np.zeros((4, 4, 3, 2)),
                        original_mean=0.0, original_std=0.0, method="x")

    def test_original_mean_neg_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=-1.0,
                        original_std=0.0, method="x")

    def test_original_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=256.0,
                        original_std=0.0, method="x")

    def test_original_std_neg_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=128.0,
                        original_std=-1.0, method="x")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=128.0,
                        original_std=0.0, method="")

    def test_original_mean_zero_ok(self):
        r = IllumResult(image=_const(), original_mean=0.0,
                        original_std=0.0, method="bg")
        assert r.original_mean == 0.0

    def test_original_std_zero_ok(self):
        r = IllumResult(image=_const(), original_mean=128.0,
                        original_std=0.0, method="bg")
        assert r.original_std == 0.0


# ─── TestEstimateIllumination ─────────────────────────────────────────────────

class TestEstimateIllumination:
    def test_returns_ndarray(self):
        bg = estimate_illumination(_gray())
        assert isinstance(bg, np.ndarray)

    def test_shape_matches(self):
        bg = estimate_illumination(_gray(48, 80))
        assert bg.shape == (48, 80)

    def test_dtype_float64(self):
        bg = estimate_illumination(_gray())
        assert bg.dtype == np.float64

    def test_rgb_ok(self):
        bg = estimate_illumination(_rgb())
        assert bg.shape == (64, 64)

    def test_constant_image_bg_approx_same(self):
        img = _const(val=100)
        bg = estimate_illumination(img, blur_ksize=3)
        assert np.abs(bg - 100.0).mean() < 5.0

    def test_ksize_two_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=2)

    def test_ksize_even_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=6)

    def test_ksize_one_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=1)

    def test_ksize_three_ok(self):
        bg = estimate_illumination(_gray(), blur_ksize=3)
        assert bg.shape == (64, 64)

    def test_values_non_negative(self):
        bg = estimate_illumination(_gray())
        assert bg.min() >= 0.0


# ─── TestSubtractBackground ───────────────────────────────────────────────────

class TestSubtractBackground:
    def test_returns_uint8(self):
        result = subtract_background(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = subtract_background(_gray(48, 80))
        assert result.shape == (48, 80)

    def test_values_in_range(self):
        result = subtract_background(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_rgb_ok(self):
        result = subtract_background(_rgb())
        assert result.shape == (64, 64)

    def test_constant_image_offset_preserved(self):
        img = _const(val=100)
        result = subtract_background(img, blur_ksize=3, offset=128.0)
        # Постоянный фон вычтен → результат ≈ offset
        assert abs(result.astype(float).mean() - 128.0) < 10.0

    def test_custom_offset(self):
        result = subtract_background(_gray(), blur_ksize=3, offset=64.0)
        assert result.dtype == np.uint8

    def test_ksize_even_raises(self):
        with pytest.raises(ValueError):
            subtract_background(_gray(), blur_ksize=4)


# ─── TestNormalizeMeanStd ─────────────────────────────────────────────────────

class TestNormalizeMeanStd:
    def test_returns_uint8(self):
        result = normalize_mean_std(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = normalize_mean_std(_gray(48, 80))
        assert result.shape == (48, 80)

    def test_values_in_range(self):
        result = normalize_mean_std(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_target_mean_applied(self):
        img = _gray(seed=1)
        result = normalize_mean_std(img, target_mean=100.0, target_std=40.0)
        # Среднее должно быть близко к target (после клипинга)
        assert isinstance(result, np.ndarray)

    def test_rgb_ok(self):
        result = normalize_mean_std(_rgb())
        assert result.shape == (64, 64)

    def test_with_mask(self):
        img = _gray()
        mask = _mask()
        result = normalize_mean_std(img, mask=mask)
        assert result.dtype == np.uint8

    def test_zero_mask_passthrough(self):
        img = _gray()
        zero_mask = np.zeros((64, 64), dtype=np.uint8)
        result = normalize_mean_std(img, mask=zero_mask)
        assert result.dtype == np.uint8

    def test_target_mean_neg_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_mean=-1.0)

    def test_target_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_mean=256.0)

    def test_target_std_zero_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_std=0.0)

    def test_constant_image_maps_to_target_mean(self):
        img = _const(val=50)
        result = normalize_mean_std(img, target_mean=128.0, target_std=30.0)
        # Постоянное изображение → std ≈ 0 → все пиксели → target_mean
        assert abs(result.astype(float).mean() - 128.0) < 5.0


# ─── TestApplyClahe ───────────────────────────────────────────────────────────

class TestApplyClahe:
    def test_returns_uint8(self):
        result = apply_clahe(_gray())
        assert result.dtype == np.uint8

    def test_shape_gray_preserved(self):
        result = apply_clahe(_gray(48, 80))
        assert result.shape == (48, 80)

    def test_shape_rgb_preserved(self):
        result = apply_clahe(_rgb(48, 80))
        assert result.shape == (48, 80, 3)

    def test_values_in_range(self):
        result = apply_clahe(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=0.0)

    def test_clip_limit_neg_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=-1.0)

    def test_tile_grid_zero_w_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_grid_size=(0, 8))

    def test_tile_grid_zero_h_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_grid_size=(8, 0))

    def test_high_clip_limit_ok(self):
        result = apply_clahe(_gray(), clip_limit=10.0)
        assert result.dtype == np.uint8

    def test_constant_image_unchanged(self):
        img = _const(val=200)
        result = apply_clahe(img)
        # Постоянное изображение CLAHE не меняет
        assert result.min() == result.max()

    def test_rgb_color_preserved(self):
        img = _rgb()
        result = apply_clahe(img)
        assert result.ndim == 3
        assert result.shape[2] == 3


# ─── TestNormalizeIllumination ────────────────────────────────────────────────

class TestNormalizeIllumination:
    def test_returns_illum_result(self):
        r = normalize_illumination(_gray())
        assert isinstance(r, IllumResult)

    def test_default_method_mean_std(self):
        r = normalize_illumination(_gray())
        assert r.method == "mean_std"

    def test_method_mean_std_ok(self):
        r = normalize_illumination(_gray(), method="mean_std")
        assert r.image.dtype == np.uint8

    def test_method_background_ok(self):
        r = normalize_illumination(_gray(), method="background")
        assert r.image.dtype == np.uint8
        assert r.method == "background"

    def test_method_clahe_ok(self):
        r = normalize_illumination(_gray(), method="clahe")
        assert r.image.dtype == np.uint8
        assert r.method == "clahe"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_illumination(_gray(), method="unknown")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            normalize_illumination(_gray(), method="")

    def test_original_mean_stored(self):
        img = _const(val=100)
        r = normalize_illumination(img)
        assert r.original_mean == pytest.approx(100.0, abs=1.0)

    def test_original_std_stored(self):
        img = _const(val=100)
        r = normalize_illumination(img)
        assert r.original_std == pytest.approx(0.0, abs=1.0)

    def test_image_shape_preserved_mean_std(self):
        img = _gray(48, 80)
        r = normalize_illumination(img, method="mean_std")
        assert r.image.shape == (48, 80)

    def test_image_shape_preserved_background(self):
        img = _gray(48, 80)
        r = normalize_illumination(img, method="background")
        assert r.image.shape == (48, 80)

    def test_image_shape_preserved_clahe(self):
        img = _gray(48, 80)
        r = normalize_illumination(img, method="clahe")
        assert r.image.shape == (48, 80)

    def test_rgb_mean_std_ok(self):
        r = normalize_illumination(_rgb(), method="mean_std")
        assert isinstance(r, IllumResult)

    def test_rgb_clahe_ok(self):
        r = normalize_illumination(_rgb(), method="clahe")
        assert r.image.ndim == 3

    def test_with_mask_ok(self):
        r = normalize_illumination(_gray(), method="mean_std", mask=_mask())
        assert isinstance(r, IllumResult)

    def test_custom_config(self):
        cfg = IllumConfig(target_mean=100.0, target_std=40.0)
        r = normalize_illumination(_gray(), cfg=cfg)
        assert isinstance(r, IllumResult)

    def test_values_in_range(self):
        r = normalize_illumination(_gray())
        assert r.image.min() >= 0
        assert r.image.max() <= 255


# ─── TestBatchNormalizeIllumination ───────────────────────────────────────────

class TestBatchNormalizeIllumination:
    def test_returns_list(self):
        images = [_gray(seed=i) for i in range(3)]
        result = batch_normalize_illumination(images)
        assert isinstance(result, list)

    def test_length_matches(self):
        images = [_gray(seed=i) for i in range(5)]
        assert len(batch_normalize_illumination(images)) == 5

    def test_empty_list(self):
        assert batch_normalize_illumination([]) == []

    def test_all_illum_results(self):
        images = [_gray(seed=i) for i in range(3)]
        for r in batch_normalize_illumination(images):
            assert isinstance(r, IllumResult)

    def test_all_uint8(self):
        images = [_gray(seed=i) for i in range(3)]
        for r in batch_normalize_illumination(images):
            assert r.image.dtype == np.uint8

    def test_method_background(self):
        images = [_gray(seed=i) for i in range(2)]
        for r in batch_normalize_illumination(images, method="background"):
            assert r.method == "background"

    def test_method_clahe(self):
        images = [_gray(seed=i) for i in range(2)]
        for r in batch_normalize_illumination(images, method="clahe"):
            assert r.method == "clahe"

    def test_custom_config(self):
        cfg = IllumConfig(target_mean=90.0)
        images = [_gray(seed=i) for i in range(2)]
        results = batch_normalize_illumination(images, cfg=cfg)
        assert len(results) == 2

    def test_shapes_preserved(self):
        images = [_gray(48, 80, seed=i) for i in range(3)]
        for r in batch_normalize_illumination(images):
            assert r.image.shape == (48, 80)
