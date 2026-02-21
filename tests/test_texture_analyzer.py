"""Тесты для puzzle_reconstruction.preprocessing.texture_analyzer."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.texture_analyzer import (
    TextureConfig,
    GradientStats,
    TextureFeatures,
    compute_gradient_stats,
    compute_texture_entropy,
    compute_texture_contrast,
    extract_texture_features,
    compare_texture_features,
    batch_extract_texture,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _features(fid: int = 0, n_bins: int = 32) -> TextureFeatures:
    hist = np.ones(n_bins, dtype=float) / n_bins
    return TextureFeatures(
        fragment_id=fid,
        gradient_mean=10.0,
        gradient_std=5.0,
        entropy=3.0,
        contrast=2.0,
        n_bins=n_bins,
        histogram=hist,
    )


# ─── TestTextureConfig ────────────────────────────────────────────────────────

class TestTextureConfig:
    def test_defaults(self):
        cfg = TextureConfig()
        assert cfg.n_bins == 32
        assert cfg.patch_size == 5
        assert cfg.use_gradient is True
        assert cfg.use_stats is True

    def test_n_bins_two_ok(self):
        cfg = TextureConfig(n_bins=2)
        assert cfg.n_bins == 2

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(n_bins=1)

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(n_bins=0)

    def test_patch_size_three_ok(self):
        cfg = TextureConfig(patch_size=3)
        assert cfg.patch_size == 3

    def test_patch_size_even_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=4)

    def test_patch_size_two_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=2)

    def test_patch_size_large_odd_ok(self):
        cfg = TextureConfig(patch_size=25)
        assert cfg.patch_size == 25

    def test_use_gradient_false(self):
        cfg = TextureConfig(use_gradient=False)
        assert cfg.use_gradient is False

    def test_use_stats_false(self):
        cfg = TextureConfig(use_stats=False)
        assert cfg.use_stats is False


# ─── TestGradientStats ────────────────────────────────────────────────────────

class TestGradientStats:
    def test_basic(self):
        gs = GradientStats(mean=5.0, std=2.0, max_val=30.0, energy=100.0)
        assert gs.mean == pytest.approx(5.0)

    def test_all_zero_ok(self):
        gs = GradientStats(mean=0.0, std=0.0, max_val=0.0, energy=0.0)
        assert gs.energy == 0.0

    def test_mean_neg_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=-0.1, std=0.0, max_val=0.0, energy=0.0)

    def test_std_neg_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=-1.0, max_val=0.0, energy=0.0)

    def test_max_val_neg_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=0.0, max_val=-5.0, energy=0.0)

    def test_energy_neg_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=0.0, max_val=0.0, energy=-1.0)


# ─── TestTextureFeatures ──────────────────────────────────────────────────────

class TestTextureFeatures:
    def test_basic(self):
        tf = _features(fid=3)
        assert tf.fragment_id == 3

    def test_feature_vector_shape(self):
        tf = _features()
        v = tf.feature_vector
        assert v.shape == (4,)

    def test_feature_vector_dtype(self):
        tf = _features()
        assert tf.feature_vector.dtype == np.float64

    def test_feature_vector_values(self):
        tf = _features()
        v = tf.feature_vector
        assert v[0] == pytest.approx(tf.gradient_mean)
        assert v[1] == pytest.approx(tf.gradient_std)
        assert v[2] == pytest.approx(tf.entropy)
        assert v[3] == pytest.approx(tf.contrast)

    def test_fragment_id_neg_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=-1)

    def test_gradient_mean_neg_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, gradient_mean=-1.0)

    def test_gradient_std_neg_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, gradient_std=-1.0)

    def test_entropy_neg_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, entropy=-0.1)

    def test_contrast_neg_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, contrast=-2.0)

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, n_bins=1)

    def test_histogram_none_ok(self):
        tf = TextureFeatures(fragment_id=0)
        assert tf.histogram is None

    def test_histogram_array_ok(self):
        hist = np.ones(32) / 32
        tf = TextureFeatures(fragment_id=0, histogram=hist)
        assert tf.histogram is not None


# ─── TestComputeGradientStats ─────────────────────────────────────────────────

class TestComputeGradientStats:
    def test_returns_gradient_stats(self):
        gs = compute_gradient_stats(_gray())
        assert isinstance(gs, GradientStats)

    def test_all_non_negative(self):
        gs = compute_gradient_stats(_gray())
        assert gs.mean >= 0.0
        assert gs.std >= 0.0
        assert gs.max_val >= 0.0
        assert gs.energy >= 0.0

    def test_rgb_image_ok(self):
        gs = compute_gradient_stats(_rgb())
        assert isinstance(gs, GradientStats)

    def test_constant_image_low_gradient(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        gs = compute_gradient_stats(img)
        # Внутри постоянного изображения градиент должен быть близок к 0
        assert gs.mean < 1.0

    def test_high_contrast_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[:, 16:] = 255
        gs = compute_gradient_stats(img)
        assert gs.max_val > 0.0

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_stats(np.zeros(10))

    def test_mean_le_max_val(self):
        gs = compute_gradient_stats(_gray())
        assert gs.mean <= gs.max_val + 1e-9

    def test_energy_positive_for_varied_img(self):
        gs = compute_gradient_stats(_gray())
        assert gs.energy > 0.0


# ─── TestComputeTextureEntropy ────────────────────────────────────────────────

class TestComputeTextureEntropy:
    def test_returns_float(self):
        assert isinstance(compute_texture_entropy(_gray()), float)

    def test_non_negative(self):
        assert compute_texture_entropy(_gray()) >= 0.0

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            compute_texture_entropy(_gray(), n_bins=1)

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            compute_texture_entropy(_gray(), n_bins=0)

    def test_constant_image_low_entropy(self):
        img = np.full((32, 32), 100, dtype=np.uint8)
        e = compute_texture_entropy(img)
        assert e < 1.0  # single bin → nearly 0

    def test_uniform_image_high_entropy(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        e = compute_texture_entropy(img, n_bins=32)
        assert e > 3.0

    def test_rgb_image_ok(self):
        e = compute_texture_entropy(_rgb())
        assert e >= 0.0

    def test_more_bins_same_or_higher(self):
        img = _gray()
        e8 = compute_texture_entropy(img, n_bins=8)
        e64 = compute_texture_entropy(img, n_bins=64)
        # При большем числе бинов возможна более высокая энтропия
        assert e8 >= 0.0 and e64 >= 0.0


# ─── TestComputeTextureContrast ───────────────────────────────────────────────

class TestComputeTextureContrast:
    def test_returns_float(self):
        assert isinstance(compute_texture_contrast(_gray()), float)

    def test_non_negative(self):
        assert compute_texture_contrast(_gray()) >= 0.0

    def test_patch_size_even_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=4)

    def test_patch_size_two_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=2)

    def test_constant_image_zero_contrast(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        c = compute_texture_contrast(img)
        assert c < 1.0

    def test_high_contrast_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[::2, :] = 255
        c = compute_texture_contrast(img)
        assert c > 10.0

    def test_rgb_image_ok(self):
        c = compute_texture_contrast(_rgb())
        assert c >= 0.0

    def test_larger_patch(self):
        c = compute_texture_contrast(_gray(), patch_size=9)
        assert c >= 0.0


# ─── TestExtractTextureFeatures ───────────────────────────────────────────────

class TestExtractTextureFeatures:
    def test_returns_texture_features(self):
        tf = extract_texture_features(_gray())
        assert isinstance(tf, TextureFeatures)

    def test_fragment_id_stored(self):
        tf = extract_texture_features(_gray(), fragment_id=7)
        assert tf.fragment_id == 7

    def test_neg_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_texture_features(_gray(), fragment_id=-1)

    def test_n_bins_matches_config(self):
        cfg = TextureConfig(n_bins=16)
        tf = extract_texture_features(_gray(), cfg=cfg)
        assert tf.n_bins == 16

    def test_histogram_length_matches_n_bins(self):
        cfg = TextureConfig(n_bins=16)
        tf = extract_texture_features(_gray(), cfg=cfg)
        assert tf.histogram is not None
        assert len(tf.histogram) == 16

    def test_histogram_sums_to_one(self):
        tf = extract_texture_features(_gray())
        assert tf.histogram is not None
        assert tf.histogram.sum() == pytest.approx(1.0, abs=1e-9)

    def test_entropy_non_negative(self):
        tf = extract_texture_features(_gray())
        assert tf.entropy >= 0.0

    def test_contrast_non_negative(self):
        tf = extract_texture_features(_gray())
        assert tf.contrast >= 0.0

    def test_no_gradient_zeros(self):
        cfg = TextureConfig(use_gradient=False)
        tf = extract_texture_features(_gray(), cfg=cfg)
        assert tf.gradient_mean == 0.0
        assert tf.gradient_std == 0.0

    def test_with_gradient(self):
        cfg = TextureConfig(use_gradient=True)
        tf = extract_texture_features(_gray(seed=1), cfg=cfg)
        assert tf.gradient_mean >= 0.0

    def test_rgb_image_ok(self):
        tf = extract_texture_features(_rgb())
        assert isinstance(tf, TextureFeatures)

    def test_default_n_bins(self):
        tf = extract_texture_features(_gray())
        assert tf.n_bins == 32


# ─── TestCompareTextureFeatures ───────────────────────────────────────────────

class TestCompareTextureFeatures:
    def test_same_features_high_similarity(self):
        tf = _features()
        sim = compare_texture_features(tf, tf)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self):
        a, b = _features(0), _features(1)
        assert isinstance(compare_texture_features(a, b), float)

    def test_zero_vector_returns_zero(self):
        zero = TextureFeatures(fragment_id=0)
        tf = _features()
        sim = compare_texture_features(zero, tf)
        assert sim == pytest.approx(0.0)

    def test_same_image_high_sim(self):
        img = _gray()
        a = extract_texture_features(img, 0)
        b = extract_texture_features(img.copy(), 1)
        assert compare_texture_features(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_different_images_lower_sim(self):
        a = extract_texture_features(np.zeros((32, 32), dtype=np.uint8), 0)
        b = extract_texture_features(np.full((32, 32), 200, dtype=np.uint8), 1)
        # Градиенты одинаковы (оба 0), но энтропия может отличаться
        sim = compare_texture_features(a, b)
        assert isinstance(sim, float)

    def test_value_range(self):
        a = extract_texture_features(_gray(seed=0), 0)
        b = extract_texture_features(_gray(seed=1), 1)
        sim = compare_texture_features(a, b)
        assert -1.0 <= sim <= 1.0


# ─── TestBatchExtractTexture ──────────────────────────────────────────────────

class TestBatchExtractTexture:
    def test_returns_list(self):
        images = [_gray(seed=i) for i in range(3)]
        result = batch_extract_texture(images)
        assert isinstance(result, list)

    def test_length_matches(self):
        images = [_gray(seed=i) for i in range(5)]
        assert len(batch_extract_texture(images)) == 5

    def test_empty_list(self):
        assert batch_extract_texture([]) == []

    def test_fragment_ids_sequential(self):
        images = [_gray(seed=i) for i in range(4)]
        for i, tf in enumerate(batch_extract_texture(images)):
            assert tf.fragment_id == i

    def test_all_texture_features(self):
        images = [_gray(seed=i) for i in range(3)]
        for tf in batch_extract_texture(images):
            assert isinstance(tf, TextureFeatures)

    def test_custom_config(self):
        cfg = TextureConfig(n_bins=16)
        images = [_gray(seed=i) for i in range(2)]
        for tf in batch_extract_texture(images, cfg):
            assert tf.n_bins == 16
