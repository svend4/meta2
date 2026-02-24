"""Extra tests for puzzle_reconstruction/preprocessing/texture_analyzer.py."""
from __future__ import annotations

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

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=32) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _features(fid=0, gmean=1.0, gstd=0.5, ent=2.0, cont=0.3) -> TextureFeatures:
    return TextureFeatures(fragment_id=fid, gradient_mean=gmean,
                           gradient_std=gstd, entropy=ent, contrast=cont)


# ─── TextureConfig ────────────────────────────────────────────────────────────

class TestTextureConfigExtra:
    def test_default_n_bins(self):
        assert TextureConfig().n_bins == 32

    def test_default_patch_size(self):
        assert TextureConfig().patch_size == 5

    def test_default_use_gradient(self):
        assert TextureConfig().use_gradient is True

    def test_default_use_stats(self):
        assert TextureConfig().use_stats is True

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(n_bins=1)

    def test_even_patch_size_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=4)

    def test_patch_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=1)

    def test_custom_n_bins(self):
        cfg = TextureConfig(n_bins=64)
        assert cfg.n_bins == 64

    def test_custom_patch_size(self):
        cfg = TextureConfig(patch_size=7)
        assert cfg.patch_size == 7


# ─── GradientStats ────────────────────────────────────────────────────────────

class TestGradientStatsExtra:
    def test_stores_mean(self):
        gs = GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=10.0)
        assert gs.mean == pytest.approx(1.0)

    def test_stores_std(self):
        gs = GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=10.0)
        assert gs.std == pytest.approx(0.5)

    def test_stores_max_val(self):
        gs = GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=10.0)
        assert gs.max_val == pytest.approx(3.0)

    def test_stores_energy(self):
        gs = GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=10.0)
        assert gs.energy == pytest.approx(10.0)

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=-1.0, std=0.0, max_val=0.0, energy=0.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=-0.1, max_val=0.0, energy=0.0)

    def test_negative_max_val_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=0.0, max_val=-1.0, energy=0.0)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=0.0, std=0.0, max_val=0.0, energy=-1.0)

    def test_zero_values_ok(self):
        gs = GradientStats(mean=0.0, std=0.0, max_val=0.0, energy=0.0)
        assert gs.mean == pytest.approx(0.0)


# ─── TextureFeatures ──────────────────────────────────────────────────────────

class TestTextureFeaturesExtra:
    def test_fragment_id_stored(self):
        f = _features(fid=5)
        assert f.fragment_id == 5

    def test_gradient_mean_stored(self):
        f = _features(gmean=2.5)
        assert f.gradient_mean == pytest.approx(2.5)

    def test_gradient_std_stored(self):
        f = _features(gstd=1.2)
        assert f.gradient_std == pytest.approx(1.2)

    def test_entropy_stored(self):
        f = _features(ent=3.0)
        assert f.entropy == pytest.approx(3.0)

    def test_contrast_stored(self):
        f = _features(cont=0.7)
        assert f.contrast == pytest.approx(0.7)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=-1)

    def test_negative_gradient_mean_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, gradient_mean=-0.1)

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, n_bins=1)

    def test_feature_vector_shape(self):
        f = _features()
        v = f.feature_vector
        assert v.shape == (4,)

    def test_feature_vector_dtype(self):
        v = _features().feature_vector
        assert v.dtype == np.float64

    def test_feature_vector_values(self):
        f = _features(gmean=1.0, gstd=0.5, ent=2.0, cont=0.3)
        v = f.feature_vector
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(0.5)
        assert v[2] == pytest.approx(2.0)
        assert v[3] == pytest.approx(0.3)


# ─── compute_gradient_stats ───────────────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_returns_gradient_stats(self):
        assert isinstance(compute_gradient_stats(_gray()), GradientStats)

    def test_uniform_mean_near_zero(self):
        gs = compute_gradient_stats(_gray(val=200))
        assert gs.mean >= 0.0

    def test_ramp_has_positive_gradient(self):
        gs = compute_gradient_stats(_ramp())
        assert gs.mean > 0.0

    def test_energy_nonneg(self):
        gs = compute_gradient_stats(_gray())
        assert gs.energy >= 0.0

    def test_max_val_gte_mean(self):
        gs = compute_gradient_stats(_ramp())
        assert gs.max_val >= gs.mean

    def test_bgr_image_ok(self):
        gs = compute_gradient_stats(_bgr())
        assert isinstance(gs, GradientStats)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_stats(np.zeros((2, 4, 4, 3)))


# ─── compute_texture_entropy ──────────────────────────────────────────────────

class TestComputeTextureEntropyExtra:
    def test_returns_float(self):
        assert isinstance(compute_texture_entropy(_gray()), float)

    def test_uniform_image_low_entropy(self):
        e = compute_texture_entropy(_gray(val=128))
        assert e >= 0.0

    def test_ramp_higher_entropy(self):
        e_uniform = compute_texture_entropy(_gray(val=128))
        e_ramp = compute_texture_entropy(_ramp())
        assert e_ramp >= e_uniform

    def test_result_nonneg(self):
        assert compute_texture_entropy(_gray()) >= 0.0

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            compute_texture_entropy(_gray(), n_bins=1)

    def test_bgr_image_ok(self):
        assert isinstance(compute_texture_entropy(_bgr()), float)


# ─── compute_texture_contrast ─────────────────────────────────────────────────

class TestComputeTextureContrastExtra:
    def test_returns_float(self):
        assert isinstance(compute_texture_contrast(_gray()), float)

    def test_result_nonneg(self):
        assert compute_texture_contrast(_gray()) >= 0.0

    def test_uniform_contrast_near_zero(self):
        c = compute_texture_contrast(_gray(val=200))
        assert c < 1.0

    def test_ramp_has_positive_contrast(self):
        c = compute_texture_contrast(_ramp())
        assert c >= 0.0

    def test_even_patch_size_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=4)

    def test_patch_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=1)

    def test_bgr_image_ok(self):
        assert isinstance(compute_texture_contrast(_bgr()), float)


# ─── extract_texture_features ─────────────────────────────────────────────────

class TestExtractTextureFeaturesExtra:
    def test_returns_texture_features(self):
        assert isinstance(extract_texture_features(_gray()), TextureFeatures)

    def test_fragment_id_stored(self):
        f = extract_texture_features(_gray(), fragment_id=3)
        assert f.fragment_id == 3

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_texture_features(_gray(), fragment_id=-1)

    def test_n_bins_matches_config(self):
        cfg = TextureConfig(n_bins=64)
        f = extract_texture_features(_gray(), cfg=cfg)
        assert f.n_bins == 64

    def test_histogram_length_matches_n_bins(self):
        cfg = TextureConfig(n_bins=16)
        f = extract_texture_features(_gray(), cfg=cfg)
        assert f.histogram is not None
        assert len(f.histogram) == 16

    def test_histogram_sums_to_one(self):
        f = extract_texture_features(_ramp())
        assert f.histogram is not None
        assert float(f.histogram.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_no_gradient_when_disabled(self):
        cfg = TextureConfig(use_gradient=False)
        f = extract_texture_features(_gray(), cfg=cfg)
        assert f.gradient_mean == pytest.approx(0.0)
        assert f.gradient_std == pytest.approx(0.0)

    def test_none_cfg(self):
        f = extract_texture_features(_gray(), cfg=None)
        assert isinstance(f, TextureFeatures)

    def test_bgr_image_ok(self):
        f = extract_texture_features(_bgr())
        assert isinstance(f, TextureFeatures)


# ─── compare_texture_features ─────────────────────────────────────────────────

class TestCompareTextureFeaturesExtra:
    def test_returns_float(self):
        f = _features()
        assert isinstance(compare_texture_features(f, f), float)

    def test_identical_features_one(self):
        f = _features(gmean=1.0, gstd=0.5, ent=2.0, cont=0.3)
        result = compare_texture_features(f, f)
        assert result == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        f1 = _features(gmean=0.0, gstd=0.0, ent=0.0, cont=0.0)
        f2 = _features(gmean=1.0, gstd=0.0, ent=0.0, cont=0.0)
        result = compare_texture_features(f1, f2)
        assert result == pytest.approx(0.0)

    def test_result_in_minus_one_to_one(self):
        f1 = _features(gmean=1.0, gstd=0.5)
        f2 = _features(gmean=2.0, gstd=1.0)
        result = compare_texture_features(f1, f2)
        assert -1.0 <= result <= 1.0

    def test_different_features_lt_one(self):
        f1 = _features(gmean=1.0, gstd=0.0, ent=0.0, cont=0.0)
        f2 = _features(gmean=0.0, gstd=1.0, ent=0.0, cont=0.0)
        result = compare_texture_features(f1, f2)
        assert result < 1.0


# ─── batch_extract_texture ────────────────────────────────────────────────────

class TestBatchExtractTextureExtra:
    def test_returns_list(self):
        assert isinstance(batch_extract_texture([_gray()]), list)

    def test_length_matches(self):
        imgs = [_gray(), _ramp(), _bgr()]
        result = batch_extract_texture(imgs)
        assert len(result) == 3

    def test_fragment_ids_sequential(self):
        imgs = [_gray(), _ramp()]
        result = batch_extract_texture(imgs)
        assert result[0].fragment_id == 0
        assert result[1].fragment_id == 1

    def test_each_element_is_texture_features(self):
        for f in batch_extract_texture([_gray(), _ramp()]):
            assert isinstance(f, TextureFeatures)

    def test_empty_list(self):
        assert batch_extract_texture([]) == []

    def test_none_cfg(self):
        result = batch_extract_texture([_gray()], cfg=None)
        assert len(result) == 1
