"""Extra tests for puzzle_reconstruction/preprocessing/texture_analyzer.py."""
from __future__ import annotations

import numpy as np
import pytest

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

def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _edge_image():
    """High contrast edge image."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[:, 25:] = 255
    return img


# ─── TextureConfig ───────────────────────────────────────────────────────────

class TestTextureConfigExtra:
    def test_defaults(self):
        c = TextureConfig()
        assert c.n_bins == 32
        assert c.patch_size == 5
        assert c.use_gradient is True
        assert c.use_stats is True

    def test_valid_patch_sizes(self):
        for ps in (3, 5, 7, 9):
            TextureConfig(patch_size=ps)

    def test_even_patch_size_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=4)

    def test_small_patch_size_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(patch_size=1)

    def test_low_n_bins_raises(self):
        with pytest.raises(ValueError):
            TextureConfig(n_bins=1)


# ─── GradientStats ──────────────────────────────────────────────────────────

class TestGradientStatsExtra:
    def test_valid(self):
        gs = GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=10.0)
        assert gs.mean == pytest.approx(1.0)

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=-1.0, std=0.5, max_val=3.0, energy=10.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=1.0, std=-0.5, max_val=3.0, energy=10.0)

    def test_negative_max_val_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=1.0, std=0.5, max_val=-1.0, energy=10.0)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError):
            GradientStats(mean=1.0, std=0.5, max_val=3.0, energy=-1.0)


# ─── TextureFeatures ────────────────────────────────────────────────────────

class TestTextureFeaturesExtra:
    def test_valid(self):
        tf = TextureFeatures(fragment_id=0, gradient_mean=1.0,
                             gradient_std=0.5, entropy=2.0, contrast=1.5)
        assert tf.fragment_id == 0

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=-1)

    def test_negative_gradient_mean_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, gradient_mean=-1.0)

    def test_negative_entropy_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, entropy=-1.0)

    def test_negative_contrast_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, contrast=-1.0)

    def test_low_n_bins_raises(self):
        with pytest.raises(ValueError):
            TextureFeatures(fragment_id=0, n_bins=1)

    def test_feature_vector(self):
        tf = TextureFeatures(fragment_id=0, gradient_mean=1.0,
                             gradient_std=2.0, entropy=3.0, contrast=4.0)
        vec = tf.feature_vector
        assert vec.shape == (4,)
        np.testing.assert_array_almost_equal(vec, [1.0, 2.0, 3.0, 4.0])


# ─── compute_gradient_stats ─────────────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_returns_gradient_stats(self):
        gs = compute_gradient_stats(_gray())
        assert isinstance(gs, GradientStats)

    def test_uniform_low_gradient(self):
        gs = compute_gradient_stats(_gray())
        assert gs.mean == pytest.approx(0.0)
        assert gs.max_val == pytest.approx(0.0)

    def test_edge_higher_gradient(self):
        gs_edge = compute_gradient_stats(_edge_image())
        gs_flat = compute_gradient_stats(_gray())
        assert gs_edge.mean > gs_flat.mean

    def test_bgr_input(self):
        gs = compute_gradient_stats(_bgr())
        assert isinstance(gs, GradientStats)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_stats(np.array([1, 2, 3]))

    def test_all_nonnegative(self):
        gs = compute_gradient_stats(_edge_image())
        assert gs.mean >= 0.0
        assert gs.std >= 0.0
        assert gs.max_val >= 0.0
        assert gs.energy >= 0.0


# ─── compute_texture_entropy ────────────────────────────────────────────────

class TestComputeTextureEntropyExtra:
    def test_nonnegative(self):
        e = compute_texture_entropy(_gray())
        assert e >= 0.0

    def test_uniform_low_entropy(self):
        e = compute_texture_entropy(_gray())
        assert e < 0.1

    def test_edge_higher_entropy(self):
        e_edge = compute_texture_entropy(_edge_image())
        e_flat = compute_texture_entropy(_gray())
        assert e_edge > e_flat

    def test_bgr_input(self):
        e = compute_texture_entropy(_bgr())
        assert isinstance(e, float)

    def test_low_n_bins_raises(self):
        with pytest.raises(ValueError):
            compute_texture_entropy(_gray(), n_bins=1)


# ─── compute_texture_contrast ───────────────────────────────────────────────

class TestComputeTextureContrastExtra:
    def test_nonnegative(self):
        c = compute_texture_contrast(_gray())
        assert c >= 0.0

    def test_uniform_zero_contrast(self):
        c = compute_texture_contrast(_gray())
        assert c == pytest.approx(0.0, abs=0.1)

    def test_edge_higher_contrast(self):
        c_edge = compute_texture_contrast(_edge_image())
        c_flat = compute_texture_contrast(_gray())
        assert c_edge > c_flat

    def test_bgr_input(self):
        c = compute_texture_contrast(_bgr())
        assert isinstance(c, float)

    def test_even_patch_size_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=4)

    def test_small_patch_size_raises(self):
        with pytest.raises(ValueError):
            compute_texture_contrast(_gray(), patch_size=1)


# ─── extract_texture_features ───────────────────────────────────────────────

class TestExtractTextureFeaturesExtra:
    def test_returns_texture_features(self):
        tf = extract_texture_features(_gray())
        assert isinstance(tf, TextureFeatures)

    def test_fragment_id(self):
        tf = extract_texture_features(_gray(), fragment_id=5)
        assert tf.fragment_id == 5

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            extract_texture_features(_gray(), fragment_id=-1)

    def test_histogram_sum(self):
        tf = extract_texture_features(_gray())
        assert tf.histogram is not None
        assert tf.histogram.sum() == pytest.approx(1.0)

    def test_bgr_input(self):
        tf = extract_texture_features(_bgr())
        assert isinstance(tf, TextureFeatures)

    def test_custom_config(self):
        cfg = TextureConfig(n_bins=16, patch_size=3)
        tf = extract_texture_features(_gray(), cfg=cfg)
        assert tf.n_bins == 16

    def test_no_gradient(self):
        cfg = TextureConfig(use_gradient=False)
        tf = extract_texture_features(_gray(), cfg=cfg)
        assert tf.gradient_mean == 0.0
        assert tf.gradient_std == 0.0


# ─── compare_texture_features ───────────────────────────────────────────────

class TestCompareTextureFeaturesExtra:
    def test_identical(self):
        tf = extract_texture_features(_edge_image())
        sim = compare_texture_features(tf, tf)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        a = extract_texture_features(_edge_image(), fragment_id=0)
        b = extract_texture_features(_gray(), fragment_id=1)
        sim = compare_texture_features(a, b)
        assert -1.0 <= sim <= 1.0

    def test_zero_vectors(self):
        a = TextureFeatures(fragment_id=0, gradient_mean=0.0,
                            gradient_std=0.0, entropy=0.0, contrast=0.0)
        b = TextureFeatures(fragment_id=1, gradient_mean=0.0,
                            gradient_std=0.0, entropy=0.0, contrast=0.0)
        sim = compare_texture_features(a, b)
        assert sim == pytest.approx(0.0)


# ─── batch_extract_texture ──────────────────────────────────────────────────

class TestBatchExtractTextureExtra:
    def test_empty(self):
        assert batch_extract_texture([]) == []

    def test_length(self):
        results = batch_extract_texture([_gray(), _gray()])
        assert len(results) == 2

    def test_fragment_ids(self):
        results = batch_extract_texture([_gray(), _edge_image()])
        assert results[0].fragment_id == 0
        assert results[1].fragment_id == 1

    def test_result_type(self):
        results = batch_extract_texture([_gray()])
        assert isinstance(results[0], TextureFeatures)
