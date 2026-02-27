"""Extra tests for puzzle_reconstruction/verification/statistical_coherence.py"""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.verification.statistical_coherence import (
    StatisticalCoherenceConfig,
    StatisticalCoherenceResult,
    StatisticalCoherenceVerifier,
    cohere_score,
    _to_gray_flat,
    _bhattacharyya_coefficient,
    _skewness,
    _glcm_contrast,
    _glcm_energy,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rng(seed=0):
    return np.random.default_rng(seed)


def _patch2d(h=16, w=16, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def _patch3d(h=16, w=16, fill=128, dtype=np.uint8):
    return np.full((h, w, 3), fill, dtype=dtype)


def _rand2d(h=16, w=16, seed=0):
    return _rng(seed).integers(0, 256, (h, w)).astype(np.uint8)


def _rand3d(h=16, w=16, seed=0):
    return _rng(seed).integers(0, 256, (h, w, 3)).astype(np.uint8)


# ─── StatisticalCoherenceConfig – edge cases ──────────────────────────────────

class TestStatCoherenceConfigExtra:

    def test_zero_bins_stored(self):
        cfg = StatisticalCoherenceConfig(n_bins=0)
        assert cfg.n_bins == 0

    def test_single_bin_stored(self):
        cfg = StatisticalCoherenceConfig(n_bins=1)
        assert cfg.n_bins == 1

    def test_threshold_zero(self):
        cfg = StatisticalCoherenceConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_one(self):
        cfg = StatisticalCoherenceConfig(threshold=1.0)
        assert cfg.threshold == 1.0

    def test_use_texture_false(self):
        cfg = StatisticalCoherenceConfig(use_texture=False)
        assert cfg.use_texture is False

    def test_method_both_stored(self):
        cfg = StatisticalCoherenceConfig(method="both")
        assert cfg.method == "both"


# ─── _to_gray_flat – additional cases ────────────────────────────────────────

class TestToGrayFlatExtra:

    def test_1d_float64_passthrough(self):
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        out = _to_gray_flat(arr)
        np.testing.assert_array_equal(out, arr)

    def test_2d_float64_flattened(self):
        arr = np.ones((3, 4), dtype=np.float64) * 99.0
        out = _to_gray_flat(arr)
        assert out.ndim == 1
        assert len(out) == 12
        assert np.all(out == 99.0)

    def test_3d_channels_averaged(self):
        arr = np.zeros((2, 2, 3), dtype=np.float64)
        arr[:, :, 0] = 0.0
        arr[:, :, 1] = 90.0
        arr[:, :, 2] = 180.0
        out = _to_gray_flat(arr)
        expected_mean = 90.0
        np.testing.assert_allclose(out, expected_mean, atol=1e-10)

    def test_3d_single_pixel_3ch(self):
        arr = np.array([[[10., 20., 30.]]])
        out = _to_gray_flat(arr)
        assert len(out) == 1
        assert out[0] == pytest.approx(20.0)

    def test_output_dtype_float64(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        out = _to_gray_flat(arr)
        assert out.dtype == np.float64

    def test_4d_raises_value_error(self):
        arr = np.ones((2, 2, 2, 2))
        with pytest.raises(ValueError):
            _to_gray_flat(arr)

    def test_large_2d_no_crash(self):
        arr = _rng(10).integers(0, 256, (200, 200)).astype(np.uint8)
        out = _to_gray_flat(arr)
        assert len(out) == 200 * 200


# ─── _bhattacharyya_coefficient – additional cases ────────────────────────────

class TestBhattacharyyaExtra:

    def test_partial_overlap(self):
        # Histograms with partial overlap should give intermediate value
        hist_a = np.array([50.0, 50.0, 0.0, 0.0])
        hist_b = np.array([0.0, 50.0, 50.0, 0.0])
        bc = _bhattacharyya_coefficient(hist_a, hist_b)
        assert 0.0 < bc < 1.0

    def test_single_bin(self):
        hist_a = np.array([5.0])
        hist_b = np.array([10.0])
        bc = _bhattacharyya_coefficient(hist_a, hist_b)
        assert bc == pytest.approx(1.0)

    def test_both_zero_returns_zero(self):
        bc = _bhattacharyya_coefficient(np.zeros(4), np.zeros(4))
        assert bc == 0.0

    def test_symmetry(self):
        rng = _rng(1)
        h1 = rng.random(16)
        h2 = rng.random(16)
        bc1 = _bhattacharyya_coefficient(h1, h2)
        bc2 = _bhattacharyya_coefficient(h2, h1)
        assert bc1 == pytest.approx(bc2, rel=1e-10)

    def test_clipped_to_one(self):
        # Due to normalisation BC ≤ 1
        hist = np.array([1.0, 2.0, 3.0])
        assert _bhattacharyya_coefficient(hist, hist) <= 1.0 + 1e-12

    def test_output_is_scalar(self):
        bc = _bhattacharyya_coefficient(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert isinstance(bc, float)


# ─── _skewness – additional cases ────────────────────────────────────────────

class TestSkewnessExtra:

    def test_right_skewed_positive(self):
        # Exponential-like distribution has positive skew
        rng = _rng(2)
        x = rng.exponential(scale=1.0, size=500)
        sk = _skewness(x)
        assert sk > 0.5

    def test_left_skewed_negative(self):
        rng = _rng(3)
        x = -rng.exponential(scale=1.0, size=500) + 10.0
        sk = _skewness(x)
        assert sk < -0.5

    def test_single_element_returns_zero(self):
        assert _skewness(np.array([42.0])) == 0.0

    def test_two_elements_returns_zero(self):
        assert _skewness(np.array([1.0, 9.0])) == 0.0

    def test_uniform_distribution_near_zero(self):
        x = np.linspace(0, 10, 1000)
        sk = _skewness(x)
        assert abs(sk) < 0.1

    def test_output_is_float(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(_skewness(x), float)


# ─── _glcm_contrast – additional cases ───────────────────────────────────────

class TestGlcmContrastExtra:

    def test_high_contrast_image(self):
        # Alternating 0/255 pattern → high contrast
        arr = np.array([[0., 255., 0., 255.]] * 4)
        c = _glcm_contrast(arr)
        assert c > 0.5

    def test_single_column_no_crash(self):
        arr = np.array([[100.], [150.], [200.]])
        c = _glcm_contrast(arr)
        assert 0.0 <= c <= 1.0

    def test_displacement_d2(self):
        arr = np.arange(100, dtype=np.float64).reshape(10, 10)
        c1 = _glcm_contrast(arr, d=1)
        c2 = _glcm_contrast(arr, d=2)
        assert 0.0 <= c1 <= 1.0
        assert 0.0 <= c2 <= 1.0

    def test_gradient_image(self):
        arr = np.tile(np.linspace(0, 255, 20), (10, 1))
        c = _glcm_contrast(arr)
        assert c > 0.0

    def test_small_1x1_image(self):
        arr = np.array([[128.]])
        c = _glcm_contrast(arr)
        # Only one pixel → falls back to flat, len <= d → 0
        assert c == pytest.approx(0.0)

    def test_result_clipped_to_0_1(self):
        rng = _rng(5)
        arr = rng.integers(0, 256, (20, 20)).astype(np.float64)
        c = _glcm_contrast(arr)
        assert 0.0 <= c <= 1.0


# ─── _glcm_energy – additional cases ─────────────────────────────────────────

class TestGlcmEnergyExtra:

    def test_high_variance_low_energy(self):
        arr = np.array([[0., 255., 0., 255.]] * 4)
        e = _glcm_energy(arr)
        # High variance → energy << 1
        assert e < 0.5

    def test_single_element_returns_one(self):
        arr = np.array([[42.]])
        e = _glcm_energy(arr)
        assert e == pytest.approx(1.0)

    def test_range_check(self):
        rng = _rng(6)
        arr = rng.integers(0, 256, (15, 15)).astype(np.float64)
        e = _glcm_energy(arr)
        assert 0.0 < e <= 1.0

    def test_always_positive(self):
        for seed in range(5):
            arr = _rng(seed).integers(0, 256, (8, 8)).astype(np.float64)
            assert _glcm_energy(arr) > 0.0

    def test_1d_fallback(self):
        arr = np.array([0.0, 100.0, 200.0, 100.0, 0.0])
        e = _glcm_energy(arr)
        assert 0.0 < e <= 1.0


# ─── StatisticalCoherenceVerifier.verify – method variants ───────────────────

class TestVerifierMethodVariants:

    def test_histogram_no_texture(self):
        cfg = StatisticalCoherenceConfig(method="histogram", use_texture=False)
        v = StatisticalCoherenceVerifier(cfg)
        p = _patch2d(10, 10, 100)
        r = v.verify(p, p)
        assert r.overall_score == pytest.approx(r.histogram_similarity)

    def test_moments_no_texture(self):
        cfg = StatisticalCoherenceConfig(method="moments", use_texture=False)
        v = StatisticalCoherenceVerifier(cfg)
        p = _patch2d(10, 10, 100)
        r = v.verify(p, p)
        assert r.overall_score == pytest.approx(r.moment_similarity, abs=0.01)

    def test_both_no_texture(self):
        cfg = StatisticalCoherenceConfig(method="both", use_texture=False)
        v = StatisticalCoherenceVerifier(cfg)
        p = _patch2d(10, 10, 150)
        r = v.verify(p, p)
        expected = 0.5 * r.histogram_similarity + 0.5 * r.moment_similarity
        assert r.overall_score == pytest.approx(expected, abs=0.01)

    def test_histogram_with_texture_weighted(self):
        cfg = StatisticalCoherenceConfig(method="histogram", use_texture=True)
        v = StatisticalCoherenceVerifier(cfg)
        p = _patch2d(10, 10, 80)
        r = v.verify(p, p)
        expected = 0.6 * r.histogram_similarity + 0.4 * r.texture_similarity
        assert r.overall_score == pytest.approx(expected, abs=0.01)

    def test_moments_with_texture_weighted(self):
        cfg = StatisticalCoherenceConfig(method="moments", use_texture=True)
        v = StatisticalCoherenceVerifier(cfg)
        p = _rand2d(8, 8, seed=7)
        r = v.verify(p, p)
        expected = 0.6 * r.moment_similarity + 0.4 * r.texture_similarity
        assert r.overall_score == pytest.approx(expected, abs=0.01)

    def test_both_with_texture_weighted(self):
        cfg = StatisticalCoherenceConfig(method="both", use_texture=True)
        v = StatisticalCoherenceVerifier(cfg)
        p = _rand2d(8, 8, seed=8)
        r = v.verify(p, p)
        expected = (0.4 * r.histogram_similarity +
                    0.3 * r.moment_similarity +
                    0.3 * r.texture_similarity)
        assert r.overall_score == pytest.approx(expected, abs=0.01)


# ─── StatisticalCoherenceVerifier.verify – input types ───────────────────────

class TestVerifierInputTypes:

    def test_3d_rgb_patches(self):
        v = StatisticalCoherenceVerifier()
        p1 = _rand3d(8, 8, seed=10)
        p2 = _rand3d(8, 8, seed=11)
        r = v.verify(p1, p2)
        assert 0.0 <= r.overall_score <= 1.0

    def test_1d_array_inputs(self):
        v = StatisticalCoherenceVerifier()
        p1 = np.linspace(0, 255, 50)
        p2 = np.linspace(0, 255, 50)
        r = v.verify(p1, p2)
        assert r.histogram_similarity == pytest.approx(1.0)

    def test_mismatched_size_patches(self):
        v = StatisticalCoherenceVerifier()
        p1 = _rand2d(4, 4, seed=12)
        p2 = _rand2d(20, 20, seed=13)
        r = v.verify(p1, p2)
        assert 0.0 <= r.overall_score <= 1.0

    def test_float32_input(self):
        v = StatisticalCoherenceVerifier()
        p = np.random.rand(10, 10).astype(np.float32) * 255
        r = v.verify(p, p)
        assert r.histogram_similarity >= 0.0

    def test_uniform_patches_histogram_sim_one(self):
        v = StatisticalCoherenceVerifier()
        p1 = _patch2d(10, 10, fill=200)
        p2 = _patch2d(10, 10, fill=200)
        r = v.verify(p1, p2)
        assert r.histogram_similarity == pytest.approx(1.0)

    def test_single_pixel_patches(self):
        v = StatisticalCoherenceVerifier()
        p1 = np.array([[50]], dtype=np.uint8)
        p2 = np.array([[50]], dtype=np.uint8)
        r = v.verify(p1, p2)
        assert isinstance(r, StatisticalCoherenceResult)


# ─── is_coherent threshold boundary ──────────────────────────────────────────

class TestIsCoherentBoundary:

    def test_threshold_exactly_met_is_coherent(self):
        # We control overall_score via identical patches (high score) and low threshold
        cfg = StatisticalCoherenceConfig(threshold=0.01)
        v = StatisticalCoherenceVerifier(cfg)
        p = _patch2d(10, 10, 128)
        r = v.verify(p, p)
        assert r.is_coherent is True

    def test_threshold_above_score_not_coherent(self):
        # Identical all-black vs all-white patches → low score
        cfg = StatisticalCoherenceConfig(threshold=0.99, method="moments",
                                          use_texture=False)
        v = StatisticalCoherenceVerifier(cfg)
        p1 = _patch2d(10, 10, 0)
        p2 = _patch2d(10, 10, 255)
        r = v.verify(p1, p2)
        assert r.is_coherent is False

    def test_overall_score_in_range(self):
        for seed in range(5):
            v = StatisticalCoherenceVerifier()
            p1 = _rand2d(8, 8, seed=seed)
            p2 = _rand2d(8, 8, seed=seed + 100)
            r = v.verify(p1, p2)
            assert 0.0 <= r.overall_score <= 1.0


# ─── cohere_score – additional cases ─────────────────────────────────────────

class TestCoheScoreExtra:

    def test_returns_float(self):
        p = _rand2d(8, 8, seed=20)
        score = cohere_score(p, p)
        assert isinstance(score, float)

    def test_symmetric(self):
        p1 = _rand2d(8, 8, seed=21)
        p2 = _rand2d(8, 8, seed=22)
        s1 = cohere_score(p1, p2)
        s2 = cohere_score(p2, p1)
        # Symmetry is approximate because histogram bins use combined range
        assert abs(s1 - s2) < 0.01

    def test_identical_high_score(self):
        p = _rand2d(16, 16, seed=23)
        score = cohere_score(p, p)
        assert score > 0.5

    def test_very_different_lower_score(self):
        p1 = _patch2d(10, 10, 0)
        p2 = _patch2d(10, 10, 255)
        identical = _patch2d(10, 10, 128)
        s_diff = cohere_score(p1, p2)
        s_same = cohere_score(identical, identical)
        assert s_same >= s_diff

    def test_with_all_methods(self):
        p1 = _rand2d(8, 8, seed=24)
        p2 = _rand2d(8, 8, seed=25)
        for method in ("histogram", "moments", "both"):
            cfg = StatisticalCoherenceConfig(method=method)
            score = cohere_score(p1, p2, config=cfg)
            assert 0.0 <= score <= 1.0

    def test_uses_default_config_when_none(self):
        p = _patch2d(10, 10, 150)
        score1 = cohere_score(p, p)
        score2 = cohere_score(p, p, config=None)
        assert score1 == pytest.approx(score2)
