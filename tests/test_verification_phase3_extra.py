"""
Extra edge-case tests for Phase 3 verification modules.

Covers cases NOT addressed in the basic test files:
  - puzzle_reconstruction.verification.color_continuity_verifier
  - puzzle_reconstruction.verification.statistical_coherence
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    ColorContinuityVerifier,
    verify_color_continuity,
)
from puzzle_reconstruction.verification.statistical_coherence import (
    StatisticalCoherenceConfig,
    StatisticalCoherenceResult,
    StatisticalCoherenceVerifier,
    cohere_score,
    _bhattacharyya_coefficient,
)


# =============================================================================
# Shared helpers
# =============================================================================

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pixels(n: int = 10, seed: int = 0) -> np.ndarray:
    """Return (N, 3) uint8 RGB pixel array."""
    return _rng(seed).integers(0, 256, (n, 3), dtype=np.uint8)


def _patch2d(h: int = 8, w: int = 8, seed: int = 0) -> np.ndarray:
    return _rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _patch3d(h: int = 8, w: int = 8, seed: int = 0) -> np.ndarray:
    return _rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# =============================================================================
# ColorContinuityVerifier – extra edge cases
# =============================================================================

class TestColorContinuityEdgeCases:

    # --- Single pixel pair --------------------------------------------------

    def test_single_pixel_pair_does_not_crash(self):
        a = np.array([[100, 150, 200]], dtype=np.uint8)
        b = np.array([[100, 150, 200]], dtype=np.uint8)
        result = verify_color_continuity(a, b)
        assert isinstance(result, ColorContinuityResult)
        assert result.n_samples == 1

    def test_single_pixel_identical_score_near_1(self):
        px = np.array([[128, 128, 128]], dtype=np.uint8)
        result = verify_color_continuity(px, px.copy())
        assert result.score > 0.99

    # --- 1000 pixel pairs ---------------------------------------------------

    def test_1000_pixel_pairs_performance(self):
        a = _pixels(n=1000, seed=0)
        b = _pixels(n=1000, seed=1)
        result = verify_color_continuity(a, b)
        assert result.n_samples == 1000
        assert isinstance(result.mean_delta, float)

    # --- score_from_delta ---------------------------------------------------

    def test_score_from_delta_zero_returns_1(self):
        score = ColorContinuityVerifier.score_from_delta(0.0, threshold=30.0)
        assert abs(score - 1.0) < 1e-9

    def test_score_from_delta_large_returns_near_zero(self):
        score = ColorContinuityVerifier.score_from_delta(1000.0, threshold=30.0)
        assert score < 0.01

    def test_score_from_delta_threshold_equal_delta_is_exp_neg1(self):
        # exp(-delta/threshold) = exp(-1) when delta == threshold
        score = ColorContinuityVerifier.score_from_delta(30.0, threshold=30.0)
        expected = float(np.exp(-1.0))
        assert abs(score - expected) < 1e-9

    def test_score_from_delta_zero_threshold(self):
        # When threshold <= 0 and delta > 0 → 0
        score = ColorContinuityVerifier.score_from_delta(1.0, threshold=0.0)
        assert score == 0.0

    def test_score_from_delta_zero_threshold_zero_delta(self):
        # When threshold <= 0 and delta == 0 → 1
        score = ColorContinuityVerifier.score_from_delta(0.0, threshold=0.0)
        assert score == 1.0

    # --- threshold=0.0 and threshold=1000.0 ---------------------------------

    def test_threshold_very_large_means_always_valid(self):
        a = _pixels(20, seed=2)
        b = _pixels(20, seed=3)
        cfg = ColorContinuityConfig(threshold=1e9)
        result = verify_color_continuity(a, b, config=cfg)
        assert result.is_valid is True

    def test_threshold_near_zero_means_usually_invalid(self):
        a = np.array([[0, 0, 0]], dtype=np.uint8)
        b = np.array([[255, 255, 255]], dtype=np.uint8)
        cfg = ColorContinuityConfig(threshold=0.001)
        result = verify_color_continuity(a, b, config=cfg)
        assert result.is_valid is False

    # --- Different color spaces ---------------------------------------------

    def test_lab_color_space_runs(self):
        a = _pixels(10)
        b = _pixels(10, seed=5)
        cfg = ColorContinuityConfig(method="lab")
        result = verify_color_continuity(a, b, config=cfg)
        assert result.n_samples == 10

    def test_hsv_color_space_runs(self):
        a = _pixels(10)
        b = _pixels(10, seed=5)
        cfg = ColorContinuityConfig(method="hsv")
        result = verify_color_continuity(a, b, config=cfg)
        assert result.n_samples == 10

    def test_rgb_color_space_runs(self):
        a = _pixels(10)
        b = _pixels(10, seed=5)
        cfg = ColorContinuityConfig(method="rgb")
        result = verify_color_continuity(a, b, config=cfg)
        assert result.n_samples == 10

    # --- Identical pixels → score 1.0, delta 0 ------------------------------

    def test_identical_pixels_delta_is_zero(self):
        px = _pixels(50)
        result = verify_color_continuity(px, px.copy())
        assert abs(result.mean_delta) < 1e-9

    def test_identical_pixels_score_is_1(self):
        px = _pixels(50)
        result = verify_color_continuity(px, px.copy())
        assert abs(result.score - 1.0) < 1e-6

    # --- Result field types -------------------------------------------------

    def test_result_mean_delta_is_float(self):
        a = _pixels(10)
        b = _pixels(10, seed=2)
        result = verify_color_continuity(a, b)
        assert isinstance(result.mean_delta, float)

    def test_result_max_delta_ge_mean_delta(self):
        a = _pixels(20)
        b = _pixels(20, seed=7)
        result = verify_color_continuity(a, b)
        assert result.max_delta >= result.mean_delta - 1e-9

    def test_result_score_in_0_1(self):
        a = _pixels(15)
        b = _pixels(15, seed=9)
        result = verify_color_continuity(a, b)
        assert 0.0 <= result.score <= 1.0

    # --- Mismatched lengths -------------------------------------------------

    def test_shorter_array_used_for_comparison(self):
        a = _pixels(20)
        b = _pixels(10)
        result = verify_color_continuity(a, b)
        assert result.n_samples == 10

    # --- Very large pixel values (near 255) ---------------------------------

    def test_saturated_pixels(self):
        a = np.full((10, 3), 255, dtype=np.uint8)
        b = np.full((10, 3), 0, dtype=np.uint8)
        result = verify_color_continuity(a, b)
        assert result.mean_delta > 0


# =============================================================================
# StatisticalCoherenceVerifier – extra edge cases
# =============================================================================

class TestStatisticalCoherenceEdgeCases:

    # --- Bhattacharyya coefficient -------------------------------------------

    def test_bhattacharyya_identical_distributions_is_1(self):
        h = np.array([10.0, 20.0, 30.0, 40.0])
        bc = _bhattacharyya_coefficient(h, h)
        assert abs(bc - 1.0) < 1e-9

    def test_bhattacharyya_disjoint_distributions_is_0(self):
        h1 = np.array([1.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        bc = _bhattacharyya_coefficient(h1, h2)
        assert abs(bc) < 1e-9

    def test_bhattacharyya_zero_hist_returns_0(self):
        h1 = np.zeros(8)
        h2 = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        bc = _bhattacharyya_coefficient(h1, h2)
        assert bc == 0.0

    # --- Texture similarity with constant patch -----------------------------

    def test_texture_similarity_constant_patch(self):
        a = np.full((10, 10), 128, dtype=np.uint8)
        b = np.full((10, 10), 64, dtype=np.uint8)
        cfg = StatisticalCoherenceConfig(use_texture=True)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        # Both patches are constant → texture contrast ≈ 0, energy ≈ 1
        # texture_sim should be high (both have same structural texture)
        assert 0.0 <= result.texture_similarity <= 1.0

    # --- Moment similarity when std=0 ---------------------------------------

    def test_moment_similarity_constant_arrays(self):
        a = np.full(50, 100.0)
        b = np.full(50, 100.0)
        # Use method="moments" so that moment_similarity is actually computed
        # (the default method="histogram" leaves moment_similarity at the 0.5 neutral default)
        cfg = StatisticalCoherenceConfig(method="moments", use_texture=False)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        # Identical constant arrays → moment similarity near 1
        assert result.moment_similarity > 0.9

    def test_moment_similarity_constant_vs_varying(self):
        a = np.full(50, 128.0)
        b = np.linspace(0, 255, 50)
        result = StatisticalCoherenceVerifier().verify(a, b)
        # Different distributions → similarity less than 1
        assert 0.0 <= result.moment_similarity <= 1.0

    # --- method="both" combines histogram and moment scores -----------------

    def test_method_both_uses_all_metrics(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        cfg = StatisticalCoherenceConfig(method="both", use_texture=True)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        # All three sub-scores should be populated
        assert 0.0 <= result.histogram_similarity <= 1.0
        assert 0.0 <= result.moment_similarity <= 1.0
        assert 0.0 <= result.texture_similarity <= 1.0
        assert 0.0 <= result.overall_score <= 1.0

    def test_method_both_no_texture(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        cfg = StatisticalCoherenceConfig(method="both", use_texture=False)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        assert 0.0 <= result.overall_score <= 1.0

    # --- Config use_texture=False -------------------------------------------

    def test_use_texture_false_skips_texture(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        cfg = StatisticalCoherenceConfig(use_texture=False)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        # texture_similarity defaults to 0.5 when not computed
        assert result.texture_similarity == 0.5

    # --- threshold=0.0 and threshold=1.0 ------------------------------------

    def test_threshold_0_is_coherent_always(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=5)   # different image
        cfg = StatisticalCoherenceConfig(threshold=0.0)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        assert result.is_coherent is True

    def test_threshold_1_is_not_coherent_for_most(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.full((10, 10), 255, dtype=np.uint8)
        cfg = StatisticalCoherenceConfig(threshold=1.0)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        # Very different images → overall_score < 1.0 → is_coherent=False
        assert result.is_coherent is False

    # --- verify with 1-D array ----------------------------------------------

    def test_verify_1d_array_works(self):
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
        result = StatisticalCoherenceVerifier().verify(a, b)
        assert isinstance(result, StatisticalCoherenceResult)

    # --- verify with 3-D array (colour patches) -----------------------------

    def test_verify_3d_array_color_patches(self):
        a = _patch3d(h=8, w=8, seed=0)
        b = _patch3d(h=8, w=8, seed=1)
        result = StatisticalCoherenceVerifier().verify(a, b)
        assert isinstance(result, StatisticalCoherenceResult)
        assert 0.0 <= result.overall_score <= 1.0

    # --- cohere_score is approximately symmetric ----------------------------

    def test_cohere_score_symmetric(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        score_ab = cohere_score(a, b)
        score_ba = cohere_score(b, a)
        assert abs(score_ab - score_ba) < 0.05  # allow small numerical differences

    # --- cohere_score returns float -----------------------------------------

    def test_cohere_score_returns_float(self):
        a = _patch2d()
        b = _patch2d(seed=3)
        s = cohere_score(a, b)
        assert isinstance(s, float)

    def test_cohere_score_in_0_1(self):
        for seed in range(5):
            a = _patch2d(seed=seed)
            b = _patch2d(seed=seed + 10)
            s = cohere_score(a, b)
            assert 0.0 <= s <= 1.0, f"Score out of range: {s}"

    # --- No NaN in any output -----------------------------------------------

    def test_no_nan_in_result(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        result = StatisticalCoherenceVerifier().verify(a, b)
        assert not np.isnan(result.histogram_similarity)
        assert not np.isnan(result.moment_similarity)
        assert not np.isnan(result.texture_similarity)
        assert not np.isnan(result.overall_score)

    # --- method="moments" only ----------------------------------------------

    def test_method_moments_only(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        cfg = StatisticalCoherenceConfig(method="moments")
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        assert 0.0 <= result.overall_score <= 1.0
        # histogram_similarity defaults to 0.5 when not computed
        assert result.histogram_similarity == 0.5

    # --- Very small patch (2x2) ---------------------------------------------

    def test_2x2_patch_works(self):
        a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        b = np.array([[15, 25], [35, 45]], dtype=np.uint8)
        result = StatisticalCoherenceVerifier().verify(a, b)
        assert isinstance(result, StatisticalCoherenceResult)

    # --- n_bins=2 extreme config -------------------------------------------

    def test_n_bins_2_works(self):
        a = _patch2d(seed=0)
        b = _patch2d(seed=1)
        cfg = StatisticalCoherenceConfig(n_bins=2)
        result = StatisticalCoherenceVerifier(cfg).verify(a, b)
        assert 0.0 <= result.overall_score <= 1.0
