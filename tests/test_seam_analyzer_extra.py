"""Extra tests for puzzle_reconstruction/verification/seam_analyzer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.seam_analyzer import (
    SeamAnalysis,
    extract_seam_profiles,
    brightness_continuity,
    gradient_continuity,
    texture_continuity,
    analyze_seam,
    score_seam_quality,
    batch_analyze_seams,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(val=128, h=32, w=32):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(val=128, h=32, w=32):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _ramp(h=32, w=32):
    row = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(row, (h, 1)).astype(np.uint8)


def _noisy(seed=0, h=32, w=32):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ─── SeamAnalysis (extra) ─────────────────────────────────────────────────────

class TestSeamAnalysisExtra:
    def test_brightness_score_stored(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.9, 0.7, 0.8, 0.8, 32)
        assert sa.brightness_score == pytest.approx(0.9)

    def test_gradient_score_stored(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.9, 0.7, 0.8, 0.8, 32)
        assert sa.gradient_score == pytest.approx(0.7)

    def test_texture_score_stored(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.9, 0.7, 0.8, 0.8, 32)
        assert sa.texture_score == pytest.approx(0.8)

    def test_quality_score_stored(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.9, 0.7, 0.8, 0.8, 32)
        assert sa.quality_score == pytest.approx(0.8)

    def test_quality_score_zero_valid(self):
        sa = SeamAnalysis(0, 1, 0, 2, 0.0, 0.0, 0.0, 0.0, 16)
        assert sa.quality_score == pytest.approx(0.0)

    def test_quality_score_one_valid(self):
        sa = SeamAnalysis(0, 1, 0, 2, 1.0, 1.0, 1.0, 1.0, 16)
        assert sa.quality_score == pytest.approx(1.0)

    def test_profile_length_stored(self):
        sa = SeamAnalysis(3, 4, 1, 3, 0.5, 0.5, 0.5, 0.5, 100)
        assert sa.profile_length == 100

    def test_custom_params_stored(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.5, 0.5, 0.5, 0.5, 32,
                          params={"border_px": 8, "method": "fast"})
        assert sa.params["border_px"] == 8
        assert sa.params["method"] == "fast"

    def test_large_idx_values(self):
        sa = SeamAnalysis(999, 1000, 3, 1, 0.7, 0.6, 0.8, 0.7, 64)
        assert sa.idx1 == 999
        assert sa.idx2 == 1000


# ─── extract_seam_profiles (extra) ────────────────────────────────────────────

class TestExtractSeamProfilesExtra:
    def test_side0_top_values_uniform(self):
        img = _gray(200)
        p, _ = extract_seam_profiles(img, img, side1=0, side2=0)
        assert np.all(p == pytest.approx(200.0, abs=1))

    def test_side1_right_values_uniform(self):
        img = _gray(80)
        p, _ = extract_seam_profiles(img, img, side1=1, side2=1)
        assert np.all(p == pytest.approx(80.0, abs=1))

    def test_larger_border_px_same_length(self):
        img = _gray(100, h=64, w=64)
        p1a, _ = extract_seam_profiles(img, img, side1=2, side2=0, border_px=4)
        p1b, _ = extract_seam_profiles(img, img, side1=2, side2=0, border_px=12)
        assert len(p1a) == len(p1b)

    def test_profiles_are_float64(self):
        img = _bgr(50)
        p1, p2 = extract_seam_profiles(img, img, side1=2, side2=0)
        assert p1.dtype == np.float64
        assert p2.dtype == np.float64

    def test_side3_length_equals_height(self):
        img = _gray(100, h=48, w=64)
        p, _ = extract_seam_profiles(img, img, side1=3, side2=1)
        assert len(p) == 48

    def test_bgr_profile_nonneg(self):
        img = _bgr(100)
        p1, p2 = extract_seam_profiles(img, img, side1=0, side2=0)
        assert (p1 >= 0).all()
        assert (p2 >= 0).all()

    def test_equal_size_images_equal_profiles(self):
        img = _gray(150)
        p1, p2 = extract_seam_profiles(img, img, side1=2, side2=0)
        np.testing.assert_allclose(p1, p2, atol=1)

    def test_profiles_1d(self):
        img = _gray(100)
        p1, p2 = extract_seam_profiles(img, img, side1=2, side2=0)
        assert p1.ndim == 1
        assert p2.ndim == 1


# ─── brightness_continuity (extra) ────────────────────────────────────────────

class TestBrightnessContinuityExtra:
    def test_returns_float(self):
        p = np.linspace(0, 200, 30, dtype=np.float64)
        assert isinstance(brightness_continuity(p, p), float)

    def test_near_identical_high_score(self):
        p1 = np.linspace(100, 150, 50, dtype=np.float64)
        p2 = p1 + 1.0
        assert brightness_continuity(p1, p2) > 0.99

    def test_small_diff_custom_max(self):
        p1 = np.zeros(20, dtype=np.float64)
        p2 = np.full(20, 10.0)
        # max_diff=50 → score = 1 - 10/50 = 0.8
        assert brightness_continuity(p1, p2, max_diff=50.0) == pytest.approx(0.8, abs=0.01)

    def test_result_nonneg(self):
        rng = np.random.default_rng(5)
        p1 = rng.uniform(0, 255, 40).astype(np.float64)
        p2 = rng.uniform(0, 255, 40).astype(np.float64)
        assert brightness_continuity(p1, p2) >= 0.0

    def test_result_le_one(self):
        rng = np.random.default_rng(6)
        p1 = rng.uniform(0, 255, 40).astype(np.float64)
        p2 = rng.uniform(0, 255, 40).astype(np.float64)
        assert brightness_continuity(p1, p2) <= 1.0

    def test_order_symmetric(self):
        p1 = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        p2 = np.array([20.0, 70.0, 120.0], dtype=np.float64)
        assert brightness_continuity(p1, p2) == pytest.approx(
            brightness_continuity(p2, p1), abs=1e-6
        )


# ─── gradient_continuity (extra) ──────────────────────────────────────────────

class TestGradientContinuityExtra:
    def test_returns_float(self):
        p = np.linspace(0, 100, 20, dtype=np.float64)
        assert isinstance(gradient_continuity(p, p), float)

    def test_result_nonneg(self):
        p1 = np.random.default_rng(10).uniform(0, 200, 40).astype(np.float64)
        p2 = np.random.default_rng(11).uniform(0, 200, 40).astype(np.float64)
        assert gradient_continuity(p1, p2) >= 0.0

    def test_result_le_one(self):
        p1 = np.random.default_rng(12).uniform(0, 200, 40).astype(np.float64)
        p2 = np.random.default_rng(13).uniform(0, 200, 40).astype(np.float64)
        assert gradient_continuity(p1, p2) <= 1.0

    def test_both_ramps_same_direction_high(self):
        p1 = np.linspace(0, 100, 30, dtype=np.float64)
        p2 = np.linspace(0, 100, 30, dtype=np.float64)
        assert gradient_continuity(p1, p2) == pytest.approx(1.0, abs=1e-5)

    def test_two_element_returns_value(self):
        p = np.array([10.0, 20.0], dtype=np.float64)
        r = gradient_continuity(p, p)
        assert 0.0 <= r <= 1.0

    def test_zero_sum_gradients_match(self):
        p1 = np.array([5.0, 0.0, 5.0, 0.0, 5.0], dtype=np.float64)
        r = gradient_continuity(p1, p1)
        assert r == pytest.approx(1.0, abs=1e-5)


# ─── texture_continuity (extra) ───────────────────────────────────────────────

class TestTextureContinuityExtra:
    def test_returns_float(self):
        p = np.linspace(0, 200, 30, dtype=np.float64)
        assert isinstance(texture_continuity(p, p), float)

    def test_result_nonneg(self):
        p1 = np.random.default_rng(20).uniform(0, 200, 50).astype(np.float64)
        p2 = np.random.default_rng(21).uniform(0, 200, 50).astype(np.float64)
        assert texture_continuity(p1, p2) >= 0.0

    def test_result_le_one(self):
        p1 = np.random.default_rng(22).uniform(0, 200, 50).astype(np.float64)
        p2 = np.random.default_rng(23).uniform(0, 200, 50).astype(np.float64)
        assert texture_continuity(p1, p2) <= 1.0

    def test_large_vs_small_variance_low(self):
        rng = np.random.default_rng(25)
        p_wide = rng.normal(100, 50, 50).astype(np.float64)
        p_narrow = rng.normal(100, 1, 50).astype(np.float64)
        r = texture_continuity(p_wide, p_narrow)
        assert r < 0.5

    def test_symmetric(self):
        rng = np.random.default_rng(30)
        p1 = rng.uniform(0, 200, 40).astype(np.float64)
        p2 = rng.uniform(0, 200, 40).astype(np.float64)
        assert texture_continuity(p1, p2) == pytest.approx(
            texture_continuity(p2, p1), abs=1e-6
        )

    def test_single_element_returns_zero_or_one(self):
        p = np.array([100.0], dtype=np.float64)
        r = texture_continuity(p, p)
        assert 0.0 <= r <= 1.0


# ─── analyze_seam (extra) ─────────────────────────────────────────────────────

class TestAnalyzeSeamExtra:
    def test_quality_score_nonneg(self):
        sa = analyze_seam(_noisy(1), _noisy(2))
        assert sa.quality_score >= 0.0

    def test_quality_score_le_one(self):
        sa = analyze_seam(_noisy(3), _noisy(4))
        assert sa.quality_score <= 1.0

    def test_brightness_score_in_range(self):
        sa = analyze_seam(_gray(100), _gray(200))
        assert 0.0 <= sa.brightness_score <= 1.0

    def test_gradient_score_in_range(self):
        sa = analyze_seam(_ramp(), _ramp())
        assert 0.0 <= sa.gradient_score <= 1.0

    def test_texture_score_in_range(self):
        sa = analyze_seam(_noisy(5), _noisy(6))
        assert 0.0 <= sa.texture_score <= 1.0

    def test_default_idx(self):
        sa = analyze_seam(_gray(100), _gray(100))
        assert sa.idx1 == 0
        assert sa.idx2 == 1

    def test_default_sides_2_0(self):
        sa = analyze_seam(_gray(100), _gray(100))
        assert sa.side1 == 2
        assert sa.side2 == 0

    def test_identical_gray_high_quality(self):
        img = _gray(128)
        sa = analyze_seam(img, img)
        assert sa.quality_score > 0.8

    def test_bgr_quality_in_range(self):
        img = _bgr(100)
        sa = analyze_seam(img, img)
        assert 0.0 <= sa.quality_score <= 1.0

    def test_rectangular_images(self):
        img1 = np.full((32, 64), 100, dtype=np.uint8)
        img2 = np.full((32, 64), 100, dtype=np.uint8)
        sa = analyze_seam(img1, img2, side1=2, side2=0)
        assert sa.profile_length > 0

    def test_custom_weights_uniform(self):
        img = _gray(128)
        w = (1.0, 1.0, 1.0)
        sa = analyze_seam(img, img, weights=w)
        assert sa.params["weights"] == w


# ─── score_seam_quality (extra) ───────────────────────────────────────────────

class TestScoreSeamQualityExtra:
    def test_perfect_score(self):
        sa = SeamAnalysis(0, 1, 2, 0, 1.0, 1.0, 1.0, 1.0, 32)
        r = score_seam_quality(sa)
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_zero_quality_returns_zero(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.0, 0.0, 0.0, 0.0, 32)
        r = score_seam_quality(sa)
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_middle_quality(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.5, 0.5, 0.5, 0.5, 32)
        r = score_seam_quality(sa)
        assert 0.0 <= r <= 1.0

    def test_clamped_above_one(self):
        sa = SeamAnalysis(0, 1, 2, 0, 1.0, 1.0, 1.0, 1.5, 32)
        assert score_seam_quality(sa) <= 1.0

    def test_clamped_below_zero(self):
        sa = SeamAnalysis(0, 1, 2, 0, 0.0, 0.0, 0.0, -0.5, 32)
        assert score_seam_quality(sa) >= 0.0


# ─── batch_analyze_seams (extra) ──────────────────────────────────────────────

class TestBatchAnalyzeSeamsExtra:
    def test_single_pair_returns_one(self):
        imgs = [_gray(100), _gray(100)]
        result = batch_analyze_seams(imgs, [(0, 1)])
        assert len(result) == 1

    def test_all_qualities_in_range(self):
        imgs = [_noisy(i) for i in range(5)]
        pairs = [(i, i + 1) for i in range(4)]
        for sa in batch_analyze_seams(imgs, pairs):
            assert 0.0 <= sa.quality_score <= 1.0

    def test_identical_images_high_quality(self):
        img = _gray(128)
        result = batch_analyze_seams([img, img], [(0, 1)])
        assert result[0].quality_score > 0.8

    def test_side_pairs_length_mismatch_uses_default(self):
        imgs = [_gray(100), _gray(100), _gray(100)]
        pairs = [(0, 1), (1, 2)]
        result = batch_analyze_seams(imgs, pairs, side_pairs=None)
        for sa in result:
            assert sa.side1 == 2
            assert sa.side2 == 0

    def test_large_batch(self):
        imgs = [_gray(i * 10 % 256) for i in range(8)]
        pairs = [(i, i + 1) for i in range(7)]
        result = batch_analyze_seams(imgs, pairs)
        assert len(result) == 7

    def test_bgr_images_batch(self):
        imgs = [_bgr(100), _bgr(200)]
        result = batch_analyze_seams(imgs, [(0, 1)])
        assert isinstance(result[0], SeamAnalysis)

    def test_results_are_seam_analysis(self):
        imgs = [_gray(50), _gray(100), _gray(150)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        result = batch_analyze_seams(imgs, pairs)
        for sa in result:
            assert isinstance(sa, SeamAnalysis)
