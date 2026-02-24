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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gradient_h(h=64, w=64):
    """Horizontal gradient image."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


# ─── SeamAnalysis ────────────────────────────────────────────────────────────

class TestSeamAnalysisExtra:
    def test_creation(self):
        sa = SeamAnalysis(
            idx1=0, idx2=1, side1=2, side2=0,
            brightness_score=0.9, gradient_score=0.8,
            texture_score=0.7, quality_score=0.8,
            profile_length=64,
        )
        assert sa.idx1 == 0
        assert sa.quality_score == pytest.approx(0.8)

    def test_repr(self):
        sa = SeamAnalysis(
            idx1=0, idx2=1, side1=2, side2=0,
            brightness_score=0.9, gradient_score=0.8,
            texture_score=0.7, quality_score=0.8,
            profile_length=64,
        )
        s = repr(sa)
        assert "idx1=0" in s
        assert "quality" in s


# ─── extract_seam_profiles ───────────────────────────────────────────────────

class TestExtractSeamProfilesExtra:
    def test_basic(self):
        p1, p2 = extract_seam_profiles(_gray(), _gray())
        assert len(p1) == len(p2)
        assert len(p1) > 0

    def test_bgr(self):
        p1, p2 = extract_seam_profiles(_bgr(), _bgr())
        assert len(p1) == len(p2)

    def test_different_sizes(self):
        p1, p2 = extract_seam_profiles(_gray(64, 100), _gray(64, 50))
        # Truncated to min length
        assert len(p1) == len(p2)

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            extract_seam_profiles(_gray(), _gray(), side1=5)

    def test_all_sides(self):
        for s1 in range(4):
            for s2 in range(4):
                p1, p2 = extract_seam_profiles(_gray(), _gray(), side1=s1, side2=s2)
                assert len(p1) > 0


# ─── brightness_continuity ───────────────────────────────────────────────────

class TestBrightnessContinuityExtra:
    def test_identical(self):
        p = np.full(64, 128.0)
        score = brightness_continuity(p, p)
        assert score == pytest.approx(1.0)

    def test_opposite(self):
        p1 = np.zeros(64)
        p2 = np.full(64, 255.0)
        score = brightness_continuity(p1, p2)
        assert score == pytest.approx(0.0)

    def test_empty(self):
        assert brightness_continuity(np.array([]), np.array([128.0])) == 0.0

    def test_partial_diff(self):
        p1 = np.full(64, 100.0)
        p2 = np.full(64, 150.0)
        score = brightness_continuity(p1, p2)
        assert 0.0 < score < 1.0


# ─── gradient_continuity ────────────────────────────────────────────────────

class TestGradientContinuityExtra:
    def test_both_flat(self):
        p = np.full(64, 128.0)
        score = gradient_continuity(p, p)
        assert score == pytest.approx(1.0)

    def test_one_flat(self):
        p1 = np.full(64, 128.0)
        # Use a profile with varying gradient (not constant diff)
        p2 = np.cumsum(np.random.RandomState(42).randn(64)) * 10 + 128
        score = gradient_continuity(p1, p2)
        # p1 gradient is flat (std~0), p2 gradient varies → returns 0.5
        assert score == pytest.approx(0.5)

    def test_same_gradient(self):
        p = np.linspace(0, 255, 64)
        score = gradient_continuity(p, p)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_short(self):
        assert gradient_continuity(np.array([1.0]), np.array([2.0])) == 0.0


# ─── texture_continuity ─────────────────────────────────────────────────────

class TestTextureContinuityExtra:
    def test_identical(self):
        p = np.linspace(0, 255, 64)
        score = texture_continuity(p, p)
        assert score == pytest.approx(1.0)

    def test_both_flat(self):
        p = np.full(64, 128.0)
        score = texture_continuity(p, p)
        assert score == pytest.approx(1.0)

    def test_different_std(self):
        p1 = np.full(64, 128.0)
        p2 = np.linspace(0, 255, 64)
        score = texture_continuity(p1, p2)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_empty(self):
        assert texture_continuity(np.array([]), np.array([1.0])) == 0.0


# ─── analyze_seam ───────────────────────────────────────────────────────────

class TestAnalyzeSeamExtra:
    def test_identical_images(self):
        sa = analyze_seam(_gray(), _gray())
        assert sa.brightness_score == pytest.approx(1.0)
        assert sa.quality_score >= 0.0

    def test_indices(self):
        sa = analyze_seam(_gray(), _gray(), idx1=5, idx2=10)
        assert sa.idx1 == 5
        assert sa.idx2 == 10

    def test_custom_weights(self):
        sa = analyze_seam(_gray(), _gray(), weights=(1.0, 0.0, 0.0))
        assert sa.quality_score == pytest.approx(sa.brightness_score)

    def test_profile_length(self):
        sa = analyze_seam(_gray(64, 64), _gray(64, 64))
        assert sa.profile_length > 0

    def test_bgr_images(self):
        sa = analyze_seam(_bgr(), _bgr())
        assert isinstance(sa, SeamAnalysis)


# ─── score_seam_quality ──────────────────────────────────────────────────────

class TestScoreSeamQualityExtra:
    def test_basic(self):
        sa = analyze_seam(_gray(), _gray())
        score = score_seam_quality(sa)
        assert 0.0 <= score <= 1.0

    def test_matches_analysis(self):
        sa = analyze_seam(_gray(), _gray())
        assert score_seam_quality(sa) == pytest.approx(sa.quality_score)


# ─── batch_analyze_seams ────────────────────────────────────────────────────

class TestBatchAnalyzeSeamsExtra:
    def test_empty(self):
        assert batch_analyze_seams([], []) == []

    def test_one_pair(self):
        images = [_gray(), _gray()]
        results = batch_analyze_seams(images, [(0, 1)])
        assert len(results) == 1
        assert isinstance(results[0], SeamAnalysis)

    def test_two_pairs(self):
        images = [_gray(), _gray(), _gray()]
        results = batch_analyze_seams(images, [(0, 1), (1, 2)])
        assert len(results) == 2

    def test_custom_sides(self):
        images = [_gray(), _gray()]
        results = batch_analyze_seams(images, [(0, 1)],
                                       side_pairs=[(1, 3)])
        assert results[0].side1 == 1
        assert results[0].side2 == 3
