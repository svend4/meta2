"""Tests for puzzle_reconstruction.verification.seam_analyzer"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

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


def make_gray(h=50, w=50, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_bgr(h=50, w=50, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_gradient_img(h=50, w=50):
    """Image with horizontal gradient."""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        img[:, i] = int(255 * i / (w - 1))
    return img


# ─── SeamAnalysis ─────────────────────────────────────────────────────────────

def test_seam_analysis_repr():
    sa = SeamAnalysis(
        idx1=0, idx2=1, side1=2, side2=0,
        brightness_score=0.8, gradient_score=0.7,
        texture_score=0.9, quality_score=0.8,
        profile_length=50,
    )
    r = repr(sa)
    assert "0" in r and "1" in r


def test_seam_analysis_quality_range():
    sa = SeamAnalysis(
        idx1=0, idx2=1, side1=2, side2=0,
        brightness_score=0.5, gradient_score=0.5,
        texture_score=0.5, quality_score=0.5,
        profile_length=50,
    )
    assert 0.0 <= sa.quality_score <= 1.0


# ─── extract_seam_profiles ────────────────────────────────────────────────────

def test_extract_seam_profiles_shape():
    img1 = make_gray(50, 60)
    img2 = make_gray(50, 60)
    p1, p2 = extract_seam_profiles(img1, img2, side1=2, side2=0)
    assert len(p1) == len(p2)
    assert len(p1) == 60


def test_extract_seam_profiles_side3():
    img1 = make_gray(50, 60)
    img2 = make_gray(50, 60)
    p1, p2 = extract_seam_profiles(img1, img2, side1=3, side2=1)
    assert len(p1) == len(p2) == 50


def test_extract_seam_profiles_invalid_side():
    img1 = make_gray()
    img2 = make_gray()
    with pytest.raises(ValueError):
        extract_seam_profiles(img1, img2, side1=4, side2=0)


def test_extract_seam_profiles_bgr():
    img1 = make_bgr(40, 60)
    img2 = make_bgr(40, 60)
    p1, p2 = extract_seam_profiles(img1, img2)
    assert len(p1) > 0


def test_extract_seam_profiles_different_sizes():
    img1 = make_gray(50, 40)
    img2 = make_gray(50, 60)
    p1, p2 = extract_seam_profiles(img1, img2, side1=2, side2=0)
    # Should be truncated to min
    assert len(p1) == len(p2) == 40


# ─── brightness_continuity ────────────────────────────────────────────────────

def test_brightness_continuity_identical():
    p = np.array([100.0, 120.0, 140.0, 160.0])
    score = brightness_continuity(p, p)
    assert score == pytest.approx(1.0)


def test_brightness_continuity_different():
    p1 = np.zeros(20)
    p2 = np.full(20, 255.0)
    score = brightness_continuity(p1, p2)
    assert score < 0.5


def test_brightness_continuity_empty():
    assert brightness_continuity(np.array([]), np.array([1.0])) == 0.0


def test_brightness_continuity_range():
    p1 = np.random.rand(50) * 255
    p2 = np.random.rand(50) * 255
    score = brightness_continuity(p1, p2)
    assert 0.0 <= score <= 1.0


# ─── gradient_continuity ──────────────────────────────────────────────────────

def test_gradient_continuity_identical():
    p = np.linspace(0, 100, 50)
    score = gradient_continuity(p, p)
    assert score == pytest.approx(1.0, abs=0.05)


def test_gradient_continuity_empty():
    assert gradient_continuity(np.array([1.0]), np.array([1.0])) == 0.0


def test_gradient_continuity_constant():
    p = np.full(20, 128.0)
    score = gradient_continuity(p, p)
    assert score == pytest.approx(1.0)


def test_gradient_continuity_range():
    p1 = np.linspace(0, 100, 50)
    p2 = np.random.rand(50) * 100
    score = gradient_continuity(p1, p2)
    assert 0.0 <= score <= 1.0


# ─── texture_continuity ───────────────────────────────────────────────────────

def test_texture_continuity_identical():
    p = np.array([0.0, 50.0, 100.0, 50.0, 0.0] * 10, dtype=float)
    score = texture_continuity(p, p)
    assert score == pytest.approx(1.0)


def test_texture_continuity_empty():
    assert texture_continuity(np.array([]), np.array([1.0])) == 0.0


def test_texture_continuity_uniform():
    p1 = np.full(20, 100.0)
    p2 = np.full(20, 200.0)
    score = texture_continuity(p1, p2)
    # Both uniform → score = 1.0
    assert score == pytest.approx(1.0)


def test_texture_continuity_range():
    p1 = np.random.rand(50) * 255
    p2 = np.random.rand(50) * 255
    score = texture_continuity(p1, p2)
    assert 0.0 <= score <= 1.0


# ─── analyze_seam ─────────────────────────────────────────────────────────────

def test_analyze_seam_basic():
    img1 = make_gray(50, 50, fill=100)
    img2 = make_gray(50, 50, fill=100)
    result = analyze_seam(img1, img2, idx1=0, idx2=1)
    assert isinstance(result, SeamAnalysis)
    assert result.idx1 == 0
    assert result.idx2 == 1


def test_analyze_seam_identical_images():
    img = make_gray(50, 50, fill=128)
    result = analyze_seam(img, img)
    assert 0.0 <= result.quality_score <= 1.0


def test_analyze_seam_custom_weights():
    img1 = make_gray(50, 50)
    img2 = make_gray(50, 50)
    result = analyze_seam(img1, img2, weights=(0.5, 0.3, 0.2))
    assert result.params.get("weights") == (0.5, 0.3, 0.2)


def test_analyze_seam_side_options():
    img1 = make_gray(50, 60)
    img2 = make_gray(50, 60)
    for side1, side2 in [(0, 2), (1, 3), (3, 1)]:
        result = analyze_seam(img1, img2, side1=side1, side2=side2)
        assert isinstance(result, SeamAnalysis)


def test_analyze_seam_profile_length():
    img1 = make_gray(50, 80)
    img2 = make_gray(50, 80)
    result = analyze_seam(img1, img2, side1=2, side2=0, border_px=5)
    assert result.profile_length == 80


# ─── score_seam_quality ───────────────────────────────────────────────────────

def test_score_seam_quality_range():
    sa = SeamAnalysis(
        idx1=0, idx2=1, side1=2, side2=0,
        brightness_score=0.8, gradient_score=0.7,
        texture_score=0.9, quality_score=0.8,
        profile_length=50,
    )
    score = score_seam_quality(sa)
    assert 0.0 <= score <= 1.0


def test_score_seam_quality_returns_float():
    sa = SeamAnalysis(
        idx1=0, idx2=1, side1=2, side2=0,
        brightness_score=0.5, gradient_score=0.5,
        texture_score=0.5, quality_score=0.5,
        profile_length=10,
    )
    assert isinstance(score_seam_quality(sa), float)


# ─── batch_analyze_seams ──────────────────────────────────────────────────────

def test_batch_analyze_seams_basic():
    images = [make_gray(50, 50, fill=f) for f in [100, 110, 120]]
    pairs = [(0, 1), (1, 2)]
    results = batch_analyze_seams(images, pairs)
    assert len(results) == 2


def test_batch_analyze_seams_empty():
    assert batch_analyze_seams([], []) == []


def test_batch_analyze_seams_with_side_pairs():
    images = [make_gray(50, 50) for _ in range(3)]
    pairs = [(0, 1), (1, 2)]
    side_pairs = [(2, 0), (1, 3)]
    results = batch_analyze_seams(images, pairs, side_pairs=side_pairs)
    assert len(results) == 2
    assert results[0].side1 == 2
    assert results[0].side2 == 0


def test_batch_analyze_seams_indices():
    images = [make_gray(50, 50) for _ in range(3)]
    pairs = [(0, 2)]
    results = batch_analyze_seams(images, pairs)
    assert results[0].idx1 == 0
    assert results[0].idx2 == 2
