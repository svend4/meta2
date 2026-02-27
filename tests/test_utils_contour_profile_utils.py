"""Tests for puzzle_reconstruction.utils.contour_profile_utils"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.contour_profile_utils import (
    ProfileConfig,
    ProfileMatchResult,
    sample_profile_along_contour,
    contour_curvature,
    smooth_profile,
    normalize_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    best_cyclic_offset,
    align_profiles,
    match_profiles,
    batch_match_profiles,
    top_k_profile_matches,
)

np.random.seed(42)


# ─── sample_profile_along_contour ─────────────────────────────────────────────

def test_sample_profile_shape():
    contour = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    result = sample_profile_along_contour(contour, n_samples=8)
    assert result.shape == (8, 2)


def test_sample_profile_single_point():
    contour = np.array([[5.0, 3.0]])
    result = sample_profile_along_contour(contour, n_samples=4)
    assert result.shape == (4, 2)
    assert np.allclose(result, [[5.0, 3.0]])


def test_sample_profile_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        sample_profile_along_contour(np.zeros((0, 2)), n_samples=4)


def test_sample_profile_n_samples_lt1_raises():
    contour = np.array([[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="n_samples"):
        sample_profile_along_contour(contour, n_samples=0)


def test_sample_profile_dtype():
    contour = np.array([[0, 0], [1, 1]], dtype=np.float32)
    result = sample_profile_along_contour(contour)
    assert result.dtype == np.float64


def test_sample_profile_endpoints():
    contour = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = sample_profile_along_contour(contour, n_samples=5)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[-1, 0] == pytest.approx(10.0)


# ─── contour_curvature ───────────────────────────────────────────────────────

def test_contour_curvature_shape():
    contour = np.array([[0, 0], [1, 0], [2, 1], [1, 2], [0, 1]], dtype=float)
    k = contour_curvature(contour)
    assert k.shape == (5,)


def test_contour_curvature_too_short():
    contour = np.array([[0, 0], [1, 1]], dtype=float)
    k = contour_curvature(contour)
    assert k.shape == (2,)
    assert np.allclose(k, 0.0)


def test_contour_curvature_straight_line_near_zero():
    contour = np.linspace([0, 0], [10, 0], 10)
    k = contour_curvature(contour)
    # For a straight line the cross product should be ~0
    assert np.abs(k).max() < 1e-6


# ─── smooth_profile ──────────────────────────────────────────────────────────

def test_smooth_profile_shape():
    v = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    result = smooth_profile(v, window=3)
    assert result.shape == v.shape


def test_smooth_profile_window_1_no_change():
    v = np.array([1.0, 5.0, 3.0])
    result = smooth_profile(v, window=1)
    assert np.allclose(result, v)


def test_smooth_profile_window_invalid():
    with pytest.raises(ValueError, match="window"):
        smooth_profile(np.array([1.0, 2.0]), window=0)


def test_smooth_profile_empty():
    v = np.array([])
    result = smooth_profile(v, window=3)
    assert result.shape == (0,)


# ─── normalize_profile ───────────────────────────────────────────────────────

def test_normalize_profile_range():
    v = np.array([2.0, 4.0, 6.0, 8.0])
    result = normalize_profile(v)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalize_profile_constant():
    v = np.array([3.0, 3.0, 3.0])
    result = normalize_profile(v)
    assert np.allclose(result, 1.0)


def test_normalize_profile_dtype():
    v = np.array([0.0, 1.0, 2.0])
    result = normalize_profile(v)
    assert result.dtype == np.float64


# ─── profile_l2_distance ─────────────────────────────────────────────────────

def test_profile_l2_distance_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert profile_l2_distance(v, v) == pytest.approx(0.0)


def test_profile_l2_distance_known():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert profile_l2_distance(a, b) == pytest.approx(5.0)


def test_profile_l2_distance_mismatch_raises():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        profile_l2_distance(a, b)


# ─── profile_cosine_similarity ───────────────────────────────────────────────

def test_profile_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert profile_cosine_similarity(v, v) == pytest.approx(1.0)


def test_profile_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert profile_cosine_similarity(a, b) == pytest.approx(0.0)


def test_profile_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert profile_cosine_similarity(a, b) == pytest.approx(0.0)


# ─── best_cyclic_offset ───────────────────────────────────────────────────────

def test_best_cyclic_offset_identical():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    offset, dist = best_cyclic_offset(v, v)
    assert dist == pytest.approx(0.0)


def test_best_cyclic_offset_known_shift():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.roll(a, 2)
    offset, dist = best_cyclic_offset(a, b)
    assert dist == pytest.approx(0.0)


def test_best_cyclic_offset_empty_raises():
    with pytest.raises(ValueError):
        best_cyclic_offset(np.array([]), np.array([]))


def test_best_cyclic_offset_mismatch_raises():
    with pytest.raises(ValueError):
        best_cyclic_offset(np.array([1.0, 2.0]), np.array([1.0]))


# ─── align_profiles ──────────────────────────────────────────────────────────

def test_align_profiles_returns_three():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 1.0, 2.0])
    a_al, b_al, offset = align_profiles(a, b)
    assert a_al.shape == a.shape
    assert b_al.shape == b.shape
    assert isinstance(offset, (int, np.integer))


# ─── match_profiles ──────────────────────────────────────────────────────────

def test_match_profiles_identical_score_near_one():
    v = np.array([0.2, 0.5, 0.8, 1.0, 0.6])
    result = match_profiles(v, v, cyclic=False, normalize=True)
    assert result.score >= 0.99


def test_match_profiles_l2_method():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([0.0, 0.5, 1.0])
    result = match_profiles(a, b, cyclic=False)
    assert result.method == "l2"


def test_match_profiles_cyclic_method():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.roll(a, 1).astype(float)
    result = match_profiles(a, b, cyclic=True)
    assert result.method == "cyclic"


def test_match_profiles_returns_result_type():
    a = np.random.rand(10)
    b = np.random.rand(10)
    result = match_profiles(a, b)
    assert isinstance(result, ProfileMatchResult)
    assert 0.0 <= result.score <= 1.0


def test_match_profiles_with_config():
    cfg = ProfileConfig(normalize=False)
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.5, 0.6, 0.7])
    result = match_profiles(a, b, cfg=cfg)
    assert result.score >= 0.9


# ─── batch_match_profiles ────────────────────────────────────────────────────

def test_batch_match_profiles_count():
    ref = np.array([1.0, 2.0, 3.0])
    candidates = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])]
    results = batch_match_profiles(ref, candidates)
    assert len(results) == 2


def test_batch_match_profiles_empty():
    ref = np.array([1.0, 2.0, 3.0])
    results = batch_match_profiles(ref, [])
    assert results == []


# ─── top_k_profile_matches ───────────────────────────────────────────────────

def test_top_k_profile_matches_order():
    results = [
        ProfileMatchResult(score=0.3, offset=0, distance=0.7),
        ProfileMatchResult(score=0.9, offset=1, distance=0.1),
        ProfileMatchResult(score=0.6, offset=0, distance=0.4),
    ]
    top2 = top_k_profile_matches(results, k=2)
    assert len(top2) == 2
    assert top2[0].score == pytest.approx(0.9)
    assert top2[1].score == pytest.approx(0.6)


def test_top_k_profile_matches_returns_fewer_if_not_enough():
    results = [ProfileMatchResult(score=0.5, offset=0, distance=0.5)]
    top5 = top_k_profile_matches(results, k=5)
    assert len(top5) == 1
