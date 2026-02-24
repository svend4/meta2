"""Extra tests for puzzle_reconstruction/utils/contour_profile_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(n=8) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def _profile(n=16) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _pmr(score=0.8, offset=0, dist=0.2) -> ProfileMatchResult:
    return ProfileMatchResult(score=score, offset=offset, distance=dist)


# ─── ProfileConfig ────────────────────────────────────────────────────────────

class TestProfileConfigExtra:
    def test_default_n_samples(self):
        assert ProfileConfig().n_samples == 64

    def test_default_smooth_window(self):
        assert ProfileConfig().smooth_window == 3

    def test_default_normalize(self):
        assert ProfileConfig().normalize is True

    def test_default_pad_mode(self):
        assert ProfileConfig().pad_mode == "edge"

    def test_custom_values(self):
        cfg = ProfileConfig(n_samples=32, smooth_window=5, normalize=False)
        assert cfg.n_samples == 32 and not cfg.normalize


# ─── ProfileMatchResult ───────────────────────────────────────────────────────

class TestProfileMatchResultExtra:
    def test_stores_score(self):
        assert _pmr(score=0.7).score == pytest.approx(0.7)

    def test_stores_offset(self):
        assert _pmr(offset=3).offset == 3

    def test_stores_distance(self):
        assert _pmr(dist=0.5).distance == pytest.approx(0.5)

    def test_default_method_dtw(self):
        r = ProfileMatchResult(score=0.5, offset=0, distance=0.3)
        assert r.method == "dtw"

    def test_repr_is_str(self):
        assert isinstance(repr(_pmr()), str)


# ─── sample_profile_along_contour ─────────────────────────────────────────────

class TestSampleProfileAlongContourExtra:
    def test_returns_ndarray(self):
        assert isinstance(sample_profile_along_contour(_square()), np.ndarray)

    def test_output_shape(self):
        out = sample_profile_along_contour(_square(), n_samples=32)
        assert out.shape == (32, 2)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sample_profile_along_contour(np.zeros((0, 2)))

    def test_n_samples_lt_1_raises(self):
        with pytest.raises(ValueError):
            sample_profile_along_contour(_square(), n_samples=0)

    def test_single_point(self):
        pts = np.array([[3.0, 4.0]])
        out = sample_profile_along_contour(pts, n_samples=8)
        assert out.shape == (8, 2)
        assert np.allclose(out, [3.0, 4.0])


# ─── contour_curvature ────────────────────────────────────────────────────────

class TestContourCurvatureExtra:
    def test_returns_ndarray(self):
        assert isinstance(contour_curvature(_square()), np.ndarray)

    def test_same_length(self):
        s = _square(10)
        assert len(contour_curvature(s)) == 10

    def test_less_than_3_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = contour_curvature(pts)
        assert np.allclose(out, 0.0)

    def test_dtype_float64(self):
        assert contour_curvature(_square()).dtype == np.float64


# ─── smooth_profile ───────────────────────────────────────────────────────────

class TestSmoothProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(smooth_profile(_profile()), np.ndarray)

    def test_same_length(self):
        v = _profile(16)
        assert len(smooth_profile(v)) == 16

    def test_window_1_identity(self):
        v = _profile(8)
        np.testing.assert_allclose(smooth_profile(v, window=1), v)

    def test_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            smooth_profile(_profile(), window=0)

    def test_empty_input(self):
        out = smooth_profile(np.array([]), window=3)
        assert len(out) == 0


# ─── normalize_profile ────────────────────────────────────────────────────────

class TestNormalizeProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_profile(_profile()), np.ndarray)

    def test_min_is_zero(self):
        v = np.array([2.0, 5.0, 8.0])
        out = normalize_profile(v)
        assert out.min() == pytest.approx(0.0)

    def test_max_is_one(self):
        v = np.array([2.0, 5.0, 8.0])
        out = normalize_profile(v)
        assert out.max() == pytest.approx(1.0)

    def test_constant_returns_ones(self):
        v = np.full(5, 3.0)
        out = normalize_profile(v)
        assert np.allclose(out, 1.0)


# ─── profile_l2_distance ──────────────────────────────────────────────────────

class TestProfileL2DistanceExtra:
    def test_returns_float(self):
        v = _profile(8)
        assert isinstance(profile_l2_distance(v, v), float)

    def test_identical_is_zero(self):
        v = _profile(8)
        assert profile_l2_distance(v, v) == pytest.approx(0.0)

    def test_nonneg(self):
        a = _profile(8)
        b = _profile(8)[::-1]
        assert profile_l2_distance(a, b) >= 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            profile_l2_distance(_profile(8), _profile(12))


# ─── profile_cosine_similarity ────────────────────────────────────────────────

class TestProfileCosineSimilarityExtra:
    def test_returns_float(self):
        v = _profile(8)
        assert isinstance(profile_cosine_similarity(v, v), float)

    def test_identical_is_one(self):
        v = np.ones(8)
        assert profile_cosine_similarity(v, v) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        v = np.zeros(8)
        w = _profile(8)
        assert profile_cosine_similarity(v, w) == pytest.approx(0.0)

    def test_in_range(self):
        a = _profile(8)
        b = _profile(8)[::-1]
        sim = profile_cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ─── best_cyclic_offset ───────────────────────────────────────────────────────

class TestBestCyclicOffsetExtra:
    def test_returns_tuple(self):
        v = _profile(8)
        result = best_cyclic_offset(v, v)
        assert isinstance(result, tuple) and len(result) == 2

    def test_identical_min_distance_near_zero(self):
        v = _profile(8)
        _, dist = best_cyclic_offset(v, v)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            best_cyclic_offset(_profile(8), _profile(12))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            best_cyclic_offset(np.array([]), np.array([]))


# ─── align_profiles ───────────────────────────────────────────────────────────

class TestAlignProfilesExtra:
    def test_returns_tuple_3(self):
        v = _profile(8)
        result = align_profiles(v, v)
        assert isinstance(result, tuple) and len(result) == 3

    def test_shapes_preserved(self):
        v = _profile(8)
        a, b, offset = align_profiles(v, v)
        assert len(a) == 8 and len(b) == 8


# ─── match_profiles ───────────────────────────────────────────────────────────

class TestMatchProfilesExtra:
    def test_returns_profile_match_result(self):
        v = _profile(8)
        assert isinstance(match_profiles(v, v), ProfileMatchResult)

    def test_identical_high_score(self):
        v = _profile(8)
        r = match_profiles(v, v)
        assert r.score >= 0.99

    def test_score_in_range(self):
        a = _profile(8)
        b = _profile(8)[::-1]
        r = match_profiles(a, b)
        assert 0.0 <= r.score <= 1.0

    def test_cyclic_returns_result(self):
        v = _profile(8)
        r = match_profiles(v, v, cyclic=True)
        assert isinstance(r, ProfileMatchResult)


# ─── batch_match_profiles ─────────────────────────────────────────────────────

class TestBatchMatchProfilesExtra:
    def test_returns_list(self):
        v = _profile(8)
        result = batch_match_profiles(v, [v])
        assert isinstance(result, list)

    def test_length_matches(self):
        v = _profile(8)
        result = batch_match_profiles(v, [v, v, v])
        assert len(result) == 3

    def test_empty_candidates(self):
        assert batch_match_profiles(_profile(8), []) == []


# ─── top_k_profile_matches ────────────────────────────────────────────────────

class TestTopKProfileMatchesExtra:
    def test_returns_list(self):
        results = [_pmr(score=0.5), _pmr(score=0.9)]
        assert isinstance(top_k_profile_matches(results, 1), list)

    def test_length_at_most_k(self):
        results = [_pmr(score=float(i) / 5) for i in range(5)]
        assert len(top_k_profile_matches(results, 3)) == 3

    def test_descending_order(self):
        results = [_pmr(score=0.2), _pmr(score=0.9), _pmr(score=0.5)]
        top = top_k_profile_matches(results, 3)
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)
