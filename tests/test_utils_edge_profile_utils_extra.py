"""Extra tests for puzzle_reconstruction/utils/edge_profile_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.edge_profile_utils import (
    EdgeProfileConfig,
    EdgeProfile,
    build_edge_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    profile_correlation,
    resample_profile,
    flip_profile,
    mean_profile,
    batch_build_profiles,
    pairwise_l2_matrix,
    best_matching_profile,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pts_horiz(n=16) -> np.ndarray:
    """Horizontal line of n points for a top/bottom profile."""
    x = np.linspace(0.0, 10.0, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def _profile(values=None, side="top") -> EdgeProfile:
    if values is None:
        values = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    return EdgeProfile(values=values, side=side)


# ─── EdgeProfileConfig ────────────────────────────────────────────────────────

class TestEdgeProfileConfigExtra:
    def test_default_n_samples(self):
        assert EdgeProfileConfig().n_samples == 64

    def test_default_smooth_sigma(self):
        assert EdgeProfileConfig().smooth_sigma == pytest.approx(1.0)

    def test_default_normalize(self):
        assert EdgeProfileConfig().normalize is True

    def test_custom_values(self):
        cfg = EdgeProfileConfig(n_samples=32, smooth_sigma=0.5, normalize=False)
        assert cfg.n_samples == 32 and not cfg.normalize


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

class TestEdgeProfileExtra:
    def test_values_converted_to_float32(self):
        p = _profile(np.array([1, 2, 3]))
        assert p.values.dtype == np.float32

    def test_values_flattened(self):
        p = EdgeProfile(values=np.ones((4, 1)))
        assert p.values.ndim == 1

    def test_n_samples_from_values(self):
        p = _profile(np.ones(20))
        assert p.n_samples == 20

    def test_len(self):
        p = _profile(np.ones(12))
        assert len(p) == 12

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            EdgeProfile(values=np.ones(8), side="diagonal")

    def test_valid_sides(self):
        for side in ("top", "bottom", "left", "right", "unknown"):
            p = EdgeProfile(values=np.ones(8), side=side)
            assert p.side == side

    def test_repr_is_str(self):
        assert isinstance(repr(_profile()), str)


# ─── build_edge_profile ───────────────────────────────────────────────────────

class TestBuildEdgeProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(build_edge_profile(_pts_horiz(), side="top"), EdgeProfile)

    def test_n_samples_from_cfg(self):
        cfg = EdgeProfileConfig(n_samples=32)
        p = build_edge_profile(_pts_horiz(), side="top", cfg=cfg)
        assert p.n_samples == 32

    def test_side_stored(self):
        p = build_edge_profile(_pts_horiz(), side="bottom")
        assert p.side == "bottom"

    def test_vertical_side(self):
        pts = np.column_stack([np.zeros(16), np.linspace(0.0, 10.0, 16)])
        p = build_edge_profile(pts, side="left")
        assert isinstance(p, EdgeProfile)


# ─── profile_l2_distance ──────────────────────────────────────────────────────

class TestProfileL2DistanceExtra:
    def test_returns_float(self):
        p = _profile()
        assert isinstance(profile_l2_distance(p, p), float)

    def test_identical_is_zero(self):
        p = _profile(np.ones(16))
        assert profile_l2_distance(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_nonneg(self):
        a = _profile(np.zeros(16))
        b = _profile(np.ones(16))
        assert profile_l2_distance(a, b) >= 0.0


# ─── profile_cosine_similarity ────────────────────────────────────────────────

class TestProfileCosineSimilarityExtra:
    def test_returns_float(self):
        p = _profile()
        assert isinstance(profile_cosine_similarity(p, p), float)

    def test_identical_is_one(self):
        p = _profile(np.ones(8))
        assert profile_cosine_similarity(p, p) == pytest.approx(1.0, abs=1e-5)

    def test_in_range(self):
        a = _profile(np.linspace(0.0, 1.0, 8))
        b = _profile(np.linspace(1.0, 0.0, 8))
        sim = profile_cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ─── profile_correlation ──────────────────────────────────────────────────────

class TestProfileCorrelationExtra:
    def test_returns_float(self):
        p = _profile()
        assert isinstance(profile_correlation(p, p), float)

    def test_identical_high(self):
        p = _profile(np.linspace(0.0, 1.0, 16))
        r = profile_correlation(p, p)
        assert r >= 0.99

    def test_in_range(self):
        a = _profile(np.linspace(0.0, 1.0, 8))
        b = _profile(np.linspace(1.0, 0.0, 8))
        r = profile_correlation(a, b)
        assert -1.0 <= r <= 1.0


# ─── resample_profile ─────────────────────────────────────────────────────────

class TestResampleProfileExtra:
    def test_returns_edge_profile(self):
        p = _profile(np.ones(16))
        assert isinstance(resample_profile(p, 32), EdgeProfile)

    def test_new_n_samples(self):
        p = _profile(np.ones(16))
        r = resample_profile(p, 32)
        assert r.n_samples == 32

    def test_constant_preserved(self):
        p = _profile(np.ones(8))
        r = resample_profile(p, 16)
        assert np.allclose(r.values, 1.0, atol=1e-5)


# ─── flip_profile ─────────────────────────────────────────────────────────────

class TestFlipProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(flip_profile(_profile()), EdgeProfile)

    def test_values_reversed(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        p = _profile(vals)
        r = flip_profile(p)
        np.testing.assert_allclose(r.values, vals[::-1])

    def test_same_n_samples(self):
        p = _profile(np.ones(16))
        assert flip_profile(p).n_samples == 16


# ─── mean_profile ─────────────────────────────────────────────────────────────

class TestMeanProfileExtra:
    def test_returns_edge_profile(self):
        profiles = [_profile(np.ones(8)), _profile(np.zeros(8))]
        assert isinstance(mean_profile(profiles), EdgeProfile)

    def test_mean_values(self):
        a = _profile(np.ones(8))
        b = _profile(np.zeros(8))
        r = mean_profile([a, b])
        np.testing.assert_allclose(r.values, 0.5, atol=1e-5)

    def test_single_profile(self):
        p = _profile(np.array([1.0, 2.0, 3.0]))
        r = mean_profile([p])
        np.testing.assert_allclose(r.values, p.values, atol=1e-5)


# ─── batch_build_profiles ─────────────────────────────────────────────────────

class TestBatchBuildProfilesExtra:
    def test_returns_list(self):
        result = batch_build_profiles([_pts_horiz(), _pts_horiz()],
                                       sides=["top", "bottom"])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_build_profiles([_pts_horiz(), _pts_horiz()],
                                       sides=["top", "bottom"])
        assert len(result) == 2

    def test_each_is_profile(self):
        for p in batch_build_profiles([_pts_horiz()], sides=["top"]):
            assert isinstance(p, EdgeProfile)


# ─── pairwise_l2_matrix ───────────────────────────────────────────────────────

class TestPairwiseL2MatrixExtra:
    def test_returns_ndarray(self):
        profiles = [_profile(np.ones(8)), _profile(np.zeros(8))]
        assert isinstance(pairwise_l2_matrix(profiles), np.ndarray)

    def test_square_shape(self):
        n = 4
        profiles = [_profile(np.ones(8)) for _ in range(n)]
        mat = pairwise_l2_matrix(profiles)
        assert mat.shape == (n, n)

    def test_diagonal_is_zero(self):
        profiles = [_profile(np.ones(8)), _profile(np.zeros(8))]
        mat = pairwise_l2_matrix(profiles)
        assert np.allclose(np.diag(mat), 0.0, atol=1e-5)


# ─── best_matching_profile ────────────────────────────────────────────────────

class TestBestMatchingProfileExtra:
    def test_returns_tuple(self):
        query = _profile(np.ones(8))
        candidates = [_profile(np.ones(8)), _profile(np.zeros(8))]
        result = best_matching_profile(query, candidates)
        assert isinstance(result, tuple)

    def test_identical_is_best(self):
        query = _profile(np.ones(8))
        candidates = [_profile(np.zeros(8)), _profile(np.ones(8))]
        idx, dist = best_matching_profile(query, candidates)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_empty_candidates_raises(self):
        query = _profile(np.ones(8))
        with pytest.raises((ValueError, IndexError)):
            best_matching_profile(query, [])
