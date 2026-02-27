"""Tests for puzzle_reconstruction.utils.edge_profile_utils"""
import numpy as np
import pytest
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

np.random.seed(42)


# ─── EdgeProfileConfig ───────────────────────────────────────────────────────

def test_edge_profile_config_defaults():
    cfg = EdgeProfileConfig()
    assert cfg.n_samples == 64
    assert cfg.smooth_sigma == pytest.approx(1.0)
    assert cfg.normalize is True


def test_edge_profile_config_custom():
    cfg = EdgeProfileConfig(n_samples=32, smooth_sigma=0.0, normalize=False)
    assert cfg.n_samples == 32
    assert cfg.smooth_sigma == 0.0


# ─── EdgeProfile ─────────────────────────────────────────────────────────────

def test_edge_profile_creation():
    values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    ep = EdgeProfile(values=values, side="top")
    assert ep.side == "top"
    assert ep.n_samples == 3
    assert len(ep) == 3


def test_edge_profile_invalid_side():
    with pytest.raises(ValueError, match="side"):
        EdgeProfile(values=np.array([0.0]), side="diagonal")


def test_edge_profile_values_cast_to_float32():
    ep = EdgeProfile(values=np.array([1.0, 2.0], dtype=np.float64))
    assert ep.values.dtype == np.float32


def test_edge_profile_repr():
    ep = EdgeProfile(values=np.array([0.0, 0.5, 1.0]))
    r = repr(ep)
    assert "EdgeProfile" in r


# ─── build_edge_profile ──────────────────────────────────────────────────────

def test_build_edge_profile_shape():
    pts = np.random.rand(20, 2) * 100
    cfg = EdgeProfileConfig(n_samples=32)
    ep = build_edge_profile(pts, side="top", cfg=cfg)
    assert ep.n_samples == 32
    assert ep.values.shape == (32,)


def test_build_edge_profile_normalized():
    pts = np.column_stack([np.linspace(0, 100, 20), np.linspace(0, 50, 20)])
    cfg = EdgeProfileConfig(n_samples=16, normalize=True, smooth_sigma=0.0)
    ep = build_edge_profile(pts, side="top", cfg=cfg)
    assert ep.values.min() >= -1e-5
    assert ep.values.max() <= 1.0 + 1e-5


def test_build_edge_profile_invalid_shape():
    pts = np.random.rand(10, 3)
    with pytest.raises(ValueError):
        build_edge_profile(pts, side="top")


def test_build_edge_profile_empty_points():
    pts = np.zeros((0, 2))
    cfg = EdgeProfileConfig(n_samples=8)
    ep = build_edge_profile(pts, side="bottom", cfg=cfg)
    assert ep.n_samples == 8
    assert np.allclose(ep.values, 0.0)


def test_build_edge_profile_3d_input():
    pts = np.random.rand(10, 1, 2) * 50
    cfg = EdgeProfileConfig(n_samples=8)
    ep = build_edge_profile(pts, side="left", cfg=cfg)
    assert ep.n_samples == 8


def test_build_edge_profile_sides():
    pts = np.column_stack([np.linspace(0, 100, 10), np.random.rand(10) * 10])
    for side in ["top", "bottom", "left", "right", "unknown"]:
        ep = build_edge_profile(pts, side=side)
        assert ep.side == side


# ─── profile_l2_distance ─────────────────────────────────────────────────────

def test_profile_l2_distance_identical():
    ep = EdgeProfile(values=np.array([0.2, 0.5, 0.8]))
    assert profile_l2_distance(ep, ep) == pytest.approx(0.0)


def test_profile_l2_distance_known():
    a = EdgeProfile(values=np.array([0.0, 0.0], dtype=np.float32))
    b = EdgeProfile(values=np.array([3.0, 4.0], dtype=np.float32))
    assert profile_l2_distance(a, b) == pytest.approx(5.0)


def test_profile_l2_distance_mismatch_raises():
    a = EdgeProfile(values=np.array([1.0, 2.0]))
    b = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        profile_l2_distance(a, b)


# ─── profile_cosine_similarity ───────────────────────────────────────────────

def test_profile_cosine_similarity_identical():
    ep = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    assert profile_cosine_similarity(ep, ep) == pytest.approx(1.0)


def test_profile_cosine_similarity_orthogonal():
    a = EdgeProfile(values=np.array([1.0, 0.0]))
    b = EdgeProfile(values=np.array([0.0, 1.0]))
    assert profile_cosine_similarity(a, b) == pytest.approx(0.0)


def test_profile_cosine_similarity_zero_vector():
    a = EdgeProfile(values=np.array([0.0, 0.0]))
    b = EdgeProfile(values=np.array([1.0, 2.0]))
    assert profile_cosine_similarity(a, b) == pytest.approx(0.0)


# ─── profile_correlation ─────────────────────────────────────────────────────

def test_profile_correlation_identical():
    ep = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    assert profile_correlation(ep, ep) == pytest.approx(1.0)


def test_profile_correlation_anticorrelated():
    a = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    b = EdgeProfile(values=np.array([3.0, 2.0, 1.0]))
    assert profile_correlation(a, b) == pytest.approx(-1.0)


def test_profile_correlation_single_sample():
    a = EdgeProfile(values=np.array([1.0]))
    b = EdgeProfile(values=np.array([2.0]))
    result = profile_correlation(a, b)
    assert result == pytest.approx(1.0)


# ─── resample_profile ────────────────────────────────────────────────────────

def test_resample_profile_shape():
    ep = EdgeProfile(values=np.array([0.0, 0.5, 1.0]))
    resampled = resample_profile(ep, n_samples=10)
    assert resampled.n_samples == 10


def test_resample_profile_preserves_side():
    ep = EdgeProfile(values=np.array([0.0, 1.0]), side="left")
    resampled = resample_profile(ep, n_samples=4)
    assert resampled.side == "left"


def test_resample_profile_invalid():
    ep = EdgeProfile(values=np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        resample_profile(ep, n_samples=0)


# ─── flip_profile ────────────────────────────────────────────────────────────

def test_flip_profile_reversed():
    ep = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    flipped = flip_profile(ep)
    assert np.allclose(flipped.values, [3.0, 2.0, 1.0])


def test_flip_profile_preserves_side():
    ep = EdgeProfile(values=np.array([0.0, 1.0]), side="right")
    flipped = flip_profile(ep)
    assert flipped.side == "right"


# ─── mean_profile ────────────────────────────────────────────────────────────

def test_mean_profile_basic():
    p1 = EdgeProfile(values=np.array([0.0, 1.0, 2.0]))
    p2 = EdgeProfile(values=np.array([2.0, 1.0, 0.0]))
    result = mean_profile([p1, p2])
    assert np.allclose(result.values, [1.0, 1.0, 1.0])


def test_mean_profile_empty():
    result = mean_profile([])
    assert len(result.values) == 0


def test_mean_profile_mismatch_raises():
    p1 = EdgeProfile(values=np.array([0.0, 1.0]))
    p2 = EdgeProfile(values=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        mean_profile([p1, p2])


# ─── batch_build_profiles ────────────────────────────────────────────────────

def test_batch_build_profiles_count():
    point_sets = [np.random.rand(10, 2) for _ in range(3)]
    profiles = batch_build_profiles(point_sets)
    assert len(profiles) == 3


def test_batch_build_profiles_with_sides():
    point_sets = [np.random.rand(10, 2) for _ in range(2)]
    profiles = batch_build_profiles(point_sets, sides=["top", "bottom"])
    assert profiles[0].side == "top"
    assert profiles[1].side == "bottom"


# ─── pairwise_l2_matrix ──────────────────────────────────────────────────────

def test_pairwise_l2_matrix_shape():
    profiles = [EdgeProfile(values=np.random.rand(8).astype(np.float32)) for _ in range(4)]
    mat = pairwise_l2_matrix(profiles)
    assert mat.shape == (4, 4)


def test_pairwise_l2_matrix_diagonal_zero():
    profiles = [EdgeProfile(values=np.array([0.0, 0.5, 1.0])) for _ in range(3)]
    mat = pairwise_l2_matrix(profiles)
    assert np.allclose(mat.diagonal(), 0.0)


def test_pairwise_l2_matrix_symmetric():
    profiles = [EdgeProfile(values=np.random.rand(5).astype(np.float32)) for _ in range(3)]
    mat = pairwise_l2_matrix(profiles)
    assert np.allclose(mat, mat.T)


# ─── best_matching_profile ───────────────────────────────────────────────────

def test_best_matching_profile_finds_identical():
    query = EdgeProfile(values=np.array([1.0, 2.0, 3.0]))
    candidates = [
        EdgeProfile(values=np.array([3.0, 1.0, 2.0])),
        EdgeProfile(values=np.array([1.0, 2.0, 3.0])),
    ]
    idx, dist = best_matching_profile(query, candidates)
    assert idx == 1
    assert dist == pytest.approx(0.0)


def test_best_matching_profile_empty_raises():
    query = EdgeProfile(values=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        best_matching_profile(query, [])
