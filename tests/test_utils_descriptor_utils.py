"""Tests for puzzle_reconstruction.utils.descriptor_utils."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.descriptor_utils import (
    DescriptorConfig,
    DescriptorMatch,
    l2_normalize,
    l1_normalize,
    batch_l2_normalize,
    l2_distance,
    cosine_distance,
    chi2_distance,
    l1_distance,
    descriptor_distance,
    pairwise_l2,
    pairwise_cosine,
    nn_match,
    ratio_test,
    mean_pool,
    max_pool,
    vlad_encode,
    batch_nn_match,
    top_k_matches,
    filter_matches_by_distance,
)

np.random.seed(55)


# ── l2_normalize ──────────────────────────────────────────────────────────────

def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0])
    n = l2_normalize(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9


def test_l2_normalize_zero_vector():
    v = np.zeros(4)
    n = l2_normalize(v)
    assert np.allclose(n, 0.0)


def test_l2_normalize_returns_copy():
    v = np.array([1.0, 0.0])
    n = l2_normalize(v)
    n[0] = 99.0
    assert v[0] == 1.0


# ── l1_normalize ──────────────────────────────────────────────────────────────

def test_l1_normalize_unit_l1():
    v = np.array([1.0, 2.0, 3.0])
    n = l1_normalize(v)
    assert abs(np.abs(n).sum() - 1.0) < 1e-9


def test_l1_normalize_zero():
    v = np.zeros(3)
    n = l1_normalize(v)
    assert np.allclose(n, 0.0)


# ── batch_l2_normalize ────────────────────────────────────────────────────────

def test_batch_l2_normalize_shape():
    mat = np.random.randn(5, 8)
    result = batch_l2_normalize(mat)
    assert result.shape == (5, 8)


def test_batch_l2_normalize_unit_rows():
    mat = np.random.randn(10, 4)
    result = batch_l2_normalize(mat)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


# ── distance metrics ──────────────────────────────────────────────────────────

def test_l2_distance_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert l2_distance(v, v) == pytest.approx(0.0)


def test_l2_distance_known():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert l2_distance(a, b) == pytest.approx(5.0)


def test_cosine_distance_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-7)


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_distance(a, b) == pytest.approx(0.5, abs=1e-6)


def test_cosine_distance_zero_vector():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(a, b) == 1.0


def test_chi2_distance_identical():
    v = np.array([0.25, 0.25, 0.25, 0.25])
    assert chi2_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_l1_distance_known():
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    assert l1_distance(a, b) == pytest.approx(7.0)


def test_descriptor_distance_invalid_metric():
    v = np.ones(4)
    with pytest.raises(ValueError):
        descriptor_distance(v, v, metric="xyz")


def test_descriptor_distance_all_metrics():
    a = np.random.rand(8)
    b = np.random.rand(8)
    for metric in ("l2", "cosine", "chi2", "l1"):
        d = descriptor_distance(a, b, metric=metric)
        assert d >= 0.0


# ── pairwise matrices ─────────────────────────────────────────────────────────

def test_pairwise_l2_shape():
    A = np.random.randn(4, 6)
    B = np.random.randn(7, 6)
    D = pairwise_l2(A, B)
    assert D.shape == (4, 7)


def test_pairwise_l2_diagonal_zero():
    A = np.random.randn(5, 4)
    D = pairwise_l2(A, A)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)


def test_pairwise_cosine_shape():
    A = np.random.randn(3, 5)
    B = np.random.randn(4, 5)
    D = pairwise_cosine(A, B)
    assert D.shape == (3, 4)


def test_pairwise_cosine_in_01():
    A = np.random.randn(4, 6)
    D = pairwise_cosine(A, A)
    assert D.min() >= -1e-6
    assert D.max() <= 1.0 + 1e-6


# ── nn_match ──────────────────────────────────────────────────────────────────

def test_nn_match_count():
    q = np.random.randn(5, 8)
    t = np.random.randn(10, 8)
    matches = nn_match(q, t)
    assert len(matches) == 5


def test_nn_match_valid_indices():
    q = np.random.randn(3, 4)
    t = np.random.randn(6, 4)
    for m in nn_match(q, t):
        assert 0 <= m.query_idx < 3
        assert 0 <= m.train_idx < 6


def test_nn_match_empty_query():
    q = np.empty((0, 4))
    t = np.random.randn(5, 4)
    assert nn_match(q, t) == []


def test_nn_match_returns_match_objects():
    q = np.random.randn(2, 4)
    t = np.random.randn(4, 4)
    for m in nn_match(q, t):
        assert isinstance(m, DescriptorMatch)


# ── ratio_test ────────────────────────────────────────────────────────────────

def test_ratio_test_fewer_matches():
    q = np.random.randn(10, 8)
    t = np.random.randn(20, 8)
    all_matches = nn_match(q, t)
    ratio_matches = ratio_test(q, t, ratio=0.75)
    assert len(ratio_matches) <= len(all_matches)


def test_ratio_test_invalid_ratio():
    q = np.random.randn(3, 4)
    t = np.random.randn(5, 4)
    with pytest.raises(ValueError):
        ratio_test(q, t, ratio=1.5)


def test_ratio_test_empty_train():
    q = np.random.randn(3, 4)
    t = np.random.randn(1, 4)  # < 2, cannot do ratio test
    assert ratio_test(q, t) == []


# ── pooling ───────────────────────────────────────────────────────────────────

def test_mean_pool_shape():
    descs = np.random.randn(8, 12)
    pooled = mean_pool(descs)
    assert pooled.shape == (12,)


def test_max_pool_shape():
    descs = np.random.randn(8, 12)
    pooled = max_pool(descs)
    assert pooled.shape == (12,)


def test_mean_pool_empty_raises():
    with pytest.raises(ValueError):
        mean_pool(np.empty((0, 4)))


def test_max_pool_empty_raises():
    with pytest.raises(ValueError):
        max_pool(np.empty((0, 4)))


def test_vlad_encode_shape():
    descs = np.random.randn(10, 8).astype(np.float32)
    codebook = np.random.randn(4, 8).astype(np.float32)
    vlad = vlad_encode(descs, codebook)
    assert vlad.shape == (32,)  # 4 * 8


def test_vlad_encode_empty_descs():
    descs = np.empty((0, 8), dtype=np.float32)
    codebook = np.random.randn(4, 8).astype(np.float32)
    vlad = vlad_encode(descs, codebook)
    assert np.allclose(vlad, 0.0)


# ── helpers ───────────────────────────────────────────────────────────────────

def test_top_k_matches():
    matches = [DescriptorMatch(i, i, float(i)) for i in range(5)]
    top2 = top_k_matches(matches, 2)
    assert len(top2) == 2
    assert top2[0].distance <= top2[1].distance


def test_filter_matches_by_distance():
    matches = [DescriptorMatch(i, i, float(i)) for i in range(5)]
    filtered = filter_matches_by_distance(matches, max_distance=2.0)
    assert all(m.distance <= 2.0 for m in filtered)


def test_batch_nn_match_length():
    sets = [np.random.randn(3, 4) for _ in range(4)]
    train = np.random.randn(6, 4)
    results = batch_nn_match(sets, train)
    assert len(results) == 4
