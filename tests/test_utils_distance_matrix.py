"""Тесты для puzzle_reconstruction/utils/distance_matrix.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.distance_matrix import (
    DistanceConfig,
    euclidean_distance_matrix,
    cosine_distance_matrix,
    manhattan_distance_matrix,
    build_distance_matrix,
    normalize_distance_matrix,
    to_similarity_matrix,
    threshold_distance_matrix,
    top_k_distance_pairs,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_X(n=4, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


# ─── DistanceConfig ───────────────────────────────────────────────────────────

class TestDistanceConfig:
    def test_defaults(self):
        cfg = DistanceConfig()
        assert cfg.metric == "euclidean"
        assert cfg.normalize is True
        assert cfg.eps == pytest.approx(1e-8)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            DistanceConfig(metric="chebyshev")

    def test_valid_metrics(self):
        for m in ("euclidean", "cosine", "manhattan"):
            cfg = DistanceConfig(metric=m)
            assert cfg.metric == m

    def test_eps_zero_raises(self):
        with pytest.raises(ValueError, match="eps"):
            DistanceConfig(eps=0.0)

    def test_eps_negative_raises(self):
        with pytest.raises(ValueError, match="eps"):
            DistanceConfig(eps=-1e-9)


# ─── euclidean_distance_matrix ────────────────────────────────────────────────

class TestEuclideanDistanceMatrix:
    def test_returns_ndarray(self):
        result = euclidean_distance_matrix(make_X())
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        X = make_X(n=4)
        result = euclidean_distance_matrix(X)
        assert result.shape == (4, 4)

    def test_diagonal_zero(self):
        result = euclidean_distance_matrix(make_X(n=4))
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_symmetric(self):
        result = euclidean_distance_matrix(make_X(n=4))
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_nonnegative(self):
        result = euclidean_distance_matrix(make_X(n=4))
        assert (result >= 0).all()

    def test_known_value(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        result = euclidean_distance_matrix(X)
        assert result[0, 1] == pytest.approx(5.0)
        assert result[1, 0] == pytest.approx(5.0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            euclidean_distance_matrix(np.zeros(4))

    def test_single_row(self):
        X = np.array([[1.0, 2.0, 3.0]])
        result = euclidean_distance_matrix(X)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_dtype_float64(self):
        result = euclidean_distance_matrix(make_X(n=3))
        assert result.dtype == np.float64


# ─── cosine_distance_matrix ───────────────────────────────────────────────────

class TestCosineDistanceMatrix:
    def test_returns_ndarray(self):
        result = cosine_distance_matrix(make_X())
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        X = make_X(n=5)
        result = cosine_distance_matrix(X)
        assert result.shape == (5, 5)

    def test_diagonal_zero(self):
        result = cosine_distance_matrix(make_X(n=4))
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)

    def test_symmetric(self):
        result = cosine_distance_matrix(make_X(n=4))
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_range_0_2(self):
        result = cosine_distance_matrix(make_X(n=4))
        assert (result >= 0).all()
        assert (result <= 2.0 + 1e-6).all()

    def test_identical_vectors_zero_distance(self):
        X = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = cosine_distance_matrix(X)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_max_distance(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0]])
        result = cosine_distance_matrix(X)
        assert result[0, 1] == pytest.approx(2.0, abs=1e-6)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            cosine_distance_matrix(np.zeros(4))


# ─── manhattan_distance_matrix ────────────────────────────────────────────────

class TestManhattanDistanceMatrix:
    def test_shape(self):
        X = make_X(n=3)
        result = manhattan_distance_matrix(X)
        assert result.shape == (3, 3)

    def test_diagonal_zero(self):
        result = manhattan_distance_matrix(make_X(n=3))
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_symmetric(self):
        result = manhattan_distance_matrix(make_X(n=4))
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_known_value(self):
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        result = manhattan_distance_matrix(X)
        assert result[0, 1] == pytest.approx(3.0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            manhattan_distance_matrix(np.zeros(4))


# ─── build_distance_matrix ────────────────────────────────────────────────────

class TestBuildDistanceMatrix:
    def test_returns_ndarray(self):
        result = build_distance_matrix(make_X(n=3))
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        X = make_X(n=5)
        result = build_distance_matrix(X)
        assert result.shape == (5, 5)

    def test_euclidean_by_default(self):
        X = make_X(n=4)
        cfg = DistanceConfig(metric="euclidean", normalize=False)
        result = build_distance_matrix(X, cfg=cfg)
        expected = euclidean_distance_matrix(X)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_normalize_in_0_1(self):
        X = make_X(n=5)
        cfg = DistanceConfig(normalize=True)
        result = build_distance_matrix(X, cfg=cfg)
        assert result.max() <= 1.0 + 1e-9
        assert result.min() >= 0.0

    def test_none_cfg_uses_defaults(self):
        result = build_distance_matrix(make_X(n=3), cfg=None)
        assert isinstance(result, np.ndarray)

    def test_cosine_metric(self):
        X = make_X(n=3)
        cfg = DistanceConfig(metric="cosine", normalize=False)
        result = build_distance_matrix(X, cfg=cfg)
        expected = cosine_distance_matrix(X)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_manhattan_metric(self):
        X = make_X(n=3)
        cfg = DistanceConfig(metric="manhattan", normalize=False)
        result = build_distance_matrix(X, cfg=cfg)
        expected = manhattan_distance_matrix(X)
        np.testing.assert_allclose(result, expected, atol=1e-9)


# ─── normalize_distance_matrix ────────────────────────────────────────────────

class TestNormalizeDistanceMatrix:
    def test_max_leq_1(self):
        mat = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        assert result.max() <= 1.0 + 1e-9

    def test_diagonal_zero(self):
        mat = np.array([[0, 3], [3, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_already_normalized_unchanged(self):
        # max off-diagonal = 1.0 → dividing by 1.0 leaves unchanged
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        np.testing.assert_allclose(result, mat, atol=1e-9)

    def test_not_square_raises(self):
        with pytest.raises(ValueError):
            normalize_distance_matrix(np.zeros((3, 4)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            normalize_distance_matrix(np.zeros(4))

    def test_all_zeros_stays_zero(self):
        mat = np.zeros((3, 3), dtype=np.float64)
        result = normalize_distance_matrix(mat)
        assert result.max() == pytest.approx(0.0)

    def test_preserves_relative_order(self):
        mat = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        assert result[0, 1] < result[0, 2] < result[1, 2]


# ─── to_similarity_matrix ─────────────────────────────────────────────────────

class TestToSimilarityMatrix:
    def test_diagonal_is_1(self):
        mat = np.array([[0, 1], [1, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_inverse_method(self):
        mat = np.array([[0, 1], [1, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat, method="inverse")
        # distance=1 → similarity = 1/(1+1) = 0.5
        assert result[0, 1] == pytest.approx(0.5)

    def test_gaussian_method(self):
        mat = np.array([[0, 1], [1, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat, method="gaussian", sigma=1.0)
        assert result[0, 1] == pytest.approx(np.exp(-0.5))

    def test_zero_distance_inverse_is_1(self):
        mat = np.zeros((2, 2), dtype=np.float64)
        result = to_similarity_matrix(mat, method="inverse")
        assert result[0, 1] == pytest.approx(1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            to_similarity_matrix(np.zeros((2, 2)), method="cosine")

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            to_similarity_matrix(np.zeros((2, 2)), method="gaussian", sigma=0.0)

    def test_values_in_0_1(self):
        mat = np.array([[0, 0.5, 1], [0.5, 0, 2], [1, 2, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat)
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-9).all()


# ─── threshold_distance_matrix ────────────────────────────────────────────────

class TestThresholdDistanceMatrix:
    def test_values_above_threshold_replaced(self):
        mat = np.array([[0, 0.3, 0.7], [0.3, 0, 0.9], [0.7, 0.9, 0]],
                       dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        assert result[0, 2] == pytest.approx(0.0)
        assert result[1, 2] == pytest.approx(0.0)

    def test_values_below_threshold_unchanged(self):
        mat = np.array([[0, 0.3], [0.3, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        assert result[0, 1] == pytest.approx(0.3)

    def test_custom_fill_value(self):
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5, fill=-1.0)
        assert result[0, 1] == pytest.approx(-1.0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            threshold_distance_matrix(np.zeros(4), threshold=0.5)

    def test_does_not_modify_original(self):
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        original = mat.copy()
        threshold_distance_matrix(mat, threshold=0.5)
        np.testing.assert_array_equal(mat, original)

    def test_zero_threshold_replaces_all_nonzero(self):
        mat = np.array([[0, 0.5], [0.5, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.0)
        assert result[0, 1] == pytest.approx(0.0)


# ─── top_k_distance_pairs ─────────────────────────────────────────────────────

class TestTopKDistancePairs:
    def test_returns_list(self):
        mat = euclidean_distance_matrix(make_X(n=4))
        result = top_k_distance_pairs(mat, k=3)
        assert isinstance(result, list)

    def test_length_k(self):
        mat = euclidean_distance_matrix(make_X(n=4))
        result = top_k_distance_pairs(mat, k=3)
        assert len(result) == 3

    def test_pairs_i_lt_j(self):
        mat = euclidean_distance_matrix(make_X(n=4))
        result = top_k_distance_pairs(mat, k=3)
        for i, j, _ in result:
            assert i < j

    def test_sorted_ascending(self):
        mat = euclidean_distance_matrix(make_X(n=4))
        result = top_k_distance_pairs(mat, k=4)
        dists = [d for _, _, d in result]
        assert dists == sorted(dists)

    def test_k_zero_raises(self):
        mat = euclidean_distance_matrix(make_X(n=3))
        with pytest.raises(ValueError):
            top_k_distance_pairs(mat, k=0)

    def test_not_square_raises(self):
        with pytest.raises(ValueError):
            top_k_distance_pairs(np.zeros((3, 4)), k=1)

    def test_k_exceeds_pairs_capped(self):
        mat = euclidean_distance_matrix(make_X(n=3))
        # 3x3 matrix → 3 unique pairs
        result = top_k_distance_pairs(mat, k=100)
        assert len(result) == 3

    def test_tuple_format(self):
        mat = euclidean_distance_matrix(make_X(n=3))
        result = top_k_distance_pairs(mat, k=1)
        i, j, d = result[0]
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(d, float)
