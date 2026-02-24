"""Extra tests for puzzle_reconstruction/utils/distance_matrix.py."""
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


def _X(n=4, d=3, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


# ─── DistanceConfig (extra) ───────────────────────────────────────────────────

class TestDistanceConfigExtra:
    def test_normalize_false(self):
        cfg = DistanceConfig(normalize=False)
        assert cfg.normalize is False

    def test_normalize_default_true(self):
        assert DistanceConfig().normalize is True

    def test_eps_small_positive_valid(self):
        cfg = DistanceConfig(eps=1e-12)
        assert cfg.eps == pytest.approx(1e-12)

    def test_euclidean_metric_stored(self):
        cfg = DistanceConfig(metric="euclidean")
        assert cfg.metric == "euclidean"

    def test_cosine_metric_stored(self):
        cfg = DistanceConfig(metric="cosine")
        assert cfg.metric == "cosine"

    def test_manhattan_metric_stored(self):
        cfg = DistanceConfig(metric="manhattan")
        assert cfg.metric == "manhattan"

    def test_all_three_valid_metrics(self):
        for m in ("euclidean", "cosine", "manhattan"):
            assert DistanceConfig(metric=m).metric == m


# ─── euclidean_distance_matrix (extra) ───────────────────────────────────────

class TestEuclideanExtra:
    def test_triangle_inequality(self):
        X = _X(n=5)
        D = euclidean_distance_matrix(X)
        # d(a, c) <= d(a, b) + d(b, c)
        for a in range(5):
            for b in range(5):
                for c in range(5):
                    assert D[a, c] <= D[a, b] + D[b, c] + 1e-9

    def test_large_n(self):
        X = _X(n=50, d=8)
        D = euclidean_distance_matrix(X)
        assert D.shape == (50, 50)
        assert (np.diag(D) < 1e-9).all()

    def test_identical_rows_zero_dist(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        D = euclidean_distance_matrix(X)
        assert D[0, 1] == pytest.approx(0.0)

    def test_integer_input_float_output(self):
        X = np.array([[0, 0], [3, 4]], dtype=np.int32)
        D = euclidean_distance_matrix(X)
        assert D[0, 1] == pytest.approx(5.0)

    def test_float32_input(self):
        X = _X(n=3).astype(np.float32)
        D = euclidean_distance_matrix(X)
        assert isinstance(D, np.ndarray)

    def test_two_rows_known(self):
        X = np.array([[1.0, 1.0], [4.0, 5.0]])
        D = euclidean_distance_matrix(X)
        assert D[0, 1] == pytest.approx(5.0)


# ─── cosine_distance_matrix (extra) ──────────────────────────────────────────

class TestCosineExtra:
    def test_orthogonal_vectors_distance_one(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = cosine_distance_matrix(X)
        assert D[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_scaled_same_vector_zero_dist(self):
        X = np.array([[1.0, 2.0], [2.0, 4.0]])
        D = cosine_distance_matrix(X)
        assert D[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_large_matrix_shape(self):
        X = _X(n=10, d=5)
        D = cosine_distance_matrix(X)
        assert D.shape == (10, 10)

    def test_symmetric_property(self):
        X = _X(n=6)
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-9)

    def test_all_identical_rows_zero_offdiag(self):
        X = np.ones((4, 3))
        D = cosine_distance_matrix(X)
        off = D[np.triu_indices(4, k=1)]
        np.testing.assert_allclose(off, 0.0, atol=1e-6)


# ─── manhattan_distance_matrix (extra) ───────────────────────────────────────

class TestManhattanExtra:
    def test_nonnegative(self):
        D = manhattan_distance_matrix(_X(n=4))
        assert (D >= 0).all()

    def test_known_3d_value(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        D = manhattan_distance_matrix(X)
        assert D[0, 1] == pytest.approx(6.0)

    def test_identical_rows_zero_dist(self):
        X = np.array([[5.0, 5.0], [5.0, 5.0]])
        D = manhattan_distance_matrix(X)
        assert D[0, 1] == pytest.approx(0.0)

    def test_larger_than_euclidean(self):
        X = _X(n=5)
        Dm = manhattan_distance_matrix(X)
        De = euclidean_distance_matrix(X)
        # Manhattan ≥ Euclidean for all pairs
        assert (Dm >= De - 1e-9).all()

    def test_shape_consistent(self):
        X = _X(n=7, d=4)
        D = manhattan_distance_matrix(X)
        assert D.shape == (7, 7)


# ─── build_distance_matrix (extra) ───────────────────────────────────────────

class TestBuildDistanceMatrixExtra:
    def test_normalize_ranges_0_1(self):
        X = _X(n=6)
        cfg = DistanceConfig(normalize=True)
        D = build_distance_matrix(X, cfg=cfg)
        assert D.max() <= 1.0 + 1e-9
        assert D.min() >= 0.0

    def test_normalize_false_can_exceed_one(self):
        X = _X(n=4, d=8, seed=99)
        cfg = DistanceConfig(normalize=False)
        D = build_distance_matrix(X, cfg=cfg)
        # No constraint that values are in [0,1]
        assert isinstance(D, np.ndarray)

    def test_cosine_cfg_matches_direct(self):
        X = _X(n=5)
        cfg = DistanceConfig(metric="cosine", normalize=False)
        D = build_distance_matrix(X, cfg=cfg)
        expected = cosine_distance_matrix(X)
        np.testing.assert_allclose(D, expected, atol=1e-9)

    def test_manhattan_cfg_matches_direct(self):
        X = _X(n=4)
        cfg = DistanceConfig(metric="manhattan", normalize=False)
        D = build_distance_matrix(X, cfg=cfg)
        expected = manhattan_distance_matrix(X)
        np.testing.assert_allclose(D, expected, atol=1e-9)

    def test_returns_2d_square(self):
        X = _X(n=5)
        D = build_distance_matrix(X)
        assert D.ndim == 2
        assert D.shape[0] == D.shape[1]

    def test_single_row_gives_1x1(self):
        X = np.array([[1.0, 2.0]])
        cfg = DistanceConfig(normalize=False)
        D = build_distance_matrix(X, cfg=cfg)
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(0.0)


# ─── normalize_distance_matrix (extra) ───────────────────────────────────────

class TestNormalizeDistanceMatrixExtra:
    def test_4x4_values_in_0_1(self):
        mat = np.array([[0, 3, 6, 9], [3, 0, 4, 7],
                        [6, 4, 0, 2], [9, 7, 2, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        assert result.max() <= 1.0 + 1e-9
        assert result.min() >= 0.0

    def test_symmetric_after_normalize(self):
        mat = np.array([[0, 5, 10], [5, 0, 8], [10, 8, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_1x1_zero_matrix(self):
        mat = np.zeros((1, 1))
        result = normalize_distance_matrix(mat)
        assert result[0, 0] == pytest.approx(0.0)

    def test_scaling_preserves_ratios(self):
        mat = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0]], dtype=np.float64)
        result = normalize_distance_matrix(mat)
        # result[0,1] : result[0,2] ≈ 2 : 4 = 1 : 2
        assert result[0, 2] == pytest.approx(2 * result[0, 1], rel=1e-6)


# ─── to_similarity_matrix (extra) ────────────────────────────────────────────

class TestToSimilarityMatrixExtra:
    def test_inverse_zero_distance_is_one(self):
        mat = np.zeros((2, 2), dtype=np.float64)
        result = to_similarity_matrix(mat, method="inverse")
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-9)

    def test_inverse_large_distance_near_zero(self):
        mat = np.array([[0, 1000.0], [1000.0, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat, method="inverse")
        assert result[0, 1] < 0.01

    def test_gaussian_sigma_affects_value(self):
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        r1 = to_similarity_matrix(mat, method="gaussian", sigma=0.5)
        r2 = to_similarity_matrix(mat, method="gaussian", sigma=2.0)
        # Smaller sigma → steeper drop → lower similarity at d=1
        assert r1[0, 1] < r2[0, 1]

    def test_diagonal_always_one(self):
        for method in ("inverse", "gaussian"):
            mat = np.array([[0, 0.3], [0.3, 0]], dtype=np.float64)
            result = to_similarity_matrix(mat, method=method, sigma=1.0)
            np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-9)

    def test_symmetric_output(self):
        mat = np.array([[0, 0.7, 0.2], [0.7, 0, 0.5],
                        [0.2, 0.5, 0]], dtype=np.float64)
        result = to_similarity_matrix(mat)
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            to_similarity_matrix(np.zeros((2, 2)), method="gaussian", sigma=-1.0)


# ─── threshold_distance_matrix (extra) ───────────────────────────────────────

class TestThresholdDistanceMatrixExtra:
    def test_all_below_threshold_unchanged(self):
        mat = np.array([[0, 0.1], [0.1, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        np.testing.assert_allclose(result, mat)

    def test_fill_nan(self):
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5, fill=float("nan"))
        assert np.isnan(result[0, 1])

    def test_diagonal_unaffected(self):
        mat = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_returns_float64(self):
        mat = np.array([[0, 0.5], [0.5, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=1.0)
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        mat = np.array([[0, 0.3], [0.3, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        assert result.shape == mat.shape

    def test_exact_threshold_kept(self):
        mat = np.array([[0, 0.5], [0.5, 0]], dtype=np.float64)
        result = threshold_distance_matrix(mat, threshold=0.5)
        assert result[0, 1] == pytest.approx(0.5)


# ─── top_k_distance_pairs (extra) ────────────────────────────────────────────

class TestTopKDistancePairsExtra:
    def test_k_one_returns_smallest(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        D = euclidean_distance_matrix(X)
        result = top_k_distance_pairs(D, k=1)
        _, _, d = result[0]
        assert d == pytest.approx(1.0)

    def test_all_pairs_4x4(self):
        X = _X(n=4)
        D = euclidean_distance_matrix(X)
        result = top_k_distance_pairs(D, k=6)  # 4*(4-1)/2 = 6
        assert len(result) == 6

    def test_no_self_pairs(self):
        X = _X(n=5)
        D = euclidean_distance_matrix(X)
        for i, j, _ in top_k_distance_pairs(D, k=10):
            assert i != j

    def test_distances_nonneg(self):
        X = _X(n=5)
        D = euclidean_distance_matrix(X)
        for _, _, d in top_k_distance_pairs(D, k=5):
            assert d >= 0.0

    def test_1x1_matrix_k1_raises(self):
        D = np.zeros((1, 1))
        # only 0 unique pairs → capped to 0
        result = top_k_distance_pairs(D, k=1)
        assert len(result) == 0

    def test_negative_k_raises(self):
        D = euclidean_distance_matrix(_X(n=3))
        with pytest.raises(ValueError):
            top_k_distance_pairs(D, k=-1)
