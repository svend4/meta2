"""Extra tests for puzzle_reconstruction/algorithms/descriptor_aggregator.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.descriptor_aggregator import (
    AggregatorConfig,
    AggregatedDescriptor,
    l2_normalize,
    concatenate_descriptors,
    weighted_average_descriptors,
    pca_reduce,
    elementwise_aggregate,
    aggregate,
    distance_matrix,
    batch_aggregate,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _vec(dim=16, seed=0):
    return np.random.default_rng(seed).random(dim).astype(np.float32)


def _descs(n=2, dim=16):
    return {f"s{i}": _vec(dim, seed=i) for i in range(n)}


# ─── TestAggregatorConfigExtra ────────────────────────────────────────────────

class TestAggregatorConfigExtra:
    def test_pca_mode_valid(self):
        cfg = AggregatorConfig(mode="pca")
        assert cfg.mode == "pca"

    def test_positive_weight_valid(self):
        cfg = AggregatorConfig(weights={"a": 2.5})
        assert cfg.weights["a"] == pytest.approx(2.5)

    def test_large_n_components_valid(self):
        cfg = AggregatorConfig(n_components=512)
        assert cfg.n_components == 512

    def test_normalize_false(self):
        cfg = AggregatorConfig(normalize=False)
        assert cfg.normalize is False

    def test_multiple_weights(self):
        cfg = AggregatorConfig(weights={"a": 1.0, "b": 2.0, "c": 0.5})
        assert len(cfg.weights) == 3


# ─── TestAggregatedDescriptorExtra ───────────────────────────────────────────

class TestAggregatedDescriptorExtra:
    def test_float64_converted_to_float32(self):
        v = _vec(8).astype(np.float64)
        ad = AggregatedDescriptor(fragment_id=0, vector=v, mode="max")
        assert ad.vector.dtype == np.float32

    def test_source_dims_default_empty(self):
        ad = AggregatedDescriptor(fragment_id=1, vector=_vec(4), mode="min")
        assert ad.source_dims == {}

    def test_zero_fragment_id_valid(self):
        ad = AggregatedDescriptor(fragment_id=0, vector=_vec(4), mode="concat")
        assert ad.fragment_id == 0

    def test_large_fragment_id(self):
        ad = AggregatedDescriptor(fragment_id=9999, vector=_vec(4), mode="concat")
        assert ad.fragment_id == 9999

    def test_dim_reflects_vector_length(self):
        v = _vec(32)
        ad = AggregatedDescriptor(fragment_id=0, vector=v, mode="weighted_avg")
        assert ad.dim == 32


# ─── TestL2NormalizeExtra ─────────────────────────────────────────────────────

class TestL2NormalizeExtra:
    def test_already_unit_norm_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n = l2_normalize(v)
        np.testing.assert_allclose(n, v, atol=1e-7)

    def test_2d_single_row_unit_norm(self):
        M = np.array([[3.0, 4.0]], dtype=np.float32)
        N = l2_normalize(M)
        assert np.linalg.norm(N[0]) == pytest.approx(1.0, abs=1e-6)

    def test_2d_zero_row_unchanged(self):
        M = np.zeros((3, 8), dtype=np.float32)
        N = l2_normalize(M)
        assert np.all(N == 0.0)

    def test_negative_values_handled(self):
        v = np.array([-3.0, -4.0], dtype=np.float32)
        n = l2_normalize(v)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-6)


# ─── TestConcatenateDescriptorsExtra ─────────────────────────────────────────

class TestConcatenateDescriptorsExtra:
    def test_three_sources_total_length(self):
        d = {"a": _vec(4), "b": _vec(6), "c": _vec(8)}
        v = concatenate_descriptors(d, normalize=False)
        assert v.shape[0] == 18

    def test_single_source_normalized(self):
        d = {"only": _vec(8)}
        v = concatenate_descriptors(d, normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_values_are_float32(self):
        d = _descs(3, 8)
        v = concatenate_descriptors(d, normalize=False)
        assert v.dtype == np.float32

    def test_no_normalize_values_match(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d = {"x": vec}
        v = concatenate_descriptors(d, normalize=False)
        np.testing.assert_allclose(v, vec, atol=1e-7)


# ─── TestWeightedAverageDescriptorsExtra ─────────────────────────────────────

class TestWeightedAverageDescriptorsExtra:
    def test_custom_weights_dominant(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        # Weight a=10, b=0 → result ~ a
        result = weighted_average_descriptors(
            {"a": a, "b": b}, weights={"a": 10.0, "b": 0.0}, normalize=False
        )
        np.testing.assert_allclose(result, a, atol=1e-5)

    def test_three_sources_same_dim(self):
        d = {"x": _vec(8, 0), "y": _vec(8, 1), "z": _vec(8, 2)}
        v = weighted_average_descriptors(d, normalize=False)
        assert v.shape[0] == 8

    def test_normalized_result_unit_norm(self):
        d = _descs(3, 16)
        v = weighted_average_descriptors(d, normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


# ─── TestPcaReduceExtra ───────────────────────────────────────────────────────

class TestPcaReduceExtra:
    def test_more_components_than_features_clamped(self):
        X = np.random.default_rng(0).random((5, 8))
        Y = pca_reduce(X, n_components=100)
        assert Y.shape[1] <= min(5, 8)

    def test_output_float32(self):
        X = np.random.default_rng(1).random((4, 16)).astype(np.float64)
        Y = pca_reduce(X, n_components=4)
        assert Y.dtype == np.float32

    def test_components_1(self):
        X = np.random.default_rng(2).random((5, 10))
        Y = pca_reduce(X, n_components=1)
        assert Y.shape == (5, 1)

    def test_square_matrix(self):
        X = np.random.default_rng(3).random((8, 8))
        Y = pca_reduce(X, n_components=4)
        assert Y.shape == (8, 4)


# ─── TestElementwiseAggregateExtra ───────────────────────────────────────────

class TestElementwiseAggregateExtra:
    def test_single_source_max(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = elementwise_aggregate({"a": v}, mode="max", normalize=False)
        np.testing.assert_allclose(result, v, atol=1e-7)

    def test_single_source_min(self):
        v = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = elementwise_aggregate({"a": v}, mode="min", normalize=False)
        np.testing.assert_allclose(result, v, atol=1e-7)

    def test_normalized_max_unit_norm(self):
        d = _descs(3, 8)
        v = elementwise_aggregate(d, mode="max", normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_three_sources_max(self):
        a = np.array([1.0, 5.0], dtype=np.float32)
        b = np.array([3.0, 2.0], dtype=np.float32)
        c = np.array([2.0, 4.0], dtype=np.float32)
        result = elementwise_aggregate({"a": a, "b": b, "c": c}, mode="max", normalize=False)
        np.testing.assert_allclose(result, [3.0, 5.0], atol=1e-7)


# ─── TestAggregateExtra ───────────────────────────────────────────────────────

class TestAggregateExtra:
    def test_pca_mode_falls_back_to_concat(self):
        d = _descs(2, 8)
        cfg = AggregatorConfig(mode="pca")
        v = aggregate(d, cfg)
        assert isinstance(v, np.ndarray)
        assert v.ndim == 1

    def test_normalize_false_concat(self):
        d = {"a": np.ones(4, dtype=np.float32)}
        cfg = AggregatorConfig(mode="concat", normalize=False)
        v = aggregate(d, cfg)
        np.testing.assert_allclose(v, np.ones(4), atol=1e-7)

    def test_min_mode_shape(self):
        d = _descs(2, 8)
        cfg = AggregatorConfig(mode="min")
        v = aggregate(d, cfg)
        assert v.shape[0] == 8

    def test_returns_float32(self):
        d = _descs(2, 8)
        v = aggregate(d, AggregatorConfig(mode="max"))
        assert v.dtype == np.float32


# ─── TestDistanceMatrixExtra ──────────────────────────────────────────────────

class TestDistanceMatrixExtra:
    def test_euclidean_symmetric(self):
        V = np.random.default_rng(5).random((4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="euclidean")
        assert np.allclose(D, D.T, atol=1e-5)

    def test_l1_diagonal_zero(self):
        V = np.random.default_rng(6).random((3, 6)).astype(np.float32)
        D = distance_matrix(V, metric="l1")
        assert np.allclose(np.diag(D), 0.0)

    def test_cosine_range_0_2(self):
        V = np.random.default_rng(7).random((5, 8)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert np.all(D >= -1e-6)
        assert np.all(D <= 2.0 + 1e-6)

    def test_two_vectors(self):
        V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        D = distance_matrix(V, metric="euclidean")
        assert D.shape == (2, 2)
        assert D[0, 1] == pytest.approx(np.sqrt(2.0), abs=1e-5)


# ─── TestBatchAggregateExtra ──────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def test_pca_config_runs(self):
        cfg = AggregatorConfig(mode="pca")
        groups = [_descs(2, 8)]
        result = batch_aggregate(groups, cfg)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_five_groups(self):
        groups = [_descs(2, 8) for _ in range(5)]
        result = batch_aggregate(groups)
        assert len(result) == 5

    def test_all_vectors_float32(self):
        groups = [_descs(2, 16) for _ in range(3)]
        result = batch_aggregate(groups)
        assert all(v.dtype == np.float32 for v in result)

    def test_min_config(self):
        cfg = AggregatorConfig(mode="min")
        groups = [_descs(3, 8)]
        result = batch_aggregate(groups, cfg)
        assert result[0].shape[0] == 8
