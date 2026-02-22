"""Тесты для puzzle_reconstruction.algorithms.descriptor_aggregator."""
import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _vec(dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


def _descs(n_sources=2, dim=16):
    return {f"src{i}": _vec(dim, seed=i) for i in range(n_sources)}


# ─── TestAggregatorConfig ─────────────────────────────────────────────────────

class TestAggregatorConfig:
    def test_defaults(self):
        cfg = AggregatorConfig()
        assert cfg.mode == "concat"
        assert cfg.n_components == 32
        assert cfg.normalize is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(mode="unknown")

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(weights={"a": -0.1})

    def test_n_components_zero_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(n_components=0)

    def test_valid_all_modes(self):
        for m in ("concat", "weighted_avg", "pca", "max", "min"):
            cfg = AggregatorConfig(mode=m)
            assert cfg.mode == m

    def test_zero_weight_valid(self):
        cfg = AggregatorConfig(weights={"a": 0.0})
        assert cfg.weights["a"] == pytest.approx(0.0)


# ─── TestAggregatedDescriptor ─────────────────────────────────────────────────

class TestAggregatedDescriptor:
    def test_basic_creation(self):
        v = _vec(8)
        ad = AggregatedDescriptor(fragment_id=0, vector=v, mode="concat")
        assert ad.fragment_id == 0

    def test_dim_property(self):
        v = _vec(16)
        ad = AggregatedDescriptor(fragment_id=0, vector=v, mode="concat")
        assert ad.dim == 16

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            AggregatedDescriptor(fragment_id=-1, vector=_vec(8), mode="concat")

    def test_2d_vector_raises(self):
        with pytest.raises(ValueError):
            AggregatedDescriptor(
                fragment_id=0,
                vector=np.ones((4, 4), dtype=np.float32),
                mode="concat",
            )

    def test_vector_converted_to_float32(self):
        v = _vec(8).astype(np.float64)
        ad = AggregatedDescriptor(fragment_id=0, vector=v, mode="concat")
        assert ad.vector.dtype == np.float32

    def test_source_dims_stored(self):
        ad = AggregatedDescriptor(
            fragment_id=1, vector=_vec(8), mode="concat",
            source_dims={"src0": 8}
        )
        assert ad.source_dims["src0"] == 8


# ─── TestL2Normalize ──────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_1d_unit_norm(self):
        v = _vec(16)
        n = l2_normalize(v)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-5)

    def test_2d_row_norms(self):
        M = np.random.default_rng(0).random((5, 16)).astype(np.float32)
        N = l2_normalize(M)
        row_norms = np.linalg.norm(N, axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-5)

    def test_zero_vector_unchanged(self):
        v = np.zeros(8, dtype=np.float32)
        n = l2_normalize(v)
        assert np.allclose(n, 0.0)

    def test_returns_float32(self):
        v = _vec(8)
        assert l2_normalize(v).dtype == np.float32


# ─── TestConcatenateDescriptors ───────────────────────────────────────────────

class TestConcatenateDescriptors:
    def test_returns_1d(self):
        d = _descs(2, 8)
        v = concatenate_descriptors(d)
        assert v.ndim == 1

    def test_length_is_sum(self):
        d = {"a": _vec(8), "b": _vec(12)}
        v = concatenate_descriptors(d, normalize=False)
        assert v.shape[0] == 20

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            concatenate_descriptors({})

    def test_normalized_unit_norm(self):
        d = _descs(2, 8)
        v = concatenate_descriptors(d, normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

    def test_no_normalize(self):
        d = {"a": np.ones(4, dtype=np.float32)}
        v = concatenate_descriptors(d, normalize=False)
        assert np.allclose(v, 1.0)

    def test_returns_float32(self):
        d = _descs(2, 8)
        assert concatenate_descriptors(d).dtype == np.float32


# ─── TestWeightedAverageDescriptors ──────────────────────────────────────────

class TestWeightedAverageDescriptors:
    def test_returns_1d(self):
        d = _descs(2, 16)
        v = weighted_average_descriptors(d)
        assert v.ndim == 1

    def test_same_dim_as_inputs(self):
        d = _descs(3, 16)
        v = weighted_average_descriptors(d, normalize=False)
        assert v.shape[0] == 16

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_average_descriptors({})

    def test_mismatched_dims_raise(self):
        d = {"a": _vec(8), "b": _vec(16)}
        with pytest.raises(ValueError):
            weighted_average_descriptors(d)

    def test_equal_weights_is_mean(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        expected = (a + b) / 2.0
        result = weighted_average_descriptors({"a": a, "b": b}, normalize=False)
        assert np.allclose(result, expected, atol=1e-5)

    def test_normalized_unit_norm(self):
        d = _descs(2, 16)
        v = weighted_average_descriptors(d, normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


# ─── TestPcaReduce ────────────────────────────────────────────────────────────

class TestPcaReduce:
    def test_output_shape(self):
        X = np.random.default_rng(0).random((10, 32)).astype(np.float64)
        Y = pca_reduce(X, n_components=8)
        assert Y.shape == (10, 8)

    def test_returns_float32(self):
        X = np.random.default_rng(0).random((5, 16))
        Y = pca_reduce(X, n_components=4)
        assert Y.dtype == np.float32

    def test_n_components_zero_raises(self):
        X = np.random.default_rng(0).random((5, 16))
        with pytest.raises(ValueError):
            pca_reduce(X, n_components=0)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(np.ones(16), n_components=4)

    def test_k_clamped_to_min_m_d(self):
        X = np.random.default_rng(0).random((3, 16))
        Y = pca_reduce(X, n_components=100)
        assert Y.shape[1] <= min(3, 16)


# ─── TestElementwiseAggregate ─────────────────────────────────────────────────

class TestElementwiseAggregate:
    def test_max_mode(self):
        a = np.array([1.0, 4.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 2.0, 5.0], dtype=np.float32)
        v = elementwise_aggregate({"a": a, "b": b}, mode="max", normalize=False)
        assert np.allclose(v, [3.0, 4.0, 5.0], atol=1e-5)

    def test_min_mode(self):
        a = np.array([1.0, 4.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 2.0, 5.0], dtype=np.float32)
        v = elementwise_aggregate({"a": a, "b": b}, mode="min", normalize=False)
        assert np.allclose(v, [1.0, 2.0, 2.0], atol=1e-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            elementwise_aggregate({})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            elementwise_aggregate({"a": _vec(4)}, mode="sum")

    def test_mismatched_dims_raise(self):
        with pytest.raises(ValueError):
            elementwise_aggregate({"a": _vec(4), "b": _vec(8)}, mode="max")

    def test_normalized_unit_norm(self):
        d = _descs(2, 16)
        v = elementwise_aggregate(d, mode="max", normalize=True)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


# ─── TestAggregate ────────────────────────────────────────────────────────────

class TestAggregate:
    def test_concat_mode(self):
        d = _descs(2, 8)
        v = aggregate(d, AggregatorConfig(mode="concat"))
        assert v.ndim == 1

    def test_weighted_avg_mode(self):
        d = _descs(2, 8)
        v = aggregate(d, AggregatorConfig(mode="weighted_avg"))
        assert v.shape[0] == 8

    def test_max_mode(self):
        d = _descs(2, 8)
        v = aggregate(d, AggregatorConfig(mode="max"))
        assert v.shape[0] == 8

    def test_min_mode(self):
        d = _descs(2, 8)
        v = aggregate(d, AggregatorConfig(mode="min"))
        assert v.shape[0] == 8

    def test_default_config(self):
        d = _descs(2, 8)
        v = aggregate(d, None)
        assert isinstance(v, np.ndarray)

    def test_returns_float32(self):
        d = _descs(2, 8)
        v = aggregate(d)
        assert v.dtype == np.float32


# ─── TestDistanceMatrix ───────────────────────────────────────────────────────

class TestDistanceMatrix:
    def test_shape(self):
        V = np.random.default_rng(0).random((5, 16)).astype(np.float32)
        D = distance_matrix(V)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        V = np.random.default_rng(0).random((4, 8)).astype(np.float32)
        D = distance_matrix(V)
        assert np.allclose(np.diag(D), 0.0)

    def test_cosine_metric(self):
        V = np.random.default_rng(0).random((4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert np.all(D >= 0.0)

    def test_euclidean_metric(self):
        V = np.random.default_rng(0).random((4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="euclidean")
        assert np.all(D >= -1e-6)

    def test_l1_metric(self):
        V = np.random.default_rng(0).random((3, 8)).astype(np.float32)
        D = distance_matrix(V, metric="l1")
        assert np.all(D >= 0.0)

    def test_invalid_metric_raises(self):
        V = np.ones((3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            distance_matrix(V, metric="hamming")

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            distance_matrix(np.ones(8, dtype=np.float32))

    def test_returns_float32(self):
        V = np.random.default_rng(0).random((3, 8)).astype(np.float32)
        D = distance_matrix(V)
        assert D.dtype == np.float32

    def test_symmetric_cosine(self):
        V = np.random.default_rng(0).random((5, 8)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert np.allclose(D, D.T, atol=1e-5)


# ─── TestBatchAggregate ───────────────────────────────────────────────────────

class TestBatchAggregate:
    def test_returns_list(self):
        groups = [_descs(2, 8), _descs(2, 8)]
        result = batch_aggregate(groups)
        assert isinstance(result, list)

    def test_length_matches(self):
        groups = [_descs(2, 8) for _ in range(4)]
        result = batch_aggregate(groups)
        assert len(result) == 4

    def test_all_ndarrays(self):
        groups = [_descs(2, 8), _descs(2, 8)]
        result = batch_aggregate(groups)
        assert all(isinstance(v, np.ndarray) for v in result)

    def test_empty_groups(self):
        result = batch_aggregate([])
        assert result == []

    def test_with_config(self):
        cfg = AggregatorConfig(mode="max")
        groups = [_descs(2, 8)]
        result = batch_aggregate(groups, cfg)
        assert result[0].shape[0] == 8
