"""Тесты для puzzle_reconstruction.algorithms.descriptor_aggregator."""
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

def _rng(seed=0):
    return np.random.default_rng(seed)


def _vec(d=16, seed=0) -> np.ndarray:
    return _rng(seed).uniform(0.1, 1.0, d).astype(np.float32)


def _descs(keys=("a", "b"), d=16) -> dict:
    return {k: _vec(d, i) for i, k in enumerate(keys)}


# ─── TestAggregatorConfig ─────────────────────────────────────────────────────

class TestAggregatorConfig:
    def test_defaults_ok(self):
        cfg = AggregatorConfig()
        assert cfg.mode == "concat"
        assert cfg.n_components == 32
        assert cfg.normalize is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(mode="unknown_mode")

    def test_all_valid_modes(self):
        for mode in ("concat", "weighted_avg", "pca", "max", "min"):
            cfg = AggregatorConfig(mode=mode)
            assert cfg.mode == mode

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(weights={"src": -0.1})

    def test_n_components_zero_raises(self):
        with pytest.raises(ValueError):
            AggregatorConfig(n_components=0)

    def test_n_components_one_ok(self):
        cfg = AggregatorConfig(n_components=1)
        assert cfg.n_components == 1


# ─── TestAggregatedDescriptor ─────────────────────────────────────────────────

class TestAggregatedDescriptor:
    def test_valid_construction(self):
        ad = AggregatedDescriptor(
            fragment_id=0, vector=np.ones(16, dtype=np.float32), mode="concat"
        )
        assert ad.dim == 16

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            AggregatedDescriptor(
                fragment_id=-1, vector=np.ones(8, dtype=np.float32), mode="concat"
            )

    def test_2d_vector_raises(self):
        with pytest.raises(ValueError):
            AggregatedDescriptor(
                fragment_id=0,
                vector=np.ones((4, 4), dtype=np.float32),
                mode="concat",
            )

    def test_dim_property(self):
        ad = AggregatedDescriptor(
            fragment_id=0, vector=np.zeros(32, dtype=np.float32), mode="concat"
        )
        assert ad.dim == 32

    def test_vector_cast_to_float32(self):
        ad = AggregatedDescriptor(
            fragment_id=0, vector=np.ones(8, dtype=np.float64), mode="concat"
        )
        assert ad.vector.dtype == np.float32


# ─── TestL2Normalize ──────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_unit_norm_1d(self):
        v = _vec(16)
        out = l2_normalize(v)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_unit_norm_each_row_2d(self):
        M = _rng(0).uniform(0.1, 1.0, (5, 16)).astype(np.float32)
        out = l2_normalize(M)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_vector_returned_unchanged(self):
        v = np.zeros(8, dtype=np.float32)
        out = l2_normalize(v)
        assert (out == 0.0).all()

    def test_returns_float32(self):
        out = l2_normalize(_vec(8))
        assert out.dtype == np.float32


# ─── TestConcatenateDescriptors ───────────────────────────────────────────────

class TestConcatenateDescriptors:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            concatenate_descriptors({})

    def test_output_dim_sum(self):
        descs = {"a": _vec(8), "b": _vec(16)}
        out = concatenate_descriptors(descs, normalize=False)
        assert out.shape == (24,)

    def test_normalize_true_unit_norm(self):
        descs = _descs(d=8)
        out = concatenate_descriptors(descs, normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_normalize_false_preserves_values(self):
        descs = {"only": np.array([3.0, 4.0], dtype=np.float32)}
        out = concatenate_descriptors(descs, normalize=False)
        np.testing.assert_allclose(out, [3.0, 4.0], atol=1e-6)

    def test_returns_float32(self):
        out = concatenate_descriptors(_descs())
        assert out.dtype == np.float32


# ─── TestWeightedAverageDescriptors ──────────────────────────────────────────

class TestWeightedAverageDescriptors:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_average_descriptors({})

    def test_dim_mismatch_raises(self):
        descs = {"a": _vec(8), "b": _vec(16)}
        with pytest.raises(ValueError):
            weighted_average_descriptors(descs)

    def test_output_shape(self):
        descs = _descs(d=16)
        out = weighted_average_descriptors(descs)
        assert out.shape == (16,)

    def test_equal_weights_average(self):
        v1 = np.array([2.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 2.0], dtype=np.float32)
        descs = {"a": v1, "b": v2}
        out = weighted_average_descriptors(descs, normalize=False)
        np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-5)

    def test_normalize_true_unit_norm(self):
        descs = _descs(d=8)
        out = weighted_average_descriptors(descs, normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


# ─── TestPcaReduce ────────────────────────────────────────────────────────────

class TestPcaReduce:
    def test_output_shape(self):
        M = _rng().uniform(0, 1, (10, 32)).astype(np.float32)
        out = pca_reduce(M, n_components=8)
        assert out.shape == (10, 8)

    def test_n_components_zero_raises(self):
        M = _rng().uniform(0, 1, (5, 10)).astype(np.float32)
        with pytest.raises(ValueError):
            pca_reduce(M, n_components=0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(np.ones(10), n_components=2)

    def test_returns_float32(self):
        M = _rng().uniform(0, 1, (5, 8)).astype(np.float64)
        out = pca_reduce(M, n_components=3)
        assert out.dtype == np.float32

    def test_n_components_capped_at_min_dim(self):
        M = _rng().uniform(0, 1, (4, 6)).astype(np.float32)
        # Request more components than samples
        out = pca_reduce(M, n_components=10)
        assert out.shape[1] <= min(4, 6)


# ─── TestElementwiseAggregate ─────────────────────────────────────────────────

class TestElementwiseAggregate:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            elementwise_aggregate({})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            elementwise_aggregate(_descs(d=8), mode="mean")

    def test_dim_mismatch_raises(self):
        descs = {"a": _vec(8), "b": _vec(16)}
        with pytest.raises(ValueError):
            elementwise_aggregate(descs, mode="max")

    def test_max_output_shape(self):
        descs = _descs(d=16)
        out = elementwise_aggregate(descs, mode="max")
        assert out.shape == (16,)

    def test_max_geq_min(self):
        descs = _descs(d=8)
        max_out = elementwise_aggregate(descs, mode="max", normalize=False)
        min_out = elementwise_aggregate(descs, mode="min", normalize=False)
        assert (max_out >= min_out - 1e-6).all()

    def test_normalize_true_unit_norm(self):
        descs = _descs(d=8)
        out = elementwise_aggregate(descs, mode="max", normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


# ─── TestAggregate ────────────────────────────────────────────────────────────

class TestAggregate:
    def test_default_concat(self):
        descs = _descs(d=8)
        out = aggregate(descs)
        assert out.shape == (16,)

    def test_weighted_avg_mode(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="weighted_avg")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (8,)

    def test_max_mode(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="max")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (8,)

    def test_min_mode(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="min")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (8,)

    def test_none_cfg_uses_defaults(self):
        descs = _descs(d=8)
        out = aggregate(descs, cfg=None)
        assert isinstance(out, np.ndarray)

    def test_returns_float32(self):
        out = aggregate(_descs(d=8))
        assert out.dtype == np.float32


# ─── TestDistanceMatrix ───────────────────────────────────────────────────────

class TestDistanceMatrix:
    def test_diagonal_zero_cosine(self):
        V = _rng().uniform(0.1, 1.0, (5, 16)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_symmetric_cosine(self):
        V = _rng().uniform(0.1, 1.0, (5, 16)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_identical_vectors_zero_dist_cosine(self):
        v = _vec(16)
        V = np.stack([v, v], axis=0)
        D = distance_matrix(V, metric="cosine")
        assert D[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_euclidean_metric(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="euclidean")
        assert D.shape == (4, 4)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_l1_metric(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="l1")
        assert D.shape == (4, 4)

    def test_invalid_metric_raises(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        with pytest.raises(ValueError):
            distance_matrix(V, metric="manhattan")

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            distance_matrix(np.ones(8))

    def test_returns_float32(self):
        V = _rng().uniform(0.1, 1.0, (3, 8)).astype(np.float32)
        D = distance_matrix(V)
        assert D.dtype == np.float32


# ─── TestBatchAggregate ───────────────────────────────────────────────────────

class TestBatchAggregate:
    def test_returns_list(self):
        groups = [_descs(d=8), _descs(d=8)]
        result = batch_aggregate(groups)
        assert isinstance(result, list)

    def test_length_matches(self):
        groups = [_descs(d=8) for _ in range(4)]
        result = batch_aggregate(groups)
        assert len(result) == 4

    def test_empty_list(self):
        result = batch_aggregate([])
        assert result == []

    def test_all_arrays(self):
        groups = [_descs(d=8), _descs(d=8)]
        for v in batch_aggregate(groups):
            assert isinstance(v, np.ndarray)

    def test_cfg_passed_through(self):
        groups = [_descs(d=8), _descs(d=8)]
        cfg = AggregatorConfig(mode="max")
        result = batch_aggregate(groups, cfg=cfg)
        # max output dim = 8 (not 16)
        assert result[0].shape == (8,)
