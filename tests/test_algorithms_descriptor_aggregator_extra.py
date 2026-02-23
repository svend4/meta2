"""Extra tests for puzzle_reconstruction.algorithms.descriptor_aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.descriptor_aggregator import (
    AggregatedDescriptor,
    AggregatorConfig,
    aggregate,
    batch_aggregate,
    concatenate_descriptors,
    distance_matrix,
    elementwise_aggregate,
    l2_normalize,
    pca_reduce,
    weighted_average_descriptors,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rng(seed=0):
    return np.random.default_rng(seed)


def _vec(d=16, seed=0):
    return _rng(seed).uniform(0.1, 1.0, d).astype(np.float32)


def _descs(keys=("a", "b"), d=16):
    return {k: _vec(d, i) for i, k in enumerate(keys)}


# ─── TestAggregatorConfigExtra ───────────────────────────────────────────────

class TestAggregatorConfigExtra:
    def test_mode_concat(self):
        cfg = AggregatorConfig(mode="concat")
        assert cfg.mode == "concat"

    def test_mode_weighted_avg(self):
        cfg = AggregatorConfig(mode="weighted_avg")
        assert cfg.mode == "weighted_avg"

    def test_mode_pca(self):
        cfg = AggregatorConfig(mode="pca")
        assert cfg.mode == "pca"

    def test_mode_max(self):
        cfg = AggregatorConfig(mode="max")
        assert cfg.mode == "max"

    def test_mode_min(self):
        cfg = AggregatorConfig(mode="min")
        assert cfg.mode == "min"

    def test_n_components_stored(self):
        cfg = AggregatorConfig(n_components=64)
        assert cfg.n_components == 64

    def test_normalize_false(self):
        cfg = AggregatorConfig(normalize=False)
        assert cfg.normalize is False

    def test_weights_stored(self):
        cfg = AggregatorConfig(weights={"a": 0.5, "b": 0.5})
        assert cfg.weights["a"] == pytest.approx(0.5)


# ─── TestAggregatedDescriptorExtra ───────────────────────────────────────────

class TestAggregatedDescriptorExtra:
    def test_dim_matches_vector(self):
        ad = AggregatedDescriptor(fragment_id=0,
                                  vector=np.zeros(64, dtype=np.float32),
                                  mode="concat")
        assert ad.dim == 64

    def test_float64_cast_to_float32(self):
        ad = AggregatedDescriptor(fragment_id=0,
                                  vector=np.ones(8, dtype=np.float64),
                                  mode="concat")
        assert ad.vector.dtype == np.float32

    def test_mode_stored(self):
        ad = AggregatedDescriptor(fragment_id=1,
                                  vector=np.zeros(4, dtype=np.float32),
                                  mode="max")
        assert ad.mode == "max"

    def test_fragment_id_stored(self):
        ad = AggregatedDescriptor(fragment_id=7,
                                  vector=np.zeros(8, dtype=np.float32),
                                  mode="concat")
        assert ad.fragment_id == 7

    def test_large_fragment_id(self):
        ad = AggregatedDescriptor(fragment_id=999,
                                  vector=np.zeros(8, dtype=np.float32),
                                  mode="concat")
        assert ad.fragment_id == 999


# ─── TestL2NormalizeExtra ────────────────────────────────────────────────────

class TestL2NormalizeExtra:
    def test_unit_norm_3d(self):
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        out = l2_normalize(v)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_all_ones_normalized(self):
        v = np.ones(8, dtype=np.float32)
        out = l2_normalize(v)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_shape_preserved_1d(self):
        v = _vec(32)
        assert l2_normalize(v).shape == (32,)

    def test_shape_preserved_2d(self):
        M = _rng().uniform(0.1, 1.0, (4, 16)).astype(np.float32)
        out = l2_normalize(M)
        assert out.shape == (4, 16)

    def test_2d_row_norms_one(self):
        M = _rng().uniform(0.1, 1.0, (6, 8)).astype(np.float32)
        out = l2_normalize(M)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_float32_output(self):
        v = _vec(16)
        assert l2_normalize(v).dtype == np.float32


# ─── TestConcatenateDescriptorsExtra ─────────────────────────────────────────

class TestConcatenateDescriptorsExtra:
    def test_three_descriptors_sum_dim(self):
        descs = {"a": _vec(8), "b": _vec(16), "c": _vec(4)}
        out = concatenate_descriptors(descs, normalize=False)
        assert out.shape == (28,)

    def test_single_descriptor_passthrough(self):
        v = _vec(16)
        descs = {"only": v}
        out = concatenate_descriptors(descs, normalize=False)
        np.testing.assert_allclose(out, v, atol=1e-6)

    def test_normalize_unit_norm(self):
        descs = _descs(d=8)
        out = concatenate_descriptors(descs, normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_dtype_float32(self):
        out = concatenate_descriptors(_descs(d=8))
        assert out.dtype == np.float32

    def test_ordering_deterministic(self):
        descs = _descs(d=4)
        out1 = concatenate_descriptors(descs, normalize=False)
        out2 = concatenate_descriptors(descs, normalize=False)
        np.testing.assert_array_equal(out1, out2)


# ─── TestWeightedAverageDescriptorsExtra ─────────────────────────────────────

class TestWeightedAverageDescriptorsExtra:
    def test_equal_weights_true_average(self):
        v1 = np.array([0.0, 1.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0], dtype=np.float32)
        out = weighted_average_descriptors({"a": v1, "b": v2}, normalize=False)
        np.testing.assert_allclose(out, [0.5, 0.5], atol=1e-6)

    def test_single_descriptor_returned(self):
        v = _vec(8)
        out = weighted_average_descriptors({"only": v}, normalize=False)
        np.testing.assert_allclose(out, v, atol=1e-6)

    def test_normalize_unit_norm(self):
        out = weighted_average_descriptors(_descs(d=8), normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_float32_output(self):
        out = weighted_average_descriptors(_descs(d=8))
        assert out.dtype == np.float32

    def test_custom_weights_applied(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        # weight a=3, b=1 → (3*[1,0]+1*[0,1])/4 = [0.75, 0.25]
        out = weighted_average_descriptors(
            {"a": v1, "b": v2},
            weights={"a": 3.0, "b": 1.0},
            normalize=False,
        )
        np.testing.assert_allclose(out, [0.75, 0.25], atol=1e-6)


# ─── TestPcaReduceExtra ──────────────────────────────────────────────────────

class TestPcaReduceExtra:
    def test_reduces_dimension(self):
        M = _rng().uniform(0, 1, (20, 64)).astype(np.float32)
        out = pca_reduce(M, n_components=8)
        assert out.shape == (20, 8)

    def test_float32_output(self):
        M = _rng().uniform(0, 1, (10, 16)).astype(np.float32)
        out = pca_reduce(M, n_components=4)
        assert out.dtype == np.float32

    def test_n_components_equal_min_dim(self):
        M = _rng().uniform(0, 1, (5, 8)).astype(np.float32)
        out = pca_reduce(M, n_components=5)
        assert out.shape[0] == 5

    def test_n_components_larger_than_samples_capped(self):
        M = _rng().uniform(0, 1, (3, 10)).astype(np.float32)
        out = pca_reduce(M, n_components=100)
        assert out.shape[1] <= 3

    def test_many_components_ok(self):
        M = _rng().uniform(0, 1, (50, 32)).astype(np.float32)
        out = pca_reduce(M, n_components=16)
        assert out.shape == (50, 16)


# ─── TestElementwiseAggregateExtra ───────────────────────────────────────────

class TestElementwiseAggregateExtra:
    def test_max_each_element(self):
        v1 = np.array([0.3, 0.8], dtype=np.float32)
        v2 = np.array([0.7, 0.2], dtype=np.float32)
        out = elementwise_aggregate({"a": v1, "b": v2}, mode="max", normalize=False)
        np.testing.assert_allclose(out, [0.7, 0.8], atol=1e-6)

    def test_min_each_element(self):
        v1 = np.array([0.3, 0.8], dtype=np.float32)
        v2 = np.array([0.7, 0.2], dtype=np.float32)
        out = elementwise_aggregate({"a": v1, "b": v2}, mode="min", normalize=False)
        np.testing.assert_allclose(out, [0.3, 0.2], atol=1e-6)

    def test_max_normalize_unit_norm(self):
        descs = _descs(d=8)
        out = elementwise_aggregate(descs, mode="max", normalize=True)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)

    def test_single_descriptor_passthrough(self):
        v = _vec(8)
        out = elementwise_aggregate({"only": v}, mode="max", normalize=False)
        np.testing.assert_allclose(out, v, atol=1e-6)

    def test_float32_output(self):
        out = elementwise_aggregate(_descs(d=8), mode="max")
        assert out.dtype == np.float32


# ─── TestAggregateExtra ──────────────────────────────────────────────────────

class TestAggregateExtra:
    def test_concat_dim_is_sum(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="concat")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (16,)  # 2 × 8

    def test_weighted_avg_dim_equals_d(self):
        descs = _descs(d=12)
        cfg = AggregatorConfig(mode="weighted_avg")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (12,)

    def test_max_dim_equals_d(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="max")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (8,)

    def test_min_dim_equals_d(self):
        descs = _descs(d=8)
        cfg = AggregatorConfig(mode="min")
        out = aggregate(descs, cfg=cfg)
        assert out.shape == (8,)

    def test_none_cfg_default(self):
        descs = _descs(d=8)
        out = aggregate(descs, cfg=None)
        assert isinstance(out, np.ndarray)

    def test_float32_output(self):
        assert aggregate(_descs(d=8)).dtype == np.float32


# ─── TestDistanceMatrixExtra ─────────────────────────────────────────────────

class TestDistanceMatrixExtra:
    def test_shape_4x4(self):
        V = _rng().uniform(0.1, 1.0, (4, 16)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert D.shape == (4, 4)

    def test_diagonal_zero_euclidean(self):
        V = _rng().uniform(0.1, 1.0, (5, 8)).astype(np.float32)
        D = distance_matrix(V, metric="euclidean")
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)

    def test_symmetric_euclidean(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="euclidean")
        np.testing.assert_allclose(D, D.T, atol=1e-5)

    def test_l1_nonneg(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="l1")
        assert D.min() >= 0.0

    def test_cosine_range_0_2(self):
        V = _rng().uniform(0.1, 1.0, (4, 8)).astype(np.float32)
        D = distance_matrix(V, metric="cosine")
        assert D.min() >= -1e-5
        assert D.max() <= 2.0 + 1e-5

    def test_identical_rows_zero_dist(self):
        v = _vec(16)
        V = np.stack([v, v, v], axis=0)
        D = distance_matrix(V, metric="euclidean")
        np.testing.assert_allclose(D, 0.0, atol=1e-5)


# ─── TestBatchAggregateExtra ─────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def test_five_groups(self):
        groups = [_descs(d=8) for _ in range(5)]
        result = batch_aggregate(groups)
        assert len(result) == 5

    def test_concat_shapes(self):
        groups = [_descs(("a", "b"), d=8), _descs(("a", "b"), d=8)]
        result = batch_aggregate(groups)
        assert all(v.shape == (16,) for v in result)

    def test_max_cfg_shapes(self):
        groups = [_descs(d=8), _descs(d=8)]
        cfg = AggregatorConfig(mode="max")
        result = batch_aggregate(groups, cfg=cfg)
        assert all(v.shape == (8,) for v in result)

    def test_all_float32(self):
        groups = [_descs(d=8) for _ in range(3)]
        result = batch_aggregate(groups)
        assert all(v.dtype == np.float32 for v in result)

    def test_single_group(self):
        result = batch_aggregate([_descs(d=8)])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
