"""Extra tests for puzzle_reconstruction/algorithms/descriptor_combiner.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.descriptor_combiner import (
    CombineConfig,
    DescriptorSet,
    CombineResult,
    combine_descriptors,
    combine_selected,
    batch_combine,
    descriptor_distance,
    build_distance_matrix,
    find_nearest,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _vec(n: int = 4, val: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) + val).astype(float)


def _dset(fid: int = 0, names=("shape", "texture", "color"),
          dim: int = 4) -> DescriptorSet:
    desc = {n: _vec(dim, seed=i) for i, n in enumerate(names)}
    return DescriptorSet(fragment_id=fid, descriptors=desc)


def _result(fid: int = 0, dim: int = 8) -> CombineResult:
    vec = np.random.default_rng(fid).random(dim)
    return CombineResult(fragment_id=fid, vector=vec,
                         used_names=["shape", "texture"],
                         original_dim=dim)


# ─── CombineConfig (extra) ───────────────────────────────────────────────────

class TestCombineConfigExtra:
    def test_default_normalize_true(self):
        assert CombineConfig().normalize is True

    def test_default_l2_final_true(self):
        assert CombineConfig().l2_final is True

    def test_default_pca_dim_none(self):
        assert CombineConfig().pca_dim is None

    def test_default_weights_empty(self):
        assert CombineConfig().weights == {}

    def test_weight_for_missing_returns_one(self):
        cfg = CombineConfig()
        assert cfg.weight_for("xyz") == pytest.approx(1.0)

    def test_weight_for_explicit(self):
        cfg = CombineConfig(weights={"color": 3.0})
        assert cfg.weight_for("color") == pytest.approx(3.0)

    def test_weight_large_ok(self):
        cfg = CombineConfig(weights={"a": 100.0})
        assert cfg.weights["a"] == pytest.approx(100.0)

    def test_multiple_weights(self):
        cfg = CombineConfig(weights={"a": 1.0, "b": 2.0, "c": 0.5})
        assert cfg.weight_for("b") == pytest.approx(2.0)
        assert cfg.weight_for("c") == pytest.approx(0.5)

    def test_pca_dim_large_ok(self):
        cfg = CombineConfig(pca_dim=512)
        assert cfg.pca_dim == 512

    def test_normalize_false_l2_false(self):
        cfg = CombineConfig(normalize=False, l2_final=False)
        assert cfg.normalize is False
        assert cfg.l2_final is False


# ─── DescriptorSet (extra) ───────────────────────────────────────────────────

class TestDescriptorSetExtra:
    def test_fragment_id_zero(self):
        ds = DescriptorSet(fragment_id=0, descriptors={})
        assert ds.fragment_id == 0

    def test_names_returns_all_keys(self):
        ds = _dset(names=("a", "b", "c", "d"))
        assert set(ds.names) == {"a", "b", "c", "d"}

    def test_total_dim_empty(self):
        ds = DescriptorSet(fragment_id=0, descriptors={})
        assert ds.total_dim == 0

    def test_total_dim_single(self):
        ds = DescriptorSet(fragment_id=0, descriptors={"x": np.ones(7)})
        assert ds.total_dim == 7

    def test_has_true_and_false(self):
        ds = _dset(names=("shape",))
        assert ds.has("shape") is True
        assert ds.has("color") is False

    def test_get_returns_copy_or_view(self):
        ds = _dset(names=("a",), dim=4)
        v = ds.get("a")
        assert v is not None
        assert v.shape == (4,)

    def test_get_missing_returns_none(self):
        ds = _dset(names=("a",))
        assert ds.get("nothere") is None

    def test_large_dim_ok(self):
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"big": np.zeros(1024)})
        assert ds.total_dim == 1024

    def test_multiple_descriptors_total_dim(self):
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(3),
                                        "b": np.ones(5),
                                        "c": np.ones(2)})
        assert ds.total_dim == 10


# ─── CombineResult (extra) ───────────────────────────────────────────────────

class TestCombineResultExtra:
    def test_dim_matches_vector_length(self):
        r = _result(dim=12)
        assert r.dim == 12

    def test_is_reduced_same_dim(self):
        r = CombineResult(fragment_id=0, vector=np.zeros(8),
                          used_names=["a"], original_dim=8)
        assert r.is_reduced is False

    def test_is_reduced_smaller_dim(self):
        r = CombineResult(fragment_id=0, vector=np.zeros(4),
                          used_names=["a"], original_dim=16)
        assert r.is_reduced is True

    def test_norm_unit_vector(self):
        v = np.array([0.0, 1.0])
        r = CombineResult(fragment_id=0, vector=v,
                          used_names=["a"], original_dim=2)
        assert r.norm == pytest.approx(1.0)

    def test_norm_nonneg(self):
        r = _result(dim=8)
        assert r.norm >= 0.0

    def test_used_names_stored(self):
        r = CombineResult(fragment_id=5, vector=np.zeros(4),
                          used_names=["x", "y"], original_dim=4)
        assert "x" in r.used_names
        assert "y" in r.used_names

    def test_fragment_id_stored(self):
        r = CombineResult(fragment_id=42, vector=np.zeros(4),
                          used_names=[], original_dim=4)
        assert r.fragment_id == 42

    def test_original_dim_larger_ok(self):
        r = CombineResult(fragment_id=0, vector=np.zeros(2),
                          used_names=[], original_dim=10)
        assert r.original_dim == 10


# ─── combine_descriptors (extra) ─────────────────────────────────────────────

class TestCombineDescriptorsExtra:
    def test_l2_final_unit_norm(self):
        cfg = CombineConfig(l2_final=True)
        r = combine_descriptors(_dset(), cfg)
        assert r.norm == pytest.approx(1.0, abs=1e-5)

    def test_no_l2_norm_any(self):
        cfg = CombineConfig(l2_final=False, normalize=False)
        r = combine_descriptors(_dset(), cfg)
        assert r.norm >= 0.0

    def test_single_descriptor_ok(self):
        ds = DescriptorSet(fragment_id=0, descriptors={"only": np.ones(5)})
        r = combine_descriptors(ds)
        assert r.dim > 0

    def test_many_descriptors_ok(self):
        desc = {f"d{i}": np.ones(4) for i in range(10)}
        ds = DescriptorSet(fragment_id=0, descriptors=desc)
        r = combine_descriptors(ds)
        assert r.dim > 0

    def test_original_dim_matches_total(self):
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(3), "b": np.ones(5)})
        r = combine_descriptors(ds)
        assert r.original_dim == 8

    def test_zero_weight_zeroes_part(self):
        cfg = CombineConfig(weights={"a": 0.0, "b": 1.0},
                            normalize=False, l2_final=False)
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(3), "b": np.ones(3)})
        r = combine_descriptors(ds, cfg)
        assert np.all(r.vector[:3] == pytest.approx(0.0))

    def test_fragment_id_preserved(self):
        r = combine_descriptors(_dset(fid=99))
        assert r.fragment_id == 99

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_descriptors(DescriptorSet(fragment_id=0, descriptors={}))


# ─── combine_selected (extra) ────────────────────────────────────────────────

class TestCombineSelectedExtra:
    def test_all_found_used(self):
        ds = _dset(names=("a", "b", "c"))
        r = combine_selected(ds, ["a", "b", "c"])
        assert set(r.used_names) == {"a", "b", "c"}

    def test_subset_used(self):
        ds = _dset(names=("a", "b", "c"))
        r = combine_selected(ds, ["b"])
        assert r.used_names == ["b"]

    def test_missing_names_filtered(self):
        ds = _dset(names=("a", "b"))
        r = combine_selected(ds, ["a", "z", "w"])
        assert "a" in r.used_names

    def test_fragment_id_preserved(self):
        ds = _dset(fid=77, names=("a",))
        r = combine_selected(ds, ["a"])
        assert r.fragment_id == 77

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError):
            combine_selected(_dset(), [])

    def test_no_matches_raises(self):
        ds = _dset(names=("a",))
        with pytest.raises(ValueError):
            combine_selected(ds, ["x", "y", "z"])

    def test_returns_combine_result(self):
        r = combine_selected(_dset(names=("a", "b")), ["a"])
        assert isinstance(r, CombineResult)


# ─── batch_combine (extra) ───────────────────────────────────────────────────

class TestBatchCombineExtra:
    def test_empty_returns_empty(self):
        assert batch_combine([]) == []

    def test_length_preserved(self):
        dsets = [_dset(fid=i) for i in range(6)]
        assert len(batch_combine(dsets)) == 6

    def test_all_instances(self):
        for r in batch_combine([_dset(fid=i) for i in range(3)]):
            assert isinstance(r, CombineResult)

    def test_fragment_ids_sequential(self):
        dsets = [_dset(fid=i) for i in range(4)]
        results = batch_combine(dsets)
        for i, r in enumerate(results):
            assert r.fragment_id == i

    def test_custom_config_applied(self):
        cfg = CombineConfig(l2_final=True)
        dsets = [_dset(fid=i) for i in range(3)]
        for r in batch_combine(dsets, cfg):
            assert r.norm == pytest.approx(1.0, abs=1e-5)

    def test_single_item(self):
        result = batch_combine([_dset(fid=0)])
        assert len(result) == 1


# ─── descriptor_distance (extra) ────────────────────────────────────────────

class TestDescriptorDistanceExtra:
    def test_cosine_same_zero(self):
        r = _result(dim=8)
        assert descriptor_distance(r, r, "cosine") == pytest.approx(0.0, abs=1e-5)

    def test_euclidean_same_zero(self):
        r = _result(dim=8)
        assert descriptor_distance(r, r, "euclidean") == pytest.approx(0.0, abs=1e-5)

    def test_l1_same_zero(self):
        r = _result(dim=8)
        assert descriptor_distance(r, r, "l1") == pytest.approx(0.0, abs=1e-5)

    def test_all_metrics_nonneg(self):
        r1 = _result(fid=0, dim=8)
        r2 = _result(fid=1, dim=8)
        for m in ("cosine", "euclidean", "l1"):
            assert descriptor_distance(r1, r2, m) >= 0.0

    def test_unknown_metric_raises(self):
        r = _result()
        with pytest.raises(ValueError):
            descriptor_distance(r, r, "manhattan_xyz")

    def test_l1_known_value(self):
        v1 = np.zeros(4)
        v2 = np.ones(4) * 2.0
        r1 = CombineResult(0, v1, [], 4)
        r2 = CombineResult(1, v2, [], 4)
        assert descriptor_distance(r1, r2, "l1") == pytest.approx(8.0)

    def test_euclidean_345(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([3.0, 4.0, 0.0])
        r1 = CombineResult(0, v1, [], 3)
        r2 = CombineResult(1, v2, [], 3)
        assert descriptor_distance(r1, r2, "euclidean") == pytest.approx(5.0)

    def test_different_dims_handled(self):
        r1 = CombineResult(0, np.ones(4), [], 4)
        r2 = CombineResult(1, np.ones(6), [], 6)
        dist = descriptor_distance(r1, r2, "euclidean")
        assert dist >= 0.0


# ─── build_distance_matrix (extra) ──────────────────────────────────────────

class TestBuildDistanceMatrixExtra:
    def test_empty_returns_empty(self):
        m = build_distance_matrix([])
        assert m.shape == (0, 0)

    def test_single_element(self):
        m = build_distance_matrix([_result()])
        assert m.shape == (1, 1)
        assert m[0, 0] == pytest.approx(0.0)

    def test_shape_nxn(self):
        for n in (2, 3, 5):
            m = build_distance_matrix([_result(fid=i) for i in range(n)])
            assert m.shape == (n, n)

    def test_diagonal_zero(self):
        results = [_result(fid=i) for i in range(4)]
        m = build_distance_matrix(results)
        np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-5)

    def test_symmetric(self):
        results = [_result(fid=i, dim=6) for i in range(5)]
        m = build_distance_matrix(results)
        assert np.allclose(m, m.T, atol=1e-5)

    def test_non_negative(self):
        results = [_result(fid=i) for i in range(4)]
        m = build_distance_matrix(results)
        assert np.all(m >= 0.0)

    def test_dtype_float32(self):
        results = [_result(fid=i) for i in range(3)]
        m = build_distance_matrix(results)
        assert m.dtype == np.float32

    def test_cosine_metric(self):
        results = [_result(fid=i) for i in range(3)]
        m = build_distance_matrix(results, metric="cosine")
        assert m.shape == (3, 3)
        np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-5)


# ─── find_nearest (extra) ────────────────────────────────────────────────────

class TestFindNearestExtra:
    def _cands(self, n=5):
        return [_result(fid=i, dim=8) for i in range(n)]

    def test_returns_list(self):
        r = find_nearest(_result(fid=99), self._cands())
        assert isinstance(r, list)

    def test_default_top_k_one(self):
        r = find_nearest(_result(fid=99), self._cands())
        assert len(r) >= 1

    def test_top_k_respected(self):
        r = find_nearest(_result(fid=99), self._cands(8), top_k=4)
        assert len(r) == 4

    def test_top_k_larger_than_candidates(self):
        r = find_nearest(_result(fid=99), self._cands(3), top_k=10)
        assert len(r) == 3

    def test_sorted_ascending(self):
        r = find_nearest(_result(fid=0), self._cands(5), top_k=5)
        dists = [d for _, d in r]
        assert dists == sorted(dists)

    def test_all_distances_nonneg(self):
        r = find_nearest(_result(fid=99), self._cands())
        for _, d in r:
            assert d >= 0.0

    def test_fragment_ids_are_ints(self):
        r = find_nearest(_result(fid=99), self._cands())
        for fid, _ in r:
            assert isinstance(fid, int)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_result(), self._cands(), top_k=0)

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_result(), [])

    def test_identical_vector_dist_zero(self):
        q = _result(fid=0, dim=8)
        cands = [CombineResult(0, q.vector.copy(), [], q.dim)]
        r = find_nearest(q, cands, top_k=1, metric="euclidean")
        assert r[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_l1_metric_ok(self):
        r = find_nearest(_result(fid=99), self._cands(), metric="l1")
        assert len(r) >= 1
