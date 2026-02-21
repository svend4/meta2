"""Тесты для puzzle_reconstruction.algorithms.descriptor_combiner."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

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


# ─── TestCombineConfig ────────────────────────────────────────────────────────

class TestCombineConfig:
    def test_defaults(self):
        cfg = CombineConfig()
        assert cfg.normalize is True
        assert cfg.l2_final is True
        assert cfg.pca_dim is None
        assert cfg.weights == {}

    def test_custom(self):
        cfg = CombineConfig(weights={"shape": 0.5}, normalize=False,
                            l2_final=False, pca_dim=16)
        assert cfg.weights["shape"] == pytest.approx(0.5)
        assert cfg.pca_dim == 16

    def test_weight_zero_ok(self):
        cfg = CombineConfig(weights={"a": 0.0})
        assert cfg.weights["a"] == 0.0

    def test_weight_neg_raises(self):
        with pytest.raises(ValueError):
            CombineConfig(weights={"a": -0.1})

    def test_pca_dim_one_ok(self):
        cfg = CombineConfig(pca_dim=1)
        assert cfg.pca_dim == 1

    def test_pca_dim_zero_raises(self):
        with pytest.raises(ValueError):
            CombineConfig(pca_dim=0)

    def test_pca_dim_neg_raises(self):
        with pytest.raises(ValueError):
            CombineConfig(pca_dim=-1)

    def test_weight_for_known(self):
        cfg = CombineConfig(weights={"shape": 2.0})
        assert cfg.weight_for("shape") == pytest.approx(2.0)

    def test_weight_for_unknown_default_one(self):
        cfg = CombineConfig()
        assert cfg.weight_for("unknown") == pytest.approx(1.0)

    def test_weight_for_explicit_one(self):
        cfg = CombineConfig(weights={"a": 1.0})
        assert cfg.weight_for("a") == pytest.approx(1.0)


# ─── TestDescriptorSet ────────────────────────────────────────────────────────

class TestDescriptorSet:
    def test_basic(self):
        ds = _dset(fid=3)
        assert ds.fragment_id == 3

    def test_names(self):
        ds = _dset(names=("a", "b", "c"))
        assert set(ds.names) == {"a", "b", "c"}

    def test_total_dim(self):
        ds = _dset(names=("a", "b"), dim=6)
        assert ds.total_dim == 12

    def test_has_true(self):
        ds = _dset(names=("shape",))
        assert ds.has("shape") is True

    def test_has_false(self):
        ds = _dset(names=("shape",))
        assert ds.has("color") is False

    def test_get_returns_array(self):
        ds = _dset(names=("shape",))
        v = ds.get("shape")
        assert isinstance(v, np.ndarray)

    def test_get_missing_none(self):
        ds = _dset(names=("shape",))
        assert ds.get("xyz") is None

    def test_empty_descriptors_ok(self):
        ds = DescriptorSet(fragment_id=0, descriptors={})
        assert ds.total_dim == 0

    def test_2d_descriptor_raises(self):
        with pytest.raises(ValueError):
            DescriptorSet(fragment_id=0,
                          descriptors={"a": np.zeros((4, 4))})

    def test_1d_ok(self):
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.zeros(8)})
        assert ds.total_dim == 8


# ─── TestCombineResult ────────────────────────────────────────────────────────

class TestCombineResult:
    def test_dim(self):
        r = _result(dim=16)
        assert r.dim == 16

    def test_is_reduced_false(self):
        r = _result(dim=8)
        assert r.is_reduced is False

    def test_is_reduced_true(self):
        r = CombineResult(fragment_id=0,
                          vector=np.zeros(4),
                          used_names=["a"],
                          original_dim=8)
        assert r.is_reduced is True

    def test_norm_unit_vector(self):
        v = np.array([3.0, 4.0])
        r = CombineResult(fragment_id=0, vector=v,
                          used_names=["a"], original_dim=2)
        assert r.norm == pytest.approx(5.0)

    def test_norm_zero_vector(self):
        r = CombineResult(fragment_id=0, vector=np.zeros(4),
                          used_names=["a"], original_dim=4)
        assert r.norm == pytest.approx(0.0)

    def test_2d_vector_raises(self):
        with pytest.raises(ValueError):
            CombineResult(fragment_id=0, vector=np.zeros((4, 4)),
                          used_names=["a"], original_dim=4)

    def test_original_dim_neg_raises(self):
        with pytest.raises(ValueError):
            CombineResult(fragment_id=0, vector=np.zeros(4),
                          used_names=["a"], original_dim=-1)

    def test_original_dim_zero_ok(self):
        r = CombineResult(fragment_id=0, vector=np.zeros(0),
                          used_names=[], original_dim=0)
        assert r.original_dim == 0


# ─── TestCombineDescriptors ───────────────────────────────────────────────────

class TestCombineDescriptors:
    def test_returns_combine_result(self):
        r = combine_descriptors(_dset())
        assert isinstance(r, CombineResult)

    def test_fragment_id_stored(self):
        r = combine_descriptors(_dset(fid=7))
        assert r.fragment_id == 7

    def test_used_names_stored(self):
        ds = _dset(names=("a", "b"))
        r = combine_descriptors(ds)
        assert set(r.used_names) == {"a", "b"}

    def test_empty_descriptors_raises(self):
        ds = DescriptorSet(fragment_id=0, descriptors={})
        with pytest.raises(ValueError):
            combine_descriptors(ds)

    def test_l2_final_unit_norm(self):
        cfg = CombineConfig(l2_final=True)
        r = combine_descriptors(_dset(), cfg)
        assert r.norm == pytest.approx(1.0, abs=1e-6)

    def test_no_l2_final_any_norm(self):
        cfg = CombineConfig(l2_final=False, normalize=False)
        r = combine_descriptors(_dset(), cfg)
        assert r.norm >= 0.0

    def test_dimension_is_sum_of_parts(self):
        cfg = CombineConfig(l2_final=False, normalize=False,
                            weights={"a": 1.0, "b": 1.0})
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(4), "b": np.ones(6)})
        r = combine_descriptors(ds, cfg)
        assert r.dim == 10

    def test_original_dim_stored(self):
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(4), "b": np.ones(6)})
        r = combine_descriptors(ds)
        assert r.original_dim == 10

    def test_weight_zero_zeroes_part(self):
        cfg = CombineConfig(weights={"a": 0.0, "b": 1.0},
                            normalize=False, l2_final=False)
        ds = DescriptorSet(fragment_id=0,
                           descriptors={"a": np.ones(4), "b": np.ones(4)})
        r = combine_descriptors(ds, cfg)
        # part "a" должна быть нулевой
        assert np.all(r.vector[:4] == pytest.approx(0.0))

    def test_single_descriptor(self):
        ds = DescriptorSet(fragment_id=0, descriptors={"a": np.ones(8)})
        r = combine_descriptors(ds)
        assert r.dim > 0

    def test_default_config_used(self):
        r = combine_descriptors(_dset())
        assert r.fragment_id >= 0


# ─── TestCombineSelected ──────────────────────────────────────────────────────

class TestCombineSelected:
    def test_selects_subset(self):
        ds = _dset(names=("a", "b", "c"), dim=4)
        r = combine_selected(ds, ["a", "b"])
        assert set(r.used_names) == {"a", "b"}

    def test_empty_names_raises(self):
        ds = _dset(names=("a",))
        with pytest.raises(ValueError):
            combine_selected(ds, [])

    def test_none_found_raises(self):
        ds = _dset(names=("a",))
        with pytest.raises(ValueError):
            combine_selected(ds, ["x", "y"])

    def test_some_missing_ok(self):
        ds = _dset(names=("a", "b"))
        r = combine_selected(ds, ["a", "z"])  # z отсутствует
        assert "a" in r.used_names

    def test_returns_combine_result(self):
        r = combine_selected(_dset(names=("a", "b")), ["a"])
        assert isinstance(r, CombineResult)

    def test_single_name(self):
        ds = _dset(names=("shape", "texture"))
        r = combine_selected(ds, ["shape"])
        assert r.used_names == ["shape"]

    def test_all_names(self):
        ds = _dset(names=("a", "b", "c"))
        r = combine_selected(ds, ["a", "b", "c"])
        assert set(r.used_names) == {"a", "b", "c"}

    def test_fragment_id_preserved(self):
        ds = _dset(fid=42, names=("a",))
        r = combine_selected(ds, ["a"])
        assert r.fragment_id == 42


# ─── TestBatchCombine ─────────────────────────────────────────────────────────

class TestBatchCombine:
    def test_returns_list(self):
        dsets = [_dset(fid=i) for i in range(3)]
        results = batch_combine(dsets)
        assert isinstance(results, list)

    def test_length_matches(self):
        dsets = [_dset(fid=i) for i in range(5)]
        assert len(batch_combine(dsets)) == 5

    def test_empty_input(self):
        assert batch_combine([]) == []

    def test_all_combine_results(self):
        dsets = [_dset(fid=i) for i in range(3)]
        for r in batch_combine(dsets):
            assert isinstance(r, CombineResult)

    def test_fragment_ids_order(self):
        dsets = [_dset(fid=i) for i in range(4)]
        results = batch_combine(dsets)
        for i, r in enumerate(results):
            assert r.fragment_id == i

    def test_custom_config(self):
        cfg = CombineConfig(l2_final=False, normalize=False)
        dsets = [_dset(fid=i) for i in range(2)]
        results = batch_combine(dsets, cfg)
        assert len(results) == 2


# ─── TestDescriptorDistance ───────────────────────────────────────────────────

class TestDescriptorDistance:
    def test_same_vector_cosine_zero(self):
        r = _result(dim=8)
        assert descriptor_distance(r, r, "cosine") == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_cosine_one(self):
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        r1 = CombineResult(0, v1, [], 4)
        r2 = CombineResult(1, v2, [], 4)
        assert descriptor_distance(r1, r2, "cosine") == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_cosine_one(self):
        r1 = CombineResult(0, np.zeros(4), [], 4)
        r2 = _result(dim=4)
        assert descriptor_distance(r1, r2, "cosine") == pytest.approx(1.0)

    def test_euclidean_same_zero(self):
        r = _result(dim=4)
        assert descriptor_distance(r, r, "euclidean") == pytest.approx(0.0)

    def test_euclidean_known_distance(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        r1 = CombineResult(0, v1, [], 2)
        r2 = CombineResult(1, v2, [], 2)
        assert descriptor_distance(r1, r2, "euclidean") == pytest.approx(5.0)

    def test_l1_same_zero(self):
        r = _result(dim=4)
        assert descriptor_distance(r, r, "l1") == pytest.approx(0.0)

    def test_l1_known(self):
        v1 = np.zeros(3)
        v2 = np.ones(3)
        r1 = CombineResult(0, v1, [], 3)
        r2 = CombineResult(1, v2, [], 3)
        assert descriptor_distance(r1, r2, "l1") == pytest.approx(3.0)

    def test_unknown_metric_raises(self):
        r = _result()
        with pytest.raises(ValueError):
            descriptor_distance(r, r, "unknown")

    def test_different_dims_padded(self):
        v1 = np.ones(4)
        v2 = np.ones(6)
        r1 = CombineResult(0, v1, [], 4)
        r2 = CombineResult(1, v2, [], 6)
        dist = descriptor_distance(r1, r2, "euclidean")
        assert dist >= 0.0

    def test_non_negative(self):
        r1 = _result(fid=0)
        r2 = _result(fid=1)
        for metric in ("cosine", "euclidean", "l1"):
            assert descriptor_distance(r1, r2, metric) >= 0.0


# ─── TestBuildDistanceMatrix ──────────────────────────────────────────────────

class TestBuildDistanceMatrix:
    def test_shape(self):
        results = [_result(fid=i) for i in range(4)]
        m = build_distance_matrix(results)
        assert m.shape == (4, 4)

    def test_dtype_float32(self):
        results = [_result(fid=i) for i in range(3)]
        m = build_distance_matrix(results)
        assert m.dtype == np.float32

    def test_diagonal_zero(self):
        results = [_result(fid=i) for i in range(4)]
        m = build_distance_matrix(results)
        for i in range(4):
            assert m[i, i] == pytest.approx(0.0)

    def test_symmetric(self):
        results = [_result(fid=i) for i in range(4)]
        m = build_distance_matrix(results)
        assert np.allclose(m, m.T, atol=1e-5)

    def test_empty_input(self):
        m = build_distance_matrix([])
        assert m.shape == (0, 0)

    def test_single_result(self):
        m = build_distance_matrix([_result()])
        assert m.shape == (1, 1)
        assert m[0, 0] == pytest.approx(0.0)

    def test_non_negative(self):
        results = [_result(fid=i) for i in range(5)]
        m = build_distance_matrix(results)
        assert np.all(m >= 0.0)

    def test_cosine_metric(self):
        results = [_result(fid=i) for i in range(3)]
        m = build_distance_matrix(results, metric="cosine")
        assert m.shape == (3, 3)


# ─── TestFindNearest ──────────────────────────────────────────────────────────

class TestFindNearest:
    def _make_results(self, n=5):
        return [_result(fid=i, dim=8) for i in range(n)]

    def test_returns_list(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results())
        assert isinstance(result, list)

    def test_length_top_k(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results(5), top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_candidates(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results(3), top_k=10)
        assert len(result) == 3

    def test_sorted_ascending(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results(), top_k=5)
        dists = [d for _, d in result]
        assert dists == sorted(dists)

    def test_fragment_ids_stored(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results())
        for fid, dist in result:
            assert isinstance(fid, int)
            assert dist >= 0.0

    def test_identical_vector_first(self):
        query = _result(fid=0, dim=8)
        # Первый кандидат — копия query
        candidates = [
            CombineResult(0, query.vector.copy(), [], query.dim),
        ] + [_result(fid=i + 1, dim=8) for i in range(4)]
        result = find_nearest(query, candidates, top_k=1, metric="euclidean")
        assert result[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_result(), self._make_results(), top_k=0)

    def test_top_k_neg_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_result(), self._make_results(), top_k=-1)

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_result(), [])

    def test_l1_metric(self):
        query = _result(fid=99)
        result = find_nearest(query, self._make_results(), metric="l1")
        assert len(result) > 0
