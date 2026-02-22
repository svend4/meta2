"""Tests for puzzle_reconstruction.algorithms.descriptor_combiner."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.descriptor_combiner import (
    CombineConfig,
    CombineResult,
    DescriptorSet,
    batch_combine,
    build_distance_matrix,
    combine_descriptors,
    combine_selected,
    descriptor_distance,
    find_nearest,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _vec(size=8, seed=0):
    return np.random.default_rng(seed).random(size).astype(np.float32)


def _desc_set(fid=0, names=("shape", "texture"), sizes=(8, 16)):
    descs = {n: _vec(s, i) for i, (n, s) in enumerate(zip(names, sizes))}
    return DescriptorSet(fragment_id=fid, descriptors=descs)


def _combine_result(fid=0, dim=8, orig_dim=16, seed=0):
    v = _vec(dim, seed)
    return CombineResult(fragment_id=fid, vector=v,
                         used_names=["a"], original_dim=orig_dim)


# ─── TestCombineConfig ────────────────────────────────────────────────────────

class TestCombineConfig:
    def test_defaults(self):
        cfg = CombineConfig()
        assert cfg.normalize is True
        assert cfg.l2_final is True
        assert cfg.pca_dim is None
        assert cfg.weights == {}

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="Вес"):
            CombineConfig(weights={"shape": -1.0})

    def test_pca_dim_zero_raises(self):
        with pytest.raises(ValueError, match="pca_dim"):
            CombineConfig(pca_dim=0)

    def test_weight_for_present(self):
        cfg = CombineConfig(weights={"shape": 2.5})
        assert cfg.weight_for("shape") == pytest.approx(2.5)

    def test_weight_for_absent_returns_one(self):
        cfg = CombineConfig()
        assert cfg.weight_for("anything") == pytest.approx(1.0)

    def test_zero_weight_allowed(self):
        cfg = CombineConfig(weights={"shape": 0.0})
        assert cfg.weight_for("shape") == pytest.approx(0.0)


# ─── TestDescriptorSet ────────────────────────────────────────────────────────

class TestDescriptorSet:
    def test_basic_creation(self):
        ds = _desc_set()
        assert ds.fragment_id == 0
        assert len(ds.descriptors) == 2

    def test_non_1d_descriptor_raises(self):
        with pytest.raises(ValueError, match="1D"):
            DescriptorSet(fragment_id=0, descriptors={"bad": np.zeros((3, 4))})

    def test_names_property(self):
        ds = _desc_set(names=("a", "b"))
        assert set(ds.names) == {"a", "b"}

    def test_total_dim(self):
        ds = _desc_set(names=("s", "t"), sizes=(8, 16))
        assert ds.total_dim == 24

    def test_has_returns_true(self):
        ds = _desc_set(names=("shape",))
        assert ds.has("shape") is True

    def test_has_returns_false(self):
        ds = _desc_set(names=("shape",))
        assert ds.has("nonexistent") is False

    def test_get_returns_vector(self):
        ds = _desc_set(names=("shape",))
        v = ds.get("shape")
        assert v is not None
        assert v.ndim == 1

    def test_get_missing_returns_none(self):
        ds = _desc_set(names=("shape",))
        assert ds.get("color") is None

    def test_default_descriptors_empty(self):
        ds = DescriptorSet(fragment_id=5)
        assert ds.descriptors == {}
        assert ds.total_dim == 0


# ─── TestCombineResult ────────────────────────────────────────────────────────

class TestCombineResult:
    def test_basic_creation(self):
        r = _combine_result()
        assert isinstance(r, CombineResult)

    def test_non_1d_vector_raises(self):
        with pytest.raises(ValueError, match="1D"):
            CombineResult(fragment_id=0, vector=np.zeros((3, 4)),
                          used_names=[], original_dim=12)

    def test_negative_original_dim_raises(self):
        with pytest.raises(ValueError, match="original_dim"):
            CombineResult(fragment_id=0, vector=np.zeros(4),
                          used_names=[], original_dim=-1)

    def test_dim_property(self):
        r = _combine_result(dim=10, orig_dim=10)
        assert r.dim == 10

    def test_is_reduced_true(self):
        r = _combine_result(dim=4, orig_dim=10)
        assert r.is_reduced is True

    def test_is_reduced_false(self):
        r = _combine_result(dim=10, orig_dim=10)
        assert r.is_reduced is False

    def test_norm_nonneg(self):
        r = _combine_result()
        assert r.norm >= 0.0


# ─── TestCombineDescriptors ───────────────────────────────────────────────────

class TestCombineDescriptors:
    def test_empty_descriptor_set_raises(self):
        ds = DescriptorSet(fragment_id=0)
        with pytest.raises(ValueError, match="пустым"):
            combine_descriptors(ds)

    def test_returns_combine_result(self):
        ds = _desc_set()
        result = combine_descriptors(ds)
        assert isinstance(result, CombineResult)

    def test_fragment_id_preserved(self):
        ds = _desc_set(fid=42)
        result = combine_descriptors(ds)
        assert result.fragment_id == 42

    def test_used_names_match_descriptors(self):
        ds = _desc_set(names=("a", "b"))
        result = combine_descriptors(ds)
        assert set(result.used_names) == {"a", "b"}

    def test_vector_is_1d(self):
        ds = _desc_set()
        result = combine_descriptors(ds)
        assert result.vector.ndim == 1

    def test_l2_final_unit_norm(self):
        ds = _desc_set()
        cfg = CombineConfig(l2_final=True)
        result = combine_descriptors(ds, cfg)
        assert result.norm == pytest.approx(1.0, abs=1e-6)

    def test_no_l2_final_not_normalized(self):
        ds = _desc_set(names=("s",), sizes=(8,))
        cfg = CombineConfig(l2_final=False, normalize=False,
                            weights={"s": 10.0})
        result = combine_descriptors(ds, cfg)
        assert result.norm > 1.0  # weights amplify

    def test_vector_dimension_is_sum_of_parts(self):
        ds = _desc_set(names=("a", "b"), sizes=(5, 7))
        cfg = CombineConfig(l2_final=False)
        result = combine_descriptors(ds, cfg)
        assert result.original_dim == 12


# ─── TestCombineSelected ──────────────────────────────────────────────────────

class TestCombineSelected:
    def test_empty_names_raises(self):
        ds = _desc_set()
        with pytest.raises(ValueError, match="names"):
            combine_selected(ds, [])

    def test_no_found_names_raises(self):
        ds = _desc_set(names=("a",))
        with pytest.raises(ValueError):
            combine_selected(ds, ["nonexistent"])

    def test_selects_subset(self):
        ds = _desc_set(names=("a", "b"), sizes=(5, 7))
        result = combine_selected(ds, ["a"])
        assert "b" not in result.used_names

    def test_returns_combine_result(self):
        ds = _desc_set()
        result = combine_selected(ds, list(ds.names))
        assert isinstance(result, CombineResult)


# ─── TestBatchCombine ─────────────────────────────────────────────────────────

class TestBatchCombine:
    def test_returns_list(self):
        sets = [_desc_set(fid=i) for i in range(3)]
        results = batch_combine(sets)
        assert len(results) == 3
        assert all(isinstance(r, CombineResult) for r in results)

    def test_empty_list(self):
        assert batch_combine([]) == []

    def test_fragment_ids_preserved(self):
        sets = [_desc_set(fid=i) for i in range(4)]
        results = batch_combine(sets)
        ids = [r.fragment_id for r in results]
        assert ids == [0, 1, 2, 3]


# ─── TestDescriptorDistance ───────────────────────────────────────────────────

class TestDescriptorDistance:
    def test_unknown_metric_raises(self):
        r1 = _combine_result()
        r2 = _combine_result()
        with pytest.raises(ValueError, match="metric"):
            descriptor_distance(r1, r2, metric="invalid")

    def test_cosine_identical_is_zero(self):
        r = _combine_result()
        d = descriptor_distance(r, r, metric="cosine")
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_euclidean_identical_is_zero(self):
        r = _combine_result()
        d = descriptor_distance(r, r, metric="euclidean")
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_l1_identical_is_zero(self):
        r = _combine_result()
        d = descriptor_distance(r, r, metric="l1")
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_nonneg(self):
        r1 = _combine_result(seed=0)
        r2 = _combine_result(seed=1)
        for metric in ("cosine", "euclidean", "l1"):
            assert descriptor_distance(r1, r2, metric) >= 0.0

    def test_different_dims_padded(self):
        r1 = _combine_result(dim=4, orig_dim=4)
        r2 = _combine_result(dim=8, orig_dim=8, seed=1)
        d = descriptor_distance(r1, r2, metric="euclidean")
        assert d >= 0.0


# ─── TestBuildDistanceMatrix ──────────────────────────────────────────────────

class TestBuildDistanceMatrix:
    def test_shape(self):
        results = [_combine_result(seed=i) for i in range(4)]
        mat = build_distance_matrix(results)
        assert mat.shape == (4, 4)

    def test_dtype_float32(self):
        results = [_combine_result(seed=i) for i in range(3)]
        mat = build_distance_matrix(results)
        assert mat.dtype == np.float32

    def test_diagonal_zero(self):
        results = [_combine_result(seed=i) for i in range(4)]
        mat = build_distance_matrix(results)
        np.testing.assert_array_almost_equal(np.diag(mat), np.zeros(4))

    def test_symmetric(self):
        results = [_combine_result(seed=i) for i in range(4)]
        mat = build_distance_matrix(results)
        np.testing.assert_array_almost_equal(mat, mat.T)


# ─── TestFindNearest ──────────────────────────────────────────────────────────

class TestFindNearest:
    def test_top_k_less_than_1_raises(self):
        q = _combine_result()
        with pytest.raises(ValueError, match="top_k"):
            find_nearest(q, [_combine_result()], top_k=0)

    def test_empty_candidates_raises(self):
        q = _combine_result()
        with pytest.raises(ValueError, match="candidates"):
            find_nearest(q, [])

    def test_returns_list_of_tuples(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(5)]
        result = find_nearest(q, cands, top_k=3)
        assert len(result) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_sorted_ascending(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(5)]
        result = find_nearest(q, cands, top_k=5)
        dists = [d for _, d in result]
        assert dists == sorted(dists)

    def test_top_k_larger_than_candidates(self):
        q = _combine_result()
        cands = [_combine_result(seed=i) for i in range(3)]
        result = find_nearest(q, cands, top_k=10)
        assert len(result) == 3  # returns all available
