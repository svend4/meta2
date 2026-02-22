"""Extra tests for puzzle_reconstruction.algorithms.descriptor_combiner."""
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


def _vec(size=8, seed=0):
    return np.random.default_rng(seed).random(size).astype(np.float32)


def _desc_set(fid=0, names=("shape", "texture"), sizes=(8, 16)):
    descs = {n: _vec(s, i) for i, (n, s) in enumerate(zip(names, sizes))}
    return DescriptorSet(fragment_id=fid, descriptors=descs)


def _combine_result(fid=0, dim=8, orig_dim=16, seed=0):
    v = _vec(dim, seed)
    return CombineResult(fragment_id=fid, vector=v,
                         used_names=["a"], original_dim=orig_dim)


# ─── TestCombineConfigExtra ─────────────────────────────────────────────────

class TestCombineConfigExtra:
    def test_default_normalize(self):
        assert CombineConfig().normalize is True

    def test_default_l2_final(self):
        assert CombineConfig().l2_final is True

    def test_default_pca_dim_none(self):
        assert CombineConfig().pca_dim is None

    def test_default_weights_empty(self):
        assert CombineConfig().weights == {}

    def test_pca_dim_positive(self):
        cfg = CombineConfig(pca_dim=4)
        assert cfg.pca_dim == 4

    def test_weight_for_known(self):
        cfg = CombineConfig(weights={"a": 3.0})
        assert cfg.weight_for("a") == pytest.approx(3.0)

    def test_weight_for_unknown(self):
        assert CombineConfig().weight_for("x") == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            CombineConfig(weights={"a": -0.1})

    def test_pca_dim_zero_raises(self):
        with pytest.raises(ValueError):
            CombineConfig(pca_dim=0)

    def test_zero_weight_ok(self):
        cfg = CombineConfig(weights={"a": 0.0})
        assert cfg.weight_for("a") == pytest.approx(0.0)


# ─── TestDescriptorSetExtra ─────────────────────────────────────────────────

class TestDescriptorSetExtra:
    def test_fragment_id(self):
        ds = _desc_set(fid=7)
        assert ds.fragment_id == 7

    def test_names(self):
        ds = _desc_set(names=("a", "b", "c"), sizes=(4, 4, 4))
        assert set(ds.names) == {"a", "b", "c"}

    def test_total_dim(self):
        ds = _desc_set(names=("x", "y"), sizes=(5, 10))
        assert ds.total_dim == 15

    def test_has_true(self):
        ds = _desc_set(names=("shape",))
        assert ds.has("shape") is True

    def test_has_false(self):
        ds = _desc_set(names=("shape",))
        assert ds.has("color") is False

    def test_get_existing(self):
        ds = _desc_set(names=("shape",), sizes=(8,))
        v = ds.get("shape")
        assert v is not None and v.ndim == 1

    def test_get_missing_none(self):
        ds = _desc_set(names=("shape",))
        assert ds.get("missing") is None

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            DescriptorSet(fragment_id=0, descriptors={"bad": np.zeros((2, 3))})

    def test_empty_descriptors(self):
        ds = DescriptorSet(fragment_id=0)
        assert ds.total_dim == 0
        assert ds.descriptors == {}


# ─── TestCombineResultExtra ─────────────────────────────────────────────────

class TestCombineResultExtra:
    def test_dim_property(self):
        r = _combine_result(dim=12, orig_dim=12)
        assert r.dim == 12

    def test_is_reduced_true(self):
        r = _combine_result(dim=4, orig_dim=8)
        assert r.is_reduced is True

    def test_is_reduced_false(self):
        r = _combine_result(dim=8, orig_dim=8)
        assert r.is_reduced is False

    def test_norm_nonneg(self):
        assert _combine_result().norm >= 0.0

    def test_fragment_id_stored(self):
        r = _combine_result(fid=42)
        assert r.fragment_id == 42

    def test_non_1d_vector_raises(self):
        with pytest.raises(ValueError):
            CombineResult(fragment_id=0, vector=np.zeros((2, 3)),
                          used_names=[], original_dim=6)

    def test_negative_original_dim_raises(self):
        with pytest.raises(ValueError):
            CombineResult(fragment_id=0, vector=np.zeros(4),
                          used_names=[], original_dim=-1)

    def test_used_names_list(self):
        r = _combine_result()
        assert isinstance(r.used_names, list)


# ─── TestCombineDescriptorsExtra ────────────────────────────────────────────

class TestCombineDescriptorsExtra:
    def test_returns_combine_result(self):
        assert isinstance(combine_descriptors(_desc_set()), CombineResult)

    def test_fragment_id_preserved(self):
        assert combine_descriptors(_desc_set(fid=5)).fragment_id == 5

    def test_vector_1d(self):
        assert combine_descriptors(_desc_set()).vector.ndim == 1

    def test_l2_final_unit_norm(self):
        cfg = CombineConfig(l2_final=True)
        r = combine_descriptors(_desc_set(), cfg)
        assert r.norm == pytest.approx(1.0, abs=1e-5)

    def test_no_l2_not_unit(self):
        cfg = CombineConfig(l2_final=False, normalize=False, weights={"shape": 10.0})
        ds = _desc_set(names=("shape",), sizes=(8,))
        r = combine_descriptors(ds, cfg)
        assert r.norm > 1.0

    def test_empty_desc_raises(self):
        with pytest.raises(ValueError):
            combine_descriptors(DescriptorSet(fragment_id=0))

    def test_used_names_match(self):
        ds = _desc_set(names=("a", "b"))
        r = combine_descriptors(ds)
        assert set(r.used_names) == {"a", "b"}

    def test_original_dim_sum(self):
        ds = _desc_set(names=("a", "b"), sizes=(5, 7))
        cfg = CombineConfig(l2_final=False)
        r = combine_descriptors(ds, cfg)
        assert r.original_dim == 12


# ─── TestCombineSelectedExtra ───────────────────────────────────────────────

class TestCombineSelectedExtra:
    def test_selects_subset(self):
        ds = _desc_set(names=("a", "b"), sizes=(4, 8))
        r = combine_selected(ds, ["a"])
        assert "b" not in r.used_names

    def test_returns_combine_result(self):
        ds = _desc_set()
        assert isinstance(combine_selected(ds, list(ds.names)), CombineResult)

    def test_empty_names_raises(self):
        with pytest.raises(ValueError):
            combine_selected(_desc_set(), [])

    def test_no_found_raises(self):
        with pytest.raises(ValueError):
            combine_selected(_desc_set(names=("a",)), ["nonexistent"])

    def test_fragment_id_preserved(self):
        ds = _desc_set(fid=10)
        assert combine_selected(ds, list(ds.names)).fragment_id == 10


# ─── TestBatchCombineExtra ──────────────────────────────────────────────────

class TestBatchCombineExtra:
    def test_returns_list(self):
        sets = [_desc_set(fid=i) for i in range(3)]
        assert isinstance(batch_combine(sets), list)

    def test_length(self):
        sets = [_desc_set(fid=i) for i in range(5)]
        assert len(batch_combine(sets)) == 5

    def test_empty(self):
        assert batch_combine([]) == []

    def test_ids_preserved(self):
        sets = [_desc_set(fid=i) for i in range(4)]
        ids = [r.fragment_id for r in batch_combine(sets)]
        assert ids == [0, 1, 2, 3]

    def test_all_combine_results(self):
        sets = [_desc_set(fid=i) for i in range(3)]
        assert all(isinstance(r, CombineResult) for r in batch_combine(sets))


# ─── TestDescriptorDistanceExtra ────────────────────────────────────────────

class TestDescriptorDistanceExtra:
    def test_cosine_identical_zero(self):
        r = _combine_result()
        assert descriptor_distance(r, r, "cosine") == pytest.approx(0.0, abs=1e-5)

    def test_euclidean_identical_zero(self):
        r = _combine_result()
        assert descriptor_distance(r, r, "euclidean") == pytest.approx(0.0, abs=1e-5)

    def test_l1_identical_zero(self):
        r = _combine_result()
        assert descriptor_distance(r, r, "l1") == pytest.approx(0.0, abs=1e-5)

    def test_nonneg(self):
        r1 = _combine_result(seed=0)
        r2 = _combine_result(seed=1)
        for m in ("cosine", "euclidean", "l1"):
            assert descriptor_distance(r1, r2, m) >= 0.0

    def test_unknown_metric_raises(self):
        r = _combine_result()
        with pytest.raises(ValueError):
            descriptor_distance(r, r, "invalid")

    def test_different_dims_handled(self):
        r1 = _combine_result(dim=4, orig_dim=4)
        r2 = _combine_result(dim=8, orig_dim=8, seed=1)
        assert descriptor_distance(r1, r2, "euclidean") >= 0.0

    def test_symmetric(self):
        r1 = _combine_result(seed=0)
        r2 = _combine_result(seed=1)
        for m in ("cosine", "euclidean", "l1"):
            assert descriptor_distance(r1, r2, m) == pytest.approx(
                descriptor_distance(r2, r1, m), abs=1e-5)


# ─── TestBuildDistanceMatrixExtra ───────────────────────────────────────────

class TestBuildDistanceMatrixExtra:
    def test_shape(self):
        results = [_combine_result(seed=i) for i in range(5)]
        assert build_distance_matrix(results).shape == (5, 5)

    def test_diagonal_zero(self):
        results = [_combine_result(seed=i) for i in range(3)]
        mat = build_distance_matrix(results)
        np.testing.assert_array_almost_equal(np.diag(mat), np.zeros(3))

    def test_symmetric(self):
        results = [_combine_result(seed=i) for i in range(4)]
        mat = build_distance_matrix(results)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_dtype_float32(self):
        results = [_combine_result(seed=i) for i in range(3)]
        assert build_distance_matrix(results).dtype == np.float32

    def test_nonneg(self):
        results = [_combine_result(seed=i) for i in range(3)]
        assert np.all(build_distance_matrix(results) >= -1e-6)


# ─── TestFindNearestExtra ───────────────────────────────────────────────────

class TestFindNearestExtra:
    def test_returns_list_of_tuples(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(5)]
        result = find_nearest(q, cands, top_k=3)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_sorted_ascending(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(5)]
        dists = [d for _, d in find_nearest(q, cands, top_k=5)]
        assert dists == sorted(dists)

    def test_top_k_respected(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(10)]
        assert len(find_nearest(q, cands, top_k=3)) == 3

    def test_top_k_more_than_cands(self):
        q = _combine_result(seed=0)
        cands = [_combine_result(seed=i) for i in range(2)]
        assert len(find_nearest(q, cands, top_k=10)) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_combine_result(), [])

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            find_nearest(_combine_result(), [_combine_result()], top_k=0)

    def test_self_distance_zero(self):
        q = _combine_result(seed=0)
        result = find_nearest(q, [q], top_k=1)
        assert result[0][1] == pytest.approx(0.0, abs=1e-5)
