"""Extra tests for puzzle_reconstruction/utils/edge_profiler.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.edge_profiler import (
    ProfileConfig,
    EdgeProfile,
    compute_brightness_profile,
    compute_gradient_profile,
    compute_diff_profile,
    normalize_profile,
    aggregate_profiles,
    compare_profiles,
    batch_profile_edges,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _strip(h=4, w=32, val=128) -> np.ndarray:
    return np.full((h, w), float(val))


def _ramp_strip(h=4, w=32) -> np.ndarray:
    row = np.linspace(0, 255, w)
    return np.tile(row, (h, 1))


def _profile(n=32, val=0.5) -> np.ndarray:
    return np.full(n, val)


# ─── ProfileConfig ────────────────────────────────────────────────────────────

class TestProfileConfigExtra:
    def test_default_n_samples(self):
        assert ProfileConfig().n_samples == 32

    def test_default_profile_type(self):
        assert ProfileConfig().profile_type == "brightness"

    def test_default_normalize(self):
        assert ProfileConfig().normalize is True

    def test_default_strip_width(self):
        assert ProfileConfig().strip_width == 4

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            ProfileConfig(n_samples=1)

    def test_invalid_profile_type_raises(self):
        with pytest.raises(ValueError):
            ProfileConfig(profile_type="unknown")

    def test_strip_width_lt_1_raises(self):
        with pytest.raises(ValueError):
            ProfileConfig(strip_width=0)

    def test_valid_profile_types(self):
        for t in ("brightness", "gradient", "diff"):
            cfg = ProfileConfig(profile_type=t)
            assert cfg.profile_type == t


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

class TestEdgeProfileExtra:
    def test_stores_profile(self):
        p = np.zeros(16)
        ep = EdgeProfile(profile=p, edge_id=0, profile_type="brightness",
                         n_samples=16)
        np.testing.assert_array_equal(ep.profile, p)

    def test_stores_edge_id(self):
        ep = EdgeProfile(profile=np.zeros(8), edge_id=3, profile_type="diff",
                         n_samples=8)
        assert ep.edge_id == 3

    def test_stores_profile_type(self):
        ep = EdgeProfile(profile=np.zeros(8), edge_id=0, profile_type="gradient",
                         n_samples=8)
        assert ep.profile_type == "gradient"

    def test_negative_edge_id_raises(self):
        with pytest.raises(ValueError):
            EdgeProfile(profile=np.zeros(8), edge_id=-1, profile_type="brightness",
                        n_samples=8)

    def test_profile_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            EdgeProfile(profile=np.zeros(5), edge_id=0, profile_type="brightness",
                        n_samples=10)

    def test_default_params_empty(self):
        ep = EdgeProfile(profile=np.zeros(8), edge_id=0, profile_type="diff",
                         n_samples=8)
        assert isinstance(ep.params, dict)

    def test_custom_params_stored(self):
        ep = EdgeProfile(profile=np.zeros(8), edge_id=0, profile_type="diff",
                         n_samples=8, params={"strip_width": 4})
        assert ep.params["strip_width"] == 4


# ─── compute_brightness_profile ───────────────────────────────────────────────

class TestComputeBrightnessProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_brightness_profile(_strip()), np.ndarray)

    def test_length_equals_n_samples(self):
        p = compute_brightness_profile(_strip(), n_samples=16)
        assert len(p) == 16

    def test_dtype_float64(self):
        p = compute_brightness_profile(_strip())
        assert p.dtype == np.float64

    def test_uniform_strip_constant_profile(self):
        p = compute_brightness_profile(_strip(val=100), n_samples=8)
        assert np.allclose(p, 100.0)

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            compute_brightness_profile(_strip(), n_samples=1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_brightness_profile(np.zeros(10), n_samples=4)

    def test_ramp_increasing(self):
        # axis=0 collapses rows → profile varies along column direction
        p = compute_brightness_profile(_ramp_strip(), n_samples=8, axis=0)
        assert p[-1] > p[0]


# ─── compute_gradient_profile ─────────────────────────────────────────────────

class TestComputeGradientProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_gradient_profile(_strip()), np.ndarray)

    def test_length_equals_n_samples(self):
        p = compute_gradient_profile(_strip(), n_samples=16)
        assert len(p) == 16

    def test_dtype_float64(self):
        assert compute_gradient_profile(_strip()).dtype == np.float64

    def test_uniform_strip_near_zero(self):
        p = compute_gradient_profile(_strip(val=128), n_samples=8)
        assert np.all(p >= 0.0)

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_profile(_strip(), n_samples=1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_profile(np.zeros(10), n_samples=4)

    def test_nonneg_values(self):
        p = compute_gradient_profile(_ramp_strip())
        assert np.all(p >= 0.0)


# ─── compute_diff_profile ─────────────────────────────────────────────────────

class TestComputeDiffProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_diff_profile(_strip()), np.ndarray)

    def test_length_equals_n_samples(self):
        p = compute_diff_profile(_strip(), n_samples=16)
        assert len(p) == 16

    def test_dtype_float64(self):
        assert compute_diff_profile(_strip()).dtype == np.float64

    def test_uniform_strip_zero_diff(self):
        p = compute_diff_profile(_strip(val=128), n_samples=8)
        assert np.allclose(p, 0.0)

    def test_n_samples_lt_2_raises(self):
        with pytest.raises(ValueError):
            compute_diff_profile(_strip(), n_samples=1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_diff_profile(np.zeros(10), n_samples=4)

    def test_nonneg_values(self):
        p = compute_diff_profile(_ramp_strip())
        assert np.all(p >= 0.0)


# ─── normalize_profile ────────────────────────────────────────────────────────

class TestNormalizeProfileExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_profile(np.array([1.0, 2.0, 3.0])), np.ndarray)

    def test_min_is_zero(self):
        p = normalize_profile(np.array([2.0, 5.0, 10.0]))
        assert p.min() == pytest.approx(0.0)

    def test_max_is_one(self):
        p = normalize_profile(np.array([2.0, 5.0, 10.0]))
        assert p.max() == pytest.approx(1.0)

    def test_constant_returns_zeros(self):
        p = normalize_profile(np.full(5, 7.0))
        assert np.allclose(p, 0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            normalize_profile(np.zeros((2, 3)))

    def test_empty_array_ok(self):
        p = normalize_profile(np.array([]))
        assert len(p) == 0

    def test_dtype_float64(self):
        p = normalize_profile(np.array([1.0, 2.0]))
        assert p.dtype == np.float64


# ─── aggregate_profiles ───────────────────────────────────────────────────────

class TestAggregateProfilesExtra:
    def test_returns_ndarray(self):
        ps = [np.ones(8), np.zeros(8)]
        assert isinstance(aggregate_profiles(ps), np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_profiles([])

    def test_equal_weights_average(self):
        ps = [np.array([1.0, 1.0]), np.array([3.0, 3.0])]
        out = aggregate_profiles(ps)
        np.testing.assert_allclose(out, [2.0, 2.0])

    def test_custom_weights(self):
        ps = [np.array([1.0]), np.array([3.0])]
        out = aggregate_profiles(ps, weights=[0.25, 0.75])
        assert out[0] == pytest.approx(2.5)

    def test_zero_weight_sum_raises(self):
        ps = [np.ones(4)]
        with pytest.raises(ValueError):
            aggregate_profiles(ps, weights=[0.0])

    def test_length_mismatch_raises(self):
        ps = [np.ones(4), np.ones(8)]
        with pytest.raises(ValueError):
            aggregate_profiles(ps)

    def test_single_profile_returned_unchanged(self):
        p = np.array([1.0, 2.0, 3.0])
        out = aggregate_profiles([p])
        np.testing.assert_allclose(out, p)


# ─── compare_profiles ─────────────────────────────────────────────────────────

class TestCompareProfilesExtra:
    def test_returns_float(self):
        p = np.ones(8)
        assert isinstance(compare_profiles(p, p), float)

    def test_identical_profiles_one(self):
        p = np.array([1.0, 2.0, 3.0])
        assert compare_profiles(p, p) == pytest.approx(1.0)

    def test_zero_profiles_one(self):
        p = np.zeros(8)
        result = compare_profiles(p, p)
        assert result == pytest.approx(1.0)

    def test_result_in_range(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        result = compare_profiles(p1, p2)
        assert 0.0 <= result <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compare_profiles(np.ones(4), np.ones(8))


# ─── batch_profile_edges ──────────────────────────────────────────────────────

class TestBatchProfileEdgesExtra:
    def test_returns_list(self):
        result = batch_profile_edges([_strip()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_profile_edges([_strip(), _ramp_strip()])
        assert len(result) == 2

    def test_each_element_is_edge_profile(self):
        for ep in batch_profile_edges([_strip()]):
            assert isinstance(ep, EdgeProfile)

    def test_edge_ids_assigned_sequentially(self):
        result = batch_profile_edges([_strip(), _strip()])
        assert result[0].edge_id == 0
        assert result[1].edge_id == 1

    def test_custom_edge_ids(self):
        result = batch_profile_edges([_strip()], edge_ids=[42])
        assert result[0].edge_id == 42

    def test_none_cfg_uses_defaults(self):
        result = batch_profile_edges([_strip()], cfg=None)
        assert result[0].profile_type == "brightness"

    def test_gradient_type(self):
        cfg = ProfileConfig(profile_type="gradient")
        result = batch_profile_edges([_strip()], cfg=cfg)
        assert result[0].profile_type == "gradient"

    def test_diff_type(self):
        cfg = ProfileConfig(profile_type="diff")
        result = batch_profile_edges([_strip()], cfg=cfg)
        assert result[0].profile_type == "diff"

    def test_profile_length_matches_n_samples(self):
        cfg = ProfileConfig(n_samples=16)
        result = batch_profile_edges([_strip()], cfg=cfg)
        assert result[0].n_samples == 16
        assert len(result[0].profile) == 16

    def test_normalize_applied(self):
        cfg = ProfileConfig(normalize=True)
        result = batch_profile_edges([_ramp_strip()], cfg=cfg)
        assert result[0].profile.max() <= 1.0 + 1e-8

    def test_empty_list_returns_empty(self):
        assert batch_profile_edges([]) == []
