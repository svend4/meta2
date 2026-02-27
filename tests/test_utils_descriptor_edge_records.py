"""Tests for puzzle_reconstruction.utils.descriptor_edge_records."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.descriptor_edge_records import (
    DescriptorCombineRecord,
    EdgeProfileRecord,
    ProfileMatchRecord,
    EdgeValidRecord,
    FeatureMatchRecord,
    make_profile_match_record,
    make_feature_match_record,
)

np.random.seed(55)


# ── 1. DescriptorCombineRecord basic ─────────────────────────────────────────
def test_descriptor_combine_basic():
    r = DescriptorCombineRecord(
        fragment_id=1, used_names=["hog", "lbp"], original_dim=512,
        combined_dim=256, normalized=True, l2_final=True,
    )
    assert r.fragment_id == 1
    assert len(r.used_names) == 2
    assert r.is_reduced is True
    assert abs(r.compression_ratio - 0.5) < 1e-9


# ── 2. DescriptorCombineRecord not reduced ───────────────────────────────────
def test_descriptor_combine_not_reduced():
    r = DescriptorCombineRecord(1, ["sift"], 128, 256)
    assert r.is_reduced is False
    assert abs(r.compression_ratio - 2.0) < 1e-9


# ── 3. DescriptorCombineRecord zero original_dim ─────────────────────────────
def test_descriptor_combine_zero_dim():
    r = DescriptorCombineRecord(1, [], 0, 0)
    assert r.compression_ratio == 0.0


# ── 4. EdgeProfileRecord basic ───────────────────────────────────────────────
def test_edge_profile_basic():
    r = EdgeProfileRecord(
        fragment_id=2, side=0, method="linear",
        n_samples=64, signal_mean=128.0, signal_std=10.0,
    )
    assert r.fragment_id == 2
    assert r.is_uniform is False


def test_edge_profile_uniform():
    r = EdgeProfileRecord(3, 1, "linear", 64, 100.0, 0.0)
    assert r.is_uniform is True


# ── 6. ProfileMatchRecord basic ──────────────────────────────────────────────
def test_profile_match_basic():
    r = ProfileMatchRecord(0, 1, 0, 1, 0.75, 0.8, 0.9)
    assert r.idx1 == 0
    assert r.idx2 == 1
    assert r.score == 0.75
    assert r.pair_key == (0, 1)
    assert r.is_good_match is True


def test_profile_match_pair_key_order():
    r = ProfileMatchRecord(5, 2, 0, 1, 0.5, 0.5, 0.5)
    assert r.pair_key == (2, 5)


def test_profile_match_not_good():
    r = ProfileMatchRecord(0, 1, 0, 1, 0.4, 0.3, 0.5)
    assert r.is_good_match is False


# ── 9. make_profile_match_record ─────────────────────────────────────────────
def test_make_profile_match_record():
    r = make_profile_match_record(1, 2, 0, 1, 0.8, 0.7, 0.9, n_samples=32)
    assert r.idx1 == 1
    assert r.idx2 == 2
    assert r.side1 == 0
    assert r.side2 == 1
    assert r.n_samples == 32


# ── 10. EdgeValidRecord basic ────────────────────────────────────────────────
def test_edge_valid_basic():
    r = EdgeValidRecord(0, 1, True, 3, 1, intensity_value=0.8,
                        gap_value=0.2, normal_value=0.9)
    assert r.valid is True
    assert abs(r.pass_rate - 0.75) < 1e-9


def test_edge_valid_zero_total():
    r = EdgeValidRecord(0, 1, False, 0, 0)
    assert r.pass_rate == 0.0


# ── 12. FeatureMatchRecord basic ─────────────────────────────────────────────
def test_feature_match_basic():
    r = FeatureMatchRecord(0, 1, "sift", 0.7, n_matches=100, n_inliers=20)
    assert r.idx1 == 0
    assert r.idx2 == 1
    assert abs(r.inlier_ratio - 0.2) < 1e-9
    assert r.is_good_match is True


def test_feature_match_no_matches():
    r = FeatureMatchRecord(0, 1, "orb", 0.3, n_matches=0, n_inliers=0)
    assert r.inlier_ratio == 0.0


def test_feature_match_not_good_score():
    r = FeatureMatchRecord(0, 1, "sift", 0.3, n_matches=10, n_inliers=5)
    assert r.is_good_match is False


def test_feature_match_not_good_inliers():
    r = FeatureMatchRecord(0, 1, "sift", 0.7, n_matches=10, n_inliers=2)
    assert r.is_good_match is False


# ── 16. make_feature_match_record ────────────────────────────────────────────
def test_make_feature_match_record():
    r = make_feature_match_record(3, 5, "orb", 0.6, 50, 10, (200, 180))
    assert r.idx1 == 3
    assert r.idx2 == 5
    assert r.method == "orb"
    assert r.n_keypoints_1 == 200
    assert r.n_keypoints_2 == 180


# ── 17. DescriptorCombineRecord with no names ─────────────────────────────────
def test_descriptor_combine_empty_names():
    r = DescriptorCombineRecord(0, [], 128, 64)
    assert r.used_names == []
    assert r.is_reduced is True
