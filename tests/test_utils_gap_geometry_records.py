"""Tests for puzzle_reconstruction.utils.gap_geometry_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.gap_geometry_records import (
    GapScoringRecord,
    GeometricMatchRecord,
    GeometryRecord,
    GlobalMatchRecord,
    make_gap_scoring_record,
    make_global_match_record,
)

np.random.seed(42)


# ── GapScoringRecord ──────────────────────────────────────────────────────────

def test_gap_scoring_pair_key_sorted():
    rec = GapScoringRecord(id_a=5, id_b=2, distance=1.0, score=0.8, penalty=0.1)
    assert rec.pair_key == (2, 5)


def test_gap_scoring_pair_key_already_sorted():
    rec = GapScoringRecord(id_a=1, id_b=3, distance=2.0, score=0.6, penalty=0.0)
    assert rec.pair_key == (1, 3)


def test_gap_scoring_is_acceptable_true():
    rec = GapScoringRecord(id_a=0, id_b=1, distance=0.5, score=0.9, penalty=0.0)
    assert rec.is_acceptable is True


def test_gap_scoring_is_acceptable_false():
    rec = GapScoringRecord(id_a=0, id_b=1, distance=0.5, score=0.3, penalty=0.0)
    assert rec.is_acceptable is False


def test_gap_scoring_acceptable_flag_boundary():
    rec = GapScoringRecord(id_a=0, id_b=1, distance=0.0, score=0.5, penalty=0.0, acceptable=True)
    assert rec.acceptable is True


def test_make_gap_scoring_record_acceptable_above_threshold():
    rec = make_gap_scoring_record(0, 1, 1.5, 0.7, 0.1)
    assert rec.acceptable is True
    assert rec.id_a == 0
    assert rec.id_b == 1
    assert rec.distance == pytest.approx(1.5)


def test_make_gap_scoring_record_not_acceptable():
    rec = make_gap_scoring_record(2, 3, 0.5, 0.4, 0.2)
    assert rec.acceptable is False


# ── GeometricMatchRecord ──────────────────────────────────────────────────────

def test_geometric_match_is_good_match_true():
    rec = GeometricMatchRecord(idx1=0, idx2=1, score=0.8,
                               aspect_score=0.9, area_score=0.8,
                               hu_score=0.7)
    assert rec.is_good_match is True


def test_geometric_match_is_good_match_false():
    rec = GeometricMatchRecord(idx1=0, idx2=1, score=0.5,
                               aspect_score=0.5, area_score=0.5,
                               hu_score=0.5)
    assert rec.is_good_match is False


def test_geometric_match_default_method():
    rec = GeometricMatchRecord(idx1=0, idx2=1, score=0.9,
                               aspect_score=1.0, area_score=1.0,
                               hu_score=1.0)
    assert rec.method == "geometric"


def test_geometric_match_custom_method():
    rec = GeometricMatchRecord(idx1=0, idx2=1, score=0.9,
                               aspect_score=1.0, area_score=1.0,
                               hu_score=1.0, method="custom")
    assert rec.method == "custom"


# ── GeometryRecord ────────────────────────────────────────────────────────────

def test_geometry_record_is_convex_true():
    rec = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                         aspect_ratio=1.0, solidity=0.97)
    assert rec.is_convex is True


def test_geometry_record_is_convex_false():
    rec = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                         aspect_ratio=1.0, solidity=0.8)
    assert rec.is_convex is False


def test_geometry_record_is_elongated_true():
    rec = GeometryRecord(fragment_id=0, area=200.0, perimeter=80.0,
                         aspect_ratio=3.0, solidity=0.9)
    assert rec.is_elongated is True


def test_geometry_record_is_elongated_false():
    rec = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                         aspect_ratio=1.5, solidity=0.9)
    assert rec.is_elongated is False


def test_geometry_record_n_contours_default():
    rec = GeometryRecord(fragment_id=1, area=50.0, perimeter=30.0,
                         aspect_ratio=1.2, solidity=0.88)
    assert rec.n_contours == 0


# ── GlobalMatchRecord ─────────────────────────────────────────────────────────

def test_global_match_is_top_true():
    rec = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.9, rank=1)
    assert rec.is_top is True


def test_global_match_is_top_false():
    rec = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.9, rank=2)
    assert rec.is_top is False


def test_global_match_is_strong_true():
    rec = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.85, rank=1)
    assert rec.is_strong is True


def test_global_match_is_strong_false():
    rec = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.7, rank=1)
    assert rec.is_strong is False


def test_make_global_match_record():
    rec = make_global_match_record(3, 7, 0.95, 1, n_channels=3)
    assert rec.fragment_id == 3
    assert rec.candidate_id == 7
    assert rec.score == pytest.approx(0.95)
    assert rec.rank == 1
    assert rec.n_channels == 3


def test_make_global_match_record_default_channels():
    rec = make_global_match_record(0, 1, 0.5, 2)
    assert rec.n_channels == 0
