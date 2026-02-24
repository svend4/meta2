"""Extra tests for puzzle_reconstruction/utils/gap_geometry_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.gap_geometry_records import (
    GapScoringRecord,
    GeometricMatchRecord,
    GeometryRecord,
    GlobalMatchRecord,
    make_gap_scoring_record,
    make_global_match_record,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gap(a=0, b=1, score=0.6) -> GapScoringRecord:
    return GapScoringRecord(id_a=a, id_b=b, distance=1.0, score=score, penalty=0.1)


def _geom_match(s=0.8) -> GeometricMatchRecord:
    return GeometricMatchRecord(idx1=0, idx2=1, score=s,
                                 aspect_score=0.7, area_score=0.8, hu_score=0.9)


# ─── GapScoringRecord ─────────────────────────────────────────────────────────

class TestGapScoringRecordExtra:
    def test_stores_ids(self):
        r = _gap(a=3, b=7)
        assert r.id_a == 3 and r.id_b == 7

    def test_stores_score(self):
        assert _gap(score=0.75).score == pytest.approx(0.75)

    def test_pair_key_ordered(self):
        r = _gap(a=5, b=2)
        assert r.pair_key == (2, 5)

    def test_pair_key_same_order(self):
        r = _gap(a=1, b=3)
        assert r.pair_key == (1, 3)

    def test_is_acceptable_high_score(self):
        r = _gap(score=0.8)
        assert r.is_acceptable is True

    def test_is_acceptable_low_score(self):
        r = _gap(score=0.3)
        assert r.is_acceptable is False


# ─── GeometricMatchRecord ─────────────────────────────────────────────────────

class TestGeometricMatchRecordExtra:
    def test_stores_score(self):
        assert _geom_match(s=0.85).score == pytest.approx(0.85)

    def test_is_good_match_true(self):
        assert _geom_match(s=0.9).is_good_match is True

    def test_is_good_match_false(self):
        assert _geom_match(s=0.5).is_good_match is False

    def test_stores_method(self):
        r = GeometricMatchRecord(idx1=0, idx2=1, score=0.7,
                                  aspect_score=0.6, area_score=0.7, hu_score=0.8,
                                  method="custom")
        assert r.method == "custom"


# ─── GeometryRecord ───────────────────────────────────────────────────────────

class TestGeometryRecordExtra:
    def test_stores_fragment_id(self):
        r = GeometryRecord(fragment_id=5, area=100.0, perimeter=40.0,
                            aspect_ratio=1.0, solidity=0.9)
        assert r.fragment_id == 5

    def test_is_convex_true(self):
        r = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                            aspect_ratio=1.0, solidity=0.98)
        assert r.is_convex is True

    def test_is_convex_false(self):
        r = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                            aspect_ratio=1.0, solidity=0.5)
        assert r.is_convex is False

    def test_is_elongated_true(self):
        r = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                            aspect_ratio=3.0, solidity=0.8)
        assert r.is_elongated is True

    def test_is_elongated_false(self):
        r = GeometryRecord(fragment_id=0, area=100.0, perimeter=40.0,
                            aspect_ratio=1.2, solidity=0.8)
        assert r.is_elongated is False


# ─── GlobalMatchRecord ────────────────────────────────────────────────────────

class TestGlobalMatchRecordExtra:
    def test_stores_fragment_id(self):
        r = GlobalMatchRecord(fragment_id=3, candidate_id=7, score=0.9, rank=1)
        assert r.fragment_id == 3

    def test_is_top_true(self):
        r = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.9, rank=1)
        assert r.is_top is True

    def test_is_top_false(self):
        r = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.9, rank=2)
        assert r.is_top is False

    def test_is_strong_true(self):
        r = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.85, rank=1)
        assert r.is_strong is True

    def test_is_strong_false(self):
        r = GlobalMatchRecord(fragment_id=0, candidate_id=1, score=0.5, rank=1)
        assert r.is_strong is False


# ─── make_gap_scoring_record ──────────────────────────────────────────────────

class TestMakeGapScoringRecordExtra:
    def test_returns_record(self):
        r = make_gap_scoring_record(0, 1, 2.0, 0.7, 0.1)
        assert isinstance(r, GapScoringRecord)

    def test_acceptable_set(self):
        r = make_gap_scoring_record(0, 1, 1.0, 0.8, 0.0)
        assert r.acceptable is True

    def test_not_acceptable_low_score(self):
        r = make_gap_scoring_record(0, 1, 1.0, 0.3, 0.0)
        assert r.acceptable is False


# ─── make_global_match_record ─────────────────────────────────────────────────

class TestMakeGlobalMatchRecordExtra:
    def test_returns_record(self):
        r = make_global_match_record(0, 1, 0.8, 1)
        assert isinstance(r, GlobalMatchRecord)

    def test_values_stored(self):
        r = make_global_match_record(3, 5, 0.75, 2, n_channels=3)
        assert r.fragment_id == 3 and r.score == pytest.approx(0.75)
        assert r.n_channels == 3
