"""Tests for puzzle_reconstruction.utils.orient_topology_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.orient_topology_utils import (
    OrientMatchRecord,
    OrientMatchSummary,
    TopologyRecord,
    TopologySummary,
    summarize_orient_matches,
    filter_orient_records,
    topology_records_from_dicts,
    summarize_topology,
    rank_orient_matches,
    top_k_orient_matches,
)

np.random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orient_records():
    return [
        OrientMatchRecord(fragment_a=0, fragment_b=1, best_score=0.9,
                          best_angle=30.0, is_flipped=False),
        OrientMatchRecord(fragment_a=1, fragment_b=2, best_score=0.5,
                          best_angle=60.0, is_flipped=True),
        OrientMatchRecord(fragment_a=2, fragment_b=3, best_score=0.3,
                          best_angle=90.0, is_flipped=False),
    ]


def _topo_records():
    return [
        TopologyRecord(solidity=0.95, extent=0.8, convexity=0.92,
                       compactness=0.91, complexity=0.1),
        TopologyRecord(solidity=0.7, extent=0.6, convexity=0.7,
                       compactness=0.7, complexity=0.3),
        TopologyRecord(solidity=0.5, extent=0.5, convexity=0.5,
                       compactness=0.5, complexity=0.5),
    ]


# ── OrientMatchRecord ─────────────────────────────────────────────────────────

def test_orient_match_record_pair():
    r = OrientMatchRecord(fragment_a=2, fragment_b=5,
                          best_score=0.8, best_angle=45.0)
    assert r.pair == (2, 5)


def test_orient_match_record_is_good_match_true():
    r = OrientMatchRecord(fragment_a=0, fragment_b=1,
                          best_score=0.7, best_angle=0.0)
    assert r.is_good_match is True


def test_orient_match_record_is_good_match_false():
    r = OrientMatchRecord(fragment_a=0, fragment_b=1,
                          best_score=0.4, best_angle=0.0)
    assert r.is_good_match is False


def test_orient_match_record_invalid_fragment_a():
    with pytest.raises(ValueError):
        OrientMatchRecord(fragment_a=-1, fragment_b=0,
                          best_score=0.5, best_angle=0.0)


def test_orient_match_record_invalid_score():
    with pytest.raises(ValueError):
        OrientMatchRecord(fragment_a=0, fragment_b=1,
                          best_score=1.5, best_angle=0.0)


def test_orient_match_record_invalid_angle():
    with pytest.raises(ValueError):
        OrientMatchRecord(fragment_a=0, fragment_b=1,
                          best_score=0.5, best_angle=-10.0)


# ── OrientMatchSummary ────────────────────────────────────────────────────────

def test_orient_match_summary_good_ratio_zero():
    s = OrientMatchSummary(total_pairs=0)
    assert s.good_ratio == pytest.approx(0.0)


def test_orient_match_summary_flip_ratio():
    s = OrientMatchSummary(total_pairs=10, flipped_pairs=4)
    assert s.flip_ratio == pytest.approx(0.4)


def test_orient_match_summary_invalid_total():
    with pytest.raises(ValueError):
        OrientMatchSummary(total_pairs=-1)


# ── summarize_orient_matches ──────────────────────────────────────────────────

def test_summarize_orient_matches_empty():
    s = summarize_orient_matches([])
    assert s.total_pairs == 0
    assert s.good_pairs == 0


def test_summarize_orient_matches_normal():
    records = _orient_records()
    s = summarize_orient_matches(records)
    assert s.total_pairs == 3
    # scores 0.9 and 0.5 >= 0.5
    assert s.good_pairs == 2
    assert s.flipped_pairs == 1
    assert s.max_score == pytest.approx(0.9)
    assert s.min_score == pytest.approx(0.3)


def test_summarize_orient_matches_mean_score():
    records = _orient_records()
    s = summarize_orient_matches(records)
    assert s.mean_score == pytest.approx((0.9 + 0.5 + 0.3) / 3)


# ── filter_orient_records ─────────────────────────────────────────────────────

def test_filter_orient_records_by_score():
    records = _orient_records()
    filtered = filter_orient_records(records, min_score=0.5)
    assert all(r.best_score >= 0.5 for r in filtered)
    assert len(filtered) == 2


def test_filter_orient_records_exclude_flipped():
    records = _orient_records()
    filtered = filter_orient_records(records, exclude_flipped=True)
    assert all(not r.is_flipped for r in filtered)
    assert len(filtered) == 2


def test_filter_orient_records_combined():
    records = _orient_records()
    filtered = filter_orient_records(records, min_score=0.5, exclude_flipped=True)
    assert len(filtered) == 1
    assert filtered[0].best_score == pytest.approx(0.9)


# ── TopologyRecord ────────────────────────────────────────────────────────────

def test_topology_record_is_convex_true():
    r = TopologyRecord(convexity=0.95)
    assert r.is_convex is True


def test_topology_record_is_convex_false():
    r = TopologyRecord(convexity=0.8)
    assert r.is_convex is False


def test_topology_record_is_compact_true():
    r = TopologyRecord(compactness=0.95)
    assert r.is_compact is True


def test_topology_record_is_compact_false():
    r = TopologyRecord(compactness=0.85)
    assert r.is_compact is False


def test_topology_record_to_dict():
    r = TopologyRecord(solidity=0.9, extent=0.7, convexity=0.8,
                       compactness=0.85, complexity=0.2)
    d = r.to_dict()
    assert set(d.keys()) == {"solidity", "extent", "convexity", "compactness", "complexity"}
    assert d["solidity"] == pytest.approx(0.9)


def test_topology_record_negative_value_raises():
    with pytest.raises(ValueError):
        TopologyRecord(solidity=-0.1)


# ── topology_records_from_dicts ───────────────────────────────────────────────

def test_topology_records_from_dicts():
    dicts = [
        {"solidity": 0.9, "extent": 0.8, "convexity": 0.85,
         "compactness": 0.88, "complexity": 0.15},
        {"solidity": 0.7, "extent": 0.6},
    ]
    records = topology_records_from_dicts(dicts)
    assert len(records) == 2
    assert records[0].solidity == pytest.approx(0.9)
    assert records[1].extent == pytest.approx(0.6)
    assert records[1].convexity == pytest.approx(0.0)


# ── TopologySummary ───────────────────────────────────────────────────────────

def test_topology_summary_convex_ratio_zero():
    s = TopologySummary(n_contours=0)
    assert s.convex_ratio == pytest.approx(0.0)


def test_topology_summary_compact_ratio():
    s = TopologySummary(n_contours=5, n_compact=2)
    assert s.compact_ratio == pytest.approx(0.4)


# ── summarize_topology ────────────────────────────────────────────────────────

def test_summarize_topology_empty():
    s = summarize_topology([])
    assert s.n_contours == 0


def test_summarize_topology_normal():
    records = _topo_records()
    s = summarize_topology(records)
    assert s.n_contours == 3
    # convexity: 0.92, 0.7, 0.5 -> only first > 0.9
    assert s.n_convex == 1
    # compactness: 0.91, 0.7, 0.5 -> only first > 0.9
    assert s.n_compact == 1


def test_summarize_topology_mean_solidity():
    records = _topo_records()
    s = summarize_topology(records)
    assert s.mean_solidity == pytest.approx((0.95 + 0.7 + 0.5) / 3)


# ── rank / top_k ──────────────────────────────────────────────────────────────

def test_rank_orient_matches():
    records = _orient_records()
    ranked = rank_orient_matches(records)
    scores = [r.best_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_k_orient_matches():
    records = _orient_records()
    top = top_k_orient_matches(records, 2)
    assert len(top) == 2
    assert top[0].best_score >= top[1].best_score


def test_top_k_orient_matches_zero():
    records = _orient_records()
    top = top_k_orient_matches(records, 0)
    assert top == []


def test_top_k_orient_matches_negative_k():
    with pytest.raises(ValueError):
        top_k_orient_matches(_orient_records(), -1)
