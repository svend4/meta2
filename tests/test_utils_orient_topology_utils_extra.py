"""Extra tests for puzzle_reconstruction/utils/orient_topology_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _orient(frag_a=0, frag_b=1, score=0.7, angle=45.0,
            is_flipped=False, n_angles=8) -> OrientMatchRecord:
    return OrientMatchRecord(fragment_a=frag_a, fragment_b=frag_b,
                             best_score=score, best_angle=angle,
                             is_flipped=is_flipped, n_angles_tested=n_angles)


def _topo(solidity=0.9, extent=0.8, convexity=0.95,
          compactness=0.85, complexity=0.5) -> TopologyRecord:
    return TopologyRecord(solidity=solidity, extent=extent,
                          convexity=convexity, compactness=compactness,
                          complexity=complexity)


# ─── OrientMatchRecord ────────────────────────────────────────────────────────

class TestOrientMatchRecordExtra:
    def test_negative_frag_a_raises(self):
        with pytest.raises(ValueError):
            _orient(frag_a=-1)

    def test_negative_frag_b_raises(self):
        with pytest.raises(ValueError):
            _orient(frag_b=-1)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _orient(score=1.5)

    def test_negative_angle_raises(self):
        with pytest.raises(ValueError):
            _orient(angle=-1.0)

    def test_pair_property(self):
        r = _orient(frag_a=2, frag_b=5)
        assert r.pair == (2, 5)

    def test_is_good_match_true(self):
        assert _orient(score=0.6).is_good_match is True

    def test_is_good_match_false(self):
        assert _orient(score=0.4).is_good_match is False


# ─── OrientMatchSummary ───────────────────────────────────────────────────────

class TestOrientMatchSummaryExtra:
    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            OrientMatchSummary(total_pairs=-1)

    def test_negative_good_raises(self):
        with pytest.raises(ValueError):
            OrientMatchSummary(good_pairs=-1)

    def test_good_ratio_zero_when_no_pairs(self):
        s = OrientMatchSummary()
        assert s.good_ratio == pytest.approx(0.0)

    def test_flip_ratio_computed(self):
        s = OrientMatchSummary(total_pairs=4, flipped_pairs=2)
        assert s.flip_ratio == pytest.approx(0.5)


# ─── summarize_orient_matches ────────────────────────────────────────────────

class TestSummarizeOrientMatchesExtra:
    def test_empty_returns_summary(self):
        s = summarize_orient_matches([])
        assert s.total_pairs == 0

    def test_good_pairs_counted(self):
        records = [_orient(score=0.8), _orient(score=0.3)]
        s = summarize_orient_matches(records)
        assert s.good_pairs == 1

    def test_flipped_pairs_counted(self):
        records = [_orient(is_flipped=True), _orient(is_flipped=False)]
        s = summarize_orient_matches(records)
        assert s.flipped_pairs == 1

    def test_mean_score(self):
        records = [_orient(score=0.4), _orient(score=0.8)]
        s = summarize_orient_matches(records)
        assert s.mean_score == pytest.approx(0.6)

    def test_min_max_score(self):
        records = [_orient(score=0.2), _orient(score=0.9)]
        s = summarize_orient_matches(records)
        assert s.min_score == pytest.approx(0.2)
        assert s.max_score == pytest.approx(0.9)


# ─── filter_orient_records ────────────────────────────────────────────────────

class TestFilterOrientRecordsExtra:
    def test_filter_by_min_score(self):
        records = [_orient(score=0.3), _orient(score=0.8)]
        result = filter_orient_records(records, min_score=0.5)
        assert len(result) == 1

    def test_exclude_flipped(self):
        records = [_orient(is_flipped=True), _orient(is_flipped=False)]
        result = filter_orient_records(records, exclude_flipped=True)
        assert len(result) == 1 and not result[0].is_flipped

    def test_no_filter(self):
        records = [_orient(), _orient()]
        result = filter_orient_records(records)
        assert len(result) == 2


# ─── TopologyRecord ───────────────────────────────────────────────────────────

class TestTopologyRecordExtra:
    def test_negative_solidity_raises(self):
        with pytest.raises(ValueError):
            TopologyRecord(solidity=-0.1)

    def test_negative_complexity_raises(self):
        with pytest.raises(ValueError):
            TopologyRecord(complexity=-1.0)

    def test_is_convex_true(self):
        assert _topo(convexity=0.95).is_convex is True

    def test_is_convex_false(self):
        assert _topo(convexity=0.5).is_convex is False

    def test_is_compact_true(self):
        assert _topo(compactness=0.95).is_compact is True

    def test_to_dict_keys(self):
        d = _topo().to_dict()
        for k in ("solidity", "extent", "convexity", "compactness", "complexity"):
            assert k in d


# ─── topology_records_from_dicts ─────────────────────────────────────────────

class TestTopologyRecordsFromDictsExtra:
    def test_returns_list(self):
        records = topology_records_from_dicts([{"solidity": 0.9}])
        assert len(records) == 1

    def test_defaults_for_missing_keys(self):
        records = topology_records_from_dicts([{}])
        assert records[0].solidity == pytest.approx(0.0)

    def test_empty_input(self):
        assert topology_records_from_dicts([]) == []


# ─── summarize_topology ───────────────────────────────────────────────────────

class TestSummarizeTopologyExtra:
    def test_empty_returns_summary(self):
        s = summarize_topology([])
        assert s.n_contours == 0

    def test_n_convex_counted(self):
        records = [_topo(convexity=0.95), _topo(convexity=0.5)]
        s = summarize_topology(records)
        assert s.n_convex == 1

    def test_n_compact_counted(self):
        records = [_topo(compactness=0.95), _topo(compactness=0.5)]
        s = summarize_topology(records)
        assert s.n_compact == 1

    def test_mean_solidity(self):
        records = [_topo(solidity=0.6), _topo(solidity=0.8)]
        s = summarize_topology(records)
        assert s.mean_solidity == pytest.approx(0.7)

    def test_convex_ratio(self):
        records = [_topo(convexity=0.95), _topo(convexity=0.95),
                   _topo(convexity=0.5)]
        s = summarize_topology(records)
        assert s.convex_ratio == pytest.approx(2 / 3)


# ─── rank / top_k orient matches ─────────────────────────────────────────────

class TestRankOrientMatchesExtra:
    def test_rank_descending(self):
        records = [_orient(score=0.3), _orient(score=0.9), _orient(score=0.5)]
        ranked = rank_orient_matches(records)
        assert ranked[0].best_score == pytest.approx(0.9)

    def test_top_k_returns_k(self):
        records = [_orient(score=0.3), _orient(score=0.9), _orient(score=0.5)]
        top = top_k_orient_matches(records, 2)
        assert len(top) == 2

    def test_top_k_zero(self):
        assert top_k_orient_matches([_orient()], 0) == []

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError):
            top_k_orient_matches([_orient()], -1)
