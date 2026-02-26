"""Тесты для puzzle_reconstruction/matching/global_matcher.py."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.global_matcher import (
    GlobalMatchConfig,
    GlobalMatch,
    GlobalMatchResult,
    aggregate_pair_scores,
    rank_candidates,
    global_match,
    filter_matches,
    merge_match_results,
)


def _make_scores():
    return {
        "css": {(0, 1): 0.9, (0, 2): 0.6, (1, 2): 0.7},
        "dtw": {(0, 1): 0.8, (0, 2): 0.5, (1, 2): 0.75},
    }


class TestGlobalMatchConfig:
    def test_defaults(self):
        c = GlobalMatchConfig()
        assert c.top_k == 5
        assert c.min_score == 0.0
        assert c.aggregate == "mean"

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=1.5)

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="invalid")


class TestGlobalMatch:
    def test_is_top_property(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        assert m.is_top

    def test_not_top_for_rank_2(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=2)
        assert not m.is_top

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.5, rank=1)

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)


class TestGlobalMatchResult:
    def _make_result(self):
        matches = {
            0: [GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1),
                GlobalMatch(fragment_id=0, candidate_id=2, score=0.6, rank=2)],
            1: [GlobalMatch(fragment_id=1, candidate_id=0, score=0.9, rank=1)],
        }
        return GlobalMatchResult(matches=matches, n_fragments=3, n_channels=2)

    def test_top_match(self):
        r = self._make_result()
        top = r.top_match(0)
        assert top is not None and top.rank == 1

    def test_top_match_none_for_missing(self):
        r = self._make_result()
        assert r.top_match(99) is None

    def test_all_top_matches(self):
        r = self._make_result()
        assert len(r.all_top_matches()) == 2

    def test_fragment_ids(self):
        r = self._make_result()
        assert set(r.fragment_ids()) == {0, 1}


class TestAggregatePairScores:
    def test_aggregates_channels(self):
        agg = aggregate_pair_scores(_make_scores())
        assert (0, 1) in agg
        assert 0.0 <= agg[(0, 1)] <= 1.0

    def test_all_pairs_present(self):
        agg = aggregate_pair_scores(_make_scores())
        assert len(agg) == 3

    def test_symmetric_aggregation(self):
        scores = {"ch": {(0, 1): 0.8, (1, 0): 0.6}}
        cfg = GlobalMatchConfig(symmetric=True)
        agg = aggregate_pair_scores(scores, cfg)
        val = agg.get((0, 1)) or agg.get((1, 0))
        assert val == pytest.approx(0.7)


class TestRankCandidates:
    def test_rank_1_highest_score(self):
        pair_scores = {(0, 1): 0.9, (0, 2): 0.6, (0, 3): 0.7}
        ranked = rank_candidates(0, pair_scores)
        assert ranked[0].score == pytest.approx(0.9) and ranked[0].rank == 1

    def test_top_k_limits_results(self):
        pair_scores = {(0, i): float(i) / 10 for i in range(1, 10)}
        ranked = rank_candidates(0, pair_scores, GlobalMatchConfig(top_k=3))
        assert len(ranked) <= 3

    def test_min_score_filter(self):
        pair_scores = {(0, 1): 0.9, (0, 2): 0.3}
        ranked = rank_candidates(0, pair_scores, GlobalMatchConfig(min_score=0.5))
        assert all(m.score >= 0.5 for m in ranked)


class TestGlobalMatchFn:
    def test_basic_run(self):
        result = global_match([0, 1, 2], _make_scores())
        assert isinstance(result, GlobalMatchResult)
        assert result.n_fragments == 3
        assert result.n_channels == 2

    def test_all_fragment_ids_present(self):
        result = global_match([0, 1, 2], _make_scores())
        assert set(result.matches.keys()) == {0, 1, 2}


class TestFilterMatches:
    def test_filters_low_scores(self):
        matches = {0: [
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1),
            GlobalMatch(fragment_id=0, candidate_id=2, score=0.3, rank=2),
        ]}
        r = GlobalMatchResult(matches=matches, n_fragments=2, n_channels=1)
        filtered = filter_matches(r, min_score=0.5)
        assert len(filtered.matches[0]) == 1
        assert filtered.matches[0][0].score == pytest.approx(0.9)

    def test_invalid_min_score_raises(self):
        r = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        with pytest.raises(ValueError):
            filter_matches(r, min_score=1.5)


class TestMergeMatchResults:
    def test_merge_two_results(self):
        r1 = global_match([0, 1, 2], _make_scores())
        r2 = global_match([0, 1, 2], {"seam": {(0, 1): 0.85, (1, 2): 0.65}})
        merged = merge_match_results([r1, r2])
        assert isinstance(merged, GlobalMatchResult)

    def test_empty_list_returns_empty(self):
        assert merge_match_results([]).n_fragments == 0
