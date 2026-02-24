"""Extra tests for puzzle_reconstruction/matching/global_matcher.py."""
from __future__ import annotations

import pytest

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


# ─── GlobalMatchConfig ──────────────────────────────────────────────────────

class TestGlobalMatchConfigExtra:
    def test_defaults(self):
        cfg = GlobalMatchConfig()
        assert cfg.top_k == 5
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.aggregate == "mean"
        assert cfg.symmetric is True

    def test_zero_top_k_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=-0.1)

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=1.5)

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="sum")

    def test_valid_aggregates(self):
        for agg in ("mean", "max", "min"):
            GlobalMatchConfig(aggregate=agg)


# ─── GlobalMatch ────────────────────────────────────────────────────────────

class TestGlobalMatchExtra:
    def test_fields_stored(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        assert m.fragment_id == 0 and m.candidate_id == 1

    def test_is_top_true(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        assert m.is_top is True

    def test_is_top_false(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=2)
        assert m.is_top is False

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.5, rank=1)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=-0.1, rank=1)

    def test_zero_rank_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)


# ─── GlobalMatchResult ──────────────────────────────────────────────────────

class TestGlobalMatchResultExtra:
    def test_top_match_found(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1)
        r = GlobalMatchResult(matches={0: [m]}, n_fragments=1, n_channels=1)
        assert r.top_match(0) is not None
        assert r.top_match(0).score == pytest.approx(0.9)

    def test_top_match_not_found(self):
        r = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        assert r.top_match(99) is None

    def test_all_top_matches(self):
        m0 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1)
        m1 = GlobalMatch(fragment_id=1, candidate_id=0, score=0.8, rank=1)
        r = GlobalMatchResult(matches={0: [m0], 1: [m1]},
                              n_fragments=2, n_channels=1)
        tops = r.all_top_matches()
        assert len(tops) == 2

    def test_fragment_ids(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=1)
        r = GlobalMatchResult(matches={0: [m], 1: []},
                              n_fragments=2, n_channels=1)
        assert 0 in r.fragment_ids()
        assert 1 not in r.fragment_ids()

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=-1, n_channels=0)

    def test_negative_n_channels_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=0, n_channels=-1)


# ─── aggregate_pair_scores ──────────────────────────────────────────────────

class TestAggregatePairScoresExtra:
    def test_single_channel(self):
        scores = {"ch1": {(0, 1): 0.8, (0, 2): 0.5}}
        agg = aggregate_pair_scores(scores)
        assert (0, 1) in agg
        assert agg[(0, 1)] == pytest.approx(0.8)

    def test_two_channels_mean(self):
        scores = {
            "ch1": {(0, 1): 0.6},
            "ch2": {(0, 1): 0.8},
        }
        agg = aggregate_pair_scores(scores)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_symmetric_averaging(self):
        scores = {"ch1": {(0, 1): 0.6, (1, 0): 0.8}}
        cfg = GlobalMatchConfig(symmetric=True)
        agg = aggregate_pair_scores(scores, cfg)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_aggregate_max(self):
        scores = {
            "ch1": {(0, 1): 0.4},
            "ch2": {(0, 1): 0.9},
        }
        cfg = GlobalMatchConfig(aggregate="max")
        agg = aggregate_pair_scores(scores, cfg)
        assert agg[(0, 1)] == pytest.approx(0.9)

    def test_aggregate_min(self):
        scores = {
            "ch1": {(0, 1): 0.4},
            "ch2": {(0, 1): 0.9},
        }
        cfg = GlobalMatchConfig(aggregate="min")
        agg = aggregate_pair_scores(scores, cfg)
        assert agg[(0, 1)] == pytest.approx(0.4)


# ─── rank_candidates ────────────────────────────────────────────────────────

class TestRankCandidatesExtra:
    def test_basic(self):
        pair_scores = {(0, 1): 0.8, (0, 2): 0.5, (0, 3): 0.9}
        cands = rank_candidates(0, pair_scores)
        assert cands[0].candidate_id == 3
        assert cands[0].rank == 1

    def test_top_k_limits(self):
        pair_scores = {(0, i): 0.1 * i for i in range(1, 10)}
        cfg = GlobalMatchConfig(top_k=3)
        cands = rank_candidates(0, pair_scores, cfg)
        assert len(cands) == 3

    def test_min_score_filters(self):
        pair_scores = {(0, 1): 0.3, (0, 2): 0.8}
        cfg = GlobalMatchConfig(min_score=0.5)
        cands = rank_candidates(0, pair_scores, cfg)
        assert len(cands) == 1


# ─── global_match ────────────────────────────────────────────────────────────

class TestGlobalMatchFuncExtra:
    def test_basic(self):
        scores = {"ch1": {(0, 1): 0.8, (1, 2): 0.6}}
        result = global_match([0, 1, 2], scores)
        assert isinstance(result, GlobalMatchResult)
        assert result.n_fragments == 3
        assert result.n_channels == 1

    def test_empty(self):
        result = global_match([], {})
        assert result.n_fragments == 0


# ─── filter_matches ─────────────────────────────────────────────────────────

class TestFilterMatchesExtra:
    def test_basic(self):
        m1 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.3, rank=1)
        m2 = GlobalMatch(fragment_id=0, candidate_id=2, score=0.8, rank=2)
        r = GlobalMatchResult(matches={0: [m1, m2]},
                              n_fragments=1, n_channels=1)
        filtered = filter_matches(r, min_score=0.5)
        assert len(filtered.matches[0]) == 1
        assert filtered.matches[0][0].rank == 1

    def test_invalid_min_score_raises(self):
        r = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        with pytest.raises(ValueError):
            filter_matches(r, min_score=1.5)


# ─── merge_match_results ────────────────────────────────────────────────────

class TestMergeMatchResultsExtra:
    def test_empty(self):
        result = merge_match_results([])
        assert result.n_fragments == 0

    def test_single(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        r = GlobalMatchResult(matches={0: [m]}, n_fragments=1, n_channels=1)
        merged = merge_match_results([r])
        assert merged.n_fragments == 1

    def test_two_results(self):
        m1 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.6, rank=1)
        m2 = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        r1 = GlobalMatchResult(matches={0: [m1]}, n_fragments=1, n_channels=1)
        r2 = GlobalMatchResult(matches={0: [m2]}, n_fragments=1, n_channels=1)
        merged = merge_match_results([r1, r2])
        # Score should be average of 0.6 and 0.8 = 0.7
        assert merged.matches[0][0].score == pytest.approx(0.7)
