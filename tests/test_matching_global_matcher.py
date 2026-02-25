"""Тесты для puzzle_reconstruction.matching.global_matcher."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.global_matcher import (
    GlobalMatch,
    GlobalMatchConfig,
    GlobalMatchResult,
    aggregate_pair_scores,
    filter_matches,
    global_match,
    merge_match_results,
    rank_candidates,
)


# ─── TestGlobalMatchConfig ────────────────────────────────────────────────────

class TestGlobalMatchConfig:
    def test_defaults(self):
        cfg = GlobalMatchConfig()
        assert cfg.top_k == 5
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.aggregate == "mean"
        assert cfg.symmetric is True

    def test_custom_values(self):
        cfg = GlobalMatchConfig(top_k=3, min_score=0.2, aggregate="max")
        assert cfg.top_k == 3
        assert cfg.min_score == pytest.approx(0.2)
        assert cfg.aggregate == "max"

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=-0.1)

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=1.1)

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="invalid_method")

    def test_valid_aggregates(self):
        for m in ("mean", "max", "min"):
            cfg = GlobalMatchConfig(aggregate=m)
            assert cfg.aggregate == m


# ─── TestGlobalMatch ──────────────────────────────────────────────────────────

class TestGlobalMatch:
    def test_basic_construction(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8)
        assert m.fragment_id == 0
        assert m.candidate_id == 1
        assert m.score == pytest.approx(0.8)
        assert m.rank == 1

    def test_is_top_true(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=1)
        assert m.is_top is True

    def test_is_top_false(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.8, rank=2)
        assert m.is_top is False

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.1)

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)


# ─── TestGlobalMatchResult ────────────────────────────────────────────────────

class TestGlobalMatchResult:
    def _make(self) -> GlobalMatchResult:
        matches = {
            0: [GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1),
                GlobalMatch(fragment_id=0, candidate_id=2, score=0.7, rank=2)],
            1: [GlobalMatch(fragment_id=1, candidate_id=0, score=0.8, rank=1)],
        }
        return GlobalMatchResult(matches=matches, n_fragments=2, n_channels=2)

    def test_top_match_returns_best(self):
        r = self._make()
        top = r.top_match(0)
        assert top is not None
        assert top.rank == 1

    def test_top_match_missing_fragment(self):
        r = self._make()
        assert r.top_match(99) is None

    def test_all_top_matches_count(self):
        r = self._make()
        tops = r.all_top_matches()
        assert len(tops) == 2

    def test_fragment_ids(self):
        r = self._make()
        fids = r.fragment_ids()
        assert set(fids) == {0, 1}

    def test_n_fragments_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=-1, n_channels=0)

    def test_n_channels_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=0, n_channels=-1)


# ─── TestAggregatePairScores ──────────────────────────────────────────────────

class TestAggregatePairScores:
    def test_returns_dict(self):
        scores = {"ch1": {(0, 1): 0.8, (1, 2): 0.6}}
        result = aggregate_pair_scores(scores)
        assert isinstance(result, dict)

    def test_keys_are_ordered_pairs(self):
        scores = {"ch1": {(1, 0): 0.8}}  # reversed key
        result = aggregate_pair_scores(scores)
        for a, b in result:
            assert a <= b

    def test_values_in_range(self):
        scores = {"ch1": {(0, 1): 0.7}, "ch2": {(0, 1): 0.5}}
        result = aggregate_pair_scores(scores)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_symmetric_averaging(self):
        scores = {"ch1": {(0, 1): 0.8, (1, 0): 0.6}}
        cfg = GlobalMatchConfig(symmetric=True)
        result = aggregate_pair_scores(scores, cfg)
        assert result[(0, 1)] == pytest.approx(0.7)

    def test_empty_input(self):
        result = aggregate_pair_scores({})
        assert result == {}

    def test_mean_aggregation(self):
        scores = {"c1": {(0, 1): 0.6}, "c2": {(0, 1): 0.8}}
        cfg = GlobalMatchConfig(aggregate="mean", symmetric=False)
        result = aggregate_pair_scores(scores, cfg)
        assert result[(0, 1)] == pytest.approx(0.7)


# ─── TestRankCandidates ───────────────────────────────────────────────────────

class TestRankCandidates:
    def _pair_scores(self):
        return {(0, 1): 0.9, (0, 2): 0.7, (0, 3): 0.5, (1, 2): 0.6}

    def test_returns_list(self):
        result = rank_candidates(0, self._pair_scores())
        assert isinstance(result, list)

    def test_sorted_by_score_desc(self):
        result = rank_candidates(0, self._pair_scores())
        scores = [m.score for m in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        cfg = GlobalMatchConfig(top_k=2)
        result = rank_candidates(0, self._pair_scores(), cfg)
        assert len(result) <= 2

    def test_rank_starts_at_1(self):
        result = rank_candidates(0, self._pair_scores())
        if result:
            assert result[0].rank == 1

    def test_min_score_filter(self):
        cfg = GlobalMatchConfig(min_score=0.8)
        result = rank_candidates(0, self._pair_scores(), cfg)
        assert all(m.score >= 0.8 for m in result)

    def test_no_candidates_for_absent_fragment(self):
        result = rank_candidates(99, self._pair_scores())
        assert result == []


# ─── TestGlobalMatch Function ─────────────────────────────────────────────────

class TestGlobalMatchFn:
    def test_returns_global_match_result(self):
        fragment_ids = [0, 1, 2]
        scores = {"ch1": {(0, 1): 0.8, (1, 2): 0.6, (0, 2): 0.5}}
        r = global_match(fragment_ids, scores)
        assert isinstance(r, GlobalMatchResult)

    def test_n_fragments_correct(self):
        fragment_ids = [0, 1, 2]
        scores = {"ch1": {(0, 1): 0.8}}
        r = global_match(fragment_ids, scores)
        assert r.n_fragments == 3

    def test_n_channels_correct(self):
        fragment_ids = [0, 1]
        scores = {"c1": {(0, 1): 0.8}, "c2": {(0, 1): 0.7}}
        r = global_match(fragment_ids, scores)
        assert r.n_channels == 2

    def test_empty_fragments(self):
        r = global_match([], {})
        assert r.n_fragments == 0


# ─── TestFilterMatches ────────────────────────────────────────────────────────

class TestFilterMatches:
    def _result(self) -> GlobalMatchResult:
        matches = {
            0: [GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1),
                GlobalMatch(fragment_id=0, candidate_id=2, score=0.3, rank=2)],
        }
        return GlobalMatchResult(matches=matches, n_fragments=1, n_channels=1)

    def test_filters_low_scores(self):
        r = self._result()
        filtered = filter_matches(r, min_score=0.5)
        assert all(m.score >= 0.5 for ms in filtered.matches.values() for m in ms)

    def test_invalid_min_score_raises(self):
        r = self._result()
        with pytest.raises(ValueError):
            filter_matches(r, min_score=1.5)

    def test_reindexes_ranks(self):
        r = self._result()
        filtered = filter_matches(r, min_score=0.5)
        for cands in filtered.matches.values():
            for i, m in enumerate(cands):
                assert m.rank == i + 1


# ─── TestMergeMatchResults ────────────────────────────────────────────────────

class TestMergeMatchResults:
    def test_empty_returns_empty(self):
        r = merge_match_results([])
        assert r.n_fragments == 0

    def test_single_result_preserved(self):
        matches = {0: [GlobalMatch(fragment_id=0, candidate_id=1, score=0.9, rank=1)]}
        r = GlobalMatchResult(matches=matches, n_fragments=1, n_channels=1)
        merged = merge_match_results([r])
        assert 0 in merged.matches

    def test_multiple_results_merged(self):
        r1 = GlobalMatchResult(
            matches={0: [GlobalMatch(0, 1, 0.8, rank=1)]},
            n_fragments=1, n_channels=1
        )
        r2 = GlobalMatchResult(
            matches={0: [GlobalMatch(0, 2, 0.7, rank=1)]},
            n_fragments=1, n_channels=1
        )
        merged = merge_match_results([r1, r2])
        # Both candidates should be present (up to top_k)
        assert len(merged.matches.get(0, [])) >= 1
