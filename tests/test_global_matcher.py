"""Тесты для puzzle_reconstruction.matching.global_matcher."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ch(pairs: dict) -> dict:
    """Канал с попарными оценками."""
    return pairs


def _simple_channels():
    """Два канала для фрагментов 0-3."""
    return {
        "color": {(0, 1): 0.9, (0, 2): 0.5, (1, 2): 0.7, (2, 3): 0.6},
        "shape": {(0, 1): 0.8, (0, 2): 0.4, (1, 2): 0.6, (2, 3): 0.8},
    }


def _simple_result(cfg=None):
    return global_match([0, 1, 2, 3], _simple_channels(), cfg)


# ─── TestGlobalMatchConfig ────────────────────────────────────────────────────

class TestGlobalMatchConfig:
    def test_defaults(self):
        cfg = GlobalMatchConfig()
        assert cfg.top_k == 5
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.weights is None
        assert cfg.aggregate == "mean"
        assert cfg.symmetric is True

    def test_valid_custom(self):
        cfg = GlobalMatchConfig(top_k=3, min_score=0.3,
                                aggregate="max", symmetric=False)
        assert cfg.top_k == 3
        assert cfg.min_score == pytest.approx(0.3)
        assert cfg.aggregate == "max"

    def test_top_k_one_ok(self):
        cfg = GlobalMatchConfig(top_k=1)
        assert cfg.top_k == 1

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_top_k_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=-1)

    def test_min_score_zero_ok(self):
        cfg = GlobalMatchConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_min_score_one_ok(self):
        cfg = GlobalMatchConfig(min_score=1.0)
        assert cfg.min_score == 1.0

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=-0.1)

    def test_min_score_above_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=1.1)

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="sum")

    def test_valid_aggregate_max(self):
        cfg = GlobalMatchConfig(aggregate="max")
        assert cfg.aggregate == "max"

    def test_valid_aggregate_min(self):
        cfg = GlobalMatchConfig(aggregate="min")
        assert cfg.aggregate == "min"

    def test_weights_set(self):
        cfg = GlobalMatchConfig(weights={"color": 0.7, "shape": 0.3})
        assert cfg.weights == {"color": 0.7, "shape": 0.3}


# ─── TestGlobalMatch ──────────────────────────────────────────────────────────

class TestGlobalMatch:
    def _make(self, rank=1, score=0.8):
        return GlobalMatch(fragment_id=0, candidate_id=1, score=score, rank=rank)

    def test_basic(self):
        m = self._make()
        assert m.fragment_id == 0
        assert m.candidate_id == 1
        assert m.score == pytest.approx(0.8)

    def test_is_top_true(self):
        m = self._make(rank=1)
        assert m.is_top is True

    def test_is_top_false(self):
        m = self._make(rank=2)
        assert m.is_top is False

    def test_channel_scores_default_empty(self):
        m = self._make()
        assert m.channel_scores == {}

    def test_channel_scores_set(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.7,
                        channel_scores={"color": 0.8, "shape": 0.6}, rank=1)
        assert m.channel_scores["color"] == pytest.approx(0.8)

    def test_invalid_score_neg(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=-0.1, rank=1)

    def test_invalid_score_above(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.1, rank=1)

    def test_invalid_rank_zero(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)

    def test_invalid_rank_neg(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=-1)


# ─── TestGlobalMatchResult ────────────────────────────────────────────────────

class TestGlobalMatchResult:
    def _make(self):
        matches = {
            0: [
                GlobalMatch(0, 1, 0.9, rank=1),
                GlobalMatch(0, 2, 0.5, rank=2),
            ],
            1: [GlobalMatch(1, 0, 0.9, rank=1)],
            2: [],
        }
        return GlobalMatchResult(matches=matches, n_fragments=3, n_channels=2)

    def test_top_match_found(self):
        r = self._make()
        m = r.top_match(0)
        assert m is not None
        assert m.candidate_id == 1
        assert m.rank == 1

    def test_top_match_empty_list(self):
        r = self._make()
        assert r.top_match(2) is None

    def test_top_match_not_found(self):
        r = self._make()
        assert r.top_match(99) is None

    def test_all_top_matches_count(self):
        r = self._make()
        tops = r.all_top_matches()
        assert len(tops) == 2  # fragment 2 has empty list

    def test_all_top_matches_rank_one(self):
        r = self._make()
        for m in r.all_top_matches():
            assert m.rank == 1

    def test_fragment_ids_nonempty_only(self):
        r = self._make()
        ids = r.fragment_ids()
        assert 0 in ids
        assert 1 in ids
        assert 2 not in ids

    def test_invalid_n_fragments_neg(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=-1, n_channels=0)

    def test_invalid_n_channels_neg(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=0, n_channels=-1)


# ─── TestAggregatePairScores ──────────────────────────────────────────────────

class TestAggregatePairScores:
    def test_mean_two_channels(self):
        chs = {
            "c1": {(0, 1): 0.8},
            "c2": {(0, 1): 0.6},
        }
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_symmetric_averaging(self):
        chs = {"c1": {(0, 1): 0.8, (1, 0): 0.6}}
        cfg = GlobalMatchConfig(symmetric=True)
        agg = aggregate_pair_scores(chs, cfg)
        # Symmetric: avg of 0.8 and 0.6 = 0.7
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_non_symmetric_uses_one_direction(self):
        chs = {"c1": {(0, 1): 0.9}}
        cfg = GlobalMatchConfig(symmetric=False)
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.9)

    def test_aggregate_max(self):
        chs = {
            "c1": {(0, 1): 0.3},
            "c2": {(0, 1): 0.9},
        }
        cfg = GlobalMatchConfig(aggregate="max")
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.9)

    def test_aggregate_min(self):
        chs = {
            "c1": {(0, 1): 0.3},
            "c2": {(0, 1): 0.9},
        }
        cfg = GlobalMatchConfig(aggregate="min")
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.3)

    def test_weighted_mean(self):
        chs = {
            "c1": {(0, 1): 1.0},
            "c2": {(0, 1): 0.0},
        }
        cfg = GlobalMatchConfig(weights={"c1": 3.0, "c2": 1.0})
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.75)

    def test_pair_normalised_order(self):
        chs = {"c1": {(3, 1): 0.7}}
        agg = aggregate_pair_scores(chs)
        assert (1, 3) in agg

    def test_empty_channels_returns_empty(self):
        assert aggregate_pair_scores({}) == {}

    def test_scores_clipped_to_1(self):
        chs = {"c1": {(0, 1): 1.5}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] <= 1.0

    def test_scores_clipped_to_0(self):
        chs = {"c1": {(0, 1): -0.5}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] >= 0.0


# ─── TestRankCandidates ───────────────────────────────────────────────────────

class TestRankCandidates:
    def _pair_scores(self):
        return {
            (0, 1): 0.9,
            (0, 2): 0.5,
            (0, 3): 0.2,
            (1, 2): 0.7,
        }

    def test_basic(self):
        ps = self._pair_scores()
        ranked = rank_candidates(0, ps)
        assert len(ranked) <= 5
        assert all(isinstance(m, GlobalMatch) for m in ranked)

    def test_sorted_descending(self):
        ranked = rank_candidates(0, self._pair_scores())
        scores = [m.score for m in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_fragment_id_correct(self):
        ranked = rank_candidates(0, self._pair_scores())
        for m in ranked:
            assert m.fragment_id == 0

    def test_rank_sequential(self):
        ranked = rank_candidates(0, self._pair_scores())
        for i, m in enumerate(ranked):
            assert m.rank == i + 1

    def test_top_k_limits(self):
        ps = {(0, i): 0.9 - i * 0.05 for i in range(1, 10)}
        cfg = GlobalMatchConfig(top_k=3)
        ranked = rank_candidates(0, ps, cfg)
        assert len(ranked) == 3

    def test_min_score_filters(self):
        ps = self._pair_scores()
        cfg = GlobalMatchConfig(min_score=0.6)
        ranked = rank_candidates(0, ps, cfg)
        for m in ranked:
            assert m.score >= 0.6

    def test_no_candidates_returns_empty(self):
        # Fragment 5 has no pairs
        ranked = rank_candidates(5, self._pair_scores())
        assert ranked == []

    def test_candidate_ids_not_equal_fragment(self):
        ranked = rank_candidates(0, self._pair_scores())
        for m in ranked:
            assert m.candidate_id != 0


# ─── TestGlobalMatch (function) ───────────────────────────────────────────────

class TestGlobalMatchFunction:
    def test_basic(self):
        r = _simple_result()
        assert isinstance(r, GlobalMatchResult)
        assert r.n_fragments == 4
        assert r.n_channels == 2

    def test_all_fragments_have_matches_dict_entry(self):
        r = _simple_result()
        for fid in [0, 1, 2, 3]:
            assert fid in r.matches

    def test_top_match_not_none_for_connected(self):
        r = _simple_result()
        assert r.top_match(0) is not None

    def test_custom_top_k(self):
        cfg = GlobalMatchConfig(top_k=1)
        r = global_match([0, 1, 2, 3], _simple_channels(), cfg)
        for fid, cands in r.matches.items():
            assert len(cands) <= 1

    def test_empty_fragments_returns_empty(self):
        r = global_match([], {})
        assert r.n_fragments == 0
        assert r.matches == {}

    def test_single_fragment_no_pairs(self):
        r = global_match([0], {})
        assert r.n_fragments == 1
        assert r.matches[0] == []

    def test_min_score_filters(self):
        cfg = GlobalMatchConfig(min_score=0.85)
        r = global_match([0, 1, 2, 3], _simple_channels(), cfg)
        for cands in r.matches.values():
            for m in cands:
                assert m.score >= 0.85


# ─── TestFilterMatches ────────────────────────────────────────────────────────

class TestFilterMatches:
    def test_basic_filter(self):
        r = _simple_result()
        filtered = filter_matches(r, min_score=0.7)
        for cands in filtered.matches.values():
            for m in cands:
                assert m.score >= 0.7

    def test_zero_keeps_all(self):
        r = _simple_result()
        filtered = filter_matches(r, min_score=0.0)
        total_before = sum(len(v) for v in r.matches.values())
        total_after = sum(len(v) for v in filtered.matches.values())
        assert total_after == total_before

    def test_one_keeps_none_or_exact(self):
        r = _simple_result()
        filtered = filter_matches(r, min_score=1.0)
        for cands in filtered.matches.values():
            for m in cands:
                assert m.score == pytest.approx(1.0)

    def test_ranks_reindexed(self):
        r = _simple_result()
        filtered = filter_matches(r, min_score=0.5)
        for cands in filtered.matches.values():
            for i, m in enumerate(cands):
                assert m.rank == i + 1

    def test_n_fragments_preserved(self):
        r = _simple_result()
        filtered = filter_matches(r, min_score=0.5)
        assert filtered.n_fragments == r.n_fragments

    def test_invalid_min_score_neg(self):
        r = _simple_result()
        with pytest.raises(ValueError):
            filter_matches(r, min_score=-0.1)

    def test_invalid_min_score_above(self):
        r = _simple_result()
        with pytest.raises(ValueError):
            filter_matches(r, min_score=1.5)


# ─── TestMergeMatchResults ────────────────────────────────────────────────────

class TestMergeMatchResults:
    def _two_results(self):
        chs1 = {"color": {(0, 1): 0.9, (0, 2): 0.5}}
        chs2 = {"shape": {(0, 1): 0.7, (0, 2): 0.8}}
        r1 = global_match([0, 1, 2], chs1)
        r2 = global_match([0, 1, 2], chs2)
        return r1, r2

    def test_basic_merge(self):
        r1, r2 = self._two_results()
        merged = merge_match_results([r1, r2])
        assert isinstance(merged, GlobalMatchResult)

    def test_merged_n_fragments_covers_all(self):
        r1, r2 = self._two_results()
        merged = merge_match_results([r1, r2])
        assert merged.n_fragments >= 1

    def test_empty_list_returns_empty(self):
        merged = merge_match_results([])
        assert merged.n_fragments == 0
        assert merged.matches == {}

    def test_single_result_identity(self):
        r = _simple_result()
        merged = merge_match_results([r])
        for fid in r.matches:
            assert len(merged.matches.get(fid, [])) == len(r.matches[fid])

    def test_ranks_sequential_after_merge(self):
        r1, r2 = self._two_results()
        merged = merge_match_results([r1, r2])
        for cands in merged.matches.values():
            for i, m in enumerate(cands):
                assert m.rank == i + 1

    def test_top_k_respected(self):
        r1, r2 = self._two_results()
        cfg = GlobalMatchConfig(top_k=1)
        merged = merge_match_results([r1, r2], cfg)
        for cands in merged.matches.values():
            assert len(cands) <= 1

    def test_total_channels_summed(self):
        r1 = _simple_result()
        r2 = _simple_result()
        merged = merge_match_results([r1, r2])
        assert merged.n_channels == r1.n_channels + r2.n_channels
