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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _simple_channels():
    return {
        "color": {(0, 1): 0.9, (0, 2): 0.5, (1, 2): 0.7, (2, 3): 0.6},
        "shape": {(0, 1): 0.8, (0, 2): 0.4, (1, 2): 0.6, (2, 3): 0.8},
    }


def _simple_result(cfg=None):
    return global_match([0, 1, 2, 3], _simple_channels(), cfg)


def _make_match(fid=0, cid=1, score=0.8, rank=1):
    return GlobalMatch(fragment_id=fid, candidate_id=cid, score=score, rank=rank)


# ─── GlobalMatchConfig (extra) ────────────────────────────────────────────────

class TestGlobalMatchConfigExtra:
    def test_default_top_k(self):
        assert GlobalMatchConfig().top_k == 5

    def test_default_min_score(self):
        assert GlobalMatchConfig().min_score == pytest.approx(0.0)

    def test_default_weights_none(self):
        assert GlobalMatchConfig().weights is None

    def test_default_aggregate_mean(self):
        assert GlobalMatchConfig().aggregate == "mean"

    def test_default_symmetric_true(self):
        assert GlobalMatchConfig().symmetric is True

    def test_top_k_1_ok(self):
        assert GlobalMatchConfig(top_k=1).top_k == 1

    def test_top_k_large(self):
        assert GlobalMatchConfig(top_k=100).top_k == 100

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=0)

    def test_top_k_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(top_k=-1)

    def test_min_score_zero_ok(self):
        assert GlobalMatchConfig(min_score=0.0).min_score == pytest.approx(0.0)

    def test_min_score_one_ok(self):
        assert GlobalMatchConfig(min_score=1.0).min_score == pytest.approx(1.0)

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=-0.1)

    def test_min_score_above_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(min_score=1.1)

    def test_aggregate_mean_ok(self):
        assert GlobalMatchConfig(aggregate="mean").aggregate == "mean"

    def test_aggregate_max_ok(self):
        assert GlobalMatchConfig(aggregate="max").aggregate == "max"

    def test_aggregate_min_ok(self):
        assert GlobalMatchConfig(aggregate="min").aggregate == "min"

    def test_aggregate_invalid_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="sum")

    def test_aggregate_invalid_product_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchConfig(aggregate="product")

    def test_weights_custom(self):
        w = {"color": 0.6, "shape": 0.4}
        cfg = GlobalMatchConfig(weights=w)
        assert cfg.weights == w

    def test_symmetric_false(self):
        assert GlobalMatchConfig(symmetric=False).symmetric is False


# ─── GlobalMatch (extra) ──────────────────────────────────────────────────────

class TestGlobalMatchExtra:
    def test_fragment_id_stored(self):
        m = _make_match(fid=5)
        assert m.fragment_id == 5

    def test_candidate_id_stored(self):
        m = _make_match(cid=10)
        assert m.candidate_id == 10

    def test_score_stored(self):
        m = _make_match(score=0.65)
        assert m.score == pytest.approx(0.65)

    def test_rank_stored(self):
        m = _make_match(rank=3)
        assert m.rank == 3

    def test_is_top_rank_1(self):
        assert _make_match(rank=1).is_top is True

    def test_is_top_rank_2(self):
        assert _make_match(rank=2).is_top is False

    def test_is_top_rank_5(self):
        assert _make_match(rank=5).is_top is False

    def test_channel_scores_default_empty(self):
        assert _make_match().channel_scores == {}

    def test_channel_scores_custom(self):
        m = GlobalMatch(fragment_id=0, candidate_id=1, score=0.7,
                        rank=1, channel_scores={"color": 0.8, "shape": 0.6})
        assert m.channel_scores["color"] == pytest.approx(0.8)
        assert m.channel_scores["shape"] == pytest.approx(0.6)

    def test_score_zero_ok(self):
        m = _make_match(score=0.0)
        assert m.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        m = _make_match(score=1.0)
        assert m.score == pytest.approx(1.0)

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=-0.1, rank=1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=1.1, rank=1)

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=0)

    def test_rank_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatch(fragment_id=0, candidate_id=1, score=0.5, rank=-1)


# ─── GlobalMatchResult (extra) ────────────────────────────────────────────────

class TestGlobalMatchResultExtra:
    def _make(self):
        matches = {
            0: [GlobalMatch(0, 1, 0.9, rank=1), GlobalMatch(0, 2, 0.5, rank=2)],
            1: [GlobalMatch(1, 0, 0.9, rank=1)],
            2: [],
        }
        return GlobalMatchResult(matches=matches, n_fragments=3, n_channels=2)

    def test_n_fragments_stored(self):
        r = self._make()
        assert r.n_fragments == 3

    def test_n_channels_stored(self):
        r = self._make()
        assert r.n_channels == 2

    def test_top_match_rank_1(self):
        r = self._make()
        m = r.top_match(0)
        assert m is not None and m.rank == 1

    def test_top_match_correct_candidate(self):
        r = self._make()
        m = r.top_match(0)
        assert m.candidate_id == 1

    def test_top_match_empty_list_none(self):
        r = self._make()
        assert r.top_match(2) is None

    def test_top_match_missing_key_none(self):
        r = self._make()
        assert r.top_match(99) is None

    def test_all_top_matches_count(self):
        r = self._make()
        tops = r.all_top_matches()
        assert len(tops) == 2

    def test_all_top_matches_all_rank_one(self):
        r = self._make()
        for m in r.all_top_matches():
            assert m.rank == 1

    def test_fragment_ids_excludes_empty(self):
        r = self._make()
        ids = r.fragment_ids()
        assert 2 not in ids

    def test_fragment_ids_includes_nonempty(self):
        r = self._make()
        ids = r.fragment_ids()
        assert 0 in ids and 1 in ids

    def test_n_fragments_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=-1, n_channels=0)

    def test_n_channels_neg_raises(self):
        with pytest.raises(ValueError):
            GlobalMatchResult(matches={}, n_fragments=0, n_channels=-1)

    def test_zero_fragments_ok(self):
        r = GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)
        assert r.n_fragments == 0

    def test_matches_dict_stored(self):
        r = self._make()
        assert isinstance(r.matches, dict)


# ─── aggregate_pair_scores (extra) ────────────────────────────────────────────

class TestAggregatePairScoresExtra:
    def test_empty_channels_returns_empty(self):
        assert aggregate_pair_scores({}) == {}

    def test_single_channel_mean_same(self):
        chs = {"c1": {(0, 1): 0.7}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_two_channels_mean(self):
        chs = {"c1": {(0, 1): 0.6}, "c2": {(0, 1): 0.8}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_aggregate_max(self):
        chs = {"c1": {(0, 1): 0.3}, "c2": {(0, 1): 0.9}}
        agg = aggregate_pair_scores(chs, GlobalMatchConfig(aggregate="max"))
        assert agg[(0, 1)] == pytest.approx(0.9)

    def test_aggregate_min(self):
        chs = {"c1": {(0, 1): 0.3}, "c2": {(0, 1): 0.9}}
        agg = aggregate_pair_scores(chs, GlobalMatchConfig(aggregate="min"))
        assert agg[(0, 1)] == pytest.approx(0.3)

    def test_symmetric_averaging(self):
        chs = {"c1": {(0, 1): 0.8, (1, 0): 0.6}}
        cfg = GlobalMatchConfig(symmetric=True)
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.7)

    def test_normalised_key_order(self):
        chs = {"c1": {(3, 1): 0.7}}
        agg = aggregate_pair_scores(chs)
        assert (1, 3) in agg

    def test_weighted_mean(self):
        chs = {"c1": {(0, 1): 1.0}, "c2": {(0, 1): 0.0}}
        cfg = GlobalMatchConfig(weights={"c1": 3.0, "c2": 1.0})
        agg = aggregate_pair_scores(chs, cfg)
        assert agg[(0, 1)] == pytest.approx(0.75)

    def test_score_clipped_to_one(self):
        chs = {"c1": {(0, 1): 1.5}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] <= 1.0

    def test_score_clipped_to_zero(self):
        chs = {"c1": {(0, 1): -0.5}}
        agg = aggregate_pair_scores(chs)
        assert agg[(0, 1)] >= 0.0

    def test_multiple_pairs(self):
        chs = {"c1": {(0, 1): 0.5, (1, 2): 0.8, (0, 2): 0.3}}
        agg = aggregate_pair_scores(chs)
        assert len(agg) == 3

    def test_returns_dict(self):
        chs = {"c1": {(0, 1): 0.5}}
        assert isinstance(aggregate_pair_scores(chs), dict)


# ─── rank_candidates (extra) ──────────────────────────────────────────────────

class TestRankCandidatesExtra:
    def _pair_scores(self):
        return {(0, 1): 0.9, (0, 2): 0.5, (0, 3): 0.2, (1, 2): 0.7}

    def test_returns_list(self):
        assert isinstance(rank_candidates(0, self._pair_scores()), list)

    def test_all_global_match(self):
        for m in rank_candidates(0, self._pair_scores()):
            assert isinstance(m, GlobalMatch)

    def test_sorted_desc(self):
        ranked = rank_candidates(0, self._pair_scores())
        scores = [m.score for m in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_fragment_id_correct(self):
        for m in rank_candidates(0, self._pair_scores()):
            assert m.fragment_id == 0

    def test_rank_sequential_from_1(self):
        ranked = rank_candidates(0, self._pair_scores())
        for i, m in enumerate(ranked):
            assert m.rank == i + 1

    def test_top_k_limits_results(self):
        ps = {(0, i): 0.9 - i * 0.05 for i in range(1, 10)}
        cfg = GlobalMatchConfig(top_k=3)
        ranked = rank_candidates(0, ps, cfg)
        assert len(ranked) == 3

    def test_min_score_filters_low(self):
        ps = self._pair_scores()
        cfg = GlobalMatchConfig(min_score=0.6)
        ranked = rank_candidates(0, ps, cfg)
        for m in ranked:
            assert m.score >= 0.6

    def test_no_candidates_returns_empty(self):
        assert rank_candidates(99, self._pair_scores()) == []

    def test_candidate_ids_not_equal_fragment(self):
        for m in rank_candidates(0, self._pair_scores()):
            assert m.candidate_id != 0

    def test_pair_both_orders_included(self):
        ps = {(0, 1): 0.9, (1, 0): 0.8}
        ranked = rank_candidates(0, ps)
        assert len(ranked) >= 1

    def test_default_cfg_top_k_5(self):
        ps = {(0, i): 0.9 - i * 0.05 for i in range(1, 10)}
        ranked = rank_candidates(0, ps)
        assert len(ranked) <= 5


# ─── global_match (extra) ─────────────────────────────────────────────────────

class TestGlobalMatchFunctionExtra:
    def test_returns_global_match_result(self):
        r = _simple_result()
        assert isinstance(r, GlobalMatchResult)

    def test_n_fragments_correct(self):
        r = _simple_result()
        assert r.n_fragments == 4

    def test_n_channels_correct(self):
        r = _simple_result()
        assert r.n_channels == 2

    def test_all_fragments_in_matches(self):
        r = _simple_result()
        for fid in [0, 1, 2, 3]:
            assert fid in r.matches

    def test_top_match_exists_for_connected(self):
        r = _simple_result()
        assert r.top_match(0) is not None

    def test_custom_top_k_limits(self):
        cfg = GlobalMatchConfig(top_k=1)
        r = _simple_result(cfg)
        for cands in r.matches.values():
            assert len(cands) <= 1

    def test_empty_fragments(self):
        r = global_match([], {})
        assert r.n_fragments == 0 and r.matches == {}

    def test_single_fragment_no_peers(self):
        r = global_match([0], {})
        assert r.n_fragments == 1
        assert r.matches.get(0) == []

    def test_min_score_filters(self):
        cfg = GlobalMatchConfig(min_score=0.85)
        r = _simple_result(cfg)
        for cands in r.matches.values():
            for m in cands:
                assert m.score >= 0.85

    def test_no_channels_all_empty(self):
        r = global_match([0, 1, 2], {})
        for cands in r.matches.values():
            assert cands == []

    def test_ranks_start_at_one(self):
        r = _simple_result()
        for cands in r.matches.values():
            if cands:
                assert cands[0].rank == 1


# ─── filter_matches (extra) ───────────────────────────────────────────────────

class TestFilterMatchesExtra:
    def test_returns_global_match_result(self):
        r = filter_matches(_simple_result(), 0.5)
        assert isinstance(r, GlobalMatchResult)

    def test_min_zero_keeps_all(self):
        r = _simple_result()
        total_before = sum(len(v) for v in r.matches.values())
        filtered = filter_matches(r, 0.0)
        total_after = sum(len(v) for v in filtered.matches.values())
        assert total_after == total_before

    def test_min_one_removes_all_below(self):
        filtered = filter_matches(_simple_result(), 1.0)
        for cands in filtered.matches.values():
            for m in cands:
                assert m.score == pytest.approx(1.0)

    def test_ranks_reindexed_from_1(self):
        filtered = filter_matches(_simple_result(), 0.5)
        for cands in filtered.matches.values():
            for i, m in enumerate(cands):
                assert m.rank == i + 1

    def test_n_fragments_preserved(self):
        r = _simple_result()
        filtered = filter_matches(r, 0.5)
        assert filtered.n_fragments == r.n_fragments

    def test_n_channels_preserved(self):
        r = _simple_result()
        filtered = filter_matches(r, 0.5)
        assert filtered.n_channels == r.n_channels

    def test_all_remaining_above_threshold(self):
        threshold = 0.7
        filtered = filter_matches(_simple_result(), threshold)
        for cands in filtered.matches.values():
            for m in cands:
                assert m.score >= threshold

    def test_neg_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_matches(_simple_result(), -0.1)

    def test_above_one_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_matches(_simple_result(), 1.5)


# ─── merge_match_results (extra) ──────────────────────────────────────────────

class TestMergeMatchResultsExtra:
    def _two_results(self):
        chs1 = {"color": {(0, 1): 0.9, (0, 2): 0.5}}
        chs2 = {"shape": {(0, 1): 0.7, (0, 2): 0.8}}
        r1 = global_match([0, 1, 2], chs1)
        r2 = global_match([0, 1, 2], chs2)
        return r1, r2

    def test_returns_global_match_result(self):
        r1, r2 = self._two_results()
        assert isinstance(merge_match_results([r1, r2]), GlobalMatchResult)

    def test_empty_list_returns_empty(self):
        merged = merge_match_results([])
        assert merged.n_fragments == 0 and merged.matches == {}

    def test_single_result_identity(self):
        r = _simple_result()
        merged = merge_match_results([r])
        for fid in r.matches:
            assert len(merged.matches.get(fid, [])) == len(r.matches[fid])

    def test_n_channels_summed(self):
        r1, r2 = self._two_results()
        merged = merge_match_results([r1, r2])
        assert merged.n_channels == r1.n_channels + r2.n_channels

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

    def test_n_fragments_covers_all(self):
        r1, r2 = self._two_results()
        merged = merge_match_results([r1, r2])
        assert merged.n_fragments >= 1

    def test_four_results_n_channels(self):
        r = _simple_result()
        merged = merge_match_results([r, r, r, r])
        assert merged.n_channels == 4 * r.n_channels

    def test_same_result_twice(self):
        r = _simple_result()
        merged = merge_match_results([r, r])
        assert merged.n_channels == 2 * r.n_channels
