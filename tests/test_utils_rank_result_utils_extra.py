"""Extra tests for puzzle_reconstruction/utils/rank_result_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.rank_result_utils import (
    RankResultConfig,
    RankResultEntry,
    RankResultSummary,
    make_rank_result_entry,
    entries_from_ranked_pairs,
    summarise_rank_results,
    filter_high_rank_entries,
    filter_low_rank_entries,
    filter_by_rank_position,
    filter_rank_by_score_range,
    filter_rank_by_dominant_channel,
    top_k_rank_entries,
    best_rank_entry,
    rerank_entries,
    rank_result_stats,
    compare_rank_summaries,
    batch_summarise_rank_results,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(frag_i=0, frag_j=1, score=0.7, rank=1,
           channels=None, method="rank") -> RankResultEntry:
    return RankResultEntry(frag_i=frag_i, frag_j=frag_j, score=score,
                           rank=rank, channel_scores=channels or {},
                           method=method)


# ─── RankResultConfig ─────────────────────────────────────────────────────────

class TestRankResultConfigExtra:
    def test_default_good_threshold(self):
        assert RankResultConfig().good_threshold == pytest.approx(0.7)

    def test_default_top_k(self):
        assert RankResultConfig().top_k == 10

    def test_good_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RankResultConfig(good_threshold=1.5)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            RankResultConfig(top_k=-1)


# ─── RankResultEntry ──────────────────────────────────────────────────────────

class TestRankResultEntryExtra:
    def test_pair_key_ordered(self):
        e = _entry(frag_i=3, frag_j=1)
        assert e.pair_key == (1, 3)

    def test_is_top_match_true(self):
        e = _entry(rank=1)
        assert e.is_top_match is True

    def test_is_top_match_false(self):
        e = _entry(rank=2)
        assert e.is_top_match is False

    def test_dominant_channel_none(self):
        e = _entry(channels={})
        assert e.dominant_channel is None

    def test_dominant_channel(self):
        e = _entry(channels={"edge": 0.9, "color": 0.4})
        assert e.dominant_channel == "edge"


# ─── make_rank_result_entry ───────────────────────────────────────────────────

class TestMakeRankResultEntryExtra:
    def test_returns_entry(self):
        e = make_rank_result_entry(0, 1, 0.7, 1)
        assert isinstance(e, RankResultEntry)

    def test_channel_scores_stored(self):
        e = make_rank_result_entry(0, 1, 0.7, 1, channel_scores={"edge": 0.8})
        assert e.channel_scores["edge"] == pytest.approx(0.8)

    def test_params_stored(self):
        e = make_rank_result_entry(0, 1, 0.7, 1, params={"k": 5})
        assert e.params["k"] == 5


# ─── entries_from_ranked_pairs ────────────────────────────────────────────────

class TestEntriesFromRankedPairsExtra:
    def test_returns_list(self):
        result = entries_from_ranked_pairs([(0, 1)], [0.8])
        assert isinstance(result, list) and len(result) == 1

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            entries_from_ranked_pairs([(0, 1), (1, 2)], [0.8])

    def test_auto_ranks(self):
        result = entries_from_ranked_pairs([(0, 1), (1, 2)], [0.8, 0.6])
        assert result[0].rank == 1 and result[1].rank == 2

    def test_custom_ranks(self):
        result = entries_from_ranked_pairs([(0, 1)], [0.8], ranks=[5])
        assert result[0].rank == 5


# ─── summarise_rank_results ───────────────────────────────────────────────────

class TestSummariseRankResultsExtra:
    def test_empty_returns_summary(self):
        s = summarise_rank_results([])
        assert s.n_entries == 0

    def test_n_top_matches(self):
        entries = [_entry(rank=1), _entry(rank=2), _entry(rank=1)]
        s = summarise_rank_results(entries)
        assert s.n_top_matches == 2

    def test_mean_score(self):
        entries = [_entry(score=0.4), _entry(score=0.8)]
        s = summarise_rank_results(entries)
        assert s.mean_score == pytest.approx(0.6)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterRankResultExtra:
    def test_filter_high(self):
        entries = [_entry(score=0.8), _entry(score=0.3)]
        result = filter_high_rank_entries(entries, 0.7)
        assert len(result) == 1

    def test_filter_low(self):
        entries = [_entry(score=0.8), _entry(score=0.2)]
        result = filter_low_rank_entries(entries, 0.3)
        assert len(result) == 1

    def test_filter_by_rank_position(self):
        entries = [_entry(rank=1), _entry(rank=5), _entry(rank=2)]
        result = filter_by_rank_position(entries, 2)
        assert len(result) == 2

    def test_filter_score_range(self):
        entries = [_entry(score=0.2), _entry(score=0.5), _entry(score=0.9)]
        result = filter_rank_by_score_range(entries, 0.4, 0.7)
        assert len(result) == 1

    def test_filter_by_dominant_channel(self):
        entries = [_entry(channels={"edge": 0.9, "color": 0.3}),
                   _entry(channels={"edge": 0.3, "color": 0.9})]
        result = filter_rank_by_dominant_channel(entries, "edge")
        assert len(result) == 1


# ─── top_k / best ─────────────────────────────────────────────────────────────

class TestRankingHelpersExtra:
    def test_top_k(self):
        entries = [_entry(score=0.3), _entry(score=0.9), _entry(score=0.6)]
        top = top_k_rank_entries(entries, 2)
        assert top[0].score == pytest.approx(0.9)
        assert len(top) == 2

    def test_best_entry_none(self):
        assert best_rank_entry([]) is None

    def test_best_entry(self):
        entries = [_entry(score=0.3), _entry(score=0.9)]
        best = best_rank_entry(entries)
        assert best.score == pytest.approx(0.9)

    def test_rerank_descending(self):
        entries = [_entry(score=0.3, rank=1), _entry(score=0.9, rank=2)]
        reranked = rerank_entries(entries)
        assert reranked[0].score == pytest.approx(0.9)
        assert reranked[0].rank == 1


# ─── rank_result_stats ────────────────────────────────────────────────────────

class TestRankResultStatsExtra:
    def test_empty_returns_zeros(self):
        s = rank_result_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = rank_result_stats([_entry(), _entry()])
        assert s["count"] == 2

    def test_min_max(self):
        entries = [_entry(score=0.2), _entry(score=0.8)]
        s = rank_result_stats(entries)
        assert s["min_score"] == pytest.approx(0.2)
        assert s["max_score"] == pytest.approx(0.8)


# ─── compare / batch ──────────────────────────────────────────────────────────

class TestCompareRankSummariesExtra:
    def test_returns_dict(self):
        s = summarise_rank_results([_entry()])
        d = compare_rank_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_rank_results([_entry()])
        d = compare_rank_summaries(s, s)
        assert d["d_mean_score"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_rank_results([[_entry()], [_entry(), _entry()]])
        assert len(result) == 2
