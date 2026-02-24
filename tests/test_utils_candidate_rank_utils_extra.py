"""Extra tests for puzzle_reconstruction/utils/candidate_rank_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.candidate_rank_utils import (
    CandidateRankConfig,
    CandidateRankEntry,
    CandidateRankSummary,
    make_candidate_entry,
    entries_from_pairs,
    summarise_rankings,
    filter_selected,
    filter_rejected_candidates,
    filter_by_score_range,
    filter_by_rank,
    top_k_candidate_entries,
    candidate_rank_stats,
    compare_rankings,
    batch_summarise_rankings,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(idx1=0, idx2=1, score=0.7, rank=1, selected=True) -> CandidateRankEntry:
    return CandidateRankEntry(
        idx1=idx1, idx2=idx2, score=score, rank=rank, is_selected=selected,
    )


def _entries(n=5) -> list:
    return [_entry(idx1=i, idx2=i+1, score=float(i+1)/n, rank=i+1)
            for i in range(n)]


def _summary(entries=None) -> CandidateRankSummary:
    if entries is None:
        entries = _entries(5)
    return summarise_rankings(entries)


def _pair(idx1=0, idx2=1, score=0.7) -> dict:
    return {"idx1": idx1, "idx2": idx2, "score": score}


# ─── CandidateRankConfig ──────────────────────────────────────────────────────

class TestCandidateRankConfigExtra:
    def test_default_min_score(self):
        assert CandidateRankConfig().min_score == pytest.approx(0.5)

    def test_default_max_pairs(self):
        assert CandidateRankConfig().max_pairs == 0

    def test_default_deduplicate(self):
        assert CandidateRankConfig().deduplicate is True

    def test_min_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            CandidateRankConfig(min_score=-0.1)

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError):
            CandidateRankConfig(min_score=1.1)

    def test_max_pairs_negative_raises(self):
        with pytest.raises(ValueError):
            CandidateRankConfig(max_pairs=-1)

    def test_valid_custom_values(self):
        cfg = CandidateRankConfig(min_score=0.3, max_pairs=10)
        assert cfg.min_score == pytest.approx(0.3)
        assert cfg.max_pairs == 10


# ─── CandidateRankEntry ───────────────────────────────────────────────────────

class TestCandidateRankEntryExtra:
    def test_stores_idx1(self):
        assert _entry(idx1=3).idx1 == 3

    def test_stores_idx2(self):
        assert _entry(idx2=7).idx2 == 7

    def test_stores_score(self):
        assert _entry(score=0.85).score == pytest.approx(0.85)

    def test_stores_rank(self):
        assert _entry(rank=5).rank == 5

    def test_stores_is_selected(self):
        assert _entry(selected=False).is_selected is False

    def test_default_meta_empty(self):
        assert _entry().meta == {}

    def test_repr_is_str(self):
        assert isinstance(repr(_entry()), str)


# ─── make_candidate_entry ─────────────────────────────────────────────────────

class TestMakeCandidateEntryExtra:
    def test_returns_entry(self):
        e = make_candidate_entry(0, 1, 0.7, 1)
        assert isinstance(e, CandidateRankEntry)

    def test_stores_values(self):
        e = make_candidate_entry(2, 5, 0.8, 3)
        assert e.idx1 == 2 and e.idx2 == 5
        assert e.score == pytest.approx(0.8) and e.rank == 3

    def test_selected_above_min_score(self):
        cfg = CandidateRankConfig(min_score=0.5)
        e = make_candidate_entry(0, 1, 0.8, 1, cfg=cfg)
        assert e.is_selected is True

    def test_not_selected_below_min_score(self):
        cfg = CandidateRankConfig(min_score=0.5)
        e = make_candidate_entry(0, 1, 0.3, 2, cfg=cfg)
        assert e.is_selected is False

    def test_none_meta_empty(self):
        e = make_candidate_entry(0, 1, 0.5, 1, meta=None)
        assert e.meta == {}


# ─── entries_from_pairs ───────────────────────────────────────────────────────

class TestEntriesFromPairsExtra:
    def test_returns_list(self):
        result = entries_from_pairs([_pair()])
        assert isinstance(result, list)

    def test_length_matches(self):
        pairs = [_pair(idx1=i, score=0.5) for i in range(5)]
        result = entries_from_pairs(pairs)
        assert len(result) == 5

    def test_empty_pairs(self):
        assert entries_from_pairs([]) == []

    def test_all_are_entries(self):
        for e in entries_from_pairs([_pair()]):
            assert isinstance(e, CandidateRankEntry)

    def test_sorted_desc_by_score(self):
        pairs = [_pair(score=0.3), _pair(score=0.9), _pair(score=0.6)]
        result = entries_from_pairs(pairs)
        scores = [e.score for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_assigned(self):
        pairs = [_pair(score=0.8), _pair(score=0.5)]
        result = entries_from_pairs(pairs)
        assert all(e.rank >= 0 for e in result)


# ─── summarise_rankings ───────────────────────────────────────────────────────

class TestSummariseRankingsExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_rankings(_entries()), CandidateRankSummary)

    def test_n_total_correct(self):
        assert summarise_rankings(_entries(7)).n_total == 7

    def test_empty_entries(self):
        s = summarise_rankings([])
        assert s.n_total == 0

    def test_n_selected_plus_rejected_eq_total(self):
        s = summarise_rankings(_entries(5))
        assert s.n_selected + s.n_rejected == s.n_total

    def test_mean_in_range(self):
        s = summarise_rankings(_entries(5))
        assert s.min_score <= s.mean_score <= s.max_score


# ─── filter_selected / filter_rejected_candidates ────────────────────────────

class TestFilterSelectedExtra:
    def test_returns_list(self):
        assert isinstance(filter_selected(_entries()), list)

    def test_all_selected(self):
        entries = [_entry(selected=True)]
        result = filter_selected(entries)
        assert len(result) == 1

    def test_empty_input(self):
        assert filter_selected([]) == []

    def test_only_selected(self):
        entries = [_entry(selected=True), _entry(selected=False)]
        result = filter_selected(entries)
        assert all(e.is_selected for e in result)


class TestFilterRejectedCandidatesExtra:
    def test_returns_list(self):
        assert isinstance(filter_rejected_candidates(_entries()), list)

    def test_only_rejected(self):
        entries = [_entry(selected=True), _entry(selected=False)]
        result = filter_rejected_candidates(entries)
        assert all(not e.is_selected for e in result)

    def test_empty_input(self):
        assert filter_rejected_candidates([]) == []


# ─── filter_by_score_range ────────────────────────────────────────────────────

class TestFilterByScoreRangeExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_score_range(_entries()), list)

    def test_keeps_in_range(self):
        entries = [_entry(score=0.2), _entry(score=0.6), _entry(score=0.9)]
        result = filter_by_score_range(entries, min_score=0.5, max_score=0.8)
        assert all(0.5 <= e.score <= 0.8 for e in result)

    def test_wide_range_keeps_all(self):
        result = filter_by_score_range(_entries(5), min_score=0.0, max_score=1.0)
        assert len(result) == 5


# ─── filter_by_rank ───────────────────────────────────────────────────────────

class TestFilterByRankExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_rank(_entries(), 3), list)

    def test_keeps_rank_le_max(self):
        entries = [_entry(rank=1), _entry(rank=3), _entry(rank=5)]
        result = filter_by_rank(entries, max_rank=3)
        assert all(e.rank <= 3 for e in result)

    def test_max_rank_zero_empty(self):
        result = filter_by_rank(_entries(), max_rank=0)
        assert result == []


# ─── top_k_candidate_entries ──────────────────────────────────────────────────

class TestTopKCandidateEntriesExtra:
    def test_returns_list(self):
        assert isinstance(top_k_candidate_entries(_entries(), 3), list)

    def test_length_at_most_k(self):
        result = top_k_candidate_entries(_entries(5), 3)
        assert len(result) <= 3

    def test_k_larger_than_n(self):
        result = top_k_candidate_entries(_entries(3), 10)
        assert len(result) == 3

    def test_empty_input(self):
        assert top_k_candidate_entries([], 3) == []


# ─── candidate_rank_stats ─────────────────────────────────────────────────────

class TestCandidateRankStatsExtra:
    def test_returns_dict(self):
        assert isinstance(candidate_rank_stats(_entries()), dict)

    def test_keys_present(self):
        stats = candidate_rank_stats(_entries(5))
        for k in ("count", "mean", "std", "min", "max", "n_selected", "n_rejected"):
            assert k in stats

    def test_count_correct(self):
        assert candidate_rank_stats(_entries(7))["count"] == 7

    def test_empty_entries(self):
        stats = candidate_rank_stats([])
        assert stats["count"] == 0


# ─── compare_rankings ─────────────────────────────────────────────────────────

class TestCompareRankingsExtra:
    def test_returns_dict(self):
        a = _summary()
        b = _summary()
        assert isinstance(compare_rankings(a, b), dict)

    def test_keys_present(self):
        d = compare_rankings(_summary(), _summary())
        for k in ("n_total_delta", "n_selected_delta",
                  "mean_score_delta", "max_score_delta"):
            assert k in d

    def test_identical_zero_delta(self):
        s = _summary()
        d = compare_rankings(s, s)
        assert d["mean_score_delta"] == pytest.approx(0.0)


# ─── batch_summarise_rankings ─────────────────────────────────────────────────

class TestBatchSummariseRankingsExtra:
    def test_returns_list(self):
        result = batch_summarise_rankings([[_pair()]])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_summarise_rankings([[_pair()], [_pair(), _pair()]])
        assert len(result) == 2

    def test_each_is_summary(self):
        for s in batch_summarise_rankings([[_pair()]]):
            assert isinstance(s, CandidateRankSummary)

    def test_empty_returns_empty(self):
        assert batch_summarise_rankings([]) == []
