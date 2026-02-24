"""Extra tests for puzzle_reconstruction/utils/ranking_layout_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.ranking_layout_utils import (
    GlobalRankingConfig,
    GlobalRankingEntry,
    GlobalRankingSummary,
    LayoutScoringConfig,
    LayoutScoringEntry,
    LayoutScoringSummary,
    make_global_ranking_entry,
    summarise_global_ranking_entries,
    filter_ranking_by_min_score,
    filter_ranking_by_fragment,
    filter_ranking_by_top_k,
    top_k_ranking_entries,
    best_ranking_entry,
    ranking_score_stats,
    compare_global_ranking_summaries,
    batch_summarise_global_ranking_entries,
    make_layout_scoring_entry,
    summarise_layout_scoring_entries,
    filter_layout_by_min_score,
    filter_layout_by_quality,
    filter_layout_by_max_overlap,
    top_k_layout_entries,
    best_layout_entry,
    layout_score_stats,
    compare_layout_scoring_summaries,
    batch_summarise_layout_scoring_entries,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rank_entry(idx1=0, idx2=1, score=0.8, rank=0) -> GlobalRankingEntry:
    return GlobalRankingEntry(idx1=idx1, idx2=idx2, score=score, rank=rank)


def _layout_entry(layout_id=0, total=0.7, coverage=0.9, overlap=0.1,
                   uniformity=0.8, n_frag=10, quality="good") -> LayoutScoringEntry:
    return LayoutScoringEntry(layout_id=layout_id, total_score=total,
                               coverage=coverage, overlap_ratio=overlap,
                               uniformity=uniformity, n_fragments=n_frag,
                               quality_level=quality)


# ─── GlobalRankingConfig ──────────────────────────────────────────────────────

class TestGlobalRankingConfigExtra:
    def test_default_min_score(self):
        assert GlobalRankingConfig().min_score == pytest.approx(0.0)

    def test_default_top_k(self):
        assert GlobalRankingConfig().top_k == 10

    def test_default_source_names_empty(self):
        assert GlobalRankingConfig().source_names == []


# ─── make_global_ranking_entry ────────────────────────────────────────────────

class TestMakeGlobalRankingEntryExtra:
    def test_returns_entry(self):
        e = make_global_ranking_entry(0, 1, 0.8, 0)
        assert isinstance(e, GlobalRankingEntry)

    def test_component_scores_stored(self):
        e = make_global_ranking_entry(0, 1, 0.8, 0, edge=0.9, color=0.7)
        assert e.component_scores["edge"] == pytest.approx(0.9)


# ─── summarise_global_ranking_entries ─────────────────────────────────────────

class TestSummariseGlobalRankingEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_global_ranking_entries([])
        assert s.n_pairs == 0 and s.top_pair is None

    def test_single_entry_top_pair(self):
        e = _rank_entry(idx1=2, idx2=3, rank=0)
        s = summarise_global_ranking_entries([e])
        assert s.top_pair == (2, 3)

    def test_mean_score(self):
        entries = [_rank_entry(score=0.4), _rank_entry(score=0.8)]
        s = summarise_global_ranking_entries(entries)
        assert s.mean_score == pytest.approx(0.6)

    def test_min_max_score(self):
        entries = [_rank_entry(score=0.2), _rank_entry(score=0.9)]
        s = summarise_global_ranking_entries(entries)
        assert s.min_score == pytest.approx(0.2)
        assert s.max_score == pytest.approx(0.9)


# ─── filter global ranking ────────────────────────────────────────────────────

class TestFilterGlobalRankingExtra:
    def test_filter_by_min_score(self):
        entries = [_rank_entry(score=0.3), _rank_entry(score=0.9)]
        result = filter_ranking_by_min_score(entries, 0.5)
        assert len(result) == 1

    def test_filter_by_fragment(self):
        entries = [_rank_entry(idx1=0, idx2=1),
                   _rank_entry(idx1=2, idx2=3)]
        result = filter_ranking_by_fragment(entries, 0)
        assert len(result) == 1

    def test_filter_by_top_k(self):
        entries = [_rank_entry(rank=2), _rank_entry(rank=0), _rank_entry(rank=5)]
        result = filter_ranking_by_top_k(entries, 2)
        assert all(e.rank <= 2 for e in result)

    def test_top_k_by_score(self):
        entries = [_rank_entry(score=0.3), _rank_entry(score=0.9),
                   _rank_entry(score=0.6)]
        top = top_k_ranking_entries(entries, 2)
        assert top[0].score == pytest.approx(0.9)

    def test_best_ranking_entry_empty(self):
        assert best_ranking_entry([]) is None

    def test_best_ranking_entry(self):
        entries = [_rank_entry(score=0.3), _rank_entry(score=0.9)]
        best = best_ranking_entry(entries)
        assert best.score == pytest.approx(0.9)


# ─── ranking_score_stats ──────────────────────────────────────────────────────

class TestRankingScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = ranking_score_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = ranking_score_stats([_rank_entry(), _rank_entry()])
        assert s["count"] == 2


# ─── compare / batch global ───────────────────────────────────────────────────

class TestCompareGlobalRankingExtra:
    def test_returns_dict(self):
        s = summarise_global_ranking_entries([_rank_entry()])
        d = compare_global_ranking_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_global_ranking_entries([_rank_entry()])
        d = compare_global_ranking_summaries(s, s)
        assert d["delta_mean_score"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_global_ranking_entries([[_rank_entry()], []])
        assert len(result) == 2


# ─── LayoutScoringConfig ──────────────────────────────────────────────────────

class TestLayoutScoringConfigExtra:
    def test_default_min_score(self):
        assert LayoutScoringConfig().min_total_score == pytest.approx(0.0)

    def test_default_quality_filter_none(self):
        assert LayoutScoringConfig().quality_filter is None


# ─── make_layout_scoring_entry ────────────────────────────────────────────────

class TestMakeLayoutScoringEntryExtra:
    def test_returns_entry(self):
        e = make_layout_scoring_entry(0, 0.7, 0.9, 0.1, 0.8, 10)
        assert isinstance(e, LayoutScoringEntry)

    def test_quality_level_stored(self):
        e = make_layout_scoring_entry(0, 0.9, 0.95, 0.05, 0.9, 12, "excellent")
        assert e.quality_level == "excellent"

    def test_extra_params_stored(self):
        e = make_layout_scoring_entry(0, 0.7, 0.9, 0.1, 0.8, 10, threshold=0.5)
        assert e.params["threshold"] == pytest.approx(0.5)


# ─── summarise_layout_scoring_entries ─────────────────────────────────────────

class TestSummariseLayoutScoringEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_layout_scoring_entries([])
        assert s.n_layouts == 0 and s.best_layout_id is None

    def test_single_entry_best(self):
        e = _layout_entry(layout_id=7, total=0.8)
        s = summarise_layout_scoring_entries([e])
        assert s.best_layout_id == 7

    def test_quality_counts(self):
        entries = [_layout_entry(quality="good"), _layout_entry(quality="good"),
                   _layout_entry(quality="poor")]
        s = summarise_layout_scoring_entries(entries)
        assert s.quality_counts["good"] == 2 and s.quality_counts["poor"] == 1

    def test_mean_score(self):
        entries = [_layout_entry(total=0.4), _layout_entry(total=0.8)]
        s = summarise_layout_scoring_entries(entries)
        assert s.mean_total_score == pytest.approx(0.6)


# ─── filter layout ────────────────────────────────────────────────────────────

class TestFilterLayoutScoringExtra:
    def test_filter_by_min_score(self):
        entries = [_layout_entry(total=0.3), _layout_entry(total=0.9)]
        result = filter_layout_by_min_score(entries, 0.5)
        assert len(result) == 1

    def test_filter_by_quality(self):
        entries = [_layout_entry(quality="good"), _layout_entry(quality="poor")]
        result = filter_layout_by_quality(entries, "good")
        assert all(e.quality_level == "good" for e in result)

    def test_filter_by_max_overlap(self):
        entries = [_layout_entry(overlap=0.05), _layout_entry(overlap=0.4)]
        result = filter_layout_by_max_overlap(entries, 0.1)
        assert len(result) == 1

    def test_top_k_layout(self):
        entries = [_layout_entry(total=0.3), _layout_entry(total=0.9),
                   _layout_entry(total=0.6)]
        top = top_k_layout_entries(entries, 2)
        assert top[0].total_score == pytest.approx(0.9)

    def test_best_layout_empty(self):
        assert best_layout_entry([]) is None

    def test_best_layout(self):
        entries = [_layout_entry(total=0.3), _layout_entry(total=0.9)]
        best = best_layout_entry(entries)
        assert best.total_score == pytest.approx(0.9)


# ─── layout_score_stats / compare / batch ────────────────────────────────────

class TestLayoutScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = layout_score_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = layout_score_stats([_layout_entry(), _layout_entry()])
        assert s["count"] == 2

    def test_compare_returns_dict(self):
        s = summarise_layout_scoring_entries([_layout_entry()])
        d = compare_layout_scoring_summaries(s, s)
        assert isinstance(d, dict)

    def test_compare_delta_zero(self):
        s = summarise_layout_scoring_entries([_layout_entry()])
        d = compare_layout_scoring_summaries(s, s)
        assert d["delta_mean_total_score"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_layout_scoring_entries([[_layout_entry()], []])
        assert len(result) == 2
