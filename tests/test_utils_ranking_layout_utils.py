"""Tests for puzzle_reconstruction.utils.ranking_layout_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.ranking_layout_utils import (
    GlobalRankingConfig,
    GlobalRankingEntry,
    GlobalRankingSummary,
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
    LayoutScoringConfig,
    LayoutScoringEntry,
    LayoutScoringSummary,
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

np.random.seed(0)


def _make_entries(n=5):
    entries = []
    for i in range(n):
        e = make_global_ranking_entry(i, i + 1, score=float(i) / n, rank=n - i - 1)
        entries.append(e)
    return entries


def _make_layout_entries(n=4):
    levels = ["poor", "fair", "good", "excellent"]
    entries = []
    for i in range(n):
        e = make_layout_scoring_entry(
            layout_id=i,
            total_score=float(i) / n,
            coverage=0.5 + 0.1 * i,
            overlap_ratio=0.1 * i,
            uniformity=0.8,
            n_fragments=10 + i,
            quality_level=levels[i % len(levels)],
        )
        entries.append(e)
    return entries


# ── GlobalRankingConfig ───────────────────────────────────────────────────────

def test_global_ranking_config_defaults():
    cfg = GlobalRankingConfig()
    assert cfg.min_score == 0.0
    assert cfg.top_k == 10
    assert cfg.source_names == []


def test_global_ranking_config_custom():
    cfg = GlobalRankingConfig(min_score=0.5, top_k=3)
    assert cfg.top_k == 3


# ── make_global_ranking_entry ─────────────────────────────────────────────────

def test_make_global_ranking_entry_basic():
    e = make_global_ranking_entry(0, 1, 0.8, 0, color=0.7)
    assert e.idx1 == 0
    assert e.idx2 == 1
    assert e.score == pytest.approx(0.8)
    assert e.rank == 0
    assert e.component_scores["color"] == pytest.approx(0.7)


def test_make_global_ranking_entry_no_components():
    e = make_global_ranking_entry(2, 3, 0.5, 1)
    assert e.component_scores == {}


# ── summarise_global_ranking_entries ─────────────────────────────────────────

def test_summarise_empty():
    s = summarise_global_ranking_entries([])
    assert s.n_pairs == 0
    assert s.top_pair is None


def test_summarise_entries():
    entries = _make_entries(4)
    s = summarise_global_ranking_entries(entries)
    assert s.n_pairs == 4
    assert s.max_score == pytest.approx(max(e.score for e in entries))
    assert s.min_score == pytest.approx(min(e.score for e in entries))
    assert s.top_pair is not None


def test_summarise_single_entry():
    e = make_global_ranking_entry(0, 1, 1.0, 0)
    s = summarise_global_ranking_entries([e])
    assert s.mean_score == pytest.approx(1.0)
    assert s.top_pair == (0, 1)


# ── filter functions ──────────────────────────────────────────────────────────

def test_filter_ranking_by_min_score():
    entries = _make_entries(5)
    filtered = filter_ranking_by_min_score(entries, 0.5)
    assert all(e.score >= 0.5 for e in filtered)


def test_filter_ranking_by_fragment():
    entries = _make_entries(5)
    filtered = filter_ranking_by_fragment(entries, 0)
    assert all(e.idx1 == 0 or e.idx2 == 0 for e in filtered)


def test_filter_ranking_by_top_k():
    entries = _make_entries(5)
    top = filter_ranking_by_top_k(entries, 3)
    assert len(top) == 3
    ranks = [e.rank for e in top]
    assert ranks == sorted(ranks)


def test_top_k_ranking_entries():
    entries = _make_entries(5)
    top = top_k_ranking_entries(entries, 2)
    assert len(top) == 2
    assert top[0].score >= top[1].score


def test_best_ranking_entry_none():
    assert best_ranking_entry([]) is None


def test_best_ranking_entry():
    entries = _make_entries(4)
    best = best_ranking_entry(entries)
    assert best.score == max(e.score for e in entries)


# ── ranking_score_stats ───────────────────────────────────────────────────────

def test_ranking_score_stats_empty():
    stats = ranking_score_stats([])
    assert stats["count"] == 0


def test_ranking_score_stats():
    entries = _make_entries(4)
    stats = ranking_score_stats(entries)
    assert stats["count"] == 4
    assert stats["max"] >= stats["min"]
    assert stats["std"] >= 0.0


# ── compare_global_ranking_summaries ─────────────────────────────────────────

def test_compare_global_ranking_summaries():
    a = summarise_global_ranking_entries(_make_entries(3))
    b = summarise_global_ranking_entries(_make_entries(5))
    cmp = compare_global_ranking_summaries(a, b)
    assert "delta_mean_score" in cmp
    assert "delta_n_pairs" in cmp
    assert cmp["delta_n_pairs"] == 2


# ── batch_summarise_global_ranking_entries ────────────────────────────────────

def test_batch_summarise_global_ranking_entries():
    groups = [_make_entries(3), _make_entries(2)]
    summaries = batch_summarise_global_ranking_entries(groups)
    assert len(summaries) == 2
    assert summaries[0].n_pairs == 3


# ── LayoutScoringConfig ───────────────────────────────────────────────────────

def test_layout_scoring_config_defaults():
    cfg = LayoutScoringConfig()
    assert cfg.min_total_score == 0.0
    assert cfg.quality_filter is None


# ── make_layout_scoring_entry ─────────────────────────────────────────────────

def test_make_layout_scoring_entry():
    e = make_layout_scoring_entry(0, 0.8, 0.9, 0.05, 0.7, 10, "good", extra=1)
    assert e.layout_id == 0
    assert e.total_score == pytest.approx(0.8)
    assert e.quality_level == "good"
    assert e.params["extra"] == 1


# ── summarise_layout_scoring_entries ─────────────────────────────────────────

def test_summarise_layout_empty():
    s = summarise_layout_scoring_entries([])
    assert s.n_layouts == 0
    assert s.best_layout_id is None


def test_summarise_layout_entries():
    entries = _make_layout_entries(4)
    s = summarise_layout_scoring_entries(entries)
    assert s.n_layouts == 4
    assert s.best_layout_id is not None
    assert s.worst_layout_id is not None


# ── layout filter functions ───────────────────────────────────────────────────

def test_filter_layout_by_min_score():
    entries = _make_layout_entries(4)
    filtered = filter_layout_by_min_score(entries, 0.5)
    assert all(e.total_score >= 0.5 for e in filtered)


def test_filter_layout_by_quality():
    entries = _make_layout_entries(4)
    filtered = filter_layout_by_quality(entries, "good")
    assert all(e.quality_level == "good" for e in filtered)


def test_filter_layout_by_max_overlap():
    entries = _make_layout_entries(4)
    filtered = filter_layout_by_max_overlap(entries, 0.15)
    assert all(e.overlap_ratio <= 0.15 for e in filtered)


def test_top_k_layout_entries():
    entries = _make_layout_entries(4)
    top = top_k_layout_entries(entries, 2)
    assert len(top) == 2
    assert top[0].total_score >= top[1].total_score


def test_best_layout_entry_none():
    assert best_layout_entry([]) is None


def test_best_layout_entry():
    entries = _make_layout_entries(4)
    best = best_layout_entry(entries)
    assert best.total_score == max(e.total_score for e in entries)


# ── layout_score_stats ────────────────────────────────────────────────────────

def test_layout_score_stats_empty():
    stats = layout_score_stats([])
    assert stats["count"] == 0


def test_layout_score_stats():
    entries = _make_layout_entries(4)
    stats = layout_score_stats(entries)
    assert stats["count"] == 4
    assert stats["std"] >= 0.0


# ── compare_layout_scoring_summaries ─────────────────────────────────────────

def test_compare_layout_scoring_summaries():
    a = summarise_layout_scoring_entries(_make_layout_entries(2))
    b = summarise_layout_scoring_entries(_make_layout_entries(4))
    cmp = compare_layout_scoring_summaries(a, b)
    assert "delta_mean_total_score" in cmp
    assert "delta_n_layouts" in cmp
    assert cmp["delta_n_layouts"] == 2


# ── batch_summarise_layout_scoring_entries ────────────────────────────────────

def test_batch_summarise_layout_entries():
    groups = [_make_layout_entries(2), _make_layout_entries(3)]
    summaries = batch_summarise_layout_scoring_entries(groups)
    assert len(summaries) == 2
    assert summaries[1].n_layouts == 3
