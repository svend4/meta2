"""Tests for puzzle_reconstruction.utils.candidate_rank_utils"""
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


# ─── CandidateRankConfig ──────────────────────────────────────────────────────

def test_candidate_rank_config_defaults():
    cfg = CandidateRankConfig()
    assert cfg.min_score == 0.5
    assert cfg.max_pairs == 0
    assert cfg.deduplicate is True


def test_candidate_rank_config_invalid_min_score():
    with pytest.raises(ValueError, match="min_score"):
        CandidateRankConfig(min_score=1.5)


def test_candidate_rank_config_invalid_max_pairs():
    with pytest.raises(ValueError, match="max_pairs"):
        CandidateRankConfig(max_pairs=-1)


def test_candidate_rank_config_boundary():
    cfg = CandidateRankConfig(min_score=0.0)
    assert cfg.min_score == 0.0
    cfg2 = CandidateRankConfig(min_score=1.0)
    assert cfg2.min_score == 1.0


# ─── make_candidate_entry ─────────────────────────────────────────────────────

def test_make_candidate_entry_selected():
    cfg = CandidateRankConfig(min_score=0.5)
    e = make_candidate_entry(0, 1, score=0.8, rank=0, cfg=cfg)
    assert e.is_selected is True
    assert e.score == pytest.approx(0.8)


def test_make_candidate_entry_rejected():
    cfg = CandidateRankConfig(min_score=0.5)
    e = make_candidate_entry(0, 1, score=0.3, rank=1, cfg=cfg)
    assert e.is_selected is False


def test_make_candidate_entry_at_threshold():
    cfg = CandidateRankConfig(min_score=0.5)
    e = make_candidate_entry(0, 1, score=0.5, rank=0, cfg=cfg)
    assert e.is_selected is True


def test_make_candidate_entry_meta():
    e = make_candidate_entry(2, 3, score=0.7, rank=0, meta={"extra": 42})
    assert e.meta["extra"] == 42


# ─── entries_from_pairs ───────────────────────────────────────────────────────

def test_entries_from_pairs_ranking_order():
    pairs = [
        {"idx1": 0, "idx2": 1, "score": 0.3},
        {"idx1": 2, "idx2": 3, "score": 0.9},
        {"idx1": 4, "idx2": 5, "score": 0.6},
    ]
    entries = entries_from_pairs(pairs)
    assert entries[0].score == pytest.approx(0.9)
    assert entries[1].score == pytest.approx(0.6)
    assert entries[2].score == pytest.approx(0.3)


def test_entries_from_pairs_rank_assigned():
    pairs = [{"idx1": 0, "idx2": 1, "score": 0.7}]
    entries = entries_from_pairs(pairs)
    assert entries[0].rank == 0


def test_entries_from_pairs_empty():
    entries = entries_from_pairs([])
    assert entries == []


# ─── summarise_rankings ───────────────────────────────────────────────────────

def test_summarise_rankings_empty():
    s = summarise_rankings([])
    assert s.n_total == 0
    assert s.mean_score == 0.0


def test_summarise_rankings_basic():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = [
        make_candidate_entry(0, 1, 0.9, 0, cfg),
        make_candidate_entry(2, 3, 0.3, 1, cfg),
    ]
    s = summarise_rankings(entries)
    assert s.n_total == 2
    assert s.n_selected == 1
    assert s.n_rejected == 1
    assert s.mean_score == pytest.approx(0.6)
    assert s.max_score == pytest.approx(0.9)
    assert s.min_score == pytest.approx(0.3)


# ─── filter_selected / filter_rejected_candidates ────────────────────────────

def test_filter_selected():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = [
        make_candidate_entry(0, 1, 0.9, 0, cfg),
        make_candidate_entry(2, 3, 0.2, 1, cfg),
        make_candidate_entry(4, 5, 0.7, 2, cfg),
    ]
    sel = filter_selected(entries)
    assert all(e.is_selected for e in sel)
    assert len(sel) == 2


def test_filter_rejected_candidates():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = [
        make_candidate_entry(0, 1, 0.9, 0, cfg),
        make_candidate_entry(2, 3, 0.2, 1, cfg),
    ]
    rej = filter_rejected_candidates(entries)
    assert len(rej) == 1
    assert rej[0].score == pytest.approx(0.2)


# ─── filter_by_score_range ───────────────────────────────────────────────────

def test_filter_by_score_range():
    entries = [
        make_candidate_entry(0, 1, 0.1, 0),
        make_candidate_entry(2, 3, 0.5, 1),
        make_candidate_entry(4, 5, 0.9, 2),
    ]
    result = filter_by_score_range(entries, min_score=0.4, max_score=0.8)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.5)


# ─── filter_by_rank ──────────────────────────────────────────────────────────

def test_filter_by_rank():
    entries = [
        make_candidate_entry(0, 1, 0.9, 0),
        make_candidate_entry(2, 3, 0.8, 1),
        make_candidate_entry(4, 5, 0.7, 2),
    ]
    result = filter_by_rank(entries, max_rank=1)
    assert len(result) == 2


# ─── top_k_candidate_entries ─────────────────────────────────────────────────

def test_top_k_candidate_entries():
    entries = [
        make_candidate_entry(0, 1, 0.3, 2),
        make_candidate_entry(2, 3, 0.9, 0),
        make_candidate_entry(4, 5, 0.6, 1),
    ]
    top2 = top_k_candidate_entries(entries, 2)
    assert len(top2) == 2
    assert top2[0].score == pytest.approx(0.9)


def test_top_k_candidate_entries_k_zero():
    entries = [make_candidate_entry(0, 1, 0.9, 0)]
    result = top_k_candidate_entries(entries, 0)
    assert result == []


# ─── candidate_rank_stats ────────────────────────────────────────────────────

def test_candidate_rank_stats_empty():
    stats = candidate_rank_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_candidate_rank_stats_values():
    cfg = CandidateRankConfig(min_score=0.5)
    entries = [
        make_candidate_entry(0, 1, 0.4, 0, cfg),
        make_candidate_entry(2, 3, 0.8, 1, cfg),
    ]
    stats = candidate_rank_stats(entries)
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(0.6)
    assert stats["min"] == pytest.approx(0.4)
    assert stats["max"] == pytest.approx(0.8)
    assert stats["n_selected"] == 1
    assert stats["n_rejected"] == 1


# ─── compare_rankings ────────────────────────────────────────────────────────

def test_compare_rankings():
    entries_a = [make_candidate_entry(0, 1, 0.9, 0), make_candidate_entry(2, 3, 0.7, 1)]
    entries_b = [make_candidate_entry(4, 5, 0.5, 0)]
    s_a = summarise_rankings(entries_a)
    s_b = summarise_rankings(entries_b)
    diff = compare_rankings(s_a, s_b)
    assert diff["n_total_delta"] == 1
    assert diff["n_selected_delta"] == 1


# ─── batch_summarise_rankings ────────────────────────────────────────────────

def test_batch_summarise_rankings():
    pair_lists = [
        [{"idx1": 0, "idx2": 1, "score": 0.9}],
        [{"idx1": 2, "idx2": 3, "score": 0.4}],
    ]
    results = batch_summarise_rankings(pair_lists)
    assert len(results) == 2
    assert results[0].n_total == 1
    assert results[1].n_total == 1
