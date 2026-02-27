"""Tests for puzzle_reconstruction.utils.seq_gap_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.seq_gap_utils import (
    SequenceScoreConfig,
    SequenceScoreEntry,
    SequenceScoreSummary,
    make_sequence_score_entry,
    summarise_sequence_score_entries,
    filter_full_sequences,
    filter_sequence_by_min_score,
    filter_sequence_by_algorithm,
    top_k_sequence_entries,
    best_sequence_entry,
    sequence_score_stats,
    compare_sequence_summaries,
    batch_summarise_sequence_score_entries,
    GapScoreConfig,
    GapScoreEntry,
    GapScoreSummary,
    make_gap_score_entry,
    summarise_gap_score_entries,
    filter_overlapping_gaps,
    filter_gap_by_category,
    filter_gap_by_max_distance,
    top_k_closest_gaps,
    best_gap_entry,
    gap_score_stats,
    compare_gap_summaries,
    batch_summarise_gap_score_entries,
)

np.random.seed(21)


def _make_seq_entries(n=5):
    algos = ["greedy", "beam", "genetic"]
    return [
        make_sequence_score_entry(
            seq_id=i, order=list(range(i + 1)),
            total_score=float(i) / n, n_fragments=i + 1,
            algorithm=algos[i % len(algos)],
            is_full=(i % 2 == 0),
        )
        for i in range(n)
    ]


def _make_gap_entries(n=5):
    cats = ["overlap", "touching", "near", "far"]
    return [
        make_gap_score_entry(
            id1=i, id2=i + 1,
            gap_x=float(i), gap_y=float(i),
            distance=float(i) * 2.0,
            category=cats[i % len(cats)],
        )
        for i in range(n)
    ]


# ── SequenceScoreConfig ───────────────────────────────────────────────────────

def test_sequence_score_config_defaults():
    cfg = SequenceScoreConfig()
    assert cfg.min_score == 0.0
    assert cfg.require_full is True


# ── make_sequence_score_entry ─────────────────────────────────────────────────

def test_make_sequence_score_entry_basic():
    e = make_sequence_score_entry(0, [1, 2, 3], 0.9, 3, "greedy", True)
    assert e.seq_id == 0
    assert e.total_score == pytest.approx(0.9)
    assert e.n_fragments == 3
    assert e.algorithm == "greedy"
    assert e.is_full is True


# ── summarise_sequence_score_entries ─────────────────────────────────────────

def test_summarise_seq_empty():
    s = summarise_sequence_score_entries([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0
    assert s.algorithms == []


def test_summarise_seq_entries():
    entries = _make_seq_entries(5)
    s = summarise_sequence_score_entries(entries)
    assert s.n_entries == 5
    assert s.min_score <= s.max_score
    assert len(s.algorithms) > 0


def test_summarise_seq_full_count():
    entries = _make_seq_entries(5)
    s = summarise_sequence_score_entries(entries)
    assert s.n_full == sum(1 for e in entries if e.is_full)


# ── filter functions (sequence) ───────────────────────────────────────────────

def test_filter_full_sequences():
    entries = _make_seq_entries(5)
    full = filter_full_sequences(entries)
    assert all(e.is_full for e in full)


def test_filter_sequence_by_min_score():
    entries = _make_seq_entries(5)
    filtered = filter_sequence_by_min_score(entries, 0.5)
    assert all(e.total_score >= 0.5 for e in filtered)


def test_filter_sequence_by_algorithm():
    entries = _make_seq_entries(6)
    filtered = filter_sequence_by_algorithm(entries, "greedy")
    assert all(e.algorithm == "greedy" for e in filtered)


def test_top_k_sequence_entries():
    entries = _make_seq_entries(5)
    top = top_k_sequence_entries(entries, 3)
    assert len(top) == 3
    assert top[0].total_score >= top[-1].total_score


def test_best_sequence_entry_none():
    assert best_sequence_entry([]) is None


def test_best_sequence_entry():
    entries = _make_seq_entries(4)
    best = best_sequence_entry(entries)
    assert best.total_score == max(e.total_score for e in entries)


# ── sequence_score_stats ──────────────────────────────────────────────────────

def test_sequence_score_stats_empty():
    stats = sequence_score_stats([])
    assert stats["count"] == 0


def test_sequence_score_stats():
    entries = _make_seq_entries(5)
    stats = sequence_score_stats(entries)
    assert stats["count"] == 5.0
    assert stats["min"] <= stats["max"]


# ── compare_sequence_summaries ────────────────────────────────────────────────

def test_compare_sequence_summaries():
    a = summarise_sequence_score_entries(_make_seq_entries(3))
    b = summarise_sequence_score_entries(_make_seq_entries(5))
    cmp = compare_sequence_summaries(a, b)
    assert "mean_score_delta" in cmp
    assert "n_full_delta" in cmp


# ── batch_summarise_sequence_score_entries ────────────────────────────────────

def test_batch_summarise_seq():
    groups = [_make_seq_entries(2), _make_seq_entries(4)]
    summaries = batch_summarise_sequence_score_entries(groups)
    assert len(summaries) == 2
    assert summaries[1].n_entries == 4


# ── GapScoreConfig ────────────────────────────────────────────────────────────

def test_gap_score_config_defaults():
    cfg = GapScoreConfig()
    assert cfg.near_threshold == pytest.approx(10.0)
    assert cfg.overlap_penalty == pytest.approx(1.0)


# ── make_gap_score_entry ──────────────────────────────────────────────────────

def test_make_gap_score_entry_basic():
    e = make_gap_score_entry(0, 1, 2.0, 3.0, 5.0, "near")
    assert e.id1 == 0
    assert e.id2 == 1
    assert e.gap_x == pytest.approx(2.0)
    assert e.distance == pytest.approx(5.0)
    assert e.category == "near"


# ── summarise_gap_score_entries ───────────────────────────────────────────────

def test_summarise_gap_empty():
    s = summarise_gap_score_entries([])
    assert s.n_entries == 0
    assert s.mean_distance == 0.0


def test_summarise_gap_entries():
    entries = _make_gap_entries(5)
    s = summarise_gap_score_entries(entries)
    assert s.n_entries == 5
    assert s.min_distance <= s.max_distance


def test_summarise_gap_category_counts():
    entries = _make_gap_entries(8)
    s = summarise_gap_score_entries(entries)
    total = s.n_overlapping + s.n_touching + s.n_near + s.n_far
    assert total == 8


# ── filter functions (gap) ────────────────────────────────────────────────────

def test_filter_overlapping_gaps():
    entries = _make_gap_entries(5)
    overlapping = filter_overlapping_gaps(entries)
    assert all(e.category == "overlap" for e in overlapping)


def test_filter_gap_by_category():
    entries = _make_gap_entries(5)
    near = filter_gap_by_category(entries, "near")
    assert all(e.category == "near" for e in near)


def test_filter_gap_by_max_distance():
    entries = _make_gap_entries(5)
    filtered = filter_gap_by_max_distance(entries, 5.0)
    assert all(e.distance <= 5.0 for e in filtered)


def test_top_k_closest_gaps():
    entries = _make_gap_entries(5)
    top = top_k_closest_gaps(entries, 2)
    assert len(top) == 2
    assert top[0].distance <= top[1].distance


def test_best_gap_entry_none():
    assert best_gap_entry([]) is None


def test_best_gap_entry():
    entries = _make_gap_entries(4)
    best = best_gap_entry(entries)
    assert best.distance == min(e.distance for e in entries)


# ── gap_score_stats ───────────────────────────────────────────────────────────

def test_gap_score_stats_empty():
    stats = gap_score_stats([])
    assert stats["count"] == 0


def test_gap_score_stats():
    entries = _make_gap_entries(4)
    stats = gap_score_stats(entries)
    assert stats["count"] == 4.0
    assert stats["min"] <= stats["max"]


# ── compare / batch ───────────────────────────────────────────────────────────

def test_compare_gap_summaries():
    a = summarise_gap_score_entries(_make_gap_entries(3))
    b = summarise_gap_score_entries(_make_gap_entries(5))
    cmp = compare_gap_summaries(a, b)
    assert "mean_distance_delta" in cmp
    assert "n_overlapping_delta" in cmp


def test_batch_summarise_gap():
    groups = [_make_gap_entries(2), _make_gap_entries(3)]
    summaries = batch_summarise_gap_score_entries(groups)
    assert len(summaries) == 2
    assert summaries[0].n_entries == 2
