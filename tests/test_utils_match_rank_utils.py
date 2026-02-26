"""Tests for puzzle_reconstruction.utils.match_rank_utils"""
import pytest
from puzzle_reconstruction.utils.match_rank_utils import (
    RankingConfig,
    RankingEntry,
    RankingSummary,
    EvalResultConfig,
    EvalResultEntry,
    EvalResultSummary,
    make_ranking_entry,
    summarise_ranking_entries,
    filter_ranking_by_algorithm,
    filter_ranking_by_min_top_score,
    filter_ranking_by_min_acceptance,
    top_k_ranking_entries,
    best_ranking_entry,
    ranking_score_stats,
    compare_ranking_summaries,
    batch_summarise_ranking_entries,
    make_eval_result_entry,
    summarise_eval_result_entries,
    filter_eval_by_min_f1,
    filter_eval_by_algorithm,
    top_k_eval_entries,
    best_eval_entry,
    eval_f1_stats,
    compare_eval_summaries,
    batch_summarise_eval_entries,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_ranking_entries():
    return [
        make_ranking_entry(0, n_pairs=10, n_accepted=8, top_score=0.9, mean_score=0.75, algorithm="greedy"),
        make_ranking_entry(1, n_pairs=20, n_accepted=12, top_score=0.7, mean_score=0.6, algorithm="greedy"),
        make_ranking_entry(2, n_pairs=15, n_accepted=5, top_score=0.5, mean_score=0.4, algorithm="ranked"),
    ]


def make_eval_entries():
    return [
        make_eval_result_entry(0, n_pairs=10, mean_score=0.8, mean_f1=0.75, best_f1=0.9, algorithm="iou"),
        make_eval_result_entry(1, n_pairs=15, mean_score=0.6, mean_f1=0.55, best_f1=0.7, algorithm="iou"),
        make_eval_result_entry(2, n_pairs=12, mean_score=0.7, mean_f1=0.65, best_f1=0.8, algorithm="nms"),
    ]


# ─── RankingConfig ────────────────────────────────────────────────────────────

def test_ranking_config_defaults():
    cfg = RankingConfig()
    assert cfg.min_score == 0.0
    assert cfg.top_k == 10
    assert cfg.deduplicate is False


# ─── RankingEntry ─────────────────────────────────────────────────────────────

def test_ranking_entry_acceptance_rate():
    e = make_ranking_entry(0, n_pairs=10, n_accepted=7, top_score=0.9, mean_score=0.7, algorithm="a")
    assert abs(e.acceptance_rate - 0.7) < 1e-9


def test_ranking_entry_zero_pairs():
    e = make_ranking_entry(0, n_pairs=0, n_accepted=0, top_score=0.0, mean_score=0.0, algorithm="a")
    assert e.acceptance_rate == 0.0


def test_ranking_entry_types():
    e = make_ranking_entry(1, "20", "15", "0.9", "0.7", "algo", extra=True)
    assert isinstance(e.n_pairs, int)
    assert isinstance(e.top_score, float)
    assert e.params == {"extra": True}


# ─── summarise_ranking_entries ────────────────────────────────────────────────

def test_summarise_ranking_empty():
    s = summarise_ranking_entries([])
    assert s.n_batches == 0
    assert s.best_batch_id is None
    assert s.worst_batch_id is None


def test_summarise_ranking_basic():
    entries = make_ranking_entries()
    s = summarise_ranking_entries(entries)
    assert s.n_batches == 3
    assert s.best_batch_id == 0  # top_score=0.9
    assert s.worst_batch_id == 2  # top_score=0.5
    assert s.total_pairs == 45
    assert s.total_accepted == 25


def test_summarise_ranking_mean_top_score():
    entries = make_ranking_entries()
    s = summarise_ranking_entries(entries)
    assert abs(s.mean_top_score - (0.9 + 0.7 + 0.5) / 3) < 1e-9


# ─── filter_ranking_by_algorithm ──────────────────────────────────────────────

def test_filter_by_algorithm():
    entries = make_ranking_entries()
    filtered = filter_ranking_by_algorithm(entries, "greedy")
    assert len(filtered) == 2


def test_filter_by_algorithm_none():
    entries = make_ranking_entries()
    filtered = filter_ranking_by_algorithm(entries, "nonexistent")
    assert len(filtered) == 0


# ─── filter_ranking_by_min_top_score ─────────────────────────────────────────

def test_filter_by_min_top_score():
    entries = make_ranking_entries()
    filtered = filter_ranking_by_min_top_score(entries, 0.7)
    assert len(filtered) == 2


# ─── filter_ranking_by_min_acceptance ────────────────────────────────────────

def test_filter_by_min_acceptance():
    entries = make_ranking_entries()
    # acceptance_rates: 0.8, 0.6, 0.333
    filtered = filter_ranking_by_min_acceptance(entries, 0.5)
    assert len(filtered) == 2


# ─── top_k_ranking_entries ────────────────────────────────────────────────────

def test_top_k_ranking_entries():
    entries = make_ranking_entries()
    top = top_k_ranking_entries(entries, 2)
    assert len(top) == 2
    assert top[0].top_score >= top[1].top_score


def test_top_k_ranking_more_than_available():
    entries = make_ranking_entries()
    top = top_k_ranking_entries(entries, 10)
    assert len(top) == 3


# ─── best_ranking_entry ───────────────────────────────────────────────────────

def test_best_ranking_entry():
    entries = make_ranking_entries()
    best = best_ranking_entry(entries)
    assert best.batch_id == 0


def test_best_ranking_entry_empty():
    assert best_ranking_entry([]) is None


# ─── ranking_score_stats ──────────────────────────────────────────────────────

def test_ranking_score_stats_empty():
    d = ranking_score_stats([])
    assert d["count"] == 0


def test_ranking_score_stats_basic():
    entries = make_ranking_entries()
    d = ranking_score_stats(entries)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(0.5)
    assert d["max"] == pytest.approx(0.9)
    assert d["std"] >= 0.0


# ─── compare_ranking_summaries ────────────────────────────────────────────────

def test_compare_ranking_summaries():
    entries = make_ranking_entries()
    a = summarise_ranking_entries(entries[:2])
    b = summarise_ranking_entries(entries)
    diff = compare_ranking_summaries(a, b)
    assert "delta_mean_top_score" in diff
    assert "same_best" in diff


# ─── batch_summarise_ranking_entries ─────────────────────────────────────────

def test_batch_summarise_ranking():
    entries = make_ranking_entries()
    result = batch_summarise_ranking_entries([entries[:2], entries[2:]])
    assert len(result) == 2
    assert result[0].n_batches == 2


# ─── EvalResultEntry ──────────────────────────────────────────────────────────

def test_make_eval_result_entry():
    e = make_eval_result_entry(0, 10, 0.8, 0.75, 0.9, "iou", k=5)
    assert isinstance(e, EvalResultEntry)
    assert e.params == {"k": 5}


# ─── summarise_eval_result_entries ────────────────────────────────────────────

def test_summarise_eval_empty():
    s = summarise_eval_result_entries([])
    assert s.n_runs == 0
    assert s.best_run_id is None


def test_summarise_eval_basic():
    entries = make_eval_entries()
    s = summarise_eval_result_entries(entries)
    assert s.n_runs == 3
    assert s.total_pairs == 37
    assert s.best_run_id == 0  # best_f1=0.9
    assert s.worst_run_id == 1  # best_f1=0.7


# ─── filter_eval_by_min_f1 ────────────────────────────────────────────────────

def test_filter_eval_by_min_f1():
    entries = make_eval_entries()
    filtered = filter_eval_by_min_f1(entries, 0.6)
    assert all(e.mean_f1 >= 0.6 for e in filtered)


# ─── filter_eval_by_algorithm ─────────────────────────────────────────────────

def test_filter_eval_by_algorithm():
    entries = make_eval_entries()
    filtered = filter_eval_by_algorithm(entries, "iou")
    assert len(filtered) == 2


# ─── top_k_eval_entries ───────────────────────────────────────────────────────

def test_top_k_eval_entries():
    entries = make_eval_entries()
    top = top_k_eval_entries(entries, 2)
    assert len(top) == 2
    assert top[0].best_f1 >= top[1].best_f1


# ─── best_eval_entry ──────────────────────────────────────────────────────────

def test_best_eval_entry():
    entries = make_eval_entries()
    best = best_eval_entry(entries)
    assert best.run_id == 0


def test_best_eval_entry_empty():
    assert best_eval_entry([]) is None


# ─── eval_f1_stats ────────────────────────────────────────────────────────────

def test_eval_f1_stats():
    entries = make_eval_entries()
    d = eval_f1_stats(entries)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(0.55)
    assert d["max"] == pytest.approx(0.75)


def test_eval_f1_stats_empty():
    d = eval_f1_stats([])
    assert d["count"] == 0


# ─── compare_eval_summaries ───────────────────────────────────────────────────

def test_compare_eval_summaries():
    entries = make_eval_entries()
    a = summarise_eval_result_entries(entries[:2])
    b = summarise_eval_result_entries(entries)
    diff = compare_eval_summaries(a, b)
    assert "delta_mean_f1" in diff
    assert "same_best" in diff


# ─── batch_summarise_eval_entries ─────────────────────────────────────────────

def test_batch_summarise_eval():
    entries = make_eval_entries()
    result = batch_summarise_eval_entries([entries[:2], entries[2:]])
    assert len(result) == 2
