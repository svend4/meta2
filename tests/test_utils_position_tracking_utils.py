"""Tests for puzzle_reconstruction.utils.position_tracking_utils"""
import pytest
from puzzle_reconstruction.utils.position_tracking_utils import (
    PositionQualityRecord,
    PositionQualitySummary,
    AssemblyHistoryEntry,
    AssemblyHistorySummary,
    make_position_quality_record,
    summarise_position_quality,
    filter_by_placement_rate,
    filter_by_method,
    top_k_position_records,
    best_position_record,
    position_quality_stats,
    make_assembly_history_entry,
    summarise_assembly_history,
    filter_converged,
    filter_by_min_best_score,
    top_k_assembly_entries,
    best_assembly_entry,
    assembly_score_stats,
    compare_assembly_summaries,
    batch_summarise_assembly_history,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_pos_records():
    return [
        make_position_quality_record(0, 10, 8, 0.9, 0.85, "grid"),
        make_position_quality_record(1, 10, 6, 0.7, 0.65, "grid"),
        make_position_quality_record(2, 10, 10, 0.95, 0.95, "refine"),
    ]


def make_history_entries():
    return [
        make_assembly_history_entry(0, 50, 0.8, True, 30, "greedy"),
        make_assembly_history_entry(1, 100, 0.6, False, None, "annealing"),
        make_assembly_history_entry(2, 75, 0.9, True, 60, "greedy"),
    ]


# ─── PositionQualityRecord ────────────────────────────────────────────────────

def test_make_position_quality_record_basic():
    r = make_position_quality_record(0, 10, 8, 0.9, 0.85, "grid")
    assert r.run_id == 0
    assert r.n_fragments == 10
    assert r.method == "grid"


def test_placement_rate_basic():
    r = make_position_quality_record(0, 10, 7, 0.8, 0.7, "grid")
    assert abs(r.placement_rate - 0.7) < 1e-9


def test_placement_rate_zero_fragments():
    r = make_position_quality_record(0, 0, 0, 0.0, 0.0, "grid")
    assert r.placement_rate == 0.0


def test_placement_rate_all_placed():
    r = make_position_quality_record(0, 5, 5, 1.0, 1.0, "refine")
    assert r.placement_rate == 1.0


def test_position_quality_record_params():
    r = make_position_quality_record(0, 5, 5, 1.0, 1.0, "refine", alpha=0.5)
    assert r.params == {"alpha": 0.5}


# ─── summarise_position_quality ───────────────────────────────────────────────

def test_summarise_position_quality_empty():
    s = summarise_position_quality([])
    assert s.n_runs == 0
    assert s.best_run_id is None
    assert s.worst_run_id is None


def test_summarise_position_quality_basic():
    records = make_pos_records()
    s = summarise_position_quality(records)
    assert s.n_runs == 3
    assert s.best_run_id == 2  # coverage=0.95
    assert s.worst_run_id == 1  # coverage=0.65
    assert s.total_fragments == 30


def test_summarise_mean_confidence():
    records = make_pos_records()
    s = summarise_position_quality(records)
    expected = (0.9 + 0.7 + 0.95) / 3
    assert abs(s.mean_confidence - expected) < 1e-9


def test_summarise_mean_coverage():
    records = make_pos_records()
    s = summarise_position_quality(records)
    expected = (0.85 + 0.65 + 0.95) / 3
    assert abs(s.mean_coverage - expected) < 1e-9


# ─── filter_by_placement_rate ─────────────────────────────────────────────────

def test_filter_by_placement_rate():
    records = make_pos_records()
    # rates: 0.8, 0.6, 1.0
    filtered = filter_by_placement_rate(records, 0.8)
    assert len(filtered) == 2


def test_filter_by_placement_rate_none():
    records = make_pos_records()
    filtered = filter_by_placement_rate(records, 1.1)
    assert len(filtered) == 0


# ─── filter_by_method ─────────────────────────────────────────────────────────

def test_filter_by_method():
    records = make_pos_records()
    filtered = filter_by_method(records, "grid")
    assert len(filtered) == 2


def test_filter_by_method_none():
    records = make_pos_records()
    filtered = filter_by_method(records, "unknown")
    assert len(filtered) == 0


# ─── top_k_position_records ───────────────────────────────────────────────────

def test_top_k_position_records():
    records = make_pos_records()
    top = top_k_position_records(records, 2)
    assert len(top) == 2
    assert top[0].canvas_coverage >= top[1].canvas_coverage


def test_top_k_position_records_more():
    records = make_pos_records()
    top = top_k_position_records(records, 10)
    assert len(top) == 3


# ─── best_position_record ────────────────────────────────────────────────────

def test_best_position_record():
    records = make_pos_records()
    best = best_position_record(records)
    assert best.run_id == 2


def test_best_position_record_empty():
    assert best_position_record([]) is None


# ─── position_quality_stats ───────────────────────────────────────────────────

def test_position_quality_stats_empty():
    d = position_quality_stats([])
    assert d["count"] == 0


def test_position_quality_stats_basic():
    records = make_pos_records()
    d = position_quality_stats(records)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(0.65)
    assert d["max"] == pytest.approx(0.95)
    assert d["std"] >= 0.0


# ─── AssemblyHistoryEntry ─────────────────────────────────────────────────────

def test_make_assembly_history_entry():
    e = make_assembly_history_entry(0, 50, 0.8, True, 30, "greedy", lr=0.01)
    assert e.run_id == 0
    assert e.converged is True
    assert e.params == {"lr": 0.01}


def test_assembly_history_entry_not_converged():
    e = make_assembly_history_entry(1, 100, 0.5, False, None, "annealing")
    assert e.converged is False
    assert e.convergence_iter is None


# ─── summarise_assembly_history ───────────────────────────────────────────────

def test_summarise_assembly_history_empty():
    s = summarise_assembly_history([])
    assert s.n_runs == 0
    assert s.best_run_id is None


def test_summarise_assembly_history_basic():
    entries = make_history_entries()
    s = summarise_assembly_history(entries)
    assert s.n_runs == 3
    assert s.n_converged == 2
    assert abs(s.convergence_rate - 2/3) < 1e-9
    assert s.best_run_id == 2  # best_score=0.9


def test_summarise_mean_best_score():
    entries = make_history_entries()
    s = summarise_assembly_history(entries)
    assert abs(s.mean_best_score - (0.8 + 0.6 + 0.9) / 3) < 1e-9


# ─── filter_converged ────────────────────────────────────────────────────────

def test_filter_converged():
    entries = make_history_entries()
    filtered = filter_converged(entries)
    assert len(filtered) == 2
    assert all(e.converged for e in filtered)


# ─── filter_by_min_best_score ────────────────────────────────────────────────

def test_filter_by_min_best_score():
    entries = make_history_entries()
    filtered = filter_by_min_best_score(entries, 0.75)
    assert len(filtered) == 2


# ─── top_k_assembly_entries ───────────────────────────────────────────────────

def test_top_k_assembly_entries():
    entries = make_history_entries()
    top = top_k_assembly_entries(entries, 2)
    assert len(top) == 2
    assert top[0].best_score >= top[1].best_score


# ─── best_assembly_entry ──────────────────────────────────────────────────────

def test_best_assembly_entry():
    entries = make_history_entries()
    best = best_assembly_entry(entries)
    assert best.run_id == 2


def test_best_assembly_entry_empty():
    assert best_assembly_entry([]) is None


# ─── assembly_score_stats ─────────────────────────────────────────────────────

def test_assembly_score_stats_empty():
    d = assembly_score_stats([])
    assert d["count"] == 0


def test_assembly_score_stats_basic():
    entries = make_history_entries()
    d = assembly_score_stats(entries)
    assert d["count"] == 3
    assert d["min"] == pytest.approx(0.6)
    assert d["max"] == pytest.approx(0.9)


# ─── compare_assembly_summaries ───────────────────────────────────────────────

def test_compare_assembly_summaries():
    entries = make_history_entries()
    a = summarise_assembly_history(entries[:2])
    b = summarise_assembly_history(entries)
    diff = compare_assembly_summaries(a, b)
    assert "delta_mean_best_score" in diff
    assert "delta_convergence_rate" in diff


# ─── batch_summarise_assembly_history ────────────────────────────────────────

def test_batch_summarise_assembly_history():
    entries = make_history_entries()
    result = batch_summarise_assembly_history([entries[:2], entries[2:]])
    assert len(result) == 2
    assert result[0].n_runs == 2
