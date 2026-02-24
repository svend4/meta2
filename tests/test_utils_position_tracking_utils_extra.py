"""Extra tests for puzzle_reconstruction/utils/position_tracking_utils.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pqr(run_id=0, n_frags=10, n_placed=8, confidence=0.9,
         coverage=0.7, method="default") -> PositionQualityRecord:
    return PositionQualityRecord(run_id=run_id, n_fragments=n_frags,
                                  n_placed=n_placed, mean_confidence=confidence,
                                  canvas_coverage=coverage, method=method)


def _ahe(run_id=0, n_iter=100, best_score=0.8, converged=True,
         convergence_iter=50, method="greedy") -> AssemblyHistoryEntry:
    return AssemblyHistoryEntry(run_id=run_id, n_iterations=n_iter,
                                best_score=best_score, converged=converged,
                                convergence_iter=convergence_iter, method=method)


# ─── PositionQualityRecord ────────────────────────────────────────────────────

class TestPositionQualityRecordExtra:
    def test_placement_rate_full(self):
        r = _pqr(n_frags=10, n_placed=10)
        assert r.placement_rate == pytest.approx(1.0)

    def test_placement_rate_partial(self):
        r = _pqr(n_frags=10, n_placed=5)
        assert r.placement_rate == pytest.approx(0.5)

    def test_placement_rate_zero_frags(self):
        r = _pqr(n_frags=0, n_placed=0)
        assert r.placement_rate == pytest.approx(0.0)

    def test_fields_stored(self):
        r = _pqr(run_id=42, method="custom")
        assert r.run_id == 42 and r.method == "custom"


# ─── make_position_quality_record ────────────────────────────────────────────

class TestMakePositionQualityRecordExtra:
    def test_returns_record(self):
        r = make_position_quality_record(0, 10, 8, 0.9, 0.7, "default")
        assert isinstance(r, PositionQualityRecord)

    def test_params_stored(self):
        r = make_position_quality_record(0, 10, 8, 0.9, 0.7, "default",
                                          threshold=0.5)
        assert r.params["threshold"] == pytest.approx(0.5)

    def test_empty_params_default(self):
        r = make_position_quality_record(1, 5, 3, 0.8, 0.6, "test")
        assert r.params == {}


# ─── summarise_position_quality ───────────────────────────────────────────────

class TestSummarisePositionQualityExtra:
    def test_empty_returns_summary(self):
        s = summarise_position_quality([])
        assert s.n_runs == 0 and s.best_run_id is None

    def test_n_runs(self):
        records = [_pqr(run_id=0), _pqr(run_id=1)]
        s = summarise_position_quality(records)
        assert s.n_runs == 2

    def test_best_run_id(self):
        records = [_pqr(run_id=0, coverage=0.3), _pqr(run_id=1, coverage=0.9)]
        s = summarise_position_quality(records)
        assert s.best_run_id == 1

    def test_worst_run_id(self):
        records = [_pqr(run_id=0, coverage=0.3), _pqr(run_id=1, coverage=0.9)]
        s = summarise_position_quality(records)
        assert s.worst_run_id == 0

    def test_total_fragments(self):
        records = [_pqr(n_frags=5), _pqr(n_frags=7)]
        s = summarise_position_quality(records)
        assert s.total_fragments == 12


# ─── filters / ranking ───────────────────────────────────────────────────────

class TestFilterPositionQualityExtra:
    def test_filter_by_placement_rate(self):
        records = [_pqr(n_frags=10, n_placed=3), _pqr(n_frags=10, n_placed=9)]
        result = filter_by_placement_rate(records, min_rate=0.5)
        assert len(result) == 1

    def test_filter_by_method(self):
        records = [_pqr(method="a"), _pqr(method="b"), _pqr(method="a")]
        result = filter_by_method(records, "a")
        assert len(result) == 2

    def test_top_k_by_coverage(self):
        records = [_pqr(coverage=0.3), _pqr(coverage=0.9), _pqr(coverage=0.6)]
        top = top_k_position_records(records, 2)
        assert top[0].canvas_coverage == pytest.approx(0.9)

    def test_best_position_record(self):
        records = [_pqr(coverage=0.3), _pqr(coverage=0.9)]
        best = best_position_record(records)
        assert best.canvas_coverage == pytest.approx(0.9)

    def test_best_position_empty(self):
        assert best_position_record([]) is None


# ─── position_quality_stats ───────────────────────────────────────────────────

class TestPositionQualityStatsExtra:
    def test_empty_returns_zeros(self):
        s = position_quality_stats([])
        assert s["count"] == 0

    def test_count_and_min_max(self):
        records = [_pqr(coverage=0.3), _pqr(coverage=0.7)]
        s = position_quality_stats(records)
        assert s["count"] == 2
        assert s["min"] == pytest.approx(0.3)
        assert s["max"] == pytest.approx(0.7)


# ─── AssemblyHistoryEntry ────────────────────────────────────────────────────

class TestAssemblyHistoryEntryExtra:
    def test_fields_stored(self):
        e = _ahe(run_id=5, best_score=0.85)
        assert e.run_id == 5 and e.best_score == pytest.approx(0.85)

    def test_converged_flag(self):
        e = _ahe(converged=True)
        assert e.converged is True


# ─── make_assembly_history_entry ─────────────────────────────────────────────

class TestMakeAssemblyHistoryEntryExtra:
    def test_returns_entry(self):
        e = make_assembly_history_entry(0, 100, 0.8, True, 50, "greedy")
        assert isinstance(e, AssemblyHistoryEntry)

    def test_params_stored(self):
        e = make_assembly_history_entry(0, 50, 0.7, False, None, "random",
                                         seed=42)
        assert e.params["seed"] == 42


# ─── summarise_assembly_history ───────────────────────────────────────────────

class TestSummariseAssemblyHistoryExtra:
    def test_empty_returns_summary(self):
        s = summarise_assembly_history([])
        assert s.n_runs == 0 and s.best_run_id is None

    def test_convergence_rate(self):
        entries = [_ahe(converged=True), _ahe(converged=False),
                   _ahe(converged=True)]
        s = summarise_assembly_history(entries)
        assert s.convergence_rate == pytest.approx(2 / 3)

    def test_best_run_id(self):
        entries = [_ahe(run_id=0, best_score=0.5),
                   _ahe(run_id=1, best_score=0.9)]
        s = summarise_assembly_history(entries)
        assert s.best_run_id == 1

    def test_mean_best_score(self):
        entries = [_ahe(best_score=0.4), _ahe(best_score=0.8)]
        s = summarise_assembly_history(entries)
        assert s.mean_best_score == pytest.approx(0.6)


# ─── filters / ranking assembly ──────────────────────────────────────────────

class TestFilterAssemblyHistoryExtra:
    def test_filter_converged(self):
        entries = [_ahe(converged=True), _ahe(converged=False)]
        result = filter_converged(entries)
        assert len(result) == 1

    def test_filter_by_min_best_score(self):
        entries = [_ahe(best_score=0.3), _ahe(best_score=0.8)]
        result = filter_by_min_best_score(entries, 0.5)
        assert len(result) == 1

    def test_top_k_assembly(self):
        entries = [_ahe(best_score=0.3), _ahe(best_score=0.9),
                   _ahe(best_score=0.6)]
        top = top_k_assembly_entries(entries, 2)
        assert top[0].best_score == pytest.approx(0.9)

    def test_best_assembly_empty(self):
        assert best_assembly_entry([]) is None

    def test_best_assembly(self):
        entries = [_ahe(best_score=0.3), _ahe(best_score=0.9)]
        best = best_assembly_entry(entries)
        assert best.best_score == pytest.approx(0.9)


# ─── assembly_score_stats ────────────────────────────────────────────────────

class TestAssemblyScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = assembly_score_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        entries = [_ahe(best_score=0.4), _ahe(best_score=0.8)]
        s = assembly_score_stats(entries)
        assert s["count"] == 2


# ─── compare / batch assembly ────────────────────────────────────────────────

class TestCompareAssemblyHistoryExtra:
    def test_returns_dict(self):
        s = summarise_assembly_history([_ahe()])
        d = compare_assembly_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_assembly_history([_ahe()])
        d = compare_assembly_summaries(s, s)
        assert d["delta_mean_best_score"] == pytest.approx(0.0)

    def test_same_best_flag(self):
        s = summarise_assembly_history([_ahe(run_id=0)])
        d = compare_assembly_summaries(s, s)
        assert d["same_best"] is True

    def test_batch_length(self):
        result = batch_summarise_assembly_history([[_ahe()], []])
        assert len(result) == 2
