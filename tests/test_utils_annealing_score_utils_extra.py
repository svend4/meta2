"""Extra tests for puzzle_reconstruction/utils/annealing_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.annealing_score_utils import (
    AnnealingScoreConfig,
    AnnealingScoreEntry,
    AnnealingSummary,
    make_annealing_entry,
    entries_from_log,
    summarise_annealing,
    filter_accepted,
    filter_rejected,
    filter_by_min_score,
    filter_by_temperature_range,
    top_k_entries,
    annealing_score_stats,
    best_entry,
    compare_summaries,
    batch_summarise,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(iter=0, temp=1.0, cur=0.5, best=0.5, accepted=True) -> AnnealingScoreEntry:
    return AnnealingScoreEntry(
        iteration=iter, temperature=temp,
        current_score=cur, best_score=best, accepted=accepted,
    )


def _entries(n=5) -> list:
    return [_entry(iter=i, cur=float(i) / n, best=float(i) / n) for i in range(n)]


def _make_summary(entries=None, best=0.8, final=0.7) -> AnnealingSummary:
    if entries is None:
        entries = _entries(5)
    return AnnealingSummary(
        entries=entries,
        n_iterations=len(entries),
        final_score=final,
        best_score=best,
        n_accepted=len(entries),
        acceptance_rate=1.0,
        converged=True,
    )


# ─── AnnealingScoreConfig ─────────────────────────────────────────────────────

class TestAnnealingScoreConfigExtra:
    def test_default_min_score(self):
        assert AnnealingScoreConfig().min_score == pytest.approx(0.0)

    def test_default_convergence_window(self):
        assert AnnealingScoreConfig().convergence_window == 10

    def test_default_improvement_threshold(self):
        assert AnnealingScoreConfig().improvement_threshold == pytest.approx(1e-4)

    def test_default_prefer_high_score(self):
        assert AnnealingScoreConfig().prefer_high_score is True

    def test_convergence_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            AnnealingScoreConfig(convergence_window=0)

    def test_improvement_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            AnnealingScoreConfig(improvement_threshold=-0.01)

    def test_custom_values(self):
        cfg = AnnealingScoreConfig(convergence_window=5, min_score=0.3)
        assert cfg.convergence_window == 5 and cfg.min_score == pytest.approx(0.3)


# ─── AnnealingScoreEntry ──────────────────────────────────────────────────────

class TestAnnealingScoreEntryExtra:
    def test_stores_iteration(self):
        e = _entry(iter=3)
        assert e.iteration == 3

    def test_stores_temperature(self):
        e = _entry(temp=2.5)
        assert e.temperature == pytest.approx(2.5)

    def test_stores_current_score(self):
        e = _entry(cur=0.75)
        assert e.current_score == pytest.approx(0.75)

    def test_stores_best_score(self):
        e = _entry(best=0.9)
        assert e.best_score == pytest.approx(0.9)

    def test_stores_accepted(self):
        e = _entry(accepted=False)
        assert e.accepted is False

    def test_negative_iteration_raises(self):
        with pytest.raises(ValueError):
            AnnealingScoreEntry(iteration=-1, temperature=1.0,
                                current_score=0.5, best_score=0.5, accepted=True)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            AnnealingScoreEntry(iteration=0, temperature=-0.1,
                                current_score=0.5, best_score=0.5, accepted=True)

    def test_default_meta_empty(self):
        e = _entry()
        assert e.meta == {}


# ─── AnnealingSummary ─────────────────────────────────────────────────────────

class TestAnnealingSummaryExtra:
    def test_stores_best_score(self):
        s = _make_summary(best=0.95)
        assert s.best_score == pytest.approx(0.95)

    def test_stores_final_score(self):
        s = _make_summary(final=0.8)
        assert s.final_score == pytest.approx(0.8)

    def test_n_iterations(self):
        s = _make_summary(entries=_entries(7))
        assert s.n_iterations == 7

    def test_repr_is_str(self):
        s = _make_summary()
        assert isinstance(repr(s), str)


# ─── make_annealing_entry ─────────────────────────────────────────────────────

class TestMakeAnnealingEntryExtra:
    def test_returns_entry(self):
        e = make_annealing_entry(0, 1.0, 0.5, 0.5, True)
        assert isinstance(e, AnnealingScoreEntry)

    def test_values_stored(self):
        e = make_annealing_entry(3, 2.0, 0.7, 0.9, False)
        assert e.iteration == 3
        assert e.temperature == pytest.approx(2.0)
        assert e.current_score == pytest.approx(0.7)

    def test_none_meta_empty(self):
        e = make_annealing_entry(0, 1.0, 0.5, 0.5, True, meta=None)
        assert e.meta == {}

    def test_custom_meta(self):
        e = make_annealing_entry(0, 1.0, 0.5, 0.5, True, meta={"x": 1})
        assert e.meta["x"] == 1


# ─── entries_from_log ─────────────────────────────────────────────────────────

class TestEntriesFromLogExtra:
    def test_returns_list(self):
        log = [{"iteration": 0, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}]
        assert isinstance(entries_from_log(log), list)

    def test_length_matches(self):
        log = [{"iteration": i, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}
               for i in range(5)]
        assert len(entries_from_log(log)) == 5

    def test_empty_log(self):
        assert entries_from_log([]) == []

    def test_all_are_entries(self):
        log = [{"iteration": 0, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}]
        for e in entries_from_log(log):
            assert isinstance(e, AnnealingScoreEntry)

    def test_extra_keys_in_meta(self):
        log = [{"iteration": 0, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True,
                "extra_key": 42}]
        entries = entries_from_log(log)
        assert entries[0].meta.get("extra_key") == 42


# ─── summarise_annealing ──────────────────────────────────────────────────────

class TestSummariseAnnealingExtra:
    def test_returns_summary(self):
        s = summarise_annealing(_entries(5))
        assert isinstance(s, AnnealingSummary)

    def test_n_iterations(self):
        s = summarise_annealing(_entries(8))
        assert s.n_iterations == 8

    def test_empty_entries(self):
        s = summarise_annealing([])
        assert s.n_iterations == 0

    def test_none_cfg(self):
        s = summarise_annealing(_entries(3), cfg=None)
        assert isinstance(s, AnnealingSummary)

    def test_best_score_nonneg(self):
        s = summarise_annealing(_entries(5))
        assert s.best_score >= 0.0

    def test_acceptance_rate_in_range(self):
        s = summarise_annealing(_entries(5))
        assert 0.0 <= s.acceptance_rate <= 1.0


# ─── filter_accepted / filter_rejected ───────────────────────────────────────

class TestFilterAcceptedExtra:
    def test_returns_list(self):
        assert isinstance(filter_accepted(_entries()), list)

    def test_all_accepted(self):
        entries = _entries(5)
        result = filter_accepted(entries)
        assert len(result) == 5

    def test_empty_input(self):
        assert filter_accepted([]) == []

    def test_only_accepted(self):
        entries = [_entry(accepted=True), _entry(accepted=False)]
        result = filter_accepted(entries)
        assert all(e.accepted for e in result)
        assert len(result) == 1


class TestFilterRejectedExtra:
    def test_returns_list(self):
        assert isinstance(filter_rejected(_entries()), list)

    def test_all_accepted_returns_empty(self):
        entries = [_entry(accepted=True)]
        assert filter_rejected(entries) == []

    def test_only_rejected(self):
        entries = [_entry(accepted=True), _entry(accepted=False)]
        result = filter_rejected(entries)
        assert all(not e.accepted for e in result)
        assert len(result) == 1


# ─── filter_by_min_score ──────────────────────────────────────────────────────

class TestFilterByMinScoreExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_min_score(_entries()), list)

    def test_keeps_above_min(self):
        entries = [_entry(cur=0.3), _entry(cur=0.7), _entry(cur=0.5)]
        result = filter_by_min_score(entries, min_score=0.5)
        assert all(e.current_score >= 0.5 for e in result)

    def test_min_zero_keeps_all(self):
        entries = _entries(5)
        result = filter_by_min_score(entries, min_score=0.0)
        assert len(result) == len(entries)

    def test_min_one_keeps_only_one_score(self):
        entries = [_entry(cur=1.0), _entry(cur=0.5)]
        result = filter_by_min_score(entries, min_score=1.0)
        assert len(result) == 1


# ─── filter_by_temperature_range ─────────────────────────────────────────────

class TestFilterByTemperatureRangeExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_temperature_range(_entries()), list)

    def test_keeps_in_range(self):
        entries = [_entry(temp=0.5), _entry(temp=1.5), _entry(temp=2.5)]
        result = filter_by_temperature_range(entries, t_min=1.0, t_max=2.0)
        assert all(1.0 <= e.temperature <= 2.0 for e in result)

    def test_wide_range_keeps_all(self):
        entries = _entries(5)
        result = filter_by_temperature_range(entries, t_min=0.0, t_max=1000.0)
        assert len(result) == len(entries)


# ─── top_k_entries ────────────────────────────────────────────────────────────

class TestTopKEntriesExtra:
    def test_returns_list(self):
        assert isinstance(top_k_entries(_entries(5), 3), list)

    def test_length_at_most_k(self):
        result = top_k_entries(_entries(5), 3)
        assert len(result) <= 3

    def test_sorted_descending(self):
        entries = [_entry(cur=0.3), _entry(cur=0.7), _entry(cur=0.5)]
        result = top_k_entries(entries, 2)
        if len(result) > 1:
            assert result[0].current_score >= result[-1].current_score

    def test_k_larger_than_n(self):
        result = top_k_entries(_entries(3), 10)
        assert len(result) == 3

    def test_empty_returns_empty(self):
        assert top_k_entries([], 3) == []


# ─── annealing_score_stats ────────────────────────────────────────────────────

class TestAnnealingScoreStatsExtra:
    def test_returns_dict(self):
        assert isinstance(annealing_score_stats(_entries()), dict)

    def test_keys_present(self):
        stats = annealing_score_stats(_entries(5))
        for k in ("count", "mean", "std", "min", "max", "acceptance_rate"):
            assert k in stats

    def test_empty_all_zero(self):
        stats = annealing_score_stats([])
        assert stats["count"] == 0

    def test_count_correct(self):
        assert annealing_score_stats(_entries(7))["count"] == 7

    def test_mean_in_range(self):
        stats = annealing_score_stats(_entries(5))
        assert stats["min"] <= stats["mean"] <= stats["max"]


# ─── best_entry ───────────────────────────────────────────────────────────────

class TestBestEntryExtra:
    def test_returns_entry_or_none(self):
        result = best_entry(_entries(5))
        assert result is None or isinstance(result, AnnealingScoreEntry)

    def test_empty_returns_none(self):
        assert best_entry([]) is None

    def test_best_score(self):
        entries = [_entry(cur=0.3), _entry(cur=0.9), _entry(cur=0.6)]
        result = best_entry(entries)
        assert result.current_score == pytest.approx(0.9)


# ─── compare_summaries ────────────────────────────────────────────────────────

class TestCompareSummariesExtra:
    def test_returns_dict(self):
        a = _make_summary(best=0.8)
        b = _make_summary(best=0.9)
        assert isinstance(compare_summaries(a, b), dict)

    def test_keys_present(self):
        a = _make_summary()
        b = _make_summary()
        d = compare_summaries(a, b)
        for k in ("best_score_delta", "final_score_delta",
                  "acceptance_rate_delta", "n_iter_delta",
                  "a_converged", "b_converged"):
            assert k in d

    def test_identical_summaries_zero_delta(self):
        s = _make_summary(best=0.7, final=0.7)
        d = compare_summaries(s, s)
        assert d["best_score_delta"] == pytest.approx(0.0)


# ─── batch_summarise ──────────────────────────────────────────────────────────

class TestBatchSummariseExtra:
    def test_returns_list(self):
        log = [{"iteration": 0, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}]
        assert isinstance(batch_summarise([log]), list)

    def test_length_matches(self):
        log = [{"iteration": i, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}
               for i in range(3)]
        result = batch_summarise([log, log])
        assert len(result) == 2

    def test_each_is_summary(self):
        log = [{"iteration": 0, "temperature": 1.0,
                "current_score": 0.5, "best_score": 0.5, "accepted": True}]
        for s in batch_summarise([log]):
            assert isinstance(s, AnnealingSummary)

    def test_empty_returns_empty(self):
        assert batch_summarise([]) == []
