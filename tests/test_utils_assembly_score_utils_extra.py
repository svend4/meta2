"""Extra tests for puzzle_reconstruction/utils/assembly_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.assembly_score_utils import (
    AssemblyScoreConfig,
    AssemblyScoreEntry,
    AssemblySummary,
    make_assembly_entry,
    summarise_assemblies,
    filter_good_assemblies,
    filter_poor_assemblies,
    filter_by_method,
    filter_by_score_range,
    filter_by_min_fragments,
    top_k_assembly_entries,
    best_assembly_entry,
    assembly_score_stats,
    compare_assembly_summaries,
    batch_summarise_assemblies,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(run_id=0, method="sa", n_frags=5, score=0.7,
           n_iter=10, rank=1) -> AssemblyScoreEntry:
    return AssemblyScoreEntry(
        run_id=run_id, method=method, n_fragments=n_frags,
        total_score=score, n_iterations=n_iter, rank=rank,
    )


def _entries(n=5) -> list:
    return [_entry(run_id=i, score=float(i) / n) for i in range(n)]


def _summary(entries=None) -> AssemblySummary:
    if entries is None:
        entries = _entries(5)
    return summarise_assemblies(entries)


# ─── AssemblyScoreConfig ──────────────────────────────────────────────────────

class TestAssemblyScoreConfigExtra:
    def test_default_min_score(self):
        assert AssemblyScoreConfig().min_score == pytest.approx(0.0)

    def test_default_max_entries(self):
        assert AssemblyScoreConfig().max_entries == 1000

    def test_default_method(self):
        assert AssemblyScoreConfig().method == "any"

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreConfig(min_score=-0.1)

    def test_max_entries_zero_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreConfig(max_entries=0)

    def test_max_entries_negative_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreConfig(max_entries=-1)


# ─── AssemblyScoreEntry ───────────────────────────────────────────────────────

class TestAssemblyScoreEntryExtra:
    def test_stores_run_id(self):
        assert _entry(run_id=7).run_id == 7

    def test_stores_method(self):
        assert _entry(method="greedy").method == "greedy"

    def test_stores_n_fragments(self):
        assert _entry(n_frags=10).n_fragments == 10

    def test_stores_total_score(self):
        assert _entry(score=0.85).total_score == pytest.approx(0.85)

    def test_negative_run_id_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreEntry(run_id=-1, method="m", n_fragments=5,
                               total_score=0.5)

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreEntry(run_id=0, method="m", n_fragments=-1,
                               total_score=0.5)

    def test_negative_total_score_raises(self):
        with pytest.raises(ValueError):
            AssemblyScoreEntry(run_id=0, method="m", n_fragments=5,
                               total_score=-0.1)

    def test_is_good_true(self):
        assert _entry(score=0.8).is_good is True

    def test_is_good_false(self):
        assert _entry(score=0.3).is_good is False

    def test_score_per_fragment(self):
        e = _entry(n_frags=4, score=0.8)
        assert e.score_per_fragment == pytest.approx(0.2)

    def test_score_per_fragment_zero_frags(self):
        e = AssemblyScoreEntry(run_id=0, method="m", n_fragments=0,
                               total_score=0.0)
        assert e.score_per_fragment == pytest.approx(0.0)

    def test_default_meta_empty(self):
        assert _entry().meta == {}


# ─── make_assembly_entry ──────────────────────────────────────────────────────

class TestMakeAssemblyEntryExtra:
    def test_returns_entry(self):
        e = make_assembly_entry(0, "sa", 5, 0.7)
        assert isinstance(e, AssemblyScoreEntry)

    def test_stores_values(self):
        e = make_assembly_entry(3, "greedy", 8, 0.6)
        assert e.run_id == 3 and e.method == "greedy"
        assert e.n_fragments == 8 and e.total_score == pytest.approx(0.6)

    def test_none_meta_empty(self):
        e = make_assembly_entry(0, "m", 5, 0.5, meta=None)
        assert e.meta == {}

    def test_custom_n_iterations(self):
        e = make_assembly_entry(0, "m", 5, 0.5, n_iterations=100)
        assert e.n_iterations == 100


# ─── summarise_assemblies ─────────────────────────────────────────────────────

class TestSummariseAssembliesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_assemblies(_entries()), AssemblySummary)

    def test_n_total_correct(self):
        assert summarise_assemblies(_entries(7)).n_total == 7

    def test_empty_entries(self):
        s = summarise_assemblies([])
        assert s.n_total == 0

    def test_n_good_plus_n_poor_le_n_total(self):
        s = summarise_assemblies(_entries(10))
        assert s.n_good + s.n_poor <= s.n_total

    def test_mean_in_range(self):
        s = summarise_assemblies(_entries(5))
        assert s.min_score <= s.mean_score <= s.max_score


# ─── filter_good_assemblies / filter_poor_assemblies ─────────────────────────

class TestFilterGoodAssembliesExtra:
    def test_returns_list(self):
        assert isinstance(filter_good_assemblies(_entries()), list)

    def test_all_good(self):
        entries = [_entry(score=0.8), _entry(score=0.9)]
        result = filter_good_assemblies(entries)
        assert all(e.is_good for e in result)

    def test_empty_input(self):
        assert filter_good_assemblies([]) == []


class TestFilterPoorAssembliesExtra:
    def test_returns_list(self):
        assert isinstance(filter_poor_assemblies(_entries()), list)

    def test_all_poor(self):
        entries = [_entry(score=0.1), _entry(score=0.2)]
        result = filter_poor_assemblies(entries)
        assert all(not e.is_good for e in result)

    def test_empty_input(self):
        assert filter_poor_assemblies([]) == []


# ─── filter_by_method ─────────────────────────────────────────────────────────

class TestFilterByMethodExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_method(_entries(), "sa"), list)

    def test_keeps_matching_method(self):
        entries = [_entry(method="sa"), _entry(method="greedy")]
        result = filter_by_method(entries, "sa")
        assert all(e.method == "sa" for e in result)
        assert len(result) == 1

    def test_no_match_empty(self):
        result = filter_by_method(_entries(), "nonexistent")
        assert result == []


# ─── filter_by_score_range ────────────────────────────────────────────────────

class TestFilterByScoreRangeExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_score_range(_entries()), list)

    def test_keeps_in_range(self):
        entries = [_entry(score=0.2), _entry(score=0.6), _entry(score=0.9)]
        result = filter_by_score_range(entries, lo=0.5, hi=0.8)
        assert all(0.5 <= e.total_score <= 0.8 for e in result)

    def test_wide_range_keeps_all(self):
        result = filter_by_score_range(_entries(5), lo=0.0, hi=100.0)
        assert len(result) == 5


# ─── filter_by_min_fragments ──────────────────────────────────────────────────

class TestFilterByMinFragmentsExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_min_fragments(_entries(), 1), list)

    def test_keeps_ge_min(self):
        entries = [_entry(n_frags=2), _entry(n_frags=5), _entry(n_frags=8)]
        result = filter_by_min_fragments(entries, min_fragments=5)
        assert all(e.n_fragments >= 5 for e in result)

    def test_zero_min_keeps_all(self):
        result = filter_by_min_fragments(_entries(5), min_fragments=0)
        assert len(result) == 5


# ─── top_k_assembly_entries ───────────────────────────────────────────────────

class TestTopKAssemblyEntriesExtra:
    def test_returns_list(self):
        assert isinstance(top_k_assembly_entries(_entries(), 3), list)

    def test_length_at_most_k(self):
        result = top_k_assembly_entries(_entries(5), 3)
        assert len(result) <= 3

    def test_k_le_0_empty(self):
        result = top_k_assembly_entries(_entries(), 0)
        assert result == []

    def test_k_larger_than_n(self):
        result = top_k_assembly_entries(_entries(3), 10)
        assert len(result) == 3

    def test_sorted_desc(self):
        entries = [_entry(score=0.3), _entry(score=0.8), _entry(score=0.5)]
        result = top_k_assembly_entries(entries, 3)
        scores = [e.total_score for e in result]
        assert scores == sorted(scores, reverse=True)


# ─── best_assembly_entry ──────────────────────────────────────────────────────

class TestBestAssemblyEntryExtra:
    def test_returns_entry_or_none(self):
        result = best_assembly_entry(_entries(5))
        assert result is None or isinstance(result, AssemblyScoreEntry)

    def test_empty_returns_none(self):
        assert best_assembly_entry([]) is None

    def test_highest_score(self):
        entries = [_entry(score=0.2), _entry(score=0.9), _entry(score=0.5)]
        result = best_assembly_entry(entries)
        assert result.total_score == pytest.approx(0.9)


# ─── assembly_score_stats ─────────────────────────────────────────────────────

class TestAssemblyScoreStatsExtra:
    def test_returns_dict(self):
        assert isinstance(assembly_score_stats(_entries()), dict)

    def test_keys_present(self):
        stats = assembly_score_stats(_entries(5))
        for k in ("n", "mean", "min", "max", "std"):
            assert k in stats

    def test_n_correct(self):
        assert assembly_score_stats(_entries(7))["n"] == 7

    def test_empty_entries(self):
        stats = assembly_score_stats([])
        assert stats["n"] == 0


# ─── compare_assembly_summaries ───────────────────────────────────────────────

class TestCompareAssemblySummariesExtra:
    def test_returns_dict(self):
        a = _summary()
        b = _summary()
        assert isinstance(compare_assembly_summaries(a, b), dict)

    def test_keys_present(self):
        d = compare_assembly_summaries(_summary(), _summary())
        for k in ("delta_mean_score", "delta_max_score", "delta_n_good", "a_better"):
            assert k in d

    def test_identical_zero_delta(self):
        s = _summary()
        d = compare_assembly_summaries(s, s)
        assert d["delta_mean_score"] == pytest.approx(0.0)


# ─── batch_summarise_assemblies ───────────────────────────────────────────────

class TestBatchSummariseAssembliesExtra:
    def test_returns_list(self):
        assert isinstance(batch_summarise_assemblies([_entries(5)]), list)

    def test_length_matches(self):
        result = batch_summarise_assemblies([_entries(3), _entries(5)])
        assert len(result) == 2

    def test_each_is_summary(self):
        for s in batch_summarise_assemblies([_entries(3)]):
            assert isinstance(s, AssemblySummary)

    def test_empty_returns_empty(self):
        assert batch_summarise_assemblies([]) == []
