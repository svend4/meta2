"""Tests for puzzle_reconstruction.utils.assembly_score_utils."""
import pytest
from puzzle_reconstruction.utils.assembly_score_utils import (
    AssemblyScoreConfig,
    AssemblyScoreEntry,
    AssemblySummary,
    make_assembly_entry,
    entries_from_assemblies,
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


# ── AssemblyScoreConfig ────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = AssemblyScoreConfig()
    assert cfg.min_score == 0.0
    assert cfg.max_entries == 1000
    assert cfg.method == "any"


def test_config_invalid_min_score():
    with pytest.raises(ValueError):
        AssemblyScoreConfig(min_score=-0.1)


def test_config_invalid_max_entries():
    with pytest.raises(ValueError):
        AssemblyScoreConfig(max_entries=0)


# ── AssemblyScoreEntry ────────────────────────────────────────────────────────

def test_entry_properties():
    e = AssemblyScoreEntry(run_id=0, method="ga", n_fragments=10, total_score=0.8)
    assert e.is_good is True
    assert abs(e.score_per_fragment - 0.08) < 1e-9


def test_entry_poor():
    e = AssemblyScoreEntry(run_id=1, method="ga", n_fragments=5, total_score=0.3)
    assert e.is_good is False


def test_entry_score_per_fragment_zero_frags():
    e = AssemblyScoreEntry(run_id=2, method="ga", n_fragments=0, total_score=0.0)
    assert e.score_per_fragment == 0.0


def test_entry_invalid_run_id():
    with pytest.raises(ValueError):
        AssemblyScoreEntry(run_id=-1, method="ga", n_fragments=5, total_score=0.5)


def test_entry_invalid_score():
    with pytest.raises(ValueError):
        AssemblyScoreEntry(run_id=0, method="ga", n_fragments=5, total_score=-0.1)


# ── make_assembly_entry ────────────────────────────────────────────────────────

def test_make_assembly_entry_returns_entry():
    e = make_assembly_entry(0, "ga", 5, 0.7)
    assert isinstance(e, AssemblyScoreEntry)
    assert e.total_score == 0.7


def test_make_assembly_entry_meta_none():
    e = make_assembly_entry(0, "ga", 5, 0.7, meta=None)
    assert e.meta == {}


# ── entries_from_assemblies ────────────────────────────────────────────────────

class _FakeAssembly:
    def __init__(self, score, n_placements):
        self.total_score = score
        self.placements = {i: None for i in range(n_placements)}


def test_entries_from_assemblies_count():
    asms = [_FakeAssembly(0.9, 10), _FakeAssembly(0.4, 5)]
    entries = entries_from_assemblies(asms, method="test")
    assert len(entries) == 2


def test_entries_from_assemblies_ranks():
    asms = [_FakeAssembly(0.4, 5), _FakeAssembly(0.9, 10)]
    entries = entries_from_assemblies(asms)
    # Best score gets rank 1
    best = max(entries, key=lambda e: e.total_score)
    assert best.rank == 1


def test_entries_from_assemblies_empty():
    entries = entries_from_assemblies([])
    assert entries == []


# ── summarise_assemblies ───────────────────────────────────────────────────────

def test_summarise_empty():
    s = summarise_assemblies([])
    assert s.n_total == 0
    assert s.mean_score == 0.0


def test_summarise_basic():
    entries = [
        make_assembly_entry(0, "ga", 5, 0.8),
        make_assembly_entry(1, "ga", 5, 0.2),
    ]
    s = summarise_assemblies(entries)
    assert s.n_total == 2
    assert s.n_good == 1
    assert s.n_poor == 1
    assert abs(s.mean_score - 0.5) < 1e-9


# ── Filters ───────────────────────────────────────────────────────────────────

def _make_entries():
    return [
        make_assembly_entry(0, "ga", 10, 0.9),
        make_assembly_entry(1, "sa", 5, 0.3),
        make_assembly_entry(2, "ga", 3, 0.6),
        make_assembly_entry(3, "aco", 8, 0.1),
    ]


def test_filter_good():
    entries = _make_entries()
    good = filter_good_assemblies(entries)
    assert all(e.is_good for e in good)


def test_filter_poor():
    entries = _make_entries()
    poor = filter_poor_assemblies(entries)
    assert all(not e.is_good for e in poor)


def test_filter_by_method():
    entries = _make_entries()
    ga_only = filter_by_method(entries, "ga")
    assert all(e.method == "ga" for e in ga_only)
    assert len(ga_only) == 2


def test_filter_by_score_range():
    entries = _make_entries()
    filtered = filter_by_score_range(entries, lo=0.4, hi=0.8)
    assert all(0.4 <= e.total_score <= 0.8 for e in filtered)


def test_filter_by_min_fragments():
    entries = _make_entries()
    filtered = filter_by_min_fragments(entries, min_fragments=6)
    assert all(e.n_fragments >= 6 for e in filtered)


# ── Ranking ───────────────────────────────────────────────────────────────────

def test_top_k_entries():
    entries = _make_entries()
    top2 = top_k_assembly_entries(entries, 2)
    assert len(top2) == 2
    assert top2[0].total_score >= top2[1].total_score


def test_top_k_zero():
    entries = _make_entries()
    assert top_k_assembly_entries(entries, 0) == []


def test_best_assembly_entry():
    entries = _make_entries()
    best = best_assembly_entry(entries)
    assert best is not None
    assert best.total_score == max(e.total_score for e in entries)


def test_best_assembly_entry_empty():
    assert best_assembly_entry([]) is None


# ── Statistics ────────────────────────────────────────────────────────────────

def test_assembly_score_stats_empty():
    stats = assembly_score_stats([])
    assert stats["n"] == 0
    assert stats["mean"] == 0.0


def test_assembly_score_stats_values():
    entries = [make_assembly_entry(i, "ga", 5, float(i) / 4) for i in range(5)]
    stats = assembly_score_stats(entries)
    assert stats["n"] == 5
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert "std" in stats


# ── compare_assembly_summaries ────────────────────────────────────────────────

def test_compare_summaries():
    s1 = summarise_assemblies([make_assembly_entry(0, "ga", 5, 0.9)])
    s2 = summarise_assemblies([make_assembly_entry(0, "sa", 5, 0.4)])
    cmp = compare_assembly_summaries(s1, s2)
    assert "delta_mean_score" in cmp
    assert cmp["a_better"] is True


# ── batch_summarise ───────────────────────────────────────────────────────────

def test_batch_summarise():
    groups = [_make_entries()[:2], _make_entries()[2:]]
    summaries = batch_summarise_assemblies(groups)
    assert len(summaries) == 2
    assert all(isinstance(s, AssemblySummary) for s in summaries)
