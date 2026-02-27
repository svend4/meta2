"""Tests for puzzle_reconstruction.utils.path_plan_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.path_plan_utils import (
    PathPlanConfig,
    PathPlanEntry,
    PathPlanSummary,
    AssemblyPlanConfig,
    AssemblyPlanEntry,
    AssemblyPlanSummary,
    make_path_entry,
    entries_from_path_results,
    summarise_path_entries,
    filter_found_paths,
    filter_not_found_paths,
    filter_path_by_cost_range,
    filter_path_by_max_hops,
    top_k_shortest_paths,
    cheapest_path_entry,
    path_cost_stats,
    compare_path_summaries,
    batch_summarise_path_entries,
    make_assembly_plan_entry,
    summarise_assembly_plans,
    filter_full_coverage_plans,
    filter_assembly_plans_by_coverage,
    filter_assembly_plans_by_score,
    filter_assembly_plans_by_strategy,
    top_k_assembly_plan_entries,
    best_assembly_plan_entry,
    assembly_plan_stats,
    compare_assembly_plan_summaries,
    batch_summarise_assembly_plans,
)

np.random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_entries():
    return [
        make_path_entry(0, 5, [0, 2, 5], 3.5, True),
        make_path_entry(1, 4, [1, 3, 4], 2.0, True),
        make_path_entry(2, 3, [], 0.0, False),
        make_path_entry(0, 3, [0, 1, 2, 3], 5.0, True),
    ]


def _assembly_entries():
    return [
        make_assembly_plan_entry(0, 10, 10, 1.0, 0.85, "greedy", list(range(10))),
        make_assembly_plan_entry(1, 10, 8, 0.8, 0.75, "greedy", list(range(8))),
        make_assembly_plan_entry(2, 10, 9, 0.9, 0.8, "beam", list(range(9))),
    ]


# ── PathPlanEntry ─────────────────────────────────────────────────────────────

def test_path_entry_hops_computed():
    e = make_path_entry(0, 3, [0, 1, 2, 3], 5.0, True)
    assert e.hops == 3


def test_path_entry_hops_empty_path():
    e = make_path_entry(0, 1, [], 0.0, False)
    assert e.hops == 0


def test_path_entry_fields():
    e = make_path_entry(2, 7, [2, 5, 7], 4.0, True)
    assert e.start == 2
    assert e.end == 7
    assert e.path == [2, 5, 7]
    assert e.cost == pytest.approx(4.0)
    assert e.found is True


# ── entries_from_path_results ─────────────────────────────────────────────────

def test_entries_from_path_results():
    results = [
        (0, 3, [0, 1, 2, 3], 3.0, True),
        (1, 4, [], 0.0, False),
    ]
    entries = entries_from_path_results(results)
    assert len(entries) == 2
    assert entries[0].found is True
    assert entries[1].found is False


# ── summarise_path_entries ────────────────────────────────────────────────────

def test_summarise_path_empty():
    s = summarise_path_entries([])
    assert s.n_entries == 0
    assert s.found_rate == pytest.approx(0.0)


def test_summarise_path_normal():
    entries = _path_entries()
    s = summarise_path_entries(entries)
    assert s.n_entries == 4
    assert s.n_found == 3
    assert s.n_not_found == 1
    assert s.found_rate == pytest.approx(0.75)


def test_summarise_path_min_cost():
    entries = _path_entries()
    s = summarise_path_entries(entries)
    assert s.min_cost == pytest.approx(2.0)


def test_summarise_path_max_cost():
    entries = _path_entries()
    s = summarise_path_entries(entries)
    assert s.max_cost == pytest.approx(5.0)


# ── Filters ───────────────────────────────────────────────────────────────────

def test_filter_found_paths():
    entries = _path_entries()
    found = filter_found_paths(entries)
    assert all(e.found for e in found)
    assert len(found) == 3


def test_filter_not_found_paths():
    entries = _path_entries()
    not_found = filter_not_found_paths(entries)
    assert all(not e.found for e in not_found)
    assert len(not_found) == 1


def test_filter_path_by_cost_range():
    entries = _path_entries()
    filtered = filter_path_by_cost_range(entries, lo=2.0, hi=4.0)
    assert all(2.0 <= e.cost <= 4.0 for e in filtered)


def test_filter_path_by_max_hops():
    entries = _path_entries()
    filtered = filter_path_by_max_hops(entries, max_hops=2)
    assert all(e.hops <= 2 for e in filtered)


def test_top_k_shortest_paths():
    entries = _path_entries()
    top = top_k_shortest_paths(entries, 2)
    assert len(top) == 2
    assert top[0].cost <= top[1].cost


def test_cheapest_path_entry():
    entries = _path_entries()
    cheapest = cheapest_path_entry(entries)
    assert cheapest is not None
    assert cheapest.cost == pytest.approx(2.0)


def test_cheapest_path_entry_all_not_found():
    entries = [make_path_entry(0, 1, [], 0.0, False)]
    assert cheapest_path_entry(entries) is None


def test_path_cost_stats_empty():
    stats = path_cost_stats([])
    assert stats["count"] == 0


def test_path_cost_stats_values():
    entries = _path_entries()
    stats = path_cost_stats(entries)
    assert stats["min"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["count"] == pytest.approx(3.0)


def test_compare_path_summaries():
    entries = _path_entries()
    s1 = summarise_path_entries(entries[:3])
    s2 = summarise_path_entries(entries[1:])
    delta = compare_path_summaries(s1, s2)
    assert "found_rate_delta" in delta
    assert "mean_cost_delta" in delta


def test_batch_summarise_path_entries():
    groups = [_path_entries()[:2], _path_entries()[2:]]
    summaries = batch_summarise_path_entries(groups)
    assert len(summaries) == 2


# ── AssemblyPlanEntry ─────────────────────────────────────────────────────────

def test_assembly_plan_entry_fields():
    e = make_assembly_plan_entry(0, 10, 9, 0.9, 0.8, "greedy", [0, 1, 2])
    assert e.plan_id == 0
    assert e.n_fragments == 10
    assert e.n_placed == 9
    assert e.coverage == pytest.approx(0.9)
    assert e.mean_score == pytest.approx(0.8)
    assert e.strategy == "greedy"
    assert e.placement_order == [0, 1, 2]


# ── summarise_assembly_plans ──────────────────────────────────────────────────

def test_summarise_assembly_empty():
    s = summarise_assembly_plans([])
    assert s.n_plans == 0
    assert s.strategy == "greedy"


def test_summarise_assembly_normal():
    entries = _assembly_entries()
    s = summarise_assembly_plans(entries)
    assert s.n_plans == 3
    assert s.min_coverage == pytest.approx(0.8)
    assert s.max_coverage == pytest.approx(1.0)
    assert s.strategy == "mixed"


def test_summarise_assembly_single_strategy():
    entries = _assembly_entries()[:2]  # both greedy
    s = summarise_assembly_plans(entries)
    assert s.strategy == "greedy"


# ── Assembly plan filters ─────────────────────────────────────────────────────

def test_filter_full_coverage_plans():
    entries = _assembly_entries()
    full = filter_full_coverage_plans(entries)
    assert all(e.coverage >= 1.0 - 1e-9 for e in full)
    assert len(full) == 1


def test_filter_assembly_plans_by_coverage():
    entries = _assembly_entries()
    filtered = filter_assembly_plans_by_coverage(entries, min_coverage=0.85)
    assert all(e.coverage >= 0.85 for e in filtered)


def test_filter_assembly_plans_by_score():
    entries = _assembly_entries()
    filtered = filter_assembly_plans_by_score(entries, min_score=0.8)
    assert all(e.mean_score >= 0.8 for e in filtered)


def test_filter_assembly_plans_by_strategy():
    entries = _assembly_entries()
    filtered = filter_assembly_plans_by_strategy(entries, "greedy")
    assert all(e.strategy == "greedy" for e in filtered)
    assert len(filtered) == 2


def test_top_k_assembly_plan_entries():
    entries = _assembly_entries()
    top = top_k_assembly_plan_entries(entries, 2)
    assert len(top) == 2
    scores = [e.coverage * e.mean_score for e in top]
    assert scores[0] >= scores[1]


def test_best_assembly_plan_entry():
    entries = _assembly_entries()
    best = best_assembly_plan_entry(entries)
    assert best is not None
    # plan 0: 1.0 * 0.85 = 0.85, plan 2: 0.9 * 0.8 = 0.72, plan 1: 0.8 * 0.75 = 0.6
    assert best.plan_id == 0


def test_best_assembly_plan_entry_empty():
    assert best_assembly_plan_entry([]) is None


def test_assembly_plan_stats_empty():
    stats = assembly_plan_stats([])
    assert stats["count"] == 0


def test_assembly_plan_stats_values():
    entries = _assembly_entries()
    stats = assembly_plan_stats(entries)
    assert stats["count"] == pytest.approx(3.0)
    assert stats["min_coverage"] == pytest.approx(0.8)
    assert stats["max_coverage"] == pytest.approx(1.0)


def test_compare_assembly_plan_summaries():
    entries = _assembly_entries()
    s1 = summarise_assembly_plans(entries[:2])
    s2 = summarise_assembly_plans(entries[1:])
    delta = compare_assembly_plan_summaries(s1, s2)
    assert "mean_coverage_delta" in delta
    assert "mean_score_delta" in delta


def test_batch_summarise_assembly_plans():
    groups = [_assembly_entries()[:2], _assembly_entries()[1:]]
    summaries = batch_summarise_assembly_plans(groups)
    assert len(summaries) == 2
