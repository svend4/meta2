"""Extra tests for puzzle_reconstruction/utils/path_plan_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _path(start=0, end=5, path=None, cost=3.0, found=True) -> PathPlanEntry:
    return PathPlanEntry(start=start, end=end,
                         path=path if path is not None else [0, 2, 5],
                         cost=cost, found=found)


def _plan(plan_id=0, n_fragments=10, n_placed=8, coverage=0.8,
          mean_score=0.7, strategy="greedy",
          placement_order=None) -> AssemblyPlanEntry:
    return AssemblyPlanEntry(plan_id=plan_id, n_fragments=n_fragments,
                              n_placed=n_placed, coverage=coverage,
                              mean_score=mean_score, strategy=strategy,
                              placement_order=placement_order or list(range(n_placed)))


# ─── PathPlanConfig ───────────────────────────────────────────────────────────

class TestPathPlanConfigExtra:
    def test_default_min_cost(self):
        assert PathPlanConfig().min_cost == pytest.approx(0.0)

    def test_require_found_default(self):
        assert PathPlanConfig().require_found is True


# ─── PathPlanEntry hops ───────────────────────────────────────────────────────

class TestPathPlanEntryExtra:
    def test_hops_computed(self):
        e = _path(path=[0, 1, 2, 3])
        assert e.hops == 3

    def test_empty_path_hops_zero(self):
        e = _path(path=[])
        assert e.hops == 0

    def test_single_node_hops_zero(self):
        e = _path(path=[0])
        assert e.hops == 0

    def test_stores_cost(self):
        e = _path(cost=7.5)
        assert e.cost == pytest.approx(7.5)


# ─── make_path_entry / entries_from_path_results ─────────────────────────────

class TestMakePathEntryExtra:
    def test_returns_entry(self):
        e = make_path_entry(0, 5, [0, 2, 5], 3.0, True)
        assert isinstance(e, PathPlanEntry)

    def test_not_found_entry(self):
        e = make_path_entry(0, 5, [], 0.0, False)
        assert e.found is False

    def test_entries_from_results(self):
        data = [(0, 1, [0, 1], 1.0, True), (1, 2, [], 0.0, False)]
        entries = entries_from_path_results(data)
        assert len(entries) == 2
        assert entries[0].found is True


# ─── summarise_path_entries ───────────────────────────────────────────────────

class TestSummarisePathEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_path_entries([])
        assert s.n_entries == 0 and s.found_rate == pytest.approx(0.0)

    def test_found_rate(self):
        entries = [_path(found=True), _path(found=False), _path(found=True)]
        s = summarise_path_entries(entries)
        assert s.found_rate == pytest.approx(2/3)

    def test_mean_cost_only_found(self):
        entries = [_path(cost=2.0, found=True), _path(cost=4.0, found=True),
                   _path(found=False)]
        s = summarise_path_entries(entries)
        assert s.mean_cost == pytest.approx(3.0)


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterPathExtra:
    def test_filter_found(self):
        entries = [_path(found=True), _path(found=False)]
        result = filter_found_paths(entries)
        assert all(e.found for e in result)

    def test_filter_not_found(self):
        entries = [_path(found=True), _path(found=False)]
        result = filter_not_found_paths(entries)
        assert not any(e.found for e in result)

    def test_filter_cost_range(self):
        entries = [_path(cost=1.0, found=True), _path(cost=10.0, found=True)]
        result = filter_path_by_cost_range(entries, 0.0, 5.0)
        assert len(result) == 1

    def test_filter_max_hops(self):
        entries = [_path(path=[0, 1, 2], found=True),
                   _path(path=[0, 1, 2, 3, 4], found=True)]
        result = filter_path_by_max_hops(entries, 2)
        assert len(result) == 1

    def test_top_k_shortest(self):
        entries = [_path(cost=5.0, found=True), _path(cost=1.0, found=True)]
        top = top_k_shortest_paths(entries, 1)
        assert top[0].cost == pytest.approx(1.0)

    def test_cheapest_path_none_not_found(self):
        entries = [_path(found=False)]
        assert cheapest_path_entry(entries) is None

    def test_cheapest_path(self):
        entries = [_path(cost=3.0, found=True), _path(cost=1.0, found=True)]
        cheapest = cheapest_path_entry(entries)
        assert cheapest.cost == pytest.approx(1.0)


# ─── path_cost_stats ──────────────────────────────────────────────────────────

class TestPathCostStatsExtra:
    def test_empty_returns_zeros(self):
        s = path_cost_stats([])
        assert s["count"] == 0

    def test_only_found_counted(self):
        entries = [_path(cost=2.0, found=True), _path(cost=9.0, found=False)]
        s = path_cost_stats(entries)
        assert s["count"] == pytest.approx(1.0)


# ─── compare / batch path ─────────────────────────────────────────────────────

class TestComparePathSummariesExtra:
    def test_returns_dict(self):
        s = summarise_path_entries([_path()])
        d = compare_path_summaries(s, s)
        assert isinstance(d, dict)

    def test_delta_zero_identical(self):
        s = summarise_path_entries([_path()])
        d = compare_path_summaries(s, s)
        assert d["found_rate_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_path_entries([[_path()], [_path(), _path()]])
        assert len(result) == 2


# ─── AssemblyPlanEntry ────────────────────────────────────────────────────────

class TestAssemblyPlanEntryExtra:
    def test_stores_coverage(self):
        e = _plan(coverage=0.9)
        assert e.coverage == pytest.approx(0.9)

    def test_stores_strategy(self):
        e = _plan(strategy="beam")
        assert e.strategy == "beam"


# ─── make_assembly_plan_entry ────────────────────────────────────────────────

class TestMakeAssemblyPlanEntryExtra:
    def test_returns_entry(self):
        e = make_assembly_plan_entry(0, 10, 8, 0.8, 0.7, "greedy", [0, 1, 2])
        assert isinstance(e, AssemblyPlanEntry)

    def test_placement_order_stored(self):
        e = make_assembly_plan_entry(0, 5, 3, 0.6, 0.5, "greedy", [2, 0, 1])
        assert e.placement_order == [2, 0, 1]


# ─── summarise_assembly_plans ────────────────────────────────────────────────

class TestSummariseAssemblyPlansExtra:
    def test_empty_returns_summary(self):
        s = summarise_assembly_plans([])
        assert s.n_plans == 0

    def test_single_strategy(self):
        entries = [_plan(strategy="greedy"), _plan(strategy="greedy")]
        s = summarise_assembly_plans(entries)
        assert s.strategy == "greedy"

    def test_mixed_strategy(self):
        entries = [_plan(strategy="greedy"), _plan(strategy="beam")]
        s = summarise_assembly_plans(entries)
        assert s.strategy == "mixed"

    def test_mean_coverage(self):
        entries = [_plan(coverage=0.6), _plan(coverage=0.8)]
        s = summarise_assembly_plans(entries)
        assert s.mean_coverage == pytest.approx(0.7)


# ─── filter assembly plans ────────────────────────────────────────────────────

class TestFilterAssemblyPlansExtra:
    def test_filter_full_coverage(self):
        entries = [_plan(coverage=1.0), _plan(coverage=0.5)]
        result = filter_full_coverage_plans(entries)
        assert len(result) == 1

    def test_filter_by_coverage(self):
        entries = [_plan(coverage=0.3), _plan(coverage=0.9)]
        result = filter_assembly_plans_by_coverage(entries, 0.7)
        assert len(result) == 1

    def test_filter_by_score(self):
        entries = [_plan(mean_score=0.4), _plan(mean_score=0.9)]
        result = filter_assembly_plans_by_score(entries, 0.7)
        assert len(result) == 1

    def test_filter_by_strategy(self):
        entries = [_plan(strategy="greedy"), _plan(strategy="beam")]
        result = filter_assembly_plans_by_strategy(entries, "beam")
        assert len(result) == 1

    def test_top_k(self):
        entries = [_plan(coverage=0.5, mean_score=0.5),
                   _plan(coverage=0.9, mean_score=0.9)]
        top = top_k_assembly_plan_entries(entries, 1)
        assert top[0].coverage == pytest.approx(0.9)

    def test_best_plan_none_empty(self):
        assert best_assembly_plan_entry([]) is None

    def test_best_plan(self):
        entries = [_plan(coverage=0.5, mean_score=0.5),
                   _plan(coverage=0.9, mean_score=0.9)]
        best = best_assembly_plan_entry(entries)
        assert best.coverage == pytest.approx(0.9)


# ─── assembly_plan_stats / compare / batch ───────────────────────────────────

class TestAssemblyPlanStatsExtra:
    def test_empty_returns_zeros(self):
        s = assembly_plan_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = assembly_plan_stats([_plan(), _plan()])
        assert s["count"] == pytest.approx(2.0)

    def test_compare_returns_dict(self):
        s = summarise_assembly_plans([_plan()])
        d = compare_assembly_plan_summaries(s, s)
        assert isinstance(d, dict)

    def test_compare_delta_zero(self):
        s = summarise_assembly_plans([_plan()])
        d = compare_assembly_plan_summaries(s, s)
        assert d["mean_coverage_delta"] == pytest.approx(0.0)

    def test_batch_length(self):
        result = batch_summarise_assembly_plans([[_plan()], [_plan(), _plan()]])
        assert len(result) == 2
