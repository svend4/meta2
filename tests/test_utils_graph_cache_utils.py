"""Tests for puzzle_reconstruction.utils.graph_cache_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.graph_cache_utils import (
    GraphAlgoConfig,
    GraphAlgoEntry,
    GraphAlgoSummary,
    BatchResultConfig,
    BatchResultEntry,
    BatchResultSummary,
    make_graph_algo_entry,
    summarise_graph_algo_entries,
    filter_graph_algo_by_found,
    filter_graph_algo_by_max_cost,
    filter_graph_algo_by_algorithm,
    filter_graph_algo_by_min_path_length,
    top_k_cheapest_paths,
    best_graph_algo_entry,
    graph_algo_cost_stats,
    compare_graph_algo_summaries,
    batch_summarise_graph_algo_entries,
    make_batch_result_entry,
    summarise_batch_result_entries,
    filter_batch_results_by_min_ratio,
    filter_batch_results_by_algorithm,
    filter_batch_results_by_max_retries,
    top_k_batch_results,
    best_batch_result_entry,
    batch_result_success_stats,
    compare_batch_result_summaries,
    batch_summarise_batch_result_entries,
)

np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entries():
    return [
        make_graph_algo_entry(0, "dijkstra", True, 5.0, 3, 10),
        make_graph_algo_entry(1, "astar", True, 3.0, 4, 10),
        make_graph_algo_entry(2, "dijkstra", False, 0.0, 0, 10),
        make_graph_algo_entry(3, "dijkstra", True, 7.0, 2, 10),
    ]


def _make_batch_entries():
    return [
        make_batch_result_entry(0, 100, 90, 10, 2, "algo_a"),
        make_batch_result_entry(1, 100, 60, 40, 5, "algo_b"),
        make_batch_result_entry(2, 50, 50, 0, 0, "algo_a"),
    ]


# ── GraphAlgoEntry ────────────────────────────────────────────────────────────

def test_make_graph_algo_entry_types():
    e = make_graph_algo_entry(0, "dijkstra", True, 2.5, 3, 8, alpha=0.1)
    assert e.run_id == 0
    assert e.algorithm == "dijkstra"
    assert e.found is True
    assert e.cost == pytest.approx(2.5)
    assert e.path_length == 3
    assert e.n_nodes == 8
    assert e.params["alpha"] == pytest.approx(0.1)


def test_make_graph_algo_entry_found_coercion():
    e = make_graph_algo_entry(1, "bfs", 1, 0.0, 2, 5)
    assert isinstance(e.found, bool)


# ── summarise_graph_algo_entries ──────────────────────────────────────────────

def test_summarise_empty_entries():
    s = summarise_graph_algo_entries([])
    assert s.n_runs == 0
    assert s.n_found == 0
    assert s.mean_cost == pytest.approx(0.0)
    assert s.min_cost is None


def test_summarise_all_not_found():
    entries = [make_graph_algo_entry(i, "d", False, 0.0, 0, 5) for i in range(3)]
    s = summarise_graph_algo_entries(entries)
    assert s.n_found == 0
    assert s.best_run_id is None


def test_summarise_normal():
    entries = _make_entries()
    s = summarise_graph_algo_entries(entries)
    assert s.n_runs == 4
    assert s.n_found == 3
    assert s.min_cost == pytest.approx(3.0)
    assert s.best_run_id == 1


def test_summarise_mean_cost():
    entries = _make_entries()
    s = summarise_graph_algo_entries(entries)
    assert s.mean_cost == pytest.approx((5.0 + 3.0 + 7.0) / 3)


# ── Filters ───────────────────────────────────────────────────────────────────

def test_filter_by_found():
    entries = _make_entries()
    found = filter_graph_algo_by_found(entries)
    assert all(e.found for e in found)
    assert len(found) == 3


def test_filter_by_max_cost():
    entries = _make_entries()
    filtered = filter_graph_algo_by_max_cost(entries, 5.0)
    costs = [e.cost for e in filtered]
    assert all(c <= 5.0 for c in costs)


def test_filter_by_algorithm():
    entries = _make_entries()
    filtered = filter_graph_algo_by_algorithm(entries, "dijkstra")
    assert all(e.algorithm == "dijkstra" for e in filtered)
    assert len(filtered) == 3


def test_filter_by_min_path_length():
    entries = _make_entries()
    filtered = filter_graph_algo_by_min_path_length(entries, 3)
    assert all(e.path_length >= 3 for e in filtered)


def test_top_k_cheapest_paths():
    entries = _make_entries()
    top = top_k_cheapest_paths(entries, 2)
    assert len(top) == 2
    assert top[0].cost <= top[1].cost


def test_best_graph_algo_entry():
    entries = _make_entries()
    best = best_graph_algo_entry(entries)
    assert best is not None
    assert best.cost == pytest.approx(3.0)


def test_best_graph_algo_entry_none():
    assert best_graph_algo_entry([]) is None


# ── graph_algo_cost_stats ─────────────────────────────────────────────────────

def test_graph_algo_cost_stats_empty():
    stats = graph_algo_cost_stats([])
    assert stats["count"] == 0


def test_graph_algo_cost_stats_values():
    entries = _make_entries()
    stats = graph_algo_cost_stats(entries)
    assert stats["min"] == pytest.approx(3.0)
    assert stats["max"] == pytest.approx(7.0)
    assert stats["count"] == 3


# ── compare / batch ───────────────────────────────────────────────────────────

def test_compare_graph_algo_summaries():
    e1 = _make_entries()[:2]
    e2 = _make_entries()[2:]
    s1 = summarise_graph_algo_entries(e1)
    s2 = summarise_graph_algo_entries(e2)
    delta = compare_graph_algo_summaries(s1, s2)
    assert "delta_mean_cost" in delta
    assert "same_best" in delta


def test_batch_summarise_graph_algo_entries():
    groups = [_make_entries()[:2], _make_entries()[2:]]
    summaries = batch_summarise_graph_algo_entries(groups)
    assert len(summaries) == 2


# ── BatchResultEntry ──────────────────────────────────────────────────────────

def test_batch_result_entry_success_ratio():
    e = make_batch_result_entry(0, 100, 80, 20, 3, "algo")
    assert e.success_ratio == pytest.approx(0.8)


def test_batch_result_entry_zero_total():
    e = BatchResultEntry(batch_id=0, total=0, n_success=0,
                         n_failed=0, n_retried=0, algorithm="x")
    assert e.success_ratio == pytest.approx(0.0)


# ── summarise_batch_result_entries ────────────────────────────────────────────

def test_summarise_batch_empty():
    s = summarise_batch_result_entries([])
    assert s.n_batches == 0
    assert s.best_batch_id is None


def test_summarise_batch_normal():
    entries = _make_batch_entries()
    s = summarise_batch_result_entries(entries)
    assert s.n_batches == 3
    assert s.best_batch_id == 2  # 50/50 = 1.0


def test_filter_batch_by_min_ratio():
    entries = _make_batch_entries()
    filtered = filter_batch_results_by_min_ratio(entries, 0.9)
    assert all(e.success_ratio >= 0.9 for e in filtered)


def test_filter_batch_by_algorithm():
    entries = _make_batch_entries()
    filtered = filter_batch_results_by_algorithm(entries, "algo_a")
    assert all(e.algorithm == "algo_a" for e in filtered)
    assert len(filtered) == 2


def test_filter_batch_by_max_retries():
    entries = _make_batch_entries()
    filtered = filter_batch_results_by_max_retries(entries, 2)
    assert all(e.n_retried <= 2 for e in filtered)


def test_top_k_batch_results():
    entries = _make_batch_entries()
    top = top_k_batch_results(entries, 2)
    assert len(top) == 2
    assert top[0].success_ratio >= top[1].success_ratio


def test_best_batch_result_entry():
    entries = _make_batch_entries()
    best = best_batch_result_entry(entries)
    assert best is not None
    assert best.success_ratio == pytest.approx(1.0)


def test_batch_result_success_stats_empty():
    stats = batch_result_success_stats([])
    assert stats["count"] == 0


def test_compare_batch_result_summaries():
    entries = _make_batch_entries()
    s1 = summarise_batch_result_entries(entries[:2])
    s2 = summarise_batch_result_entries(entries[1:])
    delta = compare_batch_result_summaries(s1, s2)
    assert "delta_mean_success_ratio" in delta
    assert "same_best" in delta


def test_batch_summarise_batch_result_entries():
    groups = [_make_batch_entries()[:2], _make_batch_entries()[1:]]
    summaries = batch_summarise_batch_result_entries(groups)
    assert len(summaries) == 2
