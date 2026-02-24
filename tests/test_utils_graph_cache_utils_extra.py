"""Extra tests for puzzle_reconstruction/utils/graph_cache_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.graph_cache_utils import (
    GraphAlgoConfig,
    GraphAlgoEntry,
    GraphAlgoSummary,
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
    BatchResultConfig,
    BatchResultEntry,
    BatchResultSummary,
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gae(rid=0, algo="dijkstra", found=True, cost=1.0, pl=3, nn=5) -> GraphAlgoEntry:
    return make_graph_algo_entry(rid, algo, found, cost, pl, nn)


def _gaes(n=4) -> list:
    return [_gae(rid=i, cost=float(i+1)) for i in range(n)]


def _bre(bid=0, total=10, ns=8, nf=2, nr=1, algo="algo") -> BatchResultEntry:
    return make_batch_result_entry(bid, total, ns, nf, nr, algo)


def _bres(n=4) -> list:
    return [_bre(bid=i, ns=i+1, total=10) for i in range(n)]


# ─── GraphAlgoConfig ──────────────────────────────────────────────────────────

class TestGraphAlgoConfigExtra:
    def test_default_require_found(self):
        assert GraphAlgoConfig().require_found is True

    def test_default_min_path_length(self):
        assert GraphAlgoConfig().min_path_length == 1

    def test_custom_values(self):
        cfg = GraphAlgoConfig(max_cost=5.0, min_path_length=2, require_found=False)
        assert cfg.max_cost == pytest.approx(5.0)
        assert cfg.min_path_length == 2


# ─── GraphAlgoEntry ───────────────────────────────────────────────────────────

class TestGraphAlgoEntryExtra:
    def test_stores_run_id(self):
        assert _gae(rid=7).run_id == 7

    def test_stores_algorithm(self):
        assert _gae(algo="bfs").algorithm == "bfs"

    def test_stores_found(self):
        assert _gae(found=True).found is True

    def test_stores_cost(self):
        assert _gae(cost=3.5).cost == pytest.approx(3.5)


# ─── summarise_graph_algo_entries ─────────────────────────────────────────────

class TestSummariseGraphAlgoEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_graph_algo_entries(_gaes()), GraphAlgoSummary)

    def test_n_runs_correct(self):
        assert summarise_graph_algo_entries(_gaes(3)).n_runs == 3

    def test_empty_entries(self):
        s = summarise_graph_algo_entries([])
        assert s.n_runs == 0 and s.best_run_id is None

    def test_n_found_only_found(self):
        entries = [_gae(found=True), _gae(rid=1, found=False)]
        s = summarise_graph_algo_entries(entries)
        assert s.n_found == 1

    def test_min_cost_correct(self):
        s = summarise_graph_algo_entries(_gaes(4))
        assert s.min_cost == pytest.approx(1.0)


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterGraphAlgoExtra:
    def test_filter_by_found(self):
        entries = [_gae(found=True), _gae(rid=1, found=False)]
        result = filter_graph_algo_by_found(entries)
        assert all(e.found for e in result)

    def test_filter_by_max_cost(self):
        entries = _gaes(4)
        result = filter_graph_algo_by_max_cost(entries, max_cost=2.0)
        assert all(e.cost <= 2.0 for e in result)

    def test_filter_by_algorithm(self):
        entries = [_gae(algo="dijkstra"), _gae(rid=1, algo="bfs")]
        result = filter_graph_algo_by_algorithm(entries, "dijkstra")
        assert all(e.algorithm == "dijkstra" for e in result)

    def test_filter_by_min_path_length(self):
        entries = [_gae(pl=2), _gae(rid=1, pl=5)]
        result = filter_graph_algo_by_min_path_length(entries, 4)
        assert all(e.path_length >= 4 for e in result)

    def test_empty_input(self):
        assert filter_graph_algo_by_found([]) == []


# ─── top_k and best ───────────────────────────────────────────────────────────

class TestTopKBestGraphAlgoExtra:
    def test_top_k_cheapest(self):
        result = top_k_cheapest_paths(_gaes(5), 3)
        assert len(result) == 3
        costs = [e.cost for e in result]
        assert costs == sorted(costs)

    def test_best_entry_min_cost(self):
        entries = _gaes(4)
        best = best_graph_algo_entry(entries)
        assert best.cost == pytest.approx(1.0)

    def test_best_empty_is_none(self):
        assert best_graph_algo_entry([]) is None


# ─── graph_algo_cost_stats ────────────────────────────────────────────────────

class TestGraphAlgoCostStatsExtra:
    def test_returns_dict(self):
        assert isinstance(graph_algo_cost_stats(_gaes()), dict)

    def test_keys_present(self):
        s = graph_algo_cost_stats(_gaes(3))
        for k in ("min", "max", "mean", "std", "count"):
            assert k in s

    def test_empty_entries(self):
        assert graph_algo_cost_stats([])["count"] == 0


# ─── compare and batch ────────────────────────────────────────────────────────

class TestCompareGraphAlgoExtra:
    def test_returns_dict(self):
        s = summarise_graph_algo_entries(_gaes(3))
        assert isinstance(compare_graph_algo_summaries(s, s), dict)

    def test_identical_zero_deltas(self):
        s = summarise_graph_algo_entries(_gaes(3))
        d = compare_graph_algo_summaries(s, s)
        assert d["delta_mean_cost"] == pytest.approx(0.0)

    def test_batch_summarise(self):
        result = batch_summarise_graph_algo_entries([_gaes(2), _gaes(3)])
        assert len(result) == 2


# ─── BatchResultEntry ─────────────────────────────────────────────────────────

class TestBatchResultEntryExtra:
    def test_success_ratio(self):
        e = _bre(total=10, ns=7)
        assert e.success_ratio == pytest.approx(0.7)

    def test_success_ratio_zero_total(self):
        e = _bre(total=0, ns=0)
        assert e.success_ratio == pytest.approx(0.0)

    def test_stores_algorithm(self):
        assert _bre(algo="my_algo").algorithm == "my_algo"


# ─── summarise_batch_result_entries ──────────────────────────────────────────

class TestSummariseBatchResultEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_batch_result_entries(_bres()), BatchResultSummary)

    def test_n_batches_correct(self):
        assert summarise_batch_result_entries(_bres(3)).n_batches == 3

    def test_empty_entries(self):
        s = summarise_batch_result_entries([])
        assert s.n_batches == 0 and s.best_batch_id is None


# ─── batch result filter / top_k / best ──────────────────────────────────────

class TestBatchResultFiltersExtra:
    def test_filter_by_min_ratio(self):
        entries = [_bre(bid=0, total=10, ns=9), _bre(bid=1, total=10, ns=3)]
        result = filter_batch_results_by_min_ratio(entries, min_ratio=0.8)
        assert all(e.success_ratio >= 0.8 for e in result)

    def test_filter_by_algorithm(self):
        entries = [_bre(algo="a"), _bre(bid=1, algo="b")]
        result = filter_batch_results_by_algorithm(entries, "a")
        assert all(e.algorithm == "a" for e in result)

    def test_filter_by_max_retries(self):
        entries = [_bre(nr=0), _bre(bid=1, nr=5)]
        result = filter_batch_results_by_max_retries(entries, max_retries=2)
        assert all(e.n_retried <= 2 for e in result)

    def test_top_k(self):
        result = top_k_batch_results(_bres(5), 3)
        assert len(result) == 3

    def test_best_entry(self):
        entries = [_bre(bid=0, total=10, ns=3), _bre(bid=1, total=10, ns=9)]
        best = best_batch_result_entry(entries)
        assert best.success_ratio == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_batch_result_entry([]) is None


# ─── batch_result_success_stats ───────────────────────────────────────────────

class TestBatchResultSuccessStatsExtra:
    def test_returns_dict(self):
        assert isinstance(batch_result_success_stats(_bres()), dict)

    def test_keys_present(self):
        for k in ("min", "max", "mean", "std", "count"):
            assert k in batch_result_success_stats(_bres(3))

    def test_empty_returns_zero_count(self):
        assert batch_result_success_stats([])["count"] == 0


# ─── compare and batch_summarise ──────────────────────────────────────────────

class TestCompareBatchResultExtra:
    def test_returns_dict(self):
        s = summarise_batch_result_entries(_bres(3))
        assert isinstance(compare_batch_result_summaries(s, s), dict)

    def test_identical_zero_deltas(self):
        s = summarise_batch_result_entries(_bres(3))
        d = compare_batch_result_summaries(s, s)
        assert d["delta_mean_success_ratio"] == pytest.approx(0.0)

    def test_batch_summarise(self):
        result = batch_summarise_batch_result_entries([_bres(2), _bres(3)])
        assert len(result) == 2
