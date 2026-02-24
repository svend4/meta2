"""Extra tests for puzzle_reconstruction/utils/match_rank_utils.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(batch_id=0, n_pairs=10, n_accepted=5, top_score=0.8,
           mean_score=0.6, algorithm="greedy") -> RankingEntry:
    return RankingEntry(batch_id=batch_id, n_pairs=n_pairs,
                        n_accepted=n_accepted, top_score=top_score,
                        mean_score=mean_score, algorithm=algorithm)


def _eval_entry(run_id=0, n_pairs=10, mean_score=0.7, mean_f1=0.6,
                best_f1=0.9, algorithm="orb") -> EvalResultEntry:
    return EvalResultEntry(run_id=run_id, n_pairs=n_pairs,
                           mean_score=mean_score, mean_f1=mean_f1,
                           best_f1=best_f1, algorithm=algorithm)


# ─── RankingConfig ────────────────────────────────────────────────────────────

class TestRankingConfigExtra:
    def test_default_min_score(self):
        assert RankingConfig().min_score == pytest.approx(0.0)

    def test_default_top_k(self):
        assert RankingConfig().top_k == 10

    def test_default_deduplicate(self):
        assert RankingConfig().deduplicate is False

    def test_custom_values(self):
        cfg = RankingConfig(min_score=0.5, top_k=5, deduplicate=True)
        assert cfg.top_k == 5


# ─── RankingEntry ─────────────────────────────────────────────────────────────

class TestRankingEntryExtra:
    def test_acceptance_rate(self):
        e = _entry(n_pairs=10, n_accepted=4)
        assert e.acceptance_rate == pytest.approx(0.4)

    def test_acceptance_rate_zero_pairs(self):
        e = _entry(n_pairs=0, n_accepted=0)
        assert e.acceptance_rate == pytest.approx(0.0)

    def test_stores_algorithm(self):
        e = _entry(algorithm="hungarian")
        assert e.algorithm == "hungarian"

    def test_params_default_empty(self):
        assert _entry().params == {}

    def test_stores_batch_id(self):
        e = _entry(batch_id=42)
        assert e.batch_id == 42


# ─── make_ranking_entry ───────────────────────────────────────────────────────

class TestMakeRankingEntryExtra:
    def test_returns_entry(self):
        e = make_ranking_entry(0, 10, 5, 0.8, 0.6, "greedy")
        assert isinstance(e, RankingEntry)

    def test_params_stored(self):
        e = make_ranking_entry(0, 10, 5, 0.8, 0.6, "greedy", threshold=0.3)
        assert e.params.get("threshold") == pytest.approx(0.3)

    def test_top_score_stored(self):
        e = make_ranking_entry(1, 20, 10, 0.95, 0.7, "orb")
        assert e.top_score == pytest.approx(0.95)


# ─── summarise_ranking_entries ────────────────────────────────────────────────

class TestSummariseRankingEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_ranking_entries([])
        assert isinstance(s, RankingSummary)
        assert s.n_batches == 0
        assert s.best_batch_id is None

    def test_single_entry(self):
        s = summarise_ranking_entries([_entry(batch_id=3, top_score=0.7)])
        assert s.n_batches == 1
        assert s.best_batch_id == 3

    def test_total_pairs_summed(self):
        entries = [_entry(n_pairs=10), _entry(n_pairs=20)]
        s = summarise_ranking_entries(entries)
        assert s.total_pairs == 30


# ─── filter helpers ───────────────────────────────────────────────────────────

class TestFilterRankingExtra:
    def test_filter_by_algorithm(self):
        entries = [_entry(algorithm="a"), _entry(algorithm="b")]
        result = filter_ranking_by_algorithm(entries, "a")
        assert all(e.algorithm == "a" for e in result)

    def test_filter_by_min_top_score(self):
        entries = [_entry(top_score=0.3), _entry(top_score=0.9)]
        result = filter_ranking_by_min_top_score(entries, 0.5)
        assert all(e.top_score >= 0.5 for e in result)

    def test_filter_by_min_acceptance(self):
        entries = [_entry(n_pairs=10, n_accepted=2),
                   _entry(n_pairs=10, n_accepted=8)]
        result = filter_ranking_by_min_acceptance(entries, 0.5)
        assert len(result) == 1

    def test_top_k_ranking_entries(self):
        entries = [_entry(top_score=0.1), _entry(top_score=0.9),
                   _entry(top_score=0.5)]
        top = top_k_ranking_entries(entries, 2)
        assert len(top) == 2
        assert top[0].top_score == pytest.approx(0.9)

    def test_best_ranking_entry_empty(self):
        assert best_ranking_entry([]) is None

    def test_best_ranking_entry(self):
        entries = [_entry(top_score=0.3), _entry(top_score=0.8)]
        best = best_ranking_entry(entries)
        assert best.top_score == pytest.approx(0.8)


# ─── ranking_score_stats ──────────────────────────────────────────────────────

class TestRankingScoreStatsExtra:
    def test_empty_returns_zeros(self):
        s = ranking_score_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        entries = [_entry(top_score=0.5), _entry(top_score=0.7)]
        s = ranking_score_stats(entries)
        assert s["count"] == 2

    def test_min_max(self):
        entries = [_entry(top_score=0.2), _entry(top_score=0.8)]
        s = ranking_score_stats(entries)
        assert s["min"] == pytest.approx(0.2)
        assert s["max"] == pytest.approx(0.8)


# ─── compare_ranking_summaries ────────────────────────────────────────────────

class TestCompareRankingSummariesExtra:
    def test_returns_dict(self):
        s = summarise_ranking_entries([_entry()])
        result = compare_ranking_summaries(s, s)
        assert isinstance(result, dict)

    def test_delta_zero_identical(self):
        s = summarise_ranking_entries([_entry()])
        d = compare_ranking_summaries(s, s)
        assert d["delta_mean_top_score"] == pytest.approx(0.0)

    def test_same_best(self):
        s = summarise_ranking_entries([_entry(batch_id=1)])
        d = compare_ranking_summaries(s, s)
        assert d["same_best"] is True


# ─── batch_summarise_ranking_entries ──────────────────────────────────────────

class TestBatchSummariseRankingExtra:
    def test_returns_list(self):
        result = batch_summarise_ranking_entries([[_entry()], [_entry()]])
        assert isinstance(result, list) and len(result) == 2

    def test_empty_groups(self):
        result = batch_summarise_ranking_entries([[], []])
        assert all(s.n_batches == 0 for s in result)


# ─── EvalResultEntry ──────────────────────────────────────────────────────────

class TestEvalResultEntryExtra:
    def test_stores_mean_f1(self):
        e = _eval_entry(mean_f1=0.75)
        assert e.mean_f1 == pytest.approx(0.75)

    def test_stores_best_f1(self):
        e = _eval_entry(best_f1=0.95)
        assert e.best_f1 == pytest.approx(0.95)

    def test_params_default_empty(self):
        assert _eval_entry().params == {}


# ─── make_eval_result_entry ───────────────────────────────────────────────────

class TestMakeEvalResultEntryExtra:
    def test_returns_entry(self):
        e = make_eval_result_entry(0, 10, 0.7, 0.6, 0.9, "orb")
        assert isinstance(e, EvalResultEntry)

    def test_extra_params_stored(self):
        e = make_eval_result_entry(0, 5, 0.5, 0.4, 0.8, "sift", k=2)
        assert e.params.get("k") == 2


# ─── summarise_eval_result_entries ────────────────────────────────────────────

class TestSummariseEvalResultEntriesExtra:
    def test_empty_returns_summary(self):
        s = summarise_eval_result_entries([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_single_entry_best(self):
        s = summarise_eval_result_entries([_eval_entry(run_id=7)])
        assert s.best_run_id == 7

    def test_total_pairs(self):
        entries = [_eval_entry(n_pairs=5), _eval_entry(n_pairs=15)]
        s = summarise_eval_result_entries(entries)
        assert s.total_pairs == 20


# ─── eval filter helpers ──────────────────────────────────────────────────────

class TestFilterEvalExtra:
    def test_filter_by_min_f1(self):
        entries = [_eval_entry(mean_f1=0.3), _eval_entry(mean_f1=0.8)]
        result = filter_eval_by_min_f1(entries, 0.5)
        assert len(result) == 1

    def test_filter_by_algorithm(self):
        entries = [_eval_entry(algorithm="orb"), _eval_entry(algorithm="sift")]
        result = filter_eval_by_algorithm(entries, "sift")
        assert all(e.algorithm == "sift" for e in result)

    def test_top_k_eval_entries(self):
        entries = [_eval_entry(best_f1=0.5), _eval_entry(best_f1=0.9)]
        top = top_k_eval_entries(entries, 1)
        assert top[0].best_f1 == pytest.approx(0.9)

    def test_best_eval_entry_empty(self):
        assert best_eval_entry([]) is None

    def test_best_eval_entry(self):
        entries = [_eval_entry(best_f1=0.4), _eval_entry(best_f1=0.85)]
        best = best_eval_entry(entries)
        assert best.best_f1 == pytest.approx(0.85)


# ─── eval_f1_stats ────────────────────────────────────────────────────────────

class TestEvalF1StatsExtra:
    def test_empty_returns_zeros(self):
        s = eval_f1_stats([])
        assert s["count"] == 0

    def test_count_correct(self):
        s = eval_f1_stats([_eval_entry(), _eval_entry()])
        assert s["count"] == 2


# ─── compare_eval_summaries / batch ───────────────────────────────────────────

class TestCompareEvalSummariesExtra:
    def test_returns_dict(self):
        s = summarise_eval_result_entries([_eval_entry()])
        result = compare_eval_summaries(s, s)
        assert isinstance(result, dict)

    def test_delta_f1_zero_identical(self):
        s = summarise_eval_result_entries([_eval_entry()])
        d = compare_eval_summaries(s, s)
        assert d["delta_mean_f1"] == pytest.approx(0.0)

    def test_batch_summarise_eval_length(self):
        groups = [[_eval_entry()], [_eval_entry(), _eval_entry()]]
        result = batch_summarise_eval_entries(groups)
        assert len(result) == 2
