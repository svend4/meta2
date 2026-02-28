"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.annealing_score_utils
  - puzzle_reconstruction.utils.assembly_score_utils
  - puzzle_reconstruction.utils.candidate_rank_utils

annealing_score_utils:
    make_annealing_entry:     fields preserved; accepted bool
    entries_from_log:         len = len(log); fields copied
    summarise_annealing:      n_iterations = len; acceptance_rate ∈ [0,1];
                              best_score >= all current_scores
    filter_accepted:          all .accepted = True
    filter_rejected:          all .accepted = False
    filter_by_min_score:      all .current_score >= min_score
    filter_by_temperature_range: all .temperature ∈ [t_min, t_max]
    top_k_entries:            len <= k; sorted descending
    annealing_score_stats:    count = n; min <= mean <= max
    best_entry:               .current_score = max of all
    batch_summarise:          len = len(logs)

assembly_score_utils:
    make_assembly_entry:      fields set; total_score >= 0
    summarise_assemblies:     n_total = len; n_good + n_poor = n_total;
                              max_score >= mean >= min_score
    filter_good_assemblies:   all .total_score > 0.5
    top_k_assembly_entries:   len <= k; sorted descending
    assembly_score_stats:     count = n; max >= mean >= min

candidate_rank_utils:
    make_candidate_entry:     score = input; is_selected = (score >= min_score)
    entries_from_pairs:       ranks ascending; len = len(pairs)
    summarise_rankings:       n_selected + n_rejected = n_total; mean ∈ [min, max]
    filter_selected:          all .is_selected = True
    top_k_candidate_entries:  len <= k; sorted descending
    candidate_rank_stats:     count = n; max >= mean >= min
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
    top_k_entries as annealing_top_k,
    annealing_score_stats,
    best_entry as annealing_best,
    batch_summarise as annealing_batch_summarise,
)
from puzzle_reconstruction.utils.assembly_score_utils import (
    AssemblyScoreConfig,
    AssemblyScoreEntry,
    AssemblySummary,
    make_assembly_entry,
    summarise_assemblies,
    filter_good_assemblies,
    top_k_assembly_entries,
    assembly_score_stats,
)
from puzzle_reconstruction.utils.candidate_rank_utils import (
    CandidateRankConfig,
    CandidateRankEntry,
    CandidateRankSummary,
    make_candidate_entry,
    entries_from_pairs,
    summarise_rankings,
    filter_selected,
    top_k_candidate_entries,
    candidate_rank_stats,
)

import numpy as np

RNG = np.random.default_rng(7777)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_annealing_entries(n: int = 10, seed: int = 0) -> List[AnnealingScoreEntry]:
    rng = np.random.default_rng(seed)
    entries = []
    best = 0.0
    for i in range(n):
        t = float(rng.uniform(0.01, 100.0))
        s = float(rng.uniform(0.0, 1.0))
        best = max(best, s)
        accepted = bool(rng.integers(0, 2))
        entries.append(make_annealing_entry(
            iteration=i, temperature=t, current_score=s,
            best_score=best, accepted=accepted,
        ))
    return entries


def _make_assembly_entries(n: int = 6, seed: int = 0) -> List[AssemblyScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_assembly_entry(
            run_id=i,
            method="greedy",
            n_fragments=int(rng.integers(2, 20)),
            total_score=float(rng.uniform(0.0, 1.0)),
        )
        for i in range(n)
    ]


def _make_candidate_entries(n: int = 8, seed: int = 0) -> List[CandidateRankEntry]:
    rng = np.random.default_rng(seed)
    pairs = [
        {"idx1": i, "idx2": i + 1, "score": float(rng.uniform(0.0, 1.0))}
        for i in range(n)
    ]
    return entries_from_pairs(pairs)


# ═══════════════════════════════════════════════════════════════════════════════
# annealing_score_utils — make_annealing_entry, entries_from_log
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnealingEntry:
    """make_annealing_entry and entries_from_log invariants."""

    def test_make_entry_fields(self) -> None:
        e = make_annealing_entry(
            iteration=3, temperature=50.0, current_score=0.7,
            best_score=0.8, accepted=True,
        )
        assert e.iteration == 3
        assert e.temperature == pytest.approx(50.0)
        assert e.current_score == pytest.approx(0.7)
        assert e.best_score == pytest.approx(0.8)
        assert e.accepted is True

    @pytest.mark.parametrize("n", [5, 10, 15])
    def test_entries_from_log_length(self, n: int) -> None:
        log = [
            {"iteration": i, "temperature": 10.0, "current_score": 0.5,
             "best_score": 0.5, "accepted": True}
            for i in range(n)
        ]
        entries = entries_from_log(log)
        assert len(entries) == n

    def test_entries_from_log_fields_copied(self) -> None:
        log = [{"iteration": 5, "temperature": 25.0, "current_score": 0.6,
                "best_score": 0.7, "accepted": False}]
        entries = entries_from_log(log)
        assert entries[0].iteration == 5
        assert entries[0].temperature == pytest.approx(25.0)
        assert entries[0].accepted is False


# ═══════════════════════════════════════════════════════════════════════════════
# annealing_score_utils — summarise_annealing
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummariseAnnealing:
    """summarise_annealing: summary invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1), (20, 2)])
    def test_n_iterations(self, n: int, seed: int) -> None:
        entries = _make_annealing_entries(n, seed)
        s = summarise_annealing(entries)
        assert s.n_iterations == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_acceptance_rate_in_range(self, n: int, seed: int) -> None:
        entries = _make_annealing_entries(n, seed)
        s = summarise_annealing(entries)
        assert 0.0 <= s.acceptance_rate <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_best_score_geq_final(self, seed: int) -> None:
        entries = _make_annealing_entries(10, seed)
        s = summarise_annealing(entries)
        assert s.best_score >= s.final_score - 1e-9

    def test_empty_entries_zero_iterations(self) -> None:
        s = summarise_annealing([])
        assert s.n_iterations == 0


# ═══════════════════════════════════════════════════════════════════════════════
# annealing_score_utils — filters and top_k
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnealingFilters:
    """filter_accepted, filter_rejected, filter_by_min_score, top_k_entries."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_accepted_all_accepted(self, seed: int) -> None:
        entries = _make_annealing_entries(15, seed)
        accepted = filter_accepted(entries)
        for e in accepted:
            assert e.accepted is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_rejected_all_rejected(self, seed: int) -> None:
        entries = _make_annealing_entries(15, seed)
        rejected = filter_rejected(entries)
        for e in rejected:
            assert e.accepted is False

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_accepted_plus_rejected_eq_total(self, seed: int) -> None:
        entries = _make_annealing_entries(15, seed)
        assert len(filter_accepted(entries)) + len(filter_rejected(entries)) == len(entries)

    @pytest.mark.parametrize("min_s", [0.3, 0.5, 0.7])
    def test_filter_by_min_score(self, min_s: float) -> None:
        entries = _make_annealing_entries(20, seed=42)
        filtered = filter_by_min_score(entries, min_s)
        for e in filtered:
            assert e.current_score >= min_s

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_top_k_length_leq_k(self, k: int) -> None:
        entries = _make_annealing_entries(10, seed=0)
        top = annealing_top_k(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("k", [3, 5])
    def test_top_k_sorted_descending(self, k: int) -> None:
        entries = _make_annealing_entries(10, seed=1)
        top = annealing_top_k(entries, k)
        scores = [e.current_score for e in top]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_temperature_range(self) -> None:
        entries = _make_annealing_entries(20, seed=5)
        filtered = filter_by_temperature_range(entries, t_min=10.0, t_max=80.0)
        for e in filtered:
            assert 10.0 <= e.temperature <= 80.0


# ═══════════════════════════════════════════════════════════════════════════════
# annealing_score_utils — stats and best_entry
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnealingStats:
    """annealing_score_stats and best_entry invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1), (15, 2)])
    def test_stats_count(self, n: int, seed: int) -> None:
        entries = _make_annealing_entries(n, seed)
        stats = annealing_score_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_annealing_entries(10, seed)
        stats = annealing_score_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_best_entry_max_score(self, seed: int) -> None:
        entries = _make_annealing_entries(10, seed)
        best = annealing_best(entries)
        assert best is not None
        max_score = max(e.current_score for e in entries)
        assert best.current_score == pytest.approx(max_score)

    def test_best_entry_empty_is_none(self) -> None:
        assert annealing_best([]) is None

    @pytest.mark.parametrize("n_logs", [3, 5])
    def test_batch_summarise_length(self, n_logs: int) -> None:
        logs = [
            [{"iteration": i, "temperature": 10.0, "current_score": 0.5,
              "best_score": 0.5, "accepted": True} for i in range(5)]
            for _ in range(n_logs)
        ]
        results = annealing_batch_summarise(logs)
        assert len(results) == n_logs


# ═══════════════════════════════════════════════════════════════════════════════
# assembly_score_utils — make_assembly_entry, summarise_assemblies
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssemblyScoreUtils:
    """assembly_score_utils invariants."""

    def test_make_entry_fields(self) -> None:
        e = make_assembly_entry(
            run_id=1, method="genetic", n_fragments=10, total_score=0.75,
        )
        assert e.run_id == 1
        assert e.method == "genetic"
        assert e.n_fragments == 10
        assert e.total_score == pytest.approx(0.75)

    @pytest.mark.parametrize("n", [3, 6, 10])
    def test_summarise_n_total(self, n: int) -> None:
        entries = _make_assembly_entries(n, seed=n)
        s = summarise_assemblies(entries)
        assert s.n_total == n

    @pytest.mark.parametrize("n", [6, 8])
    def test_summarise_good_plus_poor_eq_total(self, n: int) -> None:
        entries = _make_assembly_entries(n, seed=n)
        s = summarise_assemblies(entries)
        assert s.n_good + s.n_poor == s.n_total

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_summarise_max_geq_mean_geq_min(self, seed: int) -> None:
        entries = _make_assembly_entries(8, seed)
        s = summarise_assemblies(entries)
        assert s.min_score <= s.mean_score + 1e-9
        assert s.mean_score <= s.max_score + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_good_assemblies(self, seed: int) -> None:
        entries = _make_assembly_entries(12, seed)
        good = filter_good_assemblies(entries)
        for e in good:
            assert e.total_score > 0.5

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_assembly_length(self, k: int) -> None:
        entries = _make_assembly_entries(8, seed=k)
        top = top_k_assembly_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_assembly_stats_count(self, n: int, seed: int) -> None:
        entries = _make_assembly_entries(n, seed)
        stats = assembly_score_stats(entries)
        assert stats["n"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_assembly_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_assembly_entries(8, seed)
        stats = assembly_score_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9

    def test_score_per_fragment_nonneg(self) -> None:
        e = make_assembly_entry(
            run_id=0, method="sa", n_fragments=5, total_score=0.6,
        )
        assert e.score_per_fragment >= 0.0
        assert e.score_per_fragment == pytest.approx(0.6 / 5)


# ═══════════════════════════════════════════════════════════════════════════════
# candidate_rank_utils — make_candidate_entry, entries_from_pairs
# ═══════════════════════════════════════════════════════════════════════════════

class TestCandidateRankUtils:
    """candidate_rank_utils invariants."""

    @pytest.mark.parametrize("score,min_s,expected", [
        (0.8, 0.5, True), (0.3, 0.5, False), (0.5, 0.5, True),
    ])
    def test_is_selected(self, score: float, min_s: float, expected: bool) -> None:
        cfg = CandidateRankConfig(min_score=min_s)
        e = make_candidate_entry(0, 1, score=score, rank=0, cfg=cfg)
        assert e.is_selected is expected

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_entries_from_pairs_length(self, n: int) -> None:
        rng = np.random.default_rng(n)
        pairs = [{"idx1": i, "idx2": i + 1, "score": float(rng.uniform(0, 1))}
                 for i in range(n)]
        entries = entries_from_pairs(pairs)
        assert len(entries) == n

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_entries_from_pairs_ranks_ascending(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pairs = [{"idx1": i, "idx2": i + 1, "score": float(rng.uniform(0, 1))}
                 for i in range(n)]
        entries = entries_from_pairs(pairs)
        ranks = [e.rank for e in entries]
        assert ranks == list(range(n))

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_entries_sorted_by_score_desc(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pairs = [{"idx1": i, "idx2": i + 1, "score": float(rng.uniform(0, 1))}
                 for i in range(n)]
        entries = entries_from_pairs(pairs)
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_summarise_n_total(self, n: int, seed: int) -> None:
        entries = _make_candidate_entries(n, seed)
        s = summarise_rankings(entries)
        assert s.n_total == n

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_summarise_selected_plus_rejected_eq_total(self, n: int, seed: int) -> None:
        entries = _make_candidate_entries(n, seed)
        s = summarise_rankings(entries)
        assert s.n_selected + s.n_rejected == s.n_total

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_selected(self, seed: int) -> None:
        entries = _make_candidate_entries(12, seed)
        selected = filter_selected(entries)
        for e in selected:
            assert e.is_selected is True

    @pytest.mark.parametrize("k", [2, 4, 6])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_candidate_entries(10, seed=k)
        top = top_k_candidate_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_candidate_stats_count(self, n: int, seed: int) -> None:
        entries = _make_candidate_entries(n, seed)
        stats = candidate_rank_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_candidate_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_candidate_entries(10, seed)
        stats = candidate_rank_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9
