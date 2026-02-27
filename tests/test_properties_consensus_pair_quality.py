"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.consensus_score_utils
  - puzzle_reconstruction.utils.pair_score_utils
  - puzzle_reconstruction.utils.quality_score_utils

consensus_score_utils:
    make_consensus_entry:       vote_fraction ∈ [0, 1]; is_consensus = (frac >= threshold)
    entries_from_votes:         len = len(pair_votes); all fracs ∈ [0, 1]
    summarise_consensus:        n_pairs = len; n_consensus <= n_pairs
    filter_consensus_pairs:     all .is_consensus = True
    filter_non_consensus:       all .is_consensus = False
    filter_by_vote_fraction:    all .vote_fraction >= min_fraction
    top_k_consensus_entries:    len <= k; sorted descending
    consensus_score_stats:      count = n; fracs ∈ [0, 1]
    agreement_score:            ∈ [0, 1]

pair_score_utils:
    make_pair_score_entry:      fields set; pair_key canonical
    entries_from_pair_results:  len = len(pairs)
    summarise_pair_scores:      n_entries = len; min <= mean <= max
    filter_strong_pair_matches: all .score >= threshold
    filter_weak_pair_matches:   all .score <= threshold
    top_k_pair_entries:         len <= k; sorted descending
    pair_score_stats:           count = n; max >= mean >= min

quality_score_utils:
    make_quality_entry:         is_acceptable = (overall >= min_overall)
    entries_from_reports:       len = len(reports)
    summarise_quality:          n_total = len; n_acceptable + n_rejected = n_total
    filter_acceptable:          all .is_acceptable = True
    filter_rejected:            all .is_acceptable = False
    top_k_quality_entries:      len <= k; sorted descending
    quality_score_stats:        count = n; n_acceptable + n_rejected = count
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.consensus_score_utils import (
    ConsensusScoreConfig,
    ConsensusScoreEntry,
    ConsensusSummary,
    make_consensus_entry,
    entries_from_votes,
    summarise_consensus,
    filter_consensus_pairs,
    filter_non_consensus,
    filter_by_vote_fraction,
    top_k_consensus_entries,
    consensus_score_stats,
    agreement_score,
)
from puzzle_reconstruction.utils.pair_score_utils import (
    PairScoreConfig,
    PairScoreEntry,
    PairScoreSummary,
    make_pair_score_entry,
    entries_from_pair_results,
    summarise_pair_scores,
    filter_strong_pair_matches,
    filter_weak_pair_matches,
    top_k_pair_entries,
    pair_score_stats,
)
from puzzle_reconstruction.utils.quality_score_utils import (
    QualityScoreConfig,
    QualityScoreEntry,
    QualitySummary,
    make_quality_entry,
    entries_from_reports,
    summarise_quality,
    filter_acceptable,
    filter_rejected,
    top_k_quality_entries,
    quality_score_stats,
)

RNG = np.random.default_rng(5555)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_consensus_entries(
    n: int = 8, n_methods: int = 5, seed: int = 0
) -> List[ConsensusScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_consensus_entry(
            pair=(i, i + 1),
            vote_count=int(rng.integers(0, n_methods + 1)),
            n_methods=n_methods,
            threshold=0.5,
        )
        for i in range(n)
    ]


def _make_pair_entries(n: int = 8, seed: int = 0) -> List[PairScoreEntry]:
    rng = np.random.default_rng(seed)
    pairs = [(i, i + 1) for i in range(n)]
    scores = [float(rng.uniform(0.0, 1.0)) for _ in range(n)]
    return entries_from_pair_results(pairs, scores)


def _make_quality_entries(n: int = 8, seed: int = 0) -> List[QualityScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_quality_entry(
            image_id=i,
            blur_score=float(rng.uniform(0, 1)),
            noise_score=float(rng.uniform(0, 1)),
            contrast_score=float(rng.uniform(0, 1)),
            completeness=float(rng.uniform(0, 1)),
            overall=float(rng.uniform(0, 1)),
        )
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# consensus_score_utils — make_consensus_entry, vote_fraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsensusEntry:
    """make_consensus_entry: vote_fraction and is_consensus invariants."""

    @pytest.mark.parametrize("votes,n_methods,threshold,expected", [
        (3, 5, 0.5, True),   # 3/5 = 0.6 >= 0.5
        (2, 5, 0.5, False),  # 2/5 = 0.4 < 0.5
        (5, 5, 0.5, True),   # 5/5 = 1.0 >= 0.5
        (0, 5, 0.5, False),  # 0/5 = 0.0 < 0.5
    ])
    def test_is_consensus(self, votes: int, n_methods: int,
                          threshold: float, expected: bool) -> None:
        e = make_consensus_entry((0, 1), vote_count=votes, n_methods=n_methods,
                                 threshold=threshold)
        assert e.is_consensus is expected

    @pytest.mark.parametrize("votes,n_methods", [(3, 5), (0, 5), (5, 5), (2, 4)])
    def test_vote_fraction_in_range(self, votes: int, n_methods: int) -> None:
        e = make_consensus_entry((0, 1), vote_count=votes, n_methods=n_methods)
        assert 0.0 <= e.vote_fraction <= 1.0

    def test_vote_fraction_zero_methods(self) -> None:
        e = ConsensusScoreEntry(pair=(0, 1), vote_count=0,
                                n_methods=0, is_consensus=False)
        assert e.vote_fraction == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# consensus_score_utils — entries_from_votes, summarise_consensus
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsensusFromVotes:
    """entries_from_votes and summarise_consensus invariants."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_entries_from_votes_length(self, n: int) -> None:
        pair_votes: Dict[FrozenSet[int], int] = {
            frozenset({i, i + 1}): i % 4 for i in range(n)
        }
        entries = entries_from_votes(pair_votes, n_methods=5)
        assert len(entries) == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_summarise_n_pairs(self, n: int, seed: int) -> None:
        entries = _make_consensus_entries(n, seed=seed)
        s = summarise_consensus(entries)
        assert s.n_pairs == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_summarise_n_consensus_leq_total(self, n: int, seed: int) -> None:
        entries = _make_consensus_entries(n, seed=seed)
        s = summarise_consensus(entries)
        assert 0 <= s.n_consensus <= s.n_pairs

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mean_vote_fraction_in_range(self, seed: int) -> None:
        entries = _make_consensus_entries(10, seed=seed)
        s = summarise_consensus(entries)
        assert 0.0 <= s.mean_vote_fraction <= 1.0 + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_agreement_score_in_range(self, seed: int) -> None:
        entries = _make_consensus_entries(10, seed=seed)
        score = agreement_score(entries)
        assert 0.0 <= score <= 1.0

    def test_agreement_score_empty_is_zero(self) -> None:
        assert agreement_score([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# consensus_score_utils — filters and top_k
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsensusFilters:
    """filter_consensus_pairs, filter_non_consensus, filter_by_vote_fraction."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_consensus_all_consensus(self, seed: int) -> None:
        entries = _make_consensus_entries(12, seed=seed)
        for e in filter_consensus_pairs(entries):
            assert e.is_consensus is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_non_consensus_none_consensus(self, seed: int) -> None:
        entries = _make_consensus_entries(12, seed=seed)
        for e in filter_non_consensus(entries):
            assert e.is_consensus is False

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_consensus_plus_non_eq_total(self, seed: int) -> None:
        entries = _make_consensus_entries(12, seed=seed)
        assert (len(filter_consensus_pairs(entries)) +
                len(filter_non_consensus(entries))) == len(entries)

    @pytest.mark.parametrize("min_f", [0.2, 0.4, 0.6])
    def test_filter_by_vote_fraction(self, min_f: float) -> None:
        entries = _make_consensus_entries(15, seed=42)
        filtered = filter_by_vote_fraction(entries, min_fraction=min_f)
        for e in filtered:
            assert e.vote_fraction >= min_f - 1e-9

    @pytest.mark.parametrize("k", [2, 4, 6])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_consensus_entries(10, seed=k)
        top = top_k_consensus_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("k", [3, 5])
    def test_top_k_sorted_descending(self, k: int) -> None:
        entries = _make_consensus_entries(10, seed=k + 10)
        top = top_k_consensus_entries(entries, k)
        fracs = [e.vote_fraction for e in top]
        assert fracs == sorted(fracs, reverse=True)

    @pytest.mark.parametrize("n,seed", [(8, 0), (12, 1)])
    def test_consensus_stats_count(self, n: int, seed: int) -> None:
        entries = _make_consensus_entries(n, seed=seed)
        stats = consensus_score_stats(entries)
        assert stats["count"] == n


# ═══════════════════════════════════════════════════════════════════════════════
# pair_score_utils — make_pair_score_entry, pair_key, summarise
# ═══════════════════════════════════════════════════════════════════════════════

class TestPairScoreEntry:
    """make_pair_score_entry: fields and pair_key canonical."""

    def test_pair_key_canonical(self) -> None:
        e = make_pair_score_entry(5, 2, score=0.7)
        assert e.pair_key == (2, 5)

    @pytest.mark.parametrize("i,j", [(0, 1), (3, 7), (5, 2)])
    def test_pair_key_min_first(self, i: int, j: int) -> None:
        e = make_pair_score_entry(i, j, score=0.5)
        key = e.pair_key
        assert key[0] <= key[1]

    def test_dominant_channel_max(self) -> None:
        e = make_pair_score_entry(0, 1, score=0.5,
                                  channels={"a": 0.3, "b": 0.9, "c": 0.1})
        assert e.dominant_channel == "b"

    def test_dominant_channel_empty(self) -> None:
        e = make_pair_score_entry(0, 1, score=0.5)
        assert e.dominant_channel is None

    @pytest.mark.parametrize("score,expected", [
        (0.8, True), (0.7, True), (0.69, False), (0.5, False),
    ])
    def test_is_strong_match(self, score: float, expected: bool) -> None:
        e = make_pair_score_entry(0, 1, score=score)
        assert e.is_strong_match is expected


class TestPairScoreSummary:
    """summarise_pair_scores and pair_score_stats invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_summarise_n_entries(self, n: int, seed: int) -> None:
        entries = _make_pair_entries(n, seed)
        s = summarise_pair_scores(entries)
        assert s.n_entries == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_summarise_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_pair_entries(10, seed)
        s = summarise_pair_scores(entries)
        assert s.min_score <= s.mean_score + 1e-9
        assert s.mean_score <= s.max_score + 1e-9

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_pair_stats_count(self, n: int, seed: int) -> None:
        entries = _make_pair_entries(n, seed)
        stats = pair_score_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_pair_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_pair_entries(10, seed)
        stats = pair_score_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9


class TestPairScoreFilters:
    """filter_strong_pair_matches, filter_weak_pair_matches, top_k."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_strong(self, seed: int) -> None:
        entries = _make_pair_entries(15, seed)
        strong = filter_strong_pair_matches(entries)
        for e in strong:
            assert e.score >= 0.7

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_weak(self, seed: int) -> None:
        entries = _make_pair_entries(15, seed)
        weak = filter_weak_pair_matches(entries)
        for e in weak:
            assert e.score <= 0.3

    @pytest.mark.parametrize("k", [2, 4, 6])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_pair_entries(10, seed=k)
        top = top_k_pair_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("k", [3, 5])
    def test_top_k_sorted_descending(self, k: int) -> None:
        entries = _make_pair_entries(10, seed=k + 1)
        top = top_k_pair_entries(entries, k)
        scores = [e.score for e in top]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_entries_from_pair_results_length(self, n: int) -> None:
        pairs = [(i, i + 1) for i in range(n)]
        scores = [0.5] * n
        entries = entries_from_pair_results(pairs, scores)
        assert len(entries) == n


# ═══════════════════════════════════════════════════════════════════════════════
# quality_score_utils — make_quality_entry, summarise_quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityEntry:
    """make_quality_entry: is_acceptable invariant."""

    @pytest.mark.parametrize("overall,min_ov,expected", [
        (0.7, 0.5, True), (0.3, 0.5, False), (0.5, 0.5, True),
    ])
    def test_is_acceptable(self, overall: float, min_ov: float,
                           expected: bool) -> None:
        cfg = QualityScoreConfig(min_overall=min_ov)
        e = make_quality_entry(0, 0.5, 0.5, 0.5, 0.5, overall, cfg=cfg)
        assert e.is_acceptable is expected

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_entries_from_reports_length(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        reports = [
            {
                "image_id": i,
                "blur_score": float(rng.uniform(0, 1)),
                "noise_score": float(rng.uniform(0, 1)),
                "contrast_score": float(rng.uniform(0, 1)),
                "completeness": float(rng.uniform(0, 1)),
                "overall": float(rng.uniform(0, 1)),
            }
            for i in range(n)
        ]
        entries = entries_from_reports(reports)
        assert len(entries) == n


class TestQualitySummary:
    """summarise_quality and quality_score_stats invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_summarise_n_total(self, n: int, seed: int) -> None:
        entries = _make_quality_entries(n, seed)
        s = summarise_quality(entries)
        assert s.n_total == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_acceptable_plus_rejected_eq_total(self, n: int, seed: int) -> None:
        entries = _make_quality_entries(n, seed)
        s = summarise_quality(entries)
        assert s.n_acceptable + s.n_rejected == s.n_total

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_acceptable(self, seed: int) -> None:
        entries = _make_quality_entries(12, seed)
        for e in filter_acceptable(entries):
            assert e.is_acceptable is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_rejected(self, seed: int) -> None:
        entries = _make_quality_entries(12, seed)
        for e in filter_rejected(entries):
            assert e.is_acceptable is False

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_acceptable_plus_rejected_eq_entries(self, seed: int) -> None:
        entries = _make_quality_entries(12, seed)
        assert (len(filter_acceptable(entries)) +
                len(filter_rejected(entries))) == len(entries)

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_quality_entries(10, seed=k)
        top = top_k_quality_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_quality_stats_count(self, n: int, seed: int) -> None:
        entries = _make_quality_entries(n, seed)
        stats = quality_score_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_quality_stats_acceptable_plus_rejected_eq_count(self, seed: int) -> None:
        entries = _make_quality_entries(10, seed)
        stats = quality_score_stats(entries)
        assert stats["n_acceptable"] + stats["n_rejected"] == stats["count"]
