"""Utilities for aggregating and analysing ranked fragment pair results.

Provides dataclasses for storing per-pair ranking entries, summary
statistics, and functions for filtering, ranking, and batch analysis
of pair ranking outcomes.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class RankResultConfig:
    """Configuration for rank-result analysis utilities."""

    good_threshold: float = 0.7
    poor_threshold: float = 0.3
    top_k: int = 10

    def __post_init__(self) -> None:
        if not (0 <= self.good_threshold <= 1):
            raise ValueError("good_threshold must be in [0, 1]")
        if not (0 <= self.poor_threshold <= 1):
            raise ValueError("poor_threshold must be in [0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class RankResultEntry:
    """A single ranked fragment pair entry."""

    frag_i: int
    frag_j: int
    score: float
    rank: int
    channel_scores: Dict[str, float] = field(default_factory=dict)
    method: str = "rank"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> Tuple[int, int]:
        return (min(self.frag_i, self.frag_j), max(self.frag_i, self.frag_j))

    @property
    def is_top_match(self) -> bool:
        return self.rank == 1

    @property
    def dominant_channel(self) -> Optional[str]:
        if not self.channel_scores:
            return None
        return max(self.channel_scores, key=lambda k: self.channel_scores[k])


@dataclass
class RankResultSummary:
    """Aggregated statistics over a collection of rank entries."""

    n_entries: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    mean_rank: float
    n_top_matches: int
    params: Dict[str, Any] = field(default_factory=dict)


# ── Factory functions ─────────────────────────────────────────────────────────

def make_rank_result_entry(
    frag_i: int,
    frag_j: int,
    score: float,
    rank: int,
    *,
    channel_scores: Optional[Dict[str, float]] = None,
    method: str = "rank",
    params: Optional[Dict[str, Any]] = None,
) -> RankResultEntry:
    """Create a *RankResultEntry* from raw values."""
    return RankResultEntry(
        frag_i=frag_i,
        frag_j=frag_j,
        score=score,
        rank=rank,
        channel_scores=channel_scores or {},
        method=method,
        params=params or {},
    )


def entries_from_ranked_pairs(
    pairs: Sequence[Tuple[int, int]],
    scores: Sequence[float],
    ranks: Optional[Sequence[int]] = None,
    *,
    method: str = "rank",
) -> List[RankResultEntry]:
    """Build a list of *RankResultEntry* from parallel sequences."""
    if len(pairs) != len(scores):
        raise ValueError("pairs and scores must have the same length")
    _ranks = ranks if ranks is not None else list(range(1, len(pairs) + 1))
    return [
        make_rank_result_entry(i, j, s, r, method=method)
        for (i, j), s, r in zip(pairs, scores, _ranks)
    ]


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarise_rank_results(
    entries: Sequence[RankResultEntry],
) -> RankResultSummary:
    """Compute aggregate statistics over *entries*."""
    if not entries:
        return RankResultSummary(
            n_entries=0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            mean_rank=0.0,
            n_top_matches=0,
        )
    scores = [e.score for e in entries]
    return RankResultSummary(
        n_entries=len(entries),
        mean_score=statistics.mean(scores),
        std_score=statistics.pstdev(scores),
        min_score=min(scores),
        max_score=max(scores),
        mean_rank=statistics.mean(e.rank for e in entries),
        n_top_matches=sum(1 for e in entries if e.is_top_match),
    )


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_high_rank_entries(
    entries: Sequence[RankResultEntry],
    threshold: float = 0.7,
) -> List[RankResultEntry]:
    """Return entries whose score >= *threshold*."""
    return [e for e in entries if e.score >= threshold]


def filter_low_rank_entries(
    entries: Sequence[RankResultEntry],
    threshold: float = 0.3,
) -> List[RankResultEntry]:
    """Return entries whose score < *threshold*."""
    return [e for e in entries if e.score < threshold]


def filter_by_rank_position(
    entries: Sequence[RankResultEntry],
    max_rank: int,
) -> List[RankResultEntry]:
    """Keep entries whose rank <= *max_rank*."""
    return [e for e in entries if e.rank <= max_rank]


def filter_rank_by_score_range(
    entries: Sequence[RankResultEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[RankResultEntry]:
    """Keep entries whose score is in [lo, hi]."""
    return [e for e in entries if lo <= e.score <= hi]


def filter_rank_by_dominant_channel(
    entries: Sequence[RankResultEntry],
    channel: str,
) -> List[RankResultEntry]:
    """Keep entries whose dominant channel is *channel*."""
    return [e for e in entries if e.dominant_channel == channel]


# ── Ranking helpers ───────────────────────────────────────────────────────────

def top_k_rank_entries(
    entries: Sequence[RankResultEntry],
    k: int = 10,
) -> List[RankResultEntry]:
    """Return the *k* highest-scoring entries."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_rank_entry(
    entries: Sequence[RankResultEntry],
) -> Optional[RankResultEntry]:
    """Return the highest-scoring entry, or None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.score)


def rerank_entries(
    entries: Sequence[RankResultEntry],
    *,
    ascending: bool = False,
) -> List[RankResultEntry]:
    """Return entries with ranks reassigned by score order."""
    sorted_entries = sorted(entries, key=lambda e: e.score,
                            reverse=not ascending)
    result = []
    for new_rank, entry in enumerate(sorted_entries, start=1):
        result.append(make_rank_result_entry(
            entry.frag_i, entry.frag_j, entry.score, new_rank,
            channel_scores=entry.channel_scores, method=entry.method,
            params=entry.params,
        ))
    return result


# ── Statistics ────────────────────────────────────────────────────────────────

def rank_result_stats(
    entries: Sequence[RankResultEntry],
) -> Dict[str, float]:
    """Return a dict with descriptive statistics of the entries."""
    if not entries:
        return {"count": 0, "mean_score": 0.0, "std_score": 0.0,
                "min_score": 0.0, "max_score": 0.0, "mean_rank": 0.0}
    scores = [e.score for e in entries]
    return {
        "count": len(entries),
        "mean_score": statistics.mean(scores),
        "std_score": statistics.pstdev(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "mean_rank": statistics.mean(e.rank for e in entries),
        "n_top": sum(1 for e in entries if e.is_top_match),
    }


def compare_rank_summaries(
    a: RankResultSummary,
    b: RankResultSummary,
) -> Dict[str, float]:
    """Return deltas (a − b) for main scalar fields."""
    return {
        "d_mean_score": a.mean_score - b.mean_score,
        "d_std_score": a.std_score - b.std_score,
        "d_mean_rank": a.mean_rank - b.mean_rank,
        "d_n_top": a.n_top_matches - b.n_top_matches,
        "d_n_entries": a.n_entries - b.n_entries,
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_summarise_rank_results(
    groups: Sequence[Sequence[RankResultEntry]],
) -> List[RankResultSummary]:
    """Summarise multiple groups of entries in one call."""
    return [summarise_rank_results(g) for g in groups]
