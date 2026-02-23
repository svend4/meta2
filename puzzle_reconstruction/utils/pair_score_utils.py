"""Utilities for aggregating and analysing fragment pair scoring results.

Provides dataclasses for storing per-pair scoring entries, summary
statistics, and functions for filtering, ranking, and batch analysis
of multi-channel pair match scores.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class PairScoreConfig:
    """Configuration for pair-score analysis utilities."""

    good_threshold: float = 0.7
    poor_threshold: float = 0.3
    dominant_channel: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0 <= self.good_threshold <= 1):
            raise ValueError("good_threshold must be in [0, 1]")
        if not (0 <= self.poor_threshold <= 1):
            raise ValueError("poor_threshold must be in [0, 1]")


@dataclass
class PairScoreEntry:
    """A scored fragment pair with per-channel breakdown."""

    frag_i: int
    frag_j: int
    score: float
    channels: Dict[str, float] = field(default_factory=dict)
    method: str = "pair_scorer"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Canonical ordered pair key."""
        return (min(self.frag_i, self.frag_j), max(self.frag_i, self.frag_j))

    @property
    def dominant_channel(self) -> Optional[str]:
        """Channel with the highest score, or None if no channels."""
        if not self.channels:
            return None
        return max(self.channels, key=lambda k: self.channels[k])

    @property
    def is_strong_match(self) -> bool:
        return self.score >= 0.7


@dataclass
class PairScoreSummary:
    """Aggregated statistics over a collection of pair score entries."""

    n_entries: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    n_strong_matches: int
    channel_means: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


# ── Factory functions ─────────────────────────────────────────────────────────

def make_pair_score_entry(
    frag_i: int,
    frag_j: int,
    score: float,
    *,
    channels: Optional[Dict[str, float]] = None,
    method: str = "pair_scorer",
    params: Optional[Dict[str, Any]] = None,
) -> PairScoreEntry:
    """Create a *PairScoreEntry* from raw values."""
    return PairScoreEntry(
        frag_i=frag_i,
        frag_j=frag_j,
        score=score,
        channels=channels or {},
        method=method,
        params=params or {},
    )


def entries_from_pair_results(
    pairs: Sequence[Tuple[int, int]],
    scores: Sequence[float],
    channel_lists: Optional[Sequence[Dict[str, float]]] = None,
    *,
    method: str = "pair_scorer",
) -> List[PairScoreEntry]:
    """Build a list of *PairScoreEntry* from parallel sequences."""
    if len(pairs) != len(scores):
        raise ValueError("pairs and scores must have the same length")
    chs = channel_lists or [{}] * len(pairs)
    return [
        make_pair_score_entry(i, j, s, channels=ch, method=method)
        for (i, j), s, ch in zip(pairs, scores, chs)
    ]


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarise_pair_scores(
    entries: Sequence[PairScoreEntry],
) -> PairScoreSummary:
    """Compute aggregate statistics over *entries*."""
    if not entries:
        return PairScoreSummary(
            n_entries=0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            n_strong_matches=0,
        )
    scores = [e.score for e in entries]
    # per-channel means
    channel_keys: set = set()
    for e in entries:
        channel_keys.update(e.channels.keys())
    channel_means: Dict[str, float] = {}
    for k in channel_keys:
        vals = [e.channels[k] for e in entries if k in e.channels]
        channel_means[k] = statistics.mean(vals) if vals else 0.0
    return PairScoreSummary(
        n_entries=len(entries),
        mean_score=statistics.mean(scores),
        std_score=statistics.pstdev(scores),
        min_score=min(scores),
        max_score=max(scores),
        n_strong_matches=sum(1 for e in entries if e.is_strong_match),
        channel_means=channel_means,
    )


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_strong_pair_matches(
    entries: Sequence[PairScoreEntry],
    threshold: float = 0.7,
) -> List[PairScoreEntry]:
    """Return entries whose score >= *threshold*."""
    return [e for e in entries if e.score >= threshold]


def filter_weak_pair_matches(
    entries: Sequence[PairScoreEntry],
    threshold: float = 0.3,
) -> List[PairScoreEntry]:
    """Return entries whose score < *threshold*."""
    return [e for e in entries if e.score < threshold]


def filter_pair_by_score_range(
    entries: Sequence[PairScoreEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[PairScoreEntry]:
    """Keep entries whose score is in [lo, hi]."""
    return [e for e in entries if lo <= e.score <= hi]


def filter_pair_by_channel(
    entries: Sequence[PairScoreEntry],
    channel: str,
    min_val: float = 0.0,
) -> List[PairScoreEntry]:
    """Keep entries where *channel* score >= *min_val*."""
    return [e for e in entries
            if channel in e.channels and e.channels[channel] >= min_val]


def filter_pair_by_dominant_channel(
    entries: Sequence[PairScoreEntry],
    channel: str,
) -> List[PairScoreEntry]:
    """Keep entries whose dominant channel is *channel*."""
    return [e for e in entries if e.dominant_channel == channel]


# ── Ranking ───────────────────────────────────────────────────────────────────

def top_k_pair_entries(
    entries: Sequence[PairScoreEntry],
    k: int = 10,
) -> List[PairScoreEntry]:
    """Return the *k* highest-scoring entries."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_pair_entry(
    entries: Sequence[PairScoreEntry],
) -> Optional[PairScoreEntry]:
    """Return the entry with the highest score, or None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.score)


# ── Statistics ────────────────────────────────────────────────────────────────

def pair_score_stats(
    entries: Sequence[PairScoreEntry],
) -> Dict[str, float]:
    """Return a dict with descriptive statistics of the scores."""
    if not entries:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    scores = [e.score for e in entries]
    return {
        "count": len(entries),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "min": min(scores),
        "max": max(scores),
        "n_strong": sum(1 for e in entries if e.is_strong_match),
    }


def compare_pair_summaries(
    a: PairScoreSummary,
    b: PairScoreSummary,
) -> Dict[str, float]:
    """Return deltas (a − b) for the main scalar fields."""
    return {
        "d_mean_score": a.mean_score - b.mean_score,
        "d_std_score": a.std_score - b.std_score,
        "d_min_score": a.min_score - b.min_score,
        "d_max_score": a.max_score - b.max_score,
        "d_n_strong": a.n_strong_matches - b.n_strong_matches,
        "d_n_entries": a.n_entries - b.n_entries,
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_summarise_pair_scores(
    groups: Sequence[Sequence[PairScoreEntry]],
) -> List[PairScoreSummary]:
    """Summarise multiple groups of entries in one call."""
    return [summarise_pair_scores(g) for g in groups]
