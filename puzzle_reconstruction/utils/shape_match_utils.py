"""Утилиты анализа и ранжирования результатов сопоставления форм.

Provides lightweight dataclasses and helper functions for tracking
shape matching results: configuration, entries, summaries,
filtering, comparison, and batch operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ShapeMatchConfig:
    """Configuration for shape matching analysis."""
    min_score: float = 0.0
    max_pairs: int = 100
    method: str = "hu"

    def __post_init__(self) -> None:
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score must be >= 0.0, got {self.min_score}")
        if self.max_pairs < 1:
            raise ValueError(
                f"max_pairs must be >= 1, got {self.max_pairs}")
        valid_methods = {"hu", "zernike", "combined"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method!r}")


@dataclass
class ShapeMatchEntry:
    """A single shape matching result entry."""
    idx1: int
    idx2: int
    score: float
    hu_dist: float = 0.0
    iou: float = 0.0
    chamfer: float = 0.0
    rank: int = 0
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0 or self.idx2 < 0:
            raise ValueError(
                f"idx1 and idx2 must be >= 0, got ({self.idx1}, {self.idx2})")

    @property
    def is_good(self) -> bool:
        """Whether the match has score > 0.5."""
        return self.score > 0.5


@dataclass
class ShapeMatchSummary:
    """Summary of a batch of shape match entries."""
    entries: List[ShapeMatchEntry]
    n_total: int
    n_good: int
    n_poor: int
    mean_score: float
    max_score: float
    min_score: float

    def __repr__(self) -> str:
        return (
            f"ShapeMatchSummary(n={self.n_total}, good={self.n_good}, "
            f"poor={self.n_poor}, mean={self.mean_score:.3f}, "
            f"max={self.max_score:.3f}, min={self.min_score:.3f})"
        )


def make_match_entry(
    idx1: int,
    idx2: int,
    score: float,
    hu_dist: float = 0.0,
    iou: float = 0.0,
    chamfer: float = 0.0,
    rank: int = 0,
    meta: Optional[Dict] = None,
) -> ShapeMatchEntry:
    """Create a single shape match entry."""
    return ShapeMatchEntry(
        idx1=idx1, idx2=idx2, score=score,
        hu_dist=hu_dist, iou=iou, chamfer=chamfer,
        rank=rank, meta=meta or {},
    )


def entries_from_results(
    results: List[Tuple[int, int, float]],
) -> List[ShapeMatchEntry]:
    """Build entries from list of (idx1, idx2, score) tuples."""
    return [
        make_match_entry(idx1=r[0], idx2=r[1], score=r[2], rank=i)
        for i, r in enumerate(results)
    ]


def summarise_matches(
    entries: List[ShapeMatchEntry],
) -> ShapeMatchSummary:
    """Compute a summary over shape match entries."""
    if not entries:
        return ShapeMatchSummary(
            entries=entries, n_total=0, n_good=0, n_poor=0,
            mean_score=0.0, max_score=0.0, min_score=0.0,
        )
    scores = [e.score for e in entries]
    n_good = sum(1 for e in entries if e.is_good)
    return ShapeMatchSummary(
        entries=entries,
        n_total=len(entries),
        n_good=n_good,
        n_poor=len(entries) - n_good,
        mean_score=sum(scores) / len(scores),
        max_score=max(scores),
        min_score=min(scores),
    )


def filter_good_matches(
    entries: List[ShapeMatchEntry],
) -> List[ShapeMatchEntry]:
    """Keep only entries with score > 0.5."""
    return [e for e in entries if e.is_good]


def filter_poor_matches(
    entries: List[ShapeMatchEntry],
) -> List[ShapeMatchEntry]:
    """Keep entries with score <= 0.5."""
    return [e for e in entries if not e.is_good]


def filter_by_hu_dist(
    entries: List[ShapeMatchEntry],
    max_hu: float = 10.0,
) -> List[ShapeMatchEntry]:
    """Keep entries with hu_dist <= max_hu."""
    return [e for e in entries if e.hu_dist <= max_hu]


def filter_match_by_score_range(
    entries: List[ShapeMatchEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[ShapeMatchEntry]:
    """Keep entries with score in [lo, hi]."""
    return [e for e in entries if lo <= e.score <= hi]


def top_k_match_entries(
    entries: List[ShapeMatchEntry],
    k: int = 10,
) -> List[ShapeMatchEntry]:
    """Return top-k entries by score (descending)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranked = sorted(entries, key=lambda e: e.score, reverse=True)
    return ranked[:k]


def match_entry_stats(entries: List[ShapeMatchEntry]) -> Dict:
    """Compute basic statistics over match entries."""
    if not entries:
        return {
            "n": 0,
            "mean_score": 0.0,
            "mean_hu_dist": 0.0,
            "mean_iou": 0.0,
            "mean_chamfer": 0.0,
        }
    n = len(entries)
    return {
        "n": n,
        "mean_score": sum(e.score for e in entries) / n,
        "mean_hu_dist": sum(e.hu_dist for e in entries) / n,
        "mean_iou": sum(e.iou for e in entries) / n,
        "mean_chamfer": sum(e.chamfer for e in entries) / n,
    }


def compare_match_summaries(
    summary_a: ShapeMatchSummary,
    summary_b: ShapeMatchSummary,
) -> Dict:
    """Compare two shape match summaries."""
    return {
        "n_total_delta": summary_a.n_total - summary_b.n_total,
        "n_good_delta": summary_a.n_good - summary_b.n_good,
        "mean_score_delta": summary_a.mean_score - summary_b.mean_score,
        "max_score_delta": summary_a.max_score - summary_b.max_score,
        "min_score_delta": summary_a.min_score - summary_b.min_score,
    }


def batch_summarise_matches(
    entry_lists: List[List[ShapeMatchEntry]],
) -> List[ShapeMatchSummary]:
    """Summarise multiple match entry lists."""
    return [summarise_matches(entries) for entries in entry_lists]
