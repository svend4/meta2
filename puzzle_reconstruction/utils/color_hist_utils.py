"""Utilities for aggregating and analysing color-histogram comparison results.

Provides dataclasses for storing per-pair histogram similarity entries,
summary statistics, and functions for filtering, ranking, and batch
analysis of histogram-based fragment matching scores.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ColorHistConfig:
    """Configuration for color-histogram analysis utilities."""

    min_score: float = 0.0
    max_score: float = 1.0
    good_threshold: float = 0.7
    poor_threshold: float = 0.3
    space: str = "hsv"

    def __post_init__(self) -> None:
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")
        if self.max_score < self.min_score:
            raise ValueError("max_score must be >= min_score")
        if not (0 <= self.good_threshold <= 1):
            raise ValueError("good_threshold must be in [0, 1]")
        if not (0 <= self.poor_threshold <= 1):
            raise ValueError("poor_threshold must be in [0, 1]")


@dataclass
class ColorHistEntry:
    """A single color-histogram comparison result between two fragments."""

    frag_i: int
    frag_j: int
    intersection: float
    chi2: float
    space: str = "hsv"
    n_bins: int = 32
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Combined score: average of intersection and chi2."""
        return (self.intersection + self.chi2) / 2.0


@dataclass
class ColorHistSummary:
    """Aggregated statistics over a collection of histogram entries."""

    n_entries: int
    mean_intersection: float
    mean_chi2: float
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    space: str = "hsv"
    params: Dict[str, Any] = field(default_factory=dict)


# ── Factory functions ────────────────────────────────────────────────────────

def make_color_hist_entry(
    frag_i: int,
    frag_j: int,
    intersection: float,
    chi2: float,
    *,
    space: str = "hsv",
    n_bins: int = 32,
    params: Optional[Dict[str, Any]] = None,
) -> ColorHistEntry:
    """Create a *ColorHistEntry* from raw comparison values."""
    return ColorHistEntry(
        frag_i=frag_i,
        frag_j=frag_j,
        intersection=intersection,
        chi2=chi2,
        space=space,
        n_bins=n_bins,
        params=params or {},
    )


def entries_from_comparisons(
    pairs: Sequence[Tuple[int, int]],
    intersections: Sequence[float],
    chi2s: Sequence[float],
    *,
    space: str = "hsv",
    n_bins: int = 32,
) -> List[ColorHistEntry]:
    """Build a list of *ColorHistEntry* from parallel sequences."""
    if not (len(pairs) == len(intersections) == len(chi2s)):
        raise ValueError("All sequences must have the same length")
    return [
        make_color_hist_entry(i, j, inter, c2, space=space, n_bins=n_bins)
        for (i, j), inter, c2 in zip(pairs, intersections, chi2s)
    ]


# ── Summarisation ────────────────────────────────────────────────────────────

def summarise_color_hist(
    entries: Sequence[ColorHistEntry],
    *,
    space: str = "hsv",
) -> ColorHistSummary:
    """Compute aggregate statistics over *entries*."""
    if not entries:
        return ColorHistSummary(
            n_entries=0,
            mean_intersection=0.0,
            mean_chi2=0.0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            space=space,
        )
    scores = [e.score for e in entries]
    return ColorHistSummary(
        n_entries=len(entries),
        mean_intersection=statistics.mean(e.intersection for e in entries),
        mean_chi2=statistics.mean(e.chi2 for e in entries),
        mean_score=statistics.mean(scores),
        std_score=statistics.pstdev(scores),
        min_score=min(scores),
        max_score=max(scores),
        space=space,
    )


# ── Filtering ────────────────────────────────────────────────────────────────

def filter_good_hist_entries(
    entries: Sequence[ColorHistEntry],
    threshold: float = 0.7,
) -> List[ColorHistEntry]:
    """Return entries whose combined score >= *threshold*."""
    return [e for e in entries if e.score >= threshold]


def filter_poor_hist_entries(
    entries: Sequence[ColorHistEntry],
    threshold: float = 0.3,
) -> List[ColorHistEntry]:
    """Return entries whose combined score < *threshold*."""
    return [e for e in entries if e.score < threshold]


def filter_by_intersection_range(
    entries: Sequence[ColorHistEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[ColorHistEntry]:
    """Keep entries with intersection in [lo, hi]."""
    return [e for e in entries if lo <= e.intersection <= hi]


def filter_by_chi2_range(
    entries: Sequence[ColorHistEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[ColorHistEntry]:
    """Keep entries with chi2 in [lo, hi]."""
    return [e for e in entries if lo <= e.chi2 <= hi]


def filter_by_space(
    entries: Sequence[ColorHistEntry],
    space: str,
) -> List[ColorHistEntry]:
    """Keep entries computed in a specific color space."""
    return [e for e in entries if e.space == space]


# ── Ranking ──────────────────────────────────────────────────────────────────

def top_k_hist_entries(
    entries: Sequence[ColorHistEntry],
    k: int = 10,
) -> List[ColorHistEntry]:
    """Return the *k* entries with the highest combined score."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_hist_entry(
    entries: Sequence[ColorHistEntry],
) -> Optional[ColorHistEntry]:
    """Return the entry with the highest combined score, or *None*."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.score)


# ── Statistics ───────────────────────────────────────────────────────────────

def color_hist_stats(
    entries: Sequence[ColorHistEntry],
) -> Dict[str, float]:
    """Return a dict with basic descriptive statistics of the scores."""
    if not entries:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean_intersection": 0.0,
            "mean_chi2": 0.0,
        }
    scores = [e.score for e in entries]
    return {
        "count": len(entries),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "min": min(scores),
        "max": max(scores),
        "mean_intersection": statistics.mean(e.intersection for e in entries),
        "mean_chi2": statistics.mean(e.chi2 for e in entries),
    }


def compare_hist_summaries(
    a: ColorHistSummary,
    b: ColorHistSummary,
) -> Dict[str, float]:
    """Return deltas (a − b) for the main fields."""
    return {
        "d_mean_intersection": a.mean_intersection - b.mean_intersection,
        "d_mean_chi2": a.mean_chi2 - b.mean_chi2,
        "d_mean_score": a.mean_score - b.mean_score,
        "d_std_score": a.std_score - b.std_score,
        "d_n_entries": a.n_entries - b.n_entries,
    }


# ── Batch ────────────────────────────────────────────────────────────────────

def batch_summarise_color_hist(
    groups: Sequence[Sequence[ColorHistEntry]],
    *,
    space: str = "hsv",
) -> List[ColorHistSummary]:
    """Summarise multiple groups of entries in one call."""
    return [summarise_color_hist(g, space=space) for g in groups]
