"""Утилиты нормализации и калибровки оценок совместимости.

Provides lightweight dataclasses and helper functions for tracking
score normalization results: configuration, entries, summaries,
filtering, comparison, and batch operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ScoreNormConfig:
    """Configuration for score normalization."""
    method: str = "minmax"
    clip: bool = True
    feature_range: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        valid = {"minmax", "zscore", "rank", "calibrated"}
        if self.method not in valid:
            raise ValueError(
                f"method must be one of {valid}, got {self.method!r}")
        lo, hi = self.feature_range
        if lo >= hi:
            raise ValueError(
                f"feature_range low must be < high, got ({lo}, {hi})")


@dataclass
class ScoreNormEntry:
    """A single normalized-score entry."""
    idx: int
    original_score: float
    normalized_score: float
    method: str = "minmax"
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx < 0:
            raise ValueError(f"idx must be >= 0, got {self.idx}")

    @property
    def delta(self) -> float:
        """Difference between normalized and original score."""
        return self.normalized_score - self.original_score


@dataclass
class ScoreNormSummary:
    """Summary of a batch of score-normalisation entries."""
    entries: List[ScoreNormEntry]
    n_total: int
    method: str
    original_min: float
    original_max: float
    normalized_min: float
    normalized_max: float

    def __repr__(self) -> str:
        return (
            f"ScoreNormSummary(n={self.n_total}, method={self.method!r}, "
            f"orig=[{self.original_min:.3f}, {self.original_max:.3f}], "
            f"norm=[{self.normalized_min:.3f}, {self.normalized_max:.3f}])"
        )


def make_norm_entry(
    idx: int,
    original_score: float,
    normalized_score: float,
    method: str = "minmax",
    meta: Optional[Dict] = None,
) -> ScoreNormEntry:
    """Create a single score normalisation entry."""
    return ScoreNormEntry(
        idx=idx,
        original_score=original_score,
        normalized_score=normalized_score,
        method=method,
        meta=meta or {},
    )


def entries_from_scores(
    original: List[float],
    normalized: List[float],
    method: str = "minmax",
) -> List[ScoreNormEntry]:
    """Build entries from parallel original / normalised score lists."""
    if len(original) != len(normalized):
        raise ValueError("original and normalized must have same length")
    return [
        make_norm_entry(i, o, n, method=method)
        for i, (o, n) in enumerate(zip(original, normalized))
    ]


def summarise_norm(
    entries: List[ScoreNormEntry],
) -> ScoreNormSummary:
    """Compute a summary from a list of norm entries."""
    if not entries:
        return ScoreNormSummary(
            entries=entries, n_total=0, method="",
            original_min=0.0, original_max=0.0,
            normalized_min=0.0, normalized_max=0.0,
        )
    orig = [e.original_score for e in entries]
    norm = [e.normalized_score for e in entries]
    return ScoreNormSummary(
        entries=entries,
        n_total=len(entries),
        method=entries[0].method,
        original_min=min(orig),
        original_max=max(orig),
        normalized_min=min(norm),
        normalized_max=max(norm),
    )


def filter_by_normalized_range(
    entries: List[ScoreNormEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[ScoreNormEntry]:
    """Keep entries with normalized_score in [lo, hi]."""
    return [e for e in entries if lo <= e.normalized_score <= hi]


def filter_by_original_range(
    entries: List[ScoreNormEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[ScoreNormEntry]:
    """Keep entries with original_score in [lo, hi]."""
    return [e for e in entries if lo <= e.original_score <= hi]


def top_k_norm_entries(
    entries: List[ScoreNormEntry],
    k: int = 10,
) -> List[ScoreNormEntry]:
    """Return top-k entries by normalised score (descending)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranked = sorted(entries, key=lambda e: e.normalized_score, reverse=True)
    return ranked[:k]


def norm_entry_stats(entries: List[ScoreNormEntry]) -> Dict:
    """Compute basic statistics over norm entries."""
    if not entries:
        return {
            "n": 0,
            "mean_original": 0.0,
            "mean_normalized": 0.0,
            "mean_delta": 0.0,
        }
    n = len(entries)
    return {
        "n": n,
        "mean_original": sum(e.original_score for e in entries) / n,
        "mean_normalized": sum(e.normalized_score for e in entries) / n,
        "mean_delta": sum(e.delta for e in entries) / n,
    }


def compare_norm_summaries(
    summary_a: ScoreNormSummary,
    summary_b: ScoreNormSummary,
) -> Dict:
    """Compare two score-normalisation summaries."""
    return {
        "n_total_delta": summary_a.n_total - summary_b.n_total,
        "original_min_delta": summary_a.original_min - summary_b.original_min,
        "original_max_delta": summary_a.original_max - summary_b.original_max,
        "normalized_min_delta": (summary_a.normalized_min
                                  - summary_b.normalized_min),
        "normalized_max_delta": (summary_a.normalized_max
                                  - summary_b.normalized_max),
    }


def batch_summarise_norm(
    score_lists: List[Tuple[List[float], List[float]]],
    method: str = "minmax",
) -> List[ScoreNormSummary]:
    """Summarise multiple normalisation runs at once."""
    return [
        summarise_norm(entries_from_scores(orig, norm, method=method))
        for orig, norm in score_lists
    ]
