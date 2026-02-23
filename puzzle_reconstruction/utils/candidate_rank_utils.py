"""Утилиты ранжирования и фильтрации кандидатных пар.

Provides lightweight dataclasses and helper functions for analysing
candidate pair rankings: scoring, filtering, deduplication, top-k
selection, and batch summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CandidateRankConfig:
    """Configuration for candidate ranking analysis."""
    min_score: float = 0.5
    max_pairs: int = 0
    deduplicate: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be in [0, 1], got {self.min_score}")
        if self.max_pairs < 0:
            raise ValueError(
                f"max_pairs must be >= 0, got {self.max_pairs}")


@dataclass
class CandidateRankEntry:
    """Record for a single candidate pair ranking result."""
    idx1: int
    idx2: int
    score: float
    rank: int
    is_selected: bool
    meta: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CandidateRankEntry(pair=({self.idx1},{self.idx2}), "
            f"score={self.score:.3f}, rank={self.rank}, "
            f"selected={self.is_selected})"
        )


@dataclass
class CandidateRankSummary:
    """Summary of candidate pair ranking across a batch."""
    entries: List[CandidateRankEntry]
    n_total: int
    n_selected: int
    n_rejected: int
    mean_score: float
    max_score: float
    min_score: float

    def __repr__(self) -> str:
        return (
            f"CandidateRankSummary(n={self.n_total}, "
            f"selected={self.n_selected}, "
            f"mean_score={self.mean_score:.3f})"
        )


def make_candidate_entry(
    idx1: int,
    idx2: int,
    score: float,
    rank: int,
    cfg: Optional[CandidateRankConfig] = None,
    meta: Optional[Dict] = None,
) -> CandidateRankEntry:
    """Create a single candidate rank entry."""
    cfg = cfg or CandidateRankConfig()
    is_selected = score >= cfg.min_score
    return CandidateRankEntry(
        idx1=idx1,
        idx2=idx2,
        score=float(score),
        rank=rank,
        is_selected=is_selected,
        meta=meta or {},
    )


def entries_from_pairs(
    pairs: List[Dict],
    cfg: Optional[CandidateRankConfig] = None,
) -> List[CandidateRankEntry]:
    """Convert a list of pair dicts to CandidateRankEntry list.

    Expected keys: ``idx1``, ``idx2``, ``score``.
    Rank is assigned by descending score order.
    """
    sorted_pairs = sorted(pairs, key=lambda p: float(p.get("score", 0.0)),
                          reverse=True)
    result = []
    for rank, pair in enumerate(sorted_pairs):
        entry = make_candidate_entry(
            idx1=int(pair.get("idx1", 0)),
            idx2=int(pair.get("idx2", 0)),
            score=float(pair.get("score", 0.0)),
            rank=rank,
            cfg=cfg,
            meta={k: v for k, v in pair.items()
                  if k not in ("idx1", "idx2", "score")},
        )
        result.append(entry)
    return result


def summarise_rankings(
    entries: List[CandidateRankEntry],
) -> CandidateRankSummary:
    """Compute a summary from a list of candidate rank entries."""
    if not entries:
        return CandidateRankSummary(
            entries=entries, n_total=0, n_selected=0, n_rejected=0,
            mean_score=0.0, max_score=0.0, min_score=0.0,
        )
    n = len(entries)
    scores = [e.score for e in entries]
    n_sel = sum(1 for e in entries if e.is_selected)
    return CandidateRankSummary(
        entries=entries,
        n_total=n,
        n_selected=n_sel,
        n_rejected=n - n_sel,
        mean_score=sum(scores) / n,
        max_score=max(scores),
        min_score=min(scores),
    )


def filter_selected(
    entries: List[CandidateRankEntry],
) -> List[CandidateRankEntry]:
    """Return only selected entries."""
    return [e for e in entries if e.is_selected]


def filter_rejected_candidates(
    entries: List[CandidateRankEntry],
) -> List[CandidateRankEntry]:
    """Return only rejected entries."""
    return [e for e in entries if not e.is_selected]


def filter_by_score_range(
    entries: List[CandidateRankEntry],
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> List[CandidateRankEntry]:
    """Keep entries where min_score <= score <= max_score."""
    return [e for e in entries if min_score <= e.score <= max_score]


def filter_by_rank(
    entries: List[CandidateRankEntry],
    max_rank: int,
) -> List[CandidateRankEntry]:
    """Keep entries where rank <= max_rank."""
    return [e for e in entries if e.rank <= max_rank]


def top_k_candidate_entries(
    entries: List[CandidateRankEntry],
    k: int,
) -> List[CandidateRankEntry]:
    """Return top-k entries by score (descending)."""
    sorted_entries = sorted(entries, key=lambda e: e.score, reverse=True)
    return sorted_entries[:max(0, k)]


def candidate_rank_stats(entries: List[CandidateRankEntry]) -> Dict:
    """Compute basic statistics over candidate scores."""
    if not entries:
        return {
            "count": 0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0,
            "n_selected": 0, "n_rejected": 0,
        }
    scores = [e.score for e in entries]
    n = len(scores)
    mean_s = sum(scores) / n
    var = sum((s - mean_s) ** 2 for s in scores) / n
    std_s = var ** 0.5
    n_sel = sum(1 for e in entries if e.is_selected)
    return {
        "count": n,
        "mean": mean_s,
        "std": std_s,
        "min": min(scores),
        "max": max(scores),
        "n_selected": n_sel,
        "n_rejected": n - n_sel,
    }


def compare_rankings(
    summary_a: CandidateRankSummary,
    summary_b: CandidateRankSummary,
) -> Dict:
    """Compare two candidate ranking summaries."""
    return {
        "n_total_delta": summary_a.n_total - summary_b.n_total,
        "n_selected_delta": summary_a.n_selected - summary_b.n_selected,
        "mean_score_delta": summary_a.mean_score - summary_b.mean_score,
        "max_score_delta": summary_a.max_score - summary_b.max_score,
    }


def batch_summarise_rankings(
    pair_lists: List[List[Dict]],
    cfg: Optional[CandidateRankConfig] = None,
) -> List[CandidateRankSummary]:
    """Summarise multiple candidate pair batches at once."""
    return [
        summarise_rankings(entries_from_pairs(pairs, cfg))
        for pairs in pair_lists
    ]
