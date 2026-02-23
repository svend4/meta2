"""Утилиты оценки качества консенсуса (consensus scoring).

Provides lightweight dataclasses and helper functions for analysing
consensus results from multiple assembly methods: pair agreement,
vote distributions, stability metrics, and batch summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class ConsensusScoreConfig:
    """Configuration for consensus score analysis."""
    min_vote_fraction: float = 0.5
    min_pairs: int = 1
    weight_by_score: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_vote_fraction <= 1.0:
            raise ValueError(
                f"min_vote_fraction must be in [0, 1], got {self.min_vote_fraction}"
            )
        if self.min_pairs < 1:
            raise ValueError(
                f"min_pairs must be >= 1, got {self.min_pairs}"
            )


@dataclass
class ConsensusScoreEntry:
    """Score record for a single fragment pair in consensus."""
    pair: Tuple[int, int]
    vote_count: int
    n_methods: int
    is_consensus: bool
    meta: Dict = field(default_factory=dict)

    @property
    def vote_fraction(self) -> float:
        if self.n_methods == 0:
            return 0.0
        return self.vote_count / self.n_methods

    def __repr__(self) -> str:
        return (
            f"ConsensusScoreEntry(pair={self.pair}, "
            f"votes={self.vote_count}/{self.n_methods}, "
            f"frac={self.vote_fraction:.3f}, "
            f"consensus={self.is_consensus})"
        )


@dataclass
class ConsensusSummary:
    """Summary of a consensus scoring run."""
    entries: List[ConsensusScoreEntry]
    n_pairs: int
    n_consensus: int
    n_methods: int
    mean_vote_fraction: float
    agreement_score: float

    def __repr__(self) -> str:
        return (
            f"ConsensusSummary(n_pairs={self.n_pairs}, "
            f"n_consensus={self.n_consensus}, "
            f"agreement={self.agreement_score:.3f})"
        )


def make_consensus_entry(
    pair: Tuple[int, int],
    vote_count: int,
    n_methods: int,
    threshold: float = 0.5,
    meta: Optional[Dict] = None,
) -> ConsensusScoreEntry:
    """Create a single consensus score entry."""
    is_consensus = (vote_count / n_methods >= threshold) if n_methods > 0 else False
    return ConsensusScoreEntry(
        pair=pair,
        vote_count=vote_count,
        n_methods=n_methods,
        is_consensus=is_consensus,
        meta=meta or {},
    )


def entries_from_votes(
    pair_votes: Dict[FrozenSet[int], int],
    n_methods: int,
    threshold: float = 0.5,
) -> List[ConsensusScoreEntry]:
    """Convert a pair_votes dict to a list of ConsensusScoreEntry."""
    result = []
    for pair_set, count in pair_votes.items():
        ids = sorted(pair_set)
        pair = (ids[0], ids[1]) if len(ids) >= 2 else (ids[0], ids[0])
        entry = make_consensus_entry(pair, count, n_methods, threshold)
        result.append(entry)
    return result


def summarise_consensus(
    entries: List[ConsensusScoreEntry],
    cfg: Optional[ConsensusScoreConfig] = None,
) -> ConsensusSummary:
    """Compute a summary from a list of consensus entries."""
    cfg = cfg or ConsensusScoreConfig()
    if not entries:
        return ConsensusSummary(
            entries=entries, n_pairs=0, n_consensus=0,
            n_methods=0, mean_vote_fraction=0.0, agreement_score=0.0,
        )
    n = len(entries)
    n_consensus = sum(1 for e in entries if e.is_consensus)
    n_methods = entries[0].n_methods if entries else 0
    fracs = [e.vote_fraction for e in entries]
    mean_frac = sum(fracs) / n
    agreement = n_consensus / n if n > 0 else 0.0
    return ConsensusSummary(
        entries=entries,
        n_pairs=n,
        n_consensus=n_consensus,
        n_methods=n_methods,
        mean_vote_fraction=mean_frac,
        agreement_score=agreement,
    )


def filter_consensus_pairs(
    entries: List[ConsensusScoreEntry],
) -> List[ConsensusScoreEntry]:
    """Return only entries marked as consensus."""
    return [e for e in entries if e.is_consensus]


def filter_non_consensus(
    entries: List[ConsensusScoreEntry],
) -> List[ConsensusScoreEntry]:
    """Return only entries NOT marked as consensus."""
    return [e for e in entries if not e.is_consensus]


def filter_by_vote_fraction(
    entries: List[ConsensusScoreEntry],
    min_fraction: float = 0.5,
) -> List[ConsensusScoreEntry]:
    """Keep entries where vote_fraction >= min_fraction."""
    return [e for e in entries if e.vote_fraction >= min_fraction]


def top_k_consensus_entries(
    entries: List[ConsensusScoreEntry],
    k: int,
) -> List[ConsensusScoreEntry]:
    """Return top-k entries by vote_fraction (descending)."""
    sorted_entries = sorted(entries, key=lambda e: e.vote_fraction, reverse=True)
    return sorted_entries[:max(0, k)]


def consensus_score_stats(entries: List[ConsensusScoreEntry]) -> Dict:
    """Compute basic statistics over vote fractions."""
    if not entries:
        return {
            "count": 0, "mean_fraction": 0.0, "std_fraction": 0.0,
            "min_fraction": 0.0, "max_fraction": 0.0,
            "n_consensus": 0, "n_non_consensus": 0,
        }
    fracs = [e.vote_fraction for e in entries]
    n = len(fracs)
    mean_f = sum(fracs) / n
    var = sum((f - mean_f) ** 2 for f in fracs) / n
    std_f = var ** 0.5
    n_cons = sum(1 for e in entries if e.is_consensus)
    return {
        "count": n,
        "mean_fraction": mean_f,
        "std_fraction": std_f,
        "min_fraction": min(fracs),
        "max_fraction": max(fracs),
        "n_consensus": n_cons,
        "n_non_consensus": n - n_cons,
    }


def agreement_score(entries: List[ConsensusScoreEntry]) -> float:
    """Fraction of pairs that reached consensus."""
    if not entries:
        return 0.0
    return sum(1 for e in entries if e.is_consensus) / len(entries)


def compare_consensus(
    summary_a: ConsensusSummary,
    summary_b: ConsensusSummary,
) -> Dict:
    """Compare two consensus summaries."""
    return {
        "n_pairs_delta": summary_a.n_pairs - summary_b.n_pairs,
        "n_consensus_delta": summary_a.n_consensus - summary_b.n_consensus,
        "agreement_delta": summary_a.agreement_score - summary_b.agreement_score,
        "mean_fraction_delta": (summary_a.mean_vote_fraction
                                - summary_b.mean_vote_fraction),
    }


def batch_summarise_consensus(
    vote_dicts: List[Dict[FrozenSet[int], int]],
    n_methods_list: List[int],
    threshold: float = 0.5,
    cfg: Optional[ConsensusScoreConfig] = None,
) -> List[ConsensusSummary]:
    """Summarise multiple consensus results at once."""
    results = []
    for votes, nm in zip(vote_dicts, n_methods_list):
        entries = entries_from_votes(votes, nm, threshold)
        results.append(summarise_consensus(entries, cfg))
    return results
