"""Utility records for global ranking and boundary-validation pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Global ranking records ───────────────────────────────────────────────────

@dataclass
class RankingRunRecord:
    """Summary of one global-ranking run over a set of fragments."""
    n_fragments: int
    n_pairs_ranked: int
    top_score: float = 0.0
    mean_score: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(f"n_fragments must be >= 0, got {self.n_fragments}")
        if not (0.0 <= self.top_score <= 1.0):
            raise ValueError(f"top_score must be in [0, 1], got {self.top_score}")

    @property
    def has_results(self) -> bool:
        return self.n_pairs_ranked > 0


@dataclass
class CandidateSummary:
    """Per-fragment candidate summary."""
    fragment_id: int
    n_candidates: int
    best_score: float = 0.0
    best_partner: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(f"fragment_id must be >= 0, got {self.fragment_id}")
        if not (0.0 <= self.best_score <= 1.0):
            raise ValueError(f"best_score must be in [0, 1], got {self.best_score}")

    @property
    def has_candidates(self) -> bool:
        return self.n_candidates > 0


@dataclass
class ScoreVectorRecord:
    """Stores a fragment score vector together with metadata."""
    n_fragments: int
    scores: List[float] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.scores) not in (0, self.n_fragments):
            raise ValueError("scores length must be 0 or n_fragments")

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


# ─── Boundary validation records ─────────────────────────────────────────────

@dataclass
class ValidationRunRecord:
    """Aggregates validation results for an assembly step."""
    step: int
    n_pairs: int
    n_violations: int
    quality_score: float
    label: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(
                f"quality_score must be in [0, 1], got {self.quality_score}"
            )

    @property
    def violation_rate(self) -> float:
        if self.n_pairs == 0:
            return 0.0
        return self.n_violations / self.n_pairs

    @property
    def is_clean(self) -> bool:
        return self.n_violations == 0


@dataclass
class BoundaryCheckSummary:
    """High-level summary of boundary checking across multiple assemblies."""
    n_assemblies: int
    mean_quality: float
    best_quality: float = 1.0
    worst_quality: float = 0.0
    violation_types: Dict[str, int] = field(default_factory=dict)

    @property
    def dominant_violation(self) -> Optional[str]:
        if not self.violation_types:
            return None
        return max(self.violation_types, key=lambda k: self.violation_types[k])


# ─── Color-palette records ────────────────────────────────────────────────────

@dataclass
class PaletteComparisonRecord:
    """Records the distance between two palettes."""
    fragment_id_a: int
    fragment_id_b: int
    distance: float
    n_colors: int

    def __post_init__(self) -> None:
        if self.distance < 0.0:
            raise ValueError(f"distance must be >= 0, got {self.distance}")

    @property
    def similarity(self) -> float:
        """Approximate similarity ∈ [0, 1] derived from distance."""
        return max(0.0, 1.0 - self.distance / 255.0)


@dataclass
class PaletteRankingRecord:
    """Records palette-based ranking results for a query fragment."""
    query_id: int
    ranked_ids: List[int] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)

    def top_k(self, k: int) -> List[Tuple[int, float]]:
        pairs = list(zip(self.ranked_ids, self.similarities))
        return pairs[:k]


# ─── Convenience constructors ─────────────────────────────────────────────────

def make_ranking_record(n_fragments: int,
                         n_pairs: int,
                         top_score: float = 0.0,
                         label: str = "") -> RankingRunRecord:
    return RankingRunRecord(n_fragments=n_fragments, n_pairs_ranked=n_pairs,
                             top_score=top_score, label=label)


def make_validation_record(step: int,
                             n_pairs: int,
                             n_violations: int,
                             quality_score: float) -> ValidationRunRecord:
    return ValidationRunRecord(step=step, n_pairs=n_pairs,
                                n_violations=n_violations,
                                quality_score=quality_score)
