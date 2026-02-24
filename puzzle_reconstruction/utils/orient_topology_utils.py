"""Utilities for combining orientation matching and topology analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class OrientMatchRecord:
    """Summary record for a single orientation match pair."""
    fragment_a: int
    fragment_b: int
    best_score: float
    best_angle: float
    is_flipped: bool = False
    n_angles_tested: int = 0

    def __post_init__(self) -> None:
        if self.fragment_a < 0:
            raise ValueError("fragment_a must be >= 0")
        if self.fragment_b < 0:
            raise ValueError("fragment_b must be >= 0")
        if not (0.0 <= self.best_score <= 1.0):
            raise ValueError("best_score must be in [0, 1]")
        if self.best_angle < 0.0:
            raise ValueError("best_angle must be >= 0")

    @property
    def pair(self):
        return (self.fragment_a, self.fragment_b)

    @property
    def is_good_match(self) -> bool:
        return self.best_score >= 0.5


@dataclass
class OrientMatchSummary:
    """Aggregate summary over a collection of orientation match records."""
    total_pairs: int = 0
    good_pairs: int = 0
    flipped_pairs: int = 0
    mean_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 1.0

    def __post_init__(self) -> None:
        if self.total_pairs < 0:
            raise ValueError("total_pairs must be >= 0")
        if self.good_pairs < 0:
            raise ValueError("good_pairs must be >= 0")
        if self.flipped_pairs < 0:
            raise ValueError("flipped_pairs must be >= 0")

    @property
    def good_ratio(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.good_pairs / self.total_pairs

    @property
    def flip_ratio(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.flipped_pairs / self.total_pairs


@dataclass
class TopologyRecord:
    """Topology metrics for a single contour."""
    solidity: float = 0.0
    extent: float = 0.0
    convexity: float = 0.0
    compactness: float = 0.0
    complexity: float = 0.0

    def __post_init__(self) -> None:
        for name, val in [
            ("solidity", self.solidity),
            ("extent", self.extent),
            ("convexity", self.convexity),
            ("compactness", self.compactness),
            ("complexity", self.complexity),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def is_convex(self) -> bool:
        return self.convexity > 0.9

    @property
    def is_compact(self) -> bool:
        return self.compactness > 0.9

    def to_dict(self) -> Dict[str, float]:
        return {
            "solidity": self.solidity,
            "extent": self.extent,
            "convexity": self.convexity,
            "compactness": self.compactness,
            "complexity": self.complexity,
        }


@dataclass
class TopologySummary:
    """Aggregate statistics over a batch of TopologyRecord objects."""
    n_contours: int = 0
    mean_solidity: float = 0.0
    mean_compactness: float = 0.0
    mean_complexity: float = 0.0
    n_convex: int = 0
    n_compact: int = 0

    def __post_init__(self) -> None:
        if self.n_contours < 0:
            raise ValueError("n_contours must be >= 0")
        if self.n_convex < 0:
            raise ValueError("n_convex must be >= 0")
        if self.n_compact < 0:
            raise ValueError("n_compact must be >= 0")

    @property
    def convex_ratio(self) -> float:
        if self.n_contours == 0:
            return 0.0
        return self.n_convex / self.n_contours

    @property
    def compact_ratio(self) -> float:
        if self.n_contours == 0:
            return 0.0
        return self.n_compact / self.n_contours


def summarize_orient_matches(
    records: Sequence[OrientMatchRecord],
) -> OrientMatchSummary:
    """Aggregate a list of OrientMatchRecord objects into a summary."""
    if not records:
        return OrientMatchSummary()

    total = len(records)
    good = sum(1 for r in records if r.is_good_match)
    flipped = sum(1 for r in records if r.is_flipped)
    scores = [r.best_score for r in records]
    mean_s = float(sum(scores) / total)
    max_s = float(max(scores))
    min_s = float(min(scores))

    return OrientMatchSummary(
        total_pairs=total,
        good_pairs=good,
        flipped_pairs=flipped,
        mean_score=mean_s,
        max_score=max_s,
        min_score=min_s,
    )


def filter_orient_records(
    records: Sequence[OrientMatchRecord],
    min_score: float = 0.0,
    exclude_flipped: bool = False,
) -> List[OrientMatchRecord]:
    """Filter orientation match records by score and flip status."""
    result = [r for r in records if r.best_score >= min_score]
    if exclude_flipped:
        result = [r for r in result if not r.is_flipped]
    return result


def topology_records_from_dicts(
    dicts: Sequence[Dict[str, float]],
) -> List[TopologyRecord]:
    """Convert a list of topology dicts (from batch_topology) to TopologyRecord objects."""
    result = []
    for d in dicts:
        result.append(TopologyRecord(
            solidity=float(d.get("solidity", 0.0)),
            extent=float(d.get("extent", 0.0)),
            convexity=float(d.get("convexity", 0.0)),
            compactness=float(d.get("compactness", 0.0)),
            complexity=float(d.get("complexity", 0.0)),
        ))
    return result


def summarize_topology(
    records: Sequence[TopologyRecord],
) -> TopologySummary:
    """Aggregate a list of TopologyRecord objects into a summary."""
    if not records:
        return TopologySummary()

    n = len(records)
    mean_sol = float(sum(r.solidity for r in records) / n)
    mean_com = float(sum(r.compactness for r in records) / n)
    mean_cpx = float(sum(r.complexity for r in records) / n)
    n_convex = sum(1 for r in records if r.is_convex)
    n_compact = sum(1 for r in records if r.is_compact)

    return TopologySummary(
        n_contours=n,
        mean_solidity=mean_sol,
        mean_compactness=mean_com,
        mean_complexity=mean_cpx,
        n_convex=n_convex,
        n_compact=n_compact,
    )


def rank_orient_matches(
    records: Sequence[OrientMatchRecord],
) -> List[OrientMatchRecord]:
    """Return orient match records sorted by best_score descending."""
    return sorted(records, key=lambda r: r.best_score, reverse=True)


def top_k_orient_matches(
    records: Sequence[OrientMatchRecord],
    k: int,
) -> List[OrientMatchRecord]:
    """Return the top-k orient match records by best_score."""
    if k < 0:
        raise ValueError("k must be >= 0")
    ranked = rank_orient_matches(records)
    return ranked[:k]
