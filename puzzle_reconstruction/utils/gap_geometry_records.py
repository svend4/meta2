"""Records for gap scoring, geometric matching, geometry, and global matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GapScoringRecord:
    """Record of a gap scoring result between two fragments."""

    id_a: int
    id_b: int
    distance: float
    score: float
    penalty: float
    acceptable: bool = False

    @property
    def pair_key(self) -> tuple:
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))

    @property
    def is_acceptable(self) -> bool:
        return self.score > 0.5


@dataclass
class GeometricMatchRecord:
    """Record of a geometric match between two fragments."""

    idx1: int
    idx2: int
    score: float
    aspect_score: float
    area_score: float
    hu_score: float
    method: str = "geometric"

    @property
    def is_good_match(self) -> bool:
        return self.score > 0.7


@dataclass
class GeometryRecord:
    """Record of geometry features extracted from a fragment mask."""

    fragment_id: int
    area: float
    perimeter: float
    aspect_ratio: float
    solidity: float
    n_contours: int = 0

    @property
    def is_convex(self) -> bool:
        return self.solidity > 0.95

    @property
    def is_elongated(self) -> bool:
        return self.aspect_ratio > 2.0


@dataclass
class GlobalMatchRecord:
    """Record of a global match result for a fragment."""

    fragment_id: int
    candidate_id: int
    score: float
    rank: int
    n_channels: int = 0

    @property
    def is_top(self) -> bool:
        return self.rank == 1

    @property
    def is_strong(self) -> bool:
        return self.score > 0.8


def make_gap_scoring_record(
    id_a: int,
    id_b: int,
    distance: float,
    score: float,
    penalty: float,
) -> GapScoringRecord:
    """Create a GapScoringRecord."""
    return GapScoringRecord(
        id_a=id_a,
        id_b=id_b,
        distance=distance,
        score=score,
        penalty=penalty,
        acceptable=score > 0.5,
    )


def make_global_match_record(
    fragment_id: int,
    candidate_id: int,
    score: float,
    rank: int,
    n_channels: int = 0,
) -> GlobalMatchRecord:
    """Create a GlobalMatchRecord."""
    return GlobalMatchRecord(
        fragment_id=fragment_id,
        candidate_id=candidate_id,
        score=score,
        rank=rank,
        n_channels=n_channels,
    )
