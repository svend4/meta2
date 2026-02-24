"""Records and utilities for assembly collision, cost matrix, scoring, and overlap resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CollisionRecord:
    """Record of a collision detection result."""

    id1: int
    id2: int
    overlap_w: int
    overlap_h: int
    overlap_area: int

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.id1, self.id2), max(self.id1, self.id2))


@dataclass
class CostMatrixRecord:
    """Record of a cost matrix build result."""

    n_fragments: int
    method: str
    min_cost: float
    max_cost: float
    mean_cost: float
    n_forbidden: int = 0


@dataclass
class FragmentScoreRecord:
    """Record of a fragment scoring result."""

    fragment_idx: int
    local_score: float
    n_neighbors: int
    is_reliable: bool


@dataclass
class AssemblyScoreRecord:
    """Record of an assembly scoring result."""

    global_score: float
    coverage: float
    mean_local: float
    n_placed: int
    n_reliable: int
    fragment_scores: dict[int, FragmentScoreRecord] = field(default_factory=dict)


@dataclass
class OverlapRecord:
    """Record of an overlap between two bounding boxes."""

    id_a: int
    id_b: int
    area: float
    dx: float
    dy: float

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))

    @property
    def has_overlap(self) -> bool:
        return self.area > 0.0


@dataclass
class ResolveRecord:
    """Record of an overlap resolution result."""

    n_iter: int
    resolved: bool
    final_n_overlaps: int
    n_fragments: int


def make_collision_record(
    id1: int,
    id2: int,
    overlap_w: int = 0,
    overlap_h: int = 0,
    overlap_area: int = 0,
) -> CollisionRecord:
    """Create a CollisionRecord."""
    return CollisionRecord(
        id1=id1, id2=id2,
        overlap_w=overlap_w, overlap_h=overlap_h,
        overlap_area=overlap_area,
    )


def make_cost_matrix_record(
    n_fragments: int,
    method: str,
    min_cost: float = 0.0,
    max_cost: float = 1.0,
    mean_cost: float = 0.5,
    n_forbidden: int = 0,
) -> CostMatrixRecord:
    """Create a CostMatrixRecord."""
    return CostMatrixRecord(
        n_fragments=n_fragments, method=method,
        min_cost=min_cost, max_cost=max_cost,
        mean_cost=mean_cost, n_forbidden=n_forbidden,
    )


def make_overlap_record(
    id_a: int,
    id_b: int,
    area: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
) -> OverlapRecord:
    """Create an OverlapRecord."""
    return OverlapRecord(
        id_a=id_a, id_b=id_b,
        area=area, dx=dx, dy=dy,
    )
