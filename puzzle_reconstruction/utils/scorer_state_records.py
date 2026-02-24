"""Records and utilities for assembly scoring, state tracking, and binarization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AssemblyScoringRecord:
    """Record of a single assembly scoring run."""

    n_placed: int
    n_total: int
    total_score: float
    geometry_score: float
    coverage_score: float
    seam_score: float
    uniqueness_score: float

    @property
    def coverage_ratio(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_placed / self.n_total

    @property
    def is_complete(self) -> bool:
        return self.n_placed >= self.n_total


@dataclass
class StateTransitionRecord:
    """Record of a fragment placement transition."""

    fragment_idx: int
    from_step: int
    to_step: int
    position: tuple[float, float]
    angle: float = 0.0
    scale: float = 1.0

    @property
    def delta_step(self) -> int:
        return self.to_step - self.from_step


@dataclass
class AdjacencyRecord:
    """Record of an adjacency edge added between two fragments."""

    idx_a: int
    idx_b: int
    step: int

    @property
    def edge_key(self) -> tuple[int, int]:
        return (min(self.idx_a, self.idx_b), max(self.idx_a, self.idx_b))


@dataclass
class BeamSearchRecord:
    """Record of a beam search run."""

    n_fragments: int
    n_entries: int
    beam_width: int
    max_depth: Optional[int]
    n_placed: int
    final_score: float

    @property
    def placement_ratio(self) -> float:
        if self.n_fragments == 0:
            return 0.0
        return self.n_placed / self.n_fragments


@dataclass
class BinarizationRecord:
    """Record of an image binarization operation."""

    method: str
    threshold: float
    foreground_ratio: float
    inverted: bool = False
    image_height: int = 0
    image_width: int = 0

    @property
    def background_ratio(self) -> float:
        return 1.0 - self.foreground_ratio

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_height, self.image_width)


def make_assembly_scoring_record(
    n_placed: int,
    n_total: int,
    total_score: float,
    geometry_score: float = 0.0,
    coverage_score: float = 0.0,
    seam_score: float = 0.0,
    uniqueness_score: float = 0.0,
) -> AssemblyScoringRecord:
    """Create an AssemblyScoringRecord."""
    return AssemblyScoringRecord(
        n_placed=n_placed,
        n_total=n_total,
        total_score=total_score,
        geometry_score=geometry_score,
        coverage_score=coverage_score,
        seam_score=seam_score,
        uniqueness_score=uniqueness_score,
    )


def make_binarization_record(
    method: str,
    threshold: float,
    foreground_ratio: float,
    inverted: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> BinarizationRecord:
    """Create a BinarizationRecord from method and image metadata."""
    h, w = image_shape
    return BinarizationRecord(
        method=method,
        threshold=threshold,
        foreground_ratio=foreground_ratio,
        inverted=inverted,
        image_height=h,
        image_width=w,
    )
