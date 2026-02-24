"""Records and utilities for region scoring, seam evaluation, and array operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegionPairRecord:
    """Record of a scored region pair."""

    idx_a: int
    idx_b: int
    score: float
    color_score: float
    texture_score: float
    shape_score: float
    boundary_score: float

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.idx_a, self.idx_b), max(self.idx_a, self.idx_b))


@dataclass
class SeamRecord:
    """Record of a seam evaluation between two fragments."""

    fragment_a: int
    side_a: int
    fragment_b: int
    side_b: int
    score: float
    color_score: float
    gradient_score: float
    texture_score: float

    @property
    def side_pair(self) -> tuple[int, int]:
        return (self.side_a, self.side_b)


@dataclass
class ArrayChunkRecord:
    """Record of a chunked array operation."""

    total_elements: int
    chunk_size: int
    n_chunks: int
    last_chunk_size: int

    @property
    def has_remainder(self) -> bool:
        return self.last_chunk_size < self.chunk_size


@dataclass
class PairwiseNormRecord:
    """Record of a pairwise norm computation."""

    n_rows: int
    n_cols: int
    metric: str
    min_dist: float
    max_dist: float
    mean_dist: float


def make_region_pair_record(
    idx_a: int,
    idx_b: int,
    score: float,
    color_score: float = 0.0,
    texture_score: float = 0.0,
    shape_score: float = 0.0,
    boundary_score: float = 0.0,
) -> RegionPairRecord:
    """Create a RegionPairRecord."""
    return RegionPairRecord(
        idx_a=idx_a,
        idx_b=idx_b,
        score=score,
        color_score=color_score,
        texture_score=texture_score,
        shape_score=shape_score,
        boundary_score=boundary_score,
    )


def make_seam_record(
    fragment_a: int,
    side_a: int,
    fragment_b: int,
    side_b: int,
    score: float,
    color_score: float = 0.0,
    gradient_score: float = 0.0,
    texture_score: float = 0.0,
) -> SeamRecord:
    """Create a SeamRecord."""
    return SeamRecord(
        fragment_a=fragment_a,
        side_a=side_a,
        fragment_b=fragment_b,
        side_b=side_b,
        score=score,
        color_score=color_score,
        gradient_score=gradient_score,
        texture_score=texture_score,
    )


def make_array_chunk_record(
    total_elements: int,
    chunk_size: int,
) -> ArrayChunkRecord:
    """Create an ArrayChunkRecord from total elements and chunk size."""
    n_chunks = (total_elements + chunk_size - 1) // chunk_size
    last_chunk_size = total_elements - (n_chunks - 1) * chunk_size if n_chunks > 0 else 0
    return ArrayChunkRecord(
        total_elements=total_elements,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        last_chunk_size=last_chunk_size,
    )
