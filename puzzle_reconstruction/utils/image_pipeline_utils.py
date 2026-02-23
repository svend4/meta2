"""Utility types and helpers for image processing pipeline result aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class FrequencyMatchRecord:
    """Stores frequency-domain comparison result for a pair of images."""
    id_a: int
    id_b: int
    similarity: float
    method: str = "radial"

    def __post_init__(self) -> None:
        if self.id_a < 0:
            raise ValueError("id_a must be >= 0")
        if self.id_b < 0:
            raise ValueError("id_b must be >= 0")
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError("similarity must be in [0, 1]")

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.id_a, self.id_b)

    @property
    def is_similar(self) -> bool:
        return self.similarity >= 0.5


@dataclass
class FrequencyMatchSummary:
    """Aggregate summary over FrequencyMatchRecord objects."""
    total_pairs: int = 0
    similar_pairs: int = 0
    mean_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 1.0

    def __post_init__(self) -> None:
        if self.total_pairs < 0:
            raise ValueError("total_pairs must be >= 0")
        if self.similar_pairs < 0:
            raise ValueError("similar_pairs must be >= 0")

    @property
    def similar_ratio(self) -> float:
        if self.total_pairs == 0:
            return 0.0
        return self.similar_pairs / self.total_pairs


@dataclass
class PatchMatchRecord:
    """Compact record of a single patch match result."""
    src_row: int
    src_col: int
    dst_row: int
    dst_col: int
    score: float
    method: str = "ncc"

    def __post_init__(self) -> None:
        if self.src_row < 0:
            raise ValueError("src_row must be >= 0")
        if self.src_col < 0:
            raise ValueError("src_col must be >= 0")
        if self.dst_row < 0:
            raise ValueError("dst_row must be >= 0")
        if self.dst_col < 0:
            raise ValueError("dst_col must be >= 0")

    @property
    def displacement(self) -> Tuple[int, int]:
        return (self.dst_row - self.src_row, self.dst_col - self.src_col)


@dataclass
class PatchMatchSummary:
    """Summary for a batch of patch matches."""
    n_pairs: int = 0
    n_total_matches: int = 0
    mean_matches_per_pair: float = 0.0
    method: str = "ncc"

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be >= 0")
        if self.n_total_matches < 0:
            raise ValueError("n_total_matches must be >= 0")


@dataclass
class CanvasBuildRecord:
    """Result metadata from a single canvas build operation."""
    n_fragments: int
    coverage: float
    canvas_w: int
    canvas_h: int
    blend_mode: str = "overwrite"

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError("n_fragments must be >= 0")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0, 1]")
        if self.canvas_w < 1:
            raise ValueError("canvas_w must be >= 1")
        if self.canvas_h < 1:
            raise ValueError("canvas_h must be >= 1")

    @property
    def canvas_area(self) -> int:
        return self.canvas_w * self.canvas_h

    @property
    def is_well_covered(self) -> bool:
        return self.coverage >= 0.7


@dataclass
class CanvasBuildSummary:
    """Aggregate over multiple canvas build operations."""
    n_canvases: int = 0
    mean_coverage: float = 0.0
    well_covered_count: int = 0
    total_fragments: int = 0

    def __post_init__(self) -> None:
        if self.n_canvases < 0:
            raise ValueError("n_canvases must be >= 0")
        if self.well_covered_count < 0:
            raise ValueError("well_covered_count must be >= 0")
        if self.total_fragments < 0:
            raise ValueError("total_fragments must be >= 0")

    @property
    def well_covered_ratio(self) -> float:
        if self.n_canvases == 0:
            return 0.0
        return self.well_covered_count / self.n_canvases


def summarize_frequency_matches(
    records: Sequence[FrequencyMatchRecord],
) -> FrequencyMatchSummary:
    """Build a FrequencyMatchSummary from a list of records."""
    if not records:
        return FrequencyMatchSummary()
    total = len(records)
    similar = sum(1 for r in records if r.is_similar)
    scores = [r.similarity for r in records]
    return FrequencyMatchSummary(
        total_pairs=total,
        similar_pairs=similar,
        mean_similarity=float(sum(scores) / total),
        max_similarity=float(max(scores)),
        min_similarity=float(min(scores)),
    )


def filter_frequency_matches(
    records: Sequence[FrequencyMatchRecord],
    min_similarity: float = 0.0,
) -> List[FrequencyMatchRecord]:
    """Filter frequency match records by minimum similarity."""
    if min_similarity < 0.0 or min_similarity > 1.0:
        raise ValueError("min_similarity must be in [0, 1]")
    return [r for r in records if r.similarity >= min_similarity]


def summarize_canvas_builds(
    records: Sequence[CanvasBuildRecord],
) -> CanvasBuildSummary:
    """Build a CanvasBuildSummary from a list of build records."""
    if not records:
        return CanvasBuildSummary()
    n = len(records)
    mean_cov = float(sum(r.coverage for r in records) / n)
    well_covered = sum(1 for r in records if r.is_well_covered)
    total_frags = sum(r.n_fragments for r in records)
    return CanvasBuildSummary(
        n_canvases=n,
        mean_coverage=mean_cov,
        well_covered_count=well_covered,
        total_fragments=total_frags,
    )


def summarize_patch_matches(
    batch_results: Sequence[Sequence[PatchMatchRecord]],
    method: str = "ncc",
) -> PatchMatchSummary:
    """Build a PatchMatchSummary from batch match results."""
    n_pairs = len(batch_results)
    total = sum(len(r) for r in batch_results)
    mean = float(total / n_pairs) if n_pairs > 0 else 0.0
    return PatchMatchSummary(
        n_pairs=n_pairs,
        n_total_matches=total,
        mean_matches_per_pair=mean,
        method=method,
    )


def top_frequency_matches(
    records: Sequence[FrequencyMatchRecord],
    k: int,
) -> List[FrequencyMatchRecord]:
    """Return top-k most similar frequency match records."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return sorted(records, key=lambda r: r.similarity, reverse=True)[:k]
