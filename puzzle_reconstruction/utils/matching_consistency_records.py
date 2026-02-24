"""Records for boundary matching, color matching, and consistency checking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundaryMatchRecord:
    """Record of a boundary matching operation between two fragments."""

    idx1: int
    idx2: int
    side1: int
    side2: int
    hausdorff_score: float
    chamfer_score: float
    frechet_score: float
    total_score: float
    n_points: int = 20
    max_dist: float = 100.0

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.idx1, self.idx2), max(self.idx1, self.idx2))

    @property
    def is_good_match(self) -> bool:
        return self.total_score >= 0.7


@dataclass
class ColorMatchRecord:
    """Record of a color matching operation."""

    idx1: int
    idx2: int
    score: float
    hist_score: float
    moment_score: float
    profile_score: float
    colorspace: str = "hsv"
    metric: str = "bhatt"

    @property
    def is_compatible(self) -> bool:
        return self.score >= 0.6


@dataclass
class ConsistencyCheckRecord:
    """Record summarizing a consistency check run."""

    n_fragments: int
    n_violations: int
    score: float
    line_spacing_score: float = 1.0
    char_height_score: float = 1.0
    text_angle_score: float = 1.0
    margin_align_score: float = 1.0

    @property
    def is_consistent(self) -> bool:
        return self.n_violations == 0

    @property
    def mean_score(self) -> float:
        return (
            self.line_spacing_score
            + self.char_height_score
            + self.text_angle_score
            + self.margin_align_score
        ) / 4.0


@dataclass
class ColorHistogramRecord:
    """Record of a color histogram computation."""

    bins: int
    colorspace: str
    n_channels: int
    histogram_length: int
    min_value: float
    max_value: float
    mean_value: float


def make_boundary_match_record(
    idx1: int,
    idx2: int,
    side1: int,
    side2: int,
    hausdorff_score: float,
    chamfer_score: float,
    frechet_score: float,
    total_score: float,
    n_points: int = 20,
    max_dist: float = 100.0,
) -> BoundaryMatchRecord:
    """Create a BoundaryMatchRecord."""
    return BoundaryMatchRecord(
        idx1=idx1,
        idx2=idx2,
        side1=side1,
        side2=side2,
        hausdorff_score=hausdorff_score,
        chamfer_score=chamfer_score,
        frechet_score=frechet_score,
        total_score=total_score,
        n_points=n_points,
        max_dist=max_dist,
    )


def make_consistency_check_record(
    n_fragments: int,
    n_violations: int,
    score: float,
    method_scores: Optional[dict] = None,
) -> ConsistencyCheckRecord:
    """Create a ConsistencyCheckRecord from a method_scores dict."""
    ms = method_scores or {}
    return ConsistencyCheckRecord(
        n_fragments=n_fragments,
        n_violations=n_violations,
        score=score,
        line_spacing_score=ms.get("line_spacing", 1.0),
        char_height_score=ms.get("char_height", 1.0),
        text_angle_score=ms.get("text_angle", 1.0),
        margin_align_score=ms.get("margin_align", 1.0),
    )
