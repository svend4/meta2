"""Records and utilities for edge comparison, fragment classification/quality, and gradient flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EdgeCompareRecord:
    """Record of an edge comparison operation."""

    edge_id_a: int
    edge_id_b: int
    score: float
    dtw_dist: float
    css_sim: float
    fd_diff: float
    ifs_sim: float

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.edge_id_a, self.edge_id_b),
                max(self.edge_id_a, self.edge_id_b))


@dataclass
class CompatMatrixRecord:
    """Record of a compatibility matrix build."""

    n_edges: int
    min_score: float
    max_score: float
    mean_score: float


@dataclass
class FragmentClassifyRecord:
    """Record of a fragment classification result."""

    fragment_type: str
    confidence: float
    has_text: bool
    text_lines: int
    n_straight_sides: int
    texture_variance: float
    fill_ratio: float


@dataclass
class BatchClassifyRecord:
    """Record of a batch classification operation."""

    n_fragments: int
    records: list[FragmentClassifyRecord] = field(default_factory=list)

    @property
    def n_text_fragments(self) -> int:
        return sum(1 for r in self.records if r.has_text)

    @property
    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.fragment_type] = counts.get(r.fragment_type, 0) + 1
        return counts


@dataclass
class FragmentQualityRecord:
    """Record of a fragment quality assessment."""

    fragment_id: int
    score: float
    blur: float
    contrast: float
    coverage: float
    sharpness: float
    is_usable: bool


@dataclass
class GradientFlowRecord:
    """Record of a gradient field computation."""

    height: int
    width: int
    mean_magnitude: float
    std_magnitude: float
    edge_density: float
    dominant_angle: float
    ksize: int
    normalized: bool


def make_edge_compare_record(
    edge_id_a: int,
    edge_id_b: int,
    score: float,
    dtw_dist: float = 0.0,
    css_sim: float = 0.0,
    fd_diff: float = 0.0,
    ifs_sim: float = 0.0,
) -> EdgeCompareRecord:
    """Create an EdgeCompareRecord."""
    return EdgeCompareRecord(
        edge_id_a=edge_id_a,
        edge_id_b=edge_id_b,
        score=score,
        dtw_dist=dtw_dist,
        css_sim=css_sim,
        fd_diff=fd_diff,
        ifs_sim=ifs_sim,
    )


def make_fragment_classify_record(
    fragment_type: str,
    confidence: float,
    has_text: bool = False,
    text_lines: int = 0,
    n_straight_sides: int = 0,
    texture_variance: float = 0.0,
    fill_ratio: float = 1.0,
) -> FragmentClassifyRecord:
    """Create a FragmentClassifyRecord."""
    return FragmentClassifyRecord(
        fragment_type=fragment_type,
        confidence=confidence,
        has_text=has_text,
        text_lines=text_lines,
        n_straight_sides=n_straight_sides,
        texture_variance=texture_variance,
        fill_ratio=fill_ratio,
    )


def make_gradient_flow_record(
    height: int,
    width: int,
    mean_magnitude: float = 0.0,
    std_magnitude: float = 0.0,
    edge_density: float = 0.0,
    dominant_angle: float = 0.0,
    ksize: int = 3,
    normalized: bool = False,
) -> GradientFlowRecord:
    """Create a GradientFlowRecord."""
    return GradientFlowRecord(
        height=height,
        width=width,
        mean_magnitude=mean_magnitude,
        std_magnitude=std_magnitude,
        edge_density=edge_density,
        dominant_angle=dominant_angle,
        ksize=ksize,
        normalized=normalized,
    )
