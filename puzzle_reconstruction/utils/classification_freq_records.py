"""Records for fragment classification, mapping, validation, and frequency analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FragmentClassifyRecord:
    """Record of a fragment classification result."""

    fragment_id: int
    fragment_type: str
    confidence: float
    has_text: bool
    text_lines: int
    n_straight_sides: int = 0

    @property
    def is_corner(self) -> bool:
        return self.fragment_type == "corner"

    @property
    def is_edge(self) -> bool:
        return self.fragment_type == "edge"

    @property
    def is_inner(self) -> bool:
        return self.fragment_type == "inner"


@dataclass
class FragmentMapRecord:
    """Record of a fragment mapping operation."""

    n_fragments: int
    n_zones: int
    n_assigned: int
    canvas_w: int
    canvas_h: int

    @property
    def coverage_ratio(self) -> float:
        if self.n_zones == 0:
            return 0.0
        return min(self.n_assigned / self.n_zones, 1.0)

    @property
    def assignment_ratio(self) -> float:
        if self.n_fragments == 0:
            return 0.0
        return self.n_assigned / self.n_fragments


@dataclass
class FragmentValidationRecord:
    """Record of a fragment validation run."""

    fragment_id: int
    passed: bool
    n_issues: int
    n_errors: int
    n_warnings: int
    width: float = 0.0
    height: float = 0.0
    coverage: float = 0.0

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return min(self.width, self.height) / max(self.width, self.height)


@dataclass
class FreqDescriptorRecord:
    """Record of a frequency descriptor extraction."""

    fragment_id: int
    n_bands: int
    centroid: float
    entropy: float
    dominant_band: int
    high_freq_ratio: float

    @property
    def is_high_frequency(self) -> bool:
        return self.high_freq_ratio > 0.5

    @property
    def is_smooth(self) -> bool:
        return self.centroid < 0.3


def make_fragment_classify_record(
    fragment_id: int,
    fragment_type: str,
    confidence: float,
    has_text: bool,
    text_lines: int,
    straight_sides: list,
) -> FragmentClassifyRecord:
    """Create a FragmentClassifyRecord."""
    return FragmentClassifyRecord(
        fragment_id=fragment_id,
        fragment_type=fragment_type,
        confidence=confidence,
        has_text=has_text,
        text_lines=text_lines,
        n_straight_sides=len(straight_sides),
    )


def make_freq_descriptor_record(
    fragment_id: int,
    n_bands: int,
    centroid: float,
    entropy: float,
    dominant_band: int,
    high_freq_ratio: float,
) -> FreqDescriptorRecord:
    """Create a FreqDescriptorRecord."""
    return FreqDescriptorRecord(
        fragment_id=fragment_id,
        n_bands=n_bands,
        centroid=centroid,
        entropy=entropy,
        dominant_band=dominant_band,
        high_freq_ratio=high_freq_ratio,
    )
