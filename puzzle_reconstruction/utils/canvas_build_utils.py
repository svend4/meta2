"""Утилиты планирования и анализа сборки холста.

Provides lightweight dataclasses and helper functions for tracking
canvas build results: configuration, placement entries, summaries,
filtering, comparison, and batch operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CanvasBuildConfig:
    """Configuration for canvas build analysis."""
    min_coverage: float = 0.0
    max_fragments: int = 1000
    blend_mode: str = "overwrite"

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(
                f"min_coverage must be in [0, 1], got {self.min_coverage}")
        if self.max_fragments < 1:
            raise ValueError(
                f"max_fragments must be >= 1, got {self.max_fragments}")
        valid_modes = {"overwrite", "average"}
        if self.blend_mode not in valid_modes:
            raise ValueError(
                f"blend_mode must be one of {valid_modes}, "
                f"got {self.blend_mode!r}")


@dataclass
class PlacementEntry:
    """A single fragment placement record."""
    fragment_id: int
    x: int
    y: int
    w: int
    h: int
    coverage_contribution: float = 0.0
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id must be >= 0, got {self.fragment_id}")
        if self.w < 1 or self.h < 1:
            raise ValueError(
                f"w and h must be >= 1, got ({self.w}, {self.h})")

    @property
    def area(self) -> int:
        """Pixel area of this placement."""
        return self.w * self.h

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class CanvasBuildSummary:
    """Summary of a canvas build operation."""
    entries: List[PlacementEntry]
    n_placed: int
    canvas_w: int
    canvas_h: int
    coverage: float
    total_area: int

    def __repr__(self) -> str:
        return (
            f"CanvasBuildSummary(n_placed={self.n_placed}, "
            f"canvas=({self.canvas_w}×{self.canvas_h}), "
            f"coverage={self.coverage:.3f})"
        )


def make_placement_entry(
    fragment_id: int,
    x: int,
    y: int,
    w: int,
    h: int,
    coverage_contribution: float = 0.0,
    meta: Optional[Dict] = None,
) -> PlacementEntry:
    """Create a single placement entry."""
    return PlacementEntry(
        fragment_id=fragment_id,
        x=x, y=y, w=w, h=h,
        coverage_contribution=coverage_contribution,
        meta=meta or {},
    )


def entries_from_placements(
    placements: List[Tuple[int, int, int, int, int]],
) -> List[PlacementEntry]:
    """Build entries from list of (fid, x, y, w, h) tuples."""
    return [
        make_placement_entry(fid, x, y, w, h)
        for fid, x, y, w, h in placements
    ]


def summarise_canvas_build(
    entries: List[PlacementEntry],
    canvas_w: int,
    canvas_h: int,
    coverage: float,
) -> CanvasBuildSummary:
    """Compute a summary of a canvas build from placement entries."""
    total_area = sum(e.area for e in entries)
    return CanvasBuildSummary(
        entries=entries,
        n_placed=len(entries),
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        coverage=coverage,
        total_area=total_area,
    )


def filter_by_area(
    entries: List[PlacementEntry],
    min_area: int = 0,
    max_area: int = 10 ** 9,
) -> List[PlacementEntry]:
    """Keep placements whose area is in [min_area, max_area]."""
    return [e for e in entries if min_area <= e.area <= max_area]


def filter_by_coverage_contribution(
    entries: List[PlacementEntry],
    min_contrib: float = 0.0,
) -> List[PlacementEntry]:
    """Keep placements with coverage_contribution >= min_contrib."""
    return [e for e in entries if e.coverage_contribution >= min_contrib]


def top_k_by_coverage(
    entries: List[PlacementEntry],
    k: int = 10,
) -> List[PlacementEntry]:
    """Return top-k placements by coverage_contribution (descending)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranked = sorted(entries, key=lambda e: e.coverage_contribution,
                    reverse=True)
    return ranked[:k]


def canvas_build_stats(entries: List[PlacementEntry]) -> Dict:
    """Compute basic statistics over placement entries."""
    if not entries:
        return {"n": 0, "total_area": 0, "mean_area": 0.0,
                "mean_coverage_contribution": 0.0}
    n = len(entries)
    return {
        "n": n,
        "total_area": sum(e.area for e in entries),
        "mean_area": sum(e.area for e in entries) / n,
        "mean_coverage_contribution": sum(
            e.coverage_contribution for e in entries) / n,
    }


def compare_canvas_summaries(
    a: CanvasBuildSummary,
    b: CanvasBuildSummary,
) -> Dict:
    """Compare two canvas build summaries."""
    return {
        "n_placed_delta": a.n_placed - b.n_placed,
        "coverage_delta": a.coverage - b.coverage,
        "total_area_delta": a.total_area - b.total_area,
        "canvas_w_delta": a.canvas_w - b.canvas_w,
        "canvas_h_delta": a.canvas_h - b.canvas_h,
    }


def batch_summarise_canvas_builds(
    build_specs: List[Tuple[List[PlacementEntry], int, int, float]],
) -> List[CanvasBuildSummary]:
    """Summarise multiple canvas builds at once.

    Each element of build_specs is (entries, canvas_w, canvas_h, coverage).
    """
    return [
        summarise_canvas_build(entries, cw, ch, cov)
        for entries, cw, ch, cov in build_specs
    ]
