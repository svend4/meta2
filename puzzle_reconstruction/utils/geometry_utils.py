"""Utilities for geometry-based comparisons in puzzle reconstruction."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0.0:
            raise ValueError("width must be >= 0")
        if self.height < 0.0:
            raise ValueError("height must be >= 0")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0.0:
            return 0.0
        return self.width / self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        """Axis-aligned IoU between this box and another."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x + self.width, other.x + other.width)
        iy2 = min(self.y + self.height, other.y + other.height)
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


@dataclass
class OverlapSummary:
    """Summary of polygon overlap analysis for a set of fragments."""
    n_pairs: int
    n_conflicting: int
    mean_iou: float
    max_iou: float
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be >= 0")
        if self.n_conflicting < 0:
            raise ValueError("n_conflicting must be >= 0")
        if not (0.0 <= self.mean_iou <= 1.0):
            raise ValueError("mean_iou must be in [0, 1]")
        if not (0.0 <= self.max_iou <= 1.0):
            raise ValueError("max_iou must be in [0, 1]")

    @property
    def conflict_ratio(self) -> float:
        if self.n_pairs == 0:
            return 0.0
        return self.n_conflicting / self.n_pairs


@dataclass
class GeometryComparisonRecord:
    """Record from a single geometry comparison."""
    idx1: int
    idx2: int
    aspect_score: float
    area_score: float
    total_score: float

    def __post_init__(self) -> None:
        if self.idx1 < 0 or self.idx2 < 0:
            raise ValueError("indices must be >= 0")
        for name, val in [
            ("aspect_score", self.aspect_score),
            ("area_score", self.area_score),
            ("total_score", self.total_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


def bbox_from_points(
    points: Sequence[Tuple[float, float]],
) -> BoundingBox:
    """Compute axis-aligned bounding box from a sequence of (x, y) points."""
    if not points:
        raise ValueError("points must be non-empty")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return BoundingBox(x=x_min, y=y_min,
                        width=x_max - x_min, height=y_max - y_min)


def summarize_overlaps(
    iou_values: Sequence[float],
    iou_threshold: float = 0.05,
) -> OverlapSummary:
    """Build an OverlapSummary from a list of pairwise IoU values."""
    if not iou_values:
        return OverlapSummary(n_pairs=0, n_conflicting=0,
                               mean_iou=0.0, max_iou=0.0)
    n = len(iou_values)
    conflicting = sum(1 for v in iou_values if v > iou_threshold)
    mean_iou = sum(iou_values) / n
    max_iou = max(iou_values)
    return OverlapSummary(
        n_pairs=n,
        n_conflicting=conflicting,
        mean_iou=min(mean_iou, 1.0),
        max_iou=min(max_iou, 1.0),
    )


def rank_geometry_comparisons(
    records: Sequence[GeometryComparisonRecord],
) -> List[Tuple[int, GeometryComparisonRecord]]:
    """Return (rank, record) tuples sorted by total_score descending."""
    sorted_recs = sorted(records, key=lambda r: r.total_score, reverse=True)
    return [(i + 1, r) for i, r in enumerate(sorted_recs)]
