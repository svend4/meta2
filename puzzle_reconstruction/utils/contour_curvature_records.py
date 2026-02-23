"""Records and utilities for contour sampling and curvature analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnnealingRunRecord:
    """Record of a single simulated-annealing temperature run."""

    kind: str
    n_steps: int
    t_start: float
    t_end: float
    n_temperatures: int
    min_temp: float
    max_temp: float


@dataclass
class BlendOpRecord:
    """Record of a blend operation."""

    blend_type: str
    alpha: float
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass
class ContourSampleRecord:
    """Record of a contour sampling operation."""

    strategy: str
    n_source: int
    n_sampled: int
    closed: bool
    total_arc_length: float


@dataclass
class CurvatureAnalysisRecord:
    """Record of a curvature analysis operation."""

    n_points: int
    smooth_sigma: float
    total_curvature: float
    turning_angle: float
    n_corners: int
    n_inflections: int
    corner_threshold: float
    min_distance: int


@dataclass
class BatchCurvatureRecord:
    """Record of a batch curvature computation."""

    n_curves: int
    records: list[CurvatureAnalysisRecord] = field(default_factory=list)

    @property
    def mean_total_curvature(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.total_curvature for r in self.records) / len(self.records)

    @property
    def mean_n_corners(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.n_corners for r in self.records) / len(self.records)


@dataclass
class ContourNormRecord:
    """Record of a contour normalization operation."""

    n_points: int
    original_scale: float
    original_centroid_x: float
    original_centroid_y: float


def make_annealing_run_record(
    kind: str,
    n_steps: int,
    t_start: float,
    t_end: float,
    temperatures: list[float],
) -> AnnealingRunRecord:
    """Create an AnnealingRunRecord from a list of temperatures."""
    if not temperatures:
        return AnnealingRunRecord(
            kind=kind,
            n_steps=n_steps,
            t_start=t_start,
            t_end=t_end,
            n_temperatures=0,
            min_temp=0.0,
            max_temp=0.0,
        )
    return AnnealingRunRecord(
        kind=kind,
        n_steps=n_steps,
        t_start=t_start,
        t_end=t_end,
        n_temperatures=len(temperatures),
        min_temp=min(temperatures),
        max_temp=max(temperatures),
    )


def make_contour_sample_record(
    strategy: str,
    n_source: int,
    n_sampled: int,
    closed: bool = False,
    total_arc_length: float = 0.0,
) -> ContourSampleRecord:
    """Create a ContourSampleRecord."""
    return ContourSampleRecord(
        strategy=strategy,
        n_source=n_source,
        n_sampled=n_sampled,
        closed=closed,
        total_arc_length=total_arc_length,
    )


def make_curvature_analysis_record(
    n_points: int,
    smooth_sigma: float = 1.0,
    total_curvature: float = 0.0,
    turning_angle: float = 0.0,
    n_corners: int = 0,
    n_inflections: int = 0,
    corner_threshold: float = 0.1,
    min_distance: int = 3,
) -> CurvatureAnalysisRecord:
    """Create a CurvatureAnalysisRecord."""
    return CurvatureAnalysisRecord(
        n_points=n_points,
        smooth_sigma=smooth_sigma,
        total_curvature=total_curvature,
        turning_angle=turning_angle,
        n_corners=n_corners,
        n_inflections=n_inflections,
        corner_threshold=corner_threshold,
        min_distance=min_distance,
    )
