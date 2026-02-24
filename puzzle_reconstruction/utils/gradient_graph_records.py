"""Records for gradient flow, graph matching, histogram utils, and homography."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GradientFlowRecord:
    """Record of gradient field analysis for a single fragment."""

    fragment_id: int
    mean_magnitude: float
    std_magnitude: float
    edge_density: float
    dominant_angle: float
    n_boundary_points: int = 0

    @property
    def is_high_texture(self) -> bool:
        return self.mean_magnitude > 20.0

    @property
    def is_edge_rich(self) -> bool:
        return self.edge_density > 0.3


@dataclass
class GraphMatchRecord:
    """Record of graph-based matching result for a fragment pair."""

    fid_a: int
    fid_b: int
    edge_weight: float
    mst_rank: int = 0
    spectral_position: int = 0

    @property
    def pair_key(self) -> Tuple[int, int]:
        return (min(self.fid_a, self.fid_b), max(self.fid_a, self.fid_b))

    @property
    def is_strong_edge(self) -> bool:
        return self.edge_weight > 0.7

    @property
    def is_mst_top(self) -> bool:
        return self.mst_rank == 1


@dataclass
class HistogramRecord:
    """Record of histogram comparison between two fragments."""

    id_a: int
    id_b: int
    chi_squared: float
    intersection: float
    emd: float
    n_bins: int = 256

    @property
    def is_similar(self) -> bool:
        return self.intersection > 0.7

    @property
    def is_dissimilar(self) -> bool:
        return self.chi_squared > 1.0 or self.intersection < 0.3


@dataclass
class HomographyRecord:
    """Record of a homography estimation between two fragment images."""

    id_src: int
    id_dst: int
    n_inliers: int
    reproj_err: float
    is_valid: bool
    method: str = "ransac"

    @property
    def is_good(self) -> bool:
        return self.is_valid and self.reproj_err < 2.0 and self.n_inliers >= 8

    @property
    def quality_score(self) -> float:
        if not self.is_valid or self.n_inliers == 0:
            return 0.0
        err_term = 1.0 / (1.0 + self.reproj_err)
        inlier_term = min(self.n_inliers / 20.0, 1.0)
        return 0.5 * err_term + 0.5 * inlier_term


def make_gradient_flow_record(
    fragment_id: int,
    mean_magnitude: float,
    std_magnitude: float,
    edge_density: float,
    dominant_angle: float,
    n_boundary_points: int = 0,
) -> GradientFlowRecord:
    """Create a GradientFlowRecord."""
    return GradientFlowRecord(
        fragment_id=fragment_id,
        mean_magnitude=mean_magnitude,
        std_magnitude=std_magnitude,
        edge_density=edge_density,
        dominant_angle=dominant_angle,
        n_boundary_points=n_boundary_points,
    )


def make_homography_record(
    id_src: int,
    id_dst: int,
    n_inliers: int,
    reproj_err: float,
    is_valid: bool,
    method: str = "ransac",
) -> HomographyRecord:
    """Create a HomographyRecord."""
    return HomographyRecord(
        id_src=id_src,
        id_dst=id_dst,
        n_inliers=n_inliers,
        reproj_err=reproj_err,
        is_valid=is_valid,
        method=method,
    )
