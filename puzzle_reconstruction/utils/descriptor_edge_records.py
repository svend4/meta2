"""Records for descriptor combining, edge profile matching, and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DescriptorCombineRecord:
    """Record of a descriptor combine operation."""

    fragment_id: int
    used_names: list
    original_dim: int
    combined_dim: int
    normalized: bool = True
    l2_final: bool = True

    @property
    def is_reduced(self) -> bool:
        return self.combined_dim < self.original_dim

    @property
    def compression_ratio(self) -> float:
        if self.original_dim == 0:
            return 0.0
        return self.combined_dim / self.original_dim


@dataclass
class EdgeProfileRecord:
    """Record of an edge profile extraction."""

    fragment_id: int
    side: int
    method: str
    n_samples: int
    signal_mean: float
    signal_std: float

    @property
    def is_uniform(self) -> bool:
        return self.signal_std < 1e-6


@dataclass
class ProfileMatchRecord:
    """Record of an edge profile match operation."""

    idx1: int
    idx2: int
    side1: int
    side2: int
    score: float
    correlation: float
    dtw_score: float
    n_samples: int = 64

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.idx1, self.idx2), max(self.idx1, self.idx2))

    @property
    def is_good_match(self) -> bool:
        return self.score >= 0.6


@dataclass
class EdgeValidRecord:
    """Record of an edge pair validation."""

    idx1: int
    idx2: int
    valid: bool
    n_passed: int
    n_failed: int
    intensity_value: float = 0.0
    gap_value: float = 0.0
    normal_value: float = 0.0

    @property
    def pass_rate(self) -> float:
        total = self.n_passed + self.n_failed
        if total == 0:
            return 0.0
        return self.n_passed / total


@dataclass
class FeatureMatchRecord:
    """Record of a feature match between two images."""

    idx1: int
    idx2: int
    method: str
    score: float
    n_matches: int
    n_inliers: int
    n_keypoints_1: int = 0
    n_keypoints_2: int = 0

    @property
    def inlier_ratio(self) -> float:
        if self.n_matches == 0:
            return 0.0
        return self.n_inliers / self.n_matches

    @property
    def is_good_match(self) -> bool:
        return self.score >= 0.5 and self.n_inliers >= 4


def make_profile_match_record(
    idx1: int,
    idx2: int,
    side1: int,
    side2: int,
    score: float,
    correlation: float,
    dtw_score: float,
    n_samples: int = 64,
) -> ProfileMatchRecord:
    """Create a ProfileMatchRecord."""
    return ProfileMatchRecord(
        idx1=idx1,
        idx2=idx2,
        side1=side1,
        side2=side2,
        score=score,
        correlation=correlation,
        dtw_score=dtw_score,
        n_samples=n_samples,
    )


def make_feature_match_record(
    idx1: int,
    idx2: int,
    method: str,
    score: float,
    n_matches: int,
    n_inliers: int,
    n_keypoints: tuple[int, int] = (0, 0),
) -> FeatureMatchRecord:
    """Create a FeatureMatchRecord."""
    return FeatureMatchRecord(
        idx1=idx1,
        idx2=idx2,
        method=method,
        score=score,
        n_matches=n_matches,
        n_inliers=n_inliers,
        n_keypoints_1=n_keypoints[0],
        n_keypoints_2=n_keypoints[1],
    )
