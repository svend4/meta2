"""Utility records for score normalization, gradient analysis, and seam metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Score normalization records ──────────────────────────────────────────────

@dataclass
class NormalizationRecord:
    """Records parameters and output of a score normalization run."""
    method: str
    n_scores: int
    min_val: float = 0.0
    max_val: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.method not in ("minmax", "zscore", "rank"):
            raise ValueError(f"Unknown normalization method: {self.method!r}")
        if self.n_scores < 0:
            raise ValueError(f"n_scores must be >= 0, got {self.n_scores}")

    @property
    def value_range(self) -> float:
        return self.max_val - self.min_val


@dataclass
class ScoreCalibrationRecord:
    """Stores calibration mapping between raw and calibrated score distributions."""
    n_reference: int
    n_target: int
    shift: float = 0.0
    scale: float = 1.0
    label: str = ""

    @property
    def is_identity(self) -> bool:
        return self.shift == 0.0 and self.scale == 1.0


# ─── Gradient analysis records ────────────────────────────────────────────────

@dataclass
class GradientRunRecord:
    """Summary record for a gradient analysis run."""
    n_images: int
    kernel: str
    ksize: int
    mean_energy: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_images < 0:
            raise ValueError(f"n_images must be >= 0, got {self.n_images}")
        if self.ksize < 1:
            raise ValueError(f"ksize must be >= 1, got {self.ksize}")

    @property
    def has_data(self) -> bool:
        return self.n_images > 0


@dataclass
class OrientationHistogramRecord:
    """Stores an orientation histogram and its metadata."""
    n_bins: int
    histogram: List[float] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.histogram) not in (0, self.n_bins):
            raise ValueError("histogram length must be 0 or n_bins")

    @property
    def dominant_bin(self) -> Optional[int]:
        if not self.histogram:
            return None
        return int(np.argmax(self.histogram))

    @property
    def is_normalized(self) -> bool:
        if not self.histogram:
            return True
        return abs(sum(self.histogram) - 1.0) < 1e-5


# ─── Seam analysis records ────────────────────────────────────────────────────

@dataclass
class SeamRunRecord:
    """Summary of a batch seam analysis run."""
    n_pairs: int
    mean_quality: float = 0.0
    min_quality: float = 1.0
    max_quality: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError(f"n_pairs must be >= 0, got {self.n_pairs}")
        if not (0.0 <= self.mean_quality <= 1.0):
            raise ValueError(
                f"mean_quality must be in [0, 1], got {self.mean_quality}"
            )

    @property
    def has_pairs(self) -> bool:
        return self.n_pairs > 0


@dataclass
class SeamScoreMatrix:
    """Sparse matrix of seam quality scores for a set of fragment pairs."""
    n_fragments: int
    scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    label: str = ""

    def get(self, i: int, j: int, default: float = 0.0) -> float:
        """Return quality score for pair (i, j) or its mirror (j, i)."""
        return self.scores.get((i, j), self.scores.get((j, i), default))

    @property
    def n_scored_pairs(self) -> int:
        return len(self.scores)


# ─── Score aggregation records ────────────────────────────────────────────────

@dataclass
class AggregationRunRecord:
    """Records parameters and results of a batch score aggregation run."""
    method: str
    n_items: int
    n_channels: int
    mean_score: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.method not in ("weighted_avg", "harmonic", "min", "max"):
            raise ValueError(f"Unknown aggregation method: {self.method!r}")

    @property
    def is_empty(self) -> bool:
        return self.n_items == 0


# ─── Convenience constructors ─────────────────────────────────────────────────

def make_normalization_record(method: str, scores: List[float],
                               label: str = "") -> NormalizationRecord:
    """Build a NormalizationRecord from a list of scores."""
    v = np.asarray(scores, dtype=float)
    return NormalizationRecord(
        method=method,
        n_scores=len(scores),
        min_val=float(v.min()) if len(v) > 0 else 0.0,
        max_val=float(v.max()) if len(v) > 0 else 1.0,
        label=label,
    )


def make_seam_run_record(qualities: List[float],
                          label: str = "") -> SeamRunRecord:
    """Build a SeamRunRecord from a list of quality scores."""
    if not qualities:
        return SeamRunRecord(n_pairs=0, label=label)
    arr = np.asarray(qualities, dtype=float)
    return SeamRunRecord(
        n_pairs=len(qualities),
        mean_quality=float(arr.mean()),
        min_quality=float(arr.min()),
        max_quality=float(arr.max()),
        label=label,
    )
