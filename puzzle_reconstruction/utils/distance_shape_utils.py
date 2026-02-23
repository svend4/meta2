"""Utility records for distance matrix and shape-context pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Distance matrix records ──────────────────────────────────────────────────

@dataclass
class DistanceMatrixRecord:
    """Stores a labelled distance matrix together with its metadata."""
    label: str
    metric: str
    matrix: np.ndarray
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be a square 2-D array")

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    @property
    def max_value(self) -> float:
        return float(self.matrix.max())

    @property
    def min_offdiag(self) -> float:
        mask = ~np.eye(self.n, dtype=bool)
        return float(self.matrix[mask].min())


@dataclass
class SimilarityPair:
    """Represents a (i, j, similarity) triple."""
    i: int
    j: int
    similarity: float

    def __post_init__(self) -> None:
        if self.i < 0 or self.j < 0:
            raise ValueError("indices must be non-negative")
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(f"similarity must be in [0, 1], got {self.similarity}")

    @property
    def is_high(self) -> bool:
        return self.similarity >= 0.5


@dataclass
class DistanceBatchResult:
    """Summary of a batch distance computation."""
    n_queries: int
    metric: str
    top_pairs: List[Tuple[int, int, float]] = field(default_factory=list)

    @property
    def best_pair(self) -> Optional[Tuple[int, int, float]]:
        return self.top_pairs[0] if self.top_pairs else None


# ─── Shape-context records ────────────────────────────────────────────────────

@dataclass
class ContourMatchRecord:
    """Records the result of matching two contours via shape context."""
    contour_id_a: int
    contour_id_b: int
    cost: float
    n_correspondences: int
    similarity: float = 0.0

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(f"similarity must be in [0, 1], got {self.similarity}")

    @property
    def is_match(self) -> bool:
        return self.similarity >= 0.5


@dataclass
class ShapeContextBatchSummary:
    """Summary over a batch of shape-context computations."""
    n_contours: int
    mean_similarity: float
    best_pair: Optional[Tuple[int, int]] = None
    worst_pair: Optional[Tuple[int, int]] = None

    @property
    def is_valid(self) -> bool:
        return self.n_contours > 0 and 0.0 <= self.mean_similarity <= 1.0


# ─── Reconstruction metrics records ──────────────────────────────────────────

@dataclass
class MetricsRunRecord:
    """Stores metrics for one evaluation run."""
    run_id: str
    precision: float
    recall: float
    f1: float
    n_fragments: int
    extra: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [("precision", self.precision),
                          ("recall", self.recall), ("f1", self.f1)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    @property
    def is_perfect(self) -> bool:
        return (self.precision == 1.0 and self.recall == 1.0
                and self.f1 == 1.0)


@dataclass
class EvidenceAggregationRecord:
    """Record for one evidence aggregation step in the pipeline."""
    step: int
    pair_id: Tuple[int, int]
    n_channels: int
    confidence: float
    dominant_channel: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


# ─── Convenience constructors ─────────────────────────────────────────────────

def make_distance_record(label: str, metric: str,
                          matrix: np.ndarray,
                          normalized: bool = False) -> DistanceMatrixRecord:
    return DistanceMatrixRecord(label=label, metric=metric,
                                 matrix=matrix, normalized=normalized)


def make_contour_match(id_a: int, id_b: int,
                        cost: float, n_corr: int,
                        similarity: float = 0.0) -> ContourMatchRecord:
    return ContourMatchRecord(contour_id_a=id_a, contour_id_b=id_b,
                               cost=cost, n_correspondences=n_corr,
                               similarity=similarity)
