"""Utilities for assembling and summarising multi-stage scoring pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class StageResult:
    """Result of one scoring stage in a pipeline."""
    stage_name: str
    score: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("stage_name must be non-empty")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be in [0, 1]")
        if self.weight < 0.0:
            raise ValueError("weight must be >= 0")

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class PipelineReport:
    """Aggregated report from a multi-stage scoring pipeline."""
    stages: List[StageResult] = field(default_factory=list)

    def add_stage(self, result: StageResult) -> None:
        self.stages.append(result)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.stages)

    @property
    def weighted_score(self) -> float:
        """Weighted average score across all stages."""
        total_w = self.total_weight
        if total_w == 0.0:
            return 0.0
        return sum(s.weighted_score for s in self.stages) / total_w

    def stage_by_name(self, name: str) -> Optional[StageResult]:
        for s in self.stages:
            if s.stage_name == name:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_stages": self.n_stages,
            "weighted_score": self.weighted_score,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "score": s.score,
                    "weight": s.weight,
                }
                for s in self.stages
            ],
        }


@dataclass
class BoundaryScoreRecord:
    """Stores scores from a single boundary comparison."""
    idx1: int
    idx2: int
    hausdorff_score: float
    chamfer_score: float
    frechet_score: float
    total_score: float

    def __post_init__(self) -> None:
        if self.idx1 < 0 or self.idx2 < 0:
            raise ValueError("idx1 and idx2 must be >= 0")
        for name, val in [
            ("hausdorff_score", self.hausdorff_score),
            ("chamfer_score", self.chamfer_score),
            ("frechet_score", self.frechet_score),
            ("total_score", self.total_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class PatchComparisonRecord:
    """Result of a single patch comparison."""
    row: int
    col: int
    method: str
    value: float

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("method must be non-empty")
        if self.row < 0 or self.col < 0:
            raise ValueError("row and col must be >= 0")


def build_pipeline_report(
    stage_results: Sequence[StageResult],
) -> PipelineReport:
    """Build a PipelineReport from a sequence of StageResults."""
    report = PipelineReport()
    for r in stage_results:
        report.add_stage(r)
    return report


def summarize_boundary_scores(
    records: Sequence[BoundaryScoreRecord],
) -> Dict[str, Any]:
    """Return aggregate summary of boundary score records."""
    if not records:
        return {"n_pairs": 0, "mean_total": 0.0, "max_total": 0.0}
    totals = [r.total_score for r in records]
    return {
        "n_pairs": len(records),
        "mean_total": sum(totals) / len(totals),
        "max_total": max(totals),
    }


def rank_stage_results(
    results: Sequence[StageResult],
) -> List[tuple[int, StageResult]]:
    """Return (rank, StageResult) tuples sorted by score descending."""
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    return [(i + 1, r) for i, r in enumerate(sorted_results)]
