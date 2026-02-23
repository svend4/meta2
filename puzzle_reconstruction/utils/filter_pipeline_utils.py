"""Утилиты конвейерной фильтрации кандидатных пар.

Provides lightweight dataclasses and helper functions for building
multi-step candidate filtering pipelines: step configuration, chaining
filters, tracking filter statistics, and batch summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FilterStepConfig:
    """Configuration for a single filter step."""
    name: str = "threshold"
    threshold: float = 0.5
    top_k: int = 0
    deduplicate: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {self.threshold}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")


@dataclass
class FilterStepResult:
    """Result of applying a single filter step."""
    step_name: str
    n_input: int
    n_output: int
    n_removed: int
    meta: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"FilterStepResult(step={self.step_name!r}, "
            f"in={self.n_input}, out={self.n_output}, "
            f"removed={self.n_removed})"
        )

    @property
    def removal_rate(self) -> float:
        """Fraction of entries removed in this step."""
        if self.n_input == 0:
            return 0.0
        return self.n_removed / self.n_input


@dataclass
class FilterPipelineSummary:
    """Summary of a complete filter pipeline run."""
    steps: List[FilterStepResult]
    n_initial: int
    n_final: int
    total_removed: int
    overall_removal_rate: float

    def __repr__(self) -> str:
        return (
            f"FilterPipelineSummary(steps={len(self.steps)}, "
            f"initial={self.n_initial}, final={self.n_final}, "
            f"removal_rate={self.overall_removal_rate:.3f})"
        )


def make_filter_step(
    step_name: str,
    n_input: int,
    n_output: int,
    meta: Optional[Dict] = None,
) -> FilterStepResult:
    """Create a single filter step result."""
    return FilterStepResult(
        step_name=step_name,
        n_input=n_input,
        n_output=n_output,
        n_removed=n_input - n_output,
        meta=meta or {},
    )


def steps_from_log(
    log_entries: List[Dict],
) -> List[FilterStepResult]:
    """Convert a list of log dicts to FilterStepResult list.

    Expected keys: ``step_name``, ``n_input``, ``n_output``.
    """
    result = []
    for entry in log_entries:
        step = make_filter_step(
            step_name=str(entry.get("step_name", "unknown")),
            n_input=int(entry.get("n_input", 0)),
            n_output=int(entry.get("n_output", 0)),
            meta={k: v for k, v in entry.items()
                  if k not in ("step_name", "n_input", "n_output")},
        )
        result.append(step)
    return result


def summarise_pipeline(
    steps: List[FilterStepResult],
) -> FilterPipelineSummary:
    """Compute a summary from a list of filter steps."""
    if not steps:
        return FilterPipelineSummary(
            steps=steps, n_initial=0, n_final=0,
            total_removed=0, overall_removal_rate=0.0,
        )
    n_initial = steps[0].n_input
    n_final = steps[-1].n_output
    total_removed = sum(s.n_removed for s in steps)
    rate = total_removed / n_initial if n_initial > 0 else 0.0
    return FilterPipelineSummary(
        steps=steps,
        n_initial=n_initial,
        n_final=n_final,
        total_removed=total_removed,
        overall_removal_rate=rate,
    )


def filter_effective_steps(
    steps: List[FilterStepResult],
) -> List[FilterStepResult]:
    """Return only steps that actually removed entries."""
    return [s for s in steps if s.n_removed > 0]


def filter_by_removal_rate(
    steps: List[FilterStepResult],
    min_rate: float = 0.0,
) -> List[FilterStepResult]:
    """Keep steps where removal_rate >= min_rate."""
    return [s for s in steps if s.removal_rate >= min_rate]


def most_aggressive_step(
    steps: List[FilterStepResult],
) -> Optional[FilterStepResult]:
    """Return the step that removed the most entries."""
    if not steps:
        return None
    return max(steps, key=lambda s: s.n_removed)


def least_aggressive_step(
    steps: List[FilterStepResult],
) -> Optional[FilterStepResult]:
    """Return the step that removed the fewest entries (including zero)."""
    if not steps:
        return None
    return min(steps, key=lambda s: s.n_removed)


def pipeline_stats(steps: List[FilterStepResult]) -> Dict:
    """Compute basic statistics over filter pipeline steps."""
    if not steps:
        return {
            "n_steps": 0, "total_removed": 0,
            "mean_removal_rate": 0.0,
            "max_removal_rate": 0.0,
            "min_removal_rate": 0.0,
        }
    rates = [s.removal_rate for s in steps]
    n = len(rates)
    return {
        "n_steps": n,
        "total_removed": sum(s.n_removed for s in steps),
        "mean_removal_rate": sum(rates) / n,
        "max_removal_rate": max(rates),
        "min_removal_rate": min(rates),
    }


def compare_pipelines(
    summary_a: FilterPipelineSummary,
    summary_b: FilterPipelineSummary,
) -> Dict:
    """Compare two filter pipeline summaries."""
    return {
        "n_steps_delta": len(summary_a.steps) - len(summary_b.steps),
        "n_final_delta": summary_a.n_final - summary_b.n_final,
        "total_removed_delta": summary_a.total_removed - summary_b.total_removed,
        "removal_rate_delta": (summary_a.overall_removal_rate
                               - summary_b.overall_removal_rate),
    }


def batch_summarise_pipelines(
    log_lists: List[List[Dict]],
) -> List[FilterPipelineSummary]:
    """Summarise multiple filter pipeline logs at once."""
    return [
        summarise_pipeline(steps_from_log(log))
        for log in log_lists
    ]
