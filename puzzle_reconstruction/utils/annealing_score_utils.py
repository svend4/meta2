"""Утилиты оценки качества результатов отжига (simulated annealing).

Provides lightweight dataclasses and helper functions for analysing
the output of simulated-annealing placement: per-iteration scores,
temperature schedules, convergence, and batch summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AnnealingScoreConfig:
    """Configuration for annealing score analysis."""
    min_score: float = 0.0
    convergence_window: int = 10
    improvement_threshold: float = 1e-4
    prefer_high_score: bool = True

    def __post_init__(self) -> None:
        if self.convergence_window < 1:
            raise ValueError(
                f"convergence_window must be >= 1, got {self.convergence_window}"
            )
        if self.improvement_threshold < 0.0:
            raise ValueError(
                f"improvement_threshold must be >= 0, got {self.improvement_threshold}"
            )


@dataclass
class AnnealingScoreEntry:
    """Score record for one SA iteration."""
    iteration: int
    temperature: float
    current_score: float
    best_score: float
    accepted: bool
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


@dataclass
class AnnealingSummary:
    """Summary of an SA run."""
    entries: List[AnnealingScoreEntry]
    n_iterations: int
    final_score: float
    best_score: float
    n_accepted: int
    acceptance_rate: float
    converged: bool

    def __repr__(self) -> str:
        return (
            f"AnnealingSummary(n_iter={self.n_iterations}, "
            f"best={self.best_score:.4f}, "
            f"accept_rate={self.acceptance_rate:.3f}, "
            f"converged={self.converged})"
        )


def make_annealing_entry(
    iteration: int,
    temperature: float,
    current_score: float,
    best_score: float,
    accepted: bool,
    meta: Optional[Dict] = None,
) -> AnnealingScoreEntry:
    """Create a single annealing score entry."""
    return AnnealingScoreEntry(
        iteration=iteration,
        temperature=temperature,
        current_score=current_score,
        best_score=best_score,
        accepted=accepted,
        meta=meta or {},
    )


def entries_from_log(log: List[Dict]) -> List[AnnealingScoreEntry]:
    """Convert a list of dicts (SA iteration logs) to AnnealingScoreEntry list.

    Expected keys: ``iteration``, ``temperature``, ``current_score``,
    ``best_score``, ``accepted``.  Missing keys default to 0 / False.
    """
    result = []
    for i, rec in enumerate(log):
        entry = AnnealingScoreEntry(
            iteration=int(rec.get("iteration", i)),
            temperature=float(rec.get("temperature", 0.0)),
            current_score=float(rec.get("current_score", 0.0)),
            best_score=float(rec.get("best_score", 0.0)),
            accepted=bool(rec.get("accepted", False)),
            meta={k: v for k, v in rec.items()
                  if k not in ("iteration", "temperature", "current_score",
                               "best_score", "accepted")},
        )
        result.append(entry)
    return result


def summarise_annealing(
    entries: List[AnnealingScoreEntry],
    cfg: Optional[AnnealingScoreConfig] = None,
) -> AnnealingSummary:
    """Compute a summary from a list of annealing entries."""
    cfg = cfg or AnnealingScoreConfig()
    if not entries:
        return AnnealingSummary(
            entries=entries, n_iterations=0, final_score=0.0,
            best_score=0.0, n_accepted=0, acceptance_rate=0.0, converged=False,
        )
    n = len(entries)
    n_accepted = sum(1 for e in entries if e.accepted)
    best_score = max(e.best_score for e in entries)
    final_score = entries[-1].current_score
    acceptance_rate = n_accepted / n
    converged = _check_convergence(entries, cfg)
    return AnnealingSummary(
        entries=entries,
        n_iterations=n,
        final_score=final_score,
        best_score=best_score,
        n_accepted=n_accepted,
        acceptance_rate=acceptance_rate,
        converged=converged,
    )


def _check_convergence(
    entries: List[AnnealingScoreEntry],
    cfg: AnnealingScoreConfig,
) -> bool:
    """Return True if the best score stopped improving within the window."""
    w = cfg.convergence_window
    if len(entries) < w:
        return False
    recent = [e.best_score for e in entries[-w:]]
    return (max(recent) - min(recent)) <= cfg.improvement_threshold


def filter_accepted(
    entries: List[AnnealingScoreEntry],
) -> List[AnnealingScoreEntry]:
    """Return only accepted moves."""
    return [e for e in entries if e.accepted]


def filter_rejected(
    entries: List[AnnealingScoreEntry],
) -> List[AnnealingScoreEntry]:
    """Return only rejected moves."""
    return [e for e in entries if not e.accepted]


def filter_by_min_score(
    entries: List[AnnealingScoreEntry],
    min_score: float = 0.0,
) -> List[AnnealingScoreEntry]:
    """Keep entries where current_score >= min_score."""
    return [e for e in entries if e.current_score >= min_score]


def filter_by_temperature_range(
    entries: List[AnnealingScoreEntry],
    t_min: float = 0.0,
    t_max: float = float("inf"),
) -> List[AnnealingScoreEntry]:
    """Keep entries within a temperature range [t_min, t_max]."""
    return [e for e in entries if t_min <= e.temperature <= t_max]


def top_k_entries(
    entries: List[AnnealingScoreEntry],
    k: int,
    cfg: Optional[AnnealingScoreConfig] = None,
) -> List[AnnealingScoreEntry]:
    """Return top-k entries by current_score (descending)."""
    cfg = cfg or AnnealingScoreConfig()
    sorted_entries = sorted(entries, key=lambda e: e.current_score, reverse=True)
    return sorted_entries[:max(0, k)]


def annealing_score_stats(entries: List[AnnealingScoreEntry]) -> Dict:
    """Compute basic statistics over current_score values."""
    if not entries:
        return {
            "count": 0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "acceptance_rate": 0.0,
        }
    scores = [e.current_score for e in entries]
    n = len(scores)
    mean_s = sum(scores) / n
    variance = sum((s - mean_s) ** 2 for s in scores) / n
    std_s = variance ** 0.5
    n_acc = sum(1 for e in entries if e.accepted)
    return {
        "count": n,
        "mean": mean_s,
        "std": std_s,
        "min": min(scores),
        "max": max(scores),
        "acceptance_rate": n_acc / n,
    }


def best_entry(
    entries: List[AnnealingScoreEntry],
    cfg: Optional[AnnealingScoreConfig] = None,
) -> Optional[AnnealingScoreEntry]:
    """Return the entry with the highest current_score."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.current_score)


def compare_summaries(
    summary_a: AnnealingSummary,
    summary_b: AnnealingSummary,
) -> Dict:
    """Compare two annealing summaries and return delta metrics."""
    return {
        "best_score_delta": summary_a.best_score - summary_b.best_score,
        "final_score_delta": summary_a.final_score - summary_b.final_score,
        "acceptance_rate_delta": summary_a.acceptance_rate - summary_b.acceptance_rate,
        "n_iter_delta": summary_a.n_iterations - summary_b.n_iterations,
        "a_converged": summary_a.converged,
        "b_converged": summary_b.converged,
    }


def batch_summarise(
    logs: List[List[Dict]],
    cfg: Optional[AnnealingScoreConfig] = None,
) -> List[AnnealingSummary]:
    """Summarise multiple SA run logs at once."""
    return [summarise_annealing(entries_from_log(log), cfg) for log in logs]
