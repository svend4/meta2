"""Utility types for tracking assembly state changes and config evolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class AssemblyStateRecord:
    """Snapshot of an assembly state at a particular step."""
    step: int
    n_placed: int
    n_fragments: int
    coverage: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")
        if self.n_placed < 0:
            raise ValueError("n_placed must be >= 0")
        if self.n_fragments < 1:
            raise ValueError("n_fragments must be >= 1")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0, 1]")

    @property
    def is_complete(self) -> bool:
        return self.n_placed == self.n_fragments


@dataclass
class AssemblyStateHistory:
    """Ordered history of assembly state records."""
    records: List[AssemblyStateRecord] = field(default_factory=list)

    def append(self, record: AssemblyStateRecord) -> None:
        self.records.append(record)

    @property
    def n_steps(self) -> int:
        return len(self.records)

    @property
    def last_coverage(self) -> float:
        if not self.records:
            return 0.0
        return self.records[-1].coverage

    @property
    def is_monotone(self) -> bool:
        """Return True if coverage never decreases."""
        covs = [r.coverage for r in self.records]
        return all(covs[i] <= covs[i + 1] for i in range(len(covs) - 1))


@dataclass
class ConfigChangeRecord:
    """Records a single config change event."""
    key: str
    old_value: Any
    new_value: Any
    step: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.step < 0:
            raise ValueError("step must be >= 0")

    @property
    def changed(self) -> bool:
        return self.old_value != self.new_value


@dataclass
class ConfigChangeLog:
    """Log of config changes over multiple steps."""
    records: List[ConfigChangeRecord] = field(default_factory=list)

    def append(self, record: ConfigChangeRecord) -> None:
        self.records.append(record)

    @property
    def n_changes(self) -> int:
        return sum(1 for r in self.records if r.changed)

    @property
    def changed_keys(self) -> List[str]:
        return sorted({r.key for r in self.records if r.changed})


@dataclass
class CandidateFilterRecord:
    """Tracks results of a single filter operation."""
    filter_name: str
    n_input: int
    n_kept: int
    n_removed: int
    threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.filter_name:
            raise ValueError("filter_name must be non-empty")
        if self.n_input < 0:
            raise ValueError("n_input must be >= 0")
        if self.n_kept < 0:
            raise ValueError("n_kept must be >= 0")
        if self.n_removed < 0:
            raise ValueError("n_removed must be >= 0")

    @property
    def keep_ratio(self) -> float:
        if self.n_input == 0:
            return 0.0
        return self.n_kept / self.n_input


@dataclass
class FilterPipelineSummary:
    """Summary of a multi-step filter pipeline."""
    stages: List[CandidateFilterRecord] = field(default_factory=list)

    def add_stage(self, record: CandidateFilterRecord) -> None:
        self.stages.append(record)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def total_removed(self) -> int:
        return sum(s.n_removed for s in self.stages)

    @property
    def final_n_kept(self) -> int:
        if not self.stages:
            return 0
        return self.stages[-1].n_kept


def summarize_assembly_history(
    history: AssemblyStateHistory,
) -> Dict[str, Any]:
    """Produce a summary dict from an AssemblyStateHistory."""
    if not history.records:
        return {"n_steps": 0, "final_coverage": 0.0, "is_monotone": True}
    return {
        "n_steps": history.n_steps,
        "final_coverage": history.last_coverage,
        "is_monotone": history.is_monotone,
        "is_complete": history.records[-1].is_complete,
    }


def build_filter_pipeline_summary(
    records: Sequence[CandidateFilterRecord],
) -> FilterPipelineSummary:
    """Build a FilterPipelineSummary from a sequence of records."""
    summary = FilterPipelineSummary()
    for r in records:
        summary.add_stage(r)
    return summary


def build_config_change_log(
    config_diffs: Sequence[Dict[str, Tuple[Any, Any]]],
) -> ConfigChangeLog:
    """Build a ConfigChangeLog from a sequence of diff dicts."""
    log = ConfigChangeLog()
    for step, diff in enumerate(config_diffs):
        for key, (old_val, new_val) in diff.items():
            log.append(ConfigChangeRecord(
                key=key,
                old_value=old_val,
                new_value=new_val,
                step=step,
            ))
    return log
