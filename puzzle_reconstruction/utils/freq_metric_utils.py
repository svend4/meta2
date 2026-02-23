"""Utility helpers for frequency analysis and metric tracking pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Frequency analysis records ───────────────────────────────────────────────

@dataclass
class BandEnergyRecord:
    """Stores per-band energy values for a single fragment."""
    fragment_id: int
    band_energies: List[float]
    n_bands: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_bands = len(self.band_energies)

    @property
    def dominant_band(self) -> int:
        """Index of the band with highest energy."""
        if not self.band_energies:
            return 0
        return int(max(range(self.n_bands), key=lambda i: self.band_energies[i]))

    @property
    def total_energy(self) -> float:
        return sum(self.band_energies)

    @property
    def normalized_energies(self) -> List[float]:
        total = self.total_energy
        if total == 0.0:
            return [0.0] * self.n_bands
        return [e / total for e in self.band_energies]


@dataclass
class SpectrumComparisonRecord:
    """Result of comparing two frequency descriptors."""
    fragment_id_a: int
    fragment_id_b: int
    similarity: float
    centroid_diff: float = 0.0
    entropy_diff: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(f"similarity must be in [0, 1], got {self.similarity}")

    @property
    def is_match(self) -> bool:
        return self.similarity >= 0.5


@dataclass
class FreqBatchSummary:
    """Summary statistics over a batch of frequency descriptors."""
    n_fragments: int
    mean_entropy: float
    mean_centroid: float
    n_bands: int

    @property
    def is_valid(self) -> bool:
        return self.n_fragments > 0 and self.n_bands > 0


# ─── Metric tracking records ──────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    """A named snapshot of metric values at a given step."""
    step: int
    values: Dict[str, float]
    label: str = ""

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")

    @property
    def metric_names(self) -> List[str]:
        return list(self.values.keys())

    @property
    def n_metrics(self) -> int:
        return len(self.values)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)


@dataclass
class MetricRunSummary:
    """Aggregated summary over a tracked training/eval run."""
    namespace: str
    total_steps: int
    best_values: Dict[str, float] = field(default_factory=dict)
    worst_values: Dict[str, float] = field(default_factory=dict)
    final_values: Dict[str, float] = field(default_factory=dict)

    def best(self, name: str) -> Optional[float]:
        return self.best_values.get(name)

    def final(self, name: str) -> Optional[float]:
        return self.final_values.get(name)

    @property
    def tracked_metrics(self) -> List[str]:
        return list(self.final_values.keys())


@dataclass
class MovingAverageResult:
    """Result of a moving-average smoothing over a metric series."""
    metric_name: str
    window: int
    smoothed: List[float]

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    @property
    def length(self) -> int:
        return len(self.smoothed)

    def at(self, idx: int) -> float:
        return self.smoothed[idx]


# ─── Assembly / greedy pipeline records ───────────────────────────────────────

@dataclass
class GreedyStepRecord:
    """Records a single greedy assembly step."""
    step: int
    fragment_id: int
    anchor_id: int
    score: float
    position: Tuple[float, float] = (0.0, 0.0)
    angle: float = 0.0

    def __post_init__(self) -> None:
        if self.score < 0.0:
            raise ValueError(f"score must be >= 0, got {self.score}")


@dataclass
class AssemblyRunRecord:
    """Full record of a greedy assembly run."""
    n_fragments: int
    steps: List[GreedyStepRecord] = field(default_factory=list)
    total_score: float = 0.0
    n_orphans: int = 0

    @property
    def n_placed(self) -> int:
        return len(self.steps)

    @property
    def placement_rate(self) -> float:
        if self.n_fragments == 0:
            return 0.0
        return self.n_placed / self.n_fragments


def make_band_energy_record(fragment_id: int,
                             band_energies: List[float]) -> BandEnergyRecord:
    """Convenience constructor for BandEnergyRecord."""
    return BandEnergyRecord(fragment_id=fragment_id, band_energies=band_energies)


def make_metric_snapshot(step: int,
                          values: Dict[str, float],
                          label: str = "") -> MetricSnapshot:
    """Convenience constructor for MetricSnapshot."""
    return MetricSnapshot(step=step, values=dict(values), label=label)


def make_greedy_step(step: int, fragment_id: int, anchor_id: int,
                     score: float) -> GreedyStepRecord:
    """Convenience constructor for GreedyStepRecord."""
    return GreedyStepRecord(step=step, fragment_id=fragment_id,
                            anchor_id=anchor_id, score=score)
