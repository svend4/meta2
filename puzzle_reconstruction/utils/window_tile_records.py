"""Records for window processing and tile operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Window-processing records ────────────────────────────────────────────────

@dataclass
class WindowOpRecord:
    """Records parameters and stats of a windowed signal operation."""
    operation: str
    signal_length: int
    window_size: int
    step: int
    n_windows: int
    label: str = ""

    _VALID_OPS = frozenset({"mean", "std", "max", "min", "split", "merge"})

    def __post_init__(self) -> None:
        if self.operation not in self._VALID_OPS:
            raise ValueError(f"Unknown operation: {self.operation!r}")
        if self.signal_length <= 0:
            raise ValueError(f"signal_length must be > 0, got {self.signal_length}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.step <= 0:
            raise ValueError(f"step must be > 0, got {self.step}")
        if self.n_windows < 0:
            raise ValueError(f"n_windows must be >= 0, got {self.n_windows}")

    @property
    def coverage(self) -> float:
        """Fraction of signal covered by windows."""
        covered = min(self.n_windows * self.window_size, self.signal_length)
        return covered / self.signal_length

    @property
    def has_overlap(self) -> bool:
        return self.step < self.window_size


@dataclass
class WindowFunctionRecord:
    """Records the result of applying a window function to a signal."""
    func_name: str
    input_length: int
    sum_before: float = 0.0
    sum_after: float = 0.0
    label: str = ""

    _VALID_FUNCS = frozenset({"rect", "hann", "hamming", "bartlett", "blackman"})

    def __post_init__(self) -> None:
        if self.func_name not in self._VALID_FUNCS:
            raise ValueError(f"Unknown window function: {self.func_name!r}")
        if self.input_length <= 0:
            raise ValueError(f"input_length must be > 0, got {self.input_length}")

    @property
    def attenuation_ratio(self) -> float:
        if self.sum_before == 0.0:
            return 1.0
        return self.sum_after / self.sum_before


# ─── Tile operation records ───────────────────────────────────────────────────

@dataclass
class TileOpRecord:
    """Records parameters and summary of a tile operation."""
    operation: str
    image_shape: Tuple[int, int]
    tile_h: int
    tile_w: int
    n_tiles: int
    label: str = ""

    _VALID_OPS = frozenset({"tile", "reassemble", "filter"})

    def __post_init__(self) -> None:
        if self.operation not in self._VALID_OPS:
            raise ValueError(f"Unknown tile operation: {self.operation!r}")
        if self.tile_h <= 0:
            raise ValueError(f"tile_h must be > 0, got {self.tile_h}")
        if self.tile_w <= 0:
            raise ValueError(f"tile_w must be > 0, got {self.tile_w}")
        if self.n_tiles < 0:
            raise ValueError(f"n_tiles must be >= 0, got {self.n_tiles}")

    @property
    def tile_area(self) -> int:
        return self.tile_h * self.tile_w

    @property
    def image_area(self) -> int:
        return self.image_shape[0] * self.image_shape[1]

    @property
    def coverage_ratio(self) -> float:
        if self.image_area == 0:
            return 0.0
        return min(self.n_tiles * self.tile_area / self.image_area, 1.0)


@dataclass
class TileFilterRecord:
    """Records the result of filtering tiles by content."""
    n_input: int
    n_kept: int
    min_foreground: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_input < 0:
            raise ValueError(f"n_input must be >= 0, got {self.n_input}")
        if self.n_kept < 0:
            raise ValueError(f"n_kept must be >= 0, got {self.n_kept}")
        if self.n_kept > self.n_input:
            raise ValueError("n_kept cannot exceed n_input")
        if not (0.0 <= self.min_foreground <= 1.0):
            raise ValueError(f"min_foreground must be in [0,1], got {self.min_foreground}")

    @property
    def n_removed(self) -> int:
        return self.n_input - self.n_kept

    @property
    def retention_rate(self) -> float:
        if self.n_input == 0:
            return 1.0
        return self.n_kept / self.n_input


# ─── Overlap / validation records ────────────────────────────────────────────

@dataclass
class OverlapSummaryRecord:
    """Summarises overlap detection results for an assembly."""
    n_fragments: int
    n_pairs_checked: int
    n_overlapping_pairs: int
    total_overlap_area: float = 0.0
    max_iou: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(f"n_fragments must be >= 0")
        if self.n_overlapping_pairs < 0:
            raise ValueError("n_overlapping_pairs must be >= 0")
        if self.total_overlap_area < 0.0:
            raise ValueError("total_overlap_area must be >= 0")
        if not (0.0 <= self.max_iou <= 1.0):
            raise ValueError(f"max_iou must be in [0,1], got {self.max_iou}")

    @property
    def is_valid(self) -> bool:
        return self.n_overlapping_pairs == 0

    @property
    def overlap_rate(self) -> float:
        if self.n_pairs_checked == 0:
            return 0.0
        return self.n_overlapping_pairs / self.n_pairs_checked


# ─── Score reporter records ───────────────────────────────────────────────────

@dataclass
class ScoreSummaryRecord:
    """Compact record of a scoring run."""
    n_metrics: int
    total_score: float
    passed: bool
    pass_threshold: float
    worst_metric: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.total_score <= 1.0):
            raise ValueError(f"total_score must be in [0,1], got {self.total_score}")
        if not (0.0 <= self.pass_threshold <= 1.0):
            raise ValueError(f"pass_threshold must be in [0,1], got {self.pass_threshold}")

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"

    @property
    def margin(self) -> float:
        """Distance from total_score to pass_threshold (positive = passing)."""
        return self.total_score - self.pass_threshold


# ─── Convenience constructors ─────────────────────────────────────────────────

def make_window_op_record(stat: str, signal_len: int, size: int,
                           step: int, n_windows: int,
                           label: str = "") -> WindowOpRecord:
    """Build a WindowOpRecord from rolling-stat parameters."""
    return WindowOpRecord(
        operation=stat,
        signal_length=signal_len,
        window_size=size,
        step=step,
        n_windows=n_windows,
        label=label,
    )


def make_tile_op_record(image_shape: Tuple[int, int],
                         tile_h: int, tile_w: int,
                         n_tiles: int,
                         operation: str = "tile",
                         label: str = "") -> TileOpRecord:
    """Build a TileOpRecord from tile operation parameters."""
    return TileOpRecord(
        operation=operation,
        image_shape=image_shape,
        tile_h=tile_h,
        tile_w=tile_w,
        n_tiles=n_tiles,
        label=label,
    )
