"""Utility records for mask operations, layout refinement, and feature selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Mask operation records ───────────────────────────────────────────────────

@dataclass
class MaskOpRecord:
    """Records parameters and result summary of a mask operation."""
    operation: str
    input_shape: Tuple[int, int]
    n_nonzero_before: int
    n_nonzero_after: int
    label: str = ""

    _VALID_OPS = frozenset({"erode", "dilate", "invert", "and", "or", "xor", "crop"})

    def __post_init__(self) -> None:
        if self.operation not in self._VALID_OPS:
            raise ValueError(f"Unknown mask operation: {self.operation!r}")
        if self.n_nonzero_before < 0:
            raise ValueError("n_nonzero_before must be >= 0")

    @property
    def area_change(self) -> int:
        return self.n_nonzero_after - self.n_nonzero_before

    @property
    def coverage_ratio(self) -> float:
        total = self.input_shape[0] * self.input_shape[1]
        if total == 0:
            return 0.0
        return self.n_nonzero_after / total


@dataclass
class MaskCoverageRecord:
    """Records spatial coverage information across a set of masks."""
    n_masks: int
    canvas_shape: Tuple[int, int]
    n_covered_pixels: int
    n_total_pixels: int
    label: str = ""

    @property
    def coverage_ratio(self) -> float:
        if self.n_total_pixels == 0:
            return 0.0
        return self.n_covered_pixels / self.n_total_pixels

    @property
    def is_fully_covered(self) -> bool:
        return self.n_covered_pixels >= self.n_total_pixels


# ─── Completeness records ─────────────────────────────────────────────────────

@dataclass
class FragmentPlacementRecord:
    """Records placement state for a fragment assembly run."""
    n_total: int
    n_placed: int
    missing_ids: List[int] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        if self.n_total < 0:
            raise ValueError(f"n_total must be >= 0, got {self.n_total}")
        if self.n_placed < 0:
            raise ValueError(f"n_placed must be >= 0, got {self.n_placed}")
        if self.n_placed > self.n_total:
            raise ValueError("n_placed cannot exceed n_total")

    @property
    def coverage(self) -> float:
        if self.n_total == 0:
            return 1.0
        return self.n_placed / self.n_total

    @property
    def n_missing(self) -> int:
        return self.n_total - self.n_placed


# ─── Layout refinement records ────────────────────────────────────────────────

@dataclass
class LayoutDiffRecord:
    """Records the difference between two layout states."""
    n_fragments: int
    mean_shift: float
    max_shift: float
    n_moved: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.mean_shift < 0.0:
            raise ValueError(f"mean_shift must be >= 0, got {self.mean_shift}")
        if self.max_shift < 0.0:
            raise ValueError(f"max_shift must be >= 0, got {self.max_shift}")

    @property
    def is_stable(self) -> bool:
        return self.n_moved == 0


@dataclass
class LayoutScoreRecord:
    """Records layout quality scores over a refinement run."""
    n_pairs: int
    initial_score: float = 0.0
    final_score: float = 0.0
    n_iter: int = 0
    label: str = ""

    @property
    def score_improvement(self) -> float:
        return self.final_score - self.initial_score

    @property
    def converged(self) -> bool:
        return abs(self.score_improvement) < 1e-6


# ─── Feature selection records ────────────────────────────────────────────────

@dataclass
class FeatureSelectionRecord:
    """Records which features were selected and their scores."""
    method: str
    n_input_features: int
    n_selected_features: int
    mean_score: float = 0.0
    label: str = ""

    _VALID_METHODS = frozenset({"variance", "correlation", "rank", "pca"})

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(f"Unknown selection method: {self.method!r}")
        if self.n_selected_features > self.n_input_features:
            raise ValueError("n_selected_features cannot exceed n_input_features")

    @property
    def selection_ratio(self) -> float:
        if self.n_input_features == 0:
            return 0.0
        return self.n_selected_features / self.n_input_features


@dataclass
class PcaRecord:
    """Records PCA reduction results."""
    n_input_features: int
    n_components: int
    explained_variance_ratio: List[float] = field(default_factory=list)
    label: str = ""

    @property
    def total_variance_explained(self) -> float:
        return sum(self.explained_variance_ratio)

    @property
    def dominant_component_ratio(self) -> float:
        if not self.explained_variance_ratio:
            return 0.0
        return self.explained_variance_ratio[0]


# ─── Convenience constructors ─────────────────────────────────────────────────

def make_mask_coverage_record(masks: List[np.ndarray],
                               canvas_shape: Tuple[int, int],
                               label: str = "") -> MaskCoverageRecord:
    """Build a MaskCoverageRecord from a list of masks."""
    h, w = canvas_shape
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        mh, mw = m.shape[:2]
        sh, sw = min(h, mh), min(w, mw)
        union[:sh, :sw] |= (m[:sh, :sw] > 0).astype(np.uint8)
    return MaskCoverageRecord(
        n_masks=len(masks),
        canvas_shape=canvas_shape,
        n_covered_pixels=int(np.count_nonzero(union)),
        n_total_pixels=h * w,
        label=label,
    )


def make_layout_diff_record(diff_dict: dict, label: str = "") -> LayoutDiffRecord:
    """Build a LayoutDiffRecord from a compare_layouts result dict."""
    return LayoutDiffRecord(
        n_fragments=diff_dict.get("n_fragments", 0),
        mean_shift=float(diff_dict.get("mean_shift", 0.0)),
        max_shift=float(diff_dict.get("max_shift", 0.0)),
        n_moved=int(diff_dict.get("n_moved", 0)),
        label=label,
    )
