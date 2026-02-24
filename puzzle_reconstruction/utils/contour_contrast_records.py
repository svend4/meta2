"""Records for contour processing, contrast enhancement, and cost matrix operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContourProcessRecord:
    """Record of a contour processing operation."""

    fragment_id: int
    n_points_before: int
    n_points_after: int
    perimeter: float
    area: float
    compactness: float
    normalized: bool = True
    simplified: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.n_points_before == 0:
            return 0.0
        return self.n_points_after / self.n_points_before


@dataclass
class ContrastEnhanceRecord:
    """Record of a contrast enhancement operation."""

    method: str
    contrast_before: float
    contrast_after: float
    image_height: int
    image_width: int
    n_channels: int = 1

    @property
    def improvement(self) -> float:
        return self.contrast_after - self.contrast_before

    @property
    def improvement_ratio(self) -> float:
        if self.contrast_before == 0.0:
            return 0.0
        return self.improvement / self.contrast_before

    @property
    def is_grayscale(self) -> bool:
        return self.n_channels == 1


@dataclass
class CostMatrixRecord:
    """Record of a cost matrix construction."""

    n_fragments: int
    method: str
    min_cost: float
    max_cost: float
    mean_cost: float
    n_forbidden: int = 0
    normalized: bool = False

    @property
    def cost_range(self) -> float:
        return self.max_cost - self.min_cost


@dataclass
class ContourBatchRecord:
    """Record of a batch contour processing run."""

    n_contours: int
    n_points_config: int
    smooth_sigma: float
    rdp_epsilon: float
    normalize: bool
    n_successful: int = 0

    @property
    def success_rate(self) -> float:
        if self.n_contours == 0:
            return 0.0
        return self.n_successful / self.n_contours


def make_contour_process_record(
    fragment_id: int,
    n_points_before: int,
    n_points_after: int,
    perimeter: float,
    area: float,
    compactness: float,
    normalized: bool = True,
    simplified: bool = False,
) -> ContourProcessRecord:
    """Create a ContourProcessRecord."""
    return ContourProcessRecord(
        fragment_id=fragment_id,
        n_points_before=n_points_before,
        n_points_after=n_points_after,
        perimeter=perimeter,
        area=area,
        compactness=compactness,
        normalized=normalized,
        simplified=simplified,
    )


def make_contrast_enhance_record(
    method: str,
    contrast_before: float,
    contrast_after: float,
    image_shape: tuple[int, ...],
) -> ContrastEnhanceRecord:
    """Create a ContrastEnhanceRecord from image shape."""
    h = image_shape[0]
    w = image_shape[1]
    c = image_shape[2] if len(image_shape) == 3 else 1
    return ContrastEnhanceRecord(
        method=method,
        contrast_before=contrast_before,
        contrast_after=contrast_after,
        image_height=h,
        image_width=w,
        n_channels=c,
    )
