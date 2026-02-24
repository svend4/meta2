"""Records for illumination normalization, image stats, keypoints, and layout."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class IllumNormRecord:
    """Record of illumination normalization applied to a fragment."""

    fragment_id: int
    method: str
    original_mean: float
    original_std: float
    target_mean: float = 128.0
    target_std: float = 60.0

    @property
    def was_bright(self) -> bool:
        return self.original_mean > 150.0

    @property
    def was_dark(self) -> bool:
        return self.original_mean < 80.0

    @property
    def contrast_ratio(self) -> float:
        if self.original_std < 1e-9:
            return 0.0
        return self.target_std / self.original_std


@dataclass
class ImageStatsRecord:
    """Record of image statistics computed for a fragment."""

    fragment_id: int
    mean: float
    std: float
    entropy: float
    sharpness: float
    n_pixels: int

    @property
    def is_sharp(self) -> bool:
        return self.sharpness > 500.0

    @property
    def is_high_contrast(self) -> bool:
        return self.std > 50.0

    @property
    def is_informative(self) -> bool:
        return self.entropy > 4.0


@dataclass
class KeypointRecord:
    """Record of keypoint detection result for a fragment."""

    fragment_id: int
    n_keypoints: int
    n_matches: int = 0
    match_score: float = 0.0
    detector: str = "orb"

    @property
    def is_well_textured(self) -> bool:
        return self.n_keypoints >= 20

    @property
    def has_good_match(self) -> bool:
        return self.match_score > 0.5


@dataclass
class LayoutCellRecord:
    """Record of a single cell placement in the assembly layout."""

    fragment_idx: int
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def is_rotated(self) -> bool:
        return abs(self.rotation % 90.0) > 1.0


def make_illum_norm_record(
    fragment_id: int,
    method: str,
    original_mean: float,
    original_std: float,
    target_mean: float = 128.0,
    target_std: float = 60.0,
) -> IllumNormRecord:
    """Create an IllumNormRecord."""
    return IllumNormRecord(
        fragment_id=fragment_id,
        method=method,
        original_mean=original_mean,
        original_std=original_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def make_layout_cell_record(
    fragment_idx: int,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float = 0.0,
    confidence: float = 1.0,
) -> LayoutCellRecord:
    """Create a LayoutCellRecord."""
    return LayoutCellRecord(
        fragment_idx=fragment_idx,
        x=x,
        y=y,
        width=width,
        height=height,
        rotation=rotation,
        confidence=confidence,
    )
