"""
Утилиты анализа сегментов маски.

Post-processing helpers for binary segmentation masks: connected-component
analysis, region filtering, mask statistics, and boundary extraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """Parameters for segment analysis."""

    min_area: int = 50
    """Minimum region area (pixels) to keep."""

    max_aspect_ratio: float = 10.0
    """Maximum bounding-box aspect ratio (long / short); drop thin slivers."""

    border_margin: int = 2
    """Pixels from image border; regions touching this margin may be filtered."""

    def __post_init__(self) -> None:
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0")
        if self.max_aspect_ratio <= 0:
            raise ValueError("max_aspect_ratio must be > 0")
        if self.border_margin < 0:
            raise ValueError("border_margin must be >= 0")


# ---------------------------------------------------------------------------
# Region descriptor
# ---------------------------------------------------------------------------

@dataclass
class RegionInfo:
    """Describes a single connected component in a binary mask."""

    label: int
    area: int
    bbox: Tuple[int, int, int, int]   # (y0, x0, y1, x1)
    centroid: np.ndarray              # shape (2,), (y, x)
    aspect_ratio: float

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": self.centroid.tolist(),
            "aspect_ratio": self.aspect_ratio,
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def label_mask(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask using a simple BFS flood-fill
    (no OpenCV dependency).

    Parameters
    ----------
    mask : ndarray, shape (H, W), uint8 or bool
        Non-zero pixels are considered foreground.

    Returns
    -------
    labels : ndarray int32, shape (H, W)   — 0 = background, 1..N = regions
    n_labels : int                          — number of regions found
    """
    binary = (np.asarray(mask) > 0).astype(np.int32)
    h, w = binary.shape
    labels = np.zeros_like(binary)
    current_label = 0

    for start_y in range(h):
        for start_x in range(w):
            if binary[start_y, start_x] == 0 or labels[start_y, start_x] != 0:
                continue
            current_label += 1
            queue = [(start_y, start_x)]
            labels[start_y, start_x] = current_label
            while queue:
                cy, cx = queue.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary[ny, nx] != 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            queue.append((ny, nx))

    return labels, current_label


def region_info(labels: np.ndarray, label_id: int) -> RegionInfo:
    """
    Compute RegionInfo for a single labelled component.

    Parameters
    ----------
    labels   : int32 label array from :func:`label_mask`
    label_id : which label to describe
    """
    ys, xs = np.where(labels == label_id)
    area = int(len(ys))
    if area == 0:
        return RegionInfo(
            label=label_id,
            area=0,
            bbox=(0, 0, 0, 0),
            centroid=np.zeros(2),
            aspect_ratio=1.0,
        )
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    long_side = max(h, w)
    short_side = max(min(h, w), 1)
    return RegionInfo(
        label=label_id,
        area=area,
        bbox=(y0, x0, y1, x1),
        centroid=np.array([float(ys.mean()), float(xs.mean())]),
        aspect_ratio=float(long_side) / float(short_side),
    )


def all_regions(labels: np.ndarray, n_labels: int) -> List[RegionInfo]:
    """Return a list of :class:`RegionInfo` for all ``n_labels`` regions."""
    return [region_info(labels, i) for i in range(1, n_labels + 1)]


def filter_regions(
    regions: List[RegionInfo],
    cfg: Optional[SegmentConfig] = None,
) -> List[RegionInfo]:
    """
    Drop regions that are too small or too elongated.

    Parameters
    ----------
    regions : list of RegionInfo
    cfg     : SegmentConfig (optional)

    Returns
    -------
    Filtered list (subset of *regions*).
    """
    if cfg is None:
        cfg = SegmentConfig()
    return [r for r in regions
            if r.area >= cfg.min_area and r.aspect_ratio <= cfg.max_aspect_ratio]


def largest_region(regions: List[RegionInfo]) -> Optional[RegionInfo]:
    """Return the region with the largest area, or None if *regions* is empty."""
    if not regions:
        return None
    return max(regions, key=lambda r: r.area)


def mask_from_labels(
    labels: np.ndarray,
    keep_ids: List[int],
) -> np.ndarray:
    """
    Build a uint8 binary mask (255 = foreground) from selected label IDs.

    Parameters
    ----------
    labels   : int32 label array
    keep_ids : list of label IDs to include as foreground

    Returns
    -------
    ndarray uint8, shape (H, W)
    """
    keep_set = set(keep_ids)
    mask = np.zeros(labels.shape, dtype=np.uint8)
    for lid in keep_set:
        mask[labels == lid] = 255
    return mask


def mask_statistics(mask: np.ndarray) -> dict:
    """
    Compute basic statistics of a binary mask.

    Returns a dict with keys:
        ``foreground_pixels``, ``background_pixels``,
        ``foreground_fraction``, ``total_pixels``.
    """
    mask = np.asarray(mask)
    total = int(mask.size)
    fg = int(np.count_nonzero(mask))
    bg = total - fg
    return {
        "foreground_pixels": fg,
        "background_pixels": bg,
        "foreground_fraction": float(fg) / float(total) if total > 0 else 0.0,
        "total_pixels": total,
    }


def mask_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return the tight bounding box of all foreground pixels as
    ``(y0, x0, y1, x1)``, or ``None`` if the mask is all-zero.
    """
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract the boundary pixels of the foreground region via morphological
    erosion (pure numpy, no OpenCV).

    Returns
    -------
    ndarray uint8, shape (H, W) — 255 at boundary pixels, 0 elsewhere.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    fg = mask > 0
    # Erode by checking 4-connectivity
    h, w = fg.shape
    eroded = fg.copy()
    eroded[0, :] = False
    eroded[-1, :] = False
    eroded[:, 0] = False
    eroded[:, -1] = False
    # A pixel survives erosion only if all 4 neighbours are also foreground
    interior = (fg[1:-1, 1:-1] &
                fg[0:-2, 1:-1] & fg[2:, 1:-1] &
                fg[1:-1, 0:-2] & fg[1:-1, 2:])
    eroded[1:-1, 1:-1] = interior
    boundary = fg & ~eroded
    return boundary.astype(np.uint8) * 255
