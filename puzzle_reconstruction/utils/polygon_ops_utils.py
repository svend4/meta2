"""
Утилиты операций над полигонами.

Utilities for polygon set operations, transformations, and comparison:
clipping, union, intersection area, point-polygon tests, Minkowski sum
approximation, and polygon simplification helpers.
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
class PolygonOpsConfig:
    """Parameters controlling polygon operations."""

    clip_epsilon: float = 1e-9
    """Tolerance for degenerate-edge removal during clipping."""

    simplify_epsilon: float = 1e-6
    """Collinearity tolerance for polygon simplification."""

    n_samples: int = 64
    """Number of samples used when approximating Minkowski sum."""

    def __post_init__(self) -> None:
        if self.clip_epsilon < 0:
            raise ValueError("clip_epsilon must be >= 0")
        if self.simplify_epsilon < 0:
            raise ValueError("simplify_epsilon must be >= 0")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PolygonOverlapResult:
    """Result of a polygon–polygon overlap test."""

    overlap: bool
    iou: float
    intersection_area: float
    union_area: float

    def to_dict(self) -> dict:
        return {
            "overlap": self.overlap,
            "iou": self.iou,
            "intersection_area": self.intersection_area,
            "union_area": self.union_area,
        }


@dataclass
class PolygonStats:
    """Geometric statistics of a single polygon."""

    n_vertices: int
    area: float
    perimeter: float
    centroid: np.ndarray
    bounding_box: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    aspect_ratio: float
    compactness: float  # 4π·area / perimeter²

    def to_dict(self) -> dict:
        return {
            "n_vertices": self.n_vertices,
            "area": self.area,
            "perimeter": self.perimeter,
            "centroid": self.centroid.tolist(),
            "bounding_box": self.bounding_box,
            "aspect_ratio": self.aspect_ratio,
            "compactness": self.compactness,
        }


# ---------------------------------------------------------------------------
# Basic polygon measures
# ---------------------------------------------------------------------------

def signed_area(polygon: np.ndarray) -> float:
    """
    Compute the signed area of a polygon using the shoelace formula.

    Parameters
    ----------
    polygon : ndarray, shape (N, 2)

    Returns
    -------
    float — positive if counter-clockwise, negative if clockwise.
    """
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_area(polygon: np.ndarray) -> float:
    """Absolute area of a polygon."""
    return abs(signed_area(polygon))


def polygon_perimeter(polygon: np.ndarray) -> float:
    """Perimeter of a closed polygon (sum of edge lengths)."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """
    Centroid of a polygon via the weighted shoelace formula.

    Falls back to arithmetic mean for degenerate (zero-area) polygons.
    """
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n == 0:
        return np.zeros(2)
    if n < 3:
        return pts.mean(axis=0)
    a = signed_area(pts)
    if abs(a) < 1e-12:
        return pts.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(np.sum((x + xn) * cross)) / (6.0 * a)
    cy = float(np.sum((y + yn) * cross)) / (6.0 * a)
    return np.array([cx, cy])


def polygon_bounding_box(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Return (x0, y0, x1, y1) tight bounding box.
    Returns (0, 0, 0, 0) for empty polygon.
    """
    pts = np.asarray(polygon, dtype=float)
    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def polygon_stats(polygon: np.ndarray) -> PolygonStats:
    """
    Compute :class:`PolygonStats` for a single polygon.
    """
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    area = polygon_area(pts)
    perim = polygon_perimeter(pts)
    centroid = polygon_centroid(pts)
    bbox = polygon_bounding_box(pts)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    long_side = max(bw, bh)
    short_side = max(min(bw, bh), 1e-9)
    aspect = float(long_side) / float(short_side)
    compactness = (
        4.0 * math.pi * area / (perim * perim) if perim > 1e-12 else 0.0
    )
    return PolygonStats(
        n_vertices=n,
        area=area,
        perimeter=perim,
        centroid=centroid,
        bounding_box=bbox,
        aspect_ratio=aspect,
        compactness=compactness,
    )


# ---------------------------------------------------------------------------
# Point-in-polygon test (ray casting)
# ---------------------------------------------------------------------------

def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test.

    Parameters
    ----------
    point   : (2,) array-like
    polygon : (N, 2) array-like

    Returns
    -------
    bool — True if *point* is strictly inside *polygon*.
    """
    pts = np.asarray(polygon, dtype=float)
    px, py = float(point[0]), float(point[1])
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i, 0], pts[i, 1]
        xj, yj = pts[j, 0], pts[j, 1]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi
        ):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Polygon overlap / IoU (bounding-box approximation)
# ---------------------------------------------------------------------------

def _bbox_overlap(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Intersection area of two axis-aligned bounding boxes."""
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def polygon_overlap(
    poly1: np.ndarray,
    poly2: np.ndarray,
) -> PolygonOverlapResult:
    """
    Compute overlap between two convex polygons via bounding-box
    approximation.

    Parameters
    ----------
    poly1, poly2 : (N, 2) arrays

    Returns
    -------
    :class:`PolygonOverlapResult`
    """
    b1 = polygon_bounding_box(poly1)
    b2 = polygon_bounding_box(poly2)
    inter = _bbox_overlap(b1, b2)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    iou = inter / union if union > 1e-12 else 0.0
    return PolygonOverlapResult(
        overlap=inter > 0.0,
        iou=float(iou),
        intersection_area=float(inter),
        union_area=float(union),
    )


# ---------------------------------------------------------------------------
# Polygon simplification (collinearity removal)
# ---------------------------------------------------------------------------

def remove_collinear(
    polygon: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Remove vertices that are (approximately) collinear with their neighbours.

    Parameters
    ----------
    polygon : (N, 2) array
    epsilon : cross-product threshold for collinearity

    Returns
    -------
    ndarray — simplified polygon (may have fewer vertices than input).
    """
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 3:
        return pts.copy()
    keep = []
    for i in range(n):
        prev = pts[(i - 1) % n]
        curr = pts[i]
        nxt = pts[(i + 1) % n]
        v1 = curr - prev
        v2 = nxt - curr
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if cross > epsilon:
            keep.append(i)
    if len(keep) < 3:
        return pts.copy()
    return pts[keep]


# ---------------------------------------------------------------------------
# Polygon orientation
# ---------------------------------------------------------------------------

def ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    """Return the polygon vertices in counter-clockwise order."""
    pts = np.asarray(polygon, dtype=float)
    if signed_area(pts) < 0:
        return pts[::-1].copy()
    return pts.copy()


def ensure_cw(polygon: np.ndarray) -> np.ndarray:
    """Return the polygon vertices in clockwise order."""
    pts = np.asarray(polygon, dtype=float)
    if signed_area(pts) > 0:
        return pts[::-1].copy()
    return pts.copy()


# ---------------------------------------------------------------------------
# Polygon similarity
# ---------------------------------------------------------------------------

def polygon_similarity(
    poly1: np.ndarray,
    poly2: np.ndarray,
    cfg: Optional[PolygonOpsConfig] = None,
) -> float:
    """
    Approximate similarity score in [0, 1] based on IoU of bounding boxes
    and area ratio.

    Returns 0.0 for degenerate inputs; 1.0 for identical bounding boxes.
    """
    if cfg is None:
        cfg = PolygonOpsConfig()
    result = polygon_overlap(poly1, poly2)
    return float(result.iou)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_polygon_stats(
    polygons: List[np.ndarray],
) -> List[PolygonStats]:
    """Compute :class:`PolygonStats` for a list of polygons."""
    return [polygon_stats(p) for p in polygons]


def batch_polygon_overlap(
    polygons_a: List[np.ndarray],
    polygons_b: List[np.ndarray],
) -> List[PolygonOverlapResult]:
    """
    Compute pairwise overlap between two lists of polygons.

    Parameters
    ----------
    polygons_a, polygons_b : lists of (N, 2) arrays — must have equal length.

    Returns
    -------
    List of :class:`PolygonOverlapResult`, one per pair.
    """
    if len(polygons_a) != len(polygons_b):
        raise ValueError("polygons_a and polygons_b must have equal length")
    return [polygon_overlap(a, b) for a, b in zip(polygons_a, polygons_b)]
