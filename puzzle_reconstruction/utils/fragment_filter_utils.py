"""
Утилиты фильтрации и отбора фрагментов.

Fragment filter and selection utilities: quality checks, deduplication,
size and shape filters, and batch filtering helpers used in preprocessing
and assembly pipelines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FragmentFilterConfig:
    """Parameters controlling fragment filtering logic."""

    min_area: float = 0.0
    """Minimum fragment area in pixels.  Fragments below this are dropped."""

    max_area: float = math.inf
    """Maximum fragment area in pixels.  Fragments above this are dropped."""

    min_aspect: float = 0.0
    """Minimum aspect ratio (short / long side).  0 = no lower bound."""

    max_aspect: float = math.inf
    """Maximum aspect ratio (short / long side).  Unused if inf."""

    min_fill_ratio: float = 0.0
    """Minimum fill ratio (mask area / bounding-box area).  0 = disabled."""

    deduplicate: bool = True
    """If True, drop duplicate fragments (by image hash)."""

    def __post_init__(self) -> None:
        if self.min_area < 0:
            raise ValueError("min_area must be >= 0")
        if self.max_area <= 0:
            raise ValueError("max_area must be > 0")
        if self.min_area > self.max_area:
            raise ValueError("min_area must be <= max_area")
        if not (0.0 <= self.min_fill_ratio <= 1.0):
            raise ValueError("min_fill_ratio must be in [0, 1]")


# ---------------------------------------------------------------------------
# Per-fragment quality metrics
# ---------------------------------------------------------------------------

@dataclass
class FragmentQuality:
    """Quality metrics for a single fragment."""

    fragment_id: int
    area: float
    aspect_ratio: float
    fill_ratio: float
    passed: bool
    reject_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Core filter functions
# ---------------------------------------------------------------------------

def compute_fragment_area(mask: np.ndarray) -> float:
    """
    Return the number of non-zero pixels in *mask* as a float.

    Parameters
    ----------
    mask : (H, W) uint8 array.

    Returns
    -------
    float — pixel area.
    """
    return float(np.count_nonzero(mask))


def compute_aspect_ratio(mask: np.ndarray) -> float:
    """
    Compute the aspect ratio of the bounding box of non-zero pixels in *mask*.

    Returns short_side / long_side so the result is always in (0, 1].
    Returns 1.0 for empty or square masks.

    Parameters
    ----------
    mask : (H, W) uint8 array.

    Returns
    -------
    float in (0, 1].
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return 1.0
    h = int(rows.sum())
    w = int(cols.sum())
    long_side = max(h, w)
    short_side = min(h, w)
    if long_side == 0:
        return 1.0
    return short_side / long_side


def compute_fill_ratio(mask: np.ndarray) -> float:
    """
    Compute fill ratio = (mask area) / (bounding-box area).

    Returns 1.0 for empty masks (degenerate case treated as filled).

    Parameters
    ----------
    mask : (H, W) uint8 array.

    Returns
    -------
    float in [0, 1].
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 1.0
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    bbox_area = h * w
    if bbox_area == 0:
        return 1.0
    return float(len(ys)) / bbox_area


def evaluate_fragment(
    fragment_id: int,
    mask: np.ndarray,
    cfg: FragmentFilterConfig,
) -> FragmentQuality:
    """
    Evaluate a single fragment against *cfg* quality criteria.

    Parameters
    ----------
    fragment_id : identifier for the fragment.
    mask        : (H, W) uint8 binary mask.
    cfg         : filter configuration.

    Returns
    -------
    :class:`FragmentQuality` describing whether the fragment passes.
    """
    area = compute_fragment_area(mask)
    aspect = compute_aspect_ratio(mask)
    fill = compute_fill_ratio(mask)

    if area < cfg.min_area:
        return FragmentQuality(fragment_id, area, aspect, fill,
                               passed=False, reject_reason="area_too_small")
    if area > cfg.max_area:
        return FragmentQuality(fragment_id, area, aspect, fill,
                               passed=False, reject_reason="area_too_large")
    if aspect < cfg.min_aspect:
        return FragmentQuality(fragment_id, area, aspect, fill,
                               passed=False, reject_reason="aspect_too_small")
    if not math.isinf(cfg.max_aspect) and aspect > cfg.max_aspect:
        return FragmentQuality(fragment_id, area, aspect, fill,
                               passed=False, reject_reason="aspect_too_large")
    if fill < cfg.min_fill_ratio:
        return FragmentQuality(fragment_id, area, aspect, fill,
                               passed=False, reject_reason="fill_too_low")

    return FragmentQuality(fragment_id, area, aspect, fill, passed=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _image_hash(image: np.ndarray) -> int:
    """Quick 64-bit hash of image bytes using numpy."""
    flat = image.ravel()
    # Use a simple polynomial hash over uint8 values
    h = int(np.packbits(flat[:64] if len(flat) >= 64 else flat).view(np.uint8).sum())
    h ^= flat.shape[0]
    return h


def deduplicate_fragments(
    fragments: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    """
    Remove duplicate fragments based on image content hash.

    Parameters
    ----------
    fragments : list of (fragment_id, image) tuples.

    Returns
    -------
    Deduplicated list preserving first occurrence of each unique image.
    """
    seen: Dict[int, bool] = {}
    result = []
    for fid, img in fragments:
        h = _image_hash(img)
        if h not in seen:
            seen[h] = True
            result.append((fid, img))
    return result


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------

def filter_fragments(
    fragments: List[Tuple[int, np.ndarray, np.ndarray]],
    cfg: Optional[FragmentFilterConfig] = None,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], List[FragmentQuality]]:
    """
    Filter a list of fragments by quality criteria.

    Parameters
    ----------
    fragments : list of (fragment_id, image, mask) triples.
    cfg       : optional :class:`FragmentFilterConfig`.

    Returns
    -------
    (kept, qualities) where:
      - kept     : fragments that passed all criteria.
      - qualities: :class:`FragmentQuality` for every input fragment.
    """
    if cfg is None:
        cfg = FragmentFilterConfig()

    qualities: List[FragmentQuality] = []
    kept: List[Tuple[int, np.ndarray, np.ndarray]] = []

    for fid, img, mask in fragments:
        q = evaluate_fragment(fid, mask, cfg)
        qualities.append(q)
        if q.passed:
            kept.append((fid, img, mask))

    if cfg.deduplicate:
        img_list = [(fid, img) for fid, img, _ in kept]
        deduped_ids = {fid for fid, _ in deduplicate_fragments(img_list)}
        kept = [(fid, img, mask) for fid, img, mask in kept if fid in deduped_ids]

    return kept, qualities


# ---------------------------------------------------------------------------
# Sorting and selection helpers
# ---------------------------------------------------------------------------

def sort_by_area(
    fragments: List[Tuple[int, np.ndarray, np.ndarray]],
    descending: bool = True,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Sort fragments by mask area."""
    return sorted(
        fragments,
        key=lambda t: compute_fragment_area(t[2]),
        reverse=descending,
    )


def top_k_fragments(
    fragments: List[Tuple[int, np.ndarray, np.ndarray]],
    k: int,
    key: Optional[Callable] = None,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Select top-*k* fragments by an optional key function.

    Parameters
    ----------
    fragments : list of (fragment_id, image, mask) triples.
    k         : number of fragments to keep.
    key       : callable taking (fid, img, mask) → float.  Defaults to area.

    Returns
    -------
    Top *k* fragments sorted descending by key.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if key is None:
        key = lambda t: compute_fragment_area(t[2])
    return sorted(fragments, key=key, reverse=True)[:k]


def fragment_quality_summary(qualities: List[FragmentQuality]) -> Dict[str, int]:
    """
    Summarise filter results.

    Returns
    -------
    dict with keys: total, passed, rejected, and each reject_reason count.
    """
    summary: Dict[str, int] = {"total": len(qualities), "passed": 0, "rejected": 0}
    for q in qualities:
        if q.passed:
            summary["passed"] += 1
        else:
            summary["rejected"] += 1
            reason = q.reject_reason or "unknown"
            summary[reason] = summary.get(reason, 0) + 1
    return summary
