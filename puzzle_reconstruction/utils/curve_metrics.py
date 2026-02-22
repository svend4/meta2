"""
Метрики сравнения кривых/контуров.

Lightweight utilities for computing geometric similarity between
parametric curves (edge contours) used throughout the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CurveMetricConfig:
    """Parameters shared across curve comparison metrics."""

    n_samples: int = 64
    """Number of points to resample curves to before comparison."""

    eps: float = 1e-10
    """Small constant to prevent division by zero."""

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(pts: np.ndarray, n: int, eps: float = 1e-10) -> np.ndarray:
    """Arc-length resample *pts* to exactly *n* points."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((n, 2))
    if len(pts) == 1:
        return np.tile(pts[0], (n, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total < eps:
        return np.tile(pts[0], (n, 1))
    target = np.linspace(0.0, total, n)
    x_r = np.interp(target, cumlen, pts[:, 0])
    y_r = np.interp(target, cumlen, pts[:, 1])
    return np.stack([x_r, y_r], axis=1)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def curve_l2(
    a: np.ndarray,
    b: np.ndarray,
    cfg: Optional[CurveMetricConfig] = None,
) -> float:
    """
    Mean L2 (Euclidean) distance between corresponding points after
    arc-length resampling.

    Returns
    -------
    float  — mean distance in the same units as the input coordinates.
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    ar = _resample(np.asarray(a, dtype=np.float64), cfg.n_samples, cfg.eps)
    br = _resample(np.asarray(b, dtype=np.float64), cfg.n_samples, cfg.eps)
    return float(np.linalg.norm(ar - br, axis=1).mean())


def curve_l2_mirror(
    a: np.ndarray,
    b: np.ndarray,
    cfg: Optional[CurveMetricConfig] = None,
) -> float:
    """
    Minimum of ``curve_l2(a, b)`` and ``curve_l2(a, b[::-1])``.
    Handles reversed-orientation pairs (complementary edges).
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    b_arr = np.asarray(b, dtype=np.float64)
    return min(curve_l2(a, b_arr, cfg), curve_l2(a, b_arr[::-1], cfg))


def hausdorff_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: Optional[CurveMetricConfig] = None,
) -> float:
    """
    Symmetric Hausdorff distance between two curves after arc-length
    resampling:  max(directed_hd(a→b), directed_hd(b→a)).
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    ar = _resample(np.asarray(a, dtype=np.float64), cfg.n_samples, cfg.eps)
    br = _resample(np.asarray(b, dtype=np.float64), cfg.n_samples, cfg.eps)
    # ||ar - br||  via broadcasting  (n, 1, 2) - (1, n, 2)
    diff = ar[:, None, :] - br[None, :, :]          # (n, n, 2)
    dists = np.linalg.norm(diff, axis=2)             # (n, n)
    hd_ab = float(dists.min(axis=1).max())
    hd_ba = float(dists.min(axis=0).max())
    return max(hd_ab, hd_ba)


def frechet_distance_approx(
    a: np.ndarray,
    b: np.ndarray,
    cfg: Optional[CurveMetricConfig] = None,
) -> float:
    """
    Discrete approximation of the Fréchet distance via dynamic programming.

    This O(n²) DP-based implementation operates on resampled curves.
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    n = cfg.n_samples
    ar = _resample(np.asarray(a, dtype=np.float64), n, cfg.eps)
    br = _resample(np.asarray(b, dtype=np.float64), n, cfg.eps)
    # Pre-compute pairwise distances
    diff = ar[:, None, :] - br[None, :, :]           # (n, n, 2)
    d = np.linalg.norm(diff, axis=2)                 # (n, n)
    # DP
    dp = np.full((n, n), np.inf)
    dp[0, 0] = d[0, 0]
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], d[i, 0])
    for j in range(1, n):
        dp[0, j] = max(dp[0, j - 1], d[0, j])
    for i in range(1, n):
        for j in range(1, n):
            dp[i, j] = max(min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                           d[i, j])
    return float(dp[n - 1, n - 1])


def curve_length(pts: np.ndarray) -> float:
    """
    Arc length of a polyline (sum of segment lengths).

    Parameters
    ----------
    pts : array-like, shape (N, 2)

    Returns
    -------
    float  — total arc length.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def length_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """
    Ratio min(len_a, len_b) / max(len_a, len_b) ∈ [0, 1].
    Returns 0.0 when both lengths are 0.
    """
    la = curve_length(np.asarray(a, dtype=np.float64))
    lb = curve_length(np.asarray(b, dtype=np.float64))
    denom = max(la, lb)
    if denom < 1e-12:
        return 0.0
    return float(min(la, lb) / denom)


# ---------------------------------------------------------------------------
# Composite / scoring
# ---------------------------------------------------------------------------

@dataclass
class CurveComparisonResult:
    """Bundle of curve comparison metrics."""

    l2: float
    hausdorff: float
    frechet: float
    length_ratio: float

    def similarity(self, sigma: float = 1.0) -> float:
        """
        Combine metrics into a single similarity score ∈ [0, 1].

        score = length_ratio * exp(-mean_distance / sigma)
        """
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        mean_dist = (self.l2 + self.hausdorff) / 2.0
        return float(self.length_ratio * math.exp(-mean_dist / sigma))

    def to_dict(self) -> dict:
        return {
            "l2": self.l2,
            "hausdorff": self.hausdorff,
            "frechet": self.frechet,
            "length_ratio": self.length_ratio,
        }


def compare_curves(
    a: np.ndarray,
    b: np.ndarray,
    cfg: Optional[CurveMetricConfig] = None,
) -> CurveComparisonResult:
    """
    Compute all four curve metrics between *a* and *b*.

    Parameters
    ----------
    a, b : array-like, shape (N, 2)
    cfg  : CurveMetricConfig (optional)

    Returns
    -------
    CurveComparisonResult
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    return CurveComparisonResult(
        l2=curve_l2(a, b, cfg),
        hausdorff=hausdorff_distance(a, b, cfg),
        frechet=frechet_distance_approx(a, b, cfg),
        length_ratio=length_ratio(a, b),
    )


def batch_compare_curves(
    pairs: list,
    cfg: Optional[CurveMetricConfig] = None,
) -> list:
    """
    Compare a list of ``(a, b)`` curve pairs.

    Parameters
    ----------
    pairs : list of (array-like, array-like)
    cfg   : CurveMetricConfig (optional)

    Returns
    -------
    List of CurveComparisonResult, one per pair.
    """
    if cfg is None:
        cfg = CurveMetricConfig()
    return [compare_curves(a, b, cfg) for a, b in pairs]
