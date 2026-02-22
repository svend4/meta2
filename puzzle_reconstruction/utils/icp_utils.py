"""ICP (Iterative Closest Point) utility functions for point-cloud alignment.

Provides low-level building blocks used by the matching pipeline:
nearest-neighbour queries, SVD-based rotation/translation estimation,
RMSE bookkeeping, convergence detection, and lightweight wrappers for
batch-alignment scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ICPConfig:
    """Parameters controlling an ICP run."""
    max_iter: int = 50
    tol: float = 1e-5
    max_dist: Optional[float] = None   # reject correspondences farther than this
    track_history: bool = False
    allow_reflection: bool = False     # if False, force det(R) = +1


# ─── Point-cloud helpers ──────────────────────────────────────────────────────

def centroid(pts: np.ndarray) -> np.ndarray:
    """Return the 2-D centroid of *pts* (shape N×2)."""
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be shape (N, 2)")
    return pts.mean(axis=0)


def center_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the centroid from *pts*.

    Returns
    -------
    centered : np.ndarray
        Zero-mean version of *pts*.
    c : np.ndarray
        The centroid that was subtracted (shape (2,)).
    """
    c = centroid(pts)
    return pts - c, c


def scale_points(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale *pts* so the RMS distance from the centroid equals 1.

    Returns
    -------
    scaled : np.ndarray
    scale  : float   – the divisor applied (> 0, or 1.0 if pts is degenerate)
    """
    centered, _ = center_points(pts)
    rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if rms < 1e-12:
        return pts.copy(), 1.0
    return centered / rms, rms


def resample_uniform(pts: np.ndarray, n: int) -> np.ndarray:
    """Re-sample *pts* to exactly *n* points by uniform linear interpolation.

    Works on an open polyline (no wrap-around).  Useful for normalising
    contour lengths before ICP.
    """
    if len(pts) == 0 or n <= 0:
        return np.empty((0, 2), dtype=float)
    if len(pts) == 1:
        return np.tile(pts[0], (n, 1)).astype(float)

    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-12:
        return np.tile(pts[0], (n, 1)).astype(float)

    t_new = np.linspace(0.0, total, n)
    out = np.empty((n, 2), dtype=float)
    for i, t in enumerate(t_new):
        idx = int(np.searchsorted(cum, t, side="right")) - 1
        idx = min(idx, len(pts) - 2)
        seg = cum[idx + 1] - cum[idx]
        alpha = 0.0 if seg < 1e-12 else (t - cum[idx]) / seg
        out[i] = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]
    return out


# ─── Correspondence finding ───────────────────────────────────────────────────

def nearest_neighbours(src: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find, for each point in *src*, the nearest point in *tgt*.

    Parameters
    ----------
    src, tgt : np.ndarray  shape (M, 2) and (N, 2)

    Returns
    -------
    indices : np.ndarray  shape (M,)  – index into *tgt*
    distances : np.ndarray  shape (M,)  – Euclidean distances
    """
    if len(src) == 0 or len(tgt) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    diff = src[:, None, :] - tgt[None, :, :]   # M × N × 2
    dist2 = (diff ** 2).sum(axis=2)             # M × N
    idx = dist2.argmin(axis=1)                  # M
    dists = np.sqrt(dist2[np.arange(len(src)), idx])
    return idx, dists


def filter_correspondences(
    src: np.ndarray,
    tgt: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    max_dist: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove correspondences whose distance exceeds *max_dist*.

    Returns the filtered (src_pts, tgt_pts) ready for SVD.
    """
    tgt_matched = tgt[indices]
    if max_dist is not None and max_dist > 0:
        mask = distances <= max_dist
        return src[mask], tgt_matched[mask]
    return src.copy(), tgt_matched


# ─── SVD rotation / translation ───────────────────────────────────────────────

def svd_rotation(src_c: np.ndarray, tgt_c: np.ndarray, allow_reflection: bool = False) -> np.ndarray:
    """Compute the optimal 2-D rotation matrix via SVD.

    Both arrays must already be **centred** (zero mean).

    Parameters
    ----------
    src_c, tgt_c : np.ndarray  shape (K, 2)
    allow_reflection : bool
        If False (default) the sign of the last singular vector is flipped
        to ensure det(R) = +1 (proper rotation, no reflection).

    Returns
    -------
    R : np.ndarray  shape (2, 2)
    """
    if len(src_c) == 0:
        return np.eye(2)

    H = src_c.T @ tgt_c          # 2 × 2
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def svd_translation(src_c: np.ndarray, tgt_c: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute the translation that minimises ||R·src_c - tgt_c||²  in a
    centred coordinate frame.

    In practice: t = mean(tgt_c) - R @ mean(src_c).  When both inputs are
    already centred (mean ≈ 0) this returns ~0; the caller is responsible for
    adding the centroid offset.
    """
    return tgt_c.mean(axis=0) - R @ src_c.mean(axis=0)


# ─── RMSE helpers ─────────────────────────────────────────────────────────────

def compute_rmse(src: np.ndarray, tgt: np.ndarray) -> float:
    """Root-mean-squared distance between paired points."""
    if len(src) == 0:
        return 0.0
    diff = src - tgt
    return float(np.sqrt(np.mean((diff ** 2).sum(axis=1))))


def rmse_after_transform(
    src: np.ndarray,
    tgt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """Apply (R, t) to *src* and compute RMSE against *tgt*."""
    transformed = (src @ R.T) + t
    return compute_rmse(transformed, tgt)


# ─── Convergence detection ────────────────────────────────────────────────────

def has_converged(prev_rmse: float, curr_rmse: float, tol: float) -> bool:
    """Return True when the improvement in RMSE falls below *tol*."""
    return abs(prev_rmse - curr_rmse) < tol


# ─── Compose transforms ───────────────────────────────────────────────────────

def compose_transforms(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two rigid transforms: first apply (R1, t1) then (R2, t2).

    Returns (R_out, t_out) such that
    ``x_out = R_out @ x + t_out``.
    """
    R_out = R2 @ R1
    t_out = R2 @ t1 + t2
    return R_out, t_out


def invert_transform(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the inverse rigid transform of (R, t)."""
    R_inv = R.T
    t_inv = -(R_inv @ t)
    return R_inv, t_inv


# ─── Batch helpers ────────────────────────────────────────────────────────────

@dataclass
class PairAlignResult:
    """Alignment result for one (src, tgt) pair."""
    R: np.ndarray
    t: np.ndarray
    rmse: float
    converged: bool
    n_iter: int
    rmse_history: List[float] = field(default_factory=list)


def batch_nearest_neighbours(
    clouds: List[np.ndarray],
    reference: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Apply :func:`nearest_neighbours` from each cloud to *reference*.

    Returns a list of (indices, distances) tuples, one per input cloud.
    """
    return [nearest_neighbours(cloud, reference) for cloud in clouds]


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform (R, t) to *pts*: ``pts @ R.T + t``."""
    return pts @ R.T + t


def align_to_first(clouds: List[np.ndarray]) -> List[np.ndarray]:
    """Translate each cloud so its centroid coincides with the origin of the
    first cloud.  Only a translational alignment is performed.

    Returns a list of translated clouds (same length as input).
    """
    if not clouds:
        return []
    anchor = centroid(clouds[0])
    result = []
    for cloud in clouds:
        c = centroid(cloud)
        result.append(cloud - c + anchor)
    return result
