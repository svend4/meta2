"""
Утилиты выравнивания и регистрации контуров/кривых.

Provides lightweight ICP-style and Procrustes alignment helpers for
matching edge curves during puzzle-piece compatibility scoring.
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
class AlignmentConfig:
    """Parameters controlling curve alignment behaviour."""

    n_samples: int = 64
    """Number of points to resample both curves to before alignment."""

    max_icp_iter: int = 50
    """Maximum ICP iterations."""

    icp_tol: float = 1e-6
    """Convergence tolerance for ICP (change in mean-squared error)."""

    allow_reflection: bool = False
    """If True, Procrustes alignment may include a reflection."""

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if self.max_icp_iter < 1:
            raise ValueError("max_icp_iter must be >= 1")
        if self.icp_tol <= 0.0:
            raise ValueError("icp_tol must be > 0")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Output of a curve alignment operation."""

    rotation: float
    """Rotation angle (radians) applied to the source curve."""

    translation: np.ndarray
    """2-D translation vector applied after rotation."""

    scale: float
    """Isotropic scale factor (Procrustes) or 1.0 (ICP)."""

    error: float
    """Mean squared distance between aligned source and target."""

    aligned: np.ndarray
    """Source curve after applying (scale, rotation, translation)."""

    converged: bool = True
    """Whether the iterative method converged within max_icp_iter."""

    def to_dict(self) -> dict:
        return {
            "rotation": float(self.rotation),
            "translation": self.translation.tolist(),
            "scale": float(self.scale),
            "error": float(self.error),
            "converged": self.converged,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample *pts* to *n* equally-spaced points by arc-length."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) == 1 else np.zeros(2), (n, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total == 0.0:
        return np.tile(pts[0], (n, 1))
    target = np.linspace(0.0, total, n)
    x_r = np.interp(target, cumlen, pts[:, 0])
    y_r = np.interp(target, cumlen, pts[:, 1])
    return np.stack([x_r, y_r], axis=1)


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _svd_rotation(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return rotation R (2×2) that minimises ||R A^T - B^T||_F
    using SVD (Kabsch algorithm).  Also returns the angle in radians.
    """
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    angle = math.atan2(R[1, 0], R[0, 0])
    return R, angle


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def normalize_for_alignment(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Translate *pts* to zero mean and scale so RMS distance = 1.

    Returns
    -------
    normalized : ndarray  shape (N, 2)
    centroid   : ndarray  shape (2,)
    scale      : float    (original RMS; multiply to undo)
    """
    pts = np.asarray(pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = float(np.sqrt((centered ** 2).sum() / len(pts)))
    if scale == 0.0:
        scale = 1.0
    return centered / scale, centroid, scale


def find_best_rotation(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Find the rotation (radians) that best aligns *source* onto *target*
    (both assumed zero-mean, same number of points).

    Returns ``(angle, R)`` where *R* is the 2×2 rotation matrix.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    R, angle = _svd_rotation(source, target)
    return angle, R


def find_best_translation(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Return the translation vector that minimises mean squared error
    after applying it to *source* (assumes rotation already applied).
    """
    return target.mean(axis=0) - source.mean(axis=0)


def compute_alignment_error(source: np.ndarray, target: np.ndarray) -> float:
    """
    Mean squared distance between corresponding points of *source* and *target*.
    Both arrays must have the same shape (N, 2).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) == 0:
        return float("inf")
    diff = source - target
    return float((diff ** 2).sum(axis=1).mean())


def align_curves_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    cfg: Optional[AlignmentConfig] = None,
) -> AlignmentResult:
    """
    Align *source* onto *target* using ordinary Procrustes analysis
    (optimal rotation + isotropic scale + translation, no iteration needed).

    Parameters
    ----------
    source, target : array-like, shape (N, 2)
    cfg            : AlignmentConfig (optional)

    Returns
    -------
    AlignmentResult
    """
    if cfg is None:
        cfg = AlignmentConfig()
    src = _resample(np.asarray(source, dtype=np.float64), cfg.n_samples)
    tgt = _resample(np.asarray(target, dtype=np.float64), cfg.n_samples)

    src_n, src_c, src_s = normalize_for_alignment(src)
    tgt_n, tgt_c, tgt_s = normalize_for_alignment(tgt)

    R, angle = _svd_rotation(src_n, tgt_n)

    if cfg.allow_reflection is False and np.linalg.det(R) < 0:
        # Force proper rotation
        U, _, Vt = np.linalg.svd(src_n.T @ tgt_n)
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        angle = math.atan2(R[1, 0], R[0, 0])

    scale = tgt_s / src_s
    aligned_n = src_n @ R.T
    aligned = aligned_n * tgt_s + tgt_c
    translation = tgt_c - (src_c @ R.T) * scale
    error = compute_alignment_error(aligned, tgt)

    return AlignmentResult(
        rotation=angle,
        translation=translation,
        scale=scale,
        error=error,
        aligned=aligned,
        converged=True,
    )


def align_curves_icp(
    source: np.ndarray,
    target: np.ndarray,
    cfg: Optional[AlignmentConfig] = None,
) -> AlignmentResult:
    """
    Align *source* onto *target* using a simplified ICP (Iterative Closest
    Point) that assumes point-to-point correspondence (no nearest-neighbour
    search) and estimates rotation + translation each step.

    Parameters
    ----------
    source, target : array-like, shape (N, 2)
    cfg            : AlignmentConfig (optional)

    Returns
    -------
    AlignmentResult
    """
    if cfg is None:
        cfg = AlignmentConfig()
    src = _resample(np.asarray(source, dtype=np.float64), cfg.n_samples)
    tgt = _resample(np.asarray(target, dtype=np.float64), cfg.n_samples)

    current = src.copy()
    total_angle = 0.0
    total_t = np.zeros(2)
    prev_err = float("inf")
    converged = False

    for _ in range(cfg.max_icp_iter):
        src_c = current.mean(axis=0)
        tgt_c = tgt.mean(axis=0)
        R, dangle = _svd_rotation(current - src_c, tgt - tgt_c)
        dt = tgt_c - (current - src_c).mean(axis=0) @ R.T - src_c
        current = (current - src_c) @ R.T + tgt_c
        total_angle += dangle
        total_t += dt
        err = compute_alignment_error(current, tgt)
        if abs(prev_err - err) < cfg.icp_tol:
            converged = True
            break
        prev_err = err

    return AlignmentResult(
        rotation=total_angle,
        translation=total_t,
        scale=1.0,
        error=compute_alignment_error(current, tgt),
        aligned=current,
        converged=converged,
    )


def alignment_score(result: AlignmentResult, sigma: float = 1.0) -> float:
    """
    Convert an AlignmentResult's *error* to a similarity score in [0, 1].

    score = exp(-error / sigma)
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return float(math.exp(-result.error / sigma))


def batch_align_curves(
    sources: List[np.ndarray],
    targets: List[np.ndarray],
    method: str = "procrustes",
    cfg: Optional[AlignmentConfig] = None,
) -> List[AlignmentResult]:
    """
    Align each ``sources[i]`` onto ``targets[i]``.

    Parameters
    ----------
    sources, targets : lists of (N_i, 2) arrays (need not have equal lengths)
    method           : ``"procrustes"`` or ``"icp"``
    cfg              : AlignmentConfig (optional; shared across all pairs)

    Returns
    -------
    List of AlignmentResult, one per pair.
    """
    if len(sources) != len(targets):
        raise ValueError("sources and targets must have the same length")
    if method not in ("procrustes", "icp"):
        raise ValueError("method must be 'procrustes' or 'icp'")
    align_fn = align_curves_procrustes if method == "procrustes" else align_curves_icp
    return [align_fn(s, t, cfg) for s, t in zip(sources, targets)]
