"""
Rotation-aware DTW (R-DTW) for matching edge curves.

Extends basic DTW by searching over discrete rotations of the reference
curve before computing the alignment distance.  Two torn-paper edges may
need to be compared after one is rotated/reflected, so the best alignment
angle is found by grid-search.

Algorithm:
    1. Resample both curves to the same length.
    2. For each candidate rotation angle θ in a grid:
         a. Rotate curve B by θ around its centroid.
         b. Compute DTW distance between A and rotated B.
    3. Return the minimum distance and the best angle.

Additionally provides a mirror-aware variant (checks rotation + reflection).
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple

from .dtw import dtw_distance


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RotationDTWResult(NamedTuple):
    """Result of rotation-aware DTW matching."""
    distance: float        # minimum DTW distance found
    best_angle_deg: float  # rotation angle (degrees) that minimises distance
    mirrored: bool         # whether the best match used a mirror-flip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_curve(curve: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a (K, 2) curve to exactly n equally-spaced points using linear
    interpolation along the arc length.

    Args:
        curve: (K, 2) array of 2-D points.
        n:     Target number of points.

    Returns:
        (n, 2) resampled curve.
    """
    if len(curve) < 2:
        return np.zeros((n, 2))
    diffs  = np.diff(curve, axis=0)
    segs   = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(segs)])
    total  = cumlen[-1]
    if total == 0:
        return np.tile(curve[0], (n, 1))
    t_new = np.linspace(0.0, total, n)
    x_new = np.interp(t_new, cumlen, curve[:, 0])
    y_new = np.interp(t_new, cumlen, curve[:, 1])
    return np.column_stack([x_new, y_new])


def _rotate_curve(curve: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a (N, 2) curve by *angle_deg* around its centroid.

    Args:
        curve:     (N, 2) array.
        angle_deg: Rotation angle in degrees.

    Returns:
        (N, 2) rotated curve.
    """
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    centroid = curve.mean(axis=0)
    shifted  = curve - centroid
    rotated  = shifted @ R.T
    return rotated + centroid


def _mirror_curve(curve: np.ndarray) -> np.ndarray:
    """
    Mirror curve horizontally (flip x-coordinates around centroid).

    Args:
        curve: (N, 2) array.

    Returns:
        (N, 2) mirrored curve.
    """
    cx = curve[:, 0].mean()
    mirrored = curve.copy()
    mirrored[:, 0] = 2 * cx - curve[:, 0]
    return mirrored


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def rotation_dtw(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    n_angles: int = 36,
    n_points: int = 64,
    dtw_window: int = 10,
    check_mirror: bool = False,
) -> RotationDTWResult:
    """
    Rotation-aware DTW: find best rotation of curve_b that minimises DTW
    distance to curve_a.

    Args:
        curve_a:      (N, 2) reference curve.
        curve_b:      (M, 2) query curve (will be rotated).
        n_angles:     Number of rotation angles to try (evenly in [0°, 360°)).
        n_points:     Number of points to resample both curves to.
        dtw_window:   Sakoe-Chiba band width for DTW.
        check_mirror: If True, also check all rotations of the mirrored B.

    Returns:
        RotationDTWResult with (distance, best_angle_deg, mirrored).
    """
    if len(curve_a) < 2 or len(curve_b) < 2:
        return RotationDTWResult(distance=float("inf"), best_angle_deg=0.0, mirrored=False)

    a = _resample_curve(curve_a, n_points)
    b = _resample_curve(curve_b, n_points)

    angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)

    best_dist  = float("inf")
    best_angle = 0.0
    best_mirror = False

    for theta in angles:
        b_rot = _rotate_curve(b, theta)
        dist  = dtw_distance(a, b_rot, window=dtw_window)
        if dist < best_dist:
            best_dist  = dist
            best_angle = float(theta)
            best_mirror = False

    if check_mirror:
        b_mir = _mirror_curve(b)
        for theta in angles:
            b_rot = _rotate_curve(b_mir, theta)
            dist  = dtw_distance(a, b_rot, window=dtw_window)
            if dist < best_dist:
                best_dist   = dist
                best_angle  = float(theta)
                best_mirror = True

    return RotationDTWResult(
        distance=best_dist,
        best_angle_deg=best_angle,
        mirrored=best_mirror,
    )


def rotation_dtw_similarity(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    n_angles: int = 36,
    n_points: int = 64,
    dtw_window: int = 10,
    check_mirror: bool = True,
) -> float:
    """
    Convert rotation-aware DTW distance to a similarity score in [0, 1].

    score = exp(-distance)  so distance=0 → 1.0, distance→∞ → 0.0.

    Args:
        curve_a, curve_b:  (N, 2) / (M, 2) edge curves.
        n_angles:          Grid size for rotation search.
        n_points:          Resampling resolution.
        dtw_window:        Sakoe-Chiba band width.
        check_mirror:      Whether to also check mirrored orientations.

    Returns:
        Similarity ∈ [0, 1].
    """
    result = rotation_dtw(
        curve_a, curve_b,
        n_angles=n_angles,
        n_points=n_points,
        dtw_window=dtw_window,
        check_mirror=check_mirror,
    )
    if result.distance == float("inf"):
        return 0.0
    return float(np.exp(-result.distance))


def batch_rotation_dtw(
    query: np.ndarray,
    candidates: list[np.ndarray],
    n_angles: int = 36,
    n_points: int = 64,
    dtw_window: int = 10,
    check_mirror: bool = True,
) -> list[RotationDTWResult]:
    """
    Compute rotation-aware DTW between one query curve and multiple candidates.

    Args:
        query:       (N, 2) query curve.
        candidates:  List of (M_i, 2) candidate curves.
        n_angles:    Grid size for rotation search.
        n_points:    Resampling resolution.
        dtw_window:  Sakoe-Chiba band width.
        check_mirror: Whether to check mirrored variants.

    Returns:
        List of RotationDTWResult, one per candidate.
    """
    return [
        rotation_dtw(
            query, cand,
            n_angles=n_angles,
            n_points=n_points,
            dtw_window=dtw_window,
            check_mirror=check_mirror,
        )
        for cand in candidates
    ]
