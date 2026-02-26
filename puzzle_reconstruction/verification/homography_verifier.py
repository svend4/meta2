"""
Homography-based verification of fragment placement.

Given two overlapping (or adjacent) fragments with known key-point
correspondences, this module estimates the homography transformation
between them and scores the quality of the fit.

Uses direct linear transform (DLT) for 4-point minimal homography,
and RANSAC-like robust estimation for larger correspondences.

Typical use:
    verifier = HomographyVerifier()
    result = verifier.verify(points_src, points_dst, fragment_size=(h, w))
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HomographyConfig:
    """Configuration for HomographyVerifier."""
    ransac_threshold: float = 5.0   # Reprojection error threshold in pixels
    min_inliers: int = 4            # Minimum inlier count to accept
    max_iterations: int = 100       # RANSAC iterations
    confidence: float = 0.99        # RANSAC confidence level
    min_inlier_ratio: float = 0.5   # Minimum fraction of inliers to accept


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HomographyResult:
    """Result of homography-based fragment verification."""
    H: Optional[np.ndarray]         # (3, 3) homography matrix, or None if failed
    inlier_mask: np.ndarray         # Boolean mask of inliers
    n_inliers: int                  # Number of inliers
    inlier_ratio: float             # Fraction of inliers
    reprojection_error: float       # Mean reprojection error of inliers (pixels)
    is_valid: bool                  # Whether the homography passes quality checks
    score: float                    # Overall score ∈ [0, 1]


# ---------------------------------------------------------------------------
# DLT homography estimation
# ---------------------------------------------------------------------------

def estimate_homography_dlt(
    src: np.ndarray,
    dst: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate homography using the Direct Linear Transform (DLT).

    Requires at least 4 point correspondences.

    Args:
        src: (N, 2) source points.
        dst: (N, 2) destination points.

    Returns:
        (3, 3) normalised homography matrix, or None if estimation fails.
    """
    if len(src) < 4 or len(dst) < 4 or len(src) != len(dst):
        return None

    n = len(src)

    # Normalise points (improves numerical stability)
    src_norm, T_src = _normalise_points(src)
    dst_norm, T_dst = _normalise_points(dst)

    # Build DLT matrix A (2N × 9)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        xs, ys = float(src_norm[i, 0]), float(src_norm[i, 1])
        xd, yd = float(dst_norm[i, 0]), float(dst_norm[i, 1])
        A[2*i]     = [-xs, -ys, -1,  0,   0,  0, xd*xs, xd*ys, xd]
        A[2*i + 1] = [0,   0,   0,  -xs, -ys, -1, yd*xs, yd*ys, yd]

    # SVD solution: h = last row of V^T
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    h = Vt[-1].reshape(3, 3)

    # Denormalise: H = T_dst^-1 @ h @ T_src
    try:
        H = np.linalg.inv(T_dst) @ h @ T_src
    except np.linalg.LinAlgError:
        return None

    # Normalise so H[2,2] = 1
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


def _normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotropic normalisation: centroid at origin, mean distance = sqrt(2).

    Returns:
        (normalised_pts, T) where T is the (3, 3) normalisation matrix.
    """
    centroid = pts.mean(axis=0)
    shifted  = pts - centroid
    dist     = np.linalg.norm(shifted, axis=1)
    mean_dist = dist.mean()
    if mean_dist == 0:
        mean_dist = 1.0
    scale = np.sqrt(2.0) / mean_dist

    T = np.array([
        [scale,  0,    -scale * centroid[0]],
        [0,      scale, -scale * centroid[1]],
        [0,      0,      1],
    ])
    n = len(pts)
    pts_h = np.column_stack([pts, np.ones(n)])
    normalised = (T @ pts_h.T).T[:, :2]
    return normalised, T


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------

def reprojection_error(
    H: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """
    Compute per-point reprojection error.

    Args:
        H:   (3, 3) homography matrix.
        src: (N, 2) source points.
        dst: (N, 2) destination points.

    Returns:
        (N,) array of Euclidean reprojection errors.
    """
    n = len(src)
    src_h  = np.column_stack([src, np.ones(n)])        # (N, 3)
    dst_h  = np.column_stack([dst, np.ones(n)])        # (N, 3)

    # Forward projection: H @ src
    projected_fwd = (H @ src_h.T).T                    # (N, 3)
    w_fwd = projected_fwd[:, 2:3]
    w_fwd = np.where(np.abs(w_fwd) < 1e-12, 1e-12, w_fwd)
    proj_fwd = projected_fwd[:, :2] / w_fwd

    err_fwd = np.linalg.norm(proj_fwd - dst, axis=1)

    # Backward projection: H^-1 @ dst (symmetric error)
    try:
        H_inv = np.linalg.inv(H)
        projected_bwd = (H_inv @ dst_h.T).T
        w_bwd = projected_bwd[:, 2:3]
        w_bwd = np.where(np.abs(w_bwd) < 1e-12, 1e-12, w_bwd)
        proj_bwd = projected_bwd[:, :2] / w_bwd
        err_bwd = np.linalg.norm(proj_bwd - src, axis=1)
        return (err_fwd + err_bwd) / 2.0
    except np.linalg.LinAlgError:
        return err_fwd


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def estimate_homography_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    cfg: Optional[HomographyConfig] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate homography with RANSAC.

    Args:
        src: (N, 2) source points.
        dst: (N, 2) destination points.
        cfg: Configuration.

    Returns:
        (H, inlier_mask) where H is the (3, 3) homography or None,
        and inlier_mask is a boolean (N,) array.
    """
    if cfg is None:
        cfg = HomographyConfig()

    n = len(src)
    if n < 4:
        return None, np.zeros(n, dtype=bool)

    rng = np.random.default_rng(42)
    best_H     = None
    best_mask  = np.zeros(n, dtype=bool)
    best_count = 0

    # Adaptive RANSAC: compute iterations needed
    def _n_iters(inlier_ratio: float, n_sample: int, confidence: float) -> int:
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            return cfg.max_iterations
        return int(np.ceil(np.log(1 - confidence) /
                           np.log(1 - max(inlier_ratio ** n_sample, 1e-15))))

    n_iter = cfg.max_iterations
    for it in range(n_iter):
        idx = rng.choice(n, 4, replace=False)
        H_cand = estimate_homography_dlt(src[idx], dst[idx])
        if H_cand is None:
            continue

        errs  = reprojection_error(H_cand, src, dst)
        mask  = errs < cfg.ransac_threshold
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask  = mask
            best_H     = H_cand

            # Update number of iterations
            ratio = count / n
            n_iter = min(cfg.max_iterations,
                         _n_iters(ratio, 4, cfg.confidence))

    # Refit on all inliers
    if best_H is not None and best_count >= cfg.min_inliers:
        H_refined = estimate_homography_dlt(src[best_mask], dst[best_mask])
        if H_refined is not None:
            errs_r = reprojection_error(H_refined, src, dst)
            best_mask = errs_r < cfg.ransac_threshold
            best_H    = H_refined

    return best_H, best_mask


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def check_homography_quality(
    H: np.ndarray,
    fragment_size: Tuple[int, int],
    max_skew_ratio: float = 10.0,
) -> bool:
    """
    Check if H represents a geometrically plausible transformation.

    Args:
        H:             (3, 3) homography.
        fragment_size: (height, width) of the fragment in pixels.
        max_skew_ratio: Maximum ratio of singular values (det check).

    Returns:
        True if H is plausible.
    """
    # Det must be positive (orientation-preserving)
    if np.linalg.det(H) <= 0:
        return False

    # Singular value ratio (condition number) shouldn't be too large
    try:
        U, s, Vt = np.linalg.svd(H[:2, :2])
        if s[1] == 0:
            return False
        if s[0] / s[1] > max_skew_ratio:
            return False
    except np.linalg.LinAlgError:
        return False

    return True


# ---------------------------------------------------------------------------
# High-level verifier
# ---------------------------------------------------------------------------

class HomographyVerifier:
    """
    High-level homography-based fragment placement verifier.

    Estimates the homography between two sets of corresponding points
    and returns a quality score.
    """

    def __init__(self, cfg: Optional[HomographyConfig] = None) -> None:
        self.cfg = cfg or HomographyConfig()

    def verify(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        fragment_size: Tuple[int, int] = (256, 256),
    ) -> HomographyResult:
        """
        Verify placement by estimating homography from correspondences.

        Args:
            src:           (N, 2) source key-points.
            dst:           (N, 2) destination key-points.
            fragment_size: (height, width) used for plausibility checks.

        Returns:
            HomographyResult.
        """
        n = len(src)

        if n < 4:
            return HomographyResult(
                H=None,
                inlier_mask=np.zeros(n, dtype=bool),
                n_inliers=0,
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                is_valid=False,
                score=0.0,
            )

        H, mask = estimate_homography_ransac(src, dst, self.cfg)

        n_inliers = int(mask.sum())
        inlier_ratio = n_inliers / n if n > 0 else 0.0

        if H is None or n_inliers < self.cfg.min_inliers:
            return HomographyResult(
                H=None,
                inlier_mask=mask,
                n_inliers=n_inliers,
                inlier_ratio=inlier_ratio,
                reprojection_error=float("inf"),
                is_valid=False,
                score=0.0,
            )

        # Mean reprojection error on inliers
        errs = reprojection_error(H, src[mask], dst[mask])
        mean_err = float(errs.mean()) if len(errs) > 0 else float("inf")

        # Plausibility check
        is_valid = (
            check_homography_quality(H, fragment_size) and
            inlier_ratio >= self.cfg.min_inlier_ratio and
            n_inliers >= self.cfg.min_inliers and
            np.isfinite(mean_err)
        )

        # Score: combination of inlier ratio and reprojection accuracy
        if is_valid:
            err_score    = float(np.exp(-mean_err / 10.0))    # 0→1.0, 10px→0.37
            inlier_score = inlier_ratio
            score = float(0.5 * err_score + 0.5 * inlier_score)
        else:
            score = 0.0

        return HomographyResult(
            H=H,
            inlier_mask=mask,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=mean_err,
            is_valid=is_valid,
            score=float(np.clip(score, 0.0, 1.0)),
        )

    def verify_batch(
        self,
        correspondences: List[Tuple[np.ndarray, np.ndarray]],
        fragment_size: Tuple[int, int] = (256, 256),
    ) -> List[HomographyResult]:
        """
        Verify multiple correspondence sets.

        Args:
            correspondences: List of (src, dst) pairs.
            fragment_size:   Fragment size for quality checks.

        Returns:
            List of HomographyResult.
        """
        return [self.verify(src, dst, fragment_size) for src, dst in correspondences]
