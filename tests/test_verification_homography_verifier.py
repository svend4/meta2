"""Tests for puzzle_reconstruction/verification/homography_verifier.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.homography_verifier import (
    HomographyConfig,
    HomographyResult,
    HomographyVerifier,
    estimate_homography_dlt,
    estimate_homography_ransac,
    reprojection_error,
    check_homography_quality,
    _normalise_points,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_points_and_H(n: int = 20, seed: int = 42) -> tuple:
    """
    Create a valid planar scene: generate src points, apply a homography,
    and return (src, dst, H_true).
    """
    rng = np.random.default_rng(seed)
    src = rng.uniform(10, 200, (n, 2))
    # Simple translation + slight rotation (stays well-conditioned)
    theta = 0.05  # radians
    c, s  = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 30.0], [s,  c, 15.0], [0, 0, 1.0]])
    src_h = np.column_stack([src, np.ones(n)])
    dst_h = (R @ src_h.T).T
    dst   = dst_h[:, :2] / dst_h[:, 2:3]
    return src, dst, R


def _identity_H() -> np.ndarray:
    return np.eye(3)


def _scale_H(sx: float, sy: float) -> np.ndarray:
    H = np.diag([sx, sy, 1.0])
    return H


# ── HomographyConfig ──────────────────────────────────────────────────────────

class TestHomographyConfig:

    def test_defaults(self):
        cfg = HomographyConfig()
        assert cfg.ransac_threshold > 0
        assert cfg.min_inliers >= 4
        assert 0 < cfg.confidence < 1

    def test_custom(self):
        cfg = HomographyConfig(ransac_threshold=2.0, min_inliers=6)
        assert cfg.ransac_threshold == 2.0
        assert cfg.min_inliers == 6


# ── HomographyResult ──────────────────────────────────────────────────────────

class TestHomographyResult:

    def test_fields(self):
        r = HomographyResult(
            H=np.eye(3),
            inlier_mask=np.array([True, True]),
            n_inliers=2,
            inlier_ratio=1.0,
            reprojection_error=0.5,
            is_valid=True,
            score=0.9,
        )
        assert r.n_inliers == 2
        assert r.is_valid is True


# ── _normalise_points ─────────────────────────────────────────────────────────

class TestNormalisePoints:

    def test_centroid_at_origin(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        norm, T = _normalise_points(pts)
        np.testing.assert_allclose(norm.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_T_is_3x3(self):
        pts = np.random.default_rng(0).uniform(0, 100, (10, 2))
        _, T = _normalise_points(pts)
        assert T.shape == (3, 3)

    def test_no_crash_constant_points(self):
        pts = np.tile([5.0, 5.0], (5, 1))
        norm, T = _normalise_points(pts)
        assert norm.shape == (5, 2)


# ── estimate_homography_dlt ───────────────────────────────────────────────────

class TestEstimateHomographyDLT:

    def test_returns_3x3(self):
        src, dst, H_true = _make_points_and_H(4)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_h_22_is_one(self):
        src, dst, _ = _make_points_and_H(4)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert abs(H[2, 2] - 1.0) < 1e-9

    def test_exact_4_points(self):
        """With exactly 4 noise-free points DLT should recover H exactly."""
        src, dst, H_true = _make_points_and_H(4, seed=1)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        # Check reprojection error is near zero
        errs = reprojection_error(H, src, dst)
        assert float(errs.max()) < 1e-6

    def test_too_few_points(self):
        src, dst, _ = _make_points_and_H(3)
        H = estimate_homography_dlt(src[:3], dst[:3])
        assert H is None

    def test_returns_none_mismatched_lengths(self):
        src, dst, _ = _make_points_and_H(6)
        H = estimate_homography_dlt(src[:5], dst[:4])
        assert H is None

    def test_overdetermined_4plus(self):
        src, dst, _ = _make_points_and_H(10)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert float(errs.mean()) < 1.0  # Noise-free → near-zero


# ── reprojection_error ─────────────────────────────────────────────────────────

class TestReprojectionError:

    def test_identity_zero_error(self):
        pts = np.array([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
        errs = reprojection_error(np.eye(3), pts, pts)
        np.testing.assert_allclose(errs, 0.0, atol=1e-10)

    def test_returns_array_of_correct_length(self):
        src, dst, H = _make_points_and_H(10)
        errs = reprojection_error(H, src, dst)
        assert errs.shape == (10,)

    def test_non_negative(self):
        src, dst, H = _make_points_and_H(10)
        errs = reprojection_error(H, src, dst)
        assert np.all(errs >= 0)

    def test_exact_homography_zero_error(self):
        src, dst, _ = _make_points_and_H(8)
        H = estimate_homography_dlt(src, dst)
        errs = reprojection_error(H, src, dst)
        assert float(errs.max()) < 1e-5

    def test_wrong_H_nonzero_error(self):
        src, dst, _ = _make_points_and_H(8)
        # Use identity instead of correct H → nonzero error
        errs = reprojection_error(np.eye(3), src, dst)
        assert float(errs.mean()) > 1.0


# ── estimate_homography_ransac ────────────────────────────────────────────────

class TestEstimateHomographyRansac:

    def test_clean_data_returns_H(self):
        src, dst, _ = _make_points_and_H(20)
        H, mask = estimate_homography_ransac(src, dst)
        assert H is not None

    def test_inlier_mask_shape(self):
        src, dst, _ = _make_points_and_H(20)
        H, mask = estimate_homography_ransac(src, dst)
        assert mask.shape == (20,)
        assert mask.dtype == bool

    def test_all_inliers_clean(self):
        src, dst, _ = _make_points_and_H(20)
        H, mask = estimate_homography_ransac(src, dst)
        assert mask.sum() >= 16  # At least 80% inliers for clean data

    def test_outliers_rejected(self):
        src, dst, _ = _make_points_and_H(20, seed=5)
        # Add 5 outliers
        src_noisy = src.copy()
        dst_noisy = dst.copy()
        rng = np.random.default_rng(99)
        dst_noisy[:5] += rng.uniform(50, 200, (5, 2))
        H, mask = estimate_homography_ransac(src_noisy, dst_noisy)
        # Outliers should not all be inliers
        n_outlier_inliers = int(mask[:5].sum())
        assert n_outlier_inliers <= 3  # At most 3 of 5 outliers classified as inliers

    def test_too_few_points(self):
        src = np.array([[0., 0.], [1., 0.], [0., 1.]])
        dst = src + 1.0
        H, mask = estimate_homography_ransac(src, dst)
        assert H is None

    def test_returns_h_3x3(self):
        src, dst, _ = _make_points_and_H(10)
        H, _ = estimate_homography_ransac(src, dst)
        assert H is not None
        assert H.shape == (3, 3)


# ── check_homography_quality ──────────────────────────────────────────────────

class TestCheckHomographyQuality:

    def test_identity_is_valid(self):
        assert check_homography_quality(np.eye(3), (100, 100)) is True

    def test_negative_det_invalid(self):
        H = np.diag([-1.0, 1.0, 1.0])
        assert check_homography_quality(H, (100, 100)) is False

    def test_large_skew_invalid(self):
        H = np.diag([1000.0, 0.001, 1.0])
        assert check_homography_quality(H, (100, 100)) is False

    def test_clean_homography_valid(self):
        src, dst, _ = _make_points_and_H(8)
        H = estimate_homography_dlt(src, dst)
        assert check_homography_quality(H, (256, 256)) is True


# ── HomographyVerifier ────────────────────────────────────────────────────────

class TestHomographyVerifier:

    def test_instantiation_default(self):
        v = HomographyVerifier()
        assert v.cfg is not None

    def test_instantiation_custom(self):
        cfg = HomographyConfig(ransac_threshold=2.0)
        v = HomographyVerifier(cfg=cfg)
        assert v.cfg.ransac_threshold == 2.0

    def test_verify_returns_result(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert isinstance(r, HomographyResult)

    def test_clean_data_valid(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert bool(r.is_valid) is True

    def test_clean_data_high_score(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert r.score > 0.5

    def test_score_range(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert 0.0 <= r.score <= 1.0

    def test_few_points_invalid(self):
        src = np.array([[0., 0.], [1., 0.], [0., 1.]])
        dst = src + 1.0
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert r.is_valid is False
        assert r.score == pytest.approx(0.0)

    def test_inlier_ratio_range(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert 0.0 <= r.inlier_ratio <= 1.0

    def test_n_inliers_le_n_points(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert r.n_inliers <= 20

    def test_H_shape(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        assert r.H is not None
        assert r.H.shape == (3, 3)

    def test_verify_batch_empty(self):
        v = HomographyVerifier()
        results = v.verify_batch([])
        assert results == []

    def test_verify_batch_multiple(self):
        v = HomographyVerifier()
        src, dst, _ = _make_points_and_H(20)
        results = v.verify_batch([(src, dst), (src, dst)])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, HomographyResult)

    def test_reprojection_error_finite_on_valid(self):
        src, dst, _ = _make_points_and_H(20)
        v = HomographyVerifier()
        r = v.verify(src, dst)
        if r.is_valid:
            assert np.isfinite(r.reprojection_error)
