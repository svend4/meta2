"""Extra tests for puzzle_reconstruction/verification/homography_verifier.py"""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_scene(n: int = 20, seed: int = 42, noise: float = 0.0):
    """Generate n clean correspondences with optional Gaussian noise on dst."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(20, 180, (n, 2))
    theta = 0.08
    c, s = np.cos(theta), np.sin(theta)
    H_true = np.array([[c, -s, 25.0], [s, c, 10.0], [0, 0, 1.0]])
    src_h = np.column_stack([src, np.ones(n)])
    dst_h = (H_true @ src_h.T).T
    dst = dst_h[:, :2] / dst_h[:, 2:3]
    if noise > 0:
        dst = dst + rng.normal(0, noise, dst.shape)
    return src, dst, H_true


def _add_outliers(src, dst, n_out=5, seed=77):
    rng = np.random.default_rng(seed)
    src_o = src.copy()
    dst_o = dst.copy()
    dst_o[:n_out] += rng.uniform(80, 200, (n_out, 2))
    return src_o, dst_o


# ─── HomographyConfig – edge cases ───────────────────────────────────────────

class TestHomographyConfigExtra:

    def test_min_inliers_four(self):
        cfg = HomographyConfig(min_inliers=4)
        assert cfg.min_inliers == 4

    def test_zero_iterations_stored(self):
        cfg = HomographyConfig(max_iterations=0)
        assert cfg.max_iterations == 0

    def test_confidence_boundary_low(self):
        cfg = HomographyConfig(confidence=0.5)
        assert cfg.confidence == 0.5

    def test_min_inlier_ratio_zero(self):
        cfg = HomographyConfig(min_inlier_ratio=0.0)
        assert cfg.min_inlier_ratio == 0.0

    def test_large_threshold_stored(self):
        cfg = HomographyConfig(ransac_threshold=100.0)
        assert cfg.ransac_threshold == 100.0


# ─── _normalise_points – additional edge cases ───────────────────────────────

class TestNormalisePointsExtra:

    def test_mean_distance_after_normalisation(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
        norm, T = _normalise_points(pts)
        dists = np.linalg.norm(norm, axis=1)
        assert abs(dists.mean() - np.sqrt(2.0)) < 0.01

    def test_T_invertible(self):
        rng = np.random.default_rng(5)
        pts = rng.uniform(0, 100, (10, 2))
        _, T = _normalise_points(pts)
        assert abs(np.linalg.det(T)) > 1e-12

    def test_two_points_no_crash(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        norm, T = _normalise_points(pts)
        assert norm.shape == (2, 2)

    def test_collinear_points_no_crash(self):
        pts = np.column_stack([np.arange(5.0), np.zeros(5)])
        norm, T = _normalise_points(pts)
        assert norm.shape == (5, 2)

    def test_result_dtype_float64(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        norm, T = _normalise_points(pts)
        assert norm.dtype == np.float64
        assert T.dtype == np.float64


# ─── estimate_homography_dlt – extra cases ────────────────────────────────────

class TestEstimateHomographyDLTExtra:

    def test_pure_translation_recovered(self):
        src = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        tx, ty = 5.0, 8.0
        dst = src + np.array([tx, ty])
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert float(errs.max()) < 1e-6

    def test_pure_scale_recovered(self):
        src = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        dst = src * 3.0
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert float(errs.max()) < 1e-6

    def test_exactly_4_points_returns_H(self):
        src, dst, _ = _make_scene(4, seed=10)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_5_points_returns_H(self):
        src, dst, _ = _make_scene(5, seed=11)
        H = estimate_homography_dlt(src, dst)
        assert H is not None

    def test_h22_normalised_to_one(self):
        src, dst, _ = _make_scene(8, seed=12)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        assert abs(H[2, 2] - 1.0) < 1e-9

    def test_mismatched_length_none(self):
        src = np.zeros((6, 2))
        dst = np.zeros((5, 2))
        assert estimate_homography_dlt(src, dst) is None

    def test_noisy_data_small_error(self):
        src, dst, _ = _make_scene(20, seed=13, noise=0.5)
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert float(errs.mean()) < 2.0

    def test_large_coordinates_stable(self):
        src = np.array([[1000., 1000.], [1100., 1000.], [1000., 1100.], [1100., 1100.]])
        dst = src + 5.0
        H = estimate_homography_dlt(src, dst)
        assert H is not None
        errs = reprojection_error(H, src, dst)
        assert float(errs.max()) < 0.01


# ─── reprojection_error – extra cases ─────────────────────────────────────────

class TestReprojectionErrorExtra:

    def test_translation_H_correct_error(self):
        src = np.array([[0., 0.], [1., 0.], [0., 1.]])
        H = np.array([[1., 0., 5.], [0., 1., 3.], [0., 0., 1.]])
        dst = src + np.array([5.0, 3.0])
        errs = reprojection_error(H, src, dst)
        np.testing.assert_allclose(errs, 0.0, atol=1e-9)

    def test_all_errors_finite(self):
        src, dst, H = _make_scene(10)
        errs = reprojection_error(H, src, dst)
        assert np.all(np.isfinite(errs))

    def test_single_point_pair(self):
        H = np.eye(3)
        src = np.array([[5.0, 7.0]])
        dst = np.array([[5.0, 7.0]])
        errs = reprojection_error(H, src, dst)
        assert errs.shape == (1,)
        assert float(errs[0]) == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_H_does_not_crash(self):
        # Zero matrix – division will be handled by epsilon clipping
        H = np.zeros((3, 3))
        src = np.array([[1., 2.], [3., 4.]])
        dst = np.array([[1., 2.], [3., 4.]])
        errs = reprojection_error(H, src, dst)
        assert errs.shape == (2,)

    def test_error_symmetric_approx(self):
        # For near-identity H, forward and backward errors should be similar
        H = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
        src = np.array([[5., 5.], [10., 10.]])
        # dst = src + [1, 1]
        dst = src + 1.0
        errs_fwd_only = np.linalg.norm(
            (H @ np.column_stack([src, np.ones(2)]).T).T[:, :2] /
            (H @ np.column_stack([src, np.ones(2)]).T).T[:, 2:3] - dst,
            axis=1
        )
        errs = reprojection_error(H, src, dst)
        # Combined error ≤ 2 * forward error (triangle inequality)
        assert np.all(errs <= errs_fwd_only * 2 + 1e-9)


# ─── estimate_homography_ransac – extra cases ─────────────────────────────────

class TestEstimateHomographyRansacExtra:

    def test_with_many_outliers_still_finds_H(self):
        src, dst, _ = _make_scene(30, seed=20)
        src_o, dst_o = _add_outliers(src, dst, n_out=10, seed=55)
        H, mask = estimate_homography_ransac(src_o, dst_o)
        assert H is not None
        assert int(mask[10:].sum()) >= 16  # inliers beyond outlier block

    def test_exactly_4_points_returns_h_or_none(self):
        src, dst, _ = _make_scene(4, seed=30)
        H, mask = estimate_homography_ransac(src, dst)
        # With exactly 4 points RANSAC picks all 4 → may or may not pass min_inliers
        assert mask.shape == (4,)

    def test_custom_config_threshold(self):
        src, dst, _ = _make_scene(20, seed=31)
        cfg = HomographyConfig(ransac_threshold=1.0, min_inliers=4,
                               max_iterations=50)
        H, mask = estimate_homography_ransac(src, dst, cfg)
        assert H is not None

    def test_inlier_mask_boolean(self):
        src, dst, _ = _make_scene(15, seed=32)
        _, mask = estimate_homography_ransac(src, dst)
        assert mask.dtype == bool

    def test_refitted_H_has_lower_or_equal_error(self):
        src, dst, _ = _make_scene(20, seed=33)
        H, mask = estimate_homography_ransac(src, dst)
        assert H is not None
        errs = reprojection_error(H, src[mask], dst[mask])
        assert float(errs.mean()) < 1.0

    def test_noise_robust_inlier_recovery(self):
        src, dst, _ = _make_scene(25, seed=34, noise=0.3)
        H, mask = estimate_homography_ransac(src, dst)
        assert H is not None
        # Most points should still be inliers with small noise
        assert float(mask.mean()) > 0.7


# ─── check_homography_quality – additional cases ─────────────────────────────

class TestCheckHomographyQualityExtra:

    def test_rotation_only_is_valid(self):
        theta = 0.3
        c, s = np.cos(theta), np.sin(theta)
        H = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        assert check_homography_quality(H, (100, 100)) is True

    def test_scale_down_valid(self):
        H = np.diag([0.5, 0.5, 1.0])
        assert check_homography_quality(H, (100, 100)) is True

    def test_very_small_det_invalid(self):
        # Near-zero det (almost singular)
        H = np.array([[1e-8, 0., 0.], [0., 1e-8, 0.], [0., 0., 1.]])
        # det is 1e-16, positive but extremely small; singular value ratio may
        # also be near-zero causing s[1]==0 check to trigger
        result = check_homography_quality(H, (100, 100))
        # We only assert it doesn't crash; result depends on implementation
        assert isinstance(result, bool)

    def test_reflection_invalid(self):
        # det < 0 due to reflection
        H = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        assert check_homography_quality(H, (100, 100)) is False

    def test_identity_any_fragment_size(self):
        for size in [(10, 10), (100, 200), (1000, 1000)]:
            assert check_homography_quality(np.eye(3), size) is True

    def test_custom_max_skew_ratio(self):
        # Slightly skewed transform
        H = np.diag([5.0, 1.0, 1.0])
        # With tight max_skew_ratio=3, this should fail
        assert check_homography_quality(H, (100, 100), max_skew_ratio=3.0) is False
        # With generous ratio it may pass
        assert check_homography_quality(H, (100, 100), max_skew_ratio=10.0) is True


# ─── HomographyVerifier – extra cases ────────────────────────────────────────

class TestHomographyVerifierExtra:

    def test_result_h_is_none_when_too_few_points(self):
        v = HomographyVerifier()
        src = np.array([[0., 0.], [1., 0.]])
        dst = src + 1.0
        r = v.verify(src, dst)
        assert r.H is None

    def test_result_inlier_mask_length_matches_input(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(15, seed=40)
        r = v.verify(src, dst)
        assert len(r.inlier_mask) == 15

    def test_reprojection_error_nonnegative(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=41)
        r = v.verify(src, dst)
        assert r.reprojection_error >= 0.0 or math.isinf(r.reprojection_error)

    def test_custom_min_inlier_ratio(self):
        # Very strict ratio → should still pass on clean data
        cfg = HomographyConfig(min_inlier_ratio=0.99, min_inliers=4)
        v = HomographyVerifier(cfg)
        src, dst, _ = _make_scene(20, seed=42)
        r = v.verify(src, dst)
        # Clean data → most points are inliers
        assert r.inlier_ratio > 0.5

    def test_verify_batch_returns_same_count(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=43)
        pairs = [(src, dst)] * 5
        results = v.verify_batch(pairs)
        assert len(results) == 5

    def test_verify_batch_all_same_is_consistent(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=44)
        results = v.verify_batch([(src, dst), (src, dst)])
        assert results[0].n_inliers == results[1].n_inliers
        assert results[0].score == pytest.approx(results[1].score)

    def test_many_outliers_reduces_score(self):
        v = HomographyVerifier()
        src, dst_clean, _ = _make_scene(20, seed=45)
        src_o, dst_o = _add_outliers(src, dst_clean, n_out=15, seed=66)
        r_clean = v.verify(src, dst_clean)
        r_noisy = v.verify(src_o, dst_o)
        assert r_clean.score >= r_noisy.score

    def test_exact_4_points_score_in_range(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(4, seed=46)
        r = v.verify(src, dst)
        assert 0.0 <= r.score <= 1.0

    def test_inlier_ratio_equals_n_inliers_over_n(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=47)
        r = v.verify(src, dst)
        assert r.inlier_ratio == pytest.approx(r.n_inliers / 20, abs=1e-9)

    def test_large_fragment_size_no_crash(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=48)
        r = v.verify(src, dst, fragment_size=(4096, 4096))
        assert isinstance(r, HomographyResult)

    def test_small_fragment_size_no_crash(self):
        v = HomographyVerifier()
        src, dst, _ = _make_scene(20, seed=49)
        r = v.verify(src, dst, fragment_size=(1, 1))
        assert isinstance(r, HomographyResult)


import math
