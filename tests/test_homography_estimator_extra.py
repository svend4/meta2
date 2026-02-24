"""Extra tests for puzzle_reconstruction/algorithms/homography_estimator.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.homography_estimator import (
    HomographyConfig,
    HomographyResult,
    batch_estimate_homographies,
    compute_reprojection_error,
    decompose_homography,
    dlt_homography,
    estimate_homography,
    normalize_points,
    warp_points,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_pts(n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(10, 90, (n, 2)).astype(np.float64)


def _identity_H() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _translation_H(tx: float, ty: float) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


def _scale_H(sx: float, sy: float) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = sx
    H[1, 1] = sy
    return H


def _apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    denom = proj[:, 2:3]
    denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
    return proj[:, :2] / denom


# ─── HomographyConfig (extra) ─────────────────────────────────────────────────

class TestHomographyConfigExtra:
    def test_default_method(self):
        assert HomographyConfig().method == "ransac"

    def test_default_ransac_thresh(self):
        assert HomographyConfig().ransac_thresh == pytest.approx(3.0)

    def test_default_max_iters(self):
        assert HomographyConfig().max_iters == 2000

    def test_default_confidence(self):
        assert HomographyConfig().confidence == pytest.approx(0.995)

    def test_default_min_inliers(self):
        assert HomographyConfig().min_inliers == 4

    def test_method_dlt_valid(self):
        cfg = HomographyConfig(method="dlt")
        assert cfg.method == "dlt"

    def test_method_lmeds_valid(self):
        cfg = HomographyConfig(method="lmeds")
        assert cfg.method == "lmeds"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(method="invalid")

    def test_negative_ransac_thresh_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(ransac_thresh=-1.0)

    def test_zero_ransac_thresh_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(ransac_thresh=0.0)

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(max_iters=0)

    def test_confidence_zero_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(confidence=0.0)

    def test_confidence_one_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(confidence=1.0)

    def test_min_inliers_3_raises(self):
        with pytest.raises(ValueError):
            HomographyConfig(min_inliers=3)

    def test_min_inliers_4_valid(self):
        assert HomographyConfig(min_inliers=4).min_inliers == 4

    def test_min_inliers_10_valid(self):
        assert HomographyConfig(min_inliers=10).min_inliers == 10

    def test_custom_ransac_thresh(self):
        cfg = HomographyConfig(ransac_thresh=5.0)
        assert cfg.ransac_thresh == pytest.approx(5.0)

    def test_custom_max_iters(self):
        cfg = HomographyConfig(max_iters=500)
        assert cfg.max_iters == 500


# ─── HomographyResult (extra) ─────────────────────────────────────────────────

class TestHomographyResultExtra:
    def test_valid_result_has_homography_true(self):
        r = HomographyResult(
            H=np.eye(3), n_inliers=8, is_valid=True, reproj_err=0.1
        )
        assert r.has_homography is True

    def test_none_H_has_homography_false(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.has_homography is False

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            HomographyResult(H=None, n_inliers=-1, is_valid=False, reproj_err=0.0)

    def test_negative_reproj_err_raises(self):
        with pytest.raises(ValueError):
            HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=-0.1)

    def test_zero_n_inliers_valid(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.n_inliers == 0

    def test_method_stored(self):
        r = HomographyResult(
            H=None, n_inliers=0, is_valid=False, reproj_err=0.0, method="dlt"
        )
        assert r.method == "dlt"

    def test_is_valid_false_when_no_H(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.is_valid is False

    def test_is_valid_true_when_H_found(self):
        r = HomographyResult(
            H=np.eye(3), n_inliers=10, is_valid=True, reproj_err=0.5
        )
        assert r.is_valid is True


# ─── normalize_points (extra) ─────────────────────────────────────────────────

class TestNormalizePointsExtra:
    def test_returns_tuple(self):
        pts = _make_pts()
        result = normalize_points(pts)
        assert isinstance(result, tuple) and len(result) == 2

    def test_T_is_3x3(self):
        pts = _make_pts()
        _, T = normalize_points(pts)
        assert T.shape == (3, 3)

    def test_pts_norm_same_shape(self):
        pts = _make_pts(10)
        pts_norm, _ = normalize_points(pts)
        assert pts_norm.shape == pts.shape

    def test_centroid_near_zero(self):
        pts = _make_pts(20)
        pts_norm, _ = normalize_points(pts)
        centroid = pts_norm.mean(axis=0)
        assert np.allclose(centroid, 0.0, atol=1e-6)

    def test_less_than_2_points_raises(self):
        pts = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            normalize_points(pts)

    def test_wrong_shape_raises(self):
        pts = np.ones((5, 3))
        with pytest.raises(ValueError):
            normalize_points(pts)

    def test_1d_array_raises(self):
        pts = np.ones(6)
        with pytest.raises(ValueError):
            normalize_points(pts)

    def test_two_points_valid(self):
        pts = np.array([[0.0, 0.0], [10.0, 10.0]])
        pts_norm, T = normalize_points(pts)
        assert pts_norm.shape == (2, 2)

    def test_T_last_row(self):
        pts = _make_pts()
        _, T = normalize_points(pts)
        assert np.allclose(T[2], [0.0, 0.0, 1.0], atol=1e-10)


# ─── dlt_homography (extra) ───────────────────────────────────────────────────

class TestDltHomographyExtra:
    def test_identity_transform(self):
        src = _make_pts(8)
        H = dlt_homography(src, src)
        warped = _apply_H(H, src)
        assert np.allclose(warped, src, atol=1e-4)

    def test_translation_recovered(self):
        src = _make_pts(8)
        H_true = _translation_H(10.0, -5.0)
        dst = _apply_H(H_true, src)
        H_est = dlt_homography(src, dst)
        assert H_est is not None
        warped = _apply_H(H_est, src)
        assert np.allclose(warped, dst, atol=1e-4)

    def test_returns_3x3(self):
        src = _make_pts(8)
        H = dlt_homography(src, src)
        assert H is not None
        assert H.shape == (3, 3)

    def test_h33_normalized_to_1(self):
        src = _make_pts(8)
        H = dlt_homography(src, src)
        assert H is not None
        assert abs(H[2, 2] - 1.0) < 1e-6

    def test_fewer_than_4_points_raises(self):
        src = np.ones((3, 2))
        dst = np.ones((3, 2))
        with pytest.raises(ValueError):
            dlt_homography(src, dst)

    def test_mismatched_shapes_raises(self):
        src = _make_pts(8)
        dst = _make_pts(10)
        with pytest.raises(ValueError):
            dlt_homography(src, dst)

    def test_exactly_4_points(self):
        src = _make_pts(4)
        H = dlt_homography(src, src)
        assert H is not None

    def test_float64_output(self):
        src = _make_pts(8)
        H = dlt_homography(src, src)
        assert H.dtype == np.float64


# ─── compute_reprojection_error (extra) ───────────────────────────────────────

class TestComputeReprojectionErrorExtra:
    def test_identity_zero_error(self):
        pts = _make_pts(10)
        err = compute_reprojection_error(_identity_H(), pts, pts)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation_known_error(self):
        pts = _make_pts(10)
        H = _translation_H(5.0, 0.0)
        dst = _apply_H(H, pts)
        err = compute_reprojection_error(H, pts, dst)
        assert err == pytest.approx(0.0, abs=1e-4)

    def test_nonneg(self):
        pts = _make_pts(10)
        H = _translation_H(100.0, 100.0)
        dst = _make_pts(10, seed=42)
        err = compute_reprojection_error(H, pts, dst)
        assert err >= 0.0

    def test_non_3x3_H_raises(self):
        H = np.eye(2, dtype=np.float64)
        pts = _make_pts(8)
        with pytest.raises(ValueError):
            compute_reprojection_error(H, pts, pts)

    def test_returns_float(self):
        err = compute_reprojection_error(_identity_H(), _make_pts(5), _make_pts(5))
        assert isinstance(err, float)

    def test_larger_offset_larger_error(self):
        pts = _make_pts(10)
        dst_close = _apply_H(_translation_H(1.0, 0.0), pts)
        dst_far = _apply_H(_translation_H(100.0, 0.0), pts)
        err_close = compute_reprojection_error(_identity_H(), pts, dst_close)
        err_far = compute_reprojection_error(_identity_H(), pts, dst_far)
        assert err_far > err_close


# ─── estimate_homography (extra) ──────────────────────────────────────────────

class TestEstimateHomographyExtra:
    def test_returns_homography_result(self):
        src = _make_pts(20)
        dst = _apply_H(_translation_H(5.0, 3.0), src)
        result = estimate_homography(src, dst, HomographyConfig(method="dlt"))
        assert isinstance(result, HomographyResult)

    def test_too_few_points_invalid(self):
        src = np.ones((2, 2), dtype=np.float32)
        dst = np.ones((2, 2), dtype=np.float32)
        result = estimate_homography(src, dst)
        assert result.is_valid is False
        assert result.H is None

    def test_dlt_exact_identity(self):
        src = _make_pts(8)
        result = estimate_homography(
            src, src, HomographyConfig(method="dlt")
        )
        assert result.is_valid is True
        assert result.H is not None

    def test_ransac_returns_result(self):
        src = _make_pts(20)
        H_true = _translation_H(10.0, 5.0)
        dst = _apply_H(H_true, src)
        cfg = HomographyConfig(method="ransac", min_inliers=4)
        result = estimate_homography(src.astype(np.float32), dst.astype(np.float32), cfg)
        assert isinstance(result, HomographyResult)

    def test_reproj_err_nonneg(self):
        src = _make_pts(10)
        dst = _apply_H(_translation_H(3.0, 1.0), src)
        result = estimate_homography(
            src, dst, HomographyConfig(method="dlt")
        )
        assert result.reproj_err >= 0.0

    def test_default_config_used_when_none(self):
        src = np.ones((2, 2), dtype=np.float32)
        dst = np.ones((2, 2), dtype=np.float32)
        result = estimate_homography(src, dst, cfg=None)
        assert isinstance(result, HomographyResult)

    def test_n_inliers_nonneg(self):
        src = _make_pts(10)
        dst = _apply_H(_translation_H(5.0, 0.0), src)
        result = estimate_homography(src, dst, HomographyConfig(method="dlt"))
        assert result.n_inliers >= 0


# ─── decompose_homography (extra) ─────────────────────────────────────────────

class TestDecomposeHomographyExtra:
    def test_identity_zero_rotation(self):
        d = decompose_homography(_identity_H())
        assert d["rotation_deg"] == pytest.approx(0.0, abs=1e-5)

    def test_identity_unit_scale(self):
        d = decompose_homography(_identity_H())
        assert d["scale_x"] == pytest.approx(1.0, abs=1e-5)

    def test_identity_zero_translation(self):
        d = decompose_homography(_identity_H())
        assert d["tx"] == pytest.approx(0.0, abs=1e-5)
        assert d["ty"] == pytest.approx(0.0, abs=1e-5)

    def test_translation_extracted(self):
        H = _translation_H(15.0, -8.0)
        d = decompose_homography(H)
        assert d["tx"] == pytest.approx(15.0, abs=1e-5)
        assert d["ty"] == pytest.approx(-8.0, abs=1e-5)

    def test_scale_extracted(self):
        H = _scale_H(2.0, 3.0)
        d = decompose_homography(H)
        assert d["scale_x"] == pytest.approx(2.0, abs=1e-4)

    def test_non_3x3_raises(self):
        H = np.eye(2, dtype=np.float64)
        with pytest.raises(ValueError):
            decompose_homography(H)

    def test_returns_dict_with_keys(self):
        d = decompose_homography(_identity_H())
        for key in ("scale_x", "scale_y", "rotation_deg", "shear", "tx", "ty"):
            assert key in d

    def test_rotation_90_degrees(self):
        angle = math.pi / 2
        H = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle),  math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        d = decompose_homography(H)
        assert abs(d["rotation_deg"]) == pytest.approx(90.0, abs=0.5)

    def test_values_are_floats(self):
        d = decompose_homography(_identity_H())
        for v in d.values():
            assert isinstance(v, float)


# ─── warp_points (extra) ──────────────────────────────────────────────────────

class TestWarpPointsExtra:
    def test_identity_unchanged(self):
        pts = _make_pts(10)
        warped = warp_points(_identity_H(), pts)
        assert np.allclose(warped, pts, atol=1e-6)

    def test_translation_correct(self):
        pts = _make_pts(10)
        H = _translation_H(5.0, -3.0)
        warped = warp_points(H, pts)
        expected = pts + np.array([5.0, -3.0])
        assert np.allclose(warped, expected, atol=1e-6)

    def test_output_shape(self):
        pts = _make_pts(15)
        warped = warp_points(_identity_H(), pts)
        assert warped.shape == (15, 2)

    def test_output_float64(self):
        pts = _make_pts(10)
        warped = warp_points(_identity_H(), pts)
        assert warped.dtype == np.float64

    def test_non_3x3_H_raises(self):
        pts = _make_pts(5)
        H = np.eye(2, dtype=np.float64)
        with pytest.raises(ValueError):
            warp_points(H, pts)

    def test_wrong_pts_shape_raises(self):
        pts = np.ones((5, 3))
        with pytest.raises(ValueError):
            warp_points(_identity_H(), pts)

    def test_1d_pts_raises(self):
        pts = np.ones(6)
        with pytest.raises(ValueError):
            warp_points(_identity_H(), pts)

    def test_single_point(self):
        pts = np.array([[10.0, 20.0]])
        warped = warp_points(_translation_H(1.0, 2.0), pts)
        assert np.allclose(warped, [[11.0, 22.0]], atol=1e-6)

    def test_scale_transform(self):
        pts = np.array([[2.0, 3.0], [4.0, 6.0]])
        H = _scale_H(2.0, 2.0)
        warped = warp_points(H, pts)
        assert np.allclose(warped, pts * 2.0, atol=1e-6)


# ─── batch_estimate_homographies (extra) ──────────────────────────────────────

class TestBatchEstimateHomographiesExtra:
    def test_empty_list_returns_empty(self):
        result = batch_estimate_homographies([])
        assert result == []

    def test_single_pair(self):
        src = _make_pts(10)
        dst = _apply_H(_translation_H(5.0, 0.0), src)
        results = batch_estimate_homographies(
            [(src, dst)], HomographyConfig(method="dlt")
        )
        assert len(results) == 1
        assert isinstance(results[0], HomographyResult)

    def test_multiple_pairs(self):
        src = _make_pts(8)
        pairs = [(src, _apply_H(_translation_H(float(i), 0.0), src)) for i in range(3)]
        results = batch_estimate_homographies(pairs, HomographyConfig(method="dlt"))
        assert len(results) == 3

    def test_all_are_homography_results(self):
        src = _make_pts(8)
        dst = _apply_H(_translation_H(3.0, 2.0), src)
        results = batch_estimate_homographies(
            [(src, dst), (src, src)], HomographyConfig(method="dlt")
        )
        for r in results:
            assert isinstance(r, HomographyResult)

    def test_too_few_points_returns_invalid(self):
        src = np.ones((2, 2), dtype=np.float32)
        results = batch_estimate_homographies([(src, src)])
        assert results[0].is_valid is False

    def test_default_config(self):
        src = np.ones((2, 2), dtype=np.float32)
        results = batch_estimate_homographies([(src, src)], cfg=None)
        assert len(results) == 1
