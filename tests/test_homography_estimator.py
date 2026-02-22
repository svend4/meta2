"""Тесты для puzzle_reconstruction.algorithms.homography_estimator."""
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
    return rng.uniform(0, 100, (n, 2)).astype(np.float64)


def _identity_H() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _translation_H(tx: float, ty: float) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


# ─── TestHomographyConfigDefaults ─────────────────────────────────────────────

class TestHomographyConfigDefaults:
    def test_default_method(self):
        cfg = HomographyConfig()
        assert cfg.method == "ransac"

    def test_default_ransac_thresh(self):
        cfg = HomographyConfig()
        assert cfg.ransac_thresh > 0

    def test_default_max_iters(self):
        cfg = HomographyConfig()
        assert cfg.max_iters >= 1

    def test_default_confidence_in_range(self):
        cfg = HomographyConfig()
        assert 0.0 < cfg.confidence < 1.0

    def test_default_min_inliers(self):
        cfg = HomographyConfig()
        assert cfg.min_inliers >= 4

    def test_explicit_dlt(self):
        cfg = HomographyConfig(method="dlt")
        assert cfg.method == "dlt"

    def test_explicit_lmeds(self):
        cfg = HomographyConfig(method="lmeds")
        assert cfg.method == "lmeds"


# ─── TestHomographyConfigValidation ───────────────────────────────────────────

class TestHomographyConfigValidation:
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            HomographyConfig(method="bad")

    def test_ransac_thresh_zero(self):
        with pytest.raises(ValueError):
            HomographyConfig(ransac_thresh=0.0)

    def test_ransac_thresh_negative(self):
        with pytest.raises(ValueError):
            HomographyConfig(ransac_thresh=-1.0)

    def test_max_iters_zero(self):
        with pytest.raises(ValueError):
            HomographyConfig(max_iters=0)

    def test_confidence_zero(self):
        with pytest.raises(ValueError):
            HomographyConfig(confidence=0.0)

    def test_confidence_one(self):
        with pytest.raises(ValueError):
            HomographyConfig(confidence=1.0)

    def test_min_inliers_three(self):
        with pytest.raises(ValueError):
            HomographyConfig(min_inliers=3)


# ─── TestHomographyResult ─────────────────────────────────────────────────────

class TestHomographyResult:
    def test_has_homography_true(self):
        H = np.eye(3, dtype=np.float64)
        res = HomographyResult(H=H, n_inliers=10, is_valid=True,
                               reproj_err=0.5)
        assert res.has_homography is True

    def test_has_homography_false(self):
        res = HomographyResult(H=None, n_inliers=0, is_valid=False,
                               reproj_err=0.0)
        assert res.has_homography is False

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            HomographyResult(H=None, n_inliers=-1, is_valid=False,
                             reproj_err=0.0)

    def test_negative_reproj_err_raises(self):
        with pytest.raises(ValueError):
            HomographyResult(H=None, n_inliers=0, is_valid=False,
                             reproj_err=-0.1)

    def test_method_stored(self):
        res = HomographyResult(H=None, n_inliers=0, is_valid=False,
                               reproj_err=0.0, method="dlt")
        assert res.method == "dlt"


# ─── TestNormalizePoints ──────────────────────────────────────────────────────

class TestNormalizePoints:
    def test_returns_two_elements(self):
        pts = _make_pts(6)
        result = normalize_points(pts)
        assert len(result) == 2

    def test_normalized_shape(self):
        pts = _make_pts(6)
        pts_n, T = normalize_points(pts)
        assert pts_n.shape == pts.shape

    def test_T_is_3x3(self):
        pts = _make_pts(6)
        _, T = normalize_points(pts)
        assert T.shape == (3, 3)

    def test_centroid_near_origin(self):
        pts = _make_pts(20)
        pts_n, _ = normalize_points(pts)
        centroid = pts_n.mean(axis=0)
        assert abs(centroid[0]) < 1e-8
        assert abs(centroid[1]) < 1e-8

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_points(np.ones((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            normalize_points(np.array([[1.0, 2.0]]))

    def test_1d_array_raises(self):
        with pytest.raises(ValueError):
            normalize_points(np.array([1.0, 2.0, 3.0, 4.0]))


# ─── TestDltHomography ────────────────────────────────────────────────────────

class TestDltHomography:
    def test_identity_recovered(self):
        src = _make_pts(8, seed=1)
        H = dlt_homography(src, src)
        assert H is not None
        assert H.shape == (3, 3)
        assert abs(H[2, 2] - 1.0) < 1e-6

    def test_translation_recovered(self):
        src = _make_pts(10, seed=2)
        H_true = _translation_H(5.0, -3.0)
        dst = warp_points(H_true, src)
        H_est = dlt_homography(src, dst)
        assert H_est is not None
        err = compute_reprojection_error(H_est, src, dst)
        assert err < 1e-4

    def test_returns_3x3(self):
        src = _make_pts(6)
        dst = _make_pts(6, seed=5)
        H = dlt_homography(src, dst)
        if H is not None:
            assert H.shape == (3, 3)

    def test_too_few_points_raises(self):
        src = _make_pts(3)
        dst = _make_pts(3, seed=10)
        with pytest.raises(ValueError):
            dlt_homography(src, dst)

    def test_shape_mismatch_raises(self):
        src = _make_pts(5)
        dst = _make_pts(6)
        with pytest.raises(ValueError):
            dlt_homography(src, dst)

    def test_normalised_h22_is_one(self):
        src = _make_pts(8)
        dst = warp_points(_translation_H(2.0, 3.0), src)
        H = dlt_homography(src, dst)
        assert H is not None
        assert abs(H[2, 2] - 1.0) < 1e-6


# ─── TestComputeReprojectionError ─────────────────────────────────────────────

class TestComputeReprojectionError:
    def test_identity_zero_error(self):
        pts = _make_pts(8)
        H = _identity_H()
        err = compute_reprojection_error(H, pts, pts)
        assert err < 1e-10

    def test_perfect_translation_zero_error(self):
        src = _make_pts(8)
        H = _translation_H(10.0, -5.0)
        dst = warp_points(H, src)
        err = compute_reprojection_error(H, src, dst)
        assert err < 1e-8

    def test_wrong_H_shape_raises(self):
        with pytest.raises(ValueError):
            compute_reprojection_error(np.eye(4), _make_pts(4), _make_pts(4))

    def test_returns_float(self):
        err = compute_reprojection_error(_identity_H(), _make_pts(5), _make_pts(5))
        assert isinstance(err, float)

    def test_error_is_non_negative(self):
        err = compute_reprojection_error(_identity_H(), _make_pts(5), _make_pts(5, seed=99))
        assert err >= 0.0


# ─── TestEstimateHomography ───────────────────────────────────────────────────

class TestEstimateHomography:
    def test_returns_homography_result(self):
        src = _make_pts(10)
        dst = warp_points(_translation_H(3.0, 2.0), src)
        res = estimate_homography(src, dst, HomographyConfig(method="dlt"))
        assert isinstance(res, HomographyResult)

    def test_dlt_valid_for_clean_data(self):
        src = _make_pts(10, seed=7)
        dst = warp_points(_translation_H(4.0, 1.0), src)
        res = estimate_homography(src, dst, HomographyConfig(method="dlt",
                                                              min_inliers=4))
        assert res.has_homography

    def test_too_few_points_returns_invalid(self):
        src = _make_pts(2)
        dst = _make_pts(2, seed=3)
        res = estimate_homography(src, dst)
        assert not res.is_valid
        assert res.H is None

    def test_ransac_method(self):
        src = _make_pts(20, seed=8)
        dst = warp_points(_translation_H(6.0, -4.0), src)
        res = estimate_homography(src, dst, HomographyConfig(method="ransac",
                                                              min_inliers=4))
        assert isinstance(res, HomographyResult)

    def test_default_cfg(self):
        src = _make_pts(10)
        dst = warp_points(_translation_H(1.0, 1.0), src)
        res = estimate_homography(src, dst)
        assert isinstance(res, HomographyResult)

    def test_reproj_err_non_negative(self):
        src = _make_pts(10)
        dst = warp_points(_translation_H(2.0, 2.0), src)
        res = estimate_homography(src, dst, HomographyConfig(method="dlt",
                                                              min_inliers=4))
        assert res.reproj_err >= 0.0


# ─── TestDecomposeHomography ──────────────────────────────────────────────────

class TestDecomposeHomography:
    def test_identity_keys(self):
        d = decompose_homography(_identity_H())
        assert set(d.keys()) == {"scale_x", "scale_y", "rotation_deg",
                                  "shear", "tx", "ty"}

    def test_identity_zero_translation(self):
        d = decompose_homography(_identity_H())
        assert abs(d["tx"]) < 1e-10
        assert abs(d["ty"]) < 1e-10

    def test_identity_scale_one(self):
        d = decompose_homography(_identity_H())
        assert abs(d["scale_x"] - 1.0) < 1e-10

    def test_pure_translation(self):
        H = _translation_H(7.5, -3.0)
        d = decompose_homography(H)
        assert abs(d["tx"] - 7.5) < 1e-10
        assert abs(d["ty"] - (-3.0)) < 1e-10

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            decompose_homography(np.eye(4))

    def test_returns_dict(self):
        d = decompose_homography(_identity_H())
        assert isinstance(d, dict)

    def test_rotation_in_degrees(self):
        angle_rad = math.pi / 4
        H = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0.0],
            [0.0,                  0.0,                  1.0],
        ])
        d = decompose_homography(H)
        assert abs(d["rotation_deg"] - 45.0) < 1e-6


# ─── TestWarpPoints ───────────────────────────────────────────────────────────

class TestWarpPoints:
    def test_identity_unchanged(self):
        pts = _make_pts(5)
        warped = warp_points(_identity_H(), pts)
        assert np.allclose(warped, pts, atol=1e-10)

    def test_translation_applied(self):
        pts = _make_pts(5)
        H = _translation_H(10.0, -5.0)
        warped = warp_points(H, pts)
        expected = pts + np.array([10.0, -5.0])
        assert np.allclose(warped, expected, atol=1e-8)

    def test_output_shape(self):
        pts = _make_pts(7)
        warped = warp_points(_identity_H(), pts)
        assert warped.shape == pts.shape

    def test_wrong_H_shape_raises(self):
        with pytest.raises(ValueError):
            warp_points(np.eye(4), _make_pts(4))

    def test_wrong_pts_shape_raises(self):
        with pytest.raises(ValueError):
            warp_points(_identity_H(), np.ones((5, 3)))

    def test_output_dtype_float64(self):
        pts = _make_pts(5).astype(np.float32)
        warped = warp_points(_identity_H(), pts)
        assert warped.dtype == np.float64


# ─── TestBatchEstimateHomographies ────────────────────────────────────────────

class TestBatchEstimateHomographies:
    def _pair(self, tx, ty, n=10, seed=0):
        src = _make_pts(n, seed=seed)
        dst = warp_points(_translation_H(tx, ty), src)
        return src, dst

    def test_empty_list(self):
        result = batch_estimate_homographies([])
        assert result == []

    def test_single_pair(self):
        pair = self._pair(3.0, 2.0)
        result = batch_estimate_homographies([pair],
                                             HomographyConfig(method="dlt",
                                                               min_inliers=4))
        assert len(result) == 1
        assert isinstance(result[0], HomographyResult)

    def test_multiple_pairs(self):
        pairs = [self._pair(float(i), 0.0, seed=i) for i in range(4)]
        result = batch_estimate_homographies(pairs,
                                              HomographyConfig(method="dlt",
                                                                min_inliers=4))
        assert len(result) == 4

    def test_returns_list(self):
        result = batch_estimate_homographies([self._pair(1.0, 1.0)],
                                              HomographyConfig(method="dlt",
                                                                min_inliers=4))
        assert isinstance(result, list)

    def test_default_cfg(self):
        pair = self._pair(2.0, 2.0)
        result = batch_estimate_homographies([pair])
        assert len(result) == 1
