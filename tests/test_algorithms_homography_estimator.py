"""Tests for puzzle_reconstruction.algorithms.homography_estimator."""
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

def _random_pts(n=8, scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, scale, (n, 2))


def _identity_H():
    return np.eye(3, dtype=np.float64)


def _translation_H(tx, ty):
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


def _apply_H_to_pts(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


# ─── TestHomographyConfig ─────────────────────────────────────────────────────

class TestHomographyConfig:
    def test_defaults(self):
        cfg = HomographyConfig()
        assert cfg.method == "ransac"
        assert cfg.ransac_thresh == pytest.approx(3.0)
        assert cfg.max_iters == 2000
        assert cfg.min_inliers == 4

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            HomographyConfig(method="invalid")

    def test_ransac_thresh_nonpositive_raises(self):
        with pytest.raises(ValueError, match="ransac_thresh"):
            HomographyConfig(ransac_thresh=0.0)

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            HomographyConfig(max_iters=0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            HomographyConfig(confidence=1.0)
        with pytest.raises(ValueError, match="confidence"):
            HomographyConfig(confidence=0.0)

    def test_min_inliers_less_than_4_raises(self):
        with pytest.raises(ValueError, match="min_inliers"):
            HomographyConfig(min_inliers=3)

    def test_dlt_method_allowed(self):
        cfg = HomographyConfig(method="dlt")
        assert cfg.method == "dlt"

    def test_lmeds_method_allowed(self):
        cfg = HomographyConfig(method="lmeds")
        assert cfg.method == "lmeds"


# ─── TestHomographyResult ─────────────────────────────────────────────────────

class TestHomographyResult:
    def test_valid_creation(self):
        r = HomographyResult(H=_identity_H(), n_inliers=10,
                             is_valid=True, reproj_err=0.5)
        assert r.n_inliers == 10
        assert r.is_valid is True

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError, match="n_inliers"):
            HomographyResult(H=None, n_inliers=-1, is_valid=False,
                             reproj_err=0.0)

    def test_negative_reproj_err_raises(self):
        with pytest.raises(ValueError, match="reproj_err"):
            HomographyResult(H=None, n_inliers=0, is_valid=False,
                             reproj_err=-1.0)

    def test_has_homography_true(self):
        r = HomographyResult(H=_identity_H(), n_inliers=5,
                             is_valid=True, reproj_err=0.1)
        assert r.has_homography is True

    def test_has_homography_false(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False,
                             reproj_err=0.0)
        assert r.has_homography is False


# ─── TestNormalizePoints ──────────────────────────────────────────────────────

class TestNormalizePoints:
    def test_returns_normalized_and_transform(self):
        pts = _random_pts(10)
        pts_norm, T = normalize_points(pts)
        assert pts_norm.shape == pts.shape
        assert T.shape == (3, 3)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="pts"):
            normalize_points(np.zeros((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="2"):
            normalize_points(np.zeros((1, 2)))

    def test_normalized_centroid_near_zero(self):
        pts = _random_pts(20)
        pts_norm, _ = normalize_points(pts)
        centroid = pts_norm.mean(axis=0)
        np.testing.assert_array_almost_equal(centroid, [0.0, 0.0], decimal=10)

    def test_transform_reconstructs_points(self):
        pts = _random_pts(6)
        pts_norm, T = normalize_points(pts)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        reconstructed = (T @ pts_h.T).T[:, :2]
        np.testing.assert_array_almost_equal(reconstructed, pts_norm,
                                             decimal=10)


# ─── TestDltHomography ────────────────────────────────────────────────────────

class TestDltHomography:
    def test_too_few_points_raises(self):
        src = _random_pts(3)
        with pytest.raises(ValueError, match="4"):
            dlt_homography(src, src)

    def test_shape_mismatch_raises(self):
        src = _random_pts(6)
        dst = _random_pts(5)
        with pytest.raises(ValueError, match="Формы"):
            dlt_homography(src, dst)

    def test_returns_3x3_or_none(self):
        src = _random_pts(8)
        H_gt = _translation_H(10, 20)
        dst = _apply_H_to_pts(H_gt, src)
        H = dlt_homography(src, dst)
        if H is not None:
            assert H.shape == (3, 3)

    def test_identity_transform(self):
        """DLT with src == dst should give near-identity."""
        pts = _random_pts(10, seed=7)
        H = dlt_homography(pts, pts.copy())
        if H is not None:
            assert H[2, 2] == pytest.approx(1.0, abs=0.01)


# ─── TestComputeReprojectionError ────────────────────────────────────────────

class TestComputeReprojectionError:
    def test_non_3x3_raises(self):
        H = np.eye(2)
        with pytest.raises(ValueError, match="3×3"):
            compute_reprojection_error(H, np.zeros((3, 2)), np.zeros((3, 2)))

    def test_identity_zero_error(self):
        pts = _random_pts(10)
        err = compute_reprojection_error(_identity_H(), pts, pts.copy())
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_known_translation_error(self):
        """Pure translation H, compare to manually shifted pts."""
        pts = _random_pts(5)
        H = _translation_H(10.0, 0.0)
        dst = pts.copy()
        dst[:, 0] += 10.0
        err = compute_reprojection_error(H, pts, dst)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_returns_nonneg(self):
        pts = _random_pts(8)
        dst = _random_pts(8, seed=1)
        err = compute_reprojection_error(_identity_H(), pts, dst)
        assert err >= 0.0


# ─── TestEstimateHomography ───────────────────────────────────────────────────

class TestEstimateHomography:
    def test_too_few_points_returns_invalid(self):
        pts = _random_pts(3)
        result = estimate_homography(pts, pts)
        assert result.is_valid is False
        assert result.H is None

    def test_dlt_with_identity_transform(self):
        pts = _random_pts(10, seed=0)
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        result = estimate_homography(pts, pts.copy(), cfg)
        assert result.H is not None
        assert result.reproj_err >= 0.0

    def test_returns_homography_result(self):
        pts = _random_pts(8)
        result = estimate_homography(pts, pts.copy())
        assert isinstance(result, HomographyResult)

    def test_method_stored(self):
        pts = _random_pts(8)
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        result = estimate_homography(pts, pts, cfg)
        assert result.method == "dlt"


# ─── TestDecomposeHomography ──────────────────────────────────────────────────

class TestDecomposeHomography:
    def test_non_3x3_raises(self):
        with pytest.raises(ValueError, match="3×3"):
            decompose_homography(np.eye(2))

    def test_returns_dict_with_keys(self):
        d = decompose_homography(_identity_H())
        for key in ("scale_x", "scale_y", "rotation_deg", "shear", "tx", "ty"):
            assert key in d

    def test_identity_decomposition(self):
        d = decompose_homography(_identity_H())
        assert d["scale_x"] == pytest.approx(1.0, abs=1e-6)
        assert d["rotation_deg"] == pytest.approx(0.0, abs=1e-4)
        assert d["tx"] == pytest.approx(0.0, abs=1e-6)
        assert d["ty"] == pytest.approx(0.0, abs=1e-6)

    def test_translation_decomposition(self):
        H = _translation_H(15.0, -7.0)
        d = decompose_homography(H)
        assert d["tx"] == pytest.approx(15.0, abs=1e-6)
        assert d["ty"] == pytest.approx(-7.0, abs=1e-6)


# ─── TestWarpPoints ───────────────────────────────────────────────────────────

class TestWarpPoints:
    def test_non_3x3_H_raises(self):
        with pytest.raises(ValueError, match="3×3"):
            warp_points(np.eye(2), np.zeros((3, 2)))

    def test_wrong_pts_shape_raises(self):
        with pytest.raises(ValueError, match="pts"):
            warp_points(_identity_H(), np.zeros((3, 3)))

    def test_identity_unchanged(self):
        pts = _random_pts(6)
        result = warp_points(_identity_H(), pts)
        np.testing.assert_array_almost_equal(result, pts, decimal=8)

    def test_translation_shifts_points(self):
        pts = _random_pts(5)
        H = _translation_H(20.0, 0.0)
        result = warp_points(H, pts)
        np.testing.assert_array_almost_equal(result[:, 0], pts[:, 0] + 20.0,
                                             decimal=6)

    def test_output_shape(self):
        pts = _random_pts(7)
        result = warp_points(_identity_H(), pts)
        assert result.shape == (7, 2)

    def test_output_dtype_float64(self):
        result = warp_points(_identity_H(), _random_pts(4))
        assert result.dtype == np.float64


# ─── TestBatchEstimateHomographies ───────────────────────────────────────────

class TestBatchEstimateHomographies:
    def test_returns_list(self):
        pairs = [(_random_pts(6), _random_pts(6))]
        results = batch_estimate_homographies(pairs)
        assert len(results) == 1
        assert isinstance(results[0], HomographyResult)

    def test_empty_list(self):
        assert batch_estimate_homographies([]) == []

    def test_multiple_pairs(self):
        pairs = [(_random_pts(4, seed=i), _random_pts(4, seed=i + 10))
                 for i in range(3)]
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        results = batch_estimate_homographies(pairs, cfg)
        assert len(results) == 3
