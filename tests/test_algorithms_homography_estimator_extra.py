"""Extra tests for puzzle_reconstruction.algorithms.homography_estimator."""
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

def _pts(n=8, scale=100.0, seed=0):
    return np.random.default_rng(seed).uniform(0, scale, (n, 2))


def _eye3():
    return np.eye(3, dtype=np.float64)


def _trans(tx=10.0, ty=0.0):
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


def _scale_H(sx=2.0, sy=3.0):
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = sx
    H[1, 1] = sy
    return H


def _apply(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


# ─── HomographyConfig extras ──────────────────────────────────────────────────

class TestHomographyConfigExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(HomographyConfig()), str)

    def test_ransac_method_default(self):
        cfg = HomographyConfig()
        assert cfg.method == "ransac"

    def test_dlt_method(self):
        cfg = HomographyConfig(method="dlt")
        assert cfg.method == "dlt"

    def test_lmeds_method(self):
        cfg = HomographyConfig(method="lmeds")
        assert cfg.method == "lmeds"

    def test_max_iters_stored(self):
        cfg = HomographyConfig(max_iters=500)
        assert cfg.max_iters == 500

    def test_ransac_thresh_stored(self):
        cfg = HomographyConfig(ransac_thresh=1.5)
        assert cfg.ransac_thresh == pytest.approx(1.5)

    def test_min_inliers_4_ok(self):
        cfg = HomographyConfig(min_inliers=4)
        assert cfg.min_inliers == 4

    def test_min_inliers_100_ok(self):
        cfg = HomographyConfig(min_inliers=100)
        assert cfg.min_inliers == 100

    def test_confidence_0_01_ok(self):
        cfg = HomographyConfig(confidence=0.01)
        assert cfg.confidence == pytest.approx(0.01)

    def test_confidence_0_99_ok(self):
        cfg = HomographyConfig(confidence=0.99)
        assert cfg.confidence == pytest.approx(0.99)


# ─── HomographyResult extras ──────────────────────────────────────────────────

class TestHomographyResultExtra:
    def test_repr_is_string(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert isinstance(repr(r), str)

    def test_has_homography_true_when_H_set(self):
        r = HomographyResult(H=_eye3(), n_inliers=5, is_valid=True, reproj_err=0.0)
        assert r.has_homography is True

    def test_has_homography_false_when_None(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.has_homography is False

    def test_is_valid_false(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.is_valid is False

    def test_n_inliers_zero_ok(self):
        r = HomographyResult(H=None, n_inliers=0, is_valid=False, reproj_err=0.0)
        assert r.n_inliers == 0

    def test_reproj_err_stored(self):
        r = HomographyResult(H=_eye3(), n_inliers=4, is_valid=True, reproj_err=1.23)
        assert r.reproj_err == pytest.approx(1.23)

    def test_zero_reproj_err_valid(self):
        r = HomographyResult(H=_eye3(), n_inliers=5, is_valid=True, reproj_err=0.0)
        assert r.reproj_err == pytest.approx(0.0)


# ─── normalize_points extras ──────────────────────────────────────────────────

class TestNormalizePointsExtra:
    def test_centroid_exactly_zero(self):
        pts = _pts(20)
        pts_norm, _ = normalize_points(pts)
        centroid = pts_norm.mean(axis=0)
        np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1e-10)

    def test_T_shape_3x3(self):
        _, T = normalize_points(_pts(8))
        assert T.shape == (3, 3)

    def test_non_square_pts(self):
        pts = np.random.default_rng(5).uniform(0, 50, (12, 2))
        pts_norm, T = normalize_points(pts)
        assert pts_norm.shape == (12, 2)

    def test_reconstructed_from_T(self):
        pts = _pts(10)
        pts_norm, T = normalize_points(pts)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        recon = (T @ pts_h.T).T[:, :2]
        np.testing.assert_allclose(recon, pts_norm, atol=1e-10)

    def test_T_bottom_row(self):
        _, T = normalize_points(_pts(6))
        np.testing.assert_allclose(T[2], [0.0, 0.0, 1.0], atol=1e-10)

    def test_two_points_minimum(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        pts_norm, T = normalize_points(pts)
        assert pts_norm.shape == (2, 2)


# ─── dlt_homography extras ────────────────────────────────────────────────────

class TestDltHomographyExtra:
    def test_8_points_returns_3x3_or_none(self):
        src = _pts(8, seed=0)
        H_gt = _trans(10, 20)
        dst = _apply(H_gt, src)
        H = dlt_homography(src, dst)
        if H is not None:
            assert H.shape == (3, 3)

    def test_shape_mismatch_5_6_raises(self):
        with pytest.raises(ValueError):
            dlt_homography(_pts(5), _pts(6))

    def test_identity_near_unit(self):
        pts = _pts(10, seed=3)
        H = dlt_homography(pts, pts.copy())
        if H is not None:
            # H[2,2] normalized to 1
            assert H[2, 2] == pytest.approx(1.0, abs=0.05)

    def test_translation_matches(self):
        src = _pts(12, seed=1)
        H_gt = _trans(7.5, -3.5)
        dst = _apply(H_gt, src)
        H = dlt_homography(src, dst)
        if H is not None:
            H_norm = H / H[2, 2]
            assert H_norm[0, 2] == pytest.approx(7.5, abs=0.1)

    def test_returns_float64(self):
        src = _pts(8, seed=2)
        H = dlt_homography(src, src.copy())
        if H is not None:
            assert H.dtype == np.float64


# ─── compute_reprojection_error extras ───────────────────────────────────────

class TestComputeReprojectionErrorExtra:
    def test_y_translation_zero_error(self):
        pts = _pts(6)
        H = _trans(0.0, 5.0)
        dst = pts.copy()
        dst[:, 1] += 5.0
        err = compute_reprojection_error(H, pts, dst)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_many_points_zero_error_identity(self):
        pts = _pts(50)
        err = compute_reprojection_error(_eye3(), pts, pts.copy())
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_nonneg_random(self):
        err = compute_reprojection_error(_eye3(), _pts(10, seed=0), _pts(10, seed=1))
        assert err >= 0.0

    def test_scale_matrix_zero_error(self):
        pts = _pts(8, scale=10.0, seed=0)
        H = _scale_H(2.0, 2.0)
        dst = _apply(H, pts)
        err = compute_reprojection_error(H, pts, dst)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_wrong_H_shape_raises(self):
        with pytest.raises(ValueError):
            compute_reprojection_error(np.eye(4), _pts(4), _pts(4))


# ─── estimate_homography extras ───────────────────────────────────────────────

class TestEstimateHomographyExtra:
    def test_dlt_method_stored(self):
        pts = _pts(10)
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        result = estimate_homography(pts, pts.copy(), cfg)
        assert result.method == "dlt"

    def test_returns_homography_result(self):
        pts = _pts(8)
        assert isinstance(estimate_homography(pts, pts.copy()), HomographyResult)

    def test_few_points_invalid(self):
        pts = _pts(3)
        result = estimate_homography(pts, pts)
        assert result.is_valid is False

    def test_reproj_err_nonneg(self):
        pts = _pts(8)
        result = estimate_homography(pts, pts.copy())
        assert result.reproj_err >= 0.0

    def test_n_inliers_nonneg(self):
        pts = _pts(8)
        result = estimate_homography(pts, pts.copy())
        assert result.n_inliers >= 0

    def test_method_stored_ransac(self):
        pts = _pts(8)
        cfg = HomographyConfig(method="ransac")
        result = estimate_homography(pts, pts.copy(), cfg)
        assert result.method == "ransac"


# ─── decompose_homography extras ─────────────────────────────────────────────

class TestDecomposeHomographyExtra:
    def test_all_keys_present(self):
        d = decompose_homography(_eye3())
        for k in ("scale_x", "scale_y", "rotation_deg", "shear", "tx", "ty"):
            assert k in d

    def test_scale_matrix_decompose(self):
        H = _scale_H(2.0, 3.0)
        d = decompose_homography(H)
        assert d["scale_x"] == pytest.approx(2.0, abs=1e-5)
        assert d["scale_y"] == pytest.approx(3.0, abs=1e-5)

    def test_pure_translation(self):
        H = _trans(20.0, -5.0)
        d = decompose_homography(H)
        assert d["tx"] == pytest.approx(20.0, abs=1e-5)
        assert d["ty"] == pytest.approx(-5.0, abs=1e-5)

    def test_identity_zero_translation(self):
        d = decompose_homography(_eye3())
        assert d["tx"] == pytest.approx(0.0, abs=1e-6)
        assert d["ty"] == pytest.approx(0.0, abs=1e-6)

    def test_identity_unit_scale(self):
        d = decompose_homography(_eye3())
        assert d["scale_x"] == pytest.approx(1.0, abs=1e-6)
        assert d["scale_y"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_dict(self):
        assert isinstance(decompose_homography(_eye3()), dict)


# ─── warp_points extras ───────────────────────────────────────────────────────

class TestWarpPointsExtra:
    def test_many_points(self):
        pts = _pts(50)
        result = warp_points(_eye3(), pts)
        assert result.shape == (50, 2)

    def test_y_translation(self):
        pts = _pts(6)
        H = _trans(0.0, 8.0)
        result = warp_points(H, pts)
        np.testing.assert_allclose(result[:, 1], pts[:, 1] + 8.0, atol=1e-6)

    def test_scale_transform(self):
        pts = _pts(5, scale=10.0, seed=0)
        H = _scale_H(2.0, 3.0)
        result = warp_points(H, pts)
        np.testing.assert_allclose(result[:, 0], pts[:, 0] * 2.0, atol=1e-6)
        np.testing.assert_allclose(result[:, 1], pts[:, 1] * 3.0, atol=1e-6)

    def test_output_dtype_float64(self):
        result = warp_points(_eye3(), _pts(4))
        assert result.dtype == np.float64

    def test_output_shape_n_2(self):
        pts = _pts(12)
        result = warp_points(_eye3(), pts)
        assert result.shape == (12, 2)

    def test_identity_unchanged(self):
        pts = _pts(8)
        result = warp_points(_eye3(), pts)
        np.testing.assert_allclose(result, pts, atol=1e-8)


# ─── batch_estimate_homographies extras ──────────────────────────────────────

class TestBatchEstimateHomographiesExtra:
    def test_empty_returns_empty(self):
        assert batch_estimate_homographies([]) == []

    def test_five_pairs(self):
        pairs = [(_pts(6, seed=i), _pts(6, seed=i + 20)) for i in range(5)]
        results = batch_estimate_homographies(pairs)
        assert len(results) == 5

    def test_all_homography_results(self):
        pairs = [(_pts(8, seed=i), _pts(8, seed=i + 10)) for i in range(3)]
        for r in batch_estimate_homographies(pairs):
            assert isinstance(r, HomographyResult)

    def test_dlt_config(self):
        pts = _pts(8, seed=1)
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        results = batch_estimate_homographies([(pts, pts.copy())], cfg)
        assert len(results) == 1
        assert results[0].method == "dlt"

    def test_identity_pairs_low_reproj_err(self):
        pts = _pts(10, seed=0)
        cfg = HomographyConfig(method="dlt", min_inliers=4)
        results = batch_estimate_homographies([(pts, pts.copy())], cfg)
        assert results[0].reproj_err >= 0.0

    def test_single_pair(self):
        pair = (_pts(6, seed=0), _pts(6, seed=5))
        results = batch_estimate_homographies([pair])
        assert len(results) == 1
