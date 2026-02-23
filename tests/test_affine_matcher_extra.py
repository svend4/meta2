"""Extra tests for puzzle_reconstruction.matching.affine_matcher."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.affine_matcher import (
    AffineMatchResult,
    affine_reprojection_error,
    apply_affine_pts,
    batch_affine_match,
    estimate_affine,
    match_fragments_affine,
    score_affine_match,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rng_img(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _texture_img(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 4):
        img[i, :] = 200
    for j in range(0, w, 4):
        img[:, j] = 200
    return img


def _trans_pts(n=20, tx=5.0, ty=-3.0, seed=1):
    rng = np.random.default_rng(seed)
    pts1 = rng.uniform(5, 55, (n, 2)).astype(np.float32)
    pts2 = pts1 + np.array([tx, ty], dtype=np.float32)
    return pts1, pts2


def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


# ─── TestAffineMatchResultExtra ───────────────────────────────────────────────

class TestAffineMatchResultExtra:
    def test_idx1_idx2_stored(self):
        r = AffineMatchResult(idx1=2, idx2=5, M=None, n_inliers=0,
                              reprojection_error=0.0, score=0.0)
        assert r.idx1 == 2
        assert r.idx2 == 5

    def test_n_inliers_stored(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None, n_inliers=10,
                              reprojection_error=1.5, score=0.7)
        assert r.n_inliers == 10

    def test_score_stored(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None, n_inliers=0,
                              reprojection_error=0.0, score=0.85)
        assert r.score == pytest.approx(0.85)

    def test_M_none_ok(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None, n_inliers=0,
                              reprojection_error=0.0, score=0.0)
        assert r.M is None

    def test_M_shape_stored(self):
        M = _identity()
        r = AffineMatchResult(idx1=0, idx2=1, M=M, n_inliers=5,
                              reprojection_error=0.5, score=0.6)
        assert r.M.shape == (2, 3)

    def test_params_default_empty(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None, n_inliers=0,
                              reprojection_error=0.0, score=0.0)
        assert r.params == {}

    def test_params_stored(self):
        r = AffineMatchResult(0, 1, None, 0, 0.0, 0.0,
                              params={"ratio_thresh": 0.75})
        assert r.params["ratio_thresh"] == pytest.approx(0.75)

    def test_reprojection_error_stored(self):
        r = AffineMatchResult(0, 1, None, 0, 2.3, 0.5)
        assert r.reprojection_error == pytest.approx(2.3)


# ─── TestEstimateAffineExtra ─────────────────────────────────────────────────

class TestEstimateAffineExtra:
    def test_returns_tuple_len_2(self):
        pts1, pts2 = _trans_pts()
        result = estimate_affine(pts1, pts2)
        assert isinstance(result, tuple) and len(result) == 2

    def test_M_shape_2x3_if_found(self):
        pts1, pts2 = _trans_pts(n=30)
        M, _ = estimate_affine(pts1, pts2)
        if M is not None:
            assert M.shape == (2, 3)

    def test_fewer_than_3_pts_raises(self):
        pts1 = np.float32([[0, 0], [1, 1]])
        pts2 = np.float32([[1, 0], [2, 1]])
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2)

    def test_shape_mismatch_raises(self):
        pts1 = np.float32([[0, 0], [1, 1], [2, 2]])
        pts2 = np.float32([[0, 0], [1, 1]])
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2)

    def test_unknown_method_raises(self):
        pts1, pts2 = _trans_pts()
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2, method="homography")

    def test_known_translation(self):
        pts1, pts2 = _trans_pts(n=30, tx=10.0, ty=5.0)
        M, _ = estimate_affine(pts1, pts2, ransac_threshold=1.0)
        if M is not None:
            assert M[0, 2] == pytest.approx(10.0, abs=1.5)
            assert M[1, 2] == pytest.approx(5.0, abs=1.5)

    def test_inlier_mask_bool(self):
        pts1, pts2 = _trans_pts(n=20)
        _, mask = estimate_affine(pts1, pts2)
        if mask is not None:
            assert mask.dtype == bool
            assert mask.shape == (20,)


# ─── TestApplyAffinePtsExtra ─────────────────────────────────────────────────

class TestApplyAffinePtsExtra:
    def test_returns_float32(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = apply_affine_pts(_identity(), pts)
        assert result.dtype == np.float32

    def test_shape_n_by_2(self):
        pts = np.random.default_rng(0).random((5, 2)).astype(np.float32)
        result = apply_affine_pts(_identity(), pts)
        assert result.shape == (5, 2)

    def test_identity_unchanged(self):
        pts = np.array([[3.0, 7.0], [10.0, 2.0]], dtype=np.float32)
        result = apply_affine_pts(_identity(), pts)
        np.testing.assert_array_almost_equal(result, pts, decimal=4)

    def test_known_translation(self):
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float64)
        pts = np.array([[0.0, 0.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(-3.0)

    def test_wrong_M_shape_raises(self):
        with pytest.raises(ValueError):
            apply_affine_pts(np.eye(3), np.ones((4, 2), dtype=np.float32))

    def test_cv2_format_pts(self):
        pts = np.ones((3, 1, 2), dtype=np.float32)
        result = apply_affine_pts(_identity(), pts)
        assert result.shape == (3, 2)

    def test_single_point(self):
        M = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 2.0]], dtype=np.float64)
        pts = np.array([[3.0, 4.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(7.0)
        assert result[0, 1] == pytest.approx(10.0)


# ─── TestAffineReprojectionErrorExtra ────────────────────────────────────────

class TestAffineReprojectionErrorExtra:
    def test_identity_zero_error(self):
        pts = np.random.default_rng(0).random((10, 2)).astype(np.float32)
        assert affine_reprojection_error(_identity(), pts, pts) == pytest.approx(0.0, abs=1e-5)

    def test_imperfect_positive_error(self):
        pts1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        pts2 = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
        err = affine_reprojection_error(_identity(), pts1, pts2)
        assert err > 0.0

    def test_empty_pts_zero(self):
        err = affine_reprojection_error(
            _identity(),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
        assert err == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            affine_reprojection_error(_identity(),
                                      np.ones((3, 2), dtype=np.float32),
                                      np.ones((4, 2), dtype=np.float32))

    def test_inlier_mask_reduces_error(self):
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pts2 = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        err_all = affine_reprojection_error(_identity(), pts1, pts2)
        err_masked = affine_reprojection_error(_identity(), pts1, pts2,
                                               inlier_mask=np.array([True, False]))
        assert err_masked < err_all

    def test_all_false_mask_zero(self):
        pts = np.ones((3, 2), dtype=np.float32)
        mask = np.array([False, False, False])
        err = affine_reprojection_error(_identity(), pts, pts * 2, inlier_mask=mask)
        assert err == pytest.approx(0.0)


# ─── TestScoreAffineMatchExtra ────────────────────────────────────────────────

class TestScoreAffineMatchExtra:
    def test_in_unit_interval(self):
        s = score_affine_match(10, 2.0, max_inliers=50, max_error=5.0)
        assert 0.0 <= s <= 1.0

    def test_zero_inliers_zero_score_component(self):
        s = score_affine_match(0, 0.0, max_inliers=100, max_error=5.0,
                               w_inliers=1.0, w_error=0.0)
        assert s == pytest.approx(0.0)

    def test_max_inliers_zero_error_one(self):
        s = score_affine_match(100, 0.0, max_inliers=100, max_error=5.0)
        assert s == pytest.approx(1.0)

    def test_max_inliers_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_inliers=0)

    def test_max_error_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_error=0.0)

    def test_weights_not_summing_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, w_inliers=0.5, w_error=0.1)

    def test_high_error_lower_score(self):
        s_low = score_affine_match(50, 1.0, max_inliers=100, max_error=5.0)
        s_high = score_affine_match(50, 4.9, max_inliers=100, max_error=5.0)
        assert s_high < s_low

    def test_excess_inliers_capped(self):
        s_normal = score_affine_match(100, 0.0, max_inliers=100, max_error=5.0)
        s_excess = score_affine_match(200, 0.0, max_inliers=100, max_error=5.0)
        assert s_excess == pytest.approx(s_normal)


# ─── TestMatchFragmentsAffineExtra ───────────────────────────────────────────

class TestMatchFragmentsAffineExtra:
    def test_returns_affine_match_result(self):
        assert isinstance(match_fragments_affine(_texture_img(), _texture_img()),
                           AffineMatchResult)

    def test_idx_stored(self):
        r = match_fragments_affine(_texture_img(), _texture_img(), idx1=3, idx2=7)
        assert r.idx1 == 3 and r.idx2 == 7

    def test_score_in_0_1(self):
        r = match_fragments_affine(_rng_img(), _rng_img(seed=5))
        assert 0.0 <= r.score <= 1.0

    def test_n_inliers_nonneg(self):
        r = match_fragments_affine(_rng_img(), _rng_img(seed=3))
        assert r.n_inliers >= 0

    def test_reprojection_nonneg(self):
        r = match_fragments_affine(_rng_img(), _rng_img(seed=2))
        assert r.reprojection_error >= 0.0

    def test_uniform_image_zero_score(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        r = match_fragments_affine(img, img)
        assert r.score == pytest.approx(0.0) and r.M is None

    def test_bgr_accepted(self):
        img = np.stack([_texture_img()] * 3, axis=-1)
        assert isinstance(match_fragments_affine(img, img), AffineMatchResult)

    def test_params_stored(self):
        r = match_fragments_affine(_texture_img(), _texture_img(), max_keypoints=50)
        assert r.params["max_keypoints"] == 50


# ─── TestBatchAffineMatchExtra ────────────────────────────────────────────────

class TestBatchAffineMatchExtra:
    def test_empty_candidates_empty_result(self):
        assert batch_affine_match(_texture_img(), []) == []

    def test_length_equals_candidates(self):
        results = batch_affine_match(_texture_img(),
                                     [_rng_img(seed=i) for i in range(4)])
        assert len(results) == 4

    def test_all_affine_match_results(self):
        results = batch_affine_match(_texture_img(),
                                     [_rng_img(seed=i) for i in range(3)])
        assert all(isinstance(r, AffineMatchResult) for r in results)

    def test_idx2_sequential(self):
        results = batch_affine_match(_texture_img(),
                                     [_rng_img(seed=i) for i in range(3)],
                                     query_idx=10)
        assert [r.idx2 for r in results] == [0, 1, 2]

    def test_query_idx_stored(self):
        results = batch_affine_match(_texture_img(), [_rng_img()], query_idx=7)
        assert results[0].idx1 == 7
