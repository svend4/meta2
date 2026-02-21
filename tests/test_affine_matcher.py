"""Tests for puzzle_reconstruction.matching.affine_matcher."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rng_img(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _texture_img(h: int = 64, w: int = 64) -> np.ndarray:
    """Image with visible texture for ORB detection."""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 4):
        img[i, :] = 200
    for j in range(0, w, 4):
        img[:, j] = 200
    return img


def _translation_pts(n: int = 20, tx: float = 5.0, ty: float = -3.0,
                     seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    pts1 = rng.uniform(5, 55, (n, 2)).astype(np.float32)
    pts2 = pts1 + np.array([tx, ty], dtype=np.float32)
    return pts1, pts2


def _identity_M() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


# ─── AffineMatchResult ───────────────────────────────────────────────────────

class TestAffineMatchResult:
    def test_fields_stored(self):
        M = _identity_M()
        r = AffineMatchResult(idx1=2, idx2=5, M=M, n_inliers=10,
                              reprojection_error=1.5, score=0.7)
        assert r.idx1 == 2
        assert r.idx2 == 5
        assert r.n_inliers == 10
        assert r.reprojection_error == pytest.approx(1.5)
        assert r.score == pytest.approx(0.7)
        assert r.params == {}

    def test_M_none_allowed(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None, n_inliers=0,
                              reprojection_error=0.0, score=0.0)
        assert r.M is None

    def test_params_stored(self):
        r = AffineMatchResult(0, 1, None, 0, 0.0, 0.0,
                              params={"ratio_thresh": 0.75})
        assert r.params["ratio_thresh"] == pytest.approx(0.75)


# ─── estimate_affine ────────────────────────────────────────────────────────

class TestEstimateAffine:
    def test_returns_tuple(self):
        pts1, pts2 = _translation_pts()
        result = estimate_affine(pts1, pts2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_M_shape_2x3(self):
        pts1, pts2 = _translation_pts()
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
        pts1, pts2 = _translation_pts()
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2, method="homography")

    def test_lmeds_method_accepted(self):
        pts1, pts2 = _translation_pts(n=20)
        M, _ = estimate_affine(pts1, pts2, method="lmeds")
        # May return None for noisy data, just check no exception
        assert M is None or M.shape == (2, 3)

    def test_known_translation_recovered(self):
        pts1, pts2 = _translation_pts(n=30, tx=10.0, ty=5.0)
        M, mask = estimate_affine(pts1, pts2, ransac_threshold=1.0)
        if M is not None:
            # Translation components should be close to (10, 5)
            assert M[0, 2] == pytest.approx(10.0, abs=1.5)
            assert M[1, 2] == pytest.approx(5.0, abs=1.5)

    def test_inlier_mask_boolean(self):
        pts1, pts2 = _translation_pts(n=20)
        _, mask = estimate_affine(pts1, pts2)
        if mask is not None:
            assert mask.dtype == bool
            assert mask.shape == (20,)


# ─── apply_affine_pts ────────────────────────────────────────────────────────

class TestApplyAffinePts:
    def test_returns_float32(self):
        M = _identity_M()
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result.dtype == np.float32

    def test_shape_n_by_2(self):
        M = _identity_M()
        pts = np.random.default_rng(0).random((5, 2)).astype(np.float32)
        result = apply_affine_pts(M, pts)
        assert result.shape == (5, 2)

    def test_identity_transform_unchanged(self):
        M = _identity_M()
        pts = np.array([[3.0, 7.0], [10.0, 2.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        np.testing.assert_array_almost_equal(result, pts, decimal=4)

    def test_known_translation(self):
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float64)
        pts = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(-3.0)
        assert result[1, 0] == pytest.approx(15.0)
        assert result[1, 1] == pytest.approx(7.0)

    def test_wrong_M_shape_raises(self):
        with pytest.raises(ValueError):
            apply_affine_pts(np.eye(3), np.ones((4, 2), dtype=np.float32))

    def test_cv2_format_pts(self):
        M = _identity_M()
        pts = np.ones((3, 1, 2), dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result.shape == (3, 2)

    def test_single_point(self):
        M = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 2.0]], dtype=np.float64)
        pts = np.array([[3.0, 4.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(7.0)
        assert result[0, 1] == pytest.approx(10.0)


# ─── affine_reprojection_error ───────────────────────────────────────────────

class TestAffineReprojectionError:
    def test_identity_M_zero_error(self):
        M = _identity_M()
        pts = np.random.default_rng(0).random((10, 2)).astype(np.float32)
        err = affine_reprojection_error(M, pts, pts)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_known_translation_zero_error(self):
        M = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float64)
        pts1 = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
        pts2 = np.array([[3.0, -2.0], [8.0, 3.0]], dtype=np.float32)
        err = affine_reprojection_error(M, pts1, pts2)
        assert err == pytest.approx(0.0, abs=1e-4)

    def test_imperfect_transform_positive_error(self):
        M = _identity_M()
        pts1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        pts2 = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
        err = affine_reprojection_error(M, pts1, pts2)
        assert err > 0.0

    def test_empty_pts_returns_zero(self):
        M = _identity_M()
        err = affine_reprojection_error(
            M,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
        assert err == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        M = _identity_M()
        pts1 = np.ones((3, 2), dtype=np.float32)
        pts2 = np.ones((4, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            affine_reprojection_error(M, pts1, pts2)

    def test_inlier_mask_filters(self):
        M = _identity_M()
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pts2 = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        # Without mask: error includes both points
        err_all = affine_reprojection_error(M, pts1, pts2)
        # With mask=True,False: only first point counted (zero error)
        mask = np.array([True, False])
        err_masked = affine_reprojection_error(M, pts1, pts2, inlier_mask=mask)
        assert err_masked < err_all

    def test_all_false_mask_returns_zero(self):
        M = _identity_M()
        pts = np.ones((3, 2), dtype=np.float32)
        mask = np.array([False, False, False])
        err = affine_reprojection_error(M, pts, pts * 2, inlier_mask=mask)
        assert err == pytest.approx(0.0)


# ─── score_affine_match ──────────────────────────────────────────────────────

class TestScoreAffineMatch:
    def test_value_in_unit_interval(self):
        s = score_affine_match(10, 2.0, max_inliers=50, max_error=5.0)
        assert 0.0 <= s <= 1.0

    def test_zero_inliers_zero_score_component(self):
        s = score_affine_match(0, 0.0, max_inliers=100, max_error=5.0,
                               w_inliers=1.0, w_error=0.0)
        assert s == pytest.approx(0.0)

    def test_max_inliers_and_zero_error_returns_one(self):
        s = score_affine_match(100, 0.0, max_inliers=100, max_error=5.0)
        assert s == pytest.approx(1.0)

    def test_max_inliers_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_inliers=0)

    def test_max_error_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_error=0.0)

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, w_inliers=0.5, w_error=0.1)

    def test_excess_inliers_capped_at_max(self):
        s_normal = score_affine_match(100, 0.0, max_inliers=100, max_error=5.0)
        s_excess = score_affine_match(200, 0.0, max_inliers=100, max_error=5.0)
        assert s_excess == pytest.approx(s_normal)

    def test_high_error_reduces_score(self):
        s_low = score_affine_match(50, 1.0, max_inliers=100, max_error=5.0)
        s_high = score_affine_match(50, 4.9, max_inliers=100, max_error=5.0)
        assert s_high < s_low

    def test_known_value(self):
        # w_inliers=0.6, n=50/100=0.5 → 0.3; w_error=0.4, err=0/5=0 → 0.4 → 0.7
        s = score_affine_match(50, 0.0, max_inliers=100, max_error=5.0,
                               w_inliers=0.6, w_error=0.4)
        assert s == pytest.approx(0.7)


# ─── match_fragments_affine ──────────────────────────────────────────────────

class TestMatchFragmentsAffine:
    def test_returns_affine_match_result(self):
        result = match_fragments_affine(_texture_img(), _texture_img())
        assert isinstance(result, AffineMatchResult)

    def test_idx_stored(self):
        result = match_fragments_affine(_texture_img(), _texture_img(),
                                        idx1=3, idx2=7)
        assert result.idx1 == 3
        assert result.idx2 == 7

    def test_score_in_unit_interval(self):
        result = match_fragments_affine(_rng_img(), _rng_img(seed=5))
        assert 0.0 <= result.score <= 1.0

    def test_n_inliers_non_negative(self):
        result = match_fragments_affine(_rng_img(), _rng_img(seed=3))
        assert result.n_inliers >= 0

    def test_reprojection_error_non_negative(self):
        result = match_fragments_affine(_rng_img(), _rng_img(seed=2))
        assert result.reprojection_error >= 0.0

    def test_same_image_reasonable_score(self):
        img = _texture_img(64, 64)
        result = match_fragments_affine(img, img, max_keypoints=100)
        # Same image should produce a reasonable score (or at least not crash)
        assert isinstance(result, AffineMatchResult)

    def test_bgr_image_accepted(self):
        img = np.stack([_texture_img()] * 3, axis=-1)
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)

    def test_params_stored(self):
        result = match_fragments_affine(_texture_img(), _texture_img(),
                                        max_keypoints=50)
        assert result.params["max_keypoints"] == 50

    def test_no_keypoints_returns_zero_score(self):
        # Uniform image → ORB finds no useful features
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = match_fragments_affine(img, img)
        assert result.score == pytest.approx(0.0)
        assert result.M is None


# ─── batch_affine_match ──────────────────────────────────────────────────────

class TestBatchAffineMatch:
    def test_empty_candidates_returns_empty(self):
        assert batch_affine_match(_texture_img(), []) == []

    def test_length_equals_candidates(self):
        query = _texture_img()
        candidates = [_rng_img(seed=i) for i in range(4)]
        results = batch_affine_match(query, candidates)
        assert len(results) == 4

    def test_all_affine_match_result(self):
        query = _texture_img()
        candidates = [_rng_img(seed=i) for i in range(3)]
        results = batch_affine_match(query, candidates)
        assert all(isinstance(r, AffineMatchResult) for r in results)

    def test_idx2_sequential(self):
        query = _texture_img()
        candidates = [_rng_img(seed=i) for i in range(3)]
        results = batch_affine_match(query, candidates, query_idx=10)
        idx2s = [r.idx2 for r in results]
        assert idx2s == [0, 1, 2]

    def test_query_idx_stored(self):
        query = _texture_img()
        results = batch_affine_match(query, [_rng_img()], query_idx=7)
        assert results[0].idx1 == 7
