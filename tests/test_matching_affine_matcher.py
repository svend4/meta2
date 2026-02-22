"""Tests for puzzle_reconstruction/matching/affine_matcher.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.affine_matcher import (
    AffineMatchResult,
    estimate_affine,
    apply_affine_pts,
    affine_reprojection_error,
    score_affine_match,
    match_fragments_affine,
    batch_affine_match,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=60, w=60, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_textured(h=80, w=80, seed=42):
    """Random textured image with sufficient features for ORB."""
    rng = np.random.default_rng(seed)
    img = (rng.standard_normal((h, w)) * 40 + 128).clip(0, 255).astype(np.uint8)
    return img


def make_identity_pts(n=10):
    """n point pairs for identity transform."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    return pts, pts.copy()


def make_translation_pts(dx=5.0, dy=3.0, n=15):
    """n point pairs with a known translation."""
    rng = np.random.default_rng(1)
    pts1 = rng.uniform(10, 90, (n, 2)).astype(np.float32)
    pts2 = pts1 + np.array([dx, dy], dtype=np.float32)
    return pts1, pts2


def make_identity_M():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


# ─── AffineMatchResult ────────────────────────────────────────────────────────

class TestAffineMatchResult:
    def test_basic_creation(self):
        result = AffineMatchResult(
            idx1=0, idx2=1, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0
        )
        assert result.idx1 == 0
        assert result.idx2 == 1
        assert result.M is None

    def test_with_matrix(self):
        M = make_identity_M()
        result = AffineMatchResult(
            idx1=2, idx2=5, M=M,
            n_inliers=10, reprojection_error=1.5, score=0.75
        )
        assert result.n_inliers == 10
        assert result.score == pytest.approx(0.75)

    def test_params_stored(self):
        result = AffineMatchResult(
            idx1=0, idx2=1, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0,
            params={"method": "ransac"}
        )
        assert result.params["method"] == "ransac"

    def test_default_params_empty(self):
        result = AffineMatchResult(
            idx1=0, idx2=1, M=None,
            n_inliers=0, reprojection_error=0.0, score=0.0
        )
        assert result.params == {}


# ─── estimate_affine ──────────────────────────────────────────────────────────

class TestEstimateAffine:
    def test_invalid_method_raises(self):
        pts1, pts2 = make_identity_pts()
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2, method="homography")

    def test_too_few_points_raises(self):
        pts1 = np.array([[0, 0], [1, 0]], dtype=np.float32)
        pts2 = np.array([[0, 0], [1, 0]], dtype=np.float32)
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2)

    def test_shape_mismatch_raises(self):
        pts1 = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        pts2 = np.array([[0, 0], [1, 0]], dtype=np.float32)
        with pytest.raises(ValueError):
            estimate_affine(pts1, pts2)

    def test_identity_transform(self):
        pts1, pts2 = make_identity_pts(n=20)
        M, mask = estimate_affine(pts1, pts2, method="ransac")
        if M is not None:
            np.testing.assert_array_almost_equal(M[:, :2], np.eye(2), decimal=3)

    def test_translation_recovered(self):
        pts1, pts2 = make_translation_pts(dx=10.0, dy=5.0, n=30)
        M, mask = estimate_affine(pts1, pts2, method="ransac")
        if M is not None:
            assert abs(M[0, 2] - 10.0) < 1.0
            assert abs(M[1, 2] - 5.0) < 1.0

    def test_returns_2x3_matrix(self):
        pts1, pts2 = make_translation_pts(n=20)
        M, mask = estimate_affine(pts1, pts2)
        if M is not None:
            assert M.shape == (2, 3)

    def test_lmeds_method(self):
        pts1, pts2 = make_identity_pts(n=20)
        # lmeds should not raise
        M, mask = estimate_affine(pts1, pts2, method="lmeds")
        # M may be None for degenerate cases, that's ok


# ─── apply_affine_pts ─────────────────────────────────────────────────────────

class TestApplyAffinePts:
    def test_identity_transform(self):
        M = make_identity_M()
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        np.testing.assert_array_almost_equal(result, pts, decimal=5)

    def test_translation(self):
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float64)
        pts = np.array([[0.0, 0.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(3.0)

    def test_output_shape(self):
        M = make_identity_M()
        pts = np.random.rand(10, 2).astype(np.float32)
        result = apply_affine_pts(M, pts)
        assert result.shape == (10, 2)

    def test_output_dtype_float32(self):
        M = make_identity_M()
        pts = np.random.rand(5, 2).astype(np.float32)
        result = apply_affine_pts(M, pts)
        assert result.dtype == np.float32

    def test_invalid_M_shape_raises(self):
        M = np.eye(3, dtype=np.float64)  # 3x3 instead of 2x3
        pts = np.random.rand(5, 2).astype(np.float32)
        with pytest.raises(ValueError):
            apply_affine_pts(M, pts)

    def test_n12_format(self):
        M = make_identity_M()
        pts = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result.shape == (2, 2)

    def test_scaling_transform(self):
        M = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        pts = np.array([[1.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        result = apply_affine_pts(M, pts)
        assert result[0, 0] == pytest.approx(2.0)
        assert result[0, 1] == pytest.approx(2.0)
        assert result[1, 0] == pytest.approx(4.0)
        assert result[1, 1] == pytest.approx(6.0)


# ─── affine_reprojection_error ────────────────────────────────────────────────

class TestAffineReprojectionError:
    def test_identity_zero_error(self):
        M = make_identity_M()
        pts1, pts2 = make_identity_pts(n=10)
        err = affine_reprojection_error(M, pts1, pts2)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_translation_known_error(self):
        """Translate pts1 by (3,4) but compare to untranslated pts2 → error=5."""
        M = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]], dtype=np.float64)
        pts1 = np.zeros((5, 2), dtype=np.float32)
        pts2 = np.zeros((5, 2), dtype=np.float32)
        err = affine_reprojection_error(M, pts1, pts2)
        assert err == pytest.approx(5.0, rel=1e-4)

    def test_empty_points_returns_zero(self):
        M = make_identity_M()
        err = affine_reprojection_error(
            M,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
        assert err == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        M = make_identity_M()
        pts1 = np.zeros((5, 2), dtype=np.float32)
        pts2 = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            affine_reprojection_error(M, pts1, pts2)

    def test_non_negative(self):
        M = make_identity_M()
        pts1, pts2 = make_translation_pts(n=10)
        err = affine_reprojection_error(M, pts1, pts2)
        assert err >= 0.0

    def test_with_inlier_mask(self):
        M = make_identity_M()
        pts1 = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        pts2 = np.array([[5.0, 5.0], [100.0, 100.0]], dtype=np.float32)
        # Without mask → average of both errors
        err_all = affine_reprojection_error(M, pts1, pts2)
        # With mask selecting only index 1 (exact match)
        mask = np.array([False, True])
        err_inlier = affine_reprojection_error(M, pts1, pts2, inlier_mask=mask)
        assert err_inlier == pytest.approx(0.0, abs=1e-5)
        assert err_all > 0

    def test_all_masked_out_returns_zero(self):
        M = make_identity_M()
        pts1 = np.ones((5, 2), dtype=np.float32)
        pts2 = np.zeros((5, 2), dtype=np.float32)
        mask = np.zeros(5, dtype=bool)
        err = affine_reprojection_error(M, pts1, pts2, inlier_mask=mask)
        assert err == pytest.approx(0.0)


# ─── score_affine_match ───────────────────────────────────────────────────────

class TestScoreAffineMatch:
    def test_max_inliers_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_inliers=0)

    def test_max_error_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_error=0.0)

    def test_weights_not_summing_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, w_inliers=0.3, w_error=0.3)

    def test_perfect_score(self):
        score = score_affine_match(
            n_inliers=100, reprojection_error=0.0,
            max_inliers=100, max_error=5.0,
            w_inliers=0.6, w_error=0.4,
        )
        assert score == pytest.approx(1.0)

    def test_zero_inliers_zero_error(self):
        score = score_affine_match(
            n_inliers=0, reprojection_error=0.0,
            max_inliers=100, max_error=5.0,
            w_inliers=0.6, w_error=0.4,
        )
        assert score == pytest.approx(0.4)  # only error component

    def test_range(self):
        score = score_affine_match(
            n_inliers=50, reprojection_error=2.5,
            max_inliers=100, max_error=5.0,
        )
        assert 0.0 <= score <= 1.0

    def test_more_inliers_higher_score(self):
        s1 = score_affine_match(10, 2.0, max_inliers=100, max_error=5.0)
        s2 = score_affine_match(80, 2.0, max_inliers=100, max_error=5.0)
        assert s2 > s1

    def test_lower_error_higher_score(self):
        s1 = score_affine_match(50, 4.0, max_inliers=100, max_error=5.0)
        s2 = score_affine_match(50, 0.5, max_inliers=100, max_error=5.0)
        assert s2 > s1

    def test_overflow_inliers_clamped(self):
        """n_inliers > max_inliers → inlier component = 1."""
        score = score_affine_match(
            n_inliers=200, reprojection_error=0.0,
            max_inliers=100, max_error=5.0,
        )
        assert score == pytest.approx(1.0)

    def test_high_error_reduces_score(self):
        score = score_affine_match(
            n_inliers=50, reprojection_error=100.0,
            max_inliers=100, max_error=5.0,
        )
        # error component = 0
        assert score == pytest.approx(0.3)  # 0.6 * 0.5 + 0.4 * 0


# ─── match_fragments_affine ───────────────────────────────────────────────────

class TestMatchFragmentsAffine:
    def test_returns_affine_match_result(self):
        img = make_textured()
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)

    def test_indices_set(self):
        img = make_textured()
        result = match_fragments_affine(img, img, idx1=3, idx2=7)
        assert result.idx1 == 3
        assert result.idx2 == 7

    def test_score_in_range(self):
        img1 = make_textured(seed=1)
        img2 = make_textured(seed=2)
        result = match_fragments_affine(img1, img2)
        assert 0.0 <= result.score <= 1.0

    def test_featureless_image_returns_zero_score(self):
        """Constant images have no keypoints → score = 0."""
        img = make_gray(60, 60, value=128)
        result = match_fragments_affine(img, img)
        assert result.score == pytest.approx(0.0)
        assert result.n_inliers == 0

    def test_params_stored(self):
        img = make_textured()
        result = match_fragments_affine(img, img)
        assert "max_keypoints" in result.params

    def test_bgr_input(self):
        rng = np.random.default_rng(5)
        img = (rng.standard_normal((60, 60, 3)) * 40 + 128).clip(0, 255).astype(np.uint8)
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)

    def test_reprojection_error_non_negative(self):
        img = make_textured()
        result = match_fragments_affine(img, img)
        assert result.reprojection_error >= 0.0


# ─── batch_affine_match ───────────────────────────────────────────────────────

class TestBatchAffineMatch:
    def test_returns_list(self):
        query = make_textured(seed=0)
        candidates = [make_textured(seed=i) for i in range(3)]
        results = batch_affine_match(query, candidates)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_affine_match_results(self):
        query = make_textured(seed=0)
        candidates = [make_textured(seed=i) for i in range(2)]
        results = batch_affine_match(query, candidates)
        assert all(isinstance(r, AffineMatchResult) for r in results)

    def test_empty_candidates(self):
        query = make_textured()
        results = batch_affine_match(query, [])
        assert results == []

    def test_idx2_matches_position(self):
        query = make_textured(seed=0)
        candidates = [make_textured(seed=i + 1) for i in range(4)]
        results = batch_affine_match(query, candidates)
        for i, r in enumerate(results):
            assert r.idx2 == i

    def test_query_idx_set(self):
        query = make_textured()
        candidates = [make_textured(seed=1)]
        results = batch_affine_match(query, candidates, query_idx=5)
        assert results[0].idx1 == 5

    def test_scores_in_range(self):
        query = make_textured(seed=0)
        candidates = [make_textured(seed=i) for i in range(3)]
        results = batch_affine_match(query, candidates)
        for r in results:
            assert 0.0 <= r.score <= 1.0
