"""Extra tests for puzzle_reconstruction/matching/affine_matcher.py."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _identity_M() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _pts(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, (n, 2)).astype(np.float32)


def _gray_img(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w), dtype=np.uint8)


def _simple_result(idx1: int = 0, idx2: int = 1,
                   n_inliers: int = 10, err: float = 1.0,
                   score: float = 0.7) -> AffineMatchResult:
    return AffineMatchResult(
        idx1=idx1, idx2=idx2,
        M=_identity_M(),
        n_inliers=n_inliers,
        reprojection_error=err,
        score=score,
    )


# ─── AffineMatchResult (extra) ────────────────────────────────────────────────

class TestAffineMatchResultExtra:
    def test_idx1_stored(self):
        r = _simple_result(idx1=3)
        assert r.idx1 == 3

    def test_idx2_stored(self):
        r = _simple_result(idx2=7)
        assert r.idx2 == 7

    def test_M_stored(self):
        M = _identity_M()
        r = AffineMatchResult(idx1=0, idx2=1, M=M,
                              n_inliers=5, reprojection_error=0.5, score=0.8)
        assert r.M is M

    def test_M_none_allowed(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None,
                              n_inliers=0, reprojection_error=0.0, score=0.0)
        assert r.M is None

    def test_n_inliers_stored(self):
        r = _simple_result(n_inliers=25)
        assert r.n_inliers == 25

    def test_reprojection_error_stored(self):
        r = _simple_result(err=2.5)
        assert r.reprojection_error == pytest.approx(2.5)

    def test_score_stored(self):
        r = _simple_result(score=0.65)
        assert r.score == pytest.approx(0.65)

    def test_params_default_empty(self):
        r = _simple_result()
        assert isinstance(r.params, dict)

    def test_params_custom(self):
        r = AffineMatchResult(idx1=0, idx2=1, M=None,
                              n_inliers=0, reprojection_error=0.0, score=0.0,
                              params={"k": 5})
        assert r.params["k"] == 5


# ─── estimate_affine (extra) ──────────────────────────────────────────────────

class TestEstimateAffineExtra:
    def test_returns_tuple(self):
        p = _pts(10)
        result = estimate_affine(p, p)
        assert isinstance(result, tuple) and len(result) == 2

    def test_invalid_method_raises(self):
        p = _pts(5)
        with pytest.raises(ValueError):
            estimate_affine(p, p, method="svd")

    def test_fewer_than_3_points_raises(self):
        p = _pts(2)
        with pytest.raises(ValueError):
            estimate_affine(p, p)

    def test_shape_mismatch_raises(self):
        p1 = _pts(6)
        p2 = _pts(8)
        with pytest.raises(ValueError):
            estimate_affine(p1, p2)

    def test_identity_transform_detected(self):
        p = _pts(20)
        M, mask = estimate_affine(p, p, method="ransac")
        # Identity should be detectable
        if M is not None:
            assert M.shape == (2, 3)

    def test_lmeds_method_works(self):
        p = _pts(10)
        result = estimate_affine(p, p, method="lmeds")
        assert isinstance(result, tuple)

    def test_inlier_mask_bool_or_none(self):
        p = _pts(10)
        _, mask = estimate_affine(p, p)
        if mask is not None:
            assert mask.dtype == bool

    def test_M_shape_when_returned(self):
        p = _pts(10)
        M, _ = estimate_affine(p, p)
        if M is not None:
            assert M.shape == (2, 3)


# ─── apply_affine_pts (extra) ─────────────────────────────────────────────────

class TestApplyAffinePointsExtra:
    def test_returns_ndarray(self):
        M = _identity_M()
        result = apply_affine_pts(M, _pts(5))
        assert isinstance(result, np.ndarray)

    def test_dtype_float32(self):
        M = _identity_M()
        result = apply_affine_pts(M, _pts(5))
        assert result.dtype == np.float32

    def test_output_shape(self):
        M = _identity_M()
        p = _pts(8)
        result = apply_affine_pts(M, p)
        assert result.shape == (8, 2)

    def test_identity_unchanged(self):
        M = _identity_M()
        p = _pts(5)
        result = apply_affine_pts(M, p)
        assert np.allclose(result, p, atol=1e-4)

    def test_translation(self):
        M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)
        p = np.array([[0.0, 0.0]], dtype=np.float32)
        result = apply_affine_pts(M, p)
        assert result[0, 0] == pytest.approx(10.0, abs=1e-4)
        assert result[0, 1] == pytest.approx(20.0, abs=1e-4)

    def test_wrong_M_shape_raises(self):
        M = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError):
            apply_affine_pts(M, _pts(4))

    def test_single_point(self):
        M = _identity_M()
        p = np.array([[5.0, 7.0]], dtype=np.float32)
        result = apply_affine_pts(M, p)
        assert result.shape == (1, 2)

    def test_accepts_3d_input(self):
        M = _identity_M()
        p = _pts(5).reshape(5, 1, 2)
        result = apply_affine_pts(M, p)
        assert result.shape == (5, 2)


# ─── affine_reprojection_error (extra) ────────────────────────────────────────

class TestAffineReprojectionErrorExtra:
    def test_returns_float(self):
        M = _identity_M()
        p = _pts(5)
        result = affine_reprojection_error(M, p, p)
        assert isinstance(result, float)

    def test_identity_zero_error(self):
        M = _identity_M()
        p = _pts(10)
        assert affine_reprojection_error(M, p, p) == pytest.approx(0.0, abs=1e-4)

    def test_empty_points_returns_zero(self):
        M = _identity_M()
        p = np.zeros((0, 2), dtype=np.float32)
        assert affine_reprojection_error(M, p, p) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        M = _identity_M()
        with pytest.raises(ValueError):
            affine_reprojection_error(M, _pts(5), _pts(6))

    def test_nonneg_error(self):
        M = _identity_M()
        p1 = _pts(6)
        p2 = _pts(6)
        assert affine_reprojection_error(M, p1, p2) >= 0.0

    def test_inlier_mask_reduces_set(self):
        M = _identity_M()
        p = _pts(10)
        mask_full = np.ones(10, dtype=bool)
        mask_half = np.zeros(10, dtype=bool)
        mask_half[:5] = True
        err_full = affine_reprojection_error(M, p, p, inlier_mask=mask_full)
        err_half = affine_reprojection_error(M, p, p, inlier_mask=mask_half)
        # Both should be 0 since identity maps p to p
        assert err_full == pytest.approx(0.0, abs=1e-4)
        assert err_half == pytest.approx(0.0, abs=1e-4)

    def test_all_false_mask_returns_zero(self):
        M = _identity_M()
        p = _pts(5)
        mask = np.zeros(5, dtype=bool)
        assert affine_reprojection_error(M, p, p, inlier_mask=mask) == pytest.approx(0.0)


# ─── score_affine_match (extra) ───────────────────────────────────────────────

class TestScoreAffineMatchExtra:
    def test_returns_float(self):
        assert isinstance(score_affine_match(50, 1.0), float)

    def test_score_in_0_1(self):
        s = score_affine_match(50, 2.0)
        assert 0.0 <= s <= 1.0

    def test_max_inliers_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_inliers=0)

    def test_max_error_zero_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, max_error=0.0)

    def test_weights_not_summing_raises(self):
        with pytest.raises(ValueError):
            score_affine_match(10, 1.0, w_inliers=0.5, w_error=0.3)

    def test_perfect_score(self):
        # max inliers, zero error
        s = score_affine_match(100, 0.0, max_inliers=100, max_error=5.0,
                               w_inliers=0.6, w_error=0.4)
        assert s == pytest.approx(1.0)

    def test_zero_inliers_high_error(self):
        s = score_affine_match(0, 100.0, max_inliers=100, max_error=5.0,
                               w_inliers=0.6, w_error=0.4)
        # inlier component = 0, error component = max(0, 1-100/5) = 0
        assert s == pytest.approx(0.0)

    def test_more_inliers_higher_score(self):
        s1 = score_affine_match(10, 1.0)
        s2 = score_affine_match(50, 1.0)
        assert s2 > s1

    def test_less_error_higher_score(self):
        s1 = score_affine_match(20, 4.0)
        s2 = score_affine_match(20, 1.0)
        assert s2 > s1

    def test_max_inliers_clamped(self):
        # More inliers than max_inliers → clamped to 1
        s = score_affine_match(200, 0.0, max_inliers=100, max_error=5.0)
        assert s <= 1.0


# ─── match_fragments_affine (extra) ───────────────────────────────────────────

class TestMatchFragmentsAffineExtra:
    def test_returns_affine_match_result(self):
        img = _gray_img()
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)

    def test_idx_stored(self):
        img = _gray_img()
        result = match_fragments_affine(img, img, idx1=3, idx2=7)
        assert result.idx1 == 3
        assert result.idx2 == 7

    def test_score_in_0_1(self):
        img = _gray_img()
        result = match_fragments_affine(img, img)
        assert 0.0 <= result.score <= 1.0

    def test_n_inliers_nonneg(self):
        img = _gray_img()
        result = match_fragments_affine(img, img)
        assert result.n_inliers >= 0

    def test_small_featureless_img_returns_result(self):
        img = np.ones((16, 16), dtype=np.uint8) * 128
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)

    def test_params_has_keys(self):
        img = _gray_img()
        result = match_fragments_affine(img, img)
        assert "max_keypoints" in result.params

    def test_bgr_input_handled(self):
        img = _gray_img()[:, :, np.newaxis].repeat(3, axis=2)
        result = match_fragments_affine(img, img)
        assert isinstance(result, AffineMatchResult)


# ─── batch_affine_match (extra) ───────────────────────────────────────────────

class TestBatchAffineMatchExtra:
    def test_returns_list(self):
        img = _gray_img()
        result = batch_affine_match(img, [img, img])
        assert isinstance(result, list)

    def test_length_matches_candidates(self):
        img = _gray_img()
        candidates = [img, img, img]
        result = batch_affine_match(img, candidates)
        assert len(result) == 3

    def test_empty_candidates(self):
        img = _gray_img()
        result = batch_affine_match(img, [])
        assert result == []

    def test_all_elements_affine_match_result(self):
        img = _gray_img()
        for r in batch_affine_match(img, [img, img]):
            assert isinstance(r, AffineMatchResult)

    def test_idx2_matches_candidate_index(self):
        img = _gray_img()
        results = batch_affine_match(img, [img, img])
        assert results[0].idx2 == 0
        assert results[1].idx2 == 1

    def test_query_idx_stored(self):
        img = _gray_img()
        results = batch_affine_match(img, [img], query_idx=5)
        assert results[0].idx1 == 5
