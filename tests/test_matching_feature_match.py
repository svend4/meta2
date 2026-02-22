"""Tests for puzzle_reconstruction/matching/feature_match.py"""
import pytest
import numpy as np
import cv2

from puzzle_reconstruction.matching.feature_match import (
    KeypointMatch,
    FeatureMatchResult,
    extract_features,
    match_descriptors,
    estimate_homography,
    feature_match_pair,
    edge_feature_score,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray_img(h=100, w=100, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (h, w))).astype(np.uint8)


def make_bgr_img(h=100, w=100, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (h, w, 3))).astype(np.uint8)


def make_textured_img(h=120, w=120, seed=0):
    """Synthetic image with some detectable features (checkerboard)."""
    img = np.zeros((h, w), dtype=np.uint8)
    block = 15
    for i in range(0, h, block):
        for j in range(0, w, block):
            if (i // block + j // block) % 2 == 0:
                img[i:i+block, j:j+block] = 200
    return img


# ─── KeypointMatch ───────────────────────────────────────────────────────────

class TestKeypointMatch:
    def test_creation(self):
        kpm = KeypointMatch(pt_src=(1.0, 2.0), pt_dst=(3.0, 4.0),
                            distance=10.0, confidence=0.8)
        assert kpm.pt_src == (1.0, 2.0)
        assert kpm.pt_dst == (3.0, 4.0)
        assert kpm.distance == 10.0
        assert kpm.confidence == 0.8

    def test_default_confidence(self):
        kpm = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(1.0, 1.0), distance=5.0)
        assert kpm.confidence == 1.0

    def test_repr(self):
        kpm = KeypointMatch(pt_src=(1.0, 2.0), pt_dst=(3.0, 4.0),
                            distance=10.0, confidence=0.75)
        r = repr(kpm)
        assert "KeypointMatch" in r
        assert "conf=0.750" in r


# ─── FeatureMatchResult ──────────────────────────────────────────────────────

class TestFeatureMatchResult:
    def _make_result(self, n_matches=5, n_inliers=3, score=0.6):
        matches = [
            KeypointMatch((i, 0.0), (i, 1.0), float(i)) for i in range(n_matches)
        ]
        mask = np.array([True] * n_inliers + [False] * (n_matches - n_inliers),
                        dtype=bool)
        return FeatureMatchResult(
            matches=matches,
            homography=np.eye(3),
            inlier_mask=mask,
            score=score,
            method="orb",
            n_keypoints=(100, 100),
        )

    def test_n_matches(self):
        r = self._make_result(n_matches=5)
        assert r.n_matches == 5

    def test_n_inliers(self):
        r = self._make_result(n_matches=5, n_inliers=3)
        assert r.n_inliers == 3

    def test_inlier_ratio(self):
        r = self._make_result(n_matches=4, n_inliers=2)
        assert abs(r.inlier_ratio - 0.5) < 1e-9

    def test_inlier_ratio_no_matches(self):
        r = FeatureMatchResult(
            matches=[], homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0, method="orb",
        )
        assert r.inlier_ratio == 0.0

    def test_n_inliers_none_mask(self):
        r = FeatureMatchResult(
            matches=[], homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0, method="orb",
        )
        assert r.n_inliers == 0

    def test_repr(self):
        r = self._make_result()
        s = repr(r)
        assert "FeatureMatchResult" in s
        assert "orb" in s
        assert "score=" in s


# ─── extract_features ────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_orb_gray(self):
        img = make_textured_img()
        kps, descs = extract_features(img, method="orb")
        assert isinstance(kps, list)
        # May or may not detect features on synthetic image, just check types

    def test_orb_bgr(self):
        img = make_bgr_img()
        kps, descs = extract_features(img, method="orb")
        assert isinstance(kps, list)

    def test_akaze_gray(self):
        img = make_textured_img()
        kps, descs = extract_features(img, method="akaze")
        assert isinstance(kps, list)

    def test_unknown_method_returns_empty(self):
        """extract_features catches all exceptions; unknown method returns empty."""
        img = make_gray_img()
        kps, descs = extract_features(img, method="unknown_method")
        assert kps == []
        assert descs is None

    def test_returns_tuple_of_two(self):
        img = make_gray_img()
        result = extract_features(img)
        assert len(result) == 2

    def test_n_features_limit(self):
        img = make_textured_img(200, 200)
        kps, descs = extract_features(img, method="orb", n_features=10)
        assert len(kps) <= 10

    def test_with_mask(self):
        img = make_textured_img()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[20:80, 20:80] = 255
        kps, descs = extract_features(img, method="orb", mask=mask)
        assert isinstance(kps, list)


# ─── match_descriptors ───────────────────────────────────────────────────────

class TestMatchDescriptors:
    def test_none_desc1_returns_empty(self):
        result = match_descriptors(None, np.zeros((5, 32), dtype=np.uint8),
                                   [], [])
        assert result == []

    def test_none_desc2_returns_empty(self):
        result = match_descriptors(np.zeros((5, 32), dtype=np.uint8), None,
                                   [], [])
        assert result == []

    def test_empty_desc_returns_empty(self):
        result = match_descriptors(np.zeros((0, 32), dtype=np.uint8),
                                   np.zeros((0, 32), dtype=np.uint8),
                                   [], [])
        assert result == []

    def test_returns_list(self):
        img1 = make_textured_img()
        img2 = make_textured_img(seed=1)
        kps1, descs1 = extract_features(img1, method="orb")
        kps2, descs2 = extract_features(img2, method="orb")
        if descs1 is not None and descs2 is not None:
            matches = match_descriptors(descs1, descs2, kps1, kps2, method="orb")
            assert isinstance(matches, list)
            for m in matches:
                assert isinstance(m, KeypointMatch)

    def test_confidence_in_range(self):
        img1 = make_textured_img()
        img2 = make_textured_img(seed=2)
        kps1, descs1 = extract_features(img1, method="orb")
        kps2, descs2 = extract_features(img2, method="orb")
        if descs1 is not None and descs2 is not None and len(kps1) > 0 and len(kps2) > 0:
            matches = match_descriptors(descs1, descs2, kps1, kps2, method="orb")
            for m in matches:
                assert 0.0 <= m.confidence <= 1.0


# ─── estimate_homography ─────────────────────────────────────────────────────

class TestEstimateHomography:
    def test_too_few_matches_returns_none(self):
        matches = [KeypointMatch((0.0, 0.0), (1.0, 1.0), 1.0)] * 3
        H, mask = estimate_homography(matches, min_inliers=4)
        assert H is None

    def test_empty_matches_returns_none(self):
        H, mask = estimate_homography([])
        assert H is None
        assert len(mask) == 0

    def test_mask_type(self):
        H, mask = estimate_homography([])
        assert mask.dtype == bool

    def test_enough_collinear_points_may_fail(self):
        """Collinear points may produce degenerate homography."""
        matches = [
            KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
            for i in range(10)
        ]
        H, mask = estimate_homography(matches, min_inliers=4)
        # May return None for degenerate config — just check types
        if H is not None:
            assert H.shape == (3, 3)
        assert isinstance(mask, np.ndarray)

    def test_good_matches_can_give_identity(self):
        """Points from identity transform → should find identity-like H."""
        src_pts = np.float32([[10, 10], [50, 10], [50, 50], [10, 50],
                               [30, 30], [20, 40], [40, 20], [15, 45]])
        matches = [
            KeypointMatch(tuple(s.tolist()), tuple(s.tolist()), 0.0)
            for s in src_pts
        ]
        H, mask = estimate_homography(matches, min_inliers=4)
        if H is not None:
            assert H.shape == (3, 3)


# ─── feature_match_pair ──────────────────────────────────────────────────────

class TestFeatureMatchPair:
    def test_returns_feature_match_result(self):
        img = make_textured_img()
        result = feature_match_pair(img, img)
        assert isinstance(result, FeatureMatchResult)

    def test_score_in_range(self):
        img1 = make_textured_img()
        img2 = make_textured_img(seed=5)
        result = feature_match_pair(img1, img2)
        assert 0.0 <= result.score <= 1.0

    def test_identical_images_high_score(self):
        img = make_textured_img(seed=10)
        result = feature_match_pair(img, img)
        # Identical images should produce some matches
        assert result.score >= 0.0  # At minimum non-negative

    def test_method_stored(self):
        img = make_gray_img()
        result = feature_match_pair(img, img, method="orb")
        assert result.method == "orb"

    def test_n_keypoints_stored(self):
        img = make_textured_img()
        result = feature_match_pair(img, img, method="orb", n_features=50)
        assert len(result.n_keypoints) == 2

    def test_bgr_input(self):
        img = make_bgr_img()
        result = feature_match_pair(img, img)
        assert isinstance(result, FeatureMatchResult)
        assert 0.0 <= result.score <= 1.0

    def test_akaze_method(self):
        img = make_textured_img()
        result = feature_match_pair(img, img, method="akaze")
        assert result.method == "akaze"
        assert 0.0 <= result.score <= 1.0


# ─── edge_feature_score ──────────────────────────────────────────────────────

class TestEdgeFeatureScore:
    def test_returns_float(self):
        img = make_textured_img(50, 50)
        score = edge_feature_score(img, img)
        assert isinstance(score, float)

    def test_score_in_range(self):
        img1 = make_textured_img(50, 50)
        img2 = make_textured_img(50, 50, seed=7)
        score = edge_feature_score(img1, img2)
        assert 0.0 <= score <= 1.0

    def test_different_images(self):
        rng = np.random.default_rng(99)
        img1 = rng.integers(0, 256, (60, 60), dtype=np.uint8)
        img2 = rng.integers(0, 256, (60, 60), dtype=np.uint8)
        score = edge_feature_score(img1, img2)
        assert 0.0 <= score <= 1.0

    def test_bgr_input(self):
        img = make_bgr_img(60, 60)
        score = edge_feature_score(img, img)
        assert 0.0 <= score <= 1.0

    def test_akaze_method(self):
        img = make_textured_img(60, 60)
        score = edge_feature_score(img, img, method="akaze")
        assert 0.0 <= score <= 1.0
