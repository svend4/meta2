"""Extra tests for puzzle_reconstruction.matching.feature_match."""
import numpy as np
import pytest
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=100, w=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=100, w=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checker(h=120, w=120, block=15):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block):
        for j in range(0, w, block):
            if (i // block + j // block) % 2 == 0:
                img[i:i+block, j:j+block] = 200
    return img


def _kpm(pt_src=(0.0, 0.0), pt_dst=(1.0, 1.0), dist=1.0, conf=1.0):
    return KeypointMatch(pt_src=pt_src, pt_dst=pt_dst,
                         distance=dist, confidence=conf)


# ─── TestKeypointMatchExtra ───────────────────────────────────────────────────

class TestKeypointMatchExtra:
    def test_pt_src_stored(self):
        kpm = _kpm(pt_src=(3.0, 4.0))
        assert kpm.pt_src == (3.0, 4.0)

    def test_pt_dst_stored(self):
        kpm = _kpm(pt_dst=(7.0, 8.0))
        assert kpm.pt_dst == (7.0, 8.0)

    def test_distance_stored(self):
        kpm = _kpm(dist=12.5)
        assert kpm.distance == pytest.approx(12.5)

    def test_confidence_stored(self):
        kpm = _kpm(conf=0.77)
        assert kpm.confidence == pytest.approx(0.77)

    def test_default_confidence_is_one(self):
        kpm = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(1.0, 1.0), distance=5.0)
        assert kpm.confidence == pytest.approx(1.0)

    def test_repr_is_string(self):
        kpm = _kpm()
        assert isinstance(repr(kpm), str)

    def test_repr_contains_keypointmatch(self):
        kpm = _kpm()
        assert "KeypointMatch" in repr(kpm)

    def test_repr_contains_conf(self):
        kpm = _kpm(conf=0.75)
        assert "conf=0.750" in repr(kpm)

    def test_zero_distance(self):
        kpm = _kpm(dist=0.0)
        assert kpm.distance == pytest.approx(0.0)

    def test_large_distance(self):
        kpm = _kpm(dist=1e6)
        assert kpm.distance == pytest.approx(1e6)


# ─── TestFeatureMatchResultExtra ──────────────────────────────────────────────

class TestFeatureMatchResultExtra:
    def _make(self, n=5, n_in=3, score=0.5, method="orb"):
        matches = [_kpm(pt_src=(float(i), 0.0), pt_dst=(float(i), 1.0),
                        dist=float(i)) for i in range(n)]
        mask = np.array([True] * n_in + [False] * (n - n_in), dtype=bool)
        return FeatureMatchResult(
            matches=matches, homography=np.eye(3),
            inlier_mask=mask, score=score, method=method,
            n_keypoints=(100, 100),
        )

    def test_n_matches_correct(self):
        r = self._make(n=7)
        assert r.n_matches == 7

    def test_n_inliers_correct(self):
        r = self._make(n=6, n_in=4)
        assert r.n_inliers == 4

    def test_inlier_ratio_correct(self):
        r = self._make(n=8, n_in=2)
        assert r.inlier_ratio == pytest.approx(0.25)

    def test_inlier_ratio_zero_when_empty(self):
        r = FeatureMatchResult(
            matches=[], homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0, method="orb",
        )
        assert r.inlier_ratio == pytest.approx(0.0)

    def test_score_stored(self):
        r = self._make(score=0.88)
        assert r.score == pytest.approx(0.88)

    def test_method_stored(self):
        r = self._make(method="akaze")
        assert r.method == "akaze"

    def test_repr_is_string(self):
        r = self._make()
        assert isinstance(repr(r), str)

    def test_repr_contains_method(self):
        r = self._make(method="orb")
        assert "orb" in repr(r)

    def test_repr_contains_score(self):
        r = self._make(score=0.5)
        assert "score=" in repr(r)

    def test_n_keypoints_stored(self):
        r = FeatureMatchResult(
            matches=[], homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0, method="orb",
            n_keypoints=(50, 60),
        )
        assert r.n_keypoints == (50, 60)

    def test_all_inliers(self):
        r = self._make(n=4, n_in=4)
        assert r.inlier_ratio == pytest.approx(1.0)

    def test_no_inliers(self):
        r = self._make(n=4, n_in=0)
        assert r.n_inliers == 0
        assert r.inlier_ratio == pytest.approx(0.0)


# ─── TestExtractFeaturesExtra ─────────────────────────────────────────────────

class TestExtractFeaturesExtra:
    def test_returns_tuple(self):
        img = _gray()
        result = extract_features(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_kps_is_list(self):
        img = _checker()
        kps, _ = extract_features(img, method="orb")
        assert isinstance(kps, list)

    def test_orb_gray(self):
        kps, descs = extract_features(_checker(), method="orb")
        assert isinstance(kps, list)

    def test_orb_bgr(self):
        kps, descs = extract_features(_bgr(), method="orb")
        assert isinstance(kps, list)

    def test_akaze_gray(self):
        kps, descs = extract_features(_checker(), method="akaze")
        assert isinstance(kps, list)

    def test_unknown_method_returns_empty(self):
        kps, descs = extract_features(_gray(), method="totally_unknown_xyz")
        assert kps == []
        assert descs is None

    def test_n_features_limit(self):
        kps, _ = extract_features(_checker(200, 200), method="orb", n_features=10)
        assert len(kps) <= 10

    def test_with_mask(self):
        img = _checker()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[10:90, 10:90] = 255
        kps, _ = extract_features(img, method="orb", mask=mask)
        assert isinstance(kps, list)

    def test_small_image(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        kps, descs = extract_features(img, method="orb")
        assert isinstance(kps, list)

    def test_descs_none_or_array(self):
        _, descs = extract_features(_checker(), method="orb")
        assert descs is None or isinstance(descs, np.ndarray)


# ─── TestMatchDescriptorsExtra ────────────────────────────────────────────────

class TestMatchDescriptorsExtra:
    def test_none_desc1_empty(self):
        result = match_descriptors(None, np.zeros((5, 32), dtype=np.uint8), [], [])
        assert result == []

    def test_none_desc2_empty(self):
        result = match_descriptors(np.zeros((5, 32), dtype=np.uint8), None, [], [])
        assert result == []

    def test_empty_desc_empty(self):
        result = match_descriptors(
            np.zeros((0, 32), dtype=np.uint8),
            np.zeros((0, 32), dtype=np.uint8),
            [], [],
        )
        assert result == []

    def test_returns_list_type(self):
        img1 = _checker()
        img2 = _checker(block=10)
        kps1, d1 = extract_features(img1, method="orb")
        kps2, d2 = extract_features(img2, method="orb")
        if d1 is not None and d2 is not None:
            result = match_descriptors(d1, d2, kps1, kps2, method="orb")
            assert isinstance(result, list)

    def test_keypoint_matches_type(self):
        img = _checker()
        kps, descs = extract_features(img, method="orb")
        if descs is not None and len(kps) > 0:
            result = match_descriptors(descs, descs, kps, kps, method="orb")
            for m in result:
                assert isinstance(m, KeypointMatch)

    def test_confidence_in_range(self):
        img = _checker()
        kps, descs = extract_features(img, method="orb")
        if descs is not None and len(kps) > 1:
            result = match_descriptors(descs, descs, kps, kps, method="orb")
            for m in result:
                assert 0.0 <= m.confidence <= 1.0


# ─── TestEstimateHomographyExtra ──────────────────────────────────────────────

class TestEstimateHomographyExtra:
    def test_empty_returns_none(self):
        H, mask = estimate_homography([])
        assert H is None

    def test_empty_mask_bool(self):
        _, mask = estimate_homography([])
        assert mask.dtype == bool

    def test_too_few_points_none(self):
        matches = [_kpm() for _ in range(3)]
        H, mask = estimate_homography(matches, min_inliers=4)
        assert H is None

    def test_identity_points_shape(self):
        pts = np.float32([[10, 10], [50, 10], [50, 50], [10, 50],
                          [30, 30], [20, 40], [40, 20], [15, 45]])
        matches = [KeypointMatch(tuple(p.tolist()), tuple(p.tolist()), 0.0)
                   for p in pts]
        H, mask = estimate_homography(matches, min_inliers=4)
        if H is not None:
            assert H.shape == (3, 3)

    def test_mask_is_ndarray(self):
        _, mask = estimate_homography([])
        assert isinstance(mask, np.ndarray)

    def test_collinear_may_fail(self):
        matches = [KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
                   for i in range(8)]
        H, mask = estimate_homography(matches, min_inliers=4)
        if H is not None:
            assert H.shape == (3, 3)


# ─── TestFeatureMatchPairExtra ────────────────────────────────────────────────

class TestFeatureMatchPairExtra:
    def test_returns_feature_match_result(self):
        img = _checker()
        result = feature_match_pair(img, img)
        assert isinstance(result, FeatureMatchResult)

    def test_score_in_0_1(self):
        result = feature_match_pair(_checker(), _checker())
        assert 0.0 <= result.score <= 1.0

    def test_score_nonneg(self):
        result = feature_match_pair(_gray(), _gray(seed=99))
        assert result.score >= 0.0

    def test_method_orb_stored(self):
        result = feature_match_pair(_checker(), _checker(), method="orb")
        assert result.method == "orb"

    def test_method_akaze_stored(self):
        result = feature_match_pair(_checker(), _checker(), method="akaze")
        assert result.method == "akaze"

    def test_n_keypoints_two_elements(self):
        result = feature_match_pair(_checker(), _checker())
        assert len(result.n_keypoints) == 2

    def test_bgr_input(self):
        result = feature_match_pair(_bgr(), _bgr())
        assert isinstance(result, FeatureMatchResult)
        assert 0.0 <= result.score <= 1.0

    def test_different_sized_images(self):
        img1 = _checker(100, 100)
        img2 = _checker(80, 80)
        result = feature_match_pair(img1, img2)
        assert isinstance(result, FeatureMatchResult)

    def test_inlier_mask_bool(self):
        result = feature_match_pair(_checker(), _checker())
        assert result.inlier_mask.dtype == bool


# ─── TestEdgeFeatureScoreExtra ────────────────────────────────────────────────

class TestEdgeFeatureScoreExtra:
    def test_returns_float(self):
        img = _checker(60, 60)
        score = edge_feature_score(img, img)
        assert isinstance(score, float)

    def test_score_in_range(self):
        score = edge_feature_score(_checker(50, 50), _checker(50, 50))
        assert 0.0 <= score <= 1.0

    def test_score_nonneg_different_images(self):
        score = edge_feature_score(_gray(60, 60, seed=1),
                                   _gray(60, 60, seed=2))
        assert score >= 0.0

    def test_bgr_input(self):
        score = edge_feature_score(_bgr(60, 60), _bgr(60, 60))
        assert 0.0 <= score <= 1.0

    def test_orb_method(self):
        score = edge_feature_score(_checker(50, 50), _checker(50, 50),
                                   method="orb")
        assert 0.0 <= score <= 1.0

    def test_akaze_method(self):
        score = edge_feature_score(_checker(60, 60), _checker(60, 60),
                                   method="akaze")
        assert 0.0 <= score <= 1.0

    def test_identical_images_nonneg(self):
        img = _checker(60, 60)
        score = edge_feature_score(img, img)
        assert score >= 0.0

    def test_totally_different_images(self):
        img1 = np.zeros((60, 60), dtype=np.uint8)
        img2 = np.full((60, 60), 255, dtype=np.uint8)
        score = edge_feature_score(img1, img2)
        assert 0.0 <= score <= 1.0
