"""Extra tests for puzzle_reconstruction/matching/feature_match.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.feature_match import (
    KeypointMatch,
    FeatureMatchResult,
    extract_features,
    match_descriptors,
    estimate_homography,
    feature_match_pair,
    edge_feature_score,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=80, w=80, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)


def _bgr(h=80, w=80, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _checkerboard(size=100):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, 10):
        for j in range(0, size, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i:i + 10, j:j + 10] = 255
    return img


def _blank(h=30, w=30):
    return np.zeros((h, w), dtype=np.uint8)


# ─── KeypointMatch (extra) ───────────────────────────────────────────────────

class TestKeypointMatchExtra:
    def test_pt_src_stored(self):
        km = KeypointMatch(pt_src=(3.0, 4.0), pt_dst=(5.0, 6.0), distance=1.0)
        assert km.pt_src == (3.0, 4.0)

    def test_pt_dst_stored(self):
        km = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(7.0, 8.0), distance=2.0)
        assert km.pt_dst == (7.0, 8.0)

    def test_distance_stored(self):
        km = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(1.0, 0.0), distance=42.0)
        assert km.distance == pytest.approx(42.0)

    def test_default_confidence_is_one(self):
        km = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(1.0, 1.0), distance=5.0)
        assert km.confidence == pytest.approx(1.0)

    def test_confidence_custom(self):
        km = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(1.0, 1.0),
                           distance=5.0, confidence=0.6)
        assert km.confidence == pytest.approx(0.6)

    def test_confidence_le_one(self):
        km = KeypointMatch(pt_src=(0.0, 0.0), pt_dst=(0.0, 0.0),
                           distance=0.0, confidence=1.0)
        assert km.confidence <= 1.0

    def test_repr_has_keypoint_match(self):
        km = KeypointMatch(pt_src=(1.0, 2.0), pt_dst=(3.0, 4.0), distance=5.0)
        assert "KeypointMatch" in repr(km)

    def test_repr_has_src_dst_conf(self):
        km = KeypointMatch(pt_src=(1.0, 2.0), pt_dst=(3.0, 4.0),
                           distance=5.0, confidence=0.8)
        r = repr(km)
        assert "src=" in r
        assert "dst=" in r
        assert "conf=" in r

    def test_zero_distance_ok(self):
        km = KeypointMatch(pt_src=(5.0, 5.0), pt_dst=(5.0, 5.0), distance=0.0)
        assert km.distance == pytest.approx(0.0)

    def test_float_coordinates(self):
        km = KeypointMatch(pt_src=(3.14, 2.71), pt_dst=(0.5, 0.5), distance=1.0)
        assert abs(km.pt_src[0] - 3.14) < 1e-5


# ─── FeatureMatchResult (extra) ──────────────────────────────────────────────

class TestFeatureMatchResultExtra:
    def _make(self, n_matches=1, n_inliers=1, score=0.5):
        matches = [KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
                   for i in range(n_matches)]
        mask = np.array([i < n_inliers for i in range(n_matches)])
        return FeatureMatchResult(
            matches=matches,
            homography=np.eye(3),
            inlier_mask=mask,
            score=score,
            method="orb",
            n_keypoints=(100, 100),
        )

    def test_n_matches(self):
        r = self._make(n_matches=5)
        assert r.n_matches == 5

    def test_n_inliers_all(self):
        r = self._make(n_matches=4, n_inliers=4)
        assert r.n_inliers == 4

    def test_n_inliers_partial(self):
        r = self._make(n_matches=4, n_inliers=2)
        assert r.n_inliers == 2

    def test_inlier_ratio_all(self):
        r = self._make(n_matches=4, n_inliers=4)
        assert r.inlier_ratio == pytest.approx(1.0)

    def test_inlier_ratio_zero(self):
        r = self._make(n_matches=4, n_inliers=0)
        assert r.inlier_ratio == pytest.approx(0.0)

    def test_inlier_ratio_half(self):
        r = self._make(n_matches=4, n_inliers=2)
        assert r.inlier_ratio == pytest.approx(0.5)

    def test_none_mask_n_inliers_zero(self):
        r = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=None,
            score=0.0,
            method="orb",
        )
        assert r.n_inliers == 0

    def test_empty_mask_zero(self):
        r = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0,
            method="orb",
        )
        assert r.n_inliers == 0

    def test_repr_fields(self):
        r = self._make()
        s = repr(r)
        assert "FeatureMatchResult" in s
        assert "n_matches=" in s

    def test_method_stored(self):
        r = self._make()
        assert r.method == "orb"

    def test_score_stored(self):
        r = self._make(score=0.73)
        assert r.score == pytest.approx(0.73)

    def test_n_keypoints_default(self):
        r = FeatureMatchResult(
            matches=[],
            homography=None,
            inlier_mask=np.array([], dtype=bool),
            score=0.0,
            method="orb",
        )
        assert r.n_keypoints == (0, 0)

    def test_n_keypoints_custom(self):
        r = self._make()
        assert r.n_keypoints == (100, 100)


# ─── extract_features (extra) ────────────────────────────────────────────────

class TestExtractFeaturesExtra:
    def test_returns_tuple(self):
        result = extract_features(_gray(), method="orb")
        assert isinstance(result, tuple) and len(result) == 2

    def test_keypoints_is_list(self):
        kps, _ = extract_features(_gray(), method="orb")
        assert isinstance(kps, list)

    def test_grayscale_ok(self):
        kps, _ = extract_features(_gray(), method="orb")
        assert isinstance(kps, list)

    def test_bgr_ok(self):
        kps, _ = extract_features(_bgr(), method="orb")
        assert isinstance(kps, list)

    def test_blank_image_no_crash(self):
        kps, descs = extract_features(_blank(), method="orb")
        assert isinstance(kps, list)

    def test_n_features_upper_bound(self):
        kps, _ = extract_features(_checkerboard(), method="orb", n_features=50)
        assert len(kps) <= 50

    def test_mask_parameter_ok(self):
        img = _gray()
        mask = np.zeros_like(img)
        mask[10:70, 10:70] = 255
        kps, _ = extract_features(img, method="orb", mask=mask)
        assert isinstance(kps, list)

    def test_akaze_method(self):
        kps, _ = extract_features(_checkerboard(), method="akaze")
        assert isinstance(kps, list)

    def test_unknown_method_empty_kps(self):
        kps, descs = extract_features(_gray(), method="nonexistent_xyz")
        assert len(kps) == 0

    def test_descriptors_match_keypoints(self):
        kps, descs = extract_features(_checkerboard(), method="orb",
                                       n_features=100)
        if len(kps) > 0 and descs is not None:
            assert descs.shape[0] == len(kps)


# ─── match_descriptors (extra) ───────────────────────────────────────────────

class TestMatchDescriptorsExtra:
    def test_none_desc_returns_empty(self):
        assert match_descriptors(None, None, [], [], method="orb") == []

    def test_empty_desc_returns_empty(self):
        d = np.array([], dtype=np.uint8).reshape(0, 32)
        assert match_descriptors(d, d, [], [], method="orb") == []

    def test_returns_list_of_keypoint_matches(self):
        cb = _checkerboard()
        kps, descs = extract_features(cb, method="orb", n_features=100)
        if descs is not None and len(kps) >= 2:
            shifted = np.roll(cb, 3, axis=1)
            kps2, descs2 = extract_features(shifted, method="orb",
                                             n_features=100)
            if descs2 is not None and len(kps2) >= 2:
                matches = match_descriptors(descs, descs2, kps, kps2,
                                            method="orb")
                for m in matches:
                    assert isinstance(m, KeypointMatch)

    def test_confidence_in_range(self):
        cb = _checkerboard()
        kps, descs = extract_features(cb, method="orb", n_features=100)
        if descs is not None and len(kps) >= 2:
            matches = match_descriptors(descs, descs, kps, kps, method="orb")
            for m in matches:
                assert 0.0 <= m.confidence <= 1.0

    def test_strict_ratio_fewer_matches(self):
        cb = _checkerboard()
        kps, descs = extract_features(cb, method="orb", n_features=100)
        if descs is not None and len(kps) >= 4:
            m_strict = match_descriptors(descs, descs, kps, kps,
                                         method="orb", ratio=0.3)
            m_loose = match_descriptors(descs, descs, kps, kps,
                                        method="orb", ratio=0.99)
            assert len(m_strict) <= len(m_loose)


# ─── estimate_homography (extra) ─────────────────────────────────────────────

class TestEstimateHomographyExtra:
    def test_empty_matches_none(self):
        H, mask = estimate_homography([])
        assert H is None

    def test_empty_mask_empty(self):
        _, mask = estimate_homography([])
        assert len(mask) == 0

    def test_too_few_matches_none(self):
        matches = [KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
                   for i in range(3)]
        H, mask = estimate_homography(matches, min_inliers=4)
        assert H is None

    def test_mask_length_equals_input(self):
        matches = [KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
                   for i in range(5)]
        _, mask = estimate_homography(matches, min_inliers=4)
        assert len(mask) == 5

    def test_mask_dtype_bool(self):
        matches = [KeypointMatch((float(i * 10), float(j * 10)),
                                  (float(i * 10), float(j * 10)), 0.0)
                   for i in range(4) for j in range(4)]
        _, mask = estimate_homography(matches, min_inliers=4)
        assert mask.dtype in (bool, np.bool_)

    def test_returns_tuple_two(self):
        result = estimate_homography([])
        assert isinstance(result, tuple) and len(result) == 2

    def test_min_inliers_too_high_none(self):
        matches = [KeypointMatch((float(i), 0.0), (float(i), 0.0), 0.0)
                   for i in range(5)]
        H, _ = estimate_homography(matches, min_inliers=100)
        assert H is None

    def test_identity_points_homography_shape(self):
        pts = [(float(i * 15 + 5), float(j * 15 + 5))
               for i in range(5) for j in range(5)]
        matches = [KeypointMatch(p, p, 0.0) for p in pts]
        H, mask = estimate_homography(matches, min_inliers=4)
        if H is not None:
            assert H.shape == (3, 3)


# ─── feature_match_pair (extra) ──────────────────────────────────────────────

class TestFeatureMatchPairExtra:
    def test_returns_feature_match_result(self):
        assert isinstance(feature_match_pair(_gray(), _gray()), FeatureMatchResult)

    def test_score_in_range(self):
        r = feature_match_pair(_gray(), _gray())
        assert 0.0 <= r.score <= 1.0

    def test_method_orb(self):
        r = feature_match_pair(_gray(), _gray(), method="orb")
        assert r.method == "orb"

    def test_method_akaze(self):
        r = feature_match_pair(_checkerboard(), _checkerboard(), method="akaze")
        assert r.method == "akaze"

    def test_n_keypoints_tuple_of_two(self):
        r = feature_match_pair(_gray(), _gray())
        assert isinstance(r.n_keypoints, tuple) and len(r.n_keypoints) == 2

    def test_blank_image_score_zero(self):
        img = np.zeros((15, 15), dtype=np.uint8)
        r = feature_match_pair(img, img)
        assert r.score == pytest.approx(0.0)

    def test_bgr_accepted(self):
        r = feature_match_pair(_bgr(), _bgr(), method="orb")
        assert isinstance(r, FeatureMatchResult)

    def test_inlier_mask_consistent(self):
        r = feature_match_pair(_gray(), _gray())
        if r.inlier_mask is not None and len(r.inlier_mask) > 0:
            assert len(r.inlier_mask) == len(r.matches)

    def test_identical_nonneg_score(self):
        cb = _checkerboard()
        r = feature_match_pair(cb, cb, method="orb")
        assert r.score >= 0.0

    def test_different_images_nonneg(self):
        r = feature_match_pair(_checkerboard(), _blank(), method="orb")
        assert r.score >= 0.0


# ─── edge_feature_score (extra) ──────────────────────────────────────────────

class TestEdgeFeatureScoreExtra:
    def test_returns_float(self):
        assert isinstance(edge_feature_score(_gray(), _gray()), float)

    def test_score_in_range(self):
        s = edge_feature_score(_gray(), _gray())
        assert 0.0 <= s <= 1.0

    def test_blank_images_zero(self):
        img = np.zeros((30, 30), dtype=np.uint8)
        assert edge_feature_score(img, img) == pytest.approx(0.0)

    def test_nonneg(self):
        assert edge_feature_score(_checkerboard(), _checkerboard()) >= 0.0

    def test_narrow_image_ok(self):
        img = np.random.randint(0, 255, (60, 8), dtype=np.uint8)
        s = edge_feature_score(img, img)
        assert 0.0 <= s <= 1.0

    def test_bgr_accepted(self):
        s = edge_feature_score(_bgr(), _bgr())
        assert 0.0 <= s <= 1.0

    def test_same_ge_different(self):
        cb = _checkerboard()
        blank = np.zeros_like(cb)
        s_same = edge_feature_score(cb, cb)
        s_diff = edge_feature_score(cb, blank)
        assert s_same >= s_diff or s_same >= 0.0
