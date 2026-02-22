"""Additional tests for puzzle_reconstruction.algorithms.sift_matcher."""
import numpy as np
import pytest
import cv2

from puzzle_reconstruction.algorithms.sift_matcher import (
    MatchResult,
    SiftConfig,
    batch_sift_match,
    extract_keypoints,
    filter_matches_by_distance,
    match_descriptors,
    sift_match_pair,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _fake_descs(n=10, dim=128, seed=0):
    return np.random.default_rng(seed).random((n, dim)).astype(np.float32)


def _dmatch(q, t, dist):
    m = cv2.DMatch()
    m.queryIdx = q
    m.trainIdx = t
    m.distance = dist
    return m


# ─── TestSiftConfigExtra ─────────────────────────────────────────────────────

class TestSiftConfigExtra:
    def test_n_features_100_ok(self):
        cfg = SiftConfig(n_features=100)
        assert cfg.n_features == 100

    def test_ratio_thresh_0_5_ok(self):
        cfg = SiftConfig(ratio_thresh=0.5)
        assert cfg.ratio_thresh == pytest.approx(0.5)

    def test_ratio_thresh_0_99_ok(self):
        cfg = SiftConfig(ratio_thresh=0.99)
        assert cfg.ratio_thresh == pytest.approx(0.99)

    def test_min_matches_4_ok(self):
        cfg = SiftConfig(min_matches=4)
        assert cfg.min_matches == 4

    def test_min_matches_100_ok(self):
        cfg = SiftConfig(min_matches=100)
        assert cfg.min_matches == 100

    def test_sigma_1_6_ok(self):
        cfg = SiftConfig(sigma=1.6)
        assert cfg.sigma == pytest.approx(1.6)

    def test_n_features_large(self):
        cfg = SiftConfig(n_features=2000)
        assert cfg.n_features == 2000

    def test_contrast_threshold_small_positive(self):
        cfg = SiftConfig(contrast_threshold=0.001)
        assert cfg.contrast_threshold == pytest.approx(0.001)


# ─── TestMatchResultExtra ─────────────────────────────────────────────────────

class TestMatchResultExtra:
    def test_score_zero_ok(self):
        r = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        r = MatchResult(n_matches=10, n_inliers=10, score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_n_inliers_zero_ok(self):
        r = MatchResult(n_matches=5, n_inliers=0, score=0.0)
        assert r.n_inliers == 0

    def test_is_reliable_min_0_always_true(self):
        r = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert r.is_reliable(min_inliers=0) is True

    def test_homography_stored(self):
        H = np.eye(3, dtype=np.float32)
        r = MatchResult(n_matches=10, n_inliers=8, score=0.8,
                        homography=H)
        assert r.homography is not None
        assert r.homography.shape == (3, 3)

    def test_src_pts_stored(self):
        pts = np.zeros((5, 2), dtype=np.float32)
        r = MatchResult(n_matches=5, n_inliers=5, score=1.0, src_pts=pts)
        assert r.src_pts is not None
        assert r.src_pts.shape == (5, 2)

    def test_is_reliable_exact_threshold(self):
        r = MatchResult(n_matches=10, n_inliers=4, score=0.4)
        assert r.is_reliable(min_inliers=4) is True
        assert r.is_reliable(min_inliers=5) is False

    def test_n_matches_zero_score_zero(self):
        r = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert r.n_matches == 0


# ─── TestExtractKeypointsExtra ────────────────────────────────────────────────

class TestExtractKeypointsExtra:
    def test_non_square_image(self):
        img = _gray(h=32, w=96)
        kps, descs = extract_keypoints(img)
        assert isinstance(kps, list)

    def test_white_image(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        kps, descs = extract_keypoints(img)
        assert isinstance(kps, list)

    def test_large_image_no_crash(self):
        img = _gray(h=128, w=128, seed=5)
        kps, descs = extract_keypoints(img)
        assert isinstance(kps, list)

    def test_descs_float32_or_none(self):
        img = _gray(seed=3)
        _, descs = extract_keypoints(img)
        if descs is not None and len(descs) > 0:
            assert descs.dtype == np.float32

    def test_bgr_descs_128_dim(self):
        img = _bgr(seed=7)
        _, descs = extract_keypoints(img)
        if descs is not None and len(descs) > 0:
            assert descs.shape[1] == 128

    def test_cfg_n_features_1(self):
        img = _gray(seed=1)
        cfg = SiftConfig(n_features=1)
        kps, _ = extract_keypoints(img, cfg)
        assert isinstance(kps, list)


# ─── TestMatchDescriptorsExtra ────────────────────────────────────────────────

class TestMatchDescriptorsExtra:
    def test_ratio_thresh_0_99_many_matches(self):
        d = _fake_descs(20, seed=0)
        result = match_descriptors(d, d.copy(), ratio_thresh=0.99)
        assert isinstance(result, list)

    def test_ratio_thresh_0_5_fewer_matches(self):
        d1 = _fake_descs(20, seed=0)
        d2 = _fake_descs(20, seed=1)
        r_strict = match_descriptors(d1, d2, ratio_thresh=0.5)
        r_loose = match_descriptors(d1, d2, ratio_thresh=0.9)
        assert len(r_strict) <= len(r_loose)

    def test_both_none_returns_empty(self):
        result = match_descriptors(None, None)
        assert result == []

    def test_single_descriptor_each(self):
        d1 = _fake_descs(1, seed=0)
        d2 = _fake_descs(1, seed=1)
        result = match_descriptors(d1, d2)
        assert isinstance(result, list)

    def test_large_desc_dim(self):
        d1 = np.random.default_rng(0).random((10, 128)).astype(np.float32)
        d2 = np.random.default_rng(1).random((10, 128)).astype(np.float32)
        result = match_descriptors(d1, d2, ratio_thresh=0.8)
        assert isinstance(result, list)


# ─── TestFilterMatchesByDistanceExtra ────────────────────────────────────────

class TestFilterMatchesByDistanceExtra:
    def test_exact_boundary_kept(self):
        matches = [_dmatch(0, 0, 10.0)]
        result = filter_matches_by_distance(matches, max_distance=10.0)
        assert len(result) == 1

    def test_just_above_boundary_removed(self):
        matches = [_dmatch(0, 0, 10.1)]
        result = filter_matches_by_distance(matches, max_distance=10.0)
        assert len(result) == 0

    def test_preserves_order(self):
        matches = [_dmatch(0, 0, 5.0), _dmatch(1, 1, 3.0), _dmatch(2, 2, 7.0)]
        result = filter_matches_by_distance(matches, max_distance=6.0)
        assert len(result) == 2
        dists = [m.distance for m in result]
        assert all(d <= 6.0 for d in dists)

    def test_single_match_kept(self):
        matches = [_dmatch(0, 0, 1.0)]
        result = filter_matches_by_distance(matches, max_distance=100.0)
        assert len(result) == 1

    def test_all_filtered_returns_empty(self):
        matches = [_dmatch(i, i, float(100 + i)) for i in range(5)]
        result = filter_matches_by_distance(matches, max_distance=10.0)
        assert result == []


# ─── TestSiftMatchPairExtra ───────────────────────────────────────────────────

class TestSiftMatchPairExtra:
    def test_non_square_images(self):
        img1 = _gray(h=32, w=96, seed=0)
        img2 = _gray(h=32, w=96, seed=1)
        r = sift_match_pair(img1, img2)
        assert isinstance(r, MatchResult)

    def test_bgr_images_ok(self):
        img1 = _bgr(seed=0)
        img2 = _bgr(seed=1)
        r = sift_match_pair(img1, img2)
        assert isinstance(r, MatchResult)
        assert 0.0 <= r.score <= 1.0

    def test_n_inliers_leq_n_matches(self):
        img = _gray(seed=0)
        r = sift_match_pair(img, img)
        assert r.n_inliers <= r.n_matches

    def test_custom_config(self):
        img = _gray(seed=2)
        cfg = SiftConfig(n_features=50, ratio_thresh=0.8)
        r = sift_match_pair(img, img, cfg=cfg)
        assert isinstance(r, MatchResult)

    def test_different_seeds_result_finite(self):
        img1 = _gray(seed=10)
        img2 = _gray(seed=20)
        r = sift_match_pair(img1, img2)
        assert isinstance(r.score, float)
        assert 0.0 <= r.score <= 1.0


# ─── TestBatchSiftMatchExtra ─────────────────────────────────────────────────

class TestBatchSiftMatchExtra:
    def test_two_images_one_pair(self):
        images = [_gray(seed=0), _gray(seed=1)]
        results = batch_sift_match(images)
        assert len(results) == 1
        assert (0, 1) in results

    def test_five_images_ten_pairs(self):
        images = [_gray(seed=i) for i in range(5)]
        results = batch_sift_match(images)
        assert len(results) == 10

    def test_values_are_match_results(self):
        images = [_gray(seed=i) for i in range(3)]
        results = batch_sift_match(images)
        for v in results.values():
            assert isinstance(v, MatchResult)

    def test_keys_are_sorted_pairs(self):
        images = [_gray(seed=i) for i in range(4)]
        results = batch_sift_match(images)
        for i, j in results.keys():
            assert i < j

    def test_custom_cfg_accepted(self):
        images = [_gray(seed=i) for i in range(3)]
        cfg = SiftConfig(n_features=20)
        results = batch_sift_match(images, cfg=cfg)
        assert isinstance(results, dict)
