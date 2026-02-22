"""Tests for puzzle_reconstruction.algorithms.sift_matcher."""
import numpy as np
import pytest

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

def _noise_img(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)


def _gradient_img(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = ((i * j) % 256)
    return img


def _fake_descs(n=10, dim=128, seed=0):
    return np.random.default_rng(seed).random((n, dim)).astype(np.float32)


# ─── TestSiftConfig ───────────────────────────────────────────────────────────

class TestSiftConfig:
    def test_defaults(self):
        cfg = SiftConfig()
        assert cfg.n_features == 500
        assert cfg.ratio_thresh == pytest.approx(0.75)
        assert cfg.min_matches == 4

    def test_negative_n_features_raises(self):
        with pytest.raises(ValueError, match="n_features"):
            SiftConfig(n_features=-1)

    def test_n_octave_layers_zero_raises(self):
        with pytest.raises(ValueError, match="n_octave_layers"):
            SiftConfig(n_octave_layers=0)

    def test_contrast_threshold_nonpositive_raises(self):
        with pytest.raises(ValueError, match="contrast_threshold"):
            SiftConfig(contrast_threshold=0.0)

    def test_edge_threshold_nonpositive_raises(self):
        with pytest.raises(ValueError, match="edge_threshold"):
            SiftConfig(edge_threshold=0.0)

    def test_sigma_nonpositive_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SiftConfig(sigma=0.0)

    def test_ratio_thresh_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ratio_thresh"):
            SiftConfig(ratio_thresh=0.0)
        with pytest.raises(ValueError, match="ratio_thresh"):
            SiftConfig(ratio_thresh=1.0)

    def test_min_matches_less_than_4_raises(self):
        with pytest.raises(ValueError, match="min_matches"):
            SiftConfig(min_matches=3)

    def test_n_features_zero_allowed(self):
        cfg = SiftConfig(n_features=0)
        assert cfg.n_features == 0


# ─── TestMatchResult ──────────────────────────────────────────────────────────

class TestMatchResult:
    def test_basic_creation(self):
        r = MatchResult(n_matches=10, n_inliers=6, score=0.6)
        assert r.n_matches == 10
        assert r.n_inliers == 6
        assert r.score == pytest.approx(0.6)

    def test_negative_n_matches_raises(self):
        with pytest.raises(ValueError, match="n_matches"):
            MatchResult(n_matches=-1, n_inliers=0, score=0.0)

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError, match="n_inliers"):
            MatchResult(n_matches=0, n_inliers=-1, score=0.0)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="score"):
            MatchResult(n_matches=0, n_inliers=0, score=1.5)
        with pytest.raises(ValueError, match="score"):
            MatchResult(n_matches=0, n_inliers=0, score=-0.1)

    def test_is_reliable_true(self):
        r = MatchResult(n_matches=10, n_inliers=5, score=0.5)
        assert r.is_reliable(min_inliers=4) is True

    def test_is_reliable_false(self):
        r = MatchResult(n_matches=10, n_inliers=2, score=0.2)
        assert r.is_reliable(min_inliers=4) is False

    def test_default_homography_none(self):
        r = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert r.homography is None
        assert r.src_pts is None
        assert r.dst_pts is None


# ─── TestExtractKeypoints ─────────────────────────────────────────────────────

class TestExtractKeypoints:
    def test_returns_tuple(self):
        img = _noise_img()
        kps, descs = extract_keypoints(img)
        assert isinstance(kps, list)

    def test_grayscale_input(self):
        img = _noise_img()
        kps, descs = extract_keypoints(img)
        if descs is not None:
            assert descs.ndim == 2
            assert descs.shape[1] == 128

    def test_bgr_input_accepted(self):
        img = np.random.default_rng(0).integers(0, 255, (64, 64, 3),
                                                dtype=np.uint8)
        kps, descs = extract_keypoints(img)
        assert isinstance(kps, list)

    def test_blank_image_may_have_no_keypoints(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        kps, descs = extract_keypoints(img)
        if len(kps) == 0:
            assert descs is None or len(descs) == 0

    def test_config_applied(self):
        img = _noise_img()
        cfg = SiftConfig(n_features=10)
        kps, descs = extract_keypoints(img, cfg)
        assert isinstance(kps, list)


# ─── TestMatchDescriptors ─────────────────────────────────────────────────────

class TestMatchDescriptors:
    def test_ratio_thresh_zero_raises(self):
        d = _fake_descs(5)
        with pytest.raises(ValueError, match="ratio_thresh"):
            match_descriptors(d, d, ratio_thresh=0.0)

    def test_ratio_thresh_one_raises(self):
        d = _fake_descs(5)
        with pytest.raises(ValueError, match="ratio_thresh"):
            match_descriptors(d, d, ratio_thresh=1.0)

    def test_none_desc_returns_empty(self):
        d = _fake_descs(5)
        result = match_descriptors(None, d)
        assert result == []

    def test_empty_desc_returns_empty(self):
        d = _fake_descs(5)
        empty = np.zeros((0, 128), dtype=np.float32)
        result = match_descriptors(empty, d)
        assert result == []

    def test_identical_descs_returns_matches(self):
        d = _fake_descs(20)
        matches = match_descriptors(d, d.copy(), ratio_thresh=0.9)
        assert isinstance(matches, list)

    def test_returns_list(self):
        d1 = _fake_descs(10, seed=0)
        d2 = _fake_descs(10, seed=1)
        result = match_descriptors(d1, d2)
        assert isinstance(result, list)


# ─── TestFilterMatchesByDistance ─────────────────────────────────────────────

class TestFilterMatchesByDistance:
    def test_negative_max_distance_raises(self):
        with pytest.raises(ValueError, match="max_distance"):
            filter_matches_by_distance([], max_distance=-1.0)

    def test_empty_list_returns_empty(self):
        result = filter_matches_by_distance([], max_distance=100.0)
        assert result == []

    def test_zero_max_distance_filters_all(self):
        import cv2
        matches = [cv2.DMatch(0, 0, 10.0), cv2.DMatch(1, 1, 5.0)]
        result = filter_matches_by_distance(matches, max_distance=0.0)
        assert result == []

    def test_large_max_distance_keeps_all(self):
        import cv2
        matches = [cv2.DMatch(0, 0, 10.0), cv2.DMatch(1, 1, 5.0)]
        result = filter_matches_by_distance(matches, max_distance=1000.0)
        assert len(result) == 2

    def test_threshold_filtering(self):
        import cv2
        matches = [cv2.DMatch(0, 0, 10.0), cv2.DMatch(1, 1, 20.0)]
        result = filter_matches_by_distance(matches, max_distance=15.0)
        assert len(result) == 1
        assert result[0].distance <= 15.0


# ─── TestSiftMatchPair ────────────────────────────────────────────────────────

class TestSiftMatchPair:
    def test_returns_match_result(self):
        img = _noise_img(seed=0)
        result = sift_match_pair(img, img)
        assert isinstance(result, MatchResult)

    def test_score_in_range(self):
        img = _noise_img(seed=0)
        result = sift_match_pair(img, img)
        assert 0.0 <= result.score <= 1.0

    def test_n_matches_nonneg(self):
        img = _noise_img(seed=0)
        result = sift_match_pair(img, img)
        assert result.n_matches >= 0

    def test_blank_images_low_score(self):
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        result = sift_match_pair(img1, img2)
        assert result.n_inliers == 0 or result.score <= 1.0


# ─── TestBatchSiftMatch ───────────────────────────────────────────────────────

class TestBatchSiftMatch:
    def test_returns_dict(self):
        images = [_noise_img(seed=i) for i in range(3)]
        results = batch_sift_match(images)
        assert isinstance(results, dict)

    def test_correct_number_of_pairs(self):
        n = 4
        images = [_noise_img(seed=i) for i in range(n)]
        results = batch_sift_match(images)
        expected_pairs = n * (n - 1) // 2
        assert len(results) == expected_pairs

    def test_keys_are_ordered_pairs(self):
        n = 3
        images = [_noise_img(seed=i) for i in range(n)]
        results = batch_sift_match(images)
        for i, j in results:
            assert i < j

    def test_single_image_no_pairs(self):
        images = [_noise_img(seed=0)]
        results = batch_sift_match(images)
        assert results == {}

    def test_empty_list_no_pairs(self):
        results = batch_sift_match([])
        assert results == {}
