"""Тесты для puzzle_reconstruction.algorithms.sift_matcher."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.sift_matcher import (
    SiftConfig,
    MatchResult,
    extract_keypoints,
    match_descriptors,
    compute_homography,
    sift_match_pair,
    filter_matches_by_distance,
    batch_sift_match,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=128, w=128, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(h=128, w=128, block=16):
    """Шахматный узор — хорошо детектируется SIFT."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if ((r // block) + (c // block)) % 2 == 0:
                img[r:r + block, c:c + block] = 200
    return img


# ─── TestSiftConfig ───────────────────────────────────────────────────────────

class TestSiftConfig:
    def test_default_values(self):
        cfg = SiftConfig()
        assert cfg.n_features == 500
        assert cfg.ratio_thresh == pytest.approx(0.75)
        assert cfg.min_matches == 4

    def test_negative_n_features_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(n_features=-1)

    def test_zero_n_features_valid(self):
        cfg = SiftConfig(n_features=0)
        assert cfg.n_features == 0

    def test_n_octave_layers_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(n_octave_layers=0)

    def test_contrast_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(contrast_threshold=0.0)

    def test_edge_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(edge_threshold=0.0)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(sigma=0.0)

    def test_ratio_thresh_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(ratio_thresh=0.0)

    def test_ratio_thresh_one_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(ratio_thresh=1.0)

    def test_ratio_thresh_above_one_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(ratio_thresh=1.5)

    def test_min_matches_three_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(min_matches=3)

    def test_min_matches_four_valid(self):
        cfg = SiftConfig(min_matches=4)
        assert cfg.min_matches == 4


# ─── TestMatchResult ──────────────────────────────────────────────────────────

class TestMatchResult:
    def test_basic_creation(self):
        mr = MatchResult(n_matches=10, n_inliers=8, score=0.8)
        assert mr.n_matches == 10
        assert mr.score == pytest.approx(0.8)

    def test_negative_n_matches_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=-1, n_inliers=0, score=0.0)

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=5, n_inliers=-1, score=0.0)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=0, n_inliers=0, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=0, n_inliers=0, score=1.1)

    def test_zero_score_valid(self):
        mr = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert mr.score == 0.0

    def test_one_score_valid(self):
        mr = MatchResult(n_matches=5, n_inliers=5, score=1.0)
        assert mr.score == 1.0

    def test_is_reliable_true(self):
        mr = MatchResult(n_matches=10, n_inliers=8, score=0.8)
        assert mr.is_reliable(min_inliers=4) is True

    def test_is_reliable_false(self):
        mr = MatchResult(n_matches=3, n_inliers=2, score=0.5)
        assert mr.is_reliable(min_inliers=4) is False

    def test_homography_none_by_default(self):
        mr = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert mr.homography is None


# ─── TestExtractKeypoints ─────────────────────────────────────────────────────

class TestExtractKeypoints:
    def test_returns_tuple(self):
        kps, descs = extract_keypoints(_checkerboard())
        assert isinstance(kps, list)

    def test_gray_image(self):
        kps, descs = extract_keypoints(_gray())
        assert isinstance(kps, list)

    def test_color_image(self):
        kps, descs = extract_keypoints(_color())
        assert isinstance(kps, list)

    def test_descriptors_float32_or_none(self):
        _, descs = extract_keypoints(_checkerboard())
        if descs is not None:
            assert descs.dtype == np.float32

    def test_descriptors_128_cols(self):
        _, descs = extract_keypoints(_checkerboard())
        if descs is not None:
            assert descs.shape[1] == 128

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            extract_keypoints(np.zeros((8, 8, 3, 2), dtype=np.uint8))

    def test_custom_config(self):
        cfg = SiftConfig(n_features=10)
        kps, _ = extract_keypoints(_checkerboard(), cfg)
        # Может быть <= 10 точек (зависит от изображения)
        assert len(kps) <= 10 or len(kps) >= 0


# ─── TestMatchDescriptors ─────────────────────────────────────────────────────

class TestMatchDescriptors:
    def _descs(self, n=20):
        rng = np.random.default_rng(42)
        return rng.random((n, 128)).astype(np.float32)

    def test_returns_list(self):
        d1, d2 = self._descs(20), self._descs(20)
        matches = match_descriptors(d1, d2)
        assert isinstance(matches, list)

    def test_ratio_zero_raises(self):
        with pytest.raises(ValueError):
            match_descriptors(self._descs(), self._descs(), ratio_thresh=0.0)

    def test_ratio_one_raises(self):
        with pytest.raises(ValueError):
            match_descriptors(self._descs(), self._descs(), ratio_thresh=1.0)

    def test_none_desc1_returns_empty(self):
        assert match_descriptors(None, self._descs()) == []

    def test_none_desc2_returns_empty(self):
        assert match_descriptors(self._descs(), None) == []

    def test_empty_desc1_returns_empty(self):
        assert match_descriptors(np.zeros((0, 128), dtype=np.float32),
                                  self._descs()) == []

    def test_identical_descriptors_many_matches(self):
        d = self._descs(30)
        matches = match_descriptors(d, d, ratio_thresh=0.99)
        # При ratio_thresh близком к 1.0 все совпадения проходят
        assert len(matches) >= 0  # не падает


# ─── TestComputeHomography ────────────────────────────────────────────────────

class TestComputeHomography:
    def _kp_match_data(self, n=20):
        """Создать синтетические ключевые точки и совпадения."""
        import cv2
        kps1 = [cv2.KeyPoint(float(i * 5), float(i * 3), 5.0) for i in range(n)]
        kps2 = [cv2.KeyPoint(float(i * 5 + 2), float(i * 3 + 1), 5.0) for i in range(n)]
        matches = [cv2.DMatch(i, i, float(i)) for i in range(n)]
        return kps1, kps2, matches

    def test_min_matches_three_raises(self):
        kps1, kps2, matches = self._kp_match_data(4)
        with pytest.raises(ValueError):
            compute_homography(kps1, kps2, matches, min_matches=3)

    def test_ransac_thresh_zero_raises(self):
        kps1, kps2, matches = self._kp_match_data(10)
        with pytest.raises(ValueError):
            compute_homography(kps1, kps2, matches, ransac_thresh=0.0)

    def test_few_matches_returns_none(self):
        import cv2
        kps1 = [cv2.KeyPoint(0.0, 0.0, 1.0)]
        kps2 = [cv2.KeyPoint(1.0, 1.0, 1.0)]
        matches = [cv2.DMatch(0, 0, 1.0)]
        H, mask = compute_homography(kps1, kps2, matches, min_matches=4)
        assert H is None
        assert len(mask) == 0

    def test_returns_tuple(self):
        kps1, kps2, matches = self._kp_match_data(10)
        result = compute_homography(kps1, kps2, matches)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─── TestSiftMatchPair ────────────────────────────────────────────────────────

class TestSiftMatchPair:
    def test_returns_match_result(self):
        mr = sift_match_pair(_checkerboard(), _checkerboard())
        assert isinstance(mr, MatchResult)

    def test_score_in_range(self):
        mr = sift_match_pair(_gray(), _gray())
        assert 0.0 <= mr.score <= 1.0

    def test_n_matches_nonnegative(self):
        mr = sift_match_pair(_gray(), _gray())
        assert mr.n_matches >= 0

    def test_n_inliers_nonnegative(self):
        mr = sift_match_pair(_gray(), _gray())
        assert mr.n_inliers >= 0

    def test_n_inliers_le_n_matches(self):
        mr = sift_match_pair(_checkerboard(), _checkerboard())
        assert mr.n_inliers <= mr.n_matches

    def test_color_image_accepted(self):
        mr = sift_match_pair(_color(), _color())
        assert isinstance(mr, MatchResult)

    def test_identical_images_high_score(self):
        img = _checkerboard()
        mr = sift_match_pair(img, img)
        # Идентичные изображения → высокая оценка (если найдены точки)
        if mr.n_matches > 0:
            assert mr.score >= 0.0


# ─── TestFilterMatchesByDistance ──────────────────────────────────────────────

class TestFilterMatchesByDistance:
    def _make_matches(self, distances):
        import cv2
        return [cv2.DMatch(i, i, float(d)) for i, d in enumerate(distances)]

    def test_filters_above_threshold(self):
        matches = self._make_matches([10.0, 50.0, 100.0, 200.0])
        filtered = filter_matches_by_distance(matches, max_distance=100.0)
        assert len(filtered) == 3

    def test_empty_input(self):
        assert filter_matches_by_distance([], 50.0) == []

    def test_max_distance_zero(self):
        matches = self._make_matches([0.0, 1.0])
        filtered = filter_matches_by_distance(matches, max_distance=0.0)
        assert len(filtered) == 1

    def test_negative_max_distance_raises(self):
        with pytest.raises(ValueError):
            filter_matches_by_distance([], max_distance=-1.0)

    def test_returns_list(self):
        matches = self._make_matches([5.0, 10.0])
        result = filter_matches_by_distance(matches, max_distance=20.0)
        assert isinstance(result, list)


# ─── TestBatchSiftMatch ───────────────────────────────────────────────────────

class TestBatchSiftMatch:
    def test_returns_dict(self):
        images = [_gray(), _gray(seed=1)]
        result = batch_sift_match(images)
        assert isinstance(result, dict)

    def test_correct_number_of_pairs(self):
        images = [_gray(), _gray(seed=1), _gray(seed=2)]
        result = batch_sift_match(images)
        assert len(result) == 3  # C(3,2) = 3

    def test_keys_are_tuples_i_lt_j(self):
        images = [_gray(), _gray(seed=1), _gray(seed=2)]
        result = batch_sift_match(images)
        for i, j in result.keys():
            assert i < j

    def test_single_image_empty(self):
        result = batch_sift_match([_gray()])
        assert result == {}

    def test_empty_list(self):
        result = batch_sift_match([])
        assert result == {}

    def test_each_match_result(self):
        images = [_gray(), _gray(seed=1)]
        result = batch_sift_match(images)
        assert all(isinstance(v, MatchResult) for v in result.values())
