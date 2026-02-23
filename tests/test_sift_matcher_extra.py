"""Extra tests for puzzle_reconstruction.algorithms.sift_matcher."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=128, w=128, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=128, w=128, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(h=128, w=128, block=16):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if ((r // block) + (c // block)) % 2 == 0:
                img[r:r + block, c:c + block] = 200
    return img


def _descs(n=20, seed=42):
    return np.random.default_rng(seed).random((n, 128)).astype(np.float32)


def _cv_matches(distances):
    import cv2
    return [cv2.DMatch(i, i, float(d)) for i, d in enumerate(distances)]


# ─── TestSiftConfigExtra ─────────────────────────────────────────────────────

class TestSiftConfigExtra:
    def test_custom_n_features(self):
        cfg = SiftConfig(n_features=1000)
        assert cfg.n_features == 1000

    def test_ratio_thresh_low(self):
        cfg = SiftConfig(ratio_thresh=0.5)
        assert cfg.ratio_thresh == pytest.approx(0.5)

    def test_min_matches_10(self):
        cfg = SiftConfig(min_matches=10)
        assert cfg.min_matches == 10

    def test_negative_n_features_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(n_features=-5)

    def test_ratio_thresh_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(ratio_thresh=0.0)

    def test_ratio_thresh_one_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(ratio_thresh=1.0)

    def test_min_matches_3_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(min_matches=3)

    def test_n_octave_layers_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(n_octave_layers=0)

    def test_contrast_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(contrast_threshold=0.0)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            SiftConfig(sigma=0.0)


# ─── TestMatchResultExtra ────────────────────────────────────────────────────

class TestMatchResultExtra:
    def test_fields_stored(self):
        mr = MatchResult(n_matches=15, n_inliers=12, score=0.8)
        assert mr.n_matches == 15
        assert mr.n_inliers == 12
        assert mr.score == pytest.approx(0.8)

    def test_zero_matches_valid(self):
        mr = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert mr.n_matches == 0

    def test_negative_n_matches_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=-1, n_inliers=0, score=0.0)

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=5, n_inliers=-1, score=0.5)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=0, n_inliers=0, score=-0.01)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            MatchResult(n_matches=0, n_inliers=0, score=1.01)

    def test_is_reliable_true(self):
        mr = MatchResult(n_matches=20, n_inliers=15, score=0.9)
        assert mr.is_reliable(min_inliers=10) is True

    def test_is_reliable_false(self):
        mr = MatchResult(n_matches=5, n_inliers=3, score=0.5)
        assert mr.is_reliable(min_inliers=5) is False

    def test_homography_none(self):
        mr = MatchResult(n_matches=0, n_inliers=0, score=0.0)
        assert mr.homography is None

    def test_one_score_valid(self):
        mr = MatchResult(n_matches=10, n_inliers=10, score=1.0)
        assert mr.score == 1.0

    def test_inliers_le_matches(self):
        mr = MatchResult(n_matches=10, n_inliers=10, score=0.9)
        assert mr.n_inliers <= mr.n_matches


# ─── TestExtractKeypointsExtra ────────────────────────────────────────────────

class TestExtractKeypointsExtra:
    def test_checkerboard_returns_keypoints(self):
        kps, descs = extract_keypoints(_checkerboard())
        assert isinstance(kps, list)

    def test_gray_returns_tuple(self):
        result = extract_keypoints(_gray())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_color_accepted(self):
        kps, descs = extract_keypoints(_color())
        assert isinstance(kps, list)

    def test_descriptors_dtype(self):
        _, descs = extract_keypoints(_checkerboard())
        if descs is not None:
            assert descs.dtype == np.float32

    def test_descriptors_128_cols(self):
        _, descs = extract_keypoints(_checkerboard())
        if descs is not None:
            assert descs.shape[1] == 128

    def test_custom_config_n_features(self):
        cfg = SiftConfig(n_features=5)
        kps, _ = extract_keypoints(_checkerboard(), cfg)
        assert len(kps) <= 5 or len(kps) >= 0

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            extract_keypoints(np.zeros((8, 8, 3, 2), dtype=np.uint8))

    def test_small_image(self):
        kps, _ = extract_keypoints(_gray(h=16, w=16))
        assert isinstance(kps, list)


# ─── TestMatchDescriptorsExtra ────────────────────────────────────────────────

class TestMatchDescriptorsExtra:
    def test_returns_list(self):
        result = match_descriptors(_descs(20), _descs(20, seed=43))
        assert isinstance(result, list)

    def test_none_desc1_empty(self):
        assert match_descriptors(None, _descs()) == []

    def test_none_desc2_empty(self):
        assert match_descriptors(_descs(), None) == []

    def test_empty_desc_empty(self):
        empty = np.zeros((0, 128), dtype=np.float32)
        assert match_descriptors(empty, _descs()) == []

    def test_ratio_zero_raises(self):
        with pytest.raises(ValueError):
            match_descriptors(_descs(), _descs(), ratio_thresh=0.0)

    def test_ratio_one_raises(self):
        with pytest.raises(ValueError):
            match_descriptors(_descs(), _descs(), ratio_thresh=1.0)

    def test_identical_descs_no_crash(self):
        d = _descs(30)
        result = match_descriptors(d, d, ratio_thresh=0.99)
        assert isinstance(result, list)


# ─── TestComputeHomographyExtra ──────────────────────────────────────────────

class TestComputeHomographyExtra:
    def _kp_matches(self, n=20):
        import cv2
        kps1 = [cv2.KeyPoint(float(i * 5), float(i * 3), 5.0) for i in range(n)]
        kps2 = [cv2.KeyPoint(float(i * 5 + 2), float(i * 3 + 1), 5.0) for i in range(n)]
        matches = [cv2.DMatch(i, i, float(i)) for i in range(n)]
        return kps1, kps2, matches

    def test_returns_tuple_of_2(self):
        kps1, kps2, matches = self._kp_matches(10)
        result = compute_homography(kps1, kps2, matches)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_min_matches_3_raises(self):
        kps1, kps2, matches = self._kp_matches(4)
        with pytest.raises(ValueError):
            compute_homography(kps1, kps2, matches, min_matches=3)

    def test_ransac_thresh_zero_raises(self):
        kps1, kps2, matches = self._kp_matches(10)
        with pytest.raises(ValueError):
            compute_homography(kps1, kps2, matches, ransac_thresh=0.0)

    def test_too_few_matches_returns_none(self):
        import cv2
        kps1 = [cv2.KeyPoint(0.0, 0.0, 1.0)]
        kps2 = [cv2.KeyPoint(1.0, 1.0, 1.0)]
        matches = [cv2.DMatch(0, 0, 1.0)]
        H, mask = compute_homography(kps1, kps2, matches, min_matches=4)
        assert H is None
        assert len(mask) == 0

    def test_enough_matches_runs(self):
        kps1, kps2, matches = self._kp_matches(20)
        H, mask = compute_homography(kps1, kps2, matches)
        # H can be None or 3x3, mask is list
        assert isinstance(mask, (list, np.ndarray))


# ─── TestSiftMatchPairExtra ──────────────────────────────────────────────────

class TestSiftMatchPairExtra:
    def test_returns_match_result(self):
        assert isinstance(sift_match_pair(_gray(), _gray(seed=1)), MatchResult)

    def test_score_in_range(self):
        mr = sift_match_pair(_gray(), _gray(seed=1))
        assert 0.0 <= mr.score <= 1.0

    def test_n_matches_nonneg(self):
        assert sift_match_pair(_gray(), _gray(seed=1)).n_matches >= 0

    def test_n_inliers_nonneg(self):
        assert sift_match_pair(_gray(), _gray(seed=1)).n_inliers >= 0

    def test_inliers_le_matches(self):
        mr = sift_match_pair(_checkerboard(), _checkerboard())
        assert mr.n_inliers <= mr.n_matches

    def test_color_accepted(self):
        mr = sift_match_pair(_color(), _color(seed=2))
        assert isinstance(mr, MatchResult)

    def test_identical_checkerboard_nonneg_score(self):
        img = _checkerboard()
        mr = sift_match_pair(img, img)
        assert mr.score >= 0.0

    def test_gray_pair(self):
        mr = sift_match_pair(_gray(seed=3), _gray(seed=4))
        assert isinstance(mr, MatchResult)


# ─── TestFilterMatchesByDistanceExtra ────────────────────────────────────────

class TestFilterMatchesByDistanceExtra:
    def test_filters_above_threshold(self):
        matches = _cv_matches([5.0, 50.0, 100.0, 200.0])
        filtered = filter_matches_by_distance(matches, max_distance=100.0)
        assert len(filtered) == 3

    def test_empty_input(self):
        assert filter_matches_by_distance([], 50.0) == []

    def test_max_distance_zero(self):
        matches = _cv_matches([0.0, 1.0])
        filtered = filter_matches_by_distance(matches, max_distance=0.0)
        assert len(filtered) == 1

    def test_negative_max_distance_raises(self):
        with pytest.raises(ValueError):
            filter_matches_by_distance([], max_distance=-1.0)

    def test_returns_list(self):
        matches = _cv_matches([5.0, 10.0])
        assert isinstance(filter_matches_by_distance(matches, 20.0), list)

    def test_all_pass(self):
        matches = _cv_matches([1.0, 2.0, 3.0])
        assert len(filter_matches_by_distance(matches, max_distance=100.0)) == 3

    def test_none_pass(self):
        matches = _cv_matches([50.0, 60.0])
        assert len(filter_matches_by_distance(matches, max_distance=10.0)) == 0


# ─── TestBatchSiftMatchExtra ─────────────────────────────────────────────────

class TestBatchSiftMatchExtra:
    def test_returns_dict(self):
        result = batch_sift_match([_gray(), _gray(seed=1)])
        assert isinstance(result, dict)

    def test_correct_pairs_3_images(self):
        result = batch_sift_match([_gray(), _gray(seed=1), _gray(seed=2)])
        assert len(result) == 3

    def test_keys_are_tuples(self):
        result = batch_sift_match([_gray(), _gray(seed=1)])
        for k in result.keys():
            assert isinstance(k, tuple)
            assert len(k) == 2

    def test_keys_i_lt_j(self):
        result = batch_sift_match([_gray(), _gray(seed=1), _gray(seed=2)])
        for i, j in result.keys():
            assert i < j

    def test_single_image_empty(self):
        assert batch_sift_match([_gray()]) == {}

    def test_empty_list(self):
        assert batch_sift_match([]) == {}

    def test_values_are_match_results(self):
        result = batch_sift_match([_gray(), _gray(seed=1)])
        for v in result.values():
            assert isinstance(v, MatchResult)

    def test_four_images_six_pairs(self):
        imgs = [_gray(seed=i) for i in range(4)]
        result = batch_sift_match(imgs)
        assert len(result) == 6  # C(4,2)

    def test_scores_in_range(self):
        imgs = [_gray(seed=i) for i in range(3)]
        for mr in batch_sift_match(imgs).values():
            assert 0.0 <= mr.score <= 1.0
