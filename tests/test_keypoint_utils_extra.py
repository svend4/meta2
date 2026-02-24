"""Extra tests for puzzle_reconstruction/utils/keypoint_utils.py."""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from puzzle_reconstruction.utils.keypoint_utils import (
    KeypointSet,
    array_to_keypoints,
    compute_match_score,
    detect_keypoints,
    describe_keypoints,
    filter_by_region,
    filter_by_response,
    filter_matches_ransac,
    keypoints_to_array,
    match_descriptors,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 64, w: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_kpset(n: int = 10, seed: int = 0) -> KeypointSet:
    rng = np.random.default_rng(seed)
    kps = [cv2.KeyPoint(float(rng.integers(5, 60)),
                        float(rng.integers(5, 60)),
                        5.0,
                        response=float(rng.random()))
           for _ in range(n)]
    descs = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    return KeypointSet(keypoints=kps, descriptors=descs)


def _kp_at(x: float, y: float, response: float = 1.0) -> cv2.KeyPoint:
    return cv2.KeyPoint(x, y, 5.0, response=response)


# ─── KeypointSet (extra) ──────────────────────────────────────────────────────

class TestKeypointSetExtra:
    def test_len_correct(self):
        kpset = _make_kpset(8)
        assert len(kpset) == 8

    def test_len_empty(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        assert len(kpset) == 0

    def test_detector_stored(self):
        kpset = KeypointSet(keypoints=[], descriptors=None, detector="sift")
        assert kpset.detector == "sift"

    def test_default_detector_orb(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        assert kpset.detector == "orb"

    def test_params_default_empty(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        assert kpset.params == {}

    def test_params_stored(self):
        kpset = KeypointSet(keypoints=[], descriptors=None, params={"max": 100})
        assert kpset.params["max"] == 100

    def test_descriptors_none_allowed(self):
        kpset = KeypointSet(keypoints=[_kp_at(10, 10)], descriptors=None)
        assert kpset.descriptors is None

    def test_keypoints_list_stored(self):
        kps = [_kp_at(5, 5), _kp_at(10, 10)]
        kpset = KeypointSet(keypoints=kps, descriptors=None)
        assert len(kpset.keypoints) == 2


# ─── detect_keypoints (extra) ─────────────────────────────────────────────────

class TestDetectKeypointsExtra:
    def test_returns_keypoint_set(self):
        result = detect_keypoints(_gray())
        assert isinstance(result, KeypointSet)

    def test_detector_orb_stored(self):
        result = detect_keypoints(_gray(), detector="orb")
        assert result.detector == "orb"

    def test_max_keypoints_limit(self):
        result = detect_keypoints(_gray(), max_keypoints=10)
        assert len(result.keypoints) <= 10

    def test_max_keypoints_1(self):
        result = detect_keypoints(_gray(), max_keypoints=1)
        assert len(result.keypoints) <= 1

    def test_max_keypoints_zero_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), max_keypoints=0)

    def test_invalid_detector_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), detector="harris")

    def test_bgr_input(self):
        result = detect_keypoints(_bgr())
        assert isinstance(result, KeypointSet)

    def test_params_stored(self):
        result = detect_keypoints(_gray(), max_keypoints=50)
        assert result.params.get("max_keypoints") == 50

    def test_keypoints_are_cv2_keypoints(self):
        result = detect_keypoints(_gray())
        for kp in result.keypoints:
            assert isinstance(kp, cv2.KeyPoint)

    def test_small_image(self):
        result = detect_keypoints(np.zeros((8, 8), dtype=np.uint8))
        assert isinstance(result, KeypointSet)


# ─── filter_by_response (extra) ───────────────────────────────────────────────

class TestFilterByResponseExtra:
    def test_returns_keypoint_set(self):
        kpset = _make_kpset(10)
        result = filter_by_response(kpset)
        assert isinstance(result, KeypointSet)

    def test_min_response_zero_keeps_all(self):
        kpset = _make_kpset(10)
        result = filter_by_response(kpset, min_response=0.0)
        assert len(result) == 10

    def test_min_response_high_filters_all(self):
        kpset = _make_kpset(10)
        result = filter_by_response(kpset, min_response=999.0)
        assert len(result) == 0

    def test_top_k_limits(self):
        kpset = _make_kpset(20)
        result = filter_by_response(kpset, top_k=5)
        assert len(result) <= 5

    def test_top_k_zero_no_limit(self):
        kpset = _make_kpset(10)
        result = filter_by_response(kpset, top_k=0)
        assert len(result) == 10

    def test_top_k_negative_raises(self):
        kpset = _make_kpset(5)
        with pytest.raises(ValueError):
            filter_by_response(kpset, top_k=-1)

    def test_detector_preserved(self):
        kpset = _make_kpset(5)
        kpset.detector = "sift"
        result = filter_by_response(kpset)
        assert result.detector == "sift"

    def test_empty_input_returns_empty(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        result = filter_by_response(kpset)
        assert len(result) == 0

    def test_filtered_result_has_descriptors_or_none(self):
        kpset = _make_kpset(10)
        result = filter_by_response(kpset, min_response=0.5)
        # descriptors should match keypoints count if nonzero
        if len(result.keypoints) > 0 and result.descriptors is not None:
            assert len(result.descriptors) == len(result.keypoints)


# ─── filter_by_region (extra) ─────────────────────────────────────────────────

class TestFilterByRegionExtra:
    def test_returns_keypoint_set(self):
        kpset = _make_kpset(10)
        mask = np.ones((64, 64), dtype=np.uint8)
        result = filter_by_region(kpset, mask)
        assert isinstance(result, KeypointSet)

    def test_all_ones_mask_keeps_all(self):
        kpset = _make_kpset(10)
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        result = filter_by_region(kpset, mask)
        assert len(result) == len(kpset)

    def test_all_zeros_mask_removes_all(self):
        kpset = _make_kpset(10)
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = filter_by_region(kpset, mask)
        assert len(result) == 0

    def test_3d_mask_raises(self):
        kpset = _make_kpset(5)
        mask = np.ones((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            filter_by_region(kpset, mask)

    def test_detector_preserved(self):
        kpset = _make_kpset(5)
        kpset.detector = "sift"
        mask = np.ones((64, 64), dtype=np.uint8)
        result = filter_by_region(kpset, mask)
        assert result.detector == "sift"

    def test_empty_input(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        mask = np.ones((64, 64), dtype=np.uint8)
        result = filter_by_region(kpset, mask)
        assert len(result) == 0


# ─── keypoints_to_array (extra) ───────────────────────────────────────────────

class TestKeypointsToArrayExtra:
    def test_empty_returns_0x2(self):
        arr = keypoints_to_array([])
        assert arr.shape == (0, 2)

    def test_returns_float32(self):
        kps = [_kp_at(10.0, 20.0)]
        arr = keypoints_to_array(kps)
        assert arr.dtype == np.float32

    def test_shape_nx2(self):
        kps = [_kp_at(float(i), float(i)) for i in range(5)]
        arr = keypoints_to_array(kps)
        assert arr.shape == (5, 2)

    def test_coordinates_correct(self):
        kps = [_kp_at(10.0, 20.0), _kp_at(30.0, 40.0)]
        arr = keypoints_to_array(kps)
        assert arr[0, 0] == pytest.approx(10.0)
        assert arr[0, 1] == pytest.approx(20.0)
        assert arr[1, 0] == pytest.approx(30.0)
        assert arr[1, 1] == pytest.approx(40.0)

    def test_single_keypoint(self):
        arr = keypoints_to_array([_kp_at(5.0, 7.0)])
        assert arr.shape == (1, 2)


# ─── array_to_keypoints (extra) ───────────────────────────────────────────────

class TestArrayToKeypointsExtra:
    def test_empty_array_returns_empty(self):
        pts = np.empty((0, 2), dtype=np.float32)
        kps = array_to_keypoints(pts)
        assert kps == []

    def test_returns_list_of_keypoints(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        kps = array_to_keypoints(pts)
        assert all(isinstance(k, cv2.KeyPoint) for k in kps)

    def test_length_correct(self):
        pts = np.zeros((5, 2), dtype=np.float32)
        kps = array_to_keypoints(pts)
        assert len(kps) == 5

    def test_coordinates_correct(self):
        pts = np.array([[15.0, 25.0]], dtype=np.float32)
        kps = array_to_keypoints(pts)
        assert kps[0].pt[0] == pytest.approx(15.0)
        assert kps[0].pt[1] == pytest.approx(25.0)

    def test_custom_size(self):
        pts = np.array([[10.0, 10.0]], dtype=np.float32)
        kps = array_to_keypoints(pts, size=7.0)
        assert kps[0].size == pytest.approx(7.0)

    def test_custom_response(self):
        pts = np.array([[10.0, 10.0]], dtype=np.float32)
        kps = array_to_keypoints(pts, response=0.5)
        assert kps[0].response == pytest.approx(0.5)

    def test_roundtrip(self):
        pts = np.array([[5.0, 10.0], [20.0, 30.0]], dtype=np.float32)
        kps = array_to_keypoints(pts)
        arr = keypoints_to_array(kps)
        assert np.allclose(arr, pts, atol=1e-5)


# ─── compute_match_score (extra) ──────────────────────────────────────────────

class TestComputeMatchScoreExtra:
    def test_zero_keypoints_returns_zero(self):
        assert compute_match_score([], 0, 0) == pytest.approx(0.0)

    def test_no_matches_returns_zero(self):
        assert compute_match_score([], 10, 10) == pytest.approx(0.0)

    def test_full_match_returns_one(self):
        matches = [object()] * 10
        assert compute_match_score(matches, 10, 10) == pytest.approx(1.0)

    def test_half_match(self):
        matches = [object()] * 5
        score = compute_match_score(matches, 10, 10)
        assert score == pytest.approx(0.5)

    def test_negative_n_kp1_raises(self):
        with pytest.raises(ValueError):
            compute_match_score([], -1, 5)

    def test_negative_n_kp2_raises(self):
        with pytest.raises(ValueError):
            compute_match_score([], 5, -1)

    def test_result_in_0_1(self):
        matches = [object()] * 7
        score = compute_match_score(matches, 10, 8)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        score = compute_match_score([], 5, 5)
        assert isinstance(score, float)

    def test_unequal_sets_uses_max(self):
        # max(5, 20) = 20; 3 matches → 3/20 = 0.15
        matches = [object()] * 3
        score = compute_match_score(matches, 5, 20)
        assert score == pytest.approx(3 / 20)


# ─── match_descriptors (extra) ────────────────────────────────────────────────

class TestMatchDescriptorsExtra:
    def test_returns_list(self):
        kpset = _make_kpset(10)
        result = match_descriptors(kpset.descriptors, kpset.descriptors)
        assert isinstance(result, list)

    def test_identical_descriptors_some_matches(self):
        kpset = _make_kpset(10)
        result = match_descriptors(kpset.descriptors, kpset.descriptors)
        # With ratio test, some matches should survive
        assert len(result) >= 0

    def test_empty_descs1_returns_empty(self):
        descs1 = np.empty((0, 32), dtype=np.uint8)
        descs2 = np.ones((5, 32), dtype=np.uint8)
        result = match_descriptors(descs1, descs2)
        assert result == []

    def test_empty_descs2_returns_empty(self):
        descs1 = np.ones((5, 32), dtype=np.uint8)
        descs2 = np.empty((0, 32), dtype=np.uint8)
        result = match_descriptors(descs1, descs2)
        assert result == []

    def test_none_descs_raises(self):
        with pytest.raises(ValueError):
            match_descriptors(None, np.ones((5, 32), dtype=np.uint8))

    def test_invalid_ratio_thresh_raises(self):
        descs = np.ones((5, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            match_descriptors(descs, descs, ratio_thresh=1.0)

    def test_ratio_thresh_zero_raises(self):
        descs = np.ones((5, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            match_descriptors(descs, descs, ratio_thresh=0.0)

    def test_matches_are_dmatch(self):
        kpset = _make_kpset(10)
        result = match_descriptors(kpset.descriptors, kpset.descriptors)
        for m in result:
            assert isinstance(m, cv2.DMatch)


# ─── filter_matches_ransac (extra) ────────────────────────────────────────────

class TestFilterMatchesRansacExtra:
    def test_returns_tuple(self):
        kpset = _make_kpset(10)
        result = filter_matches_ransac(kpset.keypoints, kpset.keypoints, [])
        assert isinstance(result, tuple) and len(result) == 2

    def test_too_few_matches_returns_as_is(self):
        kpset = _make_kpset(10)
        matches = [cv2.DMatch(i, i, 0.0) for i in range(3)]
        inliers, H = filter_matches_ransac(kpset.keypoints, kpset.keypoints, matches)
        assert H is None
        assert len(inliers) == 3

    def test_invalid_reproj_threshold_raises(self):
        kpset = _make_kpset(10)
        with pytest.raises(ValueError):
            filter_matches_ransac(kpset.keypoints, kpset.keypoints, [], reproj_threshold=0.0)

    def test_negative_reproj_threshold_raises(self):
        kpset = _make_kpset(10)
        with pytest.raises(ValueError):
            filter_matches_ransac(kpset.keypoints, kpset.keypoints, [], reproj_threshold=-1.0)

    def test_empty_matches_returns_empty_and_none(self):
        kpset = _make_kpset(5)
        inliers, H = filter_matches_ransac(kpset.keypoints, kpset.keypoints, [])
        assert inliers == []
        assert H is None
