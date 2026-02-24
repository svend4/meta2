"""Extra tests for puzzle_reconstruction/utils/keypoint_utils.py."""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from puzzle_reconstruction.utils.keypoint_utils import (
    KeypointSet,
    detect_keypoints,
    filter_by_response,
    filter_by_region,
    keypoints_to_array,
    array_to_keypoints,
    compute_match_score,
    match_descriptors,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _kp(x=10.0, y=10.0, resp=5.0) -> cv2.KeyPoint:
    return cv2.KeyPoint(x, y, 10.0, response=resp)


def _kpset(n=5) -> KeypointSet:
    kps = [_kp(float(i), float(i), float(i+1)) for i in range(n)]
    return KeypointSet(keypoints=kps, descriptors=None, detector="orb")


# ─── KeypointSet ──────────────────────────────────────────────────────────────

class TestKeypointSetExtra:
    def test_len(self):
        ks = _kpset(5)
        assert len(ks) == 5

    def test_stores_detector(self):
        ks = _kpset()
        assert ks.detector == "orb"

    def test_none_descriptors(self):
        ks = _kpset()
        assert ks.descriptors is None


# ─── detect_keypoints ─────────────────────────────────────────────────────────

class TestDetectKeypointsExtra:
    def test_returns_keypoint_set(self):
        img = _gray()
        result = detect_keypoints(img, detector="orb")
        assert isinstance(result, KeypointSet)

    def test_max_keypoints_zero_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), max_keypoints=0)

    def test_invalid_detector_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), detector="surf")

    def test_respects_max_keypoints(self):
        img = _gray()
        result = detect_keypoints(img, max_keypoints=10)
        assert len(result) <= 10


# ─── filter_by_response ───────────────────────────────────────────────────────

class TestFilterByResponseExtra:
    def test_returns_keypoint_set(self):
        ks = _kpset(5)
        assert isinstance(filter_by_response(ks), KeypointSet)

    def test_filters_low_response(self):
        ks = _kpset(5)
        result = filter_by_response(ks, min_response=3.0)
        assert all(kp.response >= 3.0 for kp in result.keypoints)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            filter_by_response(_kpset(), top_k=-1)

    def test_top_k_limits_results(self):
        ks = _kpset(10)
        result = filter_by_response(ks, top_k=3)
        assert len(result) <= 3


# ─── filter_by_region ─────────────────────────────────────────────────────────

class TestFilterByRegionExtra:
    def test_returns_keypoint_set(self):
        ks = _kpset(3)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = filter_by_region(ks, mask)
        assert isinstance(result, KeypointSet)

    def test_non_2d_mask_raises(self):
        ks = _kpset(3)
        with pytest.raises(ValueError):
            filter_by_region(ks, np.ones((10, 10, 3), dtype=np.uint8))

    def test_zero_mask_removes_all(self):
        ks = _kpset(5)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = filter_by_region(ks, mask)
        assert len(result) == 0


# ─── keypoints_to_array ───────────────────────────────────────────────────────

class TestKeypointsToArrayExtra:
    def test_returns_array(self):
        kps = [_kp(1.0, 2.0)]
        arr = keypoints_to_array(kps)
        assert isinstance(arr, np.ndarray) and arr.shape == (1, 2)

    def test_empty_returns_0x2(self):
        arr = keypoints_to_array([])
        assert arr.shape == (0, 2)

    def test_values_correct(self):
        kps = [_kp(3.0, 7.0)]
        arr = keypoints_to_array(kps)
        assert arr[0, 0] == pytest.approx(3.0)
        assert arr[0, 1] == pytest.approx(7.0)


# ─── array_to_keypoints ───────────────────────────────────────────────────────

class TestArrayToKeypointsExtra:
    def test_returns_list(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = array_to_keypoints(pts)
        assert isinstance(result, list) and len(result) == 2

    def test_values_stored(self):
        pts = np.array([[5.0, 6.0]])
        kps = array_to_keypoints(pts)
        assert kps[0].pt == pytest.approx((5.0, 6.0))

    def test_empty_input(self):
        result = array_to_keypoints(np.empty((0, 2)))
        assert result == []


# ─── compute_match_score ──────────────────────────────────────────────────────

class TestComputeMatchScoreExtra:
    def test_no_matches_zero(self):
        assert compute_match_score([], 10, 10) == pytest.approx(0.0)

    def test_all_match_one(self):
        matches = [object()] * 10
        assert compute_match_score(matches, 10, 10) == pytest.approx(1.0)

    def test_negative_n_kp_raises(self):
        with pytest.raises(ValueError):
            compute_match_score([], -1, 5)

    def test_zero_denominator_returns_zero(self):
        assert compute_match_score([], 0, 0) == pytest.approx(0.0)
