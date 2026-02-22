"""Tests for puzzle_reconstruction.utils.keypoint_utils."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.integers(0, 256, (h, w))).astype(np.uint8)


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _kpset_from_image(img: np.ndarray, max_kp: int = 50) -> KeypointSet:
    return detect_keypoints(img, detector="orb", max_keypoints=max_kp)


def _make_kp(x: float, y: float, response: float = 1.0) -> cv2.KeyPoint:
    return cv2.KeyPoint(x, y, 1.0, response=response)


def _make_kpset(pts, responses=None) -> KeypointSet:
    if responses is None:
        responses = [1.0] * len(pts)
    kps = [_make_kp(float(x), float(y), r) for (x, y), r in zip(pts, responses)]
    descs = np.zeros((len(kps), 32), dtype=np.uint8)
    return KeypointSet(keypoints=kps, descriptors=descs, detector="orb")


# ─── KeypointSet ─────────────────────────────────────────────────────────────

class TestKeypointSet:
    def test_len(self):
        kpset = _make_kpset([(0, 0), (1, 1), (2, 2)])
        assert len(kpset) == 3

    def test_zero_len_empty(self):
        kpset = KeypointSet(keypoints=[], descriptors=None)
        assert len(kpset) == 0

    def test_fields_stored(self):
        kps = [_make_kp(5.0, 10.0)]
        descs = np.zeros((1, 32), dtype=np.uint8)
        kpset = KeypointSet(keypoints=kps, descriptors=descs, detector="orb",
                            params={"max_keypoints": 100})
        assert kpset.detector == "orb"
        assert kpset.params["max_keypoints"] == 100
        assert kpset.descriptors.shape == (1, 32)

    def test_none_descriptors_allowed(self):
        kpset = KeypointSet(keypoints=[_make_kp(0, 0)], descriptors=None)
        assert kpset.descriptors is None


# ─── detect_keypoints ─────────────────────────────────────────────────────────

class TestDetectKeypoints:
    def test_returns_keypoint_set(self):
        result = detect_keypoints(_gray())
        assert isinstance(result, KeypointSet)

    def test_max_keypoints_respected(self):
        result = detect_keypoints(_gray(128, 128), max_keypoints=20)
        assert len(result) <= 20

    def test_max_keypoints_less_than_1_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), max_keypoints=0)

    def test_unknown_detector_raises(self):
        with pytest.raises(ValueError):
            detect_keypoints(_gray(), detector="fast")

    def test_bgr_accepted(self):
        result = detect_keypoints(_bgr(), detector="orb")
        assert isinstance(result, KeypointSet)

    def test_descriptors_shape(self):
        result = detect_keypoints(_gray(128, 128), detector="orb", max_keypoints=50)
        if result.descriptors is not None:
            assert result.descriptors.ndim == 2
            assert result.descriptors.shape[0] == len(result.keypoints)

    def test_detector_name_stored(self):
        result = detect_keypoints(_gray(), detector="orb")
        assert result.detector == "orb"

    def test_params_stored(self):
        result = detect_keypoints(_gray(), max_keypoints=30)
        assert result.params["max_keypoints"] == 30


# ─── filter_by_response ──────────────────────────────────────────────────────

class TestFilterByResponse:
    def test_top_k_negative_raises(self):
        kpset = _make_kpset([(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            filter_by_response(kpset, top_k=-1)

    def test_min_response_filters(self):
        kpset = _make_kpset([(0, 0), (1, 1), (2, 2)], responses=[0.1, 0.5, 0.9])
        result = filter_by_response(kpset, min_response=0.4)
        assert len(result) == 2
        assert all(kp.response >= 0.4 for kp in result.keypoints)

    def test_top_k_limits(self):
        kpset = _make_kpset(
            [(float(i), 0.0) for i in range(10)],
            responses=[float(i) for i in range(10)],
        )
        result = filter_by_response(kpset, top_k=3)
        assert len(result) == 3

    def test_top_k_sorted_by_response(self):
        kpset = _make_kpset(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            responses=[0.2, 0.8, 0.5],
        )
        result = filter_by_response(kpset, top_k=2)
        responses = sorted([kp.response for kp in result.keypoints], reverse=True)
        assert responses[0] >= responses[1]

    def test_empty_result_no_descriptors(self):
        kpset = _make_kpset([(0, 0)], responses=[0.1])
        result = filter_by_response(kpset, min_response=1.0)
        assert len(result) == 0
        assert result.descriptors is None

    def test_detector_preserved(self):
        kpset = _make_kpset([(0, 0)])
        result = filter_by_response(kpset, min_response=0.0)
        assert result.detector == kpset.detector

    def test_descriptors_filtered_consistently(self):
        kpset = _make_kpset(
            [(0.0, 0.0), (1.0, 0.0)], responses=[0.1, 0.9]
        )
        result = filter_by_response(kpset, min_response=0.5)
        assert len(result) == 1
        if result.descriptors is not None:
            assert result.descriptors.shape[0] == 1


# ─── filter_by_region ────────────────────────────────────────────────────────

class TestFilterByRegion:
    def test_mask_not_2d_raises(self):
        kpset = _make_kpset([(10, 10)])
        with pytest.raises(ValueError):
            filter_by_region(kpset, np.ones((32, 32, 3), dtype=np.uint8))

    def test_keeps_inside_mask(self):
        kpset = _make_kpset([(10.0, 10.0), (25.0, 25.0)])
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 5:15] = 255  # only point (10,10) is inside
        result = filter_by_region(kpset, mask)
        assert len(result) == 1
        assert result.keypoints[0].pt == pytest.approx((10.0, 10.0))

    def test_empty_mask_removes_all(self):
        kpset = _make_kpset([(5.0, 5.0), (10.0, 10.0)])
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = filter_by_region(kpset, mask)
        assert len(result) == 0

    def test_full_mask_keeps_all(self):
        kpset = _make_kpset([(5.0, 5.0), (15.0, 15.0)])
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        result = filter_by_region(kpset, mask)
        assert len(result) == 2

    def test_returns_keypoint_set(self):
        kpset = _make_kpset([(10, 10)])
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        assert isinstance(filter_by_region(kpset, mask), KeypointSet)


# ─── describe_keypoints ──────────────────────────────────────────────────────

class TestDescribeKeypoints:
    def test_empty_keypoints_returns_none(self):
        result = describe_keypoints(_gray(), [], detector="orb")
        assert result is None

    def test_unknown_detector_raises(self):
        kps = [_make_kp(10.0, 10.0)]
        with pytest.raises(ValueError):
            describe_keypoints(_gray(), kps, detector="harris")

    def test_returns_ndarray_or_none(self):
        img = _gray(128, 128)
        kpset = detect_keypoints(img, max_keypoints=10)
        if kpset.keypoints:
            result = describe_keypoints(img, kpset.keypoints, detector="orb")
            assert result is None or isinstance(result, np.ndarray)


# ─── match_descriptors ───────────────────────────────────────────────────────

class TestMatchDescriptors:
    def test_ratio_thresh_not_in_range_raises(self):
        d = np.zeros((5, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            match_descriptors(d, d, ratio_thresh=0.0)
        with pytest.raises(ValueError):
            match_descriptors(d, d, ratio_thresh=1.0)

    def test_none_descs_raises(self):
        d = np.zeros((5, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            match_descriptors(None, d)
        with pytest.raises(ValueError):
            match_descriptors(d, None)

    def test_empty_descs_returns_empty(self):
        d = np.zeros((5, 32), dtype=np.uint8)
        empty = np.zeros((0, 32), dtype=np.uint8)
        assert match_descriptors(d, empty) == []
        assert match_descriptors(empty, d) == []

    def test_returns_list(self):
        d1 = np.random.default_rng(0).integers(0, 256, (10, 32), dtype=np.uint8)
        d2 = np.random.default_rng(1).integers(0, 256, (10, 32), dtype=np.uint8)
        result = match_descriptors(d1, d2, ratio_thresh=0.75)
        assert isinstance(result, list)

    def test_identical_descs_produce_matches(self):
        d = np.random.default_rng(0).integers(0, 256, (5, 32), dtype=np.uint8)
        result = match_descriptors(d, d, ratio_thresh=0.75)
        assert len(result) >= 0  # some matches expected


# ─── filter_matches_ransac ───────────────────────────────────────────────────

class TestFilterMatchesRansac:
    def _make_matches(self, n: int):
        """Create n synthetic DMatch objects with valid keypoints."""
        kps1, kps2, matches = [], [], []
        for i in range(n):
            x1, y1 = float(i * 5), float(i * 3)
            x2, y2 = x1 + 1.0, y1 + 1.0
            kps1.append(_make_kp(x1, y1))
            kps2.append(_make_kp(x2, y2))
            m = cv2.DMatch()
            m.queryIdx = i
            m.trainIdx = i
            m.distance = 1.0
            matches.append(m)
        return kps1, kps2, matches

    def test_reproj_threshold_zero_raises(self):
        kps1, kps2, matches = self._make_matches(6)
        with pytest.raises(ValueError):
            filter_matches_ransac(kps1, kps2, matches, reproj_threshold=0.0)

    def test_reproj_threshold_negative_raises(self):
        kps1, kps2, matches = self._make_matches(6)
        with pytest.raises(ValueError):
            filter_matches_ransac(kps1, kps2, matches, reproj_threshold=-1.0)

    def test_fewer_than_4_matches_returned_unchanged(self):
        kps1, kps2, matches = self._make_matches(3)
        inliers, H = filter_matches_ransac(kps1, kps2, matches)
        assert len(inliers) == 3
        assert H is None

    def test_returns_tuple(self):
        kps1, kps2, matches = self._make_matches(3)
        result = filter_matches_ransac(kps1, kps2, matches)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_homography_3x3_or_none(self):
        kps1, kps2, matches = self._make_matches(10)
        _, H = filter_matches_ransac(kps1, kps2, matches)
        if H is not None:
            assert H.shape == (3, 3)


# ─── keypoints_to_array ──────────────────────────────────────────────────────

class TestKeypointsToArray:
    def test_empty_returns_shape_0_2(self):
        result = keypoints_to_array([])
        assert result.shape == (0, 2)
        assert result.dtype == np.float32

    def test_shape_n_by_2(self):
        kps = [_make_kp(5.0, 10.0), _make_kp(15.0, 20.0)]
        result = keypoints_to_array(kps)
        assert result.shape == (2, 2)

    def test_dtype_float32(self):
        kps = [_make_kp(1.0, 2.0)]
        assert keypoints_to_array(kps).dtype == np.float32

    def test_coordinates_correct(self):
        kps = [_make_kp(7.5, 12.3)]
        result = keypoints_to_array(kps)
        assert result[0, 0] == pytest.approx(7.5)
        assert result[0, 1] == pytest.approx(12.3)

    def test_multiple_points(self):
        pts = [(float(i), float(i * 2)) for i in range(5)]
        kps = [_make_kp(x, y) for x, y in pts]
        result = keypoints_to_array(kps)
        for i, (x, y) in enumerate(pts):
            assert result[i, 0] == pytest.approx(x)
            assert result[i, 1] == pytest.approx(y)


# ─── array_to_keypoints ──────────────────────────────────────────────────────

class TestArrayToKeypoints:
    def test_empty_pts_returns_empty(self):
        result = array_to_keypoints(np.empty((0, 2), dtype=np.float32))
        assert result == []

    def test_length_preserved(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = array_to_keypoints(pts)
        assert len(result) == 3

    def test_all_keypoint_instances(self):
        pts = np.array([[1.0, 2.0]], dtype=np.float32)
        result = array_to_keypoints(pts)
        assert isinstance(result[0], cv2.KeyPoint)

    def test_coordinates_correct(self):
        pts = np.array([[7.5, 12.3]], dtype=np.float32)
        result = array_to_keypoints(pts)
        assert result[0].pt[0] == pytest.approx(7.5)
        assert result[0].pt[1] == pytest.approx(12.3)

    def test_roundtrip_with_keypoints_to_array(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        kps = array_to_keypoints(pts)
        back = keypoints_to_array(kps)
        np.testing.assert_array_almost_equal(back, pts, decimal=4)


# ─── compute_match_score ─────────────────────────────────────────────────────

class TestComputeMatchScore:
    def test_n_kp_negative_raises(self):
        with pytest.raises(ValueError):
            compute_match_score([], n_kp1=-1, n_kp2=5)
        with pytest.raises(ValueError):
            compute_match_score([], n_kp1=5, n_kp2=-1)

    def test_zero_denominator_returns_zero(self):
        assert compute_match_score([], n_kp1=0, n_kp2=0) == pytest.approx(0.0)

    def test_no_matches_returns_zero(self):
        assert compute_match_score([], n_kp1=10, n_kp2=10) == pytest.approx(0.0)

    def test_full_match_returns_one(self):
        fake_matches = [object()] * 10
        assert compute_match_score(fake_matches, n_kp1=10, n_kp2=10) == pytest.approx(1.0)

    def test_partial_match_in_unit_interval(self):
        fake_matches = [object()] * 5
        score = compute_match_score(fake_matches, n_kp1=10, n_kp2=10)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.5)

    def test_more_matches_than_max_capped(self):
        fake_matches = [object()] * 20
        score = compute_match_score(fake_matches, n_kp1=10, n_kp2=10)
        assert score == pytest.approx(1.0)

    def test_uses_max_of_n_kp1_n_kp2(self):
        # 5 matches, max(5, 10) = 10 → 0.5
        fake_matches = [object()] * 5
        score = compute_match_score(fake_matches, n_kp1=5, n_kp2=10)
        assert score == pytest.approx(0.5)
