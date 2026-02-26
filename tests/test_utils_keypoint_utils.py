"""Tests for puzzle_reconstruction.utils.keypoint_utils
Using only numpy and pytest (no direct cv2 in assertions).
"""
import numpy as np
import pytest
import cv2
from puzzle_reconstruction.utils.keypoint_utils import (
    KeypointSet, detect_keypoints, filter_by_response,
    filter_by_region, keypoints_to_array, array_to_keypoints,
    compute_match_score, match_descriptors, describe_keypoints,
)

np.random.seed(42)


def _make_gray_img(h=128, w=128):
    """Create a synthetic grayscale image with some features."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Add some corners/edges
    img[20:40, 20:40] = 200
    img[60:80, 60:80] = 150
    img[10:30, 70:90] = 180
    return img


def _make_bgr_img(h=128, w=128):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[20:40, 20:40] = (200, 100, 50)
    img[60:80, 60:80] = (50, 200, 100)
    return img


# ── KeypointSet ───────────────────────────────────────────────────────────────

def test_keypoint_set_len():
    kps = [cv2.KeyPoint(10.0, 20.0, 5.0) for _ in range(7)]
    ks = KeypointSet(keypoints=kps, descriptors=None)
    assert len(ks) == 7


def test_keypoint_set_empty():
    ks = KeypointSet(keypoints=[], descriptors=None)
    assert len(ks) == 0


# ── detect_keypoints ──────────────────────────────────────────────────────────

def test_detect_keypoints_returns_keypoint_set():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=50)
    assert isinstance(ks, KeypointSet)


def test_detect_keypoints_bgr_image():
    img = _make_bgr_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=50)
    assert isinstance(ks, KeypointSet)


def test_detect_keypoints_max_keypoints_respected():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=10)
    assert len(ks) <= 10


def test_detect_keypoints_invalid_detector_raises():
    img = _make_gray_img()
    with pytest.raises(ValueError):
        detect_keypoints(img, detector="unknown_det")


def test_detect_keypoints_invalid_max_keypoints_raises():
    img = _make_gray_img()
    with pytest.raises(ValueError):
        detect_keypoints(img, max_keypoints=0)


def test_detect_keypoints_detector_stored():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb")
    assert ks.detector == "orb"


# ── filter_by_response ────────────────────────────────────────────────────────

def test_filter_by_response_min_response():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=100)
    if len(ks) == 0:
        pytest.skip("No keypoints detected")
    max_r = max(kp.response for kp in ks.keypoints)
    filtered = filter_by_response(ks, min_response=max_r)
    assert all(kp.response >= max_r for kp in filtered.keypoints)


def test_filter_by_response_top_k():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=100)
    if len(ks) < 5:
        pytest.skip("Not enough keypoints")
    filtered = filter_by_response(ks, top_k=5)
    assert len(filtered) <= 5


def test_filter_by_response_negative_top_k_raises():
    ks = KeypointSet(keypoints=[], descriptors=None)
    with pytest.raises(ValueError):
        filter_by_response(ks, top_k=-1)


def test_filter_by_response_zero_min():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=50)
    filtered = filter_by_response(ks, min_response=0.0)
    assert len(filtered) == len(ks)


# ── filter_by_region ──────────────────────────────────────────────────────────

def test_filter_by_region_full_mask():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=50)
    mask = np.ones((128, 128), dtype=np.uint8)
    filtered = filter_by_region(ks, mask)
    assert len(filtered) == len(ks)


def test_filter_by_region_empty_mask():
    img = _make_gray_img()
    ks = detect_keypoints(img, detector="orb", max_keypoints=50)
    mask = np.zeros((128, 128), dtype=np.uint8)
    filtered = filter_by_region(ks, mask)
    assert len(filtered) == 0


def test_filter_by_region_non_2d_mask_raises():
    ks = KeypointSet(keypoints=[], descriptors=None)
    with pytest.raises(ValueError):
        filter_by_region(ks, np.ones((10, 10, 3), dtype=np.uint8))


# ── keypoints_to_array ────────────────────────────────────────────────────────

def test_keypoints_to_array_empty():
    arr = keypoints_to_array([])
    assert arr.shape == (0, 2)
    assert arr.dtype == np.float32


def test_keypoints_to_array_shape():
    kps = [cv2.KeyPoint(float(i), float(i*2), 1.0) for i in range(5)]
    arr = keypoints_to_array(kps)
    assert arr.shape == (5, 2)


def test_keypoints_to_array_values():
    kps = [cv2.KeyPoint(3.0, 7.0, 1.0)]
    arr = keypoints_to_array(kps)
    assert arr[0, 0] == pytest.approx(3.0)
    assert arr[0, 1] == pytest.approx(7.0)


# ── array_to_keypoints ────────────────────────────────────────────────────────

def test_array_to_keypoints_empty():
    kps = array_to_keypoints(np.empty((0, 2), dtype=np.float32))
    assert kps == []


def test_array_to_keypoints_roundtrip():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    kps = array_to_keypoints(pts)
    arr = keypoints_to_array(kps)
    assert np.allclose(arr, pts, atol=1e-5)


def test_array_to_keypoints_count():
    pts = np.random.rand(8, 2).astype(np.float32)
    kps = array_to_keypoints(pts)
    assert len(kps) == 8


# ── compute_match_score ───────────────────────────────────────────────────────

def test_compute_match_score_zero_kp():
    score = compute_match_score([], 0, 0)
    assert score == 0.0


def test_compute_match_score_perfect():
    matches = [object()] * 5  # 5 dummy matches
    score = compute_match_score(matches, 5, 5)
    assert score == pytest.approx(1.0)


def test_compute_match_score_range():
    matches = [object()] * 3
    score = compute_match_score(matches, 10, 10)
    assert 0.0 <= score <= 1.0


def test_compute_match_score_negative_n_raises():
    with pytest.raises(ValueError):
        compute_match_score([], -1, 5)


def test_compute_match_score_both_negative_raises():
    with pytest.raises(ValueError):
        compute_match_score([], -1, -1)


# ── describe_keypoints ────────────────────────────────────────────────────────

def test_describe_keypoints_empty_returns_none():
    img = _make_gray_img()
    result = describe_keypoints(img, [], detector="orb")
    assert result is None


def test_describe_keypoints_invalid_detector_raises():
    img = _make_gray_img()
    kps = [cv2.KeyPoint(30.0, 30.0, 5.0)]
    with pytest.raises(ValueError):
        describe_keypoints(img, kps, detector="surf")
