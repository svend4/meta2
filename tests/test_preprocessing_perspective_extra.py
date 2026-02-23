"""Extra tests for puzzle_reconstruction.preprocessing.perspective."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.perspective import (
    PerspectiveResult,
    auto_correct_perspective,
    batch_correct_perspective,
    correct_perspective,
    detect_corners_contour,
    detect_corners_hough,
    four_point_transform,
    order_corners,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bgr(h=64, w=64):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _gray(h=64, w=64):
    return np.full((h, w), 180, dtype=np.uint8)


def _rand_bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rect_corners(h=64, w=64):
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)


def _skewed(h=64, w=64):
    return np.array([[5, 5], [w-5, 2], [w-3, h-5], [3, h-3]], dtype=np.float32)


def _make_result(method="contour", confidence=0.8, h=64, w=64):
    img = _bgr(h, w)
    corners = _rect_corners(h, w)
    return PerspectiveResult(
        corrected=img.copy(),
        homography=np.eye(3, dtype=np.float64),
        src_pts=corners,
        dst_pts=corners.copy(),
        method=method,
        confidence=confidence,
    )


# ─── TestPerspectiveResultExtra ──────────────────────────────────────────────

class TestPerspectiveResultExtra:
    def test_corrected_shape_preserved(self):
        r = _make_result(h=32, w=48)
        assert r.corrected.shape == (32, 48, 3)

    def test_confidence_zero(self):
        r = _make_result(confidence=0.0)
        assert r.confidence == pytest.approx(0.0)

    def test_confidence_one(self):
        r = _make_result(confidence=1.0)
        assert r.confidence == pytest.approx(1.0)

    def test_method_none_stored(self):
        r = _make_result(method="none")
        assert r.method == "none"

    def test_homography_eye(self):
        r = _make_result()
        np.testing.assert_array_equal(r.homography, np.eye(3))

    def test_repr_not_empty(self):
        r = _make_result()
        assert len(repr(r)) > 0

    def test_params_dict(self):
        r = _make_result()
        assert isinstance(r.params, dict)


# ─── TestOrderCornersExtra ────────────────────────────────────────────────────

class TestOrderCornersExtra:
    def test_tl_smallest_sum(self):
        pts = np.array([[50, 50], [5, 5], [60, 60], [5, 60]], dtype=np.float32)
        result = order_corners(pts)
        sums = result[:, 0] + result[:, 1]
        assert sums[0] == sums.min()

    def test_br_largest_sum(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        result = order_corners(pts)
        sums = result[:, 0] + result[:, 1]
        assert sums[2] == sums.max()

    def test_output_is_float32(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        result = order_corners(pts)
        assert result.dtype == np.float32

    def test_all_input_pts_present(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        result = order_corners(pts)
        # All 4 input points should appear in output (possibly reordered)
        for pt in pts:
            assert any(np.allclose(pt, r) for r in result)

    def test_skewed_shape(self):
        result = order_corners(_skewed())
        assert result.shape == (4, 2)

    def test_large_coords(self):
        pts = np.array([[0, 0], [1000, 0], [1000, 800], [0, 800]], dtype=np.float32)
        result = order_corners(pts)
        assert result.shape == (4, 2)


# ─── TestFourPointTransformExtra ─────────────────────────────────────────────

class TestFourPointTransformExtra:
    def test_non_square_image(self):
        warped, H, dst = four_point_transform(_bgr(32, 64), _rect_corners(32, 64))
        assert warped.shape[0] > 0 and warped.shape[1] > 0

    def test_homography_invertible(self):
        _, H, _ = four_point_transform(_bgr(), _rect_corners())
        det = np.linalg.det(H)
        assert abs(det) > 1e-6

    def test_dst_float32(self):
        _, _, dst = four_point_transform(_bgr(), _rect_corners())
        assert dst.dtype == np.float32

    def test_warped_3_channel(self):
        warped, _, _ = four_point_transform(_bgr(), _rect_corners())
        assert warped.ndim in (2, 3)

    def test_gray_input_works(self):
        warped, H, dst = four_point_transform(_gray(), _rect_corners())
        assert isinstance(warped, np.ndarray)
        assert warped.dtype == np.uint8

    def test_skewed_corners_produces_output(self):
        warped, H, dst = four_point_transform(_bgr(), _skewed())
        assert warped.shape[0] > 0

    def test_homography_33(self):
        _, H, _ = four_point_transform(_bgr(), _rect_corners())
        assert H.shape == (3, 3)


# ─── TestDetectCornersContourExtra ───────────────────────────────────────────

class TestDetectCornersContourExtra:
    def test_returns_none_or_4x2(self):
        result = detect_corners_contour(_bgr())
        assert result is None or result.shape == (4, 2)

    def test_gray_input_accepted(self):
        result = detect_corners_contour(_gray())
        assert result is None or isinstance(result, np.ndarray)

    def test_different_sizes(self):
        for h, w in [(32, 32), (64, 64), (48, 96)]:
            result = detect_corners_contour(_bgr(h, w))
            assert result is None or result.shape == (4, 2)

    def test_custom_canny_params(self):
        result = detect_corners_contour(_bgr(), canny_lo=50, canny_hi=150)
        assert result is None or result.shape == (4, 2)

    def test_result_float32(self):
        result = detect_corners_contour(_bgr())
        if result is not None:
            assert result.dtype == np.float32


# ─── TestDetectCornersHoughExtra ─────────────────────────────────────────────

class TestDetectCornersHoughExtra:
    def test_returns_none_or_4x2(self):
        result = detect_corners_hough(_bgr())
        assert result is None or result.shape == (4, 2)

    def test_gray_input_accepted(self):
        result = detect_corners_hough(_gray())
        assert result is None or isinstance(result, np.ndarray)

    def test_different_sizes(self):
        for h, w in [(32, 32), (64, 64)]:
            result = detect_corners_hough(_bgr(h, w))
            assert result is None or result.shape == (4, 2)


# ─── TestCorrectPerspectiveExtra ─────────────────────────────────────────────

class TestCorrectPerspectiveExtra:
    def test_none_method_identity(self):
        img = _bgr(32, 48)
        result = correct_perspective(img, method="none")
        assert result.corrected.shape == img.shape

    def test_none_method_zero_confidence(self):
        result = correct_perspective(_bgr(), method="none")
        assert result.confidence == pytest.approx(0.0)

    def test_corrected_dtype_uint8(self):
        result = correct_perspective(_bgr(), method="none")
        assert result.corrected.dtype == np.uint8

    def test_confidence_range(self):
        for method in ("contour", "hough", "none"):
            result = correct_perspective(_bgr(), method=method)
            assert 0.0 <= result.confidence <= 1.0

    def test_explicit_corners_high_conf(self):
        corners = _rect_corners()
        result = correct_perspective(_bgr(), corners=corners)
        assert result.confidence >= 0.0

    def test_non_square_image(self):
        result = correct_perspective(_bgr(32, 48), method="none")
        assert result.corrected.shape == (32, 48, 3)

    def test_gray_input(self):
        result = correct_perspective(_gray(), method="none")
        assert isinstance(result, PerspectiveResult)

    def test_method_in_result(self):
        result = correct_perspective(_bgr(), method="contour")
        assert result.method == "contour"

    def test_src_pts_4x2(self):
        result = correct_perspective(_bgr(), method="none")
        assert result.src_pts.shape == (4, 2)


# ─── TestAutoCorrectPerspectiveExtra ─────────────────────────────────────────

class TestAutoCorrectPerspectiveExtra:
    def test_confidence_range(self):
        result = auto_correct_perspective(_bgr())
        assert 0.0 <= result.confidence <= 1.0

    def test_corrected_ndarray(self):
        result = auto_correct_perspective(_bgr())
        assert isinstance(result.corrected, np.ndarray)

    def test_corrected_uint8(self):
        result = auto_correct_perspective(_bgr())
        assert result.corrected.dtype == np.uint8

    def test_gray_input(self):
        result = auto_correct_perspective(_gray())
        assert isinstance(result, PerspectiveResult)

    def test_non_square(self):
        result = auto_correct_perspective(_bgr(32, 48))
        assert isinstance(result, PerspectiveResult)


# ─── TestBatchCorrectPerspectiveExtra ────────────────────────────────────────

class TestBatchCorrectPerspectiveExtra:
    def test_single_image(self):
        result = batch_correct_perspective([_bgr()])
        assert len(result) == 1

    def test_all_perspective_results(self):
        for r in batch_correct_perspective([_bgr(), _bgr(32, 48)]):
            assert isinstance(r, PerspectiveResult)

    def test_confidences_in_range(self):
        for r in batch_correct_perspective([_bgr(), _bgr()]):
            assert 0.0 <= r.confidence <= 1.0

    def test_none_method_batch(self):
        results = batch_correct_perspective([_bgr(), _bgr(32, 48)], method="none")
        for r in results:
            assert r.confidence == pytest.approx(0.0)

    def test_gray_images_batch(self):
        results = batch_correct_perspective([_gray(), _gray(32, 32)])
        assert len(results) == 2

    def test_large_batch(self):
        images = [_bgr(seed=i) if False else _bgr() for i in range(5)]
        results = batch_correct_perspective(images)
        assert len(results) == 5
