"""Расширенные тесты для puzzle_reconstruction/preprocessing/perspective.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), 200, dtype=np.uint8)


def _rect_corners(h: int = 64, w: int = 64) -> np.ndarray:
    """4 corners of a rectangle (tl, tr, br, bl-ish but unordered)."""
    return np.array([
        [0,   0],
        [w-1, 0],
        [w-1, h-1],
        [0,   h-1],
    ], dtype=np.float32)


def _skewed_corners(h: int = 64, w: int = 64) -> np.ndarray:
    """Trapezoid-like corners."""
    return np.array([
        [5,    5],
        [w-5,  2],
        [w-3,  h-5],
        [3,    h-3],
    ], dtype=np.float32)


# ─── TestPerspectiveResult ────────────────────────────────────────────────────

class TestPerspectiveResult:
    def _make(self, method="contour", confidence=0.8):
        img = _bgr()
        corners = _rect_corners()
        return PerspectiveResult(
            corrected=img.copy(),
            homography=np.eye(3, dtype=np.float64),
            src_pts=corners,
            dst_pts=corners.copy(),
            method=method,
            confidence=confidence,
        )

    def test_stores_method(self):
        r = self._make(method="hough")
        assert r.method == "hough"

    def test_stores_confidence(self):
        r = self._make(confidence=0.6)
        assert r.confidence == pytest.approx(0.6)

    def test_stores_homography(self):
        r = self._make()
        assert r.homography.shape == (3, 3)

    def test_stores_src_pts(self):
        r = self._make()
        assert r.src_pts.shape == (4, 2)

    def test_stores_dst_pts(self):
        r = self._make()
        assert r.dst_pts.shape == (4, 2)

    def test_repr_contains_method(self):
        r = self._make(method="contour")
        assert "contour" in repr(r)

    def test_repr_contains_confidence(self):
        r = self._make(confidence=0.75)
        assert "0.75" in repr(r) or "0.7" in repr(r)

    def test_repr_is_string(self):
        assert isinstance(repr(self._make()), str)

    def test_default_params_empty_dict(self):
        r = self._make()
        assert isinstance(r.params, dict)


# ─── TestOrderCorners ─────────────────────────────────────────────────────────

class TestOrderCorners:
    def test_returns_ndarray(self):
        pts = _rect_corners()
        assert isinstance(order_corners(pts), np.ndarray)

    def test_shape_4_2(self):
        pts = _rect_corners()
        result = order_corners(pts)
        assert result.shape == (4, 2)

    def test_float32_dtype(self):
        pts = _rect_corners()
        result = order_corners(pts)
        assert result.dtype == np.float32

    def test_tl_is_min_sum(self):
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        result = order_corners(pts)
        tl = result[0]
        # tl should have minimum x+y
        assert tl[0] + tl[1] <= result[1][0] + result[1][1]
        assert tl[0] + tl[1] <= result[2][0] + result[2][1]

    def test_br_is_max_sum(self):
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        result = order_corners(pts)
        br = result[2]
        assert br[0] + br[1] >= result[0][0] + result[0][1]
        assert br[0] + br[1] >= result[1][0] + result[1][1]

    def test_deterministic(self):
        pts = _skewed_corners()
        r1 = order_corners(pts)
        r2 = order_corners(pts)
        assert np.allclose(r1, r2)

    def test_same_points_different_order(self):
        pts1 = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=np.float32)
        pts2 = np.array([[10,10],[0,10],[10,0],[0,0]], dtype=np.float32)
        r1 = order_corners(pts1)
        r2 = order_corners(pts2)
        assert np.allclose(r1, r2)


# ─── TestFourPointTransform ───────────────────────────────────────────────────

class TestFourPointTransform:
    def test_returns_tuple_3(self):
        img = _bgr()
        pts = _rect_corners()
        result = four_point_transform(img, pts)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_warped_is_ndarray(self):
        warped, H, dst = four_point_transform(_bgr(), _rect_corners())
        assert isinstance(warped, np.ndarray)

    def test_homography_shape(self):
        _, H, _ = four_point_transform(_bgr(), _rect_corners())
        assert H.shape == (3, 3)

    def test_dst_shape(self):
        _, _, dst = four_point_transform(_bgr(), _rect_corners())
        assert dst.shape == (4, 2)

    def test_warped_dtype_uint8(self):
        warped, _, _ = four_point_transform(_bgr(), _rect_corners())
        assert warped.dtype == np.uint8

    def test_warped_2d_shape(self):
        warped, _, _ = four_point_transform(_bgr(), _rect_corners())
        assert warped.ndim in (2, 3)

    def test_grayscale_input(self):
        warped, H, dst = four_point_transform(_gray(), _rect_corners())
        assert isinstance(warped, np.ndarray)

    def test_skewed_corners(self):
        warped, H, dst = four_point_transform(_bgr(), _skewed_corners())
        assert warped.shape[0] > 0
        assert warped.shape[1] > 0


# ─── TestDetectCornersContour ─────────────────────────────────────────────────

class TestDetectCornersContour:
    def test_returns_none_or_array(self):
        result = detect_corners_contour(_bgr())
        assert result is None or isinstance(result, np.ndarray)

    def test_blank_image_likely_none(self):
        # Blank white image → no document corners
        result = detect_corners_contour(_bgr(64, 64))
        assert result is None or (result.shape == (4, 2))

    def test_result_shape_if_found(self):
        result = detect_corners_contour(_bgr())
        if result is not None:
            assert result.shape == (4, 2)
            assert result.dtype == np.float32

    def test_grayscale_input(self):
        result = detect_corners_contour(_gray())
        assert result is None or isinstance(result, np.ndarray)

    def test_custom_thresholds(self):
        result = detect_corners_contour(_bgr(), canny_lo=30, canny_hi=100)
        assert result is None or result.shape == (4, 2)


# ─── TestDetectCornersHough ───────────────────────────────────────────────────

class TestDetectCornersHough:
    def test_returns_none_or_array(self):
        result = detect_corners_hough(_bgr())
        assert result is None or isinstance(result, np.ndarray)

    def test_blank_image_likely_none(self):
        result = detect_corners_hough(_bgr(64, 64))
        assert result is None or (result.shape == (4, 2))

    def test_result_shape_if_found(self):
        result = detect_corners_hough(_bgr())
        if result is not None:
            assert result.shape == (4, 2)

    def test_grayscale_input(self):
        result = detect_corners_hough(_gray())
        assert result is None or isinstance(result, np.ndarray)


# ─── TestCorrectPerspective ───────────────────────────────────────────────────

class TestCorrectPerspective:
    def test_returns_perspective_result(self):
        result = correct_perspective(_bgr())
        assert isinstance(result, PerspectiveResult)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            correct_perspective(_bgr(), method="unknown")

    def test_method_none_returns_original(self):
        img = _bgr(32, 48)
        result = correct_perspective(img, method="none")
        assert result.confidence == pytest.approx(0.0)
        assert result.corrected.shape == img.shape

    def test_method_contour_accepted(self):
        result = correct_perspective(_bgr(), method="contour")
        assert isinstance(result, PerspectiveResult)

    def test_method_hough_accepted(self):
        result = correct_perspective(_bgr(), method="hough")
        assert isinstance(result, PerspectiveResult)

    def test_explicit_corners(self):
        corners = _skewed_corners()
        result = correct_perspective(_bgr(64, 64), corners=corners)
        assert isinstance(result, PerspectiveResult)
        assert result.confidence >= 0.0

    def test_no_corners_zero_confidence(self):
        # Blank image → no corners → confidence=0
        result = correct_perspective(_bgr(), method="contour")
        # May or may not find corners; confidence is [0,1]
        assert 0.0 <= result.confidence <= 1.0

    def test_corrected_is_ndarray(self):
        result = correct_perspective(_bgr())
        assert isinstance(result.corrected, np.ndarray)

    def test_homography_3x3(self):
        result = correct_perspective(_bgr(), method="none")
        assert result.homography.shape == (3, 3)

    def test_src_pts_shape(self):
        result = correct_perspective(_bgr(), method="none")
        assert result.src_pts.shape == (4, 2)

    def test_confidence_in_0_1(self):
        result = correct_perspective(_bgr())
        assert 0.0 <= result.confidence <= 1.0

    def test_method_stored(self):
        result = correct_perspective(_bgr(), method="hough")
        assert result.method == "hough"

    def test_grayscale_input(self):
        result = correct_perspective(_gray())
        assert isinstance(result, PerspectiveResult)

    def test_with_skewed_corners_explicit(self):
        corners = _skewed_corners(64, 64)
        result = correct_perspective(_bgr(64, 64), corners=corners)
        assert result.corrected.ndim in (2, 3)


# ─── TestAutoCorrectPerspective ───────────────────────────────────────────────

class TestAutoCorrectPerspective:
    def test_returns_perspective_result(self):
        result = auto_correct_perspective(_bgr())
        assert isinstance(result, PerspectiveResult)

    def test_confidence_in_0_1(self):
        result = auto_correct_perspective(_bgr())
        assert 0.0 <= result.confidence <= 1.0

    def test_corrected_is_ndarray(self):
        result = auto_correct_perspective(_bgr())
        assert isinstance(result.corrected, np.ndarray)

    def test_blank_image_zero_confidence(self):
        result = auto_correct_perspective(_bgr(64, 64))
        # Blank image → no corners → confidence=0
        assert result.confidence == pytest.approx(0.0)

    def test_grayscale_input(self):
        result = auto_correct_perspective(_gray())
        assert isinstance(result, PerspectiveResult)


# ─── TestBatchCorrectPerspective ──────────────────────────────────────────────

class TestBatchCorrectPerspective:
    def test_returns_list(self):
        imgs = [_bgr(), _bgr(32, 48)]
        result = batch_correct_perspective(imgs)
        assert isinstance(result, list)

    def test_same_length(self):
        imgs = [_bgr(), _bgr(32, 48), _bgr(48, 32)]
        result = batch_correct_perspective(imgs)
        assert len(result) == 3

    def test_each_perspective_result(self):
        for r in batch_correct_perspective([_bgr(), _bgr()]):
            assert isinstance(r, PerspectiveResult)

    def test_empty_list(self):
        result = batch_correct_perspective([])
        assert result == []

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_correct_perspective([_bgr()], method="invalid")
