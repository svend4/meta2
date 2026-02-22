"""Extra tests for puzzle_reconstruction.preprocessing.perspective."""
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


# ─── fixtures / helpers ───────────────────────────────────────────────────────

def _rect_img(h=200, w=300):
    """BGR image with a clear dark rectangle on a light background."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 20
    return img


def _blank(h=100, w=150):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _gray(h=200, w=300):
    import cv2
    return cv2.cvtColor(_rect_img(h, w), cv2.COLOR_BGR2GRAY)


def _quad(h=190, w=290, x0=10, y0=10):
    return np.array([
        [x0,      y0],
        [x0 + w,  y0],
        [x0 + w,  y0 + h],
        [x0,      y0 + h],
    ], dtype=np.float32)


# ─── order_corners extras ─────────────────────────────────────────────────────

class TestOrderCornersExtra:
    def test_wide_rectangle(self):
        pts = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], dtype=np.float32)
        r = order_corners(pts)
        assert r.shape == (4, 2)
        assert r.dtype == np.float32

    def test_tall_rectangle(self):
        pts = np.array([[0, 0], [20, 0], [20, 100], [0, 100]], dtype=np.float32)
        r = order_corners(pts)
        assert r.shape == (4, 2)

    def test_large_coordinates(self):
        pts = np.array([[1000, 1000], [2000, 1000],
                        [2000, 1500], [1000, 1500]], dtype=np.float32)
        r = order_corners(pts)
        assert r.dtype == np.float32
        assert r.shape == (4, 2)

    def test_integer_input_accepted(self):
        pts = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.int32)
        r = order_corners(pts)
        assert r.dtype == np.float32

    def test_tl_is_first_row(self):
        pts = np.array([[5, 5], [50, 5], [50, 40], [5, 40]], dtype=np.float32)
        r = order_corners(pts)
        assert r[0, 0] < r[1, 0]   # tl.x < tr.x
        assert r[0, 1] < r[3, 1]   # tl.y < bl.y is debatable; just check no crash

    def test_output_all_unique(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        r = order_corners(pts)
        rows = [tuple(r[i]) for i in range(4)]
        assert len(set(rows)) == 4


# ─── four_point_transform extras ──────────────────────────────────────────────

class TestFourPointTransformExtra:
    def test_non_square_quad(self):
        img = _rect_img()
        pts = np.array([[10, 10], [150, 10], [150, 80], [10, 80]],
                       dtype=np.float32)
        warped, H, dst = four_point_transform(img, pts)
        assert warped.dtype == np.uint8
        assert H.shape == (3, 3)

    def test_large_image(self):
        img = _rect_img(h=500, w=600)
        pts = _quad(h=450, w=550)
        warped, _, _ = four_point_transform(img, pts)
        assert warped.shape[0] > 0

    def test_small_image(self):
        img = _rect_img(h=20, w=20)
        pts = np.array([[1, 1], [18, 1], [18, 18], [1, 18]], dtype=np.float32)
        warped, _, _ = four_point_transform(img, pts)
        assert warped.dtype == np.uint8

    def test_grayscale_output_2d(self):
        gray = _gray()
        pts = _quad()
        warped, _, _ = four_point_transform(gray, pts)
        assert warped.ndim == 2

    def test_homography_invertible(self):
        img = _rect_img()
        pts = _quad()
        _, H, _ = four_point_transform(img, pts)
        det = np.linalg.det(H)
        assert abs(det) > 1e-6

    def test_dst_pts_dtype_float32(self):
        img = _rect_img()
        pts = _quad()
        _, _, dst = four_point_transform(img, pts)
        assert dst.dtype == np.float32

    def test_warped_nonnzero_dimensions(self):
        img = _rect_img()
        pts = _quad()
        warped, _, _ = four_point_transform(img, pts)
        assert warped.shape[0] > 0 and warped.shape[1] > 0


# ─── detect_corners_contour extras ────────────────────────────────────────────

class TestDetectCornersContourExtra:
    def test_very_small_image(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 200
        result = detect_corners_contour(img)
        # Just must not crash; result may be None
        assert result is None or result.shape == (4, 2)

    def test_noisy_image_no_crash(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        result = detect_corners_contour(img)
        if result is not None:
            assert result.shape == (4, 2)

    def test_high_min_area_frac_no_detection(self):
        img = _rect_img()
        result = detect_corners_contour(img, min_area_frac=0.99)
        assert result is None

    def test_rect_img_returns_float32_or_none(self):
        img = _rect_img()
        result = detect_corners_contour(img)
        if result is not None:
            assert result.dtype == np.float32

    def test_gray_input(self):
        gray = _gray()
        result = detect_corners_contour(gray)
        if result is not None:
            assert result.shape == (4, 2)


# ─── detect_corners_hough extras ──────────────────────────────────────────────

class TestDetectCornersHoughExtra:
    def test_very_high_threshold(self):
        img = _rect_img()
        result = detect_corners_hough(img, threshold=100000)
        assert result is None

    def test_gray_input(self):
        gray = _gray()
        result = detect_corners_hough(gray)
        if result is not None:
            assert result.shape == (4, 2)
            assert result.dtype == np.float32

    def test_small_image_no_crash(self):
        img = np.ones((20, 20, 3), dtype=np.uint8) * 200
        result = detect_corners_hough(img)
        assert result is None or result.shape == (4, 2)

    def test_blank_always_none(self):
        result = detect_corners_hough(_blank())
        assert result is None


# ─── correct_perspective extras ───────────────────────────────────────────────

class TestCorrectPerspectiveExtra:
    def test_blank_hough_confidence_zero(self):
        r = correct_perspective(_blank(), method="hough")
        assert r.confidence == pytest.approx(0.0)

    def test_grayscale_hough_no_crash(self):
        r = correct_perspective(_gray(), method="hough")
        assert isinstance(r, PerspectiveResult)

    def test_confidence_range_contour(self):
        r = correct_perspective(_rect_img(), method="contour")
        assert 0.0 <= r.confidence <= 1.0

    def test_confidence_range_hough(self):
        r = correct_perspective(_rect_img(), method="hough")
        assert 0.0 <= r.confidence <= 1.0

    def test_method_hough_stored(self):
        r = correct_perspective(_blank(), method="hough")
        assert r.method == "hough"

    def test_manual_corners_confidence_positive(self):
        pts = _quad()
        r = correct_perspective(_rect_img(), corners=pts, method="manual")
        assert r.confidence > 0.0

    def test_corrected_dtype_uint8(self):
        pts = _quad()
        r = correct_perspective(_rect_img(), corners=pts, method="manual")
        assert r.corrected.dtype == np.uint8

    def test_homography_3x3(self):
        pts = _quad()
        r = correct_perspective(_rect_img(), corners=pts, method="manual")
        assert r.homography.shape == (3, 3)

    def test_repr_is_string(self):
        r = correct_perspective(_blank(), method="contour")
        assert isinstance(repr(r), str)

    def test_src_pts_float32(self):
        pts = _quad()
        r = correct_perspective(_rect_img(), corners=pts, method="manual")
        assert r.src_pts.dtype == np.float32

    def test_dst_pts_float32(self):
        pts = _quad()
        r = correct_perspective(_rect_img(), corners=pts, method="manual")
        assert r.dst_pts.dtype == np.float32


# ─── auto_correct_perspective extras ─────────────────────────────────────────

class TestAutoCorrectPerspectiveExtra:
    def test_confidence_range(self):
        r = auto_correct_perspective(_rect_img())
        assert 0.0 <= r.confidence <= 1.0

    def test_corrected_uint8(self):
        r = auto_correct_perspective(_rect_img())
        assert r.corrected.dtype == np.uint8

    def test_homography_shape(self):
        r = auto_correct_perspective(_rect_img())
        assert r.homography.shape == (3, 3)

    def test_src_dst_pts_shape(self):
        r = auto_correct_perspective(_rect_img())
        assert r.src_pts.shape == (4, 2)
        assert r.dst_pts.shape == (4, 2)

    def test_blank_corrected_shape_same(self):
        img = _blank()
        r = auto_correct_perspective(img)
        assert r.corrected.shape == img.shape

    def test_non_square_image(self):
        img = _rect_img(h=100, w=300)
        r = auto_correct_perspective(img)
        assert isinstance(r, PerspectiveResult)


# ─── batch_correct_perspective extras ────────────────────────────────────────

class TestBatchCorrectPerspectiveExtra:
    def test_five_images(self):
        imgs = [_rect_img() for _ in range(5)]
        results = batch_correct_perspective(imgs)
        assert len(results) == 5

    def test_all_confidence_in_range(self):
        imgs = [_rect_img(), _blank(), _rect_img()]
        for r in batch_correct_perspective(imgs):
            assert 0.0 <= r.confidence <= 1.0

    def test_hough_method_stored(self):
        imgs = [_blank(), _blank()]
        results = batch_correct_perspective(imgs, method="hough")
        assert all(r.method == "hough" for r in results)

    def test_contour_method_stored(self):
        imgs = [_blank()]
        results = batch_correct_perspective(imgs, method="contour")
        assert results[0].method == "contour"

    def test_corrected_all_uint8(self):
        imgs = [_rect_img(), _blank()]
        for r in batch_correct_perspective(imgs):
            assert r.corrected.dtype == np.uint8

    def test_homography_all_3x3(self):
        imgs = [_rect_img(), _blank()]
        for r in batch_correct_perspective(imgs):
            assert r.homography.shape == (3, 3)
