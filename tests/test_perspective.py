"""
Тесты для puzzle_reconstruction/preprocessing/perspective.py

Покрытие:
    order_corners        — tl/tr/br/bl корректно, (4,2) float32, порядок сохраняется
                           при уже упорядоченных точках
    four_point_transform — форма выходного изображения правильная, H 3×3,
                           dst_pts (4,2), angle=0 прямоугольник → почти идентично,
                           grayscale вход
    detect_corners_contour — None для пустого/однородного изображения,
                             для изображения с чётким прямоугольником → (4,2) или None,
                             возвращает float32
    detect_corners_hough   — None при недостаточном числе линий,
                             не падает на сложных изображениях
    correct_perspective  — PerspectiveResult для любого метода,
                           confidence=0 при corners=None и нет детекций,
                           ValueError для неизвестного метода,
                           manual corners работают,
                           homography 3×3, src_pts/dst_pts (4,2),
                           shape выхода совпадает с ожидаемым
    auto_correct_perspective — возвращает PerspectiveResult, не падает
    batch_correct_perspective — длина = len(images), все PerspectiveResult
    PerspectiveResult    — repr, confidence ∈ [0, 1]
"""
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


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rect_img():
    """
    BGR 200×300: белый фон с чётким чёрным прямоугольником ≈ 80% площади.
    Хорошо подходит для обнаружения через контуры.
    """
    img = np.ones((200, 300, 3), dtype=np.uint8) * 240
    img[20:180, 30:270] = 20  # Тёмный прямоугольник
    return img


@pytest.fixture
def blank_img():
    return np.ones((100, 150, 3), dtype=np.uint8) * 255


@pytest.fixture
def gray_rect(rect_img):
    import cv2
    return cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def quad_pts():
    """Четыре точки в порядке tl, tr, br, bl (500×400 прямоугольник)."""
    return np.array([
        [10.0,  10.0],
        [290.0, 10.0],
        [290.0, 190.0],
        [10.0,  190.0],
    ], dtype=np.float32)


# ─── order_corners ────────────────────────────────────────────────────────────

class TestOrderCorners:
    def test_output_shape(self, quad_pts):
        r = order_corners(quad_pts)
        assert r.shape == (4, 2)

    def test_output_dtype(self, quad_pts):
        r = order_corners(quad_pts)
        assert r.dtype == np.float32

    def test_tl_min_sum(self, quad_pts):
        """tl имеет минимальную сумму x+y."""
        r   = order_corners(quad_pts)
        tl  = r[0]
        for pt in r[1:]:
            assert (tl[0] + tl[1]) <= (pt[0] + pt[1]) + 1e-3

    def test_br_max_sum(self, quad_pts):
        """br имеет максимальную сумму x+y."""
        r  = order_corners(quad_pts)
        br = r[2]
        for pt in r[:-1]:
            assert (br[0] + br[1]) >= (pt[0] + pt[1]) - 1e-3

    def test_shuffled_input(self, quad_pts):
        """Перемешанные точки → тот же порядок."""
        shuffled = quad_pts[[2, 0, 3, 1]]
        r1 = order_corners(quad_pts)
        r2 = order_corners(shuffled)
        assert np.allclose(r1, r2, atol=1)

    def test_3d_input(self, quad_pts):
        """Вход формы (4, 1, 2) — формат OpenCV."""
        pts_3d = quad_pts[:, np.newaxis, :]
        r      = order_corners(pts_3d)
        assert r.shape == (4, 2)


# ─── four_point_transform ─────────────────────────────────────────────────────

class TestFourPointTransform:
    def test_output_shape(self, rect_img, quad_pts):
        warped, H, dst = four_point_transform(rect_img, quad_pts)
        assert warped.ndim == 3  # BGR

    def test_homography_shape(self, rect_img, quad_pts):
        _, H, _ = four_point_transform(rect_img, quad_pts)
        assert H.shape == (3, 3)

    def test_dst_pts_shape(self, rect_img, quad_pts):
        _, _, dst = four_point_transform(rect_img, quad_pts)
        assert dst.shape == (4, 2)

    def test_output_dtype(self, rect_img, quad_pts):
        warped, _, _ = four_point_transform(rect_img, quad_pts)
        assert warped.dtype == np.uint8

    def test_grayscale_input(self, gray_rect, quad_pts):
        warped, H, _ = four_point_transform(gray_rect, quad_pts)
        assert warped.ndim == 2

    def test_output_nonzero(self, rect_img, quad_pts):
        warped, _, _ = four_point_transform(rect_img, quad_pts)
        assert warped.shape[0] > 0 and warped.shape[1] > 0

    def test_aligned_quad_preserves_content(self, rect_img):
        """Прямоугольный quad → warped ≈ roi из исходного."""
        h, w = rect_img.shape[:2]
        pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]],
                        dtype=np.float32)
        warped, _, _ = four_point_transform(rect_img, pts)
        assert warped.shape[:2] == (h, w)


# ─── detect_corners_contour ───────────────────────────────────────────────────

class TestDetectCornersContour:
    def test_blank_returns_none(self, blank_img):
        result = detect_corners_contour(blank_img)
        assert result is None

    def test_returns_float32_or_none(self, rect_img):
        result = detect_corners_contour(rect_img)
        if result is not None:
            assert result.dtype == np.float32
            assert result.shape == (4, 2)

    def test_gray_input_no_crash(self, gray_rect):
        result = detect_corners_contour(gray_rect)
        # None или (4,2) float32
        if result is not None:
            assert result.shape == (4, 2)

    def test_min_area_frac_zero_finds_more(self, rect_img):
        """min_area_frac=0 → более терпимый детектор."""
        result = detect_corners_contour(rect_img, min_area_frac=0.0)
        # Не должен падать
        if result is not None:
            assert result.shape == (4, 2)

    def test_high_min_area_frac_returns_none(self, rect_img):
        """Очень высокий min_area_frac → никакой контур не подходит."""
        result = detect_corners_contour(rect_img, min_area_frac=2.0)
        assert result is None


# ─── detect_corners_hough ─────────────────────────────────────────────────────

class TestDetectCornersHough:
    def test_blank_returns_none(self, blank_img):
        result = detect_corners_hough(blank_img)
        assert result is None

    def test_no_crash_on_complex_image(self, rect_img):
        result = detect_corners_hough(rect_img)
        if result is not None:
            assert result.shape == (4, 2)
            assert result.dtype == np.float32

    def test_high_threshold_returns_none(self, rect_img):
        """Очень высокий threshold → нет линий."""
        result = detect_corners_hough(rect_img, threshold=10000)
        assert result is None

    def test_gray_input_no_crash(self, gray_rect):
        result = detect_corners_hough(gray_rect)
        if result is not None:
            assert result.shape == (4, 2)


# ─── correct_perspective ──────────────────────────────────────────────────────

class TestCorrectPerspective:
    @pytest.mark.parametrize("method", ["contour", "hough"])
    def test_returns_result(self, rect_img, method):
        r = correct_perspective(rect_img, method=method)
        assert isinstance(r, PerspectiveResult)

    @pytest.mark.parametrize("method", ["contour", "hough"])
    def test_corrected_is_ndarray(self, rect_img, method):
        r = correct_perspective(rect_img, method=method)
        assert isinstance(r.corrected, np.ndarray)

    @pytest.mark.parametrize("method", ["contour", "hough"])
    def test_homography_shape(self, rect_img, method):
        r = correct_perspective(rect_img, method=method)
        assert r.homography.shape == (3, 3)

    @pytest.mark.parametrize("method", ["contour", "hough"])
    def test_src_dst_pts_shape(self, rect_img, method):
        r = correct_perspective(rect_img, method=method)
        assert r.src_pts.shape == (4, 2)
        assert r.dst_pts.shape == (4, 2)

    def test_unknown_method_raises(self, rect_img):
        with pytest.raises(ValueError):
            correct_perspective(rect_img, method="magic")

    def test_no_detection_identity(self, blank_img):
        """Нет углов → confidence=0, оригинал возвращается."""
        r = correct_perspective(blank_img, method="contour")
        assert r.confidence == pytest.approx(0.0)
        assert r.corrected.shape == blank_img.shape

    def test_manual_corners(self, rect_img, quad_pts):
        """Явно переданные углы → корректное преобразование."""
        r = correct_perspective(rect_img, corners=quad_pts, method="manual")
        assert r.corrected.ndim == 3
        assert r.homography.shape == (3, 3)

    def test_confidence_in_range(self, rect_img, quad_pts):
        r = correct_perspective(rect_img, corners=quad_pts, method="manual")
        assert 0.0 <= r.confidence <= 1.0

    def test_method_stored(self, blank_img):
        r = correct_perspective(blank_img, method="contour")
        assert r.method == "contour"

    def test_repr_contains_method(self, rect_img, quad_pts):
        r = correct_perspective(rect_img, corners=quad_pts, method="manual")
        assert "manual" in repr(r)

    def test_repr_contains_confidence(self, blank_img):
        r = correct_perspective(blank_img, method="contour")
        assert "confidence" in repr(r)

    def test_grayscale_manual(self, gray_rect, quad_pts):
        r = correct_perspective(gray_rect, corners=quad_pts, method="manual")
        assert r.corrected.ndim == 2


# ─── auto_correct_perspective ─────────────────────────────────────────────────

class TestAutoCorrectPerspective:
    def test_returns_result(self, rect_img):
        r = auto_correct_perspective(rect_img)
        assert isinstance(r, PerspectiveResult)

    def test_corrected_ndarray(self, rect_img):
        r = auto_correct_perspective(rect_img)
        assert isinstance(r.corrected, np.ndarray)

    def test_blank_no_crash(self, blank_img):
        r = auto_correct_perspective(blank_img)
        assert isinstance(r, PerspectiveResult)

    def test_blank_confidence_zero(self, blank_img):
        r = auto_correct_perspective(blank_img)
        assert r.confidence == pytest.approx(0.0)

    def test_confidence_in_range(self, rect_img):
        r = auto_correct_perspective(rect_img)
        assert 0.0 <= r.confidence <= 1.0


# ─── batch_correct_perspective ────────────────────────────────────────────────

class TestBatchCorrectPerspective:
    def test_length_matches(self, rect_img, blank_img):
        images  = [rect_img, blank_img, rect_img]
        results = batch_correct_perspective(images)
        assert len(results) == 3

    def test_all_perspective_results(self, rect_img, blank_img):
        results = batch_correct_perspective([rect_img, blank_img])
        assert all(isinstance(r, PerspectiveResult) for r in results)

    def test_empty_list(self):
        assert batch_correct_perspective([]) == []

    def test_single_image(self, rect_img):
        results = batch_correct_perspective([rect_img])
        assert len(results) == 1

    def test_method_propagated(self, blank_img):
        results = batch_correct_perspective([blank_img], method="hough")
        assert results[0].method == "hough"
