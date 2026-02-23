"""Extra tests for puzzle_reconstruction.preprocessing.edge_detector."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.edge_detector import (
    EdgeDetectionResult,
    adaptive_canny,
    detect_edges,
    edge_density,
    edge_orientation_hist,
    laplacian_edges,
    refine_edge_contour,
    sobel_edges,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr_rect(h=64, w=64):
    """BGR with dark rectangle on white background — clear edges."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 220
    img[20:44, 20:44] = 30
    return img


def _gray_rect(h=64, w=64):
    return cv2.cvtColor(_bgr_rect(h, w), cv2.COLOR_BGR2GRAY)


def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _full_white(h=64, w=64):
    return np.ones((h, w), dtype=np.uint8) * 255


def _uniform_bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ─── TestEdgeDetectionResultExtra ─────────────────────────────────────────────

class TestEdgeDetectionResultExtra:
    def test_edge_map_shape_stored(self):
        em = _blank(48, 32)
        r = EdgeDetectionResult(edge_map=em, method="test")
        assert r.edge_map.shape == (48, 32)

    def test_method_stored(self):
        r = EdgeDetectionResult(edge_map=_blank(), method="canny")
        assert r.method == "canny"

    def test_density_between_0_and_1(self):
        em = np.zeros((100, 100), dtype=np.uint8)
        em[:25, :] = 255
        r = EdgeDetectionResult(edge_map=em, method="test")
        assert 0.0 <= r.density <= 1.0

    def test_n_contours_nonneg(self):
        r = EdgeDetectionResult(edge_map=_blank(), method="test")
        assert r.n_contours >= 0

    def test_params_default_empty(self):
        r = EdgeDetectionResult(edge_map=_blank(), method="test")
        assert r.params == {}

    def test_repr_is_string(self):
        r = EdgeDetectionResult(edge_map=_blank(), method="canny")
        assert isinstance(repr(r), str)


# ─── TestDetectEdgesExtra ─────────────────────────────────────────────────────

class TestDetectEdgesExtra:
    def test_auto_returns_result(self):
        result = detect_edges(_bgr_rect(), method="auto")
        assert isinstance(result, EdgeDetectionResult)

    def test_canny_gray_input(self):
        result = detect_edges(_gray_rect(), method="canny")
        assert result.edge_map.shape == _gray_rect().shape[:2]

    def test_sobel_gray_input(self):
        result = detect_edges(_gray_rect(), method="sobel")
        assert result.edge_map.dtype == np.uint8

    def test_all_methods_binary_output(self):
        for method in ("canny", "adaptive_canny", "sobel"):
            result = detect_edges(_bgr_rect(), method=method)
            unique = set(np.unique(result.edge_map))
            assert unique.issubset({0, 255})

    def test_non_square_image(self):
        result = detect_edges(_bgr_rect(32, 48), method="adaptive_canny")
        assert result.edge_map.shape == (32, 48)


# ─── TestAdaptiveCannyExtra ───────────────────────────────────────────────────

class TestAdaptiveCannyExtra:
    def test_density_positive_on_edges(self):
        r = adaptive_canny(_bgr_rect())
        assert r.density > 0.0

    def test_method_label_correct(self):
        r = adaptive_canny(_bgr_rect())
        assert r.method == "adaptive_canny"

    def test_blur_k_1_ok(self):
        r = adaptive_canny(_bgr_rect(), blur_k=1)
        assert r.edge_map.shape == (64, 64)

    def test_blur_k_7_ok(self):
        r = adaptive_canny(_bgr_rect(), blur_k=7)
        assert r.edge_map.shape == (64, 64)

    def test_non_square_image(self):
        r = adaptive_canny(_bgr_rect(32, 48))
        assert r.edge_map.shape == (32, 48)

    def test_uniform_image_low_density(self):
        r = adaptive_canny(_uniform_bgr())
        assert r.density < 0.1


# ─── TestSobelEdgesExtra ──────────────────────────────────────────────────────

class TestSobelEdgesExtra:
    def test_rect_has_edges(self):
        r = sobel_edges(_bgr_rect(), threshold=30)
        assert r.density > 0.0

    def test_method_label(self):
        r = sobel_edges(_bgr_rect())
        assert r.method == "sobel"

    def test_ksize_5(self):
        r = sobel_edges(_bgr_rect(), ksize=5)
        assert r.edge_map.shape == (64, 64)

    def test_non_square_image(self):
        r = sobel_edges(_bgr_rect(32, 48))
        assert r.edge_map.shape == (32, 48)

    def test_uniform_low_density(self):
        r = sobel_edges(_uniform_bgr(), threshold=50)
        assert r.density < 0.05


# ─── TestRefineEdgeContourExtra ───────────────────────────────────────────────

class TestRefineEdgeContourExtra:
    def test_output_shape_preserved(self):
        em = adaptive_canny(_bgr_rect()).edge_map
        ref = refine_edge_contour(em)
        assert ref.shape == em.shape

    def test_full_white_stays_some(self):
        ref = refine_edge_contour(_full_white(), min_area=0)
        # should not be all zeros for big area
        assert ref.sum() > 0

    def test_close_iter_3_ok(self):
        em = adaptive_canny(_bgr_rect()).edge_map
        ref = refine_edge_contour(em, close_iter=3)
        assert ref.shape == em.shape

    def test_dilate_iter_2_ok(self):
        em = adaptive_canny(_bgr_rect()).edge_map
        ref = refine_edge_contour(em, dilate_iter=2)
        assert ref.shape == em.shape

    def test_min_area_0_keeps_all(self):
        em = np.zeros((64, 64), dtype=np.uint8)
        em[32, 32] = 255
        ref = refine_edge_contour(em, close_iter=0, dilate_iter=0, min_area=0)
        assert ref.sum() > 0


# ─── TestEdgeDensityExtra ─────────────────────────────────────────────────────

class TestEdgeDensityExtra:
    def test_half_white_approx_half(self):
        em = np.zeros((100, 100), dtype=np.uint8)
        em[:50, :] = 255
        d = edge_density(em, edge_map=em)
        assert abs(d - 0.5) < 0.01

    def test_quarter_white_approx_quarter(self):
        em = np.zeros((100, 100), dtype=np.uint8)
        em[:25, :] = 255
        d = edge_density(em, edge_map=em)
        assert abs(d - 0.25) < 0.01

    def test_bgr_input_auto(self):
        d = edge_density(_bgr_rect())
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        d = edge_density(_bgr_rect())
        assert isinstance(d, float)


# ─── TestEdgeOrientationHistExtra ─────────────────────────────────────────────

class TestEdgeOrientationHistExtra:
    def test_custom_bins_16(self):
        h = edge_orientation_hist(_bgr_rect(), n_bins=16)
        assert h.shape == (16,)

    def test_custom_bins_4(self):
        h = edge_orientation_hist(_bgr_rect(), n_bins=4)
        assert h.shape == (4,)

    def test_values_nonneg(self):
        h = edge_orientation_hist(_bgr_rect())
        assert (h >= 0).all()

    def test_gray_input(self):
        h = edge_orientation_hist(_gray_rect())
        assert h.shape == (8,)

    def test_uniform_image_near_zero(self):
        h = edge_orientation_hist(_uniform_bgr(), normalize=False)
        assert h.sum() == pytest.approx(0.0, abs=1.0)

    def test_dtype_float(self):
        h = edge_orientation_hist(_bgr_rect())
        assert h.dtype in (np.float32, np.float64)
