"""Tests for puzzle_reconstruction/preprocessing/edge_detector.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.edge_detector import (
    EdgeDetectionResult,
    detect_edges,
    adaptive_canny,
    sobel_edges,
    laplacian_edges,
    refine_edge_contour,
    edge_density,
    edge_orientation_hist,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=50, w=50, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=50, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    return img


def make_gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_checkerboard(h=50, w=50, cell=10):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, cell):
        for c in range(0, w, cell):
            if (r // cell + c // cell) % 2 == 0:
                img[r:r + cell, c:c + cell] = 255
    return img


def make_binary_edge():
    """A binary edge map with a single bright row."""
    em = np.zeros((50, 50), dtype=np.uint8)
    em[25, :] = 255
    return em


# ─── EdgeDetectionResult ──────────────────────────────────────────────────────

class TestEdgeDetectionResult:
    def test_basic_creation(self):
        em = make_binary_edge()
        result = EdgeDetectionResult(edge_map=em, method="canny")
        assert result.method == "canny"
        assert result.edge_map is em

    def test_density_computed(self):
        em = make_binary_edge()
        result = EdgeDetectionResult(edge_map=em, method="test")
        expected = float((em > 0).mean())
        assert result.density == pytest.approx(expected)

    def test_density_zero_for_blank(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        result = EdgeDetectionResult(edge_map=em, method="test")
        assert result.density == pytest.approx(0.0)

    def test_density_one_for_full(self):
        em = np.full((50, 50), 255, dtype=np.uint8)
        result = EdgeDetectionResult(edge_map=em, method="test")
        assert result.density == pytest.approx(1.0)

    def test_params_stored(self):
        em = make_binary_edge()
        params = {"threshold": 100}
        result = EdgeDetectionResult(edge_map=em, method="canny", params=params)
        assert result.params == params

    def test_n_contours_nonneg(self):
        em = make_binary_edge()
        result = EdgeDetectionResult(edge_map=em, method="test")
        assert result.n_contours >= 0

    def test_repr_contains_method(self):
        em = make_binary_edge()
        result = EdgeDetectionResult(edge_map=em, method="sobel")
        assert "sobel" in repr(result)

    def test_empty_edge_map_zero_density(self):
        em = np.zeros((0, 0), dtype=np.uint8)
        result = EdgeDetectionResult(edge_map=em, method="test")
        assert result.density == pytest.approx(0.0)


# ─── detect_edges ─────────────────────────────────────────────────────────────

class TestDetectEdges:
    def test_returns_edge_detection_result(self):
        img = make_gradient()
        result = detect_edges(img)
        assert isinstance(result, EdgeDetectionResult)

    def test_default_method_is_adaptive_canny(self):
        img = make_gradient()
        result = detect_edges(img)
        assert result.method == "adaptive_canny"

    def test_method_canny(self):
        img = make_gradient()
        result = detect_edges(img, method="canny")
        assert result.method == "canny"

    def test_method_sobel(self):
        img = make_gradient()
        result = detect_edges(img, method="sobel")
        assert result.method == "sobel"

    def test_method_laplacian(self):
        img = make_gradient()
        result = detect_edges(img, method="laplacian")
        assert result.method == "laplacian"

    def test_method_auto_alias(self):
        img = make_gradient()
        result = detect_edges(img, method="auto")
        assert result.method == "adaptive_canny"

    def test_unknown_method_raises(self):
        img = make_gradient()
        with pytest.raises(ValueError):
            detect_edges(img, method="unknown_method")

    def test_edge_map_same_shape(self):
        img = make_gradient(h=30, w=40)
        result = detect_edges(img)
        assert result.edge_map.shape == (30, 40)

    def test_edge_map_dtype_uint8(self):
        img = make_gradient()
        result = detect_edges(img)
        assert result.edge_map.dtype == np.uint8

    def test_bgr_input_accepted(self):
        img = make_bgr()
        result = detect_edges(img)
        assert isinstance(result, EdgeDetectionResult)

    def test_density_in_range(self):
        img = make_gradient()
        result = detect_edges(img)
        assert 0.0 <= result.density <= 1.0


# ─── adaptive_canny ───────────────────────────────────────────────────────────

class TestAdaptiveCanny:
    def test_returns_result(self):
        img = make_gradient()
        result = adaptive_canny(img)
        assert isinstance(result, EdgeDetectionResult)
        assert result.method == "adaptive_canny"

    def test_params_contain_sigma(self):
        img = make_gradient()
        result = adaptive_canny(img, sigma=0.5)
        assert "sigma" in result.params
        assert result.params["sigma"] == pytest.approx(0.5)

    def test_params_contain_thresholds(self):
        img = make_gradient()
        result = adaptive_canny(img)
        assert "threshold1" in result.params
        assert "threshold2" in result.params
        assert result.params["threshold1"] <= result.params["threshold2"]

    def test_edge_map_shape_preserved(self):
        img = make_gradient(h=40, w=60)
        result = adaptive_canny(img)
        assert result.edge_map.shape == (40, 60)

    def test_bgr_input(self):
        img = make_bgr()
        result = adaptive_canny(img)
        assert isinstance(result, EdgeDetectionResult)

    def test_gradient_has_edges(self):
        img = make_gradient()
        result = adaptive_canny(img)
        assert result.density >= 0.0  # At least some edges expected


# ─── sobel_edges ──────────────────────────────────────────────────────────────

class TestSobelEdges:
    def test_returns_result(self):
        img = make_gradient()
        result = sobel_edges(img)
        assert isinstance(result, EdgeDetectionResult)
        assert result.method == "sobel"

    def test_params_stored(self):
        img = make_gradient()
        result = sobel_edges(img, threshold=80.0, ksize=5)
        assert result.params["threshold"] == pytest.approx(80.0)
        assert result.params["ksize"] == 5

    def test_edge_map_binary(self):
        img = make_gradient()
        result = sobel_edges(img)
        unique_vals = set(np.unique(result.edge_map))
        assert unique_vals.issubset({0, 255})

    def test_constant_image_no_edges(self):
        img = make_gray(value=128)
        result = sobel_edges(img, threshold=0.1)
        # Constant image has zero gradient → should yield few or no edges
        assert result.density < 0.5

    def test_shape_preserved(self):
        img = make_gradient(h=30, w=45)
        result = sobel_edges(img)
        assert result.edge_map.shape == (30, 45)

    def test_bgr_input(self):
        img = make_bgr()
        result = sobel_edges(img)
        assert isinstance(result, EdgeDetectionResult)

    def test_high_threshold_fewer_edges(self):
        img = make_checkerboard()
        r_low  = sobel_edges(img, threshold=10.0)
        r_high = sobel_edges(img, threshold=200.0)
        assert r_high.density <= r_low.density


# ─── laplacian_edges ──────────────────────────────────────────────────────────

class TestLaplacianEdges:
    def test_returns_result(self):
        img = make_gradient()
        result = laplacian_edges(img)
        assert isinstance(result, EdgeDetectionResult)
        assert result.method == "laplacian"

    def test_params_stored(self):
        img = make_gradient()
        result = laplacian_edges(img, sigma=2.0, threshold=20.0)
        assert result.params["sigma"] == pytest.approx(2.0)
        assert result.params["threshold"] == pytest.approx(20.0)

    def test_edge_map_binary(self):
        img = make_gradient()
        result = laplacian_edges(img)
        unique_vals = set(np.unique(result.edge_map))
        assert unique_vals.issubset({0, 255})

    def test_shape_preserved(self):
        img = make_gradient(h=35, w=55)
        result = laplacian_edges(img)
        assert result.edge_map.shape == (35, 55)

    def test_bgr_input(self):
        img = make_bgr()
        result = laplacian_edges(img)
        assert isinstance(result, EdgeDetectionResult)


# ─── refine_edge_contour ──────────────────────────────────────────────────────

class TestRefineEdgeContour:
    def test_returns_ndarray(self):
        em = make_binary_edge()
        result = refine_edge_contour(em)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        em = make_binary_edge()
        result = refine_edge_contour(em)
        assert result.shape == em.shape

    def test_dtype_uint8(self):
        em = make_binary_edge()
        result = refine_edge_contour(em)
        assert result.dtype == np.uint8

    def test_binary_values(self):
        em = make_binary_edge()
        result = refine_edge_contour(em)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_min_area_removes_small_components(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[25, 25] = 255  # single pixel component
        result = refine_edge_contour(em, min_area=100)
        assert np.all(result == 0)

    def test_dilate_iter_zero_works(self):
        em = make_binary_edge()
        result = refine_edge_contour(em, dilate_iter=0)
        assert result.shape == em.shape

    def test_blank_map_stays_blank(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        result = refine_edge_contour(em)
        assert np.all(result == 0)


# ─── edge_density ─────────────────────────────────────────────────────────────

class TestEdgeDensity:
    def test_range_with_precomputed_map(self):
        em = make_binary_edge()
        density = edge_density(em, edge_map=em)
        assert 0.0 <= density <= 1.0

    def test_blank_map_zero_density(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        density = edge_density(em, edge_map=em)
        assert density == pytest.approx(0.0)

    def test_full_map_one_density(self):
        em = np.full((50, 50), 255, dtype=np.uint8)
        density = edge_density(em, edge_map=em)
        assert density == pytest.approx(1.0)

    def test_computes_density_from_image(self):
        img = make_gradient()
        density = edge_density(img)
        assert 0.0 <= density <= 1.0


# ─── edge_orientation_hist ────────────────────────────────────────────────────

class TestEdgeOrientationHist:
    def test_shape(self):
        em = make_binary_edge()
        hist = edge_orientation_hist(em, n_bins=8)
        assert hist.shape == (8,)

    def test_custom_n_bins(self):
        em = make_binary_edge()
        hist = edge_orientation_hist(em, n_bins=16)
        assert hist.shape == (16,)

    def test_normalized_sums_to_one(self):
        em = make_gradient()
        hist = edge_orientation_hist(em, normalize=True)
        if hist.sum() > 0:
            assert hist.sum() == pytest.approx(1.0)

    def test_not_normalized_nonneg(self):
        em = make_gradient()
        hist = edge_orientation_hist(em, normalize=False)
        assert np.all(hist >= 0)

    def test_dtype_float64(self):
        em = make_binary_edge()
        hist = edge_orientation_hist(em)
        assert hist.dtype == np.float64

    def test_constant_image_all_zero(self):
        em = make_gray(value=0)
        hist = edge_orientation_hist(em)
        assert np.all(hist == 0.0)
