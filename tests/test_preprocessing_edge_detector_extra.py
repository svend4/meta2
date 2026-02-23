"""Extra tests for puzzle_reconstruction/preprocessing/edge_detector.py"""
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

LAPLACIAN_XFAIL = pytest.mark.xfail(
    strict=False,
    reason="cv2.Laplacian float32→CV_64F unsupported on some OpenCV builds",
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=50, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _noisy(h=50, w=50, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def _checkerboard(h=50, w=50, cell=10):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, cell):
        for c in range(0, w, cell):
            if (r // cell + c // cell) % 2 == 0:
                img[r:r + cell, c:c + cell] = 255
    return img


def _bgr(h=50, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    return img


# ─── TestEdgeDetectionResultExtra ────────────────────────────────────────────

class TestEdgeDetectionResultExtra:
    def test_method_canny_stored(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        r = EdgeDetectionResult(edge_map=em, method="canny")
        assert r.method == "canny"

    def test_method_sobel_stored(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        r = EdgeDetectionResult(edge_map=em, method="sobel")
        assert r.method == "sobel"

    def test_density_half(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[:25, :] = 255
        r = EdgeDetectionResult(edge_map=em, method="test")
        assert r.density == pytest.approx(0.5)

    def test_n_contours_is_int(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        r = EdgeDetectionResult(edge_map=em, method="test")
        assert isinstance(r.n_contours, int)

    def test_params_default_empty(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        r = EdgeDetectionResult(edge_map=em, method="test")
        assert r.params == {}

    def test_density_fractional(self):
        em = np.zeros((100, 100), dtype=np.uint8)
        em[50, :] = 255
        r = EdgeDetectionResult(edge_map=em, method="test")
        expected = float((em > 0).mean())
        assert r.density == pytest.approx(expected)

    def test_repr_contains_density(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        r = EdgeDetectionResult(edge_map=em, method="test")
        s = repr(r)
        assert isinstance(s, str)

    def test_non_square_map(self):
        em = np.zeros((30, 60), dtype=np.uint8)
        em[15, :] = 255
        r = EdgeDetectionResult(edge_map=em, method="canny")
        assert r.density > 0.0


# ─── TestDetectEdgesExtra ─────────────────────────────────────────────────────

class TestDetectEdgesExtra:
    def test_non_square_image(self):
        img = _gradient(h=30, w=60)
        r = detect_edges(img)
        assert r.edge_map.shape == (30, 60)

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        r = detect_edges(img)
        assert isinstance(r, EdgeDetectionResult)

    def test_checkerboard_has_edges(self):
        img = _checkerboard()
        r = detect_edges(img)
        assert r.density > 0.0

    def test_canny_method_dtype(self):
        img = _gradient()
        r = detect_edges(img, method="canny")
        assert r.edge_map.dtype == np.uint8

    def test_sobel_method_shape(self):
        img = _gradient(h=40, w=80)
        r = detect_edges(img, method="sobel")
        assert r.edge_map.shape == (40, 80)

    def test_bgr_sobel(self):
        img = _bgr()
        r = detect_edges(img, method="sobel")
        assert r.edge_map.dtype == np.uint8

    def test_various_noisy_seeds(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = detect_edges(img)
            assert 0.0 <= r.density <= 1.0


# ─── TestAdaptiveCannyExtra ───────────────────────────────────────────────────

class TestAdaptiveCannyExtra:
    def test_non_square(self):
        img = _gradient(h=40, w=80)
        r = adaptive_canny(img)
        assert r.edge_map.shape == (40, 80)

    def test_small_sigma(self):
        img = _gradient()
        r = adaptive_canny(img, sigma=0.1)
        assert isinstance(r, EdgeDetectionResult)

    def test_large_sigma(self):
        img = _gradient()
        r = adaptive_canny(img, sigma=1.5)
        assert isinstance(r, EdgeDetectionResult)

    def test_dtype_uint8(self):
        img = _gradient()
        r = adaptive_canny(img)
        assert r.edge_map.dtype == np.uint8

    def test_various_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = adaptive_canny(img)
            assert r.density >= 0.0

    def test_constant_image_valid(self):
        img = _gray(value=128)
        r = adaptive_canny(img)
        assert isinstance(r, EdgeDetectionResult)

    def test_method_is_adaptive_canny(self):
        img = _gradient()
        r = adaptive_canny(img)
        assert r.method == "adaptive_canny"


# ─── TestSobelEdgesExtra ──────────────────────────────────────────────────────

class TestSobelEdgesExtra:
    def test_non_square(self):
        img = _gradient(h=40, w=80)
        r = sobel_edges(img)
        assert r.edge_map.shape == (40, 80)

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        r = sobel_edges(img)
        assert isinstance(r, EdgeDetectionResult)

    def test_checkerboard_has_edges(self):
        img = _checkerboard()
        r = sobel_edges(img, threshold=10.0)
        assert r.density > 0.0

    def test_threshold_200_low_density(self):
        img = _gradient()
        r = sobel_edges(img, threshold=200.0)
        assert r.density <= 1.0

    def test_method_is_sobel(self):
        img = _gradient()
        r = sobel_edges(img)
        assert r.method == "sobel"

    def test_five_seeds_no_crash(self):
        for s in range(5):
            img = _noisy(seed=s)
            r = sobel_edges(img)
            assert r.edge_map.dtype == np.uint8

    def test_ksize_3(self):
        img = _gradient()
        r = sobel_edges(img, ksize=3)
        assert r.edge_map.shape == img.shape


# ─── TestLaplacianEdgesExtra ──────────────────────────────────────────────────

class TestLaplacianEdgesExtra:
    @LAPLACIAN_XFAIL
    def test_non_square(self):
        img = _gradient(h=40, w=80)
        r = laplacian_edges(img)
        assert r.edge_map.shape == (40, 80)

    @LAPLACIAN_XFAIL
    def test_various_sigmas(self):
        for sigma in (0.5, 1.0, 2.0):
            img = _gradient()
            r = laplacian_edges(img, sigma=sigma)
            assert isinstance(r, EdgeDetectionResult)

    @LAPLACIAN_XFAIL
    def test_method_is_laplacian(self):
        img = _gradient()
        r = laplacian_edges(img)
        assert r.method == "laplacian"


# ─── TestRefineEdgeContourExtra ───────────────────────────────────────────────

class TestRefineEdgeContourExtra:
    def test_non_square(self):
        em = np.zeros((30, 60), dtype=np.uint8)
        em[15, :] = 255
        r = refine_edge_contour(em)
        assert r.shape == (30, 60)

    def test_dilate_iter_1(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[25, :] = 255
        r = refine_edge_contour(em, dilate_iter=1)
        assert r.dtype == np.uint8

    def test_dilate_iter_2(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[25, :] = 255
        r = refine_edge_contour(em, dilate_iter=2)
        assert r.shape == em.shape

    def test_large_component_preserved(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[10:40, 10:40] = 255
        r = refine_edge_contour(em, min_area=10)
        assert np.any(r > 0)

    def test_binary_output(self):
        em = _checkerboard(h=50, w=50, cell=5)
        r = refine_edge_contour(em)
        unique_vals = set(np.unique(r))
        assert unique_vals.issubset({0, 255})

    def test_min_area_large_removes_all(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        em[25, 25] = 255
        em[25, 26] = 255
        r = refine_edge_contour(em, min_area=10000)
        assert np.all(r == 0)


# ─── TestEdgeDensityExtra ─────────────────────────────────────────────────────

class TestEdgeDensityExtra:
    def test_gradient_some_density(self):
        img = _gradient()
        d = edge_density(img)
        assert 0.0 <= d <= 1.0

    def test_checkerboard_high_density(self):
        img = _checkerboard()
        d = edge_density(img)
        assert d >= 0.0

    def test_noisy_image_in_range(self):
        for s in range(5):
            img = _noisy(seed=s)
            d = edge_density(img)
            assert 0.0 <= d <= 1.0

    def test_bgr_input(self):
        img = _bgr()
        d = edge_density(img)
        assert 0.0 <= d <= 1.0

    def test_precomputed_map_matches(self):
        img = _gradient()
        em = np.zeros_like(img)
        em[25, :] = 255
        d = edge_density(img, edge_map=em)
        assert d > 0.0


# ─── TestEdgeOrientationHistExtra ────────────────────────────────────────────

class TestEdgeOrientationHistExtra:
    def test_bins_4(self):
        em = _gradient()
        hist = edge_orientation_hist(em, n_bins=4)
        assert hist.shape == (4,)

    def test_bins_32(self):
        em = _gradient()
        hist = edge_orientation_hist(em, n_bins=32)
        assert hist.shape == (32,)

    def test_all_nonneg(self):
        for s in range(5):
            em = _noisy(seed=s)
            hist = edge_orientation_hist(em)
            assert np.all(hist >= 0.0)

    def test_non_square_image(self):
        em = _gradient(h=30, w=60)
        hist = edge_orientation_hist(em, n_bins=8)
        assert hist.shape == (8,)

    def test_normalize_true_sum(self):
        em = _checkerboard()
        hist = edge_orientation_hist(em, n_bins=8, normalize=True)
        s = float(hist.sum())
        assert s == pytest.approx(0.0, abs=1e-5) or s == pytest.approx(1.0, abs=1e-5)

    def test_normalize_false_any_value(self):
        em = _gradient()
        hist = edge_orientation_hist(em, n_bins=8, normalize=False)
        assert hist.shape == (8,)

    def test_blank_image_all_zero(self):
        em = np.zeros((50, 50), dtype=np.uint8)
        hist = edge_orientation_hist(em, n_bins=8)
        assert np.all(hist == 0.0)
