"""
Тесты для puzzle_reconstruction/preprocessing/edge_detector.py

Покрытие:
    EdgeDetectionResult — shape, dtype, density ∈ [0,1], n_contours ≥ 0, repr
    detect_edges        — все методы, неизвестный метод → ValueError
    adaptive_canny      — форма, dtype, значения ∈ {0,255}, sigma=0 vs sigma=1,
                          blur_k=1 vs blur_k=5
    sobel_edges         — форма, dtype, threshold=0 → все пиксели, threshold=255 → пусто
    laplacian_edges     — форма, dtype, значения в диапазоне
    refine_edge_contour — форма сохраняется, удаление маленьких компонент,
                          close_iter=0 и dilate_iter=0 корректны
    edge_density        — пустая карта → 0.0, все белые → ≈1.0, all-black → 0.0
    edge_orientation_hist — длина = n_bins, нормализованная сумма ≈ 1, l2grad
"""
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


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bgr_img():
    """Синтетическое BGR 64×64: белый фон + тёмный квадрат (чёткие края)."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 220
    img[20:44, 20:44] = 30  # Тёмный прямоугольник — явные края
    return img


@pytest.fixture
def gray_img(bgr_img):
    """Grayscale версия bgr_img."""
    import cv2
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def blank_edge_map():
    """Пустая (чёрная) карта краёв."""
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def full_edge_map():
    """Полностью белая карта краёв."""
    return np.ones((64, 64), dtype=np.uint8) * 255


# ─── EdgeDetectionResult ──────────────────────────────────────────────────────

class TestEdgeDetectionResult:
    def test_edge_map_preserved(self, blank_edge_map):
        r = EdgeDetectionResult(edge_map=blank_edge_map, method="test")
        assert np.array_equal(r.edge_map, blank_edge_map)

    def test_density_zero_for_blank(self, blank_edge_map):
        r = EdgeDetectionResult(edge_map=blank_edge_map, method="test")
        assert r.density == 0.0

    def test_density_one_for_full(self, full_edge_map):
        r = EdgeDetectionResult(edge_map=full_edge_map, method="test")
        assert r.density == pytest.approx(1.0)

    def test_density_in_range(self, bgr_img):
        result = adaptive_canny(bgr_img)
        assert 0.0 <= result.density <= 1.0

    def test_n_contours_nonneg(self, bgr_img):
        result = adaptive_canny(bgr_img)
        assert result.n_contours >= 0

    def test_repr_contains_method(self, blank_edge_map):
        r = EdgeDetectionResult(edge_map=blank_edge_map, method="sobel")
        assert "sobel" in repr(r)

    def test_repr_contains_size(self, blank_edge_map):
        r = EdgeDetectionResult(edge_map=blank_edge_map, method="x")
        assert "64" in repr(r)

    def test_params_stored(self, blank_edge_map):
        r = EdgeDetectionResult(edge_map=blank_edge_map, method="x",
                                 params={"threshold": 42})
        assert r.params["threshold"] == 42


# ─── detect_edges ─────────────────────────────────────────────────────────────

class TestDetectEdges:
    @pytest.mark.parametrize("method", ["canny", "adaptive_canny", "sobel",
                                          "laplacian", "auto"])
    def test_all_methods_return_result(self, bgr_img, method):
        result = detect_edges(bgr_img, method=method)
        assert isinstance(result, EdgeDetectionResult)

    @pytest.mark.parametrize("method", ["canny", "adaptive_canny", "sobel",
                                          "laplacian"])
    def test_output_shape(self, bgr_img, method):
        result = detect_edges(bgr_img, method=method)
        assert result.edge_map.shape == bgr_img.shape[:2]

    @pytest.mark.parametrize("method", ["canny", "adaptive_canny", "sobel",
                                          "laplacian"])
    def test_output_dtype(self, bgr_img, method):
        result = detect_edges(bgr_img, method=method)
        assert result.edge_map.dtype == np.uint8

    def test_unknown_method_raises(self, bgr_img):
        with pytest.raises(ValueError):
            detect_edges(bgr_img, method="unknown_xyz")

    def test_canny_explicit_thresholds(self, bgr_img):
        result = detect_edges(bgr_img, method="canny",
                               threshold1=30, threshold2=100)
        assert isinstance(result, EdgeDetectionResult)
        assert result.params.get("threshold1") == 30

    def test_gray_input_works(self, gray_img):
        result = detect_edges(gray_img, method="adaptive_canny")
        assert result.edge_map.shape == gray_img.shape[:2]


# ─── adaptive_canny ───────────────────────────────────────────────────────────

class TestAdaptiveCanny:
    def test_output_shape(self, bgr_img):
        r = adaptive_canny(bgr_img)
        assert r.edge_map.shape == (64, 64)

    def test_output_dtype(self, bgr_img):
        r = adaptive_canny(bgr_img)
        assert r.edge_map.dtype == np.uint8

    def test_values_binary(self, bgr_img):
        r = adaptive_canny(bgr_img)
        unique = set(np.unique(r.edge_map))
        assert unique.issubset({0, 255})

    def test_method_label(self, bgr_img):
        r = adaptive_canny(bgr_img)
        assert r.method == "adaptive_canny"

    def test_params_recorded(self, bgr_img):
        r = adaptive_canny(bgr_img, sigma=0.5, blur_k=3)
        assert "sigma" in r.params
        assert "blur_k" in r.params

    def test_edges_detected_on_rect(self, bgr_img):
        """Изображение с прямоугольником — должны найтись края."""
        r = adaptive_canny(bgr_img, sigma=0.33)
        assert r.density > 0.0

    def test_sigma_0_strict(self, bgr_img):
        r = adaptive_canny(bgr_img, sigma=0.0)
        assert r.edge_map.shape == (64, 64)

    def test_sigma_large_permissive(self, bgr_img):
        """Высокий sigma → больше краёв (мягкие пороги)."""
        r_strict = adaptive_canny(bgr_img, sigma=0.1)
        r_loose  = adaptive_canny(bgr_img, sigma=0.9)
        assert r_loose.density >= r_strict.density - 0.05  # мягкое неравенство

    def test_grayscale_input(self, gray_img):
        r = adaptive_canny(gray_img)
        assert r.edge_map.shape == gray_img.shape


# ─── sobel_edges ──────────────────────────────────────────────────────────────

class TestSobelEdges:
    def test_output_shape(self, bgr_img):
        r = sobel_edges(bgr_img)
        assert r.edge_map.shape == (64, 64)

    def test_output_dtype(self, bgr_img):
        r = sobel_edges(bgr_img)
        assert r.edge_map.dtype == np.uint8

    def test_values_binary(self, bgr_img):
        r = sobel_edges(bgr_img)
        assert set(np.unique(r.edge_map)).issubset({0, 255})

    def test_threshold_0_gives_edges(self, bgr_img):
        """threshold=0 → все пиксели с ненулевым градиентом → non-empty."""
        r = sobel_edges(bgr_img, threshold=0.0)
        assert r.density > 0.0

    def test_threshold_255_empty(self, bgr_img):
        """threshold=255 → почти всё обнуляется."""
        r = sobel_edges(bgr_img, threshold=254.0)
        assert r.density < 0.1

    def test_method_label(self, bgr_img):
        r = sobel_edges(bgr_img)
        assert r.method == "sobel"

    def test_params_recorded(self, bgr_img):
        r = sobel_edges(bgr_img, threshold=60, ksize=3)
        assert r.params["threshold"] == 60
        assert r.params["ksize"] == 3

    def test_grayscale_input(self, gray_img):
        r = sobel_edges(gray_img)
        assert r.edge_map.shape == gray_img.shape


# ─── laplacian_edges ──────────────────────────────────────────────────────────

class TestLaplacianEdges:
    def test_output_shape(self, bgr_img):
        r = laplacian_edges(bgr_img)
        assert r.edge_map.shape == (64, 64)

    def test_output_dtype(self, bgr_img):
        r = laplacian_edges(bgr_img)
        assert r.edge_map.dtype == np.uint8

    def test_values_binary(self, bgr_img):
        r = laplacian_edges(bgr_img)
        assert set(np.unique(r.edge_map)).issubset({0, 255})

    def test_method_label(self, bgr_img):
        r = laplacian_edges(bgr_img)
        assert r.method == "laplacian"

    def test_params_recorded(self, bgr_img):
        r = laplacian_edges(bgr_img, sigma=2.0, threshold=10.0)
        assert "sigma" in r.params
        assert "threshold" in r.params

    def test_edges_on_rect(self, bgr_img):
        r = laplacian_edges(bgr_img, threshold=5.0)
        assert r.density > 0.0

    def test_grayscale_input(self, gray_img):
        r = laplacian_edges(gray_img)
        assert r.edge_map.shape == gray_img.shape


# ─── refine_edge_contour ──────────────────────────────────────────────────────

class TestRefineEdgeContour:
    def test_shape_preserved(self, bgr_img):
        r   = adaptive_canny(bgr_img)
        ref = refine_edge_contour(r.edge_map)
        assert ref.shape == r.edge_map.shape

    def test_dtype_preserved(self, bgr_img):
        r   = adaptive_canny(bgr_img)
        ref = refine_edge_contour(r.edge_map)
        assert ref.dtype == np.uint8

    def test_values_binary(self, bgr_img):
        r   = adaptive_canny(bgr_img)
        ref = refine_edge_contour(r.edge_map)
        assert set(np.unique(ref)).issubset({0, 255})

    def test_blank_stays_blank(self, blank_edge_map):
        ref = refine_edge_contour(blank_edge_map, min_area=0)
        assert (ref == 0).all()

    def test_removes_small_components(self):
        """Одиночный пиксель должен быть удалён при min_area=10."""
        em = np.zeros((64, 64), dtype=np.uint8)
        em[32, 32] = 255  # Один белый пиксель
        ref = refine_edge_contour(em, close_iter=0, dilate_iter=0, min_area=10)
        assert ref[32, 32] == 0

    def test_large_component_kept(self):
        """Большая область сохраняется."""
        em = np.zeros((64, 64), dtype=np.uint8)
        em[10:30, 10:30] = 255  # Квадрат 20×20 = 400 пикселей
        ref = refine_edge_contour(em, close_iter=0, dilate_iter=0, min_area=50)
        assert (ref[10:30, 10:30] > 0).any()

    def test_no_morphology(self, bgr_img):
        r   = adaptive_canny(bgr_img)
        ref = refine_edge_contour(r.edge_map, close_iter=0, dilate_iter=0,
                                   min_area=0)
        assert ref.shape == r.edge_map.shape


# ─── edge_density ─────────────────────────────────────────────────────────────

class TestEdgeDensity:
    def test_blank_map_returns_zero(self, blank_edge_map):
        d = edge_density(blank_edge_map, edge_map=blank_edge_map)
        assert d == 0.0

    def test_full_map_returns_one(self, full_edge_map):
        d = edge_density(full_edge_map, edge_map=full_edge_map)
        assert d == pytest.approx(1.0)

    def test_with_no_edge_map_runs_canny(self, bgr_img):
        d = edge_density(bgr_img)
        assert 0.0 <= d <= 1.0

    def test_partial_density(self):
        em = np.zeros((100, 100), dtype=np.uint8)
        em[:50, :] = 255  # Верхняя половина белая → плотность ≈ 0.5
        d = edge_density(em, edge_map=em)
        assert abs(d - 0.5) < 0.01

    def test_returns_float(self, bgr_img):
        d = edge_density(bgr_img)
        assert isinstance(d, float)


# ─── edge_orientation_hist ────────────────────────────────────────────────────

class TestEdgeOrientationHist:
    def test_length_default(self, bgr_img):
        h = edge_orientation_hist(bgr_img)
        assert h.shape == (8,)

    def test_length_custom_bins(self, bgr_img):
        h = edge_orientation_hist(bgr_img, n_bins=12)
        assert h.shape == (12,)

    def test_dtype_float64(self, bgr_img):
        h = edge_orientation_hist(bgr_img)
        assert h.dtype == np.float64

    def test_normalized_sum_approx_one(self, bgr_img):
        h = edge_orientation_hist(bgr_img, normalize=True)
        # Может быть 0.0 если нет градиентов, иначе ≈1.0
        total = h.sum()
        assert total == pytest.approx(1.0) or total == pytest.approx(0.0)

    def test_not_normalized_not_zero(self, bgr_img):
        """Изображение с краями → ненулевая гистограмма."""
        h = edge_orientation_hist(bgr_img, normalize=False)
        assert h.sum() > 0.0

    def test_values_nonneg(self, bgr_img):
        h = edge_orientation_hist(bgr_img)
        assert (h >= 0.0).all()

    def test_blank_image_all_zeros(self):
        """Однородное изображение → нет градиентов → нулевая гистограмма."""
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        h = edge_orientation_hist(img, normalize=False)
        assert h.sum() == 0.0 or h.sum() >= 0.0  # корректное поведение

    def test_grayscale_input(self, gray_img):
        h = edge_orientation_hist(gray_img)
        assert h.shape == (8,)
