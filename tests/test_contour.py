"""Тесты для puzzle_reconstruction.preprocessing.contour."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.contour import (
    extract_contour,
    rdp_simplify,
    split_contour_to_edges,
    resample_curve,
    normalize_contour,
)
from puzzle_reconstruction.models import EdgeSide


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square_mask(h: int = 64, w: int = 64, margin: int = 10) -> np.ndarray:
    """Квадратная маска: белый прямоугольник на чёрном фоне."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


def _circle_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Круговая маска."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx, r = h // 2, w // 2, min(h, w) // 2 - 4
    yy, xx = np.mgrid[0:h, 0:w]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 255
    return mask


def _rect_contour(h: int = 40, w: int = 60) -> np.ndarray:
    """Прямоугольный контур из 4 углов + промежуточных точек."""
    pts = []
    for x in range(0, w):
        pts.append([x, 0])
    for y in range(0, h):
        pts.append([w - 1, y])
    for x in range(w - 1, -1, -1):
        pts.append([x, h - 1])
    for y in range(h - 1, -1, -1):
        pts.append([0, y])
    return np.array(pts, dtype=np.float32)


def _line_contour(n: int = 50) -> np.ndarray:
    """Прямолинейный контур."""
    pts = np.column_stack([np.linspace(0, 100, n),
                           np.linspace(0, 0, n)]).astype(np.float32)
    return pts


# ─── TestExtractContour ───────────────────────────────────────────────────────

class TestExtractContour:
    def test_returns_ndarray(self):
        c = extract_contour(_square_mask())
        assert isinstance(c, np.ndarray)

    def test_shape_n_times_2(self):
        c = extract_contour(_square_mask())
        assert c.ndim == 2
        assert c.shape[1] == 2

    def test_dtype_float32(self):
        c = extract_contour(_square_mask())
        assert c.dtype == np.float32

    def test_multiple_points(self):
        c = extract_contour(_square_mask())
        assert c.shape[0] > 4

    def test_circle_mask_ok(self):
        c = extract_contour(_circle_mask())
        assert c.shape[0] > 10

    def test_empty_mask_raises(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_contour(empty)

    def test_contour_stays_within_mask_bounds(self):
        h, w = 64, 64
        mask = _square_mask(h, w, margin=10)
        c = extract_contour(mask)
        assert c[:, 0].min() >= 0
        assert c[:, 0].max() < w
        assert c[:, 1].min() >= 0
        assert c[:, 1].max() < h

    def test_full_mask_returns_border(self):
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        c = extract_contour(mask)
        assert c.shape[0] > 0


# ─── TestRdpSimplify ──────────────────────────────────────────────────────────

class TestRdpSimplify:
    def test_returns_ndarray(self):
        c = _rect_contour()
        s = rdp_simplify(c)
        assert isinstance(s, np.ndarray)

    def test_shape_n_times_2(self):
        s = rdp_simplify(_rect_contour())
        assert s.ndim == 2
        assert s.shape[1] == 2

    def test_simplified_le_original(self):
        c = _rect_contour()
        s = rdp_simplify(c)
        assert s.shape[0] <= c.shape[0]

    def test_high_epsilon_fewer_points(self):
        c = _rect_contour()
        s_low = rdp_simplify(c, epsilon_ratio=0.001)
        s_high = rdp_simplify(c, epsilon_ratio=0.05)
        assert s_high.shape[0] <= s_low.shape[0]

    def test_zero_epsilon_keeps_all(self):
        c = _rect_contour()
        s = rdp_simplify(c, epsilon_ratio=0.0)
        assert s.shape[0] == c.shape[0]

    def test_at_least_two_points(self):
        c = _rect_contour()
        s = rdp_simplify(c, epsilon_ratio=0.99)
        assert s.shape[0] >= 2

    def test_dtype_float32(self):
        s = rdp_simplify(_rect_contour())
        assert s.dtype == np.float32

    def test_from_extracted_mask(self):
        c = extract_contour(_square_mask())
        s = rdp_simplify(c)
        assert s.shape[0] <= c.shape[0]


# ─── TestSplitContourToEdges ──────────────────────────────────────────────────

class TestSplitContourToEdges:
    def test_returns_list(self):
        c = _rect_contour()
        edges = split_contour_to_edges(c, n_sides=4)
        assert isinstance(edges, list)

    def test_length_n_sides(self):
        c = _rect_contour()
        edges = split_contour_to_edges(c, n_sides=4)
        assert len(edges) == 4

    def test_each_element_tuple(self):
        c = _rect_contour()
        for edge in split_contour_to_edges(c, n_sides=4):
            assert isinstance(edge, tuple)
            assert len(edge) == 2

    def test_points_ndarray(self):
        c = _rect_contour()
        for pts, side in split_contour_to_edges(c, n_sides=4):
            assert isinstance(pts, np.ndarray)
            assert pts.ndim == 2
            assert pts.shape[1] == 2

    def test_side_is_edge_side(self):
        c = _rect_contour()
        for pts, side in split_contour_to_edges(c, n_sides=4):
            assert isinstance(side, EdgeSide)

    def test_edge_side_values(self):
        c = _rect_contour()
        sides = {side for _, side in split_contour_to_edges(c, n_sides=4)}
        valid = {EdgeSide.TOP, EdgeSide.BOTTOM, EdgeSide.LEFT, EdgeSide.RIGHT, EdgeSide.UNKNOWN}
        assert sides.issubset(valid)

    def test_three_sides(self):
        c = _rect_contour()
        edges = split_contour_to_edges(c, n_sides=3)
        assert len(edges) == 3

    def test_points_non_empty(self):
        c = _rect_contour()
        for pts, _ in split_contour_to_edges(c, n_sides=4):
            assert len(pts) > 0

    def test_from_real_mask(self):
        mask = _square_mask()
        c = extract_contour(mask)
        edges = split_contour_to_edges(c, n_sides=4)
        assert len(edges) == 4


# ─── TestResampleCurve ────────────────────────────────────────────────────────

class TestResampleCurve:
    def test_returns_ndarray(self):
        c = _rect_contour()
        r = resample_curve(c, n_points=64)
        assert isinstance(r, np.ndarray)

    def test_shape_n_points_times_2(self):
        c = _rect_contour()
        for n in (32, 64, 128, 256):
            r = resample_curve(c, n_points=n)
            assert r.shape == (n, 2)

    def test_zero_length_curve_tiles_first(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        r = resample_curve(pts, n_points=10)
        assert r.shape == (10, 2)
        assert np.allclose(r[:, 0], 5.0)

    def test_start_close_to_first_point(self):
        c = _line_contour(50)
        r = resample_curve(c, n_points=10)
        assert r[0, 0] == pytest.approx(0.0, abs=1.0)

    def test_end_close_to_last_point(self):
        c = _line_contour(50)
        r = resample_curve(c, n_points=10)
        assert r[-1, 0] == pytest.approx(100.0, abs=1.0)

    def test_single_n_points(self):
        c = _rect_contour()
        r = resample_curve(c, n_points=1)
        assert r.shape == (1, 2)

    def test_values_within_curve_bbox(self):
        c = _rect_contour(40, 60)
        r = resample_curve(c, n_points=64)
        assert r[:, 0].min() >= -1.0
        assert r[:, 0].max() <= 61.0


# ─── TestNormalizeContour ─────────────────────────────────────────────────────

class TestNormalizeContour:
    def test_returns_tuple_three(self):
        c = _rect_contour()
        result = normalize_contour(c)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_normalized_ndarray(self):
        c = _rect_contour()
        norm, centroid, scale = normalize_contour(c)
        assert isinstance(norm, np.ndarray)

    def test_centroid_is_array(self):
        c = _rect_contour()
        _, centroid, _ = normalize_contour(c)
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (2,)

    def test_scale_positive(self):
        c = _rect_contour()
        _, _, scale = normalize_contour(c)
        assert scale > 0.0

    def test_normalized_mean_near_zero(self):
        c = _rect_contour()
        norm, _, _ = normalize_contour(c)
        assert norm.mean(axis=0) == pytest.approx([0.0, 0.0], abs=0.5)

    def test_scale_zero_edge_case(self):
        # Всего одна уникальная точка → scale = 1.0
        pts = np.array([[5.0, 5.0]] * 4, dtype=np.float32)
        norm, centroid, scale = normalize_contour(pts)
        assert scale == pytest.approx(1.0)

    def test_normalized_within_unit_range(self):
        c = _rect_contour()
        norm, _, _ = normalize_contour(c)
        # После центрирования по scale = diagonal → значения < 1
        assert np.abs(norm).max() <= 2.0

    def test_centroid_close_to_geometric_center(self):
        c = _rect_contour(40, 60)
        _, centroid, _ = normalize_contour(c)
        # Центр прямоугольника ~(30, 20)
        assert centroid[0] == pytest.approx(30.0, abs=5.0)
        assert centroid[1] == pytest.approx(20.0, abs=5.0)

    def test_reconstruction(self):
        c = _rect_contour()
        norm, centroid, scale = normalize_contour(c)
        restored = norm * scale + centroid
        assert np.allclose(restored, c, atol=1e-3)
