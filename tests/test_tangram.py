"""
Тесты для модуля танграм-аппроксимации.
"""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.tangram.hull import (
    convex_hull, rdp_simplify, normalize_polygon
)
from puzzle_reconstruction.algorithms.tangram.classifier import (
    classify_shape, compute_interior_angles
)
from puzzle_reconstruction.algorithms.tangram.inscriber import (
    fit_tangram, extract_tangram_edge
)
from puzzle_reconstruction.models import ShapeClass


# ─── Тестовые фигуры ──────────────────────────────────────────────────────

def make_rectangle(w=200, h=100):
    """Прямоугольник по часовой стрелке."""
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def make_triangle():
    return np.array([[0, 0], [100, 0], [50, 80]], dtype=np.float32)


def make_parallelogram():
    return np.array([[20, 0], [120, 0], [100, 60], [0, 60]], dtype=np.float32)


def make_circle_contour(n=64, r=100):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t) + 100, r * np.sin(t) + 100]).astype(np.float32)


# ─── Convex Hull ──────────────────────────────────────────────────────────

class TestConvexHull:

    def test_rectangle_hull_is_4_points(self):
        rect = make_rectangle()
        hull = convex_hull(rect)
        assert len(hull) == 4

    def test_hull_contains_all_vertices(self):
        """Все вершины прямоугольника должны быть на выпуклой оболочке."""
        rect = make_rectangle()
        hull = convex_hull(rect)
        for vertex in rect:
            dists = np.linalg.norm(hull - vertex, axis=1)
            assert dists.min() < 1.0, f"Вершина {vertex} не на hull"

    def test_circle_hull_all_points(self):
        """Для окружности hull ≈ все точки."""
        circle = make_circle_contour(n=32)
        hull = convex_hull(circle)
        assert len(hull) >= 20  # Почти все точки должны быть на hull


class TestRDPSimplify:

    def test_rectangle_stays_4(self):
        rect = make_rectangle()
        simplified = rdp_simplify(rect, epsilon_ratio=0.01)
        assert 3 <= len(simplified) <= 6

    def test_circle_simplified_fewer_points(self):
        circle = make_circle_contour(n=256)
        simplified = rdp_simplify(circle, epsilon_ratio=0.02)
        assert len(simplified) < len(circle)
        assert len(simplified) >= 4

    def test_epsilon_zero_keeps_all(self):
        rect = make_rectangle()
        simplified = rdp_simplify(rect, epsilon_ratio=0.0)
        assert len(simplified) == len(rect)


class TestNormalizePolygon:

    def test_centroid_at_origin(self):
        rect = make_rectangle()
        normalized, centroid, scale, angle = normalize_polygon(rect)
        assert np.linalg.norm(normalized.mean(axis=0)) < 1e-6

    def test_scale_diagonal_one(self):
        rect = make_rectangle(200, 100)
        normalized, centroid, scale, angle = normalize_polygon(rect)
        bbox = normalized.max(axis=0) - normalized.min(axis=0)
        diagonal = float(np.hypot(bbox[0], bbox[1]))
        assert abs(diagonal - 1.0) < 0.01

    def test_centroid_matches(self):
        rect = make_rectangle()
        expected_centroid = rect.mean(axis=0)
        _, centroid, _, _ = normalize_polygon(rect)
        assert np.linalg.norm(centroid - expected_centroid) < 1e-4


# ─── Classifier ───────────────────────────────────────────────────────────

class TestClassifier:

    def test_triangle_classified(self):
        tri = normalize_polygon(make_triangle())[0]
        assert classify_shape(tri) == ShapeClass.TRIANGLE

    def test_rectangle_classified(self):
        rect = normalize_polygon(make_rectangle())[0]
        result = classify_shape(rect)
        assert result == ShapeClass.RECTANGLE

    def test_parallelogram_classified(self):
        para = normalize_polygon(make_parallelogram())[0]
        result = classify_shape(para)
        assert result in (ShapeClass.PARALLELOGRAM, ShapeClass.TRAPEZOID,
                           ShapeClass.POLYGON)

    def test_pentagon(self):
        t = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        penta = np.column_stack([np.cos(t), np.sin(t)])
        assert classify_shape(penta) == ShapeClass.PENTAGON

    def test_hexagon(self):
        t = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hex_ = np.column_stack([np.cos(t), np.sin(t)])
        assert classify_shape(hex_) == ShapeClass.HEXAGON

    def test_interior_angles_triangle_sum(self):
        """Сумма углов треугольника ≈ π."""
        tri = make_triangle()
        angles = compute_interior_angles(tri)
        assert abs(angles.sum() - np.pi) < 0.05

    def test_interior_angles_rectangle(self):
        """Все углы прямоугольника ≈ π/2."""
        rect = make_rectangle()
        angles = compute_interior_angles(rect)
        assert np.all(np.abs(angles - np.pi / 2) < 0.1)


# ─── Inscriber ────────────────────────────────────────────────────────────

class TestInscriber:

    def test_fit_tangram_returns_signature(self):
        circle = make_circle_contour(n=64)
        sig = fit_tangram(circle)
        assert sig is not None
        assert sig.polygon is not None
        assert sig.centroid is not None
        assert sig.scale > 0

    def test_tangram_polygon_normalized(self):
        """Полигон должен быть центрирован и нормализован."""
        circle = make_circle_contour(n=64)
        sig = fit_tangram(circle)
        norm = np.linalg.norm(sig.polygon.mean(axis=0))
        assert norm < 0.05, f"Центроид не в нуле: {norm}"

    def test_tangram_shape_classified(self):
        """Форма должна быть определена."""
        rect = make_rectangle()
        sig = fit_tangram(rect)
        assert isinstance(sig.shape_class, ShapeClass)

    def test_extract_edge_shape(self):
        """Кривая края должна иметь правильную форму."""
        rect = make_rectangle()
        sig = fit_tangram(rect)
        n_points = 64
        edge = extract_tangram_edge(sig, edge_index=0, n_points=n_points)
        assert edge.shape == (n_points, 2)

    def test_extract_all_edges(self):
        """Все 4 края должны быть извлечены."""
        rect = make_rectangle()
        sig = fit_tangram(rect)
        edges = [extract_tangram_edge(sig, i) for i in range(4)]
        assert all(e.shape[1] == 2 for e in edges)
        # Края должны различаться
        assert not np.allclose(edges[0], edges[1])

    def test_area_positive(self):
        rect = make_rectangle()
        sig = fit_tangram(rect)
        assert sig.area > 0
