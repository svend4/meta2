"""
Классификация геометрической формы упрощённого полигона.
Определяет тип фигуры: треугольник, прямоугольник, трапеция, и т.д.
"""
import numpy as np
from ...models import ShapeClass


def classify_shape(polygon: np.ndarray) -> ShapeClass:
    """
    Классифицирует форму полигона по числу вершин и углам.

    Args:
        polygon: (K, 2) нормализованные вершины.

    Returns:
        ShapeClass
    """
    n = len(polygon)

    if n <= 2:
        return ShapeClass.POLYGON

    if n == 3:
        return ShapeClass.TRIANGLE

    if n == 4:
        return _classify_quad(polygon)

    if n == 5:
        return ShapeClass.PENTAGON

    if n == 6:
        return ShapeClass.HEXAGON

    return ShapeClass.POLYGON


def compute_interior_angles(polygon: np.ndarray) -> np.ndarray:
    """
    Вычисляет внутренние углы (в радианах) для каждой вершины полигона.
    """
    n = len(polygon)
    angles = np.zeros(n)
    for i in range(n):
        p0 = polygon[(i - 1) % n]
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angles[i] = np.arccos(np.clip(cos_a, -1.0, 1.0))
    return angles


def _classify_quad(polygon: np.ndarray) -> ShapeClass:
    """Классифицирует четырёхугольник."""
    angles = compute_interior_angles(polygon)
    angle_deg = np.degrees(angles)

    # Прямоугольник: все углы ~90°
    if np.all(np.abs(angle_deg - 90) < 15):
        return ShapeClass.RECTANGLE

    # Параллелограмм: противоположные стороны параллельны
    sides = np.array([polygon[(i + 1) % 4] - polygon[i] for i in range(4)])
    dot01 = abs(np.dot(sides[0], sides[2]))  # стороны 0 и 2
    dot13 = abs(np.dot(sides[1], sides[3]))  # стороны 1 и 3
    cross01 = abs(sides[0][0] * sides[2][1] - sides[0][1] * sides[2][0])
    cross13 = abs(sides[1][0] * sides[3][1] - sides[1][1] * sides[3][0])
    if cross01 < 0.1 * dot01 and cross13 < 0.1 * dot13:
        return ShapeClass.PARALLELOGRAM

    # Трапеция: одна пара параллельных сторон
    if cross01 < 0.15 * (dot01 + 1e-5) or cross13 < 0.15 * (dot13 + 1e-5):
        return ShapeClass.TRAPEZOID

    return ShapeClass.POLYGON
