"""Утилиты для работы с полигонами фрагментов.

Модуль предоставляет функции для вычисления геометрических свойств
полигонов: площадь, периметр, центроид, выпуклая оболочка, проверка
принадлежности точки и пересечение полигонов.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# Тип точки: (x, y)
Point = Tuple[float, float]
# Тип полигона: список точек
Polygon = List[Point]


# ─── _to_array ────────────────────────────────────────────────────────────────

def _to_array(polygon: Polygon) -> np.ndarray:
    """Конвертировать полигон в numpy-массив формы (N, 2)."""
    return np.array(polygon, dtype=float)


# ─── polygon_area ─────────────────────────────────────────────────────────────

def polygon_area(polygon: Polygon) -> float:
    """Площадь полигона (формула Гаусса / shoelace).

    Аргументы:
        polygon: Список вершин [(x, y), ...] (N >= 3).

    Возвращает:
        Площадь (>= 0).

    Исключения:
        ValueError: если меньше 3 вершин.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"Полигон должен иметь >= 3 вершин, получено {len(polygon)}"
        )
    pts = _to_array(polygon)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# ─── polygon_perimeter ────────────────────────────────────────────────────────

def polygon_perimeter(polygon: Polygon) -> float:
    """Периметр полигона.

    Аргументы:
        polygon: Список вершин (N >= 2).

    Возвращает:
        Периметр (>= 0).

    Исключения:
        ValueError: если меньше 2 вершин.
    """
    if len(polygon) < 2:
        raise ValueError(
            f"Полигон должен иметь >= 2 вершин, получено {len(polygon)}"
        )
    pts = _to_array(polygon)
    shifted = np.roll(pts, -1, axis=0)
    return float(np.sum(np.linalg.norm(shifted - pts, axis=1)))


# ─── polygon_centroid ─────────────────────────────────────────────────────────

def polygon_centroid(polygon: Polygon) -> Point:
    """Центроид полигона.

    Для выпуклых и невыпуклых полигонов — по формуле через знаковую площадь.

    Аргументы:
        polygon: Список вершин (N >= 3).

    Возвращает:
        (cx, cy).

    Исключения:
        ValueError: если меньше 3 вершин или площадь = 0.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"Полигон должен иметь >= 3 вершин, получено {len(polygon)}"
        )
    pts = _to_array(polygon)
    n = len(pts)
    cx = cy = 0.0
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        signed_area += cross
        cx += (pts[i, 0] + pts[j, 0]) * cross
        cy += (pts[i, 1] + pts[j, 1]) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-12:
        # Вырожденный полигон — простое среднее
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    factor = 1.0 / (6.0 * signed_area)
    return float(cx * factor), float(cy * factor)


# ─── point_in_polygon ─────────────────────────────────────────────────────────

def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Проверить принадлежность точки полигону (алгоритм трассировки луча).

    Аргументы:
        point:   Точка (x, y).
        polygon: Вершины полигона (N >= 3).

    Возвращает:
        True если точка внутри полигона.

    Исключения:
        ValueError: если меньше 3 вершин.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"Полигон должен иметь >= 3 вершин, получено {len(polygon)}"
        )
    px, py = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ─── convex_hull ──────────────────────────────────────────────────────────────

def convex_hull(points: List[Point]) -> Polygon:
    """Выпуклая оболочка набора точек (алгоритм Грэхема).

    Аргументы:
        points: Список точек (N >= 1).

    Возвращает:
        Список вершин выпуклой оболочки в порядке обхода против часовой стрелки.

    Исключения:
        ValueError: если список пуст.
    """
    if not points:
        raise ValueError("Список точек не должен быть пустым")

    pts = _to_array(points)
    pts = np.unique(pts, axis=0)
    if len(pts) == 1:
        return [(float(pts[0, 0]), float(pts[0, 1]))]
    if len(pts) == 2:
        return [(float(pts[0, 0]), float(pts[0, 1])),
                (float(pts[1, 0]), float(pts[1, 1]))]

    # Якорная точка — нижняя левая
    anchor = pts[np.lexsort((pts[:, 0], pts[:, 1]))[0]]

    def _angle_and_dist(p: np.ndarray):
        d = p - anchor
        return np.arctan2(d[1], d[0]), np.linalg.norm(d)

    others = pts[~np.all(pts == anchor, axis=1)]
    if len(others) == 0:
        return [(float(anchor[0]), float(anchor[1]))]

    sorted_pts = sorted(others.tolist(), key=lambda p: _angle_and_dist(np.array(p)))

    hull = [anchor.tolist(), sorted_pts[0]]
    for p in sorted_pts[1:]:
        while len(hull) > 1:
            # Проверка поворота: cross product
            o = np.array(hull[-2])
            a = np.array(hull[-1])
            b = np.array(p)
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    return [(float(v[0]), float(v[1])) for v in hull]


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

def polygon_bounding_box(
    polygon: Polygon,
) -> Tuple[float, float, float, float]:
    """Ограничивающий прямоугольник полигона.

    Аргументы:
        polygon: Вершины полигона (N >= 1).

    Возвращает:
        (x_min, y_min, x_max, y_max).

    Исключения:
        ValueError: если список пуст.
    """
    if not polygon:
        raise ValueError("Полигон не должен быть пустым")
    pts = _to_array(polygon)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


# ─── polygon_aspect_ratio ─────────────────────────────────────────────────────

def polygon_aspect_ratio(polygon: Polygon) -> float:
    """Соотношение сторон bounding box полигона (ширина / высота).

    Аргументы:
        polygon: Вершины полигона (N >= 1).

    Возвращает:
        width / height; если height == 0, возвращает 0.

    Исключения:
        ValueError: если список пуст.
    """
    x_min, y_min, x_max, y_max = polygon_bounding_box(polygon)
    w = x_max - x_min
    h = y_max - y_min
    if h < 1e-12:
        return 0.0
    return float(w / h)


# ─── translate_polygon ────────────────────────────────────────────────────────

def translate_polygon(polygon: Polygon, dx: float, dy: float) -> Polygon:
    """Сместить полигон на (dx, dy).

    Аргументы:
        polygon: Вершины.
        dx:      Смещение по X.
        dy:      Смещение по Y.

    Возвращает:
        Новый полигон (исходный не изменяется).
    """
    return [(x + dx, y + dy) for x, y in polygon]


# ─── scale_polygon ────────────────────────────────────────────────────────────

def scale_polygon(
    polygon: Polygon,
    scale: float,
    center: Optional[Point] = None,
) -> Polygon:
    """Масштабировать полигон относительно центра.

    Аргументы:
        polygon: Вершины.
        scale:   Масштаб (> 0).
        center:  Центр масштабирования (None → центроид).

    Возвращает:
        Масштабированный полигон.

    Исключения:
        ValueError: если scale <= 0.
    """
    if scale <= 0.0:
        raise ValueError(f"scale должен быть > 0, получено {scale}")
    if center is None:
        if len(polygon) < 3:
            pts = _to_array(polygon)
            center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        else:
            center = polygon_centroid(polygon)
    cx, cy = center
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale)
            for x, y in polygon]


# ─── rotate_polygon ───────────────────────────────────────────────────────────

def rotate_polygon(
    polygon: Polygon,
    angle_deg: float,
    center: Optional[Point] = None,
) -> Polygon:
    """Повернуть полигон на угол (градусы) вокруг центра.

    Аргументы:
        polygon:   Вершины.
        angle_deg: Угол поворота в градусах (против часовой стрелки).
        center:    Центр поворота (None → центроид).

    Возвращает:
        Повёрнутый полигон.
    """
    if center is None:
        if len(polygon) < 3:
            pts = _to_array(polygon)
            center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        else:
            center = polygon_centroid(polygon)
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    result = []
    for x, y in polygon:
        tx, ty = x - cx, y - cy
        rx = cos_t * tx - sin_t * ty + cx
        ry = sin_t * tx + cos_t * ty + cy
        result.append((float(rx), float(ry)))
    return result


# ─── polygon_similarity ───────────────────────────────────────────────────────

def polygon_similarity(a: Polygon, b: Polygon) -> float:
    """Близость двух полигонов по площади и периметру [0, 1].

    Использует соотношение min/max для площади и периметра.

    Аргументы:
        a: Первый полигон (N >= 3).
        b: Второй полигон (N >= 3).

    Возвращает:
        Значение в [0, 1].
    """
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    peri_a = polygon_perimeter(a)
    peri_b = polygon_perimeter(b)

    def _ratio(x: float, y: float) -> float:
        if max(x, y) < 1e-12:
            return 1.0
        return float(min(x, y) / max(x, y))

    return float((_ratio(area_a, area_b) + _ratio(peri_a, peri_b)) / 2.0)
