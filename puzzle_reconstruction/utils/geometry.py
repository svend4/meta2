"""
Геометрические утилиты для работы с контурами и полигонами.

Функции:
    rotation_matrix_2d  — матрица поворота 2×2
    rotate_points       — поворот массива точек N×2
    polygon_area        — площадь полигона (формула Гаусса/Shoelace)
    polygon_centroid    — центроид полигона
    bbox_from_points    — axis-aligned bounding box
    resample_curve      — равномерная передискретизация кривой
    align_centroids     — совмещение центроидов двух наборов точек
    poly_iou            — Intersection-over-Union двух выпуклых полигонов
    point_in_polygon    — тест «точка внутри полигона» (winding number)
    normalize_contour   — нормализация контура: перенос в начало, масштаб → 1
    smooth_contour      — сглаживание кривой скользящим средним
    curvature           — кривизна в каждой точке кривой
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ─── Матрица поворота ─────────────────────────────────────────────────────────

def rotation_matrix_2d(angle: float) -> np.ndarray:
    """
    Матрица поворота 2×2 на угол *angle* (радианы, против часовой стрелки).

    Returns:
        np.ndarray shape (2, 2), dtype float64.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def rotate_points(pts: np.ndarray,
                  angle: float,
                  center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Поворачивает массив точек (N, 2) на угол *angle* вокруг *center*.

    Args:
        pts:    (N, 2) — координаты точек.
        angle:  Угол поворота (радианы).
        center: Точка (2,) вокруг которой вращать. None → начало координат.

    Returns:
        (N, 2) повёрнутые точки, dtype float64.
    """
    pts = np.asarray(pts, dtype=np.float64)
    R   = rotation_matrix_2d(angle)
    if center is not None:
        c   = np.asarray(center, dtype=np.float64)
        return (pts - c) @ R.T + c
    return pts @ R.T


# ─── Площадь и центроид ───────────────────────────────────────────────────────

def polygon_area(pts: np.ndarray) -> float:
    """
    Площадь полигона по формуле Гаусса (Shoelace).

    Возвращает *знаковую* площадь:
        > 0  — вершины в порядке CCW (против часовой)
        < 0  — вершины в порядке CW
        0    — вырожденный полигон

    Args:
        pts: (N, 2) — вершины полигона.

    Returns:
        Знаковая площадь (float).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n   = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """
    Центроид полигона (формула для невырожденного многоугольника).

    Args:
        pts: (N, 2) — вершины.

    Returns:
        (2,) — координаты центроида.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n   = len(pts)
    if n == 0:
        return np.zeros(2)
    if n < 3:
        return pts.mean(axis=0)

    x, y   = pts[:, 0], pts[:, 1]
    xn     = np.roll(x, -1)
    yn     = np.roll(y, -1)
    cross  = x * yn - xn * y
    area6  = cross.sum()   # = 6 * signed_area

    if abs(area6) < 1e-12:
        return pts.mean(axis=0)

    cx = np.dot(x + xn, cross) / (3 * area6)
    cy = np.dot(y + yn, cross) / (3 * area6)
    return np.array([cx, cy])


# ─── Bounding box ─────────────────────────────────────────────────────────────

def bbox_from_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Axis-aligned bounding box.

    Args:
        pts: (N, 2).

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


# ─── Передискретизация кривой ─────────────────────────────────────────────────

def resample_curve(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Равномерная передискретизация кривой до *n* точек (по длине дуги).

    Args:
        pts: (M, 2) — исходные точки кривой.
        n:   Число точек в результате (≥ 2).

    Returns:
        (n, 2) — передискретизированная кривая.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2 or n < 2:
        return pts[:n] if len(pts) >= n else pts

    # Длины дуг
    deltas   = np.diff(pts, axis=0)
    seg_lens = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_len  = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = arc_len[-1]

    if total < 1e-12:
        return np.tile(pts[0], (n, 1))

    targets  = np.linspace(0.0, total, n)
    out      = np.empty((n, 2))
    for i, t in enumerate(targets):
        idx = np.searchsorted(arc_len, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(pts) - 2))
        seg = seg_lens[idx]
        if seg < 1e-12:
            out[i] = pts[idx]
        else:
            alpha  = (t - arc_len[idx]) / seg
            out[i] = pts[idx] + alpha * deltas[idx]

    return out


# ─── Совмещение центроидов ────────────────────────────────────────────────────

def align_centroids(source: np.ndarray,
                    target: np.ndarray) -> np.ndarray:
    """
    Переносит *source* так, чтобы его центроид совпал с центроидом *target*.

    Args:
        source: (N, 2) — точки, которые нужно переместить.
        target: (M, 2) — точки, центроид которых используется как цель.

    Returns:
        (N, 2) — смещённый source.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    delta = tgt.mean(axis=0) - src.mean(axis=0)
    return src + delta


# ─── IoU полигонов ────────────────────────────────────────────────────────────

def poly_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Intersection-over-Union двух выпуклых полигонов методом Sutherland–Hodgman.

    Args:
        pts_a, pts_b: (N, 2) — вершины выпуклых полигонов (CCW).

    Returns:
        IoU ∈ [0, 1].
    """
    def _clip(subject, clip):
        """Sutherland–Hodgman: обрезает subject по одному клип-ребру."""
        output = list(subject)
        if not output:
            return output
        n = len(clip)
        for i in range(n):
            if not output:
                break
            inp   = output
            output = []
            E = clip[(i + 1) % n]
            S = clip[i]
            for j in range(len(inp)):
                P = inp[j]
                Q = inp[j - 1]
                inside_P = _left(S, E, P)
                inside_Q = _left(S, E, Q)
                if inside_P:
                    if not inside_Q:
                        output.append(_intersect(S, E, Q, P))
                    output.append(P)
                elif inside_Q:
                    output.append(_intersect(S, E, Q, P))
        return output

    def _left(a, b, c):
        """True если c слева от прямой a→b (или на ней)."""
        return ((b[0] - a[0]) * (c[1] - a[1])
              - (b[1] - a[1]) * (c[0] - a[0])) >= -1e-9

    def _intersect(a, b, c, d):
        """Точка пересечения прямых a–b и c–d."""
        r = np.array(b) - np.array(a)
        s = np.array(d) - np.array(c)
        denom = r[0] * s[1] - r[1] * s[0]
        if abs(denom) < 1e-12:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
        return (a[0] + t * r[0], a[1] + t * r[1])

    def _poly_area(poly):
        if len(poly) < 3:
            return 0.0
        n = len(poly)
        s = 0.0
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2.0

    pa = [tuple(p) for p in np.asarray(pts_a, dtype=np.float64)]
    pb = [tuple(p) for p in np.asarray(pts_b, dtype=np.float64)]

    inter = _clip(pa, pb)
    area_i = _poly_area(inter)
    area_a = _poly_area(pa)
    area_b = _poly_area(pb)
    union  = area_a + area_b - area_i

    return float(area_i / union) if union > 1e-12 else 0.0


# ─── Тест «точка внутри полигона» ─────────────────────────────────────────────

def point_in_polygon(point: np.ndarray,
                     polygon: np.ndarray) -> bool:
    """
    Проверяет принадлежность точки полигону (метод winding number).

    Args:
        point:   (2,) — проверяемая точка.
        polygon: (N, 2) — вершины полигона (любой порядок).

    Returns:
        True если точка внутри или на границе полигона.
    """
    pt  = np.asarray(point, dtype=np.float64)
    pol = np.asarray(polygon, dtype=np.float64)
    n   = len(pol)
    if n < 3:
        return False

    winding = 0
    x, y    = pt
    for i in range(n):
        x0, y0 = pol[i]
        x1, y1 = pol[(i + 1) % n]
        if y0 <= y:
            if y1 > y:
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                    winding += 1
        else:
            if y1 <= y:
                if (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                    winding -= 1

    return winding != 0


# ─── Нормализация контура ─────────────────────────────────────────────────────

def normalize_contour(pts: np.ndarray,
                       target_scale: float = 1.0) -> np.ndarray:
    """
    Нормализует контур: переносит в начало координат, масштабирует.

    Алгоритм:
        1. Вычитает центроид.
        2. Делит на диагональ описывающего прямоугольника (→ масштаб).

    Args:
        pts:          (N, 2) — точки контура.
        target_scale: Целевой масштаб диагонали (по умолчанию 1.0).

    Returns:
        (N, 2) — нормализованные точки.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts

    centroid  = pts.mean(axis=0)
    shifted   = pts - centroid

    x0, y0, x1, y1 = bbox_from_points(shifted)
    diag = np.hypot(x1 - x0, y1 - y0)
    if diag < 1e-12:
        return shifted

    return shifted * (target_scale / diag)


# ─── Сглаживание контура ──────────────────────────────────────────────────────

def smooth_contour(pts: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Сглаживает кривую скользящим средним с периодическими граничными условиями.

    Args:
        pts:    (N, 2) — точки кривой.
        window: Ширина окна (нечётное ≥ 3).

    Returns:
        (N, 2) — сглаженная кривая.
    """
    pts    = np.asarray(pts, dtype=np.float64)
    n      = len(pts)
    w      = max(3, window | 1)   # Гарантируем нечётность ≥ 3
    half   = w // 2

    # Расширяем массив за счёт периодической «упаковки»
    padded = np.concatenate([pts[-half:], pts, pts[:half]], axis=0)
    kernel = np.ones(w) / w

    out = np.stack([
        np.convolve(padded[:, 0], kernel, mode="valid"),
        np.convolve(padded[:, 1], kernel, mode="valid"),
    ], axis=-1)

    return out[:n]


# ─── Кривизна ────────────────────────────────────────────────────────────────

def curvature(pts: np.ndarray) -> np.ndarray:
    """
    Вычисляет дискретную кривизну κ в каждой точке открытой кривой.

    Формула: κ_i = |r' × r''| / |r'|³
        где r' и r'' — первая и вторая разности.

    Args:
        pts: (N, 2) — точки кривой.

    Returns:
        (N,) — кривизна в каждой точке (граничные точки → 0).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n   = len(pts)
    if n < 3:
        return np.zeros(n)

    dx  = np.gradient(pts[:, 0])
    dy  = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    cross   = dx * ddy - dy * ddx
    denom   = (dx ** 2 + dy ** 2) ** 1.5
    denom   = np.where(denom < 1e-12, 1e-12, denom)

    return np.abs(cross) / denom
