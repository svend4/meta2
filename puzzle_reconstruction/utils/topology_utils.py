"""
Топологические утилиты для анализа форм фрагментов.

Предоставляет функции для анализа топологических свойств бинарных масок
и контуров фрагментов: числа Эйлера, связных компонент, дефектов выпуклости,
показателей компактности и пакетной обработки.

Экспортирует:
    TopologyConfig         — параметры топологических операций
    compute_euler_number   — число Эйлера бинарной маски
    count_holes            — количество дырок (внутренних компонент фона)
    compute_solidity       — отношение площади к площади выпуклой оболочки
    compute_extent         — отношение площади к площади ограничивающего прямоугольника
    compute_convexity      — мера выпуклости контура [0, 1]
    compute_compactness    — изопериметрический коэффициент [0, 1]
    is_simply_connected    — проверка простой связности (без дырок)
    shape_complexity       — агрегированный показатель сложности формы
    batch_topology         — вычисление метрик для пакета масок
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ─── TopologyConfig ───────────────────────────────────────────────────────────

@dataclass
class TopologyConfig:
    """Параметры топологических операций.

    Attributes:
        connectivity:  Связность при анализе компонент (4 или 8).
        min_area:      Минимальная площадь компоненты (пиксели, >= 1).
    """
    connectivity: int = 8
    min_area:     int = 1

    def __post_init__(self) -> None:
        if self.connectivity not in (4, 8):
            raise ValueError(
                f"connectivity must be 4 or 8, got {self.connectivity}"
            )
        if self.min_area < 1:
            raise ValueError(
                f"min_area must be >= 1, got {self.min_area}"
            )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Привести маску к bool 2-D."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got ndim={mask.ndim}")
    return mask


def _label_components(binary: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Пометить связные компоненты наивным flood-fill (BFS).

    Returns:
        Массив int32 той же формы: 0 = фон, 1..K = компоненты.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    current_label = 0

    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
    else:
        nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for r in range(h):
        for c in range(w):
            if binary[r, c] and not visited[r, c]:
                current_label += 1
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    labels[cr, cc] = current_label
                    for dr, dc in nbrs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if binary[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
    return labels


# ─── compute_euler_number ─────────────────────────────────────────────────────

def compute_euler_number(mask: np.ndarray) -> int:
    """Вычислить число Эйлера бинарной маски.

    Число Эйлера = C - H, где C — число связных компонент переднего плана,
    H — число дырок (внутренних компонент фона).

    Args:
        mask: 2-D бинарный массив (bool или uint8, 0/non-0).

    Returns:
        Целое число — топологический инвариант.

    Raises:
        ValueError: Если mask не 2-D.
    """
    mask = _validate_mask(mask)
    n_fg = int(np.max(_label_components(mask, connectivity=8)))
    n_holes = count_holes(mask)
    return n_fg - n_holes


# ─── count_holes ──────────────────────────────────────────────────────────────

def count_holes(mask: np.ndarray) -> int:
    """Подсчитать количество дырок (внутренних компонент фона).

    Дырка — связная компонента фоновых пикселей, полностью окружённая
    передним планом (не касается границы маски).

    Args:
        mask: 2-D бинарный массив.

    Returns:
        Неотрицательное целое число.

    Raises:
        ValueError: Если mask не 2-D.
    """
    mask = _validate_mask(mask)
    bg = ~mask
    bg_labels = _label_components(bg, connectivity=4)
    n_bg = int(np.max(bg_labels)) if bg_labels.max() > 0 else 0

    # Компоненты, касающиеся границы, — внешний фон
    border_labels: set[int] = set()
    border_labels.update(bg_labels[0, :].tolist())
    border_labels.update(bg_labels[-1, :].tolist())
    border_labels.update(bg_labels[:, 0].tolist())
    border_labels.update(bg_labels[:, -1].tolist())
    border_labels.discard(0)

    return max(0, n_bg - len(border_labels))


# ─── compute_solidity ─────────────────────────────────────────────────────────

def compute_solidity(contour: np.ndarray) -> float:
    """Вычислить солидность: area / convex_hull_area.

    Солидность ∈ (0, 1]: 1.0 — идеально выпуклый контур.

    Args:
        contour: (N, 2) массив точек контура.

    Returns:
        float в (0, 1]. Если площадь выпуклой оболочки ~0, возвращает 0.0.

    Raises:
        ValueError: Если contour не (N, 2) или N < 3.
    """
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"contour must be (N, 2), got shape {pts.shape}"
        )
    if len(pts) < 3:
        raise ValueError(f"contour must have >= 3 points, got {len(pts)}")

    def _poly_area(p: np.ndarray) -> float:
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    area = _poly_area(pts)

    # Convex hull via gift wrapping
    hull = _convex_hull_pts(pts)
    hull_area = _poly_area(hull)

    if hull_area < 1e-12:
        return 0.0
    return min(1.0, area / hull_area)


def _convex_hull_pts(pts: np.ndarray) -> np.ndarray:
    """Выпуклая оболочка методом Грэхема (упрощённый)."""
    if len(pts) <= 3:
        return pts.copy()

    # Find bottom-left point
    idx = np.lexsort((pts[:, 0], pts[:, 1]))
    pts = pts[idx]
    pivot = pts[0]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    angles = np.arctan2(pts[1:, 1] - pivot[1], pts[1:, 0] - pivot[0])
    order = np.argsort(angles)
    sorted_pts = np.vstack([pivot, pts[1:][order]])

    hull = [sorted_pts[0], sorted_pts[1]]
    for p in sorted_pts[2:]:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return np.array(hull)


# ─── compute_extent ───────────────────────────────────────────────────────────

def compute_extent(contour: np.ndarray) -> float:
    """Вычислить экстент: area / bounding_box_area.

    Экстент ∈ (0, 1]: 1.0 — контур заполняет весь ограничивающий прямоугольник.

    Args:
        contour: (N, 2) массив точек контура.

    Returns:
        float в (0, 1]. Если bounding_box_area ~0, возвращает 0.0.

    Raises:
        ValueError: Если contour не (N, 2) или N < 3.
    """
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"contour must be (N, 2), got shape {pts.shape}"
        )
    if len(pts) < 3:
        raise ValueError(f"contour must have >= 3 points, got {len(pts)}")

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    bbox_area = float((x.max() - x.min()) * (y.max() - y.min()))

    if bbox_area < 1e-12:
        return 0.0
    return min(1.0, area / bbox_area)


# ─── compute_convexity ────────────────────────────────────────────────────────

def compute_convexity(contour: np.ndarray) -> float:
    """Вычислить выпуклость: perimeter_hull / perimeter_contour.

    Значение ∈ (0, 1]: 1.0 — контур является выпуклым.

    Args:
        contour: (N, 2) массив точек контура.

    Returns:
        float в (0, 1].

    Raises:
        ValueError: Если contour не (N, 2) или N < 3.
    """
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"contour must be (N, 2), got shape {pts.shape}"
        )
    if len(pts) < 3:
        raise ValueError(f"contour must have >= 3 points, got {len(pts)}")

    def _perimeter(p: np.ndarray) -> float:
        diffs = np.diff(np.vstack([p, p[:1]]), axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    perim = _perimeter(pts)
    if perim < 1e-12:
        return 1.0

    hull = _convex_hull_pts(pts)
    hull_perim = _perimeter(hull)

    return min(1.0, hull_perim / perim)


# ─── compute_compactness ──────────────────────────────────────────────────────

def compute_compactness(contour: np.ndarray) -> float:
    """Вычислить компактность (изопериметрический коэффициент).

    compactness = 4π × area / perimeter²

    Значение ∈ (0, 1]: 1.0 — идеальная окружность.

    Args:
        contour: (N, 2) массив точек контура.

    Returns:
        float в (0, 1].

    Raises:
        ValueError: Если contour не (N, 2) или N < 3.
    """
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"contour must be (N, 2), got shape {pts.shape}"
        )
    if len(pts) < 3:
        raise ValueError(f"contour must have >= 3 points, got {len(pts)}")

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))

    if perimeter < 1e-12:
        return 1.0
    return min(1.0, 4.0 * np.pi * area / (perimeter ** 2))


# ─── is_simply_connected ──────────────────────────────────────────────────────

def is_simply_connected(mask: np.ndarray) -> bool:
    """Проверить, является ли маска просто связной (без дырок).

    Args:
        mask: 2-D бинарный массив.

    Returns:
        True если нет дырок (count_holes == 0), иначе False.

    Raises:
        ValueError: Если mask не 2-D.
    """
    return count_holes(mask) == 0


# ─── shape_complexity ─────────────────────────────────────────────────────────

def shape_complexity(contour: np.ndarray) -> float:
    """Агрегированный показатель сложности формы.

    Вычисляется как 1 − (compactness × convexity).
    Значение ∈ [0, 1]: 0 — идеальная окружность, 1 — максимально сложная форма.

    Args:
        contour: (N, 2) массив точек контура.

    Returns:
        float в [0, 1].

    Raises:
        ValueError: Если contour не (N, 2) или N < 3.
    """
    return float(1.0 - compute_compactness(contour) * compute_convexity(contour))


# ─── batch_topology ───────────────────────────────────────────────────────────

def batch_topology(
    contours: List[np.ndarray],
) -> List[dict]:
    """Вычислить топологические метрики для пакета контуров.

    Args:
        contours: Список массивов (N_i, 2).

    Returns:
        Список словарей с ключами:
        ``solidity``, ``extent``, ``convexity``, ``compactness``, ``complexity``.

    Raises:
        ValueError: Если contours пуст.
    """
    if not contours:
        raise ValueError("contours must not be empty")
    results = []
    for c in contours:
        results.append({
            "solidity":    compute_solidity(c),
            "extent":      compute_extent(c),
            "convexity":   compute_convexity(c),
            "compactness": compute_compactness(c),
            "complexity":  shape_complexity(c),
        })
    return results
