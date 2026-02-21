"""
Утилиты для работы с контурами (набор точек формы (N, 2) float64).

Экспортирует:
    simplify_contour         — упрощение методом Дугласа–Пёккера
    interpolate_contour      — ресэмплинг до заданного числа точек
    contour_area             — площадь контура (формула Гаусса)
    contour_perimeter        — периметр контура
    contour_bbox             — ограничивающий прямоугольник (x, y, w, h)
    contour_centroid         — центроид контура
    contour_iou              — IoU двух контуров через растеризацию масок
    align_contour_orientation — нормализация ориентации (CW/CCW)
    contours_to_mask         — растеризация контура в бинарную маску
    mask_to_contour          — извлечение внешнего контура из маски
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ─── Приватные утилиты ────────────────────────────────────────────────────────

def _to_float64(contour: np.ndarray) -> np.ndarray:
    """Привести контур к float64 (N, 2)."""
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim == 3 and c.shape[2] == 2:
        c = c.reshape(-1, 2)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(
            f"Contour must have shape (N, 2), got {tuple(contour.shape)}"
        )
    return c


def _contour_to_int32(contour: np.ndarray) -> np.ndarray:
    """Привести контур к int32 для cv2."""
    pts = np.round(contour).astype(np.int32)
    return pts.reshape(-1, 1, 2)


# ─── Публичные функции ────────────────────────────────────────────────────────

def simplify_contour(
    contour: np.ndarray,
    epsilon: float = 2.0,
) -> np.ndarray:
    """Упростить контур методом Дугласа–Пёккера.

    Args:
        contour: Массив точек (N, 2).
        epsilon: Максимальное допустимое отклонение (в пикселях).

    Returns:
        Упрощённый контур (M, 2) float64, M ≤ N.

    Raises:
        ValueError: Если форма массива некорректна.
    """
    c = _to_float64(contour)
    if len(c) == 0:
        return c
    pts = c.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(pts, epsilon=float(epsilon), closed=True)
    return simplified.reshape(-1, 2).astype(np.float64)


def interpolate_contour(
    contour: np.ndarray,
    n_points: int = 100,
) -> np.ndarray:
    """Ресэмплировать контур до ``n_points`` равноотстоящих точек.

    Args:
        contour:  Исходный контур (N, 2).
        n_points: Желаемое количество точек (≥ 2).

    Returns:
        Массив (n_points, 2) float64.

    Raises:
        ValueError: Если ``n_points`` < 2 или контур пуст.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    c = _to_float64(contour)
    if len(c) == 0:
        raise ValueError("Cannot interpolate an empty contour")
    # Вычисляем кумулятивные длины сегментов (с замыканием)
    closed = np.vstack([c, c[:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-12:
        # Вырожденный контур — все точки совпадают
        return np.tile(c[0], (n_points, 1))
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    result = np.empty((n_points, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cum, t, side="right")) - 1
        idx = min(idx, len(c) - 1)
        seg_start = cum[idx]
        seg_len = seg_lens[idx] if idx < len(seg_lens) else 1.0
        alpha = (t - seg_start) / seg_len if seg_len > 1e-12 else 0.0
        p0 = c[idx]
        p1 = closed[idx + 1]
        result[i] = p0 + alpha * (p1 - p0)
    return result


def contour_area(contour: np.ndarray) -> float:
    """Площадь контура (формула Гаусса / «обуви»).

    Args:
        contour: Массив точек (N, 2).

    Returns:
        Неотрицательная площадь в пикселях².
    """
    c = _to_float64(contour)
    if len(c) < 3:
        return 0.0
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def contour_perimeter(contour: np.ndarray, closed: bool = True) -> float:
    """Периметр контура.

    Args:
        contour: Массив точек (N, 2).
        closed:  Если ``True``, добавить расстояние от последней точки к первой.

    Returns:
        Периметр в пикселях.
    """
    c = _to_float64(contour)
    if len(c) < 2:
        return 0.0
    segs = np.diff(c, axis=0)
    total = float(np.sum(np.hypot(segs[:, 0], segs[:, 1])))
    if closed:
        closing = c[-1] - c[0]
        total += float(np.hypot(closing[0], closing[1]))
    return total


def contour_bbox(contour: np.ndarray) -> Tuple[float, float, float, float]:
    """Ограничивающий прямоугольник контура.

    Args:
        contour: Массив точек (N, 2).

    Returns:
        Кортеж (x_min, y_min, width, height).
        Для пустого контура — (0, 0, 0, 0).
    """
    c = _to_float64(contour)
    if len(c) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x_min, y_min = c.min(axis=0)
    x_max, y_max = c.max(axis=0)
    return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))


def contour_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """Центроид контура.

    Для непустого контура использует моменты изображения (cv2.moments)
    с дополнительным обратным механизмом — среднее точек.

    Args:
        contour: Массив точек (N, 2).

    Returns:
        Кортеж (cx, cy).
        Для пустого контура — (0.0, 0.0).
    """
    c = _to_float64(contour)
    if len(c) == 0:
        return (0.0, 0.0)
    pts = _contour_to_int32(c)
    M = cv2.moments(pts)
    if M["m00"] > 1e-12:
        return (M["m10"] / M["m00"], M["m01"] / M["m00"])
    mean = c.mean(axis=0)
    return (float(mean[0]), float(mean[1]))


def contour_iou(
    contour1: np.ndarray,
    contour2: np.ndarray,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> float:
    """IoU двух контуров через растеризацию масок.

    Args:
        contour1:    Первый контур (N, 2).
        contour2:    Второй контур (M, 2).
        canvas_size: Размер холста (h, w) для растеризации.
                     Если ``None``, определяется автоматически.

    Returns:
        Значение IoU в диапазоне [0, 1].
    """
    c1 = _to_float64(contour1)
    c2 = _to_float64(contour2)
    if len(c1) < 3 or len(c2) < 3:
        return 0.0
    if canvas_size is None:
        all_pts = np.vstack([c1, c2])
        x_max = int(np.ceil(all_pts[:, 0].max())) + 2
        y_max = int(np.ceil(all_pts[:, 1].max())) + 2
        canvas_size = (max(y_max, 1), max(x_max, 1))
    h, w = canvas_size
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [_contour_to_int32(c1)], 1)
    cv2.fillPoly(m2, [_contour_to_int32(c2)], 1)
    inter = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    return float(inter) / float(union) if union > 0 else 0.0


def align_contour_orientation(
    contour: np.ndarray,
    clockwise: bool = True,
) -> np.ndarray:
    """Нормализовать ориентацию контура (CW или CCW).

    Args:
        contour:   Массив точек (N, 2).
        clockwise: Если ``True``, ориентация должна быть по часовой стрелке.

    Returns:
        Контур (N, 2) float64 с заданной ориентацией.
    """
    c = _to_float64(contour)
    if len(c) < 3:
        return c
    # Знак площади: > 0 → CCW (в системе координат экрана — CW)
    x, y = c[:, 0], c[:, 1]
    signed_area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    is_ccw = signed_area > 0
    # В системе координат изображения (ось Y вниз) CCW = signed_area > 0 ≡ CW
    # Поэтому: clockwise=True → хотим signed_area > 0 → is_ccw должно быть True
    if clockwise != is_ccw:
        c = c[::-1].copy()
    return c


def contours_to_mask(
    contour: np.ndarray,
    shape: Tuple[int, int],
    filled: bool = True,
) -> np.ndarray:
    """Растеризовать контур в бинарную маску.

    Args:
        contour: Массив точек (N, 2).
        shape:   Размер маски (height, width).
        filled:  Если ``True``, заливает область; иначе — только граница.

    Returns:
        Бинарная маска dtype uint8, значения 0/255.
    """
    c = _to_float64(contour)
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(c) < 2:
        return mask
    pts = _contour_to_int32(c)
    if filled:
        cv2.fillPoly(mask, [pts], 255)
    else:
        cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=1)
    return mask


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Извлечь внешний контур из бинарной маски.

    Args:
        mask: Бинарная маска (H, W) uint8 (ненулевые пиксели — объект).

    Returns:
        Контур (N, 2) float64 — самый длинный внешний контур.
        Пустой массив (0, 2), если контуров не найдено.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    longest = max(contours, key=lambda c: len(c))
    return longest.reshape(-1, 2).astype(np.float64)
