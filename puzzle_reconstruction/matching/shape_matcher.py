"""
Сопоставление фрагментов по форме контура.

Экспортирует:
    ShapeMatchResult    — результат сопоставления двух контуров
    hu_moments          — вектор инвариантных моментов Ху (7 значений)
    hu_distance         — расстояние между двумя наборами моментов Ху
    zernike_approx      — приближённые моменты Цернике через радиальный профиль
    match_shapes        — комплексное сопоставление по форме (Ху + IoU + Chamfer)
    find_best_shape_match — лучший кандидат для данного контура из списка
    batch_match_shapes  — матрица сходства N×M для двух наборов контуров
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class ShapeMatchResult:
    """Результат сопоставления двух контуров по форме.

    Attributes:
        idx1:        Индекс первого контура.
        idx2:        Индекс второго контура.
        hu_dist:     Расстояние Ху (меньше — похожее).
        iou:         IoU масок (больше — похожее).
        chamfer:     Расстояние Шамфера (меньше — похожее).
        score:       Итоговая оценка сходства [0, 1] (1 — идентично).
        params:      Доп. параметры сопоставления.
    """
    idx1: int
    idx2: int
    hu_dist: float
    iou: float
    chamfer: float
    score: float
    params: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ShapeMatchResult({self.idx1}↔{self.idx2}, "
            f"score={self.score:.4f}, iou={self.iou:.4f})"
        )


# ─── Приватные утилиты ────────────────────────────────────────────────────────

def _to_float64(contour: np.ndarray) -> np.ndarray:
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim == 3 and c.shape[2] == 2:
        c = c.reshape(-1, 2)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(
            f"Contour must have shape (N, 2) or (N, 1, 2), got {contour.shape}"
        )
    return c


def _contour_to_cv2(contour: np.ndarray) -> np.ndarray:
    return np.round(contour).astype(np.int32).reshape(-1, 1, 2)


def _contour_iou(
    c1: np.ndarray,
    c2: np.ndarray,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> float:
    if len(c1) < 3 or len(c2) < 3:
        return 0.0
    if canvas_size is None:
        all_pts = np.vstack([c1, c2])
        h = int(np.ceil(all_pts[:, 1].max())) + 2
        w = int(np.ceil(all_pts[:, 0].max())) + 2
        canvas_size = (max(h, 1), max(w, 1))
    h, w = canvas_size
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [_contour_to_cv2(c1)], 1)
    cv2.fillPoly(m2, [_contour_to_cv2(c2)], 1)
    inter = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    return float(inter) / float(union) if union > 0 else 0.0


def _chamfer(pts1: np.ndarray, pts2: np.ndarray) -> float:
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0
    d12 = np.sqrt(((pts1[:, None, :] - pts2[None, :, :]) ** 2).sum(axis=2))
    return float(0.5 * (d12.min(axis=1).mean() + d12.min(axis=0).mean()))


# ─── Публичные функции ────────────────────────────────────────────────────────

def hu_moments(contour: np.ndarray) -> np.ndarray:
    """Вычислить 7 инвариантных моментов Ху для контура.

    Args:
        contour: Массив точек контура (N, 2) или (N, 1, 2).

    Returns:
        Массив float64 формы (7,).
        Для пустого или вырожденного контура — нулевой вектор.

    Raises:
        ValueError: Если форма массива некорректна.
    """
    c = _to_float64(contour)
    if len(c) < 3:
        return np.zeros(7, dtype=np.float64)
    pts = c.reshape(-1, 1, 2).astype(np.float32)
    M = cv2.moments(pts)
    hm = cv2.HuMoments(M).flatten()
    # Логарифмическая нормализация (стандартный приём)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hm = np.where(
            hm != 0,
            -np.sign(hm) * np.log10(np.abs(hm) + 1e-30),
            0.0,
        )
    return log_hm.astype(np.float64)


def hu_distance(hm1: np.ndarray, hm2: np.ndarray) -> float:
    """L2-расстояние между двумя наборами моментов Ху.

    Args:
        hm1: Вектор моментов (7,).
        hm2: Вектор моментов (7,).

    Returns:
        Неотрицательное расстояние.

    Raises:
        ValueError: Если длина векторов не совпадает.
    """
    a = np.asarray(hm1, dtype=np.float64).ravel()
    b = np.asarray(hm2, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError(
            f"Moment vectors must have the same length, got {len(a)} and {len(b)}"
        )
    return float(np.linalg.norm(a - b))


def zernike_approx(
    contour: np.ndarray,
    n_radii: int = 8,
) -> np.ndarray:
    """Приближённые моменты Цернике через радиальный профиль контура.

    Вычисляет нормализованный гистограммный профиль расстояний от
    центроида до точек контура. Это быстрое приближение, не требующее
    полного вычисления полиномов Цернике.

    Args:
        contour: Массив точек контура (N, 2).
        n_radii: Количество радиальных корзин (≥ 2).

    Returns:
        Нормализованный вектор float64 формы (n_radii,).
        Для пустого контура — нулевой вектор.

    Raises:
        ValueError: Если ``n_radii`` < 2.
    """
    if n_radii < 2:
        raise ValueError(f"n_radii must be >= 2, got {n_radii}")
    c = _to_float64(contour)
    if len(c) < 3:
        return np.zeros(n_radii, dtype=np.float64)
    centroid = c.mean(axis=0)
    radii = np.linalg.norm(c - centroid, axis=1)
    max_r = radii.max()
    if max_r < 1e-12:
        return np.zeros(n_radii, dtype=np.float64)
    hist, _ = np.histogram(radii / max_r, bins=n_radii, range=(0.0, 1.0))
    total = hist.sum()
    return hist.astype(np.float64) / (total + 1e-12)


def match_shapes(
    contour1: np.ndarray,
    contour2: np.ndarray,
    idx1: int = 0,
    idx2: int = 1,
    canvas_size: Optional[Tuple[int, int]] = None,
    max_chamfer: float = 50.0,
    weights: Optional[Tuple[float, float, float]] = None,
) -> ShapeMatchResult:
    """Комплексное сопоставление двух контуров по форме.

    Объединяет три меры:
    - Расстояние Ху (нормализованное): `d_hu / (d_hu + 1)`
    - IoU масок
    - Нормализованное расстояние Шамфера: `1 - min(chamfer, max_chamfer) / max_chamfer`

    Args:
        contour1:    Первый контур (N, 2).
        contour2:    Второй контур (M, 2).
        idx1:        Индекс первого контура.
        idx2:        Индекс второго контура.
        canvas_size: Размер холста для IoU; ``None`` → автоматически.
        max_chamfer: Нормировочное расстояние Шамфера.
        weights:     Веса (w_hu, w_iou, w_chamfer); по умолчанию (1/3, 1/3, 1/3).

    Returns:
        :class:`ShapeMatchResult`.

    Raises:
        ValueError: Если форма массивов некорректна.
    """
    if weights is None:
        weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    w_hu, w_iou, w_chamfer = weights
    total_w = w_hu + w_iou + w_chamfer
    if total_w < 1e-9:
        raise ValueError("Sum of weights must be > 0")

    c1 = _to_float64(contour1)
    c2 = _to_float64(contour2)

    # Моменты Ху
    hm1 = hu_moments(c1)
    hm2 = hu_moments(c2)
    d_hu = hu_distance(hm1, hm2)
    hu_sim = 1.0 / (1.0 + d_hu)

    # IoU
    iou = _contour_iou(c1, c2, canvas_size=canvas_size)

    # Шамфер
    n_sample = min(len(c1), len(c2), 100)
    if n_sample >= 2:
        idx_arr1 = np.round(np.linspace(0, len(c1) - 1, n_sample)).astype(int)
        idx_arr2 = np.round(np.linspace(0, len(c2) - 1, n_sample)).astype(int)
        pts1 = c1[idx_arr1]
        pts2 = c2[idx_arr2]
        chamfer = _chamfer(pts1, pts2)
    else:
        chamfer = float(max_chamfer)
    chamfer_sim = 1.0 - min(chamfer, max_chamfer) / max_chamfer

    score = (w_hu * hu_sim + w_iou * iou + w_chamfer * chamfer_sim) / total_w
    score = float(np.clip(score, 0.0, 1.0))

    return ShapeMatchResult(
        idx1=idx1, idx2=idx2,
        hu_dist=d_hu,
        iou=iou,
        chamfer=chamfer,
        score=score,
        params={
            "w_hu": w_hu, "w_iou": w_iou, "w_chamfer": w_chamfer,
            "max_chamfer": max_chamfer,
        },
    )


def find_best_shape_match(
    query: np.ndarray,
    candidates: List[np.ndarray],
    query_idx: int = 0,
    **match_kwargs,
) -> Optional[ShapeMatchResult]:
    """Найти наиболее похожий контур из списка кандидатов.

    Args:
        query:        Контур-запрос (N, 2).
        candidates:   Список контуров-кандидатов.
        query_idx:    Индекс контура-запроса (для поля ``idx1``).
        **match_kwargs: Дополнительные аргументы для :func:`match_shapes`.

    Returns:
        :class:`ShapeMatchResult` с наибольшим ``score``,
        или ``None``, если список кандидатов пуст.
    """
    if not candidates:
        return None
    results = [
        match_shapes(query, c, idx1=query_idx, idx2=i, **match_kwargs)
        for i, c in enumerate(candidates)
    ]
    return max(results, key=lambda r: r.score)


def batch_match_shapes(
    contours1: List[np.ndarray],
    contours2: List[np.ndarray],
    **match_kwargs,
) -> np.ndarray:
    """Построить матрицу оценок сходства N×M для двух наборов контуров.

    Args:
        contours1:    Список из N контуров.
        contours2:    Список из M контуров.
        **match_kwargs: Дополнительные аргументы для :func:`match_shapes`.

    Returns:
        Матрица float32 формы (N, M).
        Значение [i, j] — оценка сходства contours1[i] и contours2[j].
        Для пустых списков — матрица (0, 0).
    """
    n, m = len(contours1), len(contours2)
    if n == 0 or m == 0:
        return np.empty((n, m), dtype=np.float32)
    matrix = np.zeros((n, m), dtype=np.float32)
    for i, c1 in enumerate(contours1):
        for j, c2 in enumerate(contours2):
            result = match_shapes(c1, c2, idx1=i, idx2=j, **match_kwargs)
            matrix[i, j] = float(result.score)
    return matrix
