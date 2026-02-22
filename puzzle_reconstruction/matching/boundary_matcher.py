"""
Сопоставление границ (контуров) фрагментов по геометрическим метрикам.

Вычисляет расстояние между граничными точками двух фрагментов с использованием
расстояний Хаусдорфа, Чамфера и приближения метрики Фреше, а затем агрегирует
их в единый балл совместимости.

Классы:
    BoundaryMatch — результат сравнения одной пары граничных контуров

Функции:
    extract_boundary_points — выборка точек вдоль указанной стороны контура
    hausdorff_distance      — направленное + симметричное расстояние Хаусдорфа
    chamfer_distance        — симметричное среднее минимальное расстояние
    frechet_approx          — приближённое дискретное расстояние Фреше
    score_boundary_pair     — нормированные оценки из трёх метрик + total
    match_boundary_pair     — полное сопоставление пары фрагментов
    batch_match_boundaries  — пакетное сопоставление списка пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── BoundaryMatch ────────────────────────────────────────────────────────────

@dataclass
class BoundaryMatch:
    """
    Результат сравнения граничных контуров двух фрагментов.

    Attributes:
        idx1:         Индекс первого фрагмента.
        idx2:         Индекс второго фрагмента.
        side1:        Сторона первого (0=верх, 1=право, 2=низ, 3=лево).
        side2:        Сторона второго.
        hausdorff:    Балл Хаусдорфа ∈ [0, 1] (выше → лучше).
        chamfer:      Балл Чамфера ∈ [0, 1].
        frechet:      Балл Фреше ∈ [0, 1].
        total_score:  Взвешенное среднее ∈ [0, 1].
        params:       Параметры сопоставления.
    """
    idx1:        int
    idx2:        int
    side1:       int
    side2:       int
    hausdorff:   float
    chamfer:     float
    frechet:     float
    total_score: float
    params:      Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"BoundaryMatch(idx1={self.idx1}, idx2={self.idx2}, "
                f"side1={self.side1}, side2={self.side2}, "
                f"total={self.total_score:.3f})")


# ─── extract_boundary_points ──────────────────────────────────────────────────

def extract_boundary_points(
    contour:  np.ndarray,
    side:     int,
    n_points: int = 50,
) -> np.ndarray:
    """
    Выбирает n_points точек вдоль указанной стороны ограничивающего прямоугольника
    контура путём равномерной выборки точек, ближайших к соответствующему краю.

    Args:
        contour:  Контур формы (N, 2) или (N, 1, 2), dtype float или int.
        side:     0=верх (мин y), 1=право (макс x), 2=низ (макс y), 3=лево (мин x).
        n_points: Количество возвращаемых точек.

    Returns:
        np.ndarray формы (n_points, 2) float64.

    Raises:
        ValueError: Если side не в [0, 3] или контур пустой.
    """
    if side not in (0, 1, 2, 3):
        raise ValueError(f"side must be 0, 1, 2, or 3; got {side!r}.")

    pts = contour.reshape(-1, 2).astype(np.float64)
    if pts.size == 0:
        raise ValueError("contour must not be empty.")

    # Определяем ось и направление сортировки
    if side == 0:   # верх → min y → ось y, сортируем по x
        axis, reverse = 1, False
    elif side == 1: # право → max x → ось x, сортируем по y
        axis, reverse = 0, True
    elif side == 2: # низ → max y → ось y, сортируем по x
        axis, reverse = 1, True
    else:           # лево → min x → ось x, сортируем по y
        axis, reverse = 0, False

    # Выбираем 20% точек, ближайших к соответствующему краю
    k      = max(1, len(pts) // 5)
    values = pts[:, 1 - axis]  # для side 0/2 (y) → берём x; для side 1/3 (x) → берём y
    # Но нам нужна проекция на ось, по которой ищем крайние точки
    edge_vals = pts[:, axis]
    if reverse:
        idx_sort = np.argsort(-edge_vals)
    else:
        idx_sort = np.argsort(edge_vals)
    edge_pts = pts[idx_sort[:k]]

    # Равномерная выборка из крайних точек
    if len(edge_pts) <= n_points:
        selected = edge_pts
    else:
        indices  = np.round(np.linspace(0, len(edge_pts) - 1, n_points)).astype(int)
        selected = edge_pts[indices]

    # Если не набрали n_points — дополним повторами последней точки
    if len(selected) < n_points:
        repeat = np.tile(selected[-1:], (n_points - len(selected), 1))
        selected = np.vstack([selected, repeat])

    return selected[:n_points]


# ─── hausdorff_distance ───────────────────────────────────────────────────────

def hausdorff_distance(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> float:
    """
    Симметричное расстояние Хаусдорфа между двумя облаками точек.

    H(A, B) = max(directed_H(A→B), directed_H(B→A))
    directed_H(A→B) = max_{a ∈ A} min_{b ∈ B} ||a - b||

    Args:
        pts1: (N, 2) float64 — первое облако точек.
        pts2: (M, 2) float64 — второе облако точек.

    Returns:
        Расстояние Хаусдорфа (≥ 0).
    """
    if pts1.size == 0 or pts2.size == 0:
        return 0.0
    # Матрица попарных расстояний (N, M)
    diff = pts1[:, None, :] - pts2[None, :, :]   # (N, M, 2)
    dist = np.linalg.norm(diff, axis=-1)           # (N, M)
    d_12 = dist.min(axis=1).max()
    d_21 = dist.min(axis=0).max()
    return float(max(d_12, d_21))


# ─── chamfer_distance ─────────────────────────────────────────────────────────

def chamfer_distance(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> float:
    """
    Симметричное расстояние Чамфера (среднее минимальное).

    C(A, B) = (mean_{a ∈ A} min_b ||a-b|| + mean_{b ∈ B} min_a ||b-a||) / 2.

    Args:
        pts1: (N, 2) float64.
        pts2: (M, 2) float64.

    Returns:
        Расстояние Чамфера (≥ 0).
    """
    if pts1.size == 0 or pts2.size == 0:
        return 0.0
    diff = pts1[:, None, :] - pts2[None, :, :]  # (N, M, 2)
    dist = np.linalg.norm(diff, axis=-1)         # (N, M)
    c_12 = dist.min(axis=1).mean()
    c_21 = dist.min(axis=0).mean()
    return float((c_12 + c_21) / 2.0)


# ─── frechet_approx ───────────────────────────────────────────────────────────

def frechet_approx(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> float:
    """
    Приближённое дискретное расстояние Фреше через динамическое программирование.

    Предполагает, что точки отсортированы вдоль кривых (как при обходе контура).

    Args:
        pts1: (N, 2) float64.
        pts2: (M, 2) float64.

    Returns:
        Приближённое расстояние Фреше (≥ 0).
    """
    if pts1.size == 0 or pts2.size == 0:
        return 0.0
    n, m = len(pts1), len(pts2)
    ca   = np.full((n, m), -1.0, dtype=np.float64)

    def _dist(i: int, j: int) -> float:
        return float(np.linalg.norm(pts1[i] - pts2[j]))

    def _c(i: int, j: int) -> float:
        if ca[i, j] > -1.0:
            return ca[i, j]
        d = _dist(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    # Ограничиваем размер для производительности
    MAX_PTS = 30
    if n > MAX_PTS or m > MAX_PTS:
        idx1 = np.round(np.linspace(0, n - 1, MAX_PTS)).astype(int)
        idx2 = np.round(np.linspace(0, m - 1, MAX_PTS)).astype(int)
        pts1 = pts1[idx1]
        pts2 = pts2[idx2]
        n, m = MAX_PTS, MAX_PTS
        ca   = np.full((n, m), -1.0, dtype=np.float64)

    return _c(n - 1, m - 1)


# ─── score_boundary_pair ──────────────────────────────────────────────────────

def score_boundary_pair(
    pts1:     np.ndarray,
    pts2:     np.ndarray,
    max_dist: float = 100.0,
    weights:  Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Вычисляет нормированные оценки [0, 1] для пары граничных облаков точек.

    score_x = exp(-distance_x / max_dist).

    Args:
        pts1:     (N, 2) float64.
        pts2:     (M, 2) float64.
        max_dist: Нормировочный масштаб расстояния.
        weights:  (w_hausdorff, w_chamfer, w_frechet). None → равные.

    Returns:
        (hausdorff_score, chamfer_score, frechet_score, total_score).
    """
    max_d = max(max_dist, 1e-9)

    h_dist = hausdorff_distance(pts1, pts2)
    c_dist = chamfer_distance(pts1, pts2)
    f_dist = frechet_approx(pts1, pts2)

    h_score = float(np.exp(-h_dist / max_d))
    c_score = float(np.exp(-c_dist / max_d))
    f_score = float(np.exp(-f_dist / max_d))

    if weights is None:
        wh, wc, wf = 1.0 / 3, 1.0 / 3, 1.0 / 3
    else:
        wh, wc, wf = weights
        s = wh + wc + wf
        if s > 1e-9:
            wh /= s; wc /= s; wf /= s

    total = float(np.clip(wh * h_score + wc * c_score + wf * f_score, 0.0, 1.0))
    return h_score, c_score, f_score, total


# ─── match_boundary_pair ──────────────────────────────────────────────────────

def match_boundary_pair(
    contour1:  np.ndarray,
    contour2:  np.ndarray,
    idx1:      int   = 0,
    idx2:      int   = 1,
    side1:     int   = 2,
    side2:     int   = 0,
    n_points:  int   = 50,
    max_dist:  float = 100.0,
    weights:   Optional[Tuple[float, float, float]] = None,
) -> BoundaryMatch:
    """
    Сопоставляет границы двух фрагментов.

    Args:
        contour1:  Контур первого фрагмента (N, 2) или (N, 1, 2).
        contour2:  Контур второго фрагмента.
        idx1:      Индекс первого фрагмента.
        idx2:      Индекс второго фрагмента.
        side1:     Сторона первого (0–3).
        side2:     Сторона второго (0–3).
        n_points:  Число точек выборки на каждой стороне.
        max_dist:  Нормировочный масштаб расстояния.
        weights:   Веса метрик.

    Returns:
        BoundaryMatch с полными метриками.
    """
    pts1 = extract_boundary_points(contour1, side1, n_points)
    pts2 = extract_boundary_points(contour2, side2, n_points)

    h_score, c_score, f_score, total = score_boundary_pair(
        pts1, pts2, max_dist=max_dist, weights=weights,
    )

    return BoundaryMatch(
        idx1=idx1, idx2=idx2,
        side1=side1, side2=side2,
        hausdorff=h_score,
        chamfer=c_score,
        frechet=f_score,
        total_score=total,
        params={
            "n_points": n_points,
            "max_dist": max_dist,
            "weights": weights,
        },
    )


# ─── batch_match_boundaries ───────────────────────────────────────────────────

def batch_match_boundaries(
    contours:   List[np.ndarray],
    pairs:      List[Tuple[int, int]],
    side_pairs: Optional[List[Tuple[int, int]]] = None,
    n_points:   int   = 50,
    max_dist:   float = 100.0,
    weights:    Optional[Tuple[float, float, float]] = None,
) -> List[BoundaryMatch]:
    """
    Пакетное сопоставление границ для списка пар фрагментов.

    Args:
        contours:   Список контуров.
        pairs:      Список пар [(idx1, idx2), ...].
        side_pairs: [(side1, side2), ...] или None → (2, 0) для всех.
        n_points:   Число точек выборки.
        max_dist:   Нормировочный масштаб расстояния.
        weights:    Веса метрик.

    Returns:
        Список BoundaryMatch длиной len(pairs).
    """
    if not pairs:
        return []

    if side_pairs is None:
        side_pairs = [(2, 0)] * len(pairs)

    return [
        match_boundary_pair(
            contours[i1], contours[i2],
            idx1=i1, idx2=i2,
            side1=s1, side2=s2,
            n_points=n_points,
            max_dist=max_dist,
            weights=weights,
        )
        for (i1, i2), (s1, s2) in zip(pairs, side_pairs)
    ]
