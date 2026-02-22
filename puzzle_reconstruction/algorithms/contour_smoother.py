"""Сглаживание и передискретизация контуров фрагментов документа.

Предоставляет инструменты для сглаживания и равномерной передискретизации
кривых краёв, извлечённых из фрагментов документа, с вычислением
сходства между кривыми.

Публичный API:
    SmootherConfig      — параметры сглаживания и передискретизации
    SmoothedContour     — сглаженный контур с метаданными
    smooth_gaussian     — гауссово сглаживание по координатам
    resample_contour    — равномерная передискретизация по длине дуги
    compute_arc_length  — длина ломаной линии (дуги контура)
    smooth_and_resample — сгладить + передискретизировать за один вызов
    align_contours      — циклическое выравнивание двух замкнутых контуров
    contour_similarity  — скалярная оценка сходства двух контуров
    batch_smooth        — пакетное сглаживание списка контуров
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─── SmootherConfig ───────────────────────────────────────────────────────────

@dataclass
class SmootherConfig:
    """Параметры сглаживания и передискретизации контура.

    Атрибуты:
        sigma:    СКО гауссова сглаживания (>= 0; 0 → без сглаживания).
        n_points: Число точек после передискретизации (>= 2).
        closed:   True если контур замкнут (последняя точка → первая).
    """

    sigma: float = 1.0
    n_points: int = 64
    closed: bool = False

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError(
                f"sigma должен быть >= 0, получено {self.sigma}"
            )
        if self.n_points < 2:
            raise ValueError(
                f"n_points должен быть >= 2, получено {self.n_points}"
            )


# ─── SmoothedContour ──────────────────────────────────────────────────────────

@dataclass
class SmoothedContour:
    """Сглаженный и передискретизированный контур.

    Атрибуты:
        points:     Массив формы (N, 2) с координатами (x, y) float64.
        original_n: Исходное число точек до обработки.
        method:     Имя метода сглаживания ('gaussian' или 'none').
        params:     Параметры, использованные при обработке.
    """

    points: np.ndarray       # shape (N, 2), dtype float64
    original_n: int
    method: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original_n < 0:
            raise ValueError(
                f"original_n должен быть >= 0, получено {self.original_n}"
            )

    @property
    def n_points(self) -> int:
        """Число точек после обработки."""
        return len(self.points)

    @property
    def length(self) -> float:
        """Длина дуги контура (пикселей)."""
        return compute_arc_length(self.points, closed=self.params.get("closed", False))

    @property
    def is_closed(self) -> bool:
        """True если контур замкнут."""
        return bool(self.params.get("closed", False))


# ─── compute_arc_length ───────────────────────────────────────────────────────

def compute_arc_length(points: np.ndarray, closed: bool = False) -> float:
    """Вычислить длину ломаной (длину дуги контура).

    Аргументы:
        points: Массив формы (N, 2) или (N, 1, 2) с координатами.
        closed: Если True, добавить сегмент от последней точки к первой.

    Возвращает:
        Суммарная длина дуги >= 0.

    Исключения:
        ValueError: Если points содержит менее 1 точки.
    """
    pts = points.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("points не должен быть пустым")
    if n == 1:
        return 0.0
    diffs = np.diff(pts, axis=0)
    length = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
    if closed:
        closing = pts[0] - pts[-1]
        length += float(np.sqrt((closing ** 2).sum()))
    return length


# ─── smooth_gaussian ──────────────────────────────────────────────────────────

def smooth_gaussian(
    points: np.ndarray,
    sigma: float = 1.0,
    closed: bool = False,
) -> np.ndarray:
    """Гауссово сглаживание координат контура.

    Каждая координата (x, y) сглаживается независимо вдоль индекса точки.
    Для замкнутых контуров применяется режим 'wrap'.

    Аргументы:
        points: Массив формы (N, 2) или (N, 1, 2).
        sigma:  СКО гауссова ядра (>= 0; 0 → без сглаживания).
        closed: Замкнутый контур (wrap-mode).

    Возвращает:
        Сглаженный массив формы (N, 2) float64.

    Исключения:
        ValueError: Если sigma < 0 или points пуст.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma должен быть >= 0, получено {sigma}")

    pts = points.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("points не должен быть пустым")

    if sigma < 1e-9:
        return pts.copy()

    mode = "wrap" if closed else "nearest"
    x_smooth = gaussian_filter1d(pts[:, 0], sigma=sigma, mode=mode)
    y_smooth = gaussian_filter1d(pts[:, 1], sigma=sigma, mode=mode)
    return np.stack([x_smooth, y_smooth], axis=1)


# ─── resample_contour ─────────────────────────────────────────────────────────

def resample_contour(
    points: np.ndarray,
    n_points: int,
    closed: bool = False,
) -> np.ndarray:
    """Равномерная передискретизация контура по длине дуги.

    Аргументы:
        points:   Массив формы (N, 2) или (N, 1, 2).
        n_points: Число точек в результате (>= 2).
        closed:   Если True, добавить замыкающий сегмент.

    Возвращает:
        Массив формы (n_points, 2) float64.

    Исключения:
        ValueError: Если n_points < 2 или points содержит < 1 точки.
    """
    if n_points < 2:
        raise ValueError(
            f"n_points должен быть >= 2, получено {n_points}"
        )

    pts = points.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n == 0:
        raise ValueError("points не должен быть пустым")
    if n == 1:
        return np.tile(pts[0], (n_points, 1))

    if closed:
        pts = np.vstack([pts, pts[:1]])

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    if total < 1e-9:
        return np.tile(pts[0], (n_points, 1))

    sample_positions = np.linspace(0.0, total, n_points)

    x_new = np.interp(sample_positions, cumlen, pts[:, 0])
    y_new = np.interp(sample_positions, cumlen, pts[:, 1])
    return np.stack([x_new, y_new], axis=1)


# ─── smooth_and_resample ──────────────────────────────────────────────────────

def smooth_and_resample(
    points: np.ndarray,
    cfg: Optional[SmootherConfig] = None,
) -> SmoothedContour:
    """Сгладить и передискретизировать контур.

    Аргументы:
        points: Массив формы (N, 2) или (N, 1, 2).
        cfg:    Параметры обработки (None → SmootherConfig()).

    Возвращает:
        SmoothedContour с обработанными точками.

    Исключения:
        ValueError: Если points пуст.
    """
    if cfg is None:
        cfg = SmootherConfig()

    pts = points.reshape(-1, 2).astype(np.float64)
    original_n = len(pts)
    if original_n == 0:
        raise ValueError("points не должен быть пустым")

    method = "none"
    if cfg.sigma > 1e-9:
        pts = smooth_gaussian(pts, sigma=cfg.sigma, closed=cfg.closed)
        method = "gaussian"

    pts = resample_contour(pts, n_points=cfg.n_points, closed=cfg.closed)

    return SmoothedContour(
        points=pts,
        original_n=original_n,
        method=method,
        params={
            "sigma": cfg.sigma,
            "n_points": cfg.n_points,
            "closed": cfg.closed,
        },
    )


# ─── align_contours ───────────────────────────────────────────────────────────

def align_contours(
    c1: np.ndarray,
    c2: np.ndarray,
) -> tuple:
    """Найти циклический сдвиг c2, минимизирующий среднеквадратичное расстояние до c1.

    Оба контура должны иметь одинаковое число точек.

    Аргументы:
        c1: Массив формы (N, 2) — эталонный контур.
        c2: Массив формы (N, 2) — выравниваемый контур.

    Возвращает:
        Кортеж (c1, c2_aligned), где c2_aligned — сдвинутая копия c2.

    Исключения:
        ValueError: Если контуры имеют разную длину или пусты.
    """
    pts1 = c1.reshape(-1, 2).astype(np.float64)
    pts2 = c2.reshape(-1, 2).astype(np.float64)
    n = len(pts1)
    if n == 0:
        raise ValueError("Контуры не должны быть пустыми")
    if len(pts2) != n:
        raise ValueError(
            f"Контуры должны иметь одинаковую длину: {n} vs {len(pts2)}"
        )

    best_shift = 0
    best_msd = float("inf")
    for shift in range(n):
        c2_shifted = np.roll(pts2, shift, axis=0)
        msd = float(np.mean(((pts1 - c2_shifted) ** 2).sum(axis=1)))
        if msd < best_msd:
            best_msd = msd
            best_shift = shift

    return pts1, np.roll(pts2, best_shift, axis=0)


# ─── contour_similarity ───────────────────────────────────────────────────────

def contour_similarity(
    c1: np.ndarray,
    c2: np.ndarray,
    metric: str = "l2",
) -> float:
    """Вычислить скалярную оценку сходства двух контуров ∈ [0, 1].

    Контуры передискретизируются до одинаковой длины перед сравнением.

    Аргументы:
        c1:     Массив (N, 2) или (N, 1, 2).
        c2:     Массив (M, 2) или (M, 1, 2).
        metric: Метрика ('l2', 'hausdorff').

    Возвращает:
        Скалярная оценка сходства ∈ [0, 1].

    Исключения:
        ValueError: Если metric неизвестна или контуры пусты.
    """
    valid_metrics = {"l2", "hausdorff"}
    if metric not in valid_metrics:
        raise ValueError(
            f"metric должен быть одним из {valid_metrics}, получено {metric!r}"
        )

    pts1 = c1.reshape(-1, 2).astype(np.float64)
    pts2 = c2.reshape(-1, 2).astype(np.float64)

    if len(pts1) == 0 or len(pts2) == 0:
        raise ValueError("Контуры не должны быть пустыми")

    n = max(len(pts1), len(pts2))
    if len(pts1) != n:
        pts1 = resample_contour(pts1, n_points=n)
    if len(pts2) != n:
        pts2 = resample_contour(pts2, n_points=n)

    if metric == "l2":
        dists = np.sqrt(((pts1 - pts2) ** 2).sum(axis=1))
        mean_dist = float(dists.mean())
        # Нормировка: экспоненциальный спад с масштабом 50 пикселей
        return float(math.exp(-mean_dist / 50.0))

    else:  # hausdorff
        # Направленный Хаусдорф: max(min dist от c1 до c2, min dist от c2 до c1)
        diffs1 = pts1[:, None, :] - pts2[None, :, :]
        dist_matrix = np.sqrt((diffs1 ** 2).sum(axis=2))
        h1 = float(dist_matrix.min(axis=1).max())
        h2 = float(dist_matrix.min(axis=0).max())
        hausdorff = max(h1, h2)
        return float(math.exp(-hausdorff / 50.0))


# ─── batch_smooth ─────────────────────────────────────────────────────────────

def batch_smooth(
    points_list: List[np.ndarray],
    cfg: Optional[SmootherConfig] = None,
) -> List[SmoothedContour]:
    """Пакетное сглаживание списка контуров.

    Аргументы:
        points_list: Список массивов формы (Ni, 2).
        cfg:         Параметры обработки (None → SmootherConfig()).

    Возвращает:
        Список SmoothedContour той же длины.
    """
    if cfg is None:
        cfg = SmootherConfig()
    return [smooth_and_resample(pts, cfg) for pts in points_list]
