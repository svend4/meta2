"""Постобработка контуров фрагментов пазла.

Модуль предоставляет функции для сглаживания, упрощения, нормализации
и анализа контуров фрагментов, полученных после сегментации.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── ContourConfig ────────────────────────────────────────────────────────────

@dataclass
class ContourConfig:
    """Параметры обработки контуров.

    Атрибуты:
        n_points:      Целевое число точек при ресемплинге (>= 3).
        smooth_sigma:  Сигма гауссового сглаживания (>= 0).
        rdp_epsilon:   Допуск алгоритма Рамера–Дугласа–Пекера (>= 0).
        normalize:     Нормализовать контур в [-1, 1].
    """

    n_points: int = 128
    smooth_sigma: float = 1.0
    rdp_epsilon: float = 2.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.n_points < 3:
            raise ValueError(
                f"n_points должен быть >= 3, получено {self.n_points}"
            )
        if self.smooth_sigma < 0.0:
            raise ValueError(
                f"smooth_sigma должна быть >= 0, получено {self.smooth_sigma}"
            )
        if self.rdp_epsilon < 0.0:
            raise ValueError(
                f"rdp_epsilon должен быть >= 0, получено {self.rdp_epsilon}"
            )


# ─── ContourStats ─────────────────────────────────────────────────────────────

@dataclass
class ContourStats:
    """Статистики контура.

    Атрибуты:
        n_points:    Число точек контура.
        perimeter:   Периметр.
        area:        Площадь (по формуле Гаусса).
        compactness: 4π·Area / Perimeter² (1 = окружность).
        mean_curvature: Среднее абсолютное значение кривизны.
    """

    n_points: int
    perimeter: float
    area: float
    compactness: float
    mean_curvature: float

    def __post_init__(self) -> None:
        if self.n_points < 0:
            raise ValueError(
                f"n_points должен быть >= 0, получено {self.n_points}"
            )
        if self.perimeter < 0.0:
            raise ValueError(
                f"perimeter должен быть >= 0, получено {self.perimeter}"
            )
        if self.area < 0.0:
            raise ValueError(
                f"area должна быть >= 0, получено {self.area}"
            )


# ─── ContourResult ────────────────────────────────────────────────────────────

@dataclass
class ContourResult:
    """Результат обработки контура.

    Атрибуты:
        points:       Numpy-массив (N, 2) float.
        stats:        ContourStats.
        fragment_id:  ID фрагмента.
        simplified:   True если был применён RDP.
    """

    points: np.ndarray
    stats: ContourStats
    fragment_id: int
    simplified: bool = False

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points должен быть (N, 2), получено shape={self.points.shape}"
            )

    @property
    def n_points(self) -> int:
        """Число точек результирующего контура."""
        return int(self.points.shape[0])

    @property
    def centroid(self) -> Tuple[float, float]:
        """Центроид контура."""
        return (float(self.points[:, 0].mean()),
                float(self.points[:, 1].mean()))


# ─── resample_contour ─────────────────────────────────────────────────────────

def resample_contour(
    points: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Ресемплировать контур до фиксированного числа точек.

    Использует равномерную параметризацию по длине дуги.

    Аргументы:
        points:   Контур (N, 2), N >= 2.
        n_points: Целевое число точек (>= 2).

    Возвращает:
        Numpy-массив (n_points, 2).

    Исключения:
        ValueError: При некорректных входных данных.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен быть (N, 2), получено shape={pts.shape}"
        )
    if pts.shape[0] < 2:
        raise ValueError(
            f"Контур должен иметь >= 2 точек, получено {pts.shape[0]}"
        )
    if n_points < 2:
        raise ValueError(
            f"n_points должен быть >= 2, получено {n_points}"
        )

    # Длины отрезков (замкнутый контур)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]

    if total < 1e-12:
        return np.tile(pts[0], (n_points, 1))

    # Равномерно распределённые параметры
    t_new = np.linspace(0.0, total, n_points)
    result = np.zeros((n_points, 2))
    for i, t in enumerate(t_new):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg = seg_lens[idx] if idx < len(seg_lens) else 1.0
        alpha = (t - cum[idx]) / (seg + 1e-12)
        result[i] = pts[idx] * (1.0 - alpha) + pts[idx + 1] * alpha

    return result


# ─── smooth_contour ───────────────────────────────────────────────────────────

def smooth_contour(
    points: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Сгладить контур гауссовым фильтром (по каждой оси отдельно).

    Аргументы:
        points: Контур (N, 2).
        sigma:  Стандартное отклонение (>= 0; 0 = без сглаживания).

    Возвращает:
        Сглаженный контур (N, 2).

    Исключения:
        ValueError: При sigma < 0 или некорректной форме points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен быть (N, 2), получено shape={pts.shape}"
        )
    if sigma < 0.0:
        raise ValueError(f"sigma должна быть >= 0, получено {sigma}")
    if sigma < 1e-12:
        return pts.copy()

    from scipy.ndimage import gaussian_filter1d
    smoothed = np.stack([
        gaussian_filter1d(pts[:, 0], sigma, mode="wrap"),
        gaussian_filter1d(pts[:, 1], sigma, mode="wrap"),
    ], axis=1)
    return smoothed


# ─── rdp_simplify ─────────────────────────────────────────────────────────────

def rdp_simplify(
    points: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Упростить контур алгоритмом Рамера–Дугласа–Пекера.

    Аргументы:
        points:  Контур (N, 2).
        epsilon: Допустимое отклонение (>= 0).

    Возвращает:
        Упрощённый контур (M, 2) с M <= N.

    Исключения:
        ValueError: При epsilon < 0 или некорректной форме points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен быть (N, 2), получено shape={pts.shape}"
        )
    if epsilon < 0.0:
        raise ValueError(f"epsilon должен быть >= 0, получено {epsilon}")
    if len(pts) <= 2:
        return pts.copy()

    def _perp_dist(p, a, b):
        ab = b - a
        l2 = np.dot(ab, ab)
        if l2 < 1e-12:
            return np.linalg.norm(p - a)
        t = np.clip(np.dot(p - a, ab) / l2, 0, 1)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _rdp(pts_sub: np.ndarray) -> List[int]:
        if len(pts_sub) <= 2:
            return list(range(len(pts_sub)))
        max_dist = 0.0
        max_idx = len(pts_sub) // 2  # default to midpoint; prevents infinite recursion when collinear+epsilon=0
        a, b = pts_sub[0], pts_sub[-1]
        for i in range(1, len(pts_sub) - 1):
            d = _perp_dist(pts_sub[i], a, b)
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist < epsilon:
            return [0, len(pts_sub) - 1]
        left = _rdp(pts_sub[:max_idx + 1])
        right = _rdp(pts_sub[max_idx:])
        return left[:-1] + [i + max_idx for i in right]

    indices = _rdp(pts)
    return pts[indices]


# ─── normalize_contour ────────────────────────────────────────────────────────

def normalize_contour(points: np.ndarray) -> np.ndarray:
    """Нормализовать контур в квадрат [-1, 1] × [-1, 1].

    Центрирует по центроиду и масштабирует по максимальному отклонению.

    Аргументы:
        points: Контур (N, 2).

    Возвращает:
        Нормализованный контур (N, 2).

    Исключения:
        ValueError: При некорректной форме points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен быть (N, 2), получено shape={pts.shape}"
        )
    if len(pts) == 0:
        return pts.copy()

    center = pts.mean(axis=0)
    centered = pts - center
    scale = np.abs(centered).max() + 1e-12
    return centered / scale


# ─── contour_area ─────────────────────────────────────────────────────────────

def contour_area(points: np.ndarray) -> float:
    """Площадь замкнутого контура (формула Гаусса / shoelace).

    Аргументы:
        points: Контур (N, 2), N >= 3.

    Возвращает:
        Площадь (>= 0).

    Исключения:
        ValueError: При N < 3.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError(
            f"Контур должен иметь >= 3 точек, получено {pts.shape[0]}"
        )
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# ─── contour_perimeter ────────────────────────────────────────────────────────

def contour_perimeter(points: np.ndarray) -> float:
    """Периметр замкнутого контура.

    Аргументы:
        points: Контур (N, 2), N >= 2.

    Возвращает:
        Периметр (>= 0).

    Исключения:
        ValueError: При N < 2.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError(
            f"Контур должен иметь >= 2 точек, получено {pts.shape[0]}"
        )
    shifted = np.roll(pts, -1, axis=0)
    return float(np.sum(np.linalg.norm(shifted - pts, axis=1)))


# ─── compute_contour_stats ────────────────────────────────────────────────────

def compute_contour_stats(points: np.ndarray) -> ContourStats:
    """Вычислить статистики контура.

    Аргументы:
        points: Контур (N, 2), N >= 3.

    Возвращает:
        ContourStats.
    """
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    perim = contour_perimeter(pts) if n >= 2 else 0.0
    area = contour_area(pts) if n >= 3 else 0.0
    compactness = (4.0 * np.pi * area / (perim ** 2 + 1e-12)
                   if perim > 1e-12 else 0.0)

    # Кривизна: приближение через угол между соседними отрезками
    if n >= 3:
        curvs = []
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            ab = b - a
            bc = c - b
            len_ab = np.linalg.norm(ab) + 1e-12
            len_bc = np.linalg.norm(bc) + 1e-12
            cos_a = np.clip(np.dot(ab / len_ab, bc / len_bc), -1.0, 1.0)
            curvs.append(float(np.arccos(cos_a)))
        mean_curv = float(np.mean(curvs))
    else:
        mean_curv = 0.0

    return ContourStats(n_points=n, perimeter=perim, area=area,
                        compactness=float(compactness),
                        mean_curvature=mean_curv)


# ─── process_contour ──────────────────────────────────────────────────────────

def process_contour(
    points: np.ndarray,
    fragment_id: int = 0,
    cfg: Optional[ContourConfig] = None,
) -> ContourResult:
    """Применить полный пайплайн обработки контура.

    Последовательность: ресемплинг → сглаживание → RDP → нормализация.

    Аргументы:
        points:      Исходный контур (N, 2).
        fragment_id: ID фрагмента.
        cfg:         Параметры (None → ContourConfig()).

    Возвращает:
        ContourResult.

    Исключения:
        ValueError: При некорректных входных данных.
    """
    if cfg is None:
        cfg = ContourConfig()

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен быть (N, 2), получено shape={pts.shape}"
        )

    # 1. Ресемплинг
    pts = resample_contour(pts, cfg.n_points)

    # 2. Сглаживание
    if cfg.smooth_sigma > 0.0:
        pts = smooth_contour(pts, cfg.smooth_sigma)

    # 3. Упрощение RDP
    simplified = False
    if cfg.rdp_epsilon > 0.0:
        pts_rdp = rdp_simplify(pts, cfg.rdp_epsilon)
        if len(pts_rdp) < len(pts):
            simplified = True
        pts = pts_rdp

    # 4. Нормализация
    if cfg.normalize:
        pts = normalize_contour(pts)

    stats = compute_contour_stats(pts)
    return ContourResult(points=pts, stats=stats,
                         fragment_id=fragment_id, simplified=simplified)


# ─── batch_process_contours ───────────────────────────────────────────────────

def batch_process_contours(
    contours: List[np.ndarray],
    cfg: Optional[ContourConfig] = None,
) -> List[ContourResult]:
    """Обработать список контуров.

    Аргументы:
        contours: Список контуров (N_i, 2).
        cfg:      Параметры.

    Возвращает:
        Список ContourResult.
    """
    if cfg is None:
        cfg = ContourConfig()
    return [process_contour(c, fid, cfg) for fid, c in enumerate(contours)]
