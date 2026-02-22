"""
Утилиты кривизны контуров и кривых.

Используются для анализа формы краёв фрагментов: вычисления кривизны,
нахождения угловых точек, точек перегиба и агрегации кривизны.

Экспортирует:
    CurvatureConfig        — параметры вычисления кривизны
    compute_curvature      — кривизна κ(t) кривой (N, 2)
    compute_total_curvature — суммарная абсолютная кривизна
    find_inflection_points — точки смены знака кривизны
    compute_turning_angle  — накопленный угол поворота касательной
    smooth_curvature       — сглаживание профиля кривизны (Гаусс)
    corner_score           — оценка «угловатости» каждой точки
    find_corners           — поиск угловых точек по порогу кривизны
    batch_curvature        — вычисление кривизны для набора кривых
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─── CurvatureConfig ──────────────────────────────────────────────────────────

@dataclass
class CurvatureConfig:
    """Параметры вычисления кривизны.

    Attributes:
        smooth_sigma:   Стандартное отклонение сглаживания координат перед
                        дифференцированием (> 0, иначе сглаживание не применяется).
        corner_threshold: Порог кривизны для определения угловых точек (> 0).
        min_distance:   Минимальное расстояние между угловыми точками (>= 1).
    """
    smooth_sigma:     float = 1.0
    corner_threshold: float = 0.1
    min_distance:     int   = 3

    def __post_init__(self) -> None:
        if self.corner_threshold <= 0:
            raise ValueError(
                f"corner_threshold must be > 0, got {self.corner_threshold}"
            )
        if self.min_distance < 1:
            raise ValueError(
                f"min_distance must be >= 1, got {self.min_distance}"
            )


# ─── compute_curvature ────────────────────────────────────────────────────────

def compute_curvature(
    curve: np.ndarray,
    cfg: CurvatureConfig | None = None,
) -> np.ndarray:
    """Вычислить знаковую кривизну κ(t) для параметрической кривой.

    Использует формулу: κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2)

    Args:
        curve: Массив (N, 2) — точки кривой.
        cfg:   Параметры (None → CurvatureConfig()).

    Returns:
        Массив float64 (N,) — кривизна в каждой точке.
        Для граничных точек кривизна вычисляется с односторонними разностями.

    Raises:
        ValueError: Если curve не (N, 2) или N < 3.
    """
    if cfg is None:
        cfg = CurvatureConfig()
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"curve must be (N, 2), got shape {pts.shape}")
    if len(pts) < 3:
        raise ValueError(f"curve must have >= 3 points, got {len(pts)}")

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    if cfg.smooth_sigma > 0:
        x = gaussian_filter1d(x, sigma=cfg.smooth_sigma)
        y = gaussian_filter1d(y, sigma=cfg.smooth_sigma)

    dx  = np.gradient(x)
    dy  = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-12, 1e-12, denom)

    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


# ─── compute_total_curvature ──────────────────────────────────────────────────

def compute_total_curvature(
    curve: np.ndarray,
    cfg: CurvatureConfig | None = None,
) -> float:
    """Вычислить суммарную абсолютную кривизну: ∫|κ| dt.

    Args:
        curve: Массив (N, 2).
        cfg:   Параметры.

    Returns:
        Неотрицательное float.

    Raises:
        ValueError: Если curve не (N, 2) или N < 3.
    """
    kappa = compute_curvature(curve, cfg)
    return float(np.sum(np.abs(kappa)))


# ─── find_inflection_points ───────────────────────────────────────────────────

def find_inflection_points(
    curve: np.ndarray,
    cfg: CurvatureConfig | None = None,
) -> np.ndarray:
    """Найти индексы точек перегиба (смена знака кривизны).

    Args:
        curve: Массив (N, 2).
        cfg:   Параметры.

    Returns:
        Массив int64 индексов точек перегиба.

    Raises:
        ValueError: Если curve не (N, 2) или N < 3.
    """
    kappa = compute_curvature(curve, cfg)
    signs = np.sign(kappa)
    inflections = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i - 1] != 0 and signs[i] != 0:
            inflections.append(i)
    return np.array(inflections, dtype=np.int64)


# ─── compute_turning_angle ────────────────────────────────────────────────────

def compute_turning_angle(curve: np.ndarray) -> float:
    """Вычислить суммарный угол поворота касательной (радианы).

    Суммирует угловые изменения вектора касательной между соседними сегментами.
    Для замкнутой окружности равен ±2π.

    Args:
        curve: Массив (N, 2) с N >= 2.

    Returns:
        float — суммарный угол поворота.

    Raises:
        ValueError: Если curve не (N, 2) или N < 2.
    """
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"curve must be (N, 2), got shape {pts.shape}")
    if len(pts) < 2:
        raise ValueError(f"curve must have >= 2 points, got {len(pts)}")

    tangents = np.diff(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    tangents = tangents / norms

    total = 0.0
    for i in range(len(tangents) - 1):
        cross = tangents[i, 0] * tangents[i + 1, 1] - tangents[i, 1] * tangents[i + 1, 0]
        dot   = np.clip(np.dot(tangents[i], tangents[i + 1]), -1.0, 1.0)
        angle = float(np.arctan2(cross, dot))
        total += angle
    return total


# ─── smooth_curvature ─────────────────────────────────────────────────────────

def smooth_curvature(
    kappa: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Сгладить профиль кривизны гауссовым фильтром.

    Args:
        kappa: Массив float (N,) — кривизна.
        sigma: Параметр Гаусса (> 0).

    Returns:
        Сглаженный массив float64 той же длины.

    Raises:
        ValueError: Если sigma <= 0 или kappa не 1-D.
    """
    k = np.asarray(kappa, dtype=np.float64)
    if k.ndim != 1:
        raise ValueError(f"kappa must be 1-D, got ndim={k.ndim}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return gaussian_filter1d(k, sigma=sigma)


# ─── corner_score ─────────────────────────────────────────────────────────────

def corner_score(
    curve: np.ndarray,
    cfg: CurvatureConfig | None = None,
) -> np.ndarray:
    """Вычислить оценку угловатости каждой точки кривой.

    Оценка = |κ(t)| нормированная на [0, 1].
    Значение 1.0 соответствует максимальной кривизне на данной кривой.

    Args:
        curve: Массив (N, 2).
        cfg:   Параметры.

    Returns:
        Массив float64 (N,) ∈ [0, 1].

    Raises:
        ValueError: Если curve не (N, 2) или N < 3.
    """
    kappa = compute_curvature(curve, cfg)
    abs_k = np.abs(kappa)
    max_k = abs_k.max()
    if max_k < 1e-12:
        return np.zeros_like(abs_k)
    return abs_k / max_k


# ─── find_corners ─────────────────────────────────────────────────────────────

def find_corners(
    curve: np.ndarray,
    cfg: CurvatureConfig | None = None,
) -> np.ndarray:
    """Найти угловые точки кривой по порогу кривизны.

    Args:
        curve: Массив (N, 2).
        cfg:   Параметры (corner_threshold, min_distance).

    Returns:
        Массив int64 индексов угловых точек.

    Raises:
        ValueError: Если curve не (N, 2) или N < 3.
    """
    if cfg is None:
        cfg = CurvatureConfig()

    kappa = compute_curvature(curve, cfg)
    abs_k = np.abs(kappa)
    candidates = np.where(abs_k >= cfg.corner_threshold)[0]

    if len(candidates) == 0 or cfg.min_distance <= 1:
        return candidates.astype(np.int64)

    selected: List[int] = [int(candidates[0])]
    for idx in candidates[1:]:
        if int(idx) - selected[-1] >= cfg.min_distance:
            selected.append(int(idx))
    return np.array(selected, dtype=np.int64)


# ─── batch_curvature ──────────────────────────────────────────────────────────

def batch_curvature(
    curves: List[np.ndarray],
    cfg: CurvatureConfig | None = None,
) -> List[np.ndarray]:
    """Вычислить кривизну для набора кривых.

    Args:
        curves: Список массивов (N_i, 2).
        cfg:    Параметры.

    Returns:
        Список массивов float64 (N_i,).

    Raises:
        ValueError: Если curves пуст.
    """
    if not curves:
        raise ValueError("curves must not be empty")
    return [compute_curvature(c, cfg) for c in curves]
