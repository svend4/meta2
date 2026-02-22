"""
Компактные дескрипторы кривых для сопоставления краёв фрагментов.

Вычисляет инвариантные к вращению и масштабу дескрипторы контурных кривых
на основе дискретного преобразования Фурье (DFT) и профиля кривизны.
Используется для быстрого предварительного отбора кандидатов перед более
дорогостоящими методами (DTW, ICP).

Классы:
    CurveDescriptorConfig — параметры вычисления дескриптора
    CurveDescriptor       — компактный дескриптор одной кривой

Функции:
    compute_fourier_descriptor  — DFT-дескриптор кривой (амплитудный спектр)
    compute_curvature_profile   — профиль дискретной кривизны
    describe_curve              — единая точка входа: CurveDescriptor
    descriptor_distance         — евклидово расстояние между дескрипторами
    descriptor_similarity       — сходство ∈ [0, 1]
    batch_describe_curves       — пакетное вычисление дескрипторов
    find_best_match             — индекс ближайшего дескриптора
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── CurveDescriptorConfig ────────────────────────────────────────────────────

@dataclass
class CurveDescriptorConfig:
    """Параметры вычисления дескриптора кривой.

    Атрибуты:
        n_harmonics:  Число гармоник Фурье (>= 1). Чем больше — тем точнее,
                      но медленнее сопоставление.
        normalize:    Нормализовать амплитуды на первую гармонику
                      (масштабная инвариантность).
        center:       Вычитать нулевую частоту (центрирование).
        resample_n:   Число точек для передискретизации кривой перед DFT.
                      None → использовать исходное число точек.
    """
    n_harmonics: int  = 8
    normalize:   bool = True
    center:      bool = True
    resample_n:  Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_harmonics < 1:
            raise ValueError(
                f"n_harmonics must be >= 1, got {self.n_harmonics}"
            )
        if self.resample_n is not None and self.resample_n < 2:
            raise ValueError(
                f"resample_n must be >= 2 or None, got {self.resample_n}"
            )


# ─── CurveDescriptor ──────────────────────────────────────────────────────────

@dataclass
class CurveDescriptor:
    """Компактный дескриптор одной кривой.

    Атрибуты:
        fourier_desc:    Амплитудный Фурье-дескриптор формы (n_harmonics,).
        arc_length:      Суммарная длина дуги кривой.
        curvature_mean:  Среднее абсолютное значение кривизны.
        curvature_std:   Стандартное отклонение абсолютной кривизны.
        n_points:        Число точек исходной кривой.
    """
    fourier_desc:   np.ndarray
    arc_length:     float
    curvature_mean: float
    curvature_std:  float
    n_points:       int

    def __post_init__(self) -> None:
        if self.arc_length < 0.0:
            raise ValueError(
                f"arc_length must be >= 0, got {self.arc_length}"
            )
        if self.n_points < 0:
            raise ValueError(
                f"n_points must be >= 0, got {self.n_points}"
            )

    def __repr__(self) -> str:
        return (
            f"CurveDescriptor(n_harmonics={len(self.fourier_desc)}, "
            f"arc_length={self.arc_length:.2f}, "
            f"n_points={self.n_points})"
        )


# ─── compute_fourier_descriptor ───────────────────────────────────────────────

def compute_fourier_descriptor(
    curve: np.ndarray,
    n_harmonics: int = 8,
    normalize: bool = True,
    center: bool = True,
) -> np.ndarray:
    """Вычисляет DFT-дескриптор кривой.

    Представляет кривую как комплексный сигнал z[n] = x[n] + j·y[n],
    вычисляет DFT и возвращает амплитуды n_harmonics гармоник (без нулевой).

    Args:
        curve:       (N, 2) массив float — координаты точек кривой.
        n_harmonics: Число возвращаемых гармоник.
        normalize:   True → делить на |F[1]| (инвариантность к масштабу).
        center:      True → вычитать нулевую частоту (инвариантность к сдвигу).

    Returns:
        np.ndarray формы (n_harmonics,) float64 — амплитудный спектр.

    Raises:
        ValueError: Если curve не двумерный (N, 2) массив.
    """
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"curve must have shape (N, 2), got {pts.shape}"
        )
    if len(pts) == 0:
        return np.zeros(n_harmonics, dtype=np.float64)

    # Комплексное представление
    z = pts[:, 0] + 1j * pts[:, 1]

    if center:
        z = z - z.mean()

    # DFT
    Z = np.fft.fft(z)
    amplitudes = np.abs(Z)

    # Амплитуды гармоник 1..n_harmonics (пропускаем нулевую)
    n = len(amplitudes)
    n_take = min(n_harmonics, n - 1)
    desc = amplitudes[1 : n_take + 1]

    # Дополняем нулями, если кривая короче n_harmonics + 1 точек
    if len(desc) < n_harmonics:
        desc = np.pad(desc, (0, n_harmonics - len(desc)))

    # Нормализация на первую гармонику (масштабная инвариантность)
    if normalize and desc[0] > 1e-9:
        desc = desc / desc[0]

    return desc.astype(np.float64)


# ─── compute_curvature_profile ────────────────────────────────────────────────

def compute_curvature_profile(curve: np.ndarray) -> np.ndarray:
    """Вычисляет профиль дискретной кривизны.

    Кривизна в точке i аппроксимируется синусом угла между соседними
    отрезками (знаковая кривизна).

    Args:
        curve: (N, 2) массив float — координаты точек кривой.

    Returns:
        np.ndarray формы (N,) float64 — кривизна в каждой точке.
        Крайние точки имеют нулевую кривизну.

    Raises:
        ValueError: Если curve не (N, 2).
    """
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"curve must have shape (N, 2), got {pts.shape}"
        )
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    curv = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 < 1e-12 or l2 < 1e-12:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]   # знаковая площадь
        curv[i] = cross / (l1 * l2)

    return curv


# ─── describe_curve ───────────────────────────────────────────────────────────

def describe_curve(
    curve: np.ndarray,
    cfg: Optional[CurveDescriptorConfig] = None,
) -> CurveDescriptor:
    """Единая точка входа: вычислить полный дескриптор кривой.

    Args:
        curve: (N, 2) массив float — координаты точек кривой.
        cfg:   Конфигурация (None → CurveDescriptorConfig() с умолчаниями).

    Returns:
        CurveDescriptor.

    Raises:
        ValueError: Если curve имеет неверную форму.
    """
    if cfg is None:
        cfg = CurveDescriptorConfig()

    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"curve must have shape (N, 2), got {pts.shape}"
        )

    n_pts = len(pts)

    # Передискретизация
    if cfg.resample_n is not None and n_pts >= 2:
        pts = _resample_curve(pts, cfg.resample_n)

    # Длина дуги
    if len(pts) >= 2:
        diffs = np.diff(pts, axis=0)
        arc_len = float(np.linalg.norm(diffs, axis=1).sum())
    else:
        arc_len = 0.0

    # Кривизна
    curv = compute_curvature_profile(pts)
    abs_curv = np.abs(curv)
    curv_mean = float(abs_curv.mean()) if len(abs_curv) > 0 else 0.0
    curv_std  = float(abs_curv.std())  if len(abs_curv) > 0 else 0.0

    # Фурье-дескриптор
    fd = compute_fourier_descriptor(
        pts,
        n_harmonics=cfg.n_harmonics,
        normalize=cfg.normalize,
        center=cfg.center,
    )

    return CurveDescriptor(
        fourier_desc=fd,
        arc_length=arc_len,
        curvature_mean=curv_mean,
        curvature_std=curv_std,
        n_points=n_pts,
    )


# ─── descriptor_distance ──────────────────────────────────────────────────────

def descriptor_distance(
    d1: CurveDescriptor,
    d2: CurveDescriptor,
) -> float:
    """Евклидово расстояние между двумя Фурье-дескрипторами.

    Сравниваются только fourier_desc (усечение до min длины).

    Args:
        d1: Первый CurveDescriptor.
        d2: Второй CurveDescriptor.

    Returns:
        Расстояние ∈ [0, +∞).
    """
    v1 = d1.fourier_desc
    v2 = d2.fourier_desc
    n  = min(len(v1), len(v2))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(v1[:n] - v2[:n]))


# ─── descriptor_similarity ────────────────────────────────────────────────────

def descriptor_similarity(
    d1: CurveDescriptor,
    d2: CurveDescriptor,
    sigma: float = 1.0,
) -> float:
    """Сходство между двумя дескрипторами ∈ [0, 1].

    Использует гауссово убывание: sim = exp(-dist² / (2σ²)).

    Args:
        d1:    Первый CurveDescriptor.
        d2:    Второй CurveDescriptor.
        sigma: Ширина гауссовой функции (> 0).

    Returns:
        Сходство ∈ [0, 1].

    Raises:
        ValueError: Если sigma <= 0.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    dist = descriptor_distance(d1, d2)
    return float(np.exp(-dist ** 2 / (2.0 * sigma ** 2)))


# ─── batch_describe_curves ────────────────────────────────────────────────────

def batch_describe_curves(
    curves: List[np.ndarray],
    cfg: Optional[CurveDescriptorConfig] = None,
) -> List[CurveDescriptor]:
    """Пакетное вычисление дескрипторов.

    Args:
        curves: Список кривых [(N_i, 2) float].
        cfg:    Конфигурация (None → умолчания).

    Returns:
        Список CurveDescriptor той же длины.
    """
    if cfg is None:
        cfg = CurveDescriptorConfig()
    return [describe_curve(c, cfg) for c in curves]


# ─── find_best_match ──────────────────────────────────────────────────────────

def find_best_match(
    query: CurveDescriptor,
    candidates: List[CurveDescriptor],
) -> Tuple[int, float]:
    """Находит индекс ближайшего дескриптора из списка.

    Args:
        query:      Эталонный дескриптор.
        candidates: Список дескрипторов-кандидатов (не пустой).

    Returns:
        (idx, distance) — индекс ближайшего и расстояние до него.

    Raises:
        ValueError: Если candidates пуст.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_idx  = 0
    best_dist = descriptor_distance(query, candidates[0])
    for i, c in enumerate(candidates[1:], start=1):
        d = descriptor_distance(query, c)
        if d < best_dist:
            best_dist = d
            best_idx  = i

    return (best_idx, best_dist)


# ─── Внутренние утилиты ───────────────────────────────────────────────────────

def _resample_curve(pts: np.ndarray, n: int) -> np.ndarray:
    """Передискретизирует кривую до n равноотстоящих точек по длине дуги."""
    dists = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(pts, axis=0), axis=1)
    )])
    total = dists[-1]
    if total < 1e-12:
        return np.tile(pts[0], (n, 1))
    t_new = np.linspace(0.0, total, n)
    xs = np.interp(t_new, dists, pts[:, 0])
    ys = np.interp(t_new, dists, pts[:, 1])
    return np.column_stack([xs, ys])
