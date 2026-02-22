"""Дискретизация и выборка точек из контуров фрагментов.

Предоставляет несколько стратегий выборки точек вдоль замкнутых или открытых
контуров: равномерная по дуговой длине, взвешенная по кривизне, случайная и
по углам. Используется для подготовки точек к алгоритмам сопоставления
(DTW, ICP, Fourier-дескрипторы).

Экспортирует:
    SamplerConfig      — конфигурация выборки
    SampledContour     — результат выборки (точки + метаданные)
    sample_uniform     — равномерная выборка по дуговой длине
    sample_curvature   — взвешенная выборка по значению кривизны
    sample_random      — случайная выборка
    sample_corners     — выборка угловых точек
    sample_contour     — единая точка входа по стратегии
    normalize_contour  — нормировать контур в [-1, 1]
    batch_sample       — пакетная выборка
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─── SamplerConfig ────────────────────────────────────────────────────────────

@dataclass
class SamplerConfig:
    """Конфигурация выборки точек контура.

    Атрибуты:
        n_points:  Число точек в результирующей выборке (>= 2).
        strategy:  Стратегия выборки: ``'uniform'``, ``'curvature'``,
                   ``'random'`` или ``'corners'``.
        closed:    Если True, контур считается замкнутым.
        seed:      Начальное значение RNG (для воспроизводимости в random).
        corner_threshold: Порог кривизны для стратегии ``'corners'`` (>= 0).
    """
    n_points: int = 32
    strategy: str = "uniform"
    closed: bool = False
    seed: int = 0
    corner_threshold: float = 0.1

    _STRATEGIES = frozenset({"uniform", "curvature", "random", "corners"})

    def __post_init__(self) -> None:
        if self.n_points < 2:
            raise ValueError(
                f"n_points должен быть >= 2, получено {self.n_points}"
            )
        if self.strategy not in self._STRATEGIES:
            raise ValueError(
                f"strategy должна быть одной из {sorted(self._STRATEGIES)}, "
                f"получено {self.strategy!r}"
            )
        if self.corner_threshold < 0.0:
            raise ValueError(
                f"corner_threshold должен быть >= 0, получено {self.corner_threshold}"
            )


# ─── SampledContour ───────────────────────────────────────────────────────────

@dataclass
class SampledContour:
    """Результат выборки точек контура.

    Атрибуты:
        points:       Массив (n_points, 2) float64 — выбранные точки.
        indices:      Исходные индексы в контуре (или -1 для интерполированных).
        arc_lengths:  Накопленные дуговые длины для каждой точки.
        strategy:     Использованная стратегия.
        n_source:     Число точек в исходном контуре.
    """
    points: np.ndarray
    indices: np.ndarray
    arc_lengths: np.ndarray
    strategy: str
    n_source: int

    def __post_init__(self) -> None:
        if self.n_source < 0:
            raise ValueError(
                f"n_source должен быть >= 0, получено {self.n_source}"
            )
        self.points = np.asarray(self.points, dtype=np.float64)
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.arc_lengths = np.asarray(self.arc_lengths, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points должен быть (N, 2), получено {self.points.shape}"
            )

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def total_arc_length(self) -> float:
        if len(self.arc_lengths) == 0:
            return 0.0
        return float(self.arc_lengths[-1])


# ─── Внутренние утилиты ───────────────────────────────────────────────────────

def _validate_contour(contour: np.ndarray) -> np.ndarray:
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"contour должен быть (N, 2), получено {pts.shape}"
        )
    if len(pts) < 2:
        raise ValueError(
            f"contour должен содержать >= 2 точек, получено {len(pts)}"
        )
    return pts


def _arc_cumsum(pts: np.ndarray) -> np.ndarray:
    """Накопленная дуговая длина, начиная с 0."""
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _discrete_curvature(pts: np.ndarray) -> np.ndarray:
    """Абсолютная кривизна в каждой точке (граничные = 0)."""
    n = len(pts)
    curv = np.zeros(n)
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 < 1e-12 or l2 < 1e-12:
            continue
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        curv[i] = cross / (l1 * l2)
    return curv


# ─── sample_uniform ───────────────────────────────────────────────────────────

def sample_uniform(
    contour: np.ndarray,
    n_points: int = 32,
    closed: bool = False,
) -> SampledContour:
    """Равномерная выборка по дуговой длине.

    Точки расставляются равномерно вдоль контура с линейной интерполяцией.

    Args:
        contour:  (N, 2) массив float — исходный контур.
        n_points: Число выходных точек (>= 2).
        closed:   Если True, добавить первую точку в конец перед расчётом.

    Returns:
        :class:`SampledContour` с равномерно расставленными точками.

    Raises:
        ValueError: Если contour не (N, 2) или n_points < 2.
    """
    if n_points < 2:
        raise ValueError(f"n_points должен быть >= 2, получено {n_points}")
    pts = _validate_contour(contour)
    if closed:
        pts = np.vstack([pts, pts[0]])

    arc = _arc_cumsum(pts)
    total = arc[-1]
    if total < 1e-12:
        # Вырожденный контур: все точки совпадают
        sampled_pts = np.tile(pts[0], (n_points, 1))
        sampled_arc = np.zeros(n_points)
    else:
        t_new = np.linspace(0.0, total, n_points)
        xs = np.interp(t_new, arc, pts[:, 0])
        ys = np.interp(t_new, arc, pts[:, 1])
        sampled_pts = np.column_stack([xs, ys])
        sampled_arc = t_new

    return SampledContour(
        points=sampled_pts,
        indices=np.full(n_points, -1, dtype=np.int64),
        arc_lengths=sampled_arc,
        strategy="uniform",
        n_source=len(contour),
    )


# ─── sample_curvature ─────────────────────────────────────────────────────────

def sample_curvature(
    contour: np.ndarray,
    n_points: int = 32,
    closed: bool = False,
) -> SampledContour:
    """Выборка, взвешенная по значению кривизны.

    Точки с высокой кривизной (углы) семплируются чаще.
    Веса пропорциональны |кривизна| + ε.

    Args:
        contour:  (N, 2) массив float.
        n_points: Число выходных точек (>= 2).
        closed:   Замкнутый контур.

    Returns:
        :class:`SampledContour`.

    Raises:
        ValueError: Если contour некорректен или n_points < 2.
    """
    if n_points < 2:
        raise ValueError(f"n_points должен быть >= 2, получено {n_points}")
    pts = _validate_contour(contour)
    if closed:
        pts = np.vstack([pts, pts[0]])

    curv = _discrete_curvature(pts)
    weights = curv + 1e-6
    weights /= weights.sum()

    rng = np.random.default_rng(seed=0)
    chosen = np.sort(rng.choice(len(pts), size=n_points, replace=True, p=weights))

    arc = _arc_cumsum(pts)

    return SampledContour(
        points=pts[chosen],
        indices=chosen.astype(np.int64),
        arc_lengths=arc[chosen],
        strategy="curvature",
        n_source=len(contour),
    )


# ─── sample_random ────────────────────────────────────────────────────────────

def sample_random(
    contour: np.ndarray,
    n_points: int = 32,
    seed: int = 0,
    closed: bool = False,
) -> SampledContour:
    """Случайная выборка индексов без повторений (с заменой, если N < n_points).

    Args:
        contour:  (N, 2) массив float.
        n_points: Число выходных точек (>= 2).
        seed:     Начальное значение RNG.
        closed:   Замкнутый контур.

    Returns:
        :class:`SampledContour`.

    Raises:
        ValueError: Если contour некорректен или n_points < 2.
    """
    if n_points < 2:
        raise ValueError(f"n_points должен быть >= 2, получено {n_points}")
    pts = _validate_contour(contour)
    if closed:
        pts = np.vstack([pts, pts[0]])

    rng = np.random.default_rng(seed=seed)
    replace = len(pts) < n_points
    chosen = np.sort(rng.choice(len(pts), size=n_points, replace=replace))

    arc = _arc_cumsum(pts)

    return SampledContour(
        points=pts[chosen],
        indices=chosen.astype(np.int64),
        arc_lengths=arc[chosen],
        strategy="random",
        n_source=len(contour),
    )


# ─── sample_corners ───────────────────────────────────────────────────────────

def sample_corners(
    contour: np.ndarray,
    n_points: int = 32,
    corner_threshold: float = 0.1,
    closed: bool = False,
) -> SampledContour:
    """Выборка угловых точек с дополнением равномерными при нехватке.

    Сначала отбираются точки с кривизной >= corner_threshold, затем
    при необходимости дополняются равномерно расставленными точками.

    Args:
        contour:          (N, 2) массив float.
        n_points:         Итоговое число точек (>= 2).
        corner_threshold: Минимальная кривизна для угловой точки (>= 0).
        closed:           Замкнутый контур.

    Returns:
        :class:`SampledContour`.

    Raises:
        ValueError: Если contour некорректен, n_points < 2
                    или corner_threshold < 0.
    """
    if n_points < 2:
        raise ValueError(f"n_points должен быть >= 2, получено {n_points}")
    if corner_threshold < 0.0:
        raise ValueError(
            f"corner_threshold должен быть >= 0, получено {corner_threshold}"
        )
    pts = _validate_contour(contour)
    if closed:
        pts = np.vstack([pts, pts[0]])

    curv = _discrete_curvature(pts)
    corner_idx = np.where(curv >= corner_threshold)[0]

    arc = _arc_cumsum(pts)
    total = arc[-1]

    if len(corner_idx) >= n_points:
        # Слишком много углов — выбрать с наибольшей кривизной
        sorted_by_curv = corner_idx[np.argsort(curv[corner_idx])[::-1]]
        chosen = np.sort(sorted_by_curv[:n_points])
    elif len(corner_idx) > 0:
        # Дополнить равномерными точками
        n_extra = n_points - len(corner_idx)
        if total < 1e-12:
            extra_idx = np.zeros(n_extra, dtype=np.int64)
        else:
            t_extra = np.linspace(0.0, total, n_extra + 2)[1:-1]
            extra_idx = np.searchsorted(arc, t_extra).clip(0, len(pts) - 1)
        chosen = np.sort(np.concatenate([corner_idx, extra_idx]))
    else:
        # Нет углов — равномерная выборка
        if total < 1e-12:
            chosen = np.zeros(n_points, dtype=np.int64)
        else:
            t_unif = np.linspace(0.0, total, n_points)
            chosen = np.searchsorted(arc, t_unif).clip(0, len(pts) - 1)

    chosen = chosen.astype(np.int64)

    return SampledContour(
        points=pts[chosen],
        indices=chosen,
        arc_lengths=arc[chosen],
        strategy="corners",
        n_source=len(contour),
    )


# ─── sample_contour ───────────────────────────────────────────────────────────

def sample_contour(
    contour: np.ndarray,
    cfg: Optional[SamplerConfig] = None,
) -> SampledContour:
    """Единая точка входа: выбрать стратегию из cfg и вернуть выборку.

    Args:
        contour: (N, 2) массив float.
        cfg:     Конфигурация (None → SamplerConfig()).

    Returns:
        :class:`SampledContour`.

    Raises:
        ValueError: Если contour некорректен или стратегия неизвестна.
    """
    if cfg is None:
        cfg = SamplerConfig()

    if cfg.strategy == "uniform":
        return sample_uniform(contour, n_points=cfg.n_points, closed=cfg.closed)
    if cfg.strategy == "curvature":
        return sample_curvature(contour, n_points=cfg.n_points, closed=cfg.closed)
    if cfg.strategy == "random":
        return sample_random(
            contour, n_points=cfg.n_points, seed=cfg.seed, closed=cfg.closed
        )
    if cfg.strategy == "corners":
        return sample_corners(
            contour,
            n_points=cfg.n_points,
            corner_threshold=cfg.corner_threshold,
            closed=cfg.closed,
        )
    raise ValueError(f"Неизвестная стратегия: {cfg.strategy!r}")


# ─── normalize_contour ────────────────────────────────────────────────────────

def normalize_contour(contour: np.ndarray) -> np.ndarray:
    """Нормализовать контур в [-1, 1] по каждой оси.

    Вычитает centroid, делит на полуразмах (max abs по обеим осям).
    Вырожденный контур (все точки совпадают) возвращается без изменений.

    Args:
        contour: (N, 2) массив float.

    Returns:
        Нормализованный контур (N, 2) float64.

    Raises:
        ValueError: Если contour не (N, 2) или содержит < 2 точек.
    """
    pts = _validate_contour(contour)
    center = pts.mean(axis=0)
    shifted = pts - center
    scale = np.abs(shifted).max()
    if scale < 1e-12:
        return shifted
    return shifted / scale


# ─── batch_sample ─────────────────────────────────────────────────────────────

def batch_sample(
    contours: List[np.ndarray],
    cfg: Optional[SamplerConfig] = None,
) -> List[SampledContour]:
    """Пакетная выборка для списка контуров.

    Args:
        contours: Список контуров [(N_i, 2) float].
        cfg:      Конфигурация (None → SamplerConfig()).

    Returns:
        Список :class:`SampledContour` той же длины.
    """
    if cfg is None:
        cfg = SamplerConfig()
    return [sample_contour(c, cfg) for c in contours]
