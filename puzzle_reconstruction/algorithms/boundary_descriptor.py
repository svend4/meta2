"""Дескрипторы граничных сегментов фрагментов пазла.

Модуль вычисляет компактные дескрипторы граничных кривых на основе
кривизны, распределения направлений и хорд — для последующего
сопоставления краёв фрагментов.

Публичный API:
    DescriptorConfig   — параметры вычисления дескриптора
    BoundaryDescriptor — совокупный дескриптор граничного сегмента
    compute_curvature  — кривизна в каждой точке цепочки
    curvature_histogram — гистограмма кривизны
    direction_histogram — гистограмма направлений касательных
    chord_distribution  — гистограмма длин хорд
    extract_descriptor  — извлечение BoundaryDescriptor из массива точек
    descriptor_similarity — схожесть двух дескрипторов [0, 1]
    batch_extract_descriptors — пакетное извлечение
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─── DescriptorConfig ─────────────────────────────────────────────────────────

@dataclass
class DescriptorConfig:
    """Параметры вычисления граничного дескриптора.

    Атрибуты:
        n_bins:       Число бинов в гистограммах (>= 4).
        smooth_sigma: Сигма сглаживания точек перед кривизной (>= 0).
        normalize:    Нормировать гистограммы (L1-норма = 1).
        max_chord:    Максимальная хорда для гистограммы (> 0; None = auto).
    """

    n_bins: int = 32
    smooth_sigma: float = 1.0
    normalize: bool = True
    max_chord: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_bins < 4:
            raise ValueError(
                f"n_bins должен быть >= 4, получено {self.n_bins}"
            )
        if self.smooth_sigma < 0.0:
            raise ValueError(
                f"smooth_sigma должен быть >= 0, получено {self.smooth_sigma}"
            )
        if self.max_chord is not None and self.max_chord <= 0.0:
            raise ValueError(
                f"max_chord должен быть > 0, получено {self.max_chord}"
            )


# ─── BoundaryDescriptor ───────────────────────────────────────────────────────

@dataclass
class BoundaryDescriptor:
    """Совокупный дескриптор граничного сегмента.

    Атрибуты:
        fragment_id:     Идентификатор фрагмента (>= 0).
        edge_id:         Идентификатор края (>= 0).
        curvature_hist:  Гистограмма кривизны (1-D float array).
        direction_hist:  Гистограмма направлений касательных.
        chord_hist:      Гистограмма длин хорд.
        length:          Длина дуги сегмента (>= 0).
    """

    fragment_id: int
    edge_id: int
    curvature_hist: np.ndarray
    direction_hist: np.ndarray
    chord_hist: np.ndarray
    length: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.edge_id < 0:
            raise ValueError(
                f"edge_id должен быть >= 0, получено {self.edge_id}"
            )
        for name, arr in (
            ("curvature_hist", self.curvature_hist),
            ("direction_hist", self.direction_hist),
            ("chord_hist", self.chord_hist),
        ):
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(
                    f"{name} должен быть одномерным numpy-массивом"
                )
        if self.length < 0.0:
            raise ValueError(
                f"length должен быть >= 0, получено {self.length}"
            )

    @property
    def n_bins(self) -> int:
        """Число бинов в каждой гистограмме."""
        return len(self.curvature_hist)

    @property
    def feature_vector(self) -> np.ndarray:
        """Конкатенация трёх гистограмм в единый вектор признаков."""
        return np.concatenate([
            self.curvature_hist,
            self.direction_hist,
            self.chord_hist,
        ]).astype(np.float32)


# ─── compute_curvature ────────────────────────────────────────────────────────

def compute_curvature(
    points: np.ndarray,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Вычислить кривизну в каждой точке цепочки.

    Аргументы:
        points:       Массив точек формы (N, 2), N >= 3.
        smooth_sigma: Сигма сглаживания (>= 0).

    Возвращает:
        Массив кривизн формы (N,) (знаковая кривизна).

    Исключения:
        ValueError: Если points.shape не (N, 2) или N < 3.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен иметь форму (N, 2), получено {pts.shape}"
        )
    n = pts.shape[0]
    if n < 3:
        raise ValueError(
            f"Требуется минимум 3 точки, получено {n}"
        )

    # Опциональное сглаживание по каждой координате
    if smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
        pts = np.stack([
            gaussian_filter1d(pts[:, 0], smooth_sigma),
            gaussian_filter1d(pts[:, 1], smooth_sigma),
        ], axis=1)

    # Производные первого и второго порядка (центральные разности)
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-12
    curvature = (dx * ddy - dy * ddx) / denom
    return curvature.astype(np.float32)


# ─── curvature_histogram ──────────────────────────────────────────────────────

def curvature_histogram(
    curvature: np.ndarray,
    n_bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Гистограмма кривизны.

    Аргументы:
        curvature: Массив кривизн (1-D).
        n_bins:    Число бинов (>= 4).
        normalize: Нормировать L1.

    Возвращает:
        Гистограмма формы (n_bins,).
    """
    if n_bins < 4:
        raise ValueError(f"n_bins >= 4, получено {n_bins}")
    c = np.asarray(curvature, dtype=np.float64)
    if c.size == 0:
        return np.zeros(n_bins, dtype=np.float32)

    # Симметричный диапазон по максимальному |кривизна|
    vmax = max(np.abs(c).max(), 1e-6)
    hist, _ = np.histogram(c, bins=n_bins, range=(-vmax, vmax))
    hist = hist.astype(np.float32)
    if normalize:
        s = hist.sum()
        if s > 0:
            hist /= s
    return hist


# ─── direction_histogram ──────────────────────────────────────────────────────

def direction_histogram(
    points: np.ndarray,
    n_bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Гистограмма направлений касательных.

    Аргументы:
        points:    Массив точек (N, 2).
        n_bins:    Число бинов (>= 4).
        normalize: Нормировать L1.

    Возвращает:
        Гистограмма формы (n_bins,).
    """
    if n_bins < 4:
        raise ValueError(f"n_bins >= 4, получено {n_bins}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.zeros(n_bins, dtype=np.float32)

    diff = np.diff(pts, axis=0)
    angles = np.arctan2(diff[:, 1], diff[:, 0])  # [-π, π]
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32)
    if normalize:
        s = hist.sum()
        if s > 0:
            hist /= s
    return hist


# ─── chord_distribution ───────────────────────────────────────────────────────

def chord_distribution(
    points: np.ndarray,
    n_bins: int = 32,
    max_chord: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Гистограмма длин хорд (расстояний между всеми парами точек).

    Для больших контуров используется выборка равномерных пар.

    Аргументы:
        points:    Массив точек (N, 2).
        n_bins:    Число бинов (>= 4).
        max_chord: Верхняя граница диапазона (None = max расстояние).
        normalize: Нормировать L1.

    Возвращает:
        Гистограмма формы (n_bins,).
    """
    if n_bins < 4:
        raise ValueError(f"n_bins >= 4, получено {n_bins}")
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return np.zeros(n_bins, dtype=np.float32)

    # Ограничить до 200 точек для производительности
    if n > 200:
        idx = np.linspace(0, n - 1, 200, dtype=int)
        pts = pts[idx]
        n = 200

    # Все попарные расстояния (верхний треугольник)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    upper = dists[np.triu_indices(n, k=1)]

    if max_chord is None:
        max_chord = float(upper.max()) if upper.size > 0 else 1.0
    max_chord = max(max_chord, 1e-6)

    hist, _ = np.histogram(upper, bins=n_bins, range=(0.0, max_chord))
    hist = hist.astype(np.float32)
    if normalize:
        s = hist.sum()
        if s > 0:
            hist /= s
    return hist


# ─── extract_descriptor ───────────────────────────────────────────────────────

def extract_descriptor(
    points: np.ndarray,
    fragment_id: int = 0,
    edge_id: int = 0,
    cfg: Optional[DescriptorConfig] = None,
) -> BoundaryDescriptor:
    """Извлечь BoundaryDescriptor из массива точек края.

    Аргументы:
        points:      Массив точек (N, 2); N >= 3.
        fragment_id: Идентификатор фрагмента (>= 0).
        edge_id:     Идентификатор края (>= 0).
        cfg:         Параметры (по умолчанию DescriptorConfig()).

    Возвращает:
        BoundaryDescriptor.

    Исключения:
        ValueError: Если points имеет неверную форму или < 3 точек.
    """
    if cfg is None:
        cfg = DescriptorConfig()

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points должен иметь форму (N, 2), получено {pts.shape}"
        )
    if pts.shape[0] < 3:
        raise ValueError(
            f"Требуется минимум 3 точки, получено {pts.shape[0]}"
        )

    # Длина дуги
    diffs = np.diff(pts, axis=0)
    length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

    curv = compute_curvature(pts, smooth_sigma=cfg.smooth_sigma)
    c_hist = curvature_histogram(curv, n_bins=cfg.n_bins,
                                  normalize=cfg.normalize)
    d_hist = direction_histogram(pts, n_bins=cfg.n_bins,
                                  normalize=cfg.normalize)
    ch_hist = chord_distribution(pts, n_bins=cfg.n_bins,
                                  max_chord=cfg.max_chord,
                                  normalize=cfg.normalize)

    return BoundaryDescriptor(
        fragment_id=fragment_id,
        edge_id=edge_id,
        curvature_hist=c_hist,
        direction_hist=d_hist,
        chord_hist=ch_hist,
        length=length,
    )


# ─── descriptor_similarity ────────────────────────────────────────────────────

def descriptor_similarity(
    desc_a: BoundaryDescriptor,
    desc_b: BoundaryDescriptor,
    w_curvature: float = 0.4,
    w_direction: float = 0.4,
    w_chord: float = 0.2,
) -> float:
    """Схожесть двух граничных дескрипторов [0, 1].

    Использует гистограммное пересечение для каждой компоненты.

    Аргументы:
        desc_a:      Первый дескриптор.
        desc_b:      Второй дескриптор.
        w_curvature: Вес кривизны (>= 0).
        w_direction: Вес направлений (>= 0).
        w_chord:     Вес хорд (>= 0).

    Возвращает:
        Схожесть в диапазоне [0, 1].

    Исключения:
        ValueError: Если дескрипторы имеют разное n_bins.
    """
    if desc_a.n_bins != desc_b.n_bins:
        raise ValueError(
            f"n_bins дескрипторов не совпадают: "
            f"{desc_a.n_bins} != {desc_b.n_bins}"
        )

    def _intersection(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(np.minimum(h1, h2).sum())

    sim_c = _intersection(desc_a.curvature_hist, desc_b.curvature_hist)
    sim_d = _intersection(desc_a.direction_hist, desc_b.direction_hist)
    sim_ch = _intersection(desc_a.chord_hist, desc_b.chord_hist)

    w_sum = w_curvature + w_direction + w_chord + 1e-12
    sim = (w_curvature * sim_c + w_direction * sim_d + w_chord * sim_ch) / w_sum
    return float(np.clip(sim, 0.0, 1.0))


# ─── batch_extract_descriptors ────────────────────────────────────────────────

def batch_extract_descriptors(
    points_list: List[np.ndarray],
    fragment_id: int = 0,
    cfg: Optional[DescriptorConfig] = None,
) -> List[BoundaryDescriptor]:
    """Пакетно извлечь дескрипторы из списка массивов точек.

    Аргументы:
        points_list: Список массивов точек (N_i, 2).
        fragment_id: Идентификатор фрагмента (>= 0).
        cfg:         Параметры.

    Возвращает:
        Список BoundaryDescriptor (edge_id = индекс в списке).
    """
    return [
        extract_descriptor(pts, fragment_id=fragment_id,
                           edge_id=i, cfg=cfg)
        for i, pts in enumerate(points_list)
    ]
