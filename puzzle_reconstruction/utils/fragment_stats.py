"""Статистические сводки фрагментов и сборок документа.

Модуль вычисляет агрегированные метрики по коллекции фрагментов:
распределение площадей, аспектных отношений, плотностей, а также
сравнительные сводки между несколькими сборками.

Публичный API:
    FragmentMetrics     — метрики одного фрагмента
    CollectionStats     — агрегированная статистика коллекции
    compute_fragment_metrics  — метрики одного фрагмента
    compute_collection_stats  — статистика набора фрагментов
    area_histogram            — гистограмма площадей
    compare_collections       — разница метрик двух коллекций
    outlier_indices           — индексы выбросов по площади/аспекту
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── FragmentMetrics ──────────────────────────────────────────────────────────

@dataclass
class FragmentMetrics:
    """Метрики одного фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента.
        area:        Площадь (число ненулевых пикселей маски).
        aspect:      Отношение ширины к высоте bounding-box (w/h).
        density:     Доля ненулевых пикселей в bounding-box.
        n_edges:     Число обработанных краёв.
        perimeter:   Периметр (число граничных пикселей маски, приближённо).
    """

    fragment_id: int
    area: float
    aspect: float
    density: float
    n_edges: int
    perimeter: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.area < 0:
            raise ValueError(
                f"area должна быть >= 0, получено {self.area}"
            )
        if self.aspect <= 0:
            raise ValueError(
                f"aspect должен быть > 0, получено {self.aspect}"
            )
        if not (0.0 <= self.density <= 1.0):
            raise ValueError(
                f"density должна быть в [0, 1], получено {self.density}"
            )
        if self.n_edges < 0:
            raise ValueError(
                f"n_edges должен быть >= 0, получено {self.n_edges}"
            )
        if self.perimeter < 0:
            raise ValueError(
                f"perimeter должен быть >= 0, получено {self.perimeter}"
            )


# ─── CollectionStats ──────────────────────────────────────────────────────────

@dataclass
class CollectionStats:
    """Агрегированная статистика набора фрагментов.

    Атрибуты:
        n_fragments:   Число фрагментов.
        total_area:    Суммарная площадь.
        mean_area:     Средняя площадь.
        std_area:      СКО площадей.
        min_area:      Минимальная площадь.
        max_area:      Максимальная площадь.
        mean_aspect:   Среднее аспектное отношение.
        mean_density:  Средняя плотность.
        mean_edges:    Среднее число краёв.
        extras:        Произвольный словарь дополнительных метрик.
    """

    n_fragments: int
    total_area: float
    mean_area: float
    std_area: float
    min_area: float
    max_area: float
    mean_aspect: float
    mean_density: float
    mean_edges: float
    extras: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments должен быть >= 0, получено {self.n_fragments}"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_fragments": self.n_fragments,
            "total_area": self.total_area,
            "mean_area": self.mean_area,
            "std_area": self.std_area,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "mean_aspect": self.mean_aspect,
            "mean_density": self.mean_density,
            "mean_edges": self.mean_edges,
            **self.extras,
        }


# ─── compute_fragment_metrics ─────────────────────────────────────────────────

def compute_fragment_metrics(
    fragment_id: int,
    mask: np.ndarray,
    n_edges: int = 0,
) -> FragmentMetrics:
    """Вычислить метрики одного фрагмента по его маске.

    Аргументы:
        fragment_id: Идентификатор фрагмента (>= 0).
        mask:        Бинарная маска (H, W), dtype uint8 или bool.
        n_edges:     Число известных краёв фрагмента (>= 0).

    Возвращает:
        FragmentMetrics.

    Исключения:
        ValueError: Если mask не 2-D.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"mask должна быть 2-D, получено ndim={mask.ndim}"
        )

    area = float(np.count_nonzero(mask))

    # Bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any():
        rmin, rmax = int(rows.argmax()), int(len(rows) - 1 - rows[::-1].argmax())
        cmin, cmax = int(cols.argmax()), int(len(cols) - 1 - cols[::-1].argmax())
        bb_h = max(rmax - rmin + 1, 1)
        bb_w = max(cmax - cmin + 1, 1)
    else:
        bb_h, bb_w = 1, 1

    aspect = float(bb_w) / float(bb_h)
    density = float(area) / float(bb_h * bb_w)

    # Approximate perimeter via erosion difference
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        perimeter = float(np.count_nonzero(mask.astype(np.uint8) - eroded))
    except ImportError:
        # fallback: count boundary pixels manually
        padded = np.pad(mask.astype(np.uint8), 1, constant_values=0)
        boundary = (
            (padded[1:-1, 1:-1] > 0) &
            (
                (padded[:-2, 1:-1] == 0) | (padded[2:, 1:-1] == 0) |
                (padded[1:-1, :-2] == 0) | (padded[1:-1, 2:] == 0)
            )
        )
        perimeter = float(boundary.sum())

    return FragmentMetrics(
        fragment_id=fragment_id,
        area=area,
        aspect=aspect,
        density=density,
        n_edges=n_edges,
        perimeter=perimeter,
    )


# ─── compute_collection_stats ─────────────────────────────────────────────────

def compute_collection_stats(
    metrics: List[FragmentMetrics],
) -> CollectionStats:
    """Вычислить агрегированную статистику по набору метрик фрагментов.

    Аргументы:
        metrics: Список FragmentMetrics.

    Возвращает:
        CollectionStats.

    Исключения:
        ValueError: Если metrics пуст.
    """
    if not metrics:
        raise ValueError("metrics не должен быть пустым")

    areas = np.array([m.area for m in metrics], dtype=np.float64)
    aspects = np.array([m.aspect for m in metrics], dtype=np.float64)
    densities = np.array([m.density for m in metrics], dtype=np.float64)
    edges = np.array([m.n_edges for m in metrics], dtype=np.float64)

    return CollectionStats(
        n_fragments=len(metrics),
        total_area=float(areas.sum()),
        mean_area=float(areas.mean()),
        std_area=float(areas.std()),
        min_area=float(areas.min()),
        max_area=float(areas.max()),
        mean_aspect=float(aspects.mean()),
        mean_density=float(densities.mean()),
        mean_edges=float(edges.mean()),
    )


# ─── area_histogram ───────────────────────────────────────────────────────────

def area_histogram(
    metrics: List[FragmentMetrics],
    n_bins: int = 10,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Гистограмма площадей фрагментов.

    Аргументы:
        metrics:   Список FragmentMetrics.
        n_bins:    Число бинов (>= 1).
        normalize: Нормировать счётчики.

    Возвращает:
        Кортеж (counts, edges) как из np.histogram.

    Исключения:
        ValueError: Если metrics пуст или n_bins < 1.
    """
    if not metrics:
        raise ValueError("metrics не должен быть пустым")
    if n_bins < 1:
        raise ValueError(f"n_bins должен быть >= 1, получено {n_bins}")

    areas = np.array([m.area for m in metrics], dtype=np.float64)
    counts, edges = np.histogram(areas, bins=n_bins)
    counts = counts.astype(np.float64)
    if normalize:
        s = counts.sum()
        if s > 0:
            counts /= s
    return counts, edges


# ─── compare_collections ──────────────────────────────────────────────────────

def compare_collections(
    stats_a: CollectionStats,
    stats_b: CollectionStats,
) -> Dict[str, float]:
    """Сравнить две коллекции по ключевым метрикам.

    Возвращает словарь разностей (A − B) для каждой числовой метрики.

    Аргументы:
        stats_a: Статистика первой коллекции.
        stats_b: Статистика второй коллекции.

    Возвращает:
        Словарь с ключами вида "delta_<metric>" и значениями A − B.
    """
    keys = [
        "total_area", "mean_area", "std_area",
        "min_area", "max_area",
        "mean_aspect", "mean_density", "mean_edges",
    ]
    return {
        f"delta_{k}": getattr(stats_a, k) - getattr(stats_b, k)
        for k in keys
    }


# ─── outlier_indices ──────────────────────────────────────────────────────────

def outlier_indices(
    metrics: List[FragmentMetrics],
    z_threshold: float = 2.5,
    by: str = "area",
) -> List[int]:
    """Найти индексы фрагментов-выбросов по заданной метрике.

    Выброс определяется по z-оценке: |z| > z_threshold.

    Аргументы:
        metrics:     Список FragmentMetrics.
        z_threshold: Порог z-оценки (> 0).
        by:          Метрика: 'area' | 'aspect' | 'density' | 'perimeter'.

    Возвращает:
        Список индексов выбросов (пустой, если их нет).

    Исключения:
        ValueError: Если z_threshold <= 0 или by неизвестна.
    """
    if z_threshold <= 0:
        raise ValueError(
            f"z_threshold должен быть > 0, получено {z_threshold}"
        )
    valid = ("area", "aspect", "density", "perimeter")
    if by not in valid:
        raise ValueError(
            f"by должен быть одним из {valid}, получено '{by}'"
        )
    if len(metrics) < 2:
        return []

    values = np.array([getattr(m, by) for m in metrics], dtype=np.float64)
    mu = values.mean()
    sigma = values.std()
    if sigma < 1e-12:
        return []
    z_scores = np.abs((values - mu) / sigma)
    return [int(i) for i in np.where(z_scores > z_threshold)[0]]
