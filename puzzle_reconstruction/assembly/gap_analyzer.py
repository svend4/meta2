"""Анализ зазоров между размещёнными фрагментами пазла.

Модуль вычисляет зазоры между соседними прямоугольниками фрагментов,
классифицирует пары по характеру зазора (перекрытие / касание / зазор),
строит гистограммы и сводные отчёты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── FragmentBounds ───────────────────────────────────────────────────────────

@dataclass
class FragmentBounds:
    """Прямоугольник размещённого фрагмента.

    Атрибуты:
        fragment_id: Уникальный идентификатор (>= 0).
        x, y:        Координаты верхнего левого угла (>= 0).
        width:       Ширина (>= 1).
        height:      Высота (>= 1).
    """

    fragment_id: int
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            raise ValueError(
                f"fragment_id должен быть >= 0, получено {self.fragment_id}"
            )
        if self.x < 0:
            raise ValueError(f"x должен быть >= 0, получено {self.x}")
        if self.y < 0:
            raise ValueError(f"y должен быть >= 0, получено {self.y}")
        if self.width < 1:
            raise ValueError(
                f"width должен быть >= 1, получено {self.width}"
            )
        if self.height < 1:
            raise ValueError(
                f"height должен быть >= 1, получено {self.height}"
            )

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


# ─── GapInfo ──────────────────────────────────────────────────────────────────

@dataclass
class GapInfo:
    """Зазор между двумя фрагментами.

    Атрибуты:
        id1, id2:       Идентификаторы фрагментов (id1 < id2).
        gap_x:          Горизонтальный зазор (< 0 = перекрытие).
        gap_y:          Вертикальный зазор (< 0 = перекрытие).
        distance:       Евклидово расстояние между центрами (>= 0).
        category:       'overlap' | 'touching' | 'near' | 'far'.
    """

    id1: int
    id2: int
    gap_x: float
    gap_y: float
    distance: float
    category: str = "far"

    def __post_init__(self) -> None:
        if self.id1 < 0:
            raise ValueError(f"id1 должен быть >= 0, получено {self.id1}")
        if self.id2 < 0:
            raise ValueError(f"id2 должен быть >= 0, получено {self.id2}")
        if self.distance < 0:
            raise ValueError(
                f"distance должен быть >= 0, получено {self.distance}"
            )
        _valid_cats = {"overlap", "touching", "near", "far"}
        if self.category not in _valid_cats:
            raise ValueError(
                f"category должен быть одним из {_valid_cats}, "
                f"получено '{self.category}'"
            )

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.id1, self.id2)

    @property
    def is_overlapping(self) -> bool:
        return self.gap_x < 0 and self.gap_y < 0


# ─── GapStats ─────────────────────────────────────────────────────────────────

@dataclass
class GapStats:
    """Сводная статистика зазоров.

    Атрибуты:
        n_pairs:       Число проанализированных пар.
        n_overlapping: Число перекрывающихся пар.
        n_touching:    Число касающихся пар (зазор = 0).
        n_near:        Число пар с малым положительным зазором.
        n_far:         Число пар с большим зазором.
        mean_distance: Среднее расстояние между центрами.
        std_distance:  СКО расстояния.
    """

    n_pairs: int = 0
    n_overlapping: int = 0
    n_touching: int = 0
    n_near: int = 0
    n_far: int = 0
    mean_distance: float = 0.0
    std_distance: float = 0.0

    def __post_init__(self) -> None:
        for name, val in (
            ("n_pairs", self.n_pairs),
            ("n_overlapping", self.n_overlapping),
            ("n_touching", self.n_touching),
            ("n_near", self.n_near),
            ("n_far", self.n_far),
        ):
            if val < 0:
                raise ValueError(
                    f"{name} должен быть >= 0, получено {val}"
                )
        if self.mean_distance < 0:
            raise ValueError(
                f"mean_distance должен быть >= 0, получено {self.mean_distance}"
            )
        if self.std_distance < 0:
            raise ValueError(
                f"std_distance должен быть >= 0, получено {self.std_distance}"
            )


# ─── compute_gap ──────────────────────────────────────────────────────────────

def compute_gap(
    a: FragmentBounds,
    b: FragmentBounds,
    near_threshold: float = 5.0,
) -> GapInfo:
    """Вычислить зазор между двумя фрагментами.

    Горизонтальный зазор:
        max(0, max(a.x, b.x) - min(a.x2, b.x2))  если нет перекрытия
        иначе отрицательное значение (глубина перекрытия).
    Аналогично для вертикального.

    Аргументы:
        a, b:            Фрагменты.
        near_threshold:  Порог «близко» (пикселей, >= 0).

    Возвращает:
        GapInfo.

    Исключения:
        ValueError: Если near_threshold < 0.
    """
    if near_threshold < 0:
        raise ValueError(
            f"near_threshold должен быть >= 0, получено {near_threshold}"
        )

    id1, id2 = sorted([a.fragment_id, b.fragment_id])

    # Горизонтальный зазор
    overlap_x = min(a.x2, b.x2) - max(a.x, b.x)
    gap_x = -overlap_x if overlap_x > 0 else max(a.x, b.x) - min(a.x2, b.x2)

    # Вертикальный зазор
    overlap_y = min(a.y2, b.y2) - max(a.y, b.y)
    gap_y = -overlap_y if overlap_y > 0 else max(a.y, b.y) - min(a.y2, b.y2)

    # Расстояние между центрами
    cx1, cy1 = a.center
    cx2, cy2 = b.center
    distance = float(np.hypot(cx2 - cx1, cy2 - cy1))

    # Категория
    if gap_x < 0 and gap_y < 0:
        category = "overlap"
    elif gap_x == 0 or gap_y == 0:
        category = "touching"
    elif max(gap_x, gap_y) <= near_threshold:
        category = "near"
    else:
        category = "far"

    return GapInfo(
        id1=id1, id2=id2,
        gap_x=float(gap_x), gap_y=float(gap_y),
        distance=distance,
        category=category,
    )


# ─── find_adjacent ────────────────────────────────────────────────────────────

def find_adjacent(
    fragments: List[FragmentBounds],
    distance_threshold: float = 20.0,
) -> List[GapInfo]:
    """Найти все пары фрагментов, расположенных на расстоянии <= threshold.

    Аргументы:
        fragments:          Список фрагментов.
        distance_threshold: Максимальное расстояние между центрами (>= 0).

    Возвращает:
        Список GapInfo для близких пар.

    Исключения:
        ValueError: Если distance_threshold < 0.
    """
    if distance_threshold < 0:
        raise ValueError(
            f"distance_threshold должен быть >= 0, получено {distance_threshold}"
        )
    result = []
    n = len(fragments)
    for i in range(n):
        for j in range(i + 1, n):
            gi = compute_gap(fragments[i], fragments[j])
            if gi.distance <= distance_threshold:
                result.append(gi)
    return result


# ─── analyze_all_gaps ─────────────────────────────────────────────────────────

def analyze_all_gaps(
    fragments: List[FragmentBounds],
    near_threshold: float = 5.0,
) -> List[GapInfo]:
    """Проанализировать зазоры между всеми парами фрагментов.

    Аргументы:
        fragments:       Список фрагментов.
        near_threshold:  Порог «близко» (>= 0).

    Возвращает:
        Список GapInfo (C(N,2) элементов).
    """
    result = []
    n = len(fragments)
    for i in range(n):
        for j in range(i + 1, n):
            result.append(compute_gap(fragments[i], fragments[j], near_threshold))
    return result


# ─── gap_histogram ────────────────────────────────────────────────────────────

def gap_histogram(
    gaps: List[GapInfo],
    bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Гистограмма расстояний между центрами фрагментов.

    Аргументы:
        gaps: Список GapInfo.
        bins: Число бинов (>= 1).

    Возвращает:
        (counts, bin_edges) — результат np.histogram.

    Исключения:
        ValueError: Если bins < 1.
    """
    if bins < 1:
        raise ValueError(f"bins должен быть >= 1, получено {bins}")
    if not gaps:
        return np.zeros(bins, dtype=np.int64), np.zeros(bins + 1)
    distances = np.array([g.distance for g in gaps], dtype=np.float64)
    counts, edges = np.histogram(distances, bins=bins)
    return counts.astype(np.int64), edges


# ─── classify_gaps ────────────────────────────────────────────────────────────

def classify_gaps(gaps: List[GapInfo]) -> Dict[str, List[GapInfo]]:
    """Сгруппировать зазоры по категориям.

    Возвращает:
        Словарь {'overlap': [...], 'touching': [...], 'near': [...], 'far': [...]}.
    """
    result: Dict[str, List[GapInfo]] = {
        "overlap": [], "touching": [], "near": [], "far": []
    }
    for g in gaps:
        result[g.category].append(g)
    return result


# ─── summarize ────────────────────────────────────────────────────────────────

def summarize(gaps: List[GapInfo]) -> GapStats:
    """Вычислить сводную статистику по списку зазоров.

    Аргументы:
        gaps: Список GapInfo.

    Возвращает:
        GapStats.
    """
    if not gaps:
        return GapStats()

    distances = np.array([g.distance for g in gaps], dtype=np.float64)
    by_cat = classify_gaps(gaps)

    return GapStats(
        n_pairs=len(gaps),
        n_overlapping=len(by_cat["overlap"]),
        n_touching=len(by_cat["touching"]),
        n_near=len(by_cat["near"]),
        n_far=len(by_cat["far"]),
        mean_distance=float(distances.mean()),
        std_distance=float(distances.std()),
    )


# ─── batch_analyze ────────────────────────────────────────────────────────────

def batch_analyze(
    layouts: List[List[FragmentBounds]],
    near_threshold: float = 5.0,
) -> List[GapStats]:
    """Анализ зазоров для нескольких раскладок.

    Аргументы:
        layouts:         Список раскладок (каждая — список FragmentBounds).
        near_threshold:  Порог «близко» (>= 0).

    Возвращает:
        Список GapStats.
    """
    return [
        summarize(analyze_all_gaps(layout, near_threshold))
        for layout in layouts
    ]
