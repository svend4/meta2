"""Пространственный индекс для быстрого поиска ближайших фрагментов.

Модуль предоставляет структуры данных и функции для организации фрагментов
в пространстве (2D grid, kd-tree-подобный поиск), быстрого поиска соседей
и вычисления попарных расстояний.

Публичный API:
    SpatialConfig       — параметры пространственного индекса
    SpatialEntry        — запись об одном элементе в индексе
    SpatialIndex        — пространственный индекс (grid-based)
    build_spatial_index — построить индекс из массива позиций
    query_radius        — найти всех соседей в радиусе R
    query_knn           — найти K ближайших соседей
    pairwise_distances  — матрица попарных расстояний
    cluster_by_distance — разбить точки на кластеры по расстоянию
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── SpatialConfig ────────────────────────────────────────────────────────────

@dataclass
class SpatialConfig:
    """Параметры пространственного индекса.

    Атрибуты:
        cell_size:    Размер ячейки сетки (> 0).
        metric:       Метрика расстояния: 'euclidean' | 'manhattan' | 'chebyshev'.
        max_results:  Максимальное число результатов запроса (0 = не ограничено).
    """

    cell_size: float = 50.0
    metric: str = "euclidean"
    max_results: int = 0

    def __post_init__(self) -> None:
        if self.cell_size <= 0.0:
            raise ValueError(
                f"cell_size должен быть > 0, получено {self.cell_size}"
            )
        if self.metric not in ("euclidean", "manhattan", "chebyshev"):
            raise ValueError(
                f"metric должен быть 'euclidean', 'manhattan' или 'chebyshev', "
                f"получено '{self.metric}'"
            )
        if self.max_results < 0:
            raise ValueError(
                f"max_results должен быть >= 0, получено {self.max_results}"
            )


# ─── SpatialEntry ─────────────────────────────────────────────────────────────

@dataclass
class SpatialEntry:
    """Запись об одном элементе в пространственном индексе.

    Атрибуты:
        item_id:  Идентификатор элемента.
        position: Позиция в 2D-пространстве (2,).
        payload:  Произвольные данные.
    """

    item_id: int
    position: np.ndarray
    payload: object = None

    def __post_init__(self) -> None:
        if self.item_id < 0:
            raise ValueError(
                f"item_id должен быть >= 0, получено {self.item_id}"
            )
        pos = np.asarray(self.position)
        if pos.shape != (2,):
            raise ValueError(
                f"position должен иметь форму (2,), получено {pos.shape}"
            )
        self.position = pos.astype(np.float64)


# ─── SpatialIndex ─────────────────────────────────────────────────────────────

class SpatialIndex:
    """Пространственный индекс на основе сетки.

    Аргументы:
        cfg: Параметры (None → SpatialConfig()).
    """

    def __init__(self, cfg: Optional[SpatialConfig] = None) -> None:
        if cfg is None:
            cfg = SpatialConfig()
        self._cfg = cfg
        self._grid: Dict[Tuple[int, int], List[SpatialEntry]] = {}
        self._entries: List[SpatialEntry] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def config(self) -> SpatialConfig:
        return self._cfg

    def __len__(self) -> int:
        return self.size

    def __contains__(self, item_id: int) -> bool:
        return any(e.item_id == item_id for e in self._entries)

    # ── Core operations ───────────────────────────────────────────────────────

    def insert(self, entry: SpatialEntry) -> None:
        """Добавить элемент в индекс.

        Аргументы:
            entry: SpatialEntry.
        """
        cell = self._cell(entry.position)
        if cell not in self._grid:
            self._grid[cell] = []
        self._grid[cell].append(entry)
        self._entries.append(entry)

    def remove(self, item_id: int) -> bool:
        """Удалить элемент из индекса.

        Возвращает:
            True если элемент найден и удалён.
        """
        removed = False
        self._entries = [e for e in self._entries if e.item_id != item_id]
        for cell, entries in self._grid.items():
            before = len(entries)
            self._grid[cell] = [e for e in entries if e.item_id != item_id]
            if len(self._grid[cell]) < before:
                removed = True
        return removed

    def clear(self) -> None:
        """Очистить индекс."""
        self._grid.clear()
        self._entries.clear()

    def get_all(self) -> List[SpatialEntry]:
        """Вернуть все записи."""
        return list(self._entries)

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_radius(
        self,
        center: np.ndarray,
        radius: float,
    ) -> List[Tuple[float, SpatialEntry]]:
        """Найти все элементы в радиусе.

        Аргументы:
            center: Центр запроса (2,).
            radius: Радиус поиска (>= 0).

        Возвращает:
            Список (dist, entry), отсортированный по расстоянию.

        Исключения:
            ValueError: Если radius < 0.
        """
        if radius < 0:
            raise ValueError(f"radius должен быть >= 0, получено {radius}")
        center = np.asarray(center, dtype=np.float64)

        # Определить диапазон ячеек
        cells = self._candidate_cells(center, radius)
        seen: set = set()
        results = []

        for cell in cells:
            for entry in self._grid.get(cell, []):
                if entry.item_id in seen:
                    continue
                seen.add(entry.item_id)
                d = self._distance(center, entry.position)
                if d <= radius:
                    results.append((d, entry))

        results.sort(key=lambda x: x[0])
        if self._cfg.max_results > 0:
            results = results[:self._cfg.max_results]
        return results

    def query_knn(
        self,
        center: np.ndarray,
        k: int,
    ) -> List[Tuple[float, SpatialEntry]]:
        """Найти K ближайших соседей.

        Аргументы:
            center: Центр запроса (2,).
            k:      Число ближайших соседей (>= 1).

        Возвращает:
            Список (dist, entry) длиной min(k, size).

        Исключения:
            ValueError: Если k < 1.
        """
        if k < 1:
            raise ValueError(f"k должен быть >= 1, получено {k}")
        center = np.asarray(center, dtype=np.float64)

        # Начинаем с малого радиуса и расширяем при необходимости
        if not self._entries:
            return []

        all_dists = [
            (self._distance(center, e.position), e)
            for e in self._entries
        ]
        all_dists.sort(key=lambda x: x[0])
        return all_dists[:k]

    # ── Private ───────────────────────────────────────────────────────────────

    def _cell(self, pos: np.ndarray) -> Tuple[int, int]:
        s = self._cfg.cell_size
        return (int(math.floor(pos[0] / s)), int(math.floor(pos[1] / s)))

    def _candidate_cells(
        self,
        center: np.ndarray,
        radius: float,
    ) -> List[Tuple[int, int]]:
        s = self._cfg.cell_size
        cx, cy = center[0] / s, center[1] / s
        r = math.ceil(radius / s) + 1
        cx0, cy0 = int(math.floor(cx - r)), int(math.floor(cy - r))
        cx1, cy1 = int(math.ceil(cx + r)), int(math.ceil(cy + r))
        return [
            (ix, iy)
            for ix in range(cx0, cx1 + 1)
            for iy in range(cy0, cy1 + 1)
        ]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._cfg.metric == "euclidean":
            return float(np.linalg.norm(a - b))
        if self._cfg.metric == "manhattan":
            return float(np.abs(a - b).sum())
        # chebyshev
        return float(np.abs(a - b).max())


# ─── build_spatial_index ─────────────────────────────────────────────────────

def build_spatial_index(
    positions: np.ndarray,
    payloads: Optional[List[object]] = None,
    cfg: Optional[SpatialConfig] = None,
) -> SpatialIndex:
    """Построить пространственный индекс из массива позиций.

    Аргументы:
        positions: Массив позиций (N, 2).
        payloads:  Список данных (None → None для каждого).
        cfg:       Параметры (None → SpatialConfig()).

    Возвращает:
        SpatialIndex.

    Исключения:
        ValueError: Если positions не (N, 2).
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"positions должен иметь форму (N, 2), получено {pts.shape}"
        )
    if payloads is not None and len(payloads) != len(pts):
        raise ValueError(
            f"len(payloads)={len(payloads)} != len(positions)={len(pts)}"
        )

    idx = SpatialIndex(cfg)
    for i, pos in enumerate(pts):
        payload = payloads[i] if payloads is not None else None
        idx.insert(SpatialEntry(item_id=i, position=pos, payload=payload))
    return idx


# ─── query_radius ─────────────────────────────────────────────────────────────

def query_radius(
    index: SpatialIndex,
    center: np.ndarray,
    radius: float,
) -> List[Tuple[float, SpatialEntry]]:
    """Удобная обёртка вокруг SpatialIndex.query_radius."""
    return index.query_radius(center, radius)


# ─── query_knn ────────────────────────────────────────────────────────────────

def query_knn(
    index: SpatialIndex,
    center: np.ndarray,
    k: int,
) -> List[Tuple[float, SpatialEntry]]:
    """Удобная обёртка вокруг SpatialIndex.query_knn."""
    return index.query_knn(center, k)


# ─── pairwise_distances ───────────────────────────────────────────────────────

def pairwise_distances(
    positions: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Вычислить матрицу попарных расстояний.

    Аргументы:
        positions: Массив позиций (N, 2).
        metric:    'euclidean' | 'manhattan' | 'chebyshev'.

    Возвращает:
        Симметричная матрица (N, N), float64. Диагональ = 0.

    Исключения:
        ValueError: Если positions не (N, 2) или metric неверная.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"positions должен иметь форму (N, 2), получено {pts.shape}"
        )
    if metric not in ("euclidean", "manhattan", "chebyshev"):
        raise ValueError(
            f"metric должен быть 'euclidean', 'manhattan' или 'chebyshev', "
            f"получено '{metric}'"
        )

    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)

    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=2))
    if metric == "manhattan":
        return np.abs(diff).sum(axis=2)
    # chebyshev
    return np.abs(diff).max(axis=2)


# ─── cluster_by_distance ──────────────────────────────────────────────────────

def cluster_by_distance(
    positions: np.ndarray,
    threshold: float,
    metric: str = "euclidean",
) -> List[List[int]]:
    """Разбить точки на кластеры: две точки в одном кластере, если dist < threshold.

    Использует алгоритм union-find для объединения компонент.

    Аргументы:
        positions:  Массив позиций (N, 2).
        threshold:  Порог расстояния для слияния (>= 0).
        metric:     Метрика расстояния.

    Возвращает:
        Список кластеров (каждый — список индексов точек).

    Исключения:
        ValueError: Если threshold < 0.
    """
    if threshold < 0:
        raise ValueError(
            f"threshold должен быть >= 0, получено {threshold}"
        )

    pts = np.asarray(positions, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return []

    dist_mat = pairwise_distances(pts, metric=metric)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if dist_mat[i, j] < threshold:
                union(i, j)

    # Collect clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    return sorted(clusters.values(), key=lambda c: c[0])
