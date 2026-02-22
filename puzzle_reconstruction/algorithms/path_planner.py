"""
Алгоритмы планирования пути для обхода фрагментов документа.

Предоставляет инструменты поиска кратчайших и оптимальных путей
на графе совместимостей фрагментов.

Экспортирует:
    PathResult              — результат поиска пути
    dijkstra                — алгоритм Дейкстры на матрице стоимостей
    shortest_path           — кратчайший путь по инвертированным оценкам
    all_pairs_shortest_paths — матрица расстояний (Флойд-Уоршелл)
    topological_sort        — топологическая сортировка (обнаружение циклов)
    find_connected_components — связные компоненты графа
    minimum_spanning_tree   — минимальное остовное дерево
    batch_dijkstra          — пакетный поиск путей из нескольких стартов
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class PathResult:
    """Результат поиска пути на графе.

    Attributes:
        path:  Список вершин (индексов) в найденном пути.
        cost:  Суммарная стоимость пути (≥ 0).
        found: ``True`` если путь найден.
        params: Параметры алгоритма.
    """
    path: List[int]
    cost: float
    found: bool = True
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")

    def __len__(self) -> int:
        return len(self.path)

    def __repr__(self) -> str:  # pragma: no cover
        status = "found" if self.found else "not_found"
        return f"PathResult({status}, len={len(self.path)}, cost={self.cost:.4f})"


# ─── Публичные функции ────────────────────────────────────────────────────────

def dijkstra(
    cost_matrix: np.ndarray,
    start: int,
    end: int,
) -> PathResult:
    """Алгоритм Дейкстры для поиска кратчайшего пути.

    Нулевые и отрицательные элементы матрицы трактуются как отсутствие ребра
    (бесконечная стоимость).

    Args:
        cost_matrix: Квадратная матрица (N, N) float неотрицательных стоимостей.
                     ``cost_matrix[i, j]`` — стоимость перехода i→j
                     (0 означает нет ребра, кроме диагонали).
        start:       Начальная вершина (0-based).
        end:         Конечная вершина (0-based).

    Returns:
        :class:`PathResult` с найденным путём. Если пути нет —
        ``found=False``, ``path=[]``.

    Raises:
        ValueError: Если матрица не квадратная или индексы вне диапазона.
    """
    mat = np.asarray(cost_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]
    _validate_index(start, n, "start")
    _validate_index(end, n, "end")

    if start == end:
        return PathResult(path=[start], cost=0.0,
                          params={"algorithm": "dijkstra"})

    INF = float("inf")
    dist = [INF] * n
    prev: List[Optional[int]] = [None] * n
    dist[start] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in range(n):
            if v == u:
                continue
            w = mat[u, v]
            if w <= 0:
                continue  # нет ребра
            new_d = dist[u] + w
            if new_d < dist[v]:
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(heap, (new_d, v))

    if dist[end] == INF:
        return PathResult(path=[], cost=0.0, found=False,
                          params={"algorithm": "dijkstra"})

    path = _reconstruct_path(prev, start, end)
    return PathResult(
        path=path,
        cost=float(dist[end]),
        found=True,
        params={"algorithm": "dijkstra"},
    )


def shortest_path(
    score_matrix: np.ndarray,
    start: int,
    end: int,
) -> PathResult:
    """Кратчайший путь по инвертированным оценкам совместимости.

    Оценки [0, 1] преобразуются в стоимости ``1 - score``;
    нулевые оценки трактуются как отсутствие ребра.

    Args:
        score_matrix: Квадратная матрица (N, N) оценок ∈ [0, 1].
        start:        Начальная вершина.
        end:          Конечная вершина.

    Returns:
        :class:`PathResult`.

    Raises:
        ValueError: Если матрица не квадратная.
    """
    mat = np.asarray(score_matrix, dtype=np.float64)
    _validate_square(mat)
    cost_mat = np.where(mat > 0, 1.0 - mat, 0.0)
    return dijkstra(cost_mat, start, end)


def all_pairs_shortest_paths(cost_matrix: np.ndarray) -> np.ndarray:
    """Вычислить матрицу кратчайших расстояний алгоритмом Флойда-Уоршелла.

    Args:
        cost_matrix: Квадратная матрица (N, N) float стоимостей (0 = нет ребра).

    Returns:
        Матрица float64 (N, N); ``np.inf`` означает недостижимость.

    Raises:
        ValueError: Если матрица не квадратная.
    """
    mat = np.asarray(cost_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]
    INF = float("inf")

    dist = np.where(mat > 0, mat, INF)
    np.fill_diagonal(dist, 0.0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                new_d = dist[i, k] + dist[k, j]
                if new_d < dist[i, j]:
                    dist[i, j] = new_d
    return dist


def topological_sort(adj_matrix: np.ndarray) -> List[int]:
    """Топологическая сортировка DAG (обнаружение циклов).

    Args:
        adj_matrix: Квадратная матрица (N, N); ненулевой элемент [i, j]
                    означает ребро i→j.

    Returns:
        Список вершин в топологическом порядке.

    Raises:
        ValueError: Если матрица не квадратная.
        RuntimeError: Если граф содержит цикл.
    """
    mat = np.asarray(adj_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]

    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] != 0:
                in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    result: List[int] = []

    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in range(n):
            if u != v and mat[u, v] != 0:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if len(result) != n:
        raise RuntimeError(
            "Graph contains a cycle; topological sort is not possible"
        )
    return result


def find_connected_components(adj_matrix: np.ndarray) -> List[List[int]]:
    """Найти связные компоненты ненаправленного графа.

    Args:
        adj_matrix: Квадратная матрица (N, N); ненулевой [i, j] — ребро.

    Returns:
        Список компонент; каждая компонента — список вершин (сортированный).

    Raises:
        ValueError: Если матрица не квадратная.
    """
    mat = np.asarray(adj_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]

    visited = [False] * n
    components: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        component: List[int] = []
        stack = [start]
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            component.append(u)
            for v in range(n):
                if not visited[v] and (mat[u, v] != 0 or mat[v, u] != 0):
                    stack.append(v)
        components.append(sorted(component))

    return components


def minimum_spanning_tree(adj_matrix: np.ndarray) -> np.ndarray:
    """Построить минимальное остовное дерево алгоритмом Прима.

    Args:
        adj_matrix: Квадратная матрица (N, N) неотрицательных весов
                    (0 = нет ребра). Граф считается ненаправленным.

    Returns:
        Матрица float64 (N, N) — весовая матрица MST (симметричная).

    Raises:
        ValueError: Если матрица не квадратная.
    """
    mat = np.asarray(adj_matrix, dtype=np.float64)
    _validate_square(mat)
    n = mat.shape[0]
    if n == 0:
        return mat.copy()

    INF = float("inf")
    in_tree = [False] * n
    key = [INF] * n
    parent = [-1] * n
    key[0] = 0.0

    for _ in range(n):
        # Вершина с минимальным key не в дереве
        u = min(
            (v for v in range(n) if not in_tree[v]),
            key=lambda v: key[v],
            default=-1,
        )
        if u == -1:
            break
        in_tree[u] = True
        for v in range(n):
            w = mat[u, v]
            if v != u and w > 0 and not in_tree[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    mst = np.zeros((n, n), dtype=np.float64)
    for v in range(1, n):
        p = parent[v]
        if p >= 0:
            w = mat[p, v]
            mst[p, v] = w
            mst[v, p] = w
    return mst


def batch_dijkstra(
    cost_matrix: np.ndarray,
    starts: List[int],
    end: int,
) -> List[PathResult]:
    """Пакетный поиск путей из нескольких стартовых вершин.

    Args:
        cost_matrix: Квадратная матрица стоимостей (N, N).
        starts:      Список начальных вершин.
        end:         Конечная вершина.

    Returns:
        Список :class:`PathResult` той же длины, что и ``starts``.
    """
    return [dijkstra(cost_matrix, s, end) for s in starts]


# ─── Приватные ───────────────────────────────────────────────────────────────

def _validate_square(mat: np.ndarray) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Matrix must be square 2-D, got shape {mat.shape}"
        )


def _validate_index(idx: int, n: int, name: str) -> None:
    if not (0 <= idx < n):
        raise ValueError(
            f"{name} must be in [0, {n - 1}], got {idx}"
        )


def _reconstruct_path(
    prev: List[Optional[int]],
    start: int,
    end: int,
) -> List[int]:
    path: List[int] = []
    cur: Optional[int] = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path[0] == start else []
