"""Утилиты для работы с графами совместимости фрагментов.

Модуль предоставляет функции для построения и анализа графов:
создание взвешенного графа из матрицы оценок, поиск кратчайших путей
(Dijkstra), минимального остовного дерева (Prim), выявление связных
компонент, вычисление степени вершин и пакетная обработка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─── GraphEdge ────────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    """Ребро взвешенного графа.

    Атрибуты:
        src:    Исходная вершина (>= 0).
        dst:    Целевая вершина (>= 0).
        weight: Вес ребра (>= 0).
        params: Дополнительные параметры.
    """

    src: int
    dst: int
    weight: float
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.src < 0:
            raise ValueError(f"src должен быть >= 0, получено {self.src}")
        if self.dst < 0:
            raise ValueError(f"dst должен быть >= 0, получено {self.dst}")
        if self.weight < 0.0:
            raise ValueError(f"weight должен быть >= 0, получено {self.weight}")


# ─── FragmentGraph ────────────────────────────────────────────────────────────

@dataclass
class FragmentGraph:
    """Взвешенный неориентированный граф фрагментов.

    Атрибуты:
        n_nodes:    Количество вершин (>= 1).
        edges:      Список рёбер.
        adj:        Список смежности {вершина: [(сосед, вес)]}.
        params:     Дополнительные параметры.
    """

    n_nodes: int
    edges: List[GraphEdge]
    adj: Dict[int, List[Tuple[int, float]]]
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(
                f"n_nodes должен быть >= 1, получено {self.n_nodes}"
            )

    def __len__(self) -> int:
        return self.n_nodes


# ─── build_graph ──────────────────────────────────────────────────────────────

def build_graph(
    score_matrix: np.ndarray,
    threshold: float = 0.0,
) -> FragmentGraph:
    """Построить граф из матрицы оценок совместимости.

    Аргументы:
        score_matrix: Квадратная симметричная матрица (N, N), score[i,j] >= 0.
        threshold:    Минимальный вес для включения ребра (>= 0).

    Возвращает:
        FragmentGraph с рёбрами, где вес >= threshold.

    Исключения:
        ValueError: Если матрица не квадратная или threshold < 0.
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix должна быть 2-D, получено ndim={score_matrix.ndim}"
        )
    if score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError(
            f"score_matrix должна быть квадратной, получено {score_matrix.shape}"
        )
    if threshold < 0.0:
        raise ValueError(f"threshold должен быть >= 0, получено {threshold}")

    N = score_matrix.shape[0]
    edges: List[GraphEdge] = []
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(N)}

    for i in range(N):
        for j in range(i + 1, N):
            w = float(score_matrix[i, j])
            if w > threshold:
                edges.append(GraphEdge(src=i, dst=j, weight=w))
                adj[i].append((j, w))
                adj[j].append((i, w))

    return FragmentGraph(n_nodes=N, edges=edges, adj=adj)


# ─── dijkstra ────────────────────────────────────────────────────────────────

def dijkstra(
    graph: FragmentGraph,
    source: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Алгоритм Дейкстры для кратчайших путей из source.

    Веса рёбер интерпретируются как расстояния (чем меньше — тем лучше).
    Для графов совместимости (чем больше — тем лучше) используйте
    inverted_weight = 1 / (weight + eps) перед вызовом.

    Аргументы:
        graph:  FragmentGraph.
        source: Исходная вершина (0 <= source < n_nodes).

    Возвращает:
        Кортеж (dist, prev):
          - dist (float64, shape=(N,)): кратчайшие расстояния от source.
          - prev (int64, shape=(N,)): предшественники на кратчайшем пути (-1 = нет).

    Исключения:
        ValueError: Если source вне диапазона.
    """
    N = graph.n_nodes
    if not (0 <= source < N):
        raise ValueError(
            f"source должен быть в [0, {N - 1}], получено {source}"
        )

    INF = float("inf")
    dist = np.full(N, INF, dtype=np.float64)
    prev = np.full(N, -1, dtype=np.int64)
    dist[source] = 0.0
    visited: Set[int] = set()

    # Простая O(N²) реализация (достаточно для малых графов)
    for _ in range(N):
        # Выбираем непосещённую вершину с минимальным расстоянием
        u = -1
        min_d = INF
        for v in range(N):
            if v not in visited and dist[v] < min_d:
                min_d = dist[v]
                u = v
        if u == -1:
            break
        visited.add(u)
        for v, w in graph.adj[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    return dist, prev


# ─── shortest_path ────────────────────────────────────────────────────────────

def shortest_path(
    graph: FragmentGraph, source: int, target: int
) -> List[int]:
    """Восстановить кратчайший путь от source до target.

    Аргументы:
        graph:  FragmentGraph.
        source: Исходная вершина.
        target: Целевая вершина.

    Возвращает:
        Список вершин пути [source, ..., target] или пустой список, если
        target недостижим.

    Исключения:
        ValueError: Если source или target вне диапазона.
    """
    N = graph.n_nodes
    if not (0 <= target < N):
        raise ValueError(
            f"target должен быть в [0, {N - 1}], получено {target}"
        )
    _, prev = dijkstra(graph, source)
    if prev[target] == -1 and target != source:
        return []
    path = []
    cur = target
    while cur != -1:
        path.append(int(cur))
        cur = int(prev[cur])
    return path[::-1]


# ─── minimum_spanning_tree ────────────────────────────────────────────────────

def minimum_spanning_tree(graph: FragmentGraph) -> List[GraphEdge]:
    """Алгоритм Прима: минимальное остовное дерево.

    Аргументы:
        graph: FragmentGraph (связный).

    Возвращает:
        Список рёбер MST (n_nodes - 1 рёбер или меньше для несвязного графа).
    """
    N = graph.n_nodes
    INF = float("inf")
    in_mst = [False] * N
    key = [INF] * N
    parent = [-1] * N
    key[0] = 0.0
    mst_edges: List[GraphEdge] = []

    for _ in range(N):
        # Выбираем вершину с минимальным key вне MST
        u = -1
        for v in range(N):
            if not in_mst[v] and (u == -1 or key[v] < key[u]):
                u = v
        if u == -1 or key[u] == INF:
            break
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append(GraphEdge(src=parent[u], dst=u, weight=key[u]))
        for v, w in graph.adj[u]:
            if not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u

    return mst_edges


# ─── connected_components ─────────────────────────────────────────────────────

def connected_components(graph: FragmentGraph) -> List[List[int]]:
    """Найти связные компоненты графа (BFS).

    Аргументы:
        graph: FragmentGraph.

    Возвращает:
        Список компонент, каждая — список вершин.
    """
    N = graph.n_nodes
    visited = [False] * N
    components: List[List[int]] = []

    for start in range(N):
        if visited[start]:
            continue
        component: List[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            u = queue.pop(0)
            component.append(u)
            for v, _ in graph.adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        components.append(sorted(component))

    return components


# ─── node_degrees ─────────────────────────────────────────────────────────────

def node_degrees(graph: FragmentGraph) -> np.ndarray:
    """Вычислить степень каждой вершины.

    Аргументы:
        graph: FragmentGraph.

    Возвращает:
        Массив (N,) int64 со степенями вершин.
    """
    degrees = np.array(
        [len(graph.adj[v]) for v in range(graph.n_nodes)], dtype=np.int64
    )
    return degrees


# ─── subgraph ─────────────────────────────────────────────────────────────────

def subgraph(graph: FragmentGraph, nodes: List[int]) -> FragmentGraph:
    """Извлечь индуцированный подграф на заданных вершинах.

    Аргументы:
        graph: FragmentGraph.
        nodes: Список вершин подграфа (уникальные, >= 0).

    Возвращает:
        Новый FragmentGraph с перенумерованными вершинами (0..len(nodes)-1).

    Исключения:
        ValueError: Если nodes пуст или содержит недопустимые индексы.
    """
    if not nodes:
        raise ValueError("nodes не может быть пустым")
    for v in nodes:
        if v < 0 or v >= graph.n_nodes:
            raise ValueError(
                f"Вершина {v} выходит за пределы [0, {graph.n_nodes - 1}]"
            )

    idx_map = {old: new for new, old in enumerate(nodes)}
    new_n = len(nodes)
    new_adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(new_n)}
    new_edges: List[GraphEdge] = []

    for edge in graph.edges:
        if edge.src in idx_map and edge.dst in idx_map:
            u = idx_map[edge.src]
            v = idx_map[edge.dst]
            new_edges.append(GraphEdge(src=u, dst=v, weight=edge.weight))
            new_adj[u].append((v, edge.weight))
            new_adj[v].append((u, edge.weight))

    return FragmentGraph(n_nodes=new_n, edges=new_edges, adj=new_adj)


# ─── batch_build_graphs ────────────────────────────────────────────────────────

def batch_build_graphs(
    matrices: List[np.ndarray], threshold: float = 0.0
) -> List[FragmentGraph]:
    """Построить графы из списка матриц оценок.

    Аргументы:
        matrices:  Список квадратных матриц.
        threshold: Порог веса рёбер.

    Возвращает:
        Список FragmentGraph.
    """
    return [build_graph(m, threshold=threshold) for m in matrices]
