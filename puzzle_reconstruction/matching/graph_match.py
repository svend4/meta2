"""
Графовый подход к сопоставлению фрагментов.

Фрагменты документа моделируются как узлы взвешенного графа; рёбра — пары
фрагментов с весом = score CompatEntry. Это открывает доступ к классическим
алгоритмам теории графов: минимальное остовное дерево (MST), спектральная
упорядочивание, случайные блуждания.

Алгоритмы:
    Kruskal MST          — оптимальная связка фрагментов при наилучших стыках.
    Спектральный порядок — упорядочение по Fiedler-вектору (λ₂ лапласиана).
    Случайное блуждание  — мера «достижимости» между парами фрагментов.
    Центральность степени — рейтинг фрагментов по числу сильных соседей.

Классы:
    FragmentGraph   — взвешенный неориентированный граф фрагментов
    GraphMatchResult — результат графового анализа

Функции:
    build_fragment_graph   — строит FragmentGraph из CompatEntry
    mst_ordering           — порядок обхода MST (DFS)
    spectral_ordering      — порядок по Fiedler-вектору
    random_walk_similarity — матрица стационарных вероятностей
    degree_centrality      — нормированная степень узлов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from ..models import Assembly, CompatEntry, Fragment


# ─── FragmentGraph ────────────────────────────────────────────────────────────

class FragmentGraph:
    """
    Взвешенный неориентированный граф фрагментов.

    Узлы: fragment_id (int).
    Рёбра: {frozenset{fid_i, fid_j}: weight} где weight = max compat score.

    Attributes:
        nodes:      Множество fragment_id.
        edges:      {frozenset{i,j}: weight} — веса рёбер ∈ [0,1].
        frag_map:   {fragment_id: Fragment} — ссылки на объекты.
    """

    def __init__(self) -> None:
        self.nodes:    Set[int]                  = set()
        self.edges:    Dict[FrozenSet[int], float] = {}
        self.frag_map: Dict[int, Fragment]       = {}

    # ── Построение ────────────────────────────────────────────────────────

    def add_node(self, fragment: Fragment) -> None:
        self.nodes.add(fragment.fragment_id)
        self.frag_map[fragment.fragment_id] = fragment

    def add_edge(self, fid_i: int, fid_j: int, weight: float) -> None:
        """Добавляет/обновляет ребро (берётся максимальный вес)."""
        key = frozenset({fid_i, fid_j})
        self.edges[key] = max(self.edges.get(key, 0.0), float(weight))

    # ── Базовые свойства ──────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def weight(self, fid_i: int, fid_j: int) -> float:
        """Вес ребра (0.0 если ребра нет)."""
        return self.edges.get(frozenset({fid_i, fid_j}), 0.0)

    def neighbors(self, fid: int) -> List[Tuple[int, float]]:
        """Возвращает [(fid_neighbor, weight), ...] для данного узла."""
        result = []
        for key, w in self.edges.items():
            if fid in key:
                other = next(iter(key - {fid}))
                result.append((other, w))
        return result

    def sorted_nodes(self) -> List[int]:
        return sorted(self.nodes)

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """
        Строит матрицу смежности (N×N) и соответствующий список fid.

        Returns:
            (A, fid_order) где A[i,j] = weight(fid_i, fid_j).
        """
        fids = self.sorted_nodes()
        n    = len(fids)
        idx  = {fid: i for i, fid in enumerate(fids)}
        A    = np.zeros((n, n), dtype=np.float64)
        for key, w in self.edges.items():
            ids = list(key)
            if len(ids) != 2:
                continue
            i, j = idx.get(ids[0]), idx.get(ids[1])
            if i is not None and j is not None:
                A[i, j] = w
                A[j, i] = w
        return A, fids

    def laplacian(self) -> Tuple[np.ndarray, List[int]]:
        """L = D - A (нормированный граф-лапласиан)."""
        A, fids = self.adjacency_matrix()
        D = np.diag(A.sum(axis=1))
        return D - A, fids

    def __repr__(self) -> str:
        return f"FragmentGraph(nodes={self.n_nodes}, edges={self.n_edges})"


# ─── GraphMatchResult ─────────────────────────────────────────────────────────

@dataclass
class GraphMatchResult:
    """
    Результат графового анализа.

    Attributes:
        mst_edges:      Рёбра MST — список (fid_i, fid_j, weight).
        mst_order:      DFS-порядок обхода MST (начиная от узла с макс. степенью).
        spectral_order: Порядок фрагментов по Fiedler-вектору.
        centrality:     {fragment_id: centrality_score}.
        graph:          Ссылка на исходный FragmentGraph.
    """
    mst_edges:      List[Tuple[int, int, float]]
    mst_order:      List[int]
    spectral_order: List[int]
    centrality:     Dict[int, float]
    graph:          FragmentGraph

    def summary(self) -> str:
        return (f"GraphMatchResult(nodes={self.graph.n_nodes}, "
                f"mst_edges={len(self.mst_edges)}, "
                f"spectral_order={self.spectral_order[:5]}...)")


# ─── Основные функции ─────────────────────────────────────────────────────────

def build_fragment_graph(fragments: List[Fragment],
                          entries:   List[CompatEntry],
                          threshold: float = 0.0) -> FragmentGraph:
    """
    Строит FragmentGraph из списка фрагментов и CompatEntry.

    Args:
        fragments:  Список Fragment.
        entries:    Список CompatEntry с оценками совместимости.
        threshold:  Минимальный score для включения ребра (0 = все рёбра).

    Returns:
        FragmentGraph.
    """
    graph = FragmentGraph()

    for frag in fragments:
        graph.add_node(frag)

    for e in entries:
        if e.score < threshold:
            continue
        fid_i = e.edge_i.edge_id // 10
        fid_j = e.edge_j.edge_id // 10
        if fid_i == fid_j:
            continue
        graph.add_edge(fid_i, fid_j, e.score)

    return graph


def mst_ordering(graph: FragmentGraph) -> List[int]:
    """
    Строит MST (Kruskal) и возвращает DFS-порядок обхода.

    Кружинь Крускала: сортируем рёбра по убыванию веса (высокая compat = близко),
    добавляем ребро если оно не создаёт цикл (Union-Find).

    Args:
        graph: FragmentGraph.

    Returns:
        Список fragment_id в порядке DFS-обхода MST.
    """
    fids = graph.sorted_nodes()
    if not fids:
        return []
    if len(fids) == 1:
        return list(fids)

    # Сортируем рёбра по убыванию веса
    sorted_edges = sorted(graph.edges.items(), key=lambda kv: kv[1], reverse=True)

    # Union-Find
    parent = {fid: fid for fid in fids}
    rank   = {fid: 0   for fid in fids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    # MST
    mst_adj: Dict[int, List[int]] = {fid: [] for fid in fids}
    for key, _w in sorted_edges:
        ids = list(key)
        if len(ids) != 2:
            continue
        a, b = ids[0], ids[1]
        if union(a, b):
            mst_adj[a].append(b)
            mst_adj[b].append(a)

    # DFS-обход от узла с наибольшей степенью
    start   = max(fids, key=lambda f: len(mst_adj[f]))
    order   = []
    visited = set()
    stack   = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in sorted(mst_adj[node], reverse=True):
            if neighbor not in visited:
                stack.append(neighbor)

    # Добавляем изолированные узлы
    for fid in fids:
        if fid not in visited:
            order.append(fid)

    return order


def spectral_ordering(graph: FragmentGraph) -> List[int]:
    """
    Упорядочивает фрагменты по Fiedler-вектору лапласиана графа.

    Fiedler-вектор = собственный вектор для второго наименьшего собственного
    значения L. Его компоненты отражают «расположение» узлов вдоль главного
    структурного разрыва графа.

    Args:
        graph: FragmentGraph.

    Returns:
        Список fragment_id, упорядоченных по Fiedler-вектору.
        Пустой граф → [].
    """
    fids = graph.sorted_nodes()
    n    = len(fids)

    if n == 0:
        return []
    if n == 1:
        return list(fids)
    if n == 2:
        return list(fids)

    L, fid_list = graph.laplacian()

    try:
        # Только нужные собственные векторы (быстрее для больших матриц)
        from scipy.linalg import eigh
        vals, vecs = eigh(L)
    except ImportError:
        vals, vecs = np.linalg.eigh(L)

    # Fiedler-вектор = второй (индекс 1) после нулевого
    fiedler = vecs[:, 1] if n > 1 else vecs[:, 0]

    # Сортировка по компоненте вектора
    order_idx = np.argsort(fiedler)
    return [fid_list[i] for i in order_idx]


def random_walk_similarity(graph:   FragmentGraph,
                             alpha:   float = 0.85,
                             n_iter:  int   = 30) -> np.ndarray:
    """
    Матрица «достижимости» через случайное блуждание с рестартом (PageRank).

    M[i, j] = стационарная вероятность достичь j из i за случайное блуждание.

    Args:
        graph:  FragmentGraph.
        alpha:  Вероятность продолжить блуждание (1 - вероятность рестарта).
        n_iter: Число итераций степенного метода.

    Returns:
        (N, N) float64 матрица, каждая строка нормирована (сумма = 1).
        Возвращает матрицу 0×0 для пустого графа.
    """
    fids = graph.sorted_nodes()
    n    = len(fids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    A, _ = graph.adjacency_matrix()
    # Нормируем строки → матрица переходов P
    row_sums = A.sum(axis=1, keepdims=True)
    P = np.where(row_sums > 0, A / row_sums, 1.0 / n)  # Равномерный fallback

    # PageRank для каждого стартового узла
    R = np.zeros((n, n), dtype=np.float64)
    teleport = np.eye(n, dtype=np.float64) / n

    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        for _ in range(n_iter):
            v = alpha * (v @ P) + (1.0 - alpha) * v[i]  # не совсем PageRank
        R[i] = v / (v.sum() + 1e-10)

    # Симметризация (приближённо)
    R = 0.5 * (R + R.T)
    # Нормировка строк
    row_s = R.sum(axis=1, keepdims=True)
    R = np.where(row_s > 0, R / row_s, 1.0 / n)
    return R


def degree_centrality(graph: FragmentGraph) -> Dict[int, float]:
    """
    Нормированная степень каждого узла (взвешенная сумма весов рёбер).

    centrality[fid] = Σ w(fid, neighbor) / max_possible_weight.

    Args:
        graph: FragmentGraph.

    Returns:
        {fragment_id: centrality} ∈ [0, 1].
    """
    if not graph.nodes:
        return {}

    total_weight: Dict[int, float] = {fid: 0.0 for fid in graph.nodes}
    for key, w in graph.edges.items():
        ids = list(key)
        if len(ids) != 2:
            continue
        total_weight[ids[0]] += w
        total_weight[ids[1]] += w

    max_w = max(total_weight.values()) if total_weight else 1.0
    if max_w == 0:
        max_w = 1.0

    return {fid: v / max_w for fid, v in total_weight.items()}


def analyze_graph(graph: FragmentGraph) -> GraphMatchResult:
    """
    Выполняет полный графовый анализ и возвращает GraphMatchResult.

    Args:
        graph: Построенный FragmentGraph.

    Returns:
        GraphMatchResult со всеми вычисленными полями.
    """
    # MST
    sorted_edges_kv = sorted(graph.edges.items(), key=lambda kv: kv[1],
                               reverse=True)
    mst_edges_list: List[Tuple[int, int, float]] = []
    for key, w in sorted_edges_kv:
        ids = list(key)
        if len(ids) == 2:
            mst_edges_list.append((ids[0], ids[1], w))

    mst_ord      = mst_ordering(graph)
    spectral_ord = spectral_ordering(graph)
    centrality   = degree_centrality(graph)

    return GraphMatchResult(
        mst_edges=mst_edges_list[:graph.n_nodes - 1],  # MST имеет N-1 рёбер
        mst_order=mst_ord,
        spectral_order=spectral_ord,
        centrality=centrality,
        graph=graph,
    )
