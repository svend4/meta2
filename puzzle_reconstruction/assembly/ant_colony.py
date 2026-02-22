"""
Муравьиный алгоритм (Ant Colony Optimization, ACO) для сборки документа.

Источник: адаптация ACO для задачи TSP (Dorigo et al., 1996) применительно
к восстановлению порядка фрагментов. Каждый «муравей» строит перестановку
фрагментов, руководствуясь феромонами и эвристикой (оценкой совместимости).

Ключевые компоненты:
    τ[i,j] — феромонная матрица: усиливается когда пара (i,j) даёт хороший
             результат, испаряется с каждой итерацией.
    η[i,j] — эвристика: score совместимости из CompatEntry.
    p[j|i] — вероятность перехода к фрагменту j от i:
             p[j|i] ∝ τ[i,j]^α · η[i,j]^β   (j не посещён)

Гиперпараметры:
    n_ants:      Число муравьёв (рекомендуется N .. 2N, где N = число фрагментов)
    n_iterations: Число итераций (по умолчанию 100)
    alpha:       Вес феромона (1.0–2.0)
    beta:        Вес эвристики (2.0–5.0)
    rho:         Коэффициент испарения феромона (0.1–0.5)
    Q:           Константа усиления феромона
    elite_count: Число «элитных» муравьёв (дополнительно усиливают феромон)

Преимущества перед SA/Genetic:
    - Явно накапливает информацию о хороших парах → быстро находит
      «локально уверенные» стыки.
    - Распределённый поиск без зависимости от начального решения.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import Fragment, Assembly, CompatEntry


def ant_colony_assembly(fragments:    List[Fragment],
                         entries:      List[CompatEntry],
                         n_ants:       int   = 0,
                         n_iterations: int   = 100,
                         alpha:        float = 1.0,
                         beta:         float = 3.0,
                         rho:          float = 0.2,
                         Q:            float = 1.0,
                         elite_count:  int   = 3,
                         allow_rotation: bool = True,
                         seed:         int   = 42) -> Assembly:
    """
    Оптимизирует расстановку фрагментов алгоритмом ACO.

    Args:
        fragments:     Список фрагментов.
        entries:       Список CompatEntry (чем выше score, тем лучше стык).
        n_ants:        Число муравьёв. 0 → автоматически = max(10, N).
        n_iterations:  Число итераций.
        alpha:         Влияние феромона на вероятность выбора.
        beta:          Влияние эвристики на вероятность выбора.
        rho:           Скорость испарения феромона (0 < rho < 1).
        Q:             Константа усиления феромона.
        elite_count:   Число элитных муравьёв для дополнительного усиления.
        allow_rotation: Разрешить ли учёт поворотов в оценке.
        seed:          Random seed.

    Returns:
        Assembly с лучшей найденной конфигурацией.
    """
    if not fragments:
        return Assembly(fragments=[], placements={}, compat_matrix=np.array([]))

    rng      = np.random.RandomState(seed)
    frag_ids = [f.fragment_id for f in fragments]
    n        = len(frag_ids)
    idx_map  = {fid: i for i, fid in enumerate(frag_ids)}  # frag_id → matrix_index

    if n_ants <= 0:
        n_ants = max(10, n)

    # ── Матрица эвристики η[i,j] ─────────────────────────────────────────
    eta = _build_eta_matrix(fragments, entries, n, idx_map)

    # ── Феромонная матрица τ[i,j] (инициализация единицами) ──────────────
    tau = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(tau, 0.0)

    best_order:  Optional[List[int]] = None
    best_score:  float               = -1.0
    best_angles: Optional[np.ndarray] = None

    rotations = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])

    # ── Основной цикл итераций ────────────────────────────────────────────
    for iteration in range(n_iterations):
        ant_solutions: List[Tuple[List[int], np.ndarray, float]] = []

        # Каждый муравей строит перестановку
        for _ in range(n_ants):
            order, angles, score = _build_ant_solution(
                n, frag_ids, tau, eta, alpha, beta,
                eta, rotations, rng, allow_rotation,
            )
            ant_solutions.append((order, angles, score))

            if score > best_score:
                best_score  = score
                best_order  = list(order)
                best_angles = angles.copy()

        # Испарение феромона
        tau *= (1.0 - rho)
        tau  = np.maximum(tau, 1e-10)

        # Усиление феромона на маршрутах всех муравьёв
        ant_solutions.sort(key=lambda x: x[2], reverse=True)

        for rank, (order, _, score) in enumerate(ant_solutions):
            delta = Q * score / (1.0 + rank)
            for k in range(n - 1):
                i = idx_map[order[k]]
                j = idx_map[order[k + 1]]
                tau[i, j] += delta
                tau[j, i] += delta  # Симметричность

        # Дополнительное усиление для элитных муравьёв
        for elite_idx in range(min(elite_count, len(ant_solutions))):
            order, _, score = ant_solutions[elite_idx]
            delta_elite = Q * score * elite_count
            for k in range(n - 1):
                i = idx_map[order[k]]
                j = idx_map[order[k + 1]]
                tau[i, j] += delta_elite
                tau[j, i] += delta_elite

    # ── Формируем Assembly из лучшего решения ─────────────────────────────
    if best_order is None:
        best_order  = frag_ids[:]
        best_angles = np.zeros(n)

    return _order_to_assembly(best_order, best_angles, fragments, best_score)


# ─── Построение решения (один муравей) ───────────────────────────────────────

def _build_ant_solution(n:         int,
                         frag_ids:  List[int],
                         tau:       np.ndarray,
                         eta:       np.ndarray,
                         alpha:     float,
                         beta:      float,
                         score_mat: np.ndarray,
                         rotations: np.ndarray,
                         rng:       np.random.RandomState,
                         allow_rotation: bool,
                         ) -> Tuple[List[int], np.ndarray, float]:
    """
    Один муравей строит перестановку, руководствуясь τ и η.

    Returns:
        (order: List[frag_id], angles: np.ndarray(N), total_score: float)
    """
    visited = [False] * n
    order   = []
    angles  = np.zeros(n)

    # Случайная стартовая вершина
    start = rng.randint(0, n)
    order.append(start)
    visited[start] = True

    for step in range(1, n):
        current = order[-1]

        # Вероятности перехода
        unvisited = [j for j in range(n) if not visited[j]]
        weights   = np.array([
            (tau[current, j] ** alpha) * (eta[current, j] ** beta)
            for j in unvisited
        ], dtype=np.float64)

        w_sum = weights.sum()
        if w_sum < 1e-12:
            weights = np.ones(len(unvisited))
            w_sum   = float(len(unvisited))

        probs   = weights / w_sum
        chosen  = unvisited[rng.choice(len(unvisited), p=probs)]
        order.append(chosen)
        visited[chosen] = True

        if allow_rotation:
            angles[step] = rng.choice(rotations)

    # Переводим индексы → fragment_id
    order_ids = [frag_ids[i] for i in order]

    # Считаем score решения
    total = sum(
        float(score_mat[order[k], order[k + 1]])
        for k in range(n - 1)
    )

    return order_ids, angles, total


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _build_eta_matrix(fragments: List[Fragment],
                       entries:   List[CompatEntry],
                       n:         int,
                       idx_map:   Dict[int, int]) -> np.ndarray:
    """
    Строит матрицу эвристики η[i,j] из списка CompatEntry.

    η[i,j] = max score между всеми рёбрами фрагментов i и j.
    Самостыки (i==j) и отсутствующие пары → η = 1e-6 (малая ненулевая константа).
    """
    eta = np.full((n, n), 1e-6, dtype=np.float64)

    edge_to_frag: Dict[int, int] = {
        e.edge_id: f.fragment_id
        for f in fragments
        for e in f.edges
    }

    for entry in entries:
        fi = edge_to_frag.get(entry.edge_i.edge_id)
        fj = edge_to_frag.get(entry.edge_j.edge_id)
        if fi is None or fj is None or fi == fj:
            continue
        ii = idx_map.get(fi)
        ij = idx_map.get(fj)
        if ii is None or ij is None:
            continue
        score = float(entry.score)
        if score > eta[ii, ij]:
            eta[ii, ij] = score
            eta[ij, ii] = score

    np.fill_diagonal(eta, 0.0)
    return eta


def _order_to_assembly(order:     List[int],
                        angles:    np.ndarray,
                        fragments: List[Fragment],
                        total_score: float) -> Assembly:
    """Конвертирует порядок (список frag_id) в объект Assembly."""
    frag_map = {f.fragment_id: f for f in fragments}
    spacing  = 120.0
    n_cols   = max(1, math.ceil(math.sqrt(len(order))))
    placements: Dict[int, Tuple[np.ndarray, float]] = {}

    for idx, fid in enumerate(order):
        col = idx % n_cols
        row = idx // n_cols
        pos = np.array([col * spacing, row * spacing])
        placements[fid] = (pos, float(angles[idx]))

    # Добавляем фрагменты без позиции (если order неполный)
    for frag in fragments:
        if frag.fragment_id not in placements:
            placements[frag.fragment_id] = (np.zeros(2), 0.0)

    return Assembly(
        fragments=fragments,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=total_score,
    )
