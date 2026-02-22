"""
Построение матрицы совместимости всех краёв всех фрагментов.
"""
import numpy as np
from typing import List
from ..models import Fragment, CompatEntry
from .pairwise import match_score


def build_compat_matrix(fragments: List[Fragment],
                        threshold: float = 0.0) -> tuple[np.ndarray, list[CompatEntry]]:
    """
    Строит матрицу совместимости всех пар краёв.

    Args:
        fragments:  Список фрагментов с заполненными edges.
        threshold:  Порог: пары с оценкой ниже threshold не включаются в список.

    Returns:
        (matrix, entries)
        matrix:  (N_edges, N_edges) float32 — оценки совместимости.
        entries: Список CompatEntry выше threshold, отсортированный по убыванию score.
    """
    # Собираем все края с их глобальными индексами
    all_edges = []
    frag_of_edge = []  # frag_id для каждого края

    for frag in fragments:
        for edge in frag.edges:
            all_edges.append(edge)
            frag_of_edge.append(frag.fragment_id)

    N = len(all_edges)
    matrix  = np.zeros((N, N), dtype=np.float32)
    entries = []

    for i in range(N):
        for j in range(i + 1, N):
            # Нельзя совмещать края одного фрагмента
            if frag_of_edge[i] == frag_of_edge[j]:
                continue

            entry = match_score(all_edges[i], all_edges[j])
            matrix[i, j] = entry.score
            matrix[j, i] = entry.score

            if entry.score >= threshold:
                entries.append(entry)

    entries.sort(key=lambda e: e.score, reverse=True)
    return matrix, entries


def top_candidates(matrix: np.ndarray,
                   all_edges: list,
                   edge_idx: int,
                   k: int = 5) -> list:
    """
    Возвращает топ-k кандидатов для заданного края.
    """
    row = matrix[edge_idx].copy()
    row[edge_idx] = 0  # Исключаем сам себя
    top_idx = np.argsort(row)[::-1][:k]
    return [(int(idx), float(row[idx])) for idx in top_idx if row[idx] > 0]
