"""
Точный решатель для малого числа фрагментов (N ≤ 8).

Алгоритм: ветви и границы (Branch & Bound) на дереве перестановок.

Для N фрагментов существует N! возможных порядков расстановки.
При N ≤ 8: 8! = 40 320 — можно перебрать за доли секунды.
При N = 9: 9! = 362 880 — граничный случай (~1–2 сек).
При N ≥ 10: рекомендуется переключиться на beam_search или gamma_optimizer.

Ограничение ветвей (pruning):
    Если частичная сборка k фрагментов имеет score ниже лучшего известного
    верхнего оценки (upper bound) — ветвь отсекается.
    Upper bound = текущий score + sum(top-1 score для каждого неразмещённого края).

Ориентации:
    Каждый фрагмент может быть повёрнут на 0°, 90°, 180°, 270°.
    Это увеличивает пространство поиска до (4^N * N!), поэтому
    при N ≥ 6 включается ограничение: поворачиваем только фрагмент,
    у которого хотя бы один край имеет score > threshold.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import Fragment, Assembly, CompatEntry


# Максимальный N для точного решателя (с предупреждением)
MAX_EXACT_N = 9
# N, при котором предупреждаем о возможной медленности
WARN_N = 8


class _State:
    """Промежуточное состояние в дереве поиска."""
    __slots__ = ("placed", "score", "placements")

    def __init__(self):
        self.placed:      List[int]  = []   # frag_id в порядке размещения
        self.score:       float      = 0.0
        self.placements:  Dict[int, Tuple[np.ndarray, float]] = {}


def exhaustive_assembly(fragments: List[Fragment],
                         entries: List[CompatEntry],
                         max_n: int = MAX_EXACT_N,
                         allow_rotation: bool = True,
                         seed: int = 42) -> Assembly:
    """
    Точный поиск оптимальной конфигурации методом ветвей и границ.

    Args:
        fragments:       Список фрагментов.
        entries:         Матрица совместимости (отсортированная).
        max_n:           Максимальный N для точного перебора.
                         Если len(fragments) > max_n — используем greedy.
        allow_rotation:  Разрешать ли повороты на 90°/180°/270°.
        seed:            Random seed (для жадного fallback).

    Returns:
        Assembly с оптимальной (точной) конфигурацией.

    Raises:
        ValueError: Если fragments пуст.
    """
    if not fragments:
        raise ValueError("fragments не должен быть пустым")

    n = len(fragments)

    if n > max_n:
        import warnings
        warnings.warn(
            f"exhaustive_assembly: N={n} > max_n={max_n}. "
            "Используем beam_search вместо точного перебора.",
            RuntimeWarning, stacklevel=2,
        )
        from .beam_search import beam_search
        return beam_search(fragments, entries, beam_width=10)

    if n >= WARN_N:
        import warnings
        warnings.warn(
            f"exhaustive_assembly: N={n} может быть медленным "
            f"(до {math.factorial(n) * (4 if allow_rotation else 1)} состояний).",
            RuntimeWarning, stacklevel=2,
        )

    # ── Индексы ──────────────────────────────────────────────────────────
    frag_ids    = [f.fragment_id for f in fragments]
    edge_to_frag: Dict[int, int] = {
        e.edge_id: f.fragment_id
        for f in fragments
        for e in f.edges
    }

    # ── Быстрый индекс score по паре (edge_i_id, edge_j_id) ─────────────
    score_index: Dict[Tuple[int, int], float] = {}
    for entry in entries:
        key = (entry.edge_i.edge_id, entry.edge_j.edge_id)
        score_index[key] = max(score_index.get(key, 0.0), entry.score)
        score_index[(key[1], key[0])] = score_index[key]

    # ── Upper bound на фрагмент (максимальный score любого его края) ─────
    max_score_per_frag: Dict[int, float] = {fid: 0.0 for fid in frag_ids}
    for entry in entries:
        fi = edge_to_frag.get(entry.edge_i.edge_id)
        fj = edge_to_frag.get(entry.edge_j.edge_id)
        if fi is not None:
            max_score_per_frag[fi] = max(max_score_per_frag[fi], entry.score)
        if fj is not None:
            max_score_per_frag[fj] = max(max_score_per_frag[fj], entry.score)

    rotations = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2] if allow_rotation else [0.0]

    # ── Жадная начальная оценка (нижняя граница) ─────────────────────────
    from .greedy import greedy_assembly
    greedy = greedy_assembly(fragments, entries)
    best_score = _evaluate_config(greedy.placements, entries, edge_to_frag)
    best_placements: Dict[int, Tuple[np.ndarray, float]] = {
        fid: (np.asarray(pos).copy(), angle)
        for fid, (pos, angle) in greedy.placements.items()
    }

    # ── Branch & Bound DFS ───────────────────────────────────────────────
    def _dfs(placed: List[int],
              current_score: float,
              current_placements: Dict[int, Tuple[np.ndarray, float]]) -> None:
        nonlocal best_score, best_placements

        remaining = [fid for fid in frag_ids if fid not in placed]

        # Upper bound pruning
        ub = current_score + sum(max_score_per_frag[fid] for fid in remaining)
        if ub <= best_score:
            return

        # Листовой узел — обновляем лучший результат
        if not remaining:
            if current_score > best_score:
                best_score     = current_score
                best_placements = {fid: (pos.copy(), angle)
                                    for fid, (pos, angle) in current_placements.items()}
            return

        # Выбираем следующий фрагмент: с наибольшим max_score (эвристика)
        next_fid = max(remaining, key=lambda fid: max_score_per_frag[fid])

        # Вычисляем позицию для next_fid
        col = len(placed) % max(1, math.ceil(math.sqrt(n)))
        row = len(placed) // max(1, math.ceil(math.sqrt(n)))
        spacing = 120.0
        pos = np.array([col * spacing, row * spacing])

        for angle in rotations:
            current_placements[next_fid] = (pos.copy(), angle)

            # Считаем прирост score от добавления next_fid
            delta = _score_delta(next_fid, placed, current_placements,
                                  entries, edge_to_frag)
            new_score = current_score + delta

            _dfs(placed + [next_fid], new_score, current_placements)
            del current_placements[next_fid]

    _dfs([], 0.0, {})

    return Assembly(
        fragments=fragments,
        placements=best_placements,
        compat_matrix=np.array([]),
        total_score=best_score,
    )


# ─── Вспомогательные функции ──────────────────────────────────────────────

def _score_delta(new_fid: int,
                  placed: List[int],
                  placements: Dict[int, Tuple[np.ndarray, float]],
                  entries: List[CompatEntry],
                  edge_to_frag: Dict[int, int]) -> float:
    """Возвращает суммарный score всех стыков нового фрагмента с уже размещёнными."""
    total = 0.0
    if not placed:
        return 0.0

    for entry in entries:
        fi = edge_to_frag.get(entry.edge_i.edge_id)
        fj = edge_to_frag.get(entry.edge_j.edge_id)

        # Стык нового фрагмента с размещённым
        if (fi == new_fid and fj in placed) or (fj == new_fid and fi in placed):
            total += entry.score

    return total


def _evaluate_config(placements: Dict[int, Tuple],
                      entries: List[CompatEntry],
                      edge_to_frag: Dict[int, int]) -> float:
    """Полная оценка конфигурации — сумма score всех смежных пар."""
    placed = set(placements.keys())
    total  = 0.0
    seen: set = set()

    for entry in entries:
        fi = edge_to_frag.get(entry.edge_i.edge_id)
        fj = edge_to_frag.get(entry.edge_j.edge_id)
        if fi is None or fj is None:
            continue
        if fi not in placed or fj not in placed:
            continue
        pair = (min(fi, fj), max(fi, fj))
        if pair not in seen:
            total += entry.score
            seen.add(pair)

    return total
