"""
Оптимизация порядка размещения фрагментов при сборке.

Предоставляет жадный планировщик, поиск наилучшего следующего фрагмента
и итеративную стратегию с удалением слабых мест.

Классы:
    PlacementResult — результат одного прохода оптимизации размещения

Функции:
    score_placement      — суммарный балл текущего состояния сборки
    find_best_next       — выбор фрагмента с наивысшим суммарным краевым баллом
    greedy_place         — жадное последовательное размещение всех фрагментов
    remove_worst_placed  — удаляет наименее выгодный фрагмент из сборки
    iterative_place      — итеративное улучшение жадной сборки
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
    remove_fragment,
    add_adjacency,
    compute_coverage,
    is_complete,
)


# ─── PlacementResult ──────────────────────────────────────────────────────────

@dataclass
class PlacementResult:
    """
    Результат оптимизации размещения фрагментов.

    Attributes:
        state:    Финальное состояние сборки (AssemblyState).
        score:    Суммарный балл сборки по матрице совместимости.
        n_placed: Количество размещённых фрагментов.
        history:  Список шагов [{step, idx, score_delta}, ...].
        params:   Параметры оптимизации.
    """
    state:    AssemblyState
    score:    float
    n_placed: int
    history:  List[Dict]       = field(default_factory=list)
    params:   Dict             = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"PlacementResult(n_placed={self.n_placed}, "
                f"score={self.score:.4f})")


# ─── score_placement ──────────────────────────────────────────────────────────

def score_placement(
    state:        AssemblyState,
    score_matrix: np.ndarray,
) -> float:
    """
    Вычисляет суммарный балл текущей сборки.

    Для каждой размещённой пары (i, j) с отношением смежности
    добавляет score_matrix[i, j] к общей сумме.
    Каждое ребро считается один раз (i < j).

    Args:
        state:        Текущее состояние AssemblyState.
        score_matrix: Матрица совместимости (N, N) float.

    Returns:
        Суммарный балл (float).
    """
    total = 0.0
    n     = score_matrix.shape[0]
    for i, neighbors in state.adjacency.items():
        for j in neighbors:
            if i < j and i < n and j < n:
                total += float(score_matrix[i, j])
    return total


# ─── find_best_next ───────────────────────────────────────────────────────────

def find_best_next(
    state:        AssemblyState,
    score_matrix: np.ndarray,
    candidates:   Optional[List[int]] = None,
) -> Tuple[int, float]:
    """
    Выбирает следующий фрагмент для размещения с наибольшей пользой.

    Полезность фрагмента idx = sum(score_matrix[idx, j])
    по всем j, уже размещённым.

    Args:
        state:        Текущее состояние (с уже размещёнными фрагментами).
        score_matrix: Матрица совместимости (N, N).
        candidates:   Список кандидатов (None → все незанятые).

    Returns:
        (best_idx, best_gain). Если нет кандидатов → (-1, 0.0).
    """
    placed_ids: Set[int] = set(state.placed.keys())
    n           = score_matrix.shape[0]

    if candidates is None:
        candidates = [i for i in range(n) if i not in placed_ids]

    if not candidates:
        return -1, 0.0

    best_idx   = candidates[0]
    best_gain  = -np.inf

    for idx in candidates:
        if idx in placed_ids:
            continue
        gain = sum(
            float(score_matrix[idx, j])
            for j in placed_ids if j < n
        )
        if gain > best_gain:
            best_gain = gain
            best_idx  = idx

    return best_idx, float(best_gain)


# ─── greedy_place ─────────────────────────────────────────────────────────────

def greedy_place(
    n_fragments:  int,
    score_matrix: np.ndarray,
    root:         int = 0,
) -> PlacementResult:
    """
    Жадное последовательное размещение фрагментов.

    Начинает с root, каждый следующий фрагмент выбирается
    как наиболее совместимый с уже размещёнными.

    Args:
        n_fragments:  Общее число фрагментов.
        score_matrix: Матрица совместимости (N, N).
        root:         Индекс первого фрагмента.

    Returns:
        PlacementResult с финальным состоянием и историей шагов.

    Raises:
        ValueError: Если n_fragments < 1 или root не в [0, n_fragments).
    """
    if n_fragments < 1:
        raise ValueError(f"n_fragments must be >= 1, got {n_fragments}.")
    if not (0 <= root < n_fragments):
        raise ValueError(f"root={root} is out of range [0, {n_fragments}).")

    state   = create_state(n_fragments)
    state   = place_fragment(state, root, position=(0.0, 0.0))
    history = [{"step": 0, "idx": root, "score_delta": 0.0}]

    for step in range(1, n_fragments):
        best_idx, gain = find_best_next(state, score_matrix)
        if best_idx < 0:
            break

        # Простая эвристика позиции: смещение по шагу
        pos   = (float(step), 0.0)
        state = place_fragment(state, best_idx, position=pos)

        # Добавить смежность со всеми уже размещёнными
        for j in list(state.placed.keys()):
            if j != best_idx:
                try:
                    state = add_adjacency(state, best_idx, j)
                except ValueError:
                    pass  # петля (не должна возникнуть)

        history.append({"step": step, "idx": best_idx, "score_delta": gain})

    total = score_placement(state, score_matrix)
    return PlacementResult(
        state=state,
        score=total,
        n_placed=len(state.placed),
        history=history,
        params={"root": root, "method": "greedy"},
    )


# ─── remove_worst_placed ──────────────────────────────────────────────────────

def remove_worst_placed(
    result:       PlacementResult,
    score_matrix: np.ndarray,
) -> PlacementResult:
    """
    Удаляет из сборки фрагмент с наименьшим вкладом в суммарный балл.

    Не трогает первый размещённый фрагмент (корень из history[0]).

    Args:
        result:       Текущий PlacementResult.
        score_matrix: Матрица совместимости.

    Returns:
        Новый PlacementResult без «слабейшего» фрагмента.
        Если размещён только 1 фрагмент, возвращает result без изменений.
    """
    if result.n_placed <= 1:
        return result

    state = result.state
    n     = score_matrix.shape[0]

    # Вклад фрагмента i = сумма score_matrix[i, j] по соседям j
    worst_idx  = -1
    worst_gain = np.inf

    root_idx = result.history[0]["idx"] if result.history else -1

    for idx in state.placed:
        if idx == root_idx:
            continue
        gain = sum(
            float(score_matrix[idx, j])
            for j in state.adjacency.get(idx, set())
            if j < n
        )
        if gain < worst_gain:
            worst_gain = gain
            worst_idx  = idx

    if worst_idx < 0:
        return result

    new_state = remove_fragment(state, worst_idx)
    new_score = score_placement(new_state, score_matrix)
    new_hist  = [h for h in result.history if h.get("idx") != worst_idx]

    return PlacementResult(
        state=new_state,
        score=new_score,
        n_placed=len(new_state.placed),
        history=new_hist,
        params=dict(result.params),
    )


# ─── iterative_place ──────────────────────────────────────────────────────────

def iterative_place(
    n_fragments:  int,
    score_matrix: np.ndarray,
    root:         int = 0,
    max_iter:     int = 10,
    patience:     int = 3,
) -> PlacementResult:
    """
    Итеративное улучшение жадной сборки.

    На каждой итерации: удаляет наименее выгодный фрагмент,
    затем заново дополняет сборку жадным шагом. Останавливается,
    если балл не улучшается patience итераций подряд.

    Args:
        n_fragments:  Общее число фрагментов.
        score_matrix: Матрица совместимости (N, N).
        root:         Индекс стартового фрагмента.
        max_iter:     Максимальное число итераций.
        patience:     Остановка после patience безуспешных итераций.

    Returns:
        Лучший PlacementResult за все итерации.
    """
    best = greedy_place(n_fragments, score_matrix, root=root)
    no_improve = 0

    for it in range(max_iter):
        # Удалить слабейший фрагмент
        candidate = remove_worst_placed(best, score_matrix)

        # Дополнить жадным шагом
        state = candidate.state
        placed_ids = set(state.placed.keys())
        remaining  = [i for i in range(n_fragments) if i not in placed_ids]

        for idx in remaining:
            _, gain = find_best_next(state, score_matrix,
                                     candidates=[idx])
            step = state.step
            pos  = (float(step + len(placed_ids)), 0.0)
            state = place_fragment(state, idx, position=pos)
            for j in list(state.placed.keys()):
                if j != idx:
                    try:
                        state = add_adjacency(state, idx, j)
                    except ValueError:
                        pass
            placed_ids.add(idx)

        new_score = score_placement(state, score_matrix)
        new_result = PlacementResult(
            state=state,
            score=new_score,
            n_placed=len(state.placed),
            history=candidate.history,
            params={"root": root, "method": "iterative", "iter": it},
        )

        if new_score > best.score:
            best       = new_result
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best
