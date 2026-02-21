"""
Оценка абсолютных позиций фрагментов по попарным смещениям.

Использует граф смещений (offset graph) для восстановления координат
каждого фрагмента методом обхода в ширину (BFS). Опциональный шаг уточнения
усредняет позицию из нескольких независимых путей в графе.

Классы:
    PositionEstimate — оценка положения одного фрагмента

Функции:
    build_offset_graph   — строит словарь смежности с попарными смещениями
    estimate_positions   — BFS-оценка позиций из графа
    refine_positions     — уточнение позиций усреднением по всем путям
    positions_to_array   — словарь позиций → numpy-массив (N, 2)
    align_to_origin      — сдвиг позиций так, чтобы минимум оказался в (0, 0)
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── PositionEstimate ─────────────────────────────────────────────────────────

@dataclass
class PositionEstimate:
    """
    Оценка положения одного фрагмента в координатной плоскости.

    Attributes:
        idx:           Индекс фрагмента.
        x:             Горизонтальная координата (пикселей или условных единиц).
        y:             Вертикальная координата.
        confidence:    Уверенность ∈ [0, 1]. 1.0 по умолчанию (прямое смещение).
        n_constraints: Количество ограничений (рёбер), использованных для оценки.
    """
    idx:           int
    x:             float
    y:             float
    confidence:    float = 1.0
    n_constraints: int   = 0

    def __repr__(self) -> str:
        return (f"PositionEstimate(idx={self.idx}, "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"conf={self.confidence:.2f})")


# ─── build_offset_graph ───────────────────────────────────────────────────────

def build_offset_graph(
    pairs:   List[Tuple[int, int]],
    offsets: List[Tuple[float, float]],
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Строит неориентированный граф попарных смещений.

    Для каждой пары (idx1, idx2) с смещением (dx, dy) добавляет ребро:
        idx1 → idx2: (dx,  dy)
        idx2 → idx1: (-dx, -dy)

    Args:
        pairs:   Список пар индексов [(idx1, idx2), ...].
        offsets: Список соответствующих смещений [(dx, dy), ...].

    Returns:
        Словарь {idx: [(neighbor, dx, dy), ...]}.

    Raises:
        ValueError: Если длины pairs и offsets не совпадают.
    """
    if len(pairs) != len(offsets):
        raise ValueError(
            f"pairs and offsets must have the same length, "
            f"got {len(pairs)} and {len(offsets)}."
        )
    graph: Dict[int, List[Tuple[int, float, float]]] = {}
    for (idx1, idx2), (dx, dy) in zip(pairs, offsets):
        graph.setdefault(idx1, []).append((idx2, float(dx), float(dy)))
        graph.setdefault(idx2, []).append((idx1, float(-dx), float(-dy)))
    return graph


# ─── estimate_positions ───────────────────────────────────────────────────────

def estimate_positions(
    offset_graph: Dict[int, List[Tuple[int, float, float]]],
    root:         Optional[int] = None,
) -> Dict[int, PositionEstimate]:
    """
    Оценивает абсолютные позиции фрагментов методом BFS.

    Выбирает корневой фрагмент (наибольшее число рёбер или root),
    помещает его в (0, 0) и распространяет позиции по смещениям.

    Args:
        offset_graph: Граф смещений из build_offset_graph.
        root:         Индекс корневого фрагмента. None → выбирается
                      фрагмент с наибольшим количеством соседей.

    Returns:
        Словарь {idx: PositionEstimate}.
        Изолированные вершины (не в BFS-дереве) не включаются в результат.
    """
    if not offset_graph:
        return {}

    # Выбор корня
    if root is None or root not in offset_graph:
        root = max(offset_graph, key=lambda k: len(offset_graph[k]))

    positions: Dict[int, PositionEstimate] = {}
    visited:   set                          = set()
    queue:     deque                        = deque()

    positions[root] = PositionEstimate(idx=root, x=0.0, y=0.0,
                                       confidence=1.0, n_constraints=0)
    visited.add(root)
    queue.append(root)

    while queue:
        current = queue.popleft()
        cx      = positions[current].x
        cy      = positions[current].y

        for neighbor, dx, dy in offset_graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                positions[neighbor] = PositionEstimate(
                    idx=neighbor,
                    x=cx + dx,
                    y=cy + dy,
                    confidence=1.0,
                    n_constraints=1,
                )
                queue.append(neighbor)

    return positions


# ─── refine_positions ─────────────────────────────────────────────────────────

def refine_positions(
    offset_graph:    Dict[int, List[Tuple[int, float, float]]],
    initial:         Dict[int, PositionEstimate],
) -> Dict[int, PositionEstimate]:
    """
    Уточняет позиции усреднением предсказаний от всех соседей.

    Для каждого фрагмента i собирает оценки его позиции от каждого
    уже-оценённого соседа j: pos_i = pos_j + (-offset_j→i). Итоговая
    позиция — среднее всех оценок. Уверенность = 1 / (1 + std).

    Args:
        offset_graph: Граф смещений.
        initial:      Начальные позиции (из estimate_positions).

    Returns:
        Новый словарь с уточнёнными PositionEstimate.
    """
    refined: Dict[int, PositionEstimate] = {}

    for idx in initial:
        xs: List[float] = []
        ys: List[float] = []

        for neighbor, dx, dy in offset_graph.get(idx, []):
            if neighbor in initial:
                # позиция idx = позиция neighbor - смещение neighbor→idx
                # (смещение idx→neighbor = (dx, dy), значит neighbor→idx = (-dx, -dy)
                #  pos[idx] = pos[neighbor] + (-(-dx)) ... нет:
                #  если edge idx→neighbor имеет (dx,dy), то pos[neighbor] = pos[idx] + (dx,dy)
                #  → pos[idx] = pos[neighbor] - (dx, dy) ... но в offset_graph
                #  edge neighbor→idx имеет (-dx, -dy)
                #  pos[idx] = pos[neighbor] + (-dx_rev) где dx_rev = -dx
                #  = pos[neighbor] + dx ... это правильно?
                # Пересчёт: ребро idx→neighbor: offset = (dx,dy)
                #           ребро neighbor→idx: offset = (-dx,-dy)
                # pos[neighbor] = pos[idx] + dx,dy → pos[idx] = pos[neighbor] - dx,dy
                # Используем initial[neighbor] и dx,dy (ребро idx→neighbor)
                xs.append(initial[neighbor].x - dx)
                ys.append(initial[neighbor].y - dy)

        if xs:
            mean_x    = float(np.mean(xs))
            mean_y    = float(np.mean(ys))
            std_xy    = float(np.std(xs + ys))
            conf      = float(np.clip(1.0 / (1.0 + std_xy), 0.0, 1.0))
            n_constr  = len(xs)
        else:
            # Изолированная вершина — берём из initial
            mean_x   = initial[idx].x
            mean_y   = initial[idx].y
            conf     = initial[idx].confidence
            n_constr = initial[idx].n_constraints

        refined[idx] = PositionEstimate(
            idx=idx, x=mean_x, y=mean_y,
            confidence=conf, n_constraints=n_constr,
        )

    return refined


# ─── positions_to_array ───────────────────────────────────────────────────────

def positions_to_array(
    positions: Dict[int, PositionEstimate],
    n:         Optional[int] = None,
) -> np.ndarray:
    """
    Преобразует словарь позиций в массив (N, 2) float32.

    Args:
        positions: Словарь {idx: PositionEstimate}.
        n:         Ожидаемое количество фрагментов. Если задано и некоторые
                   индексы отсутствуют, их строки заполняются NaN.
                   Если None, массив строится только из присутствующих idx.

    Returns:
        np.ndarray shape (N, 2), dtype float32, где колонки — [x, y].
    """
    if n is None:
        if not positions:
            return np.empty((0, 2), dtype=np.float32)
        indices = sorted(positions.keys())
        arr = np.array([[positions[i].x, positions[i].y] for i in indices],
                        dtype=np.float32)
        return arr

    arr = np.full((n, 2), np.nan, dtype=np.float32)
    for idx, pe in positions.items():
        if 0 <= idx < n:
            arr[idx, 0] = pe.x
            arr[idx, 1] = pe.y
    return arr


# ─── align_to_origin ──────────────────────────────────────────────────────────

def align_to_origin(
    positions: Dict[int, PositionEstimate],
) -> Dict[int, PositionEstimate]:
    """
    Сдвигает все позиции так, чтобы минимальные x и y стали равны 0.

    Args:
        positions: Словарь {idx: PositionEstimate}.

    Returns:
        Новый словарь с откорректированными позициями.
        Уверенность и n_constraints сохраняются без изменений.
    """
    if not positions:
        return {}

    min_x = min(pe.x for pe in positions.values())
    min_y = min(pe.y for pe in positions.values())

    result: Dict[int, PositionEstimate] = {}
    for idx, pe in positions.items():
        result[idx] = PositionEstimate(
            idx=idx,
            x=pe.x - min_x,
            y=pe.y - min_y,
            confidence=pe.confidence,
            n_constraints=pe.n_constraints,
        )
    return result
