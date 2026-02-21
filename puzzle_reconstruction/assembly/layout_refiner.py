"""Итеративное уточнение компоновки фрагментов.

Модуль реализует уточнение позиций фрагментов путём локальной оптимизации:
каждый фрагмент пошагово перемещается в направлении, увеличивающем
суммарную оценку соседних стыков.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── RefineConfig ─────────────────────────────────────────────────────────────

@dataclass
class RefineConfig:
    """Параметры уточнения компоновки.

    Атрибуты:
        max_iter:      Максимальное число итераций (>= 1).
        step_size:     Размер шага смещения в пикселях (> 0).
        convergence_eps: Минимальное суммарное смещение для продолжения (>= 0).
        frozen_ids:    Идентификаторы фрагментов, которые не двигаются.
    """

    max_iter: int = 20
    step_size: float = 1.0
    convergence_eps: float = 0.01
    frozen_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter должен быть >= 1, получено {self.max_iter}"
            )
        if self.step_size <= 0:
            raise ValueError(
                f"step_size должен быть > 0, получено {self.step_size}"
            )
        if self.convergence_eps < 0:
            raise ValueError(
                f"convergence_eps должен быть >= 0, "
                f"получено {self.convergence_eps}"
            )


# ─── FragmentPosition ─────────────────────────────────────────────────────────

@dataclass
class FragmentPosition:
    """Позиция фрагмента на холсте.

    Атрибуты:
        fragment_id: Идентификатор фрагмента.
        x:           Координата X (любое вещественное).
        y:           Координата Y (любое вещественное).
        rotation:    Угол поворота в градусах.
    """

    fragment_id: int
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Координаты (x, y)."""
        return (self.x, self.y)

    def distance_to(self, other: "FragmentPosition") -> float:
        """Евклидово расстояние до другой позиции."""
        return float(np.hypot(self.x - other.x, self.y - other.y))


# ─── RefineStep ───────────────────────────────────────────────────────────────

@dataclass
class RefineStep:
    """Итоговая информация об одной итерации уточнения.

    Атрибуты:
        iteration:     Номер итерации (>= 0).
        total_shift:   Суммарное смещение всех фрагментов на данной итерации.
        score_delta:   Изменение суммарной оценки (может быть < 0).
        n_moved:       Число реально сдвинутых фрагментов.
    """

    iteration: int
    total_shift: float
    score_delta: float
    n_moved: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"iteration >= 0, получено {self.iteration}"
            )
        if self.total_shift < 0:
            raise ValueError(
                f"total_shift >= 0, получено {self.total_shift}"
            )

    @property
    def improved(self) -> bool:
        """True если суммарная оценка возросла."""
        return self.score_delta > 0


# ─── RefineResult ─────────────────────────────────────────────────────────────

@dataclass
class RefineResult:
    """Результат процедуры уточнения.

    Атрибуты:
        positions:  {fragment_id: FragmentPosition} после уточнения.
        history:    Список RefineStep по итерациям.
        n_iter:     Фактическое число выполненных итераций.
        converged:  True если алгоритм сошёлся до исчерпания max_iter.
    """

    positions: Dict[int, FragmentPosition]
    history: List[RefineStep]
    n_iter: int
    converged: bool

    def __post_init__(self) -> None:
        if self.n_iter < 0:
            raise ValueError(
                f"n_iter >= 0, получено {self.n_iter}"
            )

    @property
    def total_shift(self) -> float:
        """Суммарное смещение за все итерации."""
        return sum(s.total_shift for s in self.history)

    @property
    def improved_iters(self) -> int:
        """Число итераций с положительным score_delta."""
        return sum(1 for s in self.history if s.improved)

    def get_position(self, fragment_id: int) -> Optional[FragmentPosition]:
        """Позиция фрагмента или None."""
        return self.positions.get(fragment_id)


# ─── _score_pair ─────────────────────────────────────────────────────────────

def _score_pair(
    pos_a: FragmentPosition,
    pos_b: FragmentPosition,
    adjacency_score: float,
    target_gap: float,
) -> float:
    """Оценить стык двух соседних фрагментов.

    Оценка убывает от adjacency_score при нулевом разрыве к 0
    по мере роста отклонения от target_gap.
    """
    dist = pos_a.distance_to(pos_b)
    deviation = abs(dist - target_gap)
    decay = np.exp(-deviation)
    return float(adjacency_score * decay)


# ─── compute_layout_score ─────────────────────────────────────────────────────

def compute_layout_score(
    positions: Dict[int, FragmentPosition],
    adjacency: Dict[Tuple[int, int], float],
    target_gap: float = 1.0,
) -> float:
    """Вычислить суммарную оценку текущей компоновки.

    Аргументы:
        positions:  {fragment_id: FragmentPosition}.
        adjacency:  {(a, b): score} — оценки соседних стыков.
        target_gap: Целевое расстояние между соседями (px, > 0).

    Возвращает:
        Суммарный балл компоновки (>= 0).
    """
    total = 0.0
    for (a, b), score in adjacency.items():
        pos_a = positions.get(a)
        pos_b = positions.get(b)
        if pos_a is None or pos_b is None:
            continue
        total += _score_pair(pos_a, pos_b, score, target_gap)
    return total


# ─── refine_layout ────────────────────────────────────────────────────────────

def refine_layout(
    positions: Dict[int, FragmentPosition],
    adjacency: Dict[Tuple[int, int], float],
    cfg: Optional[RefineConfig] = None,
    target_gap: float = 1.0,
) -> RefineResult:
    """Уточнить компоновку итеративно.

    На каждой итерации каждый незамороженный фрагмент пробует 4 шага
    (±x, ±y), принимает тот, что улучшает суммарную оценку, или остаётся
    на месте.

    Аргументы:
        positions:  Начальные позиции {fragment_id: FragmentPosition}.
        adjacency:  Матрица соседних стыков {(a, b): score}.
        cfg:        Параметры.
        target_gap: Целевой разрыв между соседями.

    Возвращает:
        RefineResult.
    """
    if cfg is None:
        cfg = RefineConfig()

    # Глубокая копия позиций
    current: Dict[int, FragmentPosition] = {
        fid: FragmentPosition(
            fragment_id=pos.fragment_id,
            x=pos.x,
            y=pos.y,
            rotation=pos.rotation,
        )
        for fid, pos in positions.items()
    }

    frozen = set(cfg.frozen_ids)
    movable = [fid for fid in current if fid not in frozen]

    history: List[RefineStep] = []
    converged = False

    current_score = compute_layout_score(current, adjacency, target_gap)

    for iteration in range(cfg.max_iter):
        total_shift = 0.0
        n_moved = 0
        prev_score = current_score

        for fid in movable:
            pos = current[fid]
            best_x, best_y = pos.x, pos.y
            best_score = compute_layout_score(current, adjacency, target_gap)

            for dx, dy in [(cfg.step_size, 0.0), (-cfg.step_size, 0.0),
                           (0.0, cfg.step_size), (0.0, -cfg.step_size)]:
                pos.x = best_x + dx
                pos.y = best_y + dy
                s = compute_layout_score(current, adjacency, target_gap)
                if s > best_score:
                    best_score = s
                    best_x, best_y = pos.x, pos.y

            shift = np.hypot(best_x - pos.x + (pos.x - current[fid].x),
                             best_y - pos.y + (pos.y - current[fid].y))
            # Применить лучшую найденную позицию
            old_x, old_y = current[fid].x, current[fid].y
            current[fid].x = best_x
            current[fid].y = best_y
            moved = np.hypot(best_x - old_x, best_y - old_y)
            total_shift += moved
            if moved > 1e-9:
                n_moved += 1

        current_score = compute_layout_score(current, adjacency, target_gap)
        score_delta = current_score - prev_score

        history.append(RefineStep(
            iteration=iteration,
            total_shift=max(total_shift, 0.0),
            score_delta=score_delta,
            n_moved=n_moved,
        ))

        if total_shift < cfg.convergence_eps:
            converged = True
            break

    return RefineResult(
        positions=current,
        history=history,
        n_iter=len(history),
        converged=converged,
    )


# ─── apply_offset ─────────────────────────────────────────────────────────────

def apply_offset(
    positions: Dict[int, FragmentPosition],
    dx: float,
    dy: float,
    fragment_ids: Optional[List[int]] = None,
) -> Dict[int, FragmentPosition]:
    """Сдвинуть позиции фрагментов на фиксированный вектор.

    Аргументы:
        positions:    {fragment_id: FragmentPosition}.
        dx:           Смещение по X.
        dy:           Смещение по Y.
        fragment_ids: Список фрагментов для сдвига (None = все).

    Возвращает:
        Новый словарь позиций.
    """
    ids = set(fragment_ids) if fragment_ids is not None else set(positions.keys())
    result: Dict[int, FragmentPosition] = {}
    for fid, pos in positions.items():
        if fid in ids:
            result[fid] = FragmentPosition(
                fragment_id=pos.fragment_id,
                x=pos.x + dx,
                y=pos.y + dy,
                rotation=pos.rotation,
            )
        else:
            result[fid] = FragmentPosition(
                fragment_id=pos.fragment_id,
                x=pos.x,
                y=pos.y,
                rotation=pos.rotation,
            )
    return result


# ─── compare_layouts ──────────────────────────────────────────────────────────

def compare_layouts(
    before: Dict[int, FragmentPosition],
    after: Dict[int, FragmentPosition],
) -> Dict[str, float]:
    """Сравнить две компоновки.

    Аргументы:
        before: Позиции до уточнения.
        after:  Позиции после уточнения.

    Возвращает:
        Словарь {"mean_shift", "max_shift", "n_moved"}.
    """
    common = set(before.keys()) & set(after.keys())
    if not common:
        return {"mean_shift": 0.0, "max_shift": 0.0, "n_moved": 0}

    shifts = [
        np.hypot(after[fid].x - before[fid].x,
                 after[fid].y - before[fid].y)
        for fid in common
    ]
    shifts_arr = np.array(shifts)
    return {
        "mean_shift": float(np.mean(shifts_arr)),
        "max_shift": float(np.max(shifts_arr)),
        "n_moved": int(np.sum(shifts_arr > 1e-9)),
    }
