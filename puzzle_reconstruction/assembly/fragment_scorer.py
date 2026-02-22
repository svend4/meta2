"""Оценка качества размещения отдельных фрагментов в сборке.

Вычисляет локальные оценки совместимости каждого размещённого фрагмента
с его соседями и глобальную оценку всей сборки на основе матрицы стоимостей.

Экспортирует:
    FragmentScore      — оценка одного размещённого фрагмента
    AssemblyScore      — суммарная оценка состояния сборки
    ScoreConfig        — конфигурация весов
    score_fragment     — оценить один фрагмент по матрице стоимостей
    score_assembly     — оценить всё текущее состояние сборки
    top_k_placed       — топ-K фрагментов с наибольшей локальной оценкой
    bottom_k_placed    — K фрагментов с наименьшей локальной оценкой
    batch_score        — оценить список состояний сборки
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .assembly_state import AssemblyState
from .cost_matrix import CostMatrix


# ─── ScoreConfig ──────────────────────────────────────────────────────────────

@dataclass
class ScoreConfig:
    """Конфигурация весов для агрегации оценок.

    Атрибуты:
        neighbor_weight:  Вес оценки от соседей (>= 0).
        coverage_weight:  Вес оценки покрытия (>= 0).
        min_neighbors:    Минимальное число соседей для надёжной оценки.
    """
    neighbor_weight: float = 0.7
    coverage_weight: float = 0.3
    min_neighbors: int = 1

    def __post_init__(self) -> None:
        if self.neighbor_weight < 0.0:
            raise ValueError(
                f"neighbor_weight должен быть >= 0, получено {self.neighbor_weight}"
            )
        if self.coverage_weight < 0.0:
            raise ValueError(
                f"coverage_weight должен быть >= 0, получено {self.coverage_weight}"
            )
        if self.min_neighbors < 1:
            raise ValueError(
                f"min_neighbors должен быть >= 1, получено {self.min_neighbors}"
            )

    @property
    def total_weight(self) -> float:
        return self.neighbor_weight + self.coverage_weight


# ─── FragmentScore ────────────────────────────────────────────────────────────

@dataclass
class FragmentScore:
    """Оценка качества одного размещённого фрагмента.

    Атрибуты:
        fragment_idx:   Индекс фрагмента.
        local_score:    Средняя стоимость среди соседей (меньше = лучше).
        n_neighbors:    Число учтённых соседей.
        is_reliable:    True, если n_neighbors >= cfg.min_neighbors.
    """
    fragment_idx: int
    local_score: float
    n_neighbors: int
    is_reliable: bool = False

    def __post_init__(self) -> None:
        if self.fragment_idx < 0:
            raise ValueError(
                f"fragment_idx должен быть >= 0, получено {self.fragment_idx}"
            )
        if not (0.0 <= self.local_score <= 1.0):
            raise ValueError(
                f"local_score должен быть в [0, 1], получено {self.local_score}"
            )
        if self.n_neighbors < 0:
            raise ValueError(
                f"n_neighbors должен быть >= 0, получено {self.n_neighbors}"
            )


# ─── AssemblyScore ────────────────────────────────────────────────────────────

@dataclass
class AssemblyScore:
    """Суммарная оценка текущего состояния сборки.

    Атрибуты:
        global_score:    Взвешенная агрегированная оценка ∈ [0, 1].
        coverage:        Доля размещённых фрагментов ∈ [0, 1].
        mean_local:      Среднее local_score по всем фрагментам.
        fragment_scores: Детальные оценки каждого фрагмента.
        n_reliable:      Число фрагментов с is_reliable=True.
    """
    global_score: float
    coverage: float
    mean_local: float
    fragment_scores: Dict[int, FragmentScore] = field(default_factory=dict)
    n_reliable: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.global_score <= 1.0):
            raise ValueError(
                f"global_score должен быть в [0, 1], получено {self.global_score}"
            )
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError(
                f"coverage должен быть в [0, 1], получено {self.coverage}"
            )

    @property
    def n_placed(self) -> int:
        return len(self.fragment_scores)

    def summary(self) -> str:
        return (
            f"AssemblyScore(global={self.global_score:.3f}, "
            f"coverage={self.coverage:.2%}, "
            f"placed={self.n_placed})"
        )


# ─── score_fragment ───────────────────────────────────────────────────────────

def score_fragment(
    state: AssemblyState,
    fragment_idx: int,
    cm: CostMatrix,
    cfg: Optional[ScoreConfig] = None,
) -> FragmentScore:
    """Оценить один размещённый фрагмент по его соседям в матрице стоимостей.

    Оценка = среднее cost[fragment_idx, neighbor] по всем размещённым соседям.
    Если соседей нет — local_score = 0.5 (нейтральная оценка).

    Args:
        state:        Текущее состояние сборки.
        fragment_idx: Индекс оцениваемого фрагмента.
        cm:           Матрица стоимостей (n_fragments × n_fragments).
        cfg:          Конфигурация (None → ScoreConfig()).

    Returns:
        FragmentScore с вычисленными полями.

    Raises:
        ValueError: Если fragment_idx не размещён или выходит за диапазон.
        ValueError: Если n_fragments матрицы не совпадает с state.n_fragments.
    """
    if cfg is None:
        cfg = ScoreConfig()
    if cm.n_fragments != state.n_fragments:
        raise ValueError(
            f"cm.n_fragments ({cm.n_fragments}) != state.n_fragments ({state.n_fragments})"
        )
    if fragment_idx not in state.placed:
        raise ValueError(f"Фрагмент {fragment_idx} не размещён в state.")

    neighbors = [j for j in state.adjacency.get(fragment_idx, set())
                 if j in state.placed]
    n_neighbors = len(neighbors)

    if n_neighbors == 0:
        # Нет соседей — нейтральная оценка
        local_score = 0.5
    else:
        costs = [float(cm.matrix[fragment_idx, j]) for j in neighbors]
        local_score = float(np.mean(costs))
        local_score = min(1.0, max(0.0, local_score))

    return FragmentScore(
        fragment_idx=fragment_idx,
        local_score=local_score,
        n_neighbors=n_neighbors,
        is_reliable=n_neighbors >= cfg.min_neighbors,
    )


# ─── score_assembly ───────────────────────────────────────────────────────────

def score_assembly(
    state: AssemblyState,
    cm: CostMatrix,
    cfg: Optional[ScoreConfig] = None,
) -> AssemblyScore:
    """Оценить всё текущее состояние сборки.

    Args:
        state: Текущее состояние сборки.
        cm:    Матрица стоимостей.
        cfg:   Конфигурация весов.

    Returns:
        AssemblyScore.

    Raises:
        ValueError: Если n_fragments матриц не совпадают.
    """
    if cfg is None:
        cfg = ScoreConfig()
    if cm.n_fragments != state.n_fragments:
        raise ValueError(
            f"cm.n_fragments ({cm.n_fragments}) != state.n_fragments ({state.n_fragments})"
        )

    coverage = len(state.placed) / state.n_fragments if state.n_fragments > 0 else 0.0

    fragment_scores: Dict[int, FragmentScore] = {}
    for idx in state.placed:
        fragment_scores[idx] = score_fragment(state, idx, cm, cfg)

    if fragment_scores:
        mean_local = float(np.mean([fs.local_score for fs in fragment_scores.values()]))
    else:
        mean_local = 0.5

    n_reliable = sum(1 for fs in fragment_scores.values() if fs.is_reliable)

    # Глобальная оценка: инвертируем mean_local (меньше стоимость → лучше)
    # и добавляем вклад покрытия
    total_w = cfg.total_weight
    if total_w < 1e-12:
        global_score = 0.0
    else:
        neighbor_contrib = (1.0 - mean_local) * cfg.neighbor_weight
        coverage_contrib = coverage * cfg.coverage_weight
        global_score = (neighbor_contrib + coverage_contrib) / total_w
        global_score = min(1.0, max(0.0, global_score))

    return AssemblyScore(
        global_score=global_score,
        coverage=coverage,
        mean_local=mean_local,
        fragment_scores=fragment_scores,
        n_reliable=n_reliable,
    )


# ─── top_k_placed ─────────────────────────────────────────────────────────────

def top_k_placed(
    assembly_score: AssemblyScore,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Вернуть K фрагментов с наименьшей локальной стоимостью (лучшие).

    Args:
        assembly_score: Результат score_assembly.
        k:              Число возвращаемых фрагментов (>= 1).

    Returns:
        Список (fragment_idx, local_score), отсортированный по возрастанию.

    Raises:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    items = sorted(
        [(idx, fs.local_score) for idx, fs in assembly_score.fragment_scores.items()],
        key=lambda x: x[1],
    )
    return items[:k]


# ─── bottom_k_placed ──────────────────────────────────────────────────────────

def bottom_k_placed(
    assembly_score: AssemblyScore,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Вернуть K фрагментов с наибольшей локальной стоимостью (худшие).

    Args:
        assembly_score: Результат score_assembly.
        k:              Число возвращаемых фрагментов (>= 1).

    Returns:
        Список (fragment_idx, local_score), отсортированный по убыванию.

    Raises:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    items = sorted(
        [(idx, fs.local_score) for idx, fs in assembly_score.fragment_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return items[:k]


# ─── batch_score ──────────────────────────────────────────────────────────────

def batch_score(
    states: List[AssemblyState],
    cm: CostMatrix,
    cfg: Optional[ScoreConfig] = None,
) -> List[AssemblyScore]:
    """Оценить список состояний сборки.

    Args:
        states: Список AssemblyState.
        cm:     Матрица стоимостей (общая для всех).
        cfg:    Конфигурация весов.

    Returns:
        Список AssemblyScore той же длины.
    """
    if cfg is None:
        cfg = ScoreConfig()
    return [score_assembly(s, cm, cfg) for s in states]
