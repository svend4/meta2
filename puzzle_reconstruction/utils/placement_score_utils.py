"""placement_score_utils — утилиты оценки размещения фрагментов.

Предоставляет конфигурацию, структуры данных и функции для анализа,
агрегации и сравнения результатов размещения фрагментов паззла на холсте.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class PlacementScoreConfig:
    """Параметры оценки и агрегации размещений.

    Attributes
    ----------
    min_score:
        Минимально приемлемый score [0, 1].
    coverage_weight:
        Вес покрытия (0..1) при итоговой агрегации.
    overlap_penalty_weight:
        Вес штрафа за перекрытие.
    prefer_full_placement:
        Если True, незаполненные холсты штрафуются.
    """
    min_score:             float = 0.0
    coverage_weight:       float = 0.5
    overlap_penalty_weight: float = 0.5
    prefer_full_placement: bool  = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError(
                f"min_score должен быть в [0, 1]: {self.min_score}"
            )
        if not (0.0 <= self.coverage_weight <= 1.0):
            raise ValueError(
                f"coverage_weight должен быть в [0, 1]: {self.coverage_weight}"
            )
        if not (0.0 <= self.overlap_penalty_weight <= 1.0):
            raise ValueError(
                f"overlap_penalty_weight должен быть в [0, 1]: "
                f"{self.overlap_penalty_weight}"
            )


# ─── Запись о размещении ──────────────────────────────────────────────────────

@dataclass
class PlacementScoreEntry:
    """Оценка одного шага размещения.

    Attributes
    ----------
    step:
        Номер шага размещения (≥ 0).
    fragment_idx:
        Индекс размещённого фрагмента.
    score_delta:
        Прирост score на данном шаге.
    cumulative_score:
        Накопленный score после шага.
    position:
        Позиция размещения (x, y).
    meta:
        Дополнительные данные.
    """
    step:             int
    fragment_idx:     int
    score_delta:      float
    cumulative_score: float
    position:         Tuple[float, float] = (0.0, 0.0)
    meta:             Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step не может быть отрицательным: {self.step}")
        if self.fragment_idx < 0:
            raise ValueError(
                f"fragment_idx не может быть отрицательным: {self.fragment_idx}"
            )

    def __repr__(self) -> str:
        return (f"PlacementScoreEntry(step={self.step}, "
                f"idx={self.fragment_idx}, "
                f"delta={self.score_delta:+.3f}, "
                f"cum={self.cumulative_score:.3f})")


# ─── Сводный результат ────────────────────────────────────────────────────────

@dataclass
class PlacementSummary:
    """Сводка серии оценок размещения.

    Attributes
    ----------
    entries:
        Упорядоченный список шагов.
    n_placed:
        Итоговое число размещённых фрагментов.
    final_score:
        Финальный score.
    mean_delta:
        Средний прирост score за шаг.
    max_delta:
        Наибольший прирост за один шаг.
    min_delta:
        Наименьший прирост (может быть отрицательным).
    """
    entries:     List[PlacementScoreEntry]
    n_placed:    int
    final_score: float
    mean_delta:  float
    max_delta:   float
    min_delta:   float

    def __post_init__(self) -> None:
        if self.n_placed < 0:
            raise ValueError(
                f"n_placed не может быть отрицательным: {self.n_placed}"
            )

    def __repr__(self) -> str:
        return (f"PlacementSummary(n={self.n_placed}, "
                f"final={self.final_score:.3f}, "
                f"mean_delta={self.mean_delta:+.3f})")


# ─── Конструирование записей ──────────────────────────────────────────────────

def make_placement_entry(
    step:             int,
    fragment_idx:     int,
    score_delta:      float,
    cumulative_score: float,
    position:         Tuple[float, float] = (0.0, 0.0),
    meta:             Optional[Dict] = None,
) -> PlacementScoreEntry:
    """Создать PlacementScoreEntry с проверкой.

    Returns
    -------
    PlacementScoreEntry
    """
    return PlacementScoreEntry(
        step=step,
        fragment_idx=fragment_idx,
        score_delta=float(score_delta),
        cumulative_score=float(cumulative_score),
        position=tuple(position),
        meta=dict(meta) if meta else {},
    )


def entries_from_history(
    history: List[Dict],
) -> List[PlacementScoreEntry]:
    """Преобразовать список history (из PlacementResult) в PlacementScoreEntry.

    Parameters
    ----------
    history:
        Список словарей вида {'step': int, 'idx': int, 'score_delta': float, ...}

    Returns
    -------
    Список PlacementScoreEntry (упорядоченный по step).
    """
    entries = []
    cum = 0.0
    for record in history:
        delta = float(record.get("score_delta", 0.0))
        cum += delta
        entries.append(PlacementScoreEntry(
            step=int(record.get("step", len(entries))),
            fragment_idx=int(record.get("idx", 0)),
            score_delta=delta,
            cumulative_score=cum,
            position=tuple(record.get("position", (0.0, 0.0))),
            meta={k: v for k, v in record.items()
                  if k not in ("step", "idx", "score_delta", "position")},
        ))
    return sorted(entries, key=lambda e: e.step)


# ─── Сводка ───────────────────────────────────────────────────────────────────

def summarise_placement(
    entries: List[PlacementScoreEntry],
) -> PlacementSummary:
    """Создать PlacementSummary по списку записей.

    Parameters
    ----------
    entries:
        Список PlacementScoreEntry.

    Returns
    -------
    PlacementSummary
    """
    if not entries:
        return PlacementSummary(
            entries=[], n_placed=0,
            final_score=0.0, mean_delta=0.0,
            max_delta=0.0, min_delta=0.0,
        )
    deltas = np.array([e.score_delta for e in entries], dtype=np.float64)
    return PlacementSummary(
        entries=list(entries),
        n_placed=len(entries),
        final_score=float(entries[-1].cumulative_score),
        mean_delta=float(deltas.mean()),
        max_delta=float(deltas.max()),
        min_delta=float(deltas.min()),
    )


# ─── Фильтрация и ранжирование ────────────────────────────────────────────────

def filter_positive_steps(
    entries: List[PlacementScoreEntry],
) -> List[PlacementScoreEntry]:
    """Оставить только шаги с положительным приростом."""
    return [e for e in entries if e.score_delta > 0.0]


def filter_by_min_score(
    entries:   List[PlacementScoreEntry],
    min_score: float = 0.0,
) -> List[PlacementScoreEntry]:
    """Оставить только шаги с накопленным score ≥ min_score."""
    return [e for e in entries if e.cumulative_score >= min_score]


def top_k_steps(
    entries: List[PlacementScoreEntry],
    k:       int,
) -> List[PlacementScoreEntry]:
    """Вернуть k шагов с наибольшим score_delta."""
    return sorted(entries, key=lambda e: e.score_delta, reverse=True)[:k]


def rank_fragments(
    entries: List[PlacementScoreEntry],
) -> List[Tuple[int, float]]:
    """Ранжировать фрагменты по score_delta убывания.

    Returns
    -------
    Список (fragment_idx, score_delta), отсортированный по убыванию.
    """
    return sorted(
        [(e.fragment_idx, e.score_delta) for e in entries],
        key=lambda x: x[1],
        reverse=True,
    )


# ─── Статистика ───────────────────────────────────────────────────────────────

def placement_score_stats(
    entries: List[PlacementScoreEntry],
) -> Dict:
    """Вычислить статистику по набору записей.

    Returns
    -------
    dict с ключами: 'n', 'final_score', 'mean_delta', 'std_delta',
                    'max_delta', 'min_delta', 'n_positive', 'n_negative'.
    """
    if not entries:
        return {
            "n": 0, "final_score": 0.0,
            "mean_delta": 0.0, "std_delta": 0.0,
            "max_delta": 0.0, "min_delta": 0.0,
            "n_positive": 0, "n_negative": 0,
        }
    deltas = np.array([e.score_delta for e in entries], dtype=np.float64)
    return {
        "n":           len(entries),
        "final_score": float(entries[-1].cumulative_score),
        "mean_delta":  float(deltas.mean()),
        "std_delta":   float(deltas.std()),
        "max_delta":   float(deltas.max()),
        "min_delta":   float(deltas.min()),
        "n_positive":  int((deltas > 0).sum()),
        "n_negative":  int((deltas < 0).sum()),
    }


# ─── Сравнение размещений ─────────────────────────────────────────────────────

def compare_placements(
    summary_a: PlacementSummary,
    summary_b: PlacementSummary,
) -> Dict:
    """Сравнить два PlacementSummary.

    Returns
    -------
    dict с ключами: 'score_diff', 'n_placed_diff', 'better'.
    """
    score_diff = summary_b.final_score - summary_a.final_score
    n_diff = summary_b.n_placed - summary_a.n_placed
    return {
        "score_diff":    score_diff,
        "n_placed_diff": n_diff,
        "better":        "b" if score_diff > 0 else ("a" if score_diff < 0 else "tie"),
    }


# ─── Пакетная обработка ───────────────────────────────────────────────────────

def batch_summarise(
    histories: List[List[Dict]],
) -> List[PlacementSummary]:
    """Пакетно преобразовать список history в PlacementSummary.

    Parameters
    ----------
    histories:
        Список history (каждый — список словарей шагов).

    Returns
    -------
    Список PlacementSummary.
    """
    return [summarise_placement(entries_from_history(h)) for h in histories]
