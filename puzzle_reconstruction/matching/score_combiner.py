"""Комбинирование оценок совпадения из нескольких источников.

Модуль предоставляет функции для агрегации скоров совпадения фрагментов
(взвешенное среднее, минимум, максимум, голосование, ранговое слияние)
и нормализацию перед объединением.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ─── ScoreVector ─────────────────────────────────────────────────────────────

@dataclass
class ScoreVector:
    """Набор оценок от нескольких матчеров для одной пары фрагментов.

    Атрибуты:
        idx1:    Индекс первого фрагмента (>= 0).
        idx2:    Индекс второго фрагмента (>= 0).
        scores:  Словарь {название_матчера: оценка ∈ [0, 1]}.
        params:  Дополнительные параметры.
    """

    idx1: int
    idx2: int
    scores: Dict[str, float]
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0:
            raise ValueError(f"idx1 должен быть >= 0, получено {self.idx1}")
        if self.idx2 < 0:
            raise ValueError(f"idx2 должен быть >= 0, получено {self.idx2}")
        for name, val in self.scores.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Оценка '{name}' должна быть в [0, 1], получено {val}"
                )

    @property
    def pair(self):
        """Пара индексов (idx1, idx2)."""
        return (self.idx1, self.idx2)

    def __len__(self) -> int:
        return len(self.scores)


# ─── CombinedScore ────────────────────────────────────────────────────────────

@dataclass
class CombinedScore:
    """Итоговая комбинированная оценка для пары фрагментов.

    Атрибуты:
        idx1:          Индекс первого фрагмента.
        idx2:          Индекс второго фрагмента.
        score:         Итоговая оценка ∈ [0, 1].
        contributions: Вклад каждого источника в итоговую оценку.
        params:        Дополнительные параметры.
    """

    idx1: int
    idx2: int
    score: float
    contributions: Dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0:
            raise ValueError(f"idx1 должен быть >= 0, получено {self.idx1}")
        if self.idx2 < 0:
            raise ValueError(f"idx2 должен быть >= 0, получено {self.idx2}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    @property
    def pair(self):
        return (self.idx1, self.idx2)


# ─── weighted_combine ─────────────────────────────────────────────────────────

def weighted_combine(
    sv: ScoreVector,
    weights: Optional[Dict[str, float]] = None,
) -> CombinedScore:
    """Взвешенное среднее оценок из ScoreVector.

    Аргументы:
        sv:      Вектор оценок.
        weights: Словарь весов {имя_матчера: вес > 0}.
                 Если None — равные веса для всех матчеров.

    Возвращает:
        CombinedScore с итоговой взвешенной оценкой.

    Исключения:
        ValueError: Если scores пуст или веса отрицательны/нулевые.
    """
    if not sv.scores:
        raise ValueError("ScoreVector.scores не может быть пустым")

    names = list(sv.scores.keys())

    if weights is None:
        w = {n: 1.0 for n in names}
    else:
        for n, val in weights.items():
            if val < 0.0:
                raise ValueError(f"Вес '{n}' должен быть >= 0, получено {val}")
        w = {n: weights.get(n, 1.0) for n in names}

    total_w = sum(w[n] for n in names)
    if total_w <= 0.0:
        raise ValueError("Сумма весов должна быть > 0")

    combined = sum(sv.scores[n] * w[n] for n in names) / total_w
    combined = float(np.clip(combined, 0.0, 1.0))

    contributions = {n: sv.scores[n] * w[n] / total_w for n in names}
    return CombinedScore(idx1=sv.idx1, idx2=sv.idx2, score=combined,
                         contributions=contributions)


# ─── min_combine ─────────────────────────────────────────────────────────────

def min_combine(sv: ScoreVector) -> CombinedScore:
    """Комбинирование минимумом: итог = min(scores).

    Аргументы:
        sv: Вектор оценок.

    Возвращает:
        CombinedScore с минимальной оценкой.

    Исключения:
        ValueError: Если scores пуст.
    """
    if not sv.scores:
        raise ValueError("ScoreVector.scores не может быть пустым")

    score = float(min(sv.scores.values()))
    return CombinedScore(idx1=sv.idx1, idx2=sv.idx2, score=score,
                         contributions=dict(sv.scores))


# ─── max_combine ─────────────────────────────────────────────────────────────

def max_combine(sv: ScoreVector) -> CombinedScore:
    """Комбинирование максимумом: итог = max(scores).

    Аргументы:
        sv: Вектор оценок.

    Возвращает:
        CombinedScore с максимальной оценкой.

    Исключения:
        ValueError: Если scores пуст.
    """
    if not sv.scores:
        raise ValueError("ScoreVector.scores не может быть пустым")

    score = float(max(sv.scores.values()))
    return CombinedScore(idx1=sv.idx1, idx2=sv.idx2, score=score,
                         contributions=dict(sv.scores))


# ─── rank_combine ─────────────────────────────────────────────────────────────

def rank_combine(
    score_vectors: List[ScoreVector],
) -> List[CombinedScore]:
    """Ранговое слияние: средний нормализованный ранг по каждому матчеру.

    Для каждого матчера ранжирует пары по убыванию оценки (ранг 1 = лучший).
    Итоговая оценка = 1 − (средний_ранг − 1) / (N − 1), нормирована в [0, 1].

    Аргументы:
        score_vectors: Список ScoreVector для разных пар.

    Возвращает:
        Список CombinedScore в том же порядке, что score_vectors.

    Исключения:
        ValueError: Если список пуст или матчеры отличаются между парами.
    """
    if not score_vectors:
        return []

    n = len(score_vectors)
    matcher_names = list(score_vectors[0].scores.keys())

    for sv in score_vectors:
        if set(sv.scores.keys()) != set(matcher_names):
            raise ValueError(
                "Все ScoreVector должны содержать одинаковые ключи матчеров"
            )

    # Для каждого матчера определяем ранги (0-based, меньше = лучше)
    rank_sums = np.zeros(n, dtype=np.float64)
    for name in matcher_names:
        values = np.array([sv.scores[name] for sv in score_vectors], dtype=np.float64)
        # Ранг 0 у лучшего (наибольшей оценки)
        order = np.argsort(-values)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        rank_sums += ranks

    if n == 1:
        norm_scores = np.ones(n, dtype=np.float64)
    else:
        avg_ranks = rank_sums / len(matcher_names)
        norm_scores = 1.0 - avg_ranks / (n - 1)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)

    results = []
    for i, sv in enumerate(score_vectors):
        results.append(
            CombinedScore(
                idx1=sv.idx1,
                idx2=sv.idx2,
                score=float(norm_scores[i]),
                contributions=dict(sv.scores),
            )
        )
    return results


# ─── normalize_score_vectors ─────────────────────────────────────────────────

def normalize_score_vectors(
    score_vectors: List[ScoreVector],
) -> List[ScoreVector]:
    """Min-max нормализация оценок каждого матчера по всем парам.

    Для каждого матчера масштабирует оценки в [0, 1].
    Если все оценки одного матчера равны — они становятся 0.0.

    Аргументы:
        score_vectors: Список ScoreVector для разных пар.

    Возвращает:
        Новый список ScoreVector с нормализованными оценками.
    """
    if not score_vectors:
        return []

    matcher_names = list(score_vectors[0].scores.keys())
    all_scores: Dict[str, List[float]] = {n: [] for n in matcher_names}
    for sv in score_vectors:
        for n in matcher_names:
            all_scores[n].append(sv.scores.get(n, 0.0))

    mins = {n: min(v) for n, v in all_scores.items()}
    maxs = {n: max(v) for n, v in all_scores.items()}

    result = []
    for sv in score_vectors:
        new_scores: Dict[str, float] = {}
        for n in matcher_names:
            mn, mx = mins[n], maxs[n]
            if mx - mn < 1e-12:
                new_scores[n] = 0.0
            else:
                new_scores[n] = (sv.scores[n] - mn) / (mx - mn)
        result.append(
            ScoreVector(idx1=sv.idx1, idx2=sv.idx2,
                        scores=new_scores, params=dict(sv.params))
        )
    return result


# ─── batch_combine ────────────────────────────────────────────────────────────

def batch_combine(
    score_vectors: List[ScoreVector],
    method: str = "weighted",
    weights: Optional[Dict[str, float]] = None,
) -> List[CombinedScore]:
    """Пакетное комбинирование списка ScoreVector.

    Аргументы:
        score_vectors: Список векторов оценок.
        method:        Метод комбинирования: 'weighted', 'min', 'max', 'rank'.
        weights:       Веса для метода 'weighted'.

    Возвращает:
        Список CombinedScore, отсортированный по убыванию score.

    Исключения:
        ValueError: Если метод неизвестен или список пуст (для rank).
    """
    valid = {"weighted", "min", "max", "rank"}
    if method not in valid:
        raise ValueError(
            f"Неизвестный метод '{method}'. Допустимые: {sorted(valid)}"
        )
    if not score_vectors:
        return []

    if method == "rank":
        results = rank_combine(score_vectors)
    else:
        dispatch = {
            "weighted": lambda sv: weighted_combine(sv, weights),
            "min": min_combine,
            "max": max_combine,
        }
        results = [dispatch[method](sv) for sv in score_vectors]

    results.sort(key=lambda cs: cs.score, reverse=True)
    return results
