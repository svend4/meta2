"""
Агрегация нескольких оценок совместимости рёбер фрагментов.

Объединяет оценки из разных каналов (цвет, текстура, геометрия,
текстовые профили) в единую итоговую оценку совместимости пары
фрагментов. Поддерживает взвешенное среднее, гармоническое среднее,
мин/макс.

Классы:
    AggregationResult — итоговая оценка с детализацией по каналам

Функции:
    weighted_avg       — взвешенное среднее оценок ∈ [0,1]
    harmonic_mean      — гармоническое среднее (штрафует за слабые каналы)
    aggregate_scores   — агрегация по выбранному методу → AggregationResult
    threshold_filter   — булева маска пар, прошедших порог
    top_k_pairs        — топ-K индексов пар по итоговой оценке
    batch_aggregate    — пакетная агрегация матрицы пар → 1D массив оценок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── AggregationResult ────────────────────────────────────────────────────────

@dataclass
class AggregationResult:
    """
    Результат агрегации оценок по нескольким каналам.

    Attributes:
        score:   Итоговая оценка ∈ [0,1].
        scores:  Оценки по каналам {name: value}.
        weights: Веса каналов {name: weight}.
        method:  Метод агрегации ('weighted_avg', 'harmonic', 'min', 'max').
        params:  Дополнительные параметры.
    """
    score:   float
    scores:  Dict[str, float]
    weights: Dict[str, float]
    method:  str
    params:  Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        channels = len(self.scores)
        return (f"AggregationResult(score={self.score:.4f}, "
                f"channels={channels}, method={self.method!r})")


# ─── weighted_avg ─────────────────────────────────────────────────────────────

def weighted_avg(scores:  Dict[str, float],
                  weights: Optional[Dict[str, float]] = None) -> float:
    """
    Вычисляет взвешенное среднее оценок.

    Args:
        scores:  Словарь {канал: оценка ∈ [0,1]}.
        weights: Словарь {канал: вес ≥ 0}; None → равные веса.

    Returns:
        Взвешенное среднее ∈ [0,1].

    Raises:
        ValueError: Если scores пуст или сумма весов равна 0.
    """
    if not scores:
        raise ValueError("scores must not be empty.")

    keys = list(scores.keys())

    if weights is None:
        w = {k: 1.0 for k in keys}
    else:
        w = {k: float(weights.get(k, 1.0)) for k in keys}

    total_w = sum(w[k] for k in keys)
    if total_w <= 0.0:
        raise ValueError("Sum of weights must be > 0.")

    weighted_sum = sum(scores[k] * w[k] for k in keys)
    return float(np.clip(weighted_sum / total_w, 0.0, 1.0))


# ─── harmonic_mean ────────────────────────────────────────────────────────────

def harmonic_mean(scores: Dict[str, float]) -> float:
    """
    Вычисляет гармоническое среднее оценок.

    Штрафует результат при наличии каналов с низкими оценками.
    Если хотя бы один канал равен 0, возвращает 0.0.

    Args:
        scores: Словарь {канал: оценка ∈ [0,1]}.

    Returns:
        Гармоническое среднее ∈ [0,1].

    Raises:
        ValueError: Если scores пуст.
    """
    if not scores:
        raise ValueError("scores must not be empty.")

    vals = [float(v) for v in scores.values()]
    for v in vals:
        if v <= 0.0:
            return 0.0

    n = len(vals)
    return float(np.clip(n / sum(1.0 / v for v in vals), 0.0, 1.0))


# ─── aggregate_scores ─────────────────────────────────────────────────────────

def aggregate_scores(scores:  Dict[str, float],
                      weights: Optional[Dict[str, float]] = None,
                      method:  str = "weighted_avg") -> AggregationResult:
    """
    Агрегирует оценки по нескольким каналам в единую оценку.

    Args:
        scores:  Словарь {канал: оценка ∈ [0,1]}.
        weights: Словарь {канал: вес ≥ 0} (используется для 'weighted_avg').
        method:  'weighted_avg' | 'harmonic' | 'min' | 'max'.

    Returns:
        AggregationResult с итоговой оценкой и детализацией.

    Raises:
        ValueError: Неизвестный метод или пустой scores.
    """
    if not scores:
        raise ValueError("scores must not be empty.")

    if method == "weighted_avg":
        score = weighted_avg(scores, weights)
    elif method == "harmonic":
        score = harmonic_mean(scores)
    elif method == "min":
        score = float(np.clip(min(scores.values()), 0.0, 1.0))
    elif method == "max":
        score = float(np.clip(max(scores.values()), 0.0, 1.0))
    else:
        raise ValueError(
            f"Unknown aggregation method {method!r}. "
            f"Choose from: 'weighted_avg', 'harmonic', 'min', 'max'."
        )

    eff_weights: Dict[str, float]
    if weights is not None:
        eff_weights = {k: float(weights.get(k, 1.0)) for k in scores}
    else:
        eff_weights = {k: 1.0 for k in scores}

    return AggregationResult(
        score=score,
        scores=dict(scores),
        weights=eff_weights,
        method=method,
        params={"n_channels": len(scores)},
    )


# ─── threshold_filter ─────────────────────────────────────────────────────────

def threshold_filter(results:   List[AggregationResult],
                      threshold: float = 0.5) -> List[bool]:
    """
    Возвращает булеву маску: True, если score > threshold.

    Args:
        results:   Список AggregationResult.
        threshold: Порог отбора ∈ [0,1].

    Returns:
        Список bool той же длины, что results.
    """
    return [r.score > threshold for r in results]


# ─── top_k_pairs ──────────────────────────────────────────────────────────────

def top_k_pairs(pairs:   List[Tuple[int, int]],
                 results: List[AggregationResult],
                 k:       int) -> List[Tuple[int, int]]:
    """
    Возвращает топ-K пар по убыванию итоговой оценки.

    Args:
        pairs:   Список кортежей (idx1, idx2).
        results: Соответствующие AggregationResult.
        k:       Количество возвращаемых пар (k ≤ len(pairs)).

    Returns:
        Список из min(k, len(pairs)) пар, отсортированных по score↓.

    Raises:
        ValueError: Если len(pairs) ≠ len(results).
    """
    if len(pairs) != len(results):
        raise ValueError(
            f"pairs ({len(pairs)}) and results ({len(results)}) "
            f"must have the same length."
        )
    k = min(k, len(pairs))
    if k == 0:
        return []

    indexed = sorted(enumerate(results), key=lambda x: x[1].score, reverse=True)
    return [pairs[i] for i, _ in indexed[:k]]


# ─── batch_aggregate ──────────────────────────────────────────────────────────

def batch_aggregate(score_matrix: np.ndarray,
                     channel_names: Optional[List[str]] = None,
                     weights:       Optional[List[float]] = None,
                     method:        str = "weighted_avg") -> np.ndarray:
    """
    Пакетная агрегация матрицы оценок (N пар × M каналов).

    Args:
        score_matrix:  Массив формы (N, M), значения ∈ [0,1].
        channel_names: Имена M каналов; None → ['ch_0', 'ch_1', ...].
        weights:       Список M весов; None → равные веса.
        method:        Метод агрегации (см. aggregate_scores).

    Returns:
        Одномерный массив float32 длины N с итоговыми оценками.

    Raises:
        ValueError: Если score_matrix не двумерный или пустой.
    """
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    if score_matrix.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D (N_pairs × N_channels), "
            f"got shape {score_matrix.shape}."
        )
    n_pairs, n_channels = score_matrix.shape
    if n_channels == 0:
        raise ValueError("score_matrix must have at least one channel (column).")

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    if weights is not None:
        w_dict = {channel_names[i]: float(weights[i]) for i in range(n_channels)}
    else:
        w_dict = None

    out = np.empty(n_pairs, dtype=np.float32)
    for i in range(n_pairs):
        row  = {channel_names[j]: float(score_matrix[i, j]) for j in range(n_channels)}
        res  = aggregate_scores(row, weights=w_dict, method=method)
        out[i] = float(res.score)

    return out
