"""Агрегация и слияние оценок совместимости из нескольких источников.

Модуль предоставляет утилиты для объединения оценок совместимости пар
фрагментов, поступающих от различных матчеров (цвет, текстура, геометрия
и т.д.), в единый вектор и итоговую скалярную оценку.

Публичный API:
    AggregationMethod   — перечисление методов агрегации
    ScoreVector         — вектор оценок по каналам для одной пары
    AggregationResult   — итог агрегации набора пар
    weighted_sum        — взвешенная сумма нормализованных каналов
    harmonic_mean       — гармоническое среднее каналов
    geometric_mean      — геометрическое среднее каналов
    aggregate_pair      — агрегировать один ScoreVector в скаляр
    aggregate_matrix    — агрегировать матрицу пар
    top_k_pairs         — отобрать top-K пар по итоговой оценке
    batch_aggregate     — пакетная агрегация списка ScoreVector
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── AggregationMethod ────────────────────────────────────────────────────────

class AggregationMethod(str, Enum):
    """Метод агрегации канальных оценок."""
    WEIGHTED = "weighted"
    HARMONIC = "harmonic"
    GEOMETRIC = "geometric"
    MIN = "min"
    MAX = "max"


# ─── ScoreVector ──────────────────────────────────────────────────────────────

@dataclass
class ScoreVector:
    """Вектор оценок совместимости для одной пары фрагментов.

    Атрибуты:
        idx_a:    Индекс первого фрагмента.
        idx_b:    Индекс второго фрагмента.
        channels: Словарь {channel_name: score}; каждое значение ∈ [0, 1].
        weights:  Опциональные веса каналов {channel_name: weight >= 0}.
    """

    idx_a: int
    idx_b: int
    channels: Dict[str, float]
    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for ch, v in self.channels.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"Оценка канала '{ch}' должна быть в [0, 1], получено {v}"
                )
        for ch, w in self.weights.items():
            if w < 0.0:
                raise ValueError(
                    f"Вес канала '{ch}' должен быть >= 0, получено {w}"
                )

    @property
    def n_channels(self) -> int:
        """Число каналов."""
        return len(self.channels)

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Канонический ключ (min, max)."""
        return (min(self.idx_a, self.idx_b), max(self.idx_a, self.idx_b))

    @property
    def mean_score(self) -> float:
        """Невзвешенное среднее оценок каналов."""
        if not self.channels:
            return 0.0
        return float(np.mean(list(self.channels.values())))

    @property
    def max_score(self) -> float:
        """Максимальная оценка среди каналов."""
        if not self.channels:
            return 0.0
        return float(max(self.channels.values()))

    @property
    def min_score(self) -> float:
        """Минимальная оценка среди каналов."""
        if not self.channels:
            return 0.0
        return float(min(self.channels.values()))


# ─── AggregationResult ────────────────────────────────────────────────────────

@dataclass
class AggregationResult:
    """Итог агрегации набора пар фрагментов.

    Атрибуты:
        scores:   Словарь {pair_key: score} со скалярными оценками.
        method:   Использованный метод агрегации.
        n_pairs:  Число агрегированных пар (>= 0).
        mean:     Средняя оценка по всем парам [0, 1].
        top_pair: Пара с наивысшей оценкой или None если пусто.
    """

    scores: Dict[Tuple[int, int], float]
    method: str
    n_pairs: int
    mean: float
    top_pair: Optional[Tuple[int, int]]

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError(
                f"n_pairs должен быть >= 0, получено {self.n_pairs}"
            )
        if not (0.0 <= self.mean <= 1.0):
            raise ValueError(
                f"mean должен быть в [0, 1], получено {self.mean}"
            )

    def get_score(self, idx_a: int, idx_b: int) -> Optional[float]:
        """Найти оценку по паре (None если не найдена)."""
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        return self.scores.get(key)


# ─── weighted_sum ─────────────────────────────────────────────────────────────

def weighted_sum(
    channels: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Взвешенная сумма нормализованных оценок каналов.

    Аргументы:
        channels: Словарь {channel: score ∈ [0, 1]}.
        weights:  Словарь весов (None → равные веса).

    Возвращает:
        Взвешенная средняя оценка ∈ [0, 1].

    Исключения:
        ValueError: Если channels пуст.
    """
    if not channels:
        raise ValueError("channels не должен быть пустым")

    if weights is None:
        weights = {k: 1.0 for k in channels}

    w_sum = sum(weights.get(k, 1.0) for k in channels) + 1e-12
    score = sum(
        channels[k] * weights.get(k, 1.0)
        for k in channels
    ) / w_sum
    return float(np.clip(score, 0.0, 1.0))


# ─── harmonic_mean ────────────────────────────────────────────────────────────

def harmonic_mean(channels: Dict[str, float]) -> float:
    """Гармоническое среднее оценок каналов.

    Аргументы:
        channels: Словарь {channel: score ∈ [0, 1]}.

    Возвращает:
        Гармоническое среднее ∈ [0, 1].

    Исключения:
        ValueError: Если channels пуст.
    """
    if not channels:
        raise ValueError("channels не должен быть пустым")

    values = list(channels.values())
    denom = sum(1.0 / max(v, 1e-9) for v in values)
    score = len(values) / denom
    return float(np.clip(score, 0.0, 1.0))


# ─── geometric_mean ───────────────────────────────────────────────────────────

def geometric_mean(channels: Dict[str, float]) -> float:
    """Геометрическое среднее оценок каналов.

    Аргументы:
        channels: Словарь {channel: score ∈ [0, 1]}.

    Возвращает:
        Геометрическое среднее ∈ [0, 1].

    Исключения:
        ValueError: Если channels пуст.
    """
    if not channels:
        raise ValueError("channels не должен быть пустым")

    values = list(channels.values())
    log_sum = sum(math.log(max(v, 1e-9)) for v in values)
    score = math.exp(log_sum / len(values))
    return float(np.clip(score, 0.0, 1.0))


# ─── aggregate_pair ───────────────────────────────────────────────────────────

def aggregate_pair(
    sv: ScoreVector,
    method: AggregationMethod = AggregationMethod.WEIGHTED,
) -> float:
    """Агрегировать ScoreVector в скалярную оценку.

    Аргументы:
        sv:     ScoreVector с канальными оценками.
        method: Метод агрегации.

    Возвращает:
        Скалярная оценка ∈ [0, 1].

    Исключения:
        ValueError: Если channels пуст.
    """
    if not sv.channels:
        raise ValueError("ScoreVector.channels не должен быть пустым")

    if method == AggregationMethod.WEIGHTED:
        return weighted_sum(sv.channels, sv.weights or None)
    elif method == AggregationMethod.HARMONIC:
        return harmonic_mean(sv.channels)
    elif method == AggregationMethod.GEOMETRIC:
        return geometric_mean(sv.channels)
    elif method == AggregationMethod.MIN:
        return float(min(sv.channels.values()))
    elif method == AggregationMethod.MAX:
        return float(max(sv.channels.values()))
    else:
        raise ValueError(f"Неизвестный метод: {method!r}")


# ─── aggregate_matrix ─────────────────────────────────────────────────────────

def aggregate_matrix(
    vectors: List[ScoreVector],
    n_fragments: int,
    method: AggregationMethod = AggregationMethod.WEIGHTED,
) -> np.ndarray:
    """Построить матрицу агрегированных оценок (n_fragments × n_fragments).

    Аргументы:
        vectors:     Список ScoreVector.
        n_fragments: Размер матрицы (>= 1).
        method:      Метод агрегации.

    Возвращает:
        Симметричная матрица float32.

    Исключения:
        ValueError: Если n_fragments < 1.
    """
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments должен быть >= 1, получено {n_fragments}"
        )
    mat = np.zeros((n_fragments, n_fragments), dtype=np.float32)
    for sv in vectors:
        if not sv.channels:
            continue
        a, b = sv.idx_a, sv.idx_b
        if 0 <= a < n_fragments and 0 <= b < n_fragments:
            score = aggregate_pair(sv, method)
            mat[a, b] = score
            mat[b, a] = score
    return mat


# ─── top_k_pairs ──────────────────────────────────────────────────────────────

def top_k_pairs(
    result: AggregationResult,
    k: int,
) -> List[Tuple[Tuple[int, int], float]]:
    """Отобрать top-K пар по убыванию оценки.

    Аргументы:
        result: AggregationResult.
        k:      Число пар (>= 1).

    Возвращает:
        Список [(pair_key, score)], отсортированный по убыванию.

    Исключения:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    sorted_pairs = sorted(result.scores.items(),
                          key=lambda x: x[1], reverse=True)
    return sorted_pairs[:k]


# ─── batch_aggregate ──────────────────────────────────────────────────────────

def batch_aggregate(
    vectors: List[ScoreVector],
    method: AggregationMethod = AggregationMethod.WEIGHTED,
) -> AggregationResult:
    """Пакетно агрегировать список ScoreVector в AggregationResult.

    Аргументы:
        vectors: Список ScoreVector.
        method:  Метод агрегации.

    Возвращает:
        AggregationResult.
    """
    scores: Dict[Tuple[int, int], float] = {}
    for sv in vectors:
        if not sv.channels:
            continue
        key = sv.pair_key
        score = aggregate_pair(sv, method)
        # Лучшая оценка для дедупликации
        if key not in scores or score > scores[key]:
            scores[key] = score

    n_pairs = len(scores)
    if scores:
        mean = float(np.mean(list(scores.values())))
        top_pair = max(scores, key=lambda k: scores[k])
    else:
        mean = 0.0
        top_pair = None

    return AggregationResult(
        scores=scores,
        method=method.value,
        n_pairs=n_pairs,
        mean=float(np.clip(mean, 0.0, 1.0)),
        top_pair=top_pair,
    )


