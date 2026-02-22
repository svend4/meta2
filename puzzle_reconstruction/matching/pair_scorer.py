"""Агрегированная оценка совместимости пары фрагментов.

Модуль объединяет несколько каналов совместимости (цвет, текстура,
геометрия, градиент) в единую оценку пары фрагментов.  Поддерживает
взвешенную агрегацию, нормализацию и пакетную обработку.

Публичный API:
    ScoringWeights     — весовые коэффициенты каналов
    PairScoreResult    — итоговая оценка пары с разбивкой по каналам
    aggregate_channels — взвешенная агрегация словаря channel→score
    score_pair         — оценка одной пары по словарю канальных оценок
    select_top_pairs   — отбор лучших пар по порогу или top-K
    build_score_matrix — матрица попарных оценок (N × N)
    batch_score_pairs  — пакетная оценка списка пар
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── ScoringWeights ───────────────────────────────────────────────────────────

@dataclass
class ScoringWeights:
    """Весовые коэффициенты каналов совместимости.

    Атрибуты:
        color:    Вес цветового канала (>= 0).
        texture:  Вес текстурного канала (>= 0).
        geometry: Вес геометрического канала (>= 0).
        gradient: Вес градиентного канала (>= 0).
    """

    color: float = 1.0
    texture: float = 1.0
    geometry: float = 1.0
    gradient: float = 1.0

    def __post_init__(self) -> None:
        for name in ("color", "texture", "geometry", "gradient"):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(
                    f"Вес '{name}' должен быть >= 0, получено {val}"
                )
        if self.total == 0.0:
            raise ValueError(
                "Сумма весов должна быть > 0"
            )

    @property
    def total(self) -> float:
        """Суммарный вес."""
        return self.color + self.texture + self.geometry + self.gradient

    def as_dict(self) -> Dict[str, float]:
        """Словарь channel → weight."""
        return {
            "color": self.color,
            "texture": self.texture,
            "geometry": self.geometry,
            "gradient": self.gradient,
        }

    def normalized(self) -> "ScoringWeights":
        """Нормализованные веса (сумма = 1)."""
        t = self.total
        return ScoringWeights(
            color=self.color / t,
            texture=self.texture / t,
            geometry=self.geometry / t,
            gradient=self.gradient / t,
        )


# ─── PairScoreResult ──────────────────────────────────────────────────────────

@dataclass
class PairScoreResult:
    """Итоговая оценка совместимости пары фрагментов.

    Атрибуты:
        idx_a:    Индекс первого фрагмента.
        idx_b:    Индекс второго фрагмента.
        score:    Агрегированная оценка [0, 1].
        channels: Канальные оценки {channel: score}.
        n_channels: Число использованных каналов.
    """

    idx_a: int
    idx_b: int
    score: float
    channels: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )

    @property
    def n_channels(self) -> int:
        """Число использованных каналов."""
        return len(self.channels)

    @property
    def pair_key(self) -> Tuple[int, int]:
        """Упорядоченная пара (min, max)."""
        return (min(self.idx_a, self.idx_b), max(self.idx_a, self.idx_b))

    @property
    def dominant_channel(self) -> Optional[str]:
        """Канал с наибольшей оценкой (None если channels пустой)."""
        if not self.channels:
            return None
        return max(self.channels, key=lambda k: self.channels[k])

    @property
    def is_strong_match(self) -> bool:
        """True если score >= 0.7."""
        return self.score >= 0.7


# ─── aggregate_channels ───────────────────────────────────────────────────────

def aggregate_channels(
    channel_scores: Dict[str, float],
    weights: Optional[ScoringWeights] = None,
) -> float:
    """Вычислить взвешенную агрегированную оценку из словаря канальных оценок.

    Аргументы:
        channel_scores: Словарь {channel: score}; score ∈ [0, 1].
        weights:        Весовые коэффициенты (None → ScoringWeights()).

    Возвращает:
        Взвешенная средняя оценка [0, 1].

    Исключения:
        ValueError: Если channel_scores пуст или содержит значения вне [0, 1].
    """
    if not channel_scores:
        raise ValueError("channel_scores не должен быть пустым")
    for ch, v in channel_scores.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"Оценка канала '{ch}' должна быть в [0, 1], получено {v}"
            )

    if weights is None:
        weights = ScoringWeights()

    w_dict = weights.as_dict()
    w_sum = sum(w_dict.get(ch, 1.0) for ch in channel_scores) + 1e-12
    score = sum(
        channel_scores[ch] * w_dict.get(ch, 1.0)
        for ch in channel_scores
    ) / w_sum
    return float(np.clip(score, 0.0, 1.0))


# ─── score_pair ───────────────────────────────────────────────────────────────

def score_pair(
    idx_a: int,
    idx_b: int,
    channel_scores: Dict[str, float],
    weights: Optional[ScoringWeights] = None,
) -> PairScoreResult:
    """Оценить совместимость пары фрагментов.

    Аргументы:
        idx_a:          Индекс первого фрагмента.
        idx_b:          Индекс второго фрагмента.
        channel_scores: Словарь {channel: score}.
        weights:        Весовые коэффициенты.

    Возвращает:
        PairScoreResult.

    Исключения:
        ValueError: Если channel_scores пуст.
    """
    agg = aggregate_channels(channel_scores, weights)
    return PairScoreResult(
        idx_a=idx_a,
        idx_b=idx_b,
        score=agg,
        channels=dict(channel_scores),
    )


# ─── select_top_pairs ─────────────────────────────────────────────────────────

def select_top_pairs(
    results: List[PairScoreResult],
    threshold: float = 0.0,
    top_k: int = 0,
) -> List[PairScoreResult]:
    """Отобрать лучшие пары по порогу и/или top-K.

    Аргументы:
        results:   Список PairScoreResult.
        threshold: Минимальная оценка (включительно; >= 0).
        top_k:     Ограничение сверху (0 = без ограничения; >= 0).

    Возвращает:
        Отфильтрованный список, отсортированный по убыванию score.

    Исключения:
        ValueError: Если threshold < 0 или top_k < 0.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold должен быть >= 0, получено {threshold}")
    if top_k < 0:
        raise ValueError(f"top_k должен быть >= 0, получено {top_k}")

    filtered = [r for r in results if r.score >= threshold]
    filtered.sort(key=lambda r: r.score, reverse=True)

    if top_k > 0:
        filtered = filtered[:top_k]

    return filtered


# ─── build_score_matrix ───────────────────────────────────────────────────────

def build_score_matrix(
    results: List[PairScoreResult],
    n_fragments: int,
) -> np.ndarray:
    """Построить матрицу попарных оценок (n_fragments × n_fragments).

    Аргументы:
        results:     Список PairScoreResult.
        n_fragments: Размерность матрицы (>= 1).

    Возвращает:
        Симметричная матрица float32 размером (n_fragments, n_fragments).
        Пары, не вошедшие в results, получают значение 0.

    Исключения:
        ValueError: Если n_fragments < 1.
    """
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments должен быть >= 1, получено {n_fragments}"
        )
    mat = np.zeros((n_fragments, n_fragments), dtype=np.float32)
    for r in results:
        a, b = r.idx_a, r.idx_b
        if 0 <= a < n_fragments and 0 <= b < n_fragments:
            mat[a, b] = r.score
            mat[b, a] = r.score
    return mat


# ─── batch_score_pairs ────────────────────────────────────────────────────────

def batch_score_pairs(
    pairs: List[Tuple[int, int]],
    channel_scores_list: List[Dict[str, float]],
    weights: Optional[ScoringWeights] = None,
) -> List[PairScoreResult]:
    """Пакетно оценить список пар.

    Аргументы:
        pairs:               Список пар (idx_a, idx_b).
        channel_scores_list: Список словарей канальных оценок.
        weights:             Весовые коэффициенты.

    Возвращает:
        Список PairScoreResult той же длины.

    Исключения:
        ValueError: Если длины pairs и channel_scores_list не совпадают.
    """
    if len(pairs) != len(channel_scores_list):
        raise ValueError(
            f"Длины pairs ({len(pairs)}) и channel_scores_list "
            f"({len(channel_scores_list)}) не совпадают"
        )
    return [
        score_pair(a, b, cs, weights)
        for (a, b), cs in zip(pairs, channel_scores_list)
    ]
