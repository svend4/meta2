"""
Объединение ранговых списков от нескольких алгоритмов оценки.

При сборке пазла несколько алгоритмов (цветовая, геометрическая,
текстурная совместимость) независимо ранжируют кандидатов. Этот модуль
реализует методы слияния ранговых списков в единый консенсусный список.

Функции:
    reciprocal_rank_fusion  — RRF (обратная сумма рангов)
    borda_count             — подсчёт голосов по методу Борда
    score_fusion            — взвешенное объединение оценок
    normalize_scores        — нормировка вектора оценок в [0, 1]
    fuse_rankings           — удобный фасад для всех методов слияния
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── normalize_scores ─────────────────────────────────────────────────────────

def normalize_scores(
    scores: List[float],
    eps:    float = 1e-9,
) -> List[float]:
    """Нормировать список оценок в [0, 1] (min-max).

    Если все оценки одинаковы — возвращает список из единиц.

    Args:
        scores: Список вещественных оценок.
        eps:    Порог нулевого диапазона.

    Returns:
        Список float в [0, 1].

    Raises:
        ValueError: Если список пуст.
    """
    if not scores:
        raise ValueError("scores не должен быть пустым")
    arr = np.asarray(scores, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        return [1.0] * len(scores)
    return list(((arr - mn) / (mx - mn)).tolist())


# ─── reciprocal_rank_fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k:            int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion (RRF) нескольких ранговых списков.

    RRF(d) = Σ_r 1 / (k + rank_r(d)), где rank начинается с 1.

    Args:
        ranked_lists: Список ранговых списков (каждый — список id в порядке убывания релевантности).
        k:            Константа сглаживания (> 0, рекомендуется 60).

    Returns:
        Список (id, rrf_score), отсортированный по убыванию score.

    Raises:
        ValueError: Если k <= 0 или список пуст.
    """
    if k <= 0:
        raise ValueError(f"k должен быть > 0, получено {k}")
    if not ranked_lists:
        raise ValueError("ranked_lists не должен быть пустым")

    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─── borda_count ──────────────────────────────────────────────────────────────

def borda_count(
    ranked_lists: List[List[int]],
) -> List[Tuple[int, float]]:
    """Подсчёт Борда: суммируем обратные ранги.

    Для списка длиной N: кандидат на позиции rank (0-based) получает N - rank - 1 очков.

    Args:
        ranked_lists: Список ранговых списков.

    Returns:
        Список (id, borda_score), отсортированный по убыванию.

    Raises:
        ValueError: Если список пуст.
    """
    if not ranked_lists:
        raise ValueError("ranked_lists не должен быть пустым")

    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        n = len(ranked)
        for rank, item_id in enumerate(ranked):
            points = float(n - rank - 1)
            scores[item_id] = scores.get(item_id, 0.0) + points

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─── score_fusion ─────────────────────────────────────────────────────────────

def score_fusion(
    score_lists: List[List[Tuple[int, float]]],
    weights:     Optional[List[float]] = None,
    normalize:   bool = True,
) -> List[Tuple[int, float]]:
    """Взвешенное объединение нескольких списков (id, score).

    Каждый список нормализуется (если normalize=True), затем суммируются
    взвешенные оценки.

    Args:
        score_lists: Список списков (id, score). Каждый список — отдельный алгоритм.
        weights:     Веса для каждого списка (None → равные веса).
        normalize:   Нормировать каждый список перед слиянием.

    Returns:
        Список (id, fused_score), отсортированный по убыванию.

    Raises:
        ValueError: Если score_lists пуст, или weights не совпадает по длине.
    """
    if not score_lists:
        raise ValueError("score_lists не должен быть пустым")
    if weights is not None and len(weights) != len(score_lists):
        raise ValueError(
            f"Длины weights ({len(weights)}) и score_lists ({len(score_lists)}) "
            f"должны совпадать"
        )
    if weights is None:
        weights = [1.0] * len(score_lists)

    total_w = sum(weights)
    if total_w < 1e-12:
        total_w = 1.0

    fused: Dict[int, float] = {}
    for sl, w in zip(score_lists, weights):
        if not sl:
            continue
        ids = [item_id for item_id, _ in sl]
        raw_scores = [s for _, s in sl]
        if normalize:
            raw_scores = normalize_scores(raw_scores)
        for item_id, sc in zip(ids, raw_scores):
            fused[item_id] = fused.get(item_id, 0.0) + w * sc / total_w

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ─── fuse_rankings ────────────────────────────────────────────────────────────

def fuse_rankings(
    ranked_lists: List[List[int]],
    method:       str = "rrf",
    k:            int = 60,
) -> List[Tuple[int, float]]:
    """Фасад для объединения ранговых списков.

    Args:
        ranked_lists: Список ранговых списков.
        method:       ``'rrf'`` или ``'borda'``.
        k:            Параметр k для RRF.

    Returns:
        Объединённый ранговый список (id, score).

    Raises:
        ValueError: Если method неизвестен.
    """
    if method == "rrf":
        return reciprocal_rank_fusion(ranked_lists, k=k)
    if method == "borda":
        return borda_count(ranked_lists)
    raise ValueError(
        f"Неизвестный метод {method!r}. Доступны: 'rrf', 'borda'"
    )
