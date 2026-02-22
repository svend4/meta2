"""Фильтрация и прореживание кандидатных совпадений краёв.

Предоставляет набор функций для постобработки результатов
edge_comparator: фильтрацию по оценке, удаление дублей,
выбор топ-K кандидатов и пакетную фильтрацию.

Экспортирует:
    EdgeFilterConfig   — конфигурация фильтрации
    filter_by_score    — отфильтровать по минимальной оценке
    filter_top_k       — оставить K лучших результатов
    filter_compatible  — оставить только совместимые (score >= 0.6)
    deduplicate_pairs  — убрать дублирующиеся пары (a,b) ≡ (b,a)
    apply_edge_filter  — применить все активные фильтры последовательно
    batch_filter_edges — фильтровать список списков результатов
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .edge_comparator import EdgeCompareResult


# ─── EdgeFilterConfig ─────────────────────────────────────────────────────────

@dataclass
class EdgeFilterConfig:
    """Конфигурация фильтрации совпадений краёв.

    Атрибуты:
        min_score:      Минимальный score для включения (None = без фильтра).
        top_k:          Оставить только топ-K результатов (None = все).
        deduplicate:    Убрать зеркальные дубли (a,b) и (b,a).
        only_compatible: Оставить только is_compatible (score >= 0.6).
    """
    min_score:       Optional[float] = None
    top_k:           Optional[int]   = None
    deduplicate:     bool             = True
    only_compatible: bool             = False

    def __post_init__(self) -> None:
        if self.min_score is not None and not (0.0 <= self.min_score <= 1.0):
            raise ValueError(
                f"min_score должен быть в [0, 1], получено {self.min_score}"
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(
                f"top_k должен быть >= 1, получено {self.top_k}"
            )


# ─── Публичные функции ────────────────────────────────────────────────────────

def filter_by_score(
    results: List[EdgeCompareResult],
    min_score: float,
) -> List[EdgeCompareResult]:
    """Отфильтровать результаты по минимальной оценке.

    Args:
        results:   Список EdgeCompareResult.
        min_score: Минимальный score (включительно) для включения. ∈ [0, 1].

    Returns:
        Отфильтрованный список (порядок сохранён).

    Raises:
        ValueError: Если min_score не в [0, 1].
    """
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(
            f"min_score должен быть в [0, 1], получено {min_score}"
        )
    return [r for r in results if r.score >= min_score]


def filter_top_k(
    results: List[EdgeCompareResult],
    k: int,
) -> List[EdgeCompareResult]:
    """Оставить K лучших результатов по убыванию score.

    Args:
        results: Список EdgeCompareResult.
        k:       Число лучших результатов (>= 1).

    Returns:
        Список длиной min(k, len(results)), отсортированный по убыванию score.

    Raises:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    return sorted_results[:k]


def filter_compatible(
    results: List[EdgeCompareResult],
) -> List[EdgeCompareResult]:
    """Оставить только совместимые результаты (score >= 0.6).

    Args:
        results: Список EdgeCompareResult.

    Returns:
        Список результатов, для которых is_compatible == True.
    """
    return [r for r in results if r.is_compatible]


def deduplicate_pairs(
    results: List[EdgeCompareResult],
) -> List[EdgeCompareResult]:
    """Убрать зеркальные дубли (a, b) и (b, a).

    При наличии дублей оставляется первый встреченный экземпляр.

    Args:
        results: Список EdgeCompareResult.

    Returns:
        Список без дублей (порядок первых вхождений сохранён).
    """
    seen: set = set()
    unique: List[EdgeCompareResult] = []
    for r in results:
        key = r.pair_key  # (min_id, max_id)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def apply_edge_filter(
    results: List[EdgeCompareResult],
    cfg: Optional[EdgeFilterConfig] = None,
) -> List[EdgeCompareResult]:
    """Применить все активные фильтры последовательно.

    Порядок применения:
        1. filter_by_score     (если cfg.min_score задан)
        2. filter_compatible   (если cfg.only_compatible)
        3. deduplicate_pairs   (если cfg.deduplicate)
        4. filter_top_k        (если cfg.top_k задан)

    Args:
        results: Список EdgeCompareResult.
        cfg:     Конфигурация (None → EdgeFilterConfig() с умолчаниями).

    Returns:
        Отфильтрованный список.
    """
    if cfg is None:
        cfg = EdgeFilterConfig()

    filtered = list(results)

    if cfg.min_score is not None:
        filtered = filter_by_score(filtered, cfg.min_score)

    if cfg.only_compatible:
        filtered = filter_compatible(filtered)

    if cfg.deduplicate:
        filtered = deduplicate_pairs(filtered)

    if cfg.top_k is not None:
        filtered = filter_top_k(filtered, cfg.top_k)

    return filtered


def batch_filter_edges(
    batches: List[List[EdgeCompareResult]],
    cfg: Optional[EdgeFilterConfig] = None,
) -> List[List[EdgeCompareResult]]:
    """Применить фильтрацию к списку списков результатов.

    Args:
        batches: Список списков EdgeCompareResult.
        cfg:     Конфигурация фильтрации.

    Returns:
        Список отфильтрованных списков.
    """
    if cfg is None:
        cfg = EdgeFilterConfig()
    return [apply_edge_filter(batch, cfg) for batch in batches]
