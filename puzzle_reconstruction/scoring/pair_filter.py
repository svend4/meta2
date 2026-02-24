"""Фильтрация кандидатных пар фрагментов пазла.

Модуль выполняет многоэтапную фильтрацию пар фрагментов-кандидатов
по порогу оценки, числу инлаеров и глобальному ранжированию,
формируя итоговый список наиболее вероятных совпадений.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


_FILTER_METHODS = {"score", "inlier", "rank", "combined"}


# ─── FilterConfig ─────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """Параметры фильтрации пар.

    Атрибуты:
        method:       Метод фильтрации: 'score' | 'inlier' | 'rank' | 'combined'.
        min_score:    Минимальная оценка совместимости (>= 0).
        min_inliers:  Минимальное число инлаеров (>= 0).
        max_pairs:    Максимальное число пар на выходе (>= 1).
        top_k_per_id: Максимальное число пар на каждый фрагмент (>= 1).
    """

    method: str = "combined"
    min_score: float = 0.0
    min_inliers: int = 0
    max_pairs: int = 100
    top_k_per_id: int = 5

    def __post_init__(self) -> None:
        if self.method not in _FILTER_METHODS:
            raise ValueError(
                f"method должен быть одним из {_FILTER_METHODS}, получено '{self.method}'"
            )
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )
        if self.min_inliers < 0:
            raise ValueError(
                f"min_inliers должен быть >= 0, получено {self.min_inliers}"
            )
        if self.max_pairs < 1:
            raise ValueError(
                f"max_pairs должен быть >= 1, получено {self.max_pairs}"
            )
        if self.top_k_per_id < 1:
            raise ValueError(
                f"top_k_per_id должен быть >= 1, получено {self.top_k_per_id}"
            )


# ─── CandidatePair ────────────────────────────────────────────────────────────

@dataclass
class CandidatePair:
    """Кандидатная пара фрагментов.

    Атрибуты:
        id_a:      Идентификатор первого фрагмента (>= 0).
        id_b:      Идентификатор второго фрагмента (>= 0, id_b != id_a).
        score:     Оценка совместимости (>= 0).
        n_inliers: Число инлаеров (>= 0).
        rank:      Глобальный ранг (>= 0).
    """

    id_a: int
    id_b: int
    score: float
    n_inliers: int = 0
    rank: int = 0

    def __post_init__(self) -> None:
        if self.id_a < 0:
            raise ValueError(f"id_a должен быть >= 0, получено {self.id_a}")
        if self.id_b < 0:
            raise ValueError(f"id_b должен быть >= 0, получено {self.id_b}")
        if self.score < 0.0:
            raise ValueError(f"score должен быть >= 0, получено {self.score}")
        if self.n_inliers < 0:
            raise ValueError(f"n_inliers должен быть >= 0, получено {self.n_inliers}")
        if self.rank < 0:
            raise ValueError(f"rank должен быть >= 0, получено {self.rank}")

    @property
    def pair(self) -> Tuple[int, int]:
        """Каноническая пара (min_id, max_id)."""
        return (min(self.id_a, self.id_b), max(self.id_a, self.id_b))


# ─── FilterReport ─────────────────────────────────────────────────────────────

@dataclass
class FilterReport:
    """Отчёт о результатах фильтрации.

    Атрибуты:
        n_input:    Число пар до фильтрации (>= 0).
        n_output:   Число пар после фильтрации (>= 0).
        n_rejected: Число отвергнутых пар (>= 0).
        method:     Использованный метод.
    """

    n_input: int = 0
    n_output: int = 0
    n_rejected: int = 0
    method: str = "combined"

    def __post_init__(self) -> None:
        if self.n_input < 0:
            raise ValueError(f"n_input должен быть >= 0, получено {self.n_input}")
        if self.n_output < 0:
            raise ValueError(f"n_output должен быть >= 0, получено {self.n_output}")
        if self.n_rejected < 0:
            raise ValueError(f"n_rejected должен быть >= 0, получено {self.n_rejected}")

    @property
    def rejection_rate(self) -> float:
        """Доля отвергнутых пар [0, 1]."""
        if self.n_input == 0:
            return 0.0
        return float(self.n_rejected) / float(self.n_input)


# ─── filter_by_score ──────────────────────────────────────────────────────────

def filter_by_score(
    pairs: List[CandidatePair],
    min_score: float = 0.0,
) -> List[CandidatePair]:
    """Отфильтровать пары по порогу оценки.

    Аргументы:
        pairs:     Список кандидатных пар.
        min_score: Минимальная оценка совместимости (>= 0).

    Возвращает:
        Отфильтрованный список.

    Исключения:
        ValueError: Если min_score < 0.
    """
    if min_score < 0.0:
        raise ValueError(f"min_score должен быть >= 0, получено {min_score}")
    return [p for p in pairs if p.score >= min_score]


# ─── filter_by_inlier_count ───────────────────────────────────────────────────

def filter_by_inlier_count(
    pairs: List[CandidatePair],
    min_inliers: int = 0,
) -> List[CandidatePair]:
    """Отфильтровать пары по числу инлаеров.

    Аргументы:
        pairs:       Список кандидатных пар.
        min_inliers: Минимальное число инлаеров (>= 0).

    Возвращает:
        Отфильтрованный список.

    Исключения:
        ValueError: Если min_inliers < 0.
    """
    if min_inliers < 0:
        raise ValueError(f"min_inliers должен быть >= 0, получено {min_inliers}")
    return [p for p in pairs if p.n_inliers >= min_inliers]


# ─── filter_top_k ─────────────────────────────────────────────────────────────

def filter_top_k(
    pairs: List[CandidatePair],
    k: int,
) -> List[CandidatePair]:
    """Оставить top-k пар по убыванию оценки.

    Аргументы:
        pairs: Список кандидатных пар.
        k:     Число пар (>= 1).

    Возвращает:
        Не более k лучших пар.

    Исключения:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")
    return sorted(pairs, key=lambda p: p.score, reverse=True)[:k]


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

def deduplicate_pairs(
    pairs: List[CandidatePair],
) -> List[CandidatePair]:
    """Удалить симметричные дубликаты.

    Для каждой канонической пары (min_id, max_id) сохраняется экземпляр
    с наивысшей оценкой.

    Аргументы:
        pairs: Список кандидатных пар.

    Возвращает:
        Список без дубликатов.
    """
    best: Dict[Tuple[int, int], CandidatePair] = {}
    for p in pairs:
        key = p.pair
        if key not in best or p.score > best[key].score:
            best[key] = p
    return list(best.values())


# ─── filter_top_k_per_fragment ────────────────────────────────────────────────

def filter_top_k_per_fragment(
    pairs: List[CandidatePair],
    k: int = 5,
) -> List[CandidatePair]:
    """Ограничить число пар на каждый фрагмент (top-k по оценке).

    Аргументы:
        pairs: Список кандидатных пар.
        k:     Максимальное число пар на фрагмент (>= 1).

    Возвращает:
        Список, где каждый фрагмент участвует не более k раз.

    Исключения:
        ValueError: Если k < 1.
    """
    if k < 1:
        raise ValueError(f"k должен быть >= 1, получено {k}")

    sorted_pairs = sorted(pairs, key=lambda x: x.score, reverse=True)
    counts: Dict[int, int] = defaultdict(int)
    result: List[CandidatePair] = []
    for p in sorted_pairs:
        if counts[p.id_a] < k and counts[p.id_b] < k:
            result.append(p)
            counts[p.id_a] += 1
            counts[p.id_b] += 1
    return result


# ─── filter_pairs ─────────────────────────────────────────────────────────────

def filter_pairs(
    pairs: List[CandidatePair],
    cfg: Optional[FilterConfig] = None,
) -> Tuple[List[CandidatePair], FilterReport]:
    """Фильтровать пары согласно конфигурации.

    Аргументы:
        pairs: Список кандидатных пар.
        cfg:   Параметры (None → FilterConfig()).

    Возвращает:
        (filtered_pairs, FilterReport).
    """
    if cfg is None:
        cfg = FilterConfig()

    n_input = len(pairs)
    result = list(pairs)

    if cfg.method in ("score", "combined"):
        result = filter_by_score(result, cfg.min_score)
    if cfg.method in ("inlier", "combined"):
        result = filter_by_inlier_count(result, cfg.min_inliers)

    result = deduplicate_pairs(result)

    if len(result) > cfg.top_k_per_id:
        result = filter_top_k_per_fragment(result, cfg.top_k_per_id)

    result = filter_top_k(result, cfg.max_pairs)

    report = FilterReport(
        n_input=n_input,
        n_output=len(result),
        n_rejected=n_input - len(result),
        method=cfg.method,
    )
    return result, report


# ─── merge_filter_results ─────────────────────────────────────────────────────

def merge_filter_results(
    results: List[List[CandidatePair]],
) -> List[CandidatePair]:
    """Объединить несколько отфильтрованных списков, удалив дубликаты.

    Аргументы:
        results: Список списков пар.

    Возвращает:
        Единый список без дубликатов (по наивысшей оценке).
    """
    all_pairs: List[CandidatePair] = []
    for r in results:
        all_pairs.extend(r)
    return deduplicate_pairs(all_pairs)


# ─── batch_filter ─────────────────────────────────────────────────────────────

def batch_filter(
    pair_lists: List[List[CandidatePair]],
    cfg: Optional[FilterConfig] = None,
) -> List[Tuple[List[CandidatePair], FilterReport]]:
    """Фильтровать несколько списков пар.

    Аргументы:
        pair_lists: Список списков пар.
        cfg:        Параметры.

    Возвращает:
        Список (filtered_pairs, FilterReport).
    """
    return [filter_pairs(pl, cfg) for pl in pair_lists]
