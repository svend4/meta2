"""
Фильтрация кандидатов для сборки фрагментов документа.

Предоставляет инструменты для отбора, ранжирования и отсева
кандидатов на совмещение фрагментов на основе пороговых значений,
топ-k отбора и инкрементального фильтра.

Экспортирует:
    Candidate             — кандидат с индексами и оценкой
    FilterResult          — результат фильтрации
    filter_by_threshold   — отбор по минимальной оценке
    filter_top_k          — топ-k кандидатов
    filter_by_rank        — отбор по ранговому порогу
    deduplicate_candidates — удалить дублирующие пары
    normalize_scores      — нормировать оценки к [0, 1]
    merge_candidate_lists — объединить несколько списков кандидатов
    batch_filter          — пакетная фильтрация
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """Кандидат на совмещение двух фрагментов.

    Attributes:
        idx1:   Индекс первого фрагмента (≥ 0).
        idx2:   Индекс второго фрагмента (≥ 0).
        score:  Оценка совместимости ∈ [0, 1].
        params: Дополнительные параметры.
    """
    idx1: int
    idx2: int
    score: float
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.idx1 < 0:
            raise ValueError(f"idx1 must be >= 0, got {self.idx1}")
        if self.idx2 < 0:
            raise ValueError(f"idx2 must be >= 0, got {self.idx2}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [0, 1], got {self.score}"
            )

    @property
    def pair(self):
        """Пара индексов (idx1, idx2)."""
        return (self.idx1, self.idx2)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Candidate(idx1={self.idx1}, idx2={self.idx2}, "
            f"score={self.score:.4f})"
        )


@dataclass
class FilterResult:
    """Результат фильтрации кандидатов.

    Attributes:
        candidates:  Отфильтрованный список :class:`Candidate`.
        n_kept:      Число оставленных кандидатов.
        n_removed:   Число удалённых кандидатов.
        params:      Параметры фильтрации.
    """
    candidates: List[Candidate]
    n_kept: int
    n_removed: int
    params: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_kept < 0:
            raise ValueError(f"n_kept must be >= 0, got {self.n_kept}")
        if self.n_removed < 0:
            raise ValueError(f"n_removed must be >= 0, got {self.n_removed}")

    def __len__(self) -> int:
        return self.n_kept

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FilterResult(n_kept={self.n_kept}, "
            f"n_removed={self.n_removed})"
        )


# ─── Публичные функции ────────────────────────────────────────────────────────

def filter_by_threshold(
    candidates: List[Candidate],
    threshold: float,
) -> FilterResult:
    """Отобрать кандидатов с оценкой выше порога.

    Args:
        candidates: Список :class:`Candidate`.
        threshold:  Минимальная оценка ∈ [0, 1].

    Returns:
        :class:`FilterResult` с отсортированными по убыванию кандидатами.

    Raises:
        ValueError: Если ``threshold`` вне [0, 1].
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold}"
        )
    kept = [c for c in candidates if c.score >= threshold]
    kept.sort(key=lambda c: c.score, reverse=True)
    return FilterResult(
        candidates=kept,
        n_kept=len(kept),
        n_removed=len(candidates) - len(kept),
        params={"threshold": threshold},
    )


def filter_top_k(
    candidates: List[Candidate],
    k: int,
) -> FilterResult:
    """Оставить топ-k кандидатов по оценке.

    Args:
        candidates: Список :class:`Candidate`.
        k:          Максимальное число кандидатов (≥ 1).

    Returns:
        :class:`FilterResult` с топ-k кандидатами.

    Raises:
        ValueError: Если ``k`` < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept = sorted_cands[:k]
    return FilterResult(
        candidates=kept,
        n_kept=len(kept),
        n_removed=len(candidates) - len(kept),
        params={"k": k},
    )


def filter_by_rank(
    candidates: List[Candidate],
    rank_threshold: float = 0.5,
) -> FilterResult:
    """Отобрать кандидатов из верхней части ранговой шкалы.

    Args:
        candidates:      Список :class:`Candidate`.
        rank_threshold:  Доля лучших кандидатов ∈ (0, 1].

    Returns:
        :class:`FilterResult` с кандидатами из верхней доли.

    Raises:
        ValueError: Если ``rank_threshold`` вне (0, 1].
    """
    if not (0.0 < rank_threshold <= 1.0):
        raise ValueError(
            f"rank_threshold must be in (0, 1], got {rank_threshold}"
        )
    if not candidates:
        return FilterResult(
            candidates=[],
            n_kept=0,
            n_removed=0,
            params={"rank_threshold": rank_threshold},
        )
    k = max(1, int(np.ceil(len(candidates) * rank_threshold)))
    return filter_top_k(candidates, k)


def deduplicate_candidates(
    candidates: List[Candidate],
) -> FilterResult:
    """Удалить дублирующие пары кандидатов, оставив лучшую оценку.

    Пара (i, j) и (j, i) считаются одной парой.

    Args:
        candidates: Список :class:`Candidate`.

    Returns:
        :class:`FilterResult` без дублей.
    """
    seen: Dict[tuple, Candidate] = {}
    for c in candidates:
        key = (min(c.idx1, c.idx2), max(c.idx1, c.idx2))
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    kept = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    return FilterResult(
        candidates=kept,
        n_kept=len(kept),
        n_removed=len(candidates) - len(kept),
        params={"dedup": True},
    )


def normalize_scores(
    candidates: List[Candidate],
) -> List[Candidate]:
    """Нормировать оценки кандидатов к диапазону [0, 1].

    Args:
        candidates: Список :class:`Candidate`.

    Returns:
        Новый список :class:`Candidate` с нормированными оценками.
        Если все оценки равны — возвращает оценки 0.0.
    """
    if not candidates:
        return []
    scores = np.array([c.score for c in candidates], dtype=np.float64)
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max == s_min:
        norm_scores = [0.0] * len(candidates)
    else:
        norm_scores = ((scores - s_min) / (s_max - s_min)).tolist()
    return [
        Candidate(
            idx1=c.idx1,
            idx2=c.idx2,
            score=float(np.clip(ns, 0.0, 1.0)),
            params=c.params,
        )
        for c, ns in zip(candidates, norm_scores)
    ]


def merge_candidate_lists(
    lists: List[List[Candidate]],
    dedup: bool = True,
) -> List[Candidate]:
    """Объединить несколько списков кандидатов в один.

    Args:
        lists: Список списков :class:`Candidate`.
        dedup: Если ``True`` — дедуплицировать по парам.

    Returns:
        Объединённый список, отсортированный по убыванию score.
    """
    merged: List[Candidate] = []
    for lst in lists:
        merged.extend(lst)
    if dedup:
        result = deduplicate_candidates(merged)
        return result.candidates
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


def batch_filter(
    candidate_lists: List[List[Candidate]],
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> List[FilterResult]:
    """Пакетная фильтрация нескольких списков кандидатов.

    Args:
        candidate_lists: Список списков :class:`Candidate`.
        threshold:       Минимальная оценка ∈ [0, 1].
        top_k:           Если задано — дополнительно ограничить топ-k.

    Returns:
        Список :class:`FilterResult` той же длины.

    Raises:
        ValueError: Если ``threshold`` вне [0, 1] или ``top_k`` < 1.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold}"
        )
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    results = []
    for cands in candidate_lists:
        fr = filter_by_threshold(cands, threshold)
        if top_k is not None and fr.n_kept > top_k:
            fr = filter_top_k(fr.candidates, top_k)
        results.append(fr)
    return results
