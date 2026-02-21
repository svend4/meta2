"""
Ранжирование пар-кандидатов для сборки пазла.

Принимает матрицу или список оценок совместимости пар фрагментов
и возвращает их в порядке убывания приоритета. Поддерживает
фильтрацию по порогу, топ-K выборку и жадную дедупликацию (каждый
фрагмент используется не более одного раза).

Классы:
    CandidatePair — оценённая пара кандидатов с метаданными

Функции:
    score_pair        — создать CandidatePair из индексов и оценки
    rank_pairs        — сортировка списка CandidatePair по score↓
    filter_by_score   — фильтрация пар по нижнему порогу
    top_k             — топ-K пар (с опциональной жадной дедупликацией)
    deduplicate_pairs — жадное удаление конфликтов (каждый фрагмент ≤ 1 раз)
    batch_rank        — ранжирование квадратной матрицы N×N оценок
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── CandidatePair ────────────────────────────────────────────────────────────

@dataclass
class CandidatePair:
    """
    Оценённая пара фрагментов-кандидатов.

    Attributes:
        idx1:  Индекс первого фрагмента.
        idx2:  Индекс второго фрагмента.
        score: Оценка совместимости ∈ [0,1]; выше → лучше.
        meta:  Дополнительные метаданные (каналы, метод, ...).
    """
    idx1:  int
    idx2:  int
    score: float
    meta:  Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"CandidatePair(({self.idx1},{self.idx2}), "
                f"score={self.score:.4f})")

    # Сравнение для сортировки: большая оценка → «меньший» элемент
    # (чтобы sorted() давал убывание при reverse=True)
    def __lt__(self, other: "CandidatePair") -> bool:
        return self.score > other.score


# ─── score_pair ───────────────────────────────────────────────────────────────

def score_pair(idx1:  int,
                idx2:  int,
                score: float,
                **meta) -> CandidatePair:
    """
    Создаёт CandidatePair из индексов и оценки.

    Args:
        idx1:  Индекс первого фрагмента.
        idx2:  Индекс второго фрагмента.
        score: Оценка совместимости.
        **meta: Произвольные метаданные.

    Returns:
        CandidatePair.
    """
    return CandidatePair(idx1=idx1, idx2=idx2, score=float(score), meta=dict(meta))


# ─── rank_pairs ───────────────────────────────────────────────────────────────

def rank_pairs(pairs: List[CandidatePair]) -> List[CandidatePair]:
    """
    Сортирует список пар по убыванию оценки.

    Args:
        pairs: Список CandidatePair.

    Returns:
        Новый список, отсортированный по score↓.
    """
    return sorted(pairs, key=lambda p: p.score, reverse=True)


# ─── filter_by_score ──────────────────────────────────────────────────────────

def filter_by_score(pairs:     List[CandidatePair],
                     threshold: float = 0.5) -> List[CandidatePair]:
    """
    Возвращает пары с оценкой выше порога.

    Args:
        pairs:     Список CandidatePair.
        threshold: Нижний порог (включительно не принимается: score > threshold).

    Returns:
        Отфильтрованный список, отсортированный по score↓.
    """
    return [p for p in rank_pairs(pairs) if p.score > threshold]


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

def deduplicate_pairs(pairs: List[CandidatePair]) -> List[CandidatePair]:
    """
    Жадно удаляет пары, в которых повторяются уже использованные фрагменты.

    Пары обрабатываются в порядке убывания оценки. Пара включается в
    результат только если оба её индекса ещё не «заняты».

    Args:
        pairs: Список CandidatePair (не обязательно отсортированный).

    Returns:
        Подсписок без повторяющихся индексов фрагментов.
    """
    used:   set = set()
    result: List[CandidatePair] = []
    for p in rank_pairs(pairs):
        if p.idx1 not in used and p.idx2 not in used:
            result.append(p)
            used.add(p.idx1)
            used.add(p.idx2)
    return result


# ─── top_k ────────────────────────────────────────────────────────────────────

def top_k(pairs:       List[CandidatePair],
           k:           int,
           deduplicate: bool = False) -> List[CandidatePair]:
    """
    Возвращает топ-K пар по убыванию оценки.

    Args:
        pairs:       Список CandidatePair.
        k:           Количество возвращаемых пар.
        deduplicate: Применить жадную дедупликацию перед отбором топ-K.

    Returns:
        Список из min(k, len(pairs)) пар.
    """
    k = max(0, k)
    if deduplicate:
        ranked = deduplicate_pairs(pairs)
    else:
        ranked = rank_pairs(pairs)
    return ranked[:k]


# ─── batch_rank ───────────────────────────────────────────────────────────────

def batch_rank(score_matrix: np.ndarray,
                threshold:    float = 0.0,
                symmetric:    bool  = True) -> List[CandidatePair]:
    """
    Превращает квадратную матрицу оценок N×N в ранжированный список пар.

    Args:
        score_matrix: Массив (N, N) с оценками ∈ [0,1].
                      score_matrix[i, j] — оценка совместимости фрагментов i и j.
        threshold:    Нижний порог; пары с score ≤ threshold отбрасываются.
        symmetric:    True → рассматривать только пары i < j (верхний треугольник).

    Returns:
        Список CandidatePair, отсортированный по score↓.

    Raises:
        ValueError: Если score_matrix не квадратная или не 2D.
    """
    mat = np.asarray(score_matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D, got shape {mat.shape}."
        )
    n, m = mat.shape
    if n != m:
        raise ValueError(
            f"score_matrix must be square (N×N), got {n}×{m}."
        )

    pairs: List[CandidatePair] = []
    for i in range(n):
        j_start = i + 1 if symmetric else 0
        for j in range(j_start, n):
            if i == j:
                continue
            s = float(mat[i, j])
            if s > threshold:
                pairs.append(CandidatePair(idx1=i, idx2=j, score=s))

    return rank_pairs(pairs)
