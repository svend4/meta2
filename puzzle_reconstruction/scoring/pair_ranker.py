"""Ранжирование пар фрагментов по совокупным оценкам.

Модуль предоставляет классы и функции для создания ранжированных
списков пар: взвешенное ранжирование, ранжирование по множеству
метрик, фильтрация дубликатов и построение матрицы рангов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── RankConfig ───────────────────────────────────────────────────────────────

@dataclass
class RankConfig:
    """Параметры ранжирования пар.

    Атрибуты:
        top_k:           Число лучших пар (0 = все; >= 0).
        ascending:       Сортировать по возрастанию (по умолчанию — по убыванию).
        deduplicate:     Удалять симметричные дубликаты (i,j) == (j,i).
        min_score:       Минимальная оценка для включения в ранг (>= 0).
        score_field:     Поле для сортировки ('score' | 'rank').
    """

    top_k: int = 0
    ascending: bool = False
    deduplicate: bool = True
    min_score: float = 0.0
    score_field: str = "score"

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ValueError(
                f"top_k должен быть >= 0, получено {self.top_k}"
            )
        if self.min_score < 0.0:
            raise ValueError(
                f"min_score должен быть >= 0, получено {self.min_score}"
            )
        if self.score_field not in ("score", "rank"):
            raise ValueError(
                f"score_field должен быть 'score' или 'rank', "
                f"получено '{self.score_field}'"
            )


# ─── RankedPair ───────────────────────────────────────────────────────────────

@dataclass
class RankedPair:
    """Пара фрагментов с рангом и оценкой.

    Атрибуты:
        pair:   Пара идентификаторов (a, b); a < b если deduplicate=True.
        score:  Итоговая оценка пары (>= 0).
        rank:   Ранг (1-based; >= 1).
        scores: Исходные оценки по отдельным метрикам.
    """

    pair: Tuple[int, int]
    score: float
    rank: int
    scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score < 0.0:
            raise ValueError(
                f"score должен быть >= 0, получено {self.score}"
            )
        if self.rank < 1:
            raise ValueError(
                f"rank должен быть >= 1, получено {self.rank}"
            )

    @property
    def fragment_a(self) -> int:
        """Первый фрагмент."""
        return self.pair[0]

    @property
    def fragment_b(self) -> int:
        """Второй фрагмент."""
        return self.pair[1]

    @property
    def n_metrics(self) -> int:
        """Число метрик в scores."""
        return len(self.scores)


# ─── RankResult ───────────────────────────────────────────────────────────────

@dataclass
class RankResult:
    """Итог ранжирования набора пар.

    Атрибуты:
        ranked:       Список RankedPair в порядке ранга.
        n_pairs:      Общее число пар до фильтрации (>= 0).
        n_ranked:     Число пар в итоговом списке (>= 0).
        top_score:    Наивысшая оценка (>= 0).
        mean_score:   Среднее значение оценок (>= 0).
    """

    ranked: List[RankedPair]
    n_pairs: int
    n_ranked: int
    top_score: float
    mean_score: float

    def __post_init__(self) -> None:
        for name, val in (
            ("n_pairs", self.n_pairs),
            ("n_ranked", self.n_ranked),
        ):
            if val < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {val}")
        if self.top_score < 0.0:
            raise ValueError(
                f"top_score должен быть >= 0, получено {self.top_score}"
            )
        if self.mean_score < 0.0:
            raise ValueError(
                f"mean_score должен быть >= 0, получено {self.mean_score}"
            )

    @property
    def top_pair(self) -> Optional[Tuple[int, int]]:
        """Пара с наивысшей оценкой (None если пусто)."""
        return self.ranked[0].pair if self.ranked else None

    @property
    def compression_ratio(self) -> float:
        """Доля оставленных пар (n_ranked / n_pairs; 0 если n_pairs == 0)."""
        if self.n_pairs == 0:
            return 0.0
        return float(self.n_ranked) / float(self.n_pairs)


# ─── _normalize_pair ──────────────────────────────────────────────────────────

def _normalize_pair(pair: Tuple[int, int]) -> Tuple[int, int]:
    """Привести пару к каноническому виду (меньший ID первым)."""
    a, b = pair
    return (min(a, b), max(a, b))


# ─── compute_pair_score ───────────────────────────────────────────────────────

def compute_pair_score(
    metric_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Вычислить итоговую оценку пары из словаря метрик.

    Аргументы:
        metric_scores: Словарь {metric_name: score} (все значения >= 0).
        weights:       Словарь весов (None → равные веса 1.0).

    Возвращает:
        Взвешенная средняя оценка (>= 0).

    Исключения:
        ValueError: Если metric_scores пуст или содержит отрицательные значения.
    """
    if not metric_scores:
        raise ValueError("metric_scores не должен быть пустым")
    for k, v in metric_scores.items():
        if v < 0.0:
            raise ValueError(
                f"Оценка '{k}' должна быть >= 0, получено {v}"
            )

    if weights is None:
        weights = {k: 1.0 for k in metric_scores}

    w_sum = sum(weights.get(k, 1.0) for k in metric_scores) + 1e-12
    score = sum(
        metric_scores[k] * weights.get(k, 1.0)
        for k in metric_scores
    ) / w_sum
    return float(np.clip(score, 0.0, None))


# ─── rank_pairs ───────────────────────────────────────────────────────────────

def rank_pairs(
    pairs: List[Tuple[int, int]],
    scores: List[float],
    cfg: Optional[RankConfig] = None,
    metric_scores_list: Optional[List[Dict[str, float]]] = None,
) -> RankResult:
    """Ранжировать пары фрагментов по оценкам.

    Аргументы:
        pairs:              Список пар идентификаторов.
        scores:             Список итоговых оценок (по одной на пару).
        cfg:                Параметры ранжирования.
        metric_scores_list: Список словарей с оценками по метрикам (опционально).

    Возвращает:
        RankResult.

    Исключения:
        ValueError: Если len(pairs) != len(scores).
    """
    if cfg is None:
        cfg = RankConfig()
    if len(pairs) != len(scores):
        raise ValueError(
            f"Длины pairs ({len(pairs)}) и scores ({len(scores)}) не совпадают"
        )

    # Нормализовать пары
    norm_pairs = [_normalize_pair(p) if cfg.deduplicate else p
                  for p in pairs]

    # Дедупликация
    seen: dict = {}
    for i, (p, s) in enumerate(zip(norm_pairs, scores)):
        if p not in seen or s > seen[p][0]:
            seen[p] = (s, i)

    dedup_pairs = list(seen.keys())
    dedup_scores = [seen[p][0] for p in dedup_pairs]
    dedup_indices = [seen[p][1] for p in dedup_pairs]

    # Фильтр по min_score
    filtered = [(p, s, idx) for p, s, idx in
                zip(dedup_pairs, dedup_scores, dedup_indices)
                if s >= cfg.min_score]

    n_pairs_total = len(pairs)

    if not filtered:
        return RankResult(
            ranked=[], n_pairs=n_pairs_total, n_ranked=0,
            top_score=0.0, mean_score=0.0
        )

    # Сортировка
    filtered.sort(key=lambda x: x[1], reverse=not cfg.ascending)

    # Ограничение top_k
    if cfg.top_k > 0:
        filtered = filtered[:cfg.top_k]

    # Построение результата
    ranked: List[RankedPair] = []
    for rank_idx, (p, s, orig_idx) in enumerate(filtered, start=1):
        ms = (metric_scores_list[orig_idx]
              if metric_scores_list is not None else {})
        ranked.append(RankedPair(pair=p, score=s, rank=rank_idx, scores=ms))

    top_score = float(max(rp.score for rp in ranked))
    mean_score = float(np.mean([rp.score for rp in ranked]))

    return RankResult(
        ranked=ranked,
        n_pairs=n_pairs_total,
        n_ranked=len(ranked),
        top_score=top_score,
        mean_score=mean_score,
    )


# ─── build_rank_matrix ────────────────────────────────────────────────────────

def build_rank_matrix(
    result: RankResult,
    n_fragments: int,
) -> np.ndarray:
    """Построить матрицу рангов n_fragments × n_fragments.

    Аргументы:
        result:       RankResult.
        n_fragments:  Число фрагментов (>= 1).

    Возвращает:
        Целочисленная матрица рангов (0 = пара не вошла в результат).

    Исключения:
        ValueError: Если n_fragments < 1.
    """
    if n_fragments < 1:
        raise ValueError(
            f"n_fragments должен быть >= 1, получено {n_fragments}"
        )
    matrix = np.zeros((n_fragments, n_fragments), dtype=int)
    for rp in result.ranked:
        a, b = rp.pair
        if 0 <= a < n_fragments and 0 <= b < n_fragments:
            matrix[a, b] = rp.rank
            matrix[b, a] = rp.rank
    return matrix


# ─── merge_rank_results ───────────────────────────────────────────────────────

def merge_rank_results(
    results: List[RankResult],
    cfg: Optional[RankConfig] = None,
) -> RankResult:
    """Объединить несколько RankResult в один.

    Аргументы:
        results: Список RankResult.
        cfg:     Параметры (для повторного ранжирования).

    Возвращает:
        Объединённый RankResult.

    Исключения:
        ValueError: Если results пуст.
    """
    if not results:
        raise ValueError("results не должен быть пустым")

    all_pairs: List[Tuple[int, int]] = []
    all_scores: List[float] = []
    for r in results:
        for rp in r.ranked:
            all_pairs.append(rp.pair)
            all_scores.append(rp.score)

    return rank_pairs(all_pairs, all_scores, cfg)
