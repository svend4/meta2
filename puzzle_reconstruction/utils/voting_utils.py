"""
Утилиты голосования и консенсуса для реконструкции пазла.

Объединяет результаты нескольких независимых алгоритмов (жадного,
beam search, отжига, ACO и др.) в единое согласованное решение
на основе взвешенного голосования по парам и позициям фрагментов.

Классы:
    VoteConfig   — параметры голосования

Функции:
    cast_pair_votes      — собрать голоса за пары из нескольких списков
    aggregate_pair_votes — агрегировать голоса в ранжированный список пар
    cast_position_votes  — собрать голоса за позиции фрагментов
    majority_vote        — выбрать значение большинством голосов
    weighted_vote        — взвешенное голосование по числовым значениям
    rank_fusion          — слияние нескольких ранжированных списков (RRF)
    batch_vote           — пакетное голосование по нескольким группам
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── VoteConfig ───────────────────────────────────────────────────────────────

@dataclass
class VoteConfig:
    """Параметры голосования и консенсуса.

    Attributes:
        min_votes:   Минимальное число голосов для включения результата (>= 1).
        weights:     Веса для каждого источника (None → равные веса 1.0).
        rrf_k:       Константа сглаживания для Reciprocal Rank Fusion (> 0).
        normalize:   Нормировать ли итоговые оценки в [0, 1].
    """
    min_votes: int = 1
    weights: Optional[List[float]] = None
    rrf_k: float = 60.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.min_votes < 1:
            raise ValueError(
                f"min_votes must be >= 1, got {self.min_votes}"
            )
        if self.rrf_k <= 0.0:
            raise ValueError(
                f"rrf_k must be > 0, got {self.rrf_k}"
            )
        if self.weights is not None:
            for i, w in enumerate(self.weights):
                if w < 0.0:
                    raise ValueError(
                        f"weights[{i}] must be >= 0, got {w}"
                    )


# ─── Публичные функции ────────────────────────────────────────────────────────

def cast_pair_votes(
    pair_lists: List[List[Tuple[int, int]]],
    weights: Optional[List[float]] = None,
) -> Dict[Tuple[int, int], float]:
    """Собрать взвешенные голоса за пары фрагментов.

    Каждая пара (a, b) нормализуется в каноническую форму (min, max).
    Результирующий словарь отображает каноническую пару → суммарный вес.

    Args:
        pair_lists: Список списков пар (каждый список от одного алгоритма).
        weights:    Веса для каждого списка (None → 1.0 для всех).

    Returns:
        Dict {canonical_pair: total_weight}.

    Raises:
        ValueError: Если weights задан и его длина != len(pair_lists).
    """
    if weights is not None and len(weights) != len(pair_lists):
        raise ValueError(
            f"len(weights)={len(weights)} != len(pair_lists)={len(pair_lists)}"
        )
    votes: Dict[Tuple[int, int], float] = {}
    for i, pairs in enumerate(pair_lists):
        w = 1.0 if weights is None else float(weights[i])
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            votes[key] = votes.get(key, 0.0) + w
    return votes


def aggregate_pair_votes(
    votes: Dict[Tuple[int, int], float],
    cfg: VoteConfig | None = None,
) -> List[Tuple[Tuple[int, int], float]]:
    """Агрегировать голоса в ранжированный список пар.

    Args:
        votes: Словарь {canonical_pair: weight} от cast_pair_votes.
        cfg:   Параметры (None → VoteConfig()).

    Returns:
        Список [(pair, score)] отсортированный по убыванию score.
        Пары с суммой голосов < cfg.min_votes исключаются.
        Если cfg.normalize — оценки нормируются в [0, 1].
    """
    if cfg is None:
        cfg = VoteConfig()
    filtered = [(p, v) for p, v in votes.items() if v >= cfg.min_votes]
    if not filtered:
        return []
    filtered.sort(key=lambda x: x[1], reverse=True)
    if cfg.normalize and filtered:
        max_v = filtered[0][1]
        if max_v > 0.0:
            filtered = [(p, v / max_v) for p, v in filtered]
    return filtered


def cast_position_votes(
    position_lists: List[Dict[int, float]],
    weights: Optional[List[float]] = None,
) -> Dict[int, float]:
    """Собрать взвешенные голоса за позиции (числовые значения) фрагментов.

    Args:
        position_lists: Список словарей {fragment_id: position_score}.
        weights:        Веса для каждого словаря (None → 1.0).

    Returns:
        Dict {fragment_id: weighted_sum_of_scores}.

    Raises:
        ValueError: Если weights задан и длина не совпадает.
    """
    if weights is not None and len(weights) != len(position_lists):
        raise ValueError(
            f"len(weights)={len(weights)} != len(position_lists)={len(position_lists)}"
        )
    result: Dict[int, float] = {}
    for i, pos_dict in enumerate(position_lists):
        w = 1.0 if weights is None else float(weights[i])
        for fid, score in pos_dict.items():
            result[fid] = result.get(fid, 0.0) + score * w
    return result


def majority_vote(values: List[Any]) -> Optional[Any]:
    """Выбрать значение, получившее наибольшее число голосов.

    Args:
        values: Список значений (могут повторяться).

    Returns:
        Наиболее часто встречающееся значение, или None при пустом списке.
        При равенстве голосов возвращает первое из лидирующих.
    """
    if not values:
        return None
    counts: Dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def weighted_vote(
    values: List[float],
    weights: Optional[List[float]] = None,
) -> float:
    """Взвешенное голосование по числовым значениям (взвешенное среднее).

    Args:
        values:  Список числовых значений.
        weights: Веса (None → равные веса 1.0).

    Returns:
        Взвешенное среднее; 0.0 при пустом списке или нулевых весах.

    Raises:
        ValueError: Если weights задан и его длина != len(values).
        ValueError: Если любой вес < 0.
    """
    if not values:
        return 0.0
    if weights is not None:
        if len(weights) != len(values):
            raise ValueError(
                f"len(weights)={len(weights)} != len(values)={len(values)}"
            )
        for i, w in enumerate(weights):
            if w < 0.0:
                raise ValueError(f"weights[{i}] must be >= 0, got {w}")
        total_w = sum(weights)
        if total_w == 0.0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_w
    return sum(values) / len(values)


def rank_fusion(
    ranked_lists: List[List[Any]],
    cfg: VoteConfig | None = None,
) -> List[Tuple[Any, float]]:
    """Слияние нескольких ранжированных списков методом Reciprocal Rank Fusion.

    RRF-оценка элемента: ∑_list  w_list / (k + rank(element, list))
    где rank начинается с 1.

    Args:
        ranked_lists: Список упорядоченных списков элементов
                      (первый элемент — лучший).
        cfg:          Параметры (None → VoteConfig()).

    Returns:
        Список [(element, rrf_score)] отсортированный по убыванию score.
        Если cfg.normalize — нормировать в [0, 1].
    """
    if cfg is None:
        cfg = VoteConfig()
    k = cfg.rrf_k
    weights = cfg.weights

    scores: Dict[Any, float] = {}
    for lst_idx, ranked in enumerate(ranked_lists):
        w = 1.0 if (weights is None or lst_idx >= len(weights)) else float(weights[lst_idx])
        for rank, item in enumerate(ranked, start=1):
            try:
                hashable = item if not isinstance(item, list) else tuple(item)
            except TypeError:
                hashable = id(item)
            scores[hashable] = scores.get(hashable, 0.0) + w / (k + rank)

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if cfg.normalize and result:
        max_s = result[0][1]
        if max_s > 0.0:
            result = [(item, s / max_s) for item, s in result]
    return result


def batch_vote(
    batch_pair_lists: List[List[List[Tuple[int, int]]]],
    cfg: VoteConfig | None = None,
) -> List[List[Tuple[Tuple[int, int], float]]]:
    """Пакетное голосование по нескольким независимым группам.

    Args:
        batch_pair_lists: Список групп; каждая группа — список списков пар
                          (как входные данные cast_pair_votes).
        cfg:              Параметры голосования.

    Returns:
        Список агрегированных результатов для каждой группы.

    Raises:
        ValueError: Если batch_pair_lists пуст.
    """
    if cfg is None:
        cfg = VoteConfig()
    results = []
    for pair_lists in batch_pair_lists:
        votes = cast_pair_votes(pair_lists, weights=cfg.weights)
        aggregated = aggregate_pair_votes(votes, cfg=cfg)
        results.append(aggregated)
    return results
