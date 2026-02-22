"""Глобальное сопоставление фрагментов по совокупности попарных оценок.

Модуль агрегирует результаты нескольких попарных методов сопоставления
и формирует единый рейтинг наилучших кандидатов для каждого фрагмента.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── GlobalMatchConfig ────────────────────────────────────────────────────────

@dataclass
class GlobalMatchConfig:
    """Параметры глобального матчера.

    Атрибуты:
        top_k:         Число лучших кандидатов на фрагмент (>= 1).
        min_score:     Минимальный агрегированный балл [0, 1].
        weights:       Веса каналов оценки (None = равные; если задан,
                       должен содержать ключи каналов).
        aggregate:     Метод агрегации: ``"mean"``, ``"max"``, ``"min"``.
        symmetric:     Если True, score(a, b) == score(b, a) усредняется.
    """

    top_k: int = 5
    min_score: float = 0.0
    weights: Optional[Dict[str, float]] = None
    aggregate: str = "mean"
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(
                f"top_k должен быть >= 1, получено {self.top_k}"
            )
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError(
                f"min_score должен быть в [0, 1], получено {self.min_score}"
            )
        valid_agg = {"mean", "max", "min"}
        if self.aggregate not in valid_agg:
            raise ValueError(
                f"aggregate должен быть одним из {valid_agg}, "
                f"получено '{self.aggregate}'"
            )


# ─── GlobalMatch ─────────────────────────────────────────────────────────────

@dataclass
class GlobalMatch:
    """Одна запись кандидата.

    Атрибуты:
        fragment_id: Идентификатор фрагмента-источника.
        candidate_id: Идентификатор кандидата.
        score:        Агрегированный балл [0, 1].
        channel_scores: Баллы по отдельным каналам.
        rank:          Порядковый номер в рейтинге (1 = лучший, >= 1).
    """

    fragment_id: int
    candidate_id: int
    score: float
    channel_scores: Dict[str, float] = field(default_factory=dict)
    rank: int = 1

    def __post_init__(self) -> None:
        if self.score < 0.0 or self.score > 1.0:
            raise ValueError(
                f"score должен быть в [0, 1], получено {self.score}"
            )
        if self.rank < 1:
            raise ValueError(
                f"rank должен быть >= 1, получено {self.rank}"
            )

    @property
    def is_top(self) -> bool:
        """True если rank == 1."""
        return self.rank == 1


# ─── GlobalMatchResult ────────────────────────────────────────────────────────

@dataclass
class GlobalMatchResult:
    """Результат глобального сопоставления.

    Атрибуты:
        matches: {fragment_id: [GlobalMatch, ...]} в порядке убывания score.
        n_fragments: Число обработанных фрагментов.
        n_channels: Число каналов оценки.
    """

    matches: Dict[int, List[GlobalMatch]]
    n_fragments: int
    n_channels: int

    def __post_init__(self) -> None:
        if self.n_fragments < 0:
            raise ValueError(
                f"n_fragments >= 0, получено {self.n_fragments}"
            )
        if self.n_channels < 0:
            raise ValueError(
                f"n_channels >= 0, получено {self.n_channels}"
            )

    def top_match(self, fragment_id: int) -> Optional[GlobalMatch]:
        """Вернуть лучший кандидат или None."""
        cands = self.matches.get(fragment_id, [])
        return cands[0] if cands else None

    def all_top_matches(self) -> List[GlobalMatch]:
        """Лучшие кандидаты для всех фрагментов."""
        result = []
        for cands in self.matches.values():
            if cands:
                result.append(cands[0])
        return result

    def fragment_ids(self) -> List[int]:
        """Список фрагментов с хотя бы одним кандидатом."""
        return [fid for fid, cands in self.matches.items() if cands]


# ─── _aggregate_scores ────────────────────────────────────────────────────────

def _aggregate_scores(
    channel_scores: Dict[str, float],
    weights: Optional[Dict[str, float]],
    method: str,
) -> float:
    """Агрегировать каналы в один балл."""
    if not channel_scores:
        return 0.0

    keys = list(channel_scores.keys())
    values = np.array([channel_scores[k] for k in keys], dtype=float)

    if method == "mean":
        if weights is not None:
            w = np.array([weights.get(k, 1.0) for k in keys], dtype=float)
            w_sum = w.sum()
            if w_sum < 1e-12:
                return float(np.mean(values))
            return float(np.dot(values, w) / w_sum)
        return float(np.mean(values))
    elif method == "max":
        return float(np.max(values))
    elif method == "min":
        return float(np.min(values))
    return float(np.mean(values))


# ─── aggregate_pair_scores ────────────────────────────────────────────────────

def aggregate_pair_scores(
    scores_per_channel: Dict[str, Dict[Tuple[int, int], float]],
    cfg: Optional[GlobalMatchConfig] = None,
) -> Dict[Tuple[int, int], float]:
    """Агрегировать попарные оценки из нескольких каналов.

    Аргументы:
        scores_per_channel: {канал: {(a, b): score}}.
        cfg:                Параметры.

    Возвращает:
        {(a, b): aggregated_score}.
    """
    if cfg is None:
        cfg = GlobalMatchConfig()

    all_pairs: set = set()
    for ch_scores in scores_per_channel.values():
        for pair in ch_scores:
            a, b = pair
            all_pairs.add((min(a, b), max(a, b)))

    result: Dict[Tuple[int, int], float] = {}

    for pair in all_pairs:
        a, b = pair
        channel_vals: Dict[str, float] = {}
        for ch, ch_scores in scores_per_channel.items():
            # Проверяем оба порядка
            score_ab = ch_scores.get((a, b))
            score_ba = ch_scores.get((b, a))
            if score_ab is not None and score_ba is not None and cfg.symmetric:
                channel_vals[ch] = (score_ab + score_ba) / 2.0
            elif score_ab is not None:
                channel_vals[ch] = score_ab
            elif score_ba is not None:
                channel_vals[ch] = score_ba

        agg = _aggregate_scores(channel_vals, cfg.weights, cfg.aggregate)
        agg = float(np.clip(agg, 0.0, 1.0))
        result[pair] = agg

    return result


# ─── rank_candidates ──────────────────────────────────────────────────────────

def rank_candidates(
    fragment_id: int,
    pair_scores: Dict[Tuple[int, int], float],
    cfg: Optional[GlobalMatchConfig] = None,
) -> List[GlobalMatch]:
    """Ранжировать кандидатов для одного фрагмента.

    Аргументы:
        fragment_id: Идентификатор фрагмента.
        pair_scores: Агрегированные попарные оценки.
        cfg:         Параметры.

    Возвращает:
        Список GlobalMatch по убыванию score (не более top_k записей).
    """
    if cfg is None:
        cfg = GlobalMatchConfig()

    candidates: List[Tuple[int, float]] = []

    for (a, b), score in pair_scores.items():
        if a == fragment_id:
            cand = b
        elif b == fragment_id:
            cand = a
        else:
            continue
        if score >= cfg.min_score:
            candidates.append((cand, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[: cfg.top_k]

    return [
        GlobalMatch(
            fragment_id=fragment_id,
            candidate_id=cand_id,
            score=score,
            rank=rank + 1,
        )
        for rank, (cand_id, score) in enumerate(candidates)
    ]


# ─── global_match ─────────────────────────────────────────────────────────────

def global_match(
    fragment_ids: List[int],
    scores_per_channel: Dict[str, Dict[Tuple[int, int], float]],
    cfg: Optional[GlobalMatchConfig] = None,
) -> GlobalMatchResult:
    """Выполнить глобальное сопоставление для набора фрагментов.

    Аргументы:
        fragment_ids:       Список идентификаторов фрагментов.
        scores_per_channel: {канал: {(a, b): score}}.
        cfg:                Параметры.

    Возвращает:
        GlobalMatchResult.
    """
    if cfg is None:
        cfg = GlobalMatchConfig()

    pair_scores = aggregate_pair_scores(scores_per_channel, cfg)

    matches: Dict[int, List[GlobalMatch]] = {}
    for fid in fragment_ids:
        matches[fid] = rank_candidates(fid, pair_scores, cfg)

    return GlobalMatchResult(
        matches=matches,
        n_fragments=len(fragment_ids),
        n_channels=len(scores_per_channel),
    )


# ─── filter_matches ───────────────────────────────────────────────────────────

def filter_matches(
    result: GlobalMatchResult,
    min_score: float = 0.0,
) -> GlobalMatchResult:
    """Удалить совпадения с баллом ниже порога.

    Аргументы:
        result:    GlobalMatchResult.
        min_score: Порог [0, 1].

    Возвращает:
        Новый GlobalMatchResult с обновлёнными рангами.
    """
    if not (0.0 <= min_score <= 1.0):
        raise ValueError(
            f"min_score должен быть в [0, 1], получено {min_score}"
        )

    new_matches: Dict[int, List[GlobalMatch]] = {}
    for fid, cands in result.matches.items():
        filtered = [c for c in cands if c.score >= min_score]
        # Переиндексировать ранги
        new_matches[fid] = [
            GlobalMatch(
                fragment_id=m.fragment_id,
                candidate_id=m.candidate_id,
                score=m.score,
                channel_scores=m.channel_scores,
                rank=i + 1,
            )
            for i, m in enumerate(filtered)
        ]

    return GlobalMatchResult(
        matches=new_matches,
        n_fragments=result.n_fragments,
        n_channels=result.n_channels,
    )


# ─── merge_match_results ──────────────────────────────────────────────────────

def merge_match_results(
    results: List[GlobalMatchResult],
    cfg: Optional[GlobalMatchConfig] = None,
) -> GlobalMatchResult:
    """Объединить несколько GlobalMatchResult в один.

    Аргументы:
        results: Список результатов.
        cfg:     Параметры (используется top_k).

    Возвращает:
        GlobalMatchResult с объединёнными и переранжированными кандидатами.
    """
    if cfg is None:
        cfg = GlobalMatchConfig()
    if not results:
        return GlobalMatchResult(matches={}, n_fragments=0, n_channels=0)

    all_fids: set = set()
    for r in results:
        all_fids.update(r.matches.keys())

    merged_matches: Dict[int, List[GlobalMatch]] = {}
    for fid in all_fids:
        cand_scores: Dict[int, List[float]] = {}
        for r in results:
            for m in r.matches.get(fid, []):
                cand_scores.setdefault(m.candidate_id, []).append(m.score)

        # Усреднить оценки по результатам
        aggregated: List[Tuple[int, float]] = [
            (cid, float(np.mean(scores)))
            for cid, scores in cand_scores.items()
        ]
        aggregated.sort(key=lambda x: x[1], reverse=True)
        aggregated = aggregated[: cfg.top_k]

        merged_matches[fid] = [
            GlobalMatch(
                fragment_id=fid,
                candidate_id=cid,
                score=score,
                rank=rank + 1,
            )
            for rank, (cid, score) in enumerate(aggregated)
        ]

    total_channels = sum(r.n_channels for r in results)
    return GlobalMatchResult(
        matches=merged_matches,
        n_fragments=len(all_fids),
        n_channels=total_channels,
    )
