"""Утилиты анализа результатов последовательности фрагментов и зазоров.

Предоставляет Config/Entry/Summary-структуры и вспомогательные функции
для анализа результатов последовательного упорядочивания фрагментов
и статистик зазоров между фрагментами в собранном пазле.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple


# ─── SequenceScore utilities ──────────────────────────────────────────────────

@dataclass
class SequenceScoreConfig:
    """Конфигурация для анализа результатов последовательности."""
    min_score: float = 0.0
    require_full: bool = True


@dataclass
class SequenceScoreEntry:
    """Запись одного результата упорядочивания фрагментов."""
    seq_id: int
    order: List[int]
    total_score: float
    n_fragments: int
    algorithm: str = "greedy"
    is_full: bool = True


@dataclass
class SequenceScoreSummary:
    """Сводная статистика набора SequenceScoreEntry."""
    n_entries: int
    mean_score: float
    mean_fragments: float
    min_score: float
    max_score: float
    n_full: int
    algorithms: List[str]


def make_sequence_score_entry(
    seq_id: int,
    order: List[int],
    total_score: float,
    n_fragments: int,
    algorithm: str = "greedy",
    is_full: bool = True,
) -> SequenceScoreEntry:
    """Создать SequenceScoreEntry."""
    return SequenceScoreEntry(
        seq_id=seq_id,
        order=order,
        total_score=total_score,
        n_fragments=n_fragments,
        algorithm=algorithm,
        is_full=is_full,
    )


def summarise_sequence_score_entries(
    entries: Sequence[SequenceScoreEntry],
) -> SequenceScoreSummary:
    """Вычислить сводную статистику по списку SequenceScoreEntry."""
    n = len(entries)
    if n == 0:
        return SequenceScoreSummary(
            n_entries=0, mean_score=0.0, mean_fragments=0.0,
            min_score=0.0, max_score=0.0, n_full=0, algorithms=[],
        )
    scores = [e.total_score for e in entries]
    algos = sorted({e.algorithm for e in entries})
    return SequenceScoreSummary(
        n_entries=n,
        mean_score=mean(scores),
        mean_fragments=mean(e.n_fragments for e in entries),
        min_score=min(scores),
        max_score=max(scores),
        n_full=sum(1 for e in entries if e.is_full),
        algorithms=algos,
    )


def filter_full_sequences(
    entries: Sequence[SequenceScoreEntry],
) -> List[SequenceScoreEntry]:
    """Оставить только полные последовательности."""
    return [e for e in entries if e.is_full]


def filter_sequence_by_min_score(
    entries: Sequence[SequenceScoreEntry],
    min_score: float,
) -> List[SequenceScoreEntry]:
    """Оставить записи с total_score >= min_score."""
    return [e for e in entries if e.total_score >= min_score]


def filter_sequence_by_algorithm(
    entries: Sequence[SequenceScoreEntry],
    algorithm: str,
) -> List[SequenceScoreEntry]:
    """Оставить записи с указанным алгоритмом."""
    return [e for e in entries if e.algorithm == algorithm]


def top_k_sequence_entries(
    entries: Sequence[SequenceScoreEntry],
    k: int,
) -> List[SequenceScoreEntry]:
    """Вернуть k записей с наибольшим total_score."""
    return sorted(entries, key=lambda e: e.total_score, reverse=True)[:k]


def best_sequence_entry(
    entries: Sequence[SequenceScoreEntry],
) -> Optional[SequenceScoreEntry]:
    """Запись с наибольшим total_score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.total_score)


def sequence_score_stats(
    entries: Sequence[SequenceScoreEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по total_score."""
    scores = [e.total_score for e in entries]
    if not scores:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(scores)),
        "mean": mean(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def compare_sequence_summaries(
    a: SequenceScoreSummary,
    b: SequenceScoreSummary,
) -> Dict[str, float]:
    """Сравнить две SequenceScoreSummary; вернуть разности метрик."""
    return {
        "mean_score_delta": a.mean_score - b.mean_score,
        "mean_fragments_delta": a.mean_fragments - b.mean_fragments,
        "n_full_delta": float(a.n_full - b.n_full),
    }


def batch_summarise_sequence_score_entries(
    groups: Sequence[Sequence[SequenceScoreEntry]],
) -> List[SequenceScoreSummary]:
    """Вычислить сводку для каждой группы SequenceScoreEntry."""
    return [summarise_sequence_score_entries(g) for g in groups]


# ─── GapScore utilities ───────────────────────────────────────────────────────

@dataclass
class GapScoreConfig:
    """Конфигурация для анализа результатов зазоров."""
    near_threshold: float = 10.0
    overlap_penalty: float = 1.0


@dataclass
class GapScoreEntry:
    """Запись одного результата анализа зазора между парой фрагментов."""
    id1: int
    id2: int
    gap_x: float
    gap_y: float
    distance: float
    category: str = "near"


@dataclass
class GapScoreSummary:
    """Сводная статистика набора GapScoreEntry."""
    n_entries: int
    mean_distance: float
    mean_gap_x: float
    mean_gap_y: float
    n_overlapping: int
    n_touching: int
    n_near: int
    n_far: int
    min_distance: float
    max_distance: float


def make_gap_score_entry(
    id1: int,
    id2: int,
    gap_x: float,
    gap_y: float,
    distance: float,
    category: str = "near",
) -> GapScoreEntry:
    """Создать GapScoreEntry."""
    return GapScoreEntry(
        id1=id1, id2=id2, gap_x=gap_x, gap_y=gap_y,
        distance=distance, category=category,
    )


def summarise_gap_score_entries(
    entries: Sequence[GapScoreEntry],
) -> GapScoreSummary:
    """Вычислить сводную статистику по списку GapScoreEntry."""
    n = len(entries)
    if n == 0:
        return GapScoreSummary(
            n_entries=0, mean_distance=0.0, mean_gap_x=0.0, mean_gap_y=0.0,
            n_overlapping=0, n_touching=0, n_near=0, n_far=0,
            min_distance=0.0, max_distance=0.0,
        )
    dists = [e.distance for e in entries]
    return GapScoreSummary(
        n_entries=n,
        mean_distance=mean(dists),
        mean_gap_x=mean(e.gap_x for e in entries),
        mean_gap_y=mean(e.gap_y for e in entries),
        n_overlapping=sum(1 for e in entries if e.category == "overlap"),
        n_touching=sum(1 for e in entries if e.category == "touching"),
        n_near=sum(1 for e in entries if e.category == "near"),
        n_far=sum(1 for e in entries if e.category == "far"),
        min_distance=min(dists),
        max_distance=max(dists),
    )


def filter_overlapping_gaps(
    entries: Sequence[GapScoreEntry],
) -> List[GapScoreEntry]:
    """Оставить только перекрывающиеся пары (category == 'overlap')."""
    return [e for e in entries if e.category == "overlap"]


def filter_gap_by_category(
    entries: Sequence[GapScoreEntry],
    category: str,
) -> List[GapScoreEntry]:
    """Оставить записи с указанной категорией."""
    return [e for e in entries if e.category == category]


def filter_gap_by_max_distance(
    entries: Sequence[GapScoreEntry],
    max_distance: float,
) -> List[GapScoreEntry]:
    """Оставить записи с distance <= max_distance."""
    return [e for e in entries if e.distance <= max_distance]


def top_k_closest_gaps(
    entries: Sequence[GapScoreEntry],
    k: int,
) -> List[GapScoreEntry]:
    """Вернуть k записей с наименьшим distance."""
    return sorted(entries, key=lambda e: e.distance)[:k]


def best_gap_entry(
    entries: Sequence[GapScoreEntry],
) -> Optional[GapScoreEntry]:
    """Запись с минимальным distance или None."""
    if not entries:
        return None
    return min(entries, key=lambda e: e.distance)


def gap_score_stats(
    entries: Sequence[GapScoreEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по distance."""
    dists = [e.distance for e in entries]
    if not dists:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(dists)),
        "mean": mean(dists),
        "std": stdev(dists) if len(dists) > 1 else 0.0,
        "min": min(dists),
        "max": max(dists),
    }


def compare_gap_summaries(
    a: GapScoreSummary,
    b: GapScoreSummary,
) -> Dict[str, float]:
    """Сравнить две GapScoreSummary; вернуть разности метрик."""
    return {
        "mean_distance_delta": a.mean_distance - b.mean_distance,
        "n_overlapping_delta": float(a.n_overlapping - b.n_overlapping),
        "n_far_delta": float(a.n_far - b.n_far),
    }


def batch_summarise_gap_score_entries(
    groups: Sequence[Sequence[GapScoreEntry]],
) -> List[GapScoreSummary]:
    """Вычислить сводку для каждой группы GapScoreEntry."""
    return [summarise_gap_score_entries(g) for g in groups]
