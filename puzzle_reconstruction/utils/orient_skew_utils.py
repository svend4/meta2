"""Утилиты ориентационного анализа, коррекции наклона и сравнения краёв.

Предоставляет Config/Entry/Summary-структуры и вспомогательные функции
для анализа результатов вычисления ориентационных профилей фрагментов,
коррекции перспективного наклона, а также попарного сравнения краёв.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple


# ─── OrientMatch utilities ────────────────────────────────────────────────────

@dataclass
class OrientMatchConfig:
    """Конфигурация для анализа результатов ориентационного сопоставления."""
    min_score: float = 0.0
    max_angle: float = 180.0


@dataclass
class OrientMatchEntry:
    """Запись результата ориентационного сопоставления пары фрагментов."""
    fragment_a: int
    fragment_b: int
    best_angle: float
    best_score: float
    n_angles_tested: int


@dataclass
class OrientMatchSummary:
    """Сводная статистика набора OrientMatchEntry."""
    n_entries: int
    mean_score: float
    mean_angle: float
    min_score: float
    max_score: float
    high_score_count: int


def make_orient_match_entry(
    fragment_a: int,
    fragment_b: int,
    best_angle: float,
    best_score: float,
    n_angles_tested: int,
) -> OrientMatchEntry:
    """Создать OrientMatchEntry."""
    return OrientMatchEntry(
        fragment_a=fragment_a,
        fragment_b=fragment_b,
        best_angle=best_angle,
        best_score=best_score,
        n_angles_tested=n_angles_tested,
    )


def summarise_orient_match_entries(
    entries: Sequence[OrientMatchEntry],
) -> OrientMatchSummary:
    """Вычислить сводную статистику по списку OrientMatchEntry."""
    n = len(entries)
    if n == 0:
        return OrientMatchSummary(
            n_entries=0, mean_score=0.0, mean_angle=0.0,
            min_score=0.0, max_score=0.0, high_score_count=0,
        )
    scores = [e.best_score for e in entries]
    angles = [e.best_angle for e in entries]
    return OrientMatchSummary(
        n_entries=n,
        mean_score=mean(scores),
        mean_angle=mean(angles),
        min_score=min(scores),
        max_score=max(scores),
        high_score_count=sum(1 for s in scores if s >= 0.7),
    )


def filter_high_orient_matches(
    entries: Sequence[OrientMatchEntry],
    threshold: float = 0.7,
) -> List[OrientMatchEntry]:
    """Оставить записи с best_score >= threshold."""
    return [e for e in entries if e.best_score >= threshold]


def filter_low_orient_matches(
    entries: Sequence[OrientMatchEntry],
    threshold: float = 0.7,
) -> List[OrientMatchEntry]:
    """Оставить записи с best_score < threshold."""
    return [e for e in entries if e.best_score < threshold]


def filter_orient_by_score_range(
    entries: Sequence[OrientMatchEntry],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[OrientMatchEntry]:
    """Фильтровать по диапазону best_score."""
    return [e for e in entries if lo <= e.best_score <= hi]


def filter_orient_by_max_angle(
    entries: Sequence[OrientMatchEntry],
    max_angle: float,
) -> List[OrientMatchEntry]:
    """Оставить записи с best_angle <= max_angle."""
    return [e for e in entries if e.best_angle <= max_angle]


def top_k_orient_match_entries(
    entries: Sequence[OrientMatchEntry],
    k: int,
) -> List[OrientMatchEntry]:
    """Вернуть k записей с наибольшим best_score."""
    return sorted(entries, key=lambda e: e.best_score, reverse=True)[:k]


def best_orient_match_entry(
    entries: Sequence[OrientMatchEntry],
) -> Optional[OrientMatchEntry]:
    """Запись с наибольшим best_score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.best_score)


def orient_match_stats(
    entries: Sequence[OrientMatchEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по best_score."""
    scores = [e.best_score for e in entries]
    if not scores:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(scores)),
        "mean": mean(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def compare_orient_summaries(
    a: OrientMatchSummary,
    b: OrientMatchSummary,
) -> Dict[str, float]:
    """Сравнить две OrientMatchSummary; вернуть разности метрик."""
    return {
        "mean_score_delta": a.mean_score - b.mean_score,
        "mean_angle_delta": a.mean_angle - b.mean_angle,
        "high_score_count_delta": float(a.high_score_count - b.high_score_count),
    }


def batch_summarise_orient_match_entries(
    groups: Sequence[Sequence[OrientMatchEntry]],
) -> List[OrientMatchSummary]:
    """Вычислить сводку для каждой группы OrientMatchEntry."""
    return [summarise_orient_match_entries(g) for g in groups]


# ─── SkewCorrection utilities ─────────────────────────────────────────────────

@dataclass
class SkewCorrConfig:
    """Конфигурация для анализа результатов коррекции наклона."""
    min_confidence: float = 0.0
    method: str = "auto"


@dataclass
class SkewCorrEntry:
    """Запись результата коррекции наклона одного изображения."""
    image_id: int
    angle_deg: float
    confidence: float
    method: str


@dataclass
class SkewCorrSummary:
    """Сводная статистика набора SkewCorrEntry."""
    n_entries: int
    mean_angle_deg: float
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    dominant_method: str


def make_skew_corr_entry(
    image_id: int,
    angle_deg: float,
    confidence: float,
    method: str,
) -> SkewCorrEntry:
    """Создать SkewCorrEntry."""
    return SkewCorrEntry(
        image_id=image_id,
        angle_deg=angle_deg,
        confidence=confidence,
        method=method,
    )


def summarise_skew_corr_entries(
    entries: Sequence[SkewCorrEntry],
) -> SkewCorrSummary:
    """Вычислить сводную статистику по списку SkewCorrEntry."""
    n = len(entries)
    if n == 0:
        return SkewCorrSummary(
            n_entries=0, mean_angle_deg=0.0, mean_confidence=0.0,
            min_confidence=0.0, max_confidence=0.0, dominant_method="",
        )
    angles = [e.angle_deg for e in entries]
    confs = [e.confidence for e in entries]
    methods = [e.method for e in entries]
    dominant = max(set(methods), key=methods.count)
    return SkewCorrSummary(
        n_entries=n,
        mean_angle_deg=mean(angles),
        mean_confidence=mean(confs),
        min_confidence=min(confs),
        max_confidence=max(confs),
        dominant_method=dominant,
    )


def filter_high_confidence_skew(
    entries: Sequence[SkewCorrEntry],
    threshold: float = 0.5,
) -> List[SkewCorrEntry]:
    """Оставить записи с confidence >= threshold."""
    return [e for e in entries if e.confidence >= threshold]


def filter_skew_by_method(
    entries: Sequence[SkewCorrEntry],
    method: str,
) -> List[SkewCorrEntry]:
    """Фильтровать по методу коррекции."""
    return [e for e in entries if e.method == method]


def filter_skew_by_angle_range(
    entries: Sequence[SkewCorrEntry],
    lo: float,
    hi: float,
) -> List[SkewCorrEntry]:
    """Оставить записи с angle_deg в диапазоне [lo, hi]."""
    return [e for e in entries if lo <= e.angle_deg <= hi]


def top_k_skew_entries(
    entries: Sequence[SkewCorrEntry],
    k: int,
) -> List[SkewCorrEntry]:
    """Вернуть k записей с наибольшей confidence."""
    return sorted(entries, key=lambda e: e.confidence, reverse=True)[:k]


def best_skew_entry(
    entries: Sequence[SkewCorrEntry],
) -> Optional[SkewCorrEntry]:
    """Запись с наибольшей confidence или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.confidence)


def skew_corr_stats(
    entries: Sequence[SkewCorrEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по confidence."""
    confs = [e.confidence for e in entries]
    if not confs:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(confs)),
        "mean": mean(confs),
        "std": stdev(confs) if len(confs) > 1 else 0.0,
        "min": min(confs),
        "max": max(confs),
    }


def compare_skew_summaries(
    a: SkewCorrSummary,
    b: SkewCorrSummary,
) -> Dict[str, float]:
    """Сравнить две SkewCorrSummary; вернуть разности метрик."""
    return {
        "mean_confidence_delta": a.mean_confidence - b.mean_confidence,
        "mean_angle_delta": a.mean_angle_deg - b.mean_angle_deg,
    }


def batch_summarise_skew_corr_entries(
    groups: Sequence[Sequence[SkewCorrEntry]],
) -> List[SkewCorrSummary]:
    """Вычислить сводку для каждой группы SkewCorrEntry."""
    return [summarise_skew_corr_entries(g) for g in groups]
