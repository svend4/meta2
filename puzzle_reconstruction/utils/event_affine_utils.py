"""Утилиты журналирования событий и аффинного сопоставления фрагментов.

Предоставляет Config/Entry/Summary-структуры и вспомогательные функции
для анализа записей журнала событий пайплайна и результатов аффинного
сопоставления пар фрагментов.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence


# ─── EventRecord utilities ────────────────────────────────────────────────────

@dataclass
class EventRecordConfig:
    """Конфигурация для анализа записей событий."""
    min_level: str = "debug"
    namespace: str = "default"


@dataclass
class EventRecordEntry:
    """Запись одного события пайплайна для анализа."""
    event_id: int
    name: str
    level: str
    timestamp: float
    is_error: bool = False


@dataclass
class EventRecordSummary:
    """Сводная статистика набора EventRecordEntry."""
    n_entries: int
    n_errors: int
    n_warnings: int
    error_rate: float
    unique_names: int
    time_span: float


def make_event_record_entry(
    event_id: int,
    name: str,
    level: str,
    timestamp: float,
    is_error: bool = False,
) -> EventRecordEntry:
    """Создать EventRecordEntry."""
    return EventRecordEntry(
        event_id=event_id,
        name=name,
        level=level,
        timestamp=timestamp,
        is_error=is_error,
    )


def summarise_event_record_entries(
    entries: Sequence[EventRecordEntry],
) -> EventRecordSummary:
    """Вычислить сводную статистику по списку EventRecordEntry."""
    n = len(entries)
    if n == 0:
        return EventRecordSummary(
            n_entries=0, n_errors=0, n_warnings=0,
            error_rate=0.0, unique_names=0, time_span=0.0,
        )
    n_errors = sum(1 for e in entries if e.is_error)
    n_warnings = sum(1 for e in entries if e.level == "warning")
    names = {e.name for e in entries}
    ts = [e.timestamp for e in entries]
    return EventRecordSummary(
        n_entries=n,
        n_errors=n_errors,
        n_warnings=n_warnings,
        error_rate=n_errors / n,
        unique_names=len(names),
        time_span=max(ts) - min(ts),
    )


def filter_error_events(
    entries: Sequence[EventRecordEntry],
) -> List[EventRecordEntry]:
    """Оставить только записи ошибок."""
    return [e for e in entries if e.is_error]


def filter_events_by_level(
    entries: Sequence[EventRecordEntry],
    level: str,
) -> List[EventRecordEntry]:
    """Оставить записи с указанным уровнем."""
    return [e for e in entries if e.level == level]


def filter_events_by_name(
    entries: Sequence[EventRecordEntry],
    name: str,
) -> List[EventRecordEntry]:
    """Оставить записи с указанным именем."""
    return [e for e in entries if e.name == name]


def filter_events_by_time_range(
    entries: Sequence[EventRecordEntry],
    t_lo: float,
    t_hi: float,
) -> List[EventRecordEntry]:
    """Оставить записи в диапазоне timestamp [t_lo, t_hi]."""
    return [e for e in entries if t_lo <= e.timestamp <= t_hi]


def top_k_recent_events(
    entries: Sequence[EventRecordEntry],
    k: int,
) -> List[EventRecordEntry]:
    """Вернуть k наиболее свежих записей (по timestamp)."""
    return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:k]


def latest_event_entry(
    entries: Sequence[EventRecordEntry],
) -> Optional[EventRecordEntry]:
    """Наиболее свежая запись или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.timestamp)


def event_record_stats(
    entries: Sequence[EventRecordEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по меткам времени."""
    ts = [e.timestamp for e in entries]
    if not ts:
        return {"count": 0, "min": 0.0, "max": 0.0, "span": 0.0}
    return {
        "count": float(len(ts)),
        "min": min(ts),
        "max": max(ts),
        "span": max(ts) - min(ts),
        "error_rate": sum(1 for e in entries if e.is_error) / len(ts),
    }


def compare_event_summaries(
    a: EventRecordSummary,
    b: EventRecordSummary,
) -> Dict[str, float]:
    """Сравнить две EventRecordSummary; вернуть разности метрик."""
    return {
        "error_rate_delta": a.error_rate - b.error_rate,
        "n_errors_delta": float(a.n_errors - b.n_errors),
        "unique_names_delta": float(a.unique_names - b.unique_names),
    }


def batch_summarise_event_record_entries(
    groups: Sequence[Sequence[EventRecordEntry]],
) -> List[EventRecordSummary]:
    """Вычислить сводку для каждой группы EventRecordEntry."""
    return [summarise_event_record_entries(g) for g in groups]


# ─── AffineMatch utilities ─────────────────────────────────────────────────────

@dataclass
class AffineMatchConfig:
    """Конфигурация для анализа результатов аффинного сопоставления."""
    min_score: float = 0.0
    min_inliers: int = 0


@dataclass
class AffineMatchEntry:
    """Запись результата аффинного сопоставления пары фрагментов."""
    idx1: int
    idx2: int
    score: float
    n_inliers: int
    reprojection_error: float
    has_transform: bool = True


@dataclass
class AffineMatchSummary:
    """Сводная статистика набора AffineMatchEntry."""
    n_entries: int
    mean_score: float
    mean_inliers: float
    mean_reprojection: float
    min_score: float
    max_score: float
    n_no_transform: int


def make_affine_match_entry(
    idx1: int,
    idx2: int,
    score: float,
    n_inliers: int,
    reprojection_error: float,
    has_transform: bool = True,
) -> AffineMatchEntry:
    """Создать AffineMatchEntry."""
    return AffineMatchEntry(
        idx1=idx1,
        idx2=idx2,
        score=score,
        n_inliers=n_inliers,
        reprojection_error=reprojection_error,
        has_transform=has_transform,
    )


def summarise_affine_match_entries(
    entries: Sequence[AffineMatchEntry],
) -> AffineMatchSummary:
    """Вычислить сводную статистику по списку AffineMatchEntry."""
    n = len(entries)
    if n == 0:
        return AffineMatchSummary(
            n_entries=0, mean_score=0.0, mean_inliers=0.0,
            mean_reprojection=0.0, min_score=0.0, max_score=0.0,
            n_no_transform=0,
        )
    scores = [e.score for e in entries]
    return AffineMatchSummary(
        n_entries=n,
        mean_score=mean(scores),
        mean_inliers=mean(e.n_inliers for e in entries),
        mean_reprojection=mean(e.reprojection_error for e in entries),
        min_score=min(scores),
        max_score=max(scores),
        n_no_transform=sum(1 for e in entries if not e.has_transform),
    )


def filter_strong_affine_matches(
    entries: Sequence[AffineMatchEntry],
    threshold: float = 0.5,
) -> List[AffineMatchEntry]:
    """Оставить записи с score >= threshold."""
    return [e for e in entries if e.score >= threshold]


def filter_weak_affine_matches(
    entries: Sequence[AffineMatchEntry],
    threshold: float = 0.5,
) -> List[AffineMatchEntry]:
    """Оставить записи с score < threshold."""
    return [e for e in entries if e.score < threshold]


def filter_affine_by_inliers(
    entries: Sequence[AffineMatchEntry],
    min_inliers: int,
) -> List[AffineMatchEntry]:
    """Оставить записи с n_inliers >= min_inliers."""
    return [e for e in entries if e.n_inliers >= min_inliers]


def filter_affine_with_transform(
    entries: Sequence[AffineMatchEntry],
) -> List[AffineMatchEntry]:
    """Оставить только записи с найденным аффинным преобразованием."""
    return [e for e in entries if e.has_transform]


def top_k_affine_match_entries(
    entries: Sequence[AffineMatchEntry],
    k: int,
) -> List[AffineMatchEntry]:
    """Вернуть k записей с наибольшим score."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_affine_match_entry(
    entries: Sequence[AffineMatchEntry],
) -> Optional[AffineMatchEntry]:
    """Запись с наибольшим score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.score)


def affine_match_stats(
    entries: Sequence[AffineMatchEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по score."""
    scores = [e.score for e in entries]
    if not scores:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(scores)),
        "mean": mean(scores),
        "std": stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def compare_affine_summaries(
    a: AffineMatchSummary,
    b: AffineMatchSummary,
) -> Dict[str, float]:
    """Сравнить две AffineMatchSummary; вернуть разности метрик."""
    return {
        "mean_score_delta": a.mean_score - b.mean_score,
        "mean_inliers_delta": a.mean_inliers - b.mean_inliers,
        "mean_reprojection_delta": a.mean_reprojection - b.mean_reprojection,
    }


def batch_summarise_affine_match_entries(
    groups: Sequence[Sequence[AffineMatchEntry]],
) -> List[AffineMatchSummary]:
    """Вычислить сводку для каждой группы AffineMatchEntry."""
    return [summarise_affine_match_entries(g) for g in groups]
