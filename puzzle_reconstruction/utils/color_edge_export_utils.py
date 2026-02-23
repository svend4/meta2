"""Утилиты анализа цветового сопоставления, детекции краёв и экспорта.

Предоставляет Config/Entry/Summary-структуры и вспомогательные функции
для анализа результатов цветового сопоставления фрагментов, статистик
краёв и результатов экспорта сборки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence


# ─── ColorMatch utilities ─────────────────────────────────────────────────────

@dataclass
class ColorMatchAnalysisConfig:
    """Конфигурация для анализа результатов цветового сопоставления."""
    min_score: float = 0.0
    colorspace: str = "hsv"
    metric: str = "bhatt"


@dataclass
class ColorMatchAnalysisEntry:
    """Запись результата цветового сопоставления пары фрагментов."""
    idx1: int
    idx2: int
    score: float
    hist_score: float
    moment_score: float
    profile_score: float
    method: str = "hsv"


@dataclass
class ColorMatchAnalysisSummary:
    """Сводная статистика набора ColorMatchAnalysisEntry."""
    n_entries: int
    mean_score: float
    mean_hist: float
    mean_moment: float
    mean_profile: float
    min_score: float
    max_score: float


def make_color_match_analysis_entry(
    idx1: int,
    idx2: int,
    score: float,
    hist_score: float,
    moment_score: float,
    profile_score: float,
    method: str = "hsv",
) -> ColorMatchAnalysisEntry:
    """Создать ColorMatchAnalysisEntry."""
    return ColorMatchAnalysisEntry(
        idx1=idx1, idx2=idx2, score=score,
        hist_score=hist_score, moment_score=moment_score,
        profile_score=profile_score, method=method,
    )


def summarise_color_match_analysis(
    entries: Sequence[ColorMatchAnalysisEntry],
) -> ColorMatchAnalysisSummary:
    """Вычислить сводную статистику по списку ColorMatchAnalysisEntry."""
    n = len(entries)
    if n == 0:
        return ColorMatchAnalysisSummary(
            n_entries=0, mean_score=0.0, mean_hist=0.0,
            mean_moment=0.0, mean_profile=0.0,
            min_score=0.0, max_score=0.0,
        )
    scores = [e.score for e in entries]
    return ColorMatchAnalysisSummary(
        n_entries=n,
        mean_score=mean(scores),
        mean_hist=mean(e.hist_score for e in entries),
        mean_moment=mean(e.moment_score for e in entries),
        mean_profile=mean(e.profile_score for e in entries),
        min_score=min(scores),
        max_score=max(scores),
    )


def filter_strong_color_matches(
    entries: Sequence[ColorMatchAnalysisEntry],
    threshold: float = 0.5,
) -> List[ColorMatchAnalysisEntry]:
    """Оставить записи с score >= threshold."""
    return [e for e in entries if e.score >= threshold]


def filter_weak_color_matches(
    entries: Sequence[ColorMatchAnalysisEntry],
    threshold: float = 0.5,
) -> List[ColorMatchAnalysisEntry]:
    """Оставить записи с score < threshold."""
    return [e for e in entries if e.score < threshold]


def filter_color_by_method(
    entries: Sequence[ColorMatchAnalysisEntry],
    method: str,
) -> List[ColorMatchAnalysisEntry]:
    """Оставить записи с указанным методом."""
    return [e for e in entries if e.method == method]


def top_k_color_match_entries(
    entries: Sequence[ColorMatchAnalysisEntry],
    k: int,
) -> List[ColorMatchAnalysisEntry]:
    """Вернуть k записей с наибольшим score."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_color_match_entry(
    entries: Sequence[ColorMatchAnalysisEntry],
) -> Optional[ColorMatchAnalysisEntry]:
    """Запись с наибольшим score или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.score)


def color_match_analysis_stats(
    entries: Sequence[ColorMatchAnalysisEntry],
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


def compare_color_match_summaries(
    a: ColorMatchAnalysisSummary,
    b: ColorMatchAnalysisSummary,
) -> Dict[str, float]:
    """Сравнить две ColorMatchAnalysisSummary."""
    return {
        "mean_score_delta": a.mean_score - b.mean_score,
        "mean_hist_delta": a.mean_hist - b.mean_hist,
        "mean_moment_delta": a.mean_moment - b.mean_moment,
        "mean_profile_delta": a.mean_profile - b.mean_profile,
    }


def batch_summarise_color_match_analysis(
    groups: Sequence[Sequence[ColorMatchAnalysisEntry]],
) -> List[ColorMatchAnalysisSummary]:
    """Вычислить сводку для каждой группы ColorMatchAnalysisEntry."""
    return [summarise_color_match_analysis(g) for g in groups]


# ─── EdgeDetection utilities ──────────────────────────────────────────────────

@dataclass
class EdgeDetectionAnalysisConfig:
    """Конфигурация для анализа результатов детекции краёв."""
    min_density: float = 0.0
    method: str = "canny"


@dataclass
class EdgeDetectionAnalysisEntry:
    """Запись результата детекции краёв фрагмента."""
    fragment_id: int
    density: float
    n_contours: int
    method: str = "canny"


@dataclass
class EdgeDetectionAnalysisSummary:
    """Сводная статистика набора EdgeDetectionAnalysisEntry."""
    n_entries: int
    mean_density: float
    mean_contours: float
    min_density: float
    max_density: float
    methods: List[str]


def make_edge_detection_entry(
    fragment_id: int,
    density: float,
    n_contours: int,
    method: str = "canny",
) -> EdgeDetectionAnalysisEntry:
    """Создать EdgeDetectionAnalysisEntry."""
    return EdgeDetectionAnalysisEntry(
        fragment_id=fragment_id, density=density,
        n_contours=n_contours, method=method,
    )


def summarise_edge_detection_entries(
    entries: Sequence[EdgeDetectionAnalysisEntry],
) -> EdgeDetectionAnalysisSummary:
    """Вычислить сводную статистику по списку EdgeDetectionAnalysisEntry."""
    n = len(entries)
    if n == 0:
        return EdgeDetectionAnalysisSummary(
            n_entries=0, mean_density=0.0, mean_contours=0.0,
            min_density=0.0, max_density=0.0, methods=[],
        )
    densities = [e.density for e in entries]
    methods = sorted({e.method for e in entries})
    return EdgeDetectionAnalysisSummary(
        n_entries=n,
        mean_density=mean(densities),
        mean_contours=mean(e.n_contours for e in entries),
        min_density=min(densities),
        max_density=max(densities),
        methods=methods,
    )


def filter_edge_by_min_density(
    entries: Sequence[EdgeDetectionAnalysisEntry],
    min_density: float,
) -> List[EdgeDetectionAnalysisEntry]:
    """Оставить записи с density >= min_density."""
    return [e for e in entries if e.density >= min_density]


def filter_edge_by_method(
    entries: Sequence[EdgeDetectionAnalysisEntry],
    method: str,
) -> List[EdgeDetectionAnalysisEntry]:
    """Оставить записи с указанным методом."""
    return [e for e in entries if e.method == method]


def top_k_edge_density_entries(
    entries: Sequence[EdgeDetectionAnalysisEntry],
    k: int,
) -> List[EdgeDetectionAnalysisEntry]:
    """Вернуть k записей с наибольшей density."""
    return sorted(entries, key=lambda e: e.density, reverse=True)[:k]


def best_edge_density_entry(
    entries: Sequence[EdgeDetectionAnalysisEntry],
) -> Optional[EdgeDetectionAnalysisEntry]:
    """Запись с наибольшей density или None."""
    if not entries:
        return None
    return max(entries, key=lambda e: e.density)


def edge_detection_stats(
    entries: Sequence[EdgeDetectionAnalysisEntry],
) -> Dict[str, float]:
    """Словарь базовых статистик по density."""
    densities = [e.density for e in entries]
    if not densities:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(densities)),
        "mean": mean(densities),
        "std": stdev(densities) if len(densities) > 1 else 0.0,
        "min": min(densities),
        "max": max(densities),
    }


def compare_edge_detection_summaries(
    a: EdgeDetectionAnalysisSummary,
    b: EdgeDetectionAnalysisSummary,
) -> Dict[str, float]:
    """Сравнить две EdgeDetectionAnalysisSummary."""
    return {
        "mean_density_delta": a.mean_density - b.mean_density,
        "mean_contours_delta": a.mean_contours - b.mean_contours,
    }


def batch_summarise_edge_detection_entries(
    groups: Sequence[Sequence[EdgeDetectionAnalysisEntry]],
) -> List[EdgeDetectionAnalysisSummary]:
    """Вычислить сводку для каждой группы EdgeDetectionAnalysisEntry."""
    return [summarise_edge_detection_entries(g) for g in groups]
