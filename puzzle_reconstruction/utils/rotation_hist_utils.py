"""Утилиты анализа поворотов изображений и статистики гистограмм.

Модуль предоставляет вспомогательные структуры и функции для:
- Отслеживания и агрегирования результатов поворотов фрагментов
- Статистического анализа гистограммных расстояний между фрагментами
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RotationAnalysisConfig:
    """Конфигурация анализа результатов поворота.

    Атрибуты:
        min_confidence: Минимальное значение достоверности для фильтрации.
        discrete_steps: Количество дискретных шагов угла (4 → 0°, 90°, 180°, 270°).
        angle_tolerance_deg: Допуск (°) при сравнении углов.
    """
    min_confidence: float = 0.0
    discrete_steps: int = 4
    angle_tolerance_deg: float = 5.0


@dataclass
class RotationAnalysisEntry:
    """Запись результата анализа поворота одного фрагмента.

    Атрибуты:
        fragment_id: Идентификатор фрагмента.
        angle_deg:   Оценённый угол поворота в градусах.
        confidence:  Достоверность оценки ∈ [0, 1].
        method:      Метод оценки (например, 'procrustes', 'phase').
        params:      Дополнительные параметры.
    """
    fragment_id: int
    angle_deg: float
    confidence: float
    method: str
    params: Dict = field(default_factory=dict)


@dataclass
class RotationAnalysisSummary:
    """Сводка анализа поворотов набора фрагментов.

    Атрибуты:
        n_entries:       Число записей.
        mean_angle_deg:  Средний угол.
        std_angle_deg:   СКО угла.
        mean_confidence: Средняя достоверность.
        dominant_angle:  Наиболее частый дискретный угол.
        methods_used:    Список использованных методов.
    """
    n_entries: int
    mean_angle_deg: float
    std_angle_deg: float
    mean_confidence: float
    dominant_angle: float
    methods_used: List[str]


def make_rotation_analysis_entry(
    fragment_id: int,
    angle_deg: float,
    confidence: float,
    method: str = "procrustes",
    **params,
) -> RotationAnalysisEntry:
    """Создать запись анализа поворота.

    Args:
        fragment_id: Идентификатор фрагмента.
        angle_deg:   Оценённый угол в градусах.
        confidence:  Достоверность ∈ [0, 1].
        method:      Метод оценки.
        **params:    Дополнительные параметры.

    Returns:
        :class:`RotationAnalysisEntry`.
    """
    return RotationAnalysisEntry(
        fragment_id=fragment_id,
        angle_deg=float(angle_deg),
        confidence=float(confidence),
        method=method,
        params=dict(params),
    )


def summarise_rotation_analysis(
    entries: Sequence[RotationAnalysisEntry],
    cfg: Optional[RotationAnalysisConfig] = None,
) -> RotationAnalysisSummary:
    """Сформировать сводку анализа поворотов.

    Args:
        entries: Список записей анализа.
        cfg:     Конфигурация (None → RotationAnalysisConfig()).

    Returns:
        :class:`RotationAnalysisSummary`.
    """
    if cfg is None:
        cfg = RotationAnalysisConfig()
    if not entries:
        return RotationAnalysisSummary(
            n_entries=0,
            mean_angle_deg=0.0,
            std_angle_deg=0.0,
            mean_confidence=0.0,
            dominant_angle=0.0,
            methods_used=[],
        )
    angles = [e.angle_deg for e in entries]
    confs = [e.confidence for e in entries]
    mean_a = sum(angles) / len(angles)
    variance = sum((a - mean_a) ** 2 for a in angles) / len(angles)
    std_a = variance ** 0.5
    mean_c = sum(confs) / len(confs)

    step = 360.0 / max(cfg.discrete_steps, 1)
    bins: Dict[float, int] = {}
    for a in angles:
        key = round((a % 360.0) / step) * step % 360.0
        bins[key] = bins.get(key, 0) + 1
    dominant = max(bins, key=lambda k: bins[k]) if bins else 0.0

    methods = list(dict.fromkeys(e.method for e in entries))
    return RotationAnalysisSummary(
        n_entries=len(entries),
        mean_angle_deg=mean_a,
        std_angle_deg=std_a,
        mean_confidence=mean_c,
        dominant_angle=dominant,
        methods_used=methods,
    )


def filter_rotation_by_confidence(
    entries: Sequence[RotationAnalysisEntry],
    min_confidence: float,
) -> List[RotationAnalysisEntry]:
    """Отфильтровать записи по минимальной достоверности."""
    return [e for e in entries if e.confidence >= min_confidence]


def filter_rotation_by_method(
    entries: Sequence[RotationAnalysisEntry],
    method: str,
) -> List[RotationAnalysisEntry]:
    """Отфильтровать записи по методу оценки."""
    return [e for e in entries if e.method == method]


def filter_rotation_by_angle_range(
    entries: Sequence[RotationAnalysisEntry],
    min_deg: float,
    max_deg: float,
) -> List[RotationAnalysisEntry]:
    """Отфильтровать записи по диапазону углов."""
    return [e for e in entries if min_deg <= e.angle_deg <= max_deg]


def top_k_rotation_entries(
    entries: Sequence[RotationAnalysisEntry],
    k: int,
) -> List[RotationAnalysisEntry]:
    """Вернуть k записей с наибольшей достоверностью."""
    return sorted(entries, key=lambda e: e.confidence, reverse=True)[:k]


def best_rotation_entry(
    entries: Sequence[RotationAnalysisEntry],
) -> Optional[RotationAnalysisEntry]:
    """Вернуть запись с наибольшей достоверностью."""
    return max(entries, key=lambda e: e.confidence) if entries else None


def rotation_angle_stats(
    entries: Sequence[RotationAnalysisEntry],
) -> Dict:
    """Вычислить статистику углов: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    angles = [e.angle_deg for e in entries]
    mean = sum(angles) / len(angles)
    var = sum((a - mean) ** 2 for a in angles) / len(angles)
    return {
        "min": min(angles),
        "max": max(angles),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(angles),
    }


def compare_rotation_summaries(
    a: RotationAnalysisSummary,
    b: RotationAnalysisSummary,
) -> Dict:
    """Сравнить две сводки анализа поворотов."""
    return {
        "delta_mean_angle_deg": b.mean_angle_deg - a.mean_angle_deg,
        "delta_mean_confidence": b.mean_confidence - a.mean_confidence,
        "delta_n_entries": b.n_entries - a.n_entries,
        "same_dominant_angle": abs(b.dominant_angle - a.dominant_angle) < 1e-6,
    }


def batch_summarise_rotation_analysis(
    groups: Sequence[Sequence[RotationAnalysisEntry]],
    cfg: Optional[RotationAnalysisConfig] = None,
) -> List[RotationAnalysisSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_rotation_analysis(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Histogram Distance Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HistogramDistanceConfig:
    """Конфигурация анализа гистограммных расстояний.

    Атрибуты:
        max_distance:  Максимальное допустимое расстояние для фильтрации.
        metric:        Метрика расстояния ('emd', 'chi2', 'intersection').
    """
    max_distance: float = 1.0
    metric: str = "emd"


@dataclass
class HistogramDistanceEntry:
    """Запись гистограммного расстояния между парой фрагментов.

    Атрибуты:
        frag_a:   Идентификатор первого фрагмента.
        frag_b:   Идентификатор второго фрагмента.
        distance: Расстояние между гистограммами.
        metric:   Используемая метрика.
        params:   Дополнительные параметры.
    """
    frag_a: int
    frag_b: int
    distance: float
    metric: str
    params: Dict = field(default_factory=dict)


@dataclass
class HistogramDistanceSummary:
    """Сводка гистограммных расстояний.

    Атрибуты:
        n_pairs:       Число пар.
        mean_distance: Среднее расстояние.
        min_distance:  Минимальное расстояние.
        max_distance:  Максимальное расстояние.
        metric:        Метрика.
    """
    n_pairs: int
    mean_distance: float
    min_distance: float
    max_distance: float
    metric: str


def make_histogram_distance_entry(
    frag_a: int,
    frag_b: int,
    distance: float,
    metric: str = "emd",
    **params,
) -> HistogramDistanceEntry:
    """Создать запись гистограммного расстояния.

    Args:
        frag_a:   ID первого фрагмента.
        frag_b:   ID второго фрагмента.
        distance: Расстояние.
        metric:   Метрика ('emd', 'chi2', 'intersection').
        **params: Дополнительные параметры.

    Returns:
        :class:`HistogramDistanceEntry`.
    """
    return HistogramDistanceEntry(
        frag_a=frag_a,
        frag_b=frag_b,
        distance=float(distance),
        metric=metric,
        params=dict(params),
    )


def summarise_histogram_distance_entries(
    entries: Sequence[HistogramDistanceEntry],
    cfg: Optional[HistogramDistanceConfig] = None,
) -> HistogramDistanceSummary:
    """Сформировать сводку по записям гистограммных расстояний.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → HistogramDistanceConfig()).

    Returns:
        :class:`HistogramDistanceSummary`.
    """
    if cfg is None:
        cfg = HistogramDistanceConfig()
    metric = entries[0].metric if entries else cfg.metric
    if not entries:
        return HistogramDistanceSummary(
            n_pairs=0,
            mean_distance=0.0,
            min_distance=0.0,
            max_distance=0.0,
            metric=metric,
        )
    dists = [e.distance for e in entries]
    mean = sum(dists) / len(dists)
    return HistogramDistanceSummary(
        n_pairs=len(entries),
        mean_distance=mean,
        min_distance=min(dists),
        max_distance=max(dists),
        metric=metric,
    )


def filter_histogram_by_max_distance(
    entries: Sequence[HistogramDistanceEntry],
    max_distance: float,
) -> List[HistogramDistanceEntry]:
    """Отфильтровать записи, оставив только пары с малым расстоянием."""
    return [e for e in entries if e.distance <= max_distance]


def filter_histogram_by_metric(
    entries: Sequence[HistogramDistanceEntry],
    metric: str,
) -> List[HistogramDistanceEntry]:
    """Отфильтровать записи по метрике расстояния."""
    return [e for e in entries if e.metric == metric]


def filter_histogram_by_fragment(
    entries: Sequence[HistogramDistanceEntry],
    fragment_id: int,
) -> List[HistogramDistanceEntry]:
    """Отфильтровать записи, содержащие заданный фрагмент."""
    return [e for e in entries if e.frag_a == fragment_id or e.frag_b == fragment_id]


def top_k_closest_histogram_pairs(
    entries: Sequence[HistogramDistanceEntry],
    k: int,
) -> List[HistogramDistanceEntry]:
    """Вернуть k пар с наименьшим расстоянием."""
    return sorted(entries, key=lambda e: e.distance)[:k]


def best_histogram_distance_entry(
    entries: Sequence[HistogramDistanceEntry],
) -> Optional[HistogramDistanceEntry]:
    """Вернуть пару с минимальным расстоянием."""
    return min(entries, key=lambda e: e.distance) if entries else None


def histogram_distance_stats(
    entries: Sequence[HistogramDistanceEntry],
) -> Dict:
    """Вычислить статистику расстояний: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    dists = [e.distance for e in entries]
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    return {
        "min": min(dists),
        "max": max(dists),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(dists),
    }


def compare_histogram_distance_summaries(
    a: HistogramDistanceSummary,
    b: HistogramDistanceSummary,
) -> Dict:
    """Сравнить две сводки гистограммных расстояний."""
    return {
        "delta_mean_distance": b.mean_distance - a.mean_distance,
        "delta_n_pairs": b.n_pairs - a.n_pairs,
        "same_metric": a.metric == b.metric,
    }


def batch_summarise_histogram_distance_entries(
    groups: Sequence[Sequence[HistogramDistanceEntry]],
    cfg: Optional[HistogramDistanceConfig] = None,
) -> List[HistogramDistanceSummary]:
    """Сформировать сводки для нескольких групп записей расстояний."""
    return [summarise_histogram_distance_entries(g, cfg) for g in groups]
