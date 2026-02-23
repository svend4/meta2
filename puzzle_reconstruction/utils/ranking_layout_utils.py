"""Утилиты анализа результатов глобального ранжирования и компоновки.

Модуль предоставляет вспомогательные структуры и функции для:
- Отслеживания и агрегирования результатов глобального ранжирования пар
- Статистического анализа оценок компоновки фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Global Ranking Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlobalRankingConfig:
    """Конфигурация анализа результатов глобального ранжирования.

    Атрибуты:
        min_score:    Минимальный порог оценки для фильтрации.
        top_k:        Число лучших пар для выборки.
        source_names: Ожидаемые источники оценок.
    """
    min_score: float = 0.0
    top_k: int = 10
    source_names: List[str] = field(default_factory=list)


@dataclass
class GlobalRankingEntry:
    """Запись результата глобального ранжирования одной пары.

    Атрибуты:
        idx1:             Индекс первого фрагмента.
        idx2:             Индекс второго фрагмента.
        score:            Агрегированная оценка ∈ [0, 1].
        rank:             Ранг пары (0 = лучший).
        component_scores: Оценки по источникам.
    """
    idx1: int
    idx2: int
    score: float
    rank: int
    component_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class GlobalRankingSummary:
    """Сводка глобального ранжирования.

    Атрибуты:
        n_pairs:       Число пар.
        mean_score:    Средняя оценка.
        max_score:     Максимальная оценка.
        min_score:     Минимальная оценка.
        top_pair:      Индексы лучшей пары (idx1, idx2) или None.
    """
    n_pairs: int
    mean_score: float
    max_score: float
    min_score: float
    top_pair: Optional[tuple]


def make_global_ranking_entry(
    idx1: int,
    idx2: int,
    score: float,
    rank: int,
    **component_scores: float,
) -> GlobalRankingEntry:
    """Создать запись результата глобального ранжирования.

    Args:
        idx1:              Индекс первого фрагмента.
        idx2:              Индекс второго фрагмента.
        score:             Агрегированная оценка.
        rank:              Ранг.
        **component_scores: Оценки по источникам.

    Returns:
        :class:`GlobalRankingEntry`.
    """
    return GlobalRankingEntry(
        idx1=idx1,
        idx2=idx2,
        score=float(score),
        rank=rank,
        component_scores=dict(component_scores),
    )


def summarise_global_ranking_entries(
    entries: Sequence[GlobalRankingEntry],
    cfg: Optional[GlobalRankingConfig] = None,
) -> GlobalRankingSummary:
    """Сформировать сводку по результатам глобального ранжирования.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → GlobalRankingConfig()).

    Returns:
        :class:`GlobalRankingSummary`.
    """
    if cfg is None:
        cfg = GlobalRankingConfig()
    if not entries:
        return GlobalRankingSummary(
            n_pairs=0,
            mean_score=0.0,
            max_score=0.0,
            min_score=0.0,
            top_pair=None,
        )
    scores = [e.score for e in entries]
    mean = sum(scores) / len(scores)
    best = min(entries, key=lambda e: e.rank)
    return GlobalRankingSummary(
        n_pairs=len(entries),
        mean_score=mean,
        max_score=max(scores),
        min_score=min(scores),
        top_pair=(best.idx1, best.idx2),
    )


def filter_ranking_by_min_score(
    entries: Sequence[GlobalRankingEntry],
    min_score: float,
) -> List[GlobalRankingEntry]:
    """Отфильтровать записи по минимальной оценке."""
    return [e for e in entries if e.score >= min_score]


def filter_ranking_by_fragment(
    entries: Sequence[GlobalRankingEntry],
    fragment_idx: int,
) -> List[GlobalRankingEntry]:
    """Отфильтровать записи, содержащие заданный фрагмент."""
    return [e for e in entries if e.idx1 == fragment_idx or e.idx2 == fragment_idx]


def filter_ranking_by_top_k(
    entries: Sequence[GlobalRankingEntry],
    k: int,
) -> List[GlobalRankingEntry]:
    """Вернуть k записей с наименьшим рангом."""
    return sorted(entries, key=lambda e: e.rank)[:k]


def top_k_ranking_entries(
    entries: Sequence[GlobalRankingEntry],
    k: int,
) -> List[GlobalRankingEntry]:
    """Вернуть k записей с наибольшей оценкой."""
    return sorted(entries, key=lambda e: e.score, reverse=True)[:k]


def best_ranking_entry(
    entries: Sequence[GlobalRankingEntry],
) -> Optional[GlobalRankingEntry]:
    """Вернуть запись с наибольшей оценкой."""
    return max(entries, key=lambda e: e.score) if entries else None


def ranking_score_stats(
    entries: Sequence[GlobalRankingEntry],
) -> Dict:
    """Вычислить статистику оценок: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    scores = [e.score for e in entries]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(scores),
    }


def compare_global_ranking_summaries(
    a: GlobalRankingSummary,
    b: GlobalRankingSummary,
) -> Dict:
    """Сравнить две сводки глобального ранжирования."""
    return {
        "delta_mean_score": b.mean_score - a.mean_score,
        "delta_n_pairs": b.n_pairs - a.n_pairs,
        "delta_max_score": b.max_score - a.max_score,
    }


def batch_summarise_global_ranking_entries(
    groups: Sequence[Sequence[GlobalRankingEntry]],
    cfg: Optional[GlobalRankingConfig] = None,
) -> List[GlobalRankingSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_global_ranking_entries(g, cfg) for g in groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Layout Scoring Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayoutScoringConfig:
    """Конфигурация анализа результатов оценки компоновки.

    Атрибуты:
        min_total_score: Минимальный порог итогового балла.
        quality_filter:  Фильтр по уровню качества ('poor', 'fair', 'good', 'excellent').
    """
    min_total_score: float = 0.0
    quality_filter: Optional[str] = None


@dataclass
class LayoutScoringEntry:
    """Запись результата оценки одного варианта компоновки.

    Атрибуты:
        layout_id:       Идентификатор варианта.
        total_score:     Итоговый балл.
        coverage:        Покрытие холста.
        overlap_ratio:   Доля перекрытий.
        uniformity:      Равномерность распределения.
        n_fragments:     Число размещённых фрагментов.
        quality_level:   Уровень качества.
        params:          Дополнительные параметры.
    """
    layout_id: int
    total_score: float
    coverage: float
    overlap_ratio: float
    uniformity: float
    n_fragments: int
    quality_level: str = "poor"
    params: Dict = field(default_factory=dict)


@dataclass
class LayoutScoringSummary:
    """Сводка оценки нескольких вариантов компоновки.

    Атрибуты:
        n_layouts:        Число вариантов.
        mean_total_score: Средний итоговый балл.
        best_layout_id:   ID лучшего варианта.
        worst_layout_id:  ID худшего варианта.
        quality_counts:   Количество вариантов каждого уровня качества.
    """
    n_layouts: int
    mean_total_score: float
    best_layout_id: Optional[int]
    worst_layout_id: Optional[int]
    quality_counts: Dict[str, int]


def make_layout_scoring_entry(
    layout_id: int,
    total_score: float,
    coverage: float,
    overlap_ratio: float,
    uniformity: float,
    n_fragments: int,
    quality_level: str = "poor",
    **params,
) -> LayoutScoringEntry:
    """Создать запись результата оценки компоновки.

    Args:
        layout_id:     ID варианта.
        total_score:   Итоговый балл.
        coverage:      Покрытие.
        overlap_ratio: Доля перекрытий.
        uniformity:    Равномерность.
        n_fragments:   Число фрагментов.
        quality_level: Уровень качества.
        **params:      Дополнительные параметры.

    Returns:
        :class:`LayoutScoringEntry`.
    """
    return LayoutScoringEntry(
        layout_id=layout_id,
        total_score=float(total_score),
        coverage=float(coverage),
        overlap_ratio=float(overlap_ratio),
        uniformity=float(uniformity),
        n_fragments=n_fragments,
        quality_level=quality_level,
        params=dict(params),
    )


def summarise_layout_scoring_entries(
    entries: Sequence[LayoutScoringEntry],
    cfg: Optional[LayoutScoringConfig] = None,
) -> LayoutScoringSummary:
    """Сформировать сводку по записям оценки компоновки.

    Args:
        entries: Список записей.
        cfg:     Конфигурация (None → LayoutScoringConfig()).

    Returns:
        :class:`LayoutScoringSummary`.
    """
    if cfg is None:
        cfg = LayoutScoringConfig()
    if not entries:
        return LayoutScoringSummary(
            n_layouts=0,
            mean_total_score=0.0,
            best_layout_id=None,
            worst_layout_id=None,
            quality_counts={},
        )
    scores = [e.total_score for e in entries]
    mean = sum(scores) / len(scores)
    best = max(entries, key=lambda e: e.total_score)
    worst = min(entries, key=lambda e: e.total_score)
    qc: Dict[str, int] = {}
    for e in entries:
        qc[e.quality_level] = qc.get(e.quality_level, 0) + 1
    return LayoutScoringSummary(
        n_layouts=len(entries),
        mean_total_score=mean,
        best_layout_id=best.layout_id,
        worst_layout_id=worst.layout_id,
        quality_counts=qc,
    )


def filter_layout_by_min_score(
    entries: Sequence[LayoutScoringEntry],
    min_score: float,
) -> List[LayoutScoringEntry]:
    """Отфильтровать варианты по минимальному итоговому баллу."""
    return [e for e in entries if e.total_score >= min_score]


def filter_layout_by_quality(
    entries: Sequence[LayoutScoringEntry],
    quality: str,
) -> List[LayoutScoringEntry]:
    """Отфильтровать варианты по уровню качества."""
    return [e for e in entries if e.quality_level == quality]


def filter_layout_by_max_overlap(
    entries: Sequence[LayoutScoringEntry],
    max_overlap: float,
) -> List[LayoutScoringEntry]:
    """Отфильтровать варианты по максимально допустимому перекрытию."""
    return [e for e in entries if e.overlap_ratio <= max_overlap]


def top_k_layout_entries(
    entries: Sequence[LayoutScoringEntry],
    k: int,
) -> List[LayoutScoringEntry]:
    """Вернуть k лучших вариантов по итоговому баллу."""
    return sorted(entries, key=lambda e: e.total_score, reverse=True)[:k]


def best_layout_entry(
    entries: Sequence[LayoutScoringEntry],
) -> Optional[LayoutScoringEntry]:
    """Вернуть вариант с наибольшим итоговым баллом."""
    return max(entries, key=lambda e: e.total_score) if entries else None


def layout_score_stats(
    entries: Sequence[LayoutScoringEntry],
) -> Dict:
    """Вычислить статистику итоговых баллов: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    scores = [e.total_score for e in entries]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(scores),
    }


def compare_layout_scoring_summaries(
    a: LayoutScoringSummary,
    b: LayoutScoringSummary,
) -> Dict:
    """Сравнить две сводки оценки компоновки."""
    return {
        "delta_mean_total_score": b.mean_total_score - a.mean_total_score,
        "delta_n_layouts": b.n_layouts - a.n_layouts,
        "same_best": a.best_layout_id == b.best_layout_id,
    }


def batch_summarise_layout_scoring_entries(
    groups: Sequence[Sequence[LayoutScoringEntry]],
    cfg: Optional[LayoutScoringConfig] = None,
) -> List[LayoutScoringSummary]:
    """Сформировать сводки для нескольких групп записей."""
    return [summarise_layout_scoring_entries(g, cfg) for g in groups]
