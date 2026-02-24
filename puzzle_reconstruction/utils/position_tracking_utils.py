"""Утилиты для анализа результатов оценки позиций и трекинга сборки.

Модуль предоставляет структуры данных и функции для:
- Записи и анализа истории позиционных оценок по итерациям сборки
- Сравнения результатов разных прогонов позиционирования
- Агрегации метрик качества расстановки фрагментов
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Position Quality Records
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionQualityRecord:
    """Запись качества расстановки фрагментов для одного набора позиций.

    Атрибуты:
        run_id:          Идентификатор прогона.
        n_fragments:     Число фрагментов.
        n_placed:        Число успешно размещённых фрагментов.
        mean_confidence: Среднее доверие к позициям.
        canvas_coverage: Доля покрытия холста (0–1).
        method:          Метка метода позиционирования.
        params:          Дополнительные параметры.
    """
    run_id: int
    n_fragments: int
    n_placed: int
    mean_confidence: float
    canvas_coverage: float
    method: str
    params: Dict = field(default_factory=dict)

    @property
    def placement_rate(self) -> float:
        """Доля размещённых фрагментов."""
        return self.n_placed / self.n_fragments if self.n_fragments > 0 else 0.0


@dataclass
class PositionQualitySummary:
    """Сводка результатов нескольких прогонов позиционирования.

    Атрибуты:
        n_runs:               Число прогонов.
        total_fragments:      Суммарное число фрагментов.
        mean_placement_rate:  Среднее placement_rate.
        mean_confidence:      Среднее mean_confidence.
        mean_coverage:        Среднее canvas_coverage.
        best_run_id:          ID прогона с наивысшим coverage или None.
        worst_run_id:         ID прогона с наименьшим coverage или None.
    """
    n_runs: int
    total_fragments: int
    mean_placement_rate: float
    mean_confidence: float
    mean_coverage: float
    best_run_id: Optional[int]
    worst_run_id: Optional[int]


def make_position_quality_record(
    run_id: int,
    n_fragments: int,
    n_placed: int,
    mean_confidence: float,
    canvas_coverage: float,
    method: str,
    **params,
) -> PositionQualityRecord:
    """Создать запись качества позиций.

    Args:
        run_id:          Идентификатор прогона.
        n_fragments:     Число фрагментов.
        n_placed:        Число размещённых фрагментов.
        mean_confidence: Среднее доверие.
        canvas_coverage: Покрытие холста.
        method:          Метод позиционирования.
        **params:        Дополнительные параметры.

    Returns:
        :class:`PositionQualityRecord`.
    """
    return PositionQualityRecord(
        run_id=int(run_id),
        n_fragments=int(n_fragments),
        n_placed=int(n_placed),
        mean_confidence=float(mean_confidence),
        canvas_coverage=float(canvas_coverage),
        method=method,
        params=dict(params),
    )


def summarise_position_quality(
    records: Sequence[PositionQualityRecord],
) -> PositionQualitySummary:
    """Сформировать сводку качества позиций.

    Args:
        records: Список записей.

    Returns:
        :class:`PositionQualitySummary`.
    """
    if not records:
        return PositionQualitySummary(
            n_runs=0,
            total_fragments=0,
            mean_placement_rate=0.0,
            mean_confidence=0.0,
            mean_coverage=0.0,
            best_run_id=None,
            worst_run_id=None,
        )
    rates = [r.placement_rate for r in records]
    confs = [r.mean_confidence for r in records]
    covs = [r.canvas_coverage for r in records]
    best = max(records, key=lambda r: r.canvas_coverage)
    worst = min(records, key=lambda r: r.canvas_coverage)
    return PositionQualitySummary(
        n_runs=len(records),
        total_fragments=sum(r.n_fragments for r in records),
        mean_placement_rate=sum(rates) / len(rates),
        mean_confidence=sum(confs) / len(confs),
        mean_coverage=sum(covs) / len(covs),
        best_run_id=best.run_id,
        worst_run_id=worst.run_id,
    )


def filter_by_placement_rate(
    records: Sequence[PositionQualityRecord],
    min_rate: float,
) -> List[PositionQualityRecord]:
    """Отфильтровать записи по минимальному placement_rate."""
    return [r for r in records if r.placement_rate >= min_rate]


def filter_by_method(
    records: Sequence[PositionQualityRecord],
    method: str,
) -> List[PositionQualityRecord]:
    """Отфильтровать записи по методу позиционирования."""
    return [r for r in records if r.method == method]


def top_k_position_records(
    records: Sequence[PositionQualityRecord],
    k: int,
) -> List[PositionQualityRecord]:
    """Вернуть k записей с наивысшим canvas_coverage."""
    return sorted(records, key=lambda r: r.canvas_coverage, reverse=True)[:k]


def best_position_record(
    records: Sequence[PositionQualityRecord],
) -> Optional[PositionQualityRecord]:
    """Вернуть запись с наивысшим canvas_coverage."""
    return max(records, key=lambda r: r.canvas_coverage) if records else None


def position_quality_stats(
    records: Sequence[PositionQualityRecord],
) -> Dict:
    """Вычислить статистику canvas_coverage: min, max, mean, std."""
    if not records:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [r.canvas_coverage for r in records]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Assembly Tracking History
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AssemblyHistoryEntry:
    """Запись истории одного прогона сборки.

    Атрибуты:
        run_id:          Идентификатор прогона.
        n_iterations:    Число итераций.
        best_score:      Лучшая оценка сборки.
        converged:       Флаг сходимости.
        convergence_iter: Итерация сходимости или None.
        method:          Метка алгоритма сборки.
        params:          Дополнительные параметры.
    """
    run_id: int
    n_iterations: int
    best_score: float
    converged: bool
    convergence_iter: Optional[int]
    method: str
    params: Dict = field(default_factory=dict)


@dataclass
class AssemblyHistorySummary:
    """Сводка истории нескольких прогонов сборки.

    Атрибуты:
        n_runs:               Число прогонов.
        n_converged:          Число сошедшихся прогонов.
        convergence_rate:     Доля сошедшихся прогонов.
        mean_best_score:      Средняя лучшая оценка.
        mean_n_iterations:    Среднее число итераций.
        best_run_id:          ID прогона с наилучшей оценкой или None.
    """
    n_runs: int
    n_converged: int
    convergence_rate: float
    mean_best_score: float
    mean_n_iterations: float
    best_run_id: Optional[int]


def make_assembly_history_entry(
    run_id: int,
    n_iterations: int,
    best_score: float,
    converged: bool,
    convergence_iter: Optional[int],
    method: str,
    **params,
) -> AssemblyHistoryEntry:
    """Создать запись истории сборки.

    Args:
        run_id:           Идентификатор прогона.
        n_iterations:     Число итераций.
        best_score:       Лучшая оценка.
        converged:        Флаг сходимости.
        convergence_iter: Итерация сходимости.
        method:           Метка алгоритма.
        **params:         Дополнительные параметры.

    Returns:
        :class:`AssemblyHistoryEntry`.
    """
    return AssemblyHistoryEntry(
        run_id=int(run_id),
        n_iterations=int(n_iterations),
        best_score=float(best_score),
        converged=bool(converged),
        convergence_iter=convergence_iter,
        method=method,
        params=dict(params),
    )


def summarise_assembly_history(
    entries: Sequence[AssemblyHistoryEntry],
) -> AssemblyHistorySummary:
    """Сформировать сводку истории сборок.

    Args:
        entries: Список записей.

    Returns:
        :class:`AssemblyHistorySummary`.
    """
    if not entries:
        return AssemblyHistorySummary(
            n_runs=0,
            n_converged=0,
            convergence_rate=0.0,
            mean_best_score=0.0,
            mean_n_iterations=0.0,
            best_run_id=None,
        )
    n_conv = sum(1 for e in entries if e.converged)
    best = max(entries, key=lambda e: e.best_score)
    return AssemblyHistorySummary(
        n_runs=len(entries),
        n_converged=n_conv,
        convergence_rate=n_conv / len(entries),
        mean_best_score=sum(e.best_score for e in entries) / len(entries),
        mean_n_iterations=sum(e.n_iterations for e in entries) / len(entries),
        best_run_id=best.run_id,
    )


def filter_converged(
    entries: Sequence[AssemblyHistoryEntry],
) -> List[AssemblyHistoryEntry]:
    """Вернуть только сошедшиеся прогоны."""
    return [e for e in entries if e.converged]


def filter_by_min_best_score(
    entries: Sequence[AssemblyHistoryEntry],
    min_score: float,
) -> List[AssemblyHistoryEntry]:
    """Отфильтровать прогоны по минимальной лучшей оценке."""
    return [e for e in entries if e.best_score >= min_score]


def top_k_assembly_entries(
    entries: Sequence[AssemblyHistoryEntry],
    k: int,
) -> List[AssemblyHistoryEntry]:
    """Вернуть k прогонов с наивысшей best_score."""
    return sorted(entries, key=lambda e: e.best_score, reverse=True)[:k]


def best_assembly_entry(
    entries: Sequence[AssemblyHistoryEntry],
) -> Optional[AssemblyHistoryEntry]:
    """Вернуть прогон с наивысшей best_score."""
    return max(entries, key=lambda e: e.best_score) if entries else None


def assembly_score_stats(
    entries: Sequence[AssemblyHistoryEntry],
) -> Dict:
    """Вычислить статистику best_score: min, max, mean, std."""
    if not entries:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    vals = [e.best_score for e in entries]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": mean,
        "std": var ** 0.5,
        "count": len(vals),
    }


def compare_assembly_summaries(
    a: AssemblyHistorySummary,
    b: AssemblyHistorySummary,
) -> Dict:
    """Сравнить две сводки истории сборок."""
    return {
        "delta_mean_best_score": b.mean_best_score - a.mean_best_score,
        "delta_convergence_rate": b.convergence_rate - a.convergence_rate,
        "delta_mean_iterations": b.mean_n_iterations - a.mean_n_iterations,
        "same_best": a.best_run_id == b.best_run_id,
    }


def batch_summarise_assembly_history(
    groups: Sequence[Sequence[AssemblyHistoryEntry]],
) -> List[AssemblyHistorySummary]:
    """Сформировать сводки для нескольких групп прогонов сборки."""
    return [summarise_assembly_history(g) for g in groups]
