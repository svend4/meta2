"""Формирование отчётов по оценкам верификации сборки.

Модуль предоставляет структуры и функции для сбора оценок из
нескольких верификаторов, формирования агрегированного отчёта
и экспорта его в различные форматы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ─── ReportConfig ─────────────────────────────────────────────────────────────

@dataclass
class ReportConfig:
    """Параметры формирования отчёта.

    Атрибуты:
        weights:       Веса отдельных метрик {metric_name: weight} (>= 0).
        pass_threshold: Минимальная итоговая оценка для статуса 'pass' [0, 1].
        include_meta:  Включать ли метаданные в экспорт.
    """

    weights: Dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.5
    include_meta: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.pass_threshold <= 1.0):
            raise ValueError(
                f"pass_threshold должен быть в [0, 1], "
                f"получено {self.pass_threshold}"
            )
        for k, v in self.weights.items():
            if v < 0.0:
                raise ValueError(
                    f"Вес '{k}' должен быть >= 0, получено {v}"
                )


# ─── ScoreEntry ───────────────────────────────────────────────────────────────

@dataclass
class ScoreEntry:
    """Одна запись оценки метрики.

    Атрибуты:
        metric:   Название метрики.
        value:    Значение оценки [0, 1].
        weight:   Вес метрики при агрегации (>= 0).
        meta:     Дополнительные данные.
    """

    metric: str
    value: float
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric не должен быть пустым")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"value должен быть в [0, 1], получено {self.value}"
            )
        if self.weight < 0.0:
            raise ValueError(
                f"weight должен быть >= 0, получено {self.weight}"
            )

    @property
    def weighted_value(self) -> float:
        """Произведение value × weight."""
        return self.value * self.weight


# ─── ScoringReport ────────────────────────────────────────────────────────────

@dataclass
class ScoringReport:
    """Итоговый отчёт по оценкам верификации.

    Атрибуты:
        entries:         Список ScoreEntry.
        total_score:     Взвешенная итоговая оценка [0, 1].
        n_metrics:       Число метрик (>= 0).
        passed:          True если total_score >= pass_threshold.
        pass_threshold:  Применённый порог прохождения [0, 1].
    """

    entries: List[ScoreEntry]
    total_score: float
    n_metrics: int
    passed: bool
    pass_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.total_score <= 1.0):
            raise ValueError(
                f"total_score должен быть в [0, 1], получено {self.total_score}"
            )
        if self.n_metrics < 0:
            raise ValueError(
                f"n_metrics должен быть >= 0, получено {self.n_metrics}"
            )
        if not (0.0 <= self.pass_threshold <= 1.0):
            raise ValueError(
                f"pass_threshold должен быть в [0, 1], "
                f"получено {self.pass_threshold}"
            )

    @property
    def status(self) -> str:
        """'pass' или 'fail'."""
        return "pass" if self.passed else "fail"

    @property
    def metric_names(self) -> List[str]:
        """Список имён метрик."""
        return [e.metric for e in self.entries]

    @property
    def worst_metric(self) -> Optional[str]:
        """Метрика с наименьшим value (None если нет записей)."""
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e.value).metric


# ─── add_score ────────────────────────────────────────────────────────────────

def add_score(
    entries: List[ScoreEntry],
    metric: str,
    value: float,
    weight: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
) -> ScoreEntry:
    """Добавить запись оценки в список.

    Аргументы:
        entries: Список, в который добавляется запись.
        metric:  Название метрики.
        value:   Значение оценки [0, 1].
        weight:  Вес метрики.
        meta:    Метаданные.

    Возвращает:
        Созданный ScoreEntry.
    """
    entry = ScoreEntry(metric=metric, value=value, weight=weight,
                       meta=meta or {})
    entries.append(entry)
    return entry


# ─── compute_summary ──────────────────────────────────────────────────────────

def compute_summary(
    entries: List[ScoreEntry],
    cfg: Optional[ReportConfig] = None,
) -> ScoringReport:
    """Агрегировать записи в итоговый отчёт.

    Аргументы:
        entries: Список ScoreEntry.
        cfg:     Параметры формирования отчёта (None → ReportConfig()).

    Возвращает:
        ScoringReport.
    """
    if cfg is None:
        cfg = ReportConfig()

    if not entries:
        return ScoringReport(
            entries=[],
            total_score=0.0,
            n_metrics=0,
            passed=False,
            pass_threshold=cfg.pass_threshold,
        )

    # Применить веса из cfg.weights (переопределяют entry.weight если указаны)
    effective_entries: List[ScoreEntry] = []
    for e in entries:
        w = cfg.weights.get(e.metric, e.weight)
        effective_entries.append(ScoreEntry(
            metric=e.metric, value=e.value, weight=w, meta=dict(e.meta)
        ))

    w_sum = sum(e.weight for e in effective_entries) + 1e-12
    total = sum(e.weighted_value for e in effective_entries) / w_sum
    total = float(np.clip(total, 0.0, 1.0))
    passed = total >= cfg.pass_threshold

    return ScoringReport(
        entries=effective_entries,
        total_score=total,
        n_metrics=len(effective_entries),
        passed=passed,
        pass_threshold=cfg.pass_threshold,
    )


# ─── format_report ────────────────────────────────────────────────────────────

def format_report(report: ScoringReport) -> str:
    """Форматировать отчёт в виде читаемой строки.

    Аргументы:
        report: ScoringReport.

    Возвращает:
        Многострочная строка с результатами.
    """
    lines = [
        f"Status: {report.status.upper()}",
        f"Total score: {report.total_score:.4f} "
        f"(threshold: {report.pass_threshold:.4f})",
        f"Metrics ({report.n_metrics}):",
    ]
    for e in report.entries:
        lines.append(f"  {e.metric}: {e.value:.4f} (w={e.weight:.2f})")
    if report.worst_metric:
        lines.append(f"Worst metric: {report.worst_metric}")
    return "\n".join(lines)


# ─── filter_report ────────────────────────────────────────────────────────────

def filter_report(
    report: ScoringReport,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> ScoringReport:
    """Вернуть новый ScoringReport только с метриками в диапазоне [min_value, max_value].

    Аргументы:
        report:    Исходный ScoringReport.
        min_value: Минимальное значение оценки [0, 1].
        max_value: Максимальное значение оценки [0, 1].

    Возвращает:
        Новый ScoringReport с отфильтрованными записями.

    Исключения:
        ValueError: Если min_value > max_value.
    """
    if min_value > max_value:
        raise ValueError(
            f"min_value ({min_value}) > max_value ({max_value})"
        )
    filtered = [e for e in report.entries
                if min_value <= e.value <= max_value]
    return ScoringReport(
        entries=filtered,
        total_score=report.total_score,
        n_metrics=len(filtered),
        passed=report.passed,
        pass_threshold=report.pass_threshold,
    )


# ─── compare_reports ──────────────────────────────────────────────────────────

def compare_reports(
    a: ScoringReport,
    b: ScoringReport,
) -> Dict[str, float]:
    """Сравнить два отчёта по общим метрикам.

    Аргументы:
        a: Первый ScoringReport.
        b: Второй ScoringReport.

    Возвращает:
        Словарь {metric: b.value - a.value} для общих метрик.
    """
    a_map = {e.metric: e.value for e in a.entries}
    b_map = {e.metric: e.value for e in b.entries}
    common = set(a_map.keys()) & set(b_map.keys())
    return {m: b_map[m] - a_map[m] for m in sorted(common)}


# ─── export_report ────────────────────────────────────────────────────────────

def export_report(
    report: ScoringReport,
    cfg: Optional[ReportConfig] = None,
) -> Dict[str, Any]:
    """Экспортировать отчёт в словарь.

    Аргументы:
        report: ScoringReport.
        cfg:    Параметры (если include_meta=False, мета не экспортируется).

    Возвращает:
        Словарь с полями 'total_score', 'status', 'n_metrics', 'entries'.
    """
    if cfg is None:
        cfg = ReportConfig()

    entries_data = []
    for e in report.entries:
        ed: Dict[str, Any] = {
            "metric": e.metric,
            "value": e.value,
            "weight": e.weight,
        }
        if cfg.include_meta:
            ed["meta"] = dict(e.meta)
        entries_data.append(ed)

    return {
        "total_score": report.total_score,
        "status": report.status,
        "n_metrics": report.n_metrics,
        "pass_threshold": report.pass_threshold,
        "entries": entries_data,
    }


# ─── batch_score_report ───────────────────────────────────────────────────────

def batch_score_report(
    entry_lists: List[List[ScoreEntry]],
    cfg: Optional[ReportConfig] = None,
) -> List[ScoringReport]:
    """Сформировать отчёты для нескольких наборов оценок.

    Аргументы:
        entry_lists: Список списков ScoreEntry.
        cfg:         Общие параметры формирования отчёта.

    Возвращает:
        Список ScoringReport.
    """
    return [compute_summary(entries, cfg) for entries in entry_lists]
