"""Формирование отчётов о качестве восстановленного документа.

Модуль собирает метрики из нескольких источников (OCR, покрытие,
перекрытия, оценки краёв) и генерирует структурированный отчёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─── ReporterConfig ───────────────────────────────────────────────────────────

@dataclass
class ReporterConfig:
    """Параметры формирования отчёта.

    Атрибуты:
        min_coverage:     Минимально приемлемое покрытие [0, 1].
        max_overlap:      Максимально допустимое перекрытие [0, 1].
        min_ocr_score:    Минимально приемлемый OCR-балл [0, 1].
        include_warnings: Включать предупреждения (не только ошибки).
    """

    min_coverage: float = 0.9
    max_overlap: float = 0.05
    min_ocr_score: float = 0.7
    include_warnings: bool = True

    def __post_init__(self) -> None:
        for name, val in (
            ("min_coverage", self.min_coverage),
            ("max_overlap", self.max_overlap),
            ("min_ocr_score", self.min_ocr_score),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} должен быть в [0, 1], получено {val}"
                )


# ─── QualityMetric ────────────────────────────────────────────────────────────

@dataclass
class QualityMetric:
    """Одна метрика качества.

    Атрибуты:
        name:     Название (непустая строка).
        value:    Измеренное значение.
        threshold: Порог (None если порог не применяется).
        passed:   True если метрика в норме.
        note:     Дополнительный комментарий.
    """

    name: str
    value: float
    threshold: Optional[float] = None
    passed: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name не должен быть пустой строкой")

    @property
    def has_threshold(self) -> bool:
        """True если задан порог."""
        return self.threshold is not None

    @property
    def margin(self) -> Optional[float]:
        """Запас до порога или None."""
        if self.threshold is None:
            return None
        return float(self.threshold - self.value)


# ─── QualityIssue ─────────────────────────────────────────────────────────────

@dataclass
class QualityIssue:
    """Обнаруженная проблема качества.

    Атрибуты:
        severity: ``"error"`` или ``"warning"``.
        source:   Источник (имя метрики / модуля).
        message:  Описание проблемы (непустая строка).
    """

    severity: str
    source: str
    message: str

    _SEVERITIES = {"error", "warning"}

    def __post_init__(self) -> None:
        if self.severity not in self._SEVERITIES:
            raise ValueError(
                f"severity должен быть одним из {self._SEVERITIES}, "
                f"получено '{self.severity}'"
            )
        if not self.message:
            raise ValueError("message не должен быть пустой строкой")

    @property
    def is_error(self) -> bool:
        """True если severity == 'error'."""
        return self.severity == "error"


# ─── QualityReport ────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Итоговый отчёт о качестве.

    Атрибуты:
        metrics:  Список QualityMetric.
        issues:   Список QualityIssue.
        passed:   True если ни одной ошибки.
        summary:  Краткое описание.
    """

    metrics: List[QualityMetric]
    issues: List[QualityIssue]
    passed: bool
    summary: str = ""

    @property
    def n_errors(self) -> int:
        """Число ошибок."""
        return sum(1 for i in self.issues if i.is_error)

    @property
    def n_warnings(self) -> int:
        """Число предупреждений."""
        return sum(1 for i in self.issues if not i.is_error)

    @property
    def metric_names(self) -> List[str]:
        """Имена всех метрик."""
        return [m.name for m in self.metrics]

    @property
    def failed_metrics(self) -> List[QualityMetric]:
        """Метрики, не прошедшие порог."""
        return [m for m in self.metrics if not m.passed]

    def get_metric(self, name: str) -> Optional[QualityMetric]:
        """Вернуть метрику по имени или None."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None


# ─── build_metric ─────────────────────────────────────────────────────────────

def build_metric(
    name: str,
    value: float,
    threshold: Optional[float] = None,
    higher_is_better: bool = True,
    note: str = "",
) -> QualityMetric:
    """Создать QualityMetric с автоматическим определением passed.

    Аргументы:
        name:             Название метрики.
        value:            Значение.
        threshold:        Порог (None = без порога, passed=True).
        higher_is_better: True → passed если value >= threshold.
        note:             Комментарий.

    Возвращает:
        QualityMetric.
    """
    if threshold is None:
        passed = True
    elif higher_is_better:
        passed = value >= threshold
    else:
        passed = value <= threshold

    return QualityMetric(name=name, value=value, threshold=threshold,
                         passed=passed, note=note)


# ─── add_issue ────────────────────────────────────────────────────────────────

def add_issue(
    issues: List[QualityIssue],
    severity: str,
    source: str,
    message: str,
) -> None:
    """Добавить QualityIssue в список.

    Аргументы:
        issues:   Список для добавления.
        severity: ``"error"`` или ``"warning"``.
        source:   Источник.
        message:  Описание.
    """
    issues.append(QualityIssue(severity=severity, source=source,
                                message=message))


# ─── build_report ─────────────────────────────────────────────────────────────

def build_report(
    coverage: float,
    overlap: float,
    ocr_score: float,
    extra_metrics: Optional[Dict[str, float]] = None,
    cfg: Optional[ReporterConfig] = None,
) -> QualityReport:
    """Сформировать отчёт о качестве по основным метрикам.

    Аргументы:
        coverage:      Покрытие документа [0, 1].
        overlap:       Доля перекрытий [0, 1].
        ocr_score:     OCR-балл [0, 1].
        extra_metrics: Дополнительные метрики {название: значение}.
        cfg:           Параметры.

    Возвращает:
        QualityReport.
    """
    if cfg is None:
        cfg = ReporterConfig()

    metrics: List[QualityMetric] = []
    issues: List[QualityIssue] = []

    # Покрытие
    cov_m = build_metric("coverage", coverage,
                          threshold=cfg.min_coverage, higher_is_better=True)
    metrics.append(cov_m)
    if not cov_m.passed:
        add_issue(issues, "error", "coverage",
                  f"Покрытие {coverage:.3f} ниже порога {cfg.min_coverage:.3f}")

    # Перекрытия
    ovl_m = build_metric("overlap", overlap,
                          threshold=cfg.max_overlap, higher_is_better=False)
    metrics.append(ovl_m)
    if not ovl_m.passed:
        add_issue(issues, "error", "overlap",
                  f"Перекрытие {overlap:.3f} превышает порог {cfg.max_overlap:.3f}")

    # OCR
    ocr_m = build_metric("ocr_score", ocr_score,
                          threshold=cfg.min_ocr_score, higher_is_better=True)
    metrics.append(ocr_m)
    if not ocr_m.passed and cfg.include_warnings:
        add_issue(issues, "warning", "ocr",
                  f"OCR-балл {ocr_score:.3f} ниже порога {cfg.min_ocr_score:.3f}")

    # Дополнительные метрики
    for name, val in (extra_metrics or {}).items():
        m = build_metric(name, val)
        metrics.append(m)

    passed = all(i.severity != "error" for i in issues)
    n_fail = len([m for m in metrics if not m.passed])
    summary = (
        f"OK — все {len(metrics)} метрик в норме" if passed
        else f"FAIL — {n_fail} из {len(metrics)} метрик не прошли порог"
    )

    return QualityReport(
        metrics=metrics,
        issues=issues,
        passed=passed,
        summary=summary,
    )


# ─── merge_reports ────────────────────────────────────────────────────────────

def merge_reports(reports: List[QualityReport]) -> QualityReport:
    """Объединить несколько отчётов в один.

    Аргументы:
        reports: Список QualityReport.

    Возвращает:
        Объединённый QualityReport (метрики и проблемы складываются).
    """
    if not reports:
        return QualityReport(metrics=[], issues=[], passed=True,
                             summary="Нет отчётов")

    metrics: List[QualityMetric] = []
    issues: List[QualityIssue] = []

    for r in reports:
        metrics.extend(r.metrics)
        issues.extend(r.issues)

    passed = all(i.severity != "error" for i in issues)
    summary = (
        f"Merged {len(reports)} отчётов — "
        + ("OK" if passed else "FAIL")
    )
    return QualityReport(metrics=metrics, issues=issues,
                         passed=passed, summary=summary)


# ─── filter_issues ────────────────────────────────────────────────────────────

def filter_issues(
    report: QualityReport,
    severity: str,
) -> List[QualityIssue]:
    """Отфильтровать проблемы по уровню severity.

    Аргументы:
        report:   QualityReport.
        severity: ``"error"`` или ``"warning"``.

    Возвращает:
        Список QualityIssue с указанным severity.
    """
    valid = {"error", "warning"}
    if severity not in valid:
        raise ValueError(
            f"severity должен быть одним из {valid}, получено '{severity}'"
        )
    return [i for i in report.issues if i.severity == severity]


# ─── export_report ────────────────────────────────────────────────────────────

def export_report(report: QualityReport) -> List[Dict]:
    """Экспортировать отчёт в список словарей.

    Аргументы:
        report: QualityReport.

    Возвращает:
        Список {"name", "value", "threshold", "passed"} для каждой метрики.
    """
    return [
        {
            "name": m.name,
            "value": m.value,
            "threshold": m.threshold,
            "passed": m.passed,
        }
        for m in report.metrics
    ]
