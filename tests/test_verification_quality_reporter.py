"""Tests for puzzle_reconstruction.verification.quality_reporter"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.quality_reporter import (
    ReporterConfig,
    QualityMetric,
    QualityIssue,
    QualityReport,
    build_metric,
    add_issue,
    build_report,
    merge_reports,
    filter_issues,
    export_report,
)


# ─── ReporterConfig ───────────────────────────────────────────────────────────

def test_reporter_config_defaults():
    cfg = ReporterConfig()
    assert cfg.min_coverage == 0.9
    assert cfg.max_overlap == 0.05
    assert cfg.min_ocr_score == 0.7
    assert cfg.include_warnings is True


def test_reporter_config_invalid_coverage():
    with pytest.raises(ValueError):
        ReporterConfig(min_coverage=1.5)


def test_reporter_config_invalid_overlap():
    with pytest.raises(ValueError):
        ReporterConfig(max_overlap=-0.1)


def test_reporter_config_invalid_ocr():
    with pytest.raises(ValueError):
        ReporterConfig(min_ocr_score=2.0)


# ─── QualityMetric ────────────────────────────────────────────────────────────

def test_quality_metric_has_threshold_true():
    m = QualityMetric(name="coverage", value=0.9, threshold=0.8)
    assert m.has_threshold is True


def test_quality_metric_has_threshold_false():
    m = QualityMetric(name="coverage", value=0.9)
    assert m.has_threshold is False


def test_quality_metric_margin_positive():
    m = QualityMetric(name="coverage", value=0.7, threshold=0.8)
    assert m.margin == pytest.approx(0.1)


def test_quality_metric_margin_none():
    m = QualityMetric(name="coverage", value=0.7)
    assert m.margin is None


def test_quality_metric_empty_name():
    with pytest.raises(ValueError):
        QualityMetric(name="", value=0.5)


# ─── QualityIssue ─────────────────────────────────────────────────────────────

def test_quality_issue_is_error_true():
    issue = QualityIssue(severity="error", source="coverage", message="Low coverage")
    assert issue.is_error is True


def test_quality_issue_is_error_false():
    issue = QualityIssue(severity="warning", source="ocr", message="Low OCR")
    assert issue.is_error is False


def test_quality_issue_invalid_severity():
    with pytest.raises(ValueError):
        QualityIssue(severity="critical", source="test", message="Error")


def test_quality_issue_empty_message():
    with pytest.raises(ValueError):
        QualityIssue(severity="error", source="test", message="")


# ─── QualityReport ────────────────────────────────────────────────────────────

def test_quality_report_n_errors():
    issues = [
        QualityIssue(severity="error", source="a", message="E1"),
        QualityIssue(severity="warning", source="b", message="W1"),
        QualityIssue(severity="error", source="c", message="E2"),
    ]
    report = QualityReport(metrics=[], issues=issues, passed=False)
    assert report.n_errors == 2
    assert report.n_warnings == 1


def test_quality_report_metric_names():
    metrics = [
        QualityMetric(name="coverage", value=0.9),
        QualityMetric(name="overlap", value=0.05),
    ]
    report = QualityReport(metrics=metrics, issues=[], passed=True)
    assert "coverage" in report.metric_names
    assert "overlap" in report.metric_names


def test_quality_report_failed_metrics():
    metrics = [
        QualityMetric(name="coverage", value=0.9, passed=True),
        QualityMetric(name="overlap", value=0.9, passed=False),
    ]
    report = QualityReport(metrics=metrics, issues=[], passed=False)
    failed = report.failed_metrics
    assert len(failed) == 1
    assert failed[0].name == "overlap"


def test_quality_report_get_metric():
    metrics = [QualityMetric(name="ocr_score", value=0.8)]
    report = QualityReport(metrics=metrics, issues=[], passed=True)
    m = report.get_metric("ocr_score")
    assert m is not None
    assert m.value == 0.8


def test_quality_report_get_metric_missing():
    report = QualityReport(metrics=[], issues=[], passed=True)
    assert report.get_metric("nonexistent") is None


# ─── build_metric ─────────────────────────────────────────────────────────────

def test_build_metric_higher_is_better_pass():
    m = build_metric("coverage", 0.95, threshold=0.9, higher_is_better=True)
    assert m.passed is True


def test_build_metric_higher_is_better_fail():
    m = build_metric("coverage", 0.7, threshold=0.9, higher_is_better=True)
    assert m.passed is False


def test_build_metric_lower_is_better_pass():
    m = build_metric("overlap", 0.02, threshold=0.05, higher_is_better=False)
    assert m.passed is True


def test_build_metric_lower_is_better_fail():
    m = build_metric("overlap", 0.08, threshold=0.05, higher_is_better=False)
    assert m.passed is False


def test_build_metric_no_threshold():
    m = build_metric("extra", 0.5)
    assert m.passed is True


# ─── add_issue ────────────────────────────────────────────────────────────────

def test_add_issue_appends():
    issues = []
    add_issue(issues, "error", "coverage", "Low coverage")
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_add_issue_multiple():
    issues = []
    add_issue(issues, "error", "a", "E1")
    add_issue(issues, "warning", "b", "W1")
    assert len(issues) == 2


# ─── build_report ─────────────────────────────────────────────────────────────

def test_build_report_all_pass():
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    assert report.passed
    assert len(report.metrics) >= 3


def test_build_report_low_coverage():
    report = build_report(coverage=0.5, overlap=0.02, ocr_score=0.8,
                          cfg=ReporterConfig(min_coverage=0.9))
    assert not report.passed
    errors = [i for i in report.issues if i.is_error]
    assert len(errors) >= 1


def test_build_report_high_overlap():
    report = build_report(coverage=0.95, overlap=0.5, ocr_score=0.8,
                          cfg=ReporterConfig(max_overlap=0.05))
    assert not report.passed


def test_build_report_low_ocr_warning():
    cfg = ReporterConfig(min_ocr_score=0.9, include_warnings=True)
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.5, cfg=cfg)
    warnings = [i for i in report.issues if not i.is_error]
    assert len(warnings) >= 1


def test_build_report_extra_metrics():
    extra = {"custom_metric": 0.7}
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8,
                          extra_metrics=extra)
    assert "custom_metric" in report.metric_names


def test_build_report_summary_contains_ok_or_fail():
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    assert "OK" in report.summary or "FAIL" in report.summary


# ─── merge_reports ────────────────────────────────────────────────────────────

def test_merge_reports_empty():
    merged = merge_reports([])
    assert merged.passed
    assert "Нет отчётов" in merged.summary


def test_merge_reports_combines_metrics():
    r1 = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    r2 = build_report(coverage=0.9, overlap=0.01, ocr_score=0.85)
    merged = merge_reports([r1, r2])
    assert len(merged.metrics) == len(r1.metrics) + len(r2.metrics)


def test_merge_reports_passed_if_no_errors():
    r1 = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    r2 = build_report(coverage=0.9, overlap=0.01, ocr_score=0.85)
    merged = merge_reports([r1, r2])
    assert merged.passed


# ─── filter_issues ────────────────────────────────────────────────────────────

def test_filter_issues_errors():
    report = build_report(coverage=0.1, overlap=0.9, ocr_score=0.1)
    errors = filter_issues(report, "error")
    assert all(i.is_error for i in errors)


def test_filter_issues_warnings():
    cfg = ReporterConfig(min_ocr_score=0.9, include_warnings=True)
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.5, cfg=cfg)
    warnings = filter_issues(report, "warning")
    assert all(not i.is_error for i in warnings)


def test_filter_issues_invalid_severity():
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    with pytest.raises(ValueError):
        filter_issues(report, "critical")


# ─── export_report ────────────────────────────────────────────────────────────

def test_export_report_structure():
    report = build_report(coverage=0.95, overlap=0.02, ocr_score=0.8)
    exported = export_report(report)
    assert isinstance(exported, list)
    assert len(exported) >= 3
    for item in exported:
        assert "name" in item
        assert "value" in item
        assert "passed" in item


def test_export_report_empty():
    report = QualityReport(metrics=[], issues=[], passed=True)
    exported = export_report(report)
    assert exported == []
