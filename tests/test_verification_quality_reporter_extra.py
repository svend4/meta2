"""Extra tests for puzzle_reconstruction/verification/quality_reporter.py."""
from __future__ import annotations

import pytest

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


# ─── ReporterConfig ──────────────────────────────────────────────────────────

class TestReporterConfigExtra:
    def test_defaults(self):
        c = ReporterConfig()
        assert c.min_coverage == pytest.approx(0.9)
        assert c.max_overlap == pytest.approx(0.05)
        assert c.min_ocr_score == pytest.approx(0.7)
        assert c.include_warnings is True

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_coverage=1.5)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(max_overlap=-0.1)

    def test_invalid_ocr_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_ocr_score=2.0)


# ─── QualityMetric ───────────────────────────────────────────────────────────

class TestQualityMetricExtra:
    def test_basic(self):
        m = QualityMetric(name="test", value=0.5)
        assert m.name == "test"
        assert m.value == 0.5
        assert m.passed is True

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            QualityMetric(name="", value=0.0)

    def test_has_threshold(self):
        m1 = QualityMetric(name="a", value=0.5, threshold=0.7)
        assert m1.has_threshold is True
        m2 = QualityMetric(name="b", value=0.5)
        assert m2.has_threshold is False

    def test_margin(self):
        m = QualityMetric(name="a", value=0.5, threshold=0.7)
        assert m.margin == pytest.approx(0.2)

    def test_margin_none(self):
        m = QualityMetric(name="a", value=0.5)
        assert m.margin is None


# ─── QualityIssue ────────────────────────────────────────────────────────────

class TestQualityIssueExtra:
    def test_error(self):
        i = QualityIssue(severity="error", source="cov", message="low")
        assert i.is_error is True

    def test_warning(self):
        i = QualityIssue(severity="warning", source="ocr", message="low")
        assert i.is_error is False

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="fatal", source="a", message="b")

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="error", source="a", message="")


# ─── QualityReport ───────────────────────────────────────────────────────────

class TestQualityReportExtra:
    def test_passed(self):
        r = QualityReport(metrics=[], issues=[], passed=True)
        assert r.n_errors == 0
        assert r.n_warnings == 0
        assert r.metric_names == []
        assert r.failed_metrics == []

    def test_with_issues(self):
        issues = [
            QualityIssue(severity="error", source="a", message="bad"),
            QualityIssue(severity="warning", source="b", message="meh"),
        ]
        r = QualityReport(metrics=[], issues=issues, passed=False)
        assert r.n_errors == 1
        assert r.n_warnings == 1

    def test_get_metric(self):
        m = QualityMetric(name="cov", value=0.9)
        r = QualityReport(metrics=[m], issues=[], passed=True)
        assert r.get_metric("cov") is m
        assert r.get_metric("unknown") is None

    def test_failed_metrics(self):
        m1 = QualityMetric(name="a", value=0.5, passed=True)
        m2 = QualityMetric(name="b", value=0.3, passed=False)
        r = QualityReport(metrics=[m1, m2], issues=[], passed=False)
        assert len(r.failed_metrics) == 1
        assert r.failed_metrics[0].name == "b"


# ─── build_metric ────────────────────────────────────────────────────────────

class TestBuildMetricExtra:
    def test_no_threshold(self):
        m = build_metric("test", 0.5)
        assert m.passed is True
        assert m.threshold is None

    def test_higher_is_better_pass(self):
        m = build_metric("cov", 0.95, threshold=0.9)
        assert m.passed is True

    def test_higher_is_better_fail(self):
        m = build_metric("cov", 0.8, threshold=0.9)
        assert m.passed is False

    def test_lower_is_better_pass(self):
        m = build_metric("ovl", 0.01, threshold=0.05,
                          higher_is_better=False)
        assert m.passed is True

    def test_lower_is_better_fail(self):
        m = build_metric("ovl", 0.1, threshold=0.05,
                          higher_is_better=False)
        assert m.passed is False

    def test_with_note(self):
        m = build_metric("x", 0.5, note="info")
        assert m.note == "info"


# ─── add_issue ───────────────────────────────────────────────────────────────

class TestAddIssueExtra:
    def test_adds(self):
        issues = []
        add_issue(issues, "error", "src", "msg")
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].source == "src"

    def test_multiple(self):
        issues = []
        add_issue(issues, "error", "a", "e1")
        add_issue(issues, "warning", "b", "w1")
        assert len(issues) == 2


# ─── build_report ────────────────────────────────────────────────────────────

class TestBuildReportExtra:
    def test_all_pass(self):
        r = build_report(coverage=0.95, overlap=0.01, ocr_score=0.8)
        assert r.passed is True
        assert "OK" in r.summary
        assert len(r.metrics) == 3

    def test_low_coverage_fails(self):
        r = build_report(coverage=0.5, overlap=0.01, ocr_score=0.8)
        assert r.passed is False

    def test_high_overlap_fails(self):
        r = build_report(coverage=0.95, overlap=0.2, ocr_score=0.8)
        assert r.passed is False

    def test_low_ocr_warning(self):
        r = build_report(coverage=0.95, overlap=0.01, ocr_score=0.3)
        # OCR failure is a warning, not error → still passes
        assert r.passed is True
        assert any(i.severity == "warning" for i in r.issues)

    def test_no_warnings(self):
        cfg = ReporterConfig(include_warnings=False)
        r = build_report(coverage=0.95, overlap=0.01, ocr_score=0.3, cfg=cfg)
        assert all(i.severity == "error" for i in r.issues)

    def test_extra_metrics(self):
        r = build_report(0.95, 0.01, 0.8, extra_metrics={"custom": 0.5})
        assert len(r.metrics) == 4
        assert r.get_metric("custom").value == pytest.approx(0.5)


# ─── merge_reports ───────────────────────────────────────────────────────────

class TestMergeReportsExtra:
    def test_empty(self):
        r = merge_reports([])
        assert r.passed is True
        assert r.metrics == []

    def test_two_passing(self):
        r1 = build_report(0.95, 0.01, 0.8)
        r2 = build_report(0.92, 0.02, 0.9)
        merged = merge_reports([r1, r2])
        assert merged.passed is True
        assert len(merged.metrics) == 6

    def test_one_failing(self):
        r1 = build_report(0.95, 0.01, 0.8)
        r2 = build_report(0.3, 0.01, 0.8)  # low coverage
        merged = merge_reports([r1, r2])
        assert merged.passed is False


# ─── filter_issues ───────────────────────────────────────────────────────────

class TestFilterIssuesExtra:
    def test_errors(self):
        r = build_report(0.3, 0.2, 0.3)  # coverage + overlap fail
        errors = filter_issues(r, "error")
        assert all(i.severity == "error" for i in errors)

    def test_warnings(self):
        r = build_report(0.95, 0.01, 0.3)
        warnings = filter_issues(r, "warning")
        assert all(i.severity == "warning" for i in warnings)

    def test_invalid_severity_raises(self):
        r = build_report(0.95, 0.01, 0.8)
        with pytest.raises(ValueError):
            filter_issues(r, "fatal")


# ─── export_report ───────────────────────────────────────────────────────────

class TestExportReportExtra:
    def test_basic(self):
        r = build_report(0.95, 0.01, 0.8)
        exported = export_report(r)
        assert len(exported) == 3
        assert all("name" in d for d in exported)
        assert all("value" in d for d in exported)
        assert all("threshold" in d for d in exported)
        assert all("passed" in d for d in exported)

    def test_empty(self):
        r = QualityReport(metrics=[], issues=[], passed=True)
        assert export_report(r) == []
