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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _metric(name="m", value=0.8, passed=True) -> QualityMetric:
    return QualityMetric(name=name, value=value, passed=passed)


def _issue(severity="error", source="src", message="msg") -> QualityIssue:
    return QualityIssue(severity=severity, source=source, message=message)


def _report(metrics=None, issues=None, passed=True) -> QualityReport:
    return QualityReport(
        metrics=metrics or [],
        issues=issues or [],
        passed=passed,
        summary="",
    )


# ─── ReporterConfig ───────────────────────────────────────────────────────────

class TestReporterConfigExtra:
    def test_default_min_coverage(self):
        assert ReporterConfig().min_coverage == pytest.approx(0.9)

    def test_default_max_overlap(self):
        assert ReporterConfig().max_overlap == pytest.approx(0.05)

    def test_default_min_ocr_score(self):
        assert ReporterConfig().min_ocr_score == pytest.approx(0.7)

    def test_default_include_warnings(self):
        assert ReporterConfig().include_warnings is True

    def test_min_coverage_negative_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_coverage=-0.1)

    def test_min_coverage_gt_one_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_coverage=1.1)

    def test_max_overlap_negative_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(max_overlap=-0.01)

    def test_min_ocr_score_gt_one_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_ocr_score=1.5)

    def test_custom_values(self):
        cfg = ReporterConfig(min_coverage=0.8, max_overlap=0.1, min_ocr_score=0.5)
        assert cfg.min_coverage == pytest.approx(0.8)
        assert cfg.max_overlap == pytest.approx(0.1)


# ─── QualityMetric ────────────────────────────────────────────────────────────

class TestQualityMetricExtra:
    def test_name_stored(self):
        m = _metric(name="coverage")
        assert m.name == "coverage"

    def test_value_stored(self):
        m = _metric(value=0.75)
        assert m.value == pytest.approx(0.75)

    def test_passed_stored(self):
        m = _metric(passed=False)
        assert m.passed is False

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            QualityMetric(name="", value=0.5)

    def test_threshold_default_none(self):
        m = _metric()
        assert m.threshold is None

    def test_has_threshold_false(self):
        assert _metric().has_threshold is False

    def test_has_threshold_true(self):
        m = QualityMetric(name="m", value=0.5, threshold=0.7)
        assert m.has_threshold is True

    def test_margin_none_without_threshold(self):
        assert _metric().margin is None

    def test_margin_computed(self):
        m = QualityMetric(name="m", value=0.6, threshold=0.9)
        assert m.margin == pytest.approx(0.3)

    def test_note_default_empty(self):
        assert _metric().note == ""

    def test_note_stored(self):
        m = QualityMetric(name="m", value=0.5, note="info")
        assert m.note == "info"


# ─── QualityIssue ─────────────────────────────────────────────────────────────

class TestQualityIssueExtra:
    def test_severity_stored(self):
        i = _issue(severity="warning")
        assert i.severity == "warning"

    def test_source_stored(self):
        i = _issue(source="ocr")
        assert i.source == "ocr"

    def test_message_stored(self):
        i = _issue(message="bad coverage")
        assert i.message == "bad coverage"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="critical", source="s", message="m")

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="error", source="s", message="")

    def test_is_error_true(self):
        assert _issue(severity="error").is_error is True

    def test_is_error_false_for_warning(self):
        assert _issue(severity="warning").is_error is False


# ─── QualityReport ────────────────────────────────────────────────────────────

class TestQualityReportExtra:
    def test_n_errors_zero(self):
        r = _report(issues=[_issue(severity="warning")])
        assert r.n_errors == 0

    def test_n_errors_counted(self):
        r = _report(issues=[_issue("error"), _issue("error"), _issue("warning")])
        assert r.n_errors == 2

    def test_n_warnings_counted(self):
        r = _report(issues=[_issue("warning"), _issue("error")])
        assert r.n_warnings == 1

    def test_metric_names(self):
        r = _report(metrics=[_metric("cov"), _metric("ocr")])
        assert set(r.metric_names) == {"cov", "ocr"}

    def test_failed_metrics_empty(self):
        r = _report(metrics=[_metric(passed=True)])
        assert r.failed_metrics == []

    def test_failed_metrics_returned(self):
        m1 = _metric("a", passed=True)
        m2 = _metric("b", passed=False)
        r = _report(metrics=[m1, m2])
        assert len(r.failed_metrics) == 1
        assert r.failed_metrics[0].name == "b"

    def test_get_metric_found(self):
        r = _report(metrics=[_metric("cov")])
        m = r.get_metric("cov")
        assert m is not None
        assert m.name == "cov"

    def test_get_metric_not_found(self):
        r = _report()
        assert r.get_metric("missing") is None

    def test_passed_stored(self):
        assert _report(passed=False).passed is False


# ─── build_metric ─────────────────────────────────────────────────────────────

class TestBuildMetricExtra:
    def test_returns_quality_metric(self):
        m = build_metric("m", 0.8)
        assert isinstance(m, QualityMetric)

    def test_no_threshold_passed_true(self):
        m = build_metric("m", 0.5, threshold=None)
        assert m.passed is True

    def test_higher_is_better_pass(self):
        m = build_metric("m", 0.9, threshold=0.8, higher_is_better=True)
        assert m.passed is True

    def test_higher_is_better_fail(self):
        m = build_metric("m", 0.5, threshold=0.8, higher_is_better=True)
        assert m.passed is False

    def test_lower_is_better_pass(self):
        m = build_metric("m", 0.02, threshold=0.05, higher_is_better=False)
        assert m.passed is True

    def test_lower_is_better_fail(self):
        m = build_metric("m", 0.1, threshold=0.05, higher_is_better=False)
        assert m.passed is False

    def test_note_stored(self):
        m = build_metric("m", 0.5, note="test note")
        assert m.note == "test note"

    def test_threshold_stored(self):
        m = build_metric("m", 0.5, threshold=0.7)
        assert m.threshold == pytest.approx(0.7)


# ─── add_issue ────────────────────────────────────────────────────────────────

class TestAddIssueExtra:
    def test_appends_to_list(self):
        issues = []
        add_issue(issues, "error", "src", "message")
        assert len(issues) == 1

    def test_created_issue_type(self):
        issues = []
        add_issue(issues, "warning", "src", "message")
        assert isinstance(issues[0], QualityIssue)

    def test_severity_correct(self):
        issues = []
        add_issue(issues, "error", "s", "m")
        assert issues[0].severity == "error"

    def test_multiple_add(self):
        issues = []
        add_issue(issues, "error", "s1", "m1")
        add_issue(issues, "warning", "s2", "m2")
        assert len(issues) == 2


# ─── build_report ─────────────────────────────────────────────────────────────

class TestBuildReportExtra:
    def _good(self):
        return build_report(coverage=1.0, overlap=0.0, ocr_score=1.0)

    def _bad(self):
        return build_report(coverage=0.1, overlap=0.9, ocr_score=0.1)

    def test_returns_quality_report(self):
        assert isinstance(self._good(), QualityReport)

    def test_good_report_passed(self):
        assert self._good().passed is True

    def test_bad_coverage_fails(self):
        r = build_report(coverage=0.1, overlap=0.0, ocr_score=1.0)
        assert r.passed is False

    def test_bad_overlap_fails(self):
        r = build_report(coverage=1.0, overlap=0.5, ocr_score=1.0)
        assert r.passed is False

    def test_coverage_metric_in_report(self):
        r = self._good()
        assert r.get_metric("coverage") is not None

    def test_overlap_metric_in_report(self):
        r = self._good()
        assert r.get_metric("overlap") is not None

    def test_ocr_metric_in_report(self):
        r = self._good()
        assert r.get_metric("ocr_score") is not None

    def test_extra_metrics_included(self):
        r = build_report(1.0, 0.0, 1.0, extra_metrics={"sharpness": 0.9})
        assert r.get_metric("sharpness") is not None

    def test_none_cfg_uses_defaults(self):
        r = build_report(1.0, 0.0, 1.0, cfg=None)
        assert isinstance(r, QualityReport)

    def test_ocr_warning_generated(self):
        cfg = ReporterConfig(min_ocr_score=0.9, include_warnings=True)
        r = build_report(1.0, 0.0, 0.5, cfg=cfg)
        warnings = [i for i in r.issues if i.severity == "warning"]
        assert len(warnings) > 0

    def test_summary_not_empty(self):
        r = self._good()
        assert len(r.summary) > 0


# ─── merge_reports ────────────────────────────────────────────────────────────

class TestMergeReportsExtra:
    def _ok(self) -> QualityReport:
        return build_report(1.0, 0.0, 1.0)

    def test_returns_quality_report(self):
        assert isinstance(merge_reports([self._ok()]), QualityReport)

    def test_empty_input(self):
        r = merge_reports([])
        assert isinstance(r, QualityReport)
        assert r.passed is True

    def test_single_report_merged(self):
        r = self._ok()
        merged = merge_reports([r])
        assert len(merged.metrics) == len(r.metrics)

    def test_two_reports_metrics_combined(self):
        r1 = self._ok()
        r2 = self._ok()
        merged = merge_reports([r1, r2])
        assert len(merged.metrics) == len(r1.metrics) + len(r2.metrics)

    def test_passed_false_if_any_error(self):
        ok = self._ok()
        bad = build_report(0.0, 1.0, 0.0)
        merged = merge_reports([ok, bad])
        assert merged.passed is False

    def test_summary_contains_count(self):
        merged = merge_reports([self._ok(), self._ok()])
        assert "2" in merged.summary


# ─── filter_issues ────────────────────────────────────────────────────────────

class TestFilterIssuesExtra:
    def _report_with_issues(self) -> QualityReport:
        return _report(issues=[
            _issue("error"), _issue("warning"), _issue("error"),
        ])

    def test_filter_errors(self):
        r = self._report_with_issues()
        errors = filter_issues(r, "error")
        assert len(errors) == 2
        assert all(i.severity == "error" for i in errors)

    def test_filter_warnings(self):
        r = self._report_with_issues()
        warnings = filter_issues(r, "warning")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            filter_issues(_report(), "critical")

    def test_empty_report_empty_result(self):
        assert filter_issues(_report(), "error") == []


# ─── export_report ────────────────────────────────────────────────────────────

class TestExportReportExtra:
    def test_returns_list(self):
        r = build_report(1.0, 0.0, 1.0)
        assert isinstance(export_report(r), list)

    def test_length_matches_metrics(self):
        r = build_report(1.0, 0.0, 1.0)
        exported = export_report(r)
        assert len(exported) == len(r.metrics)

    def test_each_element_has_keys(self):
        r = build_report(1.0, 0.0, 1.0)
        for item in export_report(r):
            for key in ("name", "value", "threshold", "passed"):
                assert key in item

    def test_name_preserved(self):
        r = build_report(1.0, 0.0, 1.0)
        names = {item["name"] for item in export_report(r)}
        assert "coverage" in names

    def test_empty_report(self):
        r = _report()
        assert export_report(r) == []
