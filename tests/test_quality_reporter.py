"""Тесты для puzzle_reconstruction.verification.quality_reporter."""
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

def _metric(name="coverage", value=0.95, threshold=None, passed=True):
    return QualityMetric(name=name, value=value, threshold=threshold,
                         passed=passed)


def _issue(severity="error", source="test", message="bad"):
    return QualityIssue(severity=severity, source=source, message=message)


def _report(coverage=0.95, overlap=0.01, ocr_score=0.85, cfg=None):
    return build_report(coverage, overlap, ocr_score, cfg=cfg)


# ─── TestReporterConfig ───────────────────────────────────────────────────────

class TestReporterConfig:
    def test_defaults(self):
        cfg = ReporterConfig()
        assert cfg.min_coverage == pytest.approx(0.9)
        assert cfg.max_overlap == pytest.approx(0.05)
        assert cfg.min_ocr_score == pytest.approx(0.7)
        assert cfg.include_warnings is True

    def test_valid_custom(self):
        cfg = ReporterConfig(min_coverage=0.8, max_overlap=0.1,
                             min_ocr_score=0.6, include_warnings=False)
        assert cfg.min_coverage == pytest.approx(0.8)
        assert cfg.include_warnings is False

    def test_zero_ok(self):
        cfg = ReporterConfig(min_coverage=0.0, max_overlap=0.0,
                             min_ocr_score=0.0)
        assert cfg.min_coverage == 0.0

    def test_one_ok(self):
        cfg = ReporterConfig(min_coverage=1.0, max_overlap=1.0,
                             min_ocr_score=1.0)
        assert cfg.min_coverage == 1.0

    def test_min_coverage_neg_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_coverage=-0.1)

    def test_min_coverage_above_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_coverage=1.1)

    def test_max_overlap_neg_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(max_overlap=-0.01)

    def test_max_overlap_above_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(max_overlap=1.1)

    def test_min_ocr_score_neg_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_ocr_score=-0.5)

    def test_min_ocr_score_above_raises(self):
        with pytest.raises(ValueError):
            ReporterConfig(min_ocr_score=1.5)


# ─── TestQualityMetric ────────────────────────────────────────────────────────

class TestQualityMetric:
    def test_basic(self):
        m = _metric()
        assert m.name == "coverage"
        assert m.value == pytest.approx(0.95)
        assert m.passed is True

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            QualityMetric(name="", value=0.5)

    def test_has_threshold_true(self):
        m = QualityMetric(name="x", value=0.5, threshold=0.9)
        assert m.has_threshold is True

    def test_has_threshold_false(self):
        m = QualityMetric(name="x", value=0.5, threshold=None)
        assert m.has_threshold is False

    def test_margin_with_threshold(self):
        m = QualityMetric(name="x", value=0.7, threshold=0.9)
        assert m.margin == pytest.approx(0.2)

    def test_margin_none_without_threshold(self):
        m = QualityMetric(name="x", value=0.7, threshold=None)
        assert m.margin is None

    def test_margin_negative_when_failed(self):
        m = QualityMetric(name="x", value=1.0, threshold=0.8)
        assert m.margin == pytest.approx(-0.2)

    def test_note_default_empty(self):
        m = _metric()
        assert m.note == ""

    def test_note_set(self):
        m = QualityMetric(name="x", value=0.5, note="custom")
        assert m.note == "custom"

    def test_passed_false(self):
        m = QualityMetric(name="x", value=0.4, passed=False)
        assert m.passed is False


# ─── TestQualityIssue ─────────────────────────────────────────────────────────

class TestQualityIssue:
    def test_basic_error(self):
        i = _issue(severity="error")
        assert i.is_error is True

    def test_basic_warning(self):
        i = _issue(severity="warning")
        assert i.is_error is False

    def test_empty_message_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="error", source="x", message="")

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            QualityIssue(severity="info", source="x", message="msg")

    def test_severity_stored(self):
        i = _issue(severity="warning")
        assert i.severity == "warning"

    def test_source_stored(self):
        i = _issue(source="ocr")
        assert i.source == "ocr"

    def test_message_stored(self):
        i = _issue(message="low coverage")
        assert i.message == "low coverage"


# ─── TestQualityReport ────────────────────────────────────────────────────────

class TestQualityReport:
    def _make(self):
        metrics = [
            _metric("coverage", 0.95, threshold=0.9, passed=True),
            _metric("overlap", 0.06, threshold=0.05, passed=False),
            _metric("ocr", 0.8, threshold=0.7, passed=True),
        ]
        issues = [
            _issue("error", "overlap", "overlap too high"),
            _issue("warning", "ocr", "ocr borderline"),
        ]
        return QualityReport(metrics=metrics, issues=issues,
                             passed=False, summary="FAIL")

    def test_n_errors(self):
        r = self._make()
        assert r.n_errors == 1

    def test_n_warnings(self):
        r = self._make()
        assert r.n_warnings == 1

    def test_metric_names(self):
        r = self._make()
        assert "coverage" in r.metric_names
        assert "overlap" in r.metric_names

    def test_failed_metrics(self):
        r = self._make()
        failed = r.failed_metrics
        assert len(failed) == 1
        assert failed[0].name == "overlap"

    def test_get_metric_found(self):
        r = self._make()
        m = r.get_metric("coverage")
        assert m is not None
        assert m.value == pytest.approx(0.95)

    def test_get_metric_not_found(self):
        r = self._make()
        assert r.get_metric("unknown") is None

    def test_passed_stored(self):
        r = self._make()
        assert r.passed is False

    def test_summary_stored(self):
        r = self._make()
        assert r.summary == "FAIL"

    def test_no_issues_all_pass(self):
        metrics = [_metric("x", 0.9, passed=True)]
        r = QualityReport(metrics=metrics, issues=[], passed=True, summary="OK")
        assert r.n_errors == 0
        assert r.n_warnings == 0


# ─── TestBuildMetric ──────────────────────────────────────────────────────────

class TestBuildMetric:
    def test_no_threshold_passed(self):
        m = build_metric("x", 0.5)
        assert m.passed is True
        assert m.threshold is None

    def test_higher_better_above_threshold(self):
        m = build_metric("cov", 0.95, threshold=0.9, higher_is_better=True)
        assert m.passed is True

    def test_higher_better_below_threshold(self):
        m = build_metric("cov", 0.8, threshold=0.9, higher_is_better=True)
        assert m.passed is False

    def test_lower_better_below_threshold(self):
        m = build_metric("ovl", 0.02, threshold=0.05, higher_is_better=False)
        assert m.passed is True

    def test_lower_better_above_threshold(self):
        m = build_metric("ovl", 0.08, threshold=0.05, higher_is_better=False)
        assert m.passed is False

    def test_equal_to_threshold_passes_higher_better(self):
        m = build_metric("x", 0.9, threshold=0.9, higher_is_better=True)
        assert m.passed is True

    def test_equal_to_threshold_passes_lower_better(self):
        m = build_metric("x", 0.05, threshold=0.05, higher_is_better=False)
        assert m.passed is True

    def test_note_passed_through(self):
        m = build_metric("x", 0.5, note="custom note")
        assert m.note == "custom note"

    def test_returns_quality_metric(self):
        m = build_metric("x", 0.5)
        assert isinstance(m, QualityMetric)


# ─── TestAddIssue ─────────────────────────────────────────────────────────────

class TestAddIssue:
    def test_appends(self):
        issues = []
        add_issue(issues, "error", "src", "msg")
        assert len(issues) == 1

    def test_issue_type(self):
        issues = []
        add_issue(issues, "warning", "s", "m")
        assert isinstance(issues[0], QualityIssue)

    def test_severity_stored(self):
        issues = []
        add_issue(issues, "warning", "s", "m")
        assert issues[0].severity == "warning"

    def test_multiple_append(self):
        issues = []
        add_issue(issues, "error", "a", "m1")
        add_issue(issues, "warning", "b", "m2")
        assert len(issues) == 2

    def test_source_stored(self):
        issues = []
        add_issue(issues, "error", "coverage", "msg")
        assert issues[0].source == "coverage"


# ─── TestBuildReport ──────────────────────────────────────────────────────────

class TestBuildReport:
    def test_returns_quality_report(self):
        r = _report()
        assert isinstance(r, QualityReport)

    def test_all_pass_high_values(self):
        cfg = ReporterConfig(min_coverage=0.5, max_overlap=0.5,
                             min_ocr_score=0.5)
        r = _report(0.95, 0.01, 0.9, cfg)
        assert r.passed is True

    def test_coverage_fail_generates_error(self):
        cfg = ReporterConfig(min_coverage=0.99)
        r = _report(coverage=0.80, cfg=cfg)
        assert r.passed is False
        errors = [i for i in r.issues if i.severity == "error"
                  and i.source == "coverage"]
        assert len(errors) >= 1

    def test_overlap_fail_generates_error(self):
        cfg = ReporterConfig(max_overlap=0.01)
        r = _report(overlap=0.10, cfg=cfg)
        assert r.passed is False
        errors = [i for i in r.issues if i.severity == "error"
                  and i.source == "overlap"]
        assert len(errors) >= 1

    def test_low_ocr_warning_with_warnings_enabled(self):
        cfg = ReporterConfig(min_ocr_score=0.95, include_warnings=True)
        r = _report(ocr_score=0.5, cfg=cfg)
        warnings = [i for i in r.issues if i.severity == "warning"
                    and i.source == "ocr"]
        assert len(warnings) >= 1

    def test_low_ocr_no_warning_when_disabled(self):
        cfg = ReporterConfig(min_ocr_score=0.95, include_warnings=False)
        r = _report(ocr_score=0.5, cfg=cfg)
        warnings = [i for i in r.issues if i.severity == "warning"
                    and i.source == "ocr"]
        assert len(warnings) == 0

    def test_extra_metrics_included(self):
        r = build_report(0.95, 0.01, 0.85,
                         extra_metrics={"sharpness": 0.7, "noise": 0.1})
        names = [m.name for m in r.metrics]
        assert "sharpness" in names
        assert "noise" in names

    def test_default_config_used_when_none(self):
        r = build_report(0.95, 0.01, 0.85)
        assert isinstance(r, QualityReport)

    def test_summary_contains_ok_on_pass(self):
        cfg = ReporterConfig(min_coverage=0.5, max_overlap=0.5,
                             min_ocr_score=0.5)
        r = build_report(0.95, 0.01, 0.85, cfg=cfg)
        assert "OK" in r.summary or "ok" in r.summary.lower()

    def test_summary_contains_fail_on_fail(self):
        cfg = ReporterConfig(min_coverage=1.0)
        r = build_report(0.5, 0.01, 0.85, cfg=cfg)
        assert "FAIL" in r.summary or "fail" in r.summary.lower()

    def test_three_core_metrics_always_present(self):
        r = _report()
        names = [m.name for m in r.metrics]
        assert "coverage" in names
        assert "overlap" in names
        assert "ocr_score" in names


# ─── TestMergeReports ─────────────────────────────────────────────────────────

class TestMergeReports:
    def test_empty_list_returns_report(self):
        r = merge_reports([])
        assert isinstance(r, QualityReport)
        assert r.passed is True

    def test_empty_list_no_metrics(self):
        r = merge_reports([])
        assert r.metrics == []

    def test_single_report_identity(self):
        r1 = _report()
        merged = merge_reports([r1])
        assert len(merged.metrics) == len(r1.metrics)
        assert len(merged.issues) == len(r1.issues)

    def test_metrics_summed(self):
        r1 = _report(0.95, 0.01, 0.85)
        r2 = _report(0.92, 0.02, 0.80)
        merged = merge_reports([r1, r2])
        assert len(merged.metrics) == len(r1.metrics) + len(r2.metrics)

    def test_issues_summed(self):
        cfg = ReporterConfig(min_coverage=0.99)
        r1 = build_report(0.5, 0.01, 0.85, cfg=cfg)
        r2 = build_report(0.6, 0.01, 0.85, cfg=cfg)
        merged = merge_reports([r1, r2])
        assert len(merged.issues) == len(r1.issues) + len(r2.issues)

    def test_passed_false_if_any_error(self):
        cfg = ReporterConfig(min_coverage=0.99)
        r_fail = build_report(0.5, 0.01, 0.85, cfg=cfg)
        r_ok = _report()
        merged = merge_reports([r_fail, r_ok])
        assert merged.passed is False

    def test_passed_true_if_no_errors(self):
        cfg = ReporterConfig(min_coverage=0.5, max_overlap=0.5,
                             min_ocr_score=0.5)
        r1 = build_report(0.95, 0.01, 0.85, cfg=cfg)
        r2 = build_report(0.90, 0.02, 0.80, cfg=cfg)
        merged = merge_reports([r1, r2])
        assert merged.passed is True


# ─── TestFilterIssues ─────────────────────────────────────────────────────────

class TestFilterIssues:
    def _make_report(self):
        issues = [
            _issue("error", "cov", "bad coverage"),
            _issue("warning", "ocr", "borderline"),
            _issue("error", "ovl", "overlap high"),
        ]
        return QualityReport(metrics=[], issues=issues, passed=False)

    def test_filter_errors(self):
        r = self._make_report()
        errs = filter_issues(r, "error")
        assert len(errs) == 2
        for i in errs:
            assert i.severity == "error"

    def test_filter_warnings(self):
        r = self._make_report()
        warns = filter_issues(r, "warning")
        assert len(warns) == 1
        assert warns[0].severity == "warning"

    def test_invalid_severity_raises(self):
        r = self._make_report()
        with pytest.raises(ValueError):
            filter_issues(r, "info")

    def test_empty_issues_returns_empty(self):
        r = QualityReport(metrics=[], issues=[], passed=True)
        assert filter_issues(r, "error") == []

    def test_no_errors_returns_empty(self):
        issues = [_issue("warning", "s", "m")]
        r = QualityReport(metrics=[], issues=issues, passed=True)
        assert filter_issues(r, "error") == []


# ─── TestExportReport ─────────────────────────────────────────────────────────

class TestExportReport:
    def test_returns_list(self):
        r = _report()
        exported = export_report(r)
        assert isinstance(exported, list)

    def test_length_matches_metrics(self):
        r = _report()
        exported = export_report(r)
        assert len(exported) == len(r.metrics)

    def test_each_item_has_required_keys(self):
        r = _report()
        for item in export_report(r):
            assert "name" in item
            assert "value" in item
            assert "threshold" in item
            assert "passed" in item

    def test_values_match_metrics(self):
        r = _report(0.95, 0.01, 0.85)
        exported = export_report(r)
        exported_by_name = {d["name"]: d for d in exported}
        assert "coverage" in exported_by_name
        assert exported_by_name["coverage"]["value"] == pytest.approx(0.95)

    def test_empty_report(self):
        r = QualityReport(metrics=[], issues=[], passed=True)
        assert export_report(r) == []

    def test_passed_flag_exported(self):
        r = _report()
        for item in export_report(r):
            assert isinstance(item["passed"], bool)

    def test_threshold_exported(self):
        r = _report()
        exported = export_report(r)
        # coverage, overlap, ocr_score have thresholds
        names_with_threshold = [d["name"] for d in exported
                                 if d["threshold"] is not None]
        assert "coverage" in names_with_threshold
