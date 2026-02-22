"""Тесты для puzzle_reconstruction.verification.score_reporter."""
import pytest
from puzzle_reconstruction.verification.score_reporter import (
    ReportConfig,
    ScoreEntry,
    ScoringReport,
    add_score,
    compute_summary,
    format_report,
    filter_report,
    compare_reports,
    export_report,
    batch_score_report,
)


class TestReportConfig:
    def test_defaults(self):
        cfg = ReportConfig()
        assert cfg.pass_threshold == 0.5
        assert cfg.include_meta is True
        assert cfg.weights == {}

    def test_valid_custom(self):
        cfg = ReportConfig(weights={"a": 2.0, "b": 1.0}, pass_threshold=0.7)
        assert cfg.weights["a"] == 2.0
        assert cfg.pass_threshold == 0.7

    def test_invalid_pass_threshold_below(self):
        with pytest.raises(ValueError):
            ReportConfig(pass_threshold=-0.1)

    def test_invalid_pass_threshold_above(self):
        with pytest.raises(ValueError):
            ReportConfig(pass_threshold=1.1)

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            ReportConfig(weights={"x": -1.0})


class TestScoreEntry:
    def test_basic(self):
        e = ScoreEntry(metric="boundary", value=0.8)
        assert e.metric == "boundary"
        assert e.value == 0.8
        assert e.weight == 1.0
        assert e.meta == {}

    def test_weighted_value(self):
        e = ScoreEntry(metric="x", value=0.5, weight=2.0)
        assert abs(e.weighted_value - 1.0) < 1e-9

    def test_zero_weight(self):
        e = ScoreEntry(metric="x", value=0.9, weight=0.0)
        assert e.weighted_value == 0.0

    def test_invalid_metric_empty(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="", value=0.5)

    def test_invalid_value_below(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="x", value=-0.1)

    def test_invalid_value_above(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="x", value=1.1)

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="x", value=0.5, weight=-1.0)


class TestScoringReport:
    def _make(self, total_score=0.8, passed=True):
        e = ScoreEntry(metric="a", value=0.8)
        return ScoringReport(
            entries=[e],
            total_score=total_score,
            n_metrics=1,
            passed=passed,
            pass_threshold=0.5,
        )

    def test_status_pass(self):
        r = self._make(passed=True)
        assert r.status == "pass"

    def test_status_fail(self):
        r = self._make(total_score=0.3, passed=False)
        assert r.status == "fail"

    def test_metric_names(self):
        r = self._make()
        assert r.metric_names == ["a"]

    def test_worst_metric(self):
        e1 = ScoreEntry(metric="a", value=0.9)
        e2 = ScoreEntry(metric="b", value=0.3)
        r = ScoringReport(entries=[e1, e2], total_score=0.6, n_metrics=2,
                          passed=True, pass_threshold=0.5)
        assert r.worst_metric == "b"

    def test_worst_metric_empty(self):
        r = ScoringReport(entries=[], total_score=0.0, n_metrics=0,
                          passed=False, pass_threshold=0.5)
        assert r.worst_metric is None

    def test_invalid_total_score(self):
        with pytest.raises(ValueError):
            ScoringReport(entries=[], total_score=1.5, n_metrics=0,
                          passed=False, pass_threshold=0.5)

    def test_invalid_n_metrics(self):
        with pytest.raises(ValueError):
            ScoringReport(entries=[], total_score=0.5, n_metrics=-1,
                          passed=False, pass_threshold=0.5)


class TestAddScore:
    def test_basic(self):
        entries = []
        entry = add_score(entries, "color", 0.7)
        assert len(entries) == 1
        assert entry.metric == "color"
        assert abs(entry.value - 0.7) < 1e-9

    def test_with_weight_and_meta(self):
        entries = []
        add_score(entries, "texture", 0.5, weight=2.0, meta={"k": 1})
        assert entries[0].weight == 2.0
        assert entries[0].meta["k"] == 1

    def test_appends(self):
        entries = []
        add_score(entries, "a", 0.5)
        add_score(entries, "b", 0.6)
        assert len(entries) == 2


class TestComputeSummary:
    def test_equal_weights(self):
        entries = [
            ScoreEntry("a", 0.8),
            ScoreEntry("b", 0.6),
        ]
        report = compute_summary(entries)
        assert abs(report.total_score - 0.7) < 1e-5
        assert report.n_metrics == 2

    def test_pass_threshold_pass(self):
        entries = [ScoreEntry("a", 0.9)]
        report = compute_summary(entries, ReportConfig(pass_threshold=0.8))
        assert report.passed is True

    def test_pass_threshold_fail(self):
        entries = [ScoreEntry("a", 0.4)]
        report = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert report.passed is False

    def test_custom_weights_override(self):
        entries = [
            ScoreEntry("a", 1.0),
            ScoreEntry("b", 0.0),
        ]
        cfg = ReportConfig(weights={"a": 3.0, "b": 1.0})
        report = compute_summary(entries, cfg)
        # weighted: (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert abs(report.total_score - 0.75) < 1e-5

    def test_empty_entries(self):
        report = compute_summary([])
        assert report.total_score == 0.0
        assert report.n_metrics == 0
        assert report.passed is False

    def test_total_score_clipped(self):
        # Should be exactly clipped to [0,1]
        entries = [ScoreEntry("a", 1.0)]
        report = compute_summary(entries)
        assert 0.0 <= report.total_score <= 1.0


class TestFormatReport:
    def test_contains_status(self):
        entries = [ScoreEntry("a", 0.8)]
        report = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        text = format_report(report)
        assert "PASS" in text or "FAIL" in text

    def test_contains_metrics(self):
        entries = [ScoreEntry("boundary", 0.7), ScoreEntry("color", 0.9)]
        report = compute_summary(entries)
        text = format_report(report)
        assert "boundary" in text
        assert "color" in text

    def test_contains_worst(self):
        entries = [ScoreEntry("a", 0.9), ScoreEntry("b", 0.2)]
        report = compute_summary(entries)
        text = format_report(report)
        assert "b" in text


class TestFilterReport:
    def test_filter_by_max(self):
        entries = [
            ScoreEntry("a", 0.9),
            ScoreEntry("b", 0.4),
            ScoreEntry("c", 0.6),
        ]
        report = compute_summary(entries)
        filtered = filter_report(report, max_value=0.7)
        assert all(e.value <= 0.7 for e in filtered.entries)

    def test_filter_by_min(self):
        entries = [ScoreEntry("a", 0.9), ScoreEntry("b", 0.3)]
        report = compute_summary(entries)
        filtered = filter_report(report, min_value=0.5)
        assert all(e.value >= 0.5 for e in filtered.entries)

    def test_invalid_range(self):
        entries = [ScoreEntry("a", 0.5)]
        report = compute_summary(entries)
        with pytest.raises(ValueError):
            filter_report(report, min_value=0.8, max_value=0.3)

    def test_preserves_total_score(self):
        entries = [ScoreEntry("a", 0.8)]
        report = compute_summary(entries)
        filtered = filter_report(report)
        assert filtered.total_score == report.total_score


class TestCompareReports:
    def test_positive_delta(self):
        entries_a = [ScoreEntry("a", 0.6)]
        entries_b = [ScoreEntry("a", 0.9)]
        ra = compute_summary(entries_a)
        rb = compute_summary(entries_b)
        diff = compare_reports(ra, rb)
        assert abs(diff["a"] - 0.3) < 1e-6

    def test_only_common_metrics(self):
        entries_a = [ScoreEntry("a", 0.5), ScoreEntry("b", 0.7)]
        entries_b = [ScoreEntry("b", 0.9), ScoreEntry("c", 0.8)]
        ra = compute_summary(entries_a)
        rb = compute_summary(entries_b)
        diff = compare_reports(ra, rb)
        assert set(diff.keys()) == {"b"}

    def test_empty_common(self):
        ra = compute_summary([ScoreEntry("a", 0.5)])
        rb = compute_summary([ScoreEntry("b", 0.5)])
        assert compare_reports(ra, rb) == {}


class TestExportReport:
    def test_keys_present(self):
        entries = [ScoreEntry("a", 0.8)]
        report = compute_summary(entries)
        exported = export_report(report)
        assert "total_score" in exported
        assert "status" in exported
        assert "n_metrics" in exported
        assert "entries" in exported
        assert "pass_threshold" in exported

    def test_include_meta_true(self):
        entries = [ScoreEntry("a", 0.8, meta={"k": 1})]
        report = compute_summary(entries)
        exported = export_report(report, ReportConfig(include_meta=True))
        assert "meta" in exported["entries"][0]

    def test_include_meta_false(self):
        entries = [ScoreEntry("a", 0.8, meta={"k": 1})]
        report = compute_summary(entries)
        exported = export_report(report, ReportConfig(include_meta=False))
        assert "meta" not in exported["entries"][0]

    def test_empty_entries(self):
        report = compute_summary([])
        exported = export_report(report)
        assert exported["entries"] == []


class TestBatchScoreReport:
    def test_basic(self):
        entry_lists = [
            [ScoreEntry("a", 0.9)],
            [ScoreEntry("a", 0.3)],
        ]
        reports = batch_score_report(entry_lists)
        assert len(reports) == 2

    def test_pass_threshold_applied(self):
        cfg = ReportConfig(pass_threshold=0.7)
        entry_lists = [
            [ScoreEntry("a", 0.8)],
            [ScoreEntry("a", 0.5)],
        ]
        reports = batch_score_report(entry_lists, cfg)
        assert reports[0].passed is True
        assert reports[1].passed is False

    def test_empty(self):
        assert batch_score_report([]) == []
