"""Tests for puzzle_reconstruction/verification/score_reporter.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _entry(metric="m1", value=0.7, weight=1.0):
    return ScoreEntry(metric=metric, value=value, weight=weight)


def _report(entries=None, total_score=0.7, n_metrics=1, passed=True, pass_threshold=0.5):
    if entries is None:
        entries = [_entry()]
    return ScoringReport(
        entries=entries,
        total_score=total_score,
        n_metrics=n_metrics,
        passed=passed,
        pass_threshold=pass_threshold,
    )


# ─── TestReportConfig ─────────────────────────────────────────────────────────

class TestReportConfig:
    def test_defaults(self):
        cfg = ReportConfig()
        assert cfg.pass_threshold == pytest.approx(0.5)
        assert cfg.include_meta is True
        assert cfg.weights == {}

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ReportConfig(weights={"m1": -0.1})

    def test_zero_weight_ok(self):
        cfg = ReportConfig(weights={"m1": 0.0})
        assert cfg.weights["m1"] == pytest.approx(0.0)

    def test_pass_threshold_above_1_raises(self):
        with pytest.raises(ValueError):
            ReportConfig(pass_threshold=1.1)

    def test_pass_threshold_below_0_raises(self):
        with pytest.raises(ValueError):
            ReportConfig(pass_threshold=-0.1)

    def test_pass_threshold_at_zero(self):
        cfg = ReportConfig(pass_threshold=0.0)
        assert cfg.pass_threshold == 0.0

    def test_pass_threshold_at_one(self):
        cfg = ReportConfig(pass_threshold=1.0)
        assert cfg.pass_threshold == 1.0

    def test_include_meta_false(self):
        cfg = ReportConfig(include_meta=False)
        assert cfg.include_meta is False


# ─── TestScoreEntry ───────────────────────────────────────────────────────────

class TestScoreEntry:
    def test_valid_creation(self):
        e = ScoreEntry(metric="accuracy", value=0.8, weight=1.0)
        assert e.metric == "accuracy"

    def test_empty_metric_raises(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="", value=0.5, weight=1.0)

    def test_value_above_1_raises(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="m", value=1.1, weight=1.0)

    def test_value_below_0_raises(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="m", value=-0.1, weight=1.0)

    def test_value_at_zero_boundary(self):
        e = ScoreEntry(metric="m", value=0.0)
        assert e.value == 0.0

    def test_value_at_one_boundary(self):
        e = ScoreEntry(metric="m", value=1.0)
        assert e.value == 1.0

    def test_weight_negative_raises(self):
        with pytest.raises(ValueError):
            ScoreEntry(metric="m", value=0.5, weight=-1.0)

    def test_weight_zero_ok(self):
        e = ScoreEntry(metric="m", value=0.5, weight=0.0)
        assert e.weight == 0.0

    def test_weighted_value(self):
        e = ScoreEntry(metric="m", value=0.4, weight=2.0)
        assert e.weighted_value == pytest.approx(0.8)

    def test_default_weight_one(self):
        e = ScoreEntry(metric="m", value=0.5)
        assert e.weight == pytest.approx(1.0)

    def test_default_meta_dict(self):
        e = ScoreEntry(metric="m", value=0.5)
        assert isinstance(e.meta, dict)

    def test_weighted_value_zero_weight(self):
        e = ScoreEntry(metric="m", value=0.9, weight=0.0)
        assert e.weighted_value == pytest.approx(0.0)


# ─── TestScoringReport ────────────────────────────────────────────────────────

class TestScoringReport:
    def test_valid_creation(self):
        r = _report()
        assert r.total_score == pytest.approx(0.7)

    def test_total_score_above_1_raises(self):
        with pytest.raises(ValueError):
            _report(total_score=1.1)

    def test_total_score_below_0_raises(self):
        with pytest.raises(ValueError):
            _report(total_score=-0.1)

    def test_n_metrics_negative_raises(self):
        with pytest.raises(ValueError):
            _report(n_metrics=-1)

    def test_pass_threshold_above_1_raises(self):
        with pytest.raises(ValueError):
            _report(pass_threshold=1.5)

    def test_status_pass(self):
        r = _report(passed=True)
        assert r.status == "pass"

    def test_status_fail(self):
        r = _report(passed=False)
        assert r.status == "fail"

    def test_metric_names(self):
        entries = [_entry("acc", 0.8), _entry("iou", 0.6)]
        r = _report(entries=entries, total_score=0.7, n_metrics=2)
        assert r.metric_names == ["acc", "iou"]

    def test_metric_names_empty(self):
        r = ScoringReport(entries=[], total_score=0.0, n_metrics=0,
                          passed=False, pass_threshold=0.5)
        assert r.metric_names == []

    def test_worst_metric(self):
        entries = [_entry("acc", 0.8), _entry("iou", 0.3)]
        r = _report(entries=entries, total_score=0.55, n_metrics=2)
        assert r.worst_metric == "iou"

    def test_worst_metric_none_if_empty(self):
        r = ScoringReport(entries=[], total_score=0.0, n_metrics=0,
                          passed=False, pass_threshold=0.5)
        assert r.worst_metric is None

    def test_worst_metric_single_entry(self):
        r = _report(entries=[_entry("solo", 0.5)])
        assert r.worst_metric == "solo"


# ─── TestAddScore ─────────────────────────────────────────────────────────────

class TestAddScore:
    def test_returns_score_entry(self):
        entries = []
        result = add_score(entries, "m1", 0.7)
        assert isinstance(result, ScoreEntry)

    def test_appends_to_list(self):
        entries = []
        add_score(entries, "m1", 0.7)
        assert len(entries) == 1

    def test_stores_metric(self):
        entries = []
        e = add_score(entries, "precision", 0.8)
        assert e.metric == "precision"

    def test_stores_value(self):
        entries = []
        e = add_score(entries, "m1", 0.6)
        assert e.value == pytest.approx(0.6)

    def test_default_weight_one(self):
        entries = []
        e = add_score(entries, "m1", 0.5)
        assert e.weight == pytest.approx(1.0)

    def test_custom_weight(self):
        entries = []
        e = add_score(entries, "m1", 0.5, weight=2.0)
        assert e.weight == pytest.approx(2.0)

    def test_meta_stored(self):
        entries = []
        e = add_score(entries, "m1", 0.5, meta={"k": "v"})
        assert e.meta["k"] == "v"

    def test_multiple_adds(self):
        entries = []
        add_score(entries, "a", 0.5)
        add_score(entries, "b", 0.6)
        assert len(entries) == 2

    def test_none_meta_becomes_empty_dict(self):
        entries = []
        e = add_score(entries, "m", 0.5, meta=None)
        assert isinstance(e.meta, dict)


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def test_empty_entries_zero_score(self):
        r = compute_summary([])
        assert r.total_score == pytest.approx(0.0)

    def test_empty_entries_not_passed(self):
        r = compute_summary([])
        assert r.passed is False

    def test_empty_entries_n_metrics_zero(self):
        r = compute_summary([])
        assert r.n_metrics == 0

    def test_single_entry_score(self):
        entries = [_entry("m1", 0.8)]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(0.8, rel=1e-4)

    def test_equal_weighted_average(self):
        entries = [
            ScoreEntry(metric="a", value=1.0, weight=2.0),
            ScoreEntry(metric="b", value=0.0, weight=2.0),
        ]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(0.5, rel=1e-3)

    def test_passed_above_threshold(self):
        entries = [_entry("m1", 0.8)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert r.passed is True

    def test_not_passed_below_threshold(self):
        entries = [_entry("m1", 0.3)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert r.passed is False

    def test_n_metrics_correct(self):
        entries = [_entry("a"), _entry("b"), _entry("c")]
        r = compute_summary(entries)
        assert r.n_metrics == 3

    def test_returns_scoring_report(self):
        r = compute_summary([_entry()])
        assert isinstance(r, ScoringReport)

    def test_cfg_weights_override(self):
        entries = [_entry("m1", 0.5, weight=1.0)]
        cfg = ReportConfig(weights={"m1": 10.0})
        r = compute_summary(entries, cfg)
        assert r.total_score == pytest.approx(0.5, rel=1e-3)

    def test_total_score_in_zero_one(self):
        entries = [_entry("a", 0.3), _entry("b", 0.9)]
        r = compute_summary(entries)
        assert 0.0 <= r.total_score <= 1.0

    def test_pass_threshold_stored(self):
        r = compute_summary([], ReportConfig(pass_threshold=0.75))
        assert r.pass_threshold == pytest.approx(0.75)


# ─── TestFormatReport ─────────────────────────────────────────────────────────

class TestFormatReport:
    def test_returns_string(self):
        r = compute_summary([_entry()])
        assert isinstance(format_report(r), str)

    def test_contains_status(self):
        r = compute_summary([_entry()])
        text = format_report(r)
        assert "PASS" in text or "FAIL" in text

    def test_contains_metric_name(self):
        r = compute_summary([_entry("precision", 0.9)])
        text = format_report(r)
        assert "precision" in text

    def test_contains_total_score(self):
        r = compute_summary([_entry("m1", 0.75)])
        text = format_report(r)
        assert "0.75" in text

    def test_empty_report_no_error(self):
        r = compute_summary([])
        text = format_report(r)
        assert isinstance(text, str)

    def test_contains_threshold(self):
        r = compute_summary([_entry()], ReportConfig(pass_threshold=0.5))
        text = format_report(r)
        assert "0.5" in text


# ─── TestFilterReport ─────────────────────────────────────────────────────────

class TestFilterReport:
    def test_min_greater_than_max_raises(self):
        r = compute_summary([_entry()])
        with pytest.raises(ValueError):
            filter_report(r, min_value=0.8, max_value=0.2)

    def test_filter_keeps_in_range(self):
        entries = [_entry("a", 0.3), _entry("b", 0.7), _entry("c", 0.9)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=1.0)
        assert all(e.value >= 0.5 for e in filtered.entries)

    def test_filter_removes_out_of_range(self):
        entries = [_entry("a", 0.2), _entry("b", 0.8)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=1.0)
        assert len(filtered.entries) == 1
        assert filtered.entries[0].metric == "b"

    def test_filter_all_out_returns_empty(self):
        entries = [_entry("a", 0.2), _entry("b", 0.3)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=1.0)
        assert filtered.n_metrics == 0

    def test_filter_returns_scoring_report(self):
        r = compute_summary([_entry()])
        filtered = filter_report(r)
        assert isinstance(filtered, ScoringReport)

    def test_filter_equal_min_max(self):
        entries = [_entry("a", 0.7), _entry("b", 0.8)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.7, max_value=0.7)
        assert len(filtered.entries) == 1


# ─── TestCompareReports ───────────────────────────────────────────────────────

class TestCompareReports:
    def test_returns_dict(self):
        r1 = compute_summary([_entry("m1", 0.5)])
        r2 = compute_summary([_entry("m1", 0.7)])
        result = compare_reports(r1, r2)
        assert isinstance(result, dict)

    def test_common_metric_delta(self):
        r1 = compute_summary([_entry("acc", 0.5)])
        r2 = compute_summary([_entry("acc", 0.8)])
        result = compare_reports(r1, r2)
        assert result["acc"] == pytest.approx(0.3)

    def test_no_common_metrics_empty(self):
        r1 = compute_summary([_entry("a", 0.5)])
        r2 = compute_summary([_entry("b", 0.7)])
        result = compare_reports(r1, r2)
        assert result == {}

    def test_negative_delta(self):
        r1 = compute_summary([_entry("m1", 0.8)])
        r2 = compute_summary([_entry("m1", 0.4)])
        result = compare_reports(r1, r2)
        assert result["m1"] < 0.0

    def test_only_common_metrics_returned(self):
        r1 = compute_summary([_entry("a", 0.5), _entry("b", 0.6)])
        r2 = compute_summary([_entry("a", 0.7), _entry("c", 0.8)])
        result = compare_reports(r1, r2)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result

    def test_zero_delta(self):
        r1 = compute_summary([_entry("m", 0.6)])
        r2 = compute_summary([_entry("m", 0.6)])
        result = compare_reports(r1, r2)
        assert result["m"] == pytest.approx(0.0)


# ─── TestExportReport ─────────────────────────────────────────────────────────

class TestExportReport:
    def test_returns_dict(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert isinstance(result, dict)

    def test_has_total_score(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert "total_score" in result

    def test_has_status(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert "status" in result

    def test_has_n_metrics(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert "n_metrics" in result

    def test_has_entries(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert "entries" in result

    def test_include_meta_true(self):
        e = ScoreEntry(metric="m", value=0.5, meta={"k": "v"})
        r = compute_summary([e])
        cfg = ReportConfig(include_meta=True)
        result = export_report(r, cfg)
        assert "meta" in result["entries"][0]

    def test_include_meta_false(self):
        e = ScoreEntry(metric="m", value=0.5, meta={"k": "v"})
        r = compute_summary([e])
        cfg = ReportConfig(include_meta=False)
        result = export_report(r, cfg)
        assert "meta" not in result["entries"][0]

    def test_pass_threshold_in_export(self):
        r = compute_summary([_entry()])
        result = export_report(r)
        assert "pass_threshold" in result

    def test_entries_have_metric_and_value(self):
        r = compute_summary([_entry("acc", 0.9)])
        result = export_report(r)
        entry = result["entries"][0]
        assert entry["metric"] == "acc"
        assert entry["value"] == pytest.approx(0.9)


# ─── TestBatchScoreReport ─────────────────────────────────────────────────────

class TestBatchScoreReport:
    def test_returns_list(self):
        result = batch_score_report([[_entry()], [_entry("b", 0.6)]])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_score_report([[_entry()], [_entry("b", 0.6)], []])
        assert len(result) == 3

    def test_each_is_scoring_report(self):
        for r in batch_score_report([[_entry()]]):
            assert isinstance(r, ScoringReport)

    def test_empty_list_returns_empty(self):
        result = batch_score_report([])
        assert result == []

    def test_cfg_applied_to_all(self):
        cfg = ReportConfig(pass_threshold=0.9)
        entries1 = [_entry("m", 0.5)]
        entries2 = [_entry("m", 0.5)]
        results = batch_score_report([entries1, entries2], cfg)
        assert all(not r.passed for r in results)

    def test_empty_entry_list_gives_zero_score(self):
        results = batch_score_report([[]])
        assert results[0].total_score == pytest.approx(0.0)
