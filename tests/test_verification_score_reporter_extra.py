"""Extra tests for puzzle_reconstruction/verification/score_reporter.py."""
from __future__ import annotations

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

def _e(metric="m", value=0.5, weight=1.0):
    return ScoreEntry(metric=metric, value=value, weight=weight)


def _sum(*entries):
    return compute_summary(list(entries))


# ─── ReportConfig (extra) ─────────────────────────────────────────────────────

class TestReportConfigExtra:
    def test_multiple_weights(self):
        cfg = ReportConfig(weights={"a": 0.5, "b": 2.0})
        assert cfg.weights["a"] == pytest.approx(0.5)
        assert cfg.weights["b"] == pytest.approx(2.0)

    def test_weights_large_value(self):
        cfg = ReportConfig(weights={"m": 100.0})
        assert cfg.weights["m"] == pytest.approx(100.0)

    def test_include_meta_default_true(self):
        cfg = ReportConfig()
        assert cfg.include_meta is True

    def test_threshold_midpoint(self):
        cfg = ReportConfig(pass_threshold=0.5)
        assert cfg.pass_threshold == pytest.approx(0.5)

    def test_threshold_0_75(self):
        cfg = ReportConfig(pass_threshold=0.75)
        assert cfg.pass_threshold == pytest.approx(0.75)

    def test_independent_instances(self):
        c1 = ReportConfig(pass_threshold=0.3)
        c2 = ReportConfig(pass_threshold=0.7)
        assert c1.pass_threshold != c2.pass_threshold


# ─── ScoreEntry (extra) ───────────────────────────────────────────────────────

class TestScoreEntryExtra:
    def test_large_weight_ok(self):
        e = ScoreEntry(metric="m", value=0.5, weight=100.0)
        assert e.weight == pytest.approx(100.0)
        assert e.weighted_value == pytest.approx(50.0)

    def test_weighted_value_with_half_weight(self):
        e = ScoreEntry(metric="m", value=0.6, weight=0.5)
        assert e.weighted_value == pytest.approx(0.3)

    def test_meta_custom_keys(self):
        e = ScoreEntry(metric="m", value=0.5, meta={"source": "ocr", "run": 3})
        assert e.meta["source"] == "ocr"
        assert e.meta["run"] == 3

    def test_metric_name_preserved(self):
        for name in ("iou", "precision", "recall", "f1"):
            e = ScoreEntry(metric=name, value=0.5)
            assert e.metric == name

    def test_value_boundaries(self):
        for v in (0.0, 0.001, 0.5, 0.999, 1.0):
            e = ScoreEntry(metric="m", value=v)
            assert e.value == pytest.approx(v)

    def test_different_entries_independent(self):
        e1 = ScoreEntry(metric="a", value=0.3)
        e2 = ScoreEntry(metric="b", value=0.7)
        assert e1.metric != e2.metric
        assert e1.value != e2.value


# ─── ScoringReport (extra) ────────────────────────────────────────────────────

class TestScoringReportExtra:
    def test_worst_metric_is_min_value(self):
        entries = [_e("a", 0.9), _e("b", 0.2), _e("c", 0.5)]
        r = compute_summary(entries)
        assert r.worst_metric == "b"

    def test_best_metric_present(self):
        entries = [_e("a", 0.3), _e("b", 0.9), _e("c", 0.6)]
        r = compute_summary(entries)
        # The entry with highest value should be accessible
        vals = {e.metric: e.value for e in r.entries}
        assert max(vals, key=lambda k: vals[k]) == "b"

    def test_metric_names_order_preserved(self):
        entries = [_e("x"), _e("y"), _e("z")]
        r = compute_summary(entries)
        assert r.metric_names == ["x", "y", "z"]

    def test_status_pass_when_passed(self):
        r = ScoringReport(entries=[_e()], total_score=0.8,
                          n_metrics=1, passed=True, pass_threshold=0.5)
        assert r.status == "pass"

    def test_status_fail_when_not_passed(self):
        r = ScoringReport(entries=[_e()], total_score=0.2,
                          n_metrics=1, passed=False, pass_threshold=0.5)
        assert r.status == "fail"

    def test_total_score_boundaries(self):
        for ts in (0.0, 0.5, 1.0):
            r = ScoringReport(entries=[], total_score=ts,
                              n_metrics=0, passed=True, pass_threshold=0.5)
            assert r.total_score == pytest.approx(ts)

    def test_n_metrics_matches_entries(self):
        entries = [_e("a"), _e("b"), _e("c")]
        r = compute_summary(entries)
        assert r.n_metrics == len(entries)


# ─── add_score (extra) ────────────────────────────────────────────────────────

class TestAddScoreExtra:
    def test_add_many_metrics(self):
        entries = []
        for i in range(10):
            add_score(entries, f"m{i}", i / 10.0)
        assert len(entries) == 10

    def test_weight_applied_correctly(self):
        entries = []
        e = add_score(entries, "m", 0.4, weight=5.0)
        assert e.weighted_value == pytest.approx(2.0)

    def test_entry_in_list_is_same_object(self):
        entries = []
        e = add_score(entries, "m", 0.5)
        assert entries[0] is e

    def test_meta_dict_preserved(self):
        entries = []
        e = add_score(entries, "m", 0.5, meta={"x": 42})
        assert entries[0].meta["x"] == 42

    def test_value_boundary_zero(self):
        entries = []
        e = add_score(entries, "m", 0.0)
        assert e.value == pytest.approx(0.0)

    def test_value_boundary_one(self):
        entries = []
        e = add_score(entries, "m", 1.0)
        assert e.value == pytest.approx(1.0)


# ─── compute_summary (extra) ──────────────────────────────────────────────────

class TestComputeSummaryExtra:
    def test_all_zero_scores_zero_total(self):
        entries = [_e("a", 0.0), _e("b", 0.0)]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(0.0)

    def test_all_one_scores_one_total(self):
        entries = [_e("a", 1.0), _e("b", 1.0)]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(1.0)

    def test_weight_override_does_not_change_score_value(self):
        # Regardless of weight scale (all same), score is same
        entries = [_e("a", 0.6, weight=3.0), _e("b", 0.4, weight=3.0)]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(0.5, abs=1e-4)

    def test_pass_when_score_clearly_above_threshold(self):
        entries = [_e("m", 0.8)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert r.passed is True

    def test_fail_when_score_clearly_below_threshold(self):
        entries = [_e("m", 0.3)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.7))
        assert r.passed is False

    def test_single_metric_score_equals_value(self):
        entries = [_e("m", 0.83)]
        r = compute_summary(entries)
        assert r.total_score == pytest.approx(0.83, rel=1e-3)

    def test_many_metrics_score_in_range(self):
        entries = [_e(f"m{i}", i / 10.0) for i in range(10)]
        r = compute_summary(entries)
        assert 0.0 <= r.total_score <= 1.0


# ─── format_report (extra) ────────────────────────────────────────────────────

class TestFormatReportExtra:
    def test_fail_status_in_text(self):
        entries = [_e("m", 0.1)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.8))
        text = format_report(r)
        assert "FAIL" in text

    def test_pass_status_in_text(self):
        entries = [_e("m", 0.9)]
        r = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        text = format_report(r)
        assert "PASS" in text

    def test_multiple_metrics_in_text(self):
        entries = [_e("precision", 0.8), _e("recall", 0.7)]
        r = compute_summary(entries)
        text = format_report(r)
        assert "precision" in text
        assert "recall" in text

    def test_non_empty_string(self):
        r = compute_summary([_e("m", 0.5)])
        text = format_report(r)
        assert len(text) > 0

    def test_format_with_zero_metrics(self):
        r = compute_summary([])
        text = format_report(r)
        assert isinstance(text, str)


# ─── filter_report (extra) ────────────────────────────────────────────────────

class TestFilterReportExtra:
    def test_no_filter_keeps_all(self):
        entries = [_e("a", 0.3), _e("b", 0.7)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.0, max_value=1.0)
        assert len(filtered.entries) == 2

    def test_filter_exact_match(self):
        entries = [_e("a", 0.5), _e("b", 0.5), _e("c", 0.8)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=0.5)
        assert len(filtered.entries) == 2

    def test_filtered_entries_nonneg_score(self):
        entries = [_e("a", 0.2), _e("b", 0.8)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=1.0)
        assert filtered.total_score >= 0.0

    def test_filter_metric_by_name_if_supported(self):
        entries = [_e("iou", 0.6), _e("f1", 0.4)]
        r = compute_summary(entries)
        # Filter to include only metrics >= 0.5
        filtered = filter_report(r, min_value=0.5)
        metrics = {e.metric for e in filtered.entries}
        assert "iou" in metrics
        assert "f1" not in metrics

    def test_filter_all_entries_gives_zero_metrics(self):
        entries = [_e("a", 0.1), _e("b", 0.2)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=1.0)
        assert filtered.n_metrics == 0


# ─── compare_reports (extra) ──────────────────────────────────────────────────

class TestCompareReportsExtra:
    def test_many_common_metrics(self):
        r1 = compute_summary([_e(f"m{i}", i / 10.0) for i in range(5)])
        r2 = compute_summary([_e(f"m{i}", (9 - i) / 10.0) for i in range(5)])
        result = compare_reports(r1, r2)
        assert len(result) == 5

    def test_positive_delta_when_r2_better(self):
        r1 = compute_summary([_e("acc", 0.5)])
        r2 = compute_summary([_e("acc", 0.9)])
        result = compare_reports(r1, r2)
        assert result["acc"] > 0.0

    def test_negative_delta_when_r1_better(self):
        r1 = compute_summary([_e("acc", 0.9)])
        r2 = compute_summary([_e("acc", 0.5)])
        result = compare_reports(r1, r2)
        assert result["acc"] < 0.0

    def test_delta_within_range(self):
        r1 = compute_summary([_e("m", 0.3)])
        r2 = compute_summary([_e("m", 0.8)])
        result = compare_reports(r1, r2)
        assert -1.0 <= result["m"] <= 1.0

    def test_empty_both_reports_empty_result(self):
        r1 = compute_summary([])
        r2 = compute_summary([])
        result = compare_reports(r1, r2)
        assert result == {}


# ─── export_report (extra) ────────────────────────────────────────────────────

class TestExportReportExtra:
    def test_total_score_value_correct(self):
        r = compute_summary([_e("m", 0.75)])
        result = export_report(r)
        assert result["total_score"] == pytest.approx(0.75, rel=1e-3)

    def test_status_pass_in_export(self):
        r = compute_summary([_e("m", 0.9)], ReportConfig(pass_threshold=0.5))
        result = export_report(r)
        assert result["status"] == "pass"

    def test_status_fail_in_export(self):
        r = compute_summary([_e("m", 0.1)], ReportConfig(pass_threshold=0.5))
        result = export_report(r)
        assert result["status"] == "fail"

    def test_entries_is_list(self):
        r = compute_summary([_e("a"), _e("b")])
        result = export_report(r)
        assert isinstance(result["entries"], list)
        assert len(result["entries"]) == 2

    def test_n_metrics_exported(self):
        entries = [_e(f"m{i}") for i in range(4)]
        r = compute_summary(entries)
        result = export_report(r)
        assert result["n_metrics"] == 4

    def test_meta_excluded_when_not_included(self):
        e = ScoreEntry(metric="m", value=0.5, meta={"k": "v"})
        r = compute_summary([e])
        cfg = ReportConfig(include_meta=False)
        result = export_report(r, cfg)
        assert "meta" not in result["entries"][0]

    def test_meta_included_when_requested(self):
        e = ScoreEntry(metric="m", value=0.5, meta={"key": "val"})
        r = compute_summary([e])
        cfg = ReportConfig(include_meta=True)
        result = export_report(r, cfg)
        assert "meta" in result["entries"][0]
        assert result["entries"][0]["meta"]["key"] == "val"


# ─── batch_score_report (extra) ───────────────────────────────────────────────

class TestBatchScoreReportExtra:
    def test_single_batch(self):
        result = batch_score_report([[_e("m", 0.7)]])
        assert len(result) == 1

    def test_all_pass_with_low_threshold(self):
        cfg = ReportConfig(pass_threshold=0.0)
        entries_list = [[_e("m", 0.01)], [_e("m", 0.5)], [_e("m", 1.0)]]
        results = batch_score_report(entries_list, cfg)
        assert all(r.passed for r in results)

    def test_all_fail_with_high_threshold(self):
        cfg = ReportConfig(pass_threshold=1.0)
        entries_list = [[_e("m", 0.9)], [_e("m", 0.8)]]
        results = batch_score_report(entries_list, cfg)
        assert all(not r.passed for r in results)

    def test_mixed_pass_fail(self):
        cfg = ReportConfig(pass_threshold=0.6)
        entries_list = [[_e("m", 0.8)], [_e("m", 0.3)]]
        results = batch_score_report(entries_list, cfg)
        assert results[0].passed is True
        assert results[1].passed is False

    def test_scores_independent_per_batch(self):
        entries_list = [[_e("a", 0.2)], [_e("a", 0.8)]]
        results = batch_score_report(entries_list)
        assert results[0].total_score != results[1].total_score

    def test_returns_scoring_report_list(self):
        entries_list = [[_e()], [_e("b", 0.3)]]
        results = batch_score_report(entries_list)
        assert all(isinstance(r, ScoringReport) for r in results)

    def test_empty_entries_in_batch(self):
        results = batch_score_report([[], [_e()]])
        assert results[0].total_score == pytest.approx(0.0)
        assert results[1].total_score > 0.0
