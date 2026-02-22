"""Extra tests for puzzle_reconstruction.verification.score_reporter."""
import pytest
from puzzle_reconstruction.verification.score_reporter import (
    ReportConfig,
    ScoreEntry,
    ScoringReport,
    add_score,
    batch_score_report,
    compare_reports,
    compute_summary,
    export_report,
    filter_report,
    format_report,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(metric="a", value=0.5, weight=1.0):
    return ScoreEntry(metric=metric, value=value, weight=weight)


def _report(entries=None, threshold=0.5):
    if entries is None:
        entries = [_entry()]
    return compute_summary(entries, ReportConfig(pass_threshold=threshold))


# ─── ReportConfig extras ──────────────────────────────────────────────────────

class TestReportConfigExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(ReportConfig()), str)

    def test_include_meta_false(self):
        cfg = ReportConfig(include_meta=False)
        assert cfg.include_meta is False

    def test_pass_threshold_zero_valid(self):
        cfg = ReportConfig(pass_threshold=0.0)
        assert cfg.pass_threshold == pytest.approx(0.0)

    def test_pass_threshold_one_valid(self):
        cfg = ReportConfig(pass_threshold=1.0)
        assert cfg.pass_threshold == pytest.approx(1.0)

    def test_multiple_weights(self):
        cfg = ReportConfig(weights={"x": 1.0, "y": 2.0, "z": 3.0})
        assert cfg.weights["z"] == pytest.approx(3.0)

    def test_weight_positive_required(self):
        with pytest.raises(ValueError):
            ReportConfig(weights={"x": -1.0})

    def test_empty_weights_ok(self):
        cfg = ReportConfig(weights={})
        assert cfg.weights == {}


# ─── ScoreEntry extras ────────────────────────────────────────────────────────

class TestScoreEntryExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(_entry()), str)

    def test_value_zero_valid(self):
        e = ScoreEntry(metric="x", value=0.0)
        assert e.value == pytest.approx(0.0)

    def test_value_one_valid(self):
        e = ScoreEntry(metric="x", value=1.0)
        assert e.value == pytest.approx(1.0)

    def test_weight_zero_valid(self):
        e = ScoreEntry(metric="x", value=0.5, weight=0.0)
        assert e.weight == pytest.approx(0.0)
        assert e.weighted_value == pytest.approx(0.0)

    def test_meta_stored(self):
        e = ScoreEntry(metric="x", value=0.5, meta={"tag": "color"})
        assert e.meta["tag"] == "color"

    def test_metric_with_underscore(self):
        e = ScoreEntry(metric="boundary_score", value=0.7)
        assert e.metric == "boundary_score"

    def test_weighted_value_fractional(self):
        e = ScoreEntry(metric="x", value=0.4, weight=5.0)
        assert e.weighted_value == pytest.approx(2.0)

    def test_high_weight_scales_correctly(self):
        e = ScoreEntry(metric="x", value=1.0, weight=100.0)
        assert e.weighted_value == pytest.approx(100.0)


# ─── ScoringReport extras ─────────────────────────────────────────────────────

class TestScoringReportExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(_report()), str)

    def test_status_pass_when_passed(self):
        r = _report([_entry("a", 0.9)], threshold=0.5)
        assert r.status == "pass"

    def test_status_fail_when_failed(self):
        r = _report([_entry("a", 0.1)], threshold=0.5)
        assert r.status == "fail"

    def test_metric_names_multiple(self):
        entries = [_entry("x"), _entry("y"), _entry("z")]
        r = compute_summary(entries)
        assert set(r.metric_names) == {"x", "y", "z"}

    def test_metric_names_single(self):
        r = _report([_entry("alpha")])
        assert r.metric_names == ["alpha"]

    def test_worst_metric_with_three(self):
        entries = [
            ScoreEntry("a", 0.9),
            ScoreEntry("b", 0.2),
            ScoreEntry("c", 0.7),
        ]
        r = compute_summary(entries)
        assert r.worst_metric == "b"

    def test_total_score_in_range(self):
        entries = [ScoreEntry("a", 0.3), ScoreEntry("b", 0.7)]
        r = compute_summary(entries)
        assert 0.0 <= r.total_score <= 1.0

    def test_n_metrics_stored(self):
        entries = [_entry() for _ in range(5)]
        r = compute_summary(entries)
        assert r.n_metrics == 5

    def test_pass_threshold_stored(self):
        r = compute_summary([_entry()], ReportConfig(pass_threshold=0.75))
        assert r.pass_threshold == pytest.approx(0.75)


# ─── add_score extras ─────────────────────────────────────────────────────────

class TestAddScoreExtra:
    def test_return_value_is_score_entry(self):
        entries = []
        result = add_score(entries, "x", 0.5)
        assert isinstance(result, ScoreEntry)

    def test_meta_default_empty(self):
        entries = []
        e = add_score(entries, "x", 0.5)
        assert e.meta == {}

    def test_five_consecutive_appends(self):
        entries = []
        for i in range(5):
            add_score(entries, f"m{i}", 0.1 * (i + 1))
        assert len(entries) == 5

    def test_weight_stored(self):
        entries = []
        e = add_score(entries, "x", 0.5, weight=3.0)
        assert e.weight == pytest.approx(3.0)

    def test_metric_name_preserved(self):
        entries = []
        e = add_score(entries, "boundary_match", 0.6)
        assert e.metric == "boundary_match"


# ─── compute_summary extras ───────────────────────────────────────────────────

class TestComputeSummaryExtra:
    def test_single_entry_score_equals_value(self):
        report = compute_summary([ScoreEntry("a", 0.65)])
        assert report.total_score == pytest.approx(0.65)

    def test_three_equal_weight_mean(self):
        entries = [ScoreEntry("a", 0.2), ScoreEntry("b", 0.4), ScoreEntry("c", 0.6)]
        report = compute_summary(entries)
        assert report.total_score == pytest.approx(0.4, abs=1e-5)

    def test_weighted_two_entries(self):
        entries = [
            ScoreEntry("a", 1.0, weight=3.0),
            ScoreEntry("b", 0.0, weight=1.0),
        ]
        report = compute_summary(entries)
        # (1.0*3 + 0.0*1) / 4 = 0.75
        assert report.total_score == pytest.approx(0.75, abs=1e-5)

    def test_clearly_above_threshold_passes(self):
        entries = [ScoreEntry("a", 0.8)]
        report = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert report.passed is True

    def test_just_below_threshold_fails(self):
        entries = [ScoreEntry("a", 0.499)]
        report = compute_summary(entries, ReportConfig(pass_threshold=0.5))
        assert report.passed is False

    def test_empty_passed_is_false(self):
        report = compute_summary([])
        assert report.passed is False

    def test_config_weights_applied(self):
        entries = [ScoreEntry("a", 0.0), ScoreEntry("b", 1.0)]
        cfg = ReportConfig(weights={"a": 1.0, "b": 9.0})
        report = compute_summary(entries, cfg)
        # (0.0*1 + 1.0*9) / 10 = 0.9
        assert report.total_score == pytest.approx(0.9, abs=1e-5)


# ─── format_report extras ─────────────────────────────────────────────────────

class TestFormatReportExtra:
    def test_returns_string(self):
        r = _report([_entry("boundary", 0.7)])
        assert isinstance(format_report(r), str)

    def test_contains_metric_name(self):
        r = _report([ScoreEntry("edge_quality", 0.8)])
        text = format_report(r)
        assert "edge_quality" in text

    def test_contains_score(self):
        r = _report([ScoreEntry("x", 0.55)])
        text = format_report(r)
        # Total score 0.55 should appear somewhere
        assert "0.5" in text or "55" in text

    def test_nonempty_string(self):
        r = _report()
        assert len(format_report(r)) > 0

    def test_multi_entry_all_names(self):
        entries = [ScoreEntry("alpha", 0.9), ScoreEntry("beta", 0.4),
                   ScoreEntry("gamma", 0.6)]
        r = compute_summary(entries)
        text = format_report(r)
        for name in ("alpha", "beta", "gamma"):
            assert name in text


# ─── filter_report extras ─────────────────────────────────────────────────────

class TestFilterReportExtra:
    def test_min_0_keeps_all(self):
        entries = [ScoreEntry("a", 0.1), ScoreEntry("b", 0.9)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.0)
        assert filtered.n_metrics == 2

    def test_max_1_keeps_all(self):
        entries = [ScoreEntry("a", 0.1), ScoreEntry("b", 0.9)]
        r = compute_summary(entries)
        filtered = filter_report(r, max_value=1.0)
        assert filtered.n_metrics == 2

    def test_single_entry_kept(self):
        r = _report([ScoreEntry("x", 0.5)])
        filtered = filter_report(r, min_value=0.4)
        assert filtered.n_metrics == 1

    def test_single_entry_removed(self):
        r = _report([ScoreEntry("x", 0.3)])
        filtered = filter_report(r, min_value=0.5)
        assert filtered.n_metrics == 0

    def test_returns_scoring_report(self):
        r = _report()
        filtered = filter_report(r)
        assert isinstance(filtered, ScoringReport)

    def test_min_max_exact_boundary(self):
        entries = [ScoreEntry("a", 0.5)]
        r = compute_summary(entries)
        filtered = filter_report(r, min_value=0.5, max_value=0.5)
        assert filtered.n_metrics == 1


# ─── compare_reports extras ───────────────────────────────────────────────────

class TestCompareReportsExtra:
    def test_negative_delta(self):
        ra = compute_summary([ScoreEntry("a", 0.9)])
        rb = compute_summary([ScoreEntry("a", 0.6)])
        diff = compare_reports(ra, rb)
        assert diff["a"] == pytest.approx(-0.3, abs=1e-6)

    def test_zero_delta_same_values(self):
        ra = compute_summary([ScoreEntry("a", 0.7)])
        rb = compute_summary([ScoreEntry("a", 0.7)])
        diff = compare_reports(ra, rb)
        assert diff["a"] == pytest.approx(0.0, abs=1e-6)

    def test_multiple_common_metrics(self):
        ra = compute_summary([ScoreEntry("a", 0.5), ScoreEntry("b", 0.6)])
        rb = compute_summary([ScoreEntry("a", 0.7), ScoreEntry("b", 0.4)])
        diff = compare_reports(ra, rb)
        assert set(diff.keys()) == {"a", "b"}
        assert diff["a"] == pytest.approx(0.2, abs=1e-6)
        assert diff["b"] == pytest.approx(-0.2, abs=1e-6)

    def test_returns_dict(self):
        ra = compute_summary([ScoreEntry("a", 0.5)])
        rb = compute_summary([ScoreEntry("a", 0.5)])
        assert isinstance(compare_reports(ra, rb), dict)

    def test_no_common_returns_empty(self):
        ra = compute_summary([ScoreEntry("x", 0.5)])
        rb = compute_summary([ScoreEntry("y", 0.5)])
        assert compare_reports(ra, rb) == {}


# ─── export_report extras ─────────────────────────────────────────────────────

class TestExportReportExtra:
    def test_status_pass_in_export(self):
        r = _report([ScoreEntry("a", 0.9)], threshold=0.5)
        exported = export_report(r)
        assert exported["status"] == "pass"

    def test_status_fail_in_export(self):
        r = _report([ScoreEntry("a", 0.1)], threshold=0.5)
        exported = export_report(r)
        assert exported["status"] == "fail"

    def test_n_metrics_correct(self):
        entries = [ScoreEntry("a", 0.5), ScoreEntry("b", 0.6), ScoreEntry("c", 0.7)]
        r = compute_summary(entries)
        exported = export_report(r)
        assert exported["n_metrics"] == 3

    def test_entries_length_correct(self):
        entries = [ScoreEntry("a", 0.5), ScoreEntry("b", 0.6)]
        r = compute_summary(entries)
        exported = export_report(r)
        assert len(exported["entries"]) == 2

    def test_total_score_in_export(self):
        r = _report([ScoreEntry("a", 0.8)])
        exported = export_report(r)
        assert abs(exported["total_score"] - 0.8) < 1e-5

    def test_meta_absent_when_excluded(self):
        entries = [ScoreEntry("a", 0.7, meta={"k": 1})]
        r = compute_summary(entries)
        exported = export_report(r, ReportConfig(include_meta=False))
        assert "meta" not in exported["entries"][0]


# ─── batch_score_report extras ────────────────────────────────────────────────

class TestBatchScoreReportExtra:
    def test_three_reports(self):
        entry_lists = [[ScoreEntry("a", v)] for v in (0.2, 0.5, 0.8)]
        reports = batch_score_report(entry_lists)
        assert len(reports) == 3

    def test_all_pass_high_scores(self):
        cfg = ReportConfig(pass_threshold=0.5)
        entry_lists = [[ScoreEntry("a", 0.9)] for _ in range(4)]
        reports = batch_score_report(entry_lists, cfg)
        assert all(r.passed for r in reports)

    def test_all_fail_low_scores(self):
        cfg = ReportConfig(pass_threshold=0.8)
        entry_lists = [[ScoreEntry("a", 0.3)] for _ in range(3)]
        reports = batch_score_report(entry_lists, cfg)
        assert all(not r.passed for r in reports)

    def test_all_scoring_reports(self):
        entry_lists = [[ScoreEntry("a", 0.5)] for _ in range(3)]
        for r in batch_score_report(entry_lists):
            assert isinstance(r, ScoringReport)

    def test_single_entry_list(self):
        reports = batch_score_report([[ScoreEntry("x", 0.7)]])
        assert len(reports) == 1
        assert reports[0].n_metrics == 1
