"""Extra tests for puzzle_reconstruction/scoring/consistency_checker.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.scoring.consistency_checker import (
    ConsistencyIssue,
    ConsistencyReport,
    check_unique_ids,
    check_all_present,
    check_canvas_bounds,
    check_score_threshold,
    check_gap_uniformity,
    run_consistency_check,
    batch_consistency_check,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _issue(code="ERR", desc="desc", severity="error",
           fids=None) -> ConsistencyIssue:
    return ConsistencyIssue(code=code, description=desc,
                             severity=severity,
                             fragment_ids=fids or [])


def _report(issues=None, errors=0, warnings=0,
             consistent=True, checked=0) -> ConsistencyReport:
    return ConsistencyReport(
        issues=issues or [],
        is_consistent=consistent,
        n_errors=errors,
        n_warnings=warnings,
        checked_pairs=checked,
    )


def _good_check():
    return run_consistency_check(
        fragment_ids=[0, 1, 2],
        expected_ids=[0, 1, 2],
        positions=[(0, 0), (20, 0), (40, 0)],
        sizes=[(10, 10)] * 3,
        canvas_w=100, canvas_h=100,
    )


# ─── ConsistencyIssue ─────────────────────────────────────────────────────────

class TestConsistencyIssueExtra:
    def test_code_stored(self):
        i = _issue(code="DUPLICATE_ID")
        assert i.code == "DUPLICATE_ID"

    def test_description_stored(self):
        i = _issue(desc="some description")
        assert i.description == "some description"

    def test_severity_stored(self):
        i = _issue(severity="warning")
        assert i.severity == "warning"

    def test_fragment_ids_stored(self):
        i = _issue(fids=[1, 2])
        assert i.fragment_ids == [1, 2]

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="X", description="d", severity="critical")

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="", description="d", severity="error")

    def test_info_severity_ok(self):
        i = ConsistencyIssue(code="X", description="d", severity="info")
        assert i.severity == "info"

    def test_default_fragment_ids_empty(self):
        i = _issue()
        assert i.fragment_ids == []


# ─── ConsistencyReport ────────────────────────────────────────────────────────

class TestConsistencyReportExtra:
    def test_is_consistent_stored(self):
        r = _report(consistent=False)
        assert r.is_consistent is False

    def test_n_errors_stored(self):
        r = _report(errors=3)
        assert r.n_errors == 3

    def test_n_warnings_stored(self):
        r = _report(warnings=2)
        assert r.n_warnings == 2

    def test_checked_pairs_stored(self):
        r = _report(checked=5)
        assert r.checked_pairs == 5

    def test_negative_n_errors_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=-1, n_warnings=0)

    def test_negative_n_warnings_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=0, n_warnings=-1)

    def test_negative_checked_pairs_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=0, n_warnings=0, checked_pairs=-1)

    def test_len_counts_issues(self):
        r = _report(issues=[_issue(), _issue()])
        assert len(r) == 2

    def test_len_empty(self):
        assert len(_report()) == 0


# ─── check_unique_ids ─────────────────────────────────────────────────────────

class TestCheckUniqueIdsExtra:
    def test_no_duplicates_empty_result(self):
        assert check_unique_ids([0, 1, 2]) == []

    def test_duplicate_detected(self):
        issues = check_unique_ids([0, 1, 0])
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_ID"

    def test_duplicate_fragment_ids_recorded(self):
        issues = check_unique_ids([0, 1, 0])
        assert 0 in issues[0].fragment_ids

    def test_empty_list_no_issues(self):
        assert check_unique_ids([]) == []

    def test_all_duplicates(self):
        issues = check_unique_ids([5, 5, 5])
        assert len(issues) == 1

    def test_severity_error(self):
        issues = check_unique_ids([0, 0])
        assert issues[0].severity == "error"


# ─── check_all_present ────────────────────────────────────────────────────────

class TestCheckAllPresentExtra:
    def test_all_present_no_issues(self):
        assert check_all_present([0, 1, 2], [0, 1, 2]) == []

    def test_missing_fragment_detected(self):
        issues = check_all_present([0, 2], [0, 1, 2])
        codes = [i.code for i in issues]
        assert "MISSING_FRAGMENT" in codes

    def test_extra_fragment_detected(self):
        issues = check_all_present([0, 1, 2, 3], [0, 1, 2])
        codes = [i.code for i in issues]
        assert "EXTRA_FRAGMENT" in codes

    def test_missing_severity_error(self):
        issues = check_all_present([], [0, 1])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert missing[0].severity == "error"

    def test_extra_severity_warning(self):
        issues = check_all_present([0, 1, 2], [0, 1])
        extra = [i for i in issues if i.code == "EXTRA_FRAGMENT"]
        assert extra[0].severity == "warning"

    def test_empty_both_no_issues(self):
        assert check_all_present([], []) == []


# ─── check_canvas_bounds ──────────────────────────────────────────────────────

class TestCheckCanvasBoundsExtra:
    def test_all_in_bounds_no_issues(self):
        pos = [(0, 0), (50, 50)]
        siz = [(10, 10), (10, 10)]
        assert check_canvas_bounds(pos, siz, 100, 100) == []

    def test_out_of_bounds_detected(self):
        pos = [(95, 0)]
        siz = [(10, 10)]
        issues = check_canvas_bounds(pos, siz, 100, 100)
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_canvas_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([(0, 0)], [(5, 5)], 0, 100)

    def test_canvas_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([(0, 0)], [(5, 5)], 100, 0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([(0, 0)], [(5, 5), (10, 10)], 100, 100)

    def test_negative_x_out_of_bounds(self):
        pos = [(-1, 0)]
        siz = [(5, 5)]
        issues = check_canvas_bounds(pos, siz, 100, 100)
        assert len(issues) == 1

    def test_severity_error(self):
        issues = check_canvas_bounds([(95, 0)], [(10, 10)], 100, 100)
        assert issues[0].severity == "error"


# ─── check_score_threshold ────────────────────────────────────────────────────

class TestCheckScoreThresholdExtra:
    def test_all_above_threshold_no_issues(self):
        scores = {(0, 1): 0.8, (1, 2): 0.9}
        assert check_score_threshold(scores, min_score=0.5) == []

    def test_low_score_detected(self):
        scores = {(0, 1): 0.3}
        issues = check_score_threshold(scores, min_score=0.5)
        assert len(issues) == 1
        assert issues[0].code == "LOW_SCORE"

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            check_score_threshold({}, min_score=-0.1)

    def test_severity_warning(self):
        issues = check_score_threshold({(0, 1): 0.1}, min_score=0.5)
        assert issues[0].severity == "warning"

    def test_empty_scores_no_issues(self):
        assert check_score_threshold({}) == []

    def test_fragment_ids_in_issue(self):
        issues = check_score_threshold({(3, 7): 0.1}, min_score=0.5)
        assert 3 in issues[0].fragment_ids
        assert 7 in issues[0].fragment_ids


# ─── check_gap_uniformity ─────────────────────────────────────────────────────

class TestCheckGapUniformityExtra:
    def test_uniform_gaps_no_issues(self):
        assert check_gap_uniformity([5.0, 5.0, 5.0]) == []

    def test_variable_gaps_issue(self):
        issues = check_gap_uniformity([1.0, 50.0, 2.0], max_std=3.0)
        assert len(issues) == 1
        assert issues[0].code == "NONUNIFORM_GAP"

    def test_single_gap_no_issues(self):
        assert check_gap_uniformity([5.0]) == []

    def test_empty_gaps_no_issues(self):
        assert check_gap_uniformity([]) == []

    def test_negative_max_std_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity([1.0, 2.0], max_std=-1.0)

    def test_severity_warning(self):
        issues = check_gap_uniformity([1.0, 100.0], max_std=1.0)
        assert issues[0].severity == "warning"


# ─── run_consistency_check ────────────────────────────────────────────────────

class TestRunConsistencyCheckExtra:
    def test_returns_consistency_report(self):
        assert isinstance(_good_check(), ConsistencyReport)

    def test_good_assembly_consistent(self):
        assert _good_check().is_consistent is True

    def test_good_assembly_no_errors(self):
        assert _good_check().n_errors == 0

    def test_duplicate_ids_fail(self):
        r = run_consistency_check(
            fragment_ids=[0, 0, 1],
            expected_ids=[0, 1],
            positions=[(0, 0), (10, 0), (20, 0)],
            sizes=[(5, 5)] * 3,
            canvas_w=100, canvas_h=100,
        )
        assert r.n_errors >= 1

    def test_out_of_bounds_fail(self):
        r = run_consistency_check(
            fragment_ids=[0],
            expected_ids=[0],
            positions=[(95, 0)],
            sizes=[(10, 10)],
            canvas_w=100, canvas_h=100,
        )
        assert r.n_errors >= 1

    def test_pair_scores_checked(self):
        r = run_consistency_check(
            fragment_ids=[0, 1],
            expected_ids=[0, 1],
            positions=[(0, 0), (20, 0)],
            sizes=[(10, 10)] * 2,
            canvas_w=100, canvas_h=100,
            pair_scores={(0, 1): 0.1},
            min_score=0.5,
        )
        assert r.checked_pairs == 1

    def test_no_pair_scores_zero_checked(self):
        r = _good_check()
        assert r.checked_pairs == 0


# ─── batch_consistency_check ──────────────────────────────────────────────────

class TestBatchConsistencyCheckExtra:
    def _asm(self, fids=None, expected=None):
        fids = fids or [0, 1]
        expected = expected or [0, 1]
        return {
            "fragment_ids": fids,
            "expected_ids": expected,
            "positions": [(i * 20, 0) for i in range(len(fids))],
            "sizes": [(10, 10)] * len(fids),
            "canvas_w": 200,
            "canvas_h": 100,
        }

    def test_returns_list(self):
        result = batch_consistency_check([self._asm()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_consistency_check([self._asm(), self._asm()])
        assert len(result) == 2

    def test_empty_batch(self):
        assert batch_consistency_check([]) == []

    def test_each_element_is_report(self):
        for r in batch_consistency_check([self._asm()]):
            assert isinstance(r, ConsistencyReport)

    def test_good_assembly_consistent(self):
        result = batch_consistency_check([self._asm()])
        assert result[0].is_consistent is True
