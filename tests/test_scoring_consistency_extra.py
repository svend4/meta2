"""Extra tests for puzzle_reconstruction/scoring/consistency_checker.py"""
import pytest

from puzzle_reconstruction.scoring.consistency_checker import (
    ConsistencyIssue,
    ConsistencyReport,
    batch_consistency_check,
    check_all_present,
    check_canvas_bounds,
    check_gap_uniformity,
    check_score_threshold,
    check_unique_ids,
    run_consistency_check,
)


# ─── TestConsistencyIssueExtra ────────────────────────────────────────────────

class TestConsistencyIssueExtra:
    def test_severity_warning(self):
        iss = ConsistencyIssue(code="LOW_SCORE", description="low", severity="warning")
        assert iss.severity == "warning"

    def test_severity_info(self):
        iss = ConsistencyIssue(code="INFO_CODE", description="info", severity="info")
        assert iss.severity == "info"

    def test_severity_error_default(self):
        iss = ConsistencyIssue(code="ERR", description="error")
        assert iss.severity == "error"

    def test_fragment_ids_many(self):
        iss = ConsistencyIssue(code="X", description="d",
                               fragment_ids=[0, 1, 2, 3, 4])
        assert len(iss.fragment_ids) == 5

    def test_code_long_string(self):
        code = "CANVAS_BOUNDS_EXCEEDED"
        iss = ConsistencyIssue(code=code, description="desc")
        assert iss.code == code

    def test_description_long_string(self):
        desc = "Fragment 5 is positioned outside canvas boundaries"
        iss = ConsistencyIssue(code="OUT_OF_BOUNDS", description=desc)
        assert iss.description == desc

    def test_fragment_ids_empty_by_default(self):
        iss = ConsistencyIssue(code="X", description="d")
        assert iss.fragment_ids == []


# ─── TestConsistencyReportExtra ───────────────────────────────────────────────

class TestConsistencyReportExtra:
    def _make(self, n_err=0, n_warn=0, issues=None, checked=0):
        return ConsistencyReport(
            issues=issues or [],
            is_consistent=(n_err == 0),
            n_errors=n_err,
            n_warnings=n_warn,
            checked_pairs=checked,
        )

    def test_len_with_issues(self):
        iss = [ConsistencyIssue(code="X", description="d") for _ in range(4)]
        rpt = ConsistencyReport(
            issues=iss, is_consistent=False, n_errors=4, n_warnings=0
        )
        assert len(rpt) == 4

    def test_n_errors_stored(self):
        rpt = self._make(n_err=3)
        assert rpt.n_errors == 3

    def test_n_warnings_stored(self):
        rpt = self._make(n_warn=5)
        assert rpt.n_warnings == 5

    def test_is_consistent_false_when_errors(self):
        rpt = self._make(n_err=1)
        assert rpt.is_consistent is False

    def test_checked_pairs_stored(self):
        rpt = self._make(checked=10)
        assert rpt.checked_pairs == 10

    def test_no_issues_zero_len(self):
        rpt = self._make()
        assert len(rpt) == 0


# ─── TestCheckUniqueIdsExtra ─────────────────────────────────────────────────

class TestCheckUniqueIdsExtra:
    def test_large_unique_list(self):
        ids = list(range(100))
        assert check_unique_ids(ids) == []

    def test_multiple_duplicates_one_issue(self):
        ids = [0, 0, 1, 1, 2, 2]
        issues = check_unique_ids(ids)
        assert len(issues) == 1

    def test_triplicate_detected(self):
        ids = [0, 0, 0, 1, 2]
        issues = check_unique_ids(ids)
        assert len(issues) == 1
        assert 0 in issues[0].fragment_ids

    def test_all_same_detected(self):
        ids = [5, 5, 5, 5]
        issues = check_unique_ids(ids)
        assert len(issues) == 1

    def test_two_elements_unique(self):
        assert check_unique_ids([7, 8]) == []

    def test_severity_is_error(self):
        issues = check_unique_ids([1, 1])
        assert issues[0].severity == "error"


# ─── TestCheckAllPresentExtra ─────────────────────────────────────────────────

class TestCheckAllPresentExtra:
    def test_large_all_present(self):
        ids = list(range(50))
        assert check_all_present(ids, ids) == []

    def test_multiple_missing(self):
        issues = check_all_present([0, 3], [0, 1, 2, 3])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert len(missing) >= 1

    def test_multiple_extra(self):
        issues = check_all_present([0, 1, 2, 3, 4], [0, 1, 2])
        extra = [i for i in issues if i.code == "EXTRA_FRAGMENT"]
        assert len(extra) >= 1

    def test_single_missing(self):
        issues = check_all_present([0, 2], [0, 1, 2])
        codes = [i.code for i in issues]
        assert "MISSING_FRAGMENT" in codes

    def test_single_extra(self):
        issues = check_all_present([0, 1, 2, 3], [0, 1, 2])
        codes = [i.code for i in issues]
        assert "EXTRA_FRAGMENT" in codes

    def test_empty_present_empty_expected_no_issues(self):
        assert check_all_present([], []) == []

    def test_extra_severity_warning(self):
        issues = check_all_present([0, 1, 2], [0, 1])
        extra = [i for i in issues if i.code == "EXTRA_FRAGMENT"]
        assert extra[0].severity == "warning"


# ─── TestCheckCanvasBoundsExtra ───────────────────────────────────────────────

class TestCheckCanvasBoundsExtra:
    def test_many_fragments_in_bounds(self):
        n = 10
        positions = [(i * 5, 0) for i in range(n)]
        sizes = [(4, 4)] * n
        issues = check_canvas_bounds(positions, sizes, canvas_w=200, canvas_h=100)
        assert issues == []

    def test_negative_y_detected(self):
        issues = check_canvas_bounds([(0, -1)], [(10, 10)],
                                     canvas_w=100, canvas_h=100)
        assert len(issues) == 1

    def test_boundary_exactly_at_edge(self):
        # Fragment at (90,0) with size (10,10) → x+w == 100 == canvas_w → in bounds
        issues = check_canvas_bounds([(90, 0)], [(10, 10)],
                                     canvas_w=100, canvas_h=100)
        assert issues == []

    def test_one_pixel_over_x_detected(self):
        # x + w = 101 > 100
        issues = check_canvas_bounds([(91, 0)], [(10, 10)],
                                     canvas_w=100, canvas_h=100)
        assert len(issues) == 1

    def test_one_pixel_over_y_detected(self):
        issues = check_canvas_bounds([(0, 91)], [(10, 10)],
                                     canvas_w=100, canvas_h=100)
        assert len(issues) == 1

    def test_canvas_h_negative_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([(0, 0)], [(5, 5)], canvas_w=100, canvas_h=-1)

    def test_multiple_oob_fragments(self):
        positions = [(200, 0), (0, 200), (0, 0)]
        sizes = [(10, 10)] * 3
        issues = check_canvas_bounds(positions, sizes, canvas_w=100, canvas_h=100)
        assert len(issues) == 2

    def test_empty_positions_no_issues(self):
        issues = check_canvas_bounds([], [], canvas_w=100, canvas_h=100)
        assert issues == []


# ─── TestCheckScoreThresholdExtra ────────────────────────────────────────────

class TestCheckScoreThresholdExtra:
    def test_many_above_threshold(self):
        scores = {(i, i + 1): 0.8 for i in range(10)}
        assert check_score_threshold(scores, min_score=0.5) == []

    def test_many_below_threshold(self):
        scores = {(i, i + 1): 0.1 for i in range(5)}
        issues = check_score_threshold(scores, min_score=0.5)
        assert len(issues) == 5

    def test_fragment_ids_in_issue(self):
        issues = check_score_threshold({(3, 7): 0.1}, min_score=0.5)
        assert 3 in issues[0].fragment_ids
        assert 7 in issues[0].fragment_ids

    def test_severity_warning(self):
        issues = check_score_threshold({(0, 1): 0.1}, min_score=0.5)
        assert issues[0].severity == "warning"

    def test_exactly_at_threshold_no_issue(self):
        assert check_score_threshold({(0, 1): 0.5}, min_score=0.5) == []

    def test_min_score_zero_all_ok(self):
        scores = {(i, i + 1): 0.01 for i in range(5)}
        assert check_score_threshold(scores, min_score=0.0) == []

    def test_mixed_scores(self):
        scores = {(0, 1): 0.8, (1, 2): 0.3, (2, 3): 0.9, (3, 4): 0.1}
        issues = check_score_threshold(scores, min_score=0.5)
        assert len(issues) == 2


# ─── TestCheckGapUniformityExtra ──────────────────────────────────────────────

class TestCheckGapUniformityExtra:
    def test_large_uniform_gaps(self):
        gaps = [5.0] * 20
        assert check_gap_uniformity(gaps, max_std=0.5) == []

    def test_max_std_large(self):
        gaps = [1.0, 100.0, 50.0]
        # Large max_std → no issue
        issues = check_gap_uniformity(gaps, max_std=1000.0)
        assert issues == []

    def test_slightly_nonuniform_below_threshold(self):
        gaps = [4.8, 5.0, 5.2]
        issues = check_gap_uniformity(gaps, max_std=1.0)
        assert issues == []

    def test_three_equal_gaps(self):
        assert check_gap_uniformity([3.0, 3.0, 3.0], max_std=0.0) == []

    def test_two_very_different_gaps(self):
        issues = check_gap_uniformity([0.0, 100.0], max_std=1.0)
        assert len(issues) == 1
        assert issues[0].code == "NONUNIFORM_GAP"

    def test_severity_warning(self):
        issues = check_gap_uniformity([0.0, 100.0], max_std=1.0)
        assert issues[0].severity == "warning"


# ─── TestRunConsistencyCheckExtra ─────────────────────────────────────────────

class TestRunConsistencyCheckExtra:
    def _valid(self, n=4):
        return dict(
            fragment_ids=list(range(n)),
            expected_ids=list(range(n)),
            positions=[(i * 20, 0) for i in range(n)],
            sizes=[(15, 15)] * n,
            canvas_w=200,
            canvas_h=100,
        )

    def test_five_fragments_valid(self):
        rpt = run_consistency_check(**self._valid(5))
        assert rpt.is_consistent is True

    def test_single_fragment_valid(self):
        rpt = run_consistency_check(**self._valid(1))
        assert rpt.is_consistent is True

    def test_duplicate_ids_inconsistent(self):
        args = self._valid(4)
        args["fragment_ids"] = [0, 0, 2, 3]
        rpt = run_consistency_check(**args)
        assert rpt.is_consistent is False
        assert rpt.n_errors > 0

    def test_extra_fragment_warning(self):
        args = self._valid(4)
        args["fragment_ids"] = [0, 1, 2, 3, 4]  # extra
        rpt = run_consistency_check(**args)
        assert rpt.n_warnings > 0

    def test_multiple_low_scores_warnings(self):
        args = self._valid(4)
        args["pair_scores"] = {(0, 1): 0.1, (1, 2): 0.2, (2, 3): 0.3}
        rpt = run_consistency_check(**args, min_score=0.5)
        assert rpt.n_warnings >= 3

    def test_checked_pairs_zero_when_no_scores(self):
        rpt = run_consistency_check(**self._valid(4))
        assert rpt.checked_pairs == 0

    def test_checked_pairs_counted(self):
        args = self._valid(4)
        args["pair_scores"] = {(0, 1): 0.9, (1, 2): 0.8, (2, 3): 0.7}
        rpt = run_consistency_check(**args)
        assert rpt.checked_pairs == 3

    def test_returns_consistency_report(self):
        assert isinstance(run_consistency_check(**self._valid(3)), ConsistencyReport)


# ─── TestBatchConsistencyCheckExtra ──────────────────────────────────────────

class TestBatchConsistencyCheckExtra:
    def _asm(self, n=3, offset=0):
        return {
            "fragment_ids": list(range(offset, offset + n)),
            "expected_ids": list(range(offset, offset + n)),
            "positions": [(i * 20, 0) for i in range(n)],
            "sizes": [(15, 15)] * n,
            "canvas_w": 300,
            "canvas_h": 100,
        }

    def test_five_assemblies(self):
        result = batch_consistency_check([self._asm(3, i * 3) for i in range(5)])
        assert len(result) == 5

    def test_all_consistent(self):
        result = batch_consistency_check([self._asm(3, i * 3) for i in range(3)])
        assert all(r.is_consistent for r in result)

    def test_mixed_consistent_and_not(self):
        asm_good = self._asm(3, 0)
        asm_bad = self._asm(3, 0)
        asm_bad["fragment_ids"] = [0, 0, 2]  # duplicate
        result = batch_consistency_check([asm_good, asm_bad])
        assert result[0].is_consistent is True
        assert result[1].is_consistent is False

    def test_returns_list(self):
        result = batch_consistency_check([self._asm()])
        assert isinstance(result, list)

    def test_each_is_consistency_report(self):
        result = batch_consistency_check([self._asm(2), self._asm(3)])
        assert all(isinstance(r, ConsistencyReport) for r in result)

    def test_single_assembly(self):
        result = batch_consistency_check([self._asm(4)])
        assert len(result) == 1
        assert result[0].is_consistent is True
