"""Тесты для puzzle_reconstruction.scoring.consistency_checker."""
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


# ─── TestConsistencyIssue ─────────────────────────────────────────────────────

class TestConsistencyIssue:
    def test_basic_creation(self):
        iss = ConsistencyIssue(code="DUPLICATE_ID", description="dup")
        assert iss.code == "DUPLICATE_ID"
        assert iss.severity == "error"

    def test_valid_severities(self):
        for sev in ("error", "warning", "info"):
            iss = ConsistencyIssue(code="X", description="d", severity=sev)
            assert iss.severity == sev

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="X", description="d", severity="critical")

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="", description="d")

    def test_fragment_ids_default_empty(self):
        iss = ConsistencyIssue(code="X", description="d")
        assert iss.fragment_ids == []

    def test_fragment_ids_stored(self):
        iss = ConsistencyIssue(code="X", description="d", fragment_ids=[1, 2])
        assert iss.fragment_ids == [1, 2]


# ─── TestConsistencyReport ────────────────────────────────────────────────────

class TestConsistencyReport:
    def _make(self, n_err=0, n_warn=0):
        return ConsistencyReport(
            issues=[],
            is_consistent=(n_err == 0),
            n_errors=n_err,
            n_warnings=n_warn,
        )

    def test_len_zero(self):
        rpt = self._make()
        assert len(rpt) == 0

    def test_len_nonzero(self):
        iss = ConsistencyIssue(code="X", description="d")
        rpt = ConsistencyReport(
            issues=[iss], is_consistent=False, n_errors=1, n_warnings=0
        )
        assert len(rpt) == 1

    def test_is_consistent_no_errors(self):
        rpt = self._make(n_err=0)
        assert rpt.is_consistent is True

    def test_negative_n_errors_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(
                issues=[], is_consistent=True, n_errors=-1, n_warnings=0
            )

    def test_negative_n_warnings_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(
                issues=[], is_consistent=True, n_errors=0, n_warnings=-1
            )

    def test_negative_checked_pairs_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(
                issues=[], is_consistent=True,
                n_errors=0, n_warnings=0, checked_pairs=-1
            )

    def test_checked_pairs_default_zero(self):
        rpt = self._make()
        assert rpt.checked_pairs == 0


# ─── TestCheckUniqueIds ───────────────────────────────────────────────────────

class TestCheckUniqueIds:
    def test_all_unique_no_issues(self):
        assert check_unique_ids([0, 1, 2, 3]) == []

    def test_duplicate_detected(self):
        issues = check_unique_ids([0, 1, 1, 2])
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_ID"
        assert 1 in issues[0].fragment_ids

    def test_duplicate_is_error(self):
        issues = check_unique_ids([0, 0])
        assert issues[0].severity == "error"

    def test_multiple_duplicates(self):
        issues = check_unique_ids([0, 0, 1, 1])
        assert len(issues) == 1  # один issue на все дубликаты

    def test_empty_list_no_issues(self):
        assert check_unique_ids([]) == []

    def test_single_element(self):
        assert check_unique_ids([5]) == []


# ─── TestCheckAllPresent ──────────────────────────────────────────────────────

class TestCheckAllPresent:
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

    def test_missing_is_error(self):
        issues = check_all_present([0], [0, 1])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert missing[0].severity == "error"

    def test_extra_is_warning(self):
        issues = check_all_present([0, 1, 2], [0, 1])
        extra = [i for i in issues if i.code == "EXTRA_FRAGMENT"]
        assert extra[0].severity == "warning"

    def test_empty_lists_no_issues(self):
        assert check_all_present([], []) == []


# ─── TestCheckCanvasBounds ────────────────────────────────────────────────────

class TestCheckCanvasBounds:
    def _args(self):
        return dict(
            positions=[(0, 0), (10, 10)],
            sizes=[(10, 10), (10, 10)],
            canvas_w=100,
            canvas_h=100,
        )

    def test_all_in_bounds_no_issues(self):
        issues = check_canvas_bounds(
            [(0, 0), (5, 5)], [(10, 10), (10, 10)],
            canvas_w=30, canvas_h=30
        )
        assert issues == []

    def test_out_of_bounds_x(self):
        issues = check_canvas_bounds(
            [(25, 0)], [(10, 10)], canvas_w=30, canvas_h=30
        )
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_out_of_bounds_y(self):
        issues = check_canvas_bounds(
            [(0, 25)], [(10, 10)], canvas_w=30, canvas_h=30
        )
        assert len(issues) == 1

    def test_negative_x_detected(self):
        issues = check_canvas_bounds(
            [(-1, 0)], [(10, 10)], canvas_w=30, canvas_h=30
        )
        assert len(issues) == 1

    def test_out_of_bounds_is_error(self):
        issues = check_canvas_bounds(
            [(90, 0)], [(20, 20)], canvas_w=100, canvas_h=100
        )
        assert issues[0].severity == "error"

    def test_canvas_w_zero_raises(self):
        args = self._args()
        with pytest.raises(ValueError):
            check_canvas_bounds(**{**args, "canvas_w": 0})

    def test_canvas_h_zero_raises(self):
        args = self._args()
        with pytest.raises(ValueError):
            check_canvas_bounds(**{**args, "canvas_h": 0})

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds(
                [(0, 0)], [(10, 10), (5, 5)], canvas_w=100, canvas_h=100
            )


# ─── TestCheckScoreThreshold ──────────────────────────────────────────────────

class TestCheckScoreThreshold:
    def test_all_above_no_issues(self):
        assert check_score_threshold({(0, 1): 0.9, (1, 2): 0.8},
                                     min_score=0.5) == []

    def test_low_score_detected(self):
        issues = check_score_threshold({(0, 1): 0.3}, min_score=0.5)
        assert len(issues) == 1
        assert issues[0].code == "LOW_SCORE"

    def test_low_score_is_warning(self):
        issues = check_score_threshold({(0, 1): 0.1}, min_score=0.5)
        assert issues[0].severity == "warning"

    def test_fragment_ids_in_issue(self):
        issues = check_score_threshold({(2, 5): 0.1}, min_score=0.5)
        assert 2 in issues[0].fragment_ids
        assert 5 in issues[0].fragment_ids

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            check_score_threshold({}, min_score=-0.1)

    def test_empty_scores_no_issues(self):
        assert check_score_threshold({}, min_score=0.5) == []

    def test_exact_threshold_no_issue(self):
        # Оценка == порог → не нарушение
        issues = check_score_threshold({(0, 1): 0.5}, min_score=0.5)
        assert issues == []


# ─── TestCheckGapUniformity ───────────────────────────────────────────────────

class TestCheckGapUniformity:
    def test_uniform_gaps_no_issue(self):
        issues = check_gap_uniformity([4.0, 4.0, 4.0], max_std=1.0)
        assert issues == []

    def test_nonuniform_issue(self):
        issues = check_gap_uniformity([1.0, 100.0], max_std=1.0)
        assert len(issues) == 1
        assert issues[0].code == "NONUNIFORM_GAP"

    def test_nonuniform_is_warning(self):
        issues = check_gap_uniformity([1.0, 100.0], max_std=1.0)
        assert issues[0].severity == "warning"

    def test_less_than_two_no_issue(self):
        assert check_gap_uniformity([5.0], max_std=1.0) == []
        assert check_gap_uniformity([], max_std=1.0) == []

    def test_negative_max_std_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity([1.0, 2.0], max_std=-1.0)

    def test_std_at_boundary_no_issue(self):
        # std == max_std не должно давать нарушение
        issues = check_gap_uniformity([0.0, 2.0], max_std=1.0)
        assert issues == []


# ─── TestRunConsistencyCheck ──────────────────────────────────────────────────

class TestRunConsistencyCheck:
    def _valid(self):
        return dict(
            fragment_ids=[0, 1, 2],
            expected_ids=[0, 1, 2],
            positions=[(0, 0), (20, 0), (40, 0)],
            sizes=[(15, 15)] * 3,
            canvas_w=100,
            canvas_h=100,
        )

    def test_returns_consistency_report(self):
        rpt = run_consistency_check(**self._valid())
        assert isinstance(rpt, ConsistencyReport)

    def test_valid_assembly_consistent(self):
        rpt = run_consistency_check(**self._valid())
        assert rpt.is_consistent is True
        assert rpt.n_errors == 0

    def test_duplicate_id_makes_inconsistent(self):
        args = self._valid()
        args["fragment_ids"] = [0, 0, 2]
        rpt = run_consistency_check(**args)
        assert rpt.is_consistent is False

    def test_missing_fragment_raises_error(self):
        args = self._valid()
        args["fragment_ids"] = [0, 2]
        rpt = run_consistency_check(**args)
        assert rpt.n_errors > 0

    def test_out_of_bounds_raises_error(self):
        args = self._valid()
        args["positions"] = [(0, 0), (200, 0), (40, 0)]
        rpt = run_consistency_check(**args)
        assert rpt.n_errors > 0

    def test_low_score_adds_warning(self):
        args = self._valid()
        args["pair_scores"] = {(0, 1): 0.1}
        rpt = run_consistency_check(**args, min_score=0.5)
        assert rpt.n_warnings > 0

    def test_checked_pairs_counted(self):
        args = self._valid()
        args["pair_scores"] = {(0, 1): 0.9, (1, 2): 0.8}
        rpt = run_consistency_check(**args)
        assert rpt.checked_pairs == 2

    def test_no_pair_scores_zero_checked(self):
        rpt = run_consistency_check(**self._valid())
        assert rpt.checked_pairs == 0


# ─── TestBatchConsistencyCheck ────────────────────────────────────────────────

class TestBatchConsistencyCheck:
    def _asm(self, n=3):
        return {
            "fragment_ids": list(range(n)),
            "expected_ids": list(range(n)),
            "positions": [(i * 20, 0) for i in range(n)],
            "sizes": [(15, 15)] * n,
            "canvas_w": 200,
            "canvas_h": 100,
        }

    def test_returns_list(self):
        result = batch_consistency_check([self._asm()])
        assert isinstance(result, list)

    def test_correct_length(self):
        result = batch_consistency_check([self._asm(3), self._asm(4)])
        assert len(result) == 2

    def test_empty_list(self):
        assert batch_consistency_check([]) == []

    def test_each_consistency_report(self):
        result = batch_consistency_check([self._asm(), self._asm(2)])
        assert all(isinstance(r, ConsistencyReport) for r in result)

    def test_valid_assemblies_consistent(self):
        result = batch_consistency_check([self._asm(3), self._asm(2)])
        assert all(r.is_consistent for r in result)
