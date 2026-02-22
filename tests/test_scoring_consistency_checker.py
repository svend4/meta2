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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _issue(code="DUPLICATE_ID", severity="error") -> ConsistencyIssue:
    return ConsistencyIssue(code=code, description="test", severity=severity)


def _report(n_err=0, n_warn=0) -> ConsistencyReport:
    return ConsistencyReport(issues=[], is_consistent=(n_err == 0),
                             n_errors=n_err, n_warnings=n_warn)


def _good_assembly(n: int = 4, canvas: int = 200) -> dict:
    ids = list(range(n))
    positions = [(i * 40, 0) for i in range(n)]
    sizes = [(40, 40)] * n
    return dict(
        fragment_ids=ids, expected_ids=ids,
        positions=positions, sizes=sizes,
        canvas_w=canvas, canvas_h=canvas,
    )


# ─── TestConsistencyIssue ─────────────────────────────────────────────────────

class TestConsistencyIssue:
    def test_basic(self):
        iss = _issue()
        assert iss.code == "DUPLICATE_ID"
        assert iss.severity == "error"

    def test_warning_ok(self):
        iss = _issue(severity="warning")
        assert iss.severity == "warning"

    def test_info_ok(self):
        iss = _issue(severity="info")
        assert iss.severity == "info"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="X", description="y", severity="critical")

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            ConsistencyIssue(code="", description="y")

    def test_fragment_ids_default_empty(self):
        iss = _issue()
        assert iss.fragment_ids == []

    def test_fragment_ids_stored(self):
        iss = ConsistencyIssue(code="X", description="y",
                               fragment_ids=[1, 2, 3])
        assert iss.fragment_ids == [1, 2, 3]


# ─── TestConsistencyReport ────────────────────────────────────────────────────

class TestConsistencyReport:
    def test_basic(self):
        r = _report(0, 1)
        assert r.is_consistent is True
        assert r.n_warnings == 1

    def test_len_matches_issues(self):
        r = ConsistencyReport(issues=[_issue(), _issue(severity="warning")],
                              is_consistent=False, n_errors=1, n_warnings=1)
        assert len(r) == 2

    def test_len_empty(self):
        assert len(_report()) == 0

    def test_n_errors_neg_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=-1, n_warnings=0)

    def test_n_warnings_neg_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=0, n_warnings=-1)

    def test_checked_pairs_neg_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport(issues=[], is_consistent=True,
                              n_errors=0, n_warnings=0, checked_pairs=-1)

    def test_checked_pairs_zero_ok(self):
        r = _report()
        assert r.checked_pairs == 0

    def test_consistent_no_errors(self):
        r = ConsistencyReport(issues=[_issue(severity="warning")],
                              is_consistent=True, n_errors=0, n_warnings=1)
        assert r.is_consistent is True


# ─── TestCheckUniqueIds ───────────────────────────────────────────────────────

class TestCheckUniqueIds:
    def test_all_unique_empty_list(self):
        assert check_unique_ids([0, 1, 2, 3]) == []

    def test_duplicate_returns_issue(self):
        issues = check_unique_ids([0, 1, 0, 2])
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_ID"

    def test_duplicate_severity_error(self):
        issues = check_unique_ids([0, 0])
        assert issues[0].severity == "error"

    def test_duplicate_id_in_fragment_ids(self):
        issues = check_unique_ids([1, 2, 2, 3])
        assert 2 in issues[0].fragment_ids

    def test_multiple_duplicates(self):
        issues = check_unique_ids([0, 0, 1, 1])
        assert len(issues) == 1  # один общий issue
        assert len(issues[0].fragment_ids) == 2

    def test_empty_list_ok(self):
        assert check_unique_ids([]) == []

    def test_single_element_ok(self):
        assert check_unique_ids([5]) == []


# ─── TestCheckAllPresent ──────────────────────────────────────────────────────

class TestCheckAllPresent:
    def test_all_present_no_issues(self):
        assert check_all_present([0, 1, 2], [0, 1, 2]) == []

    def test_missing_fragment_issue(self):
        issues = check_all_present([0, 2], [0, 1, 2])
        codes = {iss.code for iss in issues}
        assert "MISSING_FRAGMENT" in codes

    def test_extra_fragment_issue(self):
        issues = check_all_present([0, 1, 2, 3], [0, 1, 2])
        codes = {iss.code for iss in issues}
        assert "EXTRA_FRAGMENT" in codes

    def test_missing_is_error(self):
        issues = check_all_present([0], [0, 1])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert missing[0].severity == "error"

    def test_extra_is_warning(self):
        issues = check_all_present([0, 1, 99], [0, 1])
        extra = [i for i in issues if i.code == "EXTRA_FRAGMENT"]
        assert extra[0].severity == "warning"

    def test_missing_id_stored(self):
        issues = check_all_present([0], [0, 5])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert 5 in missing[0].fragment_ids

    def test_both_missing_and_extra(self):
        issues = check_all_present([0, 99], [0, 1])
        codes = {i.code for i in issues}
        assert codes == {"MISSING_FRAGMENT", "EXTRA_FRAGMENT"}

    def test_empty_expected_no_missing(self):
        issues = check_all_present([0, 1], [])
        missing = [i for i in issues if i.code == "MISSING_FRAGMENT"]
        assert missing == []


# ─── TestCheckCanvasBounds ────────────────────────────────────────────────────

class TestCheckCanvasBounds:
    def test_all_inside_no_issues(self):
        positions = [(0, 0), (50, 50)]
        sizes = [(40, 40), (30, 30)]
        assert check_canvas_bounds(positions, sizes, 200, 200) == []

    def test_fragment_out_right(self):
        positions = [(180, 0)]
        sizes = [(40, 40)]
        issues = check_canvas_bounds(positions, sizes, 200, 200)
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_fragment_out_bottom(self):
        positions = [(0, 180)]
        sizes = [(40, 40)]
        issues = check_canvas_bounds(positions, sizes, 200, 200)
        assert len(issues) == 1

    def test_fragment_neg_x(self):
        positions = [(-1, 0)]
        sizes = [(40, 40)]
        issues = check_canvas_bounds(positions, sizes, 200, 200)
        assert len(issues) == 1

    def test_fragment_neg_y(self):
        positions = [(0, -1)]
        sizes = [(40, 40)]
        issues = check_canvas_bounds(positions, sizes, 200, 200)
        assert len(issues) == 1

    def test_out_of_bounds_is_error(self):
        issues = check_canvas_bounds([(180, 0)], [(40, 40)], 200, 200)
        assert issues[0].severity == "error"

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([], [], 0, 100)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([], [], 100, 0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            check_canvas_bounds([(0, 0), (10, 10)], [(10, 10)], 100, 100)

    def test_empty_lists_ok(self):
        assert check_canvas_bounds([], [], 100, 100) == []


# ─── TestCheckScoreThreshold ──────────────────────────────────────────────────

class TestCheckScoreThreshold:
    def test_all_above_threshold_empty(self):
        scores = {(0, 1): 0.8, (1, 2): 0.9}
        assert check_score_threshold(scores, min_score=0.7) == []

    def test_below_threshold_issue(self):
        scores = {(0, 1): 0.3}
        issues = check_score_threshold(scores, min_score=0.5)
        assert len(issues) == 1
        assert issues[0].code == "LOW_SCORE"

    def test_low_score_is_warning(self):
        scores = {(0, 1): 0.1}
        assert check_score_threshold(scores, 0.5)[0].severity == "warning"

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            check_score_threshold({}, min_score=-0.1)

    def test_empty_scores_ok(self):
        assert check_score_threshold({}) == []

    def test_boundary_exact_ok(self):
        scores = {(0, 1): 0.5}
        assert check_score_threshold(scores, min_score=0.5) == []

    def test_multiple_low_pairs(self):
        scores = {(0, 1): 0.2, (1, 2): 0.1, (0, 2): 0.9}
        issues = check_score_threshold(scores, min_score=0.5)
        assert len(issues) == 2


# ─── TestCheckGapUniformity ───────────────────────────────────────────────────

class TestCheckGapUniformity:
    def test_uniform_gaps_ok(self):
        assert check_gap_uniformity([5.0, 5.0, 5.0]) == []

    def test_nonuniform_gaps_issue(self):
        issues = check_gap_uniformity([1.0, 50.0], max_std=5.0)
        assert len(issues) == 1
        assert issues[0].code == "NONUNIFORM_GAP"

    def test_nonuniform_is_warning(self):
        issues = check_gap_uniformity([1.0, 100.0], max_std=5.0)
        assert issues[0].severity == "warning"

    def test_single_gap_ok(self):
        assert check_gap_uniformity([10.0]) == []

    def test_empty_gaps_ok(self):
        assert check_gap_uniformity([]) == []

    def test_max_std_neg_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity([1.0, 2.0], max_std=-1.0)

    def test_max_std_zero_ok(self):
        assert check_gap_uniformity([5.0, 5.0], max_std=0.0) == []

    def test_max_std_zero_nonuniform(self):
        issues = check_gap_uniformity([5.0, 6.0], max_std=0.0)
        assert len(issues) == 1


# ─── TestRunConsistencyCheck ──────────────────────────────────────────────────

class TestRunConsistencyCheck:
    def _run(self, **kw):
        asm = _good_assembly(**kw)
        return run_consistency_check(**asm)

    def test_returns_consistency_report(self):
        assert isinstance(self._run(), ConsistencyReport)

    def test_consistent_assembly(self):
        r = self._run()
        assert r.is_consistent is True

    def test_no_issues_for_good_assembly(self):
        r = self._run()
        assert r.n_errors == 0

    def test_duplicate_detected(self):
        asm = _good_assembly()
        asm["fragment_ids"] = [0, 0, 1, 2]
        r = run_consistency_check(**asm)
        assert r.n_errors > 0
        assert r.is_consistent is False

    def test_missing_detected(self):
        asm = _good_assembly()
        asm["fragment_ids"] = [0, 1]  # missing 2, 3
        r = run_consistency_check(**asm)
        assert r.n_errors > 0

    def test_out_of_bounds_detected(self):
        asm = _good_assembly(canvas=50)  # canvas too small for 4 × 40px
        r = run_consistency_check(**asm)
        assert r.n_errors > 0

    def test_with_pair_scores(self):
        asm = _good_assembly()
        asm["pair_scores"] = {(0, 1): 0.2}
        asm["min_score"] = 0.5
        r = run_consistency_check(**asm)
        assert r.n_warnings > 0

    def test_checked_pairs_counted(self):
        asm = _good_assembly()
        asm["pair_scores"] = {(0, 1): 0.9, (1, 2): 0.8}
        r = run_consistency_check(**asm)
        assert r.checked_pairs == 2

    def test_no_pair_scores_checked_pairs_zero(self):
        r = self._run()
        assert r.checked_pairs == 0


# ─── TestBatchConsistencyCheck ────────────────────────────────────────────────

class TestBatchConsistencyCheck:
    def test_returns_list(self):
        assemblies = [_good_assembly() for _ in range(3)]
        assert isinstance(batch_consistency_check(assemblies), list)

    def test_length_matches(self):
        assemblies = [_good_assembly() for _ in range(4)]
        assert len(batch_consistency_check(assemblies)) == 4

    def test_empty_list(self):
        assert batch_consistency_check([]) == []

    def test_all_consistency_reports(self):
        assemblies = [_good_assembly() for _ in range(2)]
        for r in batch_consistency_check(assemblies):
            assert isinstance(r, ConsistencyReport)

    def test_consistent_assemblies(self):
        assemblies = [_good_assembly() for _ in range(3)]
        for r in batch_consistency_check(assemblies):
            assert r.is_consistent is True

    def test_bad_assembly_detected(self):
        good = _good_assembly()
        bad = _good_assembly()
        bad["fragment_ids"] = [0, 0, 1, 2]
        results = batch_consistency_check([good, bad])
        assert results[0].is_consistent is True
        assert results[1].is_consistent is False
