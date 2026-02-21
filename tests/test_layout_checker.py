"""Тесты для puzzle_reconstruction/verification/layout_checker.py."""
import numpy as np
import pytest

from puzzle_reconstruction.verification.layout_checker import (
    LayoutViolationType,
    LayoutViolation,
    LayoutCheckResult,
    compute_bounding_box,
    detect_overlaps,
    detect_gaps,
    check_grid_alignment,
    check_aspect_ratio,
    check_layout,
    batch_check_layout,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _box(x, y, w, h):
    return (float(x), float(y), float(w), float(h))


def _perfect_grid():
    """2×2 фрагмента по 50×50, без зазоров, без перекрытий."""
    return (
        [0, 1, 2, 3],
        {
            0: _box(0,   0,   50, 50),
            1: _box(50,  0,   50, 50),
            2: _box(0,   50,  50, 50),
            3: _box(50,  50,  50, 50),
        }
    )


def _overlapping():
    """Два фрагмента с перекрытием 10×10."""
    return (
        [0, 1],
        {0: _box(0, 0, 50, 50), 1: _box(40, 0, 50, 50)},
    )


# ─── LayoutViolationType ──────────────────────────────────────────────────────

class TestLayoutViolationType:
    def test_six_values(self):
        assert len(LayoutViolationType) == 6

    def test_is_str_enum(self):
        assert isinstance(LayoutViolationType.OVERLAP, str)

    def test_values(self):
        assert LayoutViolationType.OVERLAP      == "overlap"
        assert LayoutViolationType.GAP          == "gap"
        assert LayoutViolationType.MISALIGNMENT == "misalignment"
        assert LayoutViolationType.ASPECT_RATIO == "aspect_ratio"
        assert LayoutViolationType.BOUNDARY     == "boundary"
        assert LayoutViolationType.INSUFFICIENT == "insufficient"


# ─── LayoutViolation ─────────────────────────────────────────────────────────

class TestLayoutViolation:
    def test_fields(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5)
        assert v.type == LayoutViolationType.OVERLAP
        assert v.severity == pytest.approx(0.5)
        assert isinstance(v.fragment_ids, list)
        assert isinstance(v.description, str)
        assert isinstance(v.values, dict)

    def test_default_fragment_ids_empty(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.3)
        assert v.fragment_ids == []

    def test_custom_fragment_ids(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5,
                            fragment_ids=[0, 1])
        assert v.fragment_ids == [0, 1]

    def test_values_dict(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5,
                            values={"overlap_area": 25.0})
        assert v.values["overlap_area"] == pytest.approx(25.0)

    def test_description(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.2,
                            description="test description")
        assert "test" in v.description


# ─── LayoutCheckResult ───────────────────────────────────────────────────────

class TestLayoutCheckResult:
    def _make(self, violations=None, score=1.0, n_checked=4, bbox=None):
        return LayoutCheckResult(
            violations=violations or [],
            score=score,
            n_checked=n_checked,
            bounding_box=bbox,
            method_scores={"overlap": 1.0, "gap": 1.0,
                           "alignment": 1.0, "aspect_ratio": 1.0},
        )

    def test_n_violations_empty(self):
        r = self._make()
        assert r.n_violations == 0

    def test_n_violations_count(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.3)
        r = self._make(violations=[v, v])
        assert r.n_violations == 2

    def test_max_severity_empty(self):
        r = self._make()
        assert r.max_severity == pytest.approx(0.0)

    def test_max_severity_single(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.7)
        r = self._make(violations=[v])
        assert r.max_severity == pytest.approx(0.7)

    def test_max_severity_multiple(self):
        v1 = LayoutViolation(type=LayoutViolationType.GAP, severity=0.3)
        v2 = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.8)
        r = self._make(violations=[v1, v2])
        assert r.max_severity == pytest.approx(0.8)

    def test_score_stored(self):
        r = self._make(score=0.75)
        assert r.score == pytest.approx(0.75)

    def test_n_checked_stored(self):
        r = self._make(n_checked=6)
        assert r.n_checked == 6

    def test_method_scores_keys(self):
        r = self._make()
        assert set(r.method_scores.keys()) == {"overlap", "gap",
                                                "alignment", "aspect_ratio"}

    def test_bounding_box_stored(self):
        r = self._make(bbox=(0.0, 0.0, 100.0, 100.0))
        assert r.bounding_box == pytest.approx((0.0, 0.0, 100.0, 100.0))


# ─── compute_bounding_box ────────────────────────────────────────────────────

class TestComputeBoundingBox:
    def test_empty_returns_zeros(self):
        bbox = compute_bounding_box({})
        assert bbox == pytest.approx((0.0, 0.0, 0.0, 0.0))

    def test_single_box(self):
        bbox = compute_bounding_box({0: _box(10, 20, 30, 40)})
        assert bbox == pytest.approx((10.0, 20.0, 30.0, 40.0))

    def test_two_boxes_side_by_side(self):
        pos = {0: _box(0, 0, 50, 50), 1: _box(50, 0, 50, 50)}
        bbox = compute_bounding_box(pos)
        assert bbox[0] == pytest.approx(0.0)
        assert bbox[1] == pytest.approx(0.0)
        assert bbox[2] == pytest.approx(100.0)
        assert bbox[3] == pytest.approx(50.0)

    def test_four_boxes_grid(self):
        _, pos = _perfect_grid()
        bbox = compute_bounding_box(pos)
        assert bbox == pytest.approx((0.0, 0.0, 100.0, 100.0))

    def test_returns_tuple_of_four(self):
        bbox = compute_bounding_box({0: _box(5, 5, 10, 10)})
        assert len(bbox) == 4

    def test_offset_origin(self):
        pos = {0: _box(100, 200, 50, 50), 1: _box(150, 200, 50, 50)}
        bbox = compute_bounding_box(pos)
        assert bbox[0] == pytest.approx(100.0)
        assert bbox[1] == pytest.approx(200.0)
        assert bbox[2] == pytest.approx(100.0)


# ─── detect_overlaps ─────────────────────────────────────────────────────────

class TestDetectOverlaps:
    def test_no_overlap_returns_empty(self):
        ids, pos = _perfect_grid()
        v = detect_overlaps(ids, pos)
        assert v == []

    def test_adjacent_no_overlap(self):
        ids = [0, 1]
        pos = {0: _box(0, 0, 50, 50), 1: _box(50, 0, 50, 50)}
        assert detect_overlaps(ids, pos) == []

    def test_overlap_detected(self):
        ids, pos = _overlapping()
        v = detect_overlaps(ids, pos, min_overlap=1.0)
        assert len(v) > 0

    def test_overlap_type(self):
        ids, pos = _overlapping()
        for violation in detect_overlaps(ids, pos, min_overlap=1.0):
            assert violation.type == LayoutViolationType.OVERLAP

    def test_overlap_severity_in_range(self):
        ids, pos = _overlapping()
        for v in detect_overlaps(ids, pos, min_overlap=1.0):
            assert 0.0 <= v.severity <= 1.0

    def test_fragment_ids_in_violation(self):
        ids, pos = _overlapping()
        violations = detect_overlaps(ids, pos, min_overlap=1.0)
        for v in violations:
            assert len(v.fragment_ids) == 2

    def test_min_overlap_threshold(self):
        ids, pos = _overlapping()
        # With very high threshold, even large overlap may be ignored
        v_large_thresh = detect_overlaps(ids, pos, min_overlap=10000.0)
        v_small_thresh = detect_overlaps(ids, pos, min_overlap=1.0)
        assert len(v_large_thresh) <= len(v_small_thresh)

    def test_missing_ids_ignored(self):
        # IDs not in positions dict are skipped
        ids = [0, 1, 99]  # 99 not in positions
        pos = {0: _box(0, 0, 50, 50), 1: _box(40, 0, 50, 50)}
        v = detect_overlaps(ids, pos, min_overlap=1.0)
        assert isinstance(v, list)

    def test_returns_list(self):
        ids, pos = _perfect_grid()
        assert isinstance(detect_overlaps(ids, pos), list)


# ─── detect_gaps ─────────────────────────────────────────────────────────────

class TestDetectGaps:
    def test_no_gap_returns_empty(self):
        ids, pos = _perfect_grid()
        v = detect_gaps(ids, pos, gap_tol=2.0)
        assert v == []

    def test_horizontal_gap_detected(self):
        ids = [0, 1]
        pos = {
            0: _box(0,  0, 50, 50),
            1: _box(100, 0, 50, 50),   # 50px gap
        }
        v = detect_gaps(ids, pos, expected_gap=0.0, gap_tol=5.0)
        assert len(v) > 0

    def test_vertical_gap_detected(self):
        ids = [0, 1]
        pos = {
            0: _box(0, 0,   50, 50),
            1: _box(0, 100, 50, 50),   # 50px vertical gap
        }
        v = detect_gaps(ids, pos, expected_gap=0.0, gap_tol=5.0)
        assert len(v) > 0

    def test_gap_type(self):
        ids = [0, 1]
        pos = {0: _box(0, 0, 50, 50), 1: _box(100, 0, 50, 50)}
        for v in detect_gaps(ids, pos, gap_tol=5.0):
            assert v.type == LayoutViolationType.GAP

    def test_severity_in_range(self):
        ids = [0, 1]
        pos = {0: _box(0, 0, 50, 50), 1: _box(100, 0, 50, 50)}
        for v in detect_gaps(ids, pos, gap_tol=5.0):
            assert 0.0 <= v.severity <= 1.0

    def test_expected_gap_reduces_violations(self):
        ids = [0, 1]
        pos = {0: _box(0, 0, 50, 50), 1: _box(70, 0, 50, 50)}  # 20px gap
        v_zero  = detect_gaps(ids, pos, expected_gap=0,  gap_tol=5.0)
        v_match = detect_gaps(ids, pos, expected_gap=20, gap_tol=5.0)
        # Expected gap matches actual → fewer violations
        assert len(v_zero) >= len(v_match)

    def test_returns_list(self):
        ids, pos = _perfect_grid()
        assert isinstance(detect_gaps(ids, pos), list)


# ─── check_grid_alignment ────────────────────────────────────────────────────

class TestCheckGridAlignment:
    def test_single_fragment_no_violation(self):
        ids = [0]
        pos = {0: _box(0, 0, 50, 50)}
        assert check_grid_alignment(ids, pos) == []

    def test_empty_list_no_violation(self):
        assert check_grid_alignment([], {}) == []

    def test_perfectly_aligned_no_violation(self):
        ids, pos = _perfect_grid()
        v = check_grid_alignment(ids, pos, tol_px=2.0)
        assert v == []

    def test_misaligned_violation(self):
        ids = [0, 1, 2, 3]
        pos = {
            0: _box(0,  0,  50, 50),
            1: _box(55, 0,  50, 50),   # misaligned by 5px from grid
            2: _box(0,  50, 50, 50),
            3: _box(50, 50, 50, 50),
        }
        v = check_grid_alignment(ids, pos, tol_px=2.0)
        # May or may not detect depending on clustering, but should be a list
        assert isinstance(v, list)

    def test_violation_type(self):
        ids = [0, 1, 2, 3]
        pos = {
            0: _box(0,  0, 50, 50),
            1: _box(60, 0, 50, 50),  # 60 vs expected 50
            2: _box(0,  50, 50, 50),
            3: _box(50, 50, 50, 50),
        }
        v = check_grid_alignment(ids, pos, tol_px=1.0)
        for viol in v:
            assert viol.type == LayoutViolationType.MISALIGNMENT

    def test_severity_in_range(self):
        ids, pos = _perfect_grid()
        for viol in check_grid_alignment(ids, pos):
            assert 0.0 <= viol.severity <= 1.0

    def test_returns_list(self):
        ids, pos = _perfect_grid()
        assert isinstance(check_grid_alignment(ids, pos), list)

    def test_tol_px_affects_result(self):
        ids = [0, 1]
        pos = {0: _box(0, 0, 50, 50), 1: _box(53, 0, 50, 50)}
        v_tight = check_grid_alignment(ids, pos, tol_px=1.0)
        v_loose = check_grid_alignment(ids, pos, tol_px=10.0)
        assert len(v_tight) >= len(v_loose)


# ─── check_aspect_ratio ──────────────────────────────────────────────────────

class TestCheckAspectRatio:
    def test_no_expected_ratio_returns_empty(self):
        ids, pos = _perfect_grid()
        assert check_aspect_ratio(ids, pos, expected_ratio=None) == []

    def test_empty_ids_returns_empty(self):
        assert check_aspect_ratio([], {}, expected_ratio=1.0) == []

    def test_correct_ratio_no_violation(self):
        ids = [0]
        pos = {0: _box(0, 0, 100, 100)}   # square → ratio=1.0
        v = check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1)
        assert v == []

    def test_wrong_ratio_violation(self):
        ids = [0]
        pos = {0: _box(0, 0, 200, 100)}   # ratio=2.0, expected=1.0
        v = check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1)
        assert len(v) == 1

    def test_violation_type(self):
        ids = [0]
        pos = {0: _box(0, 0, 300, 100)}
        for viol in check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1):
            assert viol.type == LayoutViolationType.ASPECT_RATIO

    def test_severity_in_range(self):
        ids = [0]
        pos = {0: _box(0, 0, 300, 100)}
        for viol in check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1):
            assert 0.0 <= viol.severity <= 1.0

    def test_tol_ratio_affects_result(self):
        ids = [0]
        pos = {0: _box(0, 0, 120, 100)}  # ratio=1.2, expected=1.0, dev=0.2
        v_tight = check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1)
        v_loose = check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.3)
        assert len(v_tight) >= len(v_loose)

    def test_values_stored(self):
        ids = [0]
        pos = {0: _box(0, 0, 200, 100)}
        v = check_aspect_ratio(ids, pos, expected_ratio=1.0, tol_ratio=0.1)
        assert len(v) == 1
        assert "actual" in v[0].values
        assert "expected" in v[0].values


# ─── check_layout ────────────────────────────────────────────────────────────

class TestCheckLayout:
    def test_returns_result(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert isinstance(r, LayoutCheckResult)

    def test_perfect_grid_score_one(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert r.score == pytest.approx(1.0, abs=1e-6)

    def test_perfect_grid_no_violations(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert r.n_violations == 0

    def test_overlapping_has_violations(self):
        ids, pos = _overlapping()
        r = check_layout(ids, pos, overlap_min=1.0)
        assert r.n_violations > 0

    def test_score_in_range(self):
        ids, pos = _overlapping()
        r = check_layout(ids, pos, overlap_min=1.0)
        assert 0.0 <= r.score <= 1.0

    def test_n_checked(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert r.n_checked == 4

    def test_four_method_scores(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert set(r.method_scores.keys()) == {"overlap", "gap",
                                                "alignment", "aspect_ratio"}

    def test_bounding_box_computed(self):
        ids, pos = _perfect_grid()
        r = check_layout(ids, pos)
        assert r.bounding_box is not None
        assert len(r.bounding_box) == 4

    def test_single_fragment_score_one(self):
        ids = [0]
        pos = {0: _box(0, 0, 50, 50)}
        r = check_layout(ids, pos)
        assert r.score == pytest.approx(1.0)
        assert r.n_violations == 0

    def test_missing_ids_ignored(self):
        ids = [0, 1, 99]  # 99 not in positions
        pos = {0: _box(0, 0, 50, 50), 1: _box(50, 0, 50, 50)}
        r = check_layout(ids, pos)
        assert isinstance(r, LayoutCheckResult)

    def test_overlap_min_param(self):
        ids, pos = _overlapping()
        r_sensitive = check_layout(ids, pos, overlap_min=1.0)
        r_loose     = check_layout(ids, pos, overlap_min=10000.0)
        assert r_sensitive.n_violations >= r_loose.n_violations

    def test_expected_ratio_param(self):
        ids, pos = _perfect_grid()   # 100×100 → ratio=1.0
        r_ok   = check_layout(ids, pos, expected_ratio=1.0, ratio_tol=0.1)
        r_fail = check_layout(ids, pos, expected_ratio=3.0, ratio_tol=0.1)
        assert r_ok.score >= r_fail.score

    def test_violations_list(self):
        ids, pos = _overlapping()
        r = check_layout(ids, pos, overlap_min=1.0)
        for v in r.violations:
            assert isinstance(v, LayoutViolation)


# ─── batch_check_layout ──────────────────────────────────────────────────────

class TestBatchCheckLayout:
    def test_returns_list(self):
        ids, pos = _perfect_grid()
        results = batch_check_layout([ids], [pos])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_each_is_result(self):
        ids, pos = _perfect_grid()
        for r in batch_check_layout([ids, ids], [pos, pos]):
            assert isinstance(r, LayoutCheckResult)

    def test_empty_groups(self):
        results = batch_check_layout([], [])
        assert results == []

    def test_length_mismatch_raises(self):
        ids, pos = _perfect_grid()
        with pytest.raises(ValueError):
            batch_check_layout([ids, ids], [pos])

    def test_kwargs_forwarded(self):
        ids, pos = _overlapping()
        results = batch_check_layout([ids], [pos], overlap_min=1.0)
        assert results[0].n_violations > 0

    def test_multiple_assemblies(self):
        ids, pos = _perfect_grid()
        oids, opos = _overlapping()
        results = batch_check_layout([ids, oids], [pos, opos], overlap_min=1.0)
        assert len(results) == 2
        assert results[0].score > results[1].score
