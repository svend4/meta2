"""Extra tests for puzzle_reconstruction/verification/completeness_checker.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.completeness_checker import (
    CompletenessReport,
    check_fragment_coverage,
    find_missing_fragments,
    check_spatial_coverage,
    find_uncovered_regions,
    completeness_score,
    generate_completeness_report,
    batch_check_coverage,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _full(h=30, w=30):
    return np.full((h, w), 255, dtype=np.uint8)


def _empty(h=30, w=30):
    return np.zeros((h, w), dtype=np.uint8)


def _half(h=30, w=30):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h // 2, :] = 255
    return mask


# ─── CompletenessReport (extra) ───────────────────────────────────────────────

class TestCompletenessReportExtra:
    def test_fragment_coverage_zero_valid(self):
        r = CompletenessReport(fragment_coverage=0.0, spatial_coverage=0.5,
                               total_score=0.25)
        assert r.fragment_coverage == pytest.approx(0.0)

    def test_spatial_coverage_one_valid(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=1.0,
                               total_score=0.75)
        assert r.spatial_coverage == pytest.approx(1.0)

    def test_total_score_zero_valid(self):
        r = CompletenessReport(fragment_coverage=0.0, spatial_coverage=0.0,
                               total_score=0.0)
        assert r.total_score == pytest.approx(0.0)

    def test_is_complete_at_threshold_exact(self):
        r = CompletenessReport(fragment_coverage=0.8, spatial_coverage=0.8,
                               total_score=0.8)
        assert r.is_complete(threshold=0.8) is True

    def test_is_complete_just_below(self):
        r = CompletenessReport(fragment_coverage=0.79, spatial_coverage=0.79,
                               total_score=0.79)
        assert r.is_complete(threshold=0.8) is False

    def test_default_missing_ids_empty(self):
        r = CompletenessReport(fragment_coverage=1.0, spatial_coverage=1.0,
                               total_score=1.0)
        assert r.missing_ids == []

    def test_n_placed_n_total_default_zero(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5)
        assert r.n_placed == 0
        assert r.n_total == 0

    def test_params_stored(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5, params={"w_count": 0.6})
        assert r.params["w_count"] == pytest.approx(0.6)

    def test_fragment_coverage_neg_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=-0.1, spatial_coverage=0.5,
                               total_score=0.5)

    def test_total_score_neg_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=-0.1)


# ─── check_fragment_coverage (extra) ─────────────────────────────────────────

class TestCheckFragmentCoverageExtra:
    def test_one_of_five_placed(self):
        cov = check_fragment_coverage([0], [0, 1, 2, 3, 4])
        assert cov == pytest.approx(0.2)

    def test_three_of_three(self):
        cov = check_fragment_coverage([0, 1, 2], [0, 1, 2])
        assert cov == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(check_fragment_coverage([0], [0, 1]), float)

    def test_result_in_range(self):
        cov = check_fragment_coverage([2, 3], [0, 1, 2, 3])
        assert 0.0 <= cov <= 1.0

    def test_large_set(self):
        all_ids = list(range(100))
        placed = list(range(0, 100, 2))  # every other
        cov = check_fragment_coverage(placed, all_ids)
        assert cov == pytest.approx(0.5)

    def test_single_fragment_placed(self):
        cov = check_fragment_coverage([0], [0])
        assert cov == pytest.approx(1.0)


# ─── find_missing_fragments (extra) ──────────────────────────────────────────

class TestFindMissingFragmentsExtra:
    def test_returns_list(self):
        assert isinstance(find_missing_fragments([0], [0, 1, 2]), list)

    def test_missing_ids_correct(self):
        missing = find_missing_fragments([1, 3], [0, 1, 2, 3, 4])
        assert set(missing) == {0, 2, 4}

    def test_all_placed_empty_missing(self):
        assert find_missing_fragments([0, 1, 2], [0, 1, 2]) == []

    def test_none_placed_all_missing(self):
        missing = find_missing_fragments([], [0, 1, 2, 3])
        assert len(missing) == 4

    def test_result_no_duplicates(self):
        missing = find_missing_fragments([0, 0], [0, 1, 2])
        assert len(missing) == len(set(missing))

    def test_large_set_missing_count(self):
        all_ids = list(range(20))
        placed = list(range(10))
        missing = find_missing_fragments(placed, all_ids)
        assert len(missing) == 10


# ─── check_spatial_coverage (extra) ──────────────────────────────────────────

class TestCheckSpatialCoverageExtra:
    def test_returns_float(self):
        assert isinstance(check_spatial_coverage([_full()], (30, 30)), float)

    def test_two_half_masks_full_coverage(self):
        m1 = np.zeros((20, 20), dtype=np.uint8)
        m1[:10, :] = 255
        m2 = np.zeros((20, 20), dtype=np.uint8)
        m2[10:, :] = 255
        cov = check_spatial_coverage([m1, m2], (20, 20))
        assert cov == pytest.approx(1.0)

    def test_overlapping_masks_no_double_count(self):
        m = _full(20, 20)
        # Two identical full masks should still give coverage 1.0
        cov = check_spatial_coverage([m, m], (20, 20))
        assert cov == pytest.approx(1.0)

    def test_single_partial_mask(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[:20, :] = 255
        cov = check_spatial_coverage([mask], (40, 40))
        assert cov == pytest.approx(0.5)

    def test_coverage_nonneg(self):
        cov = check_spatial_coverage([_empty()], (30, 30))
        assert cov >= 0.0

    def test_coverage_le_one(self):
        cov = check_spatial_coverage([_full(), _full()], (30, 30))
        assert cov <= 1.0


# ─── find_uncovered_regions (extra) ──────────────────────────────────────────

class TestFindUncoveredRegionsExtra:
    def test_returns_ndarray(self):
        result = find_uncovered_regions([], (10, 10))
        assert isinstance(result, np.ndarray)

    def test_two_full_masks_all_covered(self):
        uncov = find_uncovered_regions([_full(20, 20), _full(20, 20)], (20, 20))
        assert (uncov == 0).all()

    def test_partial_masks_some_uncovered(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:10, :] = 255
        uncov = find_uncovered_regions([mask], (20, 20))
        assert uncov[15, 10] == 255  # bottom half uncovered
        assert uncov[5, 10] == 0     # top half covered

    def test_shape_matches_target(self):
        uncov = find_uncovered_regions([], (15, 25))
        assert uncov.shape == (15, 25)

    def test_binary_values_only(self):
        mask = _half()
        uncov = find_uncovered_regions([mask], (30, 30))
        assert set(np.unique(uncov)).issubset({0, 255})


# ─── completeness_score (extra) ───────────────────────────────────────────────

class TestCompletenessScoreExtra:
    def test_returns_float(self):
        s = completeness_score(5, 10, pixel_coverage=0.5)
        assert isinstance(s, float)

    def test_result_in_range(self):
        s = completeness_score(3, 8, pixel_coverage=0.4)
        assert 0.0 <= s <= 1.0

    def test_full_count_full_pixel_one(self):
        s = completeness_score(7, 7, pixel_coverage=1.0)
        assert s == pytest.approx(1.0)

    def test_zero_count_zero_pixel_zero(self):
        s = completeness_score(0, 5, pixel_coverage=0.0)
        assert s == pytest.approx(0.0)

    def test_pixel_only_weighting(self):
        s = completeness_score(0, 10, pixel_coverage=0.8,
                               w_count=0.0, w_pixel=1.0)
        assert s == pytest.approx(0.8)

    def test_count_only_weighting(self):
        s = completeness_score(6, 10, pixel_coverage=0.0,
                               w_count=1.0, w_pixel=0.0)
        assert s == pytest.approx(0.6)

    def test_equal_weights_average(self):
        s = completeness_score(5, 10, pixel_coverage=0.8,
                               w_count=1.0, w_pixel=1.0)
        assert s == pytest.approx((0.5 + 0.8) / 2, abs=1e-5)


# ─── generate_completeness_report (extra) ─────────────────────────────────────

class TestGenerateCompletenessReportExtra:
    def test_report_fragment_coverage_correct(self):
        report = generate_completeness_report([0, 1, 2], [0, 1, 2, 3])
        assert report.fragment_coverage == pytest.approx(0.75)

    def test_report_n_placed(self):
        report = generate_completeness_report([0, 1], [0, 1, 2, 3])
        assert report.n_placed == 2

    def test_report_n_total(self):
        report = generate_completeness_report([0], [0, 1, 2, 3, 4])
        assert report.n_total == 5

    def test_missing_ids_correct(self):
        report = generate_completeness_report([0, 2], [0, 1, 2, 3])
        assert 1 in report.missing_ids
        assert 3 in report.missing_ids

    def test_full_placement_score_one(self):
        report = generate_completeness_report([0, 1, 2], [0, 1, 2])
        assert report.total_score == pytest.approx(1.0)

    def test_total_score_in_range(self):
        report = generate_completeness_report([0], [0, 1, 2])
        assert 0.0 <= report.total_score <= 1.0


# ─── batch_check_coverage (extra) ─────────────────────────────────────────────

class TestBatchCheckCoverageExtra:
    def test_returns_float_list(self):
        results = batch_check_coverage([[0, 1]], [0, 1, 2])
        assert all(isinstance(r, float) for r in results)

    def test_all_full_returns_ones(self):
        results = batch_check_coverage([[0, 1, 2], [0, 1, 2]], [0, 1, 2])
        assert all(r == pytest.approx(1.0) for r in results)

    def test_empty_batch_returns_empty(self):
        results = batch_check_coverage([], [0, 1, 2])
        assert results == []

    def test_single_entry_correct(self):
        results = batch_check_coverage([[0, 1]], [0, 1, 2, 3])
        assert results[0] == pytest.approx(0.5)

    def test_length_matches_input(self):
        placed_sets = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        results = batch_check_coverage(placed_sets, [0, 1, 2, 3])
        assert len(results) == 4

    def test_coverage_nondecreasing_when_adding(self):
        placed_sets = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        results = batch_check_coverage(placed_sets, [0, 1, 2, 3])
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]
