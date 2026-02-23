"""Extra tests for puzzle_reconstruction.verification.completeness_checker."""
from __future__ import annotations
import numpy as np
import pytest

from puzzle_reconstruction.verification.completeness_checker import (
    CompletenessReport,
    batch_check_coverage,
    check_fragment_coverage,
    check_spatial_coverage,
    completeness_score,
    find_missing_fragments,
    find_uncovered_regions,
    generate_completeness_report,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _full(h=32, w=32):
    return np.full((h, w), 255, dtype=np.uint8)


def _empty(h=32, w=32):
    return np.zeros((h, w), dtype=np.uint8)


def _half(h=32, w=32):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h // 2, :] = 255
    return mask


# ─── TestCompletenessReportExtra ──────────────────────────────────────────────

class TestCompletenessReportExtra:
    def test_fields(self):
        r = CompletenessReport(fragment_coverage=0.9, spatial_coverage=0.8,
                               total_score=0.85, n_placed=9, n_total=10)
        assert r.fragment_coverage == pytest.approx(0.9)
        assert r.spatial_coverage == pytest.approx(0.8)
        assert r.total_score == pytest.approx(0.85)

    def test_missing_ids_default(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.missing_ids == []

    def test_params_default(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.params == {}

    def test_fragment_coverage_neg_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=-0.01,
                               spatial_coverage=0.5, total_score=0.5)

    def test_fragment_coverage_above_one_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=1.01,
                               spatial_coverage=0.5, total_score=0.5)

    def test_spatial_coverage_neg_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5,
                               spatial_coverage=-0.01, total_score=0.5)

    def test_total_score_above_one_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5,
                               spatial_coverage=0.5, total_score=1.5)

    def test_is_complete_true(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.is_complete() is True

    def test_is_complete_false(self):
        r = CompletenessReport(fragment_coverage=0.5,
                               spatial_coverage=0.5, total_score=0.5)
        assert r.is_complete(threshold=1.0) is False

    def test_is_complete_custom_threshold(self):
        r = CompletenessReport(fragment_coverage=0.8,
                               spatial_coverage=0.8, total_score=0.8)
        assert r.is_complete(threshold=0.7) is True

    def test_missing_ids_stored(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5, missing_ids=[1, 4, 7])
        assert r.missing_ids == [1, 4, 7]


# ─── TestCheckFragmentCoverageExtra ──────────────────────────────────────────

class TestCheckFragmentCoverageExtra:
    def test_full(self):
        assert check_fragment_coverage([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_zero(self):
        assert check_fragment_coverage([], [0, 1, 2]) == pytest.approx(0.0)

    def test_partial(self):
        assert check_fragment_coverage([0], [0, 1, 2, 3]) == pytest.approx(0.25)

    def test_empty_all_returns_one(self):
        assert check_fragment_coverage([], []) == pytest.approx(1.0)

    def test_extra_id_raises(self):
        with pytest.raises(ValueError):
            check_fragment_coverage([99], [0, 1, 2])

    def test_in_unit_interval(self):
        cov = check_fragment_coverage([0, 2, 4], [0, 1, 2, 3, 4])
        assert 0.0 <= cov <= 1.0

    def test_duplicates_deduplicated(self):
        cov = check_fragment_coverage([0, 0, 1], [0, 1, 2])
        assert cov == pytest.approx(2.0 / 3.0)


# ─── TestFindMissingFragmentsExtra ───────────────────────────────────────────

class TestFindMissingFragmentsExtra:
    def test_none_missing(self):
        assert find_missing_fragments([0, 1, 2], [0, 1, 2]) == []

    def test_all_missing(self):
        assert find_missing_fragments([], [0, 1, 2]) == [0, 1, 2]

    def test_partial(self):
        assert find_missing_fragments([0, 2], [0, 1, 2, 3]) == [1, 3]

    def test_sorted(self):
        result = find_missing_fragments([4], [1, 2, 3, 4, 5])
        assert result == sorted(result)

    def test_empty_all(self):
        assert find_missing_fragments([], []) == []

    def test_extra_placed_ignored(self):
        result = find_missing_fragments([0, 1, 99], [0, 1, 2])
        assert 2 in result

    def test_single_missing(self):
        assert find_missing_fragments([0, 2], [0, 1, 2]) == [1]


# ─── TestCheckSpatialCoverageExtra ───────────────────────────────────────────

class TestCheckSpatialCoverageExtra:
    def test_empty_masks_zero(self):
        assert check_spatial_coverage([], (32, 32)) == pytest.approx(0.0)

    def test_full_mask_one(self):
        assert check_spatial_coverage([_full()], (32, 32)) == pytest.approx(1.0)

    def test_empty_mask_zero(self):
        assert check_spatial_coverage([_empty()], (32, 32)) == pytest.approx(0.0)

    def test_half_mask(self):
        assert check_spatial_coverage([_half()], (32, 32)) == pytest.approx(0.5)

    def test_two_halves_full(self):
        top = np.zeros((32, 32), dtype=np.uint8)
        top[:16, :] = 255
        bot = np.zeros((32, 32), dtype=np.uint8)
        bot[16:, :] = 255
        assert check_spatial_coverage([top, bot], (32, 32)) == pytest.approx(1.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            check_spatial_coverage([], (0, 32))

    def test_in_unit_interval(self):
        cov = check_spatial_coverage([_half()], (32, 32))
        assert 0.0 <= cov <= 1.0

    def test_big_mask_clamped(self):
        big = np.full((64, 64), 255, dtype=np.uint8)
        assert check_spatial_coverage([big], (32, 32)) == pytest.approx(1.0)


# ─── TestFindUncoveredRegionsExtra ───────────────────────────────────────────

class TestFindUncoveredRegionsExtra:
    def test_dtype_uint8(self):
        assert find_uncovered_regions([_full()], (32, 32)).dtype == np.uint8

    def test_shape_matches(self):
        assert find_uncovered_regions([_full()], (24, 48)).shape == (24, 48)

    def test_fully_covered_all_zero(self):
        assert np.all(find_uncovered_regions([_full()], (32, 32)) == 0)

    def test_no_masks_all_255(self):
        assert np.all(find_uncovered_regions([], (16, 16)) == 255)

    def test_partial_both_values(self):
        result = find_uncovered_regions([_half()], (32, 32))
        assert np.any(result == 0)
        assert np.any(result == 255)

    def test_values_only_0_or_255(self):
        result = find_uncovered_regions([_half()], (32, 32))
        assert set(np.unique(result)).issubset({0, 255})

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            find_uncovered_regions([], (-1, 32))


# ─── TestCompletenessScoreExtra ──────────────────────────────────────────────

class TestCompletenessScoreExtra:
    def test_full_one(self):
        assert completeness_score(10, 10, 1.0) == pytest.approx(1.0)

    def test_zero_placed(self):
        assert completeness_score(0, 10, 0.0) == pytest.approx(0.0)

    def test_in_unit_interval(self):
        s = completeness_score(7, 10, 0.6)
        assert 0.0 <= s <= 1.0

    def test_n_total_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(0, 0)

    def test_negative_placed_raises(self):
        with pytest.raises(ValueError):
            completeness_score(-1, 10)

    def test_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            completeness_score(11, 10)

    def test_invalid_pixel_coverage_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, pixel_coverage=1.5)

    def test_both_weights_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_count=0.0, w_pixel=0.0)

    def test_only_count_weight(self):
        s = completeness_score(5, 10, w_count=1.0, w_pixel=0.0)
        assert s == pytest.approx(0.5)

    def test_only_pixel_weight(self):
        s = completeness_score(5, 10, pixel_coverage=0.8,
                               w_count=0.0, w_pixel=1.0)
        assert s == pytest.approx(0.8)


# ─── TestGenerateCompletenessReportExtra ─────────────────────────────────────

class TestGenerateCompletenessReportExtra:
    def test_returns_report(self):
        assert isinstance(generate_completeness_report([0], [0, 1]),
                          CompletenessReport)

    def test_n_placed(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert r.n_placed == 2

    def test_n_total(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert r.n_total == 3

    def test_missing_ids(self):
        r = generate_completeness_report([0], [0, 1, 2])
        assert r.missing_ids == [1, 2]

    def test_full_placement(self):
        r = generate_completeness_report([0, 1, 2], [0, 1, 2])
        assert r.fragment_coverage == pytest.approx(1.0)

    def test_with_masks(self):
        r = generate_completeness_report([0], [0, 1],
                                         masks=[_full()], target_shape=(32, 32))
        assert 0.0 <= r.spatial_coverage <= 1.0

    def test_params_stored(self):
        r = generate_completeness_report([0], [0], w_count=0.7, w_pixel=0.3)
        assert r.params.get("w_count") == pytest.approx(0.7)

    def test_empty_all_ids(self):
        r = generate_completeness_report([], [])
        assert r.total_score == pytest.approx(1.0)


# ─── TestBatchCheckCoverageExtra ─────────────────────────────────────────────

class TestBatchCheckCoverageExtra:
    def test_returns_list(self):
        assert isinstance(batch_check_coverage([[0], [1]], [0, 1, 2]), list)

    def test_length(self):
        assert len(batch_check_coverage([[0], [1], [2]], [0, 1, 2])) == 3

    def test_empty_returns_empty(self):
        assert batch_check_coverage([], [0, 1]) == []

    def test_values_in_unit(self):
        for v in batch_check_coverage([[0, 1], [0]], [0, 1, 2]):
            assert 0.0 <= v <= 1.0

    def test_full_coverage_one(self):
        result = batch_check_coverage([[0, 1, 2]], [0, 1, 2])
        assert result[0] == pytest.approx(1.0)

    def test_empty_placed_zero(self):
        result = batch_check_coverage([[]], [0, 1, 2])
        assert result[0] == pytest.approx(0.0)

    def test_multiple_groups(self):
        result = batch_check_coverage([[0], [0, 1], [0, 1, 2]], [0, 1, 2])
        assert len(result) == 3
        assert result[0] < result[1] < result[2]
