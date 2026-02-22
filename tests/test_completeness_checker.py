"""Tests for puzzle_reconstruction.verification.completeness_checker."""
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

def _full_mask(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _empty_mask(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _half_mask(h: int = 32, w: int = 32) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h // 2, :] = 255
    return mask


# ─── CompletenessReport ──────────────────────────────────────────────────────

class TestCompletenessReport:
    def test_fields_stored(self):
        r = CompletenessReport(
            fragment_coverage=0.8,
            spatial_coverage=0.7,
            total_score=0.75,
            n_placed=8,
            n_total=10,
        )
        assert r.fragment_coverage == pytest.approx(0.8)
        assert r.spatial_coverage == pytest.approx(0.7)
        assert r.total_score == pytest.approx(0.75)
        assert r.n_placed == 8
        assert r.n_total == 10

    def test_default_missing_ids_empty(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.missing_ids == []

    def test_default_params_empty(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.params == {}

    def test_fragment_coverage_below_zero_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=-0.1,
                               spatial_coverage=0.5, total_score=0.5)

    def test_fragment_coverage_above_one_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=1.1,
                               spatial_coverage=0.5, total_score=0.5)

    def test_spatial_coverage_invalid_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5,
                               spatial_coverage=-0.1, total_score=0.5)

    def test_total_score_invalid_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5,
                               spatial_coverage=0.5, total_score=1.5)

    def test_is_complete_true(self):
        r = CompletenessReport(fragment_coverage=1.0,
                               spatial_coverage=1.0, total_score=1.0)
        assert r.is_complete() is True

    def test_is_complete_false(self):
        r = CompletenessReport(fragment_coverage=0.8,
                               spatial_coverage=0.8, total_score=0.8)
        assert r.is_complete(threshold=1.0) is False

    def test_is_complete_custom_threshold(self):
        r = CompletenessReport(fragment_coverage=0.9,
                               spatial_coverage=0.9, total_score=0.9)
        assert r.is_complete(threshold=0.8) is True

    def test_missing_ids_stored(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5, missing_ids=[3, 7])
        assert r.missing_ids == [3, 7]


# ─── check_fragment_coverage ────────────────────────────────────────────────

class TestCheckFragmentCoverage:
    def test_full_coverage(self):
        assert check_fragment_coverage([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_zero_coverage(self):
        assert check_fragment_coverage([], [0, 1, 2]) == pytest.approx(0.0)

    def test_partial_coverage(self):
        assert check_fragment_coverage([0, 1], [0, 1, 2, 3]) == pytest.approx(0.5)

    def test_empty_all_ids_returns_one(self):
        assert check_fragment_coverage([], []) == pytest.approx(1.0)

    def test_extra_id_raises(self):
        with pytest.raises(ValueError):
            check_fragment_coverage([0, 5], [0, 1, 2])

    def test_result_in_unit_interval(self):
        cov = check_fragment_coverage([0, 1, 3], [0, 1, 2, 3, 4])
        assert 0.0 <= cov <= 1.0

    def test_duplicates_in_placed_deduplicated(self):
        # Duplicate placed IDs should count as one
        cov = check_fragment_coverage([0, 0, 1], [0, 1, 2])
        assert cov == pytest.approx(2.0 / 3.0)


# ─── find_missing_fragments ──────────────────────────────────────────────────

class TestFindMissingFragments:
    def test_none_missing(self):
        assert find_missing_fragments([0, 1, 2], [0, 1, 2]) == []

    def test_all_missing(self):
        assert find_missing_fragments([], [0, 1, 2]) == [0, 1, 2]

    def test_partial_missing(self):
        assert find_missing_fragments([0, 2], [0, 1, 2, 3]) == [1, 3]

    def test_returns_sorted_list(self):
        result = find_missing_fragments([4], [1, 2, 3, 4, 5])
        assert result == sorted(result)

    def test_empty_all_returns_empty(self):
        assert find_missing_fragments([], []) == []

    def test_extra_placed_ids_ignored(self):
        # IDs not in all_ids are irrelevant
        result = find_missing_fragments([0, 1, 99], [0, 1, 2])
        assert 2 in result


# ─── check_spatial_coverage ──────────────────────────────────────────────────

class TestCheckSpatialCoverage:
    def test_empty_masks_returns_zero(self):
        assert check_spatial_coverage([], (32, 32)) == pytest.approx(0.0)

    def test_full_mask_returns_one(self):
        cov = check_spatial_coverage([_full_mask()], (32, 32))
        assert cov == pytest.approx(1.0)

    def test_no_coverage_returns_zero(self):
        cov = check_spatial_coverage([_empty_mask()], (32, 32))
        assert cov == pytest.approx(0.0)

    def test_half_mask_approx_half(self):
        cov = check_spatial_coverage([_half_mask()], (32, 32))
        assert cov == pytest.approx(0.5)

    def test_union_of_two_halves(self):
        mask_top = np.zeros((32, 32), dtype=np.uint8)
        mask_top[:16, :] = 255
        mask_bot = np.zeros((32, 32), dtype=np.uint8)
        mask_bot[16:, :] = 255
        cov = check_spatial_coverage([mask_top, mask_bot], (32, 32))
        assert cov == pytest.approx(1.0)

    def test_invalid_target_shape_raises(self):
        with pytest.raises(ValueError):
            check_spatial_coverage([], (0, 32))
        with pytest.raises(ValueError):
            check_spatial_coverage([], (32, 0))

    def test_result_in_unit_interval(self):
        cov = check_spatial_coverage([_half_mask()], (32, 32))
        assert 0.0 <= cov <= 1.0

    def test_mask_larger_than_target_clamped(self):
        big_mask = np.full((64, 64), 255, dtype=np.uint8)
        cov = check_spatial_coverage([big_mask], (32, 32))
        assert cov == pytest.approx(1.0)


# ─── find_uncovered_regions ──────────────────────────────────────────────────

class TestFindUncoveredRegions:
    def test_returns_uint8(self):
        result = find_uncovered_regions([_full_mask()], (32, 32))
        assert result.dtype == np.uint8

    def test_shape_matches_target(self):
        result = find_uncovered_regions([_full_mask()], (24, 32))
        assert result.shape == (24, 32)

    def test_fully_covered_all_zero(self):
        result = find_uncovered_regions([_full_mask()], (32, 32))
        assert np.all(result == 0)

    def test_no_masks_all_255(self):
        result = find_uncovered_regions([], (16, 16))
        assert np.all(result == 255)

    def test_partial_coverage(self):
        result = find_uncovered_regions([_half_mask()], (32, 32))
        # Bottom half should be uncovered (255)
        assert np.any(result == 255)
        assert np.any(result == 0)

    def test_values_only_0_or_255(self):
        result = find_uncovered_regions([_half_mask()], (32, 32))
        assert set(np.unique(result)).issubset({0, 255})

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            find_uncovered_regions([], (-1, 32))


# ─── completeness_score ──────────────────────────────────────────────────────

class TestCompletenessScore:
    def test_full_score_one(self):
        assert completeness_score(10, 10, 1.0) == pytest.approx(1.0)

    def test_zero_placed_zero_score(self):
        assert completeness_score(0, 10, 0.0) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        s = completeness_score(7, 10, 0.6)
        assert 0.0 <= s <= 1.0

    def test_n_total_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(0, 0)

    def test_negative_n_placed_raises(self):
        with pytest.raises(ValueError):
            completeness_score(-1, 10)

    def test_n_placed_exceeds_total_raises(self):
        with pytest.raises(ValueError):
            completeness_score(11, 10)

    def test_invalid_pixel_coverage_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, pixel_coverage=1.5)

    def test_both_weights_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_count=0.0, w_pixel=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_count=-0.5)

    def test_only_count_weight(self):
        s = completeness_score(5, 10, w_count=1.0, w_pixel=0.0)
        assert s == pytest.approx(0.5)

    def test_only_pixel_weight(self):
        s = completeness_score(5, 10, pixel_coverage=0.8,
                               w_count=0.0, w_pixel=1.0)
        assert s == pytest.approx(0.8)


# ─── generate_completeness_report ───────────────────────────────────────────

class TestGenerateCompletenessReport:
    def test_returns_report(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert isinstance(r, CompletenessReport)

    def test_n_placed_correct(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert r.n_placed == 2

    def test_n_total_correct(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert r.n_total == 3

    def test_missing_ids_correct(self):
        r = generate_completeness_report([0, 1], [0, 1, 2])
        assert r.missing_ids == [2]

    def test_full_placement(self):
        r = generate_completeness_report([0, 1, 2], [0, 1, 2])
        assert r.fragment_coverage == pytest.approx(1.0)
        assert r.missing_ids == []

    def test_with_masks(self):
        r = generate_completeness_report(
            [0], [0, 1],
            masks=[_full_mask()],
            target_shape=(32, 32),
        )
        assert 0.0 <= r.spatial_coverage <= 1.0

    def test_params_stored(self):
        r = generate_completeness_report([0], [0], w_count=0.6, w_pixel=0.4)
        assert r.params.get("w_count") == pytest.approx(0.6)
        assert r.params.get("w_pixel") == pytest.approx(0.4)

    def test_empty_all_ids(self):
        r = generate_completeness_report([], [])
        assert r.total_score == pytest.approx(1.0)


# ─── batch_check_coverage ────────────────────────────────────────────────────

class TestBatchCheckCoverage:
    def test_returns_list(self):
        result = batch_check_coverage([[0, 1], [0]], [0, 1, 2])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        result = batch_check_coverage([[0], [1], [2]], [0, 1, 2])
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert batch_check_coverage([], [0, 1, 2]) == []

    def test_values_in_unit_interval(self):
        result = batch_check_coverage([[0, 1], [0]], [0, 1, 2])
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_full_coverage_is_one(self):
        result = batch_check_coverage([[0, 1, 2]], [0, 1, 2])
        assert result[0] == pytest.approx(1.0)

    def test_empty_placed_is_zero(self):
        result = batch_check_coverage([[]], [0, 1, 2])
        assert result[0] == pytest.approx(0.0)
