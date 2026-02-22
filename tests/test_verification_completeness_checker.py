"""Tests for puzzle_reconstruction/verification/completeness_checker.py"""
import pytest
import numpy as np

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_full_mask(h=50, w=50):
    return np.full((h, w), 255, dtype=np.uint8)


def make_partial_mask(h=50, w=50, row_end=25):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:row_end, :] = 255
    return mask


def make_empty_mask(h=50, w=50):
    return np.zeros((h, w), dtype=np.uint8)


# ─── CompletenessReport ───────────────────────────────────────────────────────

class TestCompletenessReport:
    def test_basic_creation(self):
        r = CompletenessReport(
            fragment_coverage=0.8,
            spatial_coverage=0.7,
            total_score=0.75,
        )
        assert r.fragment_coverage == pytest.approx(0.8)
        assert r.spatial_coverage == pytest.approx(0.7)
        assert r.total_score == pytest.approx(0.75)

    def test_defaults(self):
        r = CompletenessReport(
            fragment_coverage=1.0,
            spatial_coverage=1.0,
            total_score=1.0,
        )
        assert r.n_placed == 0
        assert r.n_total == 0
        assert r.missing_ids == []
        assert r.params == {}

    def test_fragment_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=1.5, spatial_coverage=0.5,
                               total_score=0.5)

    def test_spatial_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5, spatial_coverage=-0.1,
                               total_score=0.5)

    def test_total_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=1.2)

    def test_is_complete_at_one(self):
        r = CompletenessReport(fragment_coverage=1.0, spatial_coverage=1.0,
                               total_score=1.0)
        assert r.is_complete(threshold=1.0) is True

    def test_is_complete_below_threshold(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5)
        assert r.is_complete(threshold=0.9) is False

    def test_is_complete_above_threshold(self):
        r = CompletenessReport(fragment_coverage=0.9, spatial_coverage=0.8,
                               total_score=0.85)
        assert r.is_complete(threshold=0.8) is True

    def test_boundary_values(self):
        r = CompletenessReport(fragment_coverage=0.0, spatial_coverage=0.0,
                               total_score=0.0)
        assert r.total_score == pytest.approx(0.0)

    def test_missing_ids_stored(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5, missing_ids=[2, 4, 6])
        assert r.missing_ids == [2, 4, 6]

    def test_n_placed_n_total_stored(self):
        r = CompletenessReport(fragment_coverage=0.5, spatial_coverage=0.5,
                               total_score=0.5, n_placed=3, n_total=6)
        assert r.n_placed == 3
        assert r.n_total == 6


# ─── check_fragment_coverage ─────────────────────────────────────────────────

class TestCheckFragmentCoverage:
    def test_all_placed(self):
        cov = check_fragment_coverage([0, 1, 2], [0, 1, 2])
        assert cov == pytest.approx(1.0)

    def test_none_placed(self):
        cov = check_fragment_coverage([], [0, 1, 2])
        assert cov == pytest.approx(0.0)

    def test_partial_placement(self):
        cov = check_fragment_coverage([0, 1], [0, 1, 2, 3])
        assert cov == pytest.approx(0.5)

    def test_empty_all_ids_returns_one(self):
        cov = check_fragment_coverage([], [])
        assert cov == pytest.approx(1.0)

    def test_extra_placed_ids_raises(self):
        with pytest.raises(ValueError):
            check_fragment_coverage([0, 1, 99], [0, 1, 2])

    def test_duplicate_placed_ids(self):
        """Duplicates in placed_ids count once (set)."""
        cov = check_fragment_coverage([0, 0, 1], [0, 1, 2])
        assert cov == pytest.approx(2.0 / 3.0)

    def test_range_zero_to_one(self):
        cov = check_fragment_coverage([1, 3], [0, 1, 2, 3, 4])
        assert 0.0 <= cov <= 1.0


# ─── find_missing_fragments ───────────────────────────────────────────────────

class TestFindMissingFragments:
    def test_all_placed_returns_empty(self):
        missing = find_missing_fragments([0, 1, 2], [0, 1, 2])
        assert missing == []

    def test_none_placed_returns_all(self):
        missing = find_missing_fragments([], [0, 1, 2])
        assert missing == [0, 1, 2]

    def test_partial_placement(self):
        missing = find_missing_fragments([1, 3], [0, 1, 2, 3])
        assert missing == [0, 2]

    def test_result_sorted(self):
        missing = find_missing_fragments([2], [0, 1, 2, 3])
        assert missing == sorted(missing)

    def test_empty_all_ids(self):
        missing = find_missing_fragments([], [])
        assert missing == []

    def test_duplicates_in_placed(self):
        missing = find_missing_fragments([0, 0, 1], [0, 1, 2])
        assert 2 in missing
        assert 0 not in missing


# ─── check_spatial_coverage ───────────────────────────────────────────────────

class TestCheckSpatialCoverage:
    def test_full_coverage(self):
        masks = [make_full_mask(50, 50)]
        cov = check_spatial_coverage(masks, (50, 50))
        assert cov == pytest.approx(1.0)

    def test_empty_masks_zero_coverage(self):
        cov = check_spatial_coverage([], (50, 50))
        assert cov == pytest.approx(0.0)

    def test_half_coverage(self):
        masks = [make_partial_mask(50, 50, row_end=25)]
        cov = check_spatial_coverage(masks, (50, 50))
        assert cov == pytest.approx(0.5)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            check_spatial_coverage([], (0, 50))

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            check_spatial_coverage([], (50, 0))

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError):
            check_spatial_coverage([], (-1, 50))

    def test_no_coverage_mask(self):
        masks = [make_empty_mask(50, 50)]
        cov = check_spatial_coverage(masks, (50, 50))
        assert cov == pytest.approx(0.0)

    def test_two_masks_union(self):
        m1 = make_partial_mask(50, 50, row_end=25)
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[25:, :] = 255
        cov = check_spatial_coverage([m1, m2], (50, 50))
        assert cov == pytest.approx(1.0)

    def test_coverage_in_range(self):
        masks = [make_partial_mask(50, 50, row_end=10)]
        cov = check_spatial_coverage(masks, (50, 50))
        assert 0.0 <= cov <= 1.0

    def test_smaller_mask_than_target(self):
        """Mask smaller than target → partial coverage."""
        mask = make_full_mask(25, 25)
        cov = check_spatial_coverage([mask], (50, 50))
        assert cov == pytest.approx(0.25)


# ─── find_uncovered_regions ───────────────────────────────────────────────────

class TestFindUncoveredRegions:
    def test_full_coverage_no_uncovered(self):
        masks = [make_full_mask(20, 20)]
        uncov = find_uncovered_regions(masks, (20, 20))
        assert (uncov == 0).all()

    def test_no_masks_all_uncovered(self):
        uncov = find_uncovered_regions([], (10, 10))
        assert (uncov == 255).all()

    def test_output_shape(self):
        masks = [make_partial_mask(30, 30, row_end=15)]
        uncov = find_uncovered_regions(masks, (30, 30))
        assert uncov.shape == (30, 30)

    def test_output_dtype_uint8(self):
        uncov = find_uncovered_regions([], (10, 10))
        assert uncov.dtype == np.uint8

    def test_output_binary(self):
        masks = [make_partial_mask(20, 20, row_end=10)]
        uncov = find_uncovered_regions(masks, (20, 20))
        assert set(np.unique(uncov)).issubset({0, 255})

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            find_uncovered_regions([], (0, 10))

    def test_partial_coverage(self):
        mask = make_partial_mask(20, 20, row_end=10)
        uncov = find_uncovered_regions([mask], (20, 20))
        # Bottom half (rows 10:20) should be uncovered
        assert (uncov[10:, :] == 255).all()
        assert (uncov[:10, :] == 0).all()


# ─── completeness_score ───────────────────────────────────────────────────────

class TestCompletenessScore:
    def test_full_placement(self):
        score = completeness_score(10, 10, pixel_coverage=1.0)
        assert score == pytest.approx(1.0)

    def test_zero_placement(self):
        score = completeness_score(0, 10, pixel_coverage=0.0)
        assert score == pytest.approx(0.0)

    def test_partial_placement(self):
        score = completeness_score(5, 10, pixel_coverage=0.5)
        assert score == pytest.approx(0.5)

    def test_n_total_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(0, 0)

    def test_n_total_negative_raises(self):
        with pytest.raises(ValueError):
            completeness_score(0, -1)

    def test_n_placed_negative_raises(self):
        with pytest.raises(ValueError):
            completeness_score(-1, 10)

    def test_n_placed_gt_n_total_raises(self):
        with pytest.raises(ValueError):
            completeness_score(11, 10)

    def test_pixel_coverage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, pixel_coverage=1.5)

    def test_negative_w_count_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_count=-0.1)

    def test_negative_w_pixel_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_pixel=-0.5)

    def test_both_weights_zero_raises(self):
        with pytest.raises(ValueError):
            completeness_score(5, 10, w_count=0.0, w_pixel=0.0)

    def test_score_in_range(self):
        score = completeness_score(3, 7, pixel_coverage=0.6)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        """Count-only weighting."""
        score = completeness_score(5, 10, pixel_coverage=0.0,
                                   w_count=1.0, w_pixel=0.0)
        assert score == pytest.approx(0.5)


# ─── generate_completeness_report ────────────────────────────────────────────

class TestGenerateCompletenessReport:
    def test_returns_completeness_report(self):
        report = generate_completeness_report([0, 1], [0, 1, 2])
        assert isinstance(report, CompletenessReport)

    def test_full_placement(self):
        report = generate_completeness_report([0, 1, 2], [0, 1, 2])
        assert report.fragment_coverage == pytest.approx(1.0)

    def test_no_placement(self):
        report = generate_completeness_report([], [0, 1, 2])
        assert report.fragment_coverage == pytest.approx(0.0)

    def test_missing_ids_in_report(self):
        report = generate_completeness_report([0], [0, 1, 2])
        assert 1 in report.missing_ids
        assert 2 in report.missing_ids

    def test_n_placed_n_total(self):
        report = generate_completeness_report([0, 1], [0, 1, 2])
        assert report.n_placed == 2
        assert report.n_total == 3

    def test_with_masks_and_target(self):
        masks = [make_full_mask(30, 30)]
        report = generate_completeness_report(
            [0], [0, 1], masks=masks, target_shape=(30, 30)
        )
        assert isinstance(report, CompletenessReport)

    def test_empty_all_ids(self):
        report = generate_completeness_report([], [])
        assert report.total_score == pytest.approx(1.0)

    def test_score_in_range(self):
        report = generate_completeness_report([0, 1], [0, 1, 2, 3])
        assert 0.0 <= report.total_score <= 1.0


# ─── batch_check_coverage ─────────────────────────────────────────────────────

class TestBatchCheckCoverage:
    def test_returns_list(self):
        results = batch_check_coverage([[0], [0, 1], [0, 1, 2]], [0, 1, 2])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_values_in_range(self):
        results = batch_check_coverage([[0], [0, 1]], [0, 1, 2])
        for r in results:
            assert 0.0 <= r <= 1.0

    def test_empty_placed_sets(self):
        results = batch_check_coverage([[], []], [0, 1, 2])
        assert all(r == pytest.approx(0.0) for r in results)

    def test_full_placement(self):
        results = batch_check_coverage([[0, 1, 2]], [0, 1, 2])
        assert results[0] == pytest.approx(1.0)

    def test_empty_placed_sets_list(self):
        results = batch_check_coverage([], [0, 1, 2])
        assert results == []

    def test_coverage_increases_with_more_placed(self):
        results = batch_check_coverage([[0], [0, 1], [0, 1, 2]], [0, 1, 2])
        assert results[0] <= results[1] <= results[2]
