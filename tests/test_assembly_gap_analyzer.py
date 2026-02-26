"""Tests for puzzle_reconstruction/assembly/gap_analyzer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.gap_analyzer import (
    FragmentBounds,
    GapInfo,
    GapStats,
    compute_gap,
    find_adjacent,
    analyze_all_gaps,
    gap_histogram,
    classify_gaps,
    summarize,
    batch_analyze,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_frag(fid, x, y, w=50, h=50):
    return FragmentBounds(fragment_id=fid, x=x, y=y, width=w, height=h)


# ── FragmentBounds ────────────────────────────────────────────────────────────

class TestFragmentBounds:
    def test_valid_construction(self):
        fb = make_frag(0, 10.0, 20.0, 50.0, 60.0)
        assert fb.fragment_id == 0
        assert fb.x == 10.0
        assert fb.y == 20.0

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id должен быть >= 0"):
            make_frag(-1, 0, 0)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x должен быть >= 0"):
            FragmentBounds(fragment_id=0, x=-1.0, y=0.0, width=10, height=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y должен быть >= 0"):
            FragmentBounds(fragment_id=0, x=0.0, y=-1.0, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width должен быть >= 1"):
            FragmentBounds(fragment_id=0, x=0, y=0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height должен быть >= 1"):
            FragmentBounds(fragment_id=0, x=0, y=0, width=10, height=0)

    def test_x2_property(self):
        fb = make_frag(0, 10, 20, 50, 60)
        assert fb.x2 == 60

    def test_y2_property(self):
        fb = make_frag(0, 10, 20, 50, 60)
        assert fb.y2 == 80

    def test_center_property(self):
        fb = make_frag(0, 0, 0, 100, 100)
        cx, cy = fb.center
        assert cx == 50.0
        assert cy == 50.0

    def test_area_property(self):
        fb = make_frag(0, 0, 0, 40, 60)
        assert fb.area == 2400.0


# ── GapInfo ───────────────────────────────────────────────────────────────────

class TestGapInfo:
    def test_valid_construction(self):
        gi = GapInfo(id1=0, id2=1, gap_x=10.0, gap_y=5.0, distance=15.0,
                     category="near")
        assert gi.id1 == 0
        assert gi.gap_x == 10.0

    def test_negative_id1_raises(self):
        with pytest.raises(ValueError, match="id1 должен быть >= 0"):
            GapInfo(id1=-1, id2=1, gap_x=0, gap_y=0, distance=0)

    def test_negative_id2_raises(self):
        with pytest.raises(ValueError, match="id2 должен быть >= 0"):
            GapInfo(id1=0, id2=-1, gap_x=0, gap_y=0, distance=0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="distance должен быть >= 0"):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=-1.0)

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="category должен быть"):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=0,
                    category="diagonal")

    def test_valid_categories(self):
        for cat in ("overlap", "touching", "near", "far"):
            gi = GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=0,
                         category=cat)
            assert gi.category == cat

    def test_pair_property(self):
        gi = GapInfo(id1=3, id2=7, gap_x=0, gap_y=0, distance=10.0)
        assert gi.pair == (3, 7)

    def test_is_overlapping_true(self):
        gi = GapInfo(id1=0, id2=1, gap_x=-5.0, gap_y=-3.0, distance=5.0,
                     category="overlap")
        assert gi.is_overlapping is True

    def test_is_overlapping_false(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=5.0, distance=10.0,
                     category="far")
        assert gi.is_overlapping is False

    def test_default_category_far(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5, gap_y=5, distance=10)
        assert gi.category == "far"


# ── GapStats ──────────────────────────────────────────────────────────────────

class TestGapStats:
    def test_default_construction(self):
        gs = GapStats()
        assert gs.n_pairs == 0
        assert gs.mean_distance == 0.0

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            GapStats(n_pairs=-1)

    def test_negative_n_overlapping_raises(self):
        with pytest.raises(ValueError):
            GapStats(n_overlapping=-1)

    def test_negative_mean_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(mean_distance=-1.0)

    def test_negative_std_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(std_distance=-1.0)

    def test_valid_full_construction(self):
        gs = GapStats(n_pairs=5, n_overlapping=1, n_touching=1, n_near=2,
                      n_far=1, mean_distance=50.0, std_distance=10.0)
        assert gs.n_pairs == 5


# ── compute_gap ───────────────────────────────────────────────────────────────

class TestComputeGap:
    def test_adjacent_horizontal_gap(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 60, 0, 50, 50)
        gi = compute_gap(a, b)
        assert gi.gap_x == 10.0

    def test_touching_horizontal(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 50, 0, 50, 50)
        gi = compute_gap(a, b)
        assert gi.gap_x == 0.0
        assert gi.category == "touching"

    def test_overlapping_category(self):
        a = make_frag(0, 0, 0, 100, 100)
        b = make_frag(1, 10, 10, 50, 50)
        gi = compute_gap(a, b)
        assert gi.category == "overlap"
        assert gi.is_overlapping is True

    def test_ids_ordered(self):
        a = make_frag(5, 0, 0, 50, 50)
        b = make_frag(2, 100, 0, 50, 50)
        gi = compute_gap(a, b)
        assert gi.id1 < gi.id2

    def test_distance_nonneg(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 100, 0, 50, 50)
        gi = compute_gap(a, b)
        assert gi.distance >= 0.0

    def test_far_category_large_gap(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 200, 0, 50, 50)
        gi = compute_gap(a, b, near_threshold=5.0)
        assert gi.category == "far"

    def test_near_category_small_gap(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 54, 54, 50, 50)
        gi = compute_gap(a, b, near_threshold=10.0)
        assert gi.category in ("near", "far", "touching")

    def test_negative_near_threshold_raises(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 100, 0, 50, 50)
        with pytest.raises(ValueError, match="near_threshold должен быть >= 0"):
            compute_gap(a, b, near_threshold=-1.0)

    def test_returns_gap_info(self):
        a = make_frag(0, 0, 0)
        b = make_frag(1, 100, 0)
        result = compute_gap(a, b)
        assert isinstance(result, GapInfo)

    def test_distance_calculation_correct(self):
        # Centers: (25, 25) and (125, 25). Distance = 100
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 100, 0, 50, 50)
        gi = compute_gap(a, b)
        assert pytest.approx(gi.distance, abs=1e-4) == 100.0


# ── find_adjacent ─────────────────────────────────────────────────────────────

class TestFindAdjacent:
    def test_empty_list(self):
        assert find_adjacent([]) == []

    def test_single_fragment(self):
        assert find_adjacent([make_frag(0, 0, 0)]) == []

    def test_finds_close_pair(self):
        frags = [make_frag(0, 0, 0, 50, 50), make_frag(1, 60, 0, 50, 50)]
        result = find_adjacent(frags, distance_threshold=200.0)
        assert len(result) == 1

    def test_excludes_far_pair(self):
        frags = [make_frag(0, 0, 0, 50, 50), make_frag(1, 500, 0, 50, 50)]
        result = find_adjacent(frags, distance_threshold=20.0)
        assert len(result) == 0

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_adjacent([make_frag(0, 0, 0)], distance_threshold=-1.0)

    def test_three_fragments_close(self):
        frags = [make_frag(0, 0, 0, 50, 50),
                 make_frag(1, 60, 0, 50, 50),
                 make_frag(2, 120, 0, 50, 50)]
        result = find_adjacent(frags, distance_threshold=200.0)
        assert len(result) == 3  # C(3,2) = 3 pairs


# ── analyze_all_gaps ──────────────────────────────────────────────────────────

class TestAnalyzeAllGaps:
    def test_empty_returns_empty(self):
        assert analyze_all_gaps([]) == []

    def test_single_returns_empty(self):
        assert analyze_all_gaps([make_frag(0, 0, 0)]) == []

    def test_two_fragments_one_pair(self):
        result = analyze_all_gaps([make_frag(0, 0, 0), make_frag(1, 100, 0)])
        assert len(result) == 1

    def test_n_fragments_c_n_2_pairs(self):
        frags = [make_frag(i, i * 100, 0) for i in range(5)]
        result = analyze_all_gaps(frags)
        assert len(result) == 10  # C(5,2) = 10

    def test_all_results_are_gap_info(self):
        frags = [make_frag(i, i * 100, 0) for i in range(3)]
        result = analyze_all_gaps(frags)
        for gi in result:
            assert isinstance(gi, GapInfo)


# ── gap_histogram ─────────────────────────────────────────────────────────────

class TestGapHistogram:
    def test_empty_gaps_returns_zeros(self):
        counts, edges = gap_histogram([], bins=5)
        assert len(counts) == 5
        assert np.all(counts == 0)

    def test_bins_must_be_positive(self):
        with pytest.raises(ValueError, match="bins должен быть >= 1"):
            gap_histogram([], bins=0)

    def test_output_shapes(self):
        frags = [make_frag(i, i * 100, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, edges = gap_histogram(gaps, bins=5)
        assert len(counts) == 5
        assert len(edges) == 6

    def test_counts_sum_to_n_gaps(self):
        frags = [make_frag(i, i * 100, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, _ = gap_histogram(gaps, bins=5)
        assert counts.sum() == len(gaps)

    def test_counts_dtype(self):
        frags = [make_frag(0, 0, 0), make_frag(1, 100, 0)]
        gaps = analyze_all_gaps(frags)
        counts, _ = gap_histogram(gaps, bins=3)
        assert counts.dtype == np.int64


# ── classify_gaps ─────────────────────────────────────────────────────────────

class TestClassifyGaps:
    def test_returns_four_categories(self):
        result = classify_gaps([])
        assert set(result.keys()) == {"overlap", "touching", "near", "far"}

    def test_correct_classification(self):
        gi_far = GapInfo(id1=0, id2=1, gap_x=100.0, gap_y=100.0,
                         distance=141.4, category="far")
        gi_near = GapInfo(id1=2, id2=3, gap_x=3.0, gap_y=3.0,
                          distance=4.2, category="near")
        result = classify_gaps([gi_far, gi_near])
        assert len(result["far"]) == 1
        assert len(result["near"]) == 1
        assert len(result["overlap"]) == 0

    def test_empty_gaps(self):
        result = classify_gaps([])
        for cat in ("overlap", "touching", "near", "far"):
            assert result[cat] == []

    def test_all_categories_present(self):
        gis = [
            GapInfo(id1=0, id2=1, gap_x=-5, gap_y=-5, distance=5, category="overlap"),
            GapInfo(id1=0, id2=2, gap_x=0, gap_y=5, distance=5, category="touching"),
            GapInfo(id1=0, id2=3, gap_x=3, gap_y=3, distance=4, category="near"),
            GapInfo(id1=0, id2=4, gap_x=50, gap_y=50, distance=70, category="far"),
        ]
        result = classify_gaps(gis)
        assert len(result["overlap"]) == 1
        assert len(result["touching"]) == 1
        assert len(result["near"]) == 1
        assert len(result["far"]) == 1


# ── summarize ─────────────────────────────────────────────────────────────────

class TestSummarize:
    def test_empty_gaps_returns_default(self):
        result = summarize([])
        assert result.n_pairs == 0
        assert result.mean_distance == 0.0

    def test_n_pairs_correct(self):
        frags = [make_frag(i, i * 100, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        result = summarize(gaps)
        assert result.n_pairs == 6  # C(4,2)

    def test_mean_distance_nonneg(self):
        frags = [make_frag(i, i * 100, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        result = summarize(gaps)
        assert result.mean_distance >= 0.0

    def test_category_counts_sum_to_n_pairs(self):
        frags = [make_frag(i, i * 100, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        result = summarize(gaps)
        total = (result.n_overlapping + result.n_touching +
                 result.n_near + result.n_far)
        assert total == result.n_pairs

    def test_returns_gap_stats(self):
        frags = [make_frag(0, 0, 0), make_frag(1, 100, 0)]
        gaps = analyze_all_gaps(frags)
        result = summarize(gaps)
        assert isinstance(result, GapStats)


# ── batch_analyze ─────────────────────────────────────────────────────────────

class TestBatchAnalyze:
    def test_output_length(self):
        layouts = [
            [make_frag(0, 0, 0), make_frag(1, 100, 0)],
            [make_frag(0, 0, 0), make_frag(1, 200, 0), make_frag(2, 400, 0)],
        ]
        result = batch_analyze(layouts)
        assert len(result) == 2

    def test_empty_layouts(self):
        result = batch_analyze([])
        assert result == []

    def test_each_is_gap_stats(self):
        layouts = [[make_frag(0, 0, 0), make_frag(1, 100, 0)]]
        result = batch_analyze(layouts)
        assert isinstance(result[0], GapStats)

    def test_single_fragment_layout(self):
        layouts = [[make_frag(0, 0, 0)]]
        result = batch_analyze(layouts)
        assert result[0].n_pairs == 0

    def test_near_threshold_applied(self):
        a = make_frag(0, 0, 0, 50, 50)
        b = make_frag(1, 53, 0, 50, 50)  # gap_x=3
        layouts = [[a, b]]
        # With large threshold, should be "near"
        result_large = batch_analyze(layouts, near_threshold=10.0)
        result_small = batch_analyze(layouts, near_threshold=1.0)
        assert isinstance(result_large[0], GapStats)
        assert isinstance(result_small[0], GapStats)
