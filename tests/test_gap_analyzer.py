"""Тесты для puzzle_reconstruction.assembly.gap_analyzer."""
import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fb(fid, x, y, w=20, h=20):
    return FragmentBounds(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── TestFragmentBounds ───────────────────────────────────────────────────────

class TestFragmentBounds:
    def test_basic_creation(self):
        fb = _fb(0, 10.0, 5.0, 30, 20)
        assert fb.fragment_id == 0
        assert fb.x == pytest.approx(10.0)

    def test_x2(self):
        fb = _fb(0, 10.0, 5.0, 30, 20)
        assert fb.x2 == pytest.approx(40.0)

    def test_y2(self):
        fb = _fb(0, 10.0, 5.0, 30, 20)
        assert fb.y2 == pytest.approx(25.0)

    def test_center(self):
        fb = _fb(0, 0.0, 0.0, 20, 10)
        cx, cy = fb.center
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(5.0)

    def test_area(self):
        fb = _fb(0, 0.0, 0.0, 10, 5)
        assert fb.area == pytest.approx(50.0)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=-1, x=0, y=0, width=10, height=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_width_below_one_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=0, width=0, height=10)

    def test_height_below_one_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=0, width=10, height=0)


# ─── TestGapInfo ──────────────────────────────────────────────────────────────

class TestGapInfo:
    def test_basic_creation(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=3.0, distance=8.0)
        assert gi.id1 == 0
        assert gi.gap_x == pytest.approx(5.0)

    def test_pair_property(self):
        gi = GapInfo(id1=2, id2=5, gap_x=0.0, gap_y=0.0, distance=10.0)
        assert gi.pair == (2, 5)

    def test_is_overlapping_true(self):
        gi = GapInfo(id1=0, id2=1, gap_x=-5.0, gap_y=-3.0, distance=5.0,
                     category="overlap")
        assert gi.is_overlapping is True

    def test_is_overlapping_false(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=3.0, distance=10.0)
        assert gi.is_overlapping is False

    def test_negative_id1_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=-1, id2=0, gap_x=0, gap_y=0, distance=0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=-1.0)

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=5.0,
                    category="huge")

    def test_valid_categories(self):
        for cat in ("overlap", "touching", "near", "far"):
            gi = GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=0,
                         category=cat)
            assert gi.category == cat


# ─── TestGapStats ─────────────────────────────────────────────────────────────

class TestGapStats:
    def test_defaults(self):
        s = GapStats()
        assert s.n_pairs == 0
        assert s.mean_distance == pytest.approx(0.0)

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            GapStats(n_pairs=-1)

    def test_negative_mean_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(mean_distance=-1.0)

    def test_negative_std_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(std_distance=-1.0)

    def test_valid_creation(self):
        s = GapStats(n_pairs=3, n_far=2, n_near=1, mean_distance=15.0)
        assert s.n_pairs == 3


# ─── TestComputeGap ───────────────────────────────────────────────────────────

class TestComputeGap:
    def test_positive_gap_x(self):
        a = _fb(0, 0, 0, 10, 10)   # x: 0-10
        b = _fb(1, 15, 0, 10, 10)  # x: 15-25
        gi = compute_gap(a, b)
        assert gi.gap_x == pytest.approx(5.0)

    def test_overlap_negative_gap(self):
        a = _fb(0, 0, 0, 20, 20)
        b = _fb(1, 5, 5, 20, 20)
        gi = compute_gap(a, b)
        assert gi.gap_x < 0
        assert gi.gap_y < 0

    def test_category_overlap(self):
        a = _fb(0, 0, 0, 20, 20)
        b = _fb(1, 5, 5, 20, 20)
        gi = compute_gap(a, b)
        assert gi.category == "overlap"

    def test_category_far(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 100, 100, 10, 10)
        gi = compute_gap(a, b, near_threshold=5.0)
        assert gi.category == "far"

    def test_distance_nonneg(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 50, 50, 10, 10)
        gi = compute_gap(a, b)
        assert gi.distance >= 0.0

    def test_id_order_normalized(self):
        a = _fb(5, 0, 0, 10, 10)
        b = _fb(2, 50, 0, 10, 10)
        gi = compute_gap(a, b)
        assert gi.id1 == 2
        assert gi.id2 == 5

    def test_negative_threshold_raises(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 20, 0, 10, 10)
        with pytest.raises(ValueError):
            compute_gap(a, b, near_threshold=-1.0)


# ─── TestFindAdjacent ─────────────────────────────────────────────────────────

class TestFindAdjacent:
    def test_returns_list(self):
        frags = [_fb(0, 0, 0), _fb(1, 25, 0), _fb(2, 200, 200)]
        result = find_adjacent(frags, distance_threshold=50.0)
        assert isinstance(result, list)

    def test_all_gap_infos(self):
        frags = [_fb(0, 0, 0), _fb(1, 25, 0)]
        result = find_adjacent(frags, distance_threshold=100.0)
        assert all(isinstance(g, GapInfo) for g in result)

    def test_far_fragments_excluded(self):
        frags = [_fb(0, 0, 0), _fb(1, 1000, 1000)]
        result = find_adjacent(frags, distance_threshold=10.0)
        assert result == []

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_adjacent([_fb(0, 0, 0)], distance_threshold=-1.0)

    def test_empty_list(self):
        result = find_adjacent([], distance_threshold=10.0)
        assert result == []

    def test_single_fragment(self):
        result = find_adjacent([_fb(0, 0, 0)], distance_threshold=10.0)
        assert result == []


# ─── TestAnalyzeAllGaps ───────────────────────────────────────────────────────

class TestAnalyzeAllGaps:
    def test_count_c_n_2(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        result = analyze_all_gaps(frags)
        assert len(result) == 6  # C(4,2)

    def test_empty_returns_empty(self):
        assert analyze_all_gaps([]) == []

    def test_single_returns_empty(self):
        assert analyze_all_gaps([_fb(0, 0, 0)]) == []

    def test_two_frags_one_pair(self):
        frags = [_fb(0, 0, 0), _fb(1, 30, 0)]
        result = analyze_all_gaps(frags)
        assert len(result) == 1

    def test_all_gap_infos(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        result = analyze_all_gaps(frags)
        assert all(isinstance(g, GapInfo) for g in result)


# ─── TestGapHistogram ─────────────────────────────────────────────────────────

class TestGapHistogram:
    def test_returns_tuple(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        result = gap_histogram(gaps, bins=5)
        assert len(result) == 2

    def test_counts_shape(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, edges = gap_histogram(gaps, bins=5)
        assert len(counts) == 5
        assert len(edges) == 6

    def test_empty_gaps(self):
        counts, edges = gap_histogram([], bins=5)
        assert len(counts) == 5

    def test_bins_below_one_raises(self):
        with pytest.raises(ValueError):
            gap_histogram([], bins=0)

    def test_counts_nonneg(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, _ = gap_histogram(gaps)
        assert np.all(counts >= 0)


# ─── TestClassifyGaps ─────────────────────────────────────────────────────────

class TestClassifyGaps:
    def test_returns_dict(self):
        result = classify_gaps([])
        assert isinstance(result, dict)

    def test_keys(self):
        result = classify_gaps([])
        assert set(result.keys()) == {"overlap", "touching", "near", "far"}

    def test_overlapping_goes_to_overlap(self):
        a = _fb(0, 0, 0, 20, 20)
        b = _fb(1, 5, 5, 20, 20)
        gi = compute_gap(a, b)
        result = classify_gaps([gi])
        assert gi in result["overlap"]

    def test_far_classification(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 200, 200, 10, 10)
        gi = compute_gap(a, b, near_threshold=5.0)
        result = classify_gaps([gi])
        assert gi in result["far"]


# ─── TestSummarize ────────────────────────────────────────────────────────────

class TestSummarize:
    def test_empty_returns_default(self):
        s = summarize([])
        assert s.n_pairs == 0

    def test_n_pairs_correct(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert s.n_pairs == 3

    def test_returns_gap_stats(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert isinstance(s, GapStats)

    def test_category_counts_sum(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        total = s.n_overlapping + s.n_touching + s.n_near + s.n_far
        assert total == s.n_pairs

    def test_mean_distance_nonneg(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert s.mean_distance >= 0.0


# ─── TestBatchAnalyze ─────────────────────────────────────────────────────────

class TestBatchAnalyze:
    def test_returns_list(self):
        layouts = [[_fb(i, i * 30, 0) for i in range(3)]]
        result = batch_analyze(layouts)
        assert isinstance(result, list)

    def test_length_matches_layouts(self):
        layouts = [
            [_fb(i, i * 30, 0) for i in range(3)],
            [_fb(i, i * 50, 0) for i in range(4)],
        ]
        result = batch_analyze(layouts)
        assert len(result) == 2

    def test_all_gap_stats(self):
        layouts = [[_fb(i, i * 30, 0) for i in range(3)]]
        result = batch_analyze(layouts)
        assert isinstance(result[0], GapStats)

    def test_empty_layouts(self):
        result = batch_analyze([])
        assert result == []

    def test_empty_layout_gives_zero_pairs(self):
        result = batch_analyze([[]])
        assert result[0].n_pairs == 0
