"""Extra tests for puzzle_reconstruction.assembly.gap_analyzer."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.gap_analyzer import (
    FragmentBounds,
    GapInfo,
    GapStats,
    analyze_all_gaps,
    batch_analyze,
    classify_gaps,
    compute_gap,
    find_adjacent,
    gap_histogram,
    summarize,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fb(fid, x, y, w=20, h=20):
    return FragmentBounds(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── TestFragmentBoundsExtra ──────────────────────────────────────────────────

class TestFragmentBoundsExtra:
    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=-1, width=10, height=10)

    def test_x2_y2_consistent(self):
        fb = _fb(0, 5.0, 3.0, 15, 10)
        assert fb.x2 == pytest.approx(20.0)
        assert fb.y2 == pytest.approx(13.0)

    def test_center_non_zero_origin(self):
        fb = _fb(0, 10.0, 20.0, 40, 20)
        cx, cy = fb.center
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(30.0)

    def test_area_large(self):
        fb = _fb(0, 0, 0, 100, 50)
        assert fb.area == pytest.approx(5000.0)

    def test_fragment_id_zero_ok(self):
        fb = _fb(0, 0, 0, 1, 1)
        assert fb.fragment_id == 0

    def test_width_one_ok(self):
        fb = FragmentBounds(fragment_id=0, x=0, y=0, width=1, height=1)
        assert fb.area == pytest.approx(1.0)


# ─── TestGapInfoExtra ─────────────────────────────────────────────────────────

class TestGapInfoExtra:
    def test_is_overlapping_false_for_near(self):
        gi = GapInfo(id1=0, id2=1, gap_x=2.0, gap_y=2.0, distance=5.0,
                     category="near")
        assert gi.is_overlapping is False

    def test_is_overlapping_false_for_far(self):
        gi = GapInfo(id1=0, id2=1, gap_x=50.0, gap_y=50.0, distance=100.0,
                     category="far")
        assert gi.is_overlapping is False

    def test_pair_always_smaller_first(self):
        # pair property returns (id1, id2) as stored
        gi = GapInfo(id1=1, id2=4, gap_x=0, gap_y=0, distance=10.0)
        assert gi.pair == (1, 4)

    def test_touching_category_valid(self):
        gi = GapInfo(id1=0, id2=1, gap_x=0.0, gap_y=0.0, distance=0.0,
                     category="touching")
        assert gi.category == "touching"

    def test_default_category(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=0.0, distance=5.0)
        # default category should be one of the valid ones
        assert gi.category in ("overlap", "touching", "near", "far")

    def test_zero_distance_nonneg(self):
        gi = GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=0.0)
        assert gi.distance == pytest.approx(0.0)


# ─── TestGapStatsExtra ────────────────────────────────────────────────────────

class TestGapStatsExtra:
    def test_n_far_default_zero(self):
        s = GapStats()
        assert s.n_far == 0

    def test_n_near_default_zero(self):
        s = GapStats()
        assert s.n_near == 0

    def test_n_overlapping_default_zero(self):
        s = GapStats()
        assert s.n_overlapping == 0

    def test_n_touching_default_zero(self):
        s = GapStats()
        assert s.n_touching == 0

    def test_std_distance_default_zero(self):
        s = GapStats()
        assert s.std_distance == pytest.approx(0.0)

    def test_fields_positive(self):
        s = GapStats(n_pairs=4, n_far=2, n_near=1, n_overlapping=1,
                     mean_distance=10.0, std_distance=2.0)
        assert s.n_pairs == 4
        assert s.mean_distance == pytest.approx(10.0)


# ─── TestComputeGapExtra ──────────────────────────────────────────────────────

class TestComputeGapExtra:
    def test_touching_same_edge(self):
        # b starts exactly where a ends
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 10, 0, 10, 10)
        gi = compute_gap(a, b)
        # gap_x should be 0 (touching)
        assert gi.gap_x == pytest.approx(0.0)

    def test_near_category_within_threshold(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 13, 0, 10, 10)
        gi = compute_gap(a, b, near_threshold=5.0)
        assert gi.category == "near"

    def test_gap_y_positive(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 0, 15, 10, 10)
        gi = compute_gap(a, b)
        assert gi.gap_y == pytest.approx(5.0)

    def test_id_order_normalized_when_reversed(self):
        a = _fb(3, 0, 0, 10, 10)
        b = _fb(1, 50, 0, 10, 10)
        gi = compute_gap(a, b)
        assert gi.id1 < gi.id2

    def test_distance_symmetric(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 20, 0, 10, 10)
        gi1 = compute_gap(a, b)
        gi2 = compute_gap(b, a)
        assert gi1.distance == pytest.approx(gi2.distance)

    def test_zero_threshold_ok(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 50, 0, 10, 10)
        gi = compute_gap(a, b, near_threshold=0.0)
        assert gi.category in ("overlap", "touching", "near", "far")


# ─── TestFindAdjacentExtra ────────────────────────────────────────────────────

class TestFindAdjacentExtra:
    def test_three_frags_two_nearby(self):
        frags = [_fb(0, 0, 0), _fb(1, 25, 0), _fb(2, 500, 0)]
        result = find_adjacent(frags, distance_threshold=50.0)
        ids = {(g.id1, g.id2) for g in result}
        assert (0, 1) in ids
        assert not any(2 in pair for pair in ids)

    def test_all_pairs_close(self):
        frags = [_fb(i, i * 5, 0) for i in range(3)]
        result = find_adjacent(frags, distance_threshold=100.0)
        assert len(result) == 3  # C(3,2)

    def test_large_threshold_includes_all(self):
        frags = [_fb(i, i * 200, 0) for i in range(4)]
        result = find_adjacent(frags, distance_threshold=1e9)
        assert len(result) == 6  # C(4,2)

    def test_result_all_gap_infos(self):
        frags = [_fb(0, 0, 0), _fb(1, 10, 0)]
        result = find_adjacent(frags, distance_threshold=100.0)
        assert all(isinstance(g, GapInfo) for g in result)


# ─── TestAnalyzeAllGapsExtra ──────────────────────────────────────────────────

class TestAnalyzeAllGapsExtra:
    def test_five_frags_ten_pairs(self):
        frags = [_fb(i, i * 30, 0) for i in range(5)]
        result = analyze_all_gaps(frags)
        assert len(result) == 10  # C(5,2)

    def test_distances_nonneg(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        for gi in analyze_all_gaps(frags):
            assert gi.distance >= 0.0

    def test_ids_in_range(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        all_ids = {g.id1 for g in analyze_all_gaps(frags)} | \
                  {g.id2 for g in analyze_all_gaps(frags)}
        assert all_ids == {0, 1, 2, 3}

    def test_each_pair_once(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        pairs = [g.pair for g in gaps]
        assert len(pairs) == len(set(pairs))


# ─── TestGapHistogramExtra ────────────────────────────────────────────────────

class TestGapHistogramExtra:
    def test_single_gap(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 15, 0, 10, 10)
        gaps = [compute_gap(a, b)]
        counts, edges = gap_histogram(gaps, bins=3)
        assert counts.sum() == 1

    def test_default_bins_positive(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, edges = gap_histogram(gaps)
        assert len(counts) > 0

    def test_edges_ascending(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        _, edges = gap_histogram(gaps, bins=5)
        assert (np.diff(edges) >= 0).all()

    def test_counts_sum_equals_n_gaps(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        counts, _ = gap_histogram(gaps, bins=4)
        assert counts.sum() == len(gaps)


# ─── TestClassifyGapsExtra ────────────────────────────────────────────────────

class TestClassifyGapsExtra:
    def test_near_category(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 12, 0, 10, 10)
        gi = compute_gap(a, b, near_threshold=5.0)
        result = classify_gaps([gi])
        if gi.category == "near":
            assert gi in result["near"]

    def test_touching_category(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 10, 0, 10, 10)
        gi = compute_gap(a, b)
        result = classify_gaps([gi])
        assert gi in result.get(gi.category, [])

    def test_empty_all_keys_empty_lists(self):
        result = classify_gaps([])
        for v in result.values():
            assert v == []

    def test_multiple_gaps_classified(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        result = classify_gaps(gaps)
        total = sum(len(v) for v in result.values())
        assert total == len(gaps)


# ─── TestSummarizeExtra ───────────────────────────────────────────────────────

class TestSummarizeExtra:
    def test_std_distance_nonneg(self):
        frags = [_fb(i, i * 30, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert s.std_distance >= 0.0

    def test_category_counts_nonneg(self):
        frags = [_fb(i, i * 30, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert s.n_far >= 0
        assert s.n_near >= 0
        assert s.n_overlapping >= 0
        assert s.n_touching >= 0

    def test_two_frags_n_pairs_one(self):
        gaps = analyze_all_gaps([_fb(0, 0, 0), _fb(1, 30, 0)])
        s = summarize(gaps)
        assert s.n_pairs == 1

    def test_mean_distance_positive_when_far(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 100, 0, 10, 10)
        s = summarize([compute_gap(a, b)])
        assert s.mean_distance > 0.0


# ─── TestBatchAnalyzeExtra ────────────────────────────────────────────────────

class TestBatchAnalyzeExtra:
    def test_three_layouts(self):
        layouts = [
            [_fb(i, i * 30, 0) for i in range(3)],
            [_fb(i, i * 30, 0) for i in range(4)],
            [_fb(i, i * 30, 0) for i in range(2)],
        ]
        result = batch_analyze(layouts)
        assert len(result) == 3

    def test_correct_n_pairs_per_layout(self):
        layouts = [
            [_fb(i, i * 30, 0) for i in range(3)],
            [_fb(i, i * 30, 0) for i in range(4)],
        ]
        result = batch_analyze(layouts)
        assert result[0].n_pairs == 3   # C(3,2)
        assert result[1].n_pairs == 6   # C(4,2)

    def test_single_frag_layout_zero_pairs(self):
        result = batch_analyze([[_fb(0, 0, 0)]])
        assert result[0].n_pairs == 0

    def test_all_gap_stats_type(self):
        layouts = [[_fb(i, i * 30, 0) for i in range(3)] for _ in range(3)]
        for s in batch_analyze(layouts):
            assert isinstance(s, GapStats)
