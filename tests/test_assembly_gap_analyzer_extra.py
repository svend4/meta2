"""Extra tests for puzzle_reconstruction/assembly/gap_analyzer.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fb(fid, x, y, w=50, h=50):
    return FragmentBounds(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── FragmentBounds ─────────────────────────────────────────────────────────

class TestFragmentBoundsExtra:
    def test_valid(self):
        fb = _fb(0, 10, 20)
        assert fb.x2 == 60
        assert fb.y2 == 70
        assert fb.center == (35.0, 45.0)
        assert fb.area == 2500.0

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=-1, x=0, y=0, width=10, height=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=-1, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            FragmentBounds(fragment_id=0, x=0, y=0, width=10, height=0)


# ─── GapInfo ────────────────────────────────────────────────────────────────

class TestGapInfoExtra:
    def test_valid(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=5.0,
                     distance=10.0, category="far")
        assert gi.pair == (0, 1)

    def test_negative_id1_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=-1, id2=0, gap_x=0, gap_y=0, distance=0)

    def test_negative_id2_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=0, id2=-1, gap_x=0, gap_y=0, distance=0)

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=-1.0)

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError):
            GapInfo(id1=0, id2=1, gap_x=0, gap_y=0, distance=0, category="bad")

    def test_is_overlapping(self):
        gi = GapInfo(id1=0, id2=1, gap_x=-5.0, gap_y=-5.0,
                     distance=5.0, category="overlap")
        assert gi.is_overlapping is True

    def test_not_overlapping(self):
        gi = GapInfo(id1=0, id2=1, gap_x=5.0, gap_y=5.0,
                     distance=10.0, category="far")
        assert gi.is_overlapping is False


# ─── GapStats ───────────────────────────────────────────────────────────────

class TestGapStatsExtra:
    def test_defaults(self):
        gs = GapStats()
        assert gs.n_pairs == 0
        assert gs.mean_distance == 0.0

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            GapStats(n_pairs=-1)

    def test_negative_mean_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(mean_distance=-1.0)

    def test_negative_std_distance_raises(self):
        with pytest.raises(ValueError):
            GapStats(std_distance=-1.0)


# ─── compute_gap ────────────────────────────────────────────────────────────

class TestComputeGapExtra:
    def test_far_apart(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 100, 100, 10, 10)
        gi = compute_gap(a, b)
        assert gi.category == "far"
        assert gi.distance > 0

    def test_overlapping(self):
        a = _fb(0, 0, 0, 50, 50)
        b = _fb(1, 25, 25, 50, 50)
        gi = compute_gap(a, b)
        assert gi.category == "overlap"

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            compute_gap(_fb(0, 0, 0), _fb(1, 100, 100), near_threshold=-1)

    def test_ids_sorted(self):
        a = _fb(5, 0, 0)
        b = _fb(2, 100, 100)
        gi = compute_gap(a, b)
        assert gi.id1 == 2
        assert gi.id2 == 5

    def test_near(self):
        a = _fb(0, 0, 0, 50, 50)
        b = _fb(1, 53, 0, 50, 50)  # small gap of 3
        gi = compute_gap(a, b, near_threshold=5.0)
        assert gi.category in ("near", "touching", "overlap")


# ─── find_adjacent ──────────────────────────────────────────────────────────

class TestFindAdjacentExtra:
    def test_empty(self):
        assert find_adjacent([]) == []

    def test_single(self):
        assert find_adjacent([_fb(0, 0, 0)]) == []

    def test_two_close(self):
        a = _fb(0, 0, 0, 50, 50)
        b = _fb(1, 10, 10, 50, 50)
        result = find_adjacent([a, b], distance_threshold=100.0)
        assert len(result) == 1

    def test_two_far(self):
        a = _fb(0, 0, 0, 10, 10)
        b = _fb(1, 1000, 1000, 10, 10)
        result = find_adjacent([a, b], distance_threshold=10.0)
        assert len(result) == 0

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            find_adjacent([_fb(0, 0, 0)], distance_threshold=-1)


# ─── analyze_all_gaps ───────────────────────────────────────────────────────

class TestAnalyzeAllGapsExtra:
    def test_empty(self):
        assert analyze_all_gaps([]) == []

    def test_pair_count(self):
        frags = [_fb(i, i * 60, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        assert len(gaps) == 3  # C(3,2) = 3

    def test_four_fragments(self):
        frags = [_fb(i, i * 60, 0) for i in range(4)]
        gaps = analyze_all_gaps(frags)
        assert len(gaps) == 6  # C(4,2) = 6


# ─── gap_histogram ──────────────────────────────────────────────────────────

class TestGapHistogramExtra:
    def test_empty(self):
        counts, edges = gap_histogram([], bins=5)
        assert len(counts) == 5
        assert np.all(counts == 0)

    def test_with_data(self):
        frags = [_fb(i, i * 60, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        counts, edges = gap_histogram(gaps, bins=3)
        assert len(counts) == 3
        assert counts.sum() == 3

    def test_zero_bins_raises(self):
        with pytest.raises(ValueError):
            gap_histogram([], bins=0)


# ─── classify_gaps ──────────────────────────────────────────────────────────

class TestClassifyGapsExtra:
    def test_empty(self):
        result = classify_gaps([])
        assert all(v == [] for v in result.values())

    def test_categories_present(self):
        result = classify_gaps([])
        assert set(result.keys()) == {"overlap", "touching", "near", "far"}


# ─── summarize ──────────────────────────────────────────────────────────────

class TestSummarizeExtra:
    def test_empty(self):
        s = summarize([])
        assert s.n_pairs == 0

    def test_with_data(self):
        frags = [_fb(i, i * 60, 0) for i in range(3)]
        gaps = analyze_all_gaps(frags)
        s = summarize(gaps)
        assert s.n_pairs == 3
        assert s.mean_distance > 0


# ─── batch_analyze ──────────────────────────────────────────────────────────

class TestBatchAnalyzeExtra:
    def test_empty(self):
        assert batch_analyze([]) == []

    def test_length(self):
        layout1 = [_fb(i, i * 60, 0) for i in range(3)]
        layout2 = [_fb(i, i * 60, 0) for i in range(2)]
        results = batch_analyze([layout1, layout2])
        assert len(results) == 2

    def test_result_type(self):
        layout = [_fb(i, i * 60, 0) for i in range(3)]
        results = batch_analyze([layout])
        assert isinstance(results[0], GapStats)
