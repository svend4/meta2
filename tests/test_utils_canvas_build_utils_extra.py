"""Extra tests for puzzle_reconstruction/utils/canvas_build_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.canvas_build_utils import (
    CanvasBuildConfig,
    PlacementEntry,
    CanvasBuildSummary,
    make_placement_entry,
    entries_from_placements,
    summarise_canvas_build,
    filter_by_area,
    filter_by_coverage_contribution,
    top_k_by_coverage,
    canvas_build_stats,
    compare_canvas_summaries,
    batch_summarise_canvas_builds,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(fid=0, x=0, y=0, w=10, h=10, contrib=0.5) -> PlacementEntry:
    return PlacementEntry(
        fragment_id=fid, x=x, y=y, w=w, h=h,
        coverage_contribution=contrib,
    )


def _entries(n=5) -> list:
    return [_entry(fid=i, w=i + 1, h=i + 1, contrib=float(i + 1) / n)
            for i in range(n)]


def _summary(entries=None, canvas_w=100, canvas_h=100, coverage=0.5) -> CanvasBuildSummary:
    if entries is None:
        entries = _entries(3)
    return summarise_canvas_build(entries, canvas_w, canvas_h, coverage)


# ─── CanvasBuildConfig ────────────────────────────────────────────────────────

class TestCanvasBuildConfigExtra:
    def test_default_min_coverage(self):
        assert CanvasBuildConfig().min_coverage == pytest.approx(0.0)

    def test_default_max_fragments(self):
        assert CanvasBuildConfig().max_fragments == 1000

    def test_default_blend_mode(self):
        assert CanvasBuildConfig().blend_mode == "overwrite"

    def test_min_coverage_below_zero_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(min_coverage=-0.1)

    def test_min_coverage_above_one_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(min_coverage=1.1)

    def test_max_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(max_fragments=0)

    def test_max_fragments_negative_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(max_fragments=-5)

    def test_invalid_blend_mode_raises(self):
        with pytest.raises(ValueError):
            CanvasBuildConfig(blend_mode="multiply")

    def test_average_blend_mode_valid(self):
        cfg = CanvasBuildConfig(blend_mode="average")
        assert cfg.blend_mode == "average"

    def test_custom_values(self):
        cfg = CanvasBuildConfig(min_coverage=0.2, max_fragments=50)
        assert cfg.min_coverage == pytest.approx(0.2)
        assert cfg.max_fragments == 50


# ─── PlacementEntry ───────────────────────────────────────────────────────────

class TestPlacementEntryExtra:
    def test_stores_fragment_id(self):
        assert _entry(fid=7).fragment_id == 7

    def test_stores_x_y(self):
        e = _entry(x=3, y=5)
        assert e.x == 3 and e.y == 5

    def test_stores_w_h(self):
        e = _entry(w=20, h=30)
        assert e.w == 20 and e.h == 30

    def test_area_computed(self):
        assert _entry(w=4, h=5).area == 20

    def test_x2_computed(self):
        assert _entry(x=3, w=10).x2 == 13

    def test_y2_computed(self):
        assert _entry(y=2, h=8).y2 == 10

    def test_default_meta_empty(self):
        assert _entry().meta == {}

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=-1, x=0, y=0, w=5, h=5)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=0, x=0, y=0, w=0, h=5)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            PlacementEntry(fragment_id=0, x=0, y=0, w=5, h=0)


# ─── make_placement_entry ─────────────────────────────────────────────────────

class TestMakePlacementEntryExtra:
    def test_returns_entry(self):
        assert isinstance(make_placement_entry(0, 0, 0, 10, 10), PlacementEntry)

    def test_stores_values(self):
        e = make_placement_entry(3, 5, 7, 20, 15)
        assert e.fragment_id == 3 and e.x == 5 and e.y == 7
        assert e.w == 20 and e.h == 15

    def test_none_meta_empty(self):
        e = make_placement_entry(0, 0, 0, 5, 5, meta=None)
        assert e.meta == {}

    def test_coverage_contribution_stored(self):
        e = make_placement_entry(0, 0, 0, 10, 10, coverage_contribution=0.3)
        assert e.coverage_contribution == pytest.approx(0.3)

    def test_area_correct(self):
        e = make_placement_entry(0, 0, 0, 6, 7)
        assert e.area == 42


# ─── entries_from_placements ──────────────────────────────────────────────────

class TestEntriesFromPlacementsExtra:
    def test_returns_list(self):
        result = entries_from_placements([(0, 0, 0, 10, 10)])
        assert isinstance(result, list)

    def test_length_matches(self):
        placements = [(i, 0, 0, 5, 5) for i in range(4)]
        assert len(entries_from_placements(placements)) == 4

    def test_empty_returns_empty(self):
        assert entries_from_placements([]) == []

    def test_all_are_placement_entries(self):
        for e in entries_from_placements([(0, 1, 2, 8, 6)]):
            assert isinstance(e, PlacementEntry)

    def test_values_preserved(self):
        e = entries_from_placements([(7, 3, 4, 12, 9)])[0]
        assert e.fragment_id == 7 and e.x == 3 and e.y == 4
        assert e.w == 12 and e.h == 9


# ─── summarise_canvas_build ───────────────────────────────────────────────────

class TestSummariseCanvasBuildExtra:
    def test_returns_summary(self):
        assert isinstance(_summary(), CanvasBuildSummary)

    def test_n_placed_correct(self):
        entries = _entries(4)
        s = summarise_canvas_build(entries, 100, 100, 0.5)
        assert s.n_placed == 4

    def test_canvas_dims_stored(self):
        s = summarise_canvas_build(_entries(2), 200, 150, 0.3)
        assert s.canvas_w == 200 and s.canvas_h == 150

    def test_coverage_stored(self):
        s = summarise_canvas_build(_entries(2), 100, 100, 0.75)
        assert s.coverage == pytest.approx(0.75)

    def test_total_area_sum(self):
        entries = [_entry(w=5, h=4), _entry(w=3, h=6)]
        s = summarise_canvas_build(entries, 100, 100, 0.5)
        assert s.total_area == 20 + 18

    def test_empty_entries(self):
        s = summarise_canvas_build([], 100, 100, 0.0)
        assert s.n_placed == 0 and s.total_area == 0

    def test_repr_is_str(self):
        assert isinstance(repr(_summary()), str)


# ─── filter_by_area ───────────────────────────────────────────────────────────

class TestFilterByAreaExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_area(_entries()), list)

    def test_keeps_in_range(self):
        entries = [_entry(w=2, h=2), _entry(w=5, h=5), _entry(w=10, h=10)]
        result = filter_by_area(entries, min_area=10, max_area=50)
        assert all(10 <= e.area <= 50 for e in result)

    def test_empty_input(self):
        assert filter_by_area([], min_area=0) == []

    def test_wide_range_keeps_all(self):
        entries = _entries(5)
        result = filter_by_area(entries, min_area=0, max_area=10 ** 9)
        assert len(result) == 5

    def test_zero_max_area_empty(self):
        result = filter_by_area(_entries(3), min_area=0, max_area=0)
        assert result == []


# ─── filter_by_coverage_contribution ─────────────────────────────────────────

class TestFilterByCoverageContributionExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_coverage_contribution(_entries()), list)

    def test_keeps_above_min(self):
        entries = [_entry(contrib=0.1), _entry(contrib=0.5), _entry(contrib=0.9)]
        result = filter_by_coverage_contribution(entries, min_contrib=0.4)
        assert all(e.coverage_contribution >= 0.4 for e in result)

    def test_empty_input(self):
        assert filter_by_coverage_contribution([]) == []

    def test_zero_min_keeps_all(self):
        entries = _entries(4)
        result = filter_by_coverage_contribution(entries, min_contrib=0.0)
        assert len(result) == 4


# ─── top_k_by_coverage ────────────────────────────────────────────────────────

class TestTopKByCoverageExtra:
    def test_returns_list(self):
        assert isinstance(top_k_by_coverage(_entries(), 3), list)

    def test_length_at_most_k(self):
        result = top_k_by_coverage(_entries(5), 3)
        assert len(result) <= 3

    def test_k_larger_than_n(self):
        result = top_k_by_coverage(_entries(3), 10)
        assert len(result) == 3

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_by_coverage(_entries(), 0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            top_k_by_coverage(_entries(), -1)

    def test_descending_order(self):
        entries = [_entry(contrib=0.3), _entry(contrib=0.8), _entry(contrib=0.1)]
        result = top_k_by_coverage(entries, 3)
        contribs = [e.coverage_contribution for e in result]
        assert contribs == sorted(contribs, reverse=True)

    def test_empty_input(self):
        assert top_k_by_coverage([], 3) == []


# ─── canvas_build_stats ───────────────────────────────────────────────────────

class TestCanvasBuildStatsExtra:
    def test_returns_dict(self):
        assert isinstance(canvas_build_stats(_entries()), dict)

    def test_keys_present(self):
        stats = canvas_build_stats(_entries(3))
        for k in ("n", "total_area", "mean_area", "mean_coverage_contribution"):
            assert k in stats

    def test_n_correct(self):
        assert canvas_build_stats(_entries(6))["n"] == 6

    def test_empty_entries(self):
        stats = canvas_build_stats([])
        assert stats["n"] == 0

    def test_total_area_correct(self):
        entries = [_entry(w=4, h=5), _entry(w=3, h=3)]
        stats = canvas_build_stats(entries)
        assert stats["total_area"] == 29

    def test_mean_area_positive(self):
        stats = canvas_build_stats(_entries(4))
        assert stats["mean_area"] > 0.0


# ─── compare_canvas_summaries ─────────────────────────────────────────────────

class TestCompareCanvasSummariesExtra:
    def test_returns_dict(self):
        a = _summary()
        b = _summary()
        assert isinstance(compare_canvas_summaries(a, b), dict)

    def test_keys_present(self):
        d = compare_canvas_summaries(_summary(), _summary())
        for k in ("n_placed_delta", "coverage_delta", "total_area_delta",
                  "canvas_w_delta", "canvas_h_delta"):
            assert k in d

    def test_identical_zero_deltas(self):
        s = _summary()
        d = compare_canvas_summaries(s, s)
        assert d["n_placed_delta"] == 0
        assert d["coverage_delta"] == pytest.approx(0.0)

    def test_n_placed_delta(self):
        a = summarise_canvas_build(_entries(4), 100, 100, 0.5)
        b = summarise_canvas_build(_entries(2), 100, 100, 0.5)
        d = compare_canvas_summaries(a, b)
        assert d["n_placed_delta"] == 2


# ─── batch_summarise_canvas_builds ────────────────────────────────────────────

class TestBatchSummariseCanvasBuildsExtra:
    def test_returns_list(self):
        specs = [(_entries(2), 100, 100, 0.5)]
        assert isinstance(batch_summarise_canvas_builds(specs), list)

    def test_length_matches(self):
        specs = [
            (_entries(2), 100, 100, 0.5),
            (_entries(3), 200, 200, 0.8),
        ]
        assert len(batch_summarise_canvas_builds(specs)) == 2

    def test_each_is_summary(self):
        specs = [(_entries(2), 100, 100, 0.5)]
        for s in batch_summarise_canvas_builds(specs):
            assert isinstance(s, CanvasBuildSummary)

    def test_empty_returns_empty(self):
        assert batch_summarise_canvas_builds([]) == []

    def test_values_correct(self):
        entries = _entries(3)
        specs = [(entries, 50, 80, 0.6)]
        result = batch_summarise_canvas_builds(specs)
        assert result[0].canvas_w == 50
        assert result[0].canvas_h == 80
        assert result[0].coverage == pytest.approx(0.6)
