"""Tests for puzzle_reconstruction.utils.canvas_build_utils"""
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


# ─── CanvasBuildConfig ────────────────────────────────────────────────────────

def test_canvas_build_config_defaults():
    cfg = CanvasBuildConfig()
    assert cfg.min_coverage == 0.0
    assert cfg.max_fragments == 1000
    assert cfg.blend_mode == "overwrite"


def test_canvas_build_config_invalid_min_coverage():
    with pytest.raises(ValueError, match="min_coverage"):
        CanvasBuildConfig(min_coverage=-0.1)


def test_canvas_build_config_invalid_max_fragments():
    with pytest.raises(ValueError, match="max_fragments"):
        CanvasBuildConfig(max_fragments=0)


def test_canvas_build_config_invalid_blend_mode():
    with pytest.raises(ValueError, match="blend_mode"):
        CanvasBuildConfig(blend_mode="invalid")


def test_canvas_build_config_valid_blend_modes():
    cfg1 = CanvasBuildConfig(blend_mode="overwrite")
    cfg2 = CanvasBuildConfig(blend_mode="average")
    assert cfg1.blend_mode == "overwrite"
    assert cfg2.blend_mode == "average"


# ─── PlacementEntry ───────────────────────────────────────────────────────────

def test_placement_entry_area():
    e = PlacementEntry(fragment_id=0, x=10, y=20, w=30, h=40)
    assert e.area == 30 * 40


def test_placement_entry_x2_y2():
    e = PlacementEntry(fragment_id=1, x=5, y=7, w=10, h=15)
    assert e.x2 == 15
    assert e.y2 == 22


def test_placement_entry_invalid_fragment_id():
    with pytest.raises(ValueError, match="fragment_id"):
        PlacementEntry(fragment_id=-1, x=0, y=0, w=10, h=10)


def test_placement_entry_invalid_wh():
    with pytest.raises(ValueError, match="w and h"):
        PlacementEntry(fragment_id=0, x=0, y=0, w=0, h=10)


def test_placement_entry_coverage_contribution():
    e = PlacementEntry(fragment_id=2, x=0, y=0, w=5, h=5, coverage_contribution=0.1)
    assert e.coverage_contribution == pytest.approx(0.1)


# ─── make_placement_entry ─────────────────────────────────────────────────────

def test_make_placement_entry_basic():
    e = make_placement_entry(0, 0, 0, 10, 20)
    assert e.fragment_id == 0
    assert e.area == 200


def test_make_placement_entry_with_meta():
    e = make_placement_entry(1, 5, 5, 10, 10, meta={"tag": "test"})
    assert e.meta["tag"] == "test"


# ─── entries_from_placements ─────────────────────────────────────────────────

def test_entries_from_placements():
    placements = [(0, 0, 0, 10, 20), (1, 5, 5, 8, 12)]
    entries = entries_from_placements(placements)
    assert len(entries) == 2
    assert entries[0].fragment_id == 0
    assert entries[1].area == 96


def test_entries_from_placements_empty():
    entries = entries_from_placements([])
    assert entries == []


# ─── summarise_canvas_build ───────────────────────────────────────────────────

def test_summarise_canvas_build_basic():
    entries = [
        make_placement_entry(0, 0, 0, 10, 10),
        make_placement_entry(1, 10, 0, 5, 5),
    ]
    s = summarise_canvas_build(entries, canvas_w=100, canvas_h=100, coverage=0.5)
    assert s.n_placed == 2
    assert s.total_area == 100 + 25
    assert s.canvas_w == 100
    assert s.canvas_h == 100
    assert s.coverage == pytest.approx(0.5)


def test_summarise_canvas_build_empty():
    s = summarise_canvas_build([], canvas_w=50, canvas_h=50, coverage=0.0)
    assert s.n_placed == 0
    assert s.total_area == 0


# ─── filter_by_area ──────────────────────────────────────────────────────────

def test_filter_by_area():
    entries = [
        make_placement_entry(0, 0, 0, 2, 2),    # area=4
        make_placement_entry(1, 0, 0, 5, 5),    # area=25
        make_placement_entry(2, 0, 0, 10, 10),  # area=100
    ]
    result = filter_by_area(entries, min_area=10, max_area=50)
    assert len(result) == 1
    assert result[0].fragment_id == 1


# ─── filter_by_coverage_contribution ─────────────────────────────────────────

def test_filter_by_coverage_contribution():
    entries = [
        make_placement_entry(0, 0, 0, 1, 1, coverage_contribution=0.1),
        make_placement_entry(1, 0, 0, 1, 1, coverage_contribution=0.5),
        make_placement_entry(2, 0, 0, 1, 1, coverage_contribution=0.9),
    ]
    result = filter_by_coverage_contribution(entries, min_contrib=0.4)
    assert len(result) == 2


# ─── top_k_by_coverage ───────────────────────────────────────────────────────

def test_top_k_by_coverage():
    entries = [
        make_placement_entry(0, 0, 0, 1, 1, coverage_contribution=0.1),
        make_placement_entry(1, 0, 0, 1, 1, coverage_contribution=0.9),
        make_placement_entry(2, 0, 0, 1, 1, coverage_contribution=0.5),
    ]
    top2 = top_k_by_coverage(entries, k=2)
    assert len(top2) == 2
    assert top2[0].coverage_contribution == pytest.approx(0.9)


def test_top_k_by_coverage_invalid_k():
    with pytest.raises(ValueError, match="k must be"):
        top_k_by_coverage([], k=0)


# ─── canvas_build_stats ──────────────────────────────────────────────────────

def test_canvas_build_stats_empty():
    stats = canvas_build_stats([])
    assert stats["n"] == 0
    assert stats["total_area"] == 0


def test_canvas_build_stats_basic():
    entries = [
        make_placement_entry(0, 0, 0, 4, 5, coverage_contribution=0.2),
        make_placement_entry(1, 0, 0, 2, 5, coverage_contribution=0.4),
    ]
    stats = canvas_build_stats(entries)
    assert stats["n"] == 2
    assert stats["total_area"] == 20 + 10
    assert stats["mean_area"] == pytest.approx(15.0)
    assert stats["mean_coverage_contribution"] == pytest.approx(0.3)


# ─── compare_canvas_summaries ─────────────────────────────────────────────────

def test_compare_canvas_summaries():
    e1 = [make_placement_entry(0, 0, 0, 10, 10)]
    e2 = [make_placement_entry(0, 0, 0, 5, 5), make_placement_entry(1, 0, 0, 3, 3)]
    s1 = summarise_canvas_build(e1, 100, 100, 0.3)
    s2 = summarise_canvas_build(e2, 200, 150, 0.5)
    diff = compare_canvas_summaries(s1, s2)
    assert diff["n_placed_delta"] == -1
    assert diff["coverage_delta"] == pytest.approx(-0.2)
    assert diff["canvas_w_delta"] == -100


# ─── batch_summarise_canvas_builds ───────────────────────────────────────────

def test_batch_summarise_canvas_builds():
    entries1 = [make_placement_entry(0, 0, 0, 10, 10)]
    entries2 = [make_placement_entry(1, 0, 0, 5, 5)]
    specs = [(entries1, 100, 100, 0.4), (entries2, 50, 50, 0.2)]
    results = batch_summarise_canvas_builds(specs)
    assert len(results) == 2
    assert results[0].n_placed == 1
    assert results[1].coverage == pytest.approx(0.2)
