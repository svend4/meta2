"""Extra tests for puzzle_reconstruction/assembly/fragment_mapper.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.fragment_mapper import (
    MapConfig,
    MapResult,
    FragmentZone,
    assign_to_zone,
    batch_build_fragment_maps,
    build_fragment_map,
    compute_zone_grid,
    remap_fragments,
    score_mapping,
)


# ─── MapConfig (extra) ───────────────────────────────────────────────────────

class TestMapConfigExtra:
    def test_default_canvas_w(self):
        assert MapConfig().canvas_w == 512

    def test_default_canvas_h(self):
        assert MapConfig().canvas_h == 512

    def test_default_n_zones_x(self):
        assert MapConfig().n_zones_x == 4

    def test_default_n_zones_y(self):
        assert MapConfig().n_zones_y == 4

    def test_default_allow_multi_false(self):
        assert MapConfig().allow_multi is False

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=0)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_h=0)

    def test_canvas_w_neg_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=-1)

    def test_n_zones_x_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_x=0)

    def test_n_zones_y_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_y=0)

    def test_allow_multi_true(self):
        assert MapConfig(allow_multi=True).allow_multi is True

    def test_large_canvas_ok(self):
        cfg = MapConfig(canvas_w=4096, canvas_h=4096)
        assert cfg.canvas_w == 4096

    def test_custom_zones(self):
        cfg = MapConfig(n_zones_x=8, n_zones_y=6)
        assert cfg.n_zones_x == 8
        assert cfg.n_zones_y == 6


# ─── FragmentZone (extra) ────────────────────────────────────────────────────

class TestFragmentZoneExtra:
    def test_fields_stored(self):
        fz = FragmentZone(fragment_id=3, zone_x=1, zone_y=2, confidence=0.75)
        assert fz.fragment_id == 3
        assert fz.zone_x == 1
        assert fz.zone_y == 2
        assert fz.confidence == pytest.approx(0.75)

    def test_default_confidence_one(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        assert fz.confidence == pytest.approx(1.0)

    def test_fragment_id_neg_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=-1, zone_x=0, zone_y=0)

    def test_zone_x_neg_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=-1, zone_y=0)

    def test_zone_y_neg_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=-1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.1)

    def test_confidence_zero_ok(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=0.0)
        assert fz.confidence == pytest.approx(0.0)

    def test_confidence_one_ok(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.0)
        assert fz.confidence == pytest.approx(1.0)

    def test_zone_index_property(self):
        fz = FragmentZone(fragment_id=5, zone_x=3, zone_y=2)
        assert fz.zone_index == (3, 2)

    def test_zone_index_zero(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        assert fz.zone_index == (0, 0)


# ─── MapResult (extra) ───────────────────────────────────────────────────────

class TestMapResultExtra:
    def _make(self, n=4):
        assignments = [
            FragmentZone(fragment_id=i, zone_x=i % 2, zone_y=i // 2)
            for i in range(n)
        ]
        return MapResult(assignments=assignments, n_fragments=n,
                         n_zones=4, n_assigned=n)

    def test_fields_stored(self):
        r = self._make(4)
        assert r.n_fragments == 4
        assert r.n_zones == 4
        assert r.n_assigned == 4

    def test_n_fragments_neg_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=-1, n_zones=1, n_assigned=0)

    def test_n_zones_zero_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=0, n_assigned=0)

    def test_n_assigned_neg_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=1, n_assigned=-1)

    def test_by_fragment_dict(self):
        r = self._make(3)
        bf = r.by_fragment
        assert isinstance(bf, dict)
        for i in range(3):
            assert i in bf

    def test_by_zone_dict(self):
        r = self._make(4)
        bz = r.by_zone
        assert isinstance(bz, dict)
        for k in bz.keys():
            assert isinstance(k, tuple) and len(k) == 2

    def test_coverage_ratio_in_range(self):
        r = self._make(4)
        assert 0.0 <= r.coverage_ratio <= 1.0

    def test_coverage_ratio_full(self):
        r = MapResult(
            assignments=[FragmentZone(fragment_id=i, zone_x=i % 2, zone_y=i // 2)
                         for i in range(4)],
            n_fragments=4, n_zones=4, n_assigned=4,
        )
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_empty(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert r.coverage_ratio == pytest.approx(0.0)

    def test_assignments_stored(self):
        r = self._make(3)
        assert len(r.assignments) == 3


# ─── compute_zone_grid (extra) ───────────────────────────────────────────────

class TestComputeZoneGridExtra:
    def test_returns_list(self):
        assert isinstance(compute_zone_grid(), list)

    def test_default_length_16(self):
        assert len(compute_zone_grid()) == 16  # 4×4

    def test_custom_config_length(self):
        cfg = MapConfig(n_zones_x=3, n_zones_y=2)
        assert len(compute_zone_grid(cfg)) == 6

    def test_each_zone_is_tuple_4(self):
        for zone in compute_zone_grid():
            assert isinstance(zone, tuple) and len(zone) == 4

    def test_x0_le_x1(self):
        for x0, y0, x1, y1 in compute_zone_grid():
            assert x0 <= x1

    def test_y0_le_y1(self):
        for x0, y0, x1, y1 in compute_zone_grid():
            assert y0 <= y1

    def test_first_zone_at_origin(self):
        x0, y0, x1, y1 = compute_zone_grid()[0]
        assert x0 == 0 and y0 == 0

    def test_last_zone_ends_at_canvas(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        x0, y0, x1, y1 = zones[-1]
        assert x1 == 100 and y1 == 100

    def test_none_cfg_default(self):
        assert len(compute_zone_grid(None)) == 16

    def test_1x1_zones(self):
        cfg = MapConfig(n_zones_x=1, n_zones_y=1)
        zones = compute_zone_grid(cfg)
        assert len(zones) == 1


# ─── assign_to_zone (extra) ──────────────────────────────────────────────────

class TestAssignToZoneExtra:
    def test_returns_tuple_of_two(self):
        result = assign_to_zone(0, 0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_origin_maps_00(self):
        zx, zy = assign_to_zone(0, 0)
        assert zx == 0 and zy == 0

    def test_center_zone(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zx, zy = assign_to_zone(50, 50, cfg)
        assert zx == 1 and zy == 1

    def test_beyond_canvas_clamped(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(999, 999, cfg)
        assert zx == 3 and zy == 3

    def test_negative_clamped_to_zero(self):
        zx, zy = assign_to_zone(-50, -50)
        assert zx == 0 and zy == 0

    def test_known_point_zone(self):
        cfg = MapConfig(canvas_w=400, canvas_h=400, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(150, 50, cfg)
        assert zx == 1 and zy == 0

    def test_zone_integers(self):
        zx, zy = assign_to_zone(100, 200)
        assert isinstance(zx, int) and isinstance(zy, int)


# ─── build_fragment_map (extra) ──────────────────────────────────────────────

class TestBuildFragmentMapExtra:
    def test_returns_map_result(self):
        r = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        assert isinstance(r, MapResult)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            build_fragment_map([0, 1], [(0, 0)])

    def test_n_fragments_correct(self):
        r = build_fragment_map([0, 1, 2], [(0, 0), (100, 0), (200, 0)])
        assert r.n_fragments == 3

    def test_n_zones_from_config(self):
        cfg = MapConfig(n_zones_x=3, n_zones_y=3)
        r = build_fragment_map([0], [(0, 0)], cfg)
        assert r.n_zones == 9

    def test_all_assigned(self):
        r = build_fragment_map([0, 1, 2], [(0, 0), (100, 100), (300, 300)])
        assert r.n_assigned == 3

    def test_correct_zone_assignment(self):
        cfg = MapConfig(canvas_w=200, canvas_h=200, n_zones_x=2, n_zones_y=2)
        r = build_fragment_map([0], [(150, 50)], cfg)
        assert r.assignments[0].zone_x == 1
        assert r.assignments[0].zone_y == 0

    def test_empty_input(self):
        r = build_fragment_map([], [])
        assert r.n_fragments == 0 and r.n_assigned == 0

    def test_coverage_ratio_in_range(self):
        r = build_fragment_map([0, 1], [(0, 0), (400, 400)])
        assert 0.0 <= r.coverage_ratio <= 1.0


# ─── remap_fragments (extra) ─────────────────────────────────────────────────

class TestRemapFragmentsExtra:
    def _make_result(self):
        assignments = [
            FragmentZone(fragment_id=10, zone_x=0, zone_y=0),
            FragmentZone(fragment_id=20, zone_x=1, zone_y=1),
            FragmentZone(fragment_id=30, zone_x=2, zone_y=2),
        ]
        return MapResult(assignments=assignments, n_fragments=3,
                         n_zones=16, n_assigned=3)

    def test_returns_map_result(self):
        r = remap_fragments(self._make_result(), {10: 1, 20: 2, 30: 3})
        assert isinstance(r, MapResult)

    def test_ids_remapped(self):
        r = remap_fragments(self._make_result(), {10: 100, 20: 200, 30: 300})
        ids = {fz.fragment_id for fz in r.assignments}
        assert ids == {100, 200, 300}

    def test_unknown_id_skipped(self):
        r = remap_fragments(self._make_result(), {10: 100})
        assert len(r.assignments) == 1
        assert r.assignments[0].fragment_id == 100

    def test_n_fragments_preserved(self):
        orig = self._make_result()
        r = remap_fragments(orig, {10: 1, 20: 2, 30: 3})
        assert r.n_fragments == orig.n_fragments

    def test_n_zones_preserved(self):
        orig = self._make_result()
        r = remap_fragments(orig, {10: 1})
        assert r.n_zones == orig.n_zones

    def test_empty_mapping(self):
        orig = self._make_result()
        r = remap_fragments(orig, {})
        assert len(r.assignments) == 0


# ─── score_mapping (extra) ───────────────────────────────────────────────────

class TestScoreMappingExtra:
    def test_returns_float(self):
        r = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        assert isinstance(score_mapping(r), float)

    def test_range_0_to_1(self):
        r = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        s = score_mapping(r)
        assert 0.0 <= s <= 1.0

    def test_empty_returns_zero(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert score_mapping(r) == pytest.approx(0.0)

    def test_full_coverage_high_score(self):
        assignments = [
            FragmentZone(fragment_id=i, zone_x=i % 4, zone_y=i // 4, confidence=1.0)
            for i in range(16)
        ]
        r = MapResult(assignments=assignments, n_fragments=16,
                      n_zones=16, n_assigned=16)
        assert score_mapping(r) > 0.8

    def test_single_fragment(self):
        r = build_fragment_map([0], [(0, 0)])
        s = score_mapping(r)
        assert 0.0 <= s <= 1.0


# ─── batch_build_fragment_maps (extra) ───────────────────────────────────────

class TestBatchBuildFragmentMapsExtra:
    def test_returns_list(self):
        r = batch_build_fragment_maps([[0, 1], [2, 3]],
                                      [[(0, 0), (100, 100)],
                                       [(200, 0), (300, 100)]])
        assert isinstance(r, list)

    def test_length_matches(self):
        r = batch_build_fragment_maps([[0], [1], [2]],
                                      [[(0, 0)], [(100, 0)], [(200, 0)]])
        assert len(r) == 3

    def test_each_map_result(self):
        r = batch_build_fragment_maps([[0, 1]], [[(0, 0), (100, 100)]])
        for item in r:
            assert isinstance(item, MapResult)

    def test_empty_lists(self):
        assert batch_build_fragment_maps([], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            batch_build_fragment_maps([[0]], [])

    def test_single_item(self):
        r = batch_build_fragment_maps([[0]], [[(50, 50)]])
        assert len(r) == 1
