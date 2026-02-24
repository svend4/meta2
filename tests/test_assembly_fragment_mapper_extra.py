"""Extra tests for puzzle_reconstruction/assembly/fragment_mapper.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.fragment_mapper import (
    MapConfig,
    FragmentZone,
    MapResult,
    compute_zone_grid,
    assign_to_zone,
    build_fragment_map,
    remap_fragments,
    score_mapping,
    batch_build_fragment_maps,
)


# ─── MapConfig ───────────────────────────────────────────────────────────────

class TestMapConfigExtra:
    def test_defaults(self):
        c = MapConfig()
        assert c.canvas_w == 512
        assert c.canvas_h == 512
        assert c.n_zones_x == 4
        assert c.n_zones_y == 4
        assert c.allow_multi is False

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=0)

    def test_zero_canvas_h_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_h=0)

    def test_zero_zones_x_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_x=0)

    def test_zero_zones_y_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_y=0)


# ─── FragmentZone ────────────────────────────────────────────────────────────

class TestFragmentZoneExtra:
    def test_valid(self):
        fz = FragmentZone(fragment_id=0, zone_x=1, zone_y=2, confidence=0.9)
        assert fz.zone_index == (1, 2)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=-1, zone_x=0, zone_y=0)

    def test_negative_zone_x_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=-1, zone_y=0)

    def test_negative_zone_y_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=-1)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.5)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=-0.1)

    def test_default_confidence(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        assert fz.confidence == pytest.approx(1.0)


# ─── MapResult ───────────────────────────────────────────────────────────────

class TestMapResultExtra:
    def test_valid(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=4, n_assigned=1)
        assert r.n_assigned == 1

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=-1, n_zones=1, n_assigned=0)

    def test_zero_n_zones_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=0, n_assigned=0)

    def test_by_fragment(self):
        fz = FragmentZone(fragment_id=5, zone_x=1, zone_y=2)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=16, n_assigned=1)
        assert 5 in r.by_fragment
        assert r.by_fragment[5].zone_x == 1

    def test_by_zone(self):
        fz = FragmentZone(fragment_id=0, zone_x=1, zone_y=2)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=16, n_assigned=1)
        assert (1, 2) in r.by_zone
        assert 0 in r.by_zone[(1, 2)]

    def test_coverage_ratio(self):
        fz1 = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        fz2 = FragmentZone(fragment_id=1, zone_x=1, zone_y=0)
        r = MapResult(assignments=[fz1, fz2], n_fragments=2,
                      n_zones=4, n_assigned=2)
        assert r.coverage_ratio == pytest.approx(0.5)


# ─── compute_zone_grid ──────────────────────────────────────────────────────

class TestComputeZoneGridExtra:
    def test_default(self):
        zones = compute_zone_grid()
        assert len(zones) == 16  # 4x4

    def test_custom(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        assert len(zones) == 4

    def test_zone_covers_canvas(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        # Last zone should reach canvas edge
        assert zones[-1][2] == 100
        assert zones[-1][3] == 100


# ─── assign_to_zone ─────────────────────────────────────────────────────────

class TestAssignToZoneExtra:
    def test_origin(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zx, zy = assign_to_zone(0, 0, cfg)
        assert (zx, zy) == (0, 0)

    def test_center(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zx, zy = assign_to_zone(50, 50, cfg)
        assert (zx, zy) == (1, 1)

    def test_clamped_high(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zx, zy = assign_to_zone(200, 200, cfg)
        assert zx == 1
        assert zy == 1


# ─── build_fragment_map ─────────────────────────────────────────────────────

class TestBuildFragmentMapExtra:
    def test_basic(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        r = build_fragment_map([0, 1], [(10, 10), (60, 60)], cfg)
        assert isinstance(r, MapResult)
        assert r.n_fragments == 2
        assert r.n_assigned == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_fragment_map([0, 1], [(10, 10)], MapConfig())

    def test_assigns_correct_zones(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        r = build_fragment_map([0], [(10, 10)], cfg)
        assert r.assignments[0].zone_x == 0
        assert r.assignments[0].zone_y == 0


# ─── remap_fragments ────────────────────────────────────────────────────────

class TestRemapFragmentsExtra:
    def test_basic(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=4, n_assigned=1)
        remapped = remap_fragments(r, {0: 10})
        assert remapped.assignments[0].fragment_id == 10

    def test_missing_id_skipped(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=4, n_assigned=1)
        remapped = remap_fragments(r, {99: 10})
        assert len(remapped.assignments) == 0


# ─── score_mapping ──────────────────────────────────────────────────────────

class TestScoreMappingExtra:
    def test_empty(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert score_mapping(r) == pytest.approx(0.0)

    def test_nonzero(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.0)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=4, n_assigned=1)
        s = score_mapping(r)
        assert 0.0 <= s <= 1.0

    def test_range(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=0.5)
        r = MapResult(assignments=[fz], n_fragments=1, n_zones=1, n_assigned=1)
        s = score_mapping(r)
        assert 0.0 <= s <= 1.0


# ─── batch_build_fragment_maps ──────────────────────────────────────────────

class TestBatchBuildFragmentMapsExtra:
    def test_empty(self):
        assert batch_build_fragment_maps([], []) == []

    def test_length(self):
        results = batch_build_fragment_maps(
            [[0], [1]], [[(10, 10)], [(20, 20)]],
        )
        assert len(results) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            batch_build_fragment_maps([[0]], [])

    def test_result_type(self):
        results = batch_build_fragment_maps([[0]], [[(10, 10)]])
        assert isinstance(results[0], MapResult)
