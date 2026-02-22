"""Расширенные тесты для puzzle_reconstruction/assembly/fragment_mapper.py."""
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


# ─── TestMapConfig ────────────────────────────────────────────────────────────

class TestMapConfig:
    def test_defaults(self):
        c = MapConfig()
        assert c.canvas_w == 512
        assert c.canvas_h == 512
        assert c.n_zones_x == 4
        assert c.n_zones_y == 4
        assert c.allow_multi is False

    def test_custom_values(self):
        c = MapConfig(canvas_w=256, canvas_h=128, n_zones_x=2, n_zones_y=3)
        assert c.canvas_w == 256
        assert c.canvas_h == 128
        assert c.n_zones_x == 2
        assert c.n_zones_y == 3

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=0)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_h=0)

    def test_n_zones_x_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_x=0)

    def test_n_zones_y_zero_raises(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_y=0)

    def test_canvas_w_negative_raises(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=-1)

    def test_allow_multi_true(self):
        c = MapConfig(allow_multi=True)
        assert c.allow_multi is True


# ─── TestFragmentZone ─────────────────────────────────────────────────────────

class TestFragmentZone:
    def test_stores_fields(self):
        fz = FragmentZone(fragment_id=1, zone_x=2, zone_y=3, confidence=0.8)
        assert fz.fragment_id == 1
        assert fz.zone_x == 2
        assert fz.zone_y == 3
        assert fz.confidence == pytest.approx(0.8)

    def test_default_confidence(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        assert fz.confidence == pytest.approx(1.0)

    def test_fragment_id_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=-1, zone_x=0, zone_y=0)

    def test_zone_x_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=-1, zone_y=0)

    def test_zone_y_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=-1)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=-0.1)

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.1)

    def test_confidence_0_ok(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=0.0)
        assert fz.confidence == pytest.approx(0.0)

    def test_confidence_1_ok(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.0)
        assert fz.confidence == pytest.approx(1.0)

    def test_zone_index_property(self):
        fz = FragmentZone(fragment_id=5, zone_x=2, zone_y=3)
        assert fz.zone_index == (2, 3)


# ─── TestMapResult ────────────────────────────────────────────────────────────

class TestMapResult:
    def _make_result(self, n=3):
        assignments = [
            FragmentZone(fragment_id=i, zone_x=i % 2, zone_y=i // 2)
            for i in range(n)
        ]
        return MapResult(
            assignments=assignments,
            n_fragments=n,
            n_zones=4,
            n_assigned=n,
        )

    def test_stores_fields(self):
        r = self._make_result(3)
        assert r.n_fragments == 3
        assert r.n_zones == 4
        assert r.n_assigned == 3

    def test_n_fragments_negative_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=-1, n_zones=1, n_assigned=0)

    def test_n_zones_zero_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=0, n_assigned=0)

    def test_n_assigned_negative_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=1, n_assigned=-1)

    def test_by_fragment_property(self):
        r = self._make_result(3)
        bf = r.by_fragment
        assert isinstance(bf, dict)
        for i in range(3):
            assert i in bf
            assert isinstance(bf[i], FragmentZone)

    def test_by_zone_property(self):
        r = self._make_result(3)
        bz = r.by_zone
        assert isinstance(bz, dict)
        # All zone indices must be tuples
        for k in bz.keys():
            assert isinstance(k, tuple)
            assert len(k) == 2

    def test_coverage_ratio_in_range(self):
        r = self._make_result(3)
        cr = r.coverage_ratio
        assert 0.0 <= cr <= 1.0

    def test_coverage_ratio_empty_assignments(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert r.coverage_ratio == pytest.approx(0.0)

    def test_coverage_ratio_full(self):
        # All 4 zones occupied
        assignments = [
            FragmentZone(fragment_id=i, zone_x=i % 2, zone_y=i // 2)
            for i in range(4)
        ]
        r = MapResult(assignments=assignments, n_fragments=4, n_zones=4, n_assigned=4)
        assert r.coverage_ratio == pytest.approx(1.0)


# ─── TestComputeZoneGrid ──────────────────────────────────────────────────────

class TestComputeZoneGrid:
    def test_returns_list(self):
        result = compute_zone_grid()
        assert isinstance(result, list)

    def test_default_length_16(self):
        result = compute_zone_grid()
        assert len(result) == 16  # 4 * 4

    def test_custom_config_length(self):
        cfg = MapConfig(n_zones_x=3, n_zones_y=2)
        result = compute_zone_grid(cfg)
        assert len(result) == 6

    def test_each_zone_is_tuple_of_4(self):
        result = compute_zone_grid()
        for zone in result:
            assert isinstance(zone, tuple)
            assert len(zone) == 4

    def test_x0_le_x1(self):
        for x0, y0, x1, y1 in compute_zone_grid():
            assert x0 <= x1

    def test_y0_le_y1(self):
        for x0, y0, x1, y1 in compute_zone_grid():
            assert y0 <= y1

    def test_first_zone_starts_at_origin(self):
        x0, y0, x1, y1 = compute_zone_grid()[0]
        assert x0 == 0
        assert y0 == 0

    def test_last_zone_ends_at_canvas(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        x0, y0, x1, y1 = zones[-1]
        assert x1 == 100
        assert y1 == 100

    def test_none_cfg_uses_defaults(self):
        result = compute_zone_grid(None)
        assert len(result) == 16


# ─── TestAssignToZone ─────────────────────────────────────────────────────────

class TestAssignToZone:
    def test_returns_tuple_of_2(self):
        result = assign_to_zone(0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_origin_maps_to_00(self):
        zx, zy = assign_to_zone(0, 0)
        assert zx == 0
        assert zy == 0

    def test_center_of_canvas(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zx, zy = assign_to_zone(50, 50, cfg)
        # x=50 in canvas_w=100 with n_zones_x=2 → zone 1
        assert zx == 1
        assert zy == 1

    def test_beyond_canvas_clamped(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(200, 200, cfg)
        assert zx == 3
        assert zy == 3

    def test_negative_coords_clamped_to_0(self):
        zx, zy = assign_to_zone(-50, -50)
        assert zx == 0
        assert zy == 0

    def test_correct_zone_for_known_point(self):
        cfg = MapConfig(canvas_w=400, canvas_h=400, n_zones_x=4, n_zones_y=4)
        # Point at (150, 50): zone_w=100, zone_h=100
        # x=150 → zx=1, y=50 → zy=0
        zx, zy = assign_to_zone(150, 50, cfg)
        assert zx == 1
        assert zy == 0


# ─── TestBuildFragmentMap ─────────────────────────────────────────────────────

class TestBuildFragmentMap:
    def test_returns_map_result(self):
        result = build_fragment_map([0, 1, 2], [(0, 0), (100, 0), (200, 0)])
        assert isinstance(result, MapResult)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_fragment_map([0, 1], [(0, 0)])

    def test_n_fragments_stored(self):
        result = build_fragment_map([0, 1, 2], [(0, 0), (100, 0), (200, 0)])
        assert result.n_fragments == 3

    def test_n_zones_equals_n_zones_x_times_y(self):
        cfg = MapConfig(n_zones_x=3, n_zones_y=2)
        result = build_fragment_map([0], [(0, 0)], cfg)
        assert result.n_zones == 6

    def test_all_assigned(self):
        ids = [0, 1, 2]
        positions = [(0, 0), (100, 100), (300, 300)]
        result = build_fragment_map(ids, positions)
        assert result.n_assigned == 3

    def test_correct_zone_assignment(self):
        cfg = MapConfig(canvas_w=200, canvas_h=200, n_zones_x=2, n_zones_y=2)
        result = build_fragment_map([0], [(150, 50)], cfg)
        fz = result.assignments[0]
        assert fz.zone_x == 1
        assert fz.zone_y == 0

    def test_coverage_ratio_in_range(self):
        result = build_fragment_map([0, 1], [(0, 0), (511, 511)])
        assert 0.0 <= result.coverage_ratio <= 1.0

    def test_empty_fragments(self):
        result = build_fragment_map([], [])
        assert result.n_fragments == 0
        assert result.n_assigned == 0


# ─── TestRemapFragments ───────────────────────────────────────────────────────

class TestRemapFragments:
    def _make_result(self):
        assignments = [
            FragmentZone(fragment_id=10, zone_x=0, zone_y=0),
            FragmentZone(fragment_id=20, zone_x=1, zone_y=1),
            FragmentZone(fragment_id=30, zone_x=2, zone_y=2),
        ]
        return MapResult(assignments=assignments, n_fragments=3,
                         n_zones=16, n_assigned=3)

    def test_returns_map_result(self):
        result = self._make_result()
        assert isinstance(remap_fragments(result, {10: 1, 20: 2, 30: 3}), MapResult)

    def test_known_ids_remapped(self):
        result = self._make_result()
        remapped = remap_fragments(result, {10: 100, 20: 200, 30: 300})
        ids = {fz.fragment_id for fz in remapped.assignments}
        assert ids == {100, 200, 300}

    def test_unknown_id_skipped(self):
        result = self._make_result()
        remapped = remap_fragments(result, {10: 100})
        assert len(remapped.assignments) == 1
        assert remapped.assignments[0].fragment_id == 100

    def test_n_fragments_preserved(self):
        result = self._make_result()
        remapped = remap_fragments(result, {10: 1, 20: 2, 30: 3})
        assert remapped.n_fragments == result.n_fragments

    def test_n_zones_preserved(self):
        result = self._make_result()
        remapped = remap_fragments(result, {10: 1})
        assert remapped.n_zones == result.n_zones


# ─── TestScoreMapping ─────────────────────────────────────────────────────────

class TestScoreMapping:
    def test_returns_float(self):
        result = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        assert isinstance(score_mapping(result), float)

    def test_range_0_to_1(self):
        result = build_fragment_map([0, 1], [(0, 0), (100, 100)])
        s = score_mapping(result)
        assert 0.0 <= s <= 1.0

    def test_empty_assignments_returns_0(self):
        result = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert score_mapping(result) == pytest.approx(0.0)

    def test_high_confidence_high_coverage_high_score(self):
        # All 16 zones occupied with confidence=1.0
        assignments = [
            FragmentZone(fragment_id=i, zone_x=i % 4, zone_y=i // 4,
                         confidence=1.0)
            for i in range(16)
        ]
        result = MapResult(assignments=assignments, n_fragments=16,
                           n_zones=16, n_assigned=16)
        s = score_mapping(result)
        assert s > 0.8


# ─── TestBatchBuildFragmentMaps ───────────────────────────────────────────────

class TestBatchBuildFragmentMaps:
    def test_returns_list(self):
        result = batch_build_fragment_maps([[0, 1], [2, 3]],
                                            [[(0, 0), (100, 100)],
                                             [(200, 0), (300, 100)]])
        assert isinstance(result, list)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            batch_build_fragment_maps([[0]], [])

    def test_length_matches(self):
        result = batch_build_fragment_maps(
            [[0], [1], [2]],
            [[(0, 0)], [(100, 0)], [(200, 0)]],
        )
        assert len(result) == 3

    def test_each_is_map_result(self):
        result = batch_build_fragment_maps([[0, 1]], [[(0, 0), (100, 100)]])
        assert all(isinstance(r, MapResult) for r in result)

    def test_empty_lists(self):
        result = batch_build_fragment_maps([], [])
        assert result == []
