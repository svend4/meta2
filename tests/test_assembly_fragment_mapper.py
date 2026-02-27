"""Tests for puzzle_reconstruction/assembly/fragment_mapper.py"""
import pytest
import numpy as np

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


# ── MapConfig ──────────────────────────────────────────────────────────────────

class TestMapConfig:
    def test_default_values(self):
        cfg = MapConfig()
        assert cfg.canvas_w == 512
        assert cfg.canvas_h == 512
        assert cfg.n_zones_x == 4
        assert cfg.n_zones_y == 4
        assert cfg.allow_multi is False

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

    def test_valid_custom_values(self):
        cfg = MapConfig(canvas_w=1024, canvas_h=768, n_zones_x=8, n_zones_y=6)
        assert cfg.canvas_w == 1024
        assert cfg.n_zones_x == 8

    def test_allow_multi_true(self):
        cfg = MapConfig(allow_multi=True)
        assert cfg.allow_multi is True


# ── FragmentZone ──────────────────────────────────────────────────────────────

class TestFragmentZone:
    def test_valid_construction(self):
        fz = FragmentZone(fragment_id=0, zone_x=1, zone_y=2)
        assert fz.fragment_id == 0
        assert fz.zone_x == 1
        assert fz.zone_y == 2

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id должен быть >= 0"):
            FragmentZone(fragment_id=-1, zone_x=0, zone_y=0)

    def test_negative_zone_x_raises(self):
        with pytest.raises(ValueError, match="zone_x должен быть >= 0"):
            FragmentZone(fragment_id=0, zone_x=-1, zone_y=0)

    def test_negative_zone_y_raises(self):
        with pytest.raises(ValueError, match="zone_y должен быть >= 0"):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=-1)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence должен быть в"):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.5)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=-0.1)

    def test_zone_index_property(self):
        fz = FragmentZone(fragment_id=0, zone_x=2, zone_y=3)
        assert fz.zone_index == (2, 3)

    def test_default_confidence_one(self):
        fz = FragmentZone(fragment_id=0, zone_x=0, zone_y=0)
        assert fz.confidence == 1.0

    def test_confidence_boundary_values(self):
        fz0 = FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=0.0)
        fz1 = FragmentZone(fragment_id=1, zone_x=0, zone_y=0, confidence=1.0)
        assert fz0.confidence == 0.0
        assert fz1.confidence == 1.0


# ── MapResult ──────────────────────────────────────────────────────────────────

class TestMapResult:
    def _make_zone(self, fid, zx=0, zy=0):
        return FragmentZone(fragment_id=fid, zone_x=zx, zone_y=zy)

    def test_valid_construction(self):
        mr = MapResult(assignments=[], n_fragments=0, n_zones=1, n_assigned=0)
        assert mr.n_fragments == 0

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=-1, n_zones=1, n_assigned=0)

    def test_n_zones_zero_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=0, n_assigned=0)

    def test_negative_n_assigned_raises(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=1, n_assigned=-1)

    def test_by_fragment_property(self):
        fz = self._make_zone(42, zx=1, zy=2)
        mr = MapResult(assignments=[fz], n_fragments=1, n_zones=16, n_assigned=1)
        bf = mr.by_fragment
        assert 42 in bf
        assert bf[42].zone_x == 1

    def test_by_zone_property(self):
        fz1 = self._make_zone(0, zx=0, zy=0)
        fz2 = self._make_zone(1, zx=0, zy=0)
        mr = MapResult(assignments=[fz1, fz2], n_fragments=2, n_zones=16, n_assigned=2)
        bz = mr.by_zone
        assert (0, 0) in bz
        assert set(bz[(0, 0)]) == {0, 1}

    def test_coverage_ratio_empty(self):
        mr = MapResult(assignments=[], n_fragments=0, n_zones=16, n_assigned=0)
        assert mr.coverage_ratio == 0.0

    def test_coverage_ratio_all_different_zones(self):
        fzs = [self._make_zone(i, zx=i, zy=0) for i in range(4)]
        mr = MapResult(assignments=fzs, n_fragments=4, n_zones=4, n_assigned=4)
        assert mr.coverage_ratio == 1.0

    def test_coverage_ratio_partial(self):
        fzs = [self._make_zone(0, zx=0, zy=0), self._make_zone(1, zx=0, zy=0)]
        mr = MapResult(assignments=fzs, n_fragments=2, n_zones=4, n_assigned=2)
        assert mr.coverage_ratio == 0.25


# ── compute_zone_grid ─────────────────────────────────────────────────────────

class TestComputeZoneGrid:
    def test_default_config_returns_16_zones(self):
        zones = compute_zone_grid()
        assert len(zones) == 16  # 4x4

    def test_custom_config(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=3)
        zones = compute_zone_grid(cfg)
        assert len(zones) == 6

    def test_zones_are_tuples_of_4(self):
        zones = compute_zone_grid()
        for z in zones:
            assert len(z) == 4

    def test_first_zone_starts_at_origin(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        zones = compute_zone_grid(cfg)
        x0, y0, x1, y1 = zones[0]
        assert x0 == 0
        assert y0 == 0

    def test_zones_cover_canvas(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zones = compute_zone_grid(cfg)
        max_x1 = max(z[2] for z in zones)
        max_y1 = max(z[3] for z in zones)
        assert max_x1 == 100
        assert max_y1 == 100

    def test_none_config_uses_default(self):
        zones = compute_zone_grid(None)
        assert len(zones) == 16


# ── assign_to_zone ────────────────────────────────────────────────────────────

class TestAssignToZone:
    def test_origin_is_zone_0_0(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(0, 0, cfg)
        assert zx == 0
        assert zy == 0

    def test_far_right_bottom_clamped(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(200, 200, cfg)
        assert zx == cfg.n_zones_x - 1
        assert zy == cfg.n_zones_y - 1

    def test_middle_point(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(50, 50, cfg)
        assert 0 <= zx < 4
        assert 0 <= zy < 4

    def test_returns_tuple(self):
        result = assign_to_zone(10, 10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_none_config_uses_default(self):
        zx, zy = assign_to_zone(0, 0, None)
        assert zx == 0
        assert zy == 0


# ── build_fragment_map ────────────────────────────────────────────────────────

class TestBuildFragmentMap:
    def test_empty_lists(self):
        result = build_fragment_map([], [])
        assert result.n_fragments == 0
        assert result.n_assigned == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_fragment_map([0, 1], [(0, 0)])

    def test_single_fragment(self):
        result = build_fragment_map([0], [(10, 20)])
        assert result.n_fragments == 1
        assert result.n_assigned == 1

    def test_multiple_fragments(self):
        ids = [0, 1, 2]
        positions = [(0, 0), (100, 100), (200, 200)]
        result = build_fragment_map(ids, positions)
        assert result.n_fragments == 3
        assert result.n_assigned == 3

    def test_n_zones_correct(self):
        cfg = MapConfig(n_zones_x=3, n_zones_y=3)
        result = build_fragment_map([0], [(0, 0)], cfg)
        assert result.n_zones == 9

    def test_assignments_fragment_ids_match(self):
        ids = [5, 10, 15]
        positions = [(0, 0), (100, 0), (200, 0)]
        result = build_fragment_map(ids, positions)
        assigned_ids = {fz.fragment_id for fz in result.assignments}
        assert assigned_ids == set(ids)

    def test_zone_indices_within_bounds(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        ids = list(range(5))
        positions = [(i * 20, i * 20) for i in range(5)]
        result = build_fragment_map(ids, positions, cfg)
        for fz in result.assignments:
            assert 0 <= fz.zone_x < 4
            assert 0 <= fz.zone_y < 4

    def test_default_confidence_one(self):
        result = build_fragment_map([0], [(0, 0)])
        assert result.assignments[0].confidence == 1.0


# ── remap_fragments ───────────────────────────────────────────────────────────

class TestRemapFragments:
    def _make_result(self, fids):
        fzs = [FragmentZone(fragment_id=f, zone_x=0, zone_y=0) for f in fids]
        return MapResult(assignments=fzs, n_fragments=len(fids), n_zones=16,
                         n_assigned=len(fids))

    def test_basic_remap(self):
        result = self._make_result([0, 1, 2])
        mapping = {0: 10, 1: 11, 2: 12}
        new_result = remap_fragments(result, mapping)
        ids = {fz.fragment_id for fz in new_result.assignments}
        assert ids == {10, 11, 12}

    def test_unknown_ids_skipped(self):
        result = self._make_result([0, 1, 99])
        mapping = {0: 100, 1: 101}
        new_result = remap_fragments(result, mapping)
        assert new_result.n_assigned == 2

    def test_empty_mapping(self):
        result = self._make_result([0, 1])
        new_result = remap_fragments(result, {})
        assert new_result.n_assigned == 0

    def test_zone_info_preserved(self):
        fz = FragmentZone(fragment_id=5, zone_x=2, zone_y=3, confidence=0.8)
        mr = MapResult(assignments=[fz], n_fragments=1, n_zones=16, n_assigned=1)
        new_mr = remap_fragments(mr, {5: 99})
        assert new_mr.assignments[0].zone_x == 2
        assert new_mr.assignments[0].zone_y == 3
        assert new_mr.assignments[0].confidence == 0.8


# ── score_mapping ─────────────────────────────────────────────────────────────

class TestScoreMapping:
    def _make_result_with_confidence(self, confs, n_zones=None):
        fzs = [FragmentZone(fragment_id=i, zone_x=i, zone_y=0, confidence=c)
               for i, c in enumerate(confs)]
        nz = n_zones if n_zones else max(len(confs), 1)
        return MapResult(assignments=fzs, n_fragments=len(confs),
                         n_zones=nz, n_assigned=len(confs))

    def test_empty_assignments_returns_zero(self):
        mr = MapResult(assignments=[], n_fragments=0, n_zones=1, n_assigned=0)
        assert score_mapping(mr) == 0.0

    def test_perfect_score(self):
        # All confidence=1, coverage_ratio=1 (each in unique zone, n_zones=2)
        mr = self._make_result_with_confidence([1.0, 1.0], n_zones=2)
        score = score_mapping(mr)
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_score_in_range(self):
        mr = self._make_result_with_confidence([0.5, 0.8])
        score = score_mapping(mr)
        assert 0.0 <= score <= 1.0

    def test_low_confidence_lowers_score(self):
        mr_high = self._make_result_with_confidence([1.0], n_zones=1)
        mr_low = self._make_result_with_confidence([0.1], n_zones=1)
        assert score_mapping(mr_high) > score_mapping(mr_low)


# ── batch_build_fragment_maps ─────────────────────────────────────────────────

class TestBatchBuildFragmentMaps:
    def test_output_length(self):
        id_lists = [[0, 1], [2, 3]]
        pos_lists = [[(0, 0), (100, 0)], [(50, 50), (150, 50)]]
        results = batch_build_fragment_maps(id_lists, pos_lists)
        assert len(results) == 2

    def test_mismatched_lists_raise(self):
        with pytest.raises(ValueError):
            batch_build_fragment_maps([[0]], [])

    def test_empty_lists(self):
        results = batch_build_fragment_maps([], [])
        assert results == []

    def test_each_result_is_map_result(self):
        results = batch_build_fragment_maps([[0]], [[(0, 0)]])
        assert isinstance(results[0], MapResult)

    def test_custom_cfg_applied(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        results = batch_build_fragment_maps([[0]], [[(0, 0)]], cfg)
        assert results[0].n_zones == 4
