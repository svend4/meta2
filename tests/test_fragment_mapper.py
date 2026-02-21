"""Тесты для puzzle_reconstruction.assembly.fragment_mapper."""
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


class TestMapConfig:
    def test_defaults(self):
        cfg = MapConfig()
        assert cfg.canvas_w == 512
        assert cfg.canvas_h == 512
        assert cfg.n_zones_x == 4
        assert cfg.n_zones_y == 4
        assert cfg.allow_multi is False

    def test_valid_custom(self):
        cfg = MapConfig(canvas_w=256, canvas_h=128, n_zones_x=2, n_zones_y=2)
        assert cfg.canvas_w == 256
        assert cfg.canvas_h == 128

    def test_invalid_canvas_w(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_w=0)

    def test_invalid_canvas_h(self):
        with pytest.raises(ValueError):
            MapConfig(canvas_h=0)

    def test_invalid_n_zones_x(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_x=0)

    def test_invalid_n_zones_y(self):
        with pytest.raises(ValueError):
            MapConfig(n_zones_y=0)


class TestFragmentZone:
    def test_basic(self):
        fz = FragmentZone(fragment_id=0, zone_x=1, zone_y=2)
        assert fz.fragment_id == 0
        assert fz.zone_index == (1, 2)
        assert fz.confidence == 1.0

    def test_custom_confidence(self):
        fz = FragmentZone(fragment_id=5, zone_x=0, zone_y=0, confidence=0.75)
        assert fz.confidence == 0.75

    def test_zone_index_property(self):
        fz = FragmentZone(fragment_id=1, zone_x=3, zone_y=2)
        assert fz.zone_index == (3, 2)

    def test_invalid_fragment_id(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=-1, zone_x=0, zone_y=0)

    def test_invalid_zone_x(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=-1, zone_y=0)

    def test_invalid_zone_y(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=-1)

    def test_invalid_confidence_below(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=-0.1)

    def test_invalid_confidence_above(self):
        with pytest.raises(ValueError):
            FragmentZone(fragment_id=0, zone_x=0, zone_y=0, confidence=1.1)


class TestMapResult:
    def _make(self):
        fzs = [
            FragmentZone(0, 0, 0),
            FragmentZone(1, 1, 0),
            FragmentZone(2, 0, 1),
        ]
        return MapResult(assignments=fzs, n_fragments=3, n_zones=4, n_assigned=3)

    def test_by_fragment(self):
        r = self._make()
        d = r.by_fragment
        assert 0 in d and 1 in d and 2 in d
        assert d[0].zone_x == 0

    def test_by_zone(self):
        r = self._make()
        bz = r.by_zone
        assert 0 in bz[(0, 0)]
        assert 1 in bz[(1, 0)]

    def test_coverage_ratio(self):
        r = self._make()
        # 3 different zones out of 4
        assert abs(r.coverage_ratio - 3 / 4) < 1e-9

    def test_coverage_ratio_empty(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert r.coverage_ratio == 0.0

    def test_invalid_n_fragments(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=-1, n_zones=4, n_assigned=0)

    def test_invalid_n_zones(self):
        with pytest.raises(ValueError):
            MapResult(assignments=[], n_fragments=0, n_zones=0, n_assigned=0)


class TestComputeZoneGrid:
    def test_default_count(self):
        cfg = MapConfig(n_zones_x=2, n_zones_y=3)
        zones = compute_zone_grid(cfg)
        assert len(zones) == 6

    def test_zone_structure(self):
        zones = compute_zone_grid(MapConfig(canvas_w=100, canvas_h=100,
                                            n_zones_x=2, n_zones_y=2))
        for z in zones:
            x0, y0, x1, y1 = z
            assert x0 < x1
            assert y0 < y1

    def test_default_none(self):
        zones = compute_zone_grid()
        assert len(zones) == 16  # 4×4


class TestAssignToZone:
    def test_top_left(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(0, 0, cfg)
        assert zx == 0 and zy == 0

    def test_bottom_right(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(99, 99, cfg)
        assert zx == 3 and zy == 3

    def test_clamping_over_bounds(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=4, n_zones_y=4)
        zx, zy = assign_to_zone(200, 200, cfg)
        assert zx == cfg.n_zones_x - 1
        assert zy == cfg.n_zones_y - 1

    def test_default_none(self):
        zx, zy = assign_to_zone(0, 0)
        assert zx >= 0 and zy >= 0


class TestBuildFragmentMap:
    def test_basic(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        ids = [0, 1, 2]
        positions = [(10, 10), (60, 10), (10, 60)]
        result = build_fragment_map(ids, positions, cfg)
        assert result.n_fragments == 3
        assert result.n_assigned == 3
        assert result.n_zones == 4

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_fragment_map([0, 1], [(10, 10)], MapConfig())

    def test_all_same_zone(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        ids = [0, 1]
        positions = [(0, 0), (5, 5)]
        result = build_fragment_map(ids, positions, cfg)
        zones = {fz.zone_index for fz in result.assignments}
        assert len(zones) == 1

    def test_empty_input(self):
        result = build_fragment_map([], [], MapConfig())
        assert result.n_fragments == 0
        assert result.n_assigned == 0


class TestRemapFragments:
    def test_basic_remap(self):
        ids = [0, 1, 2]
        positions = [(10, 10), (60, 10), (10, 60)]
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        result = build_fragment_map(ids, positions, cfg)
        remapped = remap_fragments(result, {0: 10, 1: 11, 2: 12})
        fids = {fz.fragment_id for fz in remapped.assignments}
        assert fids == {10, 11, 12}

    def test_unknown_ids_skipped(self):
        ids = [0, 1]
        positions = [(10, 10), (60, 10)]
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        result = build_fragment_map(ids, positions, cfg)
        remapped = remap_fragments(result, {0: 99})
        assert remapped.n_assigned == 1
        assert remapped.assignments[0].fragment_id == 99


class TestScoreMapping:
    def test_full_coverage_perfect_confidence(self):
        # All fragments in different zones with confidence=1.0
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        ids = [0, 1, 2, 3]
        positions = [(10, 10), (60, 10), (10, 60), (60, 60)]
        result = build_fragment_map(ids, positions, cfg)
        score = score_mapping(result)
        assert score > 0.0
        assert score <= 1.0

    def test_empty_gives_zero(self):
        r = MapResult(assignments=[], n_fragments=0, n_zones=4, n_assigned=0)
        assert score_mapping(r) == 0.0


class TestBatchBuildFragmentMaps:
    def test_basic(self):
        cfg = MapConfig(canvas_w=100, canvas_h=100, n_zones_x=2, n_zones_y=2)
        id_lists = [[0, 1], [2, 3]]
        pos_lists = [[(10, 10), (60, 10)], [(10, 60), (60, 60)]]
        results = batch_build_fragment_maps(id_lists, pos_lists, cfg)
        assert len(results) == 2
        for r in results:
            assert r.n_assigned == 2

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_build_fragment_maps([[0]], [[]], MapConfig())

    def test_empty(self):
        results = batch_build_fragment_maps([], [], MapConfig())
        assert results == []
