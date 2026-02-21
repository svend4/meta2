"""
Тесты для puzzle_reconstruction.algorithms.position_estimator.
"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.position_estimator import (
    PositionEstimate,
    build_offset_graph,
    estimate_positions,
    refine_positions,
    positions_to_array,
    align_to_origin,
)


# ─── PositionEstimate ─────────────────────────────────────────────────────────

class TestPositionEstimate:
    def test_basic_fields(self):
        pe = PositionEstimate(idx=3, x=10.5, y=-2.0)
        assert pe.idx == 3
        assert pe.x == pytest.approx(10.5)
        assert pe.y == pytest.approx(-2.0)

    def test_default_confidence(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert pe.confidence == pytest.approx(1.0)

    def test_default_n_constraints(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert pe.n_constraints == 0

    def test_custom_confidence_and_constraints(self):
        pe = PositionEstimate(idx=5, x=1.0, y=2.0, confidence=0.7, n_constraints=4)
        assert pe.confidence == pytest.approx(0.7)
        assert pe.n_constraints == 4

    def test_repr_contains_idx(self):
        pe = PositionEstimate(idx=7, x=3.0, y=4.0)
        r = repr(pe)
        assert "7" in r
        assert "PositionEstimate" in r


# ─── build_offset_graph ───────────────────────────────────────────────────────

class TestBuildOffsetGraph:
    def test_empty(self):
        g = build_offset_graph([], [])
        assert g == {}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_offset_graph([(0, 1)], [])

    def test_single_pair_bidirectional(self):
        g = build_offset_graph([(0, 1)], [(5.0, -3.0)])
        assert 0 in g and 1 in g
        # 0→1: (5, -3)
        assert any(nb == 1 and abs(dx - 5.0) < 1e-9 and abs(dy - (-3.0)) < 1e-9
                   for nb, dx, dy in g[0])
        # 1→0: (-5, 3)
        assert any(nb == 0 and abs(dx - (-5.0)) < 1e-9 and abs(dy - 3.0) < 1e-9
                   for nb, dx, dy in g[1])

    def test_multiple_pairs(self):
        pairs   = [(0, 1), (1, 2), (0, 2)]
        offsets = [(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
        g = build_offset_graph(pairs, offsets)
        assert len(g[0]) == 2   # 0→1 и 0→2
        assert len(g[1]) == 2   # 1→0 и 1→2
        assert len(g[2]) == 2   # 2→1 и 2→0

    def test_offsets_are_floats(self):
        g = build_offset_graph([(0, 1)], [(3, 7)])
        nb, dx, dy = g[0][0]
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_self_loop_accepted(self):
        # build_offset_graph не запрещает петли (это дело estimate_positions)
        g = build_offset_graph([(0, 0)], [(0.0, 0.0)])
        assert 0 in g


# ─── estimate_positions ───────────────────────────────────────────────────────

class TestEstimatePositions:
    def test_empty_graph(self):
        result = estimate_positions({})
        assert result == {}

    def test_single_node(self):
        g = build_offset_graph([(0, 1)], [(10.0, 5.0)])
        pos = estimate_positions(g, root=0)
        assert 0 in pos and 1 in pos
        assert pos[0].x == pytest.approx(0.0)
        assert pos[0].y == pytest.approx(0.0)

    def test_root_at_origin(self):
        g = build_offset_graph([(0, 1)], [(20.0, -10.0)])
        pos = estimate_positions(g, root=0)
        assert pos[0].x == pytest.approx(0.0)
        assert pos[0].y == pytest.approx(0.0)

    def test_chain_positions(self):
        # 0 --(10,0)--> 1 --(10,0)--> 2
        pairs   = [(0, 1), (1, 2)]
        offsets = [(10.0, 0.0), (10.0, 0.0)]
        g   = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert pos[0].x == pytest.approx(0.0)
        assert pos[1].x == pytest.approx(10.0)
        assert pos[2].x == pytest.approx(20.0)
        # y всё время 0
        assert pos[0].y == pytest.approx(0.0)
        assert pos[1].y == pytest.approx(0.0)
        assert pos[2].y == pytest.approx(0.0)

    def test_2d_layout(self):
        # 0 --(5,0)--> 1
        # |               |
        # (0,5)         (0,5)
        # v               v
        # 2 --(5,0)--> 3
        pairs   = [(0, 1), (0, 2), (1, 3), (2, 3)]
        offsets = [(5.0, 0.0), (0.0, 5.0), (0.0, 5.0), (5.0, 0.0)]
        g   = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert pos[1].x == pytest.approx(5.0)
        assert pos[2].y == pytest.approx(5.0)

    def test_root_default_selects_highest_degree(self):
        # узел 1 имеет 3 соседа, 0 и 2 — по одному (через 1)
        pairs   = [(0, 1), (1, 2), (1, 3)]
        offsets = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        g   = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g)           # root не задан
        # Корень должен быть 1 (3 соседа)
        assert 1 in pos
        assert pos[1].x == pytest.approx(0.0)
        assert pos[1].y == pytest.approx(0.0)

    def test_unknown_root_falls_back_to_highest_degree(self):
        g   = build_offset_graph([(0, 1)], [(3.0, 0.0)])
        pos = estimate_positions(g, root=99)   # 99 не существует
        # Должен выбраться один из 0 или 1 в качестве корня, без исключений
        assert len(pos) == 2

    def test_n_constraints_of_non_root_is_1(self):
        g   = build_offset_graph([(0, 1)], [(7.0, 0.0)])
        pos = estimate_positions(g, root=0)
        assert pos[1].n_constraints == 1

    def test_all_fragments_visited(self):
        pairs   = [(i, i + 1) for i in range(9)]
        offsets = [(1.0, 0.0)] * 9
        g   = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert len(pos) == 10


# ─── refine_positions ─────────────────────────────────────────────────────────

class TestRefinePositions:
    def _simple_triangle(self):
        pairs   = [(0, 1), (1, 2), (0, 2)]
        offsets = [(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
        g   = build_offset_graph(pairs, offsets)
        ini = estimate_positions(g, root=0)
        return g, ini

    def test_returns_all_keys(self):
        g, ini = self._simple_triangle()
        ref = refine_positions(g, ini)
        assert set(ref.keys()) == set(ini.keys())

    def test_returns_position_estimate_objects(self):
        g, ini = self._simple_triangle()
        ref = refine_positions(g, ini)
        for v in ref.values():
            assert isinstance(v, PositionEstimate)

    def test_coordinates_close_to_initial(self):
        # Для согласованных смещений уточнение не должно сильно менять позиции
        g, ini = self._simple_triangle()
        ref = refine_positions(g, ini)
        for idx in ini:
            assert abs(ref[idx].x - ini[idx].x) < 5.0
            assert abs(ref[idx].y - ini[idx].y) < 5.0

    def test_empty_graph_and_initial(self):
        ref = refine_positions({}, {})
        assert ref == {}

    def test_confidence_in_range(self):
        g, ini = self._simple_triangle()
        ref = refine_positions(g, ini)
        for pe in ref.values():
            assert 0.0 <= pe.confidence <= 1.0

    def test_n_constraints_positive_for_connected(self):
        g, ini = self._simple_triangle()
        ref = refine_positions(g, ini)
        for pe in ref.values():
            assert pe.n_constraints >= 1


# ─── positions_to_array ───────────────────────────────────────────────────────

class TestPositionsToArray:
    def _make_positions(self, n=3):
        return {i: PositionEstimate(idx=i, x=float(i * 10), y=float(i * 5))
                for i in range(n)}

    def test_shape_without_n(self):
        pos = self._make_positions(4)
        arr = positions_to_array(pos)
        assert arr.shape == (4, 2)

    def test_dtype_float32(self):
        pos = self._make_positions(3)
        arr = positions_to_array(pos)
        assert arr.dtype == np.float32

    def test_values_correct(self):
        pos = {0: PositionEstimate(0, 3.0, 7.0),
               1: PositionEstimate(1, -1.0, 5.0)}
        arr = positions_to_array(pos)
        # Сортировка по индексу
        assert arr[0, 0] == pytest.approx(3.0)
        assert arr[0, 1] == pytest.approx(7.0)
        assert arr[1, 0] == pytest.approx(-1.0)
        assert arr[1, 1] == pytest.approx(5.0)

    def test_with_n_fills_nan_for_missing(self):
        pos = {0: PositionEstimate(0, 1.0, 2.0),
               2: PositionEstimate(2, 5.0, 6.0)}
        arr = positions_to_array(pos, n=4)
        assert arr.shape == (4, 2)
        assert np.isnan(arr[1, 0])
        assert np.isnan(arr[3, 1])

    def test_with_n_no_nan_when_all_present(self):
        pos = {i: PositionEstimate(i, float(i), float(i)) for i in range(3)}
        arr = positions_to_array(pos, n=3)
        assert not np.any(np.isnan(arr))

    def test_empty_without_n(self):
        arr = positions_to_array({})
        assert arr.shape == (0, 2)
        assert arr.dtype == np.float32

    def test_empty_with_n(self):
        arr = positions_to_array({}, n=5)
        assert arr.shape == (5, 2)
        assert np.all(np.isnan(arr))

    def test_out_of_range_idx_ignored(self):
        pos = {0: PositionEstimate(0, 1.0, 2.0),
               10: PositionEstimate(10, 99.0, 99.0)}
        arr = positions_to_array(pos, n=3)
        # idx=10 >= n=3, поэтому не записывается
        assert not np.isnan(arr[0, 0])
        assert np.isnan(arr[1, 0])


# ─── align_to_origin ──────────────────────────────────────────────────────────

class TestAlignToOrigin:
    def test_empty(self):
        assert align_to_origin({}) == {}

    def test_min_x_min_y_are_zero(self):
        pos = {i: PositionEstimate(i, float(i + 5), float(i + 3))
               for i in range(4)}
        aligned = align_to_origin(pos)
        xs = [pe.x for pe in aligned.values()]
        ys = [pe.y for pe in aligned.values()]
        assert min(xs) == pytest.approx(0.0)
        assert min(ys) == pytest.approx(0.0)

    def test_preserves_relative_distances(self):
        pos = {0: PositionEstimate(0, 10.0, 20.0),
               1: PositionEstimate(1, 13.0, 27.0)}
        aligned = align_to_origin(pos)
        dx = aligned[1].x - aligned[0].x
        dy = aligned[1].y - aligned[0].y
        assert dx == pytest.approx(3.0)
        assert dy == pytest.approx(7.0)

    def test_confidence_preserved(self):
        pos = {0: PositionEstimate(0, 5.0, 5.0, confidence=0.8, n_constraints=3)}
        aligned = align_to_origin(pos)
        assert aligned[0].confidence == pytest.approx(0.8)
        assert aligned[0].n_constraints == 3

    def test_already_at_origin_unchanged(self):
        pos = {0: PositionEstimate(0, 0.0, 0.0),
               1: PositionEstimate(1, 5.0, 3.0)}
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[0].y == pytest.approx(0.0)

    def test_negative_coordinates_shifted(self):
        pos = {0: PositionEstimate(0, -10.0, -20.0),
               1: PositionEstimate(1, 0.0, 0.0)}
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[0].y == pytest.approx(0.0)
        assert aligned[1].x == pytest.approx(10.0)
        assert aligned[1].y == pytest.approx(20.0)

    def test_returns_new_dict_not_mutated(self):
        pos = {0: PositionEstimate(0, 5.0, 5.0)}
        aligned = align_to_origin(pos)
        # Оригинал не изменён
        assert pos[0].x == pytest.approx(5.0)

    def test_single_fragment(self):
        pos = {0: PositionEstimate(0, 7.0, 3.0)}
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[0].y == pytest.approx(0.0)
