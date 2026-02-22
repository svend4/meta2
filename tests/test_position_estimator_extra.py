"""Extra tests for puzzle_reconstruction.algorithms.position_estimator."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.position_estimator import (
    PositionEstimate,
    align_to_origin,
    build_offset_graph,
    estimate_positions,
    positions_to_array,
    refine_positions,
)


# ─── TestPositionEstimateExtra ───────────────────────────────────────────────

class TestPositionEstimateExtra:
    def test_zero_coords(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert pe.x == pytest.approx(0.0)
        assert pe.y == pytest.approx(0.0)

    def test_negative_coords(self):
        pe = PositionEstimate(idx=1, x=-5.5, y=-10.3)
        assert pe.x == pytest.approx(-5.5)
        assert pe.y == pytest.approx(-10.3)

    def test_large_coords(self):
        pe = PositionEstimate(idx=0, x=1e6, y=1e6)
        assert pe.x == pytest.approx(1e6)

    def test_confidence_zero(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0, confidence=0.0)
        assert pe.confidence == pytest.approx(0.0)

    def test_confidence_one(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0, confidence=1.0)
        assert pe.confidence == pytest.approx(1.0)

    def test_n_constraints_positive(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0, n_constraints=10)
        assert pe.n_constraints == 10

    def test_idx_stored(self):
        pe = PositionEstimate(idx=42, x=1.0, y=2.0)
        assert pe.idx == 42

    def test_repr_type(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert isinstance(repr(pe), str)


# ─── TestBuildOffsetGraphExtra ───────────────────────────────────────────────

class TestBuildOffsetGraphExtra:
    def test_single_pair_both_keys(self):
        g = build_offset_graph([(0, 1)], [(3.0, 4.0)])
        assert 0 in g
        assert 1 in g

    def test_reverse_offsets(self):
        g = build_offset_graph([(0, 1)], [(10.0, -5.0)])
        # 1→0 should be (-10, 5)
        for nb, dx, dy in g[1]:
            if nb == 0:
                assert dx == pytest.approx(-10.0)
                assert dy == pytest.approx(5.0)

    def test_two_pairs_separate(self):
        g = build_offset_graph([(0, 1), (2, 3)], [(1.0, 0.0), (0.0, 1.0)])
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[2]) == 1

    def test_chain_three(self):
        g = build_offset_graph(
            [(0, 1), (1, 2)],
            [(5.0, 0.0), (5.0, 0.0)]
        )
        assert len(g[1]) == 2  # 1→0 and 1→2

    def test_empty(self):
        g = build_offset_graph([], [])
        assert g == {}

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            build_offset_graph([(0, 1), (1, 2)], [(1.0, 0.0)])

    def test_float_offsets(self):
        g = build_offset_graph([(0, 1)], [(3.14, 2.72)])
        nb, dx, dy = g[0][0]
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_zero_offset(self):
        g = build_offset_graph([(0, 1)], [(0.0, 0.0)])
        nb, dx, dy = g[0][0]
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)


# ─── TestEstimatePositionsExtra ──────────────────────────────────────────────

class TestEstimatePositionsExtra:
    def test_empty_graph_empty_result(self):
        assert estimate_positions({}) == {}

    def test_single_edge(self):
        g = build_offset_graph([(0, 1)], [(10.0, 0.0)])
        pos = estimate_positions(g, root=0)
        assert pos[0].x == pytest.approx(0.0)
        assert pos[1].x == pytest.approx(10.0)

    def test_root_at_origin(self):
        g = build_offset_graph([(0, 1)], [(5.0, 3.0)])
        pos = estimate_positions(g, root=0)
        assert pos[0].x == pytest.approx(0.0)
        assert pos[0].y == pytest.approx(0.0)

    def test_chain_y_axis(self):
        pairs = [(0, 1), (1, 2)]
        offsets = [(0.0, 10.0), (0.0, 10.0)]
        g = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert pos[2].y == pytest.approx(20.0)
        assert pos[2].x == pytest.approx(0.0)

    def test_all_visited(self):
        pairs = [(i, i + 1) for i in range(5)]
        offsets = [(1.0, 0.0)] * 5
        g = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert len(pos) == 6

    def test_star_topology(self):
        # 0 is center, connected to 1,2,3,4
        pairs = [(0, i) for i in range(1, 5)]
        offsets = [(float(i) * 10, 0.0) for i in range(1, 5)]
        g = build_offset_graph(pairs, offsets)
        pos = estimate_positions(g, root=0)
        assert len(pos) == 5
        assert pos[0].x == pytest.approx(0.0)

    def test_unknown_root_handled(self):
        g = build_offset_graph([(0, 1)], [(5.0, 0.0)])
        pos = estimate_positions(g, root=999)
        assert len(pos) == 2

    def test_returns_position_estimates(self):
        g = build_offset_graph([(0, 1)], [(1.0, 1.0)])
        pos = estimate_positions(g, root=0)
        for v in pos.values():
            assert isinstance(v, PositionEstimate)


# ─── TestRefinePositionsExtra ────────────────────────────────────────────────

class TestRefinePositionsExtra:
    def _triangle(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        offsets = [(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
        g = build_offset_graph(pairs, offsets)
        ini = estimate_positions(g, root=0)
        return g, ini

    def test_returns_dict(self):
        g, ini = self._triangle()
        assert isinstance(refine_positions(g, ini), dict)

    def test_all_keys_preserved(self):
        g, ini = self._triangle()
        ref = refine_positions(g, ini)
        assert set(ref.keys()) == set(ini.keys())

    def test_values_are_position_estimates(self):
        g, ini = self._triangle()
        ref = refine_positions(g, ini)
        for v in ref.values():
            assert isinstance(v, PositionEstimate)

    def test_empty_graph(self):
        assert refine_positions({}, {}) == {}

    def test_confidence_range(self):
        g, ini = self._triangle()
        for pe in refine_positions(g, ini).values():
            assert 0.0 <= pe.confidence <= 1.0

    def test_close_to_initial(self):
        g, ini = self._triangle()
        ref = refine_positions(g, ini)
        for idx in ini:
            assert abs(ref[idx].x - ini[idx].x) < 10.0
            assert abs(ref[idx].y - ini[idx].y) < 10.0

    def test_n_constraints_positive(self):
        g, ini = self._triangle()
        for pe in refine_positions(g, ini).values():
            assert pe.n_constraints >= 1


# ─── TestPositionsToArrayExtra ───────────────────────────────────────────────

class TestPositionsToArrayExtra:
    def test_empty_no_n(self):
        arr = positions_to_array({})
        assert arr.shape == (0, 2)

    def test_empty_with_n(self):
        arr = positions_to_array({}, n=3)
        assert arr.shape == (3, 2)
        assert np.all(np.isnan(arr))

    def test_single_entry(self):
        pos = {0: PositionEstimate(0, 5.0, 7.0)}
        arr = positions_to_array(pos)
        assert arr.shape == (1, 2)
        assert arr[0, 0] == pytest.approx(5.0)
        assert arr[0, 1] == pytest.approx(7.0)

    def test_dtype_float32(self):
        pos = {0: PositionEstimate(0, 1.0, 2.0)}
        assert positions_to_array(pos).dtype == np.float32

    def test_sorted_by_idx(self):
        pos = {
            2: PositionEstimate(2, 20.0, 0.0),
            0: PositionEstimate(0, 0.0, 0.0),
            1: PositionEstimate(1, 10.0, 0.0),
        }
        arr = positions_to_array(pos)
        assert arr[0, 0] == pytest.approx(0.0)
        assert arr[1, 0] == pytest.approx(10.0)
        assert arr[2, 0] == pytest.approx(20.0)

    def test_with_n_gaps_nan(self):
        pos = {0: PositionEstimate(0, 1.0, 2.0)}
        arr = positions_to_array(pos, n=3)
        assert not np.isnan(arr[0, 0])
        assert np.isnan(arr[1, 0])
        assert np.isnan(arr[2, 0])

    def test_out_of_range_idx_ignored(self):
        pos = {0: PositionEstimate(0, 1.0, 2.0),
               50: PositionEstimate(50, 99.0, 99.0)}
        arr = positions_to_array(pos, n=2)
        assert arr.shape == (2, 2)
        assert not np.isnan(arr[0, 0])
        assert np.isnan(arr[1, 0])


# ─── TestAlignToOriginExtra ─────────────────────────────────────────────────

class TestAlignToOriginExtra:
    def test_empty(self):
        assert align_to_origin({}) == {}

    def test_single_at_origin(self):
        pos = {0: PositionEstimate(0, 3.0, 7.0)}
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[0].y == pytest.approx(0.0)

    def test_already_at_origin(self):
        pos = {
            0: PositionEstimate(0, 0.0, 0.0),
            1: PositionEstimate(1, 5.0, 5.0),
        }
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[1].x == pytest.approx(5.0)

    def test_negative_shifted(self):
        pos = {
            0: PositionEstimate(0, -10.0, -20.0),
            1: PositionEstimate(1, -5.0, -10.0),
        }
        aligned = align_to_origin(pos)
        assert aligned[0].x == pytest.approx(0.0)
        assert aligned[0].y == pytest.approx(0.0)
        assert aligned[1].x == pytest.approx(5.0)
        assert aligned[1].y == pytest.approx(10.0)

    def test_relative_distance_preserved(self):
        pos = {
            0: PositionEstimate(0, 100.0, 200.0),
            1: PositionEstimate(1, 103.0, 207.0),
        }
        aligned = align_to_origin(pos)
        dx = aligned[1].x - aligned[0].x
        dy = aligned[1].y - aligned[0].y
        assert dx == pytest.approx(3.0)
        assert dy == pytest.approx(7.0)

    def test_confidence_preserved(self):
        pos = {0: PositionEstimate(0, 5.0, 5.0, confidence=0.75, n_constraints=2)}
        aligned = align_to_origin(pos)
        assert aligned[0].confidence == pytest.approx(0.75)
        assert aligned[0].n_constraints == 2

    def test_does_not_mutate_original(self):
        pos = {0: PositionEstimate(0, 10.0, 20.0)}
        align_to_origin(pos)
        assert pos[0].x == pytest.approx(10.0)

    def test_min_is_zero(self):
        pos = {i: PositionEstimate(i, float(i * 3 + 7), float(i * 2 + 5))
               for i in range(4)}
        aligned = align_to_origin(pos)
        xs = [pe.x for pe in aligned.values()]
        ys = [pe.y for pe in aligned.values()]
        assert min(xs) == pytest.approx(0.0)
        assert min(ys) == pytest.approx(0.0)
