"""Additional tests for puzzle_reconstruction.algorithms.position_estimator."""
import math
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _chain(n=4, step=10.0):
    pairs = [(i, i + 1) for i in range(n - 1)]
    offsets = [(step, 0.0)] * (n - 1)
    return build_offset_graph(pairs, offsets)


def _grid(rows=2, cols=2, step=10.0):
    """2-D grid graph."""
    pairs, offsets = [], []
    def idx(r, c):
        return r * cols + c
    for r in range(rows):
        for c in range(cols - 1):
            pairs.append((idx(r, c), idx(r, c + 1)))
            offsets.append((step, 0.0))
    for r in range(rows - 1):
        for c in range(cols):
            pairs.append((idx(r, c), idx(r + 1, c)))
            offsets.append((0.0, step))
    return build_offset_graph(pairs, offsets)


def _pe(idx=0, x=0.0, y=0.0, conf=1.0):
    return PositionEstimate(idx=idx, x=x, y=y, confidence=conf)


# ─── TestPositionEstimateExtra ────────────────────────────────────────────────

class TestPositionEstimateExtra:
    def test_negative_coords(self):
        pe = PositionEstimate(idx=1, x=-100.0, y=-50.0)
        assert pe.x == pytest.approx(-100.0)
        assert pe.y == pytest.approx(-50.0)

    def test_large_coords(self):
        pe = PositionEstimate(idx=0, x=1e6, y=2e6)
        assert math.isfinite(pe.x)
        assert math.isfinite(pe.y)

    def test_n_constraints_set(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0, n_constraints=5)
        assert pe.n_constraints == 5

    def test_confidence_zero(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0, confidence=0.0)
        assert pe.confidence == pytest.approx(0.0)

    def test_idx_large(self):
        pe = PositionEstimate(idx=999, x=1.0, y=2.0)
        assert pe.idx == 999

    def test_fields_numeric(self):
        pe = PositionEstimate(idx=0, x=3.0, y=4.0)
        assert pe.x == pytest.approx(3.0)
        assert pe.y == pytest.approx(4.0)


# ─── TestBuildOffsetGraphExtra ────────────────────────────────────────────────

class TestBuildOffsetGraphExtra:
    def test_two_edges_both_nodes_present(self):
        g = build_offset_graph([(0, 1), (1, 2)], [(1.0, 0.0), (1.0, 0.0)])
        assert 0 in g and 1 in g and 2 in g

    def test_forward_backward_offsets(self):
        g = build_offset_graph([(0, 1)], [(3.0, 4.0)])
        fwd = next(dx for n, dx, dy in g[0] if n == 1)
        bwd = next(dx for n, dx, dy in g[1] if n == 0)
        assert fwd == pytest.approx(3.0)
        assert bwd == pytest.approx(-3.0)

    def test_diagonal_offset(self):
        g = build_offset_graph([(0, 1)], [(3.0, 4.0)])
        dy_fwd = next(dy for n, dx, dy in g[0] if n == 1)
        assert dy_fwd == pytest.approx(4.0)

    def test_n_node_chain_all_nodes(self):
        g = _chain(n=6)
        for i in range(6):
            assert i in g

    def test_node_neighbor_count_in_chain(self):
        g = _chain(n=4)
        # Interior nodes: 2 neighbors each
        assert len(g[1]) == 2
        assert len(g[2]) == 2

    def test_grid_4_nodes_all_present(self):
        g = _grid(2, 2)
        for i in range(4):
            assert i in g

    def test_empty_pairs_empty_graph(self):
        g = build_offset_graph([], [])
        assert g == {}

    def test_returns_dict(self):
        g = build_offset_graph([(0, 1)], [(1.0, 0.0)])
        assert isinstance(g, dict)


# ─── TestEstimatePositionsExtra ───────────────────────────────────────────────

class TestEstimatePositionsExtra:
    def test_root_1_is_at_origin(self):
        g = _chain(4, step=10.0)
        result = estimate_positions(g, root=1)
        assert result[1].x == pytest.approx(0.0)
        assert result[1].y == pytest.approx(0.0)

    def test_grid_all_4_nodes(self):
        g = _grid(2, 2, step=10.0)
        result = estimate_positions(g, root=0)
        assert len(result) == 4

    def test_single_isolated_node(self):
        g = {5: []}
        result = estimate_positions(g, root=5)
        assert 5 in result
        assert result[5].x == pytest.approx(0.0)

    def test_chain_length_5(self):
        g = _chain(5, step=7.0)
        result = estimate_positions(g, root=0)
        assert len(result) == 5
        assert result[4].x == pytest.approx(28.0)

    def test_returns_position_estimate_type(self):
        g = _chain(3)
        for pe in estimate_positions(g, root=0).values():
            assert isinstance(pe, PositionEstimate)

    def test_all_y_zero_for_horizontal_chain(self):
        g = _chain(5, step=5.0)
        result = estimate_positions(g, root=0)
        for pe in result.values():
            assert pe.y == pytest.approx(0.0, abs=1e-9)

    def test_large_step(self):
        g = _chain(3, step=1000.0)
        result = estimate_positions(g, root=0)
        assert result[2].x == pytest.approx(2000.0)


# ─── TestRefinePositionsExtra ─────────────────────────────────────────────────

class TestRefinePositionsExtra:
    def test_single_node_unchanged(self):
        g = {0: []}
        initial = {0: _pe(0, 5.0, 3.0)}
        refined = refine_positions(g, initial)
        assert refined[0].x == pytest.approx(5.0)
        assert refined[0].y == pytest.approx(3.0)

    def test_grid_keys_match(self):
        g = _grid(2, 2)
        initial = estimate_positions(g, root=0)
        refined = refine_positions(g, initial)
        assert set(refined.keys()) == set(initial.keys())

    def test_refined_positions_all_finite(self):
        g = _chain(5)
        initial = estimate_positions(g, root=0)
        refined = refine_positions(g, initial)
        for pe in refined.values():
            assert math.isfinite(pe.x) and math.isfinite(pe.y)

    def test_large_chain_refined(self):
        g = _chain(8, step=10.0)
        initial = estimate_positions(g, root=0)
        refined = refine_positions(g, initial)
        assert len(refined) == 8

    def test_n_constraints_nonneg(self):
        g = _chain(4)
        initial = estimate_positions(g, root=0)
        refined = refine_positions(g, initial)
        for pe in refined.values():
            assert pe.n_constraints >= 0


# ─── TestPositionsToArrayExtra ────────────────────────────────────────────────

class TestPositionsToArrayExtra:
    def test_three_positions_shape(self):
        positions = {i: _pe(i, float(i), float(i * 2)) for i in range(3)}
        arr = positions_to_array(positions)
        assert arr.shape == (3, 2)

    def test_x_y_values_correct_multi(self):
        positions = {0: _pe(0, 1.0, 2.0), 1: _pe(1, 3.0, 4.0)}
        arr = positions_to_array(positions)
        vals = {tuple(row) for row in arr.tolist()}
        assert (1.0, 2.0) in vals
        assert (3.0, 4.0) in vals

    def test_n_larger_than_positions(self):
        positions = {0: _pe(0, 5.0, 6.0)}
        arr = positions_to_array(positions, n=5)
        assert arr.shape == (5, 2)
        assert np.count_nonzero(~np.isnan(arr)) == 2  # only idx=0

    def test_n_equal_to_positions(self):
        positions = {i: _pe(i, float(i), 0.0) for i in range(3)}
        arr = positions_to_array(positions, n=3)
        assert not np.any(np.isnan(arr))

    def test_empty_n_0(self):
        arr = positions_to_array({}, n=0)
        assert arr.shape == (0, 2)


# ─── TestAlignToOriginExtra ───────────────────────────────────────────────────

class TestAlignToOriginExtra:
    def test_all_negative_min_xy_become_zero(self):
        positions = {0: _pe(0, -20.0, -15.0), 1: _pe(1, -10.0, -5.0)}
        result = align_to_origin(positions)
        assert min(pe.x for pe in result.values()) == pytest.approx(0.0)
        assert min(pe.y for pe in result.values()) == pytest.approx(0.0)

    def test_single_node_at_zero(self):
        positions = {0: _pe(0, 100.0, 200.0)}
        result = align_to_origin(positions)
        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)

    def test_three_nodes_relative_y_preserved(self):
        positions = {
            0: _pe(0, 0.0, 0.0),
            1: _pe(1, 0.0, 10.0),
            2: _pe(2, 0.0, 20.0),
        }
        result = align_to_origin(positions)
        assert result[2].y - result[1].y == pytest.approx(10.0)
        assert result[1].y - result[0].y == pytest.approx(10.0)

    def test_keys_preserved(self):
        positions = {5: _pe(5, 1.0, 2.0), 9: _pe(9, 3.0, 4.0)}
        result = align_to_origin(positions)
        assert set(result.keys()) == {5, 9}

    def test_n_constraints_preserved(self):
        pe = PositionEstimate(idx=0, x=5.0, y=3.0, n_constraints=7)
        result = align_to_origin({0: pe})
        assert result[0].n_constraints == 7
