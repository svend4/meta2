"""Тесты для puzzle_reconstruction.algorithms.position_estimator."""
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

def _chain_graph(n=4, step=10.0):
    """Builds a graph: 0→1→2→...→(n-1) with constant horizontal step."""
    pairs = [(i, i + 1) for i in range(n - 1)]
    offsets = [(step, 0.0)] * (n - 1)
    return build_offset_graph(pairs, offsets)


def _make_pe(idx=0, x=0.0, y=0.0) -> PositionEstimate:
    return PositionEstimate(idx=idx, x=x, y=y)


# ─── TestPositionEstimate ─────────────────────────────────────────────────────

class TestPositionEstimate:
    def test_construction(self):
        pe = PositionEstimate(idx=3, x=10.5, y=-5.0)
        assert pe.idx == 3
        assert pe.x == pytest.approx(10.5)
        assert pe.y == pytest.approx(-5.0)

    def test_default_confidence(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert pe.confidence == pytest.approx(1.0)

    def test_default_n_constraints(self):
        pe = PositionEstimate(idx=0, x=0.0, y=0.0)
        assert pe.n_constraints == 0

    def test_custom_fields(self):
        pe = PositionEstimate(idx=1, x=5.0, y=3.0, confidence=0.8, n_constraints=2)
        assert pe.confidence == pytest.approx(0.8)
        assert pe.n_constraints == 2


# ─── TestBuildOffsetGraph ─────────────────────────────────────────────────────

class TestBuildOffsetGraph:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            build_offset_graph([(0, 1)], [(1.0, 0.0), (2.0, 0.0)])

    def test_empty_returns_empty(self):
        graph = build_offset_graph([], [])
        assert graph == {}

    def test_single_edge_bidirectional(self):
        graph = build_offset_graph([(0, 1)], [(5.0, 3.0)])
        assert 0 in graph
        assert 1 in graph
        # 0→1: (5, 3); 1→0: (-5, -3)
        neighbors_0 = graph[0]
        assert any(n == 1 and dx == pytest.approx(5.0) and dy == pytest.approx(3.0)
                   for n, dx, dy in neighbors_0)
        neighbors_1 = graph[1]
        assert any(n == 0 and dx == pytest.approx(-5.0) and dy == pytest.approx(-3.0)
                   for n, dx, dy in neighbors_1)

    def test_chain_graph_all_nodes_present(self):
        graph = _chain_graph(4)
        for idx in range(4):
            assert idx in graph

    def test_offsets_stored_as_float(self):
        graph = build_offset_graph([(0, 1)], [(1, 0)])
        _, dx, dy = graph[0][0]
        assert isinstance(dx, float)
        assert isinstance(dy, float)


# ─── TestEstimatePositions ────────────────────────────────────────────────────

class TestEstimatePositions:
    def test_empty_graph_returns_empty(self):
        result = estimate_positions({})
        assert result == {}

    def test_single_node(self):
        graph = {0: []}
        result = estimate_positions(graph, root=0)
        assert 0 in result
        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)

    def test_chain_positions_correct(self):
        graph = _chain_graph(4, step=10.0)
        result = estimate_positions(graph, root=0)
        assert len(result) == 4
        assert result[0].x == pytest.approx(0.0)
        assert result[1].x == pytest.approx(10.0)
        assert result[2].x == pytest.approx(20.0)
        assert result[3].x == pytest.approx(30.0)

    def test_y_coordinates_zero_in_chain(self):
        graph = _chain_graph(3, step=5.0)
        result = estimate_positions(graph, root=0)
        for pe in result.values():
            assert pe.y == pytest.approx(0.0, abs=1e-9)

    def test_root_at_origin(self):
        graph = _chain_graph(4, step=10.0)
        result = estimate_positions(graph, root=0)
        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)

    def test_all_fragment_ids_present(self):
        graph = _chain_graph(5)
        result = estimate_positions(graph, root=0)
        for i in range(5):
            assert i in result

    def test_returns_position_estimates(self):
        graph = _chain_graph(3)
        result = estimate_positions(graph, root=0)
        for pe in result.values():
            assert isinstance(pe, PositionEstimate)

    def test_auto_root_selection(self):
        graph = _chain_graph(4)
        result = estimate_positions(graph)  # root=None → auto
        assert len(result) >= 1


# ─── TestRefinePositions ──────────────────────────────────────────────────────

class TestRefinePositions:
    def test_returns_dict(self):
        graph = _chain_graph(3)
        initial = estimate_positions(graph, root=0)
        refined = refine_positions(graph, initial)
        assert isinstance(refined, dict)

    def test_same_keys_as_initial(self):
        graph = _chain_graph(4)
        initial = estimate_positions(graph, root=0)
        refined = refine_positions(graph, initial)
        assert set(refined.keys()) == set(initial.keys())

    def test_chain_positions_close_to_initial(self):
        graph = _chain_graph(4, step=10.0)
        initial = estimate_positions(graph, root=0)
        refined = refine_positions(graph, initial)
        for idx in initial:
            assert refined[idx].x == pytest.approx(initial[idx].x, abs=1.0)

    def test_returns_position_estimates(self):
        graph = _chain_graph(3)
        initial = estimate_positions(graph, root=0)
        for pe in refine_positions(graph, initial).values():
            assert isinstance(pe, PositionEstimate)

    def test_confidence_in_range(self):
        graph = _chain_graph(4)
        initial = estimate_positions(graph, root=0)
        refined = refine_positions(graph, initial)
        for pe in refined.values():
            assert 0.0 <= pe.confidence <= 1.0


# ─── TestPositionsToArray ─────────────────────────────────────────────────────

class TestPositionsToArray:
    def test_empty_returns_empty(self):
        arr = positions_to_array({})
        assert arr.shape == (0, 2)

    def test_shape_n_2(self):
        positions = {0: _make_pe(0, 1.0, 2.0), 1: _make_pe(1, 3.0, 4.0)}
        arr = positions_to_array(positions)
        assert arr.shape == (2, 2)

    def test_dtype_float32(self):
        positions = {0: _make_pe(0, 1.0, 2.0)}
        arr = positions_to_array(positions)
        assert arr.dtype == np.float32

    def test_values_correct(self):
        positions = {0: _make_pe(0, 5.0, 7.0)}
        arr = positions_to_array(positions)
        assert arr[0, 0] == pytest.approx(5.0)
        assert arr[0, 1] == pytest.approx(7.0)

    def test_with_n_fills_nan_for_missing(self):
        positions = {0: _make_pe(0, 1.0, 2.0)}
        arr = positions_to_array(positions, n=3)
        assert arr.shape == (3, 2)
        assert np.isnan(arr[1, 0])
        assert np.isnan(arr[2, 0])

    def test_with_n_full_coverage(self):
        positions = {i: _make_pe(i, float(i), float(i)) for i in range(4)}
        arr = positions_to_array(positions, n=4)
        assert arr.shape == (4, 2)
        assert not np.any(np.isnan(arr))


# ─── TestAlignToOrigin ────────────────────────────────────────────────────────

class TestAlignToOrigin:
    def test_empty_returns_empty(self):
        result = align_to_origin({})
        assert result == {}

    def test_min_x_is_zero(self):
        positions = {0: _make_pe(0, 3.0, 5.0), 1: _make_pe(1, 7.0, 2.0)}
        result = align_to_origin(positions)
        min_x = min(pe.x for pe in result.values())
        assert min_x == pytest.approx(0.0, abs=1e-9)

    def test_min_y_is_zero(self):
        positions = {0: _make_pe(0, 3.0, 5.0), 1: _make_pe(1, 7.0, 2.0)}
        result = align_to_origin(positions)
        min_y = min(pe.y for pe in result.values())
        assert min_y == pytest.approx(0.0, abs=1e-9)

    def test_relative_distances_preserved(self):
        positions = {0: _make_pe(0, 10.0, 0.0), 1: _make_pe(1, 20.0, 0.0)}
        result = align_to_origin(positions)
        dx = result[1].x - result[0].x
        assert dx == pytest.approx(10.0, abs=1e-9)

    def test_already_at_origin_unchanged(self):
        positions = {0: _make_pe(0, 0.0, 0.0), 1: _make_pe(1, 5.0, 3.0)}
        result = align_to_origin(positions)
        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)

    def test_confidence_preserved(self):
        positions = {0: PositionEstimate(idx=0, x=5.0, y=3.0, confidence=0.7)}
        result = align_to_origin(positions)
        assert result[0].confidence == pytest.approx(0.7)

    def test_returns_position_estimates(self):
        positions = {0: _make_pe(0, 2.0, 3.0)}
        for pe in align_to_origin(positions).values():
            assert isinstance(pe, PositionEstimate)
